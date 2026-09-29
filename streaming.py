"""Server-Sent Events (SSE) streaming of crossmatch results.

``GET /api/v1/search/stream`` runs :meth:`crossmatch.CrossmatchService.crossmatch_stream`
and sends each event as it happens, so a client sees the fast archives' rows (Gaia,
SIMBAD) while slow ones (IRSA, HEASARC) are still running::

    event: start      data: {"target": {...}, "catalogs": [...], ...}
    event: catalog    data: {"catalog": "gaia_dr3", "status": "success", "count": 1, "elapsed_ms": 812.4, "sources": [...]}
    ...
    event: group      data: {"group_id": "object-1", "catalogs": [...], "match_probability": 0.99, ...}
    event: done       data: {"record": {...UnifiedRecord...}}

Framing follows the WHATWG HTML "Server-sent events" specification: ``event:``/``id:``/
``data:`` fields, one JSON document per ``data:`` line, a blank line after each event, and
``: keepalive`` comment lines every ``SSE_KEEPALIVE_SECONDS`` (default 15 s) so proxies
keep an idle connection open. When the client disconnects, the catalogue queries still
running are cancelled. An upstream error after the stream has started is sent as an
``error`` event (the HTTP status is already 200).

The pure-Python helpers (:func:`sse_frame`, :func:`jsonable`, :func:`event_stream`) are
usable without FastAPI; ``register_cli`` adds the ``stream`` CLI subcommand.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import json
import math
import os
from collections.abc import AsyncIterator, Mapping
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

DEFAULT_KEEPALIVE_SECONDS = 15.0
# How often the stream checks for a client disconnect when nothing else happens.
DISCONNECT_POLL_SECONDS = 0.5
MEDIA_TYPE = "text/event-stream"

router = APIRouter(prefix="/api/v1", tags=["streaming"])


# ---------------------------------------------------------------------------
# Serialisation & framing
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Convert crossmatch output (dataclasses, numpy scalars, tuples, NaN, bytes, dates)
    into strict JSON values (NaN / infinity become null)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(bytes(value)).decode("ascii")
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)  # numpy array
    if callable(tolist):
        return jsonable(tolist())
    return str(value)


def sse_frame(event: str, data: Any, event_id: int | str | None = None) -> str:
    """One SSE event: ``event:``, optional ``id:``, one ``data:`` line per line of the
    JSON payload (compact JSON has none), and the terminating blank line."""
    if "\n" in event or "\r" in event:
        raise ValueError("event names must be single-line")
    payload = json.dumps(jsonable(data), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


def sse_comment(text: str = "keepalive") -> str:
    """An SSE comment line (ignored by clients; keeps idle connections open)."""
    return f": {text}\n\n"


def keepalive_seconds() -> float:
    try:
        value = float(os.getenv("SSE_KEEPALIVE_SECONDS", DEFAULT_KEEPALIVE_SECONDS))
    except ValueError:
        return DEFAULT_KEEPALIVE_SECONDS
    return value if value > 0 else DEFAULT_KEEPALIVE_SECONDS


_DONE = object()


async def event_stream(
    events: AsyncIterator[dict[str, Any]],
    *,
    keepalive: float | None = None,
    is_disconnected: Any = None,
) -> AsyncIterator[str]:
    """SSE text for an async iterator of ``{"event", "data"}`` dicts.

    The events are consumed by a background task so keepalive comments can be sent
    while the next event is pending. ``is_disconnected`` (an async callable) is polled;
    when it returns True -- or when this generator is closed or cancelled, which is what
    Starlette does on a client disconnect -- the producer is cancelled, which closes
    ``events`` (so :meth:`crossmatch_stream` cancels its pending catalogue queries).
    """
    interval = keepalive if keepalive is not None else keepalive_seconds()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for item in events:
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - forwarded to the client as an SSE error event
            await queue.put({"event": "error", "data": {"error_type": exc.__class__.__name__, "message": str(exc)}})
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await asyncio.shield(_quiet(aclose()))
            queue.put_nowait(_DONE)

    producer = asyncio.create_task(produce(), name="sse-producer")
    counter = 0
    loop = asyncio.get_running_loop()
    last_sent = loop.time()
    try:
        while True:
            wait = min(interval, DISCONNECT_POLL_SECONDS) if is_disconnected is not None else interval
            try:
                item = await asyncio.wait_for(queue.get(), timeout=max(0.0, min(wait, interval - (loop.time() - last_sent))))
            except TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    return
                if loop.time() - last_sent >= interval - 1e-3:
                    last_sent = loop.time()
                    yield sse_comment("keepalive")
                continue
            if item is _DONE:
                return
            counter += 1
            last_sent = loop.time()
            yield sse_frame(str(item.get("event", "message")), item.get("data"), counter)
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


async def _quiet(awaitable: Any) -> None:
    """Await ``awaitable`` (closing an event generator), ignoring its errors: the stream
    is ending anyway and the generator's own cleanup (cancelling queries) has run."""
    with contextlib.suppress(BaseException):
        await awaitable


# ---------------------------------------------------------------------------
# FastAPI route
# ---------------------------------------------------------------------------


class _ResolvedName:
    """A resolver that returns an object resolved beforehand (by the route, so an
    unknown name is a 404 before the 200 stream starts)."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    async def resolve(self, _query: str) -> Any:
        return self.obj


def _service_for(request: Request) -> tuple[Any, Any]:
    """(service, client to close afterwards or None): app.state.service when present,
    else a service built by :func:`main.build_service` on a new HTTP client."""
    state = request.app.state
    service = getattr(state, "service", None)
    if service is not None:
        return service, None
    import httpx

    from main import build_service

    client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    return build_service(client=client), client


def parse_catalogs(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    return names or None


@router.get("/search/stream", response_class=StreamingResponse,
            responses={200: {"content": {MEDIA_TYPE: {}}, "description": "Server-sent events"}})
async def search_stream(
    request: Request,
    ra: float | None = Query(None, ge=0.0, lt=360.0, description="Right ascension (deg, ICRS)"),
    dec: float | None = Query(None, ge=-90.0, le=90.0, description="Declination (deg, ICRS)"),
    name: str | None = Query(None, description="Object name resolved with CDS Sesame (instead of ra/dec)"),
    radius_arcsec: float | None = Query(None, gt=0.0, le=3600.0),
    profile: str | None = Query(None),
    catalogs: str | None = Query(None, description="Comma-separated registry catalogue names"),
    epoch: float | None = Query(None, description="Julian year of ra/dec"),
    pm_ra_masyr: float | None = Query(None),
    pm_dec_masyr: float | None = Query(None),
    parallax_mas: float | None = Query(None, gt=0.0),
    target_uncertainty_arcsec: float | None = Query(None, gt=0.0, description="1-sigma per-axis target position error"),
) -> StreamingResponse:
    """Stream a crossmatch as server-sent events (``start``, ``catalog`` per archive as it
    completes, ``group`` per associated object, ``done`` with the full record)."""
    if name is None and (ra is None or dec is None):
        raise HTTPException(status_code=422, detail="Provide ra and dec, or name.")
    service, own_client = _service_for(request)
    try:
        catalog_list = parse_catalogs(catalogs)
        params: dict[str, Any] = {
            "radius_arcsec": radius_arcsec, "profile": profile, "catalogs": catalog_list, "epoch": epoch,
            "pm_ra_masyr": pm_ra_masyr, "pm_dec_masyr": pm_dec_masyr, "parallax_mas": parallax_mas,
            "target_uncertainty_arcsec": target_uncertainty_arcsec,
        }
        if name is not None:
            from models import ObjectResolutionError
            from providers import SesameResolver

            import httpx

            client = getattr(request.app.state, "client", None) or own_client
            temporary = httpx.AsyncClient(timeout=30.0, follow_redirects=True) if client is None else None
            try:
                obj = await SesameResolver(client or temporary).resolve(name)
            except ObjectResolutionError as exc:
                raise HTTPException(status_code=404, detail=f"Object name could not be resolved: {exc}") from exc
            finally:
                if temporary is not None:
                    await temporary.aclose()
            params["name"] = name
            params["resolver"] = _ResolvedName(obj)
            service.prepare(obj.ra_deg, obj.dec_deg, radius_arcsec=radius_arcsec, profile=profile, catalogs=catalog_list,
                            target_uncertainty_arcsec=target_uncertainty_arcsec)
        else:
            # Validate before the 200 response starts (bad profile / catalogue / radius -> 422).
            service.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile,
                            pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas,
                            catalogs=catalog_list, target_uncertainty_arcsec=target_uncertainty_arcsec)
            params["ra"], params["dec"] = ra, dec
    except HTTPException:
        if own_client is not None:
            await own_client.aclose()
        raise
    except ValueError as exc:
        if own_client is not None:
            await own_client.aclose()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    spec = tuple(int(x) for x in str(request.scope.get("asgi", {}).get("spec_version", "2.0")).split(".")[:2])
    # With ASGI spec >= 2.4 Starlette does not listen for http.disconnect itself: poll it.
    poll = request.is_disconnected if spec >= (2, 4) else None
    body = event_stream(service.crossmatch_stream(**params), is_disconnected=poll)
    return StreamingResponse(
        body,
        media_type=MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        background=BackgroundTask(own_client.aclose) if own_client is not None else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _run_stream(args: argparse.Namespace, service: Any = None) -> int:
    import httpx

    client = None
    if service is None:
        from main import build_service

        client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        service = build_service(client=client)
    try:
        events = service.crossmatch_stream(
            args.ra, args.dec, name=args.name, radius_arcsec=args.radius, profile=args.profile,
            catalogs=parse_catalogs(args.catalogs), target_uncertainty_arcsec=args.target_sigma,
        )
        if args.format == "sse":
            async for text in event_stream(events):
                print(text, end="", flush=True)
            return 0
        async for event in events:
            data = event["data"]
            if event["event"] == "catalog" and not args.sources:
                data = {k: v for k, v in data.items() if k != "sources"}
            if event["event"] == "done" and not args.record:
                record = data["record"]
                data = {"groups": len(record.get("crossmatch_groups") or []), "failures": record.get("failures"),
                        "p_any": (record.get("provenance") or {}).get("association", {}).get("p_any"),
                        "elapsed_ms": data.get("elapsed_ms")}
            print(json.dumps({"event": event["event"], "data": jsonable(data)}, separators=(",", ":")), flush=True)
        return 0
    finally:
        if client is not None:
            await client.aclose()


def _cli_stream(args: argparse.Namespace) -> int:
    if args.name is None and (args.ra is None or args.dec is None):
        print("error: provide --ra and --dec, or --name")
        return 2
    try:
        return asyncio.run(_run_stream(args, getattr(args, "service", None)))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2


def register_cli(subparsers: Any) -> None:
    """Add the ``stream`` subcommand: print crossmatch events as each archive answers."""
    parser = subparsers.add_parser("stream", help="Stream a crossmatch catalogue by catalogue (JSON lines or SSE)")
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--name", type=str, help="Object name resolved with CDS Sesame")
    parser.add_argument("--radius", type=float, default=None, help="Search radius (arcsec)")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--catalogs", type=str, default=None, help="Comma-separated catalogue names")
    parser.add_argument("--target-sigma", type=float, default=None, help="Target position 1-sigma (arcsec)")
    parser.add_argument("--format", choices=("jsonl", "sse"), default="jsonl")
    parser.add_argument("--sources", action="store_true", help="Include the rows in catalog events")
    parser.add_argument("--record", action="store_true", help="Print the full record in the done event")
    parser.set_defaults(handler=_cli_stream)
