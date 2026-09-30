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
running are cancelled. Every input is validated before the 200 response starts (422 bad
input -- coordinates, epoch, motion, parallax, catalogues, priors; 404 unknown name; 503
name resolver unavailable). An error after the stream has started is sent as an
``error`` event whose data holds ``status`` (HTTP-like: 422, 404, 502, 504 or 500) and
``detail`` (as the web UI reads them) plus ``error_type`` and ``message``.

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
import sys
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


# Characters Python's str.splitlines() and some line readers treat as line breaks although
# SSE does not (it splits on CR / LF only): JSON-escaped so no client splits a payload.
_UNICODE_LINE_BREAKS = {" ": "\\u2028", " ": "\\u2029", "\u0085": "\\u0085", "\x0b": "\\u000b",
                        "\x0c": "\\u000c", "\x1c": "\\u001c", "\x1d": "\\u001d", "\x1e": "\\u001e"}


def sse_payload(data: Any) -> str:
    """The one-line compact JSON text of an event's data (see :func:`sse_frame`)."""
    payload = json.dumps(jsonable(data), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    for raw, escaped in _UNICODE_LINE_BREAKS.items():
        if raw in payload:
            payload = payload.replace(raw, escaped)
    return payload


def sse_frame(event: str, data: Any, event_id: int | str | None = None, *, payload: str | None = None) -> str:
    """One SSE event: ``event:``, optional ``id:``, the compact JSON payload on one
    ``data:`` line, and the terminating blank line.

    Compact JSON never contains CR or LF (they are escaped inside strings); Unicode line
    separators (U+2028, U+2029, U+0085, ...) that ``ensure_ascii=False`` would leave raw
    are escaped too, so the payload is always exactly one line. ``payload``: the text
    :func:`sse_payload` already made of ``data`` (e.g. in a worker thread).
    """
    if any(ch in str(event) for ch in "\r\n") or any(ch in str(event) for ch in _UNICODE_LINE_BREAKS):
        raise ValueError("event names must be single-line")
    if payload is None:
        payload = sse_payload(data)
    elif "\n" in payload or "\r" in payload:
        raise ValueError("a pre-serialised payload must be a single line")
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.extend(f"data: {line}" for line in payload.split("\n"))
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
            await queue.put({"event": "error", "data": error_event_data(exc)})
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
            # Events of crossmatch_stream(serializer=sse_payload) carry their text, made in a
            # worker thread: the loop only writes it.
            yield sse_frame(str(item.get("event", "message")), item.get("data"), counter, payload=item.get("json"))
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


def error_status(exc: BaseException) -> int:
    """HTTP-like status of an error event: 422 bad input, 404 unknown object name, 502 resolver
    or archive failure (an upstream error, as the web UI assumes by default), 504 timeout,
    500 anything else. (The route answers a resolver outage before the stream starts with the
    statuses of POST /api/v1/search: 503 with Retry-After.)"""
    from models import (
        CatalogUnavailableError,
        InvalidCoordinateError,
        ObjectResolutionError,
        QueryTimeoutError,
        ResolverUnavailableError,
    )

    if isinstance(exc, ResolverUnavailableError | CatalogUnavailableError):
        return 502
    if isinstance(exc, QueryTimeoutError | TimeoutError):
        return 504
    if isinstance(exc, ObjectResolutionError):
        return 404
    if isinstance(exc, InvalidCoordinateError | ValueError):
        return 422
    return 500


def error_event_data(exc: BaseException) -> dict[str, Any]:
    """Payload of an SSE ``error`` event: ``status`` (see :func:`error_status`) and
    ``detail`` (what the web UI reads), plus ``error_type`` and ``message``."""
    return {"status": error_status(exc), "detail": str(exc) or exc.__class__.__name__,
            "error_type": exc.__class__.__name__, "message": str(exc)}


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
    """Comma-separated catalogue names; None when not given. A value naming no catalogue
    (``""``, ``","``) is an error rather than 'all catalogues'."""
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("catalogs must name at least one catalogue (omit it to query all of them)")
    return names


@router.get("/search/stream", response_class=StreamingResponse,
            responses={200: {"content": {MEDIA_TYPE: {}}, "description": "Server-sent events"}})
async def search_stream(
    request: Request,
    ra: str | None = Query(None, description="Right ascension as typed: decimal degrees in [0, 360) or sexagesimal "
                                            "hours ('12 29 06.7'); its rounding sets the target uncertainty"),
    dec: str | None = Query(None, description="Declination as typed: decimal degrees in [-90, 90] or sexagesimal "
                                             "('+02 03 08.6')"),
    name: str | None = Query(None, description="Object name resolved with CDS Sesame (instead of ra/dec)"),
    radius_arcsec: float | None = Query(None, gt=0.0, le=3600.0),
    profile: str | None = Query(None),
    catalogs: str | None = Query(None, description="Comma-separated registry catalogue names"),
    epoch: float | None = Query(None, description="Julian year of ra/dec (with name: the epoch to move the "
                                                 "resolved position to)"),
    pm_ra_masyr: float | None = Query(None),
    pm_dec_masyr: float | None = Query(None),
    parallax_mas: float | None = Query(None, gt=0.0),
    target_uncertainty_arcsec: float | None = Query(None, gt=0.0, le=3600.0,
                                                    description="1-sigma per-axis target position error (arcsec)"),
    target_pm_error_masyr: float | None = Query(None, ge=0.0, description="1-sigma per-axis target proper-motion error"),
    completeness: float | None = Query(None, gt=0.0, lt=1.0,
                                       description="Prior probability of a counterpart in each catalogue "
                                                   "(default: by the target's class)"),
    target_class: str | None = Query(None, description="unknown, star, extragalactic or extended (default: inferred)"),
) -> StreamingResponse:
    """Stream a crossmatch as server-sent events (``start``, ``catalog`` per archive as it
    completes, ``group`` per associated object, ``done`` with the full record).

    Every input is validated before the 200 response starts: bad input is a 422, an
    object name Sesame does not know a 404, and a Sesame outage a 503. ``ra``/``dec`` are
    forwarded as the text typed, so their rounding (``187.278``: 1" per axis; ``150.500000``:
    exact to 1 microdegree) sets the target uncertainty (see ``CrossmatchService.prepare``)."""
    from crossmatch import parse_target_coordinates, resolved_search_target, validate_search_inputs
    from main import check_search_radius
    from models import InvalidCoordinateError, ObjectResolutionError, resolution_failure_status

    if name is None and (ra is None or dec is None):
        raise HTTPException(status_code=422, detail="Provide ra and dec, or name.")
    if name is not None and (ra is not None or dec is not None):
        raise HTTPException(status_code=422, detail="Give either an object name or ra/dec, not both.")
    try:
        # API_MAX_RADIUS_ARCSEC, as POST /api/v1/search (the Query bound is the absolute 3600").
        check_search_radius(radius_arcsec, settings=getattr(request.app.state, "settings", None))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    service, own_client = _service_for(request)
    try:
        catalog_list = parse_catalogs(catalogs)
        params: dict[str, Any] = {
            "radius_arcsec": radius_arcsec, "profile": profile, "catalogs": catalog_list, "epoch": epoch,
            "pm_ra_masyr": pm_ra_masyr, "pm_dec_masyr": pm_dec_masyr, "parallax_mas": parallax_mas,
            "target_uncertainty_arcsec": target_uncertainty_arcsec, "target_pm_error_masyr": target_pm_error_masyr,
            "completeness": completeness, "target_class": target_class,
        }
        if name is not None:
            import httpx

            from providers import SesameResolver

            # The requested epoch / motion / parallax are checked before resolving the name.
            validate_search_inputs(epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                   parallax_mas=parallax_mas)
            client = getattr(request.app.state, "client", None) or own_client
            temporary = httpx.AsyncClient(timeout=30.0, follow_redirects=True) if client is None else None
            try:
                obj = await SesameResolver(client or temporary).resolve(name)
            except ObjectResolutionError as exc:
                # The statuses of POST /api/v1/search (api.search_error): 404 unknown name, 503 (with
                # Retry-After) resolver unreachable, 502 unusable answer, 422 empty name.
                code = resolution_failure_status(exc)
                if code == 503:
                    raise HTTPException(status_code=503, detail=f"Name resolver unavailable: {exc}",
                                        headers={"Retry-After": "30"}) from exc
                if code == 404:
                    raise HTTPException(status_code=404, detail=f"Object name could not be resolved: {exc}") from exc
                if code == 422:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                raise HTTPException(status_code=502, detail=f"Name resolver failed: {exc}") from exc
            finally:
                if temporary is not None:
                    await temporary.aclose()
            params["name"] = name
            params["resolver"] = _ResolvedName(obj)
            spec = resolved_search_target(obj, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                          parallax_mas=parallax_mas, target_uncertainty_arcsec=target_uncertainty_arcsec,
                                          target_pm_error_masyr=target_pm_error_masyr)
            service.prepare(spec["ra"], spec["dec"], radius_arcsec=radius_arcsec, epoch=spec["epoch"], profile=profile,
                            pm_ra_masyr=spec["pm_ra_masyr"], pm_dec_masyr=spec["pm_dec_masyr"],
                            parallax_mas=spec["parallax_mas"], catalogs=catalog_list,
                            target_uncertainty_arcsec=spec["target_uncertainty_arcsec"],
                            target_pm_error_masyr=spec["target_pm_error_masyr"], completeness=completeness,
                            target_class=target_class)
        else:
            ra_value, dec_value, _ = parse_target_coordinates(ra, dec)
            try:
                in_range = 0.0 <= float(ra_value) < 360.0 and -90.0 <= float(dec_value) <= 90.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ra and dec must be decimal degrees or sexagesimal text, got {ra!r}, {dec!r}") from exc
            if not in_range:
                raise ValueError("ra must lie in [0, 360) degrees and dec in [-90, 90] degrees")
            # Validate before the 200 response starts (bad profile / catalogue / radius -> 422).
            service.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile,
                            pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas,
                            catalogs=catalog_list, target_uncertainty_arcsec=target_uncertainty_arcsec,
                            target_pm_error_masyr=target_pm_error_masyr, completeness=completeness,
                            target_class=target_class)
            params["ra"], params["dec"] = ra, dec
    except HTTPException:
        if own_client is not None:
            await own_client.aclose()
        raise
    except (ValueError, InvalidCoordinateError, OverflowError) as exc:  # OverflowError: '0e400' typed
        if own_client is not None:
            await own_client.aclose()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    spec = tuple(int(x) for x in str(request.scope.get("asgi", {}).get("spec_version", "2.0")).split(".")[:2])
    # With ASGI spec >= 2.4 Starlette does not listen for http.disconnect itself: poll it.
    poll = request.is_disconnected if spec >= (2, 4) else None
    body = event_stream(service.crossmatch_stream(**params, serializer=sse_payload), is_disconnected=poll)
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

    from main import check_search_radius

    check_search_radius(args.radius)  # ValueError: _cli_stream reports invalid input (status 2)
    client = None
    if service is None:
        from main import build_service

        client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        service = build_service(client=client)
    try:
        events = service.crossmatch_stream(
            args.ra, args.dec, name=args.name, radius_arcsec=args.radius, profile=args.profile,
            catalogs=parse_catalogs(args.catalogs), target_uncertainty_arcsec=args.target_sigma,
            epoch=getattr(args, "epoch", None), pm_ra_masyr=getattr(args, "pm_ra", None),
            pm_dec_masyr=getattr(args, "pm_dec", None), parallax_mas=getattr(args, "parallax", None),
            target_pm_error_masyr=getattr(args, "target_pm_error", None),
            completeness=getattr(args, "completeness", None), target_class=getattr(args, "target_class", None),
        )
        from models import AstroSearchError, every_catalog_failed

        if args.format == "sse":
            status = 0
            async for text in event_stream(events):
                if text.startswith("event: error\n"):
                    data = json.loads(text.split("data: ", 1)[1])
                    status = 2 if data.get("status") in (404, 422) else 1
                elif text.startswith("event: done\n") and status == 0:
                    done = json.loads(text.split("data: ", 1)[1])
                    if every_catalog_failed(done.get("record") or {}):
                        status = 1  # a total outage, as `search` reports it
                print(text, end="", flush=True)
            return status

        started = False
        outage = False
        try:
            async for event in events:
                started = True
                data = event["data"]
                if event["event"] == "done":
                    outage = every_catalog_failed(data.get("record") or {})
                if event["event"] == "catalog" and not args.sources:
                    data = {k: v for k, v in data.items() if k != "sources"}
                if event["event"] == "done" and not args.record:
                    record = data["record"]
                    data = {"groups": len(record.get("crossmatch_groups") or []), "failures": record.get("failures"),
                            "p_any": (record.get("provenance") or {}).get("association", {}).get("p_any"),
                            "elapsed_ms": data.get("elapsed_ms")}
                print(json.dumps({"event": event["event"], "data": jsonable(data)}, separators=(",", ":")), flush=True)
        except Exception as exc:  # reported as the error event the SSE format sends
            if not started and isinstance(exc, ValueError | AstroSearchError):
                raise  # invalid input / unknown name: _cli_stream reports it (status 2)
            print(json.dumps({"event": "error", "data": error_event_data(exc)}, separators=(",", ":")), flush=True)
            return 1
        if outage:
            print("Error: every queried catalog failed; no data was retrieved.", file=sys.stderr)
            return 1
        return 0
    finally:
        if client is not None:
            await client.aclose()


def _cli_stream(args: argparse.Namespace) -> int:
    """Exit status: 0 when the stream completed, 1 when it ended with an error event (both
    formats print it), 2 for invalid input or an unknown name (before the stream starts)."""
    from models import AstroSearchError

    if args.name is None and (args.ra is None or args.dec is None):
        print("error: provide --ra and --dec, or --name")
        return 2
    if args.name is not None and (args.ra is not None or args.dec is not None):
        print("error: give either --name or --ra/--dec, not both")
        return 2
    try:
        return asyncio.run(_run_stream(args, getattr(args, "service", None)))
    except (ValueError, AstroSearchError) as exc:  # bad input, unknown name, resolver outage
        print(f"error: {exc.__class__.__name__}: {exc}")
        return 2


def register_cli(subparsers: Any) -> None:
    """Add the ``stream`` subcommand: print crossmatch events as each archive answers."""
    parser = subparsers.add_parser("stream", help="Stream a crossmatch catalogue by catalogue (JSON lines or SSE)")
    # Coordinates as typed (decimal degrees or sexagesimal): their rounding sets the target
    # uncertainty, so they are passed on as text rather than parsed to floats here.
    parser.add_argument("--ra", type=str, help="Right ascension (decimal degrees, or sexagesimal hours '12 29 06.7')")
    parser.add_argument("--dec", type=str, help="Declination (decimal degrees, or sexagesimal '+02 03 08.6')")
    parser.add_argument("--name", type=str, help="Object name resolved with CDS Sesame")
    parser.add_argument("--radius", type=float, default=None, help="Search radius (arcsec)")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--catalogs", type=str, default=None, help="Comma-separated catalogue names")
    parser.add_argument("--epoch", type=float, default=None,
                        help="Julian year of --ra/--dec (with --name: move the resolved position to it)")
    parser.add_argument("--pm-ra", type=float, default=None, help="Target proper motion in RA*cos(Dec), mas/yr")
    parser.add_argument("--pm-dec", type=float, default=None, help="Target proper motion in Dec, mas/yr")
    parser.add_argument("--parallax", type=float, default=None, help="Target parallax (mas)")
    parser.add_argument("--target-pm-error", type=float, default=None, help="Target proper-motion 1-sigma (mas/yr)")
    parser.add_argument("--completeness", type=float, default=None,
                        help="Prior probability of a counterpart per catalogue (default: by target class)")
    parser.add_argument("--target-class", choices=("unknown", "star", "extragalactic", "extended"), default=None)
    parser.add_argument("--target-sigma", type=float, default=None, help="Target position 1-sigma (arcsec, at most 3600)")
    parser.add_argument("--format", choices=("jsonl", "sse"), default="jsonl")
    parser.add_argument("--sources", action="store_true", help="Include the rows in catalog events")
    parser.add_argument("--record", action="store_true", help="Print the full record in the done event")
    parser.set_defaults(handler=_cli_stream)
