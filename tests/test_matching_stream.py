"""Offline tests of the streaming crossmatch (CrossmatchService.crossmatch_stream) and
its SSE route (streaming.router, GET /api/v1/search/stream).

Fake providers answer with fixed delays so the event order is deterministic; the
recorded real responses of 3C 273 check the streamed record against crossmatch().
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client

import streaming
from crossmatch import CrossmatchService
from models import CatalogRegistry, CatalogSource, CatalogUnavailableError, offset_radec
from providers import QueryResult

RA, DEC = 150.0, 20.0
DELAYS = {"gaia_dr3": 0.05, "twomass_psc": 0.25, "allwise": 1.2}


class DelayedProvider:
    """Answers every catalogue after DELAYS[catalog] seconds with the target itself
    (0.05" away) plus one unrelated row 6" away; records cancellations."""

    def __init__(self, delays: dict[str, float], *, failing: set[str] | None = None) -> None:
        self.delays = delays
        self.failing = failing or set()
        self.cancelled: list[str] = []
        self.finished: dict[str, float] = {}
        self.started = time.perf_counter()

    async def query(self, catalog, target, radius_arcsec):
        try:
            await asyncio.sleep(self.delays[catalog.name])
        except asyncio.CancelledError:
            self.cancelled.append(catalog.name)
            raise
        self.finished[catalog.name] = time.perf_counter()
        if catalog.name in self.failing:
            raise CatalogUnavailableError(f"{catalog.name}: HTTP 503 (simulated)")
        rows = []
        for idx, (east, north) in enumerate([(0.05, 0.0), (6.0, 0.0)]):
            ra, dec = offset_radec(target.ra, target.dec, east, north)
            rows.append(CatalogSource(catalog.name, f"{catalog.name}-{idx}", ra, dec, 0.08, {"mag": 12.0 + idx},
                                      {"wavelength": catalog.wavelength}, {}))
        return QueryResult(rows, {"max_rows": catalog.max_rows, "row_limit": catalog.max_rows,
                                  "citation": catalog.citation})


def make_fake_service(delays: dict[str, float] | None = None, **kwargs) -> tuple[CrossmatchService, DelayedProvider]:
    registry = CatalogRegistry()
    registry._catalogs = {name: registry.get(name) for name in DELAYS}
    provider = DelayedProvider(delays or DELAYS, **kwargs)
    return CrossmatchService(registry, {"tap": provider}, radius_arcsec=10.0), provider


async def collect(agen) -> list[tuple[float, dict]]:
    started = time.perf_counter()
    return [(time.perf_counter() - started, event) async for event in agen]


# ---------------------------------------------------------------------------
# crossmatch_stream
# ---------------------------------------------------------------------------


async def test_stream_event_order_and_first_catalogue_before_the_slowest_finishes() -> None:
    service, provider = make_fake_service()
    events = await collect(service.crossmatch_stream(RA, DEC, radius_arcsec=10.0))
    kinds = [e["event"] for _, e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds[1:4] == ["catalog"] * 3
    assert set(kinds[4:-1]) == {"group"}
    catalogs = [e["data"]["catalog"] for _, e in events if e["event"] == "catalog"]
    assert catalogs == ["gaia_dr3", "twomass_psc", "allwise"]  # completion order
    first_at = next(t for t, e in events if e["event"] == "catalog")
    slowest_done = provider.finished["allwise"] - provider.started
    assert first_at < 0.6 < slowest_done  # Gaia's rows arrive ~1 s before AllWISE answers
    gaia = next(e["data"] for _, e in events if e["event"] == "catalog")
    assert gaia["status"] == "success" and gaia["count"] == 2 and len(gaia["sources"]) == 2
    assert gaia["elapsed_ms"] is not None and gaia["elapsed_ms"] < 600
    assert {s["source_id"] for s in gaia["sources"]} == {"gaia_dr3-0", "gaia_dr3-1"}
    start = events[0][1]["data"]
    assert start["catalogs"] == list(DELAYS) and start["radius_arcsec"] == 10.0
    groups = [e["data"] for _, e in events if e["event"] == "group"]
    assert groups[0]["contains_target"] and groups[0]["catalogs"] == sorted(DELAYS)
    record = events[-1][1]["data"]["record"]
    assert [g["group_id"] for g in record["crossmatch_groups"]] == [g["group_id"] for g in groups]
    assert record["catalogs_queried"] == 3 and record["failures"] == []


async def test_stream_done_record_equals_crossmatch() -> None:
    service, _ = make_fake_service()
    events = await collect(service.crossmatch_stream(RA, DEC, radius_arcsec=10.0))
    streamed = events[-1][1]["data"]["record"]
    direct = (await service.crossmatch(RA, DEC, radius_arcsec=10.0)).as_dict()
    strip = lambda groups: [(g["group_id"], g["catalogs"], [m["source_id"] for m in g["members"]],
                             g["match_probability"]) for g in groups]
    assert strip(streamed["crossmatch_groups"]) == strip(direct["crossmatch_groups"])
    assert streamed["provenance"]["matches"] == direct["provenance"]["matches"]


async def test_stream_reports_failed_catalogues_and_keeps_going() -> None:
    service, _ = make_fake_service(failing={"twomass_psc"})
    events = [e for _, e in await collect(service.crossmatch_stream(RA, DEC, radius_arcsec=10.0))]
    failed = next(e["data"] for e in events if e["event"] == "catalog" and e["data"]["catalog"] == "twomass_psc")
    assert failed["status"] == "failed" and failed["error_type"] == "CatalogUnavailableError"
    assert "503" in failed["message"] and failed["sources"] == []
    record = events[-1]["data"]["record"]
    assert [f["catalog"] for f in record["failures"]] == ["twomass_psc"]
    assert "twomass_psc" not in record["crossmatch_groups"][0]["catalogs"]


async def test_closing_the_stream_cancels_pending_catalogue_queries() -> None:
    service, provider = make_fake_service()
    agen = service.crossmatch_stream(RA, DEC, radius_arcsec=10.0)
    assert (await agen.__anext__())["event"] == "start"
    first = await agen.__anext__()
    assert first["data"]["catalog"] == "gaia_dr3"
    await agen.aclose()
    assert sorted(provider.cancelled) == ["allwise", "twomass_psc"]
    await asyncio.sleep(1.3)
    assert "allwise" not in provider.finished


async def test_stream_validates_inputs_before_querying() -> None:
    service, provider = make_fake_service()
    with pytest.raises(ValueError, match="Unknown catalog"):
        await collect(service.crossmatch_stream(RA, DEC, catalogs=["nope"]))
    with pytest.raises(ValueError, match="ra and dec"):
        await collect(service.crossmatch_stream())
    assert provider.finished == {}


async def test_stream_with_name_uses_the_resolver_position_and_errors() -> None:
    from providers import SesameResolver

    xml = (Path(__file__).resolve().parent / "fixtures" / "sesame" / "3c273.xml").read_text(encoding="utf-8")
    obj = SesameResolver.parse_response("3C 273", xml)

    class Resolver:
        async def resolve(self, name):
            assert name == "3C 273"
            return obj

    service, _ = make_fake_service()
    events = [e for _, e in await collect(service.crossmatch_stream(name="3C 273", resolver=Resolver()))]
    start = events[0]["data"]
    assert start["resolved_object"]["canonical_name"] == obj.canonical_name
    assert start["target"]["ra"] == pytest.approx(obj.ra_deg) and start["target"]["dec"] == pytest.approx(obj.dec_deg)
    record = events[-1]["data"]["record"]
    assert record["resolved_object"]["query"] == "3C 273"


async def test_stream_on_recorded_3c273_matches_the_non_streaming_record() -> None:
    catalogs = ["gaia_dr3", "simbad", "twomass_psc", "first", "nvss", "rosat"]
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273", catalogs)))
        async with offline_client() as client:
            service = make_service(client)
            events = [e for _, e in await collect(service.crossmatch_stream(
                187.2779154, 2.0523883, radius_arcsec=10.0, catalogs=catalogs))]
            direct = await service.crossmatch(187.2779154, 2.0523883, radius_arcsec=10.0, catalogs=catalogs)
    assert sorted(e["data"]["catalog"] for e in events if e["event"] == "catalog") == sorted(catalogs)
    streamed = events[-1]["data"]["record"]
    assert streamed["crossmatch_groups"][0]["catalogs"] == direct.crossmatch_groups[0]["catalogs"] == sorted(catalogs)
    assert [g["group_id"] for g in streamed["crossmatch_groups"]] == [g["group_id"] for g in direct.crossmatch_groups]


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------


def test_sse_frame_and_jsonable() -> None:
    import numpy as np

    frame = streaming.sse_frame("catalog", {"x": np.float64(1.5), "n": np.int64(3), "bad": float("nan"),
                                            "t": (1, 2), "b": b"\xff"}, 7)
    assert frame.startswith("event: catalog\nid: 7\ndata: ") and frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"x": 1.5, "n": 3, "bad": None, "t": [1, 2], "b": "/w=="}
    assert streaming.sse_comment() == ": keepalive\n\n"
    with pytest.raises(ValueError):
        streaming.sse_frame("bad\nname", {})


async def test_event_stream_sends_keepalive_comments_while_waiting() -> None:
    async def slow():
        yield {"event": "start", "data": {}}
        await asyncio.sleep(0.35)
        yield {"event": "done", "data": {}}

    chunks = [c async for c in streaming.event_stream(slow(), keepalive=0.1)]
    assert chunks[0].startswith("event: start\nid: 1\n") and chunks[-1].startswith("event: done\nid: 2\n")
    assert 2 <= sum(c == ": keepalive\n\n" for c in chunks) <= 4


async def test_event_stream_turns_errors_into_an_error_event() -> None:
    async def broken():
        yield {"event": "start", "data": {}}
        raise RuntimeError("archive exploded")

    chunks = [c async for c in streaming.event_stream(broken(), keepalive=5.0)]
    assert chunks[-1].startswith("event: error\n") and "archive exploded" in chunks[-1]


# ---------------------------------------------------------------------------
# Router (minimal FastAPI app)
# ---------------------------------------------------------------------------


def make_app(service):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(streaming.router)
    app.state.service = service
    return app


def test_router_streams_sse_parsed_by_httpx_sse() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    service, _ = make_fake_service()
    with TestClient(make_app(service)) as client, connect_sse(client, "GET", "/api/v1/search/stream",
                     params={"ra": RA, "dec": DEC, "radius_arcsec": 10.0, "catalogs": "gaia_dr3,allwise"}) as source:
        assert source.response.status_code == 200
        assert source.response.headers["content-type"].startswith("text/event-stream")
        assert source.response.headers["cache-control"] == "no-cache"
        events = list(source.iter_sse())
    kinds = [e.event for e in events]
    assert kinds[0] == "start" and kinds[1:3] == ["catalog", "catalog"] and kinds[-1] == "done"
    assert [int(e.id) for e in events] == list(range(1, len(events) + 1))
    data = [e.json() for e in events]
    assert [d["catalog"] for d, k in zip(data, kinds) if k == "catalog"] == ["gaia_dr3", "allwise"]
    record = data[-1]["record"]
    assert record["catalogs_queried"] == 2 and record["crossmatch_groups"][0]["catalogs"] == ["allwise", "gaia_dr3"]


def test_router_rejects_bad_requests_before_streaming() -> None:
    from fastapi.testclient import TestClient

    service, provider = make_fake_service()
    with TestClient(make_app(service)) as client:
        assert client.get("/api/v1/search/stream", params={"ra": RA}).status_code == 422
        response = client.get("/api/v1/search/stream", params={"ra": RA, "dec": DEC, "catalogs": "gaia_dr3,nope"})
        assert response.status_code == 422 and "Unknown catalog" in response.json()["detail"]
        assert client.get("/api/v1/search/stream", params={"ra": RA, "dec": DEC, "profile": "nope"}).status_code == 422
        assert client.get("/api/v1/search/stream", params={"ra": RA, "dec": 95.0}).status_code == 422
    assert provider.finished == {}


def test_router_unknown_name_is_404() -> None:
    from fastapi.testclient import TestClient

    service, _ = make_fake_service()
    empty = "<?xml version='1.0'?><Sesame><Target><name>zzz</name><INFO>*** Nothing found ***</INFO></Target></Sesame>"
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(200, text=empty)
        with TestClient(make_app(service)) as client:
            response = client.get("/api/v1/search/stream", params={"name": "zzz no such object"})
    assert response.status_code == 404


async def test_client_disconnect_cancels_pending_catalogue_queries() -> None:
    """Drive the ASGI app directly: the client disconnects after the first catalogue
    event; the slow catalogue queries must be cancelled and the response must end."""
    service, provider = make_fake_service()
    app = make_app(service)
    sent: list[dict] = []
    got_catalog = asyncio.Event()
    body = b""
    requested = False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await got_catalog.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal body
        sent.append(message)
        if message["type"] == "http.response.body":
            body += message.get("body", b"")
            if b"event: catalog" in body:
                got_catalog.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": "/api/v1/search/stream", "raw_path": b"/api/v1/search/stream",
             "query_string": f"ra={RA}&dec={DEC}&radius_arcsec=10".encode(), "headers": [(b"host", b"test")],
             "client": ("127.0.0.1", 1234), "server": ("test", 80), "root_path": "", "app": app}
    started = time.perf_counter()
    await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
    assert time.perf_counter() - started < 1.0  # returned long before AllWISE's 1.2 s
    assert b"event: catalog" in body and b"event: done" not in body
    await asyncio.sleep(0.05)
    assert sorted(provider.cancelled) == ["allwise", "twomass_psc"]


async def test_disconnect_polling_with_asgi_2_4() -> None:
    """With ASGI spec 2.4 Starlette does not listen for disconnects: the route polls
    ``request.is_disconnected()`` (which only takes a message that is already there)."""
    service, provider = make_fake_service({"gaia_dr3": 0.05, "twomass_psc": 0.25, "allwise": 3.0})
    app = make_app(service)
    state = {"calls": 0, "body": b""}

    async def receive():
        state["calls"] += 1
        if state["calls"] == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        if b"event: catalog" in state["body"]:
            return {"type": "http.disconnect"}
        await asyncio.sleep(3600)  # nothing yet (cancelled by is_disconnected)

    async def send(message):
        if message["type"] == "http.response.body":
            state["body"] += message.get("body", b"")

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": "/api/v1/search/stream", "raw_path": b"/api/v1/search/stream",
             "query_string": f"ra={RA}&dec={DEC}".encode(), "headers": [(b"host", b"test")],
             "client": ("127.0.0.1", 1234), "server": ("test", 80), "root_path": "", "app": app}
    started = time.perf_counter()
    await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
    assert time.perf_counter() - started < 2.5  # AllWISE would answer at 3 s
    assert b"event: catalog" in state["body"] and b"event: done" not in state["body"]
    await asyncio.sleep(0.05)
    assert "allwise" in provider.cancelled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    streaming.register_cli(sub)
    return parser.parse_args(argv)


def test_cli_stream_prints_json_lines(capsys: pytest.CaptureFixture[str]) -> None:
    service, _ = make_fake_service()
    args = _parse(["stream", "--ra", str(RA), "--dec", str(DEC), "--radius", "10"])
    args.service = service
    assert args.handler(args) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["event"] for line in lines][:4] == ["start", "catalog", "catalog", "catalog"]
    assert lines[-1]["event"] == "done" and lines[-1]["data"]["groups"] >= 1
    assert "sources" not in lines[1]["data"]


def test_cli_stream_sse_format_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    service, _ = make_fake_service()
    args = _parse(["stream", "--ra", str(RA), "--dec", str(DEC), "--format", "sse", "--catalogs", "gaia_dr3"])
    args.service = service
    assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("event: start\nid: 1\n") and "event: done" in out
    bad = _parse(["stream", "--ra", str(RA)])
    assert bad.handler(bad) == 2
    bad = _parse(["stream", "--ra", str(RA), "--dec", str(DEC), "--catalogs", "nope"])
    bad.service = service
    assert bad.handler(bad) == 2
