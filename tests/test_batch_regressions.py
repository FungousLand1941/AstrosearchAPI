"""Offline regression tests for the round-1 review of batch.py.

Covers: batch ``confidence`` equals the single-object CrossmatchService posterior; XMatch requests per radius
bucket (not per target); host-before-planet ranking; REST JSON aliases; data-loss guards; timeout splitting and
time budgets; no cone fallback on deterministic 4xx; circuit breaker settled on cancellation; response/body size
caps; shared guards; per-endpoint upload concurrency; event-loop responsiveness; CLI encoding errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from fixture_io import TARGETS, load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_batch import canary_cone_exchanges, replay_batch
from test_batch_fixtures import (
    CANARY_RADIUS,
    CANARY_TARGETS,
    HD209458_OFFSETS,
    UPLOAD_CATALOGS,
    batch_replay_side_effect,
    load_batch_exchanges,
    multipart_fields,
    request_body,
)

import batch
from batch import BatchCrossmatcher, BatchError, BatchResult, parse_targets, radius_buckets
from models import InvalidCoordinateError, plan_cone, validate_target

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GAIA_XMATCH_FIELDS = (
    ("angDist", "double", "arcsec", "pos.angDistance"), ("t_idx", "int", None, None),
    ("t_ra", "double", None, None), ("t_dec", "double", None, None),
    ("Source", "long", None, "meta.id;meta.main"), ("RAdeg", "double", "deg", "pos.eq.ra;meta.main"),
    ("DEdeg", "double", "deg", "pos.eq.dec;meta.main"), ("e_RAdeg", "double", "mas", "stat.error;pos.eq.ra"),
    ("e_DEdeg", "double", "mas", "stat.error;pos.eq.dec"),
)


def xmatch_votable(rows: list[tuple[Any, ...]]) -> bytes:
    """A CDS-XMatch-like VOTable answer (Gaia columns) with the given rows."""
    fields = "".join(
        f'<FIELD name="{n}" datatype="{d}"' + (f' unit="{u}"' if u else "") + (f' ucd="{c}"' if c else "") + "/>"
        for n, d, u, c in GAIA_XMATCH_FIELDS
    )
    body = "".join("<TR>" + "".join(f"<TD>{v}</TD>" for v in row) + "</TR>" for row in rows)
    return (
        '<?xml version="1.0" encoding="UTF-8"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        '<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>' + fields
        + "<DATA><TABLEDATA>" + body + "</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"
    ).encode()


def uploaded_targets(request: httpx.Request) -> list[tuple[int, float, float]]:
    fields = multipart_fields(request.headers["content-type"], request_body(request))
    lines = fields["cat1"].strip().splitlines()[1:]
    return [(int(a), float(b), float(c)) for a, b, c in (line.split(",") for line in lines)]


def simbad_json(rows: list[list[Any]]) -> dict[str, Any]:
    meta = [{"name": "t_idx", "datatype": "INTEGER"},
            {"name": "main_id", "datatype": "CHAR", "ucd": "meta.id;meta.main"},
            {"name": "ra", "datatype": "DOUBLE", "unit": "deg", "ucd": "pos.eq.ra;meta.main"},
            {"name": "dec", "datatype": "DOUBLE", "unit": "deg", "ucd": "pos.eq.dec;meta.main"}]
    return {"metadata": meta, "data": rows}


def simbad_rows_for(request: httpx.Request, per_target: int = 1) -> list[list[Any]]:
    fields = multipart_fields(request.headers["content-type"], request_body(request))
    rows = re.findall(r"<TR><TD>(\d+)</TD><TD>([^<]+)</TD><TD>([^<]+)</TD>", fields["targets"])
    return [[int(i), f"obj-{int(i)}-{k}", float(ra), float(dec)] for i, ra, dec in rows for k in range(per_target)]


async def run_with(handler, targets, catalogs, radius=5.0, **kwargs) -> BatchResult:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=handler)
        async with offline_client() as client:
            engine = BatchCrossmatcher(client=client, **kwargs)
            engine.retry_backoff_seconds = 0.0
            return await engine.run(targets, catalogs, radius_arcsec=radius)


# ---------------------------------------------------------------------------
# confidence == CrossmatchService posterior
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canary() -> BatchResult:
    return asyncio.run(replay_batch("canary", CANARY_TARGETS, UPLOAD_CATALOGS, CANARY_RADIUS))


@pytest.mark.parametrize("key", list(TARGETS))
def test_batch_confidence_equals_crossmatch_service(canary, key):
    """Same rows -> the same posterior: batch (uploads + XMatch) vs CrossmatchService (recorded cone searches)."""

    async def service_record():
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=replay_side_effect(load_exchanges(key, UPLOAD_CATALOGS)))
            async with offline_client() as client:
                return await make_service(client).crossmatch(*TARGETS[key], radius_arcsec=CANARY_RADIUS,
                                                             catalogs=UPLOAD_CATALOGS)

    record = asyncio.run(service_record())
    service = {(m["catalog"], m["source_id"]): m["confidence"] for m in record.provenance["matches"]}
    got = {(name, m["source_id"]): m["confidence"] for name, ms in canary.target_matches(key).items() for m in ms}
    assert got and set(got) == set(service)
    for pair, value in service.items():
        assert got[pair] == pytest.approx(value, abs=1e-6), pair
    assert canary.target_association(key)["p_any"] == record.provenance["association"]["p_any"]


def test_confidence_is_a_posterior_for_true_counterparts(canary):
    """The reviewer's canary rows: correct identities now carry posteriors near 1 (were 3e-5 .. 0.015)."""
    for key, catalog, sid in (("hd209458", "simbad", "HD 209458"), ("hd209458", "gaia_dr3", "1779546757669063552"),
                              ("hd209458", "twomass_psc", "22031077+1853036"), ("3c273", "first", "FIRST J122906.7+020308"),
                              ("3c273", "gaia_dr3", "3700386905605055360")):
        match = next(m for m in canary.target_matches(key)[catalog] if m["source_id"] == sid)
        assert match["confidence"] > 0.5, (key, catalog, match["confidence"])


# ---------------------------------------------------------------------------
# XMatch: one request per radius bucket
# ---------------------------------------------------------------------------


def test_radius_buckets_are_geometric():
    radii = {i: r for i, r in enumerate([1.0, 1.5, 2.0, 2.01, 3.9, 4.0, 4.1, 170.0])}
    assert radius_buckets(list(radii), radii) == [[0, 1, 2], [3, 4, 5], [6], [7]]


def _count_xmatch(targets, catalogs=("gaia_dr3",), radius=5.0):
    sent: list[tuple[float, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fields = multipart_fields(request.headers["content-type"], request_body(request))
        sent.append((float(fields["distMaxArcsec"]), len(uploaded_targets(request))))
        return httpx.Response(200, content=xmatch_votable([]), headers={"content-type": "text/xml"})

    result = asyncio.run(run_with(handler, targets, list(catalogs), radius, fallback_to_cone=False))
    return result, sent


def test_xmatch_targets_with_individual_epochs_share_requests():
    """1000 targets with distinct epochs (2010-2015, no PM): 3 radius buckets, not ~1000 requests."""
    targets = [{"id": f"t{i}", "ra": (i * 0.37) % 360, "dec": -60 + (i * 0.11) % 120, "epoch": 2010.0 + 5.0 * i / 999}
               for i in range(1000)]
    result, sent = _count_xmatch(targets)
    run = result.runs["gaia_dr3"]
    reg = batch.BatchCrossmatcher().registry
    catalog, _ = batch.xmatch_catalog_definition("gaia_dr3", reg)
    radii = [plan_cone(catalog, t.target, 5.0).radius_arcsec for t in result.targets]
    assert max(radii) / min(radii) > 4.0  # 15.5" .. 68" (10.5"/yr x 1-6 yr + 5")
    assert run.requests == len(sent) <= math.ceil(math.log2(max(radii) / min(radii))) + 1
    assert sum(n for _, n in sent) == 1000 and run.failed_targets == 0


def test_xmatch_per_target_radii_and_fast_movers_share_requests():
    radii_targets = [{"id": f"r{i}", "ra": i * 0.5, "dec": 10.0, "radius_arcsec": 1.0 + 4.0 * i / 59} for i in range(60)]
    result, sent = _count_xmatch(radii_targets)
    assert result.runs["gaia_dr3"].requests == len(sent) == 3  # [1,2), [2,4), [4,5]
    movers = [{"id": f"m{i}", "ra": 20 + i * 0.5, "dec": -5.0, "epoch": 2000.0,
               "pmra": 300.0 + 37.0 * i, "pmdec": -200.0 - 23.0 * i} for i in range(200)]
    result, sent = _count_xmatch(movers)
    assert result.runs["gaia_dr3"].requests == len(sent) <= 4


def test_xmatch_rows_are_cut_to_each_targets_own_cone():
    """Two targets in one bucket (2" and 3.5"): XMatch is asked for 3.5"; each keeps only its own cone."""
    targets = [{"id": "A", "ra": 150.0, "dec": 2.0, "radius_arcsec": 2.0},
               {"id": "B", "ra": 151.0, "dec": 2.0, "radius_arcsec": 3.5}]

    def handler(request: httpx.Request) -> httpx.Response:
        fields = multipart_fields(request.headers["content-type"], request_body(request))
        assert fields["distMaxArcsec"] == "3.500"
        rows = []
        for idx, ra, dec in uploaded_targets(request):
            for k, off in enumerate((1.0, 3.0)):
                rows.append((off, idx, ra, dec, 1000 + 10 * idx + k, ra, dec + off / 3600.0, 0.1, 0.1))
        return httpx.Response(200, content=xmatch_votable(rows), headers={"content-type": "text/xml"})

    result = asyncio.run(run_with(handler, targets, ["gaia_dr3"], 5.0, fallback_to_cone=False))
    assert result.runs["gaia_dr3"].requests == 1
    assert [m["source_id"] for m in result.target_matches("A")["gaia_dr3"]] == ["1000"]
    assert [m["source_id"] for m in result.target_matches("B")["gaia_dr3"]] == ["1010", "1011"]


def test_quadratic_too_wide_filter_is_gone_and_limit_checked_first():
    """20000 targets at epoch 1995 (every Gaia cone > 180"): no XMatch request, fast, reported per target."""
    targets = [{"id": str(i), "ra": (i * 0.017) % 360, "dec": 0.0, "epoch": 1995.0} for i in range(20000)]

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("no request expected")

    started = time.perf_counter()
    result = asyncio.run(run_with(handler, parse_targets(targets), ["gaia_dr3"], 5.0, max_cone_targets=100))
    elapsed = time.perf_counter() - started
    run = result.runs["gaia_dr3"]
    assert run.requests == 0 and run.failed_targets == 20000 and run.fallback_targets == 0
    assert "BATCH_MAX_CONE_TARGETS" in run.errors[-1]
    assert elapsed < 20.0, elapsed  # was ~39 s (quadratic set rebuild) before the fix


# ---------------------------------------------------------------------------
# Ranking: host star before planet
# ---------------------------------------------------------------------------


def test_offset_hd209458_ranks_host_before_planet():
    result = asyncio.run(replay_batch("hd209458_offsets", HD209458_OFFSETS, ["simbad"], 3.0))
    planet_strictly_nearer = 0
    for item in HD209458_OFFSETS:
        matches = result.target_matches(item["id"])["simbad"]
        assert [m["source_id"] for m in matches] == ["HD 209458", "HD 209458b"], item["id"]
        assert [m["rank"] for m in matches] == [1, 2]
        host, planet = matches
        planet_strictly_nearer += planet["separation_arcsec"] < host["separation_arcsec"]
    assert planet_strictly_nearer >= 1  # the float-noise case the tie rule exists for is in the recording

    async def nearest():
        exchanges = load_batch_exchanges("hd209458_offsets")
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=batch_replay_side_effect(exchanges))
            async with offline_client() as client:
                return await BatchCrossmatcher(client=client).run(HD209458_OFFSETS, ["simbad"], radius_arcsec=3.0,
                                                                  nearest_only=True)

    only = asyncio.run(nearest())
    for item in HD209458_OFFSETS:
        assert [m["source_id"] for m in only.target_matches(item["id"])["simbad"]] == ["HD 209458"]


# ---------------------------------------------------------------------------
# REST JSON aliases
# ---------------------------------------------------------------------------


def _router_capture(body: Any, monkeypatch) -> tuple[int, list[Any]]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    captured: list[Any] = []

    async def fake_run(self, targets, catalogs=None, **kwargs):
        captured.extend(targets)
        return BatchResult(list(targets), ["simbad"], 3.0, {"simbad": batch.CatalogRun("simbad", "upload", None, targets=len(targets))},
                           {}, {}, 0.0)

    monkeypatch.setattr(BatchCrossmatcher, "run", fake_run)
    app = FastAPI()
    app.include_router(batch.router)
    with TestClient(app) as client:
        response = client.post("/api/v1/batch/crossmatch?catalogs=simbad", json=body)
    return response.status_code, captured


def test_router_json_body_keeps_alias_fields(monkeypatch):
    body = {"targets": [{"name": "Vega", "ra": 279.2347, "dec": 38.7837, "epoch": 2000, "pmra": 200.94,
                         "pmdec": 286.23, "plx": 130.23, "radius": 10},
                        {"designation": "3C 273", "RAJ2000": 187.2779154, "DEJ2000": 2.0523883}]}
    status, targets = _router_capture(body, monkeypatch)
    assert status == 200
    vega, quasar = targets
    assert vega.id == "Vega" and vega.target.proper_motion == (200.94, 286.23)
    assert vega.target.parallax_mas == 130.23 and vega.radius_arcsec == 10.0 and vega.target.epoch == 2000.0
    assert quasar.id == "3C 273" and quasar.target.ra == pytest.approx(187.2779154)


def test_router_json_body_bad_alias_value_is_422(monkeypatch):
    status, _ = _router_capture({"targets": [{"ra": 1, "dec": 1, "pmra": 5}]}, monkeypatch)
    assert status == 422


def test_router_rejects_oversized_body_from_content_length(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BATCH_MAX_UPLOAD_BYTES", "200")
    app = FastAPI()
    app.include_router(batch.router)
    with TestClient(app) as client:
        response = client.post("/api/v1/batch/crossmatch", content=b"ra,dec\n" + b"1,1\n" * 200,
                               headers={"content-type": "text/csv"})
    assert response.status_code == 413 and "BATCH_MAX_UPLOAD_BYTES" in response.text


# ---------------------------------------------------------------------------
# Target validation speed / equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"ra": -172.7220846, "dec": 2.0523883}, {"ra": "359.9999999", "dec": "90"}, {"ra": -1e-14, "dec": 0},
    {"ra": 10, "dec": -5, "epoch": 2000, "pm_ra_masyr": 5.0, "pm_dec_masyr": -3.0, "parallax_mas": 12.0},
    {"ra": 1, "dec": 95}, {"ra": "x", "dec": 1}, {"ra": float("inf"), "dec": 1}, {"ra": 1, "dec": 1, "epoch": 1500},
    {"ra": 1, "dec": 1, "pm_ra_masyr": 3}, {"ra": 1, "dec": 1, "pm_ra_masyr": 30000, "pm_dec_masyr": 0},
    {"ra": 1, "dec": 1, "parallax_mas": 1200}, {"ra": 1, "dec": 1, "epoch": "abc"},
])
def test_icrs_target_equals_validate_target(kwargs):
    try:
        expected = validate_target(**kwargs)
    except InvalidCoordinateError as exc:
        with pytest.raises(InvalidCoordinateError, match=re.escape(str(exc))):
            batch.icrs_target(**kwargs)
    else:
        assert batch.icrs_target(**kwargs) == expected


def test_parse_targets_scales():
    items = [{"id": str(i), "ra": (i * 0.0171) % 360, "dec": -80 + (i * 0.0013) % 160, "epoch": 2000.0,
              "pmra": 1.0, "pmdec": -2.0} for i in range(20000)]
    started = time.perf_counter()
    targets = parse_targets(items)
    assert len(targets) == 20000
    assert time.perf_counter() - started < 5.0  # was ~11 s (one SkyCoord per target)


# ---------------------------------------------------------------------------
# Silent data loss
# ---------------------------------------------------------------------------


def test_rows_without_target_index_fail_the_chunk():
    """The real SIMBAD answer with its index column renamed (as Xamin prefixes unaliased join columns)."""
    exchange = load_batch_exchanges("canary", ["simbad"])[0]
    renamed = exchange.content.replace(b'"t_idx"', b'"targets_t_idx"')
    assert renamed != exchange.content

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=renamed, headers={"content-type": exchange.content_type})

    result = asyncio.run(run_with(handler, CANARY_TARGETS, ["simbad"], CANARY_RADIUS, fallback_to_cone=False))
    run = result.runs["simbad"]
    assert run.failed_targets == 4 and run.matched_targets == 0
    assert "ResponseParseError" in run.errors[0] and "t_idx" in run.errors[0]
    assert all("ResponseParseError" in result.failures[i]["simbad"] for i in range(4))


def test_targets_whose_rows_cannot_be_converted_are_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        rows = simbad_rows_for(request)
        rows = [[r[0], r[1], None, None] if r[0] == 1 else r for r in rows]  # target 1: no usable position
        return httpx.Response(200, json=simbad_json(rows))

    result = asyncio.run(run_with(handler, CANARY_TARGETS, ["simbad"], 5.0))
    run = result.runs["simbad"]
    assert run.failed_targets == 1 and "ResponseParseError" in result.failures[1]["simbad"]
    assert run.matched_targets == 3 and run.fallback_targets == 0


# ---------------------------------------------------------------------------
# Timeouts, budgets, 4xx
# ---------------------------------------------------------------------------


def test_timeouts_split_chunks_instead_of_resending():
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rows = simbad_rows_for(request)
        sizes.append(len(rows))
        if len(rows) > 1:
            raise httpx.ReadTimeout("simulated 5-minute limit", request=request)
        return httpx.Response(200, json=simbad_json(rows))

    result = asyncio.run(run_with(handler, CANARY_TARGETS, ["simbad"], 5.0))
    run = result.runs["simbad"]
    assert sizes == [4, 4, 2, 2, 2, 2, 1, 1, 1, 1]  # one retry per size, then split
    assert run.split_chunks == 3 and run.matched_targets == 4 and run.fallback_targets == 0


def test_catalog_time_budget_bounds_a_hanging_archive():
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json=simbad_json([]))

    started = time.perf_counter()
    result = asyncio.run(run_with(hang, CANARY_TARGETS, ["simbad"], 5.0, catalog_budget_seconds=0.5))
    assert time.perf_counter() - started < 5.0
    run = result.runs["simbad"]
    assert run.failed_targets == 4 and run.fallback_targets == 0
    assert all("time budget" in result.failures[i]["simbad"] for i in range(4))


def test_deterministic_4xx_does_not_fall_back_to_cones():
    calls = {"upload": 0, "cone": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        kind = "upload" if request.headers.get("content-type", "").startswith("multipart") else "cone"
        calls[kind] += 1
        return httpx.Response(400, text='<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="ERROR">bad</INFO>'
                                        "</RESOURCE></VOTABLE>", headers={"content-type": "text/xml"})

    result = asyncio.run(run_with(handler, [{"ra": i, "dec": 1} for i in range(300)], ["first"], 5.0))
    run = result.runs["first"]
    assert calls == {"upload": 1, "cone": 0}
    assert run.failed_targets == 300 and run.fallback_targets == 0 and run.requests == 1


def test_oversized_answers_are_split_before_parsing():
    one_row = len(json.dumps(simbad_json([[0, "obj-0-0", 1.0, 1.0]])).encode())
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rows = simbad_rows_for(request)
        sizes.append(len(rows))
        return httpx.Response(200, json=simbad_json(rows))

    result = asyncio.run(run_with(handler, CANARY_TARGETS, ["simbad"], 5.0, max_response_bytes=one_row + 40))
    run = result.runs["simbad"]
    assert sizes == [4, 2, 2, 1, 1, 1, 1] and run.split_chunks == 3 and run.matched_targets == 4


# ---------------------------------------------------------------------------
# Circuit breaker, shared guards, per-endpoint concurrency
# ---------------------------------------------------------------------------


def test_cancelled_upload_settles_the_circuit_breaker():
    guards: dict[str, Any] = {}

    async def scenario() -> BatchResult:
        async def hang(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(30)
            return httpx.Response(200, json=simbad_json([]))

        def healthy(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

        async with offline_client() as client:
            engine = BatchCrossmatcher(client=client, guards=guards, fallback_to_cone=False)
            guard = engine._guard(batch.SIMBAD_TAP)
            guard.recovery_seconds = 0.0
            for _ in range(guard.failure_threshold):
                guard.record_failure()  # circuit open; next request is the half-open probe
            with respx.mock(assert_all_called=False) as router:
                router.route().mock(side_effect=hang)
                task = asyncio.create_task(engine.run(CANARY_TARGETS, ["simbad"], radius_arcsec=5.0))
                await asyncio.sleep(0.3)
                assert guard.probe_in_flight
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not guard.probe_in_flight
            with respx.mock(assert_all_called=False) as router:
                router.route().mock(side_effect=healthy)
                return await engine.run(CANARY_TARGETS, ["simbad"], radius_arcsec=5.0)

    result = asyncio.run(scenario())
    assert result.runs["simbad"].requests == 1 and result.runs["simbad"].matched_targets == 4


def test_guard_uses_probe_timeout_setting(monkeypatch):
    monkeypatch.setenv("PROVIDER_PROBE_TIMEOUT_SECONDS", "42")
    assert BatchCrossmatcher()._guard("http://x.test/").probe_timeout_seconds == 42.0


def test_router_engine_shares_api_provider_guards_and_cache():
    from providers import CacheManager, provider_map

    guards: dict[str, Any] = {}
    cache = CacheManager(None)
    providers = provider_map(None, guards=guards, cache=cache)
    state = SimpleNamespace(providers=providers, registry=None, service=None, client=None)
    engine = batch._engine_for(SimpleNamespace(app=SimpleNamespace(state=state)))
    assert engine.guards is guards and engine.cache is cache
    again = batch._engine_for(SimpleNamespace(app=SimpleNamespace(state=state)))
    assert again.upload_slots is engine.upload_slots


def test_uploads_are_limited_per_endpoint_across_catalogs():
    """Five HEASARC catalogs x chunk_concurrency 2 = 10 possible joins; at most BATCH_ENDPOINT_CONCURRENCY run."""
    state = {"now": 0, "peak": 0}

    async def xamin(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return httpx.Response(200, text='<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="ERROR">x</INFO>'
                                        "</RESOURCE></VOTABLE>", headers={"content-type": "text/xml"})

    targets = [{"ra": i, "dec": 1} for i in range(8)]
    result = asyncio.run(run_with(xamin, targets, ["first", "nvss", "rosat", "chandra", "xmm"], 5.0,
                                  chunk_sizes={"heasarc": 2}, endpoint_concurrency=2, chunk_concurrency=2))
    assert sum(r.requests for r in result.runs.values()) == 20
    assert state["peak"] == 2


# ---------------------------------------------------------------------------
# Event loop responsiveness
# ---------------------------------------------------------------------------


def test_large_answer_does_not_block_the_event_loop():
    """3000 targets x 2 SIMBAD rows: parsing, conversion and association run off the event loop."""
    targets = [{"id": str(i), "ra": (i * 0.113) % 360, "dec": -60 + (i * 0.037) % 120} for i in range(3000)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request, per_target=2)))

    async def scenario() -> tuple[BatchResult, float]:
        gaps: list[float] = []
        done = asyncio.Event()

        async def ticker() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=handler)
            async with offline_client() as client:
                result = await BatchCrossmatcher(client=client).run(targets, ["simbad"], radius_arcsec=5.0)
        done.set()
        await tick
        return result, max(gaps)

    result, worst = asyncio.run(scenario())
    assert result.runs["simbad"].matched_targets == 3000 and result.runs["simbad"].total_matches == 6000
    assert worst < 1.0, worst  # the whole conversion (several seconds) used to run on the loop


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_non_utf8_json_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe[\x00{\x00")
    parser = argparse.ArgumentParser(prog="astrosearch")
    batch.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["batch", "--targets", str(bad)])
    assert args.handler(args) == 2
    assert capsys.readouterr().out.startswith("Error:")
    with pytest.raises(BatchError, match="UTF-8"):
        batch.load_targets_file(bad)
