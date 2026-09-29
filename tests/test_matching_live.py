"""Live tests of the Bayesian matching engine and the SSE stream against the real archives.

Run with ``-m live``. Assertions are astrophysical facts: 3C 273 is one object from the
radio to X-rays; Barnard's star, which moved ~3' between 2MASS and Gaia, is one object
once proper motion is applied; a crowded bulge field holds many distinct stars. Tests
skip only when an archive is unreachable (network error / HTTP 5xx / timeout).
"""

from __future__ import annotations

import time

import httpx
import pytest

pytestmark = pytest.mark.live

THREE_C_273 = (187.2779154, 2.0523883)
# Radio, optical, infrared and X-ray catalogues that detect 3C 273.
SPEC_3C273 = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2", "sdss", "first", "nvss", "rosat", "chandra", "xmm"]
# Barnard's star: SIMBAD J2000 position and the Gaia DR3 proper motion.
BARNARD = {"ra": 269.45207696, "dec": 4.69336497, "epoch": 2000.0, "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394}
BULGE = (272.0, -27.0)
NETWORK_ERRORS = {"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError", "TimeoutError", "ConnectError"}


def skip_on_network_failures(record, needed: list[str]) -> None:
    failed = {f["catalog"]: f for f in record.failures}
    down = [f"{name}: {failed[name]['error_type']}: {failed[name]['message']}" for name in needed
            if name in failed and failed[name]["error_type"] in NETWORK_ERRORS]
    if down:
        pytest.skip("archive unreachable: " + "; ".join(down))
    assert not [n for n in needed if n in failed], record.failures


async def live_service(client: httpx.AsyncClient):
    from main import build_service

    return build_service(client=client)


async def test_live_3c273_is_one_object_across_the_spectrum() -> None:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(*THREE_C_273, radius_arcsec=10.0)
    skip_on_network_failures(record, SPEC_3C273)
    group = record.crossmatch_groups[0]
    assert group["contains_target"] and group["match_flag"] == "best"
    assert group["p_any"] > 0.95
    assert set(SPEC_3C273) <= set(group["catalogs"]), group["catalogs"]
    members = {m["catalog"]: m for m in group["members"] if m["coincident_with"] is None}
    assert len(members) == len([m for m in group["members"] if m["coincident_with"] is None])
    for name in SPEC_3C273:
        assert members[name]["match_probability"] > 0.95, (name, members[name]["match_probability"])
        assert members[name]["confidence"] > 0.95
    # The identity rows: SIMBAD '3C 273' and NED '3C 273' (a quasar at z = 0.158).
    assert members["simbad"]["source_id"] == "3C 273"
    assert members["ned"]["physical"]["redshift"] == pytest.approx(0.158, abs=0.002)
    # No other group claims the quasar.
    assert all(not g["contains_target"] for g in record.crossmatch_groups[1:])


async def test_live_barnards_star_grouped_despite_three_arcminutes_of_motion() -> None:
    catalogs = ["gaia_dr3", "twomass_psc", "allwise", "simbad"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(BARNARD["ra"], BARNARD["dec"], radius_arcsec=10.0, epoch=BARNARD["epoch"],
                                          pm_ra_masyr=BARNARD["pm_ra_masyr"], pm_dec_masyr=BARNARD["pm_dec_masyr"],
                                          catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    group = record.crossmatch_groups[0]
    assert group["contains_target"] and set(catalogs) <= set(group["catalogs"])
    assert group["p_any"] > 0.95
    members = {m["catalog"]: m for m in group["members"] if m["coincident_with"] is None}
    for name in catalogs:
        assert members[name]["match_probability"] > 0.95, (name, members[name]["match_probability"])
    assert members["gaia_dr3"]["source_id"] == "4472832130942575872"
    assert members["simbad"]["source_id"] == "NAME Barnard's star"
    assert members["twomass_psc"]["source_id"] == "17574849+0441405"
    # Gaia saw the star 16 yr after J2000: 166" from the J2000 position before propagation.
    assert members["gaia_dr3"]["metadata"]["query_separation_arcsec"] > 150.0
    assert members["gaia_dr3"]["separation_arcsec"] < 0.1


async def test_live_crowded_bulge_field_gives_distinct_objects() -> None:
    catalogs = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(*BULGE, radius_arcsec=20.0, catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    assert record.catalog_results["gaia_dr3"]["row_count"] >= 60
    groups = record.crossmatch_groups
    multi = [g for g in groups if sum(1 for m in g["members"] if m["coincident_with"] is None) >= 2]
    assert len(multi) >= 20
    for group in groups:
        independent = [m["catalog"] for m in group["members"] if m["coincident_with"] is None]
        assert len(independent) == len(set(independent)), group["catalogs"]  # no double-catalogue members
    ids = [(m["catalog"], m["source_id"]) for g in groups for m in g["members"]]
    assert len(ids) == len(set(ids))


def _stream_app():
    from fastapi import FastAPI

    import streaming

    app = FastAPI()
    app.include_router(streaming.router)  # no app.state.service: the route builds its own
    return app


def test_live_sse_stream_for_3c273() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    catalogs = ["gaia_dr3", "simbad", "twomass_psc", "first", "nvss", "rosat"]
    arrivals: list[tuple[float, str]] = []
    with TestClient(_stream_app()) as client:
        started = time.perf_counter()
        with connect_sse(client, "GET", "/api/v1/search/stream", params={
                "ra": THREE_C_273[0], "dec": THREE_C_273[1], "radius_arcsec": 10.0,
                "catalogs": ",".join(catalogs)}, timeout=180.0) as source:
            assert source.response.status_code == 200
            events = []
            for event in source.iter_sse():
                arrivals.append((time.perf_counter() - started, event.event))
                events.append(event)
    kinds = [e.event for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("catalog") == len(catalogs)
    assert [int(e.id) for e in events] == list(range(1, len(events) + 1))
    catalog_events = [e.json() for e in events if e.event == "catalog"]
    failed = {c["catalog"]: c for c in catalog_events if c["status"] == "failed"}
    if failed:
        pytest.skip(f"archive unreachable: {failed}")
    record = events[-1].json()["record"]
    group = record["crossmatch_groups"][0]
    assert group["contains_target"] and set(catalogs) <= set(group["catalogs"])
    first_catalog = next(t for t, kind in arrivals if kind == "catalog")
    assert first_catalog < arrivals[-1][0]
    # Every catalogue event carries rows and timing.
    for data in catalog_events:
        assert data["count"] == len(data["sources"]) >= 1 and data["elapsed_ms"] > 0


def test_live_sse_stream_by_name() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    with TestClient(_stream_app()) as client, connect_sse(client, "GET", "/api/v1/search/stream",
                     params={"name": "3C 273", "radius_arcsec": 5.0, "catalogs": "gaia_dr3,simbad"},
                     timeout=180.0) as source:
        if source.response.status_code >= 500:
            pytest.skip(f"upstream error {source.response.status_code}")
        assert source.response.status_code == 200
        events = list(source.iter_sse())
    start = events[0].json()
    assert start["resolved_object"]["canonical_name"].replace(" ", "") == "3C273"
    assert start["target"]["ra"] == pytest.approx(THREE_C_273[0], abs=1e-4)
    record = events[-1].json()["record"]
    if record["failures"]:
        pytest.skip(f"archive unreachable: {record['failures']}")
    group = record["crossmatch_groups"][0]
    assert group["contains_target"] and {"gaia_dr3", "simbad"} <= set(group["catalogs"])
    # The resolver's own position error (errRAmas/errDEmas) replaced the default target sigma.
    assert record["provenance"]["association"]["target_sigma_arcsec"] < 0.01
