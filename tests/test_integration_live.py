"""End to end against the real archives: api.app under uvicorn, no replay (run with -m live).

The same server as tests/test_integration_e2e.py, with every store in a temporary directory,
queried the way a client would. Upstream outages are not failures of this code: a 502/503/504
answer (the API's report of an archive network error or 5xx) or a catalog failure caused by
one skips the test with the reason.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fixture_io import TARGETS
from test_integration_e2e import UNSET, ServerThread, free_port, server_environment, sse_events, target_group

import alerts
import api
import vizier
from providers import CacheManager

pytestmark = pytest.mark.live

RA, DEC = TARGETS["3c273"]
UPSTREAM_DOWN = {502, 503, 504}


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[httpx.Client]:
    root = tmp_path_factory.mktemp("live")
    with pytest.MonkeyPatch.context() as mp:
        for name, value in server_environment(root).items():
            mp.setenv(name, value)
        for name in UNSET:
            if name != "ANTHROPIC_API_KEY":
                mp.delenv(name, raising=False)
        mp.setenv("PROVIDER_REQUESTS_PER_SECOND", "5")  # polite pacing (the offline suite uses 1000)
        mp.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
        mp.delenv("ASTROSEARCH_SED_OFFLINE_FILTERS", raising=False)
        mp.setattr(api, "cache", CacheManager(None))
        mp.setattr(api.app.state, "quota", api.RequestQuota(None))
        alerts.clear_class_list_cache()
        vizier.clear_describe_cache()
        with ServerThread(free_port()) as server, httpx.Client(base_url=server.base_url, timeout=600.0,
                                                               trust_env=False) as client:
            yield client


def ok(response: httpx.Response, expected: int = 200) -> Any:
    """The JSON body of a successful answer; skip when an upstream archive was unreachable."""
    if response.status_code in UPSTREAM_DOWN:
        pytest.skip(f"upstream unavailable ({response.status_code}): {response.text[:300]}")
    assert response.status_code == expected, response.text[:2000]
    return response.json() if response.headers.get("content-type", "").startswith("application/json") else None


def skip_on_failures(record: dict[str, Any]) -> None:
    failures = record.get("failures") or []
    if failures:
        pytest.skip("archive failures (network or 5xx): " + "; ".join(
            f"{f['catalog']}: {f.get('error_type')}: {str(f.get('message'))[:120]}" for f in failures))


def test_live_search_by_coordinates_name_and_stream(live: httpx.Client) -> None:
    body = {"ra": RA, "dec": DEC, "radius_arcsec": 10.0, "catalogs": ["gaia_dr3", "simbad", "nvss", "twomass_psc"]}
    record = ok(live.post("/api/v1/search", json=body))
    skip_on_failures(record)
    members = target_group(record)["members"]
    assert any(m["catalog"] == "simbad" and m["source_id"] == "3C 273" and m["target_probability"] > 0.9
               for m in members)
    assert {"gaia_dr3", "nvss"} <= {m["catalog"] for m in members}
    named = ok(live.post("/api/v1/search", json={"name": "3C 273", "radius_arcsec": 10.0, "catalogs": ["simbad"]}))
    skip_on_failures(named)
    assert named["resolved_object"]["canonical_name"] == "3C 273"
    with live.stream("GET", "/api/v1/search/stream", params={"ra": str(RA), "dec": str(DEC), "radius_arcsec": 10,
                                                             "catalogs": "simbad,gaia_dr3"}) as response:
        assert response.status_code == 200
        events = sse_events(response)
    assert events[0][0] == "start" and events[-1][0] in {"done", "error"}
    if events[-1][0] == "error":
        pytest.skip(f"stream ended with an upstream error: {events[-1][1]}")
    skip_on_failures(events[-1][1]["record"])


def test_live_vo_cone_search_and_tap(live: httpx.Client) -> None:
    scs = live.get("/vo/scs", params={"RA": RA, "DEC": DEC, "SR": 10.0 / 3600.0})
    ok(scs)
    assert b"<VOTABLE" in scs.content and b"3C 273" in scs.content
    tap = live.get("/vo/tap/sync", params={"REQUEST": "doQuery", "LANG": "ADQL", "QUERY":
                   f"SELECT catalog, source_id FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                   f"CIRCLE('ICRS', {RA}, {DEC}, 0.002)) AND catalog = 'simbad'"})
    ok(tap)
    assert b"3C 273" in tap.content


def test_live_sed_lightcurves_cutouts(live: httpx.Client) -> None:
    sed = ok(live.get("/api/v1/sed", params={"name": "3C 273", "radius_arcsec": 10}))
    assert sed["classification"]["label"] == "qso" and sed["redshift"]["value"] == pytest.approx(0.158, abs=0.002)
    curves = ok(live.get("/api/v1/lightcurves", params={"name": "3C 273", "surveys": "gaia"}))
    assert curves["target"]["name"] == "3C 273" and any(s["key"] == "gaia:G" for s in curves["series"])
    png = live.get("/api/v1/cutouts", params={"ra": RA, "dec": DEC, "fov_arcmin": 2, "survey": "dss2", "width": 128,
                                              "height": 128})
    ok(png)
    assert png.headers["content-type"] == "image/png" and png.content.startswith(b"\x89PNG")
    stack = ok(live.get("/api/v1/cutouts/stack", params={"name": "3C 273", "fov_arcmin": 2}))
    assert stack["panels"]


def test_live_solar_system(live: httpx.Client) -> None:
    import asyncio

    import timedomain as td

    # Where JPL Horizons puts Ceres (geocentric, astrometric ICRF) on 2024-01-01 00:00 UTC ...
    epoch = 60310.0
    try:
        ceres = asyncio.run(td.horizons_ephemeris("1;", epoch))[0]
    except (httpx.HTTPError, td.UpstreamServiceError) as exc:
        pytest.skip(f"JPL Horizons unavailable: {exc}")
    # ... is where the API's SkyBoT cone search finds it.
    body = ok(live.get("/api/v1/solar-system", params={"ra": ceres.ra, "dec": ceres.dec, "epoch_mjd": epoch,
                                                       "radius_arcsec": 600}))
    found = {obj["name"]: obj for obj in body["objects"]}
    assert "Ceres" in found, sorted(found)[:10]
    assert found["Ceres"]["separation_arcsec"] < 10


def test_live_batch_and_dataset(live: httpx.Client) -> None:
    targets = [{"id": key, "ra": ra, "dec": dec} for key, (ra, dec) in list(TARGETS.items())[:2]]
    result = ok(live.post("/api/v1/batch/crossmatch", json={"targets": targets, "catalogs": ["simbad"],
                                                            "radius_arcsec": 5.0}))
    by_id = {t["id"]: t for t in result["targets"]}
    if by_id["3c273"]["failures"]:
        pytest.skip(f"batch upload failed upstream: {by_id['3c273']['failures']}")
    assert any(m["source_id"] == "3C 273" for m in by_id["3c273"]["matches"]["simbad"])
    created = ok(live.post("/api/v1/datasets/create", json={
        "name": "live", "profile": "full", "radius_arcsec": 5.0, "catalogs": ["simbad"], "count_threshold": 1,
        "export_format": "json", "targets": [{"ra": t["ra"], "dec": t["dec"]} for t in targets]}), expected=202)
    deadline = time.monotonic() + 600
    while (dataset := ok(live.get(f"/api/v1/datasets/{created['id']}")))["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(1.0)
    assert dataset["status"] == "completed", dataset
    if dataset["failures"]:
        pytest.skip(f"dataset archive failures: {dataset['failures'][:2]}")
    assert dataset["method"] == "batch" and dataset["total_sources"] >= 2
    rows = live.get(f"/api/v1/datasets/{created['id']}/export").json()
    assert any(r["source_id"] == "3C 273" and r["contains_target"] for r in rows)


def test_live_vizier_provenance_and_citations(live: httpx.Client) -> None:
    table = ok(live.get("/api/v1/vizier/catalog/I/355/gaiadr3"))
    assert table["kind"] == "table" and table["registrable"] is True
    built = ok(live.post("/api/v1/provenance/manifest", json={"ra": RA, "dec": DEC, "radius_arcsec": 5.0,
                                                              "catalogs": ["simbad"], "include_record": True}))
    skip_on_failures(built["record"])
    assert built["manifest"]["content_hash"].startswith("sha256:")
    cites = ok(live.get("/api/v1/citations", params={"catalogs": "gaia_dr3,simbad"}))
    assert cites["unknown"] == [] and cites["keys"]


def test_live_skycache_mirror_and_local_search(live: httpx.Client) -> None:
    report = ok(live.post("/api/v1/skycache/mirror", json={"catalog": "gaia_dr3", "ra": RA, "dec": DEC,
                                                           "radius_deg": 0.02}))
    if report["tiles_failed"]:
        pytest.skip(f"mirror tiles failed upstream: {report['warnings'][:2]}")
    cone = ok(live.get("/api/v1/skycache/cone", params={"catalog": "gaia_dr3", "ra": RA, "dec": DEC,
                                                        "radius_arcsec": 10}))
    assert cone["sources"]
    record = ok(live.post("/api/v1/search", json={"ra": RA, "dec": DEC, "radius_arcsec": 10, "catalogs": ["gaia_dr3"],
                                                  "min_confidence": 0}))
    sources = record["catalog_results"]["gaia_dr3"]["sources"]
    assert sources and all("skycache" in s["provenance"] for s in sources)
    ok(live.delete("/api/v1/skycache/gaia_dr3"))


def test_live_alerts_and_ai(live: httpx.Client) -> None:
    brokers = ok(live.get("/api/v1/alerts/brokers"))
    assert {"alerce", "fink"} <= {b["name"] for b in brokers}
    polled = ok(live.post("/api/v1/alerts/poll", json={"broker": "alerce", "limit": 2, "crossmatch": False}))
    if polled.get("error"):
        pytest.skip(f"ALeRCE: {polled['error']}")
    listed = ok(live.get("/api/v1/alerts"))
    assert listed["count"] >= polled["inserted"]
    response = live.post("/api/v1/ai/query", json={"text": "quasars within 2 arcmin of M87"})
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        compiled = ok(response)
        assert compiled["advanced_query"]["target"]["ra"] == pytest.approx(187.7059, abs=0.01)
    else:
        assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.json()["detail"]
