"""End to end: the real application (api.app) served by uvicorn on a free port, every route of every module.

The server runs in a thread of the test process with every store in a temporary directory.
Each test replays recorded archive answers for the upstream services its routes call (respx
patches httpx for the server's own HTTP client; requests to the server itself pass through),
so the whole stack -- middleware, routers, app.state objects, the crossmatch engine, the
archive adapters and their parsers -- runs exactly as in production, offline:

* catalog cones: the 3C 273 canary recordings (tests/fixtures/3c273), answered by URL;
* Sesame: the recorded SIMBAD answer for 3C 273;
* batch uploads / XMatch: tests/fixtures/batch/canary; sky mirror: tests/fixtures/skycache;
* the modules' own recordings for VizieR, SED, light curves, SkyBoT, hips2fits, AI and alerts.

The live-marked variant (tests/test_integration_live.py) runs the same server against the
real archives.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
import pyarrow.csv as pacsv
import pytest
import respx
import uvicorn
from astropy.io.votable import parse as parse_votable
from fixture_io import FIXTURES, TARGETS, FixtureMismatch, load_exchanges, replay_side_effect

import alerts
import api
import vizier
from providers import CacheManager

RA, DEC = TARGETS["3c273"]
SESAME_3C273 = (FIXTURES / "sesame" / "3c273.xml").read_text(encoding="utf-8")
CANARY = [{"id": key, "ra": ra, "dec": dec} for key, (ra, dec) in TARGETS.items()]
VOTABLE = "application/x-votable+xml"


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerThread:
    """uvicorn.Server running api.app (lifespan included) in a background thread."""

    def __init__(self, port: int) -> None:
        config = uvicorn.Config(api.app, host="127.0.0.1", port=port, lifespan="on", log_level="warning",
                                access_log=False, timeout_graceful_shutdown=10)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="astrosearch-e2e-server", daemon=True)
        self.base_url = f"http://127.0.0.1:{port}"

    def __enter__(self) -> Self:
        self.thread.start()
        deadline = time.monotonic() + 120
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("the API server did not start (see the log above)")
            if time.monotonic() > deadline:
                raise TimeoutError("the API server did not start within 120 s")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=60)
        assert not self.thread.is_alive(), "the API server did not shut down"


def server_environment(root: Path) -> dict[str, str]:
    return {
        "DATASET_STORAGE_PATH": str(root / "datasets"),
        "SKYCACHE_PATH": str(root / "skycache"),
        "CATALOG_REGISTRY_PATH": str(root / "catalogs.yaml"),
        "CUTOUT_CACHE_DIR": str(root / "cutouts"),
        "ASTROSEARCH_CACHE_DIR": str(root / "cache"),
        "API_SEARCH_CACHE_TTL_SECONDS": "0",
        "API_RATE_LIMIT_PER_MINUTE": "0",
        "TIMEDOMAIN_CACHE_TTL_SECONDS": "0",
        "ASTROSEARCH_SED_OFFLINE_FILTERS": "true",
        "DATASET_BATCH_MIN_TARGETS": "2",
    }


UNSET = ("API_KEYS", "JWT_SECRET", "JWT_PUBLIC_KEY", "REQUIRE_API_KEY", "REDIS_URL", "ANTHROPIC_API_KEY",
         "ANTHROPIC_AUTH_TOKEN", "CORS_ORIGINS", "MAX_REQUEST_BYTES", "ALERTS_DATABASE_URL", "SKYCACHE_ENABLED")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerThread]:
    root = tmp_path_factory.mktemp("e2e")
    with pytest.MonkeyPatch.context() as mp:
        for name, value in server_environment(root).items():
            mp.setenv(name, value)
        for name in UNSET:
            mp.delenv(name, raising=False)
        mp.setattr(api, "cache", CacheManager(None))
        mp.setattr(api.app.state, "quota", api.RequestQuota(None))
        alerts.clear_class_list_cache()
        vizier.clear_describe_cache()
        with ServerThread(free_port()) as running:
            running.root = root  # type: ignore[attr-defined]
            yield running


@pytest.fixture
def http(server: ServerThread) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=server.base_url, timeout=180.0, trust_env=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Upstream replay
# ---------------------------------------------------------------------------

CATALOG_CONES = replay_side_effect(load_exchanges("3c273"), strict=False)


def chain(*handlers: Callable[[httpx.Request], httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    """The first handler that has a recording for the request answers it."""
    def handler(request: httpx.Request) -> httpx.Response:
        misses: list[str] = []
        for candidate in handlers:
            try:
                response = candidate(request)
            except FixtureMismatch as exc:
                misses.append(str(exc).splitlines()[0])
                continue
            if response.status_code == 599:  # replay_side_effect(strict=False): nothing recorded there
                misses.append(response.text)
                continue
            return response
        raise FixtureMismatch(f"no recording for {request.method} {request.url}: {misses}")

    return handler


def _unexpected(request: httpx.Request) -> httpx.Response:
    raise FixtureMismatch(f"no upstream request expected: {request.method} {request.url}")


@contextlib.contextmanager
def upstream(*handlers: Callable[[httpx.Request], httpx.Response], sesame: str | None = SESAME_3C273
             ) -> Iterator[respx.MockRouter]:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route(host="127.0.0.1").pass_through()
    if sesame is not None:
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").respond(
            200, text=sesame, headers={"content-type": "text/xml"})
    router.route().mock(side_effect=chain(*handlers) if handlers else _unexpected)
    with router:
        yield router


def sse_events(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    kind, data = None, []
    for line in response.iter_lines():
        if line.startswith("event:"):
            kind = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].strip())
        elif line == "" and kind is not None:
            events.append((kind, json.loads("\n".join(data))))
            kind, data = None, []
    return events


def target_group(record: dict[str, Any]) -> dict[str, Any]:
    return next(g for g in record["crossmatch_groups"] if g["contains_target"])


# ---------------------------------------------------------------------------
# OpenAPI, UI, core endpoints
# ---------------------------------------------------------------------------


def test_openapi_lists_every_route_and_the_ui_is_served(http: httpx.Client) -> None:
    spec = http.get("/api/openapi.json").json()
    documented = {(m.upper(), p) for p, item in spec["paths"].items() for m in item}
    served = {(m, p.replace(":path}", "}")) for p, methods in api.route_table() for m in methods
              if m != "HEAD" and p not in {"/", "/index.html", "/app.js", "/styles.css", "/docs/oauth2-redirect",
                                          "/api/docs", "/api/redoc", "/api/openapi.json"}}
    assert served == documented
    # Exactly the routes every module is expected to contribute (tests/test_integration_api.py).
    from test_integration_api import MODULE_ROUTES

    assert documented == {(m, p.replace(":path}", "}")) for m, p in set().union(*MODULE_ROUTES.values())}
    for path, media in (("/", "text/html"), ("/index.html", "text/html"), ("/app.js", "text/javascript"),
                        ("/styles.css", "text/css")):
        response = http.get(path)
        assert response.status_code == 200 and response.headers["content-type"].startswith(media), path
    assert "AstroSearch Sky Explorer" in http.get("/").text
    assert http.head("/").status_code == 200
    assert http.get("/api/docs").status_code == 200 and http.get("/api/redoc").status_code == 200


def test_core_endpoints(http: httpx.Client) -> None:
    health = http.get("/api/v1/health")
    assert health.status_code == 200 and health.json()["status"] == "healthy"
    assert health.headers["x-request-id"]
    catalogs = http.get("/api/v1/catalogs").json()
    assert {"gaia_dr3", "simbad", "twomass_psc", "xmm"} <= set(catalogs)
    assert http.get("/api/v1/catalogs/gaia_dr3").json()["name"] == "gaia_dr3"
    assert http.get("/api/v1/catalogs/nope").status_code == 404
    stats = http.get("/api/v1/stats").json()
    assert set(stats) == {"datasets", "sources_exported", "saved_queries"}
    monitoring = http.get("/api/v1/monitoring").json()
    assert monitoring["status"] == "healthy" and monitoring["skycache"]["enabled"] is True
    metrics = http.get("/api/v1/monitoring/metrics")
    assert metrics.status_code == 200 and "astrosearch_requests_total" in metrics.text


def test_saved_queries_crud(http: httpx.Client) -> None:
    created = http.post("/api/v1/queries", json={"name": "3C 273 radio", "query": {"name": "3C 273", "profile": "radio"}})
    assert created.status_code == 201
    query_id = created.json()["id"]
    assert any(q["id"] == query_id for q in http.get("/api/v1/queries").json())
    assert http.post("/api/v1/queries", json={"name": "bad", "query": {"radius_arcsec": 3}}).status_code == 422
    assert http.delete(f"/api/v1/queries/{query_id}").status_code == 204
    assert http.delete(f"/api/v1/queries/{query_id}").status_code == 404


# ---------------------------------------------------------------------------
# Search: coordinates, names, batch of searches, streaming
# ---------------------------------------------------------------------------


def test_search_by_coordinates_and_by_name(http: httpx.Client) -> None:
    body = {"ra": RA, "dec": DEC, "radius_arcsec": 10.0, "catalogs": ["gaia_dr3", "simbad", "nvss"]}
    with upstream(CATALOG_CONES):
        record = http.post("/api/v1/search", json=body)
        named = http.post("/api/v1/search", json={"name": "3C 273", "radius_arcsec": 10.0,
                                                  "catalogs": ["gaia_dr3", "simbad", "nvss"]})
        many = http.post("/api/v1/search/batch", json=[body, {"radius_arcsec": 5}])
    assert record.status_code == 200, record.text
    data = record.json()
    assert set(data) >= {"target", "catalogs_queried", "catalog_results", "counterparts", "failures", "provenance",
                         "crossmatch_groups"}
    assert data["failures"] == [] and data["catalogs_queried"] == 3
    group = target_group(data)
    assert {"gaia_dr3", "simbad", "nvss"} <= {m["catalog"] for m in group["members"]}
    assert any(m["source_id"] == "3C 273" and m["target_probability"] > 0.9 for m in group["members"])
    assert data["provenance"]["association"]["method"].startswith("Bayesian N-way")

    assert named.status_code == 200, named.text
    named_data = named.json()
    assert named_data["resolved_object"]["canonical_name"] == "3C 273"
    assoc = named_data["provenance"]["association"]
    assert assoc["target_sigma_source"] == "resolver"
    assert any(r["source_id"] == "3C 273" and r["reason"] == "resolved name" for r in assoc["identity_rows"])

    assert many.status_code == 200
    first, second = many.json()
    assert first["failures"] == [] and second["status_code"] == 422
    metrics = http.get("/api/v1/monitoring/metrics").text
    assert 'astrosearch_catalog_queries_total{catalog="simbad",status="success"}' in metrics


def test_search_stream_sends_catalog_group_and_done_events(http: httpx.Client) -> None:
    params = {"ra": str(RA), "dec": str(DEC), "radius_arcsec": 10, "catalogs": "gaia_dr3,simbad"}
    with upstream(CATALOG_CONES):
        with http.stream("GET", "/api/v1/search/stream", params=params) as response:
            assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
            events = sse_events(response)
        with http.stream("GET", "/api/v1/search/stream", params={"name": "3C 273", "radius_arcsec": 10,
                                                                  "catalogs": "simbad"}) as named:
            named_events = sse_events(named)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("catalog") == 2 and "group" in kinds
    assert {data["catalog"] for kind, data in events if kind == "catalog"} == {"gaia_dr3", "simbad"}
    record = events[-1][1]["record"]
    assert target_group(record)["contains_target"] is True
    assert named_events[0][1]["resolved_object"]["canonical_name"] == "3C 273"
    assert named_events[-1][0] == "done"
    assert http.get("/api/v1/search/stream", params={"ra": "10"}).status_code == 422


# ---------------------------------------------------------------------------
# Datasets (batch mode) and the batch router
# ---------------------------------------------------------------------------


def _batch_uploads(catalogs: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    from test_batch_fixtures import batch_replay_side_effect, load_batch_exchanges

    return batch_replay_side_effect(load_batch_exchanges("canary", catalogs))


def test_dataset_lifecycle_uses_the_batch_engine(http: httpx.Client) -> None:
    request = {"name": "canary", "profile": "full", "radius_arcsec": 10.0, "catalogs": ["gaia_dr3", "simbad"],
               "count_threshold": 1, "export_format": "csv",
               "targets": [{"ra": t["ra"], "dec": t["dec"]} for t in CANARY]}
    with upstream(_batch_uploads(["gaia_dr3", "simbad"])):
        created = http.post("/api/v1/datasets/create", json=request)
        assert created.status_code == 202, created.text
        location = created.headers["location"]
        deadline = time.monotonic() + 120
        while (dataset := http.get(location).json())["status"] in {"queued", "running"}:
            assert time.monotonic() < deadline, dataset
            time.sleep(0.2)
    assert dataset["status"] == "completed", dataset
    assert dataset["method"] == "batch" and dataset["batch"]["request_count"] == 2
    assert dataset["total_sources"] > 0 and set(dataset["catalogs_used"]) == {"gaia_dr3", "simbad"}
    export = http.get(f"{location}/export")
    assert export.status_code == 200
    table = pacsv.read_csv(io.BytesIO(export.content))
    assert {"match_probability", "target_probability", "group_id", "contains_target", "p_any",
            "target_index"} <= set(table.column_names)
    assert table.num_rows == dataset["total_sources"]
    assert any(d["id"] == dataset["id"] for d in http.get("/api/v1/datasets").json())
    assert http.get("/api/v1/stats").json()["datasets"] >= 1
    assert http.delete(location).status_code == 204
    assert http.get(location).status_code == 404
    assert http.get(f"{location}/export").status_code == 404
    bad = http.post("/api/v1/datasets/create", json={**request, "catalogs": ["nope"]})
    assert bad.status_code == 422


def test_batch_router(http: httpx.Client) -> None:
    strategies = http.get("/api/v1/batch/strategies").json()
    assert strategies["catalogs"]["simbad"]["strategy"] == "upload" and strategies["max_radius_arcsec"] == 180.0
    body = {"targets": CANARY, "catalogs": ["simbad"], "radius_arcsec": 10.0}
    with upstream(_batch_uploads(["simbad"])):
        result = http.post("/api/v1/batch/crossmatch", json=body)
        as_csv = http.post("/api/v1/batch/crossmatch", params={"format": "csv"}, json=body)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["summary"]["targets"] == 4 and data["summary"]["strategies"] == {"simbad": "upload"}
    assert result.headers["x-batch-request-count"] == "1" and result.headers["x-batch-failed-targets"] == "0"
    by_id = {t["id"]: t for t in data["targets"]}
    assert any(m["source_id"] == "3C 273" for m in by_id["3c273"]["matches"]["simbad"])
    assert as_csv.status_code == 200 and as_csv.headers["content-type"].startswith("text/csv")
    assert "confidence" in as_csv.text.splitlines()[0]
    assert http.post("/api/v1/batch/crossmatch", json={**body, "catalogs": ["nope"]}).status_code == 422


# ---------------------------------------------------------------------------
# Sky cache: mirror, local cone, local-first search, delete
# ---------------------------------------------------------------------------


def test_skycache_mirror_cone_local_search_and_delete(http: httpx.Client) -> None:
    assert http.get("/api/v1/skycache/status").json()["catalogs"] == []
    mirror = {"catalog": "gaia_dr3", "ra": RA, "dec": DEC, "radius_deg": 0.2}
    with upstream(replay_side_effect(load_exchanges("skycache/3c273_mirror", ["gaia_dr3"]))):
        report = http.post("/api/v1/skycache/mirror", json=mirror)
    assert report.status_code == 200, report.text
    assert report.json()["tiles_failed"] == 0 and report.json()["rows_stored"] > 400
    status = {entry["catalog"]: entry for entry in http.get("/api/v1/skycache/status").json()["catalogs"]}
    assert status["gaia_dr3"]["rows"] == report.json()["rows_total"]
    with upstream():  # nothing may reach an archive now
        cone = http.get("/api/v1/skycache/cone", params={"catalog": "gaia_dr3", "ra": RA, "dec": DEC,
                                                         "radius_arcsec": 10})
        record = http.post("/api/v1/search", json={"ra": RA, "dec": DEC, "radius_arcsec": 10,
                                                   "catalogs": ["gaia_dr3"], "min_confidence": 0})
    assert cone.status_code == 200 and cone.json()["sources"]
    assert record.status_code == 200 and record.json()["failures"] == []
    sources = record.json()["catalog_results"]["gaia_dr3"]["sources"]
    assert sources and all(s["provenance"]["skycache"]["store"].startswith("file:") for s in sources)
    assert http.get("/api/v1/skycache/cone", params={"catalog": "gaia_dr3", "ra": 10, "dec": 10,
                                                     "radius_arcsec": 10}).status_code == 409
    assert http.delete("/api/v1/skycache/gaia_dr3").status_code == 200
    assert http.delete("/api/v1/skycache/gaia_dr3").status_code == 404
    assert http.post("/api/v1/skycache/mirror", json={**mirror, "catalog": "nope"}).status_code == 404


# ---------------------------------------------------------------------------
# VizieR discovery and registration into the running service
# ---------------------------------------------------------------------------


def _vizier(*cases: str) -> Callable[[httpx.Request], httpx.Response]:
    return replay_side_effect([e for case in cases for e in load_exchanges(f"vizier/{case}")])


def test_vizier_search_describe_register(http: httpx.Client, server: ServerThread) -> None:
    with upstream(_vizier("search_gaia_dr3", "describe_2sxps", "describe_catalog_vii250")):
        search = http.get("/api/v1/vizier/search", params={"q": "Gaia DR3"})
        table = http.get("/api/v1/vizier/catalog/IX/58/2sxps")
        catalog = http.get("/api/v1/vizier/catalog/VII/250")
        registered = http.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "swift_2sxps"})
    assert search.status_code == 200 and search.json()["tables"][0]["table_id"] == "I/355/gaiadr3"
    assert table.json()["kind"] == "table" and table.json()["registrable"] is True
    assert catalog.json()["kind"] == "catalog" and len(catalog.json()["tables"]) == 4
    assert registered.status_code == 200, registered.text
    assert registered.json()["attached"] is True
    assert Path(registered.json()["path"]) == server.root / "catalogs.yaml"  # type: ignore[attr-defined]
    # The running service, the catalog list and dataset validation know the new catalog at once.
    assert "swift_2sxps" in http.get("/api/v1/catalogs").json()
    listed = http.get("/api/v1/vizier/registered").json()
    assert [c["name"] for c in listed["catalogs"]] == ["swift_2sxps"] and listed["catalogs"][0]["active"]
    assert http.get("/api/v1/vizier/search").status_code == 422


# ---------------------------------------------------------------------------
# SED, light curves, solar system, cutouts
# ---------------------------------------------------------------------------


def test_sed_get_and_post(http: httpx.Client) -> None:
    supplementary = replay_side_effect(load_exchanges("sed/3c273") + load_exchanges("sed/svo"))
    with upstream(supplementary, CATALOG_CONES):
        got = http.get("/api/v1/sed", params={"ra": RA, "dec": DEC, "radius_arcsec": 10})
        posted = http.post("/api/v1/sed", json={"ra": RA, "dec": DEC, "radius_arcsec": 10})
    assert got.status_code == 200, got.text
    sed = got.json()
    assert set(sed) >= {"target", "points", "classification", "redshift", "members", "notes"}
    assert sed["classification"]["label"] == "qso" and sed["redshift"]["kind"] == "spec"
    assert {"band", "wavelength_um", "flux_jy", "nu_fnu_erg_s_cm2"} <= set(sed["points"][0])
    assert posted.status_code == 200 and posted.json()["redshift"]["value"] == pytest.approx(0.158339, abs=1e-5)
    assert http.get("/api/v1/sed", params={"ra": 400, "dec": 0}).status_code == 422


def test_lightcurves_and_solar_system(http: httpx.Client) -> None:
    from test_timedomain import load_case, replay_handler

    import timedomain as td

    with upstream(replay_handler("name_3c273_gaia"), sesame=None):
        curves = http.get("/api/v1/lightcurves", params={"name": "3C 273", "surveys": "gaia"})
    assert curves.status_code == 200, curves.text
    body = curves.json()
    assert body["target"]["name"] == "3C 273" and {s["key"] for s in body["series"]} >= {"gaia:G"}
    assert set(body) >= {"target", "series", "variability", "period", "failures", "provenance"}

    eph = next(e for e in load_case("vesta_2023") if "horizons" in e["url"])
    point = td.parse_horizons_result(json.loads(eph["content"])["result"])[0]
    with upstream(replay_handler("vesta_2023"), sesame=None):
        objects = http.get("/api/v1/solar-system", params={"ra": f"{point.ra:.7f}", "dec": f"{point.dec:.7f}",
                                                           "epoch_mjd": 60000.0, "radius_arcsec": 600})
    assert objects.status_code == 200, objects.text
    assert objects.json()["objects"][0]["name"] == "Vesta"
    assert http.get("/api/v1/solar-system", params={"ra": 10, "dec": 100}).status_code == 422
    assert http.get("/api/v1/lightcurves", params={"ra": 10, "dec": 10, "surveys": "kepler"}).status_code == 422


def test_cutouts_surveys_and_stack(http: httpx.Client) -> None:
    surveys = http.get("/api/v1/cutouts/surveys")
    assert surveys.status_code == 200 and any(s["key"] == "dss2" for s in surveys.json())
    with upstream(replay_side_effect(load_exchanges("imaging", ["dss2_3c273_png", "coverage_3c273"]))):
        png = http.get("/api/v1/cutouts", params={"ra": RA, "dec": DEC, "fov_arcmin": 2.0, "survey": "dss2",
                                                  "width": 128, "height": 128, "no_cache": True})
        stack = http.get("/api/v1/cutouts/stack", params={"ra": RA, "dec": DEC, "fov_arcmin": 3})
    assert png.status_code == 200, png.text
    assert png.headers["content-type"] == "image/png" and png.content.startswith(b"\x89PNG")
    assert png.headers["x-cutout-hips"] == "CDS/P/DSS2/color"
    assert stack.status_code == 200, stack.text
    panels = stack.json()["panels"]
    assert stack.json()["coverage_checked"] and all(p["url"].startswith("/api/v1/cutouts?") for p in panels)
    assert http.get("/api/v1/cutouts", params={"ra": 10, "dec": 10, "format": "gif"}).status_code == 422


# ---------------------------------------------------------------------------
# AI (Claude replaced by a scripted client), provenance, VO, alerts
# ---------------------------------------------------------------------------


def test_ai_query_and_explain(http: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_ai import FakeAnthropic, reply, submission, tool_use

    with upstream():
        missing = http.post("/api/v1/ai/query", json={"text": "quasars near M87"})
        assert missing.status_code == 503 and "ANTHROPIC_API_KEY" in missing.json()["detail"]
        assert http.post("/api/v1/ai/query", json={"text": "  "}).status_code == 422
    monkeypatch.setattr(api.app.state, "anthropic", FakeAnthropic(reply(tool_use("submit_query", submission()))),
                        raising=False)
    with upstream(replay_side_effect(load_exchanges("ai/sesame")), sesame=None):
        compiled = http.post("/api/v1/ai/query", json={"text": "quasars near M87 with radio emission"})
    assert compiled.status_code == 200, compiled.text
    assert set(compiled.json()) >= {"advanced_query", "plan", "explanation", "adql", "scope"}
    assert compiled.json()["advanced_query"]["target"]["ra"] == pytest.approx(187.70593077)
    with upstream(replay_side_effect(load_exchanges("ai/explain_3c273")), sesame=None):
        explained = http.post("/api/v1/ai/explain", json={"name": "3C 273", "facts_only": True,
                                                          "include_crossmatch": False})
    assert explained.status_code == 200, explained.text
    assert explained.json()["facts"]["identity"]["main_id"] == "3C 273" and explained.json()["summary"] is None


def test_provenance_manifest_replay_and_citations(http: httpx.Client) -> None:
    fields = {"ra": RA, "dec": DEC, "radius_arcsec": 10.0, "catalogs": ["gaia_dr3", "simbad"],
              "live_release_lookup": False}
    with upstream(CATALOG_CONES):
        built = http.post("/api/v1/provenance/manifest", json=fields)
        assert built.status_code == 200, built.text
        manifest = built.json()["manifest"]
        replayed = http.post("/api/v1/provenance/replay", json={"manifest": manifest})
    assert manifest["schema"].startswith("astrosearch.provenance.manifest/")
    assert set(manifest["catalogs"]) == {"gaia_dr3", "simbad"} and manifest["content_hash"].startswith("sha256:")
    assert replayed.status_code == 200, replayed.text
    result = replayed.json()
    assert result["content_hash_equal"] is True and result["identical"] is True
    cites = http.get("/api/v1/citations", params={"catalogs": "gaia_dr3,simbad"})
    assert cites.status_code == 200 and set(cites.json()) >= {"keys", "acknowledgements", "bibtex", "unknown"}
    bibtex = http.get("/api/v1/citations", params={"catalogs": "gaia_dr3", "format": "bibtex"})
    assert bibtex.status_code == 200 and "@ARTICLE" in bibtex.text.upper()
    assert http.get("/api/v1/citations/sources").status_code == 200
    assert http.post("/api/v1/provenance/replay", json={"manifest": {"schema": "x"}}).status_code == 422


def _tap(query: str, **extra: Any) -> dict[str, Any]:
    return {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query, **extra}


CONE_ADQL = (f"SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {RA}, {DEC}, {10.0 / 3600.0!r}))")


def test_vo_services(http: httpx.Client) -> None:
    for path in ("/vo/tap", "/vo/tap/availability", "/vo/tap/capabilities", "/vo/tap/tables",
                 "/vo/tap/tables/astrosearch.matches", "/vo/tap/examples"):
        response = http.get(path)
        assert response.status_code == 200, (path, response.text[:200])
    with upstream(CATALOG_CONES):
        scs = http.get("/vo/scs", params={"RA": RA, "DEC": DEC, "SR": 10.0 / 3600.0})
        sync_get = http.get("/vo/tap/sync", params=_tap(CONE_ADQL))
        sync_post = http.post("/vo/tap/sync", data=_tap(CONE_ADQL + " AND catalog = 'simbad'"))
        created = http.post("/vo/tap/async", data=_tap(CONE_ADQL, RUNID="e2e"), follow_redirects=False)
        assert created.status_code == 303
        job = urlsplit(created.headers["location"]).path
        assert http.get(job + "/phase").text == "PENDING"
        assert http.post(job + "/parameters", data={"MAXREC": "50"}).status_code in {200, 303}
        assert "MAXREC" in http.get(job + "/parameters").text.upper()
        http.post(job + "/executionduration", data={"EXECUTIONDURATION": "120"})
        assert http.get(job + "/executionduration").text == "120"
        # The service may shorten a requested destruction time to its retention limit.
        assert http.post(job + "/destruction", data={"DESTRUCTION": "2099-01-01T00:00:00Z"},
                         follow_redirects=False).status_code in {200, 303}
        assert datetime.fromisoformat(http.get(job + "/destruction").text) > datetime.now(UTC)
        assert http.get(job + "/quote").status_code == 200 and http.get(job + "/owner").status_code == 200
        assert http.post(job + "/phase", data={"PHASE": "RUN"}, follow_redirects=False).status_code == 303
        deadline = time.monotonic() + 120
        while (phase := http.get(job + "/phase").text) not in {"COMPLETED", "ERROR"}:
            assert time.monotonic() < deadline, phase
            http.get(job, params={"WAIT": 5, "PHASE": phase})
    # Cone Search 1.03 answers text/xml; TAP answers the VOTable media type.
    assert scs.status_code == 200 and scs.headers["content-type"].startswith("text/xml")
    for response in (scs, sync_get, sync_post):
        assert response.status_code == 200 and response.headers["content-type"].startswith((VOTABLE, "text/xml"))
        rows = parse_votable(io.BytesIO(response.content)).get_first_table().array
        assert len(rows) > 0
    assert phase == "COMPLETED"
    assert http.get(job).status_code == 200 and http.get(job + "/results").status_code == 200
    result = http.get(job + "/results/result")
    assert result.status_code == 200 and len(parse_votable(io.BytesIO(result.content)).get_first_table().array) > 0
    assert http.get(job + "/error").status_code in {200, 404}
    assert job.rsplit("/", 1)[1] in http.get("/vo/tap/async").text
    assert http.post(job, data={"ACTION": "DELETE"}, follow_redirects=False).status_code == 303
    assert http.get(job).status_code == 404
    other = urlsplit(http.post("/vo/tap/async", data=_tap("SELECT name FROM astrosearch.catalogs"),
                               follow_redirects=False).headers["location"]).path
    assert http.delete(other, follow_redirects=False).status_code == 303


def test_alerts_poll_list_show_and_recrossmatch(http: httpx.Client) -> None:
    from test_alerts import params

    brokers = http.get("/api/v1/alerts/brokers")
    assert brokers.status_code == 200 and {"alerce", "fink"} <= {b["name"] for b in brokers.json()}
    p = params("alerce")
    payload = {"broker": "alerce", "since_mjd": p["since_mjd"], "until_mjd": p["until_mjd"], "limit": p["limit"],
               "class_name": "SN", "classifier": "stamp_classifier", "mjd_field": "firstmjd",
               "crossmatch_mode": "wait"}
    alerts.clear_class_list_cache()
    with upstream(replay_side_effect(load_exchanges("alerts", ["alerce", "xmatch_alerce"])), sesame=None):
        polled = http.post("/api/v1/alerts/poll", json=payload)
        assert polled.status_code == 200, polled.text
        ids = polled.json()["alert_ids"]
        again = http.post(f"/api/v1/alerts/{ids[0]}/crossmatch")
    data = polled.json()
    assert data["fetched"] == 3 and data["inserted"] == 3 and data["crossmatched"] == 3
    listed = http.get("/api/v1/alerts").json()
    assert listed["count"] == 3 and {a["id"] for a in listed["alerts"]} == set(ids)
    shown = http.get(f"/api/v1/alerts/{ids[0]}")
    assert shown.status_code == 200 and shown.json()["crossmatch_status"] == "done"
    assert again.status_code == 200, again.text
    assert http.get("/api/v1/alerts/alerce:ZTF00nothere").status_code == 404
    assert http.post("/api/v1/alerts/poll", json={"broker": "antares"}).status_code == 422


# ---------------------------------------------------------------------------
# Middleware on the live server
# ---------------------------------------------------------------------------


def test_authentication_quota_cors_and_strict_json(http: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "e2e-key")
    for path in ("/api/v1/batch/strategies", "/api/v1/skycache/status", "/vo/tap/tables", "/api/v1/alerts"):
        assert http.get(path).status_code == 401, path
    key = {"X-API-Key": "e2e-key"}
    assert http.get("/api/v1/batch/strategies", headers=key).status_code == 200
    for path in ("/api/v1/health", "/vo/tap/availability", "/", "/app.js"):
        assert http.get(path).status_code == 200, path
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("API_KEYS", "e2e-quota-key")
    quota = {"X-API-Key": "e2e-quota-key"}
    assert http.get("/api/v1/stats", headers=quota).status_code == 200
    assert http.get("/api/v1/stats", headers=quota).status_code == 429
    assert http.get("/api/v1/health").status_code == 200
    monkeypatch.delenv("API_KEYS")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "0")
    preflight = http.options("/api/v1/search", headers={"Origin": "https://sky.example",
                                                         "Access-Control-Request-Method": "POST"})
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in preflight.headers
    nan = http.post("/api/v1/search", content=b'{"ra": NaN, "dec": 1}', headers={"content-type": "application/json"})
    assert nan.status_code == 422 and nan.json()["detail"][0]["type"] == "json_invalid"
    monkeypatch.setenv("MAX_REQUEST_BYTES", "64")
    assert http.post("/api/v1/queries", json={"name": "x" * 100, "query": {"ra": 1, "dec": 1}}).status_code == 413
