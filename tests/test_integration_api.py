"""Integration of the feature modules into api.py and main.py (in-process, offline).

Covers what the integration layer adds on top of the modules' own tests: the mounted routers
and the OpenAPI document, the shared app.state objects and their shutdown, authentication /
quota / body-limit / strict-JSON middleware on every route, CORS, the vizier-merged registry,
the sky cache in front of the archives, Prometheus catalog metrics for every search path, the
resolver's answer reaching the engine, and non-blocking Redis access.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from fastapi.testclient import TestClient
from fixture_io import FIXTURES, TARGETS
from helpers import offline_client

import api
import main
import skycache
from models import CatalogRegistry, CatalogSource, UnifiedRecord
from providers import CacheManager, QueryResult

SESAME_3C273 = (FIXTURES / "sesame" / "3c273.xml").read_text(encoding="utf-8")

# (method, path) of every route each feature module contributes.
MODULE_ROUTES: dict[str, set[tuple[str, str]]] = {
    "core": {("GET", "/api/v1/health"), ("GET", "/api/v1/catalogs"), ("GET", "/api/v1/catalogs/{catalog_name}"),
             ("POST", "/api/v1/search"), ("POST", "/api/v1/search/batch"), ("POST", "/api/v1/datasets/create"),
             ("GET", "/api/v1/datasets"), ("GET", "/api/v1/datasets/{dataset_name}"),
             ("DELETE", "/api/v1/datasets/{dataset_name}"), ("GET", "/api/v1/datasets/{dataset_name}/export"),
             ("GET", "/api/v1/queries"), ("POST", "/api/v1/queries"), ("DELETE", "/api/v1/queries/{query_id}"),
             ("GET", "/api/v1/stats"), ("GET", "/api/v1/monitoring"), ("GET", "/api/v1/monitoring/metrics")},
    "streaming": {("GET", "/api/v1/search/stream")},
    "batch": {("GET", "/api/v1/batch/strategies"), ("POST", "/api/v1/batch/crossmatch")},
    "skycache": {("GET", "/api/v1/skycache/status"), ("POST", "/api/v1/skycache/mirror"),
                 ("GET", "/api/v1/skycache/cone"), ("DELETE", "/api/v1/skycache/{catalog}")},
    "vizier": {("GET", "/api/v1/vizier/search"), ("GET", "/api/v1/vizier/catalog/{identifier:path}"),
               ("POST", "/api/v1/vizier/register"), ("GET", "/api/v1/vizier/registered")},
    "sed": {("GET", "/api/v1/sed"), ("POST", "/api/v1/sed")},
    "timedomain": {("GET", "/api/v1/lightcurves"), ("GET", "/api/v1/solar-system")},
    "imaging": {("GET", "/api/v1/cutouts"), ("GET", "/api/v1/cutouts/surveys"), ("GET", "/api/v1/cutouts/stack")},
    "ai": {("POST", "/api/v1/ai/query"), ("POST", "/api/v1/ai/explain")},
    "provenance": {("POST", "/api/v1/provenance/manifest"), ("POST", "/api/v1/provenance/replay"),
                   ("GET", "/api/v1/citations"), ("GET", "/api/v1/citations/sources")},
    "vo_server": {("GET", "/vo/scs"), ("GET", "/vo/tap"), ("GET", "/vo/tap/availability"),
                  ("GET", "/vo/tap/capabilities"), ("GET", "/vo/tap/tables"), ("GET", "/vo/tap/tables/{table_name}"),
                  ("GET", "/vo/tap/examples"), ("GET", "/vo/tap/sync"), ("POST", "/vo/tap/sync"),
                  ("GET", "/vo/tap/async"), ("POST", "/vo/tap/async"), ("GET", "/vo/tap/async/{job_id}"),
                  ("POST", "/vo/tap/async/{job_id}"), ("DELETE", "/vo/tap/async/{job_id}"),
                  *{(m, f"/vo/tap/async/{{job_id}}/{leaf}") for leaf in ("phase", "parameters", "executionduration",
                                                                          "destruction") for m in ("GET", "POST")},
                  *{("GET", f"/vo/tap/async/{{job_id}}/{leaf}") for leaf in ("results", "results/result", "error",
                                                                            "quote", "owner")}},
    "alerts": {("GET", "/api/v1/alerts"), ("GET", "/api/v1/alerts/brokers"), ("POST", "/api/v1/alerts/poll"),
               ("GET", "/api/v1/alerts/{alert_id}"), ("POST", "/api/v1/alerts/{alert_id}/crossmatch")},
}
UI_FILES = {"/", "/index.html", "/app.js", "/styles.css"}


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every store of the app in tmp_path; no authentication, no quota, no caches."""
    for name, value in {
        "DATASET_STORAGE_PATH": tmp_path / "datasets", "SKYCACHE_PATH": tmp_path / "skycache",
        "CATALOG_REGISTRY_PATH": tmp_path / "catalogs.yaml", "CUTOUT_CACHE_DIR": tmp_path / "cutouts",
        "ASTROSEARCH_CACHE_DIR": tmp_path / "cache",
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("API_SEARCH_CACHE_TTL_SECONDS", "0")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("ASTROSEARCH_SED_OFFLINE_FILTERS", "true")
    for name in ("API_KEYS", "JWT_SECRET", "JWT_PUBLIC_KEY", "REQUIRE_API_KEY", "REDIS_URL", "ANTHROPIC_API_KEY",
                 "ANTHROPIC_AUTH_TOKEN", "CORS_ORIGINS", "MAX_REQUEST_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    return tmp_path


def no_upstream() -> respx.MockRouter:
    """Any request to an archive fails the test (TestClient's own requests are not httpx ones)."""
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route().mock(side_effect=AssertionError("no upstream request expected"))
    return router


# ---------------------------------------------------------------------------
# Routers, OpenAPI and the web UI
# ---------------------------------------------------------------------------


def test_every_module_route_is_mounted_and_documented(isolated: Path) -> None:
    mounted = {(method, path) for path, methods in api.route_table() for method in methods}
    for module, routes in MODULE_ROUTES.items():
        assert routes <= mounted, (module, sorted(routes - mounted))
    assert UI_FILES <= {path for path, _ in api.route_table()}
    with TestClient(api.app) as client:
        spec = client.get("/api/openapi.json").json()
    documented = {(method.upper(), path) for path, item in spec["paths"].items() for method in item}
    # OpenAPI writes path parameters without their Starlette converter ('{identifier:path}').
    expected = {(m, p.replace(":path}", "}")) for m, p in set().union(*MODULE_ROUTES.values())}
    assert expected <= documented, sorted(expected - documented)
    # The UI files are not API operations; every operation id is unique (client generators need that).
    assert not UI_FILES & set(spec["paths"])
    operation_ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()]
    assert len(operation_ids) == len(set(operation_ids))
    assert {"streaming", "batch", "skycache", "vizier", "sed", "time-domain", "imaging", "ai", "provenance",
            "Virtual Observatory", "alerts"} <= {tag for item in spec["paths"].values() for op in item.values()
                                                 for tag in op.get("tags", [])}


def test_web_ui_is_served_at_root_without_authentication(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "integration-key")
    with TestClient(api.app) as client:
        root = client.get("/")
        assert root.status_code == 200 and "<title>AstroSearch Sky Explorer</title>" in root.text
        assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
        assert client.get("/styles.css").headers["content-type"].startswith("text/css")
        assert client.head("/").status_code == 200
        assert client.get("/api/v1/catalogs").status_code == 401


def test_web_ui_directory_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTROSEARCH_WEB_DIR", raising=False)
    assert api.web_ui_dir() == api.imaging.WEB_DIR
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("ASTROSEARCH_WEB_DIR", str(tmp_path))
    assert api.web_ui_dir() == tmp_path
    monkeypatch.setenv("ASTROSEARCH_WEB_DIR", str(tmp_path / "missing"))
    assert api.web_ui_dir() is None


# ---------------------------------------------------------------------------
# app.state: shared objects, the vizier-merged registry, shutdown
# ---------------------------------------------------------------------------


def test_lifespan_shares_one_client_registry_service_and_closes_module_state(isolated: Path) -> None:
    closed: list[str] = []

    class FakeAnthropic:
        async def close(self) -> None:
            closed.append("anthropic")

    with TestClient(api.app) as client:
        state = api.app.state
        assert isinstance(state.service, api.MeteredCrossmatchService)
        # One registry object: vizier registrations reach the service and the dataset engine at once.
        assert state.registry is state.service.registry is state.engine.registry
        assert state.engine.service is state.service and state.metadata is state.engine.metadata
        assert state.providers is state.service.providers
        assert all(isinstance(p, main.MirrorFirstProvider) for p in state.providers.values())
        assert {getattr(p, "client", None) for p in state.providers.values()} == {state.client}
        assert isinstance(state.skycache, skycache.SkyCache) and state.skycache.root == isolated / "skycache"
        assert state.sed_filters.offline is True
        state.anthropic = FakeAnthropic()
        alerts_client = state.alerts_client = httpx.AsyncClient()
        shared_client = state.client
        monitoring = client.get("/api/v1/monitoring").json()
        assert monitoring["skycache"] == {"enabled": True, "catalogs": []}
        assert monitoring["catalogs"] == len(state.registry.catalogs)
        assert not alerts_client.is_closed and not shared_client.is_closed
    assert closed == ["anthropic"]
    # The shutdown closes the alerts router's own client and the shared one (not only the Anthropic client).
    assert alerts_client.is_closed and shared_client.is_closed
    for name in api.STATE_OBJECTS + api.LAZY_STATE_OBJECTS:
        assert not hasattr(api.app.state, name), name


def test_vizier_registered_catalogs_are_loaded_at_startup(isolated: Path) -> None:
    from vizier import DEFAULT_CATALOGS

    entry = dict(DEFAULT_CATALOGS["vlass"], description="A user catalog registered with 'vizier add'")
    (isolated / "catalogs.yaml").write_text(yaml.safe_dump({"extends": "embedded", "catalogs": {"my_radio": entry}}),
                                            encoding="utf-8")
    with TestClient(api.app) as client:
        names = set(client.get("/api/v1/catalogs").json())
        assert names == set(DEFAULT_CATALOGS) | {"my_radio"}
        assert client.get("/api/v1/catalogs/my_radio").json()["description"].startswith("A user catalog")
        listed = client.get("/api/v1/vizier/registered").json()
        assert [c["name"] for c in listed["catalogs"]] == ["my_radio"] and listed["catalogs"][0]["active"] is True
    assert set(main.build_registry().catalogs) == set(DEFAULT_CATALOGS) | {"my_radio"}
    assert set(main.catalog_definitions()) == set(DEFAULT_CATALOGS) | {"my_radio"}


# ---------------------------------------------------------------------------
# Middleware: authentication, quotas, public paths, body limits, strict JSON, CORS
# ---------------------------------------------------------------------------


def test_authentication_applies_to_every_feature_router(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "integration-key")
    protected = ["/api/v1/catalogs", "/api/v1/batch/strategies", "/api/v1/skycache/status", "/api/v1/vizier/registered",
                 "/api/v1/cutouts/surveys", "/api/v1/citations/sources", "/vo/tap/tables", "/vo/tap/capabilities",
                 "/api/v1/alerts/brokers", "/api/v1/monitoring/metrics", "/api/openapi.json"]
    with no_upstream(), TestClient(api.app) as client:
        for path in protected:
            assert client.get(path).status_code == 401, path
            assert client.get(path, headers={"X-API-Key": "integration-key"}).status_code == 200, path
        assert client.post("/api/v1/search/stream").status_code == 401  # auth before routing
        for path in ("/api/v1/health", "/vo/tap/availability", "/"):
            assert client.get(path).status_code == 200, path


def test_quota_counts_api_calls_but_not_health_availability_or_ui(isolated: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "quota-key-integration")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "2")
    headers = {"X-API-Key": "quota-key-integration"}
    monkeypatch.setattr(api.app.state, "quota", api.RequestQuota(None))
    with no_upstream(), TestClient(api.app) as client:
        for _ in range(5):
            for path in ("/api/v1/health", "/vo/tap/availability", "/", "/app.js"):
                assert client.get(path, headers=headers).status_code == 200, path
        assert client.get("/api/v1/batch/strategies", headers=headers).status_code == 200
        assert client.get("/vo/tap/tables", headers=headers).status_code == 200
        limited = client.get("/api/v1/sed", headers=headers, params={"ra": 1, "dec": 1})
        assert limited.status_code == 429 and limited.headers["retry-after"] == "60"
        assert client.get("/api/v1/health").status_code == 200


def test_body_limit_and_the_routes_that_own_theirs(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_REQUEST_BYTES", "200")
    padding = {"name": "x" * 400, "query": {"ra": 1.0, "dec": 2.0}}
    with no_upstream(), TestClient(api.app) as client:
        assert client.post("/api/v1/queries", json=padding).status_code == 413
        too_long = client.post("/api/v1/queries", content=b"{}", headers={"content-length": "999999",
                                                                           "content-type": "application/json"})
        assert too_long.status_code == 413
        # The batch route streams its body against BATCH_MAX_UPLOAD_BYTES: a 300-byte target list is fine.
        targets = {"targets": [{"id": f"t{i}", "ra": 10.0 + i, "dec": 1.0} for i in range(6)],
                   "catalogs": ["no_such_catalog"]}
        response = client.post("/api/v1/batch/crossmatch", json=targets)
        assert response.status_code == 422 and "no_such_catalog" in response.text  # its own validation, not a 413
        # The VO services answer oversized bodies with their own DALI error document (not the API's JSON).
        query = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": "SELECT name FROM astrosearch.catalogs " + " " * 300}
        vo = client.post("/vo/tap/sync", data=query)
        assert vo.status_code == 200 and vo.headers["content-type"].startswith("application/x-votable+xml")


NON_FINITE_BODIES = [
    ("/api/v1/search", {"ra": float("nan"), "dec": 1.0}),
    ("/api/v1/search", {"ra": 1.0, "dec": 1.0, "radius_arcsec": float("inf")}),
    ("/api/v1/search/batch", [{"ra": 1.0, "dec": float("-inf")}]),
    ("/api/v1/batch/crossmatch", {"targets": [{"id": "a", "ra": 1, "dec": 1}], "catalogs": ["simbad"],
                                  "radius_arcsec": float("nan")}),
    ("/api/v1/queries", {"name": "q", "query": {"ra": float("nan"), "dec": 1.0}}),
    ("/api/v1/datasets/create", {"name": "d", "profile": "full", "targets": [{"ra": float("nan"), "dec": 0.0}]}),
    ("/api/v1/sed", {"ra": float("nan"), "dec": 1.0}),
    ("/api/v1/provenance/manifest", {"ra": 1.0, "dec": float("nan")}),
    ("/api/v1/alerts/poll", {"broker": "alerce", "since_mjd": float("nan")}),
    ("/api/v1/skycache/mirror", {"catalog": "gaia_dr3", "ra": float("nan"), "dec": 0.0, "radius_deg": 0.1}),
    ("/api/v1/vizier/register", {"table_id": "IX/58/2sxps", "overrides": {"systematic_arcsec": float("nan")}}),
    ("/api/v1/ai/explain", {"ra": float("nan"), "dec": 0.0, "facts_only": True}),
]


@pytest.mark.parametrize(("path", "body"), NON_FINITE_BODIES, ids=[f"{p}-{i}" for i, (p, _) in enumerate(NON_FINITE_BODIES)])
def test_non_finite_json_numbers_are_422_on_every_route(isolated: Path, path: str, body: Any) -> None:
    import json

    raw = json.dumps(body)  # Python writes NaN / Infinity / -Infinity, which RFC 8259 forbids
    with no_upstream(), TestClient(api.app) as client:
        response = client.post(path, content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail[0]["type"] == "json_invalid" and "must be finite" in detail[0]["msg"]


def test_overflowing_numbers_and_nan_query_parameters_are_422(isolated: Path) -> None:
    with no_upstream(), TestClient(api.app) as client:
        overflow = client.post("/api/v1/search", content=b'{"ra": 1e400, "dec": 1}',
                               headers={"content-type": "application/json"})
        assert overflow.status_code == 422 and "1e400" in overflow.text
        # A query parameter echoed back as NaN must not turn the 422 into a 500.
        for path, params in (("/api/v1/lightcurves", {"ra": "nan", "dec": "1"}),
                             ("/api/v1/skycache/cone", {"catalog": "gaia_dr3", "ra": "nan", "dec": "0",
                                                        "radius_arcsec": "5"}),
                             ("/api/v1/cutouts", {"ra": "inf", "dec": "0"}),
                             ("/api/v1/solar-system", {"ra": "1", "dec": "nan"})):
            response = client.get(path, params=params)
            assert response.status_code == 422, (path, response.text)
            response.json()


def test_non_finite_json_detection() -> None:
    assert api.non_finite_json(b'{"a": [1, 2.5, {"b": NaN}]}') == "NaN"
    assert api.non_finite_json(b'{"a": -Infinity}') == "-Infinity"
    assert api.non_finite_json(b'{"a": 1e999}') == "1e999"
    assert api.non_finite_json(b'{"a": 1e308, "s": "NaN"}') is None  # a string is not a number
    assert api.non_finite_json(b"not json") is None
    assert api.owns_body_limit("/api/v1/batch/crossmatch") and api.owns_body_limit("/vo/tap/sync")
    assert not api.owns_body_limit("/api/v1/search") and not api.owns_body_limit("/voyager")


def test_cors_never_allows_credentials_with_a_wildcard_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("", "*", "https://a.example,*"):
        monkeypatch.setenv("CORS_ORIGINS", value)
        options = api.cors_options()
        assert options["allow_origins"] == ["*"] and options["allow_credentials"] is False
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    options = api.cors_options()
    assert options["allow_origins"] == ["https://a.example", "https://b.example"] and options["allow_credentials"]
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "false")
    assert api.cors_options()["allow_credentials"] is False
    assert {"X-Request-ID", "X-Batch-Failed-Targets", "X-Cutout-Survey"} <= set(options["expose_headers"])


def test_cors_preflight_on_the_app(isolated: Path) -> None:
    with TestClient(api.app) as client:
        response = client.options("/api/v1/search", headers={"Origin": "https://sky.example",
                                                              "Access-Control-Request-Method": "POST"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


# ---------------------------------------------------------------------------
# Sky cache in front of the archives
# ---------------------------------------------------------------------------


class _Remote:
    def __init__(self) -> None:
        self.client = object()
        self.guards = {"https://archive.example/tap": "guard"}
        self.calls = 0

    async def query(self, catalog, target, radius_arcsec):
        self.calls += 1
        return QueryResult([], {"endpoint": "remote"})


class _Store:
    def __init__(self, mirrored: set[str]) -> None:
        self.mirrored = mirrored

    def has(self, catalog: str) -> bool:
        return catalog in self.mirrored

    def evict(self, catalog: str) -> None:
        pass


class _Local:
    def __init__(self, store: _Store) -> None:
        self.store = store
        self.calls = 0

    async def query(self, catalog, target, radius_arcsec):
        self.calls += 1
        raise skycache.CoverageError("not covered", catalog=catalog.name, covered_fraction=0.25)


def test_mirror_first_provider_uses_the_store_only_for_mirrored_catalogs() -> None:
    registry = CatalogRegistry()
    remote, local = _Remote(), _Local(_Store({"gaia_dr3"}))
    provider = main.MirrorFirstProvider(remote, local)  # type: ignore[arg-type]
    target = main.validate_target(187.2779154, 2.0523883)
    plain = asyncio.run(provider.query(registry.get("simbad"), target, 5.0))
    assert (local.calls, remote.calls) == (0, 1) and "skycache" not in plain.meta
    partly = asyncio.run(provider.query(registry.get("gaia_dr3"), target, 5.0))
    assert (local.calls, remote.calls) == (1, 2)
    assert partly.meta["skycache"] == {"hit": False, "reason": "not_covered", "covered_fraction": 0.25}
    # The archive adapter's attributes read through (batch guards, provenance client, monitoring).
    assert provider.client is remote.client and provider.guards == remote.guards
    assert skycache.archive_providers({"tap": provider}) == {"tap": remote}


def test_build_providers_honours_skycache_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYCACHE_PATH", str(tmp_path))
    wrapped = main.build_providers()
    assert all(isinstance(p, main.MirrorFirstProvider) for p in wrapped.values())
    assert {p.local.store.root for p in wrapped.values()} == {tmp_path.resolve()}
    assert main.skycache_store() is main.skycache_store(tmp_path)  # one store per directory
    monkeypatch.setenv("SKYCACHE_ENABLED", "false")
    assert not any(isinstance(p, skycache.SkyCacheProvider) for p in main.build_providers().values())


def test_a_mirrored_region_is_searched_without_the_network(tmp_path: Path) -> None:
    from test_skycache_replay import mirror_from_fixtures

    store = skycache.SkyCache(tmp_path / "store")
    report = asyncio.run(mirror_from_fixtures("gaia_dr3", store))
    assert report.tiles_failed == 0 and report.rows_stored > 400

    async def search() -> UnifiedRecord:
        async with offline_client() as client:
            service = main.build_service(registry=CatalogRegistry(), providers=main.build_providers(client=client,
                                                                                                   store=store))
            return await service.crossmatch(*TARGETS["3c273"], radius_arcsec=10.0, catalogs=["gaia_dr3"])

    with no_upstream():  # every Gaia row comes from the local HATS store
        record = asyncio.run(search())
    assert record.failures == []
    gaia = record.as_dict()["catalog_results"]["gaia_dr3"]
    assert gaia["sources"] and all(s["provenance"]["skycache"]["mirrored_from"] for s in gaia["sources"])
    target = next(g for g in record.crossmatch_groups if g["contains_target"])
    assert any(m["catalog"] == "gaia_dr3" for m in target["members"])


# ---------------------------------------------------------------------------
# Catalog metrics and the resolver's answer
# ---------------------------------------------------------------------------


def _catalog_counter(catalog: str, status: str) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("astrosearch_catalog_queries_total", {"catalog": catalog, "status": status}) or 0.0


class _OneRowProvider:
    async def query(self, catalog, target, radius_arcsec):
        source = CatalogSource(catalog.name, "row-1", target.ra, target.dec, 0.1, {}, {"wavelength": catalog.wavelength},
                               {})
        return QueryResult([source], {"max_rows": 50, "row_limit": 50})


def test_every_search_path_on_the_api_service_counts_catalog_outcomes() -> None:
    registry = CatalogRegistry()
    registry._catalogs = {"simbad": registry.get("simbad")}
    service = api.MeteredCrossmatchService(registry, {"tap": _OneRowProvider()})
    before = _catalog_counter("simbad", "success")

    async def stream() -> None:
        async for _event in service.crossmatch_stream(187.25, 2.05, radius_arcsec=5.0):
            pass

    asyncio.run(service.crossmatch(187.25, 2.05, radius_arcsec=5.0))
    asyncio.run(stream())
    assert _catalog_counter("simbad", "success") - before == 2.0
    # api._search does not count a metered service's records a second time.
    old = api.app.state.__dict__.get("_state", {}).get("service")
    api.app.state.service = service
    try:
        asyncio.run(api._search(api.SearchRequest(ra=187.25, dec=2.05, radius_arcsec=5.0)))
    finally:
        if old is None:
            del api.app.state.service
        else:
            api.app.state.service = old
    assert _catalog_counter("simbad", "success") - before == 3.0


class _CapturingService:
    registry = CatalogRegistry()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def crossmatch(self, ra, dec, **kwargs):
        self.calls.append({"ra": ra, "dec": dec, **kwargs})
        return UnifiedRecord({"ra": ra, "dec": dec}, 0, {}, {}, [], {"warnings": [], "association": {"notes": []}})


def _sesame_router() -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
        200, text=SESAME_3C273, headers={"content-type": "text/xml"})
    return router


def test_api_name_search_passes_the_resolvers_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CapturingService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    with _sesame_router():
        data = asyncio.run(api._search(api.SearchRequest(name="3C 273", radius_arcsec=5.0)))
    call = service.calls[0]
    query = call["query"]
    assert call["resolved_object"].canonical_name == "3C 273"
    assert query.metadata["resolved_object"]["canonical_name"] == "3C 273"
    assert call["target_uncertainty_source"] == "resolver" and call["target_uncertainty_arcsec"] > 0
    assert data["resolved_object"]["canonical_name"] == "3C 273"
    assert (query.target.ra, query.target.dec) == (call["ra"], call["dec"])


def test_api_name_search_moves_a_quasar_to_any_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CapturingService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    # Sesame's 3C 273 answer has a SIMBAD J2000 position; the quasar does not move, so any epoch is fine.
    with _sesame_router():
        asyncio.run(api._search(api.SearchRequest(name="3C 273", radius_arcsec=5.0, epoch=2016.0)))
    call = service.calls[-1]
    assert call["query"].target.epoch == 2016.0
    assert (call["ra"], call["dec"]) == pytest.approx((187.27791594, 2.05238823), abs=1e-9)  # not moved


# Sesame's answer for a Galactic star with a SIMBAD J2000 position but no proper motion (the 3C 273
# answer with its type, name, motion and redshift replaced).
SESAME_STAR_WITHOUT_MOTION = re.sub(r"<pm>.*?</pm>|<z>.*?</z>|<plx>.*?</plx>", "", SESAME_3C273, flags=re.DOTALL)     .replace("<otype>BLL</otype>", "<otype>*</otype>").replace("3C 273", "HD 0")


def test_api_name_search_rejects_an_epoch_it_cannot_move_the_position_to(monkeypatch: pytest.MonkeyPatch) -> None:
    """A star whose resolver answer has no motion cannot be moved from J2000 to another epoch: 422, and no
    catalog is queried (the search would otherwise look at the wrong position)."""
    service = _CapturingService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
        200, text=SESAME_STAR_WITHOUT_MOTION, headers={"content-type": "text/xml"})
    with router, TestClient(api.app) as client:
        rejected = client.post("/api/v1/search", json={"name": "HD 0", "radius_arcsec": 5.0, "epoch": 2016.0})
        accepted = client.post("/api/v1/search", json={"name": "HD 0", "radius_arcsec": 5.0})
    assert rejected.status_code == 422, rejected.text
    assert "no known proper motion" in rejected.json()["detail"] and "2016" in rejected.json()["detail"]
    assert accepted.status_code == 200, accepted.text
    assert len(service.calls) == 1 and service.calls[0]["query"].target.epoch == 2000.0


def test_cli_name_search_passes_the_resolvers_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CapturingService()
    monkeypatch.setattr(main, "build_service", lambda **kwargs: service)
    with _sesame_router():
        record = asyncio.run(main.search_object("3C 273", radius_arcsec=5.0, catalogs=["simbad"]))
    call = service.calls[0]
    assert call["resolved_object"].canonical_name == "3C 273" and call["catalogs"] == ["simbad"]
    assert call["target_uncertainty_source"] == "resolver" and call["epoch"] == 2000.0
    assert record.resolved_object["canonical_name"] == "3C 273" and record.provenance["resolver"]


def test_resolution_warnings_are_merged_into_the_record() -> None:
    record = UnifiedRecord({}, 0, {}, {}, [], {"warnings": ["a"], "association": {"notes": ["n"]}})
    main.merge_resolution(record, {"warnings": ["a", "resolved by VizieR"], "notes": ["moved to J2016"]})
    assert record.provenance["warnings"] == ["a", "resolved by VizieR"]
    assert record.provenance["association"]["notes"] == ["n", "moved to J2016"]


# ---------------------------------------------------------------------------
# Redis never blocks the event loop
# ---------------------------------------------------------------------------


class _SlowRedis:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.data: dict[str, bytes] = {}
        self.threads: set[str] = set()

    def get(self, key: str) -> bytes | None:
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        self.data[key] = value.encode()


def test_cache_redis_calls_run_off_the_event_loop() -> None:
    cache = CacheManager(None)
    slow = _SlowRedis(0.3)
    cache._redis = slow

    async def scenario() -> tuple[float, Any]:
        ticks: list[float] = []
        stop = asyncio.Event()

        async def ticker() -> None:
            while not stop.is_set():
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        started = time.perf_counter()
        cache.set("k", {"v": 1}, ttl=60)  # returns at once; Redis written in a worker thread
        set_time = time.perf_counter() - started
        value = await cache.aget("missing")
        await asyncio.sleep(0.35)
        stop.set()
        await task
        gaps = [b - a for a, b in itertools.pairwise(ticks)]
        return set_time, (value, max(gaps))

    set_time, (value, worst_gap) = asyncio.run(scenario())
    assert set_time < 0.1 and value is None and worst_gap < 0.2
    assert threading.current_thread().name not in slow.threads
    assert slow.data["k"] == b'{"v": 1}'
    assert cache.get("k") == {"v": 1}  # synchronous callers still work


def test_request_quota_in_memory_window() -> None:
    quota = api.RequestQuota(None)

    async def scenario() -> list[bool]:
        return [await quota.allow("id", 2) for _ in range(3)] + [await quota.allow("other", 2), await quota.allow("id", 0)]

    assert asyncio.run(scenario()) == [True, True, False, True, True]


def test_dataset_jobs_are_enqueued_off_the_event_loop(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """redis-py / RQ block: POST /api/v1/datasets/create enqueues from a worker thread."""
    seen: list[bool] = []

    def fake_submit(dataset_id: str, payload: dict[str, Any]) -> bool:
        try:
            asyncio.get_running_loop()
            seen.append(True)  # called on the event loop
        except RuntimeError:
            seen.append(False)
        return True  # "queued in Redis": no background task runs

    monkeypatch.setattr(api, "submit_to_redis", fake_submit)
    with no_upstream(), TestClient(api.app) as client:
        response = client.post("/api/v1/datasets/create", json={
            "name": "queued", "profile": "full", "catalogs": ["simbad"], "count_threshold": 1,
            "targets": [{"ra": 187.2779154, "dec": 2.0523883}]})
        assert response.status_code == 202, response.text
        assert client.get(response.headers["location"]).json()["status"] == "queued"
    assert seen == [False]


class _LoopRecordingProvider:
    """Records the event loop each archive query runs on (real HTTP connections belong to one loop)."""

    def __init__(self) -> None:
        self.loops: list[Any] = []

    async def query(self, catalog, target, radius_arcsec):
        self.loops.append(asyncio.get_running_loop())
        source = CatalogSource(catalog.name, "row-1", target.ra, target.dec, 0.1, {},
                               {"wavelength": catalog.wavelength}, {})
        return QueryResult([source], {"max_rows": 50, "row_limit": 50})


def test_vo_queries_reach_the_archives_on_the_application_event_loop() -> None:
    """vo_server runs the crossmatch on a private loop in a worker thread; every archive request of
    the engine (catalogue queries and density probes, not only executor.execute) must go back to the
    application's loop, where the shared HTTP client's connections live. Before the fix they ran on the
    worker's loop and failed live with 'Event loop is closed'."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    import vo_server
    from crossmatch import CrossmatchService

    provider = _LoopRecordingProvider()
    app_loops: list[Any] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        app_loops.append(asyncio.get_running_loop())
        registry = CatalogRegistry()
        registry._catalogs = {"simbad": registry.get("simbad")}
        application.state.service = CrossmatchService(registry, {"tap": provider})
        application.state.registry = registry
        yield

    application = FastAPI(lifespan=lifespan)
    application.include_router(vo_server.router)
    with TestClient(application) as client:
        response = client.get("/vo/scs", params={"RA": 187.2779154, "DEC": 2.0523883, "SR": 0.002})
        tap = client.get("/vo/tap/sync", params={
            "REQUEST": "doQuery", "LANG": "ADQL",
            "QUERY": "SELECT source_id FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                     "CIRCLE('ICRS', 187.2779154, 2.0523883, 0.002))"})
    assert response.status_code == 200 and b"row-1" in response.content, response.text[:500]
    assert tap.status_code == 200 and b"row-1" in tap.content, tap.text[:500]
    assert provider.loops and all(loop is app_loops[0] for loop in provider.loops)


def test_citation_check_accepts_an_article_number_encoded_in_the_doi() -> None:
    """doi.org (Crossref) registers SPIE JATIS 1, 014003 (TESS, Ricker et al. 2015) with page '1-10' --
    the pages inside article 014003 (live 2026-09-29). The live citation check reported a wrong
    first page; the article number is the DOI's own suffix. A genuinely different page still fails."""
    import provenance as P

    tess = next(r for r in P.REFERENCES.values() if r.bibcode == "2015JATIS...1a4003R")
    csl = {"title": "Transiting Exoplanet Survey Satellite (TESS)", "issued": {"date-parts": [[2014, 10, 24]]},
           "volume": "1", "page": "1-10", "DOI": "10.1117/1.JATIS.1.1.014003"}
    ok, problems, _meta = P._compare_csl(tess, csl)
    assert ok and problems == []
    wrong = {**csl, "DOI": "10.1117/1.JATIS.1.1.014004"}
    ok, problems, _meta = P._compare_csl(tess, wrong)
    assert not ok and problems == ["first page 1 != 14003"]


# ---------------------------------------------------------------------------
# Final-review regressions: status codes, radius limit, CORS with auth, body limit, exports, CLI
# ---------------------------------------------------------------------------

SESAME_UNKNOWN = (FIXTURES / "imaging" / "sesame_unknown.xml").read_text(encoding="utf-8")


def _sesame(**respond: Any) -> respx.MockRouter:
    """Sesame answers as given; any other upstream request fails the test."""
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(**respond)
    router.route().mock(side_effect=AssertionError("no archive request expected"))
    return router


@pytest.mark.parametrize("body", [{"ra": 10, "dec": 95}, {"ra": 10, "dec": -91}, {"ra": 400, "dec": 10},
                                  {"ra": 10, "dec": 10, "pm_ra_masyr": 5.0},
                                  {"ra": 10, "dec": 10, "pm_ra_masyr": 1e9, "pm_dec_masyr": 0.0}])
def test_invalid_coordinates_or_motion_are_422_never_502(isolated: Path, body: dict[str, Any]) -> None:
    """Finding: {"ra":10,"dec":95} answered 502 'Catalog search failed' (InvalidCoordinateError is
    not a ValueError); RA 400 was silently wrapped. Every search form now answers 422."""
    with no_upstream(), TestClient(api.app) as client:
        for path, payload in (("/api/v1/search", body), ("/api/v1/search/batch", [body]),
                              ("/api/v1/queries", {"name": "q", "query": body}),
                              ("/api/v1/provenance/manifest", body)):
            response = client.post(path, json=payload)
            assert response.status_code == 422, (path, response.text)


def test_search_error_maps_input_errors_to_422_for_search_and_each_batch_item(isolated: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from models import InvalidCoordinateError, ObjectResolutionError, ResolverUnavailableError

    assert api.search_error(InvalidCoordinateError("DEC must be within [-90, 90] degrees."))[0] == 422
    assert api.search_error(ValueError("bad profile"))[0] == 422
    assert api.search_error(ResolverUnavailableError("Sesame request failed: HTTP 503"))[0] == 503
    assert api.search_error(ResolverUnavailableError("x"))[2] == {"Retry-After": "30"}
    assert api.search_error(ObjectResolutionError("No coordinates found for 'x'"))[0] == 404
    assert api.search_error(RuntimeError("bug"))[0] == 500

    # An input error raised while the search runs (after the model validated) is still a 422.
    async def invalid(_req: Any) -> dict[str, Any]:
        raise InvalidCoordinateError("DEC must be within [-90, 90] degrees.")

    monkeypatch.setattr(api, "_search", invalid)
    with no_upstream(), TestClient(api.app) as client:
        single = client.post("/api/v1/search", json={"ra": 10, "dec": 10})
        items = client.post("/api/v1/search/batch", json=[{"ra": 10, "dec": 10}]).json()
    assert single.status_code == 422 and "DEC must be within" in single.json()["detail"]
    assert items[0]["status_code"] == 422 and "DEC must be within" in items[0]["error"]


def test_search_radius_is_limited_by_api_max_radius_arcsec(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding: radius_arcsec 100000 sent a 28-degree cone to every archive. The model caps it at
    3600" and API_MAX_RADIUS_ARCSEC (default 1800") applies to every search form."""
    service = _CapturingService()
    monkeypatch.setattr(api, "get_service", lambda: service)
    target = {"ra": 200.0, "dec": -30.0}
    with no_upstream(), TestClient(api.app) as client:
        for radius in (100000, 3000):
            body = {**target, "radius_arcsec": radius}
            for path, payload in (("/api/v1/search", body), ("/api/v1/queries", {"name": "wide", "query": body})):
                response = client.post(path, json=payload)
                assert response.status_code == 422, (path, radius, response.text)
            items = client.post("/api/v1/search/batch", json=[body])
            assert items.status_code == 422 or items.json()[0]["status_code"] == 422, items.text
            stream = client.get("/api/v1/search/stream", params={**target, "radius_arcsec": radius})
            assert stream.status_code == 422, stream.text
            dataset = client.post("/api/v1/datasets/create", json={"name": "d", "profile": "optical",
                                                                    "radius_arcsec": radius, "targets": [target]})
            assert dataset.status_code == 422, dataset.text
        assert "API_MAX_RADIUS_ARCSEC" in client.post("/api/v1/search", json={**target, "radius_arcsec": 3000}).text
        assert client.post("/api/v1/search", json={**target, "radius_arcsec": 1800}).status_code == 200
    assert [call["query"].radius_arcsec for call in service.calls] == [1800]
    # The limit is configurable (below the absolute 3600" of the model).
    monkeypatch.setenv("API_MAX_RADIUS_ARCSEC", "10")
    with pytest.raises(ValueError, match="API_MAX_RADIUS_ARCSEC"):
        main.check_search_radius(11.0)
    main.check_search_radius(10.0)
    with pytest.raises(ValueError, match="API_MAX_RADIUS_ARCSEC"):
        asyncio.run(main.crossmatch(10.0, 10.0, radius_arcsec=11.0))


def test_name_resolver_outage_is_503_and_unknown_name_404_on_search_batch_and_stream(isolated: Path) -> None:
    """Finding: a Sesame outage was reported by /search as 404 'could not be resolved'."""
    with _sesame(status_code=503), TestClient(api.app) as client:
        search = client.post("/api/v1/search", json={"name": "3C 273"})
        items = client.post("/api/v1/search/batch", json=[{"name": "3C 273"}]).json()
        stream = client.get("/api/v1/search/stream", params={"name": "3C 273"})
    assert search.status_code == 503 and search.headers["retry-after"] == "30", search.text
    assert "resolver unavailable" in search.json()["detail"]
    assert items[0]["status_code"] == 503
    assert stream.status_code == 503 and stream.headers["retry-after"] == "30"
    unknown = {"status_code": 200, "text": SESAME_UNKNOWN, "headers": {"content-type": "text/xml"}}
    with _sesame(**unknown), TestClient(api.app) as client:
        assert client.post("/api/v1/search", json={"name": "NoSuchObjectQzx42"}).status_code == 404
        assert client.post("/api/v1/search/batch", json=[{"name": "NoSuchObjectQzx42"}]).json()[0]["status_code"] == 404
        assert client.get("/api/v1/search/stream", params={"name": "NoSuchObjectQzx42"}).status_code == 404


def test_cors_preflight_and_errors_carry_cors_headers_when_authentication_is_on(
        isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding: with API_KEYS set, the preflight (which never carries X-API-Key) got a 401 without
    Access-Control-* headers. CORS is the outermost middleware."""
    monkeypatch.setenv("API_KEYS", "cors-key")
    monkeypatch.setenv("JWT_SECRET", "cors-secret")
    origin = {"Origin": "https://sky.example"}
    with no_upstream(), TestClient(api.app) as client:
        preflight = client.options("/api/v1/search", headers={
            **origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"})
        assert preflight.status_code == 200, preflight.text
        assert preflight.headers["access-control-allow-origin"] in {"*", "https://sky.example"}
        assert "x-api-key" in preflight.headers["access-control-allow-headers"].lower()
        denied = client.get("/api/v1/catalogs", headers=origin)
        assert denied.status_code == 401 and denied.headers.get("access-control-allow-origin")
        allowed = client.get("/api/v1/catalogs", headers={**origin, "X-API-Key": "cors-key"})
        assert allowed.status_code == 200 and allowed.headers.get("access-control-allow-origin")
    # CORSMiddleware wraps the authentication middleware (the last added is the outermost).
    classes = [m.cls.__name__ for m in api.app.user_middleware]
    assert classes.index("CORSMiddleware") < classes.index("BaseHTTPMiddleware")


def test_chunked_body_over_the_limit_is_rejected_without_reading_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding: a body without Content-Length was buffered completely before the 413."""
    from starlette.requests import Request

    monkeypatch.setenv("MAX_REQUEST_BYTES", "1000")
    received: list[int] = []

    async def receive() -> dict[str, Any]:
        received.append(1)
        return {"type": "http.request", "body": b"x" * 400, "more_body": len(received) < 1000}

    scope = {"type": "http", "method": "POST", "path": "/api/v1/queries", "headers": [
        (b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")], "query_string": b""}

    async def never(_request: Any) -> Any:
        raise AssertionError("the route must not run")

    response = asyncio.run(api._limited(Request(scope, receive), never))
    assert response.status_code == 413
    assert len(received) == 3  # 400 + 400 + 400 > 1000: stopped at the third chunk, not the 1000th

    async def receive_small() -> dict[str, Any]:
        return {"type": "http.request", "body": b'{"a": 1}', "more_body": False}

    async def route(request: Any) -> Any:
        assert await request.body() == b'{"a": 1}'  # the route is handed the bytes that were read
        return api.JSONResponse({"ok": True})

    assert asyncio.run(api._limited(Request(scope, receive_small), route)).status_code == 200


def test_ui_files_answer_conditional_requests_with_304(isolated: Path) -> None:
    """Finding: ETag/Last-Modified with Cache-Control: no-cache, but every revalidation was a 200."""
    with no_upstream(), TestClient(api.app) as client:
        for path in ("/", "/app.js", "/styles.css"):
            first = client.get(path)
            assert first.status_code == 200 and "no-cache" in first.headers["cache-control"]
            etag, modified = first.headers["etag"], first.headers["last-modified"]
            for headers in ({"If-None-Match": etag}, {"If-None-Match": f"W/{etag}"}, {"If-Modified-Since": modified}):
                again = client.get(path, headers=headers)
                assert again.status_code == 304 and again.content == b"", (path, headers)
                assert again.headers["etag"] == etag
            assert client.get(path, headers={"If-None-Match": '"stale"'}).status_code == 200
            # If-None-Match wins over If-Modified-Since (RFC 9110 13.2.2).
            stale = {"If-None-Match": '"stale"', "If-Modified-Since": modified}
            assert client.get(path, headers=stale).status_code == 200


@pytest.mark.parametrize(("fmt", "content", "media"), [
    ("csv", b"a,b\r\n1,2\r\n", "text/csv; charset=utf-8"), ("json", b"[]", "application/json"),
    ("parquet", b"PAR1", "application/vnd.apache.parquet"), ("fits", b"SIMPLE  =", "application/fits")])
def test_dataset_export_media_type_follows_the_format(isolated: Path, fmt: str, content: bytes, media: str) -> None:
    """Finding: a CSV export was served as application/vnd.ms-excel (guessed from the Windows registry)."""
    with no_upstream(), TestClient(api.app) as client:
        engine = api.app.state.engine
        path = engine.storage / f"export-{fmt}.{fmt}"
        path.write_bytes(content)
        engine.metadata.put_dataset({"id": f"ds{fmt}", "name": "x", "status": "completed", "output_format": fmt,
                                     "export_path": str(path), "created_at": "2026-09-29T00:00:00+00:00"})
        response = client.get(f"/api/v1/datasets/ds{fmt}/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media
    assert response.content == content


def test_dataset_catalogs_outside_the_profile_are_rejected_not_dropped(isolated: Path) -> None:
    """Finding: profile 'optical' with catalogs [gaia_dr3, simbad, twomass_psc] silently dropped twomass_psc."""
    body = {"name": "d", "profile": "optical", "catalogs": ["gaia_dr3", "simbad", "twomass_psc"],
            "targets": [{"ra": 10.0, "dec": 10.0}]}
    with no_upstream(), TestClient(api.app) as client:
        response = client.post("/api/v1/datasets/create", json=body)
        assert response.status_code == 422, response.text
        assert "twomass_psc" in response.json()["detail"] and "not in profile" in response.json()["detail"]
        assert client.get("/api/v1/datasets").json() == []


def test_cli_dataset_output_may_be_any_path_but_the_api_keeps_the_storage_restriction(isolated: Path) -> None:
    from datasets import DatasetEngine

    engine = DatasetEngine(registry=main.build_registry())
    outside = isolated / "elsewhere" / "cli-ds.csv"
    outside.parent.mkdir()
    assert engine.check_export_path(outside, "csv", any_path=True) == outside.resolve()
    with pytest.raises(ValueError, match="DATASET_STORAGE_PATH"):
        engine.check_export_path(outside, "csv")
    with pytest.raises(ValueError, match=r"\.parquet"):
        engine.check_export_path(outside, "parquet", any_path=True)
    # The REST API keeps the restriction (a client must not write anywhere on the server).
    body = {"name": "d", "profile": "optical", "targets": [{"ra": 10.0, "dec": 10.0}], "export_format": "csv",
            "output_path": str(outside)}
    with no_upstream(), TestClient(api.app) as client:
        response = client.post("/api/v1/datasets/create", json=body)
    assert response.status_code == 422 and "DATASET_STORAGE_PATH" in response.json()["detail"]


def test_cli_dataset_writes_outside_the_storage_and_checks_the_path_before_building(
        isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: `dataset --output elsewhere.csv` printed 'Building dataset ...' and then failed with
    'output_path must be within DATASET_STORAGE_PATH' (exit 2)."""
    import argparse
    import json

    from datasets import DatasetEngine

    calls: list[dict[str, Any]] = []

    async def create(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"export_path": kwargs["export_path"], "total_sources": 0, "method": "per-target"}

    monkeypatch.setattr(DatasetEngine, "create_dataset", create)
    targets = isolated / "targets.json"
    targets.write_text(json.dumps([{"ra": 10.0, "dec": 10.0}]), encoding="utf-8")
    (isolated / "out").mkdir()
    output = isolated / "out" / "cli-ds.csv"
    args = argparse.Namespace(targets=str(targets), catalogs=None, name="cli", profile="optical", radius=5.0,
                              count_threshold=1, min_confidence=0.0, format="csv", output=str(output))
    assert main._cmd_dataset(args) == 0, capsys.readouterr().err
    assert calls[0]["export_path"] == str(output) and calls[0]["any_export_path"] is True
    capsys.readouterr()
    output.write_text("taken", encoding="utf-8")  # an existing file is refused before anything is built
    assert main._cmd_dataset(args) == 2
    captured = capsys.readouterr()
    assert "Building" not in captured.out and "must not exist yet" in captured.err
    assert len(calls) == 1


def test_cli_search_exits_1_when_every_catalog_failed(monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: `search` exited 0 with the outage only inside the JSON."""
    import argparse

    def record(failures: list[dict[str, Any]]) -> UnifiedRecord:
        return UnifiedRecord({"ra": 187.2779154, "dec": 2.0523883}, 1, {}, {}, failures,
                             {"warnings": [], "association": {"notes": []}})

    args = argparse.Namespace(name=None, ra=187.2779154, dec=2.0523883, radius=3.0, profile=None, catalogs="simbad",
                              epoch=None, pm_ra=None, pm_dec=None, parallax=None, format="json")
    outage = record([{"catalog": "simbad", "error_type": "CatalogUnavailableError", "message": "HTTP 503"}])

    async def failed(*_a: Any, **_k: Any) -> UnifiedRecord:
        return outage

    monkeypatch.setattr(main, "crossmatch", failed)
    assert main._cmd_search(args) == 1
    captured = capsys.readouterr()
    assert "every queried catalog failed" in captured.err
    assert '"failures"' in captured.out  # the JSON is still printed

    async def empty(*_a: Any, **_k: Any) -> UnifiedRecord:
        return record([])

    monkeypatch.setattr(main, "crossmatch", empty)
    assert main._cmd_search(args) == 0  # an empty sky is not an error


def _run_python(*argv: str) -> Any:
    import os
    import subprocess
    import sys

    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, *argv], cwd=Path(api.__file__).parent, capture_output=True, env=env,
                          timeout=300, check=False)


def test_python_main_help_does_not_import_the_science_stack() -> None:
    """Finding: every CLI invocation took ~11 s (main.py imported every feature module first)."""
    done = _run_python("-X", "importtime", "main.py", "--help")
    assert done.returncode == 0, done.stderr[-2000:]
    assert b"usage: astrosearch" in done.stdout and b"cutout" in done.stdout
    imported = {line.rsplit(b"|", 1)[-1].strip() for line in done.stderr.splitlines() if line.startswith(b"import time")}
    for heavy in (b"pandas", b"astropy", b"anthropic", b"scipy", b"skycache", b"crossmatch", b"ai"):
        assert heavy not in imported, heavy


def test_cli_logs_go_to_stderr_not_stdout() -> None:
    """Finding: `cutout` printed structlog lines to stdout (mixed into the command's output)."""
    done = _run_python("-c", "import cli, structlog; cli.log_to_stderr(); "
                             "structlog.get_logger().info('cutout_fetched', bytes=1); print('RESULT')")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == b"RESULT"
    assert b"cutout_fetched" in done.stderr
