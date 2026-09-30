"""Circuit breaker, cancellation, timeouts, throttling, fallback discipline, truncation, and caching.

These go through the real provider/executor stack (httpx MockTransport or respx),
not by calling EndpointGuard.fail()/succeed() directly.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import respx
from fixture_io import FIXTURES, TARGETS
from helpers import offline_client

from crossmatch import CrossmatchService, QueryExecutor
from models import (
    CatalogQueryError,
    CatalogRegistry,
    CatalogUnavailableError,
    QueryPlan,
    RateLimitedError,
    RegistryError,
    UnifiedRecord,
    catalog_from_dict,
    validate_target,
)
from providers import (
    CacheManager,
    EndpointGuard,
    IRSAGatorProvider,
    MASTProvider,
    SDSSProvider,
    TapProvider,
    provider_map,
    source_sort_key,
)

TARGET = validate_target(187.2779154, 2.0523883)
ENDPOINT = "https://breaker.example/tap/sync"


def tap_def(name: str = "t", endpoint: str = ENDPOINT, **extra):
    entry = {
        "provider": "tap", "wavelength": "optical", "endpoint": endpoint, "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0,
    }
    entry.update(extra)
    return catalog_from_dict(name, entry)


def tap_json(rows: list[list]) -> dict:
    return {"metadata": [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}, {"name": "match_dist"}],
            "data": rows}


def fixture_response(key: str) -> httpx.Response:
    meta = json.loads((FIXTURES / "errors" / f"{key}.json").read_text(encoding="utf-8"))["exchanges"][0]
    return httpx.Response(meta["status_code"], headers={"content-type": meta["content_type"]},
                          content=(FIXTURES / "errors" / f"{key}.0.body").read_bytes())


class Archive:
    """A scriptable fake archive behind httpx.MockTransport (async handler, can hang)."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.calls = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.mode == "503":
            return httpx.Response(503, text="Service Unavailable")
        if self.mode == "hang":
            await asyncio.sleep(60)
        if self.mode == "400":
            return fixture_response("vizier_unquoted_table")
        return httpx.Response(200, json=tap_json([["OK1", 187.2779154, 2.0523883, 0.0]]))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def provider_with(guard: EndpointGuard, archive: Archive) -> TapProvider:
    return TapProvider(archive.client(), guards={ENDPOINT: guard}, cache=CacheManager(None))


# ---------------------------------------------------------------------------
# Circuit breaker through the provider stack
# ---------------------------------------------------------------------------


async def test_cancelled_half_open_probe_does_not_wedge_the_circuit() -> None:
    archive = Archive()
    guard = EndpointGuard(requests_per_second=1000, failure_threshold=1, recovery_seconds=0.2)
    provider = provider_with(guard, archive)
    catalog = tap_def()

    archive.mode = "503"
    with pytest.raises(CatalogUnavailableError):
        await provider.query(catalog, TARGET, 5.0)
    assert guard.state == "open"

    await asyncio.sleep(0.25)
    archive.mode = "hang"
    # The executor's wait_for cancels the in-flight half-open probe.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.query(catalog, TARGET, 5.0), timeout=0.3)
    assert guard.probe_in_flight is False  # cancellation reported the failed probe
    assert guard.state == "open"

    archive.mode = "ok"
    await asyncio.sleep(0.25)
    result = await provider.query(catalog, TARGET, 5.0)  # healthy archive: next probe closes it
    assert result.meta["status"] == "success"
    assert guard.state == "closed"
    for _ in range(3):
        assert (await provider.query(catalog, TARGET, 5.0)).meta["status"] == "success"


async def test_repeated_timeouts_open_the_circuit_through_the_executor() -> None:
    archive = Archive()
    archive.mode = "hang"
    guard = EndpointGuard(requests_per_second=1000, failure_threshold=2, recovery_seconds=60)
    reg = CatalogRegistry()
    reg._catalogs = {"t": tap_def(timeout_seconds=0.3)}
    executor = QueryExecutor({"tap": provider_with(guard, archive)}, registry=reg)
    plan = QueryPlan("t", "tap", ENDPOINT, {}, 5.0)
    for _ in range(2):
        _, failures = await executor.execute([plan], TARGET)
        assert failures[0].error_type == "QueryTimeoutError"
        # The provider's own (slightly shorter) budget fired, so the breaker counted it.
        assert "time budget" in failures[0].message
    assert guard.state == "open" and guard._failures == 2
    calls_before = archive.calls
    started = time.monotonic()
    _, failures = await executor.execute([plan], TARGET)
    assert failures[0].error_type == "CatalogUnavailableError" and "circuit is open" in failures[0].message
    assert archive.calls == calls_before and time.monotonic() - started < 0.2  # fails fast, no request


async def test_4xx_probe_closes_the_circuit() -> None:
    archive = Archive()
    guard = EndpointGuard(requests_per_second=1000, failure_threshold=1, recovery_seconds=0.1)
    provider = provider_with(guard, archive)
    archive.mode = "503"
    with pytest.raises(CatalogUnavailableError):
        await provider.query(tap_def(), TARGET, 5.0)
    await asyncio.sleep(0.15)
    archive.mode = "400"
    with pytest.raises(CatalogQueryError):
        await provider.query(tap_def(), TARGET, 5.0)
    assert guard.state == "closed"  # the endpoint answered: it is up


async def test_leaked_probe_expires() -> None:
    now = [100.0]
    guard = EndpointGuard(requests_per_second=1000, failure_threshold=1, recovery_seconds=10, probe_timeout_seconds=30,
                          clock=lambda: now[0])
    guard.record_failure()
    now[0] += 11
    await guard.acquire()  # probe granted, never reported (e.g. task leaked)
    with pytest.raises(CatalogUnavailableError):
        await guard.acquire()
    now[0] += 31
    await guard.acquire()  # stale probe expired: a new probe is allowed


# ---------------------------------------------------------------------------
# Throttling (429) and deterministic SQL errors
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_then_200_respects_retry_after() -> None:
    route = respx.post(ENDPOINT)
    route.side_effect = [httpx.Response(429, headers={"retry-after": "0.3"}),
                         httpx.Response(200, json=tap_json([["A", 187.2779154, 2.0523883, 0.0]]))]
    started = time.monotonic()
    async with offline_client() as client:
        result = await TapProvider(client, cache=CacheManager(None)).query(tap_def(), TARGET, 5.0)
    assert result.meta["status"] == "success" and route.call_count == 2
    assert time.monotonic() - started >= 0.29


@respx.mock
async def test_persistent_429_is_rate_limited_and_triggers_fallback() -> None:
    tap = respx.post("https://irsa.example/TAP/sync").respond(429, headers={"retry-after": "0"}, text="Too Many Requests")
    ipac = (
        "\\fixlen = T\n"
        "| designation      | ra          | dec        | err_maj | err_min | jdate        |\n"
        "| char             | double      | double     | double  | double  | double       |\n"
        "|                  | deg         | deg        | arcsec  | arcsec  |              |\n"
        "|                  |             |            |         |         |              |\n"
        " 12290669+0203085   187.277897    2.052387     0.17      0.08      2451599.8550 \n"
    )
    gator = respx.get("https://irsa.example/gator").respond(200, text=ipac, headers={"content-type": "text/plain"})
    base = CatalogRegistry().get("twomass_psc")
    params = {**base.parameters, "fallback": {"provider": "irsa_gator", "endpoint": "https://irsa.example/gator", "catalog": "fp_psc"}}
    catalog = catalog_from_dict("twomass_psc", {**base.as_dict(), "endpoint": "https://irsa.example/TAP/sync", "parameters": params})
    async with offline_client() as client:
        with pytest.raises(RateLimitedError):
            await TapProvider(client, cache=CacheManager(None)).query(catalog, TARGET, 10.0)
        reg = CatalogRegistry()
        reg._catalogs = {"twomass_psc": catalog}
        record = await CrossmatchService(reg, provider_map(client, cache=CacheManager(None))).crossmatch(*TARGETS["3c273"], radius_arcsec=10.0)
    result = record.catalog_results["twomass_psc"]
    assert tap.call_count >= 3 and gator.called
    assert result["status"] == "success" and "RateLimitedError" in result["fallback"]["reason"]
    assert issubclass(RateLimitedError, CatalogUnavailableError)


@respx.mock
async def test_sdss_sql_error_is_not_retried_and_does_not_trip_the_breaker() -> None:
    route = respx.get("https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch").mock(
        return_value=fixture_response("sdss_sql_error"))
    guards: dict[str, EndpointGuard] = {}
    catalog = CatalogRegistry().get("sdss")
    async with offline_client() as client:
        provider = SDSSProvider(client, guards=guards, cache=CacheManager(None))
        for _ in range(5):
            with pytest.raises(CatalogQueryError, match="Invalid column name 'nosuchcolumn'"):
                await provider.query(catalog, TARGET, 10.0)
    assert route.call_count == 5  # one HTTP call per query, no retries
    guard = guards[catalog.endpoint]
    assert guard.state == "closed" and guard._failures == 0


@respx.mock
async def test_query_error_does_not_trigger_fallback() -> None:
    # IRSA TAP answers HTTP 200 + QUERY_STATUS=ERROR: a broken query must be reported,
    # not masked by the Gator fallback.
    respx.post("https://irsa.example/TAP/sync").mock(return_value=fixture_response("heasarc_bad_column"))
    gator = respx.get("https://irsa.example/gator").respond(200, text="")
    base = CatalogRegistry().get("twomass_psc")
    params = {**base.parameters, "fallback": {"provider": "irsa_gator", "endpoint": "https://irsa.example/gator", "catalog": "fp_psc"}}
    catalog = catalog_from_dict("twomass_psc", {**base.as_dict(), "endpoint": "https://irsa.example/TAP/sync", "parameters": params})
    reg = CatalogRegistry()
    reg._catalogs = {"twomass_psc": catalog}
    async with offline_client() as client:
        record = await CrossmatchService(reg, provider_map(client, cache=CacheManager(None))).crossmatch(*TARGETS["3c273"], radius_arcsec=10.0)
    assert record.catalog_results["twomass_psc"]["status"] == "failed"
    assert record.catalog_results["twomass_psc"]["error_type"] == "CatalogQueryError"
    assert not gator.called


# ---------------------------------------------------------------------------
# Gator request, truncation, size limits
# ---------------------------------------------------------------------------


def _ipac(rows: list[tuple[str, float, float]]) -> str:
    lines = [
        "\\fixlen = T",
        "| designation      | ra          | dec        | err_maj | err_min | jdate        |",
        "| char             | double      | double     | double  | double  | double       |",
        "|                  | deg         | deg        | arcsec  | arcsec  |              |",
        "|                  |             |            |         |         |              |",
    ]
    widths = [len(seg) + 1 for seg in lines[1].split("|")[1:-1]]  # column slots between pipes
    for d, ra, dec in rows:
        values = [d, f"{ra:.7f}", f"{dec:.7f}", "0.10", "0.10", "2451599.8550"]
        lines.append("".join(" " + v.ljust(w - 1) for v, w in zip(values, widths)))
    return "\n".join(lines) + "\n"


def test_gator_request_uses_arcsec_radius() -> None:
    catalog = CatalogRegistry().get("twomass_psc")
    params = IRSAGatorProvider(offline_client(), cache=CacheManager(None)).build_params(catalog, TARGET, 10.0)
    assert params["radunits"] == "arcsec" and float(params["radius"]) == pytest.approx(10.0)
    assert params["objstr"] == "187.277915400 2.052388300" and params["spatial"] == "cone"


@respx.mock
async def test_gator_fallback_keeps_nearest_max_rows_and_flags_truncation() -> None:
    rows = [(f"J{i:02d}", 187.2779154, 2.0523883 + i / 3600.0) for i in (5, 1, 4, 2, 3)]  # unordered, 1-5"
    respx.get("https://irsa.example/gator").respond(200, text=_ipac(rows), headers={"content-type": "text/plain"})
    base = CatalogRegistry().get("twomass_psc")
    catalog = catalog_from_dict("twomass_psc", {**base.as_dict(), "endpoint": "https://irsa.example/gator", "max_rows": 3})
    async with offline_client() as client:
        result = await IRSAGatorProvider(client, cache=CacheManager(None)).query(catalog, TARGET, 10.0)
    assert [s.source_id for s in result] == ["J01", "J02", "J03"]
    assert result.meta["truncated"] is True and result.meta["raw_row_count"] == 5


@pytest.mark.parametrize(("n_rows", "truncated"), [(3, False), (4, True)])
@respx.mock
async def test_truncated_only_when_more_than_max_rows(n_rows: int, truncated: bool) -> None:
    rows = [[f"S{i}", 187.2779154, 2.0523883 + i / 36000.0, 0.0] for i in range(n_rows)]
    route = respx.post(ENDPOINT).respond(json=tap_json(rows))
    async with offline_client() as client:
        result = await TapProvider(client, cache=CacheManager(None)).query(tap_def(max_rows=3), TARGET, 5.0)
    assert "SELECT TOP 4 " in httpx.QueryParams(route.calls.last.request.content.decode())["QUERY"]
    assert len(result) == 3 and result.meta["truncated"] is truncated


@respx.mock
async def test_overflow_status_marks_truncated() -> None:
    votable = (
        '<?xml version="1.0"?><VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">'
        '<INFO name="QUERY_STATUS" value="OK"/><TABLE><FIELD name="id" datatype="char" arraysize="*"/>'
        '<FIELD name="ra" datatype="double" unit="deg"/><FIELD name="dec" datatype="double" unit="deg"/>'
        '<DATA><TABLEDATA><TR><TD>A</TD><TD>187.2779154</TD><TD>2.0523883</TD></TR></TABLEDATA></DATA></TABLE>'
        '<INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE></VOTABLE>'
    )
    respx.post(ENDPOINT).respond(200, text=votable, headers={"content-type": "application/x-votable+xml"})
    catalog = tap_def(parameters={"columns": ["id", "ra", "dec"], "id_field": "id", "format": "votable"})
    async with offline_client() as client:
        result = await TapProvider(client, cache=CacheManager(None)).query(catalog, TARGET, 5.0)
    assert result.meta["truncated"] is True and len(result) == 1


@respx.mock
async def test_oversized_response_is_rejected() -> None:
    respx.post(ENDPOINT).respond(json=tap_json([["A" * 500, 187.2779154, 2.0523883, 0.0]]))
    async with offline_client() as client:
        provider = TapProvider(client, cache=CacheManager(None), max_response_bytes=200)
        with pytest.raises(CatalogQueryError, match="byte limit"):
            await provider.query(tap_def(), TARGET, 5.0)


# ---------------------------------------------------------------------------
# MAST: silent column drops and wrong unit metadata
# ---------------------------------------------------------------------------


def _ps1_route() -> respx.Route:
    body = (FIXTURES / "3c273" / "panstarrs_dr2.0.body").read_bytes()
    return respx.get("https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json").respond(
        200, content=body, headers={"content-type": "application/json"})


@respx.mock
async def test_mast_missing_requested_column_fails_loudly() -> None:
    _ps1_route()
    base = CatalogRegistry().get("panstarrs_dr2")
    catalog = catalog_from_dict("panstarrs_dr2", {**base.as_dict(), "parameters": {
        **base.parameters, "columns": [*base.parameters["columns"], "raMeanErrr"]}})
    async with offline_client() as client:
        with pytest.raises(CatalogQueryError, match="raMeanErrr"):
            await MASTProvider(client, cache=CacheManager(None)).query(catalog, TARGET, 10.0)


@respx.mock
async def test_mast_unit_metadata_is_corrected() -> None:
    _ps1_route()
    async with offline_client() as client:
        result = await MASTProvider(client, cache=CacheManager(None)).query(CatalogRegistry().get("panstarrs_dr2"), TARGET, 10.0)
    units = {c["name"]: c["unit"] for c in result.meta["columns"]}
    assert units["raMeanErr"] == "arcsec" and units["decMeanErr"] == "arcsec"
    assert units["distance"] == "deg"  # 1.87e-06 for a 0.0067" separation
    assert result[0].data["distance"] * 3600 == pytest.approx(result[0].metadata["query_separation_arcsec"], abs=1e-3)


# ---------------------------------------------------------------------------
# Dropped rows, ties
# ---------------------------------------------------------------------------


@respx.mock
async def test_partially_unusable_payload_reports_dropped_rows(caplog) -> None:
    respx.post(ENDPOINT).respond(json=tap_json([
        ["good", 187.2779154, 2.0523883, 0.0], ["nullra", None, 2.0523883, 0.0], ["text", "abc", 2.05, 0.0]]))
    reg = CatalogRegistry()
    reg._catalogs = {"t": tap_def()}
    async with offline_client() as client:
        with caplog.at_level("WARNING", logger="astrosearch.providers"):
            record = await CrossmatchService(reg, provider_map(client, cache=CacheManager(None))).crossmatch(
                187.2779154, 2.0523883, radius_arcsec=5.0)
    stats = record.provenance["catalog_stats"]["t"]
    assert stats["status"] == "success" and stats["row_count"] == 1
    assert stats["raw_row_count"] == 3 and stats["dropped_rows"] == 2
    assert record.catalog_results["t"]["dropped_rows"] == 2
    assert "2 of 3" in stats["warnings"][0] and "2 of 3" in caplog.text


async def test_simbad_host_star_sorts_before_its_planet() -> None:
    from helpers import run_catalog

    sources, _ = await run_catalog("simbad", "hd209458")
    assert sources[0].source_id == "HD 209458"
    assert any(s.source_id.replace(" ", "") == "HD209458b" for s in sources[1:])


def test_sort_key_breaks_coincident_ties_deterministically() -> None:
    from models import CatalogSource

    def src(sid: str, sep: float, otype: str | None) -> CatalogSource:
        return CatalogSource("simbad", sid, 0, 0, None, {}, {"epoch_separation_arcsec": sep,
                                                              "physical": {"object_type": otype} if otype else {}}, {})

    rows = [src("Star c", 1e-12, "Pl"), src("Star b", 0.0, "Pl"), src("Star", 3e-12, "BY*"), src("Far", 0.5, "*")]
    assert [s.source_id for s in sorted(rows, key=source_sort_key)] == ["Star", "Star b", "Star c", "Far"]


# ---------------------------------------------------------------------------
# API search cache
# ---------------------------------------------------------------------------


def _fake_service(failures: list[dict]):
    calls = {"n": 0}

    class FakeService:
        registry = CatalogRegistry()

        async def crossmatch(self, ra, dec, *, query=None, **kwargs):
            calls["n"] += 1
            return UnifiedRecord({"ra": ra, "dec": dec}, 1, {}, {}, list(failures), {})

    return FakeService(), calls


@pytest.mark.parametrize(("failures", "ttl", "expected_calls"), [
    ([{"catalog": "gaia_dr3", "status": "failed", "error_type": "QueryTimeoutError"}], "300", 2),  # never cache failures
    ([], "300", 1),
    ([], "0", 2),  # API_SEARCH_CACHE_TTL_SECONDS=0 disables the cache
])
def test_api_search_cache_policy(monkeypatch, failures, ttl, expected_calls) -> None:
    import api

    service, calls = _fake_service(failures)
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    monkeypatch.setenv("API_SEARCH_CACHE_TTL_SECONDS", ttl)
    req = api.SearchRequest(ra=10.0, dec=20.0, radius_arcsec=5.0)
    for _ in range(2):
        asyncio.run(api._search(req))
    assert calls["n"] == expected_calls


# ---------------------------------------------------------------------------
# Registry loading and startup validation
# ---------------------------------------------------------------------------


def test_unparseable_registry_raises_instead_of_silent_defaults(tmp_path) -> None:
    path = tmp_path / "catalogs.yaml"
    path.write_text("catalogs: [unclosed", encoding="utf-8")
    with pytest.raises(RegistryError, match="could not be parsed"):
        CatalogRegistry(path)


def _bad_registry(tmp_path):
    path = tmp_path / "catalogs.yaml"
    path.write_text(
        "catalogs:\n  bad:\n    provider: tap\n    wavelength: x\n    endpoint: https://x.example/tap\n    table: t\n"
        "    epoch: 2000.0\n    parameters: {columns: [id, ra, dec, e]}\n"
        "    pos_error: {columns: [e], units: d, kind: sigma}\n",  # 'd' is a day, not an angle
        encoding="utf-8",
    )
    return path


def test_startup_check_logs_or_raises(tmp_path, monkeypatch, caplog) -> None:
    reg = CatalogRegistry(_bad_registry(tmp_path))
    with caplog.at_level("WARNING", logger="astrosearch.models"):
        problems = reg.startup_check(strict=False)
    assert any("is not an angle" in p for p in problems) and "is not an angle" in caplog.text
    monkeypatch.setenv("CATALOG_REGISTRY_STRICT", "true")
    with pytest.raises(RegistryError):
        reg.startup_check()


def test_api_refuses_to_start_with_invalid_registry_in_strict_mode(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from api import app

    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(_bad_registry(tmp_path)))
    monkeypatch.setenv("CATALOG_REGISTRY_STRICT", "true")
    with pytest.raises(RegistryError), TestClient(app):
        pass


def test_build_service_validates_registry(tmp_path, monkeypatch) -> None:
    from main import build_service

    monkeypatch.setenv("CATALOG_REGISTRY_STRICT", "true")
    with pytest.raises(RegistryError):
        build_service(registry_path=str(_bad_registry(tmp_path)))
