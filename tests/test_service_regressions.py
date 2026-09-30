"""Regression tests for service-level review findings (round 3).

Fallback error reporting, widened-cone truncation, transport retries, Xamin errors,
recorded Gator fallback replay, profile validation, name-resolution errors, catalog
metrics, timeout caps and the bounded provider cache.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import respx
from fixture_io import FIXTURES, TARGETS, load_exchanges, replay_side_effect
from helpers import make_service, offline_client, run_catalog

import providers as providers_module
from crossmatch import AdvancedQuery, CrossmatchService, QueryExecutor, QueryValidator
from models import (
    IRSA_GATOR,
    IRSA_TAP,
    CatalogDefinition,
    CatalogQueryError,
    CatalogRegistry,
    QueryPlan,
    Settings,
    UnifiedRecord,
    catalog_from_dict,
    validate_target,
)
from providers import CacheManager, CatalogProvider, HEASARCXaminProvider, QueryResult, SDSSProvider, TapProvider, provider_map

TARGET = TARGETS["3c273"]


def tap_def(name: str, endpoint: str, **extra) -> CatalogDefinition:
    entry = {
        "provider": "tap", "wavelength": "optical", "endpoint": endpoint, "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0, "profiles": ["full"],
    }
    entry.update(extra)
    return catalog_from_dict(name, entry)


def registry_of(*catalogs: CatalogDefinition) -> CatalogRegistry:
    reg = CatalogRegistry()
    reg._catalogs = {c.name: c for c in catalogs}
    return reg


def tap_json(rows: list[list]) -> dict:
    return {"metadata": [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}], "data": rows}


# ---------------------------------------------------------------------------
# Primary and fallback both failing: both causes are reported
# ---------------------------------------------------------------------------


@respx.mock
async def test_primary_and_fallback_failures_are_both_reported() -> None:
    base = CatalogRegistry().get("twomass_psc")
    params = {**base.parameters, "fallback": {"provider": "irsa_gator", "endpoint": "https://irsa.example/gator",
                                              "catalog": "fp_psc"}}
    catalog = catalog_from_dict("twomass_psc", {**base.as_dict(), "endpoint": "https://irsa.example/tap",
                                                "parameters": params})
    respx.post("https://irsa.example/tap").respond(503, text="TAP down")
    respx.get("https://irsa.example/gator").respond(
        200, text='[struct stat="ERROR", msg="gator broke"]', headers={"content-type": "text/plain"})
    async with offline_client() as client:
        record = await CrossmatchService(registry_of(catalog), provider_map(client, cache=CacheManager(None)),
                                         timeout=10.0).crossmatch(*TARGET, radius_arcsec=10.0)
    stats = record.catalog_results["twomass_psc"]
    assert stats["status"] == "failed" and stats["error_type"] == "CatalogQueryError"
    assert "primary CatalogUnavailableError" in stats["message"] and "HTTP 503" in stats["message"]
    assert "TAP down" in stats["message"] and "gator broke" in stats["message"]
    assert stats["fallback"]["provider"] == "irsa_gator"
    assert "HTTP 503" in stats["fallback"]["reason"] and "gator broke" in stats["fallback"]["error"]
    assert record.failures[0]["fallback"]["endpoint"] == "https://irsa.example/gator"


# ---------------------------------------------------------------------------
# Widened cone cut by TOP N: flagged, never a silent 'empty'
# ---------------------------------------------------------------------------


async def test_widened_cone_cut_at_row_limit_is_flagged(monkeypatch) -> None:
    monkeypatch.setenv("EPOCH_PAD_MAX_ROWS", "20")
    catalog = tap_def("nopm", "https://nopm.example/tap", epoch=2016.0, max_rows=5)
    # Target epoch 2000 without a proper motion: the cone is widened to 5 + 10.5 x 16 arcsec
    # and the archive returns row_limit + 1 = 21 rows, all pad rows (100-120" away).
    rows = [[f"far{i}", 10.0, 20.0 + (100.0 + i) / 3600.0] for i in range(21)]
    with respx.mock() as router:
        route = router.post("https://nopm.example/tap").respond(json=tap_json(rows))
        async with offline_client() as client:
            record = await make_service(client, registry_of(catalog)).crossmatch(10.0, 20.0, radius_arcsec=5.0,
                                                                                 epoch=2000.0)
    assert "TOP 21 " in httpx.QueryParams(route.calls.last.request.content.decode())["QUERY"]
    res = record.catalog_results["nopm"]
    assert res["truncated"] is True
    assert any("widened cone was cut" in w for w in res["warnings"])
    assert res["status"] == "empty" and res["epoch_incomplete"] is True
    assert res["pad_row_count"] == 21


async def test_provider_level_archive_truncation_uses_the_extra_probe_row(monkeypatch) -> None:
    monkeypatch.setenv("EPOCH_PAD_MAX_ROWS", "20")
    catalog = tap_def("nopm", "https://nopm.example/tap", epoch=2016.0, max_rows=5)
    target = validate_target(10.0, 20.0, epoch=2000.0)
    with respx.mock() as router:
        router.post("https://nopm.example/tap").respond(
            json=tap_json([[f"far{i}", 10.0, 20.0 + (100.0 + i) / 3600.0] for i in range(20)]))
        async with offline_client() as client:
            exact = await TapProvider(client, cache=CacheManager(None)).query(catalog, target, 5.0)
    assert exact.meta["archive_truncated"] is False  # exactly row_limit rows: complete
    with respx.mock() as router:
        router.post("https://nopm.example/tap").respond(
            json=tap_json([[f"far{i}", 10.0, 20.0 + (100.0 + i) / 3600.0] for i in range(21)]))
        async with offline_client() as client:
            cut = await TapProvider(client, cache=CacheManager(None)).query(catalog, target, 5.0)
    assert cut.meta["archive_truncated"] is True and cut.meta["truncated"] is True
    assert any("widened cone was cut at 20 rows" in w for w in cut.meta["warnings"])


# ---------------------------------------------------------------------------
# Transport paths (previously surviving mutants)
# ---------------------------------------------------------------------------


async def test_sdss_conesearch_mode_sends_radius_in_arcmin() -> None:
    meta = json.loads((FIXTURES / "errors" / "sdss_conesearch_3c273.json").read_text(encoding="utf-8"))["exchanges"][0]
    body = (FIXTURES / "errors" / "sdss_conesearch_3c273.0.body").read_bytes()
    endpoint = "https://skyserver.sdss.org/dr18/SkyServerWS/ConeSearch/ConeSearchService"
    base = CatalogRegistry().get("sdss")
    catalog = catalog_from_dict("sdss", {**base.as_dict(), "endpoint": endpoint,
                                         "parameters": {**base.parameters, "mode": "conesearch", "field_map": {}}})
    with respx.mock() as router:
        route = router.get(endpoint).respond(200, content=body, headers={"content-type": meta["content_type"]})
        async with offline_client() as client:
            result = await SDSSProvider(client, cache=CacheManager(None)).query(catalog, validate_target(*TARGET), 10.0)
    sent = route.calls.last.request.url.params
    assert sent["sr"] == f"{10.0 / 60.0:.10f}"  # arcmin (ConeSearchService applies sr in arcmin)
    assert float(sent["ra"]) == pytest.approx(TARGET[0]) and sent["format"] == "csv"
    assert result.meta["mode"] == "conesearch" and len(result) >= 1


async def test_transient_connect_then_read_errors_recover_on_the_third_attempt() -> None:
    catalog = tap_def("t", "https://flaky.example/tap")
    with respx.mock() as router:
        route = router.post("https://flaky.example/tap").mock(side_effect=[
            httpx.ConnectError("refused"), httpx.ReadError("reset"),
            httpx.Response(200, json=tap_json([["A", TARGET[0], TARGET[1]]])),
        ])
        async with offline_client() as client:
            result = await TapProvider(client, cache=CacheManager(None)).query(catalog, validate_target(*TARGET), 5.0)
    assert route.call_count == 3
    assert [s.source_id for s in result] == ["A"] and result.meta["status"] == "success"


async def test_retry_after_is_capped_at_five_seconds(monkeypatch) -> None:
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args, **kwargs):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(providers_module.asyncio, "sleep", fake_sleep)
    catalog = tap_def("t", "https://slow.example/tap")
    with respx.mock() as router:
        route = router.post("https://slow.example/tap").mock(side_effect=[
            httpx.Response(503, headers={"retry-after": "3600"}),
            httpx.Response(503, headers={"retry-after": "3600"}),
            httpx.Response(200, json=tap_json([["A", TARGET[0], TARGET[1]]])),
        ])
        async with offline_client() as client:
            started = time.monotonic()
            result = await TapProvider(client, cache=CacheManager(None)).query(catalog, validate_target(*TARGET), 5.0)
    assert route.call_count == 3 and result[0].source_id == "A"
    retry_delays = [d for d in delays if d > 0]
    assert retry_delays == [5.0, 5.0]
    assert time.monotonic() - started < 5.0


@pytest.mark.parametrize(("payload", "message"), [
    ({"success": False, "messages": [{"status": "Failure", "id": "Unknown table",
                                      "text": "No information available on nosuchtable. Is this a valid table?"}]},
     r"^HEASARC Xamin error: .*Unknown table"),
    ({"status": True, "messages": []}, "^HEASARC Xamin returned no result rows"),  # an object with no row array
])
async def test_xamin_error_documents_are_errors_not_empty(payload: dict, message: str) -> None:
    catalog = catalog_from_dict("nvss_x", {"provider": "heasarc_xamin", "wavelength": "radio", "table": "nvss",
                                           "endpoint": "https://xamin.example/query", "epoch": None})
    with respx.mock() as router:
        router.get("https://xamin.example/query").respond(200, json=payload)
        async with offline_client() as client:
            with pytest.raises(CatalogQueryError, match=message):
                await HEASARCXaminProvider(client, cache=CacheManager(None)).query(catalog, validate_target(*TARGET), 10.0)


async def test_xamin_real_empty_result_is_empty() -> None:
    catalog = catalog_from_dict("nvss_x", {"provider": "heasarc_xamin", "wavelength": "radio", "table": "nvss",
                                           "endpoint": "https://xamin.example/query", "epoch": None})
    with respx.mock() as router:
        router.get("https://xamin.example/query").respond(200, json={"request": [], "status": True, "messages": []})
        async with offline_client() as client:
            result = await HEASARCXaminProvider(client, cache=CacheManager(None)).query(
                catalog, validate_target(*TARGET), 10.0)
    assert result.meta["status"] == "empty" and list(result) == []


# ---------------------------------------------------------------------------
# Recorded real Gator responses replayed as the fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["twomass_psc", "allwise"])
async def test_recorded_gator_fallback_matches_the_tap_results(name: str) -> None:
    tap_sources, failure = await run_catalog(name, "3c273")
    assert failure is None and tap_sources
    registry = CatalogRegistry()
    catalog = registry.get(name)
    with respx.mock(assert_all_called=True) as router:
        router.post(IRSA_TAP).respond(503, text="IRSA TAP outage")
        router.get(url__startswith=IRSA_GATOR).mock(side_effect=replay_side_effect(load_exchanges("fallback/3c273", [name])))
        async with offline_client() as client:
            executor = QueryExecutor(provider_map(client, cache=CacheManager(None)), registry=registry)
            plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, 10.0, catalog.wavelength)
            successes, failures = await executor.execute([plan], validate_target(*TARGET))
    assert failures == []
    (_, gator_sources), = successes
    assert gator_sources.meta["fallback"]["provider"] == "irsa_gator"
    assert "HTTP 503" in gator_sources.meta["fallback"]["reason"]
    tap = {s.source_id: s for s in tap_sources}
    gator = {s.source_id: s for s in gator_sources}
    assert set(gator) == set(tap)
    for sid, src in gator.items():
        assert src.metadata["query_separation_arcsec"] == pytest.approx(tap[sid].metadata["query_separation_arcsec"], abs=1e-3)
        assert src.positional_error_arcsec == pytest.approx(tap[sid].positional_error_arcsec, rel=1e-6)
        assert src.epoch == pytest.approx(tap[sid].epoch, abs=1e-6)


# ---------------------------------------------------------------------------
# Unknown profile / unresolvable name
# ---------------------------------------------------------------------------


async def test_unknown_profile_is_rejected_not_silently_empty() -> None:
    async with offline_client() as client:
        service = make_service(client)
        with pytest.raises(ValueError, match="Unknown profile 'radoi'"):
            await service.crossmatch(*TARGET, radius_arcsec=5.0, profile="radoi")
        with pytest.raises(ValueError, match="Unknown profile"):
            service.planner.plan(5.0, profile="radoi")
    query = AdvancedQuery.from_dict({"ra": TARGET[0], "dec": TARGET[1], "profiles": ["radoi"]})
    with pytest.raises(ValueError, match="Unknown profile"):
        QueryValidator.validate(query, CatalogRegistry())
    QueryValidator.validate(AdvancedQuery.from_dict({"ra": 1.0, "dec": 2.0, "profiles": ["radio"]}), CatalogRegistry())


def _sesame_nothing_found(router: respx.MockRouter) -> None:
    router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
        200, headers={"content-type": "text/xml"},
        text='<?xml version="1.0" encoding="UTF-8"?><Sesame><Target option="SNV"><name>Xyzzy Nonexistent 123</name>'
             '<INFO>*** Nothing found *** </INFO></Target></Sesame>')


def test_api_unresolvable_name_is_404_not_502(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setattr(api, "cache", CacheManager(None))
    with TestClient(api.app) as client, respx.mock(assert_all_called=True) as router:
        _sesame_nothing_found(router)
        response = client.post("/api/v1/search", json={"name": "Xyzzy Nonexistent 123", "radius_arcsec": 5.0})
        assert response.status_code == 404, response.text
        assert "could not be resolved" in response.json()["detail"]
        batch = client.post("/api/v1/search/batch", json=[{"name": "Xyzzy Nonexistent 123"}])
        assert batch.status_code == 200 and batch.json()[0]["status_code"] == 404
        bad_profile = client.post("/api/v1/search", json={"ra": 10.0, "dec": 20.0, "profile": "radoi"})
        assert bad_profile.status_code == 422 and "Unknown profile" in bad_profile.json()["detail"]


def test_cli_unresolvable_name_and_unknown_profile_exit_cleanly(monkeypatch, capsys) -> None:
    import main

    with respx.mock(assert_all_called=True) as router:
        _sesame_nothing_found(router)
        monkeypatch.setattr("sys.argv", ["astrosearch", "search", "--name", "Xyzzy Nonexistent 123", "--radius", "5"])
        with pytest.raises(SystemExit) as exc:
            main.main()
    assert exc.value.code == 2  # an unknown name is the user's input (HTTP 404), not an upstream failure (exit 1)
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "No coordinates found" in err and "Traceback" not in err

    monkeypatch.setattr("sys.argv", ["astrosearch", "search", "--ra", "187.2779", "--dec", "2.05", "--radius", "5",
                                     "--profile", "radoi"])
    with respx.mock(assert_all_mocked=True), pytest.raises(SystemExit) as exc:  # no archive may be contacted
        main.main()
    assert exc.value.code == 2 and "Unknown profile 'radoi'" in capsys.readouterr().err  # invalid input (422)


# ---------------------------------------------------------------------------
# Catalog metrics
# ---------------------------------------------------------------------------


def test_catalog_query_outcomes_reach_prometheus(monkeypatch) -> None:
    from prometheus_client import REGISTRY

    import api

    def value(catalog: str, status: str) -> float:
        return REGISTRY.get_sample_value("astrosearch_catalog_queries_total", {"catalog": catalog, "status": status}) or 0.0

    class FakeService:
        registry = CatalogRegistry()

        async def crossmatch(self, ra, dec, *, query=None, **kwargs):
            stats = {"gaia_dr3": {"status": "success", "elapsed_ms": 120.0},
                     "twomass_psc": {"status": "empty", "fallback": {"provider": "irsa_gator"}, "elapsed_ms": 50.0},
                     "xmm": {"status": "failed", "elapsed_ms": 10.0}}
            return UnifiedRecord({"ra": ra, "dec": dec}, 3, {}, {}, [], {"catalog_stats": stats})

    before = (value("gaia_dr3", "success"), value("twomass_psc", "empty+fallback"), value("xmm", "failed"))
    monkeypatch.setattr(api, "get_service", lambda: FakeService())
    monkeypatch.setattr(api, "cache", CacheManager(None))
    asyncio.run(api._search(api.SearchRequest(ra=10.0, dec=20.0, radius_arcsec=5.0)))
    after = (value("gaia_dr3", "success"), value("twomass_psc", "empty+fallback"), value("xmm", "failed"))
    assert [a - b for a, b in zip(after, before)] == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Timeouts: operator cap and fallback budget
# ---------------------------------------------------------------------------


def test_settings_timeout_cap(monkeypatch) -> None:
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CATALOG_TIMEOUT_CAP_SECONDS", raising=False)
    assert Settings().catalog_timeout_cap_seconds is None  # registry timeouts (60-90 s) apply
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "5")
    assert Settings().catalog_timeout_cap_seconds == 5.0  # explicitly set: caps every catalog
    monkeypatch.setenv("CATALOG_TIMEOUT_CAP_SECONDS", "12")
    assert Settings().catalog_timeout_cap_seconds == 12.0
    assert Settings(REQUEST_TIMEOUT_SECONDS=3).catalog_timeout_cap_seconds == 12.0
    monkeypatch.delenv("CATALOG_TIMEOUT_CAP_SECONDS")
    assert Settings(REQUEST_TIMEOUT_SECONDS=3).catalog_timeout_cap_seconds == 3.0
    with pytest.raises(ValueError):
        Settings(CATALOG_TIMEOUT_CAP_SECONDS=0)


class SleepyProvider(CatalogProvider):
    def __init__(self, delay: float, fail: bool = False) -> None:
        self.delay, self.fail, self.seen_timeouts = delay, fail, []

    async def query(self, catalog, target, radius_arcsec):
        self.seen_timeouts.append(catalog.timeout_seconds)
        if self.fail:
            from models import CatalogUnavailableError
            raise CatalogUnavailableError("down")
        await asyncio.sleep(self.delay)
        return QueryResult([], {})


async def test_timeout_cap_bounds_registry_timeouts() -> None:
    catalog = tap_def("slow", "https://slow.example/tap", timeout_seconds=90.0)
    provider = SleepyProvider(5.0)
    executor = QueryExecutor({"tap": provider}, registry=registry_of(catalog), timeout_cap=0.2)
    started = time.monotonic()
    _, failures = await executor.execute([QueryPlan("slow", "tap", catalog.endpoint, {}, 5.0)], validate_target(*TARGET))
    assert time.monotonic() - started < 2.0
    assert failures[0].error_type == "QueryTimeoutError" and "0.2s" in failures[0].message
    assert provider.seen_timeouts == [0.2]  # providers size their budget from the capped value


async def test_fallback_gets_the_remaining_budget_not_a_fresh_one(monkeypatch) -> None:
    monkeypatch.setattr(QueryExecutor, "FALLBACK_MIN_SECONDS", 0.2)
    catalog = tap_def("c", "https://p.example/tap", timeout_seconds=0.5,
                      parameters={"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json",
                                  "fallback": {"provider": "slow_fallback"}})
    primary = SleepyProvider(10.0)  # times out after 0.5 s
    fallback = SleepyProvider(10.0)
    executor = QueryExecutor({"tap": primary, "slow_fallback": fallback}, registry=registry_of(catalog))
    started = time.monotonic()
    _, failures = await executor.execute([QueryPlan("c", "tap", catalog.endpoint, {}, 5.0)], validate_target(*TARGET))
    elapsed = time.monotonic() - started
    assert 0.6 < elapsed < 1.5  # 0.5 s primary + at least 0.2 s fallback, never 2 x 0.5 s + slack
    assert fallback.seen_timeouts[0] == pytest.approx(0.2, abs=0.05)
    assert "primary QueryTimeoutError" in failures[0].message and "fallback QueryTimeoutError" in failures[0].message


async def test_fast_primary_failure_leaves_the_fallback_most_of_the_budget() -> None:
    catalog = tap_def("c", "https://p.example/tap", timeout_seconds=3.0,
                      parameters={"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json",
                                  "fallback": {"provider": "fb"}})
    fallback = SleepyProvider(0.0)
    executor = QueryExecutor({"tap": SleepyProvider(0.0, fail=True), "fb": fallback}, registry=registry_of(catalog))
    successes, failures = await executor.execute([QueryPlan("c", "tap", catalog.endpoint, {}, 5.0)],
                                                 validate_target(*TARGET))
    assert failures == [] and successes[0][1].meta["fallback"]["provider"] == "fb"
    assert 2.5 < fallback.seen_timeouts[0] <= 3.0


# ---------------------------------------------------------------------------
# Bounded provider cache
# ---------------------------------------------------------------------------


def test_provider_cache_is_bounded_lru(monkeypatch) -> None:
    cache = CacheManager(None, max_entries=3, max_bytes=10_000)
    for key in "abc":
        cache.set(key, {"content": "x" * 100}, ttl=60)
    assert cache.get("a") is not None  # 'a' becomes most recently used
    cache.set("d", {"content": "x" * 100}, ttl=60)
    assert cache.get("b") is None and cache.get("a") is not None and cache.entry_count == 3
    # Byte budget: a large body evicts the oldest entries; a body above the budget is not kept.
    cache.set("big", {"content": "y" * 9_000}, ttl=60)
    assert cache.local_bytes <= 10_000 and cache.get("big") is not None
    cache.set("huge", {"content": "z" * 20_000}, ttl=60)
    assert cache.get("huge") is None and cache.local_bytes <= 10_000
    # Expired entries are swept on write, not only when read again.
    now = [1000.0]
    monkeypatch.setattr(providers_module.time, "monotonic", lambda: now[0])
    small = CacheManager(None, max_entries=100, max_bytes=10_000_000)
    for i in range(50):
        small.set(f"k{i}", {"content": "x"}, ttl=10)
    now[0] += 11.0
    small.set("fresh", {"content": "x"}, ttl=10)
    assert small.entry_count == 1


def test_provider_cache_limits_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_CACHE_MAX_ENTRIES", "7")
    monkeypatch.setenv("PROVIDER_CACHE_MAX_BYTES", "12345")
    cache = CacheManager(None)
    assert (cache.max_entries, cache.max_bytes) == (7, 12345)
    empty = CacheManager(None)
    assert bool(empty) is True  # 'cache or CacheManager()' must keep an empty cache
    provider = TapProvider(offline_client(), cache=empty)
    assert provider.cache is empty


# ---------------------------------------------------------------------------
# Catalogs outside the profile: one rule for the library, the API and the CLI
# ---------------------------------------------------------------------------


async def test_library_crossmatch_refuses_a_catalog_outside_the_profile() -> None:
    """``catalogs`` is intersected with ``profile``: through the library (not only main/api) a
    catalogue outside it is an input error with the one shared message, before any request."""
    import main
    from crossmatch import check_catalogs_in_profiles

    registry = CatalogRegistry()
    with pytest.raises(ValueError) as expected:
        check_catalogs_in_profiles(registry, ["nvss"], "optical")
    message = str(expected.value)
    assert message.startswith("Catalog(s) nvss are not in profile 'optical' and would not be queried")
    with pytest.raises(ValueError) as via_main:
        main.check_catalogs_in_profile(registry, ["nvss"], "optical")
    assert str(via_main.value) == message
    with pytest.raises(ValueError) as via_validator:
        QueryValidator.validate(AdvancedQuery.from_dict(
            {"ra": TARGET[0], "dec": TARGET[1], "catalogs": ["nvss"], "profiles": ["optical"]}), registry)
    assert str(via_validator.value) == message

    with respx.mock(assert_all_called=False) as router:
        route = router.route().respond(500)
        async with offline_client() as client:
            service = make_service(client, registry)
            with pytest.raises(ValueError) as via_crossmatch:
                await service.crossmatch(*TARGET, radius_arcsec=5.0, profile="optical", catalogs=["nvss"])
            assert str(via_crossmatch.value) == message
            with pytest.raises(ValueError) as via_stream:
                async for _event in service.crossmatch_stream(*TARGET, radius_arcsec=5.0, profile="optical",
                                                              catalogs=["nvss"]):
                    pass
            assert str(via_stream.value) == message
            with pytest.raises(ValueError) as via_prepare:
                service.prepare(*TARGET, profile="optical", catalogs=["nvss", "gaia_dr3"])
            assert str(via_prepare.value) == message  # only the catalogue outside is named
            # Inside the profile (or with no profile) the selection plans as asked.
            assert [p.catalog for p in service.prepare(*TARGET, profile="radio", catalogs=["nvss"]).plans] == ["nvss"]
            assert [p.catalog for p in service.prepare(*TARGET, catalogs=["nvss"]).plans] == ["nvss"]
        assert not route.called


def test_query_builder_plans_a_profileless_catalog_for_every_profile() -> None:
    """The validator lets a catalogue without profiles through with any profile (it is planned
    for every profile, as QueryPlanner.plan does); the builder must then plan it, not drop it."""
    registry = registry_of(tap_def("plain", "https://example.test/tap", profiles=[]),
                           tap_def("optical_only", "https://example.test/tap2", profiles=["optical"]))
    query = AdvancedQuery.from_dict({"ra": 1.0, "dec": 2.0, "catalogs": ["plain"], "profiles": ["optical"]})
    assert QueryValidator.validate(query, registry)
    from crossmatch import QueryBuilder

    assert [p.catalog for p in QueryBuilder(registry).build(query)] == ["plain"]
    everything = AdvancedQuery.from_dict({"ra": 1.0, "dec": 2.0, "profiles": ["optical"]})
    assert sorted(p.catalog for p in QueryBuilder(registry).build(everything)) == ["optical_only", "plain"]
