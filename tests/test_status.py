"""Status honesty: 'success' vs 'empty' vs 'failed', error classes, retries, fallback, caching."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx
from fixture_io import FIXTURES
from helpers import offline_client

from crossmatch import CrossmatchService, QueryExecutor
from models import (
    CatalogDefinition,
    CatalogQueryError,
    CatalogRegistry,
    CatalogUnavailableError,
    QueryPlan,
    QueryTimeoutError,
    ResponseParseError,
    catalog_from_dict,
    validate_target,
)
from providers import CacheManager, CatalogProvider, QueryResult, TapProvider, provider_map

TARGET = (187.2779154, 2.0523883)


def tap_def(name: str, endpoint: str, **extra) -> CatalogDefinition:
    entry = {
        "provider": "tap", "wavelength": "optical", "endpoint": endpoint, "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0,
    }
    entry.update(extra)
    return catalog_from_dict(name, entry)


def registry_of(*catalogs: CatalogDefinition) -> CatalogRegistry:
    reg = CatalogRegistry()
    reg._catalogs = {c.name: c for c in catalogs}
    return reg


def tap_json(rows: list[list]) -> dict:
    return {"metadata": [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}, {"name": "match_dist"}],
            "data": rows}


def fixture_response(key: str) -> httpx.Response:
    meta = json.loads((FIXTURES / "errors" / f"{key}.json").read_text(encoding="utf-8"))["exchanges"][0]
    return httpx.Response(meta["status_code"], headers={"content-type": meta["content_type"]},
                          content=(FIXTURES / "errors" / f"{key}.0.body").read_bytes())


async def crossmatch_with(reg: CatalogRegistry, radius: float = 10.0):
    async with offline_client() as client:
        service = CrossmatchService(reg, provider_map(client, cache=CacheManager(None)), timeout=10.0)
        return await service.crossmatch(*TARGET, radius_arcsec=radius)


@respx.mock
async def test_success_empty_and_failed_are_distinguished() -> None:
    respx.post("https://a.example/tap/sync").respond(json=tap_json([["A1", 187.2779, 2.0524, 0.0]]))
    respx.post("https://b.example/tap/sync").respond(json=tap_json([]))
    respx.post("https://c.example/tap/sync").mock(return_value=fixture_response("vizier_unquoted_table"))
    reg = registry_of(tap_def("a", "https://a.example/tap/sync"), tap_def("b", "https://b.example/tap/sync"),
                      tap_def("c", "https://c.example/tap/sync"))
    record = await crossmatch_with(reg)

    results = record.catalog_results
    assert results["a"]["status"] == "success" and results["a"]["row_count"] == 1
    assert results["b"]["status"] == "empty" and results["b"]["row_count"] == 0
    assert results["c"]["status"] == "failed"
    assert results["c"]["error_type"] == "CatalogQueryError"
    assert "Encountered" in results["c"]["message"]  # archive's own error text surfaced
    assert record.catalogs_queried == 3
    assert [f["catalog"] for f in record.failures] == ["c"]
    assert record.failures[0]["elapsed_ms"] is not None
    stats = record.provenance["catalog_stats"]
    assert {k: v["status"] for k, v in stats.items()} == {"a": "success", "b": "empty", "c": "failed"}
    assert all(isinstance(v["elapsed_ms"], float) for v in stats.values())


@respx.mock
async def test_http200_query_status_error_is_failed_not_empty() -> None:
    respx.post("https://heasarc.example/tap/sync").mock(return_value=fixture_response("heasarc_bad_column"))
    record = await crossmatch_with(registry_of(tap_def("x", "https://heasarc.example/tap/sync", parameters={
        "columns": ["name", "ra", "dec"], "id_field": "name", "format": "votable"})))
    assert record.catalog_results["x"]["status"] == "failed"
    assert "hard_hs" in record.catalog_results["x"]["message"]


@respx.mock
async def test_cloudflare_html_block_is_failed() -> None:
    respx.post("https://exo.example/TAP/sync").mock(return_value=fixture_response("exoplanet_get_403"))
    record = await crossmatch_with(registry_of(tap_def("exo", "https://exo.example/TAP/sync")))
    result = record.catalog_results["exo"]
    assert result["status"] == "failed" and result["error_type"] == "CatalogQueryError"
    assert "403" in result["message"] and "Cloudflare" in result["message"]


@respx.mock
async def test_html_page_with_http200_is_failed() -> None:
    respx.post("https://h.example/tap").respond(200, html="<!DOCTYPE html><html><title>Maintenance</title></html>")
    record = await crossmatch_with(registry_of(tap_def("h", "https://h.example/tap")))
    assert record.catalog_results["h"]["status"] == "failed"
    assert "Maintenance" in record.catalog_results["h"]["message"]


@respx.mock
async def test_persistent_5xx_is_unavailable_after_three_attempts() -> None:
    route = respx.post("https://down.example/tap").respond(503, text="Service Unavailable")
    record = await crossmatch_with(registry_of(tap_def("down", "https://down.example/tap")))
    assert route.call_count == 3
    assert record.catalog_results["down"]["error_type"] == "CatalogUnavailableError"


@respx.mock
async def test_transient_5xx_then_success_recovers() -> None:
    route = respx.post("https://flaky.example/tap")
    route.side_effect = [httpx.Response(502), httpx.Response(200, json=tap_json([["F1", 187.2779, 2.0524, 0.0]]))]
    record = await crossmatch_with(registry_of(tap_def("flaky", "https://flaky.example/tap")))
    assert record.catalog_results["flaky"]["status"] == "success"


@respx.mock
async def test_network_error_is_unavailable() -> None:
    respx.post("https://net.example/tap").mock(side_effect=httpx.ConnectError("refused"))
    record = await crossmatch_with(registry_of(tap_def("net", "https://net.example/tap")))
    assert record.catalog_results["net"]["error_type"] == "CatalogUnavailableError"


@respx.mock
async def test_read_timeout_is_timeout_error() -> None:
    respx.post("https://slow.example/tap").mock(side_effect=httpx.ReadTimeout("slow"))
    record = await crossmatch_with(registry_of(tap_def("slow", "https://slow.example/tap")))
    assert record.catalog_results["slow"]["error_type"] == "QueryTimeoutError"


async def test_executor_enforces_per_catalog_timeout() -> None:
    class Sleepy(CatalogProvider):
        async def query(self, catalog, target, radius_arcsec):
            await asyncio.sleep(5)
            return []

    reg = registry_of(tap_def("z", "https://z.example", timeout_seconds=0.05))
    executor = QueryExecutor({"tap": Sleepy()}, timeout=30.0, registry=reg)
    plan = QueryPlan("z", "tap", None, {}, 5.0)
    successes, failures = await executor.execute([plan], validate_target(*TARGET))
    assert successes == [] and failures[0].error_type == "QueryTimeoutError"


@respx.mock
async def test_rows_without_positions_fail_loudly() -> None:
    respx.post("https://nopos.example/tap").respond(json={"metadata": [{"name": "id"}, {"name": "flux"}], "data": [["A", 1.0]]})
    record = await crossmatch_with(registry_of(tap_def("nopos", "https://nopos.example/tap")))
    assert record.catalog_results["nopos"]["status"] == "failed"
    assert record.catalog_results["nopos"]["error_type"] == "ResponseParseError"


@respx.mock
async def test_unnamed_array_rows_fail_loudly() -> None:
    respx.post("https://arr.example/tap").respond(json={"data": [[187.2779, 2.0524]]})
    record = await crossmatch_with(registry_of(tap_def("arr", "https://arr.example/tap")))
    assert record.catalog_results["arr"]["error_type"] == "ResponseParseError"


@respx.mock
async def test_fallback_provider_used_when_primary_unavailable() -> None:
    respx.post("https://irsa.example/TAP/sync").respond(503)
    ipac = (
        "\\fixlen = T\n"
        "| designation      | ra          | dec        | err_maj | err_min | jdate        | dist   |\n"
        "| char             | double      | double     | double  | double  | double       | double |\n"
        "|                  | deg         | deg        | arcsec  | arcsec  |              | arcsec |\n"
        "|                  |             |            |         |         |              |        |\n"
        " 12290669+0203085   187.277897    2.052387     0.17      0.08      2451599.8550   0.08\n"
    )
    gator = respx.get("https://irsa.example/gator").respond(200, text=ipac, headers={"content-type": "text/plain"})
    base = CatalogRegistry().get("twomass_psc")
    params = {**base.parameters, "fallback": {"provider": "irsa_gator", "endpoint": "https://irsa.example/gator", "catalog": "fp_psc"}}
    catalog = catalog_from_dict("twomass_psc", {**base.as_dict(), "endpoint": "https://irsa.example/TAP/sync", "parameters": params})
    record = await crossmatch_with(registry_of(catalog))
    result = record.catalog_results["twomass_psc"]
    assert gator.called
    assert result["status"] == "success"
    assert result["fallback"]["provider"] == "irsa_gator"
    src = result["sources"][0]
    assert src.source_id == "12290669+0203085"
    assert src.epoch == pytest.approx(2000.15, abs=1e-2)
    assert "selcols" in str(gator.calls.last.request.url)


@respx.mock
async def test_query_errors_are_not_cached_but_successes_are(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    route = respx.post("https://cache.example/tap")
    ok = httpx.Response(200, json=tap_json([["C1", 187.2779, 2.0524, 0.0]]))
    route.side_effect = [fixture_response("heasarc_bad_column"), ok, ok]
    catalog = tap_def("cache", "https://cache.example/tap")
    async with offline_client() as client:
        provider = TapProvider(client, cache=CacheManager(None))
        target = validate_target(*TARGET)
        with pytest.raises(CatalogQueryError):
            await provider.query(catalog, target, 10.0)
        first = await provider.query(catalog, target, 10.0)
        second = await provider.query(catalog, target, 10.0)
    assert route.call_count == 2
    assert first.meta["cached"] is False and second.meta["cached"] is True
    assert [s.source_id for s in second] == ["C1"]


async def test_plain_list_provider_is_still_supported() -> None:
    from models import CatalogSource

    class Legacy(CatalogProvider):
        async def query(self, catalog, target, radius_arcsec):
            return [CatalogSource(catalog.name, "L1", target.ra, target.dec, None, {}, {"wavelength": "optical"}, {})]

    reg = registry_of(tap_def("legacy", "https://legacy.example"))
    service = CrossmatchService(reg, {"tap": Legacy()})
    record = await service.crossmatch(*TARGET, radius_arcsec=5.0)
    assert record.catalog_results["legacy"]["status"] == "success"
    assert isinstance(record.catalog_results["legacy"]["elapsed_ms"], float)


def test_query_result_is_a_list() -> None:
    result = QueryResult([], {"status": "empty"})
    assert isinstance(result, list) and result.meta["status"] == "empty"


@pytest.mark.parametrize("exc", [CatalogQueryError, CatalogUnavailableError, QueryTimeoutError, ResponseParseError])
def test_error_classes_share_base(exc) -> None:
    from models import AstroSearchError

    assert issubclass(exc, AstroSearchError)
