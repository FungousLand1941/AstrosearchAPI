"""One regression test per KNOWN BUG found by live testing (M87 / 3C 273, 10 arcsec)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import respx
from fixture_io import FIXTURES, TARGETS, Exchange, load_exchanges, replay_side_effect
from helpers import offline_client, run_catalog

from models import (
    CatalogRegistry,
    catalog_from_dict,
    normalize_source_record,
    parse_json_records,
    validate_target,
)
from providers import CacheManager, HEASARCXaminProvider, SDSSProvider, TapProvider, haversine_arcsec

ROOT = Path(__file__).resolve().parents[1]


# Bug 1 -----------------------------------------------------------------------
@pytest.mark.parametrize("catalog", ["gaia_dr3", "simbad", "ned", "vizier_2mass_reference", "lotss"])
def test_bug1_tap_json_array_rows_are_not_dropped(catalog: str) -> None:
    rows = parse_json_records((FIXTURES / "3c273" / f"{catalog}.0.body").read_bytes())
    assert rows, f"{catalog}: TAP {{metadata,data}} rows were dropped"
    assert all(isinstance(r, dict) for r in rows)


async def test_bug1_gaia_has_3c273_at_milliarcsec() -> None:
    sources, failure = await run_catalog("gaia_dr3", "3c273")
    assert failure is None and sources[0].metadata["query_separation_arcsec"] < 0.01


# Bug 2 -----------------------------------------------------------------------
def test_bug2_mast_info_data_rows_are_not_dropped() -> None:
    rows = parse_json_records((FIXTURES / "3c273" / "panstarrs_dr2.0.body").read_bytes())
    assert len(rows) == 3 and "raMean" in rows[0]


# Bug 3 -----------------------------------------------------------------------
async def test_bug3_heasarc_uses_tap_cone_on_correct_tables(registry: CatalogRegistry) -> None:
    tap = TapProvider(offline_client(), cache=CacheManager(None))
    target = validate_target(*TARGETS["3c273"])
    for name, table in (("chandra", "csc"), ("xmm", "xmmssc"), ("rosat", "rass2rxs"), ("first", "first"), ("nvss", "nvss")):
        adql = tap.build_adql(registry.get(name), target, 10.0)
        assert f"FROM {table} WHERE" in adql
        assert "CIRCLE('ICRS', 187.277915400, 2.052388300," in adql
    for name in ("chandra", "xmm", "rosat", "first", "nvss"):
        sources, failure = await run_catalog(name, "3c273")
        assert failure is None and sources and sources[0].metadata["query_separation_arcsec"] < 6.0


async def test_bug3_xamin_legacy_provider_sends_position_and_finds_right_sky() -> None:
    meta_exchange = load_exchanges("errors", ["xamin_nvss_position"])[0]
    catalog = catalog_from_dict("nvss_xamin", {
        "provider": "heasarc_xamin", "wavelength": "radio", "table": "nvss", "epoch": 1995.0,
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query", "parameters": {"id_field": "name"},
    })
    exchange = Exchange("nvss_xamin", "GET", meta_exchange.url, "", meta_exchange.status_code,
                        meta_exchange.content_type, meta_exchange.content, [])
    with respx.mock(assert_all_called=True) as router:
        # Hand-recorded request (different number formatting): match on endpoint only.
        route = router.route().mock(side_effect=replay_side_effect([exchange], strict=False))
        async with offline_client() as client:
            result = await HEASARCXaminProvider(client, cache=CacheManager(None)).query(
                catalog, validate_target(*TARGETS["3c273"]), 10.0)
    sent = route.calls.last.request.url.params
    assert "coord" not in sent and sent["position"].startswith("187.2779")
    assert result[0].source_id == "NVSS J122906+020305"
    assert result[0].metadata["query_separation_arcsec"] < 10.0


# Bug 4 -----------------------------------------------------------------------
async def test_bug4_exoplanet_archive_post_avoids_waf_and_finds_planet(registry: CatalogRegistry) -> None:
    exchange = load_exchanges("hd209458", ["exoplanet_archive"])[0]
    assert exchange.method == "POST" and "?" not in exchange.url
    sources, failure = await run_catalog("exoplanet_archive", "hd209458")
    assert failure is None and sources[0].source_id == "HD 209458 b"
    # The recorded GET with '...)) = 1' in the URL really is blocked (HTML 403).
    blocked = load_exchanges("errors", ["exoplanet_get_403"])[0]
    assert blocked.status_code == 403 and "html" in blocked.content_type


# Bug 5 -----------------------------------------------------------------------
async def test_bug5_sdss_radius_semantics(registry: CatalogRegistry) -> None:
    # Live verification showed SkyServer applies the radius in ARCMIN (the 'degrees'
    # claim was a false alarm). SqlSearch makes the arcmin unit explicit.
    sql = SDSSProvider(offline_client(), cache=CacheManager(None)).build_sql(registry.get("sdss"), validate_target(*TARGETS["3c273"]), 10.0)
    assert "0.1666666667)" in sql
    sources, failure = await run_catalog("sdss", "3c273")
    assert failure is None
    assert all(s.metadata["query_separation_arcsec"] <= 10.0 for s in sources)
    # Legacy ConeSearch with sr = 10/60 (arcmin) returned only sources inside 10".
    exchange = load_exchanges("errors", ["sdss_conesearch_3c273"])[0]
    assert "sr=0.1666666667" in exchange.url
    rows = [line.split(",") for line in exchange.content.decode().splitlines()[2:] if line.strip()]
    seps = [haversine_arcsec(*TARGETS["3c273"], float(r[1]), float(r[2])) for r in rows]
    assert seps and max(seps) < 10.0


# Bug 6 -----------------------------------------------------------------------
async def test_bug6_nearest_object_not_truncated_in_crowded_field(registry: CatalogRegistry) -> None:
    adql = TapProvider(offline_client(), cache=CacheManager(None)).build_adql(registry.get("ned"), validate_target(*TARGETS["3c273"]), 10.0)
    assert "ORDER BY match_dist ASC" in adql and "TOP 201" in adql
    sources, _ = await run_catalog("ned", "3c273")
    assert sources[0].source_id == "3C 273"  # not one of the '[HB89] 1226+023 ABSnn' absorbers
    assert len(sources) == 21
    sources, _ = await run_catalog("simbad", "m87")
    assert sources[0].source_id == "M 87" and len(sources) > 20


# Bug 7 -----------------------------------------------------------------------
async def test_bug7_broken_query_is_failed_not_success() -> None:
    from helpers import make_service

    catalog = catalog_from_dict("broken", {
        "provider": "tap", "wavelength": "radio", "table": "csc", "epoch": 2000.0,
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",
        "parameters": {"columns": ["name", "ra", "dec", "hard_hs"], "id_field": "name", "format": "votable"},
    })
    reg = CatalogRegistry()
    reg._catalogs = {"broken": catalog}
    error = load_exchanges("errors", ["heasarc_bad_column"])[0]
    with respx.mock() as router:
        router.route().mock(side_effect=replay_side_effect([error], strict=False))
        async with offline_client() as client:
            record = await make_service(client, reg).crossmatch(*TARGETS["3c273"], radius_arcsec=10.0)
    assert record.catalog_results["broken"]["status"] == "failed"
    assert record.failures and record.failures[0]["error_type"] == "CatalogQueryError"


# Bug 8 -----------------------------------------------------------------------
async def test_bug8_positional_errors_are_one_sigma_arcsec() -> None:
    gaia, _ = await run_catalog("gaia_dr3", "m87")
    nucleus = gaia[0]
    assert nucleus.data["ra_error"] > 0.1  # mas (0.30 mas for the M87 nucleus)
    assert nucleus.positional_error_arcsec < 0.001  # arcsec, not 0.3 "arcsec"
    ned, _ = await run_catalog("ned", "3c273")
    assert ned[0].positional_error_arcsec < ned[0].data["uncmaja"]  # 95% -> 1-sigma
    simbad, _ = await run_catalog("simbad", "3c273")
    assert simbad[0].metadata["positional_error"]["columns"][0] == "coo_err_maj"
    assert "position_uncertainty_arcsec" not in normalize_source_record({"ra": 1, "dec": 2, "ra_error": 0.3})


# Bug 9 -----------------------------------------------------------------------
def test_bug9_pytest_suite_configured_with_live_marker() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert config["testpaths"] == ["tests"]
    assert config["asyncio_mode"] == "auto"
    assert "not live" in config["addopts"]
    assert any(m.startswith("live:") for m in config["markers"])
    assert (ROOT / "tests" / "test_live_canary.py").exists()
