"""Live tests for vizier.py against the real VizieR, TAPVizieR and IVOA RegTAP services.

Run with: .venv/Scripts/python.exe -m pytest -m live tests/test_vizier_live.py

Tests skip only when a service is unreachable (network error, timeout, HTTP 5xx); a service
that answers wrongly fails the test. Truth values: SIMBAD positions/redshifts (J2000, ICRS)
and the published catalogues (2SXPS: Evans et al. 2020, ApJS 247, 54; VLASS CIRADA
components: Gordon et al. 2021, ApJS 255, 30; 2dFGRS: Colless et al. 2001, MNRAS 328, 1039).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable
from typing import Any

import httpx
import pytest

import vizier
from crossmatch import AdvancedQuery, CrossmatchService
from models import DEFAULT_CATALOGS, K90_2D, validate_target
from providers import CacheManager, provider_map

pytestmark = pytest.mark.live

TRANSIENT_FAILURES = {"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError"}

# SIMBAD basic.ra/dec (ICRS J2000) and rvz_redshift.
TARGET_3C273 = (187.27791594049, 2.05238823055)
TARGET_MRK421 = (166.11380868146, 38.20883291552)
TARGET_NGC3818 = (175.4889849546, -6.155683098159999)
NGC3818_SIMBAD_Z = 0.00554


async def live[T](awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except vizier.VizierUpstreamError as exc:
        if exc.transient:
            pytest.skip(f"service unavailable: {exc}")
        raise
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        pytest.skip(f"network error: {exc!r}")


async def test_live_search_gaia_dr3_finds_i_355_gaiadr3():
    result = await live(vizier.search_catalogs("Gaia DR3"))
    ids = [t.table_id for t in result.tables]
    assert "I/355/gaiadr3" in ids[:3], ids[:10]
    hit = next(t for t in result.tables if t.table_id == "I/355/gaiadr3")
    assert hit.nrows == 1_811_709_771  # Gaia DR3 gaia_source (Gaia Collaboration, Vallenari et al. 2023)
    assert "optical" in hit.wavelengths


async def test_live_search_filters_by_wavelength_and_ucd():
    result = await live(vizier.search_catalogs("galaxy redshift", ucd="src.redshift", wavelength="optical",
                                               max_catalogs=10))
    assert result.tables
    assert all("optical" in t.wavelengths and t.matching_columns for t in result.tables)
    radio = await live(vizier.search_catalogs("VLASS", wavelength="radio"))
    assert "J/ApJS/255/30/comp" in [t.table_id for t in radio.tables]


async def test_live_ivoa_registry_lists_vizier_vlass_cone_search():
    resources = await live(vizier.search_ivoa_registry("VLASS", wavelength="radio"))
    by_id = {r.ivoid: r for r in resources}
    assert "ivo://cds.vizier/j/apjs/255/30" in by_id
    vlass = by_id["ivo://cds.vizier/j/apjs/255/30"]
    assert vlass.vizier_catalog == "J/ApJS/255/30" and "radio" in vlass.wavebands
    assert any("J/ApJS/255/30" in url for url in vlass.services.get("conesearch", []))


async def test_live_describe_identifies_columns():
    xray = await live(vizier.describe_table("IX/58/2sxps"))
    assert (xray.ra_column, xray.dec_column, xray.id_column) == ("RAJ2000", "DEJ2000", "2SXPS")
    assert xray.pos_error["kind"] == "radius90" and xray.catalog.bibcode == "2020ApJS..247...54E"
    gaia = await live(vizier.describe_table("I/355/gaiadr3"))
    assert gaia.epoch == 2016.0 and gaia.field_map["pmra"] == "pmRA"
    redshift = await live(vizier.describe_table("VII/250/2dfgrs"))
    assert redshift.field_map == {"redshift": "z"}
    assert "confirming decimal degrees" in (redshift.position_unit_check or "")


async def _register_and_crossmatch(tmp_path, table: str, target: tuple[float, float], radius: float):
    path = tmp_path / "catalogs.yaml"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        registration = await live(vizier.register_table(table, path=path, client=client))
        registry = vizier.load_registry(path)
        assert set(DEFAULT_CATALOGS) <= set(registry.catalogs)
        service = CrossmatchService(registry, provider_map(client, timeout=120.0, cache=CacheManager(None)))
        ra, dec = target
        query = AdvancedQuery(target=validate_target(ra, dec), radius_arcsec=radius, catalogs=[registration.name],
                              min_confidence=0.0)
        record = await live(service.crossmatch(ra, dec, query=query))
    result: dict[str, Any] = record.catalog_results[registration.name]
    if result["status"] == "failed" and result.get("error_type") in TRANSIENT_FAILURES:
        pytest.skip(f"VizieR TAP unavailable: {result.get('message')}")
    assert result["status"] == "success", result
    return registration, result, record


async def test_live_register_xray_2sxps_and_crossmatch_3c273(tmp_path):
    registration, result, record = await _register_and_crossmatch(tmp_path, "IX/58/2sxps", TARGET_3C273, 5.0)
    assert registration.entry["wavelength"] == "xray"
    source = result["sources"][0]
    assert source.source_id == "13814"
    assert source.data["IAUName"] == "2SXPS J122906.6+020308"
    assert source.ra == pytest.approx(TARGET_3C273[0], abs=2e-4) and source.dec == pytest.approx(TARGET_3C273[1], abs=2e-4)
    # 90% Rayleigh radius -> 1-sigma per axis: Err90 / sqrt(-2 ln 0.1).
    assert source.positional_error_arcsec == pytest.approx(source.data["Err90"] / K90_2D, rel=1e-9)
    assert 0.2 < source.positional_error_arcsec < 1.0
    assert record.provenance["matches"][0]["separation_arcsec"] < 1.0


async def test_live_register_radio_vlass_and_crossmatch_mrk421(tmp_path):
    registration, result, record = await _register_and_crossmatch(tmp_path, "J/ApJS/255/30/comp", TARGET_MRK421, 3.0)
    assert registration.entry["wavelength"] == "radio"
    source = result["sources"][0]
    assert source.source_id == "VLASS1QLCIR J110427.33+381231.8"
    assert 300.0 < source.data["Ftot"] < 600.0  # Mrk 421: ~0.45 Jy at 3 GHz in VLASS epoch 1
    era, ede = source.data["e_RAJ2000"], source.data["e_DEJ2000"]  # 1-sigma, degrees
    expected = math.sqrt(((era * 3600.0) ** 2 + (ede * 3600.0) ** 2) / 2.0)
    assert source.positional_error_arcsec == pytest.approx(expected, rel=1e-9)
    assert record.provenance["matches"][0]["separation_arcsec"] < 0.5


async def test_live_register_redshift_2dfgrs_and_crossmatch_ngc3818(tmp_path):
    registration, result, record = await _register_and_crossmatch(tmp_path, "VII/250/2dfgrs", TARGET_NGC3818, 5.0)
    assert registration.entry["parameters"]["field_map"] == {"redshift": "z"}
    source = result["sources"][0]
    assert source.source_id == "TGN113Z100"
    redshift = source.metadata["physical"]["redshift"]
    assert redshift == source.data["z"]
    assert abs(redshift - NGC3818_SIMBAD_Z) < 0.001  # 2dFGRS vs SIMBAD redshift of NGC 3818
    assert record.provenance["matches"][0]["separation_arcsec"] < 3.0
