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
from models import DEFAULT_CATALOGS, K90_2D, K95_1D, CatalogRegistry, compute_positional_error, validate_target
from providers import CacheManager, provider_map

pytestmark = pytest.mark.live

TRANSIENT_FAILURES = {"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError"}

# SIMBAD basic.ra/dec (ICRS J2000) and rvz_redshift.
TARGET_3C273 = (187.27791594049, 2.05238823055)
TARGET_MRK421 = (166.11380868146, 38.20883291552)
TARGET_NGC3818 = (175.4889849546, -6.155683098159999)
NGC3818_SIMBAD_Z = 0.00554
# SIMBAD basic (ICRS J2000 position; pmra*cos(dec), pmdec in mas/yr), retrieved 2026-09-28.
BARNARD = (269.4520769586187, 4.693364966576667)
BARNARD_PM = (-801.551, 10362.394)
CM_DRA = (248.5847094419055, 57.16232469963777)
CM_DRA_PM = (-1113.797, 1180.977)


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


async def _register_and_crossmatch(tmp_path, table: str, target: tuple[float, float], radius: float,
                                   *, epoch: float | None = None, pm: tuple[float, float] | None = None):
    path = tmp_path / "catalogs.yaml"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        registration = await live(vizier.register_table(table, path=path, client=client))
        registry = vizier.load_registry(path)
        assert set(DEFAULT_CATALOGS) <= set(registry.catalogs)
        assert set(CatalogRegistry(path).catalogs) == set(registry.catalogs)  # the core reads it completely
        service = CrossmatchService(registry, provider_map(client, timeout=120.0, cache=CacheManager(None)))
        ra, dec = target
        query_target = validate_target(ra, dec, epoch=epoch, pm_ra_masyr=pm[0] if pm else None,
                                       pm_dec_masyr=pm[1] if pm else None)
        query = AdvancedQuery(target=query_target, radius_arcsec=radius, catalogs=[registration.name],
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
    statistical = math.sqrt(((era * 3600.0) ** 2 + (ede * 3600.0) ** 2) / 2.0)
    # VLASS Quick-Look astrometry: the core's 0.5" systematic is added to the fit error.
    assert registration.entry["pos_error"]["systematic_arcsec"] == 0.5
    assert source.positional_error_arcsec == pytest.approx(math.hypot(statistical, 0.5), rel=1e-9)
    match = record.provenance["matches"][0]
    assert match["separation_arcsec"] < 0.5
    assert match["confidence"] > 0.9  # the true counterpart keeps a high posterior


async def test_live_register_redshift_2dfgrs_and_crossmatch_ngc3818(tmp_path):
    registration, result, record = await _register_and_crossmatch(tmp_path, "VII/250/2dfgrs", TARGET_NGC3818, 5.0)
    assert registration.entry["parameters"]["field_map"] == {"redshift": "z"}
    source = result["sources"][0]
    assert source.source_id == "TGN113Z100"
    redshift = source.metadata["physical"]["redshift"]
    assert redshift == source.data["z"]
    assert abs(redshift - NGC3818_SIMBAD_Z) < 0.001  # 2dFGRS vs SIMBAD redshift of NGC 3818
    assert record.provenance["matches"][0]["separation_arcsec"] < 3.0


async def test_live_epoch_regressions_barnards_star_in_ppmxl_and_tycho2(tmp_path):
    # PPMXL RAJ2000 is at J2000 (COOSYS 2000.000), not at the per-row mean epoch epRA.
    registration, result, record = await _register_and_crossmatch(
        tmp_path / "ppmxl", "I/317/sample", BARNARD, 10.0, epoch=2000.0, pm=BARNARD_PM)
    assert registration.entry["epoch"] == 2000.0
    best = max(record.provenance["matches"], key=lambda m: m["confidence"])
    assert best["separation_arcsec"] < 0.3 and best["confidence"] > 0.9
    # Tycho-2's observed RA(ICRS) is at 1990 + EpRA-1990 (~1991.76 for Barnard's star).
    registration, result, record = await _register_and_crossmatch(
        tmp_path / "tycho2", "I/259/tyc2", BARNARD, 10.0, epoch=2000.0, pm=BARNARD_PM)
    source = next(s for s in result["sources"] if s.source_id == "425-2502-1")  # TYC 425-2502-1
    assert source.epoch == pytest.approx(1991.76, abs=0.05)
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "425-2502-1")
    assert match["separation_arcsec"] < 1.0 and match["confidence"] > 0.5


async def test_live_epoch_regression_cm_dra_in_vsx(tmp_path):
    # VSX 'Epoch' is the epoch of maximum/minimum light (HJD), not the epoch of the position.
    registration, _result, record = await _register_and_crossmatch(
        tmp_path, "B/vsx/vsx", CM_DRA, 5.0, epoch=2000.0, pm=CM_DRA_PM)
    assert registration.entry["epoch"] is None
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "CM Dra")
    assert match["separation_arcsec"] < 1.0 and match["confidence"] > 0.5


async def test_live_describe_regressions():
    lower = await live(vizier.describe_table("ix/58/2sxps"))
    assert lower.table_id == "IX/58/2sxps"
    eco = await live(vizier.describe_table("J/ApJ/956/51/table4"))
    assert "redshift" not in eco.field_map  # cz in km/s is a velocity
    cxogbs = await live(vizier.describe_table("J/ApJS/210/18/cxogbs"))
    assert cxogbs.pos_error.get("divisor") == 3.0  # 'Error in RAdeg (3{sigma})'
    tycho = await live(vizier.describe_table("I/259/tyc2"))
    assert tycho.id_column == "TYC1-TYC2-TYC3"
    allwise = await live(vizier.describe_table("II/328/allwise"))
    assert allwise.pos_error["columns"] == ["eeMaj", "eeMin", "eePA"]
    csc = await live(vizier.describe_table("IX/70/csc21mas"))
    assert csc.pos_error["kind"] == "ellipse95axis"
    assert compute_positional_error({"r0": 1.96, "r1": 1.96, "PA": 0.0}, csc.pos_error)[0] == pytest.approx(1.96 / K95_1D)
    nvss = await live(vizier.describe_table("VIII/65/nvss"))
    assert nvss.catalog.bibcode == "1998AJ....115.1693C" and nvss.citation.startswith("Condon et al. 1998")
    stars = await live(vizier.describe_table("J/A+A/657/A4/stars"))
    assert stars.ra_column == "RAJ2000"  # not 'RAS' (seconds of time)
    with pytest.raises(vizier.VizierInputError, match="journal-level"):
        await vizier.describe("J/A+A")


async def test_live_ucd_search_is_case_insensitive():
    spectral = await live(vizier.search_catalogs("Hipparcos", ucd="src.spType"))
    assert "I/239/hip_main" in [t.table_id for t in spectral.tables]
    ellipses = await live(vizier.search_catalogs("Fermi LAT", ucd="pos.errorellipse"))
    assert {"IX/67", "J/ApJS/247/33"} <= {c.catalog_id for c in ellipses.catalogs}


# ---------------------------------------------------------------------------
# Second review: dotted ICRS columns, per-row epoch spans, one-sided errors, wavebands
# ---------------------------------------------------------------------------

# SIMBAD basic (ICRS J2000; pm in mas/yr), retrieved 2026-09-29.
TARGET_COMA = (194.93501999999998, 27.91246)  # 'ACO 1656'; the Abell catalogue centre is 4.2' away
WD2117 = (319.73443171497, 54.21145665108)
WD2117_PM = (-85.45, 193.193)
PROXIMA = (217.42894222160578, -62.67949018907555)
PROXIMA_PM = (-3781.741, 769.465)
ERASS_STARS = {  # SIMBAD position/pm -> the 1eRASS counterpart
    "AB Dor": ((82.18696340245, -65.44866642978), (37.554, 158.574), "1eRASS J052844.6-652650"),
    "HD 102077": ((176.16008757069, -49.417334473779995), (-109.572, -51.325), "1eRASS J114437.9-492503"),
    "YY Gem": ((113.65603096981002, 31.86949434485999), (-201.406, -97.0), "1eRASS J073437.2+315207"),
}


async def test_live_dotted_icrs_columns_give_icrs_positions(tmp_path):
    # Abell: the main position is B1950 (194.35, +28.25); the ICRS one is VizieR's '_RA.icrs'.
    registration, result, record = await _register_and_crossmatch(tmp_path / "abell", "VII/110A/table3",
                                                                  TARGET_COMA, 600.0)
    assert registration.entry["parameters"]["ra_field"] == '"_RA.icrs"'
    source = next(s for s in result["sources"] if s.source_id == "1656")
    assert source.ra == pytest.approx(194.95305, abs=1e-4) and source.dec == pytest.approx(27.98070, abs=1e-4)
    # McCook & Sion white dwarfs: '_RA.icrs' is the J2000 ICRS position (B1950 is 1130" away).
    _registration, result, record = await _register_and_crossmatch(tmp_path / "wd", "B/wd/catalog", WD2117, 30.0,
                                                                  epoch=2000.0, pm=WD2117_PM)
    source = next(s for s in result["sources"] if s.source_id == "2117+539")
    assert source.ra == pytest.approx(319.72998, abs=1e-4) and source.dec == pytest.approx(54.21340, abs=1e-4)
    assert next(m for m in record.provenance["matches"] if m["source_id"] == "2117+539")["separation_arcsec"] < 15.0


async def test_live_per_row_epoch_catalogues_are_propagated(tmp_path):
    registration, _result, record = await _register_and_crossmatch(tmp_path / "tm", "II/246/out", BARNARD, 3.0,
                                                                    epoch=2000.0, pm=BARNARD_PM)
    assert registration.entry["parameters"]["epoch_range"] == [1997.4, 2001.2]
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "17574849+0441405")
    assert match["separation_arcsec"] < 0.5 and match["confidence"] > 0.99
    registration, _result, record = await _register_and_crossmatch(tmp_path / "er", "J/A+A/682/A34/erass1-m", PROXIMA,
                                                                  30.0, epoch=2000.0, pm=PROXIMA_PM)
    lo, hi = registration.entry["parameters"]["epoch_range"]
    assert 2019.9 < lo < hi < 2020.5  # eRASS1: December 2019 - June 2020
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "1eRASS J142932.3-624030")
    assert match["confidence"] > 0.9


async def test_live_erass1_counterparts_use_the_calibrated_positional_error(tmp_path):
    for i, (name, (position, pm, erass_id)) in enumerate(ERASS_STARS.items()):
        registration, result, record = await _register_and_crossmatch(
            tmp_path / str(i), "J/A+A/682/A34/erass1-m", position, 30.0, epoch=2000.0, pm=pm)
        assert registration.entry["pos_error"] == {"columns": ["posErr"], "units": ["arcsec"], "kind": "sigma"}
        source = next(s for s in result["sources"] if s.source_id == erass_id)
        assert source.positional_error_arcsec == pytest.approx(source.data["posErr"], rel=1e-9)
        match = next(m for m in record.provenance["matches"] if m["source_id"] == erass_id)
        assert match["confidence"] > 0.9, (name, match)


async def test_live_second_review_describe_regressions():
    wavelengths = {"II/246/out": "infrared", "VIII/92/first14": "radio", "IX/70/csc21mas": "xray"}
    for table, expected in wavelengths.items():
        desc = await live(vizier.describe_table(table))
        assert desc.wavelength == expected, (table, desc.wavelength)
    hip2 = await live(vizier.describe_table("I/311/hip2"))
    assert hip2.pos_error == {"columns": ["e_RArad", "e_DErad"], "units": ["mas", "mas"], "kind": "sigma"}
    bd = await live(vizier.describe_table("I/122/bd"))
    assert bd.ra_column == "_RA.icrs" and bd.registrable
    ngc = await live(vizier.describe_table("VII/118/ngc2000"))
    assert not ngc.registrable and "equinox B2000" in ngc.problems[0]
    fermi = await live(vizier.describe_table("J/ApJS/247/33/4fgl"))
    assert "spectral_type" not in fermi.field_map
    xmm = await live(vizier.describe_table("IX/69/xmm4d13s"))
    assert xmm.pos_error["kind"] == "radial"
    urat = await live(vizier.describe_table("I/329/urat1"))
    lo, hi = urat.epoch_range
    assert lo < 2012.4 and hi > 2014.9  # URAT1 observed 2012.3-2015.0 (sampled estimate, widened)


# ---------------------------------------------------------------------------
# Third review: position-angle errors, J2000 main positions, counterpart columns, renamed
# Gaia fields, per-row observation epochs, overrides
# ---------------------------------------------------------------------------

# SIMBAD basic (ICRS J2000), retrieved 2026-09-29.
HZ43 = (199.09105517071666, 29.098725329688055)
D02_PUP = (114.93256487363001, -38.1392981022)
HD37501 = (83.73930316285, -61.17604870992)


async def test_live_third_review_describe_regressions():
    galex = await live(vizier.describe_table("II/335/galex_ais"))
    assert galex.pos_error["columns"] == ["Nerr"]  # not e_nPA (position-angle error, deg)
    for table in ("II/371/des_dr2", "II/357/des_dr1"):
        des = await live(vizier.describe_table(table))
        assert des.pos_error == {}, (table, des.pos_error)
    rosat = await live(vizier.describe_table("IX/10A/1rxs"))
    assert rosat.pos_error["kind"] == "sigma"  # 'Total positional error', as the core rosat_bsc entry
    rxs = await live(vizier.describe_table("J/A+A/588/A103/cat2rxs"))
    assert not {"redshift", "object_type"} & set(rxs.field_map)  # zVV10/TypeVV10 belong to the VV10 counterpart
    sdss = await live(vizier.describe_table("V/147/sdss12"))
    assert "spectral_type" not in sdss.field_map and sdss.field_map.get("object_type") != "class"
    for table, epoch in (("I/345/gaia2", 2015.5), ("I/337/tgas", 2015.0), ("I/337/gaia", 2015.0)):
        gaia = await live(vizier.describe_table(table))
        assert gaia.registrable and gaia.epoch == epoch, (table, gaia.problems, gaia.epoch)
    smss = await live(vizier.describe_table("II/379/smssdr4"))
    assert smss.epoch == "EpMean" and smss.epoch_format == "mjd" and smss.epoch_range


async def test_live_third_review_crossmatches(tmp_path):
    _registration, result, record = await _register_and_crossmatch(tmp_path / "galex", "II/335/galex_ais", HZ43, 10.0)
    source = result["sources"][0]
    assert source.positional_error_arcsec == pytest.approx(source.data["Nerr"]) and source.positional_error_arcsec < 3.5
    assert record.provenance["matches"][0]["confidence"] > 0.3
    registration, result, record = await _register_and_crossmatch(tmp_path / "ngc2451", "J/AJ/122/1486/ccd",
                                                                  D02_PUP, 5.0)
    assert registration.entry["parameters"]["ra_field"] == '"RA1996"'
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "217300242")
    assert match["separation_arcsec"] < 0.5
    _registration, result, _record = await _register_and_crossmatch(tmp_path / "rxs", "J/A+A/588/A103/cat2rxs",
                                                                    HD37501, 20.0)
    source = next(s for s in result["sources"] if s.source_id == "2RXS J053457.2-611023")
    assert "redshift" not in source.metadata["physical"]
    _registration, result, record = await _register_and_crossmatch(tmp_path / "smss", "II/379/smssdr4", BARNARD, 5.0,
                                                                   epoch=2000.0, pm=BARNARD_PM)
    match = next(m for m in record.provenance["matches"] if m["source_id"] == "1353376183")
    assert match["separation_arcsec"] < 1.5  # Barnard's star at its 2018.6 SkyMapper position


async def test_live_epoch_override_gets_a_span(tmp_path):
    registration = await live(vizier.register_table("J/A+A/682/A34/erass1-m", path=tmp_path / "c.yaml",
                                                    overrides={"epoch": "b_MJD"}))
    lo, hi = registration.entry["parameters"]["epoch_range"]
    assert registration.entry["epoch_format"] == "mjd" and 2019.9 < lo < hi < 2020.5
