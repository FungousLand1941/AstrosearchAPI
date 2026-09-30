"""Offline tests for sed.py: conversions vs hand-computed values, SVO parsing/caching, member selection,
classification, redshift choice, full pipeline replayed from recorded archive responses, router and CLI.

Recorded fixtures (tests/fixtures/sed/) hold the supplementary requests sed.py makes on top of the
crossmatch recordings in tests/fixtures/<target>/: SVO FPS filter VOTables, Gaia DR3 per-source
astrometric errors / DSC probabilities, and SDSS SkyServer SpecObj/Photoz queries. Re-record with

    .venv/Scripts/python.exe tests/test_sed.py record
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import FIXTURES, TARGETS, load_exchanges, redact, replay_side_effect
from helpers import make_service, offline_client

import sed
from models import UnifiedRecord

SED_FIXTURES = FIXTURES / "sed"
SDSS_QSO = (180.01108, 33.171292)  # SDSS SpecObj 11568697826042206208: QSO, z = 1.008554, zWarning = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def exchanges_for(target: str) -> list:
    """Crossmatch recordings for ``target`` plus sed.py's supplementary recordings and SVO filters."""
    return load_exchanges(target) + load_exchanges(f"sed/{target}") + load_exchanges("sed/svo")


async def replay_record(target: str) -> dict[str, Any]:
    """The crossmatch UnifiedRecord for a canary target, replayed from the recorded catalog responses."""
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(target)))
        async with offline_client() as client:
            record = await make_service(client).crossmatch(*TARGETS[target], radius_arcsec=10.0)
    return record.as_dict()


async def replay_sed(target: str, tmp_path: Path) -> dict[str, Any]:
    record = await replay_record(target)
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges_for(target)))
        async with offline_client() as client:
            return await sed.sed_from_record(record, client=client, filters=filters)


def point(result: dict[str, Any], facility_prefix: str, band: str) -> dict[str, Any]:
    matches = [p for p in result["points"] if p["facility"].startswith(facility_prefix) and p["band"] == band]
    assert len(matches) == 1, f"{facility_prefix} {band}: {matches}"
    return matches[0]


@pytest.fixture(scope="module")
def sed_3c273(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return asyncio.run(replay_sed("3c273", tmp_path_factory.mktemp("sed3c273")))


@pytest.fixture(scope="module")
def sed_m87(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return asyncio.run(replay_sed("m87", tmp_path_factory.mktemp("sedm87")))


# ---------------------------------------------------------------------------
# Unit conversions (hand-computed values)
# ---------------------------------------------------------------------------


def test_ab_zero_magnitude_is_3631_jy() -> None:
    flux, err = sed.ab_mag_to_jy(0.0, 0.1)
    assert flux == pytest.approx(3631.0, rel=1e-12)
    # sigma_F = F * 0.1 * ln(10) / 2.5 = 3631 * 0.0921034 = 334.4275
    assert err == pytest.approx(334.4275, rel=1e-5)
    # 23.9 AB = 1 microjansky: 3631 * 10^-9.56 = 1.0000e-6 Jy
    assert sed.ab_mag_to_jy(23.9)[0] == pytest.approx(1.0e-6, rel=1e-3)
    assert sed.jy_to_ab_mag(3631.0) == pytest.approx(0.0, abs=1e-12)


def test_2mass_ks_vega_zero_point_from_svo_fixture() -> None:
    meta = json.loads((SED_FIXTURES / "svo" / "2MASS_2MASS.Ks.json").read_text())
    body = (SED_FIXTURES / "svo" / "2MASS_2MASS.Ks.0.body").read_bytes()
    assert "ID=2MASS%2F2MASS.Ks" in meta["exchanges"][0]["url"]
    params = sed.parse_svo_votable(body)
    assert params["PhotCalID"] == "2MASS/2MASS.Ks/Vega"
    assert params["MagSys"] == "Vega"
    assert params["ZeroPoint"] == pytest.approx(666.8)  # Cohen et al. 2003
    info = sed.build_filter_info("2MASS/2MASS.Ks", params, "svo")
    assert info.wavelength_eff_um == pytest.approx(2.159)
    # Ks = 10.0 Vega -> 666.8 * 1e-4 = 0.06668 Jy
    assert sed.vega_mag_to_jy(10.0, info.zero_point_jy)[0] == pytest.approx(0.06668, rel=1e-12)


def test_embedded_table_matches_recorded_svo_answers() -> None:
    """The fallback table must be a verbatim copy of what SVO returns (every filter)."""
    checked = 0
    for fid, embedded in sed.EMBEDDED_FILTERS.items():
        body = (SED_FIXTURES / "svo" / f"{fid.replace('/', '_')}.0.body").read_bytes()
        live = sed.parse_svo_votable(body)
        assert live["PhotCalID"] == embedded["PhotCalID"]
        for key in ("ZeroPoint", "WavelengthEff", "WavelengthMean", "WavelengthPivot", "WavelengthMin", "WavelengthMax"):
            assert float(live[key]) == pytest.approx(embedded[key], rel=1e-12), (fid, key)
        checked += 1
    assert checked == len(sed.FILTER_SYSTEMS) == 23


def test_ab_filters_use_definition_not_svo_ab_zero_point() -> None:
    info = sed.FilterCatalog(offline=True, cache_path=Path("nonexistent-dir/x.json")).info("SLOAN/SDSS.u")
    assert info.mag_system == "AB"
    assert info.zero_point_jy == 3631.0
    assert info.svo_vega_zero_point_jy == pytest.approx(1582.537065543)
    assert info.metadata_origin == "embedded"


def test_sdss_asinh_magnitudes() -> None:
    # Bright end: asinh -> Pogson. r = 15 -> 3631 * 1e-6 Jy.
    flux, _ = sed.sdss_asinh_mag_to_jy(15.0, "r")
    assert flux == pytest.approx(3631e-6, rel=1e-6)
    # u_AB = u_SDSS - 0.04: u = 15 -> 3631 * 10^(-0.4 * 14.96)
    assert sed.sdss_asinh_mag_to_jy(15.0, "u")[0] == pytest.approx(3631 * 10 ** (-0.4 * 14.96), rel=1e-6)
    # z_AB = z_SDSS + 0.02
    assert sed.sdss_asinh_mag_to_jy(15.0, "z")[0] == pytest.approx(3631 * 10 ** (-0.4 * 15.02), rel=1e-5)
    # Zero flux at m = -(2.5/ln10) ln b: SDSS table u 24.63, g 25.11, r 24.80, i 24.36, z 22.83.
    for band, m0 in {"u": 24.63, "g": 25.11, "r": 24.80, "i": 24.36, "z": 22.83}.items():
        exact = -sed.POGSON * math.log(sed.SDSS_ASINH_B[band])
        assert exact == pytest.approx(m0, abs=0.006)
        assert abs(sed.sdss_asinh_mag_to_jy(exact, band)[0]) < 1e-15
    # Negative flux beyond the zero-flux magnitude (asinh magnitudes are defined there).
    assert sed.sdss_asinh_mag_to_jy(26.0, "r")[0] < 0
    # Error: sigma_x = sigma_m (ln10/2.5) sqrt(x^2 + 4 b^2); bright end -> Pogson.
    _, err = sed.sdss_asinh_mag_to_jy(15.0, "r", 0.01)
    assert err == pytest.approx(3631e-6 * 0.01 / sed.POGSON, rel=1e-5)


def test_frequency_and_nu_fnu() -> None:
    assert sed.wavelength_um_to_hz(1.0) == pytest.approx(2.99792458e14, rel=1e-12)
    assert sed.hz_to_wavelength_um(1.4e9) == pytest.approx(214137.47, rel=1e-6)
    # FIRST 3C 273: 36983.34 mJy at 1.4 GHz -> nuFnu = 1.4e9 * 36.98334 * 1e-23
    assert sed.nu_fnu_cgs(1.4e9, 36.98334) == pytest.approx(5.177668e-13, rel=1e-6)


def test_xray_power_law_conversion() -> None:
    flux_jy, freq = sed.xray_band_flux_to_jy(1e-11, 0.5, 7.0)
    e0 = math.sqrt(3.5)
    assert freq == pytest.approx(e0 / 4.135667696e-18, rel=1e-8)
    # F_E(E0) = F / (E0 ln 14) erg/s/cm2/keV; F_nu = F_E * h[keV s]
    f_e = 1e-11 / (e0 * math.log(14.0))
    assert flux_jy == pytest.approx(f_e * 4.135667696e-18 / 1e-23, rel=1e-8)
    assert flux_jy == pytest.approx(8.3763e-7, rel=1e-4)
    # For Gamma = 2, nu F_nu = F / ln(E2/E1) at any energy.
    assert sed.nu_fnu_cgs(freq, flux_jy) == pytest.approx(1e-11 / math.log(14.0), rel=1e-8)
    # Gamma = 1.7: integrating the implied F_E over the band recovers the band flux.
    fj, fr = sed.xray_band_flux_to_jy(1e-11, 0.5, 7.0, photon_index=1.7)
    f_e0 = fj * 1e-23 / 4.135667696e-18
    e0k = fr * 4.135667696e-18
    grid = [0.5 * (14.0 ** (i / 4000)) for i in range(4001)]
    integral = sum(0.5 * (f_e0 * (a / e0k) ** -0.7 + f_e0 * (b / e0k) ** -0.7) * (b - a) for a, b in itertools.pairwise(grid))
    assert integral == pytest.approx(1e-11, rel=1e-5)
    assert sed.xray_band_scale(0.5, 7.0, 0.3, 3.5) == pytest.approx(math.log(3.5 / 0.3) / math.log(14.0))
    with pytest.raises(ValueError):
        sed.xray_band_flux_to_jy(1e-11, 7.0, 0.5)


def test_rosat_count_rate_uses_webpimms_unabsorbed_ecf() -> None:
    # WebPIMMS v4.15a, ROSAT/PSPC, Gamma = 2, N_H = 3e20: 1 ct/s -> 1.135e-11 absorbed, 1.877e-11 unabsorbed
    # (0.1-2.4 keV). The Gamma = 2 power law evaluated at E0 must be normalised by the UNABSORBED flux.
    assert sed.ROSAT_PSPC_ECF == 1.877e-11 and sed.ROSAT_PSPC_ECF_ABSORBED == 1.135e-11
    spec = sed.XRAY_SPECS["rosat"][1][0]
    p = sed.xray_point(spec, "ROSAT PSPC", "rosat", "x", {"count_rate": 7.8635, "count_rate_error": 0.1474})
    band_flux = 7.8635 * 1.877e-11
    assert p.nu_fnu_erg_s_cm2 == pytest.approx(band_flux / math.log(24.0), rel=1e-9)
    assert any("unabsorbed" in n and "N_H=3e20" in n for n in p.notes)
    assert p.flux_err_jy / p.flux_jy == pytest.approx(0.1474 / 7.8635, rel=1e-9)
    assert p.regime == "xray"
    assert any("Gamma=2" in n for n in p.notes)


def test_radio_units() -> None:
    first = sed.radio_point(sed.RADIO_SPECS["first"][1][0], "FIRST (VLA)", "first", "x",
                            {"int_flux_20_cm": 36983.34, "flux_20_cm_error": 6.196})
    assert first.flux_jy == pytest.approx(36.98334)
    assert first.flux_err_jy == pytest.approx(0.006196)
    vlass = sed.radio_point(sed.RADIO_SPECS["vlass"][1][0], "VLASS (VLA)", "vlass", "x", {"Flux": 2.801998, "e_Flux": 0.058})
    assert vlass.flux_jy == pytest.approx(2.801998)  # VLASS table5 Flux is already in Jy
    assert vlass.frequency_hz == pytest.approx(3.0e9, rel=1e-9)
    assert sed.radio_point(sed.RADIO_SPECS["nvss"][1][0], "NVSS", "nvss", "x", {"flux_20_cm": None}) is None


def test_magnitude_point_flags_upper_limits_and_sentinels() -> None:
    filt = sed.FilterCatalog(offline=True, cache_path=Path("nonexistent-dir/x.json"))
    j = sed.MAG_SPECS["twomass_psc"][1][0]
    p = sed.magnitude_point(j, "2MASS", "twomass_psc", "s", {"j_m": 17.0, "j_msigcom": None, "ph_qual": "UAA"},
                            filt.info(j.filter_id))
    assert p.is_upper_limit and p.flux_err_jy is None
    assert p.flux_jy == pytest.approx(1594.0 * 10 ** (-0.4 * 17.0))
    g = sed.MAG_SPECS["panstarrs_dr2"][1][0]
    assert sed.magnitude_point(g, "PS1", "panstarrs_dr2", "s", {"gMeanPSFMag": -999.0}, filt.info(g.filter_id)) is None
    w4 = sed.MAG_SPECS["allwise"][1][3]
    p4 = sed.magnitude_point(w4, "WISE", "allwise", "s", {"w4mpro": 8.0, "w4sigmpro": 0.1, "ph_qual": "AAAC"},
                             filt.info(w4.filter_id))
    assert not p4.is_upper_limit
    assert p4.flux_jy == pytest.approx(8.363 * 10 ** (-3.2))
    assert p4.flux_err_jy == pytest.approx(p4.flux_jy * 0.1 * math.log(10) / 2.5)


def test_proper_motion_significance_and_absolute_magnitude() -> None:
    assert sed.proper_motion_significance(3.0, 4.0, 1.0, 1.0, 0.0) == pytest.approx(5.0)
    assert sed.proper_motion_significance(3.0, 4.0, 1.0, 2.0, None) == pytest.approx(math.hypot(3.0, 2.0))
    from astropy.cosmology import Planck18

    d_pc = Planck18.luminosity_distance(0.158).to("pc").value
    assert sed.absolute_magnitude(13.0, 0.158) == pytest.approx(13.0 - 5 * math.log10(d_pc / 10))
    assert -26.5 < sed.absolute_magnitude(13.0, 0.158) < -26.0  # D_L ~ 780 Mpc


def test_type_mappings_and_redshift_kinds() -> None:
    assert sed.simbad_type_class("BLL") == ("qso", 1.0)
    assert sed.simbad_type_class("AGN") == ("agn", 1.0)
    assert sed.simbad_type_class("dS*") == ("star", 1.0)
    assert sed.simbad_type_class("G?") == ("galaxy", 0.5)
    assert sed.simbad_type_class("X") == (None, 0.0)
    assert sed.ned_type_class("QSO") == "qso" and sed.ned_type_class("G") == "galaxy" and sed.ned_type_class("*") == "star"
    assert sed.ned_redshift_kind("SUN") == "spec"
    assert sed.ned_redshift_kind("PUN") == "photo"
    assert sed.ned_redshift_kind("UUN") is None


def test_choose_redshift_priority() -> None:
    cands = [
        {"value": 0.30, "error": 0.05, "kind": "photo", "source": "sdss_photoz", "reliable": True},
        {"value": 0.1575, "error": None, "kind": "spec", "source": "simbad", "reliable": True},
        {"value": 0.1583, "error": 7e-5, "kind": "spec", "source": "ned", "reliable": True},
        {"value": 0.2, "error": 1e-3, "kind": "spec", "source": "sdss_specobj", "reliable": False},
    ]
    best = sed.choose_redshift(cands)
    assert (best["value"], best["kind"], best["source"]) == (0.1583, "spec", "ned")
    # The SDSS member spectrum paired with the flagged SpecObj row is flagged too: NED still wins.
    cands.append({"value": 0.2, "error": 1e-3, "kind": "spec", "source": "sdss", "reliable": False, "z_warning": 4})
    assert sed.choose_redshift(cands)["source"] == "ned"
    cands.append({"value": 0.1581, "error": 1e-4, "kind": "spec", "source": "sdss_specobj", "reliable": True})
    assert sed.choose_redshift(cands)["source"] == "sdss_specobj"
    only_photo = sed.choose_redshift([cands[0]])
    assert only_photo["kind"] == "photo" and only_photo["value"] == 0.30
    assert sed.choose_redshift([])["value"] is None


def test_member_tolerance_accepts_extended_radio_sources() -> None:
    # LoTSS 3C 273: 7.0" from the nucleus, sigma ~2", fitted major axis 44.6" (jet-dominated).
    assert sed.member_tolerance_arcsec("lotss", "radio", 2.01) == pytest.approx(3 * math.hypot(2.01, 0.1))
    assert sed.member_tolerance_arcsec("lotss", "radio", 2.01, 44.55) == pytest.approx(22.275)
    assert sed.member_tolerance_arcsec("gaia_dr3", "optical", 0.0003) == 1.0
    assert sed.member_tolerance_arcsec("unknown_cat", "xray", None) == 5.0


# ---------------------------------------------------------------------------
# SVO FilterCatalog: cache, network fallback
# ---------------------------------------------------------------------------


async def test_filter_catalog_fetches_caches_and_reuses(tmp_path: Path) -> None:
    cache = tmp_path / "svo.json"
    catalog = sed.FilterCatalog(cache_path=cache)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            await catalog.prefetch(["2MASS/2MASS.Ks", "WISE/WISE.W1"], client)
            assert route.call_count == 2
    info = catalog.info("WISE/WISE.W1")
    assert info.zero_point_jy == pytest.approx(309.54) and info.metadata_origin == "svo"
    assert info.wavelength_eff_um == pytest.approx(3.3526)
    stored = json.loads(cache.read_text())
    assert set(stored["filters"]) == {"2MASS/2MASS.Ks", "WISE/WISE.W1"}
    # A new catalog (new process) reads the disk cache and makes no request.
    again = sed.FilterCatalog(cache_path=cache)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(return_value=httpx.Response(500))
        await again.prefetch(["2MASS/2MASS.Ks"])
        assert route.call_count == 0
    assert again.info("2MASS/2MASS.Ks").metadata_origin == "svo-cache"


async def test_filter_catalog_falls_back_to_embedded_on_network_failure(tmp_path: Path) -> None:
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=httpx.ConnectError("offline"))
        async with offline_client() as client:
            info = await catalog.get("GAIA/GAIA3.G", client)
    assert info.metadata_origin == "embedded"
    assert info.zero_point_jy == pytest.approx(3228.7464752872)
    assert "GAIA/GAIA3.G" in catalog.errors
    assert not (tmp_path / "svo.json").exists()  # never cache fallback values as SVO answers


def test_parse_svo_votable_rejects_error_answers() -> None:
    error = b'<?xml version="1.0"?><VOTABLE version="1.1"><INFO name="QUERY_STATUS" value="ERROR">bad ID</INFO></VOTABLE>'
    with pytest.raises(sed.SEDError, match="bad ID"):
        sed.parse_svo_votable(error)
    with pytest.raises(sed.SEDError):
        sed.parse_svo_votable(b"not xml at all")


# ---------------------------------------------------------------------------
# Full pipeline replayed from recorded archive responses
# ---------------------------------------------------------------------------


def test_3c273_passport_offline(sed_3c273: dict[str, Any]) -> None:
    result = sed_3c273
    cls = result["classification"]
    assert cls["label"] == "qso"
    assert cls["confidence"] > 0.9
    assert set(cls["scores"]) == {"star", "qso", "galaxy", "agn"}
    assert sum(cls["scores"].values()) == pytest.approx(1.0, abs=1e-3)
    text = " | ".join(cls["evidence"])
    assert "Stern et al. 2012" in text and "radio-loud" in text and "SIMBAD object type 'BLL'" in text
    assert "DSC-Combmod" in text and "M_i = " in text
    # z = 0.158339 (NED, spectroscopic flag SUN)
    z = result["redshift"]
    assert z["value"] == pytest.approx(0.158339, abs=1e-5)
    assert (z["kind"], z["source"]) == ("spec", "ned")
    # Every band that the recorded catalogs contain.
    facilities = {(p["facility"], p["band"]) for p in result["points"]}
    for expected in [("Gaia DR3", "G"), ("Gaia DR3", "BP"), ("Gaia DR3", "RP"), ("2MASS", "J"), ("2MASS", "H"),
                     ("2MASS", "Ks"), ("WISE", "W1"), ("WISE", "W4"), ("Pan-STARRS1", "y"), ("SDSS", "u"),
                     ("FIRST (VLA)", "1.4 GHz"), ("NVSS (VLA)", "1.4 GHz"), ("LoTSS (LOFAR)", "144 MHz"),
                     ("Chandra ACIS", "0.5-7 keV"), ("XMM-Newton EPIC", "0.2-12 keV"), ("ROSAT PSPC", "0.1-2.4 keV"),
                     ("SIMBAD (literature)", "V")]:
        assert expected in facilities, expected
    # 2MASS Ks = 9.976 +/- 0.023 (Vega, ZP 666.8 Jy from SVO)
    ks = point(result, "2MASS", "Ks")
    assert ks["flux_jy"] == pytest.approx(666.8 * 10 ** (-0.4 * 9.97599983215332), rel=1e-9)
    assert ks["flux_err_jy"] == pytest.approx(ks["flux_jy"] * 0.023000000044703484 / sed.POGSON, rel=1e-6)
    assert ks["zero_point_origin"] == "svo" and ks["wavelength_um"] == pytest.approx(2.159)
    # Gaia G error from flux_over_error = 501.25 plus the zero-point term.
    g = point(result, "Gaia DR3", "G")
    rel = math.hypot(1 / 501.25375, 0.0027553202 / sed.POGSON)
    assert g["flux_err_jy"] / g["flux_jy"] == pytest.approx(rel, rel=1e-4)
    # PS1 AB: g = 12.9237 -> 3631 * 10^(-0.4 g)
    assert point(result, "Pan-STARRS1", "g")["flux_jy"] == pytest.approx(3631 * 10 ** (-0.4 * 12.923700332641602), rel=1e-9)
    # LoTSS component (44.6" major axis) accepted despite the 7" offset. Its SpeakTot (8.34 Jy) is a
    # REGRESSION value of the recorded catalogue row, not astrophysical truth: TGSS ADR1 gives ~116 Jy at
    # 150 MHz for 3C 273, so the point must carry a quality warning (rising 144 MHz -> 1.4 GHz spectrum).
    lotss = next(m for m in result["members"] if m["catalog"] == "lotss")
    assert lotss["used"] and "extended radio source" in lotss["reason"]
    lotss_point = point(result, "LoTSS", "144 MHz")
    assert lotss_point["flux_jy"] == pytest.approx(8.33765)
    assert lotss_point["quality_warning"] and any("probably underestimated" in n for n in lotss_point["notes"])
    # Chandra is flagged as piled up (and therefore excluded from the classification).
    chandra = point(result, "Chandra", "0.5-7 keV")
    assert any("pile-up" in n for n in chandra["notes"]) and chandra["quality_warning"]
    for p in result["points"]:
        assert p["nu_fnu_erg_s_cm2"] == pytest.approx(p["frequency_hz"] * p["flux_jy"] * 1e-23, rel=1e-12)
        assert p["wavelength_um"] * p["frequency_hz"] == pytest.approx(2.99792458e14, rel=1e-12)
    wl = [p["wavelength_um"] for p in result["points"]]
    assert wl == sorted(wl)
    assert result["notes"] == []
    assert result["gaia_extra"]["classprob_dsc_combmod_quasar"] == pytest.approx(0.8478828, rel=1e-6)
    json.dumps(result)  # JSON-serialisable


def test_m87_passport_offline(sed_m87: dict[str, Any]) -> None:
    result = sed_m87
    assert result["classification"]["label"] in {"galaxy", "agn"}
    z = result["redshift"]
    # NED's M87 zflag 'UUN' does not state the method (kind None), but NED's vetted value (cz = 1284 km/s) beats
    # SIMBAD's quality-E z = 0.0042 (round-2 review): reliable, source NED.
    assert (z["source"], z["kind"], z["reliable"]) == ("ned", None, True)
    assert z["value"] == pytest.approx(0.004283, abs=1e-6)
    simbad = next(c for c in z["candidates"] if c["source"] == "simbad")
    assert simbad["quality"] == "E" and simbad["reliable"] is False
    evidence = " | ".join(result["classification"]["evidence"])
    assert "RUWE = 2.92" in evidence  # spurious Gaia proper motion of the galaxy nucleus is not used
    assert "radio-loud" in evidence
    assert point(result, "VLASS", "3 GHz")["flux_jy"] == pytest.approx(2.801998)
    assert point(result, "FIRST", "1.4 GHz")["flux_jy"] == pytest.approx(122.19384)
    ks = point(result, "2MASS", "Ks")
    assert ks["quality_flag"] == "E" and any("poor photometric quality" in n for n in ks["notes"])
    # M87 has no SDSS photometry (coverage hole): SDSS SpecObj and Photoz were both queried, nothing found.
    assert not any(c["source"].startswith("sdss") for c in z["candidates"])


async def test_sed_without_supplementary_lookups_uses_member_data_only(tmp_path: Path) -> None:
    record = await replay_record("3c273")
    filters = sed.FilterCatalog(offline=True, cache_path=tmp_path / "none.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(return_value=httpx.Response(500))
        result = await sed.sed_from_record(record, filters=filters, supplementary=False)
        assert route.call_count == 0
    assert result["gaia_extra"] is None
    assert point(result, "Gaia DR3", "G")["flux_err_jy"] is None
    assert result["classification"]["label"] == "qso"
    assert point(result, "2MASS", "Ks")["zero_point_origin"] == "embedded"


async def test_supplementary_failures_are_reported_not_fatal(tmp_path: Path) -> None:
    record = await replay_record("3c273")
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(503, text="down"))
        async with offline_client() as client:
            result = await sed.sed_from_record(record, client=client, filters=filters)
    notes = " | ".join(result["notes"])
    assert "gaia lookup failed" in notes and "sdss lookup failed" in notes and "embedded SVO values used" in notes
    assert result["classification"]["label"] == "qso"
    assert result["redshift"]["source"] == "ned"


async def test_empty_record_gives_unknown() -> None:
    record = {"target": {"ra": 10.0, "dec": 10.0}, "crossmatch_groups": [], "failures": [], "provenance": {}}
    result = await sed.sed_from_record(record, supplementary=False, filters=sed.FilterCatalog(offline=True))
    assert result["points"] == [] and result["members"] == []
    assert result["classification"]["label"] == "unknown"
    assert result["redshift"]["value"] is None
    assert "no catalog source within the search radius" in result["notes"]
    with pytest.raises(sed.SEDError):
        await sed.sed_from_record({"crossmatch_groups": []}, supplementary=False)


async def test_sdss_specobj_redshift_from_recording() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/sdss_qso")))
        async with offline_client() as client:
            cands = await sed.fetch_sdss_redshifts(client, *SDSS_QSO, include_photoz=True)
    spec = [c for c in cands if c["kind"] == "spec"]
    assert spec and spec[0]["value"] == pytest.approx(1.008554, abs=1e-6)
    assert spec[0]["spec_class"] == "QSO" and spec[0]["z_warning"] == 0 and spec[0]["reliable"]
    assert spec[0]["id"] == "11568697826042206208"
    best = sed.choose_redshift(cands)
    assert (best["kind"], best["source"]) == ("spec", "sdss_specobj")


def test_gaia_adql_rejects_non_numeric_ids() -> None:
    assert "source_id = 3700386905605055360" in sed.gaia_extra_adql("3700386905605055360")
    with pytest.raises(ValueError):
        sed.gaia_extra_adql("1; DROP TABLE x")


# ---------------------------------------------------------------------------
# Router & CLI
# ---------------------------------------------------------------------------


class FakeService:
    """Stands in for CrossmatchService: returns a recorded UnifiedRecord for any position."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.calls: list[tuple[float, float, dict[str, Any]]] = []

    async def crossmatch(self, ra: float, dec: float, **kwargs: Any) -> UnifiedRecord:
        self.calls.append((ra, dec, kwargs))
        return UnifiedRecord(**json.loads(json.dumps(self.record, default=str)))


def make_app(service: Any, tmp_path: Path):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(sed.router)
    app.state.service = service
    app.state.sed_filters = sed.FilterCatalog(offline=True, cache_path=tmp_path / "svo.json")
    return app


def test_router_get_and_post(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    record = asyncio.run(replay_record("3c273"))
    service = FakeService(record)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges_for("3c273")))
        with TestClient(make_app(service, tmp_path)) as client:
            ra, dec = TARGETS["3c273"]
            res = client.get("/api/v1/sed", params={"ra": ra, "dec": dec, "radius_arcsec": 10})
            assert res.status_code == 200, res.text
            body = res.json()
            assert body["classification"]["label"] == "qso"
            assert body["redshift"]["kind"] == "spec"
            assert {"band", "facility", "catalog", "source_id", "wavelength_um", "frequency_hz", "flux_jy",
                    "flux_err_jy", "nu_fnu_erg_s_cm2", "is_upper_limit"} <= set(body["points"][0])
            res2 = client.post("/api/v1/sed", json={"ra": ra, "dec": dec, "radius_arcsec": 10})
            assert res2.status_code == 200 and res2.json()["redshift"]["value"] == pytest.approx(0.158339, abs=1e-5)
    assert service.calls[0][2]["radius_arcsec"] == 10.0


def test_router_validation_errors(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    with TestClient(make_app(FakeService({}), tmp_path)) as client:
        assert client.get("/api/v1/sed", params={"ra": 10}).status_code == 422
        assert client.get("/api/v1/sed", params={"ra": 400, "dec": 0}).status_code == 422
        assert client.get("/api/v1/sed", params={"ra": 10, "dec": 0, "radius_arcsec": 0}).status_code == 422
        assert client.get("/api/v1/sed", params={"ra": 10, "dec": 0, "radius_arcsec": 600}).status_code == 422
        assert client.post("/api/v1/sed", json={"radius_arcsec": 5}).status_code == 422
        assert client.post("/api/v1/sed", json={"ra": 1, "dec": 2, "bogus": 1}).status_code == 422


def test_router_upstream_failures_502_resolver_outage_503_unknown_name_404(tmp_path: Path) -> None:
    """Every catalog failing is a 502; name resolution follows models.resolution_failure_status (Sesame down:
    503 + Retry-After, unknown name: 404)."""
    from fastapi.testclient import TestClient

    failed = {
        "target": {"ra": 1.0, "dec": 2.0}, "catalogs_queried": 1, "catalog_results": {}, "counterparts": {},
        "failures": [{"catalog": "gaia_dr3", "status": "failed", "error_type": "CatalogUnavailableError"}],
        "provenance": {"catalog_stats": {"gaia_dr3": {"status": "failed"}}}, "crossmatch_groups": [],
    }
    with TestClient(make_app(FakeService(failed), tmp_path)) as client:
        res = client.get("/api/v1/sed", params={"ra": 1, "dec": 2})
        assert res.status_code == 502 and "every catalog query failed" in res.json()["detail"]
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/").mock(return_value=httpx.Response(503))
        with TestClient(make_app(FakeService(failed), tmp_path)) as client:
            res = client.post("/api/v1/sed", json={"name": "3C 273"})
            assert res.status_code == 503 and "Sesame" in res.json()["detail"]
            assert res.headers["retry-after"] == "30"
    empty = b'<?xml version="1.0"?><Sesame><Target><name>nosuch</name><INFO>*** Nothing found ***</INFO></Target></Sesame>'
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/").mock(return_value=httpx.Response(200, content=empty))
        with TestClient(make_app(FakeService(failed), tmp_path)) as client:
            res = client.get("/api/v1/sed", params={"name": "nosuchobject"})
            assert res.status_code == 404 and "Name resolution failed" in res.json()["detail"]
            assert "No coordinates found" in res.json()["detail"] and "retry-after" not in res.headers


def test_router_name_resolution_path(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    record = asyncio.run(replay_record("m87"))
    service = FakeService(record)
    sesame = (FIXTURES / "sesame" / "m87.xml").read_bytes()
    exchanges = exchanges_for("m87")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/").mock(return_value=httpx.Response(200, content=sesame))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(make_app(service, tmp_path)) as client:
            res = client.post("/api/v1/sed", json={"name": "M87"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["target"]["name"] == "M87"
    assert body["target"]["resolved_object"]["canonical_name"] == "M 87"
    assert body["classification"]["label"] in {"galaxy", "agn"}
    ra, _dec, kwargs = service.calls[0]
    assert ra == pytest.approx(187.70593076, abs=1e-6) and kwargs["epoch"] == 2000.0


def test_plot_writes_png(sed_3c273: dict[str, Any], tmp_path: Path) -> None:
    out = sed.plot_sed(sed_3c273, tmp_path / "plots" / "3c273.png")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000
    empty = sed.plot_sed({"points": [], "target": {"ra": 1.0, "dec": 2.0}, "classification": {}, "redshift": {}},
                         tmp_path / "empty.png")
    assert empty.exists()


def test_cli_handler(monkeypatch: pytest.MonkeyPatch, sed_3c273: dict[str, Any], tmp_path: Path,
                     capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))
    calls: list[tuple[Any, ...]] = []

    async def fake_build(ra, dec, *, radius_arcsec, name):
        calls.append((ra, dec, radius_arcsec, name))
        return sed_3c273

    monkeypatch.setattr(sed, "build_sed", fake_build)
    png = tmp_path / "cli.png"
    args = parser.parse_args(["sed", "--ra", "187.2779154", "--dec", "2.0523883", "--radius", "8", "--plot", str(png)])
    assert args.handler is sed.run_cli
    assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert "Classification: qso" in out and "2MASS" in out and "Redshift: 0.158" in out
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert calls == [(187.2779154, 2.0523883, 8.0, None)]
    args = parser.parse_args(["sed", "--name", "3C 273", "--json"])
    assert args.handler(args) == 0
    assert json.loads(capsys.readouterr().out)["classification"]["label"] == "qso"
    assert parser.parse_args(["sed"]).handler(parser.parse_args(["sed"])) == 2


# ---------------------------------------------------------------------------
# Recording (run as a script; needs network)
# ---------------------------------------------------------------------------


def _save_exchanges(folder: Path, name: str, log: list[tuple[httpx.Request, httpx.Response]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{name}.*.body"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (folder / f"{name}.{idx}.body").write_bytes(redact(response.content))
        exchanges.append({
            "method": request.method, "url": str(request.url),
            "request_body": request.content.decode("utf-8", "replace") if request.content else "",
            "status_code": response.status_code, "content_type": response.headers.get("content-type", ""), "match": [],
        })
    (folder / f"{name}.json").write_text(json.dumps({"catalog": name, "exchanges": exchanges}, indent=2), encoding="utf-8")


SUPPLEMENTARY_HOSTS = (("gaia_extra", "gea.esac.esa.int"), ("sdss_redshift", "skyserver.sdss.org"),
                       ("simbad_extra", "simbad.cds.unistra.fr"))

# Regression targets recorded as pruned live crossmatch records (tests/fixtures/sed/records/<key>.json) plus the
# supplementary requests sed.py makes for them. Each reproduces a reviewed failure mode (see the tests).
RECORD_TARGETS: dict[str, dict[str, Any]] = {
    "ngc5548": {"name": "NGC 5548"},  # SDSS type-3 galaxy; its SDSS spectrum has zWarning = 16
    "sbs0812": {"ra": 124.09470, "dec": 57.65254, "radius_arcsec": 5.0},  # 6C 081218+574825: SDSS zWarning = 4
    "leda37851": {"ra": 180.116508, "dec": 17.727406, "radius_arcsec": 5.0},  # SDSS galaxy (modelMag vs psfMag)
    "tmass_j1000": {"ra": 150.22660940485, "dec": 2.45665712683, "radius_arcsec": 5.0},  # star, SIMBAD photo-z 0.3
    "hizels41": {"ra": 217.54340, "dec": 33.92504, "radius_arcsec": 5.0},  # star with NED 'spec' z = 2.2
    "cyga": {"name": "Cygnus A", "radius_arcsec": 60.0},  # FR II lobes 44.5" and 52.1" from the nucleus
    "mrk421": {"name": "Mrk 421"},  # BL Lac with a 0.12 mas/yr Gaia proper motion
    "ngc1068": {"name": "NGC 1068"},  # radio-quiet Seyfert 2, saturated SDSS
    "m82": {"name": "M82"},  # starburst, Gaia BP/RP excess
    "gd71": {"name": "GD 71"},  # white dwarf, low-S/N WISE W3
    "barnard": {"name": "Barnard's star"},  # saturated SDSS/PS1
    # Round-2 review regressions.
    "3c351": {"name": "3C 351"},  # NED absorber '[HB89] 1704+608 ABS01' (AbLS, z = 0.2216) nearer than the QSO row
    "b1422": {"name": "QSO B1422+231"},  # lensed QSO z = 3.62; NED absorber row at 0.0 arcsec (z = 3.5375)
    "cena": {"name": "NGC 5128"},  # Centaurus A: Gaia member is a G = 21 knot; integrated V = 6.84
    "3c279": {"name": "3C 279"},  # blazar z = 0.536: Gaia pm 0.24 mas/yr at 5.5 sigma, excess noise sig 6.4
    "pks0118": {"name": "PKS 0118-272"},  # BL Lac: Gaia pm 0.63 mas/yr at 11 sigma, excess noise sig 7.8
    "pks2155": {"name": "PKS 2155-304"},  # point-like BL Lac with AllWISE ext_flg = 1
    "aldebaran": {"name": "Aldebaran"},  # saturated WISE: W1 'U' 5.6 mag limit for a ~5000 Jy source
    "betelgeuse": {"name": "Betelgeuse"},  # saturated WISE W1/W2
    "ngc4472": {"name": "NGC 4472"},  # NED z with zflag 'UUN' vs SIMBAD quality-E z
    "sdssj1030": {"name": "SDSS J103027.10+052455.0"},  # z = 6.3 QSO, W1-W2 = 1.0 but W2 > 15.05
    # Round-3 review regressions: multi-group Bayesian crossmatch records (the object split over several groups).
    "groombridge1830": {"name": "Groombridge 1830"},  # 2MASS row in another group at 0.40 arcsec
    "kapteyn": {"name": "Kapteyn's star"},  # AllWISE row in another group at 1.40 arcsec; planet rows 'HD 33793b/c'
    "luhman16": {"name": "Luhman 16"},  # SIMBAD/NED-only target group; 2MASS/AllWISE in another group
    "peg51": {"name": "51 Peg"},  # SIMBAD '* 51 Peg b' (Pl) co-located with the host; saturated SDSS type 3
    "t8dwarf": {"name": "2MASS J04151954-0935066"},  # T8 dwarf: AllWISE in another group at 2.79 arcsec
    "m87group": {"name": "M87"},  # Gaia nucleus in its own group at 0.107 arcsec
    "sn1998bw": {"name": "SN 1998bw"},  # SIMBAD 'SN*' with host redshift z = 0.0085
    "sn2006gy": {"name": "SN 2006gy"},  # SIMBAD 'SN*' in NGC 1260 (z = 0.019)
    "q2237": {"name": "Q2237+030"},  # Einstein Cross: lensed quasar z = 1.695, lens galaxy z = 0.039
    "adleo": {"name": "AD Leo"},  # M dwarf: Gaia DSC binarystar = 0.999999, star = 1e-6
    # Round-5 review regressions (records resolved with Sesame -oxpI: the resolution lists the object's aliases).
    "t8main": {"name": "2MASSI J0415195-093506"},  # T8 dwarf by SIMBAD's main id: 2MASS row found through an alias
    "scholz": {"name": "Scholz's star"},  # Gaia DR3 3048443305671969152 at 2.08 arcsec, named by a Sesame alias
    "pks1510": {"name": "PKS 1510-089"},  # quasar z = 0.36 (SIMBAD); NED lists z = 0.0068 (6dF): discordant spectra
    "proxima": {"name": "Proxima Centauri"},  # three 5XMM unique sources of the moving star (stack-diluted fluxes)
    "arp220": {"name": "Arp 220"},  # dusty starburst: radio excess over the optical, but q22 on the IR/radio correlation
}


def _prune_record(record: dict[str, Any], keep_per_catalog: int = 3) -> dict[str, Any]:
    """Keep what sed_from_record reads: the target, groups (nearest rows per catalog, all radio rows), failures."""
    groups = []
    for group in record.get("crossmatch_groups") or []:
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for member in group.get("members") or []:
            by_cat.setdefault(member["catalog"], []).append(member)
        members = []
        for cat, rows in by_cat.items():
            rows.sort(key=lambda m: (m.get("separation_arcsec") is None, m.get("separation_arcsec") or 0.0))
            if cat in sed.RADIO_SPECS:
                members.extend(rows)
            elif cat == "ned":  # every NED row within its tolerance: absorbers/sub-components compete with the object
                close = [m for m in rows if (m.get("separation_arcsec") or 99.0) <= sed.MATCH_FLOOR_ARCSEC["ned"]]
                members.extend(close if len(close) > keep_per_catalog else rows[:keep_per_catalog])
            else:
                members.extend(rows[:keep_per_catalog])
        groups.append({**group, "members": members})
    prov = record.get("provenance") or {}
    return {"target": record["target"], "crossmatch_groups": groups, "failures": record.get("failures") or [],
            "resolved_object": record.get("resolved_object"),
            "provenance": {"query_radius_arcsec": prov.get("query_radius_arcsec"), "catalog_stats": prov.get("catalog_stats")}}


def load_record(key: str) -> dict[str, Any]:
    return json.loads((SED_FIXTURES / "records" / f"{key}.json").read_text(encoding="utf-8"))


async def replay_record_fixture(key: str, tmp_path: Path) -> dict[str, Any]:
    """sed_from_record on a recorded crossmatch record, supplementary requests replayed from fixtures."""
    filters = sed.FilterCatalog(cache_path=tmp_path / f"svo-{key}.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"sed/{key}") + load_exchanges("sed/svo")))
        async with offline_client() as client:
            return await sed.sed_from_record(load_record(key), client=client, filters=filters,
                                             name=RECORD_TARGETS[key].get("name"))


async def _record_supplementary(client: httpx.AsyncClient, log: list, key: str, record: dict[str, Any]) -> None:
    import tempfile

    log.clear()
    filters = sed.FilterCatalog(offline=True, cache_path=Path(tempfile.mkdtemp()) / "c.json")
    result = await sed.sed_from_record(record, client=client, filters=filters)
    for note in result["notes"]:
        if "lookup failed" in note:
            print(key, "WARNING, re-record:", note[:300])
    for name, host in SUPPLEMENTARY_HOSTS:
        picked = [(q, r) for q, r in log if q.url.host == host]
        if picked:
            _save_exchanges(SED_FIXTURES / key, name, picked)
        print(key, name, [r.status_code for _, r in picked])


async def _resolve_and_save(client: httpx.AsyncClient, key: str, name: str) -> dict[str, Any]:
    """sed.resolve_name, keeping the Sesame answer as tests/fixtures/sed/sesame/<key>.xml."""
    log: list[httpx.Response] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append(response)

    client.event_hooks["response"].append(hook)
    try:
        info = await sed.resolve_name(name, client)
    finally:
        client.event_hooks["response"].remove(hook)
    answer = next(r for r in log if r.request.url.host == "cds.unistra.fr")
    (SED_FIXTURES / "sesame").mkdir(parents=True, exist_ok=True)
    (SED_FIXTURES / "sesame" / f"{key}.xml").write_bytes(answer.content)
    return info


async def _refresh_resolutions(which: list[str]) -> None:
    """Re-resolve the names of recorded records with Sesame -oxpI and store the resolution (aliases included) in
    each record, without re-running the crossmatch."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for key, spec in RECORD_TARGETS.items():
            if not spec.get("name") or (which and key not in which):
                continue
            path = SED_FIXTURES / "records" / f"{key}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            info = await _resolve_and_save(client, key, spec["name"])
            record["resolved_object"] = json.loads(json.dumps(info["resolved"], default=str))
            path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
            print(key, "aliases:", len(record["resolved_object"].get("aliases") or []))
            await asyncio.sleep(0.5)  # polite


async def _record(which: list[str]) -> None:
    import tempfile

    from main import build_service

    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        if "svo" in which:  # SVO filter answers, one file per filter.
            for fid in sed.FILTER_SYSTEMS:
                log.clear()
                await sed.FilterCatalog(cache_path=Path(tempfile.mkdtemp()) / "c.json").prefetch([fid], client)
                _save_exchanges(SED_FIXTURES / "svo", fid.replace("/", "_"), list(log))
                print("svo", fid, log[-1][1].status_code)
        if "canary" in which:  # Supplementary requests of the canary pipelines (crossmatch from tests/fixtures/<t>).
            for target in ("3c273", "m87"):
                await _record_supplementary(client, log, target, await replay_record(target))
            log.clear()
            await sed.fetch_sdss_redshifts(client, *SDSS_QSO, include_photoz=True)
            _save_exchanges(SED_FIXTURES / "sdss_qso", "sdss_redshift", list(log))
            print("sdss_qso", [r.status_code for _, r in log])
        for key, spec in RECORD_TARGETS.items():
            if key not in which and "targets" not in which:
                continue
            if "supplementary-only" not in which:
                async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as plain:
                    ra, dec, epoch, pm = spec.get("ra"), spec.get("dec"), None, (None, None)
                    resolved = None
                    if spec.get("name"):
                        info = await _resolve_and_save(plain, key, spec["name"])
                        ra, dec, epoch = info["ra"], info["dec"], info["epoch"]
                        pm, resolved = (info["pm_ra_masyr"], info["pm_dec_masyr"]), info["resolved"]
                    rec = await build_service(client=plain).crossmatch(
                        ra, dec, radius_arcsec=spec.get("radius_arcsec", 10.0), epoch=epoch,
                        pm_ra_masyr=pm[0], pm_dec_masyr=pm[1])
                    if resolved is not None:
                        rec.resolved_object = resolved
                pruned = _prune_record(json.loads(json.dumps(rec.as_dict(), default=str)))
                (SED_FIXTURES / "records").mkdir(parents=True, exist_ok=True)
                (SED_FIXTURES / "records" / f"{key}.json").write_text(json.dumps(pruned, indent=1, sort_keys=True), encoding="utf-8")
                print(key, "record", [f["catalog"] for f in pruned["failures"]])
            await _record_supplementary(client, log, key, load_record(key))


if __name__ == "__main__":
    # record [svo] [canary] [targets | <key> ...] [supplementary-only]
    if sys.argv[1:3] == ["record", "resolutions"]:
        asyncio.run(_refresh_resolutions(sys.argv[3:]))
    elif sys.argv[1:2] == ["record"]:
        asyncio.run(_record(sys.argv[2:] or ["svo", "canary", "targets"]))
    else:
        print(__doc__)
