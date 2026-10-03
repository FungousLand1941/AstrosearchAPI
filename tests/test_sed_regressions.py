"""Offline regression tests for reviewed sed.py failure modes (round-1 review).

Each full-pipeline test replays a pruned live crossmatch record (tests/fixtures/sed/records/<key>.json) plus the
supplementary Gaia / SDSS / SIMBAD requests recorded for it (tests/fixtures/sed/<key>/). Re-record with

    .venv/Scripts/python.exe tests/test_sed.py record targets
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client
from test_sed import FakeService, make_app, point, replay_record, replay_record_fixture

import sed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CACHE: dict[str, dict[str, Any]] = {}


def passport(key: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if key not in _CACHE:
        _CACHE[key] = asyncio.run(replay_record_fixture(key, tmp_path_factory.mktemp(f"sed-{key}")))
    return _CACHE[key]


def evidence(result: dict[str, Any]) -> str:
    return " | ".join(result["classification"]["evidence"])


def members(result: dict[str, Any], catalog: str) -> dict[str, Any]:
    return next(m for m in result["members"] if m["catalog"] == catalog)


def member_choice(catalog: str, data: dict[str, Any], source_id: str = "1") -> sed.MemberChoice:
    return sed.MemberChoice(catalog=catalog, source_id=source_id, ra=None, dec=None, separation_arcsec=0.0,
                            tolerance_arcsec=1.0, used=True, reason="test", wavelength=None, data=data)


# ---------------------------------------------------------------------------
# 1. SIMBAD photometric redshifts are not spectroscopic
# ---------------------------------------------------------------------------


def test_simbad_rvz_nature_mapping() -> None:
    assert sed.simbad_redshift_kind("s") == "spec"
    assert sed.simbad_redshift_kind("se") == "spec"
    assert sed.simbad_redshift_kind("sa") == "spec"
    assert sed.simbad_redshift_kind("p") == "photo"
    assert sed.simbad_redshift_kind(None) is None and sed.simbad_redshift_kind("") is None


def test_simbad_photo_candidate_is_never_spec() -> None:
    simbad = member_choice("simbad", {"rvz_redshift": 0.3}, "2MASS J10005438+0227239")
    # Without the SIMBAD lookup the kind is unknown, never assumed spectroscopic.
    (unknown,) = sed.member_redshift_candidates([simbad])
    assert unknown["kind"] is None and unknown["reliable"] is None
    extra = {"rvz_nature": "p", "rvz_qual": "E", "rvz_type": "z", "rvz_err": None, "rvz_bibcode": "2007ApJS..172...99C"}
    (photo,) = sed.member_redshift_candidates([simbad], simbad_extra=extra)
    assert photo["kind"] == "photo" and photo["reliable"] is False and photo["quality"] == "E"
    ned_photo = {"value": 0.3, "error": None, "kind": "photo", "source": "ned", "reliable": True}
    best = sed.choose_redshift([photo, ned_photo])
    assert (best["kind"], best["source"]) == ("photo", "ned")
    # A spectroscopic SIMBAD value of quality E ranks below a reliable photo-z and above nothing else.
    spec_e = dict(photo, kind="spec")
    assert sed.choose_redshift([spec_e, ned_photo])["source"] == "ned"
    assert sed.choose_redshift([spec_e])["reliable"] is False


def test_star_with_simbad_photo_z_live_record(tmp_path_factory: pytest.TempPathFactory) -> None:
    """2MASS J10005438+0227239: SIMBAD otype '*', rvz 0.3 with rvz_nature 'p' (COSMOS), Gaia parallax/error 235."""
    result = passport("tmass_j1000", tmp_path_factory)
    z = result["redshift"]
    simbad = next(c for c in z["candidates"] if c["source"] == "simbad")
    assert (simbad["kind"], simbad["nature"], simbad["quality"], simbad["reliable"]) == ("photo", "p", "E", False)
    assert z["kind"] == "photo"  # never 'spec'
    assert result["classification"]["label"] == "star"
    text = evidence(result)
    assert "spectroscopic z" not in text and "M_i =" not in text
    assert "conflicts with the stellar classification" in text
    assert z["reliable"] is False and z["conflict"] is True and "Doppler" not in z["note"]


# ---------------------------------------------------------------------------
# 2. SDSS zWarning is honoured for the member spectrum
# ---------------------------------------------------------------------------


def test_member_specz_takes_zwarning_of_the_same_spectrum() -> None:
    """6C 081218+574825: SpecObj 5797489067743795200 z = 2.074916, zWarning = 4; NED z = 0.05 (SUN)."""
    specobj = {"value": 2.074916, "error": 2.2e-4, "kind": "spec", "source": "sdss_specobj", "reliable": False,
               "z_warning": 4, "spec_class": "QSO", "id": "5797489067743795200"}
    sdss = member_choice("sdss", {"specz": 2.074916, "speczErr": 2.2e-4, "specClass": "QSO"}, "1237663788494880804")
    ned = member_choice("ned", {"z": 0.05, "zflag": "SUN"}, "SBS 0812+578")
    # Paired with the flagged SpecObj row (cone query): unreliable.
    cands = [specobj, *sed.member_redshift_candidates([sdss, ned], specobj_candidates=[specobj])]
    member = next(c for c in cands if c["source"] == "sdss")
    assert member["reliable"] is False and member["z_warning"] == 4
    best = sed.choose_redshift(cands)
    assert (best["value"], best["source"], best["kind"]) == (0.05, "ned", "spec")
    # From the per-object spectra (sdss_extra) alone.
    extra = {"spectra": [{"specObjID": "5797489067743795200", "z": 2.074916, "zWarning": 4}]}
    (m2, _) = sed.member_redshift_candidates([sdss, ned], sdss_extra=extra)
    assert m2["reliable"] is False
    # Unverifiable member spectrum: reliability unknown, and it ranks below NED's spectroscopic value.
    (m3, n3) = sed.member_redshift_candidates([sdss, ned])
    assert m3["reliable"] is None and "zWarning" in m3["note"]
    assert sed.choose_redshift([m3, n3])["source"] == "ned"
    # The SDSS spectral class of a flagged spectrum is not evidence.
    ev = sed.gather_evidence([], [sdss, ned], sed.choose_redshift(cands))
    assert not any("SDSS spectroscopic class" in e.text for e in ev)


@pytest.mark.parametrize(("key", "ned_z", "warning"), [("sbs0812", 0.05, 4), ("ngc5548", 0.017175, 16)])
def test_flagged_sdss_spectrum_live_records(key: str, ned_z: float, warning: int, tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport(key, tmp_path_factory)
    z = result["redshift"]
    assert (z["source"], z["kind"], z["reliable"]) == ("ned", "spec", True)
    assert z["value"] == pytest.approx(ned_z, abs=1e-6)
    sdss = [c for c in z["candidates"] if c["source"] in {"sdss", "sdss_specobj"}]
    assert {c["source"] for c in sdss} == {"sdss", "sdss_specobj"}
    assert all(c["reliable"] is False and c["z_warning"] == warning for c in sdss)
    assert "SDSS spectroscopic class" not in evidence(result)


# ---------------------------------------------------------------------------
# 3. One SDSS aperture for all bands
# ---------------------------------------------------------------------------


def test_sdss_aperture_requires_model_magnitudes_in_all_bands() -> None:
    registry_row = {"type": 3, "psfMag_u": 19.9, "psfMag_g": 19.41, "psfMag_r": 19.2, "psfMag_i": 18.95, "psfMag_z": 18.9,
                    "psfMagErr_r": 0.02, "modelMag_r": 15.71}
    assert sed.sdss_aperture(registry_row) == "psf"
    full = dict(registry_row, **{f"modelMag_{b}": 16.0 for b in "ugiz"}, **{f"modelMagErr_{b}": 0.01 for b in "ugriz"})
    assert sed.sdss_aperture(full) == "model"
    assert sed.sdss_aperture(dict(full, type=6)) == "psf"
    filters = sed.FilterCatalog(offline=True, cache_path=Path("nonexistent-dir/x.json"))
    pts = sed.extract_points([member_choice("sdss", registry_row)], filters)
    mags = {p.band: p.magnitude for p in pts}
    assert mags == {"u": 19.9, "g": 19.41, "r": 19.2, "i": 18.95, "z": 18.9}  # psfMag everywhere, not modelMag_r
    assert any("underestimates the total flux" in n for n in pts[0].notes)


@pytest.mark.parametrize("key", ["leda37851", "ngc5548"])
def test_sdss_galaxy_sed_has_no_aperture_spike(key: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport(key, tmp_path_factory)
    sdss = sorted((p for p in result["points"] if p["facility"] == "SDSS"), key=lambda p: p["wavelength_um"])
    assert [p["band"] for p in sdss] == ["u", "g", "r", "i", "z"]
    assert all(any("modelMag" in n and "all bands" in n for n in p["notes"]) for p in sdss)
    assert all(p["flux_err_jy"] is not None for p in sdss)  # modelMagErr fetched for every band
    # Adjacent bands within a factor 5 (u - g <= 1.75 mag covers the 4000 A break of old populations; the
    # mixed-aperture bug produced 13-80x spikes at r).
    for a, b in itertools.pairwise(sdss):
        assert 1 / 5 < b["flux_jy"] / a["flux_jy"] < 5, (a["band"], b["band"])


def test_leda37851_sdss_r_is_model_magnitude(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("leda37851", tmp_path_factory)
    r = point(result, "SDSS", "r")
    assert r["magnitude"] == pytest.approx(result["sdss_extra"]["modelMag_r"])
    g = point(result, "SDSS", "g")
    assert g["magnitude"] == pytest.approx(result["sdss_extra"]["modelMag_g"])


# ---------------------------------------------------------------------------
# 4. Quality flags (SDSS saturation, PS1, Gaia BP/RP excess)
# ---------------------------------------------------------------------------


def test_gaia_corrected_excess_hand_computed() -> None:
    # Riello et al. 2021 Table 2, 0.5 <= x < 4: f(1) = 1.162004 + 0.011464 + 0.049255 - 0.005879 = 1.216844
    assert sed.gaia_corrected_excess(1.3, 1.0) == pytest.approx(1.3 - 1.216844, abs=1e-9)
    # x < 0.5: f(0) = 1.154360; x >= 4: f(4) = 1.057572 + 4 * 0.140537 = 1.619720
    assert sed.gaia_corrected_excess(1.154360, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert sed.gaia_corrected_excess(1.7, 4.0) == pytest.approx(1.7 - 1.619720, abs=1e-9)
    # Eq. 18: sigma_C*(G = 15) = 0.0059898 + 8.817481e-12 * 15^7.618399
    assert sed.gaia_excess_sigma(15.0) == pytest.approx(0.0059898 + 8.817481e-12 * 15.0**7.618399, rel=1e-12)
    # Riello et al. 2021 Table C.2: G - V at BP-RP = 1: -0.02704 + 0.01424 - 0.2156 + 0.01426 = -0.21414
    assert sed.gaia_predicted_v(10.0, 1.0) == pytest.approx(10.21414, abs=1e-9)
    assert sed.gaia_predicted_v(10.0, 6.0) is None


def test_ps1_quality_flags() -> None:
    filters = sed.FilterCatalog(offline=True, cache_path=Path("nonexistent-dir/x.json"))
    row = {"gMeanPSFMag": 13.2, "gMeanPSFMagErr": 0.01, "rMeanPSFMag": 14.0, "rMeanPSFMagErr": 0.01, "qualityFlag": 60}
    pts = {p.band: p for p in sed.extract_points([member_choice("panstarrs_dr2", row)], filters)}
    assert pts["g"].quality_warning and "saturation limit 13.5" in " ".join(pts["g"].notes)  # Magnier et al. 2013
    assert not pts["r"].quality_warning
    bad = {p.band: p for p in sed.extract_points([member_choice("panstarrs_dr2", dict(row, qualityFlag=99))], filters)}
    assert bad["r"].quality_warning and any("QF_OBJ_GOOD not set" in n for n in bad["r"].notes)
    assert any("QF_OBJ_EXT" in n for n in bad["r"].notes)


def test_barnard_saturated_sdss_and_ps1_are_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("barnard", tmp_path_factory)
    for band in "griz":  # SkyServer flags_<band> & SATURATED for objID 1237668573088843078
        p = point(result, "SDSS", band)
        assert p["quality_warning"] and f"SDSS SATURATED flag set in {band}" in p["notes"]
    assert not point(result, "SDSS", "u")["quality_warning"]
    assert point(result, "Pan-STARRS1", "i")["quality_warning"]
    assert result["classification"]["label"] == "star"


def test_m82_gaia_bp_rp_excess_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("m82", tmp_path_factory)
    for band in ("BP", "RP"):
        p = point(result, "Gaia DR3", band)
        assert p["quality_warning"] and any("C* =" in n for n in p["notes"])
    assert not point(result, "Gaia DR3", "G")["quality_warning"]


def test_flagged_points_are_not_used_by_the_classification(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = asyncio.run(_replay_canary("3c273", tmp_path_factory))
    text = evidence(result)
    # X-ray/optical ratio from XMM, not the piled-up Chandra point.
    assert "log(fX/fV)" in text and "(XMM-Newton EPIC" in text and "flagged, not used: Chandra ACIS" in text
    # PS1 i (0.7 mag below its neighbours) and the saturated SDSS i are rejected for R_i and M_i.
    assert "Pan-STARRS1 i differs by" in text and "SDSS i quality-flagged" in text
    assert "M_i = " in text and "from i-band flux from interpolation" in text
    # SIMBAD V = 14.83 (Tycho-2) is inconsistent with Gaia (V ~ 12.9) and flagged, with its SIMBAD error.
    v = point(result, "SIMBAD", "V")
    assert v["quality_warning"] and v["magnitude"] == pytest.approx(14.83)
    assert v["magnitude_error"] == pytest.approx(0.022) and v["flux_err_jy"] is not None


async def _replay_canary(target: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from test_sed import replay_sed

    return await replay_sed(target, tmp_path_factory.mktemp(f"canary-{target}"))


# ---------------------------------------------------------------------------
# 5. Radio loudness of extended counterparts uses total optical light
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ngc1068", "m82", "ngc5548"])
def test_radio_quiet_extended_galaxies_are_not_radio_loud(key: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    text = evidence(passport(key, tmp_path_factory))
    assert "radio-loud" not in text
    assert "F_opt,total" in text and "radio-quiet or star-forming" in text


def test_m87_extended_but_radio_loud(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = asyncio.run(_replay_canary("m87", tmp_path_factory))
    text = evidence(result)
    assert "radio-loud" in text and "F_opt,total" in text and "SIMBAD V = 8.63" in text


def test_optical_extent_precedence() -> None:
    sdss6 = member_choice("sdss", {"type": 6})
    wise_ext = member_choice("allwise", {"ext_flg": 4})
    assert sed.optical_extent([sdss6, wise_ext]) == (False, "SDSS type 6")
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": 18.0, "iMeanKronMag": 17.5})
    assert sed.optical_extent([ps1, wise_ext])[0] is True
    saturated = member_choice("panstarrs_dr2", {"iMeanPSFMag": 12.0, "iMeanKronMag": 11.0})
    assert sed.optical_extent([saturated])[1] == "morphology unknown"


# ---------------------------------------------------------------------------
# 6. Gaia proper motions include the DR3 systematic floor
# ---------------------------------------------------------------------------


def test_mrk421_proper_motion_is_not_galactic(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Mrk 421: pm = 0.12 mas/yr, formal errors 0.0166/0.0170 mas/yr (6.8 sigma) at G = 13.157."""
    result = passport("mrk421", tmp_path_factory)
    text = evidence(result)
    assert "moving, Galactic" not in text
    assert "33 uas/yr systematics" in text and "inconclusive" in text
    assert "qso_candidates member" in text
    assert result["classification"]["label"] in {"qso", "agn"}


def test_proper_motion_floor_hand_computed() -> None:
    e = math.hypot(0.0166, 0.033)
    sig = sed.proper_motion_significance(-0.11226, 0.03862, e, math.hypot(0.0170, 0.033), None)
    assert 3.0 < sig < 5.0


# ---------------------------------------------------------------------------
# 7. Stars with large redshifts
# ---------------------------------------------------------------------------


def test_star_with_ned_spectroscopic_z_is_a_conflict(tmp_path_factory: pytest.TempPathFactory) -> None:
    """B-HiZELS NB392 41: SIMBAD/NED '*', NED z = 2.2 (S1L), SIMBAD z = 2.2 photometric."""
    result = passport("hizels41", tmp_path_factory)
    assert result["classification"]["label"] == "star"
    assert result["classification"]["redshift_conflict"] is True
    z = result["redshift"]
    assert z["value"] == pytest.approx(2.2) and z["reliable"] is False and z["conflict"] is True
    assert "misassociation" in z["note"] and "Doppler" not in z["note"]
    text = evidence(result)
    assert "M_i" not in text and "spectroscopic z = " not in text


def test_nearby_star_keeps_doppler_note(tmp_path_factory: pytest.TempPathFactory) -> None:
    z = passport("gd71", tmp_path_factory)["redshift"]
    assert "Doppler" in z["note"] and z.get("conflict") is None


# ---------------------------------------------------------------------------
# 8. Extended radio sources (lobe pairs, duplicate rows)
# ---------------------------------------------------------------------------


def test_cygnus_a_lobes_are_summed(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("cyga", tmp_path_factory)
    nvss = point(result, "NVSS", "1.4 GHz")
    # NVSS J195932+404347 (858423 mJy) + NVSS J195924+404419 (739767 mJy)
    assert nvss["flux_jy"] == pytest.approx(858.423 + 739.767, rel=1e-9)
    assert nvss["flux_err_jy"] == pytest.approx(math.hypot(32.3723, 24.6211), rel=1e-6)
    member = members(result, "nvss")
    assert member["used"] and [c["role"] for c in member["components"]] == ["lobe", "lobe"]
    assert "lobe pair" in member["reason"]
    assert "radio-loud" in evidence(result)
    assert result["classification"]["label"] in {"agn", "galaxy"}


def test_lobe_pair_geometry() -> None:
    centre = (10.0, 20.0)
    east = {"catalog": "nvss", "source_id": "E", "ra": 10.0 + 40 / 3600 / math.cos(math.radians(20)), "dec": 20.0,
            "separation_arcsec": 40.0, "data": {"flux_20_cm": 1000.0}}
    west = {"catalog": "nvss", "source_id": "W", "ra": 10.0 - 50 / 3600 / math.cos(math.radians(20)), "dec": 20.0,
            "separation_arcsec": 50.0, "data": {"flux_20_cm": 800.0}}
    north = {"catalog": "nvss", "source_id": "N", "ra": 10.0, "dec": 20.0 + 45 / 3600, "separation_arcsec": 45.0,
             "data": {"flux_20_cm": 900.0}}
    a, b, angle = sed._lobe_pair([east, west, north], centre)
    assert {a["source_id"], b["source_id"]} == {"E", "W"} and angle == pytest.approx(180.0, abs=0.01)
    assert sed._lobe_pair([east, north], centre) is None  # 90 degrees apart
    far = dict(west, ra=10.0 - 100 / 3600 / math.cos(math.radians(20)), separation_arcsec=100.0)
    assert sed._lobe_pair([east, far], centre) is None  # distance ratio 2.5 > 2


def test_repeated_radio_rows_are_not_summed(tmp_path_factory: pytest.TempPathFactory) -> None:
    """NGC 1068: two VLASS rows named J024240.73-000046.8 (different epochs): keep one, do not add them."""
    result = passport("ngc1068", tmp_path_factory)
    vlass = point(result, "VLASS", "3 GHz")
    assert vlass["flux_jy"] == pytest.approx(2.005865)
    assert "repeated row" in members(result, "vlass")["reason"]


def test_unassociated_bright_radio_component_is_reported(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("m82", tmp_path_factory)
    assert any("vlass component" in n and "outside the association tolerance" in n for n in result["notes"])


# ---------------------------------------------------------------------------
# 10. FilterCatalog recovers from transient SVO failures
# ---------------------------------------------------------------------------


async def test_filter_catalog_retries_after_failure_and_clears_errors(tmp_path: Path) -> None:
    cache = tmp_path / "svo.json"
    catalog = sed.FilterCatalog(cache_path=cache, retry_after_seconds=0.0)
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=httpx.ConnectError("blip"))
        async with offline_client() as client:
            failures = await catalog.prefetch(["2MASS/2MASS.Ks"], client)
    assert "ConnectError" in failures["2MASS/2MASS.Ks"]
    assert catalog.info("2MASS/2MASS.Ks").metadata_origin == "embedded"
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            failures = await catalog.prefetch(["2MASS/2MASS.Ks"], client)
        assert route.call_count == 1
    assert failures == {} and catalog.errors == {}
    assert catalog.info("2MASS/2MASS.Ks").metadata_origin == "svo"
    assert "2MASS/2MASS.Ks" in json.loads(cache.read_text())["filters"]


async def test_filter_catalog_info_does_not_pin_embedded_values(tmp_path: Path) -> None:
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    assert catalog.info("WISE/WISE.W1").metadata_origin == "embedded"
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            await catalog.prefetch(["WISE/WISE.W1"], client)
        assert route.call_count == 1
    assert catalog.info("WISE/WISE.W1").metadata_origin == "svo"


async def test_filter_catalog_backoff_reports_pending_retry(tmp_path: Path) -> None:
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json", retry_after_seconds=3600.0)
    with respx.mock(assert_all_mocked=True) as router:
        route = router.route().mock(return_value=httpx.Response(503))
        async with offline_client() as client:
            first = await catalog.prefetch(["GAIA/GAIA3.G"], client)
            second = await catalog.prefetch(["GAIA/GAIA3.G"], client)
        assert route.call_count == 1  # not hammered during the back-off
    assert "HTTP 503" in first["GAIA/GAIA3.G"] and "retry pending" in second["GAIA/GAIA3.G"]


async def test_sed_notes_report_only_this_requests_svo_failures(tmp_path: Path) -> None:
    record = await replay_record("3c273")
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json", retry_after_seconds=0.0)
    exchanges = load_exchanges("sed/3c273") + load_exchanges("sed/svo")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(503))
        async with offline_client() as client:
            first = await sed.sed_from_record(record, client=client, filters=filters)
    assert any("SVO FPS lookup" in n for n in first["notes"])
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            second = await sed.sed_from_record(record, client=client, filters=filters)
    assert second["notes"] == []
    assert point(second, "2MASS", "Ks")["zero_point_origin"] == "svo"


@pytest.mark.parametrize("payload", ["null", "[]", '{"filters": null}', '{"filters": [1, 2]}', "not json"])
def test_corrupt_disk_cache_is_ignored(tmp_path: Path, payload: str) -> None:
    cache = tmp_path / "svo.json"
    cache.write_text(payload, encoding="utf-8")
    info = sed.FilterCatalog(cache_path=cache, offline=True).info("2MASS/2MASS.Ks")
    assert info.zero_point_jy == pytest.approx(666.8) and info.metadata_origin == "embedded"


# ---------------------------------------------------------------------------
# 11. Colour rules need S/N; literature magnitudes are checked
# ---------------------------------------------------------------------------


def test_gd71_low_snr_w3_does_not_trigger_the_wright_locus(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("gd71", tmp_path_factory)
    assert point(result, "WISE", "W3")["quality_flag"] == "C"
    text = evidence(result)
    assert "dusty star-forming" not in text and "W3 ph_qual C" in text
    assert result["classification"]["label"] == "star"


def test_barnard_simbad_v_is_consistent_and_not_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    # V = 9.51 vs V predicted from Gaia G, BP-RP (|C*| = 0.024 < 0.1): consistent.
    assert not point(passport("barnard", tmp_path_factory), "SIMBAD", "V")["quality_warning"]


# ---------------------------------------------------------------------------
# 17-20. Asinh non-detections, LoTSS, plotting backend, CLI, Sesame
# ---------------------------------------------------------------------------


def test_sdss_negative_asinh_flux_is_an_upper_limit() -> None:
    filters = sed.FilterCatalog(offline=True, cache_path=Path("nonexistent-dir/x.json"))
    u = sed.MAG_SPECS["sdss"][1][0]
    p = sed.magnitude_point(u, "SDSS", "sdss", "x", {"psfMag_u": 25.5}, filters.info(u.filter_id))
    assert p is not None and p.is_upper_limit and p.flux_err_jy is None
    # 3 b f0 with u_AB = u_SDSS - 0.04: 3 * 1.4e-10 * 3631 * 10^(0.016)
    assert p.flux_jy == pytest.approx(3 * 1.4e-10 * 3631 * 10**0.016, rel=1e-9)
    flux, err = sed.sdss_asinh_mag_to_jy(25.5, "u", 0.5)
    q = sed.magnitude_point(u, "SDSS", "sdss", "x", {"psfMag_u": 25.5, "psfMagErr_u": 0.5}, filters.info(u.filter_id))
    assert flux < 0 and q.flux_jy == pytest.approx(3 * err, rel=1e-9) and q.is_upper_limit


def test_lotss_rising_spectrum_is_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = asyncio.run(_replay_canary("3c273", tmp_path_factory))
    lotss = point(result, "LoTSS", "144 MHz")
    assert lotss["quality_warning"] and any("alpha = +0.83" in n for n in lotss["notes"])


def test_plot_sed_keeps_the_callers_backend(tmp_path: Path) -> None:
    import matplotlib

    before = matplotlib.get_backend()
    try:
        matplotlib.use("svg")
        sed.plot_sed({"points": [], "target": {"ra": 1.0, "dec": 2.0}, "classification": {}, "redshift": {}}, tmp_path / "x.png")
        assert matplotlib.get_backend() == "svg"
    finally:
        matplotlib.use(before)
    assert (tmp_path / "x.png").read_bytes()[:4] == b"\x89PNG"


def test_cli_plot_without_matplotlib(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def fake_build(ra, dec, *, radius_arcsec, name):
        return {"target": {"ra": 1.0, "dec": 2.0}, "points": [], "classification": {}, "redshift": {}, "notes": []}

    def no_matplotlib(*args: Any, **kwargs: Any) -> Path:
        raise ModuleNotFoundError("No module named 'matplotlib'")

    monkeypatch.setattr(sed, "build_sed", fake_build)
    monkeypatch.setattr(sed, "plot_sed", no_matplotlib)
    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sed", "--ra", "1", "--dec", "2", "--plot", "out.png"])
    assert args.handler(args) == 1
    assert "matplotlib is required for --plot" in capsys.readouterr().out


def test_malformed_sesame_answer_is_502(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/").mock(
            return_value=httpx.Response(200, text="<html>Service Unavailable"))
        with TestClient(make_app(FakeService({}), tmp_path)) as client:
            res = client.get("/api/v1/sed", params={"name": "3C 273"})
    assert res.status_code == 502 and "could not be parsed" in res.json()["detail"]
