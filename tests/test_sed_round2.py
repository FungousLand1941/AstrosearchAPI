"""Offline regression tests for the round-2 review of sed.py.

Full-pipeline tests replay pruned live crossmatch records (tests/fixtures/sed/records/<key>.json) plus the
supplementary Gaia / SDSS / SIMBAD requests recorded for them (tests/fixtures/sed/<key>/); re-record with

    .venv/Scripts/python.exe tests/test_sed.py record <key> [supplementary-only]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client
from test_sed import FakeService, make_app, point, replay_record
from test_sed_regressions import evidence, member_choice, members, passport

import sed
from models import AstroSearchError, CatalogUnavailableError, ResponseParseError, UnifiedRecord

OFFLINE = Path("nonexistent-dir/x.json")


def offline_filters() -> sed.FilterCatalog:
    return sed.FilterCatalog(offline=True, cache_path=OFFLINE)


def evidence_line(result: dict[str, Any], needle: str) -> str:
    hits = [e for e in result["classification"]["evidence"] if needle in e]
    assert hits, f"no evidence containing {needle!r}: {result['classification']['evidence']}"
    return hits[0]


# ---------------------------------------------------------------------------
# 1. NED sub-component rows (absorbers) are never the object
# ---------------------------------------------------------------------------

ABSORBER = {"catalog": "ned", "source_id": "[HB89] 1704+608 ABS01", "ra": 256.1723, "dec": 60.7418,
            "separation_arcsec": 0.021, "data": {"prefphytype": "AbLS", "z": 0.2216, "zflag": "SUN"}}
QUASAR = {"catalog": "ned", "source_id": "SBS 1704+608", "ra": 256.1723, "dec": 60.7418,
          "separation_arcsec": 0.089, "data": {"prefphytype": "QSO", "z": 0.37152, "zflag": "SLS"}}


def test_ned_object_row_beats_a_nearer_absorber() -> None:
    (choice,) = sed.select_members({"members": [ABSORBER, QUASAR]})
    assert choice.source_id == "SBS 1704+608" and choice.used
    assert "[HB89] 1704+608 ABS01 (abls, 0.021 arcsec)" in choice.reason
    (cand,) = sed.member_redshift_candidates([choice])
    assert (cand["value"], cand["kind"], cand["id"]) == (0.37152, "spec", "SBS 1704+608")


def test_absorber_redshift_is_never_used_even_alone() -> None:
    (only,) = sed.select_members({"members": [ABSORBER]})
    assert only.used and only.source_id.endswith("ABS01")
    assert sed.member_redshift_candidates([only]) == []
    assert sed.ned_type_class("AbLS") is None and sed.ned_type_class("*Cl") is None  # not class evidence


def test_ned_untyped_and_xray_rows_rank_below_the_galaxy() -> None:
    xray = {"catalog": "ned", "source_id": "2CXO J004244.3+411607", "separation_arcsec": 0.3,
            "data": {"prefphytype": "XrayS"}}
    untyped = {"catalog": "ned", "source_id": "WISEA J004244.3+411608", "separation_arcsec": 0.5, "data": {}}
    galaxy = {"catalog": "ned", "source_id": "MESSIER 031", "separation_arcsec": 1.2,
              "data": {"prefphytype": "G", "z": -0.001001, "zflag": "SUN"}}
    (choice,) = sed.select_members({"members": [xray, untyped, galaxy]})
    assert choice.source_id == "MESSIER 031"
    # Beyond the NED tolerance the nearest row is reported (and not used), whatever its type.
    far = dict(galaxy, separation_arcsec=9.0)
    (choice,) = sed.select_members({"members": [far]})
    assert not choice.used


def test_ned_subobject_names_rank_below_the_object() -> None:
    # Live NED rows of 3C 48: the absorber is typed 'QSO' and sits at the same separation as the quasar.
    absorber = {"catalog": "ned", "source_id": "3C 048 ABS01", "separation_arcsec": 0.054,
                "data": {"prefphytype": "QSO", "z": 0.365399987, "zflag": "UUN"}}
    quasar = {"catalog": "ned", "source_id": "3C 048", "separation_arcsec": 0.054,
              "data": {"prefphytype": "QSO", "z": 0.369, "zflag": "SLS"}}
    (choice,) = sed.select_members({"members": [absorber, quasar]})
    assert choice.source_id == "3C 048"
    (only,) = sed.select_members({"members": [absorber]})
    assert sed.member_redshift_candidates([only]) == []  # an 'ABSnn' row never supplies the redshift
    part = {"catalog": "ned", "source_id": "NGC 0253:[HFE2003] ESX-02", "separation_arcsec": 0.5,
            "data": {"prefphytype": "G"}}
    assert sed.ned_is_subcomponent(part) and not sed.ned_is_absorber(part)
    assert sed.ned_is_subcomponent({"source_id": "CRATES J0047-2517 NED02", "data": {}})


def test_ned_object_record_with_redshift_beats_a_nearer_same_class_row() -> None:
    # Live NED rows of NGC 253: a 'G'-typed Chandra source at 1.38 arcsec, NGC 0253 itself (z, S1L) at 2.58 arcsec.
    xray_g = {"catalog": "ned", "source_id": "2CXO J004733.1-251718", "separation_arcsec": 1.377,
              "data": {"prefphytype": "G"}}
    galaxy = {"catalog": "ned", "source_id": "NGC 0253", "separation_arcsec": 2.581,
              "data": {"prefphytype": "G", "z": 0.000807, "zflag": "S1L"}}
    (choice,) = sed.select_members({"members": [xray_g, galaxy]})
    assert choice.source_id == "NGC 0253" and "2CXO J004733.1-251718" in choice.reason
    # A nearer object of a different class (a star in front of a galaxy) is kept.
    star = {"catalog": "ned", "source_id": "TYC 1-2-3", "separation_arcsec": 0.2, "data": {"prefphytype": "*"}}
    (choice,) = sed.select_members({"members": [star, galaxy]})
    assert choice.source_id == "TYC 1-2-3"


@pytest.mark.parametrize(("key", "z_true", "ned_id", "absorber"), [
    ("3c351", 0.3715, "SBS 1704+608", "[HB89] 1704+608 ABS01"),
    ("b1422", 3.62, "SDSS J142438.10+225600.9", "[PBW92] B1422+231 ABS01"),
])
def test_quasar_redshift_is_not_its_absorbers(key: str, z_true: float, ned_id: str, absorber: str,
                                              tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport(key, tmp_path_factory)
    z = result["redshift"]
    assert z["kind"] == "spec" and z["reliable"] is True
    assert z["value"] == pytest.approx(z_true, abs=0.01)
    ned = members(result, "ned")
    assert ned["source_id"] == ned_id and absorber in ned["reason"]
    assert not any(c["source"] == "ned" and "ABS" in str(c["id"]) for c in z["candidates"])
    assert result["classification"]["label"] == "qso"


# ---------------------------------------------------------------------------
# 2. X-ray/optical ratio of extended counterparts; explicit score ties
# ---------------------------------------------------------------------------


def test_cen_a_xray_ratio_uses_total_light_and_is_not_a_qso(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("cena", tmp_path_factory)
    line = evidence_line(result, "log(fX/fV)")
    # 3.87 + (6.84 - 21.14) / 2.5 = -1.85: the galaxy's integrated V, not the G = 21.1 knot.
    assert "log(fX/fV) = -1.85" in line and "SIMBAD V = 6.84" in line and "Gaia G" not in line
    assert "not diagnostic" in line
    cls = result["classification"]
    assert cls["label"] in {"galaxy", "agn"}, cls
    if cls["tied"]:
        assert cls["label"] == "galaxy" and "host-galaxy classes preferred" in evidence(result)


def test_score_ties_are_explicit() -> None:
    ev = [sed.Evidence("a", {"qso": 3.0}), sed.Evidence("b", {"galaxy": 3.0})]
    label, conf, scores, tied = sed._score(ev)
    assert (label, tied) == ("qso", ["qso", "galaxy"]) and conf == scores["qso"]
    assert sed._score(ev, sed.EXTENDED_TIE_ORDER)[0] == "galaxy"
    assert sed._score([sed.Evidence("c", {"star": 1.0})])[3] == []
    assert sed._score([sed.Evidence("d", {})])[0] == "unknown"


def test_xray_ratio_not_evaluated_without_total_light() -> None:
    allwise = member_choice("allwise", {"ext_flg": 5})
    gaia = member_choice("gaia_dr3", {"phot_g_mean_mag": 21.14})
    g = sed._point_from_flux(band="G", facility="Gaia DR3", catalog="gaia_dr3", source_id="1", wavelength_um=0.58,
                             flux_jy=1e-5, flux_err_jy=1e-6, magnitude=21.14)
    xray = sed._point_from_flux(band="0.2-12 keV", facility="XMM-Newton EPIC", catalog="xmm", source_id="x",
                                wavelength_um=0.0005, flux_jy=1e-6, flux_err_jy=1e-7)
    xray.regime = "xray"
    ev = [e.text for e in sed.gather_evidence([g, xray], [allwise, gaia], {})]
    assert any("X-ray/optical ratio not evaluated" in t for t in ev)
    assert not any("log(fX/fV)" in t for t in ev)


# ---------------------------------------------------------------------------
# 3. AllWISE ext_flg = 1 is a poor PSF fit, not morphology
# ---------------------------------------------------------------------------


def test_allwise_ext_flg_semantics() -> None:
    assert sed.optical_extent([member_choice("allwise", {"ext_flg": 1})]) == (None, "morphology unknown")
    assert sed.optical_extent([member_choice("allwise", {"ext_flg": 0})])[0] is False
    for flag in (2, 3, 4, 5):
        extended, why = sed.optical_extent([member_choice("allwise", {"ext_flg": flag})])
        assert extended is True and "2MASS XSC" in why
    ev = sed.gather_evidence([], [member_choice("allwise", {"ext_flg": 1})], {})
    (item,) = ev
    assert "not used as morphology" in item.text and item.weights == {}
    (item,) = sed.gather_evidence([], [member_choice("allwise", {"ext_flg": 4})], {})
    assert "association" in item.text and item.weights == {"galaxy": 0.5, "agn": 0.25}


def test_pks2155_point_like_bl_lac_is_not_treated_as_extended(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("pks2155", tmp_path_factory)
    text = evidence(result)
    assert "ext_flg = 1: profile-fit chi2 > 3" in text
    assert "extended counterpart" not in text and "half weight" not in text
    assert "F_opt,total" not in text  # the point-source radio-loudness branch is used
    assert result["classification"]["label"] == "qso"


# ---------------------------------------------------------------------------
# 4. Gaia proper motions of quasars (excess noise, physical threshold)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "aen_sig"), [("3c279", 6.4), ("pks0118", 7.8)])
def test_blazar_spurious_proper_motion_is_not_galactic(key: str, aen_sig: float,
                                                       tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport(key, tmp_path_factory)
    assert result["gaia_extra"]["astrometric_excess_noise_sig"] == pytest.approx(aen_sig, abs=0.05)
    line = evidence_line(result, "Gaia DR3 proper motion")
    assert "moving, Galactic" not in line and "proper motion not used" in line
    detail = next(d for d in result["classification"]["evidence_detail"] if d["text"] == line)
    assert detail["weights"] == {}
    assert result["classification"]["label"] == "qso"


def _gaia_case(pm: float, aen_sig: float | None) -> list[sed.Evidence]:
    gaia = member_choice("gaia_dr3", {"pmra": pm, "pmdec": 0.0, "parallax": 0.1, "parallax_error": 0.1,
                                      "phot_g_mean_mag": 17.0})
    extra = {"pmra_error": 0.01, "pmdec_error": 0.01, "parallax_over_error": 1.0, "ruwe": 1.0,
             "astrometric_excess_noise_sig": aen_sig}
    return sed.gather_evidence([], [gaia], {}, extra)


def test_proper_motion_rule_thresholds() -> None:
    strong = next(e for e in _gaia_case(2.0, 0.0) if "proper motion" in e.text)
    assert "moving, Galactic" in strong.text and strong.weights["star"] == 3.0
    weak = next(e for e in _gaia_case(0.5, 0.0) if "proper motion" in e.text)
    assert "weak evidence" in weak.text and weak.weights["star"] == 1.0
    tiny = next(e for e in _gaia_case(0.19, 0.0) if "proper motion" in e.text)  # 5.5 sigma with the 33 uas/yr floor
    assert "very weak" in tiny.text and tiny.weights["star"] == 0.5
    noisy = next(e for e in _gaia_case(2.0, 5.0) if "proper motion" in e.text)
    assert "not used" in noisy.text and noisy.weights == {}
    fast = next(e for e in _gaia_case(20.0, 5.0) if "proper motion" in e.text)  # far above spurious motions
    assert "moving, Galactic" in fast.text


# ---------------------------------------------------------------------------
# 5. WISE saturation and Rayleigh-Jeans consistency
# ---------------------------------------------------------------------------


def _allwise_at(row: dict[str, Any], ra: float | None, dec: float | None) -> sed.MemberChoice:
    choice = member_choice("allwise", row)
    choice.ra, choice.dec = ra, dec
    return choice


# Ecliptic longitudes (BarycentricMeanEcliptic): 51 Peg 354.3 deg (post-cryo range 280.6-48.1), Polaris 88.6 deg
# (outside both post-cryo ranges 89.4-221.8 and 280.6-48.1).
PEG51 = (344.367, 20.769)
POLARIS = (37.95, 89.26)


def test_wise_saturation_limits() -> None:
    """Profile-fit w?mpro are reliable up to 2.0, 1.5, -3.0, -4.0 mag (All-Sky ES VI.3.d); the 8/7/3.8/0.4 limits
    are the APERTURE-photometry saturation limits (AllWISE ES II.2.c.iii). W1 < 8 / W2 < 7 is a caveat only at
    post-cryo ecliptic longitudes (AllWISE ES II.2.c.i)."""
    assert sed.ecliptic_longitude_deg(*PEG51) == pytest.approx(354.3, abs=0.2)
    assert sed.ecliptic_longitude_deg(*POLARIS) == pytest.approx(88.6, abs=0.2)
    row = {"w1mpro": 7.9, "w1sigmpro": 0.02, "w2mpro": 6.9, "w2sigmpro": 0.02, "w3mpro": 0.58, "w3sigmpro": 0.02,
           "w4mpro": -3.9, "w4sigmpro": 0.02, "ph_qual": "AAAA"}
    outside = {p.band: p for p in sed.extract_points([_allwise_at(row, *POLARIS)], offline_filters())}
    # Polaris-like W3 = 0.58 and W4 = -3.9 are valid profile-fit measurements (brighter than 3.8 / 0.4, fainter than
    # the profile-fit limits -3.0 / -4.0); W1 < 8 and W2 < 7 outside the post-cryo longitudes get a note only.
    assert not any(p.quality_warning for p in outside.values()), {b: p.warnings for b, p in outside.items()}
    assert any("outside the post-cryo" in n for n in outside["W1"].notes)
    inside = {p.band: p for p in sed.extract_points([_allwise_at(row, *PEG51)], offline_filters())}
    assert inside["W1"].quality_warning and "post-cryo" in inside["W1"].warnings[0]
    assert inside["W2"].quality_warning and not inside["W3"].quality_warning and not inside["W4"].quality_warning
    unknown = {p.band: p for p in sed.extract_points([_allwise_at(row, None, None)], offline_filters())}
    assert not unknown["W1"].quality_warning and any("could not be checked" in n for n in unknown["W1"].notes)
    # Beyond the profile-fit limits: flagged wherever the source is (boundaries 2.0, 1.5, -3.0, -4.0).
    bright = {"w1mpro": 1.99, "w2mpro": 1.49, "w3mpro": -3.01, "w4mpro": -4.01, "ph_qual": "AAAA"}
    edge = {"w1mpro": 2.01, "w2mpro": 1.51, "w3mpro": -2.99, "w4mpro": -3.99, "ph_qual": "AAAA"}
    flagged = {p.band: p for p in sed.extract_points([_allwise_at(bright, *POLARIS)], offline_filters())}
    fine = {p.band: p for p in sed.extract_points([_allwise_at(edge, *POLARIS)], offline_filters())}
    for band in ("W1", "W2", "W3", "W4"):
        assert flagged[band].quality_warning and "VI.3.d" in flagged[band].warnings[0], band
        assert not fine[band].quality_warning, (band, fine[band].warnings)
    # A 'U' limit at a saturated brightness is a failed extraction, not a flux limit (nominal limits 8, 7, 3.8, 0.4).
    limit = {p.band: p for p in sed.extract_points([_allwise_at(dict(row, ph_qual="UUUU"), *POLARIS)],
                                                   offline_filters())}
    assert all(limit[b].is_upper_limit and limit[b].quality_warning for b in ("W1", "W2", "W3", "W4"))
    faint_limit = {p.band: p for p in sed.extract_points(
        [_allwise_at({"w1mpro": 8.1, "w2mpro": 7.1, "w3mpro": 3.9, "w4mpro": 0.5, "ph_qual": "UUUU"}, *POLARIS)],
        offline_filters())}
    assert not any(p.quality_warning for p in faint_limit.values())


def test_wise_limit_below_rayleigh_jeans_extrapolation() -> None:
    tmass = member_choice("twomass_psc", {"j_m": 9.5, "j_msigcom": 0.02, "h_m": 9.2, "h_msigcom": 0.02, "k_m": 9.0,
                                          "k_msigcom": 0.02, "ph_qual": "AAA"})
    wise = member_choice("allwise", {"w1mpro": 12.0, "w2mpro": 8.9, "w2sigmpro": 0.03, "ph_qual": "UA"})
    pts = {p.band: p for p in sed.extract_points([tmass, wise], offline_filters())}
    # Ks from the SVO Vega zero point; RJ bound at W1 = F_Ks (2.159 / 3.3526)^2; W1 < 309.54 * 10^-4.8.
    ks = 666.8 * 10 ** (-0.4 * 9.0)
    bound = ks * (2.159 / 3.3526) ** 2
    w1 = pts["W1"]
    assert w1.is_upper_limit and w1.flux_jy == pytest.approx(309.54 * 10 ** -4.8)
    assert bound / w1.flux_jy > sed.WISE_RJ_MARGIN
    assert w1.quality_warning and f"{bound / w1.flux_jy:.0f}x below the Rayleigh-Jeans extrapolation" in w1.warnings[0]
    # W2 = 171.787 * 10^-3.56 = 0.047 Jy vs the Ks bound 0.037 Jy (W1 is a limit, not a reference): consistent.
    assert not pts["W2"].quality_warning


# Expected WISE flags of the recorded bright giants (see the AllWISE rows in tests/fixtures/sed/records/<key>.json):
# Aldebaran W1/W2/W3 are 'U' limits at saturated brightnesses (5.61, 1.27, -2.78); its W4 = -2.93 (ph_qual A) is a
# valid profile-fit measurement (fainter than -4.0) consistent with the Rayleigh-Jeans tail of Ks = -3.04:
# 11.0 kJy x (2.159 / 22.088)^2 = 105 Jy vs 8.363 x 10^(0.4 x 2.927) = 124 Jy. Betelgeuse W1 = -1.37 and W4 = -4.98
# are brighter than the profile-fit limits 2.0 / -4.0; its W2/W3 are saturated 'U' limits.
BRIGHT_STAR_WISE: dict[str, dict[str, str | None]] = {
    "aldebaran": {"W1": "'U' upper limit", "W2": "'U' upper limit", "W3": "'U' upper limit", "W4": None},
    "betelgeuse": {"W1": "VI.3.d", "W2": "'U' upper limit", "W3": "'U' upper limit", "W4": "VI.3.d"},
}


@pytest.mark.parametrize("key", ["aldebaran", "betelgeuse"])
def test_bright_star_wise_points_are_flagged(key: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport(key, tmp_path_factory)
    for band, reason in BRIGHT_STAR_WISE[key].items():
        p = point(result, "WISE", band)
        if reason is None:
            assert not p["quality_warning"] and not p["is_upper_limit"], (band, p["warnings"])
        else:
            assert p["quality_warning"] and any(reason in w for w in p["warnings"]), (band, p["warnings"])
    if key == "aldebaran":
        w4, ks = point(result, "WISE", "W4"), point(result, "2MASS", "Ks")
        rj = ks["flux_jy"] * (ks["wavelength_um"] / w4["wavelength_um"]) ** 2
        assert w4["flux_jy"] == pytest.approx(124.0, rel=0.02) and rj == pytest.approx(105.0, rel=0.03)
    assert point(result, "WISE", "W1")["flux_jy"] < 0.05 * point(result, "2MASS", "Ks")["flux_jy"]
    assert "WISE W1-W2" not in evidence(result)  # saturated bands feed no colour rule
    assert result["classification"]["label"] == "star"


def test_saturated_w1_does_not_feed_the_stern_rule(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("ngc4472", tmp_path_factory)  # W1 = 7.54 < 8
    assert point(result, "WISE", "W1")["quality_warning"]
    assert "Stern et al. 2012" not in evidence(result)


# ---------------------------------------------------------------------------
# 6. Stern et al. 2012 applicability
# ---------------------------------------------------------------------------


def test_stern_criterion_not_applicable_for_faint_w2(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("sdssj1030", tmp_path_factory)
    line = evidence_line(result, "WISE W1-W2")
    assert "W1-W2 = 1.01" in line and "W2 = 15.57 >= 15.05" in line and "not applied" in line
    assert "< 0.8" not in line
    assert result["redshift"]["value"] == pytest.approx(6.3, abs=0.01)
    assert result["classification"]["label"] == "qso"


# ---------------------------------------------------------------------------
# 7. Redshift priority: NED values of unstated technique
# ---------------------------------------------------------------------------


def test_ned_unknown_technique_beats_flagged_and_unverified_values() -> None:
    ned = {"value": 0.003272, "kind": None, "source": "ned", "reliable": True}
    simbad_e = {"value": 0.00317, "kind": "spec", "source": "simbad", "reliable": False}
    photo_unverified = {"value": 0.05, "kind": "photo", "source": "sdss_photoz", "reliable": None}
    photo_reliable = {"value": 0.05, "kind": "photo", "source": "sdss_photoz", "reliable": True}
    assert sed.choose_redshift([simbad_e, photo_unverified, ned])["value"] == 0.003272
    best = sed.choose_redshift([simbad_e, ned])
    assert (best["source"], best["kind"], best["reliable"]) == ("ned", None, True)
    assert sed.choose_redshift([ned, photo_reliable])["source"] == "sdss_photoz"


def test_ngc4472_takes_ned_redshift(tmp_path_factory: pytest.TempPathFactory) -> None:
    z = passport("ngc4472", tmp_path_factory)["redshift"]
    assert (z["source"], z["kind"], z["reliable"]) == ("ned", None, True)
    assert z["value"] == pytest.approx(0.003272, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. S5-HVS1 ceiling is a field-star heuristic (not near Sgr A*)
# ---------------------------------------------------------------------------


def test_galactic_centre_stars_are_not_redshift_conflicts() -> None:
    fast = {"value": 0.01}  # cz ~ 3000 km/s
    assert sed.near_galactic_centre((266.41683, -29.00781))  # S2, 0.1 arcsec from Sgr A*
    assert not sed.near_galactic_centre((266.5, -29.0))  # ~4 arcmin away
    assert not sed.redshift_star_conflict(fast, (266.41683, -29.00781))
    assert sed.redshift_star_conflict(fast, (10.0, 10.0)) and sed.redshift_star_conflict(fast)
    star = member_choice("simbad", {"otype": "*"})
    spec = {"value": 0.01, "kind": "spec", "reliable": True, "source": "simbad"}
    near = [e.text for e in sed.gather_evidence([], [star], spec, position=(266.41683, -29.00781))]
    assert any("S5-HVS1 ceiling is not applied" in t for t in near)
    far = [e.text for e in sed.gather_evidence([], [star], spec, position=(10.0, 10.0))]
    assert any("fastest known field/hypervelocity star" in t and "heuristic" in t for t in far)


# ---------------------------------------------------------------------------
# 9. Malformed SVO answers / cache entries never break the SED
# ---------------------------------------------------------------------------


def _svo_body(filter_id: str) -> bytes:
    (exchange,) = [e for e in load_exchanges("sed/svo") if f"ID={filter_id.replace('/', '%2F')}" in e.url
                   or f"ID={filter_id}" in e.url]
    return exchange.content


@pytest.mark.parametrize(("edit", "message"), [
    (lambda b: b.replace(b'<PARAM name="ZeroPoint" value="666.8"', b'<PARAM name="ZeroPointX" value="666.8"'),
     "no positive finite ZeroPoint"),
    (lambda b: b.replace(b"2MASS/2MASS.Ks/Vega", b"2MASS/2MASS.Ks/AB").replace(b'value="Vega"', b'value="AB"'),
     "PhotCalID"),
    (lambda b: b.replace(b'name="ZeroPointUnit" value="Jy"', b'name="ZeroPointUnit" value="erg/cm2/s/A"'),
     "ZeroPointUnit"),
])
async def test_malformed_svo_answer_is_rejected_not_cached(tmp_path: Path, edit: Any, message: str) -> None:
    body = edit(_svo_body("2MASS/2MASS.Ks"))
    assert body != _svo_body("2MASS/2MASS.Ks")
    cache = tmp_path / "svo.json"
    catalog = sed.FilterCatalog(cache_path=cache)
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(200, content=body))
        async with offline_client() as client:
            failures = await catalog.prefetch(["2MASS/2MASS.Ks"], client)
    assert message in failures["2MASS/2MASS.Ks"]
    assert not cache.exists()
    info = catalog.info("2MASS/2MASS.Ks")
    assert info.metadata_origin == "embedded" and info.zero_point_jy == pytest.approx(666.8)
    fresh = sed.FilterCatalog(cache_path=cache, offline=True).info("2MASS/2MASS.Ks")
    assert fresh.metadata_origin == "embedded"


@pytest.mark.parametrize("entry", [
    {"WavelengthEff": 21590.0},
    {"WavelengthEff": 21590.0, "ZeroPoint": 666.8, "PhotCalID": "2MASS/2MASS.Ks/AB"},
    {"WavelengthEff": 21590.0, "ZeroPoint": "NaN", "PhotCalID": "2MASS/2MASS.Ks/Vega"},
    {"WavelengthEff": -1, "ZeroPoint": 666.8, "PhotCalID": "2MASS/2MASS.Ks/Vega"},
])
def test_invalid_cache_entry_is_ignored(tmp_path: Path, entry: dict[str, Any]) -> None:
    cache = tmp_path / "svo.json"
    cache.write_text(json.dumps({"filters": {"2MASS/2MASS.Ks": entry}}), encoding="utf-8")
    info = sed.FilterCatalog(cache_path=cache, offline=True).info("2MASS/2MASS.Ks")
    assert info.metadata_origin == "embedded" and info.zero_point_jy == pytest.approx(666.8)


def test_invalid_cache_entry_does_not_break_the_router(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    (tmp_path / "svo.json").write_text(json.dumps({"filters": {"2MASS/2MASS.Ks": {"WavelengthEff": 21590.0}}}),
                                       encoding="utf-8")
    record = asyncio.run(replay_record("3c273"))
    exchanges = load_exchanges("sed/3c273")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(make_app(FakeService(record), tmp_path)) as client:
            res = client.get("/api/v1/sed", params={"ra": 187.2779154, "dec": 2.0523883})
    assert res.status_code == 200, res.text
    ks = point(res.json(), "2MASS", "Ks")
    assert ks["zero_point_jy"] == pytest.approx(666.8) and ks["zero_point_origin"] == "embedded"


def test_unusable_stored_entry_is_dropped() -> None:
    catalog = offline_filters()
    catalog._disk_loaded = True
    catalog._memory["2MASS/2MASS.Ks"] = ({"WavelengthEff": 21590.0}, "svo")
    info = catalog.info("2MASS/2MASS.Ks")
    assert info.metadata_origin == "embedded" and "2MASS/2MASS.Ks" not in catalog._memory
    assert "unusable" in catalog.errors["2MASS/2MASS.Ks"]
    with pytest.raises(sed.SEDFilterError):
        catalog.info("NOPE/NOPE.x")


def test_embedded_table_passes_validation() -> None:
    for fid, params in sed.EMBEDDED_FILTERS.items():
        sed.validate_svo_params(params, fid)


# ---------------------------------------------------------------------------
# 10. A hanging SVO never blocks a request for long; in-flight fetches are shared
# ---------------------------------------------------------------------------


HANG_CALLS: list[str] = []


async def _hang(request: httpx.Request) -> httpx.Response:
    HANG_CALLS.append(str(request.url))
    await asyncio.sleep(30.0)
    return httpx.Response(504)


async def test_hanging_svo_is_bounded_by_the_deadline(tmp_path: Path) -> None:
    ids = sorted(sed.FILTER_SYSTEMS)
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json", deadline_seconds=0.3, timeout=30.0)
    try:
        HANG_CALLS.clear()
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.route().mock(side_effect=_hang)
            async with offline_client() as client:
                start = time.monotonic()
                first, second = await asyncio.gather(catalog.prefetch(ids, client), catalog.prefetch(ids, client))
                assert time.monotonic() - start < 3.0
                assert set(first) == set(ids) and all("deadline" in msg for msg in first.values())
                assert set(second) == set(ids)
                third = await catalog.prefetch(ids, client)  # still in flight: shared, no new request
                assert set(third) == set(ids)
                # 4 concurrent slots: only the first 4 fetches reached the network; nothing was requested twice.
                assert 0 < len(HANG_CALLS) <= catalog.max_concurrency and len(set(HANG_CALLS)) == len(HANG_CALLS)
                assert catalog.info("2MASS/2MASS.Ks").metadata_origin == "embedded"
                for task in list(catalog._inflight.values()):
                    task.cancel()
                await asyncio.sleep(0)
    finally:
        for task in list(catalog._inflight.values()):
            task.cancel()


async def test_owned_client_fetches_are_cancelled_at_the_deadline(tmp_path: Path) -> None:
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json", deadline_seconds=0.2, retry_after_seconds=3600.0)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_hang)
        start = time.monotonic()
        failures = await catalog.prefetch(["GAIA/GAIA3.G", "WISE/WISE.W1"])
        assert time.monotonic() - start < 3.0
    assert set(failures) == {"GAIA/GAIA3.G", "WISE/WISE.W1"} and not catalog._inflight
    assert "deadline" in catalog.errors["GAIA/GAIA3.G"]
    again = await catalog.prefetch(["GAIA/GAIA3.G"])  # back-off: no new attempt
    assert "retry pending" in again["GAIA/GAIA3.G"]


async def test_concurrent_prefetches_share_fetches(tmp_path: Path) -> None:
    ids = ["2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks", "WISE/WISE.W1"]
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        route = router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            results = await asyncio.gather(*(catalog.prefetch(ids, client) for _ in range(3)))
        assert route.call_count == len(ids)
    assert results == [{}, {}, {}]
    assert all(catalog.info(fid).metadata_origin == "svo" for fid in ids)


# ---------------------------------------------------------------------------
# 11. Router: non-finite JSON input and upstream error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ['{"ra": NaN, "dec": 0}', '{"ra": 10, "dec": Infinity}',
                                  '{"ra": 10, "dec": 0, "radius_arcsec": NaN}'])
def test_post_non_finite_json_is_422(tmp_path: Path, body: str) -> None:
    from fastapi.testclient import TestClient

    with TestClient(make_app(FakeService({}), tmp_path)) as client:
        res = client.post("/api/v1/sed", content=body, headers={"content-type": "application/json"})
    assert res.status_code == 422
    assert isinstance(res.json()["detail"], list)


class RaisingService:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def crossmatch(self, *args: Any, **kwargs: Any) -> UnifiedRecord:
        raise self.exc


@pytest.mark.parametrize(("exc", "status"), [
    (CatalogUnavailableError("gaia down"), 502),
    (ResponseParseError("garbage"), 502),
    (AstroSearchError("generic"), 502),
    (TimeoutError(), 502),
    (sed.SEDFilterError("No Vega zero point for 2MASS/2MASS.Ks"), 502),
    (sed.SEDError("bug"), 500),
    (sed.SEDInputError("bad input"), 422),
])
def test_router_maps_every_failure(tmp_path: Path, exc: BaseException, status: int) -> None:
    from fastapi.testclient import TestClient

    with TestClient(make_app(RaisingService(exc), tmp_path)) as client:
        res = client.get("/api/v1/sed", params={"ra": 10, "dec": 20})
    assert res.status_code == status, res.text
    assert res.json()["detail"]


# ---------------------------------------------------------------------------
# 12. Live-test skips are reserved for outages
# ---------------------------------------------------------------------------


def _failed_record(error_type: str) -> dict[str, Any]:
    return {"target": {"ra": 1.0, "dec": 2.0}, "catalogs_queried": 2, "catalog_results": {}, "counterparts": {},
            "failures": [{"catalog": "gaia_dr3", "status": "failed", "error_type": error_type},
                         {"catalog": "simbad", "status": "failed", "error_type": error_type}],
            "provenance": {"catalog_stats": {"gaia_dr3": {"status": "failed"}, "simbad": {"status": "failed"}}},
            "crossmatch_groups": []}


async def test_live_build_fails_on_parse_errors_and_skips_on_outages(tmp_path: Path) -> None:
    from test_sed_live import build

    with respx.mock(assert_all_mocked=True):
        with pytest.raises(sed.SEDUpstreamError) as info:
            await build(tmp_path, ra=1.0, dec=2.0, service=FakeService(_failed_record("ResponseParseError")))
        assert [f["error_type"] for f in info.value.failures] == ["ResponseParseError"] * 2
        with pytest.raises(pytest.skip.Exception):
            await build(tmp_path, ra=1.0, dec=2.0, service=FakeService(_failed_record("CatalogUnavailableError")))


# ---------------------------------------------------------------------------
# 13. Missing values never take the favourable default
# ---------------------------------------------------------------------------


async def test_null_zwarning_is_unknown_not_reliable() -> None:
    rows = [{"specObjID": "1", "dist_arcsec": 0.1, "ra": 1.0, "dec": 2.0, "z": 0.5, "zErr": 1e-4, "zWarning": None,
             "class": "QSO", "subClass": "", "survey": "sdss", "sciencePrimary": 1}]
    payload = [{"TableName": "Table1", "Rows": rows}]
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith=sed.SDSS_SQL_URL).mock(return_value=httpx.Response(200, json=payload))
        async with offline_client() as client:
            (cand,) = await sed.fetch_sdss_redshifts(client, 1.0, 2.0, include_photoz=False)
    assert cand["reliable"] is None and cand["z_warning"] is None
    assert sed.choose_redshift([cand])["reliable"] is None
    sdss = member_choice("sdss", {"specz": 0.5})
    extra = {"spectra": [{"specObjID": "1", "z": 0.5, "zWarning": None}]}
    (member,) = sed.member_redshift_candidates([sdss], sdss_extra=extra)
    assert member["reliable"] is None and "zWarning" in member["note"]
    # A flagged cone row paired with the member spectrum keeps its reliability (False), never coerced.
    flagged = dict(cand, reliable=False, z_warning=4)
    (member2,) = sed.member_redshift_candidates([sdss], specobj_candidates=[flagged])
    assert member2["reliable"] is False


def test_members_without_separation_are_never_nearest() -> None:
    no_sep = {"catalog": "gaia_dr3", "source_id": "bright", "separation_arcsec": None,
              "data": {"phot_g_mean_mag": 12.0}}
    near = {"catalog": "gaia_dr3", "source_id": "target", "separation_arcsec": 0.5, "data": {"phot_g_mean_mag": 18.0}}
    (choice,) = sed.select_members({"members": [no_sep, near]})
    assert choice.source_id == "target" and choice.used
    (alone,) = sed.select_members({"members": [no_sep]})
    assert not alone.used and alone.separation_arcsec is None and "no separation" in alone.reason
    assert json.dumps(alone.summary())  # JSON-safe (no inf)
    radio = {"catalog": "nvss", "source_id": "N1", "separation_arcsec": None, "data": {"flux_20_cm": 500.0}}
    (rchoice,) = sed.select_members({"members": [radio]}, (1.0, 2.0))
    assert not rchoice.used and rchoice.components == []


# ---------------------------------------------------------------------------
# 14. CLI table and plot show quality flags
# ---------------------------------------------------------------------------


def test_cli_table_shows_flags_and_reasons(tmp_path_factory: pytest.TempPathFactory) -> None:
    from test_sed_regressions import _replay_canary

    result = asyncio.run(_replay_canary("3c273", tmp_path_factory))
    table = sed.format_sed_table(result)
    header = next(line for line in table.splitlines() if "facility" in line)
    assert header.rstrip().endswith("flags")
    simbad_v = next(line for line in table.splitlines() if "SIMBAD (literature)" in line and " V " in line)
    assert simbad_v.rstrip().endswith("Q")
    assert "Q = quality warning" in table
    assert any(line.startswith("  Q #") and "SIMBAD V = 14.83" in line for line in table.splitlines())
    assert any(line.startswith("  Q #") and "pileup" in line for line in table.splitlines())
    lotss = next(line for line in table.splitlines() if "LoTSS" in line)
    assert lotss.rstrip().endswith("Q")


def test_plot_styles_are_unique_per_facility(tmp_path: Path) -> None:
    styles = list(sed._FACILITY_STYLE.values())
    assert len(set(styles)) == len(styles)  # (colour, marker) pairs never repeat
    facilities = {facility for facility, _ in list(sed.MAG_SPECS.values()) + list(sed.RADIO_SPECS.values())
                  + list(sed.XRAY_SPECS.values())} | {"GALEX", "SIMBAD (literature)"}
    assert facilities <= set(sed._FACILITY_STYLE)
    sed_dict = {"points": [
        {"facility": "SDSS", "band": "i", "wavelength_um": 0.75, "nu_fnu_erg_s_cm2": 1e-11, "is_upper_limit": False,
         "quality_warning": True, "nu_fnu_err_erg_s_cm2": 1e-12},
        {"facility": "SDSS", "band": "r", "wavelength_um": 0.62, "nu_fnu_erg_s_cm2": 1e-11, "is_upper_limit": False,
         "quality_warning": False, "nu_fnu_err_erg_s_cm2": None},
        {"facility": "WISE", "band": "W4", "wavelength_um": 22.0, "nu_fnu_erg_s_cm2": 1e-12, "is_upper_limit": True,
         "quality_warning": False},
    ], "target": {"ra": 1.0, "dec": 2.0}, "classification": {"label": "qso", "confidence": 0.9}, "redshift": {}}
    out = sed.plot_sed(sed_dict, tmp_path / "p.png")
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_cli_parser_unchanged() -> None:
    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sed", "--name", "3C 351", "--radius", "5"])
    assert args.handler is sed.run_cli and args.radius == 5.0 and math.isfinite(args.radius)
