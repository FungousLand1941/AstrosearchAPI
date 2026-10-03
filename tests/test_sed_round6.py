"""Round-6 review regressions for sed.py (offline).

1. A photo-z or an unverified spectrum never outranks catalogue-vetted redshifts (Hercules A: photo-z 0.134 vs NED and
   SIMBAD 0.155); the 1% discordance threshold is pinned at its boundary.
2. A small or negative cz is possible for a galaxy/AGN (M31, NGC 4395): the discordance resolver never rejects it.
3. The redshift cross-check survives a failed or skipped SIMBAD lookup (PKS 1510-089, Q2237+030): TAP lookups retry
   transient failures, and a member redshift that could not be vetted still takes part in the cross-check.
4. A quasar/blazar-typed EXTENDED counterpart whose nucleus is not quasar-luminous is an AGN (Cygnus A at 10 arcsec).
5. The X-ray/optical ratio is not AGN evidence for a star known from SIMBAD astrometry only (Procyon B).
6. An X-ray row closer to another (brighter) Gaia source than to the target is flagged (Gl 229B vs Gl 229A).
7. A refused Photoz query no longer discards the SpecObj redshifts.
8. The configured Sesame endpoint is used (with the 'I' option), and records resolved without identifiers get them.
9. SVO fetches never outlive their caller or its client; the disk cache is written once per batch, without leaking
   temporary files.
10. Rate limits (429 / Retry-After) are backed off; the CLI reports an unwritable plot path.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client
from test_sed import RECORD_TARGETS, load_record
from test_sed_regressions import evidence, member_choice, passport
from test_sed_round3 import record, row, sed_of, used
from test_sed_round5 import SESAME, _cand, point_of, replay_with, resolution

import sed
from models import offset_radec, propagate_radec

SIMBAD_HOST = "simbad.cds.unistra.fr"


@pytest.fixture(autouse=True)
def _fresh_backoff() -> Any:
    sed.default_backoff().clear()
    yield
    sed.default_backoff().clear()


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sed, "TAP_RETRY_DELAYS", (0.01, 0.01))
    monkeypatch.setattr(sed, "SDSS_CONNECT_RETRY_DELAYS", (0.01, 0.01))


def _handler(key: str, override: Any = None) -> Any:
    """Replay the recorded supplementary requests of ``key``; ``override(request)`` may answer or raise first."""
    replay = replay_side_effect(load_exchanges(f"sed/{key}") + load_exchanges("sed/svo"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if override is not None:
            answer = await override(request)
            if answer is not None:
                return answer
        return replay(request)

    return handler


async def _replay(key: str, tmp_path: Path, override: Any = None, rec: dict[str, Any] | None = None,
                  **kwargs: Any) -> dict[str, Any]:
    filters = sed.FilterCatalog(cache_path=tmp_path / f"svo-{key}.json")
    kwargs.setdefault("backoff", sed.HostBackoff())
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_handler(key, override))
        async with offline_client() as client:
            return await sed.sed_from_record(rec if rec is not None else load_record(key), client=client,
                                             filters=filters, name=RECORD_TARGETS[key].get("name"), **kwargs)


def _simbad_down(status: int = 503) -> Any:
    async def override(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(status) if request.url.host == SIMBAD_HOST else None
    return override


# ---------------------------------------------------------------------------
# 1. Photo-z and unverified spectra vs vetted values; the 1% threshold
# ---------------------------------------------------------------------------


def test_hercules_a_photoz_never_outranks_agreeing_catalogues() -> None:
    """Reviewer reproduction: photo-z 0.133519 (photoErrorClass 1) was adopted over NED 0.155 and SIMBAD 0.1549."""
    candidates = [_cand("sdss_photoz", 0.133519, kind="photo", reliable=True),
                  _cand("simbad", 0.1549, kind=None, reliable=True), _cand("ned", 0.155, kind=None, reliable=True)]
    z = sed.choose_redshift(candidates)
    assert (z["value"], z["source"], z["kind"], z["reliable"]) == (0.155, "ned", None, True)
    assert "discordant" not in z and "needs_resolution" not in z  # a photo-z never makes a discordance
    # Even a lone vetted catalogue value outranks a reliable photo-z.
    assert sed.choose_redshift(candidates[:1] + candidates[2:])["source"] == "ned"


def test_unverified_spectrum_never_outranks_a_corroborated_value() -> None:
    """An SDSS member spectrum of unknown zWarning (SkyServer lookup failed) ranks above NED values of unstated
    technique, but not when it disagrees with the value NED and SIMBAD agree on."""
    unverified = _cand("sdss", 0.30, kind="spec", reliable=None)
    ned, simbad = _cand("ned", 0.155, kind=None, reliable=True), _cand("simbad", 0.1549, kind=None, reliable=True)
    z = sed.choose_redshift([unverified, ned, simbad])
    assert (z["value"], z["source"], z["reliable"]) == (0.155, "ned", True)
    groups = {round(g["value"], 3): g for g in z["discordant"]}
    assert groups[0.3]["vetted"] is False and groups[0.3]["corroboration"] is None
    assert groups[0.155]["vetted"] is True and groups[0.155]["corroboration"] == "agreement of ned and simbad"
    assert "quality unknown" in z["resolution"] and "not vetted" in z["resolution"]
    # Against a single vetted value neither is corroborated: the classification must decide.
    single = sed.choose_redshift([unverified, ned])
    assert single["needs_resolution"] is True and single["value"] == 0.30
    # When they agree, the (more precise) spectrum keeps its rank.
    agree = sed.choose_redshift([_cand("sdss", 0.1551, kind="spec", reliable=None), ned, simbad])
    assert agree["source"] == "sdss" and "discordant" not in agree


@pytest.mark.parametrize(("simbad_z", "discordant"), [
    (0.125, True),  # |dz|/(1+z) = 0.025/1.1 = 0.023 > 0.01
    (0.1112, True),  # 0.0112/1.1 = 0.0102: just above
    (0.1109, False),  # 0.0109/1.1 = 0.0099
    (0.105, False),  # 0.005/1.1 = 0.0045
])
def test_discordance_threshold_boundary(simbad_z: float, discordant: bool) -> None:
    z = sed.choose_redshift([_cand("ned", 0.100), _cand("simbad", simbad_z)])
    assert ("discordant" in z) is discordant
    assert (z.get("needs_resolution") is True) is discordant
    if not discordant:
        assert z["value"] == 0.100 and z["reliable"] is True


def test_discordance_tolerance_constant() -> None:
    assert sed.REDSHIFT_DISCORD_TOLERANCE == 0.01
    assert sed._redshifts_agree(0.1109, 0.1) and not sed._redshifts_agree(0.1111, 0.1)


# ---------------------------------------------------------------------------
# 2. Local Volume galaxies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("z", "label", "verdict"), [
    (-0.001001, "galaxy", None),  # M31, cz = -300 km/s
    (0.00106, "galaxy", None),  # cz = 318 km/s
    (0.00110586, "agn", None),  # NGC 4395 (SIMBAD), cz = 332 km/s
    (0.0018, "agn", None),  # Cen A, cz = 547 km/s
    (-0.001, "agn", None),
    (0.0476, "galaxy", True),
    (-0.01, "galaxy", False),  # cz = -3000 km/s: no galaxy approaches that fast
])
def test_small_cz_is_possible_for_galaxies(z: float, label: str, verdict: bool | None) -> None:
    ok, why = sed.redshift_class_consistency(z, label, [], [])
    assert ok is verdict, why
    if verdict is None:
        assert "cannot be ruled out" in why


def test_ngc4395_true_redshift_is_not_rejected() -> None:
    """Reviewer reproduction: SIMBAD 0.00110586 (true) vs NED 0.0476; the resolver adopted 0.0476 as reliable."""
    simbad = member_choice("simbad", {"otype": "Sy2"}, "NGC 4395")
    ned = member_choice("ned", {"prefphytype": "G", "z": 0.0476}, "NGC 4395")
    chosen = sed.choose_redshift([_cand("simbad", 0.00110586), _cand("ned", 0.0476)])
    assert chosen["needs_resolution"] is True
    resolved = sed.resolve_discordant_redshift(chosen, [], [simbad, ned])
    assert resolved["reliable"] is False and resolved["conflict"] is True
    assert not (resolved["value"] == 0.0476 and resolved["reliable"] is True)
    assert "cannot be ruled out" in resolved["resolution"] and "impossible" not in resolved["resolution"]


# ---------------------------------------------------------------------------
# 3. The cross-check without SIMBAD's quality: retries, back-off, unvetted values
# ---------------------------------------------------------------------------


async def test_pks1510_transient_simbad_error_is_retried(tmp_path: Path) -> None:
    """Reviewer reproduction: one RemoteProtocolError from SIMBAD TAP turned PKS 1510-089 back into NED's z = 0.0068."""
    calls: list[int] = []

    async def once(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == SIMBAD_HOST:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        return None

    result = await _replay("pks1510", tmp_path, once)
    z = result["redshift"]
    assert len(calls) == 2  # retried once
    assert z["value"] == pytest.approx(0.35637, abs=1e-4) and (z["source"], z["kind"], z["reliable"]) == ("simbad", "spec", True)
    assert not any("simbad lookup failed" in n for n in result["notes"])


@pytest.mark.parametrize("failure", ["http503", "backoff", "unsupplemented"])
async def test_pks1510_without_simbad_quality_is_never_neds_z_reliable(tmp_path: Path, failure: str,
                                                                       fast_retries: None) -> None:
    """SIMBAD TAP answering 503 (after its retries), SIMBAD in back-off, or no supplementary lookups: the SIMBAD row
    still says z = 0.356, so NED's z = 0.0068 is never reported as reliable."""
    backoff = sed.HostBackoff()
    kwargs: dict[str, Any] = {"backoff": backoff}
    override = None
    if failure == "http503":
        override = _simbad_down()
    elif failure == "backoff":
        backoff.record_failure(SIMBAD_HOST, "ReadTimeout after 60 s")
    else:
        kwargs["supplementary"] = False
    result = await _replay("pks1510", tmp_path, override, **kwargs)
    z = result["redshift"]
    assert result["simbad_extra"] is None
    assert not (z["value"] == pytest.approx(0.006815) and z["reliable"] is True), z
    assert z["reliable"] is False and z["conflict"] is True
    groups = {round(g["value"], 3): g for g in z["discordant"]}
    assert set(groups) == {0.007, 0.356} and groups[0.356]["vetted"] is False and groups[0.007]["vetted"] is True
    assert "cross-check incomplete: z = 0.35637 (simbad) could not be vetted" in z["note"]
    assert "only z = 0.35637 (simbad) is consistent with the 'qso' classification, but its catalogue quality could " \
           "not be checked" in z["resolution"]
    text = evidence(result)
    assert "below the quasar luminosity threshold" not in text  # the round-2 critical output
    assert result["classification"]["label"] == "qso"


async def test_q2237_without_simbad_is_still_a_conflict(tmp_path: Path, fast_retries: None) -> None:
    result = await _replay("q2237", tmp_path, _simbad_down())
    z = result["redshift"]
    assert z["reliable"] is False and z["conflict"] is True
    assert sorted(round(g["value"], 3) for g in z["discordant"]) == [0.039, 1.695]
    assert "cross-check incomplete" in z["note"]


async def test_tap_lookups_retry_transient_failures(fast_retries: None) -> None:
    payload = (sed_fixture("pks1510", "simbad_extra.0.body")).read_bytes()
    answers = iter([httpx.ReadError("reset"), httpx.Response(502), httpx.Response(200, content=payload)])

    def flaky(request: httpx.Request) -> httpx.Response:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(sed.SIMBAD_TAP_URL).mock(side_effect=flaky)
        async with offline_client() as client:
            extra = await sed.fetch_simbad_extra(client, "QSO J1512-0906")
    assert route.call_count == 3 and extra is not None and extra["rvz_qual"] == "C"
    # Three failures: the last one is raised, with the HTTP status for the back-off.
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(sed.GAIA_TAP_URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "0"}))
        async with offline_client() as client:
            with pytest.raises(sed.SEDUpstreamError) as info:
                await sed.fetch_gaia_extra(client, "123")
    assert route.call_count == 3 and info.value.status_code == 429 and info.value.retry_after == 0.0
    # A deterministic client error is not retried.
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(sed.GAIA_TAP_URL).mock(return_value=httpx.Response(400))
        async with offline_client() as client:
            with pytest.raises(sed.SEDUpstreamError):
                await sed.fetch_gaia_extra(client, "123")
    assert route.call_count == 1


def sed_fixture(key: str, name: str) -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sed" / key / name


# ---------------------------------------------------------------------------
# 4. Quasar-typed extended counterparts (Cygnus A at the default radius)
# ---------------------------------------------------------------------------


def _cyga_within(radius: float) -> dict[str, Any]:
    rec = copy.deepcopy(load_record("cyga"))
    for group in rec["crossmatch_groups"]:
        group["members"] = [m for m in group["members"] if (m.get("separation_arcsec") or 1e9) <= radius]
    rec["crossmatch_groups"] = [g for g in rec["crossmatch_groups"] if g["members"]]
    rec["provenance"]["query_radius_arcsec"] = radius
    return rec


async def test_cygnus_a_at_the_default_radius_is_not_a_quasar(tmp_path: Path) -> None:
    """Reviewer reproduction: at 10 arcsec (no radio lobes) SIMBAD's 'Bla' made Cygnus A a 'qso' (0.622)."""
    result = await replay_with("cyga", _cyga_within(sed.DEFAULT_RADIUS_ARCSEC), tmp_path, name="Cygnus A")
    cls = result["classification"]
    assert cls["label"] == "agn", cls["scores"]
    assert cls["scores"]["qso"] < cls["scores"]["galaxy"] < cls["scores"]["agn"]
    detail = next(d for d in cls["evidence_detail"] if d["text"].startswith("SIMBAD object type 'Bla'"))
    assert "extended" in detail["text"] and "nucleus is not quasar-luminous" in detail["text"]
    assert "fainter than the quasar limit -22" in detail["text"]
    assert detail["weights"] == {"agn": 3.0, "qso": 1.0, "galaxy": 1.0}


def test_nuclear_quasar_luminosity() -> None:
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": 17.0, "iMeanKronMag": 16.0})
    z_cyg = {"value": 0.0565, "kind": "spec", "reliable": True}
    ok, why = sed.nuclear_quasar_luminosity([], [ps1], z_cyg)
    assert ok is False and "PS1 iPSF = 17.00" in why
    # A lensed quasar (Q2237+030: iKron 14.2 at z = 1.695) can be quasar-luminous: the SIMBAD type keeps its weight.
    lensed = member_choice("panstarrs_dr2", {"iMeanPSFMag": 15.0, "iMeanKronMag": 14.2})
    assert sed.nuclear_quasar_luminosity([], [lensed], {"value": 1.695, "kind": "spec", "reliable": True})[0] is True
    # Unknown or unusable redshift, saturated PSF photometry: cannot tell.
    assert sed.nuclear_quasar_luminosity([], [ps1], {"value": 0.0565, "kind": "spec", "reliable": False})[0] is None
    assert sed.nuclear_quasar_luminosity([], [ps1], {"value": 0.0565, "kind": "photo", "reliable": True})[0] is None
    saturated = member_choice("panstarrs_dr2", {"iMeanPSFMag": 12.0, "iMeanKronMag": 11.0})
    assert sed.nuclear_quasar_luminosity([], [saturated], z_cyg)[0] is None
    # The extended-source gate never touches point-like counterparts or unknown redshifts.
    simbad = member_choice("simbad", {"otype": "Bla"})
    allwise = member_choice("allwise", {"ext_flg": 5})
    for members, z in (([simbad, ps1, allwise], {"value": None}),
                       ([simbad, member_choice("allwise", {"ext_flg": 0})], z_cyg)):
        line = next(e for e in sed.gather_evidence([], members, z) if e.text.startswith("SIMBAD object type"))
        assert line.weights == {"qso": 3.0, "agn": 1.0}, line.text
    capped = next(e for e in sed.gather_evidence([], [simbad, ps1, allwise], z_cyg)
                  if e.text.startswith("SIMBAD object type"))
    assert capped.weights == {"agn": 3.0, "qso": 1.0, "galaxy": 1.0}
    ned = member_choice("ned", {"prefphytype": "QSO"})
    ned_line = next(e for e in sed.gather_evidence([], [ned, ps1, allwise], z_cyg) if e.text.startswith("NED"))
    assert ned_line.weights == {"agn": 2.5} and "not quasar-luminous" in ned_line.text


# ---------------------------------------------------------------------------
# 5. X-ray/optical ratio of stars known from SIMBAD astrometry only
# ---------------------------------------------------------------------------


def test_procyon_b_xray_ratio_is_not_agn_evidence() -> None:
    """No Gaia member; SIMBAD pm/parallax make it Galactic, so the ROSAT blend with Procyon A is not AGN-like."""
    simbad = member_choice("simbad", {"otype": "WD*", "pmra": -714.6, "pmdec": -1036.8, "plx_value": 284.6},
                           "Procyon B")
    v = sed._point_from_flux(band="V", facility="SIMBAD (literature)", catalog="simbad", source_id="Procyon B",
                             wavelength_um=0.55, flux_jy=3631.0 * 10 ** (-0.4 * 10.92), flux_err_jy=None,
                             magnitude=10.92)
    rosat = sed._point_from_flux(band="0.1-2.4 keV", facility="ROSAT PSPC", catalog="rosat",
                                 source_id="2RXS J073918.3+051332", wavelength_um=0.0012, flux_jy=1e-5, flux_err_jy=None)
    rosat.regime = "xray"
    line = next(e for e in sed.gather_evidence([v, rosat], [simbad], {}) if e.text.startswith("log(fX/fV)"))
    assert line.weights == {} and "not diagnostic" in line.text and "large proper motion" in line.text
    ratio = float(line.text.split("=")[1].split()[0])
    assert ratio > -1.0  # the ratio itself is 'AGN-like'; it is simply not evidence for a star
    # Without the stellar astrometry the same ratio is AGN evidence.
    bare = next(e for e in sed.gather_evidence([v, rosat], [member_choice("simbad", {"otype": "WD*"})], {})
                if e.text.startswith("log(fX/fV)"))
    assert bare.weights == {"agn": 1.5, "qso": 1.0, "star": -1.0}


# ---------------------------------------------------------------------------
# 6. X-ray rows of a brighter neighbour (Gl 229B)
# ---------------------------------------------------------------------------

GL229_PM = (-137.0, -714.0)  # Gl 229 A/B common proper motion (mas/yr)


def _gl229_record(xray_near: str) -> dict[str, Any]:
    """Gl 229B (target, J2000 + pm) with Gl 229A 7.8 arcsec away at PA 343 deg (Gaia DR3 row at 2016.0) and a CSC
    source at the 2010 position of A (``xray_near='A'``) or of B."""
    b_ra, b_dec = 92.64391, -21.86697
    a_ra, a_dec = offset_radec(b_ra, b_dec, -2.28, 7.46)
    a2016 = propagate_radec(a_ra, a_dec, *GL229_PM, 2000.0, 2016.0)
    src = (a_ra, a_dec) if xray_near == "A" else (b_ra, b_dec)
    x_ra, x_dec = offset_radec(*propagate_radec(*src, *GL229_PM, 2000.0, 2010.0), 0.3, 0.2)
    span = (1999.5, 2022.0)
    x_sep = sed._closest_approach_arcsec((x_ra, x_dec), b_ra, b_dec, 2000.0, GL229_PM, span)
    chandra = {"catalog": "chandra", "source_id": "2CXO J061034.2-215201", "ra": x_ra, "dec": x_dec,
               "separation_arcsec": x_sep, "positional_error_arcsec": 2.39, "epoch_range": list(span), "metadata": {},
               "data": {"b_flux_ap": 2e-13, "b_flux_ap_lo": 1.8e-13, "b_flux_ap_hi": 2.2e-13}}
    simbad = row("simbad", "Gl 229B", 0.0, {"otype": "BD*", "pmra": GL229_PM[0], "pmdec": GL229_PM[1],
                                            "plx_value": 173.6})
    simbad.update(ra=b_ra, dec=b_dec)
    gaia_a = {"catalog": "gaia_dr3", "source_id": "2940856402123426176", "ra": a2016[0], "dec": a2016[1],
              "epoch": 2016.0, "separation_arcsec": 7.8, "positional_error_arcsec": 0.02, "metadata": {},
              "data": {"phot_g_mean_mag": 7.4, "pmra": GL229_PM[0], "pmdec": GL229_PM[1], "ref_epoch": 2016.0,
                       "parallax": 173.6, "parallax_error": 0.02}}
    rec = record({"group_id": "object-1", "contains_target": True, "members": [simbad, chandra]},
                 {"group_id": "object-2", "members": [gaia_a]})
    rec["target"] = {"ra": b_ra, "dec": b_dec, "epoch": 2000.0, "pm_ra_masyr": GL229_PM[0],
                     "pm_dec_masyr": GL229_PM[1]}
    return rec


def test_gl229b_does_not_get_gl229a_xrays() -> None:
    result = sed_of(_gl229_record("A"))
    chandra = used(result, "chandra")
    assert chandra["separation_arcsec"] <= chandra["tolerance_arcsec"]  # within tolerance, yet A's
    (warning,) = chandra["warnings"]
    assert "Gaia DR3 2940856402123426176 (G = 7.40)" in warning and "not attributed to the target" in warning
    assert point_of(result, "Chandra", "0.5-7 keV")["quality_warning"]
    assert any(n.startswith("X-ray source closer to another optical source") for n in result["notes"])
    assert not [e for e in result["classification"]["evidence"] if e.startswith("log(fX/fV)")]


def test_xray_row_at_the_target_is_kept() -> None:
    result = sed_of(_gl229_record("B"))
    assert "warnings" not in used(result, "chandra")
    assert not point_of(result, "Chandra", "0.5-7 keV")["quality_warning"]


def test_fainter_neighbour_does_not_take_the_xrays() -> None:
    rec = _gl229_record("A")
    target_gaia = {"catalog": "gaia_dr3", "source_id": "1", "ra": rec["target"]["ra"], "dec": rec["target"]["dec"],
                   "epoch": 2000.0, "separation_arcsec": 0.01, "positional_error_arcsec": 0.02, "metadata": {},
                   "data": {"phot_g_mean_mag": 6.0, "pmra": GL229_PM[0], "pmdec": GL229_PM[1], "ref_epoch": 2000.0}}
    rec["crossmatch_groups"][0]["members"].append(target_gaia)
    assert "warnings" not in used(sed_of(rec), "chandra")  # the neighbour (G = 7.4) is fainter than the target


# ---------------------------------------------------------------------------
# 7. SDSS: a refused Photoz query keeps the SpecObj redshifts
# ---------------------------------------------------------------------------

_SPEC = [{"TableName": "Table1", "Rows": [{"specObjID": "123", "dist_arcsec": 0.1, "ra": 1, "dec": 1, "z": 0.3564,
                                           "zErr": 1e-4, "zWarning": 0, "class": "QSO", "subClass": "",
                                           "survey": "sdss", "sciencePrimary": 1}]}]


async def test_photoz_failure_keeps_specobj(fast_retries: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "Photoz" in str(request.url):
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return httpx.Response(200, json=_SPEC)

    notes: list[str] = []
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith=sed.SDSS_SQL_URL).mock(side_effect=handler)
        async with offline_client() as client:
            out = await sed.fetch_sdss_redshifts(client, 1.0, 1.0, lock=sed.SkyServerLock(), notes=notes)
    (spec,) = out
    assert (spec["value"], spec["reliable"], spec["z_warning"], spec["spec_class"]) == (0.3564, True, 0, "QSO")
    (note,) = notes
    assert note.startswith("SDSS Photoz query failed (ConnectError") and "SpecObj redshifts kept" in note


async def test_photoz_failure_in_a_passport(tmp_path: Path, fast_retries: None) -> None:
    async def refuse_photoz(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == "skyserver.sdss.org" and "Photoz" in str(request.url):
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return None

    result = await _replay("pks1510", tmp_path, refuse_photoz)
    assert any(n.startswith("SDSS Photoz query failed") for n in result["notes"])
    assert not any(n.startswith("sdss lookup failed") for n in result["notes"])
    assert result["redshift"]["value"] == pytest.approx(0.35637, abs=1e-4)


# ---------------------------------------------------------------------------
# 8. Sesame: the configured endpoint, identifiers of records resolved without them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("configured", "expected"), [
    ("https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV", sed.SESAME_ALIASES_ENDPOINT),
    ("https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxp/SNV",
     "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxpI/SNV"),
    ("https://cds.unistra.fr/cgi-bin/nph-sesame/-ox/S", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxI/S"),
    ("https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV"),
    ("https://cds.unistra.fr/cgi-bin/nph-sesame/SNV", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV"),
    ("http://127.0.0.1:9000/resolve", "http://127.0.0.1:9000/resolve"),
])
def test_sesame_aliases_endpoint(monkeypatch: pytest.MonkeyPatch, configured: str, expected: str) -> None:
    assert sed.sesame_aliases_endpoint(configured) == expected
    monkeypatch.setenv("SESAME_ENDPOINT", configured)
    assert sed.sesame_aliases_endpoint() == expected


async def test_resolve_name_uses_the_configured_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    mirror = "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxp/SNV"
    monkeypatch.setenv("SESAME_ENDPOINT", mirror)
    payload = (SESAME / "t8main.xml").read_bytes()
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get(url__startswith="https://vizier.cfa.harvard.edu/").mock(
            return_value=httpx.Response(200, content=payload))
        async with offline_client() as client:
            info = await sed.resolve_name("2MASSI J0415195-093506", client)
    assert route.call_count == 1
    assert str(route.calls.last.request.url).startswith("https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxpI/SNV?")
    assert "2MASS J04151954-0935066" in info["resolved"]["aliases"]


def _app_resolved_t8() -> dict[str, Any]:
    """t8main as the app's own name crossmatch records it: Sesame '-oxp' (no identifiers asked, none listed)."""
    rec = copy.deepcopy(load_record("t8main"))
    rec["resolved_object"] = resolution("t8main")
    rec["resolved_object"]["aliases"] = []
    rec["resolved_object"]["resolver_metadata"]["endpoint"] = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"
    return rec


async def test_record_resolved_without_identifiers_gets_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESAME_ENDPOINT", raising=False)
    payload = (SESAME / "t8main.xml").read_bytes()
    asked: list[str] = []

    async def sesame(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == "cds.unistra.fr":
            asked.append(str(request.url))
            return httpx.Response(200, content=payload)
        return None

    result = await _replay("t8main", tmp_path, sesame, rec=_app_resolved_t8())
    assert len(asked) == 1 and asked[0].startswith(sed.SESAME_ALIASES_ENDPOINT + "?")
    tmass = used(result, "twomass_psc")
    assert tmass["designation"] == "2MASS J04151954-0935066"
    assert any(n.startswith("identifiers of '2MASSI J0415195-093506' fetched from Sesame '-oxpI'") for n in result["notes"])
    assert "2MASS J04151954-0935066" in result["target"]["resolved_object"]["aliases"]


async def test_resolution_that_asked_for_identifiers_is_not_repeated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESAME_ENDPOINT", raising=False)
    rec = _app_resolved_t8()
    rec["resolved_object"]["resolver_metadata"]["endpoint"] = sed.SESAME_ALIASES_ENDPOINT
    asked: list[str] = []

    async def sesame(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == "cds.unistra.fr":
            asked.append(str(request.url))
            return httpx.Response(500)
        return None

    result = await _replay("t8main", tmp_path, sesame, rec=rec)
    assert asked == [] and not any("identifiers of" in n for n in result["notes"])


async def test_failed_identifier_lookup_is_reported(tmp_path: Path, fast_retries: None,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESAME_ENDPOINT", raising=False)

    async def down(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(503) if request.url.host == "cds.unistra.fr" else None

    result = await _replay("t8main", tmp_path, down, rec=_app_resolved_t8())
    assert any(n.startswith("aliases lookup failed: ResolverUnavailableError") for n in result["notes"]), result["notes"]
    assert not next(m for m in result["members"] if m["catalog"] == "twomass_psc")["used"]


async def test_hanging_identifier_lookup_leaves_time_for_the_others(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESAME_ENDPOINT", raising=False)

    async def hang(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == "cds.unistra.fr":
            await asyncio.sleep(30)
        return None

    started = time.monotonic()
    result = await _replay("t8main", tmp_path, hang, rec=_app_resolved_t8(), deadline_seconds=2.0)
    assert time.monotonic() - started < 5.0
    assert any(n.startswith("aliases lookup failed: TimeoutError (no answer from cds.unistra.fr within the 1 s")
               for n in result["notes"]), result["notes"]
    assert result["simbad_extra"] is not None  # the other lookups still ran
    assert not any(n.startswith(("simbad lookup failed", "sdss lookup failed")) for n in result["notes"])


# ---------------------------------------------------------------------------
# 9. SVO fetches and the disk cache
# ---------------------------------------------------------------------------


def _svo_fetches() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if not t.done() and t.get_coro().__qualname__ == "FilterCatalog._fetch"]


def _hanging_svo() -> Any:
    replay = replay_side_effect(load_exchanges("sed/t8dwarf"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == sed._host(sed.SVO_FPS_URL):
            await asyncio.sleep(30)
            return httpx.Response(504)
        return replay(request)

    return handler


async def test_cancelled_sed_cancels_its_svo_fetches(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Reviewer reproduction (probe_orphans A): 9 FilterCatalog._fetch tasks survived asyncio.wait_for's cancel."""
    caplog.set_level(logging.ERROR, logger="asyncio")
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json", deadline_seconds=6.0)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_hanging_svo())
        for client in (None, offline_client()):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(sed.sed_from_record(load_record("t8dwarf"), client=client, filters=filters,
                                                           deadline_seconds=20.0, backoff=sed.HostBackoff()), 0.5)
            assert _svo_fetches() == []
            if client is not None:
                await client.aclose()
    # The caller went away: that is not an SVO failure, nothing is backed off.
    assert filters.errors == {} and filters._inflight == {}
    # (C) The next request (fresh client, SVO answering) fetches normally instead of joining a dead fetch.
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            failures = await filters.prefetch(["WISE/WISE.W1", "2MASS/2MASS.J"], client)
    assert failures == {} and filters.info("WISE/WISE.W1").metadata_origin == "svo"
    assert "Task exception was never retrieved" not in caplog.text


async def test_supplementary_deadline_leaves_no_svo_fetch(tmp_path: Path) -> None:
    """Reviewer reproduction (probe_orphans B): the SED returned while 9 fetches ran on the client it had closed."""
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json", deadline_seconds=6.0)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_hanging_svo())
        started = time.monotonic()
        result = await sed.sed_from_record(load_record("t8dwarf"), filters=filters, deadline_seconds=0.5,
                                           backoff=sed.HostBackoff())
        assert time.monotonic() - started < 3.0
        assert _svo_fetches() == []
    notes = result["notes"]
    assert any(n.startswith("filters lookup failed: TimeoutError (no answer from svo2.cab.inta-csic.es within the "
                            "0.5 s supplementary deadline") for n in notes), notes
    per_filter = [n for n in notes if n.startswith("SVO FPS lookup for ")]
    assert per_filter and all("embedded SVO values used" in n for n in per_filter)
    assert {n.split()[4] for n in per_filter} == set(filters.unresolved(sed.required_filters(
        sed.select_record_members(load_record("t8dwarf"))[0])))
    # The request's deadline IS an SVO timeout (told apart from a caller's cancellation by the cancel message):
    # recorded, retried later.
    assert filters.errors and all(e == "TimeoutError: no SVO answer within the request's supplementary deadline"
                                  for e in filters.errors.values()), filters.errors


async def test_background_fetch_on_a_closed_client_is_not_an_svo_failure(tmp_path: Path) -> None:
    """A background fetch (caller's client) that only starts after the caller closed its client fails with httpx's
    'client has been closed': not recorded as an SVO failure (no 60 s retry window)."""
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json", deadline_seconds=0.2, max_concurrency=1)
    release = asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(504)

    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=slow)
        client = offline_client()
        failures = await catalog.prefetch(["WISE/WISE.W1", "WISE/WISE.W2"], client)
        assert set(failures) == {"WISE/WISE.W1", "WISE/WISE.W2"}
        assert all("still fetching in the background" in f for f in failures.values())
        await client.aclose()
        release.set()
        for _ in range(50):
            if not catalog._inflight:
                break
            await asyncio.sleep(0.02)
    assert not catalog._inflight
    # The first fetch got SVO's 504 (an SVO failure); the second started on the closed client (not one).
    assert "WISE/WISE.W2" not in catalog.errors, catalog.errors
    assert "HTTP 504" in catalog.errors.get("WISE/WISE.W1", "")


async def test_disk_cache_written_once_per_batch_and_temp_files_removed(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks", "WISE/WISE.W1"]
    real_replace = os.replace
    replaced: list[str] = []

    def counting(src: str, dst: str) -> None:
        replaced.append(str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(sed.os, "replace", counting)
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            assert await catalog.prefetch(ids, client) == {}
    assert len(replaced) == 1  # one write for the batch of four answers (was one per filter)
    assert set(json.loads((tmp_path / "svo.json").read_text(encoding="utf-8"))["filters"]) == set(ids)

    # A failed os.replace (Windows sharing violation) removes its temporary file; the next batch retries the write.
    def denied(src: str, dst: str) -> None:
        raise PermissionError(5, "Access is denied", src)

    monkeypatch.setattr(sed.os, "replace", denied)
    denied_dir = tmp_path / "denied"
    catalog = sed.FilterCatalog(cache_path=denied_dir / "svo.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            assert await catalog.prefetch(ids[:2], client) == {}
    assert list(denied_dir.iterdir()) == [] and catalog._dirty
    monkeypatch.setattr(sed.os, "replace", real_replace)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/svo")))
        async with offline_client() as client:
            assert await catalog.prefetch(ids[2:], client) == {}
    assert [p.name for p in denied_dir.iterdir()] == ["svo.json"] and not catalog._dirty
    assert set(json.loads((denied_dir / "svo.json").read_text(encoding="utf-8"))["filters"]) == set(ids)


# ---------------------------------------------------------------------------
# 10. Rate limits; the CLI plot path
# ---------------------------------------------------------------------------


def test_rate_limits_are_backed_off() -> None:
    backoff = sed.HostBackoff(retry_after_seconds=120.0, refused_retry_after_seconds=60.0)
    limited = sed.SEDUpstreamError("SIMBAD TAP HTTP 429", status_code=429, retry_after=30.0)
    assert sed.backoff_seconds_for(backoff, limited, 0.1) == 30.0
    assert sed.backoff_seconds_for(backoff, sed.SEDUpstreamError("x", status_code=429), 0.1) == 60.0
    assert sed.backoff_seconds_for(backoff, sed.SEDUpstreamError("x", status_code=503, retry_after=90.0), 0.1) == 90.0
    assert sed.backoff_seconds_for(backoff, sed.SEDUpstreamError("x", status_code=503), 0.1) is None  # fast 5xx
    huge = sed.SEDUpstreamError("x", status_code=429, retry_after=86400.0)
    assert sed.backoff_seconds_for(backoff, huge, 0.1) == sed.SUPPLEMENTARY_MAX_RETRY_AFTER_SECONDS


async def test_rate_limited_archive_is_skipped_by_the_next_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sed, "TAP_RETRY_AFTER_MAX_SECONDS", 0.01)  # the in-request waits (Retry-After capped)
    backoff = sed.HostBackoff()
    notes: list[str] = []
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(sed.GAIA_TAP_URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "30"}))
        async with offline_client() as client:
            jobs = {"gaia": (sed._host(sed.GAIA_TAP_URL), lambda: sed.fetch_gaia_extra(client, "1"))}
            await sed._run_supplementary(jobs, deadline=5.0, backoff=backoff, notes=notes)
            calls = route.call_count
            await sed._run_supplementary(jobs, deadline=5.0, backoff=backoff, notes=notes)
    assert calls == 3 and route.call_count == calls  # the second request never contacts the archive
    assert notes[0].startswith("gaia lookup failed: SEDUpstreamError (gea.esac.esa.int") and "HTTP 429" in notes[0]
    assert notes[1].startswith("gaia lookup failed: skipped, gea.esac.esa.int HTTP 429 (Retry-After 30 s); back-off")


def test_retry_after_parsing() -> None:
    assert sed.retry_after_seconds(httpx.Response(429, headers={"Retry-After": "120"})) == 120.0
    assert sed.retry_after_seconds(httpx.Response(429)) is None
    assert sed.retry_after_seconds(httpx.Response(429, headers={"Retry-After": "soon"})) is None
    past = sed.retry_after_seconds(httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert past == 0.0
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    future = format_datetime(datetime.now(UTC) + timedelta(seconds=300), usegmt=True)
    assert 290.0 <= sed.retry_after_seconds(httpx.Response(503, headers={"Retry-After": future})) <= 300.0


def test_cli_reports_an_unwritable_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    result = sed_of(load_record("t8dwarf"))

    async def fake_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return result

    monkeypatch.setattr(sed, "build_sed", fake_build)
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory", encoding="utf-8")
    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sed", "--name", "x", "--plot", str(blocker / "sub" / "x.png")])
    assert args.handler(args) == 1
    out = capsys.readouterr().out
    assert "Error: cannot write plot" in out and "Traceback" not in out
    args = parser.parse_args(["sed", "--name", "x", "--plot", str(tmp_path / "x.unknownformat")])
    assert args.handler(args) == 1
    assert "Error: cannot write plot" in capsys.readouterr().out
    args = parser.parse_args(["sed", "--name", "x", "--plot", str(tmp_path / "ok.png")])
    assert args.handler(args) == 0 and (tmp_path / "ok.png").stat().st_size > 0


# ---------------------------------------------------------------------------
# Unchanged passports (the fixes must not move any recorded object)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "label", "z", "source"), [
    ("pks1510", "qso", 0.35637, "simbad"),
    ("3c351", "qso", 0.3715, "ned"),
    ("mrk421", "qso", 0.030021, "ned"),
    ("cena", "galaxy", 0.001877, "simbad"),
    ("ngc4472", "galaxy", 0.003272, "ned"),
    ("sn2006gy", "galaxy", 0.01919, "ned"),
    ("cyga", "agn", 0.056549, "simbad"),
])
def test_recorded_passports(tmp_path_factory: pytest.TempPathFactory, key: str, label: str, z: float,
                            source: str) -> None:
    result = passport(key, tmp_path_factory)
    assert result["classification"]["label"] == label, result["classification"]["scores"]
    assert result["redshift"]["value"] == pytest.approx(z, abs=1e-4) and result["redshift"]["source"] == source
    assert result["redshift"]["reliable"] is True and not result["redshift"].get("conflict")
    assert math.isfinite(result["classification"]["confidence"])

