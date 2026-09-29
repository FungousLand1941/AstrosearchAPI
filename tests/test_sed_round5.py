"""Round-5 review regressions for sed.py (offline).

1. Name resolution lists the object's identifiers (Sesame '-oxpI'), so designations identify its survey rows: the T8
   dwarf queried by SIMBAD's main identifier '2MASSI J0415195-093506' keeps its 2MASS row 2.53 arcsec away, and
   Scholz's star keeps Gaia DR3 3048443305671969152 2.08 arcsec away. Every name-resolved fixture record carries the
   resolution parsed from its recorded Sesame answer (tests/fixtures/sed/sesame/<key>.xml).
2. Discordant vetted redshifts are cross-checked: PKS 1510-089 is a z = 0.356 quasar (SIMBAD), not z = 0.0068 (NED);
   uncorroborated discordant values are reported as a conflict, never silently ranked.
3. Stacked X-ray catalogue fluxes of a moving source (5XMM rows of Proxima Cen) are flagged.
4. SIMBAD rvz_qual E is unreliable whatever rvz_nature says; radial velocities (rvz_type 'v') are spectroscopic.
5. Radio loudness counts as AGN evidence only with a radio excess over the mid-IR/radio correlation (q22).
6. Refused SkyServer connections are retried once per run and then backed off; a success clears a back-off; the
   default deadline and the process-wide back-off hold on the router path; cancelling a SED cancels its lookups.
7. Structurally malformed records are input errors.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import FIXTURES, load_exchanges, replay_side_effect
from helpers import offline_client
from test_sed import RECORD_TARGETS, FakeService, exchanges_for, load_record, make_app, replay_record
from test_sed_regressions import evidence, member_choice, passport
from test_sed_round3 import GAIA_ROW, TMASS_ROW, bands, offline_filters, record, row, sed_of, used

import sed
from providers import SesameResolver

SESAME = FIXTURES / "sed" / "sesame"


@pytest.fixture(autouse=True)
def _fresh_backoff() -> Any:
    sed.default_backoff().clear()
    yield
    sed.default_backoff().clear()


def resolution(key: str, query: str | None = None) -> dict[str, Any]:
    """The production resolution of a recorded Sesame -oxpI answer (SesameResolver.parse_response)."""
    payload = (SESAME / f"{key}.xml").read_text(encoding="utf-8")
    resolved = SesameResolver.parse_response(query or RECORD_TARGETS[key]["name"], payload,
                                             endpoint=sed.SESAME_ALIASES_ENDPOINT)
    return json.loads(json.dumps(resolved.as_dict(), default=str))


async def replay_with(key: str, rec: dict[str, Any], tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    """sed_from_record on ``rec`` with the supplementary requests recorded for ``key``."""
    filters = sed.FilterCatalog(cache_path=tmp_path / f"svo-{key}.json")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"sed/{key}") + load_exchanges("sed/svo")))
        async with offline_client() as client:
            return await sed.sed_from_record(rec, client=client, filters=filters, **kwargs)


def point_of(result: dict[str, Any], facility: str, band: str) -> dict[str, Any]:
    (hit,) = [p for p in result["points"] if p["facility"].startswith(facility) and p["band"] == band]
    return hit


# ---------------------------------------------------------------------------
# 1. Sesame identifiers identify the object's rows
# ---------------------------------------------------------------------------


async def test_resolve_name_asks_sesame_for_the_identifiers() -> None:
    payload = (SESAME / "t8main.xml").read_bytes()
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get(url__startswith="https://cds.unistra.fr/").mock(return_value=httpx.Response(200, content=payload))
        async with offline_client() as client:
            info = await sed.resolve_name("2MASSI J0415195-093506", client)
    url = str(route.calls.last.request.url)
    assert url.startswith("https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV?"), url
    resolved = info["resolved"]
    assert resolved["canonical_name"] == "2MASSI J0415195-093506"
    assert {"2MASS J04151954-0935066", "WISEA J041521.26-093500.4"} <= set(resolved["aliases"])
    assert resolved["resolver_metadata"]["endpoint"] == sed.SESAME_ALIASES_ENDPOINT


def test_recorded_records_carry_the_resolution_of_their_sesame_answer() -> None:
    """Every name-resolved fixture record holds exactly what production resolution gives (aliases included)."""
    with_aliases = 0
    for key, spec in RECORD_TARGETS.items():
        if not spec.get("name"):
            continue
        stored = load_record(key)["resolved_object"]
        assert stored == resolution(key), key
        assert stored["resolver_metadata"]["endpoint"] == sed.SESAME_ALIASES_ENDPOINT, key
        with_aliases += bool(stored["aliases"])
    assert with_aliases >= 30  # (SN 2006gy is resolved without any other identifier)


async def test_t8_dwarf_by_its_simbad_main_identifier_keeps_its_2mass_row(tmp_path: Path) -> None:
    """Reviewer reproduction: '2MASSI J0415195-093506' (SIMBAD main id) lost 2MASS J/H/Ks (2.529" > 1.5")."""
    rec = load_record("t8main")
    rec["resolved_object"] = resolution("t8main")  # parsed from the recorded -oxpI answer, not hand-written
    result = await replay_with("t8main", rec, tmp_path, name="2MASSI J0415195-093506")
    tmass = used(result, "twomass_psc")
    assert tmass["source_id"] == "04151954-0935066" and tmass["designation"] == "2MASS J04151954-0935066"
    assert tmass["separation_arcsec"] == pytest.approx(2.529, abs=0.01) and tmass["tolerance_arcsec"] == 1.5
    assert "is a name of the object resolved by Sesame" in tmass["reason"]
    assert bands(result, "2MASS") == {"J", "H", "Ks"}
    assert point_of(result, "2MASS", "J")["magnitude"] == pytest.approx(15.695, abs=1e-3)
    assert used(result, "allwise")["designation"] == "WISEA J041521.26-093500.4"
    assert result["classification"]["label"] == "star"
    # The same record resolved without identifiers (providers' default '-oxp' answer) loses the row: the aliases are
    # what identifies it.
    bare = copy.deepcopy(rec)
    bare["resolved_object"]["aliases"] = []
    lost = await replay_with("t8main", bare, tmp_path, name="2MASSI J0415195-093506")
    assert not next(m for m in lost["members"] if m["catalog"] == "twomass_psc")["used"]
    assert not bands(lost, "2MASS")


def test_scholzs_star_keeps_its_gaia_row(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("scholz", tmp_path_factory)
    gaia = used(result, "gaia_dr3")
    assert gaia["source_id"] == "3048443305671969152" and gaia["designation"] == "Gaia DR3 3048443305671969152"
    assert gaia["separation_arcsec"] == pytest.approx(2.079, abs=0.01) and gaia["separation_arcsec"] > gaia["tolerance_arcsec"]
    assert bands(result, "Gaia") == {"G", "BP", "RP"}
    assert result["gaia_extra"] is not None and result["classification"]["label"] == "star"
    assert "Gaia DR3 DSC-Combmod" in evidence(result)


def test_the_canonical_name_alone_identifies_a_row() -> None:
    """Sesame's main identifier is a name of the object even when the query and the aliases are something else."""
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("simbad", "2MASS J04151954-0935066", 0.0, {"otype": "BD*"})]}
    moved = {"group_id": "object-2", "members": [row("twomass_psc", "04151954-0935066", 2.53, TMASS_ROW, 0.11)]}
    resolved = {"query": "T8 dwarf", "canonical_name": "2MASS J04151954-0935066", "aliases": []}
    tmass = used(sed_of(record(target, moved, resolved=resolved)), "twomass_psc")
    assert tmass["designation"] == "2MASS J04151954-0935066"
    assert "is a name of the object resolved by Sesame; 2.530 arcsec > tolerance" in tmass["reason"]


# ---------------------------------------------------------------------------
# 2. Discordant redshifts
# ---------------------------------------------------------------------------


def _cand(source: str, value: float, kind: str | None = "spec", reliable: bool | None = True, **extra: Any) -> dict[str, Any]:
    return {"source": source, "value": value, "error": None, "kind": kind, "reliable": reliable, "id": source, **extra}


def test_pks_1510_is_the_z036_quasar(tmp_path_factory: pytest.TempPathFactory) -> None:
    """NED lists PKS 1510-089 at z = 0.006815 (6dF, SLS); SIMBAD at z = 0.356. The redshift-independent evidence says
    quasar, and only z = 0.356 makes it quasar-luminous (M_i = -25.4 vs -16.3)."""
    result = passport("pks1510", tmp_path_factory)
    z = result["redshift"]
    assert z["value"] == pytest.approx(0.35637, abs=1e-4) and (z["source"], z["kind"], z["reliable"]) == ("simbad", "spec", True)
    assert not z.get("conflict")
    groups = {round(g["value"], 6): g for g in z["discordant"]}
    assert set(groups) == {0.006815, 0.356369}
    assert "too faint for a quasar" in groups[0.006815]["consistency"]
    assert "quasar-luminous" in groups[0.356369]["consistency"]
    assert "adopted, consistent with the 'qso' classification" in z["resolution"]
    text = evidence(result)
    assert "M_i = -25.4 < -22" in text and "below the quasar luminosity threshold" not in text
    assert result["classification"]["label"] == "qso"
    assert any(n.startswith("discordant redshifts cross-checked") for n in result["notes"])


def test_uncorroborated_discordant_redshifts_are_a_conflict(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Q2237+030: quasar z = 1.695 (SIMBAD) and lens galaxy z = 0.0386 (NED); an extended galaxy-like counterpart
    allows both, so neither is adopted as reliable."""
    result = passport("q2237", tmp_path_factory)
    z = result["redshift"]
    assert z["reliable"] is False and z["conflict"] is True
    assert sorted(round(g["value"], 3) for g in z["discordant"]) == [0.039, 1.695]
    assert "unresolved: several values are consistent with the 'galaxy' classification" in z["resolution"]
    assert "reported by priority only and marked unreliable" in z["note"]
    assert "is flagged unreliable: redshift rules not applied" in evidence(result)


def test_choose_redshift_cross_checks_vetted_values() -> None:
    # Agreeing values (catalogue rounding) are not discordant.
    agree = sed.choose_redshift([_cand("ned", 0.01717), _cand("simbad", 0.01669)])
    assert agree["value"] == 0.01717 and "discordant" not in agree and "needs_resolution" not in agree
    # Two single-catalogue values: nothing corroborates either, the classification must decide.
    pks = sed.choose_redshift([_cand("simbad", 0.356), _cand("ned", 0.0068)])
    assert pks["needs_resolution"] is True and len(pks["discordant"]) == 2
    assert "none is corroborated" in pks["resolution"]
    # An SDSS zWarning = 0 spectrum corroborates its value.
    sdss = sed.choose_redshift([_cand("ned", 0.0068), _cand("sdss_specobj", 0.356, z_warning=0)])
    assert sdss["value"] == 0.356 and "needs_resolution" not in sdss and sdss["reliable"] is True
    assert "corroborated by an SDSS spectrum with zWarning = 0" in sdss["resolution"]
    # Two catalogues agreeing beat one higher-priority catalogue.
    two = sed.choose_redshift([_cand("ned", 0.30), _cand("sdss", 0.1003, reliable=True), _cand("simbad", 0.1)])
    assert two["value"] == pytest.approx(0.1003) and two["source"] == "sdss"
    assert "agreement of sdss and simbad" in two["resolution"] and "needs_resolution" not in two
    # Both corroborated: undecided here.
    both = sed.choose_redshift([_cand("ned", 0.1), _cand("simbad", 0.1), _cand("sdss_specobj", 0.5, z_warning=0)])
    assert both["needs_resolution"] is True and "each is corroborated" in both["resolution"]
    # Photo-z and flagged values never make a discordance.
    quiet = sed.choose_redshift([_cand("ned", 0.1), _cand("sdss_photoz", 0.5, kind="photo"),
                                 _cand("simbad", 2.0, reliable=False)])
    assert "discordant" not in quiet and quiet["value"] == 0.1


def test_discordance_settled_by_a_stellar_classification() -> None:
    """A star with a NED 'spectroscopic' z = 2.2 and a SIMBAD radial velocity: the velocity is the star's."""
    simbad = member_choice("simbad", {"otype": "PM*", "pmra": 800.0, "pmdec": -300.0, "plx_value": 50.0}, "HD 1")
    ned = member_choice("ned", {"prefphytype": "QSO", "z": 2.2}, "NED 1")
    chosen = sed.choose_redshift([_cand("ned", 2.2), _cand("simbad", 1.2e-4)])
    resolved = sed.resolve_discordant_redshift(chosen, [], [simbad, ned])
    assert resolved["value"] == 1.2e-4 and resolved["source"] == "simbad" and resolved["reliable"] is True
    assert "consistent with the 'star' classification" in resolved["resolution"] and "needs_resolution" not in resolved
    # Without class evidence the conflict is reported, the priority value kept but unreliable.
    open_case = sed.resolve_discordant_redshift(chosen, [], [])
    assert open_case["value"] == 2.2 and open_case["reliable"] is False and open_case["conflict"] is True
    assert "gives no single class" in open_case["resolution"]
    # A redshift without discordance passes through unchanged.
    plain = sed.choose_redshift([_cand("ned", 0.1)])
    assert sed.resolve_discordant_redshift(plain, [], []) == plain


def test_redshift_class_consistency() -> None:
    assert sed.redshift_class_consistency(1e-4, "star", [], [])[0] is True
    assert sed.redshift_class_consistency(0.01, "star", [], [])[0] is False
    assert sed.redshift_class_consistency(0.01, "star", [], [], sed.SGR_A_STAR_RADEC)[0] is True
    assert sed.redshift_class_consistency(0.01, "galaxy", [], [])[0] is True
    # Round-3 review: a small or negative cz is possible for a (Local Volume) galaxy or AGN: cannot be told (None),
    # never 'impossible' (this assertion used to pin False, which rejected NGC 4395's true redshift). See
    # test_sed_round6.py for the M31 / NGC 4395 cases.
    assert sed.redshift_class_consistency(1e-4, "agn", [], [])[0] is None
    assert sed.redshift_class_consistency(1e-4, "qso", [], [])[0] is False
    assert sed.redshift_class_consistency(0.5, "qso", [], [])[0] is None  # no i-band magnitude
    assert sed.redshift_class_consistency(0.5, "unknown", [], [])[0] is None


# ---------------------------------------------------------------------------
# 3. Stacked X-ray fluxes of moving sources
# ---------------------------------------------------------------------------


def xray_evidence(result: dict[str, Any]) -> list[str]:
    return [e for e in result["classification"]["evidence"] if e.startswith("log(fX/fV)")]


def test_proxima_5xmm_stack_flux_is_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("proxima", tmp_path_factory)
    xmm = point_of(result, "XMM-Newton", "0.2-12 keV")
    assert xmm["quality_warning"]
    warning = next(w for w in xmm["warnings"] if "moving source in a stacked catalogue" in w)
    assert "moved 64.0 arcsec" in warning and "16.6 yr" in warning and "3 xmm unique sources" in warning
    assert used(result, "xmm")["warnings"] == [warning]
    assert any(n.startswith("stacked X-ray catalogue flux of a moving source flagged") for n in result["notes"])
    assert not any("XMM-Newton" in e for e in xray_evidence(result))


async def test_proxima_diluted_5xmm_row_is_not_evidence(tmp_path: Path) -> None:
    """Reviewer reproduction: the chosen row was 5XMM J142941.9-624044 (ep_flux 6.9e-15 with ep_det_ml 1.7e6, sum_flag
    1), which gave log(fX/fV) = -5.42. Without the 2017 row in the record that row is chosen: it must be flagged."""
    rec = load_record("proxima")
    for group in rec["crossmatch_groups"]:
        group["members"] = [m for m in group["members"] if m["source_id"] != "5XMM J142933.5-624032"]
    result = await replay_with("proxima", rec, tmp_path, name="Proxima Centauri")
    member = used(result, "xmm")
    assert member["source_id"] == "5XMM J142941.9-624044"
    xmm = point_of(result, "XMM-Newton", "0.2-12 keV")
    assert xmm["quality_warning"] and xmm["warnings"] == member["warnings"]  # flagged only as a moving source
    assert not any("XMM-Newton" in e for e in xray_evidence(result)), xray_evidence(result)


def test_kapteyn_stacked_xray_fluxes_are_flagged(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Kapteyn's star (8.6 arcsec/yr): the 5XMM row spans 3.1 yr (26.8 arcsec of motion); CSC has four master sources
    of the star within tolerance (its CSC row carries no broad-band flux, so only the member is flagged)."""
    result = passport("kapteyn", tmp_path_factory)
    xmm = point_of(result, "XMM-Newton", "0.2-12 keV")
    assert any("moved 26.8 arcsec" in w for w in xmm["warnings"]), xmm["warnings"]
    chandra = used(result, "chandra")
    assert "4 chandra unique sources lie within tolerance" in chandra["warnings"][0]
    assert xray_evidence(result) == []  # was log(fX/fV) = -6.79 from the diluted 5XMM flux


def test_stationary_and_slow_sources_keep_their_stacked_flux() -> None:
    xmm = {"ep_flux": 1e-13, "ep_flux_error": 1e-15, "time": 52000.0, "end_time": 58000.0}  # 16.4 yr
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("gaia_dr3", "111", 0.0, dict(GAIA_ROW, pmra=200.0, pmdec=0.0), 0.001),
        row("xmm", "5XMM J1", 0.5, xmm, 0.5)]}
    # 200 mas/yr x 16.4 yr = 3.3 arcsec < the 6 arcsec EPIC PSF: not flagged.
    slow = point_of(sed_of(record(target)), "XMM-Newton", "0.2-12 keV")
    assert not slow["quality_warning"] and slow["warnings"] == []
    fast = copy.deepcopy(target)
    fast["members"][0]["data"]["pmra"] = 1000.0  # 16.4 arcsec over the stack
    flagged = sed_of(record(fast))
    xpoint = point_of(flagged, "XMM-Newton", "0.2-12 keV")
    assert xpoint["quality_warning"] and "Gaia DR3 111" in xpoint["warnings"][0]
    # A catalogue that states no span: flagged only when several of its unique sources are within tolerance.
    chandra = {"b_flux_ap": 1e-13}
    one = {"group_id": "object-1", "contains_target": True, "members": [
        row("chandra", "2CXO A", 0.1, chandra, 0.1)]}
    moving = record(one)
    moving["target"].update(pm_ra_masyr=8000.0, pm_dec_masyr=0.0)
    assert not any(p["quality_warning"] for p in sed_of(moving)["points"])
    two = copy.deepcopy(moving)
    two["crossmatch_groups"].append({"group_id": "object-2", "members": [row("chandra", "2CXO B", 0.2, chandra, 0.1)]})
    warned = point_of(sed_of(two), "Chandra", "0.5-7 keV")
    assert warned["quality_warning"] and "2 chandra unique sources lie within tolerance" in warned["warnings"][0]
    # No proper motion known: nothing to flag.
    assert sed.flag_moving_stacked_xray([], [], None) == []


# ---------------------------------------------------------------------------
# 4. SIMBAD quality E / null nature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("nature", "rvz_type", "qual", "kind", "reliable"), [
    (None, "v", "A", "spec", True),  # Sirius-like radial velocity
    (None, "v", "E", "spec", False),  # Sco X-1, NGC 7027, SS 433: quality E radial velocities
    (None, "z", "E", None, False),
    (None, "z", "D", None, True),
    (None, "c", None, None, None),
    ("s", "z", "C", "spec", True),
    ("p", "z", "E", "photo", False),
    ("s", "z", None, "spec", None),
])
def test_simbad_reliability_whatever_the_nature(nature: Any, rvz_type: str, qual: Any, kind: Any, reliable: Any) -> None:
    simbad = member_choice("simbad", {"rvz_redshift": 1e-4}, "X")
    extra = {"rvz_nature": nature, "rvz_type": rvz_type, "rvz_qual": qual, "rvz_err": 1.0, "rvz_bibcode": "b"}
    (cand,) = sed.member_redshift_candidates([simbad], simbad_extra=extra)
    assert (cand["kind"], cand["reliable"]) == (kind, reliable)
    chosen = sed.choose_redshift([cand])
    assert chosen["value"] == 1e-4 and chosen["reliable"] is reliable  # every combination has a rank


def test_quality_e_ranks_below_vetted_values() -> None:
    e_rv = _cand("simbad", 0.0003, reliable=False)
    assert sed.choose_redshift([e_rv, _cand("sdss_photoz", 0.2, kind="photo", reliable=True)])["source"] == "sdss_photoz"
    unknown = _cand("simbad", 0.003, kind=None, reliable=True)
    assert sed.choose_redshift([unknown, _cand("simbad", 0.1, kind="spec", reliable=False)])["value"] == 0.003


def test_scholz_radial_velocity_is_a_reliable_spectrum(tmp_path_factory: pytest.TempPathFactory) -> None:
    z = passport("scholz", tmp_path_factory)["redshift"]
    simbad = next(c for c in z["candidates"] if c["source"] == "simbad")
    assert (simbad["nature"], simbad["rvz_type"], simbad["quality"]) == (None, "v", "D")
    assert (simbad["kind"], simbad["reliable"]) == ("spec", True)


# ---------------------------------------------------------------------------
# 5. Radio loudness and the mid-IR/radio correlation
# ---------------------------------------------------------------------------


def test_arp_220_radio_excess_is_star_formation(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("arp220", tmp_path_factory)
    detail = next(d for d in result["classification"]["evidence_detail"] if d["text"].startswith("radio loudness"))
    assert "R = log10(F_1.4GHz/F_opt,total) = 1.51 > 1" in detail["text"]
    assert "q22 = log10(F_W4/F_1.4GHz) = 1.15 >= 0.5" in detail["text"] and "not counted" in detail["text"]
    assert detail["weights"] == {}
    assert result["classification"]["label"] == "galaxy"


def _wise_points(w1: float, w2: float, w3: float, w4: float | None, filters: sed.FilterCatalog) -> list[sed.SEDPoint]:
    data = {"w1mpro": w1, "w1sigmpro": 0.02, "w2mpro": w2, "w2sigmpro": 0.02, "w3mpro": w3, "w3sigmpro": 0.02,
            "w4mpro": w4, "w4sigmpro": 0.05 if w4 is not None else None, "ph_qual": "AAA" + ("A" if w4 is not None else "U")}
    member = member_choice("allwise", data, "AW")
    return [p for p in sed.extract_points([member], filters) if p.band != "W4" or w4 is not None]


def test_q22_decides_whether_radio_loudness_is_agn_evidence() -> None:
    filters = offline_filters()
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": 18.0, "iMeanPSFMagErr": 0.01, "iMeanKronMag": 18.0}, "PS1")
    optical = sed.extract_points([ps1], filters)

    def radio(flux_jy: float) -> sed.SEDPoint:
        return sed.SEDPoint(band="1.4 GHz", facility="FIRST (VLA)", catalog="first", source_id="F", wavelength_um=2.1e5,
                            frequency_hz=1.4e9, flux_jy=flux_jy, flux_err_jy=None, nu_fnu_erg_s_cm2=1.4e9 * flux_jy * 1e-23,
                            is_upper_limit=False, regime="radio")

    def radio_line(points: list[sed.SEDPoint]) -> sed.Evidence:
        ev = sed.gather_evidence(points, [ps1], {"value": None})
        return next(e for e in ev if e.text.startswith("radio loudness"))

    # Radio-loud quasar: W4 ~ 0.03 Jy, 1.4 GHz 0.5 Jy -> q22 = -1.2: AGN evidence kept.
    loud = radio_line(optical + _wise_points(12.0, 10.8, 8.0, 5.0, filters) + [radio(0.5)])
    assert "q22 = log10(F_W4/F_1.4GHz) = " in loud.text and "< 0.5: radio excess" in loud.text
    assert loud.weights == {"agn": 1.5, "qso": 1.0, "galaxy": 0.3, "star": -1.0}
    # Same radio flux, strong W4 (starburst): q22 > 0.5, not counted.
    sfg = radio_line(optical + _wise_points(12.0, 11.5, 7.0, 0.5, filters) + [radio(0.5)])
    assert ">= 0.5: on the star-forming mid-IR/radio correlation" in sfg.text and sfg.weights == {}
    # No W4 but W2-W3 on the dusty locus: half weight; no W4 and no dusty colours: full weight, check stated.
    dusty = radio_line(optical + _wise_points(12.0, 11.5, 9.5, None, filters) + [radio(0.5)])
    assert "dusty star-forming locus" in dusty.text and dusty.weights == {"agn": 0.75, "qso": 0.5, "galaxy": 0.15, "star": -0.5}
    plain = radio_line(optical + [radio(0.5)])
    assert "q22 not available" in plain.text and plain.weights["agn"] == 1.5


# ---------------------------------------------------------------------------
# 6. Supplementary budget: refused connections, success, router defaults, cancellation
# ---------------------------------------------------------------------------


def _replay_3c273_except_sdss(sdss_effect: Any):
    replay = replay_side_effect(exchanges_for("3c273"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "skyserver.sdss.org":
            return await sdss_effect(request)
        return replay(request)

    return handler


async def test_refused_skyserver_is_retried_once_per_run_then_backed_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reviewer reproduction: a refusing SkyServer cost every request its retries (20 s on Windows), never backed off."""
    monkeypatch.setattr(sed, "SDSS_CONNECT_RETRY_DELAYS", (0.01, 0.01))
    attempts: list[str] = []

    async def refuse(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise httpx.ConnectError("All connection attempts failed", request=request)

    rec = await replay_record("3c273")
    backoff = sed.HostBackoff()
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(refuse))
        async with offline_client() as client:
            first = await sed.sed_from_record(rec, client=client, filters=offline_filters(), backoff=backoff,
                                              deadline_seconds=20.0)
            first_attempts = len(attempts)
            started = time.monotonic()
            second = await sed.sed_from_record(rec, client=client, filters=offline_filters(), backoff=backoff,
                                               deadline_seconds=20.0)
            elapsed = time.monotonic() - started
    assert first_attempts == 3  # one round of retries per run, not one per SDSS query
    notes = " | ".join(first["notes"])
    assert "sdss lookup failed: ConnectError (skyserver.sdss.org after" in notes
    assert "not retried in this run: ConnectError after 3 attempts" in notes
    assert (backoff.pending("skyserver.sdss.org") or "").startswith("ConnectError after")
    assert len(attempts) == first_attempts and elapsed < 1.0, elapsed
    skipped = [n for n in second["notes"] if n.startswith("sdss lookup failed: skipped")]
    assert skipped and "back-off, retried in 60 s" in skipped[0]
    assert first["classification"]["label"] == second["classification"]["label"] == "qso"


async def test_refused_local_port_is_backed_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same with a real refused connection (a closed local port)."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(sed, "SDSS_SQL_URL", f"http://127.0.0.1:{port}/SqlSearch")
    monkeypatch.setattr(sed, "SDSS_CONNECT_RETRY_DELAYS", (0.01, 0.01))
    rec = await replay_record("3c273")
    backoff = sed.HostBackoff()
    replay = replay_side_effect(exchanges_for("3c273"))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="127.0.0.1").pass_through()
        router.route().mock(side_effect=replay)
        async with offline_client() as client:
            first = await sed.sed_from_record(rec, client=client, filters=offline_filters(), backoff=backoff,
                                              deadline_seconds=30.0)
            started = time.monotonic()
            second = await sed.sed_from_record(rec, client=client, filters=offline_filters(), backoff=backoff,
                                               deadline_seconds=30.0)
            elapsed = time.monotonic() - started
    assert any(n.startswith("sdss lookup failed: ConnectError (127.0.0.1 after") for n in first["notes"]), first["notes"]
    assert any(n.startswith("sdss lookup failed: skipped, 127.0.0.1 ConnectError") and "back-off" in n
               for n in second["notes"]), second["notes"]
    assert elapsed < 1.0, elapsed


async def test_one_refused_connection_then_success_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sed, "SDSS_CONNECT_RETRY_DELAYS", (0.01, 0.01))
    replay = replay_side_effect(exchanges_for("3c273"))
    refused = [0]
    answered: list[str] = []

    async def once(request: httpx.Request) -> httpx.Response:
        if not refused[0]:
            refused[0] += 1
            raise httpx.ConnectError("All connection attempts failed", request=request)
        answered.append(str(request.url))
        return replay(request)

    rec = await replay_record("3c273")
    backoff = sed.HostBackoff()
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(once))
        async with offline_client() as client:
            result = await sed.sed_from_record(rec, client=client, filters=offline_filters(), backoff=backoff,
                                               deadline_seconds=20.0)
    assert refused[0] == 1
    assert not any("lookup failed" in n for n in result["notes"]), result["notes"]
    assert result["sdss_extra"] is not None and backoff.pending("skyserver.sdss.org") is None
    assert len(answered) == 2  # the refused SpecObj query was retried, then the object query ran
    assert result["redshift"]["value"] == pytest.approx(0.158339, abs=1e-5)


async def test_slow_failures_are_backed_off_and_success_clears_a_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sed, "SUPPLEMENTARY_SLOW_FAILURE_SECONDS", 0.05)
    backoff = sed.HostBackoff()

    async def slow_error() -> None:
        await asyncio.sleep(0.1)
        raise sed.SEDUpstreamError("HTTP 503")

    async def fast_error() -> None:
        raise sed.SEDUpstreamError("HTTP 503")

    notes: list[str] = []
    await sed._run_supplementary({"a": ("slow.example", slow_error), "b": ("fast.example", fast_error)},
                                 deadline=5.0, backoff=backoff, notes=notes)
    assert backoff.pending("slow.example") is not None and backoff.pending("fast.example") is None

    # A concurrent request backs the host off while this lookup is in flight; the answer clears it.
    async def answered_meanwhile() -> str:
        backoff.record_failure("up.example", "ReadTimeout after 60 s")
        await asyncio.sleep(0)
        return "ok"

    out = await sed._run_supplementary({"c": ("up.example", answered_meanwhile)}, deadline=5.0, backoff=backoff,
                                       notes=notes)
    assert out == {"c": "ok"} and backoff.pending("up.example") is None

    # But a costly failure of the same host in the same run wins over a success.
    async def times_out() -> None:
        raise httpx.ReadTimeout("")

    async def fine() -> str:
        return "ok"

    await sed._run_supplementary({"d": ("mixed.example", fine), "e": ("mixed.example", times_out)}, deadline=5.0,
                                 backoff=backoff, notes=notes)
    assert backoff.pending("mixed.example") is not None


def test_router_uses_the_default_deadline_and_the_shared_backoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The router/build_sed path (no explicit deadline or back-off) is bounded by the configured deadline, and the
    next request skips the hanging host (process-wide back-off)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS", "1")
    sdss_requests: list[dict[str, Any]] = []

    async def hang(request: httpx.Request) -> httpx.Response:
        sdss_requests.append(dict(request.extensions.get("timeout") or {}))
        await asyncio.sleep(30)
        return httpx.Response(500)

    rec = asyncio.run(replay_record("3c273"))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(hang))
        with TestClient(make_app(FakeService(rec), tmp_path)) as client:
            params = {"ra": 187.2779154, "dec": 2.0523883, "radius_arcsec": 10}
            started = time.monotonic()
            first = client.get("/api/v1/sed", params=params)
            elapsed = time.monotonic() - started
            calls = len(sdss_requests)
            second = client.get("/api/v1/sed", params=params)
    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    assert elapsed < 5.0, elapsed
    assert any("within the 1 s supplementary deadline" in n for n in first.json()["notes"])
    assert calls >= 1 and len(sdss_requests) == calls  # the second request never contacts SkyServer
    assert any(n.startswith("sdss lookup failed: skipped") for n in second.json()["notes"])
    # Each request's own timeout is capped by the deadline.
    assert all(t.get("read") is not None and t["read"] <= 1.0 for t in sdss_requests), sdss_requests


async def test_cancelling_a_sed_cancels_its_lookups(caplog: pytest.LogCaptureFixture) -> None:
    rec = await replay_record("3c273")

    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(3)
        raise httpx.ReadTimeout("late", request=request)

    def lookups() -> list[str]:
        # The supplementary lookups and (round-6 review) the SVO filter fetches.
        names = (t.get_coro().__qualname__ for t in asyncio.all_tasks() if not t.done())
        return [n for n in names if n.startswith("fetch_") or n == "FilterCatalog._fetch"]

    caplog.set_level(logging.ERROR, logger="asyncio")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(hang))
        async with offline_client() as client:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(sed.sed_from_record(rec, client=client, filters=offline_filters(),
                                                           deadline_seconds=20.0), 0.5)
            assert time.monotonic() - started < 1.5  # the hanging lookups are cancelled, not waited for
            assert lookups() == []
        # Owned client: the lookups never outlive the client sed_from_record closes.
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sed.sed_from_record(rec, filters=offline_filters(), deadline_seconds=20.0), 0.5)
        assert time.monotonic() - started < 1.5
        assert lookups() == []
        await asyncio.sleep(3.2)  # past the moment the orphans used to fail
    assert "Task exception was never retrieved" not in caplog.text


async def test_deadline_leaves_no_lookup_running() -> None:
    rec = await replay_record("3c273")

    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(500)

    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(hang))
        async with offline_client() as client:
            await sed.sed_from_record(rec, client=client, filters=offline_filters(), deadline_seconds=0.5,
                                      backoff=sed.HostBackoff())
            assert [t for t in asyncio.all_tasks() if not t.done()
                    and t.get_coro().__qualname__.startswith("fetch_")] == []


# ---------------------------------------------------------------------------
# 7. Malformed records
# ---------------------------------------------------------------------------


def _t8() -> dict[str, Any]:
    return copy.deepcopy(load_record("t8dwarf"))


def _set(path: str, value: Any):
    def mutate(rec: dict[str, Any]) -> dict[str, Any]:
        node: Any = rec
        keys = path.split("/")
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
        return rec
    return mutate


@pytest.mark.parametrize(("mutate", "where"), [
    (_set("resolved_object", "M87"), "resolved_object must be an object"),
    (_set("resolved_object/aliases", "2MASS J1"), "resolved_object.aliases must be a list"),
    (_set("resolved_object/aliases", [1]), "resolved_object.aliases[0] must be a string"),
    (_set("crossmatch_groups", {"a": 1}), "crossmatch_groups must be a list"),
    (_set("crossmatch_groups/0", "group"), "crossmatch_groups[0] must be an object"),
    (_set("crossmatch_groups/0/members", "x"), "crossmatch_groups[0].members must be a list"),
    (_set("crossmatch_groups/0/members/0", "x"), "crossmatch_groups[0].members[0] must be an object"),
    (_set("crossmatch_groups/0/members/0/data", "x"), "crossmatch_groups[0].members[0].data must be an object"),
    (_set("crossmatch_groups/0/members/0/metadata", "x"), "crossmatch_groups[0].members[0].metadata must be an object"),
    (_set("crossmatch_groups/0/members/0/position_at_epoch", 3), "members[0].position_at_epoch must be an object"),
    (_set("provenance", "x"), "provenance must be an object"),
    (_set("failures", "x"), "failures must be a list"),
    (_set("failures", ["x"]), "failures[0] must be an object"),
    (_set("target", "x"), "target must be an object"),
])
def test_malformed_record_shapes_are_input_errors(mutate: Any, where: str) -> None:
    with pytest.raises(sed.SEDInputError) as info:
        sed_of(mutate(_t8()))
    assert where in str(info.value)


def test_well_formed_nulls_are_accepted() -> None:
    rec = _t8()
    rec["resolved_object"] = None
    rec["provenance"] = None
    rec["failures"] = None
    rec["crossmatch_groups"][0]["members"][0]["metadata"] = None
    rec["crossmatch_groups"][0]["members"][0]["position_at_epoch"] = None
    assert sed_of(rec)["classification"]["label"] == "star"

