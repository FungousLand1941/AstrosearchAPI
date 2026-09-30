"""Round-3 review regressions for sed.py (offline).

1. Members are pooled over ALL crossmatch groups (the Bayesian association splits objects): unit records plus
   recorded multi-group records (tests/fixtures/sed/records/<key>.json, supplementary requests replayed).
2. SIMBAD object types are mapped through the otypedef hierarchy (recorded live table), transients are not stars.
3. SIMBAD rows: the resolved name first, planets never replace their host.
4. WISE colour-correction assumption; Gaia DSC white-dwarf/binary classes; NED '*' rows with a large redshift;
   SDSS morphology of saturated objects; tie-breaks consistent with the adopted redshift or 'unknown'.
5. One deadline and a per-host back-off for the supplementary lookups.
6. Boundary tests of every spec'd threshold and of conversions no other test pinned (GALEX AB, SIMBAD velocity
   errors, photoErrorClass, errors of upper limits).
7. Input handling (malformed records, booleans, name together with coordinates, radius fallback).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import FIXTURES, load_exchanges, replay_side_effect
from helpers import offline_client
from test_sed import FakeService, exchanges_for, make_app, point, replay_record
from test_sed_regressions import evidence, member_choice, passport

import sed

OFFLINE = Path("nonexistent-dir/x.json")


def offline_filters() -> sed.FilterCatalog:
    return sed.FilterCatalog(offline=True, cache_path=OFFLINE)


@pytest.fixture(autouse=True)
def _fresh_backoff() -> Any:
    sed.default_backoff().clear()
    yield
    sed.default_backoff().clear()


def used(result: dict[str, Any], catalog: str) -> dict[str, Any]:
    hits = [m for m in result["members"] if m["catalog"] == catalog and m["used"]]
    assert len(hits) == 1, (catalog, [(m["catalog"], m["used"], m["reason"]) for m in result["members"]])
    return hits[0]


def bands(result: dict[str, Any], facility: str) -> set[str]:
    return {p["band"] for p in result["points"] if p["facility"].startswith(facility)}


# ---------------------------------------------------------------------------
# 1. Members pooled over every crossmatch group
# ---------------------------------------------------------------------------

GAIA_ROW = {"phot_g_mean_mag": 15.0, "phot_bp_mean_mag": 15.4, "phot_rp_mean_mag": 14.5}
TMASS_ROW = {"j_m": 13.5, "j_msigcom": 0.02, "h_m": 13.1, "h_msigcom": 0.02, "k_m": 13.0, "k_msigcom": 0.02,
             "ph_qual": "AAA"}
WISE_ROW = {"w1mpro": 12.9, "w1sigmpro": 0.02, "w2mpro": 12.85, "w2sigmpro": 0.02, "ph_qual": "AAUU"}


def row(catalog: str, source_id: str, sep: float, data: dict[str, Any], err: float = 0.05) -> dict[str, Any]:
    return {"catalog": catalog, "source_id": source_id, "ra": 10.0, "dec": 20.0, "separation_arcsec": sep,
            "positional_error_arcsec": err, "metadata": {}, "data": data}


def record(*groups: dict[str, Any], resolved: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"target": {"ra": 10.0, "dec": 20.0}, "crossmatch_groups": list(groups), "failures": [],
            "provenance": {"query_radius_arcsec": 10.0}, "resolved_object": resolved}


def sed_of(rec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(sed.sed_from_record(rec, supplementary=False, filters=offline_filters(), **kwargs))


def test_singleton_resolver_group_nearer_than_the_target_group_keeps_all_photometry() -> None:
    """Reviewer reproduction: a lone NED row at 0.000 arcsec in its own group used to win and empty the SED."""
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("gaia_dr3", "111", 0.002, GAIA_ROW, 0.001), row("twomass_psc", "2M", 0.066, TMASS_ROW),
        row("allwise", "AW", 0.036, WISE_ROW)]}
    lone = {"group_id": "object-2", "contains_target": False, "members": [
        row("ned", "NED 1", 0.0, {"prefphytype": "G"})]}
    for groups in ((target, lone), (lone, target)):
        result = sed_of(record(*groups))
        assert result["group_id"] == "object-1"
        assert bands(result, "Gaia") == {"G", "BP", "RP"} and bands(result, "2MASS") == {"J", "H", "Ks"}
        assert bands(result, "WISE") == {"W1", "W2"}
        assert used(result, "ned")["group_id"] == "object-2"  # an in-tolerance row of another group is still used
    # Without the contains_target flag (older crossmatch output) the group with survey photometry is the target group.
    unflagged = [{k: v for k, v in g.items() if k != "contains_target"} for g in (lone, target)]
    assert sed.select_group(record(*unflagged))["group_id"] == "object-1"


def test_in_tolerance_rows_of_other_groups_are_used_and_reported() -> None:
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("gaia_dr3", "111", 0.0, GAIA_ROW, 0.001), row("simbad", "HD 1", 0.0, {"otype": "PM*"})]}
    split = {"group_id": "object-2", "members": [row("twomass_psc", "2M-near", 0.40, TMASS_ROW)]}
    far = {"group_id": "object-3", "members": [row("allwise", "AW-far", 4.5, WISE_ROW)]}
    result = sed_of(record(target, split, far))
    tmass = used(result, "twomass_psc")
    assert tmass["source_id"] == "2M-near" and tmass["group_id"] == "object-2"
    assert "from crossmatch group object-2, not the target group object-1" in tmass["reason"]
    assert bands(result, "2MASS") == {"J", "H", "Ks"} and not bands(result, "WISE")  # 4.5" > 3" tolerance
    assert any("twomass_psc (object-2)" in n for n in result["notes"])
    assert result["group_ids"] == ["object-1", "object-2"]


def test_target_group_row_is_preferred_and_the_other_row_reported() -> None:
    target = {"group_id": "object-1", "contains_target": True, "members": [row("twomass_psc", "2M-target", 0.8, TMASS_ROW)]}
    other = {"group_id": "object-2", "members": [row("twomass_psc", "2M-other", 0.3, dict(TMASS_ROW, j_m=9.0))]}
    result = sed_of(record(target, other))
    assert used(result, "twomass_psc")["source_id"] == "2M-target"
    note = next(n for n in result["notes"] if "2M-other" in n)
    assert "crossmatch group object-2, 0.30 arcsec" in note and "2M-target (0.80 arcsec, target group) was chosen" in note


def test_epoch_propagation_growth_widens_the_tolerance() -> None:
    member = row("gaia_dr3", "1", 1.2, GAIA_ROW, 0.001)
    assert not sed._within_tolerance(member)  # 1.2" > 1.0" floor
    member["position_at_epoch"] = {"pm_growth_arcsec": 0.45, "sigma_arcsec": 30.0}
    # 3 x hypot(0.001, 0.45, 0.1) = 1.38": the source-structure part of sigma_arcsec is not used.
    assert sed._row_tolerance(member) == pytest.approx(3 * math.sqrt(0.001**2 + 0.45**2 + 0.1**2), rel=1e-9)
    assert sed._within_tolerance(member)


def test_old_single_group_fixtures_have_exactly_one_group() -> None:
    """The regression records of rounds 1-2 hold one group; the round-3 records hold several (multi-group coverage)."""
    multi = {"groombridge1830", "kapteyn", "luhman16", "peg51", "t8dwarf", "m87group", "sn1998bw", "sn2006gy", "q2237",
             "adleo"}
    for key in multi:
        rec = json.loads((FIXTURES / "sed" / "records" / f"{key}.json").read_text(encoding="utf-8"))
        assert len(rec["crossmatch_groups"]) > 1 and any(g.get("contains_target") for g in rec["crossmatch_groups"]), key


def test_groombridge_1830_keeps_its_2mass_photometry(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("groombridge1830", tmp_path_factory)
    tmass = used(result, "twomass_psc")
    assert tmass["source_id"] == "11525880+3743060" and tmass["group_id"] != result["group_id"]
    assert tmass["separation_arcsec"] == pytest.approx(0.403, abs=0.01)
    assert bands(result, "2MASS") == {"J", "H", "Ks"}
    assert result["classification"]["label"] == "star"


def test_kapteyns_star_keeps_its_allwise_photometry_and_host_row(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("kapteyn", tmp_path_factory)
    wise = used(result, "allwise")
    assert wise["source_id"] == "J051146.81-450204.5" and wise["separation_arcsec"] == pytest.approx(1.404, abs=0.01)
    assert bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    simbad = used(result, "simbad")
    assert simbad["source_id"] == "HD 33793" and "HD 33793b" in simbad["reason"]
    assert "SIMBAD object type 'PM*'" in evidence(result) and "'Pl?'" not in evidence(result)
    assert result["classification"]["label"] == "star"


def test_luhman_16_is_not_three_simbad_points(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("luhman16", tmp_path_factory)
    assert used(result, "twomass_psc")["source_id"] == "10491891-5319100"
    assert used(result, "allwise")["source_id"] == "J104915.52-531906.1"
    assert bands(result, "2MASS") == {"J", "H", "Ks"} and bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    assert result["classification"]["label"] == "star"


def test_t8_dwarf_keeps_its_allwise_photometry_and_is_not_an_agn(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("t8dwarf", tmp_path_factory)
    assert used(result, "allwise")["separation_arcsec"] == pytest.approx(2.791, abs=0.01)
    assert bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    # W1-W2 = 2.85 is the methane colour of a T dwarf, not a Stern et al. 2012 AGN: the rule is not applied.
    line = next(t for t in result["classification"]["evidence"] if "WISE W1-W2" in t)
    assert "W1-W2 = 2.85" in line and "brown dwarfs" in line and "not applied" in line
    detail = next(d for d in result["classification"]["evidence_detail"] if d["text"] == line)
    assert detail["weights"] == {}
    assert result["classification"]["label"] == "star"


def test_m87_keeps_its_gaia_nucleus_from_another_group(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("m87group", tmp_path_factory)
    gaia = used(result, "gaia_dr3")
    assert gaia["source_id"] == "3907709439453756032" and gaia["group_id"] != result["group_id"]
    assert bands(result, "Gaia") == {"G", "BP", "RP"}
    assert used(result, "simbad")["source_id"] == "M 87"  # not 'NAME M87 Jet' at 0.0067"
    assert result["classification"]["label"] in {"galaxy", "agn"}
    assert result["redshift"]["value"] == pytest.approx(0.00428, abs=2e-4)


# ---------------------------------------------------------------------------
# 2. SIMBAD otypedef mapping (recorded live table)
# ---------------------------------------------------------------------------

OTYPEDEF = json.loads((FIXTURES / "sed" / "simbad" / "otypedef.json").read_text(encoding="utf-8"))


def test_embedded_otypedef_is_the_live_table() -> None:
    names = [c["name"] for c in OTYPEDEF["metadata"]]
    rows = [dict(zip(names, r, strict=True)) for r in OTYPEDEF["data"]]
    assert len(rows) == len(sed.SIMBAD_OTYPEDEF) == 226
    for r in rows:
        assert sed.SIMBAD_OTYPEDEF[r["otype"]] == (r["path"], bool(r["is_candidate"])), r


def test_every_otypedef_code_maps_by_its_path() -> None:
    names = [c["name"] for c in OTYPEDEF["metadata"]]
    for r in (dict(zip(names, r, strict=True)) for r in OTYPEDEF["data"]):
        code, path = r["otype"], r["path"] or ""
        nodes = [n.strip() for n in path.split(">")] if path else []
        cls, factor = sed.simbad_type_class(code)
        if cls is not None:
            assert factor == (0.5 if r["is_candidate"] else 1.0), code
        if code in {"SN*", "SN?", "No*", "No?"}:
            assert cls is None, code  # transients are never a class
        elif nodes[:3] == ["G", "AGN", "QSO"]:
            assert cls == "qso", code
        elif nodes[:2] == ["G", "AGN"]:
            assert cls == "agn", code
        elif nodes[:1] == ["G"]:
            assert cls == "galaxy", code
        elif nodes[:1] == ["*"]:
            assert cls == "star", (code, path)
        elif nodes[:1] in (["Cl*"], ["As*"], ["ISM"], ["X"], ["Rad"], ["IR"], ["UV"], ["gam"], ["ev"], ["err"], ["mul"]):
            assert cls is None, (code, path)
    # The reviewed cases.
    assert sed.simbad_type_class("Sy?") == ("star", 0.5)  # Symbiotic Star Candidate, not a Seyfert
    assert sed.simbad_type_class("SN*") == (None, 0.0) and sed.simbad_type_class("No*") == (None, 0.0)
    assert sed.simbad_type_class("LeQ") == ("qso", 1.0) and sed.simbad_type_class("LeG") == ("galaxy", 1.0)
    for code in ("Cl*", "As*", "St*", "GlC", "OpC"):
        assert sed.simbad_type_class(code) == (None, 0.0), code
    for code in ("RR?", "Ce?", "EB?", "TT?", "Mi?", "Y*?", "BD?", "V*?"):
        assert sed.simbad_type_class(code) == ("star", 0.5), code
    assert sed.simbad_type_class("nonsense") == (None, 0.0)


def test_supernova_is_not_a_star_and_keeps_its_host_redshift(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("sn1998bw", tmp_path_factory)
    assert used(result, "simbad")["source_id"] == "SN 1998bw"
    cls = result["classification"]
    assert cls["label"] != "star" and "SIMBAD object type 'SN*'" in evidence(result) and "transient" in evidence(result)
    z = result["redshift"]
    assert z["value"] == pytest.approx(0.0085, abs=1e-4) and not z.get("conflict") and z["reliable"] is not False
    assert "misassociation" not in evidence(result)


def test_supernova_in_a_catalogued_host_is_a_galaxy_not_a_qso(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("sn2006gy", tmp_path_factory)
    assert result["classification"]["label"] == "galaxy", result["classification"]
    assert result["redshift"]["value"] == pytest.approx(0.019, abs=5e-4)


def test_simbad_symbiotic_candidate_is_not_agn_evidence() -> None:
    simbad = member_choice("simbad", {"otype": "Sy?"})
    (item,) = sed.gather_evidence([], [simbad], {})
    assert "-> star" in item.text and item.weights == {"star": 1.5}


# ---------------------------------------------------------------------------
# 3. SIMBAD row choice
# ---------------------------------------------------------------------------


def test_planet_rows_never_replace_their_host() -> None:
    # Kapteyn's star: the planet row was 2e-11 arcsec nearer than the host.
    planet = row("simbad", "HD 33793b", 1.7520838e-05, {"otype": "Pl?"})
    host = row("simbad", "HD 33793", 1.7520860e-05, {"otype": "PM*"})
    (choice,) = sed.select_members({"members": [planet, host]})
    assert choice.source_id == "HD 33793" and "HD 33793b (Pl?" in choice.reason
    (alone,) = sed.select_members({"members": [planet]})
    assert alone.source_id == "HD 33793b"  # a planet row alone is still the SIMBAD member
    # The Sesame-resolved main identifier wins over a nearer non-planet row.
    jet = row("simbad", "NAME M87 Jet", 0.0, {"otype": "Rad"})
    galaxy = row("simbad", "M 87", 0.0067, {"otype": "AGN"})
    (named,) = sed.select_members({"members": [jet, galaxy]}, canonical_name="M  87")
    assert named.source_id == "M 87"


def test_51_peg_uses_the_star_and_its_doppler_redshift(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("peg51", tmp_path_factory)
    simbad = used(result, "simbad")
    assert simbad["source_id"] == "* 51 Peg" and "* 51 Peg b (Pl" in simbad["reason"]
    z = result["redshift"]
    assert (z["source"], z["kind"]) == ("simbad", "spec") and abs(z["value"]) < 1e-3
    assert "Doppler" in z["note"]
    assert result["classification"]["label"] == "star"
    # Saturated SDSS (sat flags u-z, psfMag_u = 7.45): its type 3 is neither morphology nor a galaxy vote.
    text = evidence(result)
    assert "SDSS type 3 not used" in text and "galaxy morphology" not in text and "half weight" not in text


# ---------------------------------------------------------------------------
# 4. Evidence details
# ---------------------------------------------------------------------------


def test_wise_colour_correction_assumption_is_stated() -> None:
    result = sed_of(record())
    text = next(a for a in result["assumptions"] if "colour correction" in a)
    # Wright et al. 2010 / ES IV.4.h Table 2: f_c(nu^0) / f_c(nu^2) in W3 = 0.9169 / 1.0088 -> constant-F_nu zero points
    # overestimate a Rayleigh-Jeans source by 1/0.9089 - 1 = 10.0%.
    assert "309.54" in text and "0.9169/1.0088" in text and "10.0%" in text and "8-10% W4" in text
    assert round((1.0088 / 0.9169 - 1) * 100, 1) == 10.0
    assert round((1.0084 / 0.9907 - 1) * 100, 1) == 1.8


def test_gaia_dsc_counts_white_dwarfs_and_binaries_as_stars(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("adleo", tmp_path_factory)
    assert result["gaia_extra"]["classprob_dsc_combmod_binarystar"] == pytest.approx(0.999999, abs=1e-6)
    line = next(d for d in result["classification"]["evidence_detail"] if "DSC-Combmod" in d["text"])
    assert "binary star=1.000" in line["text"] and "-> star = 1.000" in line["text"]
    assert line["weights"]["star"] == pytest.approx(2.0, abs=1e-3)
    # Without the astrophysical_parameters columns the text says what 'star' covers.
    gaia = member_choice("gaia_dr3", {})
    ev = sed.gather_evidence([], [gaia], {}, {"classprob_dsc_combmod_quasar": 0.1, "classprob_dsc_combmod_galaxy": 0.1,
                                             "classprob_dsc_combmod_star": 0.8})
    (dsc,) = [e for e in ev if "DSC" in e.text]
    assert "single non-WD stars only" in dsc.text and dsc.weights["star"] == pytest.approx(1.6)
    wd = sed.gather_evidence([], [gaia], {}, {"classprob_dsc_combmod_quasar": 0.0, "classprob_dsc_combmod_galaxy": 0.0,
                                             "classprob_dsc_combmod_star": 0.0, "classprob_dsc_combmod_whitedwarf": 0.99,
                                             "classprob_dsc_combmod_binarystar": 0.01})
    assert next(e for e in wd if "DSC" in e.text).weights["star"] == pytest.approx(2.0)


def test_gaia_query_joins_the_dsc_white_dwarf_and_binary_columns() -> None:
    adql = sed.gaia_extra_adql("625453654702751872")
    assert "LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id" in adql
    assert "ap.classprob_dsc_combmod_whitedwarf" in adql and "ap.classprob_dsc_combmod_binarystar" in adql
    assert adql.endswith("WHERE g.source_id = 625453654702751872")


def test_ned_stellar_type_with_a_quasar_redshift_is_ignored() -> None:
    # SDSS DR18 quasar z = 2.38 whose NED row WISEA J100058.51+030252.7 is typed '*' with z = 2.374 (zflag SLS).
    ned = member_choice("ned", {"prefphytype": "*", "z": 2.374, "zflag": "SLS"}, "WISEA J100058.51+030252.7")
    (item,) = sed.gather_evidence([], [ned], {"value": None}, use_redshift=False)
    assert "ignored" in item.text and "internally inconsistent" in item.text and item.weights == {}
    star = member_choice("ned", {"prefphytype": "*", "z": 0.0001}, "HR 4550")
    (ok,) = sed.gather_evidence([], [star], {"value": None}, use_redshift=False)
    assert ok.weights == {"star": 2.5}


def test_sdss_morphology_of_saturated_objects() -> None:
    assert sed.sdss_morphology({"type": 3, "sat_r": 1})[0] is None
    assert sed.sdss_morphology({"type": 3, "psfMag_u": 7.45, "psfMag_r": 10.0})[0] is None  # no flags: psfMag < 14
    assert sed.sdss_morphology({"type": 3, "psfMag_r": 14.95, "sat_r": 0})[0] == 3  # flags authoritative
    assert sed.sdss_morphology({"type": 3, "psfMag_r": 14.95})[0] == 3
    six, why = sed.sdss_morphology({"type": 6, "sat_g": 1})
    assert six == 6 and "saturation biases towards type 3" in why
    sat = member_choice("sdss", {"type": 3, "sat_r": 1})
    assert sed.optical_extent([sat]) == (None, "morphology unknown")
    assert sed.sdss_aperture({"type": 3, "sat_r": 1, **{f"modelMag{e}_{b}": 15.0 for b in "ugriz" for e in ("", "Err")}}) == "psf"


class _Patched:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, weights: list[dict[str, float]]) -> None:
        monkeypatch.setattr(sed, "gather_evidence", lambda *a, **k: [sed.Evidence(f"e{i}", w) for i, w in enumerate(weights)])


def test_ties_follow_the_adopted_redshift_or_are_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _Patched(monkeypatch, [{"qso": 3.0}, {"galaxy": 3.0}])
    none = sed.classify([], [], {"value": None})
    assert none["label"] == "unknown" and none["confidence"] == 0.0 and none["tied"] == ["qso", "galaxy"]
    assert "no evidence separates them" in none["evidence"][-1]
    ned_qso = member_choice("ned", {"prefphytype": "QSO", "z": 1.695}, "QSO J2240+0321")
    by_z = sed.classify([], [ned_qso], {"value": 1.695, "source": "ned", "kind": "spec", "reliable": True})
    assert by_z["label"] == "qso" and "qso-typed entry" in by_z["evidence"][-1]
    ned_g = member_choice("ned", {"prefphytype": "G", "z": 0.0386}, "CGCG 378-015")
    assert sed.classify([], [ned_g], {"value": 0.0386, "source": "ned", "kind": "spec", "reliable": True})["label"] == "galaxy"


def test_extended_tie_with_a_quasar_redshift_uses_the_luminosity(monkeypatch: pytest.MonkeyPatch) -> None:
    _Patched(monkeypatch, [{"qso": 3.0}, {"galaxy": 3.0}])
    simbad = member_choice("simbad", {"otype": "QSO"}, "QSO J2240+0321")
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": 15.0, "iMeanKronMag": 14.2})
    allwise = member_choice("allwise", {"ext_flg": 5})
    members = [simbad, ps1, allwise]
    lensed = sed.classify([], members, {"value": 1.695, "source": "simbad", "kind": "spec", "reliable": True})
    # iKron = 14.2 at z = 1.695: M = 14.2 - DM(1.695) ~ -31.6 < -22: a 'galaxy' at that z is impossible.
    m_abs = sed.absolute_magnitude(14.2, 1.695)
    assert m_abs < -30
    assert lensed["label"] == "qso" and f"M = {m_abs:.1f} < -22" in lensed["evidence"][-1]
    nearby = sed.classify([], members, {"value": 0.0018, "source": "simbad", "kind": "spec", "reliable": True})
    assert nearby["label"] == "galaxy" and "host-galaxy classes preferred" in nearby["evidence"][-1]


# ---------------------------------------------------------------------------
# 5. Supplementary deadline and back-off
# ---------------------------------------------------------------------------


def _replay_except_sdss(sdss_effect: Any):
    replay = replay_side_effect(exchanges_for("3c273"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "skyserver.sdss.org":
            return await sdss_effect(request)
        return replay(request)

    return handler


SDSS_REQUESTS: list[str] = []


async def _hang(request: httpx.Request) -> httpx.Response:
    SDSS_REQUESTS.append(str(request.url))  # counted when sent (respx records a call only once it completes)
    await asyncio.sleep(30)
    return httpx.Response(500)


async def test_hanging_skyserver_is_bounded_by_one_deadline_and_backed_off(tmp_path: Path) -> None:
    record = await replay_record("3c273")
    backoff = sed.HostBackoff(retry_after_seconds=300.0)
    SDSS_REQUESTS.clear()
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_except_sdss(_hang))
        async with offline_client() as client:
            started = time.monotonic()
            first = await sed.sed_from_record(record, client=client, filters=offline_filters(), deadline_seconds=1.0,
                                              backoff=backoff)
            elapsed = time.monotonic() - started
            sdss_calls = len(SDSS_REQUESTS)
            second = await sed.sed_from_record(record, client=client, filters=offline_filters(), deadline_seconds=1.0,
                                               backoff=backoff)
            sdss_calls_after = len(SDSS_REQUESTS)
    assert elapsed < 5.0, elapsed
    notes = " | ".join(first["notes"])
    assert "sdss lookup failed: TimeoutError (no answer from skyserver.sdss.org within the 1 s" in notes
    assert "sdss_object lookup failed: TimeoutError" in notes
    assert first["gaia_extra"] is not None and first["simbad_extra"] is not None  # other hosts unaffected
    assert first["classification"]["label"] == "qso"
    # The second request does not touch the hanging host at all.
    assert sdss_calls >= 1 and sdss_calls_after == sdss_calls
    skipped = [n for n in second["notes"] if n.startswith("sdss lookup failed: skipped")]
    assert skipped and "skyserver.sdss.org TimeoutError after the 1 s deadline; back-off" in skipped[0]
    assert second["gaia_extra"] is not None


async def test_slow_and_fast_failures(tmp_path: Path) -> None:
    record = await replay_record("3c273")

    async def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    async def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    for effect, backed_off, expected in ((read_timeout, True, "ReadTimeout (skyserver.sdss.org after "),
                                         (unavailable, False, "SEDUpstreamError (skyserver.sdss.org after ")):
        backoff = sed.HostBackoff()
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.route().mock(side_effect=_replay_except_sdss(effect))
            async with offline_client() as client:
                result = await sed.sed_from_record(record, client=client, filters=offline_filters(), backoff=backoff,
                                                   deadline_seconds=20.0)
        note = next(n for n in result["notes"] if n.startswith("sdss lookup failed"))
        assert expected in note, note
        assert (backoff.pending("skyserver.sdss.org") is not None) is backed_off


def test_backoff_expires_and_clears_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sed.time, "monotonic", lambda: clock[0])
    backoff = sed.HostBackoff(retry_after_seconds=60.0)
    backoff.record_failure("a.example", "ReadTimeout after 60 s")
    assert "ReadTimeout after 60 s; back-off, retried in 60 s" == backoff.pending("a.example")
    clock[0] += 61.0
    assert backoff.pending("a.example") is None
    backoff.record_failure("a.example", "x")
    backoff.record_success("a.example")
    assert backoff.pending("a.example") is None and backoff.pending(None) is None


def test_default_deadline_follows_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    assert sed.default_supplementary_deadline() == 30.0
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "12")
    assert sed.default_supplementary_deadline() == 12.0
    monkeypatch.setenv("ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS", "7.5")
    assert sed.default_supplementary_deadline() == 7.5
    with pytest.raises(sed.SEDInputError):
        sed_of(record(), deadline_seconds=0.0)


# ---------------------------------------------------------------------------
# 6. Thresholds and conversions pinned at their boundaries
# ---------------------------------------------------------------------------


def _wise_points(w1: float, w2: float) -> list[sed.SEDPoint]:
    wise = member_choice("allwise", {"w1mpro": w1, "w1sigmpro": 0.02, "w2mpro": w2, "w2sigmpro": 0.02, "ph_qual": "AA"})
    return sed.extract_points([wise], offline_filters())


@pytest.mark.parametrize(("w1", "applied"), [(12.81, True), (12.79, False)])
def test_stern_boundary(w1: float, applied: bool) -> None:
    ev = sed.gather_evidence(_wise_points(w1, 12.0), [], {})
    line = next(e for e in ev if "WISE W1-W2" in e.text)
    assert ("Stern et al. 2012 mid-IR AGN criterion" in line.text) is applied
    assert (line.weights == {"qso": 2.0, "agn": 2.0, "star": -1.0, "galaxy": -0.5}) is applied


@pytest.mark.parametrize(("poe", "significant"), [(5.1, True), (4.9, False)])
def test_parallax_boundary(poe: float, significant: bool) -> None:
    gaia = member_choice("gaia_dr3", {"parallax": 1.0, "parallax_error": 1.0 / poe, "pmra": 0.0, "pmdec": 0.0,
                                      "phot_g_mean_mag": 17.0})
    extra = {"parallax_over_error": poe, "ruwe": 1.0, "pmra_error": 0.1, "pmdec_error": 0.1}
    ev = sed.gather_evidence([], [gaia], {}, extra)
    assert any("significant parallax" in e.text for e in ev) is significant


def _radio_case(r: float) -> list[sed.Evidence]:
    i_mag = 18.0
    f_i = 3631.0 * 10 ** (-0.4 * i_mag)
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": i_mag, "iMeanPSFMagErr": 0.01})
    first = member_choice("first", {"int_flux_20_cm": f_i * 10**r * 1e3, "flux_20_cm_error": 0.1})
    points = sed.extract_points([ps1, first], offline_filters())
    return sed.gather_evidence(points, [ps1, first], {})


@pytest.mark.parametrize(("r", "loud"), [(1.1, True), (0.9, False)])
def test_radio_loudness_boundary(r: float, loud: bool) -> None:
    line = next(e for e in _radio_case(r) if "radio loudness" in e.text)
    assert f"R = log10(F_1.4GHz/F_i) = {r:.2f}" in line.text or f"R = {r:.2f}" in line.text
    assert ("radio-loud" in line.text) is loud and bool(line.weights) is loud


@pytest.mark.parametrize(("m_abs", "quasar"), [(-22.1, True), (-21.9, False)])
def test_quasar_luminosity_boundary(m_abs: float, quasar: bool) -> None:
    z = 0.5
    distance_modulus = 20.0 - sed.absolute_magnitude(20.0, z)
    i_mag = m_abs + distance_modulus
    ps1 = member_choice("panstarrs_dr2", {"iMeanPSFMag": i_mag, "iMeanPSFMagErr": 0.01})
    points = sed.extract_points([ps1], offline_filters())
    ev = sed.gather_evidence(points, [ps1], {"value": z, "kind": "spec", "source": "ned", "reliable": True})
    line = next(e for e in ev if e.text.startswith("M_i = "))
    assert line.text.startswith(f"M_i = {m_abs:.1f}")
    assert ("quasar luminosity (Schneider et al. 2010)" in line.text) is quasar
    assert (line.weights == {"qso": 2.0, "galaxy": -0.5}) is quasar


async def test_sdss_photoz_reliability_needs_photo_error_class_1() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        cmd = request.url.params["cmd"]
        rows: list[dict[str, Any]] = []
        if "Photoz" in cmd:
            rows = [{"objID": "1", "dist_arcsec": 0.1, "z": 0.3, "zErr": 0.05, "photoErrorClass": 2},
                    {"objID": "2", "dist_arcsec": 0.5, "z": 0.4, "zErr": 0.05, "photoErrorClass": 1},
                    {"objID": "3", "dist_arcsec": 0.9, "z": 0.5, "zErr": 0.05, "photoErrorClass": None}]
        return httpx.Response(200, json=[{"TableName": "Table1", "Rows": rows}])

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=answer)
        async with offline_client() as client:
            cands = await sed.fetch_sdss_redshifts(client, 10.0, 20.0, include_photoz=True)
    assert [(c["id"], c["reliable"]) for c in cands] == [("1", False), ("2", True), ("3", False)]
    best = sed.choose_redshift(cands)
    assert (best["value"], best["reliable"]) == (0.4, True)


def test_simbad_velocity_error_is_converted_to_redshift() -> None:
    simbad = member_choice("simbad", {"rvz_redshift": 0.00001}, "HD 1")
    extra = {"rvz_type": "v", "rvz_err": 2.99792458, "rvz_nature": "s", "rvz_qual": "A"}
    (cand,) = sed.member_redshift_candidates([simbad], simbad_extra=extra)
    assert cand["error"] == pytest.approx(1.0e-5, rel=1e-12) and cand["kind"] == "spec" and cand["reliable"] is True
    (as_z,) = sed.member_redshift_candidates([simbad], simbad_extra=dict(extra, rvz_type="z", rvz_err=0.002))
    assert as_z["error"] == 0.002


def test_upper_limits_carry_no_flux_error() -> None:
    spec = sed.MAG_SPECS["allwise"][1][2]  # W3
    filt = offline_filters().info(spec.filter_id)
    p = sed.magnitude_point(spec, "WISE", "allwise", "x", {"w3mpro": 11.0, "w3sigmpro": 0.3, "ph_qual": "AAUA"}, filt)
    assert p is not None and p.is_upper_limit and p.flux_err_jy is None and p.nu_fnu_err_erg_s_cm2 is None
    assert p.flux_jy == pytest.approx(31.674 * 10 ** (-4.4))
    assert p.magnitude_error == 0.3  # the catalogue value is kept for reference, not propagated


def test_galex_magnitudes_are_ab() -> None:
    galex = member_choice("galex", {"fuv_mag": 20.0, "fuv_magerr": 0.1, "NUVmag": 19.5, "e_NUVmag": 0.05})
    pts = {p.band: p for p in sed.extract_points([galex], offline_filters())}
    # AB: F = 3631 Jy x 10^(-0.4 m): FUV 20.0 -> 3.631e-5 Jy; NUV 19.5 -> 5.755e-5 Jy (not the SVO Vega 528.5 / 801.0 Jy).
    assert pts["FUV"].flux_jy == pytest.approx(3.631e-5, rel=1e-9)
    assert pts["FUV"].flux_err_jy == pytest.approx(3.631e-5 * 0.1 / sed.POGSON, rel=1e-9)
    assert pts["NUV"].flux_jy == pytest.approx(3631.0 * 10 ** (-0.4 * 19.5), rel=1e-9)
    assert pts["FUV"].mag_system == "AB" and pts["FUV"].zero_point_jy == 3631.0
    assert pts["FUV"].wavelength_um == pytest.approx(0.15488489655268, rel=1e-12)  # SVO GALEX.FUV WavelengthEff
    assert pts["FUV"].regime == "ultraviolet" and pts["FUV"].facility == "GALEX"


# ---------------------------------------------------------------------------
# 7. Input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [
    {"ra": None, "dec": 10.0}, {"ra": "abc", "dec": 10.0}, {"ra": float("nan"), "dec": 1.0}, {"ra": 400.0, "dec": 0.0},
    {"ra": 10.0, "dec": -91.0}, {"ra": True, "dec": 1.0}, {"dec": 1.0},
])
def test_malformed_record_targets_are_input_errors(target: dict[str, Any]) -> None:
    with pytest.raises(sed.SEDInputError):
        sed_of({"target": target, "crossmatch_groups": []})


def test_radius_falls_back_to_the_requested_radius() -> None:
    rec = record()
    rec["provenance"] = {}
    assert sed_of(rec, radius_arcsec=7.5)["target"]["radius_arcsec"] == 7.5
    assert sed_of(record(), radius_arcsec=7.5)["target"]["radius_arcsec"] == 10.0  # the record's own value wins


def test_router_rejects_booleans_and_ambiguous_targets(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    service = FakeService({})
    with TestClient(make_app(service, tmp_path)) as client:
        for body in ({"ra": True, "dec": 10}, {"ra": 10, "dec": False}, {"ra": 10, "dec": 1, "radius_arcsec": True},
                     {"name": "M87", "ra": 187.7, "dec": 12.4}, {"name": "M87", "ra": 187.7}, {"name": "   "}):
            res = client.post("/api/v1/sed", json=body)
            assert res.status_code == 422, (body, res.text)
        res = client.get("/api/v1/sed", params={"name": "M87", "ra": 187.7, "dec": 12.4})
        assert res.status_code == 422 and "not both" in res.json()["detail"]
        assert client.get("/api/v1/sed", params={"name": "  "}).status_code == 422
        assert client.get("/api/v1/sed", params={"ra": "true", "dec": 1}).status_code == 422
    assert service.calls == []


def test_router_reports_the_requested_radius(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    rec = asyncio.run(replay_record("3c273"))
    rec["provenance"] = {k: v for k, v in (rec.get("provenance") or {}).items() if k != "query_radius_arcsec"}
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/3c273") + load_exchanges("sed/svo")))
        with TestClient(make_app(FakeService(rec), tmp_path)) as client:
            res = client.get("/api/v1/sed", params={"ra": 187.2779154, "dec": 2.0523883, "radius_arcsec": 7.5})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["target"]["radius_arcsec"] == 7.5
    assert point(body, "2MASS", "Ks")["flux_jy"] > 0
