# ruff: noqa: F811  (pytest fixtures imported from test_ai are re-bound as test parameters)
"""Offline regression tests for the round-3 review of ai.py.

Replayed real SIMBAD / Sesame traffic (tests/fixtures/ai, recorded by
record_ai_fixtures.py) and a scripted fake Anthropic client, as in test_ai.py:
* SIMBAD's main type decides whether an object is extragalactic (stray "G" types),
* Hubble-flow errors honour the stated redshift precision (Ton 618),
* galaxy velocities become z = cz / c (NGC 6086),
* Gaia DR2 / DR3 / DR1 parallax systematics, asymmetric parallax errors,
* compiled type filters expand into SIMBAD's hierarchy and NED/SDSS vocabularies,
* mode-specific fields, cylinder mode on catalogs without parallax, ADQL positions,
* request-level frame hints, signed / Greek coordinates, citation dashes, CLI limits.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from astropy.cosmology import Planck18
from fastapi.testclient import TestClient
from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client
from test_ai import (  # noqa: F401  (sesame_replay and app_factory are pytest fixtures)
    C3C273_SESAME,
    M87_SESAME,
    FakeAnthropic,
    _no_credentials,
    _sources,
    app_factory,
    compile_with,
    recorder,
    reply,
    sesame_replay,
    submission,
    tool_results,
    tool_use,
)

import ai
from crossmatch import AdvancedQuery
from models import CatalogRegistry, haversine_arcsec

SGR_A_STAR = (266.41684, -29.00781)  # ICRS position given in the reviewer's request (Sgr A*)


async def _facts(set_name: str, *, max_references: int = 2, **kwargs: Any) -> ai.ObjectFacts:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"ai/{set_name}")))
        async with offline_client() as client:
            return await ai.gather_facts(http_client=client, include_crossmatch=False, max_references=max_references,
                                         **kwargs)


def _q(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["quantity"]: e for e in entries}


def _source(facts: ai.ObjectFacts, n: int) -> ai.Source:
    return next(s for s in facts.sources if s.n == n)


# ---------------------------------------------------------------------------
# SIMBAD main type decides (stray "G" in the type list)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("set_name", "name", "otype", "plx_mas", "distance_pc"),
    [
        ("explain_helix", "NGC 7293", "PN", 5.0124, 199.5),  # Helix Nebula, Gaia (E)DR3
        ("explain_mira", "Mira", "Mi*", 10.91, 92.0),  # Hipparcos
        ("explain_47tuc", "47 Tuc", "GlC", 0.232, 4310.0),  # Gaia EDR3 cluster mean (Vasiliev & Baumgardt 2021)
    ],
)
async def test_galactic_objects_with_a_stray_galaxy_type_keep_their_parallax_distance(
    set_name: str, name: str, otype: str, plx_mas: float, distance_pc: float
) -> None:
    facts = await _facts(set_name, name=name)
    assert facts.identity["otype"] == otype
    assert "G" in facts.identity["all_otypes"]  # the stray confirmed galaxy type is really there
    parallax = _q(facts.measurements)["parallax"]
    assert parallax["value"] == pytest.approx(plx_mas)
    assert "note" not in parallax or "extragalactic" not in parallax["note"]
    dist = _q(facts.derived)["parallax_distance"]
    assert dist["value"] == pytest.approx(distance_pc, rel=0.002)
    assert dist["value"] == pytest.approx(1000.0 / plx_mas, abs=dist["plus_error"] / 20)  # rounded to its error
    assert not any("withheld" in w for w in facts.warnings)
    assert not any(d["quantity"] == "luminosity_distance" for d in facts.derived)


def _types(**codes: tuple[str, int]) -> dict[str, dict[str, Any]]:
    return {code: {"path": path, "is_candidate": cand} for code, (path, cand) in codes.items()}


@pytest.mark.parametrize(
    ("otype", "path", "others", "expected"),
    [
        ("Mi*", "* > Ev* > AB* > Mi*", _types(G=("G", 0)), (False, False)),  # Mira
        ("PN", "* > Ev* > PN", _types(G=("G", 0)), (False, False)),  # Helix
        ("GlC", "Cl* > GlC", _types(G=("G", 0)), (False, False)),  # 47 Tuc
        ("HXB", "* > ** > XB* > HXB", {"BL?": {"path": "G > AGN > QSO > Bla > BLL", "is_candidate": 1}},
         (False, False)),  # Cyg X-1
        ("Rad", "Rad", _types(G=("G", 0)), (True, True)),  # generic main type, confirmed galaxy
        ("X", "X", {"AG?": {"path": "G > AGN", "is_candidate": 1}}, (False, True)),  # generic main, candidate only
        ("?", "", {"QSO": {"path": "G > AGN > QSO", "is_candidate": 0}}, (True, True)),
        ("Sy1", "G > AGN > SyG > Sy1", {}, (True, True)),
        ("LeQ", "grv > gLS > LeI > LeQ", {}, (True, True)),
    ],
)
def test_simbad_main_type_decides_extragalactic(otype: str, path: str, others: dict[str, Any],
                                                expected: tuple[bool, bool]) -> None:
    assert ai.classify_simbad_types(otype, path, others) == expected


def test_generic_main_type_with_candidate_galaxy_class_withholds_parallax_distance_only() -> None:
    facts = ai.ObjectFacts(query={}, object={})
    facts.identity = {"otype": "X", "all_otypes": ["AG?", "X"]}
    warnings: list[str] = []
    row = {"ra": 10.0, "dec": 20.0, "plx_bibcode": "2007A&A...474..653V"}
    ai._add_derived(facts, row, ai.SourceBook(), warnings, extragalactic=False, any_extragalactic=True, z_input=None,
                    rvz_qual=None, photometric=False, plx=2.0, plx_err=0.1, plx_qual="A", plx_ref=1)
    assert facts.derived == [] and any("withheld" in w for w in warnings)


# ---------------------------------------------------------------------------
# Redshift precision, velocities, quality and parallax S/N guards
# ---------------------------------------------------------------------------


async def test_ton618_hubble_flow_errors_use_the_stated_redshift_precision() -> None:
    """SIMBAD: z = 2.2 (one decimal, no error). The distance error must reflect +-0.05, not 250 km/s alone."""
    facts = await _facts("explain_ton618", name="Ton 618")
    z = _q(facts.measurements)["redshift"]
    assert z["value"] == 2.2 and "error" not in z and z["decimals"] == 1
    assert "SIMBAD gives no error" in z["note"]
    derived = _q(facts.derived)
    z_cmb = derived["redshift_cmb_frame"]
    assert z_cmb["value"] == 2.2  # rounded to the stated precision, not 2.20291
    assert z_cmb["error"] == 0.05 and z_cmb["error_inferred"] is True
    assert "SIMBAD gives no error" in z_cmb["note"]
    dl = derived["luminosity_distance"]
    assert dl["plus_error"] >= 200 and dl["minus_error"] >= 200  # was +-3.6 Mpc
    assert "assumed measurement error (SIMBAD gives none)" in dl["note"]
    # The better-determined literature redshift 2.219 lies within the quoted error (it was ~49 sigma away).
    z_better = 2.219
    dl_better = (1 + z_better) * Planck18.comoving_transverse_distance(z_better).value
    assert abs(dl_better - dl["value"]) < dl["plus_error"]
    assert derived["lookback_time"]["plus_error"] >= 0.05  # Gyr, was 0.0011
    # The SIMBAD placeholder morphology '?...' is not a fact.
    assert "morphological_type" not in _q(facts.measurements)
    assert all(m["ref"] is not None for m in facts.measurements)


def test_redshift_without_error_or_precision_gets_no_cosmology() -> None:
    facts = ai.ObjectFacts(query={}, object={})
    facts.identity = {"otype": "QSO"}
    warnings: list[str] = []
    z_input = ai.RedshiftInput(1.5, None, False, None, "redshift")
    ai._add_derived(facts, {"ra": 10.0, "dec": 20.0}, ai.SourceBook(), warnings, extragalactic=True,
                    any_extragalactic=True, z_input=z_input, rvz_qual="A", photometric=False, plx=None, plx_err=None,
                    plx_qual=None, plx_ref=None)
    assert facts.derived == [] and any("neither an error nor a precision" in w for w in warnings)


@pytest.mark.parametrize("quality", ["D", "E", None])
def test_poor_quality_redshift_gets_no_cosmology(quality: str | None) -> None:
    facts = ai.ObjectFacts(query={}, object={})
    facts.identity = {"otype": "G"}
    warnings: list[str] = []
    z_input = ai.RedshiftInput(0.05, 0.0001, False, 4, "redshift")
    ai._add_derived(facts, {"ra": 10.0, "dec": 20.0}, ai.SourceBook(), warnings, extragalactic=True,
                    any_extragalactic=True, z_input=z_input, rvz_qual=quality, photometric=False, plx=None,
                    plx_err=None, plx_qual=None, plx_ref=None)
    assert facts.derived == [] and any("quality" in w for w in warnings)
    good = ai.ObjectFacts(query={}, object={})
    good.identity = {"otype": "G"}
    ai._add_derived(good, {"ra": 10.0, "dec": 20.0}, ai.SourceBook(), [], extragalactic=True, any_extragalactic=True,
                    z_input=z_input, rvz_qual="C", photometric=False, plx=None, plx_err=None, plx_qual=None,
                    plx_ref=None)
    assert "luminosity_distance" in _q(good.derived)


@pytest.mark.parametrize(("plx", "err", "expected"), [(2.0, 0.5, False), (4.9, 1.0, False), (5.0, 1.0, True),
                                                      (10.0, 1.0, True)])
def test_parallax_distance_needs_signal_to_noise_five(plx: float, err: float, expected: bool) -> None:
    facts = ai.ObjectFacts(query={}, object={})
    facts.identity = {"otype": "*"}
    ai._add_derived(facts, {"ra": 1.0, "dec": 2.0}, ai.SourceBook(), [], extragalactic=False, any_extragalactic=False,
                    z_input=None, rvz_qual=None, photometric=False, plx=plx, plx_err=err, plx_qual="A", plx_ref=1)
    assert ("parallax_distance" in _q(facts.derived)) is expected


async def test_galaxy_velocity_becomes_cz_over_c_and_gets_hubble_flow_distances() -> None:
    """NGC 6086: v = 9581 +- 5 km/s. z = v/c = 0.031959, not SIMBAD's Doppler 0.032486."""
    facts = await _facts("explain_ngc6086", name="NGC 6086")
    measured = _q(facts.measurements)
    assert measured["radial_velocity"]["value"] == 9581 and measured["radial_velocity"]["error"] == 5
    z = measured["redshift"]
    assert z["value"] == pytest.approx(9581 / ai.SPEED_OF_LIGHT_KMS, abs=1e-6)
    assert abs(z["value"] - 0.03248618) > 5e-4
    assert z["error"] == pytest.approx(5 / ai.SPEED_OF_LIGHT_KMS, rel=0.05)
    assert "cz/c" in z["note"]
    derived = _q(facts.derived)
    assert derived["redshift_cmb_frame"]["value"] == pytest.approx(
        ai.cmb_frame_redshift(9581 / ai.SPEED_OF_LIGHT_KMS, facts.object["ra_deg"], facts.object["dec_deg"]), abs=1e-6)
    dl = derived["luminosity_distance"]
    assert 135 < dl["value"] < 155 and dl["plus_error"] >= 3  # dominated by the 250 km/s scatter


async def test_nearby_galaxy_velocity_below_hubble_flow_limit() -> None:
    """NGC 4639 (v = 981 km/s): z = v/c reported, no Hubble-flow distance at z < 0.01."""
    facts = await ai_facts_ngc4639()
    z = _q(facts.measurements)["redshift"]
    assert z["value"] == pytest.approx(981 / ai.SPEED_OF_LIGHT_KMS, abs=1e-6)
    assert abs(z["value"] - 0.0032776351913594848) > 4e-6  # SIMBAD's relativistic conversion is not used
    assert "luminosity_distance" not in _q(facts.derived)


async def ai_facts_ngc4639() -> ai.ObjectFacts:
    return await _facts("explain_ngc4639", name="NGC 4639")


# ---------------------------------------------------------------------------
# Gaia parallax systematics and asymmetric errors
# ---------------------------------------------------------------------------


async def test_pleiades_dr2_based_parallax_states_zero_point_and_systematics() -> None:
    facts = await _facts("explain_pleiades", name="Pleiades")
    dist = _q(facts.derived)["parallax_distance"]
    assert dist["gaia_release"] == "Gaia DR2"
    assert _source(facts, dist["ref"]).bibcode == "2018A&A...616A..10G"
    assert _source(facts, dist["zero_point_ref"]).bibcode == "2018A&A...616A...2L"
    assert _source(facts, dist["systematics_ref"]).bibcode == "2018A&A...616A...2L"
    assert "-0.029 mas" in dist["note"] and "not recorded" in dist["note"]
    assert "below the 0.04 mas" in dist["note"] and "understate" in dist["note"]


async def test_cyg_x1_gaia_zero_point_exceeds_formal_error() -> None:
    """Cyg X-1: plx 0.4439 +- 0.0149 mas; |-0.017| mas is 3.8 % of it, the formal error 3.4 %."""
    facts = await _facts("explain_cygx1", name="Cyg X-1")
    assert facts.identity["otype"] == "HXB" and "BL?" in facts.identity["all_otypes"]
    dist = _q(facts.derived)["parallax_distance"]
    assert dist["gaia_release"] == "Gaia (E)DR3"
    assert "catalogue value" in dist["note"] and "NOT applied" in dist["note"]
    assert "larger than the formal parallax error" in dist["note"]
    assert _source(facts, dist["zero_point_ref"]).bibcode == ai.GAIA_ZERO_POINT_BIBCODE
    assert "systematics_ref" not in dist  # 0.0149 mas is above the 0.01 mas floor


async def test_helix_zero_point_smaller_than_formal_error() -> None:
    dist = _q((await _facts("explain_helix", name="NGC 7293")).derived)["parallax_distance"]
    assert "NOT applied" in dist["note"] and "larger than the formal parallax error" not in dist["note"]


async def test_47tuc_gaia_based_parallax_below_the_systematic_floor() -> None:
    facts = await _facts("explain_47tuc", name="47 Tuc")
    dist = _q(facts.derived)["parallax_distance"]
    assert _source(facts, dist["ref"]).bibcode == "2021MNRAS.505.5978V"
    assert "not recorded" in dist["note"]  # Vasiliev & Baumgardt applied a zero point; the module does not claim otherwise
    assert "below the 0.01 mas" in dist["note"]


def test_gaia_release_attribution() -> None:
    assert ai.gaia_release_of("2016A&A...595A...2G", None)[0].systematic_floor_mas == 0.3  # type: ignore[union-attr]
    assert ai.gaia_release_of("2018yCat.1345....0G", "Gaia DR2") == (ai.GAIA_RELEASES[1], True)
    assert ai.gaia_release_of("2023A&A...674A...1G", None) == (ai.GAIA_RELEASES[0], True)
    assert ai.gaia_release_of("2099XYZ...1....1A", "{\\em Gaia} Data Release 3 stellar clusters")[0] is ai.GAIA_RELEASES[0]
    assert ai.gaia_release_of("2007A&A...474..653V", "Validation of the new Hipparcos reduction.") == (None, False)


async def test_betelgeuse_parallax_distance_errors_are_asymmetric() -> None:
    """plx 6.55 +- 0.83 mas: 1/(plx - err) - 1/plx = 21.8 pc, 1/plx - 1/(plx + err) = 17.1 pc."""
    facts = await _facts("explain_betelgeuse", name="Betelgeuse")
    plx = _q(facts.measurements)["parallax"]
    dist = _q(facts.derived)["parallax_distance"]
    plus = 1000 / (plx["value"] - plx["error"]) - 1000 / plx["value"]
    minus = 1000 / plx["value"] - 1000 / (plx["value"] + plx["error"])
    assert dist["plus_error"] == pytest.approx(plus, abs=0.6)
    assert dist["minus_error"] == pytest.approx(minus, abs=0.6)
    assert dist["plus_error"] - dist["minus_error"] >= 4


# ---------------------------------------------------------------------------
# SIMBAD position matching, bibliography and ADS
# ---------------------------------------------------------------------------


def _tap_json(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"metadata": [{"name": c, "datatype": "char"} for c in columns], "data": rows}


async def test_simbad_lookup_prefers_the_star_over_its_planet_at_the_same_position() -> None:
    columns = ["oid", "ra", "dec", "pmra", "pmdec", "nbref", "dist_deg"]
    rows = [
        [2, 10.0, 20.0, None, None, 12, 0.0],  # planet: nearer by 0.02", few references
        [1, 10.0, 20.0 + 0.02 / 3600, None, None, 900, 0.02 / 3600],  # the host star
        [3, 10.0, 20.0 + 1.0 / 3600, None, None, 5000, 1.0 / 3600],  # another object 1" away
    ]
    with respx.mock(assert_all_mocked=True) as router:
        router.post(ai.SIMBAD_TAP).mock(return_value=httpx.Response(200, json=_tap_json(columns, rows)))
        async with offline_client() as client:
            best = await ai._simbad_lookup(client, 10.0, 20.0, 5.0, ai.SIMBAD_TAP)
    assert best is not None and best["oid"] == 1


async def test_focused_papers_are_distinct_from_recent_and_foundational() -> None:
    facts = await _facts("explain_3c273", name="3C 273", max_references=8)
    biblio = facts.bibliography
    focused = {e["bibcode"] for e in biblio["focused"]}
    assert focused
    assert not focused & {e["bibcode"] for e in biblio["recent"]}
    assert not focused & {e["bibcode"] for e in biblio["foundational"]}


async def test_focused_list_drops_papers_already_listed_as_recent_or_foundational() -> None:
    """SIMBAD answers (shape of the real ref/has_ref reply) where the top obj_freq papers overlap the other lists."""
    columns = ["bibcode", "title", "pub_year", "journal", "nbobject", "doi", "ref_flag", "obj_freq"]

    def row(bibcode: str, year: int, freq: int) -> list[Any]:
        return [bibcode, f"Paper {bibcode}", year, "ApJ", 1, None, 1, freq]

    recent = [row("2025ApJ...900....1A", 2025, 50)]
    foundational = [row("1963Natur.197.1040S", 1963, 40)]
    focused = [row("2025ApJ...900....1A", 2025, 50), row("1963Natur.197.1040S", 1963, 40),
               row("2010ApJ...700....1B", 2010, 30), row("2011ApJ...701....1C", 2011, 20)]

    def answer(request: httpx.Request) -> httpx.Response:
        query = httpx.QueryParams(request.content.decode()).get("QUERY", "")
        rows = focused if "obj_freq DESC" in query else recent if "pub_year DESC" in query else foundational
        return httpx.Response(200, json=_tap_json(columns, rows))

    with respx.mock(assert_all_mocked=True) as router:
        router.post(ai.SIMBAD_TAP).mock(side_effect=answer)
        async with offline_client() as client:
            biblio = await ai.fetch_simbad_bibliography(client, 12345, limit=2)
    assert [e["bibcode"] for e in biblio["focused"]] == ["2010ApJ...700....1B", "2011ApJ...701....1C"]
    assert [e["bibcode"] for e in biblio["recent"]] == ["2025ApJ...900....1A"]
    assert [e["bibcode"] for e in biblio["foundational"]] == ["1963Natur.197.1040S"]


async def test_ads_ranking_is_merged_into_the_bibliography() -> None:
    body = {"responseHeader": {"status": 0}, "response": {"numFound": 2, "docs": [
        {"bibcode": "2016Natur.536..437A", "title": [("A terrestrial planet candidate in a temperate orbit around "
                                                      "Proxima Centauri")], "year": "2016", "citation_count": 1},
        {"bibcode": "2020Sci...368.1227S", "title": ["An Earth-like planet in the habitable zone of Proxima"],
         "year": "2020", "citation_count": 1},
    ]}}
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        ads = router.get(ai.ADS_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_proxima")))
        async with offline_client() as client:
            facts = await ai.gather_facts(http_client=client, name="Proxima Centauri", include_crossmatch=False,
                                          max_references=2, ads_token="token-xyz")
    assert ads.called and ads.calls.last.request.headers["Authorization"] == "Bearer token-xyz"
    cited = facts.bibliography["most_cited_ads"]
    assert [c["bibcode"] for c in cited] == ["2016Natur.536..437A", "2020Sci...368.1227S"]
    for entry in cited:
        source = _source(facts, entry["ref"])
        assert source.bibcode == entry["bibcode"] and source.title and source.url.endswith("/abstract")


# ---------------------------------------------------------------------------
# Compiled object-type filters
# ---------------------------------------------------------------------------


@pytest.fixture
async def otype_tree(sesame_replay) -> AsyncIterator[dict[str, Any]]:
    ai._OTYPE_TREES.clear()
    async with offline_client() as client:
        yield await ai.simbad_otype_tree(client)


async def test_class_words_expand_into_simbad_hierarchy_and_ned_vocabulary(otype_tree) -> None:
    codes, errors = ai.expand_object_types(["quasar"], ["simbad", "ned"], otype_tree)
    assert not errors
    assert {"QSO", "Q?", "Bla", "Bz?", "BLL", "BL?"} <= set(codes)
    assert "AGN" not in codes and "Sy1" not in codes
    stars, errors = ai.expand_object_types(["star"], ["simbad", "ned", "sdss"], otype_tree)
    assert not errors
    assert {"*", "PM*", "Mi*", "HXB", "WD*", "V*", "!V*", "STAR"} <= set(stars)
    assert not {"PN", "PN?", "Pl", "Pl?", "G", "QSO"} & set(stars)
    agn, errors = ai.expand_object_types(["AGN"], ["simbad"], otype_tree)
    assert not errors and {"AGN", "AG?", "Sy1", "Sy2", "LIN", "QSO", "BLL"} <= set(agn) and "G" not in agn


async def test_unknown_types_and_codes_without_ned_equivalent_are_rejected(otype_tree) -> None:
    _, errors = ai.expand_object_types(["M dwarf"], ["simbad"], otype_tree)
    assert errors and "neither a class word" in errors[0]
    _, errors = ai.expand_object_types(["Sy1"], ["simbad", "ned"], otype_tree)
    assert errors and "no NED equivalent" in errors[0]
    _, errors = ai.expand_object_types(["nebula"], ["simbad", "sdss"], otype_tree)
    assert errors and "no SDSS equivalent" in errors[0]
    codes, errors = ai.expand_object_types(["PN"], ["simbad", "ned"], otype_tree)
    assert not errors and {"PN", "PN?", "!PN"} <= set(codes)


async def test_compiled_type_filter_keeps_real_rows(sesame_replay) -> None:
    """3C 273 is 'BLL' in SIMBAD and 'QSO' in NED; Proxima is 'PM*'; Mira is 'V*' in NED."""
    ai._OTYPE_TREES.clear()
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(object_types=["quasar"], catalogs=["simbad", "ned"]))))
    compiled = await compile_with(fake)
    query = AdvancedQuery.from_dict(compiled.advanced_query)
    assert compiled.advanced_query["metadata"]["object_types_requested"] == ["quasar"]
    assert query.apply_filters({"catalog": "simbad", "data": {"object_type": "BLL"}, "separation_arcsec": 0.0})
    assert query.apply_filters({"catalog": "ned", "data": {"object_type": "QSO"}, "separation_arcsec": 0.0})
    assert not query.apply_filters({"catalog": "simbad", "data": {"object_type": "PM*"}, "separation_arcsec": 0.0})
    stars = FakeAnthropic(reply(tool_use("submit_query", submission(object_types=["star"], catalogs=["simbad", "ned"]))))
    star_query = AdvancedQuery.from_dict((await compile_with(stars)).advanced_query)
    assert star_query.apply_filters({"catalog": "simbad", "data": {"object_type": "PM*"}, "separation_arcsec": 0.0})
    assert star_query.apply_filters({"catalog": "ned", "data": {"object_type": "V*"}, "separation_arcsec": 0.0})
    assert not star_query.apply_filters({"catalog": "simbad", "data": {"object_type": "PN"}, "separation_arcsec": 0.0})


async def test_invented_type_is_fed_back(sesame_replay) -> None:
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(object_types=["x-ray bright dwarf"], catalogs=["simbad"]))),
        reply(tool_use("submit_query", submission(object_types=["Sy1"], catalogs=["simbad", "ned"]))),
        reply(tool_use("submit_query", submission(object_types=["Sy1"], catalogs=["simbad"]))),
    )
    compiled = await compile_with(fake)
    (first,) = tool_results(fake.calls[1])
    assert first["is_error"] and "neither a class word" in first["content"]
    (second,) = tool_results(fake.calls[2])
    assert second["is_error"] and "no NED equivalent" in second["content"]
    assert compiled.advanced_query["object_types"] == ["Sy1"]


# ---------------------------------------------------------------------------
# Mode-specific fields, cylinder mode, all-sky filters
# ---------------------------------------------------------------------------


async def test_cylinder_mode_on_catalogs_without_parallax_is_fed_back(sesame_replay) -> None:
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(search_mode="cylinder", max_distance_pc=20.0,
                                                  catalogs=["chandra", "xmm", "gaia_dr3"]))),
        reply(tool_use("submit_query", submission(search_mode="cylinder", max_distance_pc=20.0,
                                                  catalogs=["gaia_dr3", "simbad"]))),
    )
    compiled = await compile_with(fake, "X-ray emitting stars closer than 20 pc near M87")
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] and "would remove every row of chandra, xmm" in result["content"]
    assert "gaia_dr3, simbad" in result["content"]
    query = AdvancedQuery.from_dict(compiled.advanced_query)
    assert query.search_mode == "cylinder" and query.max_distance_pc == 20.0
    assert query.apply_filters({"catalog": "gaia_dr3", "data": {"parallax": 100.0}, "separation_arcsec": 1.0})
    assert not query.apply_filters({"catalog": "gaia_dr3", "data": {"parallax": 1.0}, "separation_arcsec": 1.0})


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"max_distance_pc": 20.0}, "only act in search_mode cylinder"),
        ({"min_distance_pc": 5.0}, "only act in search_mode cylinder"),
        ({"min_radius_arcsec": 30.0}, "only acts in search_mode shell"),
        ({"search_mode": "shell"}, "needs min_radius_arcsec > 0"),
    ],
)
async def test_ignored_mode_fields_are_fed_back(sesame_replay, overrides: dict[str, Any], fragment: str) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(**overrides))), reply(tool_use("submit_query", submission())))
    await compile_with(fake)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] and fragment in result["content"]


async def test_shell_mode_with_inner_radius_is_accepted(sesame_replay) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(search_mode="shell", min_radius_arcsec=30.0))))
    query = AdvancedQuery.from_dict((await compile_with(fake)).advanced_query)
    assert not query.apply_filters({"catalog": "nvss", "data": {}, "separation_arcsec": 10.0})
    assert query.apply_filters({"catalog": "nvss", "data": {}, "separation_arcsec": 60.0})


async def test_all_sky_scope_rejects_crossmatch_only_filters() -> None:
    adql = "SELECT TOP 10 b.main_id AS main_id FROM basic AS b WHERE b.plx_value > 50"
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(scope="all_sky", target=None, adql={"catalog": "simbad", "query": adql},
                                                  object_types=["star"], max_distance_pc=20.0, search_mode="cylinder"))),
        reply(tool_use("submit_query", submission(scope="all_sky", target=None, adql={"catalog": "simbad", "query": adql}))),
    )
    compiled = await compile_with(fake, "stars within 20 pc")
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] and "object_types, search_mode, max_distance_pc would be ignored" in result["content"]
    assert compiled.adql == adql


# ---------------------------------------------------------------------------
# ADQL positions come from the validated target
# ---------------------------------------------------------------------------


async def test_cone_adql_with_model_coordinates_is_rejected_and_placeholder_substituted(sesame_replay) -> None:
    m31_circle = ("SELECT TOP 100 b.main_id AS main_id FROM basic AS b WHERE "
                  "CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', 10.6847, 41.2690, 0.5)) = 1")
    unconstrained = "SELECT TOP 100 b.main_id AS main_id FROM basic AS b WHERE b.otype = 'QSO'"
    template = ("SELECT TOP 100 b.main_id AS main_id, DISTANCE(POINT('ICRS', b.ra, b.dec), {TARGET_POINT}) AS d "
                "FROM basic AS b WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), {TARGET_CIRCLE}) = 1 ORDER BY d")
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(adql={"catalog": "simbad", "query": m31_circle}))),
        reply(tool_use("submit_query", submission(adql={"catalog": "simbad", "query": unconstrained}))),
        reply(tool_use("submit_query", submission(adql={"catalog": "simbad", "query": template}))),
    )
    compiled = await compile_with(fake, "radio sources within 15 arcmin of 12h30m49.42s +12d23m28.0s, i.e. M87")
    (first,) = tool_results(fake.calls[1])
    assert first["is_error"] and "must not contain numeric sky positions" in first["content"]
    (second,) = tool_results(fake.calls[2])
    assert second["is_error"] and "{TARGET_CIRCLE}" in second["content"]
    ra, dec = M87_SESAME
    assert f"CIRCLE('ICRS', {ra:.9f}, {dec:.9f}, {900.0 / 3600:.10f})" in compiled.adql
    assert f"POINT('ICRS', {ra:.9f}, {dec:.9f})" in compiled.adql
    assert "{" not in compiled.adql


def test_adql_position_rules() -> None:
    assert ai.adql_position_problems("SELECT TOP 1 x FROM t WHERE CONTAINS(POINT(ra, dec), CIRCLE(1.0, 2.0, 0.1)) = 1",
                                     "all_sky")
    assert ai.adql_position_problems("SELECT TOP 1 x FROM t WHERE CONTAINS(POINT('ICRS', ra, dec), {TARGET_CIRCLE})=1",
                                     "all_sky")
    assert ai.adql_position_problems("SELECT TOP 1 x FROM t WHERE {TARGET_BOX}", "cone")
    assert not ai.adql_position_problems("SELECT TOP 1 x FROM t WHERE CONTAINS(POINT('ICRS', ra, dec), {TARGET_CIRCLE})=1",
                                         "cone")


# ---------------------------------------------------------------------------
# Frames: request-level hints, signs, Greek labels
# ---------------------------------------------------------------------------


def coords(text: str, frame: str = "icrs", equinox: str | None = None) -> dict[str, Any]:
    return {"name": None, "coordinates": {"text": text, "frame": frame, "equinox": equinox}}


async def test_b1950_designation_elsewhere_does_not_force_fk4() -> None:
    text = "radio sources within 60 arcsec of 12h29m06.7s +02d03m09s, i.e. the quasar with B1950 designation 1226+023"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(target=coords("12h29m06.7s +02d03m09s"),
                                                                   radius_arcsec=60.0))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *C3C273_SESAME) < 1.0
    assert any("B1950/FK4 elsewhere" in w for w in compiled.warnings)


async def test_galactic_mention_for_another_position_does_not_force_galactic() -> None:
    text = "X-ray sources within 60 arcsec of 266.41684 -29.00781 (Sgr A*, galactic coordinates l=359.944, b=-0.046)"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(
        target=coords("266.41684 -29.00781"), radius_arcsec=60.0, catalogs=["chandra", "xmm", "rosat"]))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *SGR_A_STAR) < 0.1


@pytest.mark.parametrize("coord_text", ["12h29m06.7s +02d03m09s (J2000)", "12h29m06.7s +02d03m09s"])
async def test_j2000_label_with_b1950_mention_is_accepted(coord_text: str) -> None:
    text = "radio sources within 60 arcsec of 12h29m06.7s +02d03m09s (J2000), the quasar with B1950 designation 1226+023"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(target=coords(coord_text), radius_arcsec=60.0))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *C3C273_SESAME) < 1.0


@pytest.mark.parametrize("text", ["radio sources within 60 arcsec of B1950 12h26m33.25s +02d19m43.3s",
                                  "radio sources within 60 arcsec of 12h26m33.25s +02d19m43.3s (B1950)"])
async def test_b1950_qualifying_the_coordinates_still_requires_fk4(text: str) -> None:
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target=coords("12h26m33.25s +02d19m43.3s"), radius_arcsec=60.0))),
        reply(tool_use("submit_query", submission(target=coords("12h26m33.25s +02d19m43.3s", "fk4"), radius_arcsec=60.0))),
    )
    compiled = await compile_with(fake, text)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] and "use frame fk4" in result["content"]
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *C3C273_SESAME) < 1.0


def test_fk4_without_equinox_means_b1950() -> None:
    """3C 273 at its B1950 position (12h26m33.25s +02d19m43.3s) lands on the ICRS source, not 0.7 deg away."""
    default = ai.parse_user_coordinates("12h26m33.25s +02d19m43.3s", "fk4", None)
    explicit = ai.parse_user_coordinates("12h26m33.25s +02d19m43.3s", "fk4", "B1950")
    assert haversine_arcsec(default.ra, default.dec, *C3C273_SESAME) < 1.0
    assert haversine_arcsec(default.ra, default.dec, explicit.ra, explicit.dec) < 1e-6


@pytest.mark.parametrize("text", ["-10.5 +20.0", "RA=-10.5, Dec=+20.0", "-01h00m00s +10d00m00s", "−10.5 20.0"])
def test_negative_ra_is_rejected_not_wrapped(text: str) -> None:
    with pytest.raises(ValueError, match="negative"):
        ai.parse_user_coordinates(text, "icrs")


def test_signed_galactic_longitude_is_accepted() -> None:
    signed = ai.parse_user_coordinates("l=-0.056, b=-0.046", "galactic")
    positive = ai.parse_user_coordinates("l=359.944, b=-0.046", "galactic")
    assert haversine_arcsec(signed.ra, signed.dec, positive.ra, positive.dec) < 1e-6
    assert haversine_arcsec(signed.ra, signed.dec, *SGR_A_STAR) < 2.0
    with pytest.raises(ValueError):
        ai.parse_user_coordinates("l=-190.0, b=0.0", "galactic")


@pytest.mark.parametrize(
    ("text", "frame", "expected"),
    [("α=10.684708, δ=+41.268750", "icrs", (10.684708, 41.26875)),
     ("α = 00h42m44.33s, δ = +41d16m07.5s", "icrs", (10.684708, 41.26875)),
     ("ℓ=359.944, b=-0.046", "galactic", SGR_A_STAR)],
)
def test_greek_coordinate_labels(text: str, frame: str, expected: tuple[float, float]) -> None:
    target = ai.parse_user_coordinates(text, frame)
    assert haversine_arcsec(target.ra, target.dec, *expected) < 2.0


def test_greek_labels_state_their_frame() -> None:
    assert ai.coordinate_hints("α=10.68, δ=41.27").kind == "equatorial"
    assert ai.coordinate_hints("ℓ=359.9, b=-0.05").kind == "galactic"
    with pytest.raises(ValueError, match="galactic"):
        ai.parse_user_coordinates("α=10.684708, δ=+41.268750", "galactic")


# ---------------------------------------------------------------------------
# Citations, CLI limits and input-before-credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["[3−5]", "[3‐5]", "[3‒5]", "[3 − 5]"])
def test_unicode_minus_and_hyphen_citation_ranges(marker: str) -> None:
    text, cited, warnings = ai.map_citations(f"Unicode minus {marker}.", _sources(5))
    assert text == "Unicode minus [1, 2, 3]."
    assert [s.bibcode for s in cited] == [s.bibcode for s in _sources(5)[2:5]]
    assert not any("Unrecognised" in w or "unparseable" in w for w in warnings)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    ai.register_cli(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.parametrize("limit", ["0", "-5", "2001", "ten"])
def test_cli_limit_is_bounded(limit: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        _parser().parse_args(["ask", "stars", "--limit", limit])
    assert info.value.code == 2
    assert "--limit" in capsys.readouterr().err


async def test_execute_compiled_query_rejects_bad_row_limits() -> None:
    compiled = ai.CompiledQuery(
        text="x", scope="all_sky", advanced_query=None, plan=["run"], explanation="x", adql=recorder.GOOD_ADQL,
        adql_catalog="simbad", adql_endpoint=ai.SIMBAD_TAP, resolved_objects=[], attempts=1, validation_history=[],
        model="test",
    )
    async with offline_client() as client:
        for bad in (0, -5, ai.ADQL_MAX_TOP + 1):
            with pytest.raises(ValueError, match="row limit"):
                await ai.execute_compiled_query(compiled, http_client=client, registry=CatalogRegistry(), row_limit=bad)


def test_cli_ask_validates_text_before_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    _no_credentials(monkeypatch, tmp_path)

    def no_client() -> Any:
        raise AssertionError("credentials must not be resolved for invalid input")

    monkeypatch.setattr(ai, "anthropic_factory", no_client)
    args = _parser().parse_args(["ask", "ab"])
    assert args.handler(args) == 2
    assert "at least 3 characters" in capsys.readouterr().err


def test_router_query_validates_text_before_credentials(app_factory) -> None:
    with TestClient(app_factory()) as client:
        response = client.post("/api/v1/ai/query", json={"text": "  ab  "})
    assert response.status_code == 422, response.text
    assert "at least 3 characters" in response.text


def test_check_query_request() -> None:
    assert ai.check_query_request("  quasars   near M87 ") == "quasars near M87"
    for text, retries in (("ab", 2), ("x" * 2001, 2), ("valid text", 3), ("valid text", True)):
        with pytest.raises(ValueError):
            ai.check_query_request(text, retries)  # type: ignore[arg-type]


def test_math_sanity_of_half_last_digit() -> None:
    assert ai._half_last_digit(1) == pytest.approx(0.05)
    assert ai._half_last_digit(4) == pytest.approx(5e-5)
    assert ai._half_last_digit(None) is None and ai._half_last_digit(32767) is None
    assert math.isclose(ai.hubble_flow_redshift_error(0.05), math.hypot(0.05, 250 / ai.SPEED_OF_LIGHT_KMS))
