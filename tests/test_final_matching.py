"""Final review: target identity in name and coordinate searches (recorded real cones, strict replay).

* M82 by name: SIMBAD's resolved 'M 82' row was merged as a duplicate into the radio source
  'EQ J095552.5+694045.4' 1.6" away and lost its identity prior (target_probability 0).
* NGC 4565, M101, NGC 7318A and RR Lyr by name: NED's own record of the object (listed as 'NGC 4565',
  'Messier 101', 'NGC 7318a', 'RR Lyr') was left out of the target group -- a galaxy's centre was
  given the resolver's precision although SIMBAD's 2MASS centre has no published error.
* Coordinates near extragalactic star clusters: several distinct clusters (M87's globular clusters,
  Stephan's Quintet's young star clusters) were all 'the extended object at the target' with P ~ 1.

Recorded with ``tests/test_matching_fixtures.py record-search <key>``.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_matching_fixtures import recorded_resolver, replay_search

import crossmatch


def target_group(record: dict[str, Any]) -> dict[str, Any]:
    return next(g for g in record["crossmatch_groups"] if g.get("contains_target"))


def member(record: dict[str, Any], catalog: str, source_id: str) -> dict[str, Any]:
    return next(m for g in record["crossmatch_groups"] for m in g["members"]
                if m["catalog"] == catalog and m["source_id"] == source_id)


def association(record: dict[str, Any]) -> dict[str, Any]:
    return record["provenance"]["association"]


async def test_m82_resolved_row_stays_the_target_and_represents_its_duplicate() -> None:
    record = (await replay_search("named_m82"))["record"]
    group = target_group(record)
    m82 = member(record, "simbad", "M 82")
    assert m82 in group["members"], [m["source_id"] for m in group["members"]]
    assert m82["target_probability"] > 0.99 and m82["coincident_with"] is None
    # The radio transient SIMBAD lists 1.6" away (5 mas error; 3.1 sigma from the galaxy's 0.5"
    # centre) is a separate source; were it a duplicate listing it would follow 'M 82', never
    # the reverse.
    radio = member(record, "simbad", "EQ J095552.5+694045.4")
    assert radio["coincident_with"] in (None, "M 82") and radio not in group["members"]
    reasons = {(r["catalog"], r["source_id"]): r["reason"] for r in association(record)["identity_rows"]}
    assert reasons[("simbad", "M 82")] == "resolved name"
    # NED's own record of M82, 2.7" from SIMBAD's centre, is the same galaxy.
    assert member(record, "ned", "Messier 082") in group["members"]
    assert member(record, "ned", "Messier 082")["target_probability"] > 0.95


async def test_m82_simbad_alone_at_two_arcsec_is_the_resolved_row() -> None:
    # The reviewed search: SIMBAD alone, 2": 'M 82' among 20 radio / X-ray / cluster entries of
    # the starburst within 2". The target is 'M 82' alone, never a starburst source.
    record = (await replay_search("named_m82_simbad_2arcsec"))["record"]
    group = target_group(record)
    assert [(m["source_id"], m["coincident_with"]) for m in group["members"]] == [("M 82", None)]
    assert group["members"][0]["target_probability"] > 0.99
    others = [m for g in record["crossmatch_groups"] if not g.get("contains_target") for m in g["members"]]
    assert len(others) >= 15 and all((m["target_probability"] or 0) < 0.01 for m in others),         [(m["source_id"], m["target_probability"]) for m in others if (m["target_probability"] or 0) >= 0.01]


async def test_m82_at_three_arcsec_is_the_resolved_row() -> None:
    # The reviewed 3" search with NED, 2MASS, Chandra and NVSS: 'M 82' is the target (live, before
    # the fix, the target group held a starburst X-ray / radio source and never 'M 82').
    record = (await replay_search("named_m82_3arcsec"))["record"]
    group = target_group(record)
    m82 = member(record, "simbad", "M 82")
    assert m82 in group["members"] and m82["target_probability"] > 0.99 and m82["coincident_with"] is None
    simbad_in_group = [m["source_id"] for m in group["members"] if m["catalog"] == "simbad"]
    assert simbad_in_group == ["M 82"], simbad_in_group
    # NED's X-ray / radio / infrared entries inside M82 are not collapsed onto its galaxy record.
    assert not [m["source_id"] for g in record["crossmatch_groups"] for m in g["members"]
                if m["coincident_with"] == "Messier 082"]


def test_identity_row_represents_its_duplicate_listing() -> None:
    # A resolved name's row (raised prior odds) collapsed with a more precise duplicate listing
    # of its catalogue stays the representative and keeps its prior (M 82 once followed the
    # radio source 'EQ J095552.5+694045.4' and got P = 0).
    from astrometry import IDENTITY_PRIOR_LN_ODDS, Detection, associate

    ra, dec = 148.9684583, 69.6797028
    offset = 1.5 / 3600.0
    detections = [
        Detection("simbad", ra, dec + offset, (0.005**2, 0.0, 0.005**2), label="radio", listing="one"),
        Detection("simbad", ra, dec, (0.25, 0.0, 0.25), label="M 82", listing="one",
                  prior_ln_odds=IDENTITY_PRIOR_LN_ODDS),
    ]
    result = associate(detections, {"simbad": 1.0e5}, target=(ra, dec))
    group = next(g for g in result.groups if 1 in g.members)
    assert 0 in group.members and group.coincident_with == {0: 1}
    assert result.target_probability[1] > 0.99


@pytest.mark.parametrize(("key", "ned_name", "simbad_name"), [
    ("named_ngc4565", "NGC 4565", "NGC 4565"),
    ("named_m101", "Messier 101", "M 101"),
    ("named_ngc7318a", "NGC 7318a", "NGC 7318A"),
])
async def test_ned_record_of_a_named_galaxy_is_in_the_target_group(key: str, ned_name: str, simbad_name: str) -> None:
    record = (await replay_search(key))["record"]
    group = target_group(record)
    names = {(m["catalog"], m["source_id"]) for m in group["members"]}
    assert ("ned", ned_name) in names and ("simbad", simbad_name) in names, names
    assert member(record, "ned", ned_name)["target_probability"] > 0.99
    identity = {(r["catalog"], r["source_id"]) for r in association(record)["identity_rows"]}
    assert ("ned", ned_name) in identity
    # The galaxy's centre carries the 1" scatter of catalogued galaxy centres, not a mas-level error.
    assert record["target"].get("target_sigma_arcsec", crossmatch.GALAXY_CENTRE_SIGMA_ARCSEC) >= 1.0 or \
        record["provenance"].get("target_uncertainty_source") == "galaxy_centre"


async def test_ned_rr_lyr_is_the_target_not_its_wise_entry() -> None:
    record = (await replay_search("named_rrlyr"))["record"]
    group = target_group(record)
    rr_lyr = member(record, "ned", "RR Lyr")
    assert rr_lyr in group["members"] and rr_lyr["target_probability"] > 0.9
    assert member(record, "simbad", "V* RR Lyr")["target_probability"] > 0.99


def _cluster_rows(record: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [m for g in record["crossmatch_groups"] for m in g["members"] if m["source_id"].startswith(prefix)]


async def test_m87_globular_clusters_are_compact_sources_not_extended_identities() -> None:
    record = (await replay_search("m87_halo_gcs"))["record"]
    info = association(record)
    assert info["target_class"]["class"] != "extended", info["target_class"]
    assert info["identity_rows"] == []
    clusters = _cluster_rows(record, "[JPB2009]")
    assert len(clusters) == 2
    likely = [m for m in clusters if (m["target_probability"] or 0) > 0.5]
    assert len(likely) <= 1, [(m["source_id"], m["target_probability"]) for m in clusters]
    # The cluster 2.6" from the searched position (1.9" from the other) is not the target.
    far = member(record, "simbad", "[JPB2009] 187.7183095+12.3783104")
    assert far["target_probability"] < 0.5


async def test_stephans_quintet_star_clusters_are_not_all_the_target() -> None:
    record = (await replay_search("stephan_yscs"))["record"]
    info = association(record)
    assert info["target_class"]["class"] != "extended", info["target_class"]
    assert info["identity_rows"] == []
    clusters = _cluster_rows(record, "[FGD2015]")
    assert len(clusters) >= 5
    assert sum((m["target_probability"] or 0) > 0.5 for m in clusters) <= 1, \
        [(m["source_id"], m["target_probability"]) for m in clusters]


def test_identifier_key_matches_names_across_compilations() -> None:
    key = crossmatch.identifier_key
    assert key("M 101") == key("Messier 101") == key("M101") == "m101"
    assert key("MESSIER 013") == key("M  13") == "m13"
    assert key("V* RR Lyr") == key("RR Lyr") == "rrlyr"
    assert key("NGC  7318A") == key("NGC 7318a") and key("NGC 0224") == "ngc224"
    assert key("NAME M 82 Group") != key("M82")


def test_galaxy_centre_without_an_error_gets_the_galaxy_centre_sigma() -> None:
    galaxy = recorded_resolver("named_m101").obj
    spec = crossmatch.resolved_search_target(galaxy)
    assert spec["target_uncertainty_arcsec"] == crossmatch.GALAXY_CENTRE_SIGMA_ARCSEC
    assert spec["target_uncertainty_source"] == "galaxy_centre"
    star = recorded_resolver("named_rrlyr").obj  # SIMBAD gives Gaia's errors: kept
    assert crossmatch.resolved_search_target(star)["target_uncertainty_source"] == "resolver"
    quasar = recorded_resolver("named_3c273").obj
    assert crossmatch.resolved_search_target(quasar)["target_uncertainty_source"] == "resolver"
