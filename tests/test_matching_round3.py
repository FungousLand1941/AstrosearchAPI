"""Regression tests for the adversarial review of the Bayesian matching engine (round 3).

Whole searches recorded end to end (``tests/test_matching_fixtures.py record-search``) are
replayed strictly, through the SSE stream (the resolver's answer passed to the engine) and
through what POST /api/v1/search does (the resolver's position only); synthetic rows check
the arithmetic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import respx

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import fixture_io
from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_matching_fixtures import (
    MATCHING,
    MATCHING_TARGETS,
    RECORDED_SEARCHES,
    SESAME_DIR,
    _register,
    replay_search,
)
from test_matching_round2 import members, row, search_endpoint_emulation, static_service, target_group

import astrometry
from crossmatch import (
    EXTENDED_CENTRE_SIGMA_ARCSEC,
    RESOLVER_UNDATED_SIGMA_ARCSEC,
    CrossmatchService,
    _adopt_proper_motion,
    _compact_for_extended_target,
    extended_family,
    is_binary_component,
    resolved_search_target,
)
from models import CatalogRegistry, CatalogSource, offset_radec, propagate_radec, validate_target
from providers import QueryResult, SesameResolver


def sesame_xml(stem: str) -> str:
    return (SESAME_DIR / f"{stem}.xml").read_text(encoding="utf-8")


def all_rows(record: dict) -> list[dict]:
    return [m for g in record["crossmatch_groups"] for m in g["members"]]


def by_catalog_id(record: dict) -> dict[tuple[str, str], dict]:
    return {(m["catalog"], m["source_id"]): m for m in all_rows(record)}


def both_paths(key: str):
    """The record of a named search through the stream (resolver identity) and through the
    POST /search emulation (resolver position only), as dicts."""
    async def run() -> list[tuple[str, dict]]:
        stream = (await replay_search(key))["record"]
        search = (await search_endpoint_emulation(key, min_confidence=0.0)).as_dict()
        return [("stream", stream), ("search", search)]
    return run()


# ---------------------------------------------------------------------------
# Every recorded fixture exists (a missing one must fail, never skip)
# ---------------------------------------------------------------------------


def test_every_recorded_fixture_is_present() -> None:
    _register()
    missing = [key for key in MATCHING_TARGETS if not any((MATCHING / key).glob("*.json"))]
    missing += [key for key in RECORDED_SEARCHES if not any((MATCHING / key).glob("*.json"))]
    missing += [f"sesame/{spec['sesame']}.xml" for spec in RECORDED_SEARCHES.values()
                if "sesame" in spec and not (SESAME_DIR / f"{spec['sesame']}.xml").exists()]
    assert missing == [], f"fixtures missing (record them; see tests/test_matching_fixtures.py): {missing}"
    for key in RECORDED_SEARCHES:
        for meta in (MATCHING / key).glob("*.json"):
            exchanges = json.loads(meta.read_text(encoding="utf-8"))["exchanges"]
            for index in range(len(exchanges)):
                assert (MATCHING / key / f"{meta.stem}.{index}.body").exists(), (key, meta.stem, index)


# ---------------------------------------------------------------------------
# Issue: named extended objects identified with point sources at their centre
# ---------------------------------------------------------------------------


async def test_coma_cluster_is_its_identity_not_the_galaxy_at_its_centre() -> None:
    # SIMBAD's 'ACO 1656' (quality E, no Sesame error) lies 0.03" from NGC 4876: the galaxy's
    # rows were the target group (SIMBAD 0.91, Gaia 0.9965, Chandra 0.999) and 'ACO 1656' got
    # 0.09 on the stream, 8e-6 through /search.
    for path, record in await both_paths("named_coma"):
        assoc = record["provenance"]["association"]
        assert assoc["target_class"]["class"] == "extended", path
        assert assoc["extended_centre_sigma_arcsec"] >= 60.0, path  # the identity row's own error
        rows = by_catalog_id(record)
        assert rows[("simbad", "ACO 1656")]["target_probability"] > 0.99, path
        assert rows[("ned", "GalWCat19 0003")]["target_probability"] > 0.9, path  # NED's GClstr: the same cluster
        assert rows[("ned", "GalWCat19 0003")]["target_identity"], path
        for key in (("simbad", "NGC 4876"), ("ned", "NGC 4876")):
            assert rows[key]["target_probability"] < 0.01, (path, key)
        for m in all_rows(record):
            if m["catalog"] in ("gaia_dr3", "twomass_psc", "allwise", "chandra", "xmm"):
                assert m["target_probability"] < 0.1, (path, m["catalog"], m["source_id"], m["target_probability"])
        assert set(target_group(record)["catalogs"]) <= {"simbad", "ned"}, path


async def test_cas_a_simbad_and_ned_centres_15_arcsec_apart_are_one_object() -> None:
    # NED lists Cas A twice ('Cas A' 15" from SIMBAD's centre, 'SRGA J232326.0+584841' at 0");
    # its 'Cas A' got 0.001 and a point IR source 2.6" away took NED's place in the target group.
    for path, record in await both_paths("named_casa"):
        assoc = record["provenance"]["association"]
        assert assoc["extended_centre_sigma_arcsec"] >= 15.1 / 2, path  # the observed spread of the centres
        rows = by_catalog_id(record)
        simbad, ned, srga = rows[("simbad", "NAME Cas A")], rows[("ned", "Cas A")], rows[("ned", "SRGA J232326.0+584841")]
        assert simbad["target_probability"] > 0.99 and ned["target_probability"] > 0.99, path
        assert srga["target_probability"] == ned["target_probability"], path
        assert {ned["coincident_with"], srga["coincident_with"]} == {None, "Cas A"} or \
            {ned["coincident_with"], srga["coincident_with"]} == {None, "SRGA J232326.0+584841"}, path
        assert rows[("ned", "WISEA J232326.22+584842.7")]["target_probability"] < 0.01, path  # was 0.378
        for m in all_rows(record):
            if m["catalog"] in ("gaia_dr3", "chandra", "xmm"):
                assert m["target_probability"] < 0.1, (path, m["source_id"])
        assert set(target_group(record)["catalogs"]) <= {"simbad", "ned", "nvss"}, path


async def test_m42_with_the_xray_catalogues_keeps_its_identity_rows() -> None:
    # The Trapezium's X-ray stars 9" from M 42's centre were the target group (5XMM 0.98, NED's
    # 2XMM entry 0.51) and NED's HII identity at 0.000" got 0.009; SIMBAD 'M 42' 0.34 via /search.
    for path, record in await both_paths("named_m42_xray"):
        rows = by_catalog_id(record)
        assert rows[("simbad", "M 42")]["target_probability"] > 0.99, path
        assert rows[("ned", "SRGA J053516.8-052315")]["target_probability"] > 0.99, path
        assert rows[("ned", "MESSIER 042")]["target_probability"] > 0.99, path
        group = target_group(record)
        assert set(group["catalogs"]) == {"simbad", "ned"}, path
        for m in all_rows(record):
            if m["catalog"] in ("chandra", "xmm", "gaia_dr3"):
                assert m["target_probability"] < 0.5, (path, m["source_id"], m["target_probability"])
            if m["catalog"] == "ned" and (m["physical"] or {}).get("object_type") in (None, "XrayS", "*", "IrS"):
                assert m["target_probability"] < 0.1, (path, m["source_id"])


async def test_perseus_cluster_is_not_its_central_galaxys_xray_nucleus() -> None:
    # NED's counterpart was '2CXO J031947.4+413052' (0.70) and its GClstr at 0.000" got 0.115.
    for path, record in await both_paths("named_perseus"):
        rows = by_catalog_id(record)
        assert rows[("simbad", "ACO 426")]["target_probability"] > 0.99, path
        assert rows[("ned", "SRGA J031947.6+413054")]["target_probability"] > 0.99, path
        for (catalog, sid), m in rows.items():
            if "CXO" in sid:
                assert m["target_probability"] < 0.05, (path, catalog, sid)


async def test_star_clusters_of_a_cluster_galaxy_are_not_the_galaxy_clusters_identity() -> None:
    # M87's globular clusters ('*Cl' in NED, 'GlC' in SIMBAD) inside the Virgo cluster's 10" cone
    # were flagged target_identity and 'secondary'.
    record = (await replay_search("named_virgo"))["record"]
    assoc = record["provenance"]["association"]
    identity = {(r["catalog"], r["source_id"]) for r in assoc["identity_rows"]}
    rows = by_catalog_id(record)
    clusters = [m for m in all_rows(record) if extended_family((m["physical"] or {}).get("object_type")) == "star_cluster"]
    assert len(clusters) > 20
    for m in clusters:
        assert (m["catalog"], m["source_id"]) not in identity and not m["target_identity"]
        assert m["target_probability"] < 0.01
    assert rows[("simbad", "NAME Virgo Cluster")]["target_probability"] > 0.99
    assert all(extended_family((m["physical"] or {}).get("object_type")) == "galaxy_group"
               for m in target_group(record)["members"] if m["coincident_with"] is None)


@pytest.mark.parametrize("cluster_error", [60.0, 5.0])
async def test_synthetic_cluster_identity_beats_a_precise_galaxy_at_its_centre(cluster_error: float) -> None:
    # The review's reproduction: a ClG row at the target plus a galaxy 0.03" away seen by
    # SIMBAD, Gaia and 2MASS. Was: identity 0.78 (resolver), 0.0004 (/search), 0.16 (5" error).
    ra0, dec0 = 194.93502, 27.91246
    galaxy = offset_radec(ra0, dec0, 0.02, 0.02)
    rows = [
        row("simbad", ra0, dec0, cluster_error, source_id="ACO 1656", data={"otype": "ClG"},
            metadata={"physical": {"object_type": "ClG"}}),
        row("simbad", *galaxy, 0.002, source_id="NGC 4876", data={"otype": "LIN"},
            metadata={"physical": {"object_type": "LIN"}}),
        row("gaia_dr3", *galaxy, 0.002, source_id="gaia", epoch=2016.0),
        row("twomass_psc", *offset_radec(ra0, dec0, 0.1, 0.09), 0.065, source_id="2mass", epoch=1999.3),
    ]
    service = static_service(rows, ["simbad", "gaia_dr3", "twomass_psc"])
    resolved = {"canonical_name": "ACO 1656", "object_type": "ClG", "resolver": "cds_sesame",
                "resolver_metadata": {"resolver_name": "Sc=Simbad (CDS, via client/server)", "extragalactic": True}}
    for extra in ({"resolved_object": resolved}, {}):
        record = await service.crossmatch(ra0, dec0, radius_arcsec=20.0, epoch=2000.0, pm_ra_masyr=0.0,
                                          pm_dec_masyr=0.0, **extra)
        p = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}
        assert record.provenance["association"]["target_class"]["class"] == "extended"
        assert p["ACO 1656"] > 0.99, (extra.keys(), p)
        assert max(p["NGC 4876"], p["gaia"], p["2mass"]) < 0.01, p


def test_compact_rows_of_an_extended_target() -> None:
    def src(catalog, data=None, otype=None):
        meta = {"physical": {"object_type": otype}} if otype else {}
        return row(catalog, 10.0, 10.0, 1.0, data=data or {}, metadata=meta)

    # Identity catalogues: anything but an extended object (a star, a galaxy, a detection).
    assert _compact_for_extended_target(src("simbad", otype="*"))
    assert _compact_for_extended_target(src("ned", otype="G"))
    assert _compact_for_extended_target(src("ned"))  # a designation without a type (WISEA, 2CXO, ...)
    assert not _compact_for_extended_target(src("simbad", otype="SNR"))
    # Radio / X-ray: compact detections vs extended emission (>= 10").
    assert _compact_for_extended_target(src("chandra", {"extent_flag": "T"}))  # CSC: compact sources only
    assert _compact_for_extended_target(src("xmm", {"extent": 0.0}))
    assert not _compact_for_extended_target(src("xmm", {"extent": 35.0}))  # a cluster's X-ray halo
    assert not _compact_for_extended_target(src("rosat", {"source_extent": 120.0}))
    assert _compact_for_extended_target(src("nvss", {"major_axis": 30.0}))  # no PA: an upper limit, unresolved
    assert not _compact_for_extended_target(src("nvss", {"major_axis": 300.0, "position_angle": 40.0}))
    # Catalogues without sizes keep their class prior.
    assert not _compact_for_extended_target(src("vlass"))
    assert not _compact_for_extended_target(src("gaia_dr3"))
    assert astrometry.emission_extent_arcsec(src("gaia_dr3")) is None


def test_extended_families() -> None:
    assert extended_family("ClG") == extended_family("!GClstr") == extended_family("GrG") == "galaxy_group"
    assert extended_family("GlC") == extended_family("*Cl") == "star_cluster"
    assert extended_family("SNR") == "supernova_remnant" and extended_family("HII") == "nebula"
    assert extended_family("PoC") == "cloud" and extended_family("*") is None


# ---------------------------------------------------------------------------
# Issue: POST /api/v1/search never passed the resolver identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "identity"), [("named_m42", "M 42"), ("named_crab", "M 1"), ("named_m13", "M 13")])
def test_post_search_endpoint_returns_the_named_extended_objects_identity(key: str, identity: str,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # The real endpoint (api.app), replaying the recorded archives and Sesame answer, with the
    # default min_confidence 0.5: 'M 42' was not returned (P = 0.19; 0.973 on the stream).
    from fastapi.testclient import TestClient

    import api
    from providers import CacheManager

    monkeypatch.setattr(api, "cache", CacheManager(None))
    spec = RECORDED_SEARCHES[key]
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
            200, text=sesame_xml(spec["sesame"]), headers={"content-type": "text/xml"})
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"matching/{key}")))
        with TestClient(api.app) as client:
            response = client.post("/api/v1/search", json={"name": spec["name"], "radius_arcsec": spec["radius_arcsec"],
                                                           "catalogs": spec["catalogs"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["failures"] == [], body["failures"]
    returned = {n: [s["source_id"] for s in r["sources"]] for n, r in body["catalog_results"].items()}
    assert identity in returned["simbad"], returned
    assert returned.get("gaia_dr3", []) == [] and returned.get("twomass_psc", []) == []
    assoc = body["provenance"]["association"]
    assert assoc["target_class"]["class"] == "extended"
    reasons = {r["source_id"]: r["reason"] for r in assoc["identity_rows"]}
    # api.py passes the resolver's position only: the row at exactly that position is the
    # identity (were api.py to pass the resolver's answer too, it would be the resolved name).
    assert reasons[identity] in ("the searched position is this row's catalogued position", "resolved name")
    # The same probability as the stream (which passes the resolver's answer).
    stream = members((asyncio.run(replay_search(key)))["record"])
    posted = {m["source_id"]: m["confidence"] for m in body["provenance"]["matches"]}
    assert posted[identity] == pytest.approx(stream[identity]["target_probability"], abs=0.02)


async def test_implicit_identity_needs_the_exact_catalogued_position() -> None:
    # A SIMBAD row within 1 mas of the target is its identity (the resolver's position was
    # searched); 0.5" away it is not.
    rows = [row("simbad", 150.0, 20.0, 0.5, source_id="exact", metadata={"physical": {"object_type": "*"}}),
            row("simbad", *offset_radec(150.0, 20.0, 0.5, 0.0), 0.5, source_id="near",
                metadata={"physical": {"object_type": "*"}})]
    service = static_service(rows, ["simbad"])
    record = await service.crossmatch(150.0, 20.0, radius_arcsec=5.0)
    assert [(r["source_id"], r["reason"]) for r in record.provenance["association"]["identity_rows"]] == [
        ("exact", "the searched position is this row's catalogued position")]
    shifted = await service.crossmatch(*offset_radec(150.0, 20.0, 0.25, 0.0), radius_arcsec=5.0)
    assert shifted.provenance["association"]["identity_rows"] == []


# ---------------------------------------------------------------------------
# Issue: several rows of one catalogue that are all the target split the posterior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "xmm"), [
    ("named_proxima", {"5XMM J142941.9-624044", "5XMM J142937.8-624040", "5XMM J142933.5-624032"}),
    ("named_61cyga", {"5XMM J210655.1+384509", "5XMM J210657.4+384530", "5XMM J210658.6+384542"}),
])
async def test_fast_movers_xmm_detections_at_several_epochs_are_one_listing(key: str, xmm: set[str]) -> None:
    # Was 0.37 / 0.36 / 0.27 (Proxima) and 0.36 / 0.35 / 0.29 (61 Cyg A): all dropped at /search's 0.5.
    for path, record in await both_paths(key):
        rows = {m["source_id"]: m for m in all_rows(record) if m["catalog"] == "xmm"}
        assert xmm <= set(rows), (path, set(rows))
        listed = [rows[s] for s in xmm]
        assert all(m["target_probability"] > 0.85 for m in listed), (path, [m["target_probability"] for m in listed])
        assert sum(m["coincident_with"] is None for m in listed) == 1, path
        assert all(m in target_group(record)["members"] for m in listed), path
    record = await search_endpoint_emulation(key)  # min_confidence 0.5
    assert xmm <= {s.source_id for s in record.catalog_results["xmm"]["sources"]}


async def test_compilation_listing_one_object_twice_does_not_split_it() -> None:
    # Cyg X-3: SIMBAD 'V* V1521 Cyg' 0.64 / 'NVSS J203225+405728' 0.27 (0.1" apart); NED
    # 'V1521 Cyg' 0.35 / its WISEA entry 0.50.
    for path, record in await both_paths("named_cygx3"):
        rows = by_catalog_id(record)
        for key in (("simbad", "V* V1521 Cyg"), ("simbad", "NVSS J203225+405728"), ("ned", "V1521 Cyg"),
                    ("ned", "WISEA J203225.78+405728.0")):
            assert rows[key]["target_probability"] > 0.99, (path, key, rows[key]["target_probability"])


async def test_field_rows_meeting_on_a_fast_track_away_from_the_target_are_not_merged() -> None:
    # Barnard-like target at J2016; two field 2MASS-like rows of different epochs whose
    # placements with the target's motion meet each other 4" from the target: two stars.
    pm = (-801.551, 10362.394)
    ra0, dec0 = 269.4486, 4.7394
    meet = offset_radec(ra0, dec0, 4.0, 0.0)
    rows = [row("gaia_dr3", ra0, dec0, 0.0002, source_id="star", epoch=2016.0, proper_motion_ra_masyr=pm[0],
                proper_motion_dec_masyr=pm[1], data={"parallax": 546.98, "parallax_error": 0.04})]
    for sid, epoch in (("a", 2010.0), ("b", 2013.0)):
        rows.append(row("sdss", *propagate_radec(*meet, pm[0], pm[1], 2016.0, epoch), 0.05, source_id=sid, epoch=epoch))
    service = static_service(rows, ["gaia_dr3", "sdss"])
    record = await service.crossmatch(ra0, dec0, radius_arcsec=10.0, epoch=2016.0, pm_ra_masyr=pm[0],
                                      pm_dec_masyr=pm[1], parallax_mas=546.98)
    rows_out = {m["source_id"]: m for m in all_rows(record.as_dict()) if m["catalog"] == "sdss"}
    assert rows_out["a"]["coincident_with"] is None and rows_out["b"]["coincident_with"] is None
    # At the target they would be one listing of it.
    rows[1:] = [row("sdss", *propagate_radec(ra0, dec0, pm[0], pm[1], 2016.0, epoch), 0.05, source_id=sid, epoch=epoch)
                for sid, epoch in (("a", 2010.0), ("b", 2013.0))]
    record = await static_service(rows, ["gaia_dr3", "sdss"]).crossmatch(
        ra0, dec0, radius_arcsec=10.0, epoch=2016.0, pm_ra_masyr=pm[0], pm_dec_masyr=pm[1], parallax_mas=546.98)
    rows_out = {m["source_id"]: m for m in all_rows(record.as_dict()) if m["catalog"] == "sdss"}
    assert {rows_out["a"]["coincident_with"], rows_out["b"]["coincident_with"]} in ({None, "a"}, {None, "b"})
    assert rows_out["a"]["target_probability"] > 0.99


# ---------------------------------------------------------------------------
# Issue: the adopted motion of a neighbouring star displaced the target's rows
# ---------------------------------------------------------------------------


def _galaxy_near_a_fast_star(pm: tuple[float, float], sep: float) -> list[CatalogSource]:
    ra0, dec0 = 150.0, 20.0
    rows = [row("twomass_psc", *offset_radec(ra0, dec0, 0.05, 0.05), 0.12, source_id="gal_2mass", epoch=1999.2),
            row("allwise", *offset_radec(ra0, dec0, -0.1, 0.05), 0.2, source_id="gal_wise", epoch=2010.5)]
    star = offset_radec(ra0, dec0, sep, 0.0)
    rows.append(row("gaia_dr3", *star, 0.02, source_id="star_gaia", epoch=2016.0, proper_motion_ra_masyr=pm[0],
                    proper_motion_dec_masyr=pm[1], data={"parallax": 20.0, "parallax_error": 0.1}))
    rows.append(row("twomass_psc", *propagate_radec(*star, pm[0], pm[1], 2016.0, 1999.2), 0.08, source_id="star_2mass",
                    epoch=1999.2))
    rows.append(row("allwise", *propagate_radec(*star, pm[0], pm[1], 2016.0, 2010.5), 0.1, source_id="star_wise",
                    epoch=2010.5))
    return rows


@pytest.mark.parametrize(("pm", "sep"), [((300.0, -200.0), 1.5), ((600.0, -400.0), 1.9)])
@pytest.mark.parametrize("epoch", [None, 2016.0])
async def test_a_neighbouring_stars_motion_is_not_adopted_for_the_target(pm, sep, epoch) -> None:
    # Was: the star's motion adopted (warning 'moves it 5.8 arcsec'); the galaxy's AllWISE row
    # fell to 0.174 / 0.013 (undated) and its 2MASS row to 0.009 at epoch 2016, where the star's
    # 2MASS row 1.5" (15 sigma) away became the counterpart at 0.935.
    service = static_service(_galaxy_near_a_fast_star(pm, sep), ["gaia_dr3", "twomass_psc", "allwise"])
    record = await service.crossmatch(150.0, 20.0, radius_arcsec=10.0, epoch=epoch)
    assert record.provenance["target_proper_motion"] is None
    assert not [w for w in record.provenance["warnings"] if w.startswith("Undated target")]
    p = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}
    assert p["gal_2mass"] > 0.99 and p["gal_wise"] > 0.99, p
    assert p.get("star_2mass", 0.0) < 0.01 and p.get("star_wise", 0.0) < 0.01, p  # (outside the cone when fast)


def test_adoption_requires_the_row_to_coincide_with_the_target() -> None:
    target = validate_target(150.0, 20.0, epoch=2016.0)
    star = row("gaia_dr3", 150.0, 20.0, 0.02, source_id="s", epoch=2016.0, proper_motion_ra_masyr=300.0,
               proper_motion_dec_masyr=-200.0, data={"parallax": 20.0, "parallax_error": 0.1})
    adopted, origin = _adopt_proper_motion(target, [("gaia_dr3", QueryResult([star], {}))], 10.0)
    assert adopted.proper_motion == (300.0, -200.0) and origin["source_id"] == "s"
    away = row("gaia_dr3", *offset_radec(150.0, 20.0, 1.5, 0.0), 0.02, source_id="s", epoch=2016.0,
               proper_motion_ra_masyr=300.0, proper_motion_dec_masyr=-200.0, data={"parallax": 20.0, "parallax_error": 0.1})
    assert _adopt_proper_motion(target, [("gaia_dr3", QueryResult([away], {}))], 10.0) is None
    # A coarse target position (1.5" typed rounding) is consistent with it.
    assert _adopt_proper_motion(target, [("gaia_dr3", QueryResult([away], {}))], 10.0, target_sigma_arcsec=1.0)
    # Undated: the J2000 / J2016 hypothesis that fits (SIMBAD J2000 coordinates of a fast star).
    undated = validate_target(*propagate_radec(150.0, 20.0, 300.0, -200.0, 2016.0, 2000.0))
    assert _adopt_proper_motion(undated, [("gaia_dr3", QueryResult([star], {}))], 10.0)[1]["source_id"] == "s"


# ---------------------------------------------------------------------------
# Issue: crowded cores missed when the small cone holds few rows
# ---------------------------------------------------------------------------


async def test_47_tuc_core_with_one_row_in_3_arcsec_is_probed() -> None:
    # 1 Gaia row in 3": no Poisson excess; the map's 5.8e4 per deg^2 was used (P = 0.937 at 3",
    # 0.684 at 30").
    small = (await replay_search("tuc47_core_3arcsec"))["record"]
    large = (await replay_search("tuc47_core_30arcsec"))["record"]
    for name in ("gaia_dr3", "twomass_psc"):
        info = small["provenance"]["association"]["densities"][name]
        assert info["probe"]["status"] == "success" and info["probe"]["reason"] == "crowded region NGC 104"
        assert info["probe"]["elapsed_ms"] is not None and info["probe"]["timeout_seconds"] <= 20.0
        other = large["provenance"]["association"]["densities"][name]["density_per_deg2"]
        assert info["density_per_deg2"] == pytest.approx(other, rel=0.05)
    gaia = "4689639266848574080"
    p_small, p_large = members(small)[gaia]["target_probability"], members(large)[gaia]["target_probability"]
    assert abs(p_small - p_large) < 0.02, (p_small, p_large)


def test_crowded_regions() -> None:
    assert astrometry.crowded_region(6.0236, -72.0813) == "NGC 104"  # 47 Tuc's core
    assert astrometry.crowded_region(250.4237, 36.4614) == "NGC 6205"  # M 13
    assert astrometry.crowded_region(81.0, -69.5) == "LMC"
    assert astrometry.crowded_region(150.0, 20.0) is None
    assert len(astrometry.CROWDED_STELLAR_SYSTEMS) == 147


async def _batch_style(key: str, **kwargs):
    """What batch.py does: the cones fetched without density probes, then finalize()."""
    spec = RECORDED_SEARCHES[key]
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"matching/{key}")))
        async with offline_client() as client:
            service = make_service(client)
            ctx = service.prepare(spec["ra"], spec["dec"], radius_arcsec=spec["radius_arcsec"],
                                  catalogs=spec["catalogs"], **kwargs)
            successes, failures = await service.executor.execute(ctx.plans, ctx.target)
    return CrossmatchService(CatalogRegistry(), {}).finalize(ctx, successes, failures)


async def test_finalize_without_a_probe_says_so_and_stays_conservative() -> None:
    # A batch finalises fetched cones without probing: P(M 13 star) was 0.671 vs 0.222 probed.
    probed = (await replay_search("m13_core_3arcsec", target_uncertainty_arcsec=0.5))["record"]
    batch = (await _batch_style("m13_core_3arcsec", target_uncertainty_arcsec=0.5)).as_dict()
    info = batch["provenance"]["association"]["densities"]["gaia_dr3"]
    assert info["probe"]["status"] == "not_run" and info["probe"]["reason"] == "poisson excess"
    assert info["density_per_deg2"] > 1.0e6  # the cone's own 6 field rows in 3" (map prior: 1.2e5)
    assert any("density probe not run" in w for w in batch["provenance"]["warnings"])
    star = "1328057905737084928"
    assert abs(members(batch)[star]["target_probability"] - members(probed)[star]["target_probability"]) < 0.05


async def test_density_probe_starts_when_its_catalogue_answers() -> None:
    # The probe of the fast crowded catalogue runs while a slow catalogue is still queried
    # (it waited for every catalogue before).
    calls: list[tuple[str, float, float]] = []
    started = time.perf_counter()

    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            calls.append((catalog.name, radius_arcsec, time.perf_counter() - started))
            if catalog.name == "twomass_psc":
                await asyncio.sleep(0.8)
                return QueryResult([], {"max_rows": 200, "row_limit": 200})
            rows = [row("gaia_dr3", *offset_radec(target.ra, target.dec, 0.4 + 0.3 * i, 0.1), 0.001, source_id=str(i))
                    for i in range(8)]
            return QueryResult(rows, {"max_rows": 200, "row_limit": 200, "query_radius_arcsec": radius_arcsec})

    registry = CatalogRegistry()
    registry._catalogs = {n: registry.get(n) for n in ("gaia_dr3", "twomass_psc")}
    provider = Provider()
    service = CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})
    events = [e async for e in service.crossmatch_stream(150.0, 20.0, radius_arcsec=3.0)]
    probe = next(t for name, radius, t in calls if name == "gaia_dr3" and radius == 30.0)
    assert probe < 0.5  # before the 0.8 s catalogue answered
    info = events[-1]["data"]["record"]["provenance"]["association"]["densities"]["gaia_dr3"]["probe"]
    assert info["status"] == "success" and info["reason"] == "poisson excess" and info["elapsed_ms"] >= 0
    # The same without the stream.
    calls.clear()
    started = time.perf_counter()
    await service.crossmatch(150.0, 20.0, radius_arcsec=3.0)
    assert next(t for name, radius, t in calls if name == "gaia_dr3" and radius == 30.0) < 0.5


# ---------------------------------------------------------------------------
# Issue: a span row's epoch was maximised over, not marginalised
# ---------------------------------------------------------------------------


def test_span_rows_of_a_fast_mover_are_marginalised_over_their_span() -> None:
    pm = (-3781.741, 769.465)
    target = validate_target(217.42894222, -62.67949019, epoch=2000.0, pm_ra_masyr=pm[0], pm_dec_masyr=pm[1])
    on_track = propagate_radec(target.ra, target.dec, pm[0], pm[1], 2000.0, 2003.0)
    span = row("xmm", *on_track, 0.88, source_id="x", epoch_range=(2001.6, 2018.2))
    det, info = astrometry.source_detection(span, target)
    assert info["propagation"] == "target_pm_span" and info["span_years"] == pytest.approx(16.6)
    # Along the track the placement spreads over the span (L / sqrt(12) = 18.5"); across it
    # keeps the row's error.
    speed = math.hypot(*pm) / 1000.0
    cov = np.array([[det.target_cov[0], det.target_cov[1]], [det.target_cov[1], det.target_cov[2]]])
    along = np.array(pm) / math.hypot(*pm)
    assert math.sqrt(along @ cov @ along) == pytest.approx(speed * 16.6 / math.sqrt(12.0), rel=0.01)
    across = np.array([-along[1], along[0]])
    assert math.sqrt(across @ cov @ across) < 1.0
    # A field source anywhere on the 64" track was a perfect match (the closest approach); its
    # Bayes factor is now that of a row within a 64"-long strip: ~20x lower than a dated row's.
    dated = row("xmm", *on_track, 0.88, source_id="d", epoch=2003.0)
    ddet, _ = astrometry.source_detection(dated, target)
    factors = []
    for d in (det, ddet):
        result = astrometry.associate([d], {"xmm": 600.0}, target=(target.ra, target.dec))
        factors.append(next(g for g in result.groups if g.contains_target).log10_bayes_factor)
    assert factors[1] - factors[0] > math.log10(10.0), factors


# ---------------------------------------------------------------------------
# Issue: supernova designations taken for binary components
# ---------------------------------------------------------------------------


def test_supernova_designations_are_not_binary_components() -> None:
    def component(name: str) -> bool:
        return is_binary_component(SimpleNamespace(object_type="SNR", canonical_name=name, query=name))

    for name in ("SN 1987A", "SN 1572A", "SN 1006A", "NAME SN 1987A", "AT 2018cow", "SN 2011fe"):
        assert not component(name), name
    for name in ("HD 239960A", "BD+56 2783A", "GJ 860 A"):
        assert component(name), name


# ---------------------------------------------------------------------------
# Issue: Sesame's VizieR fallback searched as an exact, undated position without a warning
# ---------------------------------------------------------------------------


def test_sesame_vizier_answer_is_parsed_with_its_notes() -> None:
    obj = SesameResolver.parse_response("GJ 860 A", sesame_xml("gj860a_vizier"))
    assert obj.canonical_name == "Gl 860"  # was '{Name} Gl 860'
    meta = obj.resolver_metadata
    assert SesameResolver.answer_kind(meta) == "vizier" and obj.epoch is None
    assert SesameResolver.multiple_answers(meta) == 2  # Sesame's '++++Multiple (2) answers++++' INFO
    simbad = SesameResolver.parse_response("GJ 860 A", sesame_xml("gj860a")).resolver_metadata
    assert SesameResolver.answer_kind(simbad) == "simbad" and SesameResolver.multiple_answers(simbad) is None


async def test_sesame_vizier_fallback_is_retried_against_simbad() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV").respond(
            200, text=sesame_xml("gj860a_vizier"), headers={"content-type": "text/xml"})
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/S?").respond(
            200, text=sesame_xml("gj860a"), headers={"content-type": "text/xml"})
        async with offline_client() as client:
            obj = await SesameResolver(client).resolve("GJ 860 A")
    assert obj.canonical_name == "HD 239960A" and obj.epoch == 2000.0
    assert obj.resolver_metadata["simbad_retry"]["replaced"] == "Vl=VizieR (local)"


async def test_sesame_vizier_fallback_kept_with_warnings_when_simbad_stays_silent() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV").respond(
            200, text=sesame_xml("gj860a_vizier"), headers={"content-type": "text/xml"})
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/S?").respond(503, text="busy")
        async with offline_client() as client:
            obj = await SesameResolver(client).resolve("GJ 860 A")
    assert SesameResolver.answer_kind(obj.resolver_metadata) == "vizier"
    assert obj.resolver_metadata["simbad_retry"]["error"] == "HTTP 503"
    spec = resolved_search_target(obj)
    assert spec["target_uncertainty_arcsec"] == RESOLVER_UNDATED_SIGMA_ARCSEC
    assert spec["target_uncertainty_source"] == "resolver_fallback"
    assert any("not SIMBAD" in w and "undated" in w for w in spec["warnings"])
    assert any("2 objects" in w for w in spec["warnings"])
    # A requested epoch relabels the undated position only with a warning.
    dated = resolved_search_target(obj, epoch=2016.0)
    assert dated["epoch"] == 2016.0 and any("has no epoch" in w for w in dated["warnings"])
    # The warnings reach the record's provenance.
    service = static_service([], ["gaia_dr3"])

    class Resolver:
        async def resolve(self, _name):
            return obj

    events = [e async for e in service.crossmatch_stream(name="GJ 860 A", resolver=Resolver(), radius_arcsec=5.0)]
    warnings = events[-1]["data"]["record"]["provenance"]["warnings"]
    assert any("not SIMBAD" in w for w in warnings) and any("2 objects" in w for w in warnings)


def test_simbad_answers_carry_no_resolver_warning() -> None:
    obj = SesameResolver.parse_response("GJ 860 A", sesame_xml("gj860a"))
    assert resolved_search_target(obj)["warnings"] == []


# ---------------------------------------------------------------------------
# Issue: the done payload serialised on the event loop
# ---------------------------------------------------------------------------


class HeavyRows:
    """Six catalogues x 200 rows with 40 data columns each (a multi-megabyte done frame)."""

    async def query(self, catalog, target, radius_arcsec):
        rows = []
        for i in range(200):
            x, y = (i % 20 - 10) * 1.3, (i // 20 - 5) * 1.3
            data = {f"col_{k}": f"value {k} of row {i} in {catalog.name} " * 2 for k in range(40)}
            rows.append(CatalogSource(catalog.name, f"{catalog.name}-{i}", *offset_radec(target.ra, target.dec, x, y), 0.1,
                                      data, {"wavelength": catalog.wavelength}, {}))
        return QueryResult(rows, {"max_rows": 400, "row_limit": 400})


def test_sse_done_payload_is_serialised_off_the_event_loop() -> None:
    from fastapi import FastAPI

    import streaming

    names = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2", "simbad", "sdss"]
    registry = CatalogRegistry()
    registry._catalogs = {n: registry.get(n) for n in names}
    provider = HeavyRows()
    app = FastAPI()
    app.include_router(streaming.router)
    app.state.service = CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    async def run() -> tuple[float, int]:
        latencies: list[float] = []
        done = asyncio.Event()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=None) as http:
            async def pinger() -> None:
                while not done.is_set():
                    t = time.perf_counter()
                    await http.get("/ping")
                    latencies.append(time.perf_counter() - t)
                    await asyncio.sleep(0.02)

            task = asyncio.create_task(pinger())
            await asyncio.sleep(0.1)
            response = await http.get("/api/v1/search/stream", params={"ra": "150.1234567", "dec": "20.1234567",
                                                                       "radius_arcsec": 20})
            done.set()
            await task
        frames = [f for f in response.text.split("\n\n") if f.startswith("event: done")]
        return max(latencies), len(frames[0]) if frames else 0

    worst, size = asyncio.run(run())
    assert size > 3_000_000  # a realistic multi-megabyte done frame
    assert worst < 0.5, worst  # was 1.85 s at the done chunk


def test_pre_serialised_frames_are_identical_to_the_loops() -> None:
    import streaming

    data = {"a": [1, 2.5, None], "text": "line sep", "nan": float("nan")}
    assert streaming.sse_frame("group", data, 3, payload=streaming.sse_payload(data)) == \
        streaming.sse_frame("group", data, 3)
    with pytest.raises(ValueError):
        streaming.sse_frame("group", data, 3, payload="two\nlines")


# ---------------------------------------------------------------------------
# Issue: CLI exit status after an error event
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    import streaming

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    streaming.register_cli(sub)
    astrometry.register_cli(sub)
    return parser.parse_args(argv)


@pytest.mark.parametrize("fmt", ["sse", "jsonl"])
def test_cli_stream_fails_when_the_stream_ends_with_an_error(fmt: str, capsys: pytest.CaptureFixture[str],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    service = static_service([row("gaia_dr3", 150.0, 20.0, 0.01, source_id="s")], ["gaia_dr3"])

    def broken(*_a, **_kw):
        raise RuntimeError("association failed")

    monkeypatch.setattr(service, "finalize", broken)
    args = _parse(["stream", "--ra", "150", "--dec", "20", "--format", fmt])
    args.service = service
    assert args.handler(args) == 1  # was 0 (sse) and a traceback (jsonl)
    out = capsys.readouterr().out
    assert "association failed" in out and "Traceback" not in out
    if fmt == "jsonl":
        error = json.loads(out.strip().splitlines()[-1])
        assert error["event"] == "error" and error["data"]["status"] == 500


@pytest.mark.parametrize("command", ["stream", "xmatch-calibrate"])
def test_cli_subcommands_have_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse([command, "--help"])
    assert exit_info.value.code == 0 and "usage" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Docstrings of the cone-edge treatment
# ---------------------------------------------------------------------------


def test_cone_edge_docstring_describes_observed_fractions() -> None:
    doc = astrometry.associate.__doc__ or ""
    assert "_observed_fractions" in doc and "independent field row" not in doc
    assert EXTENDED_CENTRE_SIGMA_ARCSEC == 1.0
    assert fixture_io.FIXTURES.exists()
