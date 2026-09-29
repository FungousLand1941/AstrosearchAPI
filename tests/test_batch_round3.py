"""Offline regression tests for the round-3 review of batch.py.

Covers: generic ``vizier:<table>`` answers whose own ``meta.main`` position is at J2000 (UCAC4, URAT1, USNO-B1.0,
NOMAD, Gaia DR2/EDR3) -- epochs, identifiers (bare ``meta.id``, XMatch's truncated fixed-size strings) and
``radec_err`` positional errors -- and dated targets planned at the J2000 positions XMatch matches (Hipparcos,
Tycho-2), from live recordings; Gaia-style target lists (``source_id``, negative parallaxes); booleans and
unusable coordinate precisions; fair, time-sliced CPU work (a small request beside a large batch, cancellation);
the batch routes' own body limit under an application-wide one; bounded memory for oversized target lists;
the CLI on a cp1252 console; the XMatch layout off the event loop; JSON bodies under other content types.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import tracemalloc
from typing import Any

import httpx
import pytest
import respx
from test_batch import make_app, replay_batch
from test_batch_fixtures import ASTROMETRIC_STARS, DATED_2016_TARGETS, J2000_TABLE_TARGETS, USNOB_FIELD_TARGETS
from test_batch_regressions import run_with, simbad_json, simbad_rows_for, uploaded_targets, xmatch_votable
from test_batch_round2 import _with_ticker

import batch
from batch import BatchCrossmatcher, BatchError, BatchResult, parse_targets
from models import CatalogRegistry, ColumnMeta, propagate_radec


def _replay(case: str, targets, catalogs, radius, **kwargs) -> BatchResult:
    return asyncio.run(replay_batch(case, targets, catalogs, radius, **kwargs))


def _cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    batch.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _no_upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
    raise AssertionError(f"no upstream request expected: {request.url}")


def _simbad_two_rows(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=simbad_json(simbad_rows_for(request, per_target=2)))


# ---------------------------------------------------------------------------
# 1. Generic tables whose meta.main position is at J2000 (live recordings, 2026-09-29)
# ---------------------------------------------------------------------------

UCAC4_IDS = {"Barnard": "474-068224", "Kapteyn": "225-005836", "Groombridge 1830": "639-046031",
             "61 Cyg A": "644-101660"}


@pytest.fixture(scope="module")
def j2000_tables() -> BatchResult:
    return _replay("j2000_tables", J2000_TABLE_TARGETS,
                   ["vizier:I/322A/out", "vizier:I/329/urat1", "vizier:I/284/out", "vizier:I/297/out",
                    "vizier:I/345/gaia2", "vizier:I/350/gaiaedr3"], 5.0)


def test_ucac4_rows_are_dated_at_their_j2000_positions(j2000_tables):
    """UCAC4 RAJ2000 is at J2000.0 (ReadMe note 2); EpRA (1991.25 for Barnard's star) is the central epoch of the
    mean RA. Re-dated to EpRA, the true counterparts of the undated J2000 targets got posteriors 0.0067 (Barnard,
    0.094" away), 0.0088 (Kapteyn), 0.011 (Groombridge 1830), 0.010 (61 Cyg A)."""
    catalog = "vizier:I/322A/out"
    run = j2000_tables.runs[catalog]
    assert not run.errors and run.failed_targets == 0
    for name, ucac in UCAC4_IDS.items():
        matches = j2000_tables.target_matches(name)[catalog]
        match = matches[0]
        assert match["source_id"] == ucac, (name, [m["source_id"] for m in matches])
        assert match["epoch"] == 2000.0, (name, match["epoch"])
        assert match["separation_arcsec"] < 0.3, (name, match["separation_arcsec"])
        assert match["data"]["EpRA"] < 2000.0  # the mean epoch is kept as data, not used as the position epoch
        if name == "61 Cyg A":
            # UCAC4 also lists 644-101659 2.47" away with 61 Cyg A's motion: the two share the posterior, the
            # nearer row most probable (0.010 before; the reviewer's J2000-dated run gave 0.59).
            other = next(m for m in matches if m["source_id"] == "644-101659")
            assert match["confidence"] > 0.5 > other["confidence"], (match["confidence"], other["confidence"])
            assert match["confidence"] + other["confidence"] > 0.99
        else:
            assert match["confidence"] > 0.99, (name, match["confidence"])
    for name in ASTROMETRIC_STARS:
        for match in j2000_tables.target_matches(name).get(catalog, []):
            if match["pm_ra_masyr"] is not None:
                assert match["epoch"] == 2000.0, (name, match)


def test_ucac4_and_urat1_positional_errors_come_from_radec_err(j2000_tables):
    """XMatch gives UCAC4 and URAT1 no error ellipse but a circular 'radec_err' (stat.error;pos.eq, arcsec):
    every row had positional_error_arcsec None."""
    for catalog in ("vizier:I/322A/out", "vizier:I/329/urat1"):
        rows = [m for t in j2000_tables.targets for m in j2000_tables.target_matches(t.id).get(catalog, [])]
        assert rows, catalog
        for match in rows:
            assert match["positional_error_arcsec"] == pytest.approx(match["data"]["radec_err"]), (catalog, match)
    barnard = next(m for m in j2000_tables.target_matches("Barnard")["vizier:I/322A/out"]
                   if m["source_id"] == UCAC4_IDS["Barnard"])
    assert barnard["positional_error_arcsec"] == pytest.approx(0.025)


def test_urat1_matches_on_ra2000_not_on_its_observation_epoch_position(j2000_tables):
    """URAT1: XMatch matches RA2000/DE2000 (meta.main, J2000); RAdeg/DEdeg are at 'Epoch' (~2013)."""
    catalog = "vizier:I/329/urat1"
    moving = [m for t in j2000_tables.targets for m in j2000_tables.target_matches(t.id).get(catalog, [])
              if m["pm_ra_masyr"] is not None]
    assert moving
    for match in moving:
        assert match["epoch"] == 2000.0
        assert match["ra"] == pytest.approx(match["data"]["RA2000"]) and match["dec"] == pytest.approx(match["data"]["DE2000"])
        assert match["data"]["Epoch"] > 2010.0
    hd = next(m for m in j2000_tables.target_matches("HD 80606")[catalog] if m["pm_ra_masyr"] is not None)
    assert hd["separation_arcsec"] < 0.3 and hd["confidence"] > 0.9, hd


@pytest.mark.parametrize("catalog, cyg_a", [("vizier:I/345/gaia2", "1872046574983507456"),
                                             ("vizier:I/350/gaiaedr3", "1872046609345556480")])
def test_gaia_dr2_edr3_keep_their_source_id_and_j2000_epoch(j2000_tables, catalog, cyg_a):
    """Gaia DR2/EDR3 in XMatch: 'source_id' is a bare meta.id and the matched position ra_epoch2000 is J2000.
    Barnard's star was 'J269.452082+04.693364' with epoch None."""
    barnard = j2000_tables.target_matches("Barnard")[catalog][0]
    assert barnard["source_id"] == "4472832130942575872", barnard["source_id"]
    assert barnard["epoch"] == 2000.0 and barnard["separation_arcsec"] < 0.3 and barnard["confidence"] > 0.9
    ids = [m["source_id"] for t in j2000_tables.targets for m in j2000_tables.target_matches(t.id).get(catalog, [])]
    assert ids and all(i.isdigit() for i in ids), ids
    cyg = j2000_tables.target_matches("61 Cyg A")[catalog][0]
    assert cyg["source_id"] == cyg_a and cyg["epoch"] == 2000.0 and cyg["confidence"] > 0.9
    # 2-parameter solutions (no proper motion) are at the Gaia reference epoch, which the answer does not state.
    assert all(m["epoch"] is None for t in j2000_tables.targets for m in j2000_tables.target_matches(t.id).get(catalog, [])
               if m["pm_ra_masyr"] is None)


@pytest.mark.parametrize("catalog, barnard_id", [("vizier:I/284/out", "0946-00315199"),
                                                 ("vizier:I/297/out", "0946-00320554")])
def test_usno_b1_and_nomad_ids_are_complete(j2000_tables, catalog, barnard_id):
    """XMatch declares USNO-B1.0 / NOMAD1.0 as char arraysize 12 but writes 13-character values (Barnard's star
    was '0946-0031519' in USNO-B1.0)."""
    ids = [m["source_id"] for t in j2000_tables.targets for m in j2000_tables.target_matches(t.id).get(catalog, [])]
    assert ids and all(len(i) == 13 for i in ids), ids
    barnard = next(m for m in j2000_tables.target_matches("Barnard")[catalog] if m["separation_arcsec"] < 1.0)
    assert barnard["source_id"] == barnard_id and barnard["epoch"] == 2000.0 and barnard["confidence"] > 0.9


def test_usno_b1_ids_are_unique_in_a_field():
    """3C 273 at 20": 4 different USNO-B1.0 sources were all '0920-0025857' (a real star 150 deg away) and 3
    more '0920-0025856'. VizieR's own id of 3C 273 is 0920-0258572 (XMatch writes the number on 8 digits)."""
    result = _replay("usnob_field", USNOB_FIELD_TARGETS, ["vizier:I/284/out", "vizier:I/297/out"], 20.0)
    for catalog in ("vizier:I/284/out", "vizier:I/297/out"):
        matches = result.target_matches("3C 273")[catalog]
        ids = [m["source_id"] for m in matches]
        assert len(ids) >= 3 and len(set(ids)) == len(ids), (catalog, ids)
        assert all(len(i) == 13 for i in ids)
        positions = {m["source_id"]: (m["ra"], m["dec"]) for m in matches}
        assert len(set(positions.values())) == len(positions)
    assert result.target_matches("3C 273")["vizier:I/284/out"][0]["source_id"] == "0920-00258572"


def _usno_like_answer() -> bytes:
    return (b'<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results"><TABLE>'
            b'<FIELD name="t_idx" datatype="short"/>'
            b'<FIELD name="USNO-B1.0" datatype="char" arraysize="12" ucd="meta.id;meta.main"/>'
            b'<FIELD name="Flags" datatype="char" arraysize="3"/>'
            b'<FIELD name="RAJ2000" datatype="double" ucd="pos.eq.ra;meta.main"/>'
            b"<DATA><TABLEDATA><TR><TD>0</TD><TD>0920-00258572</TD><TD>...</TD><TD>1.5</TD></TR>"
            b"</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>")


def test_fixed_size_strings_are_read_whole():
    content = _usno_like_answer()
    relaxed = batch._relax_char_arraysize(content)
    assert b'name="USNO-B1.0" datatype="char" arraysize="*"' in relaxed and b'arraysize="12"' not in relaxed
    assert relaxed.endswith(content[content.index(b"<TABLEDATA"):])
    response = httpx.Response(200, headers={"content-type": "text/xml"}, content=content,
                              request=httpx.Request("POST", "http://example.org"))
    table = BatchCrossmatcher._parse(response, "test", "votable")
    assert table.rows[0]["USNO-B1.0"] == "0920-00258572" and table.rows[0]["Flags"] == "..."
    # BINARY serialisations need the declared sizes: left alone (the very same object).
    binary = content.replace(b"<TABLEDATA>", b"<BINARY>")
    assert batch._relax_char_arraysize(binary) is binary


def _j2000_generic_columns(ra: str, dec: str, epoch_ucd: str = "time.epoch") -> list[ColumnMeta]:
    return [ColumnMeta("ID", None, "meta.id;meta.main"), ColumnMeta(ra, "deg", "pos.eq.ra;meta.main"),
            ColumnMeta(dec, "deg", "pos.eq.dec;meta.main"), ColumnMeta("pmRA", "mas/yr", "pos.pm;pos.eq.ra"),
            ColumnMeta("pmDE", "mas/yr", "pos.pm;pos.eq.dec"), ColumnMeta("Epoch", "yr", epoch_ucd)]


@pytest.mark.parametrize("ra, dec, epoch_ucd, expected", [
    ("RAJ2000", "DEJ2000", "time.epoch", (2000.0, 1991.25)),       # UCAC4, USNO-B1.0, NOMAD, PPMXL
    ("RA2000", "DE2000", "time.epoch", (2000.0, 1991.25)),         # URAT1
    ("ra_epoch2000", "dec_epoch2000", "time.epoch", (2000.0, 1991.25)),  # Gaia DR2/EDR3
    ("RAdeg", "DEdeg", "time.epoch", (None, 1991.25)),             # position epoch not stated: unknown
    ("RAdeg", "DEdeg", "meta.ref;time.epoch", (1991.25, 1991.25)),  # a reference epoch is the position's
])
def test_generic_epoch_rules(ra, dec, epoch_ucd, expected):
    catalog, _ = batch.xmatch_catalog_definition("vizier:I/999/test", CatalogRegistry())
    moving = {"ID": "a", ra: 10.0, dec: 5.0, "pmRA": 100.0, "pmDE": -50.0, "Epoch": 1991.25}
    still = {**moving, "ID": "b", "pmRA": None, "pmDE": None}
    definition, columns = batch.describe_generic_answer(catalog, _j2000_generic_columns(ra, dec, epoch_ucd),
                                                        {0: [moving, still]})
    result = batch._RowConverter(None, "cds_xmatch")._sources(definition, [moving, still], 5.0, None, {},
                                                              columns=columns)
    got = {s.source_id: s.epoch for s in result}
    assert (got["a"], got["b"]) == expected, got


def test_generic_bare_meta_id_is_used_only_when_it_identifies_sources():
    catalog, _ = batch.xmatch_catalog_definition("vizier:I/999/test", CatalogRegistry())
    columns = [ColumnMeta("source_id", None, "meta.id"), ColumnMeta("ra_epoch2000", "deg", "pos.eq.ra;meta.main"),
               ColumnMeta("dec_epoch2000", "deg", "pos.eq.dec;meta.main"), ColumnMeta("Num", None, "meta.id")]
    a = {"source_id": 11, "ra_epoch2000": 10.0, "dec_epoch2000": 5.0, "Num": 7}
    b = {"source_id": 12, "ra_epoch2000": 10.001, "dec_epoch2000": 5.0, "Num": 7}
    again = dict(a)  # the same source matched to a second target
    definition, _ = batch.describe_generic_answer(catalog, list(columns), {0: [a, b], 1: [again]})
    assert definition.parameters["id_field"] == "source_id"
    # A count-like meta.id ('Num'), or one value at two positions, is never an identifier.
    counts = [columns[3], columns[1], columns[2]]
    rows = [{"Num": 7, "ra_epoch2000": 10.0, "dec_epoch2000": 5.0},
            {"Num": 7, "ra_epoch2000": 10.001, "dec_epoch2000": 5.0}]
    definition, _ = batch.describe_generic_answer(catalog, list(counts), {0: rows})
    assert definition.parameters["id_field"] == batch.GENERIC_ID_COLUMN
    clash = [ColumnMeta("objID", None, "meta.id"), columns[1], columns[2]]
    rows = [{"objID": 1, "ra_epoch2000": 10.0, "dec_epoch2000": 5.0},
            {"objID": 1, "ra_epoch2000": 10.001, "dec_epoch2000": 5.0}]
    definition, _ = batch.describe_generic_answer(catalog, list(clash), {0: rows})
    assert definition.parameters["id_field"] == batch.GENERIC_ID_COLUMN


def test_generic_positional_error_fallbacks():
    catalog, _ = batch.xmatch_catalog_definition("vizier:I/999/test", CatalogRegistry())
    circular = [ColumnMeta("radec_err", "arcsec", "stat.error;pos.eq"), ColumnMeta("sigm", "mas", "stat.error;pos.eq")]
    assert batch._with_ellipse_errors(catalog, circular).pos_error == {"columns": ["radec_err"], "kind": "sigma"}
    pair = [ColumnMeta("e_RA", "mas", "stat.error;pos.eq.ra"), ColumnMeta("e_DE", "mas", "stat.error;pos.eq.dec"),
            ColumnMeta("e_pmRA", "mas/yr", "stat.error;pos.pm;pos.eq.ra")]
    assert batch._with_ellipse_errors(catalog, pair).pos_error == {"columns": ["e_RA", "e_DE"], "kind": "sigma"}
    ellipse = [ColumnMeta("errHalfMaj", "arcsec", "phys.angSize.smajAxis;pos.errorEllipse;meta.main"), *circular]
    assert batch._with_ellipse_errors(catalog, ellipse).pos_error["kind"] == "ellipse"
    assert batch._with_ellipse_errors(catalog, [ColumnMeta("x", None, "phot.mag")]).pos_error == {}


# ---------------------------------------------------------------------------
# 2. Dated targets are looked for at the J2000 positions XMatch matches (live recordings)
# ---------------------------------------------------------------------------

HIP_IDS = {"Barnard": "87937", "61 Cyg A": "104214", "Kapteyn": "24186", "Groombridge 1830": "57939",
           "Lacaille 9352": "114046", "tau Cet": "8102", "51 Peg": "113357", "HD 189733": "98505"}


@pytest.fixture(scope="module")
def dated_2016() -> BatchResult:
    return _replay("dated_2016", DATED_2016_TARGETS, ["vizier:I/311/hip2", "vizier:I/239/hip_main",
                                                      "vizier:I/259/tyc2", "vizier:I/345/gaia2"], 3.0)


@pytest.mark.parametrize("catalog", ["vizier:I/311/hip2", "vizier:I/239/hip_main"])
def test_dated_targets_find_their_hipparcos_rows(dated_2016, catalog):
    """Targets at epoch 2016.0 with their SIMBAD proper motions: HD 189733 and 51 Peg had no hip2 row at 3" and
    the six fast movers none in hip_main / hip2 (the cone stayed at the 2016 position, and the rows were left
    without a match -- read as 'no counterpart').

    The separations are those of the Hipparcos J2000 positions moved to 2016 with the Hipparcos proper motions
    (Barnard: 0.46", its Hipparcos pm differs from Gaia's by 34 mas/yr). XMatch's Hipparcos columns carry no
    positional or proper-motion errors; the association's error model for such a 16-year propagation keeps
    every posterior near 0.79 (0.997 for the undated J2000 targets of the 'hipparcos' case)."""
    run = dated_2016.runs[catalog]
    assert not run.errors and run.failed_targets == 0
    for name, hip in HIP_IDS.items():
        matches = dated_2016.target_matches(name).get(catalog, [])
        assert [m["source_id"] for m in matches] == [hip], (name, catalog, matches)
        assert matches[0]["separation_arcsec"] < 0.6 and matches[0]["confidence"] > 0.75, (name, matches[0])
        assert matches[0]["epoch"] == 2000.0 and matches[0]["epoch_propagation"] == "source_pm"


def test_dated_targets_find_their_tycho2_and_gaia_dr2_rows(dated_2016):
    tycho = {name: [m["source_id"] for m in dated_2016.target_matches(name).get("vizier:I/259/tyc2", [])]
             for name in ("HD 189733", "51 Peg")}
    assert tycho == {"HD 189733": ["2141-972-1"], "51 Peg": ["1717-2193-1"]}
    for name in tycho:
        match = dated_2016.target_matches(name)["vizier:I/259/tyc2"][0]
        assert match["separation_arcsec"] < 0.1 and match["confidence"] > 0.9, (name, match)
    for name in HIP_IDS:  # Gaia DR2 ra_epoch2000: every star, within 0.02"
        match = dated_2016.target_matches(name)["vizier:I/345/gaia2"][0]
        assert match["epoch"] == 2000.0 and match["separation_arcsec"] < 0.02 and match["confidence"] > 0.99, (name, match)


def test_generic_cones_are_planned_at_j2000():
    target = parse_targets([{"id": "HD 189733@2016", "ra": 300.18, "dec": 22.71, "epoch": 2016.0,
                             "pmra": -3.208, "pmdec": -250.323}])[0]
    catalog, view = batch.xmatch_catalog_definition("vizier:I/311/hip2", CatalogRegistry())
    planning = batch._planning_catalog(catalog, "xmatch", view)
    assert planning.epoch == 2000.0 and catalog.epoch is None
    from models import plan_cone
    plan = plan_cone(planning, target.target, 3.0)
    expected = propagate_radec(300.18, 22.71, -3.208, -250.323, 2016.0, 2000.0)
    assert plan.mode == "target_pm" and (plan.ra, plan.dec) == pytest.approx(expected, abs=1e-9)
    # Views keep their own epochs.
    gaia, gaia_view = batch.xmatch_catalog_definition("vizier:I/355/gaiadr3", CatalogRegistry())
    assert batch._planning_catalog(gaia, "xmatch", gaia_view) is gaia


def test_dated_target_uploads_its_j2000_position():
    """The XMatch upload carries the target at J2000 (not at its 2016 position)."""
    ra, dec, pmra, pmdec = ASTROMETRIC_STARS["Barnard"]
    ra16, dec16 = propagate_radec(ra, dec, pmra, pmdec, 2000.0, 2016.0)
    sent: list[tuple[float, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.extend((r, d) for _, r, d in uploaded_targets(request))
        return httpx.Response(200, content=xmatch_votable([]), headers={"content-type": "text/xml"})

    asyncio.run(run_with(handler, [{"id": "b", "ra": ra16, "dec": dec16, "epoch": 2016.0, "pmra": pmra,
                                    "pmdec": pmdec}], ["vizier:I/311/hip2"], 3.0))
    # Linear propagation to 2016 and back agrees to < 1 mas (the cone is 3").
    assert sent == [pytest.approx((ra, dec), abs=1e-6)]


# ---------------------------------------------------------------------------
# 3. Gaia-style target lists
# ---------------------------------------------------------------------------

GAIA_CSV = ("source_id,ra,dec,pmra,pmdec,parallax,ref_epoch\n"
            "4472832130942575872,269.44850252543836,4.739420051112412,-801.551,10362.394,546.9759,2016.0\n"
            "3700386905605055360,187.2779163,2.0523888,-0.21,0.09,-0.21,2016.0\n"
            "12,10.0,10.0,1.0,1.0,0.0,2016.0\n")


def test_gaia_target_list_with_negative_parallax():
    targets = batch.read_targets_csv(GAIA_CSV)
    assert [t.id for t in targets] == ["4472832130942575872", "3700386905605055360", "12"]
    barnard, quasar, zero = targets
    assert barnard.target.parallax_mas == pytest.approx(546.9759) and not barnard.notes
    assert quasar.target.parallax_mas is None and quasar.target.epoch == 2016.0
    assert "-0.21" in quasar.notes[0] and "not used" in quasar.notes[0]
    assert zero.target.parallax_mas == 0.0 and not zero.notes
    assert quasar.as_dict()["notes"] == list(quasar.notes)
    with pytest.raises(BatchError, match=r"parallax_mas must be in \(-1000, 1000\)"):
        parse_targets([{"ra": 1, "dec": 1, "parallax": -1000.0}])


def test_gaia_target_list_through_the_router():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=handler)
        with TestClient(make_app()) as client:
            response = client.post("/api/v1/batch/crossmatch?catalogs=simbad", content=GAIA_CSV,
                                   headers={"content-type": "text/csv"})
    assert response.status_code == 200, response.text
    by_id = {t["id"]: t for t in response.json()["targets"]}
    assert set(by_id) == {"4472832130942575872", "3700386905605055360", "12"}
    assert by_id["3700386905605055360"]["parallax_mas"] is None and by_id["3700386905605055360"]["notes"]


@pytest.mark.parametrize("item", [
    {"id": "a", "ra": True, "dec": 0.0}, {"id": "a", "ra": 1.0, "dec": False},
    {"id": "a", "ra": 1.0, "dec": 1.0, "epoch": True}, {"id": "a", "ra": 1.0, "dec": 1.0, "radius": True},
    {"id": "a", "ra": 1.0, "dec": 1.0, "pmra": True, "pmdec": 1.0}, {"id": "a", "ra": 1.0, "dec": 1.0, "parallax": True},
])
def test_booleans_are_not_numbers(item):
    with pytest.raises(BatchError, match="numeric"):
        parse_targets([item])


@pytest.mark.parametrize("ra, dec, message", [
    ("0e400", "1", "no usable precision"), ("1e300", "0", "too coarse"), ("1", "0e400", "no usable precision"),
    ("1e5", "0", "too coarse"),
])
def test_unusable_coordinate_precision_is_a_batch_error(ra, dec, message):
    with pytest.raises(BatchError, match=message):
        parse_targets([{"id": "x", "ra": ra, "dec": dec}])
    assert parse_targets([{"ra": "187", "dec": "2"}])[0].coordinate_sigma_arcsec < 1100.0  # coarse but usable


def test_cli_unusable_precision_is_exit_2(tmp_path, capsys):
    path = tmp_path / "over.csv"
    path.write_text("id,ra,dec\na,0e400,1\n", encoding="utf-8")
    args = _cli(["batch", "--targets", str(path), "--catalogs", "simbad"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_no_upstream)
        assert args.handler(args) == 2
    assert capsys.readouterr().out.startswith("Error: target 'a': coordinates as typed")


def test_cli_overflow_is_exit_2(tmp_path, capsys, monkeypatch):
    path = tmp_path / "t.csv"
    path.write_text("id,ra,dec\na,1,1\n", encoding="utf-8")

    def overflow(*args, **kwargs):
        raise OverflowError("math range error")

    monkeypatch.setattr(batch, "load_targets_file", overflow)
    args = _cli(["batch", "--targets", str(path), "--catalogs", "simbad"])
    assert args.handler(args) == 2
    assert capsys.readouterr().out.startswith("Error: math range error")


def test_cli_prints_non_cp1252_ids_on_a_cp1252_console(tmp_path, monkeypatch):
    """Redirected stdout on Windows is cp1252: 'α Cen' in the error message crashed the CLI (exit 1, no output)."""
    path = tmp_path / "alpha.csv"
    path.write_text("id,ra,dec\nα Cen,abc,1\n", encoding="utf-8")
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)
    args = _cli(["batch", "--targets", str(path), "--catalogs", "simbad"])
    code = args.handler(args)
    console.flush()
    assert code == 2
    assert raw.getvalue().decode("cp1252").strip() == "Error: target '\\u03b1 Cen': ra must be numeric."


# ---------------------------------------------------------------------------
# 4. Fair, time-sliced CPU work
# ---------------------------------------------------------------------------


def _big_targets(n: int) -> list[dict[str, Any]]:
    return [{"id": str(i), "ra": (i * 0.113) % 360, "dec": -60 + (i * 0.037) % 120} for i in range(n)]


def test_small_request_is_not_stuck_behind_a_large_association(monkeypatch):
    """A 1-target batch started while a 3000-target batch is associating (2 SIMBAD rows per target, ~5 s of
    CPU): it waited for the whole association (one worker call); now it waits for at most one time slice per
    step and finishes in a fraction of the large batch's association time."""
    n = 3000
    calls: list[float] = []
    original = BatchCrossmatcher._associate

    def spy(self, batch_, *args, **kwargs):
        if len(batch_) == n:
            calls.append(time.perf_counter())
        return original(self, batch_, *args, **kwargs)

    monkeypatch.setattr(BatchCrossmatcher, "_associate", spy)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_simbad_two_rows)) as client:
            engine = BatchCrossmatcher(client=client, chunk_sizes={"simbad": n})
            big = asyncio.create_task(engine.run(_big_targets(n), ["simbad"], radius_arcsec=5.0))
            while not calls:
                await asyncio.sleep(0.005)
            started = time.perf_counter()
            small = await engine.run([{"id": "s", "ra": 10.0, "dec": 10.0}], ["simbad"], radius_arcsec=5.0)
            small_elapsed = time.perf_counter() - started
            done_during_small = len(calls)
            large = await big
            association = calls[-1] - calls[0]
            return small, small_elapsed, done_during_small, large, association

    small, small_elapsed, done_during_small, large, association = asyncio.run(scenario())
    assert len(small.target_matches("s")["simbad"]) == 2
    assert large.runs["simbad"].matched_targets == n and len(large.association) == n
    assert done_during_small < n, "the large association finished before the small request"
    assert small_elapsed < 0.35 * association, (small_elapsed, association)


def test_cancelled_batch_frees_the_cpu_slot_within_one_slice(monkeypatch):
    """A batch cancelled during its association (client disconnect) stops after the running slice; a request
    submitted after the cancel is served at once (it waited 35.8 s for the abandoned association before)."""
    n = 3000
    calls: list[float] = []
    original = BatchCrossmatcher._associate

    def spy(self, batch_, *args, **kwargs):
        if len(batch_) == n:
            calls.append(time.perf_counter())
        return original(self, batch_, *args, **kwargs)

    monkeypatch.setattr(BatchCrossmatcher, "_associate", spy)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_simbad_two_rows)) as client:
            engine = BatchCrossmatcher(client=client, chunk_sizes={"simbad": n})
            big = asyncio.create_task(engine.run(_big_targets(n), ["simbad"], radius_arcsec=5.0))
            while len(calls) < 50:
                await asyncio.sleep(0.005)
            big.cancel()
            with pytest.raises(asyncio.CancelledError):
                await big
            cancelled_at = time.perf_counter()
            small = await engine.run([{"id": "s", "ra": 10.0, "dec": 10.0}], ["simbad"], radius_arcsec=5.0)
            small_elapsed = time.perf_counter() - cancelled_at
            await asyncio.sleep(3 * batch.CPU_SLICE_SECONDS)
            after_cancel = [t for t in calls if t > cancelled_at]
            return small, small_elapsed, len(calls), after_cancel

    small, small_elapsed, total, after_cancel = asyncio.run(scenario())
    assert len(small.target_matches("s")["simbad"]) == 2
    assert total < n / 2, total  # the abandoned association stopped
    if after_cancel:  # only the slice that was running when the batch was cancelled may still finish
        assert after_cancel[-1] - after_cancel[0] <= 2 * batch.CPU_SLICE_SECONDS + 0.5
    assert small_elapsed < 2.0, small_elapsed


def test_cpu_slot_serves_the_least_used_request_first():
    async def scenario():
        slot = batch._CpuSlot()
        heavy, light = batch._CpuAccount(), batch._CpuAccount()
        heavy.used = 10.0
        order: list[str] = []
        await slot.acquire(heavy)  # held

        async def wait(name: str, account: batch._CpuAccount) -> None:
            await slot.acquire(account)
            order.append(name)
            slot.release()

        tasks = [asyncio.create_task(wait("heavy-1", heavy)), asyncio.create_task(wait("heavy-2", heavy))]
        await asyncio.sleep(0)
        cancelled = asyncio.create_task(wait("cancelled", light))
        await asyncio.sleep(0)
        cancelled.cancel()
        tasks.append(asyncio.create_task(wait("light", light)))
        await asyncio.sleep(0)
        slot.release()
        await asyncio.gather(*tasks)
        return order, slot.busy

    order, busy = asyncio.run(scenario())
    assert order == ["light", "heavy-1", "heavy-2"] and not busy


def test_time_slices_process_every_item_once_in_order(monkeypatch):
    monkeypatch.setattr(batch, "CPU_SLICE_SECONDS", 0.0)  # one item per slice
    seen: list[int] = []
    calls = 0
    original = batch._offload

    async def counting(fn, *args, **kwargs):
        nonlocal calls
        calls += 1
        return await original(fn, *args, **kwargs)

    monkeypatch.setattr(batch, "_offload", counting)
    asyncio.run(batch._offload_steps(seen.append, iter(range(25))))
    assert seen == list(range(25)) and calls == 26


# ---------------------------------------------------------------------------
# 5. The batch routes' own body limit under an application-wide limit
# ---------------------------------------------------------------------------


def _app_with_global_body_limit(maximum: int):
    """An app with api.py's global body-size middleware (MAX_REQUEST_BYTES), exempting the batch routes."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.middleware("http")
    async def request_limit(request: Request, call_next):
        if batch.owns_request_body_limit(request.url.path):
            return await call_next(request)
        length = request.headers.get("content-length")
        if (length and int(length) > maximum) or len(await request.body()) > maximum:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.post("/other")
    async def other(request: Request):
        return {"bytes": len(await request.body())}

    app.include_router(batch.router)
    return app


def test_batch_routes_enforce_their_own_body_limit(monkeypatch):
    """A 45,000-target CSV (1.3 MB) through an app with a 1 MB global body limit: HTTP 413 before; the batch
    routes stream their bodies against BATCH_MAX_UPLOAD_BYTES themselves."""
    from fastapi.testclient import TestClient

    assert batch.owns_request_body_limit("/api/v1/batch/crossmatch")
    assert batch.owns_request_body_limit("/api/v1/batch")
    assert not batch.owns_request_body_limit("/api/v1/batchx") and not batch.owns_request_body_limit("/api/v1/search")
    csv_text = "id,ra,dec\n" + "".join(f"t{i},{(i * 0.0071) % 360:.6f},{-80 + (i * 0.0033) % 160:.6f}\n"
                                        for i in range(45_000))
    assert len(csv_text) > 1_048_576
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=lambda request: httpx.Response(200, json=simbad_json(simbad_rows_for(request))))
        with TestClient(_app_with_global_body_limit(1_048_576)) as client:
            response = client.post("/api/v1/batch/crossmatch?catalogs=simbad&format=rows&include_data=false",
                                   content=csv_text, headers={"content-type": "text/csv"})
            assert response.status_code == 200, response.text[:300]
            assert response.json()["summary"]["targets"] == 45_000
            assert client.post("/other", content=b"x" * 1_100_000).status_code == 413
            monkeypatch.setenv("BATCH_MAX_UPLOAD_BYTES", "1000000")
            refused = client.post("/api/v1/batch/crossmatch?catalogs=simbad", content=csv_text,
                                  headers={"content-type": "text/csv"})
            assert refused.status_code == 413 and "BATCH_MAX_UPLOAD_BYTES" in refused.text


# ---------------------------------------------------------------------------
# 6. Oversized target lists are refused with bounded memory
# ---------------------------------------------------------------------------


def _peak_bytes(fn) -> int:
    tracemalloc.start()
    try:
        fn()
    except BatchError as exc:
        assert "too many targets" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected a BatchError")
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def test_oversized_csv_is_refused_without_reading_every_row(monkeypatch):
    """422k CSV lines over the target limit: +271 MB and 3.7 s before; now the first excess row stops it."""
    monkeypatch.setenv("BATCH_MAX_TARGETS", "1000")
    text = "id,ra,dec\n" + "".join(f"t{i},{(i * 0.0071) % 360:.6f},{(i * 0.0033) % 80:.6f}\n" for i in range(300_000))
    started = time.perf_counter()
    peak = _peak_bytes(lambda: batch.read_targets_csv(text))
    assert time.perf_counter() - started < 2.0
    assert peak < 8 * 1024 * 1024, peak  # the text itself is ~9 MB; a list of its lines alone was ~25 MB


@pytest.mark.parametrize("wrap", [False, True], ids=["array", "object"])
def test_oversized_json_is_refused_without_decoding_every_target(monkeypatch, wrap):
    monkeypatch.setenv("BATCH_MAX_TARGETS", "1000")
    items = ",".join(f'{{"id":"t{i}","ra":{(i * 0.0071) % 360:.6f},"dec":{(i * 0.0033) % 80:.6f}}}'
                     for i in range(250_000))
    text = f'{{"catalogs":["simbad"],"targets":[{items}]}}' if wrap else f"[{items}]"
    peak = _peak_bytes(lambda: batch.read_targets_json(text))
    assert peak < 8 * 1024 * 1024, peak  # json.loads of the whole list: ~150 MB

    body = text.encode()
    job_peak = _peak_bytes(lambda: batch._json_job(body, {}))
    # The body is decoded to text once (as json.loads does); the target list is not built beyond the limit.
    assert job_peak < len(body) + 8 * 1024 * 1024, (job_peak, len(body))


def test_router_refuses_oversized_json_quickly(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BATCH_MAX_TARGETS", "100")
    body = json.dumps({"targets": [{"ra": 1.0 + i * 1e-4, "dec": 1.0} for i in range(20_000)]})
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=_no_upstream)
        with TestClient(make_app()) as client:
            response = client.post("/api/v1/batch/crossmatch?catalogs=simbad", content=body,
                                   headers={"content-type": "application/json"})
    assert response.status_code == 422 and "too many targets" in response.text


def test_incremental_json_decoder_matches_json_loads():
    for text in ['[]', '{"targets": []}', ' [ {"ra": 1, "dec": 2} , {"ra": 3, "dec": 4} ] ',
                 '{"a": {"targets": [1]}, "targets": [{"x": [1, {"y": null}]}], "b": "\\u00e9"}',
                 '{"targets": [1], "targets": [2, 3]}', '{"targets": {"not": "an array"}}', '"text"', "null",
                 '[NaN, Infinity]']:
        assert batch._decode_targets_json(text, 10) == json.loads(text), text
    for bad in ['[1,]', '{"targets": [1] ', '{"a" 1}', '[1] x', '{,}', '{"targets": [1],}']:
        with pytest.raises(json.JSONDecodeError):
            batch._decode_targets_json(bad, 10)
    with pytest.raises(BatchError, match="too many targets"):
        batch._decode_targets_json('{"targets": [1, 2, 3]}', 2)
    assert batch._decode_targets_json('{"targets": [1, 2]}', 2) == {"targets": [1, 2]}


# ---------------------------------------------------------------------------
# 7. XMatch layout (too-wide split, radius buckets) off the event loop
# ---------------------------------------------------------------------------


def test_too_wide_layout_runs_off_the_event_loop():
    """50,000 targets at epoch 1995 against gaia_dr3 with at most 100 cone targets: every cone is capped at 180".
    The split, the per-target warnings and the radius buckets ran on the event loop (0.86 s for 100,000
    targets, worst stall 1.17 s), and run.warnings held one string per target."""
    n = 50_000
    targets = parse_targets([{"id": str(i), "ra": (i * 0.0071) % 360, "dec": 0.0, "epoch": 1995.0 + (i % 7) * 0.001}
                             for i in range(n)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=xmatch_votable([]), headers={"content-type": "text/xml"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine = BatchCrossmatcher(client=client, max_cone_targets=100)
            return await _with_ticker(engine.run(targets, ["gaia_dr3"], radius_arcsec=5.0))

    result, worst = asyncio.run(scenario())
    run = result.runs["gaia_dr3"]
    assert run.failed_targets == 0 and run.refused_targets == 0
    assert any("capped at the XMatch limit" in w for w in run.warnings)
    assert len(run.warnings) == len(set(run.warnings)) <= batch._MAX_RUN_WARNINGS + 1
    assert len(run.warnings) < 20, run.warnings[:5]  # 7 distinct epochs -> 7 distinct cone warnings
    assert worst < 0.5, worst


def test_collect_warnings_keeps_each_distinct_warning_once(monkeypatch):
    monkeypatch.setattr(batch, "_MAX_RUN_WARNINGS", 3)
    run = batch.CatalogRun("c", "xmatch", None)
    results = [batch.QueryResult([], {"warnings": [f"w{i % 5}"]}) for i in range(100)]
    batch._collect_warnings(run, results)
    assert run.warnings == ["w0", "w1", "w2", "c: 2 more distinct per-target warning(s) not listed."]


# ---------------------------------------------------------------------------
# 8. JSON bodies under other content types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content_type", ["application/x-www-form-urlencoded", "text/json", "application/vnd.api+json",
                                          "text/plain", "application/json; charset=utf-8", None])
def test_json_bodies_under_any_content_type(content_type):
    body = json.dumps({"targets": [{"id": "a", "ra": 10, "dec": 10}], "catalogs": ["simbad"]})
    headers = {"content-type": content_type} if content_type else {}
    from fastapi.testclient import TestClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=handler)
        with TestClient(make_app()) as client:
            request = client.build_request("POST", "/api/v1/batch/crossmatch", content=body.encode(), headers=headers)
            if content_type is None:
                request.headers.pop("content-type", None)
            response = client.send(request)
    assert response.status_code == 200, (content_type, response.text[:300])
    assert [t["id"] for t in response.json()["targets"]] == ["a"]


def test_csv_under_a_json_content_type_is_a_json_error():
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as client:
        response = client.post("/api/v1/batch/crossmatch", content=b"id,ra,dec\na,1,1\n",
                               headers={"content-type": "application/json"})
    assert response.status_code == 422 and "invalid JSON body" in response.text


def test_multipart_form_data_is_not_sniffed_as_json():
    from fastapi.testclient import TestClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=handler)
        with TestClient(make_app()) as client:
            response = client.post("/api/v1/batch/crossmatch", data={"catalogs": "simbad"},
                                   files={"file": ("t.json", b'[{"id": "a", "ra": 1, "dec": 1}]', "application/json")})
    assert response.status_code == 200, response.text
    assert [t["id"] for t in response.json()["targets"]] == ["a"]


def test_one_cpu_account_per_batch(monkeypatch):
    """The catalogs of one batch run in separate tasks: they share the batch's CPU account (fair queuing is per
    request, not per catalog task)."""
    created: list[Any] = []
    original = batch._CpuAccount

    class Counting(original):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    monkeypatch.setattr(batch, "_CpuAccount", Counting)

    def handler(request: httpx.Request) -> httpx.Response:
        if "cdsxmatch" in (request.url.host or ""):
            return httpx.Response(200, content=xmatch_votable([]), headers={"content-type": "text/xml"})
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            targets = parse_targets(_big_targets(50))
            return await BatchCrossmatcher(client=client).run(targets, ["simbad", "gaia_dr3"], radius_arcsec=3.0)

    result = asyncio.run(scenario())
    assert result.runs["simbad"].matched_targets == 50
    assert len(created) == 1 and created[0].used > 0.0
