"""Regression tests for the second adversarial review of vizier.py (offline).

Each section names the reported defect. Upstream behaviour is replayed from the recordings in
tests/fixtures/vizier (recorded live with ``python tests/test_vizier.py record <case>``) or
produced by a scripted httpx transport for failure modes. Registry concurrency tests use real
subprocesses (file locks, files held open by another process).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from urllib.parse import unquote_plus
from xml.etree import ElementTree

import httpx
import pytest
import respx
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_vizier import (
    BARNARD,
    DESCRIBE_FOR_TABLE,
    _app,
    _client,
    _crossmatch_case,
    _exchanges,
    _match,
    replay_describe,
    replaying,
)
from test_vizier_regressions import kind_of, replay_handler, scripted_client, status

import models
import vizier
from models import (
    DEFAULT_CATALOGS,
    CatalogRegistry,
    catalog_from_dict,
    compute_positional_error,
    plan_cone,
    resolve_epoch,
    validate_catalog_definition,
    validate_target,
)

PYTHON = sys.executable
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
WINDOWS = os.name == "nt"


@pytest.fixture(autouse=True)
def _fresh_describe_cache():
    vizier.clear_describe_cache()
    yield
    vizier.clear_describe_cache()


@pytest.fixture(autouse=True)
def _no_reference_lookup(monkeypatch):
    """The replays hold no ADS/doi.org traffic: registrations store no resolved references here (the lookup
    itself is tested in test_vizier_final / test_provenance_final)."""

    async def none(*_args, **_kwargs):
        return [], []

    monkeypatch.setattr(vizier, "citation_references", none)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(vizier, "RETRY_BACKOFF_SECONDS", 0.0)


def col(name: str, unit: str | None, ucd: str | None, description: str = "", datatype: str = "DOUBLE",
        principal: bool = False) -> vizier.ColumnInfo:
    return vizier.ColumnInfo(name, unit, ucd, datatype, description, principal)


def _entry() -> dict:
    return json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))


def _sent_queries(router: respx.MockRouter) -> list[str]:
    return [unquote_plus((call.request.content or b"").decode()) for call in router.calls]


# ---------------------------------------------------------------------------
# Issue 1 (critical): dotted column names (_RA.icrs) are renamed by TAPVizieR's JSON
# ---------------------------------------------------------------------------


def test_select_item_keeps_dotted_names():
    assert vizier.select_item("_RA.icrs") == '"_RA.icrs" AS "_RA.icrs"'
    assert vizier.select_item("GSC2.3") == '"GSC2.3" AS "GSC2.3"'
    assert vizier.select_item("RA(ICRS)") == '"RA(ICRS)"'  # only '.' is rewritten (verified live)
    assert vizier.select_item("RAJ2000") == '"RAJ2000"'


async def test_abell_uses_the_icrs_columns_and_confirms_degrees_through_the_alias():
    desc = await replay_describe("VII/110A/table3")
    assert (desc.ra_column, desc.dec_column) == ("_RA.icrs", "_DE.icrs")
    assert desc.frame == "ICRS"
    # The largest RA was read back (the TOP 1 query selects "_RA.icrs" AS "_RA.icrs").
    assert "exceeds 24, confirming decimal degrees" in (desc.position_unit_check or "")
    assert not any("not confirmed by the data" in a for a in desc.assumptions)
    assert any("RAB1950 is an FK4 position" in a and "_RA.icrs/_DE.icrs" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    params = entry["parameters"]
    assert '"_RA.icrs" AS "_RA.icrs"' in params["columns"] and '"_DE.icrs" AS "_DE.icrs"' in params["columns"]
    assert params["ra_field"] == '"_RA.icrs"' and params["dec_field"] == '"_DE.icrs"'
    # The B1950 main position (pos.eq.ra;meta.main) would feed the canonical ra/dec by UCD.
    assert not {'"RAB1950"', '"DEB1950"'} & set(params["columns"])
    assert params["column_units"] == {"_RA.icrs": "deg", "_DE.icrs": "deg"}
    assert validate_catalog_definition(catalog_from_dict("abell", entry)) == []


async def test_crossmatch_abell_reports_the_icrs_position_not_the_b1950_one(tmp_path):
    _registration, result, record = await _crossmatch_case("crossmatch_abell_coma", tmp_path)
    source = result["sources"][0]
    assert source.source_id == "1656"
    # TAPVizieR row: _RA.icrs 194.95304722, _DE.icrs 27.98070028 (RAB1950 194.35, DEB1950 28.25).
    assert source.ra == pytest.approx(194.9530472, abs=1e-6) and source.dec == pytest.approx(27.9807003, abs=1e-6)
    assert source.data["_RA.icrs"] == pytest.approx(194.95304722222218)
    assert "_RA_icrs" not in source.data
    assert _match(record, "1656")["separation_arcsec"] == pytest.approx(252.27, abs=0.1)
    assert '"_RA.icrs" AS "_RA.icrs"' in result["query"]
    assert "CONTAINS(POINT('ICRS', \"_RA.icrs\", \"_DE.icrs\")" in result["query"]


async def test_white_dwarf_catalogue_uses_the_icrs_epoch_2000_columns():
    desc = await replay_describe("B/wd/catalog")
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("_RA.icrs", "_DE.icrs", "WD")
    assert desc.epoch == 2000.0 and "Epoch=J2000" in (desc.epoch_source or "")
    _, entry = vizier.build_definition(desc)
    assert '"_RA.icrs" AS "_RA.icrs"' in entry["parameters"]["columns"]
    assert '"RAB1950"' not in entry["parameters"]["columns"]


def test_rows_keyed_by_dotted_names_normalize_to_the_icrs_position():
    # What the provider receives for the self-aliased columns: the real dotted names.
    catalog = catalog_from_dict("x", {
        "provider": "tap", "table": '"X/1/t"', "parameters": {
            "columns": ['"ID"', '"_RA.icrs" AS "_RA.icrs"', '"_DE.icrs" AS "_DE.icrs"'],
            "ra_field": '"_RA.icrs"', "dec_field": '"_DE.icrs"', "id_field": '"ID"'}})
    from providers import TapProvider

    mapping = TapProvider._field_map(catalog)
    row = {"ID": 1, "_RA.icrs": 194.95, "_DE.icrs": 27.98}
    values = models.normalize_source_record(row, field_map=mapping)
    assert (values["ra"], values["dec"]) == (194.95, 27.98)


def test_dotted_identifier_error_and_epoch_columns_are_selected_under_their_own_names():
    cols = [col("GSC2.3", None, "meta.id;meta.main", "GSC2.3 name", datatype="CHAR(10)"),
            col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension (J2000)"),
            col("DEJ2000", "deg", "pos.eq.dec;meta.main", "Declination (J2000)"),
            col("e.pos", "arcsec", "stat.error;pos", "1-sigma positional uncertainty"),
            col("Obs.Date", "d", "time.epoch", "Observation date (MJD)")]
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, cols)
    assert (desc.id_column, desc.epoch) == ("GSC2.3", "Obs.Date")
    assert desc.pos_error == {"columns": ["e.pos"], "units": ["arcsec"], "kind": "sigma"}
    _, entry = vizier.build_definition(desc, overrides={"epoch_range": [1990.0, 2000.0]})
    columns = entry["parameters"]["columns"]
    for name in ("GSC2.3", "e.pos", "Obs.Date"):
        assert f'"{name}" AS "{name}"' in columns
    assert entry["parameters"]["id_field"] == '"GSC2.3"'
    definition = catalog_from_dict("x", entry)
    assert validate_catalog_definition(definition) == []
    # The row TAPVizieR returns for these select items (names kept verbatim by the alias):
    row = {"GSC2.3": "N9DZ000123", "RAJ2000": 10.0, "DEJ2000": 20.0, "e.pos": 0.3, "Obs.Date": 51544.5}
    from providers import TapProvider

    values = models.normalize_source_record(row, field_map=TapProvider._field_map(definition))
    assert values["source_id"] == "N9DZ000123"
    assert compute_positional_error(row, definition.pos_error)[0] == pytest.approx(0.3)
    assert resolve_epoch(row, definition.epoch, definition.epoch_format) == pytest.approx(2000.0)


def test_extra_column_that_would_feed_the_position_is_refused():
    cols = [col("ID", None, "meta.id;meta.main", datatype="INTEGER"),
            col("RAB1950", "deg", "pos.eq.ra;meta.main", "Right ascension (B1950)"),
            col("DEB1950", "deg", "pos.eq.dec;meta.main", "Declination (B1950)"),
            col("_RA.icrs", None, "pos.eq.ra", "Right ascension (ICRS) (computed by VizieR)"),
            col("_DE.icrs", None, "pos.eq.dec", "Declination (ICRS) (computed by VizieR)")]
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, cols, max_ra=359.0)
    assert desc.ra_column == "_RA.icrs"
    with pytest.raises(vizier.VizierInputError, match="'RAB1950' .* would be read as the ra of rows"):
        vizier.build_definition(desc, overrides={"extra_columns": ["RAB1950"]})
    with pytest.raises(vizier.VizierInputError, match="'DEB1950' .* would be read as the dec of rows"):
        vizier.build_definition(desc, overrides={"extra_columns": ["DEB1950"]})
    _, entry = vizier.build_definition(desc)
    assert not {'"RAB1950"', '"DEB1950"'} & set(entry["parameters"]["columns"])


# ---------------------------------------------------------------------------
# Issue 2 (critical): per-row-epoch catalogues had no epoch_range (no cone propagation)
# ---------------------------------------------------------------------------


async def test_2mass_epoch_range_comes_from_the_core_entry_for_the_same_table():
    desc = await replay_describe("II/246/out")
    assert desc.epoch == "JD" and desc.epoch_format == "jd"
    assert desc.epoch_range == DEFAULT_CATALOGS["vizier_2mass_reference"]["parameters"]["epoch_range"]
    assert any("core 'vizier_2mass_reference' entry" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    definition = catalog_from_dict("tm", entry)
    assert models.catalog_epoch_range(definition) == (1997.4, 2001.2)
    target = validate_target(BARNARD["ra"], BARNARD["dec"], epoch=2000.0, pm_ra_masyr=BARNARD["pm_ra_masyr"],
                             pm_dec_masyr=BARNARD["pm_dec_masyr"])
    assert plan_cone(definition, target, 3.0).mode == "target_pm"


async def test_crossmatch_2mass_finds_barnards_star(tmp_path):
    registration, _result, record = await _crossmatch_case("crossmatch_2mass_barnard", tmp_path)
    assert registration.entry["parameters"]["epoch_range"] == [1997.4, 2001.2]
    match = _match(record, "17574849+0441405")
    assert match["separation_arcsec"] < 0.5 and match["confidence"] > 0.99


async def test_erass1_epoch_range_is_scanned_and_proxima_is_found(tmp_path):
    desc = await replay_describe("J/A+A/682/A34/erass1-m")
    assert desc.epoch == {"span_columns": ["b_MJD", "B_MJD"], "point_max_years": 0.1, "format": "mjd"}
    lo, hi = desc.epoch_range
    # MIN(MJD_MIN) = 58828.922, MAX(MJD_MAX) = 59011.43 over the 930,203 rows (live 2026-09-29).
    assert lo == pytest.approx(2000.0 + (58828.922 - 51544.5) / 365.25, abs=1e-3)
    assert hi == pytest.approx(2000.0 + (59011.43 - 51544.5) / 365.25, abs=1e-3)
    assert any("MIN/MAX over all 930,203 rows" in a for a in desc.assumptions) and desc.complete
    registration, result, record = await _crossmatch_case("crossmatch_erass1_proxima", tmp_path)
    assert registration.entry["parameters"]["epoch_range"] == desc.epoch_range
    match = _match(record, "1eRASS J142932.3-624030")
    assert match["confidence"] > 0.95
    source = result["sources"][0]
    assert source.positional_error_arcsec == pytest.approx(source.data["posErr"], rel=1e-9)


async def test_large_per_row_epoch_table_is_sampled_over_the_sky():
    desc = await replay_describe("I/329/urat1")
    assert desc.epoch == "Epoch" and desc.nrows and desc.nrows > vizier.EPOCH_SCAN_MAX_ROWS
    lo, hi = desc.epoch_range
    # 26 of 48 cones held rows: 2012.311-2014.469, widened by 25% of the span on each side.
    assert (lo, hi) == pytest.approx((2011.771, 2015.009), abs=1e-3)
    assert lo < 2012.3 and hi > 2015.0  # URAT1 observed 2012.3-2015.0: the estimate covers it
    note = next(a for a in desc.assumptions if a.startswith("epoch: per-row epochs sampled"))
    assert "26 of 48 cones" in note and "overrides.epoch_range" in note


async def test_epoch_range_scan_failure_is_an_assumption_and_not_cached(no_backoff):
    async def down(request: httpx.Request) -> httpx.Response:
        return await status(503, "maintenance")

    client, transport = scripted_client(replay_handler("describe_erass1", overrides={"tap_other": down}))
    async with client:
        desc = await vizier.describe_table("J/A+A/682/A34/erass1-m", client=client)
        assert desc.epoch_range is None and desc.registrable and not desc.complete
        assert any("could not be determined" in a and "overrides.epoch_range" in a for a in desc.assumptions)
        before = len(transport.requests)
        await vizier.describe_table("J/A+A/682/A34/erass1-m", client=client)
        assert len(transport.requests) > before  # incomplete: described again


def test_epoch_expressions_and_filters():
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, [
        col("RAJ2000", "deg", "pos.eq.ra;meta.main"), col("DEJ2000", "deg", "pos.eq.dec;meta.main"),
        col("EpRA-1990", "yr", "time.epoch", "[0.81,2.13] epoch-1990 of RAdeg")])
    # The description gives the range ('[0.81,2.13]'); the expression is the derived column.
    assert desc.epoch == "EpRA-1990_jyear" and desc.epoch_range == [1990.81, 1992.13]
    exprs, fmt = vizier._epoch_expressions(desc)
    assert exprs == [('"EpRA-1990" + 1990', "both")] and fmt == "jyear"
    assert vizier._epoch_filter('"MJD"', "mjd") == '"MJD" BETWEEN -21000 AND 124000'
    assert vizier._epoch_filter('"X"', None) == '"X" IS NOT NULL'
    assert vizier._per_row_epoch("JD") and vizier._per_row_epoch({"span_columns": ["a", "b"]})
    assert not vizier._per_row_epoch(2000.0) and not vizier._per_row_epoch(None)


# ---------------------------------------------------------------------------
# Issue 3 (critical): eRASS1 one-sided lower errors used as a symmetric sigma
# ---------------------------------------------------------------------------


async def test_erass1_uses_the_calibrated_total_positional_error():
    desc = await replay_describe("J/A+A/682/A34/erass1-m")
    assert desc.pos_error == {"columns": ["posErr"], "units": ["arcsec"], "kind": "sigma"}
    assert any("e_RA_ICRS/e_DE_ICRS are one-sided" in a for a in desc.assumptions)
    assert any("total positional error posErr" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    assert '"e_RA_ICRS"' not in entry["parameters"]["columns"]  # stat.error;pos.eq.ra not chosen: guarded
    # AB Dor (1eRASS J052844.6-652650): RA_LOWERR 0.0002" but POS_ERR 3.05".
    sigma, _ = compute_positional_error({"posErr": 3.05, "e_RA_ICRS": 0.0002, "e_DE_ICRS": 2.9}, entry["pos_error"])
    assert sigma == pytest.approx(3.05)


def test_one_sided_pair_without_a_total_error_uses_the_larger_side():
    cols = [col("RA_ICRS", "deg", "pos.eq.ra;meta.main"), col("DE_ICRS", "deg", "pos.eq.dec;meta.main"),
            col("e_RA_ICRS", "arcsec", "stat.error;pos.eq.ra", "1-sigma lower error on RA"),
            col("E_RA_ICRS", "arcsec", "stat.error;stat.max", "1-sigma upper error on RA"),
            col("e_DE_ICRS", "arcsec", "stat.error;pos.eq.dec", "1-sigma lower error on Dec"),
            col("E_DE_ICRS", "arcsec", "stat.error;stat.max", "1-sigma upper error on Dec")]
    derived: dict[str, str] = {}
    spec, used, notes = vizier.detect_pos_error(cols, cols[0], cols[1], derived=derived)
    assert spec == {"columns": ["e_RA_ICRS_max", "e_DE_ICRS_max"], "units": ["arcsec", "arcsec"], "kind": "sigma"}
    assert derived["e_RA_ICRS_max"] == ('(ABS("e_RA_ICRS") + ABS("E_RA_ICRS") + '
                                        'ABS(ABS("e_RA_ICRS") - ABS("E_RA_ICRS"))) / 2')
    assert [u["role"] for u in used] == ["ra_error_lower", "ra_error_upper", "dec_error_lower", "dec_error_upper"]
    assert any("larger of the lower/upper errors" in n for n in notes)
    # max(|a|, |b|) = (|a| + |b| + ||a| - |b||) / 2
    for a, b in ((0.0002, 3.0), (-2.5, 1.0), (1.0, 1.0)):
        assert (abs(a) + abs(b) + abs(abs(a) - abs(b))) / 2 == max(abs(a), abs(b))
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, cols)
    assert set(desc.derived_columns) == {"e_RA_ICRS_max", "e_DE_ICRS_max"}
    _, entry = vizier.build_definition(desc)
    assert f'{derived["e_RA_ICRS_max"]} AS "e_RA_ICRS_max"' in entry["parameters"]["columns"]
    assert validate_catalog_definition(catalog_from_dict("x", entry)) == []


def test_one_sided_descriptions_are_recognised():
    by_name: dict[str, vizier.ColumnInfo] = {}
    for text in ("? 1-sigma lower error on RA (RA_LOWERR)", "upper error on Dec", "Lo_Unc position",
                 "negative error in RA"):
        assert vizier._is_one_sided(col("x", "arcsec", "stat.error;pos.eq.ra", text), by_name), text
    assert not vizier._is_one_sided(col("e_RA", "arcsec", "stat.error;pos.eq.ra", "Mean error on RA"), by_name)


# ---------------------------------------------------------------------------
# Issue 4 (major): wavelength of single-table catalogues from the IVOA registry
# ---------------------------------------------------------------------------


async def test_single_table_catalogue_wavelength_from_regtap():
    desc = await replay_describe("II/246/out")
    assert desc.wavelength == "infrared" and desc.catalog.wavelengths == ["IR"]
    _, entry = vizier.build_definition(desc)
    assert entry["profiles"] == ["full", "vizier", "infrared"]
    fermi = await replay_describe("J/ApJS/247/33/4fgl")
    assert fermi.catalog.wavelengths == ["Gamma-ray", "Radio"] and fermi.wavelength == "multi"
    _, entry = vizier.build_definition(fermi)
    assert entry["profiles"] == ["full", "vizier", "high-energy", "radio"]
    assert "waveband" in vizier._regtap_citation_adql("II/246")


async def test_waveband_is_looked_up_when_asu_gives_a_bibcode_but_no_wavelength():
    # ASU catalogue answer of 2SXPS with its '-kw.Wavelength' INFO removed (bibcode kept); the
    # RegTAP answer is the recorded rr.resource row of IX/70 (waveband 'x-ray').
    asu = next(e for e in _exchanges("describe_2sxps") if "-meta=" in e.url and "-meta.all" not in e.url)
    stripped = asu.content.replace(b'<INFO name="-kw.Wavelength" value="X-ray"/>', b"")
    assert stripped != asu.content
    regtap = next(e for e in _exchanges("describe_csc21") if "reg.g-vo.org" in e.url)
    seen: list[str] = []

    async def asu_catalog(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stripped, headers={"content-type": asu.content_type})

    async def registry(request: httpx.Request) -> httpx.Response:
        seen.append(unquote_plus(request.content.decode()))
        return httpx.Response(200, content=regtap.content, headers={"content-type": regtap.content_type})

    client, _ = scripted_client(replay_handler("describe_2sxps", overrides={"asu_catalog": asu_catalog,
                                                                             "regtap": registry}))
    async with client:
        desc = await vizier.describe_table("IX/58/2sxps", client=client)
    assert desc.catalog.bibcode == "2020ApJS..247...54E"  # from ASU, untouched
    assert len(seen) == 1 and "waveband" in seen[0]
    assert desc.catalog.wavelengths == ["X-ray"] and desc.wavelength == "xray" and desc.complete


# ---------------------------------------------------------------------------
# Issue 5 (minor): e_<RA>/e_<Dec> with a bare stat.error UCD (Hipparcos-2)
# ---------------------------------------------------------------------------


async def test_hipparcos2_errors_are_found_by_name():
    desc = await replay_describe("I/311/hip2")
    assert desc.pos_error == {"columns": ["e_RArad", "e_DErad"], "units": ["mas", "mas"], "kind": "sigma"}
    assert any("identified as the RA/Dec errors by their names" in a for a in desc.assumptions)
    sigma, _ = compute_positional_error({"e_RArad": 0.8, "e_DErad": 0.6}, desc.pos_error)
    assert sigma == pytest.approx(math.sqrt((0.8**2 + 0.6**2) / 2) / 1000.0)


def test_greek_axis_names_in_error_descriptions():
    assert vizier._error_axis("Mean error of alpha*cos(delta)") == "ra"
    assert vizier._error_axis("Mean error of delta") == "dec"
    assert vizier._error_axis("Error on RA*cos(dec)") == "ra"


# ---------------------------------------------------------------------------
# Issue 6 (minor): old-equinox main positions chosen over the ICRS columns
# ---------------------------------------------------------------------------


async def test_bonner_durchmusterung_uses_the_computed_icrs_columns():
    desc = await replay_describe("I/122/bd")
    assert (desc.ra_column, desc.dec_column) == ("_RA.icrs", "_DE.icrs") and desc.registrable
    assert any("RA1855 is an FK4 position (COOSYS eq_FK4, equinox B1855)" in a for a in desc.assumptions)


async def test_table_without_icrs_alternative_names_its_real_frame():
    desc = await replay_describe("VII/118/ngc2000")
    assert not desc.registrable and desc.ra_column is None
    problem = desc.problems[0]
    assert "RAB2000 is an FK4 position (COOSYS eq_FK4, equinox B2000)" in problem
    assert "B1950" not in problem and "no J2000/ICRS position column" in problem


def test_old_equinox_names_and_descriptions_without_coosys():
    for name, text in (("RA1900", "Right ascension 1900"), ("RAB1950", "Right ascension"),
                       ("RA", "Right ascension (B1950)"), ("RA", "Right ascension, equinox=B1875")):
        assert vizier._old_frame(col(name, "deg", "pos.eq.ra;meta.main", text), {}), (name, text)
    for name, text in (("RAdeg", "alpha, degrees (ICRS, Epoch=J1991.25)"), ("RAJ2000", "Right ascension (J2000)"),
                       ("_RA.icrs", "Right ascension (ICRS) (computed by VizieR)")):
        assert vizier._old_frame(col(name, "deg", "pos.eq.ra", text), {}) is None, (name, text)
    assert "FK5 position at equinox B1950" not in (vizier._old_frame(col("RA", "deg", "pos.eq.ra"), {
        "system": "eq_FK5", "equinox": "J2000"}) or "")
    assert vizier._old_frame(col("RA", "deg", "pos.eq.ra"), {"system": "eq_FK5", "equinox": "J1975"})


# ---------------------------------------------------------------------------
# Issue 7 (minor): spectral model names mapped to the canonical spectral type
# ---------------------------------------------------------------------------


async def test_fermi_spectral_model_is_not_a_spectral_type():
    desc = await replay_describe("J/ApJS/247/33/4fgl")
    assert "spectral_type" not in desc.field_map
    assert any(a.startswith("spectral_type: Mod") and "spectral model" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    assert '"Mod"' not in entry["parameters"]["columns"]  # would feed spectral_type by UCD


def test_spectral_type_mapping_is_for_stellar_classifications_only():
    stellar = [col("SpT", None, "src.spType", "Spectral type", datatype="CHAR(8)")]
    assert vizier.detect_field_map_notes(stellar, wavelength="optical")[0] == {"spectral_type": "SpT"}
    mapping, notes = vizier.detect_field_map_notes(stellar, wavelength="xray")
    assert mapping == {} and "xray catalogue" in notes[0]
    model = [col("Shape", None, "src.spType", "Best-fitting spectral shape", datatype="CHAR(12)")]
    assert vizier.detect_field_map_notes(model)[0] == {}


# ---------------------------------------------------------------------------
# Issue 8 (major): an HTTP-200 error answer to the ASU metadata call was accepted
# ---------------------------------------------------------------------------


ASU_ERROR_200 = ('<?xml version="1.0"?><VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
                 '<INFO name="Error" value="service temporarily unavailable (database connection lost)"/>'
                 '</VOTABLE>')


@pytest.mark.parametrize("case", ["describe_tycho2", "describe_ppmxl", "describe_2sxps", "describe_gaiadr3"])
async def test_asu_table_error_answer_blocks_registration_and_is_not_cached(case):
    table = next(t for t, c in DESCRIBE_FOR_TABLE.items() if c == case)

    async def error_200(request: httpx.Request) -> httpx.Response:
        return await status(200, ASU_ERROR_200, "text/xml")

    client, transport = scripted_client(replay_handler(case, overrides={"asu_table": error_200}))
    async with client:
        desc = await vizier.describe_table(table, client=client)
        assert not desc.registrable and not desc.complete
        assert any("VizieR ASU table metadata unusable (error 'service temporarily unavailable" in p
                   for p in desc.problems)
        with pytest.raises(vizier.VizierUpstreamError, match="COOSYS epoch") as info:
            vizier.build_definition(desc)
        assert info.value.transient  # an error answer of the metadata service is an outage (502)
        before = len(transport.requests)
        await vizier.describe_table(table, client=client)
        assert len(transport.requests) > before


async def test_asu_table_answer_without_the_chosen_ra_field_is_a_problem():
    async def fields_without_ra(request: httpx.Request) -> httpx.Response:
        body = ('<?xml version="1.0"?><VOTABLE version="1.4"><RESOURCE><TABLE>'
                '<FIELD name="TYC1" datatype="short"/></TABLE></RESOURCE></VOTABLE>')
        return await status(200, body, "text/xml")

    client, _ = scripted_client(replay_handler("describe_tycho2", overrides={"asu_table": fields_without_ra}))
    async with client:
        desc = await vizier.describe_table("I/259/tyc2", client=client)
    assert not desc.registrable and not desc.complete
    assert any("has no FIELD for RA(ICRS)" in p for p in desc.problems)


async def test_asu_catalogue_error_answer_is_not_cached_as_complete():
    async def error_200(request: httpx.Request) -> httpx.Response:
        return await status(200, ASU_ERROR_200, "text/xml")

    client, transport = scripted_client(replay_handler("describe_2sxps", overrides={"asu_catalog": error_200,
                                                                                     "regtap": error_200}))
    async with client:
        desc = await vizier.describe_table("IX/58/2sxps", client=client)
        assert desc.registrable and not desc.complete  # catalogue metadata is optional, but not cached
        assert any("no catalogue metadata for IX/58 (error: service temporarily unavailable" in a
                   for a in desc.assumptions)
        before = len(transport.requests)
        await vizier.describe_table("IX/58/2sxps", client=client)
        assert len(transport.requests) > before


def test_asu_error_message_detection():
    assert vizier.asu_error_message(ElementTree.fromstring(ASU_ERROR_200)).startswith("service temporarily")
    status_error = ('<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="ERROR">bad source</INFO>'
                    '</RESOURCE></VOTABLE>')
    assert vizier.asu_error_message(ElementTree.fromstring(status_error)) == "bad source"
    assert vizier.asu_error_message(ElementTree.fromstring('<VOTABLE><INFO name="Warning" value="x"/></VOTABLE>')) is None


# ---------------------------------------------------------------------------
# Issue 9 (major): stale built-in copies served by models.CatalogRegistry after an upgrade
# ---------------------------------------------------------------------------


def test_upgrade_refreshes_builtin_copies_and_keeps_colliding_user_entries(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("my_user_cat", _entry(), path=path)
    vizier.save_definition("another", _entry(), path=path)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["embedded_fingerprint"] == vizier.builtin_fingerprint()
    assert not vizier.registry_status(path)["builtin_copies_outdated"]
    # Simulated upgrade: a corrected built-in, a new built-in, and a built-in named like a user entry.
    corrected = json.loads(json.dumps(DEFAULT_CATALOGS["gaia_dr3"]))
    corrected["pos_error"] = {**corrected["pos_error"], "systematic_arcsec": 0.02}
    monkeypatch.setitem(DEFAULT_CATALOGS, "gaia_dr3", corrected)
    monkeypatch.setitem(DEFAULT_CATALOGS, "new_builtin", json.loads(json.dumps(DEFAULT_CATALOGS["nvss"])))
    monkeypatch.setitem(DEFAULT_CATALOGS, "my_user_cat", json.loads(json.dumps(DEFAULT_CATALOGS["first"])))
    status_before = vizier.registry_status(path)
    assert status_before["builtin_copies_outdated"] and status_before["shadowed_builtins"] == ["my_user_cat"]
    # What the core reads before any vizier call: the old copies (documented limitation).
    assert "systematic_arcsec" not in CatalogRegistry(path).get("gaia_dr3").pos_error
    # load_registry() (the registry main/api should build) serves the new built-ins AND refreshes the file.
    merged = vizier.load_registry(path)
    assert merged.get("gaia_dr3").pos_error["systematic_arcsec"] == 0.02
    assert "new_builtin" in merged.catalogs
    core = CatalogRegistry(path)
    assert core.get("gaia_dr3").pos_error["systematic_arcsec"] == 0.02 and "new_builtin" in core.catalogs
    # The colliding user entry is kept (never swallowed as a copy) and both registries agree on it.
    assert core.get("my_user_cat").description == DEFAULT_CATALOGS["vlass"]["description"]
    assert merged.get("my_user_cat").description == DEFAULT_CATALOGS["vlass"]["description"]
    assert set(vizier.list_registered(path)) == {"my_user_cat", "another"}
    status_after = vizier.registry_status(path)
    assert not status_after["builtin_copies_outdated"] and status_after["shadowed_builtins"] == ["my_user_cat"]
    vizier.save_definition("third", _entry(), path=path)
    assert set(vizier.list_registered(path)) == {"my_user_cat", "another", "third"}
    assert "my_user_cat" not in yaml.safe_load(path.read_text(encoding="utf-8"))["embedded_copies"]
    assert vizier.unregister("my_user_cat", path=path)
    assert CatalogRegistry(path).get("my_user_cat").description == DEFAULT_CATALOGS["first"]["description"]


def test_registered_route_and_cli_report_shadowed_and_outdated_copies(tmp_path, monkeypatch, capsys):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("my_user_cat", _entry(), path=path)
    monkeypatch.setitem(DEFAULT_CATALOGS, "my_user_cat", json.loads(json.dumps(DEFAULT_CATALOGS["first"])))
    body = TestClient(_app(tmp_path)).get("/api/v1/vizier/registered").json()
    assert body["shadowed_builtins"] == ["my_user_cat"] and body["builtin_copies_outdated"] is True
    assert body["catalogs"][0]["shadows_builtin"] is True
    assert vizier.main(["list", "--registry-path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "shadows it" in out and "outdated" in out


def test_refresh_failure_does_not_break_loading(tmp_path, monkeypatch, caplog):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    monkeypatch.setitem(DEFAULT_CATALOGS, "new_builtin", json.loads(json.dumps(DEFAULT_CATALOGS["nvss"])))

    def locked(*args, **kwargs):
        raise vizier.VizierConflictError("locked")

    monkeypatch.setattr(vizier, "refresh_embedded_copies", locked)
    with caplog.at_level("WARNING", logger="astrosearch.vizier"):
        merged = vizier.load_registry(path)
    assert {"mine", "new_builtin"} <= set(merged.catalogs)
    assert any("could not refresh" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Issue 10 (major): Windows replace/read failures, unusable paths
# ---------------------------------------------------------------------------


_HOLD_OPEN = textwrap.dedent("""
    import sys, time
    handle = open(sys.argv[1], encoding="utf-8")
    print("open", flush=True)
    time.sleep(float(sys.argv[2]))
""")


def _hold_open(tmp_path: Path, path: Path, seconds: float) -> subprocess.Popen:
    script = tmp_path / "hold_open.py"
    script.write_text(_HOLD_OPEN, encoding="utf-8")
    holder = subprocess.Popen([PYTHON, str(script), str(path), str(seconds)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=ENV)
    assert holder.stdout is not None and holder.stdout.readline().strip() == b"open"
    return holder


def test_save_waits_for_a_reader_that_holds_the_file_open(tmp_path):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("first_one", _entry(), path=path)
    holder = _hold_open(tmp_path, path, 1.0)
    try:
        vizier.save_definition("second_one", _entry(), path=path)  # replace retried until the handle closes
    finally:
        holder.communicate(timeout=30)
    assert set(vizier.list_registered(path)) == {"first_one", "second_one"}


def test_file_held_open_too_long_is_a_conflict_not_a_traceback(tmp_path, monkeypatch, capsys):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("first_one", _entry(), path=path)
    monkeypatch.setattr(vizier, "REGISTRY_REPLACE_RETRY_SECONDS", 0.3)
    holder = _hold_open(tmp_path, path, 8.0)
    try:
        if WINDOWS:  # a Windows replace fails while another process has the file open
            with pytest.raises(vizier.VizierConflictError, match="held open by another process"):
                vizier.save_definition("second_one", _entry(), path=path)
            with replaying("describe_2sxps"):
                code = vizier.main(["add", "IX/58/2sxps", "--registry-path", str(path)])
            assert code == 1 and "Error: the catalog registry" in capsys.readouterr().out
            app = _app(tmp_path)
            app.state.vizier_registry_path = str(path)
            with replaying("describe_2sxps"):
                response = TestClient(app).post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps"})
            assert response.status_code == 409 and "held open" in response.json()["detail"]
        else:  # POSIX replaces an open file
            vizier.save_definition("second_one", _entry(), path=path)
    finally:
        holder.kill()
        holder.communicate(timeout=30)
    assert not list(tmp_path.glob(".catalogs.*.yaml"))  # no temporary file left behind


def test_unusable_registry_path_fails_before_any_request(tmp_path, capsys):
    blocker = tmp_path / "regular_file"
    blocker.write_text("x", encoding="utf-8")
    bad = blocker / "reg.yaml"
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        assert vizier.main(["add", "IX/58/2sxps", "--registry-path", str(bad)]) == 1
        assert "cannot create the directory of the catalog registry" in capsys.readouterr().out
        app = _app(tmp_path)
        app.state.vizier_registry_path = str(bad)
        response = TestClient(app).post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps"})
        assert router.calls.call_count == 0
    assert response.status_code == 500 and "not writable" in response.json()["detail"]
    with pytest.raises(vizier.VizierRegistryIOError):
        vizier.save_definition("x_user", _entry(), path=bad)


def test_readers_retry_a_transient_permission_error(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    real = Path.read_text
    failures = {"left": 3}

    def flaky(self, *args, **kwargs):
        if self == path and failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "Permission denied")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert list(vizier.list_registered(path)) == ["mine"] and failures["left"] == 0
    failures["left"] = 10**6
    monkeypatch.setattr(vizier, "REGISTRY_READ_RETRY_SECONDS", 0.2)
    with pytest.raises(models.RegistryError, match="could not be read"):
        vizier.list_registered(path)


_SAVER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {root!r})
    import vizier
    from models import DEFAULT_CATALOGS
    entry = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    for i in range(12):
        vizier.save_definition(f"saved_{{i}}", entry, path=sys.argv[1])
""")


def test_readers_during_concurrent_saves_never_fail(tmp_path):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("seed", _entry(), path=path)
    script = tmp_path / "saver.py"
    script.write_text(_SAVER.format(root=str(ROOT)), encoding="utf-8")
    saver = subprocess.Popen([PYTHON, str(script), str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=ENV)
    reads = 0
    while saver.poll() is None:
        assert "seed" in vizier.list_registered(path)
        assert "seed" in vizier.load_registry(path).catalogs
        reads += 1
    _out, err = saver.communicate(timeout=120)
    assert saver.returncode == 0, err.decode("utf-8", "replace")
    assert reads > 0 and {f"saved_{i}" for i in range(12)} <= set(vizier.list_registered(path))


# ---------------------------------------------------------------------------
# Issue 11 (major): blocking registry I/O on the event loop
# ---------------------------------------------------------------------------


_LOCK_HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    import vizier
    from pathlib import Path
    with vizier._registry_lock(Path(sys.argv[1])):
        print("locked", flush=True)
        time.sleep(float(sys.argv[2]))
""")


def _hold_lock(tmp_path: Path, path: Path, seconds: float) -> subprocess.Popen:
    script = tmp_path / "lock_holder.py"
    script.write_text(_LOCK_HOLDER.format(root=str(ROOT)), encoding="utf-8")
    holder = subprocess.Popen([PYTHON, str(script), str(path), str(seconds)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=ENV)
    assert holder.stdout is not None and holder.stdout.readline().strip() == b"locked"
    return holder


async def test_held_registry_lock_does_not_stall_the_event_loop_and_a_cancelled_save_never_writes(tmp_path):
    path = tmp_path / "catalogs.yaml"
    holder = _hold_lock(tmp_path, path, 3.0)
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    try:
        with replaying("describe_2sxps"):
            async with _client() as client:
                start = time.monotonic()
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(1.0):
                        await vizier.register_table("IX/58/2sxps", path=path, client=client)
                elapsed = time.monotonic() - start
    finally:
        stop.set()
        await beat
        await asyncio.to_thread(holder.communicate, timeout=30)
    assert elapsed < 1.6, elapsed  # the deadline fired while the lock was still held
    assert max(gaps) < 0.5, max(gaps)  # the loop kept running
    await asyncio.sleep(0.3)  # the abandoned worker thread notices the cancellation
    assert vizier.list_registered(path) == {}  # ... and never wrote the registration


def test_register_route_answers_504_while_the_lock_is_held(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    monkeypatch.setattr(vizier, "ROUTE_DEADLINE_SECONDS", 1.0)
    app = _app(tmp_path)
    app.state.vizier_registry_path = str(path)
    holder = _hold_lock(tmp_path, path, 4.0)
    try:
        with replaying("describe_2sxps"):
            start = time.monotonic()
            response = TestClient(app).post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps"})
            elapsed = time.monotonic() - start
    finally:
        holder.communicate(timeout=30)
    assert response.status_code == 504 and elapsed < 2.5, (response.status_code, elapsed)
    time.sleep(0.3)
    assert vizier.list_registered(path) == {}


def test_registry_yaml_uses_libyaml_when_available():
    if hasattr(yaml, "CSafeLoader"):
        assert vizier._YAML_LOADER is yaml.CSafeLoader and vizier._YAML_DUMPER is yaml.CSafeDumper
    else:
        assert vizier._YAML_LOADER is yaml.SafeLoader


# ---------------------------------------------------------------------------
# Issue 12 (major): epoch lookup overrides were never checked or selected
# ---------------------------------------------------------------------------


async def test_epoch_lookup_override_columns_are_checked_and_selected():
    desc = await replay_describe("IX/58/2sxps")
    with pytest.raises(vizier.VizierInputError, match="NoSuchColumn"):
        vizier.build_definition(desc, overrides={"epoch": {"column": "NoSuchColumn", "values": {"1": 2005.0}}})
    with pytest.raises(vizier.VizierInputError, match="nope"):
        vizier.build_definition(desc, overrides={"epoch": {"column": "recno", "values": {"1": 2005.0},
                                                           "pm_columns": ["nope"], "pm_epoch": 2000.0}})
    lookup = {"column": "recno", "values": {"1": 2005.0, "2": [2004.0, 2006.0]}}
    _, entry = vizier.build_definition(desc, overrides={"epoch": lookup})
    assert '"recno"' in entry["parameters"]["columns"]
    definition = catalog_from_dict("x", entry)
    assert validate_catalog_definition(definition) == []
    assert resolve_epoch({"recno": 1}, definition.epoch) == 2005.0
    assert models.resolve_epoch_range({"recno": 2}, definition) == (2004.0, 2006.0)


@pytest.mark.parametrize("epoch", [
    {"column": "recno", "values": {"1": 3000}},
    {"column": "recno", "values": {"1": "soon"}},
    {"column": "recno", "values": {}},
    {"column": "recno", "values": {"1": [2006, 2004]}},
    {"column": "recno", "values": {"1": 2005}, "default_range": 2005},
    {"column": "recno", "values": {"1": 2005}, "pm_columns": "pmRA"},
    {"span_columns": ["MJD0"]},
    {"span_columns": ["MJD0", "MJD1"], "format": "unix"},
])
def test_bad_epoch_mappings_are_rejected(epoch):
    with pytest.raises(vizier.VizierInputError):
        vizier.validate_overrides({"epoch": epoch})


async def test_explicit_empty_pos_error_disables_the_detected_error():
    desc = await replay_describe("IX/58/2sxps")
    for value in ({}, None):
        _, entry = vizier.build_definition(desc, overrides={"pos_error": value})
        assert entry["pos_error"] == {}
    _, entry = vizier.build_definition(desc)
    assert entry["pos_error"]["kind"] == "radius90"


# ---------------------------------------------------------------------------
# Issue 13 (major): 4XMM SC_POSERR is radial (see also test_vizier.py 4XMM test)
# ---------------------------------------------------------------------------


def test_xmm_radial_convention_by_column_or_catalogue():
    spec = {"columns": ["ePos"], "units": ["arcsec"], "kind": "sigma"}
    used = [{"column": "ePos", "description": "Mean error on position (SC_POSERR)"}]
    out, notes = vizier.apply_error_conventions(spec, [], vizier.CatalogInfo(catalog_id="X/1"), "X/1/t", None, used)
    assert out["kind"] == "radial" and "SC_POSERR" in notes[-1]
    title = vizier.CatalogInfo(catalog_id="IX/65", title="XMM-Newton Serendipitous Source Catalogue 4XMM-DR11")
    used = [{"column": "ePos", "description": "Position error"}]
    assert vizier.apply_error_conventions(spec, [], title, "IX/65/xmm4d11s", None, used)[0]["kind"] == "radial"
    other = vizier.CatalogInfo(catalog_id="X/2", title="Some optical catalogue")
    assert vizier.apply_error_conventions(spec, [], other, "X/2/t", None, used)[0]["kind"] == "sigma"


# ---------------------------------------------------------------------------
# Issue 14 (minor): search failed when the description-match metadata call failed
# ---------------------------------------------------------------------------


async def test_description_match_metadata_outage_degrades_to_a_warning(no_backoff):
    replay = replay_handler("search_hipparcos")

    async def handler(request: httpx.Request) -> httpx.Response:
        if "-source=" in str(request.url):
            return await status(503, "maintenance")
        return await replay(request)

    client, _ = scripted_client(handler)
    async with client:
        result = await vizier.search_catalogs("Hipparcos", max_catalogs=20, client=client)
    assert [c.catalog_id for c in result.catalogs] == ["I/239"]
    assert "I/239/hip_main" in [t.table_id for t in result.tables]
    assert any("description matches unavailable" in w and "left out" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Issue 15 (minor): RegTAP enrichment used the 90 s budget and was retried every time
# ---------------------------------------------------------------------------


async def test_regtap_enrichment_has_a_short_budget_and_a_negative_cache(no_backoff):
    seen: list[dict] = []

    async def regtap_timeout(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions.get("timeout") or {}))
        raise httpx.ReadTimeout("timed out", request=request)

    client, transport = scripted_client(replay_handler("describe_nvss", overrides={"regtap": regtap_timeout}))
    async with client:
        first = await vizier.describe_table("VIII/65/nvss", client=client)
        assert not first.complete and first.catalog.bibcode is None
        assert any("RegTAP citation/waveband lookup unavailable" in a for a in first.assumptions)
        second = await vizier.describe_table("VIII/65/nvss", client=client)
        assert any("lookup skipped: it failed" in a for a in second.assumptions)
    assert len(seen) == 1  # the failure is remembered: no second wait
    assert seen[0]["read"] == vizier.REGTAP_TIMEOUT_SECONDS < vizier.DEFAULT_TIMEOUT_SECONDS
    other = [r for r in transport.requests if kind_of(r) != "regtap"]
    assert all(r.extensions["timeout"]["read"] == vizier.DEFAULT_TIMEOUT_SECONDS for r in other)
    vizier.clear_describe_cache()  # forgets the failure too
    assert vizier._regtap_recent_failure("VIII/65") is None


async def test_registry_search_uses_the_short_regtap_budget():
    with replaying("search_vlass_registry") as router:
        async with _client() as client:
            await vizier.search_catalogs("VLASS", include_registry=True, client=client)
        regtap = [c.request for c in router.calls if "reg.g-vo.org" in str(c.request.url)]
    assert regtap and all(r.extensions["timeout"]["read"] == vizier.REGTAP_TIMEOUT_SECONDS for r in regtap)


# ---------------------------------------------------------------------------
# Issue 16 (minor): a non-mapping 'source' crashed /registered, 'vizier list' and saves
# ---------------------------------------------------------------------------


def test_free_text_source_is_tolerated(tmp_path, capsys):
    path = tmp_path / "catalogs.yaml"
    entry = {**_entry(), "source": "hand-added from the VLASS paper"}
    path.write_text(yaml.safe_dump({"extends": "embedded", "catalogs": {"my_vlass": entry}}), encoding="utf-8")
    response = TestClient(_app(tmp_path)).get("/api/v1/vizier/registered")
    assert response.status_code == 200 and response.json()["catalogs"][0]["table"] is None
    assert vizier.main(["list", "--registry-path", str(path)]) == 0
    assert "my_vlass" in capsys.readouterr().out
    with pytest.raises(vizier.VizierConflictError, match="already registered"):
        vizier.save_definition("my_vlass", _entry(), path=path)
    vizier.save_definition("my_vlass", _entry(), path=path, replace=True)
    assert vizier._source_of("text") == {} and vizier._source_of({"source": {"table": "t"}}) == {"table": "t"}


# ---------------------------------------------------------------------------
# Issue 17 (minor): max_tables unvalidated; 429 treated as a permanent rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-3, 0, 1001, 2.5, "10", True])
async def test_max_tables_is_validated_before_any_request(bad):
    async def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client, transport = scripted_client(fail)
    async with client:
        with pytest.raises(vizier.VizierInputError, match="max_tables"):
            await vizier.search_catalogs("Gaia DR3", max_tables=bad, client=client)
    assert transport.requests == []


@pytest.mark.parametrize("code", [429, 408])
async def test_rate_limiting_is_transient_and_retried_with_retry_after(code, monkeypatch):
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(vizier.asyncio, "sleep", fake_sleep)

    async def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, text="slow down", headers={"Retry-After": "3"})

    client, transport = scripted_client(limited)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError) as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert info.value.transient and len(transport.requests) == vizier.HTTP_ATTEMPTS
    assert sleeps == [3.0] * (vizier.HTTP_ATTEMPTS - 1)
    assert ("rate limited" in str(info.value)) == (code == 429)


async def test_rate_limited_then_success(no_backoff):
    calls = {"n": 0}
    replay = replay_handler("describe_2sxps")

    async def once_limited(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0"})
        return await replay(request)

    client, _ = scripted_client(once_limited)
    async with client:
        desc = await vizier.describe("IX/58/2sxps", client=client)
    assert desc.table_id == "IX/58/2sxps"


def test_retry_after_parsing(monkeypatch):
    assert vizier._retry_after_seconds("7") == 7.0
    assert vizier._retry_after_seconds(None) is None and vizier._retry_after_seconds("soon") is None
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25.0 < vizier._retry_after_seconds(when) <= 30.0
    assert vizier._retry_after_seconds(format_datetime(datetime(2000, 1, 1, tzinfo=UTC), usegmt=True)) == 0.0


# ---------------------------------------------------------------------------
# Issue 18 (minor): surviving mutants -- untested claimed fixes
# ---------------------------------------------------------------------------


def test_unregister_takes_the_registry_lock(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    holder = _hold_lock(tmp_path, path, 3.0)
    try:
        monkeypatch.setattr(vizier, "REGISTRY_LOCK_TIMEOUT_SECONDS", 0.3)
        with pytest.raises(vizier.VizierConflictError, match="locked"):
            vizier.unregister("mine", path=path)
    finally:
        holder.communicate(timeout=30)
    assert "mine" in vizier.list_registered(path)
    assert vizier.unregister("mine", path=path)


async def test_connection_errors_are_retried_and_transient(no_backoff):
    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, transport = scripted_client(refused)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="unavailable") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert info.value.transient and len(transport.requests) == vizier.HTTP_ATTEMPTS


async def test_description_matches_beyond_max_catalogs_set_truncated():
    asu = next(e for e in _exchanges("search_hipparcos") if "-words=" in e.url)
    _warnings, _total, asu_truncated = vizier._asu_messages(ElementTree.fromstring(asu.content))
    assert not asu_truncated  # ASU itself answered one catalogue (I/239) ...
    with replaying("search_hipparcos"):
        async with _client() as client:
            result = await vizier.search_catalogs("Hipparcos", max_catalogs=20, client=client)
    assert len(result.catalogs) == 20 and result.truncated  # ... the description matches overflowed


def test_save_keeps_unknown_top_level_keys(tmp_path):
    path = tmp_path / "catalogs.yaml"
    path.write_text(yaml.safe_dump({"extends": "embedded", "owner": "survey team", "notes": {"v": 2},
                                    "catalogs": {"old": _entry()}}), encoding="utf-8")
    vizier.save_definition("new", _entry(), path=path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["owner"] == "survey team" and document["notes"] == {"v": 2}
    assert set(vizier.list_registered(path)) == {"old", "new"}
    assert vizier.read_user_registry(path)["owner"] == "survey team"


def test_describe_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(vizier, "_DESCRIBE_CACHE_MAX", 8)
    for i in range(20):
        vizier._cache_put([f"any:X/{i}"], vizier.CatalogInfo(catalog_id=f"X/{i}"))
        assert len(vizier._DESCRIBE_CACHE) <= 8
    assert vizier._cache_get("any:X/19") is not None and vizier._cache_get("any:X/0") is None


@pytest.mark.parametrize("code", [400, 403, 404])
async def test_tap_4xx_without_query_status_is_a_rejection(code):
    async def rejected(request: httpx.Request) -> httpx.Response:
        return await status(code, "Forbidden or bad request")

    client, transport = scripted_client(rejected)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match=f"rejected the query \\(HTTP {code}\\)") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert not info.value.transient and len(transport.requests) == 1
