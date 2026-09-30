"""Regression tests for the third adversarial review of vizier.py (offline).

Each section names the reported defect. Upstream behaviour is replayed from recordings in
tests/fixtures/vizier (recorded live with ``python tests/test_vizier.py record <case>``) or
produced by a scripted httpx transport for failure modes; registry-file races use real
subprocesses. Truth values: SIMBAD (HZ 43 A, d02 Pup, HD 37501, Barnard's star; ICRS J2000,
retrieved 2026-09-29) and the published catalogue documentation quoted in each test.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_vizier import (
    DESCRIBE_FOR_TABLE,
    _app,
    _client,
    _crossmatch_case,
    _match,
    replay_describe,
    replaying,
)
from test_vizier_regressions import ScriptedTransport, replay_handler, scripted_client, status

import vizier
from models import (
    DEFAULT_CATALOGS,
    CatalogRegistry,
    RegistryError,
    catalog_from_dict,
    compute_positional_error,
    validate_catalog_definition,
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


def col(name: str, unit: str | None, ucd: str | None, description: str = "", datatype: str = "DOUBLE",
        principal: bool = False) -> vizier.ColumnInfo:
    return vizier.ColumnInfo(name, unit, ucd, datatype, description, principal)


def _entry() -> dict:
    return json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))


RA = col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension (J2000)")
DE = col("DEJ2000", "deg", "pos.eq.dec;meta.main", "Declination (J2000)")


# ---------------------------------------------------------------------------
# Issue 1 (critical): position-angle errors (deg) registered as the circular positional error
# ---------------------------------------------------------------------------


async def test_galex_ais_uses_the_nuv_position_error_not_the_position_angle_error():
    desc = await replay_describe("II/335/galex_ais")
    # Nerr: stat.error;pos, arcsec, 'Position error of the source in the NUV image (nuv_poserr)'.
    assert desc.pos_error == {"columns": ["Nerr"], "units": ["arcsec"], "kind": "sigma"}
    assert [c["column"] for c in desc.pos_error_columns] == ["Nerr"]
    assert not any("e_nPA" in a and "taken as" in a for a in desc.assumptions)


@pytest.mark.parametrize("table", ["II/371/des_dr2", "II/357/des_dr1"])
async def test_des_position_angle_uncertainties_are_not_positional_errors(table):
    desc = await replay_describe(table)
    # DES e_PA: 'Uncertainty in position angle from isophotal model (ERRTHETA_IMAGE)', deg; the
    # other errors are in pixels or magnitudes: there is no positional-error column.
    assert desc.pos_error == {} and desc.pos_error_columns == []
    assert any(a.startswith("pos_error: e_PA (the error of PA (pos.posAng))") for a in desc.assumptions)
    assert "pos_error: no positional-error column found; matches use separation only." in desc.assumptions
    _, entry = vizier.build_definition(desc)
    assert entry["pos_error"] == {}


async def test_crossmatch_hz43_in_galex_gets_a_sub_arcsecond_error_and_a_real_confidence(tmp_path):
    registration, result, record = await _crossmatch_case("crossmatch_galex_hz43", tmp_path)
    assert registration.entry["pos_error"]["columns"] == ["Nerr"]
    source = result["sources"][0]
    assert source.positional_error_arcsec == pytest.approx(source.data["Nerr"]) == pytest.approx(0.48)
    match = _match(record, source.source_id)
    # Before: sigma = 88.73 deg (e_nPA) -> 319428" and confidence 0.0.
    assert match["confidence"] > 0.5


def test_circular_error_ranking_and_plausibility():
    explicit = col("PosErr", "arcsec", "stat.error;pos", "[0.1/5] Positional uncertainty")
    described = col("e_Pos", "arcsec", "stat.error", "Position error (1 sigma)")
    spec, _used, _notes = vizier.detect_pos_error([RA, DE, described, explicit], RA, DE)
    assert spec["columns"] == ["PosErr"]  # an explicit stat.error;pos column beats a description-only match
    negative = col("e_theta", "deg", "stat.error", "[-90/] Position error angle")
    wide = col("PosErr90", "deg", "stat.error;pos", "[0/90] Position error (deg)")
    spec, _used, notes = vizier.detect_pos_error([RA, DE, negative, wide], RA, DE)
    assert spec == {} and any("not used as a positional error" in n for n in notes)
    # Each guard on its own: position-angle wording (no PA column), the e_<stem> of a pos.posAng
    # column (whatever the wording), and a value range no position error can take.
    wording = col("e_PA", "deg", "stat.error", "Uncertainty in position angle from isophotal model")
    stem = [col("nPA", "deg", "pos.posAng", "Position angle in NUV"), col("e_nPA", "deg", "stat.error", "Position error")]
    ranged = col("e_Pos", "deg", "stat.error", "[-1/1] Position error")
    for extra in ([wording], stem, [ranged]):
        spec, _used, _notes = vizier.detect_pos_error([RA, DE, *extra], RA, DE)
        assert spec == {}, extra
    # A genuine error circle in degrees (gamma-ray / GRB catalogues) stays usable.
    grb = col("ErrRad", "deg", "stat.error;pos", "[0.01/18] Error circle radius (1 sigma)")
    spec, _used, _notes = vizier.detect_pos_error([RA, DE, grb], RA, DE)
    assert spec["columns"] == ["ErrRad"] and spec["units"] == ["deg"]


# ---------------------------------------------------------------------------
# Issue 2 (critical): a J2000 main position rejected for VizieR's COOSYS (FK4 B1996)
# ---------------------------------------------------------------------------


async def test_ngc2451_keeps_its_j2000_main_position_despite_the_fk4_coosys():
    desc = await replay_describe("J/AJ/122/1486/ccd")
    # RA1996: 'Right ascension (mean epoch=1996.6, equinox J2000.0)'; VizieR's COOSYS says eq_FK4 B1996.
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RA1996", "DE1996", "Seq")
    assert desc.frame.startswith("J2000 (RA1996 description")
    assert desc.epoch == 1996.6
    assert any("COOSYS says eq_FK4 equinox B1996" in a and "the description is trusted" in a
               for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    assert entry["parameters"]["ra_field"] == '"RA1996"' and entry["epoch"] == 1996.6
    assert '"_RA.icrs" AS "_RA.icrs"' not in entry["parameters"]["columns"]


async def test_crossmatch_d02_pup_finds_its_own_row_within_half_an_arcsecond(tmp_path):
    _registration, result, record = await _crossmatch_case("crossmatch_ngc2451_d02pup", tmp_path)
    source = result["sources"][0]
    # Seq 217300242 (V = 5.66) at RA1996/DE1996 = 114.93265, -38.13936; before the fix the 5"
    # cone was empty and the row sat 106.6" away at VizieR's _RA.icrs.
    assert source.source_id == "217300242"
    assert (source.ra, source.dec) == (pytest.approx(114.93265), pytest.approx(-38.13936))
    assert _match(record, "217300242")["separation_arcsec"] < 0.5


def test_stated_j2000_ignores_epochs_and_old_equinoxes():
    assert vizier._stated_j2000(col("RA1996", "deg", "pos.eq.ra", "Right ascension (mean epoch=1996.6, "
                                                                     "equinox J2000.0)")) == "equinox J2000.0"
    assert vizier._stated_j2000(col("RAB1950", "deg", "pos.eq.ra", "Right ascension (B1950) at Ep=J2000")) is None
    assert vizier._stated_j2000(col("RAdeg", "deg", "pos.eq.ra", "Right ascension at Ep=J2000")) is None
    # A B1950 position whose COOSYS says FK4 still gives way to VizieR's computed ICRS columns.
    cols = [col("RAB1950", "deg", "pos.eq.ra;meta.main", "Right ascension (B1950)"),
            col("DEB1950", "deg", "pos.eq.dec;meta.main", "Declination (B1950)"),
            col("_RA.icrs", "deg", "pos.eq.ra", "Right ascension (ICRS) (computed by VizieR)"),
            col("_DE.icrs", "deg", "pos.eq.dec", "Declination (ICRS) (computed by VizieR)")]
    asu = {"coosys": {"B": {"system": "eq_FK4", "equinox": "B1950"}},
           "fields": {"RAB1950": {"ref": "B"}, "DEB1950": {"ref": "B"}}}
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, cols, asu)
    assert desc.ra_column == "_RA.icrs"


# ---------------------------------------------------------------------------
# Issue 3 (critical): nearest-counterpart columns mapped to the source's redshift/class
# ---------------------------------------------------------------------------


async def test_2rxs_counterpart_columns_are_not_canonical_fields():
    desc = await replay_describe("J/A+A/588/A103/cat2rxs")
    assert "redshift" not in desc.field_map and "object_type" not in desc.field_map
    assert "spectral_type" not in desc.field_map
    assert any(a.startswith("redshift: zVV10") and "'VV10' cross-identification" in a for a in desc.assumptions)
    assert any(a.startswith("object_type: TypeVV10") for a in desc.assumptions)


async def test_crossmatch_hd37501_gets_no_quasar_redshift(tmp_path):
    registration, result, record = await _crossmatch_case("crossmatch_2rxs_hd37501", tmp_path)
    source = result["sources"][0]
    assert source.source_id == "2RXS J053457.2-611023"
    # dVV10 = 300": the nearest Veron-Cetty & Veron object (PKS 0534-611, z = 1.997) is 5' away.
    assert source.data.get("dVV10") == 300
    assert "redshift" not in source.metadata["physical"] and "object_type" not in source.metadata["physical"]
    member = record.crossmatch_groups[0]["members"][0]
    assert member["position_at_epoch"]["propagation"] != "extragalactic"
    assert "field_map" not in registration.entry["parameters"]


def test_counterpart_detection_keeps_the_sources_own_columns():
    own_z = col("z", None, "src.redshift", "Redshift")
    counterpart_z = col("zQSO", None, "src.redshift", "Redshift of the QSO")
    distance = col("dQSO", "arcsec", "pos.angDistance", "Distance to the QSO")
    fm, notes = vizier.detect_field_map_notes([own_z, counterpart_z, distance])
    assert fm["redshift"] == "z" and any(n.startswith("redshift: zQSO") for n in notes)
    nearest = col("Type", None, "src.class", "Type of the nearest SIMBAD object", datatype="CHAR(8)")
    fm, notes = vizier.detect_field_map_notes([nearest])
    assert "object_type" not in fm and "'nearest'" in notes[0]
    assert vizier._name_tails("zVV10") == {"VV10"} and vizier._name_tails("SpTypeBSC") == {"TypeBSC", "BSC"}


# ---------------------------------------------------------------------------
# Issue 4 (major): Gaia DR1/DR2 TAP names differ from the ASU FIELD names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("table", "epoch"), [("I/345/gaia2", 2015.5), ("I/337/tgas", 2015.0)])
async def test_gaia_dr2_and_tgas_register_with_the_coosys_of_the_renamed_fields(table, epoch):
    router = replaying(DESCRIBE_FOR_TABLE[table])
    with router:
        async with _client() as client:
            desc = await vizier.describe_table(table, client=client)
            assert desc.registrable and desc.complete and desc.problems == []
            assert (desc.ra_column, desc.dec_column) == ("ra", "dec") and desc.frame == "ICRS"
            assert desc.epoch == epoch and "COOSYS" in (desc.epoch_source or "")
            assert any("ra = ASU FIELD RA_ICRS" in a for a in desc.assumptions)
            calls = len(router.calls)
            await vizier.describe_table(table, client=client)
            assert len(router.calls) == calls  # complete -> cached, no repeated queries
    name, entry = vizier.build_definition(desc)
    assert entry["epoch"] == epoch and entry["parameters"]["ra_field"] == '"ra"'
    assert validate_catalog_definition(catalog_from_dict(name, entry)) == []


def test_asu_fields_are_aligned_by_ucd_only_when_unambiguous():
    columns = [col("ra", "deg", "pos.eq.ra;meta.main"), col("dec", "deg", "pos.eq.dec;meta.main"),
               col("e1", "mas", "stat.error"), col("e2", "mas", "stat.error")]
    fields = {"RA_ICRS": {"ucd": "pos.eq.ra;meta.main", "unit": "deg", "ref": "H"},
              "DE_ICRS": {"ucd": "pos.eq.dec;meta.main", "unit": "deg", "ref": "H"},
              "E1": {"ucd": "stat.error", "unit": "mas"}, "E2": {"ucd": "stat.error", "unit": "mas"}}
    aligned, matched = vizier._align_asu_fields(columns, fields)
    assert matched == {"ra": "RA_ICRS", "dec": "DE_ICRS"} and aligned["ra"]["ref"] == "H"
    assert "e1" not in aligned  # two candidates share the UCD: not guessed


async def test_missing_ra_field_is_a_permanent_metadata_mismatch_not_retry_later():
    async def fields_without_ra(request: httpx.Request) -> httpx.Response:
        body = ('<?xml version="1.0"?><VOTABLE version="1.4"><RESOURCE><TABLE>'
                '<FIELD name="TYC1" datatype="short"/></TABLE></RESOURCE></VOTABLE>')
        return await status(200, body, "text/xml")

    client, _ = scripted_client(replay_handler("describe_tycho2", overrides={"asu_table": fields_without_ra}))
    async with client:
        desc = await vizier.describe_table("I/259/tyc2", client=client)
    problem = next(p for p in desc.problems if "has no FIELD for RA(ICRS)" in p)
    assert problem.startswith("VizieR metadata mismatch") and "retry later" not in problem
    assert desc.upstream_problems == []
    with pytest.raises(vizier.VizierRegistrationError, match="metadata mismatch"):
        vizier.build_definition(desc)


# ---------------------------------------------------------------------------
# Issue 5 (major): SkyMapper DR4 COOSYS epoch 2000 trusted over per-row observation epochs
# ---------------------------------------------------------------------------


async def test_smss_dr4_uses_the_per_row_mean_observation_epoch():
    desc = await replay_describe("II/379/smssdr4")
    assert desc.epoch == "EpMean" and desc.epoch_format == "mjd"
    assert desc.epoch_range is not None and desc.epoch_range[0] < 2014.3 and desc.epoch_range[1] > 2021.7
    assert any("COOSYS epoch 2000.000 states 2000, but the table has no proper motions" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    assert entry["epoch"] == "EpMean" and entry["parameters"]["epoch_range"] == desc.epoch_range


async def test_crossmatch_barnards_star_in_smss_dr4(tmp_path):
    _registration, result, record = await _crossmatch_case("crossmatch_smss_barnard", tmp_path)
    ids = {s.source_id for s in result["sources"]}
    # SMSS object 1353376183 (EpMean 58342.3 = 2018.6) is Barnard's star, 194" from its J2000 position.
    assert "1353376183" in ids
    assert _match(record, "1353376183")["separation_arcsec"] < 1.5


def test_stated_epochs_are_kept_when_proper_motions_exist_or_the_epoch_is_deliberate():
    ep = col("EpMean", "d", "time.epoch;stat.mean", "Mean MJD epoch of the observations")
    pm = [col("pmRA", "mas/yr", "pos.pm;pos.eq.ra"), col("pmDE", "mas/yr", "pos.pm;pos.eq.dec")]
    j2000 = {"system": "ICRS", "epoch": "2000.000"}
    assert vizier.detect_epoch([RA, DE, ep], RA, j2000, [], dec=DE)["epoch"] == "EpMean"
    assert vizier.detect_epoch([RA, DE, ep, *pm], RA, j2000, [], dec=DE)["epoch"] == 2000.0  # propagated with PMs
    gaia = vizier.detect_epoch([RA, DE, ep], RA, {"system": "ICRS", "epoch": "J2016.0"}, [], dec=DE)
    assert gaia["epoch"] == 2016.0  # a deliberate non-J2000 epoch is authoritative


# ---------------------------------------------------------------------------
# Issue 6 (major): an 'epoch' override to a per-row column never got an epoch_range
# ---------------------------------------------------------------------------


async def test_epoch_override_to_a_per_row_column_gets_its_span(tmp_path):
    path = tmp_path / "catalogs.yaml"
    with replaying("register_erass1_epoch_override"):
        async with _client() as client:
            registration = await vizier.register_table("J/A+A/682/A34/erass1-m", path=path, client=client,
                                                       overrides={"epoch": "b_MJD"})
    entry = registration.entry
    assert entry["epoch"] == "b_MJD" and entry["epoch_format"] == "mjd"
    # eRASS1-Main: the first eRASS1 scan ran from 2019 Dec 12 to 2020 Jun 11.
    assert entry["parameters"]["epoch_range"] == [2019.944, 2020.44]
    assert any(a.startswith("epoch: 'b_MJD' given by the caller") for a in registration.assumptions)
    assert any("MIN/MAX over all" in a for a in registration.assumptions)
    # The detected span's notes ('positions measured between b_MJD and B_MJD ...') are replaced.
    assert not any(a.startswith("epoch: positions measured between") for a in registration.assumptions)
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))["catalogs"][registration.name]
    assert stored["source"]["epoch_source"] == "override"


async def test_build_definition_notes_an_overridden_per_row_epoch_without_a_span():
    desc = await replay_describe("J/A+A/682/A34/erass1-m")
    notes: list[str] = []
    _, entry = vizier.build_definition(desc, overrides={"epoch": "b_MJD"}, notes=notes)
    assert "epoch_range" not in entry["parameters"]
    assert any("without an epoch_range" in n for n in notes)
    _, entry = vizier.build_definition(desc, overrides={"epoch": "b_MJD", "epoch_range": [2019.9, 2020.5]})
    assert entry["parameters"]["epoch_range"] == [2019.9, 2020.5]


# ---------------------------------------------------------------------------
# Issue 7 (minor): SDSS galaxy subclasses and integer class codes as canonical fields
# ---------------------------------------------------------------------------


async def test_sdss_subclass_and_integer_class_codes_are_not_canonical():
    desc = await replay_describe("V/147/sdss12")
    assert "spectral_type" not in desc.field_map
    assert desc.field_map.get("object_type") != "class"
    assert any(a.startswith("object_type: class") and "integer class codes" in a for a in desc.assumptions)
    assert any(a.startswith("spectral_type: subCl") and "galaxies/QSOs" in a for a in desc.assumptions)


def test_stellar_subclasses_are_kept_in_stellar_tables():
    sub = col("SubClass", None, "src.spType", "Spectral subclass (MK)", datatype="CHAR(8)")
    fm, _notes = vizier.detect_field_map_notes([RA, DE, sub], wavelength="optical")
    assert fm["spectral_type"] == "SubClass"
    code = col("cl", None, "src.class", "Class (1=star, 2=galaxy)", datatype="SMALLINT")
    name = col("otype", None, "src.class", "Object type", datatype="CHAR(8)")
    fm, _notes = vizier.detect_field_map_notes([code, name])
    assert fm["object_type"] == "otype"


# ---------------------------------------------------------------------------
# Issue 8 (minor): 'total' read as a radial error (ROSAT PosErr)
# ---------------------------------------------------------------------------


async def test_rosat_total_positional_error_is_a_sigma_like_the_core_entry():
    desc = await replay_describe("IX/10A/1rxs")
    assert desc.pos_error == {"columns": ["PosErr"], "units": ["arcsec"], "kind": "sigma"}
    assert desc.pos_error["kind"] == DEFAULT_CATALOGS["rosat_bsc"]["pos_error"]["kind"]
    assert compute_positional_error({"PosErr": 10}, desc.pos_error)[0] == pytest.approx(10.0)


@pytest.mark.parametrize(("text", "kind"), [
    ("Total positional error (including 6\" systematic error)", "sigma"),
    ("Statistical and systematic errors added in quadrature", "sigma"),
    ("Radial position error", "radial"),
    ("Position error sqrt(RA_ERR^2 + DEC_ERR^2)", "radial"),
])
def test_radial_wording(text, kind):
    err = col("PosErr", "arcsec", "stat.error;pos", text)
    spec, _used, _notes = vizier.detect_pos_error([RA, DE, err], RA, DE)
    assert spec["kind"] == kind


# ---------------------------------------------------------------------------
# Issue 9 (major): POST /register did not reach the dataset engine's registry
# ---------------------------------------------------------------------------


def test_register_route_attaches_to_the_dataset_engine_registry(tmp_path, monkeypatch):
    import api

    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(tmp_path / "core.yaml"))
    monkeypatch.setenv("DATASET_STORAGE_PATH", str(tmp_path / "datasets"))

    async def no_processing(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(api, "process_dataset_async", no_processing)
    monkeypatch.setattr(api, "submit_to_redis", lambda *a, **k: False)
    routes = list(api.app.router.routes)
    if not any(getattr(r, "path", "").startswith("/api/v1/vizier") for r in routes):
        api.app.include_router(vizier.router)  # integration note 1 (not mounted by api.py yet)
    with TestClient(api.app) as client:
        api.app.state.vizier_registry_path = str(tmp_path / "catalogs.yaml")
        try:
            with replaying("describe_2sxps"):
                response = client.post("/api/v1/vizier/register",
                                       json={"table_id": "IX/58/2sxps", "name": "swift2sxps"})
            assert response.status_code == 200, response.text
            assert response.json()["attached"] is True
            state = api.app.state
            assert "swift2sxps" in state.engine.registry.catalogs
            assert "swift2sxps" in state.registry.catalogs and "swift2sxps" in state.service.registry.catalogs
            listed = client.get("/api/v1/vizier/registered").json()
            assert listed["catalogs"][0]["active"] is True
            response = client.post("/api/v1/datasets/create", json={
                "name": "xray_3c273", "profile": "full", "catalogs": ["swift2sxps"], "count_threshold": 1,
                "targets": [{"ra": 187.2779154, "dec": 2.0523883}]})
            assert response.status_code == 202, response.text  # before: 422 'Unknown catalog: swift2sxps'
        finally:
            del api.app.state.vizier_registry_path
            api.app.router.routes[:] = routes


def test_live_registries_are_deduplicated():
    from fastapi import FastAPI
    from starlette.requests import Request

    shared = CatalogRegistry()
    app = FastAPI()
    app.state.registry = shared
    app.state.service = type("S", (), {"registry": shared})()
    app.state.engine = type("E", (), {"registry": CatalogRegistry()})()
    request = Request({"type": "http", "app": app})
    found = vizier._live_registries(request)
    assert len(found) == 2 and found[0] is shared and found[1] is app.state.engine.registry


# ---------------------------------------------------------------------------
# Issue 10 (major): hand edits discarded by load_registry() and by saves
# ---------------------------------------------------------------------------


HAND_WRITTEN = """\
# Observatory registry -- curated by hand
# (do not lose these notes)
extends: embedded
defaults: &radio_defaults
  max_rows: 50
catalogs:
  my_vlass:   # our VLASS entry
{entry}
"""


def _hand_written(path: Path) -> str:
    entry = textwrap.indent(yaml.safe_dump(_entry(), sort_keys=False), "    ")
    text = HAND_WRITTEN.format(entry=entry.rstrip("\n"))
    path.write_text(text, encoding="utf-8")
    return text


def test_load_registry_never_rewrites_a_hand_written_extends_file(tmp_path, caplog):
    path = tmp_path / "catalogs.yaml"
    text = _hand_written(path)
    with caplog.at_level("WARNING", logger="astrosearch.vizier"):
        merged = vizier.load_registry(path)
    assert path.read_text(encoding="utf-8") == text  # a read does not rewrite (comments kept)
    assert {"my_vlass", "gaia_dr3"} <= set(merged.catalogs)
    assert any("not written by vizier" in r.getMessage() for r in caplog.records)
    status_ = vizier.registry_status(path)
    assert status_["written_by_vizier"] is False and status_["builtin_copies_outdated"] and status_["comments"]
    with pytest.raises(vizier.VizierRegistrationError, match="comments that rewriting would drop"):
        vizier.save_definition("another", _entry(), path=path)
    assert path.read_text(encoding="utf-8") == text


def test_hand_written_extends_file_without_comments_is_only_rewritten_by_a_save(tmp_path):
    path = tmp_path / "catalogs.yaml"
    path.write_text(yaml.safe_dump({"extends": "embedded", "catalogs": {"mine": _entry()}}), encoding="utf-8")
    before = path.read_bytes()
    vizier.load_registry(path)
    assert path.read_bytes() == before
    vizier.save_definition("another", _entry(), path=path)  # an explicit registration rewrites it
    assert set(CatalogRegistry(path).catalogs) == set(DEFAULT_CATALOGS) | {"mine", "another"}


def _edit_copies(path: Path) -> None:
    """An operator's edit of the built-in copies in a vizier-written file (header kept)."""
    text = path.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    document = yaml.safe_load(text)
    document["catalogs"]["gaia_dr3"]["enabled"] = False
    document["catalogs"]["simbad"]["timeout_seconds"] = 5
    path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("with_hashes", [True, False])
def test_edited_builtin_copies_are_honoured_and_kept(tmp_path, capsys, caplog, with_hashes):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    if not with_hashes:  # a file written before per-copy hashes (fingerprint only)
        text = path.read_text(encoding="utf-8")
        document = yaml.safe_load(text)
        del document["embedded_copy_hashes"]
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        path.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    _edit_copies(path)
    edited_text = path.read_text(encoding="utf-8")
    core = CatalogRegistry(path)
    assert core.get("gaia_dr3").enabled is False and core.get("simbad").timeout_seconds == 5
    merged = vizier.load_registry(path)
    assert merged.get("gaia_dr3").enabled is False and merged.get("simbad").timeout_seconds == 5
    assert path.read_text(encoding="utf-8") == edited_text  # nothing rewritten by the load
    status_ = vizier.registry_status(path)
    assert status_["edited_builtin_copies"] == ["gaia_dr3", "simbad"] and not status_["builtin_copies_outdated"]
    assert vizier.main(["list", "--registry-path", str(path)]) == 0
    assert "'gaia_dr3' was edited by hand" in capsys.readouterr().out
    with caplog.at_level("WARNING", logger="astrosearch.vizier"):
        vizier.save_definition("another", _entry(), path=path)
    assert any("edited by hand" in r.getMessage() for r in caplog.records)
    core = CatalogRegistry(path)  # the save kept the edits
    assert core.get("gaia_dr3").enabled is False and core.get("simbad").timeout_seconds == 5
    merged = vizier.load_registry(path)
    assert merged.get("gaia_dr3").enabled is False and merged.get("simbad").timeout_seconds == 5
    assert {"mine", "another", "gaia_dr3", "simbad"} <= set(vizier.list_registered(path))


def test_an_edited_copy_survives_an_upgrade_while_unedited_copies_are_refreshed(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    _edit_copies(path)
    corrected = json.loads(json.dumps(DEFAULT_CATALOGS["nvss"]))
    corrected["timeout_seconds"] = 77.0
    monkeypatch.setitem(DEFAULT_CATALOGS, "nvss", corrected)
    status_ = vizier.registry_status(path)
    assert status_["builtin_copies_outdated"] and status_["edited_builtin_copies"] == ["gaia_dr3", "simbad"]
    vizier.load_registry(path)  # a vizier-written file: the outdated copies are refreshed
    core = CatalogRegistry(path)
    assert core.get("nvss").timeout_seconds == 77.0
    assert core.get("gaia_dr3").enabled is False and core.get("simbad").timeout_seconds == 5


def test_comments_added_to_a_vizier_file_stop_rewrites(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    text = path.read_text(encoding="utf-8").replace("catalogs:\n", "catalogs:  # reviewed 2026-09\n", 1)
    path.write_text(text, encoding="utf-8")
    monkeypatch.setitem(DEFAULT_CATALOGS, "new_builtin", json.loads(json.dumps(DEFAULT_CATALOGS["nvss"])))
    vizier.load_registry(path)
    assert path.read_text(encoding="utf-8") == text
    with pytest.raises(vizier.VizierRegistrationError, match="reviewed 2026-09"):
        vizier.unregister("mine", path=path)


# ---------------------------------------------------------------------------
# Issue 11 (major): non-UTF-8 registry files crashed the CLI and answered a bare 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["utf-16", "latin-1"])
def test_non_utf8_registry_is_a_clear_registry_error(tmp_path, capsys, encoding):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("mine", _entry(), path=path)
    text = path.read_text(encoding="utf-8").replace("VLASS", "VLASS Montréal", 1)
    path.write_bytes(text.encode(encoding))
    with pytest.raises(RegistryError, match="is not UTF-8 text"):
        vizier.load_registry(path)
    assert vizier.main(["list", "--registry-path", str(path)]) == 1
    assert "is not UTF-8 text" in capsys.readouterr().out
    with replaying("describe_2sxps"):
        assert vizier.main(["add", "IX/58/2sxps", "--name", "x2", "--registry-path", str(path)]) == 1
    assert "is not UTF-8 text" in capsys.readouterr().out
    app = _app(tmp_path)
    app.state.vizier_registry_path = str(path)
    client = TestClient(app)
    response = client.get("/api/v1/vizier/registered")
    assert response.status_code == 500 and "is not UTF-8 text" in response.json()["detail"]
    with replaying("describe_2sxps"):
        response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "x2"})
    assert response.status_code == 500 and "is not UTF-8 text" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Issue 12 (major): an ASU outage during POST /register answered 422 instead of 502
# ---------------------------------------------------------------------------


def test_register_route_answers_502_with_retry_after_on_an_asu_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(vizier, "RETRY_BACKOFF_SECONDS", 0.0)

    async def down(request: httpx.Request) -> httpx.Response:
        return await status(503, "maintenance")

    app = _app(tmp_path)
    app.state.client = httpx.AsyncClient(transport=ScriptedTransport(
        replay_handler("describe_2sxps", overrides={"asu_table": down})))
    response = TestClient(app).post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps"})
    assert response.status_code == 502, response.text
    assert "retry later" in response.json()["detail"] and response.headers["retry-after"] == "30"
    assert not (tmp_path / "catalogs.yaml").exists()


# ---------------------------------------------------------------------------
# Issue 14 (major): a pos_error override without units disabled every positional error
# ---------------------------------------------------------------------------


async def test_pos_error_override_without_units_gets_the_tap_schema_units():
    desc = await replay_describe("IX/58/2sxps")
    notes: list[str] = []
    _, entry = vizier.build_definition(desc, overrides={"pos_error": {"columns": ["Err90"], "kind": "radius90"}},
                                       notes=notes)
    assert entry["pos_error"]["units"] == ["arcsec"]
    assert any("units taken from TAP_SCHEMA: Err90 'arcsec'" in n for n in notes)
    # Err90 = 2.0" (90% Rayleigh radius) -> sigma = 2.0 / 2.146 = 0.932"
    assert compute_positional_error({"Err90": 2.0}, entry["pos_error"])[0] == pytest.approx(0.932, abs=1e-3)
    with pytest.raises(vizier.VizierInputError, match="has no angular unit"):
        vizier.build_definition(desc, overrides={"pos_error": {"columns": ["NObs"], "kind": "sigma"}})


async def test_register_table_reports_the_overridden_pos_error(tmp_path):
    with replaying("describe_2sxps"):
        async with _client() as client:
            registration = await vizier.register_table(
                "IX/58/2sxps", path=tmp_path / "c.yaml", client=client,
                overrides={"pos_error": {"columns": ["Err90"], "kind": "radius90"}})
    assert registration.entry["pos_error"]["units"] == ["arcsec"]
    assert any(a.startswith("pos_error: {'columns': ['Err90']") and "given by the caller" in a
               for a in registration.assumptions)
    assert any(a == "pos_error: override without units; units taken from TAP_SCHEMA: Err90 'arcsec'."
               for a in registration.assumptions)
    # The detected error's notes are replaced by the override's (GALEX: Nerr -> Ferr).
    with replaying("describe_galex_ais"):
        async with _client() as client:
            registration = await vizier.register_table(
                "II/335/galex_ais", path=tmp_path / "c.yaml", client=client,
                overrides={"pos_error": {"columns": ["Ferr"], "kind": "sigma"}})
    notes = [a for a in registration.assumptions if a.startswith("pos_error:")]
    assert registration.entry["pos_error"] == {"columns": ["Ferr"], "kind": "sigma", "units": ["arcsec"]}
    assert not any(a.startswith("pos_error: Nerr (") for a in notes) and len(notes) == 2


# ---------------------------------------------------------------------------
# Issue 15 (minor): a route-deadline 504 could still persist the registration
# ---------------------------------------------------------------------------


def test_cancel_is_checked_before_every_replace_attempt(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("first_one", _entry(), path=path)
    before = path.read_bytes()
    cancel = threading.Event()
    real_replace = os.replace
    attempts = []

    def busy_replace(src, dst):
        attempts.append(dst)
        cancel.set()  # the caller gives up while the file is held open by another process
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(vizier.os, "replace", busy_replace)
    with pytest.raises(vizier._RegistryCancelled):
        vizier.save_definition("raced", _entry(), path=path, cancel=cancel)
    monkeypatch.setattr(vizier.os, "replace", real_replace)
    assert len(attempts) == 1 and path.read_bytes() == before
    assert not list(tmp_path.glob(".catalogs.*.yaml"))


async def test_registration_committed_before_the_cancel_is_completed(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    written = threading.Event()
    real_save = vizier.save_definition

    def slow_after_commit(*args, **kwargs):
        result = real_save(*args, **kwargs)
        written.set()
        time.sleep(0.5)  # still in the worker thread when the caller is cancelled
        return result

    monkeypatch.setattr(vizier, "save_definition", slow_after_commit)
    live = CatalogRegistry()
    with replaying("describe_2sxps"):
        async with _client() as client:
            task = asyncio.ensure_future(vizier.register_table("IX/58/2sxps", name="late", path=path, client=client,
                                                               registry=live))
            await asyncio.to_thread(written.wait, 30)
            task.cancel()
            registration = await task
    assert registration.attached and "late" in live.catalogs and "late" in vizier.list_registered(path)


async def test_registration_cancelled_before_the_commit_writes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    started = threading.Event()
    real_save = vizier.save_definition

    def wait_for_cancel(*args, cancel=None, **kwargs):
        started.set()
        assert cancel is not None and cancel.wait(30)
        return real_save(*args, cancel=cancel, **kwargs)

    monkeypatch.setattr(vizier, "save_definition", wait_for_cancel)
    live = CatalogRegistry()
    with replaying("describe_2sxps"):
        async with _client() as client:
            task = asyncio.ensure_future(vizier.register_table("IX/58/2sxps", name="early", path=path, client=client,
                                                               registry=live))
            await asyncio.to_thread(started.wait, 30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert not path.exists() and "early" not in live.catalogs


# Holds the file open until released through stdin (or until the test process goes away): no fixed hold
# time that a loaded machine could outlast before the route deadline fires.
_HOLD_OPEN = textwrap.dedent("""
    import sys
    handle = open(sys.argv[1], encoding="utf-8")
    print("open", flush=True)
    sys.stdin.readline()
""")


@pytest.mark.skipif(not WINDOWS, reason="only Windows refuses to replace a file another process holds open")
def test_route_deadline_during_a_blocked_replace_persists_nothing(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    vizier.save_definition("first_one", _entry(), path=path)
    monkeypatch.setattr(vizier, "ROUTE_DEADLINE_SECONDS", 1.0)
    # The refused replace is retried far beyond the route deadline, so however late the deadline is
    # delivered under load, it fires while the replace is still being refused (never a 409 instead).
    monkeypatch.setattr(vizier, "REGISTRY_REPLACE_RETRY_SECONDS", 120.0)
    script = tmp_path / "hold_open.py"
    script.write_text(_HOLD_OPEN, encoding="utf-8")
    app = _app(tmp_path)
    app.state.vizier_registry_path = str(path)
    client = TestClient(app)
    with replaying("describe_2sxps"):
        client.get("/api/v1/vizier/catalog/IX/58/2sxps")  # describe cached: only the save is slow below
        holder = subprocess.Popen([PYTHON, str(script), str(path)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
        try:
            assert holder.stdout is not None and holder.stdout.readline().strip() == b"open"
            response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "raced"})
        finally:
            holder.communicate(input=b"\n", timeout=30)  # release the file only after the route answered
        assert response.status_code == 504, response.text
        # The file is free now: a save that ignored the abandonment would replace it within one poll.
        time.sleep(max(0.5, 10 * vizier._POLL_SECONDS))
        assert "raced" not in vizier.list_registered(path)  # abandoned, not persisted behind the 504
        response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "raced"})
    assert response.status_code == 200, response.text  # the retry succeeds (no 409)


# ---------------------------------------------------------------------------
# Issue 16 (minor): epoch overrides were not type-checked
# ---------------------------------------------------------------------------


async def test_epoch_overrides_must_name_numeric_distinct_columns():
    desc = await replay_describe("IX/58/2sxps")
    with pytest.raises(vizier.VizierInputError, match="'IAUName' .* is not numeric"):
        vizier.build_definition(desc, overrides={"epoch": "IAUName"})
    with pytest.raises(vizier.VizierInputError, match="two distinct columns"):
        vizier.build_definition(desc, overrides={"epoch": {"span_columns": ["recno", "recno"], "format": "mjd"}})
    with pytest.raises(vizier.VizierInputError, match="epoch.pm_columns column 'IAUName'"):
        vizier.build_definition(desc, overrides={"epoch": {"column": "recno", "values": {"1": 2010.0},
                                                           "pm_columns": ["IAUName", "IAUName"]}})
    _, entry = vizier.build_definition(desc, overrides={"epoch": 2010.0})
    assert entry["epoch"] == 2010.0
