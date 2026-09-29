"""Offline tests for vizier.py: replays of real VizieR / TAPVizieR / RegTAP responses.

Record (or refresh) the fixtures with live network access:

    .venv/Scripts/python.exe tests/test_vizier.py record            # every case
    .venv/Scripts/python.exe tests/test_vizier.py record describe_2sxps crossmatch_2sxps

Each case is stored as ``tests/fixtures/vizier/<case>/exchanges.json`` (+ ``.body`` files) in the
format of tests/fixture_io.py and replayed strictly: a request that differs from the recorded
one (ADQL text, ASU parameters, ...) raises FixtureMismatch instead of being served.

Crossmatch targets are SIMBAD ICRS positions (J2000): 3C 273, Markarian 421 and NGC 3818.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fixture_io import FIXTURES, FixtureMismatch, load_exchanges, redact, replay_side_effect
from helpers import offline_client

import vizier
from crossmatch import AdvancedQuery, CrossmatchService
from models import (
    DEFAULT_CATALOGS,
    K90_2D,
    K95_2D,
    CatalogRegistry,
    ColumnMeta,
    RegistryError,
    Target,
    catalog_from_dict,
    compute_positional_error,
    normalize_source_record,
    validate_catalog_definition,
    validate_target,
)
from providers import CacheManager, TapProvider, provider_map

FIXTURE_DIR = "vizier"

# SIMBAD basic.ra/dec (ICRS, J2000), retrieved 2026-09-28.
TARGET_3C273 = (187.27791594049, 2.05238823055)
TARGET_MRK421 = (166.11380868146, 38.20883291552)  # SIMBAD main id 'Z 184-50' (Mrk 421)
TARGET_NGC3818 = (175.4889849546, -6.155683098159999)

CROSSMATCH_CASES: dict[str, tuple[str, tuple[float, float], float]] = {
    # case: (VizieR table, SIMBAD position, radius arcsec)
    "crossmatch_2sxps": ("IX/58/2sxps", TARGET_3C273, 5.0),
    "crossmatch_vlass": ("J/ApJS/255/30/comp", TARGET_MRK421, 3.0),
    "crossmatch_2dfgrs": ("VII/250/2dfgrs", TARGET_NGC3818, 5.0),
}
DESCRIBE_CASES: dict[str, str] = {
    "describe_2sxps": "IX/58/2sxps",
    "describe_vlass": "J/ApJS/255/30/comp",
    "describe_2dfgrs": "VII/250/2dfgrs",
    "describe_csc21": "IX/70/csc21mas",
    "describe_4xmm": "IX/69/xmm4d13s",
    "describe_gaiadr3": "I/355/gaiadr3",
    "describe_catalog_vii250": "VII/250",
    "describe_missing": "NOPE/123",
}
SEARCH_CASES: dict[str, dict[str, Any]] = {
    "search_gaia_dr3": {"query": "Gaia DR3"},
    "search_redshift_ucd": {"query": "galaxy redshift", "ucd": "src.redshift", "wavelength": "optical",
                            "max_catalogs": 10},
    "search_quasar_radio": {"query": "quasar", "wavelength": "radio", "max_catalogs": 20},
    "search_vlass_registry": {"query": "VLASS", "include_registry": True},
}
DESCRIBE_FOR_TABLE = {table: case for case, table in DESCRIBE_CASES.items()}


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def _exchanges(*cases: str):
    items = []
    for case in cases:
        folder = FIXTURES / FIXTURE_DIR / case
        assert folder.exists(), f"missing fixture folder {folder}; record with 'python tests/test_vizier.py record'"
        items += load_exchanges(f"{FIXTURE_DIR}/{case}")
    return items


def replaying(*cases: str) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route().mock(side_effect=replay_side_effect(_exchanges(*cases)))
    return router


def _client() -> httpx.AsyncClient:
    return offline_client()  # shared SSL context (creating one costs ~0.6 s on Windows)


async def replay_describe(table: str) -> vizier.TableDescription | vizier.CatalogInfo:
    with replaying(DESCRIBE_FOR_TABLE[table]):
        async with _client() as client:
            return await vizier.describe(table, client=client)


# ---------------------------------------------------------------------------
# Discovery (replayed)
# ---------------------------------------------------------------------------


async def test_search_gaia_dr3_finds_main_source_table():
    with replaying("search_gaia_dr3"):
        async with _client() as client:
            result = await vizier.search_catalogs("Gaia DR3", client=client)
    ids = [t.table_id for t in result.tables]
    assert ids[0] == "I/355/gaiadr3"
    top = result.tables[0]
    assert top.nrows == 1_811_709_771  # Gaia DR3 gaia_source row count
    assert top.catalog_id == "I/355" and "Gaia DR3" in (top.catalog_title or "")
    assert top.wavelengths == ["optical"]
    catalogs = {c.catalog_id: c for c in result.catalogs}
    assert {"I/355", "I/356", "I/357", "I/358", "I/350"} <= set(catalogs)
    assert catalogs["I/355"].bibcode == "2022yCat.1355....0G"
    assert not result.warnings  # ASU's "can't find table or catalogue: <word>" noise is dropped


async def test_search_wavelength_and_ucd_filters_are_strict():
    with replaying("search_redshift_ucd"):
        async with _client() as client:
            result = await vizier.search_catalogs("galaxy redshift", ucd="src.redshift", wavelength="optical",
                                                  max_catalogs=10, client=client)
    assert result.wavelength == "optical" and result.ucd == "src.redshift"
    assert result.tables, "expected optical tables with a src.redshift column"
    for hit in result.tables:
        assert "optical" in hit.wavelengths
        assert hit.matching_columns and all(vizier.ucd_matches("src.redshift", c["ucd"]) for c in hit.matching_columns)
    for info in result.catalogs:
        assert all(t.matching_columns for t in info.tables)
    assert result.truncated and result.total_catalog_matches and result.total_catalog_matches > 10


async def test_search_radio_quasars_keeps_only_radio_tagged_catalogues():
    with replaying("search_quasar_radio"):
        async with _client() as client:
            result = await vizier.search_catalogs("quasar", wavelength="radio", max_catalogs=20, client=client)
    assert result.catalogs and all("Radio" in c.wavelengths for c in result.catalogs)
    assert all("Radio" in t.wavelengths for t in result.tables)


async def test_search_with_ivoa_registry():
    with replaying("search_vlass_registry"):
        async with _client() as client:
            result = await vizier.search_catalogs("VLASS", include_registry=True, client=client)
    assert [t.table_id for t in result.tables] == ["J/ApJS/255/30/comp", "J/ApJS/255/30/subtile"]
    assert result.registry is not None
    by_id = {r.ivoid: r for r in result.registry}
    vlass = by_id["ivo://cds.vizier/j/apjs/255/30"]
    assert vlass.vizier_catalog == "J/ApJS/255/30"
    assert "radio" in vlass.wavebands
    assert any(url.startswith("https://vizier.cds.unistra.fr/viz-bin/conesearch/J/ApJS/255/30")
               for url in vlass.services["conesearch"])


def test_search_rejects_empty_and_bad_inputs():
    with pytest.raises(vizier.VizierInputError):
        asyncio.run(vizier.search_catalogs(""))
    with pytest.raises(vizier.VizierInputError):
        asyncio.run(vizier.search_catalogs("x", ucd="src.redshift' OR 1=1 --"))
    with pytest.raises(vizier.VizierInputError):
        asyncio.run(vizier.search_catalogs("x", wavelength="neutrino"))


def test_registry_adql_matches_regtap_schema():
    adql = vizier._registry_adql("Gaia DR3", standards=["ivo://ivoa.net/std/conesearch"], waveband="optical", limit=5)
    assert "FROM rr.resource AS r NATURAL JOIN rr.capability AS c NATURAL JOIN rr.interface AS i" in adql
    assert "ivo_hasword(r.res_title, 'Gaia DR3')" in adql
    assert "ivo_hashlist_has(r.waveband, 'optical')" in adql
    assert "i.intf_role = 'std'" in adql
    quoted = vizier._registry_adql("O'Brien", standards=["ivo://ivoa.net/std/tap"], waveband=None, limit=1)
    assert "'O''Brien'" in quoted


# ---------------------------------------------------------------------------
# Describe (replayed)
# ---------------------------------------------------------------------------


async def test_describe_2sxps_xray_radius90():
    desc = await replay_describe("IX/58/2sxps")
    assert isinstance(desc, vizier.TableDescription)
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RAJ2000", "DEJ2000", "2SXPS")
    assert desc.nrows == 206_335
    assert desc.catalog.bibcode == "2020ApJS..247...54E"
    assert desc.wavelength == "xray"
    # 'Position uncertainty, 90% confidence, radial, assumed to be Rayleigh-distributed'
    assert desc.pos_error == {"columns": ["Err90"], "units": ["arcsec"], "kind": "radius90"}
    assert desc.epoch is None and any(a.startswith("epoch:") for a in desc.assumptions)
    assert desc.citation.startswith("Evans et al. 2020, ApJS 247, 54 (2020ApJS..247...54E); VizieR IX/58")
    assert desc.registrable


async def test_describe_vlass_components_per_axis_sigma_in_degrees():
    desc = await replay_describe("J/ApJS/255/30/comp")
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RAJ2000", "DEJ2000", "CompName")
    assert desc.pos_error == {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": ["deg", "deg"], "kind": "sigma"}
    assert not any(a.startswith("pos_error") for a in desc.assumptions)  # '1-sigma uncertainty' is explicit
    assert desc.wavelength == "radio"
    assert desc.catalog.doi == "10.26093/cds/vizier.22550030"
    assert desc.nrows == 3_381_277


async def test_describe_2dfgrs_unitless_positions_and_redshift():
    desc = await replay_describe("VII/250/2dfgrs")
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RAJ2000", "DEJ2000", "Name")
    # TAP_SCHEMA gives no unit (description even says 'hours'); the largest RA proves degrees.
    assert "exceeds 24, confirming decimal degrees" in (desc.position_unit_check or "")
    assert desc.field_map == {"redshift": "z"}  # heliocentric 'z' (src.redshift), not z.obs / z.abs / z.em
    assert desc.pos_error == {}
    assert desc.wavelength == "optical"


async def test_describe_csc21_error_ellipse_and_single_table_catalogue():
    desc = await replay_describe("IX/70/csc21mas")
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RAICRS", "DEICRS", "2CXO")
    assert desc.pos_error["columns"] == ["r0", "r1", "PA"]
    assert desc.pos_error["kind"] == "ellipse95"
    assert any("ellipse95axis" in a for a in desc.assumptions)  # the per-axis reading is flagged
    assert desc.catalog.title and "Chandra Source Catalog" in desc.catalog.title
    assert desc.wavelength == "high-energy"
    assert desc.citation.startswith("Evans et al. 2024")


async def test_describe_4xmm_epoch_span_from_first_last_observation():
    desc = await replay_describe("IX/69/xmm4d13s")
    assert desc.epoch == {"span_columns": ["MJD0", "MJD1"], "point_max_years": 0.1, "format": "mjd"}
    # '[51575/59912] Date of first observation (MJD)' .. '[51575/59913] Date of last observation'
    lo, hi = desc.epoch_range
    assert lo == pytest.approx(2000.0 + (51575 - 51544.5) / 365.25, abs=1e-3)
    assert hi == pytest.approx(2000.0 + (59913 - 51544.5) / 365.25, abs=1e-3)
    assert desc.pos_error["columns"] == ["ePos"] and desc.pos_error["kind"] == "sigma"
    assert any("ePos" in a and "no confidence level" in a for a in desc.assumptions)


async def test_describe_gaia_dr3_epoch_from_coosys_and_astrometry():
    desc = await replay_describe("I/355/gaiadr3")
    assert (desc.ra_column, desc.dec_column, desc.id_column) == ("RA_ICRS", "DE_ICRS", "Source")
    assert desc.epoch == 2016.0 and "COOSYS" in (desc.epoch_source or "")
    assert desc.field_map == {"pmra": "pmRA", "pmdec": "pmDE", "parallax": "Plx"}
    assert desc.pos_error == {"columns": ["e_RA_ICRS", "e_DE_ICRS"], "units": ["mas", "mas"], "kind": "sigma"}
    assert desc.frame and desc.frame.startswith("ICRS")


async def test_describe_catalogue_lists_its_tables():
    info = await replay_describe("VII/250")
    assert isinstance(info, vizier.CatalogInfo)
    assert [t.table_id for t in info.tables] == ["VII/250/2dfgrs", "VII/250/parent", "VII/250/photo", "VII/250/spectra"]
    assert info.bibcode == "2001MNRAS.328.1039C"


async def test_describe_missing_table_is_not_found():
    with pytest.raises(vizier.VizierNotFoundError):
        await replay_describe("NOPE/123")


async def test_describe_table_refuses_multi_table_catalogue():
    with replaying("describe_catalog_vii250"):
        async with _client() as client:
            with pytest.raises(vizier.VizierInputError, match="VII/250/2dfgrs"):
                await vizier.describe_table("VII/250", client=client)


async def test_replay_is_strict_about_requests():
    with replaying("describe_2sxps"):
        async with _client() as client:
            with pytest.raises(FixtureMismatch):
                await client.post(vizier.VIZIER_TAP_URL, data={"REQUEST": "doQuery", "QUERY": "SELECT 1"})


# ---------------------------------------------------------------------------
# Registration: definitions (pure) and persistence
# ---------------------------------------------------------------------------


async def test_build_definition_is_valid_for_tap_provider():
    desc = await replay_describe("IX/58/2sxps")
    name, entry = vizier.build_definition(desc)
    assert name == "vizier_ix_58_2sxps"
    assert entry["provider"] == "tap" and entry["endpoint"] == vizier.VIZIER_TAP_URL
    assert entry["table"] == '"IX/58/2sxps"'
    params = entry["parameters"]
    assert params["columns"][:4] == ['"2SXPS"', '"RAJ2000"', '"DEJ2000"', '"Err90"']
    assert params["id_field"] == '"2SXPS"' and params["ra_field"] == '"RAJ2000"'
    assert entry["profiles"] == ["full", "vizier", "xray", "high-energy"]
    definition = catalog_from_dict(name, entry)
    assert validate_catalog_definition(definition) == []
    target = validate_target(*TARGET_3C273)
    async with offline_client() as client:
        adql = TapProvider(client, cache=CacheManager(None)).build_adql(definition, target, 5.0)
    assert adql.startswith('SELECT TOP 201 "2SXPS", "RAJ2000", "DEJ2000", "Err90"')
    assert 'FROM "IX/58/2sxps" WHERE 1 = CONTAINS(POINT(\'ICRS\', "RAJ2000", "DEJ2000")' in adql


async def test_build_definition_2dfgrs_column_units_and_guarded_columns():
    desc = await replay_describe("VII/250/2dfgrs")
    _, entry = vizier.build_definition(desc)
    params = entry["parameters"]
    assert params["column_units"] == {"RAJ2000": "deg", "DEJ2000": "deg"}
    assert params["field_map"] == {"redshift": "z"}
    selected = {c.strip('"') for c in params["columns"]}
    # Other redshift columns (z.obs, z.abs, z.em) would feed the canonical 'redshift' via UCD.
    assert "z" in selected and not {"z.obs", "z.abs", "z.em"} & selected
    assert entry["pos_error"] == {}


async def test_build_definition_gaia_fixed_epoch_and_proper_motions():
    desc = await replay_describe("I/355/gaiadr3")
    _, entry = vizier.build_definition(desc, name="gaia_vizier")
    assert entry["epoch"] == 2016.0
    assert "epoch_range" not in entry["parameters"]
    assert entry["parameters"]["field_map"] == {"pmra": "pmRA", "pmdec": "pmDE", "parallax": "Plx"}
    assert len(entry["parameters"]["columns"]) <= vizier.MAX_SELECTED_COLUMNS
    definition = catalog_from_dict("gaia_vizier", entry)
    assert validate_catalog_definition(definition) == []
    from models import catalog_has_proper_motions

    assert catalog_has_proper_motions(definition)


async def test_build_definition_4xmm_span_epoch_validates():
    desc = await replay_describe("IX/69/xmm4d13s")
    _, entry = vizier.build_definition(desc)
    assert entry["epoch"]["span_columns"] == ["MJD0", "MJD1"]
    assert entry["parameters"]["epoch_range"] == desc.epoch_range
    assert validate_catalog_definition(catalog_from_dict("x", entry)) == []


async def test_build_definition_overrides():
    desc = await replay_describe("IX/70/csc21mas")
    # CXC: CSC err_ellipse_r0/r1 are per-axis 95% intervals -> override the 2-D default.
    _, entry = vizier.build_definition(desc, overrides={"pos_error": {**desc.pos_error, "kind": "ellipse95axis"},
                                                        "wavelength": "xray", "systematic_arcsec": 0.1})
    assert entry["pos_error"]["kind"] == "ellipse95axis" and entry["pos_error"]["systematic_arcsec"] == 0.1
    assert entry["wavelength"] == "xray" and "xray" in entry["profiles"]
    with pytest.raises(vizier.VizierInputError, match="Unknown override"):
        vizier.build_definition(desc, overrides={"endpoint": "https://evil.example/tap"})
    with pytest.raises(vizier.VizierInputError, match="Invalid catalog name"):
        vizier.build_definition(desc, name="Bad Name!")
    with pytest.raises(vizier.VizierInputError, match="not in"):
        vizier.build_definition(desc, overrides={"extra_columns": ["nosuchcolumn"]})


def test_build_definition_refuses_table_without_positions():
    catalog = vizier.CatalogInfo(catalog_id="X/1")
    cols = [vizier.ColumnInfo("GLON", "deg", "pos.galactic.lon", "DOUBLE", "Galactic longitude"),
            vizier.ColumnInfo("GLAT", "deg", "pos.galactic.lat", "DOUBLE", "Galactic latitude")]
    desc = vizier.analyse_table("X/1/t", catalog, {"description": "t", "nrows": 3}, cols)
    assert not desc.registrable
    with pytest.raises(vizier.VizierRegistrationError, match="no numeric equatorial position"):
        vizier.build_definition(desc)


async def test_register_persists_merges_and_attaches(tmp_path):
    path = tmp_path / "catalogs.yaml"
    live = CatalogRegistry()
    with replaying("describe_2sxps"):
        async with _client() as client:
            reg = await vizier.register_table("IX/58/2sxps", path=path, registry=live, client=client)
    assert reg.name == "vizier_ix_58_2sxps" and reg.attached and not reg.replaced
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["extends"] == "embedded"
    stored = document["catalogs"]["vizier_ix_58_2sxps"]
    assert stored["source"]["table"] == "IX/58/2sxps" and stored["source"]["bibcode"] == "2020ApJS..247...54E"
    assert "vizier_ix_58_2sxps" in live.catalogs  # attached to a plain models.CatalogRegistry
    merged = vizier.load_registry(path)
    assert set(DEFAULT_CATALOGS) <= set(merged.catalogs)
    assert merged.get("vizier_ix_58_2sxps").table == '"IX/58/2sxps"'
    assert not [p for p in merged.validate() if p.startswith("vizier_ix_58_2sxps")]
    # Registering the same name again needs replace=True.
    with replaying("describe_2sxps"):
        async with _client() as client:
            with pytest.raises(vizier.VizierRegistrationError, match="already registered"):
                await vizier.register_table("IX/58/2sxps", path=path, client=client)
            again = await vizier.register_table("IX/58/2sxps", path=path, replace=True, client=client)
    assert again.replaced
    assert vizier.unregister("vizier_ix_58_2sxps", path=path)
    assert not vizier.unregister("vizier_ix_58_2sxps", path=path)
    assert vizier.list_registered(path) == {}


def test_user_registry_semantics(tmp_path):
    missing = tmp_path / "none.yaml"
    assert set(vizier.load_registry(missing).catalogs) == set(DEFAULT_CATALOGS)
    full = tmp_path / "full.yaml"
    entry = dict(DEFAULT_CATALOGS["vlass"])
    full.write_text(yaml.safe_dump({"catalogs": {"only_this": json.loads(json.dumps(entry))}}), encoding="utf-8")
    # Without 'extends: embedded' the file is a complete registry (models.CatalogRegistry semantics).
    assert set(vizier.load_registry(full).catalogs) == {"only_this"}
    assert set(CatalogRegistry(full).catalogs) == {"only_this"}
    with pytest.raises(vizier.VizierRegistrationError, match="built-in"):
        vizier.save_definition("gaia_dr3", entry, path=tmp_path / "x.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("catalogs: [1, 2", encoding="utf-8")
    with pytest.raises(RegistryError):
        vizier.list_registered(bad)


def test_user_registry_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("CATALOG_REGISTRY_PATH", raising=False)
    assert vizier.user_registry_path() == Path.home() / ".astrosearch" / "catalogs.yaml"
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(tmp_path / "env.yaml"))
    assert vizier.user_registry_path() == tmp_path / "env.yaml"
    assert vizier.user_registry_path(tmp_path / "explicit.yaml") == tmp_path / "explicit.yaml"


# ---------------------------------------------------------------------------
# Crossmatch through CrossmatchService (replayed describe + real cone responses)
# ---------------------------------------------------------------------------


async def _crossmatch_case(case: str, tmp_path: Path):
    table, (ra, dec), radius = CROSSMATCH_CASES[case]
    path = tmp_path / "catalogs.yaml"
    with replaying(DESCRIBE_FOR_TABLE[table], case):
        async with _client() as client:
            registration = await vizier.register_table(table, path=path, client=client)
            registry = vizier.load_registry(path)
            service = CrossmatchService(registry, provider_map(client, cache=CacheManager(None)))
            query = AdvancedQuery(target=validate_target(ra, dec), radius_arcsec=radius,
                                  catalogs=[registration.name], min_confidence=0.0)
            record = await service.crossmatch(ra, dec, query=query)
    result = record.catalog_results[registration.name]
    assert result["status"] == "success", result
    return registration, result, record


async def test_crossmatch_2sxps_finds_3c273(tmp_path):
    registration, result, record = await _crossmatch_case("crossmatch_2sxps", tmp_path)
    source = result["sources"][0]
    assert source.source_id == "13814"
    assert source.data["IAUName"] == "2SXPS J122906.6+020308"
    assert source.ra == pytest.approx(187.2779) and source.dec == pytest.approx(2.05239)
    # Err90 = 0.9" (90% Rayleigh radius) -> 1-sigma = 0.9 / sqrt(-2 ln 0.1)
    assert source.data["Err90"] == 0.9
    assert source.positional_error_arcsec == pytest.approx(0.9 / K90_2D, rel=1e-9)
    assert source.metadata["positional_error"]["kind"] == "radius90"
    assert source.metadata["citation"] == registration.entry["citation"]
    match = record.provenance["matches"][0]
    assert match["source_id"] == "13814" and match["separation_arcsec"] < 0.2


async def test_crossmatch_vlass_finds_mrk421(tmp_path):
    _, result, record = await _crossmatch_case("crossmatch_vlass", tmp_path)
    source = result["sources"][0]
    assert source.source_id == "VLASS1QLCIR J110427.33+381231.8"
    assert source.data["Ftot"] == pytest.approx(446.502)  # mJy at 3 GHz
    era, ede = source.data["e_RAJ2000"], source.data["e_DEJ2000"]  # degrees
    expected = math.sqrt(((era * 3600) ** 2 + (ede * 3600) ** 2) / 2)
    assert source.positional_error_arcsec == pytest.approx(expected, rel=1e-9)
    assert source.positional_error_arcsec == pytest.approx(0.0009007, rel=1e-3)
    assert record.provenance["matches"][0]["separation_arcsec"] < 0.5


async def test_crossmatch_2dfgrs_finds_ngc3818_redshift(tmp_path):
    _, result, record = await _crossmatch_case("crossmatch_2dfgrs", tmp_path)
    source = result["sources"][0]
    assert source.source_id == "TGN113Z100"
    assert source.metadata["physical"]["redshift"] == pytest.approx(0.0059)
    # SIMBAD's independent redshift of NGC 3818 is 0.00554 (cz ~ 1660 km/s).
    assert abs(source.metadata["physical"]["redshift"] - 0.00554) < 0.001
    assert source.positional_error_arcsec is None
    assert record.provenance["matches"][0]["separation_arcsec"] == pytest.approx(2.058, abs=0.01)


# ---------------------------------------------------------------------------
# Pure heuristics
# ---------------------------------------------------------------------------


def col(name: str, unit: str | None, ucd: str | None, description: str = "", datatype: str = "DOUBLE",
        principal: bool = False) -> vizier.ColumnInfo:
    return vizier.ColumnInfo(name, unit, ucd, datatype, description, principal)


def test_pos_error_pair_with_stated_per_axis_percent_and_time_seconds():
    cols = [col("RAJ2000", "deg", "pos.eq.ra;meta.main"), col("DEJ2000", "deg", "pos.eq.dec;meta.main"),
            col("e_RAs", "s", "stat.error;pos.eq.ra", "90% confidence error in RA (seconds of time)"),
            col("e_DEs", "arcsec", "stat.error;pos.eq.dec", "90% confidence error in Dec")]
    spec, _used, notes = vizier.detect_pos_error(cols, cols[0], cols[1])
    assert spec["columns"] == ["e_RAs", "e_DEs"] and spec["units"] == ["s_ra", "arcsec"]
    assert spec["divisor"] == pytest.approx(1.644854, abs=1e-6)  # two-sided 90% normal interval
    assert not notes
    row = {"e_RAs": 0.01, "e_DEs": 0.2}
    sigma, _details = compute_positional_error(row, spec, dec_deg=60.0)
    ra_arcsec = 0.01 * 15 * math.cos(math.radians(60.0)) / 1.6448536269514722
    dec_arcsec = 0.2 / 1.6448536269514722
    assert sigma == pytest.approx(math.sqrt((ra_arcsec**2 + dec_arcsec**2) / 2), rel=1e-6)


def test_pos_error_ellipse_2d_and_per_axis():
    base = [col("RA", "deg", "pos.eq.ra;meta.main"), col("DE", "deg", "pos.eq.dec;meta.main")]
    two_d = base + [col("maj", "arcsec", "phys.angSize;pos.errorEllipse", "Semi-major axis of 95% error ellipse"),
                    col("min", "arcsec", "phys.angSize", "Semi-minor axis of 95% error ellipse"),
                    col("pa", "deg", "pos.posAng", "Position angle of the error ellipse")]
    spec, used, _notes = vizier.detect_pos_error(two_d)
    assert spec == {"columns": ["maj", "min", "pa"], "units": ["arcsec", "arcsec", "deg"], "kind": "ellipse95"}
    assert [u["role"] for u in used] == ["major", "minor", "position_angle"]
    sigma, _ = compute_positional_error({"maj": 2.4477, "min": 2.4477, "pa": 0}, spec)
    assert sigma == pytest.approx(2.4477 / K95_2D, rel=1e-4)
    per_axis = base + [col("r0", "arcsec", "pos.errorEllipse", "95% confidence interval along each axis (major)"),
                       col("r1", "arcsec", "pos.errorEllipse", "95% confidence interval along each axis (minor)")]
    spec, _, _ = vizier.detect_pos_error(per_axis)
    assert spec["kind"] == "ellipse95axis"
    deconvolved = base + [col("PAb", "deg", "pos.posAng", "major axis of the ellipse defining the deconvolved source extent")]
    assert vizier.detect_pos_error(deconvolved)[0] == {}


def test_pos_error_circular_variants():
    base = [col("RA", "deg", "pos.eq.ra;meta.main"), col("DE", "deg", "pos.eq.dec;meta.main")]
    spec, _, _ = vizier.detect_pos_error(base + [col("r95", "arcsec", "stat.error;pos.eq", "95% error circle radius")])
    assert spec["kind"] == "radius95"
    spec, _, notes = vizier.detect_pos_error(base + [col("perr", "arcsec", "stat.error", "Total radial position error")])
    assert spec["kind"] == "radial" and notes
    spec, _, _ = vizier.detect_pos_error(base + [col("e", "arcsec", "stat.error;pos", "3-sigma positional uncertainty")])
    assert spec == {"columns": ["e"], "units": ["arcsec"], "kind": "sigma", "divisor": 3.0}
    spec, _, _ = vizier.detect_pos_error(base + [col("e", "arcsec", "stat.error;pos", "Position error, 68% confidence")])
    assert spec["divisor"] == pytest.approx(math.sqrt(-2 * math.log(0.32)), rel=1e-5)
    # A flux error or a unit-less column is never mistaken for a position error.
    spec, _, notes = vizier.detect_pos_error(base + [col("e_F", "mJy", "stat.error", "Error on flux"),
                                                     col("e_pos", None, "stat.error", "Position error")])
    assert spec == {} and any("no positional-error column" in n for n in notes)


def test_epoch_detection_variants():
    ra = col("RA_ICRS", "deg", "pos.eq.ra;meta.main", "Right ascension (ICRS) at Ep=2015.5")
    single = [ra, col("ObsDate", "d", "time.epoch", "Observation date (MJD)")]
    got = vizier.detect_epoch(single, ra, {}, [])
    assert got["epoch"] == "ObsDate" and got["epoch_format"] == "mjd" and got["single_epoch_positions"]
    mean = [ra, col("Epoch", "yr", "time.epoch", "Mean epoch of the observations")]
    got = vizier.detect_epoch(mean, ra, {}, [])
    assert got["epoch_format"] == "jyear" and not got["single_epoch_positions"]
    livetime = [ra, col("TimeA", "s", "time.epoch", "Total livetime")]  # CSC misuse of time.epoch
    got = vizier.detect_epoch(livetime, ra, {}, [])
    assert got["epoch"] == 2015.5 and "Ep=2015.5" in got["source"]
    got = vizier.detect_epoch([ra], ra, {"system": "ICRS", "epoch": "J2016.0"}, [])
    assert got["epoch"] == 2016.0 and "COOSYS" in got["source"]
    plain = col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension (J2000)")
    got = vizier.detect_epoch([plain], plain, {"system": "eq_FK5", "equinox": "J2000"}, ["Positions at mean epoch 1991.25"])
    assert got["epoch"] == 1991.25
    got = vizier.detect_epoch([plain], plain, {"system": "eq_FK5", "equinox": "J2000"}, ["Right ascension (J2000)"])
    assert got["epoch"] is None and got["notes"]


def test_position_selection_skips_b1950_and_char_columns():
    cols = [col("RAB1950", "deg", "pos.eq.ra;meta.main", "Right ascension (B1950)"),
            col("DEB1950", "deg", "pos.eq.dec;meta.main", "Declination (B1950)"),
            col("RAJ2000", "deg", "pos.eq.ra", "Right ascension (J2000, computed by VizieR)"),
            col("DEJ2000", "deg", "pos.eq.dec", "Declination (J2000, computed by VizieR)"),
            col("RAstr", None, "pos.eq.ra", "RA J2000 sexagesimal", datatype="CHAR(11)")]
    desc = vizier.analyse_table("X/1/t", vizier.CatalogInfo(catalog_id="X/1"), {"nrows": 1}, cols)
    assert (desc.ra_column, desc.dec_column) == ("RAJ2000", "DEJ2000")


def test_field_map_prefers_exact_ucd_and_skips_model_redshifts():
    cols = [col("zph", None, "src.redshift.phot", "Photometric redshift"),
            col("zfit", None, "src.redshift", "Redshift of the best fitting APEC model"),
            col("zsp", None, "src.redshift", "Spectroscopic redshift"),
            col("SpT", None, "src.spType", "Spectral type", datatype="CHAR(8)"),
            col("otype", None, "src.class", "Object class", datatype="CHAR(8)")]
    assert vizier.detect_field_map(cols) == {"redshift": "zsp", "object_type": "otype", "spectral_type": "SpT"}


def test_identifiers_names_and_small_helpers():
    assert vizier.validate_vizier_id(" J/A+A/707/A198/lotssdr3 ") == "J/A+A/707/A198/lotssdr3"
    assert vizier.validate_vizier_id('"I/355/gaiadr3"') == "I/355/gaiadr3"
    for bad in ("I", "I/355' OR '1'='1", "../../etc", "I/355; DROP", ""):
        with pytest.raises(vizier.VizierInputError):
            vizier.validate_vizier_id(bad)
    assert vizier.catalog_of("J/ApJS/255/30/comp") == "J/ApJS/255/30"
    assert vizier.default_catalog_name("J/A+A/707/A198/lotssdr3") == "vizier_j_a_a_707_a198_lotssdr3"
    assert vizier.quote_identifier('a"b') == '"a""b"'
    assert vizier.normalize_wavelength("X-ray") == "X-ray" and vizier.normalize_wavelength("xray") == "X-ray"
    assert vizier.normalize_wavelength("infrared") == "IR" and vizier.normalize_wavelength(None) is None
    assert vizier.ucd_matches("src.redshift", "src.redshift.phot")
    assert vizier.ucd_matches("phot.flux*;em.radio", "phot.flux.density;em.radio.750-1500MHz")
    assert not vizier.ucd_matches("src.redshift", "stat.error")
    assert vizier.bibcode_reference("2020A&A...641A.136W") == "A&A 641, A136"
    assert vizier.bibcode_reference("2001MNRAS.328.1039C") == "MNRAS 328, 1039"
    assert vizier.bibcode_reference("2022yCat.1355....0G") == "yCat 1355"
    assert vizier._table_title("Best observations of 2dFGRS ( Colless M., et al.)") == "Best observations of 2dFGRS"


def test_normalization_of_registered_rows_uses_field_map():
    # A registered 2dFGRS row normalizes 'z' (field_map) and not the observed-frame z.obs.
    columns = [ColumnMeta("Name", None, "meta.id;meta.main", "CHAR"), ColumnMeta("RAJ2000", None, "pos.eq.ra;meta.main"),
               ColumnMeta("DEJ2000", None, "pos.eq.dec;meta.main"), ColumnMeta("z.obs", None, "src.redshift;pos.heliocentric"),
               ColumnMeta("z", None, "src.redshift")]
    row = {"Name": "TGN113Z100", "RAJ2000": 175.4893, "DEJ2000": -6.1562, "z.obs": 0.0061, "z": 0.0059}
    values = normalize_source_record(row, columns=columns, field_map={"redshift": "z", "ra": "RAJ2000", "dec": "DEJ2000"})
    assert values["redshift"] == 0.0059


# ---------------------------------------------------------------------------
# Router and CLI
# ---------------------------------------------------------------------------


def _app(tmp_path: Path, registry: CatalogRegistry | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(vizier.router)
    app.state.vizier_registry_path = str(tmp_path / "catalogs.yaml")
    if registry is not None:
        app.state.registry = registry
    return app


def test_router_search_describe_register_registered(tmp_path):
    live = CatalogRegistry()
    client = TestClient(_app(tmp_path, live))
    with replaying("search_gaia_dr3"):
        response = client.get("/api/v1/vizier/search", params={"q": "Gaia DR3"})
    assert response.status_code == 200, response.text
    assert response.json()["tables"][0]["table_id"] == "I/355/gaiadr3"
    with replaying("describe_2sxps"):
        response = client.get("/api/v1/vizier/catalog/IX/58/2sxps")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "table" and body["pos_error"]["kind"] == "radius90" and body["registrable"] is True
    with replaying("describe_catalog_vii250"):
        response = client.get("/api/v1/vizier/catalog/VII/250")
    assert response.json()["kind"] == "catalog" and len(response.json()["tables"]) == 4
    with replaying("describe_2sxps"):
        response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "swift_2sxps"})
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "swift_2sxps" and response.json()["attached"] is True
    assert "swift_2sxps" in live.catalogs
    with replaying("describe_2sxps"):
        response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "name": "swift_2sxps"})
    assert response.status_code == 409
    response = client.get("/api/v1/vizier/registered")
    assert response.status_code == 200
    listed = response.json()
    assert listed["count"] == 1 and listed["catalogs"][0]["table"] == "IX/58/2sxps"
    assert listed["catalogs"][0]["active"] is True


def test_router_errors(tmp_path):
    client = TestClient(_app(tmp_path))
    with replaying("describe_missing"):
        assert client.get("/api/v1/vizier/catalog/NOPE/123").status_code == 404
    assert client.get("/api/v1/vizier/catalog/bad id").status_code == 422
    assert client.get("/api/v1/vizier/search").status_code == 422
    assert client.get("/api/v1/vizier/search", params={"q": "x", "wavelength": "neutrino"}).status_code == 422
    with replaying("describe_catalog_vii250"):
        response = client.post("/api/v1/vizier/register", json={"table_id": "VII/250"})
    assert response.status_code == 422 and "VII/250/2dfgrs" in response.json()["detail"]


def test_router_upstream_outage_is_502(tmp_path):
    client = TestClient(_app(tmp_path))
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(503, text="maintenance"))
        response = client.get("/api/v1/vizier/catalog/IX/58/2sxps")
    assert response.status_code == 502 and "unavailable" in response.json()["detail"]


def _cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    vizier.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_cli_wiring():
    assert _cli(["vizier", "search", "Gaia DR3"]).handler is vizier._cli_search
    assert _cli(["vizier", "describe", "IX/58/2sxps"]).handler is vizier._cli_describe
    args = _cli(["vizier", "add", "IX/58/2sxps", "--name", "x"])
    assert args.handler is vizier._cli_add and args.name == "x"
    assert _cli(["vizier", "list"]).handler is vizier._cli_list
    assert _cli(["vizier"]).handler(argparse.Namespace()) == 2


def test_cli_search_describe_add_list(tmp_path, capsys):
    with replaying("search_gaia_dr3"):
        args = _cli(["vizier", "search", "Gaia DR3"])
        assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert "I/355/gaiadr3" in out.splitlines()[2]
    with replaying("describe_2sxps"):
        args = _cli(["vizier", "describe", "IX/58/2sxps"])
        assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert "Err90" in out and "radius90" in out and "2020ApJS..247...54E" in out
    path = tmp_path / "reg.yaml"
    with replaying("describe_2sxps"):
        args = _cli(["vizier", "add", "IX/58/2sxps", "--registry-path", str(path), "--systematic", "0.2"])
        assert args.handler(args) == 0
    assert "Registered 'vizier_ix_58_2sxps'" in capsys.readouterr().out
    assert vizier.list_registered(path)["vizier_ix_58_2sxps"]["pos_error"]["systematic_arcsec"] == 0.2
    args = _cli(["vizier", "list", "--registry-path", str(path)])
    assert args.handler(args) == 0
    assert "vizier_ix_58_2sxps" in capsys.readouterr().out
    with replaying("describe_missing"):
        args = _cli(["vizier", "describe", "NOPE/123"])
        assert args.handler(args) == 1
    assert "Error" in capsys.readouterr().out


def test_crossmatch_uses_registered_catalog_target_type():
    # Guard: the crossmatch helper builds real Targets (no stub objects).
    assert isinstance(validate_target(*TARGET_3C273), Target)


# ---------------------------------------------------------------------------
# Recording (live network; not collected as tests)
# ---------------------------------------------------------------------------


def _save_case(case: str, log: list[tuple[httpx.Request, httpx.Response]]) -> None:
    folder = FIXTURES / FIXTURE_DIR / case
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("exchanges.*"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (folder / f"exchanges.{idx}.body").write_bytes(redact(response.content))
        exchanges.append({
            "method": request.method,
            "url": str(request.url),
            "request_body": request.content.decode("utf-8", "replace") if request.content else "",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "match": [],
        })
    meta = {"case": case, "exchanges": exchanges}
    (folder / "exchanges.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


async def _record(case: str, action: Callable[[httpx.AsyncClient], Awaitable[Any]],
                  keep: Callable[[httpx.Request], bool] | None = None) -> None:
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        if keep is None or keep(response.request):
            log.append((response.request, response))

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        try:
            await action(client)
        except vizier.VizierNotFoundError:
            pass  # recorded on purpose (describe_missing)
    _save_case(case, log)
    print(f"{case:<28} {len(log)} exchange(s)")


def _is_cone(request: httpx.Request) -> bool:
    return b"CONTAINS" in (request.content or b"")


async def record(cases: list[str] | None = None) -> None:
    import tempfile

    wanted = set(cases or [])

    def selected(name: str) -> bool:
        return not wanted or name in wanted

    for case, kwargs in SEARCH_CASES.items():
        if selected(case):
            await _record(case, lambda c, kw=kwargs: vizier.search_catalogs(kw["query"], client=c,
                                                                            **{k: v for k, v in kw.items() if k != "query"}))
    for case, table in DESCRIBE_CASES.items():
        if selected(case):
            await _record(case, lambda c, t=table: vizier.describe(t, client=c))
    for case, (table, (ra, dec), radius) in CROSSMATCH_CASES.items():
        if not selected(case):
            continue

        async def run(client: httpx.AsyncClient, table=table, ra=ra, dec=dec, radius=radius) -> None:
            path = Path(tempfile.mkdtemp()) / "catalogs.yaml"
            registration = await vizier.register_table(table, path=path, client=client)
            registry = vizier.load_registry(path)
            service = CrossmatchService(registry, provider_map(client, timeout=120.0, cache=CacheManager(None)))
            query = AdvancedQuery(target=validate_target(ra, dec), radius_arcsec=radius,
                                  catalogs=[registration.name], min_confidence=0.0)
            record_ = await service.crossmatch(ra, dec, query=query)
            stats = record_.catalog_results[registration.name]
            print(f"  {registration.name}: {stats['status']} rows={stats['row_count']} {stats.get('message') or ''}")

        await _record(case, run, keep=_is_cone)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "record":
        asyncio.run(record(sys.argv[2:] or None))
    else:
        print(__doc__)
