"""Catalog registry schema and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from models import (
    DEFAULT_CATALOGS,
    POS_ERROR_KINDS,
    CatalogDefinition,
    CatalogRegistry,
    bare_column_name,
    catalog_epoch_range,
    catalog_from_dict,
    validate_catalog_definition,
)
from providers import provider_map


def test_default_registry_is_valid(registry: CatalogRegistry) -> None:
    assert registry.validate() == []


def test_every_registry_provider_is_implemented(registry: CatalogRegistry) -> None:
    implemented = set(provider_map())
    for name, catalog in registry.catalogs.items():
        assert catalog.provider in implemented, name
        fallback = catalog.parameters.get("fallback")
        if fallback:
            assert fallback["provider"] in implemented, name


def test_expected_catalogs_enabled(registry: CatalogRegistry) -> None:
    enabled = registry.enabled_catalogs()
    assert len(enabled) >= 15
    for name in ("gaia_dr3", "simbad", "ned", "exoplanet_archive", "twomass_psc", "allwise", "panstarrs_dr2",
                 "sdss", "first", "nvss", "vlass", "lotss", "rosat", "chandra", "xmm"):
        assert name in enabled


@pytest.mark.parametrize("name", sorted(DEFAULT_CATALOGS))
def test_every_catalog_has_epoch_errors_citation_and_limits(name: str, registry: CatalogRegistry) -> None:
    catalog = registry.get(name)
    # Positions need an epoch: a year, per-row column(s)/lookup, or (epoch None) the span
    # within which the rows were observed at unrecorded times.
    assert catalog.epoch is not None or catalog_epoch_range(catalog) is not None, "positions need an epoch"
    assert catalog.pos_error, "positional error spec required"
    assert catalog.pos_error.get("kind", "sigma") in POS_ERROR_KINDS
    assert catalog.citation and catalog.acknowledgement
    assert 0 < catalog.max_rows <= 100_000
    assert catalog.timeout_seconds and catalog.timeout_seconds >= 30
    assert catalog.coverage


def test_epoch_columns_are_selected(registry: CatalogRegistry) -> None:
    for name, catalog in registry.catalogs.items():
        if isinstance(catalog.epoch, (int, float)):
            assert 1985 <= catalog.epoch <= 2030, name
            assert not catalog.parameters.get("epoch_range"), f"{name}: fixed epoch with an epoch_range"
            continue
        if catalog.epoch is None:
            # Unknown per-row epoch within a declared span (CSC, NVSS, LoTSS, 1RXS).
            lo, hi = catalog_epoch_range(catalog)
            assert 1985 <= lo < hi <= 2030, name
            continue
        if isinstance(catalog.epoch, dict):
            # Per-row span (5XMM time/end_time) or lookup (NED/SIMBAD bibcode), plus the
            # proper-motion columns that decide SIMBAD's J2000 epoch.
            names = list(catalog.epoch.get("span_columns") or [catalog.epoch["column"]])
            names += list(catalog.epoch.get("pm_columns") or [])
        else:
            names = [catalog.epoch] if isinstance(catalog.epoch, str) else catalog.epoch
        # Per-row epochs need a declared span so cones can be epoch-widened.
        assert catalog.parameters.get("epoch_range"), f"{name}: per-row epoch without epoch_range"
        selected = {bare_column_name(c).lower() for c in catalog.parameters.get("columns", [])}
        for column in names:
            assert column.lower() in selected, f"{name}: epoch column {column} not selected"


def test_known_bad_configuration_is_gone(registry: CatalogRegistry) -> None:
    # Bug 8: SIMBAD has no 'err_pos' column.
    for catalog in registry.catalogs.values():
        assert catalog.parameters.get("positional_error_field") != "err_pos"
        assert "err_pos" not in str(catalog.pos_error)
    # Bug 3: HEASARC table names and endpoints.
    tables = {name: registry.get(name).table for name in ("chandra", "xmm", "rosat", "first", "nvss")}
    assert tables == {"chandra": "csc", "xmm": "xmmssc", "rosat": "rass2rxs", "first": "first", "nvss": "nvss"}
    for name in tables:
        assert registry.get(name).endpoint == "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
    assert all(cat.provider != "heasarc_xamin" for cat in registry.enabled_catalogs().values())
    # HEASARC hosts no VLASS / LoTSS: they come from VizieR with quoted table names.
    assert registry.get("vlass").table == '"J/ApJ/914/42/table5"'
    assert registry.get("lotss").table == '"J/A+A/707/A198/lotssdr3"'
    assert registry.get("vizier_2mass_reference").table == '"II/246/out"'
    assert '"2MASS"' in registry.get("vizier_2mass_reference").parameters["columns"]
    # Bug 4: Exoplanet Archive uses the one-row-per-planet table.
    assert registry.get("exoplanet_archive").table == "pscomppars"
    assert registry.get("exoplanet_archive").epoch == 2015.5


def test_positional_error_units_per_catalog(registry: CatalogRegistry) -> None:
    assert registry.get("gaia_dr3").pos_error["units"] == "mas"
    assert registry.get("simbad").pos_error["units"] == "mas"
    assert registry.get("ned").pos_error["kind"] == "ellipse" and registry.get("ned").pos_error["divisor"] == 2.5
    assert registry.get("chandra").pos_error["kind"] == "ellipse95axis"
    assert "radius_pad_arcsec" not in registry.get("exoplanet_archive").parameters
    assert registry.get("nvss").pos_error["units"] == ["s_ra", "arcsec"]
    assert registry.get("vlass").pos_error["units"] == "deg"
    assert registry.get("xmm").pos_error["kind"] == "radial"
    assert registry.get("first").pos_error["kind"] == "first"


def test_minimal_definition_is_backward_compatible() -> None:
    catalog = CatalogDefinition(name="x", provider="tap", wavelength="optical")
    assert catalog.epoch is None and catalog.pos_error == {} and catalog.max_rows == 200
    assert catalog.as_dict()["name"] == "x"


def test_validation_catches_bad_definitions() -> None:
    bad = catalog_from_dict("bad", {
        "provider": "nope", "wavelength": "x", "endpoint": "ftp://x", "max_rows": 0, "epoch": 3000.0,
        "epoch_format": "weird", "pos_error": {"kind": "banana", "columns": ["a"], "units": "mJy"},
        "parameters": {"format": "fits", "distance": "furlongs"},
    })
    problems = " | ".join(validate_catalog_definition(bad))
    for fragment in ("unknown provider", "endpoint", "max_rows", "epoch 3000", "epoch_format", "banana",
                     "not an angle", "format", "distance"):
        assert fragment in problems, fragment


def test_pos_error_columns_must_be_selected() -> None:
    catalog = catalog_from_dict("c", {
        "provider": "tap", "wavelength": "x", "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["ra", "dec"]}, "pos_error": {"columns": ["e_ra"], "units": "arcsec"},
    })
    assert any("e_ra" in p for p in validate_catalog_definition(catalog))


def test_yaml_registry_override_reads_new_fields(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.yaml"
    path.write_text(yaml.safe_dump({"catalogs": {"custom": {
        "provider": "tap", "wavelength": "optical", "endpoint": "https://example.org/tap/sync", "table": '"I/1/x"',
        "epoch": ["t1", "t2"], "epoch_format": "mjd", "max_rows": 50, "timeout_seconds": 45,
        "pos_error": {"columns": ["e"], "units": "mas", "kind": "sigma"}, "citation": "Someone 2020",
        "parameters": {"columns": ["ra", "dec", "e", "t1", "t2"]},
    }}}), encoding="utf-8")
    reg = CatalogRegistry(path)
    custom = reg.get("custom")
    assert custom.epoch == ["t1", "t2"] and custom.epoch_format == "mjd"
    assert custom.max_rows == 50 and custom.timeout_seconds == 45.0
    assert custom.citation == "Someone 2020"
    assert reg.validate() == []


def test_by_profile(registry: CatalogRegistry) -> None:
    radio = registry.by_profile("radio")
    assert {"first", "nvss", "vlass", "lotss"} <= set(radio)
    assert "gaia_dr3" not in radio


def test_tap_catalog_requires_table() -> None:
    catalog = catalog_from_dict("t", {"provider": "tap", "wavelength": "x", "epoch": 2000.0})
    assert any("need a table" in p for p in validate_catalog_definition(catalog))
