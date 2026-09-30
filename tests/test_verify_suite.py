"""The built-in ``main.py verify`` checks, as individual pytest tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from crossmatch import AdvancedQuery, angular_separation_arcsec, match_score
from datasets import DatasetWriter
from main import run_verification
from models import CatalogRegistry, InvalidCoordinateError, Target, normalize_source_record, parse_ipac_records, validate_target
from providers import EndpointGuard, SesameResolver


def test_coordinate_normalization_and_bounds() -> None:
    target = validate_target(361.0, 10.0)
    assert target.ra == 1.0 and target.dec == 10.0
    with pytest.raises(InvalidCoordinateError):
        validate_target(10.0, 95.0)
    with pytest.raises(InvalidCoordinateError):
        validate_target("abc", 1.0)
    with pytest.raises(InvalidCoordinateError):
        validate_target(1.0, 1.0, epoch=3000)


def test_angular_separation_arcsec() -> None:
    assert 3590 < angular_separation_arcsec(Target(0.0, 0.0), Target(0.0, 1.0)) < 3610


def test_match_scoring_monotonic() -> None:
    close = match_score(0.2, positional_error_arcsec=1.0)
    far = match_score(5.0, positional_error_arcsec=1.0)
    assert 0.0 <= far < close <= 1.0


def test_field_alias_normalization() -> None:
    res = normalize_source_record({"RAJ2000": 12.5, "DEJ2000": -4.2, "designation": "J0001", "plx_value": 15.2, "sp_type": "G2V"})
    assert res["ra"] == 12.5 and res["dec"] == -4.2
    assert res["source_id"] == "J0001" and res["parallax"] == 15.2 and res["spectral_type"] == "G2V"


def test_ipac_parser() -> None:
    payload = (
        "\\fixlen = T\n\\RowsRetrieved = 1\n"
        "| ra        | dec       | designation |\n"
        "| double    | double    | char        |\n"
        "|           |           |             |\n"
        " 10.123456   -5.654321   J001\n"
    )
    rows = parse_ipac_records(payload)
    assert len(rows) == 1 and rows[0]["designation"] == "J001"
    assert abs(float(rows[0]["ra"]) - 10.123456) < 1e-5


def test_embedded_registry() -> None:
    enabled = CatalogRegistry().enabled_catalogs()
    assert len(enabled) >= 15
    for name in ("gaia_dr3", "twomass_psc", "allwise", "sdss"):
        assert name in enabled


def test_sesame_parser() -> None:
    # Recorded real Sesame -oxp response (tests/fixtures/sesame), not an invented schema.
    from fixture_io import FIXTURES

    res = SesameResolver.parse_response("M87", (FIXTURES / "sesame" / "m87.xml").read_text(encoding="utf-8"))
    assert res.canonical_name == "M 87"
    assert abs(res.ra_deg - 187.70593077) < 1e-7
    assert res.object_type == "AGN" and res.redshift == 0.0042
    assert res.pm_ra_masyr == -8.029 and res.pm_dec_masyr == 10.734 and res.epoch == 2000.0
    star = SesameResolver.parse_response("Barnard's star", (FIXTURES / "sesame" / "barnards_star.xml").read_text(encoding="utf-8"))
    assert star.pm_dec_masyr == 10362.394 and star.epoch == 2000.0
    # Legacy flat layout still parses.
    legacy = SesameResolver.parse_response("M87", "<Sesame><Result><oname>M 87</oname><alias>NGC 4486</alias>"
                                                  "<jradeg>187.70593</jradeg><jdedeg>12.391123</jdedeg><z_value>0.00428</z_value></Result></Sesame>")
    assert legacy.redshift == 0.00428 and "NGC 4486" in legacy.aliases


def test_advanced_query_filters() -> None:
    q = AdvancedQuery.from_dict({
        "ra": 10.0, "dec": 5.0, "radius_arcsec": 10.0, "search_mode": "shell", "min_radius_arcsec": 2.0,
        "object_types": ["star"], "spatial_constraints": {"radius_zones": [{"min_arcsec": 2.0, "max_arcsec": 8.0}]},
    })
    source_in = {"ra": 10.001, "dec": 5.0, "separation_arcsec": 3.0, "physical": {"object_type": "star", "parallax": 10.0}}
    assert q.apply_filters(source_in)
    assert not q.apply_filters({**source_in, "separation_arcsec": 1.0})
    q.search_mode = "cylinder"
    q.min_distance_pc, q.max_distance_pc = 90.0, 110.0
    assert q.apply_filters(source_in)
    q.max_distance_pc = 95.0
    assert not q.apply_filters(source_in)


def test_endpoint_guard_circuit_breaker() -> None:
    guard = EndpointGuard(requests_per_second=1000, failure_threshold=2, recovery_seconds=10)
    assert guard.state == "closed"
    asyncio.run(guard.fail())
    assert guard.state == "closed"
    asyncio.run(guard.fail())
    assert guard.state == "open"
    asyncio.run(guard.succeed())
    assert guard.state == "closed"


@pytest.mark.parametrize("fmt", ["json", "csv", "parquet", "fits"])
def test_dataset_export_formats(fmt: str, tmp_path: Path) -> None:
    sample = {"catalog": "gaia_dr3", "source_id": "gaia-001", "ra": 187.25, "dec": 2.05, "confidence": 0.98,
              "separation_arcsec": 0.5, "physical": {"object_type": "star"}}
    out = tmp_path / f"test.{fmt}"
    with DatasetWriter(out, fmt) as writer:
        writer.write(sample)
    assert out.exists() and out.stat().st_size > 0


def test_api_routes() -> None:
    from fastapi.testclient import TestClient

    from api import app

    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["status"] == "healthy"
        catalogs = client.get("/api/v1/catalogs").json()
        assert "gaia_dr3" in catalogs
        assert catalogs["gaia_dr3"]["epoch"] == "ref_epoch"  # new registry fields are exposed
        assert client.get("/api/v1/catalogs/gaia_dr3").json()["name"] == "gaia_dr3"
        created = client.post("/api/v1/queries", json={"name": "M87-query", "query": {"ra": 187.7, "dec": 12.39}})
        assert created.status_code == 201
        query_id = created.json()["id"]
        assert any(q["id"] == query_id for q in client.get("/api/v1/queries").json())
        assert client.delete(f"/api/v1/queries/{query_id}").status_code == 204
        assert client.get("/api/v1/stats").status_code == 200
        assert client.get("/api/v1/monitoring").status_code == 200


def test_builtin_verify_command_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_verification() is True
    assert "tests passed (100.0%)" in capsys.readouterr().out
