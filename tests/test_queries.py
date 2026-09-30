"""Query construction: ADQL, MAST parameters, SkyServer SQL, HTTP method and form fields."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs

import pytest
import respx
from helpers import offline_client

from models import CatalogRegistry, catalog_from_dict, haversine_arcsec, validate_target
from providers import CacheManager, HEASARCXaminProvider, MASTProvider, SDSSProvider, TapProvider

TARGET = validate_target(187.2779154, 2.0523883)


@pytest.fixture
def tap() -> TapProvider:
    return TapProvider(offline_client(), cache=CacheManager(None))


def test_adql_orders_nearest_first_with_top_n(tap: TapProvider, registry: CatalogRegistry) -> None:
    adql = tap.build_adql(registry.get("gaia_dr3"), TARGET, 10.0)
    # TOP max_rows + 1: a cone holding exactly max_rows rows is not reported as truncated.
    assert adql.startswith("SELECT TOP 201 source_id, ra, dec, ra_error, dec_error")
    assert "DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', 187.277915400, 2.052388300)) AS match_dist" in adql
    assert "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 187.277915400, 2.052388300, 0.0027777778))" in adql
    assert adql.endswith("ORDER BY match_dist ASC")
    assert "FROM gaiadr3.gaia_source WHERE" in adql


def test_max_rows_controls_top(tap: TapProvider, registry: CatalogRegistry) -> None:
    catalog = registry.get("ned")
    catalog.max_rows = 7
    assert tap.build_adql(catalog, TARGET, 10.0).startswith("SELECT TOP 8 ")


def test_simbad_join_uses_qualified_point_and_unqualified_order(tap: TapProvider, registry: CatalogRegistry) -> None:
    adql = tap.build_adql(registry.get("simbad"), TARGET, 10.0)
    assert "FROM basic AS b LEFT OUTER JOIN allfluxes AS f ON f.oidref = b.oid" in adql
    assert "POINT('ICRS', b.ra, b.dec)" in adql
    assert "ORDER BY match_dist ASC" in adql  # CDS rejects qualified ORDER BY names
    assert "b.coo_err_maj" in adql and "err_pos" not in adql


def test_vizier_identifiers_are_quoted(tap: TapProvider, registry: CatalogRegistry) -> None:
    adql = tap.build_adql(registry.get("vizier_2mass_reference"), TARGET, 10.0)
    assert 'FROM "II/246/out" WHERE' in adql and '"2MASS"' in adql
    assert "POINT('ICRS', RAJ2000, DEJ2000)" in adql


def test_irsa_distance_is_not_multiplied(tap: TapProvider, registry: CatalogRegistry) -> None:
    adql = tap.build_adql(registry.get("twomass_psc"), TARGET, 10.0)
    assert "*3600" not in adql.replace(" ", "")
    assert "AS match_dist" in adql and "ORDER BY match_dist" in adql


def test_heasarc_reserved_column_quoted(tap: TapProvider, registry: CatalogRegistry) -> None:
    adql = tap.build_adql(registry.get("rosat"), TARGET, 10.0)
    assert '"time"' in adql and "FROM rass2rxs WHERE" in adql


def test_exoplanet_cone_is_exact_without_target_epoch(tap: TapProvider, registry: CatalogRegistry) -> None:
    # No fixed pad any more: without a target epoch the cone is exactly the request.
    adql = tap.build_adql(registry.get("exoplanet_archive"), TARGET, 10.0)
    assert f"{10.0 / 3600.0:.10f}))" in adql
    assert "FROM pscomppars WHERE" in adql


def test_exoplanet_cone_follows_target_with_epoch_and_proper_motion(tap: TapProvider, registry: CatalogRegistry) -> None:
    # Barnard's star at its SIMBAD J2000 position with its proper motion: the cone is
    # centred where the star is at J2015.5 (160.7" away), not on the J2000 position.
    barnard = validate_target(269.45207696, 4.69336497, epoch=2000.0, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394)
    adql = tap.build_adql(registry.get("exoplanet_archive"), barnard, 5.0)
    center = re.search(r"CIRCLE\('ICRS', ([\d.]+), ([\d.-]+), ([\d.]+)\)", adql)
    ra, dec, radius = float(center.group(1)), float(center.group(2)), float(center.group(3)) * 3600.0
    # Independent truth: pscomppars lists Barnard b at (269.4486144, 4.7379808) J2015.5:
    # the 5" cone around it is inside the query cone ...
    assert haversine_arcsec(ra, dec, 269.4486144, 4.7379808) + 5.0 <= radius + 1e-3
    # ... and so are the target's J2000 neighbours (pscomppars has per-row pm): one
    # minimal circle enclosing both, not the whole unknown-pm pad (10.5"/yr x 15.5 yr).
    assert haversine_arcsec(ra, dec, barnard.ra, barnard.dec) + 5.0 + 0.1 * 15.5 <= radius + 1e-3
    assert radius == pytest.approx(5.0 + (160.7 + 1.55) / 2, abs=0.2)


def test_cone_widened_by_max_proper_motion_when_motion_unknown(tap: TapProvider, registry: CatalogRegistry) -> None:
    target = validate_target(269.45207696, 4.69336497, epoch=2000.0)
    adql = tap.build_adql(registry.get("gaia_dr3"), target, 10.0)
    # 10" + 10.5"/yr x 16 yr = 178"
    assert f"{178.0 / 3600.0:.10f}))" in adql


def test_where_clause_and_no_distance_mode(tap: TapProvider) -> None:
    catalog = catalog_from_dict("c", {"provider": "tap", "wavelength": "x", "table": "t", "epoch": 2000.0,
                                      "parameters": {"columns": ["id", "ra", "dec"], "distance": "none", "where": "flag < 2"}})
    adql = tap.build_adql(catalog, TARGET, 5.0)
    assert "ORDER BY" not in adql and "DISTANCE" not in adql
    assert adql.endswith("AND (flag < 2)")


def test_form_fields_and_format(tap: TapProvider, registry: CatalogRegistry) -> None:
    json_form = tap.build_request(registry.get("gaia_dr3"), "SELECT 1")
    assert json_form == {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": "SELECT 1", "FORMAT": "json"}
    vot_form = tap.build_request(registry.get("chandra"), "SELECT 1")
    assert "FORMAT" not in vot_form  # HEASARC rejects FORMAT=json; VOTable is the default


@respx.mock
async def test_tap_queries_are_sent_as_post_form_bodies(registry: CatalogRegistry) -> None:
    route = respx.post("https://exoplanetarchive.ipac.caltech.edu/TAP/sync").respond(json=[])
    async with offline_client() as client:
        result = await TapProvider(client, cache=CacheManager(None)).query(registry.get("exoplanet_archive"), TARGET, 10.0)
    request = route.calls.last.request
    assert request.method == "POST"
    assert request.url.query == b""  # nothing in the URL for a WAF to match
    form = parse_qs(request.content.decode())
    assert form["QUERY"][0].startswith("SELECT TOP 201 pl_name")
    assert result.meta["status"] == "empty" and result.meta["query"] == form["QUERY"][0]


def test_mast_parameters(registry: CatalogRegistry) -> None:
    params = MASTProvider(offline_client(), cache=CacheManager(None)).build_params(registry.get("panstarrs_dr2"), TARGET, 10.0)
    assert float(params["radius"]) == pytest.approx(10.0 / 3600.0)  # degrees
    columns = json.loads(params["columns"])
    assert "distance" in columns and "raMeanErr" in columns
    assert params["sort_by"] == "distance"
    assert params["nDetections.gt"] == "1"
    assert params["pagesize"] == "201"  # max_rows + 1 to detect truncation exactly


def test_sdss_sql_radius_is_arcmin_and_ordered(registry: CatalogRegistry) -> None:
    sql = SDSSProvider(offline_client(), cache=CacheManager(None)).build_sql(registry.get("sdss"), TARGET, 10.0)
    assert f"fGetNearbyObjEq(187.277915400, 2.052388300, {10.0 / 60.0:.10f})" in sql
    assert sql.startswith("SELECT TOP 201 n.objID AS objID")
    assert sql.endswith("ORDER BY n.distance")
    assert "JOIN PhotoPrimary AS p" in sql and "SpecObj" in sql and "Photoz" in sql


def test_xamin_uses_position_and_arcmin_radius(registry: CatalogRegistry) -> None:
    params = HEASARCXaminProvider(offline_client(), cache=CacheManager(None)).build_params(registry.get("nvss"), TARGET, 10.0)
    assert "coord" not in params
    assert params["position"] == "187.277915400,2.052388300"
    assert float(params["radius"]) == pytest.approx(10.0 / 60.0)
