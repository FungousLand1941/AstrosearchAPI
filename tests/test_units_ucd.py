"""Unit conversion, UCD-driven field mapping, positional errors, and epochs."""

from __future__ import annotations

import math

import pytest
from fixture_io import FIXTURES

from models import (
    K90_1D,
    K90_2D,
    K95_2D,
    ColumnMeta,
    angle_to_arcsec,
    angle_to_deg,
    arcsec_per_unit,
    bare_column_name,
    compute_positional_error,
    epoch_to_jyear,
    normalize_source_record,
    parallax_to_mas,
    parse_json_table,
    pm_to_masyr,
    resolve_epoch,
    ucd_field_map,
)

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("mas", 1e-3), ("arcsec", 1.0), ("deg", 3600.0), ("arcmin", 60.0), ("uas", 1e-6), ("'", 60.0), (45.0, 45.0)],
)
def test_arcsec_per_unit(unit, expected) -> None:
    assert arcsec_per_unit(unit) == pytest.approx(expected)


def test_non_angle_units_are_rejected() -> None:
    assert arcsec_per_unit("s") is None  # seconds of time are not an angle
    assert arcsec_per_unit("mJy") is None
    assert arcsec_per_unit(None) is None
    assert angle_to_arcsec(1.0, "Jy") is None


def test_angle_conversions() -> None:
    assert angle_to_arcsec(18.1, "mas") == pytest.approx(0.0181)
    assert angle_to_arcsec("5.9e-7", "deg") == pytest.approx(5.9e-7 * 3600)
    assert angle_to_deg(3600.0, "arcsec") == pytest.approx(1.0)
    assert angle_to_deg(12.5, None) == 12.5
    assert angle_to_deg(12.5, "h") == pytest.approx(187.5)
    assert angle_to_arcsec(None, "mas") is None


def test_proper_motion_and_parallax_units() -> None:
    assert pm_to_masyr(1.0, "arcsec/yr") == pytest.approx(1000.0)
    assert pm_to_masyr(5.0, "mas.yr-1") == pytest.approx(5.0)
    assert pm_to_masyr(5.0, "mas/yr") == pytest.approx(5.0)
    assert pm_to_masyr(5.0, None) == 5.0
    assert parallax_to_mas(0.02, "arcsec") == pytest.approx(20.0)
    assert parallax_to_mas(20.0, "mas") == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (2016.0, "yr", 2016.0),
        (51544.5, "mjd", 2000.0),
        (2451545.0, "jd", 2000.0),
        (51039.2, "d", 1998.6164),  # HEASARC mean_epoch: MJD with unit 'd'
        (2458590.8, "d", 2019.2903),  # VizieR 'Obs': JD with unit 'd'
        (55471.758, None, 2010.752),
        ("2000-01-01T12:00:00", None, 2000.0),
    ],
)
def test_epoch_to_jyear(value, fmt, expected) -> None:
    assert epoch_to_jyear(value, fmt) == pytest.approx(expected, abs=1e-3)


def test_epoch_out_of_range_rejected() -> None:
    assert epoch_to_jyear(12.0, "yr") is None
    assert epoch_to_jyear(None, "mjd") is None


def test_bare_column_name() -> None:
    assert bare_column_name("b.main_id") == "main_id"
    assert bare_column_name('"2MASS"') == "2MASS"
    assert bare_column_name("pz.z AS photoz") == "photoz"
    assert bare_column_name('"time"') == "time"


# ---------------------------------------------------------------------------
# UCD mapping
# ---------------------------------------------------------------------------


def test_ucd_map_on_real_gaia_metadata() -> None:
    table = parse_json_table((FIXTURES / "3c273" / "gaia_dr3.0.body").read_bytes())
    mapping = ucd_field_map(table.columns)
    assert mapping["ra"].name == "ra"
    assert mapping["dec"].name == "dec"
    assert mapping["pmra"].name == "pmra"
    assert mapping["parallax"].name == "parallax"
    assert mapping["ra_error_arcsec"].name == "ra_error"
    assert mapping["epoch"].name == "ref_epoch"
    values = normalize_source_record(table.rows[0], columns=table.columns)
    assert values["ra_error_arcsec"] == pytest.approx(table.rows[0]["ra_error"] / 1000.0)  # mas -> arcsec
    assert values["epoch"] == 2016.0


def test_vizier_computed_distance_with_ra_ucd_is_not_mapped_as_ra() -> None:
    # VizieR labels a computed 'DISTANCE(...)*3600 AS dist_arcsec' with RAJ2000's unit/UCD.
    columns = [
        ColumnMeta("dist_arcsec", unit="deg", ucd="pos.eq.ra;meta.main"),
        ColumnMeta("match_dist", unit="deg", ucd="pos.eq.ra;meta.main"),
        ColumnMeta("RAJ2000", unit="deg", ucd="pos.eq.ra;meta.main"),
        ColumnMeta("DEJ2000", unit="deg", ucd="pos.eq.dec;meta.main"),
    ]
    mapping = ucd_field_map(columns)
    assert mapping["ra"].name == "RAJ2000"
    values = normalize_source_record({"dist_arcsec": 0.066, "match_dist": 1.8e-5, "RAJ2000": 187.277897, "DEJ2000": 2.052387}, columns=columns)
    assert values["ra"] == pytest.approx(187.277897)


def test_real_vizier_payload_maps_quoted_identifier_columns() -> None:
    table = parse_json_table((FIXTURES / "3c273" / "vizier_2mass_reference.0.body").read_bytes())
    assert table.rows[0]["2MASS"].strip() == "12290669+0203085"
    values = normalize_source_record(table.rows[0], columns=table.columns, field_map={"source_id": "2MASS"})
    assert values["source_id"] == "12290669+0203085"  # trailing space stripped
    assert values["ra"] == pytest.approx(187.277897, abs=1e-5)


def test_ucd_mapping_for_arbitrary_catalog_with_unit_conversion() -> None:
    columns = [
        ColumnMeta("RA_deg", unit="deg", ucd="pos.eq.ra;meta.main"),
        ColumnMeta("DE_deg", unit="deg", ucd="pos.eq.dec;meta.main"),
        ColumnMeta("Seq", ucd="meta.id;meta.main"),
        ColumnMeta("pmRA", unit="arcsec/yr", ucd="pos.pm;pos.eq.ra"),
        ColumnMeta("pmDE", unit="mas/yr", ucd="pos.pm;pos.eq.dec"),
        ColumnMeta("Plx", unit="arcsec", ucd="pos.parallax.trig"),
        ColumnMeta("zsp", ucd="src.redshift"),
        ColumnMeta("e_zsp", ucd="stat.error;src.redshift"),
        ColumnMeta("Class", ucd="src.class"),
        ColumnMeta("SpT", ucd="src.spType"),
        ColumnMeta("e_RA", unit="mas", ucd="stat.error;pos.eq.ra"),
        ColumnMeta("e_DE", unit="deg", ucd="stat.error;pos.eq.dec"),
        ColumnMeta("Epoch", unit="yr", ucd="time.epoch"),
    ]
    row = {"RA_deg": 10.0, "DE_deg": -5.0, "Seq": "  X   1 ", "pmRA": 0.1, "pmDE": -3.0, "Plx": 0.01,
           "zsp": 0.5, "e_zsp": 0.01, "Class": "QSO", "SpT": "G2V", "e_RA": 20.0, "e_DE": 1e-5, "Epoch": 2015.5}
    values = normalize_source_record(row, columns=columns)
    assert values["ra"] == 10.0 and values["dec"] == -5.0
    assert values["source_id"] == "X 1"
    assert values["pmra"] == pytest.approx(100.0)
    assert values["pmdec"] == pytest.approx(-3.0)
    assert values["parallax"] == pytest.approx(10.0)
    assert values["redshift"] == 0.5
    assert values["object_type"] == "QSO"
    assert values["spectral_type"] == "G2V"
    assert values["ra_error_arcsec"] == pytest.approx(0.02)
    assert values["dec_error_arcsec"] == pytest.approx(0.036)
    assert values["epoch"] == 2015.5


def test_field_map_beats_ucd_beats_alias() -> None:
    columns = [ColumnMeta("objName", ucd="meta.id;meta.main"), ColumnMeta("objID")]
    row = {"objName": "PSO J1", "objID": "12345", "ra": 1.0, "dec": 2.0}
    assert normalize_source_record(row, columns=columns)["source_id"] == "PSO J1"
    assert normalize_source_record(row, columns=columns, field_map={"source_id": "objID"})["source_id"] == "12345"
    assert normalize_source_record({"designation": "J1", "ra": 1, "dec": 2})["source_id"] == "J1"


def test_legacy_aliases_still_work() -> None:
    res = normalize_source_record({"RAJ2000": 12.5, "DEJ2000": -4.2, "designation": "J0001", "plx_value": 15.2, "sp_type": "G2V"})
    assert res["ra"] == 12.5 and res["dec"] == -4.2
    assert res["source_id"] == "J0001" and res["parallax"] == 15.2 and res["spectral_type"] == "G2V"


def test_unit_ambiguous_error_names_are_not_treated_as_arcsec() -> None:
    # Bug 8: Gaia ra_error is in mas; the alias table must not call it arcsec.
    values = normalize_source_record({"ra": 1.0, "dec": 2.0, "ra_error": 0.3, "uncmaja": 30.0, "err_pos": 1.0})
    assert "position_uncertainty_arcsec" not in values


# ---------------------------------------------------------------------------
# Positional errors
# ---------------------------------------------------------------------------


def test_sigma_two_axes_rms_mas() -> None:
    sigma, details = compute_positional_error({"ra_error": 0.0181, "dec_error": 0.0129},
                                              {"columns": ["ra_error", "dec_error"], "units": "mas", "kind": "sigma"})
    assert sigma == pytest.approx(math.sqrt((0.0181**2 + 0.0129**2) / 2) / 1000)
    assert details["sigma_ra_arcsec"] == pytest.approx(1.81e-5)
    assert details["source"] == "catalog"


def test_ellipse_and_ellipse95() -> None:
    spec = {"columns": ["maj", "min", "pa"], "units": "arcsec", "kind": "ellipse"}
    sigma, details = compute_positional_error({"maj": 0.17, "min": 0.08, "pa": 0}, spec)
    assert sigma == pytest.approx(math.sqrt((0.17**2 + 0.08**2) / 2))
    assert details["ellipse_1sigma_arcsec"] == {"major": 0.17, "minor": 0.08, "pa_deg": 0.0}
    sigma95, _ = compute_positional_error({"maj": 0.71, "min": 0.71}, {**spec, "kind": "ellipse95"})
    assert sigma95 == pytest.approx(0.71 / K95_2D)
    assert K95_2D == pytest.approx(2.4477, abs=1e-4)


@pytest.mark.parametrize(("kind", "divisor"), [("radius95", K95_2D), ("radius90", K90_2D), ("axis90", K90_1D), ("radial", math.sqrt(2))])
def test_radius_kinds(kind, divisor) -> None:
    sigma, _ = compute_positional_error({"r": 2.0}, {"columns": ["r"], "units": "arcsec", "kind": kind})
    assert sigma == pytest.approx(2.0 / divisor)


def test_nvss_seconds_of_time_scale_with_cos_dec() -> None:
    spec = {"columns": ["ra_error", "dec_error"], "units": ["s_ra", "arcsec"], "kind": "sigma"}
    sigma, details = compute_positional_error({"ra_error": 0.03, "dec_error": 0.6}, spec, dec_deg=60.0)
    assert details["sigma_ra_arcsec"] == pytest.approx(0.03 * 15 * 0.5)
    assert sigma == pytest.approx(math.sqrt((0.225**2 + 0.6**2) / 2))


def test_rosat_pixels_and_systematic_in_quadrature() -> None:
    spec = {"columns": ["x", "y"], "units": 45.0, "kind": "sigma", "systematic_arcsec": 5.0}
    sigma, details = compute_positional_error({"x": 0.02, "y": 0.02}, spec)
    assert details["statistical_sigma_arcsec"] == pytest.approx(0.9)
    assert sigma == pytest.approx(math.hypot(0.9, 5.0))


def test_first_formula() -> None:
    spec = {"columns": ["maj", "min", "peak", "rms"], "units": ["arcsec", "arcsec", None, None], "kind": "first", "floor_arcsec": 0.1}
    sigma, _ = compute_positional_error({"maj": 6.0, "min": 5.0, "peak": 100.25, "rms": 0.2}, spec)
    snr = 100.0 / 0.2
    e_maj = 6.0 * (1 / snr + 1 / 20) / K90_1D
    e_min = 5.0 * (1 / snr + 1 / 20) / K90_1D
    assert sigma == pytest.approx(math.sqrt((e_maj**2 + e_min**2) / 2))


def test_fallback_by_quality_and_default() -> None:
    spec = {"columns": ["maj"], "units": "mas", "kind": "ellipse", "fallback_column": "qual", "fallback_values": {"D": 5.0}}
    sigma, details = compute_positional_error({"maj": None, "qual": "D"}, spec)
    assert sigma == 5.0 and details["source"] == "fallback"
    sigma, details = compute_positional_error({}, {"columns": [], "kind": "sigma", "default_arcsec": 0.1})
    assert sigma == 0.1 and details["source"] == "default"
    sigma, _ = compute_positional_error({"maj": None, "qual": "A?"}, spec)
    assert sigma is None


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        compute_positional_error({}, {"kind": "banana"})


def test_resolve_epoch_fixed_column_and_mean() -> None:
    assert resolve_epoch({}, 2016.0) == 2016.0
    assert resolve_epoch({"jdate": 2451545.0}, "jdate", "jd") == pytest.approx(2000.0)
    assert resolve_epoch({"time": 51544.5, "end_time": 51909.75}, ["time", "end_time"], "mjd") == pytest.approx(2000.5)
    assert resolve_epoch({"x": None}, "x", "mjd") is None
    assert resolve_epoch({"mean_epoch": 51039.2}, "mean_epoch", columns=[ColumnMeta("mean_epoch", unit="d")]) == pytest.approx(1998.616, abs=1e-3)


# ---------------------------------------------------------------------------
# Review round 1: impossible values, per-axis 95% ellipses, divisors, PA units
# ---------------------------------------------------------------------------


def test_declared_mjd_or_jd_that_cannot_be_one_is_rejected() -> None:
    assert epoch_to_jyear(1995.4, "mjd") is None  # a Julian year mislabelled as MJD (was 1864.3)
    assert epoch_to_jyear(-5.0, "mjd") is None
    assert epoch_to_jyear(55347.8, "mjd") == pytest.approx(2010.413, abs=1e-3)
    assert epoch_to_jyear(2010.4, "jd") is None
    assert epoch_to_jyear(51544.5, "jd") is None  # an MJD declared as JD
    assert epoch_to_jyear(2451545.0, "jd") == pytest.approx(2000.0)


def test_d_is_a_day_not_a_degree() -> None:
    assert arcsec_per_unit("d") is None  # VOUnit/CDS 'd' = day (VizieR JD columns)
    assert arcsec_per_unit("deg") == 3600.0


def test_ra_never_normalizes_to_360() -> None:
    from models import validate_target

    target = validate_target(-1e-14, 10.0)
    assert 0.0 <= target.ra < 360.0 and target.ra == 0.0


def test_proper_motion_validation() -> None:
    from models import InvalidCoordinateError, validate_target

    assert validate_target(1, 2, epoch=2000, pm_ra_masyr=1, pm_dec_masyr=2).proper_motion == (1.0, 2.0)
    with pytest.raises(InvalidCoordinateError):
        validate_target(1, 2, pm_ra_masyr=1.0)
    with pytest.raises(InvalidCoordinateError):
        validate_target(1, 2, pm_ra_masyr=1e6, pm_dec_masyr=0)


def test_per_axis_95_ellipse_uses_1_96() -> None:
    # CSC 2.1 (CXC columns doc): semi-axes are 95% intervals along each axis.
    spec = {"columns": ["r0", "r1", "ang"], "units": "arcsec", "kind": "ellipse95axis"}
    sigma, details = compute_positional_error({"r0": 0.6478, "r1": 0.6478, "ang": 30.0}, spec)
    assert sigma == pytest.approx(0.6478 / 1.96, rel=1e-4)
    assert details["units"] == ["arcsec", "arcsec", "deg"]
    assert details["ellipse_1sigma_arcsec"]["pa_deg"] == 30.0


def test_explicit_divisor_overrides_kind() -> None:
    # NED: uncertainty ellipse = 2.5 x the source catalog's 1-sigma values.
    spec = {"columns": ["maj", "min", "pa"], "units": "arcsec", "kind": "ellipse", "divisor": 2.5}
    sigma, details = compute_positional_error({"maj": 0.08725, "min": 0.08525, "pa": 0}, spec)
    assert sigma == pytest.approx(math.sqrt((0.0349**2 + 0.0341**2) / 2), rel=1e-6)
    assert details["divisor"] == 2.5 and details["units"][2] == "deg"
    with pytest.raises(ValueError):
        compute_positional_error({"maj": 1.0}, {**spec, "divisor": 0})


def test_mas_ellipse_position_angle_is_degrees() -> None:
    spec = {"columns": ["coo_err_maj", "coo_err_min", "coo_err_angle"], "units": "mas", "kind": "ellipse"}
    _, details = compute_positional_error({"coo_err_maj": 18.1, "coo_err_min": 12.9, "coo_err_angle": 90}, spec)
    assert details["units"] == ["mas", "mas", "deg"]


def test_registry_rejects_day_unit_and_bad_epoch_specs() -> None:
    from models import catalog_from_dict, validate_catalog_definition

    bad = catalog_from_dict("bad", {
        "provider": "tap", "wavelength": "x", "table": "t",
        "epoch": {"column": "bib", "values": {"x": 12.0}},
        "parameters": {"columns": ["ra", "dec", "e", "bib"], "epoch_range": [2000.0], "column_units": ["x"]},
        "pos_error": {"columns": ["e"], "units": "d", "kind": "sigma", "divisor": -1},
    })
    problems = " | ".join(validate_catalog_definition(bad))
    for fragment in ("not an angle", "epoch lookup values", "epoch_range", "column_units", "divisor"):
        assert fragment in problems, fragment
