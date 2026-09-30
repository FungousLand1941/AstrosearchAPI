"""Regression tests for astrometric review findings (round 3).

Each test reproduces a reported failure with a recorded real archive response (or the
exact values quoted in the report) and checks the corrected behaviour:

* 5XMM time/end_time is the span of the detection stack, not a point epoch.
* Rows without their own proper motion in catalogs that have them (Gaia 2-parameter
  solutions) are not moved with the target's motion.
* The annual parallax of the nearest stars is removed from single-epoch positions.
* NED error ellipses follow the convention of the catalogue they were copied from.
* SIMBAD rows without a proper motion keep the epoch of their original measurement.
* VLASS 'Redundant' duplicates are removed.
* Proper-motion adoption refuses ambiguous (crowded) matches.
* RA proper motions in seconds of time are converted with cos(dec).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect, target_for, target_radius
from helpers import make_service, offline_client, run_catalog

from crossmatch import (
    PARALLAX_INFLATION_MIN_MAS,
    _adopt_proper_motion,
    match_target,
    parallax_uncertainty_arcsec,
)
from models import (
    META_CATALOG_HAS_PM,
    META_SINGLE_EPOCH,
    CatalogRegistry,
    CatalogSource,
    ColumnMeta,
    compute_positional_error,
    earth_barycentric_au,
    epoch_separation_arcsec,
    normalize_source_record,
    parallax_offset_arcsec,
    pm_to_masyr,
    resolve_epoch,
    resolve_epoch_range,
    tangent_offset_arcsec,
    validate_target,
)

PROXIMA = "proxima_j2000_pm"
PROXIMA_PARALLAX = 768.0665  # mas (Gaia DR3, as Sesame supplies it)
PROXIMA_XMM = {"5XMM J142933.5-624032", "5XMM J142941.9-624044", "5XMM J142937.8-624040"}


def registry_of(*names: str) -> CatalogRegistry:
    reg = CatalogRegistry()
    reg._catalogs = {k: v for k, v in reg._catalogs.items() if k in names}
    return reg


async def crossmatch_recorded(key: str, catalogs: list[str], **kwargs):
    target = target_for(key)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(key, catalogs)))
        async with offline_client() as client:
            return await make_service(client, registry_of(*catalogs)).crossmatch(
                target.ra, target.dec, radius_arcsec=kwargs.pop("radius_arcsec", target_radius(key)),
                epoch=target.epoch, pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr, **kwargs)


def source(catalog: str = "x", sid: str = "s", ra: float = 10.0, dec: float = 20.0, **kw) -> CatalogSource:
    metadata = kw.pop("metadata", {})
    return CatalogSource(catalog, sid, ra, dec, kw.pop("err", 0.1), kw.pop("data", {}), metadata, {}, **kw)


# ---------------------------------------------------------------------------
# 5XMM: [time, end_time] is a per-row span
# ---------------------------------------------------------------------------


async def test_proxima_5xmm_detections_are_all_found_over_the_stack_span() -> None:
    # Reported: every 5XMM row near Proxima got the midpoint epoch 2009.90; the two
    # strongest Proxima detections became pad rows at 27.8" and 31.2" and a third was
    # 'inside' at 2.8". Each lies within ~1" of Proxima's track at its own epoch.
    record = await crossmatch_recorded(PROXIMA, ["xmm"])
    res = record.catalog_results["xmm"]
    found = {s.source_id: s for s in res["sources"]}
    assert PROXIMA_XMM <= set(found), found.keys()
    for sid in PROXIMA_XMM:
        src = found[sid]
        assert src.epoch is None
        assert src.epoch_range == pytest.approx((2001.61, 2018.19), abs=0.01)
        assert src.metadata["epoch_propagation"] == "target_pm_span"
        assert src.metadata["epoch_separation_arcsec"] < 1.1, sid
    assert not any(s.source_id in PROXIMA_XMM for s in res["pad_sources"])
    matched = {m["source_id"] for m in record.provenance["matches"] if m["catalog"] == "xmm"}
    assert PROXIMA_XMM <= matched


async def test_proxima_5xmm_midpoint_epoch_would_lose_the_detections() -> None:
    # The report's failure, reproduced from the same recorded rows with the old rule.
    sources, failure = await run_catalog("xmm", PROXIMA, radius_arcsec=5.0)
    assert failure is None
    target = target_for(PROXIMA)
    rows = list(sources) + list(sources.meta["pad_sources"])
    by_id = {s.source_id: s for s in rows}
    old = CatalogSource(**{**by_id["5XMM J142933.5-624032"].as_dict(), "epoch": 2009.90, "epoch_range": None})
    assert epoch_separation_arcsec(target, old)[0] > 25.0


def test_span_epoch_spec_point_when_short_and_span_when_long(registry: CatalogRegistry) -> None:
    xmm = registry.get("xmm")
    long_row = {"time": 52133.17, "end_time": 58189.11}
    assert resolve_epoch(long_row, xmm.epoch, xmm.epoch_format) is None
    assert resolve_epoch_range(long_row, xmm) == pytest.approx((2001.6117, 2018.1920), abs=1e-3)
    # A single observation (span of hours) keeps an exact epoch.
    short_row = {"time": 55000.0, "end_time": 55000.5}
    assert resolve_epoch(short_row, xmm.epoch, xmm.epoch_format) == pytest.approx(2000.0 + (55000.25 - 51544.5) / 365.25)
    assert resolve_epoch_range(short_row, xmm) is None
    assert resolve_epoch({"time": None, "end_time": 55000.0}, xmm.epoch, xmm.epoch_format) is None
    # 2RXS (two survey days) still uses the mean as a point epoch.
    rosat = registry.get("rosat")
    assert resolve_epoch({"time": 48100.0, "end_time": 48102.0}, rosat.epoch, rosat.epoch_format) == pytest.approx(
        2000.0 + (48101.0 - 51544.5) / 365.25)
    assert registry.validate() == []


# ---------------------------------------------------------------------------
# Rows without proper motion in catalogs that have them stay put
# ---------------------------------------------------------------------------


async def test_gaia_two_parameter_rows_are_not_moved_with_the_target() -> None:
    # Reported: Gaia 5853498713190524032 (2-parameter, G=18.9, 61.4" from Proxima's J2000
    # position) was a Proxima 'match' at 0.8" after being moved with Proxima's motion,
    # while genuine J2000 neighbours such as 5853495723865252096 (3.0") were dropped.
    record = await crossmatch_recorded(PROXIMA, ["gaia_dr3"])
    res = record.catalog_results["gaia_dr3"]
    inside = {s.source_id: s for s in res["sources"]}
    pad = {s.source_id: s for s in res["pad_sources"]}
    impostor = pad["5853498713190524032"]
    assert impostor.metadata["epoch_propagation"] == "stationary"
    assert impostor.metadata["epoch_separation_arcsec"] > 55.0
    assert impostor.metadata["target_pm_separation_arcsec"] < 5.0  # where it would be if it were Proxima
    assert "5853498713190524032" not in {m["source_id"] for m in record.provenance["matches"]}
    assert all(m["source_id"] != "5853498713190524032" for g in record.crossmatch_groups for m in g["members"])
    neighbour = inside["5853495723865252096"]
    assert neighbour.metadata["epoch_propagation"] == "stationary"
    assert neighbour.metadata["epoch_separation_arcsec"] == pytest.approx(3.0, abs=0.05)
    # Proxima itself (5-parameter) is found with its own motion.
    assert inside["5853498713190525696"].metadata["epoch_propagation"] == "source_pm"
    # Every in-radius row is within the radius at its own (stationary or pm) position.
    assert all(s.metadata["epoch_propagation"] in {"source_pm", "stationary"} for s in res["sources"])
    assert any("treated as stationary" in w for w in res["warnings"])


def test_stationary_only_for_catalogs_that_publish_proper_motions() -> None:
    target = validate_target(10.0, 20.0, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=1000.0)
    # 16" north of the target at 2016: where the target would be at 2016.
    ra, dec = 10.0, 20.0 + 16.0 / 3600.0
    gaia_like = source(ra=ra, dec=dec, epoch=2016.0, metadata={META_CATALOG_HAS_PM: True})
    twomass_like = source(ra=ra, dec=dec, epoch=2016.0, metadata={META_CATALOG_HAS_PM: False})
    sep, method = epoch_separation_arcsec(target, gaia_like)
    assert method == "stationary" and sep == pytest.approx(16.0, abs=1e-3)
    sep, method = epoch_separation_arcsec(target, twomass_like)
    assert method == "target_pm" and sep < 1e-3


# ---------------------------------------------------------------------------
# Annual parallax
# ---------------------------------------------------------------------------


def test_earth_position_and_parallax_offset_match_astropy() -> None:
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    ra, dec = 217.42894222, -62.67949019
    for jyear in (1997.6, 2000.194, 2005.37, 2010.9, 2016.0, 2019.45):
        earth = get_body_barycentric("earth", Time(jyear, format="jyear", scale="tdb")).xyz.to(u.AU).value
        assert np.linalg.norm(np.array(earth_barycentric_au(jyear)) - earth) < 0.015, jyear
        # Exact: direction from the Earth to a star 1/plx away.
        dist_au = 206264.806 / (PROXIMA_PARALLAX / 1000.0)
        a, d = math.radians(ra), math.radians(dec)
        v = dist_au * np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)]) - earth
        exact = tangent_offset_arcsec(ra, dec, math.degrees(math.atan2(v[1], v[0])) % 360.0,
                                      math.degrees(math.asin(v[2] / np.linalg.norm(v))))
        mine = parallax_offset_arcsec(ra, dec, PROXIMA_PARALLAX, jyear)
        assert math.hypot(mine[0] - exact[0], mine[1] - exact[1]) < 0.012, jyear  # < 1.5% of 0.77"


async def test_proxima_2mass_counterpart_needs_the_parallax() -> None:
    # Reported: 2MASS 14294291-6240465 (jdate 2000.194) at 0.709" with confidence 0.0;
    # the 768 mas parallax accounts for the whole offset.
    without = await crossmatch_recorded(PROXIMA, ["twomass_psc"])
    row = without.catalog_results["twomass_psc"]["sources"][0]
    assert row.source_id == "14294291-6240465"
    assert row.metadata["epoch_propagation"] == "target_pm"
    assert row.metadata["epoch_separation_arcsec"] == pytest.approx(0.709, abs=0.01)
    with_plx = await crossmatch_recorded(PROXIMA, ["twomass_psc"], parallax_mas=PROXIMA_PARALLAX)
    row = with_plx.catalog_results["twomass_psc"]["sources"][0]
    assert row.metadata["epoch_propagation"] == "target_pm_parallax"
    assert row.metadata["epoch_separation_arcsec"] < 0.1
    match = next(m for m in with_plx.provenance["matches"] if m["catalog"] == "twomass_psc")
    assert match["confidence"] > 0.5  # 0.0 before
    assert with_plx.provenance["target_parallax"] == {"parallax_mas": PROXIMA_PARALLAX, "source": "input"}
    # With a 0.3" radius the counterpart survives only with the parallax removed.
    spec = target_for(PROXIMA)
    plain = validate_target(spec.ra, spec.dec, epoch=spec.epoch, pm_ra_masyr=spec.pm_ra_masyr,
                            pm_dec_masyr=spec.pm_dec_masyr)
    corrected = validate_target(spec.ra, spec.dec, epoch=spec.epoch, pm_ra_masyr=spec.pm_ra_masyr,
                                pm_dec_masyr=spec.pm_dec_masyr, parallax_mas=PROXIMA_PARALLAX)
    assert match_target(plain, [row], 0.3) == []
    assert [m.source.source_id for m in match_target(corrected, [row], 0.3)] == ["14294291-6240465"]


async def test_parallax_is_adopted_from_gaia_when_only_the_motion_is_given() -> None:
    record = await crossmatch_recorded(PROXIMA, ["twomass_psc", "gaia_dr3"])
    plx = record.provenance["target_parallax"]
    assert plx["source"] == "adopted" and plx["catalog"] == "gaia_dr3"
    assert plx["parallax_mas"] == pytest.approx(768.07, abs=0.05)
    row = record.catalog_results["twomass_psc"]["sources"][0]
    assert row.metadata["epoch_separation_arcsec"] < 0.1


def test_uncorrected_rows_get_the_parallax_as_extra_uncertainty() -> None:
    target = validate_target(10.0, 20.0, epoch=2016.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0, parallax_mas=500.0)
    assert parallax_uncertainty_arcsec(target, "target_pm") == pytest.approx(0.5)
    assert parallax_uncertainty_arcsec(target, "target_pm_span") == pytest.approx(0.5)
    assert parallax_uncertainty_arcsec(target, "target_pm_parallax") is None
    assert parallax_uncertainty_arcsec(target, "source_pm") is None
    small = validate_target(10.0, 20.0, epoch=2016.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0,
                            parallax_mas=PARALLAX_INFLATION_MIN_MAS / 2)
    assert parallax_uncertainty_arcsec(small, "target_pm") is None
    # A multi-epoch mean (AllWISE-like, not single-epoch) 0.4" off: confidence stays usable.
    row = source(ra=10.0, dec=20.0 + 0.4 / 3600.0, epoch=2010.5, err=0.05, metadata={META_SINGLE_EPOCH: False})
    (match,) = match_target(target, [row], 3.0)
    assert match.confidence > 0.6
    with pytest.raises(Exception, match="parallax_mas"):
        validate_target(10.0, 20.0, parallax_mas=-1.0)


# ---------------------------------------------------------------------------
# NED error conventions by pos_bibcode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bibcode", "maj", "mino", "divisor", "source_kind"), [
    # 2CXO J123049.8+122327: NED copies the CSC 2.0 95%-per-axis ellipse unchanged.
    ("2020CSC...C2..0000:", 0.77524567, 0.75194365, 1.959963984540054, "lookup"),
    # 2XMM J123043.6+122147: NED 2.73 = 2.5 x the radial 1-sigma SC_POSERR (1.1").
    ("20102XMM.1.2..0000:", 2.73, 2.73, 2.5 * math.sqrt(2.0), "lookup"),
    # 2MASS: 2.5 x err_maj.
    ("2006AJ....131.1163S", 0.15, 0.15, 2.5, "lookup"),
    # Unknown source catalogue: 2.5 kept, but flagged as assumed.
    ("1995AJ....110..880J", 0.1, 0.1, 2.5, "assumed"),
])
def test_ned_error_divisor_follows_the_source_catalogue(registry, bibcode, maj, mino, divisor, source_kind) -> None:
    ned = registry.get("ned")
    row = {"uncmaja": maj, "uncmina": mino, "uncposa": 119.762314, "pos_bibcode": bibcode}
    sigma, details = compute_positional_error(row, ned.pos_error, dec_deg=12.4)
    assert sigma == pytest.approx(math.sqrt((maj**2 + mino**2) / 2) / divisor, rel=1e-9)
    assert details["divisor"] == pytest.approx(divisor) and details["divisor_source"] == source_kind


def test_ned_csc_sigma_matches_the_csc_documented_sigma(registry) -> None:
    # CSC 2.0 r0/r1 = 0.775/0.752 per-axis 95% -> sigma 0.390" (0.306" with the old /2.5).
    sigma, _ = compute_positional_error(
        {"uncmaja": 0.77524567, "uncmina": 0.75194365, "uncposa": 119.76, "pos_bibcode": "2020CSC...C2..0000:"},
        registry.get("ned").pos_error)
    assert sigma == pytest.approx(math.sqrt((0.775**2 + 0.752**2) / 2) / 1.96, abs=1e-3)


# ---------------------------------------------------------------------------
# SIMBAD epochs and J2000 errors
# ---------------------------------------------------------------------------


async def test_simbad_row_without_pm_keeps_its_measurement_epoch() -> None:
    # GES J17574778+0443542 (Gaia-ESO, ~2013; no pm) lies 0.09" from Barnard's track at
    # 2013.3 but was propagated from 'J2000' and rejected at 138.5".
    record = await crossmatch_recorded("barnard_gaia2016", ["simbad", "gaia_dr3"])
    res = record.catalog_results["simbad"]
    ges = next(s for s in res["sources"] if s.source_id == "GES J17574778+0443542")
    assert ges.epoch is None and ges.epoch_range == (1990.0, 2025.0)
    assert ges.metadata["epoch_propagation"] == "target_pm_span"
    assert ges.metadata["epoch_separation_arcsec"] < 0.2
    star = next(s for s in res["sources"] if s.source_id == "NAME Barnard's star")
    assert star.epoch == 2000.0 and star.metadata["epoch_propagation"] == "source_pm"


def test_simbad_epoch_rules(registry: CatalogRegistry) -> None:
    simbad = registry.get("simbad")
    with_pm = {"pmra": -801.5, "pmdec": 10362.4, "coo_bibcode": "2020yCat.1350....0G"}
    gaia_no_pm = {"pmra": None, "pmdec": None, "coo_bibcode": "2020yCat.1350....0G"}
    twomass_no_pm = {"pmra": None, "pmdec": None, "coo_bibcode": "2006AJ....131.1163S"}
    unknown = {"pmra": None, "pmdec": None, "coo_bibcode": "2023A&A...676A.129H"}
    assert resolve_epoch(with_pm, simbad.epoch) == 2000.0 and resolve_epoch_range(with_pm, simbad) is None
    assert resolve_epoch(gaia_no_pm, simbad.epoch) == 2016.0
    assert resolve_epoch(twomass_no_pm, simbad.epoch) is None
    assert resolve_epoch_range(twomass_no_pm, simbad) == (1997.4, 2001.2)
    assert resolve_epoch(unknown, simbad.epoch) is None and resolve_epoch_range(unknown, simbad) == (1990.0, 2025.0)


async def test_simbad_j2000_error_includes_the_propagation_from_gaia() -> None:
    # HD 209458: SIMBAD coo_err 0.0213/0.0227 mas equals Gaia's J2016 error, but the
    # position is at J2000: true error sqrt(0.0213^2 + (16 x 0.0277)^2) = 0.44 mas.
    simbad, _ = await run_catalog("simbad", "hd209458")
    star = next(s for s in simbad if s.source_id == "HD 209458")
    assert {star.data["coo_err_maj"], star.data["coo_err_min"]} == {0.0213, 0.0227}
    assert 0.40e-3 < star.positional_error_arcsec < 0.60e-3  # estimate 0.51 mas (conservative)
    growth = star.metadata["positional_error"]["epoch_growth"]
    assert growth["source_epoch"] == 2016.0 and "estimated" in growth["note"]


# ---------------------------------------------------------------------------
# VLASS redundant duplicates
# ---------------------------------------------------------------------------


async def test_vlass_redundant_duplicate_is_removed() -> None:
    # Reported: J025527.22+055919.5 came back twice (Flag 2 'Redundant' at 0.000" and the
    # Flag 1 row at 0.047"), inflating counts and making the match look ambiguous.
    sources, failure = await run_catalog("vlass", "vlass_duplicate", radius_arcsec=5.0)
    assert failure is None
    assert sources.meta["raw_row_count"] == 2 and sources.meta["filtered_rows"] == 1
    assert [s.data["Flag"] for s in sources] == [1]
    assert sources.meta["row_count"] == 1
    record = await crossmatch_recorded("vlass_duplicate", ["vlass"])
    stats = record.provenance["catalog_stats"]["vlass"]
    assert (stats["row_count"], stats["filtered_rows"], stats["raw_row_count"]) == (1, 1, 2)
    assert len(record.provenance["matches"]) == 1


# ---------------------------------------------------------------------------
# Proper-motion adoption in crowded fields
# ---------------------------------------------------------------------------


def _pm_row(sid: str, sep_arcsec: float, pm: tuple[float, float], catalog: str = "simbad", **kw) -> CatalogSource:
    return source(catalog, sid, 266.4168, -29.0078 + sep_arcsec / 3600.0, epoch=2000.0,
                  proper_motion_ra_masyr=pm[0], proper_motion_dec_masyr=pm[1],
                  metadata={"physical": kw.pop("physical", {"object_type": "*"})}, **kw)


def test_adoption_refuses_a_chance_alignment_in_a_crowded_field() -> None:
    # Sgr A*: SIMBAD has ~200 objects within 2"; the nearest S-star's motion was adopted.
    target = validate_target(266.4168, -29.0078, epoch=2016.0)
    rows = [_pm_row("[EG97] S31", 0.098, (9.965, -20.821)), _pm_row("S2", 0.15, (-3.0, 40.0)),
            _pm_row("S5", 0.4, (25.0, 5.0))]
    warnings: list[str] = []
    assert _adopt_proper_motion(target, [("simbad", rows)], 30.0, warnings=warnings) is None
    assert warnings and "ambiguous" in warnings[0] and "[EG97] S31" in warnings[0]


def test_adoption_accepts_agreeing_candidates_and_distant_rivals() -> None:
    target = validate_target(266.4168, -29.0078, epoch=2000.0)  # (rows are placed around this position)
    star = _pm_row("NAME Barnard's star", 0.0, (-801.551, 10362.394))
    gaia = _pm_row("4472832130942575872", 0.01, (-801.55, 10362.39), catalog="gaia_dr3",
                   data={"parallax": 546.98, "parallax_error": 0.04})
    rival = _pm_row("field", 1.5, (2.0, -3.0))  # beyond 2 x 0.0 + 0.5"
    warnings: list[str] = []
    adopted, origin = _adopt_proper_motion(target, [("simbad", [star, rival]), ("gaia_dr3", [gaia])], 10.0,
                                           warnings=warnings)
    assert warnings == [] and origin["source_id"] == "NAME Barnard's star"
    assert adopted.proper_motion == (-801.551, 10362.394)
    assert adopted.parallax_mas == pytest.approx(546.98)
    assert origin["parallax"]["catalog"] == "gaia_dr3"


async def test_crowded_field_adoption_warning_reaches_provenance() -> None:
    from models import catalog_from_dict

    cat = catalog_from_dict("crowd", {
        "provider": "tap", "wavelength": "x", "endpoint": "https://crowd.example/tap", "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["id", "ra", "dec", "pmra", "pmdec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0, "profiles": ["full"],
    })
    reg = CatalogRegistry()
    reg._catalogs = {"crowd": cat}
    meta = [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"},
            {"name": "pmra", "unit": "mas/yr"}, {"name": "pmdec", "unit": "mas/yr"}]
    data = [["a", 266.4168, -29.0078 + 0.1 / 3600, 10.0, -20.0], ["b", 266.4168, -29.0078 + 0.2 / 3600, -5.0, 40.0]]
    with respx.mock() as router:
        router.post("https://crowd.example/tap").respond(json={"metadata": meta, "data": data})
        async with offline_client() as client:
            record = await make_service(client, reg).crossmatch(266.4168, -29.0078, radius_arcsec=30.0, epoch=2016.0)
    assert record.provenance["target_proper_motion"] is None
    assert any("not adopted" in w for w in record.provenance["warnings"])


# ---------------------------------------------------------------------------
# Proper motions in seconds of time
# ---------------------------------------------------------------------------


def test_ra_proper_motion_in_time_seconds_is_converted_with_cos_dec() -> None:
    # Reported: pmRA -0.053 s/a (SAO-style) came out as -0.053 mas/yr (should be ~ -792).
    columns = [ColumnMeta("RAJ2000", "deg", "pos.eq.ra;meta.main"), ColumnMeta("DEJ2000", "deg", "pos.eq.dec;meta.main"),
               ColumnMeta("pmRA", "s/a", "pos.pm;pos.eq.ra"), ColumnMeta("pmDE", "arcsec/a", "pos.pm;pos.eq.dec")]
    row = {"RAJ2000": 10.0, "DEJ2000": 4.69, "pmRA": -0.053, "pmDE": 10.33}
    values = normalize_source_record(row, columns=columns)
    assert values["pmra"] == pytest.approx(-0.053 * 15000.0 * math.cos(math.radians(4.69)), rel=1e-9)  # -792.3
    assert values["pmdec"] == pytest.approx(10330.0)
    assert pm_to_masyr(-0.053, "s/yr", 60.0) == pytest.approx(-0.053 * 15000.0 * 0.5)
    assert pm_to_masyr(0.1, "s/ha", 0.0) == pytest.approx(0.1 * 15000.0 / 100.0)  # per century (FK4)
    assert pm_to_masyr(0.1, "s/cy", 0.0) == pytest.approx(15.0)
    assert pm_to_masyr(2.0, "ms/a", 0.0) == pytest.approx(30.0)
    assert pm_to_masyr(1.0, "arcsec/cy") == pytest.approx(10.0)
    assert pm_to_masyr(-0.053, "s/a") is None  # no declination: cannot convert
    assert pm_to_masyr(5.0, "furlong/fortnight") is None  # unknown unit: dropped, not mislabelled
    assert pm_to_masyr(5.0, "km/s") is None
    assert pm_to_masyr(5.0, None) == 5.0
