"""Offline tests of the Bayesian N-way matching engine (astrometry.py + crossmatch.py).

* Bayes factors against the closed forms of Budavari & Szalay (2008, ApJ 679, 301).
* Monte-Carlo skies with known truth: completeness, purity and calibration of the
  posteriors (reliability curve), and purity of the object partition.
* Epoch propagation: proper-motion uncertainty growth, error ellipses, radio structure.
* The crossmatch service on recorded real archive responses (3C 273, Barnard's star).
* Benchmark: grouping 5000 detections.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

import numpy as np
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect, target_for, target_radius
from helpers import make_service, offline_client

import astrometry
from astrometry import (
    ARCSEC_PER_RAD,
    ASTROMETRIC_FLOOR_ARCSEC,
    CATALOG_SKY_DENSITY,
    FULL_SKY_DEG2,
    AssociationConfig,
    Detection,
    SimCatalog,
    associate,
    bayes_factor_2way,
    calibration_run,
    cone_area_deg2,
    ellipse_covariance,
    estimate_density_deg2,
    ln_bayes_factor,
    log10_bayes_factor,
    posterior_probability,
    source_covariance,
    source_detection,
    structure_covariance,
)
from crossmatch import CrossmatchService, _group_matches, match_target
from models import CatalogRegistry, CatalogSource, offset_radec, validate_target
from providers import QueryResult

# ---------------------------------------------------------------------------
# Bayes factors (B&S 2008)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("psi", "s1", "s2"), [(0.0, 0.1, 0.1), (0.3, 0.1, 0.2), (1.7, 0.5, 1.2), (4.0, 2.0, 3.0)])
def test_two_way_bayes_factor_is_budavari_szalay_eq16(psi: float, s1: float, s2: float) -> None:
    # Hand computation in radians: B = 2 / (s1^2 + s2^2) exp(-psi^2 / (2 (s1^2 + s2^2))).
    r = ARCSEC_PER_RAD
    s = (s1 / r) ** 2 + (s2 / r) ** 2
    expected = 2.0 / s * math.exp(-((psi / r) ** 2) / (2.0 * s))
    assert bayes_factor_2way(psi, s1, s2) == pytest.approx(expected, rel=1e-12)
    general = ln_bayes_factor([(0.0, 0.0), (psi * 0.6, psi * 0.8)], [(s1**2, 0.0, s1**2), (s2**2, 0.0, s2**2)])
    assert general == pytest.approx(math.log(expected), abs=1e-9)
    assert log10_bayes_factor([(0.0, 0.0), (psi, 0.0)], [(s1**2, 0, s1**2), (s2**2, 0, s2**2)]) == pytest.approx(
        math.log10(expected), abs=1e-9)


def test_three_way_bayes_factor_is_budavari_szalay_eq17() -> None:
    # B&S eq. (17): B = 4 / (s1^2 s2^2 + s2^2 s3^2 + s3^2 s1^2)
    #   x exp(-(s3^2 p12^2 + s1^2 p23^2 + s2^2 p31^2) / (2 (s1^2 s2^2 + s2^2 s3^2 + s3^2 s1^2))).
    pos = [(0.0, 0.0), (0.4, -0.1), (0.1, 0.5)]
    sig = [0.2, 0.3, 0.5]
    r = ARCSEC_PER_RAD
    s1, s2, s3 = (x / r for x in sig)
    p = [[math.dist(a, b) / r for b in pos] for a in pos]
    s = s1**2 * s2**2 + s2**2 * s3**2 + s3**2 * s1**2
    q = s3**2 * p[0][1] ** 2 + s1**2 * p[1][2] ** 2 + s2**2 * p[2][0] ** 2
    expected = math.log(4.0 / s) - q / (2.0 * s)
    assert ln_bayes_factor(pos, [(x * x, 0.0, x * x) for x in sig]) == pytest.approx(expected, abs=1e-9)


def test_elliptical_bayes_factor_is_rotation_invariant_and_favours_the_major_axis() -> None:
    cov = ellipse_covariance(1.0, 0.1, 30.0)  # 1" x 0.1" ellipse, major axis 30 deg east of north
    t = math.radians(30.0)
    along = (0.8 * math.sin(t), 0.8 * math.cos(t))
    across = (0.8 * math.cos(t), -0.8 * math.sin(t))
    point = (0.01**2, 0.0, 0.01**2)
    b_along = ln_bayes_factor([(0.0, 0.0), along], [point, cov])
    b_across = ln_bayes_factor([(0.0, 0.0), across], [point, cov])
    assert b_along > b_across + 20.0  # 0.8" along a 1" axis is fine, across a 0.1" axis is 8 sigma
    # Rotating the whole configuration changes nothing.
    rot = math.radians(71.0)
    c, s = math.cos(rot), math.sin(rot)
    rotated_cov = ellipse_covariance(1.0, 0.1, 30.0 + 71.0)
    rx = (c * along[0] + s * along[1], -s * along[0] + c * along[1])
    assert ln_bayes_factor([(0.0, 0.0), rx], [point, rotated_cov]) == pytest.approx(b_along, abs=1e-6)
    assert math.isclose((cov[0] + cov[2]) / 2.0, (1.0 + 0.01) / 2.0)


def test_posterior_is_budavari_szalay_eq22() -> None:
    for log10_b, p0 in [(3.0, 1e-4), (10.0, 1e-9), (-2.0, 0.3), (0.0, 0.5)]:
        b = 10.0**log10_b
        assert posterior_probability(log10_b, p0) == pytest.approx(1.0 / (1.0 + (1.0 - p0) / (b * p0)), rel=1e-12)
    assert posterior_probability(400.0, 1e-9) == 1.0  # no overflow


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------


def test_every_registry_catalogue_has_a_documented_sky_density() -> None:
    registry = CatalogRegistry()
    assert set(registry.catalogs) <= set(CATALOG_SKY_DENSITY)
    for name, density in CATALOG_SKY_DENSITY.items():
        assert density.sources > 0 and 0 < density.area_deg2 <= FULL_SKY_DEG2 + 1e-6, name
        assert density.reference, name
    # Gaia DR3: 1,811,709,771 sources over the whole sky = 43,917 per deg^2.
    assert CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2 == pytest.approx(43917.4, rel=1e-4)


def test_gamma_poisson_density_tends_to_sky_mean_in_tiny_cones_and_local_count_in_big_ones() -> None:
    mean = CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2
    # A 3" cone holding only the target's own row: an empty 28 arcsec^2 barely moves the
    # prior (1 pseudo-source over 1 / 43,917 deg^2), so the estimate stays near the mean.
    tiny, info = estimate_density_deg2(1, cone_area_deg2(3.0), catalog="gaia_dr3", n_target_rows=1)
    assert tiny == pytest.approx(mean, rel=0.1) and tiny < mean and info["field_rows"] == 0
    area = cone_area_deg2(600.0)
    big, _ = estimate_density_deg2(5000, area, catalog="gaia_dr3", n_target_rows=1)
    assert big == pytest.approx(4999 / area, rel=0.01)
    # Exact formula.
    a = 1.0
    value, _ = estimate_density_deg2(7, 0.01, catalog="gaia_dr3")
    assert value == pytest.approx((a + 7) / (a / mean + 0.01))
    # Without an all-sky density: NWAY's (n + 1) / area.
    other, info = estimate_density_deg2(3, 0.02, catalog="not_a_catalogue")
    assert other == pytest.approx(4 / 0.02) and info["method"] == "nway_plus_one"
    theta = math.radians(1.0)
    assert cone_area_deg2(3600.0) == pytest.approx(2 * math.pi * (1 - math.cos(theta)) * (180 / math.pi) ** 2, rel=1e-12)
    assert cone_area_deg2(3600.0) == pytest.approx(math.pi, rel=1e-4)


# ---------------------------------------------------------------------------
# Monte-Carlo skies with known truth
# ---------------------------------------------------------------------------

SCENARIOS = {
    # Sparse precise optical + IR + a coarse X-ray-like catalogue.
    "sparse": [SimCatalog("opt", 0.1, 5.0, 0.7), SimCatalog("ir", 0.5, 2.0, 0.7), SimCatalog("xray", 2.0, 0.3, 0.7)],
    # Crowded: nu * pi * (3 sigma)^2 ~ 0.06-0.5 per catalogue, lower target completeness.
    "crowded": [SimCatalog("a", 0.3, 20.0, 0.5), SimCatalog("b", 1.0, 10.0, 0.5), SimCatalog("c", 4.0, 1.0, 0.5),
                SimCatalog("d", 0.05, 20.0, 0.5)],
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_monte_carlo_completeness_purity_and_calibration(scenario: str) -> None:
    report = calibration_run(SCENARIOS[scenario], fields=600, seed=20260928 + len(scenario), threshold=0.9)
    assert report["true_target_detections"] > 500
    # At P > 0.9: almost every selected association is true, and most true ones are selected.
    assert report["purity"] > 0.97, report
    assert report["completeness"] > (0.95 if scenario == "sparse" else 0.85), report
    # Calibration: in every posterior bin the observed fraction of true associations
    # matches the mean posterior within binomial noise (3 sigma) + 0.05.
    for row in report["reliability"]:
        p, n = row["mean_probability"], row["count"]
        sigma = math.sqrt(max(p * (1 - p), 0.01) / n)
        assert abs(row["true_fraction"] - p) <= 3.0 * sigma + 0.05, (scenario, row)
    # The object partition of the other detections is pure (members = one true object).
    assert report["field_group_purity"] > 0.95, report


def test_monte_carlo_expected_number_of_true_associations() -> None:
    # Sum of posteriors = expected number of true associations (global calibration).
    rng = np.random.default_rng(7)
    cats = SCENARIOS["crowded"]
    densities = {c.name: c.density_arcmin2 * 3600.0 for c in cats}
    cfg = AssociationConfig(completeness={c.name: c.target_completeness for c in cats})
    total_p = total_true = 0.0
    for _ in range(300):
        dets, truth, target = astrometry.simulate_field(rng, cats)
        result = associate(dets, densities, target=target, config=cfg)
        total_p += float(result.target_probability.sum())
        total_true += sum(1 for t in truth if t == 0)
    assert total_p == pytest.approx(total_true, rel=0.05)


def test_each_catalogue_contributes_at_most_one_source_per_object() -> None:
    rng = np.random.default_rng(11)
    cats = SCENARIOS["crowded"]
    densities = {c.name: c.density_arcmin2 * 3600.0 for c in cats}
    for _ in range(50):
        dets, _truth, target = astrometry.simulate_field(rng, cats)
        result = associate(dets, densities, target=target)
        seen = set()
        for group in result.groups:
            cats_in = [dets[m].catalog for m in group.members]
            assert len(cats_in) == len(set(cats_in))
            seen.update(group.members)
        assert seen == set(range(len(dets)))  # a partition


def test_no_counterpart_is_a_hypothesis() -> None:
    # A target with nothing near it: p_any ~ 0 and no row is its counterpart.
    dets = [Detection("opt", *offset_radec(150.0, 20.0, 5.0, 0.0), (0.01, 0.0, 0.01))]
    result = associate(dets, {"opt": 3600.0 * 5}, target=(150.0, 20.0))
    assert result.p_any < 1e-6 and result.target_probability[0] < 1e-6
    # The same row at the target position is its counterpart.
    dets = [Detection("opt", 150.0, 20.0, (0.01, 0.0, 0.01))]
    result = associate(dets, {"opt": 3600.0 * 5}, target=(150.0, 20.0))
    assert result.p_any > 0.99 and result.groups[0].contains_target and result.groups[0].match_flag == "best"


def test_ambiguous_counterparts_share_the_probability_and_are_flagged_secondary() -> None:
    # Two equally good optical candidates for an X-ray position: ~1/2 each.
    ra, dec = 150.0, 20.0
    a = Detection("opt", *offset_radec(ra, dec, 0.5, 0.0), (0.01, 0.0, 0.01))
    b = Detection("opt", *offset_radec(ra, dec, -0.5, 0.0), (0.01, 0.0, 0.01))
    result = associate([a, b], {"opt": 100.0}, target=(ra, dec), config=AssociationConfig(target_sigma_arcsec=2.0))
    p = result.target_probability
    assert p[0] == pytest.approx(p[1], rel=1e-6) and p[0] + p[1] == pytest.approx(result.p_any, rel=1e-9)
    target_group = next(g for g in result.groups if g.contains_target)
    other = next(g for g in result.groups if not g.contains_target)
    assert target_group.match_flag == "best" and other.match_flag == "secondary"
    assert target_group.p_i == pytest.approx(0.5, abs=1e-6) and len(target_group.alternatives) == 1


def test_coincident_rows_of_one_catalogue_follow_their_representative() -> None:
    # SIMBAD lists a star and its planets at one position.
    star = Detection("simbad", 150.0, 20.0, (1e-4, 0.0, 1e-4), priority=0)
    planet = Detection("simbad", 150.0, 20.0, (1e-4, 0.0, 1e-4), priority=1)
    gaia = Detection("gaia_dr3", 150.0, 20.0, (1e-6, 0.0, 1e-6))
    result = associate([planet, star, gaia], {"simbad": 500.0, "gaia_dr3": 40000.0}, target=(150.0, 20.0))
    group = result.groups[0]
    assert group.contains_target and sorted(group.members) == [0, 1, 2]
    assert group.coincident_with == {0: 1}  # the planet follows the star
    assert result.target_probability[0] == result.target_probability[1] > 0.99


# ---------------------------------------------------------------------------
# Detections: epoch propagation, ellipses, structure
# ---------------------------------------------------------------------------


def _row(catalog: str, ra: float, dec: float, err: float | None, **kw) -> CatalogSource:
    metadata = kw.pop("metadata", {})
    data = kw.pop("data", {})
    return CatalogSource(catalog, kw.pop("source_id", "x"), ra, dec, err, data, metadata, {}, **kw)


def test_proper_motion_uncertainty_grows_with_the_epoch_difference() -> None:
    target = validate_target(10.0, 20.0, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=500.0)
    gaia = _row("gaia_dr3", 10.0, 20.0 + 8.0 / 3600, 0.0002, epoch=2016.0, proper_motion_ra_masyr=0.0,
                proper_motion_dec_masyr=500.0)
    det, info = source_detection(gaia, target)
    assert info["propagation"] == "source_pm"
    # pm error = 1.4 x 0.2 mas = 0.28 mas/yr (Gaia DR3 ratio) over 16 yr = 4.5 mas.
    assert info["pm_growth_arcsec"] == pytest.approx(1.4 * 0.0002 * 16.0, rel=1e-9)
    assert math.hypot(det.ra - 10.0, det.dec - 20.0) * 3600 < 1e-6  # back at the J2000 position
    twomass = _row("twomass_psc", 10.0, 20.0 + 0.5 / 3600, 0.07, epoch=2001.0)
    _, info2 = source_detection(twomass, target, target_pm_sigma_masyr=2.0)
    assert info2["propagation"] == "target_pm" and info2["pm_growth_arcsec"] == pytest.approx(0.002)
    field = _row("gaia_dr3", 10.01, 20.0, 0.001, epoch=2016.0, metadata={"catalog_has_proper_motions": True})
    _, info3 = source_detection(field, target)
    assert info3["propagation"] == "stationary"
    assert info3["pm_growth_arcsec"] == pytest.approx(astrometry.UNKNOWN_PM_SIGMA_MASYR * 16.0 / 1000.0)
    expected = 0.001**2 + ASTROMETRIC_FLOOR_ARCSEC**2 + (astrometry.UNKNOWN_PM_SIGMA_MASYR * 0.016) ** 2
    assert info3["sigma_arcsec"] == pytest.approx(math.sqrt(expected))


def test_catalogue_ellipse_and_gaia_correlation_become_covariances() -> None:
    twomass = _row("twomass_psc", 1.0, 1.0, math.sqrt((0.17**2 + 0.08**2) / 2), metadata={"positional_error": {
        "ellipse_1sigma_arcsec": {"major": 0.17, "minor": 0.08, "pa_deg": 90.0}}})
    cov, shape = source_covariance(twomass)
    assert shape == "ellipse" and cov[0] == pytest.approx(0.17**2) and cov[2] == pytest.approx(0.08**2)
    gaia = _row("gaia_dr3", 1.0, 1.0, math.sqrt((0.0003**2 + 0.0002**2) / 2), data={"ra_dec_corr": 0.5},
                metadata={"positional_error": {"sigma_ra_arcsec": 0.0003, "sigma_dec_arcsec": 0.0002}})
    cov, shape = source_covariance(gaia)
    assert shape == "axes_corr" and cov[1] == pytest.approx(0.5 * 0.0003 * 0.0002)
    # A systematic term in positional_error_arcsec is added isotropically (trace preserved).
    rosat = _row("rosat", 1.0, 1.0, 3.755, metadata={"positional_error": {"sigma_ra_arcsec": 0.687,
                                                                          "sigma_dec_arcsec": 0.689}})
    cov, _ = source_covariance(rosat)
    assert (cov[0] + cov[2]) / 2 == pytest.approx(3.755**2)


def test_resolved_radio_source_gets_its_structure_as_host_offset_scatter() -> None:
    # NVSS J122906+020305 (3C 273): deconvolved 21.5" x 15.9" at PA 44.3 deg.
    nvss = _row("nvss", 187.2767, 2.0514, 0.53, data={"major_axis": 21.5, "minor_axis": 15.9, "position_angle": 44.3})
    cov = structure_covariance(nvss)
    sig = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    assert (cov[0] + cov[2]) == pytest.approx((21.5 * sig) ** 2 + (15.9 * sig) ** 2)
    # FIRST sizes are beam-convolved: a source as large as the 5.4" beam has no structure.
    first = _row("first", 1.0, 1.0, 0.2, data={"fit_major_axis": 5.4, "fit_minor_axis": 5.4})
    assert structure_covariance(first) is None
    # LoTSS-DR3 uses the per-row resolution ('Res' = 9" below Dec 10 deg).
    lotss = _row("lotss", 1.0, 1.0, 2.0, data={"Maj": 15.0, "Min": 9.0, "Res": 9})
    cov = structure_covariance(lotss)
    assert cov[0] + cov[2] == pytest.approx((12.0 * sig) ** 2)
    assert structure_covariance(_row("gaia_dr3", 1.0, 1.0, 0.001)) is None


# ---------------------------------------------------------------------------
# Crossmatch service on recorded archive responses
# ---------------------------------------------------------------------------

THREE_C_273 = (187.2779154, 2.0523883)
SPEC_3C273 = {"gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2", "sdss", "first", "nvss", "rosat", "chandra", "xmm"}


async def _recorded(key: str, *, catalogs=None, **kwargs):
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(key, catalogs)))
        async with offline_client() as client:
            service = make_service(client)
            t = target_for(key)
            return await service.crossmatch(t.ra, t.dec, radius_arcsec=target_radius(key), epoch=t.epoch,
                                            pm_ra_masyr=t.pm_ra_masyr, pm_dec_masyr=t.pm_dec_masyr,
                                            catalogs=catalogs, **kwargs)


async def test_3c273_recorded_one_group_across_the_spectrum() -> None:
    record = await _recorded("3c273")
    target_group = record.crossmatch_groups[0]
    assert target_group["contains_target"] and target_group["match_flag"] == "best"
    assert target_group["group_id"] == "object-1" and target_group["p_any"] > 0.999
    assert SPEC_3C273 <= set(target_group["catalogs"])
    by_catalog = {m["catalog"]: m for m in target_group["members"] if m["coincident_with"] is None}
    assert len(by_catalog) == len([m for m in target_group["members"] if m["coincident_with"] is None])
    for name in SPEC_3C273 | {"simbad", "ned"}:
        member = by_catalog[name]
        assert member["match_probability"] > 0.95 and member["confidence"] == member["match_probability"], name
    assert by_catalog["simbad"]["source_id"] == "3C 273" and by_catalog["ned"]["source_id"] == "3C 273"
    # The radio centroids are pulled towards the jet (NVSS 5.6", LoTSS 7.0") and are still
    # identified thanks to their resolved structure.
    assert by_catalog["nvss"]["separation_arcsec"] > 5.0
    assert "structure" in by_catalog["nvss"]["position_at_epoch"]["covariance_shape"]
    # The NED absorption-line systems at 0.8" (18 entries at one position) are not 3C 273.
    absorbers = [m for g in record.crossmatch_groups for m in g["members"] if "ABS" in m["source_id"]]
    assert len(absorbers) == 18 and all(m["target_probability"] < 1e-3 for m in absorbers)
    assert all(not g["contains_target"] for g in record.crossmatch_groups[1:])
    # Provenance of the association.
    assoc = record.provenance["association"]
    assert assoc["p_any"] > 0.999 and assoc["densities"]["gaia_dr3"]["method"] == "gamma_poisson"
    assert assoc["target_sigma_arcsec"] == astrometry.DEFAULT_TARGET_SIGMA_ARCSEC
    confidences = {(m["catalog"], m["source_id"]): m["confidence"] for m in record.provenance["matches"]}
    assert confidences[("gaia_dr3", "3700386905605055360")] > 0.99
    # Pan-STARRS objects 6.9" and 8.8" away are unrelated.
    assert confidences[("panstarrs_dr2", "110461872781485699")] < 1e-3


async def test_barnard_recorded_j2000_with_gaia_motion_groups_despite_the_motion() -> None:
    record = await _recorded("barnard_j2000_pm", catalogs=["gaia_dr3", "twomass_psc", "allwise", "simbad"])
    group = record.crossmatch_groups[0]
    assert group["contains_target"] and {"gaia_dr3", "twomass_psc", "allwise", "simbad"} <= set(group["catalogs"])
    assert group["p_any"] > 0.99
    for member in group["members"]:
        if member["coincident_with"] is None:
            assert member["match_probability"] > 0.95, member["catalog"]
    gaia = next(m for m in group["members"] if m["catalog"] == "gaia_dr3")
    # The Gaia row (J2016) is 166" from the J2000 position before propagation.
    assert gaia["metadata"]["query_separation_arcsec"] > 150.0
    assert gaia["position_at_epoch"]["propagation"] == "source_pm" and gaia["separation_arcsec"] < 0.05
    # Planets listed by SIMBAD at the star's position follow the star.
    planets = [m for m in group["members"] if m["coincident_with"] is not None]
    assert planets and all(m["coincident_with"] == "NAME Barnard's star" for m in planets)


async def test_target_uncertainty_parameter_changes_the_posterior() -> None:
    tight = await _recorded("3c273", catalogs=["first"], target_uncertainty_arcsec=0.01)
    loose = await _recorded("3c273", catalogs=["first"], target_uncertainty_arcsec=1.0)
    assert tight.provenance["association"]["target_sigma_arcsec"] == 0.01
    assert loose.provenance["association"]["target_sigma_arcsec"] == 1.0
    assert tight.provenance["association"]["config"]["target_sigma_arcsec"] == 0.01
    with pytest.raises(ValueError, match="target_uncertainty_arcsec"):
        await _recorded("3c273", catalogs=["first"], target_uncertainty_arcsec=-1.0)
    with pytest.raises(ValueError, match="Unknown catalog"):
        await _recorded("3c273", catalogs=["first", "not_a_catalog"])


def test_group_matches_standalone_keeps_match_confidence() -> None:
    # Callers of _group_matches (e.g. the VO server) keep their own Match.confidence.
    target = validate_target(150.0, 20.0)
    rows = [_row("gaia_dr3", 150.0, 20.0, 0.001, source_id="g", metadata={"wavelength": "optical"}),
            _row("twomass_psc", *offset_radec(150.0, 20.0, 0.05, 0.0), 0.07, source_id="t",
                 metadata={"wavelength": "infrared"}),
            _row("twomass_psc", *offset_radec(150.0, 20.0, 5.0, 0.0), 0.07, source_id="far",
                 metadata={"wavelength": "infrared"})]
    matches = match_target(target, rows, 10.0)
    groups = _group_matches(matches, target, 10.0)
    assert [sorted(m["source_id"] for m in g["members"]) for g in groups] == [["g", "t"], ["far"]]
    gaussian = {m.source.source_id: m.confidence for m in matches}
    for group in groups:
        for member in group["members"]:
            assert member["confidence"] == gaussian[member["source_id"]]
    assert groups[0]["contains_target"] and groups[0]["members"][0]["match_probability"] > 0.99


# ---------------------------------------------------------------------------
# crossmatch_many concurrency
# ---------------------------------------------------------------------------


class _CountingProvider:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0

    async def query(self, catalog, target, radius_arcsec):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        src = CatalogSource(catalog.name, f"{target.ra:.4f}", target.ra, target.dec, 0.01, {},
                            {"wavelength": catalog.wavelength}, {})
        return QueryResult([src], {"max_rows": catalog.max_rows, "row_limit": catalog.max_rows})


def _one_catalog_registry() -> CatalogRegistry:
    registry = CatalogRegistry()
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3")}
    return registry


async def test_crossmatch_many_is_concurrent_bounded_and_ordered() -> None:
    provider = _CountingProvider(0.2)
    service = CrossmatchService(_one_catalog_registry(), {"tap": provider}, max_concurrency=3)
    targets = [{"ra": 10.0 + i, "dec": 5.0} for i in range(7)]
    started = time.perf_counter()
    records = await service.crossmatch_many(targets, radius_arcsec=2.0)
    elapsed = time.perf_counter() - started
    assert [r.target["ra"] for r in records] == [t["ra"] for t in targets]
    assert provider.peak == 3
    assert elapsed < 7 * 0.2 * 0.7  # concurrent (sequential would take 1.4 s)
    with pytest.raises(ValueError):
        await service.crossmatch_many(targets, max_concurrency=0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _dense_matches(n_per_catalog: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    target = validate_target(201.69683, -47.47958)
    cats = [("gaia_dr3", 0.0005), ("twomass_psc", 0.07), ("allwise", 0.2), ("panstarrs_dr2", 0.02), ("sdss", 0.08)]
    rows = []
    # Stars shared between catalogues (with catalogue errors) plus unrelated ones.
    stars = rng.uniform(-110, 110, size=(n_per_catalog, 2))
    for name, err in cats:
        for i, (x, y) in enumerate(stars):
            if rng.random() < 0.3:  # an object only this catalogue sees
                x, y = rng.uniform(-110, 110, 2)
            ra, dec = offset_radec(target.ra, target.dec, x + rng.normal(0, err), y + rng.normal(0, err))
            rows.append(CatalogSource(name, f"{name}-{i}", ra, dec, err, {}, {"wavelength": "optical"}, {}))
    return target, match_target(target, rows, 160.0)


def test_benchmark_grouping_5000_sources_under_one_second() -> None:
    target, matches = _dense_matches(1000)
    assert len(matches) == 5000
    _group_matches(matches[:50], target, 160.0)  # warm-up (imports, caches)
    started = time.perf_counter()
    groups = _group_matches(matches, target, 160.0)
    elapsed = time.perf_counter() - started
    print(f"\nBENCHMARK grouping 5000 detections in {len(groups)} objects: {elapsed * 1000:.0f} ms")
    assert elapsed < 1.0
    assert sum(len(g["members"]) for g in groups) == 5000
    for g in groups:
        assert len(g["catalogs"]) == len(g["members"])  # one row per catalogue per object
    # Most multi-catalogue stars are recovered whole.
    assert sum(1 for g in groups if len(g["members"]) >= 3) > 300


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_xmatch_calibrate(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    astrometry.register_cli(sub)
    args = parser.parse_args(["xmatch-calibrate", "--fields", "40", "--seed", "5"])
    assert args.handler(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fields"] == 40 and report["purity"] > 0.9 and report["reliability"]
