"""Regression tests for the adversarial review of the Bayesian matching engine (round 1).

Recorded real archive cones (``tests/fixtures/matching/<key>``, recorded with
``tests/test_matching_fixtures.py record``) are replayed offline; synthetic detections
check the engine's arithmetic against closed forms.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import respx

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_matching_fixtures import (
    MATCHING_TARGETS,
    SESAME_DIR,
    _register,
    assert_replayed_exactly,
    replay_crossmatch,
)

import astrometry
from astrometry import (
    CATALOG_SKY_DENSITY,
    FULL_SKY_DEG2,
    AssociationConfig,
    Detection,
    associate,
    completeness_prior,
    cone_area_deg2,
    estimate_density_deg2,
    gnomonic_jacobian,
    healpix_nested_index,
    local_prior_density,
    pm_sigma_masyr,
    source_detection,
    structure_covariance,
)
from crossmatch import (
    AdvancedQuery,
    CrossmatchService,
    _adopted_pm_sigma,
    _resolver_pm_error,
    _resolver_position_error,
    catalog_densities,
    coordinate_sigma_arcsec,
    resolved_search_target,
)
from models import (
    CatalogRegistry,
    CatalogSource,
    InvalidCoordinateError,
    offset_radec,
    propagate_radec,
    validate_target,
)
from providers import QueryResult, SesameResolver


def members_by_id(record) -> dict[str, dict]:
    return {m["source_id"]: m for g in record.crossmatch_groups for m in g["members"]}


def target_group(record) -> dict | None:
    return next((g for g in record.crossmatch_groups if g["contains_target"]), None)


def row(catalog: str, ra: float, dec: float, err: float | None, **kw) -> CatalogSource:
    metadata = kw.pop("metadata", {})
    data = kw.pop("data", {})
    return CatalogSource(catalog, kw.pop("source_id", "x"), ra, dec, err, data, metadata, {}, **kw)


# ---------------------------------------------------------------------------
# Issue: undated coordinates of moving stars (identity rows got P = 0)
# ---------------------------------------------------------------------------

UNDATED = {
    # key: SIMBAD identity row
    "barnard_undated": "NAME Barnard's star",
    "tau_ceti_undated": "* tau Cet",
    "61cyga_undated": "* 61 Cyg A",
}


@pytest.mark.parametrize("key", sorted(UNDATED))
async def test_undated_simbad_coordinates_identify_the_moving_star(key: str) -> None:
    record = await replay_crossmatch(key)
    assert record.target["epoch"] is None
    members = members_by_id(record)
    identity = members[UNDATED[key]]
    # The SIMBAD row at 0" (J2000, with its own proper motion) is the star: before, it was
    # moved to J2016 while the target stayed put, and got P = 0.
    assert identity["separation_arcsec"] < 0.05
    assert identity["target_probability"] > 0.95, identity["target_probability"]
    assert identity["position_at_epoch"]["propagation"] == "source_pm_undated"
    group = target_group(record)
    assert group is not None and any(m["source_id"] == UNDATED[key] for m in group["members"])
    assert record.provenance["association"]["undated_target_epochs"] == [2000.0, 2016.0]


async def test_61_cyg_a_component_beats_the_pm_less_system_entry() -> None:
    record = await replay_crossmatch("61cyga_undated")
    members = members_by_id(record)
    assert members["* 61 Cyg A"]["target_probability"] > 0.95
    assert members["* 61 Cyg"]["target_probability"] < 0.05


async def test_undated_identity_row_survives_the_search_confidence_filter() -> None:
    # POST /api/v1/search builds an AdvancedQuery (min_confidence 0.5): the SIMBAD rows of
    # tau Ceti must be returned (before: 0 of 8 in-cone rows, returned_count 0).
    spec = MATCHING_TARGETS["tau_ceti_undated"]
    query = AdvancedQuery.from_dict({"ra": spec["ra"], "dec": spec["dec"], "radius_arcsec": spec["radius_arcsec"],
                                     "catalogs": spec["catalogs"], "min_confidence": 0.5})
    record = await replay_crossmatch("tau_ceti_undated", query=query, catalogs=None, radius_arcsec=None)
    simbad = record.catalog_results["simbad"]
    assert simbad["row_count"] == 8
    assert "* tau Cet" in {s.source_id for s in simbad["sources"]}
    assert simbad["returned_count"] >= 1
    assert any(c["source_id"] == "* tau Cet" for rows in record.counterparts.values() for c in rows)


async def test_undated_synthetic_vega_like_star_keeps_its_simbad_row_through_search() -> None:
    # A SIMBAD J2000 row with pm (200.94, 286.23) mas/yr at the searched position, its Gaia
    # J2016 row 5.6" away and a 2MASS row: the SIMBAD row is the target's counterpart.
    ra, dec = 279.23473479, 38.78368896
    pm = (200.94, 286.23)
    gra, gdec = propagate_radec(ra, dec, pm[0], pm[1], 2000.0, 2016.0)

    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            if catalog.name == "simbad":
                src = row("simbad", ra, dec, 0.00025, source_id="* alf Lyr", epoch=2000.0,
                          proper_motion_ra_masyr=pm[0], proper_motion_dec_masyr=pm[1],
                          metadata={"wavelength": "multi", "physical": {"object_type": "dS*"},
                                    "catalog_has_proper_motions": True})
            else:
                src = row("gaia_dr3", gra, gdec, 0.0002, source_id="2096...", epoch=2016.0,
                          proper_motion_ra_masyr=pm[0] + 0.5, proper_motion_dec_masyr=pm[1] - 0.4,
                          data={"parallax": 130.2, "parallax_error": 0.2},
                          metadata={"wavelength": "optical", "catalog_has_proper_motions": True})
            return QueryResult([src], {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {name: registry.get(name) for name in ("simbad", "gaia_dr3")}
    service = CrossmatchService(registry, {"tap": Provider()})
    query = AdvancedQuery.from_dict({"ra": ra, "dec": dec, "radius_arcsec": 10.0, "min_confidence": 0.5})
    record = await service.crossmatch(ra, dec, query=query)
    simbad = record.catalog_results["simbad"]
    assert simbad["row_count"] == 1 and simbad["returned_count"] == 1
    confidence = record.provenance["matches"][0]
    assert confidence["source_id"] == "* alf Lyr" and confidence["confidence"] > 0.9
    # The Gaia row (J2016, 5.6" away as given) is the same star: it is identified too.
    gaia = next(m for m in record.provenance["matches"] if m["catalog"] == "gaia_dr3")
    assert gaia["confidence"] > 0.9


def test_undated_detection_has_one_placement_per_epoch_hypothesis() -> None:
    pm = (-801.551, 10362.394)
    simbad = row("simbad", 269.45207696, 4.69336497, 0.0007, epoch=2000.0, proper_motion_ra_masyr=pm[0],
                 proper_motion_dec_masyr=pm[1])
    det, info = source_detection(simbad, None)
    assert info["propagation"] == "source_pm_undated" and info["epoch_hypotheses"] == [2000.0, 2016.0]
    (ra0, dec0, _), (ra1, dec1, _) = det.epoch_placements
    assert math.hypot((ra0 - 269.45207696) * math.cos(math.radians(4.69)), dec0 - 4.69336497) * 3600 < 1e-6
    ra16, dec16 = propagate_radec(269.45207696, 4.69336497, *pm, 2000.0, 2016.0)
    assert math.hypot((ra1 - ra16) * math.cos(math.radians(4.7)), dec1 - dec16) * 3600 < 1e-6
    # As a field object the row sits at J2016 (the common epoch of the field partition).
    assert info["field_propagation"] == "source_pm"
    assert math.hypot((det.ra - ra16) * math.cos(math.radians(4.7)), det.dec - dec16) * 3600 < 1e-6
    # A slow star keeps one placement.
    slow = row("simbad", 10.0, 10.0, 0.001, epoch=2000.0, proper_motion_ra_masyr=1.0, proper_motion_dec_masyr=1.0)
    det, info = source_detection(slow, None)
    assert det.epoch_placements is None and info["propagation"] == "source_pm"


# ---------------------------------------------------------------------------
# Issue: completeness priors by target class (stars and NVSS)
# ---------------------------------------------------------------------------

NVSS_STARS = {
    # Gaia DR3 id: (NVSS source, separation")
    "629415710493682432": ("NVSS J100331+211006", 13.0),
    "629415706201972480": ("NVSS J100331+211006", 14.8),
    "629559574718166784": ("NVSS J100535+215127", 5.5),
    "628572522514194048": ("NVSS J100613+203527", 17.9),
}


@pytest.mark.parametrize("gaia_id", sorted(NVSS_STARS))
async def test_field_stars_do_not_get_unrelated_nvss_counterparts(gaia_id: str) -> None:
    key = f"nvss_star_{gaia_id}"
    record = await replay_crossmatch(key)
    nvss_name, sep = NVSS_STARS[gaia_id]
    members = members_by_id(record)
    assert members[gaia_id]["target_probability"] > 0.99
    nvss = members[nvss_name]
    assert nvss["separation_arcsec"] == pytest.approx(sep, abs=0.3)
    assert nvss["target_probability"] < 0.1, nvss["target_probability"]
    cls = record.provenance["association"]["target_class"]
    assert cls["class"] == "star" and cls["parallax_mas"] > 1.0
    assert record.provenance["association"]["completeness"]["nvss"] < 0.01


async def test_nvss_upper_limit_size_is_not_structure_and_the_pair_is_rejected() -> None:
    # NVSS J100535+215127: major axis '< 23.3"' (no position angle), 5.2" from a G = 15.7
    # star with dec_error 0.9". The upper limit is not a size (P was 0.9975 with it).
    record = await replay_crossmatch("nvss_star_629559574718166784")
    nvss = members_by_id(record)["NVSS J100535+215127"]
    assert nvss["data"]["position_angle"] is None and nvss["data"]["major_axis"] == pytest.approx(23.3)
    assert "structure" not in nvss["position_at_epoch"]["covariance_shape"]
    assert nvss["target_probability"] < 0.01


def test_completeness_prior_by_class_and_parallax() -> None:
    assert completeness_prior("nvss", "unknown") == completeness_prior("nvss", "extragalactic") == 0.5
    assert completeness_prior("gaia_dr3", "star", 1.0) == 0.5  # optical: unchanged
    near, far = completeness_prior("rosat", "star", 300.0), completeness_prior("rosat", "star", 1.0)
    assert far < near <= 0.5
    assert completeness_prior("nvss", "star", 1.0) < 0.01
    assert completeness_prior("nvss", "star", None) == completeness_prior("nvss", "star",
                                                                           astrometry.UNKNOWN_STELLAR_PARALLAX_MAS)
    with pytest.raises(ValueError):
        completeness_prior("nvss", "planet")


def test_completeness_changes_the_posterior_by_exactly_the_prior_odds() -> None:
    # One row: W = B c / ((1 - c) N) x (lone-row factor), so the posterior odds scale by
    # the prior odds exactly.
    det = [Detection("nvss", *offset_radec(150.0, 20.0, 2.0, 1.0), (1.0, 0.0, 1.0))]
    dens = {"nvss": 52.0}
    odds = []
    for c in (0.5, 0.01):
        cfg = AssociationConfig(completeness={"nvss": c}, outlier_fraction=0.0)
        p = float(associate(det, dens, target=(150.0, 20.0), config=cfg).target_probability[0])
        odds.append(p / (1 - p))
    assert odds[0] / odds[1] == pytest.approx((0.5 / 0.5) / (0.01 / 0.99), rel=1e-9)


async def test_completeness_and_target_class_can_be_set_per_request() -> None:
    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            ra, dec = offset_radec(target.ra, target.dec, 3.0, 0.0)
            src = row("nvss", ra, dec, 1.0, source_id="n", data={"ra_error": 0.06, "dec_error": 1.0},
                      metadata={"wavelength": "radio"})
            return QueryResult([src], {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {"nvss": registry.get("nvss")}
    service = CrossmatchService(registry, {"tap": Provider()})
    ra, dec = 150.123456, 20.654321
    default = await service.crossmatch(ra, dec, radius_arcsec=10.0)
    low = await service.crossmatch(ra, dec, radius_arcsec=10.0, completeness=0.001)
    star = await service.crossmatch(ra, dec, radius_arcsec=10.0, target_class="star")
    p = [r.provenance["matches"][0]["confidence"] for r in (default, low, star)]
    assert p[1] < p[0] and p[2] < p[0]
    assert low.provenance["association"]["completeness"] == 0.001
    assert star.provenance["association"]["target_class"]["class"] == "star"
    with pytest.raises(ValueError, match="completeness"):
        await service.crossmatch(ra, dec, completeness=1.5)
    with pytest.raises(ValueError, match="target_class"):
        await service.crossmatch(ra, dec, target_class="planet")


# ---------------------------------------------------------------------------
# Issue: densities at the default 3" radius in crowded fields
# ---------------------------------------------------------------------------


async def test_baade_window_density_at_the_default_radius_is_local() -> None:
    record = await replay_crossmatch("baade_3arcsec", target_uncertainty_arcsec=0.1)
    gaia = record.provenance["association"]["densities"]["gaia_dr3"]
    # 60" Gaia DR3 COUNT: 1,114,976 per deg^2 (review); 0.1-deg VizieR COUNT: 1.09e6.
    assert 0.5 * 1.115e6 < gaia["density_per_deg2"] < 2.0 * 1.115e6
    assert gaia["prior_source"] == "density_map"
    # The all-sky mean would have been used before (80,154 per deg^2 with the 3" cone).
    assert gaia["density_per_deg2"] > 10 * CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2


def test_density_map_is_the_healpix_order5_gaia_map() -> None:
    import astropy.units as u
    import cdshealpix.nested as cdshealpix  # a test dependency (in the venv)

    rng = np.random.default_rng(4)
    ra = rng.uniform(0, 360, 500)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 500)))
    ref = cdshealpix.lonlat_to_healpix(ra * u.deg, dec * u.deg, 5)
    assert [healpix_nested_index(5, a, d) for a, d in zip(ra, dec)] == ref.tolist()
    # High latitude is ~10x below the all-sky mean, the bulge ~25x above it.
    assert local_prior_density("gaia_dr3", 187.28, 2.05)[0] < 0.2 * CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2
    assert local_prior_density("gaia_dr3", 272.0, -27.0)[0] > 20 * CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2
    assert local_prior_density("nvss", 272.0, -27.0) == (CATALOG_SKY_DENSITY["nvss"].per_deg2, "sky_mean")


def test_catalogue_without_a_published_density_gets_a_radius_independent_prior() -> None:
    # One row 0.3" from the target in a VizieR table registered at run time.
    values = {r: estimate_density_deg2(1, cone_area_deg2(r), catalog="J/some/table", n_target_rows=1)[0]
              for r in (3.0, 10.0, 60.0)}
    assert values[3.0] == pytest.approx(astrometry.GENERIC_SKY_DENSITY_DEG2, rel=0.01)
    assert max(values.values()) / min(values.values()) < 2.0  # was 4.6e5 / 1.2e3 = 380
    info = estimate_density_deg2(1, cone_area_deg2(3.0), catalog="J/some/table")[1]
    assert info["method"] == "gamma_poisson_generic" and info["prior_source"] == "generic"


def test_catalogue_definition_can_declare_its_sky_density() -> None:
    from crossmatch import catalog_sky_density
    from models import catalog_from_dict

    definition = catalog_from_dict("vz", {
        "provider": "tap", "wavelength": "optical", "endpoint": "https://example/tap", "table": "t",
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id",
                       "sky_density": {"sources": 1000, "area_deg2": 100.0, "reference": "VizieR row count"}}})
    density = catalog_sky_density(definition)
    assert density.per_deg2 == pytest.approx(10.0) and density.reference == "VizieR row count"
    target = validate_target(150.0, 20.0)
    rows = QueryResult([row("vz", 150.0, 20.0, 0.1)], {"query_radius_arcsec": 3.0})
    registry = CatalogRegistry()
    registry._catalogs = {"vz": definition}
    densities, info = catalog_densities([("vz", rows)], target, 3.0, 0.1, registry=registry)
    assert info["vz"]["sky_mean_per_deg2"] == pytest.approx(10.0) and densities["vz"] < 11.0


# ---------------------------------------------------------------------------
# Issue: heavy-tailed positions (AllWISE of quasars)
# ---------------------------------------------------------------------------

QUASAR_ALLWISE = {
    "3c279": "J125611.15-054721.7",
    "cygnus_a": "J195928.37+404402.0",
    "crf3_5763535293039121152": "J085506.82-030336.7",
    "crf3_3975233510826944640": "J115518.31+193941.8",
    "crf3_903276775341338112": "J082210.08+335402.2",
}


@pytest.mark.parametrize("key", sorted(QUASAR_ALLWISE))
async def test_quasar_allwise_detection_off_by_several_sigma_is_identified(key: str) -> None:
    record = await replay_crossmatch(key)
    allwise = members_by_id(record)[QUASAR_ALLWISE[key]]
    # 0.2-0.5" off with 0.03-0.05" formal errors: 5-15 sigma. Before: P = 1e-26 .. 7e-3.
    assert allwise["separation_arcsec"] > 0.2
    assert allwise["target_probability"] > 0.9, allwise["target_probability"]
    assert "x1.1" in allwise["position_at_epoch"]["covariance_shape"]  # AllWISE calibration


def test_heavy_tail_gives_outliers_a_small_but_not_negligible_probability() -> None:
    target = (150.0, 20.0)
    dens = {"allwise": 2.0e4}
    for offset, low, high in ((0.1, 0.99, 1.0), (0.4, 0.3, 0.999), (3.0, 0.0, 1e-3)):
        det = [Detection("allwise", *offset_radec(*target, offset, 0.0), (0.04**2, 0.0, 0.04**2))]
        p = float(associate(det, dens, target=target, config=AssociationConfig(target_sigma_arcsec=0.01)).target_probability[0])
        assert low < p <= high, (offset, p)
        gaussian = AssociationConfig(target_sigma_arcsec=0.01, outlier_fraction=0.0)
        p_gauss = float(associate(det, dens, target=target, config=gaussian).target_probability[0])
        if offset == 0.4:
            assert p_gauss < 1e-10  # 10 sigma: the Gaussian model rejects it outright


# ---------------------------------------------------------------------------
# Issue: correlated fields and calibration
# ---------------------------------------------------------------------------


def _brute_force_posterior(dets, dens_deg2, target, ts, completeness, factor) -> np.ndarray:
    """Exact marginal posteriors under the generative model of the engine (Gaussian errors):
    field objects Poisson with density factor x max(nu), detected by catalogue k with
    probability nu_k / nu_obj; every target association and field partition enumerated."""
    from models import tangent_offset_arcsec

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for part in partitions(rest):
            yield [[first], *part]
            for i in range(len(part)):
                yield [*part[:i], [first, *part[i]], *part[i + 1:]]

    sr = (180 / math.pi) ** 2
    nu = {c: d * sr for c, d in dens_deg2.items()}
    nu_obj = factor * max(nu.values())
    q = {c: min(astrometry.MAX_DETECTION_FRACTION, v / nu_obj) for c, v in nu.items()}
    pos = [tangent_offset_arcsec(target[0], target[1], d.ra, d.dec) for d in dets]

    def ln_g(members, with_target):
        p = ([(0.0, 0.0)] if with_target else []) + [pos[i] for i in members]
        c = ([(ts * ts, 0.0, ts * ts)] if with_target else []) + [dets[i].cov for i in members]
        return astrometry.ln_bayes_factor(p, c) - (len(p) - 1) * math.log(4 * math.pi)

    idx = list(range(len(dets)))
    total, marg = -math.inf, np.full(len(dets), -math.inf)
    for r in range(len(dets) + 1):
        for a in itertools.combinations(idx, r):
            cats = [dets[i].catalog for i in a]
            if len(set(cats)) != len(cats):
                continue
            lw_a = sum(math.log(completeness) - math.log1p(-completeness) for _ in cats) + ln_g(list(a), True)
            terms = []
            for part in partitions([i for i in idx if i not in a]):
                lw = 0.0
                for o in part:
                    oc = [dets[i].catalog for i in o]
                    if len(set(oc)) != len(oc):
                        break
                    lw += (math.log(nu_obj) + sum(math.log(q[c]) for c in oc)
                           + sum(math.log1p(-q[c]) for c in q if c not in oc) + ln_g(o, False))
                else:
                    terms.append(lw)
            lw = lw_a + astrometry._logsumexp(terms)
            total = np.logaddexp(total, lw)
            for i in a:
                marg[i] = np.logaddexp(marg[i], lw)
    return np.exp(marg - total)


def test_correlated_field_posterior_is_exact_for_the_generative_model() -> None:
    rng = np.random.default_rng(5)
    cats = [("a", 0.5, 30.0), ("b", 1.0, 20.0), ("c", 1.5, 15.0)]
    dens = {c: d * 3600 for c, _, d in cats}
    for _ in range(25):
        dets = []
        for _ in range(int(rng.integers(2, 6))):
            c, s, _d = cats[rng.integers(0, 3)]
            dets.append(Detection(c, *offset_radec(150.0, 20.0, *rng.normal(0, 2.0, 2)), (s * s, 0.0, s * s)))
        cfg = AssociationConfig(target_sigma_arcsec=1.0, completeness=0.5, outlier_fraction=0.0, prune_weight=1e-300)
        engine = associate(dets, dens, target=(150.0, 20.0), config=cfg).target_probability
        exact = _brute_force_posterior(dets, dens, (150.0, 20.0), 1.0, 0.5, astrometry.OBJECT_DENSITY_FACTOR)
        assert np.max(np.abs(engine - exact)) < 1e-5


def test_neighbour_seen_by_several_catalogues_is_one_field_object_in_the_null() -> None:
    # A neighbour star 1.2" away seen by three catalogues: under the independent-field null
    # each coincidence is rewarded separately; as one field object only one is.
    target = (150.0, 20.0)
    pos = offset_radec(*target, 1.2, 0.0)
    dets = [Detection(c, *pos, (s * s, 0.0, s * s)) for c, s in (("a", 0.3), ("b", 0.4), ("c", 0.5))]
    dens = {"a": 3.6e4, "b": 3.6e4, "c": 3.6e4}
    independent = associate(dets, dens, target=target,
                            config=AssociationConfig(target_sigma_arcsec=1.0, correlated_fields=False))
    correlated = associate(dets, dens, target=target, config=AssociationConfig(target_sigma_arcsec=1.0))
    assert correlated.p_any < independent.p_any


def _calibration_bins(p: np.ndarray, t: np.ndarray, min_count: int = 100):
    edges = np.quantile(p[p > 0.02], np.linspace(0, 1, 7)) if (p > 0.02).sum() > 6 * min_count else [0.02, 1.0]
    rows = []
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & (p <= hi)
        if m.sum() >= min_count:
            rows.append((float(p[m].mean()), float(t[m].mean()), int(m.sum())))
    return rows


def _simulate(config: AssociationConfig, fields: int, seed: int):
    cats = [astrometry.SimCatalog("a", 0.5, 7.5, 0.5), astrometry.SimCatalog("b", 1.0, 5.0, 0.5),
            astrometry.SimCatalog("c", 1.5, 3.75, 0.5), astrometry.SimCatalog("d", 2.0, 2.5, 0.5)]
    dens = {c.name: c.density_arcmin2 * 3600 for c in cats}
    rng = np.random.default_rng(seed)
    probs, truth = [], []
    for _ in range(fields):
        dets, true, target = astrometry.simulate_field(rng, cats, radius_arcsec=20.0, target_sigma_arcsec=1.0)
        result = associate(dets, dens, target=target, config=config, cone=(150.0, 20.0, 20.0))
        probs.extend(result.target_probability.tolist())
        truth.extend(t == 0 for t in true)
    return np.asarray(probs), np.asarray(truth)


def test_posteriors_are_calibrated_on_a_correlated_sky_without_slack() -> None:
    # Field objects seen by several catalogues (simulate_field), target sigma 1" (comparable
    # to the catalogue errors 0.5-2"), heavy-tailed errors: every posterior bin (sextiles of
    # P > 0.02, >= 100 rows each) matches the true fraction within 3 binomial sigma.
    completeness = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    p, t = _simulate(AssociationConfig(target_sigma_arcsec=1.0, completeness=completeness), 1500, 11)
    bins = _calibration_bins(p, t)
    assert len(bins) >= 5
    for mean, true, count in bins:
        sigma = math.sqrt(mean * (1 - mean) / count)
        assert abs(true - mean) <= 3.0 * sigma, (mean, true, count)
    assert p.sum() == pytest.approx(t.sum(), rel=0.03)
    # Negative control: the same test catches wrong priors (completeness 0.95 assumed).
    wrong = AssociationConfig(target_sigma_arcsec=1.0, completeness={k: 0.95 for k in completeness})
    p, t = _simulate(wrong, 1500, 11)
    failures = [(m, tr) for m, tr, c in _calibration_bins(p, t) if abs(tr - m) > 3.0 * math.sqrt(m * (1 - m) / c)]
    assert failures, "a miscalibrated prior must fail the calibration check"


# ---------------------------------------------------------------------------
# Issue: pm-less rows of pm catalogues (Kruger 60 A) and binary components
# ---------------------------------------------------------------------------


class _FixedResolver:
    def __init__(self, obj) -> None:
        self.obj = obj

    async def resolve(self, _query):
        return self.obj


async def _stream_record(key: str, stem: str, name: str, **kwargs) -> tuple[dict, dict]:
    _register()
    spec = MATCHING_TARGETS[key]
    obj = SesameResolver.parse_response(name, (SESAME_DIR / f"{stem}.xml").read_text(encoding="utf-8"))
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"matching/{key}")))
        async with offline_client() as client:
            service = make_service(client)
            events = [e async for e in service.crossmatch_stream(
                name=name, resolver=_FixedResolver(obj), radius_arcsec=spec["radius_arcsec"], catalogs=spec["catalogs"],
                **kwargs)]
    record = events[-1]["data"]["record"]
    assert_replayed_exactly(record)
    return events[0]["data"], record


async def test_kruger60a_two_parameter_gaia_row_is_the_target() -> None:
    start, record = await _stream_record("kruger60a", "gj860a", "GJ 860 A")
    members = {m["source_id"]: m for g in record["crossmatch_groups"] for m in g["members"]}
    a = members["2007876324466455040"]  # Gaia DR3 2-parameter solution of Kruger 60 A
    assert a["data"]["astrometric_params_solved"] == 3 and a["separation_arcsec"] > 12.0
    # Placed with the target's motion it lies 0.03" from the target (it is SIMBAD's source).
    assert a["position_at_epoch"]["propagation"] == "target_pm"
    assert a["position_at_epoch"]["field_propagation"] == "stationary"
    assert a["target_probability"] > 0.9  # was 0.0
    group = record["crossmatch_groups"][0]
    assert group["contains_target"] and "2007876324466455040" in {m["source_id"] for m in group["members"]}
    assert members["2007876324472098432"]["target_probability"] < 0.1  # B's 2-parameter row
    # A binary component: orbital motion allowance on the resolver motion.
    assert start["target_pm_error_masyr"] == 30.0
    assert any("binary component" in n for n in record["provenance"]["association"]["notes"])


def test_stationary_row_gets_both_hypotheses() -> None:
    target = validate_target(10.0, 20.0, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=500.0)
    ra, dec = propagate_radec(10.0, 20.0, 0.0, 500.0, 2000.0, 2016.0)
    two_param = row("gaia_dr3", ra, dec, 0.0005, epoch=2016.0, metadata={"catalog_has_proper_motions": True})
    det, info = source_detection(two_param, target)
    assert info["propagation"] == "target_pm" and info["field_propagation"] == "stationary"
    assert (det.ra, det.dec) == (ra, dec)  # field placement: where it was measured
    assert math.hypot(det.target_ra - 10.0, det.target_dec - 20.0) * 3600 < 1e-6


# ---------------------------------------------------------------------------
# Issue: coarse / extended SIMBAD identities
# ---------------------------------------------------------------------------


async def test_high_velocity_cloud_is_not_a_star_counterpart() -> None:
    record = await replay_crossmatch("hvc_star")
    hvc = members_by_id(record)["HVC 104.2-48-168"]
    assert hvc["data"]["coo_qual"] == "E" and hvc["extended_identity"]
    assert hvc["target_probability"] < 0.05  # was 0.958
    group = target_group(record)
    assert "HVC 104.2-48-168" not in {m["source_id"] for m in group["members"]}


def test_extended_types_and_quality_e_are_extended_identities() -> None:
    from crossmatch import is_extended_identity

    assert is_extended_identity(row("simbad", 1, 1, 60.0, data={"coo_qual": "E", "otype": "*"}))
    assert is_extended_identity(row("simbad", 1, 1, 0.5, data={"coo_qual": "C"},
                                    metadata={"physical": {"object_type": "GlC"}}))
    assert not is_extended_identity(row("simbad", 1, 1, 0.5, data={"coo_qual": "C"},
                                        metadata={"physical": {"object_type": "BY*"}}))
    assert not is_extended_identity(row("ned", 1, 1, 60.0, data={"coo_qual": "E"}))


# ---------------------------------------------------------------------------
# Issue: name + epoch in the stream
# ---------------------------------------------------------------------------


async def test_stream_name_with_epoch_moves_the_resolver_position() -> None:
    start, record = await _stream_record("barnard_name_2016", "barnards_star", "Barnard's star", epoch=2016.0)
    target = start["target"]
    ra16, dec16 = propagate_radec(269.45207696, 4.69336497, -801.551, 10362.394, 2000.0, 2016.0)
    assert target["epoch"] == 2016.0 and target["pm_dec_masyr"] == pytest.approx(10362.394)
    assert target["ra"] == pytest.approx(ra16, abs=1e-9) and target["dec"] == pytest.approx(dec16, abs=1e-9)
    assert target["parallax_mas"] == pytest.approx(546.976, abs=0.01)
    gaia = next(m for g in record["crossmatch_groups"] for m in g["members"] if m["source_id"] == "4472832130942575872")
    assert gaia["separation_arcsec"] < 0.05 and gaia["target_probability"] > 0.99  # was: no Gaia match


def test_resolved_search_target_rules() -> None:
    xml = (Path(__file__).resolve().parent / "fixtures" / "sesame" / "barnards_star.xml").read_text(encoding="utf-8")
    obj = SesameResolver.parse_response("Barnard's star", xml)
    same = resolved_search_target(obj)
    assert same["epoch"] == 2000.0 and same["pm_source"] == "resolver"
    assert same["target_uncertainty_arcsec"] == pytest.approx(_resolver_position_error(obj))
    moved = resolved_search_target(obj, epoch=2016.0)
    grow = _resolver_pm_error(obj) * 16.0 / 1000.0
    assert moved["target_uncertainty_arcsec"] == pytest.approx(math.hypot(_resolver_position_error(obj), grow))
    user = resolved_search_target(obj, epoch=2010.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
    assert (user["ra"], user["dec"]) == (pytest.approx(obj.ra_deg), pytest.approx(obj.dec_deg))  # user motion 0
    assert user["pm_source"] == "input"
    # No motion known (a Galactic object without one): a new epoch cannot be applied.
    static = SesameResolver.parse_response("x", xml.replace("<pmRA>", "<nopmRA>").replace("</pmRA>", "</nopmRA>"))
    with pytest.raises(ValueError, match="cannot be moved"):
        resolved_search_target(static, epoch=2016.0)
    # Extragalactic objects do not move.
    xml3 = (Path(__file__).resolve().parent / "fixtures" / "sesame" / "3c273.xml").read_text(encoding="utf-8")
    qso = resolved_search_target(SesameResolver.parse_response("3C 273", xml3), epoch=2016.0)
    assert qso["epoch"] == 2016.0 and qso["ra"] == pytest.approx(187.27791542, abs=1e-6)


async def test_stream_rejects_name_and_coordinates_together() -> None:
    service = CrossmatchService(CatalogRegistry(), {})
    with pytest.raises(ValueError, match="not both"):
        _ = [e async for e in service.crossmatch_stream(150.0, 20.0, name="3C 273")]
    with pytest.raises(InvalidCoordinateError):
        _ = [e async for e in service.crossmatch_stream(name="3C 273", pm_ra_masyr=5.0)]


# ---------------------------------------------------------------------------
# Issue: RA = 0 without a target; covariance rotation near the pole
# ---------------------------------------------------------------------------


def test_target_less_association_across_ra_zero_is_one_object() -> None:
    for ra0 in (0.0, 150.0):
        a = Detection("a", ra0 % 360.0, 10.0, (0.01, 0.0, 0.01))
        b = Detection("b", *offset_radec(ra0 % 360.0, 10.0, -0.05, 0.0), (0.01, 0.0, 0.01))
        result = associate([a, b], {"a": 1e4, "b": 1e4})
        assert [sorted(g.members) for g in result.groups] == [[0, 1]], ra0
        assert result.groups[0].match_probability > 0.999


def _position_angle(ra1, dec1, ra2, dec2) -> float:
    """Position angle (deg, east of north) of point 2 seen from point 1."""
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    return math.degrees(math.atan2(math.sin(a2 - a1) * math.cos(d2),
                                   math.cos(d1) * math.sin(d2) - math.sin(d1) * math.cos(d2) * math.cos(a2 - a1)))


@pytest.mark.parametrize("dec", [20.0, 89.0, 89.99, 89.9997])
def test_error_ellipse_keeps_its_orientation_near_the_pole(dec: float) -> None:
    # A detection 1" east of the target with a 5" x 0.05" ellipse along the direction to
    # the target (in the detection's own frame): the same configuration at every declination.
    target = (30.0, dec)
    det_pos = offset_radec(*target, 1.0, 0.0)
    pa = _position_angle(det_pos[0], det_pos[1], *target)
    det = Detection("a", *det_pos, astrometry.ellipse_covariance(5.0, 0.05, pa))
    p = associate([det], {"a": 100.0}, target=target, config=AssociationConfig(target_sigma_arcsec=0.01,
                                                                               outlier_fraction=0.0))
    assert p.target_probability[0] > 0.999, dec  # was 5.2e-17 at dec 89.9997


def test_gnomonic_jacobian_matches_finite_differences() -> None:
    from models import tangent_offset_arcsec

    for ra0, dec0, ra, dec in [(10.0, 20.0, 10.01, 20.02), (30.0, 89.999, 150.0, 89.9995), (0.0, -45.0, 359.99, -45.01)]:
        j11, j12, j21, j22 = (float(v[0]) for v in gnomonic_jacobian(ra0, dec0, np.array([ra]), np.array([dec])))
        h = 1e-4  # arcsec
        x0, y0 = tangent_offset_arcsec(ra0, dec0, ra, dec)
        xe, ye = tangent_offset_arcsec(ra0, dec0, *offset_radec(ra, dec, h, 0.0))
        xn, yn = tangent_offset_arcsec(ra0, dec0, *offset_radec(ra, dec, 0.0, h))
        assert (xe - x0) / h == pytest.approx(j11, abs=1e-4) and (ye - y0) / h == pytest.approx(j21, abs=1e-4)
        assert (xn - x0) / h == pytest.approx(j12, abs=1e-4) and (yn - y0) / h == pytest.approx(j22, abs=1e-4)


# ---------------------------------------------------------------------------
# Issue: duplicate listings (Pan-STARRS)
# ---------------------------------------------------------------------------


async def test_panstarrs_duplicates_do_not_split_the_probability() -> None:
    record = await replay_crossmatch("ps1_duplicate")
    members = members_by_id(record)
    first, second = members["133723599999802520"], members["133720000000192516"]
    # 8.8 mas apart, nDetections 44 / 38: one measurement listed twice (was 0.543 / 0.457).
    assert {first["coincident_with"], second["coincident_with"]} == {None, "133723599999802520"}
    assert first["target_probability"] == second["target_probability"] > 0.99
    group = target_group(record)
    assert {"133723599999802520", "133720000000192516"} <= {m["source_id"] for m in group["members"]}


def test_duplicates_need_consistent_positions_below_the_resolution() -> None:
    target = (150.0, 20.0)
    base = Detection("panstarrs_dr2", *target, (0.02**2, 0.0, 0.02**2), rank=-44)
    twin = Detection("panstarrs_dr2", *offset_radec(*target, 0.0088, 0.0), (0.02**2, 0.0, 0.02**2), rank=-38)
    resolved = Detection("panstarrs_dr2", *offset_radec(*target, 0.2, 0.0), (0.02**2, 0.0, 0.02**2))
    result = associate([base, twin, resolved], {"panstarrs_dr2": 7e4}, target=target)
    group = next(g for g in result.groups if g.contains_target)
    assert group.coincident_with == {1: 0}  # the row measured more often represents the pair
    assert 2 not in group.coincident_with  # 0.2" apart with 0.02" errors: a different source


# ---------------------------------------------------------------------------
# Issue: SIMBAD proper-motion errors
# ---------------------------------------------------------------------------


def test_simbad_pm_error_is_not_inflated_by_its_own_epoch_growth() -> None:
    from models import DEFAULT_CATALOGS, compute_positional_error

    spec = DEFAULT_CATALOGS["simbad"]["pos_error"]
    data = {"coo_err_maj": 0.03, "coo_err_min": 0.02, "coo_err_angle": 90, "coo_bibcode": "2020yCat.1350....0G",
            "pmra": -801.5, "pmdec": 10362.4}
    sigma, details = compute_positional_error(data, spec)
    simbad = row("simbad", 1.0, 1.0, sigma, data=data, metadata={"positional_error": details},
                 proper_motion_ra_masyr=-801.5, proper_motion_dec_masyr=10362.4, epoch=2000.0)
    # pm error = pm_error_ratio (1.4) x the J2016 error (0.03 mas major axis) per year.
    assert pm_sigma_masyr(simbad) == pytest.approx(1.4 * 0.03e-3 * 1000.0, rel=1e-6)
    assert sigma > 0.03e-3  # the J2000 position error includes the 16-yr growth ...
    assert pm_sigma_masyr(simbad) < 1.4 * sigma * 1000.0  # ... which the pm error must not


# ---------------------------------------------------------------------------
# Issue: rounded coordinates and the target sigma
# ---------------------------------------------------------------------------


async def test_rounded_3c273_coordinates_still_identify_the_quasar() -> None:
    # The coordinates as TYPED (text): their last digit sets the target uncertainty. (Numbers
    # are exact -- see test_coordinate_sigma_from_the_digits_given.)
    record = await replay_crossmatch("3c273_rounded", ra="187.278", dec="2.052")
    assert record.failures == []
    assoc = record.provenance["association"]
    assert assoc["target_sigma_source"] == "coordinate_precision"
    assert assoc["target_sigma_arcsec"] == pytest.approx(math.hypot(0.1, coordinate_sigma_arcsec("187.278", "2.052")))
    members = members_by_id(record)
    for sid in ("3C 273", "3700386905605055360", "110461872779253351", "1237651735760142397"):
        assert members[sid]["target_probability"] > 0.95, sid  # all were 0.0
    assert members["[CME2001] 3C 273 1"]["target_probability"] < 0.01  # the jet knot at 4.3"
    group = target_group(record)
    assert "3C 273" in {m["source_id"] for m in group["members"]}


def test_coordinate_sigma_from_the_digits_given() -> None:
    assert coordinate_sigma_arcsec("187.278", "2.052") == pytest.approx(
        math.sqrt(((3.6 * math.cos(math.radians(2.052))) ** 2 + 3.6**2) / 2 / 12.0))
    assert coordinate_sigma_arcsec("187.2779154", "2.0523883") < 1e-3
    assert coordinate_sigma_arcsec(187.27791542000001, 2.05238823055) < 1e-5
    # Numbers are exact: a float's repr says nothing about the digits typed (150.5 may be an
    # exact position; before, it got a 104" sigma and lost the star at the position).
    assert coordinate_sigma_arcsec(187.278, 2.052) == 0.0
    assert coordinate_sigma_arcsec(150.5, 2.2) == 0.0
    assert coordinate_sigma_arcsec(150, 2) == 0.0


async def test_target_uncertainty_can_be_given_with_a_query() -> None:
    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            return QueryResult([row("gaia_dr3", *offset_radec(target.ra, target.dec, 1.4, 0.0), 0.001,
                                    metadata={"wavelength": "optical"})], {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3")}
    service = CrossmatchService(registry, {"tap": Provider()})
    query = AdvancedQuery.from_dict({"ra": 150.123456, "dec": 20.654321, "radius_arcsec": 5.0, "min_confidence": 0.0,
                                     "metadata": {"target_uncertainty_arcsec": 1.0}})
    record = await service.crossmatch(150.123456, 20.654321, query=query)
    assert record.provenance["association"]["target_sigma_arcsec"] == 1.0
    assert record.provenance["association"]["target_sigma_source"] == "input"
    assert record.provenance["matches"][0]["confidence"] > 0.5


# ---------------------------------------------------------------------------
# Issue: contains_target / 'best' only for a probable counterpart
# ---------------------------------------------------------------------------


def test_far_outlier_is_not_flagged_as_the_target() -> None:
    target = (150.0, 20.0)
    det = [Detection("twomass_psc", *offset_radec(*target, 0.95, 0.0), (0.07**2, 0.0, 0.07**2))]
    result = associate(det, {"twomass_psc": 1.1e4}, target=target,
                       config=AssociationConfig(outlier_fraction=0.0))
    assert result.p_any < 1e-6
    assert not any(g.contains_target for g in result.groups)
    assert all(g.match_flag != "best" for g in result.groups)
    assert any("no target group" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Targeted unit tests (mutation survivors)
# ---------------------------------------------------------------------------


def test_catalog_densities_truncated_cone_and_target_rows() -> None:
    target = validate_target(150.0, 20.0)
    rows = [row("gaia_dr3", 150.0, 20.0, 0.0002, source_id="t")]
    rows += [row("gaia_dr3", *offset_radec(150.0, 20.0, 1.0 + i * 0.3, 0.0), 0.0002, source_id=f"f{i}") for i in range(10)]
    full = QueryResult(rows, {"query_radius_arcsec": 10.0})
    cut = QueryResult(rows, {"query_radius_arcsec": 10.0, "archive_truncated": True})
    _, info_full = catalog_densities([("gaia_dr3", full)], target, 10.0, 0.1)
    _, info_cut = catalog_densities([("gaia_dr3", cut)], target, 10.0, 0.1)
    assert info_full["gaia_dr3"]["field_rows"] == 10  # the target's own row is not a field source
    assert info_full["gaia_dr3"]["radius_arcsec"] == 10.0
    assert info_cut["gaia_dr3"]["radius_arcsec"] == pytest.approx(1.0 + 9 * 0.3, abs=1e-6)
    assert info_cut["gaia_dr3"]["area_deg2"] == pytest.approx(cone_area_deg2(3.7), rel=1e-6)
    for info in (info_full["gaia_dr3"], info_cut["gaia_dr3"]):
        expected = (1.0 + 10) / (1.0 / info["prior_mean_per_deg2"] + info["area_deg2"])
        assert info["density_per_deg2"] == pytest.approx(expected, rel=1e-9)
    assert info_cut["gaia_dr3"]["density_per_deg2"] > info_full["gaia_dr3"]["density_per_deg2"]


def test_resolver_errors_from_the_recorded_sesame_answer() -> None:
    xml = (Path(__file__).resolve().parent / "fixtures" / "sesame" / "barnards_star.xml").read_text(encoding="utf-8")
    obj = SesameResolver.parse_response("Barnard's star", xml)
    raw = obj.resolver_metadata["raw_fields"]
    era, ede = float(raw["errramas"][0]), float(raw["errdemas"][0])
    assert _resolver_position_error(obj) == pytest.approx(math.sqrt((era**2 + ede**2) / 2) / 1000.0)
    epm = [float(raw["pm.epmra"][0]), float(raw["pm.epmde"][0])]
    assert _resolver_pm_error(obj) == pytest.approx(math.sqrt((epm[0] ** 2 + epm[1] ** 2) / 2))


def test_adopted_pm_sigma_comes_from_the_adopted_row() -> None:
    gaia = row("gaia_dr3", 1.0, 1.0, 0.0003, source_id="g", data={"pmra_error": 0.03, "pmdec_error": 0.04},
               proper_motion_ra_masyr=5.0, proper_motion_dec_masyr=5.0, epoch=2016.0)
    successes = [("gaia_dr3", QueryResult([gaia], {}))]
    assert _adopted_pm_sigma({"source": "adopted", "catalog": "gaia_dr3", "source_id": "g"}, successes) == \
        pytest.approx(math.sqrt((0.03**2 + 0.04**2) / 2))
    assert _adopted_pm_sigma({"source": "extragalactic"}, successes) == 0.0


def test_extragalactic_row_with_a_noise_motion_stays_in_place() -> None:
    target = validate_target(187.7059308, 12.3911233, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
    m87 = row("simbad", 187.7059308, 12.3911233, 0.0002, epoch=2016.0, proper_motion_ra_masyr=-8.0,
              proper_motion_dec_masyr=10.7)
    det, info = source_detection(m87, target, extragalactic=True)
    assert info["propagation"] == "extragalactic" and (det.ra, det.dec) == (187.7059308, 12.3911233)
    moved, info2 = source_detection(m87, target)
    assert info2["propagation"] == "source_pm" and (moved.ra, moved.dec) != (187.7059308, 12.3911233)


def test_prune_weight_includes_the_target_sigma() -> None:
    # A precise row 10" off a target known to 5": consistent (chi2 = 4) only through the
    # target's own sigma; its posterior must match the closed-form two-way weight.
    target = (150.0, 20.0)
    det = [Detection("a", *offset_radec(*target, 10.0, 0.0), (0.01**2, 0.0, 0.01**2))]
    cfg = AssociationConfig(target_sigma_arcsec=5.0, outlier_fraction=0.0, correlated_fields=False)
    p = float(associate(det, {"a": 1.0}, target=target, config=cfg).target_probability[0])
    b = astrometry.bayes_factor_2way(10.0, 5.0, 0.01)
    w = b / (1.0 * FULL_SKY_DEG2)
    assert p == pytest.approx(w / (1 + w), rel=1e-6)
    assert p > 0.5


def test_exact_probability_is_the_association_alone_and_match_probability_the_joint() -> None:
    # a and b are the target's counterparts; c (3" off, 0.8" errors) may or may not be.
    target = (150.0, 20.0)
    a = Detection("a", *target, (0.1**2, 0.0, 0.1**2))
    b = Detection("b", *offset_radec(*target, 0.2, 0.0), (0.2**2, 0.0, 0.2**2))
    c = Detection("c", *offset_radec(*target, 3.0, 0.0), (0.8**2, 0.0, 0.8**2))
    cfg = AssociationConfig(target_sigma_arcsec=0.2, outlier_fraction=0.0, correlated_fields=False)
    result = associate([a, b, c], {"a": 1e5, "b": 1e5, "c": 1e4}, target=target, config=cfg)
    group = next(g for g in result.groups if g.contains_target)
    assert sorted(group.members) == [0, 1]
    p_c = float(result.target_probability[2])
    assert 0.05 < p_c < 0.5
    # joint: a and b (c free), between P(a) + P(b) - 1 and min(P(a), P(b)); exact: a and b
    # and not c, i.e. the joint minus (almost all of) P(c).
    pa, pb = (float(v) for v in result.target_probability[:2])
    assert pa + pb - 1.0 - 1e-9 <= group.match_probability <= min(pa, pb) + 1e-9
    assert group.exact_probability == pytest.approx(group.match_probability - p_c, abs=5e-3)
    assert group.exact_probability < group.match_probability


def test_field_group_p_i_is_its_share_of_the_non_null_weight() -> None:
    ra, dec = 150.0, 20.0
    a = Detection("opt", *offset_radec(ra, dec, 0.5, 0.0), (0.01, 0.0, 0.01))
    b = Detection("opt", *offset_radec(ra, dec, -0.5, 0.0), (0.01, 0.0, 0.01))
    result = associate([a, b], {"opt": 100.0}, target=(ra, dec), config=AssociationConfig(target_sigma_arcsec=2.0))
    other = next(g for g in result.groups if not g.contains_target)
    assert other.p_i == pytest.approx(0.5, abs=1e-6)


def test_candidates_beyond_the_per_catalogue_limit_keep_a_probability() -> None:
    target = (150.0, 20.0)
    dets = [Detection("a", *offset_radec(*target, 0.1 * np.cos(k), 0.1 * np.sin(k)), (0.25, 0.0, 0.25))
            for k in range(5)]
    cfg = AssociationConfig(target_sigma_arcsec=0.5, max_candidates_per_catalog=2, correlated_fields=False,
                            outlier_fraction=0.0)
    cut = associate(dets, {"a": 1e3}, target=target, config=cfg)
    full = associate(dets, {"a": 1e3}, target=target, config=AssociationConfig(
        target_sigma_arcsec=0.5, correlated_fields=False, outlier_fraction=0.0))
    assert any("most probable were enumerated" in n for n in cut.notes)
    assert np.all(cut.target_probability > 0.1)
    np.testing.assert_allclose(cut.target_probability, full.target_probability, atol=0.02)
    assert cut.p_any == pytest.approx(full.p_any, abs=1e-3)


def test_beam_search_keeps_the_null_association() -> None:
    rng = np.random.default_rng(2)
    target = (150.0, 20.0)
    dets = []
    for cat in "abcd":
        for _ in range(3):
            dets.append(Detection(cat, *offset_radec(*target, *rng.normal(0, 3.0, 2)), (1.0, 0.0, 1.0)))
    dens = {c: 3e4 for c in "abcd"}
    exact = associate(dets, dens, target=target, config=AssociationConfig(outlier_fraction=0.0, max_states=10**6))
    beam = associate(dets, dens, target=target, config=AssociationConfig(outlier_fraction=0.0, max_states=16))
    assert exact.exact and not beam.exact
    assert 0.0 < beam.p_any < 0.999
    assert beam.p_any == pytest.approx(exact.p_any, abs=0.05)


async def test_crossmatch_many_forwards_every_per_target_option() -> None:
    seen: list[tuple[str, float]] = []

    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            seen.append((catalog.name, radius_arcsec))
            return QueryResult([row(catalog.name, target.ra, target.dec, 0.01, metadata={"wavelength": "optical"})],
                               {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {n: registry.get(n) for n in ("gaia_dr3", "twomass_psc")}
    service = CrossmatchService(registry, {"tap": Provider()})
    records = await service.crossmatch_many([
        {"ra": 10.123456, "dec": 5.123456, "catalogs": ["gaia_dr3"], "target_uncertainty_arcsec": 0.3,
         "epoch": 2016.0, "pm_ra_masyr": 1.0, "pm_dec_masyr": 2.0, "pm_source": "resolver",
         "target_pm_error_masyr": 0.7, "completeness": 0.2, "target_class": "star"},
        {"ra": 11.123456, "dec": 5.123456, "radius_arcsec": 4.0},
    ], radius_arcsec=2.0)
    first, second = records
    assert first.catalogs_queried == 1 and second.catalogs_queried == 2
    assoc = first.provenance["association"]
    assert assoc["target_sigma_arcsec"] == 0.3 and assoc["target_pm_error_masyr"] == 0.7
    assert assoc["completeness"] == 0.2 and assoc["target_class"]["class"] == "star"
    assert first.provenance["target_proper_motion"]["source"] == "resolver"
    assert second.provenance["query_radius_arcsec"] == 4.0
    with pytest.raises(ValueError, match="unknown option"):
        await service.crossmatch_many([{"ra": 1.0, "dec": 1.0, "radius": 3.0}])


async def test_crossmatch_many_validates_every_target_before_querying() -> None:
    started: list[float] = []

    class Slow:
        async def query(self, catalog, target, radius_arcsec):
            started.append(target.ra)
            await asyncio.sleep(0.3)
            return QueryResult([], {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3")}
    service = CrossmatchService(registry, {"tap": Slow()})
    with pytest.raises(InvalidCoordinateError, match="target 2"):
        await service.crossmatch_many([{"ra": 150.0, "dec": 20.0}, {"ra": 151.0, "dec": 20.0},
                                       {"ra": 152.0, "dec": 95.0}, {"ra": 153.0, "dec": 20.0}])
    assert started == []  # nothing was queried


async def test_crossmatch_many_cancels_the_others_when_one_fails() -> None:
    finished: list[float] = []
    cancelled: list[float] = []

    class Provider:
        async def query(self, catalog, target, radius_arcsec):
            try:
                await asyncio.sleep(0.05 if target.ra < 151 else 0.5)
            except asyncio.CancelledError:
                cancelled.append(target.ra)
                raise
            finished.append(target.ra)
            return QueryResult([], {"max_rows": 200, "row_limit": 200})

    registry = CatalogRegistry()
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3")}
    service = CrossmatchService(registry, {"tap": Provider()})

    def boom(ctx, successes, failures):
        raise RuntimeError("finalize failed")

    service.finalize = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="finalize failed"):
        await service.crossmatch_many([{"ra": 150.5, "dec": 20.0}, {"ra": 152.0, "dec": 20.0}])
    await asyncio.sleep(0.6)
    assert 152.0 in cancelled and 152.0 not in finished


def test_target_class_from_counterparts() -> None:
    from crossmatch import Match, _class_from_association

    target = (150.0, 20.0)

    def assoc_for(rows):
        dets = [Detection(r.catalog, r.ra, r.dec, (1e-4, 0.0, 1e-4)) for r in rows]
        result = associate(dets, {r.catalog: 1e3 for r in rows}, target=target)
        return [Match(r.catalog, r, 0.0, 0.0) for r in rows], result

    # An ordinary stellar type (a high proper-motion star): the calibrated stellar priors.
    star = row("simbad", *target, 0.01, metadata={"physical": {"object_type": "PM*", "parallax": 12.0}})
    qso = row("simbad", *target, 0.01, metadata={"physical": {"object_type": "QSO", "redshift": 1.2}})
    psr = row("simbad", *target, 0.01, metadata={"physical": {"object_type": "Psr"}})
    gaia = row("gaia_dr3", *target, 0.001, data={"parallax": 3.0, "parallax_error": 0.1})
    assert _class_from_association(*assoc_for([star]))["class"] == "star"
    assert _class_from_association(*assoc_for([star]))["parallax_mas"] == 12.0
    # A chromospherically active BY Dra star is a known X-ray emitter: not the random-star
    # calibration, so the default priors ('unknown').
    active = row("simbad", *target, 0.01, metadata={"physical": {"object_type": "BY*", "parallax": 12.0}})
    assert _class_from_association(*assoc_for([active]))["class"] == "unknown"
    assert _class_from_association(*assoc_for([qso]))["class"] == "extragalactic"
    assert _class_from_association(*assoc_for([psr]))["class"] == "unknown"
    assert _class_from_association(*assoc_for([gaia]))["parallax_mas"] == 3.0
    assert _class_from_association(*assoc_for([star, gaia]))["parallax_mas"] == 3.0  # Gaia's parallax first


def test_resolved_radio_structure_needs_a_resolved_major_axis() -> None:
    sig = astrometry.FWHM_TO_SIGMA
    resolved = row("nvss", 1.0, 1.0, 0.5, data={"major_axis": 21.5, "minor_axis": 15.9, "position_angle": 44.3})
    cov = structure_covariance(resolved)
    # Major axis only (the minor axis is an upper limit for 88% of resolved NVSS sources).
    assert cov[0] + cov[2] == pytest.approx((21.5 * sig) ** 2)
    unresolved = row("nvss", 1.0, 1.0, 0.5, data={"major_axis": 97.4, "minor_axis": 72.6, "position_angle": None})
    assert structure_covariance(unresolved) is None


async def test_sesame_outage_is_distinguished_from_an_unknown_name() -> None:
    import httpx

    from models import ObjectResolutionError, ResolverUnavailableError

    url = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"
    empty = "<?xml version='1.0'?><Sesame><Target><name>zzz</name><INFO>*** Nothing found ***</INFO></Target></Sesame>"
    cases = [(respx.MockResponse(503, text="down"), ResolverUnavailableError),
             (respx.MockResponse(429, text="slow down"), ResolverUnavailableError),
             (httpx.ConnectError("refused"), ResolverUnavailableError),
             (respx.MockResponse(404, text="no"), ObjectResolutionError),
             (respx.MockResponse(200, text=empty), ObjectResolutionError)]
    for outcome, expected in cases:
        with respx.mock(assert_all_called=False) as router:
            route = router.get(url__startswith=url)
            if isinstance(outcome, Exception):
                route.mock(side_effect=outcome)
            else:
                route.mock(return_value=outcome)
            async with httpx.AsyncClient() as client:
                with pytest.raises(ObjectResolutionError) as info:
                    await SesameResolver(client).resolve("zzz")
        assert isinstance(info.value, expected)
        assert isinstance(info.value, ResolverUnavailableError) == (expected is ResolverUnavailableError)
