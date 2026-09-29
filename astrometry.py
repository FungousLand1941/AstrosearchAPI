"""Bayesian N-way probabilistic cross-identification of catalogue detections.

This module is the statistical core of the crossmatch engine. It is pure Python/numpy
(no network, no FastAPI) so it can be used directly from notebooks::

    from astrometry import Detection, associate
    result = associate(detections, densities_deg2, target=(ra, dec))

Method
======

**Bayes factor** (Budavari & Szalay 2008, ApJ 679, 301, "B&S"). For ``n`` detections of
one object at unknown true position ``m`` with Gaussian positional errors the evidence
ratio of "same object" (H) to "unrelated objects, each anywhere on the sky" (K) is

    B = (4 pi)^(n-1) Int prod_i N(x_i | m, C_i) dm
      = 2^(n-1) prod_i |W_i|^(1/2) / |W|^(1/2) exp(-chi2 / 2),     W_i = C_i^-1, W = sum W_i,

with ``chi2 = sum_i (x_i - m_hat)^T W_i (x_i - m_hat)`` about the precision-weighted mean
``m_hat``, all angles in radians (flat-sky limit of the Fisher distribution, valid for
errors << 1 rad). For circular errors this is B&S eq. (18),
``B = 2^(n-1) prod w_i / sum w_i exp(-sum_{i<j} w_i w_j psi_ij^2 / (2 sum w_i))``, and for
two detections ``B = 2 / (s1^2 + s2^2) exp(-psi^2 / (2 (s1^2 + s2^2)))`` (B&S eq. 16).
Error ellipses (full 2x2 covariances) are used when the catalogue publishes them (the
elliptical generalisation of Pineau et al. 2017, A&A 597, A89, sect. 2).

**Target-centric association (NWAY; Salvato et al. 2018, MNRAS 473, 4937, App. B).**
The search target is the "primary catalogue": a detection at the requested position
with its own uncertainty (resolver or user supplied). Every *association* -- for each
catalogue either no counterpart or exactly one of its candidates -- is a hypothesis.
Its unnormalised posterior weight relative to "no counterpart anywhere" is

    W_a = B(target + a) prod_{k in a} [c_k / (1 - c_k)] / N_k,        W_null = 1,

where ``N_k = 4 pi nu_k`` is catalogue k's local source density expressed as a number of
objects on the whole sky (so ``1 / N_k`` is the B&S prior that one given source is the
counterpart) and ``c_k`` is the prior probability that the target has a counterpart in
catalogue k. This is the exact posterior of the generative model "the target's
counterpart in catalogue k exists with probability c_k and lies around the true
position; all other rows are an independent Poisson field of density nu_k". It is
NWAY's weight ``B * nu_primary prod c / prod nu_plus`` with NWAY's completeness equal to
the prior *odds* c/(1-c): NWAY's default ``prior_completeness=1`` corresponds to
``c_k = 0.5`` here, which is our default. Following NWAY (nwaylib
``_compute_final_probabilities``):

* ``p_any = 1 - W_null / sum_a W_a`` -- probability that the target has any counterpart;
* ``p_i = W_a / sum_{a != null} W_a`` -- probability of association a given one exists;
* ``match_flag`` 'best' for the most probable non-null association and 'secondary' for
  those with ``p_i > 0.5 p_best`` (NWAY ``--acceptable-prob`` default 0.5).

The target group is the most probable non-null association. Its ``match_probability``
is the joint posterior that *every* member is a counterpart of the target (summed over
what the other catalogues contribute); ``exact_probability`` is the posterior of exactly
this association (NWAY ``p_any * p_i``).

Each catalogue contributes at most one source to an association by construction, and
every *partial* association (catalogues without a counterpart) is a hypothesis of its
own. The per-detection probability is the marginal posterior
``P(j is the target's counterpart) = sum_{a containing j} W_a / sum_a W_a``.

All associations are enumerated when there are at most ``max_states`` of them (exact);
otherwise a beam search keeps the ``max_states`` most probable partial associations
after each catalogue (the dominant terms of the sums). Candidates whose two-way weight
with the target is below ``prune_weight`` (1e-10) are left out of the enumeration; their
marginal is approximated by that two-way weight.

**Source densities.** ``nu_k`` is estimated per cone with a conjugate Gamma-Poisson
model: the catalogue's all-sky mean density (``CATALOG_SKY_DENSITY``: published source
count / footprint) is a Gamma prior with ``DENSITY_PRIOR_COUNTS`` pseudo-sources, updated
with the rows fetched from the archive over the fetched area (the nearest-first cut of a
truncated cone gives the area inside the farthest returned row). The row(s) explained by
the target itself are not field sources and are not counted. A tiny cone therefore
yields the catalogue mean; a large or crowded cone yields the local density (galactic
plane, a literature-rich field around a famous object in SIMBAD/NED). Without a published
density the NWAY estimator ``(n + 1) / area`` (``source_densities_plus``) is used.

**Other objects in the cone.** Detections outside the target's association are
partitioned into physical objects by greedy agglomeration: two groups with no catalogue
in common are merged while the posterior odds of "one object" exceed 1, using the n-way
Bayes factor and the B&S prior ``P0 = N_* / prod N_k`` with ``N_*`` the count of the
sparsest catalogue (every source of the sparsest catalogue has counterparts, i.e. NWAY
with completeness 1). The group probability is the B&S posterior
``P = [1 + (1 - P0) / (B P0)]^-1`` and a member's probability is its leave-one-out
posterior (joins the rest of the group vs. is a separate object).

Candidate pairs come from scipy ``cKDTree`` range searches on unit vectors (one tree per
catalogue) and are kept when their Mahalanobis distance satisfies
``chi2 <= LINK_CHI2 = -2 ln(1e-6)`` (a true pair is missed with probability 1e-6).
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from models import (
    TARGET_PM_METHODS,
    CatalogSource,
    Target,
    _to_float,
    source_position_at,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi  # 206264.806...
LN_ARCSEC_PER_RAD = math.log(ARCSEC_PER_RAD)
LN2 = math.log(2.0)
FULL_SKY_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2  # 41252.96 deg^2
NORTH_OF_DEC_M40_DEG2 = FULL_SKY_DEG2 * (1.0 + math.sin(math.radians(40.0))) / 2.0  # 33,884.7 deg^2
THREE_PI_SR_DEG2 = 3.0 * math.pi * (180.0 / math.pi) ** 2  # Pan-STARRS 3pi survey (Dec > -30): 30,939.7 deg^2
NORTHERN_HEMISPHERE_DEG2 = FULL_SKY_DEG2 / 2.0

# Pairs are linked when their Mahalanobis chi2 (2 d.o.f.) is below this: a true pair lies
# beyond it with probability exp(-LINK_CHI2 / 2) = 1e-6.
LINK_CHI2 = -2.0 * math.log(1e-6)
# Default 1-sigma uncertainty (per axis) of a user-supplied target position: coordinates
# quoted to 0.01 s in RA / 0.1" in Dec. Resolver positions carry their own errors.
DEFAULT_TARGET_SIGMA_ARCSEC = 0.1
# Rows without any positional error (none of the registry catalogues): assumed 1".
MISSING_SIGMA_ARCSEC = 1.0
# Numerical floor on a per-axis sigma (0.1 mas; Gaia DR3 bright stars reach ~0.01 mas,
# below the ~0.02 mas Gaia-CRF3 / ICRF3 frame alignment this makes no difference).
MIN_SIGMA_ARCSEC = 1e-4
# Systematic per-axis term added in quadrature to every catalogue position: positions of
# one object in different catalogues differ well beyond their formal errors (~0.01-0.1
# mas for Gaia DR3 and VLBI). Measured on the recorded cones: 3C 273's Gaia DR3 position
# is 2.1 mas from its VLBI position (NED, SIMBAD), and M87's -- a nucleus inside a bright
# extended galaxy -- is 16 mas from it; optical-radio offsets of this size are common
# (Petrov, Kovalev & Plavin 2019, MNRAS 482, 3023). Compilations also round coordinates
# to 1e-6 deg (1.0 mas rms). 20 mas per axis covers these while staying far below the
# separation of distinct stars in the most crowded fields Gaia resolves (~0.2").
ASTROMETRIC_FLOOR_ARCSEC = 0.02
# FWHM -> Gaussian sigma.
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
# Rows of one catalogue closer than this after epoch propagation are one measurement
# listed several times (SIMBAD and the Exoplanet Archive list planets at the host's
# coordinates): they are kept together and share the representative's probabilities.
COINCIDENT_ARCSEC = 1e-3
# Default prior probability that the target has a counterpart in a catalogue (see module
# docstring: equals NWAY's default prior_completeness = 1).
DEFAULT_PRIOR_COMPLETENESS = 0.5
# NWAY --acceptable-prob default: secondary solutions have p_i > 0.5 p_best.
SECONDARY_RATIO = 0.5
MAX_STATES = 20000
PRUNE_WEIGHT = 1e-10
MAX_CANDIDATES_PER_CATALOG = 12
MAX_ALTERNATIVES = 10
# Gamma prior on the density: weight of the all-sky mean, in pseudo-sources.
DENSITY_PRIOR_COUNTS = 1.0

# Proper-motion error per year of epoch difference, per unit position error, for Gaia DR3
# rows (their pmra_error is not fetched): measured ratio pm_error / position_error at
# G <= 21, median 1.25, 90th percentile 1.4 (see models._epoch_growth); the upper decile
# is used.
PM_ERROR_PER_POSITION_ERROR = 1.4
# Proper-motion uncertainty (per axis) of a target motion of unknown quality (user input
# without an error). Resolver (SIMBAD / Gaia) motions carry their own errors.
DEFAULT_TARGET_PM_SIGMA_MASYR = 1.0
# RMS proper motion (per axis) assumed for a row of a proper-motion catalogue that has no
# measured motion (Gaia 2-parameter solutions, faint field stars): a tangential-velocity
# dispersion of ~40 km/s at ~1 kpc gives 40 / (4.74 * 1 kpc) = 8.4 mas/yr; 10 mas/yr is used.
UNKNOWN_PM_SIGMA_MASYR = 10.0
# Common epoch of the detections when the target has no epoch: the Gaia DR3 reference
# epoch, at which the most precise positions (and their proper motions) are defined.
REFERENCE_EPOCH = 2016.0


@dataclass(frozen=True, slots=True)
class SkyDensity:
    """All-sky mean surface density of a catalogue: source count over its footprint."""

    sources: int
    area_deg2: float
    reference: str

    @property
    def per_deg2(self) -> float:
        return self.sources / self.area_deg2


# Published (or counted) source numbers and footprints. Counts marked "TAP COUNT(*)"
# were obtained from the archive service on 2026-09-28 with the same table the registry
# queries.
CATALOG_SKY_DENSITY: dict[str, SkyDensity] = {
    "gaia_dr3": SkyDensity(1_811_709_771, FULL_SKY_DEG2, "Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1"),
    "simbad": SkyDensity(22_156_637, FULL_SKY_DEG2, "SIMBAD basic table, TAP COUNT(*) 2026-09-28"),
    "ned": SkyDensity(1_107_251_569, FULL_SKY_DEG2, "NED holdings, 1,107,251,569 distinct objects (ned.ipac.caltech.edu, Oct 2025)"),
    "exoplanet_archive": SkyDensity(6_372, FULL_SKY_DEG2, "NASA Exoplanet Archive pscomppars, TAP COUNT(*) 2026-09-28"),
    "vizier_2mass_reference": SkyDensity(470_992_970, FULL_SKY_DEG2, "Skrutskie et al. 2006, AJ 131, 1163 (2MASS PSC)"),
    "twomass_psc": SkyDensity(470_992_970, FULL_SKY_DEG2, "Skrutskie et al. 2006, AJ 131, 1163 (2MASS PSC)"),
    "allwise": SkyDensity(747_634_026, FULL_SKY_DEG2, "Cutri et al. 2013, AllWISE Explanatory Supplement"),
    "panstarrs_dr2": SkyDensity(
        2_264_263_282, THREE_PI_SR_DEG2,
        "Gaia DR3 documentation sect. 15.3.1: PS1 objects with nDetections > 1 and valid astrometry; "
        "3pi survey Dec > -30 (Chambers et al. 2016, arXiv:1612.05560)"),
    "sdss": SkyDensity(469_053_874, 14_555.0, "Aihara et al. 2011, ApJS 193, 29 (SDSS DR8 imaging: unique objects)"),
    "first": SkyDensity(946_432, 10_575.0, "Helfand, White & Becker 2015, ApJ 801, 26; HEASARC first TAP COUNT(*)"),
    "nvss": SkyDensity(1_773_484, NORTH_OF_DEC_M40_DEG2, "Condon et al. 1998, AJ 115, 1693; HEASARC nvss TAP COUNT(*)"),
    "vlass": SkyDensity(2_088_223, NORTH_OF_DEC_M40_DEG2,
                        "Bruzewski et al. 2021, ApJ 914, 42 (VizieR J/ApJ/914/42 table5, Flag != 2, TAP COUNT(*))"),
    "lotss": SkyDensity(13_667_877, 0.88 * NORTHERN_HEMISPHERE_DEG2,
                        "Shimwell et al. 2026, A&A 707, A198 (LoTSS-DR3: 13,667,877 sources, 88% of the northern sky)"),
    "lotss_dr2": SkyDensity(4_396_228, 5_634.0, "Shimwell et al. 2022, A&A 659, A1 (LoTSS-DR2)"),
    "rosat": SkyDensity(135_118, FULL_SKY_DEG2, "Boller et al. 2016, A&A 588, A103; HEASARC rass2rxs TAP COUNT(*)"),
    "rosat_bsc": SkyDensity(18_811, FULL_SKY_DEG2, "Voges et al. 1999, A&A 349, 389; HEASARC rassbsc TAP COUNT(*)"),
    "chandra": SkyDensity(407_806, 730.0, "Evans et al. 2024, ApJS 274, 22 (CSC 2.1: 407,806 sources, ~730 deg^2)"),
    "xmm": SkyDensity(818_656, 1_397.0, "Webb et al. 2020, A&A 641, A136; 5XMM-DR15: 818,656 unique sources, ~1397 deg^2"),
}


# ---------------------------------------------------------------------------
# Bayes factors and posteriors (B&S 2008)
# ---------------------------------------------------------------------------


def bayes_factor_2way(psi_arcsec: float, sigma1_arcsec: float, sigma2_arcsec: float) -> float:
    """B&S 2008 eq. (16): ``B = 2/(s1^2+s2^2) exp(-psi^2 / (2 (s1^2+s2^2)))`` in radians."""
    s = (sigma1_arcsec**2 + sigma2_arcsec**2) / ARCSEC_PER_RAD**2
    psi = psi_arcsec / ARCSEC_PER_RAD
    return 2.0 / s * math.exp(-psi * psi / (2.0 * s))


def _inverse(cov: tuple[float, float, float]) -> tuple[float, float, float, float]:
    vxx, vxy, vyy = cov
    det = vxx * vyy - vxy * vxy
    if not det > 0.0:
        raise ValueError(f"covariance {cov} is not positive definite")
    return vyy / det, -vxy / det, vxx / det, -math.log(det)


def ln_bayes_factor(positions_arcsec: Sequence[tuple[float, float]],
                    covariances_arcsec2: Sequence[tuple[float, float, float]]) -> float:
    """Natural log of the n-way Bayes factor (B&S 2008 eq. 18 generalised to ellipses).

    ``positions_arcsec`` are tangent-plane (east, north) offsets of the detections and
    ``covariances_arcsec2`` their (var_east, cov_east_north, var_north) in arcsec^2. The
    result is dimensionless (angles in radians, as in B&S). One detection gives 0.
    """
    n = len(positions_arcsec)
    if n != len(covariances_arcsec2):
        raise ValueError("positions and covariances differ in length")
    if n <= 1:
        return 0.0
    inv = [_inverse(tuple(c)) for c in covariances_arcsec2]  # type: ignore[arg-type]
    return _ln_bf_python([p[0] for p in positions_arcsec], [p[1] for p in positions_arcsec],
                         [w[0] for w in inv], [w[1] for w in inv], [w[2] for w in inv], [w[3] for w in inv],
                         list(range(n)))


def log10_bayes_factor(positions_arcsec: Sequence[tuple[float, float]],
                       covariances_arcsec2: Sequence[tuple[float, float, float]]) -> float:
    """log10 of the n-way Bayes factor (see :func:`ln_bayes_factor`)."""
    return ln_bayes_factor(positions_arcsec, covariances_arcsec2) / math.log(10.0)


def _ln_bf_python(x: list[float], y: list[float], wa: list[float], wb: list[float], wc: list[float],
                  ld: list[float], members: Sequence[int]) -> float:
    """ln B for ``members`` from per-detection inverse covariances (arcsec^-2).

    Numerically stable: offsets are taken from the first member and chi2 is summed
    about the precision-weighted mean (no cancellation of large terms).
    """
    n = len(members)
    if n <= 1:
        return 0.0
    i0 = members[0]
    x0, y0 = x[i0], y[i0]
    sa = sb = sc = bx = by = lds = 0.0
    for i in members:
        dx, dy = x[i] - x0, y[i] - y0
        a, b, c = wa[i], wb[i], wc[i]
        sa += a
        sb += b
        sc += c
        bx += a * dx + b * dy
        by += b * dx + c * dy
        lds += ld[i]
    det = sa * sc - sb * sb
    mx = (sc * bx - sb * by) / det
    my = (sa * by - sb * bx) / det
    chi2 = 0.0
    for i in members:
        dx, dy = x[i] - x0 - mx, y[i] - y0 - my
        chi2 += wa[i] * dx * dx + 2.0 * wb[i] * dx * dy + wc[i] * dy * dy
    return (n - 1) * (LN2 + 2.0 * LN_ARCSEC_PER_RAD) + 0.5 * lds - 0.5 * math.log(det) - 0.5 * chi2


def posterior_probability(log10_bf: float, prior: float) -> float:
    """B&S 2008 eq. (22): ``P = [1 + (1 - P0) / (B P0)]^-1`` (overflow-safe)."""
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must lie in (0, 1)")
    ln_odds = log10_bf * math.log(10.0) + math.log(prior) - math.log1p(-prior)
    return _logistic(ln_odds)


def _logistic(ln_odds: float) -> float:
    if ln_odds >= 0:
        return 1.0 / (1.0 + math.exp(-ln_odds))
    e = math.exp(ln_odds)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Source densities
# ---------------------------------------------------------------------------


def cone_area_deg2(radius_arcsec: float) -> float:
    """Area of a spherical cap of the given radius (exact, not the flat-sky pi r^2)."""
    theta = math.radians(radius_arcsec / 3600.0)
    return 2.0 * math.pi * (1.0 - math.cos(theta)) * (180.0 / math.pi) ** 2


def estimate_density_deg2(
    n_rows: int,
    area_deg2: float,
    *,
    catalog: str | None = None,
    sky_density: SkyDensity | None = None,
    n_target_rows: int = 0,
    prior_counts: float = DENSITY_PRIOR_COUNTS,
) -> tuple[float, dict[str, Any]]:
    """Field-source density (per deg^2) of a catalogue around the target.

    Conjugate Gamma-Poisson estimate: prior mean = the catalogue's all-sky density with
    ``prior_counts`` pseudo-sources, likelihood = ``n_rows - n_target_rows`` field rows
    over ``area_deg2``: ``nu = (a + n) / (a / nu_sky + area)``. Without an all-sky density
    the NWAY estimator ``(n + 1) / area`` is used. Returns (density, provenance).
    """
    if sky_density is None and catalog is not None:
        sky_density = CATALOG_SKY_DENSITY.get(catalog)
    field_rows = max(0, int(n_rows) - max(0, int(n_target_rows)))
    area = max(float(area_deg2), 0.0)
    info: dict[str, Any] = {"rows": int(n_rows), "field_rows": field_rows, "area_deg2": area}
    if sky_density is not None:
        mean = sky_density.per_deg2
        density = (prior_counts + field_rows) / (prior_counts / mean + area)
        info.update({"method": "gamma_poisson", "sky_mean_per_deg2": mean, "prior_counts": prior_counts,
                     "reference": sky_density.reference})
    else:
        if area <= 0.0:
            raise ValueError("a density needs a positive area when the catalogue has no all-sky density")
        density = (field_rows + 1.0) / area
        info.update({"method": "nway_plus_one"})
    info["density_per_deg2"] = density
    return density, info


# ---------------------------------------------------------------------------
# Detections: positions at a common epoch and covariances
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Detection:
    """One catalogue row at the common epoch: position (deg) and covariance (arcsec^2).

    ``cov`` is (var_east, cov_east_north, var_north). ``priority`` breaks ties when rows
    of one catalogue coincide (lower first: e.g. a star before its planets).
    """

    catalog: str
    ra: float
    dec: float
    cov: tuple[float, float, float]
    priority: int = 0
    label: Any = None


def ellipse_covariance(major: float, minor: float, pa_deg: float | None) -> tuple[float, float, float]:
    """(var_east, cov_en, var_north) of an error ellipse (1-sigma semi-axes, PA east of north).

    Without a position angle the orientation is unknown and the isotropic covariance with
    the same trace is returned.
    """
    if pa_deg is None or not math.isfinite(pa_deg):
        v = (major * major + minor * minor) / 2.0
        return v, 0.0, v
    t = math.radians(pa_deg)
    ue, un = math.sin(t), math.cos(t)  # major axis direction (east, north)
    ve, vn = math.cos(t), -math.sin(t)  # minor axis
    a2, b2 = major * major, minor * minor
    return a2 * ue * ue + b2 * ve * ve, a2 * ue * un + b2 * ve * vn, a2 * un * un + b2 * vn * vn


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Where a radio catalogue publishes source sizes (FWHM, arcsec).

    ``beam_arcsec`` (or the per-row ``beam_column``) is the restoring beam FWHM to remove
    in quadrature when the sizes are fitted (beam-convolved) ones; None when the
    catalogue already publishes deconvolved sizes.
    """

    major: str
    minor: str
    pa: str | None
    beam_arcsec: float | None
    reference: str
    beam_column: str | None = None


# Resolved radio sources: the host (optical/IR/X-ray counterpart) is not at the centroid of
# an asymmetric structure (one-sided jets, lobes), so the host position relative to the
# catalogued centroid is given an extra Gaussian scatter with the source's own
# (deconvolved) FWHM / 2.3548 along each axis. Without it the NVSS centroid of 3C 273,
# pulled 5.6" towards the jet with a formal 0.53" error, could never be identified.
RADIO_STRUCTURE: dict[str, StructureSpec] = {
    # HEASARC nvss: 'The fitted (deconvolved) major axis'. The '<' limit flags of
    # unresolved sources are not fetched, so an upper limit counts as a size
    # (conservative: such positions constrain less than they could).
    "nvss": StructureSpec("major_axis", "minor_axis", "position_angle", None,
                          "Condon et al. 1998, AJ 115, 1693 (deconvolved sizes)"),
    # HEASARC first fit_major_axis/fit_minor_axis are before deconvolution; beam 5.4"
    # (6.4" x 5.4" south of +4.5 deg; the smaller value gives the larger, conservative size).
    "first": StructureSpec("fit_major_axis", "fit_minor_axis", None, 5.4,
                           "Becker, White & Helfand 1995, ApJ 450, 559 (5.4 arcsec beam)"),
    # VizieR J/A+A/659/A1: Maj/Min 'INCLUDING convolution with the 6-arcsec LOFAR beam'.
    "lotss_dr2": StructureSpec("Maj", "Min", None, 6.0, "Shimwell et al. 2022, A&A 659, A1 (6 arcsec beam)"),
    # VizieR J/A+A/707/A198: fitted FWHM; per-row resolution 'Res' (6", 9" below Dec 10 deg).
    "lotss": StructureSpec("Maj", "Min", None, 6.0, "Shimwell et al. 2026, A&A 707, A198", beam_column="Res"),
}


def structure_covariance(source: CatalogSource) -> tuple[float, float, float] | None:
    """Extra covariance (arcsec^2) of a resolved radio source's host position (see RADIO_STRUCTURE)."""
    spec = RADIO_STRUCTURE.get(source.catalog)
    if spec is None:
        return None
    data = source.data or {}
    major, minor = _to_float(data.get(spec.major)), _to_float(data.get(spec.minor))
    if major is None or major <= 0:
        return None
    minor = major if minor is None or minor <= 0 else minor
    beam = _to_float(data.get(spec.beam_column)) if spec.beam_column else None
    beam = beam if beam is not None and beam > 0 else spec.beam_arcsec
    if beam is not None:
        major = math.sqrt(max(0.0, major * major - beam * beam))
        minor = math.sqrt(max(0.0, minor * minor - beam * beam))
    if major <= 0 and minor <= 0:
        return None
    pa = _to_float(data.get(spec.pa)) if spec.pa else None
    return ellipse_covariance(major * FWHM_TO_SIGMA, minor * FWHM_TO_SIGMA, pa)


def source_covariance(source: CatalogSource) -> tuple[tuple[float, float, float], str]:
    """Positional covariance of a catalogue row (arcsec^2) and how it was obtained.

    The catalogue's error ellipse (``metadata['positional_error']['ellipse_1sigma_arcsec']``)
    or per-axis errors (with Gaia's ``ra_dec_corr``) give the shape; any systematic term,
    calibration scale, floor or epoch-growth term that ``positional_error_arcsec`` (the RMS
    1-sigma circular error) contains beyond them is added isotropically, so the covariance
    always has ``trace / 2 == positional_error_arcsec^2``.
    """
    sigma = source.positional_error_arcsec
    details = (source.metadata or {}).get("positional_error") or {}
    cov: tuple[float, float, float] | None = None
    shape = "circular"
    ellipse = details.get("ellipse_1sigma_arcsec")
    if isinstance(ellipse, Mapping) and _to_float(ellipse.get("major")) is not None:
        major = float(ellipse["major"])
        minor = _to_float(ellipse.get("minor"))
        minor = major if minor is None else minor
        pa = _to_float(ellipse.get("pa_deg"))
        cov = ellipse_covariance(major, minor, pa)
        shape = "ellipse" if pa is not None else "ellipse_no_pa"
    else:
        s_ra, s_dec = _to_float(details.get("sigma_ra_arcsec")), _to_float(details.get("sigma_dec_arcsec"))
        if s_ra is not None and s_dec is not None and s_ra > 0 and s_dec > 0:
            rho = _to_float((source.data or {}).get("ra_dec_corr"))
            rho = rho if rho is not None and -1.0 < rho < 1.0 else 0.0
            cov = (s_ra * s_ra, rho * s_ra * s_dec, s_dec * s_dec)
            shape = "axes_corr" if rho else "axes"
    if cov is None:
        s = sigma if sigma is not None and sigma > 0 else MISSING_SIGMA_ARCSEC
        v = max(s, MIN_SIGMA_ARCSEC) ** 2
        return (v, 0.0, v), ("circular" if sigma is not None and sigma > 0 else "missing_default")
    half_trace = (cov[0] + cov[2]) / 2.0
    if sigma is not None and sigma > 0 and half_trace > 0:
        target = sigma * sigma
        if target >= half_trace:
            extra = target - half_trace
            cov = (cov[0] + extra, cov[1], cov[2] + extra)
        else:  # a calibration scale < 1: shrink the shape
            f = target / half_trace
            cov = (cov[0] * f, cov[1] * f, cov[2] * f)
    return _floored(cov), shape


def detection_covariance(source: CatalogSource) -> tuple[tuple[float, float, float], str]:
    """Covariance used for association: :func:`source_covariance` plus the astrometric
    floor (``ASTROMETRIC_FLOOR_ARCSEC`` in quadrature) and, for resolved radio sources,
    the host-offset term of :func:`structure_covariance`."""
    cov, shape = source_covariance(source)
    floor = ASTROMETRIC_FLOOR_ARCSEC**2
    cov = (cov[0] + floor, cov[1], cov[2] + floor)
    extra = structure_covariance(source)
    if extra is not None:
        cov = (cov[0] + extra[0], cov[1] + extra[1], cov[2] + extra[2])
        shape += "+structure"
    return cov, shape


def _floored(cov: tuple[float, float, float]) -> tuple[float, float, float]:
    floor = MIN_SIGMA_ARCSEC**2
    vxx, vxy, vyy = max(cov[0], floor), cov[1], max(cov[2], floor)
    limit = 0.999 * math.sqrt(vxx * vyy)
    return vxx, max(-limit, min(limit, vxy)), vyy


def pm_sigma_masyr(source: CatalogSource) -> float:
    """Per-axis proper-motion uncertainty (mas/yr) of a row with its own motion.

    Published pm errors when the row carries them (``pmra_error``/``pmdec_error``);
    otherwise ``PM_ERROR_PER_POSITION_ERROR`` x the position error (Gaia DR3 ratio).
    """
    data = source.data or {}
    errs = [_to_float(data.get(k)) for k in ("pmra_error", "pmdec_error")]
    errs = [e for e in errs if e is not None and e >= 0]
    if errs:
        return math.sqrt(sum(e * e for e in errs) / len(errs))
    sigma = source.positional_error_arcsec
    return PM_ERROR_PER_POSITION_ERROR * (sigma if sigma is not None else MISSING_SIGMA_ARCSEC) * 1000.0


def _epoch_gap(source: CatalogSource, epoch: float) -> float:
    if source.epoch is not None:
        return abs(float(source.epoch) - epoch)
    if source.epoch_range is not None:
        lo, hi = source.epoch_range
        return max(abs(lo - epoch), abs(hi - epoch))
    return 0.0


def source_detection(
    source: CatalogSource,
    target: Target | None,
    *,
    target_pm_sigma_masyr: float = DEFAULT_TARGET_PM_SIGMA_MASYR,
    extra_sigma_arcsec: float | None = None,
    priority: int = 0,
    label: Any = None,
    reference_epoch: float = REFERENCE_EPOCH,
    extragalactic: bool = False,
) -> tuple[Detection, dict[str, Any]]:
    """A catalogue row brought to the common epoch, with its grown covariance.

    The common epoch is the target's epoch, or ``reference_epoch`` (J2016.0, Gaia DR3)
    when the target has none. Position: :func:`models.source_position_at` (own proper
    motion; the target's motion and parallax for rows of catalogues without motions;
    stationary for pm-less rows of pm catalogues; unchanged when no motion is known).
    Uncertainty growth with the epoch difference dt (added in quadrature, isotropically):
    own motion -> ``pm_sigma_masyr(row) dt``; the target's motion ->
    ``target_pm_sigma_masyr dt``; no known motion (stationary or unchanged rows) ->
    ``UNKNOWN_PM_SIGMA_MASYR dt``. ``extra_sigma_arcsec`` (e.g. an unremoved parallax)
    is added in quadrature too. The base covariance is :func:`detection_covariance`
    (catalogue errors, astrometric floor, radio structure).

    ``extragalactic`` rows (a galaxy / QSO type or a redshift) do not move: a catalogue
    proper motion of such a row is measurement noise (e.g. SIMBAD lists Gaia's for M87),
    so the row stays at its position with no growth ("extragalactic").
    """
    cov, shape = detection_covariance(source)
    if extragalactic:
        epoch = float(target.epoch) if target is not None and target.epoch is not None else float(reference_epoch)
        info = {"epoch": epoch, "propagation": "extragalactic", "covariance_shape": shape, "pm_growth_arcsec": 0.0,
                "extra_sigma_arcsec": extra_sigma_arcsec}
        extra = (extra_sigma_arcsec or 0.0) ** 2
        cov = (cov[0] + extra, cov[1], cov[2] + extra)
        info["sigma_arcsec"] = math.sqrt((cov[0] + cov[2]) / 2.0)
        return Detection(source.catalog, source.ra, source.dec, cov, priority, label), info
    if target is not None and target.epoch is not None:
        epoch = float(target.epoch)
        ra, dec, method = source_position_at(source, epoch, target.proper_motion, (target.ra, target.dec),
                                             target.parallax_mas)
    else:
        epoch = float(reference_epoch)
        ra, dec, method = source_position_at(source, epoch, None)
    dt = _epoch_gap(source, epoch)
    if method == "source_pm":
        growth = pm_sigma_masyr(source) * dt / 1000.0
    elif method in TARGET_PM_METHODS:
        growth = target_pm_sigma_masyr * dt / 1000.0
    else:  # "stationary" / "none": the row's motion is unknown
        growth = UNKNOWN_PM_SIGMA_MASYR * dt / 1000.0
    extra = growth * growth + (extra_sigma_arcsec or 0.0) ** 2
    if extra:
        cov = (cov[0] + extra, cov[1], cov[2] + extra)
    info = {"epoch": epoch, "propagation": method, "covariance_shape": shape,
            "pm_growth_arcsec": growth, "extra_sigma_arcsec": extra_sigma_arcsec,
            "sigma_arcsec": math.sqrt((cov[0] + cov[2]) / 2.0)}
    return Detection(source.catalog, ra, dec, cov, priority, label), info


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AssociationConfig:
    """Tunable parameters of :func:`associate` (defaults documented in the module)."""

    target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC
    # Scalar (every catalogue) or {catalog: c}: prior probability of a target counterpart.
    completeness: float | Mapping[str, float] = DEFAULT_PRIOR_COMPLETENESS
    link_chi2: float = LINK_CHI2
    max_states: int = MAX_STATES
    prune_weight: float = PRUNE_WEIGHT
    max_candidates_per_catalog: int = MAX_CANDIDATES_PER_CATALOG
    secondary_ratio: float = SECONDARY_RATIO
    coincident_arcsec: float = COINCIDENT_ARCSEC

    def completeness_of(self, catalog: str) -> float:
        value = self.completeness.get(catalog, DEFAULT_PRIOR_COMPLETENESS) if isinstance(self.completeness, Mapping) \
            else self.completeness
        c = float(value)
        if not 0.0 < c < 1.0:
            raise ValueError(f"prior completeness for {catalog} must lie in (0, 1), got {c}")
        return c

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_sigma_arcsec": self.target_sigma_arcsec,
            "completeness": dict(self.completeness) if isinstance(self.completeness, Mapping) else self.completeness,
            "link_chi2": self.link_chi2, "max_states": self.max_states, "prune_weight": self.prune_weight,
            "max_candidates_per_catalog": self.max_candidates_per_catalog,
            "secondary_ratio": self.secondary_ratio, "coincident_arcsec": self.coincident_arcsec,
        }


@dataclass(slots=True)
class AssociatedGroup:
    """One physical object: detection indices and its probabilities.

    ``members`` lists every detection (coincident duplicates included, after their
    representative). ``member_probability`` maps a detection index to the posterior that
    it belongs to this object (for the target group: that it is the target's
    counterpart). ``coincident_with`` maps a duplicate to its representative.
    """

    members: list[int]
    contains_target: bool
    log10_bayes_factor: float
    match_probability: float | None
    member_probability: dict[int, float | None]
    p_any: float | None = None
    p_i: float | None = None
    match_flag: str | None = None
    log10_prior: float | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    coincident_with: dict[int, int] = field(default_factory=dict)
    # Target group only: posterior of exactly this association (no other counterpart).
    exact_probability: float | None = None


@dataclass(slots=True)
class AssociationResult:
    """Output of :func:`associate`."""

    groups: list[AssociatedGroup]
    # Marginal posterior that each detection is the target's counterpart (0 without a target).
    target_probability: np.ndarray
    p_any: float | None
    n_states: int
    exact: bool
    n_links: int
    notes: list[str] = field(default_factory=list)

    def group_of(self) -> dict[int, int]:
        """Detection index -> index of its group in ``groups``."""
        return {m: g for g, group in enumerate(self.groups) for m in group.members}


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(d)
    return np.column_stack((cd * np.cos(a), cd * np.sin(a), np.sin(d)))


def _gnomonic_arcsec(ra0: float, dec0: float, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a0, d0 = math.radians(ra0), math.radians(dec0)
    a, d = np.radians(ra), np.radians(dec)
    cos_c = math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(a - a0)
    cos_c = np.maximum(cos_c, 1e-12)
    xi = np.cos(d) * np.sin(a - a0) / cos_c
    eta = (math.cos(d0) * np.sin(d) - math.sin(d0) * np.cos(d) * np.cos(a - a0)) / cos_c
    return xi * ARCSEC_PER_RAD, eta * ARCSEC_PER_RAD


class _Arrays:
    """Vectorised per-detection quantities in the tangent plane about ``origin``."""

    def __init__(self, detections: Sequence[Detection], origin: tuple[float, float]) -> None:
        n = len(detections)
        self.n = n
        self.ra = np.fromiter((d.ra for d in detections), float, n)
        self.dec = np.fromiter((d.dec for d in detections), float, n)
        cov = np.array([d.cov for d in detections], dtype=float).reshape(n, 3)
        self.vxx, self.vxy, self.vyy = cov[:, 0], cov[:, 1], cov[:, 2]
        det = self.vxx * self.vyy - self.vxy**2
        if n and not np.all(det > 0):
            raise ValueError("every detection needs a positive-definite covariance")
        self.wa, self.wb, self.wc = self.vyy / det, -self.vxy / det, self.vxx / det
        self.ld = -np.log(det)
        self.x, self.y = _gnomonic_arcsec(origin[0], origin[1], self.ra, self.dec)
        self.unit = _unit_vectors(self.ra, self.dec)
        tr, dd = (self.vxx + self.vyy) / 2.0, np.sqrt(((self.vxx - self.vyy) / 2.0) ** 2 + self.vxy**2)
        self.sigma_major = np.sqrt(tr + dd)
        # Python lists for the small-cluster arithmetic of the field partition.
        self.lx, self.ly = self.x.tolist(), self.y.tolist()
        self.lwa, self.lwb, self.lwc, self.lld = self.wa.tolist(), self.wb.tolist(), self.wc.tolist(), self.ld.tolist()


def _chord(arcsec: float) -> float:
    return 2.0 * math.sin(min(math.pi, arcsec / ARCSEC_PER_RAD) / 2.0)


def _collapse_coincident(arr: _Arrays, detections: Sequence[Detection], cat_index: np.ndarray,
                         radius_arcsec: float) -> tuple[np.ndarray, dict[int, int]]:
    """Representative index of every detection (rows of one catalogue within
    ``radius_arcsec`` of each other are one measurement)."""
    rep = np.arange(arr.n)
    duplicates: dict[int, int] = {}
    if arr.n < 2 or radius_arcsec <= 0:
        return rep, duplicates
    for k in np.unique(cat_index):
        idx = np.flatnonzero(cat_index == k)
        if len(idx) < 2:
            continue
        tree = cKDTree(arr.unit[idx])
        pairs = tree.query_pairs(_chord(radius_arcsec), output_type="ndarray")
        if not len(pairs):
            continue
        parent = {int(i): int(i) for i in idx}
        for a, b in pairs:
            ra_, rb_ = _find(parent, int(idx[a])), _find(parent, int(idx[b]))
            if ra_ != rb_:
                parent[rb_] = ra_
        clusters: dict[int, list[int]] = {}
        for i in idx:
            clusters.setdefault(_find(parent, int(i)), []).append(int(i))
        for members in clusters.values():
            if len(members) < 2:
                continue
            best = min(members, key=lambda i: (detections[i].priority, arr.vxx[i] + arr.vyy[i], i))
            for i in members:
                rep[i] = best
                if i != best:
                    duplicates[i] = best
    return rep, duplicates


def _find(parent: dict[int, int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _links(arr: _Arrays, reps: np.ndarray, cat_index: np.ndarray, link_chi2: float) -> np.ndarray:
    """Pairs (i, j) of representatives from different catalogues with chi2 <= link_chi2.

    One cKDTree per catalogue on unit vectors; the range of a catalogue pair is set by
    the largest error ellipse in each, then the exact Mahalanobis test is vectorised.
    """
    if len(reps) < 2:
        return np.empty((0, 2), dtype=int)
    by_cat = {int(k): reps[cat_index[reps] == k] for k in np.unique(cat_index[reps])}
    trees = {k: cKDTree(arr.unit[idx]) for k, idx in by_cat.items()}
    smax = {k: float(arr.sigma_major[idx].max()) for k, idx in by_cat.items()}
    keys = sorted(by_cat)
    found: list[np.ndarray] = []
    for pos, ka in enumerate(keys):
        for kb in keys[pos + 1:]:
            r = math.sqrt(link_chi2 * (smax[ka] ** 2 + smax[kb] ** 2))
            sdm = trees[ka].sparse_distance_matrix(trees[kb], _chord(r), output_type="ndarray")
            if len(sdm):
                found.append(np.column_stack((by_cat[ka][sdm["i"]], by_cat[kb][sdm["j"]])))
    if not found:
        return np.empty((0, 2), dtype=int)
    pairs = np.concatenate(found)
    i, j = pairs[:, 0], pairs[:, 1]
    dx, dy = arr.x[j] - arr.x[i], arr.y[j] - arr.y[i]
    sxx, sxy, syy = arr.vxx[i] + arr.vxx[j], arr.vxy[i] + arr.vxy[j], arr.vyy[i] + arr.vyy[j]
    det = sxx * syy - sxy**2
    chi2 = (syy * dx * dx - 2.0 * sxy * dx * dy + sxx * dy * dy) / det
    return pairs[chi2 <= link_chi2]


def _enumerate_target(
    arr: _Arrays, cand: list[list[int]], prior_terms: list[float], target_w: tuple[float, float, float, float],
    max_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """All (or the most probable ``max_states``) associations of the target.

    ``cand[k]`` are detection indices of catalogue slot k and ``prior_terms[k]`` the
    ln prior factor ln(c/(1-c)) - ln N of one counterpart from it. Returns (choice matrix
    [state, slot] -> detection or -1, ln weight, ln Bayes factor, exact).
    """
    ta, tb, tc, tld = target_w
    sa, sb, sc = np.array([ta]), np.array([tb]), np.array([tc])
    bx, by, q = np.zeros(1), np.zeros(1), np.zeros(1)
    ld, n, prior = np.array([tld]), np.ones(1), np.zeros(1)
    choice = np.full((1, len(cand)), -1, dtype=np.int64)
    exact = True
    const = LN2 + 2.0 * LN_ARCSEC_PER_RAD

    def ln_bf(sa: np.ndarray, sb: np.ndarray, sc: np.ndarray, bx: np.ndarray, by: np.ndarray, q: np.ndarray,
              ld: np.ndarray, n: np.ndarray) -> np.ndarray:
        det = sa * sc - sb * sb
        quad = (sc * bx * bx - 2.0 * sb * bx * by + sa * by * by) / det
        chi2 = np.maximum(q - quad, 0.0)
        return (n - 1.0) * const + 0.5 * ld - 0.5 * np.log(det) - 0.5 * chi2

    for k, members in enumerate(cand):
        if not members:
            continue
        m = np.asarray(members)
        wa, wb, wc = arr.wa[m], arr.wb[m], arr.wc[m]
        xx, yy = arr.x[m], arr.y[m]
        wx, wy = wa * xx + wb * yy, wb * xx + wc * yy
        qq = xx * wx + yy * wy
        # Broadcast: existing states (rows) x (none + candidates) columns.
        sa = np.column_stack((sa, sa[:, None] + wa[None, :])).ravel()
        sb = np.column_stack((sb, sb[:, None] + wb[None, :])).ravel()
        sc = np.column_stack((sc, sc[:, None] + wc[None, :])).ravel()
        bx = np.column_stack((bx, bx[:, None] + wx[None, :])).ravel()
        by = np.column_stack((by, by[:, None] + wy[None, :])).ravel()
        q = np.column_stack((q, q[:, None] + qq[None, :])).ravel()
        ld = np.column_stack((ld, ld[:, None] + arr.ld[m][None, :])).ravel()
        n = np.column_stack((n, n[:, None] + np.ones(len(m))[None, :])).ravel()
        prior = np.column_stack((prior, prior[:, None] + np.full((1, len(m)), prior_terms[k]))).ravel()
        choice = np.repeat(choice, len(m) + 1, axis=0)
        choice[:, k] = np.tile(np.concatenate(([-1], m)), len(choice) // (len(m) + 1))
        if len(sa) > max_states:
            exact = False
            logw = prior + ln_bf(sa, sb, sc, bx, by, q, ld, n)
            keep = np.argpartition(-logw, max_states - 1)[:max_states]
            # The null association (no counterpart) is always kept: p_any needs it.
            null = np.flatnonzero((choice == -1).all(axis=1))
            keep = np.union1d(keep, null)
            sa, sb, sc, bx, by, q, ld, n, prior = (v[keep] for v in (sa, sb, sc, bx, by, q, ld, n, prior))
            choice = choice[keep]
    lnbf = ln_bf(sa, sb, sc, bx, by, q, ld, n)
    return choice, prior + lnbf, lnbf, exact


def associate(
    detections: Sequence[Detection],
    densities_deg2: Mapping[str, float],
    *,
    target: tuple[float, float] | None = None,
    config: AssociationConfig | None = None,
) -> AssociationResult:
    """Probabilistic N-way association of ``detections`` (see the module docstring).

    ``densities_deg2`` gives each catalogue's field-source density per deg^2 (see
    :func:`estimate_density_deg2`). ``target`` is the (ra, dec) of the search target at
    the detections' common epoch; its 1-sigma per-axis uncertainty is
    ``config.target_sigma_arcsec``. Without a target only the object partition is made.
    """
    cfg = config or AssociationConfig()
    n = len(detections)
    notes: list[str] = []
    if target is None:
        origin = (float(np.mean([d.ra for d in detections])) if n else 0.0,
                  float(np.mean([d.dec for d in detections])) if n else 0.0)
    else:
        origin = (float(target[0]), float(target[1]))
    arr = _Arrays(detections, origin)
    catalogs = sorted({d.catalog for d in detections})
    cat_id = {c: i for i, c in enumerate(catalogs)}
    cat_index = np.fromiter((cat_id[d.catalog] for d in detections), int, n)
    missing = [c for c in catalogs if c not in densities_deg2 or not densities_deg2[c] > 0]
    if missing:
        raise ValueError(f"no positive source density for catalogue(s) {missing}")
    ln_n = {c: math.log(float(densities_deg2[c]) * FULL_SKY_DEG2) for c in catalogs}
    ln_n_det = np.fromiter((ln_n[d.catalog] for d in detections), float, n)

    rep, duplicates = _collapse_coincident(arr, detections, cat_index, cfg.coincident_arcsec)
    reps = np.flatnonzero(rep == np.arange(n))
    followers: dict[int, list[int]] = {}
    for dup, r in duplicates.items():
        followers.setdefault(r, []).append(dup)

    target_prob = np.zeros(n)
    p_any: float | None = None
    n_states = 0
    exact = True
    groups: list[AssociatedGroup] = []
    in_target: set[int] = set()
    state_members: list[tuple[list[int], float, float]] = []  # (reps, p_i, ln B) of secondary solutions

    if target is not None:
        st = max(float(cfg.target_sigma_arcsec), MIN_SIGMA_ARCSEC)
        tw = 1.0 / (st * st)
        target_w = (tw, 0.0, tw, -math.log(st**4))
        ln_odds = {c: math.log(cfg.completeness_of(c)) - math.log1p(-cfg.completeness_of(c)) for c in catalogs}
        # Two-way weight of every representative with the target (vectorised).
        sxx, sxy, syy = arr.vxx + st * st, arr.vxy, arr.vyy + st * st
        det2 = sxx * syy - sxy**2
        chi2 = (syy * arr.x**2 - 2.0 * sxy * arr.x * arr.y + sxx * arr.y**2) / det2
        ln_b2 = LN2 + 2.0 * LN_ARCSEC_PER_RAD - 0.5 * np.log(det2) - 0.5 * chi2
        prior_det = np.fromiter((ln_odds[d.catalog] for d in detections), float, n) - ln_n_det
        ln_w2 = ln_b2 + prior_det
        keep = reps[ln_w2[reps] >= math.log(cfg.prune_weight)]
        cand: list[list[int]] = []
        slot_catalogs: list[str] = []
        for c in catalogs:
            members = [int(i) for i in keep[cat_index[keep] == cat_id[c]]]
            if not members:
                continue
            members.sort(key=lambda i: -ln_w2[i])
            if len(members) > cfg.max_candidates_per_catalog:
                notes.append(f"{c}: {len(members)} target candidates, the {cfg.max_candidates_per_catalog} most "
                             "probable were enumerated")
                members = members[:cfg.max_candidates_per_catalog]
            cand.append(members)
            slot_catalogs.append(c)
        order = sorted(range(len(cand)), key=lambda k: -max(ln_w2[i] for i in cand[k]))
        cand = [cand[k] for k in order]
        slot_catalogs = [slot_catalogs[k] for k in order]
        prior_terms = [ln_odds[c] - ln_n[c] for c in slot_catalogs]
        choice, logw, lnbf, exact = _enumerate_target(arr, cand, prior_terms, target_w, cfg.max_states)
        n_states = len(logw)
        if not exact:
            notes.append(f"beam search: {n_states} most probable associations kept per step (not exhaustive)")
        top = float(logw.max())
        w = np.exp(logw - top)
        total = float(w.sum())
        is_null = (choice == -1).all(axis=1)
        w_null = float(w[is_null].sum())
        nonnull_total = total - w_null
        p_any = 1.0 - w_null / total
        # Marginal probability of each enumerated candidate.
        enumerated: set[int] = set()
        for k in range(choice.shape[1]):
            col = choice[:, k]
            mask = col >= 0
            if mask.any():
                sums = np.zeros(n)
                np.add.at(sums, col[mask], w[mask])
                idx = np.unique(col[mask])
                target_prob[idx] = sums[idx] / total
                enumerated.update(int(i) for i in idx)
        # Everything not enumerated: its two-way weight against the same normaliser.
        others = np.setdiff1d(reps, np.fromiter(enumerated, int, len(enumerated)))
        if len(others):
            target_prob[others] = np.minimum(1.0, np.exp(ln_w2[others] - top) / total)
        for dup, r in duplicates.items():
            target_prob[dup] = target_prob[r]
        if nonnull_total > 0:
            nonnull = np.flatnonzero(~is_null)
            best = int(nonnull[np.argmax(logw[nonnull])])
            p_best = float(w[best] / nonnull_total)
            best_reps = [int(i) for i in choice[best] if i >= 0]
            # Joint posterior that every member is a counterpart (other catalogues free).
            joint = np.ones(len(w), dtype=bool)
            for k in range(choice.shape[1]):
                if choice[best, k] >= 0:
                    joint &= choice[:, k] == choice[best, k]
            in_target = set(best_reps)
            members = []
            for r in sorted(best_reps, key=lambda i: (cat_index[i], i)):
                members.append(r)
                members.extend(sorted(followers.get(r, [])))
            alternatives = []
            ranked = nonnull[np.argsort(-logw[nonnull])]
            for s in ranked[1:]:
                p_i = float(w[s] / nonnull_total)
                if p_i < cfg.secondary_ratio * p_best:
                    break
                alt = [int(i) for i in choice[s] if i >= 0]
                state_members.append((alt, p_i, float(lnbf[s])))
                if len(alternatives) < MAX_ALTERNATIVES:
                    alternatives.append({"members": sorted(alt), "p_i": p_i, "match_probability": float(w[s] / total),
                                         "log10_bayes_factor": float(lnbf[s]) / math.log(10.0)})
            groups.append(AssociatedGroup(
                members=members,
                contains_target=True,
                log10_bayes_factor=float(lnbf[best]) / math.log(10.0),
                match_probability=float(w[joint].sum() / total),
                exact_probability=float(w[best] / total),
                member_probability={m: float(target_prob[m]) for m in members},
                p_any=p_any,
                p_i=p_best,
                match_flag="best",
                log10_prior=float(logw[best] - lnbf[best]) / math.log(10.0),
                alternatives=alternatives,
                coincident_with={m: duplicates[m] for m in members if m in duplicates},
            ))
        target_norm = (top, nonnull_total, total, target_w, prior_det)
    else:
        target_norm = None

    # --- partition the remaining detections into objects -------------------------------
    rest = np.array([r for r in reps if int(r) not in in_target], dtype=int)
    links = _links(arr, rest, cat_index, cfg.link_chi2)
    field_groups = _partition(arr, rest, links, cat_index, ln_n_det)
    secondary = {i for alt, _, _ in state_members for i in alt}
    for reps_g in field_groups:
        members = []
        for r in sorted(reps_g, key=lambda i: (cat_index[i], i)):
            members.append(r)
            members.extend(sorted(followers.get(r, [])))
        lnb = _ln_bf_python(arr.lx, arr.ly, arr.lwa, arr.lwb, arr.lwc, arr.lld, reps_g)
        lnp0 = _ln_field_prior(reps_g, ln_n_det)
        if len(reps_g) > 1:
            prob = _posterior_ln(lnb, lnp0)
            member_prob: dict[int, float | None] = {}
            for r in reps_g:
                rest_g = [i for i in reps_g if i != r]
                odds = (lnb + lnp0) - (_ln_bf_python(arr.lx, arr.ly, arr.lwa, arr.lwb, arr.lwc, arr.lld, rest_g)
                                       + _ln_field_prior(rest_g, ln_n_det))
                member_prob[r] = _logistic(odds)
            for r in reps_g:
                for f in followers.get(r, []):
                    member_prob[f] = member_prob[r]
        else:
            prob = None
            member_prob = {m: None for m in members}
        p_i = None
        if target_norm is not None:
            top, nonnull_total, _total, target_w, prior_det = target_norm
            lnw = _ln_bf_with_target(arr, reps_g, target_w) + float(sum(prior_det[i] for i in reps_g))
            p_i = float(math.exp(lnw - top) / nonnull_total) if nonnull_total > 0 else 0.0
            p_i = min(1.0, p_i)
        groups.append(AssociatedGroup(
            members=members,
            contains_target=False,
            log10_bayes_factor=lnb / math.log(10.0),
            match_probability=prob,
            member_probability=member_prob,
            p_i=p_i,
            match_flag="secondary" if secondary.intersection(reps_g) else None,
            log10_prior=lnp0 / math.log(10.0) if len(reps_g) > 1 else None,
            coincident_with={m: duplicates[m] for m in members if m in duplicates},
        ))
    return AssociationResult(groups, target_prob, p_any, n_states, exact, len(links), notes)


def _ln_field_prior(members: Sequence[int], ln_n_det: np.ndarray) -> float:
    """ln P0 = ln N_* - sum ln N_k with N_* the sparsest catalogue's count (B&S sect. 3)."""
    if len(members) <= 1:
        return 0.0
    values = [float(ln_n_det[i]) for i in members]
    return min(values) - sum(values)


def _posterior_ln(ln_bf: float, ln_p0: float) -> float:
    """B&S posterior from ln B and ln P0 (P0 < 1)."""
    ln_one_minus = math.log1p(-math.exp(ln_p0)) if ln_p0 < -1e-12 else -60.0
    return _logistic(ln_bf + ln_p0 - ln_one_minus)


def _ln_bf_with_target(arr: _Arrays, members: Sequence[int], target_w: tuple[float, float, float, float]) -> float:
    """ln B of the target (at the origin) plus ``members``."""
    ta, tb, tc, tld = target_w
    x = [0.0] + [arr.lx[i] for i in members]
    y = [0.0] + [arr.ly[i] for i in members]
    wa = [ta] + [arr.lwa[i] for i in members]
    wb = [tb] + [arr.lwb[i] for i in members]
    wc = [tc] + [arr.lwc[i] for i in members]
    ld = [tld] + [arr.lld[i] for i in members]
    return _ln_bf_python(x, y, wa, wb, wc, ld, list(range(len(x))))


def _partition(arr: _Arrays, reps: np.ndarray, links: np.ndarray, cat_index: np.ndarray,
               ln_n_det: np.ndarray) -> list[list[int]]:
    """Greedy agglomeration of linked detections into objects (posterior odds > 1).

    Clusters never share a catalogue. The most probable merge is always done first
    (priority queue with lazy invalidation), so the result does not depend on the order
    of the input.
    """
    cluster_of = {int(r): int(r) for r in reps}
    members: dict[int, list[int]] = {int(r): [int(r)] for r in reps}
    catmask: dict[int, int] = {int(r): 1 << int(cat_index[r]) for r in reps}
    lw: dict[int, float] = {int(r): 0.0 for r in reps}  # ln B + ln P0 of the cluster
    version: dict[int, int] = {int(r): 0 for r in reps}
    adjacency: dict[int, set[int]] = {int(r): set() for r in reps}
    for i, j in links.tolist():
        adjacency[i].add(j)
        adjacency[j].add(i)
    lx, ly, lwa, lwb, lwc, lld = arr.lx, arr.ly, arr.lwa, arr.lwb, arr.lwc, arr.lld

    def merged_lw(a: int, b: int) -> float:
        combined = members[a] + members[b]
        return (_ln_bf_python(lx, ly, lwa, lwb, lwc, lld, combined) + _ln_field_prior(combined, ln_n_det))

    heap: list[tuple[float, int, int, int, int, float]] = []
    for i, j in links.tolist():
        value = merged_lw(i, j)
        odds = value - lw[i] - lw[j]
        if odds > 0.0:
            heap.append((-odds, min(i, j), max(i, j), 0, 0, value))
    heapq.heapify(heap)
    next_id = (max(members) + 1) if members else 0
    while heap:
        _neg, a, b, va, vb, value = heapq.heappop(heap)
        if a not in members or b not in members or version[a] != va or version[b] != vb:
            continue
        if catmask[a] & catmask[b]:
            continue
        c = next_id
        next_id += 1
        members[c] = members.pop(a) + members.pop(b)
        catmask[c] = catmask.pop(a) | catmask.pop(b)
        lw[c] = value
        version[c] = 0
        neighbours = (adjacency.pop(a) | adjacency.pop(b)) - {a, b}
        del lw[a], lw[b], version[a], version[b]
        adjacency[c] = neighbours
        for m in members[c]:
            cluster_of[m] = c
        for nb in neighbours:
            adjacency[nb].discard(a)
            adjacency[nb].discard(b)
            adjacency[nb].add(c)
            if catmask[nb] & catmask[c]:
                continue
            value_nb = merged_lw(c, nb)
            odds = value_nb - lw[c] - lw[nb]
            if odds > 0.0:
                lo, hi = (c, nb) if c < nb else (nb, c)
                heapq.heappush(heap, (-odds, lo, hi, version[lo], version[hi], value_nb))
    return [sorted(v) for v in members.values()]


def group_member_map(result: AssociationResult) -> dict[int, tuple[int, AssociatedGroup]]:
    """Detection index -> (group index, group)."""
    return {m: (g, group) for g, group in enumerate(result.groups) for m in group.members}


# ---------------------------------------------------------------------------
# Monte-Carlo sky simulation (validation / calibration)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimCatalog:
    """A synthetic catalogue: detection probability of an object, per-axis error, and
    the density of objects it detects (per arcmin^2)."""

    name: str
    sigma_arcsec: float
    density_arcmin2: float
    target_completeness: float = 0.5


def simulate_field(
    rng: np.random.Generator,
    catalogs: Sequence[SimCatalog],
    *,
    radius_arcsec: float = 30.0,
    target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC,
    object_density_arcmin2: float | None = None,
    ra0: float = 150.0,
    dec0: float = 20.0,
) -> tuple[list[Detection], list[int], tuple[float, float]]:
    """One synthetic cone with known truth.

    Field objects are a Poisson process over a cone of ``radius_arcsec`` (density
    ``object_density_arcmin2``, default the largest catalogue density); each field
    object is detected by catalogue k with probability ``density_k / object_density``,
    so catalogue k's field density is ``density_k`` and field objects are seen by several
    catalogues at once (correlated, as in real skies). The target object sits at the
    cone centre and is detected by catalogue k with probability ``target_completeness``;
    the target position is its true position plus a Gaussian error of
    ``target_sigma_arcsec``. Returns (detections, true object id per detection -- 0 is
    the target -- and the measured target (ra, dec)).
    """
    obj_density = object_density_arcmin2 or max(c.density_arcmin2 for c in catalogs)
    area_arcmin2 = math.pi * (radius_arcsec / 60.0) ** 2
    n_obj = rng.poisson(obj_density * area_arcmin2)
    r = radius_arcsec * np.sqrt(rng.random(n_obj))
    phi = rng.random(n_obj) * 2.0 * math.pi
    true_x = np.concatenate(([0.0], r * np.cos(phi)))
    true_y = np.concatenate(([0.0], r * np.sin(phi)))
    detections: list[Detection] = []
    truth: list[int] = []
    from models import offset_radec

    for cat in catalogs:
        p_field = min(1.0, cat.density_arcmin2 / obj_density)
        seen = rng.random(n_obj + 1) < np.concatenate(([cat.target_completeness], np.full(n_obj, p_field)))
        for obj in np.flatnonzero(seen):
            ex, ey = rng.normal(0.0, cat.sigma_arcsec, 2)
            ra, dec = offset_radec(ra0, dec0, float(true_x[obj] + ex), float(true_y[obj] + ey))
            v = cat.sigma_arcsec**2
            detections.append(Detection(cat.name, ra, dec, (v, 0.0, v), label=int(obj)))
            truth.append(int(obj))
    tx, ty = rng.normal(0.0, target_sigma_arcsec, 2)
    target = offset_radec(ra0, dec0, float(tx), float(ty))
    return detections, truth, target


def calibration_run(
    catalogs: Sequence[SimCatalog],
    *,
    fields: int = 200,
    seed: int = 1,
    radius_arcsec: float = 30.0,
    target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC,
    threshold: float = 0.9,
    bins: Sequence[float] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
) -> dict[str, Any]:
    """Simulate ``fields`` cones and measure completeness, purity and calibration of the
    target-association posteriors (true densities and completeness given to the model).

    completeness = fraction of true target detections with P > threshold; purity =
    fraction of detections with P > threshold that are true; reliability: for each
    posterior bin, the mean posterior and the observed fraction of true associations.
    Also reports field-object grouping purity (multi-member groups whose members are
    one true object).
    """
    rng = np.random.default_rng(seed)
    densities = {c.name: c.density_arcmin2 * 3600.0 for c in catalogs}
    cfg = AssociationConfig(target_sigma_arcsec=target_sigma_arcsec,
                            completeness={c.name: c.target_completeness for c in catalogs})
    probs: list[float] = []
    truths: list[bool] = []
    groups_total = groups_pure = 0
    for _ in range(fields):
        dets, truth, target = simulate_field(rng, catalogs, radius_arcsec=radius_arcsec,
                                             target_sigma_arcsec=target_sigma_arcsec)
        if not dets:
            continue
        result = associate(dets, densities, target=target, config=cfg)
        probs.extend(result.target_probability.tolist())
        truths.extend(t == 0 for t in truth)
        for group in result.groups:
            if group.contains_target or len(group.members) < 2:
                continue
            groups_total += 1
            groups_pure += len({truth[m] for m in group.members}) == 1
    p = np.asarray(probs)
    t = np.asarray(truths, dtype=bool)
    selected = p > threshold
    reliability = []
    for lo, hi in itertools.pairwise(bins):
        in_bin = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if in_bin.any():
            reliability.append({"bin": [lo, hi], "count": int(in_bin.sum()), "mean_probability": float(p[in_bin].mean()),
                                "true_fraction": float(t[in_bin].mean())})
    return {
        "fields": fields,
        "detections": len(p),
        "true_target_detections": int(t.sum()),
        "completeness": float((selected & t).sum() / max(1, t.sum())),
        "purity": float((selected & t).sum() / max(1, selected.sum())),
        "threshold": threshold,
        "reliability": reliability,
        "field_groups": groups_total,
        "field_group_purity": groups_pure / groups_total if groups_total else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_calibrate(args: argparse.Namespace) -> int:
    catalogs = [SimCatalog("precise", args.sigma1, args.density1, args.completeness),
                SimCatalog("medium", args.sigma2, args.density2, args.completeness),
                SimCatalog("coarse", args.sigma3, args.density3, args.completeness)]
    report = calibration_run(catalogs, fields=args.fields, seed=args.seed, radius_arcsec=args.radius,
                             threshold=args.threshold)
    print(json.dumps(report, indent=2))
    return 0


def register_cli(subparsers: Any) -> None:
    """Add ``xmatch-calibrate``: Monte-Carlo check of the association posteriors."""
    parser = subparsers.add_parser(
        "xmatch-calibrate", help="Monte-Carlo completeness/purity/calibration of the Bayesian crossmatch")
    parser.add_argument("--fields", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--radius", type=float, default=30.0, help="cone radius (arcsec)")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--completeness", type=float, default=0.7)
    parser.add_argument("--sigma1", type=float, default=0.1)
    parser.add_argument("--density1", type=float, default=5.0, help="per arcmin^2")
    parser.add_argument("--sigma2", type=float, default=0.5)
    parser.add_argument("--density2", type=float, default=2.0)
    parser.add_argument("--sigma3", type=float, default=2.0)
    parser.add_argument("--density3", type=float, default=0.3)
    parser.set_defaults(handler=_cli_calibrate)
