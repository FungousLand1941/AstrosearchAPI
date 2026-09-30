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
errors << 1 rad). For circular errors this is B&S eq. (18), and for two detections
``B = 2 / (s1^2 + s2^2) exp(-psi^2 / (2 (s1^2 + s2^2)))`` (B&S eq. 16). Error ellipses
(full 2x2 covariances) are used when the catalogue publishes them (the elliptical
generalisation of Pineau et al. 2017, A&A 597, A89, sect. 2). Every covariance is
expressed in the detection's own (east, north) frame and mapped into the common
tangent plane with the Jacobian of the gnomonic projection, so ellipses keep their
orientation near the poles.

**Heavy-tailed positions.** Real catalogue positions have non-Gaussian tails (blends,
host offsets, proper-motion errors). Measured on 1929 isolated Gaia-CRF3 quasars
(G < 17.5, RUWE < 1.2) against AllWISE (cc_flags 0000, nb = 1, ext_flg = 0): 7.2% of
the offsets exceed the 99% Gaussian contour of sigra/sigdec (+ the 20 mas floor) and
1.8% exceed chi2 = 27.6 (Gaussian: 1e-6). A two-component fit gives a core scale of
1.10 and ``OUTLIER_FRACTION`` = 6.1% of offsets from a component ``OUTLIER_SCALE``
times wider (3.5 for clean sources, 7.1 with blends). A counterpart's position is
therefore a mixture ``(1 - eps) N(C) + eps N(kappa^2 C)`` (eps = 0.06, kappa = 5): every
candidate enters the target association as a "core" and a "tail" hypothesis (a copy
whose weight is below ``OUTLIER_PRUNE_RATIO`` of the other is left out), so a real
counterpart 5-10 sigma off gets a small but not 1e-17 probability.

**Target-centric association (NWAY; Salvato et al. 2018, MNRAS 473, 4937, App. B).**
The search target is the "primary catalogue": a detection at the requested position
with its own uncertainty (resolver or user supplied). Every *association* -- for each
catalogue either no counterpart or exactly one of its candidates -- is a hypothesis.
Its unnormalised posterior weight relative to "no counterpart anywhere" is

    W_a = B(target + a) prod_{k in a} [c_k / (1 - c_k)] / N_k  x  L(C \\ a) / L(C),   W_null = 1,

where ``N_k = 4 pi nu_k`` is catalogue k's local source density expressed as a number of
objects on the whole sky and ``c_k`` is the prior probability that the target has a
counterpart in catalogue k (see *Priors* below). Without the last factor this is NWAY's
weight ``B * nu_primary prod c / prod nu_plus`` with NWAY's completeness equal to the
prior *odds* c/(1-c), exact when the rows that are not the target's counterparts are an
independent Poisson field per catalogue.

**Correlated fields.** In the real sky a neighbouring star appears in Gaia, 2MASS and
WISE together, so its rows are not independent: the chance that they sit near the
target is that of *one* object, not the product of three. ``L(Y)`` is the partition
function of the rows Y (the target candidates C and the rows linked to them) as field
objects under an explicit generative model: field objects of density
``nu_obj = OBJECT_DENSITY_FACTOR x`` the densest catalogue's, each detected by catalogue k
with probability ``q_k = nu_k / nu_obj``. A row alone (an object seen by catalogue k only)
then has density ``nu_obj q_k prod_{l != k} (1 - q_l) = nu_k Q / (1 - q_k)``
(``Q = prod (1 - q_l)``), so its prior ``1 / nu_k`` gets the "lone-row" factor
``(1 - q_k) / Q``, and rows of one object weigh ``B_mix(o) / (N_obj Q)^(|o| - 1)`` relative to separate rows
(``B_mix``: the Bayes factor summed over which rows are outliers), so
``L(Y) = sum_{partitions of Y} prod_{objects o} B_mix(o) / (N_obj Q)^(|o| - 1)``. The
posteriors are exact under this model (checked against brute-force enumeration) when
every linked group has at most ``FIELD_EXACT_MAX_ROWS`` rows and ``FIELD_OBJECT_BUDGET``
possible objects; larger groups use their greedy partition plus the first-order
alternatives of its lone rows. Rows placed far from where they were measured (fast
movers) are left out of the model and keep NWAY's independent-row prior; rows within their
link reach of the searched cone's edge (their partners may lie outside) stay in the
partition function -- a star seen by four catalogues near the edge is one field object --
and their lone-row factor counts only the part of their partners' region inside the cone
(``_observed_fractions``): a partner beyond the edge was not fetched, so a row there that
looks alone may not be. Simulations therefore draw field objects beyond the cone edge too
and keep the detections measured inside it, as a cone search does.
Every association's correction ``ln L(C \\ a)`` is computed block by block (each exactly
summed component, or greedy object, on its remaining rows); where greedy objects exist, the
(at most ``MAX_CORRECTED_STATES``) most probable associations get the full value, which lets
the remaining rows re-pair across objects (see ``_corrected_weights``).
Skies simulated with this generative model (``simulate_field``, independent detection)
are calibrated without slack. Real skies differ from it: they detect objects *nested* by
brightness (an object seen by the shallowest catalogue is seen by the deeper ones) and
their object density is not exactly OBJECT_DENSITY_FACTOR x the densest catalogue's.
Simulated with nested detection (1500 cones, 4 catalogues) every posterior sextile stays
within 3 binomial sigma (|true fraction - mean posterior| <= 0.01); with nested detection
and twice the modelled object density, within 0.02, on the conservative side (true
fraction above the posterior) -- ``test_posteriors_on_misspecified_skies``.

Following NWAY (nwaylib ``_compute_final_probabilities``):

* ``p_any = 1 - W_null / sum_a W_a`` -- probability that the target has any counterpart;
* ``p_i = W_a / sum_{a != null} W_a`` -- probability of association a given one exists;
* ``match_flag`` 'best' for the most probable non-null association and 'secondary' for
  those with ``p_i > 0.5 p_best`` (NWAY ``--acceptable-prob`` default 0.5).

The target group is the most probable non-null association, reported only when
``p_any >= TARGET_GROUP_MIN_P_ANY`` (0.5): otherwise no row is more likely than not the
target's counterpart and every row is partitioned as a field object (its
``target_probability`` is still reported). The target group's ``match_probability`` is
the joint posterior that *every* member is a counterpart of the target (summed over
what the other catalogues contribute); ``exact_probability`` is the posterior of exactly
this association (NWAY ``p_any * p_i``). The per-detection probability is the marginal
posterior ``P(j is the target's counterpart) = sum_{a containing j} W_a / sum_a W_a``.

All associations are enumerated when there are at most ``max_states`` of them (exact);
otherwise a beam search keeps the ``max_states`` most probable partial associations
after each catalogue. Candidates whose two-way weight with the target is below
``prune_weight`` (1e-4 of the null hypothesis), or beyond the
``max_candidates_per_catalog`` most probable of their catalogue, are left out of the
enumeration; to first order each adds its two-way weight times the associations with
no row from its catalogue, which is its marginal and is counted in the normalisation.

**Two placements per row.** A row can be placed differently under "it is the target's
counterpart" and "it is a field object". A row of a catalogue without proper motions
(2MASS, AllWISE, ...) is the target only if it moves with the target: it is placed with
the target's motion for the association and where it was measured for the field
partition. A row of a proper-motion catalogue without a measured motion (a Gaia
2-parameter solution: bright close binaries such as Kruger 60) is a stationary field
star, *or* the target moving with the target's motion: both hypotheses are weighed.

**Undated targets.** Coordinates without an epoch are compared as given, but their
epoch is unknown: it is marginalised over ``UNDATED_TARGET_EPOCHS`` -- J2000.0 (SIMBAD,
NED and literature compilations) and J2016.0 (Gaia DR3), equal prior. Under each
hypothesis a row with its own proper motion is placed at that epoch, with
``UNDATED_EPOCH_SPREAD_YR`` of its motion along the track (Gaia DR2 J2015.5 positions,
compilations at other epochs); the enumeration runs once per hypothesis and the
associations of both are summed.

**Source densities.** ``nu_k`` is estimated per cone with a conjugate Gamma-Poisson
model: the prior mean is the catalogue's density *at the target position* -- the Gaia
DR3 density map (CDS HiPS ``CDS/P/DM/I/355/gaiadr3`` summed to HEALPix order 5, 1.8
deg pixels) scaled with ``DENSITY_MAP_SCALING`` for the catalogues whose counts follow
the stars, else the all-sky mean (``CATALOG_SKY_DENSITY``: published source count /
footprint) -- with ``DENSITY_PRIOR_COUNTS`` pseudo-sources, updated with the rows
fetched from the archive over the fetched area (the nearest-first cut of a truncated
cone gives the area inside the farthest returned row). The row(s) explained by the
target itself are not field sources and are not counted. A catalogue without a
published density gets ``GENERIC_SKY_DENSITY_DEG2`` as its prior mean.

**Priors.** ``completeness_prior`` gives ``c_k`` by target class: 0.5 (NWAY's default
prior_completeness = 1) for unidentified and extragalactic targets and for the
optical/infrared/identity catalogues; for an ordinary *star*, the fraction of Gaia DR3
stars of its parallax that have a counterpart in the radio or X-ray catalogue, measured
by the excess-density method (``STELLAR_COUNTERPART_FRACTION``; a radio / X-ray
catalogue without its own calibration uses the table of its wavelength,
``WAVELENGTH_STELLAR_FRACTION``, or the table its definition declares); for an *extended*
target (a cluster, nebula, supernova remnant, group of galaxies) ``EXTENDED_POINT_COMPLETENESS``
in the optical / infrared point-source catalogues, whose rows are stars or galaxies inside
the object, not the object. A per-row ``Detection.prior_ln_odds`` multiplies the prior odds
of one row: the row of the identity catalogue that *is* the resolved name, and -- for an
extended target -- compact rows of the other catalogues (stars, galaxies and compact radio /
X-ray sources inside the object; see ``emission_extent_arcsec``), which get the odds of
``EXTENDED_POINT_COMPLETENESS`` (the crossmatch service sets both).

**Other objects in the cone.** Detections outside the target's association are
partitioned into physical objects by greedy agglomeration: two groups with no catalogue
in common are merged while the posterior odds of "one object" exceed 1, using the n-way
Bayes factor and the B&S prior ``P0 = N_* / prod N_k`` with ``N_*`` the count of the
sparsest catalogue. The group probability is the B&S posterior
``P = [1 + (1 - P0) / (B P0)]^-1`` and a member's probability is its leave-one-out
posterior (joins the rest of the group vs. is a separate object).

Rows of one catalogue that are one measurement listed twice -- closer than
``COINCIDENT_ARCSEC``, or closer than ``DUPLICATE_RESOLUTION_FRACTION`` of the
catalogue's resolution and consistent within their errors (Pan-STARRS DR2 duplicates a
few mas apart) -- are collapsed onto one representative and share its probabilities; so are
rows the caller marks as listings of one object (``Detection.listing``: a compilation's
duplicate entries, a fast mover detected by one survey at several epochs).

Candidate pairs come from scipy ``cKDTree`` range searches on unit vectors (one tree per
catalogue) and are kept when their Mahalanobis distance satisfies
``chi2 <= LINK_CHI2 = -2 ln(1e-6)``.
"""

from __future__ import annotations

import argparse
import base64
import functools
import heapq
import itertools
import json
import math
import zlib
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
    offset_radec,
    propagate_radec,
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
# Floor of the 1-sigma uncertainty (per axis) of a user-supplied target position (arcsec).
# Coordinates typed with few digits get more: see ``coordinate_sigma_arcsec``. Resolver
# positions carry their own errors.
DEFAULT_TARGET_SIGMA_ARCSEC = 0.1
# Rows without any positional error (none of the registry catalogues): assumed 1".
MISSING_SIGMA_ARCSEC = 1.0
# Numerical floor on a per-axis sigma (0.1 mas; Gaia DR3 bright stars reach ~0.01 mas,
# below the ~0.02 mas Gaia-CRF3 / ICRF3 frame alignment this makes no difference).
MIN_SIGMA_ARCSEC = 1e-4
# Largest per-axis sigma a row's covariance may have (10 deg): larger (finite) published
# errors, ellipse axes or source sizes are clamped to it -- such a row carries no positional
# information on arcsecond scales either way -- so squares never overflow.
MAX_SIGMA_ARCSEC = 36_000.0
# Largest target position uncertainty accepted from a caller (1 deg per axis).
MAX_TARGET_SIGMA_ARCSEC = 3600.0
# Prior odds factor of the row of the resolver's identity catalogue whose identifier IS the
# resolved name (Sesame answered 'M 13' from SIMBAD: SIMBAD's 'M 13' row is the target by
# definition; the residual doubt is a wrong cross-identification, taken as 1 in 10^4).
IDENTITY_PRIOR_LN_ODDS = math.log(1e4)
# Systematic per-axis term added in quadrature to every catalogue position: positions of
# one object in different catalogues differ well beyond their formal errors (~0.01-0.1
# mas for Gaia DR3 and VLBI). Measured on the recorded cones: 3C 273's Gaia DR3 position
# is 2.1 mas from its VLBI position (NED, SIMBAD), and M87's -- a nucleus inside a bright
# extended galaxy -- is 16 mas from it; optical-radio offsets of this size are common
# (Petrov, Kovalev & Plavin 2019, MNRAS 482, 3023). Compilations also round coordinates
# to 1e-6 deg (1.0 mas rms). 20 mas per axis covers these while staying far below the
# separation of distinct stars in the most crowded fields Gaia resolves (~0.2").
ASTROMETRIC_FLOOR_ARCSEC = 0.02
# Heavy tails of real positional errors (see the module docstring): fraction of
# counterparts whose offset follows a component OUTLIER_SCALE times wider than the
# catalogue covariance. Fit to 1929 Gaia-CRF3 quasar / AllWISE pairs: eps = 0.061,
# kappa = 3.5 (clean) - 7.1 (all rows); kappa = 5 is used.
OUTLIER_FRACTION = 0.06
OUTLIER_SCALE = 5.0
# A core or tail hypothesis of a candidate whose two-way weight is below this fraction
# of the other's is not enumerated (it changes the candidate's weight by < 0.1%).
OUTLIER_PRUNE_RATIO = 1e-3
# FWHM -> Gaussian sigma.
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
# Rows of one catalogue closer than this after epoch propagation are one measurement
# listed several times (SIMBAD and the Exoplanet Archive list planets at the host's
# coordinates): they are kept together and share the representative's probabilities.
COINCIDENT_ARCSEC = 1e-3
# Rows of one catalogue closer than this fraction of its resolution (FWHM of the image,
# CATALOG_RESOLUTION_ARCSEC) and consistent within their errors (chi2 <= DUPLICATE_MAX_CHI2,
# 99% for 2 d.o.f.) are duplicate listings of one source: the catalogue could not have
# resolved them (Pan-STARRS DR2 lists some objects twice, 9-23 mas apart).
DUPLICATE_RESOLUTION_FRACTION = 0.25
DUPLICATE_MAX_CHI2 = -2.0 * math.log(0.01)
CATALOG_RESOLUTION_ARCSEC: dict[str, float] = {
    "gaia_dr3": 0.4,  # close pairs are complete beyond ~0.4" (Fabricius et al. 2021, A&A 649, A5)
    "panstarrs_dr2": 1.1,  # median seeing of the PS1 3pi survey (Chambers et al. 2016)
    "sdss": 1.4,  # median r-band seeing (York et al. 2000)
    "twomass_psc": 2.5,  # Skrutskie et al. 2006
    "vizier_2mass_reference": 2.5,
    "allwise": 6.1,  # W1 PSF FWHM (Wright et al. 2010)
    "first": 5.4,
    "nvss": 45.0,
    "vlass": 2.5,
    "lotss": 6.0,
    "lotss_dr2": 6.0,
    "rosat": 30.0,
    "rosat_bsc": 30.0,
    "chandra": 0.8,
    "xmm": 6.0,
}
# Default prior probability that the target has a counterpart in a catalogue (see module
# docstring: equals NWAY's default prior_completeness = 1).
DEFAULT_PRIOR_COMPLETENESS = 0.5
# NWAY --acceptable-prob default: secondary solutions have p_i > 0.5 p_best.
SECONDARY_RATIO = 0.5
# The most probable association is reported as the target group (contains_target,
# match_flag 'best') only when the target has a counterpart with at least this probability.
TARGET_GROUP_MIN_P_ANY = 0.5
MAX_STATES = 20000
PRUNE_WEIGHT = 1e-4
MAX_CANDIDATES_PER_CATALOG = 12
MAX_ALTERNATIVES = 10
# Gamma prior on the density: weight of the prior mean, in pseudo-sources.
DENSITY_PRIOR_COUNTS = 1.0
# Prior mean density of a catalogue without a published one (e.g. a VizieR table
# registered at run time): about the median of the registry catalogues (559 per deg^2),
# rounded up (a higher density gives more conservative posteriors).
GENERIC_SKY_DENSITY_DEG2 = 1000.0
# Linked groups of rows whose field partition function is summed exactly: at most this
# many rows and this many possible objects per row (see _FieldPartitionFunction).
FIELD_EXACT_MAX_ROWS = 10
FIELD_OBJECT_BUDGET = 256
# Associations whose weight, even after the largest field correction, is below e^-40 of the
# best are dropped from the sums.
CORRECTION_NEGLIGIBLE_LN = 40.0
# At most this many associations get their full field correction (most probable first by
# the block-by-block estimate); the rest keep the estimate (see _corrected_weights).
MAX_CORRECTED_STATES = 256
# Correlated field model: field objects have OBJECT_DENSITY_FACTOR x the density of the
# densest queried catalogue, and catalogue k detects each with probability
# q_k = nu_k / nu_obj (at most MAX_DETECTION_FRACTION).
OBJECT_DENSITY_FACTOR = 1.5
# Field rows linked (directly or through other rows) to the target candidates that enter
# the field partition function, at most this many (nearest hops first).
FIELD_MAX_NEIGHBOURS = 40
# Rows left out of the enumeration with a two-way weight above EXTENSION_MIN_WEIGHT (of the
# null hypothesis) enter the field partition function too, at most this many.
FIELD_MAX_EXTENSION_ROWS = 30
EXTENSION_MIN_WEIGHT = 1e-8
# Rows whose counterpart placement (with the target's motion) is farther than this from
# their measured position are left out of the correlated-field partition function.
MOVED_ROW_ARCSEC = 1.0
MAX_DETECTION_FRACTION = 0.9

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
# The Gaia DR3 reference epoch, at which the most precise positions are defined.
REFERENCE_EPOCH = 2016.0
# Epochs a target position without an epoch may have (equal prior): J2000.0 (SIMBAD, NED
# and most literature coordinates) and J2016.0 (Gaia DR3).
UNDATED_TARGET_EPOCHS = (2000.0, REFERENCE_EPOCH)
# Spread (years, 1 sigma) of the actual epoch around each of them, applied along a row's
# motion: Gaia DR2 positions are J2015.5; compilations list some positions at other epochs.
UNDATED_EPOCH_SPREAD_YR = 0.5
# A row whose motion over the undated window is below this stays at one placement.
UNDATED_MOTION_MIN_ARCSEC = 0.05
# Epoch span assumed for a row whose epoch is entirely unknown (no row epoch, no catalogue
# span): the era of the registry's surveys, ROSAT (1990) to VLASS (2020). Its epoch
# difference to any epoch is then never taken as zero.
UNKNOWN_EPOCH_RANGE = (1990.0, 2020.0)


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


def _ln_bf_python(x: Sequence[float], y: Sequence[float], wa: Sequence[float], wb: Sequence[float],
                  wc: Sequence[float], ld: Sequence[float], members: Sequence[int]) -> float:
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


def _logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(v - top) for v in values))


# ---------------------------------------------------------------------------
# Target-dependent priors
# ---------------------------------------------------------------------------

# Catalogues whose counterparts of STARS are rare and distance dependent (flux limited
# radio and X-ray surveys). For a stellar target the prior completeness is the fraction
# of Gaia DR3 stars (parallax_over_error > 10) of that parallax with a counterpart,
# measured on 12,815 random stars (315 with parallax > 100 mas, 2500 in each of the bins
# 30-100, 10-30, 3-10, 1-3 and 0.3-1 mas; nodes at the bin medians 125.3, 37.0, 12.7, 3.98,
# 1.49 and 0.63 mas) by the excess-density method on 2026-09-29 (HEASARC TAP uploads):
# stars with a row within the matching radius (NVSS 15", FIRST 3", 2RXS 30", CSC 3",
# 4XMM 6"; the star moved to the survey epoch) minus the same count 4' north, over the
# stars inside the footprint (NVSS: Dec > -40; FIRST, CSC, 4XMM: a row within 5').
# Measured (+- 1 sigma): 2RXS 0.444 +- 0.038, 0.156 +- 0.008, 0.032 +- 0.004,
# 0.0020 +- 0.0009, then consistent with 0; NVSS 0.035 +- 0.012, then consistent with 0
# (+- 0.0014); FIRST 0.083 +- 0.042, 0.0023 (1 star), then 0; CSC 0.17, 0.32, 0.16,
# 0.042, 0, 0 (of 54-104 covered stars); 4XMM 0.38, 0.43, 0.20, 0.063, 0.041, 0.008.
# Where a bin is consistent with zero, the fraction follows the flux limit from the last
# significant bin: fraction ~ parallax^2 (a star's flux ~ parallax^2). Nodes are
# interpolated in log-log, extrapolated with parallax^2 below the lowest and held
# constant above the highest. Not measured: VLASS and LoTSS use FIRST's values (mJy-level
# radio surveys), the ROSAT Bright Source Catalogue 2RXS's x 0.14 (its share of 2RXS).
_RASS = ((0.63, 5.0e-5), (1.49, 2.8e-4), (3.98, 2.0e-3), (12.7, 0.0316), (37.0, 0.156), (125.3, 0.444))
_FIRST = ((0.63, 6.7e-7), (1.49, 3.7e-6), (3.98, 2.7e-5), (12.7, 2.7e-4), (37.0, 0.0023), (125.3, 0.083))
STELLAR_COUNTERPART_FRACTION: dict[str, tuple[tuple[float, float], ...]] = {
    "nvss": ((0.63, 8.9e-7), (1.49, 5.0e-6), (3.98, 3.5e-5), (12.7, 3.6e-4), (37.0, 2.8e-3), (125.3, 0.035)),
    "first": _FIRST,
    "vlass": _FIRST,
    "lotss": _FIRST,
    "lotss_dr2": _FIRST,
    "rosat": _RASS,
    "rosat_bsc": tuple((p, 0.14 * f) for p, f in _RASS),
    "chandra": ((0.63, 1.1e-3), (1.49, 5.9e-3), (3.98, 0.042), (12.7, 0.163), (37.0, 0.317), (125.3, 0.167)),
    "xmm": ((0.63, 7.8e-3), (1.49, 0.041), (3.98, 0.063), (12.7, 0.20), (37.0, 0.425), (125.3, 0.379)),
}
# Radio / X-ray catalogues without their own calibration (e.g. a TGSS, RACS or eRASS table
# registered from VizieR) use the table of their wavelength: the mJy-level radio surveys'
# (FIRST) and the all-sky X-ray survey's (2RXS). A definition may declare its own table in
# ``parameters["stellar_counterpart_fraction"]`` ([[parallax_mas, fraction], ...]).
WAVELENGTH_STELLAR_FRACTION: dict[str, tuple[tuple[float, float], ...]] = {
    "radio": _FIRST, "millimeter": _FIRST, "xray": _RASS, "x-ray": _RASS, "euv": _RASS, "gamma": _RASS,
}
# Parallax assumed for a star whose parallax is unknown (a SIMBAD star without one):
# the median of Gaia DR3 stars with G < 17 is ~0.5 mas; 1 mas is used (more generous).
UNKNOWN_STELLAR_PARALLAX_MAS = 1.0
# Extended targets (clusters of stars or galaxies, nebulae, HII regions, supernova
# remnants): a row of an optical / infrared point-source catalogue near the nominal centre
# is a star or galaxy inside the object, not the object itself. Their prior probability of
# being "the counterpart" is this (1 in 100); identity catalogues (SIMBAD, NED) and the
# radio / X-ray catalogues, which detect the extended emission itself (a supernova
# remnant, the intracluster gas of a galaxy cluster), keep the default.
EXTENDED_POINT_COMPLETENESS = 0.01
# Wavelengths of point-source catalogues (of the registry's and of VizieR-registered ones).
POINT_SOURCE_WAVELENGTHS = frozenset({"optical", "infrared", "uv", "near-infrared", "mid-infrared", "exoplanet"})
# Built-in catalogues by wavelength, for callers that do not pass the wavelength.
CATALOG_WAVELENGTH: dict[str, str] = {
    "gaia_dr3": "optical", "panstarrs_dr2": "optical", "sdss": "optical", "twomass_psc": "infrared",
    "vizier_2mass_reference": "infrared", "allwise": "infrared", "exoplanet_archive": "exoplanet",
    "simbad": "multi", "ned": "extragalactic", "first": "radio", "nvss": "radio", "vlass": "radio", "lotss": "radio",
    "lotss_dr2": "radio", "rosat": "xray", "rosat_bsc": "xray", "chandra": "xray", "xmm": "xray",
}
TARGET_CLASSES = ("unknown", "star", "extragalactic", "extended")


def _fraction_table(value: Any) -> tuple[tuple[float, float], ...] | None:
    """A declared stellar-counterpart table ([[parallax_mas, fraction], ...], increasing
    parallax, positive values), or None when absent / malformed."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return None
    nodes: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            return None
        plx, frac = _to_float(item[0]), _to_float(item[1])
        if plx is None or frac is None or not plx > 0 or not 0 < frac < 1:
            return None
        nodes.append((plx, frac))
    nodes.sort()
    return tuple(nodes)


def completeness_prior(catalog: str, target_class: str = "unknown", parallax_mas: float | None = None, *,
                       wavelength: str | None = None, fractions: Any = None) -> float:
    """Prior probability ``c_k`` that a target of ``target_class`` has a counterpart in ``catalog``.

    ``unknown`` and ``extragalactic`` targets, and every catalogue without a stellar
    calibration: ``DEFAULT_PRIOR_COMPLETENESS``. Stars in radio / X-ray catalogues: the
    counterpart fraction at the star's parallax (``UNKNOWN_STELLAR_PARALLAX_MAS`` when
    unknown), capped at the default and floored at 1e-7 -- from ``fractions`` (a table
    declared by the catalogue's definition), else ``STELLAR_COUNTERPART_FRACTION[catalog]``,
    else the table of the catalogue's ``wavelength`` (``WAVELENGTH_STELLAR_FRACTION``).
    ``extended`` targets: ``EXTENDED_POINT_COMPLETENESS`` in optical / infrared
    point-source catalogues, the default elsewhere.
    """
    if target_class not in TARGET_CLASSES:
        raise ValueError(f"target_class must be one of {TARGET_CLASSES}, got {target_class!r}")
    band = str(wavelength or CATALOG_WAVELENGTH.get(catalog) or "").strip().lower()
    if target_class == "extended":
        return EXTENDED_POINT_COMPLETENESS if band in POINT_SOURCE_WAVELENGTHS else DEFAULT_PRIOR_COMPLETENESS
    if target_class != "star":
        return DEFAULT_PRIOR_COMPLETENESS
    table = _fraction_table(fractions) or STELLAR_COUNTERPART_FRACTION.get(catalog) or WAVELENGTH_STELLAR_FRACTION.get(band)
    if not table:
        return DEFAULT_PRIOR_COMPLETENESS
    plx = parallax_mas if parallax_mas is not None and parallax_mas > 0 else UNKNOWN_STELLAR_PARALLAX_MAS
    x = math.log(plx)
    nodes = [(math.log(p), f) for p, f in table]
    if x <= nodes[0][0]:
        value = nodes[0][1] * (plx / table[0][0]) ** 2  # flux-limited: ~ parallax^2
    elif x >= nodes[-1][0]:
        value = nodes[-1][1]
    else:
        value = nodes[-1][1]
        for (x0, f0), (x1, f1) in itertools.pairwise(nodes):
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                value = math.exp((1 - t) * math.log(f0) + t * math.log(f1))
                break
    return float(min(DEFAULT_PRIOR_COMPLETENESS, max(value, 1e-7)))


def completeness_priors(catalogs: Sequence[str], target_class: str = "unknown",
                        parallax_mas: float | None = None, *, wavelengths: Mapping[str, str] | None = None,
                        fractions: Mapping[str, Any] | None = None) -> dict[str, float]:
    """``{catalog: completeness_prior(catalog, target_class, parallax_mas, ...)}`` with each
    catalogue's wavelength and declared stellar table (``wavelengths`` / ``fractions``)."""
    wavelengths = wavelengths or {}
    fractions = fractions or {}
    return {c: completeness_prior(c, target_class, parallax_mas, wavelength=wavelengths.get(c),
                                  fractions=fractions.get(c)) for c in catalogs}


# ---------------------------------------------------------------------------
# Source densities
# ---------------------------------------------------------------------------


def cone_area_deg2(radius_arcsec: float) -> float:
    """Area of a spherical cap of the given radius (exact, not the flat-sky pi r^2)."""
    theta = math.radians(radius_arcsec / 3600.0)
    return 2.0 * math.pi * (1.0 - math.cos(theta)) * (180.0 / math.pi) ** 2


def _spread_bits(v: int) -> int:
    out = 0
    bit = 0
    while v:
        out |= (v & 1) << (2 * bit)
        v >>= 1
        bit += 1
    return out


def healpix_nested_index(order: int, ra_deg: float, dec_deg: float) -> int:
    """HEALPix NESTED pixel index of (ra, dec) at ``order`` (nside = 2^order).

    The ang2pix algorithm of Gorski et al. 2005 (ApJ 622, 759; ``ang2pix_nest_z_phi``
    of the HEALPix C++ library), in pure Python.
    """
    nside = 1 << order
    z = math.sin(math.radians(dec_deg))
    za = abs(z)
    tt = (math.radians(ra_deg) % (2.0 * math.pi)) / (0.5 * math.pi)  # [0, 4)
    if za <= 2.0 / 3.0:
        temp1 = nside * (0.5 + tt)
        temp2 = nside * (z * 0.75)
        jp = int(temp1 - temp2)
        jm = int(temp1 + temp2)
        ifp, ifm = jp >> order, jm >> order
        if ifp == ifm:
            face = 4 if ifp == 4 else ifp + 4
        elif ifp < ifm:
            face = ifp
        else:
            face = ifm + 8
        ix = jm & (nside - 1)
        iy = nside - (jp & (nside - 1)) - 1
    else:
        ntt = min(3, int(tt))
        tp = tt - ntt
        tmp = nside * math.sqrt(3.0 * (1.0 - za))
        jp = min(nside - 1, int(tp * tmp))
        jm = min(nside - 1, int((1.0 - tp) * tmp))
        if z >= 0:
            face, ix, iy = ntt, nside - jm - 1, nside - jp - 1
        else:
            face, ix, iy = ntt + 8, jp, jm
    return (face << (2 * order)) + _spread_bits(ix) + (_spread_bits(iy) << 1)


DENSITY_MAP_ORDER = 5
# log10(nu_k / nu_k,sky) = offset + alpha * log10(nu_gaia,map / nu_gaia,sky): how a
# catalogue's density follows the Gaia DR3 map. Fitted to TAP COUNT(*)s in 0.1-deg cones
# at 36 positions stratified in Gaia density (2,200 - 1.1 million per deg^2; VizieR
# I/355, II/246, II/328, II/349, V/154 and SIMBAD, 2026-09-28): rms residuals 0.12 dex
# (2MASS), 0.04 (AllWISE: confusion-limited at ~2e4 per deg^2 everywhere), 0.08 (PS1),
# 0.13 (SDSS, 6 fields), 0.33 (SIMBAD). The Gaia map itself agrees with Gaia counts
# (median ratio 1.04, 68% within 5%). Other catalogues keep their all-sky mean.
DENSITY_MAP_SCALING: dict[str, tuple[float, float]] = {
    "gaia_dr3": (1.0, 0.0),
    "twomass_psc": (0.755, 0.071),
    "vizier_2mass_reference": (0.755, 0.071),
    "allwise": (0.091, 0.048),
    "panstarrs_dr2": (0.743, 0.101),
    "sdss": (0.234, 0.174),
    "simbad": (0.496, 0.002),
}


@functools.lru_cache(maxsize=1)
def _gaia_density_map() -> np.ndarray:
    """Gaia DR3 source density (per deg^2) per HEALPix order-5 NESTED pixel.

    Built from the CDS density-map HiPS of Gaia DR3 (``CDS/P/DM/I/355/gaiadr3``, Norder0
    tiles: sources per order-9 pixel, tile pixel (x, y) = nested sub-index with
    x = 511 - row, y = column), summed to order 5, stored as log10(density) quantised
    in 0.015 dex steps from 3.0 (max error 1.7%), and rescaled to the published all-sky
    mean (the HiPS holds 95.5% of the 1.81e9 sources).
    """
    raw = zlib.decompress(base64.b85decode("".join(_GAIA_DR3_DENSITY_ORDER5_B85)))
    q = np.frombuffer(raw, dtype=np.uint8)
    if len(q) != 12 * 4**DENSITY_MAP_ORDER:
        raise ValueError("corrupt Gaia density map")
    density = 10.0 ** (3.0 + 0.015 * q.astype(float))
    return density * (CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2 / float(density.mean()))


def gaia_map_density_deg2(ra_deg: float, dec_deg: float) -> float:
    """Gaia DR3 source density (per deg^2) at (ra, dec) from the order-5 density map."""
    return float(_gaia_density_map()[healpix_nested_index(DENSITY_MAP_ORDER, ra_deg, dec_deg)])


def local_prior_density(catalog: str, ra_deg: float | None = None, dec_deg: float | None = None,
                        sky_density: SkyDensity | None = None) -> tuple[float, str]:
    """Prior mean density (per deg^2) of ``catalog`` around (ra, dec) and where it came from.

    'density_map': the Gaia DR3 map scaled with ``DENSITY_MAP_SCALING``; 'sky_mean': the
    catalogue's all-sky mean (``sky_density`` or ``CATALOG_SKY_DENSITY``); 'generic':
    ``GENERIC_SKY_DENSITY_DEG2`` for a catalogue without a published density.
    """
    sky = sky_density or CATALOG_SKY_DENSITY.get(catalog)
    if sky is None:
        return GENERIC_SKY_DENSITY_DEG2, "generic"
    scaling = DENSITY_MAP_SCALING.get(catalog)
    if scaling is None or ra_deg is None or dec_deg is None or sky_density is not None:
        return sky.per_deg2, "sky_mean"
    alpha, offset = scaling
    ratio = gaia_map_density_deg2(ra_deg, dec_deg) / CATALOG_SKY_DENSITY["gaia_dr3"].per_deg2
    return sky.per_deg2 * 10.0 ** (offset + alpha * math.log10(ratio)), "density_map"


# ---------------------------------------------------------------------------
# Crowded stellar systems the density map cannot resolve
# ---------------------------------------------------------------------------

# Globular clusters (Harris 1996, AJ 112, 1487; VizieR VII/202, 147 clusters: name, J2000
# position in degrees, half-mass radius in arcmin -- None where not listed) and the dense
# parts of Local Group galaxies. Their cores hold 10-100x the density of the 1.8-deg map
# pixel around them (47 Tuc's core: 7e5 Gaia sources per deg^2 within 30" vs a map value of
# 5.8e4), and a cone of a few arcsec there often holds 0-2 rows (Gaia is incomplete in
# crowded cores), so the local density must be measured, not inferred from the cone.
CROWDED_STELLAR_SYSTEMS: tuple[tuple[str, float, float, float | None], ...] = (
    ("NGC 104", 6.0217, -72.0808, 2.79), ("NGC 288", 13.1979, -26.59, 2.22), ("NGC 362", 15.8096, -70.8483, 0.81),
    ("NGC 1261", 48.0637, -55.2169, 0.75), ("Pal 1", 53.3458, 79.5806, 0.48), ("AM 1", 58.7612, -49.6144, 0.5),
    ("Eridanus", 66.1854, -21.1869, 0.4), ("Pal 2", 71.5246, 31.3808, 0.67), ("NGC 1851", 78.5263, -40.0472, 0.52),
    ("NGC 1904", 81.0442, -24.5242, 0.8), ("NGC 2298", 102.2467, -36.0053, 0.78),
    ("NGC 2419", 114.5354, 38.8819, 0.73), ("Pyxis", 136.9908, -37.2214, 1.35),
    ("NGC 2808", 138.0108, -64.8631, 0.76), ("E 3", 140.2471, -77.2825, 2.06), ("Pal 3", 151.3808, 0.0714, 0.66),
    ("NGC 3201", 154.4033, -46.4111, 2.68), ("Pal 4", 172.32, 28.9736, 0.54), ("NGC 4147", 182.5258, 18.5419, 0.43),
    ("NGC 4372", 186.4392, -72.6592, 3.9), ("Rup 106", 189.6675, -51.1503, 1.1),
    ("NGC 4590", 189.8667, -26.7428, 1.55), ("NGC 4833", 194.8958, -70.8747, 2.41),
    ("NGC 5024", 198.2304, 18.1692, 1.11), ("NGC 5053", 199.1125, 17.6981, 3.5),
    ("NGC 5139", 201.6913, -47.4769, 4.18), ("NGC 5272", 205.5467, 28.3756, 1.12),
    ("NGC 5286", 206.6104, -51.3733, 0.69), ("AM 4", 208.9587, -27.1728, 0.42),
    ("NGC 5466", 211.3638, 28.5344, 2.25), ("NGC 5634", 217.4054, -5.9764, 0.54),
    ("NGC 5694", 219.9021, -26.5383, 0.33), ("IC 4499", 225.0771, -82.2136, 1.5),
    ("NGC 5824", 225.9938, -33.0678, 0.36), ("Pal 5", 229.0221, -0.1114, 2.96),
    ("NGC 5897", 229.3521, -21.0103, 2.11), ("NGC 5904", 229.6408, 2.0828, 2.11),
    ("NGC 5927", 232.0021, -50.6728, 1.15), ("NGC 5946", 233.8688, -50.6594, 0.69),
    ("BH 176", 234.7804, -50.0506, None), ("NGC 5986", 236.5146, -37.7861, 1.05),
    ("Lynga 7", 242.7625, -55.3144, None), ("Pal 14", 242.7704, 14.9581, 1.15),
    ("NGC 6093", 244.2604, -22.975, 0.65), ("NGC 6121", 245.8979, -26.5253, 3.65),
    ("NGC 6101", 246.4525, -72.2017, 1.71), ("NGC 6144", 246.8087, -26.0247, 1.62),
    ("NGC 6139", 246.9183, -38.8489, 0.82), ("Terzan 3", 247.1671, -35.3536, 1.3),
    ("NGC 6171", 248.1329, -13.0536, 2.7), ("1636-283", 249.8562, -28.3978, None),
    ("NGC 6205", 250.4229, 36.4603, 1.49), ("NGC 6229", 251.7454, 47.5278, 0.37),
    ("NGC 6218", 251.8104, -1.9478, 2.16), ("NGC 6235", 253.3558, -22.1772, 0.84),
    ("NGC 6254", 254.2871, -4.0994, 1.81), ("NGC 6256", 254.8858, -37.1214, 0.85), ("Pal 15", 255.01, -0.5419, 1.21),
    ("NGC 6266", 255.3025, -30.1122, 1.23), ("NGC 6273", 255.6571, -26.2681, 1.25),
    ("NGC 6284", 256.12, -24.7647, 0.78), ("NGC 6287", 256.2892, -22.7081, 0.75),
    ("NGC 6293", 257.5433, -26.5817, 0.91), ("NGC 6304", 258.6354, -29.4622, 1.41),
    ("NGC 6316", 259.1558, -28.14, 0.71), ("NGC 6341", 259.2804, 43.1364, 1.09),
    ("NGC 6325", 259.4967, -23.7658, 0.94), ("NGC 6333", 259.7992, -18.5164, 0.95),
    ("NGC 6342", 260.2925, -19.5872, 0.88), ("NGC 6356", 260.8958, -17.8131, 0.74),
    ("NGC 6355", 260.9942, -26.3536, 0.87), ("NGC 6352", 261.3717, -48.4228, 2.0),
    ("IC 1257", 261.7854, -7.0931, None), ("Terzan 2", 261.8892, -30.8022, 1.52),
    ("NGC 6366", 261.9346, -5.0767, 2.63), ("Terzan 4", 262.6621, -31.5956, None), ("HP 1", 262.7717, -29.9817, 3.1),
    ("NGC 6362", 262.9783, -67.0481, 2.18), ("Liller 1", 263.3521, -33.3889, 0.45),
    ("NGC 6380", 263.6167, -39.0692, 0.75), ("Terzan 1", 263.9492, -30.4697, 3.82),
    ("Ton 2", 264.0438, -38.5533, 1.08), ("NGC 6388", 264.0708, -44.735, 0.67),
    ("NGC 6402", 264.4004, -3.2458, 1.29), ("NGC 6401", 264.6538, -23.9089, 1.91),
    ("NGC 6397", 265.1721, -53.6736, 2.33), ("Pal 6", 265.9258, -26.2225, 1.06),
    ("NGC 6426", 266.2279, 3.1703, 0.96), ("Djorg 1", 266.8679, -33.0656, 1.26),
    ("Terzan 5", 267.0204, -24.8125, 0.83), ("NGC 6440", 267.2192, -20.3594, 0.58),
    ("NGC 6441", 267.5538, -37.0511, 0.64), ("Terzan 6", 267.6933, -31.2753, 0.44),
    ("NGC 6453", 267.7158, -34.5986, 0.37), ("UKS 1", 268.6133, -24.1453, 0.86),
    ("NGC 6496", 269.7583, -44.265, 1.87), ("Terzan 9", 270.4117, -26.8397, 0.78),
    ("Djorg 2", 270.4546, -27.8258, 0.83), ("NGC 6517", 270.4608, -8.9589, 0.62),
    ("Terzan 10", 270.7392, -26.0667, None), ("NGC 6522", 270.8921, -30.0339, 1.04),
    ("NGC 6535", 270.9613, -0.2969, 0.77), ("NGC 6528", 271.2067, -30.0558, 0.43),
    ("NGC 6539", 271.2075, -7.5858, 1.67), ("NGC 6540", 271.5358, -27.7653, 0.24),
    ("NGC 6544", 271.8358, -24.9975, 1.77), ("NGC 6541", 272.0092, -43.7056, 1.19),
    ("NGC 6553", 272.315, -25.9078, 1.55), ("NGC 6558", 272.5767, -31.7636, 1.61),
    ("IC 1276", 272.6842, -7.2075, 2.35), ("Terzan 12", 273.0658, -22.7419, 0.84),
    ("NGC 6569", 273.4121, -31.8264, 1.33), ("NGC 6584", 274.6571, -52.215, 0.8),
    ("NGC 6624", 275.9187, -30.3611, 0.82), ("NGC 6626", 276.1371, -24.87, 1.56),
    ("NGC 6638", 277.7342, -25.4964, 0.66), ("NGC 6637", 277.8467, -32.3481, 0.83),
    ("NGC 6642", 277.9762, -23.4764, 0.73), ("NGC 6652", 278.9404, -32.9903, 0.65),
    ("NGC 6656", 279.1008, -23.9033, 3.26), ("Pal 8", 280.3746, -19.8258, 0.57),
    ("NGC 6681", 280.8029, -32.2919, 0.93), ("NGC 6712", 283.2679, -8.7061, 1.37),
    ("NGC 6715", 283.7638, -30.4783, 0.49), ("NGC 6717", 283.7758, -22.7008, 0.68),
    ("NGC 6723", 284.8883, -36.6317, 1.61), ("NGC 6749", 286.3137, 1.9008, 1.1),
    ("NGC 6752", 287.7158, -59.9819, 2.34), ("NGC 6760", 287.8004, 1.0306, 2.18),
    ("NGC 6779", 289.1479, 30.1847, 1.16), ("Terzan 7", 289.4321, -34.6575, 0.97),
    ("Pal 10", 289.5088, 18.5717, 0.99), ("Arp 2", 292.1837, -30.3539, 1.91), ("NGC 6809", 294.9975, -30.9622, 2.89),
    ("Terzan 8", 295.4375, -34.0003, 1.0), ("Pal 11", 296.31, -8.0072, 1.49), ("NGC 6838", 298.4421, 18.7783, 1.65),
    ("NGC 6864", 301.52, -21.9214, 0.47), ("NGC 6934", 308.5483, 7.4042, 0.6),
    ("NGC 6981", 313.3663, -12.5369, 0.88), ("NGC 7006", 315.3729, 16.1875, 0.38),
    ("NGC 7078", 322.4929, 12.1669, 1.06), ("NGC 7089", 323.3721, -0.8231, 0.93),
    ("NGC 7099", 325.0917, -23.1792, 1.15), ("Pal 12", 326.6617, -21.2508, 1.28), ("Pal 13", 346.685, 12.7719, 0.46),
    ("NGC 7492", 347.1112, -15.6114, 1.22),
)
# Local Group galaxies (centre, radius in arcmin) whose stellar density varies on scales
# below the map's pixel: the LMC and SMC bars, M31's bulge and inner disc, M33's centre.
CROWDED_GALAXIES: tuple[tuple[str, float, float, float], ...] = (
    ("LMC", 80.8942, -69.7561, 180.0), ("SMC", 13.1867, -72.8286, 90.0), ("M 31", 10.6847, 41.2688, 30.0),
    ("M 33", 23.4621, 30.6602, 15.0),
)
# A target within this many half-mass radii of a globular cluster's centre (at least
# CROWDED_MIN_RADIUS_ARCMIN) lies in its crowded part.
CROWDED_RH_FACTOR = 3.0
CROWDED_MIN_RADIUS_ARCMIN = 1.0


def crowded_region(ra_deg: float, dec_deg: float) -> str | None:
    """Name of the crowded stellar system (CROWDED_STELLAR_SYSTEMS, CROWDED_GALAXIES) the
    position lies in, or None."""
    for name, ra, dec, rh in CROWDED_STELLAR_SYSTEMS:
        if abs(dec - dec_deg) > 1.0:
            continue
        radius = max(CROWDED_RH_FACTOR * (rh if rh is not None else CROWDED_MIN_RADIUS_ARCMIN),
                     CROWDED_MIN_RADIUS_ARCMIN) * 60.0
        if _separation_arcsec(ra, dec, ra_deg, dec_deg) <= radius:
            return name
    for name, ra, dec, radius_arcmin in CROWDED_GALAXIES:
        if _separation_arcsec(ra, dec, ra_deg, dec_deg) <= radius_arcmin * 60.0:
            return name
    return None


def estimate_density_deg2(
    n_rows: int,
    area_deg2: float,
    *,
    catalog: str | None = None,
    sky_density: SkyDensity | None = None,
    n_target_rows: int = 0,
    prior_counts: float = DENSITY_PRIOR_COUNTS,
    position: tuple[float, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Field-source density (per deg^2) of a catalogue around the target.

    Conjugate Gamma-Poisson estimate: prior mean ``m`` (:func:`local_prior_density` at
    ``position``: the density map, else the all-sky mean, else the generic value) with
    ``prior_counts`` pseudo-sources, likelihood = ``n_rows - n_target_rows`` field rows
    over ``area_deg2``: ``nu = (a + n) / (a / m + area)``. Returns (density, provenance).
    """
    field_rows = max(0, int(n_rows) - max(0, int(n_target_rows)))
    area = max(float(area_deg2), 0.0)
    ra, dec = position if position is not None else (None, None)
    mean, source = local_prior_density(catalog or "", ra, dec, sky_density)
    sky = sky_density or (CATALOG_SKY_DENSITY.get(catalog) if catalog else None)
    density = (prior_counts + field_rows) / (prior_counts / mean + area)
    info: dict[str, Any] = {
        "rows": int(n_rows), "field_rows": field_rows, "area_deg2": area,
        "method": "gamma_poisson" if sky is not None else "gamma_poisson_generic",
        "prior_mean_per_deg2": mean, "prior_source": source, "prior_counts": prior_counts,
        "sky_mean_per_deg2": sky.per_deg2 if sky is not None else None,
        "reference": sky.reference if sky is not None else (
            f"no published density: generic prior mean {GENERIC_SKY_DENSITY_DEG2:g} per deg^2"),
        "density_per_deg2": density,
    }
    return density, info


# ---------------------------------------------------------------------------
# Detections: positions at a common epoch and covariances
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Detection:
    """One catalogue row at the common epoch: position (deg) and covariance (arcsec^2).

    ``cov`` is (var_east, cov_east_north, var_north) in the row's own local frame.
    ``target_ra``/``target_dec``/``target_cov`` place the row under the hypothesis "it is
    the target's counterpart" when that differs from where it is as a field object (None:
    the same). ``priority`` then ``rank`` break ties when rows of one catalogue coincide
    (lower first: a star before its planets, the Pan-STARRS duplicate with more
    detections first).
    """

    catalog: str
    ra: float
    dec: float
    cov: tuple[float, float, float]
    priority: int = 0
    label: Any = None
    target_ra: float | None = None
    target_dec: float | None = None
    target_cov: tuple[float, float, float] | None = None
    rank: float = 0.0
    # Counterpart placements under each hypothesis about the target's (unknown) epoch
    # (UNDATED_TARGET_EPOCHS, equal prior); None: one placement for every hypothesis.
    epoch_placements: tuple[tuple[float, float, tuple[float, float, float]], ...] | None = None
    # ln of an extra factor on this row's prior odds of being the target's counterpart
    # (0: none). The identity catalogue's row of a resolved name gets IDENTITY_PRIOR_LN_ODDS.
    prior_ln_odds: float = 0.0
    # Rows of one catalogue with the same non-None ``listing`` key are one object listed
    # several times (the caller knows it: a compilation's two entries of one object, a fast
    # mover detected at several epochs): they are collapsed onto one representative like
    # coincident rows and share its probabilities (see _collapse_coincident).
    listing: Any = None

    def target_placement(self, hypothesis: int = 0) -> tuple[float, float, tuple[float, float, float]]:
        if self.epoch_placements:
            return self.epoch_placements[hypothesis]
        if self.target_ra is None or self.target_dec is None:
            return self.ra, self.dec, self.target_cov or self.cov
        return self.target_ra, self.target_dec, self.target_cov or self.cov


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
    catalogue already publishes deconvolved sizes. ``resolved_column``: a row is resolved
    only when this column has a value (None: every row with a size). ``minor_is_limit``:
    the minor axis counts as unresolved (0) because its limit flag is not fetched.
    """

    major: str
    minor: str
    pa: str | None
    beam_arcsec: float | None
    reference: str
    beam_column: str | None = None
    resolved_column: str | None = None
    minor_is_limit: bool = False


# Resolved radio sources: the host (optical/IR/X-ray counterpart) is not at the centroid of
# an asymmetric structure (one-sided jets, lobes), so the host position relative to the
# catalogued centroid is given an extra Gaussian scatter with the source's own
# (deconvolved) FWHM / 2.3548 along each axis. Without it the NVSS centroid of 3C 273,
# pulled 5.6" towards the jet with a formal 0.53" error, could never be identified.
# Upper limits are never used as sizes: inflating the scatter of an unresolved source
# raises the posterior of chance alignments in a sparse catalogue.
RADIO_STRUCTURE: dict[str, StructureSpec] = {
    # HEASARC nvss: 'The fitted (deconvolved) major axis'; for unresolved sources it is an
    # upper limit (limit_major_axis '<', not fetched). NVSS lists a position angle exactly
    # when the major axis is resolved: 10,571 sources within 8 deg of (200, +40) queried
    # with the flags on 2026-09-28 -- 8,858 of 8,859 upper limits have no position angle,
    # all 1,712 resolved major axes have one. The minor axis is an upper limit for 88%
    # of the resolved sources (1,505 of 1,712), so only the major axis is used.
    "nvss": StructureSpec("major_axis", "minor_axis", "position_angle", None,
                          "Condon et al. 1998, AJ 115, 1693 (deconvolved sizes)",
                          resolved_column="position_angle", minor_is_limit=True),
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
    if spec.resolved_column is not None and _to_float(data.get(spec.resolved_column)) is None:
        return None  # an unresolved source: its size is an upper limit
    major, minor = _sigma_value(data.get(spec.major)), _sigma_value(data.get(spec.minor))
    if major is None:
        return None
    minor = major if minor is None else minor
    beam = _sigma_value(data.get(spec.beam_column)) if spec.beam_column else None
    beam = beam if beam is not None else spec.beam_arcsec
    if beam is not None:
        major = math.sqrt(max(0.0, major * major - beam * beam))
        minor = math.sqrt(max(0.0, minor * minor - beam * beam))
    if spec.minor_is_limit:
        minor = 0.0
    if major <= 0 and minor <= 0:
        return None
    pa = _to_float(data.get(spec.pa)) if spec.pa else None
    return ellipse_covariance(major * FWHM_TO_SIGMA, minor * FWHM_TO_SIGMA, pa)


# Radio / X-ray detections of at least this angular size (FWHM or fitted extent, arcsec) are
# extended emission -- they can be an extended object itself (the intracluster gas of a
# galaxy cluster, a supernova remnant's shell); smaller ones are compact sources, which for
# an extended target are stars, galaxies or nuclei inside it (see emission_extent_arcsec).
EXTENDED_EMISSION_MIN_ARCSEC = 10.0
# X-ray catalogues' extent columns (arcsec; 0 = consistent with the PSF): the XMM pipeline's
# EP_EXTENT (0 for point sources) and the 2RXS source extent.
XRAY_EXTENT_COLUMNS: dict[str, str] = {"xmm": "extent", "rosat": "source_extent", "rosat_bsc": "source_extent"}
# Catalogues of compact detections only: the Chandra Source Catalog lists sources of up to
# ~30" (its extent_flag marks sources wider than the PSF -- also confused point sources in
# crowded fields, e.g. the Trapezium's stars -- without a size), so none of its rows is an
# arcminute-scale object.
COMPACT_SOURCE_CATALOGS = frozenset({"chandra"})


def emission_extent_arcsec(source: CatalogSource) -> float | None:
    """Angular size (arcsec) of a radio / X-ray detection: 0 for a source consistent with a
    point, its fitted / deconvolved size otherwise; None when the catalogue publishes no
    size (optical / infrared and identity catalogues, VLASS, a registered VizieR table).

    X-ray: ``XRAY_EXTENT_COLUMNS``; the Chandra Source Catalog: 0 (compact detections only).
    Radio (``RADIO_STRUCTURE``): the deconvolved major-axis FWHM of a resolved source, 0 for
    an unresolved one (an NVSS size upper limit is not a size)."""
    if source.catalog in COMPACT_SOURCE_CATALOGS:
        return 0.0
    column = XRAY_EXTENT_COLUMNS.get(source.catalog)
    if column is not None:
        value = _to_float((source.data or {}).get(column))
        return max(0.0, value) if value is not None and math.isfinite(value) else None
    if source.catalog in RADIO_STRUCTURE:
        cov = structure_covariance(source)
        if cov is None:
            return 0.0
        half_trace = (cov[0] + cov[2]) / 2.0
        spread = math.sqrt(((cov[0] - cov[2]) / 2.0) ** 2 + cov[1] ** 2)
        return math.sqrt(max(half_trace + spread, 0.0)) / FWHM_TO_SIGMA
    return None


def _sigma_value(value: Any) -> float | None:
    """A usable 1-sigma (arcsec): finite and positive, clamped to MAX_SIGMA_ARCSEC; else None."""
    number = _to_float(value)
    if number is None or not math.isfinite(number) or number <= 0:
        return None
    return min(number, MAX_SIGMA_ARCSEC)


def source_covariance(source: CatalogSource) -> tuple[tuple[float, float, float], str]:
    """Positional covariance of a catalogue row (arcsec^2) and how it was obtained.

    The catalogue's error ellipse (``metadata['positional_error']['ellipse_1sigma_arcsec']``)
    or per-axis errors (with Gaia's ``ra_dec_corr``) give the shape; any systematic term,
    calibration scale, floor or epoch-growth term that ``positional_error_arcsec`` (the RMS
    1-sigma circular error) contains beyond them is added isotropically, so the covariance
    always has ``trace / 2 == positional_error_arcsec^2``.
    """
    sigma = _sigma_value(source.positional_error_arcsec)
    details = (source.metadata or {}).get("positional_error") or {}
    if not isinstance(details, Mapping):
        details = {}
    cov: tuple[float, float, float] | None = None
    shape = "circular"
    ellipse = details.get("ellipse_1sigma_arcsec")
    major = _sigma_value(ellipse.get("major")) if isinstance(ellipse, Mapping) else None
    if major is not None:
        minor = _sigma_value(ellipse.get("minor"))  # type: ignore[union-attr]
        minor = major if minor is None else minor
        pa = _to_float(ellipse.get("pa_deg"))  # type: ignore[union-attr]
        pa = pa if pa is not None and math.isfinite(pa) else None
        cov = ellipse_covariance(major, minor, pa)
        shape = "ellipse" if pa is not None else "ellipse_no_pa"
    else:
        s_ra, s_dec = _sigma_value(details.get("sigma_ra_arcsec")), _sigma_value(details.get("sigma_dec_arcsec"))
        if s_ra is not None and s_dec is not None:
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


# Calibration of catalogue position errors against Gaia-CRF3 quasars (the core scale of
# the two-component fit in the module docstring): AllWISE sigra/sigdec x 1.10 (1929 clean
# pairs, 20 mas floor included). The catalogue's own errors are reported unchanged; only
# the association uses the calibrated covariance.
POSITION_ERROR_CALIBRATION: dict[str, float] = {"allwise": 1.10}
# Saturated sources: their formal errors (a few mas for the brightest stars) do not include
# the error of the PSF-wing fit. Measured on 3,154 Hipparcos stars (Hp < 7; W1 < 3) and on
# 46,187 isolated Gaia DR3 stars (G < 14; W1 < 9.5) against AllWISE, moved with their
# proper motions to the row's w1mjdmean: the per-axis scatter beyond the formal error is
# 0.11-0.25" for W1 < 6 (median 0.2"; 99th percentile of the offsets 0.4-2.4") and 0.04" for
# 6 <= W1 < 7; fainter sources match their formal errors (+ the floor). The column
# (fetched with every row) and the (limit, extra sigma) steps, brightest first: the
# heavy-tail mixture (kappa = 5) then covers offsets of ~1-2" (Antares 0.82", Betelgeuse
# 1.44", Mira 1.89" from their Hipparcos positions).
# 2MASS: its error ellipse already grows to 0.29" for saturated stars, which covers the
# measured scatter down to K = -1 (10,808 Hipparcos stars, moved to the row's jdate: per-axis
# 0.12-0.17"); the brightest (K < -1, e.g. Arcturus) scatter much more (90th percentile of the
# offsets 1.0", 99th 7.7"; 61 stars): 0.4" is added.
SATURATION_SYSTEMATIC_ARCSEC: dict[str, tuple[str, tuple[tuple[float, float], ...]]] = {
    "allwise": ("w1mpro", ((6.0, 0.22), (7.0, 0.05))),
    "twomass_psc": ("k_m", ((-1.0, 0.4),)),
}


def saturation_sigma_arcsec(source: CatalogSource) -> float | None:
    """Extra per-axis sigma (arcsec) of a saturated row (SATURATION_SYSTEMATIC_ARCSEC), or None."""
    spec = SATURATION_SYSTEMATIC_ARCSEC.get(source.catalog)
    if spec is None:
        return None
    mag = _to_float((source.data or {}).get(spec[0]))
    if mag is None or not math.isfinite(mag):
        return None
    for limit, extra in spec[1]:
        if mag < limit:
            return extra
    return None


def detection_covariance(source: CatalogSource) -> tuple[tuple[float, float, float], str]:
    """Covariance used for association: :func:`source_covariance` scaled by
    ``POSITION_ERROR_CALIBRATION``, plus the astrometric floor (``ASTROMETRIC_FLOOR_ARCSEC``
    in quadrature), the saturation term of bright sources (:func:`saturation_sigma_arcsec`)
    and, for resolved radio sources, the host-offset term of :func:`structure_covariance`."""
    cov, shape = source_covariance(source)
    scale = POSITION_ERROR_CALIBRATION.get(source.catalog)
    if scale is not None:
        cov = (cov[0] * scale * scale, cov[1] * scale * scale, cov[2] * scale * scale)
        shape += f"x{scale:g}"
    floor = ASTROMETRIC_FLOOR_ARCSEC**2
    cov = (cov[0] + floor, cov[1], cov[2] + floor)
    saturated = saturation_sigma_arcsec(source)
    if saturated:
        cov = (cov[0] + saturated * saturated, cov[1], cov[2] + saturated * saturated)
        shape += "+saturation"
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


def _grow(cov: tuple[float, float, float], variance: float) -> tuple[float, float, float]:
    return (cov[0] + variance, cov[1], cov[2] + variance) if variance else cov


def pm_sigma_masyr(source: CatalogSource) -> float:
    """Per-axis proper-motion uncertainty (mas/yr) of a row with its own motion.

    Published pm errors when the row carries them (``pmra_error``/``pmdec_error``).
    Otherwise ``PM_ERROR_PER_POSITION_ERROR`` x the position error at the row's
    measurement epoch -- never the error already grown by a catalogue's own epoch
    propagation: SIMBAD moves Gaia positions to J2000.0 and its positional error then
    holds pm_error x 16 yr (``models._epoch_growth``), whose pm error is that term / 16 yr.
    """
    data = source.data or {}
    errs = [_to_float(data.get(k)) for k in ("pmra_error", "pmdec_error")]
    errs = [e for e in errs if e is not None and e >= 0]
    if errs:
        return math.sqrt(sum(e * e for e in errs) / len(errs))
    details = (source.metadata or {}).get("positional_error") or {}
    growth = details.get("epoch_growth") if isinstance(details, Mapping) else None
    if isinstance(growth, Mapping):
        term = _to_float(growth.get("term_arcsec"))
        to_epoch, from_epoch = _to_float(growth.get("to_epoch")), _to_float(growth.get("source_epoch"))
        if term is not None and to_epoch is not None and from_epoch is not None and to_epoch != from_epoch:
            return term / abs(to_epoch - from_epoch) * 1000.0
    base = _to_float(details.get("statistical_sigma_arcsec")) if isinstance(details, Mapping) else None
    if isinstance(growth, Mapping) and _to_float(growth.get("term_arcsec")) and base is not None:
        base = math.sqrt(max(base * base - float(growth["term_arcsec"]) ** 2, 0.0))
    if base is None or base <= 0:
        base = source.positional_error_arcsec
    return PM_ERROR_PER_POSITION_ERROR * (base if base is not None else MISSING_SIGMA_ARCSEC) * 1000.0


def _row_span(source: CatalogSource) -> tuple[float, float]:
    """Epoch span of a row: its epoch, else its epoch range, else UNKNOWN_EPOCH_RANGE."""
    if source.epoch is not None:
        return float(source.epoch), float(source.epoch)
    if source.epoch_range is not None:
        lo, hi = source.epoch_range
        return float(min(lo, hi)), float(max(lo, hi))
    return UNKNOWN_EPOCH_RANGE


def _epoch_gap(source: CatalogSource, epoch: float) -> float:
    """Largest difference between the row's (possibly unknown) epoch and ``epoch``: never
    0 for a row whose epoch is unknown (its span, else UNKNOWN_EPOCH_RANGE, is used)."""
    lo, hi = _row_span(source)
    return max(abs(lo - epoch), abs(hi - epoch))


def _window_gap(source: CatalogSource, window: tuple[float, float]) -> float:
    """Largest epoch difference between the row and any epoch in ``window``."""
    return max(_epoch_gap(source, window[0]), _epoch_gap(source, window[1]))


def _along_track(cov: tuple[float, float, float], pm: tuple[float, float], years: float) -> tuple[float, float, float]:
    """Add the variance of a uniform position along a track of |pm| x years (L^2 / 12)."""
    speed = math.hypot(*pm)
    if speed <= 0 or years <= 0:
        return cov
    ue, un = pm[0] / speed, pm[1] / speed
    var = (speed * years / 1000.0) ** 2 / 12.0
    return cov[0] + var * ue * ue, cov[1] + var * ue * un, cov[2] + var * un * un


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
    rank: float = 0.0,
) -> tuple[Detection, dict[str, Any]]:
    """A catalogue row brought to the target's epoch, with its grown covariance.

    The row gets two placements (see the module docstring): as the target's counterpart
    (``Detection.target_*``; what ``info`` describes) and as a field object
    (``Detection.ra/dec/cov``; ``info['field_propagation']``).

    * Target with an epoch: :func:`models.source_position_at` places the counterpart --
      the row's own proper motion ("source_pm", growth ``pm_sigma_masyr(row) dt``); the
      target's motion and parallax for rows of catalogues without motions (growth
      ``target_pm_sigma_masyr dt``, plus ``extra_sigma_arcsec``, e.g. an unremoved
      parallax); a pm-less row of a proper-motion catalogue moves with the target's
      motion too when it is known ("target_pm") -- as a field object it stays where it was
      measured ("stationary"). A row whose epoch is unknown within a span
      ("target_pm_span": XMM, CSC, SIMBAD rows without a motion) is placed with the target's
      motion from the middle of the span, with the variance of a uniform epoch along the
      track (``_along_track``): its likelihood is marginalised over the span. Field
      placements of rows without their own motion are the measured positions with growth
      ``UNKNOWN_PM_SIGMA_MASYR dt``.
    * Target without an epoch (``target`` None or ``target.epoch`` None): positions are
      compared as given, the target epoch being one of ``UNDATED_TARGET_EPOCHS``: as the
      counterpart, a row with its own motion gets one placement per hypothesis
      (``Detection.epoch_placements``: at that epoch, with ``UNDATED_EPOCH_SPREAD_YR`` of
      motion along its track; "source_pm_undated"); a row without its own motion moves
      with the target's motion when that is known (adopted from the target's identity row;
      "target_pm_undated", from the middle of its epoch span, the span as along-track
      spread), else stays put with ``UNKNOWN_PM_SIGMA_MASYR`` x the largest epoch
      difference. As field objects, rows
      are compared at ``reference_epoch`` (J2016.0): own motion, else
      ``UNKNOWN_PM_SIGMA_MASYR`` growth.
    * ``extragalactic`` rows (a galaxy / QSO type or a redshift) do not move: a catalogue
      proper motion of such a row is measurement noise (e.g. SIMBAD lists Gaia's for M87).

    """
    base, shape = detection_covariance(source)
    info: dict[str, Any] = {"covariance_shape": shape, "extra_sigma_arcsec": extra_sigma_arcsec}
    dated = target is not None and target.epoch is not None
    if extragalactic:
        epoch = float(target.epoch) if dated else None  # type: ignore[union-attr]
        cov = _grow(base, (extra_sigma_arcsec or 0.0) ** 2)
        info.update({"epoch": epoch, "propagation": "extragalactic", "pm_growth_arcsec": 0.0,
                     "field_propagation": "extragalactic", "sigma_arcsec": math.sqrt((cov[0] + cov[2]) / 2.0)})
        return Detection(source.catalog, source.ra, source.dec, cov, priority, label, rank=rank), info
    own_pm = source.proper_motion_ra_masyr is not None and source.proper_motion_dec_masyr is not None
    if not dated:
        window = (min(UNDATED_TARGET_EPOCHS), max(UNDATED_TARGET_EPOCHS))
        extra = (extra_sigma_arcsec or 0.0) ** 2
        # Field placement: every row at the common epoch reference_epoch (J2016.0), where
        # rows are compared with each other.
        pm = ((float(source.proper_motion_ra_masyr), float(source.proper_motion_dec_masyr))  # type: ignore[arg-type]
              if own_pm and source.epoch is not None else None)
        if pm is not None:
            f_ra, f_dec = propagate_radec(source.ra, source.dec, pm[0], pm[1], float(source.epoch), reference_epoch)
            f_growth = pm_sigma_masyr(source) * _epoch_gap(source, reference_epoch) / 1000.0
            f_method = "source_pm"
        else:
            f_ra, f_dec, f_method = source.ra, source.dec, "none"
            f_growth = UNKNOWN_PM_SIGMA_MASYR * _epoch_gap(source, reference_epoch) / 1000.0
        field_cov = _grow(base, f_growth * f_growth)
        placements: tuple[tuple[float, float, tuple[float, float, float]], ...] | None = None
        if pm is not None and math.hypot(*pm) / 1000.0 * (window[1] - window[0]) > UNDATED_MOTION_MIN_ARCSEC:
            # Counterpart placements: at each hypothesised target epoch.
            out = []
            growths = []
            for epoch_h in UNDATED_TARGET_EPOCHS:
                ra_h, dec_h = propagate_radec(source.ra, source.dec, pm[0], pm[1], float(source.epoch), epoch_h)
                g = pm_sigma_masyr(source) * _epoch_gap(source, epoch_h) / 1000.0
                cov_h = _grow(_along_track(_grow(base, g * g), pm, math.sqrt(12.0) * UNDATED_EPOCH_SPREAD_YR), extra)
                out.append((ra_h, dec_h, cov_h))
                growths.append(g)
            placements = tuple(out)
            ra, dec, cov = placements[0]
            growth = max(growths)
            method = "source_pm_undated"
        elif pm is not None:
            ra, dec = propagate_radec(source.ra, source.dec, pm[0], pm[1], float(source.epoch), 0.5 * sum(window))
            growth = pm_sigma_masyr(source) * _window_gap(source, window) / 1000.0
            cov = _grow(base, growth * growth + extra)
            method = "source_pm"
        elif (t_pm := target.proper_motion if target is not None else None) is not None and \
                math.hypot(*t_pm) / 1000.0 * (window[1] - window[0]) > UNDATED_MOTION_MIN_ARCSEC:
            # A row without its own motion is the (undated) target only if it moves with the
            # target's known motion: one placement per hypothesised target epoch, moved from
            # the row's epoch (its span's middle, with the span as along-track spread).
            lo, hi = _row_span(source)
            out = []
            growths = []
            for epoch_h in UNDATED_TARGET_EPOCHS:
                ra_h, dec_h = propagate_radec(source.ra, source.dec, t_pm[0], t_pm[1], 0.5 * (lo + hi), epoch_h)
                g = target_pm_sigma_masyr * max(abs(lo - epoch_h), abs(hi - epoch_h)) / 1000.0
                spread = math.hypot(hi - lo, math.sqrt(12.0) * UNDATED_EPOCH_SPREAD_YR)
                out.append((ra_h, dec_h, _grow(_along_track(_grow(base, g * g), t_pm, spread), extra)))
                growths.append(g)
            placements = tuple(out)
            ra, dec, cov = placements[0]
            growth = max(growths)
            method = "target_pm_undated"
        else:
            ra, dec = source.ra, source.dec
            growth = UNKNOWN_PM_SIGMA_MASYR * _window_gap(source, window) / 1000.0
            cov = _grow(base, growth * growth + extra)
            method = "none"
        info.update({"epoch": None, "epoch_window": list(window), "propagation": method, "pm_growth_arcsec": growth,
                     "field_propagation": f_method, "sigma_arcsec": math.sqrt((cov[0] + cov[2]) / 2.0)})
        if placements is not None:
            info["epoch_hypotheses"] = list(UNDATED_TARGET_EPOCHS)
        same = (f_ra, f_dec) == (ra, dec) and field_cov == cov and placements is None
        det = Detection(source.catalog, f_ra, f_dec, field_cov, priority, label,
                        target_ra=None if same else ra, target_dec=None if same else dec,
                        target_cov=None if same else cov, rank=rank, epoch_placements=placements)
        return det, info

    assert target is not None and target.epoch is not None
    epoch = float(target.epoch)
    pm_t = target.proper_motion
    ra, dec, method = source_position_at(source, epoch, pm_t, (target.ra, target.dec), target.parallax_mas)
    dt = _epoch_gap(source, epoch)
    unknown = UNKNOWN_PM_SIGMA_MASYR * dt / 1000.0
    field_ra, field_dec, field_method = ra, dec, method
    field_cov = _grow(base, unknown * unknown)
    if method == "source_pm":
        growth = pm_sigma_masyr(source) * dt / 1000.0
        field_cov = _grow(base, growth * growth)
    elif method == "stationary" and pm_t is not None and source.epoch is not None:
        # A pm-less row of a proper-motion catalogue: the target only if it moves with it.
        ra, dec = propagate_radec(source.ra, source.dec, pm_t[0], pm_t[1], float(source.epoch), epoch)
        method = "target_pm"
        growth = target_pm_sigma_masyr * dt / 1000.0
    elif method in TARGET_PM_METHODS:
        growth = target_pm_sigma_masyr * dt / 1000.0
        field_ra, field_dec, field_method = source.ra, source.dec, "none"
    else:  # "stationary" (target motion unknown) / "none": the row's motion is unknown
        growth = unknown
    cov = _grow(base, growth * growth + (extra_sigma_arcsec or 0.0) ** 2)
    if method == "target_pm_span" and pm_t is not None:
        # The row's epoch is unknown within its span (uniform): the target's position then is
        # anywhere on its track during the span. The likelihood is marginalised over the
        # epoch -- placed from the middle of the span with the track's L^2 / 12 variance
        # along the motion (as for undated targets) -- not maximised at the closest approach
        # (models.closest_span_epoch, which only decides whether the row is in the cone): a
        # field source near a fast mover's track is not a perfect match.
        lo, hi = _row_span(source)
        ra, dec = propagate_radec(source.ra, source.dec, pm_t[0], pm_t[1], 0.5 * (lo + hi), epoch)
        cov = _along_track(cov, pm_t, hi - lo)
        info["span_years"] = hi - lo
    info.update({"epoch": epoch, "propagation": method, "pm_growth_arcsec": growth,
                 "field_propagation": field_method, "sigma_arcsec": math.sqrt((cov[0] + cov[2]) / 2.0)})
    same = (field_ra, field_dec) == (ra, dec) and field_cov == cov
    det = Detection(source.catalog, field_ra, field_dec, field_cov, priority, label,
                    target_ra=None if same else ra, target_dec=None if same else dec,
                    target_cov=None if same else cov, rank=rank)
    return det, info


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AssociationConfig:
    """Tunable parameters of :func:`associate` (defaults documented in the module)."""

    target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC
    # Per-axis (east, north) target sigmas when they differ (coordinates rounded in RA
    # seconds of time: 12h29m07s is 4.3" east but 0.3" north for a Dec rounded to 1");
    # None: target_sigma_arcsec on both axes.
    target_sigma_axes: tuple[float, float] | None = None
    # Scalar (every catalogue) or {catalog: c}: prior probability of a target counterpart.
    completeness: float | Mapping[str, float] = DEFAULT_PRIOR_COMPLETENESS
    link_chi2: float = LINK_CHI2
    max_states: int = MAX_STATES
    prune_weight: float = PRUNE_WEIGHT
    max_candidates_per_catalog: int = MAX_CANDIDATES_PER_CATALOG
    secondary_ratio: float = SECONDARY_RATIO
    coincident_arcsec: float = COINCIDENT_ARCSEC
    outlier_fraction: float = OUTLIER_FRACTION
    outlier_scale: float = OUTLIER_SCALE
    # Weigh field objects seen by several catalogues in the null (see module docstring),
    # with an object density of object_density_factor x the densest catalogue's.
    correlated_fields: bool = True
    object_density_factor: float = OBJECT_DENSITY_FACTOR
    target_group_min_p_any: float = TARGET_GROUP_MIN_P_ANY
    duplicate_resolution_fraction: float = DUPLICATE_RESOLUTION_FRACTION
    resolution_arcsec: Mapping[str, float] = field(default_factory=lambda: dict(CATALOG_RESOLUTION_ARCSEC))

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
            "target_sigma_axes": list(self.target_sigma_axes) if self.target_sigma_axes else None,
            "completeness": dict(self.completeness) if isinstance(self.completeness, Mapping) else self.completeness,
            "link_chi2": self.link_chi2, "max_states": self.max_states, "prune_weight": self.prune_weight,
            "max_candidates_per_catalog": self.max_candidates_per_catalog,
            "secondary_ratio": self.secondary_ratio, "coincident_arcsec": self.coincident_arcsec,
            "outlier_fraction": self.outlier_fraction, "outlier_scale": self.outlier_scale,
            "correlated_fields": self.correlated_fields, "object_density_factor": self.object_density_factor,
            "target_group_min_p_any": self.target_group_min_p_any,
            "duplicate_resolution_fraction": self.duplicate_resolution_fraction,
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


def gnomonic_jacobian(ra0: float, dec0: float, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, ...]:
    """Jacobian (j11, j12, j21, j22) of the gnomonic projection about (ra0, dec0) at (ra, dec):
    local (east, north) offsets at the point -> tangent-plane (xi, eta) offsets.

    It is the identity at the origin and rotates by ~(ra - ra0) sin(dec) away from it (the
    convergence of meridians), so it matters near the poles.
    """
    s0, c0 = math.sin(math.radians(dec0)), math.cos(math.radians(dec0))
    d = np.radians(np.asarray(dec, float))
    a = np.radians(np.asarray(ra, float) - ra0)
    sd, cd, sa, ca = np.sin(d), np.cos(d), np.sin(a), np.cos(a)
    D = s0 * sd + c0 * cd * ca
    D = np.where(np.abs(D) < 1e-12, 1e-12, D)
    N = c0 * sd - s0 * cd * ca
    dD_dd = s0 * cd - c0 * sd * ca
    dN_dd = c0 * cd + s0 * sd * ca
    D2 = D * D
    j11 = (ca * D + c0 * cd * sa * sa) / D2
    j21 = (s0 * sa * D + N * c0 * sa) / D2
    j12 = -sa * (sd * D + cd * dD_dd) / D2
    j22 = (dN_dd * D - N * dD_dd) / D2
    return j11, j12, j21, j22


class _Arrays:
    """Vectorised per-detection quantities in the tangent plane about ``origin``.

    ``role`` 'field' uses ``Detection.ra/dec/cov``, 'target' the counterpart placement.
    Covariances are mapped from each detection's local frame with the gnomonic Jacobian.
    """

    def __init__(self, detections: Sequence[Detection], origin: tuple[float, float], role: str = "field",
                 hypothesis: int = 0) -> None:
        n = len(detections)
        self.n = n
        if role == "target":
            placed = [d.target_placement(hypothesis) for d in detections]
            ra = [p[0] for p in placed]
            dec = [p[1] for p in placed]
            covs = [p[2] for p in placed]
        else:
            ra = [d.ra for d in detections]
            dec = [d.dec for d in detections]
            covs = [d.cov for d in detections]
        self.ra = np.asarray(ra, dtype=float).reshape(n)
        self.dec = np.asarray(dec, dtype=float).reshape(n)
        cov = np.array(covs, dtype=float).reshape(n, 3)
        vxx, vxy, vyy = cov[:, 0], cov[:, 1], cov[:, 2]
        if n and not np.all(vxx * vyy - vxy**2 > 0):
            raise ValueError("every detection needs a positive-definite covariance")
        j11, j12, j21, j22 = gnomonic_jacobian(origin[0], origin[1], self.ra, self.dec)
        # C' = J C J^T
        self.vxx = j11 * j11 * vxx + 2.0 * j11 * j12 * vxy + j12 * j12 * vyy
        self.vxy = j11 * j21 * vxx + (j11 * j22 + j12 * j21) * vxy + j12 * j22 * vyy
        self.vyy = j21 * j21 * vxx + 2.0 * j21 * j22 * vxy + j22 * j22 * vyy
        det = self.vxx * self.vyy - self.vxy**2
        det = np.where(det > 0, det, vxx * vyy - vxy**2)
        self.wa, self.wb, self.wc = self.vyy / det, -self.vxy / det, self.vxx / det
        self.ld = -np.log(det)
        self.x, self.y = _gnomonic_arcsec(origin[0], origin[1], self.ra, self.dec)
        self.unit = _unit_vectors(self.ra, self.dec)
        tr, dd = (self.vxx + self.vyy) / 2.0, np.sqrt(((self.vxx - self.vyy) / 2.0) ** 2 + self.vxy**2)
        self.sigma_major = np.sqrt(tr + dd)
        # Python lists for the small-cluster arithmetic of the field partition.
        self.lx, self.ly = self.x.tolist(), self.y.tolist()
        self.lvxx, self.lvxy, self.lvyy = self.vxx.tolist(), self.vxy.tolist(), self.vyy.tolist()
        self.lwa, self.lwb, self.lwc, self.lld = self.wa.tolist(), self.wb.tolist(), self.wc.tolist(), self.ld.tolist()


def _chord(arcsec: float) -> float:
    return 2.0 * math.sin(min(math.pi, arcsec / ARCSEC_PER_RAD) / 2.0)


def _pair_chi2(arr: _Arrays, i: int, j: int) -> float:
    dx, dy = arr.x[j] - arr.x[i], arr.y[j] - arr.y[i]
    sxx, sxy, syy = arr.vxx[i] + arr.vxx[j], arr.vxy[i] + arr.vxy[j], arr.vyy[i] + arr.vyy[j]
    det = sxx * syy - sxy * sxy
    return float((syy * dx * dx - 2.0 * sxy * dx * dy + sxx * dy * dy) / det)


def placement_chi2_matrix(detections: Sequence[Detection], origin: tuple[float, float], *, role: str = "field",
                          hypothesis: int = 0) -> np.ndarray:
    """Pairwise Mahalanobis chi2 (2 d.o.f.) between the placements of ``detections`` -- their
    field placements (``role`` 'field') or their counterpart placements under an epoch
    ``hypothesis`` ('target') -- in the tangent plane about ``origin``."""
    arr = _Arrays(detections, origin, role, hypothesis)
    dx = arr.x[None, :] - arr.x[:, None]
    dy = arr.y[None, :] - arr.y[:, None]
    sxx = arr.vxx[:, None] + arr.vxx[None, :]
    sxy = arr.vxy[:, None] + arr.vxy[None, :]
    syy = arr.vyy[:, None] + arr.vyy[None, :]
    det = sxx * syy - sxy * sxy
    return (syy * dx * dx - 2.0 * sxy * dx * dy + sxx * dy * dy) / det


def target_chi2(detections: Sequence[Detection], target: tuple[float, float], target_sigma_arcsec: float, *,
                hypothesis: int = 0) -> np.ndarray:
    """Mahalanobis chi2 (2 d.o.f.) of each detection's counterpart placement (under an epoch
    ``hypothesis``) from the ``target`` position with its per-axis sigma."""
    arr = _Arrays(detections, target, "target", hypothesis)
    var = float(target_sigma_arcsec) ** 2
    sxx, sxy, syy = arr.vxx + var, arr.vxy, arr.vyy + var
    det = sxx * syy - sxy * sxy
    return (syy * arr.x * arr.x - 2.0 * sxy * arr.x * arr.y + sxx * arr.y * arr.y) / det


def _collapse_coincident(arr: _Arrays, detections: Sequence[Detection], cat_index: np.ndarray,
                         catalogs: Sequence[str], cfg: AssociationConfig) -> tuple[np.ndarray, dict[int, int]]:
    """Representative index of every detection: rows of one catalogue closer than
    ``cfg.coincident_arcsec``, or closer than ``cfg.duplicate_resolution_fraction`` of the
    catalogue's resolution and consistent within their errors, are one measurement; so are
    rows of one catalogue that share a ``Detection.listing`` key (one object listed several
    times, as the caller established)."""
    rep = np.arange(arr.n)
    duplicates: dict[int, int] = {}
    if arr.n < 2:
        return rep, duplicates
    for k in np.unique(cat_index):
        idx = np.flatnonzero(cat_index == k)
        if len(idx) < 2:
            continue
        parent = {int(i): int(i) for i in idx}

        def union(i: int, j: int, parent: dict[int, int] = parent) -> None:
            ra_, rb_ = _find(parent, i), _find(parent, j)
            if ra_ != rb_:
                parent[rb_] = ra_

        by_listing: dict[Any, int] = {}
        for i in idx.tolist():
            key = detections[i].listing
            if key is not None:
                if key in by_listing:
                    union(by_listing[key], int(i))
                else:
                    by_listing[key] = int(i)
        resolution = cfg.resolution_arcsec.get(catalogs[int(k)]) if cfg.resolution_arcsec else None
        dup_radius = cfg.duplicate_resolution_fraction * float(resolution) if resolution else 0.0
        radius = max(cfg.coincident_arcsec, dup_radius)
        pairs = cKDTree(arr.unit[idx]).query_pairs(_chord(radius), output_type="ndarray") if radius > 0 else []
        for a, b in pairs:
            i, j = int(idx[a]), int(idx[b])
            sep = 2.0 * math.asin(min(1.0, float(np.linalg.norm(arr.unit[i] - arr.unit[j])) / 2.0)) * ARCSEC_PER_RAD
            if sep > cfg.coincident_arcsec and _pair_chi2(arr, i, j) > DUPLICATE_MAX_CHI2:
                continue
            union(i, j)
        clusters: dict[int, list[int]] = {}
        for i in idx:
            clusters.setdefault(_find(parent, int(i)), []).append(int(i))
        for members in clusters.values():
            if len(members) < 2:
                continue
            # A row identified as the target (the resolved name's row, an extended identity at the
            # target: raised prior odds) represents its duplicates, never the reverse: collapsed
            # onto a nearby detection it would lose its identity prior (SIMBAD's 'M 82', 0.5",
            # merged into the radio source 'EQ J095552.5+694045.4' 1.6" away got P = 0).
            best = min(members, key=lambda i: (detections[i].prior_ln_odds <= 0.0, detections[i].priority,
                                               detections[i].rank, arr.vxx[i] + arr.vyy[i], i))
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


@dataclass(slots=True)
class _Virtual:
    """Candidate hypotheses of the target enumeration: one detection as a counterpart
    with its core or its tail (inflated) covariance, in the target placement."""

    det: np.ndarray
    x: np.ndarray
    y: np.ndarray
    wa: np.ndarray
    wb: np.ndarray
    wc: np.ndarray
    ld: np.ndarray
    prior: np.ndarray


def _enumerate_target(
    virt: _Virtual, slots: list[list[int]], target_w: tuple[float, float, float, float], max_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """All (or the most probable ``max_states``) associations of the target.

    ``slots[k]`` are the virtual candidates of catalogue slot k (hypotheses of its rows);
    ``virt.prior`` their ln prior factors (ln mixture weight + ln c/(1-c) - ln N). Returns
    (choice matrix [state, slot] -> virtual candidate or -1, ln weight, ln Bayes factor, exact).
    """
    ta, tb, tc, tld = target_w
    sa, sb, sc = np.array([ta]), np.array([tb]), np.array([tc])
    bx, by, q = np.zeros(1), np.zeros(1), np.zeros(1)
    ld, n, prior = np.array([tld]), np.ones(1), np.zeros(1)
    choice = np.full((1, len(slots)), -1, dtype=np.int64)
    exact = True
    const = LN2 + 2.0 * LN_ARCSEC_PER_RAD

    def ln_bf(sa: np.ndarray, sb: np.ndarray, sc: np.ndarray, bx: np.ndarray, by: np.ndarray, q: np.ndarray,
              ld: np.ndarray, n: np.ndarray) -> np.ndarray:
        det = sa * sc - sb * sb
        quad = (sc * bx * bx - 2.0 * sb * bx * by + sa * by * by) / det
        chi2 = np.maximum(q - quad, 0.0)
        return (n - 1.0) * const + 0.5 * ld - 0.5 * np.log(det) - 0.5 * chi2

    for k, members in enumerate(slots):
        if not members:
            continue
        m = np.asarray(members)
        wa, wb, wc = virt.wa[m], virt.wb[m], virt.wc[m]
        xx, yy = virt.x[m], virt.y[m]
        wx, wy = wa * xx + wb * yy, wb * xx + wc * yy
        qq = xx * wx + yy * wy
        # Broadcast: existing states (rows) x (none + candidates) columns.
        sa = np.column_stack((sa, sa[:, None] + wa[None, :])).ravel()
        sb = np.column_stack((sb, sb[:, None] + wb[None, :])).ravel()
        sc = np.column_stack((sc, sc[:, None] + wc[None, :])).ravel()
        bx = np.column_stack((bx, bx[:, None] + wx[None, :])).ravel()
        by = np.column_stack((by, by[:, None] + wy[None, :])).ravel()
        q = np.column_stack((q, q[:, None] + qq[None, :])).ravel()
        ld = np.column_stack((ld, ld[:, None] + virt.ld[m][None, :])).ravel()
        n = np.column_stack((n, n[:, None] + np.ones(len(m))[None, :])).ravel()
        prior = np.column_stack((prior, prior[:, None] + virt.prior[m][None, :])).ravel()
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


class _FieldPartitionFunction:
    """ln L(Y) for subsets Y of the candidate rows and their linked neighbours: the
    partition function of Y as field objects (see the module docstring), singletons
    weighing 1.

    ``ln_weight(members)`` is ln of the weight of ``members`` (rows of distinct
    catalogues) forming one field object, relative to separate rows. L factorises over the
    connected components of the link graph restricted to Y. A component is summed exactly
    (every set of linked rows with distinct catalogues as an object) when it has at most
    ``FIELD_EXACT_MAX_ROWS`` rows and at most ``FIELD_OBJECT_BUDGET`` possible objects;
    otherwise over the objects of its own greedy (most probable merges first) partition,
    each exact when small enough. Components are those of Y itself, so every association
    is scored with the pairings its left-over rows can actually make.
    """

    def __init__(self, arr: _Arrays, nodes: Sequence[int], cat_index: np.ndarray,
                 ln_weight: Any, link_chi2: float, near_chi2: float | None = None) -> None:
        self.nodes = [int(i) for i in nodes]
        self.n = len(self.nodes)
        self.ln_weight = ln_weight
        self.cat = [int(cat_index[i]) for i in self.nodes]
        self.cat_bits = [1 << c for c in self.cat]
        # adj: rows that can be one object (link_chi2, at the outlier scale); near: rows
        # whose core errors overlap (near_chi2), used to localise re-partitioning.
        self.adj = [0] * self.n
        self.near = [0] * self.n
        near_chi2 = link_chi2 if near_chi2 is None else near_chi2
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if self.cat[a] == self.cat[b]:
                    continue
                chi2 = _pair_chi2(arr, self.nodes[a], self.nodes[b])
                if chi2 <= link_chi2:
                    self.adj[a] |= 1 << b
                    self.adj[b] |= 1 << a
                if chi2 <= near_chi2:
                    self.near[a] |= 1 << b
                    self.near[b] |= 1 << a
        self.memo: dict[int, float] = {}
        self.comp_memo: dict[int, float] = {}
        self.lnw_memo: dict[int, float] = {}
        self.bits_memo: dict[int, list[int]] = {}
        # Blocks of the full set: an exact-summable component is one block; a larger one
        # is split into its greedy objects, which are re-partitioned locally when an
        # association removes some of their rows (see ln_l).
        self.full = (1 << self.n) - 1
        self.exact_blocks: list[int] = []
        self.greedy_blocks: list[int] = []
        for comp in self._components(self.full):
            if comp & (comp - 1) == 0:
                continue
            if self._exact_ok(comp):
                self.exact_blocks.append(comp)
            else:
                self.greedy_blocks.extend(b for b in self._greedy(comp))
        self.block_value = {b: self._block_value(b) for b in self.greedy_blocks}
        self.block_near: dict[int, int] = {}
        for b in self.greedy_blocks:
            reach = 0
            for p in self._bits(b):
                reach |= self.near[p]
            self.block_near[b] = b | sum(o for o in self.greedy_blocks if o & reach and o != b)

    def _exact_ok(self, comp: int) -> bool:
        return comp.bit_count() <= FIELD_EXACT_MAX_ROWS and self._object_count(comp) <= FIELD_OBJECT_BUDGET

    def _block_value(self, block: int) -> float:
        if block & (block - 1) == 0:
            return 0.0
        if self._exact_ok(block):
            return self._exact(block)
        return float(np.logaddexp(0.0, self._lnw(block))) if self._lnw(block) > -math.inf else 0.0

    # -- helpers ----------------------------------------------------------------------
    def _bits(self, mask: int) -> list[int]:
        cached = self.bits_memo.get(mask)
        if cached is not None:
            return cached
        out = []
        m = mask
        while m:
            low = m & -m
            out.append(low.bit_length() - 1)
            m ^= low
        if len(self.bits_memo) < 200_000:
            self.bits_memo[mask] = out
        return out

    def _lnw(self, mask: int) -> float:
        """ln weight of the rows of ``mask`` as one object (-inf if two share a catalogue)."""
        value = self.lnw_memo.get(mask)
        if value is None:
            cats = 0
            members: list[int] = []
            for p in self._bits(mask):
                if cats & self.cat_bits[p]:
                    self.lnw_memo[mask] = -math.inf
                    return -math.inf
                cats |= self.cat_bits[p]
                members.append(self.nodes[p])
            value = float(self.ln_weight(members))
            self.lnw_memo[mask] = value
        return value

    def _components(self, mask: int) -> list[int]:
        out = []
        remaining = mask
        while remaining:
            start = remaining & -remaining
            seen = start
            frontier = start
            while frontier:
                p = frontier.bit_length() - 1
                frontier &= ~(1 << p)
                new = self.adj[p] & remaining & ~seen
                seen |= new
                frontier |= new
            out.append(seen)
            remaining &= ~seen
        return out

    def _connected(self, mask: int) -> bool:
        start = mask & -mask
        seen = frontier = start
        while frontier:
            p = frontier.bit_length() - 1
            frontier &= ~(1 << p)
            new = self.adj[p] & mask & ~seen
            seen |= new
            frontier |= new
        return seen == mask

    def _object_count(self, mask: int) -> int:
        counts: dict[int, int] = {}
        for p in self._bits(mask):
            counts[self.cat[p]] = counts.get(self.cat[p], 0) + 1
        total = 1
        for c in counts.values():
            total *= 1 + c
        return total

    # -- exact sum -----------------------------------------------------------------------
    def _exact(self, mask: int) -> float:
        if mask & (mask - 1) == 0:
            return 0.0
        value = self.memo.get(mask)
        if value is not None:
            return value
        comps = self._components(mask)
        if len(comps) > 1:
            value = sum(self._exact(c) for c in comps)
            self.memo[mask] = value
            return value
        low = mask & -mask
        rest = mask & ~low
        low_cat = self.cat[low.bit_length() - 1]
        groups: dict[int, list[int]] = {}
        for p in self._bits(rest):
            if self.cat[p] != low_cat:
                groups.setdefault(self.cat[p], []).append(1 << p)
        terms = [self._exact(rest)]  # the lowest row alone
        for choice in itertools.product(*([0, *g] for g in groups.values())):
            sub = sum(choice)
            if not sub:
                continue
            obj = sub | low
            if not self._connected(obj):
                continue
            lnw = self._lnw(obj)
            if lnw > -math.inf:
                terms.append(lnw + self._exact(rest & ~sub))
        value = _logsumexp(terms)
        self.memo[mask] = value
        return value

    # -- approximation for large components ------------------------------------------------
    def _greedy(self, mask: int, units: Sequence[int] | None = None) -> list[int]:
        """Greedy agglomeration of the rows of ``mask`` (most probable merge first),
        starting from single rows or from the given ``units`` (disjoint row masks)."""
        start = list(units) if units is not None else [1 << p for p in self._bits(mask)]
        groups: dict[int, int] = {}
        reach: dict[int, int] = {}
        cats: dict[int, int] = {}
        lw: dict[int, float] = {}
        owner: dict[int, int] = {}
        for unit in start:
            key = (unit & -unit).bit_length() - 1
            groups[key] = unit
            r = 0
            c = 0
            for p in self._bits(unit):
                r |= self.adj[p]
                c |= self.cat_bits[p]
                owner[p] = key
            reach[key] = r & mask & ~unit
            cats[key] = c
            lw[key] = self._lnw(unit) if unit & (unit - 1) else 0.0
        heap: list[tuple[float, int, int]] = []
        for a, unit in groups.items():
            for b in {owner[p] for p in self._bits(reach[a]) if p in owner}:
                if b > a and not cats[a] & cats[b]:
                    gain = self._lnw(unit | groups[b]) - lw[a] - lw[b]
                    if gain > 0:
                        heap.append((-gain, a, b))
        heapq.heapify(heap)
        while heap:
            neg, a, b = heapq.heappop(heap)
            if a not in groups or b not in groups or cats[a] & cats[b]:
                continue
            merged = groups[a] | groups[b]
            value = self._lnw(merged)
            gain = value - lw[a] - lw[b]
            if gain <= 0:
                continue
            if abs(gain + neg) > 1e-9:  # stale entry: re-queue with the current gain
                heapq.heappush(heap, (-gain, a, b))
                continue
            groups[a] = merged
            lw[a] = value
            reach[a] = (reach[a] | reach[b]) & ~merged
            cats[a] |= cats[b]
            for p in self._bits(groups[b]):
                owner[p] = a
            del groups[b], lw[b], reach[b], cats[b]
            neighbours = {owner[p] for p in self._bits(reach[a]) if p in owner}
            for other in neighbours:
                if other != a and not cats[a] & cats[other]:
                    g = self._lnw(merged | groups[other]) - lw[a] - lw[other]
                    if g > 0:
                        heapq.heappush(heap, (-g, min(a, other), max(a, other)))
        return list(groups.values())

    def _partition_value(self, groups: list[int]) -> float:
        """ln L around a greedy partition: each group summed exactly (when small), plus the
        first-order alternatives of the rows the greedy left alone -- joining any linked
        group (or a later lone row) instead: a low-precision row with many possible partners
        (a 4" X-ray position among dense optical stars) contributes all of them, not one."""
        value = sum(self._block_value(g) for g in groups)
        singles = [g for g in groups if g & (g - 1) == 0]
        if not singles:
            return value
        lw = {g: (self._lnw(g) if g & (g - 1) else 0.0) for g in groups}
        for r in singles:
            p = r.bit_length() - 1
            reach = self.adj[p]
            terms = [0.0]
            for g in groups:
                if g == r or not g & reach or (g & (g - 1) == 0 and g < r):
                    continue
                joined = self._lnw(g | r)
                if joined > -math.inf:
                    terms.append(joined - lw[g])
            if len(terms) > 1:
                value += _logsumexp(terms)
        return value

    def _component(self, comp: int) -> float:
        """ln L of one connected set of rows: exact when small, else greedy objects."""
        value = self.comp_memo.get(comp)
        if value is not None:
            return value
        if self._exact_ok(comp):
            value = self._exact(comp)
        else:
            value = self._partition_value(self._greedy(comp))
        self.comp_memo[comp] = value
        return value

    def _local(self, mask: int) -> float:
        return sum(self._component(c) for c in self._components(mask) if c & (c - 1))

    def block_of(self) -> dict[int, int]:
        """Row bit position -> the block (exact component or greedy object mask) holding it."""
        out: dict[int, int] = {}
        for block in self.exact_blocks + self.greedy_blocks:
            for p in self._bits(block):
                out[p] = block
        return out

    def block_part_value(self, block: int, part: int) -> float:
        """ln L of the rows ``part`` left of one block, without re-pairing them with other
        blocks: exact components summed exactly, greedy objects as their own sub-object."""
        if part & (part - 1) == 0:
            return 0.0
        if block in self.block_value and part == block:
            return self.block_value[block]
        return self._exact(part) if self._exact_ok(part) else self._block_value(part)

    def ln_l(self, mask: int) -> float:
        """ln L of the rows in ``mask`` (bit p = ``nodes[p]``).

        Exact blocks are summed exactly on their remaining rows. Greedy objects that lost
        rows are re-partitioned together with the objects linked to them (so a left-over
        row can pair with a neighbour's row); untouched objects keep their value.
        """
        removed = self.full & ~mask
        total = 0.0
        for block in self.exact_blocks:
            part = block & mask
            if part & (part - 1):
                total += self._exact(part)
        if not self.greedy_blocks:
            return total
        touched = [b for b in self.greedy_blocks if b & removed]
        if not touched:
            return total + sum(self.block_value.values())
        local = 0
        for block in touched:
            local |= self.block_near[block]
        for block in self.greedy_blocks:
            if not block & local:
                total += self.block_value[block]
        key = local & mask
        value = self.memo.get(-key - 1)
        if value is None:
            # Units: the left-over rows of the touched objects, and the linked objects whole.
            units: list[int] = []
            touched_mask = 0
            for block in touched:
                touched_mask |= block
                units.extend(1 << p for p in self._bits(block & mask))
            units.extend(b for b in self.greedy_blocks if b & local and not b & touched_mask)
            value = self._partition_value(self._greedy(key, units))
            self.memo[-key - 1] = value
        return total + value


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a = _unit_vectors(np.array([ra1, ra2]), np.array([dec1, dec2]))
    return 2.0 * math.asin(min(1.0, float(np.linalg.norm(a[0] - a[1])) / 2.0)) * ARCSEC_PER_RAD


def _observed_fractions(arr: _Arrays, typical: np.ndarray, cone: tuple[float, float, float] | None) -> np.ndarray:
    """``f[i, l]``: the fraction of row i's partner region in catalogue l (whose typical
    1-sigma is ``typical[l]``) that lies inside the searched ``cone`` (1 without a cone).

    A partner of row i in catalogue l lies around it with the combined core sigma of the
    row and catalogue l's typical row; the fraction inside the cone of radius R is
    P(|x| <= R) for x ~ N(position of i, s^2 I): a non-central chi-square with 2 d.o.f.
    Rows well inside the cone see all of it (1); a row at the edge about half; a row whose
    partner region is much larger than the cone (a ROSAT position in a 5" cone) little.
    """
    n = arr.n
    frac = np.ones((n, len(typical)))
    if cone is None or n == 0 or not len(typical):
        return frac
    ra0, dec0, radius = float(cone[0]), float(cone[1]), float(cone[2])
    centre = _unit_vectors(np.array([ra0]), np.array([dec0]))[0]
    chord = np.linalg.norm(arr.unit - centre[None, :], axis=1)
    offset = 2.0 * np.arcsin(np.minimum(1.0, chord / 2.0)) * ARCSEC_PER_RAD
    s = np.hypot(arr.sigma_major[:, None], np.asarray(typical, dtype=float)[None, :])
    s = np.maximum(s, MIN_SIGMA_ARCSEC)
    from scipy.stats import ncx2

    frac = ncx2.cdf((radius / s) ** 2, 2, (offset[:, None] / s) ** 2)
    return np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0)


def _moved(det: Detection) -> bool:
    """True when a counterpart placement of ``det`` lies more than MOVED_ROW_ARCSEC from
    its field placement (placed with a motion far from where it was measured)."""
    placements = list(det.epoch_placements or ())
    if det.target_ra is not None and det.target_dec is not None:
        placements.append((det.target_ra, det.target_dec, det.cov))
    return any(_separation_arcsec(det.ra, det.dec, ra, dec) > MOVED_ROW_ARCSEC for ra, dec, _cov in placements)


def _linked_closure(seeds: Sequence[int], links: np.ndarray, limit: int) -> list[int]:
    """Rows connected to ``seeds`` through ``links`` (breadth first, nearest hops first),
    at most ``limit`` of them, excluding the seeds."""
    adjacency: dict[int, list[int]] = {}
    for i, j in links.tolist():
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    seen = {int(s) for s in seeds}
    frontier = list(seen)
    out: list[int] = []
    while frontier and len(out) < limit:
        nxt: list[int] = []
        for i in frontier:
            for j in adjacency.get(i, []):
                if j not in seen:
                    seen.add(j)
                    out.append(j)
                    nxt.append(j)
                    if len(out) >= limit:
                        return sorted(out)
        frontier = nxt
    return sorted(out)


def _corrected_weights(fp: _FieldPartitionFunction, nodes: Sequence[int], det_choice: np.ndarray,
                       logw: np.ndarray, max_exact: int = MAX_CORRECTED_STATES) -> tuple[np.ndarray, int]:
    """``logw`` + ln L(nodes minus a) for every association a, and the number of associations
    of non-negligible weight whose correction was estimated rather than computed.

    Associations are grouped by the set of field rows they remove (their correction depends
    on nothing else). Every removed set first gets its *block-local* value: each block of
    the partition function (an exactly summed component, or an object of the greedy
    partition of a large component) summed on its remaining rows, without letting them
    re-pair with other blocks -- exact when every component is summed exactly (the usual
    case), else a slight underestimate. Where greedy objects exist, the removed sets are
    then taken in decreasing order of estimated weight and, at most ``max_exact`` of them,
    get the full value (:meth:`_FieldPartitionFunction.ln_l`, which re-partitions the rows
    left around the removed ones); a set more than ``CORRECTION_NEGLIGIBLE_LN`` below the
    best even at its upper bound (L grows with the rows: ``ln L(Y - R) <= ln L(Y)``) keeps
    its estimate, which cannot matter. Crowded 15-catalogue cones reach the 20,000-state
    beam; their most probable associations still get exact corrections (see
    ``test_budgeted_field_corrections_match_the_exact_ones``).
    """
    logw = np.asarray(logw, dtype=float)
    n_states = len(logw)
    full = (1 << len(nodes)) - 1
    ln_all = fp.ln_l(full)
    if n_states == 0:
        return logw.copy(), 0
    pos = {int(node): p for p, node in enumerate(nodes)}
    top_det = int(det_choice.max(initial=-1))
    column = np.full(top_det + 2, -1, dtype=np.int64)
    cand: list[int] = []
    for node in sorted({int(d) for d in np.unique(det_choice).tolist() if d >= 0}):
        if node in pos:
            column[node] = len(cand)
            cand.append(node)
    if not cand:
        return logw + ln_all, 0
    cols = np.where(det_choice >= 0, column[np.maximum(det_choice, 0)], -1)
    incidence = np.zeros((n_states, len(cand)), dtype=bool)
    rows, slots = np.nonzero(cols >= 0)
    incidence[rows, cols[rows, slots]] = True
    patterns, inverse = np.unique(incidence, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_pat = len(patterns)
    bits = [1 << pos[node] for node in cand]
    block_of = fp.block_of()
    pattern_logw = np.full(n_pat, -np.inf)
    np.maximum.at(pattern_logw, inverse, logw)
    # Block-local values: ln L(Y) minus what each touched block loses.
    base = {block: fp.block_part_value(block, block) for block in set(block_of.values())}
    estimate = np.empty(n_pat)
    masks: list[int] = []
    for u in range(n_pat):
        mask = full
        lost: dict[int, int] = {}
        for c in np.flatnonzero(patterns[u]).tolist():
            mask &= ~bits[c]
            block = block_of.get(pos[cand[c]])
            if block is not None:
                lost[block] = lost.get(block, block) & ~bits[c]
        masks.append(mask)
        estimate[u] = ln_all + sum(fp.block_part_value(block, part) - base[block] for block, part in lost.items())
    value = estimate.copy()
    estimated = 0
    if fp.greedy_blocks:
        best = -np.inf
        n_exact = 0
        for u in np.argsort(-(pattern_logw + estimate)).tolist():
            if pattern_logw[u] + ln_all < best - CORRECTION_NEGLIGIBLE_LN:
                continue  # negligible even at the upper bound: the estimate is kept
            if n_exact >= max_exact:
                estimated += 1
                continue
            value[u] = fp.ln_l(masks[u])
            n_exact += 1
            best = max(best, pattern_logw[u] + value[u])
    return logw + value[inverse], estimated


def _origin(detections: Sequence[Detection]) -> tuple[float, float]:
    """Tangent-plane origin of a target-less association: the normalised mean of the unit
    vectors (an arithmetic mean of RA fails across RA = 0)."""
    if not detections:
        return 0.0, 0.0
    ra = np.radians([d.ra for d in detections])
    dec = np.radians([d.dec for d in detections])
    v = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))).sum(axis=0)
    norm = float(np.linalg.norm(v))
    if norm <= 0:
        return float(detections[0].ra), float(detections[0].dec)
    v /= norm
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))


def associate(
    detections: Sequence[Detection],
    densities_deg2: Mapping[str, float],
    *,
    target: tuple[float, float] | None = None,
    config: AssociationConfig | None = None,
    cone: tuple[float, float, float] | None = None,
) -> AssociationResult:
    """Probabilistic N-way association of ``detections`` (see the module docstring).

    ``densities_deg2`` gives each catalogue's field-source density per deg^2 (see
    :func:`estimate_density_deg2`). ``target`` is the (ra, dec) of the search target at
    the detections' common epoch; its 1-sigma per-axis uncertainty is
    ``config.target_sigma_axes`` (east, north) when given, else ``config.target_sigma_arcsec``.
    Without a target only the object partition is made.
    ``cone`` (ra, dec, radius_arcsec) is the searched cone when rows beyond it were not
    fetched: a row near its edge stays in the correlated-field partition function (its
    in-cone partners pair with it), but its lone-row factor counts only the part of each
    other catalogue's partner region that lies inside the cone (``_observed_fractions``),
    since a partner beyond the edge was not fetched.
    """
    cfg = config or AssociationConfig()
    n = len(detections)
    notes: list[str] = []
    origin = _origin(detections) if target is None else (float(target[0]), float(target[1]))
    arr = _Arrays(detections, origin, "field")
    catalogs = sorted({d.catalog for d in detections})
    cat_id = {c: i for i, c in enumerate(catalogs)}
    cat_index = np.fromiter((cat_id[d.catalog] for d in detections), int, n)
    missing = [c for c in catalogs if c not in densities_deg2 or not densities_deg2[c] > 0]
    if missing:
        raise ValueError(f"no positive source density for catalogue(s) {missing}")
    ln_n = {c: math.log(float(densities_deg2[c]) * FULL_SKY_DEG2) for c in catalogs}
    ln_n_det = np.fromiter((ln_n[d.catalog] for d in detections), float, n)

    rep, duplicates = _collapse_coincident(arr, detections, cat_index, catalogs, cfg)
    reps = np.flatnonzero(rep == np.arange(n))
    # Per-row prior odds factors (identity rows); a duplicate listing passes its factor to
    # its representative, which stands for both.
    row_odds = np.fromiter((float(d.prior_ln_odds or 0.0) for d in detections), float, n)
    for dup, r in duplicates.items():
        row_odds[r] = max(row_odds[r], row_odds[dup])
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
    target_norm: tuple[Any, ...] | None = None

    if target is not None:
        st = max(float(cfg.target_sigma_arcsec), MIN_SIGMA_ARCSEC)
        se, sn = (max(float(v), MIN_SIGMA_ARCSEC) for v in cfg.target_sigma_axes) if cfg.target_sigma_axes else (st, st)
        target_w = (1.0 / (se * se), 0.0, 1.0 / (sn * sn), -math.log(se * se * sn * sn))
        ln_odds = {c: math.log(cfg.completeness_of(c)) - math.log1p(-cfg.completeness_of(c)) for c in catalogs}
        eps = min(max(float(cfg.outlier_fraction), 0.0), 0.5)
        kappa2 = max(float(cfg.outlier_scale), 1.0) ** 2
        ln_core = math.log1p(-eps)
        ln_tail = math.log(eps) if eps > 0 else -math.inf
        prior_det = np.fromiter((ln_odds[d.catalog] for d in detections), float, n) - ln_n_det + row_odds
        field_weight = None
        lone = np.zeros(n)
        moved: set[int] = set()
        nu = {c: float(v) for c, v in densities_deg2.items() if v and float(v) > 0}
        if cfg.correlated_fields and n and nu:
            # Field objects of density nu_obj, each seen by catalogue k with probability q_k
            # (module docstring): a row alone has density nu_k Q / (1 - q_k) -- its prior
            # 1 / nu_k gets the lone-row factor (1 - q_k) / Q -- and rows of one object weigh
            # B / (N_obj Q)^(n - 1) relative to separate rows.
            nu_obj = max(float(cfg.object_density_factor), 1.0) * max(nu.values())
            q = {c: min(MAX_DETECTION_FRACTION, v / nu_obj) for c, v in nu.items()}
            ln_q_all = sum(math.log1p(-v) for v in q.values())
            # Rows placed far from where they were measured are left out of the field model
            # (no lone-row factor). Near the cone's edge a row's partners may lie outside,
            # unobserved: its lone-row factor counts only the observed part of each other
            # catalogue's partner region (see _observed_fractions).
            moved = {i for i, d in enumerate(detections) if _moved(d)}
            # Every catalogue with a density (queried), with or without rows here.
            columns = sorted(q)
            typical = []
            for c in columns:
                rows_c = cat_index == cat_id[c] if c in cat_id else np.zeros(n, dtype=bool)
                typical.append(float(np.median(arr.sigma_major[rows_c])) if rows_c.any() else 0.0)
            frac = _observed_fractions(arr, np.asarray(typical), cone)
            ln_miss_all = np.log1p(-np.array([q[c] for c in columns])[None, :] * frac)
            own = np.array([columns.index(d.catalog) for d in detections], dtype=int)
            ln_miss = ln_miss_all.copy()
            ln_miss[np.arange(n), own] = 0.0
            lone = np.where([i in moved for i in range(n)], 0.0, -ln_miss.sum(axis=1)) if n else lone
            prior_det = prior_det + lone
            ln_scale = math.log(nu_obj * FULL_SKY_DEG2) + ln_q_all
            ln_n_obj = math.log(nu_obj * FULL_SKY_DEG2)
            fully_observed = bool(np.all(frac >= 1.0 - 1e-12))
            miss_rows = ln_miss_all.tolist()
            lone_rows = lone.tolist()
            own_rows = own.tolist()
            n_col = len(columns)

            def field_weight(members: list[int]) -> float:
                if fully_observed:
                    return _ln_mixture_bf(arr, members, eps, kappa2) - (len(members) - 1) * ln_scale
                # Near the cone edge: an object seen by the member catalogues and by none of the
                # others *where observed* (sum of ln(1 - q_l f_l), the members' mean), relative to
                # its rows as separate lone rows (their lone-row factors); with every f = 1 this
                # is exactly B / (N_obj Q)^(|o| - 1).
                seen = {own_rows[i] for i in members}
                unseen = 0.0
                for c in range(n_col):
                    if c not in seen:
                        unseen += sum(miss_rows[i][c] for i in members) / len(members)
                return (_ln_mixture_bf(arr, members, eps, kappa2) - (len(members) - 1) * ln_n_obj
                        + sum(lone_rows[i] for i in members) + unseen)

        # One enumeration per hypothesis about the target's epoch (undated targets whose
        # rows move: J2000 or J2016, equal prior), summed.
        counts = {len(d.epoch_placements) for d in detections if d.epoch_placements}
        if len(counts) > 1:
            raise ValueError("every detection with epoch placements needs the same number of them")
        n_hyp = counts.pop() if counts else 1
        ln_hyp = -math.log(n_hyp)
        col_of = {c: k for k, c in enumerate(catalogs)}
        tarrs: list[_Arrays] = []
        run_choice: list[np.ndarray] = []
        run_logw: list[np.ndarray] = []
        run_lnbf: list[np.ndarray] = []
        run_w2: list[np.ndarray] = []
        run_enum: list[set[int]] = []
        run_cats: list[set[str]] = []
        cand_all: set[int] = set()
        for h in range(n_hyp):
            tarr = _Arrays(detections, origin, "target", h)
            tarrs.append(tarr)

            def two_way(scale: float, tarr: _Arrays = tarr) -> np.ndarray:
                sxx, sxy, syy = tarr.vxx * scale + se * se, tarr.vxy * scale, tarr.vyy * scale + sn * sn
                det2 = sxx * syy - sxy**2
                chi2 = (syy * tarr.x**2 - 2.0 * sxy * tarr.x * tarr.y + sxx * tarr.y**2) / det2
                return LN2 + 2.0 * LN_ARCSEC_PER_RAD - 0.5 * np.log(det2) - 0.5 * chi2

            ln_w2_core = two_way(1.0) + prior_det + ln_core
            ln_w2_tail = two_way(kappa2) + prior_det + ln_tail if eps > 0 else np.full(n, -np.inf)
            ln_w2 = np.logaddexp(ln_w2_core, ln_w2_tail)
            keep = reps[ln_w2[reps] >= math.log(cfg.prune_weight)]
            per_cat: list[tuple[str, list[int]]] = []
            for c in catalogs:
                members = [int(i) for i in keep[cat_index[keep] == cat_id[c]]]
                if not members:
                    continue
                members.sort(key=lambda i: -ln_w2[i])
                if len(members) > cfg.max_candidates_per_catalog:
                    note = (f"{c}: {len(members)} target candidates, the {cfg.max_candidates_per_catalog} most "
                            "probable were enumerated")
                    if note not in notes:
                        notes.append(note)
                    members = members[:cfg.max_candidates_per_catalog]
                per_cat.append((c, members))
            per_cat.sort(key=lambda item: -max(ln_w2[i] for i in item[1]))
            # Virtual candidates: core and/or tail hypothesis of each row.
            v_det: list[int] = []
            v_tail: list[bool] = []
            slots: list[list[int]] = []
            ln_ratio = math.log(OUTLIER_PRUNE_RATIO)
            for _c, members in per_cat:
                slot: list[int] = []
                for i in members:
                    if ln_w2_core[i] >= ln_w2_tail[i] + ln_ratio:
                        slot.append(len(v_det))
                        v_det.append(i)
                        v_tail.append(False)
                    if ln_w2_tail[i] >= ln_w2_core[i] + ln_ratio:
                        slot.append(len(v_det))
                        v_det.append(i)
                        v_tail.append(True)
                slots.append(slot)
            vd = np.asarray(v_det, dtype=int)
            vt = np.asarray(v_tail, dtype=bool)
            scale = np.where(vt, kappa2, 1.0) if len(vd) else np.ones(0)
            virt = _Virtual(
                det=vd, x=tarr.x[vd], y=tarr.y[vd], wa=tarr.wa[vd] / scale, wb=tarr.wb[vd] / scale,
                wc=tarr.wc[vd] / scale, ld=tarr.ld[vd] - 2.0 * np.log(scale),
                prior=prior_det[vd] + np.where(vt, ln_tail, ln_core))
            choice, logw_h, lnbf_h, exact_h = _enumerate_target(virt, slots, target_w, cfg.max_states)
            exact = exact and exact_h
            # Detection chosen per catalogue (columns: every catalogue, in catalogs order).
            det_h = np.full((len(logw_h), len(catalogs)), -1, dtype=np.int64)
            for k, (c, _m) in enumerate(per_cat):
                col = choice[:, k]
                det_h[:, col_of[c]] = np.where(col >= 0, vd[np.maximum(col, 0)] if len(vd) else -1, -1)
            run_choice.append(det_h)
            run_logw.append(logw_h + ln_hyp)
            run_lnbf.append(lnbf_h)
            run_w2.append(ln_w2)
            run_enum.append({int(i) for i in vd.tolist()})
            run_cats.append({c for c, _m in per_cat})
            cand_all.update(int(i) for i in vd.tolist())
        det_choice = np.concatenate(run_choice)
        logw = np.concatenate(run_logw)
        lnbf = np.concatenate(run_lnbf)
        run_of = np.concatenate([np.full(len(x), h) for h, x in enumerate(run_logw)])
        n_states = len(logw)
        if not exact:
            notes.append(f"beam search: at most {cfg.max_states} most probable associations kept per step "
                         "(not exhaustive)")
        if n_hyp > 1:
            notes.append(f"undated target: its epoch is marginalised over {list(UNDATED_TARGET_EPOCHS)} "
                         "(equal prior) for the rows that move")
        # Correlated fields: rows left out of an association may form field objects, with
        # each other or with their (non-candidate) neighbours. Rows placed with the
        # target's motion far from where they were measured (fast movers: 2MASS, AllWISE,
        # PS1 or NED rows of Barnard's star) are left out: several rows of one catalogue
        # then lie along the target's past track (PS1 splits a fast mover, NED lists it
        # twice) and would pair with each other.
        fp: _FieldPartitionFunction | None = None
        # Rows near the cone edge stay in the partition function: the in-cone rows of their
        # object pair with them (a star seen by four catalogues near the edge is one field
        # object, not four independent coincidences).
        cand_nodes = sorted(i for i in cand_all if i not in moved)
        # Rows left out of the enumeration whose two-way weight still matters: their first-
        # order extensions (below) also take them out of the field.
        ln_ext_min = math.log(EXTENSION_MIN_WEIGHT)
        ext_rows = sorted({int(j) for h in range(n_hyp) for j in reps.tolist()
                           if int(j) not in run_enum[h] and int(j) not in moved and run_w2[h][j] >= ln_ext_min},
                          key=lambda j: -max(float(run_w2[h][j]) for h in range(n_hyp)))[:FIELD_MAX_EXTENSION_ROWS]
        field_nodes: list[int] = []
        if field_weight is not None and len(cand_nodes) + len(ext_rows) >= 1:
            # Outlier (tail) positions link rows kappa times farther apart.
            field_link_chi2 = cfg.link_chi2 * (kappa2 if eps > 0 else 1.0)
            # Neighbours of the candidates: rows linked to them within their core errors
            # (their likely partners as field objects), nearest hops first.
            core_links = _links(arr, reps, cat_index, cfg.link_chi2)
            keep_links = np.asarray([pair for pair in core_links.tolist() if pair[0] not in moved and pair[1] not in moved],
                                    dtype=int).reshape(-1, 2)
            seeds = cand_nodes + [j for j in ext_rows if j not in set(cand_nodes)]
            field_nodes = seeds + _linked_closure(seeds, keep_links, FIELD_MAX_NEIGHBOURS)
            if len(field_nodes) >= 2:
                fp = _FieldPartitionFunction(arr, field_nodes, cat_index, field_weight, field_link_chi2, cfg.link_chi2)
                logw, estimated = _corrected_weights(fp, field_nodes, det_choice, logw)
                if estimated:
                    notes.append(f"field corrections of {estimated} less probable association(s) estimated block by "
                                 f"block (exact for the {MAX_CORRECTED_STATES} most probable)")
        top = float(logw.max())
        w = np.exp(logw - top)
        is_null = (det_choice < 0).all(axis=1)
        # Rows left out of the enumeration (pruned, or beyond max_candidates_per_catalog):
        # to first order each extends the associations with no row from its catalogue by
        # its two-way weight, times its field correction in the most probable association
        # (the field objects it would leave) -- counted in the normalisation and in the
        # marginals, so dropping them cannot make the enumerated rows overconfident.
        ext_corr: dict[int, float] = {}
        if fp is not None and ext_rows:
            pos = {node: p for p, node in enumerate(fp.nodes)}
            best_state = int(np.argmax(logw))
            base_mask = (1 << len(fp.nodes)) - 1
            for d in det_choice[best_state]:
                if d >= 0 and int(d) in pos:
                    base_mask &= ~(1 << pos[int(d)])
            base_value = fp.ln_l(base_mask)
            for j in ext_rows:
                if j in pos and base_mask >> pos[j] & 1:
                    ext_corr[j] = fp.ln_l(base_mask & ~(1 << pos[j])) - base_value
        empty = det_choice < 0
        ext = np.zeros(len(w))
        extra_num = np.zeros(n)
        for h in range(n_hyp):
            in_run = run_of == h
            others = [int(j) for j in reps.tolist() if int(j) not in run_enum[h]]
            if not others:
                continue
            e_col = np.zeros(len(catalogs))
            # A left-out row outside the modelled neighbourhood is an independent field row
            # (no lone-row factor, no correction).
            in_fp = set(fp.nodes) if fp is not None else set(field_nodes)
            w_j = {j: math.exp(min(float(run_w2[h][j]) + ext_corr.get(j, 0.0) - (0.0 if j in in_fp else float(lone[j])),
                                   700.0)) for j in others}
            for j in others:
                e_col[cat_index[j]] += w_j[j]
            ext[in_run] = w[in_run] * (empty[in_run] * e_col[None, :]).sum(axis=1)
            empty_weight = (w[in_run][:, None] * empty[in_run]).sum(axis=0)
            for j in others:
                extra_num[j] += w_j[j] * empty_weight[cat_index[j]]
        wx = w + ext
        total = float(wx.sum())
        w_null = float(w[is_null].sum())
        nonnull_total = total - w_null
        p_any = 1.0 - w_null / total
        # Marginal probability of each row: the associations containing it (all of their
        # core / tail and epoch hypotheses, and their extensions), plus its own extensions.
        num = extra_num.copy()
        for k in range(det_choice.shape[1]):
            col = det_choice[:, k]
            mask = col >= 0
            if mask.any():
                np.add.at(num, col[mask], wx[mask])
        target_prob[reps] = np.minimum(1.0, num[reps] / total)
        for dup, r in duplicates.items():
            target_prob[dup] = target_prob[r]
        if nonnull_total > 0 and p_any >= cfg.target_group_min_p_any:
            # An association is a set of rows: its weight sums the core / tail and epoch
            # hypotheses.
            nonnull = np.flatnonzero(~is_null)
            keys, key_of = np.unique(det_choice[nonnull], axis=0, return_inverse=True)
            key_of = np.asarray(key_of).reshape(-1)
            key_weight = np.bincount(key_of, weights=w[nonnull], minlength=len(keys))
            # The most probable state (core / tail, epoch hypothesis) of each association.
            by_key = np.lexsort((-logw[nonnull], key_of))
            first = np.ones(len(by_key), dtype=bool)
            first[1:] = key_of[by_key][1:] != key_of[by_key][:-1]
            key_state = np.empty(len(keys), dtype=np.int64)
            key_state[key_of[by_key][first]] = nonnull[by_key][first]
            order_keys = np.argsort(-key_weight, kind="stable")
            cut = cfg.secondary_ratio * float(key_weight[order_keys[0]])
            shown = order_keys[: int(np.count_nonzero(key_weight >= cut)) + 1]
            ranked = [(tuple(int(i) for i in keys[k]), [float(key_weight[k]), int(key_state[k])]) for k in shown.tolist()]
            best_key, (best_weight, best) = ranked[0]
            p_best = best_weight / nonnull_total
            best_reps = [i for i in best_key if i >= 0]
            # Joint posterior that every member is a counterpart (other catalogues free).
            joint = np.ones(len(w), dtype=bool)
            for k, d in enumerate(best_key):
                if d >= 0:
                    joint &= det_choice[:, k] == d
            in_target = set(best_reps)
            members = []
            for r in sorted(best_reps, key=lambda i: (cat_index[i], i)):
                members.append(r)
                members.extend(sorted(followers.get(r, [])))
            alternatives = []
            for key, (weight, s_idx) in ranked[1:]:
                p_i = weight / nonnull_total
                if p_i < cfg.secondary_ratio * p_best:
                    break
                alt = sorted(i for i in key if i >= 0)
                state_members.append((alt, p_i, float(lnbf[s_idx])))
                if len(alternatives) < MAX_ALTERNATIVES:
                    alternatives.append({"members": alt, "p_i": p_i, "match_probability": weight / total,
                                         "log10_bayes_factor": float(lnbf[s_idx]) / math.log(10.0)})
            groups.append(AssociatedGroup(
                members=members,
                contains_target=True,
                log10_bayes_factor=float(lnbf[best]) / math.log(10.0),
                match_probability=float(wx[joint].sum() / total),
                exact_probability=best_weight / total,
                member_probability={m: float(target_prob[m]) for m in members},
                p_any=p_any,
                p_i=p_best,
                match_flag="best",
                log10_prior=float(logw[best] - lnbf[best]) / math.log(10.0),
                alternatives=alternatives,
                coincident_with={m: duplicates[m] for m in members if m in duplicates},
            ))
        elif nonnull_total > 0:
            notes.append(f"no target group: p_any = {p_any:.3g} < {cfg.target_group_min_p_any:g} "
                         "(no row is more likely than not the target's counterpart)")
        fp_pos = {node: p for p, node in enumerate(fp.nodes)} if fp is not None else {}
        fp_all = fp.ln_l((1 << len(fp.nodes)) - 1) if fp is not None else 0.0
        target_norm = (top, nonnull_total, target_w, prior_det, tarrs, fp, eps, kappa2, run_w2, fp_pos, fp_all)

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
            (top, nonnull_total, target_w, prior_det, tarrs, fp, eps_mix, kappa2_mix, run_w2_all, fp_pos,
             fp_all) = target_norm
            correction = fp_all
            removed = [fp_pos[int(i)] for i in reps_g if int(i) in fp_pos]
            if fp is not None and removed:
                mask = (1 << len(fp.nodes)) - 1
                for p_bit in removed:
                    mask &= ~(1 << p_bit)
                correction = fp.ln_l(mask)
            weight = 0.0
            for h, tarr_h in enumerate(tarrs):
                if len(reps_g) == 1:
                    # A lone row: its two-way mixture weight with the target (prior included)
                    # was computed for the enumeration.
                    lnw = float(run_w2_all[h][int(reps_g[0])]) + correction - math.log(len(tarrs))
                else:
                    lnw = (_ln_bf_with_target_mixture(tarr_h, reps_g, target_w, eps_mix, kappa2_mix)
                           + float(sum(prior_det[i] for i in reps_g)) + correction - math.log(len(tarrs)))
                weight += math.exp(min(lnw - top, 700.0))
            p_i = min(1.0, weight / nonnull_total) if nonnull_total > 0 else 0.0
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


# Rows of an object whose core/tail combinations are all summed in its mixture Bayes
# factor; objects of up to MIXTURE_TWO_TAIL_MAX_ROWS rows sum the combinations with at most
# two tail rows, larger ones those with at most one (each further tail costs a factor eps).
MIXTURE_EXACT_MAX_ROWS = 4
MIXTURE_TWO_TAIL_MAX_ROWS = 7


@functools.lru_cache(maxsize=64)
def _tail_patterns(m: int, max_tails: int) -> tuple[np.ndarray, np.ndarray]:
    """Which rows are outliers in each term of the mixture sum (rows: terms), and their count."""
    combos = [t for n in range(max_tails + 1) for t in itertools.combinations(range(m), n)]
    mask = np.zeros((len(combos), m), dtype=bool)
    for r, combo in enumerate(combos):
        mask[r, list(combo)] = True
    return mask, mask.sum(axis=1)


def _ln_mixture_bf(arr: _Arrays, members: Sequence[int], eps: float, kappa2: float) -> float:
    """ln of the Bayes factor of ``members`` as one object when every row's position is
    the heavy-tailed mixture ``(1 - eps) N(C) + eps N(kappa^2 C)``: the sum over which rows
    are outliers (see OUTLIER_FRACTION). Pairs use the closed form of B&S eq. (16) with the
    summed covariance; larger objects sum the terms vectorised (same arithmetic as
    :func:`_ln_bf_python`)."""
    m = len(members)
    if m <= 1:
        return 0.0
    if eps <= 0:
        return _ln_bf_python(arr.lx, arr.ly, arr.lwa, arr.lwb, arr.lwc, arr.lld, members)
    ln_core, ln_tail = math.log1p(-eps), math.log(eps)
    const = LN2 + 2.0 * LN_ARCSEC_PER_RAD
    if m == 2:
        i, j = members[0], members[1]
        dx, dy = arr.lx[j] - arr.lx[i], arr.ly[j] - arr.ly[i]
        axx, axy, ayy = arr.lvxx[i], arr.lvxy[i], arr.lvyy[i]
        bxx, bxy, byy = arr.lvxx[j], arr.lvxy[j], arr.lvyy[j]
        terms = []
        for si, sj, lw in ((1.0, 1.0, 2.0 * ln_core), (kappa2, 1.0, ln_core + ln_tail),
                           (1.0, kappa2, ln_core + ln_tail), (kappa2, kappa2, 2.0 * ln_tail)):
            sxx, sxy, syy = si * axx + sj * bxx, si * axy + sj * bxy, si * ayy + sj * byy
            det = sxx * syy - sxy * sxy
            chi2 = (syy * dx * dx - 2.0 * sxy * dx * dy + sxx * dy * dy) / det
            terms.append(lw + const - 0.5 * math.log(det) - 0.5 * chi2)
        return _logsumexp(terms)
    max_tails = m if m <= MIXTURE_EXACT_MAX_ROWS else (2 if m <= MIXTURE_TWO_TAIL_MAX_ROWS else 1)
    if m <= 4:  # few terms: plain Python beats numpy's per-call overhead
        x0, y0 = arr.lx[members[0]], arr.ly[members[0]]
        rows = [(arr.lx[i] - x0, arr.ly[i] - y0, arr.lwa[i], arr.lwb[i], arr.lwc[i], arr.lld[i]) for i in members]
        inv_k2, ln_k2x2 = 1.0 / kappa2, 2.0 * math.log(kappa2)
        terms = []
        for n_tail in range(max_tails + 1):
            for tails in itertools.combinations(range(m), n_tail):
                sa = sb = sc = bx = by = lds = 0.0
                scaled = []
                for k, (dx, dy, a, b, c, ld) in enumerate(rows):
                    if k in tails:
                        a, b, c, ld = a * inv_k2, b * inv_k2, c * inv_k2, ld - ln_k2x2
                    scaled.append((dx, dy, a, b, c))
                    sa += a
                    sb += b
                    sc += c
                    bx += a * dx + b * dy
                    by += b * dx + c * dy
                    lds += ld
                det = sa * sc - sb * sb
                mx, my = (sc * bx - sb * by) / det, (sa * by - sb * bx) / det
                chi2 = 0.0
                for dx, dy, a, b, c in scaled:
                    ex, ey = dx - mx, dy - my
                    chi2 += a * ex * ex + 2.0 * b * ex * ey + c * ey * ey
                terms.append(n_tail * ln_tail + (m - n_tail) * ln_core + (m - 1) * const + 0.5 * lds
                             - 0.5 * math.log(det) - 0.5 * chi2)
        return _logsumexp(terms)
    tails, n_tail = _tail_patterns(m, max_tails)
    idx = np.asarray(members)
    x, y = arr.x[idx], arr.y[idx]
    dx, dy = x - x[0], y - y[0]
    scale = np.where(tails, 1.0 / kappa2, 1.0)
    wa, wb, wc = arr.wa[idx] * scale, arr.wb[idx] * scale, arr.wc[idx] * scale
    ld = arr.ld[idx] + np.where(tails, -2.0 * math.log(kappa2), 0.0)
    sa, sb, sc = wa.sum(axis=1), wb.sum(axis=1), wc.sum(axis=1)
    bx = (wa * dx + wb * dy).sum(axis=1)
    by = (wb * dx + wc * dy).sum(axis=1)
    det = sa * sc - sb * sb
    mx = (sc * bx - sb * by) / det
    my = (sa * by - sb * bx) / det
    ex, ey = dx[None, :] - mx[:, None], dy[None, :] - my[:, None]
    chi2 = (wa * ex * ex + 2.0 * wb * ex * ey + wc * ey * ey).sum(axis=1)
    lnbf = (m - 1) * const + 0.5 * ld.sum(axis=1) - 0.5 * np.log(det) - 0.5 * chi2
    terms = n_tail * ln_tail + (m - n_tail) * ln_core + lnbf
    top = float(terms.max())
    return top + math.log(float(np.exp(terms - top).sum()))


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


def _ln_bf_with_target_mixture(arr: _Arrays, members: Sequence[int], target_w: tuple[float, float, float, float],
                               eps: float, kappa2: float) -> float:
    """ln of the target + ``members`` Bayes factor with every member's heavy-tailed
    mixture weight (core x (1 - eps) or tail x eps; see _ln_mixture_bf), the target Gaussian."""
    if eps <= 0 or not members:
        return _ln_bf_with_target(arr, members, target_w) + len(members) * math.log1p(-max(eps, 0.0))
    ta, tb, tc, tld = target_w
    m = len(members)
    x = [0.0] + [arr.lx[i] for i in members]
    y = [0.0] + [arr.ly[i] for i in members]
    ln_core, ln_tail, ln_k2 = math.log1p(-eps), math.log(eps), math.log(kappa2)
    max_tails = m if m <= MIXTURE_EXACT_MAX_ROWS else (2 if m <= MIXTURE_TWO_TAIL_MAX_ROWS else 1)
    terms = []
    for n_tail in range(max_tails + 1):
        for tails in itertools.combinations(range(m), n_tail):
            wa = [ta] + [arr.lwa[i] for i in members]
            wb = [tb] + [arr.lwb[i] for i in members]
            wc = [tc] + [arr.lwc[i] for i in members]
            ld = [tld] + [arr.lld[i] for i in members]
            for t in tails:
                wa[t + 1] /= kappa2
                wb[t + 1] /= kappa2
                wc[t + 1] /= kappa2
                ld[t + 1] -= 2.0 * ln_k2
            terms.append(n_tail * ln_tail + (m - n_tail) * ln_core
                         + _ln_bf_python(x, y, wa, wb, wc, ld, list(range(m + 1))))
    return _logsumexp(terms)


def _partition(arr: _Arrays, reps: np.ndarray, links: np.ndarray, cat_index: np.ndarray,
               ln_n_det: np.ndarray) -> list[list[int]]:
    """Greedy agglomeration of linked detections into objects (posterior odds > 1).

    Clusters never share a catalogue. The most probable merge is always done first
    (priority queue with lazy invalidation), so the result does not depend on the order
    of the input.
    """
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
    outlier_fraction: float = OUTLIER_FRACTION,
    outlier_scale: float = OUTLIER_SCALE,
    correlated: bool = True,
    detection: str = "independent",
    beyond_edge: bool = True,
) -> tuple[list[Detection], list[int], tuple[float, float]]:
    """One synthetic cone with known truth.

    Field objects are a Poisson process over a cone of ``radius_arcsec`` (density
    ``object_density_arcmin2``, default the largest catalogue density) -- with
    ``beyond_edge`` (default) over a wider disc, of which only the detections measured
    inside the cone are returned, as an archive's cone search returns them (an object near
    the edge may have detections outside) -- ; each field
    object is detected by catalogue k with probability ``density_k / object_density``,
    so catalogue k's field density is ``density_k`` and field objects are seen by several
    catalogues at once (correlated, as in real skies). ``detection`` 'independent' (the
    engine's model) draws each catalogue's detections independently; 'nested' gives every
    object a brightness rank u ~ U(0, 1) and catalogue k sees the objects with u < q_k, so
    an object seen by a shallow catalogue is seen by every deeper one (as in real skies;
    the marginal densities are the same). ``correlated=False`` draws every
    catalogue's field rows as an independent Poisson field instead. The target object
    sits at the cone centre and is detected by catalogue k with probability
    ``target_completeness``; the target position is its true position plus a Gaussian
    error of ``target_sigma_arcsec``. Catalogue positions scatter with the engine's error
    model: Gaussian with ``sigma_arcsec``, a fraction ``outlier_fraction`` of them with an
    ``outlier_scale`` times wider Gaussian. Returns (detections, true object id per
    detection -- 0 is the target, field objects count from 1 -- and the measured target (ra, dec)).
    """
    # Objects farther than 4 outlier sigmas beyond the edge cannot scatter into the cone.
    outer = radius_arcsec + (4.0 * max(outlier_scale if outlier_fraction > 0 else 1.0, 1.0)
                             * max(c.sigma_arcsec for c in catalogs) if beyond_edge and catalogs else 0.0)
    area_arcmin2 = math.pi * (outer / 60.0) ** 2
    detections: list[Detection] = []
    truth: list[int] = []

    def place(cat: SimCatalog, x: float, y: float, obj: int) -> None:
        s = cat.sigma_arcsec * (outlier_scale if rng.random() < outlier_fraction else 1.0)
        ex, ey = rng.normal(0.0, s, 2)
        if beyond_edge and math.hypot(x + ex, y + ey) > radius_arcsec:
            return  # measured outside the cone: not fetched
        ra, dec = offset_radec(ra0, dec0, float(x + ex), float(y + ey))
        v = cat.sigma_arcsec**2
        detections.append(Detection(cat.name, ra, dec, (v, 0.0, v), label=int(obj)))
        truth.append(int(obj))

    def positions(count: int) -> tuple[np.ndarray, np.ndarray]:
        r = outer * np.sqrt(rng.random(count))
        phi = rng.random(count) * 2.0 * math.pi
        return r * np.cos(phi), r * np.sin(phi)

    if correlated:
        obj_density = object_density_arcmin2 or OBJECT_DENSITY_FACTOR * max(c.density_arcmin2 for c in catalogs)
        n_obj = rng.poisson(obj_density * area_arcmin2)
        fx, fy = positions(n_obj)
        true_x = np.concatenate(([0.0], fx))
        true_y = np.concatenate(([0.0], fy))
        if detection not in ("independent", "nested"):
            raise ValueError("detection must be 'independent' or 'nested'")
        rank = rng.random(n_obj) if detection == "nested" else None
        for cat in catalogs:
            p_field = min(1.0, cat.density_arcmin2 / obj_density)
            if rank is None:
                seen = rng.random(n_obj + 1) < np.concatenate(([cat.target_completeness], np.full(n_obj, p_field)))
            else:
                seen = np.concatenate(([rng.random() < cat.target_completeness], rank < p_field))
            for obj in np.flatnonzero(seen):
                place(cat, float(true_x[obj]), float(true_y[obj]), int(obj))
    else:
        next_obj = 1
        for cat in catalogs:
            if rng.random() < cat.target_completeness:
                place(cat, 0.0, 0.0, 0)
            count = rng.poisson(cat.density_arcmin2 * area_arcmin2)
            fx, fy = positions(count)
            for x, y in zip(fx, fy):
                place(cat, float(x), float(y), next_obj)
                next_obj += 1
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
    correlated: bool = True,
    object_density_arcmin2: float | None = None,
    config: AssociationConfig | None = None,
    ra0: float = 150.0,
    dec0: float = 20.0,
    detection: str = "independent",
) -> dict[str, Any]:
    """Simulate ``fields`` cones and measure completeness, purity and calibration of the
    target-association posteriors (true densities and completeness given to the model).

    completeness = fraction of true target detections with P > threshold; purity =
    fraction of detections with P > threshold that are true; reliability: for each
    posterior bin, the mean posterior and the observed fraction of true associations.
    Also reports the global calibration (sum of posteriors / true associations) and the
    field-object grouping purity (multi-member groups whose members are one true object).
    """
    rng = np.random.default_rng(seed)
    densities = {c.name: c.density_arcmin2 * 3600.0 for c in catalogs}
    cfg = config or AssociationConfig(target_sigma_arcsec=target_sigma_arcsec,
                                      completeness={c.name: c.target_completeness for c in catalogs})
    probs: list[float] = []
    truths: list[bool] = []
    groups_total = groups_pure = 0
    for _ in range(fields):
        dets, truth, target = simulate_field(rng, catalogs, radius_arcsec=radius_arcsec, ra0=ra0, dec0=dec0,
                                             target_sigma_arcsec=target_sigma_arcsec, correlated=correlated,
                                             object_density_arcmin2=object_density_arcmin2, detection=detection,
                                             outlier_fraction=cfg.outlier_fraction, outlier_scale=cfg.outlier_scale)
        if not dets:
            continue
        result = associate(dets, densities, target=target, config=cfg, cone=(ra0, dec0, radius_arcsec))
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
        "sum_probability": float(p.sum()),
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
                             threshold=args.threshold, target_sigma_arcsec=args.target_sigma,
                             correlated=not args.independent)
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
    parser.add_argument("--target-sigma", type=float, default=DEFAULT_TARGET_SIGMA_ARCSEC,
                        help="target position 1-sigma (arcsec)")
    parser.add_argument("--independent", action="store_true",
                        help="independent Poisson fields per catalogue instead of objects seen by several")
    parser.add_argument("--sigma1", type=float, default=0.1)
    parser.add_argument("--density1", type=float, default=5.0, help="per arcmin^2")
    parser.add_argument("--sigma2", type=float, default=0.5)
    parser.add_argument("--density2", type=float, default=2.0)
    parser.add_argument("--sigma3", type=float, default=2.0)
    parser.add_argument("--density3", type=float, default=0.3)
    parser.set_defaults(handler=_cli_calibrate)


# ---------------------------------------------------------------------------
# Embedded data
# ---------------------------------------------------------------------------

# Gaia DR3 density map (see _gaia_density_map): 12,288 HEALPix order-5 NESTED pixels,
# uint8 q with log10(sources per deg^2) = 3.0 + 0.015 q, zlib-compressed, base85.
_GAIA_DR3_DENSITY_ORDER5_B85 = (
    'c-l331!E&glJ3uVGu>vAZCT9B%*@OzSu)DDWXsfsHniOgZa2ep_wWwg?%v((&Cc$--}EBN{mLq<RLaVXjLiJLFCsNGJTyE!JTyd{gF}PEBSWL3'
    '<KtrkBg4Z3qXVNugT#SfVZo81k<p1!(!|u*#OMTRdSY^VhD4^bXe`q7)C`45XR<jQo>(fBb42ham#fLt87hNGXU$N_Gc*>J#g%Y`N{v#>5%I+e'
    'okBA-IDqvI4~@VvIx;poF+`dio17e(7$HqgPEL=)F+MhquZi)A3DWo^wnU~*PEV1kbQ)uVJVjx!8Ei6bhR)=$d3-UKCsT5GT)sdmRw#8;u!+Ox'
    '@bNpISSS<8rAm!nYgA~|dV|$$*XXorlToG6Y1BrwUS)E4-GN9ZmUQ|Yu3#b<(Ho3<t;6g#8g*v7*Xi^4d|qF~6Ak!6E?3l_DHL;sN~KWDmeb`*'
    't=7tx3)MopkgAp&<!n41O_y`^=6tJJoGsVqYK?lmRLi9z$zVBK&Lo0SpV%a~IK57X&TO(coxWhe<?_XH$-FlbN#zTLlF4QB2BN{3)9(pJ67gg#'
    'oy->t#eBMuE#`8$LaA6T)N74OuADE_vxQtaRY-(9L02Z03i%ywd#PBh=5wWNvskMY(&bbv5D&O)PHV^;@H#CH6KP^{a$<67YHFHHo}QVZPE%)S'
    'bS9HcW0+ZNE>|!^1uxSXEO@b491fq$<?#6;k(4JCN+b$}nj_!}#R{34FBZv_TD3uI)LX1Bhu37YI9z^jP^MC9bS9%!W6+zePKU?lc6$AhP}~~~'
    'L}Hm_fy3oOePStJES5=?N}B}cj9RBNX$%&#*=}<wlq#LpYqaW2W}D6B^jYmrugf2d`9k4jJd-P$pp4t^3&R<S#N+8mB9_ip%Jo#CP^&gu%aH_l'
    's$8xWs`b|F{K7(getv0jb!~B>y|&(2-Cp#C1F>W(m4@0<=?u6wU&v>2@j^aV$iyn%d@`FUl}q(TsSdTxEzTFS)ke8iskNH*<@x$jwN;<56%*NL'
    'y;3VCLkWM#8}{0ac2mIPciW9NeYsFbM*`tMF`h}p{9&KZ1#Z`x)gHUYW>g!LjfMI4((F=uxwEpkw6?rbU0z#i7HYY*Myp)O7Sbz=Ys>TV%~rFP'
    'Ew*X}IA^P+Y%$fym(uA-#2<@z!a<mm$#B%~aXXzZi^Hr_>XbH<#R$tw9`}S@HoM*Cb9o##v&CT688vE|N@~#P)N+A@&tvjAbSi_&=L)$@8jB8-'
    'V0sD`-pn+WGBr6x67qz64wJ(Y^QA&Ai_4}_SQPRUc^b>kz@i%;ofrei7?~I+!Nu4Z!pP77;>hSQU`PMt#MI>Y*u*&2H4e)dmiWlf;K=aUC|nMY'
    '49i6d36CS>DP<}-K!T9PU@_1hE{n%z(3muU3YAnO7wgqVtwN%d@?pX;>1?`yClYX&Ts9dj#ri1#G*c7PlcOW!qoV*N<71@pQ2-Q}&J^quox+5R'
    '8S*4)YH|XwWC{QTz+|GYZ=kR5`t@u5ef|A?a9qRV`j=l``||p=Yw*9nZ)gz83;;|5R1FODT_5Pb4iEJA!*$=l;I%KVeF4SbxQ@l)@A_38Pzzl4'
    '!@uia!gb$uqWZq;eb5kb^kEfP@9-e{YiI;qH!?KPk8nFMFbMAh{R4yGukn%L(b2In^xY^p6)+giVdx{ALxWHW7}P(|H#pFTHsM19fYyUUcnprf'
    'J7PC}9>h-I3vYcsOnS3HE!QY5Mw>~e(yJvxnV84sbLA2R47iAIH#@Bctx4~;dmUzj#i)|26%v6|q*dwEGMQWg@XDafP-oaoE(^wu26F>_Oq!XY'
    'P^MvQxNMj{bQXiph4EtX*;ERR3S&ZLz&(I(hS%wLTTM1|z!UV^Ee@+5z!~O(!U#a3hNU0!MFUQ&#}y03qkeBVU@<wYCWFNUFygjZoOZBRf^{oo'
    '8kJO{ltUX_CX34v38XN&#X_A1>Onv;*{nvu6onk7zf>Uu3NWbDdi?~<opB&M60pq#oH&1gh(>{I#=+;~NH&0uQ)u@LnL;L!NK@kzBqX8f$*Jin'
    'B(0GVocTCF<KqOK4&xMr2@cPb;19ne!7~$pv@rEPA0&8@Cc!sMDuuz|a$%jZc>*f<gfhdRu~|G0lfh<7#B!NXD3z*}I;~Q!QNS7z@t`c+my1O*'
    'DH#BJa+*S=Q<+Q}JUa~_J_R<TZ>caI95$cFVsW`5p;RIeh{XuvOe%}UhxNl{a5*eMb~YREokpS1=oA`<i>29YIvt*7Qt2}@SR;%npaS{<uPHP#'
    'Wtt2G44&cf*&Lye$7izWbT*g47KsD`i9{sAI{17Jmqw*fs0<2?0`<UCGfXC(&f+jxG?)eq_zzwfCru)^qi06p8r(7oz9NB_C&$J|$Iy{5*M~>Z'
    'vEXO;3<o+L{5y)wHASA9o`feRr%6*3oLp2Y1?-%L7wD8RP%D#Du!=yiOpF7xk*21h1!xaXsay<{BbD++Jb_5W7buicxl$!ph$Y~4nOMx>a@brR'
    '_zdl4F&KOSmnRVNd~E0gm&;<SwF;$H3s1-;P>)2a)2Y-tJ>HcmR2rp92z83Y0wM4wND>~;CKF2(Dw#qk7Ko&BiHu64P-)=n82~2=g-m8JU=&#l'
    'CIxzc;{hlI_D)U_(*^u84Shu|HASAB1jZ(_*dV!hJTBPBK)-@*e34Wn;qo~=XhRHj)95h9FalIMgT~<SSzPF=5X!Q7LY`olI6(R#=D>+=rU4Y0'
    '3B5NoGyu{M?gJe4^<j~rD<nZEgu?IwaDw3&MfgHU8povoGoDxh09Ys)|G&1xdPbl+RFMP(K~>igB?tQkuHyy3a6jS~QU4Ia<G^*qP5@niTs#2c'
    '`U%K`|AWNkAhOK>VImYK&=XAu>KVW`2Cy)~BLRz8?5Yn$Ik4+WS>ikcq#YyXA88860cRvk6%s%!PP$R}2OR}ofs(L_N%#b=Vw}@dm~j9HGcfn4'
    'Og5dx6>tS~)SUtjAIA>Hi$a1)3iLyvGigkKL_Upy?`%GS`sk0rE8sxSVcP_xfMvw+AdvyuqZB4k1?D<_H-Tg}MZ$V#NI0#*1{#CP;P5y+So#b$'
    'm(3-<r;sVIN~TeZ({P?}nOrCbhO)R0i`!{7+N=RzB;<Ald?3?IfSCq6V6G+LuqES#OeB<ySBkUse5P6o`l2DPJLpd(^0`<zod9{Kmdn&CgWhV^'
    'X$=OzU<IxOjYh9msdPH0!|Sy{8=+t<;`fC@DC*#Jcs;?O#~lnV&u^}^X4{L0JGYN}-TmEKdA?c9)hnyZTk8w0wUwt2-oLta^5ntCcb8AzUOv4%'
    '*u8VS)jQn3cYJ<wa=g8lN|f^PcsARpE-jSH^K<c7HXV(o(v@<nRW7%h8=dXVPG@85=I+VK_KlkdtL?4cN_%tj=E2d?-i^a;2vQV?rD~NPSFTPh'
    'g_SE)Dm5B|K?^7FhRp=_f^PzDPcUG%I2;zM)8+95!~Q@tDw9Jrqta-#29sHDwAj^Jo!MZv+AVgM$L010Y<7>=<?;K2!DuWLi6=e&U?dcar{k$?'
    'A(JmvVzEp*7E5Q!rP)TY(5NSp*<31J$QLWKtx9zcgyBYaajDzg+JY|Z-#l1e+3Iz=+uM6v$A<?8+xwYpshllT8?&wDmHCC0l~T3Qtk1R<7nfGr'
    '@Na2-V`pc*y9pjU+TXpgx87-W7h21$jm~mww$oi|UY<OEd*{JduO7WVyZ`Rv<L9ThU%Wm(dHLX*53k?8efj$N;qk`i>G{sia(lM9x;D3P_rZ<B'
    'M~{w<x*H2iz24ID?(WT_-bQ!s-o2;K&$l1$ce`nKx0PHhq$_j5m2T(a@$vEflbbu68_SF9YxDCfE8X?pR;RnN0;ArZt%1nRRcrZtA`uQJli_f!'
    'p3gUCi-lw|5>6&U0iPe_z2D`8Q7~F92AvONtlMR^n2cJOVQN*qn#)$pnKY;ae<bYnl#0njJ{yk(12AHKr_<>$Lx^BB0DD5nCznGVdb1f&S_R>T'
    'LMmCmwRv*hJ6b=uvA(`qUdr!mZFFyJ?)J(X3mZFI9hf>B5GSlHfbuUkEBV~g3JCv|a;=ojrc==D?&el!bp>YCPG@6nu3f9oE-%cNYvqMXrJSo)'
    's_o@ky;ZAJbBTB`mkW5RB?#)Wxnv^Y3&z7CKa749`|9^O96qnzmIxCo(cyMj&CpAje_E~4sL?3oSB!7a5lTQMlR*$B6^XoFkKOIC*q~b)ol&m='
    '-`M;fr`=#Q>TPC&UaM89v>NCws0esQ3|s{h7N)pN1i`R?55P8xa&i<k3xYpF{=ikpk7AOD$^pIscE*OTQUybR%|lnDF*Gv3>xa((zJwqK)a?V{'
    'MZhKQ5qLX9pfl>+!7JcJ{n<~D$-pq^uL(q6AZb(p<0FG3gjxlA;9__fG(FHXX`DPso`NzYOc9_-B+zXD*Jv<O3T7?{g^Lgmw9t?A(hv5d9vdPG'
    'jS||IU@tTZ!5;)MmH};keGr}il0dx<bkmQ7LG%G^9R?2RCn)J!|MdYvl>;vUApr~Z10|uNM?ymS8NfH_Fp}3*M+ecL1H_5t2Z(-O5f<M1{O#bC'
    '(}#z_=g3uPKin8a?_roUJV4}Vhy={=z*RB^`aL{CR6{uPGrOUA*Z?txL-_qrKUxBXv4(*wJ|VI#aEy_sDIhT>NwgU{74<rbQq;tl69B<Vf;f*!'
    '0Z9egluM_v7?>o$a-bF=o}yB4>`<GbfWu4;ns^nRV0<?^0`VWJt_d>gXwvjF(iZCENvMAUgqaZaEK9(ZN)&Q2Xm(IpgpLKZ#OHtpR)M;*%hgJQ'
    '&SU^y(n&-LsQ}_cl|rkLNmU9$ms3DN<8u&IkjaEXouQ(tqR{DFjsWZ7N<?z0P#_UAQU5Yg6^n#Co=`ACI0I)DI0O4akTLQx5dIi03PK=EqKJfp'
    'jFAG~m;&_+!gCsF8C{8Fjpc|+K)PZ%5D?Q77ywUWwh9z87)G9=fOeZf_krr8G0-oBasb7SnHKU4XjibG0dW<b&SYT>z@kH>%LQe}6$%(E7MsoC'
    '@=%3>vJ_#Q1;U+<>J+p+m(GA^SR6JRR4WKh5tjo+`2wK;RH#HEQwW4&iC7|)OJxeBLao-zK<{ewDxE|ok|?!umBntddqeJ^(W*B){SL2C!WYSu'
    '5(Qt#<%?xPiPosr+uRn1N~=(tYz9jr7t2)UOU+0slql4)rPY<CrA~KlzO`6uE-x;&12JDHnTw{}e!C}}2*gU&T%ol%+sYJDxq7W!WAj)Xfk-K0'
    'LJYzMy(Cjf71)MYDikX;aupq75H^>^q0(k35HB#m4uM1=lCil=j!?=ISsf;mH|+H5jaq}%qqjtp{y?S@DR})ZZ!8_nsC5dpQqNK;)G{Sh$oE(s'
    'Zcox1H`}!)ci0hVF3v72HkX>q?d9d=dC(BKN;VZrhHHgJDHYAcmKu!;fd4!Idt+|CxzMV#YS}~~HCLaXEoCZMkHhP-=&i<}JM6O=9cHyctB?q#'
    'B8;VFB84REj|N;8w=Euuhdd6SORq6O0hQFEw;9!Pt+F}an4fJ{XREX2N)0NXZ_PF9xms~<w$*3=%@oVIQVEnnDj&-hiiLc$QEpT!)q13yt=8++'
    'N<5s31UvzEI-XC4yb-_M<gh^78mG-|GeO*5%#?GnU?Nm4R7>edHV*bW%?7K{=L))QW`~7517R6`f&z0CVka_)LXd?p))+Hz1#I5un1hTOd5SzS'
    'Nr9Mz3BIS$Av|Hzm{NsUrjaT%La9)qP%G3riNT_EnQca^(dKZuTsVK^YMF{B;PNF(k=$T`dF8a*v<3}$&1AmPv#7i=$pP^kWqg{(n4u0sP%%NC'
    'm>Ncs8N)bY2vbTU6S&F%NTBE_i99~V<j|P{8CMLEDxD)@^Hq9<#%M8^WeSN*qgU!?XkafJV_{4@K>Q}=2;^$1lF4Io#44%G7d3~|rF7Zt_W0t3'
    'c)nh*f&c5-N~2t9l&V!vz!QunV+n`f>WU<N(Nd{UE|fFbOgU97#VYAisanaWikV8KRxaiW`H(9Xa9CaTL^u`oIs&e0JzFc~D#ds)5h)dO>2x$3'
    '4|>ABTq>822I3);)ntX?3!1=V?vTeH&&RVU5n}0ZJfF=a^k$6(2;QYP>x>SM-4zH1!_jy+8jSe8@OU~?EEOt+MkQY=WNP(tBcIM@<Do=2lgg(O'
    'k$5DR%cQG~LZgr^7P7T+xdMGl#nMSIx{yg`5+0YsX1AD)R<j8NuTE{!=`<i{VbLh%a+y?Y)akWwEh9clq+$q_#bTjIAOc9?L)@v9>!lJAAS;|A'
    'KAaHa!WR(!BL=aUe7NR8{LF!CHjD$_hwzqxu{49upvy7H7KsEBs8hh>^Wbp~7oOy>IV=_gwRjIgXEu++#t;`62%dt`#JmCk{S*rVDjE&(AMg{f'
    '6XNL^U?wVpEg<VO+y!u*L9m9f4tQt=Bj{;N7)+8Nkb=tb08j<sga+A!olNGHbpXjwFER;C9tt)|!#oKEqjO{#@(et~z~GyXWuRT?8k5Ol;Ag@{'
    'G!hKK7gz&q3r2wrgf;MQ5{r<Dd+?b&g{Dnm0v5tHOlge6B?&%2^bZ<|G&%;$k%Vz6ur0wj(?k{uZjy);$~dq#iRc>0BjN@qt1*I|@EpbEhEzu!'
    '$kFIkIyeo&MIavdqY%j+V0bVW8iM=4^T0<`CU6uOig{DQuIWi=k+28C;4$LFWelneRNPhdAOyxiqoJ}KLwX&;9Kq<7DjGx`IXJ+gGHIAsgK-1_'
    'LM3BoCdNo(*l(<nMQ209R5FpGg2g;VB2Qv&7n44d<C7$82x=W78X+{@AR#UZ1q6Z%bR^LpYDKVTVi3h4$S{ze*#01qslYl$uaqL8jK&Dbg(?8k'
    '76XJ{9YF2YKY)0I8o9q8Gr`yTCPoJF8A8TkK6RKr#2^#7DimFWCWFQX`$h+d4DfZ-mHk7Q{k@JdnGl&oE*R6pgc1b_ipgV?q1P}keWg!{{4|Qr'
    'zW%G^GF)Bz^4c{NtJjFs$TcW?4GUcByGlOe<JZ3Y;*0-t?b;V#e)+|h`0ooS0T1DmM7H|+Uw-|^KY#n_U;p#(|NG~^|M%bj{@)+J`|tn!=Hp*~'
    '{^vjc{PU0B|NQ4q-`+j>^zYrb@Bj8+|M|-=KYjnNKR$l?<BtzNeD&gwe?7SS>(_6eeEa77<jKR|e)#?G@1Fno{$S_)c(c2|^W@&!XJ<zb@BZ=o'
    '-+%g#AOHH>#f$s5e*XF7{x|P_{_gpc_pg5b>4#rGeEjz3KY#w^r(Zw)^xZcfE-ycRym<Y`zkmDVzkmM6@8A9K{_Fqx>kr=@o~`fPe|&gzX|+^c'
    'TW^#vpC6q(yF5ARc51Wh+s)=TZ-4vC>!YVPfBE$Fn`e(6KD_nt&ci1k9-iEKeDCzmgWE4IAKpB;d3fVSZ*{KKS!>PhA02F;-=3Y{+-R?@EN$$x'
    '7dBRAYm2SD&C`R<($0Fd*r?|+<>Gv6WvN+Tn7eoS^zPA(+oxZ>`ss(KcRpP7I{TX|?QZvE@6Orw&gp47QOHEY$#^*r{F*N2gZ^m9>k9f3@oXj<'
    'O2y{to#lFMVSan#V7Ifny;;sTYx#7g*q&Qoo2@O)|Ht26{q-+zzWwU<>(e{;k6xcYdh*r##}8gV|MA_^*6R;PpE`4$oyAUjy>oZ~A^*{>y}6@X'
    'rN&0({L!sDcY2TSpWV86d~^sMeSGos+2z}}H@1#$ZmjL@+&h2%_~_{2z4iMm?aLRNySq=Gzj=B2>f^f)x1Ya$@#6ihgRh<)J>NUa7L&0`E|HCe'
    '0)bQ{7%J71$#OZJibq4?R5B9xx*ZOW*Wt8SjYhlGZ1x8nE?>y)1)XTNyR43CC7W$Da;0RlkVs^bspXaC;`;jHQmLLTHk;L2G?oZP6QP*b=k^3c'
    'USBE~PZWyjOgQ56M`OXTPOnmHwF;?1j@%-V8%!#l*=jVaR0;*~w^|4UAd-NN6>zu$fs>2$Bb9?X#5^ZZhRtcVI9)cI(Wup72FB&K*}Xo83+gag'
    '!A7N8rT{`y!V@B-8l6F<F&MR4(7iIu+HUQ2HV%(Bcjnu*=E~~)!pZsW!M%sar=5+ZmF?ZNZl#*d)f<IUA{mLKGSPTzzEWLUZp`M2$#kib&0Rdb'
    'b$0Rc?%mzZ^^JqG%^UBYUOYR!c=X`>&i>Ke2RDz~E6w?}jm37gRxHoXRjQjitDT*_?)t)Ft+5E=((m`UgAuRa?yy+Ec1JuB3?)<HsMqIm2ZA1t'
    '$*co^=#471T&dNm)Hb`>?0~X*y;@^1X>{o<X1@{%^ldoe59Eu<RH2a0M5BRVEFKCwT~?bLd;-4Fo6H6yjHSctbGmRmtu~X%Y&7T%daXvMRx4CW'
    'xlE^3t27$53G)p~i$tcBD?kOv#4?#wr&c;`;BtXb1d$Ua{(;F|jaGN{{@J(Xo+)_y+s4ZGFaPKD!?UL+-#wi#95y{_p@_>?D1pow1V>V3Vh--$'
    'F+TI<jIt1XS96{@%9pL+y2dDgZ~#jJD?niinJm)?g%T#4j%omTiiQL=jk_<H^q(TpK9e!#`jI=4-;rr~1cyS@Hii62?6wd;Bf#kPbH110sJ-Ql'
    'TtKCg1QjB_h|l53K*)&20+2A6_Xi;YN`Rn95D{>auXb5*_W~j{P#IWOjM9Y%Q5gsqu|y!0fS7@2IeduF7*HOC3+DDQ*RKYh!QrzZPNNa~AgD`_'
    '!;!!7gliJ=IqrWDMicu`1WOTHewbUwJH);Syn_>_QCwgRK#5Zn8u5DzChpK+dq~eH2v9#@eOMm%ZziWnlhZR()5Ji4)8H{wA>b}lNf5DLIc$uA'
    '$_egFUv05WOc3T=C89AcJx+Wi{)nBhD?2eGFoJA22%j;(Jw)uE0uK*g<^2iv2Wst~CPBTx^OG1dUr7^SUm}Ts37esD+_)k36~>^Z$zf#e@uB_+'
    '%z*$W_F<j`DU`_1^b@I&e&oUa&wH%{1Jg*|{RHhJ!NR?(typB-f&P)JpT2-SqgVE!?a-AWe23FHOr);}j>e5vVqbwsxm-tP?n9%|CnKMqn}E(>'
    'yblf;M|WWwXLtmkB6c7KM#kVZK8E*)uocWSA)gOi>nBJXcZxr69uqv@hug_O<w)(sch^6Y`*lJhpd>)u&<~%##OwaSKHQTT!5$6_4-8$$8nCy7'
    'b|56kb=+g=@4reK*&SAg4`LjP!({OUoW4*jnoQ(lF^n+cxn#j$(i<E;yGLixLx5tjh2p_jrkE)PqP|GFlq#0%mDWPHy;)zJUFz(t@9vx&+<yH0'
    '$)n?Yr;naIe)+Mn)L2^IUq5UtG}`MoHxAD)?md6~_W8s67kBR5+T7VOnh?M(j*@-JZ?;+N-moVGRFd}u-GO*EmhHqhk`}wwb|bRsvsld*SHS6y'
    'Wn&qq*Wrz%BkB5LZLxQ>d9%7$UFzNJ-8{d0c6xGteDn6hJGamG_SdRQ<+<KrXSXm{nr&|`Z5+0@R_obBb$$Ebu(U9nzI=J}@yB;xzj^fL-HUfm'
    'pWVNF_~h(#|6ud>&6Az2)zyWw3#jVWox{ERXXmFkj<<K#y332rR&{k@wbdwA3!6Y1^R-%Gp*~v#*o7!O9g6^P$3p<fPN%Js#ng7hmyV|*5Z<}G'
    'NLCiy#kQG@I-UCV@!9cyZ>w{-*XwPrtS_w2F4W4|d~&v2&S&E3=)!VmeyOooU0qpOUaZZ-cf~Z&X`qnGCnNrlH->?r-DwHAeGa<`h*NLGbgx3M'
    '(Wx+E^Vr=s3*5us`oKMnO0JYiMG~PB2v7_p+Oj3|5HDM-I<-b`(OaB;r$3sGrky?~3~)59#F!rL%T;o<-eR=EeSZYT#T9S`;`u~A-zY9DY<IS6'
    'OZDY$Z=-k6>ns<WvDu~e+)`m}akIWsZ*Oj_b?eKOxwVaz)n;d{x7F)yt*_7Z_I9?nWNMk(0PV}wa<#!~u)2fpU^1V~x<fGj>1a}}k!uWABYv;a'
    'nY2ci&jAx89J9Ht&Op>3smzsIE3570a<kZ6>8@-n^cK6jTfNPzqqkk0%~hAz7dx3ssyNqfECEe-dYfIaa&2|Fv$D3lw6-u;sTWrk))wleMlqAf'
    '#DWlzXQJtd-{o_57rV=iQmfK!uP?VMtx7tY340+B&xA7(kIieh8SO@Jozf0Qf~yq3ltKZQ&jBBccx)aUOgBIm<Yv9upiyWQVxd?7_gP|yczGNi'
    'ySu#JhI*?T?OuDXI#<iaa&ezM-~!hs@wY_Xc4uX-HeX+DFRm;!mYRiRF&*(m{iSRr7YihUCWFPORccjslf$f4>D5x9RK$m9SuO%X<M6o-v&*Vi'
    '8`N%_$6?Z%bfC1P0<Ms!l&a)no><^?djcL`C=v*Vqw!cW?+--dp-4KL%9rcqYI8oCh!%>q`dq45EYG#u-R8p5T6?Xxv%a-+uzzyvel%GqW%HGK'
    'qq)2~ztrg!%gy;(qrJS^*}k#fJ2>3fJ~-YxJimMU!NrRwm#^O*pFOy^0}$}y-KU?v`Tn<GYKwE-_3ho`)!p@*x9&f9egxk?eD?b5mmj|V@w@Nd'
    'Jw14E_tmR!zWMFhr;p!$|Ni;ipWpp@@6O4i7cXDF`S{iQmlrP|efakMcV9pI;ql!^H&0)@{O05B{nuy5kIvq{e)I0@m(L$Qzr1|+?9shvcW>W3'
    'J=x#8czAyM;k~otqr=^;8$0Wr=G<y`vEAF***k?LfB*4H_r^hQ=jiPG?$hTFFW<a>`1I|ESMR?2@#o)u`u3MkukU<<IOFTrZ%<!7e)js!%dal?'
    'kMBM{hyJ{L^Wm$HKYo1s<nrCex8MHsw;z7^^!+b?|MdNxS9ebDJ>6fwc=hPx`<EX-zJGTAbpLU?`{BcPzkhuG=kNPx_n#i!zVq<?i;KrkA3wNv'
    '@4@Ah=lAa3zIW@^&GWs(z4Oz<{gcBR+grWWc4u{QZg*#MeRHe5x-eU-%r4iOw=PfaJiEB}bo20V@9f_G(Wk$@{Pd6CfBpN#i_^3BUq8Iq-Ro}b'
    'Zg-cLmzzy^HG4a&iwB2|cG*)er)s70SC?;}ojiU0>gAiKPhLEEaO=&*#o4{n!=2j)d%f+A?&`(){iCDZ{ms+;eZZ%cxo&4=ae02PF$e8ei}}pn'
    'c5h>|x3<z*S(;mfoACAA;zG0DsFw2OVkSLXuN2FLY$}&Y#uCwRAOn#_A{GvY0zQw=<8t)c?bckaR%}(Pg<>WXFXgkTWGouQvH`Ehl~1QY#)f>6'
    'V8H8lyB$7{8xhp(uv$&Hy%h=iy&$0-Zl~R9x0y{&hs|QMnv5n02=qFQ+F~;33?OQ?utt>-WPrTIcNv5mBB@9yoNMAQH4B-1HW^PrqX5_GQYo2='
    'hJt}m#P5%U{5~+%1@+sYYKza~ba*`uJMKVQEJkCgluDK==`<Aghr#4>DU-^@070X{KsXu*csyRa#{-HTgto<EFnGOArw8>nT;pjr0kImi8USr2'
    'ggkPzS*KO0lpwo_JwKU5EJZ^>VhZ?Nt{6ZY;vPN^v?}JVa6eD0xk}DKSqXwaKtDt_5Z^!@A`zdDzd_+bS=6&4F(33QX60}jfyD+$2VT$<_yO-a'
    'g#x`Km8cYIl^kkt8r@dh1@}7r9-GBw(`f<jR1hE-EGC^suLbKQm=Bam<Y1*(2!?SvEZkoq6f{>MQ_8^8VwFOpl1UUYAzviqarr#42&~`<`2k-j'
    'fO+v~C=m^KLVl0{RwH1g6L&R@HghVTO-6&UP(D@6#v{p?$LVw9@5BP$u+M4tI5BboE|cpu1}$bS!5*<tAQGZIVv$6Qb!pW)4b){Z=!^!XT%~~T'
    'WNMXG4cAI0g9U1yPGd14n4z;6)T{j$3XMv`d?J|xQ4Xf`pf4QUdZE)8#P^sNqe7W!GNGACxWPk291vXKZW3zc@$vCd@(hIx_a`xjh7rod#N-5K'
    ';ie|0P?3+JFEG0aQI!yJ8SH`Jh)E~fgKP91{w@+{gA{+82j$r;E=R}{33yxqpU5(jXJ)`II(3FlnI;kYcEpAg{$h*Bv{IRL+y`RO*$jwcC>TA_'
    'D0Cv9SIO1#$wDSsFU{8SsZur_h=zP_zdIgH#e=?ZV6L&)suXMG_Cja5Szl;op)ihBIae#D<JpwchCA#wi`(IK!wRwMbtZ!b{cW;Z!1rbtC%?<-'
    'bw@+-XuuZ^IczSc)#|W&y`g~1;c@pm+dZhey|;68u-UoMo2||_3fW3&d2VfGwz5<|JGp=7aQD{n<>R;S9zS^hdVBNeX0Lm&cmM9|7kAE|U#1h;'
    'bTpFA<jeC5)yhIM8cN3ezG$eHuH+NZbgH|)zrVJy*}ijnaeHU|aJNxem@VXM<<+HLcYbztQK^J^E>+6$cY^p^7O_x*xowe5tbny6kw8DpdaX%s'
    'x4Ik_lijLT=rmZ~2x3U9)@xyP5x+wf3B+P8PsC@?m@tRvR2U;R?%_!#5IzeeA_dGf8LVbL2k0NxhY-s1IYPe2<p)IZxWfK;*yn}$Z*|&DFjrxo'
    'dmVO{Gm|J}BjH56T52_N>2e_$h=zThus;#ag3wGRv}zLwJ*~!ScGwI$yFn^dLI^99s^vPhRIX6CU0#pf>UIQu(XihW44U*di$-nGyBt2Z)#SAO'
    'KW&Ea{Q'
)
