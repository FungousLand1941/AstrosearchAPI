"""Time-domain astronomy: multi-survey light curves, variability metrics, periods, solar-system check.

Data sources (every endpoint and parameter below was verified against the live
services; see ``tests/fixtures/timedomain`` for recorded responses):

* **ZTF** -- object IDs from the IRSA TAP table ``ztf_objects_drNN`` (newest
  release, discovered from TAP_SCHEMA), photometry from the IRSA ZTF light-curve
  API ``nph_light_curves`` by ID (Masci et al. 2019, PASP 131, 018003). PSF-fit
  magnitudes calibrated to Pan-STARRS1 (AB) in g/r/i; only epochs with
  ``catflags == 0`` and a PSF fit (chi, sharp) consistent with the object's other
  epochs are used for statistics. For moving targets every epoch's measured position
  must follow the target's motion.
* **NEOWISE-R** -- IRSA TAP table ``neowiser_p1bs_psd`` (Mainzer et al. 2014, ApJ
  792, 30). Single exposures are cleaned with the NEOWISE Explanatory Supplement
  (sec. II.3) cuts and averaged per ~6-monthly visit (W1/W2, Vega magnitudes);
  single exposures are available with ``neowise_binned=False``.
* **Gaia DR3** -- epoch photometry via the Gaia archive DataLink service (Eyer et
  al. 2023, A&A 674, A13; Riello et al. 2021, A&A 649, A3) for the nearest DR3
  source with ``has_epoch_photometry``; G, BP and RP (Vega).
* **TESS** -- SPOC 2-min light curves from MAST (Jenkins et al. 2016, SPIE 9913):
  PDCSAP flux, or SAP flux for sectors where PDCSAP is unphysical (<= 0), each
  normalised by its sector median (optional survey; ~2 MB download per sector;
  stars without SPOC 2-min data, i.e. FFI-only targets, are not covered).
* **Solar system** -- IMCCE SkyBoT cone search (Berthier et al. 2006, ASP Conf.
  Ser. 351, 367; 2016, MNRAS 458, 3394), plus a JPL Horizons API helper
  (Giorgini et al. 1996) for independent ephemeris checks.

All epochs are converted to Barycentric Dynamical Time at the solar-system
barycentre and reported as ``BMJD_TDB = BJD_TDB - 2400000.5`` so series from
different surveys share one time axis (Eastman et al. 2010, PASP 122, 935).

Variability metrics follow Sokolovsky et al. (2017, MNRAS 464, 274): weighted
mean, chi^2 vs a constant, robust 5-95 % amplitude, von Neumann eta (von Neumann
1941), Stetson J (Stetson 1996, PASP 108, 851) and normalised excess variance
(Vaughan et al. 2003, MNRAS 345, 1271). Per-epoch errors follow survey error models
calibrated on constant stars (:data:`ERROR_MODELS`: Stripe 82 standards for ZTF and
NEOWISE, Gaia GAPS sources and Evans et al. 2023 for Gaia G, RV-quiet dwarfs for TESS); the decision rule is
described in :data:`VARIABILITY_CRITERIA`. Periods use the generalised Lomb-Scargle
periodogram (Zechmeister & Kuerster 2009; VanderPlas 2018, ApJS 236, 16) with
Baluev (2008, MNRAS 385, 1279) false-alarm probabilities, a shared-frequency
multi-band periodogram (VanderPlas & Ivezic 2015, ApJ 812, 18), a period-doubling
test for eclipsing/ellipsoidal binaries (cluster-robust even/odd-cycle ANOVA and odd
harmonics of f/2, BIC; Schwarz 1978, Kass & Raftery 1995) and multi-harmonic refinement
with Montgomery & O'Donoghue (1999) uncertainties, inflated for correlated residuals
(red-noise continuum, effective N, time-block sandwich). False-alarm probabilities
account for red noise and are checked on rank-transformed data (outlier-robust).

Positions: a target epoch and proper motion (e.g. from name resolution) are
propagated to each survey's epoch (Gaia J2016.0, TIC J2000.0, ZTF mid-survey with a
widened cone, NEOWISE per exposure).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import os
import re
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import erfa
import httpx
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from scipy import stats

from models import (
    AstroSearchError,
    InvalidCoordinateError,
    ObjectResolutionError,
    Target,
    haversine_arcsec,
    propagate_radec,
    resolved_target,
    validate_target,
    votable_query_status,
)
from providers import CacheManager, SesameResolver

# ---------------------------------------------------------------------------
# Service Endpoints & Constants
# ---------------------------------------------------------------------------

ZTF_LIGHTCURVE_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
IRSA_TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_DATALINK_URL = "https://gea.esac.esa.int/data-server/data"
MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
SKYBOT_CONESEARCH_URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

USER_AGENT = "AstroSearch-timedomain/0.2 (+https://github.com/CoderAwesomeAbhi/AstrosearchAPI)"

SURVEYS: tuple[str, ...] = ("ztf", "neowise", "gaia", "tess")
DEFAULT_SURVEYS: tuple[str, ...] = ("ztf", "neowise", "gaia")
PERIOD_SURVEYS_ORDER: tuple[str, ...] = ("ztf", "gaia", "tess")
PERIOD_SURVEYS: frozenset[str] = frozenset(PERIOD_SURVEYS_ORDER)
"""Surveys whose sampling supports period searches. NEOWISE visits are ~6 months
apart (each averaged here), so NEOWISE is excluded from period searches."""

# Upstream latencies measured live (2026-09): the ZTF API needs ~170 s for a cone,
# the IRSA TAP NEOWISE-R query ~75-95 s. Timeouts leave margin above those.
DEFAULT_TIMEOUTS: dict[str, float] = {
    "ztf": 300.0, "neowise": 300.0, "gaia": 120.0, "tess": 180.0, "skybot": 120.0, "horizons": 60.0,
}

MAX_LIGHTCURVE_RADIUS_ARCSEC = 60.0
ZTF_MAX_RADIUS_DEG = 0.1667  # "Valid radius range: (0, 0.1667] degrees" (IRSA ZTF LC API docs; also enforced by the service)

# Source identity is decided separately from the search radius: the cone only finds
# candidates, and an object is the target only within a survey-specific match
# tolerance of the target position (at the survey's epoch). Nothing farther away is
# ever adopted as the target -- it is listed as a neighbour instead.
#  ZTF: astrometry is accurate to 45-85 mas rms against Gaia (Masci et al. 2019) and
#   the PSF FWHM is ~2" (median seeing, Bellm et al. 2019), so objects within 1.5" of the
#   target are the target (distinct stars closer than that are blended by ZTF anyway).
#  Gaia DR3: positions are propagated to J2016.0; 1" leaves room for input-position
#   errors (Gaia's own astrometry is sub-mas).
#  TESS: TIC-8 positions (J2000.0, Gaia DR2 based; Stassun et al. 2019) of SPOC targets;
#   2" as for Gaia plus the propagation from J2016 to J2000 of unknown proper motions.
ZTF_MATCH_ARCSEC = 1.5
GAIA_MATCH_ARCSEC = 1.0
TESS_MATCH_ARCSEC = 2.0
# Backwards-compatible alias (earlier versions grouped oids within 1" of an anchor).
ZTF_SAME_SOURCE_ARCSEC = ZTF_MATCH_ARCSEC
# ZTF object positions are measured on the reference image of each field/CCD/filter,
# whose epoch differs per object (reference images were built from 2018 data and
# partly rebuilt later; e.g. CM Dra's g-band object matches its J2018.2 position). A
# moving target is therefore associated with every object lying within
# ZTF_MATCH_ARCSEC of its track over this epoch range (survey start to the newest DR).
ZTF_POSITION_EPOCH_RANGE = (2017.9, 2025.6)
# A track match is necessary, not sufficient: each light-curve row carries the position
# measured at its epoch, and an object is the moving target only when at least this
# fraction of its good epochs lie within ZTF_MATCH_ARCSEC (+ parallax) of the target's
# position at that epoch (CM Dra, 1.6 "/yr: 404 of 407 g epochs within 1.5", median 0.14";
# stars crossed by Barnard's star's track: none within 4").
ZTF_MIN_TRACK_FRACTION = 0.5
# PSF-fit quality (ztf_psf_outliers). Flag value of catflags = 0 epochs with an outlying
# PSF fit (above every catflags bit, which end at 2^15). Checked live (2026-09, ZTF DR24) on
# 37 Landolt standards (V = 13-14.5) and 23 catalogued variables (RR Lyrae incl. 6 Kepler
# Blazhko stars, EBs, Cepheids, Miras, delta Scuti): the cut flags 0-0.7 % of the epochs of
# each series (of the variables' flagged epochs, 6 of 101 lie outside the 1-99 % magnitude
# range), and removes the seven 0.5-1.7-mag-faint epochs of PG1323-086B r (chi^2/dof
# 86.1 -> 0.94, no longer "variable"); no other standard changes its verdict.
ZTF_PSF_FLAG = 1 << 16
ZTF_PSF_CHI_FACTOR = 4.0
ZTF_PSF_CHI_MIN = 3.0
ZTF_PSF_SHARP_MIN = 0.3
ZTF_PSF_SHARP_MADS = 8.0
ZTF_PSF_MIN_EPOCHS = 10
ZTF_QUALITY_CUT = "catflags == 0 and PSF fit (chi, sharp) consistent with the object's other epochs"
# Per-field zero-point alignment (see ztf_field_offsets).
ZTF_ALIGN_MIN_EPOCHS = 20
ZTF_ALIGN_MIN_OVERLAP = 0.5
# Offsets are applied only when measurable to 5 mmag (they are ~5-30 mmag).
ZTF_ALIGN_MAX_STANDARD_ERROR = 0.005

# Per-epoch error model used for the variability statistics:
#     sigma_eff = sqrt((scale * sigma_pipeline)^2 + floor^2)      (mag)
# The pipeline uncertainties of every survey mis-state the observed scatter of
# constant stars, so (scale, floor) were CALIBRATED on non-variable stars such that
# the median chi^2/dof of constant stars is ~1 (calibration done live 2026-09):
#  ZTF (DR23 PSF-fit, catflags == 0): 81 SDSS Stripe 82 standard stars of Ivezic et
#   al. (2007, AJ 134, 973; catalogue J/AJ/134/973, chi^2/dof < 1.5 in g, r and i over
#   >= 10 SDSS epochs), r = 14-20, 225 series of 100-1300 epochs, after the per-field
#   zero-point alignment below. Median chi^2/dof per 1.5-mag bin: g 0.89-1.04,
#   r 0.96-1.05, i 0.90-1.07. (The previous 25 mmag floor -- the worst-airmass end of
#   the 8-25 mmag repeatability of Masci et al. 2019 -- gave chi^2/dof ~0.4 for the same
#   stars, i.e. it doubled the noise.)
#  Gaia DR3 G: the additional single-transit errors of Evans et al. (2023, A&A 674, A4,
#   Table 2; tabulated vs G) are a lower envelope derived from the best-behaved sources;
#   on 56 GAPS sources with epoch photometry but phot_variable_flag = NOT_AVAILABLE
#   (G = 12.8-20.5, ruwe < 1.4) they leave median chi^2/dof = 2.5 at G < 15. A floor of at
#   least 3 mmag and a scale of 1.2 bring the median to 0.6-1.0 in every G bin (the
#   reviewer's six G ~ 14 GAPS sources: chi^2/dof 3.5-9.9 -> 0.3-1.2).
#  NEOWISE W1/W2 (per-visit means): the calibration offsets of NEOWISE Explanatory
#   Supplement sec. II.2 (W1 stable to < 0.005 mag; W2 offsets up to 0.03 mag). Checked
#   on 16 of the Stripe 82 standards (W1 = 12.4-14.5, ~21 visits each, after the visit
#   cuts below): median chi^2/dof W1 1.3, W2 0.7, none flagged variable.
#  TESS (relative flux, SPOC 2-min PDCSAP in 10-min bins): the pipeline errors describe
#   photon and read noise only; granulation and residual PDC systematics add ~50-150 ppm per
#   bin, so bright quiet stars had chi^2/dof 5-45 and were all flagged variable (tau Cet:
#   44). Calibrated live (2026-09) on one recent sector of each of 30 RV-quiet FGK dwarfs
#   of the HARPS/HIRES planet-search samples (Tmag 2.7-7.7; tau Cet, HD 20794, HD 69830,
#   HD 102365, ... HD 136352): the floor needed for chi^2/dof = 1 has median 95 ppm; with
#   100 ppm (scale 1.0: fainter stars, where the pipeline error dominates, need no scaling)
#   the median chi^2/dof is 0.92 and 6 of 30 have chi^2/dof >= 2, all with 160-650 ppm rms
#   (active K dwarfs, and nu2 Lup = HD 136352 with its transiting planets).
ERROR_MODELS: dict[tuple[str, str], tuple[float, float]] = {
    ("ztf", "g"): (1.15, 0.004), ("ztf", "r"): (1.15, 0.004), ("ztf", "i"): (1.0, 0.007),
    ("neowise", "W1"): (1.0, 0.005), ("neowise", "W2"): (1.0, 0.03),
    ("gaia", "G"): (1.2, 0.003),  # floor = max(0.003, Evans et al. 2023 Table 2 at the source G)
    ("tess", "TESS"): (1.0, 1.0e-4),  # relative flux (100 ppm)
}
# Evans et al. (2023, A&A 674, A4) Table 2: additional single-transit G error (mag) to add
# in quadrature, per G magnitude (the entries at G = 10.5 and 14.0 are blank in the paper
# and are interpolated here).
GAIA_G_EXTRA_ERROR: tuple[tuple[float, float], ...] = (
    (9.5, 0.0003), (10.0, 0.0003), (11.0, 0.0007), (11.5, 0.0008), (12.0, 0.0013), (12.5, 0.0016),
    (13.0, 0.0009), (13.5, 0.0009), (14.5, 0.0010), (15.0, 0.0010), (15.5, 0.0012), (16.0, 0.0014),
    (16.5, 0.0017), (17.0, 0.0021), (17.5, 0.0025), (18.0, 0.0032), (18.5, 0.0041), (19.0, 0.0052),
    (19.5, 0.0071), (20.0, 0.0091),
)
# Series whose uncertainties cannot support a variability decision. Gaia DR3 BP/RP
# fluxes are integrated over a 3.5" x 2.1" window without deblending, so transits of
# faint or crowded sources are contaminated by neighbours in a scan-angle dependent way
# (Evans et al. 2023, sec. 5); on the 56 GAPS calibration sources BP/RP chi^2/dof stays at
# 2-4 (median) for G > 17 even with Table 2 errors and a scale of 1.5. Their statistics are
# reported but the variability decision rests on G.
NON_DECISIVE_SERIES: dict[tuple[str, str], str] = {
    ("gaia", "BP"): "Gaia BP epoch photometry is not used for the variability decision (window photometry "
                    "contaminated by neighbours; Evans et al. 2023 sec. 5): decision based on G",
    ("gaia", "RP"): "Gaia RP epoch photometry is not used for the variability decision (window photometry "
                    "contaminated by neighbours; Evans et al. 2023 sec. 5): decision based on G",
}
# Backwards-compatible view: survey floors (mag; relative flux for TESS).
ERROR_FLOORS_MAG: dict[tuple[str, str], float] = {k: v[1] for k, v in ERROR_MODELS.items()}

# Variability decision (see variability_metrics). A series is variable when, after
# removing isolated > 5-sigma outliers (at most 1 % of N, see isolated_outliers), chi^2
# vs a constant rejects constancy at >= 5 sigma AND chi^2/dof >= 2, i.e. the intrinsic
# variance is at least the calibrated noise variance. On the 225 ZTF calibration series
# this flags 2 (both with ~2 % of epochs 0.1-0.4 mag faint: possibly unflagged
# artefacts, possibly eclipses); the 1 % outlier allowance matches the largest fraction
# of > 5-sigma points of those constant stars (median 0-0.9 % per magnitude bin).
# An extreme epoch is "isolated" -- a candidate artefact -- only when no epoch adjacent
# in time (the previous/next epoch of the series) and no epoch of the same or another
# band of the survey within OUTLIER_COINCIDENCE_DAYS deviates in the same direction by
# more than OUTLIER_COINCIDENCE_SIGMA: single-exposure artefacts (cosmic rays, bad
# subtractions, satellite trails) do not repeat in the next exposure or in the other
# filter of the same night, while eclipses, dips and flares do. The allowance is
# floor(1 % of N) with no minimum, so short series (NEOWISE visits, Gaia transits,
# N < 100) never lose epochs.
VARIABLE_CHI2_DOF = 2.0
OUTLIER_SIGMA = 5.0
OUTLIER_MAX_FRACTION = 0.01
OUTLIER_COINCIDENCE_SIGMA = 3.0
# ~5 h: ZTF typically revisits a field in both g and r within a night (Bellm et al.
# 2019); the two Gaia fields of view see a source 106.5 min apart; the detached-binary
# eclipses this protects last hours.
OUTLIER_COINCIDENCE_DAYS = 0.2
# A reliable, cross-band confirmed periodogram peak this significant also qualifies a
# series as variable (low-amplitude periodic variables whose variance is below the noise,
# whatever its chi^2; see _apply_periodic_evidence).
PERIODIC_EVIDENCE_FAP = 1e-10

# NEOWISE single-exposure saturation begins at W1~8.0 and W2~7.5 mag
# (NEOWISE Explanatory Supplement, sec. II.2).
NEOWISE_SATURATION_MAG: dict[str, float] = {"W1": 8.0, "W2": 7.5}

# Visit grouping for NEOWISE: a new visit starts after a gap longer than this or once
# a visit spans longer than this. WISE revisits a sky position every ~6 months with a
# visit lasting ~1 day near the ecliptic and longer toward the ecliptic poles
# (Mainzer et al. 2014); 10 d separates visits everywhere except at the poles, where
# coverage is continuous and the span cap keeps bins short.
NEOWISE_VISIT_GAP_DAYS = 10.0
NEOWISE_VISIT_MAX_SPAN_DAYS = 10.0
# A visit is kept only when at least half of its detections of the target (and >= 3)
# pass the exposure cuts: latent images (persistence, cc_flags 'P') typically affect a
# whole visit, and the few exposures escaping the flag are contaminated too (seen live on
# the Stripe 82 standard at 321.620483 -0.892541: flagged visits' survivors are 0.25 mag
# bright with w1rchi2 4.6-6.6 vs ~1 in clean visits).
NEOWISE_MIN_VISIT_PASS_FRACTION = 0.5
NEOWISE_MIN_VISIT_EXPOSURES = 3
# Maximum distance of a single-exposure detection from the (epoch-propagated) target:
# half the ~6" W1/W2 PSF FWHM (Wright et al. 2010, AJ 140, 1868), independent of the cone.
NEOWISE_MATCH_ARCSEC = 3.0

# Stetson J pairs: observations closer in time than this form a pair. ZTF commonly
# revisits a field within a night separated by >= 30 min (Bellm et al. 2019), so one
# hour pairs same-night visits while staying short compared with most stellar variability.
DEFAULT_PAIR_WINDOW_DAYS = 1.0 / 24.0

MIN_POINTS_VARIABILITY = 5
MIN_POINTS_PERIOD = 20
# One-sided Gaussian 5-sigma tail probability: the chi^2 test threshold.
FIVE_SIGMA_PVALUE = float(stats.norm.sf(5.0))
# Conventional 1 % false-alarm level for reporting a period.
PERIOD_FAP_THRESHOLD = 0.01
# A reported period must be covered at least this many times by the baseline: red noise
# (AGN) readily produces "significant" periodogram peaks spanning only 2-3 cycles
# (Vaughan et al. 2016, MNRAS 461, 3145).
MIN_CYCLES_IN_BASELINE = 5.0
# Periods covering fewer cycles than this must also be phase-coherent between the two
# halves of the data (phase_coherence); so must every period of a series that no other
# band of its survey can confirm (_require_cross_band_confirmation). Each half's signal
# must be at least as significant as a 3-sigma sinusoid (chi^2 with 2 dof = 9).
COHERENCE_MAX_CYCLES = 20.0
COHERENCE_AMPLITUDE_P = float(stats.chi2.sf(9.0, 2))
# phase_coherence: a signal detected at >= 20 sigma in both halves whose amplitude/phase
# change by <= 50 % is a modulated periodic signal (Blazhko), not red noise.
COHERENCE_STRONG_SIGMA = 20.0
COHERENCE_MAX_RELATIVE_CHANGE = 0.5
# Monte Carlo draws propagating each half's template uncertainty in phase_coherence.
COHERENCE_TEMPLATE_DRAWS = 100
# A period whose FAP rests on the residual continuum alone (too few effective points for the
# effective-N test, see period_significance) must be detected at this significance in each half.
CONTINUUM_ONLY_HALF_SIGMA = 5.0
DEFAULT_MIN_PERIOD_DAYS = 0.05
DEFAULT_OVERSAMPLING = 5  # samples per peak; VanderPlas (2018) recommends n0 ~ 5-10
MAX_FREQUENCIES = 2_000_000
# Below ~2 samples per peak the grid can step over a narrow peak altogether.
MIN_OVERSAMPLING = 2.0
# Accepted period-search range. 0.01 d (14.4 min) is below the shortest period any of
# these surveys can sample (TESS 2-min cadence binned to 10 min; ZTF/Gaia far sparser)
# while keeping the grid of a ~3000-d baseline under MAX_FREQUENCIES at 5 samples/peak.
MIN_ALLOWED_PERIOD_DAYS = 0.01
MAX_ALLOWED_PERIOD_DAYS = 1.0e5
GROUND_BASED_SURVEYS: frozenset[str] = frozenset({"ztf"})
# Sidereal day = 86164.0905 s of UT1 (IERS Conventions 2010): 0.99726957 d.
SIDEREAL_DAY_FREQUENCY = 86400.0 / 86164.0905
DAYS_PER_JULIAN_YEAR = 365.25
# Period doubling (eclipsing/ellipsoidal binaries): 2P is adopted when adding the odd
# harmonics of f/2 improves the BIC by more than 10 summed over a survey's bands
# ("very strong" evidence, Kass & Raftery 1995, JASA 90, 773).
HARMONIC_BIC_THRESHOLD = 10.0
MAX_HARMONICS = 6
# Final refinement may use more harmonics (narrow eclipses need ~1/width of them).
REFINE_MAX_HARMONICS = 20
# Period ladder (see _finalise_period / parity_test): a doubling needs a >= 5-sigma
# even/odd-cycle difference not contradicted by the odd harmonics of f/2 (delta BIC >= -10),
# or delta BIC > 10, of at least 5 % of the variability amplitude, and 2P must fit >= 10
# times into the baseline (>= 5 cycles per parity).
PARITY_SIGMA = 5.0
PARITY_MIN_EFFECT = 0.05
# parity_test clusters the epochs of one cycle within one observing run (a new run after a
# gap of more than 6 h: a ground-based night; a TESS sector is one run).
PARITY_RUN_GAP_DAYS = 0.25
PARITY_MIN_CYCLES = 10.0
MAX_PERIOD_DOUBLINGS = 3
# Eclipse shape (eclipse_shape): one dip per cycle deeper than 70 % of the profile range
# below the median level, at most 25 % of the phase deeper than half the dip.
ECLIPSE_MIN_DEPTH_FRACTION = 0.7
ECLIPSE_MAX_WIDTH = 0.25
# ... and the eclipse-shape doubling (equal minima assumed, orbital period 2P) needs a dip of
# at least 5 % of the flux. Two equal eclipses require two stars of equal surface brightness,
# which eclipse each other by a sizeable fraction of the light unless the orbit is grazing
# or the light diluted; single shallow dips are what transiting planets produce (WASP-12 b,
# 1.5 % deep in TESS, was reported at twice its 1.09-d period, HAT-P-7 b likewise). Shallower
# single dips keep P, with 2P as the alternative.
ECLIPSE_MIN_DOUBLING_DEPTH = 0.05
# With the host star's radius known (TESS: the TIC RADIUS of the SPOC file header), a single
# dip is planet-like up to the depth of a 2-R_Jup body (the largest transiting planets, e.g.
# HAT-P-67 b) transiting it, at most PLANET_MAX_DEPTH: giant planets transiting M dwarfs are
# 7-11 % deep (TOI-5205 b, TOI-519 b: 0.36-0.40 R_sun hosts), deeper than the 5 % above.
PLANET_MAX_RADIUS_RSUN = 2.0 * 0.10276  # 2 R_Jup (IAU 2015 nominal equatorial R_Jup = 0.10276 R_sun)
PLANET_MAX_DEPTH = 0.25
# A second dip (e.g. a shallow secondary eclipse) counts when deeper than this fraction of
# the primary (and 5 x the noise of a bin median).
DIP_MIN_FRACTION = 0.03
# Noise of a bin median of the folded profile (eclipse_shape): the formal error, the scatter of
# the epochs about the binned profile and -- only when the profile resolves features, i.e. with
# at least this many bins -- the bin-to-bin differences of the profile (spots, flares). With the
# 10 bins of a sparse (Gaia) light curve the differences measure the eclipses themselves (0.04-0.05
# mag vs 0.002 mag formal), which hid 0.09-mag secondary eclipses (review: Gaia DR3
# 1974933547345251968 adopted at twice its 0.7044-d period).
ECLIPSE_EMPIRICAL_MIN_BINS = 30
# Explicit secondary-eclipse check (eclipse_shape): epochs within +-SECONDARY_WINDOW in phase of
# the point opposite the primary eclipse, fainter than the neighbouring phases by >= 5 sigma.
SECONDARY_WINDOW = 0.1
SECONDARY_SIGMA = 5.0
# symmetry_index below this: symmetric light curve (P vs 2P undecidable).
SYMMETRY_MAX_INDEX = 0.05
# A peak at an annual/semi-annual (ground) or Gaia scanning-law frequency is kept only with at
# least this semi-amplitude (mag), besides FAP < PERIODIC_EVIDENCE_FAP and phase coherence
# (see _real_at_window_frequency).
WINDOW_ALIAS_MIN_AMPLITUDE_MAG = 0.05
# alias_comb: competing peaks with at least this fraction of the best power.
ALIAS_COMB_POWER_RATIO = 0.9
# period_significance: epochs closer than this fraction of a cycle form one "visit"
# (sampling_visits); the effective-N correction needs at least MIN_VISITS_EFFECTIVE_N
# visits of at most MAX_MEAN_VISIT_SIZE epochs on average (dense sampling -- TESS sectors,
# a few runs of thousands of cadences -- is not corrected).
DENSE_SAMPLING_PHASE_STEP = 0.02
MIN_VISITS_EFFECTIVE_N = 8
MAX_MEAN_VISIT_SIZE = 5.0
# Cross-band confirmation (search_periods): another decisive band of the same survey with
# at least this fraction of the epochs must show the adopted period at single-frequency
# p < CONFIRMATION_P.
CONFIRMATION_MIN_FRACTION = 0.3
CONFIRMATION_P = 1e-3
# Without such a band (Gaia G, TESS, a single ZTF filter) the evidence must be stronger: FAP
# below 0.1 % instead of 1 %, besides phase coherence. A red-noise peak of the Gaia vari_agn
# source 6378923189372478592 (6.68 d, FAP 0.006) passed the (corrected) coherence test.
UNCONFIRMED_PERIOD_FAP = 1e-3
LS_CHUNK_FREQUENCIES = 32_768  # frequencies per periodogram evaluation (bounds memory; see ls_power)
# Series longer than this are period-searched on a time-binned copy (_period_search_arrays).
PERIOD_SEARCH_MAX_POINTS = 20_000
# Wall-clock budget of one period search (s; search_periods), and the number of analyses that
# run at once (their own executor: a light-curve request's analysis never occupies the default
# executor that the rest of the application uses).
PERIOD_SEARCH_BUDGET_S = float(os.getenv("TIMEDOMAIN_PERIOD_SEARCH_BUDGET_S", "60"))
ANALYSIS_WORKERS = max(1, int(os.getenv("TIMEDOMAIN_ANALYSIS_WORKERS", "2")))
_analysis_executor = ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS, thread_name_prefix="timedomain-analysis")
# Aliased red noise (red_noise_fap): window peaks with at least this normalised power carry the
# low-frequency power within RED_NOISE_ALIAS_MAX_OFFSET (c/d; 20-d time scales) to their sides.
RED_NOISE_ALIAS_MAX_OFFSET = 0.05
RED_NOISE_ALIAS_MIN_WINDOW = 0.05
# Contiguous time blocks of the cluster-robust frequency error (block_frequency_error): the
# largest error of 16, 8 and 4 blocks is used.
FREQUENCY_ERROR_BLOCKS = 16

# SkyBoT limits (IMCCE SkyBoT conesearch API documentation).
SKYBOT_MIN_JD = 2411320.0
SKYBOT_MAX_JD = 2473540.0
SKYBOT_MAX_RADIUS_ARCSEC = 36000.0
DEFAULT_SKYBOT_RADIUS_ARCSEC = 600.0
# Upper bound accepted for SkyBoT's ephemeris-uncertainty filter (-filter, arcsec): the
# widest cone radius; larger values filter nothing more.
SKYBOT_MAX_POSITION_ERROR_ARCSEC = SKYBOT_MAX_RADIUS_ARCSEC

RETRY_PAUSE_SECONDS = float(os.getenv("TIMEDOMAIN_RETRY_PAUSE_SECONDS", "2.0"))

MJD_JD_OFFSET = 2400000.5  # JD = MJD + 2400000.5 (definition of MJD)
TIME_SYSTEM = "BMJD_TDB"



# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TimeDomainError(AstroSearchError):
    """Base class for time-domain failures."""


class UpstreamServiceError(TimeDomainError):
    """An upstream archive failed, timed out, or returned an unusable answer."""

    def __init__(self, service: str, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LightCurvePoint:
    """One photometric epoch. ``mjd`` is BMJD_TDB; ``flag`` 0 means good quality."""

    mjd: float
    value: float
    error: float | None
    flag: int = 0
    n: int | None = None  # exposures averaged into this point (binned series)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mjd": self.mjd, "value": self.value, "error": self.error, "flag": self.flag}
        if self.n is not None:
            out["n"] = self.n
        return out


@dataclass(slots=True)
class LightCurveSeries:
    """A single-band light curve from one survey."""

    survey: str
    band: str
    unit: Literal["mag", "flux"]
    points: list[LightCurvePoint]
    photometric_system: str
    source_ids: list[str] = field(default_factory=list)
    n_total: int = 0
    n_rejected: int = 0
    time_system: str = TIME_SYSTEM
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.survey}:{self.band}"

    def good_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, y, dy) of good-quality points with finite, positive errors, time-sorted."""
        good = [p for p in self.points if p.flag == 0 and p.error is not None
                and math.isfinite(p.value) and math.isfinite(p.error) and p.error > 0]
        good.sort(key=lambda p: p.mjd)
        t = np.array([p.mjd for p in good], dtype=float)
        y = np.array([p.value for p in good], dtype=float)
        dy = np.array([p.error for p in good], dtype=float)
        return t, y, dy

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "survey": self.survey,
            "band": self.band,
            "unit": self.unit,
            "time_system": self.time_system,
            "photometric_system": self.photometric_system,
            "source_ids": list(self.source_ids),
            "n_total": self.n_total,
            "n_rejected": self.n_rejected,
            "metadata": dict(self.metadata),
            "points": [p.as_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightCurveSeries:
        return cls(
            survey=data["survey"], band=data["band"], unit=data["unit"],
            points=[LightCurvePoint(p["mjd"], p["value"], p.get("error"), int(p.get("flag", 0)), p.get("n"))
                    for p in data["points"]],
            photometric_system=data["photometric_system"], source_ids=list(data.get("source_ids", [])),
            n_total=int(data.get("n_total", 0)), n_rejected=int(data.get("n_rejected", 0)),
            time_system=data.get("time_system", TIME_SYSTEM), metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SurveyResult:
    """Series fetched from one survey plus notes and provenance of the upstream calls."""

    survey: str
    series: list[LightCurveSeries]
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"survey": self.survey, "series": [s.as_dict() for s in self.series],
                "notes": list(self.notes), "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyResult:
        return cls(data["survey"], [LightCurveSeries.from_dict(s) for s in data["series"]],
                   list(data.get("notes", [])), dict(data.get("provenance", {})))


@dataclass(slots=True)
class VariabilityMetrics:
    """Variability statistics of one series (see :func:`variability_metrics`)."""

    n: int
    unit: str
    time_span_days: float | None = None
    error_floor: float = 0.0
    weighted_mean: float | None = None
    weighted_mean_error: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    mad_std: float | None = None
    median_error: float | None = None
    chi2: float | None = None
    dof: int | None = None
    chi2_dof: float | None = None
    chi2_pvalue: float | None = None
    significance_sigma: float | None = None
    amplitude_5_95: float | None = None
    von_neumann_eta: float | None = None
    stetson_j: float | None = None
    stetson_n_pairs: int = 0
    excess_variance: float | None = None
    excess_variance_error: float | None = None
    fractional_variability: float | None = None
    intrinsic_scatter: float | None = None
    noise_rms: float | None = None
    error_scale: float = 1.0
    n_outliers_excluded: int = 0
    is_variable: bool | None = None
    decision_basis: str | None = None
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return _finite(asdict(self))


@dataclass(slots=True)
class PeriodogramResult:
    """Best Lomb-Scargle peak of one series (or of the multi-band combination)."""

    series: str
    n: int
    baseline_days: float
    best_period_days: float
    best_frequency_per_day: float
    power: float
    false_alarm_probability: float | None
    min_frequency: float
    max_frequency: float
    n_frequencies: int
    oversampling: float
    method: str
    peaks: list[dict[str, float]] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    reliable: bool = False
    warnings: list[str] = field(default_factory=list)
    # Refined period (multi-harmonic least squares around the grid peak) and its 1-sigma
    # uncertainty (Montgomery & O'Donoghue 1999); None when not refined.
    period_error_days: float | None = None
    n_harmonics: int | None = None
    # Period of the single-harmonic Lomb-Scargle peak before any doubling.
    lomb_scargle_period_days: float | None = None
    # Period ladder: {"doubled", "doubled_by", "ladder": [...], "shape": {...}, "delta_bic", ...}.
    harmonic_test: dict[str, Any] | None = None
    alternative_period_days: float | None = None
    # Baluev (2008) white-noise FAP of the single-harmonic Lomb-Scargle peak. The
    # decisive ``false_alarm_probability`` is the multi-harmonic FAP of the adopted period
    # against the local (red-)noise continuum (see red_noise_fap).
    lomb_scargle_false_alarm_probability: float | None = None
    # Residual noise power at the adopted frequency relative to white noise (1 = white,
    # >> 1 = red noise) and the local spectral slope of the residuals (red_noise_fap).
    noise_continuum: float | None = None
    noise_spectral_slope: float | None = None
    # Effective number of independent points of the residuals (sparse sampling only).
    effective_n: float | None = None
    # Variance inflation of the signal's coefficients by correlated residuals (>= 1; see
    # period_significance): applied to period_error_days (sqrt) and to the coherence test.
    error_inflation: float | None = None
    # Outlier-robust (rank-based) FAP of the adopted signal (rank_fap); must also be < 1 %.
    robust_false_alarm_probability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return _finite(asdict(self))


@dataclass(slots=True)
class PeriodSearch:
    """Per-series periodograms, the multi-band periodogram and the adopted period."""

    per_series: list[PeriodogramResult]
    multiband: PeriodogramResult | None
    best: PeriodogramResult | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"per_series": [p.as_dict() for p in self.per_series],
                "multiband": self.multiband.as_dict() if self.multiband else None,
                "best": self.best.as_dict() if self.best else None,
                "notes": list(self.notes)}


@dataclass(slots=True)
class LightCurveResult:
    """Everything the /lightcurves endpoint returns."""

    target: dict[str, Any]
    series: list[LightCurveSeries]
    variability: dict[str, VariabilityMetrics]
    period_search: PeriodSearch | None
    failures: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def period_dict(self) -> dict[str, Any] | None:
        best = self.period_search.best if self.period_search else None
        if best is None:
            return None
        out = best.as_dict()
        out["series"] = best.series
        return out

    def summary_is_variable(self) -> bool | None:
        """True if any series is variable, False if at least one series was decided and
        none is variable, None when no series could be decided."""
        decided = [m.is_variable for m in self.variability.values() if m.is_variable is not None]
        if not decided:
            return None
        return any(decided)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready dict; non-finite floats become None so strict JSON (RFC 8259) works."""
        variable = [k for k, m in self.variability.items() if m.is_variable]
        return _finite({
            "target": dict(self.target),
            "series": [s.as_dict() for s in self.series],
            "variability": {
                "per_series": {k: m.as_dict() for k, m in self.variability.items()},
                "summary": {"is_variable": self.summary_is_variable(),
                            "variable_series": variable,
                            "criteria": VARIABILITY_CRITERIA},
            },
            "period": self.period_dict(),
            "period_search": self.period_search.as_dict() if self.period_search else None,
            "failures": list(self.failures),
            "notes": list(self.notes),
            "provenance": dict(self.provenance),
        })


@dataclass(slots=True)
class SolarSystemObject:
    """A known solar-system body inside a SkyBoT cone at the requested epoch."""

    name: str
    number: int | None
    type: str
    object_class: str
    ra: float
    dec: float
    separation_arcsec: float
    v_mag: float | None
    position_error_arcsec: float | None
    ra_rate_arcsec_per_hour: float | None
    dec_rate_arcsec_per_hour: float | None
    geocentric_distance_au: float | None
    heliocentric_distance_au: float | None
    phase_angle_deg: float | None
    solar_elongation_deg: float | None
    ssocard_url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SolarSystemResult:
    target: dict[str, Any]
    epoch_mjd: float
    epoch_jd: float
    radius_arcsec: float
    observer: str
    objects: list[SolarSystemObject]
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"target": dict(self.target), "epoch_mjd": self.epoch_mjd, "epoch_jd": self.epoch_jd,
                "radius_arcsec": self.radius_arcsec, "observer": self.observer,
                "n_objects": len(self.objects), "objects": [o.as_dict() for o in self.objects],
                "notes": list(self.notes), "provenance": dict(self.provenance)}


@dataclass(slots=True)
class EphemerisPoint:
    """One JPL Horizons astrometric (ICRF) position."""

    target: str
    epoch_mjd: float
    ra: float
    dec: float
    v_mag: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


VARIABILITY_CRITERIA = (
    f"variable when N >= {MIN_POINTS_VARIABILITY} and, with per-epoch errors from the calibrated survey error model "
    f"and at most {OUTLIER_MAX_FRACTION:.0%} of N isolated >{OUTLIER_SIGMA:g}-sigma outliers (no same-direction "
    f">{OUTLIER_COINCIDENCE_SIGMA:g}-sigma deviation in the adjacent epochs or in any band of the survey within "
    f"{OUTLIER_COINCIDENCE_DAYS * 24:g} h) removed, chi^2 vs a constant rejects constancy at >= 5 sigma (p < {FIVE_SIGMA_PVALUE:.3g}) with "
    f"chi^2/dof >= {VARIABLE_CHI2_DOF:g} (intrinsic variance >= noise variance); or when the series has a reliable "
    f"periodogram peak with FAP < {PERIODIC_EVIDENCE_FAP:g} confirmed by another band, and constancy rejected at "
    ">= 5 sigma"
)


def _finite(value: Any) -> Any:
    """Recursively replace non-finite floats (nan, +-inf) by None (strict-JSON safe)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Time Conversion
# ---------------------------------------------------------------------------


# The conversions below call the ERFA ufuncs directly (``erfa.ufunc``), which return a
# status code instead of emitting Python warnings and use ERFA's built-in leap-second
# table. They therefore never touch process-global state: no ``warnings`` filters, no
# astropy IERS configuration, no download attempts. (An earlier version wrapped astropy
# ``Time`` in ``warnings.catch_warnings()`` / ``iers.conf.set_temp()``; both mutate global
# state and are not thread-safe, and the parsers run in worker threads.)


def utc_mjd_to_bmjd_tdb(mjd_utc: Sequence[float] | np.ndarray, ra: float, dec: float) -> np.ndarray:
    """Convert geocentric UTC MJDs to barycentric BMJD_TDB toward (ra, dec).

    UTC -> TAI -> TT with ERFA's leap-second table (``eraUtctai``, ``eraTaitt``); TT ->
    TDB with the Fairhead & Bretagnon (1990) series at the geocentre (``eraDtdb``); the
    Roemer delay is the projection of the Earth's barycentric position (``eraEpv00``,
    the ephemeris astropy calls "builtin") on the source direction, divided by c
    (Eastman et al. 2010, PASP 122, 935, eq. 1 without the sub-ms Shapiro/Einstein terms).
    This reproduces astropy's ``Time.light_travel_time(kind='barycentric')`` for a
    geocentric observer (checked in the tests). The topocentric term is <= 21 ms
    (Earth radius / c), negligible for survey photometry.
    """
    values = np.asarray(mjd_utc, dtype=float)
    if values.size == 0:
        return values.copy()
    tai1, tai2, _ = erfa.ufunc.utctai(MJD_JD_OFFSET, values)
    tt1, tt2, _ = erfa.ufunc.taitt(tai1, tai2)
    tdb_minus_tt = erfa.ufunc.dtdb(tt1, tt2, 0.0, 0.0, 0.0, 0.0)  # seconds; geocentre (u = v = 0)
    tdb2 = tt2 + tdb_minus_tt / erfa.DAYSEC
    _pvh, pvb, _ = erfa.ufunc.epv00(tt1, tdb2)  # Earth barycentric position (au) at TDB
    ra_r, dec_r = math.radians(ra), math.radians(dec)
    direction = np.array([math.cos(dec_r) * math.cos(ra_r), math.cos(dec_r) * math.sin(ra_r), math.sin(dec_r)])
    roemer_days = (np.asarray(pvb["p"]) @ direction) * erfa.AULT / erfa.DAYSEC
    return np.asarray((tt1 - MJD_JD_OFFSET) + tdb2 + roemer_days, dtype=float)


def gaia_time_to_bmjd_tdb(transit_time: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert Gaia DR3 epoch-photometry times to BMJD_TDB.

    ``g_transit_time`` / ``bp_obs_time`` / ``rp_obs_time`` are barycentric JD in TCB
    minus 2455197.5 (Gaia DR3 data model, table ``epoch_photometry``). TCB -> TDB is the
    linear IAU 2006 Resolution B3 relation (``eraTcbtdb``).
    """
    values = np.asarray(transit_time, dtype=float)
    if values.size == 0:
        return values.copy()
    tdb1, tdb2, _ = erfa.ufunc.tcbtdb(2455197.5, values)
    return np.asarray((tdb1 - MJD_JD_OFFSET) + tdb2, dtype=float)


def jyear_from_mjd(mjd: float) -> float:
    """Julian epoch of an MJD (J2000.0 = MJD 51544.5; Julian year = 365.25 d)."""
    return 2000.0 + (float(mjd) - 51544.5) / DAYS_PER_JULIAN_YEAR


def position_at(target: Target | None, jyear: float) -> tuple[float, float]:
    """``target`` position at Julian epoch ``jyear`` (linear proper motion, see
    :func:`models.propagate_radec`); unchanged when epoch or proper motion is unknown."""
    if target is None:
        raise ValueError("target is required")
    if target.epoch is None or target.proper_motion is None:
        return target.ra, target.dec
    return propagate_radec(target.ra, target.dec, target.pm_ra_masyr, target.pm_dec_masyr, target.epoch, jyear)


_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
UNIX_EPOCH_MJD = 40587.0  # 1970-01-01T00:00:00 UTC


def datetime_to_mjd(value: datetime) -> float:
    """UTC MJD of a timezone-aware datetime (POSIX days since 1970-01-01 + 40587; exact
    except during an inserted leap second, which POSIX time cannot represent)."""
    return UNIX_EPOCH_MJD + (value - _UNIX_EPOCH).total_seconds() / 86400.0


def now_mjd() -> float:
    return datetime_to_mjd(datetime.now(UTC))


# ---------------------------------------------------------------------------
# HTTP Helpers & Cache
# ---------------------------------------------------------------------------


@asynccontextmanager
async def client_scope(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """Yield ``client`` or a short-lived one when None."""
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=max(DEFAULT_TIMEOUTS.values()), follow_redirects=True) as owned:
        yield owned


def _error_detail(response: httpx.Response, limit: int = 300) -> str:
    _status, message = votable_query_status(response.content)
    if message:
        return message[:limit]
    ctype = response.headers.get("content-type", "")
    if "json" in ctype:
        try:
            data = response.json()
            if isinstance(data, dict):
                for key in ("message", "error", "detail", "msg"):
                    if data.get(key):
                        return str(data[key])[:limit]
        except ValueError:
            pass
    text = re.sub(r"<[^>]+>", " ", response.text or "")
    return re.sub(r"\s+", " ", text).strip()[:limit] or f"HTTP {response.status_code}"


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    service: str,
    timeout: float,
    params: dict[str, Any] | list[tuple[str, str]] | None = None,
    data: dict[str, Any] | None = None,
    ok_status: tuple[int, ...] = (200,),
) -> httpx.Response:
    """Send one request; map transport errors and HTTP errors to UpstreamServiceError.

    Connection failures and HTTP 429/5xx are retried once after a short pause (all
    requests here are idempotent queries); timeouts are not retried because the
    slow services would double the caller's wait.
    """
    for attempt in range(2):
        last = attempt == 1
        try:
            response = await client.request(method, url, params=params, data=data, timeout=timeout,
                                            headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise UpstreamServiceError(service, f"timed out after {timeout:.0f} s ({type(exc).__name__})", retryable=True) from exc
        except httpx.HTTPError as exc:
            if not last:
                await asyncio.sleep(RETRY_PAUSE_SECONDS)
                continue
            raise UpstreamServiceError(service, f"network error: {type(exc).__name__}: {exc}", retryable=True) from exc
        if response.status_code in ok_status:
            return response
        retryable = response.status_code >= 500 or response.status_code == 429
        if retryable and not last:
            await asyncio.sleep(RETRY_PAUSE_SECONDS)
            continue
        raise UpstreamServiceError(service, f"HTTP {response.status_code}: {_error_detail(response)}",
                                   status_code=response.status_code, retryable=retryable)
    raise AssertionError("unreachable")


# Messages of QUERY_STATUS=ERROR answers that report a server-side fault rather than a
# bad query. IRSA answers HTTP 200 with e.g. "TransientFault: INTERNAL_SERVER_ERROR:
# ORA-24459: OCISessionGet() timed out waiting for pool ..." (seen live, 2026-09) and
# the identical query succeeds minutes later.
_TRANSIENT_TAP_ERROR = re.compile(r"TransientFault|INTERNAL_SERVER_ERROR|ORA-\d+|timed? ?out|temporarily|"
                                  r"Service Unavailable|too many", re.IGNORECASE)


def _tap_error_check(response: httpx.Response, service: str) -> None:
    """TAP services may report ADQL errors inside an HTTP 200 VOTable.

    Server-side faults (see :data:`_TRANSIENT_TAP_ERROR`) are marked retryable.
    """
    if response.content[:200].lstrip().startswith(b"<"):
        status, message = votable_query_status(response.content)
        if status == "ERROR":
            transient = bool(_TRANSIENT_TAP_ERROR.search(message or ""))
            raise UpstreamServiceError(service, f"query error: {message}", status_code=response.status_code,
                                       retryable=transient)


async def _tap_query(client: httpx.AsyncClient, url: str, query: str, *, service: str, timeout: float) -> httpx.Response:
    """Synchronous TAP query (CSV); a transient fault reported inside HTTP 200 is retried once."""
    form = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    for attempt in range(2):
        response = await _request(client, "POST", url, service=service, timeout=timeout, data=form)
        try:
            _tap_error_check(response, service)
        except UpstreamServiceError as exc:
            if exc.retryable and attempt == 0:
                await asyncio.sleep(RETRY_PAUSE_SECONDS)
                continue
            raise
        return response
    raise AssertionError("unreachable")


def cache_ttl_seconds() -> int:
    raw = os.getenv("TIMEDOMAIN_CACHE_TTL_SECONDS", os.getenv("PROVIDER_CACHE_TTL_SECONDS", "3600"))
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 3600


_cache: CacheManager | None = None
# providers.CacheManager is not thread-safe (its local LRU is an OrderedDict): cache calls on a
# plain CacheManager run under this lock. _TimeDomainCache locks only its local LRU itself and
# does its Redis I/O outside any lock (redis-py clients are thread-safe).
_cache_lock = threading.Lock()
# Redis socket timeouts (s): an unreachable or stalled Redis must cost a request seconds,
# not the OS connect timeout, and the cache is only an optimisation.
REDIS_SOCKET_TIMEOUT_S = float(os.getenv("TIMEDOMAIN_REDIS_TIMEOUT_S", "2"))
# Circuit breaker: after a Redis connection error or timeout, Redis is skipped for this long (s)
# instead of costing every cache call the socket timeout.
REDIS_RETRY_AFTER_S = float(os.getenv("TIMEDOMAIN_REDIS_RETRY_AFTER_S", "30"))
# Cache I/O runs on its own small executor (never the default one the app shares), and a cache
# call is abandoned as a miss after this long (s).
CACHE_IO_WORKERS = 4
CACHE_IO_TIMEOUT_S = 2.0 * REDIS_SOCKET_TIMEOUT_S + 1.0
_cache_executor = ThreadPoolExecutor(max_workers=CACHE_IO_WORKERS, thread_name_prefix="timedomain-cache")
# Version of the cached payload: cached SurveyResults are *processed* (target identity, per-epoch
# track checks, PSF flags, zero-point offsets applied), so the key carries this version and it
# must be bumped whenever parsing, identity, flagging or the payload schema changes -- otherwise
# a deploy keeps serving the previous code's results for the whole TTL (TIMEDOMAIN_CACHE_TTL_SECONDS).
TIMEDOMAIN_CACHE_VERSION = "2026-09-r5"
CACHE_FORMAT = "timedomain.SurveyResult/json"


class _TimeDomainCache(CacheManager):
    """:class:`providers.CacheManager` for the time-domain module: a Redis client with connect/read
    timeouts and a circuit breaker, Redis I/O outside the lock, and a thread-safe local LRU."""

    def __init__(self, redis_url: str | None) -> None:
        super().__init__(None)
        self.redis_url = redis_url
        self._lock = threading.Lock()
        self._redis_down_until = 0.0
        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, socket_connect_timeout=REDIS_SOCKET_TIMEOUT_S,
                                             socket_timeout=REDIS_SOCKET_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - like CacheManager: no Redis means the local LRU only
                self._redis = None

    def _redis_client(self) -> Any:
        """The Redis client, or None while the circuit breaker is open (or without Redis)."""
        if self._redis is None or time.monotonic() < self._redis_down_until:
            return None
        return self._redis

    def _redis_failed(self) -> None:
        self._redis_down_until = time.monotonic() + REDIS_RETRY_AFTER_S

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._local_cache.get(key)
            if entry and entry[0] > time.monotonic():
                self._local_cache.move_to_end(key)
                return entry[1]
            self._drop(key)
        client = self._redis_client()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except Exception:  # noqa: BLE001 - connection errors and timeouts alike: skip Redis for a while
            self._redis_failed()
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return {"format": "undecodable"}

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if ttl <= 0:
            return
        text = json.dumps(value, default=str)
        with self._lock:
            self._drop(key)
            if len(text) <= self.max_bytes:
                self._local_cache[key] = (time.monotonic() + ttl, value, len(text))
                self._local_bytes += len(text)
            self._sweep()
        client = self._redis_client()
        if client is not None:
            try:
                client.setex(key, ttl, text)
            except Exception:  # noqa: BLE001
                self._redis_failed()

    def delete(self, key: str) -> None:
        with self._lock:
            self._drop(key)
        client = self._redis_client()
        if client is not None:
            try:
                client.delete(key)
            except Exception:  # noqa: BLE001
                self._redis_failed()


def get_cache() -> CacheManager:
    """Process-wide cache of parsed survey results (Redis when REDIS_URL is set)."""
    global _cache
    if _cache is None:
        _cache = _TimeDomainCache(os.getenv("REDIS_URL"))
    return _cache


def _cache_call(cache: CacheManager, method: str, *args: Any, **kwargs: Any) -> Any:
    if isinstance(cache, _TimeDomainCache):
        return getattr(cache, method)(*args, **kwargs)
    with _cache_lock:
        return getattr(cache, method)(*args, **kwargs)


def _cache_get(cache: CacheManager, key: str) -> SurveyResult | None:
    """Cached survey result (blocking: Redis I/O and JSON decoding; call in a worker thread).

    An entry that is not a decodable SurveyResult of the current format (an older schema, a
    truncated value) is a miss and is deleted, so a fresh fetch replaces it."""
    cached = _cache_call(cache, "get", key)
    if cached is None:
        return None
    try:
        if not isinstance(cached, dict) or cached.get("format") != CACHE_FORMAT or not isinstance(cached.get("content"), str):
            raise ValueError("not a time-domain SurveyResult entry")
        return SurveyResult.from_dict(json.loads(cached["content"]))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        _cache_call(cache, "delete", key)
        return None


def _cache_set(cache: CacheManager, key: str, result: SurveyResult, ttl: int) -> None:
    """Store a survey result (blocking: JSON encoding and Redis I/O; call in a worker thread)."""
    text = json.dumps(result.as_dict(), allow_nan=True)
    _cache_call(cache, "set", key, {"content": text, "format": CACHE_FORMAT}, ttl=ttl)


async def _cache_io(function: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking cache call on the cache executor; None when it takes longer than
    :data:`CACHE_IO_TIMEOUT_S` (the cache is an optimisation: a stalled one is a miss)."""
    future = asyncio.get_running_loop().run_in_executor(_cache_executor, function, *args)
    try:
        return await asyncio.wait_for(future, CACHE_IO_TIMEOUT_S)
    except TimeoutError:
        return None


def _csv_rows(text: str, *, required: Sequence[str] = (), service: str | None = None) -> list[dict[str, str]]:
    """Rows of a CSV answer.

    With ``service`` set the answer is validated: a TAP/light-curve CSV result always
    carries a header row, even with zero data rows, so an empty body, an HTML/XML page
    (e.g. a maintenance page served with HTTP 200) or a header lacking the ``required``
    columns is an upstream failure (:class:`UpstreamServiceError`, retryable), never
    "no source in the cone". Without ``service`` (local parsing) an empty text gives [].
    """
    text = text.lstrip("﻿")
    if not text.strip():
        if service is not None:
            raise UpstreamServiceError(service, "empty answer (no CSV header)", status_code=200, retryable=True)
        return []
    if service is not None and text.lstrip()[:1] == "<":
        snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text[:400])).strip()
        raise UpstreamServiceError(service, f"HTML/XML answer instead of CSV: {snippet[:200] or 'no text'}",
                                   status_code=200, retryable=True)
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [c for c in required if c not in header] if service is not None else []
    if missing and service is not None:
        raise UpstreamServiceError(service, f"CSV answer lacks column(s) {missing} (header: {header[:12]})",
                                   status_code=200, retryable=True)
    return list(reader)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"null", "nan", "none", "--"}:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
    return out if math.isfinite(out) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "t", "1", "yes"}


# ---------------------------------------------------------------------------
# ZTF (IRSA light-curve API)
# ---------------------------------------------------------------------------


_ZTF_COLLECTION = re.compile(r"^ztf_dr(\d{1,3})$")


def ztf_objects_adql(table: str, ra: float, dec: float, radius_arcsec: float) -> str:
    return (f"SELECT oid, ra, dec, filtercode, field, ccdid, qid, ngoodobs FROM {table} "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.8f})) = 1")


def _moving(target: Target | None) -> bool:
    return target is not None and target.epoch is not None and target.proper_motion is not None


def haversine_arcsec_array(ra1: Any, dec1: Any, ra2: Any, dec2: Any) -> np.ndarray:
    """Vectorised (broadcasting) :func:`models.haversine_arcsec`."""
    phi1, phi2 = np.radians(dec1), np.radians(dec2)
    dlmb = np.radians((np.asarray(ra2, dtype=float) - np.asarray(ra1, dtype=float) + 180.0) % 360.0 - 180.0)
    a = np.sin((phi2 - phi1) / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))) * 3600.0


def track_positions(target: Target, jyears: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`position_at`: the target's (ra, dec) at each Julian year.

    Same linear model as :func:`models.propagate_radec` (the motion is a straight line in
    the gnomonic tangent plane at the catalogue position, i.e. a great-circle arc).
    """
    years = np.asarray(jyears, dtype=float)
    if not _moving(target):
        return np.full(years.shape, target.ra), np.full(years.shape, target.dec)
    assert target.epoch is not None and target.pm_ra_masyr is not None and target.pm_dec_masyr is not None
    dt = years - target.epoch
    xi = np.radians(target.pm_ra_masyr * dt / 3.6e6)
    eta = np.radians(target.pm_dec_masyr * dt / 3.6e6)
    ra0, dec0 = math.radians(target.ra), math.radians(target.dec)
    denom = math.cos(dec0) - eta * math.sin(dec0)
    ra = np.degrees(ra0 + np.arctan2(xi, denom)) % 360.0
    dec = np.degrees(np.arctan2(math.sin(dec0) + eta * math.cos(dec0), np.hypot(xi, denom)))
    return ra, dec


def track_separations_arcsec(target: Target, obj_ra: Sequence[float] | np.ndarray, obj_dec: Sequence[float] | np.ndarray,
                             epoch_range: tuple[float, float]) -> np.ndarray:
    """Minimum distance (arcsec) of each object from the target's track over ``epoch_range``.

    The track is a great-circle arc (:func:`track_positions`); a gnomonic projection maps
    great circles to straight lines, so in the tangent plane at the track's midpoint the
    distance is the point-to-segment distance: O(N), and exact up to the gnomonic
    distortion (relative ~1e-5 at 10' from the midpoint).
    """
    ras, decs = np.asarray(obj_ra, dtype=float), np.asarray(obj_dec, dtype=float)
    ends_ra, ends_dec = track_positions(target, np.array(epoch_range, dtype=float))
    mid_ra, mid_dec = track_positions(target, np.array([0.5 * (epoch_range[0] + epoch_range[1])]))
    a0, d0 = math.radians(float(mid_ra[0])), math.radians(float(mid_dec[0]))

    def plane(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, d = np.radians(ra), np.radians(dec)
        cos_c = np.maximum(math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(a - a0), 1e-12)
        x = np.cos(d) * np.sin(a - a0) / cos_c
        y = (math.cos(d0) * np.sin(d) - math.sin(d0) * np.cos(d) * np.cos(a - a0)) / cos_c
        return np.degrees(x) * 3600.0, np.degrees(y) * 3600.0

    (x0, x1), (y0, y1) = plane(ends_ra, ends_dec)
    px, py = plane(ras, decs)
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    u_ = np.clip(((px - x0) * dx + (py - y0) * dy) / length2, 0.0, 1.0) if length2 > 0 else np.zeros_like(px)
    near_x, near_y = x0 + u_ * dx, y0 + u_ * dy
    dist = np.hypot(px - near_x, py - near_y)
    # Beyond ~1 deg the projection distorts; such objects are nowhere near a (< 2') track anyway.
    wide = haversine_arcsec_array(float(mid_ra[0]), float(mid_dec[0]), ras, decs) > 3600.0
    dist[wide] = np.maximum(dist[wide], 3600.0)
    return dist


def track_separation_arcsec(target: Target | None, ra: float, dec: float, obj_ra: float, obj_dec: float,
                            epoch_range: tuple[float, float] | None = None) -> float:
    """Separation of an object from the target (arcsec).

    For a target with epoch and proper motion and an ``epoch_range`` (Julian years), the
    minimum distance to the target's track over that range (:func:`track_separations_arcsec`);
    otherwise the distance to (ra, dec).
    """
    if target is None or epoch_range is None or not _moving(target):
        return haversine_arcsec(ra, dec, obj_ra, obj_dec)
    return float(track_separations_arcsec(target, [obj_ra], [obj_dec], epoch_range)[0])


def _ztf_band(row: dict[str, str]) -> str:
    return (row.get("filtercode") or "").strip().lower().removeprefix("z")


def _ztf_int(value: Any) -> int | None:
    """Integer of a ZTF field/ccdid/qid value (decimal in TAP answers, hex such as '0xe' in light-curve CSVs)."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        num = _num(text)
        return int(num) if num is not None else None


def _ztf_image_key(row: dict[str, str]) -> tuple[int | None, int | None, int | None, str] | None:
    """(field, ccdid, qid, band) of a ZTF object: one reference image, in which one star is one object."""
    key = (_ztf_int(row.get("field")), _ztf_int(row.get("ccdid")), _ztf_int(row.get("qid")), _ztf_band(row))
    return None if None in key[:3] else key


def select_ztf_objects(rows: Sequence[dict[str, str]], ra: float, dec: float, *, bands: Sequence[str] | None = None,
                       target: Target | None = None, match_arcsec: float = ZTF_MATCH_ARCSEC,
                       ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split ZTF object rows into (the target's candidate objects, neighbours).

    ZTF assigns one object ID per field, CCD quadrant and filter. An object is a
    candidate when it lies within ``match_arcsec`` of the target -- of (ra, dec), or, for a
    target with epoch and proper motion, of its track over
    :data:`ZTF_POSITION_EPOCH_RANGE` (object positions come from reference images of
    different epochs). Everything else in the cone is a neighbour, however few or many
    good epochs it has: the search radius never widens the identity tolerance, so a
    neighbour is never adopted in place of a target that ZTF did not measure (e.g. a
    saturated star, whose own objects have ``ngoodobs = 0``). Each returned row carries
    ``_sep`` (arcsec, to the target or its track).

    A star is one object per reference image, so for a stationary target only the
    nearest candidate of each (field, CCD, quadrant, filter) is kept (the others are
    neighbours, tagged ``_same_image_as`` = the kept object's ID). For a moving target a track match is necessary but not sufficient -- a
    fast star's track crosses unrelated stars -- so every candidate is kept here and
    identity is verified epoch by epoch on the light curves (:func:`parse_ztf_csv`).
    Vectorised: a crowded cone of hundreds of objects takes milliseconds.
    """
    usable: list[dict[str, str]] = []
    coords: list[tuple[float, float]] = []
    for r in rows:
        ora, odec = _num(r.get("ra")), _num(r.get("dec"))
        if ora is None or odec is None or (bands and _ztf_band(r) not in bands):
            continue
        usable.append(dict(r))
        coords.append((ora, odec))
    if not usable:
        return [], []
    obj_ra = np.array([c[0] for c in coords])
    obj_dec = np.array([c[1] for c in coords])
    if target is not None and _moving(target):
        seps = track_separations_arcsec(target, obj_ra, obj_dec, ZTF_POSITION_EPOCH_RANGE)
    else:
        seps = haversine_arcsec_array(ra, dec, obj_ra, obj_dec)
    out_same: list[dict[str, str]] = []
    out_other: list[dict[str, str]] = []
    for row, sep in zip(usable, seps):
        row["_sep"] = f"{float(sep):.3f}"
        (out_same if sep <= match_arcsec else out_other).append(row)
    out_same.sort(key=lambda r: float(r["_sep"]))
    out_other.sort(key=lambda r: float(r["_sep"]))
    if not _moving(target):
        seen: dict[tuple[int | None, int | None, int | None, str], str] = {}
        unique: list[dict[str, str]] = []
        for row in out_same:  # nearest first
            key = _ztf_image_key(row)
            if key is not None and key in seen:
                row["_same_image_as"] = seen[key]  # within the tolerance, but not the target's object
                out_other.append(row)
                continue
            if key is not None:
                seen[key] = str(row.get("oid", "")).strip()
            unique.append(row)
        out_same = unique
        out_other.sort(key=lambda r: float(r["_sep"]))
    return out_same, out_other


def _describe_neighbours(rows: Sequence[dict[str, str]], limit: int = 5) -> str:
    items = [f"{r.get('oid')} ({_ztf_band(r) or '?'}, {float(r['_sep']):.1f}\", ngoodobs {r.get('ngoodobs') or '?'})"
             for r in rows[:limit]]
    more = f" and {len(rows) - limit} more" if len(rows) > limit else ""
    return ", ".join(items) + more


def ztf_psf_outliers(chi: np.ndarray, sharp: np.ndarray, *, good: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of epochs whose PSF fit is an outlier among the object's epochs.

    IRSA light-curve rows carry the PSF-fit ``chi`` (reduced chi of the fit) and
    ``sharp`` (DAOPHOT-style sharpness) of each detection. An epoch whose fit is far worse
    than the object's typical fit -- ``chi`` above :data:`ZTF_PSF_CHI_FACTOR` x the median
    of the object's (``good``) epochs and above :data:`ZTF_PSF_CHI_MIN`, or ``sharp`` more
    than max(:data:`ZTF_PSF_SHARP_MIN`, :data:`ZTF_PSF_SHARP_MADS` x MAD) from the median --
    is a blend with a ghost/satellite trail/transient or a bad measurement, not the star
    (e.g. seven catflags = 0 epochs of the Landolt standard PG1323-086B, 0.5-1.7 mag faint
    with chi 2.3-9.1 and |sharp| up to 2.1). The criteria are relative to the object itself,
    so extended sources (sharp consistently > 0) and bright stars (larger chi at high S/N)
    keep their epochs. Calibration: see :data:`ZTF_PSF_CHI_FACTOR`.
    """
    chi = np.asarray(chi, dtype=float)
    sharp = np.asarray(sharp, dtype=float)
    base = np.ones(len(chi), dtype=bool) if good is None else np.asarray(good, dtype=bool)
    out = np.zeros(len(chi), dtype=bool)
    ok_chi = base & np.isfinite(chi) & (chi > 0)
    if np.sum(ok_chi) >= ZTF_PSF_MIN_EPOCHS:
        limit = max(ZTF_PSF_CHI_MIN, ZTF_PSF_CHI_FACTOR * float(np.median(chi[ok_chi])))
        out |= np.isfinite(chi) & (chi > limit)
    ok_sharp = base & np.isfinite(sharp)
    if np.sum(ok_sharp) >= ZTF_PSF_MIN_EPOCHS:
        centre = float(np.median(sharp[ok_sharp]))
        spread = float(stats.median_abs_deviation(sharp[ok_sharp], scale="normal"))
        out |= np.isfinite(sharp) & (np.abs(sharp - centre) > max(ZTF_PSF_SHARP_MIN, ZTF_PSF_SHARP_MADS * spread))
    return out


def _column(items: Sequence[dict[str, str]], name: str) -> np.ndarray:
    """Float column of CSV rows (NaN where missing or unparseable)."""
    return np.array([v if (v := _num(r.get(name))) is not None else np.nan for r in items], dtype=float)


def parse_ztf_csv(text: str, ra: float, dec: float, *, include_flagged: bool = False,
                  oids: Sequence[str] | None = None, match_arcsec: float = ZTF_MATCH_ARCSEC,
                  service: str | None = None, target: Target | None = None) -> SurveyResult:
    """Parse the ZTF light-curve CSV into per-band series of one astrophysical source.

    Rows are grouped by ZTF object ID (``oid``; one per field, CCD quadrant and
    filter). ``oids`` (from :func:`select_ztf_objects`) names the target's candidate
    objects; without it, objects whose mean light-curve positions lie within
    ``match_arcsec`` of (ra, dec) are the target's and all others are excluded
    neighbours.

    Moving targets (``target`` with epoch and proper motion): identity is verified epoch
    by epoch -- every light-curve row carries the position measured at that epoch, which
    must lie within ``match_arcsec`` (plus the target's parallax) of the target's position
    at that epoch (:func:`track_positions`). Rows failing it are dropped, and an object
    most of whose (good) epochs fail (:data:`ZTF_MIN_TRACK_FRACTION`) is not the target
    at all but a star on its path whose reference-image position the track merely crosses
    (e.g. the objects along Barnard's star's 80" track, 4-65" from the star at their
    epochs). ZTF creates new object IDs along a fast star's track, so one (field, CCD,
    quadrant, filter) can hold several objects of the moving target with disjoint epochs
    (GJ 3622 r: 367206400017938 holds its 75 good 2023-2025 epochs, 0.24" from the track);
    every object passing the per-epoch check is kept, and an exposure measured by two of
    them is used once (the measurement of the object with the most verified epochs). For a
    stationary target a star is one object per reference image: of several objects in one
    (field, CCD, quadrant, filter) only the nearest is kept.

    Epochs with ``catflags != 0`` are flagged (and dropped unless ``include_flagged``):
    only epochs without any processing flag bit set are used (bit definitions: ZTF
    Science Data System Explanatory Supplement, sec. 10.3, referenced by the IRSA
    light-curve API documentation). Epochs with ``catflags == 0`` whose PSF fit is an
    outlier (:func:`ztf_psf_outliers`) are flagged with :data:`ZTF_PSF_FLAG`.
    ``service`` validates an upstream answer (see :func:`_csv_rows`).
    """
    rows = _csv_rows(text, required=("oid", "mjd", "mag", "magerr", "catflags", "filtercode"), service=service)
    notes: list[str] = []
    by_oid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_oid.setdefault(str(row.get("oid", "")).strip(), []).append(row)
    if not by_oid:
        return SurveyResult("ztf", [], ["ZTF: no light-curve points for the target's object(s)."], {"rows": 0})
    centers: dict[str, tuple[float, float, float]] = {}
    for oid, items in by_oid.items():
        ras = [v for v in (_num(r.get("ra")) for r in items) if v is not None]
        decs = [v for v in (_num(r.get("dec")) for r in items) if v is not None]
        if not ras or not decs:
            continue
        mra, mdec = _circular_mean_deg(ras), float(np.mean(decs))
        centers[oid] = (mra, mdec, haversine_arcsec(ra, dec, mra, mdec))
    if oids is not None:
        wanted = {str(o).strip() for o in oids}
        kept = sorted(k for k in by_oid if k in wanted)
    else:
        if not centers:
            raise UpstreamServiceError("ztf", "light-curve rows carry no positions", status_code=200)
        kept = sorted(k for k, c in centers.items() if c[2] <= match_arcsec)
        if not kept:
            nearest = min(centers, key=lambda k: centers[k][2])
            return SurveyResult("ztf", [], [(f"ZTF: no object within {match_arcsec:g}\" of the target; the nearest "
                                             f"({nearest}) is {centers[nearest][2]:.1f}\" away and was not used.")],
                                {"rows": len(rows)})
    excluded = sorted(set(by_oid) - set(kept))
    if excluded:
        notes.append(f"ZTF: {len(excluded)} object ID(s) of neighbouring source(s) in the cone were excluded.")

    def good_rows(items: Sequence[dict[str, str]]) -> np.ndarray:
        return (_column(items, "catflags") == 0) & (_column(items, "magerr") > 0)

    moving = target is not None and _moving(target)
    on_track: dict[str, np.ndarray] = {}  # oid -> rows verified at their own epoch (moving targets)
    track_sep: dict[str, np.ndarray] = {}
    if moving and target is not None:
        tolerance = match_arcsec + (target.parallax_mas or 0.0) / 1000.0
        not_target: list[str] = []
        for oid in kept:
            items = by_oid[oid]
            mjd = _column(items, "mjd")
            tra, tdec = track_positions(target, 2000.0 + (mjd - 51544.5) / DAYS_PER_JULIAN_YEAR)
            with np.errstate(invalid="ignore"):
                sep = haversine_arcsec_array(tra, tdec, _column(items, "ra"), _column(items, "dec"))
                ok = np.isfinite(sep) & (sep <= tolerance)
            good = good_rows(items)
            basis = good if np.any(good) else np.ones(len(items), dtype=bool)
            fraction = float(np.mean(ok[basis]))
            if fraction < ZTF_MIN_TRACK_FRACTION:
                finite = sep[basis & np.isfinite(sep)]
                median = f"{float(np.median(finite)):.1f}\"" if len(finite) else "unknown"
                not_target.append(f"{oid} ({_ztf_band(items[0])}: {fraction:.0%} of its epochs, median distance {median})")
                continue
            on_track[oid] = ok
            track_sep[oid] = sep
            dropped = int(np.sum(~ok))
            if dropped:
                notes.append(f"ZTF: {dropped} epoch(s) of object {oid} lie more than {tolerance:.2g}\" from the "
                             "target's position at their epoch and were dropped.")
        if not_target:
            notes.append("ZTF: object(s) on the target's track are not the target (their positions at their own "
                         f"epochs do not follow it within {tolerance:.2g}\"): {', '.join(not_target)}.")
        kept = [oid for oid in kept if oid in on_track]
    if moving:
        # Several objects of one reference image can all be the moving target (new object IDs
        # along its track, disjoint epochs): all are kept, most verified good epochs first, and
        # an exposure that two of them measured is used once (below).
        kept.sort(key=lambda o: (-int(np.sum(on_track[o] & good_rows(by_oid[o]))),
                                 float(np.nanmedian(track_sep[o][on_track[o]])), o))
    else:
        # A (stationary) star is one object per reference image (field, CCD, quadrant, filter).
        by_image: dict[Any, list[str]] = {}
        for oid in kept:
            key = _ztf_image_key(by_oid[oid][0])
            by_image.setdefault(key if key is not None else ("oid", oid), []).append(oid)
        duplicates: list[str] = []
        for members in by_image.values():
            if len(members) < 2:
                continue
            best = min(members, key=lambda o: centers.get(o, (0.0, 0.0, math.inf))[2])
            duplicates.extend(o for o in members if o != best)
        if duplicates:
            kept = [oid for oid in kept if oid not in duplicates]
            notes.append(f"ZTF: {len(duplicates)} further object(s) in the same field/CCD/quadrant/filter as a target "
                         f"object were not used (one star is one object per reference image): "
                         f"{', '.join(sorted(duplicates))}.")
    if not kept:
        return SurveyResult("ztf", [], [*notes, ("ZTF: none of the candidate objects is the target: no ZTF light "
                                                "curve for the target.")], {"rows": len(rows)})
    located = [centers[k] for k in kept if k in centers]
    if located:
        nra = _circular_mean_deg([c[0] for c in located])
        ndec = float(np.mean([c[1] for c in located]))
    else:
        nra, ndec = ra, dec
    if moving:
        verified_seps = np.concatenate([track_sep[o][on_track[o]] for o in kept])
        nsep = float(np.median(verified_seps)) if len(verified_seps) else math.nan
    else:
        nsep = haversine_arcsec(ra, dec, nra, ndec)
    per_band: dict[str, list[tuple[float, float, float | None, int, str]]] = {}
    psf_flagged = 0
    exposures: set[tuple[Any, str, int]] = set()  # (image, band, exposure start in s): moving targets
    repeated = 0
    for oid in kept:
        items = by_oid[oid]
        verified = on_track.get(oid)
        good = good_rows(items) if verified is None else good_rows(items) & verified
        psf_bad = ztf_psf_outliers(_column(items, "chi"), _column(items, "sharp"), good=good)
        for idx, row in enumerate(items):
            if verified is not None and not verified[idx]:
                continue
            band = (row.get("filtercode") or "").strip().lower().removeprefix("z")
            mjd, mag, err = _num(row.get("mjd")), _num(row.get("mag")), _num(row.get("magerr"))
            if verified is not None and mjd is not None:
                image = _ztf_image_key(row) or _ztf_image_key(items[0]) or ("oid", oid)
                exposure = (image, band, round(mjd * 86400.0))
                if exposure in exposures:
                    repeated += 1
                    continue
                exposures.add(exposure)
            exptime = _num(row.get("exptime"))
            if mjd is not None and exptime is not None:
                # 'mjd' marks the exposure start: IRSA's 'hjd' matches it only after
                # adding half the exposure (verified on the recorded rows, see tests).
                mjd += 0.5 * exptime / 86400.0
            flags = _num(row.get("catflags"))
            if band not in ("g", "r", "i") or mjd is None or mag is None:
                continue
            flag = int(flags) if flags is not None else -1
            if flag == 0 and psf_bad[idx]:
                flag = ZTF_PSF_FLAG
                psf_flagged += 1
            per_band.setdefault(band, []).append((mjd, mag, err, flag, oid))
    if repeated:
        notes.append(f"ZTF: {repeated} exposure(s) measured by two object IDs of the target in the same "
                     "field/CCD/quadrant/filter were used once (the object with the most verified epochs).")
    if psf_flagged:
        notes.append(f"ZTF: {psf_flagged} epoch(s) with catflags = 0 but an outlying PSF fit (chi or sharp, relative "
                     f"to the object's other epochs) were flagged ({ZTF_PSF_FLAG}) and not used.")
    series: list[LightCurveSeries] = []
    for band in ("g", "r", "i"):
        items = per_band.get(band)
        if not items:
            continue
        bmjd = utc_mjd_to_bmjd_tdb([i[0] for i in items], nra, ndec)
        offsets = ztf_field_offsets([(float(t), mag, oid) for (_m, mag, err, flag, oid), t in zip(items, bmjd)
                                     if flag == 0 and err is not None and err > 0])
        points: list[LightCurvePoint] = []
        rejected = 0
        for (mjd, mag, err, flag, oid), t in zip(items, bmjd):
            bad = flag != 0 or err is None or err <= 0
            if bad:
                rejected += 1
                if not include_flagged:
                    continue
            points.append(LightCurvePoint(float(t), mag + offsets.get(oid, 0.0), err,
                                          flag if flag != 0 else (0 if err and err > 0 else 1)))
        points.sort(key=lambda p: p.mjd)
        scale, floor = ERROR_MODELS[("ztf", band)]
        metadata: dict[str, Any] = {"source_ra": nra, "source_dec": ndec, "source_separation_arcsec": round(nsep, 3),
                                    "quality_cut": ZTF_QUALITY_CUT, "error_scale": scale, "error_floor_mag": floor}
        if moving:
            metadata["source_separation_reference"] = ("median distance of the epochs from the target's position at "
                                                       "their epoch")
        if offsets:
            metadata["field_offsets_mag"] = {k: round(v, 5) for k, v in offsets.items()}
            notes.append(f"ZTF {band}: {len(offsets)} field/CCD object ID(s) shifted to the zero point of the "
                         f"best-sampled ID (offsets {', '.join(f'{v:+.3f}' for v in offsets.values())} mag).")
        series.append(LightCurveSeries(
            survey="ztf", band=band, unit="mag", points=points,
            photometric_system="AB (ZTF PSF-fit, PS1-calibrated)",
            source_ids=sorted({i[4] for i in items}), n_total=len(items), n_rejected=rejected, metadata=metadata,
        ))
    return SurveyResult("ztf", series, notes, {"rows": len(rows)})


def ztf_field_offsets(points: Sequence[tuple[float, float, str]], *, min_epochs: int = ZTF_ALIGN_MIN_EPOCHS,
                      min_overlap: float = ZTF_ALIGN_MIN_OVERLAP,
                      max_standard_error: float = ZTF_ALIGN_MAX_STANDARD_ERROR) -> dict[str, float]:
    """Zero-point offsets (mag) that bring each ZTF object ID onto the best-sampled one.

    ``points`` are good-quality (t, mag, oid) of one band. A star observed in several
    fields/CCD quadrants has one oid per field, each calibrated separately; their
    medians of the same star differ by up to a few 0.01 mag (0.024 mag measured live
    for the constant Stripe 82 standard at 41.1558 +0.8937), which masquerades as
    variability. An oid is aligned (offset = median(primary) - median(oid)) only when
    it has >= ``min_epochs`` good epochs and its time span overlaps the primary's by
    at least ``min_overlap`` of its own span, so both medians average over the same
    epochs of any intrinsic variability, and only when the offset is measurable: the
    standard error of the median difference (sqrt(pi/2) x MAD-sigma / sqrt(n) for each
    oid) must not exceed ``max_standard_error`` -- for large-amplitude variables a
    sparse oid's median is dominated by phase sampling, not calibration.
    """
    by_oid: dict[str, list[tuple[float, float]]] = {}
    for t, mag, oid in points:
        by_oid.setdefault(oid, []).append((t, mag))
    if len(by_oid) < 2:
        return {}
    def median_and_se(mags: np.ndarray) -> tuple[float, float]:
        # Standard error of the median: sqrt(pi/2) sigma / sqrt(n), sigma from the MAD.
        return float(np.median(mags)), math.sqrt(math.pi / 2.0) * float(
            stats.median_abs_deviation(mags, scale="normal")) / math.sqrt(len(mags))

    primary = max(by_oid, key=lambda k: (len(by_oid[k]), k))
    pt = np.array([a for a, _ in by_oid[primary]])
    pmed, pse = median_and_se(np.array([b for _, b in by_oid[primary]]))
    offsets: dict[str, float] = {}
    for oid, items in by_oid.items():
        if oid == primary or len(items) < min_epochs:
            continue
        t = np.array([a for a, _ in items])
        span = float(t.max() - t.min())
        overlap = min(t.max(), pt.max()) - max(t.min(), pt.min())
        if span <= 0 or overlap < min_overlap * span:
            continue
        omed, ose = median_and_se(np.array([b for _, b in items]))
        if math.hypot(pse, ose) > max_standard_error:
            continue  # offset not measurable against the source's own variability
        offsets[oid] = pmed - omed
    return offsets


def _circular_mean_deg(values: Sequence[float]) -> float:
    rad = np.radians(np.asarray(values, dtype=float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0)


async def latest_ztf_objects_table(client: httpx.AsyncClient, *, timeout: float) -> str:
    """Name of the newest ``ztf_objects_drNN`` table published in the IRSA TAP schema."""
    query = "SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE 'ztf_objects_dr%'"
    response = await _tap_query(client, IRSA_TAP_SYNC_URL, query, service="ztf", timeout=timeout)
    tables = []
    for row in _csv_rows(response.text, required=("table_name",), service="ztf"):
        match = re.fullmatch(r"ztf_objects_dr(\d+)", str(row.get("table_name", "")).strip())
        if match:
            tables.append((int(match.group(1)), match.group(0)))
    if not tables:
        raise UpstreamServiceError("ztf", "IRSA TAP lists no ztf_objects_drNN table", status_code=response.status_code)
    return max(tables)[1]


async def fetch_ztf(
    client: httpx.AsyncClient, ra: float, dec: float, radius_arcsec: float, *,
    bands: Sequence[str] | None = None, collection: str | None = None, include_flagged: bool = False,
    timeout: float | None = None, target: Target | None = None, match_arcsec: float = ZTF_MATCH_ARCSEC,
) -> SurveyResult:
    """ZTF g/r/i light curves: object IDs by cone search, then light curves by ID.

    The IRSA light-curve API's own cone search (``POS=CIRCLE``) took 170-210 s per
    position when tested live, whereas a TAP cone on ``ztf_objects_drNN`` followed by
    an ``ID=``-based light-curve request returns the identical rows in ~5-60 s. The
    newest objects table is discovered from TAP_SCHEMA unless ``collection``
    (``ztf_drNN``) pins a data release; the matching ``COLLECTION`` is passed to the
    light-curve API so objects and photometry come from the same release.

    Identity: only objects within ``match_arcsec`` of the target (of its track, for a
    ``target`` with epoch and proper motion; see :func:`select_ztf_objects`) are candidates.
    When they have no good epochs (``ngoodobs = 0``, typical of saturated stars) no
    series is returned -- neighbours are listed in a note, never substituted. For a
    moving target each candidate's epochs must also follow the target's motion
    (:func:`parse_ztf_csv`): stars that its track merely crosses are rejected.
    """
    if not 0 < radius_arcsec / 3600.0 <= ZTF_MAX_RADIUS_DEG:
        raise InvalidCoordinateError(f"ZTF radius must be in (0, {ZTF_MAX_RADIUS_DEG * 3600:.0f}] arcsec.")
    if bands:
        bad = [b for b in bands if b not in ("g", "r", "i")]
        if bad:
            raise ValueError(f"unknown ZTF band(s): {bad}; valid: g, r, i")
    tout = timeout or DEFAULT_TIMEOUTS["ztf"]
    started = time.perf_counter()
    if collection:
        match = _ZTF_COLLECTION.match(collection)
        if not match:
            raise ValueError("ztf collection must look like 'ztf_dr23'")
        table = f"ztf_objects_dr{match.group(1)}"
    else:
        table = await latest_ztf_objects_table(client, timeout=tout)
        collection = "ztf_dr" + table.rsplit("dr", 1)[1]
    query = ztf_objects_adql(table, ra, dec, radius_arcsec)
    response = await _tap_query(client, IRSA_TAP_SYNC_URL, query, service="ztf", timeout=tout)
    tolerance = min(match_arcsec, radius_arcsec)
    moving = _moving(target)
    tra, tdec = (target.ra, target.dec) if moving and target is not None else (ra, dec)

    def select() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        rows = _csv_rows(response.text, required=("oid", "ra", "dec", "filtercode", "ngoodobs"), service="ztf")
        return (rows, *select_ztf_objects(rows, tra, tdec, bands=bands, target=target if moving else None,
                                          match_arcsec=tolerance))

    # Parsing and identity checks of a crowded cone (hundreds of objects) run in a worker thread.
    objects, same, neighbours = await asyncio.to_thread(select)
    provenance: dict[str, Any] = {"objects_service": IRSA_TAP_SYNC_URL, "objects_table": table, "objects_query": query,
                                  "collection": collection, "n_objects_in_cone": len(objects),
                                  "match_radius_arcsec": tolerance,
                                  "target_object_ids": [str(r["oid"]).strip() for r in same],
                                  "neighbour_object_ids": {str(r["oid"]).strip(): float(r["_sep"]) for r in neighbours}}
    first, last = ZTF_POSITION_EPOCH_RANGE
    where = f"of the target's track (J{first:.1f}-J{last:.1f})" if moving else "of the target"
    notes: list[str] = []
    beyond = [r for r in neighbours if "_same_image_as" not in r]
    same_image = [r for r in neighbours if "_same_image_as" in r]
    if beyond:
        notes.append(f"ZTF: {len(beyond)} object ID(s) of neighbouring source(s) in the cone were excluded "
                     f"(more than {tolerance:g}\" {where}): {_describe_neighbours(beyond)}.")
    if same_image:
        provenance["same_image_object_ids"] = {str(r["oid"]).strip(): r["_same_image_as"] for r in same_image}
        notes.append(f"ZTF: {len(same_image)} object ID(s) within {tolerance:g}\" of the target were not used: "
                     "same field/CCD/quadrant/filter as the target's (nearer) object, and one star is one object "
                     f"per reference image: {_describe_neighbours(same_image)}.")
    with_data = [r for r in same if (_num(r.get("ngoodobs")) or 0) > 0 or r.get("ngoodobs") in (None, "")]
    if not same:
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        return SurveyResult("ztf", [], [*notes, (f"ZTF: no object in {table} within {tolerance:g}\" {where}; "
                                                 "no ZTF light curve for the target.")], provenance)
    if not with_data and not include_flagged:
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        ids = ", ".join(f"{r['oid']} ({_ztf_band(r)})" for r in same)
        return SurveyResult("ztf", [], [*notes, (
            f"ZTF: the target's object(s) {ids} have no good epochs (ngoodobs = 0; saturated or otherwise always "
            "flagged?): no ZTF light curve for the target.")], provenance)
    oids = sorted({str(r["oid"]).strip() for r in (with_data if not include_flagged else same)})
    params: list[tuple[str, str]] = [("ID", oid) for oid in oids] + [("COLLECTION", collection), ("FORMAT", "csv")]
    response = await _request(client, "GET", ZTF_LIGHTCURVE_URL, service="ztf", params=params, timeout=tout)
    result = await asyncio.to_thread(parse_ztf_csv, response.text, ra, dec, include_flagged=include_flagged,
                                     oids=oids, service="ztf", target=target if moving else None,
                                     match_arcsec=tolerance)
    result.notes[:0] = notes
    provenance["candidate_object_ids"] = provenance["target_object_ids"]
    provenance["target_object_ids"] = sorted({oid for s in result.series for oid in s.source_ids})
    provenance.update({"lightcurve_service": ZTF_LIGHTCURVE_URL, "lightcurve_params": params,
                       "rows": result.provenance.get("rows"), "elapsed_s": round(time.perf_counter() - started, 3)})
    result.provenance = provenance
    return result


# ---------------------------------------------------------------------------
# NEOWISE-R (IRSA TAP, single exposures binned per visit)
# ---------------------------------------------------------------------------

NEOWISE_COLUMNS = ("ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                   "qual_frame", "qi_fact", "saa_sep", "moon_masked", "cc_flags", "nb", "na", "w1rchi2", "w2rchi2")
NEOWISE_QUALITY_CUTS = ("qual_frame>0, qi_fact>0, saa_sep>0, moon_masked[band]=='0', cc_flags[band]=='0', "
                        "nb==1, na==0; a visit is kept only if >= 50% (and >= 3) of its detections pass")
# NEOWISE-R single exposures span 2013-12-13 .. 2024-08-01 (final data release).
NEOWISE_EPOCH_RANGE = (2013.95, 2024.59)


def neowise_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    return (f"SELECT {', '.join(NEOWISE_COLUMNS)} FROM neowiser_p1bs_psd "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.8f})) = 1")


def visit_groups(t: Sequence[float], *, gap_days: float = NEOWISE_VISIT_GAP_DAYS,
                 max_span_days: float = NEOWISE_VISIT_MAX_SPAN_DAYS) -> list[np.ndarray]:
    """Indices (into ``t``) of each visit in time order: a new visit starts after a gap
    longer than ``gap_days`` or once the visit spans more than ``max_span_days``."""
    ts = np.asarray(t, dtype=float)
    order = np.argsort(ts, kind="stable")
    groups: list[np.ndarray] = []
    start = 0
    for i in range(1, len(order) + 1):
        if (i < len(order) and ts[order[i]] - ts[order[i - 1]] <= gap_days
                and ts[order[i]] - ts[order[start]] <= max_span_days):
            continue
        groups.append(order[start:i])
        start = i
    return groups


def bin_visits(t: Sequence[float], y: Sequence[float], dy: Sequence[float], *,
               gap_days: float = NEOWISE_VISIT_GAP_DAYS, max_span_days: float = NEOWISE_VISIT_MAX_SPAN_DAYS,
               ) -> list[LightCurvePoint]:
    """Average single exposures per visit (inverse-variance weighted mean).

    The bin error is the larger of the formal weighted-mean error and the standard
    error of the mean (sample std / sqrt(n)), so unmodelled scatter within a visit
    is not hidden. The bin time is the unweighted mean epoch.
    """
    ts, ys, es = (np.asarray(a, dtype=float) for a in (t, y, dy))
    points: list[LightCurvePoint] = []
    for idx in visit_groups(ts, gap_days=gap_days, max_span_days=max_span_days):
        tt, yy, ee = ts[idx], ys[idx], es[idx]
        w = 1.0 / ee**2
        mean = float(np.sum(w * yy) / np.sum(w))
        err = float(1.0 / math.sqrt(np.sum(w)))
        if len(yy) > 1:
            err = max(err, float(np.std(yy, ddof=1) / math.sqrt(len(yy))))
        points.append(LightCurvePoint(float(np.mean(tt)), mean, err, 0, len(yy)))
    return points


def _neowise_exposure_ok(row: dict[str, str], idx: int, err: float | None) -> bool:
    moon = (row.get("moon_masked") or "").strip()
    cc = (row.get("cc_flags") or "").strip()
    nb, na = _num(row.get("nb")), _num(row.get("na"))
    return bool((_num(row.get("qual_frame")) or 0) > 0 and (_num(row.get("qi_fact")) or 0) > 0
                and (_num(row.get("saa_sep")) or 0) > 0 and len(moon) > idx and moon[idx] == "0"
                and len(cc) > idx and cc[idx] == "0" and (nb is None or nb == 1) and (na is None or na == 0)
                and err is not None and err > 0)


def parse_neowise_csv(text: str, ra: float, dec: float, *, binned: bool = True, target: Target | None = None,
                      match_arcsec: float = NEOWISE_MATCH_ARCSEC, include_flagged: bool = False,
                      service: str | None = None) -> SurveyResult:
    """Clean NEOWISE-R single exposures and (by default) bin them per visit (W1, W2).

    Visit means are robust against single-exposure outliers but average over the
    ~1 day a visit lasts, suppressing variability faster than that (e.g. RR Lyrae);
    ``binned=False`` returns the cleaned single exposures instead (``n`` = 1).

    Source association: in each frame (one ``mjd``) the detection nearest the target
    is used, and only if it lies within ``match_arcsec`` (default 3", half the W1/W2
    PSF FWHM) of the target position *at that frame's epoch* (proper motion applied
    when ``target`` carries one) -- whatever the cone radius -- so a neighbour never
    stands in for the target in frames where the target itself was not detected.

    Exposure cuts (NEOWISE Explanatory Supplement sec. II.3): ``qual_frame > 0``,
    ``qi_fact > 0``, ``saa_sep > 0``, unmasked Moon and clean contamination flag of the
    band (``moon_masked`` / ``cc_flags`` character = '0'), a single-component,
    non-deblended profile fit (``nb == 1``, ``na == 0``). Null magnitudes
    (non-detections) are dropped. A visit is used only when at least half of the band's
    detections in it (and >= 3) pass: persistence and other artefacts affect whole
    visits, and the few exposures escaping the flags are biased too.

    ``include_flagged`` also returns the rejected single exposures of the target (flag 1:
    failed the exposure cuts; flag 2: passed but in a rejected visit), never binned and
    never used for statistics. ``service`` validates an upstream answer (:func:`_csv_rows`).
    """
    rows = _csv_rows(text, required=("ra", "dec", "mjd"), service=service)
    if not rows:
        return SurveyResult("neowise", [], ["NEOWISE-R: no single-exposure detections in the cone."], {"rows": 0})
    nearest_per_frame: dict[float, tuple[float, dict[str, str]]] = {}
    frames_far: set[float] = set()
    moving = target is not None and target.epoch is not None and target.proper_motion is not None
    for row in rows:
        mjd, rra, rdec = _num(row.get("mjd")), _num(row.get("ra")), _num(row.get("dec"))
        if mjd is None or rra is None or rdec is None:
            continue
        tra, tdec = position_at(target, jyear_from_mjd(mjd)) if moving and target is not None else (ra, dec)
        sep = haversine_arcsec(tra, tdec, rra, rdec)
        if sep > match_arcsec:
            frames_far.add(mjd)
            continue
        if mjd not in nearest_per_frame or sep < nearest_per_frame[mjd][0]:
            nearest_per_frame[mjd] = (sep, row)
    notes: list[str] = []
    far_only = frames_far - set(nearest_per_frame)
    if far_only:
        notes.append(f"NEOWISE-R: {len(far_only)} frame(s) with detections only more than {match_arcsec:g} arcsec "
                     "from the target (neighbours) were ignored.")
    series: list[LightCurveSeries] = []
    for idx, band in enumerate(("W1", "W2")):
        col, ecol = f"w{idx + 1}mpro", f"w{idx + 1}sigmpro"
        det_t: list[float] = []
        det_rows: list[dict[str, str]] = []
        for mjd, (_sep, row) in nearest_per_frame.items():
            if _num(row.get(col)) is not None:
                det_t.append(mjd)
                det_rows.append(row)
        total = len(det_t)
        ts: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        flagged: list[LightCurvePoint] = []  # rejected exposures, returned only with include_flagged
        visits_rejected = 0
        for group in visit_groups(det_t):
            passing = [int(i) for i in group
                       if _neowise_exposure_ok(det_rows[int(i)], idx, _num(det_rows[int(i)].get(ecol)))]
            visit_ok = not (len(passing) < NEOWISE_MIN_VISIT_EXPOSURES
                            or len(passing) < NEOWISE_MIN_VISIT_PASS_FRACTION * len(group))
            if not visit_ok and passing:
                visits_rejected += 1
            for i in (int(j) for j in group):
                if visit_ok and i in passing:
                    ts.append(det_t[i])
                    ys.append(float(_num(det_rows[i].get(col)) or 0.0))
                    es.append(float(_num(det_rows[i].get(ecol)) or 0.0))
                elif include_flagged:
                    # flag 1: the exposure failed the cuts; 2: it passed but its visit was rejected
                    flagged.append(LightCurvePoint(det_t[i], float(_num(det_rows[i].get(col)) or 0.0),
                                                   _num(det_rows[i].get(ecol)), 2 if i in passing else 1, 1))
        rejected = total - len(ts)
        if not ts:
            if total:
                notes.append(f"NEOWISE-R {band}: none of the {total} exposures passed the quality cuts.")
            if not flagged:
                continue
        if visits_rejected:
            notes.append(f"NEOWISE-R {band}: {visits_rejected} visit(s) dropped because most of their exposures "
                         "failed the quality cuts (e.g. persistence).")
        if not ts:
            points = []
        elif binned:
            points = bin_visits(ts, ys, es)
        else:
            order = np.argsort(ts)
            points = [LightCurvePoint(ts[i], ys[i], es[i], 0, 1) for i in order]
        points = points + flagged
        bmjd = utc_mjd_to_bmjd_tdb([p.mjd for p in points], ra, dec)
        for p, t in zip(points, bmjd):
            p.mjd = float(t)
        points.sort(key=lambda p: p.mjd)
        median_mag = float(np.median(ys)) if ys else None
        if median_mag is not None and median_mag < NEOWISE_SATURATION_MAG[band]:
            notes.append(f"NEOWISE-R {band}: median {median_mag:.2f} mag is brighter than the single-exposure saturation "
                         f"limit ({NEOWISE_SATURATION_MAG[band]} mag); profile-fit photometry is biased.")
        scale, floor = ERROR_MODELS[("neowise", band)]
        n_visits = (len(bin_visits(ts, ys, es)) if not binned else len(points) - len(flagged)) if ts else 0
        series.append(LightCurveSeries(
            survey="neowise", band=band, unit="mag", points=points, photometric_system="Vega (WISE profile-fit)",
            source_ids=[], n_total=total, n_rejected=rejected,
            metadata={"binning": ("per visit (inverse-variance weighted mean of single exposures)" if binned
                                  else "none (single exposures)"),
                      "n_exposures_used": len(ts), "n_visits": n_visits,
                      "n_visits_rejected": visits_rejected, "match_radius_arcsec": match_arcsec,
                      "quality_cuts": NEOWISE_QUALITY_CUTS, "error_scale": scale, "error_floor_mag": floor,
                      "flagged_points": ("single rejected exposures, flag 1 = failed the exposure cuts, 2 = in a "
                                         "rejected visit" if include_flagged else "not returned")},
        ))
    return SurveyResult("neowise", series, notes, {"rows": len(rows)})


async def fetch_neowise(client: httpx.AsyncClient, ra: float, dec: float, radius_arcsec: float, *,
                        binned: bool = True, timeout: float | None = None, target: Target | None = None,
                        include_flagged: bool = False) -> SurveyResult:
    """NEOWISE-R W1/W2 light curves (per visit, or single exposures) from IRSA TAP ``neowiser_p1bs_psd``.

    ``target`` (with epoch and proper motion) lets each exposure be matched at the
    target's position at that exposure's epoch.
    """
    query = neowise_adql(ra, dec, radius_arcsec)
    started = time.perf_counter()
    response = await _tap_query(client, IRSA_TAP_SYNC_URL, query, service="neowise",
                                timeout=timeout or DEFAULT_TIMEOUTS["neowise"])
    result = await asyncio.to_thread(parse_neowise_csv, response.text, ra, dec, binned=binned, target=target,
                                     include_flagged=include_flagged, service="neowise")
    result.provenance = {"service": IRSA_TAP_SYNC_URL, "table": "neowiser_p1bs_psd", "query": query,
                         "rows": result.provenance.get("rows"), "elapsed_s": round(time.perf_counter() - started, 3)}
    return result


# ---------------------------------------------------------------------------
# Gaia DR3 epoch photometry (TAP + DataLink)
# ---------------------------------------------------------------------------


def gaia_cone_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    return ("SELECT TOP 5 source_id, ra, dec, phot_g_mean_mag, has_epoch_photometry, phot_variable_flag, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra:.7f}, {dec:.7f})) AS dist "
            "FROM gaiadr3.gaia_source "
            f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.8f})) "
            "ORDER BY dist ASC")


GAIA_BANDS: tuple[tuple[str, str, str, str, str], ...] = (
    # band, time column, mag column, flux/error column, variability reject flag
    ("G", "g_transit_time", "g_transit_mag", "g_transit_flux_over_error", "variability_flag_g_reject"),
    ("BP", "bp_obs_time", "bp_mag", "bp_flux_over_error", "variability_flag_bp_reject"),
    ("RP", "rp_obs_time", "rp_mag", "rp_flux_over_error", "variability_flag_rp_reject"),
)
_MAG_PER_FRACTIONAL_FLUX = 2.5 / math.log(10.0)  # sigma_m = (2.5/ln 10) * sigma_F / F


def gaia_g_error_model(g_mag: float | None) -> tuple[float, float]:
    """(scale, floor) of the Gaia DR3 G per-transit error model for a source of mean G.

    floor = max(3 mmag, Evans et al. 2023 Table 2 value interpolated at G); see
    :data:`ERROR_MODELS` for the calibration on non-variable GAPS sources.
    """
    scale, min_floor = ERROR_MODELS[("gaia", "G")]
    if g_mag is None or not math.isfinite(g_mag):
        return scale, min_floor
    mags, extra = zip(*GAIA_G_EXTRA_ERROR)
    return scale, max(min_floor, float(np.interp(g_mag, mags, extra)))


def parse_gaia_epoch_csv(text: str, *, source_id: str, include_flagged: bool = False,
                         service: str | None = None) -> list[LightCurveSeries]:
    """Parse Gaia DR3 EPOCH_PHOTOMETRY (DataLink CSV, INDIVIDUAL structure).

    Magnitude errors are propagated from ``*_flux_over_error``:
    sigma_m = 2.5 / ln(10) / (F / sigma_F). Transits with ``rejected_by_photometry`` or
    the band's ``variability_flag_*_reject`` set are flagged (flag = 1). ``service``
    validates an upstream answer (:func:`_csv_rows`).
    """
    rows = _csv_rows(text, required=("source_id", "g_transit_time", "g_transit_mag"), service=service)
    series: list[LightCurveSeries] = []
    for band, tcol, mcol, foe_col, rej_col in GAIA_BANDS:
        raw: list[tuple[float, float, float | None, int]] = []
        for row in rows:
            t, mag, foe = _num(row.get(tcol)), _num(row.get(mcol)), _num(row.get(foe_col))
            if t is None or mag is None:
                continue
            err = _MAG_PER_FRACTIONAL_FLUX / foe if foe and foe > 0 else None
            flag = 1 if (_truthy(row.get("rejected_by_photometry")) or _truthy(row.get(rej_col)) or err is None) else 0
            raw.append((t, mag, err, flag))
        if not raw:
            continue
        bmjd = gaia_time_to_bmjd_tdb([r[0] for r in raw])
        rejected = sum(1 for r in raw if r[3])
        points = [LightCurvePoint(float(t), r[1], r[2], r[3]) for r, t in zip(raw, bmjd) if include_flagged or not r[3]]
        points.sort(key=lambda p: p.mjd)
        series.append(LightCurveSeries(
            survey="gaia", band=band, unit="mag", points=points, photometric_system="Vega (Gaia DR3 passbands)",
            source_ids=[f"Gaia DR3 {source_id}"], n_total=len(raw), n_rejected=rejected,
            metadata={"quality_cut": f"rejected_by_photometry == false and {rej_col} == false",
                      "native_time": "BJD(TCB) - 2455197.5"},
        ))
    return series


async def fetch_gaia(client: httpx.AsyncClient, ra: float, dec: float, radius_arcsec: float, *,
                     include_flagged: bool = False, timeout: float | None = None,
                     match_arcsec: float = GAIA_MATCH_ARCSEC) -> SurveyResult:
    """Gaia DR3 G/BP/RP epoch photometry of the target's DR3 source.

    Gaia DR3 published epoch photometry only for ~11.7 million sources (mostly
    variability candidates; ``has_epoch_photometry``). (ra, dec) must be at J2016.0
    (see :func:`survey_cone`). The nearest DR3 source is the target only within
    ``match_arcsec`` (or the cone radius, if smaller); otherwise no series is returned
    and the sources in the cone are listed as neighbours.
    """
    tout = timeout or DEFAULT_TIMEOUTS["gaia"]
    query = gaia_cone_adql(ra, dec, radius_arcsec)
    started = time.perf_counter()
    response = await _tap_query(client, GAIA_TAP_SYNC_URL, query, service="gaia", timeout=tout)
    sources = _csv_rows(response.text, required=("source_id", "ra", "dec", "has_epoch_photometry"), service="gaia")
    tolerance = min(match_arcsec, radius_arcsec)
    provenance: dict[str, Any] = {"tap_service": GAIA_TAP_SYNC_URL, "query": query, "rows": len(sources),
                                  "match_radius_arcsec": tolerance}
    located = [(haversine_arcsec(ra, dec, _num(r.get("ra")), _num(r.get("dec"))), r) for r in sources
               if _num(r.get("ra")) is not None and _num(r.get("dec")) is not None]
    located.sort(key=lambda item: item[0])
    if not located:
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        return SurveyResult("gaia", [], ["Gaia DR3: no source in the cone."], provenance)
    sep, nearest = located[0]
    if sep > tolerance:
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        provenance["neighbour_source_ids"] = {str(r.get("source_id", "")).strip(): round(d, 3) for d, r in located}
        listed = ", ".join(f"{str(r.get('source_id', '')).strip()} ({d:.1f}\")" for d, r in located[:5])
        return SurveyResult("gaia", [], [(f"Gaia DR3: no source within {tolerance:g}\" of the target (J2016.0); "
                                          f"sources in the cone (not used): {listed}.")], provenance)
    source_id = str(nearest.get("source_id", "")).strip()
    provenance.update({"source_id": source_id, "separation_arcsec": round(sep, 3)})
    if not _truthy(nearest.get("has_epoch_photometry")):
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        note = (f"Gaia DR3 {source_id} ({sep:.2f}\" away) has no published epoch photometry "
                "(has_epoch_photometry = false).")
        return SurveyResult("gaia", [], [note], provenance)
    params = {"RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "ID": f"Gaia DR3 {source_id}", "DATA_STRUCTURE": "INDIVIDUAL",
              "FORMAT": "CSV", "RELEASE": "Gaia DR3", "VALID_DATA": "false"}
    response = await _request(client, "GET", GAIA_DATALINK_URL, service="gaia", params=params, timeout=tout)
    if response.text.lstrip().startswith("<"):
        raise UpstreamServiceError("gaia", f"DataLink returned no CSV: {_error_detail(response)}",
                                   status_code=response.status_code, retryable=True)
    series = await asyncio.to_thread(parse_gaia_epoch_csv, response.text, source_id=source_id,
                                     include_flagged=include_flagged, service="gaia")
    g_mag = _num(nearest.get("phot_g_mean_mag"))
    for s in series:
        s.metadata.update({"source_ra": _num(nearest.get("ra")), "source_dec": _num(nearest.get("dec")),
                           "source_separation_arcsec": round(sep, 3), "phot_g_mean_mag": g_mag,
                           "phot_variable_flag": (nearest.get("phot_variable_flag") or "").strip() or None})
        if s.band == "G":
            scale, floor = gaia_g_error_model(g_mag)
            s.metadata.update({"error_scale": scale, "error_floor_mag": floor})
    provenance.update({"datalink_service": GAIA_DATALINK_URL, "datalink_params": params,
                       "elapsed_s": round(time.perf_counter() - started, 3)})
    notes = [] if series else [f"Gaia DR3 {source_id}: DataLink returned no epoch photometry rows."]
    return SurveyResult("gaia", series, notes, provenance)


# ---------------------------------------------------------------------------
# TESS SPOC light curves (MAST)
# ---------------------------------------------------------------------------

_TESS_SPOC_2MIN = re.compile(r"^tess\d{13}-s(\d{4})-(\d{16})-\d{4}-s$")


async def _mast_invoke(client: httpx.AsyncClient, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """POST a MAST ``invoke`` request, re-submitting while the service reports EXECUTING."""
    body = {"request": json.dumps(request)}
    for _attempt in range(30):
        response = await _request(client, "POST", MAST_INVOKE_URL, service="tess", data=body, timeout=timeout)
        try:
            payload = response.json()
        except ValueError as exc:
            # e.g. an HTML maintenance page served with HTTP 200: a service fault, retryable.
            snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text[:400])).strip()
            raise UpstreamServiceError("tess", f"MAST returned non-JSON: {snippet[:150] or 'empty body'}",
                                       status_code=response.status_code, retryable=True) from exc
        if not isinstance(payload, dict):
            raise UpstreamServiceError("tess", f"MAST returned JSON of type {type(payload).__name__}",
                                       status_code=response.status_code, retryable=True)
        status = str(payload.get("status", "")).upper()
        if status == "COMPLETE":
            return payload
        if status not in {"EXECUTING", "PENDING"}:
            raise UpstreamServiceError("tess", f"MAST status {status}: {payload.get('msg', '')}", status_code=response.status_code)
        await asyncio.sleep(1.0)
    raise UpstreamServiceError("tess", "MAST query did not complete", retryable=True)


FITS_BLOCK_BYTES = 2880  # FITS files consist of 2880-byte blocks (FITS Standard 4.0, sec. 3.1)
_TESS_LC_COLUMNS = ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_FLUX", "SAP_FLUX_ERR", "QUALITY")


def validate_tess_download(content: bytes, content_type: str | None = None) -> None:
    """Reject a light-curve download that cannot be a complete FITS file.

    MAST may answer HTTP 200 with an HTML error/maintenance page, an empty body or a
    truncated file; those are service faults (:class:`UpstreamServiceError`, retryable),
    detected before parsing: a FITS file starts with the ``SIMPLE  =`` card, is a whole
    number of 2880-byte blocks and holds the data its primary and light-curve headers
    declare (:func:`_fits_extent`).
    """
    if not content:
        raise UpstreamServiceError("tess", "light-curve download is empty", status_code=200, retryable=True)
    if not content.startswith(b"SIMPLE  ="):
        kind = content_type or "unknown content type"
        snippet = re.sub(r"\s+", " ", re.sub(rb"<[^>]+>", b" ", content[:300]).decode("latin-1")).strip()
        raise UpstreamServiceError("tess", f"light-curve download is not a FITS file ({kind}): {snippet[:120]}",
                                   status_code=200, retryable=True)
    if len(content) % FITS_BLOCK_BYTES:
        raise UpstreamServiceError("tess", f"light-curve download is truncated ({len(content)} bytes, not a multiple "
                                   f"of the {FITS_BLOCK_BYTES}-byte FITS block)", status_code=200, retryable=True)
    needed = _fits_extent(content, n_hdus=2)
    if needed is None or needed > len(content):
        raise UpstreamServiceError("tess", f"light-curve download is truncated ({len(content)} bytes; its headers "
                                   f"declare {needed if needed is not None else 'more'} bytes)", status_code=200,
                                   retryable=True)


def _fits_extent(content: bytes, *, n_hdus: int) -> int | None:
    """Bytes spanned by the first ``n_hdus`` HDUs according to their headers (None when a
    header is incomplete). Data size = |BITPIX|/8 x GCOUNT x (PCOUNT + NAXIS1 x ... x NAXISn),
    padded to whole blocks (FITS Standard 4.0, sec. 4.4.1)."""
    pos = 0
    for _ in range(n_hdus):
        cards: dict[str, str] = {}
        while True:
            block = content[pos:pos + FITS_BLOCK_BYTES]
            if len(block) < FITS_BLOCK_BYTES:
                return None
            pos += FITS_BLOCK_BYTES
            ended = False
            for i in range(0, FITS_BLOCK_BYTES, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    ended = True
                    break
                if card[8:10] == "= ":
                    cards.setdefault(key, card[10:].split("/", 1)[0].strip())
            if ended:
                break
        try:
            naxis = int(cards.get("NAXIS", "0"))
            bitpix = abs(int(cards.get("BITPIX", "8")))
            axes = [int(cards.get(f"NAXIS{i}", "0")) for i in range(1, naxis + 1)]
            pcount, gcount = int(cards.get("PCOUNT", "0")), int(cards.get("GCOUNT", "1"))
        except ValueError:
            return None
        size = bitpix // 8 * gcount * (pcount + (math.prod(axes) if axes else 0))
        pos += math.ceil(size / FITS_BLOCK_BYTES) * FITS_BLOCK_BYTES
    return pos


def parse_tess_lc_fits(content: bytes) -> dict[str, Any]:
    """Arrays and header facts of a SPOC light-curve file.

    Returns ``{"info", "t", "pdcsap", "pdcsap_err", "sap", "sap_err", "quality"}`` with
    ``t`` in BMJD_TDB: TIME is BJD_TDB - (BJDREFI + BJDREFF) per the SPOC data
    products (TESS Science Data Products Description Document); both reference
    keywords are read from the header. ``info`` carries TIC ID, sector, TESSMAG, the TIC
    stellar RADIUS (R_sun) and TEFF (K) of the primary header, and the crowding metrics
    CROWDSAP / FLFRCSAP used by PDC. A download that is not a
    complete, readable SPOC light-curve file raises a retryable
    :class:`UpstreamServiceError` (never a raw parser exception).
    """
    validate_tess_download(content)
    try:
        return _parse_tess_lc_fits(content)
    except UpstreamServiceError:
        raise
    except (OSError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        raise UpstreamServiceError("tess", "light-curve file is unreadable (corrupt or truncated FITS)",
                                   status_code=200, retryable=True) from exc


def _parse_tess_lc_fits(content: bytes) -> dict[str, Any]:
    from astropy.io import fits

    with fits.open(io.BytesIO(content), memmap=False) as hdul:
        hdr0, hdu = hdul[0].header, hdul[1]
        hdr, data = hdu.header, hdu.data
        missing = [c for c in _TESS_LC_COLUMNS if data is None or c not in data.columns.names]
        if missing:
            raise UpstreamServiceError("tess", f"light-curve file lacks column(s) {missing}", status_code=200,
                                       retryable=True)
        ref = float(hdr.get("BJDREFI", 0)) + float(hdr.get("BJDREFF", 0.0))
        if ref == 0 or str(hdr.get("TIMESYS", "TDB")).upper() != "TDB":
            raise UpstreamServiceError("tess", "light-curve file lacks BJDREFI/TDB time reference")
        out = {
            "t": np.asarray(data["TIME"], dtype=float) + ref - MJD_JD_OFFSET,
            "pdcsap": np.asarray(data["PDCSAP_FLUX"], dtype=float),
            "pdcsap_err": np.asarray(data["PDCSAP_FLUX_ERR"], dtype=float),
            "sap": np.asarray(data["SAP_FLUX"], dtype=float),
            "sap_err": np.asarray(data["SAP_FLUX_ERR"], dtype=float),
            "quality": np.asarray(data["QUALITY"], dtype=np.int64),
            "info": {"tic_id": str(hdr0.get("TICID", "")), "sector": int(hdr0.get("SECTOR", -1)),
                     "tess_mag": _num(hdr0.get("TESSMAG")), "ra_obj": _num(hdr0.get("RA_OBJ")),
                     "dec_obj": _num(hdr0.get("DEC_OBJ")), "crowdsap": _num(hdr.get("CROWDSAP")),
                     "flfrcsap": _num(hdr.get("FLFRCSAP")), "radius_rsun": _num(hdr0.get("RADIUS")),
                     "teff": _num(hdr0.get("TEFF"))},
        }
    return out


def choose_tess_flux(lc: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Pick PDCSAP (default) or SAP flux for one sector; returns (column, good mask, flux, err, note).

    PDC removes the fraction ``1 - CROWDSAP`` of the aperture flux attributed to
    other stars using TIC magnitudes. A wrong TIC magnitude for the target over-
    subtracts and drives PDCSAP to non-physical values <= 0; stellar flux cannot be
    negative, so such sectors fall back to the uncorrected SAP flux (e.g. RR Lyr,
    TIC 159717514, whose TIC Tmag of 16.6 contradicts its V = 7.2).
    """
    t, quality = lc["t"], lc["quality"]
    base = (quality == 0) & np.isfinite(t)
    good = base & np.isfinite(lc["pdcsap"]) & np.isfinite(lc["pdcsap_err"]) & (lc["pdcsap_err"] > 0)
    if np.any(good) and np.all(lc["pdcsap"][good] > 0):
        return "PDCSAP_FLUX", good, lc["pdcsap"], lc["pdcsap_err"], None
    sap_good = base & np.isfinite(lc["sap"]) & np.isfinite(lc["sap_err"]) & (lc["sap_err"] > 0) & (lc["sap"] > 0)
    info = lc["info"]
    note = (f"TESS sector {info['sector']}: PDCSAP flux has non-positive values (CROWDSAP={info['crowdsap']}, "
            f"TIC Tmag={info['tess_mag']}); using SAP flux, which is not corrected for crowding or systematics.")
    return "SAP_FLUX", sap_good, lc["sap"], lc["sap_err"], note


def bin_uniform(t: np.ndarray, y: np.ndarray, dy: np.ndarray, width_days: float) -> list[LightCurvePoint]:
    """Mean-bin a densely sampled series in fixed-width time bins (error = rss / n); O(n)."""
    t, y, dy = (np.asarray(a, dtype=float) for a in (t, y, dy))
    if width_days <= 0:
        return [LightCurvePoint(float(a), float(b), float(c), 0) for a, b, c in zip(t, y, dy)]
    if t.size == 0:
        return []
    idx = np.floor((t - t[0]) / width_days).astype(np.int64)
    _keys, inverse = np.unique(idx, return_inverse=True)
    counts = np.bincount(inverse)
    tm = np.bincount(inverse, weights=t) / counts
    ym = np.bincount(inverse, weights=y) / counts
    em = np.sqrt(np.bincount(inverse, weights=dy**2)) / counts
    return [LightCurvePoint(float(a), float(b), float(c), 0, int(n)) for a, b, c, n in zip(tm, ym, em, counts)]


def select_tess_observations(observations: Sequence[dict[str, Any]], ra: float, dec: float, *, max_sectors: int,
                             match_arcsec: float = TESS_MATCH_ARCSEC,
                             ) -> tuple[str | None, list[tuple[int, dict[str, Any]]], dict[str, float]]:
    """Pick the SPOC 2-min observations of ONE star: the TIC nearest (ra, dec), if it lies
    within ``match_arcsec`` (otherwise none: a SPOC target farther away is another star).

    MAST returns every SPOC target whose position lies in the cone; mixing their
    light curves would combine different stars (e.g. 61 Cyg A and B, 30" apart).
    Observations are grouped by the TIC ID encoded in the SPOC ``obs_id``
    (``tess<date>-s<sector>-<16-digit TIC>-<camera/ccd>-s``); the TIC whose target
    position (``s_ra``/``s_dec``) is nearest wins, and its ``max_sectors`` most recent
    distinct sectors are returned. Returns (TIC, [(sector, obs)], {other TIC: sep}).
    """
    by_tic: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    seps: dict[str, float] = {}
    for obs in observations:
        match = _TESS_SPOC_2MIN.match(str(obs.get("obs_id", "")))
        if not match:
            continue
        tic = str(int(match.group(2)))
        by_tic.setdefault(tic, []).append((int(match.group(1)), obs))
        ora, odec = _num(obs.get("s_ra")), _num(obs.get("s_dec"))
        if ora is not None and odec is not None:
            sep = haversine_arcsec(ra, dec, ora, odec)
            seps[tic] = min(sep, seps.get(tic, sep))
    if not by_tic:
        return None, [], {}
    tic = min(by_tic, key=lambda k: (seps.get(k, math.inf), k))
    if not seps.get(tic, math.inf) <= match_arcsec:
        return None, [], {k: round(seps.get(k, math.nan), 2) for k in sorted(by_tic, key=lambda k: seps.get(k, math.inf))}
    chosen: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for sector, obs in sorted(by_tic[tic], key=lambda item: item[0], reverse=True):
        if sector in seen:
            continue
        seen.add(sector)
        chosen.append((sector, obs))
        if len(chosen) >= max(1, max_sectors):
            break
    others = {k: round(seps.get(k, math.nan), 2) for k in sorted(by_tic) if k != tic}
    return tic, chosen, others


def _assemble_tess(sectors: Sequence[tuple[int, str, dict[str, Any]]], bin_minutes: float, *,
                   include_flagged: bool = False, max_points: int | None = None,
                   ) -> tuple[list[LightCurvePoint], dict[str, Any], list[str], int, int]:
    """Normalise each sector by its median good flux, concatenate, bin (CPU-bound: run in a thread).

    With ``include_flagged`` the cadences with QUALITY != 0 (and finite flux) are appended
    unbinned, flag = the SPOC QUALITY bit mask; they never enter the statistics.

    The number of returned points is bounded by ``max_points`` (default
    :data:`MAX_TESS_POINTS`) on the actual data: request validation can only estimate it
    from nominal 27.4-d sectors, but sectors differ (sector 97 of tau Cet: 54.6 d, 39,335
    cadences). Beyond it the good cadences are binned to the smallest multiple of the
    cadence that fits (flagged cadences, if they would take more than half the budget, are
    left out), with a note; ``meta["bin_minutes"]`` is the width actually used.
    """
    limit = MAX_TESS_POINTS if max_points is None else int(max_points)
    flagged: list[LightCurvePoint] = []
    ts: list[np.ndarray] = []
    fs: list[np.ndarray] = []
    es: list[np.ndarray] = []
    notes: list[str] = []
    flux_columns: dict[str, str] = {}
    crowding: dict[str, float | None] = {}
    used: list[int] = []
    files: list[str] = []
    total = rejected = 0
    info: dict[str, Any] = {}
    for sector, uri, lc_data in sectors:
        info, t = lc_data["info"], lc_data["t"]
        column, good, flux, ferr, note = choose_tess_flux(lc_data)
        total += int(np.sum(np.isfinite(t)))
        rejected += int(np.sum(np.isfinite(t) & ~good))
        if not np.any(good):
            continue
        if note:
            notes.append(note)
        flux_columns[str(sector)] = column
        crowding[str(sector)] = info.get("crowdsap")
        norm = float(np.median(flux[good]))
        ts.append(t[good])
        fs.append(flux[good] / norm)
        es.append(ferr[good] / norm)
        if include_flagged:
            bad = np.isfinite(t) & ~good & np.isfinite(flux) & (lc_data["quality"] != 0)
            flagged.extend(LightCurvePoint(float(a), float(b) / norm,
                                           float(c) / norm if np.isfinite(c) and c > 0 else None, int(q) or 1)
                           for a, b, c, q in zip(t[bad], flux[bad], ferr[bad], lc_data["quality"][bad]))
        used.append(sector)
        files.append(uri)
    if not ts:
        return [], {"sectors": used, "files": files, "bin_minutes": bin_minutes}, notes, total, rejected
    t_all, f_all, e_all = (np.concatenate(a) for a in (ts, fs, es))
    order = np.argsort(t_all)
    t_all, f_all, e_all = t_all[order], f_all[order], e_all[order]
    width = bin_minutes / 1440.0
    points = bin_uniform(t_all, f_all, e_all, width)
    if flagged and len(points) + len(flagged) > limit and len(flagged) > limit // 2:
        notes.append(f"TESS: the {len(flagged):,} quality-flagged cadences are not returned (they would take more than "
                     f"half of the {limit:,}-point limit).")
        flagged = []
    room = limit - len(flagged)
    if len(points) > room:
        steps = np.diff(t_all)
        cadence = float(np.median(steps[steps > 0])) if np.any(steps > 0) else TESS_CADENCE_MINUTES / 1440.0
        factor = max(2, math.ceil(len(t_all) / room))
        while True:
            width = factor * cadence
            points = bin_uniform(t_all, f_all, e_all, width)
            if len(points) <= room:
                break
            factor += 1
        notes.append(f"TESS: {len(t_all):,} good cadences would exceed the {limit:,}-point limit at "
                     f"tess_bin_minutes = {bin_minutes:g}; binned to {width * 1440.0:.3g} min ({len(points):,} points).")
    if flagged:
        points = sorted(points + flagged, key=lambda p: p.mjd)
    meta = {"sectors": used, "files": files, "flux_columns": flux_columns, "crowdsap": crowding,
            "bin_minutes": round(width * 1440.0, 6),
            "tic_tess_mag": info.get("tess_mag"), "tic_radius_rsun": info.get("radius_rsun"),
            "tic_teff": info.get("teff")}
    return points, meta, notes, total, rejected


async def fetch_tess(client: httpx.AsyncClient, ra: float, dec: float, radius_arcsec: float, *,
                     max_sectors: int = 2, bin_minutes: float = 10.0, timeout: float | None = None,
                     include_flagged: bool = False, match_arcsec: float = TESS_MATCH_ARCSEC,
                     deadline_s: float | None = None) -> SurveyResult:
    """TESS SPOC 2-min light curves of the SPOC target nearest (ra, dec) (most recent ``max_sectors`` sectors).

    (ra, dec) should be at epoch J2000.0, the epoch of TIC-8 positions (Stassun et al.
    2019, AJ 158, 138). PDCSAP flux is used (SAP where PDCSAP is unphysical, see
    :func:`choose_tess_flux`); each sector is normalised by its median good flux
    (QUALITY == 0) and the series is binned to ``bin_minutes`` (0 disables binning).
    Only stars with SPOC 2-min target-pixel light curves are covered (no FFI photometry).
    Other SPOC targets inside the cone are never mixed in; they are listed in a note,
    and the nearest one is used only within ``match_arcsec`` (or the cone radius, if
    smaller) of (ra, dec).

    Sectors are fetched one by one and independently: a sector whose product list or
    download fails (after the retry of :func:`_request`), or that is still missing
    :data:`TESS_ASSEMBLY_MARGIN_S` before ``deadline_s``, is reported in a note and in
    ``provenance["sectors_failed"]`` while the others are returned (``provenance["partial"]``
    True; fetch_survey does not cache such a result). Only when every selected sector failed
    does the survey fail. A selected sector without an ``_lc.fits`` product is skipped with
    a note.
    """
    _validate_tess_options(max_sectors, bin_minutes)
    tout = timeout or DEFAULT_TIMEOUTS["tess"]
    started = time.perf_counter()
    request = {"service": "Mast.Caom.Filtered.Position", "format": "json",
               "params": {"columns": "obsid,obs_id,target_name,sequence_number,t_exptime,s_ra,s_dec",
                          "filters": [{"paramName": "obs_collection", "values": ["TESS"]},
                                      {"paramName": "dataproduct_type", "values": ["timeseries"]},
                                      {"paramName": "provenance_name", "values": ["SPOC"]}],
                          "position": f"{ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.8f}"}}
    payload = await _mast_invoke(client, request, timeout=tout)
    tolerance = min(match_arcsec, radius_arcsec)
    tic, chosen, others = select_tess_observations(payload.get("data") or [], ra, dec, max_sectors=max_sectors,
                                                   match_arcsec=tolerance)
    provenance: dict[str, Any] = {"service": MAST_INVOKE_URL, "request": request, "tic_id": tic,
                                  "match_radius_arcsec": tolerance,
                                  "sectors_selected": sorted({s for s, _ in chosen}),
                                  "other_tics_in_cone": others}
    notes: list[str] = []
    if others:
        notes.append(f"TESS: {len(others)} other SPOC target(s) in the cone were not used: "
                     + ", ".join(f"TIC {k} ({v:g}\")" for k, v in others.items()))
    if tic is None:
        provenance["elapsed_s"] = round(time.perf_counter() - started, 3)
        why = (f"no SPOC 2-min target within {tolerance:g}\" of the target (J2000.0)" if others
               else "no SPOC 2-min light curve at this position")
        return SurveyResult("tess", [], notes + [f"TESS: {why}."], provenance)
    downloaded: list[tuple[int, str, dict[str, Any]]] = []
    failed: dict[int, str] = {}
    first_error: UpstreamServiceError | None = None
    no_product: list[int] = []
    loop = asyncio.get_running_loop()
    end = None if deadline_s is None else loop.time() + max(deadline_s - TESS_ASSEMBLY_MARGIN_S, 0.0)
    for sector, obs in chosen:
        remaining = None if end is None else end - loop.time()
        if remaining is not None and remaining <= 0:
            failed[sector] = "not fetched before the survey deadline"
            continue
        try:
            async with asyncio.timeout(remaining):
                products = await _mast_invoke(client, {"service": "Mast.Caom.Products", "format": "json",
                                                       "params": {"obsid": str(obs["obsid"])}}, timeout=tout)
                lc = [p for p in products.get("data") or [] if p.get("productSubGroupDescription") == "LC"
                      and str(p.get("productFilename", "")).endswith("_lc.fits")]
                if not lc:
                    no_product.append(sector)
                    continue
                uri = lc[0]["dataURI"]
                response = await _request(client, "GET", MAST_DOWNLOAD_URL, service="tess", params={"uri": uri},
                                          timeout=tout)
                validate_tess_download(response.content, response.headers.get("content-type"))
                parsed = await asyncio.to_thread(parse_tess_lc_fits, response.content)
            downloaded.append((sector, uri, parsed))
        except UpstreamServiceError as exc:
            failed[sector] = exc.message
            first_error = first_error or exc
        except TimeoutError:
            failed[sector] = "no complete answer before the survey deadline"
    if failed and not downloaded:
        if first_error is not None:
            raise first_error
        raise UpstreamServiceError("tess", "no sector downloaded before the survey deadline", retryable=True)
    if no_product:
        notes.append(f"TESS: selected sector(s) {', '.join(str(x) for x in sorted(no_product))} list no SPOC "
                     "light-curve (_lc.fits) product and were skipped.")
    if failed:
        provenance["partial"] = True
        provenance["sectors_failed"] = {str(k): v for k, v in sorted(failed.items())}
        notes.append("TESS: sector(s) " + "; ".join(f"{k} ({v})" for k, v in sorted(failed.items()))
                     + f" failed; the light curve holds sector(s) {', '.join(str(x) for x, _u, _d in downloaded)} "
                       "only (partial result, not cached).")
    points, meta, sector_notes, total, rejected = await asyncio.to_thread(_assemble_tess, downloaded, bin_minutes,
                                                                          include_flagged=include_flagged)
    notes.extend(sector_notes)
    provenance.update({"sectors_used": meta["sectors"], "files": meta["files"],
                       "elapsed_s": round(time.perf_counter() - started, 3)})
    if not points:
        why = ("none of the selected sectors lists a light-curve product" if not downloaded
               else "the light-curve files contain no good-quality cadences")
        return SurveyResult("tess", [], notes + [f"TESS: {why}."], provenance)
    series = LightCurveSeries(
        survey="tess", band="TESS", unit="flux", points=points,
        photometric_system="relative flux (SPOC PDCSAP, or SAP where PDCSAP is unphysical; / sector median)",
        source_ids=[f"TIC {tic}"], n_total=total, n_rejected=rejected,
        metadata={"tic_id": tic, "sectors": meta["sectors"], "cadence_s": 120, "bin_minutes": meta["bin_minutes"],
                  "quality_cut": "QUALITY == 0", "flux_columns": meta["flux_columns"], "crowdsap": meta["crowdsap"],
                  "tic_tess_mag": meta["tic_tess_mag"], "tic_radius_rsun": meta["tic_radius_rsun"],
                  "tic_teff": meta["tic_teff"], "error_scale": ERROR_MODELS[("tess", "TESS")][0],
                  "error_floor_relative_flux": ERROR_MODELS[("tess", "TESS")][1]},
    )
    return SurveyResult("tess", [series], notes, provenance)


# ---------------------------------------------------------------------------
# Variability Metrics
# ---------------------------------------------------------------------------


def stetson_j(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, *, pair_window_days: float = DEFAULT_PAIR_WINDOW_DAYS,
              mean: float | None = None) -> tuple[float | None, int]:
    """Stetson (1996) J index with pairs of observations closer than ``pair_window_days``.

    delta_i = sqrt(n/(n-1)) (y_i - mean)/sigma_i; a pair contributes P = delta_i delta_j,
    an unpaired epoch P = delta_i^2 - 1; J = sum sgn(P) sqrt|P| / n_groups (unit weights).
    Returns (None, 0) when no pair exists (J then reduces to a noise statistic).
    """
    n = len(y)
    if n < 2:
        return None, 0
    if mean is None:
        w = 1.0 / sigma**2
        mean = float(np.sum(w * y) / np.sum(w))
    delta = math.sqrt(n / (n - 1)) * (y - mean) / sigma
    order = np.argsort(t)
    used = np.zeros(n, dtype=bool)
    terms: list[float] = []
    pairs = 0
    for pos, i in enumerate(order):
        if used[i]:
            continue
        j = order[pos + 1] if pos + 1 < n else None
        if j is not None and not used[j] and t[j] - t[i] <= pair_window_days:
            p = float(delta[i] * delta[j])
            used[i] = used[j] = True
            pairs += 1
        else:
            p = float(delta[i] ** 2 - 1.0)
            used[i] = True
        terms.append(math.copysign(math.sqrt(abs(p)), p))
    if pairs == 0:
        return None, 0
    return float(np.mean(terms)), pairs


def excess_variance(y: np.ndarray, sigma: np.ndarray, *, unit: str) -> tuple[float | None, float | None, float | None]:
    """Normalised excess variance and its error (Vaughan et al. 2003, eqs. 8, 10, 11).

    Magnitudes are first converted to relative flux F = 10^(-0.4 (m - median m)) with
    sigma_F = F ln(10)/2.5 sigma_m. Returns (sigma2_NXS, err, F_var) or Nones when the
    mean flux is not positive.
    """
    n = len(y)
    if n < 2:
        return None, None, None
    if unit == "mag":
        f = 10.0 ** (-0.4 * (y - np.median(y)))
        sf = f * sigma / _MAG_PER_FRACTIONAL_FLUX
    else:
        f, sf = y, sigma
    xbar = float(np.mean(f))
    if not xbar > 0:
        return None, None, None
    s2 = float(np.var(f, ddof=1))
    mse = float(np.mean(sf**2))
    nxs = (s2 - mse) / xbar**2
    fvar = math.sqrt(nxs) if nxs > 0 else 0.0
    err = math.sqrt((math.sqrt(2.0 / n) * mse / xbar**2) ** 2 + (math.sqrt(mse / n) * 2.0 * fvar / xbar) ** 2)
    return nxs, err, (fvar if nxs > 0 else None)


def isolated_outliers(y: np.ndarray, sigma: np.ndarray, *, t: np.ndarray | None = None,
                      companions: Sequence[tuple[np.ndarray, np.ndarray]] = (), nsigma: float = OUTLIER_SIGMA,
                      max_fraction: float = OUTLIER_MAX_FRACTION,
                      coincidence_sigma: float = OUTLIER_COINCIDENCE_SIGMA,
                      window_days: float = OUTLIER_COINCIDENCE_DAYS) -> np.ndarray:
    """Boolean mask of epochs treated as isolated outliers (candidate artefacts).

    An epoch deviating from the median by more than ``nsigma`` x sigma is isolated when
    (1) neither the previous nor the next epoch (time order; index order without ``t``)
    deviates in the same direction by more than ``coincidence_sigma``, and (2) with
    ``t``, no other epoch of the series and no epoch of a ``companions`` series -- other
    bands of the same survey as (t, signed z) arrays, z relative to their own median and
    error model -- within ``window_days`` does. Isolated epochs are removed only when
    there are at most floor(``max_fraction`` x N) of them (no minimum: short series keep
    every epoch); more extreme epochs are a property of the source (eclipses, dips,
    flares) and are never removed. Non-isolated extreme epochs are never removed. This
    replaces a MAD-based "robust scatter" veto (a MAD ignores anything affecting less
    than half of the epochs) and an allowance of max(3, 1 %) that dropped 60 % of a
    5-epoch series and the in-eclipse transits of detached binaries.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(y)
    none = np.zeros(n, dtype=bool)
    allowed = math.floor(max_fraction * n)
    if n == 0 or allowed < 1:
        return none
    z = (y - np.median(y)) / sigma
    extreme = np.abs(z) > nsigma
    if not np.any(extreme):
        return none
    order = np.argsort(t, kind="stable") if t is not None else np.arange(n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    isolated = extreme.copy()
    for i in np.flatnonzero(extreme):
        sign = math.copysign(1.0, z[i])
        r = rank[i]
        for nb in (r - 1, r + 1):
            if 0 <= nb < n and sign * z[order[nb]] > coincidence_sigma:
                isolated[i] = False
        if not isolated[i] or t is None:
            continue
        near = (np.abs(t - t[i]) <= window_days) & (np.arange(n) != i)
        if np.any(sign * z[near] > coincidence_sigma):
            isolated[i] = False
            continue
        for tc, zc in companions:
            close = np.abs(np.asarray(tc) - t[i]) <= window_days
            if np.any(sign * np.asarray(zc)[close] > coincidence_sigma):
                isolated[i] = False
                break
    count = int(np.sum(isolated))
    if count == 0 or count > allowed:
        return none
    return isolated


def variability_metrics(t: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray,
                        dy: Sequence[float] | np.ndarray, *, unit: str = "mag", error_floor: float = 0.0,
                        error_scale: float = 1.0, pair_window_days: float = DEFAULT_PAIR_WINDOW_DAYS,
                        decisive: bool = True, not_decisive_reason: str | None = None,
                        companions: Sequence[tuple[np.ndarray, np.ndarray]] = ()) -> VariabilityMetrics:
    """Variability statistics and a variable/non-variable decision for one series.

    Per-epoch errors follow the calibrated survey error model
    ``sigma_eff = sqrt((error_scale * dy)^2 + error_floor^2)`` (see :data:`ERROR_MODELS`).
    Up to 1 % of N isolated > 5-sigma outliers are excluded from the chi^2
    (:func:`isolated_outliers`; ``companions`` are the other bands of the survey as
    (t, signed z) arrays, used to recognise coincident deviations). The series is
    variable when chi^2 vs the weighted mean rejects constancy at >= 5 sigma AND chi^2/dof >= 2 -- the intrinsic variance
    (excess over the noise, cf. the normalised excess variance of Vaughan et al. 2003)
    is at least the calibrated noise variance. The effect-size condition keeps very
    long series of constant stars with slightly mis-modelled errors from being flagged
    (the chi^2 test alone becomes arbitrarily sensitive; Sokolovsky et al. 2017, MNRAS
    464, 274, sec. 2, on the need to compare indices with constant stars); it is
    calibrated on 81 constant Stripe 82 standards (2 of 225 ZTF series flagged).
    Low-amplitude periodic variables that fail it can still qualify through a
    periodogram peak (see :func:`search_periods`).

    Descriptive statistics (std, MAD, 5-95 % amplitude, von Neumann eta, Stetson J,
    excess variance) use all epochs.
    """
    ta, ya, ea = (np.asarray(a, dtype=float) for a in (t, y, dy))
    keep = np.isfinite(ta) & np.isfinite(ya) & np.isfinite(ea) & (ea > 0)
    ta, ya, ea = ta[keep], ya[keep], ea[keep]
    order = np.argsort(ta)
    ta, ya, ea = ta[order], ya[order], ea[order]
    n = len(ya)
    metrics = VariabilityMetrics(n=n, unit=unit, error_floor=error_floor, error_scale=error_scale)
    if n == 0:
        metrics.evidence.append("no usable epochs")
        return metrics
    sigma = np.sqrt((error_scale * ea) ** 2 + error_floor**2)
    w = 1.0 / sigma**2
    wmean = float(np.sum(w * ya) / np.sum(w))
    metrics.time_span_days = float(ta[-1] - ta[0])
    metrics.weighted_mean = wmean
    metrics.weighted_mean_error = float(1.0 / math.sqrt(np.sum(w)))
    metrics.mean = float(np.mean(ya))
    metrics.median = float(np.median(ya))
    metrics.median_error = float(np.median(ea))
    if n < 2:
        metrics.evidence.append("single epoch: variability undetermined")
        return metrics
    metrics.std = float(np.std(ya, ddof=1))
    metrics.mad_std = float(stats.median_abs_deviation(ya, scale="normal"))
    lo, hi = np.percentile(ya, [5.0, 95.0])
    metrics.amplitude_5_95 = float(hi - lo)
    var = float(np.var(ya, ddof=1))
    if var > 0:
        metrics.von_neumann_eta = float(np.sum(np.diff(ya) ** 2) / (n - 1) / var)
    metrics.stetson_j, metrics.stetson_n_pairs = stetson_j(ta, ya, sigma, pair_window_days=pair_window_days, mean=wmean)
    metrics.excess_variance, metrics.excess_variance_error, metrics.fractional_variability = excess_variance(ya, sigma, unit=unit)
    outliers = (isolated_outliers(ya, sigma, t=ta, companions=companions) if n >= MIN_POINTS_VARIABILITY
                else np.zeros(n, dtype=bool))
    metrics.n_outliers_excluded = int(np.sum(outliers))
    yc, sc = ya[~outliers], sigma[~outliers]
    wc = 1.0 / sc**2
    wmean_c = float(np.sum(wc * yc) / np.sum(wc))
    chi2_val = float(np.sum(((yc - wmean_c) / sc) ** 2))
    dof = len(yc) - 1
    metrics.chi2, metrics.dof, metrics.chi2_dof = chi2_val, dof, chi2_val / dof
    log_p = float(stats.chi2.logsf(chi2_val, dof))
    metrics.chi2_pvalue = math.exp(log_p) if log_p > -745 else 0.0
    metrics.significance_sigma = chi2_significance(chi2_val, dof)
    noise_var = float(np.median(sc**2))
    metrics.noise_rms = math.sqrt(noise_var)
    metrics.intrinsic_scatter = math.sqrt(max(float(np.var(yc, ddof=1)) - noise_var, 0.0))
    u_ = unit
    if n < MIN_POINTS_VARIABILITY:
        metrics.evidence.append(f"only {n} epochs (< {MIN_POINTS_VARIABILITY}): variability undetermined")
        return metrics
    significant = metrics.chi2_pvalue < FIVE_SIGMA_PVALUE
    strong = metrics.chi2_dof >= VARIABLE_CHI2_DOF
    model_txt = ""
    if error_scale != 1.0 or error_floor:
        model_txt = f" (errors x{error_scale:g} (+) {error_floor:g} {u_} floor)"
    metrics.evidence.append(
        f"chi2/dof = {metrics.chi2_dof:.2f} over {len(yc)} epochs{model_txt}: constancy "
        + (f"rejected at {metrics.significance_sigma:.1f} sigma" if significant
           else f"not rejected at 5 sigma ({metrics.significance_sigma:.1f} sigma)"))
    if metrics.n_outliers_excluded:
        metrics.evidence.append(f"{metrics.n_outliers_excluded} isolated >{OUTLIER_SIGMA:g}-sigma epoch(s) excluded "
                                "from the chi2 as possible outliers")
    metrics.evidence.append(
        f"intrinsic rms {metrics.intrinsic_scatter:.4g} {u_} vs per-epoch noise {metrics.noise_rms:.4g} {u_}"
        + (f" (chi2/dof >= {VARIABLE_CHI2_DOF:g}: variance dominated by the source)" if strong
           else f" (chi2/dof < {VARIABLE_CHI2_DOF:g})"))
    metrics.evidence.append(f"5-95% amplitude {metrics.amplitude_5_95:.4g} {u_}")
    if metrics.von_neumann_eta is not None:
        metrics.evidence.append(f"von Neumann eta = {metrics.von_neumann_eta:.3f} (~2 for uncorrelated noise; "
                                "small values indicate smooth, time-correlated variability)")
    if metrics.stetson_j is not None:
        metrics.evidence.append(f"Stetson J = {metrics.stetson_j:.3f} from {metrics.stetson_n_pairs} close pairs")
    if not decisive:
        metrics.evidence.append(not_decisive_reason or "series not used for the variability decision")
        return metrics
    metrics.is_variable = bool(significant and strong)
    metrics.decision_basis = "chi2" if metrics.is_variable else None
    return metrics


def chi2_significance(chi2_value: float, dof: int) -> float:
    """Gaussian-equivalent signed significance of a chi^2 deviation (finite for any input).

    z = Phi^-1(1 - p) with p the chi^2 survival probability: positive for an excess,
    negative when chi^2 < dof (over-estimated errors). Both tails are inverted from
    logarithms so neither p -> 0 nor p -> 1 overflows; beyond double-precision underflow
    of ln p the Wilson & Hilferty (1931) cube-root normal approximation is used.
    """
    k = float(dof)
    if k <= 0 or not math.isfinite(chi2_value):
        raise ValueError("chi2_significance needs dof > 0 and a finite chi2")
    log_sf = float(stats.chi2.logsf(chi2_value, dof))
    if math.isfinite(log_sf) and log_sf <= math.log(0.5):
        return _sigma_from_log_sf(log_sf)
    log_cdf = float(stats.chi2.logcdf(chi2_value, dof))
    if math.isfinite(log_cdf) and log_cdf < math.log(0.5):
        return -_sigma_from_log_sf(log_cdf)
    if math.isfinite(log_sf) and math.isfinite(log_cdf):
        return float(stats.norm.isf(math.exp(log_sf)))
    # Either tail underflows (chi2 = 0, or p beyond double precision): Wilson-Hilferty.
    return float(((max(chi2_value, 0.0) / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k)))


def _sigma_from_log_sf(log_p: float) -> float:
    """Gaussian-equivalent one-sided significance for ln(p), p <= 0.5, stable for tiny p."""
    if log_p > -700:
        return float(stats.norm.isf(math.exp(log_p)))
    # Asymptotic inversion of the Gaussian tail: ln p ~ -z^2/2 - ln(z sqrt(2 pi)).
    z = math.sqrt(-2.0 * log_p)
    for _ in range(5):
        z = math.sqrt(max(-2.0 * log_p - 2.0 * math.log(z * math.sqrt(2.0 * math.pi)), 1.0))
    return z


def series_error_model(series: LightCurveSeries) -> tuple[float, float]:
    """(scale, floor) of the calibrated per-epoch error model of a series (see :data:`ERROR_MODELS`);
    the floor is in the series' unit (mag, or relative flux for TESS)."""
    if series.survey == "gaia" and series.band == "G" and series.unit == "mag":
        return gaia_g_error_model(_num(series.metadata.get("phot_g_mean_mag")))
    return ERROR_MODELS.get((series.survey, series.band), (1.0, 0.0))


def series_error_floor(series: LightCurveSeries) -> float:
    return series_error_model(series)[1]


def _signed_deviations(series: LightCurveSeries) -> tuple[np.ndarray, np.ndarray]:
    """(t, (y - median) / sigma_eff) of a series' good epochs, sigma from its error model."""
    t, y, dy = series.good_arrays()
    if len(t) == 0:
        return t, y
    scale, floor = series_error_model(series)
    return t, (y - np.median(y)) / np.sqrt((scale * dy) ** 2 + floor**2)


def series_variability(series: LightCurveSeries, *, pair_window_days: float = DEFAULT_PAIR_WINDOW_DAYS,
                       companions: Sequence[LightCurveSeries] = ()) -> VariabilityMetrics:
    """:func:`variability_metrics` of a series with its survey error model; ``companions``
    (other bands of the same survey) let coincident deviations protect real events from
    the outlier cut."""
    t, y, dy = series.good_arrays()
    scale, floor = series_error_model(series)
    reason = NON_DECISIVE_SERIES.get((series.survey, series.band))
    comp = [_signed_deviations(c) for c in companions if c.survey == series.survey and c is not series]
    return variability_metrics(t, y, dy, unit=series.unit, error_floor=floor, error_scale=scale,
                               pair_window_days=pair_window_days, decisive=reason is None, not_decisive_reason=reason,
                               companions=[c for c in comp if len(c[0])])


# ---------------------------------------------------------------------------
# Period Search (Lomb-Scargle)
# ---------------------------------------------------------------------------


def frequency_grid(baseline_days: float, *, min_period_days: float = DEFAULT_MIN_PERIOD_DAYS,
                   max_period_days: float | None = None, oversampling: float = DEFAULT_OVERSAMPLING,
                   max_frequencies: int = MAX_FREQUENCIES, min_oversampling: float = MIN_OVERSAMPLING,
                   ) -> tuple[np.ndarray, float]:
    """Uniform frequency grid (cycles/day) with spacing 1/(oversampling * baseline).

    f_min = 1 / min(max_period, baseline); f_max = 1 / min_period. When the grid would
    exceed ``max_frequencies`` the oversampling is reduced (returned as the 2nd value);
    below ``min_oversampling`` samples per peak (narrow peaks could fall between grid
    points) a ValueError is raised instead of searching a grid that cannot resolve peaks.
    """
    if not math.isfinite(baseline_days) or baseline_days <= 0:
        raise ValueError("baseline must be positive")
    if not math.isfinite(min_period_days) or min_period_days <= 0:
        raise ValueError("min_period_days must be positive and finite")
    if max_period_days is not None and not math.isfinite(max_period_days):
        raise ValueError("max_period_days must be finite")
    longest = baseline_days if max_period_days is None else min(max_period_days, baseline_days)
    fmin, fmax = 1.0 / longest, 1.0 / min_period_days
    if fmax <= fmin:
        raise ValueError("min_period_days must be shorter than the longest searchable period")
    n = math.ceil((fmax - fmin) * oversampling * baseline_days) + 1
    if n > max_frequencies:
        oversampling = (max_frequencies - 1) / ((fmax - fmin) * baseline_days)
        n = max_frequencies
        if oversampling < min_oversampling:
            raise ValueError(
                f"a {baseline_days:.0f}-d baseline down to {min_period_days:g} d needs more than {max_frequencies} "
                f"frequencies for {min_oversampling:g} samples per peak; increase min_period_days")
    return np.linspace(fmin, fmax, n), oversampling


def ls_power(ls: LombScargle, freq: np.ndarray, *, chunk: int = LS_CHUNK_FREQUENCIES) -> np.ndarray:
    """Periodogram power on a regular grid, evaluated in regular sub-grids.

    astropy's fast (NUFFT) implementation allocates O(n_freq x 14) complex work
    arrays (~100 MB for 3e5 frequencies); evaluating contiguous slices of the same
    uniform grid gives identical powers with bounded memory.
    """
    out = np.empty(len(freq), dtype=float)
    for start in range(0, len(freq), chunk):
        part = freq[start:start + chunk]
        out[start:start + len(part)] = ls.power(part, assume_regular_frequency=len(part) > 1)
    return out


def _top_peaks(freq: np.ndarray, power: np.ndarray, *, n_peaks: int, min_separation: float) -> list[dict[str, float]]:
    if len(power) < 3:
        return []
    interior = np.where((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:]))[0] + 1
    order = interior[np.argsort(power[interior])[::-1]]
    peaks: list[dict[str, float]] = []
    for idx in order:
        f = float(freq[idx])
        if all(abs(f - p["frequency_per_day"]) > min_separation for p in peaks):
            peaks.append({"period_days": 1.0 / f, "frequency_per_day": f, "power": float(power[idx])})
        if len(peaks) >= n_peaks:
            break
    return peaks


# -- multi-harmonic (Fourier series) models ----------------------------------------


def _harmonic_design(t: np.ndarray, freqs: Sequence[float]) -> np.ndarray:
    cols = [np.ones_like(t)]
    for f in freqs:
        phase = 2.0 * np.pi * f * t
        cols += [np.sin(phase), np.cos(phase)]
    return np.column_stack(cols)


def _harmonic_fit(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, freqs: Sequence[float]) -> tuple[float, np.ndarray]:
    """Weighted least-squares Fourier model (constant + sin/cos at ``freqs``): (chi^2, coefficients)."""
    tc = t - t.mean()  # conditioning: large absolute times do not change the model
    design = _harmonic_design(tc, freqs) / sigma[:, None]
    coef, *_ = np.linalg.lstsq(design, y / sigma, rcond=None)
    resid = y / sigma - design @ coef
    return float(resid @ resid), coef


def _bic(chi2_value: float, n: int, n_params: int) -> float:
    """Gaussian BIC with the noise scale estimated from the data (n ln(chi2/n) + k ln n;
    Schwarz 1978): invariant to a common error scale, so mis-scaled errors do not bias it."""
    return n * math.log(max(chi2_value, 1e-300) / n) + n_params * math.log(n)


def best_n_harmonics(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, *,
                     max_harmonics: int = MAX_HARMONICS) -> int:
    """Number of harmonics of ``frequency`` minimising the BIC (1..max_harmonics)."""
    n = len(t)
    best_k, best_bic = 1, math.inf
    for k in range(1, max_harmonics + 1):
        if 2 * k + 1 >= n / 3:
            break
        chi2_k, _ = _harmonic_fit(t, y, sigma, [j * frequency for j in range(1, k + 1)])
        b = _bic(chi2_k, n, 2 * k + 1)
        if b < best_bic:
            best_k, best_bic = k, b
    return best_k


def odd_harmonic_delta_bic(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float) -> float:
    """BIC gain from folding at 2P: a Fourier model at ``frequency`` (K harmonics, K by
    BIC) versus the same model plus the first two odd harmonics of ``frequency``/2.

    The odd harmonics of f/2 are what distinguishes consecutive cycles, e.g. the unequal
    minima of an eclipsing or ellipsoidal binary whose single-harmonic Lomb-Scargle peak
    lies at half the orbital period. Positive values favour 2P.
    """
    n = len(t)
    k = best_n_harmonics(t, y, sigma, frequency)
    base = [j * frequency for j in range(1, k + 1)]
    chi2_base, _ = _harmonic_fit(t, y, sigma, base)
    chi2_odd, _ = _harmonic_fit(t, y, sigma, base + [0.5 * frequency, 1.5 * frequency])
    return _bic(chi2_base, n, 2 * k + 1) - _bic(chi2_odd, n, 2 * k + 5)


def refine_period(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, step: float, *,
                  n_harmonics: int | None = None, max_harmonics: int = MAX_HARMONICS,
                  n_grid: int = 201, bounds: tuple[float, float] | None = None) -> tuple[float, float | None, int]:
    """Refine a frequency by minimising the chi^2 of a multi-harmonic model.

    The chi^2 is evaluated on ``n_grid`` frequencies spanning +-``step`` around
    ``frequency`` -- clipped to ``bounds`` (the searched frequency range; a refinement
    must not leave it, e.g. drift to periods longer than the baseline) and always to
    positive frequencies -- and a parabola is fitted through the minimum; the number of harmonics
    is chosen by BIC (up to ``max_harmonics``) at the starting frequency. Returns
    (frequency, 1-sigma frequency error, number of harmonics). The error is the
    Cramer-Rao bound of the K-harmonic model in white noise, sigma_f = sigma_res /
    (pi sqrt(2 N var(t) sum_j j^2 A_j^2)), with A_j the semi-amplitude of harmonic j and
    sigma_res the rms of the residuals. For one harmonic and uniform sampling (var(t) =
    T^2/12) it is Montgomery & O'Donoghue (1999, DSSN 13, 28), sqrt(6/N) sigma_res /
    (pi T A_1); summing the harmonics matters when the fundamental is weak, e.g. an
    eclipsing binary with equal minima at its orbital period (A_1 ~ 0), whose frequency
    is fixed by the harmonics. Correlated residuals are accounted for by the caller
    (:func:`_complete_period` inflates the error by the square root of the variance
    inflation of :func:`period_significance`), a wrong cycle count across gaps by
    :func:`alias_comb`.
    """
    k = n_harmonics or best_n_harmonics(t, y, sigma, frequency, max_harmonics=max_harmonics)
    lo, hi = frequency - step, frequency + step
    if bounds is not None:
        lo, hi = max(lo, min(bounds[0], frequency)), min(hi, max(bounds[1], frequency))
    lo = max(lo, 0.5 * frequency) if lo <= 0 else lo
    grid = np.linspace(lo, hi, n_grid)
    chi2s = np.array([_harmonic_fit(t, y, sigma, [j * f for j in range(1, k + 1)])[0] for f in grid])
    i = int(np.argmin(chi2s))
    best = float(grid[i])
    if 0 < i < len(grid) - 1:
        c0, c1, c2 = chi2s[i - 1], chi2s[i], chi2s[i + 1]
        denom = c0 - 2.0 * c1 + c2
        if denom > 0:
            best = float(grid[i] + 0.5 * (grid[1] - grid[0]) * (c0 - c2) / denom)
    _chi2, coef = _harmonic_fit(t, y, sigma, [j * best for j in range(1, k + 1)])
    tc = t - t.mean()
    resid = y - _harmonic_design(tc, [j * best for j in range(1, k + 1)]) @ coef
    # Fisher information of the frequency summed over the harmonics: harmonic j of
    # semi-amplitude A_j contributes 2 pi^2 j^2 A_j^2 N var(t) / sigma^2.
    harmonic_power = sum(j * j * (coef[2 * j - 1] ** 2 + coef[2 * j] ** 2) for j in range(1, k + 1))
    time_var = float(np.var(t))
    sigma_f = None
    if harmonic_power > 0 and time_var > 0:
        sigma_f = float(np.std(resid)) / (math.pi * math.sqrt(2.0 * len(t) * time_var * float(harmonic_power)))
    return best, sigma_f, k


def block_frequency_error(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, n_harmonics: int, *,
                          n_blocks: int = FREQUENCY_ERROR_BLOCKS) -> float | None:
    """Cluster-robust (sandwich) 1-sigma error of a K-harmonic frequency estimate.

    The model is linearised in frequency: the design holds the constant, the sin/cos
    columns of every harmonic and d(model)/df = sum_j 2 pi j (t - tbar) (a_j cos - b_j sin).
    Its weighted least-squares covariance A^-1 (A = X^T W X) is the white-noise Cramer-Rao
    bound; the sandwich A^-1 B A^-1 with B = sum_g (X_g^T W_g r_g)(X_g^T W_g r_g)^T over
    G contiguous time blocks of equal size (Liang & Zeger 1986, Biometrika 73, 13; with
    the small-sample factor G/(G-1); G = ``n_blocks``, and half and a quarter of that,
    whichever gives the largest error) is valid whatever the residual correlation
    within a block -- red noise, Blazhko modulation, night-to-night zero points, a
    wandering phase (O-C) of spotted stars or of stars whose period changes. It reduces to
    the white-noise bound when the residuals are independent. Returns None when the
    design is singular or there are too few points.
    """
    n = len(t)
    if n < max(3 * (2 * n_harmonics + 2), 2 * n_blocks):
        return None
    order = np.argsort(t)
    ts, ys, ss = t[order], y[order], sigma[order]
    tc = ts - ts.mean()
    freqs = [j * frequency for j in range(1, n_harmonics + 1)]
    _chi2, coef = _harmonic_fit(ts, ys, ss, freqs)
    design = _harmonic_design(tc, freqs)
    resid = ys - design @ coef
    deriv = np.zeros_like(tc)
    for j in range(1, n_harmonics + 1):
        phase = 2.0 * np.pi * j * frequency * tc
        deriv += 2.0 * np.pi * j * tc * (coef[2 * j - 1] * np.cos(phase) - coef[2 * j] * np.sin(phase))
    x = np.column_stack([design, deriv])
    w = 1.0 / ss**2
    a = x.T @ (x * w[:, None])
    try:
        a_inv = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return None
    scores = x * (w * resid)[:, None]
    # Blocking analysis (Flyvbjerg & Petersen 1989, J. Chem. Phys. 91, 461): the estimate
    # grows with the block length until blocks outlast the residual correlations, so the
    # largest over G, G/2 and G/4 blocks is used (e.g. Blazhko cycles of ~200 d).
    var_f = 0.0
    for g_blocks in sorted({n_blocks, max(n_blocks // 2, 3), max(n_blocks // 4, 3)}):
        b = np.zeros_like(a)
        for block in np.array_split(np.arange(n), g_blocks):
            g = scores[block].sum(axis=0)
            b += np.outer(g, g)
        cov = a_inv @ b @ a_inv * g_blocks / (g_blocks - 1)
        var_f = max(var_f, float(cov[-1, -1]))
    return math.sqrt(var_f) if var_f > 0 and math.isfinite(var_f) else None


# -- cycle-to-cycle (parity) test and light-curve shape -----------------------------------


def _phase_profile(phase: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Median profile in ``n_bins`` phase bins, empty bins filled by periodic interpolation."""
    b = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
    prof = np.full(n_bins, np.nan)
    order = np.argsort(b, kind="stable")
    bounds = np.searchsorted(b[order], np.arange(n_bins + 1))
    for i in range(n_bins):
        chunk = y[order[bounds[i]:bounds[i + 1]]]
        if len(chunk):
            prof[i] = float(np.median(chunk))
    ok = np.isfinite(prof)
    idx = np.arange(n_bins)
    return np.interp(idx, idx[ok], prof[ok], period=n_bins)


def _profile_model(phase: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Periodic piecewise-linear interpolation of the binned median profile at ``phase``."""
    prof = _phase_profile(phase, y, n_bins)
    centres = (np.arange(n_bins) + 0.5) / n_bins
    return np.interp(phase, np.concatenate([centres - 1.0, centres, centres + 1.0]), np.tile(prof, 3))


def observing_runs(t: np.ndarray, *, gap_days: float = PARITY_RUN_GAP_DAYS) -> np.ndarray:
    """Run index of each epoch (in the order of ``t``): a new run starts after a gap longer
    than ``gap_days`` (one night of ground-based photometry; a TESS sector is one run)."""
    order = np.argsort(t, kind="stable")
    runs = np.empty(len(t), dtype=np.int64)
    runs[order] = np.concatenate([[0], np.cumsum(np.diff(t[order]) > gap_days)]) if len(t) else []
    return runs


def parity_test(band_data: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], frequency: float,
                ) -> dict[str, float] | None:
    """Do consecutive cycles at ``frequency`` differ (i.e. is the true period 2P)?

    Non-parametric analysis of variance in phase (cf. Schwarzenberg-Czerny 1989, MNRAS
    241, 153) that works for any light-curve shape, including the narrow eclipses a
    low-order Fourier model cannot represent. For each band, the data are folded at
    P = 1/frequency and split by cycle parity (even/odd cycles, i.e. the two halves of
    a fold at 2P). A fine binned-median profile at P is subtracted first (so steep
    branches do not bias the comparison); then in every phase bin the residuals of even
    and odd cycles are compared with a one-way ANOVA F test, whose p-value uses that
    bin's own scatter (eclipse bins are far noisier than out-of-eclipse bins). The
    per-bin p-values are combined as sum chi2_1-quantiles (a chi^2 with one degree of
    freedom per bin), giving ``sigma`` (Gaussian-equivalent significance).

    Cluster-robust: parity is a property of a cycle, and the epochs of one cycle taken in
    one observing run (:func:`observing_runs`; e.g. the 2-4 same-night ZTF exposures that
    fall into one phase bin) share that cycle's deviation from the mean profile -- the
    amplitude/phase modulation of a Blazhko RR Lyrae, spots, a night's zero point. They
    are one observation of the cycle, not independent ones, so each (run, cycle)
    cluster's mean residual in a bin is one ANOVA sample (treating them as independent
    doubled the periods of modulated pulsators up to 8x, e.g. V354 Lyr).

    ``effect`` is the rms of the even-odd difference of the bin means (noise-debiased)
    relative to the rms of the profile, i.e. how different consecutive cycles are in
    units of the variability amplitude (unequal eclipse depths give 0.1-1; cycle-to-cycle
    systematics of dense space photometry < 0.01). Returns None when fewer than 4 bins
    can be compared.
    """
    q_total = 0.0
    dof = 0
    num = den = 0.0
    for t, y, _s in band_data:
        if len(t) < MIN_POINTS_PERIOD:
            continue
        x = (t - t.min()) * frequency
        phase = x % 1.0
        cycle = np.floor(x).astype(np.int64)
        resid = y - _profile_model(phase, y, int(np.clip(len(t) // 6, 10, 200)))
        mad = float(stats.median_abs_deviation(resid, scale="normal"))
        if mad > 0:  # single wild epochs must not dominate a bin's comparison
            resid = np.clip(resid, -8.0 * mad, 8.0 * mad)
        n_bins = int(np.clip(len(t) // 12, 6, 60))
        b = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
        # One ANOVA sample per (phase bin, observing run, cycle) cluster.
        _keys, cluster = np.unique(np.column_stack([b, observing_runs(t), cycle]), axis=0, return_inverse=True)
        cluster = cluster.ravel()
        counts = np.bincount(cluster)
        c_resid = np.bincount(cluster, weights=resid) / counts
        c_bin = np.bincount(cluster, weights=b, minlength=len(counts)) / counts
        c_parity = (np.bincount(cluster, weights=cycle % 2) / counts).astype(np.int64)
        c_bin = np.rint(c_bin).astype(np.int64)
        diffs: list[float] = []
        levels: list[float] = []
        for i in range(n_bins):
            in_bin = c_bin == i
            even_idx, odd_idx = in_bin & (c_parity == 0), in_bin & (c_parity == 1)
            if np.sum(even_idx) < 2 or np.sum(odd_idx) < 2:
                continue
            p = float(stats.f_oneway(c_resid[even_idx], c_resid[odd_idx]).pvalue)
            if math.isfinite(p):
                q_total += float(stats.chi2.isf(max(p, 1e-300), 1))
                dof += 1
            # Effect size on the profile-subtracted residuals: the profile removes the part of
            # an even-odd difference due to the two parities sampling different sub-phases of
            # a steep bin, and the bin's scatter about the profile debiases d^2.
            re_, ro_ = c_resid[even_idx], c_resid[odd_idx]
            d = float(re_.mean() - ro_.mean())
            var_d = float(re_.var(ddof=1) / len(re_) + ro_.var(ddof=1) / len(ro_))
            diffs.append(d * d - var_d)
            levels.append(float(y[b == i].mean()))
        if len(levels) > 2:
            num += max(float(np.mean(diffs)), 0.0) / 4.0 * len(levels)
            den += float(np.var(levels)) * len(levels)
    if dof < 4:
        return None
    log_p = float(stats.chi2.logsf(q_total, dof))
    if not math.isfinite(log_p):
        log_p = -1.0e6  # beyond double precision: capped
    sigma = _sigma_from_log_sf(log_p) if log_p < math.log(0.5) else 0.0
    return {"sigma": round(sigma, 2), "effect": round(math.sqrt(num / den) if den > 0 else 0.0, 4), "n_bins": dof}


def eclipse_shape(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, *, unit: str = "mag",
                  ) -> dict[str, Any]:
    """Shape of the light curve folded at ``frequency``: is it a single narrow dip per cycle?

    The binned median profile is expressed as "fainter is larger" (magnitudes, or minus
    the flux). With L the median of the profile (the out-of-eclipse level for a detached
    binary) and D = max - L the dip depth, the fold is eclipse-like when D is at least
    ``ECLIPSE_MIN_DEPTH_FRACTION`` of the full profile range (the source spends most of the
    cycle at maximum light, unlike pulsators, whose median lies near minimum light, or
    sinusoids, D = half the range), the phase fraction deeper than D/2 is at most
    ``ECLIPSE_MAX_WIDTH`` (narrow dips), the deep part forms exactly one contiguous dip,
    and D exceeds 5 times the noise of a bin median; dips are counted above the level by
    max(5 noise, :data:`DIP_MIN_FRACTION` D). One narrow dip per cycle is the
    signature of an eclipsing binary whose two eclipses are (nearly) equal -- its orbital
    period is 2P -- or of a single visible eclipse per orbit (transit, faint secondary),
    which the shape alone cannot tell apart. ``depth_relative_flux`` (the dip as a
    fraction of the median flux) separates them in the period ladder: see
    :data:`ECLIPSE_MIN_DOUBLING_DEPTH`.

    The noise of a bin median is the largest of its formal error, the scatter of the epochs
    about the binned profile, and -- with at least :data:`ECLIPSE_EMPIRICAL_MIN_BINS` bins --
    the bin-to-bin differences of the profile. A secondary eclipse is also looked for
    explicitly epoch by epoch (:func:`_secondary_eclipse`): it lies half a cycle from the
    primary and may fill a single bin of a sparse fold. A detected secondary makes the fold
    two-dipped (``n_dips`` >= 2): the period is then the orbital one.
    """
    d = y if unit == "mag" else -y
    n_bins = int(np.clip(len(t) // 8, 10, 60))
    phase = ((t - t.min()) * frequency) % 1.0
    prof = _phase_profile(phase, d, n_bins)
    span = float(prof.max() - prof.min())
    level = float(np.median(prof))
    depth = float(prof.max() - level)
    deep = prof > level + depth / 2.0
    per_bin = math.sqrt(max(len(t) / n_bins, 1.0))
    formal = float(np.median(sigma)) / per_bin * 1.2533  # error of a median
    # Scatter of the epochs about the binned profile (robust; spots, flares and red noise
    # included), as the error of a bin median.
    resid = d - _profile_model(phase, d, n_bins)
    scatter = float(stats.median_abs_deviation(resid, scale="normal"))
    noise = max(formal, 1.2533 * scatter / per_bin)
    if n_bins >= ECLIPSE_EMPIRICAL_MIN_BINS:
        # Empirical bin-to-bin noise of a well-resolved profile (robust: MAD of first differences
        # / sqrt 2): coherent bumps (CM Dra in ZTF g: a 0.03-mag bump next to the 0.6-mag
        # eclipse counted as a second dip) scatter bin medians more than independent epochs would.
        # A coarse profile's differences measure the light curve itself, not noise.
        empirical = 1.4826 * float(np.median(np.abs(np.diff(np.concatenate([prof, prof[:1]]))))) / math.sqrt(2.0)
        noise = max(noise, empirical)
    # Dips are counted at a low threshold, so that a shallow secondary eclipse (e.g.
    # the 0.06-mag secondary of Algol next to its 1.3-mag primary) counts as a second dip.
    dipping = prof > level + max(5.0 * noise, DIP_MIN_FRACTION * depth)
    n_dips = int(np.sum(dipping & ~np.roll(dipping, 1)))
    if dipping.all():
        n_dips = 0
    secondary = None
    if n_dips == 1 and depth > 0:
        secondary = _secondary_eclipse(phase, d, sigma, prof, level=level, depth=depth)
        if secondary is not None and secondary["detected"]:
            n_dips = 2
    depth_flux = (1.0 - 10.0 ** (-0.4 * depth)) if unit == "mag" else depth / max(float(np.median(y)), 1e-300)
    out = {"depth": round(depth, 5), "depth_relative_flux": round(depth_flux, 5),
           "depth_fraction": round(depth / span, 3) if span > 0 else 0.0,
           "width": round(float(deep.mean()), 3), "n_dips": n_dips,
           "depth_over_noise": round(depth / noise, 1) if noise > 0 else None,
           "bin_noise": round(noise, 6), "secondary": secondary}
    out["eclipse_like"] = bool(span > 0 and depth / span >= ECLIPSE_MIN_DEPTH_FRACTION
                               and deep.mean() <= ECLIPSE_MAX_WIDTH and n_dips == 1
                               and noise > 0 and depth >= 5.0 * noise)
    return out


def _secondary_eclipse(phase: np.ndarray, d: np.ndarray, sigma: np.ndarray, prof: np.ndarray, *, level: float,
                       depth: float) -> dict[str, Any] | None:
    """Epoch-level search for a secondary eclipse opposite the primary of a single-dip fold.

    ``d`` is "fainter is larger" (see :func:`eclipse_shape`), ``prof`` its binned profile.
    The primary's centre is the circular mean phase of the epochs deeper than half its
    depth. Boxes of the primary's width (at least 0.04 in phase) centred within
    +-:data:`SECONDARY_WINDOW` of the opposite phase are compared with the neighbouring
    phases (0.5 +- [window + box/2, 0.25] from the primary): a mean excess of at least
    :data:`SECONDARY_SIGMA` standard errors -- scatter of the neighbouring epochs (at
    least the median per-epoch error) -- and at least :data:`DIP_MIN_FRACTION` of the
    primary depth is a secondary eclipse (or, equivalently for the period, the ellipsoidal
    minimum at the second conjunction: a fold at half the orbital period has its maximum
    there). Returns None when the neighbouring phases hold fewer than 3 epochs.
    """
    n_bins = len(prof)
    deep_pts = d > level + depth / 2.0
    ang = 2.0 * np.pi * (phase[deep_pts] if np.any(deep_pts) else np.array([(np.argmax(prof) + 0.5) / n_bins]))
    centre = float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) / (2.0 * np.pi)) % 1.0
    rel = (phase - centre) % 1.0
    width = float(np.mean(prof > level + depth / 2.0))
    box = max(width, 0.04)
    off = np.abs(rel - 0.5)
    shoulder = (off > SECONDARY_WINDOW + box / 2.0) & (off <= 0.25)
    if int(np.sum(shoulder)) < 3:
        return None
    base = float(np.median(d[shoulder]))
    scatter = max(float(stats.median_abs_deviation(d[shoulder], scale="normal")), float(np.median(sigma)))
    se_base = 1.2533 * scatter / math.sqrt(float(np.sum(shoulder)))
    best: dict[str, Any] | None = None
    for c in np.arange(0.5 - SECONDARY_WINDOW, 0.5 + SECONDARY_WINDOW + 1e-9, box / 4.0):
        inside = np.abs(rel - c) <= box / 2.0
        n_in = int(np.sum(inside))
        if n_in < 2:
            continue
        excess = float(np.mean(d[inside])) - base
        z = excess / math.hypot(scatter / math.sqrt(n_in), se_base)
        if best is None or z > best["sigma"]:
            best = {"phase": round(float(c), 3), "depth": round(excess, 5), "sigma": round(z, 2), "n_epochs": n_in}
    if best is None:
        return {"detected": False, "n_epochs": 0, "primary_phase": round(centre, 4)}
    best["primary_phase"] = round(centre, 4)
    best["detected"] = bool(best["sigma"] >= SECONDARY_SIGMA and best["depth"] >= DIP_MIN_FRACTION * depth)
    return best


def symmetry_index(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, n_harmonics: int) -> float:
    """Asymmetry of the multi-harmonic light curve: min over reflection points s of
    mean[(m(s + phi) - m(s - phi))^2] / (2 var m); 0 for curves symmetric about some phase
    (sinusoids, ellipsoidal and eclipsing binaries on circular orbits), ~0.3-1 for
    saw-tooth pulsators (RRab, Cepheids)."""
    _chi2, coef = _harmonic_fit(t, y, sigma, [j * frequency for j in range(1, n_harmonics + 1)])
    grid = np.arange(256) / 256.0
    model = np.zeros_like(grid)
    for j in range(1, n_harmonics + 1):
        model += coef[2 * j - 1] * np.sin(2 * np.pi * j * grid) + coef[2 * j] * np.cos(2 * np.pi * j * grid)
    var = float(np.var(model))
    if var <= 0:
        return 0.0
    idx = np.arange(256)
    best = min(float(np.mean((model[(s + idx) % 256] - model[(s - idx) % 256]) ** 2)) for s in range(0, 256, 2))
    return best / (2.0 * var)


# -- significance with correlated noise ---------------------------------------------------


def _local_continuum(t: np.ndarray, resid: np.ndarray, sigma: np.ndarray, frequency: float, *, lo: float,
                     hi: float, exclude: Sequence[float] = (), min_points: int = 8) -> tuple[float, float] | None:
    """(continuum S(frequency), spectral slope) of the residual periodogram ('psd' normalisation)
    on 160 log-spaced frequencies in [lo, hi], excluding +-2/T around each of ``exclude``.

    A power law is fitted to ln P vs ln f (bias-corrected by Euler's gamma, the mean of ln Exp(1))
    and evaluated at ``frequency``, but only where it is constrained: the retained frequencies
    must span a factor >= 2 and bracket ``frequency`` within a factor 2 (near the edges of the
    searched range, e.g. a 2-3-cycle peak, one short side remains; extrapolating a steep fit
    from it gave continua of 1e9-1e90). Otherwise the level is the median power / ln 2 (the
    median of Exp(1) is ln 2). None when fewer than ``min_points`` frequencies remain.
    """
    baseline = float(t.max() - t.min())
    grid = np.geomspace(lo, hi, 160) if hi > lo else np.array([frequency])
    for f_ex in exclude:
        grid = grid[np.abs(grid - f_ex) > 2.0 / baseline]
    if len(grid) < min_points:
        return None
    power = LombScargle(t, resid, sigma, fit_mean=True, center_data=True, normalization="psd").power(
        grid, method="cython")
    good = power > 0
    fg = grid[good]
    if np.sum(good) >= min_points and fg.max() >= 2.0 * fg.min() and fg.min() / 2.0 <= frequency <= 2.0 * fg.max():
        slope, intercept = np.polyfit(np.log(fg), np.log(power[good]), 1)
        return float(math.exp(intercept + slope * math.log(frequency) + np.euler_gamma)), float(slope)
    if np.sum(good):
        return float(max(np.median(power[good]) / math.log(2.0), 1e-300)), 0.0
    return None


def _low_frequency_continuum(t: np.ndarray, resid: np.ndarray, sigma: np.ndarray,
                             baseline: float) -> Callable[[float], float] | None:
    """Continuum S(nu) of the residual periodogram ('psd' normalisation) at low frequencies
    1/T <= nu <= :data:`RED_NOISE_ALIAS_MAX_OFFSET`: median power / ln 2 in 0.15-dex bins,
    interpolated in log-log (the first bin's level below 1/T). Bin medians are robust to the
    narrow holes a fitted signal leaves (the window couples a sinusoid at c + nu to one at nu).
    None when no bin has three frequencies."""
    lo, hi = 1.0 / baseline, max(RED_NOISE_ALIAS_MAX_OFFSET, 2.0 / baseline)
    grid = np.geomspace(lo, hi, 240)
    power = LombScargle(t, resid, sigma, fit_mean=True, center_data=True, normalization="psd").power(
        grid, method="cython")
    edges = np.arange(math.log10(lo), math.log10(hi) + 0.15, 0.15)
    idx = np.digitize(np.log10(grid), edges) - 1
    centres, levels = [], []
    for b in range(len(edges) - 1):
        chunk = power[(idx == b) & (power > 0)]
        if len(chunk) >= 3:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            levels.append(math.log10(float(np.median(chunk)) / math.log(2.0)))
    if not centres:
        return None
    xs, ys = np.array(centres), np.array(levels)

    def level(nu: float) -> float:
        return float(10.0 ** np.interp(math.log10(max(nu, lo)), xs, ys))

    return level


def window_alias_centres(survey: str | None, *, ground_based: bool) -> tuple[float, ...]:
    """Frequencies (c/d) at which a survey's sampling window peaks, i.e. to which low-frequency
    power (red noise, trends) is aliased (see :func:`red_noise_fap`).

    Ground-based (nightly, near a fixed hour angle, seasonal): n x solar/sidereal day +- m/yr
    (n <= 4, |m| <= 2) and 1, 2 /yr (cf. :func:`diurnal_alias`). Gaia: the scanning-law
    frequencies of :func:`gaia_scanning_alias`. Other surveys: none.
    """
    year = 1.0 / DAYS_PER_JULIAN_YEAR
    centres: set[float] = set()
    if ground_based:
        centres.update({year, 2.0 * year})
        for n in range(1, 5):
            for base in (1.0, SIDEREAL_DAY_FREQUENCY):
                centres.update(n * base + m * year for m in (-2, -1, 0, 1, 2))
    if survey == "gaia":
        centres.update(value for value, _name in _gaia_scanning_candidates())
    merged: list[float] = []
    for c in sorted(c for c in centres if c > 0):
        if not merged or c - merged[-1] > 1e-4:  # (the sidereal day - 1/yr is the solar day to 6e-8 c/d)
            merged.append(c)
    return tuple(merged)


def spectral_window_power(t: np.ndarray, sigma: np.ndarray, frequency: float) -> float:
    """|sum_i w_i exp(2 pi i f t_i)|^2 / (sum_i w_i)^2 with w = 1/sigma^2: the normalised power of
    the sampling window at ``frequency`` (1 at f = 0). Power at f0 leaks this fraction of itself
    to f0 +- f (Deeming 1975, Ap&SS 36, 137; VanderPlas 2018, ApJS 236, 16, sec. 3)."""
    w = 1.0 / sigma**2
    phase = 2.0 * np.pi * frequency * t
    return float((np.sum(w * np.cos(phase)) ** 2 + np.sum(w * np.sin(phase)) ** 2) / np.sum(w) ** 2)


def red_noise_fap(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, n_harmonics: int, *,
                  min_frequency: float, max_frequency: float,
                  alias_centres: Sequence[float] = ()) -> dict[str, Any]:
    """False-alarm probability of a K-harmonic signal against the local noise continuum.

    A white-noise FAP (e.g. Baluev 2008) makes red noise -- AGN variability, slow trends,
    cycle-to-cycle changes -- look periodic, because a red-noise periodogram is high at
    all low frequencies. Following Vaughan (2005, A&A 431, 391), the noise level is
    estimated from the data at the frequency of interest: the K-harmonic model is
    removed, and the continuum S(f) of the residuals' Lomb-Scargle periodogram ('psd'
    normalisation, = half the chi^2 reduction of a sinusoid, Exp(1)-distributed for white
    noise with correct errors; S = 1 then) is fitted in [f/3, 3f] (:func:`_local_continuum`),
    excluding +-2/T around f and around each fitted harmonic j f: the fit removed the
    residual power there, and those holes in ln P pulled the continuum of the red-noise
    Seyfert Mrk 817 down from ~950x to 540x the white level, making a 277-d red-noise peak
    "significant".

    Aliased red noise: the sampling window peaks at ``alias_centres``
    (:func:`window_alias_centres`; from the ground 1 c/d, the sidereal day and their +-1/yr
    side lobes), and each window peak c carries the low-frequency (red-noise) power at nu to
    c +- nu -- a narrow band of steep, strong power around 1 c/d that a continuum fitted over
    [f/3, 3f] misses (damped random walks on the real ZTF times of Mrk 817 gave "reliable"
    ~0.999-d periods with FAPs of 1e-9 to 1e-15). For every harmonic j f within
    :data:`RED_NOISE_ALIAS_MAX_OFFSET` of a centre c the aliased level W(c) S(nu) is
    computed -- W the normalised window power (:func:`spectral_window_power`), S(nu) the
    continuum of the data's own periodogram at nu = |j f - c| (at least 1/T; the signal's
    harmonics and their aliases excluded) -- and the continuum used is the largest of S(f)
    and these levels.

    The signal statistic Q = (chi2_0 - chi2_K) / (2 S) is Gamma(K, 1)-distributed for
    noise with that spectral density; the single-frequency probability p = P(Gamma(K) > Q)
    is converted to a FAP over the M = (f_max - f_min) T independent frequencies searched
    (FAP = 1 - (1 - p)^M; Horne & Baliunas 1986, ApJ 302, 757). Using S at the
    fundamental is conservative for red noise (the continuum is lower at the harmonics).
    Returns {"fap", "log_p", "continuum" (the one used), "slope", "white_level",
    "local_continuum", "alias" (None, or the dominating window centre, harmonic, offset nu,
    window power and level)}.
    """
    n = len(t)
    freqs = [j * frequency for j in range(1, n_harmonics + 1)]
    w = 1.0 / sigma**2
    chi2_0 = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))
    chi2_k, coef = _harmonic_fit(t, y, sigma, freqs)
    resid = y - _harmonic_design(t - t.mean(), freqs) @ coef
    baseline = float(t.max() - t.min())
    continuum, slope = 1.0, 0.0
    alias: dict[str, float] | None = None
    if n > 2 * n_harmonics + 3:
        local = _local_continuum(t, resid, sigma, frequency, lo=max(frequency / 3.0, min_frequency),
                                 hi=min(frequency * 3.0, max_frequency), exclude=freqs)
        if local is not None:
            continuum, slope = local
        windows = {c: spectral_window_power(t, sigma, c) for c in alias_centres
                   if any(abs(fj - c) <= RED_NOISE_ALIAS_MAX_OFFSET for fj in freqs)}
        windows = {c: v for c, v in windows.items() if v >= RED_NOISE_ALIAS_MIN_WINDOW}
        if windows:
            low = _low_frequency_continuum(t, resid, sigma, baseline)
            for fj in freqs:
                terms = [(c, v, max(abs(fj - c), 1.0 / baseline)) for c, v in windows.items()
                         if abs(fj - c) <= RED_NOISE_ALIAS_MAX_OFFSET]
                if not terms or low is None:
                    continue
                level = float(sum(v * low(nu) for _c, v, nu in terms))
                if level > (alias["level"] if alias else 0.0):
                    c, v, nu = max(terms, key=lambda item: item[1] * low(item[2]))
                    alias = {"centre": c, "harmonic_frequency": fj, "offset": nu, "window_power": v,
                             "level": level, "local_continuum": continuum}
        if alias is not None and alias["level"] <= continuum:
            alias = None  # the local continuum dominates: no aliased-red-noise band here
    effective = max(continuum, alias["level"]) if alias else continuum
    q = 0.5 * max(chi2_0 - chi2_k, 0.0) / max(effective, 1e-300)
    log_p = float(stats.gamma.logsf(q, n_harmonics))
    fap = _fap_from_log_p(log_p, (max_frequency - min_frequency) * baseline)
    # Mean residual power level in the same units (reduced chi^2 of the K-harmonic fit):
    # continuum / white_level is the red-noise excess at f over the residuals' own average.
    white_level = chi2_k / max(n - 2 * n_harmonics - 1, 1)
    return {"fap": fap, "log_p": log_p, "continuum": effective, "slope": float(slope), "white_level": white_level,
            "local_continuum": continuum, "alias": alias}


def sampling_visits(t: np.ndarray, frequency: float) -> list[np.ndarray]:
    """Indices (into ``t``, time order) of the "visits" of a series at ``frequency``: runs of
    epochs whose consecutive separation is below :data:`DENSE_SAMPLING_PHASE_STEP` of a cycle
    (e.g. the 1-3 transits of one Gaia scan, 106.5 min / 6 h apart, for periods >~ 4-13 d)."""
    order = np.argsort(t, kind="stable")
    steps = np.diff(t[order]) * frequency
    cuts = np.flatnonzero(steps >= DENSE_SAMPLING_PHASE_STEP) + 1
    return np.split(order, cuts)


def _clustered_or_sparse(t: np.ndarray, frequency: float) -> bool:
    """True when the effective-N correction applies: at least :data:`MIN_VISITS_EFFECTIVE_N`
    visits (:func:`sampling_visits`) of at most :data:`MAX_MEAN_VISIT_SIZE` epochs on average
    (sparse or clustered sampling, not the long contiguous runs of TESS sectors)."""
    n_visits = len(sampling_visits(t, frequency))
    return n_visits >= max(MIN_VISITS_EFFECTIVE_N, len(t) / MAX_MEAN_VISIT_SIZE)


def residual_effective_n(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float,
                         n_harmonics: int) -> tuple[float, float]:
    """(von Neumann eta of the time-ordered visit residuals, effective number of points).

    Residuals are taken from the K-harmonic model, in units of the errors, and grouped
    into visits (:func:`sampling_visits`). For a strictly periodic source in white noise
    all residuals are independent. Red noise (AGN), slow trends or zero-point drifts
    correlate them in two ways, both corrected:

    * within a visit: the intra-visit correlation rho (one-way ANOVA estimator) gives the
      design effect of clustered samples, deff = 1 + (m - 1) rho with m the size-weighted
      mean visit size (Kish 1965, Survey Sampling), i.e. N / deff independent points;
    * between visits: with eta the von Neumann ratio of the time-ordered visit means and
      eta = 2 (1 - rho_1) for lag-1 correlation rho_1, an AR(1) process has
      (1 - rho_1)/(1 + rho_1) = eta / (4 - eta) of its points independent (the classic
      effective-sample-size correction; e.g. von Storch & Zwiers 1999, Statistical
      Analysis in Climate Research, sec. 6.6).

    N_eff = N / deff x eta / (4 - eta). With one epoch per visit (sparse sampling) this is
    the AR(1) correction of the time-ordered residuals.
    """
    order = np.argsort(t)
    ts, ys, ss = t[order], y[order], sigma[order]
    n = len(ts)
    freqs = [j * frequency for j in range(1, n_harmonics + 1)]
    _chi2, coef = _harmonic_fit(ts, ys, ss, freqs)
    resid = (ys - _harmonic_design(ts - ts.mean(), freqs) @ coef) / ss
    if n < 3 or float(np.var(resid, ddof=1)) <= 0:
        return 2.0, float(n)
    visits = sampling_visits(ts, frequency)
    sizes = np.array([len(v) for v in visits], dtype=float)
    means = np.array([float(resid[v].mean()) for v in visits])
    deff = 1.0
    if 1 < len(visits) < n:
        grand = float(resid.mean())
        msb = float(np.sum(sizes * (means - grand) ** 2)) / (len(visits) - 1)
        msw = float(sum(float(np.sum((resid[v] - m) ** 2)) for v, m in zip(visits, means))) / (n - len(visits))
        m0 = (n - float(np.sum(sizes**2)) / n) / (len(visits) - 1)
        denom = msb + (m0 - 1.0) * msw
        rho = min(max((msb - msw) / denom, 0.0), 1.0) if denom > 0 else 0.0
        deff = 1.0 + (float(np.sum(sizes**2)) / n - 1.0) * rho
    if len(means) < 3 or float(np.var(means, ddof=1)) <= 0:
        return 2.0, n / deff
    eta = float(np.sum(np.diff(means) ** 2) / (len(means) - 1) / float(np.var(means, ddof=1)))
    e = min(max(eta, 0.0), 2.0)
    return eta, n / deff * e / (4.0 - e)


def multiharmonic_fap(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, n_harmonics: int, *,
                      min_frequency: float, max_frequency: float,
                      n_effective: float | None = None) -> tuple[float, float]:
    """False-alarm probability of a K-harmonic signal found by searching [f_min, f_max].

    Single-frequency test: F = [(chi2_0 - chi2_K)/(2K)] / [chi2_K/(N - 2K - 1)] ~
    F(2K, N - 2K - 1) (multi-harmonic analysis of variance, Schwarzenberg-Czerny 1996,
    ApJ 460, L107), times the M = (f_max - f_min) T independent frequencies searched
    (FAP = 1 - (1 - p)^M; Horne & Baliunas 1986, ApJ 302, 757). With correlated
    residuals (``n_effective`` < N) the coefficient variances grow by N/N_eff, so F is
    scaled by N_eff/N and the residual degrees of freedom become N_eff - 2K - 1.
    Returns (log p_single, FAP).
    """
    n = len(t)
    w = 1.0 / sigma**2
    chi2_0 = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))
    chi2_k, _coef = _harmonic_fit(t, y, sigma, [j * frequency for j in range(1, n_harmonics + 1)])
    d1 = 2 * n_harmonics
    n_eff = float(n if n_effective is None else min(n_effective, n))
    d2 = n_eff - d1 - 1
    if d2 < 1 or chi2_k <= 0 or chi2_0 <= chi2_k or n - d1 - 1 < 1:
        return 0.0, 1.0
    f_stat = ((chi2_0 - chi2_k) / d1) / (chi2_k / (n - d1 - 1)) * n_eff / n
    log_p = float(stats.f.logsf(f_stat, d1, d2))
    return log_p, _fap_from_log_p(log_p, (max_frequency - min_frequency) * float(t.max() - t.min()))


def rank_fap(t: np.ndarray, y: np.ndarray, frequency: float, n_harmonics: int, *, min_frequency: float,
             max_frequency: float) -> float:
    """Outlier-robust FAP of a K-harmonic signal: :func:`multiharmonic_fap` of the data's
    normal scores (ranks mapped to Gaussian quantiles, equal weights).

    A monotonic transform keeps any periodic light-curve shape, but caps the leverage of
    single epochs: a "period" that rests on a few extreme epochs -- in-eclipse transits of
    a sparsely sampled detached binary aligned by a chance frequency (Gaia DR3
    4655469709637879296: 0.0524 d, Gaussian FAP 6e-5, rank FAP 1) -- loses its
    significance, while real pulsations keep theirs (Gaia RRab stars: rank FAP 1e-17 to
    1e-35). K is capped at 4 (the rank transform flattens the finest structure).
    """
    n = len(y)
    scores = stats.norm.ppf((stats.rankdata(y) - 0.5) / n)
    _log_p, fap = multiharmonic_fap(t, scores, np.ones(n), frequency, max(1, min(n_harmonics, 4)),
                                    min_frequency=min_frequency, max_frequency=max_frequency)
    return fap


def _fap_from_log_p(log_p: float, trials: float) -> float:
    """1 - (1 - p)^M for p = exp(log_p), stable for tiny and for large p."""
    trials = max(trials, 1.0)
    if not math.isfinite(log_p) or log_p < -30:
        return min(math.exp(min(log_p + math.log(trials), 0.0)), 1.0) if math.isfinite(log_p) else 0.0
    p = math.exp(log_p)
    if p >= 1.0:
        return 1.0
    return min(max(float(-np.expm1(trials * math.log1p(-p))), 0.0), 1.0)


def period_significance(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, n_harmonics: int, *,
                        min_frequency: float, max_frequency: float,
                        alias_centres: Sequence[float] = ()) -> dict[str, Any]:
    """FAP of a K-harmonic periodic signal that does not assume white noise.

    Two estimates, the larger (more conservative) FAP is used:

    * :func:`red_noise_fap` -- the signal against the local noise continuum of the
      residual periodogram (Vaughan 2005), including red noise aliased by the sampling
      window (``alias_centres``, :func:`window_alias_centres`); valid for any sampling.
    * :func:`multiharmonic_fap` with the effective number of independent points of the
      residuals (:func:`residual_effective_n`), whenever the data form at least
      :data:`MIN_VISITS_EFFECTIVE_N` short visits at this frequency (:func:`_clustered_or_sparse`):
      sparse sampling, or clustered sampling such as Gaia scans (1-3 transits within
      6 h) at long periods. (With dense sampling -- TESS sectors -- there are only a few
      "visits" and consecutive residuals of any imperfect model are correlated, so the
      lag-1 statistic would measure the model, not the noise.) In sparse, clustered
      sampling red noise concentrates in alias spikes that a local continuum does not
      capture, while its time correlation does. The estimate needs at least two effective
      points per model parameter (N_eff >= 2 (2K + 1)): with fewer, the residuals and their
      correlations are those of the fit itself, and the F test has almost no residual degrees
      of freedom -- the misfit of a K-harmonic model to a sharp light curve, identical within
      a visit, gave strictly periodic Gaia Cepheids and binaries N_eff = 4-14 of 37-44 epochs
      and FAP = 1 (Gaia DR3 2686850313960864256, 5690870910317297408). The continuum estimate
      alone is then used (``effective_n_applicable`` False).

    ``variance_inflation`` is the factor by which correlated residuals inflate the variance
    of the signal's Fourier coefficients -- and so of its frequency -- over the white-noise
    estimate *from the residual scatter*: max(continuum / white level, N / N_eff, 1). The
    continuum is in units of the stated errors, and so is the residuals' mean power level
    (their reduced chi^2, ``white_level``), so dividing removes any error-bar scale that a
    residual-based variance already contains (multiplying the residual variance by the raw
    continuum counted that scale twice); N / N_eff is scale-free.

    Returns {"fap", "continuum", "slope", "white_level", "effective_n" (None if not used),
    "effective_n_applicable", "variance_inflation", "alias"}.
    """
    noise = red_noise_fap(t, y, sigma, frequency, n_harmonics, min_frequency=min_frequency,
                          max_frequency=max_frequency, alias_centres=alias_centres)
    white = max(noise["white_level"], 1e-300)
    out: dict[str, Any] = {"fap": noise["fap"], "continuum": noise["continuum"], "slope": noise["slope"],
                           "white_level": noise["white_level"], "effective_n": None, "effective_n_applicable": None,
                           "variance_inflation": max(noise["continuum"] / white, 1.0), "alias": noise["alias"]}
    if _clustered_or_sparse(t, frequency):
        _eta, n_eff = residual_effective_n(t, y, sigma, frequency, n_harmonics)
        out["effective_n"] = n_eff
        out["variance_inflation"] = max(out["variance_inflation"], len(t) / max(n_eff, 1.0))
        out["effective_n_applicable"] = n_eff >= 2.0 * (2 * n_harmonics + 1)
        if out["effective_n_applicable"]:
            _log_p, fap_eff = multiharmonic_fap(t, y, sigma, frequency, n_harmonics, min_frequency=min_frequency,
                                                max_frequency=max_frequency, n_effective=n_eff)
            out["fap"] = max(out["fap"], fap_eff)
    return out


def phase_coherence(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float, *,
                    n_ratio: float = 1.0, n_harmonics: int = 1) -> dict[str, Any] | None:
    """Is the periodic signal at ``frequency`` the same in the first and second half of the data?

    Red noise produces periodogram peaks whose phase and amplitude wander (Vaughan et al.
    2016, MNRAS 461, 3145); a periodic source repeats. Cross-validated: each half's light
    curve (the adopted ``n_harmonics`` K, at most (n_half - 1)/4, i.e. two epochs per
    parameter -- or, when it predicts the other half better, the order 1/(2 x the half's
    largest phase gap) that half's coverage supports; unit rms; one phase reference) is the
    template for the *other* half, which is
    fitted with mean + A x template + B x its phase derivative -- only the amplitude (A, data
    units) and a phase offset (B). The shape seen in one half must predict the other: red
    noise fails, while a template fitted to all the data would correlate with each half by
    construction (tried: on damped random walks sampled like Gaia and ZTF, 25 of 37
    significant red-noise peaks then passed, against 5 of 37 cross-validated). A stable
    signal gives equal (A, B) in both halves (B ~ 0; a phase shift gives B of opposite signs).
    The halves' (A, B) are compared with chi^2 (2 dof); each half's covariance is its
    residual scatter's, inflated by ``n_ratio`` -- the correlated-noise factor of
    :func:`period_significance` (``variance_inflation``: red-noise excess over the residuals'
    own level, or N/N_eff), which must not contain the error-bar scale again -- plus the
    uncertainty of the template it was fitted with: the other half's coefficient covariance
    propagated by Monte Carlo (:data:`COHERENCE_TEMPLATE_DRAWS` draws, fixed seed). A
    template fitted to a sparse half is uncertain where that half has no epochs, and without
    that term its misfit of a non-sinusoidal light curve was read as a phase change: a
    strictly periodic 0.8-mag saw-tooth with 3 mmag noise on the real Gaia times of four
    Cepheids was "incoherent" in 28 of 48 simulations (review), 14 of 48 with it (see also
    :func:`_require_cross_band_confirmation` for many-cycle signals). Each half's signal must
    itself be significant, with p <= that of a 3-sigma sinusoid amplitude (chi^2_2 = 9). A
    statistically significant difference is accepted as modulation of a periodic signal
    (``modulated``) when the signal is detected at >= :data:`COHERENCE_STRONG_SIGMA` in both
    halves and changed by at most :data:`COHERENCE_MAX_RELATIVE_CHANGE` of its size. Returns
    None when a half has fewer than 8 points or a template is flat.
    """
    order = np.argsort(t)
    ts, ys, ss = t[order], y[order], sigma[order]
    mid = len(ts) // 2
    if mid < 8 or len(ts) - mid < 8:
        return None
    t_ref = float(ts.mean())  # one phase reference for both halves
    grid = 2.0 * np.pi * np.arange(256) / 256.0
    halves = (slice(0, mid), slice(mid, None))
    inflate = max(n_ratio, 1.0)
    rng = np.random.default_rng(len(ts))

    def template_orders(sl: slice) -> list[int]:
        """Candidate template orders of one half: the adopted K (at most (n_half - 1)/4) and, if
        lower, the order its largest phase gap supports (1/(2 gap))."""
        th = ts[sl]
        k = max(1, min(int(n_harmonics), (len(th) - 1) // 4))
        folded = np.sort(((th - t_ref) * frequency) % 1.0)
        max_gap = float(np.max(np.diff(np.concatenate([folded, folded[:1] + 1.0]))))
        k_gap = max(1, min(k, int(0.5 / max_gap))) if max_gap > 0 else k
        return sorted({k, k_gap})

    def template_of(sl: slice, k: int) -> tuple[np.ndarray, np.ndarray, int] | None:
        """K-harmonic coefficients of one half and their covariance (residual scatter x inflation)."""
        th, yh, sh = ts[sl], ys[sl], ss[sl]
        design = _harmonic_design(th - t_ref, [j * frequency for j in range(1, k + 1)]) / sh[:, None]
        coef, *_ = np.linalg.lstsq(design, yh / sh, rcond=None)
        resid = yh / sh - design @ coef
        s2 = float(resid @ resid) / max(len(th) - (2 * k + 1), 1) * inflate
        try:
            cov = np.linalg.inv(design.T @ design) * s2
        except np.linalg.LinAlgError:
            return None
        return coef, cov, k

    def curves(coef: np.ndarray, k: int, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Unit-rms template and unit-rms phase derivative of coefficients ``coef`` at ``tt``."""
        a, b = coef[1::2], coef[2::2]
        j = np.arange(1, k + 1)[:, None]
        shape = (a[:, None] * np.sin(j * grid) + b[:, None] * np.cos(j * grid)).sum(axis=0)
        slope = (j * (a[:, None] * np.cos(j * grid) - b[:, None] * np.sin(j * grid))).sum(axis=0)
        rms_s, rms_d = float(np.std(shape)), float(np.std(slope))
        if rms_s <= 0 or rms_d <= 0:
            return None
        ph = 2.0 * np.pi * frequency * (tt - t_ref)
        m = sum(a[i] * np.sin((i + 1) * ph) + b[i] * np.cos((i + 1) * ph) for i in range(k))
        q = sum((i + 1) * (a[i] * np.cos((i + 1) * ph) - b[i] * np.sin((i + 1) * ph)) for i in range(k))
        return np.asarray(m) / rms_s, np.asarray(q) / rms_d

    candidates = [[tpl for tpl in (template_of(sl, k_) for k_ in template_orders(sl)) if tpl is not None]
                  for sl in halves]
    if not all(candidates):
        return None
    fits = []
    orders = []
    for h, sl in enumerate(halves):
        sh = ss[sl]
        yw = ys[sl] / sh
        # Of the other half's candidate templates (full order; order limited by its phase gaps)
        # the one predicting this half better is used: a gap-limited order misfits sharp light
        # curves, a full one extrapolates across the gap.
        best: tuple[float, Any, Any, np.ndarray] | None = None
        for coef_c, cov_c, k_c in candidates[1 - h]:
            cv = curves(coef_c, k_c, ts[sl])
            if cv is None:
                continue
            design_c = np.column_stack([np.ones(len(sh)), cv[0], cv[1]]) / sh[:, None]
            params_c, *_ = np.linalg.lstsq(design_c, yw, rcond=None)
            resid_c = yw - design_c @ params_c
            chi2_c = float(resid_c @ resid_c)
            if best is None or chi2_c < best[0]:
                best = (chi2_c, (coef_c, cov_c, k_c), design_c, params_c)
        if best is None:
            return None
        chi2_c, (coef_o, cov_o, k_o), design, params = best
        orders.append(k_o)
        scale = chi2_c / max(len(sh) - 3, 1) * inflate
        try:
            cov = np.linalg.inv(design.T @ design)[1:, 1:] * scale
        except np.linalg.LinAlgError:
            return None
        # The template's own uncertainty, propagated by Monte Carlo draws of its coefficients.
        evals, evecs = np.linalg.eigh(0.5 * (cov_o + cov_o.T))
        root = evecs * np.sqrt(np.clip(evals, 0.0, None))
        draws = []
        for _ in range(COHERENCE_TEMPLATE_DRAWS):
            cvd = curves(coef_o + root @ rng.standard_normal(len(coef_o)), k_o, ts[sl])
            if cvd is None:
                continue
            design_d = np.column_stack([np.ones(len(sh)), cvd[0], cvd[1]]) / sh[:, None]
            params_d, *_ = np.linalg.lstsq(design_d, yw, rcond=None)
            draws.append(params_d[1:])
        if len(draws) >= COHERENCE_TEMPLATE_DRAWS // 2:
            cov = cov + np.cov(np.array(draws).T)
        fits.append((params[1:], cov))
    (a1, c1), (a2, c2) = fits
    diff = a1 - a2
    try:
        chi2_diff = float(diff @ np.linalg.solve(c1 + c2, diff))
        amp_chi2 = [float(p_ @ np.linalg.solve(c_, p_)) for p_, c_ in fits]
    except np.linalg.LinAlgError:
        return None
    p_diff = float(stats.chi2.sf(chi2_diff, 2))
    p_amp = [float(stats.chi2.sf(q, 2)) for q in amp_chi2]
    amp_sigma = [math.sqrt(max(q, 0.0)) for q in amp_chi2]
    # Size of the change relative to the signal (~1.4 for independent phases, ~0.2 for a
    # 20 % / 0.1-rad change).
    norm = math.sqrt(0.5 * (float(a1 @ a1) + float(a2 @ a2)))
    relative_change = float(math.sqrt(float(diff @ diff)) / norm) if norm > 0 else math.inf
    same = p_diff > 1e-3
    # A signal detected at >= COHERENCE_STRONG_SIGMA in each half (red noise cannot be: the
    # variances are scaled to the local noise level) whose amplitude/phase change by a modest
    # fraction is periodic but modulated (Blazhko RR Lyrae, e.g. RR Lyr itself in one TESS
    # sector), not red noise.
    modulated = (not same and min(amp_sigma) >= COHERENCE_STRONG_SIGMA
                 and relative_change <= COHERENCE_MAX_RELATIVE_CHANGE)
    return {"chi2_difference": round(chi2_diff, 2), "p_difference": p_diff, "n_harmonics": max(orders),
            "amplitude_phase_halves": [[round(float(v), 4) for v in p_] for p_, _c in fits],
            "variance_inflation": round(inflate, 3),
            "amplitude_sigma_halves": [round(v, 2) for v in amp_sigma],
            "amplitude_p_halves": p_amp, "relative_change": round(relative_change, 3), "modulated": modulated,
            "coherent": bool((same or modulated) and max(p_amp) <= COHERENCE_AMPLITUDE_P)}


def alias_comb(gram: _Periodogram, *, ratio: float = ALIAS_COMB_POWER_RATIO) -> dict[str, float] | None:
    """Competing periodogram peaks from a long gap in the sampling (cycle-count ambiguity).

    When the data form widely separated segments (largest gap G > baseline/3, e.g. TESS
    sectors years apart) the peak splits into a comb with spacing ~1/(segment separation);
    the number of cycles across the gap is then ambiguous. Local maxima within one
    resolution element of the segments (1 / (baseline - G)) of the best peak with at
    least ``ratio`` of its power are reported: {"spacing": |delta f| of the nearest one,
    "power_ratio": ...}; None when no such peak exists.
    """
    t = np.sort(gram.t)
    baseline = float(t[-1] - t[0])
    if len(t) < 3 or baseline <= 0:
        return None
    gap = float(np.max(np.diff(t)))
    if gap <= baseline / 3.0:
        return None
    freq_step = (gram.result.max_frequency - gram.result.min_frequency) / max(gram.result.n_frequencies - 1, 1)
    if freq_step <= 0:
        return None
    best = int(np.argmax(gram.power))
    half = math.ceil(1.0 / max(baseline - gap, 1e-9) / freq_step)
    lo, hi = max(best - half, 1), min(best + half, len(gram.power) - 2)
    p = gram.power
    competitors = [i for i in range(lo, hi + 1) if i != best and p[i] > p[i - 1] and p[i] >= p[i + 1]
                   and p[i] >= ratio * p[best]]
    if not competitors:
        return None
    nearest = min(competitors, key=lambda i: abs(i - best))
    return {"spacing": abs(nearest - best) * freq_step, "power_ratio": round(float(p[nearest] / p[best]), 3),
            "n_competing_peaks": len(competitors), "gap_days": round(gap, 1)}


# -- periodograms --------------------------------------------------------------------


@dataclass(slots=True)
class _Periodogram:
    """Internal: one series' power on a grid, its data and its chi^2 about the weighted mean."""

    result: PeriodogramResult
    power: np.ndarray
    chi2_0: float
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    unit: str = "mag"
    host_radius_rsun: float | None = None  # TIC radius of a TESS target (planet-like single dips)


def _periodogram(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, freq: np.ndarray, used_os: float, *, series: str,
                 n_peaks: int = 5, requested_os: float | None = None, unit: str = "mag") -> _Periodogram:
    baseline = float(t.max() - t.min())
    ls = LombScargle(t, y, sigma, fit_mean=True, center_data=True, normalization="standard")
    power = ls_power(ls, freq)
    best = int(np.argmax(power))
    fap = float(ls.false_alarm_probability(power[best], method="baluev", minimum_frequency=freq[0],
                                           maximum_frequency=freq[-1]))
    result = PeriodogramResult(
        series=series, n=len(t), baseline_days=baseline, best_period_days=float(1.0 / freq[best]),
        best_frequency_per_day=float(freq[best]), power=float(power[best]), false_alarm_probability=fap,
        min_frequency=float(freq[0]), max_frequency=float(freq[-1]), n_frequencies=len(freq), oversampling=used_os,
        method=("generalised Lomb-Scargle search (astropy); parity/eclipse-shape period ladder; multi-harmonic "
                "refinement; multi-harmonic FAP corrected for correlated residuals"),
        peaks=_top_peaks(freq, power, n_peaks=n_peaks, min_separation=1.0 / baseline), members=[series],
        lomb_scargle_period_days=float(1.0 / freq[best]), lomb_scargle_false_alarm_probability=fap,
    )
    if requested_os is not None and used_os < requested_os - 1e-9:
        result.warnings.append(f"frequency grid reduced to {used_os:.2f} samples per peak (limit {MAX_FREQUENCIES} "
                               "frequencies)")
    w = 1.0 / sigma**2
    chi2_0 = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))
    return _Periodogram(result, power, chi2_0, t, y, sigma, unit)


@dataclass(slots=True)
class _LadderDecision:
    """Outcome of the period ladder for one signal: adopted f = refined LS f / multiplier."""

    multiplier: float
    doubled_by: str | None
    ladder: list[dict[str, Any]]
    shape: dict[str, Any]
    first_bic: tuple[float | None, list[float]]
    lead: str


def planet_max_depth(host_radius_rsun: float | None) -> float | None:
    """Deepest single transit a planet can cause on a host of this radius (relative flux):
    (PLANET_MAX_RADIUS_RSUN / R_host)^2, at most PLANET_MAX_DEPTH; None when R_host is unknown."""
    if host_radius_rsun is None or not math.isfinite(host_radius_rsun) or host_radius_rsun <= 0:
        return None
    return min((PLANET_MAX_RADIUS_RSUN / host_radius_rsun) ** 2, PLANET_MAX_DEPTH)


def _refine_peak(gram: _Periodogram, step: float) -> float:
    """Refine the grid peak (multi-harmonic chi^2 within +-1.5 resolution elements): a
    non-sinusoidal signal (e.g. narrow eclipses) pulls the single-harmonic peak."""
    baseline = float(gram.t.max() - gram.t.min())
    f, _sf, _k = refine_period(gram.t, gram.y, gram.sigma, gram.result.best_frequency_per_day,
                               max(1.5 / baseline, step), n_grid=121,
                               bounds=(gram.result.min_frequency, gram.result.max_frequency))
    return f


def _period_ladder(gram: _Periodogram, band_data: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
                   f_start: float, *, ground_based: bool = False) -> _LadderDecision:
    """Harmonic ladder and eclipse shape (steps 2-3 of :func:`_finalise_period`).

    From the ground, a single-dip fold at a diurnal sampling alias (:func:`diurnal_alias`, not
    the annual ones) is not doubled: the "eclipse" is then the window's image of low-frequency
    variability (a damped random walk's 0.99838-d alias became a 1.9967-d "eclipsing binary
    with equal minima" in the review)."""
    t, y, s = gram.t, gram.y, gram.sigma
    baseline = float(t.max() - t.min())
    f = f_start
    ladder: list[dict[str, Any]] = []
    first_bic: tuple[float | None, list[float]] = (None, [])
    for _ in range(MAX_PERIOD_DOUBLINGS):
        if baseline * f / 2.0 < PARITY_MIN_CYCLES:
            break
        test = parity_test(band_data, f)
        per_band = [round(odd_harmonic_delta_bic(bt, by, bs, f), 2) for bt, by, bs in band_data
                    if len(bt) >= MIN_POINTS_PERIOD]
        delta = float(sum(per_band)) if per_band else None
        if not ladder:
            first_bic = (delta, per_band)
        effect = test["effect"] if test else 0.0
        parity_significant = bool(test and test["sigma"] >= PARITY_SIGMA)
        # The odd harmonics of f/2 must not argue strongly against 2P: a significant parity
        # difference with delta BIC < -10 is cycle-to-cycle modulation (Blazhko effect, spots),
        # not unequal minima, whose depth difference the odd harmonics always pick up.
        vetoed = parity_significant and delta is not None and delta < -HARMONIC_BIC_THRESHOLD
        significant = (parity_significant and not vetoed) or (delta is not None and delta > HARMONIC_BIC_THRESHOLD)
        doubled = significant and effect >= PARITY_MIN_EFFECT
        ladder.append({"period_days": 1.0 / f, "parity_sigma": test["sigma"] if test else None,
                       "parity_effect": effect, "delta_bic": None if delta is None else round(delta, 2),
                       "parity_vetoed_by_bic": vetoed, "doubled": doubled})
        if not doubled:
            break
        f, _sf, _k = refine_period(t, y, s, f / 2.0, 0.5 / baseline, n_grid=121)
    n_doubled = sum(1 for step_ in ladder if step_["doubled"])
    shape = eclipse_shape(t, y, s, f, unit=gram.unit)
    doubled_by = "parity" if n_doubled else None
    multiplier = 2.0 ** n_doubled
    diurnal = None
    if ground_based:
        diurnal = next((name for name in (diurnal_alias(f_start, baseline), diurnal_alias(f, baseline))
                        if name and name not in ("annual", "semi-annual")), None)
    if shape["eclipse_like"] and diurnal is not None:
        shape["not_doubled"] = f"single dip at the {diurnal} sampling alias"
    elif shape["eclipse_like"] and baseline * f / 2.0 >= MIN_CYCLES_IN_BASELINE:
        planet_depth = planet_max_depth(gram.host_radius_rsun)
        if planet_depth is not None:
            shape["planet_max_depth"] = round(planet_depth, 4)
        if (shape["depth_relative_flux"] >= ECLIPSE_MIN_DOUBLING_DEPTH
                and (planet_depth is None or shape["depth_relative_flux"] > planet_depth)):
            multiplier *= 2.0
            doubled_by = "parity+eclipse_shape" if n_doubled else "eclipse_shape"
        else:
            # A single dip that a transiting planet can produce is transit-like: keep P (2P is
            # reported as the alternative) -- two equal eclipses are an assumption the data do
            # not support.
            shape["transit_like"] = True
    return _LadderDecision(multiplier, doubled_by, ladder, shape, first_bic, gram.result.series)


def _complete_period(gram: _Periodogram, f_refined: float, decision: _LadderDecision, *, ground_based: bool,
                     survey: str | None) -> None:
    """Adopt ``f_refined / multiplier``, refine, estimate significance and reliability."""
    r = gram.result
    t, y, s = gram.t, gram.y, gram.sigma
    baseline = float(t.max() - t.min())
    f_ls = f_refined
    f = f_refined / decision.multiplier
    alternative: float | None = None
    doubled_by = decision.doubled_by
    shape = decision.shape
    n_parity = sum(1 for step_ in decision.ladder if step_["doubled"])
    if doubled_by and "eclipse_shape" in doubled_by:
        alternative = 1.0 / (2.0 * f)
        r.warnings.append(
            f"one narrow dip per {alternative:.6g}-d cycle (depth {shape['depth']:.3g}, width {shape['width']:.2f} in "
            f"phase): eclipsing binary with (nearly) equal minima assumed, period doubled to {1.0 / f:.6g} d; if only "
            f"one eclipse per orbit is visible (e.g. a transit or an undetected secondary) the period is "
            f"{alternative:.6g} d")
    elif n_parity:
        alternative = 1.0 / (2.0 * f)
        last = decision.ladder[n_parity - 1]
        r.warnings.append(
            f"consecutive cycles differ (parity test {last['parity_sigma']} sigma, effect {last['parity_effect']:.2f}; "
            f"delta BIC {last['delta_bic']}): unequal minima, period doubled {n_parity}x from the Lomb-Scargle peak "
            f"{1.0 / f_ls:.6g} d")
    if shape.get("transit_like"):
        # The open question for a single dip is P vs 2P (two equal eclipses), whatever the ladder did.
        alternative = 2.0 / f
        planet = shape.get("planet_max_depth")
        why = (f"a transiting planet can cause it: below the {planet:.1%} of a 2-R_Jup planet on the "
               f"{gram.host_radius_rsun:.2f}-R_sun host (TIC)" if planet is not None
               and shape["depth_relative_flux"] >= ECLIPSE_MIN_DOUBLING_DEPTH
               else f"shallower than {ECLIPSE_MIN_DOUBLING_DEPTH:.0%}")
        r.warnings.append(
            f"one narrow dip per cycle (depth {shape['depth_relative_flux']:.2%} of the flux, width "
            f"{shape['width']:.2f} in phase; {why}) and no secondary eclipse: transit-like, period kept at "
            f"{1.0 / f:.6g} d; only if it is an eclipsing binary with two equal eclipses is the orbital period "
            f"{2.0 / f:.6g} d")
    r.harmonic_test = {
        "doubled": doubled_by is not None, "doubled_by": doubled_by,
        "n_doublings": round(math.log2(decision.multiplier)),
        "decided_on": decision.lead,
        "delta_bic": None if decision.first_bic[0] is None else round(decision.first_bic[0], 2),
        "per_band_delta_bic": decision.first_bic[1],
        "threshold": HARMONIC_BIC_THRESHOLD, "parity_sigma_threshold": PARITY_SIGMA,
        "parity_min_effect": PARITY_MIN_EFFECT, "ladder": decision.ladder, "shape": shape,
        "method": ("even/odd-cycle analysis of variance and BIC of the odd harmonics of f/2 (unequal consecutive "
                   "cycles), then eclipse shape (one narrow dip per cycle); decided on the survey's best series"),
    }
    freq, sigma_f, k = refine_period(t, y, s, f, 0.5 / baseline, max_harmonics=REFINE_MAX_HARMONICS, n_grid=121,
                                     bounds=(r.min_frequency / decision.multiplier, r.max_frequency))
    if (doubled_by is None and not shape.get("transit_like")
            and symmetry_index(t, y, s, freq, min(k, MAX_HARMONICS)) < SYMMETRY_MAX_INDEX):
        alternative = 2.0 / freq
        r.warnings.append(f"symmetric light curve: P and 2P are not distinguished by the data; for an "
                          f"ellipsoidal or contact binary with equal minima the orbital period is {2.0 / freq:.6g} d")
    r.alternative_period_days = alternative
    r.best_frequency_per_day = freq
    r.best_period_days = 1.0 / freq
    r.period_error_days = None if sigma_f is None else sigma_f / freq**2
    r.n_harmonics = k
    comb = alias_comb(gram)
    if comb is not None:
        spacing = comb["spacing"] * freq / (1.0 / (r.lomb_scargle_period_days or r.best_period_days))
        widened = spacing / freq**2
        r.period_error_days = max(r.period_error_days or 0.0, widened)
        r.warnings.append(f"sampling gap of {comb['gap_days']:.0f} d: {comb['n_competing_peaks']} competing alias "
                          f"peak(s) with >= {ALIAS_COMB_POWER_RATIO:.0%} of the power ({comb['power_ratio']:.2f}); the "
                          f"cycle count across the gap is ambiguous, period error widened to the alias spacing "
                          f"({widened:.2g} d)")
        r.harmonic_test["alias_comb"] = comb
    significance = period_significance(t, y, s, freq, k, min_frequency=r.min_frequency, max_frequency=r.max_frequency,
                                       alias_centres=window_alias_centres(survey, ground_based=ground_based))
    r.robust_false_alarm_probability = rank_fap(t, y, freq, k, min_frequency=r.min_frequency,
                                                max_frequency=r.max_frequency)
    if r.robust_false_alarm_probability >= PERIOD_FAP_THRESHOLD:
        r.warnings.append(f"the signal rests on a few extreme epochs: rank-based FAP "
                          f"{r.robust_false_alarm_probability:.2g} >= {PERIOD_FAP_THRESHOLD}")
    # Correlated residuals (red noise, Blazhko modulation, night-to-night offsets) carry less
    # information on the frequency than white noise of the same rms: the Cramer-Rao error of
    # refine_period grows by sqrt(variance inflation) (see period_significance).
    r.error_inflation = round(float(max(significance["variance_inflation"], 1.0)), 3)
    if sigma_f is not None:
        r.period_error_days = max(r.period_error_days or 0.0,
                                  sigma_f * math.sqrt(r.error_inflation) / freq**2)
    # ... and the time-block sandwich error captures correlations the continuum misses
    # (a slowly wandering phase: spots, period changes, multi-mode beating).
    sigma_block = block_frequency_error(t, y, s, freq, k)
    if sigma_block is not None:
        r.period_error_days = max(r.period_error_days or 0.0, sigma_block / freq**2)
    r.false_alarm_probability = significance["fap"]
    r.noise_continuum = round(significance["continuum"], 4)
    r.noise_spectral_slope = round(significance["slope"], 3)
    r.effective_n = None if significance["effective_n"] is None else round(significance["effective_n"], 1)
    if significance["continuum"] > 3.0 and significance["slope"] < -0.5:
        r.warnings.append(f"red noise: the residual power near the peak is {significance['continuum']:.1f}x the "
                          f"white-noise level (spectral slope {significance['slope']:.1f}); significance assessed "
                          "against it")
    alias = significance.get("alias")
    if alias is not None and alias["level"] > 3.0 * max(significance["white_level"], 1e-300):
        r.warnings.append(f"red noise aliased by the sampling window: the {1.0 / alias['centre']:.6g}-d window peak "
                          f"carries low-frequency power ({alias['level']:.1f}x the stated-error level) to the signal's "
                          f"frequency; significance assessed against it")
    if significance["effective_n"] is not None and significance["effective_n"] < 0.5 * len(t):
        if significance.get("effective_n_applicable") is False:
            r.warnings.append(f"residuals are correlated in time ({significance['effective_n']:.0f} effective of "
                              f"{len(t)} points, fewer than two per model parameter): significance from the "
                              "residual noise continuum")
        else:
            r.warnings.append(f"residuals are correlated in time: significance computed with "
                              f"{significance['effective_n']:.0f} effective of {len(t)} points")
    _chi2_k, coef_k = _harmonic_fit(t, y, s, [j * freq for j in range(1, k + 1)])
    phase_grid = 2.0 * np.pi * np.arange(512) / 512.0
    model = sum(coef_k[2 * j - 1] * np.sin(j * phase_grid) + coef_k[2 * j] * np.cos(j * phase_grid)
                for j in range(1, k + 1))
    r.harmonic_test["semi_amplitude"] = round(0.5 * float(np.max(model) - np.min(model)), 6)
    r.harmonic_test["unit"] = gram.unit
    r.harmonic_test["aliased_red_noise"] = significance.get("alias")
    # Significance from the residual continuum alone (too few effective points for the effective-N
    # test): the continuum misses red noise concentrated in sparse sampling's alias spikes, so the
    # signal must then also repeat in both halves of the data -- no exemption (see
    # _many_cycle_evidence), and be detected in each half at >= CONTINUUM_ONLY_HALF_SIGMA. Damped
    # random walks on real Gaia times otherwise gave 9.8-d and 18-d "periods" at FAP 1e-7 to 1e-44
    # (halves at 3.8 and 4.5 sigma).
    r.harmonic_test["continuum_only"] = significance.get("effective_n_applicable") is False
    at_window = _window_frequency(r, ground_based=ground_based, survey=survey)
    if (baseline * freq < COHERENCE_MAX_CYCLES or (at_window is not None and not at_window[1])
            or r.harmonic_test["continuum_only"]):
        coherence = phase_coherence(t, y, s, freq, n_ratio=r.error_inflation or 1.0, n_harmonics=k)
        if (coherence is not None and coherence["coherent"] and r.harmonic_test["continuum_only"]
                and min(coherence["amplitude_sigma_halves"]) < CONTINUUM_ONLY_HALF_SIGMA):
            coherence["coherent"] = False
            coherence["required_half_sigma"] = CONTINUUM_ONLY_HALF_SIGMA
        r.harmonic_test["phase_coherence"] = coherence
        if coherence is not None and not coherence["coherent"]:
            if _many_cycle_evidence(r):
                r.warnings.append(f"the two halves of the data differ more than their scatter allows, but "
                                  f"{baseline * freq:.0f} cycles at FAP {r.false_alarm_probability:.2g} < "
                                  f"{PERIODIC_EVIDENCE_FAP:g} are not red noise")
            else:
                r.warnings.append("the signal is not coherent between the two halves of the data (phase/amplitude "
                                  "differ, or a half shows no significant signal): typical of red noise")
    _assess_period(r, ground_based=ground_based, survey=survey)


def _finalise_period(gram: _Periodogram, band_data: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], *,
                     step: float, ground_based: bool, survey: str | None = None,
                     decision: _LadderDecision | None = None) -> _LadderDecision:
    """Period ladder, refinement, significance and reliability of one series' peak.

    ``band_data`` holds the decisive bands of the same survey (never Gaia BP/RP, see
    :data:`NON_DECISIVE_SERIES`); they are tested jointly. ``decision`` (from the
    survey's best-determined series, see :func:`search_periods`) is applied instead of
    running the ladder on this series when given.

    1. The Lomb-Scargle peak is refined (multi-harmonic chi^2 within +-1.5 frequency
       resolution elements: a non-sinusoidal signal pulls the single-harmonic peak).
    2. Harmonic ladder: while consecutive cycles differ (:func:`parity_test` at >= 5
       sigma unless the odd harmonics of f/2 worsen the BIC by more than 10, or the odd
       harmonics improve the BIC by > 10 -- Kass & Raftery 1995 "very strong" -- and in
       either case an effect of >= 5 % of the amplitude),
       the period is doubled (up to 3 times: P/8 -> P) and refined again. This recovers
       binaries with unequal minima whose Lomb-Scargle peak lies at P/2 or P/4. Doubling
       is only tested while 2P fits >= 10 times into the baseline.
    3. A fold showing one narrow dip per cycle (:func:`eclipse_shape`) at least
       :data:`ECLIPSE_MIN_DOUBLING_DEPTH` deep is an eclipsing binary with equal minima:
       the period is doubled once more and the single-dip period reported as alternative
       (it is right only if one eclipse per orbit is visible). A shallower single dip is
       transit-like: P is kept and 2P reported as the alternative. A symmetric fold
       (:func:`symmetry_index` < 0.05) keeps P and reports 2P as the alternative
       (ellipsoidal/contact binary with equal minima).
    4. Final multi-harmonic refinement (up to 20 harmonics, for narrow eclipses) with
       Montgomery & O'Donoghue errors, widened to the alias spacing when a sampling gap
       makes the cycle count ambiguous (:func:`alias_comb`).
    5. Significance (:func:`period_significance`): multi-harmonic FAP against the local
       noise continuum of the residuals and, for sparse sampling, with the effective
       number of independent points -- removing the white-noise assumption that made
       red-noise AGN look periodic; a phase-coherence check below 20 cycles; then
       :func:`_assess_period`.
    """
    f_refined = _refine_peak(gram, step)
    if decision is None:
        decision = _period_ladder(gram, band_data, f_refined, ground_based=ground_based)
    _complete_period(gram, f_refined, decision, ground_based=ground_based, survey=survey)
    return decision


def lomb_scargle_period(t: np.ndarray, y: np.ndarray, dy: np.ndarray, *, series: str = "",
                        min_period_days: float = DEFAULT_MIN_PERIOD_DAYS, max_period_days: float | None = None,
                        oversampling: float = DEFAULT_OVERSAMPLING, n_peaks: int = 5,
                        ground_based: bool = False, survey: str | None = None, unit: str = "mag",
                        ) -> PeriodogramResult | None:
    """Generalised (floating-mean) Lomb-Scargle periodogram of one series.

    ``standard`` normalisation; the highest peak goes through the period ladder,
    refinement and significance tests of :func:`_finalise_period`; ``ground_based``
    marks peaks at diurnal/annual sampling aliases as unreliable and ``survey='gaia'``
    those at Gaia scanning-law frequencies. Returns None with fewer than
    :data:`MIN_POINTS_PERIOD` points.
    """
    t, y, dy = (np.asarray(a, dtype=float) for a in (t, y, dy))
    if len(t) < MIN_POINTS_PERIOD:
        return None
    baseline = float(t.max() - t.min())
    if baseline <= 0:
        return None
    try:
        freq, used_os = frequency_grid(baseline, min_period_days=min_period_days, max_period_days=max_period_days,
                                       oversampling=oversampling)
    except ValueError:
        return None
    gram = _periodogram(t, y, dy, freq, used_os, series=series, n_peaks=n_peaks, requested_os=oversampling, unit=unit)
    _finalise_period(gram, [(t, y, dy)], step=float(freq[1] - freq[0]), ground_based=ground_based, survey=survey)
    return gram.result


def diurnal_alias(frequency_per_day: float, baseline_days: float) -> str | None:
    """Name of the ground-based sampling alias near ``frequency_per_day``, if any.

    Nightly sampling from the ground imprints the solar day (1 c/d) and, because a
    field is observed near a fixed hour angle, the sidereal day (1.0027379 c/d =
    1 / 0.99726957 d) and their harmonics; the yearly observing season adds +-1 and
    +-2 cycles/yr side lobes to each and power at 1 and 2 cycles/yr itself (window
    function of ground-based surveys, VanderPlas 2018, ApJS 236, 16, sec. 7). A peak
    within two frequency-resolution elements (2 / baseline) of any of these is flagged.
    """
    width = 2.0 / baseline_days
    year = 1.0 / DAYS_PER_JULIAN_YEAR
    for m in (1, 2):
        if abs(frequency_per_day - m * year) < width:
            return "annual" if m == 1 else "semi-annual"
    # (the sidereal day is itself the solar day + 1/yr, so exact diurnal aliases are named first)
    for n in range(1, 5):
        for m in (0, -1, 1, -2, 2):
            for base, label in ((1.0, "solar day"), (SIDEREAL_DAY_FREQUENCY, "sidereal day")):
                if abs(frequency_per_day - (n * base + m * year)) < width:
                    name = f"{n} x {label}" if n > 1 else label
                    return name if m == 0 else f"{name} {'+' if m > 0 else '-'} {abs(m)}/yr"
    return None


def _gaia_scanning_candidates() -> list[tuple[float, str]]:
    """(frequency c/d, name) of the Gaia scanning-law frequencies (see :func:`gaia_scanning_alias`)."""
    spin = 4.0
    fov = 1440.0 / 106.5
    precession = 1.0 / 63.12
    candidates: list[tuple[float, str]] = [(n * spin, f"{n} x 6-h spin") for n in range(1, 9)]
    candidates += [(m * fov, f"{m} x 106.5-min field-of-view separation") for m in (1, 2)]
    candidates += [(abs(fov - n * spin), f"FoV separation - {n} x spin") for n in range(1, 4)]
    candidates += [(m * precession, f"{m} x 63-d precession") for m in range(1, 6)]
    return candidates


def gaia_scanning_alias(frequency_per_day: float, baseline_days: float) -> str | None:
    """Name of the Gaia scanning-law frequency near ``frequency_per_day``, if any.

    Gaia spins with a 6-h period (4 c/d), its two fields of view are separated by the
    106.5 deg basic angle (the same source is seen again 106.5 min later: 13.52 c/d) and
    the spin axis precesses with a 63.12-d period (Gaia Collaboration, Prusti et al. 2016,
    A&A 595, A1, sec. 5.2); the resulting sampling imprints spurious signals at these
    frequencies and their harmonics/combinations in Gaia DR3 time series (Holl et al.
    2023, A&A 674, A25). A peak within two resolution elements (2 / baseline) of any of
    them is flagged.
    """
    width = 2.0 / baseline_days
    for value, name in _gaia_scanning_candidates():
        if abs(frequency_per_day - value) < width:
            return name
    return None


def _window_frequency(result: PeriodogramResult, *, ground_based: bool,
                      survey: str | None) -> tuple[str, bool, str] | None:
    """(name, strict, kind: "ground" or "gaia") of the sampling-window frequency the peak coincides with (within 2/T; the
    Lomb-Scargle or the adopted frequency), if any. ``strict``: a diurnal alias of ground-based
    photometry (always an alias); otherwise an annual/semi-annual or Gaia scanning-law frequency,
    at which a strong, coherent signal may still be real (see :func:`_assess_period`)."""
    freqs = (1.0 / (result.lomb_scargle_period_days or result.best_period_days), result.best_frequency_per_day)
    if ground_based:
        for f in freqs:
            name = diurnal_alias(f, result.baseline_days)
            if name:
                return name, name not in ("annual", "semi-annual"), "ground"
    if survey == "gaia":
        for f in freqs:
            name = gaia_scanning_alias(f, result.baseline_days)
            if name:
                return name, False, "gaia"
    return None


def _real_at_window_frequency(result: PeriodogramResult) -> str | None:
    """Why a peak at an annual/semi-annual or Gaia scanning-law frequency is a real signal, or None.

    Such a peak is a sampling artefact when it is the window's image of low-frequency power
    (trends, red noise) or a small calibration pattern with the survey's seasonal/scanning
    geometry. It is kept when it cannot be either: significant against the noise continuum
    including the window-aliased red noise (:func:`red_noise_fap`) at FAP <
    :data:`PERIODIC_EVIDENCE_FAP`, outlier-robust, coherent between the halves of the data
    (:func:`phase_coherence`; not required for >= :data:`COHERENCE_MAX_CYCLES` cycles, where
    red noise cannot reach that FAP and a sparse half may not pin down a non-sinusoidal
    shape, e.g. the 15.57-d Gaia Cepheid 2686850313960864256 at 4 x the 63-d precession
    frequency), and -- for magnitudes -- with a semi-amplitude of at least
    :data:`WINDOW_ALIAS_MIN_AMPLITUDE_MAG`, well above the seasonal zero-point patterns of the
    surveys (ZTF repeatability 8-25 mmag, Masci et al. 2019). Miras near 1/yr (ZTF 321-d and
    200-d periods of 4-5-mag amplitude, FAP ~0) were otherwise never adopted.
    """
    test = result.harmonic_test or {}
    fap = result.false_alarm_probability
    if fap is None or fap >= PERIODIC_EVIDENCE_FAP:
        return None
    if result.robust_false_alarm_probability is not None and result.robust_false_alarm_probability >= PERIOD_FAP_THRESHOLD:
        return None
    coherence = test.get("phase_coherence")
    cycles = result.baseline_days / result.best_period_days
    coherent = coherence is not None and coherence["coherent"]
    if not coherent and not _many_cycle_evidence(result):
        return None
    amplitude = test.get("semi_amplitude")
    if test.get("unit", "mag") == "mag" and (amplitude is None or amplitude < WINDOW_ALIAS_MIN_AMPLITUDE_MAG):
        return None
    return (f"FAP {fap:.2g} < {PERIODIC_EVIDENCE_FAP:g} against the window-aliased noise, "
            + ("coherent in both halves" if coherent else f"{cycles:.0f} cycles")
            + (f", semi-amplitude {amplitude:.3g} mag" if amplitude is not None and test.get("unit", "mag") == "mag"
               else ""))


def _many_cycle_evidence(result: PeriodogramResult) -> bool:
    """At least :data:`COHERENCE_MAX_CYCLES` cycles at FAP < :data:`PERIODIC_EVIDENCE_FAP` (against
    the red-noise and window-aliased continuum) with an outlier-robust FAP < 1 %: evidence of
    periodicity that red noise does not produce, and which a failed half-vs-half comparison of
    a sparse, non-sinusoidal light curve does not overturn (see :func:`phase_coherence`)."""
    fap = result.false_alarm_probability
    if (result.harmonic_test or {}).get("continuum_only"):
        return False  # a continuum-only FAP is not that evidence (see _complete_period)
    return bool(result.baseline_days / result.best_period_days >= COHERENCE_MAX_CYCLES
                and fap is not None and fap < PERIODIC_EVIDENCE_FAP
                and (result.robust_false_alarm_probability or 0.0) < PERIOD_FAP_THRESHOLD)


def _assess_period(result: PeriodogramResult, *, ground_based: bool, survey: str | None = None) -> None:
    fap_ok = result.false_alarm_probability is not None and result.false_alarm_probability < PERIOD_FAP_THRESHOLD
    cycles = result.baseline_days / result.best_period_days
    if not fap_ok and result.false_alarm_probability is not None:
        result.warnings.append(f"FAP {result.false_alarm_probability:.3g} >= {PERIOD_FAP_THRESHOLD}: not significant")
    if cycles < MIN_CYCLES_IN_BASELINE:
        result.warnings.append(f"only {cycles:.1f} cycles in the baseline (< {MIN_CYCLES_IN_BASELINE:g}): "
                               "indistinguishable from a trend or red noise")
    test = result.harmonic_test or {}
    alias = None
    window = _window_frequency(result, ground_based=ground_based, survey=survey)
    if window is not None:
        name, strict, kind = window
        where = (f"the {name} sampling alias of ground-based photometry" if kind == "ground"
                 else f"the Gaia scanning-law frequency ({name})")
        why = None if strict else _real_at_window_frequency(result)
        if why is None:
            alias = name
            result.warnings.append(f"peak coincides with {where}")
        else:
            result.warnings.append(f"peak lies at {where}, but is a real signal: {why}")
    rn_alias = test.get("aliased_red_noise")
    if alias is None and rn_alias is not None and not (
            result.false_alarm_probability is not None and result.false_alarm_probability < PERIODIC_EVIDENCE_FAP):
        alias = "aliased red noise"
        result.warnings.append(
            f"peak lies in the band of red noise aliased by the {1.0 / rn_alias['centre']:.6g}-d sampling window "
            f"(aliased level {rn_alias['level']:.3g} vs local continuum {rn_alias['local_continuum']:.3g}; offset "
            f"{rn_alias['offset']:.3g} c/d): an alias of the source's low-frequency variability unless its FAP is "
            f"below {PERIODIC_EVIDENCE_FAP:g}")
    coherence = test.get("phase_coherence")
    incoherent = coherence is not None and not coherence["coherent"] and not _many_cycle_evidence(result)
    robust_ok = (result.robust_false_alarm_probability is None
                 or result.robust_false_alarm_probability < PERIOD_FAP_THRESHOLD)
    result.reliable = bool(fap_ok and robust_ok and cycles >= MIN_CYCLES_IN_BASELINE and alias is None
                           and not incoherent)


def _period_search_arrays(t: np.ndarray, y: np.ndarray, dy: np.ndarray, *, max_points: int = PERIOD_SEARCH_MAX_POINTS,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """(t, y, dy, bin width or None) for the period analysis of one series.

    The period search costs O(N) per trial frequency and per multi-harmonic fit (a 56,000-
    cadence unbinned 3-sector TESS series took 47 s, 187,000 cadences 190 s), so a series
    with more than ``max_points`` epochs is analysed on a copy mean-binned in time
    (:func:`bin_uniform`) to a width of k x its median cadence, k the smallest integer
    giving <= ``max_points`` bins. The returned series itself is never binned.
    """
    if len(t) <= max_points:
        return t, y, dy, None
    steps = np.diff(t)
    cadence = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    if cadence <= 0:
        return t, y, dy, None
    factor = math.ceil(len(t) / max_points)
    while True:
        width = factor * cadence
        points = bin_uniform(t, y, dy, width)
        if len(points) <= max_points:
            break
        factor += 1
    return (np.array([p.mjd for p in points]), np.array([p.value for p in points]),
            np.array([p.error for p in points], dtype=float), width)


def _usable_arrays(series_list: Sequence[LightCurveSeries], notes: list[str] | None = None,
                   ) -> list[tuple[LightCurveSeries, np.ndarray, np.ndarray, np.ndarray]]:
    """(series, t, y, sigma_eff) of series with enough epochs; sigma from the calibrated error model
    (applied after the binning of very long series, see :func:`_period_search_arrays`)."""
    out = []
    for s in series_list:
        t, y, dy = s.good_arrays()
        if len(t) >= MIN_POINTS_PERIOD and float(t.max() - t.min()) > 0:
            n_raw = len(t)
            t, y, dy, width = _period_search_arrays(t, y, dy)
            if width is not None and notes is not None:
                notes.append(f"{s.key}: period search on {len(t):,} bins of {width * 1440.0:.3g} min (mean of the "
                             f"{n_raw:,} epochs; limit {PERIOD_SEARCH_MAX_POINTS:,} points)")
            scale, floor = series_error_model(s)
            out.append((s, t, y, np.sqrt((scale * dy) ** 2 + floor**2)))
    return out


def _combine_multiband(grams: Sequence[_Periodogram], freq: np.ndarray, used_os: float, *, n_points: int,
                       baseline: float, ground_based: bool, n_peaks: int = 5,
                       survey: str | None = None) -> PeriodogramResult | None:
    """chi^2-weighted sum of single-band powers on a common grid (see :func:`multiband_period`).

    Only decisive bands enter (never Gaia BP/RP, :data:`NON_DECISIVE_SERIES`: their
    neighbour-contaminated window photometry has chi^2_0 up to 10x that of G and would
    dominate the weights); ``survey='gaia'`` flags a peak at a scanning-law frequency.
    """
    grams = [g for g in grams if _is_decisive(g.result.series)]
    chi2_sum = float(sum(g.chi2_0 for g in grams))
    if len(grams) < 2 or chi2_sum <= 0:
        return None
    power = sum(g.chi2_0 * g.power for g in grams) / chi2_sum
    best = int(np.argmax(power))
    result = PeriodogramResult(
        series="multiband", n=n_points, baseline_days=baseline, best_period_days=float(1.0 / freq[best]),
        best_frequency_per_day=float(freq[best]), power=float(power[best]), false_alarm_probability=None,
        min_frequency=float(freq[0]), max_frequency=float(freq[-1]), n_frequencies=len(freq), oversampling=used_os,
        method="multi-band Lomb-Scargle (chi2-weighted sum; VanderPlas & Ivezic 2015)",
        peaks=_top_peaks(freq, power, n_peaks=n_peaks, min_separation=1.0 / baseline),
        members=[g.result.series for g in grams], lomb_scargle_period_days=float(1.0 / freq[best]),
    )
    cycles = baseline / result.best_period_days
    if cycles < MIN_CYCLES_IN_BASELINE:
        result.warnings.append(f"only {cycles:.1f} cycles in the baseline")
    alias = diurnal_alias(result.best_frequency_per_day, baseline) if ground_based else None
    if alias:
        result.warnings.append(f"peak coincides with the {alias} sampling alias of ground-based photometry")
    if survey == "gaia":
        scan = gaia_scanning_alias(result.best_frequency_per_day, baseline)
        if scan:
            result.warnings.append(f"peak coincides with the Gaia scanning-law frequency ({scan})")
    result.warnings.append("no analytic false-alarm probability for the multi-band sum")
    return result


def multiband_period(series_list: Sequence[LightCurveSeries], *, min_period_days: float = DEFAULT_MIN_PERIOD_DAYS,
                     max_period_days: float | None = None, oversampling: float = DEFAULT_OVERSAMPLING,
                     n_peaks: int = 5) -> PeriodogramResult | None:
    """Shared-frequency multi-band periodogram (independent mean/amplitude/phase per band).

    For this model (N_base = 0, N_band = 1 of VanderPlas & Ivezic 2015) the chi^2
    reductions of the bands add, so P(f) = sum_b chi2_0,b P_b(f) / sum_b chi2_0,b,
    where P_b is the standard-normalised single-band periodogram and chi2_0,b the
    chi^2 of band b about its weighted mean. No analytic FAP exists for the sum. Only
    decisive bands are summed (Gaia BP/RP are not, see :func:`_combine_multiband`).
    """
    usable = _usable_arrays(series_list)
    if len(usable) < 2:
        return None
    t_all = np.concatenate([u_[1] for u_ in usable])
    baseline = float(t_all.max() - t_all.min())
    try:
        freq, used_os = frequency_grid(baseline, min_period_days=min_period_days, max_period_days=max_period_days,
                                       oversampling=oversampling)
    except ValueError:
        return None
    ground = any(s.survey in GROUND_BASED_SURVEYS for s, *_ in usable)
    grams = [_periodogram(t, y, sg, freq, used_os, series=s.key) for s, t, y, sg in usable]
    surveys = {s.survey for s, *_ in usable}
    return _combine_multiband(grams, freq, used_os, n_points=sum(len(g.t) for g in grams if _is_decisive(g.result.series)),
                              baseline=baseline, ground_based=ground, n_peaks=n_peaks,
                              survey=surveys.pop() if len(surveys) == 1 else None)


def _frequencies_agree(f1: float, f2: float, tolerance: float = 0.01) -> bool:
    return abs(f1 - f2) <= tolerance * max(f1, f2)


def _is_decisive(series_key: str) -> bool:
    survey, _, band = series_key.partition(":")
    return (survey, band) not in NON_DECISIVE_SERIES


def _apply_periodic_evidence(per_series: Sequence[PeriodogramResult],
                             variability: dict[str, VariabilityMetrics]) -> None:
    """Upgrade a series to variable on a very significant, cross-band confirmed period.

    Criteria: the series' own peak is reliable (see :func:`_assess_period`) with FAP <
    :data:`PERIODIC_EVIDENCE_FAP`, and the same Lomb-Scargle frequency (within 1 %) is among
    the top peaks of another *decisive* series (Gaia BP/RP window photometry cannot confirm;
    :data:`NON_DECISIVE_SERIES`). This recovers low-amplitude periodic variables (e.g.
    spotted rotators, delta Scuti stars, ellipsoidal binaries) whose variance is below the
    noise variance. The periodic signal is its own test of constancy -- FAP < 1e-10 rejects
    it far beyond 5 sigma, measured against the residuals' own noise level (so independent of
    the error-bar scale) -- so the chi^2 test is not required as well: with conservative
    errors a periodic star can have chi^2/dof < 1 (the Chen et al. 2020 delta Scuti
    ZTFJ210655.21+462600.4: chi^2/dof 0.83 in g, periodogram FAP 2e-27).
    """
    for r in per_series:
        m = variability.get(r.series)
        if m is None or m.is_variable is not False or not r.reliable:
            continue
        if r.false_alarm_probability is None or r.false_alarm_probability >= PERIODIC_EVIDENCE_FAP:
            continue
        f_ls = 1.0 / (r.lomb_scargle_period_days or r.best_period_days)
        confirming = [o.series for o in per_series if o.series != r.series and _is_decisive(o.series)
                      and any(_frequencies_agree(f_ls, p["frequency_per_day"]) for p in o.peaks)]
        if not confirming:
            continue
        m.is_variable = True
        m.decision_basis = "periodic"
        m.evidence.append(f"periodic signal at {r.best_period_days:.6g} d (FAP {r.false_alarm_probability:.2g} < "
                          f"{PERIODIC_EVIDENCE_FAP:g}), also found in {', '.join(confirming)}: variable")


def _joint_fit_score(grams: Sequence[_Periodogram], frequency: float, n_harmonics: int) -> float:
    """Sum over bands of the K-harmonic F statistic at ``frequency`` (each band normalised
    by its own residual variance, so mis-scaled errors do not weight a band)."""
    total = 0.0
    freqs = [j * frequency for j in range(1, n_harmonics + 1)]
    for g in grams:
        n = len(g.t)
        dof = n - 2 * n_harmonics - 1
        if dof < 1:
            continue
        chi2_k, _coef = _harmonic_fit(g.t, g.y, g.sigma, freqs)
        if chi2_k > 0:
            total += (g.chi2_0 - chi2_k) / (chi2_k / dof)
    return total


def _flag_cross_band_aliases(per_series: Sequence[PeriodogramResult], grams: dict[str, _Periodogram]) -> None:
    """Resolve daily aliases between the peaks of different bands.

    Two peaks n = 1-3 (solar or sidereal) cycles per day (+- 0-2 cycles per year) apart
    are daily aliases of each other (ground-based sampling; e.g. V445 Lyr, whose sparse
    ZTF g band peaked at f_true + 2 c/d + 1/yr while r found the RR Lyrae period). The
    two candidate frequencies are compared on the same data -- all decisive bands of
    both surveys -- by the summed K-harmonic F statistic (:func:`_joint_fit_score`); the
    peak whose frequency fits worse is marked unreliable.
    """
    year = 1.0 / DAYS_PER_JULIAN_YEAR
    for r in per_series:
        f_r = 1.0 / (r.lomb_scargle_period_days or r.best_period_days)
        width = 2.0 / r.baseline_days
        for o in per_series:
            if o is r or not _is_decisive(o.series):
                continue
            f_o = 1.0 / (o.lomb_scargle_period_days or o.best_period_days)
            if _frequencies_agree(f_r, f_o):
                continue
            hit = None
            for n in (1, 2, 3):
                for m in (0, -1, 1, -2, 2):
                    for base, label in ((1.0, "solar"), (SIDEREAL_DAY_FREQUENCY, "sidereal")):
                        if hit is None and abs(abs(f_r - f_o) - (n * base + m * year)) < width:
                            hit = f"{n} {label} cycle(s) per day" + (f" {m:+d}/yr" if m else "")
            if not hit:
                continue
            surveys = {r.series.partition(":")[0], o.series.partition(":")[0]}
            data = [g for key, g in grams.items() if key.partition(":")[0] in surveys and _is_decisive(key)]
            k = max(1, min(max(r.n_harmonics or 1, o.n_harmonics or 1), MAX_HARMONICS))
            score_r = _joint_fit_score(data, r.best_frequency_per_day, k)
            score_o = _joint_fit_score(data, o.best_frequency_per_day, k)
            if score_o > score_r:
                r.reliable = False
                r.warnings.append(f"daily alias of the {o.series} peak ({1.0 / f_o:.6g} d, {hit} apart), which fits "
                                  f"all bands better (summed F {score_o:.0f} vs {score_r:.0f}): not adopted")
                break


def _single_frequency_log_p(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, frequency: float,
                            n_harmonics: int, *, alias_centres: Sequence[float] = ()) -> float | None:
    """ln p of a K-harmonic signal at a *given* frequency (no search, no trials factor),
    the larger of the red-noise-continuum (window-aliased red noise included) and effective-N
    estimates (see :func:`period_significance`).

    None when the effective-N estimate applies but the residuals are so correlated that
    fewer effective points than model parameters remain (N_eff <= 2K + 1, e.g. the smooth
    cycle-to-cycle and secular changes of a Mira): such a band can neither confirm nor
    refute the signal.
    """
    base = float(t.max() - t.min())
    noise = red_noise_fap(t, y, sigma, frequency, n_harmonics, min_frequency=frequency / 3.0,
                          max_frequency=frequency * 3.0, alias_centres=alias_centres)
    log_p = noise["log_p"]
    if base > 0 and _clustered_or_sparse(t, frequency):
        _eta, n_eff = residual_effective_n(t, y, sigma, frequency, n_harmonics)
        if n_eff - 2 * n_harmonics - 1 < 1:
            return None
        lp, _fap = multiharmonic_fap(t, y, sigma, frequency, n_harmonics, min_frequency=frequency,
                                     max_frequency=frequency, n_effective=n_eff)
        log_p = max(log_p, lp)
    return log_p


def _require_cross_band_confirmation(per_series: Sequence[PeriodogramResult], grams: dict[str, _Periodogram]) -> None:
    """A period adopted from one band must be present in another band of the same survey.

    When the survey has other decisive bands with at least
    :data:`CONFIRMATION_MIN_FRACTION` of the epochs overlapping in time, at least one of
    them must show a K-harmonic signal (K <= 4) at the adopted frequency with
    single-frequency p < :data:`CONFIRMATION_P` (no search, so no trials factor). A real
    periodic variable is seen in every filter; a red-noise peak (e.g. the 232-d "period"
    of the SDSS quasar J074756.99+454527.7 in ZTF g) is not.

    Without such a band (Gaia G -- BP/RP window photometry cannot confirm --, TESS, or a
    single ZTF filter) the period needs FAP < :data:`UNCONFIRMED_PERIOD_FAP` and must
    persist in the two independent halves of the series (:func:`phase_coherence`,
    variances inflated for red noise), whatever the number of cycles: Gaia-only red-noise AGN otherwise kept
    "reliable" periods (e.g. 6.68 d, FAP 0.006, for the vari_agn source Gaia DR3
    6378923189372478592). Periods established by the eclipse/parity ladder are exempt:
    repeated narrow eclipses are periodic evidence of their own, and a Fourier series
    of a few harmonics cannot represent them in half the data (CM Dra). So is a signal of
    at least :data:`COHERENCE_MAX_CYCLES` cycles with FAP < :data:`PERIODIC_EVIDENCE_FAP`
    (and an outlier-robust FAP < 1 %): red noise does not produce it, whereas the halves of
    a sparse series cannot always pin down a non-sinusoidal shape (review: the Gaia DR3
    Cepheids 2049531563002338432, 4655167172163949184 and 4661366184375995520, FAPs 1e-19
    to 1e-32, were rejected as "incoherent").
    """
    for r in per_series:
        if not r.reliable:
            continue
        survey = r.series.partition(":")[0]
        own = grams.get(r.series)
        if own is None:
            continue
        others = [g for key, g in grams.items() if key != r.series and key.partition(":")[0] == survey
                  and _is_decisive(key) and g.result.n >= max(MIN_POINTS_PERIOD, CONFIRMATION_MIN_FRACTION * r.n)
                  and min(g.t.max(), own.t.max()) - max(g.t.min(), own.t.min()) > 0.5 * r.best_period_days]
        k = max(1, min(r.n_harmonics or 1, 4))
        centres = window_alias_centres(survey, ground_based=survey in GROUND_BASED_SURVEYS)
        results = {g.result.series: _single_frequency_log_p(g.t, g.y, g.sigma, r.best_frequency_per_day, k,
                                                            alias_centres=centres)
                   for g in others}
        decided = {key: lp for key, lp in results.items() if lp is not None}
        undecided = sorted(key for key, lp in results.items() if lp is None)
        if not decided:
            # No band that can test it: the period must persist in both halves of the series.
            test = r.harmonic_test or {}
            if test.get("phase_coherence") is None and not test.get("doubled_by"):
                coherence = phase_coherence(own.t, own.y, own.sigma, r.best_frequency_per_day,
                                            n_ratio=r.error_inflation or 1.0, n_harmonics=r.n_harmonics or 1)
                if r.harmonic_test is not None:
                    r.harmonic_test["phase_coherence"] = coherence
            else:
                coherence = test.get("phase_coherence")
            if undecided:
                r.warnings.append(f"the other band(s) of the survey ({', '.join(undecided)}) cannot test the period "
                                  "(too few effective independent points); phase coherence of this band required")
            fap = r.false_alarm_probability
            if not test.get("doubled_by") and fap is not None and fap >= UNCONFIRMED_PERIOD_FAP:
                r.reliable = False
                r.warnings.append(f"no other band of the survey can confirm the period, and its FAP {fap:.2g} is not "
                                  f"below {UNCONFIRMED_PERIOD_FAP:g} (required without confirmation)")
                continue
            if coherence is not None and not coherence["coherent"]:
                cycles = r.baseline_days / r.best_period_days
                if _many_cycle_evidence(r):
                    # Many cycles at FAP < 1e-10 against the red-noise (and window-aliased) continuum:
                    # red noise cannot produce that, while a sparse half cannot pin down the shape of
                    # a non-sinusoidal light curve (Gaia-only Cepheids, FAP 1e-19 to 1e-32).
                    r.warnings.append(f"the two halves of the data differ more than their scatter allows, but "
                                      f"{cycles:.0f} cycles at FAP {fap:.2g} < {PERIODIC_EVIDENCE_FAP:g} are not red "
                                      "noise: period kept")
                else:
                    r.reliable = False
                    r.warnings.append("no other band of the survey can confirm the period, and the signal is not "
                                      "the same in the two halves of the data (phase/amplitude differ, or a half "
                                      "shows no significant signal): typical of red noise")
            continue
        if any(lp < math.log(CONFIRMATION_P) for lp in decided.values()):
            continue
        r.reliable = False
        r.warnings.append("period not confirmed by the other band(s) of the survey (" + ", ".join(
            f"{k_}: p = {math.exp(v):.2g}" for k_, v in decided.items()) + f" >= {CONFIRMATION_P:g})")


def search_periods(series_list: Sequence[LightCurveSeries], *, min_period_days: float = DEFAULT_MIN_PERIOD_DAYS,
                   max_period_days: float | None = None, oversampling: float = DEFAULT_OVERSAMPLING,
                   variability: dict[str, VariabilityMetrics] | None = None,
                   time_budget_s: float | None = None) -> PeriodSearch:
    """Run per-series and multi-band periodograms; adopt the most significant reliable period.

    Only surveys in :data:`PERIOD_SURVEYS` are searched. All bands of one survey
    share a frequency grid built from the survey's combined baseline, so each band's
    periodogram is computed once and reused for that survey's multi-band sum (the
    reported multi-band result is that of the survey with the most epochs). Errors
    follow the calibrated survey error model.

    Each series' peak then goes through :func:`_finalise_period`: the period ladder
    (cycle parity and eclipse shape, tested jointly on the survey's decisive bands --
    never on Gaia BP/RP), multi-harmonic refinement and a false-alarm probability that
    accounts for correlated residuals (red noise).

    A per-series peak is ``reliable`` when its FAP < 1 % (and its outlier-robust,
    rank-based FAP too, :func:`rank_fap`), the baseline covers >= 5
    cycles (and the signal is phase-coherent between the halves of the data for < 20
    cycles), it is not a diurnal/annual sampling alias (ground-based surveys), a Gaia
    scanning-law frequency (Gaia) or a daily alias of a stronger peak in another band,
    it is confirmed by another decisive band of the survey -- or, without one, is
    coherent between the halves of the data (:func:`_require_cross_band_confirmation`) --
    and -- when ``variability`` is given -- the series itself is variable. Before that
    gate, ``variability`` is updated in place by :func:`_apply_periodic_evidence`. The
    adopted period is the reliable peak with the lowest FAP; None when there is none.

    The search is bounded in time: once ``time_budget_s`` (default
    :data:`PERIOD_SEARCH_BUDGET_S`) has elapsed, the remaining series are not analysed and a
    note names them (the first survey's lead series is always analysed; fine grids -- min_period_days
    = 0.01 d at 50 samples per peak -- took
    ~60 s for six series); the results computed so far are returned.
    """
    per_series: list[PeriodogramResult] = []
    multibands: list[PeriodogramResult] = []
    notes: list[str] = []
    all_grams: dict[str, _Periodogram] = {}
    budget = PERIOD_SEARCH_BUDGET_S if time_budget_s is None else float(time_budget_s)
    started = time.perf_counter()
    skipped: list[str] = []

    def over_budget() -> bool:
        return time.perf_counter() - started > budget

    for survey in PERIOD_SURVEYS_ORDER:
        members = [s for s in series_list if s.survey == survey]
        if members and per_series and over_budget():  # (the first survey's lead series is always analysed)
            skipped.extend(s.key for s in members)
            continue
        usable = _usable_arrays(members, notes)
        if not usable:
            continue
        t_all = np.concatenate([u_[1] for u_ in usable])
        baseline = float(t_all.max() - t_all.min())
        try:
            freq, used_os = frequency_grid(baseline, min_period_days=min_period_days, max_period_days=max_period_days,
                                           oversampling=oversampling)
        except ValueError as exc:
            notes.append(f"{survey}: period search skipped: {exc}")
            continue
        ground = survey in GROUND_BASED_SURVEYS
        step = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0 / (oversampling * baseline)
        grams = [_periodogram(t, y, sg, freq, used_os, series=s.key, requested_os=oversampling, unit=s.unit)
                 for s, t, y, sg in usable]
        for gram, (s, *_rest) in zip(grams, usable):
            gram.host_radius_rsun = _num((s.metadata or {}).get("tic_radius_rsun"))
        decisive = [i for i, item in enumerate(usable) if (item[0].survey, item[0].band) not in NON_DECISIVE_SERIES]
        band_data = [(grams[i].t, grams[i].y, grams[i].sigma) for i in decisive]
        f_refined = [_refine_peak(g, step) for g in grams]
        # The ladder is decided once per signal, on the survey's best-determined series
        # (lowest Lomb-Scargle FAP): a less precise frequency of a sparse band would
        # dephase the well-sampled bands in the joint parity test.
        lead = min(decisive or range(len(grams)),
                   key=lambda i: (grams[i].result.lomb_scargle_false_alarm_probability or 0.0, -grams[i].result.n))
        lead_decision = _period_ladder(grams[lead], band_data, f_refined[lead], ground_based=ground)
        for i, gram in enumerate(grams):
            if i != lead and over_budget():
                skipped.append(gram.result.series)
                continue
            if i == lead or _frequencies_agree(f_refined[i], f_refined[lead]):
                decision = lead_decision
            else:
                decision = _period_ladder(gram, [(gram.t, gram.y, gram.sigma)], f_refined[i], ground_based=ground)
            _complete_period(gram, f_refined[i], decision, ground_based=ground, survey=survey)
            per_series.append(gram.result)
            all_grams[gram.result.series] = gram
        mags = [g for g in grams if g.unit == "mag" and _is_decisive(g.result.series)]
        multi = _combine_multiband(mags, freq, used_os, n_points=int(sum(g.result.n for g in mags)),
                                   baseline=baseline, ground_based=ground, survey=survey)
        if multi is not None:
            multibands.append(multi)
    if skipped:
        notes.append(f"period search stopped after {time.perf_counter() - started:.0f} s (time budget {budget:g} s): "
                     f"{', '.join(skipped)} not analysed; a coarser grid (larger min_period_days, lower oversampling) "
                     "is faster")
    _flag_cross_band_aliases(per_series, all_grams)
    _require_cross_band_confirmation(per_series, all_grams)
    if variability is not None:
        _apply_periodic_evidence(per_series, variability)
        for r in per_series:
            metrics = variability.get(r.series)
            if metrics is None or metrics.is_variable is not True:
                r.reliable = False
                r.warnings.append("series does not pass the variability test: peak not adopted")
    multiband = max(multibands, key=lambda m: m.n) if multibands else None
    reliable = [r for r in per_series if r.reliable]
    best = min(reliable, key=lambda r: (r.false_alarm_probability or 0.0, -r.n)) if reliable else None
    if best is not None and multiband is not None:
        ls_period = best.lomb_scargle_period_days or best.best_period_days
        agree = abs(multiband.best_period_days - ls_period) / ls_period < 0.01
        best.warnings.append(f"multi-band periodogram peak {multiband.best_period_days:.6g} d "
                             + ("agrees (within 1%) with the Lomb-Scargle peak" if agree else "differs"))
    return PeriodSearch(per_series, multiband, best, notes)


# ---------------------------------------------------------------------------
# Light-Curve Orchestration
# ---------------------------------------------------------------------------


def parse_surveys(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalise a comma-separated survey list; raises ValueError on unknown names."""
    if value is None or value == "" or value == []:
        return DEFAULT_SURVEYS
    items = value.split(",") if isinstance(value, str) else list(value)
    names = []
    for item in items:
        name = str(item).strip().lower()
        if not name:
            continue
        if name not in SURVEYS:
            raise ValueError(f"unknown survey '{name}'; valid: {', '.join(SURVEYS)}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("no survey requested")
    return tuple(names)


# Epoch at which each survey's positions are compared with the target:
#  Gaia DR3 source positions are at J2016.0 (Lindegren et al. 2021, A&A 649, A2);
#  TIC-8 positions (MAST s_ra/s_dec of SPOC targets) at J2000.0 (Stassun et al. 2019);
#  ZTF object positions come from reference images of different epochs (see
#  ZTF_POSITION_EPOCH_RANGE): the cone is centred on the middle of that range and
#  widened by |pm| x half-range, and each object is then matched to the target's track
#  (select_ztf_objects);
#  NEOWISE-R exposures (2013.95-2024.59) are matched one by one at their own epoch,
#  the cone widened by |pm| x half-span around the middle epoch.
SURVEY_POSITION_EPOCH: dict[str, tuple[float, float]] = {
    # survey: (reference epoch, half-span in years used to widen the cone)
    "gaia": (2016.0, 0.0),
    "tess": (2000.0, 0.0),
    "ztf": ((ZTF_POSITION_EPOCH_RANGE[0] + ZTF_POSITION_EPOCH_RANGE[1]) / 2.0,
            (ZTF_POSITION_EPOCH_RANGE[1] - ZTF_POSITION_EPOCH_RANGE[0]) / 2.0),
    "neowise": ((NEOWISE_EPOCH_RANGE[0] + NEOWISE_EPOCH_RANGE[1]) / 2.0,
                (NEOWISE_EPOCH_RANGE[1] - NEOWISE_EPOCH_RANGE[0]) / 2.0),
}

# Wall-clock limits (s). A single survey may chain several requests (ZTF: TAP schema,
# objects, light curves; TESS: MAST query, product lists and downloads), each with its
# own timeout and one retry; the survey deadline bounds the whole chain, and because
# surveys run concurrently it also bounds a light-curve request.
SURVEY_DEADLINES: dict[str, float] = {
    "ztf": float(os.getenv("TIMEDOMAIN_ZTF_DEADLINE_S", "480")),
    "neowise": float(os.getenv("TIMEDOMAIN_NEOWISE_DEADLINE_S", "360")),
    "gaia": float(os.getenv("TIMEDOMAIN_GAIA_DEADLINE_S", "240")),
    "tess": float(os.getenv("TIMEDOMAIN_TESS_DEADLINE_S", "480")),
}
# Whole-call bounds of the other upstream services (their per-read timeouts do not bound a
# response that trickles bytes, nor the retries): read at call time, so they can be tuned.
SKYBOT_DEADLINE_S = float(os.getenv("TIMEDOMAIN_SKYBOT_DEADLINE_S", "300"))
HORIZONS_DEADLINE_S = float(os.getenv("TIMEDOMAIN_HORIZONS_DEADLINE_S", "120"))
RESOLVER_DEADLINE_S = float(os.getenv("TIMEDOMAIN_RESOLVER_DEADLINE_S", "60"))


@asynccontextmanager
async def _deadline(service: str, seconds: float) -> AsyncIterator[None]:
    """Bound a whole upstream exchange (all requests, retries and pauses) by ``seconds``;
    exceeding it raises a retryable :class:`UpstreamServiceError` (HTTP 502 in the router)."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as exc:
        raise UpstreamServiceError(service, f"no complete answer within the {seconds:g}-s deadline",
                                   retryable=True) from exc
# TESS request bounds (core API, CLI and router alike): each 2-min sector is ~2 MB and
# ~18,000-20,000 cadences. The number of returned points is bounded directly: a bin
# narrower than the 2-min SPOC cadence does not bin at all (bin_uniform), so the points
# per sector are 27.4 d / max(bin, 2 min), and sectors x that must stay within the
# points of MAX_UNBINNED_TESS_SECTORS unbinned sectors (~59,000 points, ~6 MB of JSON).
MAX_TESS_SECTORS = 10
MAX_UNBINNED_TESS_SECTORS = 3
MAX_TESS_BIN_MINUTES = 1440.0
TESS_CADENCE_MINUTES = 2.0
TESS_SECTOR_DAYS = 27.4
MAX_TESS_POINTS = int(MAX_UNBINNED_TESS_SECTORS * TESS_SECTOR_DAYS * 1440.0 / TESS_CADENCE_MINUTES)
# fetch_tess stops starting sector requests this long (s) before the survey deadline, so the
# sectors already downloaded can be assembled and returned.
TESS_ASSEMBLY_MARGIN_S = 15.0


def tess_points_estimate(max_sectors: int, bin_minutes: float) -> int:
    """Upper estimate of the TESS points returned for ``max_sectors`` sectors binned to
    ``bin_minutes`` (0, or anything below the 2-min cadence, returns every cadence)."""
    width = max(float(bin_minutes), TESS_CADENCE_MINUTES)
    return int(max_sectors * TESS_SECTOR_DAYS * 1440.0 / width)


def _validate_tess_options(max_sectors: int, bin_minutes: float) -> None:
    if isinstance(max_sectors, bool) or not isinstance(max_sectors, int) or not 1 <= max_sectors <= MAX_TESS_SECTORS:
        raise ValueError(f"tess_max_sectors must be an integer in [1, {MAX_TESS_SECTORS}]")
    if not (math.isfinite(bin_minutes) and 0 <= bin_minutes <= MAX_TESS_BIN_MINUTES):
        raise ValueError(f"tess_bin_minutes must be within [0, {MAX_TESS_BIN_MINUTES:g}] (0 = no binning)")
    if tess_points_estimate(max_sectors, bin_minutes) > MAX_TESS_POINTS:
        needed = TESS_SECTOR_DAYS * 1440.0 * max_sectors / MAX_TESS_POINTS
        raise ValueError(
            f"{max_sectors} TESS sectors at tess_bin_minutes = {bin_minutes:g} would return ~"
            f"{tess_points_estimate(max_sectors, bin_minutes):,} points (limit {MAX_TESS_POINTS:,}; a bin below the "
            f"{TESS_CADENCE_MINUTES:g}-min cadence, or 0, does not bin): use tess_bin_minutes >= {math.ceil(needed * 10) / 10:g} "
            f"or at most {MAX_UNBINNED_TESS_SECTORS} unbinned sectors")


def survey_cone(target: Target, survey: str, radius_arcsec: float) -> tuple[float, float, float, str | None]:
    """(ra, dec, radius) at which ``survey`` is queried for ``target``, plus a note.

    With a target epoch and proper motion the position is propagated to the survey's
    epoch (:data:`SURVEY_POSITION_EPOCH`) and the cone widened by the motion over the
    survey's span. Without them the input position is used as given.
    """
    epoch, half_span = SURVEY_POSITION_EPOCH[survey]
    if target.epoch is None:
        return target.ra, target.dec, radius_arcsec, None
    if target.proper_motion is None:
        return (target.ra, target.dec, radius_arcsec,
                f"{survey}: proper motion unknown; the J{target.epoch:g} position is used unpropagated")
    ra, dec = position_at(target, epoch)
    pm_total = math.hypot(*target.proper_motion) / 1000.0  # arcsec / yr
    moved = pm_total * abs(epoch - target.epoch)
    pad = pm_total * half_span
    note = None
    if moved >= 0.05 or pad >= 0.05:
        note = (f"{survey}: position propagated by {moved:.2f}\" from J{target.epoch:g} to J{epoch:g} "
                f"(pm {pm_total * 1000:.0f} mas/yr)" + (f"; cone widened by {pad:.2f}\" for the survey span" if pad >= 0.05
                                                        else ""))
    return ra, dec, radius_arcsec + pad, note


async def fetch_survey(client: httpx.AsyncClient, survey: str, ra: float, dec: float, radius_arcsec: float, *,
                       include_flagged: bool = False, ztf_collection: str | None = None, tess_max_sectors: int = 2,
                       tess_bin_minutes: float = 10.0, neowise_binned: bool = True, target: Target | None = None,
                       use_cache: bool = True, deadline_s: float | None = None) -> SurveyResult:
    """Fetch one survey (cached by position/options for TIMEDOMAIN_CACHE_TTL_SECONDS).

    ``target`` (epoch + proper motion) is used by NEOWISE to match each exposure at its
    own epoch and by ZTF to match objects along the target's track; (ra, dec, radius) is
    the survey cone (see :func:`survey_cone`). The whole survey is bounded by
    ``deadline_s`` (default :data:`SURVEY_DEADLINES`); exceeding it raises a retryable
    :class:`UpstreamServiceError`. Only successful, validated, complete answers are cached
    (failures raise before the cache is written; a partial answer -- e.g. TESS with a failed
    sector, ``provenance["partial"]`` -- is not cached). Cache reads and writes -- Redis I/O
    included -- and their (de)serialisation run on the cache's own executor, each bounded by
    :data:`CACHE_IO_TIMEOUT_S` (a stalled cache is a miss), and the key carries
    :data:`TIMEDOMAIN_CACHE_VERSION`.
    """
    if survey not in SURVEYS:
        raise ValueError(f"unknown survey '{survey}'")
    ttl = cache_ttl_seconds() if use_cache else 0
    motion = None
    if target is not None and target.epoch is not None and target.proper_motion is not None:
        motion = (target.ra, target.dec, target.epoch, *target.proper_motion, target.parallax_mas)
    key = CacheManager.make_key("timedomain", TIMEDOMAIN_CACHE_VERSION, survey, round(ra, 7), round(dec, 7),
                                round(radius_arcsec, 4),
                                include_flagged, ztf_collection, tess_max_sectors, tess_bin_minutes, neowise_binned,
                                motion if survey in ("neowise", "ztf") else None)
    cache = get_cache() if ttl > 0 else None
    if cache is not None:
        # Redis I/O and the (de)serialisation of a potentially large payload are blocking: they
        # run on the cache executor, bounded in time (see _cache_io).
        hit = await _cache_io(_cache_get, cache, key)
        if hit is not None:
            hit.provenance["cache"] = "hit"
            return hit
    limit = deadline_s if deadline_s is not None else SURVEY_DEADLINES[survey]
    try:
        async with asyncio.timeout(limit):
            if survey == "ztf":
                result = await fetch_ztf(client, ra, dec, radius_arcsec, collection=ztf_collection,
                                         include_flagged=include_flagged, target=target)
            elif survey == "neowise":
                result = await fetch_neowise(client, ra, dec, radius_arcsec, binned=neowise_binned, target=target,
                                             include_flagged=include_flagged)
            elif survey == "gaia":
                result = await fetch_gaia(client, ra, dec, radius_arcsec, include_flagged=include_flagged)
            else:
                result = await fetch_tess(client, ra, dec, radius_arcsec, max_sectors=tess_max_sectors,
                                          bin_minutes=tess_bin_minutes, include_flagged=include_flagged,
                                          deadline_s=limit)
    except TimeoutError as exc:
        raise UpstreamServiceError(survey, f"no complete answer within the {limit:g}-s survey deadline",
                                   retryable=True) from exc
    result.provenance["retrieved_at"] = datetime.now(UTC).isoformat()
    if cache is not None and not result.provenance.get("partial"):
        await _cache_io(_cache_set, cache, key, result, ttl)
    return result


def _validate_period_range(min_period_days: float, max_period_days: float | None) -> None:
    if not (math.isfinite(min_period_days) and MIN_ALLOWED_PERIOD_DAYS <= min_period_days <= MAX_ALLOWED_PERIOD_DAYS):
        raise ValueError(f"min_period_days must be within [{MIN_ALLOWED_PERIOD_DAYS:g}, {MAX_ALLOWED_PERIOD_DAYS:g}] d")
    if max_period_days is not None and not (math.isfinite(max_period_days) and max_period_days <= MAX_ALLOWED_PERIOD_DAYS):
        raise ValueError(f"max_period_days must be finite and <= {MAX_ALLOWED_PERIOD_DAYS:g} d")
    if max_period_days is not None and max_period_days <= min_period_days:
        raise ValueError("require min_period_days < max_period_days")


async def get_lightcurves(
    ra: float | str,
    dec: float | str,
    *,
    radius_arcsec: float = 3.0,
    surveys: str | Sequence[str] | None = None,
    client: httpx.AsyncClient | None = None,
    name: str | None = None,
    epoch: float | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
    resolver: dict[str, Any] | None = None,
    include_flagged: bool = False,
    period: bool = True,
    min_period_days: float = DEFAULT_MIN_PERIOD_DAYS,
    max_period_days: float | None = None,
    oversampling: float = DEFAULT_OVERSAMPLING,
    pair_window_days: float = DEFAULT_PAIR_WINDOW_DAYS,
    ztf_collection: str | None = None,
    tess_max_sectors: int = 2,
    tess_bin_minutes: float = 10.0,
    neowise_binned: bool = True,
    use_cache: bool = True,
) -> LightCurveResult:
    """Multi-survey light curves with variability metrics and a period search.

    ``epoch`` (Julian year of ra/dec) with ``pm_ra_masyr``/``pm_dec_masyr`` (mas/yr,
    RA component including cos dec) propagates the position to each survey's epoch
    (:func:`survey_cone`); without an epoch the position is used as given. ``parallax_mas``
    widens the per-epoch identity tolerance of moving targets by the parallactic ellipse
    (ZTF: :func:`parse_ztf_csv`). Surveys are
    queried concurrently; a failing survey is reported in ``failures`` while the others
    still return. Raises :class:`UpstreamServiceError` only when every requested
    survey failed. A proper motion without ``epoch`` is rejected (it could not be
    applied), as are TESS options outside the bounds of :func:`_validate_tess_options`.
    """
    if epoch is None and (pm_ra_masyr is not None or pm_dec_masyr is not None):
        raise InvalidCoordinateError("epoch (Julian year of ra/dec) is required with a proper motion; "
                                     "without it the proper motion cannot be applied")
    target = validate_target(ra, dec, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                             parallax_mas=parallax_mas)
    _validate_tess_options(tess_max_sectors, tess_bin_minutes)
    if not (math.isfinite(float(radius_arcsec)) and 0 < float(radius_arcsec) <= MAX_LIGHTCURVE_RADIUS_ARCSEC):
        raise InvalidCoordinateError(f"radius_arcsec must be in (0, {MAX_LIGHTCURVE_RADIUS_ARCSEC:g}].")
    _validate_period_range(min_period_days, max_period_days)
    if not 1 <= oversampling <= 50:
        raise ValueError("oversampling must be within [1, 50]")
    if ztf_collection is not None and not _ZTF_COLLECTION.match(ztf_collection):
        raise ValueError("ztf_collection must look like 'ztf_dr23'")
    names = parse_surveys(surveys)
    cones = {s: survey_cone(target, s, float(radius_arcsec)) for s in names}
    async with client_scope(client) as active:
        outcomes = await asyncio.gather(
            *(fetch_survey(active, s, cones[s][0], cones[s][1], cones[s][2], include_flagged=include_flagged,
                           ztf_collection=ztf_collection, tess_max_sectors=tess_max_sectors,
                           tess_bin_minutes=tess_bin_minutes, neowise_binned=neowise_binned, target=target,
                           use_cache=use_cache) for s in names),
            return_exceptions=True,
        )
    series: list[LightCurveSeries] = []
    failures: list[dict[str, Any]] = []
    notes: list[str] = [c[3] for c in cones.values() if c[3]]
    provenance: dict[str, Any] = {}
    for survey, outcome in zip(names, outcomes):
        if isinstance(outcome, UpstreamServiceError):
            failures.append({"survey": survey, "service": outcome.service, "error": outcome.message,
                             "status_code": outcome.status_code, "retryable": outcome.retryable})
        elif isinstance(outcome, BaseException):
            if isinstance(outcome, InvalidCoordinateError):
                raise outcome
            # Anything else (e.g. an unparseable upstream answer) is that survey's failure.
            failures.append({"survey": survey, "service": survey, "error": f"{type(outcome).__name__}: {outcome}",
                             "status_code": None, "retryable": False})
        else:
            series.extend(outcome.series)
            notes.extend(outcome.notes)
            provenance[survey] = outcome.provenance
    if failures and len(failures) == len(names):
        detail = "; ".join(f"{f['survey']}: {f['error']}" for f in failures)
        raise UpstreamServiceError("lightcurves", f"all requested surveys failed ({detail})",
                                   retryable=any(f["retryable"] for f in failures))

    def analyse() -> tuple[dict[str, VariabilityMetrics], PeriodSearch | None]:
        metrics = {s.key: series_variability(s, pair_window_days=pair_window_days,
                                             companions=[o for o in series if o.survey == s.survey and o is not s])
                   for s in series}
        search = (search_periods(series, min_period_days=min_period_days, max_period_days=max_period_days,
                                 oversampling=oversampling, variability=metrics) if period else None)
        return metrics, search

    # CPU-bound (periodograms take seconds): on the bounded analysis executor, never the default
    # one the application shares (a thread cannot be cancelled; search_periods bounds its time).
    variability, period_search = await asyncio.get_running_loop().run_in_executor(_analysis_executor, analyse)
    provenance["time_system"] = f"{TIME_SYSTEM} (BJD_TDB - 2400000.5; geocentric observer)"
    target_dict: dict[str, Any] = {"ra": target.ra, "dec": target.dec, "radius_arcsec": float(radius_arcsec),
                                   "surveys": list(names), "epoch": target.epoch,
                                   "pm_ra_masyr": target.pm_ra_masyr, "pm_dec_masyr": target.pm_dec_masyr,
                                   "parallax_mas": target.parallax_mas,
                                   "survey_positions": {s: {"ra": c[0], "dec": c[1], "radius_arcsec": c[2]}
                                                        for s, c in cones.items()}}
    if name:
        target_dict["name"] = name
    if resolver is not None:
        target_dict["resolver"] = resolver
    return LightCurveResult(target_dict, series, variability, period_search, failures, notes, provenance)


class SesameUnusableAnswer(httpx.HTTPError):
    """Sesame answered HTTP 200 with something that is not a Sesame XML document
    (empty body, HTML maintenance page, truncated XML): a service failure, not an unknown name."""


# Every Sesame -oxp answer -- including the "*** Nothing found ***" one -- is an XML document
# whose root element is <Sesame>.
_SESAME_ROOT = re.compile(rb"^\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--.*?-->\s*|<!DOCTYPE[^>]*>\s*)*<Sesame[\s>]", re.DOTALL)


class _StatusCheckingClient:
    """Wrap an httpx client so failures of the resolver service raise httpx errors.

    :class:`providers.SesameResolver` turns transport failures into
    ObjectResolutionError chained to an httpx exception; with this wrapper an HTTP >= 400
    answer, and an HTTP 200 answer that is not a Sesame XML document (empty body, HTML
    page, truncated XML), are chained the same way, so callers can tell "service failed"
    (502) from "name not found" (404, a well-formed Sesame answer without a position) by
    exception type.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        response = await self._client.get(url, **kwargs)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
        if not _SESAME_ROOT.match(response.content[:4096]):
            snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text[:300])).strip()
            raise SesameUnusableAnswer(f"HTTP {response.status_code} answer is not a Sesame XML document"
                                       + (f": {snippet[:150]}" if snippet else " (empty body)"))
        if b"</Sesame>" not in response.content[-4096:]:
            raise SesameUnusableAnswer(f"HTTP {response.status_code} Sesame answer is truncated")
        return response


async def resolve_target(name: str, client: httpx.AsyncClient | None = None, *,
                         deadline_s: float | None = None) -> tuple[Target, dict[str, Any]]:
    """Resolve an object name with CDS Sesame; returns (Target with epoch/proper motion, resolver record).

    Raises ValueError for a blank name, :class:`UpstreamServiceError` (retryable) when
    Sesame cannot be reached, answers with an HTTP error or does not answer completely
    within ``deadline_s`` (default :data:`RESOLVER_DEADLINE_S`), and ObjectResolutionError
    when the name is unknown.
    """
    if not str(name or "").strip():
        raise ValueError("object name must not be empty")
    limit = RESOLVER_DEADLINE_S if deadline_s is None else float(deadline_s)
    async with client_scope(client) as active, _deadline("sesame", limit):
        try:
            resolved = await SesameResolver(_StatusCheckingClient(active)).resolve(name)  # type: ignore[arg-type]
        except ObjectResolutionError as exc:
            if isinstance(exc.__cause__, httpx.HTTPError):
                raise UpstreamServiceError("sesame", str(exc), retryable=True) from exc
            raise
    return resolved_target(resolved), resolved.as_dict()


async def resolve_name(name: str, client: httpx.AsyncClient | None = None) -> tuple[float, float, dict[str, Any]]:
    """Resolve an object name with CDS Sesame; returns (ra, dec, resolver record).

    Prefer :func:`resolve_target`, which keeps the epoch and proper motion needed to
    follow high proper-motion stars across survey epochs.
    """
    target, record = await resolve_target(name, client)
    return target.ra, target.dec, record


def _check_name_motion(epoch: float | None, pm_ra_masyr: float | None, pm_dec_masyr: float | None) -> None:
    """Reject motion options that cannot go with coordinates taken from the name resolver.

    The resolver's position is valid at the resolver's epoch, so ``epoch`` must not be
    given: it would relabel those coordinates (Sesame's J2000 position of Barnard's star
    taken as J2016 misses the star by 166"). A proper motion may override the resolver's,
    but only with both components. Raises ValueError (before any request is made).
    """
    if epoch is not None:
        raise ValueError("epoch describes user-supplied ra/dec; with a name (and no ra/dec) the coordinates and "
                         "their epoch come from the name resolver -- pass ra and dec to use another epoch")
    if (pm_ra_masyr is None) != (pm_dec_masyr is None):
        raise ValueError("pm_ra_masyr and pm_dec_masyr must be given together")


def _resolved_motion(resolved: Target, pm_ra_masyr: float | None, pm_dec_masyr: float | None,
                     ) -> tuple[float | None, float | None, float | None]:
    """(epoch, pm_ra, pm_dec) for resolver coordinates: the resolver's epoch, and its proper
    motion unless both components are given (see :func:`_check_name_motion`)."""
    if pm_ra_masyr is None or pm_dec_masyr is None:
        return resolved.epoch, resolved.pm_ra_masyr, resolved.pm_dec_masyr
    if resolved.epoch is None:
        raise ValueError("the name resolver gave no epoch for its position, so a proper motion cannot be applied; "
                         "pass ra, dec and epoch")
    return resolved.epoch, pm_ra_masyr, pm_dec_masyr


async def get_lightcurves_by_name(name: str, *, client: httpx.AsyncClient | None = None,
                                  **kwargs: Any) -> LightCurveResult:
    """Resolve ``name`` (Sesame) and fetch its light curves with epoch/proper-motion propagation.

    ``kwargs`` are those of :func:`get_lightcurves` except ra/dec/name/resolver. The
    position and its epoch come from the resolver, so ``epoch`` is rejected (ValueError,
    before any request); ``pm_ra_masyr``/``pm_dec_masyr`` (both) override the resolver's
    proper motion (see :func:`_check_name_motion`), ``parallax_mas`` its parallax (which is
    otherwise passed on: it widens the ZTF per-epoch tolerance of nearby stars).
    """
    for key in ("ra", "dec", "name", "resolver"):
        if key in kwargs:
            raise TypeError(f"get_lightcurves_by_name() takes no '{key}' argument (it comes from the resolver)")
    pm_ra, pm_dec = kwargs.pop("pm_ra_masyr", None), kwargs.pop("pm_dec_masyr", None)
    parallax = kwargs.pop("parallax_mas", None)
    _check_name_motion(kwargs.pop("epoch", None), pm_ra, pm_dec)
    async with client_scope(client) as active:
        target, record = await resolve_target(name, active)
        epoch, pm_ra, pm_dec = _resolved_motion(target, pm_ra, pm_dec)
        return await get_lightcurves(target.ra, target.dec, client=active, name=name, epoch=epoch,
                                     pm_ra_masyr=pm_ra, pm_dec_masyr=pm_dec,
                                     parallax_mas=parallax if parallax is not None else target.parallax_mas,
                                     resolver=record, **kwargs)


# ---------------------------------------------------------------------------
# Solar System: IMCCE SkyBoT & JPL Horizons
# ---------------------------------------------------------------------------


def _skybot_type(object_class: str) -> str:
    head = object_class.split(">")[0].strip().lower()
    if head.startswith("comet"):
        return "comet"
    if head == "planet":
        return "planet"
    if head == "satellite":
        return "satellite"
    return "asteroid"


def _sexagesimal_coords(items: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    coords = SkyCoord([str(i["RA (hms)"]) for i in items], [str(i["DEC (dms)"]) for i in items],
                      unit=(u.hourangle, u.deg), frame="icrs")
    return np.atleast_1d(np.asarray(coords.ra.deg, dtype=float)), np.atleast_1d(np.asarray(coords.dec.deg, dtype=float))


def parse_skybot_json(payload: Any, ra: float, dec: float, *, notes: list[str] | None = None) -> list[SolarSystemObject]:
    """Parse a SkyBoT ``-mime=json -output=all`` answer (list of objects).

    RA/Dec are astrometric ICRS/J2000 positions at the epoch (sexagesimal); motions are
    arcsec/hour on the sky; the separation is recomputed from the parsed position.
    Positions are converted in one vectorised call; if that fails because of malformed
    rows, rows are parsed one by one and unparseable ones are dropped (and counted in
    ``notes``). A payload of which no row can be parsed is an upstream failure
    (:class:`UpstreamServiceError`), never a client error.
    """
    if isinstance(payload, dict) and ("flag" in payload or "message" in payload):
        message = str(payload.get("message", "")).strip().splitlines()
        raise UpstreamServiceError("skybot", f"service error (flag {payload.get('flag')}): "
                                   f"{message[0][:200] if message else 'no message'}", retryable=True)
    if not isinstance(payload, list):
        raise UpstreamServiceError("skybot", f"unexpected JSON payload of type {type(payload).__name__}")
    items = [item for item in payload
             if isinstance(item, dict) and item.get("RA (hms)") and item.get("DEC (dms)")]
    if not items:
        return []
    # One vectorised conversion: per-object SkyCoord construction costs ~1 ms each and
    # wide ecliptic cones hold thousands of bodies.
    try:
        ras, decs = _sexagesimal_coords(items)
    except (ValueError, TypeError):
        kept: list[dict[str, Any]] = []
        ra_list: list[float] = []
        dec_list: list[float] = []
        for item in items:
            try:
                one_ra, one_dec = _sexagesimal_coords([item])
            except (ValueError, TypeError):
                continue
            kept.append(item)
            ra_list.append(float(one_ra[0]))
            dec_list.append(float(one_dec[0]))
        dropped = len(items) - len(kept)
        if not kept:
            raise UpstreamServiceError("skybot", f"none of the {len(items)} returned positions could be parsed "
                                       f"(e.g. RA {items[0].get('RA (hms)')!r})")
        if notes is not None:
            notes.append(f"SkyBoT: {dropped} object(s) with unparseable positions were dropped.")
        items, ras, decs = kept, np.array(ra_list), np.array(dec_list)
    seps = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").separation(
        SkyCoord(ras * u.deg, decs * u.deg, frame="icrs")).arcsec
    objects: list[SolarSystemObject] = []
    for item, ora, odec, sep in zip(items, ras, decs, np.atleast_1d(seps)):
        num = item.get("Num")
        try:
            number = int(num) if num not in (None, "", "-") else None
        except (TypeError, ValueError):
            number = None
        cls = str(item.get("Class") or "")
        ssodnet = item.get("ssodnet") if isinstance(item.get("ssodnet"), dict) else {}
        objects.append(SolarSystemObject(
            name=str(item.get("Name") or "").strip(), number=number, type=_skybot_type(cls), object_class=cls,
            ra=float(ora), dec=float(odec), separation_arcsec=float(sep),
            v_mag=_num(item.get("VMag (mag)")), position_error_arcsec=_num(item.get("Err (arcsec)")),
            ra_rate_arcsec_per_hour=_num(item.get("dRA (arcsec/h)")), dec_rate_arcsec_per_hour=_num(item.get("dDEC (arcsec/h)")),
            geocentric_distance_au=_num(item.get("dg (ua)")), heliocentric_distance_au=_num(item.get("dh (ua)")),
            phase_angle_deg=_num(item.get("Phase (deg)")), solar_elongation_deg=_num(item.get("SunElong (deg)")),
            ssocard_url=ssodnet.get("ssocard"),
        ))
    objects.sort(key=lambda o: o.separation_arcsec)
    return objects


_OBSERVER_CODE = re.compile(r"^[A-Za-z0-9]{3}$")


async def solar_system_objects(
    ra: float | str,
    dec: float | str,
    *,
    epoch_mjd: float | None = None,
    radius_arcsec: float = DEFAULT_SKYBOT_RADIUS_ARCSEC,
    observer: str = "500",
    max_position_error_arcsec: float = 120.0,
    client: httpx.AsyncClient | None = None,
    timeout: float | None = None,
    deadline_s: float | None = None,
) -> SolarSystemResult:
    """Known asteroids, comets and planets in a cone at a UTC epoch (IMCCE SkyBoT).

    ``epoch_mjd`` is a UTC MJD (default: now); ``observer`` an IAU/MPC observatory
    code (500 = geocentre); objects with ephemeris uncertainty above
    ``max_position_error_arcsec`` are filtered by SkyBoT (0 disables the filter). The
    whole exchange (requests, retries, parsing) is bounded by ``deadline_s`` (default
    :data:`SKYBOT_DEADLINE_S`; exceeding it raises a retryable UpstreamServiceError).
    """
    target = validate_target(ra, dec)
    epoch = now_mjd() if epoch_mjd is None else float(epoch_mjd)
    jd = epoch + MJD_JD_OFFSET
    if not math.isfinite(jd) or not SKYBOT_MIN_JD <= jd <= SKYBOT_MAX_JD:
        raise InvalidCoordinateError(
            f"epoch_mjd must be within [{SKYBOT_MIN_JD - MJD_JD_OFFSET:.1f}, {SKYBOT_MAX_JD - MJD_JD_OFFSET:.1f}] (SkyBoT range).")
    if not 0 < float(radius_arcsec) <= SKYBOT_MAX_RADIUS_ARCSEC:
        raise InvalidCoordinateError(f"radius_arcsec must be in (0, {SKYBOT_MAX_RADIUS_ARCSEC:g}].")
    if not _OBSERVER_CODE.match(observer or ""):
        raise InvalidCoordinateError("observer must be a 3-character IAU observatory code (e.g. 500, I41).")
    if not (math.isfinite(max_position_error_arcsec) and 0 <= max_position_error_arcsec <= SKYBOT_MAX_POSITION_ERROR_ARCSEC):
        raise InvalidCoordinateError(
            f"max_position_error_arcsec must be within [0, {SKYBOT_MAX_POSITION_ERROR_ARCSEC:g}] (0 disables the filter).")
    params = {"-ep": f"{jd:.6f}", "-ra": f"{target.ra:.7f}", "-dec": f"{target.dec:.7f}", "-rs": f"{float(radius_arcsec):.3f}",
              "-mime": "json", "-output": "all", "-observer": observer, "-filter": f"{max_position_error_arcsec:g}",
              "-objFilter": "111", "-refsys": "EQJ2000", "-from": "AstroSearch"}
    started = time.perf_counter()
    notes: list[str] = []
    limit = SKYBOT_DEADLINE_S if deadline_s is None else float(deadline_s)
    async with client_scope(client) as active, _deadline("skybot", limit):
        for attempt in range(2):
            try:
                response = await _request(active, "GET", SKYBOT_CONESEARCH_URL, service="skybot", params=params,
                                          timeout=timeout or DEFAULT_TIMEOUTS["skybot"], ok_status=(200, 204))
            except UpstreamServiceError as exc:
                # SkyBoT answers HTTP 400 'Bad request: unknown code for the observer location'
                # (seen live for observer XXX) to a parameter it rejects: the caller's input, not an outage.
                if exc.status_code == 400:
                    raise InvalidCoordinateError(f"SkyBoT rejected the request: {exc.message}"
                                                 + (f" (observer code {observer!r})" if "observer" in exc.message else "")) from exc
                raise
            if response.status_code == 204 or not response.content.strip():
                objects: list[SolarSystemObject] = []  # SkyBoT answers 204 No Content for an empty field
                break
            try:
                payload = response.json()
            except ValueError as exc:
                raise UpstreamServiceError("skybot", f"non-JSON answer: {_error_detail(response)}",
                                           status_code=response.status_code) from exc
            notes.clear()
            try:
                objects = await asyncio.to_thread(parse_skybot_json, payload, target.ra, target.dec, notes=notes)
                break
            except UpstreamServiceError as exc:
                # A crashed SkyBoT worker answers HTTP 200 with {"flag": -1, "message": <backtrace>}
                # (seen live); it succeeds on resubmission, so retry once.
                if not exc.retryable or attempt == 1:
                    raise
                await asyncio.sleep(RETRY_PAUSE_SECONDS)
    provenance = {"service": SKYBOT_CONESEARCH_URL, "params": params, "http_status": response.status_code,
                  "elapsed_s": round(time.perf_counter() - started, 3), "retrieved_at": datetime.now(UTC).isoformat(),
                  "time_scale": "UTC", "reference": "Berthier et al. 2006, ASP Conf. Ser. 351, 367"}
    return SolarSystemResult({"ra": target.ra, "dec": target.dec}, epoch, jd, float(radius_arcsec), observer, objects,
                             provenance, notes)


def parse_horizons_result(text: str) -> list[EphemerisPoint]:
    """Parse a Horizons OBSERVER table (QUANTITIES 1,9; ANG_FORMAT DEG; CSV_FORMAT YES)."""
    name_match = re.search(r"Target body name:\s*(.+?)\s{2,}", text)
    target = name_match.group(1).strip() if name_match else ""
    if "$$SOE" not in text:
        raise UpstreamServiceError("horizons", f"no ephemeris in answer: {text.strip()[:300]}")
    head, body = text.split("$$SOE", 1)
    body = body.split("$$EOE", 1)[0]
    header_line = ""
    for line in reversed(head.strip().splitlines()):
        if "," in line and "Date" in line:
            header_line = line
            break
    columns = [c.strip() for c in header_line.split(",")]

    def col(prefix: str) -> int | None:
        for i, c in enumerate(columns):
            if c.startswith(prefix):
                return i
        return None

    i_date, i_ra, i_dec = col("Date"), col("R.A."), col("DEC")
    i_mag = col("APmag")
    if i_mag is None:
        i_mag = col("T-mag")
    if i_date is None or i_ra is None or i_dec is None:
        raise UpstreamServiceError("horizons", f"unexpected ephemeris columns: {columns}")
    points: list[EphemerisPoint] = []
    for line in body.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= max(i_ra, i_dec):
            continue
        date_txt = parts[i_date]
        try:
            dt = datetime.strptime(date_txt.split(".")[0], "%Y-%b-%d %H:%M:%S").replace(tzinfo=UTC)
            frac = float("0." + date_txt.split(".")[1]) if "." in date_txt else 0.0
            ra_val, dec_val = float(parts[i_ra]), float(parts[i_dec])
        except ValueError as exc:
            raise UpstreamServiceError("horizons", f"unparseable ephemeris line: {line.strip()[:120]}") from exc
        mjd = datetime_to_mjd(dt) + frac / 86400.0
        mag = _num(parts[i_mag]) if i_mag is not None and i_mag < len(parts) else None
        points.append(EphemerisPoint(target, mjd, ra_val, dec_val, mag))
    if not points:
        raise UpstreamServiceError("horizons", "empty ephemeris")
    return points


async def horizons_ephemeris(command: str, epoch_mjd: float | Sequence[float], *, observer: str = "500",
                             client: httpx.AsyncClient | None = None, timeout: float | None = None,
                             deadline_s: float | None = None) -> list[EphemerisPoint]:
    """Astrometric ICRF RA/Dec (Horizons quantity 1) and visual magnitude at UTC epochs.

    ``command`` is a Horizons target, e.g. ``'1;'`` (Ceres) or ``'4;'`` (Vesta) for
    numbered asteroids. ``observer`` is an observatory code (``500`` = geocentre). The
    exchange is bounded by ``deadline_s`` (default :data:`HORIZONS_DEADLINE_S`).
    """
    epochs = [float(epoch_mjd)] if isinstance(epoch_mjd, (int, float)) else [float(e) for e in epoch_mjd]
    if not epochs:
        raise ValueError("at least one epoch is required")
    if not _OBSERVER_CODE.match(observer or ""):
        raise InvalidCoordinateError("observer must be a 3-character IAU observatory code.")
    tlist = " ".join(f"{e + MJD_JD_OFFSET:.6f}" for e in epochs)
    params = {"format": "json", "COMMAND": f"'{command}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
              "EPHEM_TYPE": "'OBSERVER'", "CENTER": f"'{observer}@399'", "TLIST": f"'{tlist}'",
              "QUANTITIES": "'1,9'", "ANG_FORMAT": "'DEG'", "CSV_FORMAT": "'YES'"}
    limit = HORIZONS_DEADLINE_S if deadline_s is None else float(deadline_s)
    async with client_scope(client) as active, _deadline("horizons", limit):
        response = await _request(active, "GET", HORIZONS_API_URL, service="horizons", params=params,
                                  timeout=timeout or DEFAULT_TIMEOUTS["horizons"])
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamServiceError("horizons", "non-JSON answer", status_code=response.status_code) from exc
    if payload.get("error"):
        raise UpstreamServiceError("horizons", str(payload["error"])[:300], status_code=response.status_code)
    return parse_horizons_result(str(payload.get("result", "")))


# ---------------------------------------------------------------------------
# REST API (FastAPI router)
# ---------------------------------------------------------------------------


class LightCurvePointModel(BaseModel):
    mjd: float = Field(description="BMJD_TDB (BJD_TDB - 2400000.5)")
    value: float
    error: float | None
    flag: int = Field(description="0 = good; survey quality flag otherwise (ZTF catflags, TESS QUALITY, Gaia reject = 1)")
    n: int | None = Field(None, description="Exposures averaged into this point (binned series)")


class LightCurveSeriesModel(BaseModel):
    key: str
    survey: str
    band: str
    unit: Literal["mag", "flux"]
    time_system: str
    photometric_system: str
    source_ids: list[str]
    n_total: int
    n_rejected: int
    metadata: dict[str, Any]
    points: list[LightCurvePointModel]


class VariabilityModel(BaseModel):
    per_series: dict[str, dict[str, Any]]
    summary: dict[str, Any]


class PeriodModel(BaseModel):
    best_period_days: float
    power: float
    false_alarm_probability: float | None
    series: str
    baseline_days: float
    best_frequency_per_day: float
    n: int
    method: str
    reliable: bool
    peaks: list[dict[str, float]]
    warnings: list[str]
    min_frequency: float
    max_frequency: float
    n_frequencies: int
    oversampling: float
    members: list[str]
    period_error_days: float | None = None
    n_harmonics: int | None = None
    lomb_scargle_period_days: float | None = None
    harmonic_test: dict[str, Any] | None = None
    alternative_period_days: float | None = None
    lomb_scargle_false_alarm_probability: float | None = None
    noise_continuum: float | None = None
    noise_spectral_slope: float | None = None
    effective_n: float | None = None
    error_inflation: float | None = None
    robust_false_alarm_probability: float | None = None


class LightCurveResponse(BaseModel):
    target: dict[str, Any]
    series: list[LightCurveSeriesModel]
    variability: VariabilityModel
    period: PeriodModel | None
    period_search: dict[str, Any] | None
    failures: list[dict[str, Any]]
    notes: list[str]
    provenance: dict[str, Any]


class SolarSystemObjectModel(BaseModel):
    name: str
    number: int | None
    type: str
    object_class: str
    ra: float
    dec: float
    separation_arcsec: float
    v_mag: float | None
    position_error_arcsec: float | None
    ra_rate_arcsec_per_hour: float | None
    dec_rate_arcsec_per_hour: float | None
    geocentric_distance_au: float | None
    heliocentric_distance_au: float | None
    phase_angle_deg: float | None
    solar_elongation_deg: float | None
    ssocard_url: str | None


class SolarSystemResponse(BaseModel):
    target: dict[str, Any]
    epoch_mjd: float
    epoch_jd: float
    radius_arcsec: float
    observer: str
    n_objects: int
    objects: list[SolarSystemObjectModel]
    notes: list[str] = []
    provenance: dict[str, Any]


router = APIRouter(prefix="/api/v1", tags=["time-domain"])


def _state_client(request: Request) -> httpx.AsyncClient | None:
    return getattr(request.app.state, "client", None)


def _validated_json(model: type[BaseModel], build: Callable[[], dict[str, Any]]) -> bytes:
    """Build, validate against ``model`` and JSON-encode a response body (CPU-bound)."""
    return model.model_validate(build()).model_dump_json().encode("utf-8")


async def _json_response(model: type[BaseModel], build: Callable[[], dict[str, Any]]) -> Response:
    """JSON response validated like ``response_model`` but built in a worker thread.

    A light-curve payload can hold ~10^5 points (unbinned TESS sectors); converting it to a
    dict, validating it and encoding it takes ~0.1-1 s, which must not stall the event loop.
    """
    body = await asyncio.to_thread(_validated_json, model, build)
    return Response(content=body, media_type="application/json")


@router.get("/lightcurves", response_model=LightCurveResponse, summary="Multi-survey light curves, variability and period")
async def lightcurves_endpoint(
    request: Request,
    ra: float | None = Query(None, description="ICRS right ascension (deg)"),
    dec: float | None = Query(None, description="ICRS declination (deg)"),
    name: str | None = Query(None, min_length=1, max_length=200, description="Object name (CDS Sesame) instead of ra/dec"),
    epoch: float | None = Query(None, ge=1800, le=2200, description="Julian year of ra/dec (enables proper-motion "
                                "propagation); not with name alone: resolved coordinates carry the resolver's epoch"),
    pm_ra_masyr: float | None = Query(None, description="Proper motion in RA * cos(dec), mas/yr (with epoch; with "
                                      "name alone it overrides the resolver's, together with pm_dec_masyr)"),
    pm_dec_masyr: float | None = Query(None, description="Proper motion in Dec, mas/yr (see pm_ra_masyr)"),
    parallax_mas: float | None = Query(None, ge=0, lt=1000, description="Parallax, mas (widens the per-epoch ZTF "
                                       "identity tolerance of nearby moving stars; with name alone it overrides "
                                       "the resolver's)"),
    radius_arcsec: float = Query(3.0, gt=0, le=MAX_LIGHTCURVE_RADIUS_ARCSEC),
    surveys: str | None = Query(None, description=f"Comma-separated subset of {','.join(SURVEYS)} (default {','.join(DEFAULT_SURVEYS)})"),
    include_flagged: bool = Query(False, description="Also return quality-flagged epochs (excluded from statistics)"),
    period: bool = Query(True, description="Run the Lomb-Scargle period search"),
    min_period_days: float = Query(DEFAULT_MIN_PERIOD_DAYS, ge=MIN_ALLOWED_PERIOD_DAYS, le=MAX_ALLOWED_PERIOD_DAYS),
    max_period_days: float | None = Query(None, gt=0, le=MAX_ALLOWED_PERIOD_DAYS),
    oversampling: float = Query(DEFAULT_OVERSAMPLING, ge=1, le=50),
    ztf_collection: str | None = Query(None, pattern=r"^ztf_dr\d{1,3}$", description="ZTF data release, e.g. ztf_dr23"),
    tess_max_sectors: int = Query(2, ge=1, le=10),
    tess_bin_minutes: float = Query(10.0, ge=0, le=1440),
    neowise_binned: bool = Query(True, description="Average NEOWISE exposures per visit (false: single exposures)"),
) -> Response:
    try:
        survey_names = parse_surveys(surveys)
        _validate_period_range(min_period_days, max_period_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if name is not None and not name.strip():
        raise HTTPException(status_code=422, detail="name must not be blank")
    client = _state_client(request)
    resolver: dict[str, Any] | None = None
    if name and (ra is None or dec is None):
        if (ra is None) != (dec is None):
            raise HTTPException(status_code=422, detail="give both ra and dec, or neither (with name)")
        try:
            # Coordinates from the resolver carry the resolver's epoch (and, unless both
            # components are given, its proper motion): checked before resolving.
            _check_name_motion(epoch, pm_ra_masyr, pm_dec_masyr)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            resolved, resolver = await resolve_target(name, client)
        except UpstreamServiceError as exc:
            # As on every route: 503 when Sesame is unreachable (or answers 5xx), 502 when it
            # answers something that is not a Sesame document.
            unusable = isinstance(getattr(exc.__cause__, "__cause__", None), SesameUnusableAnswer)
            raise HTTPException(status_code=502 if unusable else 503,
                                detail=f"Name resolver unavailable for '{name}': {exc}",
                                headers=None if unusable else {"Retry-After": "30"}) from exc
        except ObjectResolutionError as exc:
            raise HTTPException(status_code=404, detail=f"Could not resolve '{name}': {exc}") from exc
        try:
            epoch, pm_ra_masyr, pm_dec_masyr = _resolved_motion(resolved, pm_ra_masyr, pm_dec_masyr)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ra, dec = resolved.ra, resolved.dec
        if parallax_mas is None:
            parallax_mas = resolved.parallax_mas
    if ra is None or dec is None:
        raise HTTPException(status_code=422, detail="Either name or both ra and dec are required")
    try:
        result = await get_lightcurves(
            ra, dec, radius_arcsec=radius_arcsec, surveys=survey_names, client=client, name=name,
            epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas,
            resolver=resolver, include_flagged=include_flagged, period=period, min_period_days=min_period_days,
            max_period_days=max_period_days, oversampling=oversampling, ztf_collection=ztf_collection,
            tess_max_sectors=tess_max_sectors, tess_bin_minutes=tess_bin_minutes, neowise_binned=neowise_binned,
        )
    except (InvalidCoordinateError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Light-curve upstream failure: {exc}") from exc
    return await _json_response(LightCurveResponse, result.as_dict)


@router.get("/solar-system", response_model=SolarSystemResponse, summary="Known solar-system bodies in a cone (IMCCE SkyBoT)")
async def solar_system_endpoint(
    request: Request,
    ra: float = Query(..., description="ICRS right ascension (deg)"),
    dec: float = Query(..., description="ICRS declination (deg)"),
    radius_arcsec: float = Query(DEFAULT_SKYBOT_RADIUS_ARCSEC, gt=0, le=SKYBOT_MAX_RADIUS_ARCSEC),
    epoch_mjd: float | None = Query(None, description="UTC MJD of the observation (default: now)"),
    observer: str = Query("500", pattern=r"^[A-Za-z0-9]{3}$", description="IAU observatory code (500 = geocentre)"),
    max_position_error_arcsec: float = Query(120.0, ge=0, le=SKYBOT_MAX_POSITION_ERROR_ARCSEC,
                                             description="SkyBoT ephemeris-uncertainty filter, arcsec (0 = off)"),
) -> Response:
    try:
        result = await solar_system_objects(ra, dec, epoch_mjd=epoch_mjd, radius_arcsec=radius_arcsec, observer=observer,
                                            max_position_error_arcsec=max_position_error_arcsec,
                                            client=_state_client(request))
    except (InvalidCoordinateError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=f"SkyBoT upstream failure: {exc}") from exc
    return await _json_response(SolarSystemResponse, lambda: _finite(result.as_dict()))


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------


# Why a period was doubled (harmonic_test["doubled_by"]), as printed by the CLI summary.
DOUBLING_REASONS: dict[str, str] = {
    "parity": "unequal minima (consecutive cycles differ)",
    "eclipse_shape": "one narrow dip per cycle, eclipsing binary with equal minima assumed",
    "parity+eclipse_shape": "unequal consecutive cycles, then one narrow dip per cycle (equal minima assumed)",
}


def format_lightcurve_summary(result: LightCurveResult) -> str:
    lines = [(f"Target: RA={result.target['ra']:.6f} DEC={result.target['dec']:.6f} "
              f"radius={result.target['radius_arcsec']:g}\" surveys={','.join(result.target['surveys'])}")]
    if not result.series:
        lines.append("No light-curve data returned.")
    for s in result.series:
        m = result.variability.get(s.key)
        good = sum(1 for p in s.points if p.flag == 0)
        verdict = "undetermined" if m is None or m.is_variable is None else ("VARIABLE" if m.is_variable else "not variable")
        if m is not None and m.is_variable is None and (s.survey, s.band) in NON_DECISIVE_SERIES:
            verdict = "not assessed"
        extra = ""
        if m and m.chi2_dof is not None:
            extra = (f" chi2/dof={m.chi2_dof:.2f} amp5-95={m.amplitude_5_95:.3f} eta={m.von_neumann_eta or float('nan'):.2f}"
                     f" mean={m.weighted_mean:.3f}")
        lines.append(f"  [{s.key:<11}] {good:>5} pts ({s.n_rejected} rejected) {s.unit:<4} -> {verdict}{extra}")
    best = result.period_search.best if result.period_search else None
    if best is not None:
        fap = f"{best.false_alarm_probability:.2e}" if best.false_alarm_probability is not None else "n/a"
        err = f" +- {best.period_error_days:.2g}" if best.period_error_days is not None else ""
        lines.append(f"Period: {best.best_period_days:.6f} d{err} (power {best.power:.3f}, FAP {fap}) from {best.series}")
        test = best.harmonic_test or {}
        if test.get("doubled"):
            reason = DOUBLING_REASONS.get(str(test.get("doubled_by")), str(test.get("doubled_by")))
            alt = (f"; alternative {best.alternative_period_days:.6f} d"
                   if best.alternative_period_days is not None else "")
            lines.append(f"  (doubled from the Lomb-Scargle peak {best.lomb_scargle_period_days:.6f} d: {reason}{alt})")
        elif (test.get("shape") or {}).get("transit_like") and best.alternative_period_days is not None:
            lines.append(f"  (transit-like single dip; if an eclipsing binary with equal minima: "
                         f"{best.alternative_period_days:.6f} d)")
        elif best.alternative_period_days is not None:
            lines.append(f"  (if an eclipsing/ellipsoidal binary: {best.alternative_period_days:.6f} d)")
    elif result.period_search is not None:
        lines.append("Period: no significant period")
    if best is not None:
        for warning in best.warnings:
            lines.append(f"  period note: {warning}")
    if result.period_search and result.period_search.multiband:
        # The multi-band sum has no false-alarm probability; it is printed with its caveats
        # and, when no reliable period was adopted, explicitly as not a detection.
        mb = result.period_search.multiband
        status = "" if best is not None else " (not a detection: no reliable period adopted)"
        caveats = f" [{'; '.join(mb.warnings)}]" if mb.warnings else ""
        lines.append(f"Multi-band peak: {mb.best_period_days:.6f} d over {', '.join(mb.members)}{status}{caveats}")
    for f in result.failures:
        lines.append(f"FAILED {f['survey']}: {f['error']}")
    for note in result.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def format_solar_system_summary(result: SolarSystemResult) -> str:
    lines = [(f"SkyBoT: RA={result.target['ra']:.6f} DEC={result.target['dec']:.6f} epoch MJD {result.epoch_mjd:.5f} UTC "
              f"radius={result.radius_arcsec:g}\" observer={result.observer}: {len(result.objects)} object(s)")]
    for o in result.objects:
        vmag = f"{o.v_mag:.1f}" if o.v_mag is not None else "  ?"
        ident = f"({o.number}) {o.name}" if o.number is not None else o.name
        lines.append(f"  {ident:<28} {o.object_class:<14} RA={o.ra:.6f} DEC={o.dec:+.6f} sep={o.separation_arcsec:8.1f}\" V={vmag}")
    return "\n".join(lines)


def cli_lightcurve(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch lightcurve``."""
    if not args.name and (args.ra is None or args.dec is None):
        print("Error: specify --name or both --ra and --dec.", file=sys.stderr)
        return 2

    by_name = bool(args.name) and (args.ra is None or args.dec is None)
    try:
        if by_name:
            # Resolved coordinates carry the resolver's epoch (checked before any request).
            _check_name_motion(args.epoch, args.pm_ra, args.pm_dec)
        elif args.epoch is None and (args.pm_ra is not None or args.pm_dec is not None):
            raise ValueError("--epoch (Julian year of --ra/--dec) is required with --pm-ra/--pm-dec")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    async def run() -> LightCurveResult:
        ra, dec = args.ra, args.dec
        motion: dict[str, Any] = {"epoch": args.epoch, "pm_ra_masyr": args.pm_ra, "pm_dec_masyr": args.pm_dec,
                                  "parallax_mas": args.parallax}
        async with client_scope(None) as client:
            if by_name:
                resolved, record = await resolve_target(args.name, client)
                ra, dec = resolved.ra, resolved.dec
                epoch, pm_ra, pm_dec = _resolved_motion(resolved, args.pm_ra, args.pm_dec)
                motion = {"epoch": epoch, "pm_ra_masyr": pm_ra, "pm_dec_masyr": pm_dec, "resolver": record,
                          "parallax_mas": args.parallax if args.parallax is not None else resolved.parallax_mas}
            return await get_lightcurves(ra, dec, radius_arcsec=args.radius, surveys=args.surveys, client=client,
                                         name=args.name, include_flagged=args.include_flagged, period=not args.no_period,
                                         min_period_days=args.min_period, max_period_days=args.max_period,
                                         tess_max_sectors=args.tess_sectors, tess_bin_minutes=args.tess_bin_minutes,
                                         neowise_binned=not args.neowise_exposures, **motion)

    try:
        result = asyncio.run(run())
    except (InvalidCoordinateError, ValueError, ObjectResolutionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except UpstreamServiceError as exc:
        print(f"Upstream failure: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(result.as_dict(), indent=2, allow_nan=False))
    else:
        print(format_lightcurve_summary(result))
    return 0


def cli_solar_system(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch solar-system``."""
    try:
        result = asyncio.run(solar_system_objects(args.ra, args.dec, epoch_mjd=args.epoch_mjd, radius_arcsec=args.radius,
                                                  observer=args.observer))
    except (InvalidCoordinateError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except UpstreamServiceError as exc:
        print(f"Upstream failure: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(_finite(result.as_dict()), indent=2, allow_nan=False))
    else:
        print(format_solar_system_summary(result))
    return 0


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``lightcurve`` and ``solar-system`` subcommands (handler=callable(args) -> exit code)."""
    lc = subparsers.add_parser("lightcurve", help="Multi-survey light curves, variability metrics and period search")
    lc.add_argument("--ra", type=float, help="ICRS right ascension (deg)")
    lc.add_argument("--dec", type=float, help="ICRS declination (deg)")
    lc.add_argument("--name", help="Object name resolved with CDS Sesame (its epoch, proper motion and parallax "
                                   "are used)")
    lc.add_argument("--epoch", type=float, help="Julian year of --ra/--dec (enables proper-motion propagation; "
                                               "not with --name alone)")
    lc.add_argument("--pm-ra", type=float, help="Proper motion in RA * cos(dec), mas/yr (with --epoch; with --name "
                                               "alone it overrides the resolver's, together with --pm-dec)")
    lc.add_argument("--pm-dec", type=float, help="Proper motion in Dec, mas/yr (see --pm-ra)")
    lc.add_argument("--parallax", type=float, help="Parallax, mas (widens the per-epoch ZTF identity tolerance of "
                                                  "nearby moving stars)")
    lc.add_argument("--radius", type=float, default=3.0, help="Cone radius in arcsec (default 3)")
    lc.add_argument("--surveys", default=",".join(DEFAULT_SURVEYS),
                    help=f"Comma-separated surveys from {','.join(SURVEYS)} (default {','.join(DEFAULT_SURVEYS)})")
    lc.add_argument("--include-flagged", action="store_true", help="Also list quality-flagged epochs")
    lc.add_argument("--no-period", action="store_true", help="Skip the Lomb-Scargle period search")
    lc.add_argument("--min-period", type=float, default=DEFAULT_MIN_PERIOD_DAYS, help="Shortest period searched (d)")
    lc.add_argument("--max-period", type=float, help="Longest period searched (d; default: baseline)")
    lc.add_argument("--tess-sectors", type=int, default=2,
                    help=f"Most recent TESS sectors to download (1-{MAX_TESS_SECTORS}; default 2)")
    lc.add_argument("--tess-bin-minutes", type=float, default=10.0,
                    help=f"TESS bin width in minutes (0, or below the 2-min cadence, = unbinned; sectors x points per "
                         f"sector must stay within {MAX_TESS_POINTS:,} points, i.e. at most {MAX_UNBINNED_TESS_SECTORS} "
                         "unbinned sectors; default 10)")
    lc.add_argument("--neowise-exposures", action="store_true", help="Keep NEOWISE single exposures (no visit binning)")
    lc.add_argument("--format", choices=["summary", "json"], default="summary")
    lc.set_defaults(handler=cli_lightcurve)

    ss = subparsers.add_parser("solar-system", help="Known asteroids/comets near a position at an epoch (IMCCE SkyBoT)")
    ss.add_argument("--ra", type=float, required=True, help="ICRS right ascension (deg)")
    ss.add_argument("--dec", type=float, required=True, help="ICRS declination (deg)")
    ss.add_argument("--epoch-mjd", type=float, help="UTC MJD of the observation (default: now)")
    ss.add_argument("--radius", type=float, default=DEFAULT_SKYBOT_RADIUS_ARCSEC, help="Cone radius in arcsec (default 600)")
    ss.add_argument("--observer", default="500", help="IAU observatory code (default 500 = geocentre)")
    ss.add_argument("--format", choices=["summary", "json"], default="summary")
    ss.set_defaults(handler=cli_solar_system)
