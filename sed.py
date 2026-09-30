"""Multi-wavelength object passport: spectral energy distribution (SED), classification, and redshift.

Pipeline
--------
1. Crossmatch the position (or a resolved name) with :class:`crossmatch.CrossmatchService`, or take an
   existing ``UnifiedRecord`` dict.
2. Pool the rows of ALL crossmatch groups and take, per catalog, the row within that catalog's positional tolerance
   (astrometry (+) epoch-propagation growth, or resolution), preferring the target's own group (``contains_target``
   of the Bayesian association, else the group with most survey catalogs in tolerance), then the nearest. The
   association may split one object over several groups (high proper-motion stars, a Gaia nucleus in its own
   group, a group of name-resolver rows at 0 arcsec); rows used from other groups, and in-tolerance rows passed
   over, are reported in ``notes``. SIMBAD rows: the Sesame-resolved identifier first, planets ('Pl') never
   replace their host; NED rows: object-level entries before absorbers/sub-components. A 2MASS/AllWISE/Gaia DR3 row
   whose designation ('2MASS J...', 'WISEA J...', 'Gaia DR3 ...') is a name of the object resolved by Sesame is the
   object even beyond the positional tolerance (proper-motion offsets between mislabelled catalogue epochs).
3. Extract every photometric measurement present in those members (Gaia DR3 G/BP/RP, 2MASS JHKs,
   AllWISE W1-W4, Pan-STARRS1 grizy, SDSS ugriz, GALEX FUV/NUV, SIMBAD literature V, radio flux densities
   from FIRST/NVSS/VLASS/LoTSS, X-ray fluxes from ROSAT/Chandra/XMM-Newton) and convert them to flux
   densities in Jy with propagated 1-sigma errors, nu*F_nu in erg s^-1 cm^-2, and upper-limit flags.
4. Collect redshifts (SDSS SpecObj/Photoz via SkyServer SQL, NED, SIMBAD) and pick the best one.
5. Classify the object (star / qso / galaxy / agn) from transparent, individually-weighted evidence.

Photometric calibration
-----------------------
* Filter effective wavelengths and Vega zero points come from the SVO Filter Profile Service
  (Rodrigo & Solano 2020, "The SVO Filter Profile Service", XXX Reunion Bienal de la SEA, 182;
  Rodrigo, Solano & Bayo 2012, IVOA Working Draft). The service lives at
  ``https://svo2.cab.inta-csic.es/theory/fps/fps.php`` (the host ``svo2.cab.inaf.es`` does not resolve).
  Answers are validated (finite ZeroPoint > 0 in Jy with PhotCalID '<filter>/Vega', WavelengthEff > 0) before
  they are cached on disk; malformed answers or cache entries are never used. A request waits at most
  ``FilterCatalog.deadline_seconds`` for SVO; :data:`EMBEDDED_FILTERS`, copied verbatim from live SVO responses
  retrieved 2026-09-28, serves every filter SVO has not answered for.
* AB magnitudes (Pan-STARRS1: Tonry et al. 2012, ApJ 750, 99; SDSS: Fukugita et al. 1996, AJ 111, 1748;
  GALEX: Morrissey et al. 2007, ApJS 173, 682) use the defining AB zero point F_nu = 3631 Jy
  (Oke & Gunn 1983, ApJ 266, 713): m_AB = -2.5 log10(F_nu / 3631 Jy). SVO's per-filter "AB" ZeroPoint
  (e.g. 3767.2 Jy for SDSS u) is tied to SVO's own effective-wavelength convention and is *not* used.
* SDSS magnitudes are asinh magnitudes (Lupton, Gunn & Szalay 1999, AJ 118, 1406) with softening
  parameters b = (1.4, 0.9, 1.2, 1.8, 7.4) x 1e-10 for ugriz and AB offsets u_AB = u_SDSS - 0.04 and
  z_AB = z_SDSS + 0.02 (SDSS DR17 "Magnitudes" and "Flux calibration" algorithm pages).
* Vega magnitudes (Gaia DR3: Riello et al. 2021, A&A 649, A3; 2MASS: Cohen, Wheaton & Megeath 2003,
  AJ 126, 1090; WISE: Jarrett et al. 2011, ApJ 735, 112) use the SVO Vega zero points.
* Radio flux densities are taken as published (FIRST: Becker et al. 1995 / Helfand et al. 2015, integrated
  1.4 GHz flux in mJy; NVSS: Condon et al. 1998, 1.4 GHz, mJy; VLASS epoch 1 (Bruzewski et al. 2021),
  3 GHz, Jy; LoTSS (Shimwell et al.), 144 MHz, mJy).
* X-ray band fluxes F (erg s^-1 cm^-2 in [E1, E2]) are converted to a flux density at the logarithmic
  band centre E0 = sqrt(E1 E2) ASSUMING a power law with photon index Gamma = 2 (N(E) ~ E^-Gamma), the
  canonical AGN value (e.g. Nandra & Pounds 1994, MNRAS 268, 405): F_E(E) = K E^(1-Gamma) with
  F = integral_E1^E2 F_E dE, so for Gamma = 2, F_E(E0) = F / (E0 ln(E2/E1)) and nu F_nu = F / ln(E2/E1).
  ROSAT PSPC count rates (0.1-2.4 keV) use the UNABSORBED energy conversion factor from HEASARC
  WebPIMMS v4.15a (Gamma = 2, N_H = 3e20 cm^-2): 1 ct/s = 1.877e-11 erg s^-1 cm^-2 unabsorbed
  (1.135e-11 absorbed), so the ROSAT point is the level of the intrinsic Gamma = 2 power law, i.e.
  corrected for an assumed Galactic N_H = 3e20 cm^-2. The true N_H varies with sight line
  (1e20 to >1e22 cm^-2); at 0.49 keV this is the dominant systematic (a factor ~1.7 between the
  absorbed and unabsorbed normalisations for N_H = 3e20). Chandra and XMM-Newton band fluxes are
  the catalogues' observed fluxes (absorption at 3e20 cm^-2 is small above 0.5 keV).

Photometric quality
-------------------
Points carry ``quality_warning`` (and the reason in ``warnings``) when the catalogue flags the measurement: 2MASS/WISE
ph_qual E/F/X, AllWISE profile-fit magnitudes brighter than 2.0, 1.5, -3.0, -4.0 mag in W1-W4 (the limit of reliable
profile fitting of saturated sources, WISE All-Sky Explanatory Supplement VI.3.d), W1 < 8 / W2 < 7 at the ecliptic
longitudes observed in the NEOWISE post-cryo phase (AllWISE Explanatory Supplement II.2.c.i), 'U' limits brighter than
the nominal saturation limits 8, 7, 3.8, 0.4 mag (II.2.c.iii; a failed extraction, not a limit), WISE points more than
5x below the Rayleigh-Jeans extrapolation of a shorter 2MASS/WISE band, SDSS SATURATED per-band flags (or psfMag < 14 when the flags could not be fetched;
Riello et al. 2021 App. C: SDSS sources brighter than 14 mag are saturated), Pan-STARRS1 mean PSF
magnitudes brighter than the single-exposure saturation limits g, r, i 13.5, z 13.0, y 12.0 (Magnier
et al. 2013, ApJS 205, 20) or qualityFlag without QF_OBJ_GOOD, Gaia BP/RP when the corrected flux
excess |C*| > 3 sigma_C*(G) (Riello et al. 2021, Eqs. 6 and 18, Table 2), Chandra pile-up/saturation,
LoTSS points implying a rising 144 MHz -> 1.4 GHz spectrum, and SIMBAD literature V magnitudes
inconsistent with Gaia (G - V from BP - RP, Riello et al. 2021 Table C.2). Flagged points stay in the
SED but are never used by the classification rules.

Classification evidence
-----------------------
Gaia DR3 astrometry (parallax_over_error > 5 or proper motion > 5 sigma, only when RUWE < 1.4:
Lindegren et al. 2021, A&A 649, A2; the proper-motion errors include the 33 uas/yr RMS small-scale
systematics quoted in Gaia Collaboration, Vallenari et al. 2023, Sect. 3.3, plus the 80 uas/yr
bright-star frame bias for G < 13, Cantat-Gaudin & Brandt 2021; a significant motion counts fully only
above 1 mas/yr, weakly at 0.2-1 mas/yr where structured/variable quasars show spurious motions, and not at
all below 5 mas/yr when astrometric_excess_noise_sig > 2), Gaia DR3 DSC class probabilities
(Delchambre et al. 2023, A&A 674, A31) and qso/galaxy candidate membership, WISE W1-W2 >= 0.8 Vega
(Stern et al. 2012, ApJ 753, 30; valid for W2 < 15.05) and the WISE colour-colour loci (Wright et al.
2010, AJ 140, 1868) using only ph_qual A/B bands, radio loudness R_i = log10(F_1.4GHz / F_i) > 1
(Ivezic et al. 2002, AJ 124, 2364; defined for point-like optical counterparts -- for extended ones a
total optical magnitude is used and the rule is down-weighted), X-ray-to-optical ratio
log(fX/fV) = log fX + V/2.5 + 5.37 > -1 (Maccacaro et al. 1988, ApJ 326, 680; Stocke et al. 1991,
ApJS 76, 813), the QSO luminosity criterion M_i < -22 (Schneider et al. 2010, AJ 139, 2360; point-like
counterparts only), SDSS morphology/spectral class, AllWISE association with a 2MASS extended source
(ext_flg 2-5; ext_flg 1 is only a poor PSF fit) and SIMBAD / NED object types (NED sub-component rows such
as absorbers are never taken as the object). SIMBAD types are mapped through the otypedef hierarchy (``G > AGN >
QSO`` -> qso, ``G > AGN`` -> agn, ``G`` -> galaxy, ``*`` -> star), never by the look of the code: supernovae/novae
are transients (not class-specific), 'Sy?' is a symbiotic-star candidate. A NED stellar type on a row whose own
redshift exceeds 1017 km/s is ignored. SDSS morphology of saturated objects (type 3 of saturated stars) is not used,
and the WISE W1-W2 AGN colour is not applied to objects with a significant parallax/proper motion (brown dwarfs).
For extended counterparts the X-ray/optical ratio uses the total optical light, and a quasar/blazar catalogue type
counts as a quasar only when the nucleus can be quasar-luminous at the adopted redshift (otherwise as an AGN in its
host: Cygnus A, NGC 1275); the ratio is not AGN evidence for a star with a Gaia or SIMBAD parallax/proper motion, and an
X-ray row that a brighter neighbouring Gaia source is closer to (after proper-motion propagation over the X-ray
observations) is flagged. Every piece of
evidence is reported with its weight; scores are a softmax over summed weights, so the output is a
transparent heuristic, not a calibrated probability; ties are reported and resolved explicitly: an extended
counterpart prefers galaxy/agn (unless the adopted redshift is a quasar's and a galaxy at it would exceed the quasar
luminosity M < -22), otherwise the class of the entry whose redshift was adopted, otherwise 'unknown'. When the result is 'star', redshift-based rules are dropped and a redshift
|cz| > 1017 km/s (above the fastest known field/hypervelocity star, a heuristic not applied near Sgr A*) is
reported as conflicting.

Redshift
--------
Kinds are never guessed: SDSS SpecObj (zWarning = 0 is reliable, a null zWarning is unknown), NED zflag
(S = spec, P = photo, other codes = unknown technique but NED-vetted), SIMBAD rvz_nature ('s*' = spec,
'p' = photo; a radial velocity with a null nature = spec; quality A-D reliable, E unreliable whatever the nature)
fetched from SIMBAD TAP. See :data:`_REDSHIFT_PRIORITY`: a photometric redshift never outranks a catalogue-vetted
value. Every non-photometric value not flagged by its catalogue -- vetted, or of unknown quality because its lookup
failed -- is cross-checked (|dz|/(1+z) > 0.01 is a discordance): a vetted value corroborated by a second catalogue or
an SDSS zWarning = 0 spectrum, or else the only vetted value consistent with the redshift-independent classification
(a small or negative cz is never ruled out for a galaxy), is adopted; otherwise the redshift is reported as a conflict
(reliable = False), with every discordant value listed in ``redshift.discordant`` (``vetted`` False: its quality
could not be checked). Gaia/SIMBAD TAP lookups retry transient failures (3 attempts, as providers.py); rate limits
(HTTP 429, Retry-After) back the archive off.

Identification
--------------
Names are resolved with Sesame '-oxpI' (all identifiers; the configured $SESAME_ENDPOINT with the 'I' option added): a
2MASS/AllWISE/Gaia DR3 row whose designation is one of the object's names is the object even beyond the positional
tolerance (fast-moving stars at mismatched epochs). A record resolved without identifiers (the app's own '-oxp'
crossmatch) gets them from Sesame before the members are chosen.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import email.utils
import functools
import io
import json
import logging
import math
import os
import re
import ssl
import tempfile
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models import (
    AstroSearchError,
    InvalidCoordinateError,
    ObjectResolutionError,
    UnifiedRecord,
    haversine_arcsec,
    parse_json_table,
    propagate_radec,
    resolution_failure_status,
    resolved_target,
    tangent_offset_arcsec,
    validate_target,
)

logger = logging.getLogger("astrosearch.sed")


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """One default SSL context per process (as vizier/imaging): building one, which httpx otherwise does for every new
    client from certifi's bundle, takes ~0.5-1 s on Windows."""
    return ssl.create_default_context()


async def _owned_client(timeout: float) -> httpx.AsyncClient:
    """A private client for a call without a caller-supplied one.

    The SSL context is built off the event loop (once per process): building it synchronously blocked the loop for
    ~1 s, so a caller's cancellation (``asyncio.wait_for``) or deadline could not take effect until it was done."""
    if _ssl_context.cache_info().currsize:
        context = _ssl_context()
    else:
        context = await asyncio.to_thread(_ssl_context)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=context)


async def _close_owned(client: httpx.AsyncClient) -> None:
    """Close a private client even when the caller is cancelled again while it closes (no leaked connections)."""
    await asyncio.shield(client.aclose())


# ---------------------------------------------------------------------------
# Errors & Constants
# ---------------------------------------------------------------------------


class SEDError(AstroSearchError):
    """Base class of the errors raised while building an SED."""


class SEDInputError(SEDError, ValueError):
    """The caller's input is invalid (bad radius, missing position, malformed record): HTTP 422."""


class SEDFilterError(SEDError):
    """Filter metadata is unusable (malformed SVO answer or cache entry): a server-side fault, never a 422."""


class SEDUpstreamError(SEDError):
    """Raised when the archives needed for an SED could not be reached.

    ``failures`` holds the crossmatch failure dicts (catalog, error_type, ...) when the error comes from a
    crossmatch in which no catalog answered, so callers can tell outages (network, timeout, rate limit, 5xx)
    from parse/query regressions. ``status_code`` and ``retry_after`` (seconds, from a Retry-After header) are set
    when an archive answered with an HTTP error, so a rate limit (429) can be backed off (:func:`backoff_seconds_for`).
    """

    def __init__(self, message: str, failures: Sequence[Mapping[str, Any]] = (), *, status_code: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.failures: list[dict[str, Any]] = [dict(f) for f in failures]
        self.status_code = status_code
        self.retry_after = retry_after


SVO_FPS_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
SDSS_SQL_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS = 60.0  # ceiling of one supplementary HTTP request
SUPPLEMENTARY_BACKOFF_SECONDS = 120.0  # back-off of a host after a timeout (see HostBackoff)
SUPPLEMENTARY_REFUSED_BACKOFF_SECONDS = 60.0  # back-off after refused connections / other slow failures
SUPPLEMENTARY_SLOW_FAILURE_SECONDS = 5.0  # a failure that took longer than this is backed off whatever its type
SUPPLEMENTARY_MAX_RETRY_AFTER_SECONDS = 600.0  # cap of a back-off taken from an archive's Retry-After header

DEFAULT_RADIUS_ARCSEC = 10.0
MAX_RADIUS_ARCSEC = 60.0

AB_ZERO_POINT_JY = 3631.0  # Oke & Gunn 1983: m_AB = -2.5 log10(F_nu / 3631 Jy)
JY_CGS = 1.0e-23  # 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1
C_UM_HZ = float(const.c.to(u.um / u.s).value)  # speed of light in micron * Hz
H_KEV_S = float(const.h.to(u.keV * u.s).value)  # Planck constant in keV s
POGSON = 2.5 / math.log(10.0)  # 1.0857...: sigma_m = POGSON * sigma_F / F

# SDSS asinh softening parameters b (SDSS DR17 algorithms/magnitudes; Lupton et al. 1999).
SDSS_ASINH_B: dict[str, float] = {"u": 1.4e-10, "g": 0.9e-10, "r": 1.2e-10, "i": 1.8e-10, "z": 7.4e-10}
# m_AB - m_SDSS (SDSS DR17 algorithms/fluxcal: u_AB = u_SDSS - 0.04, z_AB = z_SDSS + 0.02; g, r, i ~ AB).
SDSS_AB_OFFSET: dict[str, float] = {"u": -0.04, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.02}

# Gaia (E)DR3 Vega zero-point uncertainties in mag (Riello et al. 2021, Gaia EDR3 documentation 5.4):
# G_ZP = 25.6873668671 +/- 0.0027553202, BP 25.3385422158 +/- 0.0027901700, RP 24.7478955012 +/- 0.0037793818.
GAIA_ZP_SIGMA_MAG: dict[str, float] = {"G": 0.0027553202, "BP": 0.0027901700, "RP": 0.0037793818}

# ROSAT PSPC energy conversion factors, erg s^-1 cm^-2 per ct/s in 0.1-2.4 keV: HEASARC WebPIMMS v4.15a,
# power law Gamma = 2.0, N_H = 3.0e20 cm^-2 (queried 2026-09-28): "PIMMS predicts a flux (0.100-2.400keV)
# of 1.135E-11" and "an unabsorbed flux (0.100-2.400keV) of 1.877E-11" for 1 ct/s.
ROSAT_PSPC_ECF_ABSORBED = 1.135e-11
ROSAT_PSPC_ECF = 1.877e-11  # unabsorbed: normalises the intrinsic Gamma = 2 power law used for F_nu(E0)
ROSAT_PSPC_ECF_ASSUMPTION = ("WebPIMMS v4.15a unabsorbed ECF 1.877e-11 erg/cm2/s per ct/s (0.1-2.4 keV, Gamma=2, "
                             "Galactic N_H=3e20 cm^-2 assumed for every sight line; absorbed ECF 1.135e-11)")
XRAY_PHOTON_INDEX = 2.0

RUWE_GOOD = 1.4  # Lindegren et al. 2021: RUWE > 1.4 indicates a poor single-star astrometric solution
# Bright-source proper-motion bias "reaching up to 80 uas/yr for sources in the range G = 11 - 13"
# (Cantat-Gaudin & Brandt 2021, A&A 649, A124): added in quadrature to each component when G < 13.
GAIA_BRIGHT_PM_BIAS_MASYR = 0.080
GAIA_BRIGHT_G_LIMIT = 13.0
# "The RMS angular (i.e. source to source) covariances of the parallaxes and proper motions on small scales
# are ~26 uas and ~33 uas/yr" (Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1, Sect. 3.3, citing
# Lindegren et al. 2021): added in quadrature to each proper-motion component for every source.
GAIA_PM_SYSTEMATIC_MASYR = 0.033
# A tangential velocity of 100 km/s at 100 kpc (outer halo) gives mu = v / (4.74 d) = 0.21 mas/yr; smaller
# significant proper motions are very weak evidence (residual systematics, or an implausibly distant star).
GAIA_PM_PHYSICAL_MASYR = 0.2
# Famous quasars reach formally significant Gaia DR3 proper motions of 0.2-0.6 mas/yr (3C 279: 0.24 mas/yr at
# 5.5 sigma; 4C 21.35: 0.24 at 5.8; PKS 0118-272: 0.63 at 11 sigma) from source structure and variability-induced
# motion. Only mu >= 1 mas/yr (v_t = 4.74 mu d = 237 km/s at 50 kpc, i.e. any halo star within 50 kpc) is strong
# stellar evidence; 0.2-1 mas/yr is weak evidence.
GAIA_PM_STRONG_MASYR = 1.0
# "A value D>2 indicates that the given epsilon_i is probably significant" (Gaia DR3 data model, gaia_source,
# astrometric_excess_noise_sig; Lindegren et al. 2012, A&A 538, A78): the 5-parameter model does not describe the
# source (structure, variability-induced motion, binarity). Its proper motion is then used only when it is far
# above the spurious motions of structured/variable quasars (>= 5 mas/yr).
GAIA_AEN_SIG_LIMIT = 2.0
GAIA_PM_NOISY_MIN_MASYR = 5.0
# Fastest known star unbound from the Galaxy's centre region: S5-HVS1, v_r = 1017 km/s (Koposov et al. 2020,
# MNRAS 491, 2465). A heuristic ceiling for field stars only: stars orbiting Sgr A* reach radial velocities of
# several thousand km/s (S2: GRAVITY Collaboration 2018, A&A 615, L15), so the rule is not applied within
# GALACTIC_CENTRE_RADIUS_ARCSEC of Sgr A* (ICRS 266.41681662, -29.00782497; Reid & Brunthaler 2004, ApJ 616, 872).
S5_HVS1_KMS = 1017.0
SGR_A_STAR_RADEC = (266.41681662, -29.00782497)
GALACTIC_CENTRE_RADIUS_ARCSEC = 60.0
S5_HVS1_Z = S5_HVS1_KMS / 299792.458
C_KMS = 299792.458

# Gaia EDR3/DR3 corrected BP/RP flux excess C* = C - f(BP-RP) (Riello et al. 2021, Eq. 6, Table 2) and its
# 1-sigma scatter for isolated stars sigma_C*(G) = c0 + c1 G^m (Eq. 18).
GAIA_EXCESS_POLY: tuple[tuple[float, tuple[float, ...]], ...] = (
    (0.5, (1.154360, 0.033772, 0.032277)),
    (4.0, (1.162004, 0.011464, 0.049255, -0.005879)),
    (math.inf, (1.057572, 0.140537)),
)
GAIA_EXCESS_SIGMA = (0.0059898, 8.817481e-12, 7.618399)
GAIA_EXCESS_NSIGMA = 3.0
# |C*| < 0.1 (BP+RP at most 10% above G) moves BP-RP by <~0.1 mag, i.e. G - V by <~0.05 mag: the colour is still
# good enough to predict V even when C* exceeds the 3-sigma photometric-quality threshold.
GAIA_EXCESS_COLOUR_LIMIT = 0.1
# G - V = f(BP - RP), valid -0.5 < BP-RP < 5.0 (Riello et al. 2021, Table C.2).
GAIA_G_MINUS_V = (-0.02704, 0.01424, -0.2156, 0.01426)
SIMBAD_V_TOLERANCE_MAG = 0.5

# Pan-STARRS1 single-exposure saturation in the 3pi survey (Magnier et al. 2013, ApJS 205, 20):
# g, r, i ~ 13.5, z ~ 13.0, y ~ 12.0 mag. Mean PSF magnitudes brighter than this are unreliable.
PS1_SATURATION_MAG: dict[str, float] = {"g": 13.5, "r": 13.5, "i": 13.5, "z": 13.0, "y": 12.0}
PS1_QF_OBJ_EXT, PS1_QF_OBJ_GOOD, PS1_QF_OBJ_SUSPECT_STACK, PS1_QF_OBJ_BAD_STACK = 1, 4, 64, 128  # PS1 ObjectQualityFlags
# PS1 star/galaxy separation iPSF - iKron > 0.05 (MAST "How to separate stars and galaxies"), reliable only
# between saturation (i ~ 14) and i ~ 21.
PS1_EXTENDED_PSF_KRON = 0.05
PS1_EXTENDED_I_RANGE = (14.0, 21.0)
# WISE saturation (AllWISE Explanatory Supplement, retrieved 2026-09-28):
# * "WISE profile-fitting photometry reliably extracts measurements of saturated sources using the non-saturated wings
#   of their profiles up to brightnesses of approximately 2.0, 1.5, -3.0 and -4.0 mag in W1, W2, W3 and W4" (All-Sky
#   Explanatory Supplement VI.3.d). w?mpro brighter than these limits is unreliable.
# * The "nominal saturation limits of 8, 7, 3.8 and 0.4 mag" (AllWISE ES II.2.c.iii) apply to the APERTURE photometry
#   (w?mag), which is not used here. For profile-fit photometry the only AllWISE caveat is W1 < 8 / W2 < 7 for sources
#   observed in the NEOWISE post-cryo phase, at ecliptic longitudes 89.4 < lambda < 221.8 deg or 280.6 < lambda < 48.1
#   deg: their fluxes "have much larger uncertainties and exhibit more scatter" (AllWISE ES II.2.c.i).
# * A ph_qual 'U' (S/N < 2) "limit" brighter than the nominal saturation limits is not a non-detection but a failed
#   extraction of a saturated source (e.g. Aldebaran W1 'U' 5.6 mag = 1.8 Jy for a ~5000 Jy source).
WISE_PROFILE_FIT_LIMIT_MAG: dict[str, float] = {"W1": 2.0, "W2": 1.5, "W3": -3.0, "W4": -4.0}
WISE_SATURATION_MAG: dict[str, float] = {"W1": 8.0, "W2": 7.0, "W3": 3.8, "W4": 0.4}
WISE_POSTCRYO_BANDS: dict[str, float] = {"W1": 8.0, "W2": 7.0}
WISE_POSTCRYO_ECLIPTIC_LON_DEG: tuple[tuple[float, float], ...] = ((89.4, 221.8), (280.6, 360.0), (0.0, 48.1))
# A WISE flux (or limit) more than this factor below the Rayleigh-Jeans extrapolation F_nu ~ lambda^-2 (the steepest
# decline of thermal emission towards long wavelengths) of an unflagged shorter-wavelength 2MASS/WISE point is
# physically inconsistent (saturation, confusion or a spurious limit); the factor allows for variability.
WISE_RJ_MARGIN = 5.0
SDSS_SATURATION_MAG = 14.0  # Riello et al. 2021 App. C: "SDSS12 sources brighter than 14 mag are saturated"
OPTICAL_GAIA_TOLERANCE_MAG = 0.5  # PS1/SDSS i vs Gaia RP flux density before i is used by the classification

Regime = Literal["radio", "infrared", "optical", "ultraviolet", "xray"]
CLASSES: tuple[str, ...] = ("star", "qso", "galaxy", "agn")

# ---------------------------------------------------------------------------
# Filter Metadata (SVO Filter Profile Service with disk cache + embedded fallback)
# ---------------------------------------------------------------------------

# Values copied verbatim from live SVO FPS responses (fps.php?ID=<filter>, default PhotCalID = Vega),
# retrieved 2026-09-28. Wavelengths in Angstrom, Vega zero point in Jy.
EMBEDDED_FILTERS: dict[str, dict[str, Any]] = {
    "GAIA/GAIA3.G": {"PhotCalID": "GAIA/GAIA3.G/Vega", "ZeroPoint": 3228.7464752872, "WavelengthEff": 5822.3887136979,
                     "WavelengthMean": 6719.5494436669, "WavelengthPivot": 6217.590403249,
                     "WavelengthMin": 3309.1300431716, "WavelengthMax": 10381.713610409},
    "GAIA/GAIA3.Gbp": {"PhotCalID": "GAIA/GAIA3.Gbp/Vega", "ZeroPoint": 3552.0128903434, "WavelengthEff": 5035.7502754443,
                       "WavelengthMean": 5319.8652108699, "WavelengthPivot": 5109.7120850159,
                       "WavelengthMin": 3301.8296034651, "WavelengthMax": 6739.3372529264},
    "GAIA/GAIA3.Grp": {"PhotCalID": "GAIA/GAIA3.Grp/Vega", "ZeroPoint": 2554.9484277488, "WavelengthEff": 7619.9599926215,
                       "WavelengthMean": 7939.1014333152, "WavelengthPivot": 7769.0226260418,
                       "WavelengthMin": 6200.4566154641, "WavelengthMax": 10465.566531006},
    "2MASS/2MASS.J": {"PhotCalID": "2MASS/2MASS.J/Vega", "ZeroPoint": 1594.0, "WavelengthEff": 12350.0,
                      "WavelengthMean": 12350.0, "WavelengthPivot": 12393.088853072,
                      "WavelengthMin": 10690.531138639, "WavelengthMax": 14208.518720603},
    "2MASS/2MASS.H": {"PhotCalID": "2MASS/2MASS.H/Vega", "ZeroPoint": 1024.0, "WavelengthEff": 16620.0,
                      "WavelengthMean": 16620.0, "WavelengthPivot": 16494.946887615,
                      "WavelengthMin": 14465.169802302, "WavelengthMax": 18277.226543491},
    "2MASS/2MASS.Ks": {"PhotCalID": "2MASS/2MASS.Ks/Vega", "ZeroPoint": 666.8, "WavelengthEff": 21590.0,
                       "WavelengthMean": 21590.0, "WavelengthPivot": 21638.60586223,
                       "WavelengthMin": 19402.190435053, "WavelengthMax": 23810.786086185},
    "WISE/WISE.W1": {"PhotCalID": "WISE/WISE.W1/Vega", "ZeroPoint": 309.54, "WavelengthEff": 33526.0,
                     "WavelengthMean": 33526.0, "WavelengthPivot": 33682.213249883,
                     "WavelengthMin": 27540.970745309, "WavelengthMax": 38723.877229677},
    "WISE/WISE.W2": {"PhotCalID": "WISE/WISE.W2/Vega", "ZeroPoint": 171.787, "WavelengthEff": 46028.0,
                     "WavelengthMean": 46028.0, "WavelengthPivot": 46179.056792133,
                     "WavelengthMin": 39633.264098975, "WavelengthMax": 53413.603102133},
    "WISE/WISE.W3": {"PhotCalID": "WISE/WISE.W3/Vega", "ZeroPoint": 31.674, "WavelengthEff": 115608.0,
                     "WavelengthMean": 115608.0, "WavelengthPivot": 120718.11765248,
                     "WavelengthMin": 74430.442564269, "WavelengthMax": 172613.42742331},
    "WISE/WISE.W4": {"PhotCalID": "WISE/WISE.W4/Vega", "ZeroPoint": 8.363, "WavelengthEff": 220883.0,
                     "WavelengthMean": 220883.0, "WavelengthPivot": 221944.03871835,
                     "WavelengthMin": 195200.83360398, "WavelengthMax": 279107.24275724},
    "PAN-STARRS/PS1.g": {"PhotCalID": "PAN-STARRS/PS1.g/Vega", "ZeroPoint": 3964.0298373521, "WavelengthEff": 4810.1596079294,
                         "WavelengthMean": 4900.1230736441, "WavelengthPivot": 4849.1149388631,
                         "WavelengthMin": 3949.5019343987, "WavelengthMax": 5593.8722819594},
    "PAN-STARRS/PS1.r": {"PhotCalID": "PAN-STARRS/PS1.r/Vega", "ZeroPoint": 3173.0162470114, "WavelengthEff": 6155.4659814728,
                         "WavelengthMean": 6241.274452634, "WavelengthPivot": 6201.2000304365,
                         "WavelengthMin": 5391.1116707617, "WavelengthMax": 7038.0787142857},
    "PAN-STARRS/PS1.i": {"PhotCalID": "PAN-STARRS/PS1.i/Vega", "ZeroPoint": 2575.3555343428, "WavelengthEff": 7503.0304998466,
                         "WavelengthMean": 7563.7581578246, "WavelengthPivot": 7534.9604644213,
                         "WavelengthMin": 6783.1724340176, "WavelengthMax": 8306.2380990737},
    "PAN-STARRS/PS1.z": {"PhotCalID": "PAN-STARRS/PS1.z/Vega", "ZeroPoint": 2261.8137088659, "WavelengthEff": 8668.3635610343,
                         "WavelengthMean": 8690.1011083385, "WavelengthPivot": 8674.2022460152,
                         "WavelengthMin": 8030.6815365551, "WavelengthMax": 9350.4721030043},
    "PAN-STARRS/PS1.y": {"PhotCalID": "PAN-STARRS/PS1.y/Vega", "ZeroPoint": 2180.3981124997, "WavelengthEff": 9613.603597131,
                         "WavelengthMean": 9644.6269336021, "WavelengthPivot": 9627.7936716745,
                         "WavelengthMin": 9100.8800657174, "WavelengthMax": 10877.587338262},
    "SLOAN/SDSS.u": {"PhotCalID": "SLOAN/SDSS.u/Vega", "ZeroPoint": 1582.537065543, "WavelengthEff": 3608.0403153219,
                     "WavelengthMean": 3572.1824003193, "WavelengthPivot": 3556.5239668607,
                     "WavelengthMin": 3055.1091291961, "WavelengthMax": 4030.6399499061},
    "SLOAN/SDSS.g": {"PhotCalID": "SLOAN/SDSS.g/Vega", "ZeroPoint": 4023.5732569791, "WavelengthEff": 4671.7822137652,
                     "WavelengthMean": 4750.8231056844, "WavelengthPivot": 4702.4953002767,
                     "WavelengthMin": 3797.6384743979, "WavelengthMax": 5553.0413712781},
    "SLOAN/SDSS.r": {"PhotCalID": "SLOAN/SDSS.r/Vega", "ZeroPoint": 3177.3783043241, "WavelengthEff": 6141.1230039377,
                     "WavelengthMean": 6204.2897962265, "WavelengthPivot": 6175.5788653918,
                     "WavelengthMin": 5418.2260933102, "WavelengthMax": 6994.4222709001},
    "SLOAN/SDSS.i": {"PhotCalID": "SLOAN/SDSS.i/Vega", "ZeroPoint": 2593.3980773509, "WavelengthEff": 7457.889035897,
                     "WavelengthMean": 7519.2690306904, "WavelengthPivot": 7489.9769680118,
                     "WavelengthMin": 6692.4081035994, "WavelengthMax": 8400.3173873874},
    "SLOAN/SDSS.z": {"PhotCalID": "SLOAN/SDSS.z/Vega", "ZeroPoint": 2238.9943970249, "WavelengthEff": 8922.7797236408,
                     "WavelengthMean": 8992.2620587087, "WavelengthPivot": 8946.7096125035,
                     "WavelengthMin": 7964.7001090513, "WavelengthMax": 10873.325548809},
    "GALEX/GALEX.FUV": {"PhotCalID": "GALEX/GALEX.FUV/Vega", "ZeroPoint": 528.52648338002, "WavelengthEff": 1548.8489655268,
                        "WavelengthMean": 1545.8249877975, "WavelengthPivot": 1535.0794801913,
                        "WavelengthMin": 1340.0352002502, "WavelengthMax": 1809.7076490256},
    "GALEX/GALEX.NUV": {"PhotCalID": "GALEX/GALEX.NUV/Vega", "ZeroPoint": 800.98905274781, "WavelengthEff": 2303.3663681246,
                        "WavelengthMean": 2344.8977812227, "WavelengthPivot": 2300.7848229251,
                        "WavelengthMin": 1693.2697798035, "WavelengthMax": 3007.5618833476},
    "Generic/Johnson.V": {"PhotCalID": "Generic/Johnson.V/Vega", "ZeroPoint": 3617.5031467419, "WavelengthEff": 5467.5739972259,
                          "WavelengthMean": 5537.155809702, "WavelengthPivot": 5501.402599255,
                          "WavelengthMin": 4698.0, "WavelengthMax": 7204.0},
}
EMBEDDED_RETRIEVED = "2026-09-28"

# Magnitude system of each catalog's photometry in a filter (not a property of SVO filters).
FILTER_SYSTEMS: dict[str, str] = {
    "GAIA/GAIA3.G": "Vega", "GAIA/GAIA3.Gbp": "Vega", "GAIA/GAIA3.Grp": "Vega",
    "2MASS/2MASS.J": "Vega", "2MASS/2MASS.H": "Vega", "2MASS/2MASS.Ks": "Vega",
    "WISE/WISE.W1": "Vega", "WISE/WISE.W2": "Vega", "WISE/WISE.W3": "Vega", "WISE/WISE.W4": "Vega",
    "PAN-STARRS/PS1.g": "AB", "PAN-STARRS/PS1.r": "AB", "PAN-STARRS/PS1.i": "AB",
    "PAN-STARRS/PS1.z": "AB", "PAN-STARRS/PS1.y": "AB",
    "SLOAN/SDSS.u": "AB", "SLOAN/SDSS.g": "AB", "SLOAN/SDSS.r": "AB", "SLOAN/SDSS.i": "AB", "SLOAN/SDSS.z": "AB",
    "GALEX/GALEX.FUV": "AB", "GALEX/GALEX.NUV": "AB",
    "Generic/Johnson.V": "Vega",
}
_SVO_KEYS = ("filterID", "PhotCalID", "MagSys", "ZeroPoint", "ZeroPointUnit", "WavelengthEff", "WavelengthMean",
             "WavelengthPivot", "WavelengthMin", "WavelengthMax", "WavelengthUnit", "Facility", "Band")


@dataclass(frozen=True, slots=True)
class FilterInfo:
    """Photometric filter: SVO wavelengths plus the zero point of the system the catalog uses."""

    filter_id: str
    mag_system: str
    zero_point_jy: float
    zero_point_origin: str  # "svo", "svo-cache", "embedded", or "AB definition (3631 Jy)"
    wavelength_eff_um: float
    wavelength_mean_um: float | None
    wavelength_pivot_um: float | None
    wavelength_min_um: float | None
    wavelength_max_um: float | None
    svo_vega_zero_point_jy: float | None
    metadata_origin: str  # where the wavelengths came from: "svo", "svo-cache", "embedded"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_svo_params(params: Mapping[str, Any], filter_id: str | None = None) -> None:
    """Reject SVO filter parameters that cannot give a Vega-system calibration (raises :class:`SEDFilterError`).

    Required: a finite ZeroPoint > 0 in Jy of the Vega system (PhotCalID '<filter>/Vega', MagSys 'Vega' when
    given) and a finite WavelengthEff > 0 in Angstrom. SVO's default calibration for ``fps.php?ID=<filter>``
    is Vega (every recorded answer carries PhotCalID '<filter>/Vega', MagSys 'Vega', ZeroPointUnit 'Jy'); an
    AB or ST answer, a missing zero point or a foreign filterID is never stored or used as a Vega zero point.
    """
    where = filter_id or str(params.get("filterID") or "?")
    zp = _num(params.get("ZeroPoint"))
    if zp is None or zp <= 0:
        raise SEDFilterError(f"SVO answer for {where} has no positive finite ZeroPoint")
    unit = str(params.get("ZeroPointUnit") or "Jy").strip()
    if unit.lower() != "jy":
        raise SEDFilterError(f"SVO answer for {where} has ZeroPointUnit {unit!r}, expected 'Jy'")
    cal_id = str(params.get("PhotCalID") or "").strip()
    if not cal_id.endswith("/Vega"):
        raise SEDFilterError(f"SVO answer for {where} has PhotCalID {cal_id!r}, expected '<filter>/Vega'")
    if filter_id is not None and cal_id != f"{filter_id}/Vega":
        raise SEDFilterError(f"SVO answer PhotCalID {cal_id!r} does not belong to filter {filter_id}")
    mag_sys = params.get("MagSys")
    if mag_sys is not None and str(mag_sys).strip().lower() != "vega":
        raise SEDFilterError(f"SVO answer for {where} has MagSys {mag_sys!r}, expected 'Vega'")
    returned_id = params.get("filterID")
    if filter_id is not None and returned_id is not None and str(returned_id).strip() != filter_id:
        raise SEDFilterError(f"SVO answered for filter {returned_id!r} instead of {filter_id}")
    eff = _num(params.get("WavelengthEff"))
    if eff is None or eff <= 0:
        raise SEDFilterError(f"SVO answer for {where} has no positive finite WavelengthEff")
    wl_unit = str(params.get("WavelengthUnit") or "Angstrom")
    if wl_unit.lower() not in {"angstrom", "a"}:
        raise SEDFilterError(f"Unexpected SVO wavelength unit {wl_unit!r} for {where}")


def parse_svo_votable(payload: bytes | str, filter_id: str | None = None) -> dict[str, Any]:
    """Extract and validate filter PARAMs from an SVO FPS VOTable answer (raises SEDFilterError on error answers)."""
    from astropy.io.votable import parse as parse_votable

    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        votable = parse_votable(io.BytesIO(data), verify="ignore")
    except Exception as exc:  # astropy raises many parser exception types
        raise SEDFilterError(f"SVO FPS response is not a VOTable: {exc}") from exc
    for info in votable.infos:
        if info.name == "QUERY_STATUS" and str(info.value).upper() != "OK":
            raise SEDFilterError(f"SVO FPS error: {info.content or info.value}")
    tables = list(votable.iter_tables())
    if not tables:
        raise SEDFilterError("SVO FPS response has no filter table (unknown filter ID?)")
    params: dict[str, Any] = {}
    for param in tables[0].params:
        if param.name in _SVO_KEYS and param.name not in params:
            value = param.value
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if hasattr(value, "item"):
                value = value.item()
            params[param.name] = value
    if params.get("WavelengthEff") is None:
        raise SEDFilterError("SVO FPS response lacks WavelengthEff")
    validate_svo_params(params, filter_id)
    return params


def default_cache_path() -> Path:
    """Disk cache for SVO filter metadata: $ASTROSEARCH_CACHE_DIR or ~/.cache/astrosearch."""
    base = os.getenv("ASTROSEARCH_CACHE_DIR") or str(Path.home() / ".cache" / "astrosearch")
    return Path(base) / "svo_fps_filters.json"


class FilterCatalog:
    """Resolve filter IDs to :class:`FilterInfo` via memory cache -> disk cache -> SVO FPS -> embedded table.

    Filter metadata never changes, so a *validated* SVO answer (:func:`validate_svo_params`) is cached forever
    on disk; malformed answers and cache entries are rejected and never stored. A failed lookup falls back to
    :data:`EMBEDDED_FILTERS` (values copied from SVO) *for that call only*: the fallback is never stored as a
    resolved filter, the filter is retried on a later prefetch once ``retry_after_seconds`` have passed, and a
    successful retry clears the error. :meth:`prefetch` returns the failures of that call, so callers report
    only their own request's problems.

    A request never waits more than ``deadline_seconds`` for SVO in total (the embedded copies are verified, so
    blocking on SVO buys nothing): fetches still running at the deadline keep going in the background when the
    caller's client stays open (``background=True`` with a caller-supplied client: their answers are cached when they
    arrive) and are cancelled otherwise. Concurrent prefetches on one event loop and the SAME client share the
    in-flight fetch of a filter; a prefetch that is interrupted (its caller cancelled, or the supplementary deadline
    of :func:`_run_supplementary`) cancels the fetches no other prefetch is waiting for, so no fetch outlives its
    caller's client. Answers are written to the disk cache once per batch (when no fetch is in flight any more, and at
    the end of each prefetch), not once per filter.
    """

    def __init__(
        self,
        *,
        cache_path: str | os.PathLike[str] | None = None,
        endpoint: str | None = None,
        offline: bool | None = None,
        timeout: float = 20.0,
        max_concurrency: int = 4,
        retry_after_seconds: float = 60.0,
        deadline_seconds: float = 6.0,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else default_cache_path()
        self.endpoint = endpoint or os.getenv("ASTROSEARCH_SVO_FPS_URL") or SVO_FPS_URL
        if offline is None:
            offline = os.getenv("ASTROSEARCH_SED_OFFLINE_FILTERS", "false").lower() in {"1", "true", "yes"}
        self.offline = offline
        self.timeout = timeout
        # A semaphore is created per prefetch call: asyncio primitives bind to one event loop and this
        # catalog is shared across loops (CLI runs, TestClient threads, the API server).
        self.max_concurrency = max_concurrency
        self.retry_after_seconds = retry_after_seconds
        self.deadline_seconds = deadline_seconds
        self._memory: dict[str, tuple[dict[str, Any], str]] = {}  # only validated SVO answers (live or disk cache)
        self._retry_at: dict[str, float] = {}  # filter -> time.monotonic() before which SVO is not retried
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}  # filter -> running fetch (one event loop)
        self._fetch_client: dict[asyncio.Task[dict[str, Any]], httpx.AsyncClient] = {}  # the client a fetch uses
        self._waiters: dict[asyncio.Task[dict[str, Any]], int] = {}  # prefetch calls waiting for a fetch
        # Why a fetch was cancelled: 'deadline' (this catalog's), 'supplementary' (the request's), 'interrupted'.
        self._stopped: dict[asyncio.Task[dict[str, Any]], str] = {}
        self._dirty = False  # SVO answers not yet written to the disk cache
        self._disk_loaded = False
        self.errors: dict[str, str] = {}  # latest unresolved failure per filter (cleared on success)

    # -- disk cache ---------------------------------------------------------
    def _load_disk(self) -> None:
        if self._disk_loaded:
            return
        self._disk_loaded = True
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        filters = payload.get("filters") if isinstance(payload, dict) else None
        if not isinstance(filters, dict):
            logger.warning("Ignoring malformed SVO filter cache %s", self.cache_path)
            return
        for fid, entry in filters.items():
            if not isinstance(entry, dict) or str(fid) in self._memory:
                continue
            try:
                validate_svo_params(entry, str(fid))
                build_filter_info(str(fid), entry, "svo-cache")
            except SEDFilterError as exc:
                logger.warning("Ignoring invalid SVO cache entry %s in %s: %s", fid, self.cache_path, exc)
                continue
            self._memory[str(fid)] = (entry, "svo-cache")

    def _save_disk(self) -> None:
        """Write the validated SVO answers atomically (temporary file + os.replace). A failed write removes its
        temporary file and leaves the cache dirty, so the next batch retries it (on Windows a freshly written cache
        can be held open by a virus scanner or indexer, and os.replace then fails with a sharing violation)."""
        entries = {fid: params for fid, (params, origin) in self._memory.items() if origin in {"svo", "svo-cache"}}
        self._dirty = False
        if not entries:
            return
        tmp: str | None = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.cache_path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"endpoint": self.endpoint, "filters": entries}, handle, indent=1, sort_keys=True)
            os.replace(tmp, self.cache_path)
            tmp = None
        except OSError as exc:
            self._dirty = True
            logger.warning("Could not write SVO filter cache %s: %s", self.cache_path, exc)
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    # -- network ------------------------------------------------------------
    async def _fetch(self, client: httpx.AsyncClient, filter_id: str, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        async with semaphore:
            response = await client.get(self.endpoint, params={"ID": filter_id}, timeout=self.timeout)
        if response.status_code >= 400:
            raise SEDFilterError(f"SVO FPS HTTP {response.status_code}")
        params = parse_svo_votable(response.content, filter_id)
        build_filter_info(filter_id, params, "svo")  # must be usable before it is stored
        params["retrieved_at"] = datetime.now(UTC).isoformat()
        return params

    def _record_failure(self, filter_id: str, message: str) -> None:
        self.errors[filter_id] = message
        self._retry_at[filter_id] = time.monotonic() + self.retry_after_seconds
        logger.warning("SVO FPS lookup failed for %s: %s", filter_id, message)

    def _on_fetch_done(self, filter_id: str, task: asyncio.Task[dict[str, Any]]) -> None:
        """Store the outcome of a fetch (runs even when no caller is waiting for it any more).

        A fetch cancelled because its caller went away ('interrupted') or failing because the caller's client was
        closed under it is not an SVO failure: nothing is recorded and the next prefetch asks SVO again. A fetch
        cancelled at a deadline is an SVO timeout (retried after ``retry_after_seconds``)."""
        if self._inflight.get(filter_id) is task:
            del self._inflight[filter_id]
        self._fetch_client.pop(task, None)
        self._waiters.pop(task, None)
        stopped = self._stopped.pop(task, "deadline")
        if task.cancelled():
            if stopped == "interrupted":
                logger.debug("SVO FPS fetch of %s cancelled with its caller", filter_id)
            elif stopped == "supplementary":
                self._record_failure(filter_id, "TimeoutError: no SVO answer within the request's supplementary deadline")
            else:
                self._record_failure(filter_id,
                                     f"TimeoutError: no SVO answer within the {self.deadline_seconds:g} s deadline")
        else:
            exc = task.exception()
            if exc is not None and _closed_client_error(exc):
                logger.debug("SVO FPS fetch of %s ended with its caller's client: %s", filter_id, exc)
            elif exc is not None:
                self._record_failure(filter_id, f"{type(exc).__name__}: {exc}")
            else:
                self._memory[filter_id] = (task.result(), "svo")
                self.errors.pop(filter_id, None)
                self._retry_at.pop(filter_id, None)
                self._dirty = True
        if self._dirty and not self._inflight:
            self._save_disk()

    def unresolved(self, filter_ids: Iterable[str]) -> list[str]:
        """The filters among ``filter_ids`` without a validated SVO answer (they use the embedded table)."""
        self._load_disk()
        return sorted({fid for fid in filter_ids if fid not in self._memory})

    async def prefetch(
        self, filter_ids: Iterable[str], client: httpx.AsyncClient | None = None, *, background: bool = True,
    ) -> dict[str, str]:
        """Resolve the requested filters (fetching missing ones from SVO concurrently, within the deadline).

        Returns ``{filter_id: error}`` for the filters of THIS call that will use embedded values (a failed
        fetch now, no answer within ``deadline_seconds``, or a failure less than ``retry_after_seconds`` ago).
        Offline catalogs never fetch and return no errors (embedded values are their documented source).
        ``background``: fetches still running at the deadline continue on the caller's ``client`` (it must stay
        open); with False, or without a client (a private one is closed here), they are cancelled. An interrupted
        prefetch (cancelled by its caller) always cancels the fetches no other prefetch waits for.
        """
        self._load_disk()
        missing = sorted({fid for fid in filter_ids if fid not in self._memory})
        if not missing or self.offline:
            return {}
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        failures: dict[str, str] = {}
        waiting: dict[str, asyncio.Task[dict[str, Any]]] = {}
        owned = client is None
        active = client or await _owned_client(self.timeout)
        keep = background and not owned  # our private client closes below: nothing may keep using it
        stop = "deadline"
        try:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            for fid in missing:
                task = self._inflight.get(fid)
                if (task is not None and not task.done() and task.get_loop() is loop
                        and self._fetch_client.get(task) is active):
                    pass  # shared with a concurrent request on the same client
                elif self._retry_at.get(fid, 0.0) > now:
                    failures[fid] = f"{self.errors.get(fid, 'earlier failure')} (retry pending)"
                    continue
                else:
                    task = loop.create_task(self._fetch(active, fid, semaphore))
                    self._fetch_client[task] = active
                    task.add_done_callback(functools.partial(self._on_fetch_done, fid))
                    self._inflight[fid] = task
                self._waiters[task] = self._waiters.get(task, 0) + 1
                waiting[fid] = task
            if waiting:
                await asyncio.wait(set(waiting.values()), timeout=self.deadline_seconds)
        except asyncio.CancelledError as exc:
            # The supplementary deadline (an SVO timeout) or the caller going away (not SVO's fault).
            stop = "supplementary" if exc.args[:1] == (SUPPLEMENTARY_DEADLINE_CANCEL,) else "interrupted"
            keep = False
            raise
        except BaseException:
            stop, keep = "interrupted", False
            raise
        finally:
            try:
                abandoned = []
                for task in set(waiting.values()):
                    left = self._waiters.get(task, 1) - 1
                    if task.done():
                        continue
                    if left > 0:
                        self._waiters[task] = left  # another prefetch still waits for it
                        continue
                    self._waiters.pop(task, None)
                    if not keep:
                        self._stopped[task] = stop
                        task.cancel()
                        abandoned.append(task)
                if abandoned:
                    # Shielded from a repeated cancellation; the fetches are cancelled, so this returns promptly.
                    await asyncio.shield(asyncio.gather(*abandoned, return_exceptions=True))
            finally:
                if owned:
                    await _close_owned(active)
                if self._dirty:
                    self._save_disk()
        for fid, task in waiting.items():
            if not task.done():
                failures[fid] = (f"TimeoutError: no SVO answer within the {self.deadline_seconds:g} s deadline "
                                 "(still fetching in the background)")
            elif task.cancelled() or task.exception() is not None:
                failures[fid] = self.errors.get(fid) or "SVO fetch failed"
        return failures

    def info(self, filter_id: str) -> FilterInfo:
        """A resolved filter, or the embedded SVO copy (origin 'embedded') when SVO has not answered yet.

        The embedded fallback is not memorised, so a later :meth:`prefetch` still asks SVO. A stored entry that
        cannot be turned into a :class:`FilterInfo` is dropped (and retried later) instead of failing the SED.
        """
        self._load_disk()
        if filter_id in self._memory:
            params, origin = self._memory[filter_id]
            try:
                return build_filter_info(filter_id, params, origin)
            except SEDFilterError as exc:
                del self._memory[filter_id]
                self.errors[filter_id] = f"stored SVO entry unusable: {exc}"
                logger.warning("Dropping unusable SVO entry for %s: %s", filter_id, exc)
        if filter_id not in EMBEDDED_FILTERS:
            raise SEDFilterError(f"Unknown filter {filter_id!r}")
        return build_filter_info(filter_id, EMBEDDED_FILTERS[filter_id], "embedded")

    async def get(self, filter_id: str, client: httpx.AsyncClient | None = None) -> FilterInfo:
        await self.prefetch([filter_id], client)
        return self.info(filter_id)


def _closed_client_error(exc: BaseException) -> bool:
    """httpx's error for a request on a closed client (the caller closed it under a background fetch)."""
    return isinstance(exc, RuntimeError) and "client has been closed" in str(exc)


def _angstrom_to_um(value: Any) -> float | None:
    number = _num(value)
    return None if number is None else number / 1.0e4


def build_filter_info(filter_id: str, params: Mapping[str, Any], origin: str) -> FilterInfo:
    """Combine SVO parameters with the catalog's magnitude system into a :class:`FilterInfo`."""
    system = FILTER_SYSTEMS.get(filter_id) or str(params.get("MagSys") or "Vega")
    vega_zp = _num(params.get("ZeroPoint"))
    if system == "AB":
        zero_point, zp_origin = AB_ZERO_POINT_JY, "AB definition (3631 Jy)"
    else:
        if vega_zp is None or vega_zp <= 0:
            raise SEDFilterError(f"No Vega zero point for {filter_id}")
        zero_point, zp_origin = vega_zp, origin
    eff = _angstrom_to_um(params.get("WavelengthEff"))
    if eff is None or eff <= 0:
        raise SEDFilterError(f"No effective wavelength for {filter_id}")
    return FilterInfo(
        filter_id=filter_id,
        mag_system=system,
        zero_point_jy=zero_point,
        zero_point_origin=zp_origin,
        wavelength_eff_um=eff,
        wavelength_mean_um=_angstrom_to_um(params.get("WavelengthMean")),
        wavelength_pivot_um=_angstrom_to_um(params.get("WavelengthPivot")),
        wavelength_min_um=_angstrom_to_um(params.get("WavelengthMin")),
        wavelength_max_um=_angstrom_to_um(params.get("WavelengthMax")),
        svo_vega_zero_point_jy=vega_zp,
        metadata_origin=origin,
    )


_DEFAULT_FILTERS: FilterCatalog | None = None


def default_filter_catalog() -> FilterCatalog:
    """Process-wide FilterCatalog (disk-cached SVO metadata)."""
    global _DEFAULT_FILTERS
    if _DEFAULT_FILTERS is None:
        _DEFAULT_FILTERS = FilterCatalog()
    return _DEFAULT_FILTERS


# ---------------------------------------------------------------------------
# Unit Conversions (pure functions)
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Finite float or None (catalog nulls, NaN, '', and non-numeric strings become None)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pogson_mag_to_jy(mag: float, zero_point_jy: float, mag_err: float | None = None) -> tuple[float, float | None]:
    """F = ZP * 10^(-0.4 m); sigma_F = F * (ln 10 / 2.5) * sigma_m."""
    flux = zero_point_jy * 10.0 ** (-0.4 * mag)
    err = None if mag_err is None else flux * mag_err / POGSON
    return flux, err


def ab_mag_to_jy(mag: float, mag_err: float | None = None) -> tuple[float, float | None]:
    """AB magnitude -> Jy (AB 0 mag = 3631 Jy)."""
    return pogson_mag_to_jy(mag, AB_ZERO_POINT_JY, mag_err)


def jy_to_ab_mag(flux_jy: float) -> float:
    return -2.5 * math.log10(flux_jy / AB_ZERO_POINT_JY)


def vega_mag_to_jy(mag: float, zero_point_jy: float, mag_err: float | None = None) -> tuple[float, float | None]:
    """Vega magnitude -> Jy given the filter's Vega zero point (e.g. 2MASS Ks: 666.8 Jy)."""
    return pogson_mag_to_jy(mag, zero_point_jy, mag_err)


def sdss_asinh_mag_to_jy(mag: float, band: str, mag_err: float | None = None) -> tuple[float, float | None]:
    """SDSS asinh magnitude -> Jy (Lupton et al. 1999; SDSS AB offsets applied).

    m = -(2.5/ln10) [asinh((f/f0)/(2b)) + ln b]  =>  f/f0 = 2b sinh(-m ln10/2.5 - ln b),
    F_nu = 3631 Jy * (f/f0) * 10^(-0.4 (m_AB - m_SDSS)), sigma_(f/f0) = sigma_m (ln10/2.5) sqrt((f/f0)^2 + 4b^2).
    """
    key = band.lower()
    b = SDSS_ASINH_B[key]
    ratio = 2.0 * b * math.sinh(-mag / POGSON - math.log(b))
    scale = AB_ZERO_POINT_JY * 10.0 ** (-0.4 * SDSS_AB_OFFSET[key])
    err = None if mag_err is None else scale * mag_err / POGSON * math.sqrt(ratio * ratio + 4.0 * b * b)
    return scale * ratio, err


def wavelength_um_to_hz(wavelength_um: float) -> float:
    return C_UM_HZ / wavelength_um


def hz_to_wavelength_um(frequency_hz: float) -> float:
    return C_UM_HZ / frequency_hz


def nu_fnu_cgs(frequency_hz: float, flux_jy: float) -> float:
    """nu * F_nu in erg s^-1 cm^-2."""
    return frequency_hz * flux_jy * JY_CGS


def xray_band_flux_to_jy(
    flux_cgs: float,
    e_lo_kev: float,
    e_hi_kev: float,
    *,
    photon_index: float = XRAY_PHOTON_INDEX,
    energy_kev: float | None = None,
) -> tuple[float, float]:
    """Band energy flux (erg s^-1 cm^-2 in [e_lo, e_hi] keV) -> (F_nu in Jy, frequency in Hz).

    Power law N(E) = K E^-Gamma: F_E(E) = K E^(1-Gamma) (erg s^-1 cm^-2 keV^-1 with E in keV),
    F = K (E2^(2-Gamma) - E1^(2-Gamma)) / (2-Gamma)  (Gamma != 2),  F = K ln(E2/E1)  (Gamma = 2).
    Evaluated at E0 = sqrt(E1 E2) by default; F_nu = F_E * h with h in keV s.
    """
    if not (0 < e_lo_kev < e_hi_kev):
        raise ValueError("X-ray band must satisfy 0 < e_lo < e_hi")
    e0 = energy_kev if energy_kev is not None else math.sqrt(e_lo_kev * e_hi_kev)
    if abs(photon_index - 2.0) < 1e-12:
        norm = flux_cgs / math.log(e_hi_kev / e_lo_kev)
    else:
        p = 2.0 - photon_index
        norm = flux_cgs * p / (e_hi_kev**p - e_lo_kev**p)
    f_e = norm * e0 ** (1.0 - photon_index)  # erg s^-1 cm^-2 keV^-1
    f_nu_cgs = f_e * H_KEV_S  # erg s^-1 cm^-2 Hz^-1
    return f_nu_cgs / JY_CGS, e0 / H_KEV_S


def xray_band_scale(e_lo: float, e_hi: float, to_lo: float, to_hi: float, photon_index: float = XRAY_PHOTON_INDEX) -> float:
    """Ratio F(to_lo..to_hi) / F(e_lo..e_hi) for the same power law."""
    def integral(a: float, b: float) -> float:
        if abs(photon_index - 2.0) < 1e-12:
            return math.log(b / a)
        p = 2.0 - photon_index
        return (b**p - a**p) / p

    return integral(to_lo, to_hi) / integral(e_lo, e_hi)


# ---------------------------------------------------------------------------
# Photometry Extraction Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MagBand:
    """One magnitude column of a catalog."""

    band: str
    filter_id: str
    mag_cols: tuple[str, ...]
    err_cols: tuple[str, ...] = ()
    qual_col: str | None = None  # per-band quality string (2MASS/WISE ph_qual)
    qual_index: int | None = None
    kind: str = "pogson"  # "pogson" (Vega or AB via FilterInfo) or "sdss_asinh"


@dataclass(frozen=True, slots=True)
class RadioBand:
    band: str
    flux_col: str
    err_col: str | None
    to_jy: float
    frequency_hz: float
    note: str


@dataclass(frozen=True, slots=True)
class XrayBand:
    band: str
    e_lo_kev: float
    e_hi_kev: float
    flux_col: str | None = None
    err_col: str | None = None
    lo_col: str | None = None
    hi_col: str | None = None
    rate_col: str | None = None
    rate_err_col: str | None = None
    ecf: float | None = None
    note: str = ""


MAG_SPECS: dict[str, tuple[str, list[MagBand]]] = {
    "gaia_dr3": ("Gaia DR3", [
        MagBand("G", "GAIA/GAIA3.G", ("phot_g_mean_mag",)),
        MagBand("BP", "GAIA/GAIA3.Gbp", ("phot_bp_mean_mag",)),
        MagBand("RP", "GAIA/GAIA3.Grp", ("phot_rp_mean_mag",)),
    ]),
    "twomass_psc": ("2MASS", [
        MagBand("J", "2MASS/2MASS.J", ("j_m",), ("j_msigcom",), "ph_qual", 0),
        MagBand("H", "2MASS/2MASS.H", ("h_m",), ("h_msigcom",), "ph_qual", 1),
        MagBand("Ks", "2MASS/2MASS.Ks", ("k_m",), ("k_msigcom",), "ph_qual", 2),
    ]),
    "vizier_2mass_reference": ("2MASS", [
        MagBand("J", "2MASS/2MASS.J", ("Jmag",), ("e_Jmag",), "Qflg", 0),
        MagBand("H", "2MASS/2MASS.H", ("Hmag",), ("e_Hmag",), "Qflg", 1),
        MagBand("Ks", "2MASS/2MASS.Ks", ("Kmag",), ("e_Kmag",), "Qflg", 2),
    ]),
    "allwise": ("WISE", [
        MagBand("W1", "WISE/WISE.W1", ("w1mpro",), ("w1sigmpro",), "ph_qual", 0),
        MagBand("W2", "WISE/WISE.W2", ("w2mpro",), ("w2sigmpro",), "ph_qual", 1),
        MagBand("W3", "WISE/WISE.W3", ("w3mpro",), ("w3sigmpro",), "ph_qual", 2),
        MagBand("W4", "WISE/WISE.W4", ("w4mpro",), ("w4sigmpro",), "ph_qual", 3),
    ]),
    "panstarrs_dr2": ("Pan-STARRS1", [
        MagBand(b, f"PAN-STARRS/PS1.{b}", (f"{b}MeanPSFMag",), (f"{b}MeanPSFMagErr",)) for b in "grizy"
    ]),
    "sdss": ("SDSS", [
        MagBand(b, f"SLOAN/SDSS.{b}", (f"psfMag_{b}",), (f"psfMagErr_{b}",), kind="sdss_asinh") for b in "ugriz"
    ]),
}

# GALEX columns as named by MAST GUVcat / GALEX GR6+7 (fuv_mag, fuv_magerr) and VizieR (FUVmag, e_FUVmag).
GALEX_BANDS: list[MagBand] = [
    MagBand("FUV", "GALEX/GALEX.FUV", ("fuv_mag", "FUVmag", "mag_fuv"), ("fuv_magerr", "e_FUVmag", "magerr_fuv")),
    MagBand("NUV", "GALEX/GALEX.NUV", ("nuv_mag", "NUVmag", "mag_nuv"), ("nuv_magerr", "e_NUVmag", "magerr_nuv")),
]

RADIO_SPECS: dict[str, tuple[str, list[RadioBand]]] = {
    # Units verified from the archives' column metadata (FIRST/NVSS mJy, VLASS Jy, LoTSS mJy).
    "first": ("FIRST (VLA)", [RadioBand("1.4 GHz", "int_flux_20_cm", "flux_20_cm_error", 1e-3, 1.4e9,
                                        "integrated flux density; error is the local rms (mJy/beam)")]),
    "nvss": ("NVSS (VLA)", [RadioBand("1.4 GHz", "flux_20_cm", "flux_20_cm_error", 1e-3, 1.4e9,
                                      "integrated flux density (45 arcsec beam)")]),
    "vlass": ("VLASS (VLA)", [RadioBand("3 GHz", "Flux", "e_Flux", 1.0, 3.0e9,
                                        "epoch-1 integrated flux density (Bruzewski et al. 2021)")]),
    "lotss": ("LoTSS (LOFAR)", [RadioBand("144 MHz", "SpeakTot", "e_SpeakTot", 1e-3, 1.44e8,
                                          "integrated Stokes I flux density")]),
    "lotss_dr2": ("LoTSS (LOFAR)", [RadioBand("144 MHz", "SpeakTot", "e_SpeakTot", 1e-3, 1.44e8,
                                              "integrated Stokes I flux density")]),
}

XRAY_SPECS: dict[str, tuple[str, list[XrayBand]]] = {
    "chandra": ("Chandra ACIS", [XrayBand("0.5-7 keV", 0.5, 7.0, flux_col="b_flux_ap", lo_col="b_flux_ap_lo",
                                          hi_col="b_flux_ap_hi", note="CSC 2.1 broad-band aperture flux")]),
    "xmm": ("XMM-Newton EPIC", [XrayBand("0.2-12 keV", 0.2, 12.0, flux_col="ep_flux", err_col="ep_flux_error",
                                         note="4XMM/5XMM EPIC band 8 flux")]),
    "rosat": ("ROSAT PSPC", [XrayBand("0.1-2.4 keV", 0.1, 2.4, rate_col="count_rate", rate_err_col="count_rate_error",
                                      ecf=ROSAT_PSPC_ECF, note="2RXS count rate; " + ROSAT_PSPC_ECF_ASSUMPTION)]),
    "rosat_bsc": ("ROSAT PSPC", [XrayBand("0.1-2.4 keV", 0.1, 2.4, rate_col="count_rate", rate_err_col="count_rate_error",
                                          ecf=ROSAT_PSPC_ECF, note="1RXS count rate; " + ROSAT_PSPC_ECF_ASSUMPTION)]),
}

def _flag_true(value: Any) -> bool:
    return str(value).strip().upper() in {"T", "TRUE", "1"}


# Catalog quality flags surfaced on the points they affect: (column, predicate, note, sets quality_warning).
QUALITY_NOTES: dict[str, list[tuple[str, Any, str, bool]]] = {
    "chandra": [
        ("pileup_flag", _flag_true, "CSC pileup_flag: pile-up, flux likely underestimated", True),
        ("sat_src_flag", _flag_true, "CSC sat_src_flag: source saturated in at least one observation", True),
        ("extent_flag", _flag_true, "CSC extent_flag: source extended", False),
    ],
    "xmm": [
        ("sum_flag", lambda v: (_num(v) or 0) >= 2, "XMM sum_flag >= 2: possible spurious detection or flux problem", True),
        ("extent", lambda v: (_num(v) or 0) > 0, "XMM extent > 0: extended source (flux includes diffuse emission)", False),
    ],
    "vlass": [("Flag", lambda v: _num(v) == 2, "VLASS Flag=2: redundant component", True)],
}

# SIMBAD 'allfluxes' literature magnitudes (Vega). SIMBAD K is overwhelmingly 2MASS Ks.
SIMBAD_BANDS: list[tuple[str, str, str]] = [
    ("V", "Generic/Johnson.V", "V"),
    ("G", "GAIA/GAIA3.G", "G"),
    ("K", "2MASS/2MASS.Ks", "K"),
]

# Separation floor (arcsec) below which a catalog's nearest member is always accepted: about half the
# angular resolution (radio beams, X-ray PSFs) or a few times typical astrometric scatter.
MATCH_FLOOR_ARCSEC: dict[str, float] = {
    "gaia_dr3": 1.0, "panstarrs_dr2": 1.0, "sdss": 1.0, "twomass_psc": 1.5, "vizier_2mass_reference": 1.5,
    "allwise": 3.0, "simbad": 3.0, "ned": 3.0, "exoplanet_archive": 2.0,
    "first": 2.7, "nvss": 22.5, "vlass": 1.25, "lotss": 3.0, "lotss_dr2": 3.0,
    "rosat": 15.0, "rosat_bsc": 15.0, "chandra": 1.0, "xmm": 2.0,
}
WAVELENGTH_FLOOR_ARCSEC: dict[str, float] = {
    "optical": 1.0, "infrared": 2.0, "ultraviolet": 2.5, "uv": 2.5, "radio": 5.0, "xray": 5.0,
}
CATALOG_PRIORITY: tuple[str, ...] = (
    "gaia_dr3", "panstarrs_dr2", "sdss", "twomass_psc", "vizier_2mass_reference", "allwise",
    "first", "nvss", "vlass", "lotss", "lotss_dr2", "chandra", "xmm", "rosat", "rosat_bsc", "simbad", "ned",
)


def regime_for_wavelength(wavelength_um: float) -> Regime:
    if wavelength_um >= 1000.0:
        return "radio"
    if wavelength_um >= 0.9:
        return "infrared"
    if wavelength_um >= 0.32:
        return "optical"
    if wavelength_um >= 0.01:
        return "ultraviolet"
    return "xray"


# ---------------------------------------------------------------------------
# SED Data Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SEDPoint:
    """One photometric measurement converted to flux density."""

    band: str
    facility: str
    catalog: str
    source_id: str
    wavelength_um: float
    frequency_hz: float
    flux_jy: float
    flux_err_jy: float | None
    nu_fnu_erg_s_cm2: float
    is_upper_limit: bool
    regime: str
    filter_id: str | None = None
    magnitude: float | None = None
    magnitude_error: float | None = None
    mag_system: str | None = None
    zero_point_jy: float | None = None
    zero_point_origin: str | None = None
    quality_flag: str | None = None
    nu_fnu_err_erg_s_cm2: float | None = None
    notes: list[str] = field(default_factory=list)
    quality_warning: bool = False  # catalogue flag / consistency check failed: never used by classification
    warnings: list[str] = field(default_factory=list)  # the reasons for quality_warning (also in notes)

    def warn(self, note: str) -> None:
        self.quality_warning = True
        if note not in self.notes:
            self.notes.append(note)
        if note not in self.warnings:
            self.warnings.append(note)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemberChoice:
    """Nearest member of one catalog in the selected crossmatch group."""

    catalog: str
    source_id: str
    ra: float | None
    dec: float | None
    separation_arcsec: float | None  # None: the row has no separation (never used)
    tolerance_arcsec: float
    used: bool
    reason: str
    wavelength: str | None
    data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Radio catalogs: every component whose flux is summed into the point (the nearest one first, if accepted).
    components: list[dict[str, Any]] = field(default_factory=list, repr=False)
    notes: list[str] = field(default_factory=list)
    group_id: str | None = None  # crossmatch group the row came from
    designation: str | None = None  # the object's name that identifies this row (see select_members)
    # Quality warnings of the row as a whole: every point extracted from it carries them (quality_warning).
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "catalog": self.catalog, "source_id": self.source_id, "ra": self.ra, "dec": self.dec,
            "separation_arcsec": self.separation_arcsec, "tolerance_arcsec": self.tolerance_arcsec,
            "used": self.used, "reason": self.reason, "wavelength": self.wavelength, "group_id": self.group_id,
        }
        if self.designation:
            out["designation"] = self.designation
        if self.components:
            out["components"] = [{k: c[k] for k in ("source_id", "separation_arcsec", "role")} for c in self.components]
        if self.notes:
            out["notes"] = list(self.notes)
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


@dataclass(slots=True)
class Evidence:
    text: str
    weights: dict[str, float]


# ---------------------------------------------------------------------------
# Group & Member Selection
# ---------------------------------------------------------------------------


def _record_dict(record: UnifiedRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, UnifiedRecord):
        return record.as_dict()
    if isinstance(record, Mapping):
        return dict(record)
    raise SEDInputError("record must be a UnifiedRecord or its dict form")


def _expect_mapping(value: Any, path: str, *, optional: bool = True) -> None:
    if value is None and optional:
        return
    if not isinstance(value, Mapping):
        raise SEDInputError(f"record {path} must be an object (mapping), not {type(value).__name__}")


def _expect_list(value: Any, path: str) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, str | bytes | Mapping) or not isinstance(value, Sequence):
        raise SEDInputError(f"record {path} must be a list, not {type(value).__name__}")
    return value


def validate_record_shape(record: Mapping[str, Any]) -> None:
    """:class:`SEDInputError` naming the offending path unless the record has the shape sed_from_record reads:
    ``target``, ``resolved_object`` and ``provenance`` objects (or null); ``resolved_object.aliases`` a list of
    names; ``crossmatch_groups`` a list of objects whose ``members`` are lists of objects with ``data``,
    ``metadata`` and ``position_at_epoch`` objects (or null); ``failures`` a list of objects."""
    for key in ("target", "resolved_object", "provenance"):
        _expect_mapping(record.get(key), key)
    resolved = record.get("resolved_object") or {}
    for index, alias in enumerate(_expect_list(resolved.get("aliases"), "resolved_object.aliases")):
        if not isinstance(alias, str):
            raise SEDInputError(f"record resolved_object.aliases[{index}] must be a string, not {type(alias).__name__}")
    for index, failure in enumerate(_expect_list(record.get("failures"), "failures")):
        _expect_mapping(failure, f"failures[{index}]", optional=False)
    for gi, group in enumerate(_expect_list(record.get("crossmatch_groups"), "crossmatch_groups")):
        _expect_mapping(group, f"crossmatch_groups[{gi}]", optional=False)
        for mi, member in enumerate(_expect_list(group.get("members"), f"crossmatch_groups[{gi}].members")):
            path = f"crossmatch_groups[{gi}].members[{mi}]"
            _expect_mapping(member, path, optional=False)
            for key in ("data", "metadata", "position_at_epoch"):
                _expect_mapping(member.get(key), f"{path}.{key}")


def _ci_get(data: Mapping[str, Any], *names: str) -> Any:
    """Case-insensitive lookup of the first present, non-null column."""
    lowered = {str(k).lower(): v for k, v in data.items()}
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


# Name resolvers: their rows sit at ~0 arcsec by construction for name queries (Sesame's position is SIMBAD's), so
# a group of resolver rows alone never outranks a group holding survey photometry.
RESOLVER_CATALOGS = frozenset({"simbad", "ned"})


def _member_sigma_arcsec(member: Mapping[str, Any]) -> float | None:
    """1-sigma position error of a row at the target epoch: the catalogue error (+) the proper-motion growth of the
    epoch propagation (``position_at_epoch.pm_growth_arcsec`` from the crossmatch). The crossmatch's full
    ``sigma_arcsec`` is not used: for radio rows it includes a source-structure term, and the SED tolerance already
    allows half the fitted radio extent (:func:`member_tolerance_arcsec`)."""
    err = _num(member.get("positional_error_arcsec"))
    growth = _num((member.get("position_at_epoch") or {}).get("pm_growth_arcsec"))
    if err is None and growth is None:
        return None
    return math.hypot(max(err or 0.0, 0.0), max(growth or 0.0, 0.0))


def _row_tolerance(member: Mapping[str, Any]) -> float:
    catalog = str(member.get("catalog"))
    size_col = RADIO_SIZE_COLUMNS.get(catalog)
    major = _num(_ci_get(member.get("data") or {}, size_col)) if size_col else None
    return member_tolerance_arcsec(catalog, (member.get("metadata") or {}).get("wavelength"),
                                   _member_sigma_arcsec(member), major)


def _within_tolerance(member: Mapping[str, Any]) -> bool:
    sep = _num(member.get("separation_arcsec"))
    return sep is not None and sep <= _row_tolerance(member)


def select_group(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The crossmatch group of the target itself.

    The group flagged ``contains_target`` by the crossmatch (Bayesian association) when present; otherwise the group
    with the most survey catalogs (not SIMBAD/NED, see :data:`RESOLVER_CATALOGS`) holding a row within that catalog's
    tolerance, then the most catalogs within tolerance, then the nearest row. Groups without a separation are skipped.
    Only the *preference* between equally acceptable rows depends on this choice: :func:`select_record_members`
    takes each catalog's row from every group.
    """
    groups = [g for g in record.get("crossmatch_groups") or []
              if any(_num(m.get("separation_arcsec")) is not None for m in g.get("members") or [])]
    flagged = [g for g in groups if g.get("contains_target") is True]
    if flagged:
        return flagged[0]
    best: tuple[tuple[int, int, float], dict[str, Any]] | None = None
    for group in groups:
        members = group.get("members") or []
        within = {str(m.get("catalog")) for m in members if _within_tolerance(m)}
        nearest = min(s for s in (_num(m.get("separation_arcsec")) for m in members) if s is not None)
        key = (-len(within - RESOLVER_CATALOGS), -len(within), nearest)
        if best is None or key < best[0]:
            best = (key, group)
    return None if best is None else best[1]


def select_record_members(
    record: Mapping[str, Any], centre: tuple[float, float] | None = None,
) -> tuple[list[MemberChoice], dict[str, Any] | None, list[str]]:
    """Members of the SED: per catalog, the best row of ANY crossmatch group (see :func:`select_members`).

    The crossmatch may split one physical object over several groups (a high proper-motion star whose 2MASS row
    the association assigned elsewhere, a Gaia nucleus in its own group, a group of name-resolver rows that sits at
    0 arcsec by construction): rows of every group are pooled, rows of the target group (:func:`select_group`) are
    preferred when several are within tolerance, and a row from another group is used (and reported) when it is
    within the catalog's tolerance. Returns (members, target group, notes).
    """
    groups = list(record.get("crossmatch_groups") or [])
    primary = select_group(record)
    if primary is None:
        return [], None, []
    pooled: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        gid = str(group.get("group_id") or f"group-{index + 1}")
        for member in group.get("members") or []:
            pooled.append({**member, "group_id": gid})
    primary_id = str(primary.get("group_id") or f"group-{groups.index(primary) + 1}")
    resolved = record.get("resolved_object") or {}
    canonical = str(resolved.get("canonical_name") or "").strip() or None
    names = [resolved.get("query"), resolved.get("canonical_name"), *(resolved.get("aliases") or [])]
    members = select_members({"members": pooled}, centre, primary_group_id=primary_id, canonical_name=canonical,
                             object_names=[str(n) for n in names if n])
    notes: list[str] = []
    elsewhere = [f"{m.catalog} ({m.group_id})" for m in members if m.used and m.group_id not in (None, primary_id)
                 and m.catalog not in RADIO_SPECS]
    if elsewhere:
        notes.append(f"members taken from crossmatch groups other than the target group {primary_id} (within each "
                     f"catalog's tolerance, or identified by designation): {', '.join(elsewhere)}")
    beyond = [f"{m.catalog} {m.source_id} ({_fmt_sep(m.separation_arcsec, 2)} arcsec > {m.tolerance_arcsec:.2f} arcsec)"
              for m in members if m.used and m.designation and m.separation_arcsec is not None
              and m.separation_arcsec > m.tolerance_arcsec]
    if beyond:
        notes.append("members beyond the positional tolerance, identified by designation (a name of the object): "
                     + ", ".join(beyond))
    flagged = flag_moving_stacked_xray(members, pooled, target_proper_motion(record, members))
    if flagged:
        notes.append("stacked X-ray catalogue flux of a moving source flagged (not used by the classification): "
                     + ", ".join(flagged))
    competing = flag_competing_xray_counterparts(members, pooled, record)
    if competing:
        notes.append("X-ray source closer to another optical source than to the target flagged (not used by the "
                     "classification): " + ", ".join(competing))
    return members, primary, notes


# X-ray catalogues whose rows are unique sources fitted at ONE fixed position over every observation of the field:
# the 5XMM stacked catalogue (HEASARC xmmssc: EPIC flux fitted over n_contrib overlapping observations, first/last
# observation MJD in time/end_time) and CSC 2.1 master sources. A source that moves by more than the tolerance over those observations is split into several unique
# sources, each with a flux averaged over observations in which the source was elsewhere: Proxima Cen has three 5XMM
# unique sources (2001.8, 2009.2, 2017.1 positions) with ep_flux 4.2e-12, 6.9e-15 and 1.5e-15 erg/s/cm2, the
# 6.9e-15 one with ep_det_ml = 1.7e6 (impossible for a 7e-15 source observed once).
STACKED_XRAY_CATALOGS = frozenset({"xmm", "chandra"})
STACKED_XRAY_SPAN_COLUMNS: dict[str, tuple[str, str]] = {"xmm": ("time", "end_time")}  # MJD of first/last observation
STACKED_XRAY_MAX_SPAN_YEARS = 25.0  # Chandra/XMM-Newton archives since 1999: bound when a row states no span
# On-axis PSF FWHM (arcsec): a source that moved less than this over the stacked observations loses little flux in a
# fixed-position fit (XMM-Newton Users Handbook 3.2.1: EPIC 4-6 arcsec; Chandra ACIS ~0.5-0.8 arcsec on axis).
STACKED_XRAY_PSF_FWHM_ARCSEC: dict[str, float] = {"xmm": 6.0, "chandra": 0.8}
DAYS_PER_YEAR = 365.25


def target_proper_motion(
    record: Mapping[str, Any], members: Sequence[MemberChoice],
) -> tuple[float, str] | None:
    """Total proper motion (mas/yr) of the target and its origin: the record target's (name resolution), else the
    used Gaia DR3 member's; None when unknown."""
    target = record.get("target") or {}
    pmra, pmdec = _num(target.get("pm_ra_masyr")), _num(target.get("pm_dec_masyr"))
    if pmra is not None and pmdec is not None:
        return math.hypot(pmra, pmdec), "target proper motion"
    gaia = next((m for m in members if m.used and m.catalog == "gaia_dr3"), None)
    if gaia is not None:
        pmra, pmdec = _num(_ci_get(gaia.data, "pmra")), _num(_ci_get(gaia.data, "pmdec"))
        if pmra is not None and pmdec is not None:
            return math.hypot(pmra, pmdec), f"Gaia DR3 {gaia.source_id}"
    return None


def stacked_xray_span_years(catalog: str, data: Mapping[str, Any]) -> float | None:
    """Years between the first and last observation contributing to a stacked X-ray row, when the row states them."""
    columns = STACKED_XRAY_SPAN_COLUMNS.get(catalog)
    if columns is None:
        return None
    first, last = (_num(_ci_get(data, c)) for c in columns)
    if first is None or last is None or last < first:
        return None
    return (last - first) / DAYS_PER_YEAR


def flag_moving_stacked_xray(
    members: Sequence[MemberChoice], rows: Sequence[Mapping[str, Any]], proper_motion: tuple[float, str] | None,
) -> list[str]:
    """Flag (``MemberChoice.warnings``) the used rows of stacked X-ray catalogues (:data:`STACKED_XRAY_CATALOGS`) of a
    target that moved by more than max(row tolerance, PSF FWHM) over the observations stacked into it: over the span
    the row states (5XMM time/end_time), or -- when it states none -- whenever several unique sources of that
    catalogue lie within tolerance (the catalogue split the moving source) and the target moves more than that within
    :data:`STACKED_XRAY_MAX_SPAN_YEARS`. Returns '<catalog> <source_id>' of the flagged rows."""
    if proper_motion is None:
        return []
    pm_masyr, origin = proper_motion
    flagged = []
    for choice in members:
        if not choice.used or choice.catalog not in STACKED_XRAY_CATALOGS:
            continue
        within = [m for m in rows if str(m.get("catalog")) == choice.catalog and _within_tolerance(m)]
        split = len({str(m.get("source_id")) for m in within} | {choice.source_id})
        span = stacked_xray_span_years(choice.catalog, choice.data)
        limit = max(choice.tolerance_arcsec, STACKED_XRAY_PSF_FWHM_ARCSEC.get(choice.catalog, 0.0))
        if span is not None:
            moved = pm_masyr / 1000.0 * span
            if moved <= limit:
                continue
            why = (f"the target ({pm_masyr:.0f} mas/yr, {origin}) moved {moved:.1f} arcsec (> {limit:.1f} arcsec, the "
                   f"tolerance or PSF FWHM) over the {span:.1f} yr of observations stacked into this row")
        elif split > 1 and pm_masyr / 1000.0 * STACKED_XRAY_MAX_SPAN_YEARS > limit:
            why = (f"the target moves {pm_masyr:.0f} mas/yr ({origin}) and {split} {choice.catalog} unique sources lie "
                   "within tolerance (the catalogue split the moving source)")
        else:
            continue
        if split > 1 and span is not None:
            why += f"; {split} {choice.catalog} unique sources within tolerance are this object at different epochs"
        choice.warnings.append(
            f"{choice.catalog} flux of a moving source in a stacked catalogue: {why}. Unique-source fluxes are fitted "
            "at one fixed position over all observations, so observations in which the source was elsewhere dilute "
            "them (5XMM rows of Proxima Cen differ by a factor 600); flux unreliable")
        flagged.append(f"{choice.catalog} {choice.source_id}")
    return flagged


def _closest_approach_arcsec(
    point: tuple[float, float], ra: float, dec: float, epoch: float | None, pm: tuple[float, float] | None,
    span: tuple[float, float] | None,
) -> float:
    """Closest approach (arcsec) to the fixed ``point`` of a source at (ra, dec, epoch) moving linearly with ``pm``
    (mas/yr) over the years ``span``; the plain separation when the motion or the span is unknown."""
    if pm is None or epoch is None or span is None:
        return haversine_arcsec(point[0], point[1], ra, dec)
    ax, ay = tangent_offset_arcsec(point[0], point[1], *propagate_radec(ra, dec, pm[0], pm[1], epoch, span[0]))
    bx, by = tangent_offset_arcsec(point[0], point[1], *propagate_radec(ra, dec, pm[0], pm[1], epoch, span[1]))
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length2))
    return math.hypot(ax + t * dx, ay + t * dy)


def _xray_epoch_span(row: Mapping[str, Any]) -> tuple[float, float] | None:
    """Years of the observations behind an X-ray row: its ``epoch_range`` (crossmatch), else its single epoch."""
    span = row.get("epoch_range")
    if isinstance(span, list | tuple) and len(span) == 2:
        lo, hi = _num(span[0]), _num(span[1])
        if lo is not None and hi is not None and hi >= lo:
            return lo, hi
    epoch = _num(row.get("epoch"))
    return None if epoch is None else (epoch, epoch)


def _target_track(
    record: Mapping[str, Any], members: Sequence[MemberChoice],
) -> tuple[float, float, float | None, tuple[float, float] | None] | None:
    """(ra, dec, epoch, proper motion) of the target: the record target with its proper motion (name resolution),
    else the used Gaia DR3 member at its epoch, else the static record target."""
    target = record.get("target") or {}
    ra, dec = _num(target.get("ra")), _num(target.get("dec"))
    pmra, pmdec, epoch = _num(target.get("pm_ra_masyr")), _num(target.get("pm_dec_masyr")), _num(target.get("epoch"))
    if ra is not None and dec is not None and pmra is not None and pmdec is not None and epoch is not None:
        return ra, dec, epoch, (pmra, pmdec)
    gaia = next((m for m in members if m.used and m.catalog == "gaia_dr3" and m.ra is not None and m.dec is not None),
                None)
    if gaia is not None:
        g_pm = (_num(_ci_get(gaia.data, "pmra")), _num(_ci_get(gaia.data, "pmdec")))
        g_epoch = _num(_ci_get(gaia.data, "ref_epoch")) or 2016.0
        return gaia.ra, gaia.dec, g_epoch, (None if None in g_pm else g_pm)  # type: ignore[return-value]
    if ra is None or dec is None:
        return None
    return ra, dec, None, None


def flag_competing_xray_counterparts(
    members: Sequence[MemberChoice], rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any],
) -> list[str]:
    """Flag (``MemberChoice.warnings``) used X-ray rows that belong to another optical source.

    An X-ray row within its tolerance of the target can still be a neighbour's: Gl 229B (a T7 dwarf, X-ray dark) got
    the Chandra flux of its M1V primary Gl 229A 7.7 arcsec away. Every Gaia DR3 row that is not the target's own
    counterpart (not the used Gaia member and farther from the target than its tolerance) and is brighter than the
    target's Gaia counterpart (when it has one) competes: positions of both the target and the competitor are
    propagated over the X-ray observations (:func:`_closest_approach_arcsec`), and a competitor within the X-ray row's
    tolerance that comes closer to the X-ray position than the target by more than the row's 1-sigma error (at least
    0.5 arcsec) takes the flux. Returns '<catalog> <source_id>' of the flagged rows."""
    track = _target_track(record, members)
    if track is None:
        return []
    gaia_member = next((m for m in members if m.used and m.catalog == "gaia_dr3"), None)
    target_g = _num(_ci_get(gaia_member.data, "phot_g_mean_mag")) if gaia_member is not None else None
    competitors = [r for r in rows if str(r.get("catalog")) == "gaia_dr3"
                   and (gaia_member is None or str(r.get("source_id")) != gaia_member.source_id)
                   and _num(r.get("ra")) is not None and _num(r.get("dec")) is not None
                   and _num(r.get("separation_arcsec")) is not None and not _within_tolerance(r)]
    flagged: list[str] = []
    for choice in members:
        if not choice.used or choice.catalog not in XRAY_SPECS or choice.ra is None or choice.dec is None:
            continue
        row = next((r for r in rows if str(r.get("catalog")) == choice.catalog
                    and str(r.get("source_id")) == choice.source_id), {})
        span = _xray_epoch_span(row)
        xpos = (choice.ra, choice.dec)
        target_sep = _closest_approach_arcsec(xpos, track[0], track[1], track[2], track[3], span)
        margin = max(_member_sigma_arcsec(row) or 0.0, 0.5)
        best: tuple[float, Mapping[str, Any]] | None = None
        for comp in competitors:
            g_mag = _num(_ci_get(comp.get("data") or {}, "phot_g_mean_mag"))
            if target_g is not None and (g_mag is None or g_mag >= target_g):
                continue  # only a brighter neighbour than the target's own counterpart takes the flux
            data = comp.get("data") or {}
            pm = (_num(_ci_get(data, "pmra")), _num(_ci_get(data, "pmdec")))
            epoch = _num(_ci_get(data, "ref_epoch")) or _num(comp.get("epoch")) or 2016.0
            sep = _closest_approach_arcsec(xpos, float(comp["ra"]), float(comp["dec"]), epoch,
                                           None if None in pm else pm, span)  # type: ignore[arg-type]
            if sep <= choice.tolerance_arcsec and sep + margin < target_sep and (best is None or sep < best[0]):
                best = (sep, comp)
        if best is None:
            continue
        sep, comp = best
        g_mag = _num(_ci_get(comp.get("data") or {}, "phot_g_mean_mag"))
        when = (f" over the X-ray observations ({span[0]:.1f}-{span[1]:.1f})" if span and span[1] > span[0]
                else (f" at the X-ray epoch {span[0]:.1f}" if span else ""))
        choice.warnings.append(
            f"{choice.catalog} source {choice.source_id} comes within {sep:.1f} arcsec of Gaia DR3 {comp.get('source_id')}"
            + (f" (G = {g_mag:.2f})" if g_mag is not None else "")
            + f", another source {_fmt_sep(comp.get('separation_arcsec'))} arcsec from the target, but only within "
            f"{target_sep:.1f} arcsec of the target (positions propagated with their proper motions{when}): the X-ray "
            "flux is probably that source's; not attributed to the target")
        flagged.append(f"{choice.catalog} {choice.source_id}")
    return flagged


# Fitted major-axis FWHM columns (arcsec) of radio components: the centroid of an extended source (core +
# jets/lobes) can sit anywhere within about half its extent from the optical nucleus.
RADIO_SIZE_COLUMNS: dict[str, str] = {"first": "fit_major_axis", "nvss": "major_axis", "lotss": "Maj", "lotss_dr2": "Maj"}


def member_tolerance_arcsec(
    catalog: str, wavelength: str | None, positional_error_arcsec: float | None, major_axis_arcsec: float | None = None,
) -> float:
    """max(3 sigma (catalog 1-sigma error (+) 0.1 arcsec), catalog resolution floor, half the fitted radio extent)."""
    floor = MATCH_FLOOR_ARCSEC.get(catalog, WAVELENGTH_FLOOR_ARCSEC.get(str(wavelength or "").lower(), 2.0))
    sigma = math.hypot(positional_error_arcsec or 0.0, 0.1)
    extent = 0.5 * major_axis_arcsec if major_axis_arcsec and major_axis_arcsec > 0 else 0.0
    return max(3.0 * sigma, floor, extent)


# Lobe pairing (radio catalogs): two components on opposite sides of the optical position (offset vectors
# more than LOBE_MIN_ANGLE_DEG apart) at comparable distances (ratio <= LOBE_MAX_DISTANCE_RATIO) are taken
# as the two lobes of one double radio source (FR I/II morphology; e.g. Cygnus A's NVSS lobes at 44.5" and
# 52.1" on either side of the nucleus). This is a heuristic: the point notes say so.
LOBE_MIN_ANGLE_DEG = 150.0
LOBE_MAX_DISTANCE_RATIO = 2.0


def _offset_arcsec(ra: float, dec: float, ra0: float, dec0: float) -> tuple[float, float]:
    """Tangent-plane offset (east, north) in arcsec of (ra, dec) from (ra0, dec0) (small-angle)."""
    d_ra = ((ra - ra0 + 180.0) % 360.0) - 180.0
    return d_ra * math.cos(math.radians(dec0)) * 3600.0, (dec - dec0) * 3600.0


def _radio_flux_jy(catalog: str, data: Mapping[str, Any]) -> float | None:
    spec = RADIO_SPECS.get(catalog)
    if spec is None:
        return None
    band = spec[1][0]
    flux = _num(_ci_get(data, band.flux_col))
    return None if flux is None else flux * band.to_jy


def _component(member: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {"source_id": str(member.get("source_id")), "separation_arcsec": _num(member.get("separation_arcsec")),
            "ra": _num(member.get("ra")), "dec": _num(member.get("dec")), "role": role,
            "data": dict(member.get("data") or {})}


def _lobe_pair(
    candidates: Sequence[Mapping[str, Any]], centre: tuple[float, float] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float] | None:
    """The best pair of components on opposite sides of ``centre`` (angle, distance-ratio criteria)."""
    if centre is None:
        return None
    vecs = []
    for member in candidates:
        ra, dec = _num(member.get("ra")), _num(member.get("dec"))
        if ra is None or dec is None or _num(member.get("separation_arcsec")) is None:
            continue
        dx, dy = _offset_arcsec(ra, dec, *centre)
        dist = math.hypot(dx, dy)
        if dist > 0:
            vecs.append((member, dx, dy, dist))
    best: tuple[Mapping[str, Any], Mapping[str, Any], float] | None = None
    best_flux = -1.0
    for i, (a, ax, ay, ad) in enumerate(vecs):
        for b, bx, by, bd in vecs[i + 1:]:
            cos_angle = max(-1.0, min(1.0, (ax * bx + ay * by) / (ad * bd)))
            angle = math.degrees(math.acos(cos_angle))
            if angle < LOBE_MIN_ANGLE_DEG or max(ad, bd) / min(ad, bd) > LOBE_MAX_DISTANCE_RATIO:
                continue
            catalog = str(a.get("catalog"))
            flux = (_radio_flux_jy(catalog, a.get("data") or {}) or 0.0) + (_radio_flux_jy(catalog, b.get("data") or {}) or 0.0)
            if flux > best_flux:
                best, best_flux = (a, b, angle), flux
    return best


def _separation_key(member: Mapping[str, Any]) -> tuple[int, float]:
    """Sort key: nearest first, rows without a separation last (never treated as 0 arcsec)."""
    sep = _num(member.get("separation_arcsec"))
    return (1, 0.0) if sep is None else (0, sep)


def _fmt_sep(value: Any, digits: int = 1) -> str:
    sep = _num(value)
    return "unknown" if sep is None else f"{sep:.{digits}f}"


# NED preferred object types that describe a *part* of, or a feature along the sight line to, an object rather
# than the object itself (NED object type codes, https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes):
# absorption/emission line systems, HII regions, parts of galaxies, supernova remnants/supernovae/novae,
# star clusters/associations, and single-band detections (X-ray/radio/IR/UV/visual/gamma-ray sources) whose
# nature NED does not know. NED lists e.g. 18 co-spatial absorbers ('[HB89] 1226+023 ABS..', AbLS) for 3C 273.
NED_SUBCOMPONENT_TYPES = frozenset({
    "abls", "emls", "hii", "pofg", "snr", "sn", "nova", "*cl", "*ass", "xrays", "radios", "irs", "uvs", "viss",
    "gammas", "emobj", "neb", "mcld", "pn", "blend", "!hii", "!snr", "!pn", "!*cl", "!*ass", "!neb", "!mcld", "!emobj",
})
# Rows of these types never supply the object's redshift (an intervening absorber's z is not the object's z).
NED_NO_REDSHIFT_TYPES = frozenset({"abls"})


def _ned_type(member: Mapping[str, Any]) -> str:
    return str(_ci_get(member.get("data") or {}, "prefphytype") or "").strip().lower()


# NED names of sub-objects: '<object> ABSnn' (absorption systems, e.g. '3C 048 ABS01', which NED types 'QSO'),
# '<object>:[REF] id' (parts of an object, e.g. 'NGC 0253:[TH85] 6') and '<source> NEDnn' (components of a
# multi-component source, e.g. 'CRATES J0047-2517 NED02').
NED_ABSORBER_NAME = re.compile(r"\sABS\d+$", re.IGNORECASE)
NED_SUBOBJECT_NAME = re.compile(r"(:\[|\sNED\d+$)", re.IGNORECASE)


def ned_is_absorber(member: Mapping[str, Any]) -> bool:
    """An intervening absorption-line system: type AbLS or an '<object> ABSnn' name."""
    return _ned_type(member) in NED_NO_REDSHIFT_TYPES or bool(NED_ABSORBER_NAME.search(str(member.get("source_id") or "")))


def ned_is_subcomponent(member: Mapping[str, Any]) -> bool:
    name = str(member.get("source_id") or "")
    return (_ned_type(member) in NED_SUBCOMPONENT_TYPES or ned_is_absorber(member)
            or bool(NED_SUBOBJECT_NAME.search(name)))


def _choose_ned_row(members: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], list[str]]:
    """The NED row describing the object itself, among rows within the NED tolerance:

    1. object-level rows (type maps to galaxy/QSO/star, see :func:`ned_type_class`), nearest first -- but when
       the nearest has no redshift and a row of the same class *with* a NED redshift is also within tolerance,
       that row (NED's record of the object, e.g. 'NGC 0253' at 2.6 arcsec behind the G-typed X-ray source
       '2CXO J004733.1-251718' at 1.4 arcsec);
    2. untyped or other-type rows;
    3. sub-component rows: types in :data:`NED_SUBCOMPONENT_TYPES` or NED sub-object names (``<object> ABSnn``
       absorbers, ``<object>:[REF] id`` parts, ``<source> NEDnn`` components -- '3C 048 ABS01' is typed 'QSO').

    Returns the row and the nearer rows it passed over. NED often lists such rows at the object's position (3C 351:
    absorber '[HB89] 1704+608 ABS01' at 0.02 arcsec, quasar at 0.09 arcsec); taking the nearest row would report
    the absorber's redshift and type as the quasar's.
    """
    ordered = sorted(members, key=_separation_key)
    within = []
    for member in ordered:
        if _within_tolerance(member):
            within.append(member)
    if not within:
        return ordered[0], []

    def rank(member: Mapping[str, Any]) -> int:
        if ned_is_subcomponent(member):
            return 2
        return 0 if ned_type_class(_ned_type(member)) is not None else 1

    best = min(within, key=lambda m: (rank(m), _separation_key(m)))
    if rank(best) == 0 and _num(_ci_get(best.get("data") or {}, "z")) is None:
        cls = ned_type_class(_ned_type(best))
        with_z = [m for m in within if rank(m) == 0 and ned_type_class(_ned_type(m)) == cls
                  and _num(_ci_get(m.get("data") or {}, "z")) is not None]
        if with_z:
            best = with_z[0]
    skipped = []
    for member in within:
        if member is best:
            break
        skipped.append(f"{member.get('source_id')} ({_ned_type(member) or 'untyped'}, "
                       f"{_fmt_sep(member.get('separation_arcsec'), 3)} arcsec)")
    return best, skipped


def _choose_simbad_row(
    members: Sequence[Mapping[str, Any]], canonical_name: str | None,
) -> tuple[Mapping[str, Any], list[str]]:
    """The SIMBAD row of the object itself among rows within tolerance: the Sesame-resolved main identifier first,
    then non-planet rows (a planet row 'Pl'/'Pl?' shares its host's coordinates, e.g. '* 51 Peg b' at the position
    of '* 51 Peg', and must never replace the host's type, radial velocity and photometry), then the nearest.
    Returns the row and the co-located rows passed over."""
    ordered = sorted(members, key=_separation_key)
    within = [m for m in ordered if _within_tolerance(m)]
    if not within:
        return ordered[0], []
    target_name = " ".join(canonical_name.split()) if canonical_name else None

    def rank(member: Mapping[str, Any]) -> tuple[int, int]:
        name = " ".join(str(member.get("source_id") or "").split())
        return (0 if target_name is not None and name == target_name else 1,
                1 if simbad_is_planet(_ci_get(member.get("data") or {}, "otype")) else 0)

    best = min(within, key=lambda m: (rank(m), _separation_key(m)))
    passed = []
    for member in within:
        if member is best:
            continue
        otype = _ci_get(member.get("data") or {}, "otype")
        if _separation_key(member) <= _separation_key(best) or simbad_is_planet(otype):
            passed.append(f"{member.get('source_id')} ({otype or 'no type'}, "
                          f"{_fmt_sep(member.get('separation_arcsec'), 3)} arcsec)")
    return best, passed


# Survey designations (IAU-style names built from the catalogue identifier) under which SIMBAD, NED and Sesame list
# an object: '2MASS J04151954-0935066' is 2MASS PSC row 04151954-0935066, 'WISEA J041521.26-093500.4' is AllWISE row
# J041521.26-093500.4, 'Gaia DR3 4034171629042489088' is that Gaia DR3 source.
DESIGNATION_PREFIXES: dict[str, tuple[str, ...]] = {
    "twomass_psc": ("2MASS J", "2MASS "),
    "allwise": ("WISEA ",),
    "gaia_dr3": ("Gaia DR3 ",),
}


def _norm_name(name: Any) -> str:
    return " ".join(str(name or "").split()).upper()


def catalog_designations(catalog: str, source_id: Any) -> set[str]:
    """Normalised designations of a catalogue row (empty for catalogues without designations)."""
    sid = " ".join(str(source_id or "").split())
    if not sid:
        return set()
    return {_norm_name(prefix + sid) for prefix in DESIGNATION_PREFIXES.get(catalog, ())}


def _designated_row(
    members: Sequence[Mapping[str, Any]], catalog: str, names: Mapping[str, str],
) -> tuple[Mapping[str, Any], str] | None:
    """The row whose designation is a name of the object (nearest if several), with that name as given.

    ``names`` maps normalised names (:func:`_norm_name`) to their original spelling."""
    hits = []
    for member in members:
        if _num(member.get("separation_arcsec")) is None:
            continue
        match = sorted(catalog_designations(catalog, member.get("source_id")) & names.keys())
        if match:
            hits.append((member, names[match[0]]))
    return min(hits, key=lambda hit: _separation_key(hit[0])) if hits else None


def _choose_row(members: Sequence[Mapping[str, Any]], primary_group_id: str | None) -> Mapping[str, Any]:
    """Nearest row within tolerance, preferring the target group's row; the nearest row when none is within."""
    ordered = sorted(members, key=_separation_key)
    within = [m for m in ordered if _within_tolerance(m)]
    if not within:
        return ordered[0]
    in_primary = [m for m in within if primary_group_id is not None and m.get("group_id") == primary_group_id]
    return (in_primary or within)[0]


def select_members(
    group: Mapping[str, Any],
    centre: tuple[float, float] | None = None,
    *,
    primary_group_id: str | None = None,
    canonical_name: str | None = None,
    object_names: Iterable[str] = (),
) -> list[MemberChoice]:
    """Best row per catalog among ``group['members']``, flagged ``used`` when its separation is within tolerance.

    Rows may carry a ``group_id`` (their crossmatch group, see :func:`select_record_members`): among rows within
    tolerance the ``primary_group_id`` group's row is preferred, then the nearest. NED rows are ranked by
    :func:`_choose_ned_row` (object-level rows before absorbers/sub-components) and SIMBAD rows by
    :func:`_choose_simbad_row` (``canonical_name`` = Sesame's main identifier first, planets last). Other rows of the
    catalog that are within tolerance but not used are reported in the member notes.

    Identification by designation: a 2MASS/AllWISE/Gaia DR3 row whose designation (:data:`DESIGNATION_PREFIXES`) is
    one of ``object_names`` (the Sesame query, main identifier and aliases: the object the user asked for) IS the object
    and is used even beyond the positional tolerance. Catalogue positions of fast-moving stars can be offset by their
    proper motion times an epoch mismatch (SIMBAD lists the T8 dwarf '2MASSI J0415195-093506' at its 2MASS epoch-1998.9
    position labelled J2000: at 2.26 arcsec/yr its own 2MASS row comes out 2.5 arcsec away after epoch propagation).
    The identifier of the chosen SIMBAD/NED entry (itself a positional match) only decides between rows that are
    within tolerance.

    Radio catalogs (FIRST/NVSS/VLASS/LoTSS) collect *components*: every row within tolerance is summed, and
    a pair of components on opposite sides of ``centre`` (the target position) is accepted as a lobe pair
    even beyond the tolerance, so extended double sources (e.g. Cygnus A) keep their radio emission.
    Brighter unassociated radio components are reported in the member notes instead of silently dropped.
    """
    by_catalog: dict[str, list[Mapping[str, Any]]] = {}
    for member in group.get("members") or []:
        by_catalog.setdefault(str(member.get("catalog")), []).append(member)
    names = {_norm_name(n): " ".join(str(n).split()) for n in object_names if _norm_name(n)}
    entry_names: dict[str, str] = {}  # identifiers of the chosen SIMBAD/NED entries
    choices: list[MemberChoice] = []
    # SIMBAD and NED first: the identifiers of their chosen entries are names of the object (designations below).
    for catalog in sorted(by_catalog, key=lambda c: c not in RESOLVER_CATALOGS):
        members = sorted(by_catalog[catalog], key=_separation_key)  # rows without a separation go last
        skipped: list[str] = []
        designated: tuple[Mapping[str, Any], str] | None = None
        authoritative = False  # named by Sesame (the object asked for), not by a positionally matched entry
        if catalog in DESIGNATION_PREFIXES:
            designated = _designated_row(members, catalog, names)
            authoritative = designated is not None
            if designated is None:
                designated = _designated_row([m for m in members if _within_tolerance(m)], catalog, entry_names)
        if catalog == "ned":
            nearest, skipped = _choose_ned_row(members)
        elif catalog == "simbad":
            nearest, skipped = _choose_simbad_row(members, canonical_name)
        elif designated is not None:
            nearest = designated[0]
        else:
            nearest = _choose_row(members, primary_group_id)
        sep = _num(nearest.get("separation_arcsec"))
        wavelength = (nearest.get("metadata") or {}).get("wavelength")
        size_col = RADIO_SIZE_COLUMNS.get(catalog)
        major = _num(_ci_get(nearest.get("data") or {}, size_col)) if size_col else None
        tol = _row_tolerance(nearest)
        designation = designated[1] if designated is not None and nearest is designated[0] else None
        by_name = designation is not None
        # A designation beyond the tolerance can only be authoritative: entry names were matched within tolerance.
        used = sep is not None and (sep <= tol or by_name)
        raw_gid = nearest.get("group_id")
        gid = None if raw_gid is None else str(raw_gid)
        if sep is None:
            reason = f"nearest {catalog} row has no separation: not used"
        elif by_name:
            within = f"<= tolerance {tol:.2f} arcsec" if sep <= tol else (
                f"> tolerance {tol:.2f} arcsec (a positional offset, e.g. proper motion across mismatched catalogue "
                "epochs; the name identifies the row)")
            source = ("a name of the object resolved by Sesame" if authoritative
                      else "the name of the chosen SIMBAD/NED entry")
            reason = (f"{catalog} row identified by designation: '{designation}' is {source}; {sep:.3f} arcsec {within}")
        elif used:
            reason = f"nearest {catalog} source, {sep:.3f} arcsec <= tolerance {tol:.2f} arcsec"
        else:
            reason = f"nearest {catalog} source is {sep:.3f} arcsec away (> tolerance {tol:.2f} arcsec)"
        if catalog == "ned" and skipped:
            reason = reason.replace(f"nearest {catalog} source", "nearest NED object-level entry", 1)
            reason += f"; nearer NED row(s) skipped (sub-components, or without an object-level type): {', '.join(skipped)}"
        if catalog == "simbad" and skipped:
            reason = reason.replace(f"nearest {catalog} source", "SIMBAD entry of the object", 1)
            reason += f"; co-located SIMBAD row(s) passed over (planets, or not the resolved name): {', '.join(skipped)}"
        if major and 0.5 * major >= tol - 1e-9 and major > 0:
            reason += f" (extended radio source, fitted major axis {major:.1f} arcsec)"
        if used and primary_group_id is not None and gid not in (None, primary_group_id) and catalog not in RADIO_SPECS:
            reason += (f"; from crossmatch group {gid}, not the target group {primary_group_id} (the association "
                       "assigned it to another object, but " + ("its designation identifies it as the target)" if by_name
                                                                 else "it is within this catalog's tolerance of the target)"))
        choice = MemberChoice(
            catalog=catalog, source_id=str(nearest.get("source_id")), ra=_num(nearest.get("ra")),
            dec=_num(nearest.get("dec")), separation_arcsec=sep, tolerance_arcsec=tol, used=used,
            reason=reason, wavelength=wavelength, data=dict(nearest.get("data") or {}), group_id=gid,
            designation=designation,
        )
        if catalog in RESOLVER_CATALOGS and used and _norm_name(choice.source_id):
            entry_names.setdefault(_norm_name(choice.source_id), " ".join(choice.source_id.split()))
        if catalog in RADIO_SPECS:
            _collect_radio_components(choice, members, centre)
        else:
            if len(members) > 1:
                choice.reason += f"; {len(members) - 1} other {catalog} row(s) not used"
            if catalog not in {"ned", "simbad"}:
                for other in members:
                    if other is nearest or not _within_tolerance(other):
                        continue
                    where = f"crossmatch group {other.get('group_id')}, " if other.get("group_id") is not None else ""
                    chosen_where = ", target group" if gid is not None and gid == primary_group_id else ""
                    choice.notes.append(
                        f"{catalog} row {other.get('source_id')} ({where}{_fmt_sep(other.get('separation_arcsec'), 2)} "
                        f"arcsec) is also within tolerance but not used: {nearest.get('source_id')} "
                        f"({_fmt_sep(sep, 2)} arcsec{chosen_where}) was chosen")
        choices.append(choice)
    order = {name: i for i, name in enumerate(CATALOG_PRIORITY)}
    choices.sort(key=lambda c: (order.get(c.catalog, len(order)), c.catalog))
    return choices


def _collect_radio_components(
    choice: MemberChoice, members: Sequence[Mapping[str, Any]], centre: tuple[float, float] | None,
) -> None:
    """Fill ``choice.components`` (core + in-tolerance components + a lobe pair) for a radio catalog."""
    catalog = choice.catalog
    accepted: list[dict[str, Any]] = []
    rest: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    duplicates = 0
    for member in members:
        # Repeated measurements of one component (e.g. VLASS rows with the same Name from overlapping
        # images/epochs) must not be summed: keep the nearest row per component name.
        key = str(member.get("source_id")).strip()
        if key in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(key)
        raw_sep = _num(member.get("separation_arcsec"))
        if raw_sep is None:  # position unknown: never associated
            rest.append(member)
            continue
        sep = raw_sep
        tol = _row_tolerance(member)
        if sep <= tol:
            accepted.append(_component(member, "core" if not accepted else "component within tolerance"))
        else:
            rest.append(member)
    if accepted and not choice.used:  # a farther, larger component whose own extent covers the target
        choice.used = True
        choice.reason += f"; component {accepted[0]['source_id']} is within its own extent-based tolerance"
    pair = _lobe_pair(rest, centre)
    if pair is not None:
        a, b, angle = pair
        accepted.extend([_component(a, "lobe"), _component(b, "lobe")])
        rest = [m for m in rest if m is not a and m is not b]
        choice.used = True
        choice.reason += (f"; lobe pair {a.get('source_id')} ({_num(a.get('separation_arcsec')):.1f} arcsec) + "
                          f"{b.get('source_id')} ({_num(b.get('separation_arcsec')):.1f} arcsec) on opposite sides "
                          f"({angle:.0f} deg apart) accepted as one double radio source (heuristic)")
    choice.components = accepted
    if len(accepted) > 1:
        choice.reason += f"; {len(accepted)} components summed"
    accepted_flux = sum(_radio_flux_jy(catalog, c["data"]) or 0.0 for c in accepted)
    for member in rest:
        flux = _radio_flux_jy(catalog, member.get("data") or {})
        if flux is not None and flux > accepted_flux:
            choice.notes.append(
                f"{catalog} component {member.get('source_id')} ({flux:.4g} Jy, "
                f"{_fmt_sep(member.get('separation_arcsec'))} arcsec) is brighter than the associated radio "
                "flux but lies outside the association tolerance: not included (check images for extended emission)")
    if duplicates:
        choice.reason += f"; {duplicates} repeated row(s) of the same component ignored"
    if rest:
        choice.reason += f"; {len(rest)} other {catalog} component(s) not associated"


# ---------------------------------------------------------------------------
# Photometry Extraction
# ---------------------------------------------------------------------------


def _point_from_flux(
    *, band: str, facility: str, catalog: str, source_id: str, wavelength_um: float, flux_jy: float,
    flux_err_jy: float | None, upper: bool = False, **extra: Any,
) -> SEDPoint:
    freq = wavelength_um_to_hz(wavelength_um)
    nufnu = nu_fnu_cgs(freq, flux_jy)
    nufnu_err = None if flux_err_jy is None else nu_fnu_cgs(freq, flux_err_jy)
    return SEDPoint(
        band=band, facility=facility, catalog=catalog, source_id=source_id, wavelength_um=wavelength_um,
        frequency_hz=freq, flux_jy=flux_jy, flux_err_jy=None if upper else flux_err_jy,
        nu_fnu_erg_s_cm2=nufnu, is_upper_limit=upper, regime=regime_for_wavelength(wavelength_um),
        nu_fnu_err_erg_s_cm2=None if upper else nufnu_err, **extra,
    )


def magnitude_point(
    spec: MagBand, facility: str, catalog: str, source_id: str, data: Mapping[str, Any], filt: FilterInfo,
    *, extra_err_mag: float | None = None, flux_rel_err: float | None = None,
) -> SEDPoint | None:
    """Convert one magnitude column to an :class:`SEDPoint` (None when absent or a null sentinel)."""
    mag = _num(_ci_get(data, *spec.mag_cols))
    if mag is None or mag <= -90.0 or mag >= 90.0:  # PS1 -999 / SDSS -9999 sentinels
        return None
    mag_err = _num(_ci_get(data, *spec.err_cols)) if spec.err_cols else None
    if mag_err is not None and (mag_err < 0 or mag_err >= 90.0):
        mag_err = None
    notes: list[str] = []
    qual = None
    upper = False
    if spec.qual_col:
        qual_str = _ci_get(data, spec.qual_col)
        if isinstance(qual_str, str) and spec.qual_index is not None and len(qual_str) > spec.qual_index:
            qual = qual_str[spec.qual_index]
            if qual.upper() == "U":
                upper = True
                notes.append("ph_qual U: magnitude is a 95% confidence upper limit (flux upper limit)")
            elif qual.upper() in {"E", "F", "X"}:
                notes.append(f"ph_qual {qual}: poor photometric quality")
    if spec.kind == "sdss_asinh":
        flux, flux_err = sdss_asinh_mag_to_jy(mag, spec.band, mag_err)
        system, zp, zp_origin = "AB (SDSS asinh)", AB_ZERO_POINT_JY, "AB definition (3631 Jy); Lupton et al. 1999 asinh"
        if SDSS_AB_OFFSET[spec.band.lower()]:
            notes.append(f"SDSS->AB offset {SDSS_AB_OFFSET[spec.band.lower()]:+.2f} mag applied")
    else:
        flux, flux_err = pogson_mag_to_jy(mag, filt.zero_point_jy, mag_err)
        system, zp, zp_origin = filt.mag_system, filt.zero_point_jy, filt.zero_point_origin
    if flux <= 0:
        if spec.kind != "sdss_asinh":
            return None
        # Asinh magnitude beyond the zero-flux magnitude: a non-detection. 3-sigma upper limit from the
        # measured flux and its error, or from b (~1-sigma sky noise in a PSF aperture, Lupton et al. 1999).
        b = SDSS_ASINH_B[spec.band.lower()]
        scale = AB_ZERO_POINT_JY * 10.0 ** (-0.4 * SDSS_AB_OFFSET[spec.band.lower()])
        limit = max(flux, 0.0) + 3.0 * flux_err if flux_err else 3.0 * b * scale
        notes.append(f"asinh magnitude {mag:.2f} implies flux {flux:.3g} Jy <= 0 (non-detection): "
                     f"3-sigma upper limit {'from the magnitude error' if flux_err else 'from the softening b (sky noise)'}")
        return _point_from_flux(
            band=spec.band, facility=facility, catalog=catalog, source_id=source_id,
            wavelength_um=filt.wavelength_eff_um, flux_jy=limit, flux_err_jy=None, upper=True,
            filter_id=filt.filter_id, magnitude=mag, magnitude_error=mag_err, mag_system=system,
            zero_point_jy=zp, zero_point_origin=zp_origin, quality_flag=qual, notes=notes,
        )
    if flux_rel_err is not None:
        flux_err = flux * flux_rel_err
    if extra_err_mag is not None and flux_err is not None:
        flux_err = math.hypot(flux_err, flux * extra_err_mag / POGSON)
        mag_err = POGSON * flux_err / flux
    elif flux_rel_err is not None:
        mag_err = POGSON * flux_rel_err
    if flux_err is None and not upper:
        notes.append("no magnitude error available")
    point = _point_from_flux(
        band=spec.band, facility=facility, catalog=catalog, source_id=source_id,
        wavelength_um=filt.wavelength_eff_um, flux_jy=flux, flux_err_jy=flux_err, upper=upper,
        filter_id=filt.filter_id, magnitude=mag, magnitude_error=mag_err, mag_system=system,
        zero_point_jy=zp, zero_point_origin=zp_origin, quality_flag=qual, notes=notes,
    )
    if qual is not None and qual.upper() in {"E", "F", "X"}:
        point.warn(f"ph_qual {qual}: poor photometric quality")
    return point


def radio_point(spec: RadioBand, facility: str, catalog: str, source_id: str, data: Mapping[str, Any]) -> SEDPoint | None:
    flux = _num(_ci_get(data, spec.flux_col))
    if flux is None or flux <= 0:
        return None
    err = _num(_ci_get(data, spec.err_col)) if spec.err_col else None
    wavelength = hz_to_wavelength_um(spec.frequency_hz)
    return _point_from_flux(
        band=spec.band, facility=facility, catalog=catalog, source_id=source_id, wavelength_um=wavelength,
        flux_jy=flux * spec.to_jy, flux_err_jy=None if err is None or err < 0 else err * spec.to_jy,
        notes=[spec.note],
    )


def radio_member_point(spec: RadioBand, facility: str, member: MemberChoice) -> SEDPoint | None:
    """Radio point of a member: the sum of its associated components (errors added in quadrature)."""
    components = member.components or [{"source_id": member.source_id, "data": member.data, "role": "core",
                                        "separation_arcsec": member.separation_arcsec}]
    parts: list[tuple[dict[str, Any], SEDPoint]] = []
    for c in components:
        p = radio_point(spec, facility, member.catalog, c["source_id"], c["data"])
        if p is not None:
            parts.append((c, p))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][1]
    total = sum(p.flux_jy for _, p in parts)
    errs = [p.flux_err_jy for _, p in parts if p.flux_err_jy is not None]
    err = math.sqrt(sum(e * e for e in errs)) if len(errs) == len(parts) else None
    detail = ", ".join(f"{c['source_id']} ({c['role']}, {p.flux_jy:.4g} Jy)" for c, p in parts)
    return _point_from_flux(
        band=spec.band, facility=facility, catalog=member.catalog, source_id="+".join(c["source_id"] for c, _ in parts),
        wavelength_um=hz_to_wavelength_um(spec.frequency_hz), flux_jy=total, flux_err_jy=err,
        notes=[spec.note, f"sum of {len(parts)} components: {detail}"],
    )


def xray_point(spec: XrayBand, facility: str, catalog: str, source_id: str, data: Mapping[str, Any]) -> SEDPoint | None:
    notes = [spec.note, f"flux density at E0=sqrt(E1*E2) assuming photon index {XRAY_PHOTON_INDEX:g}"]
    if spec.rate_col:
        rate = _num(_ci_get(data, spec.rate_col))
        if rate is None or rate <= 0 or spec.ecf is None:
            return None
        flux_cgs = rate * spec.ecf
        rate_err = _num(_ci_get(data, spec.rate_err_col)) if spec.rate_err_col else None
        err_cgs = None if rate_err is None else rate_err * spec.ecf
    else:
        measured = _num(_ci_get(data, spec.flux_col)) if spec.flux_col else None
        if measured is None or measured <= 0:
            return None
        flux_cgs = measured
        err_cgs = _num(_ci_get(data, spec.err_col)) if spec.err_col else None
        if err_cgs is None and spec.lo_col and spec.hi_col:
            lo, hi = _num(_ci_get(data, spec.lo_col)), _num(_ci_get(data, spec.hi_col))
            if lo is not None and hi is not None and hi >= lo:
                err_cgs = 0.5 * (hi - lo)  # half the 68% interval
    flux_jy, freq = xray_band_flux_to_jy(flux_cgs, spec.e_lo_kev, spec.e_hi_kev)
    err_jy = None if err_cgs is None else flux_jy * err_cgs / flux_cgs
    point = _point_from_flux(
        band=spec.band, facility=facility, catalog=catalog, source_id=source_id,
        wavelength_um=hz_to_wavelength_um(freq), flux_jy=flux_jy, flux_err_jy=err_jy, notes=notes,
    )
    point.regime = "xray"
    return point


def required_filters(members: Sequence[MemberChoice]) -> set[str]:
    ids: set[str] = set()
    for member in members:
        if not member.used:
            continue
        if member.catalog in MAG_SPECS:
            ids.update(spec.filter_id for spec in MAG_SPECS[member.catalog][1])
        if member.catalog == "simbad":
            ids.update(fid for _, fid, _ in SIMBAD_BANDS)
        if any(_ci_get(member.data, *spec.mag_cols) is not None for spec in GALEX_BANDS):
            ids.update(spec.filter_id for spec in GALEX_BANDS)
    return ids


def gaia_corrected_excess(excess_factor: float, bp_rp: float) -> float:
    """C* = C - f(BP - RP) (Riello et al. 2021, Eq. 6 with the Table 2 polynomials)."""
    for upper, coeffs in GAIA_EXCESS_POLY:
        if bp_rp < upper:
            return excess_factor - sum(a * bp_rp**i for i, a in enumerate(coeffs))
    raise AssertionError("unreachable")


def gaia_excess_sigma(g_mag: float) -> float:
    """1-sigma scatter of C* for well-behaved isolated stars, sigma_C*(G) = c0 + c1 G^m (Riello et al. 2021, Eq. 18)."""
    c0, c1, m = GAIA_EXCESS_SIGMA
    return c0 + c1 * g_mag**m


def gaia_corrected_excess_of(data: Mapping[str, Any], gaia_extra: Mapping[str, Any] | None) -> float | None:
    excess = _num((gaia_extra or {}).get("phot_bp_rp_excess_factor"))
    bp, rp = _num(_ci_get(data, "phot_bp_mean_mag")), _num(_ci_get(data, "phot_rp_mean_mag"))
    if excess is None or bp is None or rp is None:
        return None
    return gaia_corrected_excess(excess, bp - rp)


def gaia_bp_rp_excess_flag(data: Mapping[str, Any], gaia_extra: Mapping[str, Any] | None) -> str | None:
    """A note when |C*| > 3 sigma_C*(G): BP/RP fluxes contaminated (extended source, blend, bright neighbour)."""
    excess = _num((gaia_extra or {}).get("phot_bp_rp_excess_factor"))
    g, bp, rp = (_num(_ci_get(data, c)) for c in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
    if excess is None or g is None or bp is None or rp is None:
        return None
    c_star = gaia_corrected_excess(excess, bp - rp)
    sigma = gaia_excess_sigma(g)
    if abs(c_star) > GAIA_EXCESS_NSIGMA * sigma:
        return (f"Gaia BP/RP flux excess C* = {c_star:.3f} exceeds {GAIA_EXCESS_NSIGMA:g} sigma_C*(G) = "
                f"{GAIA_EXCESS_NSIGMA * sigma:.3f} (Riello et al. 2021): BP/RP contaminated (extended source or blend)")
    return None


def gaia_predicted_v(g_mag: float, bp_rp: float) -> float | None:
    """Johnson V from Gaia G and BP-RP via G - V = f(BP-RP) (Riello et al. 2021, Table C.2; -0.5 < BP-RP < 5)."""
    if not -0.5 < bp_rp < 5.0:
        return None
    return g_mag - sum(a * bp_rp**i for i, a in enumerate(GAIA_G_MINUS_V))


def sdss_saturation(data: Mapping[str, Any]) -> str | None:
    """Why the SDSS photometry of a row is saturated (None when it is not known to be).

    Per-band SATURATED flags (``sat_<band>`` from :func:`fetch_sdss_object`) when available, else any psfMag
    brighter than :data:`SDSS_SATURATION_MAG` (Riello et al. 2021 App. C: SDSS sources brighter than 14 mag are
    saturated).
    """
    flags = {b: _num(_ci_get(data, f"sat_{b}")) for b in "ugriz"}
    flagged = [b for b, v in flags.items() if v is not None and v > 0]
    if flagged:
        return f"SDSS SATURATED flag set in {', '.join(flagged)}"
    if any(v is not None for v in flags.values()):
        return None
    bright = [(b, m) for b in "ugriz" if (m := _num(_ci_get(data, f"psfMag_{b}"))) is not None and -90 < m < SDSS_SATURATION_MAG]
    if bright:
        band, mag = min(bright, key=lambda item: item[1])
        return f"SDSS psfMag_{band} = {mag:.2f} < {SDSS_SATURATION_MAG:g} (saturated; per-band flags unavailable)"
    return None


def sdss_morphology(data: Mapping[str, Any]) -> tuple[int | None, str]:
    """SDSS photometric type (3 = galaxy, 6 = star) when usable, else (None, why).

    SDSS separates stars from galaxies by psfMag - cmodelMag; saturation truncates the PSF core, so saturated POINT
    sources come out 'extended' (51 Peg, V = 5.5, and CW Leo are type 3). A saturated type 3 is therefore not used;
    a saturated type 6 is kept (saturation biases towards type 3, not away from it; heuristic, stated in the text).
    """
    obj_type = _num(_ci_get(data, "type"))
    if obj_type is None:
        return None, "no SDSS type"
    saturated = sdss_saturation(data)
    if saturated and int(obj_type) != 6:
        return None, f"SDSS type {int(obj_type)} not used: {saturated} (saturated point sources are typed extended)"
    if saturated:
        return 6, f"SDSS type 6 ({saturated}; saturation biases towards type 3, so a point-source type still holds)"
    return int(obj_type), f"SDSS type {int(obj_type)}"


def sdss_aperture(data: Mapping[str, Any]) -> str:
    """'model' for SDSS galaxies (unsaturated type 3) when modelMag AND modelMagErr exist in all five bands, else 'psf'.

    One aperture for all bands: mixing modelMag (total light) in some bands with psfMag in others creates
    spurious spikes of 10-80x in galaxy SEDs.
    """
    if sdss_morphology(data)[0] != 3:
        return "psf"
    for band in "ugriz":
        if _num(_ci_get(data, f"modelMag_{band}")) is None or _num(_ci_get(data, f"modelMagErr_{band}")) is None:
            return "psf"
    return "model"


def _sdss_spec_for(spec: MagBand, aperture: str) -> MagBand:
    if aperture == "model":
        return MagBand(spec.band, spec.filter_id, (f"modelMag_{spec.band}",), (f"modelMagErr_{spec.band}",), kind="sdss_asinh")
    return spec


def _sdss_quality(point: SEDPoint, data: Mapping[str, Any], aperture: str, has_flags: bool) -> None:
    band = point.band
    if aperture == "model":
        point.notes.append("SDSS modelMag (galaxy model, total light) in all bands; PSF photometry of other surveys "
                           "measures less of an extended source")
    elif _num(_ci_get(data, "type")) == 3:
        point.notes.append("SDSS psfMag of an extended source (modelMag not available in all bands): "
                           "underestimates the total flux")
    sat = _num(_ci_get(data, f"sat_{band}"))
    if sat is not None and sat > 0:
        point.warn(f"SDSS SATURATED flag set in {band}")
    clean = _num(_ci_get(data, "clean"))
    if clean == 0:
        point.notes.append("SDSS clean = 0 (photometry flags set)")
    if not has_flags and point.magnitude is not None and point.magnitude < SDSS_SATURATION_MAG and aperture == "psf":
        point.warn(f"SDSS {band} = {point.magnitude:.2f} < {SDSS_SATURATION_MAG:g}: likely saturated "
                   "(per-band flags unavailable; Riello et al. 2021 App. C)")


def _ps1_quality(point: SEDPoint, data: Mapping[str, Any]) -> None:
    limit = PS1_SATURATION_MAG.get(point.band)
    if limit is not None and point.magnitude is not None and point.magnitude < limit:
        point.warn(f"PS1 {point.band} = {point.magnitude:.2f} brighter than the single-exposure saturation limit "
                   f"{limit:g} (Magnier et al. 2013)")
    flag = _num(_ci_get(data, "qualityFlag"))
    if flag is not None and int(flag) & PS1_QF_OBJ_EXT:
        point.notes.append("PS1 QF_OBJ_EXT: extended source; the PSF magnitude underestimates its total flux")
    if flag is not None:
        bits = int(flag)
        if bits & PS1_QF_OBJ_BAD_STACK or not bits & PS1_QF_OBJ_GOOD:
            point.warn(f"PS1 qualityFlag = {bits}: QF_OBJ_GOOD not set"
                       + (" (QF_OBJ_BAD_STACK)" if bits & PS1_QF_OBJ_BAD_STACK else "")
                       + (" (QF_OBJ_SUSPECT_STACK)" if bits & PS1_QF_OBJ_SUSPECT_STACK else ""))


def ecliptic_longitude_deg(ra: float, dec: float) -> float:
    """Ecliptic longitude (deg, [0, 360)) of an ICRS position (astropy BarycentricMeanEcliptic, J2000 equinox)."""
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(BarycentricMeanEcliptic())
    return float(coord.lon.wrap_at(360 * u.deg).deg)


def wise_postcryo_longitude(ra: float | None, dec: float | None) -> bool | None:
    """True when the position lies in the ecliptic-longitude ranges observed during the NEOWISE post-cryo phase
    (AllWISE ES II.2.c.i); None when the position is unknown."""
    if ra is None or dec is None:
        return None
    lon = ecliptic_longitude_deg(ra, dec)
    return any(lo < lon < hi for lo, hi in WISE_POSTCRYO_ECLIPTIC_LON_DEG)


def _wise_quality(point: SEDPoint, ra: float | None = None, dec: float | None = None) -> None:
    """Flag unreliable AllWISE profile-fit photometry (see :data:`WISE_PROFILE_FIT_LIMIT_MAG`).

    * measurements brighter than the profile-fit limits (All-Sky ES VI.3.d): quality warning;
    * W1 < 8 / W2 < 7 in the post-cryo ecliptic-longitude ranges (AllWISE ES II.2.c.i): quality warning; outside them,
      or when the position is unknown, a note only (profile fitting of the unsaturated wings is reliable there);
    * 'U' limits brighter than the nominal saturation limits: quality warning (a failed extraction, not a limit).
    """
    mag = point.magnitude
    if mag is None:
        return
    band = point.band
    if point.is_upper_limit:
        limit = WISE_SATURATION_MAG.get(band)
        if limit is not None and mag < limit:
            point.warn(f"WISE {band} 'U' upper limit {mag:.2f} brighter than the nominal saturation limit {limit:g}: a "
                       "failed extraction of a saturated source, not a meaningful flux limit")
        return
    fit_limit = WISE_PROFILE_FIT_LIMIT_MAG.get(band)
    if fit_limit is not None and mag < fit_limit:
        point.warn(f"WISE {band} profile-fit magnitude {mag:.2f} brighter than {fit_limit:g}, the limit to which profile "
                   "fitting of the unsaturated wings is reliable (WISE All-Sky Explanatory Supplement VI.3.d): unreliable")
        return
    caveat = WISE_POSTCRYO_BANDS.get(band)
    if caveat is None or mag >= caveat:
        return
    postcryo = wise_postcryo_longitude(ra, dec)
    text = (f"WISE {band} = {mag:.2f} < {caveat:g} (saturated core, profile fit of the unsaturated wings)")
    if postcryo:
        point.warn(f"{text} at an ecliptic longitude observed in the NEOWISE post-cryo phase: AllWISE fluxes of such "
                   "sources 'have much larger uncertainties and exhibit more scatter' (AllWISE ES II.2.c.i)")
    elif postcryo is None:
        point.notes.append(f"{text}; position unknown, so the AllWISE post-cryo caveat (II.2.c.i) could not be checked")
    else:
        point.notes.append(f"{text}; outside the post-cryo ecliptic longitudes, reliable (All-Sky ES VI.3.d)")


def _wise_rayleigh_jeans_consistency(points: Sequence[SEDPoint]) -> None:
    """Flag WISE points (measurements or limits) far below the Rayleigh-Jeans extrapolation of shorter bands."""
    for point in sorted(points, key=lambda p: p.wavelength_um):  # shorter bands are settled first
        if point.facility != "WISE":
            continue
        shorter = [r for r in points if r is not point and _usable(r) and r.facility in {"2MASS", "WISE"}
                   and 1.0 < r.wavelength_um < point.wavelength_um]
        if not shorter:
            continue
        ref = max(shorter, key=lambda r: r.wavelength_um)
        bound = ref.flux_jy * (ref.wavelength_um / point.wavelength_um) ** 2
        if point.flux_jy * WISE_RJ_MARGIN < bound:
            point.warn(f"WISE {point.band} {'limit' if point.is_upper_limit else 'flux'} {point.flux_jy:.3g} Jy is "
                       f"{bound / point.flux_jy:.0f}x below the Rayleigh-Jeans extrapolation (F_nu ~ lambda^-2) of "
                       f"{ref.facility} {ref.band} ({ref.flux_jy:.3g} Jy): saturated or spurious, not a valid measurement")


def _lotss_consistency(points: Sequence[SEDPoint]) -> None:
    """Flag LoTSS 144 MHz points that imply a rising spectrum up to 1.4 GHz (missing flux or calibration)."""
    lotss = next((p for p in points if p.catalog in {"lotss", "lotss_dr2"}), None)
    ref = next((p for p in points if p.catalog == "nvss"), None) or next((p for p in points if p.catalog == "first"), None)
    if lotss is None:
        return
    if lotss.flux_jy > 1.0:
        lotss.notes.append("very bright LoTSS component (> 1 Jy): dynamic range and direction-dependent calibration "
                           "limit LoTSS accuracy near bright sources; compare with TGSS ADR1 (Intema et al. 2017)")
    if ref is None:
        return
    alpha = math.log10(ref.flux_jy / lotss.flux_jy) / math.log10(ref.frequency_hz / lotss.frequency_hz)
    if alpha > 0.5:
        lotss.warn(f"spectral index 144 MHz -> 1.4 GHz ({ref.facility}) alpha = {alpha:+.2f} > +0.5 (S ~ nu^alpha): "
                   "LoTSS flux probably underestimated (missing flux or calibration), unless the source is "
                   "synchrotron self-absorbed")


def extract_points(
    members: Sequence[MemberChoice],
    filters: FilterCatalog,
    gaia_extra: Mapping[str, Any] | None = None,
    *,
    sdss_extra: Mapping[str, Any] | None = None,
    simbad_extra: Mapping[str, Any] | None = None,
) -> list[SEDPoint]:
    """Every photometric measurement in the accepted members, converted to flux density.

    ``sdss_extra`` (from :func:`fetch_sdss_object`) supplies psf/model magnitudes, errors and per-band
    SATURATED flags for all five SDSS bands; ``simbad_extra`` (from :func:`fetch_simbad_extra`) supplies
    SIMBAD flux errors and bibcodes. Both are optional: without them the member columns are used.
    """
    points: list[SEDPoint] = []
    seen: set[tuple[str, str]] = set()

    def add(point: SEDPoint | None) -> SEDPoint | None:
        if point is None:
            return None
        key = (point.facility, point.band)
        if key in seen:  # e.g. 2MASS from both IRSA and VizieR: keep the higher-priority catalog
            return None
        seen.add(key)
        points.append(point)
        return point

    for member in members:
        if not member.used:
            continue
        data, cat, sid = member.data, member.catalog, member.source_id
        if cat == "sdss" and sdss_extra:
            data = {**data, **{k: v for k, v in sdss_extra.items() if k != "spectra"}}
        start = len(points)
        if cat in MAG_SPECS:
            facility, specs = MAG_SPECS[cat]
            aperture = sdss_aperture(data) if cat == "sdss" else ""
            excess_note = gaia_bp_rp_excess_flag(data, gaia_extra) if cat == "gaia_dr3" else None
            for spec in specs:
                if cat == "sdss":
                    spec = _sdss_spec_for(spec, aperture)
                kwargs: dict[str, Any] = {}
                if cat == "gaia_dr3" and gaia_extra:
                    foe = _num(gaia_extra.get(f"phot_{spec.band.lower()}_mean_flux_over_error"))
                    if foe and foe > 0:
                        kwargs = {"flux_rel_err": 1.0 / foe, "extra_err_mag": GAIA_ZP_SIGMA_MAG[spec.band]}
                point = magnitude_point(spec, facility, cat, sid, data, filters.info(spec.filter_id), **kwargs)
                if point is None:
                    continue
                if cat == "gaia_dr3":
                    point.notes.append("error from Gaia flux_over_error plus zero-point uncertainty (Riello et al. 2021)"
                                       if kwargs else "Gaia flux errors unavailable")
                    if excess_note and spec.band in {"BP", "RP"}:
                        point.warn(excess_note)
                elif cat == "sdss":
                    _sdss_quality(point, data, aperture, has_flags=bool(sdss_extra))
                elif cat == "panstarrs_dr2":
                    _ps1_quality(point, data)
                elif cat == "allwise":
                    _wise_quality(point, member.ra, member.dec)
                add(point)
        if cat in RADIO_SPECS:
            facility, rspecs = RADIO_SPECS[cat]
            for rspec in rspecs:
                add(radio_member_point(rspec, facility, member))
        if cat in XRAY_SPECS:
            facility, xspecs = XRAY_SPECS[cat]
            for xspec in xspecs:
                add(xray_point(xspec, facility, cat, sid, data))
        for column, predicate, note, warn in QUALITY_NOTES.get(cat, []):
            value = _ci_get(data, column)
            if value is not None and predicate(value):
                for point in points[start:]:
                    if warn:
                        point.warn(note)
                    else:
                        point.notes.append(note)
        for note in member.warnings:
            for point in points[start:]:
                point.warn(note)
        for gspec in GALEX_BANDS:
            if _ci_get(data, *gspec.mag_cols) is not None:
                add(magnitude_point(gspec, "GALEX", cat, sid, data, filters.info(gspec.filter_id)))

    # SIMBAD literature magnitudes only fill bands no survey measured.
    simbad = next((m for m in members if m.used and m.catalog == "simbad"), None)
    if simbad is not None:
        covered = {p.filter_id for p in points}
        fluxes = (simbad_extra or {}).get("fluxes") or {}
        for band, fid, col in SIMBAD_BANDS:
            if fid in covered:
                continue
            info = fluxes.get(col) or {}
            mag = _num(info.get("flux")) if info else _num(_ci_get(simbad.data, col))
            err = _num(info.get("flux_err"))
            spec = MagBand(band, fid, ("mag",), ("mag_err",))
            point = magnitude_point(spec, "SIMBAD (literature)", "simbad", simbad.source_id,
                                    {"mag": mag, "mag_err": err if err and err > 0 else None}, filters.info(fid))
            if point is None:
                continue
            ref = f" ({info['bibcode']})" if info.get("bibcode") else ""
            point.notes.append(f"SIMBAD compiled literature magnitude{ref}")
            if band == "K":
                point.notes.append("SIMBAD K treated as 2MASS Ks")
            if band == "V":
                _check_simbad_v(point, members, gaia_extra, points)
            add(point)
    _lotss_consistency(points)
    points.sort(key=lambda p: p.wavelength_um)
    _wise_rayleigh_jeans_consistency(points)
    return points


def _check_simbad_v(
    point: SEDPoint, members: Sequence[MemberChoice], gaia_extra: Mapping[str, Any] | None, points: Sequence[SEDPoint],
) -> None:
    """Flag a SIMBAD V inconsistent with the measured photometry (point-like sources only).

    Reference: V predicted from Gaia G and BP-RP (Riello et al. 2021, Table C.2) when |C*| < 0.1 (colour usable), else the
    log-log interpolation of the unflagged neighbouring survey bands at the V wavelength.
    """
    extended, why = optical_extent(members)
    if extended:
        point.notes.append(f"integrated magnitude of an extended source ({why}): not comparable to PSF photometry")
        return
    if point.magnitude is None:
        return
    gaia = _member(members, "gaia_dr3")
    if gaia is not None:
        g, bp, rp = (_num(_ci_get(gaia.data, c)) for c in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
        c_star = gaia_corrected_excess_of(gaia.data, gaia_extra)
        if g is not None and bp is not None and rp is not None and (c_star is None or abs(c_star) < GAIA_EXCESS_COLOUR_LIMIT):
            v_pred = gaia_predicted_v(g, bp - rp)
            if v_pred is not None:
                delta = point.magnitude - v_pred
                if abs(delta) > SIMBAD_V_TOLERANCE_MAG:
                    point.warn(f"SIMBAD V = {point.magnitude:.2f} differs by {delta:+.2f} mag from V = {v_pred:.2f} predicted "
                               "from Gaia DR3 G and BP-RP (Riello et al. 2021, Table C.2): erroneous, different epoch, or variable")
                return
    ref = interpolated_flux(points, point.wavelength_um)
    if ref is not None:
        delta = -2.5 * math.log10(point.flux_jy / ref[0])
        if abs(delta) > SIMBAD_V_TOLERANCE_MAG:
            point.warn(f"SIMBAD V = {point.magnitude:.2f} differs by {delta:+.2f} mag from the {ref[1]}: erroneous, "
                       "different epoch, or variable")


# ---------------------------------------------------------------------------
# Supplementary Queries (Gaia DR3 astrometric errors/DSC, SDSS SpecObj / Photoz)
# ---------------------------------------------------------------------------

GAIA_EXTRA_COLUMNS = (
    "source_id", "pmra_error", "pmdec_error", "pmra_pmdec_corr", "parallax_over_error", "ruwe",
    "phot_g_mean_flux_over_error", "phot_bp_mean_flux_over_error", "phot_rp_mean_flux_over_error",
    "classprob_dsc_combmod_quasar", "classprob_dsc_combmod_galaxy", "classprob_dsc_combmod_star",
    "in_qso_candidates", "in_galaxy_candidates", "phot_bp_rp_excess_factor",
    "astrometric_excess_noise", "astrometric_excess_noise_sig",
)
# gaia_source.classprob_dsc_combmod_star is "Probability from DSC-Combmod of being a single star (but not a white
# dwarf)"; the white-dwarf and binary-star classes of the five-class DSC-Combmod output are only in
# gaiadr3.astrophysical_parameters (column descriptions from the Gaia archive TAP_SCHEMA, 2026-09-28). AD Leo
# (Gaia DR3 625453654702751872): star = 1.0e-6, binarystar = 0.999999.
GAIA_AP_COLUMNS = ("classprob_dsc_combmod_whitedwarf", "classprob_dsc_combmod_binarystar")


def gaia_extra_adql(source_id: str) -> str:
    """gaia_source columns of one source, LEFT JOINed with its DSC white-dwarf/binary probabilities."""
    if not re.fullmatch(r"\d{1,20}", str(source_id)):
        raise ValueError(f"invalid Gaia source_id {source_id!r}")
    cols = [f"g.{c}" for c in GAIA_EXTRA_COLUMNS] + [f"ap.{c}" for c in GAIA_AP_COLUMNS]
    return (f"SELECT {', '.join(cols)} FROM gaiadr3.gaia_source AS g "
            "LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id "
            f"WHERE g.source_id = {source_id}")


def retry_after_seconds(response: httpx.Response) -> float | None:
    """The Retry-After header of an answer in seconds (delta-seconds or an HTTP date, RFC 9110 10.2.3), or None."""
    raw = str(response.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    number = _num(raw)
    if number is not None:
        return max(0.0, number)
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


# Transient failures of one TAP query are retried (the policy of providers.py for the crossmatch catalogs): 3
# attempts, 0.25 s and 0.5 s apart (a Retry-After of at most TAP_RETRY_AFTER_MAX_SECONDS is honoured instead), for
# dropped/refused connections and HTTP 429/5xx. Timeouts are not retried: they already used the request's time and
# put the host in back-off (:func:`backoff_seconds_for`). Everything stays within the supplementary deadline.
TAP_RETRY_DELAYS = (0.25, 0.5)
TAP_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TAP_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)
TAP_RETRY_AFTER_MAX_SECONDS = 5.0


async def _tap_query(client: httpx.AsyncClient, url: str, adql: str, *, timeout: float, label: str) -> httpx.Response:
    """POST one synchronous ADQL query (JSON output), retrying transient failures (:data:`TAP_RETRY_DELAYS`).

    Raises the last transport error, or :class:`SEDUpstreamError` with ``status_code``/``retry_after`` for an HTTP
    error answer."""
    data = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}
    response: httpx.Response | None = None
    for delay in (*TAP_RETRY_DELAYS, None):
        try:
            response = await client.post(url, data=data, timeout=timeout)
        except TAP_RETRY_EXCEPTIONS:
            if delay is None:
                raise
            await asyncio.sleep(delay)
            continue
        if response.status_code not in TAP_RETRY_STATUS or delay is None:
            break
        wait = retry_after_seconds(response)
        await asyncio.sleep(delay if wait is None else min(wait, TAP_RETRY_AFTER_MAX_SECONDS))
    assert response is not None  # the last attempt either answered or raised
    if response.status_code >= 400:
        wait = retry_after_seconds(response)
        raise SEDUpstreamError(f"{label} HTTP {response.status_code}"
                               + (f" (Retry-After {wait:.0f} s)" if wait is not None else ""),
                               status_code=response.status_code, retry_after=wait)
    return response


async def fetch_gaia_extra(
    client: httpx.AsyncClient, source_id: str, *, timeout: float = SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Proper-motion errors, flux S/N and DSC class probabilities (all five classes) for one Gaia DR3 source."""
    response = await _tap_query(client, GAIA_TAP_URL, gaia_extra_adql(source_id), timeout=timeout, label="Gaia TAP")
    rows = parse_json_table(response.content).rows
    return dict(rows[0]) if rows else None


def sdss_specobj_sql(ra: float, dec: float, radius_arcsec: float) -> str:
    return (
        "SELECT TOP 5 CAST(n.specObjID AS VARCHAR(24)) AS specObjID, n.distance*60.0 AS dist_arcsec, "
        "s.ra, s.dec, s.z, s.zErr, s.zWarning, s.class, s.subClass, s.survey, s.sciencePrimary "
        f"FROM dbo.fGetNearbySpecObjEq({ra:.9f}, {dec:.9f}, {radius_arcsec / 60.0:.10f}) AS n "
        "JOIN SpecObj AS s ON s.specObjID = n.specObjID ORDER BY s.sciencePrimary DESC, n.distance"
    )


def sdss_photoz_sql(ra: float, dec: float, radius_arcsec: float) -> str:
    return (
        "SELECT TOP 3 CAST(n.objID AS VARCHAR(24)) AS objID, n.distance*60.0 AS dist_arcsec, "
        "pz.z, pz.zErr, pz.photoErrorClass, pz.nnCount "
        f"FROM dbo.fGetNearbyObjEq({ra:.9f}, {dec:.9f}, {radius_arcsec / 60.0:.10f}) AS n "
        "JOIN PhotoPrimary AS p ON p.objID = n.objID JOIN Photoz AS pz ON pz.objID = n.objID ORDER BY n.distance"
    )


# ---------------------------------------------------------------------------
# Supplementary Lookup Budget (deadline + per-host back-off)
# ---------------------------------------------------------------------------

def default_supplementary_deadline() -> float:
    """Overall budget (s) of the supplementary lookups of one SED: $ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS,
    else the app's REQUEST_TIMEOUT_SECONDS (models.Settings, default 30 s), which also caps each crossmatch catalog."""
    raw = _num(os.getenv("ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS"))
    if raw is not None and raw > 0:
        return raw
    try:
        from models import Settings

        return float(Settings().request_timeout_seconds)
    except (ValueError, TypeError):  # invalid environment: keep the documented default
        return 30.0


class HostBackoff:
    """Per-host back-off after a costly failure of a supplementary lookup (see :func:`backoff_seconds_for`).

    A hanging or refusing archive would otherwise cost every later SED request the deadline or the connection
    retries: timeouts back off for ``retry_after_seconds``, refused connections (the host is down or refusing; the
    lookup already retried, see :func:`_sdss_rows`) and any failure slower than
    :data:`SUPPLEMENTARY_SLOW_FAILURE_SECONDS` for ``refused_retry_after_seconds``. Fast failures (an HTTP error
    answered at once) cost nothing and are retried on the next request. Hosts are keyed by hostname; the clock is
    ``time.monotonic``.
    """

    def __init__(self, retry_after_seconds: float = SUPPLEMENTARY_BACKOFF_SECONDS,
                 refused_retry_after_seconds: float = SUPPLEMENTARY_REFUSED_BACKOFF_SECONDS) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.refused_retry_after_seconds = refused_retry_after_seconds
        self._until: dict[str, tuple[float, str]] = {}

    def pending(self, host: str | None) -> str | None:
        """Why ``host`` is skipped (the failure and the remaining back-off), or None when it may be queried."""
        if not host or host not in self._until:
            return None
        until, why = self._until[host]
        remaining = until - time.monotonic()
        if remaining <= 0:
            del self._until[host]
            return None
        return f"{why}; back-off, retried in {remaining:.0f} s"

    def record_failure(self, host: str | None, why: str, retry_after_seconds: float | None = None) -> None:
        if host:
            wait = self.retry_after_seconds if retry_after_seconds is None else retry_after_seconds
            self._until[host] = (time.monotonic() + wait, why)

    def record_success(self, host: str | None) -> None:
        if host:
            self._until.pop(host, None)

    def clear(self) -> None:
        self._until.clear()


_DEFAULT_BACKOFF = HostBackoff()


def default_backoff() -> HostBackoff:
    """Process-wide back-off shared by all SED requests."""
    return _DEFAULT_BACKOFF


def _host(url: str) -> str | None:
    from urllib.parse import urlsplit

    return urlsplit(url).hostname


def _is_slow_failure(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TimeoutException | TimeoutError)


def backoff_seconds_for(backoff: HostBackoff, exc: BaseException, elapsed: float) -> float | None:
    """How long a failed lookup's host is skipped: timeouts :attr:`HostBackoff.retry_after_seconds`; a rate limit
    (HTTP 429, or any HTTP error answer carrying Retry-After, e.g. 503 during maintenance) the Retry-After time capped
    at :data:`SUPPLEMENTARY_MAX_RETRY_AFTER_SECONDS` (:attr:`HostBackoff.refused_retry_after_seconds` for a 429
    without it); refused connections (``httpx.ConnectError``, after the lookup's own retries) and failures slower than
    :data:`SUPPLEMENTARY_SLOW_FAILURE_SECONDS` :attr:`HostBackoff.refused_retry_after_seconds`; None (no back-off)
    for other fast failures."""
    if isinstance(exc, SEDUpstreamError) and exc.status_code is not None:
        if exc.retry_after is not None:
            return min(exc.retry_after, SUPPLEMENTARY_MAX_RETRY_AFTER_SECONDS)
        if exc.status_code == 429:
            return backoff.refused_retry_after_seconds
    if _is_slow_failure(exc):
        return backoff.retry_after_seconds
    if isinstance(exc, httpx.ConnectError) or elapsed > SUPPLEMENTARY_SLOW_FAILURE_SECONDS:
        return backoff.refused_retry_after_seconds
    return None


# SkyServer often refuses a second simultaneous connection ("All connection attempts failed"): SDSS queries of one
# SED run are serialised (``lock``) and a refused connection (nothing was sent) is retried after a short pause.
SDSS_CONNECT_RETRY_DELAYS = (1.0, 3.0)


class SkyServerLock(asyncio.Lock):
    """Serialises the SkyServer queries of one SED run and remembers a refusal that outlasted the retries: later
    queries of the same run then fail at once instead of paying the retries again."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: str | None = None


async def _sdss_rows(
    client: httpx.AsyncClient, sql: str, timeout: float, lock: asyncio.Lock | None = None,
) -> list[dict[str, Any]]:
    async with lock or contextlib.nullcontext():
        refused = getattr(lock, "refused", None)
        if refused:
            raise httpx.ConnectError(f"not retried in this run: {refused}")
        for delay in (*SDSS_CONNECT_RETRY_DELAYS, None):
            try:
                response = await client.get(SDSS_SQL_URL, params={"cmd": sql, "format": "json"}, timeout=timeout)
                break
            except httpx.ConnectError as exc:
                if delay is None:
                    if isinstance(lock, SkyServerLock):
                        lock.refused = (f"{type(exc).__name__} after {len(SDSS_CONNECT_RETRY_DELAYS) + 1} attempts"
                                        + (f": {exc}" if str(exc) else ""))
                    raise
                await asyncio.sleep(delay)
            except (httpx.ReadError, httpx.RemoteProtocolError):
                # A dropped connection (the read-only query is safe to repeat); not a refusal of the host.
                if delay is None:
                    raise
                await asyncio.sleep(delay)
    if response.status_code >= 400:
        detail = response.text[:200].replace("\n", " ")
        raise SEDUpstreamError(f"SDSS SkyServer HTTP {response.status_code}: {detail}")
    payload = response.json()
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "Rows" in payload[0]:
        return [dict(r) for r in payload[0]["Rows"]]
    raise SEDUpstreamError("Unexpected SDSS SkyServer JSON layout")


def sdss_object_sql(obj_id: str) -> str:
    """All-band photometry, per-band SATURATED flags, Photoz and SpecObj rows (zWarning) of one SDSS object."""
    if not re.fullmatch(r"\d{1,20}", str(obj_id)):
        raise ValueError(f"invalid SDSS objID {obj_id!r}")
    cols = ["CAST(p.objID AS VARCHAR(24)) AS objID", "p.type", "p.clean"]
    cols += [f"p.{kind}_{b}" for kind in ("psfMag", "psfMagErr", "modelMag", "modelMagErr") for b in "ugriz"]
    cols += [f"CASE WHEN (p.flags_{b} & dbo.fPhotoFlags('SATURATED')) != 0 THEN 1 ELSE 0 END AS sat_{b}" for b in "ugriz"]
    cols += ["pz.z AS photoz", "pz.zErr AS photozErr", "pz.photoErrorClass",
             "CAST(s.specObjID AS VARCHAR(24)) AS specObjID", "s.z", "s.zErr", "s.zWarning", "s.class", "s.sciencePrimary"]
    return (f"SELECT TOP 10 {', '.join(cols)} FROM PhotoObj AS p "
            "LEFT OUTER JOIN Photoz AS pz ON pz.objID = p.objID "
            "LEFT OUTER JOIN SpecObj AS s ON s.bestObjID = p.objID "
            f"WHERE p.objID = {obj_id} ORDER BY s.sciencePrimary DESC")


async def fetch_sdss_object(
    client: httpx.AsyncClient, obj_id: str, *, timeout: float = SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS,
    lock: asyncio.Lock | None = None,
) -> dict[str, Any] | None:
    """Photometry + flags of the SDSS member (the registry row has psfMagErr_r and modelMag_r only).

    Returns the photometric columns plus ``spectra``: [{specObjID, z, zErr, zWarning, class, sciencePrimary}].
    """
    rows = await _sdss_rows(client, sdss_object_sql(obj_id), timeout, lock)
    if not rows:
        return None
    spec_keys = ("specObjID", "z", "zErr", "zWarning", "class", "sciencePrimary")
    out = {k: v for k, v in rows[0].items() if k not in spec_keys}
    out["spectra"] = [{k: r.get(k) for k in spec_keys} for r in rows if _num(r.get("z")) is not None]
    return out


def simbad_extra_adql(main_id: str) -> str:
    """Redshift nature/quality and V/G/K flux errors of one SIMBAD object (quotes escaped for ADQL)."""
    escaped = str(main_id).replace("'", "''")
    return ("SELECT b.main_id, b.rvz_type, b.rvz_redshift, b.rvz_err, b.rvz_nature, b.rvz_qual, b.rvz_bibcode, "
            "f.filter, f.flux, f.flux_err, f.qual AS flux_qual, f.bibcode AS flux_bibcode "
            "FROM basic AS b LEFT OUTER JOIN flux AS f ON f.oidref = b.oid AND f.filter IN ('V', 'G', 'K') "
            f"WHERE b.main_id = '{escaped}'")


async def fetch_simbad_extra(
    client: httpx.AsyncClient, main_id: str, *, timeout: float = SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """SIMBAD ``rvz_nature`` ('s'/'se'/'sa' spectroscopic, 'p' photometric), ``rvz_qual`` (A-E) and flux errors."""
    response = await _tap_query(client, SIMBAD_TAP_URL, simbad_extra_adql(main_id), timeout=timeout,
                                label="SIMBAD TAP")
    rows = parse_json_table(response.content).rows
    if not rows:
        return None
    first = dict(rows[0])
    out = {k: first.get(k) for k in ("main_id", "rvz_type", "rvz_redshift", "rvz_err", "rvz_nature", "rvz_qual", "rvz_bibcode")}
    out["fluxes"] = {
        str(r.get("filter")): {"flux": _num(r.get("flux")), "flux_err": _num(r.get("flux_err")),
                               "qual": r.get("flux_qual"), "bibcode": r.get("flux_bibcode")}
        for r in rows if r.get("filter")
    }
    return out


def _zwarning(value: Any) -> int | None:
    """SDSS SpecObj zWarning bitmask, or None when null (never defaulted to 0 = 'no problems')."""
    number = _num(value)
    return None if number is None else int(number)


async def fetch_sdss_redshifts(
    client: httpx.AsyncClient, ra: float, dec: float, *, radius_arcsec: float = 2.0,
    include_photoz: bool = True, timeout: float = SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS,
    lock: asyncio.Lock | None = None, notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Redshift candidates from SDSS DR18 SpecObj (spectroscopic) and Photoz (Beck et al. 2016).

    The queries run one after the other (SkyServer refuses simultaneous connections, see :func:`_sdss_rows`). A
    failed SpecObj query fails the lookup; a failed Photoz query after it does not: the SpecObj candidates (the most
    reliable redshifts, and the zWarning = 0 corroboration of the cross-check) are returned and the Photoz failure is
    appended to ``notes`` (logged when no list is given).
    """
    results = [await _sdss_rows(client, sdss_specobj_sql(ra, dec, radius_arcsec), timeout, lock)]
    if include_photoz:
        started = time.monotonic()
        try:
            results.append(await _sdss_rows(client, sdss_photoz_sql(ra, dec, radius_arcsec), timeout, lock))
        except (httpx.HTTPError, SEDError, ValueError) as exc:  # ValueError: undecodable JSON answer
            detail = str(exc).strip()
            message = (f"SDSS Photoz query failed ({type(exc).__name__} after {time.monotonic() - started:.1f} s"
                       + (f": {detail}" if detail else "") + "); SpecObj redshifts kept, no SDSS photo-z")
            if notes is None:
                logger.warning("%s", message)
            else:
                notes.append(message)
    candidates: list[dict[str, Any]] = []
    for row in results[0]:
        z = _num(row.get("z"))
        if z is None:
            continue
        warning = _zwarning(row.get("zWarning"))
        candidates.append({
            "value": z, "error": _num(row.get("zErr")), "kind": "spec", "source": "sdss_specobj",
            "reliable": None if warning is None else warning == 0, "spec_class": (str(row.get("class") or "").strip() or None),
            "sub_class": (str(row.get("subClass") or "").strip() or None), "z_warning": warning,
            "id": str(row.get("specObjID")), "separation_arcsec": _num(row.get("dist_arcsec")),
            "survey": row.get("survey"),
        })
    if include_photoz and len(results) > 1:
        for row in results[1]:
            z = _num(row.get("z"))
            if z is None or z < -1.0:  # -9999 = no estimate
                continue
            candidates.append({
                "value": z, "error": _num(row.get("zErr")), "kind": "photo", "source": "sdss_photoz",
                "reliable": _num(row.get("photoErrorClass")) == 1, "id": str(row.get("objID")),
                "photo_error_class": _num(row.get("photoErrorClass")), "separation_arcsec": _num(row.get("dist_arcsec")),
            })
    return candidates


# ---------------------------------------------------------------------------
# Redshift
# ---------------------------------------------------------------------------


def ned_redshift_kind(zflag: Any) -> str | None:
    """NED objdir zflag: first letter S = spectroscopic, P = photometric (e.g. PUN for Richards et al. 2009
    photometric quasars); other codes (U, M, I) do not state the method, so the kind is unknown (None)."""
    text = str(zflag or "").strip().upper()
    if text.startswith("S"):
        return "spec"
    if text.startswith("P"):
        return "photo"
    return None


def simbad_redshift_kind(nature: Any, rvz_type: Any = None) -> str | None:
    """SIMBAD basic.rvz_nature: 's', 'se', 'sa' = spectroscopic; 'p' = photometric. SIMBAD leaves rvz_nature null
    for radial velocities (rvz_type 'v', e.g. Sirius, Proxima Cen, Sco X-1): a radial velocity is measured from
    spectral lines, so it is spectroscopic. Otherwise a missing nature is unknown (None)."""
    text = str(nature or "").strip().lower()
    if text.startswith("s"):
        return "spec"
    if text.startswith("p"):
        return "photo"
    if not text and str(rvz_type or "").strip().lower() == "v":
        return "spec"
    return None


SIMBAD_RELIABLE_QUALITIES = frozenset({"A", "B", "C", "D"})  # rvz_qual E: unreliable


def simbad_redshift_reliable(quality: Any) -> bool | None:
    """SIMBAD rvz_qual A-D = reliable, E = not reliable, missing = unknown (None); whatever rvz_nature says."""
    text = str(quality or "").strip().upper()
    if not text:
        return None
    return text in SIMBAD_RELIABLE_QUALITIES


def _same_z(a: float | None, b: float | None, tol: float = 1e-5) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def member_redshift_candidates(
    members: Sequence[MemberChoice],
    *,
    sdss_extra: Mapping[str, Any] | None = None,
    simbad_extra: Mapping[str, Any] | None = None,
    specobj_candidates: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Redshifts carried by the members, with their kind and reliability never assumed.

    * NED: kind from zflag; NED's own values are taken as reliable.
    * SIMBAD: kind from rvz_nature (a radial velocity with a null nature is spectroscopic, see
      :func:`simbad_redshift_kind`) and reliability from rvz_qual whatever the nature (A-D reliable, E not, see
      :func:`simbad_redshift_reliable`) via ``simbad_extra``; without it the kind and reliability are unknown (None).
    * SDSS member specz (registry join on bestObjID, no zWarning filter): reliable only when the same spectrum
      has zWarning = 0 in ``sdss_extra`` spectra or in the SpecObj cone query; otherwise unknown (None).
    * SDSS member photoz: reliable when photoErrorClass = 1 (``sdss_extra``), otherwise unknown.
    """
    candidates: list[dict[str, Any]] = []
    for member in members:
        if not member.used:
            continue
        data = member.data
        if member.catalog == "ned":
            z = _num(_ci_get(data, "z"))
            if z is not None and ned_is_absorber({"source_id": member.source_id, "data": data}):
                continue  # an intervening absorption-line system's redshift is not the object's
            if z is not None:
                candidates.append({"value": z, "error": _num(_ci_get(data, "zunc")), "kind": ned_redshift_kind(_ci_get(data, "zflag")),
                                   "source": "ned", "reliable": True, "id": member.source_id,
                                   "flag": _ci_get(data, "zflag")})
        elif member.catalog == "simbad":
            z = _num(_ci_get(data, "rvz_redshift"))
            if z is None:
                continue
            cand: dict[str, Any] = {"value": z, "error": None, "kind": None, "source": "simbad", "reliable": None,
                                    "id": member.source_id}
            if simbad_extra:
                qual = str(simbad_extra.get("rvz_qual") or "").strip().upper() or None
                kind = simbad_redshift_kind(simbad_extra.get("rvz_nature"), simbad_extra.get("rvz_type"))
                err = _num(simbad_extra.get("rvz_err"))
                cand.update({
                    "kind": kind, "nature": simbad_extra.get("rvz_nature"), "quality": qual,
                    "bibcode": simbad_extra.get("rvz_bibcode"), "rvz_type": simbad_extra.get("rvz_type"),
                    "error": (err if str(simbad_extra.get("rvz_type") or "").lower() == "z" else
                              (err / C_KMS if err is not None else None)),
                    "reliable": simbad_redshift_reliable(qual),
                })
            candidates.append(cand)
        elif member.catalog == "sdss":
            spectra = list((sdss_extra or {}).get("spectra") or [])
            spec_z = _num(_ci_get(data, "specz"))
            if spec_z is not None:
                cand = {"value": spec_z, "error": _num(_ci_get(data, "speczErr")), "kind": "spec", "source": "sdss",
                        "reliable": None, "id": member.source_id,
                        "spec_class": (str(_ci_get(data, "specClass") or "").strip() or None)}
                match = next((s for s in spectra if _same_z(_num(s.get("z")), spec_z)), None)
                if match is not None:
                    warning = _zwarning(match.get("zWarning"))
                    cand.update({"reliable": None if warning is None else warning == 0, "z_warning": warning,
                                 "spec_obj_id": match.get("specObjID")})
                else:
                    other = next((c for c in specobj_candidates if _same_z(_num(c.get("value")), spec_z)), None)
                    if other is not None:
                        cand.update({"reliable": other.get("reliable"), "z_warning": other.get("z_warning"),
                                     "spec_obj_id": other.get("id")})
                if cand["reliable"] is None:
                    cand["note"] = "zWarning of this spectrum unknown (SpecObj lookup unavailable or zWarning null)"
                candidates.append(cand)
            photo_z = _num((sdss_extra or {}).get("photoz")) if sdss_extra else None
            if photo_z is None:
                photo_z = _num(_ci_get(data, "photoz"))
            if photo_z is not None and photo_z > -1.0:
                pec = _num((sdss_extra or {}).get("photoErrorClass"))
                photo_err = _num((sdss_extra or {}).get("photozErr")) if sdss_extra else None
                candidates.append({"value": photo_z, "error": photo_err if photo_err is not None else _num(_ci_get(data, "photozErr")),
                                   "kind": "photo",
                                   "source": "sdss_photoz", "reliable": None if pec is None else pec == 1,
                                   "photo_error_class": pec, "id": member.source_id})
    return candidates


# (source, kind, reliable) in order of preference. Reliable spectra > unverified spectra > catalogue-vetted values
# of unstated technique > reliable photo-z > SIMBAD values of unknown technique and quality (its TAP lookup failed) >
# flagged spectra > unverified photo-z > the rest. NED values whose zflag does not state the technique (e.g. 'UUN',
# which NED gives for M87, NGC 4151, NGC 4472, Cygnus A) keep kind=None but are NED's vetted preferred redshifts, and
# SIMBAD values of null rvz_nature with quality A-D are vetted too: a photometric redshift never outranks them (SDSS
# photo-z scatter is ~0.02-0.03 in (1+z) even for photoErrorClass = 1, Beck et al. 2016, so a photo-z of 0.134 must
# not replace the 0.155 of Hercules A that NED and SIMBAD agree on). A SIMBAD value whose quality could not be
# fetched is not known to be bad (51 Peg's radial velocity), so it outranks every value its catalogue flags (SIMBAD
# quality E, SDSS zWarning != 0, photoErrorClass != 1) and photo-z of unknown quality.
_REDSHIFT_PRIORITY: tuple[tuple[str, str | None, bool | None], ...] = (
    ("sdss_specobj", "spec", True),
    ("sdss", "spec", True),
    ("ned", "spec", True),
    ("simbad", "spec", True),
    ("sdss_specobj", "spec", None),
    ("sdss", "spec", None),
    ("simbad", "spec", None),
    ("ned", None, True),
    ("simbad", None, True),
    ("ned", "photo", True),
    ("sdss_photoz", "photo", True),
    ("simbad", "photo", True),
    ("simbad", None, None),
    ("sdss_specobj", "spec", False),
    ("sdss", "spec", False),
    ("simbad", "spec", False),
    ("sdss_photoz", "photo", None),
    ("simbad", "photo", None),
    ("sdss_photoz", "photo", False),
    ("simbad", "photo", False),
    ("simbad", None, False),
)


# Two vetted redshifts of one object (catalogue rounding, cz <-> z conversions: |dz|/(1+z) ~ 1e-4-1e-3) agree within
# this fraction; a larger spread means at least one catalogue describes another object or is wrong (NED lists the
# z = 0.36 quasar PKS 1510-089 at z = 0.0068).
REDSHIFT_DISCORD_TOLERANCE = 0.01


def _redshift_rank(cand: Mapping[str, Any]) -> int | None:
    for index, (source, kind, reliable) in enumerate(_REDSHIFT_PRIORITY):
        if cand.get("source") == source and cand.get("kind") == kind and cand.get("reliable") is reliable:
            return index
    return None


def _redshift_family(source: Any) -> str:
    """Independent origin of a redshift: the SDSS member specz and the SpecObj row are the same spectrum."""
    return "sdss" if source in {"sdss", "sdss_specobj"} else str(source)


def _vetted_redshift(cand: Mapping[str, Any]) -> bool:
    """A value its catalogue vouches for: reliable, and spectroscopic or of unstated technique (not a photo-z)."""
    return cand.get("reliable") is True and cand.get("kind") != "photo" and _num(cand.get("value")) is not None


def _checkable_redshift(cand: Mapping[str, Any]) -> bool:
    """A value that takes part in the cross-check: not a photo-z and not flagged by its catalogue, i.e. vetted
    (:func:`_vetted_redshift`) or of unknown quality (``reliable`` None: a SIMBAD value whose TAP lookup failed or was
    skipped, an SDSS spectrum whose zWarning could not be fetched). A value that could not be vetted still disagrees:
    the SIMBAD row of PKS 1510-089 says z = 0.356 whether or not its quality could be fetched."""
    return cand.get("reliable") is not False and cand.get("kind") != "photo" and _num(cand.get("value")) is not None


def _redshift_result(cand: Mapping[str, Any] | None, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if cand is None:
        return {"value": None, "error": None, "kind": None, "source": None, "reliable": None,
                "candidates": [dict(c) for c in candidates]}
    return {"value": cand["value"], "error": cand.get("error"), "kind": cand.get("kind"), "source": cand.get("source"),
            "reliable": cand.get("reliable"), "candidates": [dict(c) for c in candidates]}


def _redshifts_agree(z: float, z0: float) -> bool:
    """|z - z0| / (1 + z0) <= :data:`REDSHIFT_DISCORD_TOLERANCE` (z0: the reference, best-ranked value)."""
    return abs(z - z0) / (1.0 + max(z0, 0.0)) <= REDSHIFT_DISCORD_TOLERANCE


def redshift_clusters(candidates: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Cross-checked candidates (:func:`_checkable_redshift`: vetted or of unknown quality) grouped by value
    (:func:`_redshifts_agree` with the group's best-ranked value), each group and the groups in priority order."""
    checkable = sorted((c for c in candidates if _checkable_redshift(c) and _redshift_rank(c) is not None),
                       key=lambda c: _redshift_rank(c))  # type: ignore[arg-type,return-value]
    clusters: list[list[Mapping[str, Any]]] = []
    for cand in checkable:
        z = float(cand["value"])
        for cluster in clusters:
            if _redshifts_agree(z, float(cluster[0]["value"])):
                cluster.append(cand)
                break
        else:
            clusters.append([cand])
    return clusters


def _cluster_vetted(cluster: Sequence[Mapping[str, Any]]) -> bool:
    """A group holding at least one vetted value: only such a group can be adopted from a discordance."""
    return any(_vetted_redshift(c) for c in cluster)


def _cluster_corroboration(cluster: Sequence[Mapping[str, Any]]) -> str | None:
    """Independent support of a group of agreeing redshifts holding a vetted value: two catalogues (the other one
    may be of unknown quality), or an SDSS spectrum with zWarning = 0. None for a group without a vetted value."""
    if not _cluster_vetted(cluster):
        return None
    families = sorted({_redshift_family(c.get("source")) for c in cluster})
    if len(families) >= 2:
        return f"agreement of {' and '.join(families)}"
    if any(_redshift_family(c.get("source")) == "sdss" and c.get("z_warning") == 0 for c in cluster):
        return "an SDSS spectrum with zWarning = 0"
    return None


def _describe_cluster(cluster: Sequence[Mapping[str, Any]]) -> str:
    return f"z = {float(cluster[0]['value']):.5g} (" + ", ".join(
        f"{c.get('source')}" + (f" {c.get('id')}" if c.get("id") else "")
        + (", quality unknown" if c.get("reliable") is None else "") for c in cluster) + ")"


def _group_candidate(group: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The best-ranked vetted candidate of a discordant group (an entry of ``redshift['discordant']``)."""
    z0 = float(group["value"])
    members = [c for c in candidates if _vetted_redshift(c) and _redshift_rank(c) is not None
               and str(c.get("source")) in group["sources"] and _redshifts_agree(float(c["value"]), z0)]
    return min(members, key=lambda c: _redshift_rank(c)) if members else None  # type: ignore[arg-type,return-value]


def choose_redshift(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Best redshift by :data:`_REDSHIFT_PRIORITY`: reliable spectra (SDSS zWarning = 0 > NED 'S' > SIMBAD 's*'
    with quality A-D) > unverified spectra > NED/SIMBAD vetted values of unstated technique > reliable photo-z >
    SIMBAD values of unknown quality > flagged spectra > unverified photo-z > the rest.

    Every non-photometric value not flagged by its catalogue -- vetted, or of unknown quality because its lookup
    failed (:func:`_checkable_redshift`) -- is cross-checked (:func:`redshift_clusters`). When they disagree (|dz|/(1+z)
    > :data:`REDSHIFT_DISCORD_TOLERANCE`), ``discordant`` lists every group (``vetted`` False: no value of the group
    could be vetted) and the priority order alone never decides: the one group holding a vetted value that is
    corroborated by a second catalogue or an SDSS zWarning = 0 spectrum (:func:`_cluster_corroboration`) is adopted
    (its best vetted value), so an unverified spectrum never outranks a corroborated value; otherwise
    ``needs_resolution`` is set and :func:`resolve_discordant_redshift` checks each value against the
    classification, or reports the conflict. Photo-z and flagged values never make a discordance.
    """
    ranked = sorted((c for c in candidates if _num(c.get("value")) is not None and _redshift_rank(c) is not None),
                    key=lambda c: _redshift_rank(c))  # type: ignore[arg-type,return-value]
    result = _redshift_result(ranked[0] if ranked else None, candidates)
    clusters = redshift_clusters(candidates)
    if len(clusters) < 2:
        return result
    support = [_cluster_corroboration(cluster) for cluster in clusters]
    result["discordant"] = [
        {"value": float(cluster[0]["value"]), "sources": [str(c.get("source")) for c in cluster],
         "ids": [c.get("id") for c in cluster], "corroboration": why, "vetted": _cluster_vetted(cluster)}
        for cluster, why in zip(clusters, support, strict=True)
    ]
    listing = "; ".join(_describe_cluster(cluster) for cluster in clusters)
    head = ("discordant redshifts" if all(_cluster_vetted(c) for c in clusters) else
            "discordant redshifts, some of unknown quality (not vetted: their catalogue lookup failed or was skipped)")
    corroborated = [i for i, why in enumerate(support) if why is not None]
    if len(corroborated) == 1:
        chosen = clusters[corroborated[0]]
        best = next(c for c in chosen if _vetted_redshift(c))
        result.update(_redshift_result(best, candidates))
        others = "; ".join(_describe_cluster(c) for i, c in enumerate(clusters) if i != corroborated[0])
        result["resolution"] = (f"{head} ({listing}): z = {float(best['value']):.5g} ({best.get('source')}) adopted, "
                                f"corroborated by {support[corroborated[0]]}; not corroborated: {others}")
    else:
        result["needs_resolution"] = True
        result["resolution"] = (f"{head} ({listing}): "
                                + ("each is corroborated" if corroborated else "none is corroborated by a second "
                                   "catalogue or an SDSS zWarning = 0 spectrum"))
    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# SIMBAD object-type hierarchy: otype code -> (path, is_candidate), copied verbatim from the live SIMBAD TAP table
# ``otypedef`` (SELECT otype, label, path, description, is_candidate FROM otypedef), retrieved 2026-09-28 (226 codes;
# 'var' has no path). Classes are derived from the path, never from the look of the code: 'Sy?' is a *Symbiotic Star
# Candidate* ('* > ** > Sy*'), not a Seyfert; 'SN*' sits under '*' but is a transient, often extragalactic.
SIMBAD_OTYPEDEF: dict[str, tuple[str | None, bool]] = {
    '?': ('', False), '*': ('*', False), '**': ('* > **', False), '**?': ('* > **', True), 'As*': ('As*', False),
    'As?': ('As*', True), 'MGr': ('As* > MGr', False), 'St*': ('As* > St*', False), 'BY*': ('* > ** > BY*', False),
    'BY?': ('* > ** > BY*', True), 'Cl*': ('Cl*', False), 'Cl?': ('Cl*', True), 'ClG': ('ClG', False),
    'C?G': ('ClG', True), 'GlC': ('Cl* > GlC', False), 'Gl?': ('Cl* > GlC', True), 'OpC': ('Cl* > OpC', False),
    'CV*': ('* > ** > CV*', False), 'CV?': ('* > ** > CV*', True), 'No*': ('* > ** > CV* > No*', False),
    'No?': ('* > ** > CV* > No*', True), 'EB*': ('* > ** > EB*', False), 'EB?': ('* > ** > EB*', True),
    'El?': ('* > ** > El*', True), 'El*': ('* > ** > El*', False), 'Em*': ('* > Em*', False), 'err': ('err', False),
    'ev': ('ev', False), 'Ev?': ('* > Ev*', True), 'Ev*': ('* > Ev*', False), 'AB?': ('* > Ev* > AB*', True),
    'AB*': ('* > Ev* > AB*', False), 'Mi?': ('* > Ev* > AB* > Mi*', True), 'Mi*': ('* > Ev* > AB* > Mi*', False),
    'C*?': ('* > Ev* > C*', True), 'C*': ('* > Ev* > C*', False), 'Ce*': ('* > Ev* > Ce*', False),
    'Ce?': ('* > Ev* > Ce*', True), 'cC*': ('* > Ev* > Ce* > cC*', False), 'HB*': ('* > Ev* > HB*', False),
    'HB?': ('* > Ev* > HB*', True), 'RR*': ('* > Ev* > HB* > RR*', False), 'RR?': ('* > Ev* > HB* > RR*', True),
    'HS*': ('* > Ev* > HS*', False), 'HS?': ('* > Ev* > HS*', True), 'LP*': ('* > Ev* > LP*', False),
    'LP?': ('* > Ev* > LP*', True), 'OH?': ('* > Ev* > OH*', True), 'OH*': ('* > Ev* > OH*', False),
    'pA*': ('* > Ev* > pA*', False), 'pA?': ('* > Ev* > pA*', True), 'PN': ('* > Ev* > PN', False),
    'PN?': ('* > Ev* > PN', True), 'RG*': ('* > Ev* > RG*', False), 'RB?': ('* > Ev* > RG*', True),
    'RV?': ('* > Ev* > RV*', True), 'RV*': ('* > Ev* > RV*', False), 'S*?': ('* > Ev* > S*', True),
    'S*': ('* > Ev* > S*', False), 'WD?': ('* > Ev* > WD*', True), 'WD*': ('* > Ev* > WD*', False),
    'WV?': ('* > Ev* > WV*', True), 'WV*': ('* > Ev* > WV*', False), 'G?': ('G', True), 'G': ('G', False),
    'AG?': ('G > AGN', True), 'AGN': ('G > AGN', False), 'LIN': ('G > AGN > LIN', False),
    'Q?': ('G > AGN > QSO', True), 'QSO': ('G > AGN > QSO', False), 'Bz?': ('G > AGN > QSO > Bla', True),
    'Bla': ('G > AGN > QSO > Bla', False), 'BLL': ('G > AGN > QSO > Bla > BLL', False),
    'BL?': ('G > AGN > QSO > Bla > BLL', True), 'rG': ('G > AGN > rG', False), 'SyG': ('G > AGN > SyG', False),
    'Sy1': ('G > AGN > SyG > Sy1', False), 'Sy2': ('G > AGN > SyG > Sy2', False), 'gam': ('gam', False),
    'gB': ('gam > gB', False), 'bCG': ('G > bCG', False), 'EmG': ('G > EmG', False), 'GiC': ('G > GiC', False),
    'BiC': ('G > GiC > BiC', False), 'GiG': ('G > GiG', False), 'GiP': ('G > GiP', False), 'H2G': ('G > H2G', False),
    'LSB': ('G > LSB', False), 'Gr?': ('GrG', True), 'GrG': ('GrG', False), 'CGG': ('GrG > CGG', False),
    'grv': ('grv', False), 'BH?': ('grv > BH', True), 'BH': ('grv > BH', False), 'gLS': ('grv > gLS', False),
    'LS?': ('grv > gLS', True), 'Le?': ('grv > gLS > gLe', True), 'gLe': ('grv > gLS > gLe', False),
    'LeI': ('grv > gLS > LeI', False), 'LI?': ('grv > gLS > LeI', True), 'LeG': ('grv > gLS > LeI > LeG', False),
    'LeQ': ('grv > gLS > LeI > LeQ', False), 'GWE': ('grv > GWE', False), 'Lev': ('grv > Lev', False),
    'SBG': ('G > SBG', False), 'HV*': ('* > HV*', False), 'IG': ('IG', False), 'IR': ('IR', False),
    'FIR': ('IR > FIR', False), 'MIR': ('IR > MIR', False), 'NIR': ('IR > NIR', False), 'ISM': ('ISM', False),
    'bub': ('ISM > bub', False), 'Cld': ('ISM > Cld', False), 'CGb': ('ISM > Cld > CGb', False),
    'DNe': ('ISM > Cld > DNe', False), 'glb': ('ISM > Cld > glb', False), 'GNe': ('ISM > Cld > GNe', False),
    'RNe': ('ISM > Cld > GNe > RNe', False), 'HVC': ('ISM > Cld > HVC', False), 'MoC': ('ISM > Cld > MoC', False),
    'cor': ('ISM > cor', False), 'flt': ('ISM > flt', False), 'HII': ('ISM > HII', False),
    'SFR': ('ISM > SFR', False), 'sh': ('ISM > sh', False), 'SR?': ('ISM > SNR', True), 'SNR': ('ISM > SNR', False),
    'LM*': ('* > LM*', False), 'LM?': ('* > LM*', True), 'BD?': ('* > LM* > BD*', True),
    'BD*': ('* > LM* > BD*', False), 'Ma?': ('* > Ma*', True), 'Ma*': ('* > Ma*', False),
    'bC?': ('* > Ma* > bC*', True), 'bC*': ('* > Ma* > bC*', False), 'N*?': ('* > Ma* > N*', True),
    'N*': ('* > Ma* > N*', False), 'Psr': ('* > Ma* > N* > Psr', False), 'sg*': ('* > Ma* > sg*', False),
    'sg?': ('* > Ma* > sg*', True), 's?b': ('* > Ma* > sg* > s*b', True), 's*b': ('* > Ma* > sg* > s*b', False),
    'WR?': ('* > Ma* > sg* > s*b > WR*', True), 'WR*': ('* > Ma* > sg* > s*b > WR*', False),
    's*r': ('* > Ma* > sg* > s*r', False), 's?r': ('* > Ma* > sg* > s*r', True),
    's*y': ('* > Ma* > sg* > s*y', False), 's?y': ('* > Ma* > sg* > s*y', True), 'MS?': ('* > MS*', True),
    'MS*': ('* > MS*', False), 'Be?': ('* > MS* > Be*', True), 'Be*': ('* > MS* > Be*', False),
    'BS*': ('* > MS* > BS*', False), 'BS?': ('* > MS* > BS*', True), 'SX*': ('* > MS* > BS* > SX*', False),
    'dS*': ('* > MS* > dS*', False), 'gD*': ('* > MS* > gD*', False), 'mul': ('mul', False), 'Opt': ('Opt', False),
    'blu': ('Opt > blu', False), 'EmO': ('Opt > EmO', False), 'PaG': ('PaG', False), 'PCG?': ('PCG', True),
    'PCG': ('PCG', False), 'Pe?': ('* > Pe*', True), 'Pe*': ('* > Pe*', False), 'a2?': ('* > Pe* > a2*', True),
    'a2*': ('* > Pe* > a2*', False), 'RC?': ('* > Pe* > RC*', True), 'RC*': ('* > Pe* > RC*', False),
    'Pl': ('* > Pl', False), 'Pl?': ('* > Pl', True), 'PM*': ('* > PM*', False), 'PoC': ('PoC', False),
    'PoG': ('PoG', False), 'Rad': ('Rad', False), 'cm': ('Rad > cm', False), 'HI': ('Rad > HI', False),
    'Mas': ('Rad > Mas', False), 'mm': ('Rad > mm', False), 'mR': ('Rad > mR', False), 'rB': ('Rad > rB', False),
    'smm': ('Rad > smm', False), 'reg': ('reg', False), 'RS*': ('* > ** > RS*', False),
    'RS?': ('* > ** > RS*', True), 'SB?': ('* > ** > SB*', True), 'SB*': ('* > ** > SB*', False),
    'SC?': ('SCG', True), 'SCG': ('SCG', False), 'SN*': ('* > SN*', False), 'SN?': ('* > SN*', True),
    'Sy*': ('* > ** > Sy*', False), 'Sy?': ('* > ** > Sy*', True), 'UV': ('UV', False), 'V*': ('* > V*', False),
    'V*?': ('* > V*', True), 'Er*': ('* > V* > Er*', False), 'Er?': ('* > V* > Er*', True), 'vid': ('vid', False),
    'Ir*': ('* > V* > Ir*', False), 'Pu?': ('* > V* > Pu*', True), 'Pu*': ('* > V* > Pu*', False),
    'Ro?': ('* > V* > Ro*', True), 'Ro*': ('* > V* > Ro*', False), 'X': ('X', False), 'XB*': ('* > ** > XB*', False),
    'XB?': ('* > ** > XB*', True), 'HXB': ('* > ** > XB* > HXB', False), 'HX?': ('* > ** > XB* > HXB', True),
    'LX?': ('* > ** > XB* > LXB', True), 'LXB': ('* > ** > XB* > LXB', False), 'ULX': ('X > ULX', False),
    'UX?': ('X > ULX', True), 'Y*?': ('* > Y*O', True), 'Y*O': ('* > Y*O', False), 'Ae?': ('* > Y*O > Ae*', True),
    'Ae*': ('* > Y*O > Ae*', False), 'Or*': ('* > Y*O > Or*', False), 'of?': ('* > Y*O > out', True),
    'out': ('* > Y*O > out', False), 'HH': ('* > Y*O > out > HH', False), 'TT*': ('* > Y*O > TT*', False),
    'TT?': ('* > Y*O > TT*', True), 'var': (None, False),
}
SIMBAD_OTYPEDEF_RETRIEVED = "2026-09-28"
# Transients and events: the SIMBAD row is an event, not the object class (a supernova's redshift is its host's).
SIMBAD_TRANSIENT_CODES = frozenset({"SN*", "SN?", "No*", "No?", "ev", "gB", "GWE", "Lev", "rB"})
# Aggregates of galaxies: extragalactic, and 'galaxy' is the closest class (as in the NED 'GPair'/'GGroup'/'GClstr'
# mapping). Stellar aggregates (Cl*, As*, St*, ...), superclusters and protoclusters are not class-specific.
SIMBAD_GALAXY_AGGREGATE_ROOTS = frozenset({"ClG", "GrG", "IG", "PaG"})
NED_QSO_TYPES = frozenset({"qso", "q_lens", "bllac", "blazar"})
NED_GALAXY_TYPES = frozenset({"g", "gpair", "gtrpl", "ggroup", "gclstr", "g_lens"})


def simbad_type_path(otype: Any) -> str | None:
    """The SIMBAD otypedef path of an otype code (None when unknown or without a path)."""
    entry = SIMBAD_OTYPEDEF.get(str(otype or "").strip())
    return None if entry is None else entry[0]


def simbad_type_class(otype: Any) -> tuple[str | None, float]:
    """Map a SIMBAD otype code to (class, weight factor) through its otypedef path; candidates get half weight.

    'G > AGN > QSO ...' (QSO, Bla, BLL and candidates) and the lensed quasar image 'LeQ' -> qso; the rest of
    'G > AGN ...' (AGN, LIN, rG, SyG, Sy1, Sy2) -> agn; other 'G ...' codes, the lensed galaxy image 'LeG' and
    aggregates of galaxies -> galaxy; '* ...' -> star, except the transients in :data:`SIMBAD_TRANSIENT_CODES`
    (supernovae, novae). Everything else (stellar clusters/associations, ISM, single-band sources, lens systems,
    unknown codes) is not class-specific: (None, 0).
    """
    code = str(otype or "").strip()
    entry = SIMBAD_OTYPEDEF.get(code)
    if entry is None or not entry[0] or code in SIMBAD_TRANSIENT_CODES:
        return None, 0.0
    path, candidate = entry
    factor = 0.5 if candidate else 1.0
    nodes = [n.strip() for n in (path or "").split(">")]
    if nodes[:3] == ["G", "AGN", "QSO"] or code == "LeQ":
        return "qso", factor
    if nodes[:2] == ["G", "AGN"]:
        return "agn", factor
    if nodes[0] == "G" or code == "LeG" or nodes[0] in SIMBAD_GALAXY_AGGREGATE_ROOTS:
        return "galaxy", factor
    if nodes[0] == "*" and not any(n in SIMBAD_TRANSIENT_CODES for n in nodes):
        return "star", factor
    return None, 0.0


def simbad_is_planet(otype: Any) -> bool:
    """SIMBAD 'Pl' / 'Pl?' rows share their host star's coordinates (e.g. '* 51 Peg b' and '* 51 Peg')."""
    return str(otype or "").strip() in {"Pl", "Pl?"}


def ned_type_class(ptype: Any) -> str | None:
    """Map a NED preferred object type to a class; sub-component types (star clusters, HII regions, absorbers,
    single-band sources, ...) are not class evidence for the object."""
    low = str(ptype or "").strip().lower()
    if not low or low in NED_SUBCOMPONENT_TYPES:
        return None
    if low in NED_QSO_TYPES:
        return "qso"
    if low in NED_GALAXY_TYPES:
        return "galaxy"
    if "*" in low:
        return "star"
    return None


def _member(members: Sequence[MemberChoice], catalog: str) -> MemberChoice | None:
    return next((m for m in members if m.catalog == catalog and m.used), None)


def _point(points: Sequence[SEDPoint], facility_prefix: str, band: str) -> SEDPoint | None:
    return next((p for p in points if p.facility.startswith(facility_prefix) and p.band == band), None)


def proper_motion_significance(pmra: float, pmdec: float, e_ra: float, e_dec: float, corr: float | None) -> float:
    """sqrt(mu^T C^-1 mu) for the 2-D proper motion with its covariance (Lindegren et al. 2021)."""
    rho = corr if corr is not None and abs(corr) < 1.0 else 0.0
    det = (e_ra**2) * (e_dec**2) * (1.0 - rho**2)
    if det <= 0:
        return 0.0
    chi2 = (pmra**2 * e_dec**2 - 2.0 * rho * e_ra * e_dec * pmra * pmdec + pmdec**2 * e_ra**2) / det
    return math.sqrt(max(chi2, 0.0))


def absolute_magnitude(apparent: float, z: float) -> float:
    """M = m - 5 log10(D_L / 10 pc), Planck 2018 cosmology, no K-correction."""
    d_l_pc = float(Planck18.luminosity_distance(z).to(u.pc).value)
    return apparent - 5.0 * math.log10(d_l_pc / 10.0)


def _usable(point: SEDPoint | None) -> bool:
    """A measured (not upper-limit) point without a quality warning: eligible for the classification rules."""
    return point is not None and not point.quality_warning and not point.is_upper_limit


def _wise_ok(point: SEDPoint | None) -> bool:
    """WISE band usable for colours: ph_qual A (S/N >= 10) or B (3 <= S/N < 10) (Cutri et al. 2012)."""
    return _usable(point) and (point.quality_flag is None or point.quality_flag.upper() in {"A", "B"})  # type: ignore[union-attr]


# AllWISE ext_flg values that associate the source with a 2MASS Extended Source Catalog (XSC) object (AllWISE
# Explanatory Supplement, Sect. II.1.a, ext_flg): 2/3 = inside the extrapolated isophotal footprint of an XSC
# source, 4/5 = within 5 arcsec of an XSC source (3 and 5 additionally have w?rchi2 > 3). ext_flg = 1 means only
# "the profile-fit photometry goodness-of-fit, w?rchi2, is >3.0 in one or more bands", which is common for
# bright, saturated or high-proper-motion POINT sources, so it is not used as morphology.
ALLWISE_EXT_XSC: dict[int, str] = {
    2: "inside the isophotal footprint of a 2MASS XSC source",
    3: "inside the isophotal footprint of a 2MASS XSC source (and w?rchi2 > 3)",
    4: "within 5 arcsec of a 2MASS XSC source",
    5: "within 5 arcsec of a 2MASS XSC source (and w?rchi2 > 3)",
}


def optical_extent(members: Sequence[MemberChoice]) -> tuple[bool | None, str]:
    """Is the optical counterpart extended? SDSS type > PS1 iPSF - iKron > AllWISE 2MASS-XSC association
    (ext_flg 2-5) or ext_flg = 0 (first available); None when unknown."""
    sdss = _member(members, "sdss")
    if sdss is not None:
        obj_type, why = sdss_morphology(sdss.data)
        if obj_type == 3:
            return True, why
        if obj_type == 6:
            return False, why
    ps1 = _member(members, "panstarrs_dr2")
    if ps1 is not None:
        psf, kron = _num(_ci_get(ps1.data, "iMeanPSFMag")), _num(_ci_get(ps1.data, "iMeanKronMag"))
        lo, hi = PS1_EXTENDED_I_RANGE
        if psf is not None and kron is not None and lo < psf < hi:
            diff = psf - kron
            return diff > PS1_EXTENDED_PSF_KRON, f"PS1 iPSF - iKron = {diff:.2f}"
    allwise = _member(members, "allwise")
    if allwise is not None:
        ext = _num(_ci_get(allwise.data, "ext_flg"))
        if ext is not None and int(ext) in ALLWISE_EXT_XSC:
            return True, f"AllWISE ext_flg = {int(ext)}: {ALLWISE_EXT_XSC[int(ext)]}"
        if ext == 0:
            return False, "AllWISE ext_flg = 0: point-like, no 2MASS XSC association"
        # ext_flg = 1 only says the profile fit has chi2 > 3 (bright, saturated, moving or resolved source).
    return None, "morphology unknown"


# SVO Generic/Johnson.V Vega zero point (Jy): converts a total optical flux density to a V-equivalent magnitude for
# the Maccacaro et al. 1988 X-ray/optical ratio (a band approximation when the flux is PS1/SDSS i, see notes).
JOHNSON_V_ZERO_POINT_JY = float(EMBEDDED_FILTERS["Generic/Johnson.V"]["ZeroPoint"])
STERN_W2_LIMIT = 15.05  # Stern et al. 2012: W1 - W2 >= 0.8 selects AGN for W2 < 15.05 (Vega)
# Mid-IR/radio correlation of star formation: q24 = log10(S_24um / S_1.4GHz) = 0.84 +/- 0.28 (Appleton et al. 2004,
# ApJS 154, 147), and q22 with WISE W4 alike. Radio-loud AGN lie far below (OJ 287: q22 = -0.99, NGC 1316: -0.24);
# dusty starbursts on it (Arp 220: 1.15, Mrk 231: 1.32) have radio emission from star formation, and their radio
# excess over the *optical* comes from dust extinction of the optical light. A radio-loudness R > 1 counts as AGN
# evidence only with a radio excess over the correlation, q22 < Q22_RADIO_EXCESS.
Q22_RADIO_EXCESS = 0.5
I_BAND_UM = 0.75  # PS1 i / SDSS i effective wavelength (SVO: 0.7503 / 0.7458 um)


def select_optical_i(points: Sequence[SEDPoint], extended: bool | None) -> tuple[float | None, str, list[str]]:
    """The i-band flux density (Jy) for R_i and M_i, with its provenance label and rejected candidates.

    Preference: an unflagged PS1/SDSS i that agrees (|Delta m| <= 0.5) with the log-log interpolation of the
    unflagged neighbouring bands of other facilities (e.g. 3C 273's PS1 i is 0.7 mag below its neighbours);
    else Gaia RP (lambda_eff 0.76 um) as an i proxy; else that interpolation at 0.75 um; else Gaia G. The
    consistency check is skipped for extended sources (PSF and window photometry legitimately differ).
    """
    rejected: list[str] = []
    rejected_points: list[SEDPoint] = []
    for prefix in ("Pan-STARRS1", "SDSS"):
        cand = _point(points, prefix, "i")
        if cand is None:
            continue
        if not _usable(cand):
            rejected.append(f"{prefix} i quality-flagged")
            continue
        if extended is not True:
            ref = interpolated_flux(points, cand.wavelength_um, exclude_facility=cand.facility)
            if ref is not None:
                delta = -2.5 * math.log10(cand.flux_jy / ref[0])
                if abs(delta) > OPTICAL_GAIA_TOLERANCE_MAG:
                    rejected.append(f"{prefix} i differs by {delta:+.2f} mag from {ref[1]}")
                    rejected_points.append(cand)
                    continue
        return cand.flux_jy, f"{prefix} i", rejected
    rp = _point(points, "Gaia", "RP")
    if _usable(rp):
        return rp.flux_jy, "Gaia RP as i proxy", rejected  # type: ignore[union-attr]
    ref = interpolated_flux([p for p in points if not any(p is r for r in rejected_points)], I_BAND_UM)
    if ref is not None:
        return ref[0], f"i-band flux from {ref[1]}", rejected
    g = _point(points, "Gaia", "G")
    if _usable(g):
        return g.flux_jy, "Gaia G", rejected  # type: ignore[union-attr]
    return None, "", rejected


INTERPOLATION_MAX_RATIO = 1.35  # each neighbour within a factor 1.35 in wavelength (curvature of steep SEDs)


def interpolated_flux(
    points: Sequence[SEDPoint], wavelength_um: float, *, exclude_facility: str | None = None,
) -> tuple[float, str] | None:
    """Log-log interpolation of F_nu at ``wavelength_um`` between the nearest unflagged survey points on either
    side, each within a factor :data:`INTERPOLATION_MAX_RATIO` in wavelength (SIMBAD literature values and
    ``exclude_facility`` excluded); None without such a bracketing pair."""
    lo_lim, hi_lim = wavelength_um / INTERPOLATION_MAX_RATIO, wavelength_um * INTERPOLATION_MAX_RATIO
    usable = [p for p in points if _usable(p) and p.catalog != "simbad" and p.facility != exclude_facility
              and lo_lim <= p.wavelength_um <= hi_lim and p.flux_jy > 0]
    below = [p for p in usable if p.wavelength_um < wavelength_um]
    above = [p for p in usable if p.wavelength_um > wavelength_um]
    if not below or not above:
        return None
    lo = max(below, key=lambda p: p.wavelength_um)
    hi = min(above, key=lambda p: p.wavelength_um)
    t = math.log(wavelength_um / lo.wavelength_um) / math.log(hi.wavelength_um / lo.wavelength_um)
    flux = math.exp(math.log(lo.flux_jy) + t * (math.log(hi.flux_jy) - math.log(lo.flux_jy)))
    return flux, (f"interpolation between {lo.facility} {lo.band} ({lo.wavelength_um:.2f} um) and "
                  f"{hi.facility} {hi.band} ({hi.wavelength_um:.2f} um)")


def total_optical_flux(points: Sequence[SEDPoint], members: Sequence[MemberChoice]) -> tuple[float, str] | None:
    """Largest total-light optical flux density (Jy) of an extended counterpart: PS1 iKron, SDSS modelMag i,
    or SIMBAD's integrated V (the largest is the least likely to overstate radio loudness or the X-ray/optical
    ratio; i-band values used as V are a band approximation, stated in the evidence label)."""
    found: list[tuple[float, str]] = []
    ps1 = _member(members, "panstarrs_dr2")
    if ps1 is not None:
        kron, psf = _num(_ci_get(ps1.data, "iMeanKronMag")), _num(_ci_get(ps1.data, "iMeanPSFMag"))
        if kron is not None and -90 < kron < 90 and (psf is None or psf >= PS1_SATURATION_MAG["i"]):
            found.append((ab_mag_to_jy(kron)[0], f"PS1 iKron = {kron:.2f}"))
    sdss_i = _point(points, "SDSS", "i")
    if _usable(sdss_i) and any("modelMag" in n and "all bands" in n for n in sdss_i.notes):  # type: ignore[union-attr]
        found.append((sdss_i.flux_jy, f"SDSS modelMag i = {sdss_i.magnitude:.2f}"))  # type: ignore[union-attr]
    v = _point(points, "SIMBAD", "V")
    if _usable(v):
        found.append((v.flux_jy, f"SIMBAD V = {v.magnitude:.2f}"))  # type: ignore[union-attr]
    return max(found) if found else None


def _pick_xray(points: Sequence[SEDPoint]) -> SEDPoint | None:
    """Unflagged X-ray point, preferring XMM-Newton > Chandra > ROSAT (broad band, modern calibration)."""
    order = {"xmm": 0, "chandra": 1, "rosat": 2, "rosat_bsc": 3}
    usable = [p for p in points if p.regime == "xray" and _usable(p)]
    usable.sort(key=lambda p: order.get(p.catalog, 9))
    return usable[0] if usable else None


def gather_evidence(
    points: Sequence[SEDPoint],
    members: Sequence[MemberChoice],
    redshift: Mapping[str, Any],
    gaia_extra: Mapping[str, Any] | None = None,
    *,
    use_redshift: bool = True,
    position: tuple[float, float] | None = None,
) -> list[Evidence]:
    """Each rule contributes a (text, weights) pair; weights are log-odds-like increments per class.

    Quality-flagged points are never used. ``use_redshift=False`` drops the redshift-based rules (used when
    the object is classified as a star, where a large redshift is a misassociation, not evidence).
    """
    ev: list[Evidence] = []
    gaia = _member(members, "gaia_dr3")
    gaia_astrometry_used = False
    stellar_astrometry = False  # a significant parallax or a large proper motion: a Galactic object
    extended, extent_reason = optical_extent(members)

    # --- Gaia astrometry --------------------------------------------------
    if gaia is not None:
        d = gaia.data
        ruwe = _num((gaia_extra or {}).get("ruwe")) or _num(_ci_get(d, "ruwe"))
        plx, plx_err = _num(_ci_get(d, "parallax")), _num(_ci_get(d, "parallax_error"))
        poe = _num((gaia_extra or {}).get("parallax_over_error"))
        if poe is None and plx is not None and plx_err:
            poe = plx / plx_err
        pmra, pmdec = _num(_ci_get(d, "pmra")), _num(_ci_get(d, "pmdec"))
        e_ra, e_dec = _num((gaia_extra or {}).get("pmra_error")), _num((gaia_extra or {}).get("pmdec_error"))
        corr = _num((gaia_extra or {}).get("pmra_pmdec_corr"))
        aen_sig = _num((gaia_extra or {}).get("astrometric_excess_noise_sig"))
        noisy = aen_sig is not None and aen_sig > GAIA_AEN_SIG_LIMIT
        pm_sig: float | None = None
        g_mag = _num(_ci_get(d, "phot_g_mean_mag"))
        bright = g_mag is not None and g_mag < GAIA_BRIGHT_G_LIMIT
        floor_text = f"{GAIA_PM_SYSTEMATIC_MASYR * 1000:.0f} uas/yr systematics" + (
            f" + {GAIA_BRIGHT_PM_BIAS_MASYR * 1000:.0f} uas/yr bright-star bias" if bright else "")
        if None not in (pmra, pmdec, e_ra, e_dec) and e_ra > 0 and e_dec > 0:  # type: ignore[operator]
            # Small-scale systematics (all sources) and the bright-star frame bias (G < 13) in quadrature.
            extra = math.hypot(GAIA_PM_SYSTEMATIC_MASYR, GAIA_BRIGHT_PM_BIAS_MASYR if bright else 0.0)
            e_ra = math.hypot(e_ra, extra)  # type: ignore[arg-type]
            e_dec = math.hypot(e_dec, extra)  # type: ignore[arg-type]
            pm_sig = proper_motion_significance(pmra, pmdec, e_ra, e_dec, corr)  # type: ignore[arg-type]
        if ruwe is not None and ruwe >= RUWE_GOOD:
            ev.append(Evidence(f"Gaia DR3 RUWE = {ruwe:.2f} >= {RUWE_GOOD}: astrometry unreliable (extended/blended source?); "
                               "parallax and proper motion not used", {}))
        elif poe is not None:
            gaia_astrometry_used = True
            if poe > 5.0:
                stellar_astrometry = True
                ev.append(Evidence(f"Gaia DR3 parallax/error = {poe:.1f} > 5: significant parallax (Galactic star)",
                                   {"star": 4.0, "qso": -2.0, "galaxy": -2.0, "agn": -2.0}))
            if pm_sig is not None and pm_sig > 5.0 and pmra is not None and pmdec is not None:
                pm_tot = math.hypot(pmra, pmdec)
                head = f"Gaia DR3 proper motion {pm_tot:.2f} mas/yr at {pm_sig:.1f} sigma (> 5, errors incl. {floor_text})"
                if noisy and pm_tot < GAIA_PM_NOISY_MIN_MASYR:
                    ev.append(Evidence(f"{head} but astrometric_excess_noise_sig = {aen_sig:.1f} > {GAIA_AEN_SIG_LIMIT:g} "
                                       "(significant excess noise: source structure or variability-induced motion, "
                                       f"Lindegren et al. 2012) and < {GAIA_PM_NOISY_MIN_MASYR:g} mas/yr: proper motion "
                                       "not used", {}))
                elif pm_tot >= GAIA_PM_STRONG_MASYR:
                    stellar_astrometry = True
                    ev.append(Evidence(f"{head}, >= {GAIA_PM_STRONG_MASYR:g} mas/yr: moving, Galactic",
                                       {"star": 3.0, "qso": -1.5, "galaxy": -1.5, "agn": -1.5}))
                elif pm_tot >= GAIA_PM_PHYSICAL_MASYR:
                    ev.append(Evidence(f"{head} but < {GAIA_PM_STRONG_MASYR:g} mas/yr: weak evidence (spurious motions of "
                                       "0.2-0.6 mas/yr are seen for structured/variable quasars)",
                                       {"star": 1.0, "qso": -0.5, "galaxy": -0.5, "agn": -0.5}))
                else:
                    ev.append(Evidence(f"{head} but < {GAIA_PM_PHYSICAL_MASYR} mas/yr (a star would need d > 100 kpc at "
                                       "v_t = 100 km/s): very weak evidence", {"star": 0.5, "qso": -0.25, "galaxy": -0.25,
                                                                                "agn": -0.25}))
            if abs(poe) < 3.0 and pm_sig is not None and pm_sig < 3.0:
                ev.append(Evidence(f"Gaia DR3 parallax/error = {poe:.2f} and proper motion {pm_sig:.1f} sigma (errors incl. "
                                   f"{floor_text}): astrometrically stationary, consistent with an extragalactic source",
                                   {"star": -0.5, "qso": 0.5, "galaxy": 0.5, "agn": 0.5}))
            elif abs(poe) < 3.0 and pm_sig is None:
                ev.append(Evidence(f"Gaia DR3 parallax/error = {poe:.2f}: no significant parallax", {}))
            elif abs(poe) <= 5.0 and pm_sig is not None and pm_sig <= 5.0:
                ev.append(Evidence(f"Gaia DR3 parallax/error = {poe:.2f} and proper motion {pm_sig:.1f} sigma (errors incl. "
                                   f"{floor_text}): inconclusive", {}))
        probs = {name: _num((gaia_extra or {}).get(f"classprob_dsc_combmod_{name}"))
                 for name in ("quasar", "galaxy", "star", "whitedwarf", "binarystar")}
        if any(probs[name] is not None for name in ("quasar", "galaxy", "star")):
            # DSC-Combmod has five classes; gaia_source's 'star' excludes white dwarfs and physical binaries, which
            # are stars too (astrophysical_parameters). DSC classifies point-source BP/RP spectra; halve its weight
            # for extended counterparts.
            stellar = [v for n in ("star", "whitedwarf", "binarystar") if (v := probs[n]) is not None]
            dsc = {"qso": probs["quasar"], "galaxy": probs["galaxy"], "star": sum(stellar) if stellar else None}
            scale = 0.5 if extended else 1.0
            weights = {cls: round(2.0 * scale * (v or 0.0), 3) for cls, v in dsc.items()}
            weights["agn"] = round(1.0 * scale * (dsc["qso"] or 0.0), 3)
            parts = [f"{label}={probs[name]:.3f}" for label, name in (("qso", "quasar"), ("galaxy", "galaxy"),
                                                                       ("single star", "star"),
                                                                       ("white dwarf", "whitedwarf"),
                                                                       ("binary star", "binarystar"))
                     if probs[name] is not None]
            missing = ("" if probs["whitedwarf"] is not None and probs["binarystar"] is not None else
                       "; white-dwarf/binary probabilities unavailable, so star covers single non-WD stars only")
            ev.append(Evidence("Gaia DR3 DSC-Combmod probabilities (Delchambre et al. 2023): " + ", ".join(parts)
                               + f" -> star = {dsc['star'] if dsc['star'] is not None else 0.0:.3f}{missing}"
                               + (f" (half weight: extended counterpart, {extent_reason})" if extended else ""), weights))
        if _flag_true((gaia_extra or {}).get("in_qso_candidates")):
            ev.append(Evidence("Gaia DR3 qso_candidates member (low-purity candidate table, Gaia Collaboration, "
                               "Bailer-Jones et al. 2023)", {"qso": 0.75, "agn": 0.25}))
        if _flag_true((gaia_extra or {}).get("in_galaxy_candidates")):
            ev.append(Evidence("Gaia DR3 galaxy_candidates member (low-purity candidate table)", {"galaxy": 0.75}))

    # --- SIMBAD proper motion / parallax (no errors; only extreme values are meaningful) ----------
    simbad = _member(members, "simbad")
    if simbad is not None and not gaia_astrometry_used:
        pmra, pmdec = _num(_ci_get(simbad.data, "pmra")), _num(_ci_get(simbad.data, "pmdec"))
        plx = _num(_ci_get(simbad.data, "plx_value"))
        if pmra is not None and pmdec is not None and math.hypot(pmra, pmdec) > 50.0:
            stellar_astrometry = True
            ev.append(Evidence(f"SIMBAD proper motion {math.hypot(pmra, pmdec):.1f} mas/yr > 50 mas/yr: nearby star",
                               {"star": 2.5, "qso": -1.0, "galaxy": -1.0, "agn": -1.0}))
        if plx is not None and plx > 10.0:
            stellar_astrometry = True
            ev.append(Evidence(f"SIMBAD parallax {plx:.2f} mas > 10 mas: within 100 pc", {"star": 2.0, "qso": -1.0,
                                                                                         "galaxy": -1.0, "agn": -1.0}))

    # --- WISE colours (ph_qual A/B only) ---------------------------------------
    w1, w2, w3 = _point(points, "WISE", "W1"), _point(points, "WISE", "W2"), _point(points, "WISE", "W3")
    if _wise_ok(w1) and _wise_ok(w2) and w1.magnitude is not None and w2.magnitude is not None:  # type: ignore[union-attr]
        c12 = w1.magnitude - w2.magnitude  # type: ignore[union-attr]
        if w2.magnitude >= STERN_W2_LIMIT:  # type: ignore[union-attr]
            ev.append(Evidence(f"WISE W1-W2 = {c12:.2f} but W2 = {w2.magnitude:.2f} >= {STERN_W2_LIMIT}: the Stern et al. "  # type: ignore[union-attr]
                               "2012 criterion is defined only for W2 < 15.05 (fainter sources need the W2-dependent "
                               "Assef et al. 2013 cut): not applied", {}))
        elif c12 >= 0.8 and stellar_astrometry:
            ev.append(Evidence(f"WISE W1-W2 = {c12:.2f} >= 0.8 but the source has a significant parallax or proper motion: "
                               "the red W1-W2 of a Galactic object (cool brown dwarfs reach W1-W2 ~ 1-3, Kirkpatrick et al. "
                               "2011, ApJS 197, 19), not the Stern et al. 2012 AGN signature: not applied", {}))
        elif c12 >= 0.8:
            ev.append(Evidence(f"WISE W1-W2 = {c12:.2f} >= 0.8 (Vega, W2 = {w2.magnitude:.2f} < {STERN_W2_LIMIT}): "  # type: ignore[union-attr]
                               "Stern et al. 2012 mid-IR AGN criterion", {"qso": 2.0, "agn": 2.0, "star": -1.0, "galaxy": -0.5}))
        else:
            ev.append(Evidence(f"WISE W1-W2 = {c12:.2f} < 0.8: no mid-IR AGN signature (Stern et al. 2012)", {}))
        if w3 is not None and _wise_ok(w3) and w3.magnitude is not None:
            c23 = w2.magnitude - w3.magnitude  # type: ignore[union-attr]
            if c12 < 0.3 and c23 < 0.5:
                ev.append(Evidence(f"WISE W1-W2 = {c12:.2f}, W2-W3 = {c23:.2f}: stellar Rayleigh-Jeans locus (Wright et al. 2010)",
                                   {"star": 1.0}))
            elif c12 < 0.8 and c23 >= 1.5:
                ev.append(Evidence(f"WISE W2-W3 = {c23:.2f} >= 1.5 with W1-W2 < 0.8: dusty star-forming galaxy locus "
                                   "(Wright et al. 2010)", {"galaxy": 1.0, "star": -0.5}))
        elif w3 is not None and w3.quality_flag and w3.quality_flag.upper() not in {"A", "B"}:
            ev.append(Evidence(f"WISE W3 ph_qual {w3.quality_flag} (S/N < 3): W2-W3 colour not used", {}))

    # --- Radio loudness ----------------------------------------------------
    first, nvss = _point(points, "FIRST", "1.4 GHz"), _point(points, "NVSS", "1.4 GHz")
    radio_order = (nvss, first) if extended else (first, nvss)
    radio = next((p for p in radio_order if _usable(p)), None)
    optical_i, optical_label, rejected = select_optical_i(points, extended)
    w4 = _point(points, "WISE", "W4")

    def radio_excess(loud_text: str, weights: dict[str, float]) -> Evidence:
        """The R > 1 evidence, counted only with a radio excess over the mid-IR/radio correlation (q22)."""
        if radio is None:
            raise ValueError("radio_excess needs a radio point")
        if _wise_ok(w4) and w4 is not None and w4.flux_jy > 0:
            q22 = math.log10(w4.flux_jy / radio.flux_jy)
            if q22 >= Q22_RADIO_EXCESS:
                return Evidence(f"{loud_text}, but q22 = log10(F_W4/F_1.4GHz) = {q22:.2f} >= {Q22_RADIO_EXCESS:g}: on the "
                                "star-forming mid-IR/radio correlation (q24 = 0.84 +/- 0.28, Appleton et al. 2004), so the "
                                "radio emission can come from star formation and the radio excess over the optical from "
                                "dust extinction: not counted as radio-loud AGN evidence", {})
            return Evidence(f"{loud_text}; q22 = log10(F_W4/F_1.4GHz) = {q22:.2f} < {Q22_RADIO_EXCESS:g}: radio excess over "
                            "the star-forming mid-IR/radio correlation (Appleton et al. 2004): radio-loud AGN", weights)
        if (_wise_ok(w1) and _wise_ok(w2) and _wise_ok(w3) and w1 is not None and w2 is not None and w3 is not None
                and w1.magnitude is not None and w2.magnitude is not None and w3.magnitude is not None
                and w1.magnitude - w2.magnitude < 0.8 and w2.magnitude - w3.magnitude >= 1.5):
            half = {cls: round(0.5 * w, 3) for cls, w in weights.items()}
            return Evidence(f"{loud_text}; q22 not available (no usable W4), and WISE W2-W3 = "
                            f"{w2.magnitude - w3.magnitude:.2f} >= 1.5 is on the dusty star-forming locus: the radio "
                            "excess over the optical may come from dust extinction (half weight)", half)
        return Evidence(f"{loud_text}; q22 not available (no usable W4): radio excess over the mid-IR/radio "
                        "correlation not checked", weights)

    if radio is not None and extended:
        total = total_optical_flux(points, members)
        if total is None:
            ev.append(Evidence(f"radio loudness not evaluated: the optical counterpart is extended ({extent_reason}) and no "
                               "total optical magnitude is available (a nuclear PSF flux would overstate R)", {}))
        else:
            flux, label = total
            r_tot = math.log10(radio.flux_jy / flux)
            if r_tot > 1.0:
                ev.append(radio_excess(f"radio loudness R = log10(F_1.4GHz/F_opt,total) = {r_tot:.2f} > 1 ({radio.facility} "
                                       f"{radio.flux_jy:.3g} Jy, {label}; extended counterpart, {extent_reason}): radio-loud "
                                       "(Ivezic et al. 2002 criterion is defined for point sources: down-weighted)",
                                       {"agn": 1.0, "galaxy": 0.3, "star": -0.5}))
            else:
                ev.append(Evidence(f"radio loudness R = log10(F_1.4GHz/F_opt,total) = {r_tot:.2f} <= 1 ({label}; extended "
                                   f"counterpart, {extent_reason}): radio-quiet or star-forming", {}))
    elif radio is not None and optical_i is not None:
        r_i = math.log10(radio.flux_jy / optical_i)
        skipped = f"; {'; '.join(rejected)}" if rejected else ""
        if r_i > 1.0:
            ev.append(radio_excess(f"radio loudness R = log10(F_1.4GHz/F_i) = {r_i:.2f} > 1 ({radio.facility} "
                                   f"{radio.flux_jy:.3g} Jy, {optical_label}{skipped}): radio-loud (Ivezic et al. 2002)",
                                   {"agn": 1.5, "qso": 1.0, "galaxy": 0.3, "star": -1.0}))
        else:
            ev.append(Evidence(f"radio loudness R = {r_i:.2f} <= 1 ({optical_label}{skipped}): radio-quiet or star-forming "
                               "(Ivezic et al. 2002)", {}))

    # --- X-ray / optical ------------------------------------------------------
    xray = _pick_xray(points)
    flagged_xray = [p.facility for p in points if p.regime == "xray" and p.quality_warning]
    # Point-like or unknown morphology: Gaia G (homogeneous, measured) as the V proxy; SIMBAD's compiled V only when
    # Gaia is absent. Extended counterparts: the total optical light (Maccacaro et al. 1988 and Stocke et al. 1991
    # use total V magnitudes), never the Gaia G of a nucleus/knot (Cen A: G = 21.1 vs integrated V = 6.8).
    v_mag: float | None = None
    v_label = ""
    if extended:
        total_v = total_optical_flux(points, members)
        if total_v is not None:
            v_mag = -2.5 * math.log10(total_v[0] / JOHNSON_V_ZERO_POINT_JY)
            v_label = f"total optical light as V: {total_v[1]}; extended counterpart, {extent_reason}"
        elif xray is not None:
            ev.append(Evidence(f"X-ray/optical ratio not evaluated: the optical counterpart is extended ({extent_reason}) "
                               "and no total optical magnitude is available", {}))
    else:
        gp = _point(points, "Gaia", "G")
        if gp is not None and _usable(gp) and gp.magnitude is not None:
            v_mag, v_label = gp.magnitude, "Gaia G as V proxy"
        else:
            v_point = _point(points, "SIMBAD", "V")
            if v_point is not None and _usable(v_point) and v_point.magnitude is not None:
                v_mag, v_label = v_point.magnitude, "V (SIMBAD)"
    if xray is not None and v_mag is not None:
        spec = next(s for s in XRAY_SPECS[xray.catalog][1] if s.band == xray.band)
        # Recover the band flux, then rescale to Maccacaro's 0.3-3.5 keV band with the same Gamma = 2 law.
        band_flux = xray.nu_fnu_erg_s_cm2 * math.log(spec.e_hi_kev / spec.e_lo_kev)
        fx = band_flux * xray_band_scale(spec.e_lo_kev, spec.e_hi_kev, 0.3, 3.5)
        ratio = math.log10(fx) + v_mag / 2.5 + 5.37
        skipped = f"; flagged, not used: {', '.join(flagged_xray)}" if flagged_xray else ""
        if stellar_astrometry:
            # Gaia or SIMBAD parallax/proper motion: a Galactic star (e.g. Procyon B, known to SIMBAD only, whose ROSAT
            # blend with Procyon A gives an 'AGN-like' ratio against the white dwarf's V).
            ev.append(Evidence(f"log(fX/fV) = {ratio:.2f} ({xray.facility}, {v_label}{skipped}): not diagnostic for a source with a "
                               "significant parallax or a large proper motion (coronal or hot white-dwarf X-rays, or "
                               "the X-rays of a companion blended in the X-ray PSF)", {}))
        elif ratio > -1.0:
            ev.append(Evidence(f"log(fX/fV) = {ratio:.2f} > -1 ({xray.facility}, 0.3-3.5 keV, {v_label}{skipped}): "
                               "AGN-like X-ray/optical ratio (Maccacaro et al. 1988; Stocke et al. 1991)",
                               {"agn": 1.5, "qso": 1.0, "star": -1.0}))
        else:
            ev.append(Evidence(f"log(fX/fV) = {ratio:.2f} <= -1 ({xray.facility}, {v_label}{skipped}): typical of stellar "
                               "coronae or normal galaxies, not diagnostic", {}))

    # --- Catalogued object types ----------------------------------------------
    # A quasar/blazar type of an EXTENDED counterpart counts as a quasar only when its nucleus can be quasar-luminous
    # at the adopted redshift; otherwise it is an active nucleus in a resolved host galaxy (Cygnus A and NGC 1275 are
    # typed 'Bla' by SIMBAD: an FR II and a cD galaxy, not quasars). Without a usable redshift nothing is changed.
    nucleus_ok, nucleus_why = (nuclear_quasar_luminosity(points, members, redshift)
                               if extended and use_redshift else (None, ""))
    host_dominated = nucleus_ok is False
    if simbad is not None:
        otype = _ci_get(simbad.data, "otype")
        cls, factor = simbad_type_class(otype)
        path = simbad_type_path(otype)
        where = f" (otypedef path '{path}')" if path else ""
        if cls == "qso" and host_dominated:
            ev.append(Evidence(f"SIMBAD object type '{otype}'{where} -> qso, but the counterpart is extended "
                               f"({extent_reason}) and its nucleus is not quasar-luminous ({nucleus_why}): counted as an "
                               "active nucleus in its host galaxy (agn)",
                               {"agn": 3.0 * factor, "qso": 1.0 * factor, "galaxy": 1.0 * factor}))
        elif cls is not None:
            weights = {cls: 3.0 * factor}
            if cls in {"qso", "agn"}:
                weights[{"qso": "agn", "agn": "qso"}[cls]] = 1.0 * factor
            if cls == "agn":  # SIMBAD AGN types (Sy1/Sy2/LINER/AGN) are galaxies hosting an active nucleus
                weights["galaxy"] = 1.0 * factor
            ev.append(Evidence(f"SIMBAD object type '{otype}'{where} -> {cls}", weights))
        elif str(otype or "").strip() in SIMBAD_TRANSIENT_CODES:
            ev.append(Evidence(f"SIMBAD object type '{otype}'{where} is a transient event (supernova/nova), not an object "
                               "class: not class-specific (a supernova's redshift is its host galaxy's)", {}))
        elif otype:
            ev.append(Evidence(f"SIMBAD object type '{otype}'{where} is not class-specific", {}))
    ned = _member(members, "ned")
    if ned is not None:
        ptype = _ci_get(ned.data, "prefphytype")
        cls = ned_type_class(ptype)
        ned_z = _num(_ci_get(ned.data, "z"))
        if cls == "star" and ned_z is not None and abs(ned_z) > S5_HVS1_Z and not near_galactic_centre(position):
            ev.append(Evidence(f"NED preferred type '{ptype}' ignored: the same NED row ({ned.source_id}) has z = {ned_z:.5g} "
                               f"(cz = {ned_z * C_KMS:.0f} km/s > 1017 km/s, S5-HVS1), inconsistent with a star "
                               "(internally inconsistent NED entry)", {}))
        elif cls == "qso" and host_dominated:
            ev.append(Evidence(f"NED preferred type '{ptype}' -> qso, but the counterpart is extended ({extent_reason}) "
                               f"and its nucleus is not quasar-luminous ({nucleus_why}): counted as an active nucleus "
                               "in its host galaxy (agn)", {"agn": 2.5}))
        elif cls is not None:
            ev.append(Evidence(f"NED preferred type '{ptype}' -> {cls}", {cls: 2.5}))

    # --- SDSS morphology / spectral class ------------------------------------
    sdss = _member(members, "sdss")
    sdss_type, sdss_type_why = sdss_morphology(sdss.data) if sdss is not None else (None, "")
    if sdss is not None and sdss_type is None and "not used" in sdss_type_why:
        ev.append(Evidence(f"{sdss_type_why}: SDSS morphology not used", {}))
    if sdss is not None and sdss_type is not None:
        obj_type = sdss_type
        if obj_type == 3:
            ev.append(Evidence("SDSS photometric type 3 (extended): galaxy morphology (a quasar-dominated nucleus would "
                               "look point-like)", {"galaxy": 1.0, "star": -0.5, "qso": -0.5}))
        elif obj_type == 6:
            ev.append(Evidence("SDSS photometric type 6 (point source): star or quasar", {"star": 0.5, "qso": 0.5}))
    if (sdss is None or sdss_type is None) and extended is not None and extent_reason.startswith("PS1"):
        if extended:
            ev.append(Evidence(f"{extent_reason} > {PS1_EXTENDED_PSF_KRON} (MAST PS1 star/galaxy separation): extended, galaxy "
                               "morphology", {"galaxy": 1.0, "star": -0.5, "qso": -0.5}))
        else:
            ev.append(Evidence(f"{extent_reason} <= {PS1_EXTENDED_PSF_KRON}: point-like (star or quasar)", {"star": 0.5, "qso": 0.5}))
    allwise = _member(members, "allwise")
    if allwise is not None:
        ext = _num(_ci_get(allwise.data, "ext_flg"))
        if ext is not None and int(ext) in ALLWISE_EXT_XSC:
            ev.append(Evidence(f"AllWISE ext_flg = {int(ext)}: {ALLWISE_EXT_XSC[int(ext)]} (an association with a "
                               "2MASS extended source, not proof of extension)", {"galaxy": 0.5, "agn": 0.25}))
        elif ext == 1:
            ev.append(Evidence("AllWISE ext_flg = 1: profile-fit chi2 > 3 in at least one band (poor PSF fit: bright, "
                               "saturated, moving or resolved source); not used as morphology", {}))
    spec_class = next((c.get("spec_class") for c in redshift.get("candidates") or []
                       if c.get("source") in {"sdss_specobj", "sdss"} and c.get("spec_class") and c.get("reliable") is True), None)
    if spec_class:
        mapped = {"STAR": "star", "GALAXY": "galaxy", "QSO": "qso"}.get(str(spec_class).upper())
        if mapped:
            ev.append(Evidence(f"SDSS spectroscopic class {spec_class} (zWarning = 0)", {mapped: 3.0}))

    # --- Redshift & luminosity ------------------------------------------------
    if not use_redshift:
        return ev
    z, kind, reliable = _num(redshift.get("value")), redshift.get("kind"), redshift.get("reliable")
    if z is not None and reliable is False:
        ev.append(Evidence(f"redshift z = {z:.5f} ({redshift.get('source')}) is flagged unreliable: redshift rules not applied", {}))
        return ev
    galactic_centre = near_galactic_centre(position)
    if z is not None and kind == "spec" and z > S5_HVS1_Z and not galactic_centre:
        ev.append(Evidence(f"spectroscopic z = {z:.5f} (cz = {z * C_KMS:.0f} km/s) exceeds the fastest known field/"
                           "hypervelocity star (S5-HVS1, 1017 km/s; heuristic): extragalactic",
                           {"star": -3.0, "qso": 0.5, "galaxy": 0.5, "agn": 0.5}))
    elif z is not None and kind == "spec" and z > S5_HVS1_Z:
        ev.append(Evidence(f"spectroscopic z = {z:.5f} near Sgr A* (stars there reach thousands of km/s): the S5-HVS1 "
                           "ceiling is not applied", {}))
    elif z is not None and kind == "photo" and z > 0.05:
        ev.append(Evidence(f"photometric z = {z:.3f}: extragalactic", {"star": -1.5, "qso": 0.3, "galaxy": 0.3, "agn": 0.3}))
    i_like = optical_label != "Gaia G"
    if z is not None and kind == "spec" and z > S5_HVS1_Z and optical_i is not None and i_like:
        if extended:
            ev.append(Evidence(f"M_i not evaluated: extended counterpart ({extent_reason}); the quasar luminosity criterion "
                               "applies to point-like nuclei", {}))
        else:
            m_i = jy_to_ab_mag(optical_i)
            abs_i = absolute_magnitude(m_i, z)
            if abs_i < -22.0:
                ev.append(Evidence(f"M_i = {abs_i:.1f} < -22 (m_i = {m_i:.2f} AB from {optical_label}, Planck18, no "
                                   "K-correction): quasar luminosity (Schneider et al. 2010)", {"qso": 2.0, "galaxy": -0.5}))
            else:
                ev.append(Evidence(f"M_i = {abs_i:.1f} >= -22 (m_i = {m_i:.2f} AB from {optical_label}): below the quasar "
                                   "luminosity threshold", {"qso": -1.0, "galaxy": 0.5, "agn": 0.5}))
    return ev


# Tie-break order among classes with equal summed weight. Point-like/unknown morphology keeps CLASSES order;
# an extended counterpart prefers the host-galaxy classes (a quasar is point-like by definition).
EXTENDED_TIE_ORDER: tuple[str, ...] = ("galaxy", "agn", "qso", "star")


def _score(
    evidence: Sequence[Evidence], tie_order: Sequence[str] = CLASSES,
) -> tuple[str, float, dict[str, float], list[str]]:
    """(label, confidence, softmax scores, classes tied at the top); ties are resolved by ``tie_order``."""
    totals = {cls: 0.0 for cls in CLASSES}
    for item in evidence:
        for cls, weight in item.weights.items():
            totals[cls] += weight
    peak = max(totals.values())
    exps = {cls: math.exp(v - peak) for cls, v in totals.items()}
    norm = sum(exps.values())
    scores = {cls: round(v / norm, 4) for cls, v in exps.items()}
    if not any(item.weights for item in evidence):
        return "unknown", 0.0, scores, []
    tied = [cls for cls in CLASSES if abs(totals[cls] - peak) <= 1e-9]
    label = min(tied, key=lambda c: list(tie_order).index(c))
    return label, scores[label], scores, (tied if len(tied) > 1 else [])


def near_galactic_centre(position: tuple[float, float] | None) -> bool:
    """Within :data:`GALACTIC_CENTRE_RADIUS_ARCSEC` of Sgr A* (where stars reach thousands of km/s)."""
    if position is None:
        return False
    ra, dec = position
    ra0, dec0 = SGR_A_STAR_RADEC
    cos_d = (math.sin(math.radians(dec)) * math.sin(math.radians(dec0))
             + math.cos(math.radians(dec)) * math.cos(math.radians(dec0)) * math.cos(math.radians(ra - ra0)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d)))) * 3600.0 <= GALACTIC_CENTRE_RADIUS_ARCSEC


def redshift_star_conflict(redshift: Mapping[str, Any], position: tuple[float, float] | None = None) -> bool:
    """Heuristic: |cz| above the fastest known field/hypervelocity star (S5-HVS1, 1017 km/s).

    Not applied near Sgr A* (S-stars reach several thousand km/s); compact binaries can also exceed it in single
    epochs, so this flags a *probable* misassociation, not an impossibility.
    """
    z = _num(redshift.get("value"))
    return z is not None and abs(z) > S5_HVS1_Z and not near_galactic_centre(position)


QSO_ABS_MAG_LIMIT = -22.0  # Schneider et al. 2010: SDSS DR7 quasars have M_i < -22


def _total_absolute_magnitude(
    points: Sequence[SEDPoint], members: Sequence[MemberChoice], redshift: Mapping[str, Any],
) -> tuple[float, str] | None:
    """Absolute AB magnitude of the total optical light (:func:`total_optical_flux`) at the adopted redshift
    (Planck18, no K-correction); None without a usable positive redshift or total magnitude."""
    z = _num(redshift.get("value"))
    if z is None or z <= S5_HVS1_Z or redshift.get("reliable") is False:
        return None
    total = total_optical_flux(points, members)
    if total is None:
        return None
    return absolute_magnitude(jy_to_ab_mag(total[0]), z), f"{total[1]} at z = {z:.4g}"


def nuclear_quasar_luminosity(
    points: Sequence[SEDPoint], members: Sequence[MemberChoice], redshift: Mapping[str, Any],
) -> tuple[bool | None, str]:
    """Can the nucleus of an EXTENDED counterpart be quasar-luminous (M_i < -22, Schneider et al. 2010) at the adopted
    redshift? (verdict, reason).

    The nuclear flux is bounded from above by the unsaturated PSF photometry of the counterpart (PS1 iMeanPSFMag, SDSS
    psfMag_i: the point-source flux includes the host light under the PSF), by the Gaia G of the counterpart (windowed
    photometry of the nucleus region), and by its total optical light (:func:`total_optical_flux`); the faintest
    bound is used (i and G taken alike, a band approximation). False when even that bound is fainter than M_i = -22
    (the nucleus cannot be a quasar: Cygnus A's PS1 iPSF at z = 0.056), True when it is brighter (a quasar nucleus is
    possible, e.g. a lensed quasar or a quasar in a bright host), None without a usable redshift (spectroscopic or of
    unstated technique, not flagged, cz > 1017 km/s) or magnitude (Planck18, no K-correction)."""
    z = _num(redshift.get("value"))
    if z is None or z <= S5_HVS1_Z or redshift.get("reliable") is False or redshift.get("kind") == "photo":
        return None, "no usable redshift"
    bounds: list[tuple[float, str]] = []
    ps1 = _member(members, "panstarrs_dr2")
    psf = _num(_ci_get(ps1.data, "iMeanPSFMag")) if ps1 is not None else None
    if psf is not None and PS1_SATURATION_MAG["i"] <= psf < 90:
        bounds.append((ab_mag_to_jy(psf)[0], f"PS1 iPSF = {psf:.2f}"))
    sdss = _member(members, "sdss")
    sdss_psf = _num(_ci_get(sdss.data, "psfMag_i")) if sdss is not None else None
    if sdss is not None and sdss_psf is not None and SDSS_SATURATION_MAG <= sdss_psf < 90 and not sdss_saturation(sdss.data):
        bounds.append((sdss_asinh_mag_to_jy(sdss_psf, "i")[0], f"SDSS psfMag_i = {sdss_psf:.2f}"))
    gaia_g = _point(points, "Gaia", "G")
    if _usable(gaia_g) and gaia_g is not None and gaia_g.magnitude is not None:
        # Gaia's windowed photometry of the counterpart within 1 arcsec of the target (the nucleus plus the host
        # light in the window; Cen A's obscured nucleus shows only a G = 21 knot).
        bounds.append((gaia_g.flux_jy, f"Gaia G = {gaia_g.magnitude:.2f}"))
    total = total_optical_flux(points, members)
    if total is not None:
        bounds.append((total[0], f"total light {total[1]}"))
    if not bounds:
        return None, "no unsaturated PSF or total optical magnitude"
    flux, label = min(bounds)
    abs_mag = absolute_magnitude(jy_to_ab_mag(flux), z)
    ok = abs_mag < QSO_ABS_MAG_LIMIT
    return ok, (f"nuclear M_i >= {abs_mag:.1f} ({label} at z = {z:.4g}), "
                + ("so a quasar nucleus is possible" if ok else f"fainter than the quasar limit {QSO_ABS_MAG_LIMIT:g}"))


def redshift_source_class(redshift: Mapping[str, Any], members: Sequence[MemberChoice]) -> tuple[str | None, str]:
    """Class of the catalogue entry whose redshift was adopted: the SIMBAD/NED member's type or the SDSS spectral
    class. (None, '') when unknown."""
    source = redshift.get("source")
    if source == "simbad":
        simbad = _member(members, "simbad")
        if simbad is not None:
            otype = _ci_get(simbad.data, "otype")
            return simbad_type_class(otype)[0], f"SIMBAD {simbad.source_id} '{otype}'"
    elif source == "ned":
        ned = _member(members, "ned")
        if ned is not None:
            ptype = _ci_get(ned.data, "prefphytype")
            return ned_type_class(ptype), f"NED {ned.source_id} '{ptype}'"
    elif source in {"sdss", "sdss_specobj"}:
        value = _num(redshift.get("value"))
        for cand in redshift.get("candidates") or []:
            if cand.get("source") == source and _same_z(_num(cand.get("value")), value) and cand.get("spec_class"):
                mapped = {"STAR": "star", "GALAXY": "galaxy", "QSO": "qso"}.get(str(cand["spec_class"]).upper())
                return mapped, f"SDSS spectrum class {cand['spec_class']}"
    return None, ""


def redshift_class_consistency(
    z: float, label: str, points: Sequence[SEDPoint], members: Sequence[MemberChoice],
    position: tuple[float, float] | None = None,
) -> tuple[bool | None, str]:
    """Is redshift ``z`` possible for an object of class ``label``? (verdict, reason); None when it cannot be told.

    star: |cz| within the fastest known field/hypervelocity star (S5-HVS1; anything near Sgr A*). galaxy/agn: possible
    for cz > 1017 km/s; impossible for a blueshift beyond 1017 km/s (no galaxy approaches that fast: Local Group and
    Virgo-cluster galaxies reach a few hundred km/s); cannot be told for |cz| <= 1017 km/s, where the Local Volume
    galaxies lie (M31 at cz = -300 km/s, M81, M82, NGC 253, NGC 4395, Cen A at 547 km/s): the S5-HVS1 speed is a
    ceiling on stellar velocities, not a floor on galaxy redshifts (SIMBAD: NGC 4395 z = 0.00110586, cz = 332 km/s).
    qso: extragalactic and quasar-luminous at z,
    M_i < -22 (Schneider et al. 2010) from the i-band magnitude of a point-like counterpart or the total optical light
    of an extended one (Planck18, no K-correction).
    """
    cz = z * C_KMS
    if label == "star":
        ok = abs(z) <= S5_HVS1_Z or near_galactic_centre(position)
        return ok, f"|cz| = {abs(cz):.0f} km/s {'<=' if ok else '>'} 1017 km/s (S5-HVS1) for a star"
    if label in {"galaxy", "agn"}:
        if z > S5_HVS1_Z:
            return True, f"cz = {cz:.0f} km/s > 1017 km/s: extragalactic"
        if z < -S5_HVS1_Z:
            return False, (f"cz = {cz:.0f} km/s: a blueshift beyond 1017 km/s, faster than any galaxy approaches "
                           "(Local Group and Virgo-cluster galaxies reach a few hundred km/s)")
        return None, (f"|cz| = {abs(cz):.0f} km/s <= 1017 km/s: possible for a Local Volume galaxy (M31 -300 km/s, "
                      "NGC 4395 +332 km/s), so it cannot be ruled out")
    if label != "qso":
        return None, f"class '{label}' does not constrain the redshift"
    if z <= S5_HVS1_Z:
        return False, (f"cz = {cz:.0f} km/s <= 1017 km/s: a quasar (M_i < -22) that near (within ~15 Mpc) would be "
                       "brighter than m_i ~ 9, far brighter than any known quasar (3C 273: V = 12.9)")
    extended, extent_reason = optical_extent(members)
    if extended:
        total = _total_absolute_magnitude(points, members, {"value": z, "reliable": True})
        if total is None:
            return None, f"no total optical magnitude of the extended counterpart ({extent_reason})"
        abs_mag, label_text = total[0], f"total optical light, {total[1]}"
    else:
        optical_i, optical_label, _rejected = select_optical_i(points, extended)
        if optical_i is None or optical_label == "Gaia G":
            return None, "no usable i-band magnitude"
        abs_mag = absolute_magnitude(jy_to_ab_mag(optical_i), z)
        label_text = f"m_i = {jy_to_ab_mag(optical_i):.2f} AB from {optical_label}"
    ok = abs_mag < QSO_ABS_MAG_LIMIT
    return ok, (f"M_i = {abs_mag:.1f} {'<' if ok else '>='} {QSO_ABS_MAG_LIMIT:g} at z = {z:.5g} ({label_text}): "
                + ("quasar-luminous" if ok else "too faint for a quasar"))


def resolve_discordant_redshift(
    redshift: Mapping[str, Any],
    points: Sequence[SEDPoint],
    members: Sequence[MemberChoice],
    gaia_extra: Mapping[str, Any] | None = None,
    *,
    position: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Settle discordant redshifts that no catalogue corroborates (``needs_resolution``, see :func:`choose_redshift`)
    with the classification from the redshift-independent evidence: the one group whose value is possible for that
    class (:func:`redshift_class_consistency`) while every other group's is impossible is adopted (``reliable`` True,
    corroborated by the classification) -- provided it holds a vetted value: a value whose quality could not be checked
    (its lookup failed) can reject the others but is never adopted. Otherwise the priority choice is kept with
    ``reliable`` False and ``conflict`` True (never a silent pick; the note says when the cross-check was incomplete).
    Other redshifts are returned unchanged."""
    out = dict(redshift)
    if not out.pop("needs_resolution", False):
        return out
    groups = [dict(g) for g in out.get("discordant") or []]
    out["discordant"] = groups
    extended, _reason = optical_extent(members)
    evidence = gather_evidence(points, members, out, gaia_extra, use_redshift=False, position=position)
    label, _confidence, _scores, tied = _score(evidence, EXTENDED_TIE_ORDER if extended else CLASSES)
    verdicts: list[tuple[dict[str, Any], bool | None, str]] = []
    if label != "unknown" and not tied:
        for group in groups:
            ok, why = redshift_class_consistency(float(group["value"]), label, points, members, position)
            group["consistency"] = why
            verdicts.append((group, ok, why))
    possible = [v for v in verdicts if v[1] is True]
    unvetted = [g for g in groups if g.get("vetted") is False]
    single = (bool(verdicts) and len(possible) == 1
              and all(v[1] is False for v in verdicts if v is not possible[0]))
    cand = _group_candidate(possible[0][0], out.get("candidates") or []) if single else None
    if single and cand is not None:
        chosen, _ok, why = possible[0]
        out.update({"value": cand["value"], "error": cand.get("error"), "kind": cand.get("kind"),
                    "source": cand.get("source"), "reliable": True})
        rejected = "; ".join(f"z = {float(g['value']):.5g} ({', '.join(g['sources'])}): {w}"
                             for g, ok, w in verdicts if g is not chosen)
        out["resolution"] = (f"{out.get('resolution')}; z = {float(chosen['value']):.5g} ({', '.join(chosen['sources'])}) "
                             f"adopted, consistent with the '{label}' classification from the redshift-independent "
                             f"evidence ({why}); rejected: {rejected}")
        return out
    if label == "unknown" or tied:
        reason = "the redshift-independent evidence gives no single class"
    elif single:
        reason = (f"only z = {float(possible[0][0]['value']):.5g} ({', '.join(possible[0][0]['sources'])}) is consistent "
                  f"with the '{label}' classification, but its catalogue quality could not be checked")
    elif not possible:
        reason = f"no value is consistent with the '{label}' classification"
    elif len(possible) > 1:
        reason = f"several values are consistent with the '{label}' classification"
    else:
        reason = f"the '{label}' classification cannot rule out the other values"
    details = "; ".join(f"z = {float(g['value']):.5g}: {w}" for g, _ok, w in verdicts)
    out["reliable"] = False
    out["conflict"] = True
    out["resolution"] = f"{out.get('resolution')}; unresolved: {reason}" + (f" ({details})" if details else "")
    incomplete = ""
    if unvetted:
        incomplete = ("; cross-check incomplete: " + ", ".join(
            f"z = {float(g['value']):.5g} ({', '.join(g['sources'])})" for g in unvetted)
            + " could not be vetted (catalogue quality unknown: its lookup failed or was skipped)")
    out["note"] = (f"discordant redshifts from different catalogues, none independently corroborated: "
                   f"z = {float(out['value']):.5g} ({out.get('source')}) is reported by priority only and marked "
                   f"unreliable (see redshift.discordant){incomplete}")
    return out


def classify(
    points: Sequence[SEDPoint],
    members: Sequence[MemberChoice],
    redshift: Mapping[str, Any],
    gaia_extra: Mapping[str, Any] | None = None,
    *,
    position: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Label + softmax scores over summed evidence weights; 'unknown' when no evidence carries weight.

    If the evidence (including the redshift rules, which only penalise 'star') still says 'star', the
    redshift-based rules are dropped (a large redshift of a star is a misassociation or catalogue error,
    and M_i computed from it is meaningless) and a conflicting redshift is reported in the evidence.
    Equal top scores are reported in ``tied`` and resolved explicitly: extended counterparts prefer galaxy/agn
    (unless the adopted redshift belongs to a quasar-typed entry and the total optical light at that redshift is
    quasar-luminous), else the class of the entry whose redshift was adopted, else 'unknown' (never an arbitrary
    fixed order).
    """
    extended, extent_reason = optical_extent(members)
    tie_order = EXTENDED_TIE_ORDER if extended else CLASSES
    evidence = gather_evidence(points, members, redshift, gaia_extra, position=position)
    label, confidence, scores, tied = _score(evidence, tie_order)
    conflict = False
    if label == "star" and _num(redshift.get("value")) is not None:
        evidence = gather_evidence(points, members, redshift, gaia_extra, use_redshift=False, position=position)
        label, confidence, scores, tied = _score(evidence, tie_order)
        if redshift_star_conflict(redshift, position):
            conflict = True
            z = float(redshift["value"])
            evidence.append(Evidence(
                f"redshift z = {z:.5g} ({redshift.get('source')}, {redshift.get('kind')}; cz = {z * C_KMS:.0f} km/s) conflicts "
                "with the stellar classification (|cz| > 1017 km/s, faster than the fastest known field/hypervelocity star "
                "S5-HVS1; heuristic): probable misassociation or erroneous catalogue redshift; redshift rules not applied", {}))
    if tied:
        z_class, z_why = redshift_source_class(redshift, members)
        total_abs = _total_absolute_magnitude(points, members, redshift)
        if extended and z_class == "qso" and "qso" in tied and label != "qso" and total_abs is not None \
                and total_abs[0] < QSO_ABS_MAG_LIMIT:
            # 'A galaxy at the quasar's redshift' would be brighter than the quasar luminosity threshold: the label
            # must agree with the adopted redshift (a lensed quasar, e.g. Q2237+030 at z = 1.695 with iKron = 14.2).
            label = "qso"
            why = (f"the adopted redshift belongs to a quasar-typed entry ({z_why}) and the total optical light would "
                   f"give M = {total_abs[0]:.1f} < {QSO_ABS_MAG_LIMIT:g} ({total_abs[1]}; Schneider et al. 2010 quasar "
                   "luminosity) for a galaxy at that redshift")
        elif extended:
            why = f"extended counterpart ({extent_reason}): host-galaxy classes preferred"
        elif z_class in tied:
            label = z_class
            why = f"the adopted redshift belongs to a {z_class}-typed entry ({z_why})"
        else:
            label = "unknown"
            why = "no evidence separates them"
        confidence = scores.get(label, 0.0)
        evidence.append(Evidence(f"scores tied between {', '.join(tied)}: '{label}' chosen by tie-break ({why})", {}))
    return {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "tied": tied,
        "evidence": [item.text for item in evidence],
        "evidence_detail": [{"text": item.text, "weights": item.weights} for item in evidence],
        "method": "softmax over summed rule weights (transparent heuristic, not a calibrated probability)",
        "redshift_conflict": conflict,
    }


# ---------------------------------------------------------------------------
# SED Assembly
# ---------------------------------------------------------------------------


def _reference_position(members: Sequence[MemberChoice], centre: tuple[float, float]) -> tuple[float, float]:
    for catalog in ("gaia_dr3", "panstarrs_dr2", "sdss", "simbad", "ned", "twomass_psc", "allwise"):
        member = _member(members, catalog)
        if member is not None and member.ra is not None and member.dec is not None:
            return member.ra, member.dec
    return centre


def _target_position(target: Mapping[str, Any]) -> tuple[float, float]:
    """The record's target (ra, dec) as finite ICRS degrees, or :class:`SEDInputError`."""
    if "ra" not in target or "dec" not in target:
        raise SEDInputError("record has no target position")
    ra, dec = _num(target.get("ra")), _num(target.get("dec"))
    if ra is None or dec is None or not (0.0 <= ra < 360.0) or not (-90.0 <= dec <= 90.0):
        raise SEDInputError(f"record target position is invalid: ra={target.get('ra')!r}, dec={target.get('dec')!r} "
                            "(finite degrees, 0 <= ra < 360, -90 <= dec <= 90 required)")
    return ra, dec


# Assumptions stated with every passport (see the module docstring for the references).
WISE_COLOUR_CORRECTION_ASSUMPTION = (
    "WISE flux densities use the SVO/Jarrett et al. 2011 zero points 309.54, 171.787, 31.674, 8.363 Jy, which are the "
    "WISE Explanatory Supplement IV.4.h Table 1 values for a flat spectrum (F_nu = constant); no colour correction "
    "is applied. For a Rayleigh-Jeans (stellar, F_nu ~ nu^2) spectrum the Wright et al. 2010 corrections "
    "(f_c(nu^0)/f_c(nu^2) = 0.9907/1.0084, 0.9935/1.0066, 0.9169/1.0088, 0.9905/1.0013) mean W1, W2, W3, W4 are "
    "1.8%, 1.3%, 10.0% and 1.1% too high; sources with a steeply rising mid-IR spectrum (F_nu ~ nu^-alpha, alpha >= 1) "
    "also need the ~8-10% W4 red-source reduction (Explanatory Supplement IV.4.h, Eq. 3)")


# Message of the cancellation of a lookup still running at the supplementary deadline (as opposed to the caller's
# own cancellation): :meth:`FilterCatalog.prefetch` records the former as an SVO timeout, the latter as nothing.
SUPPLEMENTARY_DEADLINE_CANCEL = "supplementary deadline"


def _mark_finished(finished: dict[str, float], key: str, _task: asyncio.Future[Any]) -> None:
    finished.setdefault(key, time.monotonic())


async def _run_supplementary(
    jobs: Mapping[str, tuple[str | None, Callable[[], Coroutine[Any, Any, Any]]]],
    *,
    deadline: float,
    backoff: HostBackoff,
    notes: list[str],
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the supplementary lookups concurrently within ONE overall deadline.

    ``jobs`` maps a key to (back-off host or None, coroutine factory); ``labels`` optionally names the host a job
    talks to in the notes when it is not backed off here (the SVO filter lookups have their own retry policy).
    Hosts in back-off are skipped; lookups still running at the deadline are cancelled (with the message
    :data:`SUPPLEMENTARY_DEADLINE_CANCEL`; their host enters back-off); failures are reported in ``notes`` with the
    exception type, host and elapsed time, and costly ones (timeouts, rate limits, refused connections, slow failures:
    :func:`backoff_seconds_for`) put their host in back-off; a success clears its host's back-off. If the caller is
    cancelled (or anything else interrupts the wait), every lookup is cancelled and awaited before the exception
    propagates: no lookup outlives the call (or its HTTP client). Returns ``{key: result}`` for the lookups that
    succeeded.
    """
    started = time.monotonic()
    finished: dict[str, float] = {}
    tasks: dict[str, asyncio.Task[Any]] = {}
    hosts: dict[str, str | None] = {}
    for key, (host, factory) in jobs.items():
        why = backoff.pending(host)
        if why is not None:
            notes.append(f"{key} lookup failed: skipped, {host} {why}")
            continue
        task = asyncio.ensure_future(factory())
        task.add_done_callback(functools.partial(_mark_finished, finished, key))
        tasks[key], hosts[key] = task, host
    if not tasks:
        return {}
    try:
        _done, pending = await asyncio.wait(set(tasks.values()), timeout=deadline)
        for task in pending:
            task.cancel(msg=SUPPLEMENTARY_DEADLINE_CANCEL)
        if pending:
            await asyncio.wait(pending)
    except BaseException:
        for task in tasks.values():
            task.cancel()
        # Shielded from a repeated cancellation; the lookups are already cancelled, so this returns promptly.
        await asyncio.shield(asyncio.gather(*tasks.values(), return_exceptions=True))
        raise
    results: dict[str, Any] = {}
    failed_hosts: set[str | None] = set()
    succeeded_hosts: set[str | None] = set()
    for key, task in tasks.items():
        host = hosts[key]
        where = host or (labels or {}).get(key) or "local"
        if task in pending or task.cancelled():
            notes.append(f"{key} lookup failed: TimeoutError (no answer from {where} within the {deadline:g} s "
                         "supplementary deadline; cancelled)")
            backoff.record_failure(host, f"TimeoutError after the {deadline:g} s deadline")
            failed_hosts.add(host)
            continue
        exc = task.exception()
        if exc is not None:
            elapsed = finished.get(key, time.monotonic()) - started
            detail = str(exc).strip()
            notes.append(f"{key} lookup failed: {type(exc).__name__} ({where} after {elapsed:.1f} s)"
                         + (f": {detail}" if detail else ""))
            wait = backoff_seconds_for(backoff, exc, elapsed)
            if wait is not None:
                if isinstance(exc, SEDUpstreamError) and exc.status_code is not None:
                    why = f"HTTP {exc.status_code}" + (f" (Retry-After {exc.retry_after:.0f} s)"
                                                      if exc.retry_after is not None else "")
                else:
                    why = f"{type(exc).__name__} after {elapsed:.0f} s"
                backoff.record_failure(host, why, wait)
                failed_hosts.add(host)
            continue
        succeeded_hosts.add(host)
        results[key] = task.result()
    # A host that answered clears a back-off recorded meanwhile (e.g. by a concurrent request), unless another lookup
    # of this run on the same host failed costly: that failure is the newer evidence.
    for host in succeeded_hosts - failed_hosts:
        backoff.record_success(host)
    return results


async def _with_sesame_aliases(
    rec: dict[str, Any], client: httpx.AsyncClient, *, deadline: float, backoff: HostBackoff, notes: list[str],
    timeout: float,
) -> dict[str, Any]:
    """``rec`` with ``resolved_object.aliases`` filled from Sesame '-oxpI' when the record was resolved by name without
    asking for identifiers (its resolver endpoint lacks the 'I' option and it lists no alias). The canonical name
    (Sesame's main identifier) is resolved again, else the query; the result is used only when it is the same object
    (within 1 arcsec of the recorded resolution). The lookup gets at most half of the supplementary ``deadline`` (a
    hanging Sesame must not starve the other lookups; its host is backed off like theirs). Failures are reported in
    ``notes`` and leave the record unchanged."""
    resolved = rec.get("resolved_object")
    if not isinstance(resolved, Mapping) or resolved.get("aliases"):
        return rec
    endpoint = (resolved.get("resolver_metadata") or {}).get("endpoint")
    if endpoint and sesame_asks_aliases(endpoint):
        return rec  # identifiers were asked for: the object has none besides its names
    name = str(resolved.get("canonical_name") or resolved.get("query") or "").strip()
    if not name:
        return rec
    url = sesame_aliases_endpoint()
    outcome = await _run_supplementary(
        {"aliases": (_host(url), lambda: resolve_name(name, client, endpoint=url))},
        deadline=min(deadline / 2.0, timeout), backoff=backoff, notes=notes)  # never all of the lookups' budget
    info = outcome.get("aliases")
    if not isinstance(info, dict):
        return rec
    fresh = info["resolved"]
    ra, dec = _num(resolved.get("ra_deg")), _num(resolved.get("dec_deg"))
    if ra is not None and dec is not None and haversine_arcsec(ra, dec, fresh["ra_deg"], fresh["dec_deg"]) > 1.0:
        notes.append(f"identifiers of '{name}' not used: Sesame '-oxpI' resolved it {haversine_arcsec(ra, dec, fresh['ra_deg'], fresh['dec_deg']):.1f} "
                     "arcsec from the record's resolution")
        return rec
    aliases = [str(a) for a in fresh.get("aliases") or []]
    if not aliases:
        return rec
    out = dict(rec)
    out["resolved_object"] = {**dict(resolved), "aliases": aliases}
    notes.append(f"identifiers of '{name}' fetched from Sesame '-oxpI' ({len(aliases)} aliases) for identification by "
                 "designation: the record's resolution" + (f" ({endpoint})" if endpoint else "") + " did not list them")
    return out


async def sed_from_record(
    record: UnifiedRecord | Mapping[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
    filters: FilterCatalog | None = None,
    supplementary: bool = True,
    name: str | None = None,
    radius_arcsec: float | None = None,
    deadline_seconds: float | None = None,
    backoff: HostBackoff | None = None,
    background_filter_fetches: bool | None = None,
) -> dict[str, Any]:
    """Build the object passport (SED + classification + redshift) from a crossmatch record.

    Members are taken per catalog from every crossmatch group (:func:`select_record_members`). ``supplementary``
    enables the extra lookups (SVO filter metadata for filters not cached yet, Gaia DR3 errors/DSC probabilities/
    BP-RP excess, SDSS SpecObj/Photoz near the object, the SDSS member's all-band photometry/flags/zWarning, SIMBAD
    redshift nature/quality and flux errors). They share ONE deadline (``deadline_seconds``, default
    :func:`default_supplementary_deadline`) and a per-host back-off (``backoff``, default :func:`default_backoff`);
    each failure is reported in ``notes`` and the SED is still built (embedded filter table, missing Gaia errors,
    member columns only, redshift kinds/reliabilities left unknown rather than guessed). ``radius_arcsec`` is reported
    as the search radius when the record's provenance does not state it.

    A record resolved by name without the object's identifiers (the app's own crossmatch resolves with Sesame '-oxp',
    which lists none) gets them first from Sesame '-oxpI' (:func:`_with_sesame_aliases`, within the same deadline), so
    survey rows are still identified by designation. ``background_filter_fetches`` lets SVO fetches still running at
    the filter deadline finish (and be cached) after the call; the default allows it only on a caller-supplied
    client, which is assumed to stay open (pass False when the client is closed right after the call).
    """
    rec = _record_dict(record)
    validate_record_shape(rec)
    target = dict(rec.get("target") or {})
    centre = _target_position(target)
    filters = filters or default_filter_catalog()
    deadline = deadline_seconds if deadline_seconds is not None else default_supplementary_deadline()
    if not math.isfinite(deadline) or deadline <= 0:
        raise SEDInputError("deadline_seconds must be finite and > 0")
    backoff = backoff or default_backoff()
    request_timeout = min(SUPPLEMENTARY_REQUEST_TIMEOUT_SECONDS, deadline)
    if background_filter_fetches is None:
        background_filter_fetches = client is not None
    notes: list[str] = []
    owned = client is None and supplementary
    active = client or (await _owned_client(request_timeout) if supplementary else None)
    gaia_extra: dict[str, Any] | None = None
    sdss_extra: dict[str, Any] | None = None
    simbad_extra: dict[str, Any] | None = None
    sdss_candidates: list[dict[str, Any]] = []
    sdss_notes: list[str] = []  # partial SkyServer failures (a Photoz query refused after a successful SpecObj query)
    svo_failures: dict[str, str] = {}
    filters_failed = False
    try:
        started = time.monotonic()
        if supplementary and active is not None:
            rec = await _with_sesame_aliases(rec, active, deadline=deadline, backoff=backoff, notes=notes,
                                             timeout=request_timeout)
        remaining = max(deadline - (time.monotonic() - started), 1e-3)
        members, group, selection_notes = select_record_members(rec, centre)
        if not group:
            notes.append("no catalog source within the search radius")
        notes.extend(selection_notes)
        for member in members:
            notes.extend(member.notes)
        gaia, sdss, simbad = _member(members, "gaia_dr3"), _member(members, "sdss"), _member(members, "simbad")
        ref_ra, ref_dec = _reference_position(members, centre)
        needed = required_filters(members)
        if supplementary and active is not None:
            http = active
            sdss_lock = SkyServerLock()  # one SkyServer connection at a time; one round of connect retries
            jobs: dict[str, tuple[str | None, Callable[[], Coroutine[Any, Any, Any]]]] = {
                # SVO: own deadline and retry policy (no host back-off here); fetches outlive the call only on a
                # client that stays open (background_filter_fetches).
                "filters": (None, lambda: filters.prefetch(needed, http, background=bool(background_filter_fetches))),
            }
            if gaia is not None and re.fullmatch(r"\d{1,20}", gaia.source_id):
                gaia_id = gaia.source_id
                jobs["gaia"] = (_host(GAIA_TAP_URL), lambda: fetch_gaia_extra(http, gaia_id, timeout=request_timeout))
            if group:
                jobs["sdss"] = (_host(SDSS_SQL_URL), lambda: fetch_sdss_redshifts(
                    http, ref_ra, ref_dec, include_photoz=sdss is None, timeout=request_timeout, lock=sdss_lock,
                    notes=sdss_notes))
            if sdss is not None and re.fullmatch(r"\d{1,20}", sdss.source_id):
                sdss_id = sdss.source_id
                jobs["sdss_object"] = (_host(SDSS_SQL_URL), lambda: fetch_sdss_object(
                    http, sdss_id, timeout=request_timeout, lock=sdss_lock))
            if simbad is not None and simbad.source_id:
                simbad_id = simbad.source_id
                jobs["simbad"] = (_host(SIMBAD_TAP_URL), lambda: fetch_simbad_extra(
                    http, simbad_id, timeout=request_timeout))
            outcome = await _run_supplementary(jobs, deadline=remaining, backoff=backoff, notes=notes,
                                               labels={"filters": _host(filters.endpoint) or filters.endpoint})
            got_filters, got_gaia = outcome.get("filters"), outcome.get("gaia")
            got_sdss, got_object, got_simbad = outcome.get("sdss"), outcome.get("sdss_object"), outcome.get("simbad")
            filters_failed = "filters" not in outcome
            if isinstance(got_filters, dict):
                svo_failures = got_filters
            if isinstance(got_gaia, dict):
                gaia_extra = got_gaia
            if isinstance(got_sdss, list):
                sdss_candidates = got_sdss
            if isinstance(got_object, dict):
                sdss_extra = got_object
            if isinstance(got_simbad, dict):
                simbad_extra = got_simbad
        # Without supplementary lookups, filters come from the memory/disk cache or the embedded table.
    finally:
        if owned and active is not None:
            await _close_owned(active)
    if filters_failed and not filters.offline:
        # The whole SVO job failed or was cancelled at the deadline: name the filters that fall back.
        for fid in filters.unresolved(needed):
            svo_failures.setdefault(fid, filters.errors.get(fid)
                                    or "no SVO answer before the filters lookup ended (see its note)")
    notes.extend(sdss_notes)
    for fid, err in sorted(svo_failures.items()):
        notes.append(f"SVO FPS lookup for {fid} failed ({err}); embedded SVO values used")
    if sdss is not None and sdss_extra:
        # All-band photometry and per-band SATURATED flags: every rule reading the SDSS member (morphology,
        # aperture, saturation) sees them.
        sdss.data.update({k: v for k, v in sdss_extra.items() if k != "spectra"})

    points = extract_points(members, filters, gaia_extra, sdss_extra=sdss_extra, simbad_extra=simbad_extra)
    member_candidates = member_redshift_candidates(members, sdss_extra=sdss_extra, simbad_extra=simbad_extra,
                                                   specobj_candidates=sdss_candidates)
    redshift = resolve_discordant_redshift(choose_redshift(sdss_candidates + member_candidates), points, members,
                                           gaia_extra, position=(ref_ra, ref_dec))
    if redshift.get("discordant"):
        notes.append(f"discordant redshifts cross-checked: {redshift.get('resolution')}")
    classification = classify(points, members, redshift, gaia_extra, position=(ref_ra, ref_dec))
    if redshift.get("value") is not None and classification["label"] == "star":
        if classification.get("redshift_conflict"):
            redshift["reliable"] = False
            redshift["conflict"] = True
            redshift["note"] = (f"|cz| = {abs(float(redshift['value'])) * C_KMS:.0f} km/s exceeds the fastest known field/"
                                "hypervelocity star (S5-HVS1, 1017 km/s) but the object is classified as a star: probable "
                                "misassociation or erroneous catalogue redshift (heuristic; not applied near Sgr A*)")
            notes.append("redshift conflicts with the stellar classification (see redshift.note)")
        elif not redshift.get("conflict"):
            redshift["note"] = "Doppler shift from the radial velocity of a Galactic star, not a cosmological redshift"

    resolved = rec.get("resolved_object") or None
    filter_meta = {}
    for fid in sorted({p.filter_id for p in points if p.filter_id}):
        filter_meta[fid] = filters.info(fid).as_dict()
    failures = list(rec.get("failures") or [])
    record_radius = _num((rec.get("provenance") or {}).get("query_radius_arcsec"))
    used_groups = sorted({m.group_id for m in members if m.used and m.group_id is not None})
    return {
        "target": {
            "ra": centre[0], "dec": centre[1], "epoch": target.get("epoch"),
            "radius_arcsec": record_radius if record_radius is not None else _num(radius_arcsec),
            "name": name or (resolved or {}).get("canonical_name"),
            "resolved_object": resolved,
        },
        "points": [p.as_dict() for p in points],
        "classification": classification,
        "redshift": redshift,
        "members": [m.summary() for m in members],
        "group_id": group.get("group_id") if group else None,
        "group_ids": used_groups,
        "reference_position": {"ra": ref_ra, "dec": ref_dec},
        "filters": filter_meta,
        "gaia_extra": gaia_extra,
        "sdss_extra": sdss_extra,
        "simbad_extra": simbad_extra,
        "failures": failures,
        "notes": notes,
        "assumptions": [
            ("AB magnitudes: 0 mag = 3631 Jy (Oke & Gunn 1983); SDSS asinh magnitudes (Lupton et al. 1999) with "
             "u_AB = u_SDSS - 0.04, z_AB = z_SDSS + 0.02; one SDSS aperture (psfMag, or modelMag for galaxies) for all bands"),
            "Vega zero points and effective wavelengths from the SVO Filter Profile Service",
            WISE_COLOUR_CORRECTION_ASSUMPTION,
            (f"X-ray flux densities at the logarithmic band centre for a photon index {XRAY_PHOTON_INDEX:g} power law; "
             "Chandra/XMM from observed band fluxes"),
            ROSAT_PSPC_ECF_ASSUMPTION,
            "optical/IR fluxes are not corrected for Galactic extinction",
            "quality-flagged points (quality_warning) are shown but excluded from the classification",
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }


# Sesame with the 'I' option: the answer lists every identifier of the object (<alias>), which
# :func:`select_record_members` uses to identify survey rows by designation. Without it (providers' default '-oxp')
# the aliases are always empty and e.g. '2MASSI J0415195-093506' (SIMBAD's main identifier of a T8 dwarf) would not
# name its own 2MASS row '2MASS J04151954-0935066'. This is the default; the endpoint actually used is the configured
# resolver (Settings.resolver_endpoint, $SESAME_ENDPOINT, e.g. a CDS mirror) with the 'I' option added
# (:func:`sesame_aliases_endpoint`).
SESAME_ALIASES_ENDPOINT = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV"
_SESAME_OPTIONS = re.compile(r"-o[A-Za-z0-9]*")


def sesame_asks_aliases(endpoint: Any) -> bool:
    """Does a Sesame endpoint URL request all identifiers (an '-o...' output-option segment containing 'I')?"""
    from urllib.parse import urlsplit

    segments = urlsplit(str(endpoint or "")).path.split("/")
    return any(_SESAME_OPTIONS.fullmatch(seg) and "I" in seg[2:] for seg in segments)


def sesame_aliases_endpoint(endpoint: str | None = None) -> str:
    """The Sesame endpoint for SED name resolution: ``endpoint``, else the app's configured resolver
    (``models.Settings().resolver_endpoint``, i.e. $SESAME_ENDPOINT, default CDS '-oxp'), with the 'I' output option
    (all identifiers) added to its '-o...' path segment ('.../nph-sesame/-oxp/SNV' -> '.../nph-sesame/-oxpI/SNV');
    '-oxpI' is inserted after 'nph-sesame' when the URL has no option segment. A URL that is not recognisably Sesame's
    (a local test resolver) is used as configured."""
    from urllib.parse import urlsplit, urlunsplit

    if endpoint is None:
        try:
            from models import Settings

            endpoint = Settings().resolver_endpoint
        except (ValueError, TypeError):  # invalid environment: the documented default
            return SESAME_ALIASES_ENDPOINT
    parts = urlsplit(endpoint)
    segments = parts.path.split("/")
    for index, segment in enumerate(segments):
        if _SESAME_OPTIONS.fullmatch(segment):
            if "I" not in segment[2:]:
                segments[index] = segment + "I"
            break
    else:
        at = next((i for i, s in enumerate(segments) if s.startswith("nph-sesame")), None)
        if at is None:
            return endpoint
        segments.insert(at + 1, "-oxpI")
    return urlunsplit(parts._replace(path="/".join(segments)))


async def resolve_name(name: str, client: httpx.AsyncClient | None = None, *, endpoint: str | None = None) -> dict[str, Any]:
    """Resolve a name with CDS Sesame (with all identifiers, :func:`sesame_aliases_endpoint` of ``endpoint`` or of the
    configured resolver); returns target kwargs (ra, dec, epoch, pm) and the resolution (``resolved['aliases']``: the
    object's other names)."""
    from providers import SesameResolver

    owned = client is None
    active = client or await _owned_client(30.0)
    try:
        resolved = await SesameResolver(active, endpoint=sesame_aliases_endpoint(endpoint)).resolve(name)
    finally:
        if owned:
            await _close_owned(active)
    target = resolved_target(resolved)
    return {
        "ra": target.ra, "dec": target.dec, "epoch": target.epoch,
        "pm_ra_masyr": target.pm_ra_masyr, "pm_dec_masyr": target.pm_dec_masyr,
        "resolved": resolved.as_dict(),
    }


async def build_sed(
    ra: float | None = None,
    dec: float | None = None,
    *,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    name: str | None = None,
    service: Any | None = None,
    client: httpx.AsyncClient | None = None,
    filters: FilterCatalog | None = None,
    supplementary: bool = True,
) -> dict[str, Any]:
    """Crossmatch a position or a resolved name (not both) and return its SED passport (see :func:`sed_from_record`)."""
    radius = validate_radius(radius_arcsec)
    try:
        _check_position_arguments(name, ra, dec)
    except ValueError as exc:
        raise SEDInputError(str(exc)) from exc
    owned = client is None
    active = client or await _owned_client(60.0)
    try:
        epoch = pm_ra = pm_dec = None
        resolved = None
        if name:
            info = await resolve_name(name, active)
            ra, dec, epoch = info["ra"], info["dec"], info["epoch"]
            pm_ra, pm_dec, resolved = info["pm_ra_masyr"], info["pm_dec_masyr"], info["resolved"]
        if ra is None or dec is None:
            raise SEDInputError(f"name {name!r} resolved without coordinates")
        target = validate_target(ra, dec)
        if service is None:
            from main import build_service

            service = build_service(client=active)
        record = await service.crossmatch(target.ra, target.dec, radius_arcsec=radius, epoch=epoch,
                                          pm_ra_masyr=pm_ra, pm_dec_masyr=pm_dec)
        if resolved is not None:
            record.resolved_object = resolved
        rec = record.as_dict()
        stats = (rec.get("provenance") or {}).get("catalog_stats") or {}
        answered = [n for n, s in stats.items() if s.get("status") in {"success", "empty"}]
        if stats and not answered:
            failures = list(rec.get("failures") or [])
            raise SEDUpstreamError("every catalog query failed: " + "; ".join(
                f"{f.get('catalog')}: {f.get('error_type')}" for f in failures), failures)
        # A client created here is closed right after: SVO fetches must not outlive the call on it.
        return await sed_from_record(rec, client=active, filters=filters, supplementary=supplementary, name=name,
                                     radius_arcsec=radius, background_filter_fetches=not owned)
    finally:
        if owned:
            await _close_owned(active)


def validate_radius(radius_arcsec: Any) -> float:
    radius = _num(radius_arcsec)
    if radius is None or radius <= 0 or radius > MAX_RADIUS_ARCSEC:
        raise SEDInputError(f"radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}]")
    return radius


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# One colour + marker per facility (colour-blind-safe Okabe-Ito / Tol hues), so every legend entry maps to its
# points; facilities not listed cycle through the fallback palette.
_FACILITY_STYLE: dict[str, tuple[str, str]] = {
    "Gaia DR3": ("#0072B2", "o"), "SDSS": ("#56B4E9", "s"), "Pan-STARRS1": ("#009E73", "D"),
    "SIMBAD (literature)": ("#999999", "P"), "GALEX": ("#CC79A7", "^"), "2MASS": ("#E69F00", "p"),
    "WISE": ("#D55E00", "<"), "FIRST (VLA)": ("#882255", "o"), "NVSS (VLA)": ("#AA4499", "s"),
    "VLASS (VLA)": ("#332288", "D"), "LoTSS (LOFAR)": ("#117733", "^"), "ROSAT PSPC": ("#44AA99", "o"),
    "Chandra ACIS": ("#DDCC77", "s"), "XMM-Newton EPIC": ("#661100", "D"),
}
_FALLBACK_STYLES: tuple[tuple[str, str], ...] = (("#000000", "o"), ("#6699CC", "s"), ("#888888", "D"))


def _facility_style(facility: str, extra: dict[str, tuple[str, str]]) -> tuple[str, str]:
    if facility in _FACILITY_STYLE:
        return _FACILITY_STYLE[facility]
    if facility not in extra:
        extra[facility] = _FALLBACK_STYLES[len(extra) % len(_FALLBACK_STYLES)]
    return extra[facility]


def plot_sed(sed: Mapping[str, Any], path: str | os.PathLike[str], *, dpi: int = 120) -> Path:
    """Write a log-log nu*F_nu vs wavelength PNG.

    One colour/marker per facility; filled symbols are measurements used by the classification, open symbols are
    quality-flagged points (shown, not trusted), downward arrows are upper limits. The legend is built from proxy
    artists, so its symbols never depend on the flag state of a facility's first point.
    """
    # An explicit Figure + Agg canvas: never touches pyplot or the caller's global backend (notebooks).
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    points = list(sed.get("points") or [])
    fig = Figure(figsize=(8.5, 5.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    extra: dict[str, tuple[str, str]] = {}
    facilities: list[str] = []
    any_flagged = any_limit = False
    for p in points:
        facility = str(p.get("facility"))
        if facility not in facilities:
            facilities.append(facility)
        color, marker = _facility_style(facility, extra)
        x, y = p["wavelength_um"], p["nu_fnu_erg_s_cm2"]
        flagged = bool(p.get("quality_warning"))
        any_flagged |= flagged
        if p.get("is_upper_limit"):
            any_limit = True
            ax.errorbar([x], [y], yerr=[[0.4 * y], [0.0]], uplims=True, fmt=marker, color=color, ms=5,
                        mfc="none" if flagged else color, alpha=0.8)
        else:
            err = p.get("nu_fnu_err_erg_s_cm2")
            yerr = None if err is None else [[min(err, 0.999 * y)], [err]]
            ax.errorbar([x], [y], yerr=yerr, fmt=marker, color=color, ms=6, capsize=2,
                        mfc="none" if flagged else color, alpha=0.65 if flagged else 1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength [micron]")
    ax.set_ylabel(r"$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]")
    cls = sed.get("classification") or {}
    z = (sed.get("redshift") or {}).get("value")
    tgt = sed.get("target") or {}
    title = tgt.get("name") or f"RA {tgt.get('ra', 0):.5f}  Dec {tgt.get('dec', 0):+.5f}"
    title += f" | {cls.get('label', '?')} ({cls.get('confidence', 0):.2f})"
    if z is not None:
        title += f" | z = {z:.4g} ({(sed.get('redshift') or {}).get('kind')})"
    ax.set_title(title, fontsize=10)
    if points:
        handles = [Line2D([], [], color=_facility_style(f, extra)[0], marker=_facility_style(f, extra)[1],
                          linestyle="none", ms=6, label=f) for f in facilities]
        if any_flagged:
            handles.append(Line2D([], [], color="#444444", marker="o", mfc="none", linestyle="none", ms=6,
                                  label="open: quality-flagged (not used)"))
        if any_limit:
            handles.append(Line2D([], [], color="#444444", marker="v", linestyle="none", ms=6, label="upper limit"))
        ax.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "no photometry", transform=ax.transAxes, ha="center")
    ax.grid(True, which="both", alpha=0.25)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    return out


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------


class SEDRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ra: float | None = Field(default=None, ge=0.0, lt=360.0)
    dec: float | None = Field(default=None, ge=-90.0, le=90.0)
    radius_arcsec: float = Field(default=DEFAULT_RADIUS_ARCSEC, gt=0.0, le=MAX_RADIUS_ARCSEC)
    name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("ra", "dec", "radius_arcsec", mode="before")
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # Pydantic's lax mode turns JSON true/false into 1.0/0.0: a boolean is never a coordinate or a radius.
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")  # noqa: TRY004 (pydantic needs ValueError -> 422)
        return value

    @model_validator(mode="after")
    def _need_position(self) -> SEDRequest:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        _check_position_arguments(self.name, self.ra, self.dec)
        return self


def _check_position_arguments(name: str | None, ra: float | None, dec: float | None) -> None:
    """Exactly one way to give the target: a name, or both ra and dec (never both: which one wins would be a guess)."""
    has_coords = ra is not None or dec is not None
    if name and has_coords:
        raise ValueError("give either name or ra/dec, not both")
    if not name and (ra is None or dec is None):
        raise ValueError("Either name or both ra and dec are required")


class SEDPointModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    band: str
    facility: str
    catalog: str
    source_id: str
    wavelength_um: float
    frequency_hz: float
    flux_jy: float
    flux_err_jy: float | None
    nu_fnu_erg_s_cm2: float
    is_upper_limit: bool


class ClassificationModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    confidence: float
    scores: dict[str, float]
    evidence: list[str]


class RedshiftModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: float | None
    error: float | None
    kind: Literal["spec", "photo"] | None
    source: str | None


class SEDResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: dict[str, Any]
    points: list[SEDPointModel]
    classification: ClassificationModel
    redshift: RedshiftModel
    members: list[dict[str, Any]]


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (NaN/Infinity echoed back in validation errors) by strings, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class _SEDRoute(APIRoute):
    """APIRoute whose 422 validation errors stay JSON-serialisable.

    FastAPI's default handler echoes the offending input; a JSON body such as ``{"ra": NaN}`` (Python's json
    accepts NaN/Infinity) then makes the error response itself unserialisable and the client gets a 500. This
    route class (FastAPI "custom APIRoute class" pattern) answers 422 with non-finite inputs rendered as strings.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError as exc:
                return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(
                    _json_safe(list(exc.errors()))))})

        return handler


router = APIRouter(prefix="/api/v1", tags=["sed"], route_class=_SEDRoute)


async def _sed_for_request(request: Request, *, ra: float | None, dec: float | None, radius: float, name: str | None) -> dict[str, Any]:
    state = request.app.state
    client = getattr(state, "client", None)
    service = getattr(state, "service", None)
    filters = getattr(state, "sed_filters", None)
    try:
        return await build_sed(ra, dec, radius_arcsec=radius, name=name, service=service, client=client, filters=filters)
    except ObjectResolutionError as exc:
        # 404 unknown name, 503 resolver down, 502 unusable resolver answer, 422 empty name.
        status = resolution_failure_status(exc)
        raise HTTPException(status_code=status, detail=f"Name resolution failed: {exc}",
                            headers={"Retry-After": "30"} if status == 503 else None) from exc
    except (SEDInputError, InvalidCoordinateError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SEDUpstreamError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream archives failed: {exc}") from exc
    except SEDFilterError as exc:  # server-side: SVO metadata unusable (never the client's fault)
        raise HTTPException(status_code=502, detail=f"Filter metadata unavailable: {exc}") from exc
    except SEDError as exc:
        raise HTTPException(status_code=500, detail=f"SED construction failed: {type(exc).__name__}: {exc}") from exc
    except AstroSearchError as exc:  # CatalogUnavailableError, ResponseParseError, ... raised by the service
        raise HTTPException(status_code=502, detail=f"Upstream failure: {type(exc).__name__}: {exc}") from exc
    except (httpx.HTTPError, TimeoutError) as exc:  # asyncio.TimeoutError is TimeoutError (Python >= 3.11)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {type(exc).__name__}: {exc}") from exc


@router.get("/sed", response_model=SEDResponse)
async def get_sed(
    request: Request,
    ra: float | None = Query(default=None, ge=0.0, lt=360.0, description="Right ascension (ICRS deg)"),
    dec: float | None = Query(default=None, ge=-90.0, le=90.0, description="Declination (ICRS deg)"),
    radius_arcsec: float = Query(default=DEFAULT_RADIUS_ARCSEC, gt=0.0, le=MAX_RADIUS_ARCSEC),
    name: str | None = Query(default=None, min_length=1, max_length=200, description="Object name (CDS Sesame)"),
) -> dict[str, Any]:
    """SED passport (photometry in Jy, classification, redshift) for a position or name (not both)."""
    if name is not None and not name.strip():
        raise HTTPException(status_code=422, detail="name must not be blank")
    try:
        _check_position_arguments(name, ra, dec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await _sed_for_request(request, ra=ra, dec=dec, radius=radius_arcsec, name=name)


@router.post("/sed", response_model=SEDResponse)
async def post_sed(request: Request, body: SEDRequest) -> dict[str, Any]:
    """SED passport for ``{ra, dec, radius_arcsec}`` or ``{name, radius_arcsec}`` (a name together with ra/dec is 422)."""
    return await _sed_for_request(request, ra=body.ra, dec=body.dec, radius=body.radius_arcsec, name=body.name)


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------


def format_sed_table(sed: Mapping[str, Any]) -> str:
    """Plain-text passport: one row per point with its flags (UL = upper limit, Q = quality warning, never used
    by the classification), then the warnings of flagged points, the classification evidence and the redshift."""
    lines = []
    tgt = sed.get("target") or {}
    ra, dec = _num(tgt.get("ra")), _num(tgt.get("dec"))
    lines.append(f"Target: {tgt.get('name') or ''} RA={ra if ra is None else f'{ra:.6f}'} "
                 f"Dec={dec if dec is None else f'{dec:+.6f}'} radius={tgt.get('radius_arcsec')}\"")
    lines.append(f"{'#':>3} {'facility':<22}{'band':<12}{'lambda[um]':>12}{'F_nu[Jy]':>13}{'err[Jy]':>12}"
                 f"{'nuFnu[cgs]':>13}  flags")
    flagged: list[tuple[int, Mapping[str, Any]]] = []
    for idx, p in enumerate(sed.get("points") or [], start=1):
        err = p.get("flux_err_jy")
        flags = " ".join(f for f, on in (("UL", p.get("is_upper_limit")), ("Q", p.get("quality_warning"))) if on)
        lines.append(f"{idx:>3} {p['facility']:<22}{p['band']:<12}{p['wavelength_um']:>12.5g}{p['flux_jy']:>13.5g}"
                     f"{(f'{err:.3g}' if err is not None else '-'):>12}{p['nu_fnu_erg_s_cm2']:>13.4g}  {flags}")
        if p.get("quality_warning"):
            flagged.append((idx, p))
    lines.append("Flags: UL = upper limit; Q = quality warning (shown, not used by the classification)")
    for idx, p in flagged:
        for note in p.get("warnings") or ["quality warning (reason not recorded)"]:
            lines.append(f"  Q #{idx} {p['facility']} {p['band']}: {note}")
    cls = sed.get("classification") or {}
    lines.append(f"Classification: {cls.get('label')} (confidence {cls.get('confidence')}) scores={cls.get('scores')}")
    for item in cls.get("evidence") or []:
        lines.append(f"  - {item}")
    z = sed.get("redshift") or {}
    lines.append(f"Redshift: {z.get('value')} +/- {z.get('error')} ({z.get('kind')}, {z.get('source')})")
    if z.get("note"):
        lines.append(f"  ({z['note']})")
    for note in sed.get("notes") or []:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def run_cli(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch sed``; returns a process exit code."""
    if args.name and (args.ra is not None or args.dec is not None):
        print("Error: specify --name or both --ra and --dec, not both.")
        return 2
    if not args.name and (args.ra is None or args.dec is None):
        print("Error: specify --name or both --ra and --dec.")
        return 2
    try:
        sed = asyncio.run(build_sed(args.ra, args.dec, radius_arcsec=args.radius, name=args.name))
    except (ValueError, InvalidCoordinateError, ObjectResolutionError, SEDError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(sed, indent=2, default=str))
    else:
        print(format_sed_table(sed))
    if args.plot:
        try:
            out = plot_sed(sed, args.plot)
        except ImportError:
            print("Error: matplotlib is required for --plot (pip install matplotlib)")
            return 1
        except (OSError, ValueError) as exc:  # unwritable path / unsupported image format
            print(f"Error: cannot write plot {args.plot!r}: {exc}")
            return 1
        print(f"SED plot written to {out}")
    return 0


def register_cli(subparsers: Any) -> None:
    """Add the ``sed`` subcommand to an argparse subparsers object."""
    parser = subparsers.add_parser("sed", help="Multi-wavelength SED, classification and redshift for one object")
    parser.add_argument("--ra", type=float, help="Right ascension in degrees (ICRS)")
    parser.add_argument("--dec", type=float, help="Declination in degrees (ICRS)")
    parser.add_argument("--name", type=str, help="Object name resolved with CDS Sesame (e.g. '3C 273')")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_ARCSEC,
                        help=f"Crossmatch radius in arcsec (default {DEFAULT_RADIUS_ARCSEC:g})")
    parser.add_argument("--plot", type=str, help="Write a PNG SED plot to this path")
    parser.add_argument("--json", action="store_true", help="Print the full JSON passport")
    parser.set_defaults(handler=run_cli)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(prog="sed")
    register_cli(_parser.add_subparsers(dest="command"))
    _args = _parser.parse_args()
    raise SystemExit(_args.handler(_args) if hasattr(_args, "handler") else 2)
