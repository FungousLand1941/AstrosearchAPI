"""Sky imaging: multi-survey HiPS cutouts (CDS hips2fits), wavelength stacks, and the web UI mount.

Cutouts are rendered server side by the CDS ``hips2fits`` service
(https://alasky.cds.unistra.fr/hips-image-services/hips2fits), which reprojects
any Hierarchical Progressive Survey (HiPS; Fernique et al. 2015, A&A 578, A114,
2015A&A...578A.114F) onto a requested WCS. The documented request parameters are
``hips``, ``width``, ``height``, ``fov`` (degrees, largest image dimension),
``projection``, ``ra``, ``dec``, ``coordsys``, ``rotation_angle``, ``format``
(``fits`` | ``jpg`` | ``png``) and, for jpg/png only, ``min_cut``, ``max_cut``,
``stretch`` and ``cmap``; one cutout may not exceed 50 million pixels. The service
has two independent endpoints (``alasky`` and ``alaskybis``); a 5xx or transport
error on the first is retried on the second.

Every HiPS identifier in :data:`SURVEYS` was checked against the CDS MocServer
registry (``/MocServer/query?ID=...&get=record``) and rendered through hips2fits
at 3C 273 on 2026-09-28. Wavelength limits (``em_min``/``em_max``, metres) and
bibcodes are the registry's own ``em_min``, ``em_max`` and ``bib_reference``
values, except where noted on the survey. Two requested surveys are *not*
available as HiPS:

* FIRST (Becker, White & Helfand 1995) has no HiPS in the CDS registry
  (MocServer searches ``ID=*FIRST*`` / ``obs_title=*FIRST*`` return nothing); the
  nearest equivalents are NVSS (same frequency, 45" beam) and VLASS (2-4 GHz, 2.5").
* VLASS (``NRAO/P/VLASS-Quicklook-MedianStack``) is registered, but hips2fits
  answered HTTP 500 on the primary endpoint and a fully transparent (no-data)
  image on the mirror when verified (and still at the Crab, where its MOC does
  have data); it is offered but not used in default stacks.

Blank images are never trusted blindly: hips2fits also emits blank images as a
*rendering-failure* mode (VLASS at the Crab: fully transparent PNG, uniformly white
JPEG). A blank image (no data pixels; for JPEG, which has no alpha, a uniform image
settled with the equivalent PNG) is cached as a genuine "no data" answer only when
the survey's MOC confirms it does not reach the field at all. Otherwise the next
endpoint is tried, and if every endpoint is blank the image comes back uncached
with ``X-Cutout-Blank: rendering-failure`` plus ``X-Cutout-Degraded`` (the MOC
overlaps the field) or ``X-Cutout-Blank: unconfirmed`` (the MocServer could not
be asked).

Colour HiPS (``color=True``) are JPEG/PNG composites; FITS cutouts of them are
8-bit RGBA display cubes. Each has a ``science`` companion (plus optional
``science_alternates`` of the same survey) with FITS tiles (tile formats checked in
the MocServer ``hips_tile_format`` field), used for the stack's FITS links. The
companion's own footprint is checked too: Legacy Surveys DR10 r, for instance,
covers far less sky (MOC sky fraction 0.50) than the DR10 colour HiPS (0.67).

FITS pixels are *single-band survey pixel values*, not necessarily flux
calibrated: each survey records ``pixel_units`` and ``calibrated`` from the
registry record (``hips_bunit``, ``hips_pixel_bitpix``, ``hips_data_range``,
``hipsgen_params``) and the survey papers. hips2fits writes no ``BUNIT`` or zero
point into its FITS headers, and it *resamples* (it does not flux-conserve), so
values stay per native survey pixel. DSS2 red is photographic plate density,
2MASS and AllWISE are DN that need each Atlas image's ``MAGZP``, Pan-STARRS1 stacks
are counts with an exposure-dependent zero point.

Survey footprints for the stack are taken from the MocServer spatial query
(``RA``/``DEC``/``SR`` + ``expr=ID=a||ID=b`` + ``get=id``), which intersects each
survey's MOC (Fernique et al. 2014, IVOA MOC 1.0) with the position.

Stacks follow proper motion: given ``pm_ra_masyr``/``pm_dec_masyr`` (or a name
that Sesame resolves with a proper motion), each panel is centred on the target's
linearly propagated position at the survey's mean observing epoch (``epoch_span``
from the registry ``t_min``/``t_max`` or the survey papers), so fast stars such as
Barnard's star (10.4"/yr) stay in the field.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import io
import json
import math
import os
import re
import ssl
import struct
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
import structlog
from astropy import constants as const
from astropy import units as u
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image  # Pillow: required (PNG alpha coverage, truncated-image detection)
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Service endpoints & limits
# ---------------------------------------------------------------------------

HIPS2FITS_URLS: tuple[str, ...] = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
)
MOCSERVER_URLS: tuple[str, ...] = (
    "https://alasky.cds.unistra.fr/MocServer/query",
    "https://alaskybis.cds.unistra.fr/MocServer/query",
)

#: hips2fits refuses cutouts larger than 50 million pixels (service documentation).
HIPS2FITS_MAX_PIXELS = 50_000_000
MIN_SIZE_PX = 8
MAX_SIZE_PX = 4096
MIN_FOV_ARCMIN = 0.05
#: ``fov`` is the full width of the largest image side, so the whole sky is 360 deg
#: (hips2fits' own HI4PI Mollweide example uses 2000 px x 0.18 deg = 360 deg).
MAX_FOV_ARCMIN = 360.0 * 60.0

#: Projections accepted by hips2fits (its documentation lists these WCS codes).
PROJECTIONS: frozenset[str] = frozenset({
    "AZP", "SZP", "TAN", "STG", "SIN", "ARC", "ZEA", "AIR", "CYP", "CEA", "CAR",
    "MER", "SFL", "PAR", "MOL", "AIT", "TSC", "CSC", "QSC", "HPX", "XPH",
})
#: Largest full field of view (arcmin) of projections that cannot show the whole sphere
#: (Calabretta & Greisen 2002, A&A 395, 1077, 2002A&A...395.1077C, sect. 5.1).
#: ``fov`` is the full width, i.e. twice the angular distance from the tangent point:
#: TAN (and AZP/SZP, which reduce to it for mu = 0) diverges 90 deg from the tangent
#: point and SIN folds over there (a hemisphere, fov 180 deg); STG and AIR diverge only
#: at the antipode (fov < 360 deg). ZEA and ARC, like the all-sky projections (MOL, AIT,
#: CAR, CEA, HPX ...), map the whole sphere, so ``MAX_FOV_ARCMIN`` applies to them.
PROJECTION_MAX_FOV_ARCMIN: dict[str, tuple[float, bool]] = {
    # projection: (limit, limit itself allowed)
    "TAN": (180.0 * 60.0, False),
    "AZP": (180.0 * 60.0, False),
    "SZP": (180.0 * 60.0, False),
    "SIN": (180.0 * 60.0, True),
    "STG": (360.0 * 60.0, False),
    "AIR": (360.0 * 60.0, False),
}
#: Largest stack field (TAN panels): just under a hemisphere.
STACK_MAX_FOV_ARCMIN = 179.0 * 60.0
STRETCHES: frozenset[str] = frozenset({"power", "linear", "sqrt", "log", "asinh"})
ImageFormat = Literal["png", "jpg", "fits"]
FORMATS: tuple[str, ...] = ("png", "jpg", "fits")
MEDIA_TYPES: dict[str, str] = {"png": "image/png", "jpg": "image/jpeg", "fits": "application/fits"}

_CUT_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?%?$")
#: hips2fits' documented defaults for an omitted cut (percentiles of the pixel distribution).
DEFAULT_MIN_CUT_PERCENT = 0.5
DEFAULT_MAX_CUT_PERCENT = 99.5


def _parse_cut(name: str, value: str) -> tuple[float, bool]:
    """Validate a hips2fits ``min_cut``/``max_cut``: ``(number, is_percentile)``.

    hips2fits takes either a pixel value or a percentile of the pixel distribution
    (``'99.5%'``); a percentile outside [0, 100] makes it answer HTTP 500, which is
    indistinguishable from an outage, so it is rejected here.
    """
    text = str(value).strip()
    if not _CUT_RE.match(text):
        raise CutoutValidationError(f"{name} must be a number or a percentile such as '99.5%'")
    percent = text.endswith("%")
    number = float(text.rstrip("%"))
    if not math.isfinite(number):
        raise CutoutValidationError(f"{name} must be finite")
    if percent and not 0.0 <= number <= 100.0:
        raise CutoutValidationError(f"{name} percentile must be in [0, 100]%")
    return number, percent


def _check_cut_order(min_cut: tuple[float, bool] | None, max_cut: tuple[float, bool] | None) -> None:
    """Reject cut pairs hips2fits cannot render (it answers them with HTTP 500).

    An omitted cut takes its documented default (``min_cut`` 0.5 %, ``max_cut``
    99.5 %), so a lone percentile is compared with that default. Verified live
    (NVSS at the Crab, 2026-09-28): ``min_cut=99.5%`` and ``max_cut=0.5%`` alone
    render, ``99.6%`` / ``0.4%`` alone give HTTP 500; equal cuts (``5%``/``5%``,
    ``50``/``50``) render. A pixel-value cut against a percentile cannot be checked
    without the data; :meth:`CutoutService.cutout` diagnoses that case afterwards.
    """
    low = min_cut if min_cut is not None else (DEFAULT_MIN_CUT_PERCENT, True)
    high = max_cut if max_cut is not None else (DEFAULT_MAX_CUT_PERCENT, True)
    if low[1] == high[1] and low[0] > high[0]:
        if min_cut is None or max_cut is None:
            which = "max_cut" if min_cut is None else "min_cut"
            default = "min_cut (default 0.5%)" if min_cut is None else "max_cut (default 99.5%)"
            relation = "at least" if min_cut is None else "at most"
            raise CutoutValidationError(f"{which} must be {relation} the hips2fits default {default}")
        raise CutoutValidationError("min_cut must not exceed max_cut")


#: Colormaps hips2fits renders (each also with a ``_r`` reversed variant; names are case-sensitive).
#: hips2fits silently falls back to its default grey scale (``Greys_r``) for any name it does not
#: know (``foobar``, ``Viridis``), so only names verified live are accepted: every Matplotlib 3.11
#: colormap was rendered on a 24x24 NVSS cutout of 3C 273 (2026-09-28) and compared with the
#: default rendering; all but ``okabe_ito``/``okabe_ito_r`` (identical to the default, i.e. unknown
#: to the service) changed the image. ``Greys``/``Grays`` are aliases, so ``Greys_r`` is the default.
HIPS2FITS_CMAPS: frozenset[str] = frozenset(
    name + suffix
    for name in ["Accent", "Blues", "BrBG", "BuGn", "BuPu", "CMRmap", "Dark2", "GnBu", "Grays", "Greens", "Greys", "OrRd", "Oranges", "PRGn", "Paired", "Pastel1", "Pastel2", "PiYG", "PuBu", "PuBuGn", "PuOr", "PuRd", "Purples", "RdBu", "RdGy", "RdPu", "RdYlBu", "RdYlGn", "Reds", "Set1", "Set2", "Set3", "Spectral", "Wistia", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "afmhot", "autumn", "berlin", "binary", "bone", "brg", "bwr", "cividis", "cool", "coolwarm", "copper", "cubehelix", "flag", "gist_earth", "gist_gray", "gist_grey", "gist_heat", "gist_ncar", "gist_rainbow", "gist_stern", "gist_yarg", "gist_yerg", "gnuplot", "gnuplot2", "gray", "grey", "hot", "hsv", "inferno", "jet", "magma", "managua", "nipy_spectral", "ocean", "pink", "plasma", "prism", "rainbow", "seismic", "spring", "summer", "tab10", "tab20", "tab20b", "tab20c", "terrain", "turbo", "twilight", "twilight_shifted", "vanimo", "viridis", "winter"]
    for suffix in ("", "_r")
)
_HIPS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*(/[A-Za-z0-9._+-]+)+$")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ImagingError(Exception):
    """Base class for imaging failures."""


class CutoutValidationError(ImagingError, ValueError):
    """The cutout request is invalid (maps to HTTP 422)."""


class UnknownSurveyError(CutoutValidationError):
    """The survey key / HiPS identifier is unknown or not available."""


class CutoutUpstreamError(ImagingError):
    """hips2fits (or the MocServer) failed or answered something unusable (HTTP 502)."""


# ---------------------------------------------------------------------------
# Survey catalogue
# ---------------------------------------------------------------------------

_HC = (const.h * const.c).to(u.keV * u.m).value  # keV * m, for X-ray energy labels
_C = const.c.to(u.m / u.s).value

REGIME_ORDER: tuple[str, ...] = ("radio", "infrared", "optical", "uv", "xray")


def _mjd_year(mjd: float) -> float:
    """Julian year of an MJD (J2000.0 = MJD 51544.5), rounded to 0.01 yr."""
    return round(2000.0 + (mjd - 51544.5) / 365.25, 2)


def _mjd_span(t_min: float, t_max: float) -> tuple[float, float]:
    """Observing span (Julian years) from a MocServer record's ``t_min``/``t_max`` (MJD)."""
    return _mjd_year(t_min), _mjd_year(t_max)


@dataclass(frozen=True, slots=True)
class HipsSurvey:
    """One HiPS image survey usable through hips2fits.

    ``em_min_m``/``em_max_m`` are the MocServer ``em_min``/``em_max`` (metres);
    ``coverage`` is ``"full"`` (MOC sky fraction 1), ``"moc"`` (check with the
    MocServer) or a ``"dec>-40"``-style declination limit (supported for surveys
    without a usable MOC; none of the catalogued surveys currently needs one).

    ``color`` surveys are RGB composites built from JPEG/PNG tiles: their FITS
    output is a 4-plane 8-bit display cube, not survey data. ``science`` names the
    single-band survey (FITS tiles) to use for pixel values, and
    ``science_alternates`` other bands of the same survey to fall back on where the
    first companion has no data.

    ``pixel_units`` says what a single-band survey's FITS pixels are (``"Jy/beam"``,
    ``"DN"``, ``"photographic density"`` ...) and ``calibrated`` whether they are in
    physical flux or surface-brightness units as delivered (``False`` for DN,
    counts, plate densities or unknown units); ``pixel_note`` explains conversions.
    ``epoch_span`` is the (first, last) Julian year of the observations, or None
    when no single span applies (multi-year coadds without a registry span).
    """

    key: str
    hips_id: str
    label: str
    regime: str
    em_min_m: float
    em_max_m: float
    bib_reference: str | None
    color: bool
    coverage: str = "moc"
    band: str | None = None
    note: str | None = None
    science: str | None = None
    science_alternates: tuple[str, ...] = ()
    pixel_units: str | None = None
    calibrated: bool = False
    pixel_note: str | None = None
    epoch_span: tuple[float, float] | None = None

    @property
    def has_wavelength(self) -> bool:
        return (math.isfinite(self.em_min_m) and math.isfinite(self.em_max_m)
                and self.em_min_m > 0 and self.em_max_m > 0)

    @property
    def wavelength_m(self) -> float:
        """Representative wavelength: geometric mean of the band limits (metres; NaN if unknown)."""
        return math.sqrt(self.em_min_m * self.em_max_m) if self.has_wavelength else math.nan

    @property
    def frequency_hz(self) -> float:
        return _C / self.wavelength_m if self.has_wavelength else math.nan

    @property
    def mean_epoch(self) -> float | None:
        """Middle of ``epoch_span`` (Julian year), or None."""
        return None if self.epoch_span is None else round(0.5 * (self.epoch_span[0] + self.epoch_span[1]), 3)

    @property
    def wavelength_label(self) -> str:
        """Human label: frequency for radio, energy for X-ray, wavelength otherwise."""
        if not self.has_wavelength:
            return "unknown"
        if self.regime == "radio":
            lo, hi = sorted((_C / self.em_max_m, _C / self.em_min_m))
            return _span(lo / 1e9, hi / 1e9, "GHz") if hi >= 1e9 else _span(lo / 1e6, hi / 1e6, "MHz")
        if self.regime == "xray":
            lo, hi = sorted((_HC / self.em_max_m, _HC / self.em_min_m))
            return _span(lo, hi, "keV")
        lo, hi = sorted((self.em_min_m, self.em_max_m))
        if hi >= 1e-6:  # unit from the upper limit: 0.39-1.02 um, not 390-1.02e+03 nm
            return _span(lo * 1e6, hi * 1e6, "µm")
        return _span(lo * 1e9, hi * 1e9, "nm")

    def covered(self, dec: float) -> bool | None:
        """Declination-limit coverage (True/False) or None when a MOC query is needed."""
        if self.coverage == "full":
            return True
        match = re.fullmatch(r"dec([<>])(-?\d+(?:\.\d+)?)", self.coverage)
        if match:
            limit = float(match.group(2))
            return dec > limit if match.group(1) == ">" else dec < limit
        return None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description (unknown wavelengths are ``None``, never NaN)."""
        data = asdict(self)
        known = self.has_wavelength
        data.update(
            em_min_m=self.em_min_m if known else None,
            em_max_m=self.em_max_m if known else None,
            wavelength_m=self.wavelength_m if known else None,
            frequency_hz=self.frequency_hz if known else None,
            wavelength=self.wavelength_label,
            science_alternates=list(self.science_alternates),
            epoch_span=list(self.epoch_span) if self.epoch_span else None,
            mean_epoch=self.mean_epoch,
            bib_url=f"https://ui.adsabs.harvard.edu/abs/{self.bib_reference}/abstract" if self.bib_reference else None,
        )
        return data


def _span(lo: float, hi: float, unit: str) -> str:
    def fmt(x: float) -> str:
        # Three significant figures without exponent notation (1020, not 1.02e+03).
        if x >= 100:
            return f"{x:.0f}"
        return f"{x:.3g}"

    if math.isclose(lo, hi, rel_tol=0.02):
        # A narrow band is one value, to four significant figures (147.5 MHz, 1.4 GHz).
        return f"{float(f'{math.sqrt(lo * hi):.4g}'):g} {unit}"
    return f"{fmt(lo)}–{fmt(hi)} {unit}"


# Registry em limits of the TGSS ADR (CDS MocServer record astron.nl/P/tgssadr, the same
# data as Leiden/P/TGSSADR, whose record has no em_* values): 2.03143-2.03363 m, i.e.
# 147.4-147.6 MHz around the 147.5 MHz centre frequency (Intema et al. 2017). "150 MHz" is
# only the survey's nominal name.
_TGSS_EM = (2.03143, 2.03363)

_RADIO_NOTE = "Pixel values per synthesised beam, as in the survey images; resampled, not flux-conserving."
_NMGY = "22.5 - 2.5 log10(value)"
# Observing spans (Julian years). Registry = MocServer t_min/t_max (MJD) of the HiPS record.
_SDSS_SPAN = (1998.5, 2009.6)  # SDSS imaging 1998-2009 (York et al. 2000; Ahn et al. 2012); registry t_* are later
_ALLWISE_SPAN = (2010.0, 2011.2)  # WISE cryogenic + NEOWISE post-cryo, Jan 2010 - Feb 2011 (Wright et al. 2010)
_2MASS_SPAN = _mjd_span(50600, 51941)  # registry: June 1997 - Feb 2001 (Skrutskie et al. 2006)
_PS1_SPAN = _mjd_span(54999.5103005881, 56896.245445359)  # registry (colour HiPS): 3pi survey 2009-2014
_LEGACY_SPAN = _mjd_span(56878, 58549)  # registry t_min/t_max of the DR10 r and i HiPS
_GALEX_SPAN = _mjd_span(52757.4858796, 56471.4858796)  # registry: GR6/7, 2003-2013
_SPITZER_SPAN = _mjd_span(52876, 55195)  # registry: cryogenic Legacy programmes 2003-2009
_DSS2_RED_SPAN = _mjd_span(45700, 51179)  # registry (DSS2 red): plates 1984-1999, epoch varies plate to plate
# Registry t_min/t_max of the DSS2 colour HiPS (MJD 42413-51179, 1975.0-1999.0): its blue component
# (SERC-J/EJ 1975-1988, POSS-II J 1987-1998, as in the DSS2 blue record) starts nine years before the red.
_DSS2_COLOR_SPAN = _mjd_span(42413, 51179)
# unWISE neo8 (Meisner et al. 2022, 2022RNAAS...6..188M): W1/W2 exposures of epochs 2010.0-2011.1 (WISE)
# and 2014.0-2022.0 (NEOWISE-R), i.e. the "nine years" of the registry description.
_UNWISE_SPAN = (2010.0, 2022.0)
_XMM_SPAN = _mjd_span(51577, 60597)  # registry: every PN pointing of 5XMM-DR15, 2000-2024
#: ``hips_bunit`` of the GALEX HiPS registry records (checked by the live registry test). They do
#: not describe the pixels, so they are deliberately not used or quoted as units.
GALEX_REGISTRY_BUNITS: dict[str, str] = {"galex_nuv": "(1/0.723778)MJy/sr", "galex_fuv": "(1/2.04142)MJy/sr"}
# Read literally, those strings do not match the pixels. Aperture photometry of 1.5"/px hips2fits FITS,
# reading pixels as counts/s per native 1.5" pixel, against GUVcat (VizieR II/335/galex_ais) implies AB zero
# points of 19.86 and 20.06 (NUV) and 18.97 (FUV) in the 3C 273 field (2026-09-28; checks on other fields
# gave 19.99-20.06 and 18.81-18.84), i.e. the Morrissey et al. (2007) count-rate zero points 20.08 and
# 18.82: the pixels are counts/s.
_GALEX_BUNIT_NOTE = ("Not flux-calibrated as delivered; the registry's hips_bunit strings for these HiPS do not "
                     "describe the pixels (which match the count-rate zero points) and are not used.")
_LEGACY_NOTE = f"AB nanomaggies per native 0.262\" pixel (Dey et al. 2019): m_AB = {_NMGY}."

SURVEYS: dict[str, HipsSurvey] = {s.key: s for s in (
    # --- radio ------------------------------------------------------------------
    HipsSurvey("tgss", "Leiden/P/TGSSADR", "TGSS ADR 150 MHz", "radio", *_TGSS_EM,
               "2017A&A...598A..78I", False, "moc", "147.5 MHz (nominal 150 MHz)",
               "em_min/em_max from the MocServer record astron.nl/P/tgssadr (same TGSS ADR data; the "
               "Leiden/P/TGSSADR record has none): 147.4-147.6 MHz, centre 147.5 MHz (Intema et al. 2017).",
               pixel_units="Jy/beam", calibrated=True, pixel_note=_RADIO_NOTE,
               epoch_span=(2010.25, 2012.25)),  # GMRT observations April 2010 - March 2012 (Intema et al. 2017)
    HipsSurvey("racs_low", "CSIRO/P/RACS/low/I", "RACS-low 888 MHz", "radio", 0.291, 0.403,
               "2020PASA...37...48M", False, "moc", "Stokes I",
               pixel_units="Jy/beam", calibrated=True, pixel_note=_RADIO_NOTE, epoch_span=_mjd_span(58594, 59021)),
    HipsSurvey("nvss", "CDS/P/NVSS", "NVSS 1.4 GHz", "radio", 0.21413747, 0.21413747,
               "1998AJ....115.1693C", False, "moc", "Stokes I",
               pixel_units="Jy/beam", calibrated=True, pixel_note=_RADIO_NOTE + " 45\" beam.",
               epoch_span=_mjd_span(49231, 50357)),
    HipsSurvey("vlass", "NRAO/P/VLASS-Quicklook-MedianStack", "VLASS 3 GHz", "radio", 0.0747, 0.1525,
               "2020PASP..132c5001L", False, "moc", "Quick Look median stack",
               "The survey covers dec > -40 deg (Lacy et al. 2020), but this HiPS does not fill it: the "
               "MocServer answers spatial queries for it (no coverage at 3C 273, coverage at the Crab), so "
               "its MOC decides. hips2fits rendering of it was failing when verified (2026-09-28).",
               pixel_units="Jy/beam", calibrated=True,
               pixel_note="Quick Look images are not science-grade: their flux densities carry known systematic "
                          "errors (Lacy et al. 2020). " + _RADIO_NOTE,
               epoch_span=_mjd_span(58004.01597222, 60588.95416667)),
    # --- infrared ----------------------------------------------------------------
    HipsSurvey("allwise", "CDS/P/allWISE/color", "AllWISE W1/W2/W4", "infrared", 2.754e-6, 2.79107e-5,
               "2010AJ....140.1868W", True, "full", "W1 W2 W4 colour", science="allwise_w1",
               epoch_span=_ALLWISE_SPAN),
    HipsSurvey("allwise_w1", "CDS/P/allWISE/W1", "AllWISE W1 3.4 µm", "infrared", 2.754e-6, 3.8723e-6,
               "2010AJ....140.1868W", False, "full", "W1",
               "Registry t_min/t_max (MJD 55378-55414) cover 36 days only; the AllWISE span (2010.0-2011.2) is used.",
               pixel_units="DN", calibrated=False,
               pixel_note="AllWISE Atlas Image DN (registry data range 4.3-2345). Magnitudes need the Atlas "
                          "image's MAGZP (Vega), which the cutout header does not carry.",
               epoch_span=_ALLWISE_SPAN),
    HipsSurvey("unwise", "CDS/P/unWISE/color-W2-W1W2-W1", "unWISE W1/W2", "infrared", 2.754e-6, 5.3413e-6,
               "2022RNAAS...6..188M", True, "full", "W1 W2 colour", science="unwise_w1", epoch_span=_UNWISE_SPAN),
    HipsSurvey("unwise_w1", "CDS/P/unWISE/W1", "unWISE W1 3.4 µm", "infrared", 2.754e-6, 3.8723e-6,
               "2022RNAAS...6..188M", False, "full", "W1",
               "neo8 coadd (registry prov_progenitor .../unwise/neo8/): exposures of 2010.0-2011.1 and 2014.0-2022.0 "
               "(Meisner et al. 2022); the registry has no t_min/t_max. The coadd blends every epoch, and the "
               "eight NEOWISE years outweigh 2010, so a fast mover appears nearer its ~2017 position than the "
               "span's middle (2016.0).",
               pixel_units="Vega nanomaggies", calibrated=True,
               pixel_note=f"unWISE coadds are in Vega nanomaggies (Lang 2014): m_Vega = {_NMGY}, per native "
                          "2.75\" pixel.",
               epoch_span=_UNWISE_SPAN),
    HipsSurvey("spitzer", "CDS/P/SPITZER/color", "Spitzer IRAC", "infrared", 3.1296e-6, 9.5875e-6,
               "2003PASP..115..953B", True, "moc", "IRAC colour", science="spitzer_irac1", epoch_span=_SPITZER_SPAN),
    HipsSurvey("spitzer_irac1", "CDS/P/SPITZER/IRAC1", "Spitzer IRAC1 3.6 µm", "infrared", 3.1296e-6, 3.9614e-6,
               "2003PASP..115..953B", False, "moc", "IRAC1",
               "Registry lists seven bib_references (GLIMPSE and other Spitzer programmes); the first is cited.",
               pixel_units="MJy/sr", calibrated=True,
               pixel_note="Spitzer Legacy-programme mosaics in surface brightness (MJy/sr).",
               epoch_span=_SPITZER_SPAN),
    HipsSurvey("2mass", "CDS/P/2MASS/color", "2MASS J/H/Ks", "infrared", 1.147e-6, 2.303e-6,
               "2006AJ....131.1163S", True, "full", "J H Ks colour", science="2mass_k", epoch_span=_2MASS_SPAN),
    HipsSurvey("2mass_k", "CDS/P/2MASS/K", "2MASS Ks", "infrared", 2.015e-6, 2.303e-6,
               "2006AJ....131.1163S", False, "full", "Ks",
               pixel_units="DN", calibrated=False,
               pixel_note="Background-subtracted Atlas Image DN (HiPS built with hipsgen skyval=SKYVAL). "
                          "Magnitudes need each Atlas image's MAGZP, which the cutout header does not carry.",
               epoch_span=_2MASS_SPAN),
    # --- optical -----------------------------------------------------------------
    HipsSurvey("panstarrs", "CDS/P/PanSTARRS/DR1/color-z-zg-g", "Pan-STARRS1 g/z", "optical", 3.9434e-7, 9.51e-7,
               "2016arXiv161205560C", True, "moc", "g z colour", science="panstarrs_g", epoch_span=_PS1_SPAN),
    HipsSurvey("panstarrs_g", "CDS/P/PanSTARRS/DR1/g", "Pan-STARRS1 g", "optical", 3.9434e-7, 5.59327e-7,
               "2016arXiv161205560C", False, "moc", "g",
               "The DR1 r/i/z/y HiPS are 16-bit with registry data ranges of about +-10 (a different pixel "
               "scale from g), so they are not offered as FITS companions.",
               pixel_units="counts", calibrated=False,
               pixel_note="PS1 stack counts (registry data range -3652 to 9688). The AB zero point depends on "
                          "each stack's exposure time (Waters et al. 2020) and is not in the cutout header.",
               epoch_span=_mjd_span(54999.5103005881, 56837.9755531207)),
    HipsSurvey("legacy", "CDS/P/DESI-Legacy-Surveys/DR10/color", "Legacy Surveys DR10", "optical", 3.9e-7, 1.02e-6,
               "2019AJ....157..168D", True, "moc", "g r i z colour", science="legacy_r",
               science_alternates=("legacy_g", "legacy_z", "legacy_i"), epoch_span=_LEGACY_SPAN),
    HipsSurvey("legacy_g", "CDS/P/DESI-Legacy-Surveys/DR10/g", "Legacy Surveys DR10 g", "optical", 3.876e-7, 5.808e-7,
               "2019AJ....157..168D", False, "moc", "g",
               pixel_units="nanomaggies", calibrated=True, pixel_note=_LEGACY_NOTE, epoch_span=_LEGACY_SPAN),
    HipsSurvey("legacy_r", "CDS/P/DESI-Legacy-Surveys/DR10/r", "Legacy Surveys DR10 r", "optical", 5.618e-7, 7.26e-7,
               "2019AJ....157..168D", False, "moc", "r",
               "MOC sky fraction 0.50, against 0.67 for the DR10 colour HiPS; g, z and i fill most of the gap.",
               pixel_units="nanomaggies", calibrated=True, pixel_note=_LEGACY_NOTE, epoch_span=_LEGACY_SPAN),
    HipsSurvey("legacy_i", "CDS/P/DESI-Legacy-Surveys/DR10/i", "Legacy Surveys DR10 i", "optical", 7.1e-7, 8.57e-7,
               "2019AJ....157..168D", False, "moc", "i",
               pixel_units="nanomaggies", calibrated=True, pixel_note=_LEGACY_NOTE, epoch_span=_LEGACY_SPAN),
    HipsSurvey("legacy_z", "CDS/P/DESI-Legacy-Surveys/DR10/z", "Legacy Surveys DR10 z", "optical", 8.252e-7,
               1.0145e-6, "2019AJ....157..168D", False, "moc", "z",
               pixel_units="nanomaggies", calibrated=True, pixel_note=_LEGACY_NOTE, epoch_span=_LEGACY_SPAN),
    HipsSurvey("sdss", "CDS/P/SDSS9/color", "SDSS DR9", "optical", 3.782e-7, 8.389e-7,
               "2012ApJS..203...21A", True, "moc", "g r i colour",
               "Registry record has no bib_reference; cited SDSS DR9 paper (Ahn et al. 2012).", science="sdss_r",
               epoch_span=_SDSS_SPAN),
    HipsSurvey("sdss_r", "CDS/P/SDSS9/r", "SDSS DR9 r", "optical", 5.415e-7, 6.989e-7,
               "2012ApJS..203...21A", False, "moc", "r",
               "Registry record has no bib_reference; cited SDSS DR9 paper (Ahn et al. 2012). Registry "
               "t_min/t_max (2008-2014) are not observing dates; SDSS imaging ran 1998-2009.",
               pixel_units="nanomaggies", calibrated=True,
               pixel_note=f"SDSS corrected-frame nanomaggies per native 0.396\" pixel: m = {_NMGY} "
                          "(SDSS magnitudes are close to, not exactly, AB).",
               epoch_span=_SDSS_SPAN),
    HipsSurvey("dss2", "CDS/P/DSS2/color", "DSS2 colour", "optical", 4e-7, 6e-7,
               "1996ASPC..101...88L", True, "full", "B R colour",
               "Composite of the DSS2 red and blue plates, which were taken at different epochs (blue from 1975, "
               "red from 1984; registry t_min/t_max 1975.0-1999.0): a fast mover can appear twice.",
               science="dss2_red", epoch_span=_DSS2_COLOR_SPAN),
    HipsSurvey("dss2_red", "CDS/P/DSS2/red", "DSS2 red", "optical", 6.4e-7, 6.58e-7,
               "1996ASPC..101...88L", False, "full", "F+R",
               pixel_units="photographic density", calibrated=False,
               pixel_note="Digitised photographic plate densities (BITPIX 16), not flux-calibrated; the plate "
                          "epoch varies across the sky (1984-1999).",
               epoch_span=_DSS2_RED_SPAN),
    # --- ultraviolet -------------------------------------------------------------
    HipsSurvey("galex", "CDS/P/GALEXGR6_7/color", "GALEX FUV/NUV", "uv", 1.344e-7, 2.831e-7,
               "2017ApJS..230...24B", True, "moc", "FUV NUV colour", science="galex_nuv",
               science_alternates=("galex_fuv",), epoch_span=_GALEX_SPAN),
    HipsSurvey("galex_nuv", "CDS/P/GALEXGR6_7/NUV", "GALEX NUV", "uv", 1.771e-7, 2.831e-7,
               "2017ApJS..230...24B", False, "moc", "NUV",
               pixel_units="counts/s", calibrated=False,
               pixel_note="Intensity-map count rate (counts/s) per native 1.5\" pixel: m_AB = 20.08 - 2.5 log10(cps) "
                          "(NUV zero point, Morrissey et al. 2007, 2007ApJS..173..682M). " + _GALEX_BUNIT_NOTE,
               epoch_span=_GALEX_SPAN),
    HipsSurvey("galex_fuv", "CDS/P/GALEXGR6_7/FUV", "GALEX FUV", "uv", 1.344e-7, 1.786e-7,
               "2017ApJS..230...24B", False, "moc", "FUV",
               pixel_units="counts/s", calibrated=False,
               pixel_note="Intensity-map count rate (counts/s) per native 1.5\" pixel: m_AB = 18.82 - 2.5 log10(cps) "
                          "(FUV zero point, Morrissey et al. 2007, 2007ApJS..173..682M). " + _GALEX_BUNIT_NOTE,
               epoch_span=_GALEX_SPAN),
    # --- X-ray --------------------------------------------------------------------
    HipsSurvey("erosita", "erosita/dr1/count/024", "eROSITA-DE DR1", "xray", 5.39066e-10, 6.19927e-9,
               "2024A&A...682A..34M", False, "moc", "0.2-2.3 keV counts",
               "German eROSITA DR1 covers the western Galactic hemisphere (l > 180 deg).",
               pixel_units="counts", calibrated=False,
               pixel_note="Photon counts per pixel (registry title 'Count Image'); divide by the exposure map "
                          "for rates.",
               epoch_span=_mjd_span(58828.8958, 59011.4583)),
    HipsSurvey("rass", "CDS/P/RASS", "ROSAT All-Sky Survey", "xray", 5.166e-10, 1.2398e-8,
               "1999A&A...349..389V", False, "full", "0.1-2.4 keV",
               pixel_units="counts", calibrated=False,
               pixel_note="16-bit integer photon counts per pixel; the registry gives no unit.",
               epoch_span=(1990.5, 1991.1)),  # survey phase Aug 1990 - Jan 1991 (Voges et al. 1999)
    HipsSurvey("xmm", "xcatdb/P/XMM/PN/color", "XMM-Newton PN", "xray", 2.76e-10, 2.48e-9,
               None, True, "moc", "PN colour",
               "Registry lists em_min/em_max swapped (0.5-4.5 keV); bib_reference is 'Submitted'.", science="xmm_eb2",
               epoch_span=_XMM_SPAN),
    HipsSurvey("xmm_eb2", "xcatdb/P/XMM/PN/eb2", "XMM-Newton PN 0.5-1 keV", "xray", 1.24e-9, 2.48e-9,
               None, False, "moc", "PN energy band 2",
               "Registry lists em_min/em_max swapped (0.5-1 keV); bib_reference is 'Submitted'.",
               pixel_units="unknown", calibrated=False,
               pixel_note="The registry gives no unit for this 5XMM-DR15 PN stack (BITPIX -64).",
               epoch_span=_XMM_SPAN),
    HipsSurvey("chandra", "cxc.harvard.edu/P/cda/hips/allsky/rgb", "Chandra (CXC)", "xray", 2.0e-10, 6.0e-9,
               None, True, "moc", "RGB composite",
               "PNG tiles only (no FITS HiPS), so no single-band FITS cutout is available."),
)}

#: Surveys that were requested but have no HiPS; asking for them gives a clear 422.
UNAVAILABLE_SURVEYS: dict[str, str] = {
    "first": ("FIRST is not published as a HiPS in the CDS registry (MocServer search, 2026-09-28); "
              "use 'nvss' (1.4 GHz) or 'vlass' (3 GHz) instead."),
}

#: Default multi-wavelength stack: per regime slot, the first survey covering the position wins.
STACK_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("radio-low", ("tgss", "racs_low")),
    ("radio", ("nvss", "racs_low")),
    ("mid-infrared", ("allwise", "unwise")),
    ("near-infrared", ("2mass",)),
    ("optical", ("panstarrs", "legacy", "sdss", "dss2")),
    ("ultraviolet", ("galex",)),
    ("soft-xray", ("erosita", "rass")),
    ("xray", ("xmm", "chandra")),
)


def get_survey(key_or_id: str) -> HipsSurvey:
    """Look up a survey by key (``dss2``) or HiPS ID (``CDS/P/DSS2/color``), case-insensitively.

    An unlisted but well-formed HiPS ID is accepted as an ad-hoc survey (hips2fits
    rejects unknown IDs itself, which surfaces as :class:`UnknownSurveyError`). Its
    colour/single-band nature is unknown here: :meth:`CutoutService.describe_hips`
    reads it from the registry, and FITS cutouts are labelled from their content.
    """
    text = (key_or_id or "").strip()
    if not text:
        raise UnknownSurveyError("survey must not be empty")
    lowered = text.lower()
    if lowered in UNAVAILABLE_SURVEYS:
        raise UnknownSurveyError(UNAVAILABLE_SURVEYS[lowered])
    if lowered in SURVEYS:
        return SURVEYS[lowered]
    for survey in SURVEYS.values():
        if survey.hips_id.lower() == lowered:
            return survey
    if _HIPS_ID_RE.match(text):
        return HipsSurvey(key=text, hips_id=text, label=text, regime="unknown", em_min_m=math.nan,
                          em_max_m=math.nan, bib_reference=None, color=False, coverage="moc")
    raise UnknownSurveyError(f"Unknown survey '{text}'. Known keys: {', '.join(SURVEYS)}")


def list_surveys() -> list[dict[str, Any]]:
    """All catalogued surveys ordered radio -> X-ray (decreasing wavelength)."""
    return [s.as_dict() for s in sorted(SURVEYS.values(), key=lambda s: -s.wavelength_m)]


def science_survey(survey: HipsSurvey) -> HipsSurvey | None:
    """The preferred survey whose FITS cutouts carry single-band pixel values for ``survey``.

    A single-band survey is its own science survey; a colour composite maps to its
    ``science`` companion, or None when no FITS HiPS exists (e.g. the Chandra RGB).
    Ad-hoc HiPS IDs are assumed single-band. :func:`fits_companion` also takes the
    companion's own footprint into account.
    """
    if not survey.color:
        return survey
    return SURVEYS.get(survey.science) if survey.science else None


def companion_surveys(survey: HipsSurvey) -> list[HipsSurvey]:
    """Single-band FITS candidates for ``survey`` in preference order (itself when single-band)."""
    if not survey.color:
        return [survey]
    keys = ((survey.science,) if survey.science else ()) + survey.science_alternates
    return [SURVEYS[k] for k in keys if k in SURVEYS]


def fits_companion(survey: HipsSurvey, covered: dict[str, bool | None]) -> tuple[HipsSurvey | None, str | None]:
    """The FITS survey to link for ``survey`` at a position, plus a note when it is not the first choice.

    ``covered`` maps survey keys to footprint answers (True / False / None = unknown;
    full-sky surveys count as covered). The first candidate known to cover the
    position wins, else the first whose footprint is unknown (MocServer down). When
    every candidate is known to miss the position there is no FITS link (``None``)
    and the note says why.
    """
    candidates = companion_surveys(survey)
    if not candidates:
        return None, f"{survey.label} has no single-band FITS HiPS."
    answers = {c.key: True if c.coverage == "full" else covered.get(c.key) for c in candidates}
    first = candidates[0]
    for candidate in candidates:
        if answers[candidate.key] is True:
            note = None if candidate is first else f"{first.label} has no data here; {candidate.label} is used instead."
            return candidate, note
    for candidate in candidates:
        if answers[candidate.key] is None:
            return candidate, None
    names = ", ".join(c.label for c in candidates)
    return None, f"No single-band FITS survey of {survey.label} covers this position ({names})."


def regime_for_wavelength(wavelength_m: float) -> str:
    """Coarse spectral regime of a wavelength (metres): used for ad-hoc HiPS only.

    Boundaries follow common usage: radio > 1 mm, infrared 0.7 um - 1 mm, optical
    0.32 - 0.7 um, ultraviolet 10 nm - 0.32 um, X-ray 10 pm - 10 nm (0.12 - 124 keV),
    gamma-ray below 10 pm.
    """
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        return "unknown"
    for limit, regime in ((1e-3, "radio"), (7e-7, "infrared"), (3.2e-7, "optical"), (1e-8, "uv"), (1e-11, "xray")):
        if wavelength_m > limit:
            return regime
    return "gamma"


# ---------------------------------------------------------------------------
# Requests & results
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """Deterministic, lossless text for a float query parameter (cache keys, fixtures)."""
    return repr(float(value))


def _projection_wcs(projection: str) -> Any:
    """Unit-scale celestial WCS of ``projection`` with its reference point at (0, 0) and pixel 0 there."""
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.crpix = [1.0, 1.0]  # 1-based FITS pixel 1 = 0-based pixel 0: pixel coordinates are plane degrees
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.set()
    return wcs


def _plane_xy(projection: str, lon: float, lat: float) -> tuple[float, float]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = _projection_wcs(projection).wcs_world2pix([[lon, lat]], 0)[0]
    return float(x), float(y)


@functools.lru_cache(maxsize=256)
def hips2fits_cdelt_deg(projection: str, fov_deg: float, size_px: int) -> float | None:
    """``|CDELT|`` of a hips2fits cutout: ``2 X(fov/2) / size_px`` (degrees per pixel).

    ``X(a)`` is the projection-plane distance (Calabretta & Greisen 2002, 2002A&A...395.1077C)
    of a point ``a`` degrees from the reference point along the native equator / first axis,
    computed with astropy's WCSLIB. This reproduces every ``CDELT1`` hips2fits returned for
    all 21 accepted projections at fov 1, 100 and 360 deg (40x20 px FITS, 2026-09-28), e.g.
    TAN 100 deg: 3.41412 (= 2 tan 50 deg in plane degrees / 40), MOL 360 deg: 8.10285
    (= 2 x 2 sqrt 2 / pi x 90 deg / 40), CAR 360 deg: 9. None when WCSLIB cannot project the edge.
    """
    x, _ = _plane_xy(projection, fov_deg / 2.0, 0.0)
    if not math.isfinite(x) or x == 0.0:
        return None
    return 2.0 * abs(x) / size_px


@functools.lru_cache(maxsize=32)
def _reference_plane_scale(projection: str) -> float:
    """Square root of the projection's areal scale at the reference point (plane deg^2 per sq deg).

    1 for zenithal, CAR and equal-area projections; pi/4 for TSC, 1.0233 for PAR, 1.0854
    for HPX, 1.3604 for XPH (WCSLIB normalisations). A pixel of ``CDELT`` plane degrees
    therefore spans ``CDELT / scale`` degrees on the sky at the image centre.
    """
    step = 1e-4
    x1, y1 = _plane_xy(projection, step, 0.0)
    x2, y2 = _plane_xy(projection, 0.0, step)
    return math.sqrt(abs(x1 * y2 - x2 * y1)) / step


@dataclass(frozen=True, slots=True)
class CutoutRequest:
    """A validated hips2fits cutout request.

    ``fov_arcmin`` is the size of the *largest* image dimension, as hips2fits
    defines ``fov``; the pixel scale is therefore ``fov / max(width, height)``.
    """

    ra: float
    dec: float
    fov_arcmin: float = 5.0
    survey: str = "dss2"
    width: int = 512
    height: int = 512
    format: str = "png"
    projection: str = "TAN"
    stretch: str | None = None
    cmap: str | None = None
    min_cut: str | None = None
    max_cut: str | None = None
    rotation_angle: float = 0.0
    coordsys: str = "icrs"

    def __post_init__(self) -> None:
        for name in ("ra", "dec", "fov_arcmin", "rotation_angle"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise CutoutValidationError(f"{name} must be a finite number")
        if not 0.0 <= self.ra < 360.0:
            raise CutoutValidationError("ra must be in [0, 360) degrees")
        if not -90.0 <= self.dec <= 90.0:
            raise CutoutValidationError("dec must be in [-90, 90] degrees")
        if not MIN_FOV_ARCMIN <= self.fov_arcmin <= MAX_FOV_ARCMIN:
            raise CutoutValidationError(f"fov_arcmin must be in [{MIN_FOV_ARCMIN}, {MAX_FOV_ARCMIN:g}]")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not MIN_SIZE_PX <= value <= MAX_SIZE_PX:
                raise CutoutValidationError(f"{name} must be an integer in [{MIN_SIZE_PX}, {MAX_SIZE_PX}] pixels")
        if self.width * self.height > HIPS2FITS_MAX_PIXELS:
            raise CutoutValidationError("cutout exceeds the hips2fits limit of 50 million pixels")
        fmt = self.format.lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in FORMATS:
            raise CutoutValidationError(f"format must be one of {', '.join(FORMATS)}")
        object.__setattr__(self, "format", fmt)
        projection = self.projection.upper()
        if projection not in PROJECTIONS:
            raise CutoutValidationError(f"projection must be one of {', '.join(sorted(PROJECTIONS))}")
        object.__setattr__(self, "projection", projection)
        if projection in PROJECTION_MAX_FOV_ARCMIN:
            limit, inclusive = PROJECTION_MAX_FOV_ARCMIN[projection]
            if self.fov_arcmin > limit or (self.fov_arcmin == limit and not inclusive):
                relation = "<=" if inclusive else "<"
                raise CutoutValidationError(
                    f"fov_arcmin must be {relation} {limit:g} for {projection} (use ZEA, ARC, MOL, AIT or CAR "
                    "for wider fields)")
        if self.coordsys not in ("icrs", "galactic"):
            raise CutoutValidationError("coordsys must be 'icrs' or 'galactic'")
        if self.stretch is not None and self.stretch not in STRETCHES:
            raise CutoutValidationError(f"stretch must be one of {', '.join(sorted(STRETCHES))}")
        if self.cmap is not None and self.cmap not in HIPS2FITS_CMAPS:
            raise CutoutValidationError(
                f"cmap '{self.cmap[:41]}' is not a colormap hips2fits renders (case-sensitive Matplotlib names such "
                "as viridis, inferno, Greys_r; hips2fits would silently ignore it)")
        cuts = {name: _parse_cut(name, getattr(self, name)) for name in ("min_cut", "max_cut")
                if getattr(self, name) is not None}
        if cuts:
            _check_cut_order(cuts.get("min_cut"), cuts.get("max_cut"))
        if fmt == "fits" and any(getattr(self, n) is not None for n in ("stretch", "cmap", "min_cut", "max_cut")):
            raise CutoutValidationError("stretch/cmap/min_cut/max_cut apply only to png/jpg output")
        get_survey(self.survey)  # raises UnknownSurveyError early

    @property
    def hips(self) -> HipsSurvey:
        return get_survey(self.survey)

    @property
    def pixel_kind(self) -> str:
        """Expected pixels: ``display`` (png/jpg), ``rgb-preview`` (FITS of a catalogued colour HiPS)
        or ``survey`` (FITS of single-band survey pixel values).

        This is what the catalogue predicts; :attr:`Cutout.pixel_kind` is what the
        returned FITS actually holds (an ad-hoc colour HiPS is only known from its content).
        """
        if self.format != "fits":
            return "display"
        return "rgb-preview" if self.hips.color else "survey"

    @property
    def nominal_pixel_scale_arcsec(self) -> float:
        """``fov / max(width, height)``: the mean angular size of a pixel along the largest side."""
        return self.fov_arcmin * 60.0 / max(self.width, self.height)

    @property
    def cdelt_deg(self) -> float | None:
        """The ``CDELT1``/``CDELT2`` magnitude hips2fits writes (projection-plane degrees per pixel).

        See :func:`hips2fits_cdelt_deg`; None only if the projection cannot place the field edge.
        """
        return hips2fits_cdelt_deg(self.projection, self.fov_arcmin / 60.0, max(self.width, self.height))

    @property
    def pixel_scale_arcsec(self) -> float:
        """Angular pixel size at the image centre: the square root of a pixel's solid angle there.

        Equal to ``CDELT`` for the zenithal (TAN, SIN, ARC, ZEA, ...), CAR and equal-area
        (MOL, AIT, CEA, SFL) projections, and to the nominal ``fov / max(width, height)``
        only for small fields: at fov 360 deg an AIT or MOL pixel is 0.9003 (= 2 sqrt 2 / pi)
        times the nominal value, a 100 deg TAN pixel 1.37 times. Falls back to the nominal
        scale if the projection cannot place the field edge.
        """
        cdelt = self.cdelt_deg
        if cdelt is None:
            return self.nominal_pixel_scale_arcsec
        return cdelt * 3600.0 / _reference_plane_scale(self.projection)

    @property
    def field_radius_deg(self) -> float:
        """Radius of the circle circumscribing the image (``fov`` spans the largest side), max 180 deg."""
        half_diagonal = self.fov_arcmin / 120.0 * math.hypot(self.width, self.height) / max(self.width, self.height)
        return min(half_diagonal, 180.0)

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[self.format]

    def params(self) -> dict[str, str]:
        """hips2fits query parameters (documented names), with deterministic number text."""
        params = {
            "hips": self.hips.hips_id,
            "width": str(self.width),
            "height": str(self.height),
            "fov": _num(self.fov_arcmin / 60.0),
            "projection": self.projection,
            "ra": _num(self.ra),
            "dec": _num(self.dec),
            "format": self.format,
        }
        if self.coordsys != "icrs":
            params["coordsys"] = self.coordsys
        if self.rotation_angle:
            params["rotation_angle"] = _num(self.rotation_angle)
        for name in ("stretch", "cmap", "min_cut", "max_cut"):
            value = getattr(self, name)
            if value is not None:
                params[name] = str(value)
        return params

    def icrs_position(self) -> tuple[float, float]:
        """Field centre in ICRS degrees (``coordsys='galactic'`` requests give l, b)."""
        if self.coordsys == "icrs":
            return self.ra, self.dec
        from astropy.coordinates import SkyCoord

        icrs = SkyCoord(l=self.ra * u.deg, b=self.dec * u.deg, frame="galactic").icrs
        return float(icrs.ra.deg), float(icrs.dec.deg)

    def cache_key(self) -> str:
        blob = json.dumps(self.params(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def filename(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.hips.key).strip("_").lower()
        return f"cutout_{slug}_{self.ra:.5f}_{self.dec:+.5f}_{self.fov_arcmin:g}arcmin.{self.format}"


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Format and pixel dimensions read from an image header (no full decode).

    ``bitpix`` is the FITS BITPIX (None for PNG/JPEG).
    """

    format: str
    width: int
    height: int
    planes: int = 1
    bitpix: int | None = None

    @property
    def is_rgb_cube(self) -> bool:
        """A hips2fits FITS of a colour HiPS: 4 planes (RGBA) of 8-bit display values."""
        return self.format == "fits" and self.planes == 4 and self.bitpix == 8


@dataclass(slots=True)
class Cutout:
    """A cutout image plus provenance."""

    request: CutoutRequest
    content: bytes
    media_type: str
    width: int
    height: int
    source_url: str
    endpoint: str
    cached: bool
    fetched_at: str
    coverage_fraction: float | None = None
    #: Why the image is suspect (e.g. blank answer from a mirror after the primary failed), else None.
    degraded: str | None = None
    #: Header of the returned image (planes, BITPIX); None only for hand-built instances.
    info: ImageInfo | None = None
    #: False when the image must not be stored by caches (degraded, or a blank image the MOC did not confirm).
    cacheable: bool = True
    #: For a blank image (no survey pixels): ``"no-data"`` when the survey's MOC confirms it does not
    #: reach the field, ``"unconfirmed"`` when the footprint could not be checked, ``"rendering-failure"``
    #: when the MOC overlaps the field (also ``degraded``). None for an image with data.
    blank: str | None = None
    #: Colour-composite FITS only: the single-band companion that covers the field (MOC-checked by
    #: :class:`CutoutService`), a note when it is not the first choice or none covers it, and
    #: whether that check ran (False: headers fall back to the catalogue's first choice).
    science: HipsSurvey | None = None
    science_note: str | None = None
    science_checked: bool = False

    @property
    def survey(self) -> HipsSurvey:
        return self.request.hips

    def science_companion(self) -> tuple[HipsSurvey | None, str | None]:
        """Single-band FITS survey to point users to for this colour cutout, plus a note."""
        if self.science_checked:
            return self.science, self.science_note
        return fits_companion(self.survey, {})

    @property
    def pixel_kind(self) -> str:
        """What the image holds, read from its content: ``display``, ``rgb-preview`` or ``survey``.

        A FITS with 4 planes of BITPIX 8 is an RGBA display cube whatever the survey
        key said (e.g. an ad-hoc colour HiPS such as ``CDS/P/Mellinger/color``).
        """
        if self.request.format != "fits":
            return "display"
        if self.info is None:
            return self.request.pixel_kind
        return "rgb-preview" if self.info.is_rgb_cube else "survey"

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, self.content)
        return target

    def headers(self) -> dict[str, str]:
        kind = self.pixel_kind
        headers = {
            "X-Cutout-Survey": self.survey.key,
            "X-Cutout-Hips": self.survey.hips_id,
            "X-Cutout-Cache": "hit" if self.cached else "miss",
            "X-Cutout-Pixel-Scale-Arcsec": f"{self.request.pixel_scale_arcsec:.6g}",
            "X-Cutout-Projection": self.request.projection,
            "X-Cutout-Source": self.source_url,
            "X-Cutout-Pixels": kind,
            "Content-Disposition": f'inline; filename="{self.request.filename()}"',
            "Cache-Control": "public, max-age=86400",
        }
        cdelt = self.request.cdelt_deg
        if cdelt is not None:
            headers["X-Cutout-Cdelt-Deg"] = f"{cdelt:.6g}"
        if self.coverage_fraction is not None:
            headers["X-Cutout-Coverage"] = f"{self.coverage_fraction:.4f}"
        if self.blank:
            headers["X-Cutout-Blank"] = self.blank
        if kind == "survey":
            headers["X-Cutout-Pixel-Units"] = self.survey.pixel_units or "unknown"
            headers["X-Cutout-Calibrated"] = "true" if self.survey.calibrated else "false"
        if kind == "rgb-preview" and self.survey.color:
            science, note = self.science_companion()
            if science is not None:
                headers["X-Cutout-Science-Survey"] = science.key
            if note:
                headers["X-Cutout-Science-Note"] = _header_text(note)
        if self.degraded:
            headers["X-Cutout-Degraded"] = _header_text(self.degraded)
        if self.degraded or not self.cacheable:
            headers["Cache-Control"] = "no-store"
        return headers

    def summary(self) -> dict[str, Any]:
        return {
            "survey": self.survey.key,
            "hips_id": self.survey.hips_id,
            "format": self.request.format,
            "width": self.width,
            "height": self.height,
            "bytes": len(self.content),
            "fov_arcmin": self.request.fov_arcmin,
            "projection": self.request.projection,
            "pixel_scale_arcsec": self.request.pixel_scale_arcsec,
            "nominal_pixel_scale_arcsec": self.request.nominal_pixel_scale_arcsec,
            "cdelt_deg": self.request.cdelt_deg,
            "coverage_fraction": self.coverage_fraction,
            "blank": self.blank,
            "pixels": self.pixel_kind,
            "pixel_units": self.survey.pixel_units if self.pixel_kind == "survey" else None,
            "degraded": self.degraded,
            "cached": self.cached,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
        }


# ---------------------------------------------------------------------------
# Image header probing, completeness & coverage
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: The PNG IEND chunk (length 0, type, CRC): every complete PNG ends with these 12 bytes.
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_FITS_BLOCK = 2880


def _fits_header(content: bytes) -> tuple[dict[str, str], int]:
    """Primary-header cards and the header length in bytes (a multiple of 2880)."""
    cards: dict[str, str] = {}
    for offset in range(0, min(len(content), _FITS_BLOCK * 20), 80):
        card = content[offset:offset + 80].decode("ascii", "replace")
        key = card[:8].strip()
        if key == "END":
            return cards, (offset // _FITS_BLOCK + 1) * _FITS_BLOCK
        if card[8:10] == "= ":
            cards[key] = card[10:].split("/")[0].strip()
    raise CutoutUpstreamError("FITS header has no END card")


def probe_image(content: bytes) -> ImageInfo:
    """Read format and dimensions from PNG (IHDR), JPEG (SOFn) or FITS (primary header)."""
    if content.startswith(_PNG_MAGIC) and len(content) >= 26 and content[12:16] == b"IHDR":
        width, height = struct.unpack(">II", content[16:24])
        color_type = content[25]
        planes = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 1)
        return ImageInfo("png", width, height, planes)
    if content.startswith(b"\xff\xd8"):
        pos = 2
        while pos + 4 <= len(content):
            if content[pos] != 0xFF:
                pos += 1
                continue
            marker = content[pos + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                pos += 1 if marker == 0xFF else 2
                continue
            length = struct.unpack(">H", content[pos + 2:pos + 4])[0]
            if marker in _JPEG_SOF and pos + 9 < len(content):
                height, width = struct.unpack(">HH", content[pos + 5:pos + 9])
                return ImageInfo("jpg", width, height, content[pos + 9])
            pos += 2 + length
        raise CutoutUpstreamError("JPEG without a frame header")
    if content.startswith(b"SIMPLE  ="):
        cards, _ = _fits_header(content)
        try:
            naxis = int(cards.get("NAXIS", "0"))
            width, height = int(cards["NAXIS1"]), int(cards["NAXIS2"])
            planes = int(cards.get("NAXIS3", "1")) if naxis >= 3 else 1
            bitpix = int(cards["BITPIX"])
        except (KeyError, ValueError) as exc:
            raise CutoutUpstreamError("FITS header lacks BITPIX/NAXIS1/NAXIS2") from exc
        return ImageInfo("fits", width, height, planes, bitpix)
    raise CutoutUpstreamError("response is not a PNG, JPEG or FITS image")


def check_complete(content: bytes, info: ImageInfo) -> None:
    """Raise :class:`CutoutUpstreamError` if the body is visibly truncated (no decoder needed).

    A PNG must end with its IEND chunk, a JPEG with the EOI marker (``FF D9``), and
    a FITS must hold every data byte its header announces (NAXISn x |BITPIX| / 8;
    trailing block padding is not required). A connection that drops mid-body can
    otherwise deliver a valid header in front of missing pixels.
    """
    if info.format == "png" and not content.endswith(_PNG_IEND):
        raise CutoutUpstreamError("truncated PNG (no IEND chunk)")
    if info.format == "jpg" and not content.rstrip(b"\x00").endswith(b"\xff\xd9"):
        raise CutoutUpstreamError("truncated JPEG (no end-of-image marker)")
    if info.format == "fits":
        cards, header_bytes = _fits_header(content)
        naxis = int(cards.get("NAXIS", "0"))
        count = math.prod(int(cards.get(f"NAXIS{i}", "0")) for i in range(1, naxis + 1)) if naxis else 0
        needed = header_bytes + count * abs(info.bitpix or 8) // 8
        if len(content) < needed:
            raise CutoutUpstreamError(f"truncated FITS ({len(content)} of {needed} bytes)")


def _decode_raster(content: bytes) -> Image.Image:
    """Fully decode a PNG/JPEG with Pillow (raises :class:`CutoutUpstreamError` on corrupt data)."""
    try:
        image = Image.open(io.BytesIO(content))
        image.load()  # Pillow refuses truncated data unless ImageFile.LOAD_TRUNCATED_IMAGES is set
    except (OSError, SyntaxError, ValueError) as exc:  # PIL raises SyntaxError for some broken files
        raise CutoutUpstreamError(f"image pixels could not be decoded: {exc}") from exc
    return image


def _raster_coverage(image: Image.Image) -> float | None:
    import numpy as np

    if image.mode in ("RGBA", "LA", "PA") or (image.mode in ("P", "L", "RGB") and "transparency" in image.info):
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
        return float((alpha > 0).mean())
    return None


def is_uniform_raster(content: bytes) -> bool:
    """True when every pixel of a PNG/JPEG has the same value in every channel (``min == max``).

    A uniform JPEG is a *blank candidate*: hips2fits renders both "no data" and its
    rendering failures as a uniform white JPEG (PS1 colour at (100, -70); VLASS at the
    Crab from alaskybis). It is not proof of either: a genuinely flat field is uniform
    too (ROSAT counts of 0 in a 12" field render uniform black, or uniform white with
    ``cmap=Greys``; verified 2026-09-28), so :class:`CutoutService` settles it with the
    MOC and the equivalent PNG.
    """
    import numpy as np

    with _decode_raster(content) as image:
        pixels = np.asarray(image)
    return pixels.size > 0 and int(pixels.min()) == int(pixels.max())


def coverage_fraction(content: bytes, fmt: str) -> float | None:
    """Fraction of pixels that carry survey data, or None when it cannot be told.

    hips2fits marks pixels outside a survey footprint as transparent (alpha 0) in
    PNG output, as a zero alpha plane in 4-plane colour FITS, and as NaN in float
    FITS. JPEG has no transparency, so coverage is unknown from the JPEG alone
    (None); :class:`CutoutService` resolves uniform (possibly blank) JPEGs with
    :func:`is_uniform_raster`, the survey MOC and the equivalent PNG. Corrupt or
    truncated pixel data raises :class:`CutoutUpstreamError`.
    """
    import numpy as np

    if fmt == "fits":
        from astropy.io import fits

        try:
            with fits.open(io.BytesIO(content), memmap=False) as hdul:
                data = hdul[0].data
        except (OSError, ValueError, TypeError) as exc:
            raise CutoutUpstreamError(f"FITS pixels could not be decoded: {exc}") from exc
        if data is None:
            return 0.0
        if data.dtype.kind == "f":
            return float(np.isfinite(data).mean())
        if data.ndim == 3 and data.shape[0] == 4:
            return float((data[3] > 0).mean())
        return None
    if fmt in ("png", "jpg"):
        with _decode_raster(content) as image:
            return _raster_coverage(image)
    return None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


#: Without a size overrun, the cache directory is still swept for expired entries every N writes.
CACHE_PRUNE_EVERY_PUTS = 256


@dataclass(slots=True)
class CutoutCache:
    """Content-addressed on-disk cache of cutouts (``<sha256>.bin`` + ``<sha256>.json``).

    Entries expire after ``ttl_seconds``; the directory is pruned oldest-first
    (by modification time, refreshed on every hit) once it exceeds ``max_bytes``.
    ``ttl_seconds=0`` disables the cache.

    The directory is scanned only when needed: the size is tracked in memory after
    one initial scan, so a write prunes only when the tracked size crosses
    ``max_bytes`` or every :data:`CACHE_PRUNE_EVERY_PUTS` writes (expiry sweep).
    All methods do blocking file I/O; :class:`CutoutService` calls them through
    :func:`asyncio.to_thread`. Share one instance per directory (see :meth:`shared`).
    """

    directory: Path
    ttl_seconds: float = 7 * 86400
    max_bytes: int = 512 * 1024 * 1024
    _tracked_bytes: int | None = field(default=None, init=False, repr=False, compare=False)
    _puts_since_prune: int = field(default=0, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    @classmethod
    def from_env(cls) -> CutoutCache:
        """``CUTOUT_CACHE_DIR`` (default ``<tmp>/astrosearch-cutouts``), ``CUTOUT_CACHE_TTL_SECONDS``, ``CUTOUT_CACHE_MAX_BYTES``."""
        directory = os.getenv("CUTOUT_CACHE_DIR") or str(Path(tempfile.gettempdir()) / "astrosearch-cutouts")
        return cls(
            Path(directory),
            ttl_seconds=float(os.getenv("CUTOUT_CACHE_TTL_SECONDS", str(7 * 86400))),
            max_bytes=int(os.getenv("CUTOUT_CACHE_MAX_BYTES", str(512 * 1024 * 1024))),
        )

    @classmethod
    def shared(cls) -> CutoutCache:
        """The process-wide cache for the current environment settings (one per configuration)."""
        probe = cls.from_env()
        return _shared_cache(str(probe.directory), probe.ttl_seconds, probe.max_bytes)

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def _paths(self, key: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("cache key must be a sha256 hex digest")
        return self.directory / f"{key}.bin", self.directory / f"{key}.json"

    def get(self, key: str) -> tuple[bytes, dict[str, Any]] | None:
        if not self.enabled:
            return None
        data_path, meta_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if time.time() - float(meta.get("stored_at", 0)) > self.ttl_seconds:
                return None
            content = data_path.read_bytes()
            if len(content) != meta.get("bytes"):
                return None
            now = time.time()
            os.utime(data_path, (now, now))
            return content, meta
        except (OSError, ValueError, TypeError):
            return None

    def put(self, key: str, content: bytes, meta: dict[str, Any]) -> None:
        if not self.enabled:
            return
        data_path, meta_path = self._paths(key)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock:
                if self._tracked_bytes is None:
                    self._tracked_bytes = self.size_bytes()  # one scan per process, then tracked
            try:
                previous = data_path.stat().st_size
            except OSError:
                previous = 0
            _atomic_write(data_path, content)
            _atomic_write(meta_path, json.dumps({**meta, "bytes": len(content), "stored_at": time.time()}).encode())
            with self._lock:
                self._tracked_bytes = (self._tracked_bytes or 0) + len(content) - previous
                self._puts_since_prune += 1
                due = self._tracked_bytes > self.max_bytes or self._puts_since_prune >= CACHE_PRUNE_EVERY_PUTS
            if due:
                self.prune()
        except OSError as exc:
            logger.warning("cutout_cache_write_failed", error=str(exc), directory=str(self.directory))

    def size_bytes(self) -> int:
        if not self.directory.exists():
            return 0
        total = 0
        for path in self.directory.glob("*.bin"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def prune(self) -> int:
        """Drop expired entries, then the least recently used until under ``max_bytes``; returns count removed."""
        if not self.directory.exists():
            return 0
        entries = []
        for data_path in self.directory.glob("*.bin"):
            try:
                stat = data_path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, data_path))
        entries.sort()
        total = sum(size for _, size, _ in entries)
        removed = 0
        now = time.time()
        for mtime, size, data_path in entries:
            if total <= self.max_bytes and now - mtime <= self.ttl_seconds:
                continue
            for path in (data_path, data_path.with_suffix(".json")):
                try:
                    path.unlink()
                except OSError:
                    pass
            total -= size
            removed += 1
        with self._lock:
            self._tracked_bytes = total
            self._puts_since_prune = 0
        return removed

    def clear(self) -> int:
        removed = 0
        if self.directory.exists():
            for path in list(self.directory.glob("*.bin")) + list(self.directory.glob("*.json")):
                try:
                    path.unlink()
                    removed += path.suffix == ".bin"
                except OSError:
                    pass
        with self._lock:
            self._tracked_bytes = None
            self._puts_since_prune = 0
        return removed


@functools.lru_cache(maxsize=8)
def _shared_cache(directory: str, ttl_seconds: float, max_bytes: int) -> CutoutCache:
    return CutoutCache(Path(directory), ttl_seconds=ttl_seconds, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# Cutout service (hips2fits + MocServer)
# ---------------------------------------------------------------------------


#: Largest accepted proper motion (mas/yr): Barnard's star, the fastest known star, moves 10.4"/yr.
MAX_PM_MASYR = 20_000.0


@dataclass(frozen=True, slots=True)
class ProperMotion:
    """Target proper motion (mas/yr; ``pm_ra_masyr`` includes cos dec) at reference ``epoch`` (Julian year)."""

    pm_ra_masyr: float
    pm_dec_masyr: float
    epoch: float = 2000.0

    def __post_init__(self) -> None:
        for name in ("pm_ra_masyr", "pm_dec_masyr", "epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise CutoutValidationError(f"{name} must be a finite number")
        if math.hypot(self.pm_ra_masyr, self.pm_dec_masyr) > MAX_PM_MASYR:
            raise CutoutValidationError(f"proper motion must not exceed {MAX_PM_MASYR:g} mas/yr")
        if not 1800.0 <= self.epoch <= 2200.0:
            raise CutoutValidationError("epoch must be a Julian year in [1800, 2200]")

    @property
    def total_arcsec_per_year(self) -> float:
        return math.hypot(self.pm_ra_masyr, self.pm_dec_masyr) / 1000.0

    def position_at(self, ra: float, dec: float, epoch: float) -> tuple[float, float]:
        """Linearly propagated (ra, dec) at ``epoch`` (parallax and radial velocity ignored:
        sub-arcsecond even for Barnard's star, far below a cutout pixel)."""
        from models import propagate_radec

        new_ra, new_dec = propagate_radec(ra, dec, self.pm_ra_masyr, self.pm_dec_masyr, self.epoch, epoch)
        return new_ra % 360.0, new_dec

    def as_dict(self) -> dict[str, float]:
        return {"pm_ra_masyr": self.pm_ra_masyr, "pm_dec_masyr": self.pm_dec_masyr, "epoch": self.epoch}


@dataclass(frozen=True, slots=True)
class PanelCentre:
    """Where a stack panel is centred and why.

    ``epoch`` is the survey mean epoch the target was moved to (None: not moved),
    ``offset_arcsec`` the distance from the reference position, and
    ``spread_arcsec`` how far the target moves either side of the centre during the
    survey's observing span (None when unknown).
    """

    ra: float
    dec: float
    epoch: float | None = None
    offset_arcsec: float = 0.0
    spread_arcsec: float | None = None
    note: str | None = None


def panel_centre(
    survey: HipsSurvey, ra: float, dec: float, motion: ProperMotion | None, fov_arcmin: float | None = None
) -> PanelCentre:
    """Centre of ``survey``'s panel for a target at (ra, dec) moving with ``motion``.

    Without a proper motion the panel is centred on (ra, dec). With one, it is
    centred on the linearly propagated position at the survey's mean observing
    epoch (middle of :attr:`HipsSurvey.epoch_span`), which bounds the error by the
    motion over half the span. A note warns when that bound exceeds half the field
    (the target may be outside the image) or when the survey epoch is unknown.
    """
    if motion is None or motion.total_arcsec_per_year == 0.0:
        return PanelCentre(ra, dec)
    rate = motion.total_arcsec_per_year
    half_field = fov_arcmin * 30.0 if fov_arcmin else math.inf
    if survey.epoch_span is None:
        return PanelCentre(ra, dec, note=(
            f"Survey epoch unknown: centred on the epoch-{motion.epoch:g} position, but the target moves "
            f"{rate:.3g}\"/yr."))
    epoch = survey.mean_epoch
    assert epoch is not None
    centre_ra, centre_dec = motion.position_at(ra, dec, epoch)
    offset = rate * abs(epoch - motion.epoch)
    spread = rate * (survey.epoch_span[1] - survey.epoch_span[0]) / 2.0
    note = None
    if spread > half_field:
        note = (f"The target moves up to {spread:.0f}\" from this centre during the survey's "
                f"{survey.epoch_span[0]:.1f}-{survey.epoch_span[1]:.1f} observations and may lie outside the field.")
    return PanelCentre(centre_ra, centre_dec, epoch, round(offset, 3), round(spread, 3), note)


@dataclass(slots=True)
class StackPanel:
    """One panel of a multi-wavelength stack."""

    slot: str
    survey: str
    label: str
    hips_id: str
    regime: str
    wavelength: str
    wavelength_m: float | None
    in_coverage: bool | None
    color: bool = False
    url: str | None = None
    #: Single-band FITS of the same field: the survey itself, or a single-band companion that covers it.
    fits_survey: str | None = None
    fits_label: str | None = None
    fits_url: str | None = None
    #: What the FITS pixels are (``Jy/beam``, ``DN``, ``photographic density`` ...) and whether they are
    #: in physical flux units as delivered.
    fits_pixel_units: str | None = None
    fits_calibrated: bool | None = None
    #: Why the FITS link is missing or is not the first-choice companion.
    fits_note: str | None = None
    #: Panel centre (ICRS deg): the target position, moved by its proper motion to ``epoch`` when known.
    center_ra: float | None = None
    center_dec: float | None = None
    epoch: float | None = None
    offset_arcsec: float = 0.0
    position_spread_arcsec: float | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["wavelength_m"] is not None and not math.isfinite(data["wavelength_m"]):
            data["wavelength_m"] = None
        return data


@dataclass(slots=True)
class CoverageResult:
    """Which surveys have data at a position (None = could not be determined).

    ``error_kind`` says why the footprint is unknown: ``transport`` (connection
    failure), ``http5xx`` (server error), ``http4xx`` (request refused) or
    ``parse`` (the MocServer answered something that is not an ID list).
    """

    covered: dict[str, bool | None]
    checked: bool
    error: str | None = None
    error_kind: str | None = None

    @property
    def outage(self) -> bool:
        """True when the MocServer was merely unreachable (transport errors or 5xx)."""
        return self.error_kind in ("transport", "http5xx")


_FAILURE_PRIORITY = ("parse", "unusable", "http4xx", "http5xx", "transport")


def _worst(kinds: Iterable[str]) -> str | None:
    found = set(kinds)
    return next((k for k in _FAILURE_PRIORITY if k in found), None)


_BLANK_KINDS = ("no-data", "unconfirmed", "rendering-failure")


def _cutout_from_cache(request: CutoutRequest, content: bytes, meta: dict[str, Any]) -> Cutout | None:
    """Rebuild a cached cutout, or None (a miss) when the entry does not describe this request.

    A shared ``CUTOUT_CACHE_DIR`` may hold entries written by another code version or
    damaged by hand: metadata with missing or ill-typed fields, or bytes that are not
    the requested image, count as a miss instead of failing the request.
    """
    try:
        info = probe_image(content)
    except CutoutUpstreamError:
        return None
    width, height = meta.get("width"), meta.get("height")
    strings = [meta.get(k) for k in ("source_url", "endpoint", "fetched_at")]
    coverage = meta.get("coverage_fraction")
    blank = meta.get("blank")
    if (info.format != request.format or (info.width, info.height) != (request.width, request.height)
            or width != request.width or height != request.height
            or not all(isinstance(s, str) and s for s in strings)
            or not (coverage is None or (isinstance(coverage, int | float) and not isinstance(coverage, bool)
                                         and 0.0 <= coverage <= 1.0))
            or blank not in (None, *_BLANK_KINDS)):
        return None
    if blank is None and coverage == 0.0:
        blank = "no-data"  # only MOC-confirmed blanks are ever cached (entries written before 'blank' existed)
    cutout = Cutout(request, content, request.media_type, request.width, request.height, strings[0], strings[1],
                    True, strings[2], None if coverage is None else float(coverage), info=info, blank=blank)
    if "science" in meta:
        science = SURVEYS.get(meta["science"]) if isinstance(meta["science"], str) else None
        note = meta.get("science_note")
        if meta["science"] is None or science is not None:
            cutout.science, cutout.science_note = science, note if isinstance(note, str) else None
            cutout.science_checked = True
    return cutout


class CutoutService:
    """Fetch cutouts from hips2fits with mirror failover, validation and a disk cache.

    ``client`` may be shared (e.g. ``app.state.client``); when None a short-lived
    client is opened per call.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        cache: CutoutCache | None = None,
        endpoints: Sequence[str] | None = None,
        moc_endpoints: Sequence[str] | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.client = client
        self.cache = cache if cache is not None else CutoutCache.shared()
        env_urls = [x.strip() for x in os.getenv("HIPS2FITS_URLS", "").split(",") if x.strip()]
        self.endpoints = tuple(endpoints or env_urls or HIPS2FITS_URLS)
        self.moc_endpoints = tuple(moc_endpoints or MOCSERVER_URLS)
        self.timeout = timeout

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.client is not None:
            yield self.client
            return
        async with _new_client(self.timeout) as client:
            yield client

    async def cutout(self, request: CutoutRequest, *, use_cache: bool = True) -> Cutout:
        """Return the cutout for ``request`` (from the disk cache when fresh).

        Endpoints are tried in order. A transport error, HTTP 5xx, 429, a 404 that
        does not name an unknown HiPS, or a body that is not the complete requested
        image (HTML error page, wrong size, truncated or undecodable pixels) moves on
        to the next mirror; only an unknown HiPS or another 4xx is final.

        Blank images get the same failover. hips2fits emits a blank image both for
        "no data here" and as a rendering failure (VLASS at the Crab), so a blank
        answer (no data pixels in PNG/FITS; a uniform JPEG, see :func:`is_uniform_raster`)
        is judged by the survey's MOC (:meth:`_blank_verdict`):

        * the MOC does not reach the field: a genuine "no data" (``blank='no-data'``),
          returned and cached at once;
        * otherwise the blank is recorded as a failure and the next endpoint is tried.
          Only when no endpoint returns a non-blank image is the (first) blank returned,
          never cached: ``blank='rendering-failure'`` and ``degraded`` when the MOC
          overlaps the field (``degraded`` also when another endpoint failed),
          ``blank='unconfirmed'`` when the footprint could not be checked.

        A uniform JPEG whose field the MOC may cover is checked against the same
        cutout as PNG from the same endpoint (its alpha plane marks the data pixels):
        a flat but real field (ROSAT counts of 0) is kept, with the PNG's coverage.

        ``use_cache=False`` neither reads nor writes the disk cache.

        When every endpoint answers HTTP 5xx to a request with ``min_cut``/``max_cut``,
        the same cutout is tried once without the cuts: if that renders, the cuts
        are the problem and :class:`CutoutValidationError` (HTTP 422) is raised.
        Blocking work (disk cache, image decoding) runs in worker threads.
        """
        key = request.cache_key()
        if use_cache:
            hit = await asyncio.to_thread(self.cache.get, key)
            cached = _cutout_from_cache(request, *hit) if hit is not None else None
            if cached is not None:
                if cached.pixel_kind == "rgb-preview" and request.hips.color and not cached.science_checked:
                    await self._attach_science(request, cached)
                return cached

        params = request.params()
        failures: list[tuple[str, str]] = []  # (kind, message), for endpoints without a usable image
        blanks: list[Cutout] = []  # blank images not confirmed as "no data"
        footprint: list[CoverageResult] = []  # the MOC answer, asked at most once per call
        async with self._client() as client:
            for endpoint in self.endpoints:
                host = _host(endpoint)
                try:
                    response = await client.get(endpoint, params=params, timeout=self.timeout)
                except httpx.HTTPError as exc:
                    failures.append(("transport", f"{host}: {type(exc).__name__}: {exc}"))
                    continue
                status = response.status_code
                if status >= 400:
                    detail = _detail(response)
                    if "unknown hips" in detail.lower() or "could not find a hips" in detail.lower():
                        raise UnknownSurveyError(f"hips2fits does not know HiPS '{request.hips.hips_id}': {detail}")
                    if status >= 500:
                        failures.append(("http5xx", f"{host}: HTTP {status} {detail}".rstrip()))
                        continue
                    if status in (404, 408, 429):
                        failures.append(("http4xx", f"{host}: HTTP {status} {detail}".rstrip()))
                        continue
                    raise CutoutUpstreamError(f"hips2fits rejected the request (HTTP {status}): {detail}")
                try:
                    cutout = await asyncio.to_thread(self._accept, request, response, endpoint)
                    uniform_jpeg = (request.format == "jpg"
                                    and await asyncio.to_thread(is_uniform_raster, cutout.content))
                except CutoutUpstreamError as exc:
                    failures.append(("unusable", f"{host}: {exc}"))
                    continue
                if cutout.coverage_fraction == 0.0 or uniform_jpeg:
                    verdict = await self._blank_verdict(request, footprint)
                    if verdict is False:  # the MOC does not reach the field: a real "no data" answer
                        cutout.coverage_fraction = 0.0
                        cutout.blank = "no-data"
                    elif not (uniform_jpeg and await self._jpeg_has_data(client, request, endpoint, cutout)):
                        blanks.append(cutout)
                        logger.warning("cutout_blank", survey=request.hips.key, endpoint=host, moc_overlap=verdict)
                        continue
                return await self._finish(request, cutout, key, params, use_cache)
            if blanks:
                return self._blank_result(request, blanks, failures, footprint[0] if footprint else None)
            errors = "; ".join(message for _, message in failures)
            kinds = {kind for kind, _ in failures}
            has_cuts = request.min_cut is not None or request.max_cut is not None
            if has_cuts and kinds == {"http5xx"} and await self._renders_without_cuts(client, request):
                raise CutoutValidationError(
                    f"hips2fits cannot render this cutout with min_cut={request.min_cut}, max_cut={request.max_cut} "
                    f"(HTTP 5xx on every endpoint, while the same cutout without cuts renders). A pixel-value cut "
                    f"outside the image's values, or a percentile/value pair that crosses (e.g. min_cut=1% above "
                    f"max_cut=0.02), causes this. Details: {errors}")
        if "unusable" in kinds:
            raise CutoutUpstreamError("hips2fits returned no usable image: " + errors)
        raise CutoutUpstreamError("hips2fits unavailable: " + errors)

    async def _finish(
        self, request: CutoutRequest, cutout: Cutout, key: str, params: dict[str, str], use_cache: bool
    ) -> Cutout:
        """Attach the colour-FITS companion, store a good cutout in the cache and return it."""
        if cutout.pixel_kind == "rgb-preview" and request.hips.color:
            await self._attach_science(request, cutout)
        if use_cache and cutout.cacheable:
            meta: dict[str, Any] = {
                "width": cutout.width, "height": cutout.height, "source_url": cutout.source_url,
                "endpoint": cutout.endpoint, "fetched_at": cutout.fetched_at,
                "coverage_fraction": cutout.coverage_fraction, "blank": cutout.blank, "params": params,
            }
            if cutout.science_checked:
                meta["science"] = cutout.science.key if cutout.science else None
                meta["science_note"] = cutout.science_note
            await asyncio.to_thread(self.cache.put, key, cutout.content, meta)
        logger.info("cutout_fetched", survey=request.hips.key, endpoint=_host(cutout.endpoint),
                    bytes=len(cutout.content), coverage=cutout.coverage_fraction, blank=cutout.blank)
        return cutout

    async def _blank_verdict(self, request: CutoutRequest, memo: list[CoverageResult]) -> bool | None:
        """Does the survey's MOC reach the field (the circle circumscribing the image)?

        True / False, or None when the MocServer cannot be asked. Asked once per
        :meth:`cutout` call (``memo``). MOCs are coarse supersets of the data, so a
        blank image at the very edge of a footprint can be judged a rendering failure
        although it is merely empty; that errs on the side of not caching.
        """
        if not memo:
            ra, dec = request.icrs_position()
            memo.append(await self.coverage(ra, dec, [request.hips], radius_deg=request.field_radius_deg))
        return memo[0].covered.get(request.hips.key)

    async def _jpeg_has_data(self, client: httpx.AsyncClient, request: CutoutRequest, endpoint: str,
                             cutout: Cutout) -> bool:
        """Settle a uniform JPEG with the same cutout as PNG from the same endpoint.

        The PNG's alpha plane marks the data pixels. With data, the JPEG is a flat but
        real field: its coverage becomes the PNG's and True is returned. A blank PNG
        sets the JPEG's coverage to 0.0; a failed PNG request leaves it unknown (None).
        """
        png_request = replace(request, format="png")
        try:
            response = await client.get(endpoint, params=png_request.params(), timeout=self.timeout)
            if response.status_code != 200:
                raise CutoutUpstreamError(f"HTTP {response.status_code}")
            png = await asyncio.to_thread(self._accept, png_request, response, endpoint)
        except (httpx.HTTPError, CutoutUpstreamError) as exc:
            logger.warning("cutout_png_check_failed", survey=request.hips.key, endpoint=_host(endpoint),
                           error=str(exc))
            return False
        if png.coverage_fraction is None:  # a PNG without alpha cannot settle it
            return False
        if png.coverage_fraction > 0.0:
            cutout.coverage_fraction = png.coverage_fraction
            return True
        cutout.coverage_fraction = 0.0
        return False

    def _blank_result(self, request: CutoutRequest, blanks: list[Cutout], failures: list[tuple[str, str]],
                      footprint: CoverageResult | None) -> Cutout:
        """The cutout to return when every usable answer was blank (never cached)."""
        cutout = blanks[0]
        cutout.cacheable = False
        overlaps = footprint.covered.get(request.hips.key) if footprint else None
        hosts = " and ".join(dict.fromkeys(_host(b.endpoint) for b in blanks))
        what = "blank (uniform) JPEG" if request.format == "jpg" else "blank image"
        footprint_text = f"the {request.hips.label} footprint (CDS MocServer MOC) overlaps this field"
        cutout.blank = "rendering-failure" if overlaps else "unconfirmed"
        if failures:
            mirror = "mirror " if cutout.endpoint != self.endpoints[0] else ""
            cutout.degraded = f"{what} from {mirror}{hosts} after " + "; ".join(m for _, m in failures)
            if overlaps:
                cutout.degraded += f"; {footprint_text}"
        elif overlaps:
            cutout.degraded = f"{what} from {hosts} although {footprint_text}"
        if cutout.degraded:
            logger.warning("cutout_degraded", survey=request.hips.key, reason=cutout.degraded)
        else:
            logger.warning("cutout_blank_unconfirmed", survey=request.hips.key,
                           error=footprint.error if footprint else None)
        return cutout

    async def _attach_science(self, request: CutoutRequest, cutout: Cutout) -> None:
        """Pick the single-band companion of a colour FITS that covers its centre (as the stack does)."""
        candidates = companion_surveys(request.hips)
        ra, dec = request.icrs_position()
        covered = (await self.coverage(ra, dec, candidates)).covered if candidates else {}
        cutout.science, cutout.science_note = fits_companion(request.hips, covered)
        cutout.science_checked = True

    async def _renders_without_cuts(self, client: httpx.AsyncClient, request: CutoutRequest) -> bool:
        params = {k: v for k, v in request.params().items() if k not in ("min_cut", "max_cut")}
        for endpoint in self.endpoints:
            try:
                response = await client.get(endpoint, params=params, timeout=self.timeout)
            except httpx.HTTPError:
                continue
            if response.status_code == 200:
                try:
                    return probe_image(response.content).format == request.format
                except CutoutUpstreamError:
                    continue
        return False

    def _accept(self, request: CutoutRequest, response: httpx.Response, endpoint: str) -> Cutout:
        content = response.content
        try:
            info = probe_image(content)
        except CutoutUpstreamError as exc:
            raise CutoutUpstreamError(f"hips2fits returned an unusable body ({_detail(response)}): {exc}") from exc
        if info.format != request.format:
            raise CutoutUpstreamError(f"hips2fits returned {info.format} instead of {request.format}")
        if (info.width, info.height) != (request.width, request.height):
            raise CutoutUpstreamError(
                f"hips2fits returned {info.width}x{info.height} pixels, expected {request.width}x{request.height}")
        check_complete(content, info)
        coverage = coverage_fraction(content, request.format)  # full decode: raises on corrupt pixels
        return Cutout(request, content, request.media_type, info.width, info.height, str(response.request.url),
                      endpoint, False, datetime.now(UTC).isoformat(), coverage, info=info)

    async def coverage(
        self, ra: float, dec: float, surveys: Iterable[HipsSurvey], *, radius_deg: float = 1.0 / 3600.0
    ) -> CoverageResult:
        """Decide per survey whether its footprint reaches within ``radius_deg`` of (ra, dec).

        Full-sky and declination-limited surveys are decided locally (declination
        limits at the centre); the rest with one MocServer spatial query (``SR`` =
        ``radius_deg``, 1 arcsec by default, i.e. "is the position covered").
        ``error_kind`` distinguishes an outage (``transport``/``http5xx``) from a
        refused request or an answer that does not parse.
        """
        covered: dict[str, bool | None] = {}
        pending: dict[str, HipsSurvey] = {}
        for survey in surveys:
            local = survey.covered(dec)
            if local is None:
                pending[survey.key] = survey
            covered[survey.key] = local
        if not pending:
            return CoverageResult(covered, True)
        expr = "||".join(f"ID={s.hips_id}" for s in pending.values())
        params = {"RA": _num(ra), "DEC": _num(dec), "SR": _num(radius_deg), "expr": expr, "get": "id", "fmt": "json"}
        failures: list[tuple[str, str]] = []
        async with self._client() as client:
            for endpoint in self.moc_endpoints:
                host = _host(endpoint)
                try:
                    response = await client.get(endpoint, params=params, timeout=min(self.timeout, 30.0))
                except httpx.HTTPError as exc:
                    failures.append(("transport", f"{host}: {type(exc).__name__}: {exc}"))
                    continue
                if response.status_code >= 400:
                    kind = "http5xx" if response.status_code >= 500 else "http4xx"
                    failures.append((kind, f"{host}: HTTP {response.status_code}"))
                    continue
                try:
                    ids = response.json()
                    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                        raise TypeError(f"expected a JSON list of HiPS IDs, got {type(ids).__name__}")
                except (ValueError, TypeError) as exc:
                    failures.append(("parse", f"{host}: unparseable answer: {exc}"))
                    continue
                found = set(ids)
                for survey in pending.values():
                    covered[survey.key] = survey.hips_id in found
                return CoverageResult(covered, True)
        worst = _worst(k for k, _ in failures)
        label = "MocServer unavailable" if worst in ("transport", "http5xx") else "MocServer answer unusable"
        return CoverageResult(covered, False, f"{label}: " + "; ".join(m for _, m in failures), worst)

    async def describe_hips(self, hips_ids: Sequence[str]) -> dict[str, HipsSurvey]:
        """Registry description (title, wavelength, bibcode, colour) of ad-hoc HiPS IDs.

        One MocServer ``get=record`` query. IDs the registry does not list, or a
        failed query, are simply absent from the result (callers keep the bare ID).
        The registry's ``em_min``/``em_max`` are sometimes swapped (e.g. XMM), so
        they are sorted; a list of ``bib_reference`` values contributes the first.
        """
        wanted = [h for h in dict.fromkeys(hips_ids) if h]
        if not wanted:
            return {}
        params = {"expr": "||".join(f"ID={h}" for h in wanted), "get": "record", "fmt": "json",
                  "fields": "ID,obs_title,em_min,em_max,bib_reference,dataproduct_subtype,hips_tile_format"}
        async with self._client() as client:
            for endpoint in self.moc_endpoints:
                try:
                    response = await client.get(endpoint, params=params, timeout=min(self.timeout, 30.0))
                    response.raise_for_status()
                    records = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("hips_describe_failed", endpoint=_host(endpoint), error=str(exc))
                    continue
                if isinstance(records, dict):
                    records = [records]
                out: dict[str, HipsSurvey] = {}
                for record in records if isinstance(records, list) else []:
                    survey = _survey_from_record(record) if isinstance(record, dict) else None
                    if survey is not None and survey.hips_id in wanted:
                        out[survey.hips_id] = survey
                return out
        return {}

    async def plan_stack(
        self,
        ra: float,
        dec: float,
        surveys: Sequence[str] | None = None,
        *,
        motion: ProperMotion | None = None,
        fov_arcmin: float | None = None,
    ) -> tuple[list[StackPanel], CoverageResult]:
        """Multi-wavelength panels ordered radio -> X-ray.

        With ``surveys`` the given surveys are used as-is (ordered by wavelength);
        otherwise each :data:`STACK_SLOTS` slot picks its first survey that covers the
        position. When the MocServer cannot be reached a slot takes a survey known to
        cover the position locally (full-sky / declination limit), else its first choice.

        The footprints of the single-band FITS companions are checked in the same
        MocServer query, so a panel links the first companion with data (or none).
        With ``motion`` each panel is centred on the target at the survey's mean epoch
        (:func:`panel_centre`; ``fov_arcmin`` sizes the "may be outside" warning).
        """
        def with_companions(items: Iterable[HipsSurvey]) -> dict[str, HipsSurvey]:
            out: dict[str, HipsSurvey] = {}
            for item in items:
                out.setdefault(item.key, item)
                for companion in companion_surveys(item):
                    out.setdefault(companion.key, companion)
            return out

        def build(slot: str, survey: HipsSurvey, covered: dict[str, bool | None]) -> StackPanel:
            return _panel(slot, survey, covered, panel_centre(survey, ra, dec, motion, fov_arcmin))

        if surveys:
            chosen = list({s.key: s for s in (get_survey(x) for x in surveys)}.values())
            adhoc = [s.hips_id for s in chosen if s.key == s.hips_id and s.key not in SURVEYS]
            if adhoc:
                described = await self.describe_hips(adhoc)
                chosen = [described.get(s.hips_id, s) if s.hips_id in adhoc else s for s in chosen]
            coverage = await self.coverage(ra, dec, with_companions(chosen).values())
            ordered = sorted(chosen, key=lambda s: -s.wavelength_m if s.has_wavelength else 0.0)
            return [build(s.regime, s, coverage.covered) for s in ordered], coverage
        candidates = with_companions(SURVEYS[key] for _, keys in STACK_SLOTS for key in keys)
        coverage = await self.coverage(ra, dec, candidates.values())
        panels: list[StackPanel] = []
        used: set[str] = set()
        for slot, keys in STACK_SLOTS:
            available = [k for k in keys if k not in used]
            if not available:
                continue
            pick = next((k for k in available if coverage.covered.get(k) is True), None)
            if pick is None and not coverage.checked:
                pick = available[0]
            if pick is None:
                continue
            used.add(pick)
            panels.append(build(slot, SURVEYS[pick], coverage.covered))
        return panels, coverage


def _survey_from_record(record: dict[str, Any]) -> HipsSurvey | None:
    """Ad-hoc :class:`HipsSurvey` from a MocServer record (``ID``, ``em_min``, ``em_max`` ...)."""
    hips_id = str(record.get("ID") or "").strip()
    if not _HIPS_ID_RE.match(hips_id):
        return None

    def as_float(value: Any) -> float:
        try:
            number = float(value[0] if isinstance(value, list) else value)
        except (TypeError, ValueError, IndexError):
            return math.nan
        return number if math.isfinite(number) and number > 0 else math.nan

    lo, hi = as_float(record.get("em_min")), as_float(record.get("em_max"))
    if math.isfinite(lo) and math.isfinite(hi):
        lo, hi = sorted((lo, hi))
    else:
        lo = hi = math.nan
    bib = record.get("bib_reference")
    bib = bib[0] if isinstance(bib, list) and bib else bib
    bib = str(bib) if bib and len(str(bib)) == 19 and str(bib)[:4].isdigit() else None  # ADS bibcodes are 19 chars
    subtype = str(record.get("dataproduct_subtype") or "")
    tiles = str(record.get("hips_tile_format") or "")
    title = record.get("obs_title")
    title = str(title[0] if isinstance(title, list) and title else title or hips_id)[:120]
    wavelength = math.sqrt(lo * hi) if math.isfinite(lo) else math.nan
    return HipsSurvey(key=hips_id, hips_id=hips_id, label=title, regime=regime_for_wavelength(wavelength),
                      em_min_m=lo, em_max_m=hi, bib_reference=bib,
                      color="color" in subtype or "fits" not in tiles.split(), coverage="moc",
                      note="Ad-hoc HiPS described from its CDS MocServer record.")


def _panel(slot: str, survey: HipsSurvey, covered: dict[str, bool | None], centre: PanelCentre) -> StackPanel:
    fits, fits_note = fits_companion(survey, covered)
    return StackPanel(
        slot, survey.key, survey.label, survey.hips_id, survey.regime, survey.wavelength_label,
        survey.wavelength_m if survey.has_wavelength else None, covered.get(survey.key),
        color=survey.color, fits_survey=fits.key if fits else None, fits_label=fits.label if fits else None,
        fits_pixel_units=(fits.pixel_units or "unknown") if fits else None,
        fits_calibrated=fits.calibrated if fits else None, fits_note=fits_note,
        center_ra=centre.ra, center_dec=centre.dec, epoch=centre.epoch, offset_arcsec=centre.offset_arcsec,
        position_spread_arcsec=centre.spread_arcsec, note=centre.note,
    )


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """One default SSL context for short-lived clients (building one costs ~0.5 s on Windows)."""
    return ssl.create_default_context()


def _new_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=_ssl_context())


def _host(url: str) -> str:
    return httpx.URL(url).host


def _header_text(text: str, limit: int = 300) -> str:
    """Single-line, latin-1-safe HTTP header value."""
    flat = " ".join(str(text).split())[:limit]
    return flat.encode("latin-1", "replace").decode("latin-1")


_TEXTUAL = ("text/", "application/json", "application/problem+json", "application/xml", "application/xhtml")


def _detail(response: httpx.Response) -> str:
    """Short, printable description of an error body: its message when textual, else its size and type.

    Binary bodies (an image or FITS served with an error, or a mislabelled answer)
    are never pasted into error messages.
    """
    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                text = " - ".join(str(payload[k]) for k in ("title", "description", "detail") if payload.get(k))
                if text:
                    return _printable(text)[:300]
        except ValueError:
            pass
    if content_type.startswith(_TEXTUAL):
        return _printable(response.text)[:300].strip()
    return f"{len(response.content)} bytes of {content_type or 'unlabelled content'}"


def _printable(text: str) -> str:
    return "".join(ch if ch.isprintable() or ch in "\n\t" else "?" for ch in text)


async def fetch_cutout(
    ra: float,
    dec: float,
    *,
    survey: str = "dss2",
    fov_arcmin: float = 5.0,
    width: int = 512,
    height: int = 512,
    format: str = "png",
    client: httpx.AsyncClient | None = None,
    cache: CutoutCache | None = None,
    **options: Any,
) -> Cutout:
    """Notebook-friendly one-call cutout: ``await fetch_cutout(187.278, 2.052, survey="nvss", format="fits")``."""
    request = CutoutRequest(ra=ra, dec=dec, survey=survey, fov_arcmin=fov_arcmin, width=width, height=height,
                            format=format, **options)
    return await CutoutService(client, cache=cache).cutout(request)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["imaging"])


class SurveyModel(BaseModel):
    key: str
    hips_id: str
    label: str
    regime: str
    band: str | None
    wavelength: str
    wavelength_m: float | None
    frequency_hz: float | None
    em_min_m: float | None
    em_max_m: float | None
    color: bool
    science: str | None = Field(None, description="Single-band companion with FITS survey pixel values (colour surveys)")
    science_alternates: list[str] = Field(default_factory=list,
                                          description="Other single-band companions, used where `science` has no data")
    pixel_units: str | None = Field(None, description="What single-band FITS pixels hold (Jy/beam, DN, counts ...)")
    calibrated: bool = Field(False, description="FITS pixels are in physical flux / surface-brightness units")
    pixel_note: str | None = None
    epoch_span: list[float] | None = Field(None, description="First and last Julian year of the observations")
    mean_epoch: float | None = None
    coverage: str
    bib_reference: str | None
    bib_url: str | None
    note: str | None


class PanelModel(BaseModel):
    slot: str
    survey: str
    label: str
    hips_id: str
    regime: str
    wavelength: str
    wavelength_m: float | None
    in_coverage: bool | None
    color: bool = False
    url: str = Field(description="Root-relative URL of the display cutout (PNG/JPEG) on this API")
    fits_survey: str | None = Field(None, description="Single-band survey whose FITS carries survey pixel values")
    fits_label: str | None = None
    fits_url: str | None = Field(None, description="Root-relative URL of the single-band FITS cutout, or null")
    fits_pixel_units: str | None = Field(None, description="What the FITS pixels hold (Jy/beam, DN, counts ...)")
    fits_calibrated: bool | None = Field(None, description="FITS pixels are in physical flux units as delivered")
    fits_note: str | None = Field(None, description="Why the FITS link is missing or not the first-choice band")
    center_ra: float | None = Field(None, description="Panel centre (ICRS deg), moved for proper motion")
    center_dec: float | None = None
    epoch: float | None = Field(None, description="Survey mean epoch the target was moved to (null: not moved)")
    offset_arcsec: float = 0.0
    position_spread_arcsec: float | None = None
    note: str | None = None


class StackResponse(BaseModel):
    target: dict[str, Any]
    fov_arcmin: float
    size_px: int
    format: str
    coverage_checked: bool
    coverage_error: str | None = None
    proper_motion: dict[str, float] | None = None
    panels: list[PanelModel] = Field(default_factory=list)


def _service(request: Request) -> CutoutService:
    state = request.app.state
    cache = getattr(state, "cutout_cache", None)
    return CutoutService(getattr(state, "client", None), cache=cache)


def name_unknown_to_sesame(exc: BaseException) -> bool:
    """True when a Sesame resolution error is the client's: an unknown name (HTTP 404) or an
    empty one (422). Transport errors, HTTP errors and answers that are not XML are upstream
    failures (503/502). See :func:`models.resolution_failure_status`."""
    from models import resolution_failure_status

    return resolution_failure_status(exc) in (404, 422)


async def _resolve_position(
    request: Request, ra: float | None, dec: float | None, name: str | None
) -> tuple[float, float, dict[str, Any] | None]:
    """The target of ``/cutouts`` and ``/cutouts/stack``: ra/dec as given, or the Sesame position of
    ``name``. A name together with ra and/or dec is a 422 (:data:`main.NAME_AND_COORDINATES`)
    before anything is resolved, as on every search route, so an image is never rendered at the
    coordinates while being requested (and labelled) as the named object."""
    from main import check_search_target  # lazy: main imports this module for its CLI

    try:
        check_search_target(name, ra, dec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if ra is not None and dec is not None:
        return ra, dec, None
    assert name  # check_search_target: a name when ra/dec are absent
    from models import ObjectResolutionError, resolution_failure_status
    from providers import SesameResolver

    client = getattr(request.app.state, "client", None)
    owned = client is None
    active = client or _new_client(30.0)
    try:
        resolved = await SesameResolver(active).resolve(name)
    except ObjectResolutionError as exc:
        # The same statuses as every route: 404 unknown name, 503 resolver down, 502 unusable answer.
        status = resolution_failure_status(exc)
        raise HTTPException(status_code=status, detail=f"Could not resolve '{name}': {exc}",
                            headers={"Retry-After": "30"} if status == 503 else None) from exc
    finally:
        if owned:
            await active.aclose()
    info: dict[str, Any] = {"name": name, "canonical_name": resolved.canonical_name, "resolver": resolved.resolver}
    # The resolver decides whether the catalogued motion is physical: for extragalactic objects
    # (3C 273: Gaia pm -0.02/+0.10 mas/yr, measurement noise) it is not applied.
    motion = resolver_motion(resolved)
    if motion is not None:
        info.update(motion.as_dict())
    return resolved.ra_deg, resolved.dec_deg, info


def resolver_motion(resolved: Any) -> ProperMotion | None:
    """The proper motion of a resolver answer the cutouts follow, or None: only when the
    resolver deems it physical (``proper_motion_applicable``; not the Gaia noise of a quasar).
    Sesame/SIMBAD positions are ICRS at epoch J2000 unless the resolver says otherwise."""
    applicable = resolved.resolver_metadata.get("proper_motion_applicable")
    if applicable is None:
        applicable = resolved.pm_ra_masyr is not None and resolved.pm_dec_masyr is not None
    if not applicable or resolved.pm_ra_masyr is None or resolved.pm_dec_masyr is None:
        return None
    return ProperMotion(resolved.pm_ra_masyr, resolved.pm_dec_masyr,
                        resolved.epoch if resolved.epoch is not None else 2000.0)


@router.get("/cutouts/surveys", response_model=list[SurveyModel])
async def cutout_surveys() -> list[dict[str, Any]]:
    """HiPS surveys available for cutouts, ordered radio -> X-ray, with FITS pixel units and epochs."""
    return list_surveys()


@router.get("/cutouts/stack", response_model=StackResponse)
async def cutout_stack(
    request: Request,
    ra: float | None = Query(None, ge=0, lt=360, description="ICRS right ascension (deg)"),
    dec: float | None = Query(None, ge=-90, le=90, description="ICRS declination (deg)"),
    name: str | None = Query(None, max_length=200, description="Object name (CDS Sesame) instead of ra/dec"),
    fov_arcmin: float = Query(3.0, ge=MIN_FOV_ARCMIN, le=STACK_MAX_FOV_ARCMIN, description="Panel side (arcmin, TAN)"),
    size: int = Query(256, ge=MIN_SIZE_PX, le=1024, description="Panel width = height (pixels)"),
    format: Literal["png", "jpg"] = Query("png"),
    surveys: str | None = Query(None, description="Comma-separated survey keys (default: best per band)"),
    pm_ra_masyr: float | None = Query(None, ge=-MAX_PM_MASYR, le=MAX_PM_MASYR,
                                      description="Proper motion in RA (mas/yr, includes cos dec)"),
    pm_dec_masyr: float | None = Query(None, ge=-MAX_PM_MASYR, le=MAX_PM_MASYR,
                                       description="Proper motion in Dec (mas/yr)"),
    epoch: float | None = Query(None, ge=1800, le=2200,
                                description="Julian year of ra/dec (default 2000.0 when a proper motion is given)"),
) -> dict[str, Any]:
    """Multi-wavelength panel list (radio -> X-ray) with cutout URLs served by this API.

    With a proper motion (given, or from Sesame when ``name`` is used) each panel is
    centred on the target's position at that survey's mean observing epoch.
    """
    if (pm_ra_masyr is None) != (pm_dec_masyr is None):
        raise HTTPException(status_code=422, detail="pm_ra_masyr and pm_dec_masyr must be given together")
    ra, dec, resolved = await _resolve_position(request, ra, dec, name)
    motion: ProperMotion | None = None
    try:
        if pm_ra_masyr is not None and pm_dec_masyr is not None:
            motion = ProperMotion(pm_ra_masyr, pm_dec_masyr, 2000.0 if epoch is None else epoch)
        elif resolved and "pm_ra_masyr" in resolved:
            motion = ProperMotion(resolved["pm_ra_masyr"], resolved["pm_dec_masyr"], resolved["epoch"])
        keys = [s.strip() for s in surveys.split(",") if s.strip()] if surveys else None
        panels, coverage = await _service(request).plan_stack(ra, dec, keys, motion=motion, fov_arcmin=fov_arcmin)
    except CutoutValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Root-relative URLs (path incl. any root_path + query): they stay valid behind a
    # TLS-terminating proxy or under a sub-path, where scheme://host from the ASGI scope may not.
    base = request.url_for("get_cutout").path

    def cutout_url(survey_key: str, fmt: str, centre_ra: float, centre_dec: float) -> str:
        query = urlencode({"ra": _num(centre_ra), "dec": _num(centre_dec), "fov_arcmin": _num(fov_arcmin),
                           "survey": survey_key, "width": size, "height": size, "format": fmt})
        return f"{base}?{query}"

    out = []
    for panel in panels:
        centre = (ra if panel.center_ra is None else panel.center_ra, dec if panel.center_dec is None else panel.center_dec)
        panel.url = cutout_url(panel.survey, format, *centre)
        panel.fits_url = cutout_url(panel.fits_survey, "fits", *centre) if panel.fits_survey else None
        out.append(panel.as_dict())
    target: dict[str, Any] = {"ra": ra, "dec": dec}
    if resolved:
        target.update(resolved)
    return {"target": target, "fov_arcmin": fov_arcmin, "size_px": size, "format": format,
            "coverage_checked": coverage.checked, "coverage_error": coverage.error,
            "proper_motion": motion.as_dict() if motion else None, "panels": out}


@router.get(
    "/cutouts",
    name="get_cutout",
    response_class=Response,
    responses={200: {"content": {"image/png": {}, "image/jpeg": {}, "application/fits": {}}}},
)
async def get_cutout(
    request: Request,
    ra: float | None = Query(None, ge=0, lt=360, description="ICRS right ascension (deg)"),
    dec: float | None = Query(None, ge=-90, le=90, description="ICRS declination (deg)"),
    name: str | None = Query(None, max_length=200, description="Object name (CDS Sesame) instead of ra/dec"),
    fov_arcmin: float = Query(5.0, ge=MIN_FOV_ARCMIN, le=MAX_FOV_ARCMIN,
                              description="Largest image side (arcmin); TAN/SIN/AZP/SZP < 180 deg"),
    survey: str = Query("dss2", max_length=120, description="Survey key (see /cutouts/surveys) or HiPS ID"),
    format: Literal["png", "jpg", "jpeg", "fits"] = Query("png"),
    width: int = Query(512, ge=MIN_SIZE_PX, le=MAX_SIZE_PX),
    height: int = Query(512, ge=MIN_SIZE_PX, le=MAX_SIZE_PX),
    projection: str = Query("TAN", max_length=3),
    stretch: str | None = Query(None),
    cmap: str | None = Query(None, max_length=41),
    min_cut: str | None = Query(None, max_length=24),
    max_cut: str | None = Query(None, max_length=24),
    rotation_angle: float = Query(0.0, ge=-360, le=360),
    pm_ra_masyr: float | None = Query(None, ge=-MAX_PM_MASYR, le=MAX_PM_MASYR,
                                      description="Proper motion in RA (mas/yr, includes cos dec)"),
    pm_dec_masyr: float | None = Query(None, ge=-MAX_PM_MASYR, le=MAX_PM_MASYR,
                                       description="Proper motion in Dec (mas/yr)"),
    epoch: float | None = Query(None, ge=1800, le=2200,
                                description="Julian year of ra/dec (default 2000.0 when a proper motion is given)"),
) -> Response:
    """Image cutout (PNG/JPEG/FITS) of any catalogued HiPS survey, rendered by CDS hips2fits.

    With a proper motion (given, or from Sesame when ``name`` is used, as for
    ``/cutouts/stack``) the image is centred on the target's position at the survey's mean
    observing epoch; ``X-Cutout-Centre-Epoch`` and ``X-Cutout-Centre-Offset-Arcsec`` then
    say where it was moved (``X-Cutout-Centre-Note`` when the survey epoch is unknown or the
    target moves out of the field during the survey).

    Headers: ``X-Cutout-Pixels`` (``display`` | ``rgb-preview`` | ``survey``, read
    from the returned image), ``X-Cutout-Pixel-Units`` / ``X-Cutout-Calibrated`` for
    survey FITS, ``X-Cutout-Coverage`` (fraction of pixels with data), ``X-Cutout-Blank``
    (``no-data`` | ``unconfirmed`` | ``rendering-failure``), ``X-Cutout-Degraded``
    (rendering failure; never cached), ``X-Cutout-Pixel-Scale-Arcsec`` (angular pixel
    size at the image centre) and ``X-Cutout-Cdelt-Deg`` (the WCS CDELT hips2fits writes).
    """
    if (pm_ra_masyr is None) != (pm_dec_masyr is None):
        raise HTTPException(status_code=422, detail="pm_ra_masyr and pm_dec_masyr must be given together")
    ra, dec, resolved = await _resolve_position(request, ra, dec, name)
    centre: PanelCentre | None = None
    try:
        motion: ProperMotion | None = None
        if pm_ra_masyr is not None and pm_dec_masyr is not None:
            motion = ProperMotion(pm_ra_masyr, pm_dec_masyr, 2000.0 if epoch is None else epoch)
        elif resolved and "pm_ra_masyr" in resolved:
            motion = ProperMotion(resolved["pm_ra_masyr"], resolved["pm_dec_masyr"], resolved["epoch"])
        if motion is not None:
            centre = panel_centre(get_survey(survey), ra, dec, motion, fov_arcmin)
        cutout_request = CutoutRequest(
            ra=centre.ra if centre else ra, dec=centre.dec if centre else dec, fov_arcmin=fov_arcmin, survey=survey,
            width=width, height=height, format=format, projection=projection, stretch=stretch, cmap=cmap,
            min_cut=min_cut, max_cut=max_cut, rotation_angle=rotation_angle,
        )
        cutout = await _service(request).cutout(cutout_request)
    except CutoutValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CutoutUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    headers = cutout.headers()
    if centre is not None:
        if centre.epoch is not None:
            headers["X-Cutout-Centre-Epoch"] = f"{centre.epoch:.3f}"
            headers["X-Cutout-Centre-Offset-Arcsec"] = f"{centre.offset_arcsec:.3f}"
        if centre.note:
            headers["X-Cutout-Centre-Note"] = _header_text(centre.note)
    return Response(content=cutout.content, media_type=cutout.media_type, headers=headers)


# ---------------------------------------------------------------------------
# Web UI mount
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent / "web"
_UI_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".txt": "text/plain; charset=utf-8",
}
RESERVED_PREFIXES = ("/api", "/vo")
_UI_CACHE_CONTROL = "no-cache"  # always revalidate (ETag/Last-Modified) so UI updates show at once


def _etag_matches(header: str, etag: str) -> bool:
    """If-None-Match comparison (RFC 9110 13.1.2: weak comparison, ``*`` matches any)."""
    strip = etag.removeprefix("W/")
    for tag in header.split(","):
        tag = tag.strip()
        if tag == "*" or tag.removeprefix("W/") == strip:
            return True
    return False


def ui_file_response(path: Path, media_type: str, request_headers: Mapping[str, str]) -> Response:
    """A UI file with ETag/Last-Modified and ``Cache-Control: no-cache``, or ``304 Not
    Modified`` when the browser's copy is current: If-None-Match takes precedence over
    If-Modified-Since (RFC 9110 13.2.2), and a 304 repeats the validators."""
    from email.utils import parsedate_to_datetime

    stat = os.stat(path)
    response = FileResponse(path, media_type=media_type, headers={"Cache-Control": _UI_CACHE_CONTROL}, stat_result=stat)
    etag = response.headers.get("etag", "")
    last_modified = response.headers.get("last-modified", "")
    not_modified = False
    if_none_match = request_headers.get("if-none-match")
    if if_none_match is not None:
        not_modified = bool(etag) and _etag_matches(if_none_match, etag)
    elif (since := request_headers.get("if-modified-since")) and last_modified:
        try:
            not_modified = parsedate_to_datetime(last_modified) <= parsedate_to_datetime(since)
        except (TypeError, ValueError, IndexError):
            not_modified = False
    if not_modified:
        headers = {"Cache-Control": _UI_CACHE_CONTROL, "ETag": etag}
        if last_modified:
            headers["Last-Modified"] = last_modified
        return Response(status_code=304, headers=headers)
    return response


def ui_files(web_dir: Path | None = None) -> dict[str, Path]:
    """URL path -> file for every servable file in ``web_dir`` (hidden files skipped)."""
    root = (web_dir or WEB_DIR).resolve()
    files: dict[str, Path] = {}
    if not root.is_dir():
        return files
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        if path.suffix.lower() not in _UI_MEDIA_TYPES:
            continue
        url = "/" + rel.as_posix()
        if url.startswith(RESERVED_PREFIXES):
            continue
        files[url] = path
    return files


def _route_path(scope: dict[str, Any]) -> str:
    """Request path relative to the app's ``root_path`` (as Starlette's router matches it)."""
    path = str(scope.get("path", ""))
    root = str(scope.get("root_path", "") or "")
    if root and path.startswith(root) and (len(path) == len(root) or path[len(root)] == "/"):
        return path[len(root):] or "/"
    return path


class UIStaticMiddleware:
    """ASGI middleware that answers GET/HEAD for the UI's static files before any other middleware.

    The single-page UI is a public shell: its files hold no data or secrets, and the
    browser cannot attach an API key to a page navigation. Serving them here, outside
    the application's authentication and quota middleware, keeps the page reachable
    in keyed deployments (``API_KEYS``) and uncounted by rate limits, while every
    ``/api`` call the page makes still carries ``X-API-Key`` and is checked as usual.
    Only the exact file paths found by :func:`ui_files` (plus ``/``) are served.
    """

    def __init__(self, app: Any, files: dict[str, Path]) -> None:
        self.app = app
        self.files = {url: (path, _UI_MEDIA_TYPES[path.suffix.lower()]) for url, path in files.items()}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("method") in ("GET", "HEAD"):
            entry = self.files.get(_route_path(scope))
            if entry is not None:
                from starlette.datastructures import Headers

                path, media_type = entry
                response = ui_file_response(path, media_type, Headers(scope=scope))
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def is_ui_path(app: FastAPI, path: str) -> bool:
    """True when ``path`` is one of the UI files :func:`mount_ui` added to ``app``."""
    return path in set(getattr(app.state, "ui_routes", ()) or ())


def mount_ui(app: FastAPI, web_dir: Path | str | None = None, *, public: bool = True) -> list[str]:
    """Serve the single-page UI at ``/`` without shadowing ``/api`` or ``/vo``.

    Each file gets an exact GET/HEAD route (``/``, ``/index.html``, ``/app.js``, ...),
    so no catch-all mount can swallow API routes registered later. With ``public``
    (default) an outermost :class:`UIStaticMiddleware` also serves those exact paths
    ahead of the app's own middleware, so API-key authentication and request quotas
    apply to the API calls the page makes, not to loading the page itself. Call it
    before the app starts (Starlette cannot add middleware afterwards; the routes
    still work then, behind the app's middleware). Returns the URL paths added and
    records them in ``app.state.ui_routes``. Calling it twice is a no-op.
    """
    if getattr(app.state, "ui_mounted", False):
        return []
    root = Path(web_dir) if web_dir else WEB_DIR
    files = ui_files(root)
    if "/index.html" not in files:
        raise FileNotFoundError(f"web UI not found: {root / 'index.html'}")
    added: list[str] = []

    def make_endpoint(path: Path):
        media_type = _UI_MEDIA_TYPES[path.suffix.lower()]

        async def endpoint(request: Request) -> Response:
            return ui_file_response(path, media_type, request.headers)

        return endpoint

    routes = {"/": files["/index.html"], **files}
    for url, path in routes.items():
        app.add_api_route(url, make_endpoint(path), methods=["GET", "HEAD"], include_in_schema=False,
                          name=f"ui:{url}")
        added.append(url)
    app.state.ui_public = False
    if public:
        try:
            app.add_middleware(UIStaticMiddleware, files=routes)
            app.state.ui_public = True
        except RuntimeError as exc:  # the app has already started
            logger.warning("ui_middleware_not_added", error=str(exc))
    app.state.ui_mounted = True
    app.state.ui_dir = root
    app.state.ui_routes = list(added)
    return added


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

#: ``cutout`` exit status when hips2fits' rendering is known to have failed (nothing saved).
EXIT_DEGRADED = 3
_OUT_FORMATS = {".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".fits": "fits", ".fit": "fits", ".fts": "fits"}


def register_cli(subparsers: Any) -> None:
    """Add ``cutout`` to an argparse subparsers object (``main.py`` calls this)."""
    parser = subparsers.add_parser(
        "cutout",
        help="Download a sky cutout (PNG/JPEG/FITS) from a HiPS survey via CDS hips2fits",
        description="Download a sky image cutout via CDS hips2fits. The format is taken from --format "
                    "or the --out extension (.png, .jpg/.jpeg, .fits/.fit/.fts). Exit status: 0 saved, "
                    f"1 upstream/network failure, 2 bad arguments, {EXIT_DEGRADED} rendering degraded "
                    "(hips2fits returned a blank image where the survey has data; not saved unless "
                    "--allow-degraded).",
    )
    parser.add_argument("--ra", type=float, help="ICRS right ascension in degrees [0, 360)")
    parser.add_argument("--dec", type=float, help="ICRS declination in degrees [-90, 90]")
    parser.add_argument("--name", help="Object name resolved with CDS Sesame (instead of --ra/--dec)")
    parser.add_argument("--fov", type=float, default=5.0, help="Field of view (largest side) in arcmin (default 5)")
    parser.add_argument("--survey", default="dss2", help="Survey key or HiPS ID (default dss2; see --list-surveys)")
    parser.add_argument("--out", help="Output file path")
    parser.add_argument("--format", choices=["png", "jpg", "fits"], help="Image format (default: from --out)")
    parser.add_argument("--width", type=int, default=512, help="Width in pixels (default 512)")
    parser.add_argument("--height", type=int, default=512, help="Height in pixels (default 512)")
    parser.add_argument("--projection", default="TAN", help="WCS projection code (default TAN)")
    parser.add_argument("--stretch", choices=sorted(STRETCHES), help="Stretch for png/jpg")
    parser.add_argument("--cmap", help="Matplotlib colormap for single-band png/jpg")
    parser.add_argument("--no-cache", action="store_true",
                        help="Neither read nor write the on-disk cutout cache (always fetch from hips2fits)")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="Save the image even when hips2fits' rendering is known to have failed (exit 0)")
    parser.add_argument("--list-surveys", action="store_true", help="List available surveys and exit")
    parser.set_defaults(handler=cli_cutout)


def _format_from_path(path: str) -> str | None:
    return _OUT_FORMATS.get(Path(path).suffix.lower())


def _angle_text(arcsec: float) -> str:
    """0.352", 3.75', 0.81 deg: three significant figures in the most readable unit."""
    if arcsec < 60.0:
        return f'{arcsec:.3g}"'
    if arcsec < 3600.0:
        return f"{arcsec / 60.0:.3g}'"
    return f"{arcsec / 3600.0:.3g} deg"


def cli_cutout(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch cutout``; returns a process exit code (see :func:`register_cli`)."""
    if args.list_surveys:
        print(f"{'key':<13} {'regime':<9} {'band':<22} {'FITS pixels':<22} HiPS ID")
        for item in list_surveys():
            pixels = "colour composite" if item["color"] else (item["pixel_units"] or "unknown")
            print(f"{item['key']:<13} {item['regime']:<9} {item['wavelength']:<22} {pixels:<22} {item['hips_id']}")
        for key, why in UNAVAILABLE_SURVEYS.items():
            print(f"{key:<13} unavailable: {why}")
        return 0
    if not args.out:
        print("Error: --out is required", file=sys.stderr)
        return 2
    from main import check_search_target  # lazy: main imports this module for its CLI

    try:  # --name or --ra/--dec, never both (the rule of every search command)
        check_search_target(args.name or None, args.ra, args.dec)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    from_path = _format_from_path(args.out)
    if args.format is None and from_path is None:
        suffix = Path(args.out).suffix or "(none)"
        print(f"Error: cannot tell the image format from the --out extension {suffix}; use "
              f"{', '.join(sorted(_OUT_FORMATS))} or give --format", file=sys.stderr)
        return 2
    fmt: str = args.format or from_path or "png"  # from_path is set when --format is absent
    if args.format and from_path not in (None, args.format):
        print(f"Warning: --out extension does not match --format {args.format}", file=sys.stderr)
    from models import ObjectResolutionError

    async def run() -> Cutout:
        async with _new_client(90.0) as client:
            ra, dec = args.ra, args.dec
            if args.name:
                from providers import SesameResolver

                resolved = await SesameResolver(client).resolve(args.name)
                ra, dec = resolved.ra_deg, resolved.dec_deg
                print(f"Resolved '{args.name}' -> RA={ra:.6f} Dec={dec:+.6f} ({resolved.resolver})")
                motion = resolver_motion(resolved)
                if motion is not None:  # centred where the star was when the survey observed it
                    centre = panel_centre(get_survey(args.survey), ra, dec, motion, args.fov)
                    ra, dec = centre.ra, centre.dec
                    if centre.epoch is not None:
                        print(f"Centred on the epoch-{centre.epoch:.1f} position ({centre.offset_arcsec:.1f}\" "
                              f"from the resolver's epoch-{motion.epoch:g} position)")
                    if centre.note:
                        print(f"Note: {centre.note}")
            request = CutoutRequest(ra=ra, dec=dec, fov_arcmin=args.fov, survey=args.survey, width=args.width,
                                    height=args.height, format=fmt, projection=args.projection,
                                    stretch=args.stretch, cmap=args.cmap)
            return await CutoutService(client).cutout(request, use_cache=not args.no_cache)

    try:
        cutout = asyncio.run(run())
    except CutoutValidationError as exc:  # bad arguments (incl. unknown survey): usage error
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ImagingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (httpx.HTTPError, ObjectResolutionError, OSError) as exc:  # resolver / network failures
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if cutout.degraded and not getattr(args, "allow_degraded", False):
        print(f"Error: hips2fits rendering degraded, nothing saved ({cutout.degraded}). Retry later, or pass "
              "--allow-degraded to save the image anyway.", file=sys.stderr)
        return EXIT_DEGRADED
    try:
        path = cutout.save(args.out)
    except OSError as exc:
        print(f"Error: cannot write {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    coverage = "unknown" if cutout.coverage_fraction is None else f"{cutout.coverage_fraction:.1%}"
    print(f"Saved {cutout.survey.label} {cutout.width}x{cutout.height} {cutout.request.format.upper()} "
          f"({len(cutout.content):,} bytes, {_angle_text(cutout.request.pixel_scale_arcsec)}/px at the centre, "
          f"data coverage {coverage}"
          f"{', cached' if cutout.cached else ''}) -> {path}")
    if cutout.degraded:
        print(f"Warning: rendering degraded, image not cached ({cutout.degraded}).", file=sys.stderr)
    elif cutout.blank == "no-data":
        print("Warning: the survey has no data at this position (blank image; its MOC does not reach the field).",
              file=sys.stderr)
    elif cutout.blank:
        print("Warning: blank image; the survey footprint could not be checked, so it may be 'no data here' or a "
              "hips2fits rendering failure (not cached).", file=sys.stderr)
    kind = cutout.pixel_kind
    if kind == "rgb-preview":
        science, science_note = cutout.science_companion() if cutout.survey.color else (None, None)
        hint = (f" Use --survey {science.key} for single-band survey pixel values ({science.pixel_units})."
                if science else "")
        if science_note:
            hint += f" {science_note}"
        print(f"Warning: {cutout.survey.label} is a colour composite; this FITS holds 8-bit RGBA display "
              f"planes, not survey data.{hint}", file=sys.stderr)
    elif kind == "survey":
        units = cutout.survey.pixel_units or "unknown"
        quality = "" if cutout.survey.calibrated else " (not flux-calibrated)"
        print(f"FITS pixels: {units}{quality}. {cutout.survey.pixel_note or ''}".rstrip())
    return 0


__all__ = [
    "EXIT_DEGRADED",
    "GALEX_REGISTRY_BUNITS",
    "HIPS2FITS_CMAPS",
    "STACK_SLOTS",
    "SURVEYS",
    "UNAVAILABLE_SURVEYS",
    "CoverageResult",
    "Cutout",
    "CutoutCache",
    "CutoutRequest",
    "CutoutService",
    "CutoutUpstreamError",
    "CutoutValidationError",
    "HipsSurvey",
    "ImageInfo",
    "ImagingError",
    "PanelCentre",
    "ProperMotion",
    "StackPanel",
    "UIStaticMiddleware",
    "UnknownSurveyError",
    "check_complete",
    "cli_cutout",
    "companion_surveys",
    "coverage_fraction",
    "fetch_cutout",
    "fits_companion",
    "get_survey",
    "hips2fits_cdelt_deg",
    "is_ui_path",
    "is_uniform_raster",
    "list_surveys",
    "mount_ui",
    "name_unknown_to_sesame",
    "panel_centre",
    "probe_image",
    "regime_for_wavelength",
    "register_cli",
    "router",
    "science_survey",
    "ui_files",
]
