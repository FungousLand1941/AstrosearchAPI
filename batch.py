"""High-throughput batch crossmatch: one archive request per catalog (per chunk) instead of one per target.

A list of targets ``[{id, ra, dec, epoch?, pm_ra_masyr?, pm_dec_masyr?, parallax_mas?, radius_arcsec?}]`` is
matched against each requested catalog with the cheapest protocol the archive supports:

``upload``
    IVOA TAP 1.1 table upload (Dowler et al. 2019, "Table Access Protocol Version 1.1", IVOA Recommendation,
    Sect. 2.5.4 UPLOAD; ADQL 2.0, Ortiz et al. 2008, Sect. 2.4 geometric functions). The target list is sent as
    an inline VOTable (``UPLOAD=targets,param:targets``, multipart/form-data) and joined server-side::

        SELECT tu.t_idx, <catalog columns> FROM <catalog table>, TAP_UPLOAD.targets AS tu
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', tu.t_ra, tu.t_dec, tu.t_rad))

    Every target carries its own cone (``t_ra, t_dec, t_rad``) from :func:`models.plan_cone`, so epoch-aware
    cones (proper-motion propagation / epoch widening) are exactly those of the per-target cone search.
    Verified live (2026-09-28) with a column-valued CIRCLE radius on SIMBAD TAP, VizieR TAP, HEASARC Xamin TAP
    and IRSA TAP. Service particulars (all verified against live responses):

    * SIMBAD (https): TAPRegExt capabilities declare ``uploadLimit`` 200000 rows and ``outputLimit`` 50000
      (default) / 2000000 (hard) rows.
    * VizieR TAP: uploads only work over ``http://tapvizier.cds.unistra.fr`` (the https endpoint used for cone
      queries rejects multipart uploads); ``uploadLimit`` 100000 rows.
    * IRSA TAP: only the comma (cross-join + WHERE) form is accepted -- ``JOIN ... ON CONTAINS(...)`` returns
      ``UsageFault: BAD_REQUEST`` -- and TAP/sync has a 5-minute execution limit (capabilities comment), so
      chunks are kept small. Upload columns are plain ASCII names (IRSA rejects unicode columns and reserved
      names such as ``uid``).
    * HEASARC Xamin: result columns of a join are prefixed with the table name (``first_ra``) unless aliased,
      so every selected column is given an explicit ``AS`` alias. The target index is uploaded as a 32-bit
      ``int``: Xamin returns BINARY-serialised VOTables and a 64-bit ``long`` column with its null sentinel
      (-9223372036854775808) overflows astropy's C ``long`` parser on Windows.

``xmatch``
    The CDS XMatch service (Boch, Pineau & Derriere 2012, ASP Conf. Ser. 461, 291;
    http://cdsxmatch.u-strasbg.fr/xmatch/doc/API-calls.html) cross-matches an uploaded CSV with any VizieR
    table (``cat2=vizier:<table>``), returning every pair within ``distMaxArcsec`` (at most 180 arcsec; uploads
    at most 100 MB; ``MAXREC`` hard limit 2000000). Used for Gaia DR3 (``vizier:I/355/gaiadr3``) because the ESA
    Gaia archive does not complete anonymous synchronous uploads. VizieR I/355 positions are the Gaia DR3
    ``ra``/``dec`` at Ep=2016.0 (I/355 ReadMe: "RAdeg ... Right ascension (ICRS) at Ep=2016.0"); its columns
    are renamed to the Gaia archive names used by the registry (``Source`` -> ``source_id``, ``e_RAdeg`` ->
    ``ra_error`` in mas, ...) so rows are normalised exactly like the cone-search rows. Any other VizieR table
    can be requested as a catalog named ``vizier:<table>`` (e.g. ``vizier:II/349/ps1``); the tables in
    :data:`VIZIER_VIEWS` (2MASS PSC, AllWISE, PS1 DR1, SDSS DR16) carry their identifier, epoch and
    positional-error conventions, other tables are mapped from the IVOA UCDs XMatch returns.

    ``distMaxArcsec`` is one value per request, so targets are grouped by cone radius into geometric buckets
    (a new bucket starts when a radius exceeds :data:`XMATCH_RADIUS_BUCKET_RATIO` times the bucket's smallest):
    targets with individual epochs, proper motions or radii still share O(log(r_max/r_min)) requests, and every
    target's rows are then cut back to its own cone.

``cone``
    Archives without upload support (NED, the NASA Exoplanet Archive, MAST Pan-STARRS, SDSS SkyServer) are
    queried with the regular per-target cone search through :class:`crossmatch.QueryExecutor` (including its
    fallbacks) with bounded concurrency; each endpoint is paced by the :class:`providers.EndpointGuard` shared
    with the API's own providers (``app.state.providers``) when the router runs inside the API.

Rows of every strategy are converted with the providers' own row pipeline (``normalize_source_record`` with
the UCD metadata of the response, the catalog's positional-error and epoch specifications, epoch-corrected
separations). Each target's rows of all catalogs are then finalised by
:meth:`crossmatch.CrossmatchService.finalize` -- the single-object pipeline: proper-motion / parallax adoption
from matched rows, final in-radius split, and the NWAY-style Bayesian association (Budavari & Szalay 2008, ApJ
679, 301; Salvato et al. 2018, MNRAS 473, 4937). A batch match's ``confidence`` is therefore the same posterior
probability that the row is the target's counterpart that a single-object search over the same catalogs
reports (it depends on which catalogs are matched together, exactly as there). Matches are ranked nearest
first with SIMBAD host stars ahead of planets listed at the same position (:func:`providers.source_sort_key`).

Large lists are chunked (per-service chunk sizes below), each chunk is retried on transient failures
(connection resets, HTTP 408/429/5xx, honouring Retry-After), a chunk whose answer hit the row or byte limit
(QUERY_STATUS OVERFLOW, ``MAXREC`` rows, :data:`MAX_RESPONSE_BYTES`) or that timed out twice is split in two and
re-sent, and a chunk that still fails for a transient reason falls back to per-target cone searches (a
deterministic HTTP 4xx query error does not: it would fail identically for every target). Each catalog has a
time budget (``BATCH_CATALOG_BUDGET_SECONDS``); targets left when it runs out are reported as failed. Request
counts, retries and wall times are reported per catalog.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import math
import os
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from email.parser import BytesParser
from email.policy import HTTP as HTTP_POLICY
from pathlib import Path
from typing import Any

import httpx

from astrometry import AssociationConfig
from crossmatch import CrossmatchService, QueryExecutor, SearchContext
from models import (
    HEASARC_TAP,
    IRSA_TAP,
    VIZIER_TAP,
    AstroSearchError,
    CatalogDefinition,
    CatalogQueryError,
    CatalogRegistry,
    CatalogUnavailableError,
    ColumnMeta,
    ConePlan,
    InvalidCoordinateError,
    ParsedTable,
    QueryPlan,
    QueryTimeoutError,
    ResponseParseError,
    Settings,
    Target,
    bare_column_name,
    convert_canonical,
    find_column,
    haversine_arcsec,
    normalize_source_record,
    plan_cone,
    row_get,
    ucd_field_map,
)
from providers import _PLANET_TYPES, _TIE_ARCSEC, CacheManager, EndpointGuard, QueryResult, TapProvider, provider_map

logger = logging.getLogger("astrosearch.batch")

__all__ = [
    "BatchCrossmatcher",
    "BatchError",
    "BatchResult",
    "BatchTarget",
    "CatalogRun",
    "UploadService",
    "XMatchView",
    "batch_crossmatch",
    "load_targets_file",
    "parse_targets",
    "read_targets_csv",
    "read_targets_json",
    "register_cli",
    "router",
]

# ---------------------------------------------------------------------------
# Service descriptions
# ---------------------------------------------------------------------------

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
# VizieR TAP accepts multipart uploads over plain http only (the https endpoint rejects them).
VIZIER_TAP_UPLOAD = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
XMATCH_ENDPOINT = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
XMATCH_MAX_DISTANCE_ARCSEC = 180.0  # API-calls doc: "Maximum allowed value is 180"
XMATCH_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # "Total size of uploaded tables can not be larger than 100 MB"
XMATCH_SERVICE_MAX_ROWS = 2_000_000  # MAXREC hard limit of the service (API-calls doc)
# MAXREC we send: an answer this long is treated as truncated and its chunk split. Kept far below the service
# limit so an overflowing answer is cheap to discard (a 100k-row XMatch VOTable is ~33 MB).
XMATCH_MAXREC = 200_000
# Radius buckets: a bucket's XMatch distance is at most this factor above its smallest cone (<= 4x the area).
XMATCH_RADIUS_BUCKET_RATIO = 2.0
# Chunks of wide cones are shrunk so that the expected rows per request stay those of a 10" cone chunk.
XMATCH_REFERENCE_RADIUS_ARCSEC = 10.0

UPLOAD_TABLE = "targets"
UPLOAD_ALIAS = "tu"
UPLOAD_INDEX = "t_idx"
UPLOAD_COLUMNS = ("t_idx", "t_ra", "t_dec", "t_rad")
# XMatch echoes the uploaded columns and adds angDist (arcsec from the uploaded position).
XMATCH_EXTRA_COLUMNS = frozenset({"angdist", "t_idx", "t_ra", "t_dec", "t_rad"})

STRATEGIES = ("upload", "xmatch", "cone")
MAX_RADIUS_ARCSEC = XMATCH_MAX_DISTANCE_ARCSEC


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


# Largest upload/XMatch answer read into memory (bytes); a larger answer is treated like an overflow (split).
MAX_RESPONSE_BYTES = _env_int("BATCH_MAX_RESPONSE_BYTES", 256 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class UploadService:
    """A TAP service that accepts table uploads.

    ``chunk_size`` targets are sent per request (below the service's upload limit and small enough to finish
    inside synchronous execution limits); ``max_rows`` is sent as MAXREC -- an answer with that many rows (or
    QUERY_STATUS=OVERFLOW) is treated as truncated and its chunk is split.
    """

    key: str
    upload_endpoint: str
    chunk_size: int
    max_rows: int
    alias_columns: bool = False
    upload_row_limit: int | None = None
    note: str = ""


UPLOAD_SERVICES: dict[str, UploadService] = {
    SIMBAD_TAP: UploadService("simbad", SIMBAD_TAP, 5000, 200_000, upload_row_limit=200_000,
                              note="uploadLimit 200000 rows, outputLimit hard 2000000 (TAPRegExt capabilities)"),
    VIZIER_TAP: UploadService("vizier", VIZIER_TAP_UPLOAD, 5000, 200_000, upload_row_limit=100_000,
                              note="uploads over http:// only; uploadLimit 100000 rows"),
    IRSA_TAP: UploadService("irsa", IRSA_TAP, 2000, 200_000,
                            note="comma-join form only; TAP/sync 5-minute execution limit"),
    HEASARC_TAP: UploadService("heasarc", HEASARC_TAP, 2000, 200_000, alias_columns=True,
                               note="join columns are table-prefixed unless aliased"),
}


@dataclass(frozen=True, slots=True)
class XMatchView:
    """How a catalog is served by CDS XMatch: VizieR table, ``cols2`` selection and its row conventions.

    ``base_catalog`` names a registry catalog whose definition normalises the rows (after ``renames`` to its
    column names); without it the rows are described by ``epoch``/``epoch_format``, ``pos_error`` (same
    specification as :class:`models.CatalogDefinition`), ``parameters`` (``id_field``, ``epoch_range``, ...),
    ``wavelength`` and ``citation``. An empty ``columns`` sends no ``cols2`` (XMatch's default column set).
    """

    vizier_table: str
    columns: tuple[str, ...] = ()
    renames: Mapping[str, str] = field(default_factory=dict)
    epoch: float | str | None = None
    chunk_size: int = 20_000
    note: str = ""
    base_catalog: str | None = None
    epoch_format: str | None = None
    pos_error: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    wavelength: str = "unknown"
    citation: str | None = None


_GAIA_VIEW = XMatchView(
    # Gaia DR3 main source table in VizieR (I/355/gaiadr3). Names verified against
    # tables?action=getColList&tabName=vizier:I/355/gaiadr3 and the I/355 ReadMe.
    "vizier:I/355/gaiadr3",
    ("Source", "RAdeg", "DEdeg", "e_RAdeg", "e_DEdeg", "RADEcor", "Plx", "e_Plx", "pmRA", "pmDE",
     "Gmag", "BPmag", "RPmag", "RUWE", "Solved"),
    {
        "Source": "source_id", "RAdeg": "ra", "DEdeg": "dec", "e_RAdeg": "ra_error", "e_DEdeg": "dec_error",
        "RADEcor": "ra_dec_corr", "Plx": "parallax", "e_Plx": "parallax_error", "pmRA": "pmra",
        "pmDE": "pmdec", "Gmag": "phot_g_mean_mag", "BPmag": "phot_bp_mean_mag", "RPmag": "phot_rp_mean_mag",
        "RUWE": "ruwe", "Solved": "astrometric_params_solved",
    },
    2016.0,
    note="VizieR I/355/gaiadr3 (Gaia DR3 at Ep=2016.0); ESA archive uploads hang anonymously",
    base_catalog="gaia_dr3",
)

XMATCH_VIEWS: dict[str, XMatchView] = {"gaia_dr3": _GAIA_VIEW}

# XMatch standardises every VizieR table's positional error into a 1-sigma error ellipse
# (UCDs phys.angSize.smajAxis/sminAxis;pos.errorEllipse;meta.main). Verified live (2026-09-28) against the
# catalogues' own 1-sigma values: 2MASS errHalfMaj/errHalfMin = IRSA err_maj/err_min (1-sigma, 2MASS Explanatory
# Supplement IV.4); AllWISE eeMaj/eeMin = the ellipse of IRSA sigra/sigdec/sigradec (1-sigma, AllWISE Explanatory
# Supplement II.1); PS1 errHalfMaj/errHalfMin = max/min of II/349 e_RAJ2000/e_DEJ2000 (1-sigma mean-position errors).
VIZIER_VIEWS: dict[str, XMatchView] = {
    "vizier:I/355/gaiadr3": _GAIA_VIEW,
    "vizier:II/246/out": XMatchView(
        "vizier:II/246/out",
        epoch="MeasureJD", epoch_format="jd",  # "Julian date of the source measurement" (time.epoch)
        pos_error={"columns": ["errHalfMaj", "errHalfMin", "errPosAng"], "units": "arcsec", "kind": "ellipse"},
        parameters={"id_field": "2MASS", "epoch_range": [1997.4, 2001.2], "single_epoch_positions": True},
        wavelength="infrared",
        citation="Skrutskie et al. 2006, AJ 131, 1163; VizieR II/246 (Cutri et al. 2003, 2003yCat.2246....0C)",
        note="2MASS PSC; each position is one observation at MeasureJD",
    ),
    "vizier:II/328/allwise": XMatchView(
        "vizier:II/328/allwise",
        # No proper motions (AllWISE pmRA/pmDE are noisy motion-fit values the registry does not use either).
        ("AllWISE", "RAJ2000", "DEJ2000", "eeMaj", "eeMin", "eePA", "W1mag", "W2mag", "W3mag", "W4mag",
         "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag", "Jmag", "Hmag", "Kmag", "ccf", "ex", "var", "qph", "ID"),
        pos_error={"columns": ["eeMaj", "eeMin", "eePA"], "units": "arcsec", "kind": "ellipse"},
        # VizieR II/328 has no per-source mean epoch (IRSA's w1mjdmean): positions are means over the
        # WISE cryogenic + NEOWISE post-cryo mission (Jan 2010 - Feb 2011), reported as an epoch range.
        parameters={"id_field": "AllWISE", "epoch_range": [2010.0, 2011.2]},
        wavelength="infrared",
        citation="Wright et al. 2010, AJ 140, 1868; VizieR II/328 (Cutri et al. 2013, 2013yCat.2328....0C)",
        note="AllWISE designation as source_id; epoch range 2010.0-2011.2 (no per-source epoch in VizieR)",
    ),
    "vizier:II/349/ps1": XMatchView(
        "vizier:II/349/ps1",
        epoch="Epoch", epoch_format="mjd",  # "Mean epoch (MJD)" (time.epoch, unit d)
        # 15 mas systematic floor as for the registry's panstarrs_dr2 (underestimated errors of bright sources).
        pos_error={"columns": ["errHalfMaj", "errHalfMin", "errPosAng"], "units": "arcsec", "kind": "ellipse",
                   "systematic_arcsec": 0.015},
        # Nd <= 1: single-detection artefacts, dropped as the registry's panstarrs_dr2 (nDetections > 1).
        parameters={"id_field": "objID", "epoch_range": [2009.5, 2015.0], "exclude_values": {"Nd": [0, 1]}},
        wavelength="optical",
        citation="Chambers et al. 2016, arXiv:1612.05560; VizieR II/349 (Chambers et al. 2017, 2017yCat.2349....0C)",
        note="Pan-STARRS1 DR1 mean objects",
    ),
    "vizier:V/154/sdss16": XMatchView(
        "vizier:V/154/sdss16",
        ("objID", "RA_ICRS", "DE_ICRS", "mode", "class", "clean", "e_RA_ICRS", "e_DE_ICRS", "umag", "gmag",
         "rmag", "imag", "zmag", "e_umag", "e_gmag", "e_rmag", "e_imag", "e_zmag", "zsp", "e_zsp", "f_zsp",
         "spCl", "subCl", "zph", "e_zph", "Q", "SDSS16", "MJD"),
        epoch="MJD", epoch_format="mjd",  # imaging MJD (time.epoch;obs)
        # e_RA_ICRS/e_DE_ICRS are SkyServer raErr/decErr (1-sigma); 40 mas systematic as the registry's sdss.
        pos_error={"columns": ["e_RA_ICRS", "e_DE_ICRS"], "units": "arcsec", "kind": "sigma", "systematic_arcsec": 0.04},
        # mode 1 = PRIMARY (the registry's sdss uses PhotoPrimary); 2 secondary, 3 family, 4 outside.
        parameters={"id_field": "objID", "epoch_range": [1998.5, 2009.6], "single_epoch_positions": True,
                    "exclude_values": {"mode": [2, 3, 4]}},
        wavelength="optical",
        citation="Ahumada et al. 2020, ApJS 249, 3 (SDSS DR16); VizieR V/154",
        note="SDSS DR16 primary photometric objects (mode 1)",
    ),
}

_VIZIER_ACK = "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France (DOI 10.26093/cds/vizier)."
_XMATCH_ACK = "This research has made use of the cross-match service provided by CDS, Strasbourg."


class BatchError(AstroSearchError, ValueError):
    """Invalid batch request (targets, catalogs, radius or strategy)."""


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchTarget:
    """One validated input target; ``radius_arcsec`` overrides the batch radius when set."""

    id: str
    target: Target
    radius_arcsec: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ra": self.target.ra, "dec": self.target.dec, "epoch": self.target.epoch,
                "pm_ra_masyr": self.target.pm_ra_masyr, "pm_dec_masyr": self.target.pm_dec_masyr,
                "parallax_mas": self.target.parallax_mas, "radius_arcsec": self.radius_arcsec}


_TARGET_KEYS: dict[str, tuple[str, ...]] = {
    "id": ("id", "target_id", "name", "objid", "obj_id", "source", "designation"),
    "ra": ("ra", "ra_deg", "radeg", "raj2000", "ra_icrs"),
    "dec": ("dec", "dec_deg", "decdeg", "dej2000", "decj2000", "de_icrs", "dec_icrs"),
    "epoch": ("epoch", "ref_epoch", "epoch_jyear"),
    "pm_ra_masyr": ("pm_ra_masyr", "pmra", "pm_ra"),
    "pm_dec_masyr": ("pm_dec_masyr", "pmdec", "pm_dec", "pmde"),
    "parallax_mas": ("parallax_mas", "parallax", "plx"),
    "radius_arcsec": ("radius_arcsec", "radius"),
}


def _pick(lowered: Mapping[str, Any], canonical: str) -> Any:
    for key in _TARGET_KEYS[canonical]:
        value = lowered.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            return value
    return None


def _optional_float(value: Any, name: str, label: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BatchError(f"{label}: {name} must be numeric (got {value!r}).") from exc
    if not math.isfinite(number):
        raise BatchError(f"{label}: {name} must be finite.")
    return number


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise InvalidCoordinateError(f"{name} must be finite.")
    return number


def icrs_target(
    ra: Any, dec: Any, *, epoch: float | None = None, pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None, parallax_mas: float | None = None,
) -> Target:
    """:func:`models.validate_target` for ICRS input, without building an astropy SkyCoord per target.

    The checks and messages are those of ``validate_target`` (RA normalised to [0, 360), Dec in [-90, 90],
    epoch a Julian year in [1800, 2200], proper motion given in both components and below 20"/yr, parallax
    in [0, 1000) mas). ``validate_target`` additionally constructs a SkyCoord only to validate the frame name,
    which is always ICRS here; that construction costs ~0.5 ms, i.e. ~60 s for 100k targets.
    """
    ra_val = _finite(ra, "ra") % 360.0
    if ra_val >= 360.0:  # float modulo of tiny negatives (e.g. -1e-14 % 360 == 360.0)
        ra_val = 0.0
    dec_val = _finite(dec, "dec")
    if not -90.0 <= dec_val <= 90.0:
        raise InvalidCoordinateError("DEC must be within [-90, 90] degrees.")
    if epoch is not None:
        try:
            epoch = float(epoch)
        except (TypeError, ValueError) as exc:
            raise InvalidCoordinateError("epoch must be numeric.") from exc
        if not math.isfinite(epoch) or epoch < 1800 or epoch > 2200:
            raise InvalidCoordinateError("epoch must be a finite Julian year between 1800 and 2200.")
    if (pm_ra_masyr is None) != (pm_dec_masyr is None):
        raise InvalidCoordinateError("pm_ra_masyr and pm_dec_masyr must be given together.")
    if pm_ra_masyr is not None:
        pm_ra_masyr = _finite(pm_ra_masyr, "pm_ra_masyr")
        pm_dec_masyr = _finite(pm_dec_masyr, "pm_dec_masyr")
        if math.hypot(pm_ra_masyr, pm_dec_masyr) > 20_000.0:
            raise InvalidCoordinateError("proper motion exceeds 20 arcsec/yr; check units (mas/yr expected).")
    if parallax_mas is not None:
        parallax_mas = _finite(parallax_mas, "parallax_mas")
        if not 0.0 <= parallax_mas < 1000.0:
            raise InvalidCoordinateError("parallax_mas must be in [0, 1000) mas (Proxima Cen, the nearest star, has 768 mas).")
    return Target(ra=ra_val, dec=dec_val, frame="icrs", epoch=epoch, pm_ra_masyr=pm_ra_masyr,
                  pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas)


def parse_targets(items: Iterable[Mapping[str, Any]], *, max_targets: int | None = None) -> list[BatchTarget]:
    """Validate target mappings; RA is normalised to [0, 360) and ids must be unique.

    Keys are case-insensitive with common aliases (``ra``/``RAJ2000``, ``pmra``, ``name``, ``radius``, ...).
    A missing id becomes the 1-based row number. Proper motion (mas/yr, RA component including cos dec) needs
    both components.
    """
    limit = max_targets if max_targets is not None else _env_int("BATCH_MAX_TARGETS", 100_000)
    targets: list[BatchTarget] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise BatchError(f"target {index}: expected an object with ra and dec, got {type(item).__name__}.")
        if len(targets) >= limit:
            raise BatchError(f"too many targets: at most {limit} per batch (BATCH_MAX_TARGETS).")
        lowered = {str(k).strip().lower().lstrip("﻿"): v for k, v in item.items()}
        raw_id = _pick(lowered, "id")
        target_id = " ".join(str(raw_id).split()) if raw_id is not None else str(index)
        label = f"target {target_id!r}"
        if target_id in seen:
            raise BatchError(f"{label}: duplicate id.")
        seen.add(target_id)
        ra, dec = _pick(lowered, "ra"), _pick(lowered, "dec")
        if ra is None or dec is None:
            raise BatchError(f"{label}: ra and dec (ICRS degrees) are required.")
        try:
            target = icrs_target(
                ra, dec,
                epoch=_optional_float(_pick(lowered, "epoch"), "epoch", label),
                pm_ra_masyr=_optional_float(_pick(lowered, "pm_ra_masyr"), "pm_ra_masyr", label),
                pm_dec_masyr=_optional_float(_pick(lowered, "pm_dec_masyr"), "pm_dec_masyr", label),
                parallax_mas=_optional_float(_pick(lowered, "parallax_mas"), "parallax_mas", label),
            )
        except InvalidCoordinateError as exc:
            raise BatchError(f"{label}: {exc}") from exc
        radius = _optional_float(_pick(lowered, "radius_arcsec"), "radius_arcsec", label)
        if radius is not None and not 0.0 < radius <= MAX_RADIUS_ARCSEC:
            raise BatchError(f"{label}: radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}].")
        targets.append(BatchTarget(target_id, target, radius))
    if not targets:
        raise BatchError("no targets given.")
    return targets


def read_targets_csv(text: str | bytes, *, max_targets: int | None = None) -> list[BatchTarget]:
    """Targets from CSV text with a header row (``id,ra,dec[,epoch,pmra,pmdec,parallax,radius_arcsec]``).

    Lines starting with ``#`` are comments.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", "replace")
    lines = [line for line in text.lstrip("﻿").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise BatchError("CSV contains no header row.")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    fields = {str(f).strip().lower() for f in reader.fieldnames or []}
    if not fields & set(_TARGET_KEYS["ra"]) or not fields & set(_TARGET_KEYS["dec"]):
        raise BatchError(f"CSV header must contain ra and dec columns (got {sorted(fields)}).")
    rows = [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None} for row in reader]
    return parse_targets(rows, max_targets=max_targets)


def read_targets_json(payload: str | bytes | Any, *, max_targets: int | None = None) -> list[BatchTarget]:
    """Targets from a JSON array of objects or ``{"targets": [...]}``."""
    try:
        data = json.loads(payload) if isinstance(payload, (str, bytes, bytearray)) else payload
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BatchError(f"invalid JSON targets: {exc}") from exc
    if isinstance(data, Mapping):
        data = data.get("targets")
    if not isinstance(data, list):
        raise BatchError("JSON targets must be an array of {id, ra, dec} objects (or {\"targets\": [...]}).")
    return parse_targets(data, max_targets=max_targets)


def load_targets_file(path: str | os.PathLike[str], *, max_targets: int | None = None) -> list[BatchTarget]:
    """Targets from a ``.csv`` or ``.json`` file (other extensions: sniffed from the first character)."""
    file = Path(path)
    content = file.read_bytes()
    suffix = file.suffix.lower()
    if suffix == ".json" or (suffix != ".csv" and content.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] in (b"[", b"{")):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BatchError(f"{file}: JSON targets must be UTF-8 encoded ({exc}).") from exc
        try:
            return read_targets_json(text, max_targets=max_targets)
        except BatchError as exc:
            raise BatchError(f"{file}: {exc}") from exc
    return read_targets_csv(content, max_targets=max_targets)


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CatalogRun:
    """Per-catalog execution report."""

    catalog: str
    strategy: str
    endpoint: str | None
    requests: int = 0
    retries: int = 0
    chunks: int = 0
    split_chunks: int = 0
    rows_returned: int = 0
    targets: int = 0
    matched_targets: int = 0
    total_matches: int = 0
    fallback_targets: int = 0
    failed_targets: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    citation: str | None = None
    acknowledgement: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog": self.catalog, "strategy": self.strategy, "endpoint": self.endpoint,
            "requests": self.requests, "retries": self.retries, "chunks": self.chunks,
            "split_chunks": self.split_chunks, "rows_returned": self.rows_returned, "targets": self.targets,
            "matched_targets": self.matched_targets, "total_matches": self.total_matches,
            "fallback_targets": self.fallback_targets, "failed_targets": self.failed_targets,
            "elapsed_s": round(self.elapsed_s, 3), "errors": list(self.errors),
            "warnings": list(dict.fromkeys(self.warnings))[:50], "queries": self.queries[:3],
            "citation": self.citation, "acknowledgement": self.acknowledgement,
        }


class _CountingClient:
    """Wraps an ``httpx.AsyncClient``: every request (each retry included) is counted for a catalog.

    Providers only call ``client.request``; upload/XMatch requests use ``stream`` (size-capped reads). The
    shared client is never mutated.
    """

    def __init__(self, client: httpx.AsyncClient, on_request: Callable[[], None]) -> None:
        self._client = client
        self._on_request = on_request

    async def request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        self._on_request()
        return await self._client.request(method, url, **kwargs)

    def stream(self, method: str, url: Any, **kwargs: Any) -> Any:
        self._on_request()
        return self._client.stream(method, url, **kwargs)

    async def get(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)


class _RowConverter(TapProvider):
    """The providers' row pipeline (normalisation, positional errors, epochs, separations) for batch rows."""

    def __init__(self, client: Any, provider_name: str) -> None:
        super().__init__(client, cache=CacheManager(None))
        self.provider_name = provider_name


class _Overflow(Exception):
    """An upload/XMatch answer hit the row or byte limit: the chunk must be split."""


class _TooSlow(Exception):
    """An upload/XMatch request timed out repeatedly: the chunk must be split."""


class _ResponseTooLarge(Exception):
    """The answer exceeded MAX_RESPONSE_BYTES (not read further)."""


_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_TIMEOUT_EXCEPTIONS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout, httpx.PoolTimeout)
_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError)
# Chunk failures after which the targets are cone-searched instead (transient or availability problems);
# CatalogQueryError (a deterministic HTTP 4xx / query error) would fail identically for every target.
_FALLBACK_ERRORS = (CatalogUnavailableError, QueryTimeoutError, ResponseParseError)
# Response headers that describe the wire encoding: dropped when a decoded body is re-wrapped.
_WIRE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

FLAT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("target_id", "string"), ("target_ra", "float64"), ("target_dec", "float64"), ("target_epoch", "float64"),
    ("catalog", "string"), ("strategy", "string"), ("rank", "int32"), ("source_id", "string"),
    ("ra", "float64"), ("dec", "float64"), ("separation_arcsec", "float64"),
    ("query_separation_arcsec", "float64"), ("epoch_propagation", "string"),
    ("positional_error_arcsec", "float64"), ("epoch", "float64"), ("epoch_range_start", "float64"),
    ("epoch_range_end", "float64"), ("pm_ra_masyr", "float64"), ("pm_dec_masyr", "float64"),
    ("confidence", "float64"), ("data_json", "string"),
)


@dataclass(slots=True)
class BatchResult:
    """Per-target matches for every catalog plus the execution report.

    ``association[idx]`` holds the per-target association summary of
    :meth:`crossmatch.CrossmatchService.finalize` (``p_any``, the proper motion / parallax used and where they
    came from, adoption warnings) for targets with at least one catalog row.
    """

    targets: list[BatchTarget]
    catalogs: list[str]
    radius_arcsec: float
    runs: dict[str, CatalogRun]
    matches: dict[int, dict[str, list[dict[str, Any]]]]
    failures: dict[int, dict[str, str]]
    wall_time_s: float
    nearest_only: bool = False
    association: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def request_count(self) -> int:
        return sum(run.requests for run in self.runs.values())

    def summary(self) -> dict[str, Any]:
        return {
            "targets": len(self.targets),
            "catalogs": list(self.catalogs),
            "radius_arcsec": self.radius_arcsec,
            "nearest_only": self.nearest_only,
            "request_count": self.request_count,
            "wall_time_s": round(self.wall_time_s, 3),
            "strategies": {name: run.strategy for name, run in self.runs.items()},
            "matched_targets": {name: run.matched_targets for name, run in self.runs.items()},
            "total_matches": sum(run.total_matches for run in self.runs.values()),
            "confidence": "posterior probability that the row is the target's counterpart (as a single-object "
                          "crossmatch over the same catalogs)",
        }

    def _index(self, target_id: str) -> int:
        for idx, item in enumerate(self.targets):
            if item.id == target_id:
                return idx
        raise KeyError(target_id)

    def target_matches(self, target_id: str) -> dict[str, list[dict[str, Any]]]:
        """Matches of one target by catalog (ranked nearest first)."""
        return self.matches.get(self._index(target_id), {})

    def target_association(self, target_id: str) -> dict[str, Any]:
        """Association summary of one target (empty when no catalog returned a row)."""
        return self.association.get(self._index(target_id), {})

    def as_dict(self, *, include_data: bool = True) -> dict[str, Any]:
        targets = []
        for idx, item in enumerate(self.targets):
            per_catalog = self.matches.get(idx, {})
            targets.append({
                **item.as_dict(),
                "matches": {name: [_strip_data(m, include_data) for m in per_catalog.get(name, [])] for name in self.catalogs},
                "failures": dict(self.failures.get(idx, {})),
                "association": dict(self.association.get(idx, {})),
            })
        return {"summary": self.summary(), "catalogs": {n: r.as_dict() for n, r in self.runs.items()}, "targets": targets}

    def rows(self, *, include_data: bool = True) -> list[dict[str, Any]]:
        """One flat row per (target, catalog, match): a dataset-like table."""
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(self.targets):
            for name in self.catalogs:
                for match in self.matches.get(idx, {}).get(name, []):
                    span = match.get("epoch_range") or (None, None)
                    out.append({
                        "target_id": item.id, "target_ra": item.target.ra, "target_dec": item.target.dec,
                        "target_epoch": item.target.epoch, "catalog": name, "strategy": match["strategy"],
                        "rank": match["rank"], "source_id": match["source_id"], "ra": match["ra"], "dec": match["dec"],
                        "separation_arcsec": match["separation_arcsec"],
                        "query_separation_arcsec": match["query_separation_arcsec"],
                        "epoch_propagation": match["epoch_propagation"],
                        "positional_error_arcsec": match["positional_error_arcsec"], "epoch": match["epoch"],
                        "epoch_range_start": span[0], "epoch_range_end": span[1],
                        "pm_ra_masyr": match["pm_ra_masyr"], "pm_dec_masyr": match["pm_dec_masyr"],
                        "confidence": match["confidence"],
                        "data_json": json.dumps(match.get("data"), default=str) if include_data else None,
                    })
        return out

    def to_arrow(self, *, include_data: bool = True):
        """The flat rows as a ``pyarrow.Table``; the summary and per-catalog report are in the schema metadata."""
        import pyarrow as pa

        schema = pa.schema([pa.field(name, getattr(pa, kind)()) for name, kind in FLAT_COLUMNS])
        table = pa.Table.from_pylist(self.rows(include_data=include_data), schema=schema)
        meta = {
            b"astrosearch.batch.summary": json.dumps(self.summary()).encode(),
            b"astrosearch.batch.catalogs": json.dumps({n: r.as_dict() for n, r in self.runs.items()}, default=str).encode(),
        }
        return table.replace_schema_metadata(meta)

    def to_parquet_bytes(self, *, include_data: bool = True) -> bytes:
        import pyarrow.parquet as pq

        sink = io.BytesIO()
        pq.write_table(self.to_arrow(include_data=include_data), sink)
        return sink.getvalue()

    def to_csv_text(self, *, include_data: bool = True) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=[name for name, _ in FLAT_COLUMNS], lineterminator="\n")
        writer.writeheader()
        for row in self.rows(include_data=include_data):
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
        return buffer.getvalue()

    def write(self, path: str | os.PathLike[str], fmt: str | None = None, *, include_data: bool = True) -> Path:
        """Write ``parquet`` (flat rows), ``csv`` (flat rows) or ``json`` (full per-target result)."""
        out = Path(path)
        kind = (fmt or out.suffix.lstrip(".") or "parquet").lower()
        if kind not in {"parquet", "csv", "json"}:
            raise BatchError(f"unsupported output format {kind!r} (parquet, csv or json).")
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "parquet":
            out.write_bytes(self.to_parquet_bytes(include_data=include_data))
        elif kind == "csv":
            out.write_text(self.to_csv_text(include_data=include_data), encoding="utf-8")
        else:
            out.write_text(json.dumps(self.as_dict(include_data=include_data), default=str), encoding="utf-8")
        return out


def _strip_data(match: dict[str, Any], include_data: bool) -> dict[str, Any]:
    return match if include_data else {k: v for k, v in match.items() if k != "data"}


# ---------------------------------------------------------------------------
# Upload payloads & queries
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    return repr(float(value))


def upload_votable(rows: Sequence[tuple[int, float, float, float]]) -> bytes:
    """Deterministic VOTable 1.3 TABLEDATA upload with ASCII columns t_idx (int32), t_ra, t_dec, t_rad (deg)."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
        f'<RESOURCE type="results"><TABLE name="{UPLOAD_TABLE}">\n'
        '<FIELD name="t_idx" datatype="int"/>\n'
        '<FIELD name="t_ra" datatype="double" unit="deg"/>\n'
        '<FIELD name="t_dec" datatype="double" unit="deg"/>\n'
        '<FIELD name="t_rad" datatype="double" unit="deg"/>\n'
        "<DATA><TABLEDATA>\n"
    ]
    for idx, ra, dec, rad in rows:
        parts.append(f"<TR><TD>{int(idx)}</TD><TD>{_num(ra)}</TD><TD>{_num(dec)}</TD><TD>{_num(rad)}</TD></TR>\n")
    parts.append("</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>\n")
    return "".join(parts).encode("utf-8")


def upload_csv(rows: Sequence[tuple[int, float, float]]) -> bytes:
    """Deterministic CSV upload (t_idx,t_ra,t_dec) for CDS XMatch."""
    lines = ["t_idx,t_ra,t_dec"] + [f"{int(i)},{_num(ra)},{_num(dec)}" for i, ra, dec in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


def _column_alias(expr: str) -> str:
    """Alias for a selected column: quoted when the column itself is a delimited identifier (``"time"``,
    an SQL reserved word) or not a plain ASCII identifier."""
    name = bare_column_name(expr)
    if expr.strip().startswith('"') or not (name.isidentifier() and name.isascii()):
        return '"' + name.replace('"', '""') + '"'
    return name


def build_upload_adql(catalog: CatalogDefinition, service: UploadService) -> str:
    """ADQL joining the uploaded targets with ``catalog`` (same columns as the per-target cone query).

    Every target row carries its own cone (``tu.t_ra, tu.t_dec, tu.t_rad`` in degrees).
    """
    params = catalog.parameters
    raw_columns = params.get("columns") or ["source_id", "ra", "dec"]
    columns = [c.strip() for c in raw_columns.split(",")] if isinstance(raw_columns, str) else [str(c) for c in raw_columns]
    ra_expr = str(params.get("ra_field", "ra"))
    dec_expr = str(params.get("dec_field", "dec"))
    id_expr = params.get("id_field", "source_id")
    selected = {bare_column_name(c).lower() for c in columns}
    for expr in (ra_expr, dec_expr, id_expr):
        if expr and bare_column_name(str(expr)).lower() not in selected:
            columns.append(str(expr))
            selected.add(bare_column_name(str(expr)).lower())
    if service.alias_columns:
        columns = [c if " as " in c.lower() else f"{c} AS {_column_alias(c)}" for c in columns]
        index = f"{UPLOAD_ALIAS}.{UPLOAD_INDEX} AS {UPLOAD_INDEX}"
    else:
        index = f"{UPLOAD_ALIAS}.{UPLOAD_INDEX}"
    point = f"POINT('ICRS', {ra_expr}, {dec_expr})"
    circle = f"CIRCLE('ICRS', {UPLOAD_ALIAS}.t_ra, {UPLOAD_ALIAS}.t_dec, {UPLOAD_ALIAS}.t_rad)"
    where = f"1 = CONTAINS({point}, {circle})"
    if params.get("where"):
        where += f" AND ({params['where']})"
    table = catalog.table or "gaiadr3.gaia_source"
    return f"SELECT {index}, {', '.join(columns)} FROM {table}, TAP_UPLOAD.{UPLOAD_TABLE} AS {UPLOAD_ALIAS} WHERE {where}"


def _xmatch_view(name: str) -> XMatchView | None:
    return XMATCH_VIEWS.get(name) or VIZIER_VIEWS.get(name)


def xmatch_catalog_definition(name: str, registry: CatalogRegistry) -> tuple[CatalogDefinition, XMatchView | None]:
    """CatalogDefinition used to normalise XMatch rows of ``name`` (a registry catalog or ``vizier:<table>``).

    ``vizier:I/355/gaiadr3`` is the table behind the ``gaia_dr3`` view and is normalised identically (epoch
    2016.0, Gaia archive column names); the other :data:`VIZIER_VIEWS` carry their own conventions; any other
    ``vizier:<table>`` is described by the UCDs of the XMatch answer.
    """
    view = _xmatch_view(name)
    if view is not None and view.base_catalog:
        base = registry.get(view.base_catalog)
        definition = replace(
            base,
            name=name,
            endpoint=XMATCH_ENDPOINT,
            table=view.vizier_table,
            epoch=view.epoch,
            epoch_format=None,
            citation=(base.citation or "") + "; VizieR " + view.vizier_table.split(":", 1)[1] + " via CDS XMatch",
            acknowledgement=(base.acknowledgement or "") + " " + _VIZIER_ACK + " " + _XMATCH_ACK,
        )
        return definition, view
    if name.startswith("vizier:") and len(name) > len("vizier:"):
        table = name.split(":", 1)[1]
        parameters: dict[str, Any] = dict(view.parameters) if view else {}
        if view and view.columns:
            parameters.setdefault("columns", list(view.columns))
        definition = CatalogDefinition(
            name=name, provider="xmatch", wavelength=view.wavelength if view else "unknown", endpoint=XMATCH_ENDPOINT,
            table=name, description=f"VizieR table {table} via CDS XMatch", query_method="CDS XMatch",
            parameters=parameters, epoch=view.epoch if view else None, epoch_format=view.epoch_format if view else None,
            pos_error=dict(view.pos_error) if view else {},
            citation=(view.citation if view and view.citation else
                      f"VizieR {table} (see https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={table})"),
            acknowledgement=_VIZIER_ACK + " " + _XMATCH_ACK, max_rows=200,
        )
        return definition, view
    raise BatchError(f"{name!r} has no CDS XMatch view (use a registry catalog listed in XMATCH_VIEWS or 'vizier:<table>').")


def _with_ellipse_errors(catalog: CatalogDefinition, columns: Sequence[ColumnMeta]) -> CatalogDefinition:
    """A catalog without a positional-error specification gets XMatch's standardised 1-sigma error ellipse.

    CDS XMatch returns, for every VizieR table with positional errors, the half-axes and position angle of a
    1-sigma error ellipse (UCDs ``phys.angSize.smajAxis;pos.errorEllipse;meta.main``,
    ``phys.angSize.sminAxis;pos.errorEllipse;meta.main``, ``pos.posAng;pos.errorEllipse;meta.main``); see
    :data:`VIZIER_VIEWS` for the live verification of the 1-sigma convention.
    """
    if catalog.pos_error:
        return catalog

    def find(token: str) -> str | None:
        for col in columns:
            ucd = (col.ucd or "").lower()
            if "pos.errorellipse" in ucd and token in ucd:
                return col.name
        return None

    major, minor, angle = find("smajaxis"), find("sminaxis"), find("pos.posang")
    if major is None:
        return catalog
    cols = [major, minor or major] + ([angle] if angle else [])
    return replace(catalog, pos_error={"columns": cols, "kind": "ellipse"})


# ---------------------------------------------------------------------------
# Batch engine
# ---------------------------------------------------------------------------


def _match_sort_key(match: Mapping[str, Any]) -> tuple[float, int, str]:
    """:func:`providers.source_sort_key` for serialised matches: nearest first, coincident rows (within 0.1 mas)
    with non-planets first, then by id -- SIMBAD lists a host star and its planets at coordinates that differ by
    ~1e-17 deg, so the exact separation alone would make the nearest identity depend on float noise."""
    otype = str((match.get("physical") or {}).get("object_type") or "").strip().lower()
    sep = float(match.get("separation_arcsec") or 0.0)
    return (round(sep / _TIE_ARCSEC) * _TIE_ARCSEC, 1 if otype in _PLANET_TYPES else 0, str(match.get("source_id")))


def radius_buckets(indices: Sequence[int], radii: Mapping[int, float], ratio: float = XMATCH_RADIUS_BUCKET_RATIO) -> list[list[int]]:
    """Group target indices by cone radius: a bucket holds radii within ``ratio`` x its smallest radius."""
    buckets: list[list[int]] = []
    low = 0.0
    for i in sorted(indices, key=lambda k: (radii[k], k)):
        if not buckets or radii[i] > ratio * low:
            buckets.append([])
            low = radii[i]
        buckets[-1].append(i)
    return buckets


def _position_reader(catalog: CatalogDefinition, columns: Sequence[ColumnMeta]) -> Callable[[Mapping[str, Any]], tuple[float, float] | None]:
    """(ra, dec) of a row with the precedence of :func:`models.normalize_source_record` (catalog field map, then
    UCD metadata), resolving the columns once per table instead of once per row."""
    field_map = TapProvider._field_map(catalog)
    ucd = ucd_field_map(columns)
    sources: dict[str, list[tuple[str, Any]]] = {}
    for canonical in ("ra", "dec"):
        options: list[tuple[str, Any]] = []
        if field_map.get(canonical):
            name = field_map[canonical]
            col = find_column(columns, name)
            options.append((name, col.unit if col else None))
        if canonical in ucd:
            options.append((ucd[canonical].name, ucd[canonical].unit))
        sources[canonical] = options

    def read(row: Mapping[str, Any]) -> tuple[float, float] | None:
        values: list[float] = []
        for canonical in ("ra", "dec"):
            found = None
            for name, unit in sources[canonical]:
                found = convert_canonical(canonical, row_get(row, name), unit)
                if found is not None:
                    break
            if found is None:  # static alias table only: the full normaliser
                found = normalize_source_record(row, columns=columns, field_map=field_map).get(canonical)
            try:
                values.append(float(found))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        return values[0], values[1]

    return read


class BatchCrossmatcher:
    """Crossmatch many targets against many catalogs with one request per catalog chunk.

    Parameters are optional: without ``client`` a client is created for each :meth:`run`; without
    ``registry`` the embedded catalog registry is used. ``guards`` (endpoint pacers/circuit breakers) and
    ``cache`` may be shared with the API's providers so batch cone searches and ordinary searches of the same
    endpoint are paced together; ``upload_slots`` (per-endpoint concurrency limits) may be shared between
    instances. ``fallback_to_cone`` sends a chunk whose upload/XMatch request keeps failing for a transient
    reason through per-target cone searches instead. ``association_config`` is the Bayesian association
    configuration (default that of :class:`crossmatch.CrossmatchService`).
    """

    retry_backoff_seconds = 1.0  # first retry delay; doubles per attempt

    def __init__(
        self,
        *,
        registry: CatalogRegistry | None = None,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
        guards: dict[str, EndpointGuard] | None = None,
        cache: CacheManager | None = None,
        cone_concurrency: int | None = None,
        chunk_concurrency: int | None = None,
        endpoint_concurrency: int | None = None,
        upload_timeout: float | None = None,
        chunk_sizes: Mapping[str, int] | None = None,
        max_cone_targets: int | None = None,
        fallback_to_cone: bool = True,
        attempts: int = 5,
        catalog_budget_seconds: float | None = None,
        max_response_bytes: int | None = None,
        association_config: AssociationConfig | None = None,
        upload_slots: Any = None,
    ) -> None:
        self.settings = settings or Settings()
        self.registry = registry or CatalogRegistry(self.settings.catalog_registry_path)
        self.client = client
        self.guards = guards if guards is not None else {}
        self.cache = cache or CacheManager(None)
        self.cone_concurrency = cone_concurrency or _env_int("BATCH_CONE_CONCURRENCY", 8)
        self.chunk_concurrency = chunk_concurrency or _env_int("BATCH_CHUNK_CONCURRENCY", 2)
        self.endpoint_concurrency = endpoint_concurrency or _env_int("BATCH_ENDPOINT_CONCURRENCY", 2)
        self.upload_timeout = upload_timeout or _env_float("BATCH_UPLOAD_TIMEOUT_SECONDS", 300.0)
        self.chunk_sizes = dict(chunk_sizes or {})
        self.max_cone_targets = max_cone_targets or _env_int("BATCH_MAX_CONE_TARGETS", 5000)
        self.fallback_to_cone = fallback_to_cone
        self.attempts = max(1, int(attempts))
        self.catalog_budget_seconds = catalog_budget_seconds or _env_float("BATCH_CATALOG_BUDGET_SECONDS", 1800.0)
        self.max_response_bytes = max_response_bytes or MAX_RESPONSE_BYTES
        self.association_config = association_config or AssociationConfig()
        # {event loop: {endpoint: Semaphore}} -- asyncio primitives belong to one loop.
        self.upload_slots = upload_slots if upload_slots is not None else weakref.WeakKeyDictionary()

    # -- planning -------------------------------------------------------------

    def default_catalogs(self) -> list[str]:
        """Enabled catalogs that need no per-target requests (upload or xmatch strategy)."""
        return [name for name in self.registry.enabled_catalogs() if self.default_strategy(name) != "cone"]

    def default_strategy(self, name: str) -> str:
        if name.startswith("vizier:") or name in XMATCH_VIEWS:
            return "xmatch"
        catalog = self.registry.get(name)
        if catalog.provider == "tap" and catalog.endpoint in UPLOAD_SERVICES:
            return "upload"
        return "cone"

    def strategy_table(self) -> dict[str, dict[str, Any]]:
        """Strategy, endpoint and chunk size of every registry catalog."""
        table: dict[str, dict[str, Any]] = {}
        for name, catalog in self.registry.catalogs.items():
            strategy = self.default_strategy(name)
            info: dict[str, Any] = {"strategy": strategy, "enabled": catalog.enabled, "provider": catalog.provider}
            if strategy == "upload":
                service = UPLOAD_SERVICES[str(catalog.endpoint)]
                info.update(endpoint=service.upload_endpoint, chunk_size=self._chunk_size(service.key, service.chunk_size),
                            note=service.note)
            elif strategy == "xmatch":
                view = XMATCH_VIEWS[name]
                info.update(endpoint=XMATCH_ENDPOINT, vizier_table=view.vizier_table,
                            chunk_size=self._chunk_size("xmatch", view.chunk_size), note=view.note)
            else:
                info.update(endpoint=catalog.endpoint, concurrency=self.cone_concurrency)
            table[name] = info
        return table

    def _chunk_size(self, key: str, default: int) -> int:
        return max(1, int(self.chunk_sizes.get(key, _env_int(f"BATCH_CHUNK_{key.upper()}", default))))

    def _resolve(self, catalogs: Sequence[str] | None, strategies: Mapping[str, str] | None) -> list[tuple[str, str]]:
        names = list(dict.fromkeys(c.strip() for c in (catalogs or self.default_catalogs()) if c and c.strip()))
        if not names:
            raise BatchError("no catalogs requested.")
        overrides = dict(strategies or {})
        unknown = set(overrides) - set(names)
        if unknown:
            raise BatchError(f"strategy given for catalog(s) not requested: {sorted(unknown)}")
        plan: list[tuple[str, str]] = []
        for name in names:
            if not name.startswith("vizier:") and name not in self.registry.catalogs:
                raise BatchError(f"unknown catalog {name!r}.")
            strategy = overrides.get(name) or self.default_strategy(name)
            if strategy not in STRATEGIES:
                raise BatchError(f"{name}: unknown strategy {strategy!r} (upload, xmatch or cone).")
            if strategy == "upload":
                catalog = self.registry.get(name) if name in self.registry.catalogs else None
                if catalog is None or catalog.provider != "tap" or catalog.endpoint not in UPLOAD_SERVICES:
                    raise BatchError(f"{name}: its archive does not support TAP uploads.")
            if strategy == "xmatch" and not (name.startswith("vizier:") or name in XMATCH_VIEWS):
                raise BatchError(f"{name}: no CDS XMatch (VizieR) view is defined.")
            if strategy == "cone" and name.startswith("vizier:"):
                raise BatchError(f"{name}: VizieR tables are matched with the xmatch strategy only.")
            plan.append((name, strategy))
        return plan

    # -- entry point ------------------------------------------------------------

    async def run(
        self,
        targets: Sequence[BatchTarget] | Sequence[Mapping[str, Any]],
        catalogs: Sequence[str] | None = None,
        *,
        radius_arcsec: float = 3.0,
        strategies: Mapping[str, str] | None = None,
        nearest_only: bool = False,
    ) -> BatchResult:
        """Crossmatch ``targets`` with ``catalogs`` (default: :meth:`default_catalogs`).

        CPU-bound work (target validation, response parsing, row conversion and the per-target association)
        runs in worker threads so an event loop serving other requests is never blocked for long.
        """
        radius = float(radius_arcsec)
        if not math.isfinite(radius) or not 0.0 < radius <= MAX_RADIUS_ARCSEC:
            raise BatchError(f"radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}].")
        items = list(targets)
        if items and all(isinstance(t, BatchTarget) for t in items):
            batch: list[BatchTarget] = items  # type: ignore[assignment]
        else:
            batch = await asyncio.to_thread(parse_targets, items)  # type: ignore[arg-type]
        plan = self._resolve(catalogs, strategies)
        started = time.perf_counter()
        owned = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.upload_timeout, follow_redirects=True)
        try:
            outcomes = await asyncio.gather(*(self._run_catalog(client, name, strategy, batch, radius)
                                              for name, strategy in plan))
        finally:
            if owned:
                await client.aclose()
        runs: dict[str, CatalogRun] = {}
        results: dict[str, dict[int, QueryResult]] = {}
        failures: dict[int, dict[str, str]] = defaultdict(dict)
        for (name, _strategy), (run, per_target, per_failure) in zip(plan, outcomes):
            runs[name] = run
            results[name] = per_target
            for idx, message in per_failure.items():
                failures[idx][name] = message
        matches, association = await asyncio.to_thread(
            self._associate, batch, [name for name, _ in plan], results, runs, radius, nearest_only)
        return BatchResult(batch, [name for name, _ in plan], radius, runs, matches, dict(failures),
                           time.perf_counter() - started, nearest_only, association)

    async def _run_catalog(
        self, client: httpx.AsyncClient, name: str, strategy: str, batch: Sequence[BatchTarget], radius: float,
    ) -> tuple[CatalogRun, dict[int, QueryResult], dict[int, str]]:
        started = time.perf_counter()
        deadline = time.monotonic() + self.catalog_budget_seconds
        if strategy == "xmatch":
            catalog, view = xmatch_catalog_definition(name, self.registry)
            endpoint: str | None = XMATCH_ENDPOINT
        else:
            catalog, view = self.registry.get(name), None
            endpoint = UPLOAD_SERVICES[str(catalog.endpoint)].upload_endpoint if strategy == "upload" else catalog.endpoint
        run = CatalogRun(name, strategy, endpoint, targets=len(batch), citation=catalog.citation,
                         acknowledgement=catalog.acknowledgement)

        def count() -> None:
            run.requests += 1

        counting = _CountingClient(client, count)
        results: dict[int, QueryResult] = {}
        failures: dict[int, str] = {}
        indices = list(range(len(batch)))
        try:
            if strategy == "upload":
                results, retry, failures = await self._run_upload(counting, catalog, batch, indices, radius, run, deadline)
            elif strategy == "xmatch":
                results, retry, failures = await self._run_xmatch(counting, catalog, view, batch, indices, radius, run,
                                                                   deadline)
            else:
                results, failures = await self._run_cone(counting, catalog, batch, indices, radius, run, deadline)
                retry = []
            if retry:
                if not self.fallback_to_cone or name.startswith("vizier:"):
                    for idx in retry:
                        failures[idx] = "; ".join(run.errors[-1:]) or "request failed"
                elif len(retry) > self.max_cone_targets:
                    message = (f"{len(retry)} targets would need per-target cone searches; at most "
                               f"{self.max_cone_targets} are allowed per batch (BATCH_MAX_CONE_TARGETS).")
                    run.errors.append(message)
                    failures.update({idx: message for idx in retry})
                else:
                    run.fallback_targets += len(retry)
                    cone_results, cone_failures = await self._run_cone(counting, self.registry.get(name), batch, retry,
                                                                       radius, run, deadline)
                    results.update(cone_results)
                    failures.update(cone_failures)
        except BatchError as exc:
            run.errors.append(str(exc))
            failures.update({idx: str(exc) for idx in indices if idx not in results})
        for result in results.values():
            run.warnings.extend(str(w) for w in (result.meta.get("warnings") or []))
        run.failed_targets = len(failures)
        run.elapsed_s = time.perf_counter() - started
        return run, results, failures

    # -- association ----------------------------------------------------------------

    def _associate(
        self, batch: Sequence[BatchTarget], catalogs: Sequence[str], results: Mapping[str, Mapping[int, QueryResult]],
        runs: Mapping[str, CatalogRun], radius: float, nearest_only: bool,
    ) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[int, dict[str, Any]]]:
        """Per target: :meth:`crossmatch.CrossmatchService.finalize` over the rows of every catalog.

        This is the single-object pipeline on the same rows: proper-motion/parallax adoption, final in-radius
        split and the Bayesian association whose target posterior becomes each match's ``confidence``.
        Targets without a single fetched row are skipped (finalize would return no match for them).
        """
        service = CrossmatchService(self.registry, {}, association_config=self.association_config)
        sigma = self.association_config.target_sigma_arcsec
        matches: dict[int, dict[str, list[dict[str, Any]]]] = {}
        association: dict[int, dict[str, Any]] = {}
        for idx, item in enumerate(batch):
            successes = [(name, results[name][idx]) for name in catalogs if idx in results[name]]
            if not any(_has_rows(result) for _, result in successes):
                continue
            requested = item.radius_arcsec or radius
            plans = [QueryPlan(name, "batch", None, {}, requested, "unknown") for name, _ in successes]
            ctx = SearchContext(target=item.target, plans=plans, search_radius=requested, query=None, profile=None,
                                pm_source=None, target_sigma_arcsec=sigma, target_pm_sigma_masyr=None)
            record = service.finalize(ctx, successes, [])
            prov = record.provenance
            info = prov.get("association") or {}
            association[idx] = {
                "p_any": info.get("p_any"),
                "best_match_probability": info.get("best_match_probability"),
                "target_proper_motion": prov.get("target_proper_motion"),
                "target_parallax": prov.get("target_parallax"),
                "warnings": [w for w in prov.get("warnings") or [] if "proper motion" in w.lower()][:5],
            }
            strategy_of = {name: str(result.meta.get("batch_strategy") or runs[name].strategy) for name, result in successes}
            by_catalog: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for group in record.counterparts.values():
                for match in group:
                    by_catalog[match["catalog"]].append(match)
            found_any: dict[str, list[dict[str, Any]]] = {}
            for name, found in by_catalog.items():
                found.sort(key=_match_sort_key)
                if nearest_only:
                    found = found[:1]
                found_any[name] = [_serialise_match(rank, match, strategy_of.get(name, runs[name].strategy))
                                   for rank, match in enumerate(found, start=1)]
                runs[name].matched_targets += 1
                runs[name].total_matches += len(found)
            if found_any:
                matches[idx] = found_any
        return matches, association

    # -- HTTP -----------------------------------------------------------------

    def _guard(self, endpoint: str) -> EndpointGuard:
        """Pacer/circuit breaker for upload/XMatch requests to ``endpoint``.

        Keyed separately from the cone-search guard of the same URL: a failing upload must not open the circuit
        that the per-target cone fallback of that archive then needs.
        """
        return self.guards.setdefault(f"batch-upload:{endpoint}", EndpointGuard(
            requests_per_second=float(os.getenv("PROVIDER_REQUESTS_PER_SECOND", "5")),
            failure_threshold=int(os.getenv("PROVIDER_FAILURE_THRESHOLD", "5")),
            recovery_seconds=float(os.getenv("PROVIDER_RECOVERY_SECONDS", "30")),
            probe_timeout_seconds=float(os.getenv("PROVIDER_PROBE_TIMEOUT_SECONDS", "180")),
        ))

    def _slot(self, endpoint: str) -> asyncio.Semaphore:
        """Per-endpoint limit on simultaneous upload/XMatch joins (across catalogs and shared engines)."""
        loop = asyncio.get_running_loop()
        per_loop = self.upload_slots.get(loop)
        if per_loop is None:
            per_loop = {}
            self.upload_slots[loop] = per_loop
        slot = per_loop.get(endpoint)
        if slot is None:
            slot = per_loop[endpoint] = asyncio.Semaphore(self.endpoint_concurrency)
        return slot

    async def _read(self, client: _CountingClient, endpoint: str, data: dict[str, str],
                    files: dict[str, tuple[str, bytes, str]], timeout: float) -> httpx.Response:
        """POST and read the answer, at most ``max_response_bytes`` (a larger answer raises _ResponseTooLarge)."""
        cap = self.max_response_bytes
        async with client.stream("POST", endpoint, data=data, files=files, timeout=timeout) as response:
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > cap:
                raise _ResponseTooLarge(f"answer of {declared} bytes exceeds {cap} bytes")
            parts: list[bytes] = []
            total = 0
            async for part in response.aiter_bytes():
                total += len(part)
                if total > cap:
                    raise _ResponseTooLarge(f"answer exceeds {cap} bytes")
                parts.append(part)
            headers = [(k, v) for k, v in response.headers.multi_items() if k.lower() not in _WIRE_HEADERS]
            return httpx.Response(response.status_code, headers=headers, content=b"".join(parts),
                                  request=response.request)

    async def _post(self, client: _CountingClient, endpoint: str, data: dict[str, str],
                    files: dict[str, tuple[str, bytes, str]], run: CatalogRun, *, deadline: float,
                    splittable: bool) -> httpx.Response:
        """POST multipart/form-data with pacing, a circuit breaker and retries on transient failures.

        Connection resets and HTTP 408/429/5xx are retried ``attempts`` times with exponential backoff
        (Retry-After honoured up to 30 s). CDS XMatch resets roughly one connection in six under load (observed
        live 2026-09-28 with curl and httpx alike), so a few retries are routine. A timeout is retried once;
        a second timeout splits a multi-target chunk (_TooSlow) instead of re-sending the same heavy join (a
        single target then falls back to a cone search). Every
        attempt reports to the circuit breaker exactly once, cancellation included.
        """
        guard = self._guard(endpoint)
        last_error = ""
        timeouts = 0
        for attempt in range(self.attempts):
            if attempt:
                run.retries += 1
                run.warnings.append(f"{run.catalog}: retried after {last_error}")
                logger.info("%s: retry %d after %s", run.catalog, attempt, last_error)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueryTimeoutError(f"{endpoint}: catalog time budget exhausted ({last_error or 'no answer'})")
            delay = self.retry_backoff_seconds * 2**attempt
            async with self._slot(endpoint):
                await guard.acquire()
                try:
                    response = await self._read(client, endpoint, data, files, min(self.upload_timeout, remaining))
                except _ResponseTooLarge as exc:
                    guard.record_success()  # the endpoint answered
                    raise _Overflow(str(exc)) from exc
                except _TIMEOUT_EXCEPTIONS as exc:
                    guard.record_failure()
                    timeouts += 1
                    last_error = f"{exc.__class__.__name__}: {exc}".rstrip(": ")
                    if timeouts >= 2:
                        if splittable:
                            raise _TooSlow(f"{endpoint}: timed out twice ({last_error})") from exc
                        raise QueryTimeoutError(f"{endpoint}: timed out twice ({last_error})") from exc
                except _RETRY_EXCEPTIONS as exc:
                    guard.record_failure()
                    last_error = f"{exc.__class__.__name__}: {exc}".rstrip(": ")
                except httpx.HTTPError as exc:
                    guard.record_failure()
                    raise CatalogUnavailableError(f"{endpoint}: {exc.__class__.__name__}: {exc}") from exc
                except BaseException:
                    # Cancellation (client disconnect, time budget) or a programming error: never leave a
                    # half-open probe dangling.
                    guard.record_failure()
                    raise
                else:
                    if response.status_code not in _RETRY_STATUS:
                        guard.record_success()
                        return response
                    guard.record_failure()
                    last_error = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = min(float(retry_after), 30.0) if retry_after else delay
                    except ValueError:
                        pass
            if attempt + 1 < self.attempts:
                await asyncio.sleep(max(0.0, min(delay, deadline - time.monotonic())))
        if "Timeout" in last_error:
            raise QueryTimeoutError(f"{endpoint}: {last_error} after {self.attempts} attempt(s)")
        raise CatalogUnavailableError(f"{endpoint}: {last_error} after {self.attempts} attempt(s)")

    @staticmethod
    def _parse(response: httpx.Response, label: str, expected: str | None) -> ParsedTable:
        TapProvider._check(response, label)
        return TapProvider._parse_table(response, label, expected)

    # -- chunk scheduling -------------------------------------------------------

    async def _chunked(
        self,
        chunks: list[list[int]],
        send: Callable[[list[int]], Any],
        run: CatalogRun,
        deadline: float,
    ) -> tuple[list[tuple[list[int], Any]], list[int], dict[int, str]]:
        """Send chunks (bounded concurrency), splitting overflowing / too slow chunks.

        Returns (answers, targets to retry by cone search, final failures). Transient/availability failures are
        retried by cone search; a deterministic query error (HTTP 4xx) or an exhausted time budget is final.
        """
        answers: list[tuple[list[int], Any]] = []
        retry: list[int] = []
        final: dict[int, str] = {}
        queue: list[list[int]] = list(chunks)
        semaphore = asyncio.Semaphore(self.chunk_concurrency)

        async def one(chunk: list[int]) -> None:
            async with semaphore:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    message = f"catalog time budget of {self.catalog_budget_seconds:g} s exhausted"
                    if message not in run.errors:
                        run.errors.append(message)
                    final.update({i: message for i in chunk})
                    return
                run.chunks += 1
                try:
                    answers.append((chunk, await asyncio.wait_for(send(chunk), timeout=remaining)))
                except (_Overflow, _TooSlow) as exc:
                    if len(chunk) > 1:
                        run.split_chunks += 1
                        half = len(chunk) // 2
                        queue.extend([chunk[:half], chunk[half:]])
                    else:
                        run.errors.append(f"single-target request failed: {exc}")
                        retry.extend(chunk)
                except TimeoutError:
                    message = f"catalog time budget of {self.catalog_budget_seconds:g} s exhausted"
                    run.errors.append(f"chunk of {len(chunk)} target(s): {message}")
                    final.update({i: message for i in chunk})
                except CatalogQueryError as exc:
                    message = f"chunk of {len(chunk)} target(s) failed: {exc.__class__.__name__}: {exc}"
                    logger.warning("%s: %s", run.catalog, message)
                    run.errors.append(message)
                    final.update({i: f"{exc.__class__.__name__}: {exc}" for i in chunk})
                except _FALLBACK_ERRORS as exc:
                    message = f"chunk of {len(chunk)} target(s) failed: {exc.__class__.__name__}: {exc}"
                    logger.warning("%s: %s", run.catalog, message)
                    run.errors.append(message)
                    retry.extend(chunk)

        while queue:
            pending, queue[:] = list(queue), []
            await asyncio.gather(*(one(chunk) for chunk in pending))
        return answers, retry, final

    # -- strategies ---------------------------------------------------------------

    async def _run_upload(
        self, client: _CountingClient, catalog: CatalogDefinition, batch: Sequence[BatchTarget], indices: list[int],
        radius: float, run: CatalogRun, deadline: float,
    ) -> tuple[dict[int, QueryResult], list[int], dict[int, str]]:
        service = UPLOAD_SERVICES[str(catalog.endpoint)]
        plans = {i: plan_cone(catalog, batch[i].target, batch[i].radius_arcsec or radius) for i in indices}
        adql = build_upload_adql(catalog, service)
        run.queries.append(adql)
        fmt = str(catalog.parameters.get("format", "json"))
        size = self._chunk_size(service.key, service.chunk_size)
        if service.upload_row_limit:
            size = min(size, service.upload_row_limit)
        converter = _RowConverter(client, "tap")
        label = f"TAP upload {catalog.name}"
        parameters = {"QUERY": adql, "UPLOAD": UPLOAD_TABLE}

        def parse_and_convert(chunk: list[int], response: httpx.Response) -> tuple[int, dict[int, QueryResult], dict[int, str]]:
            table = TapProvider._apply_column_units(catalog, self._parse(response, label, fmt))
            if table.truncated or len(table.rows) >= service.max_rows:
                raise _Overflow(f"{len(table.rows)} rows (MAXREC {service.max_rows}, status {table.query_status})")
            grouped, columns = _group_rows(table, UPLOAD_INDEX, frozenset(UPLOAD_COLUMNS), chunk, label)
            converted, failed = self._convert_chunk(converter, catalog, batch, chunk, plans, grouped, columns,
                                                    service.upload_endpoint, parameters, "upload", run)
            return len(table.rows), converted, failed

        async def send(chunk: list[int]) -> tuple[int, dict[int, QueryResult], dict[int, str]]:
            rows = [(i, plans[i].ra, plans[i].dec, plans[i].radius_arcsec / 3600.0) for i in chunk]
            form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql,
                    "UPLOAD": f"{UPLOAD_TABLE},param:{UPLOAD_TABLE}", "MAXREC": str(service.max_rows)}
            if fmt in {"json", "csv"}:
                form["FORMAT"] = fmt
            files = {UPLOAD_TABLE: (f"{UPLOAD_TABLE}.xml", upload_votable(rows), "application/x-votable+xml")}
            response = await self._post(client, service.upload_endpoint, form, files, run, deadline=deadline,
                                        splittable=len(chunk) > 1)
            return await asyncio.to_thread(parse_and_convert, chunk, response)

        chunks = [indices[k:k + size] for k in range(0, len(indices), size)]
        return await self._collect(chunks, send, run, deadline)

    async def _collect(self, chunks: list[list[int]], send: Callable[[list[int]], Any], run: CatalogRun,
                       deadline: float) -> tuple[dict[int, QueryResult], list[int], dict[int, str]]:
        answers, retry, final = await self._chunked(chunks, send, run, deadline)
        results: dict[int, QueryResult] = {}
        for _chunk, (n_rows, converted, failed) in answers:
            run.rows_returned += n_rows
            results.update(converted)
            final.update(failed)
        return results, retry, final

    async def _run_xmatch(
        self, client: _CountingClient, catalog: CatalogDefinition, view: XMatchView | None,
        batch: Sequence[BatchTarget], indices: list[int], radius: float, run: CatalogRun, deadline: float,
    ) -> tuple[dict[int, QueryResult], list[int], dict[int, str]]:
        plans = {i: plan_cone(catalog, batch[i].target, batch[i].radius_arcsec or radius) for i in indices}
        too_wide = [i for i in indices if plans[i].radius_arcsec > XMATCH_MAX_DISTANCE_ARCSEC]
        wide = set(too_wide)
        if too_wide:
            run.warnings.append(
                f"{catalog.name}: {len(too_wide)} target cone(s) wider than the XMatch limit of "
                f"{XMATCH_MAX_DISTANCE_ARCSEC:g} arcsec (epoch widening) were sent to cone searches."
            )
        usable = [i for i in indices if i not in wide]
        size = self._chunk_size("xmatch", view.chunk_size if view else 20_000)
        radii = {i: math.ceil(plans[i].radius_arcsec * 1000.0) / 1000.0 for i in usable}
        chunks: list[list[int]] = []
        for bucket in radius_buckets(usable, radii):
            widest = max(radii[i] for i in bucket)
            # Expected rows per request ~ targets x radius^2: shrink chunks of wide cones accordingly.
            scaled = max(1, min(size, int(size * (XMATCH_REFERENCE_RADIUS_ARCSEC / max(widest, XMATCH_REFERENCE_RADIUS_ARCSEC)) ** 2)))
            chunks.extend(bucket[k:k + scaled] for k in range(0, len(bucket), scaled))
        cols2 = ",".join(view.columns) if view and view.columns else None
        table_name = catalog.table or ""
        converter = _RowConverter(client, "cds_xmatch")
        label = f"CDS XMatch {catalog.name}"

        def parse_and_convert(chunk: list[int], response: httpx.Response) -> tuple[int, dict[int, QueryResult], dict[int, str]]:
            table = self._parse(response, label, "votable")
            if table.truncated or len(table.rows) >= XMATCH_MAXREC:
                raise _Overflow(f"{len(table.rows)} rows (MAXREC {XMATCH_MAXREC}, status {table.query_status})")
            if view is not None and view.renames:
                _rename_columns(table, view.renames)
            grouped, columns = _group_rows(table, UPLOAD_INDEX, XMATCH_EXTRA_COLUMNS, chunk, label)
            definition = _with_ellipse_errors(catalog, columns)
            position = _position_reader(definition, columns)
            inside: dict[int, list[dict[str, Any]]] = {}
            for i in chunk:
                plan = plans[i]
                kept = []
                for row in grouped.get(i, []):
                    pos = position(row)
                    # XMatch used the bucket's widest radius: cut back to this target's own cone.
                    if pos is None or haversine_arcsec(plan.ra, plan.dec, pos[0], pos[1]) <= plan.radius_arcsec * (1.0 + 1e-9) + 1e-6:
                        kept.append(row)
                inside[i] = kept
            converted, failed = self._convert_chunk(converter, definition, batch, chunk, plans, inside, columns,
                                                    XMATCH_ENDPOINT, {"cat2": table_name}, "xmatch", run)
            return len(table.rows), converted, failed

        async def send(chunk: list[int]) -> tuple[int, dict[int, QueryResult], dict[int, str]]:
            dist = min(XMATCH_MAX_DISTANCE_ARCSEC, max(radii[i] for i in chunk))
            body = upload_csv([(i, plans[i].ra, plans[i].dec) for i in chunk])
            if len(body) > XMATCH_MAX_UPLOAD_BYTES:
                raise _Overflow(f"upload of {len(body)} bytes exceeds 100 MB")
            form = {"request": "xmatch", "distMaxArcsec": f"{dist:.3f}", "RESPONSEFORMAT": "votable",
                    "cat2": table_name, "colRA1": "t_ra", "colDec1": "t_dec", "selection": "all",
                    "MAXREC": str(XMATCH_MAXREC)}
            if cols2:
                form["cols2"] = cols2
            files = {"cat1": ("targets.csv", body, "text/csv")}
            response = await self._post(client, XMATCH_ENDPOINT, form, files, run, deadline=deadline,
                                        splittable=len(chunk) > 1)
            return await asyncio.to_thread(parse_and_convert, chunk, response)

        run.queries.append(f"CDS XMatch cat2={table_name} selection=all" + (f" cols2={cols2}" if cols2 else ""))
        results, retry, final = await self._collect(chunks, send, run, deadline)
        return results, retry + too_wide, final

    async def _run_cone(
        self, client: _CountingClient, catalog: CatalogDefinition, batch: Sequence[BatchTarget], indices: list[int],
        radius: float, run: CatalogRun, deadline: float,
    ) -> tuple[dict[int, QueryResult], dict[int, str]]:
        if len(indices) > self.max_cone_targets:
            raise BatchError(f"{catalog.name}: {len(indices)} targets need per-target cone searches; at most "
                             f"{self.max_cone_targets} are allowed per batch (BATCH_MAX_CONE_TARGETS).")
        providers = provider_map(client, timeout=self.settings.request_timeout_seconds,  # type: ignore[arg-type]
                                 max_response_bytes=self.settings.max_response_bytes, guards=self.guards, cache=self.cache)
        executor = QueryExecutor(providers, timeout=self.settings.request_timeout_seconds, registry=self.registry,
                                 timeout_cap=self.settings.catalog_timeout_cap_seconds)
        semaphore = asyncio.Semaphore(self.cone_concurrency)
        results: dict[int, QueryResult] = {}
        failures: dict[int, str] = {}
        budget_message = f"catalog time budget of {self.catalog_budget_seconds:g} s exhausted"

        async def one(i: int) -> None:
            item = batch[i]
            plan = QueryPlan(catalog.name, catalog.provider, catalog.endpoint, {}, item.radius_arcsec or radius,
                             catalog.wavelength)
            async with semaphore:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failures[i] = budget_message
                    return
                try:
                    successes, failed = await asyncio.wait_for(executor.execute([plan], item.target), timeout=remaining)
                except TimeoutError:
                    failures[i] = budget_message
                    return
            if successes:
                result = successes[0][1]
                assert isinstance(result, QueryResult)
                result.meta["batch_strategy"] = "cone"
                results[i] = result
                run.rows_returned += int(result.meta.get("raw_row_count") or len(result))
                if result.meta.get("fallback"):
                    run.warnings.append(f"{catalog.name}: fallback {result.meta['fallback'].get('provider')} used")
            else:
                failures[i] = f"{failed[0].error_type}: {failed[0].message}"

        await asyncio.gather(*(one(i) for i in indices))
        if failures:
            run.errors.append(f"{len(failures)} cone search(es) failed; first: {next(iter(failures.values()))}")
        run.queries.append(f"per-target cone searches via provider '{catalog.provider}'")
        return results, failures

    def _convert_chunk(
        self, converter: _RowConverter, catalog: CatalogDefinition, batch: Sequence[BatchTarget], chunk: list[int],
        plans: Mapping[int, ConePlan], grouped: Mapping[int, list[dict[str, Any]]], columns: list[ColumnMeta],
        endpoint: str, parameters: dict[str, Any], strategy: str, run: CatalogRun,
    ) -> tuple[dict[int, QueryResult], dict[int, str]]:
        """Rows of every target of a chunk -> QueryResults through the providers' row pipeline.

        A target whose rows cannot be converted (no usable position in any row) is a failure, not "no match".
        The column metadata is stored once per chunk (shared by the targets' results).
        """
        column_meta = [c.as_dict() for c in columns]
        results: dict[int, QueryResult] = {}
        failed: dict[int, str] = {}
        for i in chunk:
            rows = grouped.get(i, [])
            plan = plans[i]
            # Every row inside the cone was returned (no TOP N): never mistake a full cone for an archive cut.
            cone = replace(plan, row_limit=max(plan.row_limit, len(rows)), warnings=list(plan.warnings))
            try:
                results[i] = converter._sources(
                    catalog, rows, plan.requested_radius_arcsec, endpoint, parameters, columns=columns,
                    target=batch[i].target, cone=cone,
                    meta={"endpoint": endpoint, "batch_strategy": strategy, "columns": column_meta},
                )
            except ResponseParseError as exc:
                failed[i] = f"ResponseParseError: {exc}"
                run.errors.append(f"target {batch[i].id!r}: {exc}")
        return results, failed


def _has_rows(result: QueryResult) -> bool:
    meta = getattr(result, "meta", {}) or {}
    return bool(len(result) or meta.get("pad_sources") or meta.get("excess_sources"))


def _serialise_match(rank: int, match: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    """A counterpart of :meth:`crossmatch.CrossmatchService.finalize` as a batch match row."""
    meta = match.get("metadata") or {}
    return {
        "catalog": match["catalog"],
        "strategy": strategy,
        "rank": rank,
        "source_id": match["source_id"],
        "ra": match["ra"],
        "dec": match["dec"],
        "separation_arcsec": match["separation_arcsec"],
        "query_separation_arcsec": meta.get("query_separation_arcsec"),
        "epoch_propagation": meta.get("epoch_propagation"),
        "positional_error_arcsec": match.get("positional_error_arcsec"),
        "epoch": match.get("epoch"),
        "epoch_range": match.get("epoch_range"),
        "pm_ra_masyr": match.get("proper_motion_ra_masyr"),
        "pm_dec_masyr": match.get("proper_motion_dec_masyr"),
        "physical": match.get("physical") or {},
        # Posterior probability that the row is the target's counterpart (crossmatch.associate_matches).
        "confidence": match.get("confidence"),
        "data": match.get("data"),
    }


def _group_rows(
    table: ParsedTable, index_column: str, drop: frozenset[str], chunk: Sequence[int], label: str,
) -> tuple[dict[int, list[dict[str, Any]]], list[ColumnMeta]]:
    """Split joined rows by uploaded target index; drop the upload/XMatch helper columns.

    A row without a valid index of this chunk means the answer cannot be attributed (e.g. a service that
    renamed the index column): the chunk fails with ResponseParseError rather than silently losing rows.
    """
    drop_lower = {d.lower() for d in drop}
    index_name = next((c.name for c in table.columns if c.name.lower() == index_column.lower()), None)
    columns = [c for c in table.columns if c.name.lower() not in drop_lower]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not table.rows:
        return grouped, columns
    if index_name is None:
        raise ResponseParseError(f"{label}: {len(table.rows)} row(s) but no {index_column!r} column "
                                 f"(columns: {[c.name for c in table.columns][:20]}).")
    allowed = set(chunk)
    for row in table.rows:
        raw = row.get(index_name)
        try:
            idx = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ResponseParseError(f"{label}: row with unusable {index_column} value {raw!r}.") from None
        if idx not in allowed:
            raise ResponseParseError(f"{label}: row for target index {idx}, which was not uploaded in this chunk.")
        grouped[idx].append({k: v for k, v in row.items() if str(k).lower() not in drop_lower})
    return grouped, columns


def _rename_columns(table: ParsedTable, renames: Mapping[str, str]) -> None:
    """Rename XMatch/VizieR columns to the registry (Gaia archive) names, keeping units and UCDs."""
    for col in table.columns:
        if col.name in renames:
            col.name = renames[col.name]
    table.rows = [{renames.get(k, k): v for k, v in row.items()} for row in table.rows]


async def batch_crossmatch(
    targets: Sequence[BatchTarget] | Sequence[Mapping[str, Any]],
    catalogs: Sequence[str] | None = None,
    *,
    radius_arcsec: float = 3.0,
    strategies: Mapping[str, str] | None = None,
    nearest_only: bool = False,
    client: httpx.AsyncClient | None = None,
    registry: CatalogRegistry | None = None,
) -> BatchResult:
    """Notebook-friendly one-call batch crossmatch (see :class:`BatchCrossmatcher`)."""
    engine = BatchCrossmatcher(registry=registry, client=client)
    return await engine.run(targets, catalogs, radius_arcsec=radius_arcsec, strategies=strategies, nearest_only=nearest_only)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

OUTPUT_FORMATS = ("json", "rows", "parquet", "csv")


class BatchRequest(BaseModel):
    """JSON body. ``targets`` are validated by :func:`parse_targets` (aliases such as ``name``, ``pmra``,
    ``pmdec``, ``plx``, ``radius``, ``RAJ2000`` are accepted exactly as in CSV and file uploads)."""

    targets: list[dict[str, Any]] = Field(..., min_length=1)
    catalogs: list[str] | None = None
    radius_arcsec: float = Field(default=3.0, gt=0.0, le=MAX_RADIUS_ARCSEC)
    strategies: dict[str, str] | None = None
    nearest_only: bool = False
    include_data: bool = True
    format: str = Field(default="json", pattern="^(json|rows|parquet|csv)$")


def _split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _parse_strategies(value: str | None) -> dict[str, str] | None:
    """``catalog=strategy,catalog=strategy`` (form/query parameter)."""
    if not value:
        return None
    out: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            raise BatchError(f"strategy override {part!r} must look like catalog=strategy.")
        name, strategy = part.split("=", 1)
        out[name.strip()] = strategy.strip()
    return out


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _multipart_fields(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """Parse multipart/form-data with the standard library: {field: (filename, content)}."""
    message = BytesParser(policy=HTTP_POLICY).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise BatchError("malformed multipart/form-data body.")
    fields: dict[str, tuple[str | None, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[str(name)] = (part.get_filename(), payload)
    return fields


async def _read_body(request: Request, max_bytes: int) -> bytes:
    """The request body, refused (413) as soon as it is known to exceed ``max_bytes`` -- from Content-Length
    before anything is read, else while streaming."""
    too_large = HTTPException(status_code=413, detail=f"request body exceeds {max_bytes} bytes (BATCH_MAX_UPLOAD_BYTES).")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise too_large
    parts: list[bytes] = []
    total = 0
    async for part in request.stream():
        total += len(part)
        if total > max_bytes:
            raise too_large
        parts.append(part)
    return b"".join(parts)


async def _request_to_job(request: Request, query: Mapping[str, Any]) -> dict[str, Any]:
    """Build the batch job from JSON, CSV (text/csv body) or multipart (file field) requests.

    Target parsing (CPU-bound for large lists) runs in a worker thread.
    """
    content_type = request.headers.get("content-type", "").lower()
    body = await _read_body(request, _env_int("BATCH_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    job: dict[str, Any] = {
        "catalogs": _split_list(query.get("catalogs")),
        "radius_arcsec": query.get("radius_arcsec"),
        "strategies": _parse_strategies(query.get("strategies")),
        "nearest_only": query.get("nearest_only"),
        "include_data": query.get("include_data"),
        "format": query.get("format"),
    }
    if content_type.startswith("application/json") or (not content_type and body.lstrip()[:1] in (b"{", b"[")):
        try:
            payload = json.loads(body or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}") from exc
        if isinstance(payload, list):
            payload = {"targets": payload}
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object with 'targets'.")
        merged = {**{k: v for k, v in job.items() if v is not None}, **payload}
        try:
            model = BatchRequest.model_validate(merged)
        except Exception as exc:  # pydantic.ValidationError
            errors = getattr(exc, "errors", None)
            detail = json.loads(json.dumps(errors(), default=str)) if callable(errors) else str(exc)
            raise HTTPException(status_code=422, detail=detail) from exc
        job.update(model.model_dump(exclude={"targets"}))
        job["targets"] = await asyncio.to_thread(parse_targets, model.targets)
        return job
    if content_type.startswith("multipart/form-data"):
        fields = _multipart_fields(body, request.headers.get("content-type", ""))
        upload = fields.get("file") or fields.get("targets")
        if upload is None:
            raise HTTPException(status_code=422, detail="multipart request needs a 'file' field (CSV or JSON targets).")
        filename, content = upload
        for key in ("catalogs", "radius_arcsec", "strategies", "nearest_only", "include_data", "format"):
            if key in fields and job.get(key) is None:
                text = fields[key][1].decode("utf-8", "replace")
                job[key] = _split_list(text) if key == "catalogs" else _parse_strategies(text) if key == "strategies" else text
        is_json = (filename or "").lower().endswith(".json") or content.lstrip(b"\xef\xbb\xbf \r\n\t")[:1] in (b"[", b"{")
        if is_json:
            job["targets"] = await asyncio.to_thread(read_targets_json, content.decode("utf-8-sig"))
        else:
            job["targets"] = await asyncio.to_thread(read_targets_csv, content)
        return job
    if content_type.startswith(("text/csv", "text/plain", "application/csv")) or body:
        job["targets"] = await asyncio.to_thread(read_targets_csv, body)
        return job
    raise HTTPException(status_code=422, detail="send targets as JSON, a text/csv body or a multipart 'file' field.")


def _engine_for(request: Request) -> BatchCrossmatcher:
    """Engine sharing the API's registry, client, association settings and -- via the providers of
    ``app.state.providers`` (or the service's) -- its endpoint guards and cache, so batch cone searches and
    ordinary searches of one endpoint are paced and circuit-broken together."""
    state = request.app.state
    registry = getattr(state, "registry", None)
    service = getattr(state, "service", None)
    if registry is None and service is not None:
        registry = getattr(service, "registry", None)
    client = getattr(state, "client", None)
    providers = getattr(state, "providers", None) or getattr(service, "providers", None) or {}
    shared = providers.get("tap") if isinstance(providers, Mapping) else None
    guards = getattr(shared, "guards", None)
    cache = getattr(shared, "cache", None)
    if guards is None:
        guards = getattr(state, "batch_guards", None)
        if guards is None:
            guards = {}
            try:
                state.batch_guards = guards  # shared by later batches so concurrent jobs stay paced
            except Exception:
                pass
    slots = getattr(state, "batch_upload_slots", None)
    if slots is None:
        slots = weakref.WeakKeyDictionary()
        try:
            state.batch_upload_slots = slots
        except Exception:
            pass
    config = getattr(service, "association_config", None)
    return BatchCrossmatcher(registry=registry, client=client, guards=guards, cache=cache, upload_slots=slots,
                             association_config=config if isinstance(config, AssociationConfig) else None)


@router.get("/strategies")
async def batch_strategies(request: Request) -> dict[str, Any]:
    """How each registry catalog is matched in a batch (upload | xmatch | cone), endpoints and chunk sizes."""
    engine = _engine_for(request)
    return {"default_catalogs": engine.default_catalogs(), "catalogs": engine.strategy_table(),
            "vizier_views": {name: view.note for name, view in VIZIER_VIEWS.items()},
            "max_radius_arcsec": MAX_RADIUS_ARCSEC}


@router.post("/crossmatch")
async def batch_crossmatch_endpoint(
    request: Request,
    catalogs: str | None = Query(default=None, description="Comma-separated catalog names (default: upload/xmatch catalogs)"),
    radius_arcsec: float | None = Query(default=None, gt=0.0, le=MAX_RADIUS_ARCSEC),
    strategies: str | None = Query(default=None, description="Overrides, e.g. 'simbad=cone,gaia_dr3=xmatch'"),
    nearest_only: bool | None = Query(default=None),
    include_data: bool | None = Query(default=None),
    format: str | None = Query(default=None, pattern="^(json|rows|parquet|csv)$"),
) -> Response:
    """Batch crossmatch of up to BATCH_MAX_TARGETS targets.

    Body: JSON ``{"targets": [{id, ra, dec, epoch?}], "catalogs": [...], "radius_arcsec": 3, ...}``, a
    ``text/csv`` target table, or multipart/form-data with a ``file`` field (CSV or JSON). ``format``: ``json``
    (per-target matches), ``rows`` (flat dataset-like rows), ``parquet`` or ``csv`` (flat rows as a file).
    Each match's ``confidence`` is the posterior probability that the row is the target's counterpart.
    """
    query = {"catalogs": catalogs, "radius_arcsec": radius_arcsec, "strategies": strategies,
             "nearest_only": nearest_only, "include_data": include_data, "format": format}
    try:
        job = await _request_to_job(request, query)
        engine = _engine_for(request)
        radius = float(job.get("radius_arcsec") or 3.0)
        fmt = str(job.get("format") or "json").lower()
        if fmt not in OUTPUT_FORMATS:
            raise BatchError(f"format must be one of {', '.join(OUTPUT_FORMATS)}.")
        include = True if job.get("include_data") is None else _as_bool(job["include_data"])
        result = await engine.run(job["targets"], job.get("catalogs"), radius_arcsec=radius,
                                  strategies=job.get("strategies"), nearest_only=_as_bool(job.get("nearest_only") or False))
    except BatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid batch request: {exc}") from exc
    if result.runs and all(run.failed_targets == run.targets and run.targets for run in result.runs.values()):
        raise HTTPException(status_code=502, detail={"message": "every catalog failed",
                                                     "catalogs": {n: r.as_dict() for n, r in result.runs.items()}})
    headers = {"X-Batch-Request-Count": str(result.request_count), "X-Batch-Wall-Time-S": f"{result.wall_time_s:.3f}"}
    if fmt == "parquet":
        headers["Content-Disposition"] = 'attachment; filename="batch_crossmatch.parquet"'
        content = await asyncio.to_thread(result.to_parquet_bytes, include_data=include)
        return Response(content, media_type="application/vnd.apache.parquet", headers=headers)
    if fmt == "csv":
        headers["Content-Disposition"] = 'attachment; filename="batch_crossmatch.csv"'
        text = await asyncio.to_thread(result.to_csv_text, include_data=include)
        return Response(text, media_type="text/csv", headers=headers)

    def payload() -> Any:
        if fmt == "rows":
            data = {"summary": result.summary(), "catalogs": {n: r.as_dict() for n, r in result.runs.items()},
                    "rows": result.rows(include_data=include)}
        else:
            data = result.as_dict(include_data=include)
        return json.loads(json.dumps(data, default=str))

    return JSONResponse(await asyncio.to_thread(payload), headers=headers)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def format_report(result: BatchResult) -> str:
    """Plain-text per-catalog report: strategy, requests, time, matches."""
    lines = [f"Batch crossmatch: {len(result.targets)} targets x {len(result.catalogs)} catalogs, "
             f"radius {result.radius_arcsec:g} arcsec -> {result.request_count} requests in {result.wall_time_s:.1f} s"]
    lines.append(f"{'catalog':<22}{'strategy':<9}{'requests':>9}{'retries':>8}{'time[s]':>9}{'matched':>9}{'matches':>9}  errors")
    for name in result.catalogs:
        run = result.runs[name]
        lines.append(f"{name:<22}{run.strategy:<9}{run.requests:>9}{run.retries:>8}{run.elapsed_s:>9.1f}"
                     f"{run.matched_targets:>9}{run.total_matches:>9}  {len(run.errors)}")
        for error in run.errors[:3]:
            lines.append(f"    ! {error}")
        if run.fallback_targets:
            lines.append(f"    {run.fallback_targets} target(s) fell back to cone searches")
    return "\n".join(lines)


def run_cli(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch batch``; returns a process exit code."""
    try:
        targets = load_targets_file(args.targets)
        catalogs = _split_list(args.catalogs)
        strategies = _parse_strategies(args.strategy)
        fmt = args.format or (Path(args.out).suffix.lstrip(".").lower() if args.out else "parquet") or "parquet"
        if fmt not in {"parquet", "csv", "json"}:
            raise BatchError(f"unsupported output format {fmt!r} (parquet, csv or json).")
        engine = BatchCrossmatcher()
        result = asyncio.run(engine.run(targets, catalogs, radius_arcsec=args.radius, strategies=strategies,
                                        nearest_only=args.nearest))
    except (BatchError, OSError, KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    except (AstroSearchError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1
    print(format_report(result))
    if args.out:
        path = result.write(args.out, fmt, include_data=not args.no_data)
        print(f"Wrote {len(result.rows(include_data=False))} match rows to {path}")
    return 0 if all(run.failed_targets < run.targets for run in result.runs.values()) else 1


def register_cli(subparsers: Any) -> None:
    """Add the ``batch`` subcommand to an argparse subparsers object."""
    parser = subparsers.add_parser("batch", help="Batch crossmatch a target list (TAP upload / CDS XMatch / cones)")
    parser.add_argument("--targets", required=True, help="CSV (id,ra,dec[,epoch,pmra,pmdec]) or JSON target file")
    parser.add_argument("--catalogs", help="Comma-separated catalogs (default: all upload/xmatch-capable catalogs)")
    parser.add_argument("--radius", type=float, default=3.0, help="Match radius in arcsec (default 3, max 180)")
    parser.add_argument("--out", help="Output file (.parquet, .csv or .json)")
    parser.add_argument("--format", choices=["parquet", "csv", "json"], help="Output format (default from --out suffix)")
    parser.add_argument("--strategy", help="Strategy overrides, e.g. 'simbad=cone,twomass_psc=upload'")
    parser.add_argument("--nearest", action="store_true", help="Keep only the nearest match per catalog")
    parser.add_argument("--no-data", action="store_true", help="Omit the raw catalog row (data_json) from the output")
    parser.set_defaults(handler=run_cli)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(prog="batch")
    register_cli(_parser.add_subparsers(dest="command"))
    _args = _parser.parse_args()
    raise SystemExit(_args.handler(_args) if hasattr(_args, "handler") else 2)
