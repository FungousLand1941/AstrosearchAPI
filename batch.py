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
    positional-error conventions and share their registry survey's association priors (sky density, density-map
    scaling, resolution); other tables are described by :func:`describe_generic_answer`: positions are the ones
    XMatch matched on -- VizieR's computed ``_RAJ2000``/``_DEJ2000`` (Hipparcos' ``meta.main`` columns are at
    J1991.25) or the table's ``meta.main`` columns --, dated J2000.0 for rows with a proper motion when that
    position is at J2000 (``_RAJ2000``; UCAC4/USNO-B1.0/NOMAD/PPMXL ``RAJ2000``, URAT1 ``RA2000``, Gaia DR2/EDR3
    ``ra_epoch2000``; a mean observation epoch column never re-dates them); identifiers the ``meta.id;meta.main``
    column, the ``meta.id.part;meta.main`` parts (Tycho-2 TYC1-TYC2-TYC3) or an identifier-like bare ``meta.id``
    (Gaia ``source_id``); positional errors XMatch's 1-sigma ellipse, else its circular ``radec_err``. Their
    cones are planned for J2000 (:func:`_planning_catalog`), the epoch XMatch matches them at. XMatch declares
    some string columns shorter than their values (USNO-B1.0, NOMAD1.0: arraysize 12 for 13 characters): the
    declared sizes are relaxed before parsing (:func:`_relax_char_arraysize`) so identifiers are never cut.

    ``distMaxArcsec`` is one value per request, so targets are grouped by cone radius into geometric buckets
    (a new bucket starts when a radius exceeds :data:`XMATCH_RADIUS_BUCKET_RATIO` times the bucket's smallest):
    targets with individual epochs, proper motions or radii still share O(log(r_max/r_min)) requests, and every
    target's rows are then cut back to its own cone with the ``angDist`` XMatch measured. A cone wider than 180"
    (epoch widening) is cone-searched when the catalog has a registry cone search (``fallback_to_cone``, at most
    ``BATCH_MAX_CONE_TARGETS``); otherwise a cone widened for an unknown proper motion is capped at 180" with a
    warning, and a proper-motion track that does not fit is refused with its reason (never sent upstream).

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

Large lists are chunked (per-service chunk sizes below; upload and XMatch chunks shrink with the cone area so
wide cones do not overflow), each chunk is retried on transient failures (connection resets and connect
timeouts, HTTP 408/429/5xx, honouring Retry-After), a chunk whose answer hit the row or byte limit (QUERY_STATUS
OVERFLOW -- detected before parsing --, ``MAXREC`` rows, :data:`MAX_RESPONSE_BYTES`) or whose request timed out
twice on the server side (read/write timeout) is split in two and re-sent, and a chunk that still fails for a
transient reason falls back to per-target cone searches (a deterministic HTTP 4xx query error does not: it
would fail identically for every target). Each catalog has a time budget (``BATCH_CATALOG_BUDGET_SECONDS``);
targets left when it runs out are reported as failed. Every failed (target, catalog) pair carries its own
reason, also in the flat outputs (``status`` "failed" rows), so a failed query is never read as "no
counterpart". Request counts, retries and wall times are reported per catalog. CPU-bound steps (planning,
parsing, conversion, association, serialisation) run one at a time in a worker thread (:func:`_offload`), in
short time slices served least-CPU-used request first (:func:`_offload_steps`), so a small request is not held
up by a large batch; the REST handler cancels a batch whose client disconnected, which stops its CPU work after
the running slice. The batch routes enforce their own body limit (``BATCH_MAX_UPLOAD_BYTES``) while streaming;
an application-wide body-size middleware must exempt them (:func:`owns_request_body_limit`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import functools
import heapq
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from email.parser import BytesParser
from email.policy import HTTP as HTTP_POLICY
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from astrometry import AssociationConfig
from crossmatch import CrossmatchService, QueryExecutor, SearchContext, coordinate_sigma_arcsec
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
    catalog_epoch_range,
    convert_canonical,
    epoch_pad_settings,
    epoch_to_jyear,
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
    "describe_generic_answer",
    "load_targets_file",
    "owns_request_body_limit",
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
# Column list of a table in the XMatch service (HTTP 400 {"error": ...} for a table it does not serve).
XMATCH_TABLES_ENDPOINT = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync/tables"
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
# XMatch echoes the uploaded columns and adds angDist (arcsec from the uploaded position to the catalogue position
# it matched on); angDist is used to cut every target's rows back to its own cone, then dropped as well.
XMATCH_ECHO_COLUMNS = frozenset({"t_idx", "t_ra", "t_dec", "t_rad"})
XMATCH_EXTRA_COLUMNS = XMATCH_ECHO_COLUMNS | {"angdist"}
# VizieR's computed J2000 positions: CDS XMatch matches a VizieR table on these when the table has them (verified
# live 2026-09-29: Hipparcos I/239/hip_main and I/311/hip2, whose meta.main positions RAICRS/RArad are at
# J1991.25, return angDist from _RAJ2000/_DEJ2000 -- Barnard's star 0.084" from its SIMBAD J2000 position while
# its J1991.25 position is 90" away; Tycho-2 I/259/tyc2 marks them meta.main). For rows with a proper motion
# VizieR propagates them to epoch J2000.0.
XMATCH_J2000_COLUMNS = ("_RAJ2000", "_DEJ2000")
XMATCH_J2000_EPOCH = 2000.0
# Columns added to generic (vizier:<table>) XMatch rows: the identifier composed from the table's
# meta.id.part;meta.main columns (Tycho-2 TYC1-TYC2-TYC3) or from the position, and each row's epoch.
GENERIC_ID_COLUMN = "_source_id"
GENERIC_EPOCH_COLUMN = "_epoch"
# Units VizieR mislabels in XMatch metadata (I/259/tyc2 pmRA is declared 'ma/yr'; its ReadMe says mas/yr).
_UNIT_TYPOS = {"ma/yr": "mas/yr"}

STRATEGIES = ("upload", "xmatch", "cone")
MAX_RADIUS_ARCSEC = XMATCH_MAX_DISTANCE_ARCSEC


def _env_int_or_zero(name: str, default: int) -> int:
    """An integer setting that may be 0 (disabled)."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


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
# Catalogs per batch request: each is at least one upstream join, so one call must not fan out without bound.
DEFAULT_MAX_CATALOGS = 20


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
    ``density_key`` names the registry survey whose published sky density, density-map scaling and angular
    resolution the Bayesian association uses for the view (the same table under another name must not get a
    different prior: see :func:`_register_view_priors`).
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
    density_key: str | None = None


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
    density_key="gaia_dr3",
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
        density_key="twomass_psc",
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
        density_key="allwise",
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
        density_key="panstarrs_dr2",
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
        density_key="sdss",
    ),
}

_VIZIER_ACK = "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France (DOI 10.26093/cds/vizier)."
_XMATCH_ACK = "This research has made use of the cross-match service provided by CDS, Strasbourg."


def _view_aliases() -> dict[str, str]:
    """{view catalog name: registry survey name} for views that are a registry survey under another name."""
    views = {**VIZIER_VIEWS, **XMATCH_VIEWS}
    return {name: view.density_key for name, view in views.items() if view.density_key and view.density_key != name}


def _register_view_priors() -> None:
    """Give each VizieR view the association priors of the survey it serves.

    The Bayesian association (:func:`crossmatch.catalog_densities`, :func:`astrometry.local_prior_density`)
    looks up a catalogue's published sky density and density-map scaling by catalogue name; a name it does not
    know gets the generic prior mean (1000 per deg^2), which the Gamma-Poisson update then pulls towards the rows
    inside the (tiny) cone -- e.g. >= 458,000 per deg^2 for one row in a 3" cone, 40x the 2MASS mean -- so the
    same 2MASS row matched as ``vizier:II/246/out`` got a posterior of 0.108 instead of twomass_psc's 0.833.
    The views are the registry surveys' own tables (same sources, same selection: SDSS primary objects, PS1
    nDetections > 1), so their names are registered as aliases of the survey's entries. ``setdefault`` never
    overrides a value another module has set for the name.
    """
    import astrometry

    tables = [getattr(astrometry, attr, None) for attr in ("CATALOG_SKY_DENSITY", "DENSITY_MAP_SCALING",
                                                            "CATALOG_RESOLUTION_ARCSEC")]
    for name, key in _view_aliases().items():
        for table in tables:
            if isinstance(table, dict) and key in table:
                table.setdefault(name, table[key])


_register_view_priors()


class BatchError(AstroSearchError, ValueError):
    """Invalid batch request (targets, catalogs, radius or strategy)."""


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchTarget:
    """One validated input target; ``radius_arcsec`` overrides the batch radius when set.

    ``coordinate_sigma_arcsec`` is the per-axis rounding error of the coordinates as typed
    (:func:`crossmatch.coordinate_sigma_arcsec`: 187.278 -> 1.04 arcsec), recorded by :func:`parse_targets`; None
    derives it from the float values. As in a single-object search it widens the target's position error.
    """

    id: str
    target: Target
    radius_arcsec: float | None = None
    coordinate_sigma_arcsec: float | None = None
    # How the input was interpreted where it was not taken as given (e.g. a negative parallax not used).
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ra": self.target.ra, "dec": self.target.dec, "epoch": self.target.epoch,
                "pm_ra_masyr": self.target.pm_ra_masyr, "pm_dec_masyr": self.target.pm_dec_masyr,
                "parallax_mas": self.target.parallax_mas, "radius_arcsec": self.radius_arcsec,
                "notes": list(self.notes)}


_TARGET_KEYS: dict[str, tuple[str, ...]] = {
    # 'source_id': Gaia archive exports (source_id, ra, dec, pmra, pmdec, parallax, ref_epoch).
    "id": ("id", "target_id", "name", "source_id", "objid", "obj_id", "source", "designation"),
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
    if isinstance(value, bool):  # float(True) == 1.0: a JSON true is not a number
        raise BatchError(f"{label}: {name} must be numeric (got {_short(value)}).")
    try:
        number = float(value)
    except OverflowError as exc:  # a JSON integer such as 10**400
        raise BatchError(f"{label}: {name} must be finite.") from exc
    except (TypeError, ValueError) as exc:
        raise BatchError(f"{label}: {name} must be numeric (got {_short(value)}).") from exc
    if not math.isfinite(number):
        raise BatchError(f"{label}: {name} must be finite.")
    return number


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):  # float(True) == 1.0: a JSON true is not a coordinate
        raise InvalidCoordinateError(f"{name} must be numeric.")
    try:
        number = float(value)
    except OverflowError as exc:  # a JSON integer such as 10**400
        raise InvalidCoordinateError(f"{name} must be finite.") from exc
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise InvalidCoordinateError(f"{name} must be finite.")
    return number


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."



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
        if isinstance(epoch, bool):
            raise InvalidCoordinateError("epoch must be numeric.")
        try:
            epoch = float(epoch)
        except OverflowError as exc:
            raise InvalidCoordinateError("epoch must be a finite Julian year between 1800 and 2200.") from exc
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


# Coordinates typed so coarsely that their rounding exceeds half the sky (e.g. '1e5' or '0e400', whose last
# digit is worth 10**400 degrees) do not locate a target.
_MAX_COORDINATE_SIGMA_ARCSEC = 180.0 * 3600.0


def _typed_precision(ra: Any, dec: Any, label: str) -> float:
    """:func:`crossmatch.coordinate_sigma_arcsec` of the coordinates as typed, as a BatchError when unusable."""
    try:
        sigma = coordinate_sigma_arcsec(ra, dec)
    except (OverflowError, ValueError) as exc:  # '0e400': 10.0 ** 400
        raise BatchError(f"{label}: coordinates as typed ({_short(ra, 40)}, {_short(dec, 40)}) have no usable "
                         f"precision ({exc}).") from exc
    if not math.isfinite(sigma) or sigma > _MAX_COORDINATE_SIGMA_ARCSEC:
        raise BatchError(f"{label}: coordinates as typed ({_short(ra, 40)}, {_short(dec, 40)}) are too coarse to "
                         "locate a target (their last digit is worth more than 180 degrees).")
    return sigma


class _TargetParser:
    """Incremental :func:`parse_targets`: ids stay unique and row numbers continue across :meth:`feed` calls, so a
    long list can be validated in slices (one CPU slot at a time, see :func:`_offload`)."""

    def __init__(self, max_targets: int | None = None) -> None:
        self.limit = max_targets if max_targets is not None else _env_int("BATCH_MAX_TARGETS", 100_000)
        self.targets: list[BatchTarget] = []
        self.seen: set[str] = set()
        self.index = 0

    def feed(self, items: Iterable[Any]) -> None:
        """Validate every item of ``items``."""
        for item in items:
            self.add(item)

    def finish(self) -> list[BatchTarget]:
        if not self.targets:
            raise BatchError("no targets given.")
        return self.targets

    def add(self, item: Any) -> None:
        """Validate one more item (the next row number)."""
        self.index += 1
        index = self.index
        if not isinstance(item, Mapping):
            raise BatchError(f"target {index}: expected an object with ra and dec, got {type(item).__name__}.")
        if len(self.targets) >= self.limit:
            raise BatchError(f"too many targets: at most {self.limit} per batch (BATCH_MAX_TARGETS).")
        lowered = {str(k).strip().lower().lstrip("﻿"): v for k, v in item.items()}
        raw_id = _pick(lowered, "id")
        target_id = " ".join(str(raw_id).split()) if raw_id is not None else str(index)
        label = f"target {target_id!r}"
        if target_id in self.seen:
            raise BatchError(f"{label}: duplicate id.")
        self.seen.add(target_id)
        ra, dec = _pick(lowered, "ra"), _pick(lowered, "dec")
        if ra is None or dec is None:
            raise BatchError(f"{label}: ra and dec (ICRS degrees) are required.")
        notes: list[str] = []
        parallax = _optional_float(_pick(lowered, "parallax_mas"), "parallax_mas", label)
        if parallax is not None and parallax < 0.0:
            # A measured parallax may be negative (Gaia DR3: faint stars, quasars): it is kept as a note, and
            # the target is not parallax-corrected (a negative distance has no displacement to apply).
            if parallax <= -1000.0:
                raise BatchError(f"{label}: parallax_mas must be in (-1000, 1000) mas.")
            notes.append(f"parallax {parallax:g} mas is negative (not significant): not used for the parallax "
                         "correction.")
            parallax = None
        try:
            target = icrs_target(
                ra, dec,
                epoch=_optional_float(_pick(lowered, "epoch"), "epoch", label),
                pm_ra_masyr=_optional_float(_pick(lowered, "pm_ra_masyr"), "pm_ra_masyr", label),
                pm_dec_masyr=_optional_float(_pick(lowered, "pm_dec_masyr"), "pm_dec_masyr", label),
                parallax_mas=parallax,
            )
        except InvalidCoordinateError as exc:
            raise BatchError(f"{label}: {exc}") from exc
        radius = _optional_float(_pick(lowered, "radius_arcsec"), "radius_arcsec", label)
        if radius is not None and not 0.0 < radius <= MAX_RADIUS_ARCSEC:
            raise BatchError(f"{label}: radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}].")
        self.targets.append(BatchTarget(target_id, target, radius, _typed_precision(ra, dec, label), tuple(notes)))


def parse_targets(items: Iterable[Mapping[str, Any]], *, max_targets: int | None = None) -> list[BatchTarget]:
    """Validate target mappings; RA is normalised to [0, 360) and ids must be unique.

    Keys are case-insensitive with common aliases (``ra``/``RAJ2000``, ``pmra``, ``name``/``source_id``,
    ``radius``, ...). A missing id becomes the 1-based row number. Proper motion (mas/yr, RA component including
    cos dec) needs both components. A negative parallax (a valid measurement, e.g. many Gaia DR3 sources) is not
    used for the parallax correction; the target's ``notes`` say so. Booleans are not numbers. ``items`` is
    consumed lazily: a list longer than ``max_targets`` fails at its first excess item.
    """
    parser = _TargetParser(max_targets)
    parser.feed(items)
    return parser.finish()


_CSV_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def _csv_lines(text: str) -> Iterable[str]:
    """Non-blank, non-comment lines of ``text`` (with their line ends), lazily: no list of every line and no
    StringIO copy (4 bytes per character) of a text that can be 50 MB."""
    for match in _CSV_LINE.finditer(text):
        line = match.group()
        if line.strip() and not line.lstrip().startswith("#"):
            yield line


def _csv_rows(text: str | bytes) -> Iterable[dict[str, Any]]:
    """Target mappings of CSV text, lazily; the header is checked before the first row is read.

    csv.Error while reading (malformed CSV) raises :class:`BatchError`.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", "replace")
    reader = csv.DictReader(_csv_lines(text.lstrip("﻿")))
    try:
        header = reader.fieldnames
    except csv.Error as exc:
        raise BatchError(f"malformed CSV header: {exc}") from exc
    if not header:
        raise BatchError("CSV contains no header row.")
    fields = {str(f).strip().lower() for f in header}
    if not fields & set(_TARGET_KEYS["ra"]) or not fields & set(_TARGET_KEYS["dec"]):
        raise BatchError(f"CSV header must contain ra and dec columns (got {sorted(fields)[:20]}).")

    def rows() -> Iterable[dict[str, Any]]:
        try:
            for row in reader:
                yield {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}
        except csv.Error as exc:
            raise BatchError(f"malformed CSV near data line {reader.line_num} (check for an unbalanced double "
                             f"quote): {exc}") from exc

    return rows()


def read_targets_csv(text: str | bytes, *, max_targets: int | None = None) -> list[BatchTarget]:
    """Targets from CSV text with a header row (``id,ra,dec[,epoch,pmra,pmdec,parallax,radius_arcsec]``).

    Lines starting with ``#`` are comments. Malformed CSV (e.g. an unbalanced double quote, which makes the
    rest of the file one field larger than ``csv.field_size_limit``) raises :class:`BatchError`. Rows are read
    lazily, so a list over ``max_targets`` is refused at its first excess row, without building every row.
    """
    return parse_targets(_csv_rows(text), max_targets=max_targets)


_JSON_WS = re.compile(r"[ \t\n\r]*")


class _TooManyTargets(Exception):
    pass


def _decode_targets_json(payload: str | bytes | bytearray, limit: int) -> Any:
    """``json.loads`` for a target document that stops at the first target over ``limit``.

    A bare array or the ``targets`` array of a top-level object is decoded element by element
    (``JSONDecoder.raw_decode``), so an oversized list costs at most ``limit`` decoded targets instead of the whole
    document (a 50 MB list of 1M targets would need over 1 GB as Python objects). Any other shape, and every
    malformed document, is decoded by ``json.loads`` itself (same result and error messages).
    """
    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode(json.detect_encoding(payload), "surrogatepass")
    else:
        text = payload
    if text.startswith("﻿"):
        return json.loads(text)  # json.loads' own error for a BOM
    decoder = json.JSONDecoder()

    def array(pos: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        pos = _JSON_WS.match(text, pos + 1).end()
        if text.startswith("]", pos):
            return items, pos + 1
        while True:
            if len(items) >= limit:
                raise _TooManyTargets
            value, pos = decoder.raw_decode(text, pos)
            items.append(value)
            pos = _JSON_WS.match(text, pos).end()
            if text.startswith(",", pos):
                pos = _JSON_WS.match(text, pos + 1).end()
            elif text.startswith("]", pos):
                return items, pos + 1
            else:
                raise ValueError("expected ',' or ']'")

    try:
        pos = _JSON_WS.match(text).end()
        if text.startswith("[", pos):
            data, pos = array(pos)
        elif text.startswith("{", pos):
            data = {}
            pos = _JSON_WS.match(text, pos + 1).end()
            while not text.startswith("}", pos):
                key, pos = decoder.raw_decode(text, pos)
                if not isinstance(key, str):  # invalid JSON: json.loads reports it
                    return json.loads(text)
                pos = _JSON_WS.match(text, pos).end()
                if not text.startswith(":", pos):
                    raise ValueError("expected ':'")
                pos = _JSON_WS.match(text, pos + 1).end()
                if key == "targets" and text.startswith("[", pos):
                    data[key], pos = array(pos)
                else:
                    data[key], pos = decoder.raw_decode(text, pos)
                pos = _JSON_WS.match(text, pos).end()
                if text.startswith(",", pos):
                    pos = _JSON_WS.match(text, pos + 1).end()
                    if text.startswith("}", pos):
                        raise ValueError("trailing comma")
                elif not text.startswith("}", pos):
                    raise ValueError("expected ',' or '}'")
            pos += 1
        else:
            return json.loads(text)
        if _JSON_WS.match(text, pos).end() != len(text):
            raise ValueError("extra data")
    except _TooManyTargets:
        raise BatchError(f"too many targets: at most {limit} per batch (BATCH_MAX_TARGETS).") from None
    except (ValueError, RecursionError):
        return json.loads(text)  # malformed: json.loads raises its own, exact error
    return data


def _json_target_items(payload: str | bytes | Any, limit: int) -> list[Any]:
    """The raw target items of a JSON array of objects or ``{"targets": [...]}`` (decoded up to ``limit`` + 1)."""
    try:
        data = (_decode_targets_json(payload, limit) if isinstance(payload, (str, bytes, bytearray))
                else payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BatchError(f"invalid JSON targets: {exc}") from exc
    except RecursionError as exc:  # e.g. '[' * 100000
        raise BatchError("invalid JSON targets: nested too deeply.") from exc
    if isinstance(data, Mapping):
        data = data.get("targets")
    if not isinstance(data, list):
        raise BatchError("JSON targets must be an array of {id, ra, dec} objects (or {\"targets\": [...]}).")
    return data


def read_targets_json(payload: str | bytes | Any, *, max_targets: int | None = None) -> list[BatchTarget]:
    """Targets from a JSON array of objects or ``{"targets": [...]}``; a list over ``max_targets`` is refused
    without decoding the rest of it."""
    limit = max_targets if max_targets is not None else _env_int("BATCH_MAX_TARGETS", 100_000)
    return parse_targets(_json_target_items(payload, limit), max_targets=limit)


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
    # Targets never sent upstream because the request cannot be served (a cone wider than the CDS XMatch limit
    # without a usable cone fallback): client-side refusals, not archive failures.
    refused_targets: int = 0
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
            "refused_targets": self.refused_targets, "elapsed_s": round(self.elapsed_s, 3), "errors": list(self.errors),
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
# Timeouts of a request the server received: the join may be too heavy for the service's execution limit, so a
# multi-target chunk is split. (ConnectTimeout / PoolTimeout mean the request never reached the server: they are
# connection problems, retried with backoff and never split.)
_TIMEOUT_EXCEPTIONS = (httpx.ReadTimeout, httpx.WriteTimeout)
_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError,
                     httpx.WriteError)
# Chunk failures after which the targets are cone-searched instead (transient or availability problems);
# CatalogQueryError (a deterministic HTTP 4xx / query error) would fail identically for every target.
_FALLBACK_ERRORS = (CatalogUnavailableError, QueryTimeoutError, ResponseParseError)
# Response headers that describe the wire encoding: dropped when a decoded body is re-wrapped.
_WIRE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

# Flat output (rows / csv / parquet): one row per (target, catalog, match) with status "match", plus one row per
# (target, catalog) whose query FAILED with status "failed" and the reason in "error" (every match column
# empty). A (target, catalog) pair without any row was queried successfully and has no counterpart -- so a flat
# file never shows a failed archive query as "no counterpart".
FLAT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("target_id", "string"), ("target_ra", "float64"), ("target_dec", "float64"), ("target_epoch", "float64"),
    ("catalog", "string"), ("strategy", "string"), ("status", "string"), ("rank", "int32"), ("source_id", "string"),
    ("ra", "float64"), ("dec", "float64"), ("separation_arcsec", "float64"),
    ("query_separation_arcsec", "float64"), ("epoch_propagation", "string"),
    ("positional_error_arcsec", "float64"), ("epoch", "float64"), ("epoch_range_start", "float64"),
    ("epoch_range_end", "float64"), ("pm_ra_masyr", "float64"), ("pm_dec_masyr", "float64"),
    ("confidence", "float64"), ("error", "string"), ("data_json", "string"),
)
ROW_STATUS_MATCH = "match"
ROW_STATUS_FAILED = "failed"


@dataclass(slots=True)
class BatchResult:
    """Per-target matches for every catalog plus the execution report.

    ``association[idx]`` holds the per-target association summary of
    :meth:`crossmatch.CrossmatchService.finalize` (``p_any``, the proper motion / parallax used and where they
    came from, adoption warnings) for targets with at least one catalog row. ``failures[idx][catalog]`` is the
    reason a target could not be matched against a catalog (the query failed: this is not "no counterpart").
    ``groups[idx]`` is the target's ``crossmatch_groups`` (every Bayesian object in its cone, as a single-object
    search returns them), kept only when the engine was built with ``keep_groups=True`` (dataset creation).
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
    groups: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    _positions: dict[str, int] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def request_count(self) -> int:
        return sum(run.requests for run in self.runs.values())

    @property
    def failed_target_count(self) -> int:
        """Targets for which at least one catalog query failed."""
        return sum(1 for per_catalog in self.failures.values() if per_catalog)

    def failures_by_id(self) -> dict[str, dict[str, str]]:
        """{target id: {catalog: reason}} for every failed (target, catalog) pair."""
        return {self.targets[idx].id: dict(per_catalog) for idx, per_catalog in sorted(self.failures.items())
                if per_catalog}

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
            "failed_targets": {name: run.failed_targets for name, run in self.runs.items()},
            "confidence": "posterior probability that the row is the target's counterpart (as a single-object "
                          "crossmatch over the same catalogs)",
        }

    def _index(self, target_id: str) -> int:
        """Position of a target id (ids are unique): one dict built on first use, O(1) per call."""
        if self._positions is None or len(self._positions) != len(self.targets):
            self._positions = {item.id: idx for idx, item in enumerate(self.targets)}
        return self._positions[target_id]

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
        """Flat dataset-like rows (:data:`FLAT_COLUMNS`): one per (target, catalog, match) with status "match",
        and one per failed (target, catalog) query with status "failed" and the reason in ``error``."""
        out: list[dict[str, Any]] = []
        empty = dict.fromkeys(name for name, _ in FLAT_COLUMNS)
        for idx, item in enumerate(self.targets):
            base = {"target_id": item.id, "target_ra": item.target.ra, "target_dec": item.target.dec,
                    "target_epoch": item.target.epoch}
            failed = self.failures.get(idx, {})
            for name in self.catalogs:
                if name in failed:
                    run = self.runs.get(name)
                    out.append({**empty, **base, "catalog": name, "strategy": run.strategy if run else None,
                                "status": ROW_STATUS_FAILED, "error": failed[name]})
                    continue
                for match in self.matches.get(idx, {}).get(name, []):
                    span = match.get("epoch_range") or (None, None)
                    out.append({
                        **base, "catalog": name, "strategy": match["strategy"], "status": ROW_STATUS_MATCH,
                        "rank": match["rank"], "source_id": match["source_id"], "ra": match["ra"], "dec": match["dec"],
                        "separation_arcsec": match["separation_arcsec"],
                        "query_separation_arcsec": match["query_separation_arcsec"],
                        "epoch_propagation": match["epoch_propagation"],
                        "positional_error_arcsec": match["positional_error_arcsec"], "epoch": match["epoch"],
                        "epoch_range_start": span[0], "epoch_range_end": span[1],
                        "pm_ra_masyr": match["pm_ra_masyr"], "pm_dec_masyr": match["pm_dec_masyr"],
                        "confidence": match["confidence"], "error": None,
                        "data_json": json.dumps(match.get("data"), default=str) if include_data else None,
                    })
        return out

    def to_arrow(self, *, include_data: bool = True):
        """The flat rows as a ``pyarrow.Table``; the summary, the per-catalog report and the failed (target,
        catalog) pairs (``astrosearch.batch.failures``: {target id: {catalog: reason}}) are in the schema
        metadata."""
        import pyarrow as pa

        schema = pa.schema([pa.field(name, getattr(pa, kind)()) for name, kind in FLAT_COLUMNS])
        table = pa.Table.from_pylist(self.rows(include_data=include_data), schema=schema)
        meta = {
            b"astrosearch.batch.summary": json.dumps(self.summary()).encode(),
            b"astrosearch.batch.catalogs": json.dumps({n: r.as_dict() for n, r in self.runs.items()}, default=str).encode(),
            b"astrosearch.batch.failures": json.dumps(self.failures_by_id(), default=str).encode(),
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
    parts = [(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
        f'<RESOURCE type="results"><TABLE name="{UPLOAD_TABLE}">\n'
        '<FIELD name="t_idx" datatype="int"/>\n'
        '<FIELD name="t_ra" datatype="double" unit="deg"/>\n'
        '<FIELD name="t_dec" datatype="double" unit="deg"/>\n'
        '<FIELD name="t_rad" datatype="double" unit="deg"/>\n'
        "<DATA><TABLEDATA>\n"
    )]
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


_TAPVIZIER_HOSTS = ("tapvizier.cds.unistra.fr", "tapvizier.u-strasbg.fr")


def vizier_table_of(catalog: CatalogDefinition) -> str | None:
    """The VizieR table id (``J/ApJ/914/42/table5``) of a registry catalog served by TAPVizieR -- an embedded
    survey such as ``vlass``/``lotss`` or a table registered with ``vizier add`` -- else None."""
    if catalog.provider != "tap" or not catalog.endpoint or not catalog.table:
        return None
    host = urlsplit(str(catalog.endpoint)).hostname or ""
    if host.lower() not in _TAPVIZIER_HOSTS:
        return None
    table = str(catalog.table).strip().strip('"').strip()
    return table or None


_QUOTED_NAME = re.compile(r'^"((?:[^"]|"")+)"$')
_SELF_ALIAS = re.compile(r'^"((?:[^"]|"")+)"\s+AS\s+"((?:[^"]|"")+)"$', re.IGNORECASE)


def _vizier_column(item: Any) -> str | None:
    """The VizieR column name of a registry select-list item: ``Name``, ``"Name"`` or ``"z.obs" AS "z.obs"``
    (how vizier.py selects names with a dot). A computed item (``"Epoch" + 1990 AS "ep"``) is None."""
    text = str(item or "").strip()
    if not text:
        return None
    match = _SELF_ALIAS.match(text)
    if match:
        return match.group(1).replace('""', '"') if match.group(1) == match.group(2) else None
    match = _QUOTED_NAME.match(text)
    if match:
        return match.group(1).replace('""', '"')
    return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.+-]*", text) else None


def registry_vizier_view(catalog: CatalogDefinition) -> XMatchView | None:
    """The CDS XMatch view of a TAPVizieR registry catalog (see :func:`vizier_table_of`), or None.

    CDS XMatch serves every VizieR table under the same column names as TAPVizieR, and answers a batch join in
    seconds, where TAPVizieR table uploads can stall for minutes (seen live: VLASS uploads timing out after
    2 x 300 s while XMatch returned the same rows in 3.8 s). The view is the registry entry itself: its rows are
    normalised by the entry (``base_catalog``: column mapping, epoch and epoch format, positional errors,
    exclusions), ``cols2`` asks for the columns the entry reads, and the association uses the entry's own priors.

    None when the entry reads a column computed in its ADQL (``vizier add`` derives e.g. Tycho-2 epochs stored
    as offsets, or larger-side errors): CDS XMatch returns table columns only, so such a table stays on the
    TAPVizieR strategies.
    """
    table = vizier_table_of(catalog)
    if table is None:
        return None
    params = catalog.parameters or {}
    items = [item for item in params.get("columns") or [] if isinstance(item, str)]
    names = [_vizier_column(item) for item in items]
    if any(name is None for name in names):
        return None
    needed: list[Any] = [params.get("id_field"), params.get("ra_field"), params.get("dec_field")]
    pos_cols = (catalog.pos_error or {}).get("columns")
    if isinstance(pos_cols, list):
        needed.extend(pos_cols)
    if isinstance(catalog.epoch, str):
        needed.append(catalog.epoch)
    needed.extend((params.get("exclude_values") or {}).keys())
    field_map = params.get("field_map")
    if isinstance(field_map, Mapping):
        needed.extend(field_map.values())
    needed_names = [_vizier_column(item) for item in needed if item]
    if any(name is None for name in needed_names):
        return None
    columns: tuple[str, ...] = ()
    if names:  # no column list: XMatch's default column set (every column of the table)
        columns = tuple(dict.fromkeys([*(n for n in names if n), *(n for n in needed_names if n)]))
    return XMatchView(
        f"vizier:{table}", columns, epoch=catalog.epoch, epoch_format=catalog.epoch_format,
        note=f"VizieR {table} via CDS XMatch (registry entry {catalog.name})",
        base_catalog=catalog.name, density_key=catalog.name,
    )


def is_registry_view(name: str, registry: CatalogRegistry) -> bool:
    """True when ``name``'s XMatch view is derived from its registry entry (:func:`registry_vizier_view`), not a
    predefined, verified one: whether XMatch serves the table with the entry's columns is checked at run time."""
    return not name.startswith("vizier:") and _xmatch_view(name) is None and catalog_xmatch_view(name, registry) is not None


# {vizier table: (checked at (monotonic s), XMatch column names or None when the service does not serve it)}.
_XMATCH_COLUMNS: dict[str, tuple[float, frozenset[str] | None]] = {}
XMATCH_COLUMNS_TTL_SECONDS = 86400.0


async def xmatch_table_columns(client: httpx.AsyncClient, vizier_table: str, *,
                               timeout: float = 30.0) -> frozenset[str] | None:
    """The columns CDS XMatch serves for ``vizier_table`` ('vizier:J/ApJ/914/42/table5'), or None when it does not
    serve the table (HTTP 400: 'Table ... not in the service', seen live for LoTSS-DR3, J/A+A/707/A198). XMatch
    indexes some tables with its own column set (2MASS: errHalfMaj/errHalfMin/errPosAng and MeasureJD, where
    TAPVizieR has errMaj/errMin/errPA and JD). Cached per process; network errors propagate (httpx.HTTPError)."""
    now = time.monotonic()
    cached = _XMATCH_COLUMNS.get(vizier_table)
    if cached is not None and now - cached[0] < XMATCH_COLUMNS_TTL_SECONDS:
        return cached[1]
    response = await client.get(XMATCH_TABLES_ENDPOINT, params={"action": "getColList", "tabName": vizier_table,
                                                                "RESPONSEFORMAT": "json"}, timeout=timeout)
    if response.status_code == 400:
        columns: frozenset[str] | None = None
    else:
        response.raise_for_status()
        try:
            meta = response.json().get("metadata") or []
        except (ValueError, AttributeError) as exc:
            raise httpx.DecodingError(f"XMatch column list of {vizier_table} is not JSON: {exc}") from exc
        columns = frozenset(str(c.get("name")) for c in meta if isinstance(c, Mapping) and c.get("name"))
    _XMATCH_COLUMNS[vizier_table] = (now, columns)
    return columns


def known_unusable_view(view: XMatchView) -> bool:
    """True when this process has seen CDS XMatch refuse the view's table or lack columns it reads (see
    :func:`xmatch_table_columns`); unknown (not checked yet) is False."""
    cached = _XMATCH_COLUMNS.get(view.vizier_table)
    if cached is None or time.monotonic() - cached[0] >= XMATCH_COLUMNS_TTL_SECONDS:
        return False
    served = cached[1]
    return served is None or any(c not in served for c in view.columns)


def catalog_xmatch_view(name: str, registry: CatalogRegistry) -> XMatchView | None:
    """The CDS XMatch view of ``name``: a predefined view (:data:`XMATCH_VIEWS`, :data:`VIZIER_VIEWS`), else that
    of a TAPVizieR registry catalog (:func:`registry_vizier_view`), else None."""
    view = _xmatch_view(name)
    if view is None and name in registry.catalogs:
        view = registry_vizier_view(registry.get(name))
    return view


def xmatch_catalog_definition(name: str, registry: CatalogRegistry) -> tuple[CatalogDefinition, XMatchView | None]:
    """CatalogDefinition used to normalise XMatch rows of ``name`` (a registry catalog or ``vizier:<table>``).

    ``vizier:I/355/gaiadr3`` is the table behind the ``gaia_dr3`` view and is normalised identically (epoch
    2016.0, Gaia archive column names); the other :data:`VIZIER_VIEWS` carry their own conventions; any other
    ``vizier:<table>`` is described by the UCDs of the XMatch answer.
    """
    view = catalog_xmatch_view(name, registry)
    if view is not None and view.base_catalog:
        base = registry.get(view.base_catalog)
        acknowledgement = base.acknowledgement or ""
        for ack in (_VIZIER_ACK, _XMATCH_ACK):
            if ack not in acknowledgement:
                acknowledgement = f"{acknowledgement} {ack}".strip()
        definition = replace(
            base,
            name=name,
            endpoint=XMATCH_ENDPOINT,
            table=view.vizier_table,
            epoch=view.epoch,
            epoch_format=view.epoch_format,
            citation=(base.citation or "") + "; VizieR " + view.vizier_table.split(":", 1)[1] + " via CDS XMatch",
            acknowledgement=acknowledgement,
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
    """A catalog without a positional-error specification gets XMatch's standardised 1-sigma error ellipse, or
    for tables without one (UCAC4, URAT1) XMatch's circular per-coordinate ``radec_err``.

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
    if major is not None:
        cols = [major, minor or major] + ([angle] if angle else [])
        return replace(catalog, pos_error={"columns": cols, "kind": "ellipse"})
    # No ellipse: XMatch's circular per-coordinate error (UCAC4 and URAT1 'radec_err', stat.error;pos.eq, in
    # arcsec; it comes before any table column with the same UCD, e.g. URAT1 'sigm' in mas), else a per-axis
    # stat.error;pos.eq.ra / pos.eq.dec pair. Units come from the column metadata.
    by_ucd: dict[str, str] = {}
    for col in columns:
        by_ucd.setdefault(_normalized_ucd(col.ucd), col.name)
    if "stat.error;pos.eq" in by_ucd:
        return replace(catalog, pos_error={"columns": [by_ucd["stat.error;pos.eq"]], "kind": "sigma"})
    if "stat.error;pos.eq.ra" in by_ucd and "stat.error;pos.eq.dec" in by_ucd:
        return replace(catalog, pos_error={"columns": [by_ucd["stat.error;pos.eq.ra"],
                                                       by_ucd["stat.error;pos.eq.dec"]], "kind": "sigma"})
    return catalog


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


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _normalized_ucd(ucd: str | None) -> str:
    return ";".join(part.strip().lower() for part in str(ucd or "").split(";") if part.strip())


def _j2000_named(name: str) -> bool:
    """A position column named for epoch/equinox J2000 (``RAJ2000``, ``RA2000``, ``ra_epoch2000``, ``_RAJ2000``)."""
    return name.lower().endswith("2000")


def _identifier_like(name: str) -> bool:
    """A column name that reads as an identifier (``source_id``, ``ID``, ``objID``, ``Name``), not a count such as
    Tycho-2's ``Num``."""
    low = name.lower()
    return low in {"source", "name", "designation"} or low.endswith(("id", "name", "designation"))


def _unique_identifier(columns: Sequence[ColumnMeta], rows: Sequence[Mapping[str, Any]],
                       position: Callable[[Mapping[str, Any]], tuple[float, float] | None]) -> ColumnMeta | None:
    """A bare ``meta.id`` column that identifies the rows of this answer: an identifier-like name, a value in
    every row, and one value per catalogue source (the same source matched to two targets repeats its id at the
    same position; one value at two positions -- Tycho-2 ``Num`` = 7 for 4 stars -- is not an identifier)."""
    for col in columns:
        if _normalized_ucd(col.ucd) != "meta.id" or not _identifier_like(col.name):
            continue
        where: dict[str, tuple[float, float] | None] = {}
        usable = True
        for row in rows:
            value = row.get(col.name)
            if value is None or str(value).strip() == "":
                usable = False
                break
            key = " ".join(str(value).split())
            pos = position(row)
            pos = (round(pos[0], 9), round(pos[1], 9)) if pos is not None else None
            if key in where and where[key] != pos:
                usable = False
                break
            where[key] = pos
        if usable:
            return col
    return None


def describe_generic_answer(
    catalog: CatalogDefinition, columns: list[ColumnMeta], grouped: Mapping[int, list[dict[str, Any]]],
) -> tuple[CatalogDefinition, list[ColumnMeta]]:
    """Definition for the rows of a generic ``vizier:<table>`` XMatch answer (a table without an
    :class:`XMatchView`), from what the answer itself says.

    * Position: VizieR's computed ``_RAJ2000``/``_DEJ2000`` when the answer has them -- the positions CDS XMatch
      matched on (see :data:`XMATCH_J2000_COLUMNS`) -- else the ``meta.main`` UCD columns, which XMatch matches
      on. For Hipparcos the ``meta.main`` columns are the J1991.25 positions, 90" from Barnard's star's J2000
      position.
    * Epoch of a table with proper motions (every row gets an explicit epoch, so a missing one stays unknown
      instead of being read from a ``time.epoch`` or ``Epoch`` column):

      - a row with a proper motion whose position is at J2000 is dated 2000.0: ``_RAJ2000`` (VizieR propagated
        it with the row's proper motion) or a ``meta.main`` position named for J2000 (``RAJ2000``, ``RA2000``,
        ``ra_epoch2000``: VizieR adds a computed ``_RAJ2000`` only when the table's own position is not at
        J2000). UCAC4 I/322A: "The epoch for the positions of all stars is J2000.0" (ReadMe note 2) while
        ``EpRA`` (time.epoch) is the central epoch of the mean RA; USNO-B1.0, PPMXL and NOMAD ``RAJ2000`` are
        at Ep=J2000 with a mean observation epoch column; URAT1 ``RA2000`` is at J2000 while ``RAdeg`` is at
        ``Epoch``; Gaia DR2/EDR3 ``ra_epoch2000``. A table's mean observation epoch never re-dates such a
        position (it put UCAC4 Barnard's star at 1991.25: posterior 0.007 instead of 0.995);
      - a row with a proper motion whose position is not at J2000 takes the table's reference epoch
        (``meta.ref;time.epoch``) when it has one, else its epoch is unknown (a ``time.epoch`` column is a mean
        epoch of observation, not the epoch of a propagated position);
      - a row without a proper motion keeps its ``time.epoch`` column (the epoch of its unpropagated position,
        e.g. the Tycho-2 mean epoch of a row without a mean position), else its epoch is unknown.

      A table without proper motions keeps its observation epochs (``time.epoch``: 2MASS, SDSS).
    * Identifier: the ``meta.id;meta.main`` column; else the ``meta.id.part;meta.main`` columns joined with
      '-' (Tycho-2 TYC1-TYC2-TYC3 -> '3694-1921-1'); else a bare ``meta.id`` column with an identifier's name and
      one value per source (Gaia DR2/EDR3 ``source_id``: see :func:`_unique_identifier`; Tycho-2's ``Num`` is the
      number of observations: 4 different stars in one field were all labelled '7'); else a position
      designation ``J<ra>+<dec>``.
    * Proper-motion units VizieR mislabels (``ma/yr``) are corrected.

    Rows of ``grouped`` are updated in place with :data:`GENERIC_ID_COLUMN` / :data:`GENERIC_EPOCH_COLUMN`.
    """
    for col in columns:
        unit = "".join(str(col.unit or "").split())  # the VOTable parser writes 'ma / yr'
        if unit in _UNIT_TYPOS and _normalized_ucd(col.ucd).startswith("pos.pm"):
            col.unit = _UNIT_TYPOS[unit]
    names = {c.name for c in columns}
    ucd = ucd_field_map(columns)
    params = dict(catalog.parameters)
    field_map = {str(k): str(v) for k, v in (params.get("field_map") or {}).items()}
    if all(name in names for name in XMATCH_J2000_COLUMNS):
        params["ra_field"], params["dec_field"] = XMATCH_J2000_COLUMNS
        position_j2000 = True
    else:
        ra_col, dec_col = ucd.get("ra"), ucd.get("dec")
        position_j2000 = (ra_col is not None and dec_col is not None
                          and _j2000_named(ra_col.name) and _j2000_named(dec_col.name))
    pm_cols = (ucd.get("pmra"), ucd.get("pmdec"))
    has_pm = pm_cols[0] is not None and pm_cols[1] is not None
    if has_pm:
        field_map["pmra"], field_map["pmdec"] = pm_cols[0].name, pm_cols[1].name  # type: ignore[union-attr]
    params["field_map"] = field_map
    extra: list[ColumnMeta] = []
    epoch: Any = catalog.epoch
    epoch_format = catalog.epoch_format
    rows = [row for group in grouped.values() for row in group]

    main_id = next((c for c in columns if _normalized_ucd(c.ucd) == "meta.id;meta.main"), None)
    parts = [c for c in columns if {"meta.id.part", "meta.main"} <= set(_normalized_ucd(c.ucd).split(";"))]
    if main_id is not None:
        params["id_field"] = main_id.name
    else:
        position = _position_reader(replace(catalog, parameters=params), columns)
        bare_id = None if parts else _unique_identifier(columns, rows, position)
        if bare_id is not None:
            params["id_field"] = bare_id.name
        else:
            for row in rows:
                values = [row.get(c.name) for c in parts]
                if parts and all(v is not None and str(v).strip() != "" for v in values):
                    row[GENERIC_ID_COLUMN] = "-".join(_id_part(v) for v in values)
                else:
                    pos = position(row)
                    row[GENERIC_ID_COLUMN] = f"J{pos[0]:010.6f}{pos[1]:+010.6f}" if pos is not None else None
            params["id_field"] = GENERIC_ID_COLUMN
            extra.append(ColumnMeta(GENERIC_ID_COLUMN, None, None, "char",
                                    "identifier composed by batch.describe_generic_answer"))

    if has_pm:
        epoch_col = ucd.get("epoch")
        reference_epoch = epoch_col is not None and "meta.ref" in _normalized_ucd(epoch_col.ucd).split(";")
        epochs: set[float] = set()
        for row in rows:
            moving = (_float_or_none(row.get(pm_cols[0].name)) is not None  # type: ignore[union-attr]
                      and _float_or_none(row.get(pm_cols[1].name)) is not None)  # type: ignore[union-attr]
            value: float | None
            if moving and position_j2000:
                value = XMATCH_J2000_EPOCH
            elif epoch_col is not None and (reference_epoch or not moving):
                value = epoch_to_jyear(row.get(epoch_col.name), epoch_col.unit)
            else:
                value = None
            row[GENERIC_EPOCH_COLUMN] = value
            if value is not None:
                epochs.add(value)
        # A lookup (models.resolve_epoch): a row whose value is missing stays unknown -- a plain column name
        # would fall back to the row's time.epoch / 'Epoch' column.
        epoch, epoch_format = {"column": GENERIC_EPOCH_COLUMN, "values": {str(v): v for v in sorted(epochs)}}, None
        extra.append(ColumnMeta(GENERIC_EPOCH_COLUMN, "yr", None, "double",
                                "position epoch: J2000.0 for rows at J2000 positions with a proper motion"))
    definition = replace(catalog, parameters=params, epoch=epoch, epoch_format=epoch_format)
    return definition, columns + extra


def _id_part(value: Any) -> str:
    number = _float_or_none(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return " ".join(str(value).split())


def _planning_catalog(catalog: CatalogDefinition, strategy: str, view: XMatchView | None) -> CatalogDefinition:
    """The definition cones are planned with (:func:`models.plan_cone`).

    CDS XMatch matches a generic ``vizier:<table>`` (no :class:`XMatchView`) on J2000 positions: VizieR's
    computed ``_RAJ2000`` (Hipparcos, Tycho-2: propagated to J2000.0 with each row's proper motion) or the table's
    own J2000 ``meta.main`` columns (UCAC4, USNO-B1.0, NOMAD, PPMXL ``RAJ2000``; URAT1 ``RA2000``; Gaia DR2/EDR3
    ``ra_epoch2000``); see :func:`describe_generic_answer`. The epoch that matters is therefore known before the
    request: a dated target with a proper motion is looked for at its J2000 position (HD 189733 at epoch 2016 had
    no Hipparcos or Tycho-2 row at 3" when the cone stayed at the 2016 position), and a dated target without one
    gets the cone widened for the gap to J2000, exactly as for a registry catalog at a fixed epoch.
    """
    if strategy == "xmatch" and view is None and catalog.epoch is None:
        return replace(catalog, epoch=XMATCH_J2000_EPOCH, epoch_format=None)
    return catalog


# Distinct warnings kept per catalog run (a 100k-target batch of distinct epochs has one cone warning per target).
_MAX_RUN_WARNINGS = 1000


def _collect_warnings(run: CatalogRun, results: Iterable[QueryResult]) -> None:
    """Add the distinct warnings of ``results`` to ``run`` (each once; at most :data:`_MAX_RUN_WARNINGS`)."""
    seen = set(run.warnings)
    omitted: set[str] = set()
    for result in results:
        for warning in result.meta.get("warnings") or ():
            text = str(warning)
            if text in seen or text in omitted:
                continue
            if len(seen) >= _MAX_RUN_WARNINGS:
                omitted.add(text)
                continue
            seen.add(text)
            run.warnings.append(text)
    if omitted:
        run.warnings.append(f"{run.catalog}: {len(omitted)} more distinct per-target warning(s) not listed.")


def _area_chunks(indices: Sequence[int], plans: Mapping[int, ConePlan], size: int,
                 reference_arcsec: float = XMATCH_REFERENCE_RADIUS_ARCSEC) -> list[list[int]]:
    """Consecutive chunks whose summed cone area is at most ``size`` reference cones.

    Expected rows per request grow with sum(radius^2): a target counts (max(r, reference) / reference)^2, so
    chunks of cones up to the reference radius hold ``size`` targets and wide cones proportionally fewer (a 60"
    chunk holds size / 36 targets) instead of overflowing MAXREC and being downloaded and discarded.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    load = 0.0
    for i in indices:
        cost = (max(plans[i].radius_arcsec, reference_arcsec) / reference_arcsec) ** 2
        if current and load + cost > size:
            chunks.append(current)
            current, load = [], 0.0
        current.append(i)
        load += cost
    if current:
        chunks.append(current)
    return chunks


_OVERFLOW_STATUS = re.compile(rb'name\s*=\s*"QUERY_STATUS"\s+value\s*=\s*"OVERFLOW"'
                              rb'|value\s*=\s*"OVERFLOW"\s+name\s*=\s*"QUERY_STATUS"')


_FIELD_TAG = re.compile(rb"<FIELD\b[^>]*>")
_CHAR_TYPE = re.compile(rb'\bdatatype\s*=\s*"(?:char|unicodeChar)"')
_FIXED_ARRAYSIZE = re.compile(rb'\barraysize\s*=\s*"[0-9]+\*?"')


def _relax_char_arraysize(content: bytes) -> bytes:
    """A TABLEDATA VOTable with its fixed-length string fields made variable-length (``arraysize="*"``).

    CDS XMatch declares ``<FIELD name="USNO-B1.0" datatype="char" arraysize="12">`` (NOMAD1.0 alike) and then
    writes 13-character values ('0920-00258572'); astropy truncates them to the declared length without a
    warning, so different USNO-B1.0 stars shared the id '0920-0025857' -- which is another, real star 150 deg
    away. TABLEDATA cells carry their own length, so the declared size is not needed to read them. BINARY
    serialisations (which need the fixed size to decode) and answers without such fields are returned as they
    are (the same object).
    """
    start = content.find(b"<TABLEDATA")
    if start < 0:
        return content
    header = content[:start]

    def relax(match: re.Match[bytes]) -> bytes:
        tag = match.group()
        if _CHAR_TYPE.search(tag) is None:
            return tag
        return _FIXED_ARRAYSIZE.sub(b'arraysize="*"', tag)

    relaxed = _FIELD_TAG.sub(relax, header)
    return content if relaxed == header else b"".join((relaxed, memoryview(content)[start:]))


def _declares_overflow(content: bytes) -> bool:
    """A VOTable answer whose QUERY_STATUS is OVERFLOW (truncated): detected before the costly parse."""
    return b"OVERFLOW" in content and _OVERFLOW_STATUS.search(content) is not None


# One CPU-bound batch step at a time per event loop (all engines and requests): parsing, conversion, cone planning
# and association hold the GIL, and several such worker threads at once starve the event loop of it (0.4 s
# trivial callbacks, multi-second stalls with 6 catalogs converting in parallel). With one worker the loop gets
# the GIL at every switch interval (5 ms).
#
# The work of a batch is handed to that worker in short time slices (:data:`CPU_SLICE_SECONDS`, see
# :func:`_offload_steps`), and a free slot goes to the waiting request that has used the least CPU so far (fair
# queuing over :class:`_CpuAccount`, one per request): a 1-target request waited 50.7 s behind the single
# 36-s association call of a 20,000-target batch, and a batch whose client had left (cancelled) held the slot
# until that call ended. Now a request waits for at most one slice per step, and a cancelled batch stops after
# the slice that is running.
CPU_SLICE_SECONDS = _env_float("BATCH_CPU_SLICE_SECONDS", 0.1)


class _CpuAccount:
    """CPU time used by one request (shared by all the tasks of its batch through a context variable)."""

    __slots__ = ("used",)

    def __init__(self) -> None:
        self.used = 0.0


_CPU_ACCOUNT: contextvars.ContextVar[_CpuAccount | None] = contextvars.ContextVar("batch_cpu_account", default=None)


def _cpu_account() -> _CpuAccount:
    """The current request's account (a new one, bound to the current context, when there is none)."""
    account = _CPU_ACCOUNT.get()
    if account is None:
        account = _CpuAccount()
        _CPU_ACCOUNT.set(account)
    return account


class _CpuSlot:
    """A one-holder lock whose waiters are served least-CPU-used first (then first come, first served)."""

    def __init__(self) -> None:
        self.busy = False
        self.waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self.sequence = 0

    async def acquire(self, account: _CpuAccount) -> None:
        if not self.busy and not self.waiters:
            self.busy = True
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sequence += 1
        heapq.heappush(self.waiters, (account.used, self.sequence, future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # granted and cancelled at once: pass the slot on
            raise

    def release(self) -> None:
        while self.waiters:
            _used, _seq, future = heapq.heappop(self.waiters)
            if not future.done():
                future.set_result(None)  # the slot passes to this waiter (busy stays True)
                return
        self.busy = False


_CPU_SLOTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _CpuSlot] = weakref.WeakKeyDictionary()


async def _offload(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` in a worker thread, at most one per event loop (see :data:`_CPU_SLOTS`).

    The slot is released when the thread finishes, not when the caller is cancelled (a worker thread cannot
    be interrupted), so cancellation never lets a second CPU-bound thread start alongside a running one. The
    thread's run time is charged to the caller's :class:`_CpuAccount`.
    """
    loop = asyncio.get_running_loop()
    slot = _CPU_SLOTS.get(loop)
    if slot is None:
        slot = _CPU_SLOTS[loop] = _CpuSlot()
    account = _cpu_account()
    await slot.acquire(account)
    started = time.perf_counter()
    try:
        call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
        future = loop.run_in_executor(None, call)
    except BaseException:
        slot.release()
        raise

    def settle(done: asyncio.Future[Any]) -> None:
        account.used += time.perf_counter() - started
        slot.release()
        if not done.cancelled():
            done.exception()  # retrieved: the caller may have been cancelled meanwhile

    future.add_done_callback(settle)
    return await asyncio.shield(future)


def _run_steps(step: Callable[[Any], None], iterator: Iterator[Any], budget: float) -> bool:
    """Apply ``step`` to the next items of ``iterator`` for about ``budget`` seconds; False once exhausted."""
    deadline = time.perf_counter() + budget
    for item in iterator:
        step(item)
        if time.perf_counter() >= deadline:
            return True
    return False


async def _offload_steps(step: Callable[[Any], None], items: Iterable[Any]) -> None:
    """``step(item)`` for every item, in the worker thread, one :func:`_offload` call per time slice of
    :data:`CPU_SLICE_SECONDS`: the slot changes hands between concurrent requests at every slice, and a
    cancelled caller stops after the running slice. Items are processed exactly once, in order, whatever the
    slicing (results never depend on timing)."""
    iterator = iter(items)
    while await _offload(_run_steps, step, iterator, CPU_SLICE_SECONDS):
        pass


async def _parse_in_slices(items: Iterable[Any], max_targets: int | None = None) -> list[BatchTarget]:
    """:func:`parse_targets` in time slices (``items`` may be a lazy iterator, e.g. CSV rows)."""
    parser = _TargetParser(max_targets)

    await _offload_steps(parser.add, items)
    return parser.finish()


class BatchCrossmatcher:
    """Crossmatch many targets against many catalogs with one request per catalog chunk.

    Parameters are optional: without ``client`` a client is created for each :meth:`run`; without
    ``registry`` the embedded catalog registry is used. ``guards`` (endpoint pacers/circuit breakers) and
    ``cache`` may be shared with the API's providers so batch cone searches and ordinary searches of the same
    endpoint are paced together; ``upload_slots`` (per-endpoint concurrency limits) may be shared between
    instances. ``fallback_to_cone`` sends a chunk whose upload/XMatch request keeps failing for a transient
    reason through per-target cone searches instead. ``association_config`` is the Bayesian association
    configuration (default that of :class:`crossmatch.CrossmatchService`). ``max_catalogs`` limits the catalogs
    of one run (``BATCH_MAX_CATALOGS``, default 20). ``fast_fallback_targets`` / ``fast_fallback_seconds``
    (``BATCH_FAST_FALLBACK_TARGETS`` / ``_SECONDS``, default 100 / 45 s; 0 targets disables it): a batch this small
    whose upload/XMatch join does not answer in time goes to cone searches at once (see :meth:`_post`).
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
        max_catalogs: int | None = None,
        keep_groups: bool = False,
        fast_fallback_targets: int | None = None,
        fast_fallback_seconds: float | None = None,
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
        # A small batch (at most BATCH_FAST_FALLBACK_TARGETS targets) whose upload/XMatch join does not answer
        # within BATCH_FAST_FALLBACK_SECONDS goes to per-target cone searches at once: a healthy service answers
        # such a join in seconds, while a stalled one (TAPVizieR uploads, seen live) would otherwise cost
        # 2 x BATCH_UPLOAD_TIMEOUT_SECONDS per split level before the cones run.
        self.fast_fallback_targets = (int(fast_fallback_targets) if fast_fallback_targets is not None
                                      else max(0, _env_int_or_zero("BATCH_FAST_FALLBACK_TARGETS", 100)))
        self.fast_fallback_seconds = (float(fast_fallback_seconds) if fast_fallback_seconds is not None
                                      else _env_float("BATCH_FAST_FALLBACK_SECONDS", 45.0))
        self.chunk_sizes = dict(chunk_sizes or {})
        self.max_cone_targets = max_cone_targets or _env_int("BATCH_MAX_CONE_TARGETS", 5000)
        self.fallback_to_cone = fallback_to_cone
        self.attempts = max(1, int(attempts))
        self.catalog_budget_seconds = catalog_budget_seconds or _env_float("BATCH_CATALOG_BUDGET_SECONDS", 1800.0)
        self.max_response_bytes = max_response_bytes or MAX_RESPONSE_BYTES
        self.association_config = association_config or AssociationConfig()
        self.max_catalogs = max_catalogs or _env_int("BATCH_MAX_CATALOGS", DEFAULT_MAX_CATALOGS)
        # {event loop: {endpoint: Semaphore}} -- asyncio primitives belong to one loop.
        self.upload_slots = upload_slots if upload_slots is not None else weakref.WeakKeyDictionary()
        # Keep every target's crossmatch_groups in BatchResult.groups (datasets.DatasetEngine uses them).
        self.keep_groups = bool(keep_groups)

    # -- planning -------------------------------------------------------------

    def default_catalogs(self) -> list[str]:
        """Enabled catalogs that need no per-target requests (upload or xmatch strategy)."""
        return [name for name in self.registry.enabled_catalogs() if self.default_strategy(name) != "cone"]

    def xmatch_view(self, name: str) -> XMatchView | None:
        """The CDS XMatch view of ``name`` (:func:`catalog_xmatch_view`), or None."""
        return catalog_xmatch_view(name, self.registry)

    def default_strategy(self, name: str) -> str:
        """``xmatch`` for VizieR tables (``vizier:<table>``, and registry catalogs served by TAPVizieR: CDS XMatch
        answers in seconds, TAPVizieR uploads may stall for minutes) and the predefined views; ``upload`` for the
        other TAP archives that accept uploads (TAPVizieR uploads stay available as an explicit strategy);
        ``cone`` otherwise."""
        view = self.xmatch_view(name)
        if name.startswith("vizier:") or (view is not None and not known_unusable_view(view)):
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
                view = self.xmatch_view(name)
                assert view is not None
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
        if len(names) > self.max_catalogs:
            raise BatchError(f"too many catalogs: {len(names)} requested, at most {self.max_catalogs} per batch "
                             "(BATCH_MAX_CATALOGS).")
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
            if strategy == "xmatch" and not (name.startswith("vizier:") or self.xmatch_view(name) is not None):
                raise BatchError(f"{name}: no CDS XMatch (VizieR) view is defined (only VizieR tables and "
                                 f"{', '.join(sorted(XMATCH_VIEWS))} can be matched with CDS XMatch).")
            if strategy == "cone" and name.startswith("vizier:"):
                raise BatchError(f"{name}: VizieR tables are matched with the xmatch strategy only.")
            plan.append((name, strategy))
        return plan

    def _check_limits(self, plan: Sequence[tuple[str, str]], batch: Sequence[BatchTarget]) -> None:
        """Client-side limits known before any request (so an impossible batch is a 422, not a 502)."""
        for name, strategy in plan:
            if strategy == "cone" and len(batch) > self.max_cone_targets:
                raise BatchError(f"{name}: {len(batch)} targets need per-target cone searches; at most "
                                 f"{self.max_cone_targets} are allowed per batch (BATCH_MAX_CONE_TARGETS). Use an "
                                 "upload/xmatch catalog or split the target list.")

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

        CPU-bound work (target validation, cone planning, response parsing, row conversion and the per-target
        association) runs in worker threads, one at a time (:func:`_offload`) and in short time slices
        (:func:`_offload_steps`), so an event loop serving other requests is never blocked for long, concurrent
        requests take turns (least CPU used first), and a cancelled batch stops within one slice.
        """
        radius = float(radius_arcsec)
        if not math.isfinite(radius) or not 0.0 < radius <= MAX_RADIUS_ARCSEC:
            raise BatchError(f"radius_arcsec must be in (0, {MAX_RADIUS_ARCSEC:g}].")
        # One CPU account for the whole batch (bound before the per-catalog tasks copy the context), shared with
        # the request's body parsing when the router already opened one.
        _cpu_account()
        plan = self._resolve(catalogs, strategies)
        items = list(targets)
        if items and all(isinstance(t, BatchTarget) for t in items):
            batch: list[BatchTarget] = items  # type: ignore[assignment]
        else:
            batch = await _parse_in_slices(items)
        self._check_limits(plan, batch)
        started = time.perf_counter()
        owned = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.upload_timeout, follow_redirects=True)
        try:
            chosen = set(strategies or {})
            outcomes = await asyncio.gather(*(self._run_catalog(client, name, strategy, batch, radius,
                                                                explicit=name in chosen)
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
        names = [name for name, _ in plan]
        matches: dict[int, dict[str, list[dict[str, Any]]]] = {}
        association: dict[int, dict[str, Any]] = {}
        groups: dict[int, list[dict[str, Any]]] = {}

        service = CrossmatchService(self.registry, {}, association_config=self._view_config())

        def associate(idx: int) -> None:
            found, summary = self._associate(batch, names, results, runs, radius, nearest_only, (idx,), service,
                                             groups=groups if self.keep_groups else None)
            matches.update(found)
            association.update(summary)

        await _offload_steps(associate, range(len(batch)))
        return BatchResult(batch, [name for name, _ in plan], radius, runs, matches, dict(failures),
                           time.perf_counter() - started, nearest_only, association, groups)

    async def _registry_view_problem(self, client: httpx.AsyncClient, name: str) -> str | None:
        """Why CDS XMatch cannot serve the registry catalog ``name`` through its view, or None when it can: the
        table must be in the service and offer every column the registry entry reads."""
        view = self.xmatch_view(name)
        assert view is not None
        try:
            served = await xmatch_table_columns(client, view.vizier_table)
        except httpx.HTTPError as exc:
            return None if isinstance(exc, httpx.TransportError) else f"its table list could not be read ({exc})"
        if served is None:
            return f"CDS XMatch does not serve {view.vizier_table}"
        missing = [c for c in view.columns if c not in served]
        if missing:
            return f"CDS XMatch serves {view.vizier_table} without the column(s) {', '.join(missing)} the registry reads"
        return None

    async def _run_catalog(
        self, client: httpx.AsyncClient, name: str, strategy: str, batch: Sequence[BatchTarget], radius: float,
        explicit: bool = False,
    ) -> tuple[CatalogRun, dict[int, QueryResult], dict[int, str]]:
        started = time.perf_counter()
        deadline = time.monotonic() + self.catalog_budget_seconds
        switched: str | None = None
        if strategy == "xmatch" and is_registry_view(name, self.registry):
            problem = await self._registry_view_problem(client, name)
            if problem is not None:
                catalog = self.registry.get(name)
                if explicit:
                    run = CatalogRun(name, strategy, XMATCH_ENDPOINT, targets=len(batch), citation=catalog.citation,
                                     acknowledgement=catalog.acknowledgement)
                    message = f"{name}: the xmatch strategy was requested, but {problem}."
                    run.errors.append(message)
                    run.failed_targets = len(batch)
                    run.elapsed_s = time.perf_counter() - started
                    return run, {}, {i: message for i in range(len(batch))}
                strategy = "upload" if catalog.provider == "tap" and catalog.endpoint in UPLOAD_SERVICES else "cone"
                switched = f"{name}: {problem}; matched with the {strategy} strategy instead."
        if strategy == "xmatch":
            catalog, view = xmatch_catalog_definition(name, self.registry)
            endpoint: str | None = XMATCH_ENDPOINT
        else:
            catalog, view = self.registry.get(name), None
            endpoint = UPLOAD_SERVICES[str(catalog.endpoint)].upload_endpoint if strategy == "upload" else catalog.endpoint
        run = CatalogRun(name, strategy, endpoint, targets=len(batch), citation=catalog.citation,
                         acknowledgement=catalog.acknowledgement)
        if switched:
            run.warnings.append(switched)

        def count() -> None:
            run.requests += 1

        counting = _CountingClient(client, count)
        results: dict[int, QueryResult] = {}
        failures: dict[int, str] = {}
        retry: dict[int, str] = {}
        indices = list(range(len(batch)))
        # VizieR tables have no registry cone search to fall back to.
        can_fall_back = self.fallback_to_cone and strategy != "cone" and not name.startswith("vizier:")
        quick = can_fall_back and len(batch) <= min(self.fast_fallback_targets, self.max_cone_targets)
        try:
            if strategy == "cone":
                results, failures = await self._run_cone(counting, catalog, batch, indices, radius, run, deadline)
            else:
                # Generic VizieR tables are matched by CDS XMatch at their J2000 positions (see
                # _planning_catalog): the cones are planned for that epoch, not left unpropagated.
                planning = _planning_catalog(catalog, strategy, view)
                plans: dict[int, ConePlan] = {}

                def plan(i: int) -> None:
                    plans[i] = plan_cone(planning, batch[i].target, batch[i].radius_arcsec or radius)

                await _offload_steps(plan, indices)
                if strategy == "upload":
                    results, retry, failures = await self._run_upload(counting, catalog, batch, indices, plans, run,
                                                                      deadline, quick=quick)
                else:
                    results, retry, failures = await self._run_xmatch(counting, catalog, view, batch, indices, plans,
                                                                       run, deadline, can_fall_back, planning,
                                                                       quick=quick)
            if retry:
                if not can_fall_back:
                    failures.update(retry)
                elif len(retry) > self.max_cone_targets:
                    message = (f"{len(retry)} targets would need per-target cone searches; at most "
                               f"{self.max_cone_targets} are allowed per batch (BATCH_MAX_CONE_TARGETS).")
                    run.errors.append(message)
                    failures.update({idx: f"{reason}; not cone-searched: {message}" for idx, reason in retry.items()})
                else:
                    run.fallback_targets += len(retry)
                    cone_results, cone_failures = await self._run_cone(counting, self.registry.get(name), batch,
                                                                       list(retry), radius, run, deadline)
                    results.update(cone_results)
                    failures.update(cone_failures)
        except BatchError as exc:
            run.errors.append(str(exc))
            failures.update({idx: str(exc) for idx in indices if idx not in results})
        if results:
            await _offload(_collect_warnings, run, list(results.values()))
        run.failed_targets = len(failures)
        run.elapsed_s = time.perf_counter() - started
        return run, results, failures

    # -- association ----------------------------------------------------------------

    def _view_config(self) -> AssociationConfig:
        """The association configuration with the survey resolutions (and per-catalog completeness, when set)
        of :func:`_view_aliases` applied to the view names."""
        config = self.association_config
        aliases = _view_aliases()
        resolution = dict(config.resolution_arcsec)
        for name, key in aliases.items():
            if key in resolution:
                resolution.setdefault(name, resolution[key])
        completeness = config.completeness
        if isinstance(completeness, Mapping):
            completeness = dict(completeness)
            for name, key in aliases.items():
                if key in completeness:
                    completeness.setdefault(name, completeness[key])
        return replace(config, resolution_arcsec=resolution, completeness=completeness)

    def _associate(
        self, batch: Sequence[BatchTarget], catalogs: Sequence[str], results: Mapping[str, Mapping[int, QueryResult]],
        runs: Mapping[str, CatalogRun], radius: float, nearest_only: bool, indices: Iterable[int] | None = None,
        service: CrossmatchService | None = None, groups: dict[int, list[dict[str, Any]]] | None = None,
    ) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[int, dict[str, Any]]]:
        """Per target (``indices``, default all): :meth:`crossmatch.CrossmatchService.finalize` over the rows of
        every catalog.

        This is the single-object pipeline on the same rows: proper-motion/parallax adoption, final in-radius
        split and the Bayesian association whose target posterior becomes each match's ``confidence``.
        Targets without a single fetched row are skipped (finalize would return no match for them).
        """
        if service is None:
            service = CrossmatchService(self.registry, {}, association_config=self._view_config())
        floor = float(self.association_config.target_sigma_arcsec)
        matches: dict[int, dict[str, list[dict[str, Any]]]] = {}
        association: dict[int, dict[str, Any]] = {}
        for idx in range(len(batch)) if indices is None else indices:
            item = batch[idx]
            successes = [(name, results[name][idx]) for name in catalogs if idx in results[name]]
            if not any(_has_rows(result) for _, result in successes):
                continue
            requested = item.radius_arcsec or radius
            plans = [QueryPlan(name, "batch", None, {}, requested, "unknown") for name, _ in successes]
            # The target's position error as CrossmatchService.prepare sets it: the configured floor,
            # widened by the rounding of the coordinates as typed when that is not negligible.
            quantum = item.coordinate_sigma_arcsec
            if quantum is None:
                quantum = coordinate_sigma_arcsec(item.target.ra, item.target.dec)
            precise = quantum <= 0.1 * floor
            sigma = floor if precise else math.hypot(floor, quantum)
            ctx = SearchContext(target=item.target, plans=plans, search_radius=requested, query=None, profile=None,
                                pm_source=None, target_sigma_arcsec=sigma, target_pm_sigma_masyr=None,
                                target_sigma_source="default" if precise else "coordinate_precision")
            record = service.finalize(ctx, successes, [])
            if groups is not None:
                groups[idx] = record.crossmatch_groups
            prov = record.provenance
            info = prov.get("association") or {}
            association[idx] = {
                "p_any": info.get("p_any"),
                "best_match_probability": info.get("best_match_probability"),
                "target_proper_motion": prov.get("target_proper_motion"),
                "target_parallax": prov.get("target_parallax"),
                "target_sigma_arcsec": sigma,
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
                    splittable: bool, quick: bool = False) -> httpx.Response:
        """POST multipart/form-data with pacing, a circuit breaker and retries on transient failures.

        Connection problems (resets, refused or timed-out connections) and HTTP 408/429/5xx are retried
        ``attempts`` times with exponential backoff (Retry-After honoured up to 30 s) and count as endpoint
        failures for the circuit breaker. CDS XMatch resets roughly one connection in six under load (observed
        live 2026-09-28 with curl and httpx alike), so a few retries are routine. A read/write timeout (the
        server has the request) is retried once; a second one splits a multi-target chunk (_TooSlow) instead of
        re-sending the same heavy join (a single target then falls back to a cone search). A pool timeout (no
        connection of the local client free) is retried but gives no verdict on the endpoint. Every attempt
        reports to the circuit breaker at most once, cancellation included; a timed-out multi-target join is not
        reported as an endpoint failure (it is too heavy, and is split), unless it was the half-open recovery
        probe.

        With ``quick`` (a small batch that can fall back to cone searches, see ``fast_fallback_targets``) each
        attempt waits at most ``fast_fallback_seconds``; the first timeout raises QueryTimeoutError at once (the
        targets go to cone searches) and counts as an endpoint failure, so while the service is down later
        batches meet the open circuit and go straight to cones. Retries of transient errors stop once twice
        ``fast_fallback_seconds`` have passed (then QueryTimeoutError: cone searches).
        """
        guard = self._guard(endpoint)
        last_error = ""
        timeouts = 0
        attempts = self.attempts
        quick_until = time.monotonic() + 2.0 * self.fast_fallback_seconds if quick else math.inf
        for attempt in range(attempts):
            if attempt:
                if time.monotonic() >= quick_until:
                    raise QueryTimeoutError(f"{endpoint}: still failing after {2.0 * self.fast_fallback_seconds:g} s "
                                            f"({last_error}); a small batch goes to per-target cone searches instead")
                run.retries += 1
                run.warnings.append(f"{run.catalog}: retried after {last_error}")
                logger.info("%s: retry %d after %s", run.catalog, attempt, last_error)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueryTimeoutError(f"{endpoint}: catalog time budget exhausted ({last_error or 'no answer'})")
            delay = self.retry_backoff_seconds * 2**attempt
            async with self._slot(endpoint):
                await guard.acquire()
                # acquire() marks the half-open recovery probe; a probe must always settle the breaker.
                is_probe = guard.probe_in_flight
                wait = min(self.upload_timeout, remaining)
                if quick:
                    wait = max(1.0, min(wait, self.fast_fallback_seconds, quick_until - time.monotonic()))
                try:
                    response = await self._read(client, endpoint, data, files, wait)
                except _ResponseTooLarge as exc:
                    guard.record_success()  # the endpoint answered
                    raise _Overflow(str(exc)) from exc
                except _TIMEOUT_EXCEPTIONS as exc:
                    # A multi-target join that times out is too heavy for the service's execution limit (e.g.
                    # IRSA TAP/sync's 5 minutes), not evidence that the endpoint is down: it is split, and
                    # counting it would open the circuit that the smaller halves need. Single-target timeouts
                    # (and a timed-out recovery probe) do count.
                    if is_probe or not splittable or quick:
                        guard.record_failure()
                    timeouts += 1
                    last_error = f"{exc.__class__.__name__}: {exc}".rstrip(": ")
                    if quick:
                        raise QueryTimeoutError(f"{endpoint}: no answer within {wait:g} s ({last_error}); a small "
                                                "batch goes to per-target cone searches instead") from exc
                    if timeouts >= 2:
                        if splittable:
                            raise _TooSlow(f"{endpoint}: timed out twice ({last_error})") from exc
                        raise QueryTimeoutError(f"{endpoint}: timed out twice ({last_error})") from exc
                except httpx.PoolTimeout as exc:
                    # No connection of the (shared) client became free: nothing was sent to the endpoint.
                    if is_probe:
                        guard.record_failure()
                    last_error = f"{exc.__class__.__name__}: {exc}".rstrip(": ")
                except _RETRY_EXCEPTIONS as exc:
                    guard.record_failure()
                    last_error = f"{exc.__class__.__name__}: {exc}".rstrip(": ")
                except httpx.HTTPError as exc:
                    guard.record_failure()
                    raise CatalogUnavailableError(f"{endpoint}: {exc.__class__.__name__}: {exc}") from exc
                except BaseException:
                    # Cancellation (client disconnect, time budget) or a programming error: never leave a
                    # half-open probe dangling. An ordinary request gave no verdict on the endpoint, so it is
                    # not counted (repeatedly cancelled batches must not open the circuit of a healthy archive).
                    if is_probe:
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
            if attempt + 1 < attempts:
                await asyncio.sleep(max(0.0, min(delay, deadline - time.monotonic())))
        if "Timeout" in last_error:
            raise QueryTimeoutError(f"{endpoint}: {last_error} after {attempts} attempt(s)")
        raise CatalogUnavailableError(f"{endpoint}: {last_error} after {attempts} attempt(s)")

    @staticmethod
    def _parse(response: httpx.Response, label: str, expected: str | None) -> ParsedTable:
        TapProvider._check(response, label)
        content = response.content
        relaxed = _relax_char_arraysize(content)
        if relaxed is not content:
            response = httpx.Response(response.status_code, headers=[
                (k, v) for k, v in response.headers.multi_items() if k.lower() not in _WIRE_HEADERS
            ], content=relaxed, request=response.request)
        return TapProvider._parse_table(response, label, expected)

    # -- chunk scheduling -------------------------------------------------------

    async def _chunked(
        self,
        chunks: list[list[int]],
        send: Callable[[list[int]], Any],
        run: CatalogRun,
        deadline: float,
    ) -> tuple[list[tuple[list[int], Any]], dict[int, str], dict[int, str]]:
        """Send chunks (bounded concurrency), splitting overflowing / too slow chunks.

        Returns (answers, {target: reason} to retry by cone search, {target: reason} final failures). Each
        target's reason is the error of its own chunk. Transient/availability failures are retried by cone
        search; a deterministic query error (HTTP 4xx) or an exhausted time budget is final.
        """
        answers: list[tuple[list[int], Any]] = []
        retry: dict[int, str] = {}
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
                        message = f"single-target request failed: {exc}"
                        run.errors.append(message)
                        retry.update({i: message for i in chunk})
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
                    retry.update({i: f"{exc.__class__.__name__}: {exc}" for i in chunk})

        while queue:
            pending, queue[:] = list(queue), []
            await asyncio.gather(*(one(chunk) for chunk in pending))
        return answers, retry, final

    # -- strategies ---------------------------------------------------------------

    async def _run_upload(
        self, client: _CountingClient, catalog: CatalogDefinition, batch: Sequence[BatchTarget], indices: list[int],
        plans: Mapping[int, ConePlan], run: CatalogRun, deadline: float, *, quick: bool = False,
    ) -> tuple[dict[int, QueryResult], dict[int, str], dict[int, str]]:
        service = UPLOAD_SERVICES[str(catalog.endpoint)]
        adql = build_upload_adql(catalog, service)
        run.queries.append(adql)
        fmt = str(catalog.parameters.get("format", "json"))
        size = self._chunk_size(service.key, service.chunk_size)
        if service.upload_row_limit:
            size = min(size, service.upload_row_limit)
        converter = _RowConverter(client, "tap")
        label = f"TAP upload {catalog.name}"
        parameters = {"QUERY": adql, "UPLOAD": UPLOAD_TABLE}

        def parse(chunk: list[int], response: httpx.Response) -> tuple[int, dict[int, list[dict[str, Any]]], list[ColumnMeta]]:
            if response.status_code < 400 and _declares_overflow(response.content):
                raise _Overflow(f"QUERY_STATUS OVERFLOW (MAXREC {service.max_rows})")
            table = TapProvider._apply_column_units(catalog, self._parse(response, label, fmt))
            if table.truncated or len(table.rows) >= service.max_rows:
                raise _Overflow(f"{len(table.rows)} rows (MAXREC {service.max_rows}, status {table.query_status})")
            grouped, columns = _group_rows(table, UPLOAD_INDEX, frozenset(UPLOAD_COLUMNS), chunk, label)
            return len(table.rows), grouped, columns

        async def send(chunk: list[int]) -> tuple[int, dict[int, QueryResult], dict[int, str]]:
            rows = [(i, plans[i].ra, plans[i].dec, plans[i].radius_arcsec / 3600.0) for i in chunk]
            form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql,
                    "UPLOAD": f"{UPLOAD_TABLE},param:{UPLOAD_TABLE}", "MAXREC": str(service.max_rows)}
            if fmt in {"json", "csv"}:
                form["FORMAT"] = fmt
            files = {UPLOAD_TABLE: (f"{UPLOAD_TABLE}.xml", upload_votable(rows), "application/x-votable+xml")}
            response = await self._post(client, service.upload_endpoint, form, files, run, deadline=deadline,
                                        splittable=len(chunk) > 1, quick=quick)
            n_rows, grouped, columns = await _offload(parse, chunk, response)
            converted, failed = await self._convert_in_slices(converter, catalog, batch, chunk, plans, grouped, columns,
                                                              service.upload_endpoint, parameters, "upload", run)
            return n_rows, converted, failed

        return await self._collect(_area_chunks(indices, plans, size), send, run, deadline)

    async def _convert_in_slices(
        self, converter: _RowConverter, catalog: CatalogDefinition, batch: Sequence[BatchTarget], chunk: list[int],
        plans: Mapping[int, ConePlan], grouped: Mapping[int, list[dict[str, Any]]], columns: list[ColumnMeta],
        endpoint: str, parameters: dict[str, Any], strategy: str, run: CatalogRun,
    ) -> tuple[dict[int, QueryResult], dict[int, str]]:
        """:meth:`_convert_chunk` target by target in CPU time slices (:func:`_offload_steps`): a 20,000-target
        XMatch chunk is no longer converted in one worker call."""
        results: dict[int, QueryResult] = {}
        failed: dict[int, str] = {}
        column_meta = [c.as_dict() for c in columns]

        def convert(i: int) -> None:
            converted, bad = self._convert_chunk(converter, catalog, batch, (i,), plans, grouped, columns, endpoint,
                                                 parameters, strategy, run, column_meta)
            results.update(converted)
            failed.update(bad)

        await _offload_steps(convert, chunk)
        return results, failed

    async def _collect(self, chunks: list[list[int]], send: Callable[[list[int]], Any], run: CatalogRun,
                       deadline: float) -> tuple[dict[int, QueryResult], dict[int, str], dict[int, str]]:
        answers, retry, final = await self._chunked(chunks, send, run, deadline)
        results: dict[int, QueryResult] = {}
        for _chunk, (n_rows, converted, failed) in answers:
            run.rows_returned += n_rows
            results.update(converted)
            final.update(failed)
        return results, retry, final

    def _too_wide(
        self, catalog: CatalogDefinition, batch: Sequence[BatchTarget], too_wide: list[int],
        plans: dict[int, ConePlan], run: CatalogRun, can_fall_back: bool,
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Targets whose epoch-aware cone is wider than the CDS XMatch limit (180"): (cone-search, refused).

        * With a cone fallback (a registry catalog, ``fallback_to_cone``, at most ``BATCH_MAX_CONE_TARGETS``
          targets) they are cone-searched: the per-target cone keeps the full epoch widening.
        * Otherwise a cone widened for an unknown proper motion (``unknown_pm_pad``) is capped at 180" -- the
          same trade-off :func:`models.plan_cone` makes at ``EPOCH_PAD_MAX_ARCSEC`` -- with a warning naming
          the speed beyond which a star may be missed.
        * A cone that must hold the target's own proper-motion track (``target_pm``) cannot be capped without
          losing the target: that target is refused with the reason (not sent upstream).

        ``catalog`` is the definition the cones were planned with (:func:`_planning_catalog`). Runs in a worker
        thread (:meth:`_xmatch_layout`): ~9 us per target.
        """
        limit = XMATCH_MAX_DISTANCE_ARCSEC
        retry: dict[int, str] = {}
        refused: dict[int, str] = {}

        def reason(i: int) -> str:
            return (f"cone of {plans[i].radius_arcsec:.1f} arcsec (mode {plans[i].mode}) exceeds the CDS XMatch "
                    f"limit of {limit:g} arcsec")

        if can_fall_back and len(too_wide) <= self.max_cone_targets:
            retry = {i: reason(i) for i in too_wide}
            run.warnings.append(f"{catalog.name}: {len(too_wide)} target cone(s) wider than the XMatch limit of "
                                f"{limit:g} arcsec (epoch widening) were sent to per-target cone searches.")
            return retry, refused
        span = catalog_epoch_range(catalog)
        max_pm = epoch_pad_settings()[0]
        capped: list[int] = []
        rest: list[int] = []
        for i in too_wide:
            plan = plans[i]
            epoch = batch[i].target.epoch
            if plan.mode == "unknown_pm_pad" and span is not None and epoch is not None:
                gap = max(abs(span[0] - epoch), abs(span[1] - epoch))
                # The uncapped need (plan_cone may already have capped the pad at EPOCH_PAD_MAX_ARCSEC).
                needed = max(plan.radius_arcsec, plan.requested_radius_arcsec + max_pm * gap)
                plan.radius_arcsec = limit
                plan.pad_arcsec = max(0.0, limit - plan.requested_radius_arcsec)
                plan.warnings = [w for w in plan.warnings if "epoch gap" not in w]
                plan.warnings.append(
                    f"{catalog.name}: epoch gap {gap:.1f} yr needs a {needed:.0f} arcsec cone; capped at the CDS "
                    f"XMatch limit of {limit:g} arcsec (stars faster than {plan.pad_arcsec / gap:.2f} arcsec/yr may "
                    "be missed). Supply the target proper motion."
                )
                capped.append(i)
            else:
                rest.append(i)
        if capped:
            why = ("VizieR tables have no cone fallback" if catalog.name.startswith("vizier:")
                   else "the cone fallback is disabled" if not self.fallback_to_cone
                   else f"more than BATCH_MAX_CONE_TARGETS={self.max_cone_targets} would need cone searches")
            run.warnings.append(f"{catalog.name}: {len(capped)} epoch-widened target cone(s) capped at the XMatch "
                                f"limit of {limit:g} arcsec ({why}); stars moving faster may be missed.")
        if rest:
            if can_fall_back and len(rest) <= self.max_cone_targets:
                retry = {i: reason(i) for i in rest}
                run.warnings.append(f"{catalog.name}: {len(rest)} proper-motion target cone(s) wider than the XMatch "
                                    f"limit of {limit:g} arcsec were sent to per-target cone searches.")
            else:
                why = ("VizieR tables cannot be cone-searched in a batch; use the registry catalog of this survey"
                       if catalog.name.startswith("vizier:")
                       else "the cone fallback is disabled" if not self.fallback_to_cone
                       else f"more than BATCH_MAX_CONE_TARGETS={self.max_cone_targets} would need cone searches")
                refused = {i: f"{reason(i)}: not queried ({why})" for i in rest}
                run.refused_targets += len(refused)
                run.errors.append(f"{len(refused)} target(s) not queried: {refused[rest[0]]}")
        return retry, refused

    def _xmatch_layout(
        self, catalog: CatalogDefinition, batch: Sequence[BatchTarget], indices: list[int], plans: dict[int, ConePlan],
        run: CatalogRun, can_fall_back: bool, size: int,
    ) -> tuple[dict[int, str], dict[int, str], dict[int, float], list[list[int]]]:
        """(cone-search, refused, {target: XMatch radius}, chunks): the too-wide split (:meth:`_too_wide`) and the
        radius buckets of an XMatch run -- per-target work (sorting, warnings) done in a worker thread (0.86 s on
        the event loop for 100,000 too-wide targets before)."""
        too_wide = [i for i in indices if plans[i].radius_arcsec > XMATCH_MAX_DISTANCE_ARCSEC]
        retry_wide, refused = self._too_wide(catalog, batch, too_wide, plans, run, can_fall_back) if too_wide else ({}, {})
        skip = set(retry_wide) | set(refused)
        usable = [i for i in indices if i not in skip]
        radii = {i: math.ceil(plans[i].radius_arcsec * 1000.0) / 1000.0 for i in usable}
        chunks: list[list[int]] = []
        for bucket in radius_buckets(usable, radii):
            widest = max(radii[i] for i in bucket)
            # Expected rows per request ~ targets x radius^2: shrink chunks of wide cones accordingly.
            scaled = max(1, min(size, int(size * (XMATCH_REFERENCE_RADIUS_ARCSEC / max(widest, XMATCH_REFERENCE_RADIUS_ARCSEC)) ** 2)))
            chunks.extend(bucket[k:k + scaled] for k in range(0, len(bucket), scaled))
        return retry_wide, refused, radii, chunks

    async def _run_xmatch(
        self, client: _CountingClient, catalog: CatalogDefinition, view: XMatchView | None,
        batch: Sequence[BatchTarget], indices: list[int], plans: dict[int, ConePlan], run: CatalogRun,
        deadline: float, can_fall_back: bool, planning: CatalogDefinition | None = None, *, quick: bool = False,
    ) -> tuple[dict[int, QueryResult], dict[int, str], dict[int, str]]:
        size = self._chunk_size("xmatch", view.chunk_size if view else 20_000)
        retry_wide, refused, radii, chunks = await _offload(
            self._xmatch_layout, planning or catalog, batch, indices, plans, run, can_fall_back, size)
        cols2 = ",".join(view.columns) if view and view.columns else None
        table_name = catalog.table or ""
        converter = _RowConverter(client, "cds_xmatch")
        label = f"CDS XMatch {catalog.name}"

        def parse(chunk: list[int], response: httpx.Response) -> tuple[int, CatalogDefinition, dict[int, list[dict[str, Any]]], list[ColumnMeta]]:
            if response.status_code < 400 and _declares_overflow(response.content):
                raise _Overflow(f"QUERY_STATUS OVERFLOW (MAXREC {XMATCH_MAXREC})")
            table = self._parse(response, label, "votable")
            if table.truncated or len(table.rows) >= XMATCH_MAXREC:
                raise _Overflow(f"{len(table.rows)} rows (MAXREC {XMATCH_MAXREC}, status {table.query_status})")
            if view is not None and view.renames:
                _rename_columns(table, view.renames)
            grouped, columns = _group_rows(table, UPLOAD_INDEX, XMATCH_ECHO_COLUMNS, chunk, label)
            dist_name = next((c.name for c in columns if c.name.lower() == "angdist"), None)
            columns = [c for c in columns if c.name.lower() != "angdist"]
            definition = _with_ellipse_errors(catalog, columns)
            if view is None:
                definition, columns = describe_generic_answer(definition, columns, grouped)
            position = _position_reader(definition, columns)
            inside: dict[int, list[dict[str, Any]]] = {}
            for i in chunk:
                plan = plans[i]
                limit = plan.radius_arcsec * (1.0 + 1e-9) + 1e-6
                kept = []
                for row in grouped.get(i, []):
                    # XMatch used the bucket's widest radius: cut back to this target's own cone, with the
                    # distance XMatch itself measured (to the positions it matched on).
                    dist = _float_or_none(row.pop(dist_name, None)) if dist_name else None
                    if dist is None:
                        pos = position(row)
                        dist = haversine_arcsec(plan.ra, plan.dec, pos[0], pos[1]) if pos is not None else None
                    if dist is None or dist <= limit:
                        kept.append(row)
                inside[i] = kept
            return len(table.rows), definition, inside, columns

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
                                        splittable=len(chunk) > 1, quick=quick)
            n_rows, definition, inside, columns = await _offload(parse, chunk, response)
            converted, failed = await self._convert_in_slices(converter, definition, batch, chunk, plans, inside,
                                                              columns, XMATCH_ENDPOINT, {"cat2": table_name}, "xmatch",
                                                              run)
            return n_rows, converted, failed

        run.queries.append(f"CDS XMatch cat2={table_name} selection=all" + (f" cols2={cols2}" if cols2 else ""))
        results, retry, final = await self._collect(chunks, send, run, deadline)
        final.update(refused)
        return results, {**retry, **retry_wide}, final

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
        self, converter: _RowConverter, catalog: CatalogDefinition, batch: Sequence[BatchTarget], chunk: Sequence[int],
        plans: Mapping[int, ConePlan], grouped: Mapping[int, list[dict[str, Any]]], columns: list[ColumnMeta],
        endpoint: str, parameters: dict[str, Any], strategy: str, run: CatalogRun,
        column_meta: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[int, QueryResult], dict[int, str]]:
        """Rows of every target of a chunk -> QueryResults through the providers' row pipeline.

        A target whose rows cannot be converted (no usable position in any row) is a failure, not "no match".
        The column metadata is stored once per chunk (shared by the targets' results).
        """
        if column_meta is None:
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
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

ROUTER_PREFIX = "/api/v1/batch"
router = APIRouter(prefix=ROUTER_PREFIX, tags=["batch"])


def owns_request_body_limit(path: str) -> bool:
    """True for the batch routes, which read their bodies as a stream and refuse one over
    ``BATCH_MAX_UPLOAD_BYTES`` (50 MB) themselves (:func:`_read_body`).

    An application-wide body-size check (api.py's ``request_metrics`` middleware refuses bodies over
    ``MAX_REQUEST_BYTES``, 1 MB, after buffering them) must let these paths through unread: with it, any CSV
    list above ~37,000 targets (JSON above ~10-20,000) got HTTP 413 although the documented limits are 100,000
    targets and 50 MB. api.py::

        if batch.owns_request_body_limit(request.url.path):
            response = await call_next(request)   # batch enforces BATCH_MAX_UPLOAD_BYTES while streaming
        elif (length and int(length) > maximum) or len(await request.body()) > maximum: ...
    """
    return path == ROUTER_PREFIX or path.startswith(ROUTER_PREFIX + "/")


OUTPUT_FORMATS = ("json", "rows", "parquet", "csv")
# Status returned (to nobody) when the client disconnected and the batch was cancelled (nginx convention).
CLIENT_CLOSED_REQUEST = 499


class BatchOptions(BaseModel):
    """The options of a JSON body (everything but ``targets``)."""

    catalogs: list[str] | None = None
    radius_arcsec: float = Field(default=3.0, gt=0.0, le=MAX_RADIUS_ARCSEC)
    strategies: dict[str, str] | None = None
    nearest_only: bool = False
    include_data: bool = True
    format: str = Field(default="json", pattern="^(json|rows|parquet|csv)$")


class BatchRequest(BatchOptions):
    """JSON body. ``targets`` are validated by :func:`parse_targets` (aliases such as ``name``, ``source_id``,
    ``pmra``, ``pmdec``, ``plx``, ``radius``, ``RAJ2000`` are accepted exactly as in CSV and file uploads).
    The handler validates the options with :class:`BatchOptions` and the targets with :func:`parse_targets`
    only (no second, pydantic copy of up to 100,000 target objects)."""

    targets: list[dict[str, Any]] = Field(..., min_length=1)


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


def _first_non_finite(value: Any) -> str | None:
    """The first NaN / Infinity (also an overflowing literal such as 1e400) of a decoded JSON
    body, as text, or None. Python's json module accepts them; RFC 8259 JSON has none."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            if not math.isfinite(item):
                return str(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _json_job(body: bytes, defaults: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """A JSON body (object with ``targets`` or a bare array) -> (job options, raw target items); runs in a worker
    thread. The target array is decoded at most up to BATCH_MAX_TARGETS + 1 items (:func:`_decode_targets_json`)
    and never copied through pydantic; the items are validated by :func:`parse_targets` (in slices)."""
    try:
        payload = _decode_targets_json(body or b"null", _env_int("BATCH_MAX_TARGETS", 100_000))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}") from exc
    except RecursionError as exc:  # e.g. '[' * 100000
        raise HTTPException(status_code=422, detail="invalid JSON body: nested too deeply.") from exc
    token = _first_non_finite(payload)
    if token is not None:  # the same answer as every other route (api.non_finite_json)
        raise HTTPException(status_code=422, detail=[{
            "type": "json_invalid", "loc": ["body"], "input": token,
            "msg": f"JSON numbers must be finite (RFC 8259): {token} is not allowed"}])
    if isinstance(payload, list):
        payload = {"targets": payload}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object with 'targets'.")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise HTTPException(status_code=422, detail="JSON body needs a non-empty 'targets' array of {id, ra, dec} "
                                                    "objects.")
    merged = {**{k: v for k, v in defaults.items() if v is not None},
              **{k: v for k, v in payload.items() if k != "targets"}}
    try:
        model = BatchOptions.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        errors = getattr(exc, "errors", None)
        detail = _finite_json(json.loads(json.dumps(errors(), default=str))) if callable(errors) else str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc
    job = dict(defaults)
    job.update(model.model_dump())
    return job, targets


_LEADING_BLANKS = re.compile(rb"(?:\xef\xbb\xbf)?[ \t\r\n]*")


def _looks_like_json(content: bytes) -> bool:
    """The first non-blank byte (after a UTF-8 BOM) opens a JSON object or array -- read without copying a body
    that can be 50 MB."""
    start = _LEADING_BLANKS.match(content).end()  # type: ignore[union-attr]
    return content[start:start + 1] in (b"{", b"[")


def _is_json_type(content_type: str) -> bool:
    """``application/json``, ``text/json``, ``application/*+json`` (parameters ignored)."""
    base = content_type.split(";", 1)[0].strip().lower()
    return base.endswith(("/json", "+json"))


def _file_items(filename: str | None, content: bytes) -> Iterable[Any]:
    """Raw target items of an uploaded file: a JSON array (decoded up to BATCH_MAX_TARGETS + 1 items) or lazily
    read CSV rows."""
    if (filename or "").lower().endswith(".json") or _looks_like_json(content):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BatchError(f"JSON targets must be UTF-8 encoded ({exc}).") from exc
        return _json_target_items(text, _env_int("BATCH_MAX_TARGETS", 100_000))
    return _csv_rows(content)


async def _request_to_job(request: Request, query: Mapping[str, Any]) -> dict[str, Any]:
    """Build the batch job from JSON, CSV (text/csv body) or multipart (file field) requests.

    JSON: any ``*/json`` or ``+json`` content type, or a body whose first character is ``{`` or ``[`` under any
    other non-multipart content type (``curl -d`` sends ``application/x-www-form-urlencoded``). Body decoding and
    target validation (CPU-bound for large lists) run in a worker thread, the validation in slices.
    """
    raw_type = request.headers.get("content-type", "")
    content_type = raw_type.lower()
    body = await _read_body(request, _env_int("BATCH_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    job: dict[str, Any] = {
        "catalogs": _split_list(query.get("catalogs")),
        "radius_arcsec": query.get("radius_arcsec"),
        "strategies": _parse_strategies(query.get("strategies")),
        "nearest_only": query.get("nearest_only"),
        "include_data": query.get("include_data"),
        "format": query.get("format"),
    }
    multipart = content_type.startswith("multipart/")
    if _is_json_type(content_type) or (not multipart and _looks_like_json(body)):
        job, items = await _offload(_json_job, body, job)
        job["targets"] = await _parse_in_slices(items)
        return job
    if content_type.startswith("multipart/form-data"):
        fields = await _offload(_multipart_fields, body, raw_type)
        upload = fields.get("file") or fields.get("targets")
        if upload is None:
            raise HTTPException(status_code=422, detail="multipart request needs a 'file' field (CSV or JSON targets).")
        filename, content = upload
        for key in ("catalogs", "radius_arcsec", "strategies", "nearest_only", "include_data", "format"):
            if key in fields and job.get(key) is None:
                text = fields[key][1].decode("utf-8", "replace")
                job[key] = _split_list(text) if key == "catalogs" else _parse_strategies(text) if key == "strategies" else text
        job["targets"] = await _parse_in_slices(await _offload(_file_items, filename, content))
        return job
    if body:
        job["targets"] = await _parse_in_slices(await _offload(_csv_rows, body))
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
            except (AttributeError, TypeError):  # read-only state object: guards live for this request only
                logger.debug("app.state does not accept batch_guards; guards are per request")
    slots = getattr(state, "batch_upload_slots", None)
    if slots is None:
        slots = weakref.WeakKeyDictionary()
        try:
            state.batch_upload_slots = slots
        except (AttributeError, TypeError):  # read-only state object: slots live for this request only
            logger.debug("app.state does not accept batch_upload_slots; upload slots are per request")
    config = getattr(service, "association_config", None)
    return BatchCrossmatcher(registry=registry, client=client, guards=guards, cache=cache, upload_slots=slots,
                             association_config=config if isinstance(config, AssociationConfig) else None)


class _ClientDisconnected(Exception):
    """The HTTP client went away while its batch was running."""


async def _wait_for_disconnect(request: Request) -> None:
    """Return when the client disconnects. Called after the whole body was read, so the next ASGI message is
    ``http.disconnect`` (servers deliver it when the connection closes)."""
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _unless_disconnected(request: Request, work: Any) -> Any:
    """Await ``work`` (a coroutine), cancelling it as soon as the client disconnects: an abandoned batch must
    not keep querying the archives for up to BATCH_CATALOG_BUDGET_SECONDS per catalog."""
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        watcher.cancel()
        raise
    if task.done():
        watcher.cancel()
        return task.result()
    if watcher.exception() is not None:  # no usable disconnect signal from this server: just finish the work
        logger.debug("disconnect watcher failed: %r", watcher.exception())
        return await task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _ClientDisconnected()


def _all_failed(result: BatchResult) -> bool:
    """Every catalog failed for every target."""
    return bool(result.runs) and all(run.targets and run.failed_targets == run.targets for run in result.runs.values())


def _all_refused(result: BatchResult) -> bool:
    """Every failure is a client-side refusal (nothing could be sent upstream): an invalid request."""
    return _all_failed(result) and all(run.refused_targets == run.failed_targets for run in result.runs.values())


def _finite_json(value: Any) -> Any:
    """JSON has no NaN/Infinity: non-finite floats (e.g. in raw catalog rows) become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def _json_bytes(data: Any) -> bytes:
    """Serialise once, to bytes (runs in a worker thread; the response is not re-rendered on the loop)."""
    try:
        text = json.dumps(data, default=str, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    except ValueError:  # a NaN/Infinity somewhere: replace them (one more pass only in that case)
        text = json.dumps(_finite_json(data), default=str, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


@router.get("/strategies")
async def batch_strategies(request: Request) -> dict[str, Any]:
    """How each registry catalog is matched in a batch (upload | xmatch | cone), endpoints and chunk sizes."""
    engine = _engine_for(request)
    return {"default_catalogs": engine.default_catalogs(), "catalogs": engine.strategy_table(),
            "vizier_views": {name: view.note for name, view in VIZIER_VIEWS.items()},
            "max_radius_arcsec": MAX_RADIUS_ARCSEC, "max_catalogs": engine.max_catalogs}


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
    """Batch crossmatch of up to BATCH_MAX_TARGETS targets against up to BATCH_MAX_CATALOGS catalogs.

    Body: JSON ``{"targets": [{id, ra, dec, epoch?}], "catalogs": [...], "radius_arcsec": 3, ...}``, a
    ``text/csv`` target table, or multipart/form-data with a ``file`` field (CSV or JSON). ``format``: ``json``
    (per-target matches and failures), ``rows`` (flat dataset-like rows), ``parquet`` or ``csv`` (flat rows as a
    file). Flat rows carry a ``status``: "match", or "failed" (with ``error``) for a (target, catalog) query
    that failed -- a pair without rows has no counterpart. Each match's ``confidence`` is the posterior
    probability that the row is the target's counterpart. Headers: ``X-Batch-Request-Count``,
    ``X-Batch-Wall-Time-S``, ``X-Batch-Failed-Targets`` (targets with at least one failed catalog query).
    422: invalid request (including one no catalog could be queried for, e.g. every cone wider than the CDS
    XMatch limit); 502: every catalog failed upstream for every target. The batch is cancelled when the client
    disconnects.
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
        result = await _unless_disconnected(request, engine.run(
            job["targets"], job.get("catalogs"), radius_arcsec=radius, strategies=job.get("strategies"),
            nearest_only=_as_bool(job.get("nearest_only") or False)))
    except _ClientDisconnected:
        logger.info("batch cancelled: client disconnected")
        return Response(status_code=CLIENT_CLOSED_REQUEST)
    except BatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, UnicodeDecodeError, OverflowError, RecursionError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid batch request: {exc}") from exc
    if _all_failed(result):
        report = {n: r.as_dict() for n, r in result.runs.items()}
        if _all_refused(result):
            raise HTTPException(status_code=422, detail={"message": "no catalog could be queried for any target",
                                                         "catalogs": report})
        raise HTTPException(status_code=502, detail={"message": "every catalog failed", "catalogs": report})
    headers = {"X-Batch-Request-Count": str(result.request_count), "X-Batch-Wall-Time-S": f"{result.wall_time_s:.3f}",
               "X-Batch-Failed-Targets": str(result.failed_target_count)}
    if fmt == "parquet":
        headers["Content-Disposition"] = 'attachment; filename="batch_crossmatch.parquet"'
        content = await _offload(result.to_parquet_bytes, include_data=include)
        return Response(content, media_type="application/vnd.apache.parquet", headers=headers)
    if fmt == "csv":
        headers["Content-Disposition"] = 'attachment; filename="batch_crossmatch.csv"'
        text = await _offload(lambda: result.to_csv_text(include_data=include).encode("utf-8"))
        return Response(text, media_type="text/csv; charset=utf-8", headers=headers)

    def payload() -> bytes:
        if fmt == "rows":
            data = {"summary": result.summary(), "catalogs": {n: r.as_dict() for n, r in result.runs.items()},
                    "failures": result.failures_by_id(), "rows": result.rows(include_data=include)}
        else:
            data = result.as_dict(include_data=include)
        return _json_bytes(data)

    return Response(await _offload(payload), media_type="application/json", headers=headers)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def format_report(result: BatchResult) -> str:
    """Plain-text per-catalog report: strategy, requests, time, matches, failed targets."""
    lines = [(f"Batch crossmatch: {len(result.targets)} targets x {len(result.catalogs)} catalogs, "
              f"radius {result.radius_arcsec:g} arcsec -> {result.request_count} requests in {result.wall_time_s:.1f} s")]
    lines.append(f"{'catalog':<22}{'strategy':<9}{'requests':>9}{'retries':>8}{'time[s]':>9}{'matched':>9}"
                 f"{'matches':>9}{'failed':>8}  errors")
    for name in result.catalogs:
        run = result.runs[name]
        lines.append(f"{name:<22}{run.strategy:<9}{run.requests:>9}{run.retries:>8}{run.elapsed_s:>9.1f}"
                     f"{run.matched_targets:>9}{run.total_matches:>9}{run.failed_targets:>8}  {len(run.errors)}")
        for error in run.errors[:3]:
            lines.append(f"    ! {error}")
        if run.fallback_targets:
            lines.append(f"    {run.fallback_targets} target(s) fell back to cone searches")
    return "\n".join(lines)


def _check_output_path(path: str | os.PathLike[str]) -> Path:
    """Fail before the (possibly long) batch runs when its output cannot be written."""
    out = Path(path)
    if out.is_dir():
        raise BatchError(f"--out {out} is a directory; give a file name (.parquet, .csv or .json).")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix=".batch-write-check-"):
            pass
    except OSError as exc:
        raise BatchError(f"cannot write --out {out}: {exc}") from exc
    return out


def _echo(text: str) -> None:
    """print() that never fails on characters the console encoding lacks: redirected Windows output is cp1252,
    and target ids such as 'α Cen' in an error or report would end the CLI with a UnicodeEncodeError traceback
    (exit 1) instead of its message. Unencodable characters are written as backslash escapes."""
    stream = sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        print(text.encode(encoding, "backslashreplace").decode(encoding, "replace"), file=stream)


def run_cli(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch batch``; returns a process exit code: 0 success, 2 bad input (nothing could be
    queried, or the output cannot be written), 1 upstream errors (some catalog failed for every target)."""
    try:
        targets = load_targets_file(args.targets)
        catalogs = _split_list(args.catalogs)
        strategies = _parse_strategies(args.strategy)
        fmt = args.format or (Path(args.out).suffix.lstrip(".").lower() if args.out else "parquet") or "parquet"
        if fmt not in {"parquet", "csv", "json"}:
            raise BatchError(f"unsupported output format {fmt!r} (parquet, csv or json).")
        if args.out:
            _check_output_path(args.out)
        engine = BatchCrossmatcher()
        result = asyncio.run(engine.run(targets, catalogs, radius_arcsec=args.radius, strategies=strategies,
                                        nearest_only=args.nearest))
    except (BatchError, OSError, KeyError, ValueError, RecursionError, OverflowError) as exc:
        _echo(f"Error: {exc}")
        return 2
    except (AstroSearchError, httpx.HTTPError) as exc:
        _echo(f"Error: {exc}")
        return 1
    _echo(format_report(result))
    if result.failed_target_count:
        _echo(f"{result.failed_target_count} target(s) have failed catalog queries (status 'failed' rows in the output).")
    if args.out:
        try:
            path = result.write(args.out, fmt, include_data=not args.no_data)
        except OSError as exc:
            _echo(f"Error: could not write {args.out}: {exc}")
            return 2
        _echo(f"Wrote {sum(1 for row in result.rows(include_data=False) if row['status'] == ROW_STATUS_MATCH)} match "
              f"rows to {path}")
    if _all_refused(result):
        return 2
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
