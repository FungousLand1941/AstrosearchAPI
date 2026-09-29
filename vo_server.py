"""IVOA Virtual Observatory interfaces for AstroSearch: Cone Search, TAP (ADQL), VOSI and UWS.

Exposes the multi-archive crossmatch engine through standard VO protocols so that
TOPCAT, Aladin, pyvo, astroquery and any other VO client can use it directly:

* **Simple Cone Search 1.03** -- ``GET /vo/scs?RA=&DEC=&SR=[&VERB=&CATALOGS=]``
  (Williams, Hanisch, Szalay & Plante 2008, IVOA Recommendation "Simple Cone Search
  Version 1.03", ed. R. Plante). SR is in degrees; errors are VOTables with
  ``INFO name="Error"``; the three required FIELDs carry the UCD1 names the standard
  mandates (``ID_MAIN``, ``POS_EQ_RA_MAIN``, ``POS_EQ_DEC_MAIN``, section 2.2), as the
  VizieR and HEASARC cone searches do, while TAP/VOSI metadata use UCD1+.
* **Table Access Protocol 1.1** -- ``/vo/tap/sync`` and ``/vo/tap/async`` (Dowler,
  Rixon, Tody, Demleitner 2019, IVOA REC "Table Access Protocol Version 1.1"),
  answering a documented subset of **ADQL 2.1** (Mantelet, Morris, Demleitner et al.
  2023, IVOA REC "ADQL Version 2.1") that is parsed and evaluated in-process.
* **DALI 1.1** (Dowler, Demleitner, Taylor, Tody 2017): case-insensitive parameter
  names, MAXREC, RESPONSEFORMAT and ``QUERY_STATUS`` (OK / OVERFLOW / ERROR).
* **VOSI 1.1** (Graham, Rixon, Dowler et al. 2017): ``availability``,
  ``capabilities`` (with TAPRegExt 1.0, Demleitner et al. 2012) and ``tables``
  (VODataService 1.1 tableset, ``detail=min`` plus per-table documents).
* **UWS 1.1** (Harrison, Rixon et al. 2016): the asynchronous job lifecycle behind
  ``/vo/tap/async`` (in-memory job store, blocking ``WAIT``).
* **VOTable 1.4** (Ochsenbein, Taylor, Williams et al. 2019) output, with column
  semantics from the IVOA UCD1+ controlled vocabulary (Preite Martinez, Derriere,
  Gray et al., "The UCD1+ controlled vocabulary").

Published tables
----------------
``astrosearch.matches``
    One row per archive detection matched to a cone centre. Rows are computed live
    from the upstream archives by :class:`crossmatch.CrossmatchService`, so every query
    on this table must carry a spatial constraint
    ``1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', ra0, dec0, r_deg))`` (or the
    equivalent ``DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', ra0, dec0)) < r_deg``)
    as a top-level AND term. Archives excluded by ``catalog``/``wavelength`` conditions
    are not queried -- unless the query reads a ``group_*`` column, whose values are
    computed across every archive (a WHERE clause never changes the values of the rows
    it keeps). Each archive is asked for the rows nearest the cone centre, up to
    ``min(VO_CATALOG_ROW_LIMIT, VO_MAX_CONE_ROWS / archives queried)`` (defaults 2000 and
    3000; never fewer than the registry's interactive ``max_rows``), which bounds the
    work of one cone. An archive holding more, or failing, makes the cone incomplete:
    rows are then returned with ``QUERY_STATUS=OVERFLOW`` and WARNING INFOs, while
    aggregates and filters on other columns -- whose answers could be arbitrarily wrong
    -- are refused with an error naming the archives (CSV/TSV, which cannot carry the
    warning, refuse incomplete results altogether). ``TOP n ... ORDER BY separation``
    (the "nearest n" query) is answered exactly when its n-th row lies inside the
    radius to which every truncated archive is complete and no archive failed.
    ``separation`` is measured to catalog positions as published, at each catalog's
    epoch (no proper-motion propagation). ``group_id`` labels candidate counterpart
    sets built by :func:`assign_groups` from the positional errors, proper motions and
    parallaxes (at most one detection per catalog, co-located rows of one catalog counting
    as one, see there), not confirmed physical associations.
``astrosearch.catalogs``
    The catalog registry (provider, spectral regime, endpoint, citation, ...).
``TAP_SCHEMA.schemas|tables|columns|keys|key_columns``
    Standard TAP metadata (TAP 1.1 section 4) describing all of the above.

Supported ADQL subset (anything else is rejected with a TAP error VOTable naming
the construct): ``SELECT [ALL|DISTINCT] [TOP n] <items> FROM <one table> [[AS] alias]
[WHERE <condition>] [ORDER BY <keys> [ASC|DESC]] [OFFSET n]``. Items are ``*``,
``alias.*``, value expressions with optional aliases, or (without other items) the
aggregates COUNT(*), COUNT([DISTINCT] x), MIN, MAX, AVG, SUM. Conditions support
AND/OR/NOT, parentheses, the comparison operators, BETWEEN, [I]LIKE, IN (list),
IS [NOT] NULL, arithmetic (+ - * / with SQL integer division), ``||``, the ADQL
mathematical functions, LOWER/UPPER, and the geometry functions POINT, CIRCLE,
CONTAINS and DISTANCE in the ICRS frame. Joins, GROUP BY, subqueries, set operations,
BOX/POLYGON/REGION and uploads are not supported. Mathematical domain errors
(e.g. ``SQRT(-1)``, division by zero) yield NULL rather than aborting the query.
ROUND/TRUNCATE work on the decimal value of their argument (as SQL ``numeric`` does),
and MOD of two integers is an integer.

Resource bounds: expressions deeper than ``MAX_EXPRESSION_DEPTH``, strings longer
than ``MAX_STRING_LENGTH``, more than ``MAX_SELECT_ITEMS`` output columns and request
bodies above ``MAX_REQUEST_BYTES`` (HTTP 413) are refused; planning and evaluation
together get ``VO_ADQL_EVAL_SECONDS`` (checked on every row, inside LIKE matching,
sort-key computation and aggregation); a result is at most ``VO_MAX_RESULT_BYTES``
(its values while they are built, and its serialized body); async results held by the
UWS store total at most ``VO_UWS_MAX_STORED_BYTES`` (the oldest finished jobs are
destroyed first). Parsing and planning, crossmatching (the CPU-bound part of
CrossmatchService), grouping, ADQL evaluation and serialization run in dedicated worker
threads, so neither a dense cone nor a costly constant condition stalls the server's
event loop; cancelling a request (UWS ABORT, client disconnect) cancels its archive
queries too.
"""

from __future__ import annotations

import asyncio
import csv
import email.parser
import email.policy
import functools
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any
from urllib.parse import parse_qsl
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_attr

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response

from models import AstroSearchError, CatalogRegistry, UnifiedRecord, haversine_arcsec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOTABLE_MEDIA_TYPE = "application/x-votable+xml"
SCS_MEDIA_TYPE = "text/xml"  # SCS 1.03 section 3: responses are served as text/xml
VOSI_MEDIA_TYPE = "text/xml"
UWS_NS = "http://www.ivoa.net/xml/UWS/v1.0"
XLINK_NS = "http://www.w3.org/1999/xlink"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SERVICE_TITLE = "AstroSearch multi-archive crossmatch"
_UP_SINCE = datetime.now(UTC)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(slots=True)
class VOConfig:
    """Operational limits of the VO endpoints (overridable through environment variables).

    ``max_radius_deg`` bounds cone radii (SCS ``SR`` and ADQL ``CIRCLE``) because every
    matches query fans out to all upstream archives; ``catalog_row_limit`` is the most
    rows (the nearest) requested from one archive per cone and ``max_cone_rows`` the
    budget shared by all archives of a cone (see :func:`per_archive_row_limit`) -- a
    cone holding more is reported as incomplete, never silently as complete;
    ``default_maxrec``/``hard_maxrec`` are the TAPRegExt output limits;
    ``max_eval_seconds`` bounds the in-process ADQL planning and evaluation of one query
    (together); ``max_result_bytes`` bounds the size of one query's result (the values
    it builds and its serialized body); the UWS values set job lifetimes and
    ``uws_max_stored_bytes`` the total size of the results the job store keeps.
    """

    max_radius_deg: float = 0.05
    default_maxrec: int = 10000
    hard_maxrec: int = 100000
    catalog_row_limit: int = 2000
    max_cone_rows: int = 3000
    max_eval_seconds: float = 30.0
    execution_duration_s: int = 600
    retention_s: int = 86400
    max_jobs: int = 1000
    max_wait_s: float = 60.0
    max_result_bytes: int = 16_000_000
    uws_max_stored_bytes: int = 256_000_000

    @classmethod
    def from_env(cls) -> VOConfig:
        """Read VO_MAX_RADIUS_DEG, VO_CATALOG_ROW_LIMIT, VO_MAX_CONE_ROWS, VO_TAP_DEFAULT_MAXREC,
        VO_TAP_HARD_MAXREC, VO_ADQL_EVAL_SECONDS, VO_MAX_RESULT_BYTES and VO_UWS_* variables."""
        hard = max(1, _env_int("VO_TAP_HARD_MAXREC", 100000))
        return cls(
            max_radius_deg=_env_float("VO_MAX_RADIUS_DEG", 0.05),
            default_maxrec=min(_env_int("VO_TAP_DEFAULT_MAXREC", 10000), hard),
            hard_maxrec=hard,
            catalog_row_limit=max(1, _env_int("VO_CATALOG_ROW_LIMIT", 2000)),
            max_cone_rows=max(1, _env_int("VO_MAX_CONE_ROWS", 3000)),
            max_eval_seconds=_env_float("VO_ADQL_EVAL_SECONDS", 30.0),
            execution_duration_s=_env_int("VO_UWS_EXECUTION_DURATION", 600),
            retention_s=max(60, _env_int("VO_UWS_RETENTION_SECONDS", 86400)),
            max_jobs=max(1, _env_int("VO_UWS_MAX_JOBS", 1000)),
            max_wait_s=_env_float("VO_UWS_MAX_WAIT_SECONDS", 60.0),
            max_result_bytes=max(1024, _env_int("VO_MAX_RESULT_BYTES", 16_000_000)),
            uws_max_stored_bytes=max(1024, _env_int("VO_UWS_MAX_STORED_BYTES", 256_000_000)),
        )


def per_archive_row_limit(row_limit: int, max_cone_rows: int | None, n_archives: int) -> int:
    """Rows requested from each of ``n_archives`` archives for one cone.

    ``min(row_limit, max_cone_rows // n_archives)``: one archive alone may return up to
    ``row_limit`` rows, and ``n`` archives share the ``max_cone_rows`` budget so that the
    work of one cone (crossmatch, grouping, serialization) stays bounded. The result is
    at least 1; :func:`service_with_row_limit` never lowers an archive below the
    registry's interactive ``max_rows``.
    """
    limit = max(1, int(row_limit))
    if max_cone_rows:
        limit = min(limit, max(1, int(max_cone_rows) // max(1, int(n_archives))))
    return limit


# ---------------------------------------------------------------------------
# Worker Threads (keep CPU-bound work off the event loop)
# ---------------------------------------------------------------------------

_POOLS: dict[str, ThreadPoolExecutor] = {}
_POOL_LOCK = threading.Lock()
# Worker counts: "cpu" runs ADQL evaluation, grouping and serialization; "crossmatch" runs
# CrossmatchService.crossmatch (its archive requests are sent back to the event loop, so
# these threads mostly wait on the network). Separate pools keep VO work from starving
# asyncio.to_thread users elsewhere in the application.
_POOL_SIZES = {"cpu": ("VO_CPU_THREADS", 4), "crossmatch": ("VO_CROSSMATCH_THREADS", 8)}


def _pool(kind: str) -> ThreadPoolExecutor:
    with _POOL_LOCK:
        pool = _POOLS.get(kind)
        if pool is None:
            env, default = _POOL_SIZES[kind]
            pool = ThreadPoolExecutor(max_workers=max(1, _env_int(env, default) or default),
                                      thread_name_prefix=f"vo-{kind}")
            _POOLS[kind] = pool
        return pool


async def run_in_worker(fn: Callable[..., Any], *args: Any, kind: str = "cpu", **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` in one of this module's worker pools and await the result."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool(kind), functools.partial(fn, *args, **kwargs))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VOError(Exception):
    """Base class for errors reported to VO clients as error documents."""

    status_code = 400


class VOParameterError(VOError):
    """A request parameter is missing, malformed or out of range (HTTP 400)."""


class ADQLError(VOError):
    """The ADQL query cannot be parsed or uses an unsupported construct (HTTP 400)."""


class UpstreamError(VOError):
    """Upstream archives failed, so the requested result cannot be produced (HTTP 502)."""

    status_code = 502


class IncompleteResultError(VOError):
    """An upstream archive held more rows in the cone than were fetched, so an exact answer
    (aggregate, filtered or ranked query, or a CSV/TSV table that cannot carry the warning)
    cannot be given (HTTP 400: reduce the cone radius)."""


class ResultTooLargeError(VOError):
    """A query result larger than this service's ``max_result_bytes`` (a service limit)."""

    status_code = 400


class RequestTooLargeError(VOError):
    """A request body larger than :data:`MAX_REQUEST_BYTES`."""

    status_code = 413


class ServiceUnavailableError(VOError):
    """The crossmatch service could not be built (HTTP 503)."""

    status_code = 503


# ---------------------------------------------------------------------------
# Table & Column Metadata
# ---------------------------------------------------------------------------

_NUMERIC_TYPES = frozenset({"short", "int", "long", "float", "double"})
_STRING_TYPES = frozenset({"char", "unicodeChar"})


@dataclass(frozen=True, slots=True)
class VOColumn:
    """Metadata of one published column (VOTable FIELD / TAP_SCHEMA.columns row).

    ``verb`` is the minimum Cone Search VERB level (1-3) at which the column is
    returned; ``principal``/``indexed``/``std`` follow TAP_SCHEMA.columns semantics.
    """

    name: str
    datatype: str
    description: str
    unit: str | None = None
    ucd: str | None = None
    arraysize: str | None = None
    utype: str | None = None
    xtype: str | None = None
    principal: bool = False
    indexed: bool = False
    std: bool = False
    nullable: bool = True
    verb: int = 3

    @property
    def is_string(self) -> bool:
        return self.datatype in _STRING_TYPES

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "datatype": self.datatype, "arraysize": self.arraysize, "unit": self.unit,
            "ucd": self.ucd, "utype": self.utype, "xtype": self.xtype, "description": self.description,
        }


def _char(name: str, description: str, ucd: str | None = None, **kw: Any) -> VOColumn:
    return VOColumn(name, kw.pop("datatype", "char"), description, ucd=ucd, arraysize="*", **kw)


def _num(name: str, datatype: str, description: str, ucd: str | None = None, unit: str | None = None, **kw: Any) -> VOColumn:
    return VOColumn(name, datatype, description, unit=unit, ucd=ucd, **kw)


MATCHES_COLUMNS: tuple[VOColumn, ...] = (
    _char("match_id", "Unique row identifier '<catalog>:<source_id>'.", "meta.id;meta.main",
          principal=True, indexed=True, nullable=False, verb=1),
    _num("ra", "double", "Right ascension (ICRS) as published by the source catalog, at that catalog's epoch.",
         "pos.eq.ra;meta.main", "deg", principal=True, indexed=True, nullable=False, verb=1),
    _num("dec", "double", "Declination (ICRS) as published by the source catalog, at that catalog's epoch.",
         "pos.eq.dec;meta.main", "deg", principal=True, indexed=True, nullable=False, verb=1),
    _char("catalog", "AstroSearch catalog key of the archive that returned the row (see astrosearch.catalogs.name).",
          "meta.dataset", principal=True, indexed=True, nullable=False, verb=2),
    _char("source_id", "Identifier of the source in its own catalog.", "meta.id", principal=True, nullable=False, verb=2),
    _num("separation", "double",
         "Angular distance from the query position to the catalog position as published, at that catalog's epoch "
         "(no proper-motion propagation: rows of a high-proper-motion star from catalogs of different epochs can "
         "lie far apart).",
         "pos.angDistance", "arcsec", principal=True, nullable=False, verb=2),
    _num("match_confidence", "double",
         "Positional match score in [0, 1]: exp(-sep^2 / 2 sigma^2) with sigma the 1-sigma positional error "
         "(quadrature, floor 0.1 arcsec), or a linear taper over 30 arcsec when no error is published "
         "(crossmatch.match_score). A relative likelihood, not a Bayesian association probability.",
         "stat.likelihood", principal=True, verb=2),
    _char("group_id", "Candidate counterpart set (see vo_server.assign_groups): at most one detection per catalog, "
          "each within k*sqrt(sigma^2 + sigma_group^2) of the set's inverse-variance mean position (k = 3.72, "
          "i.e. 99.9% completeness for Gaussian errors; sigma = published 1-sigma error, 0.5 arcsec when none, "
          "plus 0.1 arcsec systematic, in quadrature), comparing proper-motion carriers at a common epoch and "
          "other rows with the set's position at their own epoch, including the annual parallax of the set's "
          "proper-motion carriers (removed for single-epoch catalogs, added to sigma otherwise). Rows of one "
          "catalog at indistinguishable positions -- entries sharing identical coordinates (e.g. a SIMBAD star "
          "and its planets) or survey duplicates within their errors -- count as ONE detection and share a set. "
          "A positional candidate, NOT a confirmed physical association: extended sources whose catalogued "
          "centroid is offset (e.g. radio jets) form separate sets.",
          "meta.id", principal=True, verb=2),
    _char("wavelength", "Registry label of the source catalog's regime: the spectral regimes 'optical', 'infrared', "
          "'radio', 'xray', or -- for compilations that are not bandpasses -- 'multi' (SIMBAD), 'extragalactic' "
          "(NED) and 'exoplanet' (NASA Exoplanet Archive). wavelength = 'radio' therefore selects radio SURVEYS, "
          "not radio objects listed in SIMBAD or NED.",
          "instr.bandpass", principal=True, verb=2),
    _num("pos_error", "double", "1-sigma circular positional uncertainty of the catalog position.",
         "stat.error;pos", "arcsec", verb=2),
    _num("epoch", "double", "Julian epoch of the catalog position (NULL when the catalog does not publish one).",
         "time.epoch", "yr"),
    _num("pmra", "double", "Proper motion in right ascension, including the cos(dec) factor.",
         "pos.pm;pos.eq.ra", "mas/yr"),
    _num("pmdec", "double", "Proper motion in declination.", "pos.pm;pos.eq.dec", "mas/yr"),
    _num("parallax", "double", "Trigonometric parallax published by the catalog.", "pos.parallax", "mas"),
    _num("redshift", "double", "Redshift published by the catalog (NED, SIMBAD, ...).", "src.redshift"),
    _char("object_type", "Object classification published by the catalog (e.g. SIMBAD otype, NED type).", "src.class"),
    _num("group_n_catalogs", "int", "Number of catalogs (among those queried) with a detection in this row's "
         "candidate counterpart set (see group_id); each catalog contributes at most one detection (which may span "
         "several co-located rows of that catalog).",
         "meta.number;meta.dataset"),
    _char("group_catalogs", "Comma-separated catalogs (among those queried) with a detection in this row's candidate "
          "counterpart set (see group_id).", "meta.id;meta.dataset"),
)

# Simple Cone Search 1.03 section 2.2 requires the UCD1 names ID_MAIN, POS_EQ_RA_MAIN and
# POS_EQ_DEC_MAIN on exactly one FIELD each (pyvo's SCSRecord.id/.pos look them up); the
# TAP, TAP_SCHEMA and VOSI metadata keep the UCD1+ equivalents.
SCS_UCD1: dict[str, str] = {"match_id": "ID_MAIN", "ra": "POS_EQ_RA_MAIN", "dec": "POS_EQ_DEC_MAIN"}
_EQ_UCD_PREFIXES = ("pos.eq.ra", "pos.eq.dec", "POS_EQ_RA", "POS_EQ_DEC")

CATALOGS_COLUMNS: tuple[VOColumn, ...] = (
    _char("name", "AstroSearch catalog key.", "meta.id;meta.main", principal=True, indexed=True, nullable=False),
    _char("provider", "Access protocol/provider used to query the archive (tap, irsa_gator, mast, sdss, heasarc_xamin).",
          "meta.code", principal=True, nullable=False),
    _char("wavelength", "Spectral regime of the catalog ('optical', 'infrared', 'radio', 'xray'), or 'multi', "
          "'extragalactic' or 'exoplanet' for compilations that are not bandpasses (see astrosearch.matches.wavelength).",
          "instr.bandpass", principal=True, nullable=False),
    _num("enabled", "int", "1 when the catalog is queried by default, else 0.", "meta.code.status", principal=True, nullable=False),
    _char("endpoint", "Upstream service URL.", "meta.ref.url"),
    _char("upstream_table", "Table name at the upstream service.", "meta.id;meta.table"),
    _char("upstream_catalog", "Catalog name at the upstream service (Gator, MAST, Xamin).", "meta.id"),
    _char("description", "Human-readable description.", "meta.note", datatype="unicodeChar", principal=True),
    _char("profiles", "Comma-separated search profiles that include the catalog.", "meta.code.class"),
    _num("epoch", "double", "Epoch of the catalog positions when a single Julian year applies to every row (a fixed "
         "registry epoch, or a per-row epoch column documented as constant, e.g. Gaia DR3 ref_epoch = J2016.0); "
         "else NULL (per-row epochs: see astrosearch.matches.epoch).", "time.epoch", "yr"),
    _num("max_rows", "int", "Most rows this VO service fetches from the archive for one astrosearch.matches cone (the "
         "nearest ones): VO_CATALOG_ROW_LIMIT when it is the only archive queried; when k archives are queried each "
         "gets max(interactive max_rows, min(this value, VO_MAX_CONE_ROWS / k)).", "meta.number"),
    _num("timeout_seconds", "double", "Per-query timeout.", "time.interval", "s"),
    _char("coverage", "Sky and depth coverage summary.", "meta.note", datatype="unicodeChar"),
    _char("citation", "Reference to cite when using the catalog.", "meta.bib", datatype="unicodeChar"),
    _char("acknowledgement", "Acknowledgement text requested by the archive.", "meta.note", datatype="unicodeChar"),
)

_TS_SCHEMAS: tuple[VOColumn, ...] = (
    _char("schema_name", "Fully qualified schema name.", std=True, principal=True, nullable=False),
    _char("utype", "Data model utype of the schema.", std=True),
    _char("description", "Brief description of the schema.", std=True, datatype="unicodeChar"),
    _num("schema_index", "int", "Recommended sort order of the schema.", std=True),
)
_TS_TABLES: tuple[VOColumn, ...] = (
    _char("schema_name", "Fully qualified schema name.", std=True, principal=True, nullable=False),
    _char("table_name", "Fully qualified table name.", std=True, principal=True, nullable=False),
    _char("table_type", "One of: table, view.", std=True, nullable=False),
    _char("utype", "Data model utype of the table.", std=True),
    _char("description", "Brief description of the table.", std=True, datatype="unicodeChar"),
    _num("table_index", "int", "Recommended sort order of the table.", std=True),
)
_TS_COLUMNS: tuple[VOColumn, ...] = (
    _char("table_name", "Fully qualified table name.", std=True, principal=True, nullable=False),
    _char("column_name", "Column name.", std=True, principal=True, nullable=False),
    _char("utype", "Data model utype of the column.", std=True),
    _char("ucd", "UCD1+ describing the column content.", std=True),
    _char("unit", "VOUnit of the column values.", std=True),
    _char("description", "Brief description of the column.", std=True, datatype="unicodeChar"),
    _char("datatype", "VOTable datatype of the column.", std=True, nullable=False),
    _char("arraysize", "VOTable arraysize of the column.", std=True),
    _char("xtype", "VOTable xtype of the column.", std=True),
    _num("size", "int", "Deprecated length of variable-length columns (TAP 1.0).", std=True),
    _num("principal", "int", "1 if the column is principal (shown by default).", std=True, nullable=False),
    _num("indexed", "int", "1 if the column is indexed.", std=True, nullable=False),
    _num("std", "int", "1 if the column is defined by a standard.", std=True, nullable=False),
    _num("column_index", "int", "Recommended sort order of the column.", std=True),
)
_TS_KEYS: tuple[VOColumn, ...] = (
    _char("key_id", "Unique key identifier.", std=True, nullable=False),
    _char("from_table", "Fully qualified table name.", std=True, nullable=False),
    _char("target_table", "Fully qualified table name.", std=True, nullable=False),
    _char("utype", "Data model utype of the key.", std=True),
    _char("description", "Description of the key.", std=True, datatype="unicodeChar"),
)
_TS_KEY_COLUMNS: tuple[VOColumn, ...] = (
    _char("key_id", "Key identifier from TAP_SCHEMA.keys.", std=True, nullable=False),
    _char("from_column", "Column in the from_table.", std=True, nullable=False),
    _char("target_column", "Column in the target_table.", std=True, nullable=False),
)


@dataclass(frozen=True, slots=True)
class TableDef:
    """A queryable table: schema, qualified name, description and columns."""

    schema: str
    name: str
    description: str
    columns: tuple[VOColumn, ...]
    table_type: str = "table"
    requires_cone: bool = False

    @property
    def short_name(self) -> str:
        return self.name.split(".", 1)[1]

    def column(self, name: str, *, quoted: bool = False) -> VOColumn | None:
        for col in self.columns:
            if (col.name == name) if quoted else (col.name.casefold() == name.casefold()):
                return col
        return None


SCHEMAS: tuple[tuple[str, str], ...] = (
    ("astrosearch", ("AstroSearch multi-wavelength crossmatch results computed live from public archives "
                     "(Gaia, SIMBAD, NED, 2MASS, AllWISE, Pan-STARRS, SDSS, FIRST, NVSS, LoTSS, VLASS, ROSAT, Chandra, XMM, ...).")),
    ("TAP_SCHEMA", "Table Access Protocol metadata (TAP 1.1 section 4)."),
)

TABLES: tuple[TableDef, ...] = (
    TableDef("astrosearch", "astrosearch.matches",
             "Archive detections within a cone, crossmatched across catalogs: separation from the cone centre, "
             "positional match score and candidate counterpart set. Computed live; requires a top-level "
             "1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', ra0, dec0, radius_deg)) constraint.",
             MATCHES_COLUMNS, table_type="view", requires_cone=True),
    TableDef("astrosearch", "astrosearch.catalogs",
             "Registry of the archives queried by AstroSearch, with endpoints, spectral regimes and citations.",
             CATALOGS_COLUMNS),
    TableDef("TAP_SCHEMA", "TAP_SCHEMA.schemas", "Schemas available for ADQL querying.", _TS_SCHEMAS),
    TableDef("TAP_SCHEMA", "TAP_SCHEMA.tables", "Tables available for ADQL querying.", _TS_TABLES),
    TableDef("TAP_SCHEMA", "TAP_SCHEMA.columns", "Columns of the tables available for ADQL querying.", _TS_COLUMNS),
    TableDef("TAP_SCHEMA", "TAP_SCHEMA.keys", "Foreign keys between tables (none are declared).", _TS_KEYS),
    TableDef("TAP_SCHEMA", "TAP_SCHEMA.key_columns", "Columns participating in foreign keys (none are declared).",
             _TS_KEY_COLUMNS),
)
TABLES_BY_NAME: dict[str, TableDef] = {t.name: t for t in TABLES}


def find_table(parts: Sequence[str], quoted: Sequence[bool]) -> TableDef:
    """Resolve a (possibly schema-qualified) table reference; raises ADQLError if unknown."""
    if len(parts) > 2:
        raise ADQLError(f"Catalog-qualified table names are not supported: {'.'.join(parts)}")
    for table in TABLES:
        schema, short = table.name.split(".", 1)
        cands: tuple[tuple[str, bool, str], ...]
        if len(parts) == 2:
            cands = ((parts[0], quoted[0], schema), (parts[1], quoted[1], short))
        else:
            cands = ((parts[0], quoted[0], short),)
        if all((given == actual) if q else (given.casefold() == actual.casefold()) for given, q, actual in cands):
            return table
    known = ", ".join(t.name for t in TABLES)
    raise ADQLError(f"Unknown table '{'.'.join(parts)}'. Available tables: {known}")


# ---------------------------------------------------------------------------
# Result Tables
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResultTable:
    """A tabular result ready for VOTable/CSV/TSV/JSON serialization.

    ``status`` is the DALI QUERY_STATUS (OK or OVERFLOW); ``infos`` are (name, value)
    pairs emitted as VOTable INFO elements (warnings, citations, the query, ...).
    ``truncated``/``failed`` name the upstream archives (catalog -> explanation) whose
    rows in the cone are incomplete; such a result always has status OVERFLOW.
    """

    name: str
    columns: list[VOColumn]
    rows: list[tuple[Any, ...]]
    status: str = "OK"
    infos: list[tuple[str, str]] = field(default_factory=list)
    description: str | None = None
    truncated: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def incomplete(self) -> bool:
        """True when rows are missing because an upstream archive was truncated or failed."""
        return bool(self.truncated or self.failed)

    def as_dicts(self) -> list[dict[str, Any]]:
        names = [c.name for c in self.columns]
        return [dict(zip(names, row)) for row in self.rows]

    def column_values(self, name: str) -> list[Any]:
        index = [c.name for c in self.columns].index(name)
        return [row[index] for row in self.rows]


# ---------------------------------------------------------------------------
# Row Builders (UnifiedRecord / registry / TAP_SCHEMA -> rows)
# ---------------------------------------------------------------------------


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _record_field(record: UnifiedRecord | Mapping[str, Any], name: str) -> Any:
    """Read a UnifiedRecord attribute or the same key of its ``as_dict()`` form."""
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name, None)


def match_rows(record: UnifiedRecord | Mapping[str, Any], *, centre: tuple[float, float] | None = None
               ) -> list[dict[str, Any]]:
    """Flatten a crossmatch result into ``astrosearch.matches`` rows (one per matched source).

    Reads the members of ``crossmatch_groups`` (or ``counterparts`` when a record has no
    groups), then labels candidate counterpart sets with :func:`assign_groups` around
    ``centre`` (default: the record's target). Rows are ordered by separation, then
    catalog and source id.
    """
    members: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    groups = _record_field(record, "crossmatch_groups") or []
    if groups:
        for group in groups:
            for member in group.get("members") or []:
                members.append((member, group))
    else:
        for sources in (_record_field(record, "counterparts") or {}).values():
            for member in sources:
                members.append((member, None))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for member, group in members:
        catalog = str(member.get("catalog"))
        source_id = str(member.get("source_id"))
        match_id = f"{catalog}:{source_id}"
        if match_id in seen:
            continue
        seen.add(match_id)
        physical = member.get("physical") or (member.get("metadata") or {}).get("physical") or {}
        metadata = member.get("metadata") or {}
        rows.append({
            "match_id": match_id,
            "ra": _float_or_none(member.get("ra")),
            "dec": _float_or_none(member.get("dec")),
            "catalog": catalog,
            "source_id": source_id,
            "separation": _float_or_none(member.get("separation_arcsec")),
            "match_confidence": _float_or_none(member.get("confidence")),
            "group_id": None,
            "wavelength": _str_or_none(metadata.get("wavelength")),
            "pos_error": _float_or_none(member.get("positional_error_arcsec")),
            "epoch": _float_or_none(member.get("epoch")),
            "pmra": _float_or_none(member.get("proper_motion_ra_masyr")),
            "pmdec": _float_or_none(member.get("proper_motion_dec_masyr")),
            "parallax": _float_or_none(physical.get("parallax")),
            "redshift": _float_or_none(physical.get("redshift")),
            "object_type": _str_or_none(physical.get("object_type")),
            "group_n_catalogs": None,
            "group_catalogs": None,
            # Not published: how assign_groups treats the parallax of this row's position.
            "_single_epoch": bool(metadata.get("single_epoch_positions")),
            "_epoch_range": member.get("epoch_range"),
        })
    rows.sort(key=lambda r: (r["separation"] if r["separation"] is not None else math.inf, r["catalog"], r["source_id"]))
    if centre is None:
        target = _record_field(record, "target") or {}
        ra0, dec0 = _float_or_none(target.get("ra")), _float_or_none(target.get("dec"))
        if ra0 is None or dec0 is None:
            placed = [r for r in rows if r["ra"] is not None and r["dec"] is not None]
            ra0, dec0 = (placed[0]["ra"], placed[0]["dec"]) if placed else (0.0, 0.0)
        centre = (ra0, dec0)
    assign_groups(rows, centre[0], centre[1])
    return rows


# ---------------------------------------------------------------------------
# Candidate Counterpart Sets (group_id)
# ---------------------------------------------------------------------------

# A true counterpart is missed with probability GROUP_MISS_PROBABILITY: for Gaussian
# per-axis errors the separation of two detections of one source follows a Rayleigh
# distribution of scale sqrt(s1^2 + s2^2), so P(sep > k s) = exp(-k^2 / 2) and
# k = sqrt(-2 ln p) (the completeness-based candidate selection of Pineau et al. 2017,
# A&A 597, A89, section 3; cf. Budavari & Szalay 2008, ApJ 679, 301).
GROUP_MISS_PROBABILITY = 1e-3
GROUP_K = math.sqrt(-2.0 * math.log(GROUP_MISS_PROBABILITY))  # 3.717
# Added in quadrature to every published error: inter-catalog frame/systematic residuals
# (the same 0.1 arcsec floor as crossmatch.match_score).
GROUP_SYSTEMATIC_ARCSEC = 0.1
# Assumed 1-sigma error of a detection whose catalog publishes none.
GROUP_UNKNOWN_ERROR_ARCSEC = 0.5
# Epoch at which proper-motion carriers are compared, and assumed for rows with no epoch.
GROUP_REFERENCE_EPOCH = 2000.0
# Rows of ONE catalog closer than this (1 mas) describe one position. SIMBAD, NED and the
# NASA Exoplanet Archive list a host star and each of its planets (and other entries that
# inherit the star's astrometry) at identical coordinates -- e.g. SIMBAD "NAME Barnard's
# star" and "NAME Barnard's Star b".."e", "HD 209458" and "HD 209458b" -- which differ only
# by floating-point rounding (1e-10 deg).
GROUP_SAME_POSITION_ARCSEC = 0.001
# Registry wavelength labels of compilation catalogs (SIMBAD, NED, Exoplanet Archive):
# their rows are distinct objects even where published errors overlap (a star with a
# 1 mas error next to an X-ray source with a 5" error), so only identical positions
# (GROUP_SAME_POSITION_ARCSEC) make two of their rows one detection.
COMPILATION_WAVELENGTHS = frozenset({"multi", "extragalactic", "exoplanet"})
# Largest separation at which two rows of one SURVEY can be one detection listed twice:
# Chandra's on-axis resolution (~0.5 arcsec FWHM; Chandra Proposers' Observatory Guide,
# HRMA chapter), the sharpest of the registry's surveys whose errors can reach arcseconds.
# Two sources farther apart could have been resolved by some survey, so faint sources with
# large errors (ROSAT, XMM, NVSS) are never merged merely because their errors overlap.
GROUP_DUPLICATE_MAX_ARCSEC = 0.5
_ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi


def _tangent_arcsec(ra: float, dec: float, ra0: float, dec0: float) -> tuple[float, float]:
    """Gnomonic (TAN) projection of (ra, dec) about (ra0, dec0), in arcsec (x east, y north)."""
    a, d, a0, d0 = (math.radians(v) for v in (ra, dec, ra0, dec0))
    cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    cos_c = max(cos_c, 1e-9)  # more than 90 deg away cannot be projected (never inside a VO cone)
    x = math.cos(d) * math.sin(a - a0) / cos_c
    y = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cos_c
    return x * _ARCSEC_PER_RAD, y * _ARCSEC_PER_RAD


@dataclass(slots=True)
class _Detection:
    """One detection: a row, or several rows of one catalog at indistinguishable positions.

    ``index`` is the representative (most precise) row, ``rows`` every row it stands for.
    ``err`` is the published 1-sigma error (None when unpublished) and ``sigma`` the
    grouping error (with the systematic floor). ``parallax`` (arcsec) is the row's own
    published parallax; ``single_epoch`` marks catalogs whose rows are one observation at
    ``epoch`` (apparent, parallax-displaced positions); ``span`` is the epoch range of a
    row without an epoch.
    """

    index: int
    rows: list[int]
    catalog: str
    ra: float
    dec: float
    x: float
    y: float
    err: float | None
    sigma: float
    epoch: float | None
    pm: tuple[float, float] | None  # arcsec/yr in (x, y)
    parallax: float | None = None
    single_epoch: bool = False
    span: tuple[float, float] | None = None
    compilation: bool = False

    def at(self, epoch: float) -> tuple[float, float]:
        if self.pm is None or self.epoch is None:
            return self.x, self.y
        dt = epoch - self.epoch
        return self.x + self.pm[0] * dt, self.y + self.pm[1] * dt


@dataclass(slots=True)
class _Group:
    members: list[_Detection]
    catalogs: set[str]
    pm: tuple[float, float] | None  # adopted from the seed (the most precise proper-motion carrier)
    parallax: float = 0.0  # arcsec: of the most precise proper-motion carrier publishing one
    sx: float = 0.0
    sy: float = 0.0
    sw: float = 0.0

    @property
    def sigma(self) -> float:
        return 1.0 / math.sqrt(self.sw)

    def centre(self, epoch: float | None = None) -> tuple[float, float]:
        """Inverse-variance mean (barycentric) position, at ``epoch`` when the set has a proper motion."""
        cx, cy = self.sx / self.sw, self.sy / self.sw
        if self.pm is not None and epoch is not None:
            dt = epoch - GROUP_REFERENCE_EPOCH
            cx, cy = cx + self.pm[0] * dt, cy + self.pm[1] * dt
        return cx, cy

    def probe_epoch(self, det: _Detection) -> float:
        """Epoch at which ``det`` (no proper motion of its own) is compared with this moving set:
        its own epoch; for a row with only an epoch span, the epoch in the span at which the set's
        track passes closest to it (as models.source_position_at's "target_pm_span"); else J2000."""
        if det.epoch is not None:
            return det.epoch
        if det.span is None or self.pm is None:
            return GROUP_REFERENCE_EPOCH
        lo, hi = det.span
        speed2 = self.pm[0] ** 2 + self.pm[1] ** 2
        if speed2 == 0.0:
            return 0.5 * (lo + hi)
        cx, cy = self.centre(GROUP_REFERENCE_EPOCH)
        t = GROUP_REFERENCE_EPOCH + ((det.x - cx) * self.pm[0] + (det.y - cy) * self.pm[1]) / speed2
        return min(hi, max(lo, t))

    def offset(self, det: _Detection, *, moving_phase: bool) -> tuple[float, float, float, float]:
        """(dx, dy, extra variance, epoch) of ``det`` relative to this set's predicted position.

        Proper-motion carriers (``moving_phase``) are compared at J2000.0, each moved with its
        own motion (their positions are barycentric). Other rows are compared with a moving
        set's barycentric track at their own epoch plus, when the set has a parallax ``p``,
        the annual-parallax displacement at that epoch (models.parallax_offset_arcsec) for a
        single-epoch (apparent) position, or ``p^2`` added to the variance when the parallax
        cannot be removed (a multi-epoch mean such as AllWISE or Pan-STARRS, or an unknown
        epoch) -- the treatment of crossmatch.source_position_at / parallax_uncertainty_arcsec.
        """
        if moving_phase:
            px, py = det.at(GROUP_REFERENCE_EPOCH)
            gx, gy = self.centre(GROUP_REFERENCE_EPOCH)
            return px - gx, py - gy, 0.0, GROUP_REFERENCE_EPOCH
        if self.pm is None:
            gx, gy = self.centre()
            return det.x - gx, det.y - gy, 0.0, det.epoch if det.epoch is not None else GROUP_REFERENCE_EPOCH
        epoch = self.probe_epoch(det)
        gx, gy = self.centre(epoch)
        extra = 0.0
        if self.parallax > 0.0:
            if det.single_epoch and det.epoch is not None:
                from models import parallax_offset_arcsec

                east, north = parallax_offset_arcsec(det.ra, det.dec, self.parallax * 1000.0, epoch)
                gx, gy = gx + east, gy + north
            else:
                extra = self.parallax * self.parallax
        return det.x - gx, det.y - gy, extra, epoch

    def add(self, det: _Detection, *, moving_phase: bool) -> None:
        """Add ``det`` to the inverse-variance mean (referred to J2000.0 when the set moves)."""
        if self.members and (moving_phase or self.pm is not None):
            dx, dy, extra, _epoch = self.offset(det, moving_phase=moving_phase)
            cx, cy = self.centre(GROUP_REFERENCE_EPOCH if self.pm is not None else None)
            x, y = cx + dx, cy + dy
        elif self.pm is not None:  # the seed of a moving set
            x, y = det.at(GROUP_REFERENCE_EPOCH)
            extra = 0.0
        else:
            x, y, extra = det.x, det.y, 0.0
        w = 1.0 / (det.sigma * det.sigma + extra)
        self.sx += w * x
        self.sy += w * y
        self.sw += w
        self.members.append(det)
        self.catalogs.add(det.catalog)
        if moving_phase and self.parallax <= 0.0 and det.parallax:
            self.parallax = det.parallax


def _row_detection(i: int, row: Mapping[str, Any], ra0: float, dec0: float) -> _Detection | None:
    ra, dec = row.get("ra"), row.get("dec")
    if ra is None or dec is None:
        return None
    x, y = _tangent_arcsec(float(ra), float(dec), ra0, dec0)
    raw_err = row.get("pos_error")
    err = float(raw_err) if isinstance(raw_err, (int, float)) and math.isfinite(raw_err) and raw_err >= 0 else None
    sigma = math.hypot(err if err is not None else GROUP_UNKNOWN_ERROR_ARCSEC, GROUP_SYSTEMATIC_ARCSEC)
    raw_epoch = row.get("epoch")
    epoch = float(raw_epoch) if isinstance(raw_epoch, (int, float)) and math.isfinite(raw_epoch) else None
    pmra, pmdec = row.get("pmra"), row.get("pmdec")
    pm = None
    if epoch is not None and isinstance(pmra, (int, float)) and isinstance(pmdec, (int, float)) \
            and math.isfinite(pmra) and math.isfinite(pmdec):
        pm = (pmra / 1000.0, pmdec / 1000.0)  # pmra includes cos(dec): an offset along x
    plx = row.get("parallax")
    parallax = plx / 1000.0 if isinstance(plx, (int, float)) and math.isfinite(plx) and plx > 0 else None
    span = None
    raw_span = row.get("_epoch_range")
    if epoch is None and isinstance(raw_span, (list, tuple)) and len(raw_span) == 2 \
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw_span):
        span = (float(min(raw_span)), float(max(raw_span)))
    return _Detection(i, [i], str(row.get("catalog")), float(ra), float(dec), x, y, err, sigma, epoch, pm,
                      parallax, bool(row.get("_single_epoch")), span,
                      row.get("wavelength") in COMPILATION_WAVELENGTHS)


def _same_detection(a: _Detection, b: _Detection) -> bool:
    """True when two rows of one catalog are one detection (``a`` the more precise).

    Positions within GROUP_SAME_POSITION_ARCSEC are identical. Survey rows (not
    compilations) whose published errors both exist are also one detection when their
    distance is within ``k * sqrt(err_a^2 + err_b^2)`` and GROUP_DUPLICATE_MAX_ARCSEC --
    statistically indistinguishable positions closer than any survey resolves, i.e. one
    source listed twice (e.g. the two Pan-STARRS DR2 objects 0.13" apart at Cyg X-1,
    errors 0.015" and 0.05"). Rows with proper motions must also move alike: within
    max(10 mas/yr, 10%), crossmatch's PM_AGREE_* criterion.
    """
    from crossmatch import PM_AGREE_FRACTION, PM_AGREE_MASYR

    distance = math.hypot(a.x - b.x, a.y - b.y)
    tolerance = GROUP_SAME_POSITION_ARCSEC
    if not a.compilation and a.err is not None and b.err is not None:
        tolerance = max(tolerance, min(GROUP_K * math.hypot(a.err, b.err), GROUP_DUPLICATE_MAX_ARCSEC))
    if distance > tolerance:
        return False
    if a.pm is not None and b.pm is not None:
        dpm = 1000.0 * math.hypot(a.pm[0] - b.pm[0], a.pm[1] - b.pm[1])
        return dpm <= max(PM_AGREE_MASYR, PM_AGREE_FRACTION * 1000.0 * math.hypot(*a.pm))
    return True


class _Grid:
    """Uniform grid of item ids over the tangent plane (candidate lookup by radius)."""

    def __init__(self, cell: float) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int], set[int]] = {}
        self.ids: set[int] = set()

    def key(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self.cell), math.floor(y / self.cell))

    def add(self, item: int, cell_key: tuple[int, int]) -> None:
        self.cells.setdefault(cell_key, set()).add(item)
        self.ids.add(item)

    def discard(self, item: int, cell_key: tuple[int, int]) -> None:
        self.cells.get(cell_key, set()).discard(item)

    def near(self, x: float, y: float, radius: float) -> set[int]:
        rings = math.ceil(radius / self.cell) + 1
        if (2 * rings + 1) ** 2 > 4 * len(self.cells) + 16:  # a huge radius: scanning everything is cheaper
            return set(self.ids)
        cx, cy = self.key(x, y)
        found: set[int] = set()
        for i in range(cx - rings, cx + rings + 1):
            for j in range(cy - rings, cy + rings + 1):
                hit = self.cells.get((i, j))
                if hit:
                    found |= hit
        return found


def _merge_duplicates(detections: list[_Detection]) -> list[_Detection]:
    """Merge rows of one catalog that are one detection (:func:`_same_detection`).

    Rows are taken from the most to the least precise; each joins the first earlier
    representative it matches (never a chain), whose position, error and epoch the merged
    detection keeps; a proper motion or parallax missing from the representative is taken
    from the first member publishing one.
    """
    by_catalog: dict[str, list[_Detection]] = {}
    for det in detections:
        by_catalog.setdefault(det.catalog, []).append(det)
    merged: list[_Detection] = []
    for dets in by_catalog.values():
        if len(dets) == 1:
            merged.extend(dets)
            continue
        dets.sort(key=lambda d: (d.sigma, d.index))
        reach = [max(GROUP_SAME_POSITION_ARCSEC, min(GROUP_K * math.sqrt(2.0) * (d.err or 0.0),
                                                     GROUP_DUPLICATE_MAX_ARCSEC)) for d in dets]
        grid = _Grid(max(0.01, sorted(reach)[len(reach) // 2]))
        reps: list[_Detection] = []
        for det, radius in zip(dets, reach):
            target = None
            for rid in sorted(grid.near(det.x, det.y, radius)):
                if _same_detection(reps[rid], det):
                    target = reps[rid]
                    break
            if target is None:
                grid.add(len(reps), grid.key(det.x, det.y))
                reps.append(det)
                continue
            target.rows.extend(det.rows)
            if target.pm is None and det.pm is not None and target.epoch is not None:
                target.pm = det.pm
            if target.parallax is None and det.parallax is not None:
                target.parallax = det.parallax
            target.single_epoch = target.single_epoch and det.single_epoch
        merged.extend(reps)
    return merged


def _detections(rows: Sequence[Mapping[str, Any]], ra0: float, dec0: float) -> list[_Detection]:
    """Detections of the positioned rows, co-located rows of one catalog merged (:func:`_merge_duplicates`)."""
    found = [d for d in (_row_detection(i, row, ra0, dec0) for i, row in enumerate(rows)) if d is not None]
    return _merge_duplicates(found)


def assign_groups(rows: list[dict[str, Any]], ra0: float, dec0: float) -> list[dict[str, Any]]:
    """Label candidate counterpart sets in ``astrosearch.matches`` rows (``group_*`` columns), in place.

    Method (the candidate-tuple model of Pineau et al. 2017, A&A 597, A89: at most one
    detection per catalogue, positions consistent within their errors):

    1. Positions are projected on the tangent plane at (ra0, dec0). Each detection gets
       ``sigma = sqrt(err^2 + 0.1^2)`` arcsec (``err`` = published 1-sigma error, else
       0.5 arcsec). Rows of one catalog at indistinguishable positions are ONE detection
       (:func:`_same_detection`): a compilation's host star and its planets share the
       star's coordinates, and survey duplicates lie within their own errors. Otherwise the
       one-per-catalog rule would hand a star's survey detections to whichever co-located
       entry came first (e.g. SIMBAD "HD 209458b" instead of "HD 209458").
    2. Detections are processed from the most to the least precise, proper-motion
       carriers first (compared at J2000.0, each moved with its own motion); the others
       are compared with a moving set's barycentric track at their own epoch (J2000.0 if
       unknown; for rows with only an epoch span, the span epoch closest to the track),
       displaced by the annual parallax of the set's carriers for single-epoch catalogs
       (2MASS, SDSS, ...: Proxima Cen's 2MASS position is 0.71" off its linear track and
       0.05" off after the 768 mas parallax), or with that parallax added to ``sigma`` in
       quadrature when it cannot be removed (multi-epoch means such as AllWISE, unknown
       epochs) -- as crossmatch.source_position_at / parallax_uncertainty_arcsec do.
    3. A detection may join a set that has no detection from its catalog when its
       distance ``psi`` to the set's inverse-variance mean position satisfies
       ``psi <= k * sqrt(sigma^2 + sigma_set^2)`` with ``k = sqrt(-2 ln 1e-3)`` = 3.72
       (a true pair is missed with probability 1e-3). Among such sets it joins the one
       with the largest Bayes factor ``2/s^2 exp(-psi^2 / 2 s^2)``, ``s^2 = sigma^2 +
       sigma_set^2`` (the two-detection Bayes factor for Gaussian errors of Budavari &
       Szalay 2008, ApJ 679, 301, section 3); otherwise it seeds a new set.

    Comparing with the set's mean (not pairwise union-find) prevents chaining: a
    detection with a large error cannot merge sets that are mutually inconsistent. The
    cost is ``O(n * candidates)`` with a grid index. Sets are labelled ``group-1``,
    ``group-2``, ... by increasing separation of their nearest member from (ra0, dec0).
    Every row of a merged detection gets its set's labels; rows without a position form
    single-row sets. Returns ``rows``.
    """
    detections = _detections(rows, ra0, dec0)
    if not detections:
        for n, row in enumerate(rows, start=1):
            row.update(group_id=f"group-{n}", group_n_catalogs=1, group_catalogs=str(row.get("catalog")))
        return rows
    k = GROUP_K
    reach = sorted(k * math.sqrt(2.0) * d.sigma for d in detections)
    grid = _Grid(max(0.25, reach[len(reach) // 2]))
    groups: list[_Group] = []
    registered: dict[int, tuple[int, int]] = {}
    max_group_sigma = 0.0
    max_parallax = 0.0

    def place(det: _Detection, *, moving_phase: bool) -> None:
        nonlocal max_group_sigma, max_parallax
        px, py = det.at(GROUP_REFERENCE_EPOCH) if moving_phase else (det.x, det.y)
        best: tuple[float, int] | None = None
        radius = k * math.sqrt(det.sigma ** 2 + max_group_sigma ** 2 + max_parallax ** 2) + max_parallax
        for gid in grid.near(px, py, radius):
            group = groups[gid]
            if det.catalog in group.catalogs:
                continue
            dx, dy, extra, _epoch = group.offset(det, moving_phase=moving_phase)
            s2 = det.sigma ** 2 + group.sigma ** 2 + extra
            psi2 = dx * dx + dy * dy
            if psi2 > k * k * s2:
                continue
            log_bayes = math.log(2.0 / s2) - psi2 / (2.0 * s2)
            if best is None or log_bayes > best[0] or (log_bayes == best[0] and gid < best[1]):
                best = (log_bayes, gid)
        if best is None:
            group = _Group([], set(), det.pm if moving_phase else None)
            group.add(det, moving_phase=moving_phase)
            groups.append(group)
            gid = len(groups) - 1
            registered[gid] = grid.key(*group.centre(GROUP_REFERENCE_EPOCH if moving_phase else None))
            grid.add(gid, registered[gid])
        else:
            gid = best[1]
            group = groups[gid]
            group.add(det, moving_phase=moving_phase)
            if group.pm is None:  # static sets: follow the mean position in the index
                new_key = grid.key(*group.centre())
                if new_key != registered[gid]:
                    grid.discard(gid, registered[gid])
                    registered[gid] = new_key
                    grid.add(gid, new_key)
        max_group_sigma = max(max_group_sigma, groups[gid].sigma)
        max_parallax = max(max_parallax, groups[gid].parallax)

    def order(d: _Detection) -> tuple[float, float, str, int]:
        return (d.sigma, math.hypot(d.x, d.y), d.catalog, d.index)

    moving = sorted((d for d in detections if d.pm is not None), key=order)
    static = sorted((d for d in detections if d.pm is None), key=order)
    for det in moving:
        place(det, moving_phase=True)
    if static and groups:
        # Index every moving set along its path over the epochs (and spans) of the remaining rows.
        epochs = [GROUP_REFERENCE_EPOCH]
        for d in static:
            epochs.extend(d.span if d.epoch is None and d.span else [d.epoch if d.epoch is not None
                                                                     else GROUP_REFERENCE_EPOCH])
        lo, hi = min(epochs), max(epochs)
        for gid, group in enumerate(groups):
            if group.pm is None:
                continue
            (x0, y0), (x1, y1) = group.centre(lo), group.centre(hi)
            steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / (grid.cell / 2.0)))
            for s in range(steps + 1):
                grid.add(gid, grid.key(x0 + (x1 - x0) * s / steps, y0 + (y1 - y0) * s / steps))
    for det in static:
        place(det, moving_phase=False)

    def nearest(group: _Group) -> tuple[float, str]:
        return min((rows[i]["separation"] if rows[i]["separation"] is not None else math.inf, rows[i]["match_id"])
                   for m in group.members for i in m.rows)

    labelled: set[int] = set()
    for n, group in enumerate(sorted(groups, key=nearest), start=1):
        catalogs = ",".join(sorted(group.catalogs))
        for member in group.members:
            for i in member.rows:
                rows[i].update(group_id=f"group-{n}", group_n_catalogs=len(group.catalogs), group_catalogs=catalogs)
                labelled.add(i)
    n = len(groups)
    for i, row in enumerate(rows):
        if i not in labelled:  # no position: a set of its own
            n += 1
            row.update(group_id=f"group-{n}", group_n_catalogs=1, group_catalogs=str(row.get("catalog")))
    return rows


# Per-row epoch columns that the archive documents as holding ONE value for every row,
# keyed by (upstream table, epoch column). Gaia DR3: "ref_epoch ... Reference epoch to
# which the astrometric source parameters are referred ... For Gaia DR3 this reference
# epoch is J2016.0" (Gaia DR3 documentation, gaia_source data model; Lindegren et al.
# 2021, A&A 649, A2, section 2); every row returned by the ESA archive carries 2016.0.
UNIFORM_EPOCH_COLUMNS: dict[tuple[str, str], float] = {
    ("gaiadr3.gaia_source", "ref_epoch"): 2016.0,
}


def catalog_epoch(cat: Any) -> float | None:
    """Julian epoch that applies to every row of a catalog, or None when epochs vary per row."""
    epoch = cat.epoch
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        return float(epoch)
    if isinstance(epoch, str) and isinstance(cat.table, str):
        return UNIFORM_EPOCH_COLUMNS.get((cat.table.strip().lower(), epoch.strip().lower()))
    return None


def catalog_rows(registry: CatalogRegistry, config: VOConfig | None = None) -> list[dict[str, Any]]:
    """Rows of ``astrosearch.catalogs`` from a catalog registry (all catalogs, enabled or not).

    ``max_rows`` is the most rows the VO service fetches from the archive for one cone
    (``config.catalog_row_limit``, or the registry's ``max_rows`` when larger, since
    :func:`service_with_row_limit` never lowers it).
    """
    cfg = config or VOConfig.from_env()
    rows = []
    for name, cat in registry.catalogs.items():
        epoch = catalog_epoch(cat)
        rows.append({
            "name": name,
            "provider": cat.provider,
            "wavelength": cat.wavelength,
            "enabled": 1 if cat.enabled else 0,
            "endpoint": cat.endpoint,
            "upstream_table": cat.table,
            "upstream_catalog": cat.catalog,
            "description": cat.description,
            "profiles": ",".join(cat.profiles) if cat.profiles else None,
            "epoch": epoch,
            "max_rows": max(int(cat.max_rows or 0), cfg.catalog_row_limit),
            "timeout_seconds": _float_or_none(cat.timeout_seconds),
            "coverage": cat.coverage,
            "citation": cat.citation,
            "acknowledgement": cat.acknowledgement,
        })
    return rows


def tap_schema_rows(table_name: str) -> list[dict[str, Any]]:
    """Rows of the TAP_SCHEMA tables describing every published table (TAP 1.1 section 4)."""
    if table_name == "TAP_SCHEMA.schemas":
        return [{"schema_name": s, "utype": None, "description": d, "schema_index": i} for i, (s, d) in enumerate(SCHEMAS)]
    if table_name == "TAP_SCHEMA.tables":
        return [{"schema_name": t.schema, "table_name": t.name, "table_type": "view" if t.table_type == "view" else "table",
                 "utype": None, "description": t.description, "table_index": i} for i, t in enumerate(TABLES)]
    if table_name == "TAP_SCHEMA.columns":
        return [{
            "table_name": t.name, "column_name": c.name, "utype": c.utype, "ucd": c.ucd, "unit": c.unit,
            "description": c.description, "datatype": c.datatype, "arraysize": c.arraysize, "xtype": c.xtype,
            "size": None, "principal": int(c.principal), "indexed": int(c.indexed), "std": int(c.std), "column_index": j,
        } for t in TABLES for j, c in enumerate(t.columns)]
    if table_name in {"TAP_SCHEMA.keys", "TAP_SCHEMA.key_columns"}:
        return []
    raise ADQLError(f"Unknown TAP_SCHEMA table {table_name}")


# ---------------------------------------------------------------------------
# Crossmatch Access
# ---------------------------------------------------------------------------


def _upstream_catalogs(registry: CatalogRegistry | None, requested: Iterable[str] | None) -> list[str] | None:
    """Validate requested catalog keys against the registry's enabled catalogs."""
    if requested is None:
        return None
    names = [n.strip() for n in requested if n and n.strip()]
    if registry is None:
        return names
    enabled = registry.enabled_catalogs()
    unknown = [n for n in names if n not in enabled]
    if unknown:
        raise VOParameterError(f"Unknown or disabled catalog(s): {', '.join(unknown)}. Known: {', '.join(sorted(enabled))}")
    return names


@dataclass(slots=True)
class ConeFetch:
    """Outcome of one crossmatch cone: ``astrosearch.matches`` rows, INFO pairs and completeness.

    ``truncated`` maps each archive that held more rows in the cone than were fetched
    (the nearest ones were kept) to an explanation that includes the radius up to which
    its rows are complete; ``failed`` maps each archive that failed to
    ``"<ErrorType>: <message>"``. Either makes the cone incomplete. ``complete_to`` maps
    each truncated archive to that radius in arcsec (the separation of its farthest
    kept row; 0 when none was kept): every row of the archive nearer than it is present.
    """

    rows: list[dict[str, Any]]
    infos: list[tuple[str, str]]
    truncated: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    complete_to: dict[str, float] = field(default_factory=dict)

    @property
    def incomplete(self) -> bool:
        return bool(self.truncated or self.failed)


def incompleteness_summary(truncated: Mapping[str, str], failed: Mapping[str, str]) -> str:
    """One line naming every failed/truncated archive, as ``catalog: ErrorType: message`` segments."""
    parts = [f"{name}: {reason}" for name, reason in sorted(failed.items())]
    parts += [f"{name}: Truncated: {reason}" for name, reason in sorted(truncated.items())]
    return "; ".join(parts)


def incompleteness_error(truncated: Mapping[str, str], failed: Mapping[str, str], why: str) -> VOError:
    """The error raised when ``why`` needs every row of the cone but archives were truncated/failed.

    Failed archives give an :class:`UpstreamError` (HTTP 502); truncation alone gives an
    :class:`IncompleteResultError` (HTTP 400: a smaller cone fits the per-archive row limit).
    """
    detail = incompleteness_summary(truncated, failed)
    if failed:
        return UpstreamError(f"{why}, but upstream archives failed or were truncated: {detail}. Retry later "
                             "or restrict the query to the archives that answered")
    return IncompleteResultError(f"{why}, but the cone holds more rows than this service fetches per archive: "
                                 f"{detail}. Reduce the cone radius")


def service_with_row_limit(service: Any, row_limit: int) -> Any:
    """A view of a :class:`crossmatch.CrossmatchService` that asks each archive for up to ``row_limit`` rows.

    The registry's per-catalog ``max_rows`` (200 by default, sized for interactive
    searches) is raised to ``row_limit`` in a private copy of the registry; the
    providers, HTTP client and every other attribute are shared with ``service``.
    Other services (e.g. test doubles) are returned unchanged.
    """
    import copy

    from crossmatch import CrossmatchService, QueryPlanner

    registry = getattr(service, "registry", None)
    if not isinstance(service, CrossmatchService) or row_limit <= 0 or not isinstance(registry, CatalogRegistry) \
            or not hasattr(registry, "_catalogs"):
        return service
    raised = copy.copy(registry)
    raised._catalogs = {  # a private copy of the registry, never shared
        name: replace(cat, max_rows=max(int(cat.max_rows or 0), row_limit)) for name, cat in registry.catalogs.items()
    }
    derived = copy.copy(service)
    derived.registry = raised
    derived.planner = QueryPlanner(raised)
    derived.executor = copy.copy(service.executor)
    derived.executor.registry = raised
    return derived


class CrossmatchCancellation:
    """Cancellation token shared by a request's task and the worker thread running its crossmatch.

    :meth:`cancel` (called when the awaiting task is cancelled, e.g. by a UWS ABORT or a
    client disconnect) stops the archive requests already sent to the application's event
    loop, refuses new ones and cancels the crossmatch on the worker thread's private loop.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._futures: set[Any] = set()
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._thread_task: asyncio.Task[Any] | None = None

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def cancel(self) -> None:
        self.event.set()
        with self._lock:
            futures = list(self._futures)
            loop, task = self._thread_loop, self._thread_task
        for future in futures:
            future.cancel()
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # the private loop has just finished
                pass

    def track(self, future: Any) -> None:
        with self._lock:
            self._futures.add(future)
        if self.event.is_set():
            future.cancel()

    def untrack(self, future: Any) -> None:
        with self._lock:
            self._futures.discard(future)

    def attach(self, loop: asyncio.AbstractEventLoop, task: asyncio.Task[Any]) -> None:
        with self._lock:
            self._thread_loop, self._thread_task = loop, task


class _LoopBridgedExecutor:
    """Executor proxy for a CrossmatchService running in a worker thread: its archive
    requests (``execute``) run on the application's event loop, where the shared HTTP
    client and providers live; everything else is delegated unchanged. Requests stop when
    ``token`` is cancelled."""

    def __init__(self, inner: Any, loop: asyncio.AbstractEventLoop, token: CrossmatchCancellation | None = None) -> None:
        self._inner = inner
        self._loop = loop
        self._token = token or CrossmatchCancellation()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        if self._token.cancelled:
            raise asyncio.CancelledError
        future = asyncio.run_coroutine_threadsafe(self._inner.execute(*args, **kwargs), self._loop)
        self._token.track(future)
        try:
            waiting = asyncio.wrap_future(future)
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(waiting), timeout=1.0)
                except TimeoutError:
                    if self._loop.is_closed() or not self._loop.is_running():
                        future.cancel()
                        raise UpstreamError("The server's event loop stopped while archives were being queried") from None
        finally:
            self._token.untrack(future)


def _crossmatch_in_thread(service: Any, loop: asyncio.AbstractEventLoop, call: Mapping[str, Any],
                          token: CrossmatchCancellation | None = None) -> Any:
    """Run ``service.crossmatch(**call)`` on a private event loop in this worker thread.

    CrossmatchService.crossmatch awaits only ``self.executor.execute`` (the archive
    requests), which :class:`_LoopBridgedExecutor` sends back to ``loop``; its CPU-bound
    remainder (row classification, matching and crossmatch._group_matches, which is
    O(n^2)) runs here instead of stalling ``loop``. Cancelling ``token`` cancels the
    crossmatch and its outstanding archive requests.
    """
    import copy

    cancel = token or CrossmatchCancellation()
    bridged = copy.copy(service)
    bridged.executor = _LoopBridgedExecutor(service.executor, loop, cancel)

    async def run() -> Any:
        task = asyncio.current_task()
        assert task is not None
        cancel.attach(asyncio.get_running_loop(), task)
        if cancel.cancelled:
            raise asyncio.CancelledError
        return await bridged.crossmatch(**call)

    return asyncio.run(run())


def _enabled_count(service: Any) -> int:
    registry = getattr(service, "registry", None)
    try:
        return max(1, len(registry.enabled_catalogs())) if registry is not None else 1
    except Exception:  # noqa: BLE001 - a registry double without enabled_catalogs()
        return 1


async def fetch_match_rows(
    service: Any,
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    catalogs: Sequence[str] | None = None,
    row_limit: int | None = None,
    max_total_rows: int | None = None,
) -> ConeFetch:
    """Run the crossmatch for a cone and return its rows, INFO pairs and completeness.

    ``catalogs`` restricts the archives queried (an empty sequence queries nothing);
    ``row_limit`` raises the number of rows requested per archive, shared out of
    ``max_total_rows`` over the archives queried (see :func:`per_archive_row_limit` and
    :func:`service_with_row_limit`). A :class:`crossmatch.CrossmatchService` runs in a
    worker thread (see :func:`_crossmatch_in_thread`), and rows are flattened and
    grouped (:func:`match_rows`) in another, so the event loop is never stalled. Archive
    failures and truncations become ``WARNING`` infos and are reported in
    :attr:`ConeFetch.failed` / :attr:`ConeFetch.truncated` (a catalog is truncated when
    crossmatch reports ``truncated`` or a positive ``excess_row_count``); if every
    archive failed an :class:`UpstreamError` is raised. Invalid input raises
    :class:`VOParameterError`.
    """
    from crossmatch import AdvancedQuery, CrossmatchService

    if catalogs is not None and len(catalogs) == 0:
        return ConeFetch([], [("WARNING", "No catalogs selected; no archive was queried.")])
    if row_limit:
        n_archives = len(catalogs) if catalogs is not None else _enabled_count(service)
        service = service_with_row_limit(service, per_archive_row_limit(row_limit, max_total_rows, n_archives))
    # 1e-6 arcsec rounding keeps SR(deg)*3600 exact for decimal radii (10/3600*3600 != 10).
    radius_arcsec = round(radius_deg * 3600.0, 6)
    call: dict[str, Any] = {"ra": ra, "dec": dec, "radius_arcsec": radius_arcsec}
    try:
        if catalogs is not None:
            call["query"] = AdvancedQuery.from_dict({
                "ra": ra, "dec": dec, "radius_arcsec": radius_arcsec, "catalogs": list(catalogs), "min_confidence": 0.0,
            })
        if isinstance(service, CrossmatchService):
            token = CrossmatchCancellation()
            try:
                record = await run_in_worker(_crossmatch_in_thread, service, asyncio.get_running_loop(), call, token,
                                             kind="crossmatch")
            except asyncio.CancelledError:
                token.cancel()  # the thread would otherwise keep querying every archive
                raise
        else:
            record = await service.crossmatch(**call)
    except VOError:
        raise
    except ValueError as exc:
        raise VOParameterError(str(exc)) from exc
    except AstroSearchError as exc:
        raise UpstreamError(f"Crossmatch failed: {exc.__class__.__name__}: {exc}") from exc

    failures = list(_record_field(record, "failures") or [])
    queried = int(_record_field(record, "catalogs_queried") or 0)
    if queried and len(failures) >= queried:
        detail = "; ".join(f"{f.get('catalog')}: {f.get('error_type')}: {f.get('message')}" for f in failures)
        raise UpstreamError(f"All {queried} upstream archives failed: {detail}")
    rows = await run_in_worker(match_rows, record, centre=(ra, dec))
    infos: list[tuple[str, str]] = [("catalogs_queried", str(queried))]
    failed: dict[str, str] = {}
    for failure in failures:
        name = str(failure.get("catalog"))
        failed[name] = f"{failure.get('error_type')}: {failure.get('message')}"
        infos.append(("WARNING", f"{name} failed ({failure.get('error_type')}): {failure.get('message')}"))
    truncated: dict[str, str] = {}
    complete_to: dict[str, float] = {}
    for name, stats in sorted((_record_field(record, "catalog_results") or {}).items()):
        if not isinstance(stats, Mapping) or name in failed:
            continue
        if stats.get("truncated") or int(stats.get("excess_row_count") or 0) > 0:
            seps = [r["separation"] for r in rows if r["catalog"] == name and r["separation"] is not None]
            complete_to[name] = max(seps) if seps else 0.0
            reach = f"complete only to {max(seps):.2f} arcsec" if seps else "no row inside the cone was kept"
            truncated[name] = (f"only the {int(stats.get('row_count') or len(seps))} nearest rows were fetched "
                               f"({reach} of the {radius_arcsec:g} arcsec cone)")
    provenance = _record_field(record, "provenance") or {}
    for warning in provenance.get("warnings") or []:
        infos.append(("WARNING", str(warning)))
    for name, reason in truncated.items():
        infos.append(("WARNING", f"{name} truncated: {reason}"))
    for name, citation in sorted((provenance.get("citations") or {}).items()):
        infos.append(("citation", f"{name}: {citation}"))
    return ConeFetch(rows, infos, truncated, failed, complete_to)


def _validate_cone(ra: float, dec: float, radius_deg: float, config: VOConfig, *, allow_zero: bool = False) -> float:
    if not all(math.isfinite(v) for v in (ra, dec, radius_deg)):
        raise VOParameterError("RA, DEC and the radius must be finite numbers")
    if not 0.0 <= ra <= 360.0:
        raise VOParameterError(f"RA must be within [0, 360] degrees, got {ra:g}")
    if not -90.0 <= dec <= 90.0:
        raise VOParameterError(f"DEC must be within [-90, 90] degrees, got {dec:g}")
    if radius_deg < 0 or (radius_deg == 0 and not allow_zero):
        raise VOParameterError(f"The search radius must be positive, got {radius_deg:g} deg")
    if radius_deg > config.max_radius_deg:
        raise VOParameterError(
            f"The search radius {radius_deg:g} deg exceeds this service's maximum of {config.max_radius_deg:g} deg "
            "(every cone is fanned out to all upstream archives)")
    return ra % 360.0


# ---------------------------------------------------------------------------
# Simple Cone Search 1.03
# ---------------------------------------------------------------------------


def _parse_float_param(params: Mapping[str, str], name: str) -> float:
    raw = params.get(name)
    if raw is None or not str(raw).strip():
        raise VOParameterError(f"Missing required parameter {name}")
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise VOParameterError(f"Invalid {name} value: {raw!r} is not a number") from exc


async def cone_search(
    service: Any,
    ra: float,
    dec: float,
    sr_deg: float,
    *,
    verb: int = 2,
    catalogs: Sequence[str] | None = None,
    registry: CatalogRegistry | None = None,
    config: VOConfig | None = None,
) -> ResultTable:
    """Simple Cone Search: crossmatched sources within ``sr_deg`` degrees of (ra, dec).

    ``verb`` selects the columns (1 = id and position, 2 = default, 3 = all);
    ``SR = 0`` returns the column metadata without querying any archive. The id and
    position FIELDs carry the SCS 1.03 UCD1 names (:data:`SCS_UCD1`). ``catalogs``, when
    given, must name at least one catalog. When an archive failed or held more rows
    than it was asked for (see :func:`per_archive_row_limit`), the rows are still
    returned (each archive's rows are complete up to the radius its WARNING states) but
    QUERY_STATUS is OVERFLOW.
    """
    cfg = config or VOConfig.from_env()
    if verb not in (1, 2, 3):
        raise VOParameterError(f"VERB must be 1, 2 or 3, got {verb}")
    ra = _validate_cone(ra, dec, sr_deg, cfg, allow_zero=True)
    columns = [replace(c, ucd=SCS_UCD1.get(c.name, c.ucd)) for c in MATCHES_COLUMNS if c.verb <= verb]
    if catalogs is not None and not [c for c in catalogs if c and c.strip()]:
        raise VOParameterError("CATALOGS names no catalog; give comma-separated catalog keys or omit it to query "
                               "every enabled catalog")
    selected = _upstream_catalogs(registry or getattr(service, "registry", None), catalogs)
    fetched = ConeFetch([], [])
    if sr_deg > 0:
        fetched = await fetch_match_rows(service, ra, dec, sr_deg, catalogs=selected, row_limit=cfg.catalog_row_limit,
                                         max_total_rows=cfg.max_cone_rows)
    table_rows = [tuple(row[c.name] for c in columns) for row in fetched.rows]
    description = f"AstroSearch cone search: RA={ra:.7f} DEC={dec:.7f} SR={sr_deg:g} deg"
    infos = list(fetched.infos)
    if fetched.incomplete:
        infos.append(("INCOMPLETE", incompleteness_summary(fetched.truncated, fetched.failed)))
    return ResultTable("matches", columns, table_rows, "OVERFLOW" if fetched.incomplete else "OK", infos, description,
                       dict(fetched.truncated), dict(fetched.failed))


# ---------------------------------------------------------------------------
# Serializers (VOTable 1.4, CSV, TSV, JSON)
# ---------------------------------------------------------------------------


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


# Characters outside the XML 1.0 ``Char`` production (W3C XML 1.0, 5th ed., section 2.2).
_XML_ILLEGAL = re.compile("[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]")


def has_xml_illegal_chars(text: str) -> bool:
    """True when ``text`` contains a character that no XML 1.0 document may contain (e.g. U+0001)."""
    return _XML_ILLEGAL.search(text) is not None


def xml_safe(text: str) -> str:
    """Replace characters that XML 1.0 forbids by U+FFFD so the text can be serialized."""
    return _XML_ILLEGAL.sub("\uFFFD", text)


def _xml_text(value: Any) -> str:
    """Escape text content for XML after removing XML-illegal characters."""
    return _xml_escape(xml_safe(str(value)))


def _xml_attr_value(value: Any) -> str:
    """Quoted XML attribute value after removing XML-illegal characters."""
    return _xml_attr(xml_safe(str(value)))


def render_votable(result: ResultTable, *, extra_root_infos: Sequence[tuple[str, str]] = ()) -> bytes:
    """Serialize a result as a VOTable 1.4 document (TABLEDATA) with DALI QUERY_STATUS.

    QUERY_STATUS=OK precedes the table; when the output was truncated by MAXREC an
    INFO QUERY_STATUS=OVERFLOW follows the TABLE (DALI 1.1 section 4.4.1).
    """
    from astropy.io.votable.tree import CooSys, Field, Info, Resource, TableElement, VOTableFile

    vot = VOTableFile(version="1.4")
    used_ids: dict[str, int] = {}

    def info(name: str, value: str) -> Any:
        # XML IDs must be unique; astropy would otherwise reuse the name for repeated INFOs.
        base = "info_" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        used_ids[base] = used_ids.get(base, 0) + 1
        return Info(ID=base if used_ids[base] == 1 else f"{base}_{used_ids[base]}", name=xml_safe(name),
                    value=xml_safe(value))

    for name, value in extra_root_infos:
        vot.infos.append(info(name, value))
    resource = Resource(type="results")
    vot.resources.append(resource)
    resource.infos.append(info("QUERY_STATUS", "OK"))
    for name, value in result.infos:
        resource.infos.append(info(name, value))
    has_eq = any((c.ucd or "").startswith(_EQ_UCD_PREFIXES) for c in result.columns)
    if has_eq:
        resource.coordinate_systems.append(CooSys(ID="icrs", system="ICRS"))

    # FIELD IDs equal the column names (astropy/pyvo use IDs as table column names), made
    # unique against each other and against the INFO/COOSYS/TABLE IDs.
    taken = {"icrs", "astrosearch_result", *(i.ID for i in vot.infos), *(i.ID for i in resource.infos)}
    table = TableElement(vot, ID="astrosearch_result", name=result.name)
    if result.description:
        table.description = xml_safe(result.description)
    resource.tables.append(table)
    string_index = {i for i, c in enumerate(result.columns) if c.is_string}
    rows = [tuple(xml_safe(str(v)) if i in string_index and v is not None else v for i, v in enumerate(row))
            for row in result.rows]
    for index, col in enumerate(result.columns):
        datatype = col.datatype
        if datatype == "char" and not all(_is_ascii(str(row[index])) for row in rows if row[index] is not None):
            datatype = "unicodeChar"
        field_id = re.sub(r"[^A-Za-z0-9_.-]", "_", col.name) or "col"
        if not (field_id[0].isalpha() or field_id[0] == "_"):
            field_id = "_" + field_id
        base_id, n = field_id, 1
        while field_id in taken:
            n += 1
            field_id = f"{base_id}_{n}"
        taken.add(field_id)
        fld = Field(vot, name=xml_safe(col.name), ID=field_id, datatype=datatype,
                    arraysize=col.arraysize, unit=col.unit, ucd=col.ucd, utype=col.utype, xtype=col.xtype,
                    ref="icrs" if has_eq and (col.ucd or "").startswith(_EQ_UCD_PREFIXES) else None)
        fld.description = xml_safe(col.description) if col.description else col.description
        table.fields.append(fld)
    table.create_arrays(nrows=len(rows))
    for i, row in enumerate(rows):
        values = []
        mask = []
        for col, value in zip(result.columns, row):
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                values.append("" if col.is_string else (math.nan if col.datatype in {"float", "double"} else 0))
                mask.append(True)
            else:
                values.append(str(value) if col.is_string else value)
                mask.append(False)
        table.array[i] = tuple(values)
        table.array.mask[i] = tuple(mask)

    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vot.to_xml(buffer, tabledata_format="tabledata")
    body = buffer.getvalue()
    if result.status == "OVERFLOW":
        body = body.replace(b"</TABLE>", b'</TABLE>\n  <INFO name="QUERY_STATUS" value="OVERFLOW"/>', 1)
    return body


def error_votable(message: str, *, scs: bool = False) -> bytes:
    """Error document: DALI ``INFO name="QUERY_STATUS" value="ERROR"`` in the results RESOURCE.

    For Simple Cone Search (``scs=True``) the SCS 1.03 form is added as well: an
    ``INFO ID="Error" name="Error" value="<message>"`` directly under VOTABLE.
    """
    from astropy.io.votable.tree import Info, Resource, VOTableFile

    message = xml_safe(message)
    vot = VOTableFile(version="1.4")
    if scs:
        vot.infos.append(Info(ID="Error", name="Error", value=message))
    resource = Resource(type="results")
    status = Info(name="QUERY_STATUS", value="ERROR")
    status.content = message
    resource.infos.append(status)
    vot.resources.append(resource)
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vot.to_xml(buffer)
    return buffer.getvalue()


def _text_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def render_csv(result: ResultTable, *, delimiter: str = ",") -> bytes:
    """CSV (RFC 4180, header row) or TSV serialization; NULL is the empty field.

    TSV follows the IANA ``text/tab-separated-values`` definition, which has no quoting:
    fields are written verbatim, and the tab, CR and LF characters that it cannot
    represent inside a field are replaced by a space. The format has no place for
    QUERY_STATUS or warnings: use :func:`render`, which refuses incomplete results, and
    :func:`status_headers` for the HTTP headers.
    """
    if delimiter == "\t":
        lines = ["\t".join(_tsv_field(c.name) for c in result.columns)]
        lines.extend("\t".join(_tsv_field(_text_value(v)) for v in row) for row in result.rows)
        return ("\n".join(lines) + "\n").encode("utf-8")
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n")
    writer.writerow([c.name for c in result.columns])
    for row in result.rows:
        writer.writerow([_text_value(v) for v in row])
    return buffer.getvalue().encode("utf-8")


_TSV_FORBIDDEN = str.maketrans({"\t": " ", "\r": " ", "\n": " "})


def _tsv_field(text: str) -> str:
    return text.translate(_TSV_FORBIDDEN)


def render_json(result: ResultTable) -> bytes:
    """JSON serialization: ``{"metadata": [column...], "data": [[...]], "query_status", "infos"}``."""
    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    payload = {
        "metadata": [c.as_dict() for c in result.columns],
        "data": [[clean(v) for v in row] for row in result.rows],
        "query_status": result.status,
        "infos": [{"name": n, "value": v} for n, v in result.infos],
    }
    return json.dumps(payload, allow_nan=False).encode("utf-8")


ERROR_HEADERS: Mapping[str, str] = {"X-VO-Query-Status": "ERROR"}


def status_headers(result: ResultTable) -> dict[str, str]:
    """HTTP headers carrying the DALI QUERY_STATUS (``X-VO-Query-Status``) for every output format.

    ``X-VO-Incomplete`` lists the truncated/failed archives when the status is OVERFLOW
    because of them (header values are ASCII-sanitized and bounded in length).
    """
    headers = {"X-VO-Query-Status": result.status}
    if result.incomplete:
        text = incompleteness_summary(result.truncated, result.failed)
        headers["X-VO-Incomplete"] = " ".join(text.encode("ascii", "replace").decode("ascii").split())[:2000]
    return headers


_FORMATS: dict[str, tuple[str, str]] = {
    "votable": ("votable", VOTABLE_MEDIA_TYPE),
    "application/x-votable+xml": ("votable", VOTABLE_MEDIA_TYPE),
    "application/x-votable+xml;serialization=tabledata": ("votable", VOTABLE_MEDIA_TYPE),
    "votable/td": ("votable", VOTABLE_MEDIA_TYPE),
    "text/xml": ("votable", "text/xml"),
    "csv": ("csv", "text/csv;header=present"),
    "text/csv": ("csv", "text/csv;header=present"),
    "text/csv;header=present": ("csv", "text/csv;header=present"),
    "tsv": ("tsv", "text/tab-separated-values"),
    "text/tab-separated-values": ("tsv", "text/tab-separated-values"),
    "json": ("json", "application/json"),
    "application/json": ("json", "application/json"),
}


def resolve_format(value: str | None) -> tuple[str, str]:
    """Map a RESPONSEFORMAT/FORMAT value to (serializer key, media type)."""
    if value is None or not value.strip():
        return _FORMATS["votable"]
    key = re.sub(r"\s+", "", value).lower()
    if key not in _FORMATS:
        raise VOParameterError(
            f"Unsupported RESPONSEFORMAT {value!r}; supported: votable, csv, tsv, json (or their MIME types)")
    return _FORMATS[key]


def render(result: ResultTable, fmt: str) -> bytes:
    """Serialize with the serializer key returned by :func:`resolve_format`.

    CSV and TSV cannot carry QUERY_STATUS or WARNING INFOs, so an incomplete result
    (an archive truncated or failed) is refused with the error of
    :func:`incompleteness_error` instead of being served as if it were complete.
    """
    if fmt in {"csv", "tsv"} and result.incomplete:
        raise incompleteness_error(
            result.truncated, result.failed,
            f"{fmt.upper()} output cannot carry the incompleteness warning (use RESPONSEFORMAT=votable or json)")
    if fmt == "votable":
        return render_votable(result)
    if fmt == "csv":
        return render_csv(result)
    if fmt == "tsv":
        return render_csv(result, delimiter="\t")
    if fmt == "json":
        return render_json(result)
    raise VOParameterError(f"Unsupported format {fmt}")


# ---------------------------------------------------------------------------
# ADQL: Tokenizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # num | str | qid | id | op | eof
    text: str
    pos: int


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>'(?:[^']|'')*')
  | (?P<qid>"(?:[^"]|"")+")
  | (?P<id>[A-Za-z][A-Za-z0-9_]*)
  | (?P<op><>|!=|<=|>=|\|\||[=<>+\-*/(),.;])
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[Token]:
    """Split ADQL text into tokens; raises ADQLError on unexpected characters.

    A numeric literal immediately followed by a letter, digit or underscore (``0x10``,
    ``2abc``, ``1e5x``) is refused: numeric literals and regular identifiers are SQL-92
    "nondelimiter tokens", and a nondelimiter token "shall be followed by a delimiter token or a
    separator" (SQL-92 section 5.2, <token> and <separator>; ADQL 2.1 section 2.1.9) -- reading
    ``0x10`` as the literal 0 aliased ``x10`` would silently return a wrong value.
    ADQL 2.1 (REC 2023, section 2.1.8) has no hexadecimal literals. Delimited identifiers
    may not contain characters that XML 1.0 forbids: they become VOTable FIELD names.
    """
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            snippet = text[pos:pos + 20]
            if text[pos] == "'":
                raise ADQLError(f"Unterminated string literal at position {pos + 1}")
            raise ADQLError(f"Unexpected character {text[pos]!r} at position {pos + 1} near {snippet!r}")
        kind = m.lastgroup or ""
        if kind not in {"ws", "comment"}:
            value = m.group()
            if kind == "num" and m.end() < len(text) and (text[m.end()].isalnum() or text[m.end()] == "_"):
                junk = re.match(r"\w+", text[m.end():])
                raise ADQLError(
                    f"Numeric literal {value!r} at position {pos + 1} is immediately followed by "
                    f"{junk.group() if junk else text[m.end()]!r}; separate them with whitespace (ADQL 2.1 has no "
                    "hexadecimal or suffixed numeric literals)")
            if kind == "str":
                value = value[1:-1].replace("''", "'")
            elif kind == "qid":
                value = value[1:-1].replace('""', '"')
                if has_xml_illegal_chars(value):
                    raise ADQLError(f"The delimited identifier at position {pos + 1} contains characters that "
                                    "XML 1.0 does not allow (control characters such as U+0001)")
            tokens.append(Token(kind, value, pos))
        pos = m.end()
    tokens.append(Token("eof", "", len(text)))
    return tokens


# ---------------------------------------------------------------------------
# ADQL: Abstract Syntax Tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Col:
    parts: tuple[str, ...]
    quoted: tuple[bool, ...]
    pos: int


@dataclass(frozen=True, slots=True)
class Lit:
    value: Any
    pos: int = 0


@dataclass(frozen=True, slots=True)
class Neg:
    operand: Any


@dataclass(frozen=True, slots=True)
class Arith:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Func:
    name: str
    args: tuple[Any, ...]
    pos: int
    star: bool = False
    distinct: bool = False


@dataclass(frozen=True, slots=True)
class Cmp:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Between:
    expr: Any
    low: Any
    high: Any
    negated: bool


@dataclass(frozen=True, slots=True)
class Like:
    expr: Any
    pattern: Any
    negated: bool
    case_insensitive: bool


@dataclass(frozen=True, slots=True)
class InList:
    expr: Any
    items: tuple[Any, ...]
    negated: bool


@dataclass(frozen=True, slots=True)
class IsNull:
    expr: Any
    negated: bool


@dataclass(frozen=True, slots=True)
class And:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class Or:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class Not:
    item: Any


@dataclass(frozen=True, slots=True)
class SelectItem:
    expr: Any = None
    alias: str | None = None
    star: bool = False
    star_qualifier: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class OrderItem:
    expr: Any
    descending: bool


@dataclass(frozen=True, slots=True)
class TableRef:
    parts: tuple[str, ...]
    quoted: tuple[bool, ...]
    alias: str | None
    alias_quoted: bool


@dataclass(frozen=True, slots=True)
class SelectQuery:
    """Parsed ADQL SELECT statement."""

    distinct: bool
    top: int | None
    items: tuple[SelectItem, ...]
    table: TableRef
    where: Any | None
    order_by: tuple[OrderItem, ...]
    offset: int | None


# ---------------------------------------------------------------------------
# ADQL: Parser
# ---------------------------------------------------------------------------

_KEYWORDS = frozenset({
    "SELECT", "DISTINCT", "ALL", "TOP", "FROM", "WHERE", "AND", "OR", "NOT", "AS", "ORDER", "BY", "ASC", "DESC",
    "BETWEEN", "LIKE", "ILIKE", "IN", "IS", "NULL", "OFFSET", "GROUP", "HAVING", "JOIN", "INNER", "LEFT", "RIGHT",
    "FULL", "OUTER", "NATURAL", "CROSS", "ON", "USING", "UNION", "INTERSECT", "EXCEPT", "WITH", "CASE", "WHEN",
    "THEN", "ELSE", "END", "EXISTS",
})
_COMPARISON_OPS = frozenset({"=", "<>", "!=", "<", "<=", ">", ">="})
_UNSUPPORTED_GEOMETRY = frozenset({"BOX", "POLYGON", "REGION", "INTERSECTS", "AREA", "CENTROID", "COORD1", "COORD2",
                                   "COORDSYS", "MOC"})
_AGGREGATES = frozenset({"COUNT", "MIN", "MAX", "AVG", "SUM"})


class ADQLParser:
    """Recursive-descent parser for the supported ADQL 2.1 subset (see module docstring)."""

    def __init__(self, text: str) -> None:
        if not text or not text.strip():
            raise ADQLError("Empty QUERY")
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    # -- token helpers -------------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.kind != "eof":
            self.i += 1
        return tok

    def error(self, message: str, tok: Token | None = None) -> ADQLError:
        tok = tok or self.peek()
        where = "end of query" if tok.kind == "eof" else f"position {tok.pos + 1} ({tok.text!r})"
        return ADQLError(f"{message} at {where}")

    def at_kw(self, *words: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == "id" and tok.text.upper() in words

    def accept_kw(self, word: str) -> bool:
        if self.at_kw(word):
            self.advance()
            return True
        return False

    def expect_kw(self, word: str) -> None:
        if not self.accept_kw(word):
            raise self.error(f"Expected {word}")

    def at_op(self, *ops: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == "op" and tok.text in ops

    def accept_op(self, op: str) -> bool:
        if self.at_op(op):
            self.advance()
            return True
        return False

    def expect_op(self, op: str) -> None:
        if not self.accept_op(op):
            raise self.error(f"Expected '{op}'")

    def is_identifier(self, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == "qid" or (tok.kind == "id" and tok.text.upper() not in _KEYWORDS)

    def identifier(self) -> tuple[str, bool]:
        tok = self.peek()
        if not self.is_identifier():
            raise self.error("Expected an identifier")
        self.advance()
        return tok.text, tok.kind == "qid"

    def unsigned_int(self, what: str) -> int:
        tok = self.advance()
        if tok.kind != "num" or not tok.text.isdigit():
            raise self.error(f"{what} must be a non-negative integer", tok)
        return int(tok.text)

    # -- statement -----------------------------------------------------------

    def parse(self) -> SelectQuery:
        if self.at_kw("WITH"):
            raise self.error("Common table expressions (WITH) are not supported")
        self.expect_kw("SELECT")
        distinct = False
        if self.accept_kw("DISTINCT"):
            distinct = True
        else:
            self.accept_kw("ALL")
        top = self.unsigned_int("TOP") if self.accept_kw("TOP") else None
        items = self.parse_select_list()
        self.expect_kw("FROM")
        if self.at_op("("):
            raise self.error("Subqueries are not supported")
        table = self.parse_table_ref()
        if self.at_op(",") or self.at_kw("JOIN", "INNER", "LEFT", "RIGHT", "FULL", "NATURAL", "CROSS"):
            raise self.error("Joins are not supported; query a single table")
        where = self.parse_condition() if self.accept_kw("WHERE") else None
        if self.at_kw("GROUP", "HAVING"):
            raise self.error("GROUP BY / HAVING are not supported")
        order: list[OrderItem] = []
        if self.accept_kw("ORDER"):
            self.expect_kw("BY")
            while True:
                expr = self.parse_value()
                descending = False
                if self.accept_kw("DESC"):
                    descending = True
                else:
                    self.accept_kw("ASC")
                order.append(OrderItem(expr, descending))
                if not self.accept_op(","):
                    break
        offset = self.unsigned_int("OFFSET") if self.accept_kw("OFFSET") else None
        if self.at_kw("UNION", "INTERSECT", "EXCEPT"):
            raise self.error("Set operations (UNION/INTERSECT/EXCEPT) are not supported")
        self.accept_op(";")
        if self.peek().kind != "eof":
            raise self.error("Unexpected token")
        return SelectQuery(distinct, top, tuple(items), table, where, tuple(order), offset)

    def parse_select_list(self) -> list[SelectItem]:
        items: list[SelectItem] = []
        while True:
            if self.accept_op("*"):
                items.append(SelectItem(star=True))
            else:
                qualifier = self._qualified_star()
                if qualifier is not None:
                    items.append(SelectItem(star=True, star_qualifier=qualifier))
                else:
                    expr = self.parse_value()
                    items.append(SelectItem(expr=expr, alias=self._alias()))
            if not self.accept_op(","):
                return items

    def _qualified_star(self) -> tuple[str, ...] | None:
        j = 0
        parts: list[str] = []
        while self.is_identifier(j) and self.at_op(".", offset=j + 1):
            parts.append(self.peek(j).text)
            j += 2
        if parts and self.at_op("*", offset=j):
            for _ in range(j + 1):
                self.advance()
            return tuple(parts)
        return None

    def _alias(self) -> str | None:
        if self.accept_kw("AS"):
            return self.identifier()[0]
        if self.is_identifier():
            return self.identifier()[0]
        return None

    def parse_table_ref(self) -> TableRef:
        parts, quoted = [], []
        name, q = self.identifier()
        parts.append(name)
        quoted.append(q)
        while self.accept_op("."):
            name, q = self.identifier()
            parts.append(name)
            quoted.append(q)
        alias, alias_quoted = None, False
        if self.accept_kw("AS") or self.is_identifier():
            alias, alias_quoted = self.identifier()
        return TableRef(tuple(parts), tuple(quoted), alias, alias_quoted)

    # -- conditions ----------------------------------------------------------

    def parse_condition(self) -> Any:
        items = [self.parse_and()]
        while self.accept_kw("OR"):
            items.append(self.parse_and())
        return items[0] if len(items) == 1 else Or(tuple(items))

    def parse_and(self) -> Any:
        items = [self.parse_not()]
        while self.accept_kw("AND"):
            items.append(self.parse_not())
        return items[0] if len(items) == 1 else And(tuple(items))

    def parse_not(self) -> Any:
        if self.accept_kw("NOT"):
            return Not(self.parse_not())
        return self.parse_predicate()

    def _continues_predicate(self) -> bool:
        return (self.at_op(*_COMPARISON_OPS, "+", "-", "*", "/", "||")
                or self.at_kw("BETWEEN", "LIKE", "ILIKE", "IN", "IS")
                or (self.at_kw("NOT") and self.at_kw("BETWEEN", "LIKE", "ILIKE", "IN", offset=1)))

    def parse_predicate(self) -> Any:
        if self.at_kw("EXISTS"):
            raise self.error("EXISTS subqueries are not supported")
        if self.at_op("("):
            saved = self.i
            try:
                self.advance()
                condition = self.parse_condition()
                self.expect_op(")")
                if not self._continues_predicate():
                    return condition
            except ADQLError:
                pass
            self.i = saved
        left = self.parse_value()
        if self.at_op(*_COMPARISON_OPS):
            op = self.advance().text
            return Cmp("<>" if op == "!=" else op, left, self.parse_value())
        negated = self.accept_kw("NOT")
        if self.accept_kw("BETWEEN"):
            low = self.parse_value()
            self.expect_kw("AND")
            return Between(left, low, self.parse_value(), negated)
        if self.at_kw("LIKE", "ILIKE"):
            ci = self.advance().text.upper() == "ILIKE"
            return Like(left, self.parse_value(), negated, ci)
        if self.accept_kw("IN"):
            self.expect_op("(")
            if self.at_kw("SELECT"):
                raise self.error("IN subqueries are not supported")
            items = [self.parse_value()]
            while self.accept_op(","):
                items.append(self.parse_value())
            self.expect_op(")")
            return InList(left, tuple(items), negated)
        if not negated and self.accept_kw("IS"):
            is_not = self.accept_kw("NOT")
            self.expect_kw("NULL")
            return IsNull(left, is_not)
        raise self.error("Expected a comparison (=, <>, <, <=, >, >=, BETWEEN, LIKE, IN or IS NULL)")

    # -- value expressions ---------------------------------------------------

    def parse_value(self) -> Any:
        left = self.parse_term()
        while self.at_op("+", "-", "||"):
            op = self.advance().text
            left = Arith(op, left, self.parse_term())
        return left

    def parse_term(self) -> Any:
        left = self.parse_factor()
        while self.at_op("*", "/"):
            op = self.advance().text
            left = Arith(op, left, self.parse_factor())
        return left

    def parse_factor(self) -> Any:
        if self.accept_op("-"):
            operand = self.parse_factor()
            if isinstance(operand, Lit) and isinstance(operand.value, (int, float)) and not isinstance(operand.value, bool):
                return Lit(-operand.value, operand.pos)
            return Neg(operand)
        if self.accept_op("+"):
            return self.parse_factor()
        return self.parse_primary()

    def parse_primary(self) -> Any:
        tok = self.peek()
        if tok.kind == "num":
            self.advance()
            return Lit(_numeric_literal(tok.text, tok.pos), tok.pos)
        if tok.kind == "str":
            self.advance()
            return Lit(tok.text, tok.pos)
        if self.at_op("("):
            self.advance()
            if self.at_kw("SELECT"):
                raise self.error("Subqueries are not supported")
            expr = self.parse_value()
            self.expect_op(")")
            return expr
        if tok.kind == "id" and tok.text.upper() == "NULL":
            self.advance()
            return Lit(None, tok.pos)
        if tok.kind == "id" and tok.text.upper() == "CASE":
            raise self.error("CASE expressions are not supported")
        if tok.kind == "id" and self.at_op("(", offset=1) and tok.text.upper() not in _KEYWORDS:
            return self.parse_function()
        if self.is_identifier():
            parts, quoted = [], []
            name, q = self.identifier()
            parts.append(name)
            quoted.append(q)
            while self.at_op(".") and self.is_identifier(1):
                self.advance()
                name, q = self.identifier()
                parts.append(name)
                quoted.append(q)
            return Col(tuple(parts), tuple(quoted), tok.pos)
        raise self.error("Expected a value expression")

    def parse_function(self) -> Func:
        tok = self.advance()
        name = tok.text.upper()
        self.expect_op("(")
        if name == "COUNT" and self.accept_op("*"):
            self.expect_op(")")
            return Func(name, (), tok.pos, star=True)
        distinct = name in _AGGREGATES and self.accept_kw("DISTINCT")
        if name in _AGGREGATES and not distinct:
            self.accept_kw("ALL")
        args: list[Any] = []
        if not self.at_op(")"):
            args.append(self.parse_value())
            while self.accept_op(","):
                args.append(self.parse_value())
        self.expect_op(")")
        return Func(name, tuple(args), tok.pos, distinct=distinct)


MAX_QUERY_LENGTH = 100_000
# Deepest value/condition expression accepted: evaluation recurses once per level, and
# real queries stay far below this (1+1+...+1 with 3000 terms is 3000 levels deep).
MAX_EXPRESSION_DEPTH = 200
# Longest string an expression may build (||, LOWER, UPPER); longer ones are refused.
MAX_STRING_LENGTH = 65_536
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def _numeric_literal(text: str, pos: int) -> int | float:
    """Value of an ADQL numeric literal: exact integers must fit a 64-bit BIGINT (the
    ``long`` VOTable type, with room for the magnitude of INT64_MIN); approximate ones must
    be finite doubles."""
    if text.isdigit():
        digits = text.lstrip("0") or "0"
        if len(digits) > 19 or int(digits) > 2**63:
            raise ADQLError(f"Integer literal at position {pos + 1} is outside the 64-bit range; "
                            "write it as a floating-point literal (e.g. 1.0E30)")
        return int(digits)
    value = float(text)
    if not math.isfinite(value):
        raise ADQLError(f"Numeric literal at position {pos + 1} is outside the double-precision range")
    return value


def _check_int(value: Any) -> Any:
    """SQL BIGINT semantics: an exact integer result outside the 64-bit range is an error."""
    if isinstance(value, int) and not isinstance(value, bool) and not INT64_MIN <= value <= INT64_MAX:
        raise ADQLError("Integer overflow: a result is outside the 64-bit BIGINT range; use floating-point "
                        "arithmetic (e.g. multiply by 1.0)")
    return value


def parse_adql(text: str) -> SelectQuery:
    """Parse an ADQL query of the supported subset into a :class:`SelectQuery`.

    Expressions nested deeper than :data:`MAX_EXPRESSION_DEPTH` are refused (checked
    iteratively), so type-checking and evaluation can never exhaust the stack.
    """
    if len(text) > MAX_QUERY_LENGTH:
        raise ADQLError(f"QUERY is longer than {MAX_QUERY_LENGTH} characters")
    try:
        parsed = ADQLParser(text).parse()
    except RecursionError as exc:
        raise ADQLError("QUERY is too deeply nested") from exc
    roots = [parsed.where, *(i.expr for i in parsed.items), *(o.expr for o in parsed.order_by)]
    depth = max(expression_depth(r) for r in roots)
    if depth > MAX_EXPRESSION_DEPTH:
        raise ADQLError(f"QUERY is too deeply nested: an expression is {depth} levels deep "
                        f"(this service accepts at most {MAX_EXPRESSION_DEPTH})")
    return parsed


def ast_children(node: Any) -> tuple[Any, ...]:
    """Direct sub-expressions/conditions of an AST node (empty for leaves)."""
    if isinstance(node, (And, Or)):
        return node.items
    if isinstance(node, Not):
        return (node.item,)
    if isinstance(node, Neg):
        return (node.operand,)
    if isinstance(node, (Arith, Cmp)):
        return (node.left, node.right)
    if isinstance(node, Func):
        return node.args
    if isinstance(node, Between):
        return (node.expr, node.low, node.high)
    if isinstance(node, Like):
        return (node.expr, node.pattern)
    if isinstance(node, InList):
        return (node.expr, *node.items)
    if isinstance(node, IsNull):
        return (node.expr,)
    return ()


def expression_depth(root: Any) -> int:
    """Depth of an AST (0 for None), computed without recursion."""
    if root is None:
        return 0
    deepest = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        deepest = max(deepest, depth)
        stack.extend((child, depth + 1) for child in ast_children(node))
    return deepest


class EvaluationBudget:
    """Wall-clock budget of one query's evaluation: ``check()`` raises :class:`ADQLError` once
    ``limit_s`` seconds have elapsed (``spent_s`` of them already used by an earlier phase
    of the same query); ``tick()`` is a cheaper check for inner loops (it reads the clock
    every 256 calls); ``elapsed()`` is the time used since this budget was created."""

    __slots__ = ("_ticks", "deadline", "limit_s", "started")

    def __init__(self, limit_s: float, spent_s: float = 0.0) -> None:
        self.limit_s = limit_s
        self.started = time.perf_counter()
        self.deadline = self.started + max(0.0, limit_s - spent_s)
        self._ticks = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def check(self) -> None:
        if time.perf_counter() > self.deadline:
            raise ADQLError(f"Query evaluation exceeded this service's limit of {self.limit_s:g} s")

    def tick(self) -> None:
        self._ticks += 1
        if not self._ticks & 0xFF:
            self.check()


def _check_string_length(length: int) -> None:
    if length > MAX_STRING_LENGTH:
        raise ADQLError(f"An expression would build a string of {length} characters; this service allows at most "
                        f"{MAX_STRING_LENGTH}")


# ---------------------------------------------------------------------------
# ADQL: Binding, Type Checking & Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Point:
    ra: float
    dec: float


@dataclass(frozen=True, slots=True)
class _Circle:
    ra: float
    dec: float
    radius: float


_ICRS_FRAMES = frozenset({"", "ICRS", "ICRS GEOCENTER", "ICRS TOPOCENTER", "ICRS BARYCENTER"})


def _safe(fn: Callable[..., float]) -> Callable[..., float | None]:
    def wrapped(*args: float) -> float | None:
        try:
            value = fn(*args)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return value if not isinstance(value, float) or math.isfinite(value) else None
    return wrapped


_DECIMAL_DIGITS_LIMIT = 400  # |n| beyond this cannot change a double (|x| < 1.8e308, >= 4.9e-324)


def _decimal_places(x: float, n: Any, rounding: str) -> float:
    """ROUND/TRUNCATE ``x`` to ``n`` decimal places on its DECIMAL value, as SQL ``numeric`` does.

    A double is taken at its shortest round-tripping decimal form (``repr``: 0.29 is
    0.29, 1.005 is 1.005), so ``TRUNCATE(0.29, 2)`` = 0.29 and ``ROUND(1.005, 2)`` = 1.01
    as in PostgreSQL, where binary scaling (0.29 * 100 = 28.999999999999996) gives 0.28
    and 1.0. ``rounding`` is ROUND_HALF_UP (SQL ROUND: half away from zero) or
    ROUND_DOWN (TRUNCATE: toward zero). Integers stay integers.
    """
    places = max(-_DECIMAL_DIGITS_LIMIT, min(_DECIMAL_DIGITS_LIMIT, int(n)))
    if isinstance(x, int) and not isinstance(x, bool):
        if places >= 0:
            return x
        value = Decimal(x)
    else:
        value = Decimal(repr(float(x)))
        if not value.is_finite():
            raise ValueError("not finite")
        exponent = value.as_tuple().exponent
        assert isinstance(exponent, int)  # finite (checked above): never 'n', 'N' or 'F'
        if exponent >= -places:  # already has at most n decimals
            return float(x)
    with localcontext() as ctx:
        ctx.prec = 2 * _DECIMAL_DIGITS_LIMIT
        try:
            result = value.quantize(Decimal(1).scaleb(-places), rounding=rounding)
        except InvalidOperation as exc:
            raise ValueError(str(exc)) from exc
    return int(result) if isinstance(x, int) else float(result)


def _round(x: float, n: Any = 0) -> float:
    return _decimal_places(x, n, ROUND_HALF_UP)


def _truncate(x: float, n: Any = 0) -> float:
    return _decimal_places(x, n, ROUND_DOWN)


def _mod(a: float, b: float) -> float:
    """SQL MOD: the remainder has the sign of the dividend; integer for integer arguments."""
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
        if b == 0:
            raise ZeroDivisionError("MOD by zero")
        remainder = abs(a) % abs(b)
        return remainder if a >= 0 else -remainder
    return math.fmod(a, b)


# name -> (min args, max args, implementation, result type)
_SCALAR_FUNCTIONS: dict[str, tuple[int, int, Callable[..., Any], str]] = {
    "ABS": (1, 1, _safe(abs), "num"),
    "CEILING": (1, 1, _safe(lambda x: x if isinstance(x, int) else float(math.ceil(x))), "num"),
    "FLOOR": (1, 1, _safe(lambda x: x if isinstance(x, int) else float(math.floor(x))), "num"),
    "ROUND": (1, 2, _safe(_round), "num"),
    "TRUNCATE": (1, 2, _safe(_truncate), "num"),
    "SQRT": (1, 1, _safe(math.sqrt), "num"),
    "EXP": (1, 1, _safe(math.exp), "num"),
    "LOG": (1, 1, _safe(math.log), "num"),
    "LOG10": (1, 1, _safe(math.log10), "num"),
    "POWER": (2, 2, _safe(math.pow), "num"),
    "MOD": (2, 2, _safe(_mod), "num"),
    "PI": (0, 0, lambda: math.pi, "num"),
    "RADIANS": (1, 1, _safe(math.radians), "num"),
    "DEGREES": (1, 1, _safe(math.degrees), "num"),
    "SIN": (1, 1, _safe(math.sin), "num"),
    "COS": (1, 1, _safe(math.cos), "num"),
    "TAN": (1, 1, _safe(math.tan), "num"),
    "COT": (1, 1, _safe(lambda x: 1.0 / math.tan(x)), "num"),
    "ASIN": (1, 1, _safe(math.asin), "num"),
    "ACOS": (1, 1, _safe(math.acos), "num"),
    "ATAN": (1, 1, _safe(math.atan), "num"),
    "ATAN2": (2, 2, _safe(math.atan2), "num"),
    "LOWER": (1, 1, lambda s: s.lower(), "str"),
    "UPPER": (1, 1, lambda s: s.upper(), "str"),
}


def _is_string_literal(expr: Any) -> bool:
    return isinstance(expr, Lit) and isinstance(expr.value, str)


def _circle_args(fn: Func) -> tuple[Any, ...]:
    """CIRCLE arguments without the leading coordinate system: (ra, dec, r) or (point, r).

    Accepts the ADQL 2.0 form CIRCLE(sys, ra, dec, r), the ADQL 2.1 forms
    CIRCLE(sys, POINT(...), r) and CIRCLE(POINT(...), r), and CIRCLE(ra, dec, r).
    """
    args = fn.args
    if len(args) == 4 or (len(args) == 3 and _is_string_literal(args[0])):
        return args[1:]
    return args


def _has_column(expr: Any) -> bool:
    if isinstance(expr, Col):
        return True
    if isinstance(expr, Neg):
        return _has_column(expr.operand)
    if isinstance(expr, Arith):
        return _has_column(expr.left) or _has_column(expr.right)
    if isinstance(expr, Func):
        return any(_has_column(a) for a in expr.args)
    return False


def _has_aggregate(expr: Any) -> bool:
    if isinstance(expr, Func):
        return expr.name in _AGGREGATES or any(_has_aggregate(a) for a in expr.args)
    if isinstance(expr, Neg):
        return _has_aggregate(expr.operand)
    if isinstance(expr, Arith):
        return _has_aggregate(expr.left) or _has_aggregate(expr.right)
    return False


def _int_division_hint(expr: Any) -> str:
    def has_int_div(e: Any) -> bool:
        if isinstance(e, Arith):
            if e.op == "/" and isinstance(e.left, Lit) and isinstance(e.right, Lit) \
                    and isinstance(e.left.value, int) and isinstance(e.right.value, int):
                return True
            return has_int_div(e.left) or has_int_div(e.right)
        return isinstance(e, Neg) and has_int_div(e.operand)
    return (" (note: integer/integer is integer division in ADQL, e.g. 10/3600 = 0; write 10/3600.0)"
            if has_int_div(expr) else "")


class Binder:
    """Resolves column references against one table and type-checks/evaluates expressions."""

    def __init__(self, table: TableDef, ref: TableRef) -> None:
        self.table = table
        self.ref = ref
        # Keyed by the reference's (parts, quoted) value -- never by id(), which is reused.
        self._resolved: dict[tuple[tuple[str, ...], tuple[bool, ...]], VOColumn] = {}
        # Set while evaluating: every expression node and LIKE step counts against it.
        self.budget: EvaluationBudget | None = None

    # -- resolution ----------------------------------------------------------

    def _qualifier_ok(self, parts: Sequence[str], quoted: Sequence[bool]) -> bool:
        def eq(given: str, q: bool, actual: str) -> bool:
            return given == actual if q else given.casefold() == actual.casefold()

        if not parts:
            return True
        if self.ref.alias is not None:
            # A quoted alias (in FROM or in the reference) must match exactly; unquoted ones fold case.
            return len(parts) == 1 and eq(parts[0], quoted[0] or self.ref.alias_quoted, self.ref.alias)
        schema, short = self.table.name.split(".", 1)
        if len(parts) == 1:
            return eq(parts[0], quoted[0], short)
        if len(parts) == 2:
            return eq(parts[0], quoted[0], schema) and eq(parts[1], quoted[1], short)
        return False

    def column(self, col: Col) -> VOColumn:
        key = (col.parts, col.quoted)
        if key in self._resolved:
            return self._resolved[key]
        *qual, name = col.parts
        *qual_q, name_q = col.quoted
        if not self._qualifier_ok(qual, qual_q):
            raise ADQLError(f"Unknown table qualifier '{'.'.join(qual)}' in column reference '{'.'.join(col.parts)}'")
        found = self.table.column(name, quoted=name_q)
        if found is None:
            raise ADQLError(f"Unknown column '{name}' in table {self.table.name}. "
                            f"Columns: {', '.join(c.name for c in self.table.columns)}")
        self._resolved[key] = found
        return found

    def star_columns(self, qualifier: tuple[str, ...] | None) -> list[VOColumn]:
        if qualifier and not self._qualifier_ok(list(qualifier), [False] * len(qualifier)):
            raise ADQLError(f"Unknown table qualifier '{'.'.join(qualifier)}' in select list")
        return list(self.table.columns)

    # -- typing --------------------------------------------------------------

    def type_of(self, expr: Any, *, allow_aggregate: bool = False) -> str:
        """Return 'num', 'str', 'null', 'point' or 'circle'; raises ADQLError on type errors."""
        if isinstance(expr, Lit):
            if expr.value is None:
                return "null"
            return "str" if isinstance(expr.value, str) else "num"
        if isinstance(expr, Col):
            return "str" if self.column(expr).is_string else "num"
        if isinstance(expr, Neg):
            if self.type_of(expr.operand, allow_aggregate=allow_aggregate) not in {"num", "null"}:
                raise ADQLError("Unary minus requires a numeric operand")
            return "num"
        if isinstance(expr, Arith):
            lt = self.type_of(expr.left, allow_aggregate=allow_aggregate)
            rt = self.type_of(expr.right, allow_aggregate=allow_aggregate)
            if expr.op == "||":
                if {lt, rt} - {"str", "null"}:
                    raise ADQLError("The || operator requires string operands")
                return "str"
            if {lt, rt} - {"num", "null"}:
                raise ADQLError(f"Operator {expr.op} requires numeric operands")
            return "num"
        if isinstance(expr, Func):
            return self._func_type(expr, allow_aggregate)
        raise ADQLError(f"Unsupported expression {expr!r}")

    def _frame(self, arg: Any, fn: str) -> None:
        if not (isinstance(arg, Lit) and isinstance(arg.value, str)):
            raise ADQLError(f"The first argument of {fn} must be a coordinate system string such as 'ICRS'")
        if arg.value.strip().upper() not in _ICRS_FRAMES:
            raise ADQLError(f"Unsupported coordinate system {arg.value!r} in {fn}; only ICRS is supported")

    def _func_type(self, fn: Func, allow_aggregate: bool) -> str:
        name = fn.name
        if name in _AGGREGATES:
            if not allow_aggregate:
                raise ADQLError(f"Aggregate function {name} is only allowed in the select list")
            if fn.star:
                return "num"
            if len(fn.args) != 1:
                raise ADQLError(f"{name} takes exactly one argument")
            arg_type = self.type_of(fn.args[0])
            if arg_type in {"point", "circle"}:
                raise ADQLError(f"{name} cannot aggregate geometry values")
            if name in {"AVG", "SUM"} and arg_type not in {"num", "null"}:
                raise ADQLError(f"{name} requires a numeric argument")
            return "num" if name in {"COUNT", "AVG", "SUM"} else arg_type
        if name == "POINT":
            args = fn.args
            if len(args) == 3:
                self._frame(args[0], "POINT")
                args = args[1:]
            if len(args) != 2:
                raise ADQLError("POINT takes (coordsys, ra, dec) or (ra, dec)")
            for a in args:
                if self.type_of(a) not in {"num", "null"}:
                    raise ADQLError("POINT coordinates must be numeric")
            return "point"
        if name == "CIRCLE":
            if len(fn.args) in (3, 4) and _is_string_literal(fn.args[0]):
                self._frame(fn.args[0], "CIRCLE")
            args = _circle_args(fn)
            if len(args) == 2 and self.type_of(args[0]) == "point":
                args = args[1:]
            elif len(args) != 3:
                raise ADQLError("CIRCLE takes (coordsys, ra, dec, radius), (ra, dec, radius), (coordsys, point, radius) "
                                "or (point, radius)")
            for a in args:
                if self.type_of(a) not in {"num", "null"}:
                    raise ADQLError("CIRCLE centre and radius must be numeric")
            return "circle"
        if name == "CONTAINS":
            if len(fn.args) != 2:
                raise ADQLError("CONTAINS takes two geometry arguments")
            first, second = (self.type_of(a) for a in fn.args)
            if first != "point" or second != "circle":
                raise ADQLError("Only CONTAINS(POINT(...), CIRCLE(...)) is supported")
            return "num"
        if name == "DISTANCE":
            if len(fn.args) == 2:
                if any(self.type_of(a) != "point" for a in fn.args):
                    raise ADQLError("DISTANCE takes two POINTs or four numeric coordinates")
            elif len(fn.args) == 4:
                if any(self.type_of(a) not in {"num", "null"} for a in fn.args):
                    raise ADQLError("DISTANCE(ra1, dec1, ra2, dec2) requires numeric coordinates")
            else:
                raise ADQLError("DISTANCE takes two POINTs or four numeric coordinates")
            return "num"
        if name in _UNSUPPORTED_GEOMETRY:
            raise ADQLError(f"Geometry function {name} is not supported; use CONTAINS(POINT, CIRCLE) or DISTANCE")
        if name not in _SCALAR_FUNCTIONS:
            raise ADQLError(f"Unknown or unsupported function {name}")
        lo, hi, _impl, rtype = _SCALAR_FUNCTIONS[name]
        if not lo <= len(fn.args) <= hi:
            arity = str(lo) if lo == hi else f"{lo}-{hi}"
            raise ADQLError(f"{name} takes {arity} argument(s), got {len(fn.args)}")
        want = "str" if rtype == "str" else "num"
        for a in fn.args:
            if self.type_of(a, allow_aggregate=allow_aggregate) not in {want, "null"}:
                raise ADQLError(f"{name} requires {'string' if want == 'str' else 'numeric'} arguments")
        return rtype

    def check_condition(self, cond: Any) -> None:
        """Type-check a WHERE condition tree."""
        if isinstance(cond, (And, Or)):
            for item in cond.items:
                self.check_condition(item)
            return
        if isinstance(cond, Not):
            self.check_condition(cond.item)
            return
        if isinstance(cond, Cmp):
            self._comparable(cond.left, cond.right, cond.op)
            return
        if isinstance(cond, Between):
            self._comparable(cond.expr, cond.low, "BETWEEN")
            self._comparable(cond.expr, cond.high, "BETWEEN")
            return
        if isinstance(cond, Like):
            for e in (cond.expr, cond.pattern):
                if self.type_of(e) not in {"str", "null"}:
                    raise ADQLError("LIKE requires string operands")
            return
        if isinstance(cond, InList):
            for item in cond.items:
                self._comparable(cond.expr, item, "IN")
            return
        if isinstance(cond, IsNull):
            if self.type_of(cond.expr) in {"point", "circle"}:
                raise ADQLError("IS NULL cannot be applied to geometry values")
            return
        raise ADQLError("Invalid condition")

    def _comparable(self, left: Any, right: Any, op: str) -> None:
        lt, rt = self.type_of(left), self.type_of(right)
        if "point" in (lt, rt) or "circle" in (lt, rt):
            raise ADQLError("Geometry values cannot be compared; use CONTAINS(...) = 1 or DISTANCE(...)")
        if "null" not in (lt, rt) and lt != rt:
            raise ADQLError(f"Cannot compare {'a string' if lt == 'str' else 'a number'} with "
                            f"{'a string' if rt == 'str' else 'a number'} ({op})")

    # -- evaluation ----------------------------------------------------------

    def value(self, expr: Any, row: Mapping[str, Any]) -> Any:
        if self.budget is not None:
            self.budget.tick()
        if isinstance(expr, Lit):
            return expr.value
        if isinstance(expr, Col):
            return row.get(self.column(expr).name)
        if isinstance(expr, Neg):
            v = self.value(expr.operand, row)
            return None if v is None else _check_int(-v)
        if isinstance(expr, Arith):
            a, b = self.value(expr.left, row), self.value(expr.right, row)
            if a is None or b is None:
                return None
            if expr.op == "||":
                a, b = str(a), str(b)
                _check_string_length(len(a) + len(b))
                return a + b
            if expr.op == "+":
                return _finite(_check_int(a + b))
            if expr.op == "-":
                return _finite(_check_int(a - b))
            if expr.op == "*":
                return _finite(_check_int(a * b))
            if b == 0:
                return None
            if isinstance(a, int) and isinstance(b, int):
                quotient = abs(a) // abs(b)  # SQL integer division truncates toward zero
                return _check_int(quotient if (a >= 0) == (b >= 0) else -quotient)
            return _finite(a / b)
        if isinstance(expr, Func):
            return self._call(expr, row)
        raise ADQLError(f"Cannot evaluate {expr!r}")

    def _call(self, fn: Func, row: Mapping[str, Any]) -> Any:
        name = fn.name
        if name == "POINT":
            args = fn.args[1:] if len(fn.args) == 3 else fn.args
            ra, dec = (self.value(a, row) for a in args)
            return None if ra is None or dec is None else _Point(float(ra), float(dec))
        if name == "CIRCLE":
            args = _circle_args(fn)
            if len(args) == 2:
                centre, radius = self.value(args[0], row), self.value(args[1], row)
                if centre is None or radius is None:
                    return None
                return _Circle(centre.ra, centre.dec, float(radius))
            ra, dec, radius = (self.value(a, row) for a in args)
            return None if None in (ra, dec, radius) else _Circle(float(ra), float(dec), float(radius))
        if name == "CONTAINS":
            point, circle = (self.value(a, row) for a in fn.args)
            if point is None or circle is None:
                return None
            return 1 if haversine_arcsec(point.ra, point.dec, circle.ra, circle.dec) <= circle.radius * 3600.0 else 0
        if name == "DISTANCE":
            if len(fn.args) == 2:
                p1, p2 = (self.value(a, row) for a in fn.args)
                if p1 is None or p2 is None:
                    return None
                return haversine_arcsec(p1.ra, p1.dec, p2.ra, p2.dec) / 3600.0
            vals = [self.value(a, row) for a in fn.args]
            if any(v is None for v in vals):
                return None
            return haversine_arcsec(*(float(v) for v in vals)) / 3600.0
        _lo, _hi, impl, rtype = _SCALAR_FUNCTIONS[name]
        arg_values = [self.value(a, row) for a in fn.args]
        if any(a is None for a in arg_values):
            return None
        result = impl(*arg_values)
        if rtype == "str" and isinstance(result, str):
            _check_string_length(len(result))  # UPPER('ß') is 'SS': case mapping can grow a string
        return result

    def test(self, cond: Any, row: Mapping[str, Any]) -> bool | None:
        """Three-valued (SQL) truth of a condition for one row."""
        if isinstance(cond, And):
            result: bool | None = True
            for item in cond.items:
                v = self.test(item, row)
                if v is False:
                    return False
                if v is None:
                    result = None
            return result
        if isinstance(cond, Or):
            result = False
            for item in cond.items:
                v = self.test(item, row)
                if v is True:
                    return True
                if v is None:
                    result = None
            return result
        if isinstance(cond, Not):
            v = self.test(cond.item, row)
            return None if v is None else not v
        if isinstance(cond, Cmp):
            return _compare(cond.op, self.value(cond.left, row), self.value(cond.right, row))
        if isinstance(cond, Between):
            v, lo, hi = self.value(cond.expr, row), self.value(cond.low, row), self.value(cond.high, row)
            inside = _and3(_compare(">=", v, lo), _compare("<=", v, hi))
            return None if inside is None else (inside != cond.negated)
        if isinstance(cond, Like):
            v, pattern = self.value(cond.expr, row), self.value(cond.pattern, row)
            if v is None or pattern is None:
                return None
            matched = like_match(str(v), str(pattern), case_insensitive=cond.case_insensitive, budget=self.budget)
            return matched != cond.negated
        if isinstance(cond, InList):
            v = self.value(cond.expr, row)
            if v is None:
                return None
            saw_null = False
            for item in cond.items:
                other = self.value(item, row)
                if other is None:
                    saw_null = True
                elif v == other:
                    return not cond.negated
            return None if saw_null else cond.negated
        if isinstance(cond, IsNull):
            return (self.value(cond.expr, row) is None) != cond.negated
        raise ADQLError("Invalid condition")


def _finite(value: Any) -> Any:
    """Floating-point overflow (inf) or NaN is a domain error and yields NULL."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _and3(a: bool | None, b: bool | None) -> bool | None:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _compare(op: str, a: Any, b: Any) -> bool | None:
    if a is None or b is None:
        return None
    if op == "=":
        return a == b
    if op == "<>":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise ADQLError(f"Unknown comparison operator {op}")


def like_match(value: str, pattern: str, *, case_insensitive: bool = False,
               budget: EvaluationBudget | None = None) -> bool:
    """SQL/ADQL ``LIKE``: ``%`` matches any sequence, ``_`` exactly one character (the ADQL
    grammar's ``<like_predicate>`` has no ESCAPE clause); ``ILIKE`` compares case-folded strings.

    Greedy two-pointer matching that backtracks only to the last ``%`` (the classic
    wildcard algorithm, cf. K. J. Krauss, "Matching Wildcards: An Algorithm", Dr. Dobb's
    Journal, 2008): each restart after a mismatch resumes one character further in
    ``value`` and rescans at most the pattern segment after the last ``%``, so the time
    is O(len(value)^2 + len(pattern)) and the extra space O(1) -- unlike a regex
    translation, whose backtracking is exponential in the number of ``%``
    (``'%_%_%_%_%Q'`` took 30 s on a 278-character value). A pattern needing more
    characters than ``value`` has is rejected at once (a shortcut, not a bound). With a
    ``budget``, its deadline is checked every 4096 steps, so the quadratic worst case
    (values and patterns up to :data:`MAX_STRING_LENGTH`) cannot overrun it.
    """
    if case_insensitive:
        value, pattern = value.casefold(), pattern.casefold()
    pattern = re.sub("%+", "%", pattern)
    if len(pattern) - pattern.count("%") > len(value):
        return False
    i = j = 0
    star = -1
    mark = 0
    n, m = len(value), len(pattern)
    steps = 0
    while i < n:
        steps += 1
        if budget is not None and not steps & 0xFFF:
            budget.check()
        if j < m and pattern[j] != "%" and (pattern[j] == "_" or pattern[j] == value[i]):
            i += 1
            j += 1
        elif j < m and pattern[j] == "%":
            star, mark = j, i
            j += 1
        elif star >= 0:
            mark += 1
            i, j = mark, star + 1
        else:
            return False
    while j < m and pattern[j] == "%":
        j += 1
    return j == m


# ---------------------------------------------------------------------------
# ADQL: Query Planning (cone extraction, catalog push-down) & Execution
# ---------------------------------------------------------------------------


def _conjuncts(cond: Any) -> list[Any]:
    if cond is None:
        return []
    if isinstance(cond, And):
        out: list[Any] = []
        for item in cond.items:
            out.extend(_conjuncts(item))
        return out
    return [cond]


def _is_one(expr: Any) -> bool:
    return isinstance(expr, Lit) and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool) and expr.value == 1


def _point_args(binder: Binder, expr: Any) -> tuple[Any, Any] | None:
    if isinstance(expr, Func) and expr.name == "POINT":
        return (expr.args[-2], expr.args[-1])
    return None


def _is_position_columns(binder: Binder, ra_expr: Any, dec_expr: Any) -> bool:
    return (isinstance(ra_expr, Col) and isinstance(dec_expr, Col)
            and binder.column(ra_expr).ucd == "pos.eq.ra;meta.main"
            and binder.column(dec_expr).ucd == "pos.eq.dec;meta.main")


def _const(binder: Binder, expr: Any) -> float | None:
    if _has_column(expr):
        return None
    value = binder.value(expr, {})
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass(frozen=True, slots=True)
class Cone:
    """A cone constraint extracted from an ADQL WHERE clause (degrees, ICRS)."""

    ra: float
    dec: float
    radius: float
    radius_expr: Any = None


def _make_cone(values: Sequence[float | None], radius_expr: Any) -> Cone | None:
    ra, dec, radius = values
    if ra is None or dec is None or radius is None:
        return None
    return Cone(ra, dec, radius, radius_expr)


def _cone_of_term(binder: Binder, term: Any) -> Cone | None:
    """The cone expressed by one WHERE conjunct, if it is ``1=CONTAINS(POINT(ra,dec), CIRCLE(...))``
    or ``DISTANCE(POINT(ra,dec), POINT(a,b)) < r`` with constant centre and radius."""
    if not isinstance(term, Cmp):
        return None
    # 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', a, b, r))
    if term.op == "=":
        contains = term.right if _is_one(term.left) else term.left if _is_one(term.right) else None
        if isinstance(contains, Func) and contains.name == "CONTAINS":
            point = _point_args(binder, contains.args[0])
            circle = contains.args[1]
            if point and _is_position_columns(binder, *point) and isinstance(circle, Func) and circle.name == "CIRCLE":
                cargs = _circle_args(circle)
                if len(cargs) == 2:
                    centre = _point_args(binder, cargs[0])
                    values = [_const(binder, a) for a in (*(centre or (None, None)), cargs[1])]
                else:
                    values = [_const(binder, a) for a in cargs]
                return _make_cone(values, cargs[-1])
            return None
    # DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', a, b)) < r   (or r > DISTANCE(...))
    if term.op in {"<", "<="} and isinstance(term.left, Func) and term.left.name == "DISTANCE":
        dist, bound = term.left, term.right
    elif term.op in {">", ">="} and isinstance(term.right, Func) and term.right.name == "DISTANCE":
        dist, bound = term.right, term.left
    else:
        return None
    if len(dist.args) == 2:
        p1, p2 = _point_args(binder, dist.args[0]), _point_args(binder, dist.args[1])
        if not p1 or not p2:
            return None
    else:
        p1, p2 = (dist.args[0], dist.args[1]), (dist.args[2], dist.args[3])
    if _is_position_columns(binder, *p1):
        centre = p2
    elif _is_position_columns(binder, *p2):
        centre = p1
    else:
        return None
    return _make_cone([_const(binder, centre[0]), _const(binder, centre[1]), _const(binder, bound)], bound)


def cone_terms(binder: Binder, where: Any) -> list[tuple[Any, Cone]]:
    """Every top-level WHERE conjunct that is a cone constraint, with its cone."""
    out = []
    for term in _conjuncts(where):
        cone = _cone_of_term(binder, term)
        if cone is not None:
            out.append((term, cone))
    return out


def extract_cone(binder: Binder, where: Any) -> Cone | None:
    """Find the tightest top-level ``1=CONTAINS(POINT(ra,dec), CIRCLE(...))`` or ``DISTANCE(...) < r`` term."""
    cones = [cone for _term, cone in cone_terms(binder, where)]
    return min(cones, key=lambda c: c.radius) if cones else None


def extract_catalogs(binder: Binder, where: Any) -> list[str] | None:
    """Catalog keys implied by top-level ``catalog = '...'`` / ``catalog IN (...)`` terms (None = unconstrained)."""
    allowed: set[str] | None = None
    for term in _conjuncts(where):
        names: set[str] | None = None
        if isinstance(term, Cmp) and term.op == "=":
            col, lit = (term.left, term.right) if isinstance(term.left, Col) else (term.right, term.left)
            if isinstance(col, Col) and isinstance(lit, Lit) and isinstance(lit.value, str) \
                    and binder.column(col).name == "catalog":
                names = {lit.value}
        elif isinstance(term, InList) and not term.negated and isinstance(term.expr, Col) \
                and binder.column(term.expr).name == "catalog" \
                and all(isinstance(i, Lit) and isinstance(i.value, str) for i in term.items):
            names = {i.value for i in term.items}
        if names is not None:
            allowed = names if allowed is None else allowed & names
    return None if allowed is None else sorted(allowed)


def referenced_columns(binder: Binder, node: Any) -> set[str]:
    """Names of the table columns referenced by an expression or condition tree."""
    if node is None or isinstance(node, Lit):
        return set()
    if isinstance(node, Col):
        return {binder.column(node).name}
    if isinstance(node, (And, Or)):
        children: tuple[Any, ...] = node.items
    elif isinstance(node, Not):
        children = (node.item,)
    elif isinstance(node, Neg):
        children = (node.operand,)
    elif isinstance(node, (Arith, Cmp)):
        children = (node.left, node.right)
    elif isinstance(node, Func):
        children = node.args
    elif isinstance(node, Between):
        children = (node.expr, node.low, node.high)
    elif isinstance(node, Like):
        children = (node.expr, node.pattern)
    elif isinstance(node, InList):
        children = (node.expr, *node.items)
    elif isinstance(node, IsNull):
        children = (node.expr,)
    else:
        return set()
    out: set[str] = set()
    for child in children:
        out |= referenced_columns(binder, child)
    return out


# Columns whose value is the same for every row of one archive: a condition on them only
# selects archives (``wavelength`` is the registry's spectral regime of the catalog).
CATALOG_LEVEL_COLUMNS = frozenset({"catalog", "wavelength"})
# Columns computed across all queried archives: restricting the archives changes them.
GROUP_COLUMNS = frozenset({"group_id", "group_n_catalogs", "group_catalogs"})


def catalog_level_terms(binder: Binder, where: Any, *, constants: bool = True) -> list[Any]:
    """Top-level WHERE conjuncts that reference only ``catalog``/``wavelength`` (or, when
    ``constants``, no column at all)."""
    out = []
    for term in _conjuncts(where):
        refs = referenced_columns(binder, term)
        if refs <= CATALOG_LEVEL_COLUMNS and (refs or constants):
            out.append(term)
    return out


def plan_catalogs(binder: Binder, where: Any, registry: CatalogRegistry) -> list[str] | None:
    """Enabled archives whose rows can satisfy the catalog-level WHERE conjuncts (None = all).

    Each ``catalog``/``wavelength``-only conjunct (``catalog = 'x'``, ``catalog IN (...)``,
    ``catalog <> 'x'``, ``wavelength IN ('radio', 'xray')``, ...) is evaluated with that
    archive's key and spectral regime; an archive for which any is not TRUE cannot
    contribute a row and need not be queried. The caller must not use this push-down
    when the query refers to :data:`GROUP_COLUMNS`, whose values depend on every archive.
    """
    # Column-free terms are left to the row filter: evaluating them here would repeat them
    # once per archive for no gain (their value is the same for every archive).
    terms = catalog_level_terms(binder, where, constants=False)
    if not terms:
        return None
    enabled = registry.enabled_catalogs()
    keep = sorted(name for name, cat in enabled.items()
                  if all(binder.test(t, {"catalog": name, "wavelength": cat.wavelength}) is True for t in terms))
    return None if len(keep) == len(enabled) else keep


def _expr_label(expr: Any) -> str:
    if isinstance(expr, Col):
        return expr.parts[-1]
    if isinstance(expr, Func):
        return expr.name.lower()
    return "expr"


@functools.lru_cache(maxsize=512)
def _valid_ucd(ucd: str) -> bool:
    from astropy.io.votable.ucd import check_ucd

    return bool(check_ucd(ucd, check_controlled_vocabulary=True))


# Secondary UCD1+ words describing an aggregate of a column (UCD1+ controlled vocabulary).
_AGGREGATE_UCD_WORDS = {"MIN": "stat.min", "MAX": "stat.max", "AVG": "stat.mean", "SUM": "arith.sum"}


def derived_ucd(base: str | None, word: str) -> str | None:
    """UCD of an aggregate of a column: the column's UCD without ``meta.main`` plus the
    secondary word (e.g. ``pos.angDistance;stat.mean``); None when the vocabulary forbids it."""
    if not base:
        return None
    parts = [p.strip() for p in base.split(";") if p.strip() and p.strip() != "meta.main"]
    if not parts:
        return None
    if word not in parts:
        parts.append(word)
    ucd = ";".join(parts)
    return ucd if _valid_ucd(ucd) else None


def _output_column(binder: Binder, expr: Any, name: str) -> VOColumn:
    if isinstance(expr, Col):
        return replace(binder.column(expr), name=name)
    kind = binder.type_of(expr, allow_aggregate=True)
    if isinstance(expr, Func) and expr.name == "DISTANCE":
        return VOColumn(name, "double", "Angular distance computed by DISTANCE().", unit="deg", ucd="pos.angDistance")
    if isinstance(expr, Func) and expr.name == "COUNT":
        return VOColumn(name, "long", "Row count.", ucd="meta.number")
    if isinstance(expr, Func) and expr.name in _AGGREGATE_UCD_WORDS and expr.args and isinstance(expr.args[0], Col):
        base = binder.column(expr.args[0])
        ucd = derived_ucd(base.ucd, _AGGREGATE_UCD_WORDS[expr.name])
        description = f"{expr.name} of {base.name}: {base.description}"
        if expr.name in {"MIN", "MAX"}:
            return replace(base, name=name, description=description, ucd=ucd, principal=False, indexed=False,
                           std=False, verb=3)
        datatype = "long" if expr.name == "SUM" and _is_integer_expr(binder, expr) else "double"
        return VOColumn(name, datatype, description, unit=base.unit, ucd=ucd)
    if kind == "str":
        return VOColumn(name, "char", "Computed string value.", arraysize="*")
    if kind == "null":
        return VOColumn(name, "char", "NULL literal.", arraysize="*")
    if _is_integer_expr(binder, expr):
        return VOColumn(name, "long", "Computed integer value.")
    return VOColumn(name, "double", "Computed numeric value.")


def _is_integer_expr(binder: Binder, expr: Any) -> bool:
    """True when SQL typing makes ``expr`` an exact integer (integer literals/columns, + - * /)."""
    if isinstance(expr, Lit):
        return isinstance(expr.value, int) and not isinstance(expr.value, bool)
    if isinstance(expr, Col):
        return binder.column(expr).datatype in {"short", "int", "long"}
    if isinstance(expr, Neg):
        return _is_integer_expr(binder, expr.operand)
    if isinstance(expr, Arith) and expr.op in {"+", "-", "*", "/"}:
        return _is_integer_expr(binder, expr.left) and _is_integer_expr(binder, expr.right)
    if isinstance(expr, Func) and expr.name in {"SUM", "ABS", "CEILING", "FLOOR", "ROUND", "TRUNCATE"} and expr.args:
        return _is_integer_expr(binder, expr.args[0])
    if isinstance(expr, Func) and expr.name == "MOD" and len(expr.args) == 2:
        return _is_integer_expr(binder, expr.args[0]) and _is_integer_expr(binder, expr.args[1])
    return False


def _unique(names: list[str]) -> list[str]:
    """Output column names made unique (case-insensitively, as ADQL regular identifiers compare).

    The first column with a name keeps it; a repeated name gets the first ``_2``, ``_3``,
    ... suffix not used by ANY column, so ``name, name, name AS name_2`` gives
    ``name, name_3, name_2``.
    """
    used = {name.casefold() for name in names}
    seen: set[str] = set()
    out = []
    for name in names:
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            out.append(name)
            continue
        n = 2
        while f"{name}_{n}".casefold() in used:
            n += 1
        candidate = f"{name}_{n}"
        used.add(candidate.casefold())
        out.append(candidate)
    return out


def _aggregate(binder: Binder, fn: Func, rows: Sequence[Mapping[str, Any]]) -> Any:
    if fn.star:
        return len(rows)
    values = []
    for row in rows:
        if binder.budget is not None:
            binder.budget.check()
        value = binder.value(fn.args[0], row)
        if value is not None:
            values.append(value)
    if fn.distinct:
        values = list(dict.fromkeys(values))
    if fn.name == "COUNT":
        return len(values)
    if not values:
        return None
    if fn.name == "MIN":
        return min(values)
    if fn.name == "MAX":
        return max(values)
    if fn.name == "SUM":
        return _finite(_check_int(sum(values)))
    return _finite(math.fsum(values) / len(values))


def _eval_aggregate_expr(binder: Binder, expr: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    if isinstance(expr, Func) and expr.name in _AGGREGATES:
        return _aggregate(binder, expr, rows)
    if isinstance(expr, Lit):
        return expr.value
    if isinstance(expr, Neg):
        v = _eval_aggregate_expr(binder, expr.operand, rows)
        return None if v is None else -v
    if isinstance(expr, Arith):
        a = Lit(_eval_aggregate_expr(binder, expr.left, rows))
        b = Lit(_eval_aggregate_expr(binder, expr.right, rows))
        return binder.value(Arith(expr.op, a, b), {})
    if isinstance(expr, Func):
        return binder.value(Func(expr.name, tuple(Lit(_eval_aggregate_expr(binder, a, rows)) for a in expr.args), expr.pos), {})
    raise ADQLError("Aggregate queries may only combine aggregates and constants (GROUP BY is not supported)")


def _sort_key(value: Any) -> tuple[int, Any]:
    return (1, 0) if value is None else (0, value)


@dataclass(frozen=True, slots=True)
class _OrderKey:
    """Sort key over (source row, output row): an output column index or an expression on the source row.

    NULLs sort after every value, i.e. last for ASC and first for DESC (PostgreSQL semantics).
    """

    output_index: int | None = None
    binder: Binder | None = None
    expr: Any = None

    def __call__(self, row: Mapping[str, Any], out: tuple[Any, ...]) -> tuple[int, Any]:
        if self.output_index is not None:
            return _sort_key(out[self.output_index])
        assert self.binder is not None
        return _sort_key(self.binder.value(self.expr, row))


class RankIncompleteError(Exception):
    """A "nearest n" answer on a truncated cone would include rows beyond the completeness radius."""


def _entry_key(index: int, entry: tuple[tuple[Any, ...], Any, Any]) -> Any:
    return entry[0][index]


# Estimated serialized size of one value beyond its characters (VOTable <TD></TD>, CSV/JSON
# separators and quotes) and of a non-string value (the longest repr of a double is 24).
_VALUE_OVERHEAD_BYTES = 16
_NUMBER_BYTES = 24


class _ResultSize:
    """Running size estimate of a result's values, bounded by ``limit`` bytes (None = unbounded)."""

    __slots__ = ("limit", "total")

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.total = 0

    def add(self, values: Sequence[Any]) -> None:
        if self.limit is None:
            return
        for value in values:
            self.total += (len(value) + _VALUE_OVERHEAD_BYTES) if isinstance(value, str) else _NUMBER_BYTES
        if self.total > self.limit:
            raise ResultTooLargeError(result_too_large_message(self.limit))


def result_too_large_message(limit: int) -> str:
    return (f"The query result would exceed this service's limit of {limit} bytes (VO_MAX_RESULT_BYTES); "
            "select fewer or shorter columns, or fewer rows (TOP, MAXREC, a smaller cone)")


def _evaluate(
    binder: Binder,
    parsed: SelectQuery,
    source_rows: list[dict[str, Any]],
    out_exprs: Sequence[Any],
    order_keys: Sequence[tuple[_OrderKey, bool]],
    aggregate_query: bool,
    budget_s: float,
    rank_limit: tuple[int, float] | None = None,
    *,
    spent_s: float = 0.0,
    max_bytes: int | None = None,
) -> list[tuple[Any, ...]]:
    """Filter, project, order and de-duplicate (CPU-bound; run in a worker thread).

    The ``budget_s`` deadline (of which planning already used ``spent_s``) is checked on
    every row, inside expression evaluation and LIKE matching (:class:`EvaluationBudget`),
    while sort keys are computed (they are precomputed once per row) and in aggregates.
    The values projected for the kept rows may total at most ``max_bytes`` (estimated as
    string lengths plus a per-value overhead), else :class:`ResultTooLargeError`.
    ``rank_limit`` = ``(n, radius_arcsec)`` requires the n-th row after ordering to have
    ``separation`` < radius, else :class:`RankIncompleteError` (see :func:`execute_adql`).
    Raises :class:`ADQLError` when evaluation exceeds the budget or produces an integer
    outside the 64-bit range.
    """
    budget = EvaluationBudget(budget_s, spent_s)
    binder.budget = budget
    size = _ResultSize(max_bytes)
    try:
        if parsed.where is not None:
            kept = []
            for row in source_rows:
                budget.check()
                if binder.test(parsed.where, row) is True:
                    kept.append(row)
            source_rows = kept
        if aggregate_query:
            outputs = [tuple(_eval_aggregate_expr(binder, e, source_rows) for e in out_exprs)]
            size.add(outputs[0])
        else:
            entries: list[tuple[tuple[Any, ...], Mapping[str, Any], tuple[Any, ...]]] = []
            for row in source_rows:
                budget.check()
                out = tuple(binder.value(e, row) for e in out_exprs)
                size.add(out)
                entries.append((tuple(key(row, out) for key, _descending in order_keys), row, out))
            for index in range(len(order_keys) - 1, -1, -1):  # stable sorts, last key first
                entries.sort(key=functools.partial(_entry_key, index), reverse=order_keys[index][1])
            budget.check()
            if rank_limit is not None:
                needed, radius = rank_limit
                if needed > 0:
                    if len(entries) < needed:
                        raise RankIncompleteError(f"only {len(entries)} rows were fetched, {needed} are needed")
                    sep = entries[needed - 1][1].get("separation")
                    if sep is None or not sep < radius:
                        raise RankIncompleteError(f"row {needed} lies at {sep} arcsec")
            outputs = [out for _keys, _row, out in entries]
        for out in outputs:
            for value in out:
                _check_int(value)
        if parsed.distinct:
            outputs = list(dict.fromkeys(outputs))
        if parsed.offset:
            outputs = outputs[parsed.offset:]
        if parsed.top is not None:
            outputs = outputs[:parsed.top]
        return outputs
    except RecursionError as exc:
        raise ADQLError("QUERY is too deeply nested") from exc
    finally:
        binder.budget = None


async def execute_adql(
    query: str,
    *,
    service: Any | None,
    registry: CatalogRegistry | None = None,
    maxrec: int | None = None,
    config: VOConfig | None = None,
) -> ResultTable:
    """Parse, plan and execute an ADQL query against the published tables.

    ``maxrec`` (DALI MAXREC) caps the rows returned; exceeding it sets QUERY_STATUS
    OVERFLOW. ``MAXREC=0`` returns metadata only, with OVERFLOW (DALI 1.1: "metadata,
    no results, and an overflow indicator"), without querying any archive; ``TOP 0``
    also returns metadata only, with status OK.

    On ``astrosearch.matches``, archives that the ``catalog``/``wavelength`` conditions
    exclude are not queried, unless the query refers to a ``group_*`` column (their
    values depend on every archive, and a WHERE clause must not change the values of the
    rows it keeps). When an archive failed or held more rows in the cone than it was
    asked for (:func:`per_archive_row_limit`), a query that needs every row of the cone
    (aggregates, conditions on other columns, ORDER BY with TOP/OFFSET) fails with the
    error of :func:`incompleteness_error`; other queries return their rows with
    QUERY_STATUS OVERFLOW and WARNING INFOs. Exception: ``TOP n`` ordered by ascending
    ``separation`` (or DISTANCE to the cone centre), with no failed archive, no DISTINCT
    and no ``group_*`` condition or ordering, is answered when its last needed row lies
    nearer than the radius to which every truncated archive is complete -- the rows are
    then exact (QUERY_STATUS OK unless ``group_*`` values, which may involve missing
    rows, are selected). Raises :class:`ADQLError`, :class:`VOParameterError`,
    :class:`IncompleteResultError` or :class:`UpstreamError`.
    """
    try:
        return await _execute_adql(query, service=service, registry=registry, maxrec=maxrec, config=config)
    except RecursionError as exc:
        raise ADQLError("QUERY is too deeply nested") from exc


# Most output expressions one query may select (after ``*`` expansion): a service limit,
# far above the widest published table (astrosearch.matches, 20 columns).
MAX_SELECT_ITEMS = 500


def _ranks_by_separation(binder: Binder, order_keys: Sequence[tuple[_OrderKey, bool]], out_exprs: Sequence[Any],
                         centre_ra: float, centre_dec: float) -> bool:
    """True when the first ORDER BY key is ascending ``separation`` or DISTANCE to the cone centre."""
    if not order_keys:
        return False
    key, descending = order_keys[0]
    if descending:
        return False
    expr = out_exprs[key.output_index] if key.output_index is not None else key.expr
    if isinstance(expr, Col):
        return binder.column(expr).name == "separation"
    if isinstance(expr, Func) and expr.name == "DISTANCE":
        cone = _cone_of_term(binder, Cmp("<", expr, Lit(1.0)))
        return cone is not None and abs(cone.ra % 360.0 - centre_ra) < 1e-9 and abs(cone.dec - centre_dec) < 1e-9
    return False


def _same_expr(binder: Binder, a: Any, b: Any) -> bool:
    """Structural equality of two expressions, ignoring source positions (columns by resolved name)."""
    if isinstance(a, Col) or isinstance(b, Col):
        return isinstance(a, Col) and isinstance(b, Col) and binder.column(a).name == binder.column(b).name
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same_expr(binder, x, y) for x, y in zip(a, b))
    fields = getattr(type(a), "__dataclass_fields__", None)
    if fields is None:
        return bool(a == b) and type(a) is type(b)
    return all(_same_expr(binder, getattr(a, name), getattr(b, name)) for name in fields if name != "pos")


@dataclass(slots=True)
class _QueryPlan:
    """A parsed, type-checked and planned ADQL query (see :func:`_plan_query`)."""

    parsed: SelectQuery
    table: TableDef
    binder: Binder
    out_exprs: list[Any]
    out_cols: list[VOColumn]
    order_keys: list[tuple[_OrderKey, bool]]
    aggregate_query: bool
    limit: int
    metadata_only: bool
    infos: list[tuple[str, str]]
    source_rows: list[dict[str, Any]] = field(default_factory=list)
    cone: Cone | None = None
    centre_ra: float = 0.0
    all_cones: list[tuple[Any, Cone]] = field(default_factory=list)
    catalogs: list[str] | None = None
    where_refs: set[str] = field(default_factory=set)
    select_refs: set[str] = field(default_factory=set)
    order_refs: set[str] = field(default_factory=set)
    spent_s: float = 0.0


def _plan_query(query: str, registry: CatalogRegistry, cfg: VOConfig, maxrec: int | None) -> _QueryPlan:
    """Parse, type-check and plan ``query`` (CPU-bound; run in a worker thread).

    Evaluating constant expressions (cone centres and radii, catalog-level conditions) is
    charged to the query's ``max_eval_seconds`` budget; the time used is returned in
    :attr:`_QueryPlan.spent_s` so that evaluation gets only the remainder.
    """
    parsed = parse_adql(query)
    table = find_table(parsed.table.parts, parsed.table.quoted)
    binder = Binder(table, parsed.table)
    budget = EvaluationBudget(cfg.max_eval_seconds)
    binder.budget = budget
    try:
        plan = _plan_parsed(query, parsed, table, binder, registry, cfg, maxrec)
    finally:
        binder.budget = None
    plan.spent_s = budget.elapsed()
    return plan


def _plan_parsed(query: str, parsed: SelectQuery, table: TableDef, binder: Binder, registry: CatalogRegistry,
                 cfg: VOConfig, maxrec: int | None) -> _QueryPlan:
    # -- type-check -----------------------------------------------------------
    if parsed.where is not None:
        if _condition_has_aggregate(parsed.where):
            raise ADQLError("Aggregate functions are not allowed in WHERE")
        binder.check_condition(parsed.where)
    aggregate_query = any(item.expr is not None and _has_aggregate(item.expr) for item in parsed.items)
    out_exprs: list[Any] = []
    out_cols: list[VOColumn] = []
    names: list[str] = []
    for item in parsed.items:
        if item.star:
            if aggregate_query:
                raise ADQLError("SELECT * cannot be combined with aggregate functions (GROUP BY is not supported)")
            for col in binder.star_columns(item.star_qualifier):
                out_exprs.append(Col((col.name,), (True,), 0))
                out_cols.append(col)
                names.append(col.name)
            continue
        kind = binder.type_of(item.expr, allow_aggregate=True)
        if kind in {"point", "circle"}:
            raise ADQLError("Geometry values (POINT/CIRCLE) cannot be selected; select ra, dec or DISTANCE(...)")
        if aggregate_query and _has_column_outside_aggregate(item.expr):
            raise ADQLError("Aggregate queries may only select aggregates and constants (GROUP BY is not supported)")
        out_exprs.append(item.expr)
        out_cols.append(_output_column(binder, item.expr, item.alias or _expr_label(item.expr)))
        names.append(item.alias or _expr_label(item.expr))
    if len(out_exprs) > MAX_SELECT_ITEMS:
        raise ADQLError(f"The query selects {len(out_exprs)} columns; this service allows at most {MAX_SELECT_ITEMS}")
    names = _unique(names)
    out_cols = [replace(col, name=name) for col, name in zip(out_cols, names)]

    order_keys: list[tuple[_OrderKey, bool]] = []
    for order in parsed.order_by:
        expr = order.expr
        if isinstance(expr, Lit) and isinstance(expr.value, int):
            if not 1 <= expr.value <= len(out_cols):
                raise ADQLError(f"ORDER BY position {expr.value} is out of range (1-{len(out_cols)})")
            order_keys.append((_OrderKey(output_index=expr.value - 1), order.descending))
            continue
        if isinstance(expr, Col) and len(expr.parts) == 1:
            # An output column name (alias) takes precedence over a table column, as in SQL.
            alias_hits = [i for i, n in enumerate(names)
                          if (n == expr.parts[0] if expr.quoted[0] else n.casefold() == expr.parts[0].casefold())]
            if alias_hits:
                order_keys.append((_OrderKey(output_index=alias_hits[0]), order.descending))
                continue
        if aggregate_query:
            raise ADQLError("ORDER BY in an aggregate query must name a selected column")
        if binder.type_of(expr) in {"point", "circle"}:
            raise ADQLError("Cannot ORDER BY a geometry value")
        if parsed.distinct:
            # SQL (and PostgreSQL): for SELECT DISTINCT, ORDER BY expressions must appear in the
            # select list -- a key on a non-selected column has no single value per output row.
            same = [i for i, e in enumerate(out_exprs) if _same_expr(binder, e, expr)]
            if not same:
                raise ADQLError("For SELECT DISTINCT, ORDER BY expressions must appear in the select list")
            order_keys.append((_OrderKey(output_index=same[0]), order.descending))
            continue
        order_keys.append((_OrderKey(binder=binder, expr=expr), order.descending))

    # -- plan -------------------------------------------------------------------
    limit = cfg.default_maxrec if maxrec is None else maxrec
    limit = min(limit, cfg.hard_maxrec)
    plan = _QueryPlan(parsed, table, binder, out_exprs, out_cols, order_keys, aggregate_query, limit,
                      limit == 0 or parsed.top == 0, [("QUERY", query.strip())])
    if not table.requires_cone:
        plan.source_rows = catalog_rows(registry, cfg) if table.name == "astrosearch.catalogs" \
            else tap_schema_rows(table.name)
        return plan
    all_cones = cone_terms(binder, parsed.where)
    cone = min((c for _t, c in all_cones), key=lambda c: c.radius) if all_cones else None
    if cone is None:
        raise ADQLError(
            f"Queries on {table.name} must constrain position with a top-level "
            "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', ra0, dec0, radius_deg)) condition "
            f"(radius <= {cfg.max_radius_deg:g} deg): rows are computed live from the upstream archives")
    if cone.radius <= 0:
        raise ADQLError(f"The CIRCLE radius must be positive, got {cone.radius:g} deg"
                        f"{_int_division_hint(cone.radius_expr)}")
    try:
        plan.centre_ra = _validate_cone(cone.ra % 360.0, cone.dec, cone.radius, cfg)
    except VOParameterError as exc:
        raise ADQLError(str(exc)) from exc
    plan.cone, plan.all_cones = cone, all_cones
    # Columns the query reads: group_* values depend on every archive queried.
    plan.where_refs = referenced_columns(binder, parsed.where)
    for item in parsed.items:
        if item.star:
            plan.select_refs |= {c.name for c in binder.star_columns(item.star_qualifier)}
        else:
            plan.select_refs |= referenced_columns(binder, item.expr)
    for key, _descending in order_keys:
        plan.order_refs |= referenced_columns(binder, key.expr if key.output_index is None
                                              else out_exprs[key.output_index])
    catalogs = plan_catalogs(binder, parsed.where, registry)
    if catalogs and (plan.where_refs | plan.select_refs | plan.order_refs) & GROUP_COLUMNS:
        catalogs = None  # rows come from some archives, but their group values from all of them
    # A column-free conjunct has one value for every row: evaluated ONCE here (off the event
    # loop, within the query budget); unless TRUE no row can match and no archive is asked.
    constant_terms = [t for t in _conjuncts(parsed.where) if not referenced_columns(binder, t)]
    if any(binder.test(t, {}) is not True for t in constant_terms):
        catalogs = []
    plan.catalogs = catalogs
    plan.infos.append(("cone", f"ra={plan.centre_ra:.8f} dec={cone.dec:.8f} radius_deg={cone.radius:g}"))
    return plan


def _incomplete_rank_limit(plan: _QueryPlan, fetched: ConeFetch, cfg: VOConfig) -> tuple[int, float] | None:
    """For a truncated/failed cone: the ``rank_limit`` of an exact "nearest n" answer, None when
    the rows can be returned flagged, or the error of :func:`incompleteness_error` when the
    query needs every row (CPU-bound: run in a worker thread; charged to the query budget).

    Truncated archives hold every row up to their completeness radius, so an unfiltered cone
    is a well-defined, flagged subset; anything that selects, counts or ranks rows could be
    arbitrarily wrong and is refused -- except the nearest rows, which are exact when they
    lie inside that radius.
    """
    parsed, binder, cone = plan.parsed, plan.binder, plan.cone
    assert cone is not None
    budget = EvaluationBudget(cfg.max_eval_seconds, plan.spent_s)
    binder.budget = budget
    try:
        cone_ids = {id(t) for t, c in plan.all_cones if (c.ra, c.dec) == (cone.ra, cone.dec) and c.radius >= cone.radius}
        level_ids = {id(t) for t in catalog_level_terms(binder, parsed.where)}
        extra = [t for t in _conjuncts(parsed.where) if id(t) not in cone_ids | level_ids]
        ranked = (not fetched.failed and not plan.aggregate_query and parsed.top is not None and not parsed.distinct
                  and not ((plan.where_refs | plan.order_refs) & GROUP_COLUMNS)
                  and _ranks_by_separation(binder, plan.order_keys, plan.out_exprs, plan.centre_ra, cone.dec))
    finally:
        binder.budget = None
        plan.spent_s += budget.elapsed()
    why = None
    if plan.aggregate_query:
        why = "An aggregate (COUNT/MIN/MAX/AVG/SUM) needs every row of the cone"
    elif ranked:
        radius = min(fetched.complete_to.get(name, 0.0) for name in fetched.truncated)
        return ((parsed.offset or 0) + min(parsed.top or 0, plan.limit), radius)
    elif extra:
        why = "A WHERE condition on columns other than the cone, catalog and wavelength needs every row of the cone"
    elif parsed.order_by and (parsed.top is not None or parsed.offset):
        why = ("ORDER BY with TOP/OFFSET needs every row of the cone (only TOP n ... ORDER BY separation, "
               "without DISTINCT or group_* terms, can be answered from the complete inner part of the cone)")
    if why is not None:
        raise incompleteness_error(fetched.truncated, fetched.failed, why)
    return None


async def _execute_adql(
    query: str,
    *,
    service: Any | None,
    registry: CatalogRegistry | None,
    maxrec: int | None,
    config: VOConfig | None,
) -> ResultTable:
    cfg = config or VOConfig.from_env()
    registry = registry or getattr(service, "registry", None) or CatalogRegistry()
    # Parsing, type-checking and planning (which evaluates constant expressions) run in a
    # worker thread: a costly constant condition must never stall the event loop.
    plan = await run_in_worker(_plan_query, query, registry, cfg, maxrec)
    table, parsed = plan.table, plan.parsed
    infos = plan.infos
    source_rows = plan.source_rows
    truncated: dict[str, str] = {}
    failed: dict[str, str] = {}
    rank_limit: tuple[int, float] | None = None
    if table.requires_cone and not plan.metadata_only:
        assert plan.cone is not None
        if service is None:
            raise UpstreamError("No crossmatch service is configured")
        fetched = await fetch_match_rows(service, plan.centre_ra, plan.cone.dec, plan.cone.radius,
                                         catalogs=plan.catalogs, row_limit=cfg.catalog_row_limit,
                                         max_total_rows=cfg.max_cone_rows)
        infos.extend(fetched.infos)
        source_rows = fetched.rows
        truncated, failed = dict(fetched.truncated), dict(fetched.failed)
        if truncated or failed:
            rank_limit = await run_in_worker(_incomplete_rank_limit, plan, fetched, cfg)

    # -- filter, project, order, limit (in a worker thread, time- and size-bounded) --
    try:
        outputs = await run_in_worker(_evaluate, plan.binder, parsed, source_rows, plan.out_exprs, plan.order_keys,
                                      plan.aggregate_query, cfg.max_eval_seconds, rank_limit, spent_s=plan.spent_s,
                                      max_bytes=cfg.max_result_bytes)
    except RankIncompleteError:
        assert rank_limit is not None
        raise incompleteness_error(
            truncated, failed, f"TOP ... ORDER BY separation needs the {rank_limit[0]} nearest rows, but they do not all "
            f"lie within {rank_limit[1]:.2f} arcsec, the radius to which every truncated archive is complete") from None
    if rank_limit is not None and not plan.select_refs & GROUP_COLUMNS:
        # Every returned value is exact: the result is complete.
        infos.append(("COMPLETENESS", (f"Exact: the {rank_limit[0]} nearest rows lie within {rank_limit[1]:.2f} "
                                       "arcsec, to which every truncated archive is complete")))
        truncated = {}
    elif truncated or failed:
        infos.append(("INCOMPLETE", incompleteness_summary(truncated, failed)))
    status = "OK"
    limit = plan.limit
    if len(outputs) > limit:
        outputs = outputs[:limit]
        status = "OVERFLOW"
    if limit == 0 or truncated or failed:
        status = "OVERFLOW"
    if plan.metadata_only:
        outputs = []
    return ResultTable(table.short_name, plan.out_cols, outputs, status, infos, table.description, truncated, failed)


def _condition_has_aggregate(cond: Any) -> bool:
    if isinstance(cond, (And, Or)):
        return any(_condition_has_aggregate(i) for i in cond.items)
    if isinstance(cond, Not):
        return _condition_has_aggregate(cond.item)
    if isinstance(cond, Cmp):
        return _has_aggregate(cond.left) or _has_aggregate(cond.right)
    if isinstance(cond, Between):
        return any(_has_aggregate(e) for e in (cond.expr, cond.low, cond.high))
    if isinstance(cond, Like):
        return _has_aggregate(cond.expr) or _has_aggregate(cond.pattern)
    if isinstance(cond, InList):
        return _has_aggregate(cond.expr) or any(_has_aggregate(i) for i in cond.items)
    if isinstance(cond, IsNull):
        return _has_aggregate(cond.expr)
    return False


def _has_column_outside_aggregate(expr: Any) -> bool:
    if isinstance(expr, Col):
        return True
    if isinstance(expr, Func):
        if expr.name in _AGGREGATES:
            return False
        return any(_has_column_outside_aggregate(a) for a in expr.args)
    if isinstance(expr, Neg):
        return _has_column_outside_aggregate(expr.operand)
    if isinstance(expr, Arith):
        return _has_column_outside_aggregate(expr.left) or _has_column_outside_aggregate(expr.right)
    return False


# ---------------------------------------------------------------------------
# TAP Request Handling (shared by sync and async)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TAPResponse:
    """Serialized TAP outcome: body, media type, HTTP status, error message (if any) and
    extra HTTP headers (``X-VO-Query-Status``, see :func:`status_headers`)."""

    body: bytes
    media_type: str
    status_code: int = 200
    error: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


_SUPPORTED_LANGS = frozenset({"ADQL", "ADQL-2.0", "ADQL-2.1"})


def validate_tap_parameters(params: Mapping[str, str], config: VOConfig) -> tuple[str, int | None, str, str]:
    """Validate DALI/TAP parameters; returns (query, maxrec, serializer key, media type)."""
    request = params.get("REQUEST")
    if request is not None and request.strip() != "doQuery":
        raise VOParameterError(f"Unsupported REQUEST {request!r}; only REQUEST=doQuery is supported")
    lang = params.get("LANG")
    if lang is None or not lang.strip():
        raise VOParameterError("Missing required parameter LANG (use LANG=ADQL)")
    if lang.strip().upper() not in _SUPPORTED_LANGS:
        raise VOParameterError(f"Unsupported LANG {lang!r}; supported: ADQL, ADQL-2.0, ADQL-2.1")
    query = params.get("QUERY")
    if query is None or not query.strip():
        raise VOParameterError("Missing required parameter QUERY")
    if params.get("UPLOAD"):
        raise VOParameterError("UPLOAD is not supported by this service")
    maxrec: int | None = None
    raw_maxrec = params.get("MAXREC")
    if raw_maxrec is not None and raw_maxrec.strip():
        try:
            maxrec = int(raw_maxrec.strip())
        except ValueError as exc:
            raise VOParameterError(f"MAXREC must be a non-negative integer, got {raw_maxrec!r}") from exc
        if maxrec < 0:
            raise VOParameterError(f"MAXREC must be a non-negative integer, got {raw_maxrec!r}")
        maxrec = min(maxrec, config.hard_maxrec)
    fmt, media = resolve_format(params.get("RESPONSEFORMAT") or params.get("FORMAT"))
    return query, maxrec, fmt, media


ServiceProvider = Callable[[], tuple[Any, CatalogRegistry | None]]


async def handle_tap_query(
    params: Mapping[str, str],
    *,
    service: Any | None = None,
    registry: CatalogRegistry | None = None,
    config: VOConfig | None = None,
    service_provider: ServiceProvider | None = None,
) -> TAPResponse:
    """Execute a TAP doQuery request (upper-cased parameter names) and serialize the outcome.

    ``service_provider`` (called only when no ``service`` is given) builds the service and
    registry lazily, so that a failure to build them is reported as an error document
    too. Errors become DALI error VOTables: HTTP 400 for bad parameters/queries and
    truncated cones, 502 when upstream archives failed, 503 when the service cannot be
    built, 500 for unexpected internal errors.
    """
    cfg = config or VOConfig.from_env()
    try:
        query, maxrec, fmt, media = validate_tap_parameters(params, cfg)
        if service is None and service_provider is not None:
            try:
                service, provided_registry = service_provider()
            except Exception as exc:  # reported as an error document
                raise ServiceUnavailableError(f"Service unavailable: {exc.__class__.__name__}: {exc}") from exc
            registry = registry or provided_registry
        result = await execute_adql(query, service=service, registry=registry, maxrec=maxrec, config=cfg)
        if params.get("RUNID"):
            result.infos.insert(0, ("RUNID", params["RUNID"]))
        body = await run_in_worker(render, result, fmt)
        if len(body) > cfg.max_result_bytes:
            raise ResultTooLargeError(result_too_large_message(cfg.max_result_bytes))
        return TAPResponse(body, media, headers=status_headers(result))
    except VOError as exc:
        return TAPResponse(error_votable(str(exc)), VOTABLE_MEDIA_TYPE, exc.status_code, str(exc), dict(ERROR_HEADERS))
    except Exception as exc:  # noqa: BLE001 - reported to the client as a TAP error document
        message = f"Internal error: {exc.__class__.__name__}: {exc}"
        return TAPResponse(error_votable(message), VOTABLE_MEDIA_TYPE, 500, message, dict(ERROR_HEADERS))


# ---------------------------------------------------------------------------
# VOSI Documents (availability, capabilities, tables) and DALI examples
# ---------------------------------------------------------------------------

EXAMPLES: tuple[tuple[str, str, str], ...] = (
    ("cone-3c273", "All counterparts of 3C 273 within 10 arcsec",
     ("SELECT * FROM astrosearch.matches "
      "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 187.2779154, 2.0523883, 10/3600.0)) "
      "ORDER BY separation")),
    ("radio-xray", "Radio and X-ray detections near M 87",
     ("SELECT catalog, source_id, ra, dec, separation FROM astrosearch.matches "
      "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 187.7059308, 12.3911233, 0.003)) "
      "AND wavelength IN ('radio', 'xray') ORDER BY separation")),
    ("count-by-catalog", "Number of Gaia DR3 sources within 20 arcsec of HD 209458",
     ("SELECT COUNT(*) AS n FROM astrosearch.matches "
      "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 330.79502, 18.88432, 20/3600.0)) "
      "AND catalog = 'gaia_dr3'")),
    ("catalogs", "Enabled radio catalogs and their citations",
     "SELECT name, provider, citation FROM astrosearch.catalogs WHERE wavelength = 'radio' AND enabled = 1"),
    ("columns", "Columns of astrosearch.matches",
     ("SELECT column_name, ucd, unit, description FROM TAP_SCHEMA.columns "
      "WHERE table_name = 'astrosearch.matches' ORDER BY column_index")),
)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def availability_xml(*, available: bool = True, note: str | None = None) -> bytes:
    """VOSI 1.1 availability document."""
    notes = f"<avl:note>{_xml_text(note)}</avl:note>" if note else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<avl:availability xmlns:avl="http://www.ivoa.net/xml/VOSIAvailability/v1.0">'
        f"<avl:available>{'true' if available else 'false'}</avl:available>"
        f"<avl:upSince>{_iso(_UP_SINCE)}</avl:upSince>{notes}</avl:availability>\n"
    ).encode()


def capabilities_xml(tap_base_url: str, config: VOConfig | None = None) -> bytes:
    """VOSI capabilities with the TAPRegExt 1.0 ``tr:TableAccess`` description of this service."""
    cfg = config or VOConfig.from_env()
    base = tap_base_url.rstrip("/")

    def vosi(standard: str, path: str, version: str | None = None) -> str:
        ver = f' version="{version}"' if version else ""
        return (f'<capability standardID="ivo://ivoa.net/std/VOSI#{standard}">'
                f'<interface xsi:type="vs:ParamHTTP" role="std"{ver}>'
                f'<accessURL use="full">{_xml_escape(base + "/" + path)}</accessURL></interface></capability>')

    formats = [
        ("ivo://ivoa.net/std/TAPRegExt#output-votable-td", "application/x-votable+xml", ["votable", "votable/td"]),
        (None, "text/xml", []),
        (None, "text/csv;header=present", ["csv"]),
        (None, "text/tab-separated-values", ["tsv"]),
        (None, "application/json", ["json"]),
    ]
    output = "".join(
        f"<outputFormat{f' ivo-id={_xml_attr(ivo)}' if ivo else ''}><mime>{_xml_escape(mime)}</mime>"
        + "".join(f"<alias>{a}</alias>" for a in aliases) + "</outputFormat>"
        for ivo, mime, aliases in formats
    )
    geo = "".join(f"<feature><form>{f}</form></feature>" for f in ("POINT", "CIRCLE", "CONTAINS", "DISTANCE"))
    string_features = "".join(f"<feature><form>{f}</form></feature>" for f in ("LOWER", "UPPER", "ILIKE"))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<vosi:capabilities xmlns:vosi="http://www.ivoa.net/xml/VOSICapabilities/v1.0" '
        'xmlns:tr="http://www.ivoa.net/xml/TAPRegExt/v1.0" xmlns:vr="http://www.ivoa.net/xml/VOResource/v1.0" '
        'xmlns:vs="http://www.ivoa.net/xml/VODataService/v1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<capability standardID="ivo://ivoa.net/std/TAP" xsi:type="tr:TableAccess">'
        f'<interface xsi:type="vs:ParamHTTP" role="std" version="1.1"><accessURL use="base">{_xml_escape(base)}</accessURL></interface>'
        "<language><name>ADQL</name>"
        '<version ivo-id="ivo://ivoa.net/std/ADQL#v2.0">2.0</version>'
        '<version ivo-id="ivo://ivoa.net/std/ADQL#v2.1">2.1</version>'
        "<description>Subset of ADQL evaluated by AstroSearch: single-table SELECT with WHERE, ORDER BY, TOP, OFFSET, "
        "DISTINCT, COUNT/MIN/MAX/AVG/SUM and ICRS cone constraints (CONTAINS/POINT/CIRCLE, DISTANCE). "
        "astrosearch.matches requires a top-level CONTAINS cone of radius at most "
        f"{cfg.max_radius_deg:g} deg; each upstream archive is asked for the rows nearest the cone centre, up to "
        f"min({cfg.catalog_row_limit}, {cfg.max_cone_rows} / number of archives queried), and a cone holding more is "
        "flagged (QUERY_STATUS=OVERFLOW) or, for aggregate, filtered or ranked queries, refused -- except "
        "TOP n ... ORDER BY separation, answered exactly when the n nearest rows lie within the complete part of "
        "the cone.</description>"
        f'<languageFeatures type="ivo://ivoa.net/std/TAPRegExt#features-adqlgeo">{geo}</languageFeatures>'
        f'<languageFeatures type="ivo://ivoa.net/std/TAPRegExt#features-adql-string">{string_features}</languageFeatures>'
        '<languageFeatures type="ivo://ivoa.net/std/TAPRegExt#features-adql-offset"><feature><form>OFFSET</form></feature></languageFeatures>'
        "</language>"
        f"{output}"
        f"<retentionPeriod><default>{cfg.retention_s}</default><hard>{cfg.retention_s}</hard></retentionPeriod>"
        f"<executionDuration><default>{cfg.execution_duration_s}</default><hard>{cfg.execution_duration_s}</hard></executionDuration>"
        f'<outputLimit><default unit="row">{cfg.default_maxrec}</default><hard unit="row">{cfg.hard_maxrec}</hard></outputLimit>'
        "</capability>"
        + vosi("capabilities", "capabilities")
        + vosi("availability", "availability")
        # VOSI 1.1 (REC 2017-05-24, section 3): "tables (1.1)" is ivo://ivoa.net/std/VOSI#tables-1.1;
        # this endpoint implements 1.1 (detail=min and per-table child resources).
        + vosi("tables-1.1", "tables", "1.1")
        + '<capability standardID="ivo://ivoa.net/std/DALI#examples"><interface xsi:type="vr:WebBrowser">'
        f'<accessURL use="full">{_xml_escape(base + "/examples")}</accessURL></interface></capability>'
        "</vosi:capabilities>\n"
    ).encode("utf-8")


def _column_xml(col: VOColumn) -> str:
    parts = [f"<name>{_xml_escape(col.name)}</name>", f"<description>{_xml_escape(col.description)}</description>"]
    if col.unit:
        parts.append(f"<unit>{_xml_escape(col.unit)}</unit>")
    if col.ucd:
        parts.append(f"<ucd>{_xml_escape(col.ucd)}</ucd>")
    if col.utype:
        parts.append(f"<utype>{_xml_escape(col.utype)}</utype>")
    arraysize = f' arraysize="{col.arraysize}"' if col.arraysize else ""
    parts.append(f'<dataType xsi:type="vs:VOTableType"{arraysize}>{col.datatype}</dataType>')
    if col.indexed:
        parts.append("<flag>indexed</flag>")
    if col.name in {"match_id"} or (col.ucd or "").startswith("meta.id;meta.main"):
        parts.append("<flag>primary</flag>")
    if col.nullable:
        parts.append("<flag>nullable</flag>")
    std = ' std="true"' if col.std else ""
    return f"<column{std}>" + "".join(parts) + "</column>"


def _table_xml(table: TableDef, *, with_columns: bool, tag: str = "table", namespaces: str = "") -> str:
    kind = "view" if table.table_type == "view" else "base_table"
    body = f"<name>{_xml_escape(table.name)}</name><description>{_xml_escape(table.description)}</description>"
    if with_columns:
        body += "".join(_column_xml(c) for c in table.columns)
    return f'<{tag}{namespaces} type="{kind}">{body}</{tag}>'


_VOSI_TABLE_NS = (' xmlns:vtm="http://www.ivoa.net/xml/VOSITables/v1.0" '
                  'xmlns:vs="http://www.ivoa.net/xml/VODataService/v1.1" '
                  'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')


def tableset_xml(*, detail: str | None = None) -> bytes:
    """VOSI 1.1 tableset; ``detail=min`` omits columns (clients then fetch ``tables/<name>``)."""
    with_columns = (detail or "max").lower() != "min"
    schemas = []
    for schema, description in SCHEMAS:
        tables = "".join(_table_xml(t, with_columns=with_columns) for t in TABLES if t.schema == schema)
        schemas.append(f"<schema><name>{_xml_escape(schema)}</name><description>{_xml_escape(description)}</description>{tables}</schema>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<vtm:tableset{_VOSI_TABLE_NS}>' + "".join(schemas) + "</vtm:tableset>\n").encode("utf-8")


def table_xml(name: str) -> bytes:
    """VOSI 1.1 single-table document (``vtm:table`` root) for ``GET tables/<name>``."""
    table = next((t for t in TABLES if t.name.casefold() == name.casefold()), None)
    if table is None:
        raise KeyError(name)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + _table_xml(table, with_columns=True, tag="vtm:table", namespaces=_VOSI_TABLE_NS) + "\n").encode("utf-8")


# DALI 1.1 section 2.3: example elements "must be descendants of an element that has a vocab
# attribute" with this value (Appendix A.1 records the change from ivo://ivoa.net/std/DALI#examples).
DALI_EXAMPLES_VOCAB = "http://www.ivoa.net/rdf/examples#"


def examples_xhtml(tap_base_url: str) -> bytes:
    """DALI 1.1 examples document (XHTML + RDFa, ``property="query"`` elements under
    ``vocab="http://www.ivoa.net/rdf/examples#"``)."""
    blocks = "".join(
        f'<div typeof="example" id="{ident}" resource="#{ident}">'
        f'<h2 property="name">{_xml_escape(title)}</h2>'
        f'<pre property="query">{_xml_escape(query)}</pre></div>'
        for ident, title, query in EXAMPLES
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>AstroSearch TAP examples</title></head>'
        f'<body vocab="{DALI_EXAMPLES_VOCAB}"><h1>AstroSearch TAP examples</h1>'
        f'<p>Service: <a href="{_xml_escape(tap_base_url)}">{_xml_escape(tap_base_url)}</a></p>{blocks}</body></html>\n'
    ).encode()


# ---------------------------------------------------------------------------
# UWS 1.1 Asynchronous Jobs
# ---------------------------------------------------------------------------

_ACTIVE_PHASES = frozenset({"QUEUED", "EXECUTING"})
_FINAL_PHASES = frozenset({"COMPLETED", "ERROR", "ABORTED"})


@dataclass
class UWSJob:
    """State of one asynchronous TAP job."""

    job_id: str
    parameters: dict[str, str]
    creation_time: datetime
    destruction: datetime
    execution_duration: int
    run_id: str | None = None
    phase: str = "PENDING"
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: bytes | None = None
    result_type: str | None = None
    result_headers: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    error_document: bytes | None = None
    task: asyncio.Task[None] | None = None


Runner = Callable[[Mapping[str, str]], Awaitable[TAPResponse]]


class UWSJobManager:
    """In-memory UWS 1.1 job store: create, run, wait, abort, delete and expire jobs."""

    def __init__(self, config: VOConfig | None = None) -> None:
        self.config = config or VOConfig.from_env()
        self.jobs: dict[str, UWSJob] = {}

    def prune(self) -> None:
        now = datetime.now(UTC)
        for job_id in [j.job_id for j in self.jobs.values() if j.destruction <= now]:
            self.delete(job_id)

    def create(self, parameters: Mapping[str, str]) -> UWSJob:
        self.prune()
        if len(self.jobs) >= self.config.max_jobs:
            finished = sorted((j for j in self.jobs.values() if j.phase in _FINAL_PHASES | {"PENDING"}),
                              key=lambda j: j.creation_time)
            if not finished:
                raise ServiceUnavailableError("Too many active jobs; try again later")
            self.delete(finished[0].job_id)
        now = datetime.now(UTC)
        params = {k.upper(): v for k, v in parameters.items() if k.upper() not in {"PHASE", "ACTION"}}
        check_job_parameters(params)
        job = UWSJob(
            job_id=uuid.uuid4().hex,
            parameters=params,
            creation_time=now,
            destruction=now + timedelta(seconds=self.config.retention_s),
            execution_duration=self.config.execution_duration_s,
            run_id=params.get("RUNID"),
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> UWSJob:
        self.prune()
        return self.jobs[job_id]

    def start(self, job: UWSJob, runner: Runner) -> None:
        if job.phase != "PENDING":
            return
        job.phase = "QUEUED"
        job.task = asyncio.get_running_loop().create_task(self._run(job, runner))

    async def _run(self, job: UWSJob, runner: Runner) -> None:
        job.phase = "EXECUTING"
        job.start_time = datetime.now(UTC)
        try:
            timeout = job.execution_duration or None
            response = await asyncio.wait_for(runner(dict(job.parameters)), timeout=timeout)
        except asyncio.CancelledError:
            if job.phase != "ABORTED":
                job.phase = "ABORTED"
            job.end_time = datetime.now(UTC)
            return
        except TimeoutError:
            job.error_message = f"Execution duration of {job.execution_duration} s exceeded"
            job.error_document = error_votable(job.error_message)
            job.phase = "ERROR"
            job.end_time = datetime.now(UTC)
            return
        except Exception as exc:  # noqa: BLE001 - recorded as the job's error summary
            job.error_message = f"Internal error: {exc.__class__.__name__}: {exc}"
            job.error_document = error_votable(job.error_message)
            job.phase = "ERROR"
            job.end_time = datetime.now(UTC)
            return
        job.end_time = datetime.now(UTC)
        if response.error is None and not self._make_room(job, len(response.body)):
            message = (f"The result ({len(response.body)} bytes) exceeds this service's job result store of "
                       f"{self.config.uws_max_stored_bytes} bytes (VO_UWS_MAX_STORED_BYTES)")
            response = TAPResponse(error_votable(message), VOTABLE_MEDIA_TYPE, 400, message, dict(ERROR_HEADERS))
        if response.error is not None:
            job.error_message = response.error
            job.error_document = response.body
            job.phase = "ERROR"
        else:
            job.result = response.body
            job.result_type = response.media_type
            job.result_headers = dict(response.headers)
            job.phase = "COMPLETED"

    def stored_bytes(self) -> int:
        """Total size of the results and error documents held by the job store."""
        return sum(len(j.result or b"") + len(j.error_document or b"") for j in self.jobs.values())

    def _make_room(self, job: UWSJob, size: int) -> bool:
        """Keep the store within ``uws_max_stored_bytes`` before ``job`` stores ``size`` bytes.

        The oldest finished jobs (by end time) are destroyed early until the new result fits
        (UWS 1.1 section 2.2.3.3: the service may destroy a job before its destruction
        time); False when the result alone exceeds the store.
        """
        cap = self.config.uws_max_stored_bytes
        if size > cap:
            return False
        stored = self.stored_bytes()
        if stored + size <= cap:
            return True
        finished = sorted((j for j in self.jobs.values() if j is not job and j.phase in _FINAL_PHASES),
                          key=lambda j: j.end_time or j.creation_time)
        for old in finished:
            if stored + size <= cap:
                break
            stored -= len(old.result or b"") + len(old.error_document or b"")
            self.delete(old.job_id)
        return stored + size <= cap

    def abort(self, job: UWSJob) -> None:
        if job.phase in _FINAL_PHASES:
            return
        job.phase = "ABORTED"
        job.end_time = datetime.now(UTC)
        if job.task is not None and not job.task.done():
            job.task.cancel()

    def delete(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        if job is not None and job.task is not None and not job.task.done():
            job.task.cancel()

    async def wait(self, job: UWSJob, seconds: float) -> None:
        """Block until the job leaves its current active phase or ``seconds`` elapse (UWS 1.1 WAIT)."""
        if job.phase not in _ACTIVE_PHASES | {"PENDING"}:
            return
        start_phase = job.phase
        deadline = asyncio.get_running_loop().time() + max(0.0, min(seconds, self.config.max_wait_s))
        while job.phase == start_phase and asyncio.get_running_loop().time() < deadline:
            if job.task is not None and job.task.done() and job.phase == start_phase:
                break
            await asyncio.sleep(0.05)


def check_job_parameters(params: Mapping[str, str]) -> None:
    """Reject UWS job parameters that no XML job document could carry (XML 1.0 illegal characters)."""
    for key, value in params.items():
        if has_xml_illegal_chars(key) or has_xml_illegal_chars(value):
            raise VOParameterError(f"Parameter {xml_safe(key)!r} contains characters that XML 1.0 does not allow "
                                   "(control characters such as U+0001)")


def _nil(tag: str) -> str:
    return f'<uws:{tag} xsi:nil="true"/>'


def job_xml(job: UWSJob, job_url: str) -> bytes:
    """UWS 1.1 job representation."""
    params = "".join(f'<uws:parameter id={_xml_attr_value(k.lower())}>{_xml_text(v)}</uws:parameter>'
                     for k, v in sorted(job.parameters.items()))
    results = ""
    if job.phase == "COMPLETED":
        results = (f'<uws:result id="result" xlink:type="simple" xlink:href={_xml_attr(job_url + "/results/result")}'
                   f' mime-type={_xml_attr(job.result_type or VOTABLE_MEDIA_TYPE)} size="{len(job.result or b"")}"/>')
    error = ""
    if job.phase == "ERROR" and job.error_message:
        error = (f'<uws:errorSummary type="fatal" hasDetail="true"><uws:message>{_xml_text(job.error_message)}'
                 "</uws:message></uws:errorSummary>")
    run_id = f"<uws:runId>{_xml_text(job.run_id)}</uws:runId>" if job.run_id else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<uws:job xmlns:uws="{UWS_NS}" xmlns:xlink="{XLINK_NS}" xmlns:xsi="{XSI_NS}" version="1.1">'
        f"<uws:jobId>{job.job_id}</uws:jobId>{run_id}{_nil('ownerId')}<uws:phase>{job.phase}</uws:phase>"
        f"{_nil('quote')}<uws:creationTime>{_iso(job.creation_time)}</uws:creationTime>"
        + (f"<uws:startTime>{_iso(job.start_time)}</uws:startTime>" if job.start_time else _nil("startTime"))
        + (f"<uws:endTime>{_iso(job.end_time)}</uws:endTime>" if job.end_time else _nil("endTime"))
        + f"<uws:executionDuration>{job.execution_duration}</uws:executionDuration>"
        f"<uws:destruction>{_iso(job.destruction)}</uws:destruction>"
        f"<uws:parameters>{params}</uws:parameters><uws:results>{results}</uws:results>{error}</uws:job>\n"
    ).encode("utf-8")


def job_list_xml(jobs: Iterable[UWSJob], base_url: str) -> bytes:
    """UWS 1.1 job list."""
    refs = "".join(
        f'<uws:jobref id="{j.job_id}" xlink:href={_xml_attr(base_url.rstrip("/") + "/" + j.job_id)}>'
        f"<uws:phase>{j.phase}</uws:phase>"
        + (f"<uws:runId>{_xml_text(j.run_id)}</uws:runId>" if j.run_id else "")
        + f"{_nil('ownerId')}<uws:creationTime>{_iso(j.creation_time)}</uws:creationTime></uws:jobref>"
        for j in jobs
    )
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<uws:jobs xmlns:uws="{UWS_NS}" xmlns:xlink="{XLINK_NS}" '
            f'xmlns:xsi="{XSI_NS}" version="1.1">{refs}</uws:jobs>\n').encode()


def _parse_iso(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/vo", tags=["Virtual Observatory"])


# Largest request body read: a QUERY of MAX_QUERY_LENGTH characters at up to 4 UTF-8 bytes
# each, percent-encoded (x3), plus 64 KiB for the other parameters and multipart framing.
MAX_REQUEST_BYTES = 12 * MAX_QUERY_LENGTH + 65_536


async def read_body(request: Request, limit: int = MAX_REQUEST_BYTES) -> bytes:
    """The request body, refused (:class:`RequestTooLargeError`, HTTP 413) beyond ``limit`` bytes
    -- from its Content-Length before reading, else as soon as the stream exceeds it."""
    declared = request.headers.get("content-length", "").strip()
    if declared.isdigit() and int(declared) > limit:
        raise RequestTooLargeError(f"The request body ({declared} bytes) exceeds this service's limit of {limit} bytes")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise RequestTooLargeError(f"The request body exceeds this service's limit of {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def request_params(request: Request) -> dict[str, str]:
    """DALI parameters from the query string and a urlencoded or multipart body (names upper-cased, first wins).

    A multipart part carrying a file (TAP UPLOAD) is recorded as ``UPLOAD`` so that
    it is rejected with a clear error instead of being silently ignored. Bodies larger
    than :data:`MAX_REQUEST_BYTES` are refused before they are read (:func:`read_body`).
    """
    params: dict[str, str] = {}
    for key, value in request.query_params.multi_items():
        params.setdefault(key.upper(), value)
    if request.method in {"POST", "PUT"}:
        ctype = request.headers.get("content-type", "")
        body = await read_body(request)
        if body and "application/x-www-form-urlencoded" in ctype:
            for key, value in parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True):
                params.setdefault(key.upper(), value)
        elif body and "multipart/form-data" in ctype:
            message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                if part.get_filename() is not None:
                    params.setdefault("UPLOAD", str(name))
                    continue
                payload = part.get_payload(decode=True)
                text = _decode_part(payload, part.get_content_charset()) if isinstance(payload, bytes) else ""
                params.setdefault(str(name).upper(), text)
    return params


def _decode_part(payload: bytes, charset: str | None) -> str:
    """Decode a multipart form value; an unknown or invalid charset falls back to UTF-8."""
    try:
        return payload.decode(charset or "utf-8", "replace")
    except (LookupError, UnicodeError, TypeError):
        return payload.decode("utf-8", "replace")


async def _request_params_or_error(request: Request) -> dict[str, str]:
    """:func:`request_params`, with any failure to read the request reported as a VOParameterError."""
    try:
        return await request_params(request)
    except VOError:
        raise
    except Exception as exc:  # malformed request bodies become DALI error documents
        raise VOParameterError(f"Could not read the request parameters: {exc.__class__.__name__}: {exc}") from exc


def _vo_config(request: Request) -> VOConfig:
    state = request.app.state
    cfg = getattr(state, "vo_config", None)
    if cfg is None:
        cfg = VOConfig.from_env()
        state.vo_config = cfg
    return cfg


def _service(request: Request) -> Any:
    """The shared ``app.state.service``, else one CrossmatchService built on first use.

    The fallback service gets ONE HTTP client -- ``app.state.client`` when present, else
    a client created here and kept in ``app.state.vo_client`` (close it on shutdown with
    :func:`aclose_vo_state`) -- instead of letting every provider open its own.
    """
    state = request.app.state
    service = getattr(state, "service", None)
    if service is not None:
        return service
    service = getattr(state, "vo_service", None)
    if service is None:
        from main import build_service
        from models import Settings

        client = getattr(state, "client", None)
        if client is None:
            import httpx

            client = getattr(state, "vo_client", None)
            if client is None:
                client = httpx.AsyncClient(timeout=Settings().request_timeout_seconds, follow_redirects=True)
                state.vo_client = client
        service = build_service(client=client)
        state.vo_service = service
    return service


async def aclose_vo_state(app: Any) -> None:
    """Cancel running UWS jobs and close the HTTP client this router created (call on shutdown)."""
    state = app.state
    manager = getattr(state, "vo_jobs", None)
    if manager is not None:
        for job_id in list(manager.jobs):
            manager.delete(job_id)
    client = getattr(state, "vo_client", None)
    if client is not None:
        await client.aclose()
        state.vo_client = None
        state.vo_service = None


def _service_provider(request: Request) -> ServiceProvider:
    def provide() -> tuple[Any, CatalogRegistry | None]:
        service = _service(request)
        return service, _registry(request, service)

    return provide


def _registry(request: Request, service: Any) -> CatalogRegistry:
    return getattr(request.app.state, "registry", None) or getattr(service, "registry", None) or CatalogRegistry()


def _jobs(request: Request) -> UWSJobManager:
    state = request.app.state
    manager = getattr(state, "vo_jobs", None)
    if manager is None:
        manager = UWSJobManager(_vo_config(request))
        state.vo_jobs = manager
    return manager


def _url_without(request: Request, suffix_parts: int) -> str:
    url = str(request.url.replace(query=""))
    for _ in range(suffix_parts):
        url = url.rsplit("/", 1)[0]
    return url


def _tap_base(request: Request) -> str:
    """Base URL of the TAP service (``.../vo/tap``) derived from the current request path."""
    path = request.url.path
    marker = path.find("/tap/")
    base_path = path[: marker + len("/tap")] if marker >= 0 else path.rstrip("/")
    return str(request.url.replace(path=base_path, query=""))


@router.get("/scs", summary="Simple Cone Search 1.03 (VOTable of crossmatched sources)")
async def scs_endpoint(request: Request) -> Response:
    """``RA``/``DEC`` (ICRS deg), ``SR`` (deg), optional ``VERB`` (1-3) and ``CATALOGS`` (comma list)."""
    try:
        params = await _request_params_or_error(request)
        ra = _parse_float_param(params, "RA")
        dec = _parse_float_param(params, "DEC")
        sr = _parse_float_param(params, "SR")
        raw_verb = params.get("VERB", "2").strip() or "2"
        try:
            verb = int(raw_verb)
        except ValueError as exc:
            raise VOParameterError(f"VERB must be 1, 2 or 3, got {raw_verb!r}") from exc
        # An empty CATALOGS value means "not given"; one that names no catalog (e.g. ",") is an
        # error (cone_search), never a silent "query nothing".
        raw_catalogs = params.get("CATALOGS")
        catalogs = raw_catalogs.split(",") if raw_catalogs and raw_catalogs.strip() else None
        try:
            service = _service(request)
        except Exception as exc:  # reported as an error document
            raise ServiceUnavailableError(f"Service unavailable: {exc.__class__.__name__}: {exc}") from exc
        result = await cone_search(service, ra, dec, sr, verb=verb, catalogs=catalogs,
                                   registry=_registry(request, service), config=_vo_config(request))
        body = await run_in_worker(render_votable, result)
        headers = status_headers(result)
    except VOError as exc:
        body = error_votable(str(exc), scs=True)
        headers = dict(ERROR_HEADERS)
    except Exception as exc:  # noqa: BLE001 - SCS reports every failure as an error VOTable
        body = error_votable(f"Internal error: {exc.__class__.__name__}: {exc}", scs=True)
        headers = dict(ERROR_HEADERS)
    return Response(content=body, media_type=SCS_MEDIA_TYPE, headers=headers)


@router.get("/tap", summary="TAP service root")
async def tap_root(request: Request) -> PlainTextResponse:
    base = _tap_base(request)
    endpoints = ["sync", "async", "availability", "capabilities", "tables", "examples"]
    return PlainTextResponse(
        f"{SERVICE_TITLE} -- IVOA TAP 1.1 service\n\n" + "\n".join(f"{base}/{e}" for e in endpoints) + "\n")


@router.get("/tap/availability", summary="VOSI availability")
async def tap_availability(request: Request) -> Response:
    try:
        service = _service(request)
        enabled = len(_registry(request, service).enabled_catalogs())
        body = availability_xml(available=enabled > 0,
                                note=f"{enabled} upstream archives configured; per-archive outages are reported "
                                     "as WARNING INFOs in each result.")
    except Exception as exc:  # noqa: BLE001 - availability must still answer
        body = availability_xml(available=False, note=f"Service unavailable: {exc}")
    return Response(content=body, media_type=VOSI_MEDIA_TYPE)


@router.get("/tap/capabilities", summary="VOSI capabilities (TAPRegExt)")
async def tap_capabilities(request: Request) -> Response:
    return Response(content=capabilities_xml(_tap_base(request), _vo_config(request)), media_type=VOSI_MEDIA_TYPE)


@router.get("/tap/tables", summary="VOSI tableset")
async def tap_tables(request: Request) -> Response:
    detail = request.query_params.get("detail") or request.query_params.get("DETAIL")
    return Response(content=tableset_xml(detail=detail), media_type=VOSI_MEDIA_TYPE)


@router.get("/tap/tables/{table_name}", summary="VOSI single-table metadata")
async def tap_table(table_name: str) -> Response:
    try:
        return Response(content=table_xml(table_name), media_type=VOSI_MEDIA_TYPE)
    except KeyError:
        return PlainTextResponse(f"Unknown table {table_name}", status_code=404)


@router.get("/tap/examples", summary="DALI examples")
async def tap_examples(request: Request) -> Response:
    return Response(content=examples_xhtml(_tap_base(request)), media_type="application/xhtml+xml")


@router.api_route("/tap/sync", methods=["GET", "POST"], summary="TAP 1.1 synchronous query")
async def tap_sync(request: Request) -> Response:
    try:
        params = await _request_params_or_error(request)
    except VOError as exc:
        return _tap_error(exc)
    outcome = await handle_tap_query(params, config=_vo_config(request), service_provider=_service_provider(request))
    return Response(content=outcome.body, media_type=outcome.media_type, status_code=outcome.status_code,
                    headers=outcome.headers)


def _tap_error(exc: VOError) -> Response:
    """A DALI error document for ``exc`` with its HTTP status and ``X-VO-Query-Status: ERROR``."""
    return Response(content=error_votable(str(exc)), media_type=VOTABLE_MEDIA_TYPE, status_code=exc.status_code,
                    headers=dict(ERROR_HEADERS))


def _job_runner(request: Request) -> Runner:
    provider = _service_provider(request)
    config = _vo_config(request)

    async def run(params: Mapping[str, str]) -> TAPResponse:
        return await handle_tap_query(params, config=config, service_provider=provider)

    return run


def _async_base(request: Request) -> str:
    return _tap_base(request) + "/async"


def _get_job(request: Request, job_id: str) -> UWSJob | None:
    try:
        return _jobs(request).get(job_id)
    except KeyError:
        return None


def _no_job(job_id: str) -> PlainTextResponse:
    return PlainTextResponse(f"No such job: {job_id}", status_code=404)


async def _job_params(request: Request) -> dict[str, str] | PlainTextResponse:
    """Parameters of a UWS job-resource request, or the plain-text error response for an
    unreadable or oversized body (HTTP 400/413)."""
    try:
        return await _request_params_or_error(request)
    except VOError as exc:
        return PlainTextResponse(str(exc), status_code=exc.status_code)


@router.get("/tap/async", summary="UWS job list")
async def tap_async_list(request: Request) -> Response:
    manager = _jobs(request)
    manager.prune()
    phases = {p.upper() for p in request.query_params.getlist("PHASE") + request.query_params.getlist("phase")}
    jobs = [j for j in manager.jobs.values() if not phases or j.phase in phases]
    after = request.query_params.get("AFTER") or request.query_params.get("after")
    if after:
        try:
            cutoff = _parse_iso(after)
        except ValueError:
            return PlainTextResponse(f"Invalid AFTER timestamp {after!r}", status_code=400)
        jobs = [j for j in jobs if j.creation_time > cutoff]
    jobs.sort(key=lambda j: j.creation_time, reverse=True)
    last = request.query_params.get("LAST") or request.query_params.get("last")
    if last:
        if not last.isdigit():
            return PlainTextResponse(f"Invalid LAST value {last!r}", status_code=400)
        jobs = jobs[: int(last)]
    return Response(content=job_list_xml(jobs, _async_base(request)), media_type=VOSI_MEDIA_TYPE)


@router.post("/tap/async", summary="Create a UWS job (PHASE=RUN starts it)")
async def tap_async_create(request: Request) -> Response:
    try:
        params = await _request_params_or_error(request)
        job = _jobs(request).create(params)
    except VOError as exc:
        return _tap_error(exc)
    if params.get("PHASE", "").upper() == "RUN":
        _jobs(request).start(job, _job_runner(request))
    return RedirectResponse(f"{_async_base(request)}/{job.job_id}", status_code=303)


@router.get("/tap/async/{job_id}", summary="UWS job description (supports WAIT)")
async def tap_job(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    wait = request.query_params.get("WAIT") or request.query_params.get("wait")
    if wait is not None:
        try:
            seconds = float(wait)
        except ValueError:
            return PlainTextResponse(f"Invalid WAIT value {wait!r}", status_code=400)
        wanted = (request.query_params.get("PHASE") or request.query_params.get("phase") or "").upper()
        if not wanted or wanted == job.phase:
            await _jobs(request).wait(job, _jobs(request).config.max_wait_s if seconds < 0 else seconds)
    return Response(content=job_xml(job, f"{_async_base(request)}/{job.job_id}"), media_type=VOSI_MEDIA_TYPE)


@router.post("/tap/async/{job_id}", summary="UWS job actions (ACTION=DELETE)")
async def tap_job_post(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    params = await _job_params(request)
    if isinstance(params, Response):
        return params
    if params.get("ACTION", "").upper() == "DELETE":
        _jobs(request).delete(job_id)
        return RedirectResponse(_async_base(request), status_code=303)
    return PlainTextResponse("Unsupported job action; use ACTION=DELETE", status_code=400)


@router.delete("/tap/async/{job_id}", summary="Delete a UWS job")
async def tap_job_delete(request: Request, job_id: str) -> Response:
    if _get_job(request, job_id) is None:
        return _no_job(job_id)
    _jobs(request).delete(job_id)
    return RedirectResponse(_async_base(request), status_code=303)


@router.get("/tap/async/{job_id}/phase", summary="UWS job phase")
async def tap_job_phase(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    return _no_job(job_id) if job is None else PlainTextResponse(job.phase)


@router.post("/tap/async/{job_id}/phase", summary="Run or abort a UWS job")
async def tap_job_set_phase(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    params = await _job_params(request)
    if isinstance(params, Response):
        return params
    phase = params.get("PHASE", "").upper()
    if phase == "RUN":
        if job.phase != "PENDING":
            return PlainTextResponse(f"Job is {job.phase}; only PENDING jobs can be run", status_code=400)
        _jobs(request).start(job, _job_runner(request))
    elif phase == "ABORT":
        _jobs(request).abort(job)
    else:
        return PlainTextResponse("PHASE must be RUN or ABORT", status_code=400)
    return RedirectResponse(f"{_async_base(request)}/{job_id}", status_code=303)


@router.get("/tap/async/{job_id}/parameters", summary="UWS job parameters")
async def tap_job_parameters(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    params = "".join(f'<uws:parameter id={_xml_attr_value(k.lower())}>{_xml_text(v)}</uws:parameter>'
                     for k, v in sorted(job.parameters.items()))
    body = f'<?xml version="1.0" encoding="UTF-8"?>\n<uws:parameters xmlns:uws="{UWS_NS}">{params}</uws:parameters>\n'
    return Response(content=body.encode("utf-8"), media_type=VOSI_MEDIA_TYPE)


@router.post("/tap/async/{job_id}/parameters", summary="Update parameters of a PENDING job")
async def tap_job_set_parameters(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    if job.phase != "PENDING":
        return PlainTextResponse("Parameters can only be changed while the job is PENDING", status_code=400)
    params = await _job_params(request)
    if isinstance(params, Response):
        return params
    updates = {k: v for k, v in params.items() if k not in {"PHASE", "ACTION"}}
    try:
        check_job_parameters(updates)
    except VOError as exc:
        return PlainTextResponse(str(exc), status_code=exc.status_code)
    job.parameters.update(updates)
    job.run_id = job.parameters.get("RUNID", job.run_id)
    return RedirectResponse(f"{_async_base(request)}/{job_id}", status_code=303)


@router.get("/tap/async/{job_id}/results", summary="UWS job result list")
async def tap_job_results(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    url = f"{_async_base(request)}/{job_id}"
    results = ""
    if job.phase == "COMPLETED":
        results = f'<uws:result id="result" xlink:type="simple" xlink:href={_xml_attr(url + "/results/result")}/>'
    body = (f'<?xml version="1.0" encoding="UTF-8"?>\n<uws:results xmlns:uws="{UWS_NS}" xmlns:xlink="{XLINK_NS}">'
            f"{results}</uws:results>\n")
    return Response(content=body.encode("utf-8"), media_type=VOSI_MEDIA_TYPE)


@router.get("/tap/async/{job_id}/results/result", summary="UWS job result document")
async def tap_job_result(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    if job.phase != "COMPLETED" or job.result is None:
        return PlainTextResponse(f"Job is {job.phase}; no result available", status_code=404)
    return Response(content=job.result, media_type=job.result_type or VOTABLE_MEDIA_TYPE, headers=job.result_headers)


@router.get("/tap/async/{job_id}/error", summary="UWS job error document")
async def tap_job_error(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    if job.phase != "ERROR":
        return PlainTextResponse(f"Job is {job.phase}; no error", status_code=404)
    return Response(content=job.error_document or error_votable(job.error_message or "Unknown error"),
                    media_type=VOTABLE_MEDIA_TYPE, headers=dict(ERROR_HEADERS))


@router.get("/tap/async/{job_id}/executionduration", summary="UWS execution duration (s)")
async def tap_job_duration(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    return _no_job(job_id) if job is None else PlainTextResponse(str(job.execution_duration))


@router.post("/tap/async/{job_id}/executionduration", summary="Set UWS execution duration (s)")
async def tap_job_set_duration(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    params = await _job_params(request)
    if isinstance(params, Response):
        return params
    raw = params.get("EXECUTIONDURATION", "")
    if not raw.strip().isdigit():
        return PlainTextResponse("EXECUTIONDURATION must be a non-negative integer (seconds)", status_code=400)
    if job.phase == "PENDING":
        limit = _jobs(request).config.execution_duration_s
        requested = int(raw)
        job.execution_duration = min(requested, limit) if limit and requested else (requested or limit)
    return RedirectResponse(f"{_async_base(request)}/{job_id}", status_code=303)


@router.get("/tap/async/{job_id}/destruction", summary="UWS destruction time")
async def tap_job_destruction(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    return _no_job(job_id) if job is None else PlainTextResponse(_iso(job.destruction))


@router.post("/tap/async/{job_id}/destruction", summary="Set UWS destruction time")
async def tap_job_set_destruction(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    if job is None:
        return _no_job(job_id)
    params = await _job_params(request)
    if isinstance(params, Response):
        return params
    raw = params.get("DESTRUCTION", "")
    try:
        requested = _parse_iso(raw)
    except ValueError:
        return PlainTextResponse(f"Invalid DESTRUCTION timestamp {raw!r}", status_code=400)
    latest = job.creation_time + timedelta(seconds=_jobs(request).config.retention_s)
    job.destruction = min(requested, latest)
    return RedirectResponse(f"{_async_base(request)}/{job_id}", status_code=303)


@router.get("/tap/async/{job_id}/quote", summary="UWS quote (not estimated)")
async def tap_job_quote(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    return _no_job(job_id) if job is None else PlainTextResponse("")


@router.get("/tap/async/{job_id}/owner", summary="UWS owner (anonymous)")
async def tap_job_owner(request: Request, job_id: str) -> Response:
    job = _get_job(request, job_id)
    return _no_job(job_id) if job is None else PlainTextResponse("")


# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------


def _default_cli_service(client: Any) -> Any:
    from main import build_service

    return build_service(client=client)


# Replaced in tests to inject an offline service: callable(httpx.AsyncClient) -> service.
_cli_service_factory: Callable[[Any], Any] = _default_cli_service


async def _cli_run(args: Any) -> tuple[bytes, int]:
    import httpx

    from models import Settings

    command = getattr(args, "vo_command", None)
    if command == "tables":
        lines = []
        for table in TABLES:
            lines.append(f"{table.name}  ({len(table.columns)} columns)  {table.description}")
            if getattr(args, "columns", False):
                for col in table.columns:
                    unit = f" [{col.unit}]" if col.unit else ""
                    lines.append(f"    {col.name:<18} {col.datatype:<11} {col.ucd or '':<26}{unit}")
        return ("\n".join(lines) + "\n").encode("utf-8"), 0
    settings = Settings()
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        service = _cli_service_factory(client)
        registry = getattr(service, "registry", None)
        if command == "cone":
            catalogs = [c for c in args.catalogs.split(",") if c.strip()] if args.catalogs else None
            try:
                result = await cone_search(service, args.ra, args.dec, args.sr, verb=args.verb, catalogs=catalogs,
                                           registry=registry)
                return render(result, resolve_format(args.format)[0]), 0
            except VOError as exc:
                return f"Error: {exc}\n".encode(), 2
        if command == "adql":
            params = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": args.query, "RESPONSEFORMAT": args.format}
            if args.maxrec is not None:
                params["MAXREC"] = str(args.maxrec)
            outcome = await handle_tap_query(params, service=service, registry=registry)
            if outcome.error is not None:
                return f"Error: {outcome.error}\n".encode(), 2
            return outcome.body, 0
    return b"Usage: vo {cone,adql,tables}\n", 2


def cli_handler(args: Any) -> int:
    """Execute a ``vo`` subcommand; writes to ``args.output`` or stdout, returns the exit code."""
    body, code = asyncio.run(_cli_run(args))
    output = getattr(args, "output", None)
    if output and code == 0:
        with open(output, "wb") as handle:
            handle.write(body)
        print(f"Wrote {len(body)} bytes to {output}")
    else:
        stream = sys.stdout if code == 0 else sys.stderr
        stream.write(body.decode("utf-8", "replace"))
        stream.flush()
    return code


def register_cli(subparsers: Any) -> None:
    """Add ``vo cone``, ``vo adql`` and ``vo tables`` subcommands (handler: :func:`cli_handler`)."""
    vo = subparsers.add_parser("vo", help="Run Virtual Observatory (Cone Search / ADQL) queries locally")
    vo_sub = vo.add_subparsers(dest="vo_command")
    cone = vo_sub.add_parser("cone", help="Simple Cone Search: crossmatched sources as a VOTable/CSV/JSON")
    cone.add_argument("--ra", type=float, required=True, help="Right ascension (ICRS, degrees)")
    cone.add_argument("--dec", type=float, required=True, help="Declination (ICRS, degrees)")
    cone.add_argument("--sr", type=float, required=True, help="Search radius in DEGREES (e.g. 0.00278 = 10 arcsec)")
    cone.add_argument("--verb", type=int, default=2, choices=[1, 2, 3], help="Columns: 1 minimal, 2 default, 3 all")
    cone.add_argument("--catalogs", help="Comma-separated catalog keys to query (default: all enabled)")
    cone.add_argument("--format", default="votable", choices=["votable", "csv", "tsv", "json"])
    cone.add_argument("--output", help="Write the result to this file instead of stdout")
    adql = vo_sub.add_parser("adql", help="Run an ADQL query against astrosearch.* / TAP_SCHEMA.* tables")
    adql.add_argument("query", help="ADQL query text")
    adql.add_argument("--maxrec", type=int, help="Maximum number of rows (DALI MAXREC)")
    adql.add_argument("--format", default="votable", choices=["votable", "csv", "tsv", "json"])
    adql.add_argument("--output", help="Write the result to this file instead of stdout")
    tables = vo_sub.add_parser("tables", help="List the published tables")
    tables.add_argument("--columns", action="store_true", help="Also list columns with UCDs and units")
    vo.set_defaults(handler=cli_handler)
