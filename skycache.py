"""Local sky mirror ("skycache"): HATS-partitioned Parquet copies of archive regions for
millisecond cone searches that return exactly what the archive would.

What it does
------------
* :func:`mirror_region` downloads every row of a catalog inside a sky region (a cone or a
  list of HEALPix pixels) through the SAME provider adapters the crossmatch service uses
  (``providers.TapProvider`` & co). Only the catalog's primary archive is used: the
  registry's fallback archive (e.g. IRSA Gator for IRSA TAP) returns differently shaped
  rows (extra columns, other float precision), so a tile the primary cannot serve is
  retried, split (timeouts) or reported as failed -- never filled from the fallback. The
  adapters are used through per-mirror copies with their own circuit breakers and no HTTP
  response cache (a refresh always reaches the archive and never disturbs live traffic);
  TAP requests carry MAXREC = TOP so services with a lower default output limit (SIMBAD:
  50000 rows) do not silently cap a tile. Rows are deduplicated by POSITION (a complete
  answer holds every row at each position inside its cone, so the rows at one position
  are kept from one archive request: SDSS objIDs joined to several spectra survive, and so
  do distinct sources sharing an id and catalogs without archive ids) and stored, with
  per-column units/UCDs, in a local HATS catalog.
* The region that is known to be *complete* is recorded as a MOC (IVOA Multi-Order
  Coverage map, MOC 2.0, Fernique et al. 2022, https://www.ivoa.net/documents/MOC/)
  held as sorted disjoint ranges of order-29 NESTED HEALPix indices. Only rows inside it
  are stored, and each covered cell holds the rows of ONE archive snapshot (the mirror
  that last covered it, whose time the store records): a refresh replaces the rows of
  the cells it covers and leaves the others alone, so a source whose archive position
  changed between two mirrors is never stored twice (see ``_merge_masks``).
* Mirrors run on their own event loop in a worker thread (archive requests still go
  through the caller's HTTP client and loop), so converting tens of thousands of rows
  never stalls an API server's loop.
* :class:`LocalProvider` is a :class:`providers.CatalogProvider` that answers a cone
  from the store only when the cone (after the same epoch widening the remote provider
  applies, :func:`models.plan_cone`) lies entirely inside the MOC; otherwise it raises
  :class:`CoverageError` so the caller can fall back to the archive
  (:class:`SkyCacheProvider` / :func:`wrap_providers` do that automatically; they also
  fall back when the local store is unreadable). Rows are converted by the providers'
  own ``_HTTPProvider._sources`` so the :class:`models.CatalogSource` objects are
  identical in shape to the remote ones (same canonical fields, epochs, positional
  errors, metadata, nearest-first TOP N semantics). Each source's provenance carries the
  time its row was retrieved from the archive, not the time of the local lookup.
* Invalidation: the store records a fingerprint of the registry definition fields that
  decide which rows and columns the archive returns (provider, endpoint, table, columns,
  where/filters/joins, id/ra/dec fields, exclude_values, null_sentinels, column_units,
  ...). When the query-time definition differs, :class:`LocalProvider` raises
  ``CoverageError(reason="definition_changed")`` (the archive answers instead) and the
  next mirror of that catalog starts a fresh store. The fingerprint is normalized so the
  definitions ``crossmatch.QueryExecutor`` derives from planner/builder plans (which add
  copies of table/catalog and post-fetch AdvancedQuery filters) match the registry's.
  Optionally a maximum data age (``$SKYCACHE_MAX_AGE_DAYS`` or the catalog parameter
  ``skycache_max_age_days``) makes older data count as not covered (``reason="stale"``);
  the store records when each part of its coverage was last mirrored, so a refreshed
  region is fresh even where it has no rows.

Resources: local cones select the nearest ``TOP N+1`` rows with numpy before any row is
converted to Python, so their cost depends on the rows returned, not on the cone's
density; the per-row cost is the providers' own ``normalize_source_record`` conversion
(identical to the remote path). A mirror merges old and new rows as Arrow tables (fetched
rows are converted to Arrow per archive request) and rewrites the catalog files once
(O(store size) disk I/O, memory about the Arrow size of the store plus the new rows).
Local queries run in a worker thread (``asyncio.to_thread``), so executor timeouts apply
and the event loop never blocks on Parquet reads.

Storage layout (HATS)
---------------------
Each catalog is a HATS catalog (Hierarchical Adaptive Tiling Scheme, LINCC Frameworks;
spec https://hats.readthedocs.io/en/stable/guide/directory_scheme.html, IVOA note
"HATS: A standard for large catalogs", 2025)::

    <root>/<catalog>/hats.properties          java properties (obs_collection, dataproduct_type=object,
    <root>/<catalog>/properties               hats_col_ra/_dec, hats_col_healpix, hats_nrows, hats_order, ...)
    <root>/<catalog>/partition_info.csv       "Norder,Npix" rows, sorted
    <root>/<catalog>/dataset/_common_metadata Parquet schema (field metadata carry unit/ucd/description)
    <root>/<catalog>/dataset/_metadata        schema + row-group metadata of every leaf file
    <root>/<catalog>/dataset/Norder=K/Dir=D/Npix=P.parquet   D = (P // 10000) * 10000
    <root>/<catalog>/skycache.json            (ours) coverage MOC (+ coverage by retrieval time), column
                                              units/UCDs, mirror log
    <root>/.<catalog>.lock                    (ours, outside the HATS tree) cross-process write lock

    <root>/<catalog>/coverage_moc.fits        (ours) the coverage as an IVOA MOC 2.0 FITS file (NUNIQ)

Leaf files hold ``_healpix_29`` (order-29 NESTED index of the row, int64, HATS spatial
index, rows sorted by it), our canonical columns (``_sc_ra``, ``_sc_dec``, ``_sc_source_id``,
``_sc_epoch``, ``_sc_pmra``, ``_sc_pmdec``, ``_sc_pos_err_arcsec``, ...) and every raw
archive column. Partitions are adaptive exactly like ``hats-import``: a pixel is split into
its 4 children while it holds more than ``hats_max_rows`` rows. The layout, file names,
``Dir`` rule, ``_healpix_29`` column and properties keys were checked against the ``hats``
0.11 source (``hats.io.paths``, ``hats.pixel_math.spatial_index``,
``hats.catalog.dataset.table_properties``) and against the public Gaia DR3 HATS catalog at
https://data.lsdb.io/hats/gaia_dr3/gaia (its ``hats.properties``/``_common_metadata``: no
Norder/Dir/Npix columns inside leaf files). ``lsdb.open_catalog(<root>/<catalog>)`` reads
the store (tested). HEALPix indices use ``astropy_healpix`` (NESTED, Gorski et al. 2005,
ApJ 622, 759); they equal ``hats``' own ``cdshealpix``-based ``compute_spatial_index``
(verified on 2e5 random positions incl. poles and RA=0/360).

Differences from a hats-import catalog (documented, all allowed by the spec): no
``skymap.fits``/``point_map.fits`` (optional), no margin cache, and raw columns whose
values mix Python types (e.g. int and str) are stored JSON-encoded as strings (listed in
``skycache.json`` ``json_columns``) so they round-trip exactly.

A mirror is a PARTIAL copy of the catalog: HATS readers (``lsdb``, ``hats``) return
exactly the archive's rows inside the coverage and nothing outside it, so a query region
that leaves the coverage is silently incomplete there -- intersect it with
``coverage_moc.fits`` (e.g. ``mocpy.MOC.from_fits``; ``SkyCache.covers`` does this for
cones). HATS's own ``point_map.fits`` is not used for this: hats derives its MOC from the
point COUNTS (a covered pixel without rows would drop out) at one fixed order.

Remote HATS catalogs (optional)
-------------------------------
:class:`HatsRemoteProvider` answers cones from a public HATS catalog through ``lsdb``
(``pip install lsdb``; e.g. https://data.lsdb.io/hats/gaia_dr3/gaia) and converts the rows
with the registry definition, so it too returns ordinary CatalogSource objects.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy_healpix import HEALPix, nside_to_pixel_resolution
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from crossmatch import QueryExecutor
from models import (
    COMPUTED_COLUMNS,
    AstroSearchError,
    CatalogDefinition,
    CatalogRegistry,
    CatalogSource,
    ColumnMeta,
    InvalidCoordinateError,
    QueryPlan,
    Settings,
    Target,
    bare_column_name,
    plan_cone,
    ucd_field_map,
    validate_target,
)
from providers import CacheManager, CatalogProvider, EndpointGuard, QueryResult, TapProvider, _HTTPProvider, provider_map

logger = logging.getLogger("astrosearch.skycache")

__all__ = [
    "CoverageError",
    "HatsRemoteProvider",
    "LocalProvider",
    "MirrorError",
    "MirrorReport",
    "Moc",
    "SkyCache",
    "SkyCacheProvider",
    "cone_moc",
    "cone_pixels",
    "definition_fingerprint",
    "definition_signature",
    "healpix29",
    "mirror_region",
    "partition_orders",
    "pixels_inside_cone",
    "register_cli",
    "router",
    "wrap_providers",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMAT_NAME = "astrosearch-skycache"
FORMAT_VERSION = 1
BUILDER = f"astrosearch-skycache v{FORMAT_VERSION}"
HATS_VERSION = "v0.1"  # value written by hats-import (see the public Gaia DR3 catalog's properties)

COVERAGE_MOC_FILE = "coverage_moc.fits"  # (ours) IVOA MOC 2.0 FITS of the coverage, at the catalog root
SPATIAL_INDEX_COLUMN = "_healpix_29"  # hats.pixel_math.spatial_index.SPATIAL_INDEX_COLUMN
SPATIAL_INDEX_ORDER = 29  # hats.pixel_math.spatial_index.SPATIAL_INDEX_ORDER
NPIX_29 = 12 * 4**29  # number of order-29 pixels on the sphere (< 2**63)
DIR_DIVISOR = 10_000  # hats.io.paths.pixel_directory: Dir = int(Npix / 10000) * 10000

# Canonical columns stored next to the raw archive columns.
SC_RA = "_sc_ra"
SC_DEC = "_sc_dec"
SC_ID = "_sc_source_id"
SC_EPOCH = "_sc_epoch"
SC_PMRA = "_sc_pmra"
SC_PMDEC = "_sc_pmdec"
SC_POSERR = "_sc_pos_err_arcsec"
SC_PROVIDER = "_sc_provider"
SC_RETRIEVED = "_sc_retrieved_at"
SC_COLSET = "_sc_colset"
SYSTEM_COLUMNS = (SPATIAL_INDEX_COLUMN, SC_ID, SC_RA, SC_DEC, SC_EPOCH, SC_PMRA, SC_PMDEC, SC_POSERR, SC_PROVIDER,
                  SC_RETRIEVED, SC_COLSET)
_SYSTEM_META: dict[str, tuple[str | None, str | None, str]] = {
    SPATIAL_INDEX_COLUMN: (None, "pos.healpix", "HEALPix NESTED index of (_sc_ra, _sc_dec) at order 29 (HATS spatial index)"),
    SC_ID: (None, "meta.id;meta.main", "canonical source identifier (CatalogSource.source_id)"),
    SC_RA: ("deg", "pos.eq.ra", "canonical ICRS right ascension of the archive row (CatalogSource.ra)"),
    SC_DEC: ("deg", "pos.eq.dec", "canonical ICRS declination of the archive row (CatalogSource.dec)"),
    SC_EPOCH: ("yr", "time.epoch", "Julian year of the position (CatalogSource.epoch)"),
    SC_PMRA: ("mas/yr", "pos.pm;pos.eq.ra", "proper motion in RA * cos(dec) (CatalogSource.proper_motion_ra_masyr)"),
    SC_PMDEC: ("mas/yr", "pos.pm;pos.eq.dec", "proper motion in Dec (CatalogSource.proper_motion_dec_masyr)"),
    SC_POSERR: ("arcsec", "stat.error;pos", "1-sigma circular positional error (CatalogSource.positional_error_arcsec)"),
    SC_PROVIDER: (None, "meta.note", "provider adapter that fetched the row (primary or fallback archive)"),
    SC_RETRIEVED: (None, "time.processing", "UTC time the row was retrieved from the archive"),
    SC_COLSET: (None, "meta.code", "index into skycache.json 'colsets' (the archive columns of this row)"),
}

# Providers whose cone results reveal server-side truncation (TOP N+1 probe row, OVERFLOW
# status or the whole cone being returned), which mirroring relies on to know a tile is
# complete. HEASARC Xamin has no row limit parameter, so completeness cannot be verified.
MIRRORABLE_PROVIDERS = frozenset({"tap", "irsa_gator", "mast", "sdss"})
# Default id column each adapter passes to _sources (see providers.*.query).
_DEFAULT_ID = {"tap": None, "irsa_gator": "designation", "mast": "objID", "sdss": "objID", "heasarc_xamin": "name"}

# Coverage bookkeeping: the root cone of a mirror records the order-C pixels lying fully
# inside it, with C chosen so a pixel is <= radius / 32 (loses a < ~2-pixel rim).
COVERAGE_RES_FRACTION = 1.0 / 32.0
MAX_COVERAGE_ORDER = 20
MAX_TILE_ORDER = 18  # a tile still truncated at this order (0.8") is reported as failed
MAX_QUERY_ORDER = 24
# Safety margins for polygon-vs-circle tests, in units of the pixel resolution. Pixel edges
# are sampled with BOUNDARY_STEP points per side; the chord sagitta between samples is
# < res / 500, far below the margin.
BOUNDARY_STEP = 8
PIXEL_MARGIN = 0.02
# A position counts as inside a complete request cone (whose answer is authoritative there)
# only when it is at least this far inside the cone's edge (36 micro-arcsec: far above the
# float rounding of the archives' own separations, far below any astrometric change).
CONE_EDGE_MARGIN_DEG = 1e-8
# cone_moc: astropy-healpix's cone search enumerates pixels at a cost that grows faster than
# their number (0.3 ms for 100 pixels, 320 ms for 7000); above this estimated count the
# hierarchical search is used instead.
CONE_MOC_DIRECT_PIXELS = 600

# Environment configuration.
ENV_PATH = "SKYCACHE_PATH"
ENV_TILE_ROWS = "SKYCACHE_TILE_ROWS"
ENV_PARTITION_ROWS = "SKYCACHE_PARTITION_ROWS"
ENV_MAX_RADIUS = "SKYCACHE_MAX_RADIUS_DEG"
ENV_MAX_QUERIES = "SKYCACHE_MAX_QUERIES"
ENV_MAX_AGE_DAYS = "SKYCACHE_MAX_AGE_DAYS"
DEFAULT_TILE_ROWS = 10_000
DEFAULT_PARTITION_ROWS = 200_000
DEFAULT_MAX_RADIUS_DEG = 1.0
DEFAULT_MAX_QUERIES = 256
DEFAULT_CONCURRENCY = 4
DEFAULT_REQUEST_TIMEOUT_S = 30.0  # archive request limit for catalogs without a registry timeout_seconds
MAX_PARTITION_ORDER = 20

# Tile failures. A tile whose archive request timed out is split into its 4 children
# like a truncated one (a smaller tile is a smaller, faster query), for at most
# TIMEOUT_SPLIT_DEPTH consecutive levels: an archive that times out even on 1/64 of the
# area is slow or down, not overloaded by the tile. A tile whose archive was unavailable
# (HTTP 5xx, throttled) is retried UNAVAILABLE_RETRIES times, after RETRY_DELAY_S, then
# twice that, ... seconds. Every attempt that reaches the archive counts against max_queries.
TIMEOUT_SPLIT_DEPTH = 3
UNAVAILABLE_RETRIES = 2
RETRY_DELAY_S = 1.0
_UNAVAILABLE_ERRORS = frozenset({"CatalogUnavailableError", "RateLimitedError"})
_TIMEOUT_ERRORS = frozenset({"QueryTimeoutError"})

# A mirror fetches through its OWN copies of the archive adapters: its own circuit breakers
# (never shared with live crossmatch traffic, which the mirror's bursts of tile requests
# would otherwise block, and whose failures would otherwise abort the mirror) and no HTTP
# response cache (a refresh must reach the archive; tiles would also flood the service's
# cache). The mirror's breaker opens after MIRROR_FAILURE_THRESHOLD consecutive failed
# requests (timeouts included) and lets one probe through after MIRROR_RECOVERY_S; a tile
# refused by the open circuit did not reach the archive, so it waits (at most
# CIRCUIT_WAIT_LIMIT_S in total, polling every CIRCUIT_POLL_S) and tries again without
# using the query budget. Requests per second stay as configured for the service
# ($PROVIDER_REQUESTS_PER_SECOND): mirrors are as polite as the live service.
MIRROR_FAILURE_THRESHOLD = 10
MIRROR_RECOVERY_S = 5.0
CIRCUIT_WAIT_LIMIT_S = 60.0
CIRCUIT_POLL_S = 0.5

# TAP answers in JSON or CSV carry no QUERY_STATUS, so a service that silently caps its
# answer below TOP N+1 rows (SIMBAD: default outputLimit 50000 rows) looks complete.
# Mirror requests therefore send MAXREC = TOP (DALI/TAP 1.1 section 2.7.4; honoured by
# every registry TAP service, verified live), and the service's <outputLimit> values are
# read once per mirror from its VOSI capabilities: an answer of exactly that many rows
# (fewer than TOP) is treated as truncated and the tile is split.
_TOP_CLAUSE = re.compile(r"^\s*SELECT\s+TOP\s+(\d+)\b", re.IGNORECASE)
_OUTPUT_LIMIT = re.compile(r"<(?:[A-Za-z0-9_]+:)?outputLimit\b[^>]*>(.*?)</(?:[A-Za-z0-9_]+:)?outputLimit\s*>",
                           re.IGNORECASE | re.DOTALL)
_LIMIT_VALUE = re.compile(r"<(?:[A-Za-z0-9_]+:)?(default|hard)\b([^>]*)>\s*(\d+)\s*<", re.IGNORECASE)

# Reading the previous copy of a catalog while merging a refresh: transient errors (a
# Windows scanner holding a leaf file, a store written by another process) are retried
# READ_ATTEMPTS times READ_RETRY_DELAY_S apart; if the copy still cannot be read, the
# mirror aborts and keeps it (MirrorStoreError). Only a copy that is provably damaged
# (unparseable Parquet or skycache.json, missing leaf files) is replaced by the new mirror.
READ_ATTEMPTS = 3
READ_RETRY_DELAY_S = 0.2

# Directory swap (SkyCache.write). On Windows an antivirus or indexer handle on freshly
# written files makes os.replace fail with PermissionError for a moment: retried
# REPLACE_ATTEMPTS times with exponential backoff from REPLACE_DELAY_S. Hidden swap
# directories (".<catalog>.tmp-*" being written, ".<catalog>.old-*" backups,
# ".<catalog>.del-*" deleted catalogs) left by a crashed or failed write are repaired: a
# backup whose catalog is missing is restored, other backups, deleted trees and temporary
# trees older than STALE_SWAP_SECONDS are removed. Writes, deletes and repairs of a
# catalog hold an exclusive cross-process lock (an OS lock on "<root>/.<catalog>.lock",
# released by the OS if the process dies), so opening the store in another process never
# "repairs" a swap that is in progress; a writer waits at most LOCK_TIMEOUT_S for it.
REPLACE_ATTEMPTS = 6
REPLACE_DELAY_S = 0.05
STALE_SWAP_SECONDS = 600.0
LOCK_TIMEOUT_S = 300.0
LOCK_POLL_S = 0.05
_SWAP_DIR = re.compile(r"^\.(?P<catalog>[A-Za-z0-9_][A-Za-z0-9_.-]{0,99})\.(?P<kind>tmp|old|del)-[0-9a-f]{8}$")

# Definition fields that only the query-time conversion reads (from the definition given
# to LocalProvider): changing them does not change which rows/columns the archive returns,
# so they are left out of the store's definition fingerprint. Everything else (provider,
# endpoint, table, catalog, and all other parameters: columns, where, filters, joins,
# id/ra/dec fields, field_map, exclude_values, null_sentinels, column_units, distance,
# mode, ...) is part of it.
_QUERY_TIME_FIELDS = frozenset({
    "name", "wavelength", "enabled", "description", "query_method", "units", "profiles", "epoch", "epoch_format",
    "pos_error", "citation", "acknowledgement", "max_rows", "timeout_seconds", "coverage",
})
# Parameters no adapter sends to the archive: query-time conversion settings, and the
# filters crossmatch.QueryBuilder adds to every plan (AdvancedQuery object_types,
# spectral_types, count_threshold, time_period), which AdvancedQuery applies to the rows
# AFTER they were fetched (crossmatch.AdvancedQuery.matches).
_QUERY_TIME_PARAMETERS = frozenset({
    "fallback", "epoch_range", "single_epoch_positions", "positional_error_field", "positional_error_unit",
    "skycache_max_age_days", "object_types", "spectral_types", "count_threshold", "time_period",
})
# crossmatch.QueryPlanner / QueryBuilder copy the definition's table and catalog into every
# plan's parameters, and QueryExecutor.definition_for lifts them back into the top-level
# fields, which are what the adapters read (no adapter reads parameters['table'/'catalog']).
# A copy that equals the top-level field (or is empty) changes nothing and is not
# fingerprinted; one that differs would change the top-level field of executed plans, so
# it stays in the fingerprint.
_PLAN_COPIED_PARAMETERS = frozenset({"table", "catalog"})


def default_store_path() -> Path:
    """``$SKYCACHE_PATH`` or ``~/.astrosearch/skycache``."""
    configured = os.getenv(ENV_PATH)
    return Path(configured).expanduser() if configured else Path.home() / ".astrosearch" / "skycache"


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CoverageError(AstroSearchError):
    """The local store cannot answer a cone completely (not mirrored or only partly covered).

    ``covered_fraction`` is the fraction of the cone's HEALPix pixels (at ``order``) that lie
    inside the recorded coverage. Callers fall back to the remote archive.
    """

    def __init__(self, message: str, *, catalog: str, ra: float | None = None, dec: float | None = None,
                 radius_arcsec: float | None = None, covered_fraction: float = 0.0, order: int | None = None,
                 reason: str = "not_covered") -> None:
        super().__init__(message)
        self.catalog = catalog
        self.ra = ra
        self.dec = dec
        self.radius_arcsec = radius_arcsec
        self.covered_fraction = covered_fraction
        self.order = order
        self.reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {"catalog": self.catalog, "ra": self.ra, "dec": self.dec, "radius_arcsec": self.radius_arcsec,
                "covered_fraction": self.covered_fraction, "order": self.order, "reason": self.reason,
                "message": str(self)}


class MirrorError(AstroSearchError):
    """A mirror request is invalid or no part of the region could be fetched."""


class MirrorInputError(MirrorError, ValueError):
    """The mirror request itself is invalid (unknown catalog, region too large, ...)."""


class MirrorStoreError(MirrorError):
    """The local store could not be written (file system error); the previous state is kept."""


class _StaleStateError(Exception):
    """A cached catalog state no longer matches the files on disk (the tree was swapped)."""


# ---------------------------------------------------------------------------
# Definition fingerprint (cache invalidation)
# ---------------------------------------------------------------------------


def definition_signature(definition: CatalogDefinition | Mapping[str, Any]) -> dict[str, Any]:
    """The definition fields that decide which rows and columns the archive returns (JSON form).

    Normalized so that the definition ``crossmatch.QueryExecutor.definition_for`` builds
    from a ``QueryPlanner``/``QueryBuilder`` plan (registry parameters plus the plan's
    copies of table/catalog and the AdvancedQuery post-fetch filters) has the same
    signature as the registry definition it came from. Idempotent (a signature's
    signature is itself).
    """
    data = definition.as_dict() if isinstance(definition, CatalogDefinition) else dict(definition)
    data = json.loads(json.dumps(data, default=str, sort_keys=True))
    out = {k: v for k, v in data.items() if k not in _QUERY_TIME_FIELDS and k != "parameters"}
    params = data.get("parameters") or {}
    out["parameters"] = {
        k: v for k, v in params.items()
        if k not in _QUERY_TIME_PARAMETERS
        and not (k in _PLAN_COPIED_PARAMETERS and (v in (None, "") or v == data.get(k)))
    }
    return out


def definition_fingerprint(definition: CatalogDefinition | Mapping[str, Any]) -> str:
    """SHA-256 of :func:`definition_signature` (stored in skycache.json)."""
    text = json.dumps(definition_signature(definition), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _recorded_signature(meta: Mapping[str, Any]) -> dict[str, Any] | None:
    """The signature of the definition a store was mirrored with, normalized by the CURRENT
    :func:`definition_signature` (stores written by older versions recorded it un-normalized)."""
    source = meta.get("definition") or meta.get("definition_signature")
    return definition_signature(source) if source else None


def _signature_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Names of the signature fields that differ ('parameters.<key>' for parameters).

    A key present on one side only counts as a change even when the other side's value
    would be ``None`` (the fingerprints differ then too).
    """
    def differs(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> bool:
        return (key in a) != (key in b) or a.get(key) != b.get(key)

    changed = [k for k in sorted(set(old) | set(new)) if k != "parameters" and differs(old, new, k)]
    op, np_ = old.get("parameters") or {}, new.get("parameters") or {}
    changed += [f"parameters.{k}" for k in sorted(set(op) | set(np_)) if differs(op, np_, k)]
    return changed


# ---------------------------------------------------------------------------
# HEALPix helpers (astropy-healpix, NESTED)
# ---------------------------------------------------------------------------


@functools.cache
def _healpix(order: int) -> HEALPix:
    return HEALPix(nside=2**order, order="nested")


@functools.cache
def pixel_resolution_deg(order: int) -> float:
    """Square root of the pixel area (the usual HEALPix 'resolution'), degrees."""
    return float(nside_to_pixel_resolution(2**order).to_value(u.deg))


def order_for_resolution(res_deg: float, *, lo: int = 0, hi: int = SPATIAL_INDEX_ORDER) -> int:
    """Smallest order in [lo, hi] whose pixel resolution is <= ``res_deg``."""
    for order in range(lo, hi + 1):
        if pixel_resolution_deg(order) <= res_deg:
            return order
    return hi


def healpix29(ra_deg: Any, dec_deg: Any) -> np.ndarray:
    """Order-29 NESTED HEALPix index (the HATS ``_healpix_29`` spatial index), int64."""
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if ra.size == 0:
        return np.zeros(ra.shape, dtype=np.int64)
    return np.asarray(_healpix(SPATIAL_INDEX_ORDER).lonlat_to_healpix(ra * u.deg, dec * u.deg), dtype=np.int64)


def angular_sep_deg(ra0: float, dec0: float, ra: Any, dec: Any) -> np.ndarray:
    """Vectorized haversine separation in degrees (same formula as models.haversine_arcsec)."""
    ra = np.radians(np.asarray(ra, dtype=np.float64))
    dec = np.radians(np.asarray(dec, dtype=np.float64))
    r0, d0 = math.radians(ra0), math.radians(dec0)
    dlmb = np.remainder(ra - r0 + math.pi, 2 * math.pi) - math.pi
    a = np.sin((dec - d0) / 2) ** 2 + math.cos(d0) * np.cos(dec) * np.sin(dlmb / 2) ** 2
    return np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a))))


def position_angle_deg(ra0: float, dec0: float, ra: Any, dec: Any) -> np.ndarray:
    """Position angle of (ra, dec) seen from (ra0, dec0), degrees east of north in [0, 360)."""
    a = np.radians(np.asarray(ra, dtype=np.float64)) - math.radians(ra0)
    d = np.radians(np.asarray(dec, dtype=np.float64))
    d0 = math.radians(dec0)
    pa = np.degrees(np.arctan2(np.sin(a), math.cos(d0) * np.tan(d) - math.sin(d0) * np.cos(a)))
    return np.remainder(pa, 360.0)


def cone_pixels(ra: float, dec: float, radius_deg: float, order: int) -> np.ndarray:
    """All order-``order`` pixels overlapping the cone (inclusive; sorted int64).

    ``HEALPix.cone_search_lonlat`` returns every pixel that overlaps the cone, including
    partially (astropy-healpix docs). Checked against a dense (1/256-pixel) sampling of the
    pixels that ``cdshealpix``'s approximate search adds: none truly overlapped. The radius
    is still inflated by 2% of a pixel so slivers can never be missed.
    """
    res = pixel_resolution_deg(order)
    radius = min(180.0, radius_deg + PIXEL_MARGIN * res)
    pixels = _healpix(order).cone_search_lonlat(ra * u.deg, dec * u.deg, radius * u.deg)
    return np.unique(np.asarray(pixels, dtype=np.int64))


def _boundary_lonlat(order: int, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _healpix(order).boundaries_lonlat(pixels, step=BOUNDARY_STEP)
    return lon.to_value(u.deg), lat.to_value(u.deg)


def _inside_mask(ra: float, dec: float, radius_deg: float, order: int, pixels: np.ndarray) -> np.ndarray:
    """Which order-``order`` pixels lie entirely inside the cone: every sampled boundary point
    within ``radius - 2% res`` of the centre (the farthest point of a pixel from any point
    lies on its boundary; sampling misses it by < res/500)."""
    inside = np.zeros(pixels.size, dtype=bool)
    margin = PIXEL_MARGIN * pixel_resolution_deg(order)
    for start in range(0, pixels.size, 20_000):
        chunk = pixels[start:start + 20_000]
        lon, lat = _boundary_lonlat(order, chunk)
        far = angular_sep_deg(ra, dec, lon.ravel(), lat.ravel()).reshape(lon.shape).max(axis=1)
        inside[start:start + chunk.size] = far + margin <= radius_deg
    return inside


def pixels_inside_cone(ra: float, dec: float, radius_deg: float, order: int) -> np.ndarray:
    """Order-``order`` pixels lying entirely inside the cone (conservative; sorted int64).

    A pixel is inside when every sampled boundary point is within ``radius - 2% res`` of the
    centre (:func:`_inside_mask`). Found hierarchically (:func:`_cone_cells`).
    """
    moc = _cone_cells(ra, dec, radius_deg, order, inside_only=True)
    shift = 2 * (SPATIAL_INDEX_ORDER - int(order))
    if moc.empty:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(a >> shift, b >> shift, dtype=np.int64) for a, b in moc.ranges.tolist()])


def pixel_center(order: int, pixel: int) -> tuple[float, float]:
    lon, lat = _healpix(order).healpix_to_lonlat(np.array([pixel], dtype=np.int64))
    return float(lon.to_value(u.deg)[0]) % 360.0, float(lat.to_value(u.deg)[0])


def pixel_geometry(order: int, pixels: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ra, dec, circumradius) of many pixels at once, vectorized; see :func:`pixel_circumradius_deg`."""
    pix = np.asarray(pixels, dtype=np.int64).ravel()
    lon, lat = _healpix(order).healpix_to_lonlat(pix)
    ra = np.mod(lon.to_value(u.deg), 360.0)
    dec = lat.to_value(u.deg)
    blon, blat = _boundary_lonlat(order, pix)
    # Haversine separation of each pixel's boundary samples from its centre (as angular_sep_deg).
    r0, d0 = np.radians(ra)[:, None], np.radians(dec)[:, None]
    br, bd = np.radians(blon), np.radians(blat)
    dlmb = np.remainder(br - r0 + math.pi, 2 * math.pi) - math.pi
    a = np.sin((bd - d0) / 2) ** 2 + np.cos(d0) * np.cos(bd) * np.sin(dlmb / 2) ** 2
    far = np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))).max(axis=1)
    return ra, dec, far + PIXEL_MARGIN * pixel_resolution_deg(order)


def pixel_circumradius_deg(order: int, pixel: int) -> float:
    """Radius of a circle around the pixel centre that encloses the whole pixel (+2% res)."""
    return float(pixel_geometry(order, [pixel])[2][0])


def pixel_ranges(order: int, pixels: Any) -> np.ndarray:
    """[start, end) order-29 index ranges of order-``order`` pixels, shape (n, 2)."""
    pix = np.asarray(pixels, dtype=np.int64).ravel()
    shift = 2 * (SPATIAL_INDEX_ORDER - int(order))
    return np.stack([pix << shift, (pix + 1) << shift], axis=1) if pix.size else np.zeros((0, 2), dtype=np.int64)


def partition_orders(h29: np.ndarray, threshold: int, *, max_order: int = MAX_PARTITION_ORDER,
                     min_order: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive HATS partitioning: (order, pixel) of every row.

    Top-down: a pixel keeps its rows when it holds at most ``threshold`` of them, otherwise
    they move to its 4 children (hats-import's ``pixel_threshold`` rule); ``max_order``
    stops the recursion (such partitions may exceed the threshold).
    """
    h29 = np.asarray(h29, dtype=np.int64)
    orders = np.full(h29.size, -1, dtype=np.int16)
    pixels = np.zeros(h29.size, dtype=np.int64)
    remaining = np.arange(h29.size)
    for order in range(min_order, max_order + 1):
        if remaining.size == 0:
            break
        pix = h29[remaining] >> (2 * (SPATIAL_INDEX_ORDER - order))
        _uniq, inverse, counts = np.unique(pix, return_inverse=True, return_counts=True)
        done = np.ones(pix.size, dtype=bool) if order == max_order else counts[inverse] <= threshold
        orders[remaining[done]] = order
        pixels[remaining[done]] = pix[done]
        remaining = remaining[~done]
    return orders, pixels


# ---------------------------------------------------------------------------
# Multi-Order Coverage map
# ---------------------------------------------------------------------------


class Moc:
    """A sky region as sorted, disjoint, non-adjacent [start, end) ranges of order-29
    NESTED HEALPix indices -- the "range" representation of an IVOA MOC 2.0 space MOC
    (Fernique et al. 2022, IVOA Recommendation MOC 2.0, sections 3-4)."""

    __slots__ = ("ranges",)

    def __init__(self, ranges: Any = None) -> None:
        arr = np.asarray(ranges if ranges is not None else np.zeros((0, 2)), dtype=np.int64).reshape(-1, 2)
        self.ranges = self._normalize(arr)

    @staticmethod
    def _normalize(ranges: np.ndarray) -> np.ndarray:
        ranges = ranges[ranges[:, 1] > ranges[:, 0]]
        if ranges.shape[0] <= 1:
            return ranges.copy()
        ranges = ranges[np.argsort(ranges[:, 0], kind="stable")]
        ends = np.maximum.accumulate(ranges[:, 1])
        new_block = np.empty(ranges.shape[0], dtype=bool)
        new_block[0] = True
        new_block[1:] = ranges[1:, 0] > ends[:-1]
        first = np.flatnonzero(new_block)
        last = np.append(first[1:] - 1, ranges.shape[0] - 1)
        return np.stack([ranges[first, 0], ends[last]], axis=1)

    @classmethod
    def from_pixels(cls, order: int, pixels: Any) -> Moc:
        return cls(pixel_ranges(order, pixels))

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> Moc:
        return cls(np.asarray((data or {}).get("ranges") or [], dtype=np.int64).reshape(-1, 2))

    @classmethod
    def from_cone(cls, ra: float, dec: float, radius_deg: float, order: int) -> Moc:
        """The pixels fully inside a cone (see :func:`pixels_inside_cone`)."""
        return _cone_cells(ra, dec, radius_deg, order, inside_only=True)

    def union(self, other: Moc) -> Moc:
        return Moc(np.concatenate([self.ranges, other.ranges]))

    def __or__(self, other: Moc) -> Moc:
        return self.union(other)

    def __len__(self) -> int:
        return int(self.ranges.shape[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Moc) and np.array_equal(self.ranges, other.ranges)

    @property
    def empty(self) -> bool:
        return self.ranges.shape[0] == 0

    def intersection(self, other: Moc) -> Moc:
        a, b = self.ranges.tolist(), other.ranges.tolist()
        i = j = 0
        out: list[tuple[int, int]] = []
        while i < len(a) and j < len(b):
            start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            if start < end:
                out.append((start, end))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return Moc(np.array(out, dtype=np.int64).reshape(-1, 2))

    def __and__(self, other: Moc) -> Moc:
        return self.intersection(other)

    def difference(self, other: Moc) -> Moc:
        """The part of this region outside ``other``."""
        if self.empty or other.empty:
            return Moc(self.ranges)
        out: list[tuple[int, int]] = []
        b = other.ranges.tolist()
        j = 0
        for start, end in self.ranges.tolist():
            while j < len(b) and b[j][1] <= start:
                j += 1
            k = j
            cursor = start
            while k < len(b) and b[k][0] < end:
                if b[k][0] > cursor:
                    out.append((cursor, b[k][0]))
                cursor = max(cursor, b[k][1])
                if cursor >= end:
                    break
                k += 1
            if cursor < end:
                out.append((cursor, end))
        return Moc(np.array(out, dtype=np.int64).reshape(-1, 2))

    def __sub__(self, other: Moc) -> Moc:
        return self.difference(other)

    def overlaps_ranges(self, ranges: np.ndarray) -> bool:
        """Whether any of the [start, end) query ranges shares a cell with this region."""
        return bool(self.covered_cells(ranges).sum()) if not self.empty else False

    @property
    def cells29(self) -> int:
        """Number of order-29 cells (exact integer area measure)."""
        return int(sum(e - s for s, e in self.ranges.tolist()))

    def contains_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Boolean per query range: fully inside the coverage."""
        ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        if self.empty or ranges.shape[0] == 0:
            return np.zeros(ranges.shape[0], dtype=bool)
        idx = np.searchsorted(self.ranges[:, 0], ranges[:, 0], side="right") - 1
        ok = idx >= 0
        safe = np.clip(idx, 0, None)
        return ok & (self.ranges[safe, 1] >= ranges[:, 1]) & (self.ranges[safe, 0] <= ranges[:, 0])

    def contains_pixels(self, order: int, pixels: Any) -> np.ndarray:
        return self.contains_ranges(pixel_ranges(order, pixels))

    def contains_points(self, ra: Any, dec: Any) -> np.ndarray:
        h = healpix29(ra, dec)
        return self.contains_ranges(np.stack([h, h + 1], axis=1))

    def covered_cells(self, ranges: np.ndarray) -> np.ndarray:
        """Order-29 cells of each [start, end) query range that lie inside the coverage (vectorized)."""
        ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        if self.empty or ranges.shape[0] == 0:
            return np.zeros(ranges.shape[0], dtype=np.int64)
        starts, ends = self.ranges[:, 0], self.ranges[:, 1]
        below = np.concatenate([[0], np.cumsum(ends - starts)])  # covered cells before range i

        def covered_below(x: np.ndarray) -> np.ndarray:
            i = np.searchsorted(starts, x, side="right") - 1
            safe = np.clip(i, 0, None)
            partial = np.clip(np.minimum(x, ends[safe]) - starts[safe], 0, None)
            return np.where(i >= 0, below[safe] + partial, 0)

        return covered_below(ranges[:, 1]) - covered_below(ranges[:, 0])

    @property
    def sky_fraction(self) -> float:
        return float((self.ranges[:, 1] - self.ranges[:, 0]).sum()) / NPIX_29 if not self.empty else 0.0

    @property
    def area_deg2(self) -> float:
        return self.sky_fraction * 4.0 * math.pi * (180.0 / math.pi) ** 2

    def to_orders(self) -> dict[int, np.ndarray]:
        """Decompose into maximal aligned pixels per order (the NUNIQ/ASCII form)."""
        out: dict[int, list[int]] = {}
        for start, end in self.ranges.tolist():
            s = int(start)
            e = int(end)
            while s < e:
                order = SPATIAL_INDEX_ORDER
                while order > 0:
                    size = 4 ** (SPATIAL_INDEX_ORDER - (order - 1))
                    if s % size == 0 and s + size <= e:
                        order -= 1
                    else:
                        break
                size = 4 ** (SPATIAL_INDEX_ORDER - order)
                out.setdefault(order, []).append(s // size)
                s += size
        return {k: np.array(sorted(v), dtype=np.int64) for k, v in sorted(out.items())}

    @property
    def max_order(self) -> int:
        orders = self.to_orders()
        return max(orders) if orders else 0

    def to_ascii(self) -> str:
        """IVOA MOC 2.0 ASCII serialization ("order/ipix ipix1-ipix2 ...")."""
        parts: list[str] = []
        for order, pixels in self.to_orders().items():
            tokens: list[str] = []
            values = pixels.tolist()
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[j + 1] == values[j] + 1:
                    j += 1
                tokens.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
                i = j + 1
            parts.append(f"{order}/" + " ".join(tokens))
        return " ".join(parts)

    def as_json(self) -> dict[str, Any]:
        return {"ranges": self.ranges.tolist(), "sky_fraction": self.sky_fraction, "area_deg2": self.area_deg2}

    def uniq(self) -> np.ndarray:
        """The MOC's cells as sorted NUNIQ values (4 * 4**order + ipix, MOC 2.0 section 4.3.2)."""
        cells = [4 * 4**order + pixels for order, pixels in self.to_orders().items()]
        return np.sort(np.concatenate(cells)) if cells else np.zeros(0, dtype=np.int64)

    def write_fits(self, path: str | os.PathLike[str]) -> None:
        """Write the MOC as an IVOA MOC 2.0 FITS file (space MOC, NUNIQ ordering), readable
        with e.g. ``mocpy.MOC.from_fits``."""
        from astropy.io import fits

        column = fits.Column(name="UNIQ", format="K", array=self.uniq())
        hdu = fits.BinTableHDU.from_columns([column])
        order = self.max_order
        for key, value, comment in (
            ("PIXTYPE", "HEALPIX", "HEALPix magic code"), ("ORDERING", "NUNIQ", "NUNIQ coding method"),
            ("COORDSYS", "C", "ICRS reference frame"), ("MOCVERS", "2.0", "MOC version"),
            ("MOCDIM", "SPACE", "Physical dimension"), ("MOCORD_S", order, "MOC resolution (best order)"),
            ("MOCORDER", order, "MOC resolution (MOC 1.x keyword)"), ("MOCTOOL", BUILDER, "Name of the MOC generator"),
            ("MOCTYPE", "CATALOG", "Source type (IMAGE or CATALOG)"),
        ):
            hdu.header[key] = (value, comment)
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def cone_moc(ra: float, dec: float, radius_deg: float, order: int) -> Moc:
    """Every order-``order`` pixel overlapping the cone, as a MOC (inclusive like :func:`cone_pixels`).

    See :func:`_cone_cells`.
    """
    return _cone_cells(ra, dec, radius_deg, order, inside_only=False)


def _cone_cells(ra: float, dec: float, radius_deg: float, order: int, *, inside_only: bool) -> Moc:
    """The order-``order`` pixels overlapping the cone (``inside_only``: lying entirely inside
    it, :func:`_inside_mask`), as a MOC.

    Small cones (up to ~CONE_MOC_DIRECT_PIXELS pixels) use :func:`cone_pixels` directly.
    Larger ones are hierarchical: from a coarse order (pixels ~ radius/8), pixels lying
    entirely inside the cone are kept whole and only pixels crossing its edge are split,
    so the cost grows with the cone's perimeter in pixels, not with its area. This matters
    beyond speed: astropy-healpix's cone search enumerates every pixel in ONE C call that
    holds the GIL (~200 ms for a 600" cone at order 14, ~0.5 s for a mirror's coverage
    pixels), which would stall every other thread's event loop even when run in a worker
    thread. A coarse pixel that is inside (all sampled boundary points within ``radius -
    2% res``) has only inside descendants: theirs lie within it and the coarse margin (2%
    of the coarse res) exceeds the sampling error (< res/500) plus the fine margin, so the
    result equals the direct test. A child of a crossing pixel is kept when its nearest
    sampled boundary point is within ``radius + 2% res`` or it contains the centre.
    Sampling overestimates the nearest distance by at most (res/16)^2 / (2 * radius) <
    res/500 for the pixel sizes used (res <= radius / 8), far inside the margin, so no
    overlapping pixel is lost.
    """
    order = int(order)
    if math.pi * (radius_deg / pixel_resolution_deg(order) + 1.5) ** 2 <= CONE_MOC_DIRECT_PIXELS:
        pixels = cone_pixels(ra, dec, radius_deg, order)  # few pixels: direct is faster
        if inside_only and pixels.size:
            pixels = pixels[_inside_mask(ra, dec, radius_deg, order, pixels)]
        return Moc.from_pixels(order, pixels)
    start = min(order, order_for_resolution(radius_deg / 8.0, hi=order))
    pixels = cone_pixels(ra, dec, radius_deg, start)
    center29 = int(healpix29([ra], [dec])[0])
    ranges: list[np.ndarray] = []
    level = start
    while pixels.size:
        if level == order:
            if inside_only:
                pixels = pixels[_inside_mask(ra, dec, radius_deg, level, pixels)]
            ranges.append(pixel_ranges(level, pixels))
            break
        inside = _inside_mask(ra, dec, radius_deg, level, pixels)
        ranges.append(pixel_ranges(level, pixels[inside]))
        level += 1
        children = (pixels[~inside][:, None] * 4 + np.arange(4, dtype=np.int64)).ravel()
        if not children.size:
            break
        lon, lat = _boundary_lonlat(level, children)
        near = angular_sep_deg(ra, dec, lon.ravel(), lat.ravel()).reshape(lon.shape).min(axis=1)
        keep = (near <= radius_deg + PIXEL_MARGIN * pixel_resolution_deg(level)) | (
            children == center29 >> (2 * (SPATIAL_INDEX_ORDER - level)))
        pixels = children[keep]
    return Moc(np.concatenate(ranges) if ranges else None)


# ---------------------------------------------------------------------------
# Row <-> Arrow conversion
# ---------------------------------------------------------------------------


def _column_array(values: list[Any]) -> tuple[pa.Array, bool]:
    """Arrow array for one raw column; (array, json_encoded).

    Columns whose non-null values share one plain type (bool, int, float, str) keep it;
    anything else (mixed types, lists, dicts, ints beyond int64) is stored as JSON text so
    that it round-trips exactly.
    """
    kinds = {type(v) for v in values if v is not None}
    if not kinds:
        return pa.array(values, type=pa.string()), False
    if len(kinds) == 1:
        kind = next(iter(kinds))
        target = {bool: pa.bool_(), int: pa.int64(), float: pa.float64(), str: pa.string()}.get(kind)
        if target is not None:
            try:
                return pa.array(values, type=target), False
            except (pa.ArrowInvalid, OverflowError, TypeError):
                pass
    encoded = [None if v is None else json.dumps(v, sort_keys=True, allow_nan=True) for v in values]
    return pa.array(encoded, type=pa.string()), True


def _field_metadata(meta: ColumnMeta | None) -> dict[bytes, bytes] | None:
    if meta is None:
        return None
    out: dict[bytes, bytes] = {}
    for key in ("unit", "ucd", "datatype", "description"):
        value = getattr(meta, key)
        if value not in (None, ""):
            out[key.encode()] = str(value).encode("utf-8")
    return out or None


def _is_query_relative(name: str, provider: str) -> bool:
    """Columns computed by the archive relative to the query centre (never stored)."""
    lowered = name.lower()
    return lowered in COMPUTED_COLUMNS or (provider == "irsa_gator" and lowered == "angle")


# ---------------------------------------------------------------------------
# Stored rows, Arrow batches and catalog state
# ---------------------------------------------------------------------------


@dataclass
class StoredRow:
    """One mirrored archive row: raw columns (ordered) plus canonical fields."""

    source_id: str
    ra: float
    dec: float
    data: dict[str, Any]
    epoch: float | None = None
    pmra: float | None = None
    pmdec: float | None = None
    pos_err_arcsec: float | None = None
    provider: str | None = None
    retrieved_at: str | None = None
    columns: list[ColumnMeta] | None = None

    @classmethod
    def from_source(cls, source: CatalogSource, *, provider: str, retrieved_at: str,
                    columns: list[ColumnMeta] | None) -> StoredRow:
        data = {k: v for k, v in source.data.items() if not _is_query_relative(k, provider)}
        return cls(source.source_id, float(source.ra), float(source.dec), data, source.epoch,
                   source.proper_motion_ra_masyr, source.proper_motion_dec_masyr, source.positional_error_arcsec,
                   provider, retrieved_at, columns)


_SYSTEM_TYPES: dict[str, pa.DataType] = {
    SPATIAL_INDEX_COLUMN: pa.int64(), SC_ID: pa.string(), SC_RA: pa.float64(), SC_DEC: pa.float64(),
    SC_EPOCH: pa.float64(), SC_PMRA: pa.float64(), SC_PMDEC: pa.float64(), SC_POSERR: pa.float64(),
    SC_PROVIDER: pa.string(), SC_RETRIEVED: pa.string(), SC_COLSET: pa.int32(),
}


@dataclass
class _Batch:
    """Rows as one Arrow table (system columns first, then raw archive columns).

    ``colsets[i]`` lists the raw keys (in archive order) of the rows whose ``_sc_colset``
    is i and ``colset_columns[i]`` their column metadata; ``json_columns`` are raw columns
    stored JSON-encoded as strings.
    """

    table: pa.Table
    colsets: list[list[str]]
    colset_columns: list[list[dict[str, Any]]]
    json_columns: set[str]

    @property
    def num_rows(self) -> int:
        return self.table.num_rows

    @property
    def raw_names(self) -> list[str]:
        return [n for n in self.table.column_names if n not in _SYSTEM_TYPES]

    @classmethod
    def empty(cls) -> _Batch:
        return cls(pa.table({n: pa.array([], type=t) for n, t in _SYSTEM_TYPES.items()}), [], [], set())

    def filter(self, mask: Any) -> _Batch:
        return _Batch(self.table.filter(pa.array(np.asarray(mask, dtype=bool))), self.colsets, self.colset_columns,
                      self.json_columns)


def _batch_from_rows(rows: Sequence[StoredRow], catalog_name: str = "") -> _Batch:
    """Arrow batch of StoredRows (rows keep their order; ``_healpix_29`` computed)."""
    colset_index: dict[tuple[str, ...], int] = {}
    colsets: list[list[str]] = []
    colset_columns: list[list[dict[str, Any]]] = []
    row_colset: list[int] = []
    for row in rows:
        keys = tuple(row.data.keys())
        index = colset_index.get(keys)
        if index is None:
            index = colset_index[keys] = len(colsets)
            colsets.append(list(keys))
            colset_columns.append([c.as_dict() for c in (row.columns or [])])
        elif not colset_columns[index] and row.columns:
            colset_columns[index] = [c.as_dict() for c in row.columns]
        row_colset.append(index)
    raw_names: list[str] = []
    seen: set[str] = set()
    for keys in colsets:
        for key in keys:
            if key not in seen:
                if key in _SYSTEM_TYPES:
                    raise MirrorError(f"{catalog_name}: archive column {key!r} collides with a skycache column")
                seen.add(key)
                raw_names.append(key)
    ra = np.array([r.ra for r in rows], dtype=np.float64)
    dec = np.array([r.dec for r in rows], dtype=np.float64)

    def opt_float(values: Iterable[Any]) -> pa.Array:
        return pa.array([None if v is None else float(v) for v in values], type=pa.float64())

    arrays: dict[str, pa.Array] = {
        SPATIAL_INDEX_COLUMN: pa.array(healpix29(ra, dec), type=pa.int64()),
        SC_ID: pa.array([r.source_id for r in rows], type=pa.string()),
        SC_RA: pa.array(ra, type=pa.float64()),
        SC_DEC: pa.array(dec, type=pa.float64()),
        SC_EPOCH: opt_float(r.epoch for r in rows),
        SC_PMRA: opt_float(r.pmra for r in rows),
        SC_PMDEC: opt_float(r.pmdec for r in rows),
        SC_POSERR: opt_float(r.pos_err_arcsec for r in rows),
        SC_PROVIDER: pa.array([r.provider for r in rows], type=pa.string()),
        SC_RETRIEVED: pa.array([r.retrieved_at for r in rows], type=pa.string()),
        SC_COLSET: pa.array(row_colset, type=pa.int32()),
    }
    json_columns: set[str] = set()
    for name in raw_names:
        arr, encoded = _column_array([r.data.get(name) for r in rows])
        if encoded:
            json_columns.add(name)
        arrays[name] = arr
    return _Batch(pa.table(arrays), colsets, colset_columns, json_columns)


def _json_encode_column(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """A typed raw column as the JSON text :func:`_column_array` would have produced for it."""
    return pa.array([None if v is None else json.dumps(v, sort_keys=True, allow_nan=True)
                     for v in column.to_pylist()], type=pa.string())


def _unify_batches(batches: Sequence[_Batch]) -> _Batch:
    """Concatenate batches: colsets merged (indices remapped), raw columns unified.

    A raw column keeps its Arrow type when every batch holding non-null values stores it
    with the same type; otherwise (types differ, or a batch JSON-encodes it) it becomes a
    JSON-encoded column, exactly as :func:`_column_array` would encode the combined values.
    """
    batches = [b for b in batches if b.num_rows]
    if not batches:
        return _Batch.empty()
    if len(batches) == 1:
        return batches[0]
    colset_index: dict[tuple[str, ...], int] = {}
    colsets: list[list[str]] = []
    colset_columns: list[list[dict[str, Any]]] = []
    remaps: list[np.ndarray] = []
    for batch in batches:
        remap = np.zeros(max(1, len(batch.colsets)), dtype=np.int32)
        for i, keys in enumerate(batch.colsets):
            key = tuple(keys)
            meta = list(batch.colset_columns[i]) if i < len(batch.colset_columns) else []
            index = colset_index.get(key)
            if index is None:
                index = colset_index[key] = len(colsets)
                colsets.append(list(keys))
                colset_columns.append(meta)
            elif not colset_columns[index] and meta:
                colset_columns[index] = meta
            remap[i] = index
        remaps.append(remap)
    columns: dict[str, pa.ChunkedArray] = {}
    for name, dtype in _SYSTEM_TYPES.items():
        if name == SC_COLSET:
            chunks = [pa.array(remap[b.table.column(SC_COLSET).to_numpy()], type=pa.int32())
                      for b, remap in zip(batches, remaps)]
        else:
            chunks = [c for b in batches for c in b.table.column(name).cast(dtype).chunks]
        columns[name] = pa.chunked_array(chunks, type=dtype)
    raw_names: list[str] = []
    seen: set[str] = set()
    for batch in batches:
        for name in batch.raw_names:
            if name not in seen:
                seen.add(name)
                raw_names.append(name)
    json_columns: set[str] = set()
    for name in raw_names:
        present = [b.table.column(name) if name in b.table.column_names else None for b in batches]
        filled = [(b, col) for b, col in zip(batches, present) if col is not None and col.null_count < len(col)]
        kinds = {"json" if name in b.json_columns else str(col.type) for b, col in filled}
        chunks: list[pa.Array] = []
        if len(kinds) > 1 or "json" in kinds:
            json_columns.add(name)
            dtype = pa.string()
            for b, col in zip(batches, present):
                if col is None or col.null_count == len(col):
                    chunks.append(pa.nulls(b.num_rows, dtype))
                elif name in b.json_columns:
                    chunks.extend(col.cast(dtype).chunks)
                else:
                    chunks.append(_json_encode_column(col))
        else:
            dtype = filled[0][1].type if filled else pa.string()
            for b, col in zip(batches, present):
                if col is None or col.null_count == len(col):
                    chunks.append(pa.nulls(b.num_rows, dtype))
                else:
                    chunks.extend(col.chunks)
        columns[name] = pa.chunked_array(chunks, type=dtype)
    return _Batch(pa.table(columns), colsets, colset_columns, json_columns)


@dataclass
class _Partition:
    order: int
    pixel: int
    table: pa.Table
    h29: np.ndarray
    ra: np.ndarray
    dec: np.ndarray


@dataclass
class _CatalogState:
    name: str
    path: Path
    meta: dict[str, Any]
    moc: Moc
    stamp: tuple[int, int]
    partitions: list[tuple[int, int, int]]  # (order, pixel, rows), sorted by range start
    part_ranges: np.ndarray  # (n, 2) order-29 ranges of the partitions (disjoint)
    loaded: dict[tuple[int, int], _Partition] = field(default_factory=dict)
    colsets: list[list[str]] = field(default_factory=list)
    colset_meta: list[list[ColumnMeta]] = field(default_factory=list)
    json_columns: frozenset[str] = frozenset()
    moc_max_order: int = 0
    signature: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    # Coverage by retrieval time: (retrieved_at, region last mirrored then), newest first,
    # disjoint (skycache.json 'coverage_ages'); None for stores written before it existed.
    ages: list[tuple[str, Moc]] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class CoverageReport:
    covered: bool
    covered_fraction: float
    order: int
    pixels: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConeResult:
    """Rows of a local cone search, nearest first."""

    rows: list[dict[str, Any]]  # raw archive columns (query-relative columns not included)
    source_ids: list[str]
    ra: np.ndarray
    dec: np.ndarray
    separation_arcsec: np.ndarray
    colsets: list[int]
    elapsed_ms: float
    partitions_read: int
    rows_in_cone: int = 0  # rows inside the radius before the limit was applied
    providers: list[str | None] = field(default_factory=list)  # adapter that fetched each row
    retrieved_at: list[str | None] = field(default_factory=list)  # UTC time each row was retrieved
    catalog_meta: dict[str, Any] = field(default_factory=dict)  # skycache.json of the version read
    column_meta: list[ColumnMeta] | None = None  # archive column metadata of the returned rows
    # Retrieval times of the mirrors whose coverage the cone overlaps (None: not recorded).
    coverage_retrieved_at: list[str] | None = None

    def __len__(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


_STORE_LOCKS: dict[str, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _catalog_lock(path: Path) -> threading.RLock:
    key = str(path.resolve()).lower()
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(key, threading.RLock())


# Cross-process catalog locks held by this process: key -> [file descriptor, depth]. Only
# touched by the thread holding the catalog's RLock, so no further locking is needed.
_FILE_LOCKS: dict[str, list[int]] = {}


def _lock_file(path: Path) -> Path:
    """The lock file of a catalog directory: ``<root>/.<catalog>.lock`` (never a HATS file)."""
    return path.parent / f".{path.name}.lock"


def _os_trylock(fd: int) -> bool:
    """Take the OS lock on byte 0 of an open file without blocking; False when another process holds it."""
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _os_unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _exclusive(path: Path, *, wait: float | None = None) -> Iterator[bool]:
    """Exclusive lock on one catalog of a store, across threads AND processes (re-entrant).

    The thread lock (:func:`_catalog_lock`) serializes this process; an OS file lock on
    ``<root>/.<catalog>.lock`` (``msvcrt.locking`` / ``fcntl.flock``) serializes processes
    and is released by the OS when a process dies, so a crashed writer never leaves the
    catalog locked. Yields True once held. With ``wait=0`` it yields False instead of
    waiting when another thread or process holds the lock; otherwise it waits up to
    ``wait`` seconds (default ``LOCK_TIMEOUT_S``) and then raises :class:`MirrorStoreError`
    (also raised when the lock file cannot be created, e.g. a read-only store).
    """
    wait = LOCK_TIMEOUT_S if wait is None else float(wait)
    rlock = _catalog_lock(path)
    if not (rlock.acquire(blocking=False) if wait <= 0 else rlock.acquire(timeout=wait)):
        if wait <= 0:
            yield False
            return
        raise MirrorStoreError(f"{path.name}: the local store is locked by another writer (waited {wait:g} s)")
    try:
        key = str(path.resolve()).lower()
        held = _FILE_LOCKS.get(key)
        if held is not None:  # re-entered by the thread that holds it
            held[1] += 1
            try:
                yield True
            finally:
                held[1] -= 1
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(_lock_file(path), os.O_RDWR | os.O_CREAT, 0o666)
        except OSError as exc:
            raise MirrorStoreError(f"{path.name}: cannot lock the local store ({type(exc).__name__}: {exc})") from exc
        deadline = time.monotonic() + max(0.0, wait)
        acquired = False
        try:
            while not (acquired := _os_trylock(fd)):
                if wait <= 0 or time.monotonic() >= deadline:
                    break
                time.sleep(LOCK_POLL_S)
            if not acquired:
                if wait <= 0:
                    yield False
                    return
                raise MirrorStoreError(f"{path.name}: the local store is locked by another process "
                                       f"(waited {wait:g} s)")
            _FILE_LOCKS[key] = [fd, 1]
            try:
                yield True
            finally:
                _FILE_LOCKS.pop(key, None)
                with contextlib.suppress(OSError):
                    _os_unlock(fd)
        finally:
            os.close(fd)
    finally:
        rlock.release()


_CATALOG_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}$")


def _replace_with_retry(src: Path, dst: Path) -> None:
    """os.replace, retried with backoff on PermissionError (transient Windows file handles)."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAY_S * 2**attempt)


def _dir_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:  # removed by a concurrent swap
            continue
    return total


class SkyCache:
    """A directory of HATS catalogs mirrored from archives (see module docstring).

    Pure Python API::

        cache = SkyCache("~/.astrosearch/skycache")
        report = await mirror_region("gaia_dr3", ra=187.2779, dec=2.0524, radius_deg=0.2, store=cache)
        cache.covers("gaia_dr3", 187.2779, 2.0524, 30.0)        # CoverageReport
        cache.cone_search("gaia_dr3", 187.2779, 2.0524, 30.0)   # ConeResult (raw rows)
        await LocalProvider(cache).query(registry.get("gaia_dr3"), target, 30.0)  # CatalogSources
    """

    def __init__(self, root: str | os.PathLike[str] | None = None, *, partition_rows: int | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else default_store_path()
        self.partition_rows = int(partition_rows or _env_int(ENV_PARTITION_ROWS, DEFAULT_PARTITION_ROWS))
        self._states: dict[str, _CatalogState] = {}
        self._lock = threading.RLock()
        try:
            self.repair()
        except OSError as exc:  # a read-only or unavailable store is still usable for reads
            logger.warning("skycache: could not repair %s: %s", self.root, exc)

    # -- paths ---------------------------------------------------------------

    def catalog_path(self, catalog: str) -> Path:
        if not _CATALOG_NAME.match(catalog or ""):
            raise MirrorInputError(f"invalid catalog name {catalog!r}")
        return self.root / catalog

    def catalogs(self) -> list[str]:
        """Mirrored catalogs (hidden swap directories of in-progress or failed writes excluded)."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if _CATALOG_NAME.match(p.name) and (p / "skycache.json").is_file())

    def has(self, catalog: str) -> bool:
        try:
            return (self.catalog_path(catalog) / "skycache.json").is_file()
        except MirrorInputError:
            return False

    def swap_dirs(self) -> list[dict[str, Any]]:
        """Hidden ``.<catalog>.tmp-*`` / ``.<catalog>.old-*`` directories left in the store."""
        if not self.root.is_dir():
            return []
        out = []
        for entry in sorted(self.root.iterdir()):
            match = _SWAP_DIR.match(entry.name)
            if match and entry.is_dir():
                try:
                    age = time.time() - entry.stat().st_mtime
                except OSError:
                    continue
                out.append({"path": str(entry), "catalog": match["catalog"], "kind": match["kind"],
                            "age_s": round(age, 1)})
        return out

    def repair(self, catalog: str | None = None) -> dict[str, list[str]]:
        """Restore or remove swap directories left by an interrupted or failed write.

        A backup (``.old-``) whose catalog directory is missing is moved back (the newest
        one); other backups and deleted trees (``.del-``) are removed, as are temporary
        trees (``.tmp-``) older than ``STALE_SWAP_SECONDS``. Each catalog is repaired only
        while holding its exclusive cross-process lock (:func:`_exclusive`), taken without
        waiting: a catalog whose write (swap) is in progress in another thread or process
        is left alone and listed under ``busy`` -- a half-done swap is never "repaired".
        """
        done: dict[str, list[str]] = {"restored": [], "removed": [], "busy": []}
        leftovers = [d for d in self.swap_dirs() if catalog is None or d["catalog"] == catalog]
        # Newest backups first, so the most recent catalog version is the one restored.
        leftovers.sort(key=lambda d: (d["catalog"], d["kind"], d["age_s"]))
        for item in leftovers:
            entry = Path(item["path"])
            target = self.root / item["catalog"]
            try:
                with _exclusive(target, wait=0) as acquired:
                    if not acquired:
                        done["busy"].append(str(entry))
                        continue
                    if not entry.exists():
                        continue
                    if item["kind"] == "old" and not (target / "skycache.json").is_file() \
                            and (entry / "skycache.json").is_file():
                        if target.exists():  # a catalog directory without metadata: debris
                            self._discard_tree(target, item["catalog"])
                        _replace_with_retry(entry, target)
                        done["restored"].append(str(entry))
                        logger.warning("skycache: restored %s from the backup %s", target, entry.name)
                        with self._lock:
                            self._states.pop(item["catalog"], None)
                    elif item["kind"] in ("old", "del") or item["age_s"] > STALE_SWAP_SECONDS:
                        shutil.rmtree(entry, ignore_errors=True)
                        if not entry.exists():
                            done["removed"].append(str(entry))
            except MirrorStoreError as exc:  # the lock file cannot be created (read-only store)
                logger.warning("skycache: %s not repaired: %s", entry, exc)
        return done

    def _discard_tree(self, tree: Path, catalog: str) -> bool:
        """Remove a catalog tree (or swap directory) so that it can never be restored.

        The tree is first renamed to ``.<catalog>.del-*`` (which :meth:`repair` only ever
        removes), then deleted. If a file of it cannot be deleted (a handle held by another
        process on Windows), its ``skycache.json`` is removed at least, so it is neither a
        catalog nor a restorable backup any more. Returns True when nothing restorable is left.
        """
        grave = self.root / f".{catalog}.del-{uuid.uuid4().hex[:8]}"
        try:
            _replace_with_retry(tree, grave)
            tree = grave
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.warning("skycache: could not rename %s for deletion (%s); deleting in place", tree, exc)
        shutil.rmtree(tree, ignore_errors=True)
        if not tree.exists():
            return True
        with contextlib.suppress(OSError):
            (tree / "skycache.json").unlink()
        logger.warning("skycache: %s could not be removed completely (files in use); it will be removed later", tree)
        return not (tree / "skycache.json").exists()

    # -- loading -------------------------------------------------------------

    @staticmethod
    def _stamp(path: Path) -> tuple[int, int] | None:
        try:
            st = (path / "skycache.json").stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _state(self, catalog: str) -> _CatalogState | None:
        path = self.catalog_path(catalog)
        stamp = self._stamp(path)
        with self._lock:
            state = self._states.get(catalog)
            if stamp is None:
                self._states.pop(catalog, None)
                return None
            if state is not None and state.stamp == stamp:
                return state
            meta = json.loads((path / "skycache.json").read_text(encoding="utf-8"))
            if meta.get("format") != FORMAT_NAME:
                raise AstroSearchError(f"{path} is not a {FORMAT_NAME} catalog")
            parts = sorted(((int(k), int(p), int(n)) for k, p, n in meta.get("partitions", [])),
                           key=lambda t: t[1] << (2 * (SPATIAL_INDEX_ORDER - t[0])))
            ranges = (np.concatenate([pixel_ranges(k, [p]) for k, p, _ in parts]) if parts
                      else np.zeros((0, 2), dtype=np.int64))
            colsets = [list(c) for c in meta.get("colsets", [])]
            colset_meta = [[ColumnMeta(**c) for c in cols] for cols in meta.get("colset_columns", [])]
            signature = _recorded_signature(meta)
            ages = meta.get("coverage_ages")
            state = _CatalogState(catalog, path, meta, Moc.from_json(meta.get("coverage")), stamp, parts, ranges,
                                  colsets=colsets, colset_meta=colset_meta,
                                  json_columns=frozenset(meta.get("json_columns", [])),
                                  signature=signature or {},
                                  fingerprint=definition_fingerprint(signature) if signature else "",
                                  ages=None if ages is None else [(str(a["retrieved_at"]), Moc.from_json(a))
                                                                  for a in ages])
            cov_order = (meta.get("coverage") or {}).get("max_order")
            state.moc_max_order = int(cov_order) if cov_order is not None else state.moc.max_order
            self._states[catalog] = state
            return state

    def evict(self, catalog: str) -> None:
        """Drop the cached state (metadata and loaded partitions) of a catalog."""
        with self._lock:
            self._states.pop(catalog, None)

    def _forget(self, state: _CatalogState) -> None:
        with self._lock:
            if self._states.get(state.name) is state:
                self._states.pop(state.name, None)

    @staticmethod
    def _partition_file(base: Path, order: int, pixel: int) -> Path:
        return base / "dataset" / f"Norder={order}" / f"Dir={(pixel // DIR_DIVISOR) * DIR_DIVISOR}" / f"Npix={pixel}.parquet"

    @staticmethod
    def _read_partition(state: _CatalogState, order: int, pixel: int) -> pa.Table:
        """Read one leaf file, checking that it belongs to the catalog version of ``state``."""
        try:
            table = pq.read_table(SkyCache._partition_file(state.path, order, pixel))
        except FileNotFoundError as exc:
            raise _StaleStateError(str(exc)) from exc
        info = (table.schema.metadata or {}).get(b"astrosearch_skycache")
        version = json.loads(info).get("version") if info else None
        if version is not None and version != state.meta.get("version"):
            raise _StaleStateError(f"{state.name}: partition {order}/{pixel} is from catalog version {version}")
        return table

    def _partition(self, state: _CatalogState, order: int, pixel: int) -> _Partition:
        key = (order, pixel)
        part = state.loaded.get(key)
        if part is not None:
            return part
        with state.lock:  # one reader loads a partition; the others wait for it
            part = state.loaded.get(key)
            if part is None:
                table = self._read_partition(state, order, pixel)
                part = _Partition(order, pixel, table,
                                  table.column(SPATIAL_INDEX_COLUMN).to_numpy(),
                                  table.column(SC_RA).to_numpy(), table.column(SC_DEC).to_numpy())
                state.loaded[key] = part
        return part

    def metadata(self, catalog: str) -> dict[str, Any] | None:
        state = self._state(catalog)
        return None if state is None else state.meta

    def coverage(self, catalog: str) -> Moc:
        state = self._state(catalog)
        return Moc() if state is None else state.moc

    # -- queries -------------------------------------------------------------

    @staticmethod
    def _query_order(radius_deg: float, moc_max_order: int) -> int:
        """HEALPix order used to test a cone against the coverage.

        Fine enough that pixels are <= radius/4 and not coarser than the coverage's own
        finest pixels (a coarse pixel straddling the coverage edge would fail the test),
        but capped so the cone spans at most ~1.3e4 pixels (pixel >= radius/64).
        """
        fine = order_for_resolution(radius_deg / 4.0, hi=MAX_QUERY_ORDER)
        cap = order_for_resolution(radius_deg / 64.0, hi=MAX_QUERY_ORDER)
        return max(0, min(max(fine, min(moc_max_order, MAX_QUERY_ORDER)), cap))

    def _cone_coverage(self, state: _CatalogState, ra: float, dec: float,
                       radius_deg: float) -> tuple[CoverageReport, np.ndarray]:
        """(report, order-29 ranges of the cone's pixels). ``covered_fraction`` is by area."""
        order = self._query_order(radius_deg, state.moc_max_order)
        cone = cone_moc(ra, dec, radius_deg, order)
        total = cone.cells29
        inside = int(state.moc.covered_cells(cone.ranges).sum()) if total else 0
        fraction = inside / total if total else 0.0
        pixels = total // 4 ** (SPATIAL_INDEX_ORDER - order)
        return CoverageReport(bool(total) and inside == total, fraction, order, int(pixels)), cone.ranges

    def covers(self, catalog: str, ra: float, dec: float, radius_arcsec: float) -> CoverageReport:
        """Whether every pixel overlapping the cone lies inside the recorded coverage.

        The cone's pixels (:func:`cone_moc`, inclusive) must all be inside the MOC, so a True
        answer is never optimistic.
        """
        state = self._state(catalog)
        if state is None or state.moc.empty:
            return CoverageReport(False, 0.0, 0, 0)
        return self._cone_coverage(state, ra, dec, float(radius_arcsec) / 3600.0)[0]

    def require_coverage(self, catalog: str, ra: float, dec: float, radius_arcsec: float) -> CoverageReport:
        """Raise :class:`CoverageError` unless the cone is fully covered."""
        state = self._state(catalog)
        if state is None:
            raise self._not_mirrored(catalog, ra, dec, radius_arcsec)
        return self._require(state, ra, dec, radius_arcsec)[0]

    def _not_mirrored(self, catalog: str, ra: float | None = None, dec: float | None = None,
                      radius_arcsec: float | None = None) -> CoverageError:
        return CoverageError(f"{catalog}: not mirrored in the local sky cache ({self.root})", catalog=catalog,
                             ra=ra, dec=dec, radius_arcsec=radius_arcsec, reason="not_mirrored")

    def _require(self, state: _CatalogState, ra: float, dec: float,
                 radius_arcsec: float) -> tuple[CoverageReport, np.ndarray]:
        report, pixels = self._cone_coverage(state, ra, dec, float(radius_arcsec) / 3600.0)
        if not report.covered:
            raise CoverageError(
                f"{state.name}: cone RA={ra:.6f} Dec={dec:+.6f} r={radius_arcsec:g}\" is only "
                f"{100.0 * report.covered_fraction:.1f}% inside the mirrored coverage",
                catalog=state.name, ra=ra, dec=dec, radius_arcsec=radius_arcsec,
                covered_fraction=report.covered_fraction, order=report.order,
                reason="partially_covered" if report.covered_fraction > 0 else "not_covered")
        return report, pixels

    @staticmethod
    def _check_definition(state: _CatalogState, definition: CatalogDefinition) -> None:
        """CoverageError(reason='definition_changed') unless ``definition`` matches the mirrored one."""
        if state.fingerprint and definition_fingerprint(definition) == state.fingerprint:
            return
        changes = _signature_changes(state.signature, definition_signature(definition)) if state.signature else []
        detail = (", ".join(changes) if changes else "fingerprints differ" if state.signature
                  else "no definition recorded in the store")
        raise CoverageError(
            f"{state.name}: the catalog definition differs from the one the mirror was made with ({detail}); "
            "the local copy is not used until the region is mirrored again",
            catalog=state.name, reason="definition_changed")

    def check_definition(self, definition: CatalogDefinition) -> None:
        """Raise :class:`CoverageError` unless ``definition`` matches the mirrored catalog's."""
        state = self._state(definition.name)
        if state is None:
            raise self._not_mirrored(definition.name)
        self._check_definition(state, definition)

    def cone_search(self, catalog: str, ra: float, dec: float, radius_arcsec: float, *,
                    limit: int | None = None, require_coverage: bool = True,
                    definition: CatalogDefinition | None = None) -> ConeResult:
        """Rows within ``radius_arcsec`` of (ra, dec), nearest first (ties by source id).

        Candidate rows come from the partitions and ``_healpix_29`` ranges of the pixels
        overlapping the cone (vectorized ``searchsorted`` on each partition's sorted
        spatial index); the exact cut is a numpy haversine separation <= radius. With a
        ``limit`` the nearest rows are picked with numpy (``np.partition``) and only those
        are converted to Python, so the cost does not grow with the cone's density. With
        ``require_coverage`` a :class:`CoverageError` is raised unless the cone is fully
        inside the mirrored coverage; with ``definition`` also unless the catalog was
        mirrored with an equivalent definition (``reason='definition_changed'``).
        """
        started = time.perf_counter()
        radius_deg = float(radius_arcsec) / 3600.0
        if not (math.isfinite(radius_deg) and radius_deg > 0):
            raise InvalidCoordinateError("radius_arcsec must be finite and > 0")
        for attempt in range(2):
            state = self._state(catalog)
            if state is None:
                raise self._not_mirrored(catalog, ra, dec, radius_arcsec)
            try:
                return self._cone(state, ra, dec, radius_arcsec, limit=limit, require_coverage=require_coverage,
                                  definition=definition, started=started)
            except _StaleStateError as exc:
                # The catalog tree was replaced after the state was read: reload once.
                self._forget(state)
                if attempt:
                    raise CoverageError(f"{catalog}: the local store changed during the query ({exc})",
                                        catalog=catalog, ra=ra, dec=dec, radius_arcsec=radius_arcsec,
                                        reason="store_changed") from exc
        raise AssertionError("unreachable")

    def _cone(self, state: _CatalogState, ra: float, dec: float, radius_arcsec: float, *, limit: int | None,
              require_coverage: bool, definition: CatalogDefinition | None, started: float) -> ConeResult:
        radius_deg = float(radius_arcsec) / 3600.0
        if definition is not None:
            self._check_definition(state, definition)
        if require_coverage:
            _report, cone = self._require(state, ra, dec, radius_arcsec)
        else:
            _report, cone = self._cone_coverage(state, ra, dec, radius_deg)
        read = 0
        parts: list[_Partition] = []
        idxs: list[np.ndarray] = []
        seps: list[np.ndarray] = []
        if state.part_ranges.shape[0] and cone.shape[0]:
            # Partitions overlapping any cone range (both sets are sorted and disjoint).
            first = np.searchsorted(state.part_ranges[:, 1], cone[:, 0], side="right")
            last = np.searchsorted(state.part_ranges[:, 0], cone[:, 1], side="left")
            wanted = sorted({i for a, b in zip(first.tolist(), last.tolist()) for i in range(a, b)})
            for index in wanted:
                k, p, _n = state.partitions[index]
                part = self._partition(state, k, p)
                read += 1
                lo = np.searchsorted(part.h29, cone[:, 0], side="left")
                hi = np.searchsorted(part.h29, cone[:, 1], side="left")
                spans = [np.arange(a, b) for a, b in zip(lo.tolist(), hi.tolist()) if b > a]
                if not spans:
                    continue
                idx = np.concatenate(spans)
                sep = angular_sep_deg(ra, dec, part.ra[idx], part.dec[idx])
                keep = sep <= radius_deg
                if keep.any():
                    parts.append(part)
                    idxs.append(idx[keep])
                    seps.append(sep[keep])
        total = int(sum(i.size for i in idxs))
        if total:
            owner = np.concatenate([np.full(i.size, j, dtype=np.int64) for j, i in enumerate(idxs)])
            row_idx = np.concatenate(idxs)
            sep_all = np.concatenate(seps)
        else:
            owner = row_idx = np.zeros(0, dtype=np.int64)
            sep_all = np.zeros(0)
        keep_n = total if limit is None else max(0, min(int(limit), total))
        if keep_n == 0:
            candidates = np.zeros(0, dtype=np.int64)
        elif keep_n < total:
            # Every row as near as the keep_n-th nearest (ties included), then exact ordering.
            kth = np.partition(sep_all, keep_n - 1)[keep_n - 1]
            candidates = np.flatnonzero(sep_all <= kth)
        else:
            candidates = np.arange(total)

        def take(positions: np.ndarray, column: str | None) -> list[Any]:
            """Values (a column, or whole records) of candidate rows, in ``positions`` order."""
            out: list[Any] = [None] * positions.size
            owners = owner[positions]
            for j in np.unique(owners).tolist():
                where = np.flatnonzero(owners == j)
                rows = pa.array(row_idx[positions[where]])
                table = parts[j].table
                values = (table.column(column).take(rows) if column else table.take(rows)).to_pylist()
                for w, value in zip(where.tolist(), values):
                    out[w] = value
            return out

        cand_ids = [str(v) for v in take(candidates, SC_ID)]
        ranked = sorted(range(candidates.size), key=lambda i: (float(sep_all[candidates[i]]), cand_ids[i]))[:keep_n]
        chosen = candidates[np.asarray(ranked, dtype=np.int64)] if ranked else np.zeros(0, dtype=np.int64)
        records = take(chosen, None)
        colsets = [int(rec[SC_COLSET]) for rec in records]
        ra_arr = np.array([float(rec[SC_RA]) for rec in records], dtype=np.float64)
        dec_arr = np.array([float(rec[SC_DEC]) for rec in records], dtype=np.float64)
        return ConeResult(
            rows=[self._raw_row(state, rec, colset) for rec, colset in zip(records, colsets)],
            source_ids=[cand_ids[i] for i in ranked], ra=ra_arr, dec=dec_arr,
            separation_arcsec=sep_all[chosen] * 3600.0, colsets=colsets,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3), partitions_read=read,
            rows_in_cone=total, providers=[rec.get(SC_PROVIDER) for rec in records],
            retrieved_at=[rec.get(SC_RETRIEVED) for rec in records], catalog_meta=state.meta,
            column_meta=self._column_meta_of(state, set(colsets) if colsets else None),
            coverage_retrieved_at=None if state.ages is None else [
                when for when, moc in state.ages if cone.shape[0] and moc.overlaps_ranges(cone)],
        )

    @staticmethod
    def _raw_row(state: _CatalogState, record: Mapping[str, Any], colset: int) -> dict[str, Any]:
        keys = state.colsets[colset] if 0 <= colset < len(state.colsets) else []
        row: dict[str, Any] = {}
        for key in keys:
            value = record.get(key)
            if key in state.json_columns and value is not None:
                value = json.loads(value)
            row[key] = value
        return row

    @staticmethod
    def _column_meta_of(state: _CatalogState, colsets: Iterable[int] | None) -> list[ColumnMeta] | None:
        if not state.colset_meta:
            return None
        chosen = sorted(set(colsets)) if colsets is not None else range(len(state.colset_meta))
        merged: dict[str, ColumnMeta] = {}
        for index in chosen:
            if 0 <= index < len(state.colset_meta):
                for col in state.colset_meta[index]:
                    merged.setdefault(col.name, col)
        return [ColumnMeta(**asdict(c)) for c in merged.values()] or None

    def column_meta(self, catalog: str, colsets: Iterable[int] | None = None) -> list[ColumnMeta] | None:
        """Archive column metadata (units/UCDs) of the given column sets, merged by name."""
        state = self._state(catalog)
        return None if state is None else self._column_meta_of(state, colsets)

    def load_rows(self, catalog: str) -> list[StoredRow]:
        """Every stored row as Python objects (inspection and tests; mirrors merge Arrow batches)."""
        state = self._state(catalog)
        if state is None:
            return []
        out: list[StoredRow] = []
        for k, p, _n in state.partitions:
            part = self._partition(state, k, p)
            for rec in part.table.to_pylist():
                colset = int(rec[SC_COLSET])
                out.append(StoredRow(
                    str(rec[SC_ID]), float(rec[SC_RA]), float(rec[SC_DEC]), self._raw_row(state, rec, colset),
                    rec.get(SC_EPOCH), rec.get(SC_PMRA), rec.get(SC_PMDEC), rec.get(SC_POSERR),
                    rec.get(SC_PROVIDER), rec.get(SC_RETRIEVED),
                    state.colset_meta[colset] if 0 <= colset < len(state.colset_meta) else None,
                ))
        return out

    def load_batch(self, catalog: str) -> _Batch | None:
        """Every stored row as one Arrow batch (read straight from the files, not cached)."""
        for attempt in range(2):
            state = self._state(catalog)
            if state is None:
                return None
            try:
                tables = [self._read_partition(state, k, p).replace_schema_metadata(None)
                          for k, p, _n in state.partitions]
            except _StaleStateError:
                self._forget(state)
                if attempt:
                    raise
                continue
            if not tables:
                return _Batch.empty()
            table = pa.concat_tables(tables)
            fields = [pa.field(f.name, f.type) for f in table.schema]
            return _Batch(table.cast(pa.schema(fields)), [list(c) for c in state.colsets],
                          [list(c) for c in state.meta.get("colset_columns", [])], set(state.json_columns))
        return None

    # -- writing -------------------------------------------------------------

    def write(self, catalog: CatalogDefinition, rows: Sequence[StoredRow], coverage: Moc, *,
              mirrors: list[dict[str, Any]] | None = None,
              coverage_ages: list[tuple[str, Moc]] | None = None) -> dict[str, Any]:
        """(Re)write the whole HATS catalog atomically; returns the new skycache.json."""
        return self.write_batch(catalog, _batch_from_rows(rows, catalog.name), coverage, mirrors=mirrors,
                                coverage_ages=coverage_ages)

    def write_batch(self, catalog: CatalogDefinition, batch: _Batch, coverage: Moc, *,
                    mirrors: list[dict[str, Any]] | None = None,
                    coverage_ages: list[tuple[str, Moc]] | None = None) -> dict[str, Any]:
        """(Re)write the whole HATS catalog from an Arrow batch, atomically.

        The new tree is written to a hidden temporary directory and swapped in with two
        renames (catalog -> backup, temporary -> catalog), under the catalog's exclusive
        cross-process lock. A failed swap restores the backup; file-system errors are
        raised as :class:`MirrorStoreError` and the previous catalog stays readable.
        ``coverage_ages`` (retrieved_at, region) layers, newest first, record when each
        part of ``coverage`` was last mirrored (see :class:`LocalProvider` data age).
        """
        path = self.catalog_path(catalog.name)
        with _exclusive(path):
            tmp = self.root / f".{catalog.name}.tmp-{uuid.uuid4().hex[:8]}"
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                self.repair(catalog.name)
                meta = self._write_tree(tmp, catalog, batch, coverage, mirrors or [], coverage_ages)
                self._swap(tmp, path, catalog.name, meta["version"])
            except OSError as exc:
                raise MirrorStoreError(f"{catalog.name}: could not write the local store at {path}: "
                                       f"{type(exc).__name__}: {exc}") from exc
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
                with self._lock:
                    self._states.pop(catalog.name, None)
            return meta

    @staticmethod
    def _tree_version(path: Path) -> str | None:
        """The catalog version recorded in a tree's skycache.json (None if unreadable)."""
        try:
            return json.loads((path / "skycache.json").read_text(encoding="utf-8")).get("version")
        except (OSError, ValueError, AttributeError):
            return None

    def _swap(self, tmp: Path, path: Path, name: str, version: str) -> None:
        backup = None
        if path.exists():
            backup = self.root / f".{name}.old-{uuid.uuid4().hex[:8]}"
            _replace_with_retry(path, backup)
        try:
            _replace_with_retry(tmp, path)
        except OSError:
            if backup is not None:
                if path.exists():
                    # Never delete a tree we did not put there: only our own new tree (same
                    # version) could legitimately be at ``path`` now. Anything else is left
                    # alone, and so is our backup, for repair() to sort out.
                    if self._tree_version(path) == version:
                        shutil.rmtree(backup, ignore_errors=True)
                        return
                    logger.error("skycache: %s reappeared during a failed swap; backup %s kept", path, backup)
                    raise
                try:
                    _replace_with_retry(backup, path)
                except OSError as restore_error:  # left for repair() (runs before every write)
                    logger.error("skycache: could not restore %s from %s: %s", path, backup, restore_error)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            if backup.exists():  # a handle on a file of the old tree (Windows): never restorable
                self._discard_tree(backup, name)

    def _write_tree(self, base: Path, catalog: CatalogDefinition, batch: _Batch, coverage: Moc,
                    mirrors: list[dict[str, Any]],
                    coverage_ages: list[tuple[str, Moc]] | None = None) -> dict[str, Any]:
        dataset = base / "dataset"
        dataset.mkdir(parents=True)
        version = uuid.uuid4().hex
        col_meta: dict[str, ColumnMeta] = {}
        for cols in batch.colset_columns:
            for c in cols:
                col_meta.setdefault(c["name"], ColumnMeta(**c))

        n = batch.num_rows
        h29 = batch.table.column(SPATIAL_INDEX_COLUMN).to_numpy()
        order = np.argsort(h29, kind="stable")
        h29 = h29[order]
        sorted_table = batch.table.take(pa.array(order)).combine_chunks() if n else batch.table
        fields = [pa.field(name, dtype, metadata=_field_metadata(ColumnMeta(name, *_SYSTEM_META[name][:2],
                                                                            description=_SYSTEM_META[name][2])))
                  for name, dtype in _SYSTEM_TYPES.items()]
        arrays = [sorted_table.column(name) for name in _SYSTEM_TYPES]
        json_columns = sorted(batch.json_columns)
        for name in batch.raw_names:
            column = sorted_table.column(name)
            fmeta = _field_metadata(col_meta.get(name)) or {}
            if name in batch.json_columns:
                fmeta[b"skycache_encoding"] = b"json"
            fields.append(pa.field(name, column.type, metadata=fmeta or None))
            arrays.append(column)
        schema = pa.schema(fields, metadata={
            b"astrosearch_skycache": json.dumps({"catalog": catalog.name, "format_version": FORMAT_VERSION,
                                                 "archive_endpoint": catalog.endpoint,
                                                 "version": version}).encode("utf-8"),
        })
        table = pa.Table.from_arrays(arrays, schema=schema)

        # Adaptive partitions; rows of one partition are contiguous in _healpix_29 order.
        porders, ppixels = partition_orders(h29, self.partition_rows)
        partitions: list[list[int]] = []
        collector: list[pq.FileMetaData] = []
        if n:
            change = np.flatnonzero((np.diff(porders) != 0) | (np.diff(ppixels) != 0)) + 1
            bounds = np.concatenate([[0], change, [n]])
            for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
                k, p = int(porders[a]), int(ppixels[a])
                rel = f"Norder={k}/Dir={(p // DIR_DIVISOR) * DIR_DIVISOR}/Npix={p}.parquet"
                target = dataset / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table.slice(a, b - a), target, metadata_collector=collector)
                collector[-1].set_file_path(rel)
                partitions.append([k, p, b - a])
        pq.write_metadata(schema, dataset / "_common_metadata")
        pq.write_metadata(schema, dataset / "_metadata", metadata_collector=collector)
        (base / "partition_info.csv").write_text(
            "Norder,Npix\n" + "".join(f"{k},{p}\n" for k, p, _ in sorted(partitions)), encoding="utf-8")

        now = datetime.now(UTC)
        size_kb = math.ceil(sum(f.stat().st_size for f in dataset.rglob("*") if f.is_file()) / 1024.0)
        properties = {
            "obs_collection": catalog.name,
            "dataproduct_type": "object",
            "hats_nrows": str(n),
            "hats_col_ra": SC_RA,
            "hats_col_dec": SC_DEC,
            "hats_col_healpix": SPATIAL_INDEX_COLUMN,
            "hats_col_healpix_order": str(SPATIAL_INDEX_ORDER),
            "hats_npix_suffix": ".parquet",
            "hats_max_rows": str(self.partition_rows),
            "hats_order": str(max((k for k, _p, _n in partitions), default=0)),
            "hats_builder": BUILDER,
            "hats_creation_date": now.strftime("%Y-%m-%dT%H:%MUTC"),
            "hats_estsize": str(size_kb),
            "hats_version": HATS_VERSION,
            "moc_sky_fraction": f"{coverage.sky_fraction:.10g}",
            "obs_title": f"Local mirror of {catalog.description or catalog.name}",
            "obs_regime": catalog.wavelength,
            "bib_reference": catalog.citation or "",
            "publisher_id": "astrosearch-skycache",
        }
        text = "#HATS catalog\n" + "".join(f"{k}={_escape_property(v)}\n" for k, v in properties.items() if v != "")
        (base / "hats.properties").write_text(text, encoding="utf-8")
        (base / "properties").write_text(text, encoding="utf-8")
        # The region where the leaf files are complete, for HATS/lsdb readers (they return the
        # stored rows, which exist only inside it: see the module docstring).
        coverage.write_fits(base / COVERAGE_MOC_FILE)

        retrieved = table.column(SC_RETRIEVED)
        bounds_retrieved = pc.min_max(retrieved).as_py() if n and retrieved.null_count < n else {"min": None, "max": None}
        signature = definition_signature(catalog)
        meta = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "catalog": catalog.name,
            "version": version,
            "updated_at": now.isoformat(),
            "provider": catalog.provider,
            "archive_endpoint": catalog.endpoint,
            "citation": catalog.citation,
            "acknowledgement": catalog.acknowledgement,
            "definition": _json_safe(catalog.as_dict()),
            "definition_signature": signature,
            "definition_fingerprint": definition_fingerprint(signature),
            "rows": n,
            "row_providers": sorted(v for v in pc.unique(table.column(SC_PROVIDER)).to_pylist() if v is not None),
            "retrieved_at": {"oldest": bounds_retrieved["min"], "newest": bounds_retrieved["max"]},
            "partitions": partitions,
            "partition_rows": self.partition_rows,
            "colsets": batch.colsets,
            "colset_columns": batch.colset_columns,
            "json_columns": json_columns,
            "coverage": {**coverage.as_json(), "moc_ascii": coverage.to_ascii(), "max_order": coverage.max_order},
            "mirrors": mirrors,
            **({} if coverage_ages is None else {"coverage_ages": [
                {"retrieved_at": when, "ranges": moc.ranges.tolist(), "area_deg2": moc.area_deg2}
                for when, moc in coverage_ages if not moc.empty]}),
            "hats_properties": properties,
        }
        (base / "skycache.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return meta

    def delete(self, catalog: str) -> bool:
        """Delete a mirrored catalog and every swap directory of it (under the catalog's lock).

        Leftover backups (``.old-``) are removed too, otherwise :meth:`repair` would restore
        one on the next open and the deleted catalog would come back in an older version.
        Returns whether the catalog existed; raises :class:`MirrorStoreError` when its
        directory could not be removed.
        """
        path = self.catalog_path(catalog)
        if not self.root.is_dir():
            return False
        with _exclusive(path):
            with self._lock:
                self._states.pop(catalog, None)
            existed = path.exists()
            if existed and not self._discard_tree(path, catalog):
                raise MirrorStoreError(f"{catalog}: could not delete {path} (files in use)")
            for leftover in self.swap_dirs():
                if leftover["catalog"] == catalog and leftover["kind"] != "del":
                    self._discard_tree(Path(leftover["path"]), catalog)
            with self._lock:
                self._states.pop(catalog, None)
            return existed

    def status(self, registry: Any | None = None) -> dict[str, Any]:
        """Catalogs, coverage and mirror log; with a registry, whether each definition is current.

        A catalog whose metadata cannot be read is listed with an ``error`` instead of
        failing the whole status; leftover swap directories are listed separately.
        """
        catalogs = []
        for name in self.catalogs():
            path = self.catalog_path(name)
            try:
                state = self._state(name)
            except (OSError, ValueError, TypeError, KeyError, AstroSearchError) as exc:
                catalogs.append({"catalog": name, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if state is None:  # deleted meanwhile
                continue
            meta = state.meta
            cov = meta.get("coverage") or {}
            ascii_moc = cov.get("moc_ascii") or ""
            entry = {
                "catalog": name,
                "path": str(path),
                "rows": meta.get("rows", 0),
                "partitions": len(meta.get("partitions") or []),
                "hats_order": (meta.get("hats_properties") or {}).get("hats_order"),
                "coverage_area_deg2": cov.get("area_deg2", 0.0),
                "coverage_sky_fraction": cov.get("sky_fraction", 0.0),
                "coverage_max_order": cov.get("max_order"),
                "coverage_moc_ascii": ascii_moc if len(ascii_moc) <= 4000 else ascii_moc[:4000] + " ...",
                "coverage_moc_file": str(path / COVERAGE_MOC_FILE) if (path / COVERAGE_MOC_FILE).is_file() else None,
                "mirrors": len(meta.get("mirrors") or []),
                "last_mirror": (meta.get("mirrors") or [None])[-1],
                "updated_at": meta.get("updated_at"),
                "retrieved_at": meta.get("retrieved_at"),
                "row_providers": meta.get("row_providers"),
                "disk_bytes": _dir_bytes(path),
                "provider": meta.get("provider"),
                "archive_endpoint": meta.get("archive_endpoint"),
                "citation": meta.get("citation"),
                "definition_fingerprint": state.fingerprint or None,
            }
            if registry is not None:
                try:
                    current = registry.get(name)
                except KeyError:
                    entry["definition_current"] = None
                else:
                    same = bool(state.fingerprint) and definition_fingerprint(current) == state.fingerprint
                    entry["definition_current"] = same
                    if not same:
                        entry["definition_changes"] = _signature_changes(state.signature, definition_signature(current))
            catalogs.append(entry)
        return {"root": str(self.root), "format": FORMAT_NAME, "format_version": FORMAT_VERSION,
                "hats_compatible": True, "catalogs": catalogs, "orphaned_swap_dirs": self.swap_dirs()}


def _escape_property(value: str) -> str:
    """Java .properties value escaping as jproperties/hats write it (':' and '=' escaped)."""
    return str(value).replace("\\", "\\\\").replace("\n", " ").replace(":", "\\:").replace("=", "\\=")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Local provider
# ---------------------------------------------------------------------------


def _sdss_mode(catalog: CatalogDefinition) -> str:
    """SDSSProvider's mode rule (explicit parameter, else ConeSearch endpoints use conesearch)."""
    endpoint = str(catalog.endpoint or "")
    return str(catalog.parameters.get("mode") or ("conesearch" if "ConeSearch" in endpoint else "sqlsearch"))


def _default_id(catalog: CatalogDefinition) -> str | None:
    """The id column the catalog's own adapter passes to ``_sources`` (see providers.*.query)."""
    if catalog.provider == "sdss" and _sdss_mode(catalog) == "conesearch":
        return "objid"
    return _DEFAULT_ID.get(catalog.provider)


def _archive_row_limit(catalog: CatalogDefinition, row_limit: int) -> int | None:
    """Rows the catalog's archive returns for a cone: TOP row_limit + 1 (TAP, MAST pagesize,
    SDSS SqlSearch), or the whole cone (IRSA Gator and SDSS ConeSearch have no row limit;
    their adapters sort and trim locally)."""
    if catalog.provider == "irsa_gator" or (catalog.provider == "sdss" and _sdss_mode(catalog) == "conesearch"):
        return None
    return max(1, int(row_limit)) + 1


def chord_angle_deg(sep_deg: Any) -> np.ndarray:
    """The chord of an arc, as an angle: 2 sin(theta/2) radians in degrees (< arc by theta^3/24)."""
    return np.degrees(2.0 * np.sin(np.radians(np.asarray(sep_deg, dtype=np.float64)) / 2.0))


def _chord_distance(catalog: CatalogDefinition) -> bool:
    """IRSA TAP's ADQL DISTANCE() returns the CHORD length in arcsec, not the arc.

    Measured on the recorded 2MASS PSC answer around 3C 273 (tests/fixtures/skycache): the
    archive's match_dist minus the exact (Vincenty) separation is -9.8e-13 * sep^3 arcsec at
    every separation up to 600" (e.g. 586.8184" -> -1.98e-4"), which is theta^3/24 in radians
    (1 / (24 * 206264.8^2) = 9.79e-13): the chord 2 sin(theta/2). Other TAP services (ESA
    Gaia, HEASARC, VizieR) return the arc.
    """
    return "irsa.ipac.caltech.edu" in str(catalog.endpoint or "")


def _query_relative_values(catalog: CatalogDefinition, provider: str, ra0: float, dec0: float,
                           ra: np.ndarray, dec: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    """Columns the archive computes relative to the query centre, as its adapter returns them.

    TAP: ``match_dist`` = ADQL DISTANCE() in degrees, or arcsec where the registry says the
    service returns arcsec (IRSA, whose DISTANCE() is the chord: :func:`_chord_distance`);
    Gator: ``dist`` (arcsec) and ``angle`` (degrees E of N,
    verified live against positions); MAST: ``distance`` in degrees; SDSS SqlSearch:
    ``dist_arcsec``. Values are recomputed with the haversine formula (agree with the
    archives to ~1e-9 arcsec, float rounding only).
    """
    sep = angular_sep_deg(ra0, dec0, ra, dec)
    params = catalog.parameters
    if provider == "tap":
        mode = str(params.get("distance", "deg"))
        if mode == "none":
            return []
        if mode == "arcsec" and _chord_distance(catalog):
            return [("match_dist", chord_angle_deg(sep) * 3600.0, "arcsec")]
        return [("match_dist", sep * 3600.0 if mode == "arcsec" else sep, "arcsec" if mode == "arcsec" else "deg")]
    if provider == "irsa_gator":
        return [("dist", sep * 3600.0, "arcsec"), ("angle", position_angle_deg(ra0, dec0, ra, dec), "deg")]
    if provider == "mast":
        return [("distance", sep, "deg")] if params.get("columns") else []
    if provider == "sdss":
        return [("dist_arcsec", sep * 3600.0, "arcsec")] if _sdss_mode(catalog) != "conesearch" else []
    return []


def _conversion_columns(columns: list[ColumnMeta] | None) -> list[ColumnMeta] | None:
    """Column metadata for ``_sources`` with the UCDs that cannot matter removed.

    ``models.normalize_source_record`` recomputes ``ucd_field_map(columns)`` for every row
    (~0.5 ms per row for a 20-column table). Only UCD-carrying columns are candidates and
    only ``.ucd`` is read from them there (every other lookup is by name, for units), so
    keeping the UCD of just the columns ``ucd_field_map`` selects gives the SAME mapping:
    for each canonical field the first predicate with a hit in the full list still hits
    the same (first) column in the reduced list, and an earlier predicate that had no hit
    in the full list has none in its subset. The per-row mapping is then computed over a
    handful of candidates. Names, units and order are unchanged.
    """
    if not columns:
        return columns
    selected = {id(c) for c in ucd_field_map(columns).values()}
    return [c if id(c) in selected or not c.ucd else replace(c, ucd=None) for c in columns]


def _definition_columns(catalog: CatalogDefinition) -> list[str]:
    """Result-column names the catalog asks its archive for (parameters.columns, id/ra/dec fields)."""
    params = catalog.parameters
    columns = params.get("columns") or []
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",") if c.strip()]
    names = [bare_column_name(str(c)) for c in columns]
    names += [bare_column_name(str(params[k])) for k in ("id_field", "ra_field", "dec_field") if params.get(k)]
    return names


def _column_spellings(colsets: Iterable[Sequence[str]], catalog: CatalogDefinition) -> dict[str, str]:
    """Renames giving one spelling to each column the archive returned under several (by case).

    Some archives are not deterministic about the case of a column name (the MAST PS1 DR2
    API labels the same column 'objName' or 'ObjName' from one request to the next), so
    tiles of one mirror disagree and a local answer would mix both keys, which no single
    archive answer does. Such a column gets the spelling of the catalog definition's
    ``columns`` entry (case-insensitive match), else its first spelling seen. Columns
    that always had one spelling are never renamed (an archive that consistently spells a
    column differently from the definition keeps its spelling, as its remote answers do).
    """
    variants: dict[str, list[str]] = {}
    for keys in colsets:
        for key in keys:
            group = variants.setdefault(str(key).lower(), [])
            if key not in group:
                group.append(key)
    preferred = {name.lower(): name for name in _definition_columns(catalog)}
    mapping: dict[str, str] = {}
    for low, names in variants.items():
        if len(names) > 1:
            canonical = preferred.get(low, names[0])
            mapping.update({name: canonical for name in names if name != canonical})
    return mapping


def _rename_batch_columns(batch: _Batch, mapping: Mapping[str, str]) -> _Batch:
    """A batch with raw columns renamed; columns that become one name are coalesced.

    Every row belongs to one column set, so of two spellings of a column at most one holds
    a value in a row. Same-typed columns are coalesced as they are; otherwise the result is
    JSON-encoded like :func:`_unify_batches` does for conflicting types.
    """
    if not any(name in mapping for name in batch.raw_names):
        return batch
    groups: dict[str, list[str]] = {}
    for name in batch.table.column_names:
        groups.setdefault(mapping.get(name, name) if name not in _SYSTEM_TYPES else name, []).append(name)
    arrays: list[pa.ChunkedArray | pa.Array] = []
    json_columns: set[str] = set()
    for target, sources in groups.items():
        columns = [batch.table.column(s) for s in sources]
        encoded = any(s in batch.json_columns for s in sources)
        if len(sources) == 1:
            arrays.append(columns[0])
        elif not encoded and len({str(c.type) for c in columns if c.null_count < len(c)}) <= 1:
            dtype = next((c.type for c in columns if c.null_count < len(c)), columns[0].type)
            arrays.append(pc.coalesce(*[c.cast(dtype) if c.null_count < len(c) else pa.nulls(len(c), dtype)
                                        for c in columns]))
        else:
            encoded = True
            arrays.append(pc.coalesce(*[c.cast(pa.string()) if s in batch.json_columns else _json_encode_column(c)
                                        for s, c in zip(sources, columns)]))
        if encoded:
            json_columns.add(target)
    colsets = [[mapping.get(k, k) for k in keys] for keys in batch.colsets]
    colset_columns = [[{**c, "name": mapping.get(c["name"], c["name"])} for c in cols] for cols in batch.colset_columns]
    return _Batch(pa.table(arrays, names=list(groups)), colsets, colset_columns, json_columns)


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class LocalProvider(_HTTPProvider):
    """Answers cone queries from a :class:`SkyCache`, only for fully covered cones.

    The cone is planned exactly like the remote adapters do (:func:`models.plan_cone`:
    epoch widening, static pads, ``row_limit``), rows inside it are taken nearest first
    (``row_limit + 1`` -- the archives' TOP N+1 probe), the query-relative distance
    columns of the catalog's own adapter are recomputed, and the rows go through
    ``_HTTPProvider._sources`` -- the same conversion the remote adapter uses -- so every
    canonical field, epoch, positional error, pad/excess split and truncation flag is the
    same. Raises :class:`CoverageError` when the store does not fully cover the cone
    (``not_mirrored``, ``not_covered``, ``partially_covered``), when the catalog was
    mirrored with a different definition (``definition_changed``), when rows in the cone
    came from another archive than the catalog's provider (``fallback_rows``, stores
    written before fallback results were refused) or are older than the maximum age
    (``stale``: ``max_age_days``, the catalog parameter ``skycache_max_age_days`` or
    ``$SKYCACHE_MAX_AGE_DAYS``; no limit by default).

    ``query`` runs the synchronous lookup in a worker thread, so it never blocks the
    event loop and ``QueryExecutor`` timeouts apply to it.
    """

    provider_name = "skycache"

    def __init__(self, store: SkyCache | None = None, *, max_age_days: float | None = None) -> None:
        self.store = store or SkyCache()
        self.client = None
        self.timeout = 30.0
        self.max_response_bytes = 0
        self.guards = {}
        self.cache = None
        if max_age_days is None and os.getenv(ENV_MAX_AGE_DAYS):
            max_age_days = _env_float(ENV_MAX_AGE_DAYS, math.inf)  # invalid value: no limit
        self.max_age_days = max_age_days

    def _max_age_days(self, catalog: CatalogDefinition) -> float | None:
        value = catalog.parameters.get("skycache_max_age_days")
        try:
            value = float(value) if value is not None else self.max_age_days
        except (TypeError, ValueError):
            value = self.max_age_days
        return value if value is not None and math.isfinite(value) and value > 0 else None

    def _check_age(self, catalog: CatalogDefinition, result: ConeResult, cone_desc: dict[str, Any]) -> None:
        """CoverageError(reason='stale') when the cone's data is older than the maximum age.

        The age of a cone is that of its oldest part: the oldest retrieval time of the
        rows it returns and of the mirrors that last covered any part of it (the store's
        ``coverage_ages`` layers; a region mirrored again is judged by the new mirror, so
        an empty cone in a refreshed region is fresh). Stores written before those layers
        existed fall back, for cones without rows, to the catalog's oldest mirror.
        """
        max_age = self._max_age_days(catalog)
        if max_age is None:
            return
        times = [t for t in (_parse_time(v) for v in result.retrieved_at) if t is not None]
        if result.coverage_retrieved_at is not None:
            times += [t for t in (_parse_time(v) for v in result.coverage_retrieved_at) if t is not None]
        elif not times:  # legacy store, no rows: the age of the oldest mirror of the catalog
            times = [t for t in (_parse_time(m.get("retrieved_at")) for m in result.catalog_meta.get("mirrors") or [])
                     if t is not None]
        if not times:
            return
        age_days = (datetime.now(UTC) - min(times)).total_seconds() / 86400.0
        if age_days > max_age:
            raise CoverageError(f"{catalog.name}: mirrored data is {age_days:.1f} days old (limit {max_age:g} days)",
                                reason="stale", catalog=catalog.name, covered_fraction=1.0, **cone_desc)

    def query_sync(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        started = time.perf_counter()
        cone = plan_cone(catalog, target, radius_arcsec)
        provider = catalog.provider
        limit = _archive_row_limit(catalog, cone.row_limit)
        result = self.store.cone_search(catalog.name, cone.ra, cone.dec, cone.radius_arcsec, limit=limit,
                                        definition=catalog)
        cone_desc = {"ra": cone.ra, "dec": cone.dec, "radius_arcsec": cone.radius_arcsec}
        foreign = sorted({p for p in result.providers if p is not None and p != provider})
        if foreign:
            raise CoverageError(
                f"{catalog.name}: rows in this cone were mirrored from {', '.join(foreign)} (a fallback archive), "
                f"not from {provider}; the local copy is not used", reason="fallback_rows", catalog=catalog.name,
                covered_fraction=1.0, **cone_desc)
        self._check_age(catalog, result, cone_desc)
        computed = _query_relative_values(catalog, provider, cone.ra, cone.dec, result.ra, result.dec)
        # One spelling per column even where the archive varied its case between tiles.
        spellings = _column_spellings(result.catalog_meta.get("colsets") or [], catalog)
        rows: list[dict[str, Any]] = []
        for i, raw in enumerate(result.rows):
            row = {spellings.get(k, k): v for k, v in raw.items() if not _is_query_relative(k, provider)}
            for name, values, _unit in computed:
                row[name] = float(values[i])
            rows.append(row)
        columns = result.column_meta
        if columns is not None and spellings:
            merged: dict[str, ColumnMeta] = {}
            for col in columns:
                col.name = spellings.get(col.name, col.name)
                merged.setdefault(col.name, col)
            columns = list(merged.values())
        if columns is not None:
            present = {c.name for c in columns}
            for name, _values, unit in computed:
                if name not in present:
                    columns.append(ColumnMeta(name=name, unit=unit, datatype="double"))
        meta_info = result.catalog_meta
        location = self.store.catalog_path(catalog.name).resolve().as_uri()
        parameters = {"ra": cone.ra, "dec": cone.dec, "radius_arcsec": cone.radius_arcsec,
                      "row_limit": cone.row_limit, "store": location,
                      "mirrored_from": meta_info.get("archive_endpoint") or catalog.endpoint}
        query_text = (f"skycache cone ({cone.ra:.9f}, {cone.dec:.9f}) r={cone.radius_arcsec:.6f}\""
                      + (f" TOP {limit} ORDER BY distance" if limit is not None else ""))
        meta: dict[str, Any] = {"endpoint": location, "format": "hats-parquet", "cached": False}
        if provider == "tap":
            meta.update({"query": query_text, "http_method": None, "truncated": False, "query_status": None})
        if provider == "sdss":
            meta["mode"] = _sdss_mode(catalog)
            meta["query"] = query_text if meta["mode"] != "conesearch" else None
        if columns is not None:
            meta["columns"] = [c.as_dict() for c in columns]
        out = self._sources(
            catalog, rows, radius_arcsec, location, parameters,
            columns=_conversion_columns(columns), target=target, cone=cone,
            field_map=self._field_map(catalog, default_id=_default_id(catalog)),
            meta=meta,
        )
        # Provenance: when (and by which adapter) each row was retrieved from the archive.
        # Keyed by position: every stored row at one position comes from one archive
        # request (mirrors deduplicate and replace by position), and a row converted again
        # gets exactly its stored position -- unlike its source_id, which _sources
        # synthesizes from the row's index for catalogs without an archive id.
        origin = {(float(r), float(d)): (prov, when)
                  for r, d, prov, when in zip(result.ra, result.dec, result.providers, result.retrieved_at)}
        served_at = datetime.now(UTC).isoformat()
        endpoint = meta_info.get("archive_endpoint") or catalog.endpoint
        for src in [*out, *(out.meta.get("pad_sources") or []), *(out.meta.get("excess_sources") or [])]:
            prov, when = origin.get((float(src.ra), float(src.dec)), (None, None))
            if when is not None:
                src.provenance["retrieved_at"] = when
            src.provenance["served_at"] = served_at
            src.provenance["skycache"] = {"store": location, "mirrored_from": endpoint, "archive_provider": prov,
                                          "catalog_version": meta_info.get("version")}
        times = sorted(t for t in result.retrieved_at if t)
        mirrors = meta_info.get("mirrors") or []
        out.meta["skycache"] = {
            "hit": True,
            "store": location,
            "catalog_version": meta_info.get("version"),
            "mirrored_from": endpoint,
            "row_providers": sorted({p for p in result.providers if p}),
            "retrieved_at": {"oldest": times[0] if times else None, "newest": times[-1] if times else None},
            "last_mirror_at": mirrors[-1].get("retrieved_at") if mirrors else None,
            "definition_fingerprint": meta_info.get("definition_fingerprint"),
            "rows_in_cone": result.rows_in_cone,
            "rows_scanned": len(result.rows),
            "partitions_read": result.partitions_read,
            "lookup_ms": result.elapsed_ms,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return out

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        return await asyncio.to_thread(self.query_sync, catalog, target, radius_arcsec)


class SkyCacheProvider(CatalogProvider):
    """Local-first provider: the sky cache when it can answer, else the remote adapter.

    Falls back to the remote archive on :class:`CoverageError` (not covered, definition
    changed, stale, ...) and on any other error of the local store (a truncated or
    unreadable Parquet file, a malformed skycache.json, ...): a damaged cache never turns
    a working archive query into a failure. The outcome is recorded in
    ``meta['skycache']`` (``hit``, ``reason``, and ``error`` for local errors).
    """

    def __init__(self, remote: CatalogProvider, local: LocalProvider) -> None:
        self.remote = remote
        self.local = local

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        try:
            return await self.local.query(catalog, target, radius_arcsec)
        except CoverageError as exc:
            info: dict[str, Any] = {"hit": False, "reason": exc.reason, "covered_fraction": exc.covered_fraction}
        except Exception as exc:  # noqa: BLE001 - any local failure must fall back to the archive
            logger.warning("skycache: local query of %s failed (%s: %s); using the archive",
                           catalog.name, type(exc).__name__, exc)
            self.local.store.evict(catalog.name)
            info = {"hit": False, "reason": "local_error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        result = await self.remote.query(catalog, target, radius_arcsec)
        meta = getattr(result, "meta", None)
        if isinstance(meta, dict):
            meta["skycache"] = info
        return result


def wrap_providers(providers: Mapping[str, CatalogProvider], store: SkyCache | None = None) -> dict[str, CatalogProvider]:
    """Provider map whose adapters answer from the sky cache first (drop-in for CrossmatchService)."""
    local = LocalProvider(store)
    return {name: SkyCacheProvider(provider, local) for name, provider in providers.items()}


def archive_providers(providers: Mapping[str, CatalogProvider]) -> dict[str, CatalogProvider]:
    """The archive adapters behind a provider map (``SkyCacheProvider`` wrappers removed).

    Mirroring must always reach the archive: a wrapped provider would answer covered
    cones from the store being refreshed. A bare :class:`LocalProvider` cannot fetch
    anything and is rejected.
    """
    out: dict[str, CatalogProvider] = {}
    for name, provider in providers.items():
        while isinstance(provider, SkyCacheProvider):
            provider = provider.remote
        if isinstance(provider, LocalProvider):
            raise MirrorInputError(f"provider {name!r} is a LocalProvider (the sky cache itself); "
                                   "mirroring needs the archive adapters")
        out[name] = provider
    return out


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------


@dataclass
class MirrorReport:
    """Outcome of :func:`mirror_region`."""

    catalog: str
    region: dict[str, Any]
    queries: int = 0
    tiles_complete: int = 0
    tiles_failed: int = 0
    tiles_retried: int = 0  # requests repeated after the archive was unavailable (HTTP 5xx, throttled)
    tiles_split_timeout: int = 0  # tiles split into 4 because their request timed out
    circuit_waits: int = 0  # requests held back (not sent, not counted) while the mirror's breaker was open
    rows_fetched: int = 0
    rows_stored: int = 0
    rows_replaced: int = 0
    rows_deduplicated: int = 0  # rows at a position already fetched by another (overlapping) tile request
    rows_outside_coverage: int = 0  # fetched rows outside the cells this mirror covers (request overhang): not stored
    rows_moved: int = 0  # rows of sources the archive moved across the edge of the refreshed cells (kept once)
    rows_total: int = 0
    region_covered_fraction: float = 0.0
    coverage_area_deg2: float = 0.0
    elapsed_s: float = 0.0
    providers_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    store: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _OneCatalogRegistry:
    """Minimal registry view for QueryExecutor.definition_for (one modified definition)."""

    def __init__(self, catalog: CatalogDefinition) -> None:
        self._catalog = catalog

    @property
    def catalogs(self) -> dict[str, CatalogDefinition]:
        return {self._catalog.name: self._catalog}


@dataclass
class _Fetch:
    """One archive request. ``kind``: ok | truncated | too_big | timeout | unavailable | error |
    circuit_open (refused by the mirror's own open circuit breaker: nothing was sent)."""

    kind: str
    sources: list[CatalogSource] = field(default_factory=list)
    columns: list[ColumnMeta] | None = None
    error: str | None = None


class _NoCache(CacheManager):
    """An HTTP response cache that stores nothing: every mirror request reaches the archive."""

    def __init__(self) -> None:
        super().__init__(None)

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        return None


class _MirrorGuards(dict):
    """Circuit breakers of one mirror (see MIRROR_FAILURE_THRESHOLD): the adapters create a
    guard per endpoint with ``guards.setdefault(endpoint, EndpointGuard(<service settings>))``;
    here the new guard keeps the service's request rate but gets the mirror's breaker settings."""

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self and isinstance(default, EndpointGuard):
            default.failure_threshold = MIRROR_FAILURE_THRESHOLD
            default.recovery_seconds = MIRROR_RECOVERY_S
        return super().setdefault(key, default)


def _maxrec_request(build: Any) -> Any:
    """Wrap ``TapProvider.build_request`` to also send MAXREC = the query's TOP value."""

    def build_request(catalog: CatalogDefinition, adql: str) -> dict[str, str]:
        form = build(catalog, adql)
        top = _TOP_CLAUSE.match(adql)
        if top:
            form["MAXREC"] = top.group(1)
        return form

    return build_request


def _uses_http_transport(provider: CatalogProvider) -> bool:
    """An archive adapter whose ``query`` is the providers module's own (it goes through
    ``_HTTPProvider._request``, hence through its guards and response cache)."""
    return isinstance(provider, _HTTPProvider) and getattr(type(provider).query, "__module__", None) == \
        _HTTPProvider.__module__


async def _on_loop(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    """Run ``coro`` on another thread's event loop ``loop`` and wait for it on this one.

    Cancelling the waiting task cancels the coroutine on ``loop`` too.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


class _CallerLoopClient:
    """The caller's ``httpx.AsyncClient`` as seen from the mirror's worker loop.

    A client (its connection pool, TLS context, proxies, event hooks) belongs to the event
    loop it is used on, so every request is sent on the CALLER's loop and only awaited from
    the worker; parsing and converting the answer then happens in the worker thread. The
    adapters only use ``request`` (and ``get`` for VOSI capabilities).
    """

    def __init__(self, client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    async def request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        return await _on_loop(self._loop, self._client.request(method, url, **kwargs))

    async def get(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in ("_client", "_loop"):  # not set yet (e.g. while being copied)
            raise AttributeError(name)
        return getattr(self._client, name)


class _CallerLoopProvider(CatalogProvider):
    """A provider with its own ``query`` (not the providers module's HTTP path: a test
    stand-in or custom adapter), run on the caller's event loop exactly as before, since
    nothing is known about the loop affinity of its resources."""

    def __init__(self, provider: CatalogProvider, loop: asyncio.AbstractEventLoop) -> None:
        self.provider = provider
        self._loop = loop

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> Any:
        return await _on_loop(self._loop, self.provider.query(catalog, target, radius_arcsec))

    def __getattr__(self, name: str) -> Any:
        if name in ("provider", "_loop"):  # not set yet (e.g. while being copied)
            raise AttributeError(name)
        return getattr(self.provider, name)


def _mirror_adapters(providers: Mapping[str, CatalogProvider],
                     caller_loop: asyncio.AbstractEventLoop | None = None,
                     guards: _MirrorGuards | None = None) -> dict[str, CatalogProvider]:
    """Copies of the archive adapters for one mirror (same HTTP client and settings).

    Each copy gets the mirror's own circuit breakers (:class:`_MirrorGuards`), no response
    cache (:class:`_NoCache`) and, for TAP, MAXREC on every request. The caller's adapters
    -- e.g. the live service's -- are not modified. With ``caller_loop`` (the mirror runs
    on a worker thread's loop, see :func:`mirror_region`) each copy sends its requests
    through the caller's client on the caller's loop (:class:`_CallerLoopClient`) and
    converts the rows in the worker; other providers (test stand-ins, custom adapters with
    their own ``query``) run on the caller's loop as they are (:class:`_CallerLoopProvider`).
    """
    guards = guards if guards is not None else _MirrorGuards()
    no_cache = _NoCache()
    out: dict[str, CatalogProvider] = {}
    for name, provider in providers.items():
        if _uses_http_transport(provider):
            clone = copy.copy(provider)
            clone.guards = guards
            clone.cache = no_cache
            if isinstance(clone, TapProvider):
                clone.build_request = _maxrec_request(clone.build_request)
            if caller_loop is not None and clone.client is not None:
                clone.client = _CallerLoopClient(clone.client, caller_loop)
            provider = clone
        elif caller_loop is not None:
            provider = _CallerLoopProvider(provider, caller_loop)
        out[name] = provider
    return out


def _parse_output_limits(text: str) -> set[int]:
    """Row limits (default and hard, unit 'row') in a VOSI capabilities document."""
    limits: set[int] = set()
    for block in _OUTPUT_LIMIT.findall(text or ""):
        for _kind, attrs, value in _LIMIT_VALUE.findall(block):
            unit = re.search(r"unit\s*=\s*[\"']([^\"']*)[\"']", attrs)
            if unit is None or unit.group(1).strip().lower() in ("row", "rows"):
                limits.add(int(value))
    return limits


async def _tap_output_limits(provider: CatalogProvider, catalog: CatalogDefinition, timeout: float) -> set[int]:
    """The TAP service's <outputLimit> row counts, read once per mirror (empty if unknown).

    Only needed for JSON/CSV answers (a VOTable answer states OVERFLOW itself). The
    capabilities document is at ``<service>/capabilities`` for a ``<service>/sync`` endpoint
    (VOSI 1.1, TAP 1.1 section 2).
    """
    if not isinstance(provider, TapProvider) or not _uses_http_transport(provider) or provider.client is None:
        return set()
    if str(catalog.parameters.get("format", "json")).lower() not in ("json", "csv"):
        return set()
    endpoint = str(catalog.endpoint or "https://gea.esac.esa.int/tap-server/tap/sync").rstrip("/")
    if not endpoint.lower().endswith("/sync"):
        return set()
    url = endpoint[: -len("/sync")] + "/capabilities"
    try:
        response = await provider.client.get(url, timeout=min(float(timeout), 30.0))
        limits = _parse_output_limits(response.text) if response.status_code == 200 else set()
    except Exception as exc:  # noqa: BLE001 - the guard is optional; any failure means "unknown"
        logger.info("skycache: capabilities of %s not available (%s: %s)", url, type(exc).__name__, exc)
        return set()
    return limits


def _resolve_catalog(catalog: str | CatalogDefinition, registry: Any | None) -> CatalogDefinition:
    if isinstance(catalog, CatalogDefinition):
        return catalog
    reg = registry or CatalogRegistry(Settings().catalog_registry_path)
    try:
        definition = reg.get(catalog)
    except KeyError as exc:
        raise MirrorInputError(f"unknown catalog {catalog!r}") from exc
    return definition


async def _in_worker_loop(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Await ``factory()`` on a new event loop in a worker thread, without blocking this loop.

    Cancelling the caller cancels the worker's task and waits for it to unwind (a store
    write already in progress completes atomically), so nothing keeps running unseen.
    """
    lock = threading.Lock()
    handle: dict[str, Any] = {}

    async def main() -> Any:
        with lock:
            if handle.get("cancelled"):
                raise asyncio.CancelledError
            handle["loop"] = asyncio.get_running_loop()
            handle["task"] = asyncio.current_task()
        return await factory()

    worker = asyncio.ensure_future(asyncio.to_thread(asyncio.run, main()))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with lock:
            handle["cancelled"] = True
            loop, task = handle.get("loop"), handle.get("task")
        if task is not None:
            with contextlib.suppress(RuntimeError):  # the worker's loop already finished
                loop.call_soon_threadsafe(task.cancel)
        with contextlib.suppress(BaseException):
            await worker
        raise


async def mirror_region(
    catalog: str | CatalogDefinition,
    *,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    healpix_order: int | None = None,
    healpix_pixels: Sequence[int] | None = None,
    store: SkyCache | None = None,
    providers: Mapping[str, CatalogProvider] | None = None,
    registry: Any | None = None,
    client: httpx.AsyncClient | None = None,
    tile_rows: int | None = None,
    max_queries: int | None = None,
    max_radius_deg: float | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float | None = None,
) -> MirrorReport:
    """Mirror every row of ``catalog`` in a cone (ra, dec, radius_deg) or HEALPix pixels.

    Each archive request goes through the service's own adapters via
    :class:`crossmatch.QueryExecutor`, for a target without epoch (an exact cone) and
    ``max_rows = tile_rows``, with the registry's fallback archive DISABLED (its rows
    differ in shape and precision from the primary's, so mixing them would store the same
    source twice and serve rows unlike the archive's). ``SkyCacheProvider`` wrappers in
    ``providers`` are unwrapped: a refresh always reaches the archive. The adapters are
    used through per-mirror copies (:func:`_mirror_adapters`): the mirror's own circuit
    breakers, no HTTP response cache, and MAXREC = TOP on TAP requests.

    A request is *complete* when the adapter reports no archive truncation (fewer than
    TOP N+1 rows, no OVERFLOW, and -- for TAP answers in JSON/CSV, which cannot say
    OVERFLOW -- not exactly one of the service's published output limits). An incomplete
    one -- truncated, over the adapter's response-size limit, or timed out (at most
    ``TIMEOUT_SPLIT_DEPTH`` consecutive times) -- is split: a cone into the HEALPix pixels
    (order with resolution <= r/2) overlapping it, a pixel into its 4 children (up to
    order 18). A request the archive could not serve (HTTP 5xx, throttled) is retried
    ``UNAVAILABLE_RETRIES`` times; a request refused by the mirror's open circuit breaker
    waits for it (not counted); other failures fail the tile. Pixels are fetched with a
    cone around the pixel centre that encloses the whole pixel. The coverage MOC gains the
    complete pixels, or for a complete root cone the order-C pixels fully inside it. At
    most ``max_queries`` requests are made (so at most that many HEALPix pixels are accepted).

    Rows are merged into the store by POSITION: a complete answer holds every archive row
    at every position inside its cone, so within this mirror the rows at one position are
    kept from one request only (the first tile in HEALPix order that returned it; all its
    rows there, e.g. one per SDSS spectrum). Distinct sources sharing an identifier (e.g.
    4XMM names) and catalogs without archive ids (``_sources`` numbers such rows by their
    index in each answer) are therefore kept intact. Every covered cell then holds the
    rows of one snapshot: the cells this mirror covers get exactly the rows fetched there
    (previous rows in them are replaced), the other cells keep theirs, and rows outside the
    coverage (the overhang of tile and cone requests) are not stored; sources the archive
    moved across the edge of the refreshed cells are kept once (:func:`_merge_masks`).

    ``timeout`` (seconds), when given, bounds every archive request (each catalog's
    registry ``timeout_seconds`` is capped at it, like ``Settings.catalog_timeout_cap_seconds``
    does for searches); by default the registry's per-catalog timeout applies (30 s when a
    catalog sets none).

    The mirror runs on its own event loop in a worker thread, so converting the rows of
    large tiles (``models.normalize_source_record``, ~1 ms per row) and merging never
    block the caller's loop (an API server stays responsive). Archive requests are still
    sent through the caller's client on the caller's loop, and providers that are not the
    providers module's HTTP adapters (custom ``query``) run on the caller's loop.
    """
    caller_loop = asyncio.get_running_loop()
    own_client = None
    if providers is None and client is None:
        own_client = client = httpx.AsyncClient(timeout=timeout or DEFAULT_REQUEST_TIMEOUT_S, follow_redirects=True)
    try:
        return await _in_worker_loop(functools.partial(
            _mirror_region, catalog, ra=ra, dec=dec, radius_deg=radius_deg, healpix_order=healpix_order,
            healpix_pixels=healpix_pixels, store=store, providers=providers, registry=registry, client=client,
            tile_rows=tile_rows, max_queries=max_queries, max_radius_deg=max_radius_deg, concurrency=concurrency,
            timeout=timeout, caller_loop=caller_loop))
    finally:
        if own_client is not None:
            await own_client.aclose()


async def _mirror_region(
    catalog: str | CatalogDefinition,
    *,
    ra: float | None,
    dec: float | None,
    radius_deg: float | None,
    healpix_order: int | None,
    healpix_pixels: Sequence[int] | None,
    store: SkyCache | None,
    providers: Mapping[str, CatalogProvider] | None,
    registry: Any | None,
    client: httpx.AsyncClient | None,
    tile_rows: int | None,
    max_queries: int | None,
    max_radius_deg: float | None,
    concurrency: int,
    timeout: float | None,
    caller_loop: asyncio.AbstractEventLoop,
) -> MirrorReport:
    """The body of :func:`mirror_region`, run on the worker loop (requests go through
    ``caller_loop``, see :func:`_mirror_adapters`)."""
    started = time.perf_counter()
    if timeout is not None and not (math.isfinite(float(timeout)) and float(timeout) > 0):
        raise MirrorInputError("timeout must be a positive number of seconds")
    request_timeout = float(timeout) if timeout is not None else DEFAULT_REQUEST_TIMEOUT_S
    cache = store or SkyCache()
    definition = _resolve_catalog(catalog, registry)
    if not definition.enabled:
        raise MirrorInputError(f"catalog {definition.name!r} is disabled in the registry")
    if definition.provider not in MIRRORABLE_PROVIDERS:
        raise MirrorInputError(
            f"catalog {definition.name!r} uses provider {definition.provider!r}, whose results do not reveal "
            f"server-side truncation; mirroring supports {sorted(MIRRORABLE_PROVIDERS)}")
    tile_rows = int(tile_rows or _env_int(ENV_TILE_ROWS, DEFAULT_TILE_ROWS))
    if tile_rows < 1 or tile_rows > 100_000:
        raise MirrorInputError("tile_rows must be in 1..100000")
    max_queries = int(max_queries or _env_int(ENV_MAX_QUERIES, DEFAULT_MAX_QUERIES))
    limit_deg = float(max_radius_deg or _env_float(ENV_MAX_RADIUS, DEFAULT_MAX_RADIUS_DEG))
    if providers is not None:
        providers = archive_providers(providers)

    roots: list[tuple[str, int, int, float, float, float]] = []  # (kind, order, pixel, ra, dec, radius_deg)
    if healpix_pixels is not None:
        if healpix_order is None or not 0 <= int(healpix_order) <= MAX_TILE_ORDER:
            raise MirrorInputError(f"healpix_order must be in 0..{MAX_TILE_ORDER}")
        order = int(healpix_order)
        pixels = sorted({int(p) for p in healpix_pixels})
        if not pixels or any(p < 0 or p >= 12 * 4**order for p in pixels):
            raise MirrorInputError(f"healpix_pixels must be non-empty and within 0..{12 * 4**order - 1}")
        if len(pixels) > max_queries:
            # Every pixel needs at least one archive query: the rest could never be mirrored.
            raise MirrorInputError(f"{len(pixels)} HEALPix pixels need at least {len(pixels)} archive queries, "
                                   f"more than the limit of {max_queries} (${ENV_MAX_QUERIES}); give fewer pixels "
                                   "or pixels of a lower order")
        area = len(pixels) * pixel_resolution_deg(order) ** 2
        if area > math.pi * limit_deg**2:
            raise MirrorInputError(f"region of {area:.3f} deg^2 exceeds the {math.pi * limit_deg**2:.3f} deg^2 "
                                   f"limit (${ENV_MAX_RADIUS}={limit_deg:g} deg)")
        # Geometry of every root pixel at once, off the event loop.
        c_ras, c_decs, radii = await asyncio.to_thread(pixel_geometry, order, pixels)
        roots.extend(("pixel", order, p, float(r), float(d), float(rad))
                     for p, r, d, rad in zip(pixels, c_ras, c_decs, radii))
        region: dict[str, Any] = {"type": "healpix", "order": order, "pixels": pixels}
        requested = Moc.from_pixels(order, pixels)
    else:
        if ra is None or dec is None or radius_deg is None:
            raise MirrorInputError("give ra, dec and radius_deg, or healpix_order and healpix_pixels")
        target = validate_target(ra, dec)
        radius = float(radius_deg)
        if not math.isfinite(radius) or radius <= 0:
            raise MirrorInputError("radius_deg must be > 0")
        if radius > limit_deg:
            raise MirrorInputError(f"radius_deg {radius:g} exceeds the limit of {limit_deg:g} deg (${ENV_MAX_RADIUS})")
        roots.append(("cone", 0, 0, target.ra, target.dec, radius))
        region = {"type": "cone", "ra": target.ra, "dec": target.dec, "radius_deg": radius}
        cov_order = order_for_resolution(radius * COVERAGE_RES_FRACTION, hi=MAX_COVERAGE_ORDER)
        requested = await asyncio.to_thread(Moc.from_cone, target.ra, target.dec, radius, cov_order)

    report = MirrorReport(definition.name, region, store=str(cache.root))
    # Primary archive only: QueryExecutor tries parameters["fallback"] only when it is a dict.
    fetch_def = replace(definition, max_rows=tile_rows, parameters={**definition.parameters, "fallback": None})
    retrieved_at = datetime.now(UTC).isoformat()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    batches: list[tuple[tuple[int, int], _Batch]] = []
    new_cov: list[Moc] = []
    complete_cones: list[tuple[float, float, float]] = []  # (ra, dec, radius_deg) of every complete request
    budget_exhausted = False
    # The archive's own latest failure: a request refused by the mirror's open breaker names
    # it, so the report always carries the cause of an outage, not only "circuit is open".
    last_archive_error: str | None = None

    if providers is None:  # mirror_region created or was given a client: the default adapters on it
        providers = provider_map(client, timeout=request_timeout, cache=_NoCache())
    guards = _MirrorGuards()
    adapters = _mirror_adapters(providers, caller_loop, guards)
    # timeout_cap: an explicit timeout bounds every request, whatever the catalog's own timeout.
    executor = QueryExecutor(adapters, timeout=request_timeout, registry=_OneCatalogRegistry(fetch_def),
                             timeout_cap=None if timeout is None else request_timeout)
    output_limits: set[int] = set()
    if fetch_def.provider == "tap" and "tap" in adapters:
        output_limits = await _tap_output_limits(adapters["tap"], fetch_def, request_timeout)

    async def fetch_once(target: Target, radius_deg_: float) -> _Fetch | None:
        nonlocal budget_exhausted, last_archive_error
        if report.queries >= max_queries:
            budget_exhausted = True
            return None
        report.queries += 1
        plan = QueryPlan(fetch_def.name, fetch_def.provider, fetch_def.endpoint, {}, radius_deg_ * 3600.0,
                         fetch_def.wavelength)
        async with semaphore:
            successes, failures = await executor.execute([plan], target)
        if failures:
            failure = failures[0]
            message = f"{failure.error_type}: {failure.message}"
            if "circuit is open" in (failure.message or ""):
                # Refused by the mirror's own breaker before anything was sent to the archive.
                report.queries -= 1
                return _Fetch("circuit_open", error=message)
            if "byte limit" in (failure.message or ""):  # the adapter's response-size guard fired
                return _Fetch("too_big", error=message)
            last_archive_error = message
            if failure.error_type in _TIMEOUT_ERRORS:
                return _Fetch("timeout", error=message)
            if failure.error_type in _UNAVAILABLE_ERRORS:
                return _Fetch("unavailable", error=message)
            return _Fetch("error", error=message)
        result = successes[0][1]
        meta = getattr(result, "meta", {}) or {}
        if meta.get("fallback"):  # never enabled above; refuse rather than store foreign rows
            return _Fetch("error", error=f"answered by the fallback archive {meta['fallback'].get('provider')}")
        sources = list(result) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
        columns = [ColumnMeta(**c) for c in meta.get("columns") or []] or None
        truncated = bool(meta.get("archive_truncated"))
        raw_rows = int(meta.get("raw_row_count") or 0)
        if not truncated and output_limits and raw_rows in output_limits:
            # Exactly the service's output limit, below TOP N+1: a silently capped answer.
            return _Fetch("truncated", sources, columns,
                          error=f"was capped at the service's output limit of {raw_rows} rows")
        return _Fetch("truncated" if truncated else "ok", sources, columns)

    def circuit_refuses() -> bool:
        """Whether the mirror's breaker would refuse a request now: open (within its recovery
        time), or half-open with its recovery probe in flight (not yet expired), exactly as
        ``EndpointGuard.acquire`` decides. Waiting tiles then do not run the executor just to
        be refused: with many tiles waiting, that polling kept the worker thread busy, and
        through the GIL slowed every other thread (the caller's event loop included)."""
        for guard in guards.values():
            state = guard.state
            if state == "open":
                return True
            if state == "half_open" and guard.probe_in_flight:
                probe_started = getattr(guard, "_probe_started", None)
                if probe_started is None or guard.clock() - probe_started < guard.probe_timeout_seconds:
                    return True
        return False

    async def fetch(c_ra: float, c_dec: float, radius_deg_: float) -> _Fetch | None:
        retries = 0
        waited = 0.0
        target = validate_target(c_ra, c_dec)
        while True:
            refused = "CatalogUnavailableError: Provider circuit is open" if circuit_refuses() else None
            if refused is None:
                res = await fetch_once(target, radius_deg_)
                if res is None:
                    return None
                if res.kind == "circuit_open":  # e.g. its recovery probe is in flight
                    refused = res.error
            if refused is not None:
                if waited >= CIRCUIT_WAIT_LIMIT_S:
                    cause = f"; last archive error: {last_archive_error}" if last_archive_error else ""
                    return _Fetch("unavailable", error=f"{refused} (the archive did not recover within "
                                                        f"{CIRCUIT_WAIT_LIMIT_S:g} s{cause})")
                report.circuit_waits += 1
                await asyncio.sleep(CIRCUIT_POLL_S)
                waited += CIRCUIT_POLL_S
                continue
            if res.kind == "unavailable" and retries < UNAVAILABLE_RETRIES:
                retries += 1
                report.tiles_retried += 1
                await asyncio.sleep(RETRY_DELAY_S * 2 ** (retries - 1))
                continue
            return res

    async def accept(res: _Fetch, coverage: Moc, key: tuple[int, int], cone: tuple[float, float, float]) -> None:
        rows = [StoredRow.from_source(src, provider=fetch_def.provider, retrieved_at=retrieved_at, columns=res.columns)
                for src in res.sources]
        batch = await asyncio.to_thread(_batch_from_rows, rows, definition.name)
        report.rows_fetched += len(rows)
        report.tiles_complete += 1
        new_cov.append(coverage)
        complete_cones.append(cone)
        batches.append((key, batch))

    def splittable(res: _Fetch, order: int, timeouts: int) -> bool:
        if order >= MAX_TILE_ORDER:
            return False
        return res.kind in {"truncated", "too_big"} or (res.kind == "timeout" and timeouts < TIMEOUT_SPLIT_DEPTH)

    async def tile(order: int, pixel: int, timeouts: int = 0,
                   geometry: tuple[float, float, float] | None = None) -> None:
        nonlocal budget_exhausted
        if report.queries >= max_queries:  # nothing left to fetch it with: no geometry needed
            budget_exhausted = True
            report.tiles_failed += 1
            return
        await asyncio.sleep(0)  # let other tasks (and requests) run between tiles
        if geometry is None:
            c_ras, c_decs, radii = pixel_geometry(order, [pixel])
            geometry = (float(c_ras[0]), float(c_decs[0]), float(radii[0]))
        c_ra, c_dec, radius = geometry
        res = await fetch(c_ra, c_dec, radius)
        if res is None:
            report.tiles_failed += 1
            return
        if res.kind == "ok":
            accept_key = (pixel << (2 * (SPATIAL_INDEX_ORDER - order)), order)
            await accept(res, Moc.from_pixels(order, [pixel]), accept_key, geometry)
            return
        if splittable(res, order, timeouts):
            if res.kind == "timeout":
                report.tiles_split_timeout += 1
                report.warnings.append(f"order-{order} pixel {pixel}: {res.error}; split into 4 order-{order + 1} tiles")
            children = [pixel * 4 + i for i in range(4)]
            await asyncio.gather(*(tile(order + 1, c, timeouts + 1 if res.kind == "timeout" else 0)
                                   for c in children))
            return
        report.tiles_failed += 1
        if res.kind in {"truncated", "too_big"}:
            reason = f"still truncated at order {MAX_TILE_ORDER}"
        elif res.kind == "timeout" and order < MAX_TILE_ORDER:
            reason = f"{res.error} (after {TIMEOUT_SPLIT_DEPTH} consecutive timeout splits)"
        else:
            reason = res.error or res.kind
        report.warnings.append(f"order-{order} pixel {pixel}: {reason}")

    async def root_cone(c_ra: float, c_dec: float, radius: float) -> None:
        res = await fetch(c_ra, c_dec, radius)
        if res is None:
            report.tiles_failed += 1
            return
        if res.kind == "ok":
            cov_order = order_for_resolution(radius * COVERAGE_RES_FRACTION, hi=MAX_COVERAGE_ORDER)
            coverage = await asyncio.to_thread(Moc.from_cone, c_ra, c_dec, radius, cov_order)
            await accept(res, coverage, (-1, 0), (c_ra, c_dec, radius))
            return
        if res.kind not in {"truncated", "too_big", "timeout"}:
            report.tiles_failed += 1
            report.warnings.append(f"cone: {res.error}")
            return
        order = order_for_resolution(radius / 2.0, hi=MAX_TILE_ORDER)
        tiles = cone_pixels(c_ra, c_dec, radius, order)
        why = {"truncated": res.error or f"held more than {tile_rows} rows",
               "too_big": "exceeded the adapter's response size limit",
               "timeout": f"timed out ({res.error})"}[res.kind]
        if res.kind == "timeout":
            report.tiles_split_timeout += 1
        report.warnings.append(f"cone {why}; split into {tiles.size} order-{order} tiles")
        await asyncio.gather(*(tile(order, int(p), 1 if res.kind == "timeout" else 0) for p in tiles))

    await asyncio.gather(*(
        root_cone(r_ra, r_dec, r_rad) if kind == "cone" else tile(order, pixel, geometry=(r_ra, r_dec, r_rad))
        for kind, order, pixel, r_ra, r_dec, r_rad in roots
    ))
    if budget_exhausted:
        report.warnings.append(f"stopped after {max_queries} archive queries (${ENV_MAX_QUERIES}); "
                               "the rest of the region was not mirrored")
    if not new_cov:
        raise MirrorError(f"{definition.name}: no part of the region could be mirrored: "
                          + "; ".join(report.warnings or ["no archive query succeeded"]))

    added = Moc()
    for moc in new_cov:
        added = added | moc
    record = {
        "region": region, "retrieved_at": retrieved_at, "archive_endpoint": definition.endpoint,
        "providers": [fetch_def.provider], "queries": report.queries, "tiles_complete": report.tiles_complete,
        "tiles_failed": report.tiles_failed, "rows_fetched": report.rows_fetched, "tile_rows": tile_rows,
        "coverage_added_deg2": added.area_deg2, "warnings": [],
    }
    meta = await asyncio.to_thread(_merge_and_write, cache, definition, batches, added, record, report,
                                   complete_cones)
    report.rows_total = int(meta["rows"])
    total = Moc.from_json(meta["coverage"])
    report.coverage_area_deg2 = total.area_deg2
    wanted = requested.cells29
    report.region_covered_fraction = (total & requested).cells29 / wanted if wanted else 0.0
    report.providers_used = [fetch_def.provider]
    report.elapsed_s = round(time.perf_counter() - started, 3)
    logger.info("skycache mirror %s: %d queries, %d rows fetched, %d stored, total %d",
                definition.name, report.queries, report.rows_fetched, report.rows_stored, report.rows_total)
    return report


_STORE_READ_ERRORS = (OSError, ValueError, TypeError, KeyError, pa.ArrowException, AstroSearchError, _StaleStateError)


def _position_keys(table: pa.Table) -> np.ndarray:
    """Exact (ra, dec) of each row as one comparable value (complex128 ra + i dec)."""
    ra = np.asarray(table.column(SC_RA).to_numpy(), dtype=np.float64)
    dec = np.asarray(table.column(SC_DEC).to_numpy(), dtype=np.float64)
    return ra + 1j * dec


def _dedupe_batches(batches: Sequence[tuple[tuple[int, int], _Batch]]) -> tuple[list[_Batch], int]:
    """Keep the rows at each position from the first batch (in HEALPix tile order) holding it.

    Overlapping tile cones fetch the same rows more than once. A complete archive answer
    holds EVERY row at every position inside its cone, so each batch holding a position
    has the same rows there: all of them are kept from one batch (several rows per
    position survive, e.g. an SDSS objID joined to several spectra), none twice. Keys are
    positions, not source ids: two different sources may share an id (4XMM IAU names),
    and rows without an archive id get ids from their index in each answer.
    Returns the filtered batches and the number of rows dropped.
    """
    ordered = [batch for _key, batch in sorted(batches, key=lambda item: item[0])]
    if not ordered:
        return [], 0
    keys = np.concatenate([_position_keys(b.table) for b in ordered])
    if keys.size == 0:
        return ordered, 0
    owner = np.concatenate([np.full(b.num_rows, i, dtype=np.int64) for i, b in enumerate(ordered)])
    _unique, inverse = np.unique(keys, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    first = np.full(_unique.size, len(ordered), dtype=np.int64)
    np.minimum.at(first, inverse, owner)
    keep = owner == first[inverse]
    out: list[_Batch] = []
    offset = 0
    for batch in ordered:
        mask = keep[offset: offset + batch.num_rows]
        offset += batch.num_rows
        out.append(batch if mask.all() else batch.filter(mask))
    return out, int((~keep).sum())


def _read_previous(cache: SkyCache, definition: CatalogDefinition,
                   report: MirrorReport) -> tuple[dict[str, Any] | None, _Batch | None]:
    """(metadata, rows) of the stored copy to merge into, or (None, None) to start afresh.

    A copy mirrored with another definition is discarded (warning). Transient read errors
    (e.g. PermissionError from a scanner holding a leaf file on Windows) are retried
    ``READ_ATTEMPTS`` times; if the copy still cannot be read, :class:`MirrorStoreError`
    aborts the mirror and nothing is changed. Only a copy that is provably damaged --
    unparseable Parquet or skycache.json, or leaf files that stay missing or belong to
    another version while the caller holds the catalog's lock -- is replaced (warning).
    """
    name = definition.name
    last: BaseException | None = None
    for attempt in range(READ_ATTEMPTS):
        if attempt:
            time.sleep(READ_RETRY_DELAY_S)
        cache.evict(name)
        try:
            old_meta = cache.metadata(name)
            if old_meta is None:
                return None, None
            recorded = _recorded_signature(old_meta)
            changes = (_signature_changes(recorded, definition_signature(definition)) if recorded is not None
                       else ["no definition recorded"])
            if changes:
                report.warnings.append(
                    f"the catalog definition changed since the previous mirror ({', '.join(changes)}); "
                    f"the previous local copy ({old_meta.get('rows', 0)} rows) was discarded")
                return None, None
            return old_meta, cache.load_batch(name)
        except (FileNotFoundError, _StaleStateError) as exc:  # missing/foreign leaf: retried, then damaged
            last = exc
        except OSError as exc:  # transient (a handle on a file): retried, then abort
            last = exc
        except (ValueError, TypeError, KeyError, pa.ArrowException, AstroSearchError) as exc:
            report.warnings.append(f"the previous local copy is damaged ({type(exc).__name__}: {exc}); "
                                   "it was replaced by this mirror")
            return None, None
    if isinstance(last, (FileNotFoundError, _StaleStateError)):
        report.warnings.append(f"the previous local copy is damaged ({type(last).__name__}: {last}); "
                               "it was replaced by this mirror")
        return None, None
    raise MirrorStoreError(
        f"{name}: the previous local copy could not be read after {READ_ATTEMPTS} attempts "
        f"({type(last).__name__}: {last}); nothing was changed. Retry later, or delete the catalog "
        "(astrosearch skycache delete) to start afresh") from last


def _row_cells(table: pa.Table) -> np.ndarray:
    """The order-29 cell of each row as a one-cell range (for Moc.contains_ranges)."""
    h29 = np.asarray(table.column(SPATIAL_INDEX_COLUMN).to_numpy(), dtype=np.int64)
    return np.stack([h29, h29 + 1], axis=1)


def _archive_id_mask(table: pa.Table, catalog_name: str) -> np.ndarray:
    """Rows whose source id came from the archive (``_sources`` names rows without one
    ``<catalog>-<index in the answer>``, which identifies nothing across answers)."""
    if not table.num_rows:
        return np.zeros(0, dtype=bool)
    synthesized = pc.match_substring_regex(table.column(SC_ID), f"^{re.escape(catalog_name)}-[0-9]+$")
    return ~np.asarray(pc.fill_null(synthesized, True).to_numpy(zero_copy_only=False), dtype=bool)


def _ids_in(table: pa.Table, mask: np.ndarray, others: pa.Table, other_mask: np.ndarray) -> np.ndarray:
    """``mask`` rows of ``table`` whose source id is one of the ids of the ``other_mask`` rows of ``others``."""
    out = np.zeros(table.num_rows, dtype=bool)
    if not mask.any() or not other_mask.any():
        return out
    values = others.column(SC_ID).filter(pa.array(other_mask)).combine_chunks()
    hits = pc.is_in(table.column(SC_ID), value_set=values)
    return mask & np.asarray(pc.fill_null(hits, False).to_numpy(zero_copy_only=False), dtype=bool)


def _inside_cones(ra: np.ndarray, dec: np.ndarray, cones: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """Positions inside at least one cone (ra, dec, radius_deg), by more than CONE_EDGE_MARGIN_DEG."""
    inside = np.zeros(ra.size, dtype=bool)
    for c_ra, c_dec, radius in cones:
        if inside.all():
            break
        inside |= angular_sep_deg(c_ra, c_dec, ra, dec) <= radius - CONE_EDGE_MARGIN_DEG
    return inside


def _merge_masks(old: pa.Table, new: pa.Table, added: Moc, coverage: Moc,
                 cones: Sequence[tuple[float, float, float]], catalog_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Which old and new rows a refresh keeps: (keep_old, keep_new, moved).

    Every covered cell holds the rows of ONE archive snapshot, the one its coverage age
    records: the cells this mirror covers (``added``) hold exactly the rows it fetched
    there (older rows in them are replaced), every other covered cell keeps its previous
    rows, and nothing outside ``coverage`` is stored (tile and cone requests fetch rows
    beyond the pixels they complete; such rows can never be served, since a cone is only
    answered locally when every cell it touches is covered). Mixing snapshots inside one
    cell would store a source twice (or lose it) when the archive revised its position
    between two mirrors.

    Across the edge of ``added`` two snapshots meet, so a source the archive MOVED across
    that edge would be stored at both positions (moved in) or at neither (moved out). The
    requests of this mirror are complete for their whole cones, which reach beyond
    ``added``, and so tell such moves apart from new or deleted sources:

    * an old row outside ``added`` whose position lies inside a complete request cone but
      holds no row any more, while a row with the same archive id appeared inside
      ``added``, moved in: it is dropped (the source is kept once, at its new position);
    * a fetched row outside ``added`` (inside ``coverage``) at a position with no old row,
      carrying the archive id of an old row that vanished from inside ``added``, moved
      out: it is kept.

    Ids are used only for this pairing, never to replace rows (distinct sources may share
    an id, e.g. 4XMM IAU names; rows without archive ids are never paired). A move longer
    than the reach of the request cones beyond ``added`` (the pixel overhang of a tile
    cone, or the rim of a root cone) cannot be recognised.
    """
    keep_old = np.zeros(old.num_rows, dtype=bool)
    keep_new = np.zeros(new.num_rows, dtype=bool)
    old_added = added.contains_ranges(_row_cells(old)) if old.num_rows else keep_old.copy()
    new_added = added.contains_ranges(_row_cells(new)) if new.num_rows else keep_new.copy()
    if old.num_rows:
        keep_old = ~old_added & coverage.contains_ranges(_row_cells(old))
    keep_new |= new_added
    moved = 0
    if old.num_rows and new.num_rows:
        old_keys, new_keys = _position_keys(old), _position_keys(new)
        old_vanished = ~np.isin(old_keys, new_keys)  # no fetched row at this position any more
        new_appeared = ~np.isin(new_keys, old_keys)  # no stored row at this position before
        old_ids, new_ids = _archive_id_mask(old, catalog_name), _archive_id_mask(new, catalog_name)
        # Moved in: vanished from a position the complete requests saw, reappeared inside ``added``.
        arrived = new_added & new_appeared & new_ids
        candidates = np.flatnonzero(_ids_in(old, keep_old & old_vanished & old_ids, new, arrived))
        if candidates.size:
            ra = np.asarray(old.column(SC_RA).take(pa.array(candidates)).to_numpy(), dtype=np.float64)
            dec = np.asarray(old.column(SC_DEC).take(pa.array(candidates)).to_numpy(), dtype=np.float64)
            gone = candidates[_inside_cones(ra, dec, cones)]
            keep_old[gone] = False
            moved += int(gone.size)
        # Moved out: vanished from inside ``added``, reappeared in the fetched margin.
        departed = old_added & old_vanished & old_ids
        margin = ~new_added & new_appeared & new_ids & coverage.contains_ranges(_row_cells(new))
        landed = _ids_in(new, margin, old, departed)
        keep_new |= landed
        moved += int(landed.sum())
    return keep_old, keep_new, moved


def _merge_and_write(cache: SkyCache, definition: CatalogDefinition, batches: list[tuple[tuple[int, int], _Batch]],
                     added: Moc, record: dict[str, Any], report: MirrorReport,
                     cones: Sequence[tuple[float, float, float]] = ()) -> dict[str, Any]:
    """Merge the fetched rows into the stored copy (see :func:`_merge_masks`) and rewrite it.

    ``cones`` are the (ra, dec, radius_deg) of this mirror's complete archive requests.
    """
    path = cache.catalog_path(definition.name)
    with _exclusive(path):  # read-merge-write is atomic across threads and processes
        new_batches, dropped = _dedupe_batches(batches)
        new = _unify_batches(new_batches)
        report.rows_deduplicated = dropped
        old_meta, old = _read_previous(cache, definition, report)
        old = old if old is not None else _Batch.empty()
        old_cov = Moc.from_json(old_meta.get("coverage")) if old_meta else Moc()
        coverage = old_cov | added
        keep_old, keep_new, moved = _merge_masks(old.table, new.table, added, coverage, cones, definition.name)
        kept_old = old.filter(keep_old) if old.num_rows else old
        kept_new = new.filter(keep_new) if new.num_rows else new
        report.rows_replaced = old.num_rows - kept_old.num_rows
        report.rows_stored = kept_new.num_rows
        report.rows_outside_coverage = new.num_rows - kept_new.num_rows
        report.rows_moved = moved
        record["rows_stored"] = report.rows_stored
        record["warnings"] = list(report.warnings)
        mirrors = (list(old_meta.get("mirrors") or []) if old_meta else []) + [record]
        merged = _unify_batches([kept_old, kept_new])
        merged = _rename_batch_columns(merged, _column_spellings(merged.colsets, definition))
        return cache.write_batch(definition, merged, coverage, mirrors=mirrors,
                                 coverage_ages=_coverage_ages(old_meta, added, record["retrieved_at"]))


def _coverage_ages(old_meta: Mapping[str, Any] | None, added: Moc, retrieved_at: str) -> list[tuple[str, Moc]]:
    """Coverage layers by retrieval time after a mirror: the new region first, older layers minus it.

    Stores written before layers were recorded get one layer for their whole coverage, dated
    by their oldest mirror (the conservative age they were judged by until now).
    """
    layers: list[tuple[str, Moc]] = [(retrieved_at, added)]
    if old_meta:
        recorded = old_meta.get("coverage_ages")
        if recorded is None:
            times = [m.get("retrieved_at") for m in old_meta.get("mirrors") or [] if m.get("retrieved_at")]
            when = min(times, key=lambda t: _parse_time(t) or datetime.max.replace(tzinfo=UTC)) if times else \
                old_meta.get("updated_at")
            recorded = [{"retrieved_at": when, **Moc.from_json(old_meta.get("coverage")).as_json()}] if when else []
        for layer in recorded:
            rest = Moc.from_json(layer) - added
            if not rest.empty:
                layers.append((str(layer["retrieved_at"]), rest))
    return layers


# ---------------------------------------------------------------------------
# Remote HATS catalogs via lsdb (optional)
# ---------------------------------------------------------------------------

# Public HATS catalogs whose columns match a registry definition (verified: the Gaia DR3
# HATS catalog carries the gaia_source columns under their archive names).
HATS_REMOTE_CATALOGS: dict[str, str] = {"gaia_dr3": "https://data.lsdb.io/hats/gaia_dr3/gaia"}


def lsdb_available() -> bool:
    try:
        import lsdb  # noqa: F401
    except Exception:  # noqa: BLE001 - ImportError or a broken optional stack (numba, dask) alike
        return False
    return True


def _float32_value(value: Any) -> float:
    """A float32 value as the double the archives print for it (TAP services write float32
    columns with their shortest float32 representation, e.g. 0.00826429, not the widened
    binary value 0.008264290168881416)."""
    return float(str(np.float32(value)))


def _is_float32(dtype: Any) -> bool:
    """Whether a pandas column dtype (numpy, nullable or pyarrow-backed) holds float32 values."""
    arrow = getattr(dtype, "pyarrow_dtype", None)
    if arrow is not None:
        return pa.types.is_float32(arrow)
    try:
        return np.dtype(getattr(dtype, "numpy_dtype", dtype)) == np.float32
    except TypeError:
        return False


def _plain(value: Any, *, float32: bool = False) -> Any:
    """A JSON-like Python value of a HATS cell (``float32``: the cell's column is float32)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value]  # elements keep their numpy type (float32 included)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.float32) or (float32 and isinstance(value, (float, np.floating))):
        return _float32_value(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


class HatsRemoteProvider(_HTTPProvider):
    """Cone searches on a remote (or local) HATS catalog through ``lsdb`` (optional).

    ``lsdb.open_catalog(url, search_filter=lsdb.ConeSearch(ra, dec, radius_arcsec),
    columns=...)`` reads only the partitions overlapping the cone (LSDB docs,
    https://docs.lsdb.io). The rows are cut to the cone, ordered nearest first, limited to
    TOP ``row_limit + 1`` and converted with the registry definition, so the result has the
    same shape as the archive adapters' (units come from the definition, since HATS
    catalogs need not carry units/UCDs). Slow for cold remote reads (~10 s per cone for
    Gaia DR3 at data.lsdb.io, measured) -- a fallback, not a replacement for the store.
    """

    provider_name = "hats"

    def __init__(self, urls: Mapping[str, str] | None = None) -> None:
        self.urls = dict(urls or HATS_REMOTE_CATALOGS)
        self.client = None
        self.timeout = 120.0
        self.max_response_bytes = 0
        self.guards = {}
        self.cache = None

    def _fetch(self, url: str, columns: list[str] | None, ra: float, dec: float, radius_arcsec: float):
        import lsdb

        kwargs: dict[str, Any] = {"search_filter": lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=radius_arcsec)}
        if columns:
            kwargs["columns"] = columns
        catalog = lsdb.open_catalog(url, **kwargs)
        info = catalog.hc_structure.catalog_info
        return catalog.compute(), info.ra_column, info.dec_column

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        url = self.urls.get(catalog.name) or catalog.parameters.get("hats_url")
        if not url:
            raise CoverageError(f"{catalog.name}: no remote HATS catalog configured", catalog=catalog.name,
                                reason="no_hats_catalog")
        if not lsdb_available():
            raise AstroSearchError("HatsRemoteProvider needs the optional 'lsdb' package (pip install lsdb)")
        cone = plan_cone(catalog, target, radius_arcsec)
        wanted = [bare_column_name(str(c)) for c in (catalog.parameters.get("columns") or [])]
        frame, ra_col, dec_col = await asyncio.to_thread(self._fetch, url, wanted or None, cone.ra, cone.dec,
                                                         cone.radius_arcsec)
        float32_columns = {str(name) for name, dtype in frame.dtypes.items() if _is_float32(dtype)}
        records = frame.reset_index(drop=True).to_dict("records")
        rows = [{k: _plain(v, float32=k in float32_columns) for k, v in rec.items() if k != SPATIAL_INDEX_COLUMN}
                for rec in records]
        if rows:
            ra = np.array([float(r[ra_col]) for r in rows])
            dec = np.array([float(r[dec_col]) for r in rows])
            sep = angular_sep_deg(cone.ra, cone.dec, ra, dec)
            keep = [i for i in np.argsort(sep, kind="stable").tolist() if sep[i] <= cone.radius_arcsec / 3600.0]
            keep = keep[: max(1, int(cone.row_limit)) + 1]
            computed = _query_relative_values(catalog, catalog.provider, cone.ra, cone.dec, ra[keep], dec[keep])
            rows = [rows[i] for i in keep]
            for j, row in enumerate(rows):
                for name, values, _unit in computed:
                    row[name] = float(values[j])
        parameters = {"url": url, "ra": cone.ra, "dec": cone.dec, "radius_arcsec": cone.radius_arcsec}
        return self._sources(
            catalog, rows, radius_arcsec, url, parameters, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id=_default_id(catalog)),
            meta={"endpoint": url, "format": "hats-parquet", "cached": False,
                  "query": f"lsdb ConeSearch({cone.ra:.9f}, {cone.dec:.9f}, {cone.radius_arcsec:.6f}\")"},
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/skycache", tags=["skycache"])


class MirrorRequest(BaseModel):
    """Region to mirror: a cone (ra, dec, radius_deg) or HEALPix NESTED pixels at one order."""

    catalog: str = Field(..., min_length=1, max_length=100, description="Registry catalog name, e.g. gaia_dr3")
    ra: float | None = Field(default=None, ge=0.0, lt=360.0, description="Cone centre RA (ICRS deg)")
    dec: float | None = Field(default=None, ge=-90.0, le=90.0, description="Cone centre Dec (ICRS deg)")
    radius_deg: float | None = Field(default=None, gt=0.0, le=10.0, description="Cone radius (deg)")
    healpix_order: int | None = Field(default=None, ge=0, le=MAX_TILE_ORDER)
    healpix_pixels: list[int] | None = Field(default=None, min_length=1, max_length=10_000)
    tile_rows: int | None = Field(default=None, ge=1, le=100_000, description="Rows per archive query before splitting")

    @model_validator(mode="after")
    def _one_region(self) -> MirrorRequest:
        cone = self.ra is not None and self.dec is not None and self.radius_deg is not None
        pix = self.healpix_pixels is not None and self.healpix_order is not None
        if cone == pix:
            raise ValueError("give either ra, dec and radius_deg, or healpix_order and healpix_pixels")
        return self


def _store_for(request: Request) -> SkyCache:
    store = getattr(request.app.state, "skycache", None)
    if isinstance(store, SkyCache):
        return store
    return _default_store()


@functools.lru_cache(maxsize=4)
def _store_at(path: str) -> SkyCache:
    return SkyCache(path)


def _default_store() -> SkyCache:
    return _store_at(str(default_store_path()))


def _service_for(request: Request) -> Any:
    service = getattr(request.app.state, "service", None)
    if service is None:
        from main import build_service

        service = build_service(client=getattr(request.app.state, "client", None))
    return service


def _registry_for(request: Request, *, build: bool) -> Any:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        service = getattr(request.app.state, "service", None)
        registry = getattr(service, "registry", None)
    if registry is None and build:
        registry = CatalogRegistry(Settings().catalog_registry_path)
    return registry


@router.get("/status")
async def skycache_status(request: Request) -> dict[str, Any]:
    """Mirrored catalogs: rows, HATS partitions, coverage MOC (area, ASCII), mirror log and
    whether each catalog's registry definition still matches the mirrored one."""
    store = _store_for(request)
    return await asyncio.to_thread(store.status, _registry_for(request, build=False))


@router.post("/mirror")
async def skycache_mirror(request: Request, body: MirrorRequest) -> dict[str, Any]:
    """Mirror a region of one catalog into the local HATS store (synchronous).

    The mirror runs on a worker thread's event loop (:func:`mirror_region`), so other
    requests are served meanwhile. Archive requests are bounded like the service's own
    searches (its executor's ``timeout_cap``, i.e. ``Settings.catalog_timeout_cap_seconds``).
    """
    store = _store_for(request)
    service = _service_for(request)
    registry = getattr(request.app.state, "registry", None) or getattr(service, "registry", None)
    try:
        definition = registry.get(body.catalog)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Catalog '{body.catalog}' not found in registry") from exc
    try:
        report = await mirror_region(
            definition, ra=body.ra, dec=body.dec, radius_deg=body.radius_deg,
            healpix_order=body.healpix_order, healpix_pixels=body.healpix_pixels,
            store=store, providers=service.providers, tile_rows=body.tile_rows,
            timeout=getattr(getattr(service, "executor", None), "timeout_cap", None),
        )
    except (MirrorInputError, InvalidCoordinateError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MirrorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MirrorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return report.as_dict()


@router.get("/cone")
async def skycache_cone(
    request: Request,
    catalog: str = Query(..., min_length=1, max_length=100),
    ra: float = Query(..., ge=0.0, lt=360.0),
    dec: float = Query(..., ge=-90.0, le=90.0),
    radius_arcsec: float = Query(..., gt=0.0, le=3600.0),
) -> dict[str, Any]:
    """Local-only cone search returning CatalogSource dicts; 409 when not fully covered."""
    store = _store_for(request)
    registry = _registry_for(request, build=True)
    try:
        definition = registry.get(catalog)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Catalog '{catalog}' not found in registry") from exc
    try:
        # Parquet reads and row conversion run in a worker thread (never on the event loop).
        result = await asyncio.to_thread(LocalProvider(store).query_sync, definition, validate_target(ra, dec),
                                         radius_arcsec)
    except CoverageError as exc:
        raise HTTPException(status_code=409, detail=exc.as_dict()) from exc
    except (MirrorInputError, InvalidCoordinateError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _STORE_READ_ERRORS as exc:
        raise HTTPException(status_code=503, detail=f"the local store of {catalog!r} could not be read: "
                                                    f"{type(exc).__name__}: {exc}") from exc
    meta = {k: v for k, v in result.meta.items() if k not in {"pad_sources", "excess_sources"}}
    return _json_safe({"catalog": catalog, "count": len(result), "sources": [s.as_dict() for s in result],
                       "meta": meta})


@router.delete("/{catalog}")
async def skycache_delete(request: Request, catalog: str) -> dict[str, Any]:
    """Remove a mirrored catalog (its HATS directory) from the store."""
    store = _store_for(request)
    try:
        removed = await asyncio.to_thread(store.delete, catalog)
    except MirrorInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MirrorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"Catalog '{catalog}' is not mirrored")
    return {"catalog": catalog, "deleted": True}


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


async def _cli_mirror_async(args: argparse.Namespace) -> MirrorReport:
    settings = Settings()
    registry = CatalogRegistry(settings.catalog_registry_path)
    store = SkyCache(args.store) if args.store else SkyCache()
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        providers = provider_map(client, timeout=settings.request_timeout_seconds,
                                 max_response_bytes=settings.max_response_bytes, cache=_NoCache())
        return await mirror_region(
            args.catalog, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
            healpix_order=args.order, healpix_pixels=args.pixels, store=store, providers=providers,
            registry=registry, tile_rows=args.tile_rows,
            # Like the service's searches: registry timeouts, capped by $CATALOG_TIMEOUT_CAP_SECONDS
            # (or an explicit $REQUEST_TIMEOUT_SECONDS); --timeout overrides.
            timeout=args.timeout if getattr(args, "timeout", None) is not None else settings.catalog_timeout_cap_seconds,
        )


def cli_mirror(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch mirror``."""
    if args.pixels is None and (args.ra is None or args.dec is None or args.radius_deg is None):
        print("Error: give --ra, --dec and --radius-deg, or --order and --pixels.")
        return 2
    try:
        report = asyncio.run(_cli_mirror_async(args))
    except (MirrorError, InvalidCoordinateError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(f"Mirrored {report.catalog}: {report.rows_stored} rows from {report.queries} archive queries "
              f"({report.tiles_complete} complete tiles, {report.tiles_failed} failed) in {report.elapsed_s:.1f} s")
        print(f"Store {report.store}: {report.rows_total} rows, coverage {report.coverage_area_deg2:.4f} deg^2, "
              f"requested region {100.0 * report.region_covered_fraction:.1f}% covered")
        for warning in report.warnings:
            print(f"Warning: {warning}")
    return 0 if report.tiles_failed == 0 else 1


def cli_status(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch skycache status``."""
    store = SkyCache(args.store) if args.store else SkyCache()
    try:
        registry = CatalogRegistry(Settings().catalog_registry_path)
    except Exception as exc:  # noqa: BLE001 - status still works without the registry
        logger.warning("skycache status: registry unavailable (%s); definitions not checked", exc)
        registry = None
    status = store.status(registry)
    if args.json:
        print(json.dumps(status, indent=2, default=str))
        return 0
    print(f"Sky cache at {status['root']} ({len(status['catalogs'])} catalog(s))")
    for cat in status["catalogs"]:
        if "error" in cat:
            print(f"  {cat['catalog']:<20} UNREADABLE: {cat['error']}")
            continue
        note = "  DEFINITION CHANGED (not used; mirror again)" if cat.get("definition_current") is False else ""
        print(f"  {cat['catalog']:<20} {cat['rows']:>9} rows  {cat['partitions']:>4} partitions  "
              f"{cat['coverage_area_deg2']:.4f} deg^2  {cat['disk_bytes'] / 1e6:.2f} MB  updated {cat['updated_at']}{note}")
    for leftover in status.get("orphaned_swap_dirs") or []:
        print(f"  leftover {leftover['kind']} directory of {leftover['catalog']}: {leftover['path']}")
    return 0


def cli_delete(args: argparse.Namespace) -> int:
    store = SkyCache(args.store) if args.store else SkyCache()
    try:
        removed = store.delete(args.catalog)
    except MirrorInputError as exc:
        print(f"Error: {exc}")
        return 2
    except MirrorStoreError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Deleted {args.catalog}" if removed else f"{args.catalog} is not mirrored")
    return 0 if removed else 1


def cli_cone(args: argparse.Namespace) -> int:
    """Handler for ``astrosearch skycache cone``: exit 0, 1 (not covered, invalid input or an
    unreadable store) or 2 (unknown catalog)."""
    store = SkyCache(args.store) if args.store else SkyCache()
    registry = CatalogRegistry(Settings().catalog_registry_path)
    try:
        definition = registry.get(args.catalog)
    except KeyError:
        print(f"Error: unknown catalog {args.catalog!r}")
        return 2
    try:
        started = time.perf_counter()
        result = LocalProvider(store).query_sync(definition, validate_target(args.ra, args.dec), args.radius_arcsec)
        elapsed = (time.perf_counter() - started) * 1000.0
    except (CoverageError, InvalidCoordinateError) as exc:
        print(f"Error: {exc}")
        return 1
    except _STORE_READ_ERRORS as exc:
        print(f"Error: the local store could not be read: {type(exc).__name__}: {exc}")
        return 1
    if args.json:
        print(json.dumps(_json_safe([s.as_dict() for s in result]), indent=2))
        return 0
    print(f"{len(result)} {args.catalog} source(s) within {args.radius_arcsec:g}\" ({elapsed:.1f} ms, local)")
    for src in result:
        sep = src.metadata.get("query_separation_arcsec") or 0.0
        print(f"  {src.source_id:<28} RA={src.ra:.7f} Dec={src.dec:+.7f} sep={sep:.3f}\"")
    return 0


def register_cli(subparsers: Any) -> None:
    """Add ``mirror`` and ``skycache {status,delete,cone}`` subcommands."""
    mirror = subparsers.add_parser("mirror", help="Mirror a sky region of a catalog into the local HATS sky cache")
    mirror.add_argument("--catalog", required=True, help="Registry catalog name (e.g. gaia_dr3, twomass_psc)")
    mirror.add_argument("--ra", type=float, help="Cone centre RA (ICRS deg)")
    mirror.add_argument("--dec", type=float, help="Cone centre Dec (ICRS deg)")
    mirror.add_argument("--radius-deg", dest="radius_deg", type=float, help="Cone radius (deg)")
    mirror.add_argument("--order", type=int, help="HEALPix order of --pixels")
    mirror.add_argument("--pixels", type=int, nargs="+", help="HEALPix NESTED pixels to mirror")
    mirror.add_argument("--tile-rows", dest="tile_rows", type=int, help="Rows per archive query before splitting")
    mirror.add_argument("--timeout", type=float,
                        help="Seconds allowed per archive request (default: the catalog's registry timeout, capped "
                             "by $CATALOG_TIMEOUT_CAP_SECONDS)")
    mirror.add_argument("--store", help=f"Store directory (default ${ENV_PATH} or ~/.astrosearch/skycache)")
    mirror.add_argument("--json", action="store_true", help="Print the JSON report")
    mirror.set_defaults(handler=cli_mirror)

    sky = subparsers.add_parser("skycache", help="Inspect or query the local sky cache")
    actions = sky.add_subparsers(dest="skycache_command")
    status = actions.add_parser("status", help="List mirrored catalogs and coverage")
    status.add_argument("--store")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=cli_status)
    delete = actions.add_parser("delete", help="Delete a mirrored catalog")
    delete.add_argument("--catalog", required=True)
    delete.add_argument("--store")
    delete.set_defaults(handler=cli_delete)
    cone = actions.add_parser("cone", help="Local cone search (fails when not fully mirrored)")
    cone.add_argument("--catalog", required=True)
    cone.add_argument("--ra", type=float, required=True)
    cone.add_argument("--dec", type=float, required=True)
    cone.add_argument("--radius-arcsec", dest="radius_arcsec", type=float, required=True)
    cone.add_argument("--store")
    cone.add_argument("--json", action="store_true")
    cone.set_defaults(handler=cli_cone)
    sky.set_defaults(handler=cli_status, store=None, json=False)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(prog="skycache")
    register_cli(_parser.add_subparsers(dest="command"))
    _args = _parser.parse_args()
    raise SystemExit(_args.handler(_args) if hasattr(_args, "handler") else 2)
