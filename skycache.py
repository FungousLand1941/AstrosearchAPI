"""Local sky mirror ("skycache"): HATS-partitioned Parquet copies of archive regions for
millisecond cone searches that return exactly what the archive would.

What it does
------------
* :func:`mirror_region` downloads every row of a catalog inside a sky region (a cone or a
  list of HEALPix pixels) through the SAME provider adapters the crossmatch service uses
  (``providers.TapProvider`` & co, with the registry's fallback archives), dedupes the rows
  by source id and stores them, with per-column units/UCDs, in a local HATS catalog.
* The region that is known to be *complete* is recorded as a MOC (IVOA Multi-Order
  Coverage map, MOC 2.0, Fernique et al. 2022, https://www.ivoa.net/documents/MOC/)
  held as sorted disjoint ranges of order-29 NESTED HEALPix indices.
* :class:`LocalProvider` is a :class:`providers.CatalogProvider` that answers a cone
  from the store only when the cone (after the same epoch widening the remote provider
  applies, :func:`models.plan_cone`) lies entirely inside the MOC; otherwise it raises
  :class:`CoverageError` so the caller can fall back to the archive
  (:class:`SkyCacheProvider` / :func:`wrap_providers` do that automatically).
  Rows are converted by the providers' own ``_HTTPProvider._sources`` so the
  :class:`models.CatalogSource` objects are identical in shape to the remote ones (same
  canonical fields, epochs, positional errors, metadata, nearest-first TOP N semantics).

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
    <root>/<catalog>/skycache.json            (ours) coverage MOC, column units/UCDs, mirror log

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

Remote HATS catalogs (optional)
-------------------------------
:class:`HatsRemoteProvider` answers cones from a public HATS catalog through ``lsdb``
(``pip install lsdb``; e.g. https://data.lsdb.io/hats/gaia_dr3/gaia) and converts the rows
with the registry definition, so it too returns ordinary CatalogSource objects.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
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
    validate_target,
)
from providers import CacheManager, CatalogProvider, QueryResult, _HTTPProvider, provider_map

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
    "cone_pixels",
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

# Environment configuration.
ENV_PATH = "SKYCACHE_PATH"
ENV_TILE_ROWS = "SKYCACHE_TILE_ROWS"
ENV_PARTITION_ROWS = "SKYCACHE_PARTITION_ROWS"
ENV_MAX_RADIUS = "SKYCACHE_MAX_RADIUS_DEG"
ENV_MAX_QUERIES = "SKYCACHE_MAX_QUERIES"
DEFAULT_TILE_ROWS = 10_000
DEFAULT_PARTITION_ROWS = 200_000
DEFAULT_MAX_RADIUS_DEG = 1.0
DEFAULT_MAX_QUERIES = 256
DEFAULT_CONCURRENCY = 4
MAX_PARTITION_ORDER = 20


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


def pixels_inside_cone(ra: float, dec: float, radius_deg: float, order: int) -> np.ndarray:
    """Order-``order`` pixels lying entirely inside the cone (conservative; sorted int64).

    A pixel is inside when every sampled boundary point is within ``radius - 2% res`` of the
    centre (the farthest point of a region from any point lies on its boundary).
    """
    candidates = cone_pixels(ra, dec, radius_deg, order)
    if candidates.size == 0:
        return candidates
    inside = np.zeros(candidates.size, dtype=bool)
    margin = PIXEL_MARGIN * pixel_resolution_deg(order)
    for start in range(0, candidates.size, 20_000):
        chunk = candidates[start:start + 20_000]
        lon, lat = _boundary_lonlat(order, chunk)
        far = angular_sep_deg(ra, dec, lon.ravel(), lat.ravel()).reshape(lon.shape).max(axis=1)
        inside[start:start + chunk.size] = far + margin <= radius_deg
    return candidates[inside]


def pixel_center(order: int, pixel: int) -> tuple[float, float]:
    lon, lat = _healpix(order).healpix_to_lonlat(np.array([pixel], dtype=np.int64))
    return float(lon.to_value(u.deg)[0]) % 360.0, float(lat.to_value(u.deg)[0])


def pixel_circumradius_deg(order: int, pixel: int) -> float:
    """Radius of a circle around the pixel centre that encloses the whole pixel (+2% res)."""
    ra, dec = pixel_center(order, pixel)
    lon, lat = _boundary_lonlat(order, np.array([pixel], dtype=np.int64))
    far = float(angular_sep_deg(ra, dec, lon.ravel(), lat.ravel()).max())
    return far + PIXEL_MARGIN * pixel_resolution_deg(order)


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
        return cls.from_pixels(order, pixels_inside_cone(ra, dec, radius_deg, order))

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


def _row_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _is_query_relative(name: str, provider: str) -> bool:
    """Columns computed by the archive relative to the query centre (never stored)."""
    lowered = name.lower()
    return lowered in COMPUTED_COLUMNS or (provider == "irsa_gator" and lowered == "angle")


# ---------------------------------------------------------------------------
# Stored rows and catalog state
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

    def key(self) -> tuple[str, str]:
        return self.source_id, _row_hash(self.data)


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


_CATALOG_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}$")


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

    # -- paths ---------------------------------------------------------------

    def catalog_path(self, catalog: str) -> Path:
        if not _CATALOG_NAME.match(catalog or ""):
            raise MirrorInputError(f"invalid catalog name {catalog!r}")
        return self.root / catalog

    def catalogs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "skycache.json").is_file())

    def has(self, catalog: str) -> bool:
        try:
            return (self.catalog_path(catalog) / "skycache.json").is_file()
        except MirrorInputError:
            return False

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
            state = _CatalogState(catalog, path, meta, Moc.from_json(meta.get("coverage")), stamp, parts, ranges,
                                  colsets=colsets, colset_meta=colset_meta,
                                  json_columns=frozenset(meta.get("json_columns", [])))
            cov_order = (meta.get("coverage") or {}).get("max_order")
            state.moc_max_order = int(cov_order) if cov_order is not None else state.moc.max_order
            self._states[catalog] = state
            return state

    def _partition(self, state: _CatalogState, order: int, pixel: int) -> _Partition:
        key = (order, pixel)
        part = state.loaded.get(key)
        if part is None:
            path = state.path / "dataset" / f"Norder={order}" / f"Dir={(pixel // DIR_DIVISOR) * DIR_DIVISOR}" / f"Npix={pixel}.parquet"
            table = pq.read_table(path)
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
        order = self._query_order(radius_deg, state.moc_max_order)
        pixels = cone_pixels(ra, dec, radius_deg, order)
        inside = state.moc.contains_pixels(order, pixels) if not state.moc.empty else np.zeros(pixels.size, bool)
        fraction = float(inside.mean()) if inside.size else 0.0
        return CoverageReport(bool(inside.size and inside.all()), fraction, order, int(pixels.size)), pixels

    def covers(self, catalog: str, ra: float, dec: float, radius_arcsec: float) -> CoverageReport:
        """Whether every pixel overlapping the cone lies inside the recorded coverage.

        The cone's pixels come from an inclusive cone search (:func:`cone_pixels`) and each
        must be contained in the MOC, so a True answer is never optimistic.
        """
        state = self._state(catalog)
        if state is None or state.moc.empty:
            return CoverageReport(False, 0.0, 0, 0)
        return self._cone_coverage(state, ra, dec, float(radius_arcsec) / 3600.0)[0]

    def require_coverage(self, catalog: str, ra: float, dec: float, radius_arcsec: float) -> CoverageReport:
        """Raise :class:`CoverageError` unless the cone is fully covered."""
        return self._require(catalog, ra, dec, radius_arcsec)[1]

    def _require(self, catalog: str, ra: float, dec: float,
                 radius_arcsec: float) -> tuple[_CatalogState, CoverageReport, np.ndarray]:
        state = self._state(catalog)
        if state is None:
            raise CoverageError(f"{catalog}: not mirrored in the local sky cache ({self.root})", catalog=catalog,
                                ra=ra, dec=dec, radius_arcsec=radius_arcsec, reason="not_mirrored")
        report, pixels = self._cone_coverage(state, ra, dec, float(radius_arcsec) / 3600.0)
        if not report.covered:
            raise CoverageError(
                f"{catalog}: cone RA={ra:.6f} Dec={dec:+.6f} r={radius_arcsec:g}\" is only "
                f"{100.0 * report.covered_fraction:.1f}% inside the mirrored coverage",
                catalog=catalog, ra=ra, dec=dec, radius_arcsec=radius_arcsec,
                covered_fraction=report.covered_fraction, order=report.order,
                reason="partially_covered" if report.covered_fraction > 0 else "not_covered")
        return state, report, pixels

    def cone_search(self, catalog: str, ra: float, dec: float, radius_arcsec: float, *,
                    limit: int | None = None, require_coverage: bool = True) -> ConeResult:
        """Rows within ``radius_arcsec`` of (ra, dec), nearest first (ties by source id).

        Candidate rows come from the partitions and ``_healpix_29`` ranges of the pixels
        overlapping the cone (vectorized ``searchsorted`` on each partition's sorted
        spatial index); the exact cut is a numpy haversine separation <= radius. With
        ``require_coverage`` a :class:`CoverageError` is raised unless the cone is fully
        inside the mirrored coverage.
        """
        started = time.perf_counter()
        radius_deg = float(radius_arcsec) / 3600.0
        if not (math.isfinite(radius_deg) and radius_deg > 0):
            raise InvalidCoordinateError("radius_arcsec must be finite and > 0")
        if require_coverage:
            state, report, pixels = self._require(catalog, ra, dec, radius_arcsec)
        else:
            maybe = self._state(catalog)
            if maybe is None:
                raise CoverageError(f"{catalog}: not mirrored", catalog=catalog, reason="not_mirrored")
            state = maybe
            report, pixels = self._cone_coverage(state, ra, dec, radius_deg)
        cone = Moc.from_pixels(report.order, pixels).ranges
        read = 0
        chunks: list[tuple[_Partition, np.ndarray]] = []
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
                    chunks.append((part, idx[keep]))
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        ras: list[float] = []
        decs: list[float] = []
        colsets: list[int] = []
        for part, idx in chunks:
            records = part.table.take(pa.array(idx)).to_pylist()
            for rec in records:
                colset = int(rec[SC_COLSET])
                rows.append(self._raw_row(state, rec, colset))
                ids.append(str(rec[SC_ID]))
                ras.append(float(rec[SC_RA]))
                decs.append(float(rec[SC_DEC]))
                colsets.append(colset)
        ra_arr = np.asarray(ras, dtype=np.float64)
        dec_arr = np.asarray(decs, dtype=np.float64)
        sep_arcsec = angular_sep_deg(ra, dec, ra_arr, dec_arr) * 3600.0 if ras else np.zeros(0)
        order_idx = sorted(range(len(rows)), key=lambda i: (float(sep_arcsec[i]), ids[i]))
        if limit is not None:
            order_idx = order_idx[: max(0, int(limit))]
        return ConeResult(
            rows=[rows[i] for i in order_idx], source_ids=[ids[i] for i in order_idx],
            ra=ra_arr[order_idx] if ras else ra_arr, dec=dec_arr[order_idx] if ras else dec_arr,
            separation_arcsec=sep_arcsec[order_idx] if ras else sep_arcsec,
            colsets=[colsets[i] for i in order_idx],
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3), partitions_read=read,
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

    def column_meta(self, catalog: str, colsets: Iterable[int] | None = None) -> list[ColumnMeta] | None:
        """Archive column metadata (units/UCDs) of the given column sets, merged by name."""
        state = self._state(catalog)
        if state is None or not state.colset_meta:
            return None
        chosen = sorted(set(colsets)) if colsets is not None else range(len(state.colset_meta))
        merged: dict[str, ColumnMeta] = {}
        for index in chosen:
            if 0 <= index < len(state.colset_meta):
                for col in state.colset_meta[index]:
                    merged.setdefault(col.name, col)
        return [ColumnMeta(**asdict(c)) for c in merged.values()] or None

    def load_rows(self, catalog: str) -> list[StoredRow]:
        """Every stored row (used when merging a new mirror into the store)."""
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

    # -- writing -------------------------------------------------------------

    def write(self, catalog: CatalogDefinition, rows: Sequence[StoredRow], coverage: Moc, *,
              mirrors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """(Re)write the whole HATS catalog atomically; returns the new skycache.json."""
        path = self.catalog_path(catalog.name)
        with _catalog_lock(path):
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{catalog.name}.tmp-{uuid.uuid4().hex[:8]}"
            try:
                meta = self._write_tree(tmp, catalog, rows, coverage, mirrors or [])
                backup = None
                if path.exists():
                    backup = self.root / f".{catalog.name}.old-{uuid.uuid4().hex[:8]}"
                    os.replace(path, backup)
                os.replace(tmp, path)
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
            with self._lock:
                self._states.pop(catalog.name, None)
            return meta

    def _write_tree(self, base: Path, catalog: CatalogDefinition, rows: Sequence[StoredRow], coverage: Moc,
                    mirrors: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = base / "dataset"
        dataset.mkdir(parents=True)
        # Column sets: the ordered raw keys of each row; metadata per set.
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
                    if key in SYSTEM_COLUMNS:
                        raise MirrorError(f"{catalog.name}: archive column {key!r} collides with a skycache column")
                    seen.add(key)
                    raw_names.append(key)
        col_meta: dict[str, ColumnMeta] = {}
        for cols in colset_columns:
            for c in cols:
                col_meta.setdefault(c["name"], ColumnMeta(**c))

        n = len(rows)
        ra = np.array([r.ra for r in rows], dtype=np.float64)
        dec = np.array([r.dec for r in rows], dtype=np.float64)
        h29 = healpix29(ra, dec)
        order = np.argsort(h29, kind="stable")

        def pick(values: list[Any]) -> list[Any]:
            return [values[i] for i in order.tolist()]

        def opt_float(values: list[Any]) -> pa.Array:
            return pa.array([None if v is None else float(v) for v in pick(values)], type=pa.float64())

        arrays: list[pa.Array] = [
            pa.array(h29[order], type=pa.int64()),
            pa.array(pick([r.source_id for r in rows]), type=pa.string()),
            pa.array(ra[order], type=pa.float64()),
            pa.array(dec[order], type=pa.float64()),
            opt_float([r.epoch for r in rows]),
            opt_float([r.pmra for r in rows]),
            opt_float([r.pmdec for r in rows]),
            opt_float([r.pos_err_arcsec for r in rows]),
            pa.array(pick([r.provider for r in rows]), type=pa.string()),
            pa.array(pick([r.retrieved_at for r in rows]), type=pa.string()),
            pa.array(pick(row_colset), type=pa.int32()),
        ]
        fields = [pa.field(name, arr.type, metadata=_field_metadata(ColumnMeta(name, *_SYSTEM_META[name][:2],
                                                                               description=_SYSTEM_META[name][2])))
                  for name, arr in zip(SYSTEM_COLUMNS, arrays)]
        json_columns: list[str] = []
        for name in raw_names:
            values = pick([r.data.get(name) for r in rows])
            arr, encoded = _column_array(values)
            if encoded:
                json_columns.append(name)
            arrays.append(arr)
            meta = col_meta.get(name)
            fmeta = _field_metadata(meta) or {}
            if encoded:
                fmeta[b"skycache_encoding"] = b"json"
            fields.append(pa.field(name, arr.type, metadata=fmeta or None))
        schema = pa.schema(fields, metadata={
            b"astrosearch_skycache": json.dumps({"catalog": catalog.name, "format_version": FORMAT_VERSION,
                                                 "archive_endpoint": catalog.endpoint}).encode("utf-8"),
        })
        table = pa.Table.from_arrays(arrays, schema=schema)

        # Adaptive partitions; rows of one partition are contiguous in _healpix_29 order.
        porders, ppixels = partition_orders(h29[order], self.partition_rows)
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

        meta = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "catalog": catalog.name,
            "version": uuid.uuid4().hex,
            "updated_at": now.isoformat(),
            "provider": catalog.provider,
            "archive_endpoint": catalog.endpoint,
            "citation": catalog.citation,
            "acknowledgement": catalog.acknowledgement,
            "definition": _json_safe(catalog.as_dict()),
            "rows": n,
            "partitions": partitions,
            "partition_rows": self.partition_rows,
            "colsets": colsets,
            "colset_columns": colset_columns,
            "json_columns": json_columns,
            "coverage": {**coverage.as_json(), "moc_ascii": coverage.to_ascii(), "max_order": coverage.max_order},
            "mirrors": mirrors,
            "hats_properties": properties,
        }
        (base / "skycache.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return meta

    def delete(self, catalog: str) -> bool:
        path = self.catalog_path(catalog)
        with _catalog_lock(path):
            with self._lock:
                self._states.pop(catalog, None)
            if not path.exists():
                return False
            shutil.rmtree(path)
            return True

    def status(self) -> dict[str, Any]:
        catalogs = []
        for name in self.catalogs():
            meta = self.metadata(name) or {}
            path = self.catalog_path(name)
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            cov = meta.get("coverage") or {}
            ascii_moc = cov.get("moc_ascii") or ""
            catalogs.append({
                "catalog": name,
                "path": str(path),
                "rows": meta.get("rows", 0),
                "partitions": len(meta.get("partitions") or []),
                "hats_order": (meta.get("hats_properties") or {}).get("hats_order"),
                "coverage_area_deg2": cov.get("area_deg2", 0.0),
                "coverage_sky_fraction": cov.get("sky_fraction", 0.0),
                "coverage_max_order": cov.get("max_order"),
                "coverage_moc_ascii": ascii_moc if len(ascii_moc) <= 4000 else ascii_moc[:4000] + " ...",
                "mirrors": len(meta.get("mirrors") or []),
                "last_mirror": (meta.get("mirrors") or [None])[-1],
                "updated_at": meta.get("updated_at"),
                "disk_bytes": size,
                "provider": meta.get("provider"),
                "archive_endpoint": meta.get("archive_endpoint"),
                "citation": meta.get("citation"),
            })
        return {"root": str(self.root), "format": FORMAT_NAME, "format_version": FORMAT_VERSION,
                "hats_compatible": True, "catalogs": catalogs}


def _escape_property(value: str) -> str:
    """Java .properties value escaping as jproperties/hats write it (':' and '=' escaped)."""
    return str(value).replace("\\", "\\\\").replace("\n", " ").replace(":", "\\:").replace("=", "\\=")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Local provider
# ---------------------------------------------------------------------------


def _query_relative_values(catalog: CatalogDefinition, provider: str, ra0: float, dec0: float,
                           ra: np.ndarray, dec: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    """Columns the archive computes relative to the query centre, as its adapter returns them.

    TAP: ``match_dist`` = ADQL DISTANCE() in degrees, or arcsec where the registry says the
    service returns arcsec (IRSA); Gator: ``dist`` (arcsec) and ``angle`` (degrees E of N,
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
        return [("match_dist", sep * 3600.0 if mode == "arcsec" else sep, "arcsec" if mode == "arcsec" else "deg")]
    if provider == "irsa_gator":
        return [("dist", sep * 3600.0, "arcsec"), ("angle", position_angle_deg(ra0, dec0, ra, dec), "deg")]
    if provider == "mast":
        return [("distance", sep, "deg")] if params.get("columns") else []
    if provider == "sdss":
        endpoint = str(catalog.endpoint or "")
        mode = str(params.get("mode") or ("conesearch" if "ConeSearch" in endpoint else "sqlsearch"))
        return [("dist_arcsec", sep * 3600.0, "arcsec")] if mode != "conesearch" else []
    return []


class LocalProvider(_HTTPProvider):
    """Answers cone queries from a :class:`SkyCache`, only for fully covered cones.

    The cone is planned exactly like the remote adapters do (:func:`models.plan_cone`:
    epoch widening, static pads, ``row_limit``), rows inside it are taken nearest first
    (``row_limit + 1`` -- the archives' TOP N+1 probe), the query-relative distance
    columns of the catalog's own adapter are recomputed, and the rows go through
    ``_HTTPProvider._sources`` -- the same conversion the remote adapter uses -- so every
    canonical field, epoch, positional error, pad/excess split and truncation flag is the
    same. Raises :class:`CoverageError` when the store does not fully cover the cone.
    """

    provider_name = "skycache"

    def __init__(self, store: SkyCache | None = None) -> None:
        self.store = store or SkyCache()
        self.client = None
        self.timeout = 30.0
        self.max_response_bytes = 0
        self.guards = {}
        self.cache = None

    def query_sync(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        started = time.perf_counter()
        cone = plan_cone(catalog, target, radius_arcsec)
        result = self.store.cone_search(catalog.name, cone.ra, cone.dec, cone.radius_arcsec,
                                        limit=max(1, int(cone.row_limit)) + 1)
        provider = catalog.provider
        computed = _query_relative_values(catalog, provider, cone.ra, cone.dec, result.ra, result.dec)
        rows: list[dict[str, Any]] = []
        for i, raw in enumerate(result.rows):
            row = {k: v for k, v in raw.items() if not _is_query_relative(k, provider)}
            for name, values, _unit in computed:
                row[name] = float(values[i])
            rows.append(row)
        columns = self.store.column_meta(catalog.name, set(result.colsets) if result.colsets else None)
        if columns is not None:
            present = {c.name for c in columns}
            for name, _values, unit in computed:
                if name not in present:
                    columns.append(ColumnMeta(name=name, unit=unit, datatype="double"))
        meta_info = self.store.metadata(catalog.name) or {}
        location = self.store.catalog_path(catalog.name).resolve().as_uri()
        parameters = {"ra": cone.ra, "dec": cone.dec, "radius_arcsec": cone.radius_arcsec,
                      "row_limit": cone.row_limit, "store": location,
                      "mirrored_from": meta_info.get("archive_endpoint") or catalog.endpoint}
        query_text = (f"skycache cone ({cone.ra:.9f}, {cone.dec:.9f}) r={cone.radius_arcsec:.6f}\" "
                      f"TOP {cone.row_limit + 1} ORDER BY distance")
        meta: dict[str, Any] = {"endpoint": location, "format": "hats-parquet", "cached": False}
        if provider in {"tap", "sdss"}:
            meta["query"] = query_text
        if provider == "tap":
            meta.update({"http_method": None, "truncated": False, "query_status": None})
        if provider == "sdss":
            meta["mode"] = catalog.parameters.get("mode") or "sqlsearch"
        if columns is not None:
            meta["columns"] = [c.as_dict() for c in columns]
        default_id = _DEFAULT_ID.get(provider)
        out = self._sources(
            catalog, rows, radius_arcsec, location, parameters,
            columns=columns, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id=default_id),
            meta=meta,
        )
        out.meta["skycache"] = {
            "hit": True,
            "store": location,
            "catalog_version": meta_info.get("version"),
            "mirrored_from": meta_info.get("archive_endpoint"),
            "rows_scanned": len(result.rows),
            "partitions_read": result.partitions_read,
            "lookup_ms": result.elapsed_ms,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return out

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        return self.query_sync(catalog, target, radius_arcsec)


class SkyCacheProvider(CatalogProvider):
    """Local-first provider: the sky cache when it covers the cone, else the remote adapter."""

    def __init__(self, remote: CatalogProvider, local: LocalProvider) -> None:
        self.remote = remote
        self.local = local

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        try:
            return await self.local.query(catalog, target, radius_arcsec)
        except CoverageError as exc:
            result = await self.remote.query(catalog, target, radius_arcsec)
            meta = getattr(result, "meta", None)
            if isinstance(meta, dict):
                meta["skycache"] = {"hit": False, "reason": exc.reason, "covered_fraction": exc.covered_fraction}
            return result


def wrap_providers(providers: Mapping[str, CatalogProvider], store: SkyCache | None = None) -> dict[str, CatalogProvider]:
    """Provider map whose adapters answer from the sky cache first (drop-in for CrossmatchService)."""
    local = LocalProvider(store)
    return {name: SkyCacheProvider(provider, local) for name, provider in providers.items()}


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
    rows_fetched: int = 0
    rows_stored: int = 0
    rows_replaced: int = 0
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
    ok: bool
    truncated: bool
    sources: list[CatalogSource]
    columns: list[ColumnMeta] | None
    provider: str
    error: str | None = None


def _resolve_catalog(catalog: str | CatalogDefinition, registry: Any | None) -> CatalogDefinition:
    if isinstance(catalog, CatalogDefinition):
        return catalog
    reg = registry or CatalogRegistry(Settings().catalog_registry_path)
    try:
        definition = reg.get(catalog)
    except KeyError as exc:
        raise MirrorInputError(f"unknown catalog {catalog!r}") from exc
    return definition


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
    timeout: float = 30.0,
) -> MirrorReport:
    """Mirror every row of ``catalog`` in a cone (ra, dec, radius_deg) or HEALPix pixels.

    Each archive request goes through the service's own adapters (and the registry's
    fallback archive) via :class:`crossmatch.QueryExecutor`, for a target without epoch
    (an exact cone) and ``max_rows = tile_rows``. A request is *complete* when the
    adapter reports no archive truncation (fewer than TOP N+1 rows and no OVERFLOW); an
    incomplete one is split: a cone into the HEALPix pixels (order with resolution <= r/2)
    overlapping it, a pixel into its 4 children (up to order 18). Pixels are fetched with a
    cone around the pixel centre that encloses the whole pixel. The coverage MOC gains the
    complete pixels, or for a complete root cone the order-C pixels fully inside it.

    Rows are merged into the store: previously stored rows inside the newly covered area
    are replaced (the archive is authoritative there), rows are deduplicated by
    (source id, content) and a source id fetched again replaces the older copy.
    """
    started = time.perf_counter()
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

    roots: list[tuple[str, int, int, float, float, float]] = []  # (kind, order, pixel, ra, dec, radius_deg)
    if healpix_pixels is not None:
        if healpix_order is None or not 0 <= int(healpix_order) <= MAX_TILE_ORDER:
            raise MirrorInputError(f"healpix_order must be in 0..{MAX_TILE_ORDER}")
        order = int(healpix_order)
        pixels = sorted({int(p) for p in healpix_pixels})
        if not pixels or any(p < 0 or p >= 12 * 4**order for p in pixels):
            raise MirrorInputError(f"healpix_pixels must be non-empty and within 0..{12 * 4**order - 1}")
        area = len(pixels) * pixel_resolution_deg(order) ** 2
        if area > math.pi * limit_deg**2:
            raise MirrorInputError(f"region of {area:.3f} deg^2 exceeds the {math.pi * limit_deg**2:.3f} deg^2 "
                                   f"limit (${ENV_MAX_RADIUS}={limit_deg:g} deg)")
        for p in pixels:
            c_ra, c_dec = pixel_center(order, p)
            roots.append(("pixel", order, p, c_ra, c_dec, pixel_circumradius_deg(order, p)))
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
        requested = Moc.from_cone(target.ra, target.dec, radius, cov_order)

    report = MirrorReport(definition.name, region, store=str(cache.root))
    fetch_def = replace(definition, max_rows=tile_rows)
    retrieved_at = datetime.now(UTC).isoformat()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    collected: list[StoredRow] = []
    new_cov: list[Moc] = []
    providers_used: set[str] = set()
    budget_exhausted = False

    own_client = None
    if providers is None:
        if client is None:
            own_client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
            client = own_client
        providers = provider_map(client, timeout=timeout, cache=CacheManager(None))
    executor = QueryExecutor(dict(providers), timeout=timeout, registry=_OneCatalogRegistry(fetch_def))

    async def fetch(c_ra: float, c_dec: float, radius_deg_: float) -> _Fetch | None:
        nonlocal budget_exhausted
        if report.queries >= max_queries:
            budget_exhausted = True
            return None
        report.queries += 1
        plan = QueryPlan(fetch_def.name, fetch_def.provider, fetch_def.endpoint, {}, radius_deg_ * 3600.0,
                         fetch_def.wavelength)
        async with semaphore:
            successes, failures = await executor.execute([plan], validate_target(c_ra, c_dec))
        if failures:
            failure = failures[0]
            message = f"{failure.error_type}: {failure.message}"
            # The adapter's response-size guard fired: the tile is simply too dense.
            too_big = "byte limit" in (failure.message or "")
            return _Fetch(False, too_big, [], None, fetch_def.provider, None if too_big else message)
        result = successes[0][1]
        meta = getattr(result, "meta", {}) or {}
        fallback = meta.get("fallback")
        used = str((fallback or {}).get("provider") or fetch_def.provider)
        sources = list(result) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
        columns = [ColumnMeta(**c) for c in meta.get("columns") or []] or None
        return _Fetch(True, bool(meta.get("archive_truncated")), sources, columns, used)

    def accept(res: _Fetch, coverage: Moc) -> None:
        providers_used.add(res.provider)
        report.rows_fetched += len(res.sources)
        report.tiles_complete += 1
        new_cov.append(coverage)
        for src in res.sources:
            collected.append(StoredRow.from_source(src, provider=res.provider, retrieved_at=retrieved_at,
                                                   columns=res.columns))

    async def tile(order: int, pixel: int) -> None:
        c_ra, c_dec = pixel_center(order, pixel)
        res = await fetch(c_ra, c_dec, pixel_circumradius_deg(order, pixel))
        if res is None:
            report.tiles_failed += 1
            return
        if res.ok and not res.truncated:
            accept(res, Moc.from_pixels(order, [pixel]))
            return
        if res.truncated and order < MAX_TILE_ORDER:
            children = [pixel * 4 + i for i in range(4)]
            await asyncio.gather(*(tile(order + 1, c) for c in children))
            return
        report.tiles_failed += 1
        report.warnings.append(
            f"order-{order} pixel {pixel}: " + (res.error or f"still truncated at order {MAX_TILE_ORDER}"))

    async def root_cone(c_ra: float, c_dec: float, radius: float) -> None:
        res = await fetch(c_ra, c_dec, radius)
        if res is None:
            report.tiles_failed += 1
            return
        if res.ok and not res.truncated:
            cov_order = order_for_resolution(radius * COVERAGE_RES_FRACTION, hi=MAX_COVERAGE_ORDER)
            accept(res, Moc.from_cone(c_ra, c_dec, radius, cov_order))
            return
        if not res.truncated:
            report.tiles_failed += 1
            report.warnings.append(f"cone: {res.error}")
            return
        order = order_for_resolution(radius / 2.0, hi=MAX_TILE_ORDER)
        tiles = cone_pixels(c_ra, c_dec, radius, order)
        report.warnings.append(f"cone held more than {tile_rows} rows; split into {tiles.size} order-{order} tiles")
        await asyncio.gather(*(tile(order, int(p)) for p in tiles))

    try:
        await asyncio.gather(*(
            root_cone(r_ra, r_dec, r_rad) if kind == "cone" else tile(order, pixel)
            for kind, order, pixel, r_ra, r_dec, r_rad in roots
        ))
    finally:
        if own_client is not None:
            await own_client.aclose()
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
        "providers": sorted(providers_used), "queries": report.queries, "tiles_complete": report.tiles_complete,
        "tiles_failed": report.tiles_failed, "rows_fetched": report.rows_fetched, "tile_rows": tile_rows,
        "coverage_added_deg2": added.area_deg2, "warnings": list(report.warnings),
    }
    meta = await asyncio.to_thread(_merge_and_write, cache, definition, collected, added, record, report)
    report.rows_total = int(meta["rows"])
    total = Moc.from_json(meta["coverage"])
    report.coverage_area_deg2 = total.area_deg2
    wanted = requested.cells29
    report.region_covered_fraction = (total & requested).cells29 / wanted if wanted else 0.0
    report.providers_used = sorted(providers_used)
    report.elapsed_s = round(time.perf_counter() - started, 3)
    logger.info("skycache mirror %s: %d queries, %d rows fetched, %d stored, total %d",
                definition.name, report.queries, report.rows_fetched, report.rows_stored, report.rows_total)
    return report


def _merge_and_write(cache: SkyCache, definition: CatalogDefinition, collected: list[StoredRow], added: Moc,
                     record: dict[str, Any], report: MirrorReport) -> dict[str, Any]:
    path = cache.catalog_path(definition.name)
    with _catalog_lock(path):
        old_meta = cache.metadata(definition.name) or {}
        old_rows = cache.load_rows(definition.name)
        old_cov = Moc.from_json(old_meta.get("coverage"))
        new_rows: list[StoredRow] = []
        seen: set[tuple[str, str]] = set()
        for row in collected:
            key = row.key()
            if key not in seen:
                seen.add(key)
                new_rows.append(row)
        new_ids = {row.source_id for row in new_rows}
        kept: list[StoredRow] = []
        if old_rows:
            in_new = added.contains_points([r.ra for r in old_rows], [r.dec for r in old_rows])
            for row, inside in zip(old_rows, in_new.tolist()):
                if inside or row.source_id in new_ids:
                    continue
                kept.append(row)
        report.rows_replaced = len(old_rows) - len(kept)
        report.rows_stored = len(new_rows)
        mirrors = list(old_meta.get("mirrors") or []) + [record]
        return cache.write(definition, kept + new_rows, old_cov | added, mirrors=mirrors)


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


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if value is pd.NA or value is pd.NaT:
        return None
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
        records = frame.reset_index(drop=True).to_dict("records")
        rows = [{k: _plain(v) for k, v in rec.items() if k != SPATIAL_INDEX_COLUMN} for rec in records]
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
            field_map=self._field_map(catalog, default_id=_DEFAULT_ID.get(catalog.provider)),
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


@router.get("/status")
async def skycache_status(request: Request) -> dict[str, Any]:
    """Mirrored catalogs: rows, HATS partitions, coverage MOC (area, ASCII) and mirror log."""
    store = _store_for(request)
    return await asyncio.to_thread(store.status)


@router.post("/mirror")
async def skycache_mirror(request: Request, body: MirrorRequest) -> dict[str, Any]:
    """Mirror a region of one catalog into the local HATS store (synchronous)."""
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
        )
    except (MirrorInputError, InvalidCoordinateError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        service = getattr(request.app.state, "service", None)
        registry = getattr(service, "registry", None) or CatalogRegistry(Settings().catalog_registry_path)
    try:
        definition = registry.get(catalog)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Catalog '{catalog}' not found in registry") from exc
    try:
        result = LocalProvider(store).query_sync(definition, validate_target(ra, dec), radius_arcsec)
    except CoverageError as exc:
        raise HTTPException(status_code=409, detail=exc.as_dict()) from exc
    except InvalidCoordinateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
                                 max_response_bytes=settings.max_response_bytes)
        return await mirror_region(
            args.catalog, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
            healpix_order=args.order, healpix_pixels=args.pixels, store=store, providers=providers,
            registry=registry, tile_rows=args.tile_rows, timeout=settings.request_timeout_seconds,
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
    status = store.status()
    if args.json:
        print(json.dumps(status, indent=2, default=str))
        return 0
    print(f"Sky cache at {status['root']} ({len(status['catalogs'])} catalog(s))")
    for cat in status["catalogs"]:
        print(f"  {cat['catalog']:<20} {cat['rows']:>9} rows  {cat['partitions']:>4} partitions  "
              f"{cat['coverage_area_deg2']:.4f} deg^2  {cat['disk_bytes'] / 1e6:.2f} MB  updated {cat['updated_at']}")
    return 0


def cli_delete(args: argparse.Namespace) -> int:
    store = SkyCache(args.store) if args.store else SkyCache()
    try:
        removed = store.delete(args.catalog)
    except MirrorInputError as exc:
        print(f"Error: {exc}")
        return 2
    print(f"Deleted {args.catalog}" if removed else f"{args.catalog} is not mirrored")
    return 0 if removed else 1


def cli_cone(args: argparse.Namespace) -> int:
    store = SkyCache(args.store) if args.store else SkyCache()
    registry = CatalogRegistry(Settings().catalog_registry_path)
    try:
        definition = registry.get(args.catalog)
        started = time.perf_counter()
        result = LocalProvider(store).query_sync(definition, validate_target(args.ra, args.dec), args.radius_arcsec)
        elapsed = (time.perf_counter() - started) * 1000.0
    except KeyError:
        print(f"Error: unknown catalog {args.catalog!r}")
        return 2
    except (CoverageError, InvalidCoordinateError) as exc:
        print(f"Error: {exc}")
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
