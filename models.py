"""AstroSearch data models, coordinate astrometry, parsers, and embedded catalog registry."""

from __future__ import annotations

import csv
import functools
import html
import io
import json
import logging
import math
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii as astropy_ascii
from astropy.io.votable import parse as parse_votable

logger = logging.getLogger("astrosearch.models")

# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------


class AstroSearchError(Exception):
    """Base exception for all AstroSearch errors."""


class InvalidCoordinateError(AstroSearchError):
    """Raised when RA, DEC, or epoch coordinates fail validation."""


class CatalogUnavailableError(AstroSearchError):
    """Raised when an astronomy catalog or provider service is unavailable."""


class RateLimitedError(CatalogUnavailableError):
    """Raised when an archive throttles requests (HTTP 429) or times out the request (HTTP 408).

    A subclass of CatalogUnavailableError so fallbacks and canary skip logic apply.
    """


class QueryTimeoutError(AstroSearchError):
    """Raised when a catalog query exceeds its allotted timeout."""


class CatalogQueryError(AstroSearchError):
    """Raised when an upstream catalog HTTP request returns an error."""


class ResponseParseError(AstroSearchError):
    """Raised when a provider response cannot be parsed."""


class ObjectResolutionError(AstroSearchError):
    """Raised when an astronomical object name cannot be resolved to coordinates."""


class ResolverUnavailableError(ObjectResolutionError):
    """The name resolver itself failed (network error, HTTP 5xx): the name may well be
    valid. A subclass of ObjectResolutionError, so existing handlers keep working; HTTP
    routes report it as 503 rather than 404."""


class RegistryError(AstroSearchError, ValueError):
    """Raised when a catalog registry file cannot be parsed or fails strict validation."""


def every_catalog_failed(record: UnifiedRecord | Mapping[str, Any]) -> bool:
    """True when a search queried at least one catalog and every one of them failed (a total
    outage, not an empty sky): the CLI search and stream commands then exit 1."""
    data = record.as_dict() if isinstance(record, UnifiedRecord) else record
    queried = int(data.get("catalogs_queried") or 0)
    failed = {str(f.get("catalog")) for f in data.get("failures") or [] if isinstance(f, Mapping)}
    return queried > 0 and len(failed) >= queried


# Largest cone of POST /api/v1/search (and its batch, saved-query and manifest forms) and of
# the streamed search: one degree. A wider cone makes every archive scan huge sky areas.
MAX_SEARCH_RADIUS_ARCSEC = 3600.0


def resolution_failure_status(exc: BaseException) -> int:
    """The HTTP status every route uses for a failed name resolution.

    * 503 -- the resolver could not be reached or answered HTTP 5xx/429
      (:class:`ResolverUnavailableError`): the name may well be valid, retry later;
    * 502 -- the resolver answered, but not with a usable document (an HTML maintenance
      page, truncated XML, an HTTP 4xx of the service itself);
    * 422 -- an empty name;
    * 404 -- a well-formed answer that knows no object of that name.
    """
    from xml.etree.ElementTree import ParseError

    if isinstance(exc, ResolverUnavailableError):
        return 503
    text = str(exc)
    if "must not be empty" in text:
        return 422
    if text.startswith("Sesame request failed"):
        return 502
    if isinstance(exc.__cause__, ParseError):
        return 502
    if text.startswith("Sesame response could not be parsed") and "No coordinates found" not in text:
        return 502
    return 404


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Target:
    """Canonical representation of a search target position."""

    ra: float
    dec: float
    frame: str = "icrs"
    # Julian year of (ra, dec). None = unknown: cones are neither widened nor propagated.
    epoch: float | None = None
    # Proper motion of the target (mas/yr, pm_ra includes cos(dec)). Both or neither.
    pm_ra_masyr: float | None = None
    pm_dec_masyr: float | None = None
    # Trigonometric parallax of the target (mas). Used to remove the annual parallactic
    # displacement from single-epoch catalog positions of the target (2MASS, SDSS, ...).
    parallax_mas: float | None = None

    @property
    def proper_motion(self) -> tuple[float, float] | None:
        """(pm_ra_cosdec, pm_dec) in mas/yr when both components are known."""
        if self.pm_ra_masyr is None or self.pm_dec_masyr is None:
            return None
        return (self.pm_ra_masyr, self.pm_dec_masyr)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResolvedObject:
    """Canonical identity and position returned by an object-name resolver."""

    query: str
    canonical_name: str | None
    ra_deg: float
    dec_deg: float
    aliases: list[str]
    object_type: str | None
    redshift: float | None
    pm_ra_masyr: float | None
    pm_dec_masyr: float | None
    epoch: float | None
    resolver: str
    resolver_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogDefinition:
    """Specification of an astronomical catalog and its access parameters."""

    name: str
    provider: str
    wavelength: str
    enabled: bool = True
    endpoint: str | None = None
    table: str | None = None
    catalog: str | None = None
    description: str | None = None
    query_method: str | None = None
    units: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    profiles: tuple[str, ...] = ()
    # Epoch of the catalog positions: a Julian year (e.g. 2016.0), the name of a
    # per-row epoch column, a list of columns whose mean is the epoch, or a lookup
    # {"column": name, "values": {value: jyear}}. Per-row catalogs declare the span
    # of their epochs in parameters["epoch_range"] so cones can be epoch-widened.
    epoch: float | str | list[str] | dict[str, Any] | None = None
    # Format of per-row epoch columns: "jyear", "mjd", "jd" (None = infer from unit).
    epoch_format: str | None = None
    # Positional-error specification, converted to a 1-sigma circular arcsec value.
    # Keys: columns, units (str | list), kind (sigma|ellipse|ellipse95|ellipse95axis|
    # radius90|radius95|axis90|radial|first), divisor, systematic_arcsec, scale,
    # floor_arcsec, default_arcsec, fallback_column, fallback_values.
    pos_error: dict[str, Any] = field(default_factory=dict)
    citation: str | None = None
    acknowledgement: str | None = None
    max_rows: int = 200
    timeout_seconds: float | None = None
    coverage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogSource:
    """A detected astronomical source returned by a catalog."""

    catalog: str
    source_id: str
    ra: float
    dec: float
    positional_error_arcsec: float | None
    data: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    epoch: float | None = None
    proper_motion_ra_masyr: float | None = None
    proper_motion_dec_masyr: float | None = None
    position_uncertainty_arcsec: float | None = None
    # (earliest, latest) Julian year when the row's epoch is unknown but bounded (e.g. a
    # Chandra CSC master source observed at some time in 1999.5-2022, or a NED position
    # copied from 2MASS, 1997.4-2001.2). Only set when ``epoch`` is None.
    epoch_range: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryPlan:
    """Planned query for a single catalog."""

    catalog: str
    provider: str
    endpoint: str | None
    parameters: dict[str, Any]
    radius_arcsec: float
    wavelength: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Match:
    """Crossmatch association between a target and an archive detection."""

    catalog: str
    source: CatalogSource
    separation_arcsec: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog": self.catalog,
            "source": self.source.as_dict(),
            "separation_arcsec": self.separation_arcsec,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CatalogFailure:
    """Error record for a catalog that failed during execution."""

    catalog: str
    status: str = "failed"
    error_type: str | None = None
    message: str | None = None
    elapsed_ms: float | None = None
    # Set when a fallback archive was tried after the primary failed (both failed):
    # provider, endpoint, and the primary's error ('reason').
    fallback: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnifiedRecord:
    """Provenance-rich unified result aggregating all catalog crossmatches."""

    target: dict[str, Any]
    catalogs_queried: int
    catalog_results: dict[str, Any]
    counterparts: dict[str, list[dict[str, Any]]]
    failures: list[dict[str, Any]]
    provenance: dict[str, Any]
    crossmatch_groups: list[dict[str, Any]] = field(default_factory=list)
    resolved_object: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Runtime Settings
# ---------------------------------------------------------------------------


class Settings:
    """Runtime configuration read from environment variables or overrides."""

    def __init__(self, **overrides: Any) -> None:
        def val(name: str, default: Any) -> Any:
            return overrides.get(name, os.getenv(name, default))

        self.default_radius_arcsec = float(val("DEFAULT_RADIUS_ARCSEC", 3.0))
        self.request_timeout_seconds = float(val("REQUEST_TIMEOUT_SECONDS", 30.0))
        # Upper bound on every catalog's own timeout_seconds (registry values are 60-90 s).
        # CATALOG_TIMEOUT_CAP_SECONDS wins; otherwise an explicitly set
        # REQUEST_TIMEOUT_SECONDS caps them too, so an operator can shorten a search.
        cap = val("CATALOG_TIMEOUT_CAP_SECONDS", None)
        if cap in (None, "") and ("REQUEST_TIMEOUT_SECONDS" in overrides or os.getenv("REQUEST_TIMEOUT_SECONDS")):
            cap = self.request_timeout_seconds
        self.catalog_timeout_cap_seconds: float | None = float(cap) if cap not in (None, "") else None
        self.max_response_bytes = int(val("MAX_RESPONSE_BYTES", 10_000_000))
        self.resolver_endpoint = str(val("SESAME_ENDPOINT", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"))
        self.catalog_registry_path = val("CATALOG_REGISTRY_PATH", None)
        self.log_level = str(val("LOG_LEVEL", "INFO"))

        if not math.isfinite(self.default_radius_arcsec) or self.default_radius_arcsec <= 0:
            raise ValueError("DEFAULT_RADIUS_ARCSEC must be finite and greater than zero.")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be finite and greater than zero.")
        cap_value = self.catalog_timeout_cap_seconds
        if cap_value is not None and (not math.isfinite(cap_value) or cap_value <= 0):
            raise ValueError("CATALOG_TIMEOUT_CAP_SECONDS must be finite and greater than zero.")
        if self.max_response_bytes <= 0:
            raise ValueError("MAX_RESPONSE_BYTES must be greater than zero.")


# ---------------------------------------------------------------------------
# Coordinate Validation
# ---------------------------------------------------------------------------


def _as_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise InvalidCoordinateError(f"{name} must be finite.")
    return numeric


def validate_target(
    ra: float | str,
    dec: float | str,
    *,
    frame: str = "icrs",
    epoch: float | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
) -> Target:
    """Validate coordinates and normalize right ascension to [0, 360).

    ``pm_ra_masyr``/``pm_dec_masyr`` (mas/yr, RA component includes cos dec) must be
    given together; they let catalog cones follow the target to each catalog's epoch.
    ``parallax_mas`` (optional, >= 0 and below 1000 mas) removes the annual parallax
    from single-epoch catalog positions of the target.
    """
    ra_val = _as_float(ra, "ra") % 360.0
    if ra_val >= 360.0:  # float modulo of tiny negatives (e.g. -1e-14 % 360 == 360.0)
        ra_val = 0.0
    dec_val = _as_float(dec, "dec")
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
        pm_ra_masyr = _as_float(pm_ra_masyr, "pm_ra_masyr")
        pm_dec_masyr = _as_float(pm_dec_masyr, "pm_dec_masyr")
        if math.hypot(pm_ra_masyr, pm_dec_masyr) > 20_000.0:
            raise InvalidCoordinateError("proper motion exceeds 20 arcsec/yr; check units (mas/yr expected).")
    if parallax_mas is not None:
        parallax_mas = _as_float(parallax_mas, "parallax_mas")
        if not 0.0 <= parallax_mas < 1000.0:
            raise InvalidCoordinateError("parallax_mas must be in [0, 1000) mas (Proxima Cen, the nearest star, has 768 mas).")
    try:
        coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame=frame)
    except Exception as exc:
        raise InvalidCoordinateError(f"Unsupported coordinate frame: {frame}") from exc
    if not math.isfinite(coord.ra.deg) or not math.isfinite(coord.dec.deg):
        raise InvalidCoordinateError("Coordinate values are not finite.")
    return Target(ra=ra_val, dec=dec_val, frame=frame, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                  parallax_mas=parallax_mas)


# Object types (SIMBAD otype codes and NED prefphytype values, lower case) of
# extragalactic objects: any proper motion measured for them (e.g. a Gaia solution of
# a galaxy nucleus or quasar) is noise, so they are treated as stationary. (SIMBAD's
# 'Sy?' is a symbiotic-star candidate and 'Lev' a microlensing event, both Galactic.)
EXTRAGALACTIC_OTYPES: frozenset[str] = frozenset({
    # SIMBAD
    "g", "gig", "gic", "bic", "ig", "pag", "sbg", "syg", "sy1", "sy2", "agn", "lin", "sfg", "bcg",
    "lsb", "emg", "h2g", "rg", "qso", "bla", "bll", "grg", "clg", "cgg", "scg", "pcg", "lei", "leq",
    "leg", "lya", "dla", "mal", "lls", "bal", "als", "agn?", "qso?", "g?", "bla?", "bll?", "rg?",
    "grg?", "clg?",
    # current SIMBAD codes of candidates, pairs and lensed objects
    "gip", "ag?", "q?", "bz?", "bl?", "c?g", "gr?", "sc?", "gls", "gle", "ls?", "le?", "li?",
    # NED
    "gclstr", "ggroup", "gpair", "gtrpl", "qgroup", "g_lens", "q_lens", "qsolens",
})


def is_extragalactic_type(object_type: Any) -> bool:
    """True when a SIMBAD/NED object type denotes an extragalactic object."""
    return str(object_type or "").strip().lower() in EXTRAGALACTIC_OTYPES


def resolved_target(resolved: ResolvedObject, *, frame: str = "icrs") -> Target:
    """Build the search Target for a resolved name, carrying epoch and proper motion.

    Galactic objects with a resolver proper motion move with it (and carry the
    resolver parallax); extragalactic objects are treated as stationary (pm = 0);
    otherwise the motion is unknown.
    """
    meta = resolved.resolver_metadata or {}
    pm: tuple[float, float] | None = None
    parallax: float | None = None
    if resolved.epoch is not None:
        if meta.get("proper_motion_applicable") and resolved.pm_ra_masyr is not None and resolved.pm_dec_masyr is not None:
            pm = (resolved.pm_ra_masyr, resolved.pm_dec_masyr)
            plx = _to_float(meta.get("parallax_mas"))
            parallax = plx if plx is not None and 0.0 < plx < 1000.0 else None
        elif meta.get("extragalactic"):
            pm = (0.0, 0.0)
    return validate_target(
        resolved.ra_deg, resolved.dec_deg, frame=frame, epoch=resolved.epoch,
        pm_ra_masyr=pm[0] if pm else None, pm_dec_masyr=pm[1] if pm else None, parallax_mas=parallax,
    )


# ---------------------------------------------------------------------------
# Parsed Tables & Column Metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColumnMeta:
    """Per-column metadata (unit, UCD, datatype) reported by an archive."""

    name: str
    unit: str | None = None
    ucd: str | None = None
    datatype: str | None = None
    description: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedTable:
    """Rows keyed by column name plus column metadata and service status flags."""

    rows: list[dict[str, Any]]
    columns: list[ColumnMeta] = field(default_factory=list)
    query_status: str | None = None
    status_message: str | None = None
    format: str = "unknown"

    @property
    def truncated(self) -> bool:
        return (self.query_status or "").upper() == "OVERFLOW"

    def column(self, name: str) -> ColumnMeta | None:
        return find_column(self.columns, name)


def find_column(columns: Sequence[ColumnMeta] | None, name: str) -> ColumnMeta | None:
    """Look up a column by exact, case-insensitive, or bare (unquoted/unprefixed) name."""
    if not columns:
        return None
    bare = bare_column_name(name)
    for candidate in (name, bare):
        for col in columns:
            if col.name == candidate:
                return col
    lowered = bare.lower()
    for col in columns:
        if col.name.lower() == lowered:
            return col
    return None


def bare_column_name(expr: str) -> str:
    """Return the result-column name of an ADQL select item.

    ``b.main_id`` -> ``main_id``; ``"2MASS"`` -> ``2MASS``; ``pz.z AS photoz`` -> ``photoz``.
    """
    text = str(expr).strip()
    match = re.search(r"\s+AS\s+(\"[^\"]+\"|\w+)\s*$", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if "." in text and '"' not in text and "(" not in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip('"')


def row_get(row: Mapping[str, Any], name: str) -> Any:
    """Fetch a row value by exact, bare, or case-insensitive column name."""
    if name in row:
        return row[name]
    bare = bare_column_name(name)
    if bare in row:
        return row[bare]
    lowered = bare.lower()
    for key, val in row.items():
        if str(key).lower() == lowered:
            return val
    return None


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars, masked values, bytes, and NaN into plain Python values."""
    if value is None or value is np.ma.masked:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_clean_value(v) for v in value.tolist()]
    return value


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Response Parsers (JSON / VOTable / CSV / IPAC)
# ---------------------------------------------------------------------------


def _json_load(payload: str | bytes | bytearray | dict[str, Any] | list[Any]) -> Any:
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            return json.loads(payload)
        except ValueError as exc:
            raise ResponseParseError(f"Response is not valid JSON: {exc}") from exc
    return payload


def _columns_from_json_meta(meta: list[Any]) -> list[ColumnMeta]:
    columns: list[ColumnMeta] = []
    for idx, item in enumerate(meta):
        if isinstance(item, str):
            columns.append(ColumnMeta(name=item))
            continue
        if not isinstance(item, dict):
            raise ResponseParseError(f"Column metadata entry {idx} is not an object.")
        name = item.get("name") or item.get("column_name")
        if not name:
            raise ResponseParseError(f"Column metadata entry {idx} has no name.")
        columns.append(
            ColumnMeta(
                name=str(name),
                unit=item.get("unit") or None,
                ucd=item.get("ucd") or None,
                datatype=item.get("datatype") or item.get("db_type") or item.get("type"),
                description=item.get("description"),
            )
        )
    return columns


def _zip_rows(columns: list[ColumnMeta], data: list[Any]) -> list[dict[str, Any]]:
    names = [col.name for col in columns]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        if isinstance(row, dict):
            rows.append({str(k): _clean_value(v) for k, v in row.items()})
            continue
        if not isinstance(row, (list, tuple)):
            raise ResponseParseError(f"Row {idx} is a {type(row).__name__}, expected an array of values.")
        if len(row) != len(names):
            raise ResponseParseError(f"Row {idx} has {len(row)} values but metadata declares {len(names)} columns.")
        rows.append({name: _clean_value(val) for name, val in zip(names, row)})
    return rows


def _dict_rows(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ResponseParseError(
                f"Row {idx} is a {type(item).__name__} but no column metadata was supplied to name its values."
            )
        rows.append({str(k): _clean_value(v) for k, v in item.items()})
    return rows


def parse_json_table(payload: str | bytes | dict[str, Any] | list[Any]) -> ParsedTable:
    """Parse any archive JSON shape into named rows plus column metadata.

    Supported shapes: TAP ``{"metadata": [...], "data": [[...]]}``, MAST ``{"info": [...],
    "data": [[...]]}``, record lists ``[{...}]``, ``{"data"|"results"|"request": [{...}]}``,
    and SkyServer ``[{"TableName": ..., "Rows": [{...}]}]``. Array rows without column
    names raise :class:`ResponseParseError` instead of being dropped.
    """
    data = _json_load(payload)
    if isinstance(data, dict):
        meta = None
        for key in ("metadata", "info", "columns", "fields"):
            if isinstance(data.get(key), list) and all(isinstance(m, (dict, str)) for m in data[key]):
                meta = data[key]
                break
        rows_raw = None
        for key in ("data", "results", "request", "rows", "Rows"):
            if isinstance(data.get(key), list):
                rows_raw = data[key]
                break
        if rows_raw is None:
            if meta is not None:
                raise ResponseParseError("JSON payload has column metadata but no data array.")
            return ParsedTable(rows=[{str(k): _clean_value(v) for k, v in data.items()}], format="json-object")
        if meta is not None:
            columns = _columns_from_json_meta(meta)
            return ParsedTable(rows=_zip_rows(columns, rows_raw), columns=columns, format="json-tap")
        return ParsedTable(rows=_dict_rows(rows_raw), format="json-records")
    if isinstance(data, list):
        if data and all(isinstance(t, dict) and "Rows" in t for t in data):
            tables = [t for t in data if t.get("TableName") != "SqlQuery"]
            rows: list[dict[str, Any]] = []
            for table in tables:
                rows.extend(_dict_rows(table.get("Rows") or []))
            return ParsedTable(rows=rows, format="json-skyserver")
        return ParsedTable(rows=_dict_rows(data), format="json-records")
    raise ResponseParseError(f"Unsupported JSON payload of type {type(data).__name__}.")


def parse_json_records(payload: str | bytes | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Parse JSON records returned by public astronomy APIs (all known shapes)."""
    return parse_json_table(payload).rows


_INFO_TAG = re.compile(rb"<(?:\w+:)?INFO\b([^>]*?)(?:/>|>(.*?)</(?:\w+:)?INFO\s*>)", re.DOTALL | re.IGNORECASE)
_XML_ATTR = re.compile(rb"([\w:.-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")


def votable_query_status(payload: bytes | str) -> tuple[str | None, str | None]:
    """Return the (QUERY_STATUS value, message) declared in a VOTable, if any.

    ERROR takes precedence; OVERFLOW is reported when results were truncated.
    """
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    status: str | None = None
    message: str | None = None
    for match in _INFO_TAG.finditer(raw):
        # Attribute values and element text are XML-escaped (&quot; &apos; &amp; ...).
        attrs = {
            m.group(1).decode("utf-8", "replace").lower(): html.unescape((m.group(2) or m.group(3) or b"").decode("utf-8", "replace"))
            for m in _XML_ATTR.finditer(match.group(1))
        }
        if attrs.get("name", "").upper() != "QUERY_STATUS":
            continue
        value = attrs.get("value", "").upper()
        body = match.group(2) or b""
        cdata = re.fullmatch(rb"\s*<!\[CDATA\[(.*)\]\]>\s*", body, re.DOTALL)
        body_text = cdata.group(1).decode("utf-8", "replace") if cdata else html.unescape(body.decode("utf-8", "replace"))
        text = body_text.strip() or attrs.get("content") or None
        if value == "ERROR":
            return "ERROR", text
        if status is None or value == "OVERFLOW":
            status, message = value or None, text
    return status, message


def parse_votable_table(payload: bytes | io.BytesIO | str) -> ParsedTable:
    """Parse an IVOA VOTable (TABLEDATA or BINARY) with units, UCDs, and QUERY_STATUS.

    Raises :class:`CatalogQueryError` when the service declares QUERY_STATUS=ERROR.
    """
    if isinstance(payload, io.BytesIO):
        raw = payload.getvalue()
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = bytes(payload)
    status, message = votable_query_status(raw)
    if status == "ERROR":
        raise CatalogQueryError(f"Service reported QUERY_STATUS=ERROR: {message or 'no message'}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            votable = parse_votable(io.BytesIO(raw), verify="ignore")
            tables = list(votable.iter_tables())
    except Exception as exc:
        raise ResponseParseError(f"VOTable could not be parsed: {exc}") from exc
    if not tables:
        raise ResponseParseError(f"VOTable contains no TABLE (QUERY_STATUS={status or 'missing'}).")
    table = tables[0]
    columns = [
        ColumnMeta(
            name=str(f.name or f.ID),
            unit=str(f.unit) if f.unit is not None else None,
            ucd=f.ucd or None,
            datatype=f.datatype,
            description=f.description,
        )
        for f in table.fields
    ]
    rows: list[dict[str, Any]] = []
    array = table.array
    if array is not None and len(array):
        values = [[_clean_value(v) for v in array[name].tolist()] for name in array.dtype.names]
        for i in range(len(array)):
            rows.append({columns[j].name: values[j][i] for j in range(len(columns))})
    return ParsedTable(rows=rows, columns=columns, query_status=status, status_message=message, format="votable")


def parse_votable_records(payload: bytes | io.BytesIO | str) -> list[dict[str, Any]]:
    """Parse an IVOA VOTable XML stream via Astropy."""
    return parse_votable_table(payload).rows


def parse_csv_table(payload: str | bytes) -> ParsedTable:
    """Parse CSV text rows, skipping comment lines (e.g. SkyServer '#Table1')."""
    text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
    csv_text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = [dict(row) for row in reader]
    columns = [ColumnMeta(name=name) for name in (reader.fieldnames or [])]
    return ParsedTable(rows=rows, columns=columns, format="csv")


def parse_csv_records(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse CSV text rows, skipping comment lines."""
    return parse_csv_table(payload).rows


def parse_ipac_table(payload: str | bytes) -> ParsedTable:
    """Parse an IPAC ASCII table (such as IRSA Gator responses) with column units."""
    text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
    try:
        table = astropy_ascii.read(text, format="ipac")
    except Exception as exc:
        raise ResponseParseError(f"IPAC table could not be parsed: {exc}") from exc
    columns = [
        ColumnMeta(name=name, unit=str(table[name].unit) if table[name].unit is not None else None)
        for name in table.colnames
    ]
    values = {name: [_clean_value(v) for v in table[name].tolist()] for name in table.colnames}
    rows = [{name: values[name][i] for name in table.colnames} for i in range(len(table))]
    return ParsedTable(rows=rows, columns=columns, format="ipac")


def parse_ipac_records(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse an IPAC ASCII table (such as IRSA Gator responses) via Astropy."""
    return parse_ipac_table(payload).rows


# ---------------------------------------------------------------------------
# Unit Conversion
# ---------------------------------------------------------------------------

_ARCSEC_PER_UNIT: dict[str, float] = {
    # NB: 'd' is deliberately absent: it is a DAY in both VOUnit and CDS units.
    "deg": 3600.0, "degree": 3600.0, "degrees": 3600.0,
    "arcmin": 60.0, "arcm": 60.0, "'": 60.0, "amin": 60.0,
    "arcsec": 1.0, "arcs": 1.0, "asec": 1.0, '"': 1.0, "arcsecond": 1.0, "arcseconds": 1.0,
    "mas": 1e-3, "milliarcsec": 1e-3, "uas": 1e-6, "µas": 1e-6,
    "rad": 206264.80624709636, "hourangle": 54000.0,
}
_ANGLE_WORDS = {"deg", "degree", "degrees", "arcmin", "arcsec", "mas", "uas", "rad"}


_HOUR_WORDS = frozenset({"h", "hr", "hour", "hours", "hourangle"})


def _is_custom_unit(unit: u.UnitBase) -> bool:
    """True for a unit the VOUnit parser invented for an unknown word (e.g. 'hourangle', 'foo').

    With ``parse_strict='silent'`` astropy's VOUnit format accepts any word as a new
    custom unit of unknown physical type instead of reporting it unrecognized. Real
    dimensionless-looking units (``mag`` and other function units) are kept.
    """
    if isinstance(unit, u.FunctionUnitBase):
        return False
    return str(unit.physical_type) == "unknown"


@functools.lru_cache(maxsize=256)
def _astropy_unit(text: str) -> u.UnitBase | None:
    # CDS/VizieR write arcsec and arcmin as '"' and "'" (e.g. '"/yr' for a proper motion).
    text = text.replace('"', "arcsec").replace("'", "arcmin")
    for fmt in ("vounit", "cds", None):
        try:
            unit = u.Unit(text, format=fmt, parse_strict="silent") if fmt else u.Unit(text, parse_strict="silent")
        except Exception:
            continue
        if isinstance(unit, u.UnrecognizedUnit) or _is_custom_unit(unit):
            continue
        return unit
    return None


def arcsec_per_unit(unit: Any) -> float | None:
    """Arcseconds per one ``unit`` of angle (None when the unit is not an angle)."""
    if unit is None:
        return None
    if isinstance(unit, (int, float)) and not isinstance(unit, bool):
        return float(unit)
    text = str(unit).strip().strip("[]")
    if not text:
        return None
    if text.lower() in _ARCSEC_PER_UNIT:
        return _ARCSEC_PER_UNIT[text.lower()]
    parsed = _astropy_unit(text)
    if parsed is None:
        return None
    try:
        return float(parsed.to(u.arcsec))
    except u.UnitConversionError:
        return None


def angle_to_arcsec(value: Any, unit: Any) -> float | None:
    """Convert an angular quantity in ``unit`` to arcseconds (None if impossible)."""
    number = _to_float(value)
    factor = arcsec_per_unit(unit)
    if number is None or factor is None:
        return None
    return number * factor


def angle_to_deg(value: Any, unit: Any = "deg") -> float | None:
    """Convert an angle to degrees; a missing unit is assumed to be degrees."""
    number = _to_float(value)
    if number is None:
        return None
    factor = arcsec_per_unit(unit) if unit not in (None, "") else 3600.0
    if factor is None:
        # Hours of right ascension (units 'h', 'hr', 'hour'; 'hourangle' parses as an angle).
        word = str(unit).strip().strip("[]").lower()
        parsed = _astropy_unit(str(unit).strip())
        if word in _HOUR_WORDS and parsed is not None and parsed.physical_type == "time":
            return number * 15.0
        return number
    return number * factor / 3600.0


# Proper-motion unit parts that astropy misreads or cannot express: CDS 'cy' (century)
# parses as 'cycle', and RA motions in seconds of TIME ('s/a', 's/yr', 's/ha', 'ms/a'
# in the SAO, PPM, FK4 tables) are dimensionless to astropy (time / time).
_PM_UNIT = re.compile(r"^\s*([0-9.]*(?:[eE][+-]?\d+)?)\s*([A-Za-z\"']+)\s*/\s*([A-Za-z]+)\s*$")
_PM_TIME_DENOMINATOR_YEARS = {"a": 1.0, "yr": 1.0, "year": 1.0, "ha": 100.0, "cy": 100.0, "century": 100.0}
_RA_TIME_SECONDS = {"s": 1.0, "sec": 1.0, "ms": 1e-3}


def pm_to_masyr(value: Any, unit: Any, dec_deg: float | None = None) -> float | None:
    """Convert a proper motion to mas/yr (a missing unit is assumed mas/yr).

    RA motions in seconds of time per year/century (``s/a``, ``s/yr``, ``s/ha``,
    ``s/cy``, ``ms/a``) are converted with 1 s = 15 arcsec x cos(dec), which needs
    ``dec_deg`` (None without it). A unit that is present but cannot be converted to an
    angular rate gives None (logged), never the raw number mislabelled as mas/yr.
    """
    number = _to_float(value)
    if number is None:
        return None
    if unit in (None, ""):
        return number
    text = str(unit).strip().strip("[]")
    match = _PM_UNIT.match(text)
    if match:
        scale = float(match.group(1)) if match.group(1) else 1.0
        numerator, denominator = match.group(2), match.group(3).lower()
        years = _PM_TIME_DENOMINATOR_YEARS.get(denominator)
        if years is not None and numerator.lower() in _RA_TIME_SECONDS:
            if dec_deg is None or not math.isfinite(dec_deg):
                logger.warning("proper motion in time units %r needs the row's declination; value dropped", text)
                return None
            seconds_per_year = number * scale * _RA_TIME_SECONDS[numerator.lower()] / years
            return seconds_per_year * 15_000.0 * math.cos(math.radians(dec_deg))
        if years is not None and denominator in {"ha", "cy", "century"}:
            factor = arcsec_per_unit(numerator)
            if factor is not None:
                return number * scale * factor * 1000.0 / years
    parsed = _astropy_unit(text)
    if parsed is None:
        logger.warning("proper-motion unit %r not recognised; value dropped", text)
        return None
    try:
        return float((number * parsed).to(u.mas / u.yr).value)
    except u.UnitConversionError:
        logger.warning("proper-motion unit %r is not an angular rate; value dropped", text)
        return None


def parallax_to_mas(value: Any, unit: Any) -> float | None:
    """Convert a parallax to milliarcseconds (missing unit is assumed mas)."""
    number = _to_float(value)
    if number is None:
        return None
    factor = arcsec_per_unit(unit)
    return number if factor is None else number * factor * 1000.0


_JD_1900 = 2415020.0  # JD of Julian epoch J1900.0


def epoch_to_jyear(value: Any, fmt: Any = None) -> float | None:
    """Convert a Julian year, MJD, JD, or ISO date into a Julian year.

    ``fmt`` may be 'jyear'/'yr'/'a', 'mjd', 'jd', or a column unit such as 'd'
    (for which MJD vs JD is decided from the magnitude of the value).
    """
    if value is None or value == "":
        return None
    kind = str(fmt).strip().lower() if fmt not in (None, "") else ""
    number = _to_float(value)
    if number is None:
        try:
            from astropy.time import Time

            return float(Time(str(value).strip()).jyear)
        except Exception:
            return None
    if kind in {"jyear", "yr", "a", "year", "years", "julian year"}:
        result = number
    elif kind == "jd":
        # A declared JD before 1900.0 (JD 2415020) is a format mismatch for any positional
        # catalog (e.g. a Julian year or MJD mislabelled as JD), not an 1800s epoch.
        if number < _JD_1900:
            return None
        result = 2000.0 + (number - 2451545.0) / 365.25
    elif kind == "mjd":
        # Likewise MJD < 15020 (before 1900.0; MJD < 0 is before 1858): e.g. the Julian
        # year 1995.4 declared as MJD would otherwise become a plausible-looking 1864.3.
        if number < _JD_1900 - 2_400_000.5:
            return None
        result = 2000.0 + (number - 51544.5) / 365.25
    else:
        if number > 2_300_000:
            result = 2000.0 + (number - 2451545.0) / 365.25
        elif number > 10_000:
            result = 2000.0 + (number - 51544.5) / 365.25
        else:
            result = number
    return result if 1800.0 <= result <= 2200.0 else None


# ---------------------------------------------------------------------------
# Field Normalization (alias table + UCD-driven mapping)
# ---------------------------------------------------------------------------

FIELD_ALIASES: dict[str, str] = {
    "ra": "ra", "ra_icrs": "ra", "raj2000": "ra", "ramean": "ra",
    "dec": "dec", "dec_icrs": "dec", "de_icrs": "dec", "dej2000": "dec", "decmean": "dec",
    "source_id": "source_id", "objid": "source_id", "designation": "source_id",
    "id": "source_id", "sourceid": "source_id", "main_id": "source_id",
    "objname": "source_id", "prefname": "source_id", "pl_name": "source_id", "oid": "source_id",
    "pmra": "pmra", "pm_ra": "pmra", "pmra_cosdec": "pmra", "sy_pmra": "pmra",
    "pmdec": "pmdec", "pm_dec": "pmdec", "pmde": "pmdec", "sy_pmdec": "pmdec",
    "ref_epoch": "epoch", "epoch": "epoch", "obsepoch": "epoch",
    # Only unit-unambiguous error aliases (arcsec by convention). Unit-bearing
    # error columns (Gaia ra_error in mas, NED 95% ellipses, ...) are handled by
    # the catalog's pos_error spec or by UCD metadata, never by name alone.
    "poserr": "position_uncertainty_arcsec", "pos_error": "position_uncertainty_arcsec",
    "parallax": "parallax", "plx_value": "parallax", "sy_plx": "parallax", "plx": "parallax",
    "z": "redshift", "redshift": "redshift", "z_value": "redshift", "rvz_redshift": "redshift",
    "otype": "object_type", "objtype": "object_type", "prefphytype": "object_type",
    "morphology": "morphology", "sp_type": "spectral_type", "spectral_type": "spectral_type",
    "obsdate": "observation_date", "obs_date": "observation_date",
    "observation_date": "observation_date", "date_obs": "observation_date",
    "quality": "quality_flags", "quality_flag": "quality_flags", "flags": "quality_flags",
}

# Column names that are computed server-side and must never be mapped by UCD
# (VizieR labels computed DISTANCE() aliases with the RA column's UCD and unit).
COMPUTED_COLUMNS: frozenset[str] = frozenset({"match_dist", "dist", "dist_arcsec", "dist_deg", "_r", "angdist", "distance"})

_UCD_RULES: list[tuple[str, tuple[Any, ...]]] = [
    ("ra", (lambda s: s == "pos.eq.ra;meta.main", lambda s: s == "pos.eq.ra")),
    ("dec", (lambda s: s == "pos.eq.dec;meta.main", lambda s: s == "pos.eq.dec")),
    ("source_id", (lambda s: s == "meta.id;meta.main", lambda s: s == "meta.id")),
    ("pmra", (lambda s: s.startswith("pos.pm;pos.eq.ra"),)),
    ("pmdec", (lambda s: s.startswith("pos.pm;pos.eq.dec"),)),
    ("parallax", (lambda s: s in {"pos.parallax.trig", "pos.parallax"}, lambda s: s.startswith("pos.parallax"))),
    ("redshift", (lambda s: s.startswith("src.redshift") and "stat" not in s,)),
    ("object_type", (lambda s: s == "src.class", lambda s: s.startswith("src.class") and "stargalaxy" not in s)),
    ("spectral_type", (lambda s: s.startswith("src.sptype"),)),
    ("ra_error_arcsec", (lambda s: s == "stat.error;pos.eq.ra", lambda s: s.startswith("stat.error;pos.eq.ra"))),
    ("dec_error_arcsec", (lambda s: s == "stat.error;pos.eq.dec", lambda s: s.startswith("stat.error;pos.eq.dec"))),
    ("epoch", (lambda s: s in {"time.epoch", "meta.ref;time.epoch"}, lambda s: s.startswith("time.epoch"))),
]


def _normalize_ucd(ucd: str) -> str:
    return ";".join(part.strip().lower() for part in str(ucd).split(";") if part.strip())


def ucd_field_map(columns: Sequence[ColumnMeta], exclude: Sequence[str] | frozenset[str] = COMPUTED_COLUMNS) -> dict[str, ColumnMeta]:
    """Map canonical field names to columns using IVOA UCDs (computed columns excluded)."""
    excluded = {name.lower() for name in exclude}
    candidates = [c for c in columns if c.ucd and c.name.lower() not in excluded]
    mapping: dict[str, ColumnMeta] = {}
    for canonical, predicates in _UCD_RULES:
        for predicate in predicates:
            hit = next((c for c in candidates if predicate(_normalize_ucd(c.ucd or ""))), None)
            if hit is not None:
                mapping[canonical] = hit
                break
    return mapping


def convert_canonical(canonical: str, value: Any, unit: Any, dec_deg: float | None = None) -> Any:
    """Convert a raw column value into the canonical unit for ``canonical``.

    ``dec_deg`` (the row's declination) is needed for RA proper motions given in
    seconds of time.
    """
    if value is None or value == "":
        return None
    if canonical in {"ra", "dec"}:
        return angle_to_deg(value, unit)
    if canonical == "pmra":
        return pm_to_masyr(value, unit, dec_deg)
    if canonical == "pmdec":
        return pm_to_masyr(value, unit)
    if canonical == "parallax":
        return parallax_to_mas(value, unit)
    if canonical == "epoch":
        return epoch_to_jyear(value, unit)
    if canonical in {"ra_error_arcsec", "dec_error_arcsec", "position_uncertainty_arcsec"}:
        return angle_to_arcsec(value, unit if unit not in (None, "") else "arcsec")
    if canonical == "redshift":
        return _to_float(value)
    if canonical == "source_id":
        return " ".join(str(value).split())
    return value


def normalize_field_names(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map heterogeneous astronomical catalog field names to canonical keys."""
    normalized: dict[str, Any] = {}
    for key, val in record.items():
        clean_key = str(key).strip().lower()
        normalized[FIELD_ALIASES.get(clean_key, str(key).strip())] = val
    return normalized


def _row_declination(
    raw_record: Mapping[str, Any],
    aliased: Mapping[str, Any],
    ucd_map: Mapping[str, ColumnMeta],
    columns: Sequence[ColumnMeta] | None,
    field_map: Mapping[str, str] | None,
) -> float | None:
    """The row's declination in degrees (same precedence as the normalized 'dec')."""
    if field_map and field_map.get("dec"):
        raw = row_get(raw_record, field_map["dec"])
        col = find_column(columns, field_map["dec"])
        dec = angle_to_deg(raw, col.unit if col else None)
        if dec is not None:
            return dec
    if "dec" in ucd_map:
        dec = angle_to_deg(raw_record.get(ucd_map["dec"].name), ucd_map["dec"].unit)
        if dec is not None:
            return dec
    return _to_float(aliased.get("dec"))


def normalize_source_record(
    raw_record: Mapping[str, Any],
    *,
    columns: Sequence[ColumnMeta] | None = None,
    field_map: Mapping[str, str] | None = None,
    exclude: Sequence[str] | frozenset[str] = COMPUTED_COLUMNS,
) -> dict[str, Any]:
    """Canonicalize provider record values without hallucinating missing fields.

    Precedence: explicit ``field_map`` (canonical -> column) > UCD metadata (with
    unit conversion) > the static alias table.
    """
    values = normalize_field_names(raw_record)
    ucd_map = ucd_field_map(columns, exclude) if columns else {}
    dec_hint = _row_declination(raw_record, values, ucd_map, columns, field_map)
    for canonical, col in ucd_map.items():
        converted = convert_canonical(canonical, raw_record.get(col.name), col.unit, dec_hint)
        if converted is not None:
            values[canonical] = converted
    if field_map:
        for canonical, column_name in field_map.items():
            raw = row_get(raw_record, column_name)
            if raw is None:
                continue
            col = find_column(columns, column_name)
            converted = convert_canonical(canonical, raw, col.unit if col else None, dec_hint)
            if converted is not None:
                values[canonical] = converted
    for key in ("ra", "dec"):
        if key in values:
            try:
                values[key] = float(values[key])
            except (TypeError, ValueError):
                pass
    if "source_id" in values and values["source_id"] is not None:
        values["source_id"] = " ".join(str(values["source_id"]).split())
    for key in ("pmra", "pmdec", "epoch", "position_uncertainty_arcsec", "parallax", "redshift",
                "ra_error_arcsec", "dec_error_arcsec"):
        if key in values:
            number = _to_float(values[key])
            if number is None:
                values.pop(key, None)
            else:
                values[key] = number
    return values


# ---------------------------------------------------------------------------
# Positional Errors & Epochs
# ---------------------------------------------------------------------------

POS_ERROR_KINDS = frozenset({
    "sigma", "ellipse", "ellipse95", "ellipse95axis", "radius95", "radius90", "axis90", "radial", "first",
})
ELLIPSE_KINDS = frozenset({"ellipse", "ellipse95", "ellipse95axis"})
K95_2D = math.sqrt(-2.0 * math.log(0.05))  # 2.4477: 95% 2-D (Rayleigh) contour / sigma
K90_2D = math.sqrt(-2.0 * math.log(0.10))  # 2.1460: 90% 2-D contour / sigma
K95_1D = 1.959963984540054  # 95% two-sided 1-D interval / sigma (per-axis 95% semi-axes)
K90_1D = 1.6448536269514722  # 90% two-sided 1-D interval / sigma
SPECIAL_ERROR_UNITS = frozenset({"s_ra"})


def _error_unit_factor(unit: Any, dec_deg: float | None) -> float | None:
    if isinstance(unit, str) and unit.strip().lower() == "s_ra":
        # RA error in seconds of time: 1 s = 15 arcsec * cos(dec) on the sky.
        return 15.0 * math.cos(math.radians(dec_deg or 0.0))
    return arcsec_per_unit(unit)


def compute_positional_error(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    dec_deg: float | None = None,
    columns: Sequence[ColumnMeta] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Convert catalog-native positional errors into a 1-sigma circular error (arcsec).

    The circular value is the RMS of the two per-axis 1-sigma errors,
    ``sqrt((s1^2 + s2^2) / 2)``, plus any systematic term in quadrature.
    Returns (sigma_arcsec, details) where details keeps the raw values and ellipse.

    Kinds: ``ellipse`` (1-sigma semi-axes), ``ellipse95`` (semi-axes of the 2-D 95%
    contour, / 2.4477), ``ellipse95axis`` (semi-axes that are per-axis 95% intervals,
    / 1.96), ``radius95``/``radius90`` (2-D radii), ``axis90`` (per-axis 90%),
    ``radial`` (sqrt(s_ra^2 + s_dec^2)), ``first`` (FIRST formula) and ``sigma``.
    An explicit ``divisor`` in the spec replaces the kind's conversion factor (for
    ``sigma`` the default factor is 1, so the catalog values are divided by it).
    The third column of an ellipse kind is a position angle in degrees.

    Order: statistical sigma -> ``systematic_arcsec`` added in quadrature -> multiplied
    by ``scale`` (an empirical calibration factor, e.g. 2RXS s = 1.22) -> raised to
    ``floor_arcsec``. Fallback/default values are used as given.
    """
    kind = str(spec.get("kind", "sigma"))
    cols = [str(c) for c in (spec.get("columns") or [])]
    units = spec.get("units", spec.get("unit"))
    unit_list = list(units) if isinstance(units, (list, tuple)) else [units] * len(cols)
    unit_list += [None] * (len(cols) - len(unit_list))
    if kind in ELLIPSE_KINDS and len(unit_list) > 2:
        unit_list[2] = "deg"  # position angle, never a length
    divisor_override = _to_float(spec.get("divisor"))
    divisor_source = "spec" if divisor_override is not None else None
    lookup = spec.get("divisor_lookup")
    if isinstance(lookup, Mapping):
        # Per-row convention, e.g. NED copies each source catalogue's own error ellipse
        # (CSC 95% per axis, 2XMM 2.5 x radial sigma, 2MASS/WISE 2.5 x sigma), keyed by
        # pos_bibcode. Values missing from the table keep ``divisor``, flagged 'assumed'.
        key = row_get(row, str(lookup.get("column", "")))
        found = _to_float((lookup.get("values") or {}).get(str(key).strip())) if key is not None else None
        if found is not None:
            divisor_override, divisor_source = found, "lookup"
        else:
            divisor_source = "assumed"
    if divisor_override is not None and divisor_override <= 0:
        raise ValueError("pos_error divisor must be positive")
    raw = {c: row_get(row, c) for c in cols}

    def arcsec(i: int) -> float | None:
        if i >= len(cols):
            return None
        number = _to_float(raw[cols[i]])
        if number is None or number < 0:
            return None
        unit = unit_list[i]
        if unit in (None, ""):
            col = find_column(columns, cols[i])
            unit = col.unit if col else None
        factor = _error_unit_factor(unit, dec_deg)
        return None if factor is None else number * factor

    sigma: float | None = None
    sigma_ra: float | None = None
    sigma_dec: float | None = None
    ellipse: dict[str, float | None] | None = None

    if kind == "sigma":
        sigma_scale = 1.0 / (divisor_override or 1.0)
        vals = [None if v is None else v * sigma_scale for v in (arcsec(i) for i in range(len(cols)))]
        present = [v for v in vals if v is not None]
        if len(cols) >= 2:
            sigma_ra, sigma_dec = vals[0], vals[1]
        if present:
            sigma = math.sqrt(sum(v * v for v in present) / len(present))
    elif kind in ELLIPSE_KINDS:
        default_divisor = {"ellipse": 1.0, "ellipse95": K95_2D, "ellipse95axis": K95_1D}[kind]
        scale = 1.0 / (divisor_override or default_divisor)
        major = arcsec(0)
        minor = arcsec(1)
        if major is not None:
            major *= scale
            minor = minor * scale if minor is not None else major
            sigma = math.sqrt((major * major + minor * minor) / 2.0)
            ellipse = {"major": major, "minor": minor, "pa_deg": _to_float(raw.get(cols[2])) if len(cols) > 2 else None}
    elif kind in {"radius95", "radius90", "axis90", "radial"}:
        divisor = divisor_override or {"radius95": K95_2D, "radius90": K90_2D, "axis90": K90_1D, "radial": math.sqrt(2.0)}[kind]
        radius = arcsec(0)
        if radius is not None:
            sigma = radius / divisor
    elif kind == "first":
        # FIRST: eps90 = size * (1/SNR + 1/20) along each axis (90% per axis).
        first_divisor = divisor_override or K90_1D
        major, minor = arcsec(0), arcsec(1)
        peak, rms = _to_float(raw.get(cols[2])) if len(cols) > 2 else None, _to_float(raw.get(cols[3])) if len(cols) > 3 else None
        if major is not None and peak is not None and rms and rms > 0:
            snr = (peak - 0.25) / rms
            if snr > 0:
                minor = minor if minor is not None else major
                eps_major = major * (1.0 / snr + 1.0 / 20.0) / first_divisor
                eps_minor = minor * (1.0 / snr + 1.0 / 20.0) / first_divisor
                sigma = math.sqrt((eps_major**2 + eps_minor**2) / 2.0)
                ellipse = {"major": eps_major, "minor": eps_minor, "pa_deg": None}
    else:
        raise ValueError(f"Unknown positional error kind: {kind}")

    source = "catalog" if sigma is not None else None
    growth = _epoch_growth(row, spec.get("epoch_growth"), sigma, ellipse)
    if growth is not None and growth.get("term_arcsec") is not None and sigma is not None:
        term = float(growth["term_arcsec"])
        sigma = math.hypot(sigma, term)
        if ellipse is not None:
            ellipse = {**ellipse, "major": math.hypot(ellipse["major"] or 0.0, term),
                       "minor": math.hypot(ellipse["minor"] or 0.0, term)}
    statistical = sigma
    systematic = _to_float(spec.get("systematic_arcsec"))
    if sigma is not None and systematic:
        sigma = math.hypot(sigma, systematic)
    scale_factor = _to_float(spec.get("scale"))
    if scale_factor is not None and scale_factor <= 0:
        raise ValueError("pos_error scale must be positive")
    if sigma is not None and scale_factor is not None:
        sigma *= scale_factor
    floor = _to_float(spec.get("floor_arcsec"))
    if sigma is not None and floor is not None:
        sigma = max(sigma, floor)
    if sigma is None and spec.get("fallback_column"):
        key = row_get(row, str(spec["fallback_column"]))
        mapping = spec.get("fallback_values") or {}
        if key is not None and str(key).strip() in mapping:
            sigma = float(mapping[str(key).strip()])
            source = "fallback"
    if sigma is None and _to_float(spec.get("default_arcsec")) is not None:
        sigma = float(spec["default_arcsec"])
        source = "default"

    details: dict[str, Any] = {
        "kind": kind,
        "columns": cols,
        "units": unit_list,
        "divisor": divisor_override,
        "divisor_source": divisor_source,
        "raw": raw,
        "sigma_arcsec": sigma,
        "statistical_sigma_arcsec": statistical,
        "systematic_arcsec": systematic,
        "scale": scale_factor,
        "sigma_ra_arcsec": sigma_ra,
        "sigma_dec_arcsec": sigma_dec,
        "ellipse_1sigma_arcsec": ellipse,
        "source": source,
    }
    if growth is not None:
        details["epoch_growth"] = growth
    return sigma, details


def _epoch_growth(
    row: Mapping[str, Any],
    spec: Any,
    sigma: float | None,
    ellipse: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Error growth of a position propagated from its measurement epoch to another epoch.

    SIMBAD publishes Gaia positions moved to J2000.0 with proper motion but keeps the
    Gaia (J2016.0) coordinate errors, so the J2000 error must add pm_error x 16 yr.
    SIMBAD's basic table is queried without proper-motion errors, so they are
    estimated from the position error with ``pm_error_ratio`` (pm error per yr / position
    error), measured for Gaia DR3 (median 1.25, 90th percentile 1.4) and DR2 (1.85 / 2.1);
    the upper-decile values are used. Rows without a proper motion (not propagated) or
    whose source epoch is unknown get a note instead of a term.
    """
    if not isinstance(spec, Mapping) or sigma is None:
        return None
    pm_cols = [str(c) for c in (spec.get("pm_columns") or [])]
    if pm_cols and any(_to_float(row_get(row, c)) is None for c in pm_cols):
        return None  # not propagated: the position is at its measurement epoch
    to_epoch = _to_float(spec.get("to_epoch"))
    lookup = spec.get("epoch_lookup")
    source_epoch = _to_float(_epoch_lookup(row, lookup)) if isinstance(lookup, Mapping) else None
    info: dict[str, Any] = {"to_epoch": to_epoch, "source_epoch": source_epoch, "term_arcsec": None}
    if source_epoch is None or to_epoch is None:
        info["note"] = "position propagated from an unknown epoch; the ellipse applies at that epoch"
        return info
    ratios = {float(k): float(v) for k, v in (spec.get("pm_error_ratio") or {}).items()}
    ratio = ratios.get(source_epoch)
    if ratio is None:
        info["note"] = f"no proper-motion error estimate for source epoch {source_epoch:g}; ellipse applies at that epoch"
        return info
    base = max(float(ellipse["major"]), float(ellipse["minor"] or 0.0)) if ellipse else sigma
    info.update({
        "term_arcsec": base * ratio * abs(to_epoch - source_epoch),
        "pm_error_ratio": ratio,
        "note": "estimated: proper-motion error taken as pm_error_ratio x position error per year",
    })
    return info


def resolve_epoch(
    row: Mapping[str, Any],
    epoch_spec: float | str | Sequence[str] | Mapping[str, Any] | None,
    epoch_format: str | None = None,
    columns: Sequence[ColumnMeta] | None = None,
) -> float | None:
    """Return the Julian-year epoch of a row from a fixed epoch or per-row column(s).

    ``epoch_spec`` may also be a mapping:

    * a lookup ``{"column": name, "values": {value: jyear}}`` (e.g. NED's ``pos_bibcode``
      -> epoch of the survey the position was copied from); values missing from the
      table give None (unknown), never a guessed epoch. A lookup value
      ``[earliest, latest]`` (a survey observed over years) also gives None here;
      :func:`resolve_epoch_range` returns its span. With ``pm_columns`` and
      ``pm_epoch`` (SIMBAD), rows that have a proper motion are at ``pm_epoch`` (SIMBAD
      propagates those to J2000) and only the others use the lookup.
    * a per-row span ``{"span_columns": [start, end], "format": "mjd",
      "point_max_years": 0.1}``: the position was measured at some time between the
      two dates (5XMM unique sources: first and last observation of the stack). When
      the span is at most ``point_max_years`` its midpoint is the epoch; otherwise the
      epoch is unknown (None) and :func:`resolve_epoch_range` returns the span.
    """
    if epoch_spec is None:
        return None
    if isinstance(epoch_spec, (int, float)) and not isinstance(epoch_spec, bool):
        return float(epoch_spec)
    if isinstance(epoch_spec, Mapping):
        if epoch_spec.get("span_columns"):
            span = _row_span(row, epoch_spec, epoch_format, columns)
            if span is None:
                return None
            limit = _to_float(epoch_spec.get("point_max_years"))
            return 0.5 * (span[0] + span[1]) if limit is not None and span[1] - span[0] <= limit else None
        if _row_has_pm(row, epoch_spec):
            return _to_float(epoch_spec.get("pm_epoch"))
        value = _epoch_lookup(row, epoch_spec)
        return None if isinstance(value, (list, tuple)) else _to_float(value)
    names = [epoch_spec] if isinstance(epoch_spec, str) else list(epoch_spec)
    values: list[float] = []
    for name in names:
        fmt = epoch_format
        if fmt is None:
            col = find_column(columns, name)
            fmt = col.unit if col else None
        jyear = epoch_to_jyear(row_get(row, name), fmt)
        if jyear is not None:
            values.append(jyear)
    return sum(values) / len(values) if values else None


def _epoch_lookup(row: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    if not spec.get("column"):
        return None
    key = row_get(row, str(spec.get("column", "")))
    if key is None:
        return None
    return (spec.get("values") or {}).get(str(key).strip())


def _row_has_pm(row: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    cols = [str(c) for c in (spec.get("pm_columns") or [])]
    return bool(cols) and spec.get("pm_epoch") is not None and all(_to_float(row_get(row, c)) is not None for c in cols)


def _row_span(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    epoch_format: str | None,
    columns: Sequence[ColumnMeta] | None,
) -> tuple[float, float] | None:
    names = [str(c) for c in spec.get("span_columns") or []]
    if len(names) != 2:
        return None
    years: list[float] = []
    for name in names:
        fmt = spec.get("format") or epoch_format
        if fmt is None:
            col = find_column(columns, name)
            fmt = col.unit if col else None
        jyear = epoch_to_jyear(row_get(row, name), fmt)
        if jyear is None:
            return None
        years.append(jyear)
    return min(years), max(years)


def _as_span(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = _to_float(value[0]), _to_float(value[1])
        if lo is not None and hi is not None:
            return min(lo, hi), max(lo, hi)
    return None


def resolve_epoch_range(
    row: Mapping[str, Any],
    catalog: CatalogDefinition,
    columns: Sequence[ColumnMeta] | None = None,
) -> tuple[float, float] | None:
    """(earliest, latest) Julian year of a row whose exact epoch is unknown but bounded.

    * An epoch lookup whose value is ``[earliest, latest]`` (NED positions copied from
      2MASS or WISE, each observed once at an unrecorded date within the survey), or
      the lookup's ``default_range`` for a row missing from the table (SIMBAD rows
      without a proper motion: coordinates of the original measurement, epoch unknown).
    * A per-row span (``span_columns``) longer than ``point_max_years`` (5XMM).
    * A catalog with no epoch (``epoch: null``) and ``parameters.epoch_range``: every row
      was observed at some unrecorded time in that span (Chandra CSC master sources,
      LoTSS, NVSS, the 1RXS catalogue).
    * A row of a per-row-epoch catalog (an epoch column: AllWISE ``w1mjdmean``, 2MASS
      ``jdate``) whose value is missing: it was still observed during the survey, so the
      catalog's ``parameters.epoch_range`` bounds it (AllWISE leaves w1mjdmean empty for
      some saturated stars, e.g. Aldebaran; its span is 2010.0-2011.2).
    """
    if isinstance(catalog.epoch, Mapping):
        spec = catalog.epoch
        if spec.get("span_columns"):
            span = _row_span(row, spec, catalog.epoch_format, columns)
            limit = _to_float(spec.get("point_max_years"))
            if span is None or (limit is not None and span[1] - span[0] <= limit):
                return None
            return span
        if _row_has_pm(row, spec):
            return None
        value = _epoch_lookup(row, spec)
        if value is None:
            return _as_span(spec.get("default_range"))
        return _as_span(value)
    if catalog.epoch is None or isinstance(catalog.epoch, (str, list, tuple)):
        return _as_span(catalog.parameters.get("epoch_range"))
    return None


# ---------------------------------------------------------------------------
# Astrometric Propagation & Epoch-Aware Cones
# ---------------------------------------------------------------------------


def haversine_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcseconds (haversine, numerically stable at small angles)."""
    phi1, phi2 = math.radians(dec1), math.radians(dec2)
    dphi = phi2 - phi1
    dlmb = math.radians((ra2 - ra1 + 180.0) % 360.0 - 180.0)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def offset_radec(ra: float, dec: float, east_arcsec: float, north_arcsec: float) -> tuple[float, float]:
    """Move (ra, dec) by a tangent-plane offset (inverse gnomonic projection; valid at the poles)."""
    xi = math.radians(east_arcsec / 3600.0)
    eta = math.radians(north_arcsec / 3600.0)
    ra0, dec0 = math.radians(ra), math.radians(dec)
    denom = math.cos(dec0) - eta * math.sin(dec0)
    new_ra = ra0 + math.atan2(xi, denom)
    new_dec = math.atan2(math.sin(dec0) + eta * math.cos(dec0), math.hypot(xi, denom))
    ra_deg = math.degrees(new_ra) % 360.0
    return (0.0 if ra_deg >= 360.0 else ra_deg), math.degrees(new_dec)


def tangent_offset_arcsec(ra0: float, dec0: float, ra: float, dec: float) -> tuple[float, float]:
    """(east, north) gnomonic offset of (ra, dec) from (ra0, dec0) in arcsec (inverse of :func:`offset_radec`)."""
    a0, d0, a, d = map(math.radians, (ra0, dec0, ra, dec))
    cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    cos_c = max(cos_c, 1e-12)  # > 90 deg away: no meaningful tangent-plane offset
    xi = math.cos(d) * math.sin(a - a0) / cos_c
    eta = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cos_c
    return math.degrees(xi) * 3600.0, math.degrees(eta) * 3600.0


def propagate_radec(
    ra: float, dec: float, pm_ra_masyr: float, pm_dec_masyr: float, from_epoch: float, to_epoch: float
) -> tuple[float, float]:
    """Linear proper-motion propagation (pm_ra includes cos dec) from one Julian year to another.

    Parallax and radial-velocity (perspective acceleration) terms are ignored. They are
    NOT negligible for the nearest fast stars: for Barnard's star (plx 547 mas, RV -110
    km/s, dmu/dt ~ 1.3 mas/yr^2) rigorous space motion (astropy ``apply_space_motion``)
    differs from this linear model by ~0.16" from J2016.0 to J2000.0, ~0.28" to J1995
    and ~0.40" to J1991. The linear model is kept deliberately: SIMBAD, the Exoplanet
    Archive and Sesame propagate Gaia positions linearly too, so linear propagation
    reproduces their published epoch positions to a few mas (a rigorous model would put
    those ~0.16" off). Matching radii of arcseconds absorb the difference, but Gaussian
    confidences against sub-0.1" catalogs can be a few sigma low for such stars.

    The annual PARALLAX is not part of this barycentric motion: it is removed separately
    from single-epoch (apparent) catalog positions of the target in
    :func:`source_position_at` (``parallax_offset_arcsec``). It dominates for the nearest
    stars -- Proxima's 2MASS position (2000.194) is 0.71" off the linear track and 0.05"
    off after removing the 768 mas parallax.
    """
    dt = to_epoch - from_epoch
    if dt == 0.0:
        return ra, dec
    return offset_radec(ra, dec, pm_ra_masyr * dt / 1000.0, pm_dec_masyr * dt / 1000.0)


def earth_barycentric_au(jyear: float) -> tuple[float, float, float]:
    """Approximate barycentric position of the Earth (AU, ICRS axes) at Julian year ``jyear``.

    Low-precision solar coordinates (Astronomical Almanac, section C): the Sun's
    geocentric position is good to ~0.01 deg / 1e-4 AU and the Sun-barycentre offset
    (<= 0.01 AU) is ignored, so the result is within ~0.01 AU of the JPL ephemeris --
    a parallax error below 1% (8 mas for Proxima Cen), with no ephemeris download.
    """
    n = (jyear - 2000.0) * 365.25  # days from J2000.0 (TT ~ UTC at this precision)
    mean_longitude = math.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = mean_longitude + math.radians(1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    distance = 1.00014 - 0.01671 * math.cos(anomaly) - 0.00014 * math.cos(2 * anomaly)
    obliquity = math.radians(23.439 - 0.0000004 * n)
    sun_x = distance * math.cos(ecliptic_longitude)
    sun_y = distance * math.cos(obliquity) * math.sin(ecliptic_longitude)
    sun_z = distance * math.sin(obliquity) * math.sin(ecliptic_longitude)
    return -sun_x, -sun_y, -sun_z


def parallax_offset_arcsec(ra: float, dec: float, parallax_mas: float, jyear: float) -> tuple[float, float]:
    """(east, north) annual-parallax displacement (arcsec) of a star seen from the Earth at ``jyear``.

    Observed (geocentric) position = barycentric position + this offset:
    d(alpha*) = p (X sin a - Y cos a), d(delta) = p (X cos a sin d + Y sin a sin d - Z cos d)
    with (X, Y, Z) the Earth's barycentric position in AU.
    """
    x, y, z = earth_barycentric_au(jyear)
    p = parallax_mas / 1000.0
    a, d = math.radians(ra), math.radians(dec)
    east = p * (x * math.sin(a) - y * math.cos(a))
    north = p * (x * math.cos(a) * math.sin(d) + y * math.sin(a) * math.sin(d) - z * math.cos(d))
    return east, north


def closest_span_epoch(
    ra: float, dec: float, target_ra: float, target_dec: float, target_epoch: float,
    pm: tuple[float, float], span: tuple[float, float],
) -> float:
    """Julian year in ``span`` at which a target moving with ``pm`` passes closest to (ra, dec).

    The target track is linear in the tangent plane at its mid-span position, so the
    closest approach is the projection of the offset on the motion, clamped to the span.
    """
    lo, hi = span
    mid = 0.5 * (lo + hi)
    cra, cdec = propagate_radec(target_ra, target_dec, pm[0], pm[1], target_epoch, mid)
    east, north = tangent_offset_arcsec(cra, cdec, ra, dec)
    vx, vy = pm[0] / 1000.0, pm[1] / 1000.0
    speed2 = vx * vx + vy * vy
    if speed2 == 0.0:
        return mid
    return min(hi, max(lo, mid + (east * vx + north * vy) / speed2))


# Metadata flags set on every CatalogSource by the providers (from the catalog definition).
META_CATALOG_HAS_PM = "catalog_has_proper_motions"
META_SINGLE_EPOCH = "single_epoch_positions"
# Propagation methods that place a row with the TARGET's motion (identity test).
TARGET_PM_METHODS = frozenset({"target_pm", "target_pm_parallax", "target_pm_span"})


def source_position_at(
    source: CatalogSource,
    epoch: float | None,
    fallback_pm: tuple[float, float] | None = None,
    anchor: tuple[float, float] | None = None,
    parallax_mas: float | None = None,
) -> tuple[float, float, str]:
    """Position of ``source`` at Julian year ``epoch`` and how it was obtained.

    * "source_pm": the row's own proper motion.
    * "stationary": the row has an exact epoch but no proper motion in a catalog that
      publishes proper motions (a Gaia 2-parameter solution, a SIMBAD Gaia position
      without pm): such rows are field objects measured without detectable motion, so
      they stay where they are -- they are never moved with the target's motion (that
      would drag faint field stars from the target's old position into the cone).
    * "target_pm": a row of a catalog without proper motions (2MASS, AllWISE, ...) moved
      with ``fallback_pm``, the target's motion -- where the row would be now if it
      were the target. "target_pm_parallax": the same after removing the target's
      annual parallax (``parallax_mas``) at the row's epoch, for catalogs whose rows are
      single-epoch apparent positions (``metadata['single_epoch_positions']``).
    * "target_pm_span": a row with an unknown epoch inside a known span
      (``source.epoch_range``) moved with ``fallback_pm`` from the epoch in that span at
      which the target -- at ``anchor`` (ra, dec) at ``epoch`` -- passes closest to it:
      the row is compatible with the target if it lies near any point of the target's
      track during the span. Without an anchor the mid-span epoch is used.
    * "none": the catalog position unchanged.
    """
    if epoch is None:
        return source.ra, source.dec, "none"
    own_pm = source.proper_motion_ra_masyr is not None and source.proper_motion_dec_masyr is not None
    if source.epoch is None:
        span = source.epoch_range
        if span is None or fallback_pm is None or own_pm:
            return source.ra, source.dec, "none"
        if anchor is not None:
            from_epoch = closest_span_epoch(source.ra, source.dec, anchor[0], anchor[1], epoch, fallback_pm, span)
        else:
            from_epoch = 0.5 * (span[0] + span[1])
        ra, dec = propagate_radec(source.ra, source.dec, fallback_pm[0], fallback_pm[1], from_epoch, epoch)
        return ra, dec, "target_pm_span"
    if own_pm:
        ra, dec = propagate_radec(source.ra, source.dec, source.proper_motion_ra_masyr,  # type: ignore[arg-type]
                                  source.proper_motion_dec_masyr, source.epoch, epoch)  # type: ignore[arg-type]
        return ra, dec, "source_pm"
    if fallback_pm is None:
        return source.ra, source.dec, "none"
    if (source.metadata or {}).get(META_CATALOG_HAS_PM):
        return source.ra, source.dec, "stationary"
    ra, dec, method = source.ra, source.dec, "target_pm"
    if parallax_mas and (source.metadata or {}).get(META_SINGLE_EPOCH):
        east, north = parallax_offset_arcsec(ra, dec, parallax_mas, source.epoch)
        ra, dec = offset_radec(ra, dec, -east, -north)  # observed -> barycentric at source.epoch
        method = "target_pm_parallax"
    ra, dec = propagate_radec(ra, dec, fallback_pm[0], fallback_pm[1], source.epoch, epoch)
    return ra, dec, method


def target_pm_separation_arcsec(target: Target, source: CatalogSource) -> float | None:
    """Separation if a "stationary" row moved with the target (could it be the target?)."""
    pm = target.proper_motion
    if target.epoch is None or pm is None or source.epoch is None:
        return None
    ra, dec = propagate_radec(source.ra, source.dec, pm[0], pm[1], source.epoch, target.epoch)
    return haversine_arcsec(target.ra, target.dec, ra, dec)


def epoch_separation_arcsec(target: Target, source: CatalogSource) -> tuple[float, str]:
    """Separation between the target and a source brought to the target's epoch.

    For a row with an unknown epoch in a known span this is the closest approach of the
    target's track during that span (see :func:`source_position_at`).
    """
    ra, dec, method = source_position_at(source, target.epoch, target.proper_motion, (target.ra, target.dec),
                                         target.parallax_mas)
    return haversine_arcsec(target.ra, target.dec, ra, dec), method


def catalog_epoch_range(catalog: CatalogDefinition) -> tuple[float, float] | None:
    """(earliest, latest) Julian year of a catalog's positions, or None when unknown.

    A fixed ``epoch`` gives a single epoch. Per-row epoch catalogs, and catalogs whose
    rows were observed at unrecorded times within a span (``epoch: null``), declare the
    span in ``parameters["epoch_range"]``; cones are then padded by pm x span / 2.
    """
    if isinstance(catalog.epoch, (int, float)) and not isinstance(catalog.epoch, bool):
        return float(catalog.epoch), float(catalog.epoch)
    span = catalog.parameters.get("epoch_range")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        lo, hi = _to_float(span[0]), _to_float(span[1])
        if lo is not None and hi is not None:
            return min(lo, hi), max(lo, hi)
    return None


@dataclass(slots=True)
class ConePlan:
    """Where and how wide a catalog cone must be so the target is found at the catalog's epoch.

    ``radius_arcsec`` is what is sent to the archive; rows farther than
    ``requested_radius_arcsec`` from the target (after epoch correction) are pad rows
    and are never counted as results. ``row_limit`` is the number of rows to ask the
    archive for: ``max_rows`` for an exact cone, more for a widened one (a fast mover
    can lie beyond the N rows nearest the cone centre in a crowded field).
    """

    ra: float
    dec: float
    radius_arcsec: float
    requested_radius_arcsec: float
    pad_arcsec: float = 0.0
    mode: str = "none"  # none | static_pad | target_pm | unknown_pm_pad | epoch_unknown
    warnings: list[str] = field(default_factory=list)
    row_limit: int = 200

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Field stars in catalogs with per-row proper motions are assumed to move less than this
# (0.1"/yr covers all but a tiny fraction of stars) when a cone must also cover the
# target's neighbours at the target epoch.
FIELD_PM_ALLOWANCE_ARCSEC_PER_YR = 0.1


def catalog_has_proper_motions(catalog: CatalogDefinition) -> bool:
    """True when the catalog's selected columns include per-row proper motions."""
    if "pmra" in {str(k) for k in (catalog.parameters.get("field_map") or {})}:
        return True
    names = {bare_column_name(str(c)).lower() for c in (catalog.parameters.get("columns") or [])}
    return any(FIELD_ALIASES.get(name) == "pmra" for name in names)


def epoch_pad_settings() -> tuple[float, float]:
    """(max plausible proper motion in arcsec/yr, cap on the pad in arcsec) from the environment.

    The default 10.5 arcsec/yr covers Barnard's star (10.39 arcsec/yr), the fastest known star.
    """
    max_pm = float(os.getenv("EPOCH_PAD_MAX_PM_ARCSEC_PER_YR", "10.5"))
    cap = float(os.getenv("EPOCH_PAD_MAX_ARCSEC", "300"))
    return max(0.0, max_pm), max(0.0, cap)


def padded_row_limit(max_rows: int, requested_arcsec: float, radius_arcsec: float) -> int:
    """Rows to request for a cone widened from ``requested`` to ``radius``: max_rows scaled
    by the area ratio, capped by EPOCH_PAD_MAX_ROWS (default 2000)."""
    max_rows = max(1, int(max_rows))
    if radius_arcsec <= requested_arcsec:
        return max_rows
    cap = max(max_rows, int(os.getenv("EPOCH_PAD_MAX_ROWS", "2000")))
    scaled = math.ceil(max_rows * (radius_arcsec / max(requested_arcsec, 1e-6)) ** 2)
    return min(cap, max(max_rows, scaled))


def plan_cone(catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> ConePlan:
    """Compute the cone to send to ``catalog`` (see :func:`_plan_cone`) and its row limit."""
    plan = _plan_cone(catalog, target, radius_arcsec)
    plan.row_limit = padded_row_limit(catalog.max_rows, plan.requested_radius_arcsec, plan.radius_arcsec)
    return plan


def _plan_cone(catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> ConePlan:
    """Compute the cone to send to ``catalog`` for a target at ``target.epoch``.

    * No target epoch: the requested cone, unchanged (positions are compared as given).
    * Target epoch and proper motion known: the cone is centred on the target's path at
      the middle of the catalog's epoch span and widened by half the path length; for
      catalogs with per-row proper motions it is enlarged to also enclose the target's
      position at the target epoch (its field-star neighbours).
    * Target epoch known, proper motion unknown: the cone is widened by the largest
      plausible motion over the epoch gap (capped; a warning says when the cap bites).
    * Target epoch known, catalog epoch unknown: unchanged, with a warning.
    A static ``parameters["radius_pad_arcsec"]`` is added in every case.
    """
    static_pad = max(0.0, float(catalog.parameters.get("radius_pad_arcsec") or 0.0))
    plan = ConePlan(target.ra, target.dec, radius_arcsec + static_pad, radius_arcsec, static_pad,
                    "static_pad" if static_pad else "none")
    if target.epoch is None:
        return plan
    span = catalog_epoch_range(catalog)
    if span is None:
        plan.mode = "epoch_unknown"
        plan.warnings.append(
            f"{catalog.name}: catalog epoch unknown; cone not epoch-propagated from target epoch {target.epoch:g}."
        )
        return plan
    lo, hi = span
    pm = target.proper_motion
    if pm is not None:
        mid = 0.5 * (lo + hi)
        plan.ra, plan.dec = propagate_radec(target.ra, target.dec, pm[0], pm[1], target.epoch, mid)
        half_path = math.hypot(*pm) / 1000.0 * (hi - lo) / 2.0
        extra = half_path
        if catalog_has_proper_motions(catalog):
            # Rows with their own proper motion are compared at the target epoch, so the
            # neighbours of the target's position AT target.epoch (field stars, which move
            # slowly) must be fetched too -- not only the target's track. Cover both with
            # one enclosing circle so the in-radius set does not depend on whether the
            # target's motion was given or adopted later.
            gap = max(abs(lo - target.epoch), abs(hi - target.epoch))
            allowance = FIELD_PM_ALLOWANCE_ARCSEC_PER_YR * gap
            d = haversine_arcsec(plan.ra, plan.dec, target.ra, target.dec)
            if d + allowance > half_path:
                if d + half_path <= allowance:
                    plan.ra, plan.dec, extra = target.ra, target.dec, allowance
                else:
                    extra = 0.5 * (d + allowance + half_path)
                    east, north = tangent_offset_arcsec(plan.ra, plan.dec, target.ra, target.dec)
                    frac = (extra - half_path) / d
                    plan.ra, plan.dec = offset_radec(plan.ra, plan.dec, east * frac, north * frac)
        plan.pad_arcsec = static_pad + extra
        plan.radius_arcsec = radius_arcsec + plan.pad_arcsec
        plan.mode = "target_pm"
        return plan
    gap = max(abs(lo - target.epoch), abs(hi - target.epoch))
    max_pm, cap = epoch_pad_settings()
    needed = max_pm * gap
    if needed <= 0.0:
        return plan
    pad = min(needed, cap)
    plan.pad_arcsec = static_pad + pad
    plan.radius_arcsec = radius_arcsec + plan.pad_arcsec
    plan.mode = "unknown_pm_pad"
    if needed > cap:
        plan.warnings.append(
            f"{catalog.name}: epoch gap {gap:.1f} yr needs a {needed:.0f} arcsec pad; capped at {cap:.0f} arcsec "
            f"(stars faster than {cap / gap:.2f} arcsec/yr may be missed). Supply the target proper motion."
        )
    return plan


def build_provenance(
    catalog: str,
    *,
    provider: str,
    source_id: str,
    endpoint: str | None,
    query_parameters: Mapping[str, object],
    search_radius_arcsec: float,
) -> dict[str, object]:
    """Assemble complete provenance metadata for an archive observation."""
    return {
        "catalog": catalog,
        "provider": provider,
        "source_id": source_id,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "search_radius_arcsec": search_radius_arcsec,
        "query_parameters": dict(query_parameters),
    }


# ---------------------------------------------------------------------------
# Embedded Catalog Registry
# ---------------------------------------------------------------------------

HEASARC_TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_GATOR = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

_HEASARC_ACK = (
    "This research has made use of data obtained from the High Energy Astrophysics Science Archive "
    "Research Center (HEASARC), provided by NASA's Goddard Space Flight Center."
)
_VIZIER_ACK = "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France (DOI 10.26093/cds/vizier)."
_IRSA_ACK = (
    "This research has made use of the NASA/IPAC Infrared Science Archive, which is funded by the National "
    "Aeronautics and Space Administration and operated by the California Institute of Technology."
)
_TWOMASS_ACK = (
    "This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project "
    "of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of "
    "Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation."
)
_NRAO_ACK = (
    "The National Radio Astronomy Observatory is a facility of the National Science Foundation operated under "
    "cooperative agreement by Associated Universities, Inc."
)

# NED copies positions from source catalogs; their epoch follows the catalog, not
# the J2000 equinox. Surveys with one well-defined epoch map to a Julian year; surveys
# that observed each position once at some date over several years (2MASS, WISE) map
# to their [earliest, latest] span, so a moving target is matched anywhere on its track
# during the span instead of at a guessed mid-survey date (up to ~2 yr = 20" wrong for
# Barnard's star).
NED_POSITION_EPOCHS: dict[str, float | list[float]] = {
    "2013wise.rept....1C": [2010.0, 2011.2],  # AllWISE (WISE cryo + NEOWISE post-cryo)
    "2012wise.rept....1C": [2010.0, 2010.7],  # WISE All-Sky (cryogenic mission)
    "2006AJ....131.1163S": [1997.4, 2001.2],  # 2MASS
    "2003yCat.2246....0C": [1997.4, 2001.2],  # 2MASS PSC via VizieR
    "2003tmc..book.....C": [1997.4, 2001.2],  # 2MASS explanatory supplement
    "2020yCat.1350....0G": 2016.0,  # Gaia EDR3
    "2021A&A...649A...1G": 2016.0,  # Gaia EDR3
    "2023A&A...674A...1G": 2016.0,  # Gaia DR3
    "2022yCat.1355....0G": 2016.0,  # Gaia DR3 via VizieR
    "2018yCat.1345....0G": 2015.5,  # Gaia DR2
    "2018A&A...616A...1G": 2015.5,  # Gaia DR2
    "2016A&A...595A...2G": 2015.0,  # Gaia DR1
    "1997ESASP1200.....E": 1991.25,  # Hipparcos
    "2007A&A...474..653V": 1991.25,  # Hipparcos (new reduction)
    "2000A&A...355L..27H": 2000.0,  # Tycho-2 (positions propagated to J2000)
}

# Divisor that turns NED's uncmaja/uncmina into 1-sigma per-axis errors, by pos_bibcode.
NED_ERROR_DIVISORS: dict[str, float] = {
    "2020CSC...C2..0000:": K95_1D,  # Chandra Source Catalog 2.0: per-axis 95% copied unchanged
    "2010ApJS..189...37E": K95_1D,  # Chandra Source Catalog 1 (same convention)
    "20102XMM.1.2..0000:": 2.5 * math.sqrt(2.0),  # 2XMM: 2.5 x radial 1-sigma SC_POSERR
    "2006AJ....131.1163S": 2.5,  # 2MASS (2.5 x err_maj)
    "2003yCat.2246....0C": 2.5,  # 2MASS PSC via VizieR
    "2013wise.rept....1C": 2.5,  # AllWISE (2.5 x sigra/sigdec)
    "2012wise.rept....1C": 2.5,  # WISE All-Sky
    "1998AJ....115.1693C": 2.5,  # NVSS (2.5 x 1-sigma)
}

DEFAULT_CATALOGS: dict[str, dict[str, Any]] = {
    "gaia_dr3": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "optical",
        "endpoint": "https://gea.esac.esa.int/tap-server/tap/sync",
        "table": "gaiadr3.gaia_source",
        "description": "Gaia DR3 TAP positional and astrometric search",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "source_id", "ra", "dec", "ra_error", "dec_error", "ra_dec_corr", "ref_epoch",
                "parallax", "parallax_error", "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag",
                "phot_rp_mean_mag", "ruwe", "astrometric_params_solved",
            ],
            "id_field": "source_id",
            "format": "json",
            "distance": "deg",
            "epoch_range": [2016.0, 2016.0],
        },
        "profiles": ["full", "optical", "stellar"],
        "epoch": "ref_epoch",
        "epoch_format": "jyear",
        "pos_error": {"columns": ["ra_error", "dec_error"], "units": "mas", "kind": "sigma"},
        "citation": "Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (2023A&A...674A...1G); Gaia Collaboration, Prusti et al. 2016, A&A 595, A1",
        "acknowledgement": (
            "This work has made use of data from the European Space Agency (ESA) mission Gaia "
            "(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium "
            "(DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided "
            "by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement."
        ),
        "max_rows": 200,
        "timeout_seconds": 90.0,
        "coverage": "all-sky, G < ~21",
    },
    "simbad": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "multi",
        "endpoint": "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        "table": "basic AS b LEFT OUTER JOIN allfluxes AS f ON f.oidref = b.oid",
        "description": "SIMBAD astronomical object identities, types, and measurements",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "b.main_id", "b.ra", "b.dec", "b.coo_err_maj", "b.coo_err_min", "b.coo_err_angle",
                "b.coo_qual", "b.coo_wavelength", "b.coo_bibcode", "b.otype", "b.sp_type",
                "b.rvz_redshift", "b.plx_value", "b.pmra", "b.pmdec", "f.V", "f.G", "f.K",
            ],
            "id_field": "main_id",
            "ra_field": "b.ra",
            "dec_field": "b.dec",
            "format": "json",
            "distance": "deg",
            # Cones are planned at J2000 (rows with a proper motion are propagated there by
            # SIMBAD); rows without one are checked over the lookup's default_range.
            "epoch_range": [2000.0, 2000.0],
        },
        "profiles": ["full", "optical", "stellar", "identity"],
        # SIMBAD moves a position to J2000.0 only when it has a proper motion. Other rows
        # keep the epoch of the original measurement (e.g. GES J17574778+0443542, Gaia-ESO
        # ~2013, lies on Barnard's star's track at 2013.3): the epoch then comes from
        # coo_bibcode (Gaia -> 2016.0, 2MASS -> its span) or, when unknown, the span of the
        # modern surveys most such coordinates come from.
        "epoch": {"column": "coo_bibcode", "values": NED_POSITION_EPOCHS, "pm_columns": ["pmra", "pmdec"],
                  "pm_epoch": 2000.0, "default_range": [1990.0, 2025.0]},
        "pos_error": {
            "columns": ["coo_err_maj", "coo_err_min", "coo_err_angle"],
            "units": "mas",
            "kind": "ellipse",
            # coo_err_* are the errors at the source epoch (Gaia J2016.0 for most A-quality
            # rows) although the position was moved to J2000.0: add the pm-propagation term.
            "epoch_growth": {"epoch_lookup": {"column": "coo_bibcode", "values": NED_POSITION_EPOCHS},
                             "to_epoch": 2000.0, "pm_columns": ["pmra", "pmdec"],
                             "pm_error_ratio": {"2016.0": 1.4, "2015.5": 2.1}},
            # coo_qual when coo_err_* is null. SIMBAD user guide (coordinate quality):
            # A = Hipparcos/Gaia-class with proper motion (mas), B 0.01-0.1", C 0.1-1",
            # D 1-10", E >= 10". Values are representative 1-sigma errors inside each range;
            # E is open-ended (positions rounded to 0.1-1 deg occur, e.g. HVC 104.2-48-168
            # at exactly (0.0, 13.0)): 1' is listed, and the association treats E rows as
            # extended, non-point identities (crossmatch.is_extended_identity).
            "fallback_column": "coo_qual",
            "fallback_values": {"A": 0.005, "B": 0.05, "C": 0.5, "D": 5.0, "E": 60.0},
        },
        "citation": "Wenger et al. 2000, A&AS 143, 9 (2000A&AS..143....9W)",
        "acknowledgement": "This research has made use of the SIMBAD database, operated at CDS, Strasbourg, France.",
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "all-sky literature compilation",
    },
    "ned": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "extragalactic",
        "endpoint": "https://ned.ipac.caltech.edu/tap/sync",
        "table": "NEDTAP.objdir",
        "description": "NASA/IPAC Extragalactic Database object directory",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "prefname", "ra", "dec", "uncmaja", "uncmina", "uncposa", "pos_bibcode",
                "prefphytype", "z", "zunc", "zflag", "n_gphot",
            ],
            "id_field": "prefname",
            "format": "json",
            "distance": "deg",
            # NED positions are equinox J2000 but carry the epoch of the survey they
            # were copied from (e.g. AllWISE 2010.0-2011.2): epoch or span comes from pos_bibcode.
            "epoch_range": [1990.0, 2016.0],
        },
        "profiles": ["full", "optical", "extragalactic", "identity"],
        # Unknown bibcodes give epoch None (unknown), never a made-up 2000.0.
        "epoch": {"column": "pos_bibcode", "values": NED_POSITION_EPOCHS},
        # NED's uncertainty ellipse is 2.5 x the source catalog's 1-sigma errors
        # (verified: uncmaja/uncmina = 2.500 x AllWISE sigra/sigdec for HD 209458 and
        # HD 106785, and 2.5 x 2MASS err_maj for V0376 Peg), so divide by 2.5.
        # The convention depends on the catalogue the position came from (pos_bibcode):
        # CSC 2.0 ellipses are copied unchanged (95% per axis: 1.96; e.g. 2CXO
        # J123049.8+122327 0.775/0.752 in both), 2XMM gives 2.5 x the RADIAL 1-sigma
        # SC_POSERR (2XMM J123043.6+122147: 2.73 vs 1.1), so the per-axis sigma is
        # uncmaja / (2.5 sqrt 2). Unknown bibcodes keep 2.5, flagged 'assumed'.
        "pos_error": {"columns": ["uncmaja", "uncmina", "uncposa"], "units": "arcsec", "kind": "ellipse", "divisor": 2.5,
                      "divisor_lookup": {"column": "pos_bibcode", "values": NED_ERROR_DIVISORS}},
        "citation": "NASA/IPAC Extragalactic Database (NED), DOI 10.26132/NED1",
        "acknowledgement": (
            "This research has made use of the NASA/IPAC Extragalactic Database (NED), which is funded by the "
            "National Aeronautics and Space Administration and operated by the California Institute of Technology."
        ),
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "all-sky extragalactic compilation",
    },
    "exoplanet_archive": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "exoplanet",
        "endpoint": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        "table": "pscomppars",
        "description": "NASA Exoplanet Archive Planetary Systems Composite Parameters (one row per planet)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "pl_name", "hostname", "ra", "dec", "sy_dist", "sy_vmag", "sy_kmag", "sy_gaiamag",
                "sy_pmra", "sy_pmdec", "pl_orbper", "pl_rade", "pl_bmasse", "discoverymethod",
                "disc_year", "gaia_dr3_id",
            ],
            "id_field": "pl_name",
            "format": "json",
            "distance": "deg",
            # Positions are J2015.5 (Gaia DR2/DR3 propagated; verified to < 1 mas).
            # High proper-motion hosts are found through the epoch-aware cone (target
            # epoch + proper motion), not a fixed pad that leaks out-of-cone rows.
        },
        "profiles": ["full", "exoplanet", "stellar"],
        "epoch": 2015.5,
        "pos_error": {"default_arcsec": 0.1, "kind": "sigma", "columns": []},
        "citation": "Akeson et al. 2013, PASP 125, 989 (2013PASP..125..989A); pscomppars DOI 10.26133/NEA13",
        "acknowledgement": (
            "This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute "
            "of Technology, under contract with the National Aeronautics and Space Administration under the "
            "Exoplanet Exploration Program."
        ),
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "known exoplanet host systems",
    },
    "vizier_2mass_reference": {
        "enabled": False,
        "provider": "tap",
        "wavelength": "infrared",
        "endpoint": VIZIER_TAP,
        "table": '"II/246/out"',
        "description": "VizieR TAP 2MASS reference catalog",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                '"2MASS"', "RAJ2000", "DEJ2000", "errMaj", "errMin", "errPA", "Jmag", "e_Jmag",
                "Hmag", "e_Hmag", "Kmag", "e_Kmag", "Qflg", "JD",
            ],
            "id_field": '"2MASS"',
            "ra_field": "RAJ2000",
            "dec_field": "DEJ2000",
            "format": "json",
            "distance": "deg",
            "epoch_range": [1997.4, 2001.2],  # 2MASS observed June 1997 - Feb 2001
            "single_epoch_positions": True,
        },
        "profiles": ["full", "infrared", "stellar"],
        "epoch": "JD",
        "epoch_format": "jd",
        "pos_error": {"columns": ["errMaj", "errMin", "errPA"], "units": "arcsec", "kind": "ellipse"},
        "citation": "Skrutskie et al. 2006, AJ 131, 1163 (2006AJ....131.1163S); VizieR II/246 (Cutri et al. 2003)",
        "acknowledgement": _TWOMASS_ACK + " " + _VIZIER_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "all-sky",
    },
    "twomass_psc": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "infrared",
        "catalog": "fp_psc",
        "table": "fp_psc",
        "endpoint": IRSA_TAP,
        "description": "2MASS Point Source Catalog (IRSA TAP, Gator fallback)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "designation", "ra", "dec", "err_maj", "err_min", "err_ang", "j_m", "j_msigcom",
                "h_m", "h_msigcom", "k_m", "k_msigcom", "ph_qual", "cc_flg", "ext_key", "jdate",
            ],
            "id_field": "designation",
            "format": "votable",
            # IRSA DISTANCE() returns arcsec and cannot be multiplied (ORA-01722).
            "distance": "arcsec",
            "fallback": {"provider": "irsa_gator", "endpoint": IRSA_GATOR, "catalog": "fp_psc"},
            "epoch_range": [1997.4, 2001.2],
            # Each position is one observation at jdate: an apparent (parallax-displaced) position.
            "single_epoch_positions": True,
        },
        "profiles": ["full", "infrared", "stellar"],
        "epoch": "jdate",
        "epoch_format": "jd",
        "pos_error": {"columns": ["err_maj", "err_min", "err_ang"], "units": "arcsec", "kind": "ellipse"},
        "citation": "Skrutskie et al. 2006, AJ 131, 1163 (2006AJ....131.1163S); DOI 10.26131/IRSA2",
        "acknowledgement": _TWOMASS_ACK + " " + _IRSA_ACK,
        "max_rows": 200,
        "timeout_seconds": 90.0,
        "coverage": "all-sky",
    },
    "allwise": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "infrared",
        "catalog": "allwise_p3as_psd",
        "table": "allwise_p3as_psd",
        "endpoint": IRSA_TAP,
        "description": "AllWISE mid-infrared source catalog (IRSA TAP, Gator fallback)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "designation", "ra", "dec", "sigra", "sigdec", "sigradec", "w1mpro", "w1sigmpro",
                "w2mpro", "w2sigmpro", "w3mpro", "w3sigmpro", "w4mpro", "w4sigmpro", "cc_flags",
                "ext_flg", "ph_qual", "w1mjdmean", "tmass_key",
            ],
            "id_field": "designation",
            "format": "votable",
            "distance": "arcsec",
            "fallback": {"provider": "irsa_gator", "endpoint": IRSA_GATOR, "catalog": "allwise_p3as_psd"},
            "epoch_range": [2010.0, 2011.2],  # WISE cryogenic + NEOWISE post-cryo (Jan 2010 - Feb 2011)
        },
        "profiles": ["full", "infrared", "stellar"],
        "epoch": "w1mjdmean",
        "epoch_format": "mjd",
        "pos_error": {"columns": ["sigra", "sigdec"], "units": "arcsec", "kind": "sigma"},
        "citation": "Wright et al. 2010, AJ 140, 1868 (2010AJ....140.1868W); Mainzer et al. 2011, ApJ 731, 53; DOI 10.26131/IRSA1",
        "acknowledgement": (
            "This publication makes use of data products from the Wide-field Infrared Survey Explorer, which is a "
            "joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California "
            "Institute of Technology, and NEOWISE, which is a project of the Jet Propulsion Laboratory/California "
            "Institute of Technology. WISE and NEOWISE are funded by the National Aeronautics and Space Administration. "
        ) + _IRSA_ACK,
        "max_rows": 200,
        "timeout_seconds": 90.0,
        "coverage": "all-sky",
    },
    "panstarrs_dr2": {
        "enabled": True,
        "provider": "mast",
        "wavelength": "optical",
        "endpoint": "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json",
        "description": "Pan-STARRS DR2 mean object catalog (MAST catalogs API)",
        "query_method": "positional API",
        "units": "degrees",
        "parameters": {
            "columns": [
                "objID", "objName", "raMean", "decMean", "raMeanErr", "decMeanErr", "epochMean",
                "nDetections", "qualityFlag", "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag",
                "rMeanPSFMagErr", "iMeanPSFMag", "iMeanPSFMagErr", "zMeanPSFMag", "zMeanPSFMagErr",
                "yMeanPSFMag", "yMeanPSFMagErr", "iMeanKronMag", "distance",
            ],
            "id_field": "objID",
            "ra_field": "raMean",
            "dec_field": "decMean",
            # Drop 0-1 detection artefacts (e.g. 8 of 9 rows around 3C 273).
            "filters": {"nDetections.gt": 1},
            "null_sentinels": [-999, -999.0, "NULL"],
            "epoch_range": [2009.5, 2015.0],  # PS1 3pi survey 2010-2014
            # MAST metadata mislabels these (raMeanErr 'deg' is arcsec; 'distance'
            # labelled arcsec is in degrees): corrected before anything reads them.
            "column_units": {"raMeanErr": "arcsec", "decMeanErr": "arcsec", "distance": "deg"},
        },
        "profiles": ["full", "optical", "extragalactic"],
        "epoch": "epochMean",
        "epoch_format": "mjd",
        # raMeanErr/decMeanErr are arcsec (metadata wrongly says deg/mas) and are
        # underestimated for bright sources: add a 15 mas systematic floor.
        "pos_error": {"columns": ["raMeanErr", "decMeanErr"], "units": "arcsec", "kind": "sigma", "systematic_arcsec": 0.015},
        "citation": "Chambers et al. 2016, arXiv:1612.05560; Flewelling et al. 2020, ApJS 251, 7; DOI 10.17909/s0zg-jx37",
        "acknowledgement": (
            "The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive have been made possible through "
            "contributions by the Institute for Astronomy, the University of Hawaii, the Pan-STARRS Project Office, "
            "the Max-Planck Society and its participating institutes (full text at panstarrs.stsci.edu)."
        ),
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "Dec > -30",
    },
    "sdss": {
        "enabled": True,
        "provider": "sdss",
        "wavelength": "optical",
        "endpoint": "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
        "table": "PhotoPrimary",
        "description": "Sloan Digital Sky Survey DR18 photometry (+photo-z, spec-z) via SkyServer SqlSearch",
        "query_method": "SQL (fGetNearbyObjEq, radius in arcmin)",
        "units": "arcminutes",
        "parameters": {
            "columns": [
                "p.ra", "p.dec", "p.raErr", "p.decErr", "p.mjd", "p.type", "p.clean", "p.mode",
                "p.psfMag_u", "p.psfMag_g", "p.psfMag_r", "p.psfMag_i", "p.psfMag_z", "p.psfMagErr_r",
                "p.modelMag_r", "pz.z AS photoz", "pz.zErr AS photozErr", "s.z AS specz",
                "s.zErr AS speczErr", "s.class AS specClass",
            ],
            "id_field": "objID",
            "field_map": {"redshift": "specz", "object_type": "specClass"},
            "epoch_range": [1998.5, 2009.6],  # SDSS imaging 1998-2009
            "single_epoch_positions": True,  # one imaging run per PhotoPrimary object (mjd)
        },
        "profiles": ["full", "optical", "extragalactic", "spectroscopy"],
        "epoch": "mjd",
        "epoch_format": "mjd",
        "pos_error": {"columns": ["raErr", "decErr"], "units": "arcsec", "kind": "sigma", "systematic_arcsec": 0.04},
        "citation": "Almeida et al. 2023, ApJS 267, 44 (SDSS DR18); York et al. 2000, AJ 120, 1579; Beck et al. 2016, MNRAS 460, 1371",
        "acknowledgement": (
            "Funding for the Sloan Digital Sky Survey V has been provided by the Alfred P. Sloan Foundation, the "
            "Heising-Simons Foundation, the National Science Foundation, and the Participating Institutions "
            "(see sdss.org/collaboration/citing-sdss)."
        ),
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "SDSS imaging footprint (~14,500 deg^2); holes around very bright/extended galaxies (e.g. M87)",
    },
    "first": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "radio",
        "table": "first",
        "endpoint": HEASARC_TAP,
        "description": "FIRST radio catalogue (1.4 GHz, HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "name", "ra", "dec", "flux_20_cm", "int_flux_20_cm", "flux_20_cm_error",
                "fit_major_axis", "fit_minor_axis", "sidelobe_prob", "mean_epoch",
            ],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            "epoch_range": [1993.0, 2011.3],
            "single_epoch_positions": True,
        },
        "profiles": ["full", "radio", "extragalactic"],
        "epoch": "mean_epoch",
        "epoch_format": "mjd",
        "pos_error": {
            "columns": ["fit_major_axis", "fit_minor_axis", "flux_20_cm", "flux_20_cm_error"],
            "units": ["arcsec", "arcsec", None, None],
            "kind": "first",
            "floor_arcsec": 0.1,
        },
        "citation": "Becker, White & Helfand 1995, ApJ 450, 559; Helfand, White & Becker 2015, ApJ 801, 26",
        "acknowledgement": _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "~10,575 deg^2 (SDSS area, both Galactic caps)",
    },
    "nvss": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "radio",
        "table": "nvss",
        "endpoint": HEASARC_TAP,
        "description": "NVSS radio catalogue (1.4 GHz, Dec > -40, HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "name", "ra", "dec", "flux_20_cm", "flux_20_cm_error", "ra_error", "dec_error",
                "major_axis", "minor_axis", "position_angle", "pol_flux",
            ],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            # NVSS observed 1993.7-1996.8 (+1997 fill-in); the catalogue has no per-source date.
            "epoch_range": [1993.7, 1997.3],
        },
        "profiles": ["full", "radio", "extragalactic"],
        "epoch": None,
        # ra_error is in SECONDS OF TIME, dec_error in arcsec; both 1-sigma.
        "pos_error": {"columns": ["ra_error", "dec_error"], "units": ["s_ra", "arcsec"], "kind": "sigma"},
        "citation": "Condon et al. 1998, AJ 115, 1693 (1998AJ....115.1693C)",
        "acknowledgement": _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "Dec > -40",
    },
    "vlass": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "radio",
        "table": '"J/ApJ/914/42/table5"',
        "endpoint": VIZIER_TAP,
        "description": "VLASS Epoch 1 radio catalogue (3 GHz, Bruzewski et al. 2021, VizieR TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["Name", "RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "Flux", "e_Flux", "Fpk", "Type", "Obs", "Flag"],
            "id_field": "Name",
            "ra_field": "RAJ2000",
            "dec_field": "DEJ2000",
            "format": "json",
            "distance": "deg",
            "epoch_range": [2017.5, 2019.8],  # VLASS epoch 1
            # Flag 2 = 'Redundant' (ReadMe: 0 Alone, 1 Useful, 2 Redundant): a duplicate of a
            # Flag 1 row from an overlapping image (144,503 of 2,232,726 rows). Removed on
            # receipt (the query itself is unchanged so recorded requests stay valid).
            "exclude_values": {"Flag": [2]},
            "single_epoch_positions": True,
        },
        "profiles": ["full", "radio", "extragalactic"],
        "epoch": "Obs",
        "epoch_format": "jd",
        # e_RAJ2000/e_DEJ2000 are in DEGREES; add ~0.5" Quick-Look astrometric systematics.
        "pos_error": {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": "deg", "kind": "sigma", "systematic_arcsec": 0.5},
        "citation": "Lacy et al. 2020, PASP 132, 035001; Bruzewski et al. 2021, ApJ 914, 42 (2021ApJ...914...42B)",
        "acknowledgement": _NRAO_ACK + " " + _VIZIER_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "Dec > -40; very bright sources (e.g. 3C 273) may be absent",
    },
    "lotss": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "radio",
        "table": '"J/A+A/707/A198/lotssdr3"',
        "endpoint": VIZIER_TAP,
        "description": "LoTSS-DR3 radio catalogue (144 MHz, Shimwell et al. 2026, VizieR TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "Source", "RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "SpeakTot", "e_SpeakTot",
                "Speak", "e_Speak", "Maj", "Min", "SCode", "Res",
            ],
            "id_field": "Source",
            "ra_field": "RAJ2000",
            "dec_field": "DEJ2000",
            "format": "json",
            "distance": "deg",
            # LoTSS pointings observed from 2014.3 on (DR3 up to ~2024); no per-source date.
            "epoch_range": [2014.3, 2024.5],
        },
        "profiles": ["full", "radio", "extragalactic"],
        "epoch": None,
        "pos_error": {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": "arcsec", "kind": "sigma", "systematic_arcsec": 0.2},
        "citation": "Shimwell et al. 2026, A&A 707, A198 (LoTSS-DR3); Shimwell et al. 2022, A&A 659, A1 (DR2)",
        "acknowledgement": (
            "LOFAR is the Low Frequency Array designed and constructed by ASTRON (see the LoTSS data policy for the "
            "full acknowledgement). "
        ) + _VIZIER_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "northern sky; hole around Virgo A (M87 absent)",
    },
    "lotss_dr2": {
        "enabled": False,
        "provider": "tap",
        "wavelength": "radio",
        "table": '"J/A+A/659/A1/catalog"',
        "endpoint": VIZIER_TAP,
        "description": "LoTSS-DR2 radio catalogue (144 MHz, mostly Dec > +25, VizieR TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["Source", "RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "SpeakTot", "e_SpeakTot", "Maj", "Min", "SCode"],
            "id_field": "Source",
            "ra_field": "RAJ2000",
            "dec_field": "DEJ2000",
            "format": "json",
            "distance": "deg",
            "epoch_range": [2014.3, 2020.2],  # DR2 pointings observed 2014-2020
        },
        "profiles": ["full", "radio", "extragalactic"],
        "epoch": None,
        "pos_error": {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": "arcsec", "kind": "sigma", "systematic_arcsec": 0.2},
        "citation": "Shimwell et al. 2022, A&A 659, A1 (2022A&A...659A...1S)",
        "acknowledgement": _VIZIER_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "5,634 deg^2, mostly Dec > +25",
    },
    "rosat": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "xray",
        "table": "rass2rxs",
        "endpoint": HEASARC_TAP,
        "description": "Second ROSAT All-Sky Survey point source catalogue (2RXS, HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "name", "ra", "dec", "x_pixel_error", "y_pixel_error", "count_rate", "count_rate_error",
                "detection_likelihood", "exposure", "source_extent", "source_quality_flag", '"time"', "end_time",
            ],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            "epoch_range": [1990.5, 1991.1],  # RASS Aug 1990 - Jan 1991
            "single_epoch_positions": True,  # each source scanned over ~2 days
        },
        "profiles": ["full", "xray", "high-energy"],
        "epoch": ["time", "end_time"],
        "epoch_format": "mjd",
        # x/y_pixel_error are 1-sigma errors in 45" image pixels (HEASARC rass2rxs). Empirical
        # calibration against Gaia (Freund et al. 2022, A&A 664, A105, Eq. 2):
        # sigma = 1.22 * sqrt((45 XERR)^2/2 + (45 YERR)^2/2 + (3")^2).
        "pos_error": {"columns": ["x_pixel_error", "y_pixel_error"], "units": 45.0, "kind": "sigma",
                      "systematic_arcsec": 3.0, "scale": 1.22},
        "citation": "Boller et al. 2016, A&A 588, A103 (2016A&A...588A.103B); positional errors: Freund et al. 2022, A&A 664, A105",
        "acknowledgement": _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "all-sky (1990-1991)",
    },
    "rosat_bsc": {
        "enabled": False,
        "provider": "tap",
        "wavelength": "xray",
        "table": "rassbsc",
        "endpoint": HEASARC_TAP,
        "description": "1RXS ROSAT Bright Source Catalogue (legacy names; HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["name", "ra", "dec", "positional_error", "count_rate", "count_rate_error", "hardness_ratio_1",
                        "hardness_ratio_2", "exposure", "source_extent", "detect_likelihood"],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            "epoch_range": [1990.5, 1991.1],  # RASS Aug 1990 - Jan 1991; no per-source date
        },
        "profiles": ["full", "xray", "high-energy"],
        "epoch": None,
        "pos_error": {"columns": ["positional_error"], "units": "arcsec", "kind": "sigma"},
        "citation": "Voges et al. 1999, A&A 349, 389 (1999A&A...349..389V)",
        "acknowledgement": _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "all-sky bright sources",
    },
    "chandra": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "xray",
        "table": "csc",
        "endpoint": HEASARC_TAP,
        "description": "Chandra Source Catalog 2.1 master sources (HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "name", "ra", "dec", "error_ellipse_r0", "error_ellipse_r1", "error_ellipse_angle",
                "significance", "b_flux_ap", "b_flux_ap_lo", "b_flux_ap_hi", "hardness_ratio_hs",
                "hardness_ratio_hm", "var_flag", "conf_flag", "extent_flag", "sat_src_flag",
                "pileup_flag", "acis_num", "hrc_num",
            ],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            # Master sources combine observations from 1999.6 to the end of 2021 and the
            # HEASARC csc table has no per-source time: the epoch is unknown in this span.
            "epoch_range": [1999.5, 2022.0],
        },
        "profiles": ["full", "xray", "high-energy"],
        "epoch": None,
        # CXC: err_ellipse_r0/r1 'correspond to the 95% confidence intervals along
        # these axes' (per-axis, i.e. 1.96 sigma) and already include the CSC 2.1
        # absolute-astrometry term (0.29" 95% per axis, in quadrature): divide by 1.96.
        "pos_error": {"columns": ["error_ellipse_r0", "error_ellipse_r1", "error_ellipse_angle"], "units": "arcsec", "kind": "ellipse95axis"},
        "citation": "Evans et al. 2024, ApJS 274, 22 (CSC 2.1); Evans et al. 2010, ApJS 189, 37; DOI 10.25574/csc2.1",
        "acknowledgement": (
            "This research has made use of data obtained from the Chandra Source Catalog, provided by the Chandra "
            "X-ray Center (CXC). "
        ) + _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "pointed observations only (~1% of sky)",
    },
    "xmm": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "xray",
        "table": "xmmssc",
        "endpoint": HEASARC_TAP,
        "description": "XMM-Newton Serendipitous Source Catalog (5XMM, unique sources, HEASARC TAP)",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": [
                "srcid", "name", "ra", "dec", "error_radius", "error_ell_major", "error_ell_minor",
                "ep_flux", "ep_flux_error", "ep_det_ml", "sum_flag", "n_obs", "n_contrib", "extent",
                '"time"', "end_time",
            ],
            "id_field": "name",
            "format": "votable",
            "distance": "deg",
            "epoch_range": [2000.0, 2025.0],
        },
        "profiles": ["full", "xray", "high-energy"],
        # time/end_time are the first and last observation of the STACK in which the unique
        # source was detected (Proxima Cen: 2001.61-2018.19 for all 8 nearby sources): the
        # position was measured at an unknown time in that span, not at its midpoint.
        "epoch": {"span_columns": ["time", "end_time"], "format": "mjd", "point_max_years": 0.1},
        "epoch_format": "mjd",
        # error_radius = sqrt(ra_err^2 + dec_err^2) (statistical) + 0.88" per-axis systematic.
        "pos_error": {"columns": ["error_radius"], "units": "arcsec", "kind": "radial", "systematic_arcsec": 0.88},
        "citation": "Webb et al. 2020, A&A 641, A136 (2020A&A...641A.136W); Webb, Traulsen et al. 2026 (5XMM-DR15)",
        "acknowledgement": (
            "This research has made use of data obtained from the 5XMM serendipitous source catalog compiled by the "
            "XMM-Newton Survey Science Center, the XMM2ATHENA project and in collaboration with the XMM-Newton SOC. "
        ) + _HEASARC_ACK,
        "max_rows": 200,
        "timeout_seconds": 60.0,
        "coverage": "pointed observations only (~3% of sky)",
    },
}

KNOWN_PROVIDERS = frozenset({"tap", "irsa_gator", "mast", "sdss", "heasarc_xamin"})
_DEFINITION_FIELDS = (
    "endpoint", "table", "catalog", "description", "query_method", "units", "epoch", "epoch_format",
    "citation", "acknowledgement", "coverage",
)


def catalog_from_dict(name: str, entry: Mapping[str, Any]) -> CatalogDefinition:
    """Build a CatalogDefinition from a registry dictionary (YAML or embedded)."""
    extra = {key: entry.get(key) for key in _DEFINITION_FIELDS}
    epoch = extra.pop("epoch")
    if isinstance(epoch, (list, tuple)):
        epoch = [str(item) for item in epoch]
    timeout = entry.get("timeout_seconds")
    return CatalogDefinition(
        name=name,
        provider=str(entry.get("provider", "unknown")),
        wavelength=str(entry.get("wavelength", "unknown")),
        enabled=bool(entry.get("enabled", True)),
        parameters=dict(entry.get("parameters", {}) or {}),
        profiles=tuple(str(item) for item in entry.get("profiles", ()) or ()),
        epoch=epoch,
        pos_error=dict(entry.get("pos_error", {}) or {}),
        max_rows=int(entry["max_rows"]) if entry.get("max_rows") is not None else 200,
        timeout_seconds=float(timeout) if timeout is not None else None,
        **extra,
    )


def _plausible_epoch_value(value: Any) -> bool:
    span = _as_span(value)
    years = list(span) if span is not None else [_to_float(value)]
    return all(y is not None and 1800 <= y <= 2200 for y in years)


def validate_catalog_definition(catalog: CatalogDefinition, *, providers: frozenset[str] | set[str] = KNOWN_PROVIDERS) -> list[str]:
    """Return a list of human-readable problems with a catalog definition (empty = valid)."""
    problems: list[str] = []
    name = catalog.name
    if catalog.provider not in providers:
        problems.append(f"{name}: unknown provider '{catalog.provider}'")
    if catalog.endpoint and not str(catalog.endpoint).startswith(("https://", "http://")):
        problems.append(f"{name}: endpoint must be an http(s) URL")
    if catalog.max_rows <= 0 or catalog.max_rows > 100_000:
        problems.append(f"{name}: max_rows must be in 1..100000")
    if catalog.timeout_seconds is not None and catalog.timeout_seconds <= 0:
        problems.append(f"{name}: timeout_seconds must be positive")
    if isinstance(catalog.epoch, (int, float)) and not 1800 <= float(catalog.epoch) <= 2200:
        problems.append(f"{name}: epoch {catalog.epoch} is not a plausible Julian year")
    if isinstance(catalog.epoch, Mapping) and catalog.epoch.get("span_columns") is not None:
        span_cols = catalog.epoch.get("span_columns")
        if not isinstance(span_cols, (list, tuple)) or len(span_cols) != 2 or not all(span_cols):
            problems.append(f"{name}: epoch span_columns must name the [start, end] columns")
        limit = catalog.epoch.get("point_max_years")
        if limit is not None and ((_to_float(limit) is None) or float(limit) < 0):
            problems.append(f"{name}: epoch point_max_years must be a non-negative number")
        if catalog.epoch.get("format") not in (None, "jyear", "mjd", "jd"):
            problems.append(f"{name}: epoch span format must be jyear, mjd or jd")
    elif isinstance(catalog.epoch, Mapping):
        values = catalog.epoch.get("values")
        if not catalog.epoch.get("column") or not isinstance(values, Mapping):
            problems.append(f"{name}: epoch lookup needs 'column' and a 'values' mapping")
        elif not all(_plausible_epoch_value(v) for v in values.values()):
            problems.append(f"{name}: epoch lookup values must be Julian years or [earliest, latest] spans")
        default_range = catalog.epoch.get("default_range")
        if default_range is not None and (_as_span(default_range) is None or not _plausible_epoch_value(default_range)):
            problems.append(f"{name}: epoch default_range must be [earliest, latest] Julian years")
        if catalog.epoch.get("pm_epoch") is not None and not _plausible_epoch_value(catalog.epoch.get("pm_epoch")):
            problems.append(f"{name}: epoch pm_epoch must be a Julian year")
    exclude = catalog.parameters.get("exclude_values")
    if exclude is not None and (not isinstance(exclude, Mapping)
                                or not all(isinstance(v, (list, tuple)) for v in exclude.values())):
        problems.append(f"{name}: exclude_values must map column names to lists of values")
    if isinstance(catalog.epoch, (int, float)) and not isinstance(catalog.epoch, bool) and catalog.parameters.get("epoch_range"):
        problems.append(
            f"{name}: a fixed epoch with an epoch_range is contradictory; use epoch null for rows observed "
            "at unrecorded times within the range"
        )
    span = catalog.parameters.get("epoch_range")
    if span is not None and (
        not isinstance(span, (list, tuple)) or len(span) != 2
        or any(_to_float(v) is None or not 1800 <= float(v) <= 2200 for v in span)
    ):
        problems.append(f"{name}: epoch_range must be [earliest, latest] Julian years")
    column_units = catalog.parameters.get("column_units")
    if column_units is not None and not isinstance(column_units, Mapping):
        problems.append(f"{name}: column_units must map column names to units")
    if catalog.epoch_format not in (None, "jyear", "mjd", "jd"):
        problems.append(f"{name}: epoch_format must be jyear, mjd or jd")
    spec = catalog.pos_error or {}
    if spec:
        kind = spec.get("kind", "sigma")
        if kind not in POS_ERROR_KINDS:
            problems.append(f"{name}: unknown pos_error kind '{kind}'")
        if spec.get("divisor") is not None and not ((_to_float(spec.get("divisor")) or 0.0) > 0):
            problems.append(f"{name}: pos_error divisor must be a positive number")
        if spec.get("scale") is not None and not ((_to_float(spec.get("scale")) or 0.0) > 0):
            problems.append(f"{name}: pos_error scale must be a positive number")
        lookup = spec.get("divisor_lookup")
        if lookup is not None and (
            not isinstance(lookup, Mapping) or not lookup.get("column") or not isinstance(lookup.get("values"), Mapping)
            or not all((_to_float(v) or 0.0) > 0 for v in lookup["values"].values())
        ):
            problems.append(f"{name}: pos_error divisor_lookup needs a column and positive divisors")
        cols = spec.get("columns") or []
        if not cols and spec.get("default_arcsec") is None:
            problems.append(f"{name}: pos_error needs columns or default_arcsec")
        units = spec.get("units", spec.get("unit"))
        unit_list = list(units) if isinstance(units, (list, tuple)) else [units] * len(cols)
        if kind in ELLIPSE_KINDS and len(unit_list) > 2:
            unit_list = unit_list[:2]  # third column is a position angle (degrees)
        for unit in unit_list:
            if unit is None or (isinstance(unit, str) and unit.lower() in SPECIAL_ERROR_UNITS):
                continue
            if arcsec_per_unit(unit) is None:
                problems.append(f"{name}: pos_error unit '{unit}' is not an angle")
        selected = {bare_column_name(c).lower() for c in (catalog.parameters.get("columns") or [])}
        if catalog.provider in {"tap", "mast", "sdss"} and selected:
            for col in cols:
                if col.lower() not in selected:
                    problems.append(f"{name}: pos_error column '{col}' is not in the selected columns")
    if catalog.provider == "tap" and not catalog.table:
        problems.append(f"{name}: TAP catalogs need a table")
    fmt = catalog.parameters.get("format")
    if fmt not in (None, "json", "votable", "csv"):
        problems.append(f"{name}: unsupported format '{fmt}'")
    distance = catalog.parameters.get("distance")
    if distance not in (None, "deg", "arcsec", "none"):
        problems.append(f"{name}: distance must be deg, arcsec or none")
    return problems


def registry_strict_from_env() -> bool:
    return os.getenv("CATALOG_REGISTRY_STRICT", "false").strip().lower() in {"1", "true", "yes", "on"}


class CatalogRegistry:
    """Catalog registry managing enabled astronomy catalogs with optional YAML override.

    A registry file that exists but cannot be parsed raises :class:`RegistryError`
    (it never silently falls back to the embedded defaults). A configured path that
    does not exist falls back to the defaults with a logged warning.
    """

    def __init__(self, registry_path: str | os.PathLike[str] | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else None
        self._catalogs: dict[str, CatalogDefinition] = {}
        self.reload()

    def reload(self) -> None:
        """Load catalog definitions from the YAML file if it exists, else the embedded registry."""
        catalogs_data: dict[str, Any] = {}
        if self.registry_path and self.registry_path.exists():
            try:
                content = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.error("catalog registry %s could not be parsed: %s", self.registry_path, exc)
                raise RegistryError(f"Catalog registry {self.registry_path} could not be parsed: {exc}") from exc
            if not isinstance(content, dict) or not isinstance(content.get("catalogs", {}), dict):
                raise RegistryError(f"Catalog registry {self.registry_path} must contain a 'catalogs' mapping.")
            catalogs_data = content.get("catalogs", {})
        else:
            if self.registry_path:
                logger.warning("catalog registry %s not found; using the embedded registry", self.registry_path)
            catalogs_data = DEFAULT_CATALOGS

        self._catalogs = {}
        for name, entry in catalogs_data.items():
            if not isinstance(entry, dict):
                continue
            self._catalogs[name] = catalog_from_dict(name, entry)

    @property
    def catalogs(self) -> dict[str, CatalogDefinition]:
        return dict(self._catalogs)

    def enabled_catalogs(self) -> dict[str, CatalogDefinition]:
        return {name: cat for name, cat in self._catalogs.items() if cat.enabled}

    def get(self, name: str) -> CatalogDefinition:
        if name not in self._catalogs:
            raise KeyError(f"Catalog '{name}' not found in registry.")
        return self._catalogs[name]

    def by_profile(self, profile: str) -> dict[str, CatalogDefinition]:
        return {
            name: cat for name, cat in self.enabled_catalogs().items()
            if not cat.profiles or profile in cat.profiles
        }

    def validate(self, *, providers: frozenset[str] | set[str] = KNOWN_PROVIDERS) -> list[str]:
        """Validate every catalog definition; returns all problems found."""
        problems: list[str] = []
        for catalog in self._catalogs.values():
            problems.extend(validate_catalog_definition(catalog, providers=providers))
        return problems

    def startup_check(self, *, strict: bool | None = None, providers: frozenset[str] | set[str] = KNOWN_PROVIDERS) -> list[str]:
        """Validate at service start: log every problem; raise RegistryError when strict.

        ``strict`` defaults to the CATALOG_REGISTRY_STRICT environment variable.
        """
        problems = self.validate(providers=providers)
        for problem in problems:
            logger.warning("catalog registry problem: %s", problem)
        if problems and (registry_strict_from_env() if strict is None else strict):
            raise RegistryError("Catalog registry is invalid: " + "; ".join(problems))
        return problems
