"""Universal catalogs: discover, describe and register any VizieR table (and search the IVOA registry).

VizieR (Ochsenbein, Bauer & Marcout 2000, A&AS 143, 23; DOI 10.26093/cds/vizier) publishes tens
of thousands of catalogues; TAPVizieR's TAP_SCHEMA.tables listed 64,531 tables on 2026-09-28.
This module turns any of them into a crossmatchable AstroSearch catalog without hand-written
registry entries:

* **Discovery** -- :func:`search_catalogs`. Keyword search runs on the VizieR ASU metadata
  service (``viz-bin/votable?-words=...&-meta``, the same call astroquery's
  ``Vizier.find_catalogs`` makes; words are AND-ed over titles, descriptions and keywords).
  Wavelength filtering uses VizieR's ``-kw.Wavelength`` catalogue keywords (vocabulary
  verified live: Radio, Millimeter, IR, optical, UV, EUV, X-ray, Gamma-ray) and UCD filtering
  uses the ASU ``-ucd`` constraint, re-verified per table against TAPVizieR ``TAP_SCHEMA``.
  The tables of every matching catalogue come from ``TAP_SCHEMA.tables`` (IVOA TAP 1.1
  sec. 4). :func:`search_ivoa_registry` queries the IVOA Relational Registry (RegTAP 1.1,
  GAVO endpoint ``http://reg.g-vo.org/tap``; the query mirrors pyvo.registry) for cone-search
  and TAP services from every data centre.
* **Describe** -- :func:`describe_table` merges ``TAP_SCHEMA.columns`` (exact TAP column
  names, units, UCDs, datatypes), ``TAP_SCHEMA.tables`` (row count), the ASU catalogue
  metadata (bibcode, DOI, authors, wavelength keywords; the IVOA registry fills a missing
  bibcode or waveband), the ASU table metadata (VOTable ``COOSYS`` frame and ``epoch``) and,
  for per-row epochs, the span of the epoch column (MIN/MAX scan or sky sampling), and
  identifies the position (J2000/ICRS; FK4/old-equinox main positions give way to VizieR's
  computed ``_RA.icrs`` unless their own description states a J2000 equinox), identifier,
  positional-error and epoch columns.
* **Register** -- :func:`build_definition` / :func:`register_table` build a registry entry for
  the existing :class:`providers.TapProvider` (``provider: tap`` on the VizieR TAP endpoint,
  quoted table name, UCD-selected columns; names with '.' are selected under their own name
  because TAPVizieR's JSON otherwise renames them) and persist it in a user registry YAML
  (``CATALOG_REGISTRY_PATH`` or ``~/.astrosearch/catalogs.yaml``) under an inter-process lock.
  The file also carries fingerprinted copies of the built-in entries, so
  :class:`models.CatalogRegistry` (which reads any registry file as complete) still sees every
  built-in catalog, while :class:`UserCatalogRegistry` (:func:`load_registry`) merges the user
  entries over the current embedded registry and refreshes outdated copies.

UCD semantics follow the IVOA UCD1+ controlled vocabulary (v1.5): ``pos.eq.ra;meta.main`` /
``pos.eq.dec;meta.main`` main position, ``meta.id;meta.main`` main identifier,
``stat.error;pos.eq.ra|dec`` per-axis position errors, ``pos.errorEllipse`` error ellipses,
``time.epoch``/``time.start``/``time.end`` dates. Positional errors are converted to a 1-sigma
circular error with :func:`models.compute_positional_error`; the confidence level is read from
the column description (e.g. 2SXPS ``Err90``: "90% confidence, radial, assumed to be
Rayleigh-distributed" -> radius90, sigma = r90 / 2.146). Whenever a convention cannot be read
from the metadata the choice is recorded as an *assumption* in the registration, and every
choice can be overridden at registration time.
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
import ssl
import statistics
import tempfile
import threading
import time
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from xml.etree import ElementTree

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from models import (
    DEFAULT_CATALOGS,
    K95_1D,
    NED_ERROR_DIVISORS,
    VIZIER_TAP,
    AstroSearchError,
    CatalogDefinition,
    CatalogQueryError,
    CatalogRegistry,
    ColumnMeta,
    RegistryError,
    ResponseParseError,
    arcsec_per_unit,
    catalog_from_dict,
    epoch_to_jyear,
    parse_json_table,
    parse_votable_table,
    ucd_field_map,
    validate_catalog_definition,
    votable_query_status,
)

logger = logging.getLogger("astrosearch.vizier")

# ---------------------------------------------------------------------------
# Endpoints & constants
# ---------------------------------------------------------------------------

#: VizieR ASU metadata service (https://vizier.cds.unistra.fr/doc/asu-summary.htx).
VIZIER_ASU_URL = os.getenv("VIZIER_ASU_URL", "https://vizier.cds.unistra.fr/viz-bin/votable")
#: TAPVizieR synchronous endpoint (the same one the embedded registry uses).
VIZIER_TAP_URL = os.getenv("VIZIER_TAP_URL", VIZIER_TAP)
#: GAVO RegTAP endpoint (pyvo.registry's default; the service refuses HTTPS connections).
REGTAP_URL = os.getenv("REGTAP_URL", "http://reg.g-vo.org/tap/sync")
USER_AGENT = "AstroSearch-vizier/1.0 (VizieR/RegTAP metadata client)"
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("VIZIER_TIMEOUT_SECONDS", "90"))
#: Per-request budget of the *optional* IVOA registry calls (citation/waveband enrichment and
#: the registry search): a slow or blocked reg.g-vo.org must not hold a describe for 90 s.
REGTAP_TIMEOUT_SECONDS = float(os.getenv("VIZIER_REGTAP_TIMEOUT_SECONDS", "10"))
#: Budget of the *optional* lookup of the papers a registration's citation names (ADS link
#: gateway -> DOI -> doi.org CSL-JSON), stored in the entry for BibTeX (provenance.cite).
REFERENCE_LOOKUP_TIMEOUT_SECONDS = float(os.getenv("VIZIER_REFERENCE_TIMEOUT_SECONDS", "20"))
#: How long a failed RegTAP enrichment of a catalogue is remembered (no repeated waits).
REGTAP_NEGATIVE_TTL_SECONDS = float(os.getenv("VIZIER_REGTAP_NEGATIVE_TTL_SECONDS", "600"))
#: Per-attempt back-off before retrying a 5xx/429/408/network failure (attempts: HTTP_ATTEMPTS).
RETRY_BACKOFF_SECONDS = 0.5
HTTP_ATTEMPTS = 3
#: Largest Retry-After (seconds) honoured between attempts after HTTP 429/503.
RETRY_AFTER_MAX_SECONDS = 10.0
#: HTTP statuses retried like outages: 408 Request Timeout and 429 Too Many Requests.
_RETRY_STATUSES = frozenset({408, 429})
#: Overall budget of one API request (search/describe/register) before it answers 504.
ROUTE_DEADLINE_SECONDS = float(os.getenv("VIZIER_ROUTE_DEADLINE_SECONDS", "150"))
#: Lifetime of cached describe() results (VizieR metadata changes rarely).
DESCRIBE_CACHE_TTL_SECONDS = float(os.getenv("VIZIER_DESCRIBE_CACHE_TTL_SECONDS", str(6 * 3600)))
#: Parallel TAP_SCHEMA requests issued by one search (politeness bound).
TAP_CONCURRENCY = 3
#: Largest table count accepted for one catalogue prefix lookup (VizieR catalogues have at
#: most a few hundred tables; a larger answer means the id is a journal-level prefix).
MAX_TABLES_PER_CATALOG = 1000

VIZIER_ACKNOWLEDGEMENT = (
    "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France "
    "(DOI 10.26093/cds/vizier). The original description of the VizieR service was published in "
    "Ochsenbein, Bauer & Marcout 2000, A&AS 143, 23."
)

#: VizieR ``-kw.Wavelength`` vocabulary (verified live) -> AstroSearch wavelength names.
VIZIER_WAVELENGTHS: dict[str, str] = {
    "Radio": "radio", "Millimeter": "millimeter", "IR": "infrared", "optical": "optical",
    "UV": "uv", "EUV": "euv", "X-ray": "xray", "Gamma-ray": "gamma",
}
#: Accepted user spellings -> VizieR keyword.
_WAVELENGTH_INPUT: dict[str, str] = {
    "radio": "Radio", "millimeter": "Millimeter", "millimetre": "Millimeter", "mm": "Millimeter",
    "submm": "Millimeter", "sub-mm": "Millimeter", "infrared": "IR", "ir": "IR", "optical": "optical",
    "uv": "UV", "ultraviolet": "UV", "euv": "EUV", "xray": "X-ray", "x-ray": "X-ray", "x": "X-ray",
    "gamma": "Gamma-ray", "gamma-ray": "Gamma-ray", "gammaray": "Gamma-ray",
}
#: RegTAP (VOResource 1.1) waveband vocabulary for the same keywords.
_REGTAP_WAVEBAND: dict[str, str] = {
    "Radio": "radio", "Millimeter": "millimeter", "IR": "infrared", "optical": "optical",
    "UV": "uv", "EUV": "euv", "X-ray": "x-ray", "Gamma-ray": "gamma-ray",
}
#: Registry profiles given to a registered catalog, by wavelength (existing profile names).
_WAVELENGTH_PROFILES: dict[str, tuple[str, ...]] = {
    "radio": ("radio",), "millimeter": ("radio",), "infrared": ("infrared",), "optical": ("optical",),
    "uv": ("uv",), "euv": ("uv",), "xray": ("xray", "high-energy"), "gamma": ("high-energy",),
    "high-energy": ("high-energy",),
}
REGTAP_STANDARDS: dict[str, str] = {
    "conesearch": "ivo://ivoa.net/std/conesearch",
    "tap": "ivo://ivoa.net/std/tap",
}

# VizieR identifiers: 'I/355', 'J/A+A/707/A198/lotssdr3', 'B/avo.rad/wsrt', 'VII/250/2dfgrs'.
_VIZIER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+.\-]*(?:/[A-Za-z0-9_+.\-]+)+$")
_UCD_INPUT = re.compile(r"^[A-Za-z0-9_.;\-*]+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_NUMERIC_TYPES = frozenset({
    "double", "float", "real", "int", "integer", "smallint", "bigint", "long", "short", "tinyint",
    "byte", "unsignedbyte", "numeric", "decimal",
})
MAX_SELECTED_COLUMNS = 40
DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_PER_CATALOG = 60.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VizierError(AstroSearchError):
    """Base class for VizieR discovery/registration errors."""


class VizierInputError(VizierError, ValueError):
    """Malformed identifier, UCD, wavelength or override supplied by the caller."""


class VizierNotFoundError(VizierError, LookupError):
    """The table or catalogue does not exist in VizieR."""


class VizierUpstreamError(VizierError):
    """VizieR/RegTAP could not be reached or answered with an error.

    ``transient`` is True for outages (network errors, timeouts, HTTP 5xx) and False when the
    service answered but rejected the request or sent something unparseable.
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class VizierRegistrationError(VizierError, ValueError):
    """No valid catalog definition could be built (or the name is taken)."""


class VizierConflictError(VizierRegistrationError):
    """The catalog name is already registered, or the registry file is locked or held open by
    another process (HTTP 409; try again)."""


class VizierRegistryIOError(VizierError):
    """The registry file or its directory cannot be created, read or written (an OS error such
    as a path below a regular file or a read-only location; HTTP 500 with a JSON detail)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TableHit:
    """A VizieR table that matched a search."""

    table_id: str
    catalog_id: str
    description: str | None
    nrows: int | None
    catalog_title: str | None = None
    popularity: float | None = None
    wavelengths: list[str] = field(default_factory=list)
    bibcode: str | None = None
    matching_columns: list[dict[str, str]] = field(default_factory=list)
    relevance: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogInfo:
    """Catalogue-level (ASU RESOURCE) metadata."""

    catalog_id: str
    title: str | None = None
    bibcode: str | None = None
    doi: str | None = None
    creator: str | None = None
    year: str | None = None
    journal: str | None = None
    reference_url: str | None = None
    ivoid: str | None = None
    popularity: float | None = None
    wavelengths: list[str] = field(default_factory=list)
    missions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tables: list[TableHit] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegistryResource:
    """An IVOA registry resource offering cone-search and/or TAP access."""

    ivoid: str
    title: str | None
    short_name: str | None
    wavebands: list[str]
    services: dict[str, list[str]]
    reference_url: str | None = None
    vizier_catalog: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    """Outcome of :func:`search_catalogs`."""

    query: str
    ucd: str | None
    wavelength: str | None
    catalogs: list[CatalogInfo]
    tables: list[TableHit]
    total_catalog_matches: int | None
    truncated: bool
    warnings: list[str] = field(default_factory=list)
    registry: list[RegistryResource] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ColumnInfo:
    """One TAP column with its IVOA metadata."""

    name: str
    unit: str | None
    ucd: str | None
    datatype: str | None
    description: str | None
    principal: bool = False
    indexed: bool = False
    displayed: bool = False  # in VizieR's default column set (ASU FIELD display != 0)

    @property
    def numeric(self) -> bool:
        return _is_numeric(self.datatype)

    def as_meta(self) -> ColumnMeta:
        return ColumnMeta(name=self.name, unit=self.unit, ucd=self.ucd, datatype=self.datatype,
                          description=self.description)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TableDescription:
    """Everything needed to register a VizieR table, with the reasoning behind each choice."""

    table_id: str
    catalog: CatalogInfo
    description: str | None
    nrows: int | None
    columns: list[ColumnInfo]
    ra_column: str | None
    dec_column: str | None
    id_column: str | None
    frame: str | None
    position_unit_check: str | None
    pos_error: dict[str, Any]
    pos_error_columns: list[dict[str, Any]]
    epoch: Any
    epoch_format: str | None
    epoch_range: list[float] | None
    epoch_source: str | None
    single_epoch_positions: bool
    field_map: dict[str, str]
    citation: str
    wavelength: str
    assumptions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: ADQL expressions selected under an alias (Tycho-2 composite id
    #: ``"TYC1" || '-' || "TYC2" || '-' || "TYC3"``, epochs stored as offsets ``"EpRA-1990" + 1990``).
    derived_columns: dict[str, str] = field(default_factory=dict)
    #: False when optional metadata could not be fetched (the result is then not cached).
    complete: bool = True
    #: The problems (also listed in ``problems``) caused by an upstream outage rather than by the
    #: table's metadata: registering then fails with a transient VizierUpstreamError (HTTP 502).
    upstream_problems: list[str] = field(default_factory=list)

    @property
    def registrable(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["registrable"] = self.registrable
        return data


@dataclass(slots=True)
class Registration:
    """A persisted user catalog."""

    name: str
    table_id: str
    entry: dict[str, Any]
    path: str
    replaced: bool
    assumptions: list[str]
    attached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_numeric(datatype: str | None) -> bool:
    base = re.split(r"[(\s\[]", str(datatype or "").strip().lower(), maxsplit=1)[0]
    return base in _NUMERIC_TYPES


def ucd_atoms(ucd: str | None) -> list[str]:
    """UCD words, lower-cased (``'pos.eq.ra;meta.main'`` -> ``['pos.eq.ra', 'meta.main']``)."""
    return [part.strip().lower() for part in str(ucd or "").split(";") if part.strip()]


def _unquote(name: str) -> str:
    text = str(name).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].replace('""', '"')
    return text


def quote_identifier(name: str) -> str:
    """ADQL delimited identifier (always quoted: VizieR names such as ``z.obs``, ``2MASS`` and
    case-twins ``b_z``/``B_z`` in one table need it)."""
    return '"' + str(name).replace('"', '""') + '"'


def select_item(name: str) -> str:
    """ADQL select-list item that keeps ``name`` as the result column name.

    TAPVizieR's JSON output renames a selected column whose name contains '.' ('_RA.icrs' ->
    '_RA_icrs', 'z.obs' -> 'z_obs', 'GSC2.3' -> 'GSC2_3'; verified live 2026-09-29, no other
    character is rewritten), but keeps an explicit alias verbatim. Such columns are therefore
    selected as ``"_RA.icrs" AS "_RA.icrs"``, so every registry reference (ra/dec/id fields,
    positional-error, epoch and field_map columns) can use the real column name."""
    quoted = quote_identifier(name)
    return f"{quoted} AS {quoted}" if "." in str(name) else quoted


def validate_vizier_id(value: str) -> str:
    """Return a stripped VizieR catalogue/table identifier or raise :class:`VizierInputError`."""
    text = str(value or "").strip().strip('"').strip("/")
    if not _VIZIER_ID.match(text) or len(text) > 128:
        raise VizierInputError(
            f"Invalid VizieR identifier {value!r}: expected e.g. 'I/355/gaiadr3' or 'J/ApJS/255/30'."
        )
    return text


def catalog_of(table_id: str) -> str:
    """Catalogue id of a table (``'J/ApJS/255/30/comp'`` -> ``'J/ApJS/255/30'``)."""
    return table_id.rsplit("/", 1)[0]


def _check_catalog_depth(ident: str) -> None:
    """Refuse journal-level prefixes before any prefix query: a VizieR journal catalogue id is
    ``J/<journal>/<volume>/<page>`` (four parts, e.g. J/ApJS/255/30); 'J/A+A' or 'J/A+A/707'
    would match tens of thousands of tables (J/A+A alone: 22,406 tables on 2026-09-28)."""
    parts = ident.split("/")
    if parts[0].upper() == "J" and len(parts) < 4:
        raise VizierInputError(
            f"'{ident}' is a journal-level prefix, not a VizieR catalogue; give J/<journal>/<volume>/<page>"
            "[/<table>] (e.g. J/ApJS/255/30/comp) or use the search."
        )


def default_catalog_name(table_id: str) -> str:
    """Registry name for a table (``'IX/58/2sxps'`` -> ``'vizier_ix_58_2sxps'``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", table_id.lower()).strip("_")
    return ("vizier_" + slug)[:64].rstrip("_")


def normalize_wavelength(value: str | None) -> str | None:
    """VizieR ``-kw.Wavelength`` keyword for a user wavelength name (None when not given)."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text in VIZIER_WAVELENGTHS:
        return text
    key = text.lower().replace("_", "-")
    if key in _WAVELENGTH_INPUT:
        return _WAVELENGTH_INPUT[key]
    raise VizierInputError(
        f"Unknown wavelength {value!r}; use one of: radio, millimeter, infrared, optical, uv, euv, xray, gamma."
    )


#: UCD1+ words spelled with capitals (IVOA UCD list v1.5). VizieR's ASU ``-ucd`` constraint
#: and TAPVizieR's ``LIKE`` are case-sensitive and TAPVizieR has no LOWER()/ILIKE, so an
#: all-lower-case user UCD is re-cased with this table. Every canonical spelling was checked
#: to occur in TAPVizieR TAP_SCHEMA.columns (and its lower-case form not to) on 2026-09-28.
_UCD_CANONICAL_WORDS: dict[str, str] = {
    "sptype": "spType", "errorellipse": "errorEllipse", "posang": "posAng", "angsize": "angSize",
    "angdistance": "angDistance", "angresolution": "angResolution", "ir": "IR", "uv": "UV", "x-ray": "X-ray",
    "eqwidth": "eqWidth", "magfield": "magField", "airmass": "airMass", "dopplerveloc": "dopplerVeloc",
    "dopplerparam": "dopplerParam", "columndensity": "columnDensity", "emissmeasure": "emissMeasure", "sfr": "SFR",
    "rotmeasure": "rotMeasure", "stargalaxy": "starGalaxy", "halpha": "Halpha", "hbeta": "Hbeta", "hi": "HI",
    "lyalpha": "Lyalpha", "meananomaly": "meanAnomaly", "skylevel": "skyLevel", "antennatemp": "antennaTemp",
    "axisratio": "axisRatio", "impactparam": "impactParam",
}


def canonical_ucd(value: str) -> str:
    """UCD with UCD1+ capitalisation (``'src.sptype'`` -> ``'src.spType'``). Input that already
    contains a capital letter is trusted as typed."""
    if value != value.lower():
        return value
    atoms = []
    for atom in value.split(";"):
        atoms.append(".".join(_UCD_CANONICAL_WORDS.get(word, word) for word in atom.split(".")))
    return ";".join(atoms)


def _validate_ucd(value: str | None) -> str | None:
    """Validated UCD, casing preserved (VizieR's ``-ucd`` is case-sensitive: ``src.spType``
    finds I/239, ``src.sptype`` nothing); all-lower-case input is re-cased by canonical_ucd."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not _UCD_INPUT.match(text) or len(text) > 120:
        raise VizierInputError(f"Invalid UCD {value!r} (e.g. 'src.redshift', 'phot.flux.density;em.radio').")
    return canonical_ucd(text)


def ucd_matches(pattern: str, ucd: str | None) -> bool:
    """True when every word of ``pattern`` (``*`` wildcards allowed) matches a word of ``ucd``
    or a more specific child of it (``src.redshift`` matches ``src.redshift.phot``)."""
    have = ucd_atoms(ucd)
    for want in ucd_atoms(pattern):
        regex = re.compile("^" + re.escape(want).replace(r"\*", ".*") + r"(\..*)?$")
        if not any(regex.match(atom) for atom in have):
            return False
    return True


def _adql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _retry_after_seconds(value: str | None) -> float | None:
    """Seconds from an HTTP Retry-After header (delta-seconds or an HTTP date; RFC 9110 10.2.3)."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """One default SSL context per process (creating one costs ~0.6 s on Windows)."""
    return ssl.create_default_context()


class _Http:
    """Polite HTTP access (shared client when given; bounded retries on 5xx/network errors)."""

    def __init__(self, client: httpx.AsyncClient | None = None, *, timeout: float | None = None) -> None:
        self._own = client is None
        self.timeout = float(timeout or DEFAULT_TIMEOUT_SECONDS)
        self.client = client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, verify=_ssl_context(),
                                                  headers={"User-Agent": USER_AGENT})

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._own:
            await self.client.aclose()

    async def request(self, method: str, url: str, *, params: dict[str, Any] | None = None,
                      data: dict[str, Any] | None = None, service: str,
                      timeout: float | None = None) -> httpx.Response:
        """One HTTP exchange with bounded retries.

        Network errors, HTTP 5xx, 408 and 429 are retried HTTP_ATTEMPTS times (exponential
        back-off; a Retry-After header is honoured up to RETRY_AFTER_MAX_SECONDS) and then
        raise a *transient* VizierUpstreamError; timeouts are transient at once. Other
        statuses are returned to the caller (a 4xx is a rejection, never retried)."""
        budget = float(timeout or self.timeout)
        last: Exception | None = None
        rate_limited = False
        for attempt in range(HTTP_ATTEMPTS):
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            try:
                response = await self.client.request(method, url, params=params, data=data, timeout=budget,
                                                     headers={"User-Agent": USER_AGENT})
            except httpx.TimeoutException as exc:
                raise VizierUpstreamError(f"{service} timed out after {budget:g}s", transient=True) from exc
            except httpx.HTTPError as exc:
                last = exc
            else:
                code = response.status_code
                if code < 500 and code not in _RETRY_STATUSES:
                    return response
                rate_limited = code == 429
                last = VizierUpstreamError(f"{service} HTTP {code}" + (" (rate limited)" if rate_limited else ""))
                retry_after = _retry_after_seconds(response.headers.get("retry-after"))
                if retry_after is not None:
                    wait = max(wait, min(retry_after, RETRY_AFTER_MAX_SECONDS))
            if attempt < HTTP_ATTEMPTS - 1:
                await asyncio.sleep(wait)
        what = "rate limited" if rate_limited else "unavailable"
        raise VizierUpstreamError(f"{service} {what}: {last}", transient=True) from last

    async def tap(self, adql: str, *, url: str | None = None, service: str = "VizieR TAP", fmt: str = "json",
                  timeout: float | None = None):
        """Run a synchronous ADQL query; returns :class:`models.ParsedTable`."""
        form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql}
        if fmt == "json":
            form["FORMAT"] = "json"
        response = await self.request("POST", url or VIZIER_TAP_URL, data=form, service=service, timeout=timeout)
        status, message = (votable_query_status(response.content)
                           if b"QUERY_STATUS" in response.content[:20_000] else (None, None))
        if status == "ERROR" or response.status_code >= 400:
            raise VizierUpstreamError(
                f"{service} rejected the query (HTTP {response.status_code}): {message or response.text[:300]}"
            )
        try:
            if fmt == "json" and "json" in response.headers.get("content-type", ""):
                return parse_json_table(response.content)
            return parse_votable_table(response.content)
        except (ResponseParseError, CatalogQueryError, ValueError) as exc:
            raise VizierUpstreamError(f"{service} response could not be parsed: {exc}") from exc

    async def asu(self, params: dict[str, Any]) -> ElementTree.Element:
        """GET the VizieR ASU VOTable service and return the parsed XML root."""
        response = await self.request("GET", VIZIER_ASU_URL, params=params, service="VizieR ASU")
        if response.status_code >= 400:
            raise VizierUpstreamError(f"VizieR ASU HTTP {response.status_code}: {response.text[:300]}")
        try:
            return ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise VizierUpstreamError(f"VizieR ASU returned malformed XML: {exc}") from exc


# ---------------------------------------------------------------------------
# ASU metadata parsing
# ---------------------------------------------------------------------------


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local(child.tag) == name]


def _description(element: ElementTree.Element) -> str | None:
    for child in element:
        if _local(child.tag) == "DESCRIPTION":
            text = " ".join((child.text or "").split())
            return text or None
    return None


def _strip_prefix(value: str | None, prefix: str) -> str | None:
    if not value:
        return None
    return value[len(prefix):] if value.lower().startswith(prefix) else value


def parse_catalog_resource(resource: ElementTree.Element) -> CatalogInfo | None:
    """Catalogue metadata from one ASU ``<RESOURCE type="meta">`` element."""
    catalog_id = resource.get("name")
    if not catalog_id:
        return None
    title = None
    for child in resource:
        if _local(child.tag) == "DESCRIPTION":
            # A single-table catalogue answers with the table's RESOURCE whose description is
            # '<catalogue title>' and '<table title>' on two lines: the first is the catalogue's.
            lines = [" ".join(line.split()) for line in (child.text or "").splitlines() if line.strip()]
            title = lines[0] if lines else None
            break
    info = CatalogInfo(catalog_id=catalog_id, title=title)
    for item in _children(resource, "INFO"):
        name, value = item.get("name") or "", item.get("value")
        if value is None:
            continue
        if name == "cites" and value.lower().startswith("bibcode:"):
            info.bibcode = info.bibcode or value[len("bibcode:"):]
        elif name == "citation":
            info.doi = _strip_prefix(value, "doi:")
        elif name == "creator":
            info.creator = value
        elif name == "original_date":
            info.year = value
        elif name == "journal":
            info.journal = value
        elif name == "reference_url":
            info.reference_url = value
        elif name == "ivoid":
            info.ivoid = value
        elif name == "ipopu":
            info.popularity = _float(value)
        elif name == "-kw.Wavelength":
            info.wavelengths.append(value)
        elif name == "-kw.Mission":
            info.missions.append(value)
        elif name == "-kw.Astronomy":
            info.keywords.append(value)
    return info


def _asu_messages(root: ElementTree.Element) -> tuple[list[str], int | None, bool]:
    """(warnings, total matching catalogues, truncated) from the top-level ASU INFO elements."""
    warnings: list[str] = []
    total: int | None = None
    truncated = False
    for item in root.iter():
        if _local(item.tag) != "INFO":
            continue
        name, value = item.get("name") or "", (item.get("value") or "").strip()
        if name == "Warning":
            match = re.match(r"List of (\d+) matching catalogues truncated to (\d+)", value)
            if match:
                total, truncated = int(match.group(1)), True
            elif value.startswith("STOP, Max. number of RESOURCE"):
                truncated = True
            elif value.startswith("can't find table or catalogue") or value == "remove -kw":
                continue  # ASU first tries every word as a catalogue name; not a user-facing problem
            else:
                warnings.append(value)
        elif name == "Error" and value and not value.startswith(("--POSTGRES", "Report from")):
            if _SQL_NOISE.search(value):
                internal = "VizieR reported an internal database error; the catalogue list may be incomplete."
                if internal not in warnings:
                    warnings.append(internal)
            else:
                warnings.append(f"VizieR: {value}")
    return warnings, total, truncated


#: Fragments of VizieR's server-side SQL error dumps ('ERROR:  syntax error at or near "COMMIT"',
#: 'LINE 1: SELECT catid FROM METAcat ...', '^') -- reported once as a generic warning.
_SQL_NOISE = re.compile(r"^(?:ERROR:|LINE \d+:|\^)|\bCOMMIT\b|\bMETAcat\b|syntax error")


def asu_error_message(root: ElementTree.Element) -> str | None:
    """The error VizieR reported inside an HTTP-200 ASU answer (``<INFO name="Error">`` or
    ``QUERY_STATUS=ERROR``), or None. VizieR answers failures of its metadata service this way
    rather than with an HTTP error status."""
    for item in root.iter():
        if _local(item.tag) != "INFO":
            continue
        name = (item.get("name") or "").strip()
        value = (item.get("value") or "").strip()
        if name == "Error" or (name == "QUERY_STATUS" and value.upper() == "ERROR"):
            text = " ".join((item.text or "").split())
            message = value if name == "Error" else (text or value)
            return message or "error"
    return None


def parse_table_meta(root: ElementTree.Element) -> dict[str, Any]:
    """ASU ``-meta.all`` table metadata: COOSYS systems and the FIELD -> COOSYS references,
    plus ``error`` (the message of an HTTP-200 error answer, else None)."""
    coosys: dict[str, dict[str, str | None]] = {}
    fields: dict[str, dict[str, str | None]] = {}
    table_description: str | None = None
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "COOSYS" and element.get("ID"):
            coosys[element.get("ID") or ""] = {
                "system": element.get("system"), "equinox": element.get("equinox"), "epoch": element.get("epoch"),
            }
        elif tag == "FIELD" and element.get("name"):
            fields[element.get("name") or ""] = {
                "ref": element.get("ref"), "unit": element.get("unit"), "ucd": element.get("ucd"),
                "datatype": element.get("datatype"), "xtype": element.get("xtype"),
                "display": element.get("display"),
            }
        elif tag == "TABLE" and table_description is None:
            table_description = _description(element)
    return {"coosys": coosys, "fields": fields, "description": table_description, "error": asu_error_message(root)}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset({"the", "of", "and", "a", "an", "in", "for", "with", "catalog", "catalogue", "survey"})


def _query_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9+\-.]+", text.lower()) if w not in _STOPWORDS]


def _relevance(words: Sequence[str], *texts: str | None) -> float:
    if not words:
        return 0.0
    haystack = " ".join(t or "" for t in texts).lower()
    tokens = set(re.findall(r"[a-z0-9+\-.]+", haystack))
    return sum(1.0 for w in words if w in tokens or w in haystack) / len(words)


async def _gather_or_cancel(*awaitables: Awaitable[Any]) -> list[Any]:
    """Run awaitables concurrently; on the first failure cancel (and await) the others so no
    orphan request keeps retrying after the caller has already answered."""
    tasks = [asyncio.ensure_future(aw) for aw in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        return [task.result() for task in tasks]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _bounded(semaphore: asyncio.Semaphore, awaitable: Awaitable[Any]) -> Any:
    async with semaphore:
        return await awaitable


async def _tables_of_catalogs(http: _Http, catalog_ids: Sequence[str]) -> dict[str, list[TableHit]]:
    """TAP_SCHEMA.tables rows of the given catalogues, grouped by catalogue id.

    The prefix clauses are sent in chunks of 50 (at most TAP_CONCURRENCY queries at a time):
    one query OR-ing 500 LIKE clauses took ~20 s live."""
    grouped: dict[str, list[TableHit]] = {cid: [] for cid in catalog_ids}
    if not catalog_ids:
        return grouped
    semaphore = asyncio.Semaphore(TAP_CONCURRENCY)
    queries = []
    for start in range(0, len(catalog_ids), 50):
        chunk = catalog_ids[start:start + 50]
        clauses = " OR ".join(f"table_name LIKE {_adql_string(chr(34) + cid + '/%')}" for cid in chunk)
        queries.append(_bounded(semaphore, http.tap(
            f"SELECT table_name, description, nrows FROM TAP_SCHEMA.tables WHERE {clauses}")))
    for table in await _gather_or_cancel(*queries):
        for row in table.rows:
            table_id = _unquote(str(row.get("table_name") or ""))
            cid = catalog_of(table_id)
            if cid in grouped:  # LIKE's '_' wildcard may admit near-misses; keep exact prefixes only
                grouped[cid].append(TableHit(table_id=table_id, catalog_id=cid, description=row.get("description"),
                                             nrows=int(row["nrows"]) if row.get("nrows") is not None else None))
    for hits in grouped.values():
        hits.sort(key=lambda t: t.table_id)
    return grouped


def _like_prefix(ucd_atom: str) -> str:
    """Case-safe LIKE fragment of a UCD word: the text before its first capital letter (UCD1+
    top-level words are lower case). TAPVizieR has no LOWER()/ILIKE (verified live), so the
    case-insensitive comparison is done locally with :func:`ucd_matches`."""
    head = re.split(r"[A-Z*]", ucd_atom, maxsplit=1)[0]
    return head if "." in head else ucd_atom.split(".", 1)[0] + "."


async def _ucd_columns(http: _Http, table_ids: Sequence[str], ucd: str) -> dict[str, list[dict[str, str]]]:
    """Columns of ``table_ids`` whose UCD matches ``ucd`` (TAP_SCHEMA.columns), compared
    case-insensitively (``src.sptype`` finds ``src.spType``)."""
    found: dict[str, list[dict[str, str]]] = {}
    pattern = _like_prefix(str(ucd).split(";")[0].strip())
    semaphore = asyncio.Semaphore(TAP_CONCURRENCY)
    queries = []
    for start in range(0, len(table_ids), 100):
        chunk = table_ids[start:start + 100]
        names = ", ".join(_adql_string(quote_identifier(t)) for t in chunk)
        queries.append(_bounded(semaphore, http.tap(
            "SELECT table_name, column_name, ucd, unit FROM TAP_SCHEMA.columns "
            f"WHERE table_name IN ({names}) AND ucd LIKE {_adql_string('%' + pattern + '%')}")))
    for table in await _gather_or_cancel(*queries):
        for row in table.rows:
            if ucd_matches(ucd, row.get("ucd")):
                found.setdefault(_unquote(str(row["table_name"])), []).append(
                    {"column": _unquote(str(row["column_name"])), "ucd": str(row.get("ucd") or ""),
                     "unit": str(row.get("unit") or "")})
    return found


def _case_variants(word: str) -> list[str]:
    """Spellings tried by a case-sensitive LIKE (TAPVizieR has no LOWER/ILIKE): as typed,
    lower, Capitalised and UPPER ('eROSITA', 'erosita', 'Erosita', 'EROSITA')."""
    out: list[str] = []
    for variant in (word, word.lower(), word[:1].upper() + word[1:].lower(), word.upper()):
        if variant and variant not in out:
            out.append(variant)
    return out


def _description_search_adql(words: Sequence[str], limit: int) -> str:
    """TAP_SCHEMA.tables rows whose description contains every word (in one of its case variants)."""
    clauses = []
    for word in words:
        likes = " OR ".join(f"description LIKE {_adql_string('%' + v + '%')}" for v in _case_variants(word))
        clauses.append(f"({likes})")
    return (f"SELECT TOP {int(limit)} table_name, description, nrows FROM TAP_SCHEMA.tables "
            f"WHERE {' AND '.join(clauses)} ORDER BY nrows DESC")


async def _description_matches(http: _Http, text: str, limit: int) -> list[str]:
    """Catalogue ids (largest matching table first) whose TAP_SCHEMA table descriptions contain
    every query word. Complements ASU ``-words``, which answers a word that is also a catalogue
    alias with that one catalogue ('Hipparcos' -> only I/239)."""
    words = [w for w in re.findall(r"[A-Za-z0-9+\-.]+", text) if w.lower() not in _STOPWORDS and len(w) >= 2]
    if not words:
        return []
    table = await http.tap(_description_search_adql(words[:6], limit))
    ordered: list[str] = []
    for row in table.rows:
        cid = catalog_of(_unquote(str(row.get("table_name") or "")))
        if cid not in ordered:
            ordered.append(cid)
    return ordered


async def _catalog_infos(http: _Http, catalog_ids: Sequence[str]) -> list[CatalogInfo]:
    """ASU catalogue metadata of several catalogues (``-source`` takes a space-separated list,
    verified live)."""
    infos: list[CatalogInfo] = []
    for start in range(0, len(catalog_ids), 40):
        chunk = list(catalog_ids[start:start + 40])
        root = await http.asu({"-source": " ".join(chunk), "-meta": ""})
        error = asu_error_message(root)
        if error and not any(_local(r.tag) == "RESOURCE" and r.get("name") in chunk for r in root.iter()):
            raise VizierUpstreamError(f"VizieR ASU answered with an error: {error}")
        for res in root.iter():
            if _local(res.tag) == "RESOURCE" and res.get("name") in chunk:
                info = parse_catalog_resource(res)
                if info is not None:
                    infos.append(info)
    return infos


async def search_catalogs(
    query: str = "",
    *,
    ucd: str | None = None,
    wavelength: str | None = None,
    max_catalogs: int = 50,
    max_tables: int = 100,
    include_registry: bool = False,
    client: httpx.AsyncClient | None = None,
    timeout: float | None = None,
) -> SearchResult:
    """Find VizieR tables by keywords, wavelength (VizieR keyword) and/or column UCD.

    Keywords run on two services whose catalogues are merged (deduplicated by catalogue id):
    the ASU metadata service (``-words``, AND-ed by VizieR over catalogue titles, descriptions
    and keywords) and a TAP_SCHEMA.tables description search (every word, case variants), which
    recovers catalogues ASU hides when a word is also a catalogue alias ('Hipparcos' makes ASU
    answer only I/239; the table descriptions add I/337/tgas, J/ApJS/254/42, ...). At most
    ``max_catalogs`` catalogues are kept in total (the ASU matches first, then the description
    matches; ``truncated`` is set when some are left out). With ``wavelength`` the VizieR keyword
    (e.g. ``X-ray``) is added to the ASU words -- ASU drops ``-kw.Wavelength`` when combined
    with ``-words`` (it answers 'remove -kw') -- and catalogues not tagged with it are removed.
    With ``ucd``, VizieR's ``-ucd`` constraint (case-sensitive; all-lower-case input is
    re-cased by :func:`canonical_ucd`) selects catalogues, and only tables that really have a
    matching column (TAP_SCHEMA, compared case-insensitively) are kept. Tables are ranked by
    the fraction of query words in their catalogue title/description (plus half that fraction
    for the table's own description), then by VizieR's popularity index and row count.
    ``include_registry`` adds IVOA registry services (RegTAP, queried concurrently; a registry
    outage becomes a warning).
    """
    text = " ".join(str(query or "").split())
    ucd_value = _validate_ucd(ucd)
    keyword = normalize_wavelength(wavelength)
    if not text and not ucd_value and not keyword:
        raise VizierInputError("Give search keywords, a wavelength or a UCD.")
    for label, value, top in (("max_catalogs", max_catalogs, 500), ("max_tables", max_tables, 1000)):
        if not _is_number(value) or float(value) != int(value) or not 1 <= int(value) <= top:
            raise VizierInputError(f"{label} must be an integer in 1..{top}, got {value!r}")
    max_catalogs, max_tables = int(max_catalogs), int(max_tables)
    params: dict[str, Any] = {"-meta": "", "-meta.max": str(int(max_catalogs))}
    if text:
        params["-words"] = f"{text} {keyword}" if keyword else text
    elif keyword:
        params["-kw.Wavelength"] = keyword
    if ucd_value:
        params["-ucd"] = ucd_value

    warnings: list[str] = []
    async with _Http(client, timeout=timeout) as http:

        async def registry_lookup() -> list[RegistryResource] | None:
            if not (include_registry and text):
                return None
            try:
                return await _search_registry(http, text, keyword=keyword,
                                              timeout=min(REGTAP_TIMEOUT_SECONDS, http.timeout))
            except VizierUpstreamError as exc:
                warnings.append(f"IVOA registry search failed: {exc}")
                return []

        async def description_lookup() -> list[str]:
            if not text:
                return []
            try:
                return await _description_matches(http, text, limit=max(100, int(max_tables)))
            except VizierUpstreamError as exc:
                warnings.append(f"TAP_SCHEMA description search failed (ASU keyword results only): {exc}")
                return []

        async def vizier_lookup() -> tuple[list[CatalogInfo], list[TableHit], int | None, bool]:
            root, described = await _gather_or_cancel(http.asu(params), description_lookup())
            asu_warnings, total, truncated = _asu_messages(root)
            warnings.extend(asu_warnings)
            catalogs = [info for res in root.iter() if _local(res.tag) == "RESOURCE"
                        for info in [parse_catalog_resource(res)] if info is not None]
            known = {c.catalog_id for c in catalogs}
            extra_ids = [cid for cid in described if cid not in known]
            room = max(0, int(max_catalogs) - len(catalogs))
            if len(extra_ids) > room:
                truncated = True  # more table-description matches than max_catalogs leaves room for
                extra_ids = extra_ids[:room]
            # Catalogue metadata of the description matches and the tables of every catalogue are
            # fetched concurrently (catalogues later removed by the wavelength filter are dropped).
            # The description matches are a recall supplement: when their catalogue metadata
            # cannot be fetched they are dropped with a warning, like a failed description search.
            (extra_infos, extra_error), grouped = await _gather_or_cancel(
                _optional(_catalog_infos(http, extra_ids), "VizieR ASU metadata of the table-description matches"),
                _tables_of_catalogs(http, [c.catalog_id for c in catalogs] + extra_ids))
            if extra_error:
                warnings.append(f"{extra_error}; {len(extra_ids)} catalogue(s) found only by the TAP_SCHEMA description "
                                "search were left out (ASU keyword results only).")
                extra_infos = []
            catalogs.extend(extra_infos)
            if keyword:
                kept = [c for c in catalogs if keyword in c.wavelengths]
                if len(kept) < len(catalogs):
                    warnings.append(f"{len(catalogs) - len(kept)} catalogue(s) matched the words but are not tagged "
                                    f"'{keyword}' in VizieR and were removed.")
                catalogs = kept
            words = _query_words(text)
            hits: list[TableHit] = []
            for info in catalogs:
                info.tables = grouped.get(info.catalog_id, [])
                for hit in info.tables:
                    hit.catalog_title = info.title
                    hit.popularity = info.popularity
                    hit.wavelengths = list(info.wavelengths)
                    hit.bibcode = info.bibcode
                    # Catalogue match (title/id) plus a bonus when the table's own description matches too.
                    hit.relevance = round(_relevance(words, info.title, hit.description, hit.table_id, info.catalog_id)
                                          + 0.5 * _relevance(words, _table_title(hit.description), hit.table_id), 4)
                    hits.append(hit)
            if ucd_value and hits:
                matched = await _ucd_columns(http, [h.table_id for h in hits], ucd_value)
                for hit in hits:
                    hit.matching_columns = matched.get(hit.table_id, [])
                hits = [h for h in hits if h.matching_columns]
                for info in catalogs:
                    info.tables = [t for t in info.tables if t.matching_columns]
                catalogs = [c for c in catalogs if c.tables]
            return catalogs, hits, total, truncated

        (catalogs, hits, total, truncated), registry = await _gather_or_cancel(vizier_lookup(), registry_lookup())
    hits.sort(key=lambda h: (-h.relevance, -(h.popularity or 0.0), -(h.nrows or 0), h.table_id))
    return SearchResult(query=text, ucd=ucd_value, wavelength=keyword, catalogs=catalogs, tables=hits[:max_tables],
                        total_catalog_matches=total if total is not None else len(catalogs),
                        truncated=truncated or len(hits) > max_tables, warnings=warnings, registry=registry)


def _registry_adql(keywords: str, *, standards: Sequence[str], waveband: str | None, limit: int) -> str:
    """RegTAP 1.1 query (rr.resource x rr.capability x rr.interface), as pyvo.registry builds it."""
    word = _adql_string(keywords)
    ids = ", ".join(_adql_string(s) for s in standards)
    where = [
        f"c.standard_id IN ({ids})",
        "i.intf_role = 'std'",
        f"(1 = ivo_hasword(r.res_description, {word}) OR 1 = ivo_hasword(r.res_title, {word}))",
    ]
    if waveband:
        where.append(f"1 = ivo_hashlist_has(r.waveband, {_adql_string(waveband)})")
    return (
        f"SELECT TOP {int(limit)} r.ivoid, r.res_title, r.short_name, r.waveband, r.reference_url, "
        "c.standard_id, i.access_url "
        "FROM rr.resource AS r NATURAL JOIN rr.capability AS c NATURAL JOIN rr.interface AS i "
        f"WHERE {' AND '.join(where)} ORDER BY r.ivoid"
    )


async def _search_registry(http: _Http, keywords: str, *, keyword: str | None = None,
                           servicetypes: Sequence[str] = ("conesearch", "tap"), limit: int = 200,
                           timeout: float | None = None) -> list[RegistryResource]:
    unknown = [s for s in servicetypes if s not in REGTAP_STANDARDS]
    if unknown:
        raise VizierInputError(f"Unknown service type(s) {unknown}; use {sorted(REGTAP_STANDARDS)}")
    adql = _registry_adql(keywords, standards=[REGTAP_STANDARDS[s] for s in servicetypes],
                          waveband=_REGTAP_WAVEBAND.get(keyword or ""), limit=limit)
    table = await http.tap(adql, url=REGTAP_URL, service="IVOA RegTAP", fmt="votable", timeout=timeout)
    by_id: dict[str, RegistryResource] = {}
    kinds = {v: k for k, v in REGTAP_STANDARDS.items()}
    for row in table.rows:
        ivoid = str(row.get("ivoid") or "")
        res = by_id.get(ivoid)
        if res is None:
            wavebands = [w for w in str(row.get("waveband") or "").split("#") if w]
            short = row.get("short_name")
            res = RegistryResource(
                ivoid=ivoid, title=row.get("res_title"), short_name=short, wavebands=wavebands, services={},
                reference_url=row.get("reference_url"),
                vizier_catalog=str(short) if ivoid.startswith("ivo://cds.vizier/") and short else None,
            )
            by_id[ivoid] = res
        kind = kinds.get(str(row.get("standard_id") or "").lower(), str(row.get("standard_id")))
        url = row.get("access_url")
        if url and url not in res.services.setdefault(kind, []):
            res.services[kind].append(str(url))
    return list(by_id.values())


async def search_ivoa_registry(
    keywords: str,
    *,
    wavelength: str | None = None,
    servicetypes: Sequence[str] = ("conesearch", "tap"),
    limit: int = 200,
    client: httpx.AsyncClient | None = None,
    timeout: float | None = None,
) -> list[RegistryResource]:
    """Cone-search / TAP services from the IVOA registry (RegTAP 1.1) matching ``keywords``."""
    text = " ".join(str(keywords or "").split())
    if not text:
        raise VizierInputError("keywords are required for a registry search")
    async with _Http(client, timeout=timeout) as http:
        return await _search_registry(http, text, keyword=normalize_wavelength(wavelength),
                                      servicetypes=servicetypes, limit=limit)


# ---------------------------------------------------------------------------
# Description: column roles
# ---------------------------------------------------------------------------


#: Old-equinox positions named or described as such: RAB1950, RA1855 (BD), RA1900 (I/128),
#: RAB2000 (NGC 2000.0, FK4 B2000), '(B1950)', 'Equinox=B1950', 'FK4'. A mere year in the
#: description is not enough ('Ep=1991.25' is an epoch, not an equinox).
_OLD_EQUINOX_NAME = re.compile(r"B(?:1[89]|20)\d\d$|(?<![0-9])1[89]\d\d$", re.IGNORECASE)
_OLD_EQUINOX_TEXT = re.compile(
    r"\bB(?:1[89]|20)\d\d(?:\.\d+)?\b|\bFK4\b|\(\s*B\s*\)|\(\s*1[89]\d\d(?:\.0+)?\s*\)"
    r"|\bequinox\s*[=:]?\s*B?1[89]\d\d|\b(?:right ascension|declination|RA|Dec?)\s+1[89]\d\d\b",
    re.IGNORECASE)
_EQUATORIAL_OK = re.compile(r"j2000|icrs|fk5", re.IGNORECASE)


#: Epoch statements removed before looking for an equinox ('mean epoch=1996.6', 'Ep=J2000').
_EPOCH_PHRASE = re.compile(r"\b(?:mean\s+|central\s+)?ep(?:och)?\s*[=:]?\s*[JB]?\d{4}(?:\.\d+)?", re.IGNORECASE)
#: An explicit J2000/ICRS equinox or frame in a position column's description.
_J2000_STATED = re.compile(r"\bequinox\s*[=:]?\s*J?2000(?:\.0*)?\b|\bJ2000(?:\.0*)?\b|\bICRS\b|\bFK5\b",
                           re.IGNORECASE)


def _stated_j2000(col: ColumnInfo) -> str | None:
    """The J2000/ICRS equinox a position column's own description states, else None
    (J/AJ/122/1486/ccd RA1996: 'Right ascension (mean epoch=1996.6, equinox J2000.0)'). Epoch
    statements are ignored ('Ep=J2000' is an epoch, not an equinox)."""
    text = _EPOCH_PHRASE.sub(" ", _clean(col.description))
    if _OLD_EQUINOX_TEXT.search(text):
        return None
    match = _J2000_STATED.search(text)
    return match.group(0) if match else None


def _old_frame(col: ColumnInfo, coosys: Mapping[str, Any] | None,
               conflicts: list[str] | None = None) -> str | None:
    """Why ``col`` is not a J2000/ICRS position (FK4 or an old equinox), else None.

    The VOTable COOSYS of the column is normally authoritative (VizieR: RA1855 -> eq_FK4
    B1855, RAB2000 -> eq_FK4 B2000); without one the column name/description is read. The
    column's own description wins when it states a J2000/ICRS equinox explicitly: VizieR infers
    the COOSYS of J/AJ/122/1486/ccd RA1996 ('mean epoch=1996.6, equinox J2000.0') from its name
    (eq_FK4 B1996), and its computed _RA.icrs, precessed from that wrong equinox, is 104" off.
    Such conflicts are appended to ``conflicts``."""
    system = str((coosys or {}).get("system") or "").strip()
    equinox = str((coosys or {}).get("equinox") or "").strip()
    stated = _stated_j2000(col)
    if stated is not None:
        doubt = None
        if system.lower() in {"eq_fk4", "fk4"}:
            doubt = f"VizieR's COOSYS says {system} equinox {equinox or 'B1950'}"
        elif system.lower() in {"eq_fk5", "fk5"} and equinox and not re.fullmatch(r"J?2000(?:\.0*)?", equinox):
            doubt = f"VizieR's COOSYS says FK5 equinox {equinox}"
        elif _OLD_EQUINOX_NAME.search(col.name):
            doubt = f"its name ends in an old-equinox year ('{_OLD_EQUINOX_NAME.search(col.name).group(0)}')"  # type: ignore[union-attr]
        if doubt and conflicts is not None:
            conflicts.append(f"{col.name}: {doubt}, but its description states '{stated}' "
                             f"('{_clean(col.description)[:80]}'); the description is trusted")
        return None
    system = str((coosys or {}).get("system") or "").strip()
    equinox = str((coosys or {}).get("equinox") or "").strip()
    if system.lower() in {"eq_fk4", "fk4"}:
        return f"an FK4 position (COOSYS {system}, equinox {equinox or 'B1950'})"
    if system.lower() in {"eq_fk5", "fk5"} and equinox and not re.fullmatch(r"J?2000(?:\.0*)?", equinox):
        return f"an FK5 position at equinox {equinox} (COOSYS)"
    match = _OLD_EQUINOX_NAME.search(col.name) or _OLD_EQUINOX_TEXT.search(_clean(col.description))
    if match:
        return f"an old-equinox position ('{match.group(0).strip()}')"
    return None


def _clean(text: str | None) -> str:
    """Column description with VizieR markup removed: VizieR writes Greek letters and symbols
    in braces ('3{sigma}', '{mu}m', '{eta} Cha'), which would hide 'sigma' from the regexes."""
    return re.sub(r"\{([^{}]*)\}", r"\1", str(text or ""))


def _is_degrees(col: ColumnInfo) -> bool:
    """True when the column is in degrees, or has no unit (checked later against the data).

    ADQL POINT()/CONTAINS() need decimal degrees, so a position in seconds of time
    (J/A+A/657/A4/stars 'RAS', unit 's'), hours or radians cannot be cone-searched."""
    if _is_integer(col.datatype):
        return False  # sexagesimal parts ('DEd', unit deg, SMALLINT) are not positions
    unit = (col.unit or "").strip()
    if not unit:
        return True
    factor = arcsec_per_unit(unit)
    return factor is not None and math.isclose(factor, 3600.0, rel_tol=1e-9)


def _position_reject_reason(col: ColumnInfo) -> str:
    if _is_integer(col.datatype):
        return f"{col.name} (integer {col.datatype}: a sexagesimal part)"
    return f"{col.name} (unit '{col.unit}', not degrees)"


_INTEGER_TYPES = frozenset({"int", "integer", "smallint", "bigint", "long", "short", "tinyint", "byte",
                            "unsignedbyte"})


def _is_integer(datatype: str | None) -> bool:
    base = re.split(r"[(\s\[]", str(datatype or "").strip().lower(), maxsplit=1)[0]
    return base in _INTEGER_TYPES


def _position_candidates(columns: Sequence[ColumnInfo], axis: str,
                         rejected: list[ColumnInfo] | None = None) -> list[ColumnInfo]:
    """Numeric pos.eq.<axis> columns in degrees: meta.main first, then the other columns named
    or described as J2000/ICRS/FK5 (VizieR's computed ``_RA.icrs``/``RAJ2000``). The frame
    of each candidate (FK4/old equinox) is checked by the caller against the COOSYS.

    Columns skipped only because their unit is not degrees are appended to ``rejected``."""
    word = f"pos.eq.{axis}"
    cands = [c for c in columns if ucd_atoms(c.ucd)[:1] == [word] and c.numeric
             and "stat.error" not in ucd_atoms(c.ucd)]
    if rejected is not None:
        rejected.extend(c for c in cands if not _is_degrees(c))
    usable = [c for c in cands if _is_degrees(c)]
    main = [c for c in usable if "meta.main" in ucd_atoms(c.ucd)]
    rest = [c for c in usable if c not in main and _EQUATORIAL_OK.search(f"{c.name} {c.description or ''}")]
    return main + rest


def _pair_dec(ra: ColumnInfo, decs: Sequence[ColumnInfo]) -> ColumnInfo | None:
    if not decs:
        return None
    ra_main = "meta.main" in ucd_atoms(ra.ucd)
    guesses = {ra.name.replace("RA", "DE"), ra.name.replace("RA", "Dec"), ra.name.replace("ra", "dec"),
               ra.name.replace("RA", "DEC")}
    for dec in decs:
        if dec.name in guesses:
            return dec
    for dec in decs:
        if ("meta.main" in ucd_atoms(dec.ucd)) == ra_main:
            return dec
    return decs[0]


def _angle_factor(unit: str | None) -> float | None:
    return arcsec_per_unit(unit) if unit not in (None, "") else None


_PERCENT = re.compile(r"(\d{2}(?:\.\d+)?)\s*(?:%|per\s*cent)", re.IGNORECASE)
_NSIGMA = re.compile(r"(\d(?:\.\d+)?)\s*[- ]?\s*(?:sigma|σ|sig)\b", re.IGNORECASE)
_ONE_SIGMA_WORDS = re.compile(r"standard (?:error|deviation)|\brms\b|\bsigma\b|\bs\.e\.", re.IGNORECASE)
_PER_AXIS = re.compile(r"per[- ]axis|along (?:each|these|the) ax|each axis|one[- ]dimensional|1-?d\b", re.IGNORECASE)


def _confidence(text: str | None) -> tuple[str, float] | None:
    """('percent', p) / ('sigma', n) read from a column description, or None when unstated.

    VizieR markup is removed first: 'Error in RAdeg (3{sigma})' -> ('sigma', 3.0)."""
    if not text:
        return None
    text = _clean(text)
    match = _PERCENT.search(text)
    if match:
        value = float(match.group(1))
        if 50.0 <= value < 100.0:
            return "percent", value
    match = _NSIGMA.search(text)
    if match:
        return "sigma", float(match.group(1))
    if _ONE_SIGMA_WORDS.search(text):
        return "sigma", 1.0
    return None


#: 'RA', 'RAdeg', 'right ascension' and the Greek 'alpha' (FK6 'alpha*cos(delta)').
_RA_WORDS = re.compile(r"\bR\.?A\.?(?:deg|J2000|ICRS)?(?![a-z])|right ascension|\balpha\b", re.IGNORECASE)
#: Case-sensitive 'Dec'/'DE'/'delta' (the RA description 'RA*cos(dec)' must not read as Dec;
#: 'alpha*cos(delta)' is an RA error because the first axis mentioned wins).
_DEC_WORDS = re.compile(r"\bDec(?:l|\b)|\bDE(?:deg)?\b|[Dd]eclination|\b[Dd]elta\b")
_NOT_AXIS_ERROR = re.compile(r"proper motion|prop\.? mot|\bpm\b|parallax|flux|mag|velocity|/yr", re.IGNORECASE)


def _error_axis(text: str | None) -> str | None:
    """'ra' / 'dec' for an error column described as the error of RA or of Dec (the first axis
    mentioned wins: 'Mean error on RAdeg*cos(DEdeg)' is an RA error), else None."""
    clean = _clean(text)
    if not clean or _NOT_AXIS_ERROR.search(clean):
        return None
    ra, dec = _RA_WORDS.search(clean), _DEC_WORDS.search(clean)
    if ra and (dec is None or ra.start() < dec.start()):
        return "ra"
    if dec:
        return "dec"
    return None


def _one_d_divisor(percent: float) -> float:
    """Half-width of a two-sided 1-D normal interval with ``percent`` coverage, in sigma."""
    return statistics.NormalDist().inv_cdf(0.5 + percent / 200.0)


def _rayleigh_divisor(percent: float) -> float:
    """Radius of a circular 2-D normal region enclosing ``percent``, in per-axis sigma:
    sqrt(-2 ln(1 - p)) (90% -> 2.146, 95% -> 2.448)."""
    return math.sqrt(-2.0 * math.log(1.0 - percent / 100.0))


def _error_unit(col: ColumnInfo, axis: str | None) -> str | None:
    """Unit usable by compute_positional_error ('s_ra' for RA errors in seconds of time)."""
    unit = (col.unit or "").strip()
    if not unit:
        return None
    if axis == "ra" and unit.lower() in {"s", "sec"}:
        return "s_ra"
    return unit if _angle_factor(unit) is not None else None


_POSITION_WORDS = re.compile(r"\bpos(?:ition|itional|\.)?\b|\bastrometr|\bcoordinate|\blocali[sz]ation", re.IGNORECASE)
#: Wording of errors that are not positional errors. Position *angles* are included: GALEX AIS
#: e_nPA 'Position angle error in NUV (nuv_errtheta_j2000)' and DES e_PA 'Uncertainty in position
#: angle from isophotal model (ERRTHETA_IMAGE)' are in degrees and would otherwise be read as a
#: circular positional error of tens of degrees.
_NOT_POSITION = re.compile(r"flux|extent|\bext\b|size|pixel|magnitude|\bmag\b|velocity|offset from|distance|"
                           r"separation|proper motion|parallax|\bfwhm\b|major axis of the source|deconvolved|"
                           r"position[- ]angle|\bangle\b|\bP\.?A\.?\b|theta|\bposang|orientation",
                           re.IGNORECASE)
#: Position-angle wording: excluded even for columns tagged stat.error;pos.
_ANGLE_WORDS = re.compile(r"position[- ]angle|\bangle\b|\bP\.?A\.?\b|theta|\bposang|orientation", re.IGNORECASE)
#: VizieR's '[min/max]' value-range prefix of a column description ('[-90/]', '[0.4/3.3]', '[/7]').
_RANGE_ANY = re.compile(r"^\s*\[\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)?\s*/\s*"
                        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)?\s*\]")
#: Largest plausible 1-sigma/90%/95% circular positional error (degrees): BATSE-era GRB error
#: circles reach ~20 deg; a range reaching 90 deg (or 180) is an angle, not a position error.
_MAX_POSITION_ERROR_DEG = 45.0


def _implausible_position_error(col: ColumnInfo) -> str | None:
    """Why the values of an angular error column cannot be a positional error, else None: its
    VizieR '[min/max]' range admits negative values (GALEX e_nPA '[-90/]', DES e_PA '[-90/90]')
    or reaches _MAX_POSITION_ERROR_DEG."""
    match = _RANGE_ANY.match(col.description or "")
    if not match:
        return None
    lo, hi = _float(match.group(1)), _float(match.group(2))
    if lo is not None and lo < 0:
        return f"its values range down to {lo:g} (an error radius cannot be negative)"
    factor = _angle_factor(col.unit)
    if hi is not None and factor is not None and hi * factor >= _MAX_POSITION_ERROR_DEG * 3600.0:
        return f"its values reach {hi:g} {col.unit} (not a plausible positional error)"
    return None
_TOKENS = re.compile(r"[A-Za-z_][A-Za-z0-9_()+\-.]*")


def _referenced_columns(col: ColumnInfo, names: set[str]) -> set[str]:
    """Column names an error/epoch column refers to: its ``e_<name>`` stem and names quoted in
    its description ('Error on RA_pm', 'epoch-1990 of RAdeg')."""
    refs = {tok.rstrip(".,;:)") for tok in _TOKENS.findall(_clean(col.description))} & names
    if col.name.lower().startswith("e_") and col.name[2:] in names:
        refs.add(col.name[2:])
    return refs


#: One side of an asymmetric error ('1-sigma lower error on RA (RA_LOWERR)', 4FGL
#: 'Lo_Unc_Flux_Band1', 'upper error'): never a symmetric per-axis sigma on its own.
_ONE_SIDED = re.compile(
    r"\b(?:lower|upper|negative|positive|minus|plus)\b[^.;]*?\b(?:err|error|uncertaint)"
    r"|\b(?:LO|UP)_?(?:W?ERR|UNC)|LOWERR|UPERR", re.IGNORECASE)
#: A circular error described as a raw / uncalibrated fit output ranks below a calibrated
#: total (eRASS1 RADEC_ERR 'raw output from PSF fitting' vs POS_ERR '1-sigma positional uncertainty').
_RAW_ERROR = re.compile(r"\braw\b|uncorrected|uncalibrated|statistical(?: error)? only|without systematic",
                        re.IGNORECASE)


def _is_radial_error_text(text: str) -> bool:
    """True when a circular error is explicitly described as the radial combination of the two
    axes ('radial', 'sqrt(RA_ERR^2 + DEC_ERR^2)', 'quadrature sum of the RA and Dec errors').
    'Total'/'combined'/'quadrature' alone describe a statistical + systematic total (ROSAT 1RXS
    PosErr 'Total positional error (including 6" systematic error)', read as a 1-sigma error like
    the core rosat_bsc entry), not a radial error."""
    if re.search(r"\bradial\b", text, re.IGNORECASE):
        return True
    ra = re.search(r"(?<![a-z])(?:ra|alpha|right ascension)(?![a-z])", text, re.IGNORECASE)
    dec = re.search(r"(?<![a-z])(?:dec|de|delta|declination)(?![a-z])", text, re.IGNORECASE)
    both_axes = ra is not None and dec is not None
    return both_axes and bool(re.search(r"sqrt|quadrat|root[- ]sum|\bhypot", text, re.IGNORECASE))


def _upper_partner(col: ColumnInfo, by_name: Mapping[str, ColumnInfo]) -> ColumnInfo | None:
    """VizieR's upper-error column ``E_<x>`` of a lower-error column ``e_<x>`` (VizieR writes
    asymmetric errors as the pair e_<x> / E_<x>; a lone e_<x> is a symmetric mean error)."""
    if col.name.startswith("e_"):
        partner = by_name.get("E_" + col.name[2:])
        if partner is not None and partner.numeric:
            return partner
    return None


def _is_one_sided(col: ColumnInfo, by_name: Mapping[str, ColumnInfo]) -> bool:
    return (bool(_ONE_SIDED.search(_clean(col.description))) or _upper_partner(col, by_name) is not None
            or bool({"stat.max", "stat.min"} & set(ucd_atoms(col.ucd))))


def _larger_side_expr(lower: ColumnInfo, upper: ColumnInfo) -> str:
    """ADQL max(|lower|, |upper|) (TAPVizieR has neither GREATEST nor CASE; verified live):
    (|a| + |b| + ||a| - |b||) / 2."""
    a, b = f"ABS({quote_identifier(lower.name)})", f"ABS({quote_identifier(upper.name)})"
    return f"({a} + {b} + ABS({a} - {b})) / 2"


def detect_pos_error(columns: Sequence[ColumnInfo], ra: ColumnInfo | None = None,
                     dec: ColumnInfo | None = None, *,
                     derived: dict[str, str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Positional-error spec for :func:`models.compute_positional_error` from UCDs/descriptions.

    Order of preference (IVOA UCD1+):
    1. per-axis errors of the chosen position -> kind ``sigma``, divided by the stated per-axis
       confidence (1-D normal: 90% -> 1.645, 95% -> 1.960): the ``stat.error;pos.eq.ra`` +
       ``stat.error;pos.eq.dec`` pair, else the columns named exactly ``e_<RA>``/``e_<Dec>``
       (Hipparcos-2 e_RArad/e_DErad, UCD 'stat.error' only), else a 'stat.error' pair whose
       descriptions name RA and Dec (Tycho-2). A pair that belongs to *another* position column
       of the table (AllWISE ``e_RA_pm`` is the error of ``RA_pm``, the motion-fit position at
       MJD 55400, not of RAJ2000) is only used, with an assumption, when nothing else exists.
       A pair of *one-sided* errors (lower/upper: eRASS1 e_RA_ICRS 'lower error on RA
       (RA_LOWERR)' beside E_RA_ICRS) is never read as a symmetric sigma: an error ellipse or a
       total positional error is preferred, else the larger side per axis is selected in ADQL;
    2. an error ellipse (``pos.errorEllipse`` or 'error ellipse' in the description): major,
       minor, position angle -> ``ellipse`` (1 sigma), ``ellipse95`` (2-D 95% contour, / 2.448)
       or ``ellipse95axis`` (per-axis 95%, / 1.96);
    3. one circular error (``stat.error;pos.eq``/``stat.error;pos``, or ``stat.error`` whose
       description is about the position; calibrated totals before raw fit outputs):
       ``radius90``/``radius95`` (Rayleigh radius) for a stated 2-D percentage, ``radial`` for a
       1-sigma sqrt(s_ra^2 + s_dec^2), else ``sigma``.
    Confidence levels are read after removing VizieR markup ('3{sigma}' -> 3 sigma). ADQL
    expressions the spec needs (larger-side columns) are added to ``derived`` when given.
    Returns (spec, columns used with their descriptions, assumptions).
    """
    assumptions: list[str] = []
    numeric = [c for c in columns if c.numeric]
    by_name = {c.name: c for c in columns}
    positions = {c.name for c in columns if ucd_atoms(c.ucd)[:1] in (["pos.eq.ra"], ["pos.eq.dec"])}
    chosen = {c.name for c in (ra, dec) if c is not None}

    def described(cols: Sequence[ColumnInfo], role: Sequence[str]) -> list[dict[str, Any]]:
        return [{"column": c.name, "role": r, "unit": c.unit, "ucd": c.ucd, "description": c.description}
                for c, r in zip(cols, role)]

    def foreign(col: ColumnInfo) -> set[str]:
        """Other position columns this error column belongs to (empty when it is ours)."""
        refs = _referenced_columns(col, positions)
        return set() if refs & chosen or not chosen else refs - chosen

    def axis_errors(axis: str) -> list[ColumnInfo]:
        want = f"pos.eq.{axis}"
        hits = [c for c in numeric if (atoms := ucd_atoms(c.ucd)) and atoms[0] == "stat.error"
                and len(atoms) > 1 and atoms[1] == want and _error_unit(c, axis) is not None
                and not {"pos.pm", "stat.max", "stat.min"} & set(atoms)]
        target = ra if axis == "ra" else dec
        hits.sort(key=lambda c: (1 if foreign(c) else 0,
                                 0 if target is not None and c.name.lower() == f"e_{target.name.lower()}" else 1))
        return hits

    def named_error(target: ColumnInfo | None, axis: str) -> ColumnInfo | None:
        """``e_<target>`` in an angular unit with a stat.error UCD (Hipparcos-2 e_RArad)."""
        if target is None:
            return None
        col = by_name.get(f"e_{target.name}")
        atoms = ucd_atoms(col.ucd) if col is not None else []
        if (col is None or not col.numeric or atoms[:1] != ["stat.error"] or _error_unit(col, axis) is None
                or {"pos.pm", "pos.parallax", "stat.max", "stat.min"} & set(atoms)
                or _NOT_AXIS_ERROR.search(_clean(col.description))):
            return None
        return col

    def pair_spec(era: ColumnInfo, edec: ColumnInfo) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        spec: dict[str, Any] = {"columns": [era.name, edec.name],
                                "units": [_error_unit(era, "ra"), _error_unit(edec, "dec")], "kind": "sigma"}
        conf = _confidence(era.description) or _confidence(edec.description)
        if conf is None:
            notes.append(f"pos_error: {era.name}/{edec.name} do not state a confidence level; taken as 1-sigma.")
        elif conf[0] == "percent":
            spec["divisor"] = round(_one_d_divisor(conf[1]), 6)
        elif conf[1] != 1.0:
            spec["divisor"] = conf[1]
        return spec, described([era, edec], ["ra_error", "dec_error"]), notes

    # 1. per-axis pair of the chosen position (UCD pair, e_<RA>/e_<Dec> names, descriptions)
    fallback_pair: tuple[ColumnInfo, ColumnInfo] | None = None
    one_sided_pair: tuple[ColumnInfo, ColumnInfo] | None = None
    candidates: list[tuple[ColumnInfo, ColumnInfo, str | None]] = []
    ra_errs, dec_errs = axis_errors("ra"), axis_errors("dec")
    if ra_errs and dec_errs:
        era, edec = ra_errs[0], dec_errs[0]
        if not foreign(era) and not foreign(edec):
            candidates.append((era, edec, None))
        else:
            fallback_pair = (era, edec)
    era_named, edec_named = named_error(ra, "ra"), named_error(dec, "dec")
    if era_named is not None and edec_named is not None and ra is not None and dec is not None:
        candidates.append((era_named, edec_named,
                           (f"pos_error: {era_named.name}/{edec_named.name} (UCD '{era_named.ucd}') identified as the "
                            f"RA/Dec errors by their names (e_{ra.name}/e_{dec.name}).")))
    # Tycho-2 e_RAdeg 's.e.RA*cos(dec), of observed Tycho-2 RA' / e_DEdeg 's.e. of observed Tycho-2 Dec'
    if fallback_pair is None:
        described_axes: dict[str, list[ColumnInfo]] = {"ra": [], "dec": []}
        for c in numeric:
            if ucd_atoms(c.ucd) != ["stat.error"] or foreign(c):
                continue
            axis = _error_axis(c.description)
            if axis and _error_unit(c, axis) is not None:
                described_axes[axis].append(c)
        if described_axes["ra"] and described_axes["dec"]:
            era, edec = described_axes["ra"][0], described_axes["dec"][0]
            candidates.append((era, edec, (f"pos_error: {era.name}/{edec.name} (UCD 'stat.error' only) identified "
                                           "as the RA/Dec errors from their descriptions.")))
    for era, edec, how in candidates:
        if _is_one_sided(era, by_name) or _is_one_sided(edec, by_name):
            one_sided_pair = one_sided_pair or (era, edec)
            continue
        spec, used, notes = pair_spec(era, edec)
        if how:
            notes.insert(0, how)
        return spec, used, assumptions + notes
    if one_sided_pair is not None:
        assumptions.append(f"pos_error: {one_sided_pair[0].name}/{one_sided_pair[1].name} are one-sided (lower/upper) "
                           "errors, not a symmetric 1-sigma per-axis error; they are not used as such.")

    # 2. error ellipse
    def is_ellipse(c: ColumnInfo) -> bool:
        text = _clean(c.description)
        return ("pos.errorellipse" in ucd_atoms(c.ucd) or re.search(r"error ellipse", text, re.IGNORECASE) is not None) \
            and "deconvolved" not in text.lower() and not foreign(c)

    ellipse_cols = [c for c in numeric if is_ellipse(c)]
    axes = [c for c in ellipse_cols if _error_unit(c, None) is not None and "pos.posang" not in ucd_atoms(c.ucd)]
    major = next((c for c in axes if re.search(r"major|semi-major", c.description or "", re.IGNORECASE)
                  or "stat.max" in ucd_atoms(c.ucd)), None)
    minor = next((c for c in axes if c is not major and (re.search(r"minor|semi-minor", c.description or "", re.IGNORECASE)
                                                         or "stat.min" in ucd_atoms(c.ucd))), None)
    if major is not None:
        pa = next((c for c in ellipse_cols if "pos.posang" in ucd_atoms(c.ucd)), None)
        cols = [major] + ([minor] if minor else [major]) + ([pa] if pa else [])
        spec = {"columns": [c.name for c in cols], "units": [_error_unit(major, None),
                                                            _error_unit(minor or major, None)] + (["deg"] if pa else []),
                "kind": "ellipse"}
        major_text = _clean(major.description)
        conf = _confidence(major_text)
        per_axis = bool(_PER_AXIS.search(major_text))
        if conf is None:
            assumptions.append(f"pos_error: ellipse {major.name} does not state a confidence level; semi-axes taken "
                               "as 1-sigma.")
        elif conf[0] == "percent":
            if per_axis:
                spec["kind"] = "ellipse95axis" if conf[1] == 95.0 else "ellipse"
                if conf[1] != 95.0:
                    spec["divisor"] = round(_one_d_divisor(conf[1]), 6)
            else:
                spec["kind"] = "ellipse95" if conf[1] == 95.0 else "ellipse"
                if conf[1] != 95.0:
                    spec["divisor"] = round(_rayleigh_divisor(conf[1]), 6)
                assumptions.append(
                    f"pos_error: '{conf[1]:g}% confidence' ellipse {major.name} read as the 2-D {conf[1]:g}% contour "
                    f"(semi-axes / {_rayleigh_divisor(conf[1]):.4f}); if the archive means per-axis intervals "
                    f"(e.g. Chandra CSC) override kind to 'ellipse95axis' (/ {K95_1D:.2f})."
                )
        elif conf[1] != 1.0:
            spec["divisor"] = conf[1]
        if minor is None:
            assumptions.append(f"pos_error: no minor axis found; {major.name} used as a circular error.")
        if one_sided_pair is not None:
            assumptions.append(f"pos_error: the error ellipse {major.name} is used instead of the one-sided "
                               f"{one_sided_pair[0].name}/{one_sided_pair[1].name}.")
        if fallback_pair is not None:
            assumptions.append(f"pos_error: {fallback_pair[0].name}/{fallback_pair[1].name} are the errors of "
                               f"{', '.join(sorted(foreign(fallback_pair[0]) | foreign(fallback_pair[1])))}, not of "
                               f"the chosen position; the error ellipse is used instead.")
        roles = ["major", "minor" if minor else "major"] + (["position_angle"] if pa else [])
        return spec, described(cols, roles), assumptions

    # 3. circular error
    not_positional: list[str] = []

    def explicit(c: ColumnInfo) -> bool:
        atoms = ucd_atoms(c.ucd)
        return len(atoms) > 1 and atoms[1] in {"pos.eq", "pos"}

    def circular(c: ColumnInfo) -> bool:
        atoms = ucd_atoms(c.ucd)
        if (not atoms or atoms[0] != "stat.error" or _error_unit(c, None) is None or foreign(c)
                or _is_one_sided(c, by_name)):
            return False
        text = _clean(c.description)
        if not explicit(c) and not (len(atoms) == 1 and _POSITION_WORDS.search(text)):
            return False
        # The error of another, non-positional quantity (GALEX e_nPA is the error of nPA, a
        # pos.posAng), position-angle wording, or values no positional error can take.
        stem = by_name.get(c.name[2:]) if c.name.lower().startswith("e_") else None
        stem_atoms = ucd_atoms(stem.ucd) if stem is not None else []
        reason = None
        if stem_atoms and not stem_atoms[0].startswith(("pos.eq", "pos.errorellipse")) and stem_atoms[0] != "pos":
            reason = f"the error of {stem.name} ({stem.ucd})"  # type: ignore[union-attr]
        elif (_ANGLE_WORDS if explicit(c) else _NOT_POSITION).search(text):
            reason = f"'{text[:70]}' is not a positional error"
        else:
            reason = _implausible_position_error(c)
        if reason is not None:
            not_positional.append(f"{c.name} ({reason})")
            return False
        return True

    # Columns explicitly tagged stat.error;pos(.eq) rank before description-only matches, and
    # calibrated totals before raw fit outputs (column order otherwise).
    circles = sorted((c for c in numeric if circular(c)),
                     key=lambda c: (0 if explicit(c) else 1, 1 if _RAW_ERROR.search(_clean(c.description)) else 0))
    if circles:
        col = circles[0]
        spec = {"columns": [col.name], "units": [_error_unit(col, None)], "kind": "sigma"}
        text = _clean(col.description)
        conf = _confidence(text)
        if conf is not None and conf[0] == "percent":
            if conf[1] in (90.0, 95.0):
                spec["kind"] = "radius90" if conf[1] == 90.0 else "radius95"
            else:
                spec["divisor"] = round(_rayleigh_divisor(conf[1]), 6)
        elif _is_radial_error_text(text) and (conf is None or conf[1] == 1.0):
            spec["kind"] = "radial"
            if conf is None:
                assumptions.append(f"pos_error: {col.name} is a radial error without a stated level; taken as "
                                   "sqrt(s_ra^2 + s_dec^2) at 1 sigma.")
        elif conf is None:
            assumptions.append(f"pos_error: {col.name} ('{text[:80]}') states no confidence level; taken as a "
                               "1-sigma per-axis error.")
        elif conf[1] != 1.0:
            spec["divisor"] = conf[1]
        if fallback_pair is not None:
            assumptions.append(f"pos_error: {fallback_pair[0].name}/{fallback_pair[1].name} belong to another position "
                               f"column; the circular error {col.name} is used instead.")
        if one_sided_pair is not None:
            assumptions.append(f"pos_error: the total positional error {col.name} ('{text[:60]}') is used instead of "
                               f"the one-sided {one_sided_pair[0].name}/{one_sided_pair[1].name}.")
        return spec, described([col], ["radius"]), assumptions

    if one_sided_pair is not None:
        era, edec = one_sided_pair
        upper_ra, upper_dec = _upper_partner(era, by_name), _upper_partner(edec, by_name)
        if (upper_ra is not None and upper_dec is not None and _error_unit(upper_ra, "ra") == _error_unit(era, "ra")
                and _error_unit(upper_dec, "dec") == _error_unit(edec, "dec")):
            aliases = []
            for col, upper in ((era, upper_ra), (edec, upper_dec)):
                alias = f"{col.name}_max"
                while alias in by_name:
                    alias = "_" + alias
                aliases.append(alias)
                if derived is not None:
                    derived[alias] = _larger_side_expr(col, upper)
            spec, used, notes = pair_spec(era, edec)
            spec["columns"] = aliases
            used = described([era, upper_ra, edec, upper_dec],
                             ["ra_error_lower", "ra_error_upper", "dec_error_lower", "dec_error_upper"])
            notes.append(f"pos_error: no total positional error; the larger of the lower/upper errors per axis "
                         f"(max(|{era.name}|, |{upper_ra.name}|), max(|{edec.name}|, |{upper_dec.name}|), selected "
                         f"as {aliases[0]}/{aliases[1]}) is taken as the 1-sigma per-axis error.")
            return spec, used, assumptions + notes
        spec, used, notes = pair_spec(era, edec)
        notes.append(f"pos_error: {era.name}/{edec.name} are one-sided errors without an upper/lower partner; used "
                     "for lack of any other positional error (they may understate the uncertainty; override "
                     "pos_error if the catalogue documents a total error).")
        return spec, used, assumptions + notes

    if fallback_pair is not None:
        spec, used, notes = pair_spec(*fallback_pair)
        others = sorted(foreign(fallback_pair[0]) | foreign(fallback_pair[1]))
        notes.append(f"pos_error: {fallback_pair[0].name}/{fallback_pair[1].name} are described as the errors of "
                     f"{', '.join(others)}, not of the chosen position; used for lack of any other positional error.")
        return spec, used, assumptions + notes

    if not_positional:
        assumptions.append("pos_error: " + "; ".join(not_positional[:4]) + (" ..." if len(not_positional) > 4 else "")
                           + " -- not used as a positional error.")
    assumptions.append("pos_error: no positional-error column found; matches use separation only.")
    return {}, [], assumptions


#: Archives whose error ellipses are per-axis 95% intervals (CXC: CSC err_ellipse_r0/r1 are
#: "95% confidence intervals along these axes"; the core's chandra entry and
#: models.NED_ERROR_DIVISORS use the same K95_1D convention).
_PER_AXIS_95_ARCHIVES = re.compile(r"Chandra Source Catalog|\bCSC\s?[12](?:\.\d)?\b", re.IGNORECASE)
_PER_AXIS_95_BIBCODES = frozenset(code for code, divisor in NED_ERROR_DIVISORS.items()
                                  if math.isclose(divisor, K95_1D))
#: XMM-Newton serendipitous source catalogues (2XMM/3XMM/4XMM, 'XMM-Newton Serendipitous Source
#: Catalogue') and their radial SC_POSERR / POSERR column names.
_XMM_SSC_ARCHIVES = re.compile(r"\b[234]XMM\b|XMM-Newton Serendipitous|XMM Serendipitous", re.IGNORECASE)
_XMM_RADIAL_COLUMN = re.compile(r"\bSC_POSERR\b|\bSC_RADEC_ERR\b", re.IGNORECASE)


def _survey_systematics() -> list[tuple[re.Pattern[str], float, str]]:
    """Astrometric systematics the core registry documents for surveys also served by VizieR
    (VLASS Quick-Look ~0.5", LoTSS 0.2"), matched by survey name in the catalogue title."""
    known = []
    for core_name, pattern in (("vlass", r"\bVLASS\b"), ("lotss", r"\bLoTSS\b|LOFAR Two-metre Sky Survey")):
        value = _float(((DEFAULT_CATALOGS.get(core_name) or {}).get("pos_error") or {}).get("systematic_arcsec"))
        if value:
            known.append((re.compile(pattern, re.IGNORECASE), value, core_name))
    return known


def apply_error_conventions(spec: Mapping[str, Any], notes: Sequence[str], catalog: CatalogInfo, table_id: str,
                            table_description: str | None = None,
                            used: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Archive conventions the column metadata cannot express (returns the spec and notes).

    * Chandra Source Catalog ellipses are per-axis 95% intervals: ``ellipse95`` -> ``ellipse95axis``
      (by catalogue title or a bibcode listed with K95_1D in models.NED_ERROR_DIVISORS).
    * The XMM-Newton serendipitous catalogues' SC_POSERR / POSERR (4XMM ``ePos``) is the total
      *radial* 1-sigma error sqrt(RADEC_ERR^2 + SYSERR^2) with RADEC_ERR = sqrt(s_ra^2 + s_dec^2)
      (XMM SSC documentation; the core 'xmm' entry uses kind 'radial'): a single-column
      ``sigma`` read from such a column becomes ``radial`` (per-axis sigma = ePos / sqrt 2).
    * A VizieR table the core registry already serves (same TAPVizieR table) inherits the
      core entry's ``systematic_arcsec``; otherwise a known survey systematic (VLASS Quick-Look
      0.5", LoTSS 0.2", the values of the core's vlass/lotss entries) is added.
    Every change is recorded as an assumption; overrides still win."""
    out = dict(spec)
    notes = list(notes)
    if not out:
        return out, notes
    text = " ".join(t for t in (catalog.title, table_description, catalog.catalog_id) if t)
    csc = bool(_PER_AXIS_95_ARCHIVES.search(text)) or (catalog.bibcode in _PER_AXIS_95_BIBCODES)
    if csc and out.get("kind") == "ellipse95" and "divisor" not in out:
        out["kind"] = "ellipse95axis"
        notes = [n for n in notes if "read as the 2-D" not in n]
        notes.append("pos_error: Chandra Source Catalog error ellipses are per-axis 95% confidence intervals (CXC "
                     f"documentation; core chandra entry and NED convention) -> kind 'ellipse95axis' (/ {K95_1D:.2f}).")
    columns = list(out.get("columns") or [])
    column_text = " ".join(_clean(str(u.get("description") or "")) for u in (used or []))
    xmm = bool(_XMM_SSC_ARCHIVES.search(text))
    if (out.get("kind") == "sigma" and len(columns) == 1 and "divisor" not in out
            and (_XMM_RADIAL_COLUMN.search(column_text) or (xmm and _POSITION_WORDS.search(column_text)))):
        out["kind"] = "radial"
        col = columns[0]
        notes = [n for n in notes if not n.startswith(f"pos_error: {col} (")]
        notes.append(f"pos_error: {col} is the XMM-Newton serendipitous-catalogue SC_POSERR, the total radial 1-sigma "
                     "error sqrt(RADEC_ERR^2 + SYSERR^2) (XMM SSC documentation; core 'xmm' entry) -> kind 'radial' "
                     f"(per-axis sigma = {col} / sqrt 2).")
    if out.get("systematic_arcsec") is None:
        quoted = quote_identifier(table_id)
        for core_name, entry in DEFAULT_CATALOGS.items():
            systematic = _float((entry.get("pos_error") or {}).get("systematic_arcsec"))
            if systematic and entry.get("endpoint") == VIZIER_TAP and entry.get("table") == quoted:
                out["systematic_arcsec"] = systematic
                notes.append(f"pos_error: astrometric systematic {systematic:g}\" added in quadrature, as in the core "
                             f"'{core_name}' entry for the same VizieR table.")
                break
        else:
            for pattern, systematic, core_name in _survey_systematics():
                if pattern.search(text):
                    out["systematic_arcsec"] = systematic
                    notes.append(f"pos_error: the catalogued errors are statistical only; the survey's astrometric "
                                 f"systematic {systematic:g}\" (core '{core_name}' entry) is added in quadrature "
                                 "(override systematic_arcsec to change it).")
                    break
    return out, notes


# -- epochs ---------------------------------------------------------------------

_DATE_UNITS = frozenset({"d", "yr", "a", "year", "jyear"})
_START = re.compile(r"\b(first|start|begin|beginning|earliest|min(?:imum)?)\b", re.IGNORECASE)
_END = re.compile(r"\b(last|end|stop|latest|max(?:imum)?)\b", re.IGNORECASE)
_RANGE_PREFIX = re.compile(r"^\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*[/,]\s*([-+]?\d+(?:\.\d+)?)\s*\]")
#: 'Ep=J2000', 'Ep=2016.0', 'epoch 2000.0', 'epoch=J2000' (not 'epoch-1990', an offset).
_EP_IN_TEXT = re.compile(r"\bEp(?:och)?\s*[=:]?\s*J?((?:18|19|20|21)\d\d(?:\.\d+)?)\b(?!\s*[-+]\s*\d)", re.IGNORECASE)
#: 'at Epoch="MJD"' (SDSS), 'at epoch "Epoch"' (Pan-STARRS): the epoch is a per-row column.
_EP_COLUMN = re.compile(r"\bep(?:och)?\s*=?\s*\"([^\"]+)\"", re.IGNORECASE)
_MEAN_EPOCH = re.compile(r"\bmean\s+epoch\s*(?:=|:|of|is)?\s*J?((?:18|19|20|21)\d\d(?:\.\d+)?)\b"
                         r"|\bepoch\s*(?:=|:)\s*J?((?:18|19|20|21)\d\d(?:\.\d+)?)\b", re.IGNORECASE)
#: Values stored as an offset from a year ('EpRA-1990: epoch-1990 of RAdeg', Tycho-2).
_OFFSET_TEXT = re.compile(r"\bepoch\s*-\s*((?:18|19|20)\d\d)\b", re.IGNORECASE)
_OFFSET_NAME = re.compile(r"-((?:18|19|20)\d\d)$")
#: Times of light-curve/orbital events, never the epoch of a position (VSX 'Epoch of maximum
#: or minimum (HJD)', ephemerides 'T0', periastron/transit times).
_EVENT_TIME = re.compile(
    r"\b(?:epoch|time|date|jd|hjd|bjd)\s+of\s+(?:the\s+)?(?:max|min|light|periast|apast|transit|eclips|conjunction|"
    r"outburst|flare|burst|peak|discovery|explosion|trigger|zero|phase)"
    r"|periast|transit|eclips|\bT_?0\b|zero[- ]?point|ephemer|light[- ]?curve|\bphase\b|\bHJD\b|\bBJD\b",
    re.IGNORECASE)
#: Wording that ties a time column to the measured positions.
_POSITION_TIME = re.compile(r"observ|measure|position|coordinate|astrometr|detect|mean epoch|central epoch",
                            re.IGNORECASE)


def _epoch_format(col: ColumnInfo) -> str | None:
    text = f"{col.name} {col.description or ''}"
    unit = (col.unit or "").strip().lower()
    if unit in {"yr", "a", "year", "jyear"}:
        return "jyear"
    if re.search(r"\bmjd|modified julian", text, re.IGNORECASE):
        return "mjd"
    if re.search(r"\bjd\b|julian da(?:te|y)|^jd", text, re.IGNORECASE):
        return "jd"
    return None


def _epoch_candidates(columns: Sequence[ColumnInfo]) -> list[ColumnInfo]:
    out = []
    for col in columns:
        atoms = ucd_atoms(col.ucd)
        if not atoms or atoms[0] not in {"time.epoch", "time.start", "time.end"} or not col.numeric:
            continue
        unit = (col.unit or "").strip().lower()
        if unit in _DATE_UNITS or (unit == "" and _epoch_format(col) is not None):
            out.append(col)
    return out


def _range_years(col: ColumnInfo, fmt: str | None, offset: float = 0.0) -> tuple[float, float] | None:
    match = _RANGE_PREFIX.match(col.description or "")
    if not match:
        return None
    lo: float | None
    hi: float | None
    if offset:
        lo, hi = (float(match.group(i)) + offset for i in (1, 2))
    else:
        lo, hi = (epoch_to_jyear(match.group(i), fmt) for i in (1, 2))
    if lo is None or hi is None or not (1800 <= lo <= 2200 and 1800 <= hi <= 2200):
        return None
    return round(min(lo, hi), 3), round(max(lo, hi), 3)


def _same_axis_group(a: ColumnInfo, b: ColumnInfo) -> bool:
    """EpRA/EpDE, epRA/epDE, EpRA-1990/EpDE-1990: the RA and Dec epochs of one position."""
    swap = str.maketrans({"R": "D", "D": "R"})
    na, nb = a.name, b.name
    return na == nb or na.replace("RA", "DE") == nb or nb.replace("RA", "DE") == na \
        or na.replace("ra", "de") == nb or nb.replace("ra", "de") == na or na.translate(swap) == nb


def _fixed_year(value: Any) -> float | None:
    match = re.fullmatch(r"J?((?:18|19|20|21)\d\d(?:\.\d+)?)", str(value or "").strip())
    return float(match.group(1)) if match else None


def detect_epoch(columns: Sequence[ColumnInfo], ra: ColumnInfo | None, coosys: Mapping[str, Any] | None,
                 texts: Sequence[str | None], *, dec: ColumnInfo | None = None,
                 fields: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Epoch of the chosen positions, most authoritative statement first.

    1. An explicit statement about the position itself: a per-row column named in the RA/Dec
       description (SDSS 'at Epoch="MJD"', Pan-STARRS 'at epoch "Epoch"'), the VOTable COOSYS
       ``epoch`` of the RA field (VOTable 1.4 sec. 2.1; PPMXL/UCAC4/USNO-B1 '2000.000', Gaia
       'J2016.0'; SDSS's '0.000' is not a year and is ignored) or 'Ep=J2000' / 'epoch 2000.0' in
       the RA/Dec description. Generic ``time.epoch`` columns never override these (PPMXL's
       epRA is the mean epoch of the observations, its RAJ2000 is at epoch 2000.0) -- except a
       J2000 stamp on a table *without proper motions* that has exactly one per-row 'epoch of
       the observations' column: its positions cannot have been propagated to 2000, so that
       column is used and the conflict recorded (SkyMapper DR4 EpMean, 'Mean MJD epoch of the
       observations', beside COOSYS epoch 2000 and 'Ep=J2000').
    2. First/last observation columns (``time.start``/``time.end``) -> per-row span.
    3. One per-row ``time.epoch`` column tied to the position: its description names the chosen
       RA/Dec, it shares the RA field's COOSYS reference in VizieR's VOTable, or it speaks of
       the observation/measurement/mean epoch. Event times (VSX 'Epoch of maximum or minimum
       (HJD)', T0, periastron, transit) and epochs of *another* position column are rejected;
       values stored as offsets ('epoch-1990 of RAdeg', Tycho-2) become a derived column
       ``"<col>" + 1990``. Two equally supported, unrelated candidates are ambiguous: the epoch
       is then left unknown and the candidates are listed.
    4. A mean epoch stated in the table/catalogue description.
    Returns {epoch, epoch_format, epoch_range, source, single_epoch_positions, columns,
    derived, notes}.
    """
    result: dict[str, Any] = {"epoch": None, "epoch_format": None, "epoch_range": None, "source": None,
                              "single_epoch_positions": False, "columns": [], "derived": {}, "notes": []}
    fields = fields or {}
    by_name = {c.name: c for c in columns}
    cands = _epoch_candidates(columns)
    position_cols = {c.name for c in columns if ucd_atoms(c.ucd)[:1] in (["pos.eq.ra"], ["pos.eq.dec"])}
    chosen = {c.name for c in (ra, dec) if c is not None}

    def use_column(col: ColumnInfo, source: str) -> dict[str, Any]:
        fmt = _epoch_format(col)
        offset_match = _OFFSET_TEXT.search(_clean(col.description)) or _OFFSET_NAME.search(col.name)
        if offset_match:
            base = float(offset_match.group(1))
            alias = f"{col.name}_jyear"
            while alias in by_name:
                alias = "_" + alias
            result["derived"] = {alias: f"{quote_identifier(col.name)} + {base:g}"}
            span = _range_years(col, "jyear", offset=base)
            result.update(epoch=alias, epoch_format="jyear", columns=[col.name],
                          source=f"{source}; stored as an offset from {base:g} (derived column {col.name} + {base:g})")
            if span:
                result["epoch_range"] = list(span)
        else:
            result.update(epoch=col.name, epoch_format=fmt, source=source, columns=[col.name])
            span = _range_years(col, fmt)
            if span:
                result["epoch_range"] = list(span)
            if fmt is None:
                result["notes"].append(f"epoch: {col.name} unit '{col.unit}' -- JD/MJD/year inferred from the value.")
        mean = re.search(r"\bmean\b|\baverage\b", col.description or "", re.IGNORECASE)
        result["single_epoch_positions"] = not mean
        return result

    # 1. explicit statements about the chosen position
    position_texts = [(c.name, _clean(c.description)) for c in (ra, dec) if c is not None]
    for name, text in position_texts:
        match = _EP_COLUMN.search(text)
        if match and match.group(1) in by_name and by_name[match.group(1)].numeric:
            return use_column(by_name[match.group(1)], f"{name} description ('{match.group(0)}')")
    has_pm = any(ucd_atoms(c.ucd)[:2] == ["pos.pm", "pos.eq.ra"] and c.numeric for c in columns) \
        and any(ucd_atoms(c.ucd)[:2] == ["pos.pm", "pos.eq.dec"] and c.numeric for c in columns)

    def observation_epoch_column() -> ColumnInfo | None:
        """A per-row 'mean epoch of the observations' column of a table without proper motions
        (SkyMapper DR4 EpMean 'Mean MJD epoch of the observations')."""
        if has_pm:
            return None
        found = [c for c in cands if ucd_atoms(c.ucd)[0] == "time.epoch"
                 and not _EVENT_TIME.search(_clean(c.description))
                 and re.search(r"observ|measure|detect", _clean(c.description), re.IGNORECASE)
                 and not (_referenced_columns(c, position_cols) - chosen)]
        return found[0] if len(found) == 1 else None

    def fixed_epoch(year: float, source: str) -> dict[str, Any]:
        # Only VizieR's default J2000 stamp is doubted: it is written for tables whose positions
        # carry no epoch statement of their own, while a deliberate other epoch (Gaia 2015.5/2016.0,
        # Hipparcos 1991.25) states where the positions are.
        per_row = observation_epoch_column() if abs(year - 2000.0) < 1e-6 else None
        if per_row is None:
            result.update(epoch=year, source=source)
            return result
        # A fixed epoch needs positions propagated with proper motions; a table without any,
        # whose rows carry the epoch of their own observations, holds positions at that epoch
        # (SkyMapper DR4: COOSYS 'epoch 2000' and 'Ep=J2000' in RAICRS, but Barnard's star sits
        # at its 2018.6 position, 194" from J2000).
        use_column(per_row, f"column {per_row.name} ({per_row.ucd})")
        result["notes"].insert(0, f"epoch: {source} states {year:g}, but the table has no proper motions and "
                                  f"{per_row.name} ('{_clean(per_row.description)[:60]}') gives each row's epoch of "
                                  f"observation: positions cannot have been propagated to {year:g}, so {per_row.name} "
                                  f"is used (override 'epoch': {year:g} to use the stated epoch instead).")
        return result

    year = _fixed_year((coosys or {}).get("epoch")) if ra is not None else None
    if year is not None:
        for name, text in position_texts:
            stated = _EP_IN_TEXT.search(text)
            if stated and abs(float(stated.group(1)) - year) > 1e-6:
                result["notes"].append(f"epoch: COOSYS says {year:g} but {name} says '{stated.group(0)}'; the "
                                       "COOSYS epoch is used.")
        return fixed_epoch(year, f"VOTable COOSYS epoch {coosys.get('epoch')}")  # type: ignore[union-attr]
    for name, text in position_texts:
        stated = _EP_IN_TEXT.search(text)
        if stated:
            return fixed_epoch(float(stated.group(1)), f"{name} description ('{stated.group(0)}')")

    # 2. per-row observation span
    events = [c for c in cands if _EVENT_TIME.search(_clean(c.description))
              and not {"time.start", "time.end"} & set(ucd_atoms(c.ucd))]
    pool = [c for c in cands if c not in events]
    starts = [c for c in pool if "time.start" in ucd_atoms(c.ucd) or _START.search(c.description or "")]
    ends = [c for c in pool if c not in starts and ("time.end" in ucd_atoms(c.ucd) or _END.search(c.description or ""))]
    if starts and ends:
        start, end = starts[0], ends[0]
        fmt = _epoch_format(start) or _epoch_format(end)
        spec: dict[str, Any] = {"span_columns": [start.name, end.name], "point_max_years": 0.1}
        if fmt:
            spec["format"] = fmt
        lo, hi = _range_years(start, fmt), _range_years(end, fmt)
        result.update(epoch=spec, epoch_format=fmt, source="columns (first/last observation span)",
                      columns=[start.name, end.name])
        if lo and hi:
            result["epoch_range"] = [lo[0], hi[1]]
        result["notes"].append(f"epoch: positions measured between {start.name} and {end.name} (per row); rows whose "
                               "span exceeds 0.1 yr are matched over the whole span.")
        return result

    # 3. one per-row column tied to the position
    for col in events:
        result["notes"].append(f"epoch: {col.name} ('{(col.description or '')[:60]}') is an event time, not the epoch "
                               "of the positions; ignored.")
    ra_ref = (fields.get(ra.name) or {}).get("ref") if ra is not None else None
    scored: list[tuple[int, ColumnInfo]] = []
    for col in pool:
        if col in starts or col in ends:
            continue
        refs = _referenced_columns(col, position_cols)
        if refs and not refs & chosen:
            result["notes"].append(f"epoch: {col.name} is the epoch of {', '.join(sorted(refs))}, not of the chosen "
                                   "position; ignored.")
            continue
        score = 3 if refs & chosen else 0
        if ra_ref and (fields.get(col.name) or {}).get("ref") == ra_ref:
            score += 2
        if _POSITION_TIME.search(_clean(col.description)) or _OFFSET_NAME.search(col.name):
            score += 1
        scored.append((score, col))
    scored.sort(key=lambda item: (-item[0], 0 if "meta.main" in ucd_atoms(item[1].ucd) else 1))
    if scored and scored[0][0] >= 1:
        best_score, best = scored[0]
        rivals = [c for s, c in scored[1:] if s == best_score and not _same_axis_group(best, c)]
        if not rivals:
            return use_column(best, f"column {best.name} ({best.ucd})")
        names = ", ".join(c.name for c in [best, *rivals])
        result["notes"].append(f"epoch: ambiguous -- {names} are equally plausible epochs of the positions; the epoch "
                               "is left unknown (override 'epoch' to choose one).")
        return result
    for _score, col in scored:
        result["notes"].append(f"epoch: {col.name} ('{(col.description or '')[:60]}') is not described as the epoch "
                               "of the positions; ignored (override 'epoch' to use it).")

    # 4. a mean epoch stated in the table or catalogue description
    for description_text in texts:
        found = _MEAN_EPOCH.search(description_text or "") or _EP_IN_TEXT.search(description_text or "")
        if found:
            match = found
            year_text = next(group for group in match.groups() if group)
            result.update(epoch=float(year_text), source=f"description ('{match.group(0)}')")
            return result
    result["notes"].append("epoch: not stated in the VizieR metadata; positions are compared as given "
                           "(no proper-motion propagation for this catalog).")
    return result


def _pick_best(columns: Sequence[ColumnInfo], predicate) -> ColumnInfo | None:
    hits = [c for c in columns if predicate(c)]
    hits.sort(key=lambda c: (0 if "meta.main" in ucd_atoms(c.ucd) else 1, 0 if c.principal else 1))
    return hits[0] if hits else None


_DIMENSIONLESS = frozenset({"", "-", "---", "1", "dimensionless"})


#: Spectral *model* names tagged src.spType (4FGL 'Mod': 'Spectral type in the global model
#: ("PowerLaw", "LogParabola", ...)'): not stellar classifications.
_SPECTRAL_MODEL = re.compile(r"\bmodel|spectral shape|fit(?:ted|ting)? function|power[- ]?law|log[- ]?parabola"
                             r"|cut-?off|\bband\b function", re.IGNORECASE)
#: Wavelengths whose catalogues have no stellar (MK) spectral classifications.
_NON_STELLAR_SPTYPE_WAVELENGTHS = frozenset({"xray", "gamma", "high-energy", "radio", "millimeter"})
#: 'Spectroscopic subclass' (SDSS subClass, LAMOST subclass): MK types for stars, but emission-line
#: classes ('STARFORMING', 'BROADLINE', 'AGN') for galaxies and QSOs.
_SUBCLASS = re.compile(r"\bsub-?class", re.IGNORECASE)
_EXTRAGALACTIC_WORDS = re.compile(r"\bgalax|\bQSOs?\b|quasar|\bAGN\b", re.IGNORECASE)
#: A column describing a cross-identified object of another catalogue, not the source itself.
_COUNTERPART_TEXT = re.compile(r"\bnearest\b|counterpart|cross-?ident|cross-?match|\bmatched\s+(?:source|object)"
                               r"|\bassociated\s+(?:source|object)", re.IGNORECASE)


def _name_tails(name: str) -> set[str]:
    """Catalogue-suffix candidates of a VizieR column name: the tails that start at a word
    boundary ('zVV10' -> {'VV10'}, 'SpTypeBSC' -> {'TypeBSC', 'BSC'}, 'd2RXP' -> {'2RXP'}), at
    least 3 characters long."""
    tails = set()
    for i in range(1, len(name)):
        here, before = name[i], name[i - 1]
        boundary = before == "_" or ((here.isupper() or here.isdigit()) and not (before.isupper() or before.isdigit()))
        if boundary and here != "_" and len(name) - i >= 3:
            tails.add(name[i:])
    return tails


def _counterpart_reason(col: ColumnInfo, columns: Sequence[ColumnInfo]) -> str | None:
    """Why ``col`` describes a cross-identified counterpart from another catalogue, else None.

    Signals: its description speaks of the nearest/counterpart/cross-identified object, or a
    sibling ``pos.angDistance`` column carries the same catalogue suffix (J/A+A/588/A103/cat2rxs:
    zVV10 'VV10 Redshift' and TypeVV10 beside dVV10 'Distance to nearest VV10 counterpart', which
    reaches 300": the nearest Veron-Cetty & Veron QSO of a star's X-ray source)."""
    text = _clean(col.description)
    match = _COUNTERPART_TEXT.search(text)
    if match:
        return f"is described as a '{match.group(0)}' value"
    tails = _name_tails(col.name)
    if not tails:
        return None
    for other in columns:
        if other.name == col.name or not ucd_atoms(other.ucd)[:1] or not ucd_atoms(other.ucd)[0].startswith("pos.angdistance"):
            continue
        shared = tails & _name_tails(other.name)
        if shared:
            suffix = max(shared, key=len)
            return (f"belongs to the '{suffix}' cross-identification ({other.name}: "
                    f"'{_clean(other.description)[:60]}')")
    return None


def _has_extragalactic_classes(columns: Sequence[ColumnInfo]) -> str | None:
    """A column showing that the table classifies galaxies/QSOs (a src.redshift column, or a
    class column whose description names galaxies/QSOs), else None."""
    for c in columns:
        atoms = ucd_atoms(c.ucd)
        if atoms and atoms[0].startswith("src.redshift"):
            return f"{c.name}, {c.ucd}"
        if any(a.startswith(("src.class", "meta.code.class")) for a in atoms) \
                and _EXTRAGALACTIC_WORDS.search(_clean(c.description)):
            return f"{c.name}, '{_clean(c.description)[:40]}'"
    return None


def detect_field_map_notes(columns: Sequence[ColumnInfo], *,
                           wavelength: str | None = None) -> tuple[dict[str, str], list[str]]:
    """Canonical physical fields chosen by UCD (proper motions, parallax, redshift, class,
    spectral type), with notes on columns deliberately left out.

    ``src.redshift`` is mapped to the canonical (dimensionless) redshift only when its unit is
    empty/dimensionless: VizieR tags recession velocities cz in km/s with the same UCD
    (J/ApJ/956/51/table4 'Local Group-corrected Recession Velocity'), and z = v/c holds only for
    heliocentric velocities, so velocity columns are never converted silently.

    ``src.spType`` is mapped to the canonical (stellar MK) ``spectral_type`` only for stellar
    classifications: a column tagged ``meta.modelled`` or described as a spectral model/shape
    (4FGL 'Mod' = PowerLaw/LogParabola), or any src.spType column of an X-ray, gamma-ray or
    radio catalogue (``wavelength``), is left out with a note."""
    def atoms(c: ColumnInfo) -> list[str]:
        return ucd_atoms(c.ucd)

    def has_unit(c: ColumnInfo) -> bool:
        return bool((c.unit or "").strip())

    notes: list[str] = []
    mapping: dict[str, str] = {}
    counterpart_cache: dict[str, str | None] = {}

    def own(c: ColumnInfo, canonical: str) -> bool:
        """False (with a note) for a column describing a cross-identified object of another
        catalogue (2RXS zVV10/TypeVV10 beside dVV10 'Distance to nearest VV10 counterpart')."""
        if c.name not in counterpart_cache:
            counterpart_cache[c.name] = _counterpart_reason(c, columns)
            if counterpart_cache[c.name]:
                notes.append(f"{canonical}: {c.name} ('{_clean(c.description)[:60]}') {counterpart_cache[c.name]}; it "
                             f"describes that counterpart, not the source, and is not mapped to the canonical "
                             f"{canonical}.")
        return counterpart_cache[c.name] is None

    pmra = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:2] == ["pos.pm", "pos.eq.ra"]
                      and own(c, "pmra"))
    pmdec = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:2] == ["pos.pm", "pos.eq.dec"]
                       and own(c, "pmdec"))
    if pmra and pmdec:
        mapping.update(pmra=pmra.name, pmdec=pmdec.name)
    plx = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:1] in (["pos.parallax.trig"], ["pos.parallax"])
                     and own(c, "parallax"))
    if plx:
        mapping["parallax"] = plx.name

    def redshift_like(c: ColumnInfo) -> bool:
        a = atoms(c)
        return (c.numeric and bool(a) and a[0].startswith("src.redshift") and "stat" not in ";".join(a)
                and not re.search(r"model|fit|photometric|phot\b", c.description or "", re.IGNORECASE)
                and a[0] != "src.redshift.phot")

    def dimensionless(c: ColumnInfo) -> bool:
        return (c.unit or "").strip().lower() in _DIMENSIONLESS

    for c in columns:
        if redshift_like(c) and not dimensionless(c) and own(c, "redshift"):
            notes.append(f"redshift: {c.name} is tagged src.redshift but its unit is '{c.unit}' (a velocity such as cz, "
                         "not a redshift); it is not mapped to the canonical redshift.")
    usable_z = [c for c in columns if redshift_like(c) and dimensionless(c) and own(c, "redshift")]
    exact = [c for c in usable_z if atoms(c) == ["src.redshift"]]
    z = exact[0] if exact else _pick_best(usable_z, lambda c: True)
    if z:
        mapping["redshift"] = z.name
    extragalactic = _has_extragalactic_classes(columns)

    def object_class(c: ColumnInfo) -> bool:
        if atoms(c)[:1] != ["src.class"] or not own(c, "object_type"):
            return False
        if _is_integer(c.datatype):
            notes.append(f"object_type: {c.name} ('{_clean(c.description)[:70]}') holds integer class codes, not "
                         "object-type names; it is not mapped to the canonical object_type (select it with "
                         "extra_columns to keep the codes).")
            return False
        return True

    otype = _pick_best(columns, object_class)
    if otype:
        mapping["object_type"] = otype.name

    def stellar_sptype(c: ColumnInfo) -> bool:
        a = atoms(c)
        if not a or not a[0].startswith("src.sptype") or not own(c, "spectral_type"):
            return False
        text = _clean(c.description)
        if "meta.modelled" in a or _SPECTRAL_MODEL.search(text):
            notes.append(f"spectral_type: {c.name} ('{(c.description or '')[:70]}') names a spectral model, not a "
                         "stellar classification; it is not mapped to the canonical spectral_type.")
            return False
        if wavelength in _NON_STELLAR_SPTYPE_WAVELENGTHS:
            notes.append(f"spectral_type: {c.name} is tagged src.spType in a {wavelength} catalogue (not a stellar MK "
                         "classification); it is not mapped to the canonical spectral_type.")
            return False
        if extragalactic and _SUBCLASS.search(text):
            notes.append(f"spectral_type: {c.name} ('{text[:60]}') is a spectroscopic subclass in a table that also "
                         f"classifies galaxies/QSOs ({extragalactic}): its values include galaxy/QSO subclasses "
                         "(e.g. SDSS 'STARFORMING', 'BROADLINE'), not only stellar MK types; it is not mapped to the "
                         "canonical spectral_type.")
            return False
        return True

    sptype = _pick_best(columns, stellar_sptype)
    if sptype:
        mapping["spectral_type"] = sptype.name
    return mapping, notes


def detect_field_map(columns: Sequence[ColumnInfo]) -> dict[str, str]:
    """Canonical physical fields chosen by UCD (see :func:`detect_field_map_notes`)."""
    return detect_field_map_notes(columns)[0]


#: Counts ('Number of positions used', Tycho-2 'Num'), not identifiers ('Sequential number' is one).
_COUNT_LIKE = re.compile(r"\b(?:number|count|numb|nb|num)\b\.?\s+of\b|\bcounts?\b", re.IGNORECASE)


def _nullable(col: ColumnInfo) -> bool:
    """VizieR marks columns that may be empty with '?' after the optional '[range]' prefix."""
    text = re.sub(r"^\s*\[[^\]]*\]", "", col.description or "").lstrip()
    return text.startswith("?")


def detect_identifier(columns: Sequence[ColumnInfo]) -> tuple[str | None, dict[str, str], list[str]]:
    """(identifier column or derived alias, derived ADQL columns, notes).

    1. ``meta.id;meta.main``;
    2. several ``meta.id.part;meta.main`` columns -> one composite id joined with '-' in ADQL
       (Tycho-2 TYC1/TYC2/TYC3 -> '425-2502-1'; TAPVizieR accepts ``||`` on numeric columns,
       verified live);
    3. ``recno``, the VizieR record number, unique by construction (VizieR: 'Should Not be used
       for identification' across releases, hence recorded as an assumption);
    4. a plain ``meta.id`` column that is neither nullable ('?') nor a count ('Number of
       positions used', Tycho-2 'Num'), as an assumption.
    """
    notes: list[str] = []
    main = _pick_best(columns, lambda c: ucd_atoms(c.ucd) == ["meta.id", "meta.main"])
    if main is not None:
        return main.name, {}, notes
    parts = [c for c in columns if ucd_atoms(c.ucd) == ["meta.id.part", "meta.main"]]
    if len(parts) >= 2:
        names = {c.name for c in columns}
        alias = "-".join(c.name for c in parts)[:60]
        while alias in names:
            alias = "_" + alias
        expr = " || '-' || ".join(quote_identifier(c.name) for c in parts)
        notes.append(f"identifier: composite of {', '.join(c.name for c in parts)} (meta.id.part;meta.main) joined "
                     f"with '-' as '{alias}'.")
        return alias, {alias: expr}, notes
    if len(parts) == 1:
        return parts[0].name, {}, notes
    recno = next((c for c in columns if c.name == "recno" and ucd_atoms(c.ucd)[:1] == ["meta.record"]), None)
    if recno is not None:
        notes.append("identifier: no meta.id;meta.main column; the VizieR record number 'recno' (unique within the "
                     "table, not a published designation) is used.")
        return recno.name, {}, notes
    plain = [c for c in columns if ucd_atoms(c.ucd) == ["meta.id"] and not _nullable(c)
             and not _COUNT_LIKE.search(_clean(c.description))]
    if plain:
        col = plain[0]
        notes.append(f"identifier: no meta.id;meta.main column; {col.name} ('{(col.description or '')[:60]}', "
                     "meta.id) assumed to be a unique identifier.")
        return col.name, {}, notes
    notes.append("identifier: no usable identifier column; sources are numbered <catalog>-<row>.")
    return None, {}, notes


def bibcode_reference(bibcode: str | None) -> str | None:
    """'2001MNRAS.328.1039C' -> 'MNRAS 328, 1039' (ADS 19-character bibcode: YYYYJJJJJVVVVMPPPPA)."""
    if not bibcode or len(bibcode) != 19 or not bibcode[:4].isdigit():
        return None
    journal = bibcode[4:9].strip(".")
    volume = bibcode[9:13].strip(".")
    page = (bibcode[13] + bibcode[14:18].lstrip(".")) if bibcode[13].isalpha() else bibcode[14:18].strip(".")
    reference = journal + (f" {volume}" if volume else "")
    return reference + (f", {page}" if page and page != "0" else "")


def _citation(info: CatalogInfo, table_id: str) -> str:
    """'Evans et al. 2020, ApJS 247, 54 (2020ApJS..247...54E); VizieR IX/58 (IX/58/2sxps), DOI ...'."""
    authors = None
    match = re.search(r"\(([^()]+),\s*((?:18|19|20|21)\d\d)\)\s*$", info.title or "")
    if match:
        who = match.group(1).strip()
        authors = (who[:-1].strip() + " et al." if who.endswith("+") else who) + f" {match.group(2)}"
    elif info.creator:
        authors = f"{info.creator} {info.year or ''}".strip()
    head = authors or "VizieR catalogue"
    if info.bibcode:
        reference = bibcode_reference(info.bibcode)
        head += f", {reference} ({info.bibcode})" if reference else f" ({info.bibcode})"
    tail = f"VizieR {info.catalog_id}" + (f" ({table_id})" if table_id != info.catalog_id else "")
    if info.doi:
        tail += f", DOI {info.doi}"
    return f"{head}; {tail}"


def _wavelength_of(info: CatalogInfo) -> tuple[str, str | None]:
    """(wavelength, note): from VizieR's -kw.Wavelength keywords; without keywords, VizieR
    section IX ('High-Energy data', TAP_SCHEMA.schemas IX_HE) gives 'high-energy'."""
    mapped = {VIZIER_WAVELENGTHS[w] for w in info.wavelengths if w in VIZIER_WAVELENGTHS}
    if len(mapped) == 1:
        return mapped.pop(), None
    if mapped:
        return "multi", None
    if info.catalog_id.split("/", 1)[0] == "IX":
        return "high-energy", ("wavelength: no VizieR wavelength keyword; 'high-energy' from VizieR section IX "
                               "(High-Energy data).")
    return "unknown", "wavelength: no VizieR wavelength keyword."


def analyse_table(
    table_id: str,
    catalog: CatalogInfo,
    table_row: Mapping[str, Any],
    columns: Sequence[ColumnInfo],
    asu_table: Mapping[str, Any] | None = None,
    *,
    max_ra: float | None = None,
) -> TableDescription:
    """Pure metadata analysis (no network): position/id/error/epoch columns and problems.

    ``max_ra`` (largest RA value in the table, only needed when TAP_SCHEMA gives the
    position columns no unit) confirms decimal degrees when it exceeds 24.
    """
    asu_table = asu_table or {}
    problems: list[str] = []
    assumptions: list[str] = []
    fields = asu_table.get("fields") or {}
    for col in columns:
        display = (fields.get(col.name) or {}).get("display")
        col.displayed = display not in (None, "0")
    rejected: list[ColumnInfo] = []
    ras = _position_candidates(columns, "ra", rejected)
    decs = _position_candidates(columns, "dec", rejected)
    all_coosys = asu_table.get("coosys") or {}

    def coosys_of(col: ColumnInfo) -> dict[str, Any]:
        return dict(all_coosys.get((fields.get(col.name) or {}).get("ref") or "", {}) or {})

    # The first candidate in a J2000/ICRS frame wins: an FK4 / old-equinox main position
    # (RAB1950, BD's RA1855, RA1900, NGC 2000.0's FK4 B2000) gives way to VizieR's
    # computed ICRS columns (_RA.icrs/_DE.icrs) when the table has them.
    old_frames: list[str] = []
    frame_conflicts: list[str] = []
    usable_decs = [d for d in decs if _old_frame(d, coosys_of(d)) is None]
    ra = dec = None
    for candidate in ras:
        reason = _old_frame(candidate, coosys_of(candidate), frame_conflicts)
        if reason is not None:
            old_frames.append(f"{candidate.name} is {reason}")
            continue
        paired = _pair_dec(candidate, usable_decs)
        if paired is not None:
            ra, dec = candidate, paired
            break
    if ra is None or dec is None:
        if old_frames:
            problems.append("; ".join(old_frames) + "; the table has no J2000/ICRS position column in decimal degrees "
                            "(VizieR computes _RA.icrs/_DE.icrs only for some tables), so it cannot be cone-searched "
                            "in ICRS")
        elif rejected:
            problems.append(
                "no equatorial position columns in decimal degrees: "
                + ", ".join(_position_reject_reason(c) for c in rejected)
                + " cannot be used in an ADQL cone search (POINT/CONTAINS need decimal degrees)")
        else:
            problems.append("no numeric equatorial position columns (UCD pos.eq.ra / pos.eq.dec, J2000/ICRS) -- "
                            "the table cannot be cone-searched")
        ra = dec = None
    else:
        if old_frames:
            assumptions.append("position: " + "; ".join(old_frames) + f"; using the J2000/ICRS columns "
                               f"{ra.name}/{dec.name} instead.")
        if rejected:
            assumptions.append("position: skipped " + ", ".join(_position_reject_reason(c) for c in rejected)
                               + f" (not decimal degrees); using {ra.name}/{dec.name}.")
    frame = None
    coosys: dict[str, Any] = {}
    ra_conflicts = [c for c in frame_conflicts if ra is not None and c.startswith(f"{ra.name}:")]
    if ra is not None and ra_conflicts:
        assumptions.append("position: " + "; ".join(ra_conflicts)
                           + " (VizieR's computed _RA.icrs/_DE.icrs, precessed from the COOSYS equinox, are not "
                           "used; the COOSYS epoch is disregarded too).")
        frame = f"J2000 ({ra.name} description: '{_stated_j2000(ra)}'; COOSYS disregarded)"
    elif ra is not None:
        coosys = coosys_of(ra)
        if coosys:
            frame = coosys.get("system")
            if coosys.get("equinox"):
                frame = f"{frame} (equinox {coosys['equinox']})"
        else:
            frame = "ICRS" if re.search(r"icrs", f"{ra.name} {ra.description}", re.IGNORECASE) else "J2000 (assumed ICRS-aligned)"
    unit_check = None
    if ra is not None and dec is not None and (not (ra.unit or "").strip() or not (dec.unit or "").strip()):
        if max_ra is not None and max_ra > 24.0:
            unit_check = (f"{ra.name}/{dec.name} carry no unit in TAP_SCHEMA; the largest {ra.name} "
                          f"({max_ra:.4f}) exceeds 24, confirming decimal degrees.")
        else:
            unit_check = (f"{ra.name}/{dec.name} carry no unit in TAP_SCHEMA; taken as decimal degrees (TAPVizieR "
                          "convention) -- not confirmed by the data"
                          + (f" (largest {ra.name} = {max_ra:.4f})." if max_ra is not None else "."))
            assumptions.append("position: " + unit_check)
    derived: dict[str, str] = {}
    id_name, id_derived, id_notes = detect_identifier(columns)
    derived.update(id_derived)
    assumptions.extend(id_notes)
    error_derived: dict[str, str] = {}
    pos_error, error_cols, error_notes = detect_pos_error(columns, ra, dec, derived=error_derived)
    derived.update({alias: expr for alias, expr in error_derived.items() if alias in (pos_error.get("columns") or [])})
    wavelength, wavelength_note = _wavelength_of(catalog)
    if wavelength_note:
        assumptions.append(wavelength_note)
    pos_error, error_notes = apply_error_conventions(pos_error, error_notes, catalog, table_id,
                                                     table_row.get("description"), error_cols)
    assumptions.extend(error_notes)
    if pos_error and pos_error.get("systematic_arcsec") is None and wavelength == "radio":
        assumptions.append("pos_error: the catalogued errors are statistical only and no astrometric systematic is "
                           "known for this survey; give overrides.systematic_arcsec / --systematic if the survey "
                           "documents one.")
    texts = [table_row.get("description"), asu_table.get("description"), catalog.title]
    epoch = detect_epoch(columns, ra, coosys, texts, dec=dec, fields=fields)
    derived.update(epoch.get("derived") or {})
    assumptions.extend(epoch["notes"])
    field_map, field_notes = detect_field_map_notes(columns, wavelength=wavelength)
    assumptions.extend(field_notes)
    return TableDescription(
        table_id=table_id,
        catalog=catalog,
        description=table_row.get("description"),
        nrows=int(table_row["nrows"]) if table_row.get("nrows") is not None else None,
        columns=list(columns),
        ra_column=ra.name if ra else None,
        dec_column=dec.name if dec else None,
        id_column=id_name,
        frame=frame,
        position_unit_check=unit_check,
        pos_error=pos_error,
        pos_error_columns=error_cols,
        epoch=epoch["epoch"],
        epoch_format=epoch["epoch_format"],
        epoch_range=epoch["epoch_range"],
        epoch_source=epoch["source"],
        single_epoch_positions=bool(epoch["single_epoch_positions"]),
        field_map=field_map,
        citation=_citation(catalog, table_id),
        wavelength=wavelength,
        assumptions=assumptions,
        problems=problems,
        derived_columns=derived,
    )


def _columns_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[ColumnInfo]:
    columns = []
    for row in rows:
        columns.append(ColumnInfo(
            name=_unquote(str(row.get("column_name") or "")),
            unit=(str(row["unit"]).strip() or None) if row.get("unit") not in (None, "") else None,
            ucd=(str(row["ucd"]).strip() or None) if row.get("ucd") not in (None, "") else None,
            datatype=row.get("datatype"),
            description=row.get("description"),
            principal=bool(row.get("principal")),
            indexed=bool(row.get("indexed")),
        ))
    return columns


async def _catalog_info(http: _Http, catalog_id: str) -> CatalogInfo:
    root = await http.asu({"-source": catalog_id, "-meta": ""})
    resources = [res for res in root.iter() if _local(res.tag) == "RESOURCE" and res.get("name")]
    # Single-table catalogues (e.g. IX/70) answer with the table's RESOURCE instead.
    for res in sorted(resources, key=lambda r: 0 if r.get("name") == catalog_id else 1):
        if res.get("name") == catalog_id or str(res.get("name")).startswith(catalog_id + "/"):
            info = parse_catalog_resource(res)
            if info is not None:
                info.catalog_id = catalog_id
                return info
    # An HTTP-200 error answer, or no RESOURCE for the catalogue: never an empty (cacheable)
    # description.
    error = asu_error_message(root)
    raise VizierUpstreamError(f"VizieR ASU returned no catalogue metadata for {catalog_id}"
                              + (f" (error: {error})" if error else ""))


def _regtap_citation_adql(catalog_id: str) -> str:
    """rr.resource row of the VizieR catalogue (RegTAP 1.1: VizieR ivoids are lower case)."""
    return ("SELECT ivoid, res_title, source_format, source_value, creator_seq, waveband FROM rr.resource "
            f"WHERE ivoid = {_adql_string('ivo://cds.vizier/' + catalog_id.lower())}")


def _first_author(creator_seq: str | None) -> str | None:
    """'Monet D.G.; Levine S.E.; Casian B.; et al.' -> 'Monet et al.'; 'Condon J.J.' -> 'Condon'."""
    names = [n.strip() for n in str(creator_seq or "").split(";") if n.strip()]
    if not names:
        return None
    surname = re.sub(r"\s+(?:[A-Z][a-z]?\.-?\s*)+$", "", names[0]).strip() or names[0]
    return surname + (" et al." if len(names) > 1 else "")


#: RegTAP (VOResource 1.1) waveband -> VizieR ``-kw.Wavelength`` keyword.
_WAVEBAND_TO_VIZIER: dict[str, str] = {band: keyword for keyword, band in _REGTAP_WAVEBAND.items()}
#: catalogue id -> monotonic time of the last failed RegTAP enrichment (negative cache).
_REGTAP_FAILURES: dict[str, float] = {}
_REGTAP_FAILURES_LOCK = threading.Lock()


def _regtap_recent_failure(catalog_id: str) -> float | None:
    """Seconds since RegTAP last failed for ``catalog_id`` (None when not failed recently)."""
    with _REGTAP_FAILURES_LOCK:
        when = _REGTAP_FAILURES.get(catalog_id)
        if when is None:
            return None
        age = time.monotonic() - when
        if age > REGTAP_NEGATIVE_TTL_SECONDS:
            del _REGTAP_FAILURES[catalog_id]
            return None
        return age


def _regtap_failed(catalog_id: str) -> None:
    with _REGTAP_FAILURES_LOCK:
        if len(_REGTAP_FAILURES) > 4096:
            _REGTAP_FAILURES.clear()
        _REGTAP_FAILURES[catalog_id] = time.monotonic()


async def _regtap_enrich(http: _Http, info: CatalogInfo) -> list[str]:
    """Fill a missing bibcode/creator/year/title and wavelength from the IVOA registry.

    ASU gives no 'cites' INFO for many flagship catalogues (2MASS II/246, USNO-B1 I/284, NVSS
    VIII/65, ...), and a single-table catalogue answers ``-source=<cat>&-meta`` with its table
    RESOURCE, which carries no ``-kw.Wavelength`` (II/246, II/349, VIII/92, IX/70, 4FGL, ...);
    rr.resource holds the reference bibcode (source_value) and the VOResource ``waveband``
    ('infrared', 'optical#infrared', 'x-ray', 'gamma-ray#radio'). Returns notes on what was
    filled. The lookup is optional: it runs with the short REGTAP_TIMEOUT_SECONDS budget and a
    failure is remembered for REGTAP_NEGATIVE_TTL_SECONDS, so repeated describes do not wait
    again (raises VizierUpstreamError; the caller records the failure)."""
    age = _regtap_recent_failure(info.catalog_id)
    if age is not None:
        raise VizierUpstreamError(f"IVOA RegTAP lookup skipped: it failed {age:.0f} s ago", transient=True)
    try:
        table = await http.tap(_regtap_citation_adql(info.catalog_id), url=REGTAP_URL, service="IVOA RegTAP",
                               fmt="votable", timeout=min(REGTAP_TIMEOUT_SECONDS, http.timeout))
    except VizierUpstreamError:
        _regtap_failed(info.catalog_id)
        raise
    notes: list[str] = []
    for row in table.rows:
        if not info.bibcode and str(row.get("source_format") or "").strip().lower() == "bibcode" \
                and row.get("source_value"):
            info.bibcode = str(row["source_value"]).strip()
            notes.append(f"citation: bibcode {info.bibcode} from the IVOA registry (VizieR ASU gives none).")
        author = _first_author(row.get("creator_seq"))
        if author and not info.creator:
            info.creator = author
        if not info.year and info.bibcode and info.bibcode[:4].isdigit():
            info.year = info.bibcode[:4]
        if not info.title and row.get("res_title"):
            info.title = str(row["res_title"])
        if not info.wavelengths:
            bands = [b.strip().lower() for b in str(row.get("waveband") or "").split("#") if b.strip()]
            keywords = [_WAVEBAND_TO_VIZIER[b] for b in bands if b in _WAVEBAND_TO_VIZIER]
            if keywords:
                info.wavelengths = list(dict.fromkeys(keywords))
                notes.append(f"wavelength: {'#'.join(bands)} from the IVOA registry (rr.resource.waveband; VizieR "
                             "ASU gives no wavelength keyword for this catalogue).")
        break
    return notes


# -- describe cache -------------------------------------------------------------

_DESCRIBE_CACHE: dict[str, tuple[float, TableDescription | CatalogInfo]] = {}
_DESCRIBE_CACHE_LOCK = threading.Lock()
_DESCRIBE_CACHE_MAX = 512


def clear_describe_cache() -> None:
    """Forget every cached describe() result and every remembered RegTAP failure."""
    with _DESCRIBE_CACHE_LOCK:
        _DESCRIBE_CACHE.clear()
    with _REGTAP_FAILURES_LOCK:
        _REGTAP_FAILURES.clear()


def _cache_get(key: str) -> TableDescription | CatalogInfo | None:
    if DESCRIBE_CACHE_TTL_SECONDS <= 0:
        return None
    with _DESCRIBE_CACHE_LOCK:
        item = _DESCRIBE_CACHE.get(key)
        if item is None:
            return None
        stored, value = item
        if time.monotonic() - stored > DESCRIBE_CACHE_TTL_SECONDS:
            del _DESCRIBE_CACHE[key]
            return None
    return copy.deepcopy(value)  # callers may mutate the result


def _cache_put(keys: Sequence[str], value: TableDescription | CatalogInfo) -> None:
    if DESCRIBE_CACHE_TTL_SECONDS <= 0 or (isinstance(value, TableDescription) and not value.complete):
        return
    frozen = copy.deepcopy(value)
    with _DESCRIBE_CACHE_LOCK:
        if len(_DESCRIBE_CACHE) >= _DESCRIBE_CACHE_MAX:
            for old in sorted(_DESCRIBE_CACHE, key=lambda k: _DESCRIBE_CACHE[k][0])[:_DESCRIBE_CACHE_MAX // 4]:
                del _DESCRIBE_CACHE[old]
        for key in dict.fromkeys(keys):
            _DESCRIBE_CACHE[key] = (time.monotonic(), frozen)


# -- describe -------------------------------------------------------------------


async def _table_rows(http: _Http, ident: str) -> dict[str, Mapping[str, Any]]:
    """TAP_SCHEMA.tables rows of ``ident`` itself and of the tables below it (bounded)."""
    tables = await http.tap(
        f"SELECT TOP {MAX_TABLES_PER_CATALOG + 1} table_name, description, nrows FROM TAP_SCHEMA.tables "
        f"WHERE table_name = {_adql_string(quote_identifier(ident))} "
        f"OR table_name LIKE {_adql_string(chr(34) + ident + '/%')}"
    )
    if len(tables.rows) > MAX_TABLES_PER_CATALOG:
        raise VizierInputError(f"'{ident}' matches more than {MAX_TABLES_PER_CATALOG} VizieR tables: it is not a "
                               "single catalogue; give a catalogue or table id, or use the search.")
    return {_unquote(str(r.get("table_name"))): r for r in tables.rows}


async def _canonical_id(http: _Http, ident: str) -> str | None:
    """VizieR's own spelling of a case-mismatched id ('ix/58/2sxps' -> 'IX/58/2sxps'). TAPVizieR
    compares table names case-sensitively and has no LOWER(), while the ASU service resolves
    ``-source`` case-insensitively and answers with the canonical RESOURCE name."""
    root = await http.asu({"-source": ident, "-meta": ""})
    for res in root.iter():
        name = res.get("name") if _local(res.tag) == "RESOURCE" else None
        if name and name.lower() == ident.lower():
            return name
    return None


async def _describe(http: _Http, ident: str, *, want_table: bool) -> TableDescription | CatalogInfo:
    rows = await _table_rows(http, ident)
    if ident not in rows and not any(catalog_of(t) == ident for t in rows):
        canonical = await _canonical_id(http, ident)
        if canonical and canonical != ident:
            ident = canonical
            rows = await _table_rows(http, ident)
    if ident in rows:
        return await _describe_table(http, ident, rows[ident])
    children = sorted(t for t in rows if catalog_of(t) == ident)
    if not children:
        raise VizierNotFoundError(f"VizieR has no table or catalogue '{ident}'.")
    if want_table and len(children) == 1:
        # A single-table catalogue: describe its table from the rows already fetched.
        return await _describe_table(http, children[0], rows[children[0]])
    info = await _catalog_info(http, ident)
    info.tables = [TableHit(table_id=t, catalog_id=ident, description=rows[t].get("description"),
                            nrows=int(rows[t]["nrows"]) if rows[t].get("nrows") is not None else None,
                            catalog_title=info.title, popularity=info.popularity,
                            wavelengths=list(info.wavelengths), bibcode=info.bibcode) for t in children]
    return info


async def _describe_cached(identifier: str, *, want_table: bool, client: httpx.AsyncClient | None,
                           timeout: float | None) -> TableDescription | CatalogInfo:
    ident = validate_vizier_id(identifier)
    _check_catalog_depth(ident)
    key = f"{'table' if want_table else 'any'}:{ident}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    async with _Http(client, timeout=timeout) as http:
        result = await _describe(http, ident, want_table=want_table)
    if isinstance(result, TableDescription):
        keys = [key, f"any:{result.table_id}", f"table:{result.table_id}"]
    else:
        keys = [key, f"any:{result.catalog_id}"]
    _cache_put(keys, result)
    return result


async def describe(identifier: str, *, client: httpx.AsyncClient | None = None,
                   timeout: float | None = None) -> TableDescription | CatalogInfo:
    """Describe a VizieR table (``'IX/58/2sxps'``) or list the tables of a catalogue (``'IX/58'``).

    Identifiers are matched case-insensitively ('ix/58/2sxps' is described as 'IX/58/2sxps');
    journal-level prefixes ('J/A+A') are refused before any query. Complete results are cached
    in-process for DESCRIBE_CACHE_TTL_SECONDS (``clear_describe_cache()`` empties the cache)."""
    return await _describe_cached(identifier, want_table=False, client=client, timeout=timeout)


async def describe_table(table_id: str, *, client: httpx.AsyncClient | None = None,
                         timeout: float | None = None) -> TableDescription:
    """Describe one VizieR table; a catalogue id with exactly one table is accepted (its table
    is described from the same TAP_SCHEMA answer, without a second lookup)."""
    result = await _describe_cached(table_id, want_table=True, client=client, timeout=timeout)
    if isinstance(result, TableDescription):
        return result
    raise VizierInputError(
        f"'{result.catalog_id}' is a catalogue with {len(result.tables)} tables; choose one of: "
        + ", ".join(t.table_id for t in result.tables)
    )


async def _optional[T](awaitable: Awaitable[T], what: str) -> tuple[T | None, str | None]:
    try:
        return await awaitable, None
    except VizierUpstreamError as exc:
        return None, f"{what} unavailable ({exc})"


def _align_asu_fields(columns: Sequence[ColumnInfo],
                      fields: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """ASU FIELDs keyed by TAP column name, and {TAP name: ASU name} for fields matched by UCD.

    TAPVizieR and the ASU ``-meta.all`` answer usually use the same column names, but not for
    some tables: Gaia DR1/DR2 (I/337/gaia, I/337/tgas, I/345/gaia2, ...) are ra/dec/source_id in
    TAP_SCHEMA and RA_ICRS/DE_ICRS/Source in ASU (the COOSYS H_2015.500 is attached to RA_ICRS).
    A TAP column absent from the ASU FIELDs is matched to the one unmatched ASU FIELD with the
    same UCD (and unit), when exactly one TAP column and one FIELD share that UCD."""
    aligned = {name: dict(value) for name, value in fields.items()}
    unmatched_tap = [c for c in columns if c.name not in fields and c.ucd]
    names = {c.name for c in columns}
    unmatched_asu = {name: value for name, value in fields.items() if name not in names and value.get("ucd")}
    matched: dict[str, str] = {}

    def key(ucd: Any) -> str:
        return ";".join(ucd_atoms(str(ucd)))

    for col in unmatched_tap:
        same_tap = [c for c in unmatched_tap if key(c.ucd) == key(col.ucd)]
        same_asu = [n for n, v in unmatched_asu.items() if key(v.get("ucd")) == key(col.ucd)]
        if len(same_tap) != 1 or len(same_asu) != 1:
            continue
        asu = unmatched_asu[same_asu[0]]
        unit_tap, unit_asu = (col.unit or "").strip(), str(asu.get("unit") or "").strip()
        if unit_tap and unit_asu and unit_tap != unit_asu:
            continue
        aligned[col.name] = dict(asu)
        matched[col.name] = same_asu[0]
    return aligned, matched


async def _describe_table(http: _Http, table_id: str, table_row: Mapping[str, Any]) -> TableDescription:
    cols_q = ("SELECT column_name, description, unit, ucd, datatype, principal, indexed FROM TAP_SCHEMA.columns "
              f"WHERE table_name = {_adql_string(quote_identifier(table_id))}")
    # The columns are essential; the catalogue metadata (bibcode, wavelength keywords) and the
    # ASU table metadata (COOSYS frame/epoch) degrade to recorded assumptions/problems. On the
    # first failure the sibling requests are cancelled (no orphan retries).
    cols_table, (info, info_error), (asu_root, asu_error) = await _gather_or_cancel(
        http.tap(cols_q),
        _optional(_catalog_info(http, catalog_of(table_id)), "VizieR ASU catalogue metadata"),
        _optional(http.asu({"-source": table_id, "-meta.all": ""}), "VizieR ASU table metadata"),
    )
    columns = _columns_from_rows(cols_table.rows)
    if not columns:
        raise VizierNotFoundError(f"TAP_SCHEMA lists no columns for '{table_id}'.")
    notes: list[str] = []
    complete = True
    if info is None:
        info = CatalogInfo(catalog_id=catalog_of(table_id))
        notes.append(f"catalogue: {info_error}; title, bibcode, DOI and wavelength keywords are missing.")
        complete = False
    if not info.bibcode or not info.wavelengths:
        found, regtap_error = await _optional(_regtap_enrich(http, info), "IVOA RegTAP citation/waveband lookup")
        if regtap_error:
            notes.append(f"citation: {regtap_error}.")
            complete = False
        notes.extend(found or [])
    asu_table = parse_table_meta(asu_root) if asu_root is not None else {}
    if asu_root is not None and (asu_table.get("error") or not asu_table.get("fields")):
        # VizieR answers metadata failures with HTTP 200 and an <INFO name="Error">: without the
        # FIELD/COOSYS elements the frame (FK4?) and the COOSYS epoch cannot be checked.
        reason = f"error '{asu_table['error']}'" if asu_table.get("error") else "no FIELD elements"
        asu_error = f"VizieR ASU table metadata unusable ({reason})"
        asu_table = {}
    aligned: dict[str, str] = {}
    if asu_table.get("fields"):
        asu_table = dict(asu_table)
        asu_table["fields"], aligned = _align_asu_fields(columns, asu_table["fields"])
    description = analyse_table(table_id, info, table_row, columns, asu_table)
    if description.ra_column and description.position_unit_check:
        # No unit on the position columns: the largest RA (one indexed TOP 1 query, ~1 s even
        # for 2MASS) tells decimal degrees (> 24) from hours. The column is selected under its
        # own name: TAPVizieR's JSON renames '_RA.icrs' to '_RA_icrs' unless it is aliased.
        ra_q = quote_identifier(description.ra_column)
        top = await http.tap(f"SELECT TOP 1 {select_item(description.ra_column)} FROM {quote_identifier(table_id)} "
                             f"WHERE {ra_q} IS NOT NULL ORDER BY {ra_q} DESC")
        values = [v for v in (_float(r.get(description.ra_column)) for r in top.rows) if v is not None]
        description = analyse_table(table_id, info, table_row, columns, asu_table,
                                    max_ra=max(values) if values else None)
    description.assumptions[:0] = notes
    position_aliases = [f"{tap} = ASU FIELD {aligned[tap]}" for tap in (description.ra_column, description.dec_column)
                        if tap in aligned]
    if position_aliases:
        description.assumptions.append(
            "position: TAPVizieR and the VizieR ASU metadata name the position columns differently ("
            + ", ".join(position_aliases) + ", matched by UCD); the frame and COOSYS epoch are those of the ASU FIELDs.")
    if asu_error:
        # An outage (or an HTTP-200 error answer) of VizieR's metadata service: transient, so the
        # registration answers 502 (retry later), not 422.
        problem = (f"{asu_error}: the coordinate frame and the COOSYS epoch of the positions could not be checked; "
                   "retry later")
        description.problems.append(problem)
        description.upstream_problems.append(problem)
        complete = False
    elif description.ra_column and description.ra_column not in (asu_table.get("fields") or {}):
        description.problems.append(
            f"VizieR metadata mismatch: the ASU table metadata has no FIELD for {description.ra_column} (neither by "
            "name nor by UCD), so its coordinate frame and COOSYS epoch cannot be checked (the two VizieR services "
            "describe this table inconsistently; retrying does not help)")
        complete = False
    if _per_row_epoch(description.epoch) and description.epoch_range is None and not description.problems:
        range_notes, range_ok = await _determine_epoch_range(http, description)
        description.assumptions.extend(range_notes)
        complete = complete and range_ok
    description.complete = complete
    return description


# -- epoch span of per-row-epoch tables -------------------------------------------

#: Tables up to this many rows get an exact MIN/MAX scan of their epoch column(s) (TAPVizieR:
#: FIRST 946k rows 2 s, eRASS1 930k rows 2-13 s; 2MASS's 471M rows time out after 90 s).
EPOCH_SCAN_MAX_ROWS = int(os.getenv("VIZIER_EPOCH_SCAN_MAX_ROWS", "2000000"))
#: Larger tables are sampled with small cones spread over the sky (positional index).
EPOCH_SAMPLE_DECS = (-75.0, -45.0, -15.0, 15.0, 45.0, 75.0)
EPOCH_SAMPLE_RAS = 8
EPOCH_SAMPLE_RADIUS_DEG = 0.1
#: A sampled span is widened by this fraction of itself on each side (at least
#: EPOCH_SAMPLE_MIN_MARGIN_YEARS): 48 cones found 1997.77-2001.12 for 2MASS (true 1997.4-2001.2)
#: and 2012.31-2014.47 for URAT1 (true 2012.3-2015.0), live 2026-09-29.
EPOCH_SAMPLE_MARGIN = 0.25
EPOCH_SAMPLE_MIN_MARGIN_YEARS = 0.5
#: Plausible raw values of each epoch format (filters sentinels such as 0 or 9999).
_EPOCH_VALUE_BOUNDS: dict[str, tuple[float, float]] = {
    "jyear": (1800.0, 2200.0), "mjd": (-21000.0, 124000.0), "jd": (2378500.0, 2524600.0),
}


def _per_row_epoch(epoch: Any) -> bool:
    """True for per-row epochs (column name(s) or a span mapping), not for a fixed year/None."""
    if isinstance(epoch, str):
        return True
    if isinstance(epoch, (list, tuple)):
        return bool(epoch)
    return isinstance(epoch, Mapping) and bool(epoch.get("span_columns"))


def _epoch_expressions(description: TableDescription) -> tuple[list[tuple[str, str]], str | None]:
    """[(ADQL expression, 'lo'|'hi'|'both')] of the epoch column(s) and their format."""
    epoch = description.epoch
    fmt = description.epoch_format

    def expr(name: str) -> str:
        return description.derived_columns.get(name) or quote_identifier(name)

    if isinstance(epoch, Mapping):
        start, end = [str(c) for c in epoch.get("span_columns") or []][:2]
        return [(expr(start), "lo"), (expr(end), "hi")], str(epoch.get("format") or fmt or "") or None
    names = [epoch] if isinstance(epoch, str) else [str(c) for c in epoch]
    return [(expr(n), "both") for n in names], fmt


def _epoch_filter(expression: str, fmt: str | None) -> str:
    bounds = _EPOCH_VALUE_BOUNDS.get(str(fmt or ""))
    if bounds is None:
        return f"{expression} IS NOT NULL"
    return f"{expression} BETWEEN {bounds[0]:g} AND {bounds[1]:g}"


def _years(values: Sequence[Any], fmt: str | None) -> list[float]:
    return [y for y in (epoch_to_jyear(v, fmt) for v in values if v is not None) if y is not None]


def _core_epoch_range(table_id: str) -> tuple[list[float], str] | None:
    """The epoch_range the core registry documents for the same TAPVizieR table."""
    quoted = quote_identifier(table_id)
    for core_name, entry in DEFAULT_CATALOGS.items():
        span = (entry.get("parameters") or {}).get("epoch_range")
        if entry.get("endpoint") == VIZIER_TAP and entry.get("table") == quoted and isinstance(span, (list, tuple)) \
                and len(span) == 2 and all(_float(v) is not None for v in span):
            return [float(span[0]), float(span[1])], core_name
    return None


async def _scan_epoch_range(http: _Http, table_id: str, exprs: Sequence[tuple[str, str]],
                            fmt: str | None, where: str | None = None) -> tuple[float, float] | None:
    """MIN/MAX of the epoch expression(s) over the table (or the rows matching ``where``)."""
    items = []
    for i, (expression, role) in enumerate(exprs):
        filt = _epoch_filter(expression, fmt)
        if role in ("lo", "both"):
            items.append((f"MIN({expression})", f"lo{i}", filt))
        if role in ("hi", "both"):
            items.append((f"MAX({expression})", f"hi{i}", filt))
    lows: list[float] = []
    highs: list[float] = []
    # One query per filter (every aggregate of a query shares its WHERE clause).
    for filt in dict.fromkeys(f for _e, _a, f in items):
        select = ", ".join(f"{agg} AS {alias}" for agg, alias, f in items if f == filt)
        clause = f"({filt})" + (f" AND ({where})" if where else "")
        table = await http.tap(f"SELECT {select} FROM {quote_identifier(table_id)} WHERE {clause}")
        for row in table.rows[:1]:
            lows += _years([row.get(k) for k in row if str(k).startswith("lo")], fmt)
            highs += _years([row.get(k) for k in row if str(k).startswith("hi")], fmt)
    if not lows or not highs:
        return None
    return min(lows), max(highs)


async def _sample_epoch_range(http: _Http, description: TableDescription, exprs: Sequence[tuple[str, str]],
                              fmt: str | None) -> tuple[tuple[float, float] | None, int, list[str]]:
    """MIN/MAX of the epochs inside small cones spread over the sky (indexed positional
    queries; a full scan of a 10^8-10^9-row table times out). Returns (span, cones with
    rows, errors)."""
    ra_q, dec_q = quote_identifier(description.ra_column or ""), quote_identifier(description.dec_column or "")
    semaphore = asyncio.Semaphore(TAP_CONCURRENCY)
    errors: list[str] = []

    async def cone(ra: float, dec: float) -> tuple[float, float] | None:
        where = (f"1 = CONTAINS(POINT('ICRS', {ra_q}, {dec_q}), "
                 f"CIRCLE('ICRS', {ra:.4f}, {dec:.4f}, {EPOCH_SAMPLE_RADIUS_DEG:g}))")
        async with semaphore:
            try:
                return await _scan_epoch_range(http, description.table_id, exprs, fmt, where)
            except VizierUpstreamError as exc:
                errors.append(str(exc))
                return None

    points = [((j + 0.5 * (i % 2)) * 360.0 / EPOCH_SAMPLE_RAS, dec)
              for i, dec in enumerate(EPOCH_SAMPLE_DECS) for j in range(EPOCH_SAMPLE_RAS)]
    spans = [s for s in await asyncio.gather(*(cone(ra, dec) for ra, dec in points)) if s is not None]
    if not spans:
        return None, 0, errors
    return (min(s[0] for s in spans), max(s[1] for s in spans)), len(spans), errors


async def _determine_epoch_range(http: _Http, description: TableDescription) -> tuple[list[str], bool]:
    """Set ``description.epoch_range`` for a per-row-epoch table whose metadata gives no span.

    models.plan_cone needs the span of a per-row-epoch catalog (``parameters.epoch_range``) to
    move and widen the cone for a moving target; without it the cone is not propagated at all
    and fast movers are missed (Barnard's star in 2MASS at 3"). In order: the span the core
    registry documents for the same TAPVizieR table; an exact MIN/MAX scan for tables of at
    most EPOCH_SCAN_MAX_ROWS rows; otherwise 48 small cones spread over the sky, the sampled
    span widened by EPOCH_SAMPLE_MARGIN. Returns (notes, complete); complete is False when
    the span could not be determined because of an upstream failure (not cached then)."""
    core = _core_epoch_range(description.table_id)
    if core is not None:
        description.epoch_range = core[0]
        return [(f"epoch: per-row epochs span {core[0][0]:g}-{core[0][1]:g} (the core '{core[1]}' entry for the "
                 "same VizieR table).")], True
    exprs, fmt = _epoch_expressions(description)
    hint = " Give overrides.epoch_range ([earliest, latest] Julian years) from the survey documentation."
    nrows = description.nrows
    if nrows is not None and nrows <= EPOCH_SCAN_MAX_ROWS:
        try:
            span = await _scan_epoch_range(http, description.table_id, exprs, fmt)
        except VizierUpstreamError as exc:
            return [f"epoch: the span of the per-row epochs could not be determined ({exc}); cones are not "
                    "epoch-propagated for this catalog." + hint], False
        if span is None:
            return ["epoch: the per-row epoch column has no plausible values; cones are not epoch-propagated for "
                    "this catalog." + hint], True
        description.epoch_range = [round(span[0], 3), round(span[1], 3)]
        return [f"epoch: per-row epochs span {span[0]:.3f}-{span[1]:.3f} (MIN/MAX over all {nrows:,} rows)."], True
    span, cones, errors = await _sample_epoch_range(http, description, exprs, fmt)
    total = len(EPOCH_SAMPLE_DECS) * EPOCH_SAMPLE_RAS
    if span is None:
        if errors:
            return [f"epoch: the span of the per-row epochs could not be sampled ({errors[0]}); cones are not "
                    "epoch-propagated for this catalog." + hint], False
        return [f"epoch: none of {total} sample cones held rows with a plausible epoch; the span of the per-row "
                "epochs is unknown and cones are not epoch-propagated for this catalog." + hint], True
    margin = max(EPOCH_SAMPLE_MARGIN * (span[1] - span[0]), EPOCH_SAMPLE_MIN_MARGIN_YEARS)
    lo, hi = max(1800.0, span[0] - margin), min(2200.0, span[1] + margin)
    description.epoch_range = [round(lo, 3), round(hi, 3)]
    return [f"epoch: per-row epochs sampled in {cones} of {total} cones of radius {EPOCH_SAMPLE_RADIUS_DEG:g} deg "
            f"span {span[0]:.3f}-{span[1]:.3f}; epoch_range widened by {margin:.2f} yr on each side to "
            f"{lo:.3f}-{hi:.3f} (an estimate; the table is too large to scan)." + hint], not errors


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_OVERRIDE_KEYS = frozenset({
    "wavelength", "max_rows", "timeout_seconds", "pos_error", "epoch", "epoch_format", "epoch_range",
    "systematic_arcsec", "profiles", "enabled", "description", "coverage", "extra_columns", "id_column",
})
# Canonical fields a stray selected column could feed through models.ucd_field_map.
_GUARDED_CANONICAL = frozenset({"ra", "dec", "source_id", "pmra", "pmdec", "parallax", "redshift", "object_type",
                                "spectral_type", "epoch", "ra_error_arcsec", "dec_error_arcsec"})


def _table_title(text: str | None) -> str | None:
    """TAP_SCHEMA table description without VizieR's trailing '( authors)' list."""
    if not text:
        return None
    return re.sub(r"\s*\(\s[^()]*\)\s*$", "", text).strip() or text


#: Wavelength names accepted for the ``wavelength`` override (AstroSearch registry names).
_WAVELENGTH_NAMES = frozenset({*_WAVELENGTH_PROFILES, "multi", "unknown"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _string_list(key: str, value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v.strip() for v in value):
        raise VizierInputError(f"override {key!r} must be a list of non-empty strings, got {value!r}")
    return [v.strip() for v in value]


def validate_catalog_name(name: str | None, table_id: str) -> str:
    """Registry name for ``table_id`` (explicit ``name`` or ``vizier_<table>``), validated."""
    if name is not None and not isinstance(name, str):
        raise VizierInputError(f"Invalid catalog name {name!r}")
    catalog_name = (name or default_catalog_name(table_id)).strip().lower()
    if not _NAME.match(catalog_name):
        raise VizierInputError(f"Invalid catalog name {catalog_name!r}: lower-case letters, digits, '_' or '-' (max 64).")
    return catalog_name


def _epoch_value_ok(value: Any) -> bool:
    """A Julian year, or an [earliest, latest] span of Julian years (1800..2200)."""
    if _is_number(value):
        return 1800 <= float(value) <= 2200
    return (isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value)
            and all(1800 <= float(v) <= 2200 for v in value) and float(value[0]) <= float(value[1]))


def _validate_epoch_lookup(value: Mapping[str, Any]) -> None:
    """An epoch lookup ``{"column", "values": {key: year | [lo, hi]}, "default_range"?,
    "pm_columns"?, "pm_epoch"?}`` (models.resolve_epoch); the columns are checked against the
    table by build_definition."""
    if not isinstance(value.get("column"), str) or not value["column"].strip() \
            or not isinstance(value.get("values"), Mapping) or not value["values"]:
        raise VizierInputError("override 'epoch' mapping needs 'span_columns' or 'column' + a non-empty 'values' "
                               "mapping")
    bad = {str(k): v for k, v in value["values"].items() if not _epoch_value_ok(v)}
    if bad:
        raise VizierInputError(f"override 'epoch.values' must map to Julian years or [earliest, latest] spans "
                               f"(1800..2200), got {bad!r}")
    if value.get("default_range") is not None and not (isinstance(value["default_range"], (list, tuple))
                                                        and _epoch_value_ok(value["default_range"])):
        raise VizierInputError(f"override 'epoch.default_range' must be [earliest, latest] Julian years, "
                               f"got {value['default_range']!r}")
    if value.get("pm_columns") is not None:
        _string_list("epoch.pm_columns", value["pm_columns"])
    if value.get("pm_epoch") is not None and not (_is_number(value["pm_epoch"]) and _epoch_value_ok(value["pm_epoch"])):
        raise VizierInputError(f"override 'epoch.pm_epoch' must be a Julian year, got {value['pm_epoch']!r}")


def validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Type-check registration overrides (before any network call); raises VizierInputError.

    wavelength: an AstroSearch wavelength name (or a VizieR keyword such as 'X-ray');
    max_rows: integer 1..100000; timeout_seconds: positive number; pos_error: a mapping
    (``columns`` list of strings, ``kind``, ``units`` string or list, numeric ``divisor`` /
    ``systematic_arcsec``); epoch: Julian year, column name, list of column names, a
    ``span_columns``/lookup mapping, or null; epoch_format: jyear/mjd/jd/null; epoch_range:
    [earliest, latest] Julian years or null; systematic_arcsec: number >= 0; profiles /
    extra_columns: lists of strings; enabled: true/false; description, coverage, id_column:
    strings."""
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise VizierInputError(f"overrides must be a JSON object, got {type(overrides).__name__}")
    out = dict(overrides)
    unknown = set(out) - _OVERRIDE_KEYS
    if unknown:
        raise VizierInputError(f"Unknown override(s): {sorted(map(str, unknown))}; allowed: {sorted(_OVERRIDE_KEYS)}")
    if "wavelength" in out:
        value = out["wavelength"]
        if not isinstance(value, str) or not value.strip():
            raise VizierInputError(f"override 'wavelength' must be a string, got {value!r}")
        text = value.strip()
        if text.lower() not in _WAVELENGTH_NAMES:
            try:
                text = VIZIER_WAVELENGTHS[normalize_wavelength(text) or ""]
            except (VizierInputError, KeyError):
                raise VizierInputError(f"override 'wavelength' {value!r} is not one of "
                                       f"{sorted(_WAVELENGTH_NAMES)}") from None
        out["wavelength"] = text.lower()
    if "max_rows" in out:
        value = out["max_rows"]
        if not _is_number(value) or float(value) != int(value) or not 1 <= int(value) <= 100_000:
            raise VizierInputError(f"override 'max_rows' must be an integer in 1..100000, got {value!r}")
        out["max_rows"] = int(value)
    if "timeout_seconds" in out:
        value = out["timeout_seconds"]
        if not _is_number(value) or not 0 < float(value) <= 3600:
            raise VizierInputError(f"override 'timeout_seconds' must be a number in (0, 3600], got {value!r}")
        out["timeout_seconds"] = float(value)
    if "systematic_arcsec" in out:
        value = out["systematic_arcsec"]
        if not _is_number(value) or float(value) < 0:
            raise VizierInputError(f"override 'systematic_arcsec' must be a non-negative number, got {value!r}")
        out["systematic_arcsec"] = float(value)
    if "pos_error" in out and out["pos_error"] is not None:
        spec = out["pos_error"]
        if not isinstance(spec, Mapping):
            raise VizierInputError("override 'pos_error' must be an object such as {\"columns\": [\"Err90\"], "
                                   f"\"units\": [\"arcsec\"], \"kind\": \"radius90\"}}, got {spec!r}")
        spec = dict(spec)
        if "columns" in spec:
            spec["columns"] = _string_list("pos_error.columns", spec["columns"])
        units = spec.get("units", spec.get("unit"))
        if units is not None and not isinstance(units, str) and not (
                isinstance(units, (list, tuple)) and all(u is None or isinstance(u, str) for u in units)):
            raise VizierInputError(f"override 'pos_error.units' must be a string or a list of strings, got {units!r}")
        if "kind" in spec and not isinstance(spec["kind"], str):
            raise VizierInputError(f"override 'pos_error.kind' must be a string, got {spec['kind']!r}")
        for key in ("divisor", "systematic_arcsec", "scale", "floor_arcsec", "default_arcsec"):
            if key in spec and spec[key] is not None and (not _is_number(spec[key]) or float(spec[key]) < 0):
                raise VizierInputError(f"override 'pos_error.{key}' must be a non-negative number, got {spec[key]!r}")
        out["pos_error"] = spec
    if "epoch" in out:
        value = out["epoch"]
        if value is None or isinstance(value, str) and value.strip():
            pass
        elif _is_number(value):
            if not 1800 <= float(value) <= 2200:
                raise VizierInputError(f"override 'epoch' {value!r} is not a plausible Julian year (1800..2200)")
            out["epoch"] = float(value)
        elif isinstance(value, (list, tuple)):
            out["epoch"] = _string_list("epoch", value)
        elif isinstance(value, Mapping):
            if "span_columns" in value:
                if len(_string_list("epoch.span_columns", value["span_columns"])) != 2:
                    raise VizierInputError("override 'epoch.span_columns' must name the [start, end] columns")
                limit = value.get("point_max_years")
                if limit is not None and (not _is_number(limit) or float(limit) < 0):
                    raise VizierInputError(f"override 'epoch.point_max_years' must be a non-negative number, got {limit!r}")
                if value.get("format") not in (None, "jyear", "mjd", "jd"):
                    raise VizierInputError(f"override 'epoch.format' must be jyear, mjd or jd, got {value.get('format')!r}")
            else:
                _validate_epoch_lookup(value)
            out["epoch"] = dict(value)
        else:
            raise VizierInputError("override 'epoch' must be a Julian year, a column name, a list of column names, "
                                   f"an epoch mapping or null, got {value!r}")
    if "epoch_format" in out and out["epoch_format"] not in (None, "jyear", "mjd", "jd"):
        raise VizierInputError(f"override 'epoch_format' must be jyear, mjd, jd or null, got {out['epoch_format']!r}")
    if "epoch_range" in out and out["epoch_range"] is not None:
        value = out["epoch_range"]
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_is_number(v) for v in value)
                or not all(1800 <= float(v) <= 2200 for v in value) or float(value[0]) > float(value[1])):
            raise VizierInputError(f"override 'epoch_range' must be [earliest, latest] Julian years, got {value!r}")
        out["epoch_range"] = [float(value[0]), float(value[1])]
    for key in ("profiles", "extra_columns"):
        if key in out and out[key] is not None:
            out[key] = _string_list(key, out[key])
    if "enabled" in out and not isinstance(out["enabled"], bool):
        raise VizierInputError(f"override 'enabled' must be true or false, got {out['enabled']!r}")
    for key in ("description", "coverage", "id_column"):
        if key in out and out[key] is not None and (not isinstance(out[key], str) or not out[key].strip()):
            raise VizierInputError(f"override {key!r} must be a non-empty string, got {out[key]!r}")
    return out


def _fill_pos_error_units(spec: dict[str, Any], by_name: Mapping[str, ColumnInfo], derived: Mapping[str, str],
                          table_id: str, notes: list[str]) -> None:
    """Complete a ``pos_error`` override that names columns but no units from the columns'
    TAP_SCHEMA units (an RA error in seconds of time becomes 's_ra'); without units
    models.compute_positional_error returns no error for any row. A column without an angular
    unit (or a derived expression) needs explicit units -> VizierInputError."""
    columns = [str(c) for c in spec.get("columns") or []]
    if not columns:
        return
    given = spec.pop("units", spec.pop("unit", None))
    if isinstance(given, str) and given.strip():
        spec["units"] = given
        return
    units = list(given) if isinstance(given, (list, tuple)) else []
    units += [None] * (len(columns) - len(units))
    kind = str(spec.get("kind") or "sigma")
    filled = []
    for i, name in enumerate(columns):
        if units[i] not in (None, ""):
            continue
        if kind.startswith("ellipse") and i == 2:
            units[i] = "deg"  # the position angle (compute_positional_error reads it in degrees)
            continue
        col = by_name.get(name)
        if col is None:
            if name in derived:
                raise VizierInputError(f"override pos_error column {name!r} is a derived expression; give its units "
                                       "(pos_error.units)")
            continue  # reported as a missing column by the caller
        axis = ("ra" if i == 0 else "dec" if i == 1 else None) if kind == "sigma" and len(columns) >= 2 else None
        unit = _error_unit(col, axis)
        if unit is None:
            raise VizierInputError(f"override pos_error column {name!r} has no angular unit in TAP_SCHEMA "
                                   f"(unit {col.unit!r}); give pos_error.units explicitly")
        units[i] = unit
        filled.append(f"{name} '{unit}'")
    spec["units"] = units[:len(columns)]
    if filled:
        notes.append(f"pos_error: override without units; units taken from TAP_SCHEMA: {', '.join(filled)}.")


def _override_epoch_format(epoch: Any, by_name: Mapping[str, ColumnInfo]) -> str | None:
    """Format (jyear/mjd/jd) of an overridden per-row epoch, read from its column metadata."""
    if isinstance(epoch, Mapping):
        if epoch.get("format"):
            return str(epoch["format"])
        names = [str(c) for c in epoch.get("span_columns") or []]
    elif isinstance(epoch, str):
        names = [epoch]
    elif isinstance(epoch, (list, tuple)):
        names = [str(c) for c in epoch]
    else:
        return None
    formats = {f for f in (_epoch_format(by_name[n]) for n in names if n in by_name) if f}
    return formats.pop() if len(formats) == 1 else None


def _check_epoch_override_columns(epoch: Any, by_name: Mapping[str, ColumnInfo], table_id: str) -> None:
    """Epoch overrides must name numeric columns (a CHAR designation resolves to no epoch at
    match time); a span needs two distinct columns; lookup pm_columns must be numeric."""
    def numeric(names: Sequence[str], what: str) -> None:
        for name in names:
            col = by_name.get(name)
            if col is not None and not col.numeric:
                raise VizierInputError(f"override {what} column {name!r} of {table_id} is not numeric "
                                       f"({col.datatype}); an epoch column must hold JD/MJD/Julian-year values")
    if isinstance(epoch, str):
        numeric([epoch], "epoch")
    elif isinstance(epoch, (list, tuple)):
        numeric([str(c) for c in epoch], "epoch")
    elif isinstance(epoch, Mapping):
        span = [str(c) for c in epoch.get("span_columns") or []]
        if span:
            if len(set(span)) != len(span):
                raise VizierInputError(f"override epoch.span_columns must name two distinct columns (start, end), "
                                       f"got {span!r}")
            numeric(span, "epoch.span_columns")
        numeric([str(c) for c in epoch.get("pm_columns") or []], "epoch.pm_columns")


def build_definition(
    description: TableDescription,
    *,
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Registry entry (the dict format of models.DEFAULT_CATALOGS / the registry YAML).

    Columns: identifier, position, positional-error, epoch and canonical physical columns, then
    the table's principal columns (TAP_SCHEMA ``principal``) up to 40. A column whose UCD would
    silently feed a canonical field (``models.ucd_field_map``: another RA/Dec such as the
    B1950 main position beside ``_RA.icrs``, another main identifier, epoch, proper motion,
    redshift, ...) is only selected when it is the column chosen for that field. Columns whose
    name contains '.' are selected under their own name (:func:`select_item`), so the registry
    references them by their real name. Overrides: see :func:`validate_overrides`; an explicit
    ``pos_error: {}`` (or null) disables the detected positional error; a ``pos_error`` override
    without units gets the columns' TAP_SCHEMA units (refused when a column has no angular unit).
    Epoch overrides must name numeric columns (a span needs two distinct ones); an overridden
    per-row epoch without ``epoch_range`` keeps no span (:func:`register_table` determines one).
    Choices made here are appended to ``notes``. Raises :class:`VizierRegistrationError` when the
    table cannot be cone-searched or the definition fails
    :func:`models.validate_catalog_definition`, and a transient :class:`VizierUpstreamError`
    when the description is incomplete because of an upstream outage.
    """
    notes = notes if notes is not None else []
    overrides = validate_overrides(overrides)
    catalog_name = validate_catalog_name(name, description.table_id)
    if description.upstream_problems:
        raise VizierUpstreamError(f"{description.table_id} cannot be registered right now: "
                                  + "; ".join(description.upstream_problems), transient=True)
    if description.problems:
        raise VizierRegistrationError(f"{description.table_id} cannot be registered: " + "; ".join(description.problems))
    by_name = {c.name: c for c in description.columns}
    derived = dict(description.derived_columns)
    ra, dec = description.ra_column or "", description.dec_column or ""
    id_col = overrides.pop("id_column", None) or description.id_column
    if id_col is not None and id_col not in by_name and id_col not in derived:
        raise VizierInputError(f"id_column {id_col!r} is not a column of {description.table_id}")

    if "pos_error" in overrides:
        pos_error = dict(overrides.pop("pos_error") or {})  # {} / null: no positional error
        _fill_pos_error_units(pos_error, by_name, derived, description.table_id, notes)
    else:
        pos_error = dict(description.pos_error)
    systematic = overrides.pop("systematic_arcsec", None)
    if systematic is not None:
        value = _float(systematic)
        if value is None or value < 0:
            raise VizierInputError("systematic_arcsec must be a non-negative number")
        if not pos_error:
            raise VizierInputError("systematic_arcsec needs a positional-error column (or a pos_error override)")
        pos_error["systematic_arcsec"] = value
    epoch_overridden = "epoch" in overrides
    epoch = overrides.pop("epoch", description.epoch)
    same_epoch = not epoch_overridden or epoch == description.epoch
    if "epoch_format" in overrides:
        epoch_format = overrides.pop("epoch_format")
    elif same_epoch:
        epoch_format = description.epoch_format
    else:
        epoch_format = _override_epoch_format(epoch, by_name)
        if epoch_format and _per_row_epoch(epoch):
            notes.append(f"epoch: format of the overridden epoch {epoch!r} read as '{epoch_format}' from its column "
                         "metadata (override epoch_format to change it).")
    default_range = description.epoch_range if same_epoch else None
    epoch_range = overrides.pop("epoch_range", default_range)
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        epoch_range = None  # a fixed epoch needs no span (and the validator rejects both)

    required: list[str] = [c for c in (id_col, ra, dec) if c]
    required += [str(c) for c in pos_error.get("columns") or []]
    if isinstance(epoch, str):
        required.append(epoch)
    elif isinstance(epoch, Mapping):
        required += [str(c) for c in epoch.get("span_columns") or []]
        if epoch.get("column"):  # an epoch lookup {"column", "values"} and its pm_columns
            required.append(str(epoch["column"]))
        required += [str(c) for c in epoch.get("pm_columns") or []]
    elif isinstance(epoch, list):
        required += [str(c) for c in epoch]
    required += list(description.field_map.values())
    extra = [str(c) for c in overrides.pop("extra_columns", None) or []]
    missing = [c for c in required + extra if c not in by_name and c not in derived]
    if missing:
        raise VizierInputError(f"column(s) {missing} are not in {description.table_id}")
    if epoch_overridden and not same_epoch:
        _check_epoch_override_columns(epoch, by_name, description.table_id)
        if _per_row_epoch(epoch) and not epoch_range:
            notes.append(f"epoch: overridden to the per-row epoch {epoch!r} without an epoch_range: cones are not "
                         "epoch-propagated for a moving target (give overrides.epoch_range, or register with "
                         "register_table, which determines the span).")

    chosen: dict[str, str] = {canon: col for canon, col in description.field_map.items()}
    chosen.update({"ra": ra, "dec": dec})
    if id_col:
        chosen["source_id"] = id_col
    if isinstance(epoch, str):
        chosen["epoch"] = epoch

    def stray_roles(col: ColumnInfo) -> set[str]:
        """Canonical fields ``col`` would feed through its UCD although it is not their column."""
        roles = set(ucd_field_map([col.as_meta()])) & _GUARDED_CANONICAL
        if "source_id" in roles and ucd_atoms(col.ucd) != ["meta.id", "meta.main"]:
            roles.discard("source_id")  # plain meta.id columns (other names) are harmless beside a main id
        return {r for r in roles if chosen.get(r) != col.name}

    for col_name in extra:
        col = by_name.get(col_name)
        position_roles = stray_roles(col) & {"ra", "dec", "source_id"} if col is not None else set()
        if position_roles:
            raise VizierInputError(
                f"extra column {col_name!r} ({col.ucd}) would be read as the {'/'.join(sorted(position_roles))} of "  # type: ignore[union-attr]
                f"rows whose chosen column ({', '.join(chosen[r] or '-' for r in sorted(position_roles))}) is empty; "
                "it cannot be selected")
    selected: list[str] = []
    for col_name in required + extra:
        if col_name not in selected:
            selected.append(col_name)
    for col in description.columns:
        if len(selected) >= MAX_SELECTED_COLUMNS:
            break
        if not (col.principal or col.displayed) or col.name in selected:
            continue
        if stray_roles(col):
            continue
        selected.append(col.name)
    used_derived = {alias: expr for alias, expr in derived.items() if alias in selected}

    column_units = {}
    for col_name in (ra, dec):
        if col_name and not (by_name[col_name].unit or "").strip():
            column_units[col_name] = "deg"
    wavelength = str(overrides.pop("wavelength", None) or description.wavelength)
    profiles = overrides.pop("profiles", None)
    if profiles is None:
        bands = [wavelength]
        if wavelength == "multi":  # e.g. RegTAP 'optical#infrared': every band's profiles
            bands = [VIZIER_WAVELENGTHS[w] for w in description.catalog.wavelengths if w in VIZIER_WAVELENGTHS]
        profiles = ["full", "vizier"]
        for band in bands:
            profiles += [p for p in _WAVELENGTH_PROFILES.get(band, ()) if p not in profiles]
    parameters: dict[str, Any] = {
        "columns": [f"{used_derived[c]} AS {quote_identifier(c)}" if c in used_derived else select_item(c)
                    for c in selected],
        "id_field": quote_identifier(id_col) if id_col else "",
        "ra_field": quote_identifier(ra),
        "dec_field": quote_identifier(dec),
        "format": "json",
        "distance": "deg",
    }
    field_map = {canon: col for canon, col in description.field_map.items()}
    if field_map:
        parameters["field_map"] = field_map
    if column_units:
        parameters["column_units"] = column_units
    if epoch_range and not isinstance(epoch, (int, float)):
        parameters["epoch_range"] = [float(epoch_range[0]), float(epoch_range[1])]
    if isinstance(epoch, str) and epoch == description.epoch and description.single_epoch_positions:
        parameters["single_epoch_positions"] = True

    title = description.catalog.title or description.catalog.catalog_id
    entry: dict[str, Any] = {
        "enabled": bool(overrides.pop("enabled", True)),
        "provider": "tap",
        "wavelength": wavelength,
        "endpoint": VIZIER_TAP_URL,
        "table": quote_identifier(description.table_id),
        "description": str(overrides.pop("description", None)
                           or f"{title}: {_table_title(description.description) or description.table_id} "
                              f"(VizieR {description.table_id})"),
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": parameters,
        "profiles": [str(p) for p in profiles],
        "epoch": epoch,
        "epoch_format": epoch_format,
        "pos_error": pos_error,
        "citation": description.citation,
        "acknowledgement": VIZIER_ACKNOWLEDGEMENT,
        "max_rows": int(overrides.pop("max_rows", DEFAULT_MAX_ROWS)),
        "timeout_seconds": float(overrides.pop("timeout_seconds", DEFAULT_TIMEOUT_PER_CATALOG)),
        "coverage": str(overrides.pop("coverage", None)
                        or (f"{description.nrows:,} rows (VizieR {description.table_id})" if description.nrows is not None
                            else f"VizieR {description.table_id}")),
    }
    definition = catalog_from_dict(catalog_name, entry)
    problems = validate_catalog_definition(definition)
    if problems:
        raise VizierRegistrationError("generated definition is invalid: " + "; ".join(problems))
    return catalog_name, entry


# -- user registry ------------------------------------------------------------
#
# File format written by save_definition (readable by BOTH models.CatalogRegistry, which treats
# any YAML as a complete registry and ignores unknown keys, and UserCatalogRegistry):
#
#   extends: embedded
#   embedded_fingerprint: <hash of the built-in registry the copies were taken from>
#   embedded_copies: [gaia_dr3, simbad, ...]   # built-in entries copied into 'catalogs'
#   embedded_copy_hashes: {gaia_dr3: <hash of the copy as written>, ...}
#   catalogs: {<copies of the built-in entries>, <user entries>}
#
# The copies make the file a complete registry for the core (so pointing CATALOG_REGISTRY_PATH
# at it never hides the built-in catalogs); UserCatalogRegistry ignores them and merges the
# user entries over the *current* DEFAULT_CATALOGS. Every save, and every load_registry() of a
# file vizier wrote whose fingerprint no longer matches the running code's built-ins (an
# upgrade), rewrites the copies, so the core never keeps serving an outdated built-in.
# Operators may edit the file: a copy whose hash differs from 'embedded_copy_hashes' (or, in a
# file without hashes but with the current fingerprint, differs from its built-in) is a hand
# edit; it is treated as a user entry that shadows the built-in -- served by both registries,
# kept by saves, reported by registry_status / GET /registered / 'vizier list' -- never
# reverted. A load never rewrites a file vizier did not write (no header/fingerprint) or one
# with added comments (a warning is logged instead), and a save refuses to rewrite a file with
# comments other than vizier's header (a YAML dump would drop them). A user entry whose name
# later becomes a built-in stays a user entry (it is never listed as a copy, so a save cannot
# swallow it): both registries serve the user's entry and the collision is reported. A file
# without 'extends' is a hand-written complete registry: it keeps that meaning. Files must be
# UTF-8 (the encoding models.CatalogRegistry reads); others raise RegistryError.
#
# Concurrency: writers take an OS lock on '<file>.lock'; the file is replaced atomically. On
# Windows a replace fails while another process has the file open, so it is retried for
# REGISTRY_REPLACE_RETRY_SECONDS and readers retry a transient PermissionError. All blocking
# registry work runs in worker threads when called from the async API.

_REGISTRY_THREAD_LOCK = threading.RLock()
REGISTRY_LOCK_TIMEOUT_SECONDS = 30.0
#: Windows: os.replace onto a file another process holds open fails (sharing violation).
REGISTRY_REPLACE_RETRY_SECONDS = 3.0
#: Readers retry a PermissionError this long (Windows: the file is being replaced).
REGISTRY_READ_RETRY_SECONDS = 2.0
#: Lock budget of the best-effort refresh of outdated built-in copies done by load_registry().
REGISTRY_REFRESH_LOCK_SECONDS = 2.0
_POLL_SECONDS = 0.05

#: libyaml (C) loader/dumper when available: the registry file holds ~45 KB of built-in copies.
_YAML_LOADER: Any = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_YAML_DUMPER: Any = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


class _RegistryCancelled(VizierError):
    """A registry write abandoned because its caller gave up (route deadline)."""


def user_registry_path(path: str | os.PathLike[str] | None = None) -> Path:
    """User registry YAML: explicit path > CATALOG_REGISTRY_PATH > ~/.astrosearch/catalogs.yaml."""
    if path:
        return Path(path).expanduser()
    env = os.getenv("CATALOG_REGISTRY_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".astrosearch" / "catalogs.yaml"


def _lock_timeout_error(target: Path) -> VizierConflictError:
    return VizierConflictError(f"the catalog registry {target} is locked by another process; try again")


def _prepare_directory(target: Path) -> None:
    """Create the registry's directory; OS errors become VizierRegistryIOError (HTTP 500 with
    a JSON detail, CLI exit 1) instead of tracebacks."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VizierRegistryIOError(f"cannot create the directory of the catalog registry {target}: {exc}") from exc
    if target.is_dir():
        raise VizierRegistryIOError(f"the catalog registry path {target} is a directory")


@contextlib.contextmanager
def _registry_lock(target: Path, timeout: float | None = None,
                   cancel: threading.Event | None = None) -> Iterator[None]:
    """Exclusive inter-process lock (OS lock on ``<file>.lock``) around a registry
    read-modify-write, so concurrent 'vizier add' / POST /register calls never lose an entry.

    Waits at most ``timeout`` (REGISTRY_LOCK_TIMEOUT_SECONDS) -> VizierConflictError; gives up
    early when ``cancel`` is set (the async caller was cancelled) -> _RegistryCancelled."""
    deadline = time.monotonic() + (REGISTRY_LOCK_TIMEOUT_SECONDS if timeout is None else timeout)

    def wait() -> None:
        if cancel is not None and cancel.is_set():
            raise _RegistryCancelled(f"registry update of {target} abandoned by its caller")
        if time.monotonic() > deadline:
            raise _lock_timeout_error(target)
        time.sleep(_POLL_SECONDS)

    _prepare_directory(target)
    lock_path = target.with_name(target.name + ".lock")
    while not _REGISTRY_THREAD_LOCK.acquire(timeout=_POLL_SECONDS):
        wait()
    try:
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            raise VizierRegistryIOError(f"cannot open the registry lock file {lock_path}: {exc}") from exc
        try:
            if os.name == "nt":
                import msvcrt

                while True:
                    os.lseek(fd, 0, os.SEEK_SET)
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        wait()
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        wait()
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    finally:
        _REGISTRY_THREAD_LOCK.release()


def _read_text_retrying(target: Path) -> str | None:
    """File text (None when absent). A PermissionError -- on Windows the transient state of a
    file being replaced by a writer -- is retried for REGISTRY_READ_RETRY_SECONDS."""
    deadline = time.monotonic() + REGISTRY_READ_RETRY_SECONDS
    while True:
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            # e.g. PowerShell 5 'Out-File' (UTF-16 with a BOM) or an ANSI/Latin-1 editor save.
            raise RegistryError(f"Catalog registry {target} is not UTF-8 text ({exc}); save it as UTF-8 (the "
                                "encoding models.CatalogRegistry reads too)") from exc
        except PermissionError as exc:
            if time.monotonic() >= deadline or target.is_dir():
                raise RegistryError(f"Catalog registry {target} could not be read: {exc}") from exc
            time.sleep(_POLL_SECONDS / 2)
        except OSError as exc:
            if not target.exists():
                return None
            raise RegistryError(f"Catalog registry {target} could not be read: {exc}") from exc


def _read_document(target: Path) -> tuple[dict[str, Any], str]:
    """(parsed YAML document, raw text); ({}, '') when the file does not exist."""
    text = _read_text_retrying(target)
    if text is None:
        return {}, ""
    try:
        content = yaml.load(text, Loader=_YAML_LOADER) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"Catalog registry {target} could not be parsed: {exc}") from exc
    if not isinstance(content, dict) or not isinstance(content.get("catalogs", {}) or {}, dict):
        raise RegistryError(f"Catalog registry {target} must contain a 'catalogs' mapping.")
    content["catalogs"] = content.get("catalogs") or {}
    return content, text


def _extends_embedded(document: Mapping[str, Any]) -> bool:
    return not document or document.get("extends") == "embedded"


def _user_entries(document: Mapping[str, Any]) -> dict[str, Any]:
    """User entries of a registry document: everything except the built-in copies (files with
    ``extends: embedded``) or except built-in names (a hand-written complete registry).

    A built-in copy edited by hand (see :func:`_edited_copies`) is a user entry: the operator's
    change (``enabled: false``, a new timeout) is what models.CatalogRegistry serves from the
    file, so the merged registry serves it too and a save keeps it instead of reverting it."""
    catalogs = dict(document.get("catalogs") or {})
    if _extends_embedded(document):
        copies = {str(n) for n in document.get("embedded_copies") or []} - _edited_copies(document)
        return {name: entry for name, entry in catalogs.items() if name not in copies}
    return {name: entry for name, entry in catalogs.items() if name not in DEFAULT_CATALOGS}


def _entry_hash(entry: Any) -> str:
    """Short hash of a registry entry as stored (YAML round trip safe)."""
    payload = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _edited_copies(document: Mapping[str, Any]) -> set[str]:
    """Built-in copies of an ``extends: embedded`` file changed after vizier wrote them.

    Files written since per-copy hashes exist carry ``embedded_copy_hashes`` (the hash of each
    copy as written): a copy whose hash differs was edited. A file with only an
    ``embedded_fingerprint`` equal to the running built-ins' was written from exactly the
    current built-ins, so a copy that differs from its built-in was edited. Otherwise (an older
    fingerprint: an upgrade) a difference cannot be told from an outdated copy and the copy is
    treated as outdated."""
    if not document or not _extends_embedded(document):
        return set()
    catalogs = document.get("catalogs") or {}
    copies = [str(n) for n in document.get("embedded_copies") or [] if str(n) in catalogs]
    hashes = document.get("embedded_copy_hashes")
    if isinstance(hashes, Mapping) and hashes:
        return {name for name in copies if hashes.get(name) != _entry_hash(catalogs[name])}
    if document.get("embedded_fingerprint") == builtin_fingerprint():
        return {name for name in copies
                if name in DEFAULT_CATALOGS and _entry_hash(catalogs[name]) != _entry_hash(_plain(DEFAULT_CATALOGS[name]))}
    return set()


def _source_of(entry: Any) -> dict[str, Any]:
    """The ``source`` block of a registry entry ({} unless it is a mapping: hand-edited files
    may hold a free-text source)."""
    source = entry.get("source") if isinstance(entry, Mapping) else None
    return dict(source) if isinstance(source, Mapping) else {}


def builtin_fingerprint() -> str:
    """Short hash of the running code's built-in registry (models.DEFAULT_CATALOGS)."""
    payload = json.dumps(DEFAULT_CATALOGS, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _copies_outdated(document: Mapping[str, Any]) -> bool:
    """True when an ``extends: embedded`` file's built-in copies are not those of the running
    code (upgrade: changed/added/removed built-ins, or a file written before fingerprints)."""
    if not document or not _extends_embedded(document) or not document.get("catalogs"):
        return False
    user = set(_user_entries(document))
    copies = {str(n) for n in document.get("embedded_copies") or []} - _edited_copies(document)
    return (document.get("embedded_fingerprint") != builtin_fingerprint()
            or copies != set(DEFAULT_CATALOGS) - user)


#: Header lines vizier writes (the current header and the one of the first file format).
_LEGACY_HEADER_LINES = (
    "# AstroSearch user catalog registry. 'extends: embedded' merges these entries over the",
    "# built-in registry (vizier.UserCatalogRegistry); entries added by 'vizier add'.",
)


def _has_comment(line: str) -> bool:
    """True when a YAML line holds a comment ('#' at the start or after whitespace, outside
    quoted scalars); a YAML dump would drop it."""
    quote: str | None = None
    previous = " "
    escaped = False
    for ch in line:
        if quote == '"':
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quote = None
        elif quote == "'":
            if ch == "'":
                quote = None
        elif ch in "'\"" and previous in " \t:[{,-":
            quote = ch
        elif ch == "#" and previous in " \t":
            return True
        previous = ch
    return False


def _foreign_comments(text: str) -> list[str]:
    """Comment lines of a registry file other than vizier's own header lines."""
    own = set(_HEADER.splitlines()) | set(_LEGACY_HEADER_LINES)
    return [line for line in text.splitlines() if line.strip() and line.rstrip() not in own and _has_comment(line)]


def _written_by_vizier(document: Mapping[str, Any], text: str) -> bool:
    """True for a file vizier wrote (its header and an ``embedded_fingerprint``); only such a
    file is ever rewritten by a *load* (a hand-written file is left alone and a warning logged)."""
    first = text.lstrip("\ufeff").splitlines()[:1]
    return bool(document.get("embedded_fingerprint")) and bool(first) and first[0] in (
        _HEADER.splitlines()[0], _LEGACY_HEADER_LINES[0])


def read_user_registry(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """The user registry document with ``catalogs`` holding only the user entries (built-in
    copies removed); empty when the file does not exist."""
    document, _text = _read_document(user_registry_path(path))
    if not document:
        return {}
    out = {key: value for key, value in document.items()
           if key not in {"catalogs", "embedded_copies", "embedded_fingerprint", "embedded_copy_hashes"}}
    out["catalogs"] = _user_entries(document)
    return out


_HEADER = (
    "# AstroSearch catalog registry written by 'vizier add' / POST /api/v1/vizier/register.\n"
    "# 'catalogs' holds copies of the built-in entries (listed in 'embedded_copies', refreshed on\n"
    "# every save and whenever vizier.load_registry() finds them outdated) plus the user entries,\n"
    "# so models.CatalogRegistry reads a complete registry; vizier.UserCatalogRegistry merges the\n"
    "# user entries over the current built-in registry.\n"
    "# A built-in copy edited by hand (its hash differs from 'embedded_copy_hashes') is kept as a\n"
    "# user entry that shadows the built-in. Comments added here make vizier refuse to rewrite the file.\n"
)


def _write_document(document: Mapping[str, Any], path: Path, *, header: bool = True,
                    cancel: threading.Event | None = None) -> None:
    """Atomic write (temporary file + os.replace). On Windows the replace fails while another
    process has the file open; it is retried for REGISTRY_REPLACE_RETRY_SECONDS, then reported
    as VizierConflictError (try again). Other OS errors become VizierRegistryIOError."""
    _prepare_directory(path)
    text = yaml.dump(dict(document), Dumper=_YAML_DUMPER, sort_keys=False, allow_unicode=True, width=120)
    if header:
        text = _HEADER + text
    try:
        fd, tmp = tempfile.mkstemp(prefix=".catalogs.", suffix=".yaml", dir=str(path.parent))
    except OSError as exc:
        raise VizierRegistryIOError(f"cannot write the catalog registry {path}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        deadline = time.monotonic() + REGISTRY_REPLACE_RETRY_SECONDS
        while True:
            # Checked before every attempt: a caller that gave up (route deadline) while another
            # process held the file open must not have the file replaced behind its back.
            if cancel is not None and cancel.is_set():
                raise _RegistryCancelled(f"registry update of {path} abandoned by its caller")
            try:
                os.replace(tmp, path)
                break
            except PermissionError as exc:
                if time.monotonic() >= deadline:
                    raise VizierConflictError(
                        f"the catalog registry {path} could not be replaced ({exc.strerror or exc}): it is held open "
                        "by another process or not writable; try again") from exc
                time.sleep(_POLL_SECONDS)
    except OSError as exc:
        raise VizierRegistryIOError(f"cannot write the catalog registry {path}: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))  # YAML-safe builtin types


def _save_user_entries(target: Path, document: Mapping[str, Any], text: str, user: Mapping[str, Any],
                       cancel: threading.Event | None = None) -> None:
    """Write ``user`` entries back in the file's own format (called under _registry_lock).

    A file with comments other than vizier's header is never rewritten (a YAML dump would drop
    them): VizierRegistrationError. Built-in copies edited by hand are part of ``user`` (see
    :func:`_user_entries`), so they are kept, not reverted."""
    comments = _foreign_comments(text)
    if comments:
        kind = ("an 'extends: embedded' registry" if _extends_embedded(document)
                else "a hand-written complete catalog registry (no 'extends' key)")
        raise VizierRegistrationError(
            f"{target} is {kind} with comments that rewriting would drop (e.g. {comments[0].strip()[:60]!r}); "
            "register into a separate file (--registry-path or app.state.vizier_registry_path) or remove the "
            "comments first.")
    if _extends_embedded(document):
        extra = {k: v for k, v in document.items()
                 if k not in {"extends", "embedded_fingerprint", "embedded_copies", "embedded_copy_hashes", "catalogs"}}
        builtin = _plain(DEFAULT_CATALOGS)
        edited = sorted(_edited_copies(document) & set(user))
        if edited:
            logger.warning("built-in catalog copies %s in %s were edited by hand; the edited entries are kept as user "
                           "entries that shadow the built-ins (delete them to follow the built-ins again)", edited, target)
        shadowed = sorted(set(user) & set(builtin))
        if shadowed:
            logger.warning("user catalog(s) %s in %s have built-in names; the user entries are kept and shadow the "
                           "built-ins (rename them to use both)", shadowed, target)
        copies = {name: entry for name, entry in builtin.items() if name not in user}
        _write_document({"extends": "embedded", **extra, "embedded_fingerprint": builtin_fingerprint(),
                         "embedded_copies": list(copies),
                         "embedded_copy_hashes": {name: _entry_hash(entry) for name, entry in copies.items()},
                         "catalogs": {**copies, **dict(user)}}, target, cancel=cancel)
        return
    logger.warning("rewriting the complete catalog registry %s (no 'extends' key; it stays a complete registry)", target)
    catalogs = {name: entry for name, entry in (document.get("catalogs") or {}).items() if name in DEFAULT_CATALOGS}
    catalogs.update(user)
    _write_document({**{k: v for k, v in document.items() if k != "catalogs"}, "catalogs": catalogs}, target,
                    header=False, cancel=cancel)


def refresh_embedded_copies(path: str | os.PathLike[str] | None = None, *,
                            timeout: float | None = None) -> bool:
    """Rewrite outdated built-in copies of an ``extends: embedded`` registry file (after an
    upgrade changed models.DEFAULT_CATALOGS), keeping every user entry and every hand-edited
    copy. Returns True when the file was rewritten. load_registry() calls it (best effort) on
    every load. Only a file vizier wrote (header + fingerprint) without added comments is
    rewritten; for any other file a warning is logged and False returned (a *read* never
    rewrites a hand-written file)."""
    target = user_registry_path(path)
    document, text = _read_document(target)
    if not _copies_outdated(document):
        return False
    if not _written_by_vizier(document, text) or _foreign_comments(text):
        logger.warning("the built-in catalog copies in %s are missing or outdated, but the file was %s; it is not "
                       "rewritten (models.CatalogRegistry reading it misses or serves old built-ins; run 'vizier add' "
                       "or save it with vizier to refresh them)", target,
                       "edited with comments" if _written_by_vizier(document, text) else "not written by vizier")
        return False
    with _registry_lock(target, timeout=REGISTRY_REFRESH_LOCK_SECONDS if timeout is None else timeout):
        document, text = _read_document(target)
        if not _copies_outdated(document) or not _written_by_vizier(document, text) or _foreign_comments(text):
            return False
        _save_user_entries(target, document, text, _user_entries(document))
    logger.info("refreshed the built-in catalog copies in %s", target)
    return True


def registered_entries(path: str | os.PathLike[str] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """(user catalog entries that are mappings, names of malformed non-mapping entries)."""
    entries = read_user_registry(path).get("catalogs") or {}
    valid = {str(name): entry for name, entry in entries.items() if isinstance(entry, dict)}
    invalid = sorted(str(name) for name, entry in entries.items() if not isinstance(entry, dict))
    return valid, invalid


def registry_status(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """One read of the registry file: user entries, malformed names, user entries that shadow
    a built-in name, and whether the built-in copies are missing or outdated (the core registry
    would then miss or serve old built-ins until the next save or load_registry())."""
    target = user_registry_path(path)
    document, text = _read_document(target)
    entries = _user_entries(document) if document else {}
    valid = {str(name): entry for name, entry in entries.items() if isinstance(entry, dict)}
    return {
        "path": str(target),
        "entries": valid,
        "invalid": sorted(str(name) for name, entry in entries.items() if not isinstance(entry, dict)),
        "shadowed_builtins": sorted(name for name in valid if name in DEFAULT_CATALOGS),
        "builtin_copies_outdated": _copies_outdated(document),
        "edited_builtin_copies": sorted(_edited_copies(document)),
        "written_by_vizier": _written_by_vizier(document, text) if document else None,
        "comments": bool(_foreign_comments(text)) if document else False,
    }


def list_registered(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """User catalog entries stored in the registry YAML (malformed non-mapping entries skipped)."""
    return registered_entries(path)[0]


def _check_registrable_name(name: str, path: str | os.PathLike[str] | None, replace: bool) -> None:
    """Refuse built-in names, and names already registered unless ``replace``."""
    if name in DEFAULT_CATALOGS:
        raise VizierRegistrationError(f"'{name}' is a built-in catalog name; choose another name.")
    if replace:
        return
    existing = read_user_registry(path).get("catalogs") or {}
    if name in existing:
        previous = _source_of(existing[name]).get("table")
        raise VizierConflictError(
            f"'{name}' is already registered{f' (VizieR {previous})' if previous else ''}; pass replace=True to "
            "overwrite.")


def _preflight_registry(path: str | os.PathLike[str] | None, name: str | None, replace: bool) -> None:
    """Checks run before any network request: the registry location is usable and (for an
    explicit name) the name is free, so a doomed registration does not waste a describe."""
    target = user_registry_path(path)
    _prepare_directory(target)
    probe = target if target.exists() else target.parent
    if not os.access(probe, os.W_OK):
        raise VizierRegistryIOError(f"the catalog registry {target} is not writable")
    if name is not None:
        _check_registrable_name(name, target, replace)


def save_definition(name: str, entry: Mapping[str, Any], *, path: str | os.PathLike[str] | None = None,
                    replace: bool = False, source: Mapping[str, Any] | None = None,
                    cancel: threading.Event | None = None) -> tuple[Path, bool]:
    """Persist one entry in the user registry (locked read-modify-write); returns (path, replaced).

    Blocking (file I/O, lock wait): async callers run it in a worker thread and set ``cancel``
    when they give up, so an abandoned call never writes."""
    target = user_registry_path(path)
    if name in DEFAULT_CATALOGS:
        raise VizierRegistrationError(f"'{name}' is a built-in catalog name; choose another name.")
    with _registry_lock(target, cancel=cancel):
        document, text = _read_document(target)
        user = _user_entries(document)
        existed = name in user
        if existed and not replace:
            _check_registrable_name(name, target, replace)
        stored = _plain(dict(entry))
        if source:
            stored["source"] = _plain(dict(source))
        user[name] = stored
        _save_user_entries(target, document, text, user, cancel=cancel)
    return target, existed


def unregister(name: str, *, path: str | os.PathLike[str] | None = None) -> bool:
    """Remove a user catalog; returns False when it was not registered."""
    target = user_registry_path(path)
    with _registry_lock(target):
        document, text = _read_document(target)
        user = _user_entries(document)
        if name not in user:
            return False
        del user[name]
        _save_user_entries(target, document, text, user)
    return True


class UserCatalogRegistry(CatalogRegistry):
    """The embedded registry with the user registry YAML merged over it.

    File semantics: absent -> embedded catalogs only; ``extends: embedded`` (written by
    :func:`save_definition`) -> the current embedded catalogs + the user entries (the file's
    ``embedded_copies`` are ignored, so an updated built-in is never shadowed by a stale copy,
    and outdated copies are rewritten for models.CatalogRegistry readers, best effort, when
    vizier wrote the file and nobody added comments to it); copies edited by hand are served
    as user entries, exactly as models.CatalogRegistry serves them; a file without ``extends``
    is a complete registry, exactly as :class:`models.CatalogRegistry` reads it.
    """

    def __init__(self, registry_path: str | os.PathLike[str] | None = None) -> None:
        super().__init__(user_registry_path(registry_path))

    def reload(self) -> None:
        document, _text = _read_document(self.registry_path) if self.registry_path else ({}, "")
        if self.registry_path is not None and _copies_outdated(document):
            # refresh_embedded_copies leaves hand-written/commented files alone (with a warning).
            try:
                refresh_embedded_copies(self.registry_path)
            except (VizierError, RegistryError) as exc:
                logger.warning("could not refresh the outdated built-in copies in %s (%s); models.CatalogRegistry "
                               "readers of that file see old built-ins until the next save", self.registry_path, exc)
        merged: dict[str, Any] = {}
        if _extends_embedded(document):
            merged.update(DEFAULT_CATALOGS)
            merged.update(_user_entries(document))
        else:
            merged.update(document.get("catalogs") or {})
        self._catalogs = {name: catalog_from_dict(name, entry) for name, entry in merged.items()
                          if isinstance(entry, dict)}

    def add(self, definition: CatalogDefinition) -> None:
        """Make a definition available immediately (without touching the file)."""
        self._catalogs[definition.name] = definition


def load_registry(path: str | os.PathLike[str] | None = None) -> UserCatalogRegistry:
    """Embedded + user catalogs (see :class:`UserCatalogRegistry`); the registry main.py/api.py
    should build instead of ``CatalogRegistry(settings.catalog_registry_path)``."""
    return UserCatalogRegistry(path)


def attach_definition(registry: CatalogRegistry, name: str, entry: Mapping[str, Any]) -> CatalogDefinition:
    """Add a definition to a live registry (e.g. the API service's) so it is queried at once.

    ``models.CatalogRegistry`` has no public add method yet; for it the adapter writes the
    registry's catalog map directly (see integration notes).
    """
    definition = catalog_from_dict(name, entry)
    adder = getattr(registry, "add", None)
    if callable(adder):
        adder(definition)
    else:
        registry._catalogs[name] = definition  # adapter until the core exposes add()
    return definition


async def register_table(
    table_id: str,
    *,
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    path: str | os.PathLike[str] | None = None,
    replace: bool = False,
    registry: CatalogRegistry | Sequence[CatalogRegistry] | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float | None = None,
) -> Registration:
    """Describe a VizieR table, build its definition, save it to the user registry and
    (optionally) attach it to one or more live registries (``registry``).

    Overrides, the name and the registry location are checked before any network request. An
    overridden per-row epoch without ``epoch_range`` gets its span determined like a detected
    one (MIN/MAX scan or sky sampling), so moving targets are still followed; the returned
    assumptions describe the overridden epoch/positional error, not the replaced detection.
    The registry file work (reads, YAML, the inter-process lock wait) runs in worker threads,
    so the event loop -- and the API route deadline -- are never blocked. When the caller is
    cancelled the pending save is abandoned (the lock wait and every replace attempt check for
    it); if the file had already been replaced, the registration is completed (attached and
    returned) instead of being reported as not done."""
    overrides = validate_overrides(overrides)
    ident = validate_vizier_id(table_id)
    early_name = validate_catalog_name(name, ident) if name is not None else None
    await asyncio.to_thread(_preflight_registry, path, early_name, replace)
    description = await describe_table(ident, client=client, timeout=timeout)
    build_notes: list[str] = []
    catalog_name, entry = build_definition(description, name=name, overrides=overrides, notes=build_notes)
    assumptions = list(description.assumptions)
    epoch_source = description.epoch_source
    if "epoch" in overrides and overrides["epoch"] != description.epoch:
        assumptions = [a for a in assumptions if not a.startswith("epoch:")]
        was = f"{description.epoch!r}" + (f" from {description.epoch_source}" if description.epoch_source else "")
        assumptions.append(f"epoch: {overrides['epoch']!r} given by the caller (the metadata gave {was}).")
        epoch_source = "override"
        if _per_row_epoch(entry["epoch"]) and not entry["parameters"].get("epoch_range"):
            span_notes, span = await _override_epoch_range(description, entry, client=client, timeout=timeout)
            assumptions.extend(span_notes)
            if span is not None:
                build_notes = []
                catalog_name, entry = build_definition(description, name=name,
                                                       overrides={**overrides, "epoch_range": span}, notes=build_notes)
    if "pos_error" in overrides:
        assumptions = [a for a in assumptions if not a.startswith("pos_error:")]
        assumptions.append(f"pos_error: {entry['pos_error'] or 'none'} given by the caller (the metadata gave "
                           f"{description.pos_error or 'none'}).")
    assumptions.extend(n for n in build_notes if n not in assumptions)
    references, reference_notes = await citation_references(entry.get("citation"), client=client)
    if references:
        from provenance import REGISTRY_REFERENCES_KEY

        entry["parameters"][REGISTRY_REFERENCES_KEY] = references
    assumptions.extend(reference_notes)
    source = {
        "service": "vizier",
        "table": description.table_id,
        "catalog": description.catalog.catalog_id,
        "title": description.catalog.title,
        "bibcode": description.catalog.bibcode,
        "doi": description.catalog.doi,
        "nrows": description.nrows,
        "epoch_source": epoch_source,
        "assumptions": list(assumptions),
        "overrides": dict(overrides or {}),
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    cancel = threading.Event()
    save = asyncio.ensure_future(asyncio.to_thread(save_definition, catalog_name, entry, path=path, replace=replace,
                                                   source=source, cancel=cancel))
    try:
        saved, replaced = await asyncio.shield(save)
    except asyncio.CancelledError:
        cancel.set()
        # The worker now either abandons the write or has already replaced the file; wait for
        # it (at most one replace attempt) to know which.
        await asyncio.wait({save})
        if save.cancelled() or save.exception() is not None:
            raise
        saved, replaced = save.result()
        logger.warning("registration of %s was cancelled after %s had been written; completing it", catalog_name,
                       saved)
    live = [registry] if isinstance(registry, CatalogRegistry) else list(registry or [])
    for target in live:
        attach_definition(target, catalog_name, entry)
    return Registration(name=catalog_name, table_id=description.table_id, entry=entry, path=str(saved),
                        replaced=replaced, assumptions=assumptions, attached=bool(live))


async def citation_references(citation: str | None, *, client: httpx.AsyncClient | None = None,
                              timeout: float | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """(references, notes): the papers a registry citation names (its bibcodes), resolved through
    the ADS link gateway and doi.org (provenance.lookup_reference) and stored in the entry, so
    `astrosearch cite` and /api/v1/citations write verified BibTeX for the table's own paper.
    Optional: a paper that cannot be resolved (network, no DOI, time budget) is only noted, and
    is then cited from the citation text, marked unverified."""
    import provenance

    bibcodes = provenance.bibcodes_in(citation)
    if not bibcodes:
        return [], []
    budget = REFERENCE_LOOKUP_TIMEOUT_SECONDS if timeout is None else float(timeout)
    found: list[dict[str, Any]] = []
    notes: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        active = client or await stack.enter_async_context(
            httpx.AsyncClient(timeout=budget, headers={"User-Agent": USER_AGENT}))
        for bibcode in bibcodes:
            try:
                ref = await asyncio.wait_for(provenance.lookup_reference(active, bibcode), budget)
            except (httpx.HTTPError, TimeoutError, provenance.UnparsableAnswerError) as exc:
                notes.append(f"citation: {bibcode} not resolved through ADS/doi.org ({exc.__class__.__name__}); "
                             "cited from the citation text, unverified.")
                continue
            if ref is None:
                notes.append(f"citation: ADS gives no DOI for {bibcode}; cited from the citation text, unverified.")
                continue
            found.append(provenance.reference_to_registry(ref))
    return found, notes


async def _override_epoch_range(description: TableDescription, entry: Mapping[str, Any], *,
                                client: httpx.AsyncClient | None,
                                timeout: float | None) -> tuple[list[str], list[float] | None]:
    """Span of an overridden per-row epoch (the entry's epoch/epoch_format), determined like a
    detected one (:func:`_determine_epoch_range`). Returns (notes, [earliest, latest] or None)."""
    probe = copy.copy(description)
    probe.epoch = entry["epoch"]
    probe.epoch_format = entry.get("epoch_format")
    probe.epoch_range = None
    async with _Http(client, timeout=timeout) as http:
        notes, _complete = await _determine_epoch_range(http, probe)
    if probe.epoch_range is None:
        return notes, None
    return notes, [float(probe.epoch_range[0]), float(probe.epoch_range[1])]


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/vizier", tags=["vizier"])


class RegisterRequest(BaseModel):
    table_id: str = Field(..., min_length=3, max_length=128, description="VizieR table, e.g. 'IX/58/2sxps'")
    name: str | None = Field(default=None, max_length=64, description="Registry name (default vizier_<table>)")
    overrides: dict[str, Any] | None = Field(default=None, description="Definition overrides (pos_error, epoch, ...)")
    replace: bool = Field(default=False, description="Overwrite an existing user catalog of that name")


def _state(request: Request, name: str) -> Any:
    return getattr(request.app.state, name, None)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, VizierNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, VizierConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (VizierInputError, VizierRegistrationError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RegistryError):
        return HTTPException(status_code=500, detail=f"User registry unreadable: {exc}")
    if isinstance(exc, VizierRegistryIOError):
        return HTTPException(status_code=500, detail=f"User registry not writable: {exc}")
    headers = {"Retry-After": "30"} if isinstance(exc, VizierUpstreamError) and exc.transient else None
    return HTTPException(status_code=502, detail=f"Upstream failure: {type(exc).__name__}: {exc}", headers=headers)


async def _within_deadline[T](awaitable: Awaitable[T], what: str) -> T:
    """Run one route's work under ROUTE_DEADLINE_SECONDS (504 when exceeded); VizieR/registry
    errors become 404/409/422/500/502."""
    try:
        async with asyncio.timeout(ROUTE_DEADLINE_SECONDS):
            return await awaitable
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"VizieR {what} did not finish within "
                                                    f"{ROUTE_DEADLINE_SECONDS:g} s") from exc
    except (VizierError, RegistryError) as exc:
        raise _http_error(exc) from exc


@router.get("/search", response_model=dict[str, Any])
async def search_endpoint(
    request: Request,
    q: str = Query(default="", max_length=200, description="Keywords (AND-ed by VizieR)"),
    ucd: str | None = Query(default=None, max_length=120, description="Column UCD, e.g. src.redshift"),
    wavelength: str | None = Query(default=None, description="radio, millimeter, infrared, optical, uv, xray, gamma"),
    max_catalogs: int = Query(default=50, ge=1, le=500),
    max_tables: int = Query(default=100, ge=1, le=1000),
    registry: bool = Query(default=False, description="Also search the IVOA registry (RegTAP)"),
) -> dict[str, Any]:
    """Search VizieR catalogues/tables by keyword, wavelength and UCD."""
    result = await _within_deadline(search_catalogs(q, ucd=ucd, wavelength=wavelength, max_catalogs=max_catalogs,
                                                    max_tables=max_tables, include_registry=registry,
                                                    client=_state(request, "client")), "search")
    return result.as_dict()


@router.get("/catalog/{identifier:path}", response_model=dict[str, Any])
async def describe_endpoint(request: Request, identifier: str) -> dict[str, Any]:
    """Describe a VizieR table (columns, UCDs, units, rows, reference, position/error/epoch columns)
    or list a catalogue's tables."""
    result = await _within_deadline(describe(identifier, client=_state(request, "client")), "describe")
    data = result.as_dict()
    data["kind"] = "table" if isinstance(result, TableDescription) else "catalog"
    return data


def _live_registries(request: Request) -> list[CatalogRegistry]:
    """Every live registry of the app a new catalog must be attached to: app.state.registry,
    the crossmatch service's and the dataset engine's (api.py's DatasetEngine builds its own
    CatalogRegistry, which POST /api/v1/datasets/create validates against), each once."""
    found: list[CatalogRegistry] = []
    for candidate in (_state(request, "registry"), getattr(_state(request, "service"), "registry", None),
                      getattr(_state(request, "engine"), "registry", None)):
        if isinstance(candidate, CatalogRegistry) and all(candidate is not r for r in found):
            found.append(candidate)
    return found


@router.post("/register", response_model=dict[str, Any])
async def register_endpoint(request: Request, body: RegisterRequest) -> dict[str, Any]:
    """Register a VizieR table as a crossmatch catalog (saved to the user registry YAML and
    attached to the running service's registry)."""
    path = _state(request, "vizier_registry_path")
    registration = await _within_deadline(
        register_table(body.table_id, name=body.name, overrides=body.overrides, path=path, replace=body.replace,
                       registry=_live_registries(request), client=_state(request, "client")), "registration")
    return registration.as_dict()


@router.get("/registered", response_model=dict[str, Any])
async def registered_endpoint(request: Request) -> dict[str, Any]:
    """User catalogs from the registry YAML and whether the running service knows them
    (malformed non-mapping entries are listed under ``invalid``)."""
    path = user_registry_path(_state(request, "vizier_registry_path"))
    try:
        status = await asyncio.to_thread(registry_status, path)  # file I/O + YAML off the event loop
    except (RegistryError, VizierError) as exc:
        raise _http_error(exc) from exc
    entries = status["entries"]
    live = _live_registries(request)
    return {
        "path": str(path),
        "count": len(entries),
        "catalogs": [
            {"name": name, "table": _source_of(entry).get("table"), "wavelength": entry.get("wavelength"),
             "citation": entry.get("citation"), "enabled": entry.get("enabled", True),
             "active": any(name in r.catalogs for r in live) if live else None,
             "shadows_builtin": name in DEFAULT_CATALOGS, "entry": entry}
            for name, entry in entries.items()
        ],
        "invalid": status["invalid"],
        "shadowed_builtins": status["shadowed_builtins"],
        "builtin_copies_outdated": status["builtin_copies_outdated"],
        "edited_builtin_copies": status["edited_builtin_copies"],
        "written_by_vizier": status["written_by_vizier"],
    }


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def format_search(result: SearchResult, limit: int = 25) -> str:
    lines = [f"VizieR search '{result.query}'" + (f" wavelength={result.wavelength}" if result.wavelength else "")
             + (f" ucd={result.ucd}" if result.ucd else "")
             + f": {len(result.tables)} table(s) from {len(result.catalogs)} catalogue(s)"
             + (f" of {result.total_catalog_matches} matching (truncated)" if result.truncated else "")]
    lines.append(f"{'table':<34}{'rows':>14}  {'wavelength':<14}description")
    for hit in result.tables[:limit]:
        rows = f"{hit.nrows:,}" if hit.nrows is not None else "-"
        desc = (hit.description or hit.catalog_title or "")[:70]
        lines.append(f"{hit.table_id:<34}{rows:>14}  {','.join(hit.wavelengths)[:14]:<14}{desc}")
        for col in hit.matching_columns[:3]:
            lines.append(f"{'':<50}  {col['column']} [{col['ucd']}] {col['unit']}")
    for res in result.registry or []:
        lines.append(f"registry: {res.ivoid}  {res.title}  {', '.join(res.services)}")
    for warning in result.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def format_description(desc: TableDescription | CatalogInfo) -> str:
    if isinstance(desc, CatalogInfo):
        lines = [f"{desc.catalog_id}: {desc.title}", f"  reference: {desc.bibcode}  DOI {desc.doi}",
                 f"  tables ({len(desc.tables)}):"]
        lines += [f"    {t.table_id:<34}{(t.nrows or 0):>14,}  {(t.description or '')[:70]}" for t in desc.tables]
        return "\n".join(lines)
    lines = [
        f"{desc.table_id}: {desc.description}",
        f"  catalogue: {desc.catalog.title}",
        f"  citation:  {desc.citation}",
        f"  rows: {desc.nrows:,}" if desc.nrows is not None else "  rows: unknown",
        f"  wavelength: {desc.wavelength}   frame: {desc.frame}",
        f"  position: {desc.ra_column}, {desc.dec_column}   id: {desc.id_column}",
        f"  positional error: {json.dumps(desc.pos_error) if desc.pos_error else 'none'}",
        f"  epoch: {desc.epoch!r} ({desc.epoch_source or 'unknown'})"
        + (f" range {desc.epoch_range}" if desc.epoch_range else ""),
        f"  canonical fields: {desc.field_map or '-'}",
    ]
    for note in desc.assumptions:
        lines.append(f"  assumption: {note}")
    for problem in desc.problems:
        lines.append(f"  PROBLEM: {problem}")
    lines.append(f"  {'column':<18}{'unit':<12}{'ucd':<34}description")
    for col in desc.columns:
        flag = "*" if col.principal else " "
        lines.append(f" {flag}{col.name:<18}{(col.unit or ''):<12}{(col.ucd or ''):<34}{(col.description or '')[:60]}")
    return "\n".join(lines)


def _cli_error(exc: Exception) -> int:
    """Print a CLI error; exit code 2 for bad input (usage), 1 for everything else."""
    print(f"Error: {exc}")
    return 2 if isinstance(exc, VizierInputError) else 1


def _cli_search(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(search_catalogs(args.keywords, ucd=args.ucd, wavelength=args.wavelength,
                                             max_catalogs=args.max, include_registry=args.registry))
    except (VizierError, httpx.HTTPError) as exc:
        return _cli_error(exc)
    print(json.dumps(result.as_dict(), indent=2, default=str) if args.json else format_search(result, args.limit))
    return 0


def _cli_describe(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(describe(args.table))
    except (VizierError, httpx.HTTPError) as exc:
        return _cli_error(exc)
    print(json.dumps(result.as_dict(), indent=2, default=str) if args.json else format_description(result))
    return 0


def _cli_add(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.overrides:
        try:
            parsed = json.loads(args.overrides)
        except json.JSONDecodeError as exc:
            print(f"Error: --overrides is not valid JSON: {exc}")
            return 2
        if not isinstance(parsed, dict):
            print(f"Error: --overrides must be a JSON object such as '{{\"epoch\": 2000.0}}', got {parsed!r}")
            return 2
        overrides = parsed
    if args.systematic is not None:
        overrides["systematic_arcsec"] = args.systematic
    try:
        registration = asyncio.run(register_table(args.table, name=args.name, overrides=overrides or None,
                                                  path=args.registry_path, replace=args.replace))
    except (VizierError, RegistryError, httpx.HTTPError, OSError) as exc:
        return _cli_error(exc)
    if args.json:
        print(json.dumps(registration.as_dict(), indent=2, default=str))
    else:
        verb = "Replaced" if registration.replaced else "Registered"
        print(f"{verb} '{registration.name}' (VizieR {registration.table_id}) in {registration.path}")
        print(f"  wavelength={registration.entry['wavelength']} epoch={registration.entry['epoch']!r} "
              f"pos_error={registration.entry['pos_error'] or 'none'}")
        for note in registration.assumptions:
            print(f"  assumption: {note}")
    return 0


def _cli_list(args: argparse.Namespace) -> int:
    try:
        status = registry_status(args.registry_path)
    except (RegistryError, VizierError, OSError) as exc:
        return _cli_error(exc)
    entries, invalid = status["entries"], status["invalid"]
    if args.json:
        print(json.dumps({"catalogs": entries, "invalid": invalid, "shadowed_builtins": status["shadowed_builtins"],
                          "builtin_copies_outdated": status["builtin_copies_outdated"],
                          "edited_builtin_copies": status["edited_builtin_copies"],
                          "written_by_vizier": status["written_by_vizier"]}, indent=2, default=str))
        return 0
    print(f"{len(entries)} user catalog(s) in {status['path']}")
    for name, entry in entries.items():
        table = str(_source_of(entry).get("table") or "")
        print(f"  {name:<32}{table:<30}{entry.get('wavelength')}")
    for name in invalid:
        print(f"warning: entry '{name}' is not a mapping and is ignored")
    edited = set(status["edited_builtin_copies"])
    for name in status["shadowed_builtins"]:
        if name in edited:
            print(f"warning: the built-in copy '{name}' was edited by hand; the edit is kept and shadows the built-in "
                  "(delete the entry to follow the built-in again)")
        else:
            print(f"warning: user catalog '{name}' has the name of a built-in catalog and shadows it; rename it to use "
                  "both")
    if status["builtin_copies_outdated"]:
        refresher = ("the next 'vizier add' or vizier.load_registry()" if status["written_by_vizier"] and not status["comments"]
                     else "the next 'vizier add' (a hand-written or commented file is never rewritten by a load)")
        print("warning: the built-in catalog copies in this file are missing or outdated (models.CatalogRegistry "
              f"reading it would miss or serve old built-ins); they are refreshed by {refresher}")
    return 0


def _add_subcommands(sub: Any) -> None:
    search = sub.add_parser("search", help="Search VizieR catalogues by keywords/wavelength/UCD")
    search.add_argument("keywords", nargs="?", default="", help="Keywords, e.g. 'Gaia DR3'")
    search.add_argument("--ucd", help="Only tables with a column of this UCD (e.g. src.redshift)")
    search.add_argument("--wavelength", help="radio, millimeter, infrared, optical, uv, euv, xray, gamma")
    search.add_argument("--max", type=int, default=50, help="Maximum catalogues requested from VizieR (default 50)")
    search.add_argument("--limit", type=int, default=25, help="Tables printed (default 25)")
    search.add_argument("--registry", action="store_true", help="Also search the IVOA registry (RegTAP)")
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=_cli_search)

    desc = sub.add_parser("describe", help="Columns, UCDs, units, rows, reference of a VizieR table")
    desc.add_argument("table", help="VizieR table or catalogue id, e.g. IX/58/2sxps")
    desc.add_argument("--json", action="store_true")
    desc.set_defaults(handler=_cli_describe)

    add = sub.add_parser("add", help="Register a VizieR table as a crossmatch catalog")
    add.add_argument("table", help="VizieR table id, e.g. J/ApJS/255/30/comp")
    add.add_argument("--name", help="Registry name (default vizier_<table>)")
    add.add_argument("--registry-path", help="User registry YAML (default CATALOG_REGISTRY_PATH or ~/.astrosearch/catalogs.yaml)")
    add.add_argument("--systematic", type=float, help="Astrometric systematic (arcsec) added in quadrature")
    add.add_argument("--overrides", help="JSON object of definition overrides (pos_error, epoch, wavelength, ...)")
    add.add_argument("--replace", action="store_true", help="Overwrite an existing user catalog of that name")
    add.add_argument("--json", action="store_true")
    add.set_defaults(handler=_cli_add)

    listing = sub.add_parser("list", help="User catalogs in the registry YAML")
    listing.add_argument("--registry-path")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_cli_list)


def register_cli(subparsers: Any) -> None:
    """Add ``vizier search|describe|add|list`` to an argparse subparsers object."""
    parser = subparsers.add_parser("vizier", help="Discover, describe and register any VizieR table")
    sub = parser.add_subparsers(dest="vizier_command")
    parser.set_defaults(handler=lambda args: (parser.print_help(), 2)[1])
    _add_subcommands(sub)


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point: ``python vizier.py search "Gaia DR3"`` (the subcommands sit on the
    top-level parser; the prefixed form ``python vizier.py vizier search ...`` is accepted too)."""
    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    if arguments[:1] == ["vizier"]:
        arguments = arguments[1:]
    parser = argparse.ArgumentParser(prog="vizier", description="Discover, describe and register any VizieR table")
    _add_subcommands(parser.add_subparsers(dest="vizier_command"))
    args = parser.parse_args(arguments)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
