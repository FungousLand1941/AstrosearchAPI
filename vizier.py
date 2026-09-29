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
  metadata (bibcode, DOI, authors, wavelength keywords) and the ASU table metadata
  (VOTable ``COOSYS`` frame and ``epoch``) and identifies the position, identifier,
  positional-error and epoch columns.
* **Register** -- :func:`build_definition` / :func:`register_table` build a registry entry for
  the existing :class:`providers.TapProvider` (``provider: tap`` on the VizieR TAP endpoint,
  quoted table name, UCD-selected columns) and persist it in a user registry YAML
  (``CATALOG_REGISTRY_PATH`` or ``~/.astrosearch/catalogs.yaml``) that is merged over the
  embedded registry by :class:`UserCatalogRegistry`.

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
#: Per-attempt back-off before retrying a 5xx/network failure (attempts: HTTP_ATTEMPTS).
RETRY_BACKOFF_SECONDS = 0.5
HTTP_ATTEMPTS = 3
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
                      data: dict[str, Any] | None = None, service: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(HTTP_ATTEMPTS):
            try:
                response = await self.client.request(method, url, params=params, data=data, timeout=self.timeout,
                                                     headers={"User-Agent": USER_AGENT})
            except httpx.TimeoutException as exc:
                raise VizierUpstreamError(f"{service} timed out after {self.timeout:g}s", transient=True) from exc
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code < 500:
                    return response
                last = VizierUpstreamError(f"{service} HTTP {response.status_code}")
            if attempt < HTTP_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise VizierUpstreamError(f"{service} unavailable: {last}", transient=True) from last

    async def tap(self, adql: str, *, url: str | None = None, service: str = "VizieR TAP", fmt: str = "json"):
        """Run a synchronous ADQL query; returns :class:`models.ParsedTable`."""
        form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql}
        if fmt == "json":
            form["FORMAT"] = "json"
        response = await self.request("POST", url or VIZIER_TAP_URL, data=form, service=service)
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
            warnings.append(f"VizieR: {value}")
    return warnings, total, truncated


def parse_table_meta(root: ElementTree.Element) -> dict[str, Any]:
    """ASU ``-meta.all`` table metadata: COOSYS systems and the FIELD -> COOSYS references."""
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
    return {"coosys": coosys, "fields": fields, "description": table_description}


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
    ``max_catalogs`` catalogues come from each source. With ``wavelength`` the VizieR keyword
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
    if not 1 <= int(max_catalogs) <= 500:
        raise VizierInputError("max_catalogs must be in 1..500")
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
                return await _search_registry(http, text, keyword=keyword)
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
            extra_ids = [cid for cid in described if cid not in known][:int(max_catalogs)]
            if extra_ids:
                catalogs.extend(await _catalog_infos(http, extra_ids))
            if keyword:
                kept = [c for c in catalogs if keyword in c.wavelengths]
                if len(kept) < len(catalogs):
                    warnings.append(f"{len(catalogs) - len(kept)} catalogue(s) matched the words but are not tagged "
                                    f"'{keyword}' in VizieR and were removed.")
                catalogs = kept
            grouped = await _tables_of_catalogs(http, [c.catalog_id for c in catalogs])
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
                           servicetypes: Sequence[str] = ("conesearch", "tap"), limit: int = 200) -> list[RegistryResource]:
    unknown = [s for s in servicetypes if s not in REGTAP_STANDARDS]
    if unknown:
        raise VizierInputError(f"Unknown service type(s) {unknown}; use {sorted(REGTAP_STANDARDS)}")
    adql = _registry_adql(keywords, standards=[REGTAP_STANDARDS[s] for s in servicetypes],
                          waveband=_REGTAP_WAVEBAND.get(keyword or ""), limit=limit)
    table = await http.tap(adql, url=REGTAP_URL, service="IVOA RegTAP", fmt="votable")
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


_B1950 = re.compile(r"b1950|fk4|\(b\)|1950", re.IGNORECASE)
_EQUATORIAL_OK = re.compile(r"j2000|icrs|fk5", re.IGNORECASE)


def _clean(text: str | None) -> str:
    """Column description with VizieR markup removed: VizieR writes Greek letters and symbols
    in braces ('3{sigma}', '{mu}m', '{eta} Cha'), which would hide 'sigma' from the regexes."""
    return re.sub(r"\{([^{}]*)\}", r"\1", str(text or ""))


def _is_degrees(col: ColumnInfo) -> bool:
    """True when the column is in degrees, or has no unit (checked later against the data).

    ADQL POINT()/CONTAINS() need decimal degrees, so a position in seconds of time
    (J/A+A/657/A4/stars 'RAS', unit 's'), hours or radians cannot be cone-searched."""
    unit = (col.unit or "").strip()
    if not unit:
        return True
    factor = arcsec_per_unit(unit)
    return factor is not None and math.isclose(factor, 3600.0, rel_tol=1e-9)


def _position_candidates(columns: Sequence[ColumnInfo], axis: str,
                         rejected: list[ColumnInfo] | None = None) -> list[ColumnInfo]:
    """Numeric pos.eq.<axis> columns in degrees, main first, B1950/FK4 positions excluded.

    Columns skipped only because their unit is not degrees are appended to ``rejected``."""
    word = f"pos.eq.{axis}"
    cands = [c for c in columns if ucd_atoms(c.ucd)[:1] == [word] and c.numeric
             and "stat.error" not in ucd_atoms(c.ucd)]
    usable = [c for c in cands if not _B1950.search(f"{c.name} {c.description or ''}")]
    if rejected is not None:
        rejected.extend(c for c in usable if not _is_degrees(c))
    usable = [c for c in usable if _is_degrees(c)]
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
_ONE_SIGMA_WORDS = re.compile(r"standard (?:error|deviation)|\brms\b|\bsigma\b", re.IGNORECASE)
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
_NOT_POSITION = re.compile(r"flux|extent|\bext\b|size|pixel|magnitude|\bmag\b|velocity|offset from|distance|"
                           r"separation|proper motion|parallax|\bfwhm\b|major axis of the source|deconvolved",
                           re.IGNORECASE)
_TOKENS = re.compile(r"[A-Za-z_][A-Za-z0-9_()+\-.]*")


def _referenced_columns(col: ColumnInfo, names: set[str]) -> set[str]:
    """Column names an error/epoch column refers to: its ``e_<name>`` stem and names quoted in
    its description ('Error on RA_pm', 'epoch-1990 of RAdeg')."""
    refs = {tok.rstrip(".,;:)") for tok in _TOKENS.findall(_clean(col.description))} & names
    if col.name.lower().startswith("e_") and col.name[2:] in names:
        refs.add(col.name[2:])
    return refs


def detect_pos_error(columns: Sequence[ColumnInfo], ra: ColumnInfo | None = None,
                     dec: ColumnInfo | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Positional-error spec for :func:`models.compute_positional_error` from UCDs/descriptions.

    Order of preference (IVOA UCD1+):
    1. per-axis errors ``stat.error;pos.eq.ra`` + ``stat.error;pos.eq.dec`` of the chosen
       position -> kind ``sigma``, divided by the stated per-axis confidence (1-D normal:
       90% -> 1.645, 95% -> 1.960). A pair that belongs to *another* position column of the
       table (AllWISE ``e_RA_pm`` is the error of ``RA_pm``, the motion-fit position at MJD
       55400, not of RAJ2000) is only used, with an assumption, when nothing else exists;
    2. an error ellipse (``pos.errorEllipse`` or 'error ellipse' in the description): major,
       minor, position angle -> ``ellipse`` (1 sigma), ``ellipse95`` (2-D 95% contour, / 2.448)
       or ``ellipse95axis`` (per-axis 95%, / 1.96);
    3. one circular error (``stat.error;pos.eq``/``stat.error;pos``, or ``stat.error`` whose
       description is about the position): ``radius90``/``radius95`` (Rayleigh radius) for a
       stated 2-D percentage, ``radial`` for a 1-sigma sqrt(s_ra^2 + s_dec^2), else ``sigma``.
    Confidence levels are read after removing VizieR markup ('3{sigma}' -> 3 sigma).
    Returns (spec, columns used with their descriptions, assumptions).
    """
    assumptions: list[str] = []
    numeric = [c for c in columns if c.numeric]
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

    # 1. per-axis pair of the chosen position
    ra_errs, dec_errs = axis_errors("ra"), axis_errors("dec")
    fallback_pair: tuple[ColumnInfo, ColumnInfo] | None = None
    if ra_errs and dec_errs:
        era, edec = ra_errs[0], dec_errs[0]
        if not foreign(era) and not foreign(edec):
            spec, used, notes = pair_spec(era, edec)
            return spec, used, assumptions + notes
        fallback_pair = (era, edec)

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
        if fallback_pair is not None:
            assumptions.append(f"pos_error: {fallback_pair[0].name}/{fallback_pair[1].name} are the errors of "
                               f"{', '.join(sorted(foreign(fallback_pair[0]) | foreign(fallback_pair[1])))}, not of "
                               f"the chosen position; the error ellipse is used instead.")
        roles = ["major", "minor" if minor else "major"] + (["position_angle"] if pa else [])
        return spec, described(cols, roles), assumptions

    # 3. circular error
    def circular(c: ColumnInfo) -> bool:
        atoms = ucd_atoms(c.ucd)
        if not atoms or atoms[0] != "stat.error" or _error_unit(c, None) is None or foreign(c):
            return False
        if len(atoms) > 1 and atoms[1] in {"pos.eq", "pos"}:
            return True
        text = _clean(c.description)
        return len(atoms) == 1 and bool(_POSITION_WORDS.search(text)) and not _NOT_POSITION.search(text)

    circles = [c for c in numeric if circular(c)]
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
        elif re.search(r"\bradial\b|sqrt|quadrature|total", text, re.IGNORECASE) and (conf is None or conf[1] == 1.0):
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
        return spec, described([col], ["radius"]), assumptions

    if fallback_pair is not None:
        spec, used, notes = pair_spec(*fallback_pair)
        others = sorted(foreign(fallback_pair[0]) | foreign(fallback_pair[1]))
        notes.append(f"pos_error: {fallback_pair[0].name}/{fallback_pair[1].name} are described as the errors of "
                     f"{', '.join(others)}, not of the chosen position; used for lack of any other positional error.")
        return spec, used, assumptions + notes

    assumptions.append("pos_error: no positional-error column found; matches use separation only.")
    return {}, [], assumptions


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
       epRA is the mean epoch of the observations, its RAJ2000 is at epoch 2000.0).
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
    year = _fixed_year((coosys or {}).get("epoch")) if ra is not None else None
    if year is not None:
        result.update(epoch=year, source=f"VOTable COOSYS epoch {coosys.get('epoch')}")  # type: ignore[union-attr]
        for name, text in position_texts:
            stated = _EP_IN_TEXT.search(text)
            if stated and abs(float(stated.group(1)) - year) > 1e-6:
                result["notes"].append(f"epoch: COOSYS says {year:g} but {name} says '{stated.group(0)}'; the "
                                       "COOSYS epoch is used.")
        return result
    for name, text in position_texts:
        stated = _EP_IN_TEXT.search(text)
        if stated:
            result.update(epoch=float(stated.group(1)), source=f"{name} description ('{stated.group(0)}')")
            return result

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
    for text in texts:
        match = _MEAN_EPOCH.search(text or "") or _EP_IN_TEXT.search(text or "")
        if match:
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


def detect_field_map_notes(columns: Sequence[ColumnInfo]) -> tuple[dict[str, str], list[str]]:
    """Canonical physical fields chosen by UCD (proper motions, parallax, redshift, class,
    spectral type), with notes on columns deliberately left out.

    ``src.redshift`` is mapped to the canonical (dimensionless) redshift only when its unit is
    empty/dimensionless: VizieR tags recession velocities cz in km/s with the same UCD
    (J/ApJ/956/51/table4 'Local Group-corrected Recession Velocity'), and z = v/c holds only for
    heliocentric velocities, so velocity columns are never converted silently."""
    def atoms(c: ColumnInfo) -> list[str]:
        return ucd_atoms(c.ucd)

    def has_unit(c: ColumnInfo) -> bool:
        return bool((c.unit or "").strip())

    notes: list[str] = []
    mapping: dict[str, str] = {}
    pmra = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:2] == ["pos.pm", "pos.eq.ra"])
    pmdec = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:2] == ["pos.pm", "pos.eq.dec"])
    if pmra and pmdec:
        mapping.update(pmra=pmra.name, pmdec=pmdec.name)
    plx = _pick_best(columns, lambda c: c.numeric and has_unit(c) and atoms(c)[:1] in (["pos.parallax.trig"], ["pos.parallax"]))
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
        if redshift_like(c) and not dimensionless(c):
            notes.append(f"redshift: {c.name} is tagged src.redshift but its unit is '{c.unit}' (a velocity such as cz, "
                         "not a redshift); it is not mapped to the canonical redshift.")
    exact = [c for c in columns if redshift_like(c) and dimensionless(c) and atoms(c) == ["src.redshift"]]
    z = exact[0] if exact else _pick_best(columns, lambda c: redshift_like(c) and dimensionless(c))
    if z:
        mapping["redshift"] = z.name
    otype = _pick_best(columns, lambda c: atoms(c)[:1] == ["src.class"])
    if otype:
        mapping["object_type"] = otype.name
    sptype = _pick_best(columns, lambda c: bool(atoms(c)) and atoms(c)[0].startswith("src.sptype"))
    if sptype:
        mapping["spectral_type"] = sptype.name
    return mapping, notes


def detect_field_map(columns: Sequence[ColumnInfo]) -> dict[str, str]:
    """Canonical physical fields chosen by UCD (see :func:`detect_field_map_notes`)."""
    return detect_field_map_notes(columns)[0]


_COUNT_LIKE = re.compile(r"\b(?:number|count|numb|nb|num)\b\s+of|\bnumber\b|\bcount\b", re.IGNORECASE)


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
    ras = _position_candidates(columns, "ra")
    decs = _position_candidates(columns, "dec")
    ra = ras[0] if ras else None
    dec = _pair_dec(ra, decs) if ra else None
    if ra is None or dec is None:
        problems.append("no numeric equatorial position columns (UCD pos.eq.ra / pos.eq.dec, J2000/ICRS) -- "
                        "the table cannot be cone-searched")
    frame = None
    coosys: dict[str, Any] = {}
    if ra is not None:
        field_meta = fields.get(ra.name) or {}
        coosys = (asu_table.get("coosys") or {}).get(field_meta.get("ref") or "", {}) or {}
        if coosys:
            frame = coosys.get("system")
            if coosys.get("equinox"):
                frame = f"{frame} (equinox {coosys['equinox']})"
            if str(coosys.get("system") or "").lower() in {"eq_fk4", "fk4"}:
                problems.append(f"{ra.name} is an FK4/B1950 position; no J2000/ICRS alternative was found")
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
    id_col = _pick_best(columns, lambda c: ucd_atoms(c.ucd) == ["meta.id", "meta.main"]) or _pick_best(
        columns, lambda c: ucd_atoms(c.ucd)[:1] == ["meta.id"] and len(ucd_atoms(c.ucd)) == 1)
    if id_col is None:
        assumptions.append("identifier: no meta.id column; sources are numbered <catalog>-<row>.")
    pos_error, error_cols, error_notes = detect_pos_error(columns, ra, dec)
    assumptions.extend(error_notes)
    texts = [table_row.get("description"), asu_table.get("description"), catalog.title]
    epoch = detect_epoch(columns, ra, coosys, texts)
    assumptions.extend(epoch["notes"])
    field_map = detect_field_map(columns)
    wavelength, wavelength_note = _wavelength_of(catalog)
    if wavelength_note:
        assumptions.append(wavelength_note)
    return TableDescription(
        table_id=table_id,
        catalog=catalog,
        description=table_row.get("description"),
        nrows=int(table_row["nrows"]) if table_row.get("nrows") is not None else None,
        columns=list(columns),
        ra_column=ra.name if ra else None,
        dec_column=dec.name if dec else None,
        id_column=id_col.name if id_col else None,
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
    return CatalogInfo(catalog_id=catalog_id)


async def describe(identifier: str, *, client: httpx.AsyncClient | None = None,
                   timeout: float | None = None) -> TableDescription | CatalogInfo:
    """Describe a VizieR table (``'IX/58/2sxps'``) or list the tables of a catalogue (``'IX/58'``)."""
    ident = validate_vizier_id(identifier)
    async with _Http(client, timeout=timeout) as http:
        tables = await http.tap(
            "SELECT table_name, description, nrows FROM TAP_SCHEMA.tables "
            f"WHERE table_name = {_adql_string(quote_identifier(ident))} "
            f"OR table_name LIKE {_adql_string(chr(34) + ident + '/%')}"
        )
        rows = {_unquote(str(r.get("table_name"))): r for r in tables.rows}
        if ident in rows:
            return await _describe_table(http, ident, rows[ident])
        children = sorted(t for t in rows if catalog_of(t) == ident)
        if not children:
            raise VizierNotFoundError(f"VizieR has no table or catalogue '{ident}'.")
        info = await _catalog_info(http, ident)
        info.tables = [TableHit(table_id=t, catalog_id=ident, description=rows[t].get("description"),
                                nrows=int(rows[t]["nrows"]) if rows[t].get("nrows") is not None else None,
                                catalog_title=info.title, popularity=info.popularity,
                                wavelengths=list(info.wavelengths), bibcode=info.bibcode) for t in children]
        return info


async def describe_table(table_id: str, *, client: httpx.AsyncClient | None = None,
                         timeout: float | None = None) -> TableDescription:
    """Describe one VizieR table; a catalogue id with exactly one table is accepted."""
    result = await describe(table_id, client=client, timeout=timeout)
    if isinstance(result, TableDescription):
        return result
    if len(result.tables) == 1:
        return await describe(result.tables[0].table_id, client=client, timeout=timeout)  # type: ignore[return-value]
    raise VizierInputError(
        f"'{result.catalog_id}' is a catalogue with {len(result.tables)} tables; choose one of: "
        + ", ".join(t.table_id for t in result.tables)
    )


async def _describe_table(http: _Http, table_id: str, table_row: Mapping[str, Any]) -> TableDescription:
    cols_q = ("SELECT column_name, description, unit, ucd, datatype, principal, indexed FROM TAP_SCHEMA.columns "
              f"WHERE table_name = {_adql_string(quote_identifier(table_id))}")
    cols_table, info, asu_root = await asyncio.gather(
        http.tap(cols_q), _catalog_info(http, catalog_of(table_id)),
        http.asu({"-source": table_id, "-meta.all": ""}),
    )
    columns = _columns_from_rows(cols_table.rows)
    if not columns:
        raise VizierNotFoundError(f"TAP_SCHEMA lists no columns for '{table_id}'.")
    asu_table = parse_table_meta(asu_root)
    description = analyse_table(table_id, info, table_row, columns, asu_table)
    if description.ra_column and description.position_unit_check:
        # No unit on the position columns: the largest RA (one indexed TOP 1 query, ~1 s even
        # for 2MASS) tells decimal degrees (> 24) from hours.
        ra_q = quote_identifier(description.ra_column)
        top = await http.tap(f"SELECT TOP 1 {ra_q} FROM {quote_identifier(table_id)} "
                             f"WHERE {ra_q} IS NOT NULL ORDER BY {ra_q} DESC")
        values = [v for v in (_float(r.get(description.ra_column)) for r in top.rows) if v is not None]
        description = analyse_table(table_id, info, table_row, columns, asu_table,
                                    max_ra=max(values) if values else None)
    return description


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_OVERRIDE_KEYS = frozenset({
    "wavelength", "max_rows", "timeout_seconds", "pos_error", "epoch", "epoch_format", "epoch_range",
    "systematic_arcsec", "profiles", "enabled", "description", "coverage", "extra_columns", "id_column",
})
# Canonical fields a stray selected column could feed through models.ucd_field_map.
_GUARDED_CANONICAL = frozenset({"pmra", "pmdec", "parallax", "redshift", "object_type", "spectral_type", "epoch",
                                "ra_error_arcsec", "dec_error_arcsec"})


def _table_title(text: str | None) -> str | None:
    """TAP_SCHEMA table description without VizieR's trailing '( authors)' list."""
    if not text:
        return None
    return re.sub(r"\s*\(\s[^()]*\)\s*$", "", text).strip() or text


def build_definition(
    description: TableDescription,
    *,
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Registry entry (the dict format of models.DEFAULT_CATALOGS / the registry YAML).

    Columns: identifier, position, positional-error, epoch and canonical physical columns, then
    the table's principal columns (TAP_SCHEMA ``principal``) up to 40. A column whose UCD would
    silently feed a canonical field (``models.ucd_field_map``) is only selected when it is the
    column chosen for that field. Raises :class:`VizierRegistrationError` when the table cannot be
    cone-searched or the definition fails :func:`models.validate_catalog_definition`.
    """
    if description.problems:
        raise VizierRegistrationError(f"{description.table_id} cannot be registered: " + "; ".join(description.problems))
    overrides = dict(overrides or {})
    unknown = set(overrides) - _OVERRIDE_KEYS
    if unknown:
        raise VizierInputError(f"Unknown override(s): {sorted(unknown)}; allowed: {sorted(_OVERRIDE_KEYS)}")
    catalog_name = (name or default_catalog_name(description.table_id)).strip().lower()
    if not _NAME.match(catalog_name):
        raise VizierInputError(f"Invalid catalog name {catalog_name!r}: lower-case letters, digits, '_' or '-' (max 64).")
    by_name = {c.name: c for c in description.columns}
    ra, dec = description.ra_column or "", description.dec_column or ""
    id_col = overrides.pop("id_column", None) or description.id_column
    if id_col is not None and id_col not in by_name:
        raise VizierInputError(f"id_column {id_col!r} is not a column of {description.table_id}")

    pos_error = dict(overrides.pop("pos_error", None) or description.pos_error)
    systematic = overrides.pop("systematic_arcsec", None)
    if systematic is not None:
        value = _float(systematic)
        if value is None or value < 0:
            raise VizierInputError("systematic_arcsec must be a non-negative number")
        if not pos_error:
            raise VizierInputError("systematic_arcsec needs a positional-error column (or a pos_error override)")
        pos_error["systematic_arcsec"] = value
    epoch = overrides.pop("epoch", description.epoch)
    epoch_format = overrides.pop("epoch_format", description.epoch_format)
    epoch_range = overrides.pop("epoch_range", description.epoch_range)
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        epoch_range = None  # a fixed epoch needs no span (and the validator rejects both)

    required: list[str] = [c for c in (id_col, ra, dec) if c]
    required += [str(c) for c in pos_error.get("columns") or []]
    if isinstance(epoch, str):
        required.append(epoch)
    elif isinstance(epoch, Mapping):
        required += [str(c) for c in epoch.get("span_columns") or []]
    elif isinstance(epoch, list):
        required += [str(c) for c in epoch]
    required += list(description.field_map.values())
    extra = [str(c) for c in overrides.pop("extra_columns", None) or []]
    missing = [c for c in required + extra if c not in by_name]
    if missing:
        raise VizierInputError(f"column(s) {missing} are not in {description.table_id}")

    chosen = {canon: col for canon, col in description.field_map.items()}
    selected: list[str] = []
    for col in required + extra:
        if col not in selected:
            selected.append(col)
    for col in description.columns:
        if len(selected) >= MAX_SELECTED_COLUMNS:
            break
        if not (col.principal or col.displayed) or col.name in selected:
            continue
        roles = set(ucd_field_map([col.as_meta()]))
        if roles & _GUARDED_CANONICAL and not any(chosen.get(r) == col.name for r in roles):
            continue
        selected.append(col.name)

    column_units = {}
    for col_name in (ra, dec):
        if col_name and not (by_name[col_name].unit or "").strip():
            column_units[col_name] = "deg"
    wavelength = str(overrides.pop("wavelength", None) or description.wavelength)
    profiles = overrides.pop("profiles", None)
    if profiles is None:
        profiles = ["full", "vizier", *_WAVELENGTH_PROFILES.get(wavelength, ())]
    parameters: dict[str, Any] = {
        "columns": [quote_identifier(c) for c in selected],
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


def user_registry_path(path: str | os.PathLike[str] | None = None) -> Path:
    """User registry YAML: explicit path > CATALOG_REGISTRY_PATH > ~/.astrosearch/catalogs.yaml."""
    if path:
        return Path(path).expanduser()
    env = os.getenv("CATALOG_REGISTRY_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".astrosearch" / "catalogs.yaml"


def read_user_registry(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """The user registry document ({'extends': 'embedded', 'catalogs': {...}}); empty when absent."""
    target = user_registry_path(path)
    if not target.exists():
        return {}
    try:
        content = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"Catalog registry {target} could not be parsed: {exc}") from exc
    if not isinstance(content, dict) or not isinstance(content.get("catalogs", {}) or {}, dict):
        raise RegistryError(f"Catalog registry {target} must contain a 'catalogs' mapping.")
    content["catalogs"] = content.get("catalogs") or {}
    return content


def _write_user_registry(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# AstroSearch user catalog registry. 'extends: embedded' merges these entries over the\n"
              "# built-in registry (vizier.UserCatalogRegistry); entries added by 'vizier add'.\n")
    text = header + yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True, width=120)
    fd, tmp = tempfile.mkstemp(prefix=".catalogs.", suffix=".yaml", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def list_registered(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """Catalog entries stored in the user registry YAML."""
    return dict(read_user_registry(path).get("catalogs") or {})


def save_definition(name: str, entry: Mapping[str, Any], *, path: str | os.PathLike[str] | None = None,
                    replace: bool = False, source: Mapping[str, Any] | None = None) -> tuple[Path, bool]:
    """Persist one entry in the user registry; returns (path, replaced)."""
    target = user_registry_path(path)
    if name in DEFAULT_CATALOGS:
        raise VizierRegistrationError(f"'{name}' is a built-in catalog name; choose another name.")
    document = read_user_registry(target)
    if not document:
        document = {"extends": "embedded", "catalogs": {}}
    catalogs = document.setdefault("catalogs", {})
    existed = name in catalogs
    if existed and not replace:
        previous = (catalogs[name].get("source") or {}).get("table")
        raise VizierRegistrationError(
            f"'{name}' is already registered{f' (VizieR {previous})' if previous else ''}; pass replace=True to overwrite."
        )
    stored = json.loads(json.dumps(dict(entry)))  # plain YAML-safe types
    if source:
        stored["source"] = dict(source)
    catalogs[name] = stored
    _write_user_registry(document, target)
    return target, existed


def unregister(name: str, *, path: str | os.PathLike[str] | None = None) -> bool:
    """Remove a user catalog; returns False when it was not registered."""
    target = user_registry_path(path)
    document = read_user_registry(target)
    if name not in (document.get("catalogs") or {}):
        return False
    del document["catalogs"][name]
    _write_user_registry(document, target)
    return True


class UserCatalogRegistry(CatalogRegistry):
    """The embedded registry with the user registry YAML merged over it.

    File semantics: absent -> embedded catalogs only; ``extends: embedded`` (written by
    :func:`save_definition`) -> embedded + file entries (file wins on equal names); a file
    without ``extends`` is a complete registry, exactly as :class:`models.CatalogRegistry`
    reads it.
    """

    def __init__(self, registry_path: str | os.PathLike[str] | None = None) -> None:
        super().__init__(user_registry_path(registry_path))

    def reload(self) -> None:
        document = read_user_registry(self.registry_path)
        merged: dict[str, Any] = {}
        if not document or document.get("extends") == "embedded":
            merged.update(DEFAULT_CATALOGS)
        merged.update(document.get("catalogs") or {})
        self._catalogs = {name: catalog_from_dict(name, entry) for name, entry in merged.items()
                          if isinstance(entry, dict)}

    def add(self, definition: CatalogDefinition) -> None:
        """Make a definition available immediately (without touching the file)."""
        self._catalogs[definition.name] = definition


def load_registry(path: str | os.PathLike[str] | None = None) -> UserCatalogRegistry:
    """Embedded + user catalogs (see :class:`UserCatalogRegistry`)."""
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
    registry: CatalogRegistry | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float | None = None,
) -> Registration:
    """Describe a VizieR table, build its definition, save it to the user registry and
    (optionally) attach it to a live ``registry``."""
    description = await describe_table(table_id, client=client, timeout=timeout)
    catalog_name, entry = build_definition(description, name=name, overrides=overrides)
    source = {
        "service": "vizier",
        "table": description.table_id,
        "catalog": description.catalog.catalog_id,
        "title": description.catalog.title,
        "bibcode": description.catalog.bibcode,
        "doi": description.catalog.doi,
        "nrows": description.nrows,
        "epoch_source": description.epoch_source,
        "assumptions": list(description.assumptions),
        "overrides": dict(overrides or {}),
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    saved, replaced = save_definition(catalog_name, entry, path=path, replace=replace, source=source)
    attached = False
    if registry is not None:
        attach_definition(registry, catalog_name, entry)
        attached = True
    return Registration(name=catalog_name, table_id=description.table_id, entry=entry, path=str(saved),
                        replaced=replaced, assumptions=list(description.assumptions), attached=attached)


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
    if isinstance(exc, (VizierInputError, VizierRegistrationError)):
        status = 409 if "already registered" in str(exc) else 422
        return HTTPException(status_code=status, detail=str(exc))
    if isinstance(exc, RegistryError):
        return HTTPException(status_code=500, detail=f"User registry unreadable: {exc}")
    return HTTPException(status_code=502, detail=f"Upstream failure: {type(exc).__name__}: {exc}")


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
    try:
        result = await search_catalogs(q, ucd=ucd, wavelength=wavelength, max_catalogs=max_catalogs,
                                       max_tables=max_tables, include_registry=registry, client=_state(request, "client"))
    except (VizierError, RegistryError) as exc:
        raise _http_error(exc) from exc
    return result.as_dict()


@router.get("/catalog/{identifier:path}", response_model=dict[str, Any])
async def describe_endpoint(request: Request, identifier: str) -> dict[str, Any]:
    """Describe a VizieR table (columns, UCDs, units, rows, reference, position/error/epoch columns)
    or list a catalogue's tables."""
    try:
        result = await describe(identifier, client=_state(request, "client"))
    except (VizierError, RegistryError) as exc:
        raise _http_error(exc) from exc
    data = result.as_dict()
    data["kind"] = "table" if isinstance(result, TableDescription) else "catalog"
    return data


def _live_registries(request: Request) -> list[CatalogRegistry]:
    found: list[CatalogRegistry] = []
    for candidate in (_state(request, "registry"), getattr(_state(request, "service"), "registry", None)):
        if isinstance(candidate, CatalogRegistry) and all(candidate is not r for r in found):
            found.append(candidate)
    return found


@router.post("/register", response_model=dict[str, Any])
async def register_endpoint(request: Request, body: RegisterRequest) -> dict[str, Any]:
    """Register a VizieR table as a crossmatch catalog (saved to the user registry YAML and
    attached to the running service's registry)."""
    path = _state(request, "vizier_registry_path")
    try:
        registration = await register_table(body.table_id, name=body.name, overrides=body.overrides, path=path,
                                            replace=body.replace, client=_state(request, "client"))
    except (VizierError, RegistryError) as exc:
        raise _http_error(exc) from exc
    for live in _live_registries(request):
        attach_definition(live, registration.name, registration.entry)
        registration.attached = True
    return registration.as_dict()


@router.get("/registered", response_model=dict[str, Any])
async def registered_endpoint(request: Request) -> dict[str, Any]:
    """User catalogs from the registry YAML and whether the running service knows them."""
    path = user_registry_path(_state(request, "vizier_registry_path"))
    try:
        entries = list_registered(path)
    except RegistryError as exc:
        raise _http_error(exc) from exc
    live = _live_registries(request)
    return {
        "path": str(path),
        "count": len(entries),
        "catalogs": [
            {"name": name, "table": (entry.get("source") or {}).get("table"), "wavelength": entry.get("wavelength"),
             "citation": entry.get("citation"), "enabled": entry.get("enabled", True),
             "active": any(name in r.catalogs for r in live) if live else None, "entry": entry}
            for name, entry in entries.items()
        ],
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


def _cli_search(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(search_catalogs(args.keywords, ucd=args.ucd, wavelength=args.wavelength,
                                             max_catalogs=args.max, include_registry=args.registry))
    except (VizierError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1
    print(json.dumps(result.as_dict(), indent=2, default=str) if args.json else format_search(result, args.limit))
    return 0


def _cli_describe(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(describe(args.table))
    except (VizierError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1
    print(json.dumps(result.as_dict(), indent=2, default=str) if args.json else format_description(result))
    return 0


def _cli_add(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.overrides:
        try:
            overrides = json.loads(args.overrides)
        except json.JSONDecodeError as exc:
            print(f"Error: --overrides is not valid JSON: {exc}")
            return 2
    if args.systematic is not None:
        overrides["systematic_arcsec"] = args.systematic
    try:
        registration = asyncio.run(register_table(args.table, name=args.name, overrides=overrides or None,
                                                  path=args.registry_path, replace=args.replace))
    except (VizierError, RegistryError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1
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
        entries = list_registered(args.registry_path)
    except RegistryError as exc:
        print(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(entries, indent=2, default=str))
        return 0
    print(f"{len(entries)} user catalog(s) in {user_registry_path(args.registry_path)}")
    for name, entry in entries.items():
        print(f"  {name:<32}{(entry.get('source') or {}).get('table', ''):<30}{entry.get('wavelength')}")
    return 0


def register_cli(subparsers: Any) -> None:
    """Add ``vizier search|describe|add|list`` to an argparse subparsers object."""
    parser = subparsers.add_parser("vizier", help="Discover, describe and register any VizieR table")
    sub = parser.add_subparsers(dest="vizier_command")
    parser.set_defaults(handler=lambda args: (parser.print_help(), 2)[1])

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



if __name__ == "__main__":
    _parser = argparse.ArgumentParser(prog="vizier")
    register_cli(_parser.add_subparsers(dest="command"))
    _args = _parser.parse_args()
    raise SystemExit(_args.handler(_args) if hasattr(_args, "handler") else 2)
