"""Reproducibility manifests, replay diffs, and verified citations for AstroSearch results.

Three pieces, all usable from notebooks without the web API:

* **Manifests** -- :func:`build_manifest` turns a :class:`models.UnifiedRecord` (or its
  ``as_dict()`` form) into a :class:`ProvenanceManifest`: software version and git
  commit, UTC time, the input target and radius, the query mode/parameters, the exact
  request sent to every catalog (endpoint, HTTP method, form/query parameters, ADQL or
  SQL text), the catalog data release, row counts, and two hashes:

  - ``query_hash``: SHA-256 of the canonical JSON of *what was asked* (input target,
    radius, query options, per-catalog requests);
  - ``content_hash``: SHA-256 of the canonical JSON of *what came back* -- the
    scientific payload (target, per-catalog status and rows, matches, groups) with all
    volatile fields (``retrieved_at``, ``elapsed_ms``, cache flags, warnings, provider
    metadata) left out, so identical science gives an identical hash.

  Canonical JSON follows RFC 8785 (JSON Canonicalization Scheme) where it matters for
  browser round trips: keys sorted, no insignificant whitespace, UTF-8, and numbers
  written as ECMAScript writes them -- a whole-valued float is an integer (``2000.0``
  -> ``2000``, ``-0.0`` -> ``0``), because ``JSON.stringify`` in a browser writes it
  that way and Python then reads it back as an int. Floats are first rounded to
  :data:`FLOAT_SIGNIFICANT_DIGITS` significant digits so a result does not change hash
  through last-bit floating-point noise (12 digits of a right ascension near 180 deg
  is 1e-9 deg = 3.6 micro-arcsec). Integers beyond +-(2**53 - 1) (Gaia ``source_id``,
  SDSS ``objID``) are written as decimal strings, because JavaScript's ``JSON.parse``
  turns them into the nearest IEEE-754 double; the content hash additionally reads
  such integers *inside archive columns* at double precision, so a record that went
  through a browser hashes like the original (the row identity is the exact string
  ``source_id``, which is unaffected). The manifest's input position, radius and
  query options are stored at full float precision, so a replay asks exactly the
  original question. The tests check all of this with a real ``node`` round trip.

  A proper motion or parallax given with the query (by the caller, or by the Sesame
  name resolver as in ``main.search_object`` and ``POST /api/v1/search``) is part of
  the question and is recorded with its origin (``query.pm_source`` /
  ``query.parallax_source``); one the crossmatch adopted from a catalog row is part of
  the answer (``science.target_proper_motion`` / ``science.target_parallax``) and is
  adopted again by a replay. HEASARC tables whose release changes in place
  (``xmmssc``, ``csc``) have the release read live from ``TAP_SCHEMA.tables``
  (:func:`live_releases`) -- used only when read within an hour of the query
  (``query_time``: the search the manifest ran, or the earliest row ``retrieved_at``
  of a stored record); otherwise the label is kept and marked unverified. Curated
  release labels and citations are attached only to the built-in service/table of a
  catalog (:func:`matches_curated`). An empty catalog's request is the ADQL/SQL the
  provider recorded, checked against the registry. A catalog answered by its declared
  fallback archive (IRSA Gator for 2MASS/AllWISE when IRSA TAP is down) records the
  fallback and the primary request that failed.

* **Replay** -- :func:`replay_manifest` re-runs the manifest's query through
  :class:`crossmatch.CrossmatchService` (same input target, proper-motion origin,
  target uncertainties, radius, query options and catalog set, provider caches
  bypassed; a catalog the original got from its fallback archive asks that archive
  first) and returns a structured diff (:func:`diff_science`): added / removed / moved
  / changed sources per catalog (moves judged against each row's 1-sigma error; rows
  served by different archive services compared on their common columns at single
  precision), status changes, match separation/confidence changes, request changes
  (with the serving archive), data-release changes, catalogs that failed in both runs
  (``not_compared``), an ``identical`` flag (equal content hashes and every catalog
  compared) and a plain-language explanation. The CPU work (hashing, diffing) runs in
  a worker thread so the API's event loop keeps serving.

* **Citations** -- :func:`citations_for` returns the acknowledgement text and BibTeX
  for the catalogs and services used. Every bibcode in :data:`REFERENCES` was checked
  against NASA ADS on 2026-09-28 through the ADS link gateway
  (``https://ui.adsabs.harvard.edu/link_gateway/<bibcode>/<link type>`` redirects for a
  known bibcode with full text, answers 200/302 for the associated/data/SIMBAD links of
  a known bibcode without full text, and 404 for an unknown one; the abstract pages
  themselves sit behind a bot check) and, where the gateway redirected to a DOI, the
  DOI's CSL-JSON metadata (title, year, volume, first page) was compared with the
  entry. :func:`verify_reference` repeats that check live (``astrosearch cite --verify``).
  Acknowledgement paragraphs prescribed by the data providers are stored verbatim.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import html
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from crossmatch import AdvancedQuery, CrossmatchService, QueryValidator, validate_profile
from models import (
    MAX_SEARCH_RADIUS_ARCSEC,
    CatalogDefinition,
    CatalogRegistry,
    InvalidCoordinateError,
    Settings,
    Target,
    UnifiedRecord,
    check_search_radius,
    haversine_arcsec,
    plan_cone,
    validate_target,
)
from providers import (
    CacheManager,
    HEASARCXaminProvider,
    IRSAGatorProvider,
    MASTProvider,
    SDSSProvider,
    SesameResolver,
    TapProvider,
    provider_map,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# /2: integers beyond 2**53 as strings (JavaScript-safe), archive-column big integers
# hashed at double precision, full-precision input target and query, pm_source recorded.
# /3: whole-valued floats written as integers (ECMAScript number form, so hashes survive
# a browser round trip); target parallax and its origin in the science payload;
# query.parallax_source; catalog citations from the verified curated references.
MANIFEST_SCHEMA = "astrosearch.provenance.manifest/3"
SCIENCE_SCHEMA = "astrosearch.provenance.science/3"
HASH_ALGORITHM = "sha256"
FLOAT_SIGNIFICANT_DIGITS = 12
# Largest integer a JavaScript number (IEEE-754 binary64) holds exactly: Number.MAX_SAFE_INTEGER.
MAX_SAFE_INTEGER = 2**53 - 1
HASH_METHOD = (
    "sha256 over canonical JSON (keys sorted, compact separators, UTF-8, floats rounded to "
    f"{FLOAT_SIGNIFICANT_DIGITS} significant digits and whole-valued floats written as integers (ECMAScript "
    "number form), NaN/Infinity as strings, integers beyond 2**53-1 as "
    "decimal strings; inside archive columns such integers are read at IEEE-754 double precision) of the "
    "scientific payload"
)
# Replay tolerances for "moved" rows. A row counts as moved when its position changed
# by more than position_tolerance_arcsec OR by more than sigma_fraction x its quoted
# 1-sigma positional error, whichever is smaller. The absolute 1 mas cap alone is not
# enough: in a live Gaia DR3 cone around 3C 273, 72% of rows have ra_error and
# dec_error < 1 mas (median 0.018 mas at G<15, 0.043 at G 15-17, 0.148 at G 17-19,
# 0.379 at G 19-20), so a 1 mas cap would call a move of tens of sigma "equivalent".
DEFAULT_POSITION_TOLERANCE_ARCSEC = 1e-3
DEFAULT_SIGMA_FRACTION = 0.1
DEFAULT_NUMERIC_RTOL = 1e-9
# Relative precision of IEEE-754 single precision (2**-23, machine epsilon): values an
# archive serves as float32 (IRSA TAP VOTable columns) and as decimal text (IRSA Gator)
# agree to this, e.g. 2MASS j_m 11.766 vs 11.765999794 (1.8e-8).
FLOAT32_RTOL = 2.0**-23
# Match confidences are software scores in [0, 1]; any change above float noise means
# the scoring changed.
DEFAULT_CONFIDENCE_TOLERANCE = 1e-6
ROW_HASH_HEX_CHARS = 16

# Archive columns holding the row's RA / Dec in degrees (lower-cased names as served by
# Gaia, SIMBAD, NED, IRSA, SDSS, HEASARC ("ra"/"dec"), MAST ("raMean"/"decMean") and
# VizieR ("RAJ2000"/"DEJ2000", "RA_ICRS"/"DE_ICRS")). Compared as angles, not with rtol.
RA_COLUMNS = frozenset({"ra", "ramean", "raj2000", "_raj2000", "ra_icrs", "radeg", "ra_deg"})
DEC_COLUMNS = frozenset({"dec", "decmean", "dej2000", "_dej2000", "decj2000", "de_icrs", "dec_icrs", "dedeg", "dec_deg"})

# Target proper-motion origins (provenance.target_proper_motion.source) that were
# derived by the crossmatch from catalog rows, i.e. NOT part of the query as given.
# Anything else ("input", "resolver", ...) was supplied with the query.
DERIVED_PM_ORIGINS = frozenset({"adopted", "extragalactic"})
# Target parallax origins (provenance.target_parallax.source) derived by the crossmatch
# (crossmatch._adopt_parallax / _adopt_proper_motion: taken from a catalog row whose
# motion agrees with the target's). "input" and "resolver" were given with the query.
DERIVED_PARALLAX_ORIGINS = frozenset({"adopted"})

# Errors that mean "the archive could not be reached", not "the archive answered".
UNREACHABLE_ERRORS = frozenset({"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError"})

VIZIER_HOSTS = ("tapvizier.cds.unistra.fr", "vizier.cds.unistra.fr", "vizier.u-strasbg.fr")
HEASARC_HOSTS = ("heasarc.gsfc.nasa.gov",)
IRSA_HOSTS = ("irsa.ipac.caltech.edu",)

ADS_ABS_URL = "https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
ADS_GATEWAY_URL = "https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{link_type}"
ADS_LINK_TYPES = ("PUB_HTML", "EPRINT_HTML", "ADS_PDF", "ADS_SCAN", "PUB_PDF")
# Asked when every full-text type is 404: a real bibcode without full text (e.g. the AAS
# abstract 2015AAS...22533616H, APASS DR9) still answers 200/302 for one of these, while
# an unknown bibcode (2015AAS...22533616X) answers 404 for all of them (probed 2026-09-28).
ADS_OTHER_LINK_TYPES = ("ASSOCIATED", "DATA", "ESOURCE", "SIMBAD")
DOI_RESOLVER_URL = "https://doi.org/{doi}"
CSL_JSON = "application/vnd.citationstyles.csl+json"
USER_AGENT = "astrosearch-provenance (citation verification)"


class ManifestError(ValueError):
    """A manifest is malformed, has an unknown schema, or fails its integrity check."""


class ReplayUnavailableError(RuntimeError):
    """Every catalog of a replay failed to reach its archive (upstream outage)."""


class UnknownCitationError(KeyError):
    """A catalog or service name has no citation entry."""


class BibTeXError(ValueError):
    """BibTeX text failed validation."""


# ---------------------------------------------------------------------------
# Canonical JSON & Hashing
# ---------------------------------------------------------------------------


def _js_float(value: float) -> int | float | str:
    """A finite float the way ECMAScript (and RFC 8785, section 3.2.2.3) writes it.

    JavaScript has one number type: ``JSON.stringify(2000.0)`` is ``2000`` and
    ``JSON.stringify(-0)`` is ``0``, which Python's ``json`` then reads back as an *int*.
    So a whole-valued float is written as an integer (and an integer beyond
    +-:data:`MAX_SAFE_INTEGER` -- every float of magnitude >= 2**53 is whole -- as the
    decimal string of its exact value, the big-integer rule of :func:`canonicalize`).
    Non-whole floats keep Python's shortest round-trip repr, which parses to the same
    IEEE-754 double as ECMAScript's Number::toString output.
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    value = value + 0.0  # folds -0.0 into 0.0
    if value.is_integer():
        whole = int(value)
        return whole if abs(whole) <= MAX_SAFE_INTEGER else str(whole)
    return value


def _round_float(value: float, digits: int = FLOAT_SIGNIFICANT_DIGITS) -> int | float | str:
    if not math.isfinite(value):
        return _js_float(value)
    return _js_float(float(format(value, f".{digits}g")))


def _exact_float(value: float) -> int | float | str:
    return _js_float(value)


def canonicalize(value: Any, *, round_floats: bool = True) -> Any:
    """Convert ``value`` into plain JSON types with a deterministic representation.

    numpy scalars/arrays, Decimal, dataclasses, tuples and sets are converted; floats
    are rounded to :data:`FLOAT_SIGNIFICANT_DIGITS` significant digits (kept exact with
    ``round_floats=False``) and a whole-valued float is written as an integer, as
    ECMAScript does (:func:`_js_float`), so a number's canonical form does not depend on
    whether it last passed through Python or JavaScript; integers beyond
    +-:data:`MAX_SAFE_INTEGER` become decimal strings (JavaScript-safe); dictionary keys
    become strings; sets are sorted. Bytes are decoded as UTF-8 (invalid bytes replaced).
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return str(value) if abs(value) > MAX_SAFE_INTEGER else value
    if isinstance(value, float):
        return _round_float(value) if round_floats else _exact_float(value)
    if isinstance(value, Decimal):
        return _round_float(float(value)) if round_floats else _exact_float(float(value))
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): canonicalize(v, round_floats=round_floats) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value), round_floats=round_floats)
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(v, round_floats=round_floats) for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [canonicalize(v, round_floats=round_floats) for v in value]
    # numpy (imported lazily through duck typing: np.generic has .item(), ndarray .tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return canonicalize(value.tolist(), round_floats=round_floats)
    if hasattr(value, "item") and callable(value.item):
        return canonicalize(value.item(), round_floats=round_floats)
    return str(value)


def exact(value: Any) -> Any:
    """:func:`canonicalize` without float rounding (for values a replay must reuse exactly)."""
    return canonicalize(value, round_floats=False)


_BIG_INT_TEXT = re.compile(r"-?\d{16,}")


def js_number_value(value: Any) -> Any:
    """``value`` as a JavaScript ``JSON.parse`` would hold it, for big integers.

    An integer (or canonical decimal-integer string) beyond :data:`MAX_SAFE_INTEGER`
    becomes the decimal text of the nearest IEEE-754 double (both Python's int->float
    and JavaScript's number parsing round to nearest, ties to even), so the original
    3700386905605055360 and its browser round trip 3700386905605055000 map to the same
    value. Anything else is returned unchanged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) > MAX_SAFE_INTEGER:
        return str(int(float(value)))
    if isinstance(value, str) and _BIG_INT_TEXT.fullmatch(value):
        number = int(value)
        if abs(number) > MAX_SAFE_INTEGER:
            return str(int(float(number)))
    return value


def _js_normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _js_normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_normalized(v) for v in value]
    if isinstance(value, float) or value is None:  # fast path: nothing to change
        return value
    if isinstance(value, str) and len(value) < 16:
        return value
    return js_number_value(value)


def canonical_json(value: Any) -> str:
    """Canonical JSON text of ``value`` (sorted keys, compact, UTF-8 characters kept)."""
    return _dumps_canonical(canonicalize(value))


def _dumps_canonical(value: Any) -> str:
    """JSON text of a value that is *already* canonical (output of :func:`canonicalize`).

    :func:`canonicalize` is idempotent (a 12-digit float rounds to itself, whole floats
    are already ints, big integers already strings), so this equals
    ``canonical_json(value)`` for such values without walking them again.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(value: Any) -> str:
    """Hex SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_canonical(value: Any) -> str:
    """:func:`sha256_hex` of a value that is already canonical."""
    return hashlib.sha256(_dumps_canonical(value).encode("utf-8")).hexdigest()


def _prefixed(digest: str) -> str:
    return f"{HASH_ALGORITHM}:{digest}"


# ---------------------------------------------------------------------------
# Software Environment
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pyproject_version() -> str | None:
    path = Path(__file__).resolve().parent / "pyproject.toml"
    try:
        import tomllib

        return str(tomllib.loads(path.read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, ValueError):
        return None


def git_state_at(path: str | Path) -> dict[str, Any]:
    """Commit hash and dirty flag of the git checkout containing ``path``.

    ``dirty`` is True when tracked files are modified *or* untracked ``*.py`` files exist
    (code the commit does not contain, such as a new module the running server imports).
    Returns ``{"commit": None, ...}`` when git or the repository is unavailable.
    """
    here = str(Path(path).resolve())

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None, "source": "unavailable"}
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "*.py")
    untracked_py = sorted(line for line in (untracked or "").splitlines() if line.strip())
    dirty = None if tracked is None or untracked is None else bool(tracked) or bool(untracked_py)
    return {"commit": commit, "dirty": dirty, "untracked_python_files": len(untracked_py), "source": "git"}


@functools.lru_cache(maxsize=1)
def git_state() -> dict[str, Any]:
    """:func:`git_state_at` for the checkout this module runs from (computed once).

    ``ASTROSEARCH_GIT_COMMIT`` overrides detection (for deployments without ``.git``).
    """
    override = os.getenv("ASTROSEARCH_GIT_COMMIT")
    if override:
        return {"commit": override.strip(), "dirty": None, "source": "env:ASTROSEARCH_GIT_COMMIT"}
    return git_state_at(Path(__file__).resolve().parent)


# Modules whose code decides what is queried and how rows are matched and hashed.
CORE_MODULES = ("models.py", "providers.py", "crossmatch.py", "provenance.py")


@functools.lru_cache(maxsize=1)
def code_digests() -> dict[str, str | None]:
    """SHA-256 (first 16 hex digits) of the core modules as loaded by this process.

    Identifies the exact code even when the checkout is dirty or has untracked files.
    """
    here = Path(__file__).resolve().parent
    out: dict[str, str | None] = {}
    for name in CORE_MODULES:
        try:
            out[name] = hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
        except OSError:
            out[name] = None
    return out


@functools.lru_cache(maxsize=1)
def _software_info_cached() -> str:
    return json.dumps({
        "name": "astrosearch",
        "version": _package_version("astrosearch") or _pyproject_version() or "unknown",
        "git": git_state(),
        "code_sha256": code_digests(),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "dependencies": {
            name: _package_version(name) for name in ("astropy", "numpy", "httpx", "fastapi", "pydantic")
        },
    })


def software_info() -> dict[str, Any]:
    """Versions of AstroSearch, its git commit and code digests, Python, and dependencies.

    Computed once per process (the running code cannot change underneath it); the first
    call runs ``git`` subprocesses, so async code should use :func:`software_info_async`.
    """
    return json.loads(_software_info_cached())


async def software_info_async() -> dict[str, Any]:
    """:func:`software_info` computed off the event loop (git subprocesses block ~1 s once)."""
    return await asyncio.to_thread(software_info)


# ---------------------------------------------------------------------------
# Scientific Payload (what the content hash covers)
# ---------------------------------------------------------------------------


def record_to_dict(record: UnifiedRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Plain-dict form of a UnifiedRecord (sources as dicts), accepting either form."""
    if isinstance(record, UnifiedRecord):
        data = record.as_dict()
    elif isinstance(record, Mapping):
        data = dict(record)
    else:
        raise ManifestError("record must be a UnifiedRecord or a mapping")
    _check_record_shape(data)
    results = {}
    for name, res in (data.get("catalog_results") or {}).items():
        res = dict(res or {})
        for key in ("sources", "pad_sources"):
            if key in res:
                res[key] = [asdict(s) if is_dataclass(s) and not isinstance(s, type) else dict(s) for s in res[key] or []]
        results[str(name)] = res
    data["catalog_results"] = results
    return data


def _check_record_shape(data: Mapping[str, Any]) -> None:
    """Reject records whose containers have the wrong JSON type (ManifestError).

    Without this, a string where a list belongs is iterated character by character
    (``catalogs_planned: "gaia_dr3"`` became catalogs '3', '_', 'a', ...).
    """
    results = data.get("catalog_results")
    if results is not None and not isinstance(results, Mapping):
        raise ManifestError("record catalog_results must be an object {catalog: result}")
    for name, res in (results or {}).items():
        if res is not None and not isinstance(res, Mapping):
            raise ManifestError(f"record catalog_results[{name!r}] must be an object")
        for key in ("sources", "pad_sources"):
            rows = (res or {}).get(key)
            if rows is None:
                continue
            if not isinstance(rows, list) or not all(isinstance(r, Mapping) or (is_dataclass(r) and not isinstance(r, type))
                                                     for r in rows):
                raise ManifestError(f"record catalog_results[{name!r}].{key} must be a list of row objects")
    prov = data.get("provenance")
    if prov is not None and not isinstance(prov, Mapping):
        raise ManifestError("record provenance must be an object")
    prov = prov or {}
    planned = prov.get("catalogs_planned")
    if planned is not None and (not isinstance(planned, list) or not all(isinstance(c, str) for c in planned)):
        raise ManifestError("record provenance.catalogs_planned must be a list of catalog names")
    for key in ("matches", "warnings"):
        value = prov.get(key)
        if value is not None and not isinstance(value, list):
            raise ManifestError(f"record provenance.{key} must be a list")
    if not all(isinstance(m, Mapping) for m in prov.get("matches") or []):
        raise ManifestError("record provenance.matches must be a list of objects")
    for key in ("catalog_stats", "target_proper_motion", "target_parallax", "advanced_query"):
        value = prov.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ManifestError(f"record provenance.{key} must be an object or null")
    for key in ("failures", "crossmatch_groups"):
        value = data.get(key)
        if value is not None and (not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value)):
            raise ManifestError(f"record {key} must be a list of objects")
    for group in data.get("crossmatch_groups") or []:
        members = group.get("members")
        if members is not None and (not isinstance(members, list) or not all(isinstance(m, Mapping) for m in members)):
            raise ManifestError("record crossmatch_groups[].members must be a list of objects")
    target = data.get("target")
    if target is not None and not isinstance(target, Mapping):
        raise ManifestError("record target must be an object")


def _fold_column_case(data: Mapping[str, Any]) -> dict[str, Any]:
    """Archive columns keyed by lower-case name (unless that would merge two columns).

    Column-name case is presentation, not science: unquoted ADQL/SQL identifiers are
    case-insensitive (IVOA ADQL 2.1, section 2.1.3), and MAST's catalogs API was
    observed (2026-09-28) to name the same Pan-STARRS DR2 column "objName" in one
    answer and "ObjName" in the next, a minute apart. A row whose names collide when
    lower-cased keeps its original keys.
    """
    folded = {str(k).lower(): v for k, v in data.items()}
    return folded if len(folded) == len(data) else {str(k): v for k, v in data.items()}


def _source_science(source: Mapping[str, Any]) -> dict[str, Any]:
    data = source.get("data") or {}
    return canonicalize({
        "source_id": str(source.get("source_id")),
        "ra": source.get("ra"),
        "dec": source.get("dec"),
        "positional_error_arcsec": source.get("positional_error_arcsec"),
        "epoch": source.get("epoch"),
        "pm_ra_masyr": source.get("proper_motion_ra_masyr"),
        "pm_dec_masyr": source.get("proper_motion_dec_masyr"),
        "data": _fold_column_case(data) if isinstance(data, Mapping) else data,
    })


def _hash_form_row(entry: Mapping[str, Any], *, canonical: bool = False) -> dict[str, Any]:
    """A row as hashed: archive columns with big integers read at double precision.

    ``canonical=True``: ``entry`` is already canonical (skip re-canonicalizing its data).
    """
    body = {k: v for k, v in entry.items() if k != "row_hash"}
    if isinstance(body.get("data"), Mapping):
        body["data"] = _js_normalized(body["data"] if canonical else canonicalize(body["data"]))
    return body


def row_hash(entry: Mapping[str, Any]) -> str:
    """Short SHA-256 digest of one canonical source entry (without its own ``row_hash``)."""
    return _row_hash_canonical(canonicalize(entry))


def _row_hash_canonical(entry: Mapping[str, Any]) -> str:
    """:func:`row_hash` of an entry that is already canonical (same digest, one pass)."""
    return _sha256_canonical(_hash_form_row(entry, canonical=True))[:ROW_HASH_HEX_CHARS]


def _source_order(entry: Mapping[str, Any]) -> tuple[str, str]:
    return (str(entry.get("source_id")), canonical_json(entry))


def _sorted_sources[Row: Mapping[str, Any]](entries: Iterable[Row], *, canonical: bool = False) -> list[Row]:
    """Rows in :func:`_source_order` order (source id, then canonical text for repeated ids).

    The canonical text is only computed for ids that occur more than once, so sorting
    costs one pass instead of a canonical serialization of every row.
    """
    by_id: dict[str, list[Row]] = {}
    for entry in entries:
        by_id.setdefault(str(entry.get("source_id")), []).append(entry)
    out: list[Row] = []
    for sid in sorted(by_id):
        group = by_id[sid]
        if len(group) > 1:
            group = sorted(group, key=lambda e: _dumps_canonical(e) if canonical else canonical_json(e))
        out.extend(group)
    return out


def science_payload(record: UnifiedRecord | Mapping[str, Any]) -> dict[str, Any]:
    """The scientific content of a record: exactly what :func:`content_hash` covers.

    Included: target (position, frame, epoch, proper motion, parallax), query and
    effective radius, the target proper-motion and parallax origins (input / resolver /
    adopted from a catalog row -- the parallax removes the annual parallax from
    single-epoch positions such as 2MASS and SDSS, so it shapes separations), per
    catalog the status (``success``/``empty``/``failed`` + error type), row count,
    truncation and epoch-completeness flags, every in-cone row (id, position, 1-sigma
    error, epoch, proper motion, archive columns), the target matches (catalog, id,
    separation, confidence) and crossmatch group memberships.

    Excluded (volatile or presentation): retrieval timestamps, elapsed times, cache
    flags, warnings, error message text, per-row provider metadata/provenance/links,
    rows outside the requested radius (epoch pad rows), and the name-resolver output.
    Rows are sorted by source id and catalogs by name, so archive row order does not
    affect the hash; archive column names are compared case-insensitively
    (:func:`_fold_column_case`).
    """
    rec = record_to_dict(record)
    prov = rec.get("provenance") or {}
    target = rec.get("target") or {}
    catalogs: dict[str, dict[str, Any]] = {}
    for name, res in sorted((rec.get("catalog_results") or {}).items()):
        res = res or {}
        status = str(res.get("status") or ("failed" if res.get("error_type") else "success"))
        entries = _sorted_sources([_source_science(s) for s in res.get("sources") or []], canonical=True)
        for entry in entries:
            entry["row_hash"] = _row_hash_canonical(entry)
        item: dict[str, Any] = {
            "status": status,
            "row_count": int(res.get("row_count", len(entries)) or 0),
            "truncated": bool(res.get("truncated", False)),
            "sources": entries,
        }
        if res.get("epoch_incomplete"):
            item["epoch_incomplete"] = True
        if status == "failed":
            item["error_type"] = res.get("error_type")
        catalogs[name] = item
    for failure in rec.get("failures") or []:
        name = str(failure.get("catalog"))
        if name not in catalogs:
            catalogs[name] = {"status": "failed", "error_type": failure.get("error_type"), "row_count": 0,
                              "truncated": False, "sources": []}
    matches = sorted(
        (canonicalize({
            "catalog": m.get("catalog"),
            "source_id": str(m.get("source_id")),
            "separation_arcsec": m.get("separation_arcsec"),
            "confidence": m.get("confidence"),
        }) for m in prov.get("matches") or []),
        key=lambda m: (m["catalog"], m["source_id"], canonical_json(m)),
    )
    groups = sorted(
        sorted(f"{mem.get('catalog')}:{mem.get('source_id')}" for mem in group.get("members") or [])
        for group in rec.get("crossmatch_groups") or []
    )
    # Row entries are canonical already: set them aside so the payload is not walked twice.
    rows = {name: item.pop("sources") for name, item in catalogs.items()}
    out = canonicalize({
        "schema": SCIENCE_SCHEMA,
        "target": {k: target.get(k) for k in ("ra", "dec", "frame", "epoch", "pm_ra_masyr", "pm_dec_masyr", "parallax_mas")},
        "query_radius_arcsec": prov.get("query_radius_arcsec"),
        "effective_radius_arcsec": prov.get("effective_radius_arcsec"),
        "target_proper_motion": prov.get("target_proper_motion"),
        "target_parallax": prov.get("target_parallax"),
        "catalogs": catalogs,
        "matches": matches,
        "groups": groups,
    })
    for name, entries in rows.items():
        out["catalogs"][str(name)]["sources"] = entries
    return out


def _hash_form(science: Mapping[str, Any], *, canonical: bool = False) -> dict[str, Any]:
    """The payload as hashed. ``canonical=True``: ``science`` is already canonical (as
    built by :func:`science_payload` or parsed by :meth:`ProvenanceManifest.from_dict`),
    so it is not serialized and re-parsed first (the result is the same)."""
    if canonical:
        out = dict(science)
        if isinstance(out.get("catalogs"), Mapping):
            out["catalogs"] = {k: dict(v) if isinstance(v, Mapping) else v for k, v in out["catalogs"].items()}
    else:
        out = json.loads(canonical_json(science))
    for item in (out.get("catalogs") or {}).values():
        if isinstance(item, dict) and isinstance(item.get("sources"), list):
            item["sources"] = [
                {**_hash_form_row(s, canonical=True), "row_hash": s.get("row_hash")} if isinstance(s, Mapping) else s
                for s in item["sources"]
            ]
    return out


def _content_hash_canonical(science: Mapping[str, Any]) -> str:
    """:func:`content_hash` of a science payload that is already canonical."""
    return _prefixed(_sha256_canonical(_hash_form(science, canonical=True)))


def content_hash(record_or_science: UnifiedRecord | Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the scientific payload of a record (or of a payload itself).

    Integers beyond 2**53 inside archive columns are hashed at double precision (see
    :func:`js_number_value`), so a record or manifest that passed through a JavaScript
    client hashes like the original.
    """
    if isinstance(record_or_science, Mapping) and record_or_science.get("schema") == SCIENCE_SCHEMA:
        return _content_hash_canonical(json.loads(canonical_json(record_or_science)))
    return _content_hash_canonical(science_payload(record_or_science))


def compact_science(science: Mapping[str, Any]) -> dict[str, Any]:
    """Science payload with each row reduced to id, position, 1-sigma error and ``row_hash``.

    A compact manifest still supports replay diffs (added/removed/moved/changed rows),
    but changed rows are reported without per-column detail and the content hash can
    no longer be recomputed from the manifest.
    """
    out = json.loads(canonical_json(science))
    for item in out.get("catalogs", {}).values():
        item["sources"] = [
            {"source_id": s["source_id"], "ra": s.get("ra"), "dec": s.get("dec"),
             "positional_error_arcsec": s.get("positional_error_arcsec"), "row_hash": s["row_hash"]}
            for s in item.get("sources", [])
        ]
    out["compact"] = True
    return out


# ---------------------------------------------------------------------------
# Catalog Releases & Exact Requests
# ---------------------------------------------------------------------------

# Data release of each registry catalog. ``living`` = the service is updated in place
# (new rows / revised values appear without a release number), so a replay months
# later can legitimately differ even though nothing in AstroSearch changed.
CATALOG_RELEASES: dict[str, dict[str, Any]] = {
    "gaia_dr3": {"release": "Gaia DR3 (ESA Gaia Archive table gaiadr3.gaia_source)", "living": False},
    "simbad": {"release": "SIMBAD (CDS; continuously updated database)", "living": True},
    "ned": {"release": "NED (IPAC; continuously updated database)", "living": True},
    "exoplanet_archive": {"release": "NASA Exoplanet Archive pscomppars (updated as planets are added/revised)", "living": True},
    "vizier_2mass_reference": {"release": "2MASS All-Sky Point Source Catalog (VizieR II/246)", "living": False},
    "twomass_psc": {"release": "2MASS All-Sky Point Source Catalog (IRSA table fp_psc)", "living": False},
    "allwise": {"release": "AllWISE Source Catalog (IRSA table allwise_p3as_psd)", "living": False},
    "panstarrs_dr2": {"release": "Pan-STARRS1 DR2 mean-object table (MAST catalogs API)", "living": False},
    "sdss": {"release": "SDSS DR18 (SkyServer dr18 context)", "living": False},
    "first": {"release": "FIRST catalogue (HEASARC table first)", "living": False},
    "nvss": {"release": "NVSS catalogue (HEASARC table nvss)", "living": False},
    "vlass": {"release": "VLASS Epoch 1 components of Bruzewski et al. 2021 (VizieR J/ApJ/914/42)", "living": False},
    "lotss": {"release": "LoTSS-DR3 (VizieR J/A+A/707/A198)", "living": False},
    "lotss_dr2": {"release": "LoTSS-DR2 (VizieR J/A+A/659/A1)", "living": False},
    "rosat": {"release": "2RXS (HEASARC table rass2rxs)", "living": False},
    "rosat_bsc": {"release": "1RXS Bright Source Catalogue (HEASARC table rassbsc)", "living": False},
    # HEASARC updates csc (v2.1 -> v2.1.1) and replaces xmmssc with each new XMM-Newton
    # serendipitous catalogue (4XMM-DR14 -> 5XMM-DR15 in June 2026): the live table
    # description is recorded in the manifest (see live_releases) when it can be read.
    "chandra": {"release": "Chandra Source Catalog 2.x master sources (HEASARC table csc; tracks the latest 2.x version)",
                "living": True},
    "xmm": {"release": "XMM-Newton serendipitous source catalogue (HEASARC table xmmssc; tracks the latest DR)", "living": True},
}


@functools.cache
def _default_identity(name: str) -> tuple[Any, ...] | None:
    """(provider, endpoint, table, catalog) of the built-in definition of ``name``."""
    from models import DEFAULT_CATALOGS, catalog_from_dict

    entry = DEFAULT_CATALOGS.get(name)
    if entry is None:
        return None
    d = catalog_from_dict(name, entry)
    return (d.provider, d.endpoint or None, d.table or None, d.catalog or None)


def matches_curated(name: str, definition: CatalogDefinition | None) -> bool:
    """True when ``definition`` is the service/table the curated release and citation
    entries of ``name`` were verified for (the built-in registry definition).

    A deployment registry (``CATALOG_REGISTRY_PATH``) may point a known name at another
    endpoint or table (e.g. ``sdss`` at a ``dr19`` SkyServer); the DR18 label and
    references must not be attached to that. No definition: nothing contradicts them.
    """
    if definition is None:
        return True
    identity = _default_identity(name)
    return identity is not None and identity == (definition.provider, definition.endpoint or None,
                                                 definition.table or None, definition.catalog or None)


def catalog_release(name: str, definition: CatalogDefinition | None = None) -> dict[str, Any]:
    """Release label and 'living database' flag of a catalog.

    The curated label (:data:`CATALOG_RELEASES`) is used only when ``definition`` is
    the built-in service/table (:func:`matches_curated`); otherwise the registry's own
    text is returned with ``verified: False``.
    """
    if name in CATALOG_RELEASES and matches_curated(name, definition):
        return {**CATALOG_RELEASES[name], "verified": True}
    label = None
    if definition is not None:
        label = definition.description or definition.table or definition.catalog
        where = " ".join(str(x) for x in (definition.endpoint, definition.table or definition.catalog) if x)
        if where:
            label = f"{label} [{where}]" if label else where
    note = None
    if name in CATALOG_RELEASES:
        note = (f"registry definition differs from the curated one ({CATALOG_RELEASES[name]['release']!r}): "
                "release not verified")
    return {"release": label or "unknown", "living": None, "verified": False, "note": note}


# Live table descriptions: {(endpoint, table): (monotonic expiry, description, UTC time it was read)}.
_RELEASE_CACHE: dict[tuple[str, str], tuple[float, str, str]] = {}
RELEASE_CACHE_SECONDS = 3600.0
RELEASE_LOOKUP_TIMEOUT = 15.0


def _heasarc_tables(registry: CatalogRegistry, names: Iterable[str]) -> dict[str, tuple[str, str]]:
    """{catalog: (TAP endpoint, table)} for manifest catalogs served by HEASARC TAP."""
    out: dict[str, tuple[str, str]] = {}
    for name in names:
        definition = registry.catalogs.get(name)
        if definition is None or definition.provider != "tap" or not definition.table or not definition.endpoint:
            continue
        if httpx.URL(definition.endpoint).host in HEASARC_HOSTS and re.fullmatch(r"[A-Za-z0-9_]+", definition.table):
            out[name] = (definition.endpoint, definition.table)
    return out


def _parse_tap_schema_votable(content: bytes) -> dict[str, str]:
    import io

    from astropy.io.votable import parse

    table = parse(io.BytesIO(content)).get_first_table().to_table()
    out: dict[str, str] = {}
    for row in table:
        name, description = row["table_name"], row["description"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        description = description.decode() if isinstance(description, bytes) else str(description)
        # Tables named by an ADQL reserved word are listed quoted (HEASARC: '"first"').
        out[name.strip().strip('"')] = " ".join(description.split())
    return out


async def live_releases(
    client: httpx.AsyncClient | None, registry: CatalogRegistry, names: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Data-release descriptions read live from the archives, for the given catalogs.

    HEASARC replaces or re-versions tables in place (``xmmssc`` is "XMM-Newton
    Serendipitous Source Catalog: 5XMM-DR15 Version" since June 2026; ``csc`` is "Chandra
    Source Catalog, v2.1.1"), so the description served in ``TAP_SCHEMA.tables`` (IVOA
    TAP 1.1 section 4) at query time is what identifies the release actually queried.
    VizieR tables are frozen per catalogue and the other services name their release in
    the endpoint or table, so only HEASARC TAP tables are looked up. Answers are cached
    for :data:`RELEASE_CACHE_SECONDS`. Returns ``{catalog: {"release", "source",
    "looked_up_at"}}`` (``looked_up_at``: UTC ISO time the description was read; a
    manifest uses it only for a query made within :data:`RELEASE_TIME_WINDOW_SECONDS`);
    any lookup failure returns ``{catalog: {"source": "unavailable: <reason>"}}`` for the
    affected catalogs, and the manifest keeps the hand-maintained label.
    """
    import time

    wanted = _heasarc_tables(registry, names)
    out: dict[str, dict[str, str]] = {}
    by_endpoint: dict[str, list[str]] = {}
    now = time.monotonic()
    for name, (endpoint, table) in wanted.items():
        cached = _RELEASE_CACHE.get((endpoint, table))
        if cached and cached[0] > now:
            out[name] = {"release": f"{cached[1]} (HEASARC table {table})",
                         "source": f"live: {endpoint} TAP_SCHEMA.tables ({table})", "looked_up_at": cached[2]}
        else:
            by_endpoint.setdefault(endpoint, []).append(name)
    if not by_endpoint:
        return out
    own = client is None
    http = client or httpx.AsyncClient(timeout=RELEASE_LOOKUP_TIMEOUT, follow_redirects=True)
    try:
        for endpoint, cats in by_endpoint.items():
            tables = sorted({wanted[c][1] for c in cats})
            names_sql = [f"'{t}'" for t in tables] + [f"'\"{t}\"'" for t in tables]  # plain and quoted names
            adql = ("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name IN ("
                    + ", ".join(names_sql) + ")")
            try:
                response = await http.post(endpoint, data={"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql},
                                           timeout=RELEASE_LOOKUP_TIMEOUT)
                response.raise_for_status()
                described = _parse_tap_schema_votable(response.content)
            except Exception as exc:  # noqa: BLE001 - metadata only: never fail a manifest over it
                for c in cats:
                    out[c] = {"source": f"unavailable: {exc.__class__.__name__}"}
                continue
            read_at = datetime.now(UTC).isoformat()
            for c in cats:
                table = wanted[c][1]
                if described.get(table):
                    _RELEASE_CACHE[(endpoint, table)] = (now + RELEASE_CACHE_SECONDS, described[table], read_at)
                    out[c] = {"release": f"{described[table]} (HEASARC table {table})",
                              "source": f"live: {endpoint} TAP_SCHEMA.tables ({table})", "looked_up_at": read_at}
                else:
                    out[c] = {"source": f"unavailable: {table} not listed in TAP_SCHEMA.tables"}
    finally:
        if own:
            await http.aclose()
    return out


def _pure_builder(cls: type[Any]) -> Any:
    """Provider instance used only for its pure request builders (no HTTP client).

    ``build_adql``/``build_request``/``build_params``/``build_sql`` read nothing but the
    catalog definition, the target and class attributes, so the transport set up by
    ``__init__`` is not needed (and creating one would leak an unclosed client).
    """
    return object.__new__(cls)


def reconstruct_request(catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> dict[str, Any]:
    """The exact request the provider for ``catalog`` sends for ``target``/``radius``.

    Uses the providers' own request builders and :func:`models.plan_cone` (epoch-aware
    cone), so it equals what :mod:`providers` sends; the tests check this against the
    parameters recorded on real result rows.
    """
    cone = plan_cone(catalog, target, radius_arcsec)
    provider = catalog.provider
    if provider == "tap":
        builder = _pure_builder(TapProvider)
        adql = builder.build_adql(catalog, target, radius_arcsec, cone)
        method = str(catalog.parameters.get("http_method", "POST")).upper()
        return {"provider": provider, "endpoint": catalog.endpoint, "http_method": method, "query": adql,
                "parameters": builder.build_request(catalog, adql)}
    if provider in {"irsa_gator", "mast", "heasarc_xamin"}:
        cls = {"irsa_gator": IRSAGatorProvider, "mast": MASTProvider, "heasarc_xamin": HEASARCXaminProvider}[provider]
        params = _pure_builder(cls).build_params(catalog, target, radius_arcsec, cone)
        return {"provider": provider, "endpoint": catalog.endpoint, "http_method": "GET", "query": None, "parameters": params}
    if provider == "sdss":
        endpoint = catalog.endpoint or ""
        mode = str(catalog.parameters.get("mode") or ("conesearch" if "ConeSearch" in endpoint else "sqlsearch"))
        if mode == "conesearch":
            params = {"format": "csv", "ra": f"{cone.ra:.9f}", "dec": f"{cone.dec:.9f}", "sr": f"{cone.radius_arcsec / 60.0:.10f}"}
            return {"provider": provider, "endpoint": catalog.endpoint, "http_method": "GET", "query": None, "parameters": params}
        sql = _pure_builder(SDSSProvider).build_sql(catalog, target, radius_arcsec, cone)
        return {"provider": provider, "endpoint": catalog.endpoint, "http_method": "GET", "query": sql,
                "parameters": {"cmd": sql, "format": "json"}}
    raise ValueError(f"cannot reconstruct requests for provider {provider!r}")


def _fallback_definition(catalog: CatalogDefinition, fallback: Mapping[str, Any]) -> CatalogDefinition:
    """The definition QueryExecutor uses when a catalog's declared fallback ran."""
    raw = catalog.parameters.get("fallback")
    spec: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return replace(
        catalog,
        provider=str(fallback.get("provider") or spec.get("provider") or catalog.provider),
        endpoint=fallback.get("endpoint") or spec.get("endpoint") or catalog.endpoint,
        catalog=spec.get("catalog", catalog.catalog),
        table=spec.get("table", catalog.table),
        parameters={**catalog.parameters, **(spec.get("parameters") or {}), "fallback": None},
    )


@dataclass(slots=True)
class CatalogRequest:
    """What was sent to one catalog and what it returned (manifest entry)."""

    catalog: str
    provider: str | None
    endpoint: str | None
    http_method: str | None
    query: str | None
    parameters: dict[str, Any] | None
    request_source: str  # recorded | reconstructed | unavailable
    request_verified: bool | None
    release: str
    living_database: bool | None
    status: str
    row_count: int
    raw_row_count: int | None = None
    truncated: bool = False
    query_radius_arcsec: float | None = None
    fallback: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    citation: str | None = None
    retrieved_at: str | None = None
    elapsed_ms: float | None = None
    release_source: str = "label"  # label | live: <endpoint> TAP_SCHEMA.tables (<table>) | unavailable: <why>
    citation_source: str | None = None  # curated (verified) | registry (unverified)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Expected JSON types of each field (None always allowed except for the required ones).
    _FIELD_TYPES: ClassVar[tuple[tuple[str, tuple[type, ...]], ...]] = (
        ("catalog", (str,)), ("provider", (str,)), ("endpoint", (str,)), ("http_method", (str,)), ("query", (str,)),
        ("parameters", (Mapping,)), ("request_source", (str,)), ("request_verified", (bool,)), ("release", (str,)),
        ("living_database", (bool,)), ("status", (str,)), ("row_count", (int,)), ("raw_row_count", (int,)),
        ("truncated", (bool,)), ("query_radius_arcsec", (int, float)), ("fallback", (Mapping,)),
        ("error_type", (str,)), ("error_message", (str,)), ("citation", (str,)), ("retrieved_at", (str,)),
        ("elapsed_ms", (int, float)), ("release_source", (str,)), ("citation_source", (str,)),
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CatalogRequest:
        if not isinstance(data, Mapping):
            raise ManifestError("catalog request entry must be an object")
        missing = {"catalog", "status"} - set(data)
        if missing:
            raise ManifestError(f"catalog request entry is missing {sorted(missing)}")
        for name, types in cls._FIELD_TYPES:
            value = data.get(name)
            if value is None:
                if name in ("catalog", "status"):
                    raise ManifestError(f"catalog request entry field {name!r} must not be null")
                continue
            if not isinstance(value, types) or (isinstance(value, bool) and bool not in types):
                raise ManifestError(f"catalog request {data.get('catalog')!r}: field {name!r} has type "
                                    f"{type(value).__name__}")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Fields with a non-null default keep it when the manifest gives null.
        defaults = {"request_source": "unavailable", "release": "unknown", "row_count": 0, "truncated": False,
                    "release_source": "label"}
        values: dict[str, Any] = {
            "provider": None, "endpoint": None, "http_method": None, "query": None, "parameters": None,
            "request_verified": None, "living_database": None, **defaults,
        }
        for key, value in data.items():
            if key in known and not (value is None and key in defaults):
                values[key] = dict(value) if key in ("parameters", "fallback") and value is not None else value
        return cls(**values)

    def request_signature(self) -> dict[str, Any]:
        """The part of the entry that defines the request (hashed into ``query_hash``)."""
        return {"endpoint": self.endpoint, "http_method": self.http_method, "query": self.query, "parameters": self.parameters}


def _recorded_request(result: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """(parameters, provider, retrieved_at) recorded on the rows of one catalog result."""
    rows = list(result.get("sources") or []) + list(result.get("pad_sources") or [])
    stamps = sorted(str(r.get("provenance", {}).get("retrieved_at")) for r in rows if r.get("provenance", {}).get("retrieved_at"))
    for row in rows:
        prov = row.get("provenance") or {}
        if isinstance(prov.get("query_parameters"), Mapping):
            return dict(prov["query_parameters"]), prov.get("provider"), stamps[0] if stamps else None
    return None, None, stamps[0] if stamps else None


def _accepts(function: Any, parameter: str) -> bool:
    """True when ``function`` takes a keyword ``parameter`` (forward compatibility with
    core modules that gain options, e.g. a target parallax)."""
    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _target_kwargs(t: Mapping[str, Any], *, with_motion: bool, with_parallax: bool | None = None) -> dict[str, Any]:
    """validate_target keywords for a target dict: epoch/motion/parallax as recorded.

    ``with_parallax`` defaults to ``with_motion``.
    """
    kwargs: dict[str, Any] = {"frame": t.get("frame") or "icrs", "epoch": t.get("epoch") if with_motion else None,
                              "pm_ra_masyr": t.get("pm_ra_masyr") if with_motion else None,
                              "pm_dec_masyr": t.get("pm_dec_masyr") if with_motion else None}
    keep_parallax = with_motion if with_parallax is None else with_parallax
    if keep_parallax and t.get("parallax_mas") is not None and _accepts(validate_target, "parallax_mas"):
        kwargs["parallax_mas"] = t["parallax_mas"]
    return kwargs


def given_pm_source(record: Mapping[str, Any]) -> str | None:
    """Origin of a proper motion that was *given* with the query ("input", "resolver", ...).

    None when the target had no proper motion or the crossmatch derived it from a
    catalog row (:data:`DERIVED_PM_ORIGINS`: "adopted" from e.g. Gaia, or
    "extragalactic" = stationary because the identified object is extragalactic).
    """
    pm_info = (record.get("provenance") or {}).get("target_proper_motion")
    if not isinstance(pm_info, Mapping):
        return None
    source = pm_info.get("source") or "input"
    return None if source in DERIVED_PM_ORIGINS else str(source)


def given_parallax_source(record: Mapping[str, Any]) -> str | None:
    """Origin of a target parallax that was *given* with the query ("input", "resolver").

    ``crossmatch`` records the parallax it used in ``provenance.target_parallax`` with
    its origin: "input" (caller), "resolver" (Sesame, with the resolver proper motion)
    or "adopted" (taken from a catalog row by ``_adopt_parallax`` -- part of the answer,
    re-adopted on replay, so SIMBAD/Gaia revisions reach the replayed separations).
    None when no parallax was given. A record without the ``target_parallax`` key
    (written before the crossmatch recorded parallax origins) is read as: a
    ``target.parallax_mas`` was given.
    """
    prov = record.get("provenance") or {}
    info = prov.get("target_parallax") if isinstance(prov, Mapping) else None
    if isinstance(info, Mapping):
        if info.get("parallax_mas") is None:
            return None
        source = info.get("source") or "input"
        return None if source in DERIVED_PARALLAX_ORIGINS else str(source)
    if isinstance(prov, Mapping) and "target_parallax" in prov:
        return None  # recorded explicitly as "no parallax"
    target = record.get("target") or {}
    return "input" if isinstance(target, Mapping) and target.get("parallax_mas") is not None else None


def input_target(record: Mapping[str, Any]) -> Target:
    """The target as it was *given* to the crossmatch (before any adoption).

    A proper motion or parallax supplied with the query -- by the caller ("input") or by
    the name resolver ("resolver", e.g. ``main.search_object`` / ``POST /api/v1/search``
    with a name) -- is part of the question and is kept; one the crossmatch adopted
    from a catalog row is part of the answer and is dropped (a replay adopts it again).
    """
    prov = record.get("provenance") or {}
    adv = prov.get("advanced_query")
    if isinstance(adv, Mapping) and adv.get("target"):
        # query.to_dict() keeps the query's own target: adoption never writes into it.
        t = adv["target"]
        # CrossmatchService drops epoch, proper motion and parallax when proper_motion is False.
        use_pm = adv.get("proper_motion", True) is not False
        return validate_target(t["ra"], t["dec"], **_target_kwargs(t, with_motion=use_pm))
    target = record.get("target") or {}
    kwargs = _target_kwargs(target, with_motion=given_pm_source(record) is not None,
                            with_parallax=given_parallax_source(record) is not None)
    kwargs["epoch"] = target.get("epoch")  # the epoch of (ra, dec) is always part of the question
    return validate_target(target["ra"], target["dec"], **kwargs)


def curated_citation(name: str, definition: CatalogDefinition | None = None) -> str | None:
    """One-line citation of a catalog from the verified references (:data:`CITATION_ENTRIES`).

    Preferred over the registry's free-text ``citation`` (models.DEFAULT_CATALOGS),
    which is not verified: e.g. its Chandra entry names DOI 10.25574/csc2.1, which
    doi.org does not know (HTTP 404, checked 2026-09-28; the CSC Release 2 series DOI
    is 10.25574/csc2). None when ``definition`` is not the service/table the entry was
    verified for (:func:`matches_curated`).
    """
    entry = CITATION_ENTRIES.get(name) or CITATION_ENTRIES.get(CITATION_ALIASES.get(name.lower(), ""))
    if entry is None or entry.kind != "catalog" or not entry.references:
        return None
    if name in CATALOG_RELEASES and not matches_curated(name, definition):
        return None
    parts = []
    for key in entry.references:
        ref = REFERENCES[key]
        parts.append(f"{ref.short} ({ref.bibcode})" if ref.bibcode else f"{ref.short}, doi:{ref.doi}")
    return "; ".join(parts)


# A live release description is attached to a manifest only when it was read within
# this time of the query (the record's earliest row retrieval, or the search the
# manifest ran): HEASARC re-versions tables in place (xmmssc: 4XMM-DR14 -> 5XMM-DR15 in
# June 2026), so a description read days after a stored record was made may name a
# release that record never saw.
RELEASE_TIME_WINDOW_SECONDS = 3600.0
UNVERIFIED_RELEASE_PREFIX = "label (not verified for the query time"


def _parse_time(value: Any) -> datetime | None:
    """A timezone-aware datetime from an ISO-8601 string or datetime (None if unusable)."""
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value.strip():
        try:
            stamp = datetime.fromisoformat(value.strip())  # Python 3.11+ also reads a trailing 'Z'
        except ValueError:
            return None
    else:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def record_query_time(record: Mapping[str, Any]) -> datetime | None:
    """When the archives answered a record: the earliest ``retrieved_at`` of its rows.

    None when no catalog returned a row (empty and failed results carry no timestamp).
    """
    stamps: list[datetime] = []
    results = record.get("catalog_results")
    for res in (results.values() if isinstance(results, Mapping) else []):
        if not isinstance(res, Mapping):
            continue
        for row in [*(res.get("sources") or []), *(res.get("pad_sources") or [])]:
            row = asdict(row) if is_dataclass(row) and not isinstance(row, type) else row
            prov = row.get("provenance") if isinstance(row, Mapping) else None
            stamp = _parse_time(prov.get("retrieved_at")) if isinstance(prov, Mapping) else None
            if stamp is not None:
                stamps.append(stamp)
    return min(stamps) if stamps else None


def _served_by_fallback(fallback: Any, status: str | None) -> bool:
    """The catalog's rows came from its declared fallback archive (the primary failed).

    A fallback record with an ``error`` means the fallback failed as well (the catalog
    failed), so nothing was served by it.
    """
    return isinstance(fallback, Mapping) and status != "failed" and not fallback.get("error")


def _recorded_form(effective: CatalogDefinition, query_text: str) -> dict[str, Any] | None:
    """The request form the provider sends for a recorded ADQL/SQL text (TAP, SDSS SQL)."""
    if effective.provider == "tap":
        return _pure_builder(TapProvider).build_request(effective, query_text)
    if effective.provider == "sdss":
        return {"cmd": query_text, "format": "json"}
    return None


def _catalog_requests(
    record: Mapping[str, Any], registry: CatalogRegistry | None, target: Target, radius: float,
    releases: Mapping[str, Mapping[str, str]] | None = None, query_time: datetime | None = None,
) -> dict[str, CatalogRequest]:
    prov = record.get("provenance") or {}
    stats = prov.get("catalog_stats") or {}
    results = record.get("catalog_results") or {}
    names = list(dict.fromkeys([*(prov.get("catalogs_planned") or []), *results]))
    failures = {f.get("catalog"): f for f in record.get("failures") or []}
    out: dict[str, CatalogRequest] = {}
    for name in sorted(names):
        res = results.get(name) or {}
        st = stats.get(name) or {}
        status = str(res.get("status") or st.get("status") or ("failed" if name in failures else "unknown"))
        definition = registry.catalogs.get(name) if registry is not None else None
        fallback_raw = res.get("fallback") or st.get("fallback") or (failures.get(name) or {}).get("fallback")
        fallback = dict(fallback_raw) if isinstance(fallback_raw, Mapping) else None
        served_by_fallback = _served_by_fallback(fallback, status)
        params, provider, retrieved_at = _recorded_request(res)
        endpoint = res.get("endpoint")
        query_text = res.get("query")
        method = None
        source = "recorded" if params is not None else "unavailable"
        rebuilt: dict[str, Any] | None = None
        effective: CatalogDefinition | None = None
        if definition is not None:
            effective = _fallback_definition(definition, fallback) if served_by_fallback and fallback else definition
            try:
                rebuilt = reconstruct_request(effective, target, radius)
            except (ValueError, KeyError, TypeError, InvalidCoordinateError):
                rebuilt = None
            if fallback is not None and effective is not definition and "primary_request" not in fallback:
                # The request the primary archive was sent (and failed on) before the fallback ran.
                try:
                    primary = reconstruct_request(definition, target, radius)
                    fallback["primary_request"] = {k: primary[k] for k in ("provider", "endpoint", "http_method",
                                                                           "query", "parameters")}
                except (ValueError, KeyError, TypeError, InvalidCoordinateError):
                    pass
        verified: bool | None = None
        if rebuilt is not None:
            if params is None:
                recorded_form = _recorded_form(effective, query_text) if (effective is not None and query_text) else None
                if recorded_form is not None:
                    # No row recorded the request, but the result metadata holds the exact
                    # ADQL/SQL that was sent: that, not today's registry, is the request.
                    params, source = recorded_form, "recorded (result metadata)"
                    verified = (query_text == rebuilt["query"]
                                and (endpoint is None or endpoint == rebuilt["endpoint"]))
                else:
                    params, source = rebuilt["parameters"], "reconstructed"
                    query_text = query_text or rebuilt["query"]
                    if endpoint is not None and endpoint != rebuilt["endpoint"]:
                        verified = False  # the recorded endpoint contradicts the registry
                endpoint = endpoint or rebuilt["endpoint"]
            else:
                verified = canonicalize(params) == canonicalize(rebuilt["parameters"])
            provider = provider or rebuilt["provider"]
            method = rebuilt["http_method"]
        if endpoint is None and definition is not None:
            endpoint = definition.endpoint
        release = catalog_release(name, definition)
        live = (releases or {}).get(name) or {}
        release_source = "label" if release.get("verified", True) else f"registry ({release.get('note') or 'unverified'})"
        if live.get("release") and release.get("verified", True):
            looked_up = _parse_time(live.get("looked_up_at")) or datetime.now(UTC)
            gap = abs((looked_up - query_time).total_seconds()) if query_time is not None else None
            if gap is not None and gap <= RELEASE_TIME_WINDOW_SECONDS:
                release["release"], release_source = live["release"], live.get("source") or "live"
            else:
                when = query_time.isoformat() if query_time is not None else "at an unknown time (no row timestamps)"
                release_source = (f"{UNVERIFIED_RELEASE_PREFIX}: the archive described the table as "
                                  f"{live['release']!r} at {looked_up.isoformat()}, but the record was retrieved {when})")
        elif live.get("source") and not live.get("release"):
            release_source = f"label ({live['source']})"
        failure = failures.get(name) or {}
        curated = curated_citation(name, definition)
        registry_citation = res.get("citation") or (definition.citation if definition is not None else None)
        out[name] = CatalogRequest(
            catalog=name,
            provider=provider or (effective.provider if effective is not None else None),
            endpoint=endpoint,
            http_method=method,
            query=query_text,
            parameters=canonicalize(params) if params is not None else None,
            request_source=source,
            request_verified=verified,
            release=release["release"],
            living_database=release["living"],
            status=status,
            row_count=int(res.get("row_count", st.get("row_count", 0)) or 0),
            raw_row_count=st.get("raw_row_count", res.get("raw_row_count")),
            truncated=bool(res.get("truncated", st.get("truncated", False))),
            query_radius_arcsec=st.get("query_radius_arcsec", res.get("query_radius_arcsec")),
            fallback=canonicalize(fallback) if fallback is not None else None,
            error_type=res.get("error_type") or failure.get("error_type"),
            error_message=res.get("message") or failure.get("message"),
            citation=curated or registry_citation,
            citation_source=("curated (bibcodes checked against NASA ADS, DOIs against doi.org)" if curated
                             else "registry (unverified)" if registry_citation else None),
            retrieved_at=retrieved_at,
            elapsed_ms=res.get("elapsed_ms", st.get("elapsed_ms")),
            release_source=release_source,
        )
    return out


# ---------------------------------------------------------------------------
# Provenance Manifest
# ---------------------------------------------------------------------------


def _optional_number(section: str, data: Mapping[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ManifestError(f"manifest {section}.{key} must be a finite number or null")


def _validate_input_target(t: Mapping[str, Any]) -> None:
    for key in ("ra", "dec"):
        if t.get(key) is None:
            raise ManifestError(f"manifest input_target lacks a numeric {key!r}")
        _optional_number("input_target", t, key)
    for key in ("epoch", "pm_ra_masyr", "pm_dec_masyr", "parallax_mas"):
        _optional_number("input_target", t, key)
    if t.get("frame") is not None and not isinstance(t["frame"], str):
        raise ManifestError("manifest input_target.frame must be a string")
    # Manifests store normalized positions; anything else was edited by hand.
    if not 0.0 <= float(t["ra"]) < 360.0 or not -90.0 <= float(t["dec"]) <= 90.0:
        raise ManifestError("manifest input_target must have 0 <= ra < 360 and -90 <= dec <= 90 (degrees)")
    try:
        validate_target(t["ra"], t["dec"], **_target_kwargs(t, with_motion=True))
    except (InvalidCoordinateError, ValueError, TypeError) as exc:
        raise ManifestError(f"manifest input_target is invalid: {exc}") from exc


def _validate_query(query: Mapping[str, Any]) -> None:
    mode = query.get("mode", "basic")
    if mode not in ("basic", "advanced"):
        raise ManifestError(f"manifest query.mode must be 'basic' or 'advanced', not {mode!r}")
    for key in ("profile", "resolver", "pm_source", "parallax_source"):
        if query.get(key) is not None and not isinstance(query[key], str):
            raise ManifestError(f"manifest query.{key} must be a string or null")
    planned = query.get("catalogs_planned")
    if planned is not None and (not isinstance(planned, list) or not all(isinstance(c, str) for c in planned)):
        raise ManifestError("manifest query.catalogs_planned must be a list of catalog names")
    for key in ("target_sigma_arcsec", "target_pm_error_masyr"):
        _optional_number("query", query, key)
        if query.get(key) is not None and float(query[key]) < 0:
            raise ManifestError(f"manifest query.{key} must be >= 0")
    adv = query.get("advanced_query")
    if mode == "advanced" or adv is not None:
        if not isinstance(adv, Mapping):
            raise ManifestError("manifest query.advanced_query must be an object for an advanced query")
        parse_advanced_query(adv)


# AdvancedQuery fields and the JSON type each must have (from_dict copies them unchecked;
# a wrong type would surface as a TypeError in the middle of a replay, after archives
# were queried).
_ADV_STRING_LISTS = ("profiles", "object_types", "spectral_types", "morphology", "catalogs")
_ADV_OBJECTS = ("target", "filters", "spatial_constraints", "time_period", "metadata")


def parse_advanced_query(adv: Mapping[str, Any], registry: CatalogRegistry | None = None) -> AdvancedQuery:
    """Parse and validate a manifest's ``advanced_query`` (ManifestError on any problem).

    Checks the JSON type of every list/object field, then runs
    :meth:`crossmatch.QueryValidator.validate` -- against ``registry`` when given (the
    replay's pinned registry: catalog and profile names), otherwise without the
    registry-dependent catalog/profile checks.
    """
    if not isinstance(adv, Mapping):
        raise ManifestError("manifest advanced_query must be an object")
    for key in _ADV_STRING_LISTS:
        value = adv.get(key)
        if value is not None and (not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            raise ManifestError(f"manifest advanced_query.{key} must be a list of strings or null")
    for key in _ADV_OBJECTS:
        value = adv.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ManifestError(f"manifest advanced_query.{key} must be an object or null")
    spatial = adv.get("spatial_constraints") or {}
    polygons = spatial.get("exclude_polygons")
    if polygons is not None and not isinstance(polygons, list):
        raise ManifestError("manifest advanced_query.spatial_constraints.exclude_polygons must be a list")
    zones = spatial.get("radius_zones")
    if zones is not None and (not isinstance(zones, list) or not all(isinstance(z, Mapping) for z in zones)):
        raise ManifestError("manifest advanced_query.spatial_constraints.radius_zones must be a list of objects")
    pm_source = (adv.get("metadata") or {}).get("pm_source")
    if pm_source is not None and not isinstance(pm_source, str):
        raise ManifestError("manifest advanced_query.metadata.pm_source must be a string")
    try:
        query = AdvancedQuery.from_dict(dict(adv))
        if registry is None:
            QueryValidator.validate(replace(query, catalogs=None, profiles=None))
        else:
            QueryValidator.validate(query, registry)
    except (ValueError, TypeError, KeyError, AttributeError, InvalidCoordinateError) as exc:
        raise ManifestError(f"manifest advanced_query is invalid: {exc}") from exc
    return query


def _validate_science(science: Mapping[str, Any]) -> None:
    cats = science.get("catalogs")
    if not isinstance(cats, Mapping):
        raise ManifestError("manifest science.catalogs must be an object")
    for name, item in cats.items():
        if not isinstance(item, Mapping):
            raise ManifestError(f"manifest science.catalogs[{name!r}] must be an object")
        sources = item.get("sources", [])
        if not isinstance(sources, list):
            raise ManifestError(f"manifest science.catalogs[{name!r}].sources must be an array")
        for src in sources:
            if not isinstance(src, Mapping) or "source_id" not in src:
                raise ManifestError(f"manifest science.catalogs[{name!r}].sources holds a non-row entry")
            for key in ("ra", "dec", "positional_error_arcsec"):
                value = src.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, str))):
                    raise ManifestError(f"manifest science row {name}:{src.get('source_id')} has a bad {key!r}")
            if src.get("data") is not None and not isinstance(src["data"], Mapping):
                raise ManifestError(f"manifest science row {name}:{src.get('source_id')} data must be an object")
        for key in ("status", "error_type"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise ManifestError(f"manifest science.catalogs[{name!r}].{key} must be a string")
        count = item.get("row_count")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
            raise ManifestError(f"manifest science.catalogs[{name!r}].row_count must be an integer")
    matches = science.get("matches", [])
    if not isinstance(matches, list) or not all(isinstance(m, Mapping) for m in matches):
        raise ManifestError("manifest science.matches must be an array of objects")
    groups = science.get("groups", [])
    if not isinstance(groups, list) or not all(isinstance(g, list) and all(isinstance(x, str) for x in g) for g in groups):
        raise ManifestError("manifest science.groups must be an array of arrays of 'catalog:source_id' strings")
    for key in ("target", "target_proper_motion", "target_parallax"):
        if science.get(key) is not None and not isinstance(science[key], Mapping):
            raise ManifestError(f"manifest science.{key} must be an object or null")


@dataclass(slots=True)
class ProvenanceManifest:
    """Everything needed to reproduce, verify and cite one crossmatch result."""

    created_at: str
    software: dict[str, Any]
    target: dict[str, Any]
    input_target: dict[str, Any]
    radius_arcsec: float
    query: dict[str, Any]
    catalogs: dict[str, CatalogRequest]
    query_hash: str
    content_hash: str
    science: dict[str, Any]
    resolved_object: dict[str, Any] | None = None
    schema: str = MANIFEST_SCHEMA
    hash_method: str = HASH_METHOD
    # When the archives answered (``created_at`` is when the manifest was built, which
    # for a stored record can be much later) and how that was determined.
    query_time: str | None = None
    query_time_source: str | None = None

    @property
    def row_counts(self) -> dict[str, int]:
        return {name: req.row_count for name, req in self.catalogs.items()}

    @property
    def compact(self) -> bool:
        return bool(self.science.get("compact"))

    def as_dict(self, *, include_science: bool = True) -> dict[str, Any]:
        """JSON form. The input target, radius, query options and target are kept at full
        float precision (a replay must ask exactly the original question); everything
        else is canonical (rounded floats, JavaScript-safe integers)."""
        out = {
            "schema": self.schema,
            "created_at": self.created_at,
            "query_time": self.query_time,
            "query_time_source": self.query_time_source,
            "software": canonicalize(self.software),
            "target": exact(self.target),
            "input_target": exact(self.input_target),
            "radius_arcsec": exact(self.radius_arcsec),
            "resolved_object": exact(self.resolved_object),
            "query": exact(self.query),
            "catalogs": canonicalize({name: req.as_dict() for name, req in sorted(self.catalogs.items())}),
            "row_counts": dict(sorted(self.row_counts.items())),
            "query_hash": self.query_hash,
            "content_hash": self.content_hash,
            "hash_method": self.hash_method,
        }
        if include_science:
            out["science"] = canonicalize(self.science)
        return out

    def to_json(self, *, indent: int | None = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=ensure_ascii, sort_keys=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenanceManifest:
        """Parse and fully validate a manifest; every structural problem is a ManifestError.

        Validation happens before any use (in particular before a replay queries any
        archive): object/array types of every section, a finite valid input position,
        the query options (an advanced query must parse), each catalog entry, and the
        science payload's catalogs, rows, matches and groups.
        """
        if not isinstance(data, Mapping):
            raise ManifestError("manifest must be a JSON object")
        schema = data.get("schema")
        if schema != MANIFEST_SCHEMA:
            raise ManifestError(f"unsupported manifest schema {schema!r} (expected {MANIFEST_SCHEMA!r})")
        required = ("created_at", "software", "target", "input_target", "radius_arcsec", "query", "catalogs",
                    "query_hash", "content_hash", "science")
        missing = [k for k in required if k not in data]
        if missing:
            raise ManifestError(f"manifest is missing {missing}")
        for key in ("software", "target", "input_target", "query", "catalogs", "science"):
            if not isinstance(data[key], Mapping):
                raise ManifestError(f"manifest {key!r} must be an object, not {type(data[key]).__name__}")
        for key in ("created_at", "query_hash", "content_hash"):
            if not isinstance(data[key], str):
                raise ManifestError(f"manifest {key!r} must be a string")
        for key in ("query_time", "query_time_source"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ManifestError(f"manifest {key!r} must be a string or null")
        if data.get("resolved_object") is not None and not isinstance(data["resolved_object"], Mapping):
            raise ManifestError("manifest 'resolved_object' must be an object or null")
        radius = data["radius_arcsec"]
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ManifestError("manifest radius_arcsec must be a number")
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ManifestError("manifest radius_arcsec must be finite and > 0")
        input_dict = dict(data["input_target"])
        _validate_input_target(input_dict)
        query = dict(data["query"])
        _validate_query(query)
        catalogs: dict[str, CatalogRequest] = {}
        for key, entry in data["catalogs"].items():
            if not isinstance(entry, Mapping):
                raise ManifestError(f"manifest catalogs[{key!r}] must be an object, not {type(entry).__name__}")
            catalogs[str(key)] = CatalogRequest.from_dict({**entry, "catalog": entry.get("catalog") or key})
        science = data["science"]
        if science.get("schema") != SCIENCE_SCHEMA:
            raise ManifestError(f"manifest science payload must have schema {SCIENCE_SCHEMA!r}")
        _validate_science(science)
        return cls(
            created_at=data["created_at"],
            software=dict(data["software"]),
            target=dict(data["target"]),
            input_target=input_dict,
            radius_arcsec=radius,
            query=query,
            catalogs=catalogs,
            query_hash=data["query_hash"],
            content_hash=data["content_hash"],
            science=json.loads(canonical_json(science)),
            resolved_object=dict(data["resolved_object"]) if data.get("resolved_object") else None,
            schema=str(schema),
            hash_method=str(data.get("hash_method") or HASH_METHOD),
            query_time=data.get("query_time"),
            query_time_source=data.get("query_time_source"),
        )

    @classmethod
    def from_json(cls, text: str) -> ProvenanceManifest:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def compute_query_hash(input_target_dict: Mapping[str, Any], radius: float, query: Mapping[str, Any],
                       catalogs: Mapping[str, CatalogRequest]) -> str:
    """``sha256:<hex>`` of the query definition (input target, radius, options, requests)."""
    return _prefixed(sha256_hex({
        "input_target": dict(input_target_dict),
        "radius_arcsec": radius,
        "query": {k: v for k, v in query.items() if k != "catalogs_planned"},
        "catalogs": {name: req.request_signature() for name, req in sorted(catalogs.items())},
    }))


def association_inputs(record: Mapping[str, Any]) -> dict[str, float]:
    """Target uncertainties the crossmatch scored the matches with, when given with the query.

    ``crossmatch.CrossmatchService.crossmatch`` takes ``target_uncertainty_arcsec`` (the
    1-sigma per-axis target position error; default ``association_config``, 0.1") and
    ``target_pm_error_masyr`` (that of a *given* proper motion); the streaming name
    search, ``POST /api/v1/search`` and ``astrosearch search`` set both from the Sesame
    answer. They change every match confidence (Budavari & Szalay 2008 posterior), so they
    are part of the question. The record carries them as
    ``provenance.association.target_sigma_arcsec`` and
    ``provenance.association.target_pm_error_masyr`` (older records:
    ``provenance.target_proper_motion.pm_error_masyr``); an error of an *adopted* motion is
    derived by the crossmatch (part of the answer) and is not returned.
    """
    prov = record.get("provenance") or {}
    out: dict[str, float] = {}
    assoc = prov.get("association") if isinstance(prov, Mapping) else None
    sigma = assoc.get("target_sigma_arcsec") if isinstance(assoc, Mapping) else None
    if isinstance(sigma, (int, float)) and not isinstance(sigma, bool) and math.isfinite(sigma) and sigma > 0:
        out["target_sigma_arcsec"] = float(sigma)
    pm_info = prov.get("target_proper_motion") if isinstance(prov, Mapping) else None
    pm_error = assoc.get("target_pm_error_masyr") if isinstance(assoc, Mapping) else None
    if pm_error is None and isinstance(pm_info, Mapping):
        pm_error = pm_info.get("pm_error_masyr")
    if (given_pm_source(record) is not None and isinstance(pm_error, (int, float)) and not isinstance(pm_error, bool)
            and math.isfinite(pm_error) and pm_error >= 0):
        out["target_pm_error_masyr"] = float(pm_error)
    return out


def build_manifest(
    record: UnifiedRecord | Mapping[str, Any],
    *,
    registry: CatalogRegistry | None = None,
    include_rows: bool = True,
    created_at: datetime | str | None = None,
    releases: Mapping[str, Mapping[str, str]] | None = None,
    query_time: datetime | str | None = None,
    self_check: bool = True,
) -> ProvenanceManifest:
    """Build the provenance manifest of a crossmatch result.

    ``registry`` (the one the query ran with) lets requests that left no trace on result
    rows (empty or failed catalogs, MAST/Gator parameters) be reconstructed exactly and
    lets recorded requests be verified against the registry. ``include_rows=False``
    stores a compact science payload (row digests only; see :func:`compact_science`).
    ``releases`` (from :func:`live_releases`) replaces hand-maintained release labels by
    the table descriptions the archive served -- but only when they were read within
    :data:`RELEASE_TIME_WINDOW_SECONDS` of the query. ``query_time`` is when the
    archives answered: give it when the search was just run (``now``); otherwise it is
    the earliest row ``retrieved_at`` of the record (:func:`record_query_time`), and
    unknown when no catalog returned a row.

    The input target (position, epoch, and a proper motion given as input or by the
    name resolver), radius and query options are stored at full float precision, and
    the proper-motion origin is recorded as ``query.pm_source``, so a replay re-sends
    exactly the requests of the original run.

    ``self_check`` re-validates the finished manifest with its own parser (skipped by
    :func:`replay_manifest`, whose records come from the crossmatch itself). This is CPU
    work proportional to the number of row cells: async code should call this function
    through ``asyncio.to_thread``.
    """
    rec = record_to_dict(record)
    prov = rec.get("provenance") or {}
    if not isinstance(prov, Mapping):
        raise ManifestError("record provenance must be an object")
    if not isinstance(rec.get("target"), Mapping) or rec["target"].get("ra") is None or rec["target"].get("dec") is None:
        raise ManifestError("record has no target position")
    radius = prov.get("query_radius_arcsec")
    if radius is None:
        raise ManifestError("record provenance lacks query_radius_arcsec")
    try:
        radius = float(radius)
    except (TypeError, ValueError) as exc:
        raise ManifestError("record query_radius_arcsec must be a number") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ManifestError("record query_radius_arcsec must be finite and > 0")
    try:
        given = input_target(rec)
    except (KeyError, TypeError, ValueError, InvalidCoordinateError) as exc:
        raise ManifestError(f"record target is invalid: {exc}") from exc
    asked_source: str | None
    if query_time is not None:
        asked = _parse_time(query_time)
        if asked is None:
            raise ManifestError(f"query_time {query_time!r} is not an ISO-8601 time")
        asked_source = "the search run for this manifest"
    else:
        asked = record_query_time(rec)
        asked_source = "earliest row retrieved_at of the record" if asked is not None else None
    catalogs = _catalog_requests(rec, registry, given, radius, releases, asked)
    resolved = rec.get("resolved_object")
    if resolved is not None and not isinstance(resolved, Mapping):
        raise ManifestError("record resolved_object must be an object or null")
    query = exact({
        "mode": "advanced" if prov.get("advanced_query") else "basic",
        "profile": prov.get("profile"),
        "advanced_query": prov.get("advanced_query"),
        "catalogs_planned": sorted(prov.get("catalogs_planned") or catalogs),
        "resolver": prov.get("resolver") or (resolved.get("resolver") if isinstance(resolved, Mapping) else None),
        "pm_source": given_pm_source(rec),
        "parallax_source": given_parallax_source(rec),
        **association_inputs(rec),
    })
    science = science_payload(rec)
    digest = _content_hash_canonical(science)
    stamp = created_at if isinstance(created_at, str) else (created_at or datetime.now(UTC)).isoformat()
    input_dict = exact(given.as_dict())
    manifest = ProvenanceManifest(
        created_at=stamp,
        software=software_info(),
        target=exact(rec["target"]),
        input_target=input_dict,
        radius_arcsec=radius,
        query=query,
        catalogs=catalogs,
        query_hash=compute_query_hash(input_dict, radius, query, catalogs),
        content_hash=digest,
        science=science if include_rows else compact_science(science),
        resolved_object=exact(rec.get("resolved_object")) if rec.get("resolved_object") else None,
        query_time=asked.isoformat() if asked is not None else None,
        query_time_source=asked_source,
    )
    if self_check:  # never hand out a manifest that its own parser rejects
        try:
            # The science payload was built canonical by science_payload: check its structure
            # directly instead of serializing and re-parsing every row.
            head = manifest.as_dict(include_science=False)
            head["science"] = {"schema": SCIENCE_SCHEMA, "catalogs": {}}
            ProvenanceManifest.from_dict(head)
            _validate_science(manifest.science)
        except ManifestError as exc:
            raise ManifestError(f"record cannot be described by a valid manifest: {exc}") from exc
    return manifest


def verify_manifest(manifest: ProvenanceManifest | Mapping[str, Any]) -> dict[str, Any]:
    """Integrity check: recompute the hashes stored in a manifest.

    Returns ``{"content_hash_ok": bool|None, "query_hash_ok": bool, "row_hashes_ok": bool}``;
    ``content_hash_ok`` is None for compact manifests (rows not stored).
    """
    m = manifest if isinstance(manifest, ProvenanceManifest) else ProvenanceManifest.from_dict(manifest)
    rows_ok = True
    content_ok = None
    if not m.compact:
        science = json.loads(canonical_json(m.science))  # one canonical pass, then per-row digests
        for item in (science.get("catalogs") or {}).values():
            for src in item.get("sources", []):
                if _row_hash_canonical(src) != src.get("row_hash"):
                    rows_ok = False
        content_ok = _content_hash_canonical(science) == m.content_hash
    query_ok = compute_query_hash(m.input_target, m.radius_arcsec, m.query, m.catalogs) == m.query_hash
    return {"content_hash_ok": content_ok, "query_hash_ok": query_ok, "row_hashes_ok": rows_ok}


# ---------------------------------------------------------------------------
# Science Diff
# ---------------------------------------------------------------------------


def _keyed_sources(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Map rows by source id; repeated ids get '#2', '#3', ... in canonical order."""
    keyed: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, int] = {}
    for src in _sorted_sources(sources):
        sid = str(src.get("source_id"))
        counts[sid] = counts.get(sid, 0) + 1
        keyed[sid if counts[sid] == 1 else f"{sid}#{counts[sid]}"] = src
    return keyed


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _values_differ(old: Any, new: Any, rtol: float) -> bool:
    if isinstance(old, bool) or isinstance(new, bool):
        return old != new
    old, new = js_number_value(old), js_number_value(new)  # big integers at double precision
    a, b = _as_number(old), _as_number(new)
    if a is not None and b is not None:
        return not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)
    return old != new


def _nested_differ(old: Any, new: Any, rtol: float, arcsec_tol: float, key: str = "") -> bool:
    """Deep comparison: numbers with ``rtol``, keys ending in ``_arcsec`` with an absolute
    angular tolerance, everything else exactly."""
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        return set(old) != set(new) or any(_nested_differ(old[k], new[k], rtol, arcsec_tol, str(k)) for k in old)
    if isinstance(old, list) and isinstance(new, list):
        return len(old) != len(new) or any(_nested_differ(a, b, rtol, arcsec_tol, key) for a, b in zip(old, new))
    a, b = _as_number(old), _as_number(new)
    if a is not None and b is not None and key.endswith("_arcsec"):
        return abs(a - b) > arcsec_tol
    return _values_differ(old, new, rtol)


def _move_threshold(old: Mapping[str, Any], new: Mapping[str, Any], position_tolerance_arcsec: float,
                    sigma_fraction: float) -> float:
    """Position change (arcsec) beyond which a row counts as moved (see DEFAULT_SIGMA_FRACTION)."""
    sigmas = [v for v in (_as_number(old.get("positional_error_arcsec")), _as_number(new.get("positional_error_arcsec")))
              if v is not None and v >= 0 and math.isfinite(v)]
    if sigmas and sigma_fraction > 0:
        return min(position_tolerance_arcsec, sigma_fraction * min(sigmas))
    return position_tolerance_arcsec


def _angle_change_arcsec(column: str, old: Any, new: Any, dec_deg: float | None) -> float | None:
    """Angular size (arcsec) of a change of an RA/Dec archive column in degrees, else None."""
    a, b = _as_number(old), _as_number(new)
    if a is None or b is None:
        return None
    name = column.lower()
    if name in DEC_COLUMNS:
        return abs(a - b) * 3600.0
    if name in RA_COLUMNS:
        delta = abs(a - b) % 360.0
        delta = min(delta, 360.0 - delta)
        cos_dec = math.cos(math.radians(dec_deg)) if dec_deg is not None else 1.0
        return delta * 3600.0 * abs(cos_dec)
    return None


def _field_changes(old: Mapping[str, Any], new: Mapping[str, Any], rtol: float, angle_tol_arcsec: float,
                   *, columns: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Changed values of one row; ``columns`` restricts the archive columns compared."""
    changes: dict[str, dict[str, Any]] = {}
    for key in ("positional_error_arcsec", "epoch", "pm_ra_masyr", "pm_dec_masyr"):
        if _values_differ(old.get(key), new.get(key), rtol):
            changes[key] = {"old": old.get(key), "new": new.get(key)}
    old_data, new_data = old.get("data") or {}, new.get("data") or {}
    dec = _as_number(old.get("dec"))
    keys = set(old_data) | set(new_data)
    if columns is not None:
        keys &= columns
    for key in sorted(keys):
        if key not in old_data:
            changes[f"data.{key}"] = {"old": None, "new": new_data[key], "added": True}
        elif key not in new_data:
            changes[f"data.{key}"] = {"old": old_data[key], "new": None, "removed": True}
        else:
            angle = _angle_change_arcsec(key, old_data[key], new_data[key], dec)
            differs = angle > angle_tol_arcsec if angle is not None else _values_differ(old_data[key], new_data[key], rtol)
            if differs:
                changes[f"data.{key}"] = {"old": old_data[key], "new": new_data[key]}
    return changes


def _brief(src: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_id": src.get("source_id"), "ra": src.get("ra"), "dec": src.get("dec")}


def _match_key(match: Mapping[str, Any]) -> str:
    return f"{match.get('catalog')}:{match.get('source_id')}"


def _match_changes(old: Mapping[str, Any], new: Mapping[str, Any], position_tolerance_arcsec: float,
                   confidence_tolerance: float) -> dict[str, Any]:
    old_m = {_match_key(m): m for m in old.get("matches") or []}
    new_m = {_match_key(m): m for m in new.get("matches") or []}
    changed = []
    for key in sorted(set(old_m) & set(new_m)):
        a, b = old_m[key], new_m[key]
        entry: dict[str, Any] = {"match": key}
        sep_a, sep_b = _as_number(a.get("separation_arcsec")), _as_number(b.get("separation_arcsec"))
        if (sep_a is None) != (sep_b is None) or (sep_a is not None and sep_b is not None
                                                  and abs(sep_a - sep_b) > position_tolerance_arcsec):
            entry["separation_arcsec"] = {"old": a.get("separation_arcsec"), "new": b.get("separation_arcsec")}
        conf_a, conf_b = _as_number(a.get("confidence")), _as_number(b.get("confidence"))
        if (conf_a is None) != (conf_b is None) or (conf_a is not None and conf_b is not None
                                                    and abs(conf_a - conf_b) > confidence_tolerance):
            entry["confidence"] = {"old": a.get("confidence"), "new": b.get("confidence")}
        if len(entry) > 1:
            changed.append(entry)
    return {"added": sorted(set(new_m) - set(old_m)), "removed": sorted(set(old_m) - set(new_m)), "changed": changed}


def diff_science(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    position_tolerance_arcsec: float = DEFAULT_POSITION_TOLERANCE_ARCSEC,
    numeric_rtol: float = DEFAULT_NUMERIC_RTOL,
    sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    confidence_tolerance: float = DEFAULT_CONFIDENCE_TOLERANCE,
    transport_changed: Iterable[str] = (),
    verified_row_hashes: bool = False,
) -> dict[str, Any]:
    """Structured difference between two science payloads (see :func:`science_payload`).

    Per catalog: ``status`` / ``row_count`` changes, ``added`` and ``removed`` rows (by
    source id), ``moved`` rows (same id, great-circle position change larger than
    ``min(position_tolerance_arcsec, sigma_fraction x the row's 1-sigma positional
    error)``) and ``changed`` rows (other values differing: numbers with relative
    tolerance ``numeric_rtol``, the archive's own RA/Dec columns as angles with the
    same threshold as the row position, big integers at double precision). With compact
    payloads a changed row is reported by digest only. Also reports catalogs present
    in only one payload, target changes (position as an angle; epoch, proper motion and
    parallax with ``numeric_rtol``), changes of the radius and of the target
    proper-motion and parallax origins (``other``; ``*_arcsec`` values with the
    absolute position tolerance), added / removed matches, matches whose separation
    (beyond ``position_tolerance_arcsec``) or confidence (beyond
    ``confidence_tolerance``) changed, and group membership changes. ``equivalent`` is
    True when nothing differs beyond the tolerances.

    ``transport_changed`` names catalogs whose rows were served by different archive
    services in the two payloads (the primary archive in one run, its declared fallback
    in the other, e.g. IRSA TAP VOTable vs IRSA Gator text for 2MASS). Their column sets
    differ by construction (Gator adds ``dist``/``angle``/``clon``/``clat``/``j_h``/...,
    TAP returns ``match_dist``) and TAP serves single-precision columns as float32
    (``j_m`` 11.766 -> 11.765999794), so for those catalogs rows are compared on the
    columns both services returned, numbers at single precision (relative tolerance
    :data:`FLOAT32_RTOL`), and the one-sided columns are listed once per catalog
    (``columns``) instead of as a change of every row.

    ``verified_row_hashes``: the ``row_hash`` stored with every full row is known to
    match it (both payloads freshly built, or checked by :func:`verify_manifest`), so
    unchanged rows are recognized without re-hashing them.
    """
    for label, value in (("position_tolerance_arcsec", position_tolerance_arcsec), ("sigma_fraction", sigma_fraction),
                         ("confidence_tolerance", confidence_tolerance), ("numeric_rtol", numeric_rtol)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and >= 0")
    old_cats: Mapping[str, Any] = old.get("catalogs") or {}
    new_cats: Mapping[str, Any] = new.get("catalogs") or {}
    per_catalog: dict[str, dict[str, Any]] = {}
    transport = set(transport_changed)
    totals = {"added": 0, "removed": 0, "moved": 0, "changed": 0, "status_changed": 0, "matches_changed": 0}
    for name in sorted(set(old_cats) & set(new_cats)):
        o, n = old_cats[name], new_cats[name]
        entry: dict[str, Any] = {}
        if o.get("status") != n.get("status"):
            entry["status"] = {"old": o.get("status"), "new": n.get("status"),
                               "old_error_type": o.get("error_type"), "new_error_type": n.get("error_type")}
            totals["status_changed"] += 1
        if o.get("row_count") != n.get("row_count"):
            entry["row_count"] = {"old": o.get("row_count"), "new": n.get("row_count")}
        if bool(o.get("truncated")) != bool(n.get("truncated")):
            entry["truncated"] = {"old": bool(o.get("truncated")), "new": bool(n.get("truncated"))}
        ok, nk = _keyed_sources(o.get("sources") or []), _keyed_sources(n.get("sources") or [])
        added = [_brief(nk[k]) for k in sorted(set(nk) - set(ok))]
        removed = [_brief(ok[k]) for k in sorted(set(ok) - set(nk))]
        moved, changed = [], []
        other_service = name in transport
        rtol = max(numeric_rtol, FLOAT32_RTOL) if other_service else numeric_rtol
        common: set[str] | None = None
        if other_service:
            old_cols = {c for s in ok.values() if isinstance(s.get("data"), Mapping) for c in s["data"]}
            new_cols = {c for s in nk.values() if isinstance(s.get("data"), Mapping) for c in s["data"]}
            common = old_cols & new_cols
            if old_cols != new_cols:
                entry["columns"] = {"only_old": sorted(old_cols - new_cols), "only_new": sorted(new_cols - old_cols),
                                    "reason": "rows served by different archive services in the two runs"}
        for key in sorted(set(ok) & set(nk)):
            a, b = ok[key], nk[key]
            # Full rows are re-hashed unless their stored digests are known to be right;
            # compact rows only have theirs.
            hash_a = row_hash(a) if "data" in a and not verified_row_hashes else a.get("row_hash")
            hash_b = row_hash(b) if "data" in b and not verified_row_hashes else b.get("row_hash")
            if hash_a and hash_a == hash_b:
                continue
            threshold = _move_threshold(a, b, position_tolerance_arcsec, sigma_fraction)
            compact = "data" not in a or "data" not in b
            fields = None if compact else _field_changes(a, b, rtol, threshold, columns=common)
            if None not in (_as_number(a.get("ra")), _as_number(a.get("dec")), _as_number(b.get("ra")), _as_number(b.get("dec"))):
                sep = haversine_arcsec(float(a["ra"]), float(a["dec"]), float(b["ra"]), float(b["dec"]))
                if sep > threshold:
                    # A moved row is reported once; its other changed values (usually the
                    # archive's own RA/Dec columns) travel with it.
                    moved.append({"source_id": a.get("source_id"), "separation_arcsec": sep, "threshold_arcsec": threshold,
                                  "old": {"ra": a["ra"], "dec": a["dec"]}, "new": {"ra": b["ra"], "dec": b["dec"]},
                                  "fields": fields})
                    continue
            if compact:
                changed.append({"source_id": a.get("source_id"), "fields": None,
                                "old_row_hash": a.get("row_hash"), "new_row_hash": b.get("row_hash")})
            elif fields:
                changed.append({"source_id": a.get("source_id"), "fields": fields})
        for label, items in (("added", added), ("removed", removed), ("moved", moved), ("changed", changed)):
            if items:
                entry[label] = items
                totals[label] += len(items)
        if entry:
            per_catalog[name] = entry
    # A catalog whose only difference is the column set of another archive service is
    # still equivalent science.
    substantive = {n for n, e in per_catalog.items() if set(e) - {"columns"}}

    target_changed: dict[str, Any] = {}
    old_t, new_t = old.get("target") or {}, new.get("target") or {}
    coords = [_as_number(old_t.get("ra")), _as_number(old_t.get("dec")), _as_number(new_t.get("ra")), _as_number(new_t.get("dec"))]
    if None not in coords:
        sep = haversine_arcsec(*coords)  # type: ignore[arg-type]
        if sep > position_tolerance_arcsec:
            target_changed["position"] = {"old": {"ra": old_t.get("ra"), "dec": old_t.get("dec")},
                                          "new": {"ra": new_t.get("ra"), "dec": new_t.get("dec")}, "separation_arcsec": sep}
    for key in sorted((set(old_t) | set(new_t)) - ({"ra", "dec"} if None not in coords else set())):
        if _values_differ(old_t.get(key), new_t.get(key), numeric_rtol):
            target_changed[key] = {"old": old_t.get(key), "new": new_t.get(key)}

    matches = _match_changes(old, new, position_tolerance_arcsec, confidence_tolerance)
    totals["matches_changed"] = len(matches["changed"])
    old_g = {tuple(g) for g in old.get("groups") or []}
    new_g = {tuple(g) for g in new.get("groups") or []}
    other_changes = {}
    for key in ("query_radius_arcsec", "effective_radius_arcsec", "target_proper_motion", "target_parallax"):
        if _nested_differ(canonicalize(old.get(key)), canonicalize(new.get(key)), numeric_rtol, position_tolerance_arcsec, key):
            other_changes[key] = {"old": old.get(key), "new": new.get(key)}
    diff: dict[str, Any] = {
        "catalogs": per_catalog,
        "catalogs_added": sorted(set(new_cats) - set(old_cats)),
        "catalogs_removed": sorted(set(old_cats) - set(new_cats)),
        "target": target_changed,
        "other": other_changes,
        "matches": matches,
        "groups": {"added": [list(g) for g in sorted(new_g - old_g)], "removed": [list(g) for g in sorted(old_g - new_g)]},
        "totals": totals,
        "position_tolerance_arcsec": position_tolerance_arcsec,
        "sigma_fraction": sigma_fraction,
        "numeric_rtol": numeric_rtol,
        "confidence_tolerance": confidence_tolerance,
    }
    diff["equivalent"] = not (
        substantive or diff["catalogs_added"] or diff["catalogs_removed"] or target_changed or other_changes
        or matches["added"] or matches["removed"] or matches["changed"] or diff["groups"]["added"] or diff["groups"]["removed"]
    )
    return diff


def served_by(req: CatalogRequest | None) -> str | None:
    """Which archive service answered a catalog: ``"primary"``, ``"fallback <provider>"``
    (the declared fallback ran because the primary failed), or None (failed / unknown)."""
    if req is None or req.status in ("failed", "unknown"):
        return None
    if _served_by_fallback(req.fallback, req.status):
        return f"fallback {(req.fallback or {}).get('provider')}"
    return "primary"


def diff_requests(old: Mapping[str, CatalogRequest], new: Mapping[str, CatalogRequest]) -> dict[str, dict[str, Any]]:
    """Catalogs whose request (endpoint, method, query text, parameters) changed.

    When a different archive service answered in the two runs (primary vs declared
    fallback, :func:`served_by`), the entry also has ``served_by: {old, new}``: the
    request differs because another service was asked, not because a definition changed.
    """
    changes: dict[str, dict[str, Any]] = {}
    for name in sorted(set(old) & set(new)):
        a, b = canonicalize(old[name].request_signature()), canonicalize(new[name].request_signature())
        if a["parameters"] is None or b["parameters"] is None:
            continue  # nothing recorded on one side: cannot compare
        if a != b:
            so, sn = served_by(old[name]), served_by(new[name])
            if (so is None) != (sn is None) and (so or sn) != "primary":
                # One run failed and the other was answered by the declared fallback: the failed
                # side's entry describes the primary request, so the two are not comparable.
                continue
            change: dict[str, Any] = {k: {"old": a[k], "new": b[k]} for k in a if a[k] != b[k]}
            if so and sn and so != sn:
                change["served_by"] = {"old": so, "new": sn}
            changes[name] = change
    return changes


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class _PinnedRegistry(CatalogRegistry):
    """Registry view exposing exactly the manifest's catalogs as enabled.

    A replay must query the same catalogs as the original even if the deployment
    registry has since enabled/disabled catalogs; definitions come from the current
    registry (so a changed definition shows up as a request change in the diff).
    ``overrides`` replaces the definitions of some catalogs (see :func:`_fallback_first`).
    """

    def __init__(self, base: CatalogRegistry, names: Iterable[str],
                 overrides: Mapping[str, CatalogDefinition] | None = None) -> None:
        wanted = set(names)
        self.registry_path = getattr(base, "registry_path", None)
        self._catalogs = {n: replace((overrides or {}).get(n, c), enabled=True)
                          for n, c in base.catalogs.items() if n in wanted}


def _fallback_first(definition: CatalogDefinition, fallback: Mapping[str, Any]) -> CatalogDefinition:
    """``definition`` with its declared fallback archive asked first and the primary second.

    Used to replay a catalog the original run got from its fallback (the primary was
    down): asking the same service again compares like with like -- IRSA Gator returns
    other columns than IRSA TAP and decimal text instead of float32 -- and the primary
    is still tried if the fallback fails now.
    """
    first = _fallback_definition(definition, fallback)
    back = {"provider": definition.provider, "endpoint": definition.endpoint, "catalog": definition.catalog,
            "table": definition.table,
            "parameters": {k: v for k, v in definition.parameters.items() if k != "fallback"}}
    return replace(first, parameters={**first.parameters, "fallback": back})


@dataclass(slots=True)
class ReplayResult:
    """Outcome of re-running a manifest's query.

    ``identical``: the content hashes are equal *and* every catalog of the original was
    actually compared -- a catalog that failed in both runs (``not_compared``) had no
    data to compare, so its science is not reproduced. ``content_hash_equal`` is the
    bare hash comparison.
    """

    identical: bool
    equivalent_within_tolerance: bool
    diff: dict[str, Any]
    new_manifest: ProvenanceManifest
    original_content_hash: str
    new_content_hash: str
    explanation: list[str]
    missing_catalogs: list[str] = field(default_factory=list)
    not_compared: list[dict[str, Any]] = field(default_factory=list)
    content_hash_equal: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "content_hash_equal": (self.original_content_hash == self.new_content_hash
                                   if self.content_hash_equal is None else self.content_hash_equal),
            "equivalent_within_tolerance": self.equivalent_within_tolerance,
            "original_content_hash": self.original_content_hash,
            "new_content_hash": self.new_content_hash,
            "diff": canonicalize(self.diff),
            "explanation": list(self.explanation),
            "missing_catalogs": list(self.missing_catalogs),
            "not_compared": canonicalize(self.not_compared),
            "new_manifest": self.new_manifest.as_dict(),
        }


def _client_of(service: CrossmatchService) -> httpx.AsyncClient | None:
    for provider in service.providers.values():
        client = getattr(provider, "client", None)
        if client is not None:
            return client
    return None


def _release_changed(old: CatalogRequest | None, new: CatalogRequest | None) -> bool:
    """Both releases were read live from the archive (at query time) and differ."""
    return bool(old and new and old.release_source.startswith("live") and new.release_source.startswith("live")
                and old.release != new.release)


def _release_unverified(req: CatalogRequest | None) -> bool:
    """The manifest could not tie the archive's release description to the query time."""
    return bool(req and req.release_source.startswith(UNVERIFIED_RELEASE_PREFIX))


def _served_text(req: CatalogRequest | None, label: str | None) -> str:
    if label is None:
        return "no archive (failed)"
    if label == "primary":
        return f"the primary archive ({req.provider} {req.endpoint})" if req else "the primary archive"
    fb = (req.fallback or {}) if req else {}
    reason = fb.get("reason")
    where = " ".join(str(x) for x in (fb.get("provider"), fb.get("endpoint")) if x)
    return f"the declared fallback archive ({where})" + (f" because the primary failed ({reason})" if reason else "")


def _explain(old: ProvenanceManifest, new: ProvenanceManifest, diff: Mapping[str, Any],
             request_changes: Mapping[str, Any], missing: Sequence[str], *,
             integrity: Mapping[str, Any] | None = None, not_compared: Sequence[Mapping[str, Any]] = (),
             transport: Mapping[str, Mapping[str, Any]] | None = None,
             fallback_failed: Mapping[str, str] | None = None) -> list[str]:
    notes: list[str] = []
    integrity = integrity if integrity is not None else verify_manifest(old)
    transport = transport or {}
    if integrity["content_hash_ok"] is False or integrity["row_hashes_ok"] is False:
        notes.append("Manifest integrity check failed: its science payload does not match its content hash "
                     "(the manifest was edited after it was created).")
    if not integrity["query_hash_ok"]:
        notes.append("Manifest integrity check failed: its query definition does not match its query hash.")
    uncompared = ", ".join(f"{c['catalog']} (original {c.get('old_error_type')}, replay {c.get('new_error_type')})"
                           for c in not_compared)
    if old.content_hash == new.content_hash:
        if not_compared:
            notes.append(f"Content hash identical, but {len(not_compared)} catalog(s) failed in both runs, so their "
                         f"data were never compared and are not reproduced: {uncompared}.")
        else:
            notes.append("Content hash identical: every archive returned the same science.")
    elif not_compared:
        notes.append(f"Catalogs that failed in both runs (their data were not compared): {uncompared}.")
    if missing:
        notes.append(f"Catalogs no longer in the registry could not be replayed: {', '.join(missing)}.")
    for name in sorted(set(old.catalogs) & set(new.catalogs)):
        if _release_changed(old.catalogs[name], new.catalogs[name]):
            notes.append(f"{name}: the archive's data release changed: {old.catalogs[name].release!r} -> "
                         f"{new.catalogs[name].release!r}.")
        elif _release_unverified(old.catalogs[name]):
            notes.append(f"{name}: the data release of the original query is not known -- the manifest was built "
                         f"from a stored record after the fact, so the archive's description could not be tied to "
                         f"the query time ({old.catalogs[name].release_source[len('label ('):].rstrip(')')}).")
    for name, info in sorted(transport.items()):
        notes.append(f"{name}: the original run was served by {info['old_text']} and the replay by "
                     f"{info['new_text']}: the request, the column set and the number format differ because a "
                     "different archive service answered, not because the registry or the software changed.")
    for name, reason in sorted((fallback_failed or {}).items()):
        notes.append(f"{name}: the declared fallback archive that served the original run failed in the replay "
                     f"({reason}); the primary archive answered instead.")
    for name, entry in diff.get("catalogs", {}).items():
        req_old, req_new = old.catalogs.get(name), new.catalogs.get(name)
        req = req_new or req_old
        living = bool(req and req.living_database)
        status = entry.get("status")
        if status:
            if status["new"] == "failed":
                why = "archive unreachable" if status.get("new_error_type") in UNREACHABLE_ERRORS else "query error"
                notes.append(f"{name}: failed during replay ({status.get('new_error_type')}, {why}); its rows are missing.")
                continue  # its 'removed' rows are a consequence of the failure, not of the archive
            if status["old"] == "failed":
                notes.append(f"{name}: failed in the original run ({status.get('old_error_type')}) but answered now; "
                             "its rows were missing from the original result.")
                continue  # its 'added' rows are a consequence of the original failure
            notes.append(f"{name}: status changed {status['old']} -> {status['new']}.")
        columns = entry.get("columns")
        if columns:
            notes.append(f"{name}: columns returned by only one of the two services were not compared "
                         f"(original only: {columns.get('only_old')}; replay only: {columns.get('only_new')}); the "
                         f"common columns were compared at single precision.")
        counts = {k: len(entry[k]) for k in ("added", "removed", "moved", "changed") if entry.get(k)}
        if counts:
            text = ", ".join(f"{v} {k}" for k, v in counts.items())
            if _release_changed(req_old, req_new):
                suffix = " -- explained by the data release change above"
            elif name in transport:
                suffix = " -- the two runs were answered by different archive services (see above)"
            elif name in request_changes:
                suffix = " -- the request differs from the original (see the request change below)"
            elif _release_unverified(req_old) and living:
                suffix = (" -- the release the original query saw is unknown and this table changes release in "
                          "place, so a data release change may explain it")
            elif living:
                suffix = " -- a continuously updated database, so changes over time are expected"
            else:
                suffix = " -- a fixed data release answered differently to the same request: investigate"
            notes.append(f"{name}: {text}{suffix}.")
    for name, change in request_changes.items():
        if name in transport:
            if transport[name].get("same_archive_request_unchanged") is False:
                notes.append(f"{name}: in addition, the request the archive that answered the original run would "
                             "be sent now differs from the manifest's: the registry definition or the "
                             "request-building software changed.")
            continue
        fields = ", ".join(sorted(k for k in change if k != "served_by"))
        notes.append(f"{name}: the request sent differs from the manifest ({fields}): the registry definition "
                     "or the request-building software changed.")
    if diff.get("catalogs_added") or diff.get("catalogs_removed"):
        notes.append(f"Catalog set changed: +{diff.get('catalogs_added')} -{diff.get('catalogs_removed')}.")
    target = diff.get("target") or {}
    if target:
        notes.append(f"Target changed ({', '.join(sorted(target))}): the replay did not start from the recorded input.")
    other = diff.get("other") or {}
    if "target_proper_motion" in other:
        o, n = other["target_proper_motion"]["old"] or {}, other["target_proper_motion"]["new"] or {}
        notes.append(f"Target proper motion changed: {o.get('source')} ({o.get('pm_ra_masyr')}, {o.get('pm_dec_masyr')}) -> "
                     f"{n.get('source')} ({n.get('pm_ra_masyr')}, {n.get('pm_dec_masyr')}) mas/yr; epoch-propagated "
                     "cones and match separations follow it.")
    if "target_parallax" in other:
        o, n = other["target_parallax"]["old"] or {}, other["target_parallax"]["new"] or {}
        notes.append(f"Target parallax changed: {o.get('source')} {o.get('parallax_mas')} -> {n.get('source')} "
                     f"{n.get('parallax_mas')} mas ({o.get('catalog') or '-'} -> {n.get('catalog') or '-'}); the annual "
                     "parallax removed from single-epoch positions (2MASS, SDSS, ...) and their separations follow it.")
    for key in ("query_radius_arcsec", "effective_radius_arcsec"):
        if key in other:
            notes.append(f"{key} changed: {other[key]['old']} -> {other[key]['new']}.")
    matches = diff.get("matches") or {}
    if matches.get("changed"):
        notes.append(f"{len(matches['changed'])} target match(es) changed separation or confidence for the same rows: "
                     "the crossmatch scoring or the target model changed.")
    if matches.get("added") or matches.get("removed"):
        notes.append(f"Target matches: +{len(matches.get('added') or [])} -{len(matches.get('removed') or [])}.")
    old_sw, new_sw = old.software, new.software
    if old_sw.get("version") != new_sw.get("version") or (old_sw.get("git") or {}).get("commit") != (new_sw.get("git") or {}).get("commit"):
        notes.append(
            f"Software differs: {old_sw.get('version')}@{(old_sw.get('git') or {}).get('commit')} -> "
            f"{new_sw.get('version')}@{(new_sw.get('git') or {}).get('commit')}."
        )
    elif (old_sw.get("code_sha256") or {}) != (new_sw.get("code_sha256") or {}) and old_sw.get("code_sha256"):
        notes.append("Software differs: same commit, but the core module code changed (code_sha256).")
    if old.content_hash != new.content_hash and diff.get("equivalent"):
        if transport:
            notes.append(f"Content hashes differ because {', '.join(sorted(transport))} came from different archive "
                         "services in the two runs; compared on their common columns at single precision, the "
                         "science is equivalent.")
        else:
            notes.append("Hashes differ only below the diff tolerances (e.g. sub-threshold position or last-digit "
                         "value changes).")
    return notes


def _swap_back_fallbacks(record: UnifiedRecord, swapped: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Record, for catalogs replayed fallback-first, which service really answered.

    The replay registry made the original run's fallback archive the first choice
    (:func:`_fallback_first`), so the executor reports the *primary* as "fallback" when
    it had to use it. Rewrite the record's fallback info in the registry's own terms:
    served by the declared fallback -> ``fallback`` set; the fallback failed and the
    primary answered -> ``fallback`` None. Returns ``{catalog: why the fallback failed}``
    for the latter.
    """
    failed_now: dict[str, str] = {}
    stats = record.provenance.get("catalog_stats") or {}
    for name, original in swapped.items():
        res = record.catalog_results.get(name)
        if not isinstance(res, dict) or res.get("status") == "failed":
            continue  # both archives failed: the failure (and its message) says so
        meta = res.get("fallback")
        if meta:
            failed_now[name] = str(meta.get("reason"))
            info: dict[str, Any] | None = None
        else:
            info = {"provider": original.get("provider"), "endpoint": original.get("endpoint"),
                    "reason": ("replay: the declared fallback archive was asked first because it served the "
                               f"original run (original reason: {original.get('reason')})")}
        res["fallback"] = info
        if isinstance(stats.get(name), dict):
            stats[name]["fallback"] = info
    return failed_now


def _compare(old: ProvenanceManifest, record: UnifiedRecord, pinned: CatalogRegistry, *, missing: list[str],
             swapped: Mapping[str, Mapping[str, Any]], fallback_failed: Mapping[str, str],
             releases: Mapping[str, Mapping[str, str]] | None, query_time: datetime,
             tolerances: Mapping[str, float]) -> ReplayResult:
    """The CPU part of a replay (manifest, diff, explanation): run it in a worker thread."""
    new = build_manifest(record, registry=pinned, include_rows=not old.compact, releases=releases,
                         query_time=query_time, self_check=False)
    integrity = verify_manifest(old)
    transport: dict[str, dict[str, Any]] = {}
    for name in sorted(set(old.catalogs) & set(new.catalogs)):
        a, b = old.catalogs[name], new.catalogs[name]
        so, sn = served_by(a), served_by(b)
        if so and sn and so != sn:
            info: dict[str, Any] = {"old": so, "new": sn, "old_text": _served_text(a, so), "new_text": _served_text(b, sn)}
            definition = pinned.catalogs.get(name)
            if definition is not None and a.parameters is not None:
                archive = (_fallback_definition(definition, a.fallback or {})
                           if so != "primary" else definition)
                try:
                    now_req = reconstruct_request(archive, validate_target(
                        old.input_target["ra"], old.input_target["dec"], **_target_kwargs(old.input_target, with_motion=True)),
                        old.radius_arcsec)
                    info["same_archive_request_unchanged"] = canonicalize(
                        {k: now_req[k] for k in ("endpoint", "http_method", "query", "parameters")}
                    ) == canonicalize(a.request_signature())
                except (ValueError, KeyError, TypeError, InvalidCoordinateError):
                    pass
            transport[name] = info
    diff = diff_science(old.science, new.science, transport_changed=transport,
                        verified_row_hashes=bool(integrity["row_hashes_ok"]),
                        position_tolerance_arcsec=tolerances["position_tolerance_arcsec"],
                        numeric_rtol=tolerances["numeric_rtol"], sigma_fraction=tolerances["sigma_fraction"],
                        confidence_tolerance=tolerances["confidence_tolerance"])
    request_changes = diff_requests(old.catalogs, new.catalogs)
    diff["requests"] = request_changes
    diff["releases"] = {
        name: {"old": old.catalogs[name].release, "new": new.catalogs[name].release}
        for name in sorted(set(old.catalogs) & set(new.catalogs)) if _release_changed(old.catalogs[name], new.catalogs[name])
    }
    diff["served_by"] = {name: {"old": info["old"], "new": info["new"]} for name, info in transport.items()}
    not_compared = [
        {"catalog": name, "old_error_type": old.catalogs[name].error_type, "new_error_type": new.catalogs[name].error_type}
        for name in sorted(set(old.catalogs) & set(new.catalogs))
        if old.catalogs[name].status == "failed" and new.catalogs[name].status == "failed"
    ]
    diff["not_compared"] = [c["catalog"] for c in not_compared]
    hashes_equal = old.content_hash == new.content_hash
    return ReplayResult(
        identical=hashes_equal and not not_compared,
        equivalent_within_tolerance=bool(diff["equivalent"]) and not not_compared,
        diff=diff,
        new_manifest=new,
        original_content_hash=old.content_hash,
        new_content_hash=new.content_hash,
        explanation=_explain(old, new, diff, request_changes, missing, integrity=integrity,
                             not_compared=not_compared, transport=transport, fallback_failed=fallback_failed),
        missing_catalogs=missing,
        not_compared=not_compared,
        content_hash_equal=hashes_equal,
    )


def _check_replay_radius(old: ProvenanceManifest, settings: Settings | None) -> None:
    """ManifestError unless every search radius of a manifest's query is a positive number no
    larger than MAX_SEARCH_RADIUS_ARCSEC (3600") and API_MAX_RADIUS_ARCSEC."""
    adv = old.query.get("advanced_query") if old.query.get("mode") == "advanced" else None
    radii: list[tuple[str, Any]] = [("radius_arcsec", old.radius_arcsec)]
    if isinstance(adv, Mapping) and adv.get("radius_arcsec") is not None:
        radii.append(("advanced_query.radius_arcsec", adv.get("radius_arcsec")))
    for label, value in radii:
        if value is None:
            continue
        try:
            radius = float(value)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"manifest {label} must be a number") from exc
        if not math.isfinite(radius) or radius <= 0 or radius > MAX_SEARCH_RADIUS_ARCSEC:
            raise ManifestError(f"manifest {label} {value!r} must be in (0, {MAX_SEARCH_RADIUS_ARCSEC:g}] arcsec")
        try:
            check_search_radius(radius, settings=settings, field=f"manifest {label}")
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc


async def replay_manifest(
    manifest: ProvenanceManifest | Mapping[str, Any],
    service: CrossmatchService | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    position_tolerance_arcsec: float = DEFAULT_POSITION_TOLERANCE_ARCSEC,
    numeric_rtol: float = DEFAULT_NUMERIC_RTOL,
    sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    confidence_tolerance: float = DEFAULT_CONFIDENCE_TOLERANCE,
    use_cache: bool = False,
    live_release_lookup: bool = True,
    settings: Settings | None = None,
) -> ReplayResult:
    """Re-run a manifest's query and compare the new science with the recorded one.

    The manifest's search radius (and its advanced query's) must lie within the static
    3600" ceiling and ``settings.max_radius_arcsec`` (API_MAX_RADIUS_ARCSEC; default: the
    environment's), as for a new search: a crafted manifest cannot send a wider cone to the
    archives (:class:`ManifestError`, checked before any request).

    The replay uses the manifest's *input* target at full precision (the resolved
    position when the original query was by name -- the resolver is not re-run), its
    given proper motion and parallax with their origins (``query.pm_source`` /
    ``query.parallax_source``; an adopted motion or parallax is adopted again), the
    target uncertainties the matches were scored with (``query.target_sigma_arcsec`` /
    ``query.target_pm_error_masyr``), radius, query mode and options, and exactly the
    manifest's catalogs, under the execution limits of ``service`` (timeouts, timeout
    cap, response-size cap, endpoint guards). Provider response caches are bypassed
    (``use_cache=False``) so the archives are really asked again; endpoint rate limits
    and circuit breakers of ``service`` are shared. With ``live_release_lookup`` the
    new manifest records the data release each HEASARC table serves now.

    A catalog the original run got from its declared fallback archive (the primary was
    unavailable; ``catalogs[name].fallback``) is replayed fallback-first, so the same
    service answers again (the primary is still tried if the fallback fails). When the
    two runs were nevertheless answered by different services, the diff compares that
    catalog's rows on their common columns at single precision and the explanation says
    so (``diff["served_by"]``).

    ``identical`` requires equal content hashes and no catalog that failed in both runs
    (``not_compared``): data that were never compared are not reproduced.

    Raises :class:`ManifestError` for an invalid manifest (before any network request)
    or one whose catalogs are all unknown to the registry, and
    :class:`ReplayUnavailableError` when no catalog could reach its archive. The
    manifest building, diff and explanation run in a worker thread (they are CPU work
    proportional to the number of row cells), so the event loop keeps serving.
    """
    old = manifest if isinstance(manifest, ProvenanceManifest) else ProvenanceManifest.from_dict(manifest)
    tolerances = {"position_tolerance_arcsec": position_tolerance_arcsec, "numeric_rtol": numeric_rtol,
                  "sigma_fraction": sigma_fraction, "confidence_tolerance": confidence_tolerance}
    diff_science({}, {}, position_tolerance_arcsec=position_tolerance_arcsec, numeric_rtol=numeric_rtol,
                 sigma_fraction=sigma_fraction, confidence_tolerance=confidence_tolerance)  # validates tolerances early
    own_client = None
    if service is None:
        from main import build_service  # lazy: main imports the whole application

        if client is None:
            own_client = client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
        service = build_service(client=client)
    try:
        names = sorted(old.catalogs) or list(old.query.get("catalogs_planned") or [])
        if not names:
            raise ManifestError("manifest lists no catalogs to replay")
        _check_replay_radius(old, settings)  # before any archive is queried
        missing = sorted(n for n in names if n not in service.registry.catalogs)
        if len(missing) == len(names):
            raise ManifestError(f"none of the manifest's catalogs is in the registry: {missing}")
        await software_info_async()  # first call runs git; keep it off the event loop
        pinned = _PinnedRegistry(service.registry, names)
        # Catalogs the original run got from their declared fallback: ask that service first.
        swapped: dict[str, dict[str, Any]] = {}
        overrides: dict[str, CatalogDefinition] = {}
        for name in names:
            req, definition = old.catalogs.get(name), pinned.catalogs.get(name)
            if (req is not None and definition is not None and req.fallback is not None
                    and _served_by_fallback(req.fallback, req.status)
                    and isinstance(definition.parameters.get("fallback"), Mapping)):
                swapped[name] = dict(req.fallback)
                overrides[name] = _fallback_first(definition, req.fallback)
        run_registry = _PinnedRegistry(service.registry, names, overrides) if overrides else pinned
        t = old.input_target
        query: AdvancedQuery | None = None
        if old.query.get("mode") == "advanced" and old.query.get("advanced_query"):
            adv = dict(old.query["advanced_query"])
            if adv.get("catalogs"):
                # The original QueryBuilder planned only the listed catalogs that also match
                # the profiles; the manifest (and so the pinned registry) holds exactly those.
                kept = [c for c in adv["catalogs"] if c in pinned.catalogs]
                if not kept:
                    raise ManifestError("none of the advanced query's catalogs is both in the manifest and the registry")
                adv["catalogs"] = kept
            query = parse_advanced_query(adv, pinned)  # checked before any archive is queried
        elif old.query.get("profile") is not None:
            try:
                validate_profile(old.query["profile"], pinned)
            except ValueError as exc:
                raise ManifestError(f"manifest query.profile is invalid: {exc}") from exc

        shared_client = client or _client_of(service)
        if use_cache:
            providers = service.providers
        else:
            # The deployment's execution limits apply unchanged -- endpoint guards (rate
            # limits, circuit breakers), request timeout, response-size cap and the cap on
            # per-catalog timeouts; only the response cache is bypassed. Otherwise a catalog
            # could time out or overflow in one run and not the other for reasons unrelated
            # to the archive, and that would be reported as a science change.
            first = next(iter(service.providers.values()), None)
            guards = getattr(first, "guards", None)
            max_bytes = getattr(first, "max_response_bytes", None)
            providers = provider_map(shared_client, timeout=service.executor.timeout, guards=guards,
                                     cache=CacheManager(None),
                                     **({"max_response_bytes": int(max_bytes)} if max_bytes else {}))
        replay_service = CrossmatchService(run_registry, providers, radius_arcsec=old.radius_arcsec,
                                           timeout=service.executor.timeout,
                                           timeout_cap=getattr(service.executor, "timeout_cap", None))
        extra: dict[str, Any] = {}
        # Target uncertainties the original matches were scored with (streaming name search).
        if old.query.get("target_sigma_arcsec") is not None and _accepts(replay_service.crossmatch,
                                                                          "target_uncertainty_arcsec"):
            extra["target_uncertainty_arcsec"] = old.query["target_sigma_arcsec"]
        if old.query.get("target_pm_error_masyr") is not None and _accepts(replay_service.crossmatch,
                                                                            "target_pm_error_masyr"):
            extra["target_pm_error_masyr"] = old.query["target_pm_error_masyr"]
        # A name search passed the resolver's answer to the engine (its catalogue row is the
        # target's identity, and it sets the target class): the replay does too.
        if old.resolved_object and _accepts(replay_service.crossmatch, "resolved_object"):
            extra["resolved_object"] = dict(old.resolved_object)
        started = datetime.now(UTC)
        if query is not None:
            record = await replay_service.crossmatch(t["ra"], t["dec"], query=query, **extra)
        else:
            has_pm = t.get("pm_ra_masyr") is not None and t.get("pm_dec_masyr") is not None
            # Only a parallax given with the query is in input_target; an adopted one is
            # adopted again by the crossmatch (see given_parallax_source).
            if t.get("parallax_mas") is not None and _accepts(replay_service.crossmatch, "parallax_mas"):
                extra["parallax_mas"] = t["parallax_mas"]
            record = await replay_service.crossmatch(
                t["ra"], t["dec"], radius_arcsec=old.radius_arcsec, epoch=t.get("epoch"),
                profile=old.query.get("profile"), pm_ra_masyr=t.get("pm_ra_masyr"), pm_dec_masyr=t.get("pm_dec_masyr"),
                pm_source=(old.query.get("pm_source") or "input") if has_pm else None, **extra,
            )
        if old.resolved_object:
            record.resolved_object = dict(old.resolved_object)
        if old.query.get("resolver"):
            record.provenance["resolver"] = old.query["resolver"]
        releases = await live_releases(shared_client, pinned, names) if live_release_lookup else None
    finally:
        if own_client is not None:
            await own_client.aclose()

    failed = [f for f in record.failures]
    present = [n for n in names if n not in missing]
    if len(failed) == len(present) and all(f.get("error_type") in UNREACHABLE_ERRORS for f in failed):
        raise ReplayUnavailableError(
            "every catalog failed to reach its archive during replay: "
            + "; ".join(f"{f['catalog']}: {f.get('error_type')}" for f in failed)
        )
    fallback_failed = _swap_back_fallbacks(record, swapped)
    return await asyncio.to_thread(
        _compare, old, record, pinned, missing=missing, swapped=swapped, fallback_failed=fallback_failed,
        releases=releases, query_time=started, tolerances=tolerances,
    )


# ---------------------------------------------------------------------------
# Citations: Verified References
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reference:
    """One citable work. ``bibcode`` None = a dataset DOI (verified via doi.org)."""

    key: str
    entry_type: str  # article | inproceedings | incollection | techreport | misc
    authors: tuple[str, ...]
    title: str
    year: int
    bibcode: str | None = None
    journal: str | None = None
    booktitle: str | None = None
    series: str | None = None
    volume: str | None = None
    pages: str | None = None
    publisher: str | None = None
    institution: str | None = None
    doi: str | None = None
    eprint: str | None = None
    url: str | None = None
    note: str | None = None
    more_authors: bool = False
    howpublished: str | None = None
    #: False for a reference parsed from free text (a registry citation) and not yet resolved
    #: through ADS/doi.org; the curated REFERENCES and resolved ones are True.
    verified: bool = True

    @property
    def ads_url(self) -> str | None:
        return ADS_ABS_URL.format(bibcode=self.bibcode.replace("&", "%26")) if self.bibcode else None

    @property
    def short(self) -> str:
        first = self.authors[0].split(",")[0].strip("{}") if self.authors else "Anon."
        etal = " et al." if len(self.authors) > 2 or self.more_authors else (f" & {self.authors[1].split(',')[0].strip('{}')}" if len(self.authors) == 2 else "")
        return f"{first}{etal} {self.year}"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ads_url"] = self.ads_url
        out["short"] = self.short
        out["authors"] = list(self.authors)
        return out


def _ref(key: str, entry_type: str, authors: Sequence[str], title: str, year: int, **kw: Any) -> Reference:
    return Reference(key=key, entry_type=entry_type, authors=tuple(authors), title=title, year=year, **kw)


_AA = "Astronomy & Astrophysics"
_AAS = "Astronomy and Astrophysics Supplement Series"
_AJ = "The Astronomical Journal"
_APJ = "The Astrophysical Journal"
_APJS = "The Astrophysical Journal Supplement Series"
_PASP = "Publications of the Astronomical Society of the Pacific"
_MNRAS = "Monthly Notices of the Royal Astronomical Society"
_ASPC = "Astronomical Society of the Pacific Conference Series"

# Bibliographic data: bibcodes confirmed by the ADS link gateway; journal, volume,
# first page, year and DOI confirmed from the DOI's CSL-JSON (doi.org) or, for
# ASP Conference Series papers, from aspbooks.org; IVOA documents from ivoa.net.
REFERENCES: dict[str, Reference] = {r.key: r for r in [
    _ref("2023A&A...674A...1G", "article", ["{Gaia Collaboration}", "Vallenari, A.", "Brown, A. G. A.", "Prusti, T."],
         "Gaia Data Release 3. Summary of the content and survey properties", 2023, bibcode="2023A&A...674A...1G",
         journal=_AA, volume="674", pages="A1", doi="10.1051/0004-6361/202243940", eprint="2208.00211", more_authors=True),
    _ref("2016A&A...595A...1G", "article", ["{Gaia Collaboration}", "Prusti, T.", "de Bruijne, J. H. J.", "Brown, A. G. A."],
         "The Gaia mission", 2016, bibcode="2016A&A...595A...1G", journal=_AA, volume="595", pages="A1",
         doi="10.1051/0004-6361/201629272", more_authors=True),
    _ref("2000A&AS..143....9W", "article", ["Wenger, M.", "Ochsenbein, F.", "Egret, D.", "Dubois, P.", "Bonnarel, F.", "Borde, S."],
         "The SIMBAD astronomical database. The CDS reference database for astronomical objects", 2000,
         bibcode="2000A&AS..143....9W", journal=_AAS, volume="143", pages="9--22", doi="10.1051/aas:2000332", more_authors=True),
    _ref("2000A&AS..143...23O", "article", ["Ochsenbein, F.", "Bauer, P.", "Marcout, J."],
         "The VizieR database of astronomical catalogues", 2000, bibcode="2000A&AS..143...23O", journal=_AAS,
         volume="143", pages="23--32", doi="10.1051/aas:2000169"),
    _ref("1991ASSL..171...89H", "incollection", ["Helou, G.", "Madore, B. F.", "Schmitz, M.", "Bicay, M. D.", "Wu, X.", "Bennett, J."],
         "The NASA/IPAC Extragalactic Database", 1991, bibcode="1991ASSL..171...89H",
         booktitle="Astrophysics and Space Science Library, vol. 171", volume="171", pages="89--106",
         publisher="Springer Netherlands", doi="10.1007/978-94-011-3250-3_10"),
    _ref("2006AJ....131.1163S", "article", ["Skrutskie, M. F.", "Cutri, R. M.", "Stiening, R.", "Weinberg, M. D.", "Schneider, S.", "Carpenter, J. M."],
         "The Two Micron All Sky Survey (2MASS)", 2006, bibcode="2006AJ....131.1163S", journal=_AJ, volume="131",
         pages="1163--1183", doi="10.1086/498708", more_authors=True),
    _ref("2010AJ....140.1868W", "article", ["Wright, E. L.", "Eisenhardt, P. R. M.", "Mainzer, A. K.", "Ressler, M. E.", "Cutri, R. M."],
         "The Wide-field Infrared Survey Explorer (WISE): Mission Description and Initial On-orbit Performance", 2010,
         bibcode="2010AJ....140.1868W", journal=_AJ, volume="140", pages="1868--1881", doi="10.1088/0004-6256/140/6/1868",
         more_authors=True),
    _ref("2011ApJ...731...53M", "article", ["Mainzer, A.", "Bauer, J.", "Grav, T.", "Masiero, J.", "Cutri, R. M."],
         "Preliminary Results from NEOWISE: An Enhancement to the Wide-field Infrared Survey Explorer for Solar System Science",
         2011, bibcode="2011ApJ...731...53M", journal=_APJ, volume="731", pages="53", doi="10.1088/0004-637X/731/1/53",
         more_authors=True),
    _ref("2014ApJ...792...30M", "article", ["Mainzer, A.", "Bauer, J.", "Cutri, R. M.", "Grav, T.", "Masiero, J."],
         "Initial Performance of the NEOWISE Reactivation Mission", 2014, bibcode="2014ApJ...792...30M", journal=_APJ,
         volume="792", pages="30", doi="10.1088/0004-637X/792/1/30", more_authors=True),
    _ref("2013wise.rept....1C", "techreport", ["Cutri, R. M."], "Explanatory Supplement to the AllWISE Data Release Products",
         2013, bibcode="2013wise.rept....1C", institution="IPAC/Caltech",
         url="https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/index.html", more_authors=True),
    _ref("2016arXiv161205560C", "misc", ["Chambers, K. C.", "Magnier, E. A.", "Metcalfe, N.", "Flewelling, H. A.", "Huber, M. E."],
         "The Pan-STARRS1 Surveys", 2016, bibcode="2016arXiv161205560C", eprint="1612.05560",
         doi="10.48550/arXiv.1612.05560", note="arXiv:1612.05560", more_authors=True),
    _ref("2020ApJS..251....7F", "article", ["Flewelling, H. A.", "Magnier, E. A.", "Chambers, K. C.", "Heasley, J. N.", "Holmberg, C.", "Huber, M. E."],
         "The Pan-STARRS1 Database and Data Products", 2020, bibcode="2020ApJS..251....7F", journal=_APJS, volume="251",
         pages="7", doi="10.3847/1538-4365/abb82d", more_authors=True),
    _ref("2023ApJS..267...44A", "article", ["Almeida, A.", "Anderson, S. F.", "Argudo-Fernández, M.", "Badenes, C.", "Barger, K."],
         "The Eighteenth Data Release of the Sloan Digital Sky Surveys: Targeting and First Spectra from SDSS-V", 2023,
         bibcode="2023ApJS..267...44A", journal=_APJS, volume="267", pages="44", doi="10.3847/1538-4365/acda98", more_authors=True),
    _ref("2000AJ....120.1579Y", "article", ["York, D. G.", "Adelman, J.", "Anderson, Jr., J. E.", "Anderson, S. F.", "Annis, J."],
         "The Sloan Digital Sky Survey: Technical Summary", 2000, bibcode="2000AJ....120.1579Y", journal=_AJ,
         volume="120", pages="1579--1587", doi="10.1086/301513", more_authors=True),
    _ref("2016MNRAS.460.1371B", "article", ["Beck, R.", "Dobos, L.", "Budavári, T.", "Szalay, A. S.", "Csabai, I."],
         "Photometric redshifts for the SDSS Data Release 12", 2016, bibcode="2016MNRAS.460.1371B", journal=_MNRAS,
         volume="460", pages="1371--1381", doi="10.1093/mnras/stw1009"),
    _ref("1995ApJ...450..559B", "article", ["Becker, R. H.", "White, R. L.", "Helfand, D. J."],
         "The FIRST Survey: Faint Images of the Radio Sky at Twenty Centimeters", 1995, bibcode="1995ApJ...450..559B",
         journal=_APJ, volume="450", pages="559", doi="10.1086/176166"),
    _ref("2015ApJ...801...26H", "article", ["Helfand, D. J.", "White, R. L.", "Becker, R. H."],
         "The Last of FIRST: The Final Catalog and Source Identifications", 2015, bibcode="2015ApJ...801...26H",
         journal=_APJ, volume="801", pages="26", doi="10.1088/0004-637X/801/1/26"),
    _ref("1998AJ....115.1693C", "article", ["Condon, J. J.", "Cotton, W. D.", "Greisen, E. W.", "Yin, Q. F.", "Perley, R. A.", "Taylor, G. B.", "Broderick, J. J."],
         "The NRAO VLA Sky Survey", 1998, bibcode="1998AJ....115.1693C", journal=_AJ, volume="115", pages="1693--1716",
         doi="10.1086/300337"),
    _ref("2020PASP..132c5001L", "article", ["Lacy, M.", "Baum, S. A.", "Chandler, C. J.", "Chatterjee, S.", "Clarke, T. E."],
         "The Karl G. Jansky Very Large Array Sky Survey (VLASS). Science Case and Survey Design", 2020,
         bibcode="2020PASP..132c5001L", journal=_PASP, volume="132", pages="035001", doi="10.1088/1538-3873/ab63eb",
         more_authors=True),
    _ref("2021ApJ...914...42B", "article", ["Bruzewski, S.", "Schinzel, F. K.", "Taylor, G. B.", "Petrov, L."],
         "Radio Counterpart Candidates to Unassociated 4FGL-DR2 Sources", 2021, bibcode="2021ApJ...914...42B",
         journal=_APJ, volume="914", pages="42", doi="10.3847/1538-4357/abf73b"),
    _ref("2017A&A...598A.104S", "article", ["Shimwell, T. W.", "Röttgering, H. J. A.", "Best, P. N.", "Williams, W. L."],
         "The LOFAR Two-metre Sky Survey. I. Survey description and preliminary data release", 2017,
         bibcode="2017A&A...598A.104S", journal=_AA, volume="598", pages="A104", doi="10.1051/0004-6361/201629313",
         more_authors=True),
    _ref("2022A&A...659A...1S", "article", ["Shimwell, T. W.", "Hardcastle, M. J.", "Tasse, C.", "Best, P. N."],
         "The LOFAR Two-metre Sky Survey. V. Second data release", 2022, bibcode="2022A&A...659A...1S", journal=_AA,
         volume="659", pages="A1", doi="10.1051/0004-6361/202142484", more_authors=True),
    _ref("2026A&A...707A.198S", "article", ["Shimwell, T. W.", "Hardcastle, M. J.", "Tasse, C.", "Drabent, A."],
         "The LOFAR Two-metre Sky Survey. VII. Third Data Release", 2026, bibcode="2026A&A...707A.198S", journal=_AA,
         volume="707", pages="A198", doi="10.1051/0004-6361/202557749", more_authors=True),
    _ref("2013A&A...556A...2V", "article", ["van Haarlem, M. P.", "Wise, M. W.", "Gunst, A. W.", "Heald, G.", "McKean, J. P."],
         "LOFAR: The LOw-Frequency ARray", 2013, bibcode="2013A&A...556A...2V", journal=_AA, volume="556", pages="A2",
         doi="10.1051/0004-6361/201220873", eprint="1305.3550", more_authors=True),
    _ref("2016A&A...588A.103B", "article", ["Boller, Th.", "Freyberg, M. J.", "Trümper, J.", "Haberl, F.", "Voges, W.", "Nandra, K."],
         "Second ROSAT all-sky survey (2RXS) source catalogue", 2016, bibcode="2016A&A...588A.103B", journal=_AA,
         volume="588", pages="A103", doi="10.1051/0004-6361/201525648"),
    _ref("1999A&A...349..389V", "article", ["Voges, W.", "Aschenbach, B.", "Boller, Th.", "Bräuninger, H."],
         "The ROSAT all-sky survey bright source catalogue", 1999, bibcode="1999A&A...349..389V", journal=_AA,
         volume="349", pages="389", eprint="astro-ph/9909315", more_authors=True),
    _ref("2022A&A...664A.105F", "article", ["Freund, S.", "Czesla, S.", "Robrade, J.", "Schneider, P. C.", "Schmitt, J. H. M. M."],
         "The stellar content of the ROSAT all-sky survey", 2022, bibcode="2022A&A...664A.105F", journal=_AA,
         volume="664", pages="A105", doi="10.1051/0004-6361/202142573"),
    _ref("2010ApJS..189...37E", "article", ["Evans, I. N.", "Primini, F. A.", "Glotfelty, K. J.", "Anderson, C. S.", "Bonaventura, N. R."],
         "The Chandra Source Catalog", 2010, bibcode="2010ApJS..189...37E", journal=_APJS, volume="189", pages="37--82",
         doi="10.1088/0067-0049/189/1/37", more_authors=True),
    _ref("2024ApJS..274...22E", "article", ["Evans, I. N.", "Evans, J. D.", "Martínez-Galarza, J. R.", "Miller, J. B.", "Primini, F. A."],
         "The Chandra Source Catalog Release 2 Series", 2024, bibcode="2024ApJS..274...22E", journal=_APJS, volume="274",
         pages="22", doi="10.3847/1538-4365/ad6319", more_authors=True),
    _ref("2020A&A...641A.136W", "article", ["Webb, N. A.", "Coriat, M.", "Traulsen, I.", "Ballet, J.", "Motch, C.", "Carrera, F. J."],
         "The XMM-Newton serendipitous survey. IX. The fourth XMM-Newton serendipitous source catalogue", 2020,
         bibcode="2020A&A...641A.136W", journal=_AA, volume="641", pages="A136", doi="10.1051/0004-6361/201937353",
         more_authors=True),
    _ref("2020A&A...641A.137T", "article", ["Traulsen, I.", "Schwope, A. D.", "Lamer, G.", "Ballet, J.", "Carrera, F. J."],
         "The XMM-Newton serendipitous survey. X. The second source catalogue from overlapping XMM-Newton observations "
         "and its long-term variable content", 2020, bibcode="2020A&A...641A.137T", journal=_AA, volume="641",
         pages="A137", doi="10.1051/0004-6361/202037706", eprint="2007.02932", more_authors=True),
    _ref("2013PASP..125..989A", "article", ["Akeson, R. L.", "Chen, X.", "Ciardi, D.", "Crane, M.", "Good, J.", "Harbut, M."],
         "The NASA Exoplanet Archive: Data and Tools for Exoplanet Research", 2013, bibcode="2013PASP..125..989A",
         journal=_PASP, volume="125", pages="989--999", doi="10.1086/672273", more_authors=True),
    _ref("2012ASPC..461..291B", "inproceedings", ["Boch, T.", "Pineau, F.", "Derriere, S."], "The CDS Cross-Match Service",
         2012, bibcode="2012ASPC..461..291B", booktitle="Astronomical Data Analysis Software and Systems XXI", series=_ASPC,
         volume="461", pages="291"),
    _ref("2020ASPC..522..125P", "inproceedings", ["Pineau, F.", "Boch, T.", "Derrière, S.", "Schaaff, A."],
         "The CDS Cross-match Service: Key Figures, Internals and Future Plans", 2020, bibcode="2020ASPC..522..125P",
         booktitle="Astronomical Data Analysis Software and Systems XXVII", series=_ASPC, volume="522", pages="125"),
    _ref("2013A&A...558A..33A", "article", ["{Astropy Collaboration}", "Robitaille, T. P.", "Tollerud, E. J.", "Greenfield, P."],
         "Astropy: A community Python package for astronomy", 2013, bibcode="2013A&A...558A..33A", journal=_AA,
         volume="558", pages="A33", doi="10.1051/0004-6361/201322068", more_authors=True),
    _ref("2018AJ....156..123A", "article", ["{Astropy Collaboration}", "Price-Whelan, A. M.", "Sipőcz, B. M.", "Günther, H. M."],
         "The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package", 2018,
         bibcode="2018AJ....156..123A", journal=_AJ, volume="156", pages="123", doi="10.3847/1538-3881/aabc4f", more_authors=True),
    _ref("2022ApJ...935..167A", "article", ["{Astropy Collaboration}", "Price-Whelan, A. M.", "Lim, P. L.", "Earl, N."],
         "The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major "
         "Release (v5.0) of the Core Package", 2022, bibcode="2022ApJ...935..167A", journal=_APJ, volume="935", pages="167",
         doi="10.3847/1538-4357/ac7c74", more_authors=True),
    _ref("2000A&AS..143...33B", "article", ["Bonnarel, F.", "Fernique, P.", "Bienaymé, O.", "Egret, D.", "Genova, F.", "Louys, M."],
         "The ALADIN interactive sky atlas. A reference tool for identification of astronomical sources", 2000,
         bibcode="2000A&AS..143...33B", journal=_AAS, volume="143", pages="33--40", doi="10.1051/aas:2000331", more_authors=True),
    _ref("2014ASPC..485..277B", "inproceedings", ["Boch, T.", "Fernique, P."], "Aladin Lite: Embed your Sky in the Browser",
         2014, bibcode="2014ASPC..485..277B", booktitle="Astronomical Data Analysis Software and Systems XXIII", series=_ASPC,
         volume="485", pages="277"),
    _ref("2022ASPC..532....7B", "inproceedings", ["Baumann, M.", "Boch, T.", "Pineau, F.-X.", "Fernique, P.", "Bot, C.", "Allen, M."],
         "Aladin Lite v3: Behind the Scenes of a Major Overhaul", 2022, bibcode="2022ASPC..532....7B",
         booktitle="Astronomical Data Analysis Software and Systems XXX", series=_ASPC, volume="532", pages="7"),
    _ref("2015A&A...578A.114F", "article", ["Fernique, P.", "Allen, M. G.", "Boch, T.", "Oberto, A.", "Pineau, F.-X.", "Durand, D."],
         "Hierarchical progressive surveys. Multi-resolution HEALPix data structures for astronomical images, catalogues, "
         "and 3-dimensional data cubes", 2015, bibcode="2015A&A...578A.114F", journal=_AA, volume="578", pages="A114",
         doi="10.1051/0004-6361/201526075", more_authors=True),
    _ref("2017ivoa.spec.0519F", "techreport", ["Fernique, P.", "Allen, M.", "Boch, T.", "Donaldson, T.", "Durand, D.",
                                                "Ebisawa, K.", "Michel, L.", "Salgado, J.", "Stoehr, F."],
         "HiPS - Hierarchical Progressive Survey Version 1.0", 2017, bibcode="2017ivoa.spec.0519F",
         institution="International Virtual Observatory Alliance", note="IVOA Recommendation 19 May 2017",
         doi="10.5479/ADS/bib/2017ivoa.spec.0519F", url="https://www.ivoa.net/documents/HiPS/20170519/index.html"),
    _ref("2005ApJ...622..759G", "article", ["Górski, K. M.", "Hivon, E.", "Banday, A. J.", "Wandelt, B. D.", "Hansen, F. K.", "Reinecke, M.", "Bartelmann, M."],
         "HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere", 2005,
         bibcode="2005ApJ...622..759G", journal=_APJ, volume="622", pages="759--771", doi="10.1086/427976"),
    _ref("2024A&A...689A..93R", "article", ["Rodrigo, C.", "Cruz, P.", "Aguilar, J. F.", "Aller, A.", "Solano, E.", "Gálvez-Ortiz, M. C."],
         "Photometric segregation of dwarf and giant FGK stars using the SVO Filter Profile Service and photometric tools",
         2024, bibcode="2024A&A...689A..93R", journal=_AA, volume="689", pages="A93", doi="10.1051/0004-6361/202449998",
         more_authors=True),
    _ref("2012ivoa.rept.1015R", "techreport", ["Rodrigo, C.", "Solano, E.", "Bayo, A."], "SVO Filter Profile Service Version 1.0",
         2012, bibcode="2012ivoa.rept.1015R", institution="International Virtual Observatory Alliance",
         note="IVOA Working Draft 15 October 2012", url="https://www.ivoa.net/documents/Notes/SVOFPS/index.html"),
    _ref("2020sea..confE.182R", "inproceedings", ["Rodrigo, C.", "Solano, E."], "The SVO Filter Profile Service", 2020,
         bibcode="2020sea..confE.182R", booktitle="XIV.0 Scientific Meeting (virtual) of the Spanish Astronomical Society",
         pages="182"),
    _ref("2006ASPC..351..367B", "inproceedings", ["Berthier, J.", "Vachier, F.", "Thuillot, W.", "Fernique, P.", "Ochsenbein, F.",
                                                   "Genova, F.", "Lainey, V.", "Arlot, J.-E."],
         "SkyBoT, a new VO service to identify Solar System objects", 2006, bibcode="2006ASPC..351..367B",
         booktitle="Astronomical Data Analysis Software and Systems XV", series=_ASPC, volume="351", pages="367"),
    _ref("2016MNRAS.458.3394B", "article", ["Berthier, J.", "Carry, B.", "Vachier, F.", "Eggl, S.", "Santerne, A."],
         "Prediction of transits of Solar system objects in Kepler/K2 images: an extension of the Virtual Observatory "
         "service SkyBoT", 2016, bibcode="2016MNRAS.458.3394B", journal=_MNRAS, volume="458", pages="3394--3398",
         doi="10.1093/mnras/stw492"),
    _ref("2019PASP..131a8002B", "article", ["Bellm, E. C.", "Kulkarni, S. R.", "Graham, M. J.", "Dekany, R.", "Smith, R. M.", "Riddle, R."],
         "The Zwicky Transient Facility: System Overview, Performance, and First Results", 2019,
         bibcode="2019PASP..131a8002B", journal=_PASP, volume="131", pages="018002", doi="10.1088/1538-3873/aaecbe",
         more_authors=True),
    _ref("2019PASP..131a8003M", "article", ["Masci, F. J.", "Laher, R. R.", "Rusholme, B.", "Shupe, D. L.", "Groom, S.", "Surace, J."],
         "The Zwicky Transient Facility: Data Processing, Products, and Archive", 2019, bibcode="2019PASP..131a8003M",
         journal=_PASP, volume="131", pages="018003", doi="10.1088/1538-3873/aae8ac", more_authors=True),
    _ref("2019PASP..131g8001G", "article", ["Graham, M. J.", "Kulkarni, S. R.", "Bellm, E. C.", "Adams, S. M.", "Barbarino, C.", "Blagorodnova, N."],
         "The Zwicky Transient Facility: Science Objectives", 2019, bibcode="2019PASP..131g8001G", journal=_PASP,
         volume="131", pages="078001", doi="10.1088/1538-3873/ab006c", more_authors=True),
    _ref("2021AJ....161..242F", "article", ["Förster, F.", "Cabrera-Vives, G.", "Castillo-Navarrete, E.", "Estévez, P. A.",
                                             "Sánchez-Sáez, P.", "Arredondo, J."],
         "The Automatic Learning for the Rapid Classification of Events (ALeRCE) Alert Broker", 2021,
         bibcode="2021AJ....161..242F", journal=_AJ, volume="161", pages="242", doi="10.3847/1538-3881/abe9bc",
         more_authors=True),
    # TESS, GALEX and unWISE (light-curve, SED and imaging features): bibcodes confirmed by the
    # ADS link gateway (PUB_HTML -> the DOI below) and DOI CSL-JSON metadata, 2026-09-28.
    _ref("2015JATIS...1a4003R", "article", ["Ricker, G. R.", "Winn, J. N.", "Vanderspek, R.", "Latham, D. W.",
                                            "Bakos, G. Á.", "Bean, J. L."],
         "Transiting Exoplanet Survey Satellite (TESS)", 2015, bibcode="2015JATIS...1a4003R",
         journal="Journal of Astronomical Telescopes, Instruments, and Systems", volume="1", pages="014003",
         doi="10.1117/1.JATIS.1.1.014003", more_authors=True),
    _ref("2016SPIE.9913E..3EJ", "inproceedings", ["Jenkins, J. M.", "Twicken, J. D.", "McCauliff, S.", "Campbell, J.",
                                                   "Sanderfer, D.", "Lung, D."],
         "The TESS science processing operations center", 2016, bibcode="2016SPIE.9913E..3EJ",
         booktitle="Software and Cyberinfrastructure for Astronomy IV", series="Proc. SPIE", volume="9913",
         pages="99133E", doi="10.1117/12.2233418", more_authors=True),
    _ref("2005ApJ...619L...1M", "article", ["Martin, D. C.", "Fanson, J.", "Schiminovich, D.", "Morrissey, P.",
                                            "Friedman, P. G.", "Barlow, T. A."],
         "The Galaxy Evolution Explorer: A Space Ultraviolet Survey Mission", 2005, bibcode="2005ApJ...619L...1M",
         journal=_APJ, volume="619", pages="L1--L6", doi="10.1086/426387", more_authors=True),
    _ref("2007ApJS..173..682M", "article", ["Morrissey, P.", "Conrow, T.", "Barlow, T. A.", "Small, T.", "Seibert, M.",
                                            "Wyder, T. K."],
         "The Calibration and Data Products of GALEX", 2007, bibcode="2007ApJS..173..682M", journal=_APJS,
         volume="173", pages="682--697", doi="10.1086/520512", more_authors=True),
    _ref("2017ApJS..230...24B", "article", ["Bianchi, L.", "Shiao, B.", "Thilker, D."],
         "Revised Catalog of GALEX Ultraviolet Sources. I. The All-Sky Survey: GUVcat_AIS", 2017,
         bibcode="2017ApJS..230...24B", journal=_APJS, volume="230", pages="24", doi="10.3847/1538-4365/aa7053"),
    _ref("2014AJ....147..108L", "article", ["Lang, D."], "unWISE: Unblurred Coadds of the WISE Imaging", 2014,
         bibcode="2014AJ....147..108L", journal=_AJ, volume="147", pages="108", doi="10.1088/0004-6256/147/5/108"),
    _ref("2017AJ....154..161M", "article", ["Meisner, A. M.", "Lang, D.", "Schlegel, D. J."],
         "Deep Full-sky Coadds from Three Years of WISE and NEOWISE Observations", 2017, bibcode="2017AJ....154..161M",
         journal=_AJ, volume="154", pages="161", doi="10.3847/1538-3881/aa894e"),
    _ref("2022RNAAS...6..188M", "article", ["Meisner, A. M.", "Lang, D.", "Schlafly, E. F.", "Schlegel, D. J."],
         "9-yr Deep Sky unWISE Coadds", 2022, bibcode="2022RNAAS...6..188M",
         journal="Research Notes of the American Astronomical Society", volume="6", pages="188",
         doi="10.3847/2515-5172/ac913e"),
    # Dataset DOIs (DataCite metadata confirmed through doi.org content negotiation).
    _ref("NED_DOI_2019", "misc", ["{NASA/IPAC Extragalactic Database (NED)}"], "NASA/IPAC Extragalactic Database (NED)", 2019,
         publisher="IPAC", doi="10.26132/NED1"),
    _ref("IRSA_2MASS_PSC_DOI", "misc", ["Skrutskie, M. F.", "Cutri, R. M.", "Stiening, R."], "2MASS All-Sky Point Source Catalog (PSC)",
         2003, publisher="IPAC", doi="10.26131/IRSA2", more_authors=True),
    _ref("IRSA_AllWISE_DOI", "misc", ["Wright, E. L.", "Eisenhardt, P. R. M.", "Mainzer, A. K."], "AllWISE Source Catalog", 2019,
         publisher="IPAC", doi="10.26131/IRSA1", more_authors=True),
    _ref("MAST_PS1_DR2_DOI", "misc", ["{STScI}"], "Pan-STARRS1 DR2 Catalog", 2022, publisher="STScI/MAST",
         doi="10.17909/s0zg-jx37"),
    _ref("CDS_VizieR_DOI", "misc", ["Ochsenbein, F."], "The VizieR database of astronomical catalogues", 1996,
         publisher="CDS, Centre de Données astronomiques de Strasbourg", doi="10.26093/cds/vizier"),
    _ref("CXC_CSC2_DOI", "misc", ["{CXC-DS}"], "Chandra X-ray Observatory DOI for Chandra Source Catalog 2.1", 2024,
         publisher="Chandra X-ray Center/SAO", doi="10.25574/csc2",
         note="DOI of the Chandra Source Catalog Release 2 series (cxc.cfa.harvard.edu/csc/cite.html)"),
    _ref("NEA_PSCompPars_DOI", "misc", ["{NASA Exoplanet Science Institute}"], "Planetary Systems Composite Table", 2020,
         publisher="IPAC", doi="10.26133/NEA13"),
]}


# ---------------------------------------------------------------------------
# Citations: Catalog & Service Entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationEntry:
    """Acknowledgement text and references for one catalog, archive service or tool."""

    key: str
    name: str
    kind: Literal["catalog", "archive", "service", "software"]
    acknowledgement: str
    references: tuple[str, ...]
    source_url: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["references"] = list(self.references)
        return out


# HEASARC FAQ "How do I acknowledge data obtained from the HEASARC?" (page updated
# 21-Jan-2026, read 2026-09-28), verbatim.
_HEASARC_ACK_URL = "https://heasarc.gsfc.nasa.gov/docs/faq.html"
_HEASARC_ACK = (
    "This research has made use of data, software and/or web tools obtained from the High Energy Astrophysics "
    "Science Archive Research Center (HEASARC), a service of the Astrophysics Science Division at NASA/GSFC and of "
    "the Smithsonian Astrophysical Observatory's High Energy Astrophysics Division."
)
# NED "Acknowledging NED" (last change 2025-04-14, read 2026-09-28), verbatim.
_NED_ACK_URL = "https://ned.ipac.caltech.edu/Documents/Overview/Acknowledgments"
_NED_ACK = (
    "This research has made use of the NASA/IPAC Extragalactic Database, which is funded by the National "
    "Aeronautics and Space Administration and operated by the California Institute of Technology."
)
# NEOWISE release page ("please use this acknowledgement in any published material that
# makes use of NEOWISE data products"; last update 2024 November 14, read 2026-09-28).
_NEOWISE_ACK_URL = "https://wise2.ipac.caltech.edu/docs/release/neowise/"
_NEOWISE_ACK = (
    "This publication makes use of data products from the Near-Earth Object Wide-field Infrared Survey Explorer "
    "(NEOWISE), which is a joint project of the Jet Propulsion Laboratory/California Institute of Technology and the "
    "University of California, Los Angeles. NEOWISE is funded by the National Aeronautics and Space Administration."
)
# MAST "Mission Acknowledgements" (archive.stsci.edu/publishing/mission-acknowledgements,
# read 2026-09-28): the TESS text verbatim, and the legacy-mission template ("This
# research is based on observations made with the mission <name>, ...") with the
# mission name "Galaxy Evolution Explorer" from its list filled in.
_MAST_ACK_URL = "https://archive.stsci.edu/publishing/mission-acknowledgements"
_TESS_ACK = (
    "This paper includes data collected with the TESS mission, obtained from the MAST data archive at the Space "
    "Telescope Science Institute (STScI). Funding for US Institutions for the TESS mission is provided by the NASA "
    "Explorer Program. STScI is operated by the Association of Universities for Research in Astronomy, Inc., under "
    "NASA contract NAS5–26555."
)
_GALEX_ACK = (
    "This research is based on observations made with the Galaxy Evolution Explorer, obtained from the MAST data "
    "archive at the Space Telescope Science Institute, which is operated by the Association of Universities for "
    "Research in Astronomy, Inc., under NASA contract NAS5–26555."
)
_IRSA_ACK = (
    "This research has made use of the NASA/IPAC Infrared Science Archive, which is funded by the National Aeronautics "
    "and Space Administration and operated by the California Institute of Technology."
)
_NRAO_ACK = (
    "The National Radio Astronomy Observatory is a facility of the National Science Foundation operated under "
    "cooperative agreement by Associated Universities, Inc."
)
_CDS_ACK_URL = "https://cds.unistra.fr/help/acknowledgement/"
# ZTF DR24 release notes, section 1.a "Acknowledging ZTF and Referencing Data Product usage"
# ("Please include the following text in your acknowledgments ..."), read 2026-09-28.
_ZTF_ACK_URL = "https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/dr24/ztf_release_notes_dr24.pdf"
_ZTF_ACK = (
    "Based on observations obtained with the Samuel Oschin Telescope 48-inch and the 60-inch Telescope at the "
    "Palomar Observatory as part of the Zwicky Transient Facility project. ZTF is supported by the National "
    "Science Foundation under Grants No. AST-1440341 and AST-2034437 and a collaboration including current "
    "partners Caltech, IPAC, the Oskar Klein Center at Stockholm University, the University of Maryland, "
    "University of California, Berkeley, the University of Wisconsin at Milwaukee, University of Warwick, Ruhr "
    "University, Cornell University, Northwestern University and Drexel University. Operations are conducted by "
    "COO, IPAC, and UW."
)

# Official acknowledgement paragraphs, verbatim from the pages cited next to each (read
# 2026-09-28; link markup removed). Pan-STARRS1: panstarrs.stsci.edu ("Credit where it is
# due"). SDSS: SDSS-I/II classic.sdss.org/collaboration/credits.html, SDSS-III
# www.sdss3.org/collaboration/boiler-plate.php, SDSS-IV www.sdss4.org/collaboration/,
# SDSS-V www.sdss.org/collaboration/citing-sdss/. LOFAR: lofar-surveys.org/releases.html
# (data-use paragraph) and lofar-surveys.org/credits.html (the fuller text the releases
# page requires for DR2/DR3 data; its first two sentences repeat the data-use paragraph
# without the van Haarlem et al. 2013 citation, so they are not repeated here).
_PS1_ACK = (
    'The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive have been made possible through '
    'contributions by the Institute for Astronomy, the University of Hawaii, the Pan-STARRS Project Office, '
    'the Max-Planck Society and its participating institutes, the Max Planck Institute for Astronomy, '
    'Heidelberg and the Max Planck Institute for Extraterrestrial Physics, Garching, The Johns Hopkins '
    "University, Durham University, the University of Edinburgh, the Queen's University Belfast, the "
    'Harvard-Smithsonian Center for Astrophysics, the Las Cumbres Observatory Global Telescope Network '
    'Incorporated, the National Central University of Taiwan, the Space Telescope Science Institute, the '
    'National Aeronautics and Space Administration under Grant No. NNX08AR22G issued through the Planetary '
    'Science Division of the NASA Science Mission Directorate, the National Science Foundation Grant No. '
    'AST-1238877, the University of Maryland, Eotvos Lorand University (ELTE), the Los Alamos National '
    'Laboratory, and the Gordon and Betty Moore Foundation.'
)
_SDSS12_ACK = (
    'Funding for the SDSS and SDSS-II has been provided by the Alfred P. Sloan Foundation, the Participating '
    'Institutions, the National Science Foundation, the U.S. Department of Energy, the National Aeronautics '
    'and Space Administration, the Japanese Monbukagakusho, the Max Planck Society, and the Higher Education '
    'Funding Council for England. The SDSS Web Site is http://www.sdss.org/. The SDSS is managed by the '
    'Astrophysical Research Consortium for the Participating Institutions. The Participating Institutions are'
    ' the American Museum of Natural History, Astrophysical Institute Potsdam, University of Basel, '
    'University of Cambridge, Case Western Reserve University, University of Chicago, Drexel University, '
    'Fermilab, the Institute for Advanced Study, the Japan Participation Group, Johns Hopkins University, the'
    ' Joint Institute for Nuclear Astrophysics, the Kavli Institute for Particle Astrophysics and Cosmology, '
    'the Korean Scientist Group, the Chinese Academy of Sciences (LAMOST), Los Alamos National Laboratory, '
    'the Max-Planck-Institute for Astronomy (MPIA), the Max-Planck-Institute for Astrophysics (MPA), New '
    'Mexico State University, Ohio State University, University of Pittsburgh, University of Portsmouth, '
    'Princeton University, the United States Naval Observatory, and the University of Washington.'
)
_SDSS3_ACK = (
    'Funding for SDSS-III has been provided by the Alfred P. Sloan Foundation, the Participating '
    'Institutions, the National Science Foundation, and the U.S. Department of Energy Office of Science. The '
    'SDSS-III web site is http://www.sdss3.org/. SDSS-III is managed by the Astrophysical Research Consortium'
    ' for the Participating Institutions of the SDSS-III Collaboration including the University of Arizona, '
    'the Brazilian Participation Group, Brookhaven National Laboratory, Carnegie Mellon University, '
    'University of Florida, the French Participation Group, the German Participation Group, Harvard '
    'University, the Instituto de Astrofisica de Canarias, the Michigan State/Notre Dame/JINA Participation '
    'Group, Johns Hopkins University, Lawrence Berkeley National Laboratory, Max Planck Institute for '
    'Astrophysics, Max Planck Institute for Extraterrestrial Physics, New Mexico State University, New York '
    'University, Ohio State University, Pennsylvania State University, University of Portsmouth, Princeton '
    'University, the Spanish Participation Group, University of Tokyo, University of Utah, Vanderbilt '
    'University, University of Virginia, University of Washington, and Yale University.'
)
_SDSS4_ACK = (
    'Funding for the Sloan Digital Sky Survey IV has been provided by the Alfred P. Sloan Foundation, the '
    'U.S. Department of Energy Office of Science, and the Participating Institutions. SDSS acknowledges '
    'support and resources from the Center for High-Performance Computing at the University of Utah. The SDSS'
    ' web site is www.sdss4.org. SDSS is managed by the Astrophysical Research Consortium for the '
    'Participating Institutions of the SDSS Collaboration including the Brazilian Participation Group, the '
    'Carnegie Institution for Science, Carnegie Mellon University, Center for Astrophysics | Harvard & '
    'Smithsonian (CfA), the Chilean Participation Group, the French Participation Group, Instituto de '
    'Astrofísica de Canarias, The Johns Hopkins University, Kavli Institute for the Physics and Mathematics '
    'of the Universe (IPMU) / University of Tokyo, the Korean Participation Group, Lawrence Berkeley National'
    ' Laboratory, Leibniz Institut für Astrophysik Potsdam (AIP), Max-Planck-Institut für Astronomie (MPIA '
    'Heidelberg), Max-Planck-Institut für Astrophysik (MPA Garching), Max-Planck-Institut für '
    'Extraterrestrische Physik (MPE), National Astronomical Observatories of China, New Mexico State '
    'University, New York University, University of Notre Dame, Observatório Nacional / MCTI, The Ohio State '
    'University, Pennsylvania State University, Shanghai Astronomical Observatory, United Kingdom '
    'Participation Group, Universidad Nacional Autónoma de México, University of Arizona, University of '
    'Colorado Boulder, University of Oxford, University of Portsmouth, University of Utah, University of '
    'Virginia, University of Washington, University of Wisconsin, Vanderbilt University, and Yale University.'
)
_SDSS5_ACK = (
    'Funding for the Sloan Digital Sky Survey V has been provided by the Alfred P. Sloan Foundation, the '
    'Heising-Simons Foundation, the National Science Foundation, and the Participating Institutions. SDSS '
    'acknowledges support and resources from the Center for High-Performance Computing at the University of '
    'Utah. SDSS telescopes are located at Apache Point Observatory, funded by the Astrophysical Research '
    'Consortium and operated by New Mexico State University, and at Las Campanas Observatory, operated by the'
    ' Carnegie Institution for Science. The SDSS web site is www.sdss.org. SDSS is managed by the '
    'Astrophysical Research Consortium for the Participating Institutions of the SDSS Collaboration, '
    'including the Carnegie Institution for Science, Chilean National Time Allocation Committee (CNTAC) '
    'ratified researchers, Caltech, the Gotham Participation Group, Harvard University, Heidelberg '
    'University, The Flatiron Institute, The Johns Hopkins University, L’Ecole polytechnique fédérale de '
    'Lausanne (EPFL), Leibniz-Institut für Astrophysik Potsdam (AIP), Max-Planck-Institut für Astronomie '
    '(MPIA Heidelberg), Max-Planck-Institut für Extraterrestrische Physik (MPE), Nanjing University, National'
    ' Astronomical Observatories of China (NAOC), New Mexico State University, The Ohio State University, '
    'Pennsylvania State University, Smithsonian Astrophysical Observatory, Space Telescope Science Institute '
    '(STScI), the Stellar Astrophysics Participation Group, Universidad Nacional Autónoma de México (UNAM), '
    'University of Arizona, University of Colorado Boulder, University of Illinois at Urbana-Champaign, '
    'University of Toronto, University of Utah, University of Virginia, Yale University, and Yunnan '
    'University.'
)
_LOFAR_RELEASES_ACK = (
    'LOFAR data products were provided by the LOFAR Surveys Key Science project (LSKSP; '
    'https://lofar-surveys.org/) and were derived from observations with the International LOFAR Telescope '
    '(ILT). LOFAR (van Haarlem et al. 2013) is the Low Frequency Array, designed and constructed by ASTRON. '
    'It has observing, data processing, and data storage facilities in several countries, which are owned by '
    'various parties (each with their own funding sources), and which are collectively operated by the LOFAR '
    'ERIC under a joint scientific policy. The efforts of the LSKSP have benefited from funding from the '
    'European Research Council, NOVA, NWO, CNRS-INSU, the SURF Co-operative, the UK Science and Technology '
    'Funding Council and the Jülich Supercomputing Centre.'
)
_LOFAR_CREDITS_ACK = (
    'The LOFAR resources have benefited from the following recent major funding sources: CNRS-INSU, '
    "Observatoire de Paris and Université d'Orléans, France; BMFTR, MKW-NRW, MPG, Germany; Science Foundation"
    ' Ireland (SFI), Department of Business, Enterprise and Innovation (DBEI), Ireland; NWO, The Netherlands;'
    ' The Science and Technology Facilities Council, UK; Ministry of Science and Higher Education, Poland; '
    'The Istituto Nazionale di Astrofisica (INAF), Italy. This research made use of the Dutch national '
    'e-infrastructure with support of the SURF Cooperative (e-infra 180169) and the LOFAR e-infra group. The '
    'Jülich LOFAR Long Term Archive and the German LOFAR network are both coordinated and operated by the '
    'Jülich Supercomputing Centre (JSC), and computing resources on the supercomputer JUWELS at JSC were '
    'provided by the Gauss Centre for Supercomputing e.V. (grant CHTB00) through the John von Neumann '
    'Institute for Computing (NIC). This research made use of the University of Hertfordshire '
    'high-performance computing facility and the LOFAR-UK computing facility located at the University of '
    'Hertfordshire and supported by STFC [ST/P000096/1], and of the Italian LOFAR-IT computing infrastructure'
    ' supported and operated by INAF, including the resources within the PLEIADI special "LOFAR" project by '
    'USC-C of INAF, and by the Physics Department of Turin university (under an agreement with Consorzio '
    'Interuniversitario per la Fisica Spaziale) at the C3S Supercomputing Centre, Italy. This research is '
    'part of the project LOFAR Data Valorization (LDV) [project numbers 2020.031, 2022.033, and 2024.047] of '
    'the research programme Computing Time on National Computer Facilities using SPIDER that is (co-)funded '
    'by the Dutch Research Council (NWO), hosted by SURF through the call for proposals of Computing Time on '
    'National Computer Facilities.'
)


# Acknowledgement texts: the CDS services (SIMBAD, VizieR, Aladin, X-Match, hips2fits),
# SkyBoT, SVO, ZTF, Astropy, Pan-STARRS1, SDSS, LOFAR, Chandra and XMM-Newton verbatim
# from the pages in ``source_url`` / the constants above (any adaptation is stated in
# the entry's ``note``); other catalog texts follow the
# mission/archive wording carried in the catalog registry (models.DEFAULT_CATALOGS).
# Entries whose ``note`` says "generic" have no prescribed sentence from the data
# provider. When a registry is supplied, its own acknowledgement/citation strings are
# returned alongside (registry_* keys).
CITATION_ENTRIES: dict[str, CitationEntry] = {e.key: e for e in [
    CitationEntry("gaia_dr3", "Gaia DR3", "catalog",
                  "This work has made use of data from the European Space Agency (ESA) mission Gaia "
                  "(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium "
                  "(DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by "
                  "national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.",
                  ("2023A&A...674A...1G", "2016A&A...595A...1G"),
                  "https://www.cosmos.esa.int/web/gaia-users/credits"),
    CitationEntry("simbad", "SIMBAD", "catalog",
                  "This research has made use of the SIMBAD database, CDS, Strasbourg Astronomical Observatory, France.",
                  ("2000A&AS..143....9W",), _CDS_ACK_URL),
    CitationEntry("ned", "NASA/IPAC Extragalactic Database", "catalog", _NED_ACK,
                  ("NED_DOI_2019", "1991ASSL..171...89H"), _NED_ACK_URL),
    CitationEntry("exoplanet_archive", "NASA Exoplanet Archive (Planetary Systems Composite Parameters)", "catalog",
                  "This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute "
                  "of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet "
                  "Exploration Program.",
                  ("2013PASP..125..989A", "NEA_PSCompPars_DOI"), "https://exoplanetarchive.ipac.caltech.edu/"),
    CitationEntry("twomass_psc", "2MASS Point Source Catalog", "catalog",
                  "This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project "
                  "of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute "
                  "of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.",
                  ("2006AJ....131.1163S", "IRSA_2MASS_PSC_DOI")),
    CitationEntry("vizier_2mass_reference", "2MASS Point Source Catalog (VizieR II/246)", "catalog",
                  "This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project "
                  "of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute "
                  "of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.",
                  ("2006AJ....131.1163S",)),
    CitationEntry("allwise", "AllWISE", "catalog",
                  "This publication makes use of data products from the Wide-field Infrared Survey Explorer, which is a "
                  "joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California "
                  "Institute of Technology, and NEOWISE, which is a project of the Jet Propulsion Laboratory/California "
                  "Institute of Technology. WISE and NEOWISE are funded by the National Aeronautics and Space Administration.",
                  ("2010AJ....140.1868W", "2011ApJ...731...53M", "2013wise.rept....1C", "IRSA_AllWISE_DOI")),
    CitationEntry("panstarrs_dr2", "Pan-STARRS1 DR2", "catalog", _PS1_ACK,
                  ("2016arXiv161205560C", "2020ApJS..251....7F", "MAST_PS1_DR2_DOI"), "https://panstarrs.stsci.edu/"),
    CitationEntry("sdss", "SDSS DR18 (SkyServer)", "catalog",
                  f"{_SDSS12_ACK} {_SDSS3_ACK} {_SDSS4_ACK} {_SDSS5_ACK}",
                  ("2023ApJS..267...44A", "2000AJ....120.1579Y", "2016MNRAS.460.1371B"),
                  "https://www.sdss.org/collaboration/citing-sdss/",
                  note="SDSS asks for the acknowledgement of every phase whose data are used. The query reads "
                       "PhotoPrimary (imaging taken by SDSS-I/II, DR1-7, photometry reprocessed for DR8 by SDSS-III), "
                       "Photoz (Beck et al. 2016, DR12, SDSS-III) and SpecObj (spectra from SDSS-I to -IV) from the DR18 "
                       "data release of SDSS-V, so all four phase texts are included; drop those whose data you do not "
                       "use. Beck et al. 2016 is the photometric-redshift table joined in the query."),
    CitationEntry("first", "FIRST", "catalog", "This research has made use of the FIRST survey catalogue (VLA, NRAO). " + _NRAO_ACK,
                  ("1995ApJ...450..559B", "2015ApJ...801...26H"),
                  note="generic first sentence; the NRAO sentence is the observatory's standard acknowledgement."),
    CitationEntry("nvss", "NVSS", "catalog", "This research has made use of the NRAO VLA Sky Survey (NVSS) catalogue. " + _NRAO_ACK,
                  ("1998AJ....115.1693C",),
                  note="generic first sentence; the NRAO sentence is the observatory's standard acknowledgement."),
    CitationEntry("vlass", "VLASS Epoch 1 (Bruzewski et al. 2021)", "catalog", _NRAO_ACK,
                  ("2020PASP..132c5001L", "2021ApJ...914...42B")),
    CitationEntry("lotss", "LoTSS-DR3", "catalog", _LOFAR_RELEASES_ACK + " " + _LOFAR_CREDITS_ACK,
                  ("2026A&A...707A.198S", "2017A&A...598A.104S", "2013A&A...556A...2V"),
                  "https://lofar-surveys.org/releases.html",
                  note="LSKSP data-use paragraph (releases page) followed by the fuller credits text the releases page "
                       "requires for DR2/DR3 data (lofar-surveys.org/credits.html)."),
    CitationEntry("lotss_dr2", "LoTSS-DR2", "catalog", _LOFAR_RELEASES_ACK + " " + _LOFAR_CREDITS_ACK,
                  ("2022A&A...659A...1S", "2017A&A...598A.104S", "2013A&A...556A...2V"),
                  "https://lofar-surveys.org/releases.html",
                  note="LSKSP data-use paragraph (releases page) followed by the fuller credits text the releases page "
                       "requires for DR2/DR3 data (lofar-surveys.org/credits.html)."),
    CitationEntry("rosat", "2RXS (Second ROSAT All-Sky Survey)", "catalog",
                  "This research has made use of the Second ROSAT All-Sky Survey source catalogue (2RXS).",
                  ("2016A&A...588A.103B", "2022A&A...664A.105F"),
                  note="generic sentence; Freund et al. 2022 is the positional-error calibration applied to 2RXS."),
    CitationEntry("rosat_bsc", "1RXS Bright Source Catalogue", "catalog",
                  "This research has made use of the ROSAT All-Sky Survey Bright Source Catalogue (1RXS).",
                  ("1999A&A...349..389V",), note="generic sentence."),
    CitationEntry("chandra", "Chandra Source Catalog 2.1", "catalog",
                  "This research has made use of data obtained from the Chandra Source Catalog, provided by the Chandra "
                  "X-ray Center (CXC).",
                  ("2024ApJS..274...22E", "2010ApJS..189...37E", "CXC_CSC2_DOI"), "https://cxc.cfa.harvard.edu/csc/cite.html",
                  note="The CXC asks for the Release 2 series DOI doi:10.25574/csc2 and Evans et al. 2024."),
    CitationEntry("xmm", "XMM-Newton Serendipitous Source Catalogue (5XMM-DR15)", "catalog",
                  "This research has made use of data obtained from the 5XMM serendipitous source catalog compiled by the "
                  "XMM-Newton Survey Science Center, the XMM2ATHENA project and in collaboration with the XMM-Newton SOC.",
                  ("2020A&A...641A.136W", "2020A&A...641A.137T"),
                  "https://heasarc.gsfc.nasa.gov/W3Browse/xmm-newton/xmmssc.html",
                  note="HEASARC xmmssc (5XMM-DR15) cites the 4XMM papers Webb et al. 2020 and Traulsen et al. 2020; the "
                       "5XMM catalogue paper (Webb, Traulsen et al., 'XI') is in preparation."),
    # Archives hosting catalogs (added automatically from a catalog's endpoint).
    CitationEntry("vizier", "VizieR", "archive",
                  "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg Astronomical "
                  "Observatory, France (DOI : 10.26093/cds/vizier).",
                  ("2000A&AS..143...23O", "CDS_VizieR_DOI"), _CDS_ACK_URL),
    CitationEntry("heasarc", "HEASARC", "archive", _HEASARC_ACK, (), _HEASARC_ACK_URL),
    CitationEntry("irsa", "NASA/IPAC Infrared Science Archive", "archive", _IRSA_ACK, (), "https://irsa.ipac.caltech.edu/"),
    # Services used by other AstroSearch features.
    CitationEntry("cds_xmatch", "CDS X-Match", "service",
                  "This research has made use of the CDS cross-match service, Strasbourg Astronomical Observatory, France.",
                  ("2012ASPC..461..291B", "2020ASPC..522..125P"), _CDS_ACK_URL),
    CitationEntry("aladin", "Aladin / Aladin Lite", "service",
                  "This research has made use of Aladin sky atlas, CDS, Strasbourg Astronomical Observatory, France.",
                  ("2000A&AS..143...33B", "2014ASPC..485..277B", "2022ASPC..532....7B"), _CDS_ACK_URL),
    CitationEntry("hips", "HiPS (Hierarchical Progressive Surveys)", "service",
                  "This research has made use of Hierarchical Progressive Surveys (HiPS) served by CDS, Strasbourg "
                  "Astronomical Observatory, France.",
                  ("2015A&A...578A.114F", "2017ivoa.spec.0519F"), "https://aladin.cds.unistra.fr/hips/",
                  note="generic sentence: CDS prescribes no acknowledgement for HiPS itself (cite Fernique et al. 2015 and "
                       "the IVOA HiPS standard); for cutouts made with hips2fits use the 'hips2fits' entry. Also cite the "
                       "survey whose HiPS image is shown (e.g. 2MASS, Pan-STARRS, WISE)."),
    CitationEntry("hips2fits", "CDS hips2fits cutout service", "service",
                  "This research made use of hips2fits,\\footnote{https://alasky.cds.unistra.fr/hips-image-services/hips2fits} "
                  "a service provided by CDS.",
                  ("2015A&A...578A.114F",), "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                  note="Verbatim, including the LaTeX footnote, as prescribed on the hips2fits page. hips2fits prescribes "
                       "no paper; Fernique et al. 2015 describes the HiPS surveys it cuts. Also cite the survey whose "
                       "image is shown."),
    CitationEntry("svo_fps", "SVO Filter Profile Service", "service",
                  "This research has made use of the SVO Filter Profile Service \"Carlos Rodrigo\", funded by "
                  "MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00.",
                  ("2024A&A...689A..93R", "2012ivoa.rept.1015R", "2020sea..confE.182R"),
                  "http://svo2.cab.inta-csic.es/theory/fps/"),
    CitationEntry("skybot", "SkyBoT (IMCCE)", "service", "This research has made use of LTE SkyBoT VO tool.",
                  ("2006ASPC..351..367B", "2016MNRAS.458.3394B"), "https://ssp.imcce.fr/webservices/skybot/"),
    CitationEntry("ztf", "Zwicky Transient Facility", "service", _ZTF_ACK,
                  ("2019PASP..131a8003M", "2019PASP..131a8002B", "2019PASP..131g8001G"), _ZTF_ACK_URL,
                  note="Verbatim from section 1.a of the ZTF DR24 release notes (one stray space before a comma, "
                       "a PDF text artifact, removed). The notes ask to cite Masci et al. (2019PASP..131a8003M); "
                       "Bellm et al. and Graham et al. describe the survey. ZTF data are served by IRSA, whose "
                       "acknowledgement is added automatically."),
    CitationEntry("alerce", "ALeRCE alert broker", "service",
                  "This work has made use of the ALeRCE alert broker.",
                  ("2021AJ....161..242F",), "https://alerce.science/",
                  note="generic sentence: ALeRCE publishes no fixed acknowledgement; cite Forster et al. 2021."),
    CitationEntry("neowise", "NEOWISE", "service", _NEOWISE_ACK,
                  ("2011ApJ...731...53M", "2014ApJ...792...30M", "2010AJ....140.1868W"), _NEOWISE_ACK_URL),
    CitationEntry("tess", "TESS (SPOC light curves, MAST)", "service", _TESS_ACK,
                  ("2015JATIS...1a4003R", "2016SPIE.9913E..3EJ"), _MAST_ACK_URL,
                  note="Verbatim from MAST's mission acknowledgements. Ricker et al. 2015 describes the mission; "
                       "Jenkins et al. 2016 the SPOC pipeline that produced the 2-min light curves."),
    CitationEntry("galex", "GALEX", "service", _GALEX_ACK,
                  ("2005ApJ...619L...1M", "2007ApJS..173..682M", "2017ApJS..230...24B"), _MAST_ACK_URL,
                  note="MAST's legacy-mission template with 'Galaxy Evolution Explorer' filled in. Martin et al. 2005 "
                       "(mission), Morrissey et al. 2007 (calibration: AB zero points), Bianchi et al. 2017 (GUVcat, "
                       "the GALEX source catalogue the SED feature reads)."),
    CitationEntry("unwise", "unWISE coadds", "service",
                  "This research has made use of the unWISE coadds of the WISE and NEOWISE imaging.",
                  ("2014AJ....147..108L", "2017AJ....154..161M", "2022RNAAS...6..188M"), "https://unwise.me/",
                  note="generic sentence: unwise.me asks users to cite Lang 2014 and Meisner et al. (the 2022 note "
                       "describes the neo8 coadds served as CDS/P/unWISE HiPS) and prescribes no acknowledgement; "
                       "images shown need the attribution 'unWISE / NASA/JPL-Caltech / D. Lang (Perimeter "
                       "Institute)'. Also acknowledge WISE/NEOWISE (the 'allwise' and 'neowise' entries)."),
    CitationEntry("astropy", "Astropy", "software",
                  "This work made use of Astropy: \\footnote{https://www.astropy.org} a community-developed core Python "
                  "package and an ecosystem of tools and resources for astronomy "
                  "\\citep{2013A&A...558A..33A, 2018AJ....156..123A, 2022ApJ...935..167A}.",
                  ("2013A&A...558A..33A", "2018AJ....156..123A", "2022ApJ...935..167A"), "https://www.astropy.org/acknowledging.html",
                  note="The LaTeX snippet of astropy.org/acknowledging.html (read 2026-09-28) word for word, except that the "
                       "\\citep keys are those of the BibTeX written here (the snippet names them astropy:2013, "
                       "astropy:2018, astropy:2022)."),
    CitationEntry("healpix", "HEALPix", "software", "Some of the results in this paper have been derived using the HEALPix package.",
                  ("2005ApJ...622..759G",), "https://healpix.sourceforge.io/"),
]}

# Other names a user or feature may pass for the same entry.
CITATION_ALIASES: dict[str, str] = {
    "gaia": "gaia_dr3", "gaiadr3": "gaia_dr3", "2mass": "twomass_psc", "twomass": "twomass_psc", "wise": "allwise",
    "panstarrs": "panstarrs_dr2", "ps1": "panstarrs_dr2", "pan-starrs": "panstarrs_dr2", "sdss_dr18": "sdss",
    "exoplanets": "exoplanet_archive", "nasa_exoplanet_archive": "exoplanet_archive", "csc": "chandra",
    "4xmm": "xmm", "5xmm": "xmm", "xmm_newton": "xmm", "2rxs": "rosat", "1rxs": "rosat_bsc", "xmatch": "cds_xmatch",
    "cds-xmatch": "cds_xmatch", "cutouts": "hips2fits", "cutout": "hips2fits", "svo": "svo_fps", "filters": "svo_fps",
    "solar_system": "skybot", "solar-system": "skybot", "lightcurves_ztf": "ztf", "alerts": "alerce",
    "tess_spoc": "tess", "galex_fuv": "galex", "galex_nuv": "galex", "guvcat": "galex", "unwise_w1": "unwise",
    "unwise_w2": "unwise",
}

# Archives implied by a catalog's service endpoint (their acknowledgement is required too).
_HOST_ARCHIVES: tuple[tuple[tuple[str, ...], str], ...] = (
    (VIZIER_HOSTS, "vizier"), (HEASARC_HOSTS, "heasarc"), (IRSA_HOSTS, "irsa"),
)
# Archives serving the data of services used by other features (ZTF and NEOWISE light
# curves come from IRSA).
_SERVICE_ARCHIVES: dict[str, str] = {"ztf": "irsa", "neowise": "irsa"}  # TESS/GALEX texts already name MAST


def _archive_for_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    host = httpx.URL(endpoint).host
    for hosts, key in _HOST_ARCHIVES:
        if host in hosts:
            return key
    return None


def known_citation_keys() -> list[str]:
    return sorted(CITATION_ENTRIES)


def resolve_citation_keys(names: Iterable[str], registry: CatalogRegistry | None = None) -> list[str]:
    """Normalize names (aliases, case) and add hosting archives; raises UnknownCitationError.

    A registry catalog without a curated entry is accepted when ``registry`` is given
    (its registry citation/acknowledgement is used).
    """
    keys, unknown = _resolve_citation_keys(names, registry)
    if unknown:
        raise UnknownCitationError(f"no citation entry for {sorted(set(unknown))}; known: {known_citation_keys()}")
    return keys


def _resolve_citation_keys(names: Iterable[str], registry: CatalogRegistry | None) -> tuple[list[str], list[str]]:
    """(citation keys incl. hosting archives, names without an entry)."""
    keys: list[str] = []
    unknown: list[str] = []
    reg_catalogs = registry.catalogs if registry is not None else {}
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        low = name.lower()
        key = low if low in CITATION_ENTRIES else CITATION_ALIASES.get(low) or (name if name in reg_catalogs else None)
        if key is None:
            unknown.append(name)
            continue
        keys.append(key)
        definition = reg_catalogs.get(key)
        endpoint = definition.endpoint if definition is not None else (
            _DEFAULT_ENDPOINTS.get(key) if key in _DEFAULT_ENDPOINTS else None
        )
        archive = _archive_for_endpoint(endpoint) or _SERVICE_ARCHIVES.get(key)
        if archive:
            keys.append(archive)
    return list(dict.fromkeys(keys)), sorted(set(unknown))


def _default_endpoints() -> dict[str, str]:
    from models import DEFAULT_CATALOGS

    return {name: str(entry.get("endpoint")) for name, entry in DEFAULT_CATALOGS.items() if entry.get("endpoint")}


_DEFAULT_ENDPOINTS: dict[str, str] = _default_endpoints()

# Sentence boundary: ". " + capital, but not after an initial or abbreviation such as
# "Alfred P. Sloan", "U.S. Department", "Grant No. NNX08AR22G" or "et al. The".
_SENTENCE_SPLIT = re.compile(r"(?<!\b[A-Z]\.)(?<!\bNo\.)(?<!\bal\.)(?<!\be\.V\.)(?<=[.!?])\s+(?=[A-Z])")


def _dedupe_sentences(texts: Iterable[str]) -> str:
    seen: set[str] = set()
    sentences: list[str] = []
    for text in texts:
        for sentence in _SENTENCE_SPLIT.split(text.strip()):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                sentences.append(sentence)
    return " ".join(sentences)


@dataclass(slots=True)
class CitationBundle:
    """Acknowledgements and references for a set of catalogs/services."""

    keys: list[str]
    acknowledgements: list[dict[str, Any]]
    references: list[Reference]
    unknown: list[str] = field(default_factory=list)  # names without a citation entry (strict=False)

    @property
    def acknowledgement_text(self) -> str:
        """One paragraph for a paper's acknowledgements (repeated sentences removed)."""
        return _dedupe_sentences(a["text"] for a in self.acknowledgements if a.get("text"))

    @property
    def bibtex(self) -> str:
        return to_bibtex(self.references)

    def as_dict(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "acknowledgements": list(self.acknowledgements),
            "acknowledgement_text": self.acknowledgement_text,
            "references": [r.as_dict() for r in self.references],
            "bibtex": self.bibtex,
            "unknown": list(self.unknown),
        }


_DOI_TEXT = re.compile(r"\b10\.\d{4,9}/[^\s;,()]+[^\s;,().]")


def unverified_dois(text: str | None) -> list[str]:
    """DOIs named in free text (a registry citation) that are not verified references.

    E.g. the registry's Chandra citation names 10.25574/csc2.1, which doi.org does not
    know (HTTP 404; the verified CSC Release 2 series DOI is 10.25574/csc2).
    """
    verified = {r.doi.lower() for r in REFERENCES.values() if r.doi}
    return sorted({d for d in _DOI_TEXT.findall(text or "") if d.lower() not in verified})


# A 19-character ADS bibcode (YYYYJJJJJVVVVMPPPPA) inside free text, e.g. the registry citation
# 'Gaia Collaboration 2018, A&A 616, A1 (2018A&A...616A...1G); VizieR I/345 ...' that
# `vizier add` writes from the VizieR/IVOA registry metadata.
_BIBCODE_IN_TEXT = re.compile(r"(?<![\w&.])((?:1[89]|20)\d\d[A-Za-z&][A-Za-z&.]{4}[A-Za-z0-9.]{4}[A-Za-z.][A-Za-z0-9.]{4}[A-Z.])"
                              r"(?![\w&])")
# What precedes '(bibcode)' in such a citation: '<authors> <year>, <journal> <volume>, <page>'.
_CITATION_HEAD = re.compile(r"(?P<authors>[^;()]+?)\s+(?P<year>(?:1[89]|20)\d\d)[a-z]?\s*,\s*(?P<where>[^;()]*?)\s*\($")
#: Key under which `vizier add` stores the references it resolved through ADS/doi.org in a
#: registry entry's ``parameters`` (see :func:`lookup_reference`).
REGISTRY_REFERENCES_KEY = "citation_references"


def bibcodes_in(text: str | None) -> list[str]:
    """The ADS bibcodes named in free text, in order, without repeats."""
    return list(dict.fromkeys(_BIBCODE_IN_TEXT.findall(text or "")))


def reference_from_citation_text(bibcode: str, text: str) -> Reference:
    """An unverified reference for ``bibcode`` from the citation text that names it: authors and
    'journal volume, page' as written there, year from the bibcode. BibTeX ``@misc`` (no title is
    known offline); :func:`lookup_reference` resolves the full record through ADS and doi.org."""
    known = REFERENCES.get(bibcode)
    if known is not None:
        return known
    authors: tuple[str, ...] = ()
    where = ""
    position = text.find(bibcode)
    if position > 0:
        head = _CITATION_HEAD.search(text[:position])
        if head:
            who = head.group("authors").strip().rstrip(",")
            more = who.endswith(" et al.")
            who = who.removesuffix(" et al.").strip()
            authors = tuple("{" + part.strip() + "}" for part in re.split(r",\s*|\s+&\s+|\s+and\s+", who) if part.strip())
            where = head.group("where").strip().rstrip(",")
            if more:
                return Reference(key=bibcode, entry_type="misc", authors=authors, title="", year=int(bibcode[:4]),
                                 bibcode=bibcode, howpublished=where or None, more_authors=True, verified=False,
                                 note="unverified: parsed from the catalog registry citation (resolve it with "
                                      "'astrosearch cite --verify' or /api/v1/citations?verify=true)")
    return Reference(key=bibcode, entry_type="misc", authors=authors, title="", year=int(bibcode[:4]),
                     bibcode=bibcode, howpublished=where or f"ADS {bibcode}", verified=False,
                     note="unverified: parsed from the catalog registry citation (resolve it with "
                          "'astrosearch cite --verify' or /api/v1/citations?verify=true)")


def _csl_person(person: Mapping[str, Any]) -> str | None:
    family = str(person.get("family") or "").strip()
    given = str(person.get("given") or "").strip()
    if family:
        initials = " ".join(f"{part[0]}." for part in re.split(r"[\s.]+", given) if part) if given else ""
        return f"{family}, {initials}".strip().rstrip(",")
    literal = str(person.get("literal") or person.get("name") or "").strip()
    return "{" + literal + "}" if literal else None


def reference_from_csl(bibcode: str | None, doi: str, csl: Mapping[str, Any]) -> Reference:
    """A verified reference from a DOI's CSL-JSON (doi.org): at most four authors listed (then
    'and others'), as the curated references."""
    people = [name for name in (_csl_person(p) for p in csl.get("author") or [] if isinstance(p, Mapping)) if name]
    issued = (csl.get("issued") or csl.get("published-print") or csl.get("published-online") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] and issued[0][0] else (int(bibcode[:4]) if bibcode else 0)

    def text(value: Any) -> str | None:
        """Plain text of a CSL string (Crossref titles carry JATS/HTML markup and entities)."""
        if isinstance(value, list):
            value = value[0] if value else None
        if value in (None, ""):
            return None
        plain = html.unescape(re.sub(r"<[^>]+>", "", str(value)))
        return re.sub(r"\s+", " ", plain).strip() or None

    kind = str(csl.get("type") or "")
    entry_type = "article" if kind in ("journal-article", "article-journal", "article") else (
        "inproceedings" if kind in ("proceedings-article", "paper-conference") else "misc")
    journal = text(csl.get("container-title"))
    title = text(csl.get("title")) or ""
    subtitle = text(csl.get("subtitle"))
    if title and subtitle and subtitle.lower() not in title.lower():
        title = f"{title}. {subtitle}"
    if entry_type == "article" and not journal:
        entry_type = "misc"
    if entry_type == "inproceedings" and not journal:
        entry_type = "misc"
    return Reference(
        key=bibcode or doi, entry_type=entry_type, authors=tuple(people[:4]), title=re.sub(r"\s+", " ", title),
        year=int(year), bibcode=bibcode, journal=journal if entry_type == "article" else None,
        booktitle=journal if entry_type == "inproceedings" else None, volume=text(csl.get("volume")),
        pages=text(csl.get("page")) or text(csl.get("article-number")), doi=doi, more_authors=len(people) > 4,
        publisher=text(csl.get("publisher")) if entry_type == "misc" else None,
        howpublished=None if title else journal,
    )


async def lookup_reference(client: httpx.AsyncClient, bibcode: str) -> Reference | None:
    """Resolve a bibcode to a verified reference through the verification path: the ADS link
    gateway gives the publisher DOI, whose CSL-JSON (doi.org) gives authors, title, journal,
    volume, pages and year. None when ADS gives no DOI or the DOI is not registered; network
    errors propagate (httpx.HTTPError)."""
    if bibcode in REFERENCES:
        return REFERENCES[bibcode]
    probe = Reference(key=bibcode, entry_type="misc", authors=(), title="", year=int(bibcode[:4]), bibcode=bibcode)
    check = await verify_reference(client, probe, check_metadata=False)
    if not check.exists or not check.doi_from_ads:
        return None
    csl = await fetch_csl(client, check.doi_from_ads)
    if csl is None:
        return None
    return reference_from_csl(bibcode, check.doi_from_ads, csl)


def reference_to_registry(ref: Reference) -> dict[str, Any]:
    """A resolved reference as stored in a registry entry (``parameters[REGISTRY_REFERENCES_KEY]``)."""
    return {k: v for k, v in asdict(ref).items() if v not in (None, "", False) or k in ("year",)} | {
        "authors": list(ref.authors), "verified": ref.verified}


def reference_from_registry(data: Mapping[str, Any]) -> Reference | None:
    """The reference :func:`reference_to_registry` stored, or None when it is malformed."""
    try:
        fields_ = {f: data[f] for f in Reference.__dataclass_fields__ if f in data}
        fields_["authors"] = tuple(str(a) for a in data.get("authors") or ())
        fields_["year"] = int(data["year"])
        return Reference(**fields_)
    except (KeyError, TypeError, ValueError):
        return None


def registry_references(definition: CatalogDefinition) -> list[Reference]:
    """The papers a registry catalog's citation names: the references `vizier add` resolved
    through ADS/doi.org when it registered the table (verified), else those parsed from the
    citation text (curated ones verified, others marked unverified)."""
    stored = (definition.parameters or {}).get(REGISTRY_REFERENCES_KEY)
    resolved = {ref.bibcode: ref for ref in (reference_from_registry(item) for item in stored or []
                                            if isinstance(item, Mapping)) if ref is not None and ref.bibcode}
    refs: list[Reference] = []
    for bibcode in bibcodes_in(definition.citation):
        refs.append(resolved.get(bibcode) or reference_from_citation_text(bibcode, definition.citation or ""))
    for bibcode, ref in resolved.items():  # stored but no longer named in the text: still cited
        if all(r.bibcode != bibcode for r in refs):
            refs.append(ref)
    return refs


async def resolve_unverified(client: httpx.AsyncClient, bundle: CitationBundle) -> list[str]:
    """Replace the bundle's unverified references (parsed from registry citations) by records
    resolved through ADS/doi.org (:func:`lookup_reference`). Returns a note per reference that
    stayed unverified."""
    notes: list[str] = []
    for n, ref in enumerate(list(bundle.references)):
        if ref.verified or not ref.bibcode:
            continue
        try:
            found = await lookup_reference(client, ref.bibcode)
        except (httpx.HTTPError, UnparsableAnswerError) as exc:
            notes.append(f"{ref.bibcode}: not resolved ({exc.__class__.__name__}: {exc})")
            continue
        if found is None:
            notes.append(f"{ref.bibcode}: ADS gives no DOI for it (kept unverified)")
            continue
        bundle.references[n] = found
        for ack in bundle.acknowledgements:
            if ref.bibcode in (ack.get("bibcodes") or []):
                ack["references"] = [found.short if r == ref.short else r for r in ack.get("references") or []]
                ack["references_verified"] = all(r.verified for r in bundle.references if r.bibcode in ack["bibcodes"])
                if found.doi and found.doi not in (ack.get("dois") or []):
                    ack.setdefault("dois", []).append(found.doi)
    return notes


def _without_vizier_ack(text: str) -> str:
    """A registry acknowledgement without its VizieR sentence (the curated 'vizier' archive entry,
    added for every VizieR catalog, carries CDS's own wording)."""
    kept = [sentence for sentence in _SENTENCE_SPLIT.split(text.strip())
            if not re.search(r"VizieR catalogue access tool|description of the VizieR service", sentence)]
    return " ".join(s.strip() for s in kept if s.strip())


def citations_for(names: Iterable[str], registry: CatalogRegistry | None = None, *,
                  strict: bool = True) -> CitationBundle:
    """Acknowledgement texts and verified references for catalogs/services ``names``.

    ``strict`` (default): any name without a citation entry raises
    :class:`UnknownCitationError`. ``strict=False``: such names are listed in
    ``bundle.unknown`` and the others are cited (UnknownCitationError only when none is
    known), so a page citing several surveys still gets the citations it can have.

    Archives that host a catalog (VizieR, HEASARC, IRSA) are added automatically. When
    ``registry`` is given, a registry catalog's own citation/acknowledgement text is
    reported alongside the curated one as ``registry_citation`` (and used for catalogs
    without a curated entry); it is free text from the catalog registry and is marked
    ``registry_citation_verified: False``, with any DOI it names that is not a verified
    reference listed in ``registry_citation_unverified_dois``.
    """
    names = list(names)
    keys, unknown = _resolve_citation_keys(names, registry)
    if unknown and (strict or not keys):
        raise UnknownCitationError(f"no citation entry for {unknown}; known: {known_citation_keys()}")
    acks: list[dict[str, Any]] = []
    ref_keys: list[str] = []
    extra_refs: list[Reference] = []
    reg = registry.catalogs if registry is not None else {}
    for key in keys:
        entry = CITATION_ENTRIES.get(key)
        definition = reg.get(key)
        registry_citation = definition.citation if definition is not None else None
        if entry is not None and definition is not None and key in CATALOG_RELEASES and not matches_curated(key, definition):
            # The deployment registry points this name at another service/table: the curated
            # (verified) entry does not describe it; report the registry text, unverified.
            acks.append({
                "key": key, "catalog": key, "name": definition.description or key, "kind": "catalog",
                "text": definition.acknowledgement or "", "references": [], "bibcodes": [], "dois": [],
                "source_url": None,
                "note": (f"the registry definition of {key!r} differs from the service/table the curated entry "
                         f"({entry.name}) was verified for: registry citation text only (not verified)"),
                "registry_citation": registry_citation,
                "registry_citation_verified": False if registry_citation else None,
                "registry_citation_unverified_dois": unverified_dois(registry_citation),
                "registry_acknowledgement": definition.acknowledgement,
            })
        elif entry is not None:
            dois = [REFERENCES[r].doi for r in entry.references if REFERENCES[r].doi]
            acks.append({
                "key": key, "catalog": key, "name": entry.name, "kind": entry.kind, "text": entry.acknowledgement,
                "references": [REFERENCES[r].short for r in entry.references],
                "bibcodes": [REFERENCES[r].bibcode for r in entry.references if REFERENCES[r].bibcode],
                "dois": dois,
                "source_url": entry.source_url, "note": entry.note,
                "registry_citation": registry_citation,
                "registry_citation_verified": False if registry_citation else None,
                "registry_citation_unverified_dois": unverified_dois(registry_citation),
                "registry_acknowledgement": definition.acknowledgement if definition is not None else None,
            })
            ref_keys.extend(entry.references)
        elif definition is not None:
            papers = registry_references(definition)
            extra_refs.extend(papers)
            verified = all(ref.verified for ref in papers) if papers else None
            text = definition.acknowledgement or ""
            if "vizier" in keys:
                text = _without_vizier_ack(text)
            if papers:
                note = ("no curated entry: the paper(s) named in the registry citation"
                        + (" (resolved through ADS/doi.org)" if verified else
                           " (not all verified: resolve them with 'astrosearch cite --verify' or verify=true)"))
            else:
                note = "no curated entry: registry citation text only (it names no bibcode; not verified)"
            acks.append({
                "key": key, "catalog": key, "name": definition.description or key, "kind": "catalog",
                "text": text, "references": [ref.short for ref in papers],
                "bibcodes": [ref.bibcode for ref in papers if ref.bibcode],
                "dois": [ref.doi for ref in papers if ref.doi],
                "references_verified": verified,
                "source_url": None, "note": note,
                "registry_citation": registry_citation,
                "registry_citation_verified": False if registry_citation else None,
                "registry_citation_unverified_dois": unverified_dois(registry_citation),
                "registry_acknowledgement": definition.acknowledgement,
            })
    refs = [REFERENCES[k] for k in dict.fromkeys(ref_keys)]
    known = {ref.key for ref in refs}
    for ref in extra_refs:
        if ref.key not in known:
            known.add(ref.key)
            refs.append(ref)
    return CitationBundle(keys=keys, acknowledgements=acks, references=refs, unknown=unknown)


def _rows_of(entry: Mapping[str, Any]) -> int:
    count = entry.get("row_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    sources = entry.get("sources")
    return len(sources) if isinstance(sources, list) else 0


def catalogs_in(obj: UnifiedRecord | ProvenanceManifest | Mapping[str, Any], *, include_empty: bool = False) -> list[str]:
    """Catalog names whose data a record or manifest used: those that returned rows.

    A catalog that answered with no row inside the radius contributed no data, so it is
    not acknowledged unless ``include_empty`` (for authors who report non-detections).
    Failed catalogs are never included.
    """
    def used(status: Any, rows: int) -> bool:
        return status != "failed" and (rows > 0 or include_empty)

    if isinstance(obj, ProvenanceManifest):
        return sorted(n for n, r in obj.catalogs.items() if used(r.status, r.row_count))
    schema = obj.get("schema") if isinstance(obj, Mapping) else None
    if isinstance(schema, str) and schema.startswith("astrosearch.provenance.manifest/") and schema != MANIFEST_SCHEMA:
        raise ManifestError(f"unsupported manifest schema {schema!r} (expected {MANIFEST_SCHEMA!r})")
    if isinstance(obj, Mapping) and schema == MANIFEST_SCHEMA:
        cats = obj.get("catalogs")
        if not isinstance(cats, Mapping) or not all(isinstance(r, Mapping) for r in cats.values()):
            raise ManifestError("manifest catalogs must be an object of catalog entries")
        return sorted(str(n) for n, r in cats.items() if used(r.get("status"), _rows_of(r)))
    data = record_to_dict(obj)
    return sorted(n for n, r in (data.get("catalog_results") or {}).items()
                  if used((r or {}).get("status"), _rows_of(r or {})))


# ---------------------------------------------------------------------------
# BibTeX: Writer & Validator
# ---------------------------------------------------------------------------

_LATEX_ACCENTS = {
    "á": "{\\'a}", "é": "{\\'e}", "í": "{\\'i}", "ó": "{\\'o}", "ú": "{\\'u}", "Á": "{\\'A}", "É": "{\\'E}",
    "à": "{\\`a}", "è": "{\\`e}", "ò": "{\\`o}", "ä": '{\\"a}', "ë": '{\\"e}', "ï": '{\\"i}', "ö": '{\\"o}',
    "ü": '{\\"u}', "Ä": '{\\"A}', "Ö": '{\\"O}', "Ü": '{\\"U}', "ñ": "{\\~n}", "ç": "{\\c{c}}", "ő": "{\\H{o}}",
    "ű": "{\\H{u}}", "ø": "{\\o}", "å": "{\\aa}", "ß": "{\\ss}", "č": "{\\v{c}}", "š": "{\\v{s}}", "ž": "{\\v{z}}",
}
_SPECIAL = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_", "$": "\\$"}


def latex_escape(text: str, *, keep_braces: bool = True) -> str:
    """Escape LaTeX specials (& % # _ $) and write common accented letters as macros."""
    out = []
    for ch in text:
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _LATEX_ACCENTS:
            out.append(_LATEX_ACCENTS[ch])
        elif not keep_braces and ch in "{}":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _bib_field(name: str, value: str) -> str:
    return f"  {name:<9} = {{{value}}}"


def reference_to_bibtex(ref: Reference) -> str:
    """One BibTeX entry (key = bibcode, as ADS exports) with LaTeX-safe field values."""
    authors = [latex_escape(a) for a in ref.authors] + (["others"] if ref.more_authors else [])
    fields: list[tuple[str, str]] = []
    if authors:
        fields.append(("author", " and ".join(authors)))
    if ref.title:
        fields.append(("title", "{" + latex_escape(ref.title) + "}"))
    if ref.howpublished:
        fields.append(("howpublished", latex_escape(ref.howpublished)))
    if ref.journal:
        fields.append(("journal", latex_escape(ref.journal)))
    if ref.booktitle:
        fields.append(("booktitle", latex_escape(ref.booktitle)))
    if ref.series:
        fields.append(("series", latex_escape(ref.series)))
    fields.append(("year", str(ref.year)))
    for name in ("volume", "pages"):
        value = getattr(ref, name)
        if value:
            fields.append((name, latex_escape(value)))
    if ref.publisher:
        fields.append(("publisher", latex_escape(ref.publisher)))
    if ref.institution:
        fields.append(("institution", latex_escape(ref.institution)))
    if ref.eprint:
        fields.extend([("eprint", ref.eprint), ("archivePrefix", "arXiv")])
    if ref.doi:
        fields.append(("doi", ref.doi))  # verbatim: styles put it in a doi/href macro; ADS exports it unescaped
    if ref.url:
        fields.append(("url", ref.url))
    if ref.bibcode:
        fields.append(("adsurl", ref.ads_url or ""))
    if ref.note:
        fields.append(("note", latex_escape(ref.note)))
    body = ",\n".join(_bib_field(n, v) for n, v in fields)
    return f"@{ref.entry_type}{{{ref.key},\n{body}\n}}\n"


def to_bibtex(references: Iterable[Reference]) -> str:
    """BibTeX text for ``references`` (duplicates by key removed, order kept)."""
    seen: dict[str, Reference] = {}
    for ref in references:
        seen.setdefault(ref.key, ref)
    return "\n".join(reference_to_bibtex(r) for r in seen.values())


REQUIRED_BIBTEX_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    # Each inner tuple: at least one of these fields must be present (BibTeX standard styles).
    "article": (("author",), ("title",), ("journal",), ("year",)),
    "inproceedings": (("author",), ("title",), ("booktitle",), ("year",)),
    "incollection": (("author",), ("title",), ("booktitle",), ("publisher",), ("year",)),
    "techreport": (("author",), ("title",), ("institution",), ("year",)),
    "misc": (("title", "howpublished"),),
    "book": (("author", "editor"), ("title",), ("publisher",), ("year",)),
    "phdthesis": (("author",), ("title",), ("school",), ("year",)),
}


@dataclass(slots=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


def _read_value(text: str, i: int) -> tuple[str, int]:
    """Read a BibTeX field value starting at ``i``: {...}, "..." or a bare number/macro."""
    n = len(text)
    if text[i] == "{":
        depth, j = 0, i
        while j < n:
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j], j + 1
            j += 1
        raise BibTeXError(f"unbalanced braces in value starting at offset {i}")
    if text[i] == '"':
        depth, j = 0, i + 1
        while j < n:
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    raise BibTeXError(f"unbalanced braces in quoted value at offset {i}")
            elif ch == '"' and depth == 0:
                return text[i + 1:j], j + 1
            j += 1
        raise BibTeXError(f"unterminated quoted value at offset {i}")
    m = re.match(r"[A-Za-z0-9_.:+-]+", text[i:])
    if not m:
        raise BibTeXError(f"invalid field value at offset {i}: {text[i:i + 20]!r}")
    return m.group(0), i + m.end()


def _check_value(key: str, name: str, value: str) -> None:
    depth = 0
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise BibTeXError(f"{key}.{name}: unbalanced closing brace")
        elif ch in "&%#" and name not in {"url", "adsurl", "doi", "eprint"}:
            raise BibTeXError(f"{key}.{name}: unescaped '{ch}' (write \\{ch})")
        i += 1
    if depth != 0:
        raise BibTeXError(f"{key}.{name}: unbalanced braces")


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse and validate BibTeX; raises :class:`BibTeXError` on the first problem.

    Checks: entry syntax (``@type{key, name = value, ...}``, fields separated by commas
    as bibtex/biber require; a trailing comma is allowed), balanced braces, unique
    keys, no repeated fields, required fields per entry type
    (:data:`REQUIRED_BIBTEX_FIELDS`), four-digit years, and no unescaped ``& % #`` in
    text fields (they break LaTeX when the bibliography is typeset).
    """
    depth = 0
    for pos, ch in enumerate(text):
        if ch in "{}" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                raise BibTeXError(f"unbalanced braces: unexpected '}}' at offset {pos}")
    if depth:
        raise BibTeXError(f"unbalanced braces: {depth} '{{' not closed")
    entries: list[BibEntry] = []
    keys: set[str] = set()
    i, n = 0, len(text)
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@([A-Za-z]+)\s*([{(])", text[at:])
        if not m:
            raise BibTeXError(f"malformed entry header at offset {at}: {text[at:at + 30]!r}")
        entry_type = m.group(1).lower()
        closer = "}" if m.group(2) == "{" else ")"
        j = at + m.end()
        if entry_type in {"comment", "preamble", "string"}:
            depth = 1
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0) if closer == "}" else {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            if depth:
                raise BibTeXError(f"unterminated @{entry_type} at offset {at}")
            i = j
            continue
        km = re.match(r"\s*([^\s,{}()=\"]+)\s*,", text[j:])
        if not km:
            raise BibTeXError(f"@{entry_type} at offset {at} has no citation key")
        key = km.group(1)
        if key in keys:
            raise BibTeXError(f"duplicate citation key {key!r}")
        keys.add(key)
        j += km.end()
        fields: dict[str, str] = {}
        while True:
            ws = re.match(r"\s*", text[j:])
            j += ws.end() if ws else 0
            if j >= n:
                raise BibTeXError(f"entry {key!r} is not closed")
            if text[j] == closer:  # also after a trailing comma, which BibTeX accepts
                j += 1
                break
            fm = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", text[j:])
            if not fm:
                raise BibTeXError(f"entry {key!r}: malformed field at offset {j}: {text[j:j + 30]!r}")
            name = fm.group(1).lower()
            j += fm.end()
            if j >= n:
                raise BibTeXError(f"entry {key!r}: field {name!r} has no value")
            value, j = _read_value(text, j)
            while True:  # string concatenation with '#'
                cm = re.match(r"\s*#\s*", text[j:])
                if not cm:
                    break
                j += cm.end()
                more, j = _read_value(text, j)
                value += more
            if name in fields:
                raise BibTeXError(f"entry {key!r}: field {name!r} repeated")
            _check_value(key, name, value)
            fields[name] = value
            # bibtex/biber: "I was expecting a `,' or a `}'" -- fields are comma-separated.
            sep = re.match(r"\s*", text[j:])
            j += sep.end() if sep else 0
            if j >= n:
                raise BibTeXError(f"entry {key!r} is not closed")
            if text[j] == ",":
                j += 1
            elif text[j] != closer:
                raise BibTeXError(f"entry {key!r}: expected ',' or '{closer}' after field {name!r} at offset {j}, "
                                  f"found {text[j:j + 20]!r}")
        for group in REQUIRED_BIBTEX_FIELDS.get(entry_type, ()):
            if not any(fields.get(f, "").strip() for f in group):
                raise BibTeXError(f"@{entry_type}{{{key}}} lacks required field {' or '.join(group)}")
        if "year" in fields and not re.fullmatch(r"\d{4}", fields["year"].strip()):
            raise BibTeXError(f"entry {key!r}: year {fields['year']!r} is not a four-digit year")
        entries.append(BibEntry(entry_type, key, fields))
        i = j
    if not entries and text.strip():
        raise BibTeXError("no BibTeX entries found")
    return entries


# ---------------------------------------------------------------------------
# Live Reference Verification (NASA ADS link gateway + doi.org)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReferenceCheck:
    """Result of checking one reference against ADS and its DOI registration."""

    key: str
    bibcode: str | None
    exists: bool | None  # None = could not decide (service unreachable)
    link_type: str | None = None
    resolved_url: str | None = None
    doi_from_ads: str | None = None
    doi_matches: bool | None = None
    metadata: dict[str, Any] | None = None
    metadata_matches: bool | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.exists) and self.doi_matches is not False and self.metadata_matches is not False

    @property
    def undecided(self) -> bool:
        """A service problem (unreachable, 5xx, unparsable answer) left a check open."""
        return self.exists is None or any(p.startswith(("unreachable", "unparsable", "ADS gateway")) for p in self.problems)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        out["undecided"] = self.undecided
        return out


def _first_page(pages: Any) -> str | None:
    if pages in (None, ""):
        return None
    return re.split(r"[-–]+", str(pages))[0].strip().lstrip("0") or "0"


class UnparsableAnswerError(ValueError):
    """A service answered 2xx with something that is not the requested format."""


async def fetch_csl(client: httpx.AsyncClient, doi: str) -> dict[str, Any] | None:
    """CSL-JSON metadata of a DOI via doi.org content negotiation (None when not registered).

    Raises :class:`UnparsableAnswerError` when the answer is not a JSON object (e.g. a
    registration agency's HTML landing page served with 200 instead of CSL-JSON).
    """
    response = await client.get(DOI_RESOLVER_URL.format(doi=doi), headers={"Accept": CSL_JSON, "User-Agent": USER_AGENT},
                                follow_redirects=True)
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(f"doi.org HTTP {response.status_code}", request=response.request, response=response)
    ctype = response.headers.get("content-type", "")
    try:
        data = response.json()
    except ValueError as exc:
        raise UnparsableAnswerError(f"doi.org answered HTTP {response.status_code} with {ctype or 'no content type'}, "
                                    "not CSL-JSON") from exc
    if not isinstance(data, dict):
        raise UnparsableAnswerError(f"doi.org answered a JSON {type(data).__name__}, not a CSL-JSON object")
    return data


def _compare_csl(ref: Reference, csl: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    problems: list[str] = []
    issued = (csl.get("issued") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] else None
    meta = {"title": csl.get("title"), "year": year, "volume": csl.get("volume"), "page": csl.get("page"),
            "container": csl.get("container-title"), "doi": csl.get("DOI")}
    # Journals sometimes date the online version one year earlier than the issue (ZTF
    # PASP papers: online Dec 2018, issue/ADS 2019); accept +-1 year for articles.
    if year is not None and abs(int(year) - ref.year) > (1 if ref.entry_type == "article" else 0):
        problems.append(f"year {year} != {ref.year}")
    if ref.volume and csl.get("volume") and str(csl["volume"]).strip() != ref.volume:
        problems.append(f"volume {csl['volume']} != {ref.volume}")
    csl_page = _first_page(csl.get("page") or csl.get("article-number"))
    if (ref.pages and csl_page and csl_page != _first_page(ref.pages)
            and not _article_number_in_doi(ref.pages, csl.get("DOI") or ref.doi)):
        problems.append(f"first page {csl_page} != {_first_page(ref.pages)}")
    return not problems, problems, meta


def _article_number_in_doi(pages: str, doi: str | None) -> bool:
    """True when ``pages`` is an article number (elocation id) that the DOI itself ends with.

    Journals with article numbers (SPIE JATIS, AIP, APS, IOP) register a page range *inside*
    the article with Crossref (live 2026-09-29: 10.1117/1.JATIS.1.1.014003 has page '1-10'),
    while ADS and the citation use the article number 014003; the DOI suffix settles it."""
    number = str(pages).strip()
    if not doi or not re.fullmatch(r"[A-Za-z]?\d{4,}", number):
        return False
    return bool(re.search(rf"(?<![0-9A-Za-z]){re.escape(number)}$", str(doi).strip(), re.IGNORECASE))


async def verify_reference(client: httpx.AsyncClient, ref: Reference, *, check_metadata: bool = True) -> ReferenceCheck:
    """Check one reference live.

    Bibcodes: the ADS link gateway is asked for the full-text link types in
    :data:`ADS_LINK_TYPES`; a redirect proves ADS knows the bibcode. When all of them
    are 404 (no full text), the types in :data:`ADS_OTHER_LINK_TYPES` (associated works,
    data, e-sources, SIMBAD) are tried, where a 200 or redirect also proves the bibcode.
    ADS answers 404 to every type for an unknown bibcode *and* for a real one without
    any link, so a bibcode no link type answers for is reported undecided
    (``exists=None``), not missing. When the full-text redirect goes to doi.org, that DOI
    must equal the reference's DOI. With ``check_metadata`` the DOI's CSL-JSON (doi.org) is compared
    with the reference's year, volume and first page. Dataset references (no bibcode)
    are checked through doi.org only. Network failures and HTTP 5xx/429 give
    ``exists=None`` (undecided), never a false negative.
    """
    check = ReferenceCheck(key=ref.key, bibcode=ref.bibcode, exists=None)
    try:
        if ref.bibcode:
            statuses = []
            for link_type in ADS_LINK_TYPES:
                url = ADS_GATEWAY_URL.format(bibcode=ref.bibcode.replace("&", "%26"), link_type=link_type)
                response = await client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=False)
                statuses.append(response.status_code)
                if response.status_code in (301, 302, 303, 307, 308):
                    check.exists = True
                    check.link_type = link_type
                    check.resolved_url = response.headers.get("location")
                    break
                if response.status_code != 404:
                    check.problems.append(f"ADS gateway {link_type}: HTTP {response.status_code}")
                    return check
            else:
                for link_type in ADS_OTHER_LINK_TYPES:
                    url = ADS_GATEWAY_URL.format(bibcode=ref.bibcode.replace("&", "%26"), link_type=link_type)
                    response = await client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=False)
                    if response.status_code == 200 or response.status_code in (301, 302, 303, 307, 308):
                        check.exists = True
                        check.link_type = link_type
                        check.resolved_url = response.headers.get("location")
                        break
                    if response.status_code != 404:
                        check.problems.append(f"ADS gateway {link_type}: HTTP {response.status_code}")
                        return check
                else:
                    check.problems.append(
                        "ADS gives no full-text, data, e-source, SIMBAD or associated link for this bibcode "
                        "(HTTP 404 for every link type): existence undecided (a typo or an unlinked record)")
                    return check
            loc = check.resolved_url or ""
            if "doi.org/" in loc:
                check.doi_from_ads = loc.split("doi.org/", 1)[1]
                if ref.doi:
                    check.doi_matches = check.doi_from_ads.lower() == ref.doi.lower()
                    if not check.doi_matches:
                        check.problems.append(f"ADS DOI {check.doi_from_ads} != {ref.doi}")
        doi = ref.doi or check.doi_from_ads
        if doi and (check_metadata or not ref.bibcode):
            csl = await fetch_csl(client, doi)
            if csl is None:
                check.problems.append(f"DOI {doi} is not registered")
                if not ref.bibcode:
                    check.exists = False
                check.metadata_matches = False
                return check
            if not ref.bibcode:
                check.exists = True
            if check_metadata and ref.entry_type != "misc":
                check.metadata_matches, problems, check.metadata = _compare_csl(ref, csl)
                check.problems.extend(problems)
            else:
                check.metadata = {"title": csl.get("title"), "publisher": csl.get("publisher")}
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        check.problems.append(f"unreachable: {exc.__class__.__name__}: {exc}")
        check.metadata_matches = None
    except UnparsableAnswerError as exc:
        # Undecided, not failed: the DOI's metadata could not be read this time.
        check.problems.append(f"unparsable doi.org answer: {exc}")
        check.metadata_matches = None
    return check


async def verify_references(
    client: httpx.AsyncClient, references: Iterable[Reference], *, check_metadata: bool = True, delay_seconds: float = 0.25
) -> list[ReferenceCheck]:
    """Verify references one at a time (polite to ADS/doi.org: ``delay_seconds`` apart)."""
    results = []
    for n, ref in enumerate(references):
        if n and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        results.append(await verify_reference(client, ref, check_metadata=check_metadata))
    return results


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["provenance"])


class UpstreamServiceError(RuntimeError):
    """A service needed to build the query (the CDS Sesame name resolver) failed.

    ``status_code`` is the HTTP status the routes answer: 503 when the resolver could not be
    reached, 502 when it (or every archive) answered unusably."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnknownObjectNameError(ValueError):
    """The name resolver knows no object of this name (HTTP 404, as on every route)."""


class SearchFields(BaseModel):
    """The fields of ``POST /api/v1/search`` (``api.SearchRequest``), with the same types,
    defaults and constraints (``tests/test_provenance_round3.py`` compares them field by
    field, so a change to either model fails the tests instead of drifting silently).

    Kept as a copy rather than imported: ``api`` imports the feature routers, so importing
    ``api`` here would be circular.
    """

    model_config = ConfigDict(extra="forbid")
    # False for /api/v1/search/batch items (api.BatchSearchItem): their configured-limit check
    # runs per item when the search runs, so one item above it fails on its own.
    enforce_configured_radius_limit: ClassVar[bool] = True
    ra: float | None = Field(None, description="Right ascension in degrees, [0, 360)", ge=0, lt=360)
    dec: float | None = Field(None, description="Declination in degrees, [-90, 90]", ge=-90, le=90)
    name: str | None = Field(None, description="Astronomical object name (will be resolved)")
    radius_arcsec: float = Field(3.0, description="Search radius in arcseconds (at most API_MAX_RADIUS_ARCSEC, "
                                                  "default 1800; never above 3600, one degree)", gt=0,
                                 le=MAX_SEARCH_RADIUS_ARCSEC)
    profile: str | None = Field(None, description="Catalog profile: optical, infrared, radio, etc.")
    epoch: float | None = Field(None, ge=1800, le=2200, description="Julian epoch of ra/dec for proper motion correction")
    pm_ra_masyr: float | None = Field(None, description="Target proper motion in RA*cos(Dec), mas/yr (with pm_dec_masyr)")
    pm_dec_masyr: float | None = Field(None, description="Target proper motion in Dec, mas/yr (with pm_ra_masyr)")
    parallax_mas: float | None = Field(None, ge=0, lt=1000,
                                       description="Target parallax in mas (removes the annual parallax from single-epoch catalogs)")
    object_types: list[str] | None = Field(None, description="Filter by object types")
    spectral_types: list[str] | None = Field(None, description="Filter by spectral classification")
    morphology: list[str] | None = None
    max_results: int | None = Field(None, gt=0, description="Maximum results per catalog")
    min_confidence: float = Field(0.5, description="Minimum match confidence", ge=0, le=1)
    catalogs: list[str] | None = None
    time_period: dict[str, Any] | None = None
    spatial_constraints: dict[str, Any] | None = None
    search_mode: Literal["cone", "shell", "cylinder"] = "cone"
    min_radius_arcsec: float = Field(0.0, ge=0, le=MAX_SEARCH_RADIUS_ARCSEC)
    proper_motion: bool = True
    adaptive_radius: bool = False
    min_distance_pc: float | None = Field(None, gt=0)
    max_distance_pc: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check_motion_and_shell(self) -> SearchFields:
        """Proper-motion components come together and stay below 20"/yr (as
        models.validate_target requires); a shell's inner radius is below its outer one."""
        if (self.pm_ra_masyr is None) != (self.pm_dec_masyr is None):
            raise ValueError("pm_ra_masyr and pm_dec_masyr must be given together")
        if self.pm_ra_masyr is not None and self.pm_dec_masyr is not None                 and math.hypot(self.pm_ra_masyr, self.pm_dec_masyr) > 20_000.0:
            raise ValueError("proper motion exceeds 20 arcsec/yr; check units (mas/yr expected)")
        if self.min_radius_arcsec and self.min_radius_arcsec >= self.radius_arcsec:
            raise ValueError("min_radius_arcsec must be smaller than radius_arcsec")
        if self.enforce_configured_radius_limit:
            check_radius_limit(self.radius_arcsec)
        return self


def check_radius_limit(radius_arcsec: float | None) -> None:
    """ValueError when a search cone exceeds ``Settings.max_radius_arcsec`` (API_MAX_RADIUS_ARCSEC, default
    1800"): /api/v1/search, its batch items, saved queries and search manifests all validate their radius
    with the one check, :func:`models.check_search_radius` (as main.check_search_radius)."""
    check_search_radius(radius_arcsec)


class ManifestRequest(SearchFields):
    """Build a manifest from a UnifiedRecord dict, or run a search and manifest it.

    The search fields are those of ``POST /api/v1/search`` (:class:`SearchFields`) and
    the search runs exactly as that endpoint runs it (:func:`run_api_search`), so the
    manifest of ``{name}`` or ``{ra, dec}`` describes the result ``/api/v1/search``
    returns for the same fields. A UI should still send the ``record`` it displays: that
    is the result the user saw.
    """

    record: dict[str, Any] | None = Field(None, description="UnifiedRecord dict as returned by POST /api/v1/search")
    include_rows: bool = Field(True, description="False stores row digests only (compact manifest)")
    include_record: bool = Field(False, description="Also return the record the manifest describes")
    live_release_lookup: bool = Field(True, description="Record the data release HEASARC tables serve now")

    def search_fields(self) -> dict[str, Any]:
        """The ``POST /api/v1/search`` fields of this request."""
        return self.model_dump(include=set(SEARCH_FIELDS))


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest: dict[str, Any]
    position_tolerance_arcsec: float = Field(DEFAULT_POSITION_TOLERANCE_ARCSEC, ge=0, le=3600)
    sigma_fraction: float = Field(DEFAULT_SIGMA_FRACTION, ge=0, le=100,
                                  description="A row also counts as moved beyond this fraction of its 1-sigma error")
    numeric_rtol: float = Field(DEFAULT_NUMERIC_RTOL, ge=0, le=1)
    confidence_tolerance: float = Field(DEFAULT_CONFIDENCE_TOLERANCE, ge=0, le=1)
    use_cache: bool = Field(False, description="Allow provider response caches (default: really re-query the archives)")


def _state(request: Request, name: str) -> Any:
    return getattr(request.app.state, name, None)


@contextlib.asynccontextmanager
async def _request_service(request: Request) -> AsyncIterator[tuple[CrossmatchService, httpx.AsyncClient]]:
    """(service, HTTP client) for one request: the app's shared objects when present.

    Without ``app.state.client`` (and no client on the service's providers) one client
    is opened for this request and closed afterwards; a service built here uses it, so
    no provider or resolver creates a client of its own that nobody closes.
    """
    service = _state(request, "service")
    client = _state(request, "client") or (_client_of(service) if service is not None else None)
    own: httpx.AsyncClient | None = None
    if client is None:
        own = client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
    try:
        if service is None:
            from main import build_service  # lazy: main imports the whole application

            service = build_service(client=client)
        yield service, client
    finally:
        if own is not None:
            await own.aclose()


def _record_catalogs(record: Mapping[str, Any]) -> list[str]:
    """Catalog names a record planned or answered (for the live release lookup)."""
    prov_raw, results_raw = record.get("provenance"), record.get("catalog_results")
    prov: Mapping[str, Any] = prov_raw if isinstance(prov_raw, Mapping) else {}
    results: Mapping[str, Any] = results_raw if isinstance(results_raw, Mapping) else {}
    planned_raw = prov.get("catalogs_planned")
    planned: list[Any] = planned_raw if isinstance(planned_raw, list) else []
    return sorted({str(n) for n in [*planned, *results]})


def unreachable_outage(record: UnifiedRecord | Mapping[str, Any]) -> str | None:
    """A message when every planned catalog failed to reach its archive, else None.

    Such a result holds no science: manifesting it would describe an outage, so the
    routes answer 502 and the CLI exits 3, as a replay does (:class:`ReplayUnavailableError`).
    """
    failures_raw: Any
    prov: Any
    results: Any
    if isinstance(record, UnifiedRecord):  # attributes, not as_dict(): no deep copy on the event loop
        failures_raw, prov, results = record.failures, record.provenance, record.catalog_results
    else:
        failures_raw, prov, results = record.get("failures"), record.get("provenance"), record.get("catalog_results")
    failures = [f for f in failures_raw or [] if isinstance(f, Mapping)]
    planned = prov.get("catalogs_planned") if isinstance(prov, Mapping) else None
    names = set(planned or []) | set(results if isinstance(results, Mapping) else {})
    if not names or {f.get("catalog") for f in failures} != names:
        return None
    if not all(f.get("error_type") in UNREACHABLE_ERRORS for f in failures):
        return None
    return ("every catalog failed to reach its archive: "
            + "; ".join(f"{f.get('catalog')}: {f.get('error_type')}" for f in failures))


def _manifest_response(rec_dict: dict[str, Any], manifest_kwargs: dict[str, Any], include_record: bool) -> dict[str, Any]:
    """Worker-thread part of the manifest route (CPU work proportional to the row cells)."""
    manifest = build_manifest(rec_dict, **manifest_kwargs)
    return {"manifest": manifest.as_dict(), "record": exact(rec_dict) if include_record else None}


@router.post("/provenance/manifest")
async def manifest_endpoint(req: ManifestRequest, request: Request) -> dict[str, Any]:
    """Provenance manifest of a record (preferred: the record the UI displays), or of a
    new search run exactly as ``POST /api/v1/search`` runs it (ra/dec or name + options).

    A record that went through JavaScript hashes like the original (numbers are
    canonicalized in ECMAScript form, integers beyond 2**53 in archive columns at double
    precision), but its stored rows show those big integers rounded. 422: bad input;
    502: the name resolver or every archive was unreachable.
    """
    search_given = req.name is not None or req.ra is not None or req.dec is not None
    if req.record is not None and search_given:
        raise HTTPException(status_code=422, detail="give either a 'record' or search fields (ra/dec or name), not both")
    if req.record is None and not (req.name or (req.ra is not None and req.dec is not None)):
        raise HTTPException(status_code=422, detail="give a 'record', or 'ra' and 'dec', or 'name'")
    registry = _state(request, "registry")
    if registry is None and _state(request, "service") is not None:
        registry = request.app.state.service.registry
    async with _request_service(request) as (service, client):
        registry = registry or service.registry  # the deployment registry (CATALOG_REGISTRY_PATH)
        record: dict[str, Any] | UnifiedRecord
        query_time: datetime | None = None
        if req.record is not None:
            record = req.record
        else:
            query_time = datetime.now(UTC)
            try:
                record = await run_api_search(service, req.search_fields(), resolver=SesameResolver(client))
            except UpstreamServiceError as exc:
                raise HTTPException(status_code=exc.status_code, detail=f"upstream service error: {exc}",
                                    headers={"Retry-After": "30"} if exc.status_code == 503 else None) from exc
            except UnknownObjectNameError as exc:
                raise HTTPException(status_code=404, detail=f"Object name could not be resolved: {exc}") from exc
            except (ValueError, InvalidCoordinateError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"upstream archive error: {exc}") from exc
            outage = unreachable_outage(record)
            if outage:
                raise HTTPException(status_code=502, detail=f"upstream archives unavailable: {outage}")
        await software_info_async()
        try:
            rec_dict = await asyncio.to_thread(record_to_dict, record)
            releases = (await live_releases(client, registry, _record_catalogs(rec_dict))
                        if req.live_release_lookup else None)
            return await asyncio.to_thread(
                _manifest_response, rec_dict,
                {"registry": registry, "include_rows": req.include_rows, "releases": releases, "query_time": query_time},
                req.include_record)
        except ManifestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (TypeError, AttributeError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"record is not a valid UnifiedRecord: {exc!r}") from exc


def _resolution_error(exc: Exception) -> Exception:
    """Sesame failures, classified as every route does (models.resolution_failure_status): an
    unknown name is UnknownObjectNameError (404), an empty one a ValueError (422), an
    unreachable resolver an UpstreamServiceError with status 503, an unusable answer 502."""
    from models import resolution_failure_status

    text = str(exc)
    code = resolution_failure_status(exc)
    if code == 404:
        return UnknownObjectNameError(text)
    if code == 422:
        return ValueError(text)
    return UpstreamServiceError(text, status_code=code)


async def _resolve(resolver: Any, name: str) -> tuple[Any, Target]:
    from models import ObjectResolutionError, resolved_target

    try:
        resolved = await resolver.resolve(name)
    except ObjectResolutionError as exc:
        raise _resolution_error(exc) from exc
    return resolved, resolved_target(resolved)


# The fields of POST /api/v1/search (api.SearchRequest), in its order, and the defaults
# of those a caller may leave out -- both read from SearchFields.
SEARCH_FIELDS: tuple[str, ...] = tuple(SearchFields.model_fields)
SEARCH_DEFAULTS: dict[str, Any] = {name: f.default for name, f in SearchFields.model_fields.items()
                                   if f.default is not None}


def _search_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """api.SearchRequest fields with its defaults for the ones left out."""
    return {**dict.fromkeys(SEARCH_FIELDS), **SEARCH_DEFAULTS,
            **{k: v for k, v in fields.items() if v is not None or k in ("ra", "dec")}}


def api_search_query(fields: Mapping[str, Any], resolved_target_: Target | None = None, *,
                     spec: Mapping[str, Any] | None = None, resolved: Any = None) -> AdvancedQuery:
    """The AdvancedQuery ``POST /api/v1/search`` builds from its fields (``main.search_query``,
    the code the endpoint itself runs).

    For a name search pass ``spec`` (:func:`crossmatch.resolved_search_target` of the Sesame
    answer) and ``resolved`` (the answer); ``resolved_target_`` (:func:`models.resolved_target`
    of the answer, the former interface) is still accepted and taken as the resolver's
    position, epoch, motion and parallax.
    """
    from main import search_query  # lazy: main imports this module for its CLI

    if spec is None and resolved_target_ is not None:
        t = resolved_target_
        f = _search_fields(fields)
        motion = t.proper_motion is not None and f.get("pm_ra_masyr") is None and f.get("pm_dec_masyr") is None
        spec = {"ra": t.ra, "dec": t.dec, "epoch": t.epoch if t.epoch is not None else f.get("epoch"),
                "pm_ra_masyr": t.pm_ra_masyr if motion else f.get("pm_ra_masyr"),
                "pm_dec_masyr": t.pm_dec_masyr if motion else f.get("pm_dec_masyr"),
                "parallax_mas": (t.parallax_mas if motion and f.get("parallax_mas") is None else f.get("parallax_mas")),
                "pm_source": "resolver" if motion else ("input" if f.get("pm_ra_masyr") is not None else None)}
    info = resolved.as_dict() if hasattr(resolved, "as_dict") else resolved
    return search_query(_search_fields(fields), spec, info)


async def run_api_search(service: CrossmatchService, fields: Mapping[str, Any], *, resolver: Any | None = None) -> UnifiedRecord:
    """Run a search exactly as ``POST /api/v1/search`` does (``main.api_search``, without the
    endpoint's response cache).

    ``fields`` are api.SearchRequest fields (missing ones take its defaults);
    ``resolver`` (default: a :class:`providers.SesameResolver` on the service's client)
    resolves ``fields["name"]``. Unknown names raise ValueError, resolver outages
    :class:`UpstreamServiceError`.
    """
    from main import api_search  # lazy: main imports this module for its CLI

    active = None
    if fields.get("name"):
        active = _CheckedResolver(resolver or SesameResolver(_client_of(service)))
    return await api_search(service, _search_fields(fields), active)


class _CheckedResolver:
    """A resolver whose failures are this module's errors (unknown name: ValueError; outage:
    UpstreamServiceError), as :func:`_resolve` reports them."""

    def __init__(self, resolver: Any) -> None:
        self.resolver = resolver

    async def resolve(self, name: str) -> Any:
        resolved, _target = await _resolve(self.resolver, str(name))
        return resolved


async def run_basic_search(
    service: CrossmatchService,
    *,
    name: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_arcsec: float | None = None,
    profile: str | None = None,
    epoch: float | None = None,
    resolver: Any | None = None,
) -> UnifiedRecord:
    """Run a search exactly as ``astrosearch search`` does (``main.search_object`` for a
    name: resolver position, epoch, proper motion and parallax; ``main.crossmatch`` for
    coordinates): the basic crossmatch path, every in-radius row kept. A name and ra/dec
    together are refused (``main.check_search_target``)."""
    from main import check_search_target  # lazy: main imports this module for its CLI

    name = check_search_target(name, ra, dec)  # a blank name is no name, as on every route
    check_radius_limit(radius_arcsec)
    if name:
        from main import crossmatch_resolved  # lazy: main imports this module for its CLI

        resolved, _tgt = await _resolve(resolver or SesameResolver(_client_of(service)), name)
        return await crossmatch_resolved(service, resolved, radius_arcsec=radius_arcsec, profile=profile)
    if ra is None or dec is None:
        raise ValueError("give a name, or ra and dec")
    return await service.crossmatch(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile)


@router.post("/provenance/replay")
async def replay_endpoint(req: ReplayRequest, request: Request) -> dict[str, Any]:
    """Re-run a manifest's query; returns identical flag, structured diff and new manifest.

    422: malformed manifest (checked before any archive is queried), a search radius above
    API_MAX_RADIUS_ARCSEC or the 3600" ceiling, or catalogs all unknown to the registry.
    502: every archive unreachable.
    """
    try:
        manifest = await asyncio.to_thread(ProvenanceManifest.from_dict, req.manifest)
    except ManifestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with _request_service(request) as (service, client):
        try:
            result = await replay_manifest(manifest, service, client=client,
                                           position_tolerance_arcsec=req.position_tolerance_arcsec,
                                           numeric_rtol=req.numeric_rtol, sigma_fraction=req.sigma_fraction,
                                           confidence_tolerance=req.confidence_tolerance, use_cache=req.use_cache,
                                           settings=_state(request, "settings"))
        except ManifestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ValueError, InvalidCoordinateError) as exc:
            raise HTTPException(status_code=422, detail=f"manifest query is invalid: {exc}") from exc
        except ReplayUnavailableError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream archive error during replay: {exc}") from exc
    return await asyncio.to_thread(result.as_dict)


@router.get("/citations")
async def citations_endpoint(
    request: Request,
    catalogs: str = Query(..., min_length=1, description="Comma-separated catalog/service names, e.g. gaia_dr3,simbad,hips2fits"),
    format: Literal["json", "bibtex"] = Query("json", description="'bibtex' returns the .bib text only"),
    verify: bool = Query(False, description="Resolve papers named only in a registry citation (a table registered "
                                            "with vizier add before its references were stored) through ADS/doi.org"),
):
    """Acknowledgement texts and BibTeX (ADS-verified bibcodes) for the given catalogs/services.

    Names without a citation entry are listed in ``unknown`` (and in the
    ``X-Unknown-Citations`` header of the BibTeX response); 422 only when none is known.
    A registry catalog without a curated entry (a VizieR table registered with ``vizier add``)
    is cited with the paper(s) its registry citation names: the records ``vizier add`` resolved
    through ADS/doi.org, else entries parsed from the citation text and marked unverified
    (``verify=true`` resolves those now; ``X-Unverified-References`` lists what stays unverified).
    """
    names = [c for c in (s.strip() for s in catalogs.split(",")) if c]
    if not names:
        raise HTTPException(status_code=422, detail="catalogs must name at least one catalog or service")
    registry = _state(request, "registry")
    service = _state(request, "service")
    if registry is None and service is not None:
        registry = service.registry
    try:
        bundle = citations_for(names, registry, strict=False)
    except UnknownCitationError as exc:  # none of the names is known
        raise HTTPException(status_code=422, detail=str(exc.args[0])) from exc
    resolution_notes: list[str] = []
    if verify and any(not ref.verified for ref in bundle.references):
        client = _state(request, "client")
        if client is not None:
            resolution_notes = await resolve_unverified(client, bundle)
        else:
            async with httpx.AsyncClient(timeout=60.0) as own:
                resolution_notes = await resolve_unverified(own, bundle)
    unverified = [ref.key for ref in bundle.references if not ref.verified]
    if format == "bibtex":
        headers = {"X-Unknown-Citations": ",".join(bundle.unknown)} if bundle.unknown else {}
        if unverified:
            headers["X-Unverified-References"] = ",".join(unverified)
        return PlainTextResponse(bundle.bibtex, media_type="application/x-bibtex", headers=headers or None)
    out = bundle.as_dict()
    out["unverified_references"] = unverified
    if resolution_notes:
        out["resolution_notes"] = resolution_notes
    return out


@router.get("/citations/sources")
async def citation_sources_endpoint() -> dict[str, Any]:
    """Every catalog/service with a citation entry, plus accepted aliases."""
    return {
        "sources": [
            {"key": e.key, "name": e.name, "kind": e.kind, "references": [REFERENCES[r].short for r in e.references]}
            for e in CITATION_ENTRIES.values()
        ],
        "aliases": dict(sorted(CITATION_ALIASES.items())),
    }


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------
#
# Exit codes: 0 success (replay: identical science), 1 replay differs / reference
# verification failed, 2 invalid input or output (manifest, record, name, file paths),
# 3 upstream unavailable (every archive or the name resolver unreachable), 4 replay
# gave the same content hash but some catalogs failed in both runs (not compared).


def _deployment_registry() -> CatalogRegistry:
    """The registry the deployment runs with (as ``main.build_registry``): the embedded catalogs plus the
    VizieR tables registered with ``vizier add`` (``vizier.load_registry``, ``CATALOG_REGISTRY_PATH`` or
    ~/.astrosearch/catalogs.yaml), so ``astrosearch cite --catalogs vizier_i_345_gaia2`` cites a registered
    table's own paper instead of reporting it unknown."""
    import vizier  # lazy: vizier imports this module in its citation helpers
    from models import Settings

    return vizier.load_registry(Settings().catalog_registry_path or None)


class CLIInputError(Exception):
    """A file argument could not be read or written."""


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CLIInputError(f"cannot read {path}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CLIInputError(f"{path} is not valid JSON: {exc}") from exc


def _write_text(path: str, text: str) -> None:
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CLIInputError(f"cannot write {path}: {exc}") from exc


def _console(text: str) -> str:
    """``text`` made printable on this stdout (cp1252 when redirected on Windows):
    characters the encoding lacks become backslash escapes instead of crashing."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, "backslashreplace").decode(encoding)
    except LookupError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def _print_json(value: Any) -> None:
    # ASCII-escaped: Windows consoles and redirected stdout default to cp1252.
    print(json.dumps(value, indent=2, ensure_ascii=True))


def _print_replay(result: ReplayResult) -> None:
    print(f"Original content hash: {result.original_content_hash}")
    print(f"Replay content hash:   {result.new_content_hash}")
    print(f"Identical: {'yes' if result.identical else 'no'}"
          f"  (equivalent within tolerance: {'yes' if result.equivalent_within_tolerance else 'no'})")
    totals = result.diff.get("totals", {})
    print("Changes: " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    if result.not_compared:
        print("Not compared (failed in both runs): " + ", ".join(c["catalog"] for c in result.not_compared))
    for line in result.explanation:
        print(_console(f"  - {line}"))


async def _replay_cli(args: argparse.Namespace) -> int:
    manifest = ProvenanceManifest.from_dict(_read_json(args.manifest))
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        from main import build_service

        service = build_service(client=client)
        result = await replay_manifest(manifest, service, client=client, position_tolerance_arcsec=args.tolerance,
                                       use_cache=False)
    # Print first: a failing --out must not lose a replay that queried every archive.
    if args.json:
        _print_json(result.as_dict())
    else:
        _print_replay(result)
    if args.out:
        _write_text(args.out, result.new_manifest.to_json())
        if not args.json:
            print(f"New manifest written to {args.out}")
    if result.identical:
        return 0
    return 4 if result.content_hash_equal and result.not_compared else 1


def replay_command(args: argparse.Namespace) -> int:
    """CLI: ``replay manifest.json`` -> exit 0 identical, 1 differs, 2 bad input, 3 archives down,
    4 same content hash but catalogs that failed in both runs were not compared."""
    try:
        return asyncio.run(_replay_cli(args))
    except ManifestError as exc:
        print(f"Invalid manifest: {exc}", file=sys.stderr)
        return 2
    except CLIInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, InvalidCoordinateError) as exc:
        print(f"Invalid replay request: {exc}", file=sys.stderr)
        return 2
    except ReplayUnavailableError as exc:
        print(f"Replay impossible: {exc}", file=sys.stderr)
        return 3
    except httpx.HTTPError as exc:
        print(f"Upstream archive error during replay: {exc}", file=sys.stderr)
        return 3


def cite_command(args: argparse.Namespace) -> int:
    """CLI: ``cite --catalogs a,b [--from record_or_manifest.json] [--bibtex out.bib] [--verify]``."""
    names: list[str] = [c for c in (args.catalogs or "").split(",") if c.strip()]
    try:
        if args.source:
            data = _read_json(args.source)
            try:
                names.extend(catalogs_in(data, include_empty=args.include_empty))
            except (ManifestError, AttributeError, TypeError) as exc:
                raise CLIInputError(f"{args.source} is not a record or manifest: {exc}") from exc
    except CLIInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not names:
        print("Give --catalogs a,b or --from record/manifest.json", file=sys.stderr)
        return 2
    try:
        bundle = citations_for(names, _deployment_registry(), strict=False)
    except UnknownCitationError as exc:
        print(str(exc.args[0]), file=sys.stderr)
        return 2
    except ValueError as exc:  # models.RegistryError: CATALOG_REGISTRY_PATH cannot be parsed
        print(f"Catalog registry error: {exc}", file=sys.stderr)
        return 2
    if bundle.unknown:
        print(f"Warning: no citation entry for {bundle.unknown} (not cited); known: {known_citation_keys()}",
              file=sys.stderr)
    if args.verify and any(not ref.verified for ref in bundle.references):
        # Papers named only in a registry citation: resolved through ADS/doi.org before the
        # BibTeX is written, then verified with the others below.
        async def resolve() -> list[str]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                return await resolve_unverified(client, bundle)

        for note in asyncio.run(resolve()):
            print(_console(f"Warning: {note}"), file=sys.stderr)
    elif any(not ref.verified for ref in bundle.references):
        print("Note: " + ", ".join(ref.key for ref in bundle.references if not ref.verified)
              + " parsed from a registry citation (unverified); --verify resolves them through ADS/doi.org",
              file=sys.stderr)
    if args.json:
        _print_json(bundle.as_dict())
    else:
        print("Acknowledgements:\n")
        print(_console(bundle.acknowledgement_text))
        print("\nReferences:")
        for ref in bundle.references:
            where = ref.bibcode or f"doi:{ref.doi}"
            print(_console(f"  {ref.short:<32} {where}"))
    if args.bibtex:
        text = bundle.bibtex
        parse_bibtex(text)  # never write an invalid file
        try:
            _write_text(args.bibtex, text)
        except CLIInputError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not args.json:
            print(f"\nBibTeX ({len(bundle.references)} entries) written to {args.bibtex}")
    elif not args.json:
        print("\n" + bundle.bibtex)  # BibTeX is ASCII: accents are LaTeX macros
    if args.verify:
        async def run() -> list[ReferenceCheck]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                return await verify_references(client, bundle.references)

        checks = asyncio.run(run())
        bad = [c for c in checks if not c.ok or c.undecided]
        for c in checks:
            state = "FAILED" if not c.ok and not c.undecided else "UNDECIDED" if c.undecided else "ok"
            print(_console(f"  verify {c.key:<24} {state} {'; '.join(c.problems)}"))
        return 1 if bad else 0
    return 0


async def _manifest_cli(args: argparse.Namespace) -> int:
    if args.record:
        record: Any = _read_json(args.record)
        registry = _deployment_registry()
        async with httpx.AsyncClient(timeout=RELEASE_LOOKUP_TIMEOUT, follow_redirects=True) as client:
            releases = (await live_releases(client, registry, _record_catalogs(record))
                        if isinstance(record, Mapping) and not args.no_live_release else None)
        manifest = build_manifest(record, registry=registry, include_rows=not args.compact, releases=releases)
    else:
        if not args.name and (args.ra is None or args.dec is None):
            print("Give --record FILE, or --ra and --dec, or --name", file=sys.stderr)
            return 2
        catalogs = [c.strip() for c in (args.catalogs or "").split(",") if c.strip()] or None
        api_mode = bool(args.api or catalogs or args.min_confidence is not None)
        from main import build_service
        from models import Settings

        settings = Settings()
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
            service = build_service(settings=settings, client=client)
            asked = datetime.now(UTC)
            resolver = SesameResolver(client) if api_mode else SesameResolver(client, endpoint=settings.resolver_endpoint)
            if api_mode:
                fields = {"ra": args.ra, "dec": args.dec, "name": args.name, "radius_arcsec": args.radius,
                          "profile": args.profile, "epoch": args.epoch, "catalogs": catalogs,
                          "min_confidence": 0.5 if args.min_confidence is None else args.min_confidence}
                try:
                    fields = ManifestRequest.model_validate(fields).search_fields()
                except ValidationError as exc:
                    raise ValueError(f"invalid search parameters: {exc.errors()[0].get('msg')}") from exc
                record = await run_api_search(service, fields, resolver=resolver)
            else:
                record = await run_basic_search(service, name=args.name, ra=args.ra, dec=args.dec,
                                                radius_arcsec=args.radius, profile=args.profile, epoch=args.epoch,
                                                resolver=resolver)
            outage = unreachable_outage(record)
            if outage:
                raise UpstreamServiceError(outage)
            releases = None if args.no_live_release else await live_releases(client, service.registry, _record_catalogs(record.as_dict()))
            manifest = build_manifest(record, registry=service.registry, include_rows=not args.compact, releases=releases,
                                      query_time=asked)
    if args.out:
        _write_text(args.out, manifest.to_json())
        print(f"Manifest written to {args.out}")
        print(f"  content hash: {manifest.content_hash}")
        print(f"  query hash:   {manifest.query_hash}")
        print("  rows: " + ", ".join(f"{k}={v}" for k, v in sorted(manifest.row_counts.items())))
    else:
        print(manifest.to_json(ensure_ascii=True))
    return 0


def manifest_command(args: argparse.Namespace) -> int:
    """CLI: ``manifest (--record record.json | --ra RA --dec DEC | --name NAME) [-o manifest.json]``."""
    try:
        return asyncio.run(_manifest_cli(args))
    except ManifestError as exc:
        print(f"Cannot build manifest: {exc}", file=sys.stderr)
        return 2
    except CLIInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, InvalidCoordinateError) as exc:
        print(f"Cannot build manifest: {exc}", file=sys.stderr)
        return 2
    except UpstreamServiceError as exc:
        print(f"Upstream service error: {exc}", file=sys.stderr)
        return 3
    except httpx.HTTPError as exc:
        print(f"Upstream archive error: {exc}", file=sys.stderr)
        return 3


def register_cli(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``replay``, ``cite`` and ``manifest`` subcommands (handler returns an exit code)."""
    replay = subparsers.add_parser("replay", help="Re-run a provenance manifest and diff the science")
    replay.add_argument("manifest", help="Path to a manifest JSON file")
    replay.add_argument("--out", help="Write the replay's new manifest here")
    replay.add_argument("--tolerance", type=float, default=DEFAULT_POSITION_TOLERANCE_ARCSEC,
                        help="Position change (arcsec) below which a source is not reported as moved (default 0.001; "
                             "rows also move beyond 0.1 x their 1-sigma error)")
    replay.add_argument("--json", action="store_true", help="Print the full replay result as JSON")
    replay.set_defaults(handler=replay_command)

    cite = subparsers.add_parser("cite", help="Acknowledgements and BibTeX for catalogs/services used")
    cite.add_argument("--catalogs", default="", help="Comma-separated names, e.g. gaia_dr3,simbad,allwise")
    cite.add_argument("--from", dest="source", help="Record or manifest JSON: cite the catalogs that returned rows")
    cite.add_argument("--include-empty", action="store_true",
                      help="With --from: also cite catalogs that answered with no rows (reported non-detections)")
    cite.add_argument("--bibtex", help="Write the BibTeX file here (validated first)")
    cite.add_argument("--verify", action="store_true", help="Check every reference live against NASA ADS and doi.org")
    cite.add_argument("--json", action="store_true", help="Print the citation bundle as JSON")
    cite.set_defaults(handler=cite_command)

    manifest = subparsers.add_parser("manifest", help="Build a provenance manifest from a record or a new search")
    manifest.add_argument("--record", help="UnifiedRecord JSON: the 'record' a UI got from POST /api/v1/search, or "
                                           "the output of 'astrosearch search --ra RA --dec DEC --format json'")
    manifest.add_argument("--ra", type=float)
    manifest.add_argument("--dec", type=float)
    manifest.add_argument("--name")
    manifest.add_argument("--radius", type=float, default=3.0)
    manifest.add_argument("--profile", help="Catalog profile (optical, infrared, radio, ...)")
    manifest.add_argument("--epoch", type=float, help="Julian epoch of --ra/--dec")
    manifest.add_argument("--api", action="store_true",
                          help="Search as POST /api/v1/search does (advanced query, min confidence 0.5) instead of as "
                               "'astrosearch search' does (every in-radius row kept); implied by --catalogs/--min-confidence")
    manifest.add_argument("--catalogs", default="", help="Comma-separated registry catalogs (implies --api)")
    manifest.add_argument("--min-confidence", type=float, help="Minimum match confidence (implies --api; default 0.5)")
    manifest.add_argument("--compact", action="store_true", help="Store row digests only")
    manifest.add_argument("--no-live-release", action="store_true",
                          help="Do not read the data release HEASARC tables serve now (use the built-in labels)")
    manifest.add_argument("-o", "--out", help="Output manifest path (default: print)")
    manifest.set_defaults(handler=manifest_command)


__all__ = [
    "CATALOG_RELEASES", "CITATION_ALIASES", "CITATION_ENTRIES", "DERIVED_PARALLAX_ORIGINS", "DERIVED_PM_ORIGINS",
    "FLOAT32_RTOL", "MANIFEST_SCHEMA", "REFERENCES", "RELEASE_TIME_WINDOW_SECONDS", "SCIENCE_SCHEMA", "SEARCH_FIELDS",
    "UNVERIFIED_RELEASE_PREFIX", "BibEntry", "BibTeXError", "CatalogRequest", "CitationBundle", "CitationEntry",
    "ManifestError", "ManifestRequest", "ProvenanceManifest", "Reference", "ReferenceCheck", "ReplayResult",
    "ReplayUnavailableError", "SearchFields", "UnknownCitationError", "UnknownObjectNameError", "UnparsableAnswerError", "UpstreamServiceError",
    "api_search_query", "association_inputs", "build_manifest", "canonical_json", "canonicalize", "catalogs_in",
    "citations_for", "code_digests", "compact_science", "content_hash", "curated_citation", "diff_requests",
    "diff_science", "exact", "given_parallax_source", "given_pm_source", "input_target", "js_number_value",
    "live_releases", "matches_curated", "parse_advanced_query", "parse_bibtex", "reconstruct_request",
    "record_query_time", "register_cli", "replay_manifest", "router", "run_api_search", "run_basic_search",
    "science_payload", "served_by", "software_info", "software_info_async", "to_bibtex", "unreachable_outage",
    "unverified_dois", "verify_manifest", "verify_reference", "verify_references",
]
