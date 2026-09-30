"""Public astronomy archive adapters, CDS Sesame resolver, resilience, and caching layer."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import ssl
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from models import (
    EXTRAGALACTIC_OTYPES,
    META_CATALOG_HAS_PM,
    META_SINGLE_EPOCH,
    CatalogDefinition,
    CatalogQueryError,
    CatalogSource,
    CatalogUnavailableError,
    ColumnMeta,
    ConePlan,
    ObjectResolutionError,
    ParsedTable,
    QueryTimeoutError,
    RateLimitedError,
    ResolvedObject,
    ResolverUnavailableError,
    ResponseParseError,
    Target,
    angle_to_arcsec,
    bare_column_name,
    build_provenance,
    catalog_epoch_range,
    catalog_has_proper_motions,
    compute_positional_error,
    epoch_separation_arcsec,
    find_column,
    haversine_arcsec,
    is_extragalactic_type,
    normalize_source_record,
    parse_csv_records,
    parse_csv_table,
    parse_ipac_records,
    parse_ipac_table,
    parse_json_records,
    parse_json_table,
    parse_votable_records,
    parse_votable_table,
    plan_cone,
    propagate_radec,
    resolve_epoch,
    resolve_epoch_range,
    row_get,
    target_pm_separation_arcsec,
    votable_query_status,
)

logger = logging.getLogger("astrosearch.providers")

__all__ = [
    "CacheManager",
    "CatalogProvider",
    "EndpointGuard",
    "HEASARCXaminProvider",
    "IRSAGatorProvider",
    "MASTProvider",
    "QueryResult",
    "SDSSProvider",
    "SesameResolver",
    "TapProvider",
    "haversine_arcsec",
    "new_http_client",
    "new_http_client_async",
    "parse_csv_records",
    "parse_ipac_records",
    "parse_json_records",
    "parse_votable_records",
    "provider_map",
    "shared_ssl_context",
]


# ---------------------------------------------------------------------------
# HTTP clients owned by this package
# ---------------------------------------------------------------------------


def _ssl_ca_locations(trust_env: bool = True) -> tuple[str | None, str | None]:
    """The ``(cafile, capath)`` httpx would verify against for ``verify=True``.

    Mirrors httpx 0.28's ``create_ssl_context``: with ``trust_env`` a non-empty ``SSL_CERT_FILE``
    wins, else a non-empty ``SSL_CERT_DIR``; otherwise certifi's bundle (``(None, None)`` here).
    The environment is read on every call, so a changed variable takes effect for new clients."""
    if trust_env:
        cafile = os.environ.get("SSL_CERT_FILE")
        if cafile:
            return cafile, None
        capath = os.environ.get("SSL_CERT_DIR")
        if capath:
            return None, capath
    return None, None


_SSL_CONTEXTS: dict[tuple[str | None, str | None], ssl.SSLContext] = {}
_SSL_CONTEXTS_LOCK = threading.Lock()


def _ssl_context_for(cafile: str | None, capath: str | None) -> ssl.SSLContext:
    """One context per CA location: the bundle is loaded once, not for every client."""
    key = (cafile, capath)
    with _SSL_CONTEXTS_LOCK:  # held while building, so concurrent first calls load it once
        context = _SSL_CONTEXTS.get(key)
        if context is None:
            if cafile:
                context = ssl.create_default_context(cafile=cafile)
            elif capath:
                context = ssl.create_default_context(capath=capath)
            else:
                try:
                    import certifi
                except ImportError:  # pragma: no cover - certifi ships with httpx
                    context = ssl.create_default_context()
                else:
                    context = ssl.create_default_context(cafile=certifi.where())
            _SSL_CONTEXTS[key] = context
        return context


@dataclass(frozen=True)
class _SSLCacheInfo:
    currsize: int


class _SharedSSLContext:
    """Callable returning the shared SSL context for the current environment.

    httpx builds a fresh context from its CA bundle for every new client, which takes ~0.5-1 s
    on Windows; a client built on the event loop stalls every other request for that long. This
    trusts exactly what httpx would for ``verify=True`` (``SSL_CERT_FILE`` / ``SSL_CERT_DIR``
    when ``trust_env``, else certifi), cached per ``(cafile, capath)``."""

    def __call__(self, *, trust_env: bool = True) -> ssl.SSLContext:
        return _ssl_context_for(*_ssl_ca_locations(trust_env))

    def is_cached(self, *, trust_env: bool = True) -> bool:
        """Whether the context for the current environment is already built (no bundle load)."""
        return _ssl_ca_locations(trust_env) in _SSL_CONTEXTS

    def cache_info(self) -> _SSLCacheInfo:
        return _SSLCacheInfo(currsize=len(_SSL_CONTEXTS))

    def cache_clear(self) -> None:
        with _SSL_CONTEXTS_LOCK:
            _SSL_CONTEXTS.clear()


shared_ssl_context = _SharedSSLContext()


def _owned_verify(kwargs: dict[str, Any]) -> bool:
    """True when the client would verify with httpx's default trust (so the shared context can
    stand in); a caller's ``verify=False``/own context/path, or a ``cert`` (httpx would load it
    into the context, mutating the shared one), is passed through to httpx untouched."""
    return kwargs.get("verify", True) is True and not kwargs.get("cert")


def new_http_client(timeout: float, **kwargs: Any) -> httpx.AsyncClient:
    """A client with the shared SSL context (the CA bundle is loaded once per CA location)."""
    kwargs.setdefault("follow_redirects", True)
    if _owned_verify(kwargs):
        trust_env = bool(kwargs.get("trust_env", True))
        kwargs["verify"] = shared_ssl_context(trust_env=trust_env)
    return httpx.AsyncClient(timeout=timeout, **kwargs)


async def new_http_client_async(timeout: float, **kwargs: Any) -> httpx.AsyncClient:
    """:func:`new_http_client` from a coroutine: an uncached SSL context is built off the
    event loop, so neither other requests nor a caller's cancellation wait for it."""
    if _owned_verify(kwargs):
        trust_env = bool(kwargs.get("trust_env", True))
        if not shared_ssl_context.is_cached(trust_env=trust_env):
            await asyncio.to_thread(shared_ssl_context, trust_env=trust_env)
    return new_http_client(timeout, **kwargs)

# ---------------------------------------------------------------------------
# Resilience: Rate Limiter and Circuit Breaker
# ---------------------------------------------------------------------------


@dataclass
class EndpointGuard:
    """A fixed-interval request pacer combined with a three-state circuit breaker.

    One guard instance is shared per provider endpoint. It throttles request starts
    and transitions to 'open' when consecutive failure thresholds are exceeded.
    After ``recovery_seconds`` one probe request is let through (half-open); its
    outcome closes or re-opens the circuit. A probe that never reports back (its task
    was cancelled or leaked) expires after ``probe_timeout_seconds`` so the circuit
    can never be stuck open for the life of the process.

    ``record_success``/``record_failure`` are synchronous so they can run from
    cancellation handlers (no awaiting, hence atomic on the event loop).
    """

    requests_per_second: float = 5.0
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    probe_timeout_seconds: float = 180.0
    clock: Callable[[], float] = time.monotonic
    _next_start: float = 0.0
    _failures: int = 0
    _opened_at: float | None = None
    _probe_in_flight: bool = False
    _probe_started: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        """Wait for rate limit interval and ensure the circuit is not open."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        async with self._lock:
            now = self.clock()
            if self._opened_at is not None:
                if now - self._opened_at < self.recovery_seconds:
                    raise CatalogUnavailableError("Provider circuit is open")
                probe_alive = (
                    self._probe_in_flight
                    and self._probe_started is not None
                    and now - self._probe_started < self.probe_timeout_seconds
                )
                if probe_alive:
                    raise CatalogUnavailableError("Provider circuit is open (recovery probe in flight)")
                self._probe_in_flight = True
                self._probe_started = now
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + 1.0 / self.requests_per_second
        if delay:
            await asyncio.sleep(delay)

    def record_success(self) -> None:
        """Reset failures and close the circuit (endpoint answered)."""
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False
        self._probe_started = None

    def record_failure(self) -> None:
        """Count a failure; open (or re-open) the circuit on threshold or a failed probe."""
        self._failures += 1
        if self._probe_in_flight or self._failures >= self.failure_threshold:
            self._opened_at = self.clock()
        self._probe_in_flight = False
        self._probe_started = None

    async def succeed(self) -> None:
        """Signal successful execution, resetting circuit breaker failures."""
        self.record_success()

    async def fail(self) -> None:
        """Record an endpoint failure and open the circuit if threshold is reached."""
        self.record_failure()

    @property
    def probe_in_flight(self) -> bool:
        return self._probe_in_flight

    @property
    def state(self) -> str:
        """Return the current circuit status: 'closed', 'half_open', or 'open'."""
        if self._opened_at is None:
            return "closed"
        if self.clock() - self._opened_at >= self.recovery_seconds:
            return "half_open"
        return "open"


# ---------------------------------------------------------------------------
# Caching: Distributed and In-Memory Layer
# ---------------------------------------------------------------------------


def _estimate_size(value: Any) -> int:
    """Approximate in-memory size (bytes) of a cached value (response bodies dominate)."""
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return len(value["content"]) + 256
    try:
        return len(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return 1024


class CacheManager:
    """Manages caching for provider responses and crossmatch queries.

    The in-process layer is a bounded LRU: at most ``max_entries`` values
    (PROVIDER_CACHE_MAX_ENTRIES, default 512) and ``max_bytes`` of estimated payload
    (PROVIDER_CACHE_MAX_BYTES, default 64 MB); expired entries are swept on every write,
    so a long-running process with many distinct cones cannot grow without bound.

    The Redis layer (REDIS_URL) uses the blocking redis-py client. In async code, read with
    :meth:`aget` (the Redis round trip runs in a worker thread); :meth:`set` called while an
    event loop runs writes to Redis in a worker thread too (serialisation included), so no
    Redis latency or outage ever stalls the event loop.
    """

    def __init__(self, redis_url: str | None = None, *, max_entries: int | None = None, max_bytes: int | None = None) -> None:
        self.redis_url = redis_url
        self._local_cache: OrderedDict[str, tuple[float, Any, int]] = OrderedDict()
        self._local_bytes = 0
        self.max_entries = max(1, int(max_entries if max_entries is not None else os.getenv("PROVIDER_CACHE_MAX_ENTRIES", "512")))
        self.max_bytes = max(1, int(max_bytes if max_bytes is not None else os.getenv("PROVIDER_CACHE_MAX_BYTES", str(64 * 1024 * 1024))))
        self._redis = None
        self._pending_writes: set[asyncio.Future[None]] = set()  # Redis writes running in worker threads
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
            except Exception:
                self._redis = None

    def get(self, key: str) -> Any | None:
        """Retrieve cached value from Redis or local memory cache (blocking; see :meth:`aget`)."""
        remote = self._redis_get(key)
        if remote is not None:
            return remote
        return self._local_get(key)

    async def aget(self, key: str) -> Any | None:
        """:meth:`get` for async code: the Redis round trip runs in a worker thread."""
        if self._redis:
            remote = await asyncio.to_thread(self._redis_get, key)
            if remote is not None:
                return remote
        return self._local_get(key)

    async def aset(self, key: str, value: Any, ttl: int = 3600) -> None:
        """:meth:`set` for async code (the Redis write never blocks the event loop)."""
        self.set(key, value, ttl=ttl)

    def _redis_get(self, key: str) -> Any | None:
        if not self._redis:
            return None
        try:
            val = self._redis.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    def _redis_store(self, key: str, value: Any, ttl: int) -> None:
        try:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass

    def _local_get(self, key: str) -> Any | None:
        entry = self._local_cache.get(key)
        if entry and entry[0] > time.monotonic():
            self._local_cache.move_to_end(key)
            return entry[1]
        self._drop(key)
        return None

    def _drop(self, key: str) -> None:
        entry = self._local_cache.pop(key, None)
        if entry is not None:
            self._local_bytes -= entry[2]

    def _sweep(self) -> None:
        now = time.monotonic()
        for key in [k for k, (expires, _, _) in self._local_cache.items() if expires <= now]:
            self._drop(key)
        while self._local_cache and (len(self._local_cache) > self.max_entries or self._local_bytes > self.max_bytes):
            self._drop(next(iter(self._local_cache)))  # least recently used

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Store value with TTL in seconds (values larger than max_bytes are not kept locally)."""
        if ttl <= 0:
            return
        size = _estimate_size(value)
        self._drop(key)
        if size <= self.max_bytes:
            self._local_cache[key] = (time.monotonic() + ttl, value, size)
            self._local_bytes += size
        self._sweep()
        if self._redis:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # synchronous caller: write now
                self._redis_store(key, value, ttl)
            else:  # never block the event loop on Redis (or on serialising a large value)
                pending = loop.run_in_executor(None, self._redis_store, key, value, ttl)
                self._pending_writes.add(pending)
                pending.add_done_callback(self._pending_writes.discard)

    @property
    def entry_count(self) -> int:
        # (No __len__: callers use 'cache or CacheManager(...)', and an empty cache must stay truthy.)
        return len(self._local_cache)

    @property
    def local_bytes(self) -> int:
        return self._local_bytes

    def delete(self, key: str) -> None:
        """Invalidate cached entry."""
        self._drop(key)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all cache keys."""
        self._local_cache.clear()
        self._local_bytes = 0
        if self._redis:
            try:
                keys = list(self._redis.scan_iter(match="astrosearch:cache:*"))
                if keys:
                    self._redis.delete(*keys)
            except Exception:
                pass

    @staticmethod
    def make_key(*args: Any) -> str:
        """Generate deterministic SHA-256 cache key."""
        payload = json.dumps(args, sort_keys=True)
        return "astrosearch:cache:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Base Catalog Provider
# ---------------------------------------------------------------------------


class QueryResult(list):
    """List of CatalogSource objects that also carries per-query metadata.

    ``meta`` holds status ('success' | 'empty'), row counts, truncation flag,
    the exact query sent, elapsed time, and citation text. It is a plain list
    subclass so existing callers that expect ``list[CatalogSource]`` keep working.
    """

    def __init__(self, sources: list[CatalogSource] | None = None, meta: dict[str, Any] | None = None) -> None:
        super().__init__(sources or [])
        self.meta: dict[str, Any] = dict(meta or {})


def source_sort_key(source: CatalogSource) -> tuple[float, int, str]:
    """Nearest first; coincident rows (within 0.1 mas) put non-planets first, then by id.

    SIMBAD lists a host star and its planets at identical coordinates (differences of
    1e-17 deg), so separation alone makes the 'nearest' identity depend on float noise.
    """
    meta = source.metadata
    sep = meta.get("epoch_separation_arcsec")
    if sep is None:
        sep = meta.get("query_separation_arcsec") or 0.0
    otype = str((meta.get("physical") or {}).get("object_type") or "").strip().lower()
    return (round(float(sep) / _TIE_ARCSEC) * _TIE_ARCSEC, 1 if otype in _PLANET_TYPES else 0, source.source_id)


def _exclusions(catalog: CatalogDefinition) -> dict[str, set[str]]:
    """``parameters.exclude_values`` as {column: {normalized value, ...}}."""
    raw = catalog.parameters.get("exclude_values") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(col): {_norm_value(v) for v in (values or [])} for col, values in raw.items()}


def _norm_value(value: Any) -> str:
    number = None
    if not isinstance(value, bool):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
    if number is not None and math.isfinite(number):
        return repr(number)
    return str(value).strip()


def _excluded(row: dict[str, Any], exclusions: dict[str, set[str]]) -> bool:
    for column, values in exclusions.items():
        value = row_get(row, column)
        if value is not None and _norm_value(value) in values:
            return True
    return False


def split_by_radius(sources: list[CatalogSource], radius_arcsec: float) -> tuple[list[CatalogSource], list[CatalogSource]]:
    """Split sources into (inside radius, outside radius) by epoch-corrected separation.

    Sources outside are flagged ``metadata["outside_radius"] = True``.
    """
    limit = radius_arcsec * (1.0 + 1e-9) + 1e-6
    inside: list[CatalogSource] = []
    outside: list[CatalogSource] = []
    for src in sources:
        sep = src.metadata.get("epoch_separation_arcsec")
        if sep is None:
            sep = src.metadata.get("query_separation_arcsec")
        if sep is None or sep <= limit:
            src.metadata.pop("outside_radius", None)
            inside.append(src)
        else:
            src.metadata["outside_radius"] = True
            outside.append(src)
    return inside, outside


@dataclass
class RowSplit:
    """Result of :func:`classify_sources`: in-radius rows (nearest ``max_rows``), rows in
    radius beyond ``max_rows`` (``excess``), pad rows outside the radius, and flags."""

    inside: list[CatalogSource]
    excess: list[CatalogSource]
    pad: list[CatalogSource]
    truncated: bool
    warnings: list[str]


def classify_sources(
    sources: list[CatalogSource],
    target: Target | None,
    requested_radius_arcsec: float,
    *,
    catalog_name: str,
    max_rows: int,
    row_limit: int,
    archive_truncated: bool,
    cone_center: tuple[float, float] | None = None,
    epoch_span: tuple[float, float] | None = None,
) -> RowSplit:
    """Epoch-correct, order and split a catalog's rows for ``target``.

    Every row is kept: rows beyond ``max_rows`` inside the radius (``excess``) and all
    pad rows are returned so a later re-split (after a proper motion is adopted) can
    still find them. ``truncated`` means the result is not the complete in-radius set:
    the archive held more rows than were fetched, or more than ``max_rows`` lie inside.

    When a widened cone was cut by the archive and the target's proper motion is known
    (given or adopted), the cut only matters if the target's track over the catalog's
    ``epoch_span`` reaches beyond the fetched rows (nearest-first around ``cone_center``);
    otherwise no warning is issued.
    """
    rows = list(sources)
    would_match = 0
    if target is not None:
        limit = requested_radius_arcsec * (1.0 + 1e-9) + 1e-6
        for src in rows:
            esep, method = epoch_separation_arcsec(target, src)
            src.metadata["epoch_separation_arcsec"] = esep
            src.metadata["epoch_propagation"] = method
            src.metadata.pop("target_pm_separation_arcsec", None)
            if method == "stationary":
                # Not moved with the target's motion (a field object without a measured
                # motion); report where it would be if it WERE the target.
                alt = target_pm_separation_arcsec(target, src)
                src.metadata["target_pm_separation_arcsec"] = alt
                if alt is not None and alt <= limit < esep:
                    would_match += 1
        rows.sort(key=source_sort_key)
    inside, pad = split_by_radius(rows, requested_radius_arcsec)
    excess = inside[max_rows:]
    inside = inside[:max_rows]
    truncated = bool(archive_truncated or excess)
    warnings: list[str] = []
    if would_match:
        warnings.append(
            f"{catalog_name}: {would_match} row(s) without a proper motion (in a catalog that has them) were treated "
            f"as stationary; they would lie within {requested_radius_arcsec:g} arcsec only if they moved with the "
            "target (see metadata target_pm_separation_arcsec)."
        )
    if truncated:
        if pad and archive_truncated and not _track_covered(rows, target, requested_radius_arcsec, cone_center, epoch_span):
            warnings.append(
                f"{catalog_name}: the widened cone was cut at {row_limit} rows; a fast-moving target "
                "beyond them may be missing."
            )
        if inside and (excess or not pad):
            seps = [s.metadata.get("epoch_separation_arcsec") for s in inside]
            seps = [x if x is not None else (s.metadata.get("query_separation_arcsec") or 0.0) for x, s in zip(seps, inside)]
            warnings.append(
                f"{catalog_name}: truncated at the {len(inside)} nearest rows, which cover only "
                f"{max(seps):.2f} of the requested {requested_radius_arcsec:g} arcsec; "
                "matches and counterparts beyond that are missing."
            )
    return RowSplit(inside, excess, pad, truncated, warnings)


def _track_covered(
    rows: list[CatalogSource],
    target: Target | None,
    radius_arcsec: float,
    cone_center: tuple[float, float] | None,
    epoch_span: tuple[float, float] | None,
) -> bool:
    """True when every position the target can occupy in the catalog lies inside the
    region covered by the fetched rows (which are the nearest ones to ``cone_center``)."""
    if target is None or target.epoch is None or target.proper_motion is None or cone_center is None or epoch_span is None:
        return False
    if not rows:
        return False
    covered = max(haversine_arcsec(cone_center[0], cone_center[1], s.ra, s.dec) for s in rows)
    pm = target.proper_motion
    need = max(
        haversine_arcsec(cone_center[0], cone_center[1], *propagate_radec(target.ra, target.dec, pm[0], pm[1], target.epoch, t))
        for t in epoch_span
    ) + radius_arcsec
    return need <= covered


class CatalogProvider(ABC):
    """Abstract base class for astronomical archive providers."""

    @abstractmethod
    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        raise NotImplementedError


_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_RATE_LIMIT_STATUS = {408, 429}
_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.WriteError)
_TIMEOUT_EXCEPTIONS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)
# Rows whose separations differ by less than this are treated as coincident (ties).
_TIE_ARCSEC = 1e-4
_PLANET_TYPES = frozenset({"pl", "pl?"})


_CACHEABLE_HEADERS = frozenset({"content-type", "retry-after", "date", "last-modified", "etag"})


def _cacheable_headers(headers: Any) -> dict[str, str]:
    """Headers safe to replay with an already-decoded body (no encoding/length headers)."""
    return {str(k).lower(): str(v) for k, v in dict(headers).items() if str(k).lower() in _CACHEABLE_HEADERS}


def service_error_detail(response: httpx.Response, limit: int = 300) -> str:
    """Extract a human-readable error message from an archive error response."""
    content = response.content[:200_000]
    ctype = response.headers.get("content-type", "").lower()
    _status, message = votable_query_status(content)
    if message:
        return message[:limit]
    if "json" in ctype:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key in ("ErrorMessage", "message", "error", "detail", "msg"):
                    if data.get(key):
                        return str(data[key])[:limit]
        except ValueError:
            pass
    text = content.decode("utf-8", "replace")
    title = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if title:
        return " ".join(title.group(1).split())[:limit]
    return " ".join(text.split())[:limit]


class _HTTPProvider(CatalogProvider):
    """Base HTTP provider managing connection pooling, exponential retry, and caching."""

    provider_name = "unknown"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float = 30.0,
        max_response_bytes: int = 10_000_000,
        guards: dict[str, EndpointGuard] | None = None,
        cache: CacheManager | None = None,
    ) -> None:
        self.client = client or new_http_client(timeout)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.guards = guards if guards is not None else {}
        self.cache = cache or CacheManager(os.getenv("REDIS_URL"))

    # -- transport ----------------------------------------------------------

    def _guard(self, endpoint: str) -> EndpointGuard:
        return self.guards.setdefault(
            endpoint,
            EndpointGuard(
                requests_per_second=float(os.getenv("PROVIDER_REQUESTS_PER_SECOND", "5")),
                failure_threshold=int(os.getenv("PROVIDER_FAILURE_THRESHOLD", "5")),
                recovery_seconds=float(os.getenv("PROVIDER_RECOVERY_SECONDS", "30")),
                probe_timeout_seconds=float(os.getenv("PROVIDER_PROBE_TIMEOUT_SECONDS", "180")),
            ),
        )

    @staticmethod
    def _budget(timeout: float | None) -> float | None:
        """Total time allowed for one request incl. retries.

        Slightly shorter than the executor's per-catalog limit (which uses the same
        configured value) so the provider's own timeout path runs first and is
        counted by the circuit breaker.
        """
        if timeout is None:
            return None
        # Margin of 5% (at least 0.1 s, at most 2 s): comfortably above event-loop
        # and OS timer jitter (~16 ms on Windows) so the provider path wins the race.
        return max(0.001, timeout - min(2.0, max(0.1, 0.05 * timeout)))

    def _is_query_error(self, response: httpx.Response) -> bool:
        """True when a 5xx response is a deterministic query error (not retried, not an outage)."""
        return False

    async def _request(
        self,
        method: str,
        endpoint: str,
        provider: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[httpx.Response, str, bool]:
        """Send a request with caching, rate limiting, and retry on transient failures.

        Returns (response, cache_key, from_cache). 5xx responses that persist after
        retries are returned (not raised) so the caller can report the archive's
        own error text; network failures raise CatalogUnavailableError and read
        timeouts (or exhausting the time budget) raise QueryTimeoutError.

        Every exit reports to the endpoint's circuit breaker exactly once -- including
        cancellation by the executor and unexpected exceptions -- so a half-open probe
        can never be left dangling.
        """
        cache_key = self.cache.make_key("provider", method.upper(), endpoint, params, data)
        reader = getattr(self.cache, "aget", None)
        cached = await reader(cache_key) if reader is not None else self.cache.get(cache_key)
        if cached is not None:
            request = httpx.Request(method.upper(), endpoint, params=params, data=data)
            response = httpx.Response(
                cached["status"],
                # The cached body is already decoded: never replay content/transfer encodings
                # (entries written by older versions may still carry them).
                headers=_cacheable_headers(cached.get("headers") or {}),
                content=base64.b64decode(cached["content"]),
                request=request,
            )
            return response, cache_key, True

        guard = self._guard(endpoint)
        await guard.acquire()

        budget = self._budget(timeout)
        kwargs: dict[str, Any] = {}
        if params is not None:
            kwargs["params"] = params
        if data is not None:
            kwargs["data"] = data
        if timeout is not None:
            kwargs["timeout"] = min(timeout, budget) if budget is not None else timeout

        settled = False
        last_error: Exception | None = None
        last_response: httpx.Response | None = None
        try:
            async with asyncio.timeout(budget):
                for attempt in range(3):
                    try:
                        response = await self.client.request(method.upper(), endpoint, **kwargs)
                    except _TIMEOUT_EXCEPTIONS as exc:
                        settled = True
                        guard.record_failure()
                        raise QueryTimeoutError(f"{provider} request timed out: {exc.__class__.__name__}") from exc
                    except _RETRY_EXCEPTIONS as exc:
                        last_error = exc
                        if attempt < 2:
                            await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    except httpx.HTTPError as exc:
                        # Any other transport/protocol failure (DecodingError, TooManyRedirects,
                        # UnsupportedProtocol, ProxyError, LocalProtocolError, ...): the archive
                        # could not be used, so report it as unavailable (fallbacks apply).
                        settled = True
                        guard.record_failure()
                        raise CatalogUnavailableError(
                            f"{provider} request failed: {exc.__class__.__name__}: {exc}"
                        ) from exc
                    if response.status_code not in _RETRY_STATUS or self._is_query_error(response):
                        # The endpoint answered (4xx and deterministic query errors included).
                        settled = True
                        guard.record_success()
                        return response, cache_key, False
                    last_response = response
                    last_error = CatalogUnavailableError(f"{provider} HTTP {response.status_code}")
                    if attempt < 2:
                        retry_after = response.headers.get("retry-after")
                        try:
                            delay = min(float(retry_after), 5.0) if retry_after else 0.25 * (2**attempt)
                        except ValueError:
                            delay = 0.25 * (2**attempt)
                        await asyncio.sleep(max(0.0, delay))
        except TimeoutError as exc:
            if not settled:
                settled = True
                guard.record_failure()
            raise QueryTimeoutError(f"{provider} request exceeded its {budget:g}s time budget") from exc
        except BaseException:
            # Cancellation (executor wait_for), programming errors, ...: never leave a
            # half-open probe dangling or a hung request uncounted.
            if not settled:
                settled = True
                guard.record_failure()
            raise

        guard.record_failure()
        if last_response is not None:
            return last_response, cache_key, False
        raise CatalogUnavailableError(f"{provider} request failed after 3 attempts: {last_error!r}") from last_error

    def _remember(self, cache_key: str, response: httpx.Response) -> None:
        """Cache a response only after it parsed successfully (never cache errors).

        ``response.content`` is the *decoded* body, so headers describing the wire
        encoding (content-encoding, transfer-encoding, content-length) are not stored:
        replaying 'content-encoding: gzip' with plain bytes makes httpx fail to decode.
        """
        if response.status_code == 200 and len(response.content) <= self.max_response_bytes:
            self.cache.set(
                cache_key,
                {
                    "status": response.status_code,
                    "headers": _cacheable_headers(response.headers),
                    "content": base64.b64encode(response.content).decode(),
                },
                ttl=int(os.getenv("PROVIDER_CACHE_TTL_SECONDS", "600")),
            )

    async def _get(self, endpoint: str, provider: str, **kwargs: Any) -> httpx.Response:
        """Backward-compatible GET helper (caches every HTTP 200 response)."""
        response, key, cached = await self._request("GET", endpoint, provider, params=kwargs.get("params"))
        if not cached:
            self._remember(key, response)
        return response

    # -- response validation --------------------------------------------------

    @staticmethod
    def _check(response: httpx.Response, provider: str) -> None:
        """Raise on HTTP errors.

        5xx -> CatalogUnavailableError; 429/408 (throttled) -> RateLimitedError (a
        CatalogUnavailableError, so fallbacks apply); other 4xx -> CatalogQueryError.
        """
        if response.status_code < 400:
            return
        detail = service_error_detail(response)
        message = f"{provider} query failed: HTTP {response.status_code}" + (f": {detail}" if detail else "")
        if response.status_code in _RATE_LIMIT_STATUS:
            retry_after = response.headers.get("retry-after")
            raise RateLimitedError(message + (f" (Retry-After: {retry_after})" if retry_after else ""))
        if response.status_code >= 500:
            raise CatalogUnavailableError(message)
        raise CatalogQueryError(message)

    def _check_size(self, response: httpx.Response, provider: str) -> None:
        if len(response.content) > self.max_response_bytes:
            raise CatalogQueryError(f"{provider} response exceeded {self.max_response_bytes} byte limit.")

    @staticmethod
    def _parse_table(response: httpx.Response, provider: str, expected: str | None = None) -> ParsedTable:
        """Parse a response body according to its content type (JSON, VOTable, CSV, IPAC)."""
        ctype = response.headers.get("content-type", "").lower()
        head = response.content[:512].lstrip().lower()
        if "html" in ctype or head.startswith((b"<!doctype html", b"<html")):
            raise CatalogQueryError(f"{provider} returned an HTML page instead of data: {service_error_detail(response)}")
        try:
            if "xml" in ctype or "votable" in ctype or head.startswith((b"<?xml", b"<votable")):
                return parse_votable_table(response.content)
            if expected == "ipac" or head.startswith((b"\\", b"|")):
                return parse_ipac_table(response.content)
            if "csv" in ctype or expected == "csv":
                return parse_csv_table(response.content)
            return parse_json_table(response.content)
        except (CatalogQueryError, ResponseParseError):
            raise
        except Exception as exc:
            raise ResponseParseError(f"{provider} response could not be parsed: {exc}") from exc

    @staticmethod
    def _apply_column_units(catalog: CatalogDefinition, table: ParsedTable) -> ParsedTable:
        """Correct archive column units the registry knows to be wrong (parameters.column_units)."""
        overrides = {str(k).lower(): str(v) for k, v in (catalog.parameters.get("column_units") or {}).items()}
        if overrides:
            for col in table.columns:
                unit = overrides.get(col.name.lower())
                if unit is not None:
                    col.unit = unit
        return table

    # -- row conversion -------------------------------------------------------

    @staticmethod
    def _apply_sentinels(row: dict[str, Any], sentinels: list[Any]) -> dict[str, Any]:
        if not sentinels:
            return row
        return {k: (None if v is not None and not isinstance(v, bool) and v in sentinels else v) for k, v in row.items()}

    @staticmethod
    def _field_map(catalog: CatalogDefinition, default_id: str | None = None) -> dict[str, str]:
        params = catalog.parameters
        mapping: dict[str, str] = {}
        for canonical, key, default in (("ra", "ra_field", None), ("dec", "dec_field", None), ("source_id", "id_field", default_id)):
            value = params.get(key, default)
            if value:
                mapping[canonical] = bare_column_name(str(value))
        mapping.update({str(k): str(v) for k, v in (params.get("field_map") or {}).items()})
        return mapping

    def _positional_error(
        self,
        catalog: CatalogDefinition,
        row: dict[str, Any],
        normalized: dict[str, Any],
        dec: float,
        columns: list[ColumnMeta] | None,
        positional_error_key: str | None,
    ) -> tuple[float | None, dict[str, Any]]:
        if catalog.pos_error:
            return compute_positional_error(row, catalog.pos_error, dec_deg=dec, columns=columns)
        legacy_key = catalog.parameters.get("positional_error_field") or positional_error_key
        if legacy_key and row_get(row, str(legacy_key)) is not None:
            col = find_column(columns, str(legacy_key))
            unit = catalog.parameters.get("positional_error_unit") or (col.unit if col and col.unit else "arcsec")
            value = angle_to_arcsec(row_get(row, str(legacy_key)), unit)
            return value, {"kind": "legacy", "columns": [legacy_key], "units": [unit], "sigma_arcsec": value, "source": "catalog"}
        errs = [normalized.get("ra_error_arcsec"), normalized.get("dec_error_arcsec")]
        present = [e for e in errs if e is not None and e >= 0]
        if present:
            value = math.sqrt(sum(e * e for e in present) / len(present))
            return value, {"kind": "sigma", "columns": "ucd", "sigma_ra_arcsec": errs[0], "sigma_dec_arcsec": errs[1],
                           "sigma_arcsec": value, "source": "ucd"}
        value = normalized.get("position_uncertainty_arcsec")
        if value is not None:
            return value, {"kind": "alias", "sigma_arcsec": value, "source": "catalog"}
        return None, {}

    def _sources(
        self,
        catalog: CatalogDefinition,
        rows: list[dict[str, Any]],
        radius_arcsec: float,
        endpoint: str | None,
        parameters: dict[str, Any],
        positional_error_key: str | None = None,
        *,
        columns: list[ColumnMeta] | None = None,
        target: Target | None = None,
        field_map: dict[str, str] | None = None,
        meta: dict[str, Any] | None = None,
        cone: ConePlan | None = None,
    ) -> QueryResult:
        """Convert heterogeneous provider rows into canonical CatalogSource objects.

        Rows lacking a usable position are counted (``dropped_rows``, with a warning);
        if *every* row is unusable a ResponseParseError is raised instead of silently
        returning nothing.

        With a target, rows are ordered by epoch-corrected separation (ties broken
        deterministically) and the result holds the nearest ``catalog.max_rows`` rows
        inside the requested radius. Nothing fetched is discarded: in-radius rows beyond
        ``max_rows`` go to ``meta["excess_sources"]`` and rows outside the radius --
        fetched only because the cone was widened for proper motion -- to
        ``meta["pad_sources"]`` (never counted in ``status``/``row_count``). The
        crossmatch service re-splits all of them once it has the final target model
        (e.g. an adopted proper motion), so no counterpart is lost to an early cut.

        ``null_sentinels`` (e.g. PS1 -999) are replaced by None in ``data`` as well as in
        the normalized values, so exports and dataset filters never see them as values.
        """
        sentinels = list(catalog.parameters.get("null_sentinels") or [])
        mapping = field_map if field_map is not None else self._field_map(catalog)
        exclude = _exclusions(catalog)
        has_pm = catalog_has_proper_motions(catalog)
        single_epoch = bool(catalog.parameters.get("single_epoch_positions"))
        sources: list[CatalogSource] = []
        dropped = 0
        filtered = 0
        integer_ids = _integer_id_columns(catalog)
        for index, raw_row in enumerate(rows):
            row = self._apply_sentinels(dict(raw_row), sentinels)
            if integer_ids:
                _integer_ids(row, integer_ids)
            if exclude and _excluded(row, exclude):
                filtered += 1
                continue
            try:
                normalized = normalize_source_record(row, columns=columns, field_map=mapping)
                ra = float(str(normalized["ra"])) % 360.0
                dec = float(str(normalized["dec"]))
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            if not math.isfinite(ra) or not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
                dropped += 1
                continue
            if ra >= 360.0:  # float modulo of tiny negatives (e.g. -1e-14 % 360 == 360.0)
                ra = 0.0

            source_id = normalized.get("source_id")
            source_id = str(source_id) if source_id not in (None, "") else f"{catalog.name}-{index}"
            pos_err, err_details = self._positional_error(catalog, row, normalized, dec, columns, positional_error_key)
            epoch = resolve_epoch(row, catalog.epoch, catalog.epoch_format, columns)
            if epoch is None and not isinstance(catalog.epoch, dict):
                epoch = normalized.get("epoch")
            epoch_range = resolve_epoch_range(row, catalog, columns) if epoch is None else None
            separation = haversine_arcsec(target.ra, target.dec, ra, dec) if target is not None else None

            sources.append(
                CatalogSource(
                    catalog=catalog.name,
                    source_id=source_id,
                    ra=ra,
                    dec=dec,
                    positional_error_arcsec=pos_err,
                    data=row,
                    metadata={
                        "wavelength": catalog.wavelength,
                        "table": catalog.table,
                        "catalog": catalog.catalog,
                        # How rows without their own proper motion are epoch-corrected
                        # (see models.source_position_at).
                        META_CATALOG_HAS_PM: has_pm,
                        META_SINGLE_EPOCH: single_epoch,
                        # Separation from the target at the coordinates given (no epoch correction).
                        "query_separation_arcsec": separation,
                        "positional_error": err_details,
                        "citation": catalog.citation,
                        "physical": {
                            key: normalized[key]
                            for key in (
                                "parallax", "redshift", "object_type", "spectral_type",
                                "morphology", "observation_date", "quality_flags",
                            )
                            if normalized.get(key) is not None
                        },
                        "links": {
                            "SIMBAD": f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote(source_id)}"
                            if catalog.name == "simbad" else None,
                            "NED": f"https://ned.ipac.caltech.edu/byname?objname={quote(source_id)}"
                            if catalog.name == "ned" else None,
                            "MAST": f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}%20{dec}"
                            if catalog.provider == "mast" else None,
                            "IRSA": f"https://irsa.ipac.caltech.edu/applications/finderchart/servlet/api?locstr={ra}%20{dec}"
                            if catalog.provider == "irsa_gator" or "irsa.ipac" in str(catalog.endpoint) else None,
                            "LegacySurvey": f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&pixscale=0.262&bands=griz"
                            if catalog.wavelength in {"optical", "extragalactic"} else None,
                        },
                    },
                    provenance=build_provenance(
                        catalog.name,
                        provider=self.provider_name,
                        source_id=source_id,
                        endpoint=endpoint,
                        query_parameters=parameters,
                        search_radius_arcsec=radius_arcsec,
                    ),
                    epoch=epoch,
                    proper_motion_ra_masyr=normalized.get("pmra"),
                    proper_motion_dec_masyr=normalized.get("pmdec"),
                    position_uncertainty_arcsec=pos_err,
                    epoch_range=epoch_range,
                )
            )

        if rows and not sources and filtered < len(rows):
            raise ResponseParseError(
                f"{catalog.name}: parsed {len(rows)} row(s) but none had a usable RA/Dec "
                f"(columns: {sorted(rows[0].keys())[:20]})."
            )
        warnings = list(cone.warnings) if cone is not None else []
        if dropped:
            message = f"{catalog.name}: {dropped} of {len(rows)} archive row(s) had no usable RA/Dec and were dropped."
            warnings.append(message)
            logger.warning(message)

        requested = cone.requested_radius_arcsec if cone is not None else radius_arcsec
        row_limit = cone.row_limit if cone is not None else catalog.max_rows
        # Archive truncation: the server said OVERFLOW or returned the extra probe row.
        archive_truncated = bool((meta or {}).get("truncated")) or len(rows) > row_limit
        epoch_span = catalog_epoch_range(catalog)
        cone_center = (cone.ra, cone.dec) if cone is not None else (target.ra, target.dec) if target is not None else None
        split = classify_sources(
            sources, target, requested, catalog_name=catalog.name, max_rows=catalog.max_rows,
            row_limit=row_limit, archive_truncated=archive_truncated, cone_center=cone_center, epoch_span=epoch_span,
        )
        for message in split.warnings:
            logger.warning(message)

        info = dict(meta or {})
        info.update({
            "status": "success" if split.inside else "empty",
            "row_count": len(split.inside),
            "raw_row_count": len(rows),
            "dropped_rows": dropped,
            # Rows removed by the catalog's exclude_values (e.g. VLASS 'Redundant' duplicates).
            "filtered_rows": filtered,
            "pad_row_count": len(split.pad),
            "pad_sources": split.pad,
            "excess_row_count": len(split.excess),
            "excess_sources": split.excess,
            "archive_truncated": archive_truncated,
            "cone_center": cone_center,
            "epoch_span": epoch_span,
            "row_limit": row_limit,
            "max_rows": catalog.max_rows,
            # Cone/parse warnings; truncation warnings are recomputed on every re-split.
            "base_warnings": list(warnings),
            "requested_radius_arcsec": requested,
            "query_radius_arcsec": cone.radius_arcsec if cone is not None else requested,
            "cone": cone.as_dict() if cone is not None else None,
            "truncated": split.truncated,
            "warnings": warnings + split.warnings,
            "citation": catalog.citation,
            "acknowledgement": catalog.acknowledgement,
            "epoch": catalog.epoch,
            "epoch_range": catalog.parameters.get("epoch_range"),
        })
        return QueryResult(split.inside, info)

    def _timeout(self, catalog: CatalogDefinition) -> float:
        return float(catalog.timeout_seconds or self.timeout)


# ---------------------------------------------------------------------------
# Specific Archive Providers
# ---------------------------------------------------------------------------


def _column_list(value: Any, default: list[str]) -> list[str]:
    columns = value or default
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",") if c.strip()]
    return [str(col) for col in columns]


class TapProvider(_HTTPProvider):
    """IVOA Table Access Protocol (TAP) adapter querying databases via ADQL.

    Sends POST form requests (avoids URL-length limits and WAF false positives),
    selects a server-side DISTANCE() alias, and orders nearest-first with TOP N.
    Catalog ``parameters``: columns, ra_field, dec_field, id_field, field_map,
    format (json|votable|csv), distance (deg|arcsec|none), where, http_method,
    radius_pad_arcsec.
    """

    provider_name = "tap"

    def build_adql(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float, cone: ConePlan | None = None) -> str:
        """ADQL cone query: TOP max_rows+1 (so a full cone is not mistaken for a truncated one)."""
        cone = cone or plan_cone(catalog, target, radius_arcsec)
        params = catalog.parameters
        columns = _column_list(params.get("columns"), ["source_id", "ra", "dec"])
        ra_expr = str(params.get("ra_field", "ra"))
        dec_expr = str(params.get("dec_field", "dec"))
        id_expr = params.get("id_field", "source_id")
        selected = {bare_column_name(c).lower() for c in columns}
        for expr in (ra_expr, dec_expr, id_expr):
            if expr and bare_column_name(str(expr)).lower() not in selected:
                columns.append(str(expr))
                selected.add(bare_column_name(str(expr)).lower())

        radius_deg = cone.radius_arcsec / 3600.0
        point = f"POINT('ICRS', {ra_expr}, {dec_expr})"
        center = f"POINT('ICRS', {cone.ra:.9f}, {cone.dec:.9f})"
        distance_mode = str(params.get("distance", "deg"))
        if distance_mode != "none":
            columns.append(f"DISTANCE({point}, {center}) AS match_dist")
        where = f"1 = CONTAINS({point}, CIRCLE('ICRS', {cone.ra:.9f}, {cone.dec:.9f}, {radius_deg:.10f}))"
        if params.get("where"):
            where += f" AND ({params['where']})"
        order = " ORDER BY match_dist ASC" if distance_mode != "none" else ""
        top = max(1, int(cone.row_limit)) + 1
        table = catalog.table or "gaiadr3.gaia_source"
        return f"SELECT TOP {top} {', '.join(columns)} FROM {table} WHERE {where}{order}"

    def build_request(self, catalog: CatalogDefinition, adql: str) -> dict[str, str]:
        fmt = str(catalog.parameters.get("format", "json"))
        form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql}
        if fmt in {"json", "csv"}:
            form["FORMAT"] = fmt
        return form

    # Oracle-backed TAP services (NASA Exoplanet Archive, IRSA) evaluate DISTANCE() with
    # ACOS and fail with ORA-01428 when rounding puts its argument a hair above 1, i.e.
    # when the cone centre lies within a few mas of a row (e.g. Barnard b at the target's
    # propagated J2015.5 position, found live). The query is then repeated with the centre moved by
    # a negligible CENTRE_NUDGE_ARCSEC (the radius grows by the same amount).
    CENTRE_NUDGE_ARCSEC = 0.05  # 1 - cos(0.05") = 3e-14, far above double rounding

    @staticmethod
    def _acos_domain_error(response: httpx.Response) -> bool:
        return response.status_code == 400 and b"ORA-01428" in response.content[:20_000]

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        endpoint = catalog.endpoint or "https://gea.esac.esa.int/tap-server/tap/sync"
        cone = plan_cone(catalog, target, radius_arcsec)
        method = str(catalog.parameters.get("http_method", "POST")).upper()
        for attempt in range(2):
            adql = self.build_adql(catalog, target, radius_arcsec, cone)
            form = self.build_request(catalog, adql)
            response, key, cached = await self._request(
                method,
                endpoint,
                "TAP",
                params=form if method == "GET" else None,
                data=form if method != "GET" else None,
                timeout=self._timeout(catalog),
            )
            if attempt == 0 and self._acos_domain_error(response):
                nudge = self.CENTRE_NUDGE_ARCSEC
                cone = replace(cone, dec=cone.dec + (nudge if cone.dec < 0 else -nudge) / 3600.0,
                               radius_arcsec=cone.radius_arcsec + nudge,
                               warnings=[*cone.warnings])
                continue
            break
        self._check(response, f"TAP {catalog.name}")
        self._check_size(response, f"TAP {catalog.name}")
        table = self._apply_column_units(
            catalog, self._parse_table(response, f"TAP {catalog.name}", str(catalog.parameters.get("format", "json")))
        )
        result = self._sources(
            catalog,
            table.rows,
            radius_arcsec,
            endpoint,
            form,
            columns=table.columns,
            target=target,
            cone=cone,
            meta={
                "query": adql,
                "endpoint": endpoint,
                "http_method": method,
                "format": table.format,
                "truncated": table.truncated,
                "query_status": table.query_status,
                "cached": cached,
                "columns": [c.as_dict() for c in table.columns],
            },
        )
        if not cached:
            self._remember(key, response)
        return result


class IRSAGatorProvider(_HTTPProvider):
    """IRSA Gator cone search adapter (2MASS PSC, AllWISE); fast fallback for IRSA TAP.

    Gator returns the whole cone unordered, so rows are sorted locally and trimmed to
    ``max_rows`` (the nearest N, exactly like the TAP path), with ``truncated`` set
    when the cone held more.
    """

    provider_name = "irsa_gator"

    def build_params(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float,
                     cone: ConePlan | None = None) -> dict[str, str]:
        cone = cone or plan_cone(catalog, target, radius_arcsec)
        params = {
            "catalog": catalog.catalog or catalog.table or catalog.name,
            "spatial": "cone",
            "objstr": f"{cone.ra:.9f} {cone.dec:.9f}",
            "radius": f"{cone.radius_arcsec:.6f}",
            "radunits": "arcsec",
            "outfmt": "1",
        }
        columns = catalog.parameters.get("columns")
        if columns:
            params["selcols"] = ",".join(bare_column_name(c) for c in _column_list(columns, []))
        return params

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        endpoint = catalog.endpoint or "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
        cone = plan_cone(catalog, target, radius_arcsec)
        params = self.build_params(catalog, target, radius_arcsec, cone)
        response, key, cached = await self._request("GET", endpoint, "IRSA", params=params, timeout=self._timeout(catalog))
        self._check(response, "IRSA")
        self._check_size(response, "IRSA")
        text = response.text
        if "[struct stat=" in text[:500] and "ERROR" in text[:500].upper():
            raise CatalogQueryError(f"IRSA Gator error: {' '.join(text[:300].split())}")
        table = self._apply_column_units(catalog, self._parse_table(response, "IRSA", "ipac"))
        result = self._sources(
            catalog, table.rows, radius_arcsec, endpoint, params,
            columns=table.columns, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id="designation"),
            meta={"endpoint": endpoint, "format": table.format, "cached": cached,
                  "columns": [c.as_dict() for c in table.columns]},
        )
        if not cached:
            self._remember(key, response)
        return result


# Integer identifier columns of an archive that serialises them inconsistently: the MAST
# catalogs API returns Pan-STARRS DR2 objID (a 64-bit integer beyond 2**53) as a JSON number
# in one answer and as a string in another (seen live: a sky-cache mirror held
# '110561872774570857' where the archive then answered 110561872774570857). Such values are
# converted to int by every adapter reading the catalog (remote and sky cache), so one object
# always has one data record and one content hash.
_INTEGER_ID_COLUMNS = {"mast": ("objID",)}


def _integer_id_columns(catalog: CatalogDefinition) -> tuple[str, ...]:
    configured = catalog.parameters.get("integer_id_columns")
    if configured:
        return tuple(str(c) for c in configured)
    return _INTEGER_ID_COLUMNS.get(catalog.provider, ())


def _integer_ids(row: dict[str, Any], names: tuple[str, ...]) -> None:
    """Integer identifiers written as decimal strings become ints (in place; any key case)."""
    wanted = {n.lower() for n in names}
    for key, value in row.items():
        if key.lower() in wanted and isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                row[key] = int(text)


class MASTProvider(_HTTPProvider):
    """STScI MAST catalogs API adapter (Pan-STARRS DR2 mean objects).

    Requests explicit columns including ``distance`` (in DEGREES despite its
    label) so that ``sort_by=distance`` is honoured, filters artefacts with
    ``nDetections.gt``, and zips the ``{"info": [...], "data": [[...]]}`` rows.
    """

    provider_name = "mast"

    def build_params(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float,
                     cone: ConePlan | None = None) -> dict[str, str]:
        cone = cone or plan_cone(catalog, target, radius_arcsec)
        params: dict[str, str] = {
            "ra": f"{cone.ra:.9f}",
            "dec": f"{cone.dec:.9f}",
            "radius": f"{cone.radius_arcsec / 3600.0:.10f}",
        }
        columns = _column_list(catalog.parameters.get("columns"), [])
        if columns:
            if "distance" not in columns:
                columns.append("distance")
            params["columns"] = json.dumps(columns)
            params["sort_by"] = "distance"
        params["pagesize"] = str(max(1, int(cone.row_limit)) + 1)
        for key, value in (catalog.parameters.get("filters") or {}).items():
            params[str(key)] = str(value)
        return params

    @staticmethod
    def _missing_columns(requested: list[str], table: ParsedTable) -> list[str]:
        """Requested columns MAST silently left out of its ``info`` (typos, renamed columns)."""
        if not table.columns:
            return []
        returned = {c.name.lower() for c in table.columns}
        return [c for c in requested if c.lower() not in returned]

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        endpoint = catalog.endpoint or "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
        cone = plan_cone(catalog, target, radius_arcsec)
        params = self.build_params(catalog, target, radius_arcsec, cone)
        response, key, cached = await self._request("GET", endpoint, "MAST", params=params, timeout=self._timeout(catalog))
        self._check(response, "MAST")
        self._check_size(response, "MAST")
        table = self._apply_column_units(catalog, self._parse_table(response, "MAST", "json"))
        missing = self._missing_columns(json.loads(params["columns"]) if "columns" in params else [], table)
        if missing:
            # Unlike TAP services, MAST answers HTTP 200 and drops unknown columns.
            raise CatalogQueryError(f"MAST {catalog.name}: requested column(s) not returned: {', '.join(missing)}")
        result = self._sources(
            catalog, table.rows, radius_arcsec, endpoint, params,
            columns=table.columns, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id="objID"),
            meta={"endpoint": endpoint, "format": table.format, "cached": cached,
                  "columns": [c.as_dict() for c in table.columns]},
        )
        if not cached:
            self._remember(key, response)
        return result


class SDSSProvider(_HTTPProvider):
    """Sloan Digital Sky Survey (SDSS) SkyServer adapter.

    Default mode 'sqlsearch' runs T-SQL with ``dbo.fGetNearbyObjEq(ra, dec, r)``
    whose radius is explicitly in ARCMINUTES, ordered by distance. Legacy mode
    'conesearch' calls ConeSearchService whose ``sr`` is, despite its VOTable
    PARAM claiming degrees, applied in ARCMINUTES (verified live: sr=1 -> 60").
    """

    provider_name = "sdss"
    default_joins = (
        "LEFT OUTER JOIN Photoz AS pz ON pz.objID = n.objID "
        "LEFT OUTER JOIN SpecObj AS s ON s.bestObjID = n.objID"
    )

    def build_sql(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float, cone: ConePlan | None = None) -> str:
        cone = cone or plan_cone(catalog, target, radius_arcsec)
        columns = _column_list(catalog.parameters.get("columns"), ["p.ra", "p.dec"])
        radius_arcmin = cone.radius_arcsec / 60.0
        photo = catalog.table or "PhotoPrimary"
        joins = catalog.parameters.get("joins", self.default_joins)
        return (
            f"SELECT TOP {max(1, int(cone.row_limit)) + 1} n.objID AS objID, n.distance*60.0 AS dist_arcsec, "
            f"{', '.join(columns)} "
            f"FROM dbo.fGetNearbyObjEq({cone.ra:.9f}, {cone.dec:.9f}, {radius_arcmin:.10f}) AS n "
            f"JOIN {photo} AS p ON p.objID = n.objID {joins} "
            "ORDER BY n.distance"
        )

    @staticmethod
    def _sql_error(response: httpx.Response) -> bool:
        return response.status_code >= 500 and "ErrorMessage" in response.text[:5000]

    def _is_query_error(self, response: httpx.Response) -> bool:
        # SkyServer reports SQL errors as HTTP 500 + JSON ErrorMessage: deterministic,
        # so they are neither retried nor counted against the endpoint's breaker.
        return self._sql_error(response)

    @staticmethod
    def _check(response: httpx.Response, provider: str) -> None:
        if SDSSProvider._sql_error(response):
            raise CatalogQueryError(f"{provider} SQL error: {service_error_detail(response)}")
        _HTTPProvider._check(response, provider)

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        endpoint = catalog.endpoint or "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
        mode = str(catalog.parameters.get("mode") or ("conesearch" if "ConeSearch" in endpoint else "sqlsearch"))
        cone = plan_cone(catalog, target, radius_arcsec)
        if mode == "conesearch":
            params = {
                "format": "csv",
                "ra": f"{cone.ra:.9f}",
                "dec": f"{cone.dec:.9f}",
                "sr": f"{cone.radius_arcsec / 60.0:.10f}",  # arcmin (see class doc)
            }
            expected = "csv"
            query_text = None
        else:
            query_text = self.build_sql(catalog, target, radius_arcsec, cone)
            params = {"cmd": query_text, "format": "json"}
            expected = "json"
        response, key, cached = await self._request("GET", endpoint, "SDSS", params=params, timeout=self._timeout(catalog))
        self._check(response, "SDSS")
        self._check_size(response, "SDSS")
        table = self._apply_column_units(catalog, self._parse_table(response, "SDSS", expected))
        result = self._sources(
            catalog, table.rows, radius_arcsec, endpoint, params,
            columns=table.columns, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id="objID" if mode != "conesearch" else "objid"),
            meta={"endpoint": endpoint, "query": query_text, "mode": mode, "format": table.format, "cached": cached},
        )
        if not cached:
            self._remember(key, response)
        return result


class HEASARCXaminProvider(_HTTPProvider):
    """NASA HEASARC Xamin positional search adapter (legacy; prefer HEASARC TAP).

    Xamin ignores ``coord=``: the position must be sent as ``position=RA,DEC``
    and ``radius`` is in ARCMINUTES. Rows are unordered, so they are sorted locally.
    """

    provider_name = "heasarc_xamin"

    def build_params(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float,
                     cone: ConePlan | None = None) -> dict[str, str]:
        cone = cone or plan_cone(catalog, target, radius_arcsec)
        return {
            "table": catalog.table or catalog.name,
            "position": f"{cone.ra:.9f},{cone.dec:.9f}",
            "radius": f"{cone.radius_arcsec / 60.0:.8f}",
            "format": "json",
        }

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> QueryResult:
        endpoint = catalog.endpoint or "https://heasarc.gsfc.nasa.gov/xamin/query"
        cone = plan_cone(catalog, target, radius_arcsec)
        params = self.build_params(catalog, target, radius_arcsec, cone)
        response, key, cached = await self._request("GET", endpoint, "HEASARC Xamin", params=params, timeout=self._timeout(catalog))
        self._check(response, "HEASARC Xamin")
        self._check_size(response, "HEASARC Xamin")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResponseParseError(f"HEASARC Xamin response is not JSON: {exc}") from exc
        if isinstance(payload, dict) and payload.get("success") is False:
            raise CatalogQueryError(f"HEASARC Xamin error: {payload.get('messages') or payload}")
        table = parse_json_table(payload)
        if table.format == "json-object":
            # A JSON object without a row array is never a valid (even empty) result:
            # Xamin answers a real empty search with {"request": []}. Anything else is an
            # error document and must not become a silent 'empty'.
            detail = payload.get("messages") or payload.get("error") or payload if isinstance(payload, dict) else payload
            raise CatalogQueryError(f"HEASARC Xamin returned no result rows: {str(detail)[:300]}")
        rows = table.rows
        result = self._sources(
            catalog, rows, radius_arcsec, endpoint, params, target=target, cone=cone,
            field_map=self._field_map(catalog, default_id="name"),
            meta={"endpoint": endpoint, "format": table.format, "cached": cached},
        )
        if not cached:
            self._remember(key, response)
        return result


# ---------------------------------------------------------------------------
# Object-Name Resolver: CDS Sesame
# ---------------------------------------------------------------------------


class SesameResolver:
    """Resolve astronomical object names through the CDS Sesame web service."""

    name = "cds_sesame"
    default_endpoint = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"

    def __init__(self, client: httpx.AsyncClient | None = None, *, endpoint: str | None = None) -> None:
        # Without a client one is built on first use (:meth:`_http`), not here: constructing a
        # client can load the CA bundle, which must not happen synchronously on the event loop.
        self.client: httpx.AsyncClient | None = client or None
        self.endpoint = endpoint or self.default_endpoint

    async def _http(self) -> httpx.AsyncClient:
        """The caller's client, else a fallback one sharing the process's SSL context (built off
        the event loop the first time; see :func:`new_http_client_async`)."""
        if self.client is None:
            self.client = await new_http_client_async(30.0)
        return self.client

    async def resolve(self, query: str) -> ResolvedObject:
        clean_query = str(query).strip()
        if not clean_query:
            raise ObjectResolutionError("Object name must not be empty.")
        url = f"{self.endpoint}?{quote(clean_query, safe='')}"
        try:
            response = await (await self._http()).get(url, headers={"Accept": "application/xml, text/xml"})
        except httpx.HTTPError as exc:
            # The resolver could not be reached: not evidence that the name is unknown.
            raise ResolverUnavailableError(f"Sesame request failed: {exc}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            raise ResolverUnavailableError(f"Sesame request failed: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise ObjectResolutionError(f"Sesame request failed: HTTP {response.status_code}")
        try:
            answer = self.parse_response(clean_query, response.text, endpoint=self.endpoint)
        except (ElementTree.ParseError, ValueError, TypeError) as exc:
            raise ObjectResolutionError(f"Sesame response could not be parsed: {exc}") from exc
        return await self._retry_simbad(clean_query, answer)

    async def _retry_simbad(self, query: str, answer: ResolvedObject) -> ResolvedObject:
        """Sesame's -oxp/SNV returns the first sub-resolver that answers: when SIMBAD did not
        (an outage or a slow answer) a name SIMBAD knows can come back from VizieR -- an
        undated catalogue position without errors or motion, possibly one of several
        answers ('GJ 860 A' -> '{Name} Gl 860', 50" from Kruger 60 A). Such an answer is
        checked once more against SIMBAD alone (-oxp/S): its answer replaces the fallback,
        which is otherwise kept (``resolver_metadata['simbad_retry']`` records the attempt)."""
        meta = answer.resolver_metadata
        kind = self.answer_kind(meta)
        match = re.search(r"/-ox?p?/([A-Za-z]+)$", self.endpoint.rstrip("/"))
        if kind in ("simbad", "ned") or match is None or "S" not in match.group(1).upper() \
                or match.group(1).upper() == "S":
            return answer
        endpoint = self.endpoint.rstrip("/")[: match.start(1)] + "S"
        try:
            response = await (await self._http()).get(f"{endpoint}?{quote(query, safe='')}",
                                                      headers={"Accept": "application/xml, text/xml"})
            retried = self.parse_response(query, response.text, endpoint=endpoint) if response.status_code < 400 else None
        except Exception as exc:  # noqa: BLE001 - the fallback answer stands; the attempt is recorded
            meta["simbad_retry"] = {"endpoint": endpoint, "error": f"{exc.__class__.__name__}: {exc}"}
            return answer
        if retried is None or self.answer_kind(retried.resolver_metadata) != "simbad":
            meta["simbad_retry"] = {"endpoint": endpoint, "error": f"HTTP {response.status_code}" if retried is None
                                    else "no SIMBAD answer"}
            return answer
        retried.resolver_metadata["simbad_retry"] = {"endpoint": endpoint, "replaced": meta.get("resolver_name")}
        return retried

    @staticmethod
    def answer_kind(resolver_metadata: dict[str, Any] | None) -> str | None:
        """Which Sesame sub-resolver answered: 'simbad', 'ned', 'vizier' (an undated catalogue
        position without errors or motion), another name, or None when unknown."""
        name = str((resolver_metadata or {}).get("resolver_name") or "").lower()
        if not name:
            return None
        for kind in ("simbad", "ned", "vizier"):
            if kind in name:
                return kind
        return name.split("=")[-1].split()[0] if name.split("=")[-1].split() else None

    @staticmethod
    def multiple_answers(resolver_metadata: dict[str, Any] | None) -> int | None:
        """Number of objects Sesame found for the name ('++++Multiple (2) answers++++' in its
        INFO, kept in ``raw_fields['info']``; the first was returned), or None."""
        info = ((resolver_metadata or {}).get("raw_fields") or {}).get("info") or []
        for text in info if isinstance(info, list) else [info]:
            match = re.search(r"Multiple\s*\((\d+)\)", str(text), re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    # SIMBAD object types that are extragalactic (proper motions of these are noise).
    EXTRAGALACTIC_TYPES = EXTRAGALACTIC_OTYPES

    @classmethod
    def parse_response(cls, query: str, payload: str, *, endpoint: str | None = None) -> ResolvedObject:
        """Parse a Sesame ``-oxp`` XML answer (one <Resolver> per service tried).

        Structured elements are flattened with dotted keys: ``<pm><pmRA>`` ->
        ``pm.pmra``, ``<pm><pmDE>`` -> ``pm.pmde``, ``<z><v>`` -> ``z.v``,
        ``<plx><v>`` -> ``plx.v``, ``<Vel><v>`` -> ``vel.v``. SIMBAD positions
        (jradeg/jdedeg) are ICRS at epoch J2000, so ``epoch`` is 2000.0 for SIMBAD
        answers and None (unknown) for other resolvers. Extragalactic objects -- an
        extragalactic object type (QSO, BLL, Sy1, G, ...), or a redshift and no parallax --
        are flagged stationary even when SIMBAD lists a (Gaia-noise) parallax and proper
        motion for them (3C 273: BLL, plx 0.011 mas). Proper motions of everything else
        are marked applicable.
        """
        root = ElementTree.fromstring(payload)
        records = [el for el in root.iter() if isinstance(el.tag, str)
                   and cls._local_name(el.tag) in {"resolver", "result", "object"}]
        if not records:
            records = [root]

        for record in records:
            values = cls._flatten(record)

            ra = cls._number(values, "jradeg", "ra", "ra_deg")
            dec = cls._number(values, "jdedeg", "dec", "dec_deg")
            if ra is None or dec is None:
                continue
            if not math.isfinite(ra) or not math.isfinite(dec) or not -90 <= dec <= 90:
                continue

            aliases = cls._all_values(values, "alias", "aliases")
            canonical = cls._first(values, "oname", "name", "canonical_name") or (aliases[0] if aliases else query)
            # VizieR answers carry the name template of the catalogue column: '{Name} Gl 860'.
            canonical = " ".join(re.sub(r"^\s*\{[^}]*\}\s*", "", canonical).split()) or query
            resolver_name = str(record.get("name") or "")
            object_type = cls._first(values, "otype", "otyp", "object_type")
            redshift = cls._number(values, "z.v", "z_value", "redshift", "z")
            parallax = cls._number(values, "plx.v", "plx")
            velocity = cls._number(values, "vel.v")
            pm_ra = cls._number(values, "pm.pmra", "pmra", "pm_ra")
            pm_dec = cls._number(values, "pm.pmde", "pmde", "pmdec", "pm_dec")
            epoch = cls._number(values, "epoch", "ref_epoch", "obsepoch")
            epoch_source = "explicit" if epoch is not None else None
            if epoch is None and "simbad" in resolver_name.lower():
                epoch, epoch_source = 2000.0, "SIMBAD positions are ICRS at epoch J2000"
            extragalactic = is_extragalactic_type(object_type) or (redshift is not None and parallax is None)
            pm_applicable = pm_ra is not None and pm_dec is not None and not extragalactic

            return ResolvedObject(
                query=query,
                canonical_name=canonical,
                ra_deg=ra % 360.0,
                dec_deg=dec,
                aliases=sorted({" ".join(a.split()) for a in aliases} - {canonical}),
                object_type=object_type,
                redshift=redshift,
                pm_ra_masyr=pm_ra,
                pm_dec_masyr=pm_dec,
                epoch=epoch,
                resolver=cls.name,
                resolver_metadata={
                    "endpoint": endpoint or cls.default_endpoint,
                    "resolver_name": resolver_name or None,
                    "parallax_mas": parallax,
                    "radial_velocity_kms": velocity,
                    "extragalactic": extragalactic,
                    "proper_motion_applicable": pm_applicable,
                    "epoch_source": epoch_source,
                    "raw_fields": values,
                },
            )
        raise ValueError(f"No coordinates found for object {query!r}.")

    @classmethod
    def _flatten(cls, element: ElementTree.Element, prefix: str = "",
                 out: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
        out = {} if out is None else out
        for child in element:
            if not isinstance(child.tag, str):  # comments / processing instructions
                continue
            key = prefix + cls._local_name(child.tag)
            text = (child.text or "").strip()
            if text:
                out.setdefault(key, []).append(text)
            if len(child):
                cls._flatten(child, key + ".", out)
        return out

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].strip().lower()

    @staticmethod
    def _first(values: dict[str, list[str]], *keys: str) -> str | None:
        for k in keys:
            if values.get(k):
                return values[k][0]
        return None

    @classmethod
    def _number(cls, values: dict[str, list[str]], *keys: str) -> float | None:
        val = cls._first(values, *keys)
        if val is None:
            return None
        try:
            res = float(val)
        except (TypeError, ValueError):
            return None
        return res if math.isfinite(res) else None

    @classmethod
    def _all_values(cls, values: dict[str, list[str]], *keys: str) -> list[str]:
        result: list[str] = []
        for k in keys:
            result.extend(values.get(k, []))
        return result


# ---------------------------------------------------------------------------
# Factory: Provider Map
# ---------------------------------------------------------------------------


def provider_map(
    client: httpx.AsyncClient | None = None,
    *,
    timeout: float = 30.0,
    max_response_bytes: int = 10_000_000,
    guards: dict[str, EndpointGuard] | None = None,
    cache: CacheManager | None = None,
) -> dict[str, CatalogProvider]:
    """Instantiate all catalog provider adapters with shared client, guards, and cache."""
    shared_guards = guards if guards is not None else {}
    shared_cache = cache or CacheManager(os.getenv("REDIS_URL"))

    def make(cls: type[_HTTPProvider]) -> _HTTPProvider:
        return cls(
            client,
            timeout=timeout,
            max_response_bytes=max_response_bytes,
            guards=shared_guards,
            cache=shared_cache,
        )

    return {
        "tap": make(TapProvider),
        "irsa_gator": make(IRSAGatorProvider),
        "mast": make(MASTProvider),
        "sdss": make(SDSSProvider),
        "heasarc_xamin": make(HEASARCXaminProvider),
    }
