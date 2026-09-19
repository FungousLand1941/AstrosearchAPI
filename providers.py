"""Public astronomy archive adapters, CDS Sesame resolver, resilience, and caching layer."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from models import (
    CatalogDefinition,
    CatalogQueryError,
    CatalogSource,
    CatalogUnavailableError,
    ObjectResolutionError,
    ResolvedObject,
    ResponseParseError,
    Target,
    build_provenance,
    normalize_source_record,
    parse_csv_records,
    parse_ipac_records,
    parse_json_records,
    parse_votable_records,
)

# ---------------------------------------------------------------------------
# Resilience: Rate Limiter and Circuit Breaker
# ---------------------------------------------------------------------------


@dataclass
class EndpointGuard:
    """A fixed-interval request pacer combined with a three-state circuit breaker.

    One guard instance is shared per provider endpoint. It throttles request starts
    and transitions to 'open' when consecutive failure thresholds are exceeded.
    """

    requests_per_second: float = 5.0
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _next_start: float = 0.0
    _failures: int = 0
    _opened_at: float | None = None
    _probe_in_flight: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        """Wait for rate limit interval and ensure the circuit is not open."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        async with self._lock:
            now = self.clock()
            if self._opened_at is not None:
                if now - self._opened_at < self.recovery_seconds or self._probe_in_flight:
                    raise CatalogUnavailableError("Provider circuit is open")
                self._probe_in_flight = True
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + 1.0 / self.requests_per_second
        if delay:
            await asyncio.sleep(delay)

    async def succeed(self) -> None:
        """Signal successful execution, resetting circuit breaker failures."""
        async with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    async def fail(self) -> None:
        """Record an endpoint failure and open the circuit if threshold is reached."""
        async with self._lock:
            self._failures += 1
            if self._probe_in_flight or self._failures >= self.failure_threshold:
                self._opened_at = self.clock()
            self._probe_in_flight = False

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


class CacheManager:
    """Manages caching for provider responses and crossmatch queries."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url
        self._local_cache: dict[str, tuple[float, Any]] = {}
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
            except Exception:
                self._redis = None

    def get(self, key: str) -> Any | None:
        """Retrieve cached value from Redis or local memory cache."""
        if self._redis:
            try:
                val = self._redis.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
        entry = self._local_cache.get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        self._local_cache.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Store value with TTL in seconds."""
        if ttl <= 0:
            return
        self._local_cache[key] = (time.monotonic() + ttl, value)
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
            except Exception:
                pass

    def delete(self, key: str) -> None:
        """Invalidate cached entry."""
        self._local_cache.pop(key, None)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all cache keys."""
        self._local_cache.clear()
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


class CatalogProvider(ABC):
    """Abstract base class for astronomical archive providers."""

    @abstractmethod
    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        raise NotImplementedError


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
        self.client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self.max_response_bytes = max_response_bytes
        self.guards = guards if guards is not None else {}
        self.cache = cache or CacheManager(os.getenv("REDIS_URL"))

    async def _get(self, endpoint: str, provider: str, **kwargs: Any) -> httpx.Response:
        """Execute GET with caching, rate limiting, and exponential retry on transient failures."""
        cache_key = self.cache.make_key("provider", endpoint, kwargs.get("params"))
        cached = self.cache.get(cache_key)
        if cached is not None:
            req = httpx.Request("GET", endpoint, params=kwargs.get("params"))
            return httpx.Response(
                cached["status"],
                headers=cached["headers"],
                content=base64.b64decode(cached["content"]),
                request=req,
            )

        guard = self.guards.setdefault(
            endpoint,
            EndpointGuard(
                requests_per_second=float(os.getenv("PROVIDER_REQUESTS_PER_SECOND", "5")),
                failure_threshold=int(os.getenv("PROVIDER_FAILURE_THRESHOLD", "5")),
                recovery_seconds=float(os.getenv("PROVIDER_RECOVERY_SECONDS", "30")),
            ),
        )
        await guard.acquire()

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.get(endpoint, **kwargs)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    await guard.succeed()
                    if response.status_code == 200 and len(response.content) <= self.max_response_bytes:
                        self.cache.set(
                            cache_key,
                            {
                                "status": response.status_code,
                                "headers": dict(response.headers),
                                "content": base64.b64encode(response.content).decode(),
                            },
                            ttl=int(os.getenv("PROVIDER_CACHE_TTL_SECONDS", "600")),
                        )
                    return response
                last_error = CatalogQueryError(f"{provider} query failed: HTTP {response.status_code}")
                if attempt < 2:
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = min(float(retry_after), 5.0) if retry_after else 0.25 * (2**attempt)
                    except ValueError:
                        delay = 0.25 * (2**attempt)
                    await asyncio.sleep(max(0.0, delay))
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.25 * (2**attempt))

        await guard.fail()
        raise CatalogQueryError(f"{provider} request failed after 3 attempts: {last_error}") from last_error

    def _sources(
        self,
        catalog: CatalogDefinition,
        rows: list[dict[str, Any]],
        radius_arcsec: float,
        endpoint: str | None,
        parameters: dict[str, Any],
        positional_error_key: str | None = None,
    ) -> list[CatalogSource]:
        """Convert heterogeneous provider rows into canonical CatalogSource objects."""
        sources: list[CatalogSource] = []
        for row in rows:
            try:
                normalized = normalize_source_record(row)
            except (TypeError, ValueError):
                continue
            if "ra" not in normalized or "dec" not in normalized:
                continue
            try:
                ra = float(str(normalized["ra"])) % 360.0
                dec = float(str(normalized["dec"]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ra) or not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
                continue

            source_id = str(normalized.get("source_id", f"{catalog.name}-{len(sources)}"))
            err_val = normalized.get("position_uncertainty_arcsec")
            if err_val is None and positional_error_key:
                err_val = row.get(positional_error_key)
            try:
                pos_err = float(str(err_val)) if err_val not in (None, "") else None
            except (TypeError, ValueError):
                pos_err = None

            sources.append(
                CatalogSource(
                    catalog=catalog.name,
                    source_id=source_id,
                    ra=ra,
                    dec=dec,
                    positional_error_arcsec=pos_err,
                    data=dict(row),
                    metadata={
                        "wavelength": catalog.wavelength,
                        "table": catalog.table,
                        "catalog": catalog.catalog,
                        "physical": {
                            key: normalized[key]
                            for key in (
                                "parallax", "redshift", "object_type", "spectral_type",
                                "morphology", "observation_date", "quality_flags",
                            )
                            if key in normalized
                        },
                        "links": {
                            "SIMBAD": f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote(source_id)}"
                            if catalog.name == "simbad" else None,
                            "NED": f"https://ned.ipac.caltech.edu/byname?objname={quote(source_id)}"
                            if catalog.name == "ned" else None,
                            "MAST": f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}%20{dec}"
                            if catalog.provider == "mast" else None,
                            "IRSA": f"https://irsa.ipac.caltech.edu/applications/finderchart/servlet/api?locstr={ra}%20{dec}"
                            if catalog.provider == "irsa_gator" else None,
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
                    epoch=normalized.get("epoch"),
                    proper_motion_ra_masyr=normalized.get("pmra"),
                    proper_motion_dec_masyr=normalized.get("pmdec"),
                    position_uncertainty_arcsec=pos_err,
                )
            )
        return sources

    @staticmethod
    def _check(response: httpx.Response, provider: str) -> None:
        if response.status_code >= 400:
            raise CatalogQueryError(f"{provider} query failed: HTTP {response.status_code}")

    def _check_size(self, response: httpx.Response, provider: str) -> None:
        if len(response.content) > self.max_response_bytes:
            raise CatalogQueryError(f"{provider} response exceeded {self.max_response_bytes} byte limit.")


# ---------------------------------------------------------------------------
# Specific Archive Providers
# ---------------------------------------------------------------------------


class TapProvider(_HTTPProvider):
    """IVOA Table Access Protocol (TAP) adapter querying databases via ADQL."""

    provider_name = "tap"

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        endpoint = catalog.endpoint or "https://gea.esac.esa.int/tap-server/tap/sync"
        radius_deg = radius_arcsec / 3600.0
        params = catalog.parameters
        columns = params.get("columns") or ["source_id", "ra", "dec"]
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(",") if c.strip()]
        columns = [str(col) for col in columns]

        ra_field = str(params.get("ra_field", "ra"))
        dec_field = str(params.get("dec_field", "dec"))
        id_field = str(params.get("id_field", "source_id"))
        pos_err_field = params.get("positional_error_field")

        for f in (ra_field, dec_field, id_field):
            if f not in columns:
                columns.append(f)

        adql = (
            "SELECT TOP 100 " + ", ".join(columns) + " FROM "
            f"{catalog.table or 'gaiadr3.gaia_source'} "
            "WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', "
            f"{target.ra}, {target.dec}, {radius_deg})) = 1"
        )
        adql = adql.replace("POINT('ICRS', ra, dec)", f"POINT('ICRS', {ra_field}, {dec_field})")
        req_params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}

        response = await self._get(endpoint, "TAP", params=req_params)
        self._check(response, "TAP")
        self._check_size(response, "TAP")

        try:
            ct = response.headers.get("content-type", "").lower()
            if "votable" in ct or "xml" in ct:
                rows = parse_votable_records(response.content)
            elif "csv" in ct:
                rows = parse_csv_records(response.text)
            else:
                rows = parse_json_records(response.text)
        except Exception as exc:
            raise ResponseParseError(f"TAP response could not be parsed: {exc}") from exc

        mapped_rows = []
        for row in rows[:100]:
            mapped = dict(row)
            if ra_field in mapped:
                mapped["ra"] = mapped[ra_field]
            if dec_field in mapped:
                mapped["dec"] = mapped[dec_field]
            if id_field in mapped:
                mapped["source_id"] = mapped[id_field]
            mapped_rows.append(mapped)

        return self._sources(catalog, mapped_rows, radius_arcsec, endpoint, req_params, pos_err_field or "ra_error")


class IRSAGatorProvider(_HTTPProvider):
    """IRSA Gator cone search adapter (2MASS PSC, AllWISE)."""

    provider_name = "irsa_gator"

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        endpoint = catalog.endpoint or "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
        params = {
            "catalog": catalog.catalog or catalog.name,
            "spatial": "cone",
            "objstr": f"{target.ra} {target.dec}",
            "radius": str(radius_arcsec),
            "radunits": "arcsec",
            "outfmt": "1",
        }
        response = await self._get(endpoint, "IRSA", params=params)
        self._check(response, "IRSA")
        self._check_size(response, "IRSA")
        try:
            rows = parse_ipac_records(response.content)
        except Exception as exc:
            raise ResponseParseError(f"IRSA response could not be parsed: {exc}") from exc
        return self._sources(catalog, rows, radius_arcsec, endpoint, params)


class MASTProvider(_HTTPProvider):
    """STScI MAST API adapter (Pan-STARRS DR2)."""

    provider_name = "mast"

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        endpoint = catalog.endpoint or "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean"
        params = {"ra": str(target.ra), "dec": str(target.dec), "radius": str(radius_arcsec / 3600.0)}
        response = await self._get(endpoint, "MAST", params=params)
        self._check(response, "MAST")
        self._check_size(response, "MAST")
        try:
            rows = parse_json_records(response.json())
        except Exception as exc:
            raise ResponseParseError(f"MAST response could not be parsed: {exc}") from exc
        return self._sources(catalog, rows, radius_arcsec, endpoint, params, "raError")


class SDSSProvider(_HTTPProvider):
    """Sloan Digital Sky Survey (SDSS) SkyServer Cone Search adapter."""

    provider_name = "sdss"

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        endpoint = catalog.endpoint or "https://skyserver.sdss.org/dr18/SkyServerWS/ConeSearch/ConeSearchService"
        params = {"format": "csv", "ra": str(target.ra), "dec": str(target.dec), "sr": str(radius_arcsec / 60.0)}
        response = await self._get(endpoint, "SDSS", params=params)
        self._check(response, "SDSS")
        self._check_size(response, "SDSS")
        try:
            rows = parse_csv_records(response.text)
        except Exception as exc:
            raise ResponseParseError(f"SDSS response could not be parsed: {exc}") from exc
        return self._sources(catalog, rows, radius_arcsec, endpoint, params)


class HEASARCXaminProvider(_HTTPProvider):
    """NASA HEASARC Xamin positional search adapter (FIRST, NVSS, ROSAT, Chandra, XMM)."""

    provider_name = "heasarc_xamin"

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> list[CatalogSource]:
        endpoint = catalog.endpoint or "https://heasarc.gsfc.nasa.gov/xamin/query"
        params = {
            "table": catalog.table or catalog.name,
            "coord": f"{target.ra},{target.dec}",
            "radius": str(radius_arcsec),
            "format": "json",
        }
        response = await self._get(endpoint, "HEASARC Xamin", params=params)
        self._check(response, "HEASARC Xamin")
        self._check_size(response, "HEASARC Xamin")
        try:
            rows = parse_json_records(response.json())
        except Exception as exc:
            raise ResponseParseError(f"HEASARC Xamin response could not be parsed: {exc}") from exc
        return self._sources(catalog, rows, radius_arcsec, endpoint, params, "poserr")


# ---------------------------------------------------------------------------
# Object-Name Resolver: CDS Sesame
# ---------------------------------------------------------------------------


class SesameResolver:
    """Resolve astronomical object names through the CDS Sesame web service."""

    name = "cds_sesame"
    default_endpoint = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"

    def __init__(self, client: httpx.AsyncClient | None = None, *, endpoint: str | None = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.endpoint = endpoint or self.default_endpoint

    async def resolve(self, query: str) -> ResolvedObject:
        clean_query = str(query).strip()
        if not clean_query:
            raise ObjectResolutionError("Object name must not be empty.")
        url = f"{self.endpoint}?{quote(clean_query, safe='')}"
        try:
            response = await self.client.get(url, headers={"Accept": "application/xml, text/xml"})
        except httpx.HTTPError as exc:
            raise ObjectResolutionError(f"Sesame request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ObjectResolutionError(f"Sesame request failed: HTTP {response.status_code}")
        try:
            return self.parse_response(clean_query, response.text, endpoint=self.endpoint)
        except (ElementTree.ParseError, ValueError, TypeError) as exc:
            raise ObjectResolutionError(f"Sesame response could not be parsed: {exc}") from exc

    @classmethod
    def parse_response(cls, query: str, payload: str, *, endpoint: str | None = None) -> ResolvedObject:
        root = ElementTree.fromstring(payload)
        records = [el for el in root.iter() if cls._local_name(el.tag) in {"result", "object"}]
        if not records:
            records = [root]

        for record in records:
            values: dict[str, list[str]] = {}
            for element in record.iter():
                key = cls._local_name(element.tag)
                val = (element.text or "").strip()
                if val:
                    values.setdefault(key, []).append(val)

            ra = cls._number(values, "jradeg", "ra", "ra_deg")
            dec = cls._number(values, "jdedeg", "dec", "dec_deg")
            if ra is None or dec is None:
                continue
            if not math.isfinite(ra) or not math.isfinite(dec) or not -90 <= dec <= 90:
                continue

            aliases = cls._all_values(values, "alias", "aliases", "oid")
            canonical = cls._first(values, "oname", "name", "canonical_name") or (aliases[0] if aliases else query)

            return ResolvedObject(
                query=query,
                canonical_name=canonical,
                ra_deg=ra % 360.0,
                dec_deg=dec,
                aliases=sorted({a for a in aliases if a != canonical}),
                object_type=cls._first(values, "otyp", "otype", "object_type"),
                redshift=cls._number(values, "z_value", "redshift", "z"),
                pm_ra_masyr=cls._number(values, "pmra", "pm_ra"),
                pm_dec_masyr=cls._number(values, "pmdec", "pm_dec"),
                epoch=cls._number(values, "epoch", "ref_epoch", "obsepoch"),
                resolver=cls.name,
                resolver_metadata={"endpoint": endpoint or cls.default_endpoint, "raw_fields": values},
            )
        raise ValueError(f"No coordinates found for object {query!r}.")

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
