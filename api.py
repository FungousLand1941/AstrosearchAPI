"""AstroSearch REST API: the core search/dataset endpoints plus every feature router and the web UI.

Mounted routers (see DOCUMENTATION.md for each endpoint):

* ``/api/v1/search/stream`` (streaming), ``/api/v1/batch`` (batch), ``/api/v1/skycache``
  (skycache), ``/api/v1/vizier`` (vizier), ``/api/v1/sed`` (sed), ``/api/v1/lightcurves`` and
  ``/api/v1/solar-system`` (timedomain), ``/api/v1/cutouts`` (imaging), ``/api/v1/ai`` (ai),
  ``/api/v1/provenance`` and ``/api/v1/citations`` (provenance), ``/vo`` (vo_server),
  ``/api/v1/alerts`` (alerts);
* the single-page web UI at ``/`` (``imaging.mount_ui``).

The lifespan builds one HTTP client, one catalog registry (embedded + ``vizier add``
catalogs), one provider map (answering from the local sky cache for mirrored catalogs) and
one CrossmatchService, and publishes them on ``app.state`` for every router.
Authentication, quotas and the request-body limit apply to every route except
``/api/v1/health``, the VO availability endpoint and the UI files.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
import uuid
import weakref
from collections import defaultdict, deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

import ai
import alerts
import batch
import imaging
import provenance
import sed
import skycache
import streaming
import timedomain
import vizier
import vo_server
from crossmatch import AdvancedQuery, CrossmatchService, QueryValidator
from datasets import DatasetEngine, MetadataStore, enqueue_dataset, process_dataset_async, submit_to_redis
from main import api_search, build_providers, build_registry, build_service, skycache_store
from models import (
    CatalogQueryError,
    CatalogUnavailableError,
    InvalidCoordinateError,
    ObjectResolutionError,
    QueryTimeoutError,
    ResponseParseError,
    Settings,
    UnifiedRecord,
    resolution_failure_status,
)
from providers import CacheManager, SesameResolver

# ---------------------------------------------------------------------------
# Logging and Prometheus Metrics
# ---------------------------------------------------------------------------

logger = structlog.get_logger()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# The Anthropic SDK and astropy's cosmology are imported lazily by ai.py (CLI start-up): the API
# imports them now, before serving, so no request waits seconds for an import on the event loop.
ai.preload_heavy_dependencies()

REQUEST_COUNT = Counter("astrosearch_requests_total", "Total API requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("astrosearch_request_latency_seconds", "API request latency", ["method", "endpoint"])
ACTIVE_REQUESTS = Gauge("astrosearch_active_requests", "Active in-flight API requests", ["endpoint"])
CATALOG_QUERIES = Counter("astrosearch_catalog_queries_total", "Catalog query outcomes", ["catalog", "status"])
CATALOG_LATENCY = Histogram("astrosearch_catalog_query_seconds", "Catalog query latency", ["catalog"])

API_VERSION = "0.3.0"


def setup_metrics(application: FastAPI) -> None:
    application.state.metrics = {
        "request_count": REQUEST_COUNT,
        "request_latency": REQUEST_LATENCY,
        "active_requests": ACTIVE_REQUESTS,
    }


def record_catalog_metrics(catalog_stats: Mapping[str, Any]) -> None:
    """Count per-catalog outcomes (success/empty/failed; '+fallback' when a fallback
    archive answered or was tried) in astrosearch_catalog_queries_total, and latencies."""
    for name, stats in catalog_stats.items():
        if not isinstance(stats, Mapping):
            continue
        status_label = str(stats.get("status") or "unknown")
        if stats.get("fallback"):
            status_label += "+fallback"
        CATALOG_QUERIES.labels(catalog=name, status=status_label).inc()
        elapsed = stats.get("elapsed_ms")
        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
            CATALOG_LATENCY.labels(catalog=name).observe(float(elapsed) / 1000.0)


class MeteredCrossmatchService(CrossmatchService):
    """The API's CrossmatchService: every record it finalizes -- a search, a streamed search,
    a VO cone or ADQL query, a per-target dataset search, an SED or an AI explanation run on
    it -- counts its catalog outcomes in the Prometheus metrics."""

    records_catalog_metrics = True

    def finalize(self, ctx: Any, successes: Any, failures: Any) -> UnifiedRecord:
        record = super().finalize(ctx, successes, failures)
        record_catalog_metrics(record.provenance.get("catalog_stats") or {})
        return record


# ---------------------------------------------------------------------------
# Authentication & Request Quotas
# ---------------------------------------------------------------------------


class RequestQuota:
    """Sliding-window request quota tracker with Redis backend and in-memory fallback.

    The Redis client (redis.asyncio) is created lazily for each event loop that uses the
    quota, because asyncio connections belong to one loop (the server, or each TestClient).
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._redis_url = redis_url
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    def _redis(self) -> Any:
        if not self._redis_url:
            return None
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            try:
                import redis.asyncio as aioredis

                client = aioredis.from_url(self._redis_url)
            except Exception as exc:
                logging.getLogger(__name__).warning("Redis quota unavailable: %s", exc)
                return None
            self._clients[loop] = client
        return client

    async def allow(self, identity: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.time()
        bucket = int(now // 60)
        client = self._redis()
        if client is not None:
            key = f"astrosearch:quota:{identity}:{bucket}"
            try:
                count = await client.incr(key)
                if count == 1:
                    await client.expire(key, 120)
                return count <= limit
            except Exception as exc:
                logging.getLogger(__name__).warning("Redis quota unavailable: %s", exc)

        async with self._lock:
            events = self._events[identity]
            while events and events[0] <= now - 60:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True

    async def close(self) -> None:
        """Close the Redis client of the running event loop (call on shutdown)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        client = self._clients.pop(loop, None)
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


def authenticate(request: Request) -> str | JSONResponse:
    """Validate API key or JWT bearer token."""
    configured = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
    jwt_public_key = os.getenv("JWT_PUBLIC_KEY")
    jwt_secret = os.getenv("JWT_SECRET")
    required = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"

    if required and not (configured or jwt_public_key or jwt_secret):
        return JSONResponse({"detail": "API authentication is not configured"}, status_code=503)

    authorization = request.headers.get("authorization", "")
    supplied = request.headers.get("x-api-key") or authorization.removeprefix("Bearer ")

    if configured and any(hmac.compare_digest(supplied, k) for k in configured):
        return hashlib.sha256(supplied.encode()).hexdigest()[:20]

    if authorization.startswith("Bearer ") and (jwt_public_key or jwt_secret):
        import jwt

        signing_key = jwt_public_key or jwt_secret
        assert signing_key is not None
        try:
            claims = jwt.decode(
                authorization.removeprefix("Bearer "),
                signing_key,
                algorithms=["RS256" if jwt_public_key else "HS256"],
                audience=os.getenv("JWT_AUDIENCE") or None,
                issuer=os.getenv("JWT_ISSUER") or None,
                options={"require": ["exp", "sub"], "verify_aud": bool(os.getenv("JWT_AUDIENCE"))},
            )
            return hashlib.sha256(f"{claims.get('iss', '')}:{claims['sub']}".encode()).hexdigest()[:20]
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid bearer token"}, status_code=401)

    if configured or jwt_public_key or jwt_secret:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# FastAPI Lifespan & Application Setup
# ---------------------------------------------------------------------------

# app.state attributes the lifespan sets (and removes on shutdown).
STATE_OBJECTS = ("settings", "client", "registry", "providers", "service", "skycache", "engine", "metadata",
                 "sed_filters")
# Attributes the feature routers create lazily on app.state (cleared on shutdown so a later
# start never reuses objects bound to a closed client or event loop).
LAZY_STATE_OBJECTS = ("alert_store", "alert_service", "alerts_built_services", "alerts_client", "vo_jobs",
                      "vo_service", "vo_client", "vo_config", "anthropic", "anthropic_lock", "batch_guards",
                      "batch_upload_slots")


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = Settings()
    # Embedded catalogs + the catalogs registered with `vizier add` / POST /api/v1/vizier/register.
    # Every registry problem is logged; CATALOG_REGISTRY_STRICT=true refuses to start.
    registry = build_registry(settings=settings)
    store = skycache_store()
    state = application.state
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        providers = build_providers(settings=settings, client=client, store=store)
        service = build_service(settings=settings, registry=registry, providers=providers,
                                service_class=MeteredCrossmatchService)
        state.settings = settings
        state.client = client
        state.registry = registry  # the service's own object: vizier registrations reach it at once
        state.providers = providers
        state.service = service
        state.skycache = store
        state.engine = DatasetEngine(registry_path=settings.catalog_registry_path, service=service, registry=registry)
        state.metadata = state.engine.metadata
        state.sed_filters = sed.FilterCatalog()
        try:
            yield
        finally:
            await _close_module_state(application)
            for name in STATE_OBJECTS:
                if hasattr(state, name):
                    delattr(state, name)
    await state.quota.close()


async def _close_module_state(application: FastAPI) -> None:
    """Close what the feature routers opened on app.state: the Anthropic client, the VO UWS jobs
    and client, the alerts router's own client, then forget their lazily built objects."""
    state = application.state
    for closer in (ai.aclose_state_clients(state), vo_server.aclose_vo_state(application)):
        try:
            await closer
        except Exception as exc:
            logger.warning("shutdown_close_failed", error=str(exc))
    alerts_client = getattr(state, "alerts_client", None)
    if alerts_client is not None and not alerts_client.is_closed:
        await alerts_client.aclose()
    for name in LAZY_STATE_OBJECTS:
        if hasattr(state, name):
            delattr(state, name)


app = FastAPI(
    title="AstroSearch API",
    description="Astronomical catalog cross-matching, time-domain, imaging, VO and dataset generation service",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Response headers browsers may read cross-origin (the web UI served elsewhere, notebooks).
EXPOSED_HEADERS = [
    "X-Request-ID", "Retry-After", "Location", "Content-Disposition",
    "X-Batch-Request-Count", "X-Batch-Wall-Time-S", "X-Batch-Failed-Targets", "X-Unknown-Citations", "X-Unverified-References",
    "X-Cutout-Survey", "X-Cutout-Hips", "X-Cutout-Cache", "X-Cutout-Pixel-Scale-Arcsec", "X-Cutout-Cdelt-Deg",
    "X-Cutout-Projection", "X-Cutout-Source", "X-Cutout-Pixels", "X-Cutout-Coverage", "X-Cutout-Blank",
    "X-Cutout-Pixel-Units", "X-Cutout-Calibrated", "X-Cutout-Science-Survey", "X-Cutout-Science-Note",
    "X-Cutout-Degraded", "X-Cutout-Centre-Epoch", "X-Cutout-Centre-Offset-Arcsec", "X-Cutout-Centre-Note",
]


def cors_options() -> dict[str, Any]:
    """CORS settings from CORS_ORIGINS (comma-separated; empty or '*' = any origin) and
    CORS_ALLOW_CREDENTIALS (default true). Credentials are never allowed together with a
    wildcard origin: a browser would send cookies/Authorization to every site otherwise."""
    origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
    wildcard = not origins or "*" in origins
    credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").strip().lower() in {"1", "true", "yes"}
    return {
        "allow_origins": ["*"] if wildcard else origins,
        "allow_credentials": credentials and not wildcard,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": EXPOSED_HEADERS,
    }


setup_metrics(app)
cache = CacheManager(os.getenv("REDIS_URL"))
app.state.quota = RequestQuota(os.getenv("REDIS_URL"))


def get_service() -> CrossmatchService:
    service = getattr(app.state, "service", None)
    if service is None:
        service = build_service()
    return service


def get_engine() -> DatasetEngine:
    return getattr(app.state, "engine", None) or DatasetEngine()


def get_metadata() -> MetadataStore:
    return getattr(app.state, "metadata", None) or MetadataStore(
        local_dir=os.getenv("DATASET_STORAGE_PATH") or "datasets"
    )


# ---------------------------------------------------------------------------
# Middleware: Request Metrics, Quotas, Body Limits and Strict JSON
# ---------------------------------------------------------------------------

# Routes that need neither authentication nor quota: liveness probes. (The UI files are
# served ahead of this middleware by imaging.UIStaticMiddleware.)
PUBLIC_PATHS = frozenset({"/api/v1/health", "/vo/tap/availability"})


def owns_body_limit(path: str) -> bool:
    """Routes that read and limit their own request bodies: the batch upload (streamed, up to
    BATCH_MAX_UPLOAD_BYTES) and the VO services (DALI/UWS error documents, vo_server.MAX_REQUEST_BYTES)."""
    return batch.owns_request_body_limit(path) or path == "/vo" or path.startswith("/vo/")


class _NonFiniteNumber(ValueError):
    pass


def non_finite_json(body: bytes) -> str | None:
    """The first non-finite number (NaN, Infinity, 1e400) of a JSON body, or None.

    Python's json module accepts these tokens, but they are not JSON (RFC 8259) and no
    coordinate, radius or flux may be NaN or infinite. A body that is not valid JSON at all
    returns None: the route's own validation answers it.
    """
    def constant(token: str) -> float:
        raise _NonFiniteNumber(token)

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise _NonFiniteNumber(text)
        return value

    try:
        json.loads(body, parse_constant=constant, parse_float=number)
    except _NonFiniteNumber as exc:
        return str(exc)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    return None


def _is_json(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return content_type == "application/json" or content_type.endswith("+json")


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    started = time.monotonic()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    endpoint = request.url.path
    metrics = getattr(app.state, "metrics", {})

    if "active_requests" in metrics:
        metrics["active_requests"].labels(endpoint="all").inc()
    response: Response
    try:
        if endpoint not in PUBLIC_PATHS:
            identity = authenticate(request)
            if isinstance(identity, JSONResponse):
                response = identity
            elif not await app.state.quota.allow(identity, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))):
                response = JSONResponse(
                    {"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"}
                )
            else:
                structlog.contextvars.bind_contextvars(actor=identity)
                response = await _limited(request, call_next)
        else:
            response = await call_next(request)

        route = request.scope.get("route")
        endpoint_label = getattr(route, "path", endpoint if endpoint in PUBLIC_PATHS else "unmatched")
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            method=request.method,
            path=endpoint_label,
            resource=request.url.path,
            status=response.status_code,
        )
        if "request_count" in metrics:
            metrics["request_count"].labels(
                method=request.method, endpoint=endpoint_label, status_code=str(response.status_code)
            ).inc()
        return response
    finally:
        if "active_requests" in metrics:
            metrics["active_requests"].labels(endpoint="all").dec()
        if "request_latency" in metrics:
            metrics["request_latency"].labels(
                method=request.method,
                endpoint=endpoint_label if "endpoint_label" in locals() else "unmatched",
            ).observe(time.monotonic() - started)
        structlog.contextvars.clear_contextvars()


# CORS is the outermost middleware (Starlette wraps each added middleware around the ones
# added before it): a browser preflight (OPTIONS, never carrying X-API-Key/Authorization) is
# answered before authentication and quotas, and every response -- a 401, 413 or 429 too --
# carries the Access-Control-* headers, so browser code can read it.
app.add_middleware(CORSMiddleware, **cors_options())


async def _limited(request: Request, call_next) -> Response:
    """MAX_REQUEST_BYTES (default 1 MB) for every body a route does not limit itself, and
    strict JSON: a NaN/Infinity number in a JSON body is a 422, never a 500."""
    if owns_body_limit(request.url.path):
        return await call_next(request)
    maximum = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    length = request.headers.get("content-length")
    try:
        declared = int(length) if length else None
    except ValueError:
        return JSONResponse({"detail": "Invalid Content-Length header"}, status_code=400)
    if declared is not None and declared > maximum:
        return JSONResponse({"detail": "Request body too large"}, status_code=413)
    # Read incrementally (a chunked upload declares no Content-Length): stop with 413 as soon
    # as the running total passes the limit instead of buffering the whole stream first.
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > maximum:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        chunks.append(chunk)
    body = b"".join(chunks)
    # What Request.body() stores: the route (behind this middleware) is handed the same bytes.
    request._body = body
    if body and _is_json(request):
        token = await asyncio.to_thread(non_finite_json, body) if len(body) > 65536 else non_finite_json(body)
        if token is not None:
            return JSONResponse(status_code=422, content={"detail": [{
                "type": "json_invalid", "loc": ["body"], "input": token,
                "msg": f"JSON numbers must be finite (RFC 8259): {token} is not allowed"}]})
    return await call_next(request)


def _json_safe(value: Any) -> Any:
    """Non-finite floats (echoed back in validation errors) as strings, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with a JSON-serialisable detail: FastAPI's default handler echoes the offending
    input, and a non-finite float there (a NaN query parameter) would turn the 422 into a 500."""
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(_json_safe(list(exc.errors()))))})


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = API_VERSION


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ra: float = Field(..., description="Right ascension in degrees", ge=0, lt=360)
    dec: float = Field(..., description="Declination in degrees", ge=-90, le=90)
    epoch: float | None = Field(None, ge=1800, le=2200, description="Julian epoch for proper motion correction")


class SearchRequest(provenance.SearchFields):
    """Body of POST /api/v1/search (also of /search/batch items, saved queries and search
    manifests): provenance.SearchFields, so every route validates the same bounds -- ra in
    [0, 360), dec in [-90, 90], radius_arcsec at most MAX_SEARCH_RADIUS_ARCSEC (3600),
    proper-motion components together."""


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    profile: str = Field(..., description="Catalog profile to use")
    radius_arcsec: float = Field(10.0, description="Search radius in arcseconds", gt=0)
    object_types: list[str] | None = Field(None, description="Filter by object types")
    count_threshold: int = Field(5, description="Minimum detections per object", gt=0)
    time_period: dict[str, Any] | None = Field(None, description="Time period constraints")
    catalogs: list[str] | None = Field(None, description="Specific catalogs to query")
    filters: dict[str, Any] | None = Field(None, description="Additional filters")
    export_format: Literal["parquet", "csv", "fits", "json"] = Field("parquet", description="Output format")
    output_path: str | None = Field(None, description="Export path for the dataset")
    targets: list[TargetRequest] = Field(..., min_length=1, description="Sky positions to search")
    min_confidence: float = Field(0.0, ge=0, le=1)
    max_results: int | None = Field(None, gt=0)


class SavedQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    query: SearchRequest


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint (no authentication, no quota)."""
    return HealthResponse(status="healthy", timestamp=datetime.now(UTC).isoformat())


@app.get("/api/v1/catalogs", response_model=dict[str, Any])
async def list_catalogs():
    """All configured catalogs (embedded + registered VizieR tables)."""
    logger.info("list_catalogs", endpoint="/api/v1/catalogs")
    service = get_service()
    return {name: asdict(catalog) for name, catalog in service.registry.catalogs.items()}


@app.get("/api/v1/catalogs/{catalog_name}", response_model=dict[str, Any])
async def get_catalog(catalog_name: str):
    """Retrieve detailed specification for a given catalog."""
    logger.info("get_catalog", endpoint="/api/v1/catalogs", catalog=catalog_name)
    service = get_service()
    try:
        catalog = service.registry.get(catalog_name)
        return asdict(catalog)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Catalog {catalog_name} not found")


async def _search(req: SearchRequest) -> dict[str, Any]:
    if not req.name and (req.ra is None or req.dec is None):
        raise ValueError("Either name or both ra and dec are required")

    cache_key = cache.make_key("search", req.model_dump())
    cached = await cache.aget(cache_key)
    if cached is not None:
        return cached

    service = get_service()
    client = getattr(app.state, "client", None)
    settings = getattr(app.state, "settings", None) or Settings()
    resolver = SesameResolver(client, endpoint=settings.resolver_endpoint) if req.name else None
    # main.api_search: shared with provenance's manifests of API searches (same query, same record).
    result = await api_search(service, req.model_dump(), resolver)
    if not getattr(service, "records_catalog_metrics", False):
        record_catalog_metrics(result.provenance.get("catalog_stats") or {})

    data = result.as_dict()
    # Never replay a transient archive failure to later identical searches: only
    # records in which every catalog answered are cached.
    if not result.failures:
        await cache.aset(cache_key, data, ttl=search_cache_ttl())
    return data


def search_cache_ttl() -> int:
    """TTL (s) of the API-level search cache; API_SEARCH_CACHE_TTL_SECONDS, default 300 (0 disables)."""
    try:
        return max(0, int(os.getenv("API_SEARCH_CACHE_TTL_SECONDS", "300")))
    except ValueError:
        return 300


# Exceptions of a search that mean an archive or service failed (502), as opposed to a bug
# in this server (500): a search run's per-catalog failures are recorded in the record, so
# only failures outside the catalogs (the service's own client, the cache) reach here.
UPSTREAM_ERRORS = (CatalogUnavailableError, QueryTimeoutError, CatalogQueryError, ResponseParseError, httpx.HTTPError,
                   TimeoutError)


def search_error(exc: Exception) -> tuple[int, str, dict[str, str] | None]:
    """(status, detail, headers) of a failed search, the same for /search and each /search/batch item:
    404 unknown name, 503 (Retry-After) resolver down, 502 resolver garbage or upstream archive
    failure, 422 invalid input (InvalidCoordinateError, ValueError), 500 anything else."""
    if isinstance(exc, ObjectResolutionError):
        code = resolution_failure_status(exc)
        if code == 503:
            return 503, f"Name resolver unavailable: {exc}", {"Retry-After": "30"}
        if code == 404:
            return 404, f"Object name could not be resolved: {exc}", None
        if code == 422:
            return 422, str(exc), None
        return 502, f"Name resolver failed: {exc}", None
    if isinstance(exc, InvalidCoordinateError | ValueError):
        return 422, str(exc), None
    if isinstance(exc, UPSTREAM_ERRORS):
        return 502, f"Catalog search failed: {type(exc).__name__}: {exc}", None
    return 500, "Internal error during search", None


@app.post("/api/v1/search", response_model=dict[str, Any])
async def search_endpoint(req: SearchRequest):
    """Execute single crossmatch search by coordinates or resolved object name."""
    logger.info("search_request", endpoint="/api/v1/search", ra=req.ra, dec=req.dec, name=req.name)
    try:
        return await _search(req)
    except HTTPException:
        raise
    except Exception as e:
        code, detail, headers = search_error(e)
        (logger.error if code >= 500 else logger.info)("search_error", error=str(e), error_type=type(e).__name__,
                                                       status=code, endpoint="/api/v1/search")
        raise HTTPException(status_code=code, detail=detail, headers=headers) from e


@app.post("/api/v1/search/batch", response_model=list[dict[str, Any]])
async def batch_search(requests: list[SearchRequest], max_concurrent: int = Query(10, ge=1, le=100)):
    """Execute multiple search queries concurrently with error isolation (each failed item
    carries the status /api/v1/search would answer)."""
    if len(requests) > int(os.getenv("API_MAX_BATCH_SIZE", "100")):
        raise HTTPException(status_code=413, detail="Batch contains too many searches")
    logger.info("batch_search_request", count=len(requests), endpoint="/api/v1/search/batch")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def one(req: SearchRequest):
        try:
            async with semaphore:
                return await _search(req)
        except Exception as e:
            code, detail, _ = search_error(e)
            if code >= 500:
                logger.error("batch_search_item_error", error=str(e), error_type=type(e).__name__, status=code)
            return {"error": detail if code != 422 else str(e), "status_code": code, "request": req.model_dump()}

    return await asyncio.gather(*(one(req) for req in requests))


@app.post("/api/v1/datasets/create", response_model=dict[str, Any], status_code=status.HTTP_202_ACCEPTED)
async def create_dataset_endpoint(req: DatasetRequest, background_tasks: BackgroundTasks):
    """Submit an asynchronous dataset generation job (a Redis RQ worker when REDIS_URL is set,
    else a background task of this process). Lists of at least DATASET_BATCH_MIN_TARGETS
    targets are fetched with the batch engine; exports carry the match probabilities."""
    if len(req.targets) > int(os.getenv("API_MAX_TARGETS", "10000")):
        raise HTTPException(status_code=413, detail="Dataset contains too many targets")

    engine = get_engine()
    try:
        query = AdvancedQuery.from_dict({
            "ra": req.targets[0].ra,
            "dec": req.targets[0].dec,
            "radius_arcsec": req.radius_arcsec,
            "profiles": [req.profile],
            "object_types": req.object_types,
            "count_threshold": req.count_threshold,
            "min_confidence": req.min_confidence,
            "max_results": req.max_results,
            "catalogs": req.catalogs,
            "time_period": req.time_period,
        })
        QueryValidator.validate(query, engine.registry)
        if req.output_path:
            path = Path(req.output_path).resolve()
            if (
                not path.is_relative_to(engine.storage)
                or path.exists()
                or path.suffix.lower() != f".{req.export_format}"
            ):
                raise ValueError("output_path must be unused, inside DATASET_STORAGE_PATH, and match export_format")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("create_dataset_request", name=req.name, profile=req.profile)
    try:
        payload = req.model_dump(exclude={"output_path", "export_format"})
        payload["output_format"] = req.export_format
        payload["export_path"] = req.output_path
        metadata = enqueue_dataset(payload, engine)
        try:
            # redis-py and RQ are blocking clients: enqueue from a worker thread.
            if not await asyncio.to_thread(submit_to_redis, metadata["id"], payload):
                background_tasks.add_task(process_dataset_async, metadata["id"], payload, engine)
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = "Job queue unavailable"
            engine.metadata.put_dataset(metadata)
            raise HTTPException(status_code=503, detail="Job queue unavailable") from exc
        return JSONResponse(metadata, status_code=202, headers={"Location": f"/api/v1/datasets/{metadata['id']}"})
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dataset_create_error", error=str(e), name=req.name)
        raise HTTPException(status_code=500, detail="Dataset creation failed") from e


@app.get("/api/v1/datasets", response_model=list[dict[str, Any]])
async def list_datasets_endpoint():
    """List all created datasets."""
    logger.info("list_datasets", endpoint="/api/v1/datasets")
    return get_engine().list_datasets()


@app.get("/api/v1/datasets/{dataset_name}", response_model=dict[str, Any])
async def get_dataset_endpoint(dataset_name: str):
    """Retrieve metadata and generation status of a dataset."""
    logger.info("get_dataset", dataset=dataset_name, endpoint="/api/v1/datasets")
    dataset = get_engine().get_dataset(dataset_name)
    if dataset:
        return dataset
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")


EXPORT_MEDIA_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "parquet": "application/vnd.apache.parquet",
    "fits": "application/fits",
}


def export_media_type(output_format: str | None) -> str:
    """Content type of a dataset export by its format (application/octet-stream if unknown)."""
    return EXPORT_MEDIA_TYPES.get(str(output_format or "").lower(), "application/octet-stream")


@app.get("/api/v1/datasets/{dataset_name}/export")
async def export_dataset_endpoint(dataset_name: str):
    """Download exported dataset file or stream from object storage."""
    engine = get_engine()
    dataset = engine.get_dataset(dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset["status"] != "completed":
        raise HTTPException(status_code=409, detail="Dataset export is not ready")

    if dataset.get("export_uri"):
        body = engine.objects.get(dataset["export_uri"])

        def chunks():
            try:
                while chunk := body.read(1024 * 1024):
                    yield chunk
            finally:
                body.close()

        return StreamingResponse(
            chunks(),
            media_type=export_media_type(dataset["output_format"]),
            headers={"Content-Disposition": f'attachment; filename="{dataset_name}.{dataset["output_format"]}"'},
        )
    # An explicit type: FileResponse would otherwise guess it from the OS (the Windows registry
    # maps .csv to application/vnd.ms-excel).
    return FileResponse(dataset["export_path"], filename=f"{dataset_name}.{dataset['output_format']}",
                        media_type=export_media_type(dataset["output_format"]))


@app.delete("/api/v1/datasets/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset_endpoint(dataset_name: str):
    """Delete a dataset and its exported artifact."""
    logger.info("delete_dataset", dataset=dataset_name, endpoint="/api/v1/datasets")
    try:
        if get_engine().delete_dataset(dataset_name):
            return
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")


@app.get("/api/v1/queries", response_model=list[dict[str, Any]])
async def list_queries_endpoint():
    """List all saved search queries."""
    return get_metadata().list_queries()


@app.post("/api/v1/queries", response_model=dict[str, Any], status_code=201)
async def save_query_endpoint(req: SavedQueryRequest):
    """Save a search query for reuse."""
    if not req.query.name and (req.query.ra is None or req.query.dec is None):
        raise HTTPException(status_code=422, detail="Query requires name or ra and dec")
    return get_metadata().save_query(req.name, req.query.model_dump())


@app.delete("/api/v1/queries/{query_id}", status_code=204)
async def delete_query_endpoint(query_id: str):
    """Delete a saved search query."""
    if not get_metadata().delete_query(query_id):
        raise HTTPException(status_code=404, detail="Saved query not found")


@app.get("/api/v1/stats", response_model=dict[str, Any])
async def stats_endpoint():
    """Retrieve service usage statistics."""
    datasets = get_engine().list_datasets()
    return {
        "datasets": len(datasets),
        "sources_exported": sum(item.get("total_sources", 0) for item in datasets),
        "saved_queries": len(get_metadata().list_queries()),
    }


@app.get("/api/v1/monitoring", response_model=dict[str, Any])
async def monitoring_endpoint():
    """Provider circuit-breaker states, registry size and sky-cache catalogs."""
    service = getattr(app.state, "service", None)
    store = getattr(app.state, "skycache", None)
    return {
        "status": "healthy",
        "provider_circuits": {
            endpoint: guard.state
            for provider in (service.providers.values() if service is not None else ())
            for endpoint, guard in (getattr(provider, "guards", None) or {}).items()
        },
        "catalogs": len(service.registry.catalogs) if service is not None else None,
        "skycache": {"enabled": isinstance(store, skycache.SkyCache),
                     "catalogs": await asyncio.to_thread(store.catalogs) if store is not None else []},
    }


@app.get("/api/v1/monitoring/metrics")
async def metrics_endpoint():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Feature routers and the web UI
# ---------------------------------------------------------------------------

FEATURE_ROUTERS = (
    streaming.router,   # GET  /api/v1/search/stream (server-sent events)
    batch.router,       # /api/v1/batch/{crossmatch,strategies}
    skycache.router,    # /api/v1/skycache/{status,mirror,cone,{catalog}}
    vizier.router,      # /api/v1/vizier/{search,catalog/{id},register,registered}
    sed.router,         # /api/v1/sed
    timedomain.router,  # /api/v1/lightcurves, /api/v1/solar-system
    imaging.router,     # /api/v1/cutouts, /api/v1/cutouts/{surveys,stack}
    ai.router,          # /api/v1/ai/{query,explain}
    provenance.router,  # /api/v1/provenance/{manifest,replay}, /api/v1/citations[/sources]
    vo_server.router,   # /vo/scs, /vo/tap/...
    alerts.router,      # /api/v1/alerts[...]
)
for _router in FEATURE_ROUTERS:
    app.include_router(_router)

def _installed_web_dirs() -> list[Path]:
    """Where a wheel's data files (share/astrosearch/web) can land: the environment prefix, the
    data path of every install scheme (a --user or distro prefix install differs from
    sys.prefix), the user base, and the files the installed distribution records itself."""
    import importlib.metadata
    import site
    import sysconfig

    roots: list[Path] = [Path(sys.prefix), Path(sys.base_prefix)]
    for scheme in {sysconfig.get_default_scheme(), *(s for s in sysconfig.get_scheme_names() if "user" in s)}:
        try:
            roots.append(Path(sysconfig.get_path("data", scheme)))
        except (KeyError, TypeError):
            continue
    if getattr(site, "USER_BASE", None):
        roots.append(Path(site.USER_BASE))
    found = [root / "share" / "astrosearch" / "web" for root in roots]
    try:
        for entry in importlib.metadata.distribution("astrosearch").files or ():
            if entry.name == "index.html" and "web" in entry.parts:
                found.append(Path(str(entry.locate())).resolve().parent)
    except importlib.metadata.PackageNotFoundError:
        pass
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def web_ui_dir() -> Path | None:
    """The web UI directory: ASTROSEARCH_WEB_DIR, else web/ next to imaging.py (a checkout or an
    editable install), else share/astrosearch/web of the installation (a wheel install into a
    venv, --user, --prefix or a distro scheme; see :func:`_installed_web_dirs`); None when absent."""
    configured = os.getenv("ASTROSEARCH_WEB_DIR")
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        candidates = [imaging.WEB_DIR, *_installed_web_dirs()]
    return next((path for path in candidates if (path / "index.html").is_file()), None)


# The UI files get exact GET/HEAD routes and are served ahead of the authentication/quota
# middleware (UI_PUBLIC=false keeps them behind it). Must run after the middleware above.
_WEB_DIR = web_ui_dir()
if _WEB_DIR is not None:
    imaging.mount_ui(app, _WEB_DIR,
                     public=os.getenv("UI_PUBLIC", "true").strip().lower() not in {"0", "false", "no", "off"})
else:
    logger.warning("web_ui_not_found", searched=str(imaging.WEB_DIR),
                   hint="set ASTROSEARCH_WEB_DIR to the directory holding index.html")


def route_table(application: FastAPI | None = None) -> list[tuple[str, frozenset[str]]]:
    """(path, methods) of every HTTP route of the app, the routes of included routers
    flattened (FastAPI keeps an included router as one entry of ``app.routes``)."""
    table: list[tuple[str, frozenset[str]]] = []
    for route in (application or app).routes:
        contexts = getattr(route, "effective_route_contexts", None)
        if callable(contexts):
            table.extend((ctx.path, frozenset(ctx.methods or ())) for ctx in contexts())
        elif hasattr(route, "path"):
            table.append((route.path, frozenset(getattr(route, "methods", None) or ())))
    return table


def create_app() -> FastAPI:
    """Factory returning the configured FastAPI application instance."""
    return app
