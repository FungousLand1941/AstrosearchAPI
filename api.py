"""Complete production FastAPI REST API for AstroSearch."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from crossmatch import AdvancedQuery, CrossmatchService, QueryValidator
from datasets import DatasetEngine, MetadataStore, enqueue_dataset, process_dataset_async, submit_to_redis
from models import CatalogRegistry, Settings
from providers import CacheManager, SesameResolver, provider_map
from astronomy import ArchiveError, object_summary, summarize_system

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

REQUEST_COUNT = Counter("astrosearch_requests_total", "Total API requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("astrosearch_request_latency_seconds", "API request latency", ["method", "endpoint"])
ACTIVE_REQUESTS = Gauge("astrosearch_active_requests", "Active in-flight API requests", ["endpoint"])
CATALOG_QUERIES = Counter("astrosearch_catalog_queries_total", "Catalog query outcomes", ["catalog", "status"])
CATALOG_LATENCY = Histogram("astrosearch_catalog_query_seconds", "Catalog query latency", ["catalog"])


def setup_metrics(application: FastAPI) -> None:
    application.state.metrics = {
        "request_count": REQUEST_COUNT,
        "request_latency": REQUEST_LATENCY,
        "active_requests": ACTIVE_REQUESTS,
    }


# ---------------------------------------------------------------------------
# Authentication & Request Quotas
# ---------------------------------------------------------------------------


class RequestQuota:
    """Sliding-window request quota tracker with Redis backend and in-memory fallback."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._redis = None
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url)
            except Exception:
                self._redis = None

    async def allow(self, identity: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.time()
        bucket = int(now // 60)
        if self._redis is not None:
            key = f"astrosearch:quota:{identity}:{bucket}"
            try:
                count = await self._redis.incr(key)
                if count == 1:
                    await self._redis.expire(key, 120)
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
        if self._redis is not None:
            try:
                await self._redis.aclose()
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


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = Settings()
    registry = CatalogRegistry(settings.catalog_registry_path)
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
        application.state.client = client
        application.state.registry = registry
        application.state.providers = provider_map(client, timeout=settings.request_timeout_seconds)
        application.state.service = CrossmatchService(
            registry,
            application.state.providers,
            radius_arcsec=settings.default_radius_arcsec,
            timeout=settings.request_timeout_seconds,
        )
        application.state.engine = DatasetEngine(
            registry_path=settings.catalog_registry_path,
            service=application.state.service,
        )
        application.state.metadata = application.state.engine.metadata
        yield
        del application.state.service
        del application.state.client
        del application.state.engine
        del application.state.metadata
        await application.state.quota.close()


app = FastAPI(
    title="AstroSearch API",
    description="Astronomical catalog cross-matching and dataset generation service",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials="*" not in (cors_origins or ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)
cache = CacheManager(os.getenv("REDIS_URL"))
app.state.quota = RequestQuota(os.getenv("REDIS_URL"))


def get_service() -> CrossmatchService:
    service = getattr(app.state, "service", None)
    if service is None:
        settings = Settings()
        registry = CatalogRegistry(settings.catalog_registry_path)
        service = CrossmatchService(registry, provider_map(timeout=settings.request_timeout_seconds))
    return service


def get_engine() -> DatasetEngine:
    return getattr(app.state, "engine", None) or DatasetEngine()


def get_metadata() -> MetadataStore:
    return getattr(app.state, "metadata", None) or MetadataStore(
        local_dir=os.getenv("DATASET_STORAGE_PATH") or "datasets"
    )


# ---------------------------------------------------------------------------
# Middleware: Request Metrics, Quotas, and Body Limits
# ---------------------------------------------------------------------------


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
    try:
        if endpoint != "/api/v1/health":
            identity = authenticate(request)
            if isinstance(identity, JSONResponse):
                response = identity
            elif not await app.state.quota.allow(identity, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))):
                response = JSONResponse(
                    {"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"}
                )
            else:
                structlog.contextvars.bind_contextvars(actor=identity)
                maximum = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
                length = request.headers.get("content-length")
                if (length and int(length) > maximum) or len(await request.body()) > maximum:
                    response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                else:
                    response = await call_next(request)
        else:
            response = await call_next(request)

        route = request.scope.get("route")
        endpoint_label = getattr(route, "path", endpoint if endpoint == "/api/v1/health" else "unmatched")
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


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "0.2.0"


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ra: float = Field(..., description="Right ascension in degrees", ge=0, lt=360)
    dec: float = Field(..., description="Declination in degrees", ge=-90, le=90)
    epoch: float | None = Field(None, ge=1800, le=2200, description="Julian epoch for proper motion correction")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ra: float | None = Field(None, description="Right ascension in degrees")
    dec: float | None = Field(None, description="Declination in degrees")
    name: str | None = Field(None, description="Astronomical object name (will be resolved)")
    radius_arcsec: float = Field(3.0, description="Search radius in arcseconds", gt=0)
    profile: str | None = Field(None, description="Catalog profile: optical, infrared, radio, etc.")
    epoch: float | None = Field(None, ge=1800, le=2200, description="Julian epoch for proper motion correction")
    object_types: list[str] | None = Field(None, description="Filter by object types")
    spectral_types: list[str] | None = Field(None, description="Filter by spectral classification")
    morphology: list[str] | None = None
    max_results: int | None = Field(None, gt=0, description="Maximum results per catalog")
    min_confidence: float = Field(0.5, description="Minimum match confidence", ge=0, le=1)
    catalogs: list[str] | None = None
    time_period: dict[str, Any] | None = None
    spatial_constraints: dict[str, Any] | None = None
    search_mode: Literal["cone", "shell", "cylinder"] = "cone"
    min_radius_arcsec: float = Field(0.0, ge=0)
    proper_motion: bool = True
    adaptive_radius: bool = False
    min_distance_pc: float | None = Field(None, gt=0)
    max_distance_pc: float | None = Field(None, gt=0)


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


class SystemSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=300)


@app.post("/api/v1/summaries/system")
async def system_summary_endpoint(req: SystemSummaryRequest):
    try:
        return await summarize_system(app.state.client, req.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ArchiveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/v1/summaries/object")
async def object_summary_endpoint(req: SearchRequest):
    try:
        return object_summary(await _search(req))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint."""
    logger.info("health_check", endpoint="/api/v1/health")
    return HealthResponse(status="healthy", timestamp=datetime.now(UTC).isoformat())


@app.get("/api/v1/catalogs", response_model=dict[str, Any])
async def list_catalogs():
    """List all available astronomical catalogs."""
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
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    resolved_info = None
    service = get_service()
    client = getattr(app.state, "client", None)

    if req.name:
        resolver = SesameResolver(client)
        resolved = await resolver.resolve(req.name)
        resolved_info = resolved.as_dict()
        ra, dec = resolved.ra_deg, resolved.dec_deg
    else:
        ra, dec = req.ra, req.dec

    query = AdvancedQuery.from_dict({
        "ra": ra,
        "dec": dec,
        "epoch": req.epoch,
        "radius_arcsec": req.radius_arcsec,
        "profiles": [req.profile] if req.profile else None,
        "object_types": req.object_types,
        "spectral_types": req.spectral_types,
        "morphology": req.morphology,
        "min_confidence": req.min_confidence,
        "max_results": req.max_results,
        "catalogs": req.catalogs,
        "time_period": req.time_period,
        "spatial_constraints": req.spatial_constraints,
        "search_mode": req.search_mode,
        "min_radius_arcsec": req.min_radius_arcsec,
        "proper_motion": req.proper_motion,
        "adaptive_radius": req.adaptive_radius,
        "min_distance_pc": req.min_distance_pc,
        "max_distance_pc": req.max_distance_pc,
    })

    result = await service.crossmatch(ra, dec, query=query)
    if resolved_info:
        result.resolved_object = resolved_info

    data = result.as_dict()
    cache.set(cache_key, data, ttl=300)
    return data


@app.post("/api/v1/search", response_model=dict[str, Any])
async def search_endpoint(req: SearchRequest):
    """Execute single crossmatch search by coordinates or resolved object name."""
    logger.info("search_request", endpoint="/api/v1/search", ra=req.ra, dec=req.dec, name=req.name)
    try:
        return await _search(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("search_error", error=str(e), endpoint="/api/v1/search")
        raise HTTPException(status_code=502, detail="Catalog search failed") from e


@app.post("/api/v1/search/batch", response_model=list[dict[str, Any]])
async def batch_search(requests: list[SearchRequest], max_concurrent: int = Query(10, ge=1, le=100)):
    """Execute multiple search queries concurrently with error isolation."""
    if len(requests) > int(os.getenv("API_MAX_BATCH_SIZE", "100")):
        raise HTTPException(status_code=413, detail="Batch contains too many searches")
    logger.info("batch_search_request", count=len(requests), endpoint="/api/v1/search/batch")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def one(req: SearchRequest):
        try:
            async with semaphore:
                return await _search(req)
        except Exception as e:
            return {"error": str(e), "request": req.model_dump()}

    return await asyncio.gather(*(one(req) for req in requests))


@app.post("/api/v1/datasets/create", response_model=dict[str, Any], status_code=status.HTTP_202_ACCEPTED)
async def create_dataset_endpoint(req: DatasetRequest, background_tasks: BackgroundTasks):
    """Submit asynchronous dataset generation job."""
    if len(req.targets) > int(os.getenv("API_MAX_TARGETS", "1000")):
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
            if not submit_to_redis(metadata["id"], payload):
                background_tasks.add_task(process_dataset_async, metadata["id"], payload)
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
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{dataset_name}.{dataset["output_format"]}"'},
        )
    return FileResponse(dataset["export_path"], filename=f"{dataset_name}.{dataset['output_format']}")


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
    """Inspect provider circuit breaker states and service health."""
    service = getattr(app.state, "service", None)
    return {
        "status": "healthy",
        "provider_circuits": {
            endpoint: guard.state
            for provider in (service.providers.values() if service is not None else ())
            for endpoint, guard in getattr(provider, "guards", {}).items()
        },
    }


@app.get("/api/v1/monitoring/metrics")
async def metrics_endpoint():
    """Prometheus scrape endpoint."""
    logger.info("metrics_request", endpoint="/api/v1/monitoring/metrics")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app() -> FastAPI:
    """Factory creating the configured FastAPI application instance."""
    return app
