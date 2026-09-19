"""Streaming multi-format dataset generation, storage backends (SQLite/PostgreSQL/S3), and worker jobs."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, TextIO

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from crossmatch import AdvancedQuery, CrossmatchService, QueryValidator
from models import CatalogRegistry, validate_target

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Multi-Format Streaming Exporter
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = ("catalog", "source_id", "physical", "data", "metadata", "provenance", "links")
_FLOAT_COLUMNS = ("ra", "dec", "separation_arcsec", "confidence", "epoch", "positional_error_arcsec")
_COLUMNS = _TEXT_COLUMNS + _FLOAT_COLUMNS


def _flat(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(source.get(key), default=str)
        if key in {"physical", "data", "metadata", "provenance", "links"}
        else str(source[key])
        if key in {"catalog", "source_id"} and key in source
        else float(source[key])
        if source.get(key) is not None and key in _FLOAT_COLUMNS
        else None
        for key in _COLUMNS
    }


class DatasetWriter:
    """Context manager for streaming multi-format dataset exports (JSON, CSV, Parquet, FITS)."""

    def __init__(self, path: Path, output_format: str) -> None:
        self.path = path
        self.output_format = output_format.lower()
        self.count = 0
        self._buffer: list[dict[str, Any]] = []
        self._handle: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._parquet_writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_format == "json":
            self._handle = self.path.open("w", encoding="utf-8")
            self._handle.write("[")
        elif self.output_format == "csv":
            self._handle = self.path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._handle, fieldnames=_COLUMNS)
            self._csv_writer.writeheader()
        elif self.output_format == "parquet":
            self._schema = pa.schema(
                [(name, pa.string()) for name in _TEXT_COLUMNS]
                + [(name, pa.float64()) for name in _FLOAT_COLUMNS]
            )
            self._parquet_writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")
        elif self.output_format != "fits":
            raise ValueError(f"Unsupported output format: {self.output_format}")
        return self

    def write(self, source: dict[str, Any]) -> None:
        """Write a single catalog detection to the dataset stream."""
        if self.output_format == "json":
            assert self._handle is not None
            if self.count:
                self._handle.write(",\n")
            json.dump(source, self._handle, default=str)
        elif self.output_format == "csv":
            assert self._csv_writer is not None
            self._csv_writer.writerow(_flat(source))
        else:
            self._buffer.append(_flat(source))
            if self.output_format == "parquet" and len(self._buffer) >= 1000:
                self._flush_parquet()
        self.count += 1

    def _flush_parquet(self) -> None:
        if self._buffer:
            assert self._parquet_writer is not None and self._schema is not None
            self._parquet_writer.write_table(pa.Table.from_pylist(self._buffer, schema=self._schema))
            self._buffer.clear()

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self.output_format == "json":
                assert self._handle is not None
                self._handle.write("]")
            elif self.output_format == "parquet" and exc_type is None:
                self._flush_parquet()
            elif self.output_format == "fits" and exc_type is None:
                from astropy.table import Table

                fits_rows = [
                    {
                        key: val if val is not None else float("nan") if key in _FLOAT_COLUMNS else ""
                        for key, val in row.items()
                    }
                    for row in self._buffer
                ]
                table = Table(rows=fits_rows) if fits_rows else Table(
                    names=_COLUMNS,
                    dtype=["U1"] * len(_TEXT_COLUMNS) + ["f8"] * len(_FLOAT_COLUMNS),
                )
                table.write(self.path, format="fits", overwrite=True)
        finally:
            if self._handle:
                self._handle.close()
            if self._parquet_writer:
                self._parquet_writer.close()
            if exc_type is not None:
                self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Storage Layer: MetadataStore and ObjectStore
# ---------------------------------------------------------------------------


class MetadataStore:
    """Durable metadata and query history store supporting SQLite and PostgreSQL."""

    def __init__(self, database_url: str | None = None, *, local_dir: str | Path = "datasets") -> None:
        url = database_url or os.getenv("DATABASE_URL")
        self.postgres = bool(url and url.startswith(("postgresql://", "postgres://")))
        if url and not self.postgres and not url.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must use postgresql:// or sqlite:///")
        self.database_url = url
        self.sqlite_path = (
            Path(url.removeprefix("sqlite:///"))
            if url and not self.postgres
            else Path(local_dir) / "metadata.sqlite3"
        )
        if not self.postgres:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        if self.postgres:
            import psycopg
            return psycopg.connect(self.database_url)
        return sqlite3.connect(self.sqlite_path, timeout=30)

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.postgres else statement

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                self._sql(
                    "CREATE TABLE IF NOT EXISTS datasets ("
                    "id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
            )
            conn.execute(
                self._sql(
                    "CREATE TABLE IF NOT EXISTS saved_queries ("
                    "id TEXT PRIMARY KEY, name TEXT NOT NULL, query_json TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
            )

    def put_dataset(self, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                self._sql(
                    "INSERT INTO datasets (id, metadata_json, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT (id) DO UPDATE SET metadata_json = excluded.metadata_json"
                ),
                (metadata["id"], json.dumps(metadata, default=str), metadata["created_at"]),
            )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(self._sql("SELECT metadata_json FROM datasets WHERE id = ?"), (dataset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT metadata_json FROM datasets ORDER BY created_at DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(self._sql("DELETE FROM datasets WHERE id = ?"), (dataset_id,))
            return cur.rowcount > 0

    def save_query(self, name: str, query: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex,
            "name": name,
            "query": query,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._connect() as conn:
            conn.execute(
                self._sql("INSERT INTO saved_queries (id, name, query_json, created_at) VALUES (?, ?, ?, ?)"),
                (item["id"], name, json.dumps(query), item["created_at"]),
            )
        return item

    def list_queries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, name, query_json, created_at FROM saved_queries ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "name": r[1], "query": json.loads(r[2]), "created_at": r[3]} for r in rows]

    def delete_query(self, query_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(self._sql("DELETE FROM saved_queries WHERE id = ?"), (query_id,))
            return cur.rowcount > 0


class ObjectStore:
    """Optional S3 / MinIO cloud storage adapter for dataset files."""

    def __init__(self, bucket: str | None = None) -> None:
        self.bucket = bucket or os.getenv("S3_BUCKET")
        self._client = None
        if self.bucket:
            try:
                import boto3
                self._client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL") or None)
            except Exception:
                self._client = None

    def put(self, dataset_id: str, path: Path) -> str | None:
        if not self._client or not self.bucket:
            return None
        key = f"datasets/{dataset_id}/{path.name}"
        self._client.upload_file(str(path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def get(self, uri: str):
        if not self._client or not self.bucket or not uri.startswith(f"s3://{self.bucket}/"):
            raise ValueError("Invalid dataset object URI")
        key = uri.removeprefix(f"s3://{self.bucket}/")
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"]

    def delete(self, uri: str) -> None:
        if self._client and self.bucket and uri.startswith(f"s3://{self.bucket}/"):
            self._client.delete_object(Bucket=self.bucket, Key=uri.removeprefix(f"s3://{self.bucket}/"))


# ---------------------------------------------------------------------------
# Dataset Generation Engine
# ---------------------------------------------------------------------------


class DatasetEngine:
    """Processes search targets into filtered, deduplicated datasets with durable metadata."""

    def __init__(
        self,
        registry_path: str | None = None,
        storage_path: str | None = None,
        service: CrossmatchService | None = None,
    ) -> None:
        self.registry = CatalogRegistry(registry_path)
        self.storage = Path(storage_path or os.getenv("DATASET_STORAGE_PATH") or "datasets").resolve()
        self.storage.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.metadata = MetadataStore(local_dir=self.storage)
        self.objects = ObjectStore()

    async def create_dataset(
        self,
        name: str,
        profile: str,
        radius_arcsec: float,
        object_types: list[str] | None = None,
        count_threshold: int = 1,
        time_period: dict[str, Any] | None = None,
        catalogs: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        output_format: str = "parquet",
        export_path: str | None = None,
        targets: list[dict[str, Any]] | None = None,
        min_confidence: float = 0.0,
        max_results: int | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute crossmatches across targets and stream filtered detections into an export file."""
        if not name.strip() or not profile.strip():
            raise ValueError("name and profile are required")
        if output_format.lower() not in {"json", "csv", "parquet", "fits"}:
            raise ValueError("Unsupported output format")
        if not targets:
            raise ValueError("At least one target with ra and dec is required")

        unknown = set(catalogs or ()) - set(self.registry.enabled_catalogs())
        if unknown:
            raise ValueError(f"Unknown catalogs: {', '.join(sorted(unknown))}")

        query = AdvancedQuery.from_dict({
            "ra": targets[0]["ra"],
            "dec": targets[0]["dec"],
            "radius_arcsec": radius_arcsec,
            "object_types": object_types,
            "count_threshold": count_threshold,
            "min_confidence": min_confidence,
            "time_period": time_period,
            "catalogs": catalogs,
            "filters": filters,
            "max_results": max_results,
            "profiles": [profile],
        })
        QueryValidator.validate(query, self.registry)

        for target in targets:
            validate_target(target["ra"], target["dec"], epoch=target.get("epoch"))

        dataset_id = dataset_id or uuid.uuid4().hex
        path = Path(export_path).resolve() if export_path else self.storage / f"{dataset_id}.{output_format.lower()}"
        if not path.is_relative_to(self.storage):
            raise ValueError("output_path must be within DATASET_STORAGE_PATH")
        if path.exists() or path.suffix.lower() != f".{output_format.lower()}":
            raise ValueError("output_path must be unused and match output_format")

        seen: set[tuple[str, str]] = set()
        catalogs_used: set[str] = set()
        failures: list[dict[str, Any]] = []

        async def collect(active_service: CrossmatchService, writer: DatasetWriter) -> None:
            async for result in self._search_targets(active_service, targets, query):
                failures.extend(result.get("failures", []))
                for group in result["crossmatch_groups"]:
                    members = [
                        source for source in group["members"]
                        if source["confidence"] >= min_confidence
                        and query.apply_filters(source)
                        and (not catalogs or source["catalog"] in catalogs)
                        and all(self._passes_filter(source, k, v) for k, v in (filters or {}).items())
                    ]
                    if len(members) < count_threshold:
                        continue
                    for source in members:
                        key = (source["catalog"], source["source_id"])
                        if key in seen:
                            continue
                        seen.add(key)
                        catalogs_used.add(source["catalog"])
                        writer.write(source)

        with DatasetWriter(path, output_format) as writer:
            if self.service is None:
                from providers import provider_map
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    svc = CrossmatchService(self.registry, provider_map(client))
                    await collect(svc, writer)
            else:
                await collect(self.service, writer)

        export_uri = self.objects.put(dataset_id, path)
        if export_uri:
            path.unlink()

        previous = self.metadata.get_dataset(dataset_id)
        metadata = {
            "id": dataset_id,
            "name": name,
            "profile": profile,
            "status": "completed",
            "total_sources": writer.count,
            "catalogs_used": sorted(catalogs_used),
            "failures": failures,
            "output_format": output_format.lower(),
            "export_path": str(path),
            "created_at": previous["created_at"] if previous else datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "export_uri": export_uri,
        }
        self.metadata.put_dataset(metadata)
        return metadata

    @staticmethod
    async def _search_targets(service: CrossmatchService, targets: list[dict[str, Any]], base_query: AdvancedQuery):
        semaphore = asyncio.Semaphore(10)

        async def search(t: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                q_dict = base_query.to_dict()
                q_dict["target"] = t
                q = AdvancedQuery.from_dict(q_dict)
                result = await service.crossmatch(t["ra"], t["dec"], query=q)
                return result.as_dict()

        for offset in range(0, len(targets), 10):
            batch = targets[offset:offset + 10]
            for res in await asyncio.gather(*(search(t) for t in batch)):
                yield res

    @staticmethod
    def _passes_filter(source: dict[str, Any], name: str, limit: Any) -> bool:
        field = name.removeprefix("min_").removeprefix("max_")
        val = source.get(field)
        if val is None:
            val = source.get("physical", {}).get(field, source.get("data", {}).get(field))
        if val is None and field == "proper_motion_masyr":
            ra = source.get("proper_motion_ra_masyr")
            dec = source.get("proper_motion_dec_masyr")
            if ra is not None and dec is not None:
                try:
                    val = math.hypot(float(ra), float(dec))
                except (TypeError, ValueError):
                    return False
        if val is None:
            return False
        try:
            if name.startswith("min_"):
                return float(val) >= float(limit)
            if name.startswith("max_"):
                return float(val) <= float(limit)
        except (TypeError, ValueError):
            return False
        return str(val) == str(limit)

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.metadata.list_datasets()

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        if not dataset_id.isalnum():
            return None
        return self.metadata.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> bool:
        metadata = self.get_dataset(dataset_id)
        if metadata is None:
            return False
        if metadata.get("status") in {"queued", "running"}:
            raise ValueError("A running dataset cannot be deleted")
        if metadata.get("export_uri"):
            self.objects.delete(metadata["export_uri"])
        elif metadata.get("export_path"):
            p = Path(metadata["export_path"]).resolve()
            if p.is_relative_to(self.storage):
                p.unlink(missing_ok=True)
        return self.metadata.delete_dataset(dataset_id)


# ---------------------------------------------------------------------------
# Asynchronous Jobs and Queue Submission
# ---------------------------------------------------------------------------


def enqueue_dataset(payload: dict[str, Any], engine: DatasetEngine | None = None) -> dict[str, Any]:
    """Register a new dataset in the metadata store with status 'queued'."""
    eng = engine or DatasetEngine()
    dataset_id = uuid.uuid4().hex
    metadata = {
        "id": dataset_id,
        "name": payload["name"],
        "profile": payload["profile"],
        "status": "queued",
        "total_sources": 0,
        "catalogs_used": [],
        "failures": [],
        "output_format": payload["output_format"],
        "export_path": None,
        "export_uri": None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    eng.metadata.put_dataset(metadata)
    return metadata


async def process_dataset_async(dataset_id: str, payload: dict[str, Any]) -> None:
    """Execute dataset generation asynchronously, updating job status."""
    engine = DatasetEngine()
    metadata = engine.get_dataset(dataset_id)
    if metadata is None:
        return
    metadata["status"] = "running"
    engine.metadata.put_dataset(metadata)
    logger.info("dataset_started", dataset_id=dataset_id)
    try:
        await engine.create_dataset(**payload, dataset_id=dataset_id)
        logger.info("dataset_completed", dataset_id=dataset_id)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        engine.metadata.put_dataset(metadata)
        logger.error("dataset_failed", dataset_id=dataset_id, error=str(exc))
        raise


def process_dataset(dataset_id: str, payload: dict[str, Any]) -> None:
    """Sync wrapper for worker executors."""
    asyncio.run(process_dataset_async(dataset_id, payload))


def submit_to_redis(dataset_id: str, payload: dict[str, Any]) -> bool:
    """Attempt enqueueing to Redis RQ; return False if Redis is not configured."""
    url = os.getenv("REDIS_URL")
    if not url:
        return False
    try:
        import redis
        from rq import Queue

        queue = Queue("datasets", connection=redis.from_url(url))
        queue.enqueue(process_dataset, dataset_id, payload, job_id=dataset_id, job_timeout=3600)
        return True
    except Exception as exc:
        logger.warning("redis_enqueue_failed", error=str(exc))
        return False
