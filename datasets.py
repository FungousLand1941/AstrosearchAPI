"""Streaming multi-format dataset generation, storage backends (SQLite/PostgreSQL/S3), and worker jobs.

A dataset is the crossmatch of a list of targets, filtered (confidence, query filters, count
threshold) and exported as JSON, CSV, Parquet or FITS. Target lists of at least
``DATASET_BATCH_MIN_TARGETS`` targets (default 50) are fetched with :mod:`batch` (TAP uploads,
CDS XMatch and paced cone searches: a few requests per catalog instead of one per target) and
then associated per target exactly as a single-object search (:meth:`CrossmatchService.finalize`);
targets for which a batch query failed, and every target when the batch cannot run at all, are
searched one by one. Every exported row carries its Bayesian match probabilities.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sqlite3
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, TextIO

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from crossmatch import AdvancedQuery, CrossmatchService, QueryBuilder, QueryValidator
from models import AstroSearchError, CatalogRegistry, Settings, validate_target

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Multi-Format Streaming Exporter
# ---------------------------------------------------------------------------

_JSON_COLUMNS = ("physical", "data", "metadata", "provenance", "links")
# Text columns: identifiers, then the JSON-encoded nested fields, then the association labels
# (group_id: the Bayesian object the row belongs to within its target's cone; match_flag:
# 'best' / 'secondary' / None; coincident_with: the row this one duplicates in its catalog).
_TEXT_COLUMNS = ("catalog", "source_id", *_JSON_COLUMNS, "group_id", "match_flag", "coincident_with")
# Probabilities (crossmatch.associate_matches, Budavari & Szalay 2008 / NWAY):
#   confidence / target_probability -- posterior that the row is the target's counterpart;
#   match_probability -- posterior that the row belongs to its group's object;
#   group_match_probability -- posterior of the group's association as a whole;
#   p_any -- probability that the target has any counterpart among the rows of its cone.
_FLOAT_COLUMNS = ("ra", "dec", "separation_arcsec", "confidence", "epoch", "positional_error_arcsec",
                  "match_probability", "target_probability", "group_match_probability", "p_any",
                  "target_ra", "target_dec")
_INT_COLUMNS = ("target_index",)
_BOOL_COLUMNS = ("contains_target",)
_COLUMNS = _TEXT_COLUMNS + _FLOAT_COLUMNS + _INT_COLUMNS + _BOOL_COLUMNS
#: Exported columns, in file order (CSV header, Parquet and FITS schema).
EXPORT_COLUMNS = _COLUMNS


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flat(source: dict[str, Any]) -> dict[str, Any]:
    """One export row: nested fields JSON-encoded, numbers as floats (non-finite -> None)."""
    row: dict[str, Any] = {}
    for key in _COLUMNS:
        value = source.get(key)
        if key in _JSON_COLUMNS:
            row[key] = json.dumps(value, default=str)
        elif key in _FLOAT_COLUMNS:
            row[key] = _float_or_none(value)
        elif key in _INT_COLUMNS:
            row[key] = int(value) if isinstance(value, int) and not isinstance(value, bool) else None
        elif key in _BOOL_COLUMNS:
            row[key] = bool(value) if isinstance(value, bool) else None
        else:
            row[key] = str(value) if value is not None else None
    return row


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
                + [(name, pa.int64()) for name in _INT_COLUMNS]
                + [(name, pa.bool_()) for name in _BOOL_COLUMNS]
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

                # FITS has no nulls for these types: NaN for a missing float, -1 for a missing
                # target index, False for an unknown contains_target, '' for missing text.
                def fits_value(key: str, val: Any) -> Any:
                    if val is not None:
                        return val
                    if key in _FLOAT_COLUMNS:
                        return float("nan")
                    if key in _INT_COLUMNS:
                        return -1
                    if key in _BOOL_COLUMNS:
                        return False
                    return ""

                import numpy as np

                kinds = {**dict.fromkeys(_TEXT_COLUMNS, "U"), **dict.fromkeys(_FLOAT_COLUMNS, "f8"),
                         **dict.fromkeys(_INT_COLUMNS, "i8"), **dict.fromkeys(_BOOL_COLUMNS, "bool")}
                columns = {}
                for key in _COLUMNS:
                    values = [fits_value(key, row[key]) for row in self._buffer]
                    kind = kinds[key]
                    if kind == "U":  # at least one character wide (FITS cannot store zero-width strings)
                        width = max([1, *(len(v) for v in values)])
                        columns[key] = np.array(values, dtype=f"U{width}")
                    else:
                        columns[key] = np.array(values, dtype=kind)
                Table(columns).write(self.path, format="fits", overwrite=True)
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


def crowded_targets(targets: list[dict[str, Any]], *, chunk: int = 20000) -> dict[int, str]:
    """{target index: crowded stellar system} for the targets inside one, exactly as
    ``astrometry.crowded_region`` decides it. A vectorised angular-distance screen against every
    system (with 1" of margin) selects the candidates, which ``crowded_region`` then confirms:
    100,000 targets take well under a second instead of ~13 s one by one."""
    import numpy as np

    from astrometry import (
        CROWDED_GALAXIES,
        CROWDED_MIN_RADIUS_ARCMIN,
        CROWDED_RH_FACTOR,
        CROWDED_STELLAR_SYSTEMS,
        crowded_region,
    )

    if not targets:
        return {}
    centres = [(ra, dec, max(CROWDED_RH_FACTOR * (rh if rh is not None else CROWDED_MIN_RADIUS_ARCMIN),
                              CROWDED_MIN_RADIUS_ARCMIN) / 60.0) for _, ra, dec, rh in CROWDED_STELLAR_SYSTEMS]
    centres += [(ra, dec, radius / 60.0) for _, ra, dec, radius in CROWDED_GALAXIES]
    region_ra = np.radians([c[0] for c in centres])
    region_dec = np.radians([c[1] for c in centres])
    limit = np.radians([c[2] for c in centres]) + np.radians(1.0 / 3600.0)
    ra = np.radians(np.array([float(t["ra"]) for t in targets]))
    dec = np.radians(np.array([float(t["dec"]) for t in targets]))
    found: dict[int, str] = {}
    for start in range(0, len(targets), chunk):
        r, d = ra[start:start + chunk, None], dec[start:start + chunk, None]
        # Haversine distance of every target to every system centre (radians).
        hav = (np.sin((d - region_dec) / 2.0) ** 2
               + np.cos(d) * np.cos(region_dec) * np.sin((r - region_ra) / 2.0) ** 2)
        distance = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        for offset in np.flatnonzero((distance <= limit).any(axis=1)):
            index = start + int(offset)
            region = crowded_region(float(targets[index]["ra"]), float(targets[index]["dec"]))
            if region is not None:
                found[index] = region
    return found


def batch_min_targets() -> int:
    """DATASET_BATCH_MIN_TARGETS (default 50): target lists at least this long are fetched with
    :mod:`batch`; 0 disables batch mode."""
    try:
        return max(0, int(os.getenv("DATASET_BATCH_MIN_TARGETS", "50")))
    except ValueError:
        return 50


class DatasetEngine:
    """Processes search targets into filtered, deduplicated datasets with durable metadata.

    ``registry``: the catalogs datasets may use (default: the service's, else the embedded
    registry plus the user catalogs of ``registry_path`` / CATALOG_REGISTRY_PATH, merged as
    ``vizier.load_registry`` does). ``service``: the CrossmatchService per-target searches run
    on (default: ``main.build_service`` on a client opened per dataset). ``min_batch_targets``:
    target lists at least this long are fetched with :mod:`batch` (default
    :func:`batch_min_targets`; 0 disables).
    """

    def __init__(
        self,
        registry_path: str | None = None,
        storage_path: str | None = None,
        service: CrossmatchService | None = None,
        *,
        registry: CatalogRegistry | None = None,
        min_batch_targets: int | None = None,
    ) -> None:
        if registry is None and service is not None:
            registry = service.registry
        if registry is None:
            import vizier

            registry = vizier.load_registry(registry_path or Settings().catalog_registry_path or None)
        self.registry = registry
        self.storage = Path(storage_path or os.getenv("DATASET_STORAGE_PATH") or "datasets").resolve()
        self.storage.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.min_batch_targets = batch_min_targets() if min_batch_targets is None else max(0, int(min_batch_targets))
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
        any_export_path: bool = False,
    ) -> dict[str, Any]:
        """Crossmatch every target and stream the filtered detections into an export file.

        ``export_path`` must lie inside DATASET_STORAGE_PATH (the REST API's rule, so a
        client cannot write anywhere on the server) unless ``any_export_path`` (the local
        CLI: any unused path of the user's).

        Each group of a target's crossmatch (one Bayesian object in its cone) contributes its
        members whose confidence is at least ``min_confidence`` and that pass the query and
        ``filters``, when at least ``count_threshold`` of them do. A row found for several
        targets is written once (for the first). Rows carry the target they were found for
        (``target_index``, ``target_ra``, ``target_dec``) and the association probabilities
        (``confidence`` = ``target_probability``, ``match_probability``, ``group_id``,
        ``group_match_probability``, ``contains_target``, ``match_flag``, ``p_any``).
        """
        if not name.strip() or not profile.strip():
            raise ValueError("name and profile are required")
        if output_format.lower() not in {"json", "csv", "parquet", "fits"}:
            raise ValueError("Unsupported output format")
        if not targets:
            raise ValueError("At least one target with ra and dec is required")
        limit = Settings().max_radius_arcsec
        if not math.isfinite(float(radius_arcsec)) or not 0.0 < float(radius_arcsec) <= limit:
            # Every target's cone goes to every archive of the profile (main.check_search_radius).
            raise ValueError(f"radius_arcsec must be in (0, {limit:g}] arcsec (API_MAX_RADIUS_ARCSEC), "
                             f"got {radius_arcsec!r}")

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
            validate_target(target["ra"], target["dec"], epoch=target.get("epoch"),
                            pm_ra_masyr=target.get("pm_ra_masyr"), pm_dec_masyr=target.get("pm_dec_masyr"))

        dataset_id = dataset_id or uuid.uuid4().hex
        path = (self.check_export_path(export_path, output_format, any_path=any_export_path) if export_path
                else self.storage / f"{dataset_id}.{output_format.lower()}")

        seen: set[tuple[str, str]] = set()
        catalogs_used: set[str] = set()
        failures: list[dict[str, Any]] = []
        run_info: dict[str, Any] = {"method": "per-target", "batch": None}

        def export_row(index: int, group: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
            target = targets[index]
            return {
                **source,
                "target_index": index,
                "target_ra": target.get("ra"),
                "target_dec": target.get("dec"),
                "group_id": group.get("group_id"),
                "group_match_probability": group.get("match_probability"),
                "contains_target": group.get("contains_target"),
                "match_flag": group.get("match_flag"),
                "p_any": group.get("p_any"),
            }

        async def collect(active_service: CrossmatchService, writer: DatasetWriter) -> None:
            async for index, result in self._search_all(active_service, targets, query, run_info):
                failures.extend({**failure, "target_index": index} for failure in result.get("failures", []))
                for group in result["crossmatch_groups"]:
                    members = [
                        source for source in group["members"]
                        if (source.get("confidence") or 0.0) >= min_confidence
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
                        writer.write(export_row(index, group, source))

        with DatasetWriter(path, output_format) as writer:
            if self.service is None:
                from main import build_service  # lazy: main imports this module

                settings = Settings()
                async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
                    await collect(build_service(settings=settings, client=client, registry=self.registry), writer)
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
            "targets": len(targets),
            "catalogs_used": sorted(catalogs_used),
            "failures": failures,
            "method": run_info["method"],
            "batch": run_info["batch"],
            "columns": list(EXPORT_COLUMNS) if output_format.lower() != "json" else None,
            "output_format": output_format.lower(),
            "export_path": str(path),
            "created_at": previous["created_at"] if previous else datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "export_uri": export_uri,
        }
        self.metadata.put_dataset(metadata)
        return metadata

    async def _search_all(
        self,
        service: CrossmatchService,
        targets: list[dict[str, Any]],
        query: AdvancedQuery,
        run_info: dict[str, Any],
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """(target index, record dict) for every target, in target order: from one batch run when
        the list is long enough (:meth:`_batch_results`), per target otherwise and for every
        target the batch did not serve completely.

        Targets inside a crowded stellar system (``astrometry.crowded_region``: the Harris
        globular clusters, the LMC/SMC/M31/M33 cores) are always searched one by one: there a
        single-object search measures each catalogue's local source density with a wider density
        probe, which a batch (whose cones are fetched in bulk) cannot run, so the batch would
        fall back on the density map and overstate the posteriors.
        """
        batch_results: dict[int, dict[str, Any]] = {}
        if self.min_batch_targets and len(targets) >= self.min_batch_targets:
            crowded = await asyncio.to_thread(crowded_targets, targets)
            eligible = [i for i in range(len(targets)) if i not in crowded]
            if len(eligible) >= self.min_batch_targets:
                batch_results = await self._batch_results(service, targets, eligible, query, run_info)
            else:
                run_info["batch"] = {"min_targets": self.min_batch_targets,
                                     "error": f"only {len(eligible)} targets outside crowded stellar systems"}
            if crowded:
                run_info["batch"]["crowded_targets"] = len(crowded)
                run_info["batch"]["crowded_regions"] = sorted(set(crowded.values()))
        pending = [i for i in range(len(targets)) if i not in batch_results]
        if batch_results:
            run_info["method"] = "batch+per-target" if pending else "batch"
        # Per-target results arrive in the (ascending) order of ``pending``.
        singles = self._search_targets(service, [targets[i] for i in pending], query).__aiter__()
        for index in range(len(targets)):
            if index in batch_results:
                yield index, batch_results.pop(index)
            else:
                yield index, await singles.__anext__()

    async def _batch_results(
        self,
        service: CrossmatchService,
        targets: list[dict[str, Any]],
        indices: list[int],
        query: AdvancedQuery,
        run_info: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        """Record-like dicts ({crossmatch_groups, failures}), keyed by target index, of the targets
        (``indices`` of ``targets``) one batch run served
        completely (every catalog of the dataset answered for them), from
        :class:`batch.BatchCrossmatcher` with ``keep_groups``: the rows are fetched by TAP upload,
        CDS XMatch or paced cones and associated per target by ``CrossmatchService.finalize``,
        the single-object pipeline. Targets with a failed (target, catalog) query are left out
        and searched one by one afterwards; a batch that cannot run (a radius beyond the XMatch
        limit, too many catalogs, too many cone-only targets, an unexpected error) leaves every
        target to the per-target path, with the reason in the dataset metadata (``batch.error``).
        """
        import batch as batch_module

        catalogs = [plan.catalog for plan in QueryBuilder(self.registry).build(query)]
        info: dict[str, Any] = {"min_targets": self.min_batch_targets, "catalogs": catalogs}
        run_info["batch"] = info
        if not catalogs:
            info["error"] = "no enabled catalog matches the dataset's profile and catalogs"
            return {}
        if not 0.0 < float(query.radius_arcsec) <= batch_module.MAX_RADIUS_ARCSEC:
            info["error"] = (f"radius {query.radius_arcsec:g} arcsec exceeds the batch limit of "
                             f"{batch_module.MAX_RADIUS_ARCSEC:g} arcsec")
            return {}
        items = []
        for index in indices:
            target = targets[index]
            item = {"id": str(index), "ra": target["ra"], "dec": target["dec"]}
            for key in ("epoch", "pm_ra_masyr", "pm_dec_masyr", "parallax_mas"):
                if target.get(key) is not None:
                    item[key] = target[key]
            items.append(item)
        providers = getattr(service, "providers", None)
        providers = providers if isinstance(providers, dict) else {}
        shared = providers.get("tap")
        client = next((getattr(p, "client", None) for p in providers.values() if getattr(p, "client", None) is not None),
                      None)
        engine = batch_module.BatchCrossmatcher(
            registry=self.registry,
            client=client,
            guards=getattr(shared, "guards", None),
            cache=getattr(shared, "cache", None),
            association_config=getattr(service, "association_config", None),
            keep_groups=True,
        )
        try:
            result = await engine.run(items, catalogs, radius_arcsec=float(query.radius_arcsec))
        except (AstroSearchError, ValueError) as exc:
            info["error"] = str(exc)
            logger.warning("dataset_batch_unavailable", error=str(exc))
            return {}
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("dataset_batch_failed", error=info["error"])
            return {}
        failed = sorted(int(target_id) for target_id in result.failures_by_id())
        info.update({
            "strategies": {name: run.strategy for name, run in result.runs.items()},
            "request_count": result.request_count,
            "wall_time_s": round(result.wall_time_s, 3),
            "fallback_targets": failed,
        })
        skipped = set(failed)
        # BatchResult.groups is keyed by position in the batch; the batch ids are target indices.
        return {index: {"crossmatch_groups": result.groups.get(position, []), "failures": []}
                for position, index in enumerate(indices) if index not in skipped}

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
            chunk = targets[offset:offset + 10]
            for res in await asyncio.gather(*(search(t) for t in chunk)):
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

    def check_export_path(self, export_path: str | os.PathLike[str], output_format: str, *,
                          any_path: bool = False) -> Path:
        """The resolved export path, or ValueError: it must be unused, carry the format's
        extension, and (unless ``any_path``, the local CLI) lie inside DATASET_STORAGE_PATH:
        a REST client may not write anywhere on the server."""
        path = Path(export_path).expanduser().resolve()
        if not any_path and not path.is_relative_to(self.storage):
            raise ValueError("output_path must be within DATASET_STORAGE_PATH")
        if path.exists() or path.suffix.lower() != f".{output_format.lower()}":
            raise ValueError(f"output path must not exist yet and must end in .{output_format.lower()}: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"output directory does not exist: {path.parent}")
        return path

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


async def process_dataset_async(dataset_id: str, payload: dict[str, Any], engine: DatasetEngine | None = None) -> None:
    """Execute dataset generation asynchronously, updating job status. ``engine``: the API's
    engine (its service, registry and storage); a worker process builds its own."""
    engine = engine or DatasetEngine()
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
