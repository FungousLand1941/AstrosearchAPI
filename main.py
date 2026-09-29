"""Programmatic facade, command-line interface (CLI) and built-in offline verification suite.

Every feature module plugs into the CLI through ``register_cli(subparsers)``, which adds its
subcommands and sets ``handler`` (a function returning the exit status). The core commands
(serve, search, dataset, catalogs, benchmark, verify) are defined here.

The service factories used by the API (``api.py``), the CLI and the feature modules' own
fallbacks live here too:

* :func:`build_registry` -- the embedded catalog registry merged with the user registry YAML
  that ``vizier add`` / ``POST /api/v1/vizier/register`` write (``vizier.load_registry``);
* :func:`build_providers` -- the archive adapters, answering from the local sky cache first
  for catalogs mirrored there (``skycache``), with the archive as fallback;
* :func:`build_service` -- a :class:`crossmatch.CrossmatchService` over both.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

# Only the modules the service factories need are imported here; the CLI (cli.py) imports a
# feature module when its command runs, so `import main` stays cheap.
import cli
import skycache
import streaming
import vizier
from crossmatch import (
    AdvancedQuery,
    CrossmatchService,
    angular_separation_arcsec,
    match_score,
    resolved_search_target,
)
from datasets import DatasetEngine, DatasetWriter
from models import (
    AstroSearchError,
    CatalogDefinition,
    CatalogRegistry,
    InvalidCoordinateError,
    Settings,
    Target,
    UnifiedRecord,
    every_catalog_failed,
    normalize_source_record,
    parse_ipac_records,
    validate_target,
)
from providers import (
    CatalogProvider,
    EndpointGuard,
    SesameResolver,
    provider_map,
)

# Feature modules whose ``register_cli(subparsers)`` adds subcommands, in help order
# (cli.FEATURE_COMMANDS lists their commands, so --help needs none of them imported).
CLI_MODULE_NAMES = tuple(cli.FEATURE_COMMANDS)

# ---------------------------------------------------------------------------
# Service factories
# ---------------------------------------------------------------------------


def build_registry(*, settings: Settings | None = None, registry_path: str | None = None) -> CatalogRegistry:
    """The catalog registry the service queries: the embedded catalogs plus the user
    catalogs registered with ``vizier add`` (``vizier.load_registry``; a registry file
    without ``extends: embedded`` is read as a complete registry, as before).

    Every registry problem is logged; CATALOG_REGISTRY_STRICT=true raises instead.
    """
    active = settings or Settings()
    registry = vizier.load_registry(registry_path or active.catalog_registry_path or None)
    registry.startup_check()
    return registry


def skycache_enabled() -> bool:
    """SKYCACHE_ENABLED (default true): answer cones of mirrored catalogs from the local store."""
    return os.getenv("SKYCACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


@functools.lru_cache(maxsize=8)
def _store_at(path: str) -> skycache.SkyCache:
    return skycache.SkyCache(path)


def skycache_store(path: str | os.PathLike[str] | None = None) -> skycache.SkyCache:
    """The sky cache at ``path`` (default ``$SKYCACHE_PATH`` or ``~/.astrosearch/skycache``),
    one instance per directory per process, so every service shares its loaded partitions."""
    root = Path(path).expanduser() if path else skycache.default_store_path()
    return _store_at(str(root.resolve()))


class MirrorFirstProvider(skycache.SkyCacheProvider):
    """An archive adapter that answers from the local sky cache when the catalog is mirrored.

    For a catalog present in the store, the local cone search runs first; a cone it does not
    fully cover (or a stale / changed / damaged mirror) falls back to the archive, recorded
    in ``meta['skycache']`` (see :class:`skycache.SkyCacheProvider`). A catalog that was
    never mirrored goes straight to the archive, without a worker-thread hop or a
    ``skycache`` entry, so a service without mirrors behaves exactly like the bare adapters.
    The store is checked on every query, so a catalog mirrored while the API runs
    (``POST /api/v1/skycache/mirror``) is used at once.

    Attributes of the archive adapter (``client``, ``guards``, ``cache``, ``timeout``, ...)
    are read through, so the batch router, the provenance replay and the monitoring
    endpoint see the same endpoint guards and HTTP client. :func:`skycache.archive_providers`
    unwraps it (mirroring always queries the archive).
    """

    def __getattr__(self, name: str) -> Any:
        if name in {"remote", "local"}:  # not set yet (copy / unpickling): no recursion
            raise AttributeError(name)
        return getattr(self.remote, name)

    async def query(self, catalog: CatalogDefinition, target: Target, radius_arcsec: float) -> Any:
        if not self.local.store.has(catalog.name):
            return await self.remote.query(catalog, target, radius_arcsec)
        return await super().query(catalog, target, radius_arcsec)


def build_providers(
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    store: skycache.SkyCache | None = None,
) -> dict[str, CatalogProvider]:
    """Archive adapters on one HTTP client (one guard map and cache shared between them),
    wrapped in :class:`MirrorFirstProvider` unless SKYCACHE_ENABLED=false."""
    active = settings or Settings()
    providers: dict[str, CatalogProvider] = provider_map(
        client,
        timeout=active.request_timeout_seconds,
        max_response_bytes=active.max_response_bytes,
    )
    if skycache_enabled():
        local = skycache.LocalProvider(store or skycache_store())
        providers = {name: MirrorFirstProvider(provider, local) for name, provider in providers.items()}
    return providers


def build_service(
    *,
    settings: Settings | None = None,
    registry_path: str | None = None,
    client: httpx.AsyncClient | None = None,
    registry: CatalogRegistry | None = None,
    providers: Mapping[str, CatalogProvider] | None = None,
    service_class: type[CrossmatchService] = CrossmatchService,
) -> CrossmatchService:
    """A configured CrossmatchService: :func:`build_registry` (unless ``registry`` is given)
    and :func:`build_providers` on ``client`` (unless ``providers`` is given)."""
    active_settings = settings or Settings()
    if providers is None:
        providers = build_providers(settings=active_settings, client=client)
    return service_class(
        registry if registry is not None else build_registry(settings=active_settings, registry_path=registry_path),
        # The same mapping object (app.state.providers is the service's providers).
        providers if isinstance(providers, dict) else dict(providers),
        radius_arcsec=active_settings.default_radius_arcsec,
        timeout=active_settings.request_timeout_seconds,
        timeout_cap=active_settings.catalog_timeout_cap_seconds,
    )


def resolution_kwargs(spec: Mapping[str, Any], resolved: Any) -> dict[str, Any]:
    """CrossmatchService.crossmatch keywords carrying a resolver's answer: the target's position
    and proper-motion uncertainties (and where they came from) and the resolved object (its
    catalogue row is the target's identity)."""
    kwargs = {"target_uncertainty_arcsec": spec.get("target_uncertainty_arcsec"),
              "target_pm_error_masyr": spec.get("target_pm_error_masyr"),
              "target_uncertainty_source": spec.get("target_uncertainty_source"),
              "resolved_object": resolved}
    return {key: value for key, value in kwargs.items() if value is not None}


def search_query(fields: Mapping[str, Any], spec: Mapping[str, Any] | None = None,
                 resolved_info: Mapping[str, Any] | None = None) -> AdvancedQuery:
    """The AdvancedQuery of ``POST /api/v1/search`` from api.SearchRequest ``fields`` (all present).

    For a name search ``spec`` is :func:`crossmatch.resolved_search_target` of the resolver's
    answer (position, epoch, motion, parallax and pm_source) and ``resolved_info`` that answer's
    ``as_dict()``; otherwise the coordinates and motion are the fields' ("input" motion).
    """
    epoch, pm_ra, pm_dec, parallax = (fields.get("epoch"), fields.get("pm_ra_masyr"), fields.get("pm_dec_masyr"),
                                      fields.get("parallax_mas"))
    pm_source = "input" if pm_ra is not None else None
    if spec is not None:
        ra, dec, epoch = spec["ra"], spec["dec"], spec["epoch"]
        pm_ra, pm_dec, parallax = spec["pm_ra_masyr"], spec["pm_dec_masyr"], spec["parallax_mas"]
        pm_source = spec["pm_source"]
    else:
        if fields.get("ra") is None or fields.get("dec") is None:
            raise ValueError("Either name or both ra and dec are required")
        ra, dec = fields["ra"], fields["dec"]
    metadata: dict[str, Any] = {}
    if pm_source:
        metadata["pm_source"] = pm_source
    if resolved_info:
        metadata["resolved_object"] = dict(resolved_info)
    profile = fields.get("profile")
    return AdvancedQuery.from_dict({
        "ra": ra,
        "dec": dec,
        "epoch": epoch,
        "pm_ra_masyr": pm_ra,
        "pm_dec_masyr": pm_dec,
        "parallax_mas": parallax,
        "radius_arcsec": fields["radius_arcsec"],
        "profiles": [profile] if profile else None,
        "object_types": fields.get("object_types"),
        "spectral_types": fields.get("spectral_types"),
        "morphology": fields.get("morphology"),
        "min_confidence": fields["min_confidence"],
        "max_results": fields.get("max_results"),
        "catalogs": fields.get("catalogs"),
        "time_period": fields.get("time_period"),
        "spatial_constraints": fields.get("spatial_constraints"),
        "search_mode": fields["search_mode"],
        "min_radius_arcsec": fields["min_radius_arcsec"],
        "proper_motion": fields["proper_motion"],
        "adaptive_radius": fields["adaptive_radius"],
        "min_distance_pc": fields.get("min_distance_pc"),
        "max_distance_pc": fields.get("max_distance_pc"),
        "metadata": metadata or None,
    })


async def api_search(service: Any, fields: Mapping[str, Any], resolver: Any = None) -> UnifiedRecord:
    """A search exactly as ``POST /api/v1/search`` runs it (without its response cache), shared
    with provenance's ``run_api_search``: ``fields`` are the api.SearchRequest fields (all
    present) and ``resolver`` resolves a name.

    A name's resolver answer is the target: its position at the resolver epoch (J2000 for
    SIMBAD) with its motion and parallax, its errors as the target uncertainty, and the object
    itself (its catalogue row is the target's identity). A requested epoch moves the position
    there with the requested (else the resolver's) proper motion.
    """
    spec: dict[str, Any] | None = None
    resolved_info: dict[str, Any] | None = None
    extra: dict[str, Any] = {}
    if fields.get("name"):
        resolved = await resolver.resolve(fields["name"])
        resolved_info = resolved.as_dict()
        spec = resolved_search_target(resolved, epoch=fields.get("epoch"), pm_ra_masyr=fields.get("pm_ra_masyr"),
                                      pm_dec_masyr=fields.get("pm_dec_masyr"), parallax_mas=fields.get("parallax_mas"))
        extra = resolution_kwargs(spec, resolved)
    query = search_query(fields, spec, resolved_info)
    ra, dec = (spec["ra"], spec["dec"]) if spec is not None else (fields["ra"], fields["dec"])
    result = await service.crossmatch(ra, dec, query=query, **extra)
    if spec is not None:
        merge_resolution(result, spec)
    if resolved_info:
        result.resolved_object = resolved_info
    return result


def merge_resolution(record: UnifiedRecord, spec: Mapping[str, Any]) -> UnifiedRecord:
    """Add the notes and warnings of :func:`crossmatch.resolved_search_target` (VizieR-only
    resolver answers, several objects for one name, a position moved to another epoch) to a
    record's provenance, as the streaming search does."""
    provenance_block = record.provenance if isinstance(record.provenance, dict) else {}
    warnings = provenance_block.setdefault("warnings", [])
    if isinstance(warnings, list):
        warnings.extend(w for w in spec.get("warnings") or [] if w not in warnings)
    association = provenance_block.get("association")
    if isinstance(association, dict):
        notes = association.setdefault("notes", [])
        if isinstance(notes, list):
            notes.extend(n for n in spec.get("notes") or [] if n not in notes)
    return record


# ---------------------------------------------------------------------------
# Public Programmatic API
# ---------------------------------------------------------------------------


async def crossmatch(
    ra: float | str,
    dec: float | str,
    *,
    radius_arcsec: float | None = None,
    epoch: float | None = None,
    profile: str | None = None,
    settings: Settings | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
    catalogs: list[str] | None = None,
) -> UnifiedRecord:
    """Execute a single coordinate crossmatch using a fresh client session.

    ``epoch`` is the Julian year of (ra, dec); with ``pm_ra_masyr``/``pm_dec_masyr``
    the target is followed to each catalog's epoch.
    """
    active_settings = settings or Settings()
    async with httpx.AsyncClient(timeout=active_settings.request_timeout_seconds, follow_redirects=True) as client:
        service = build_service(settings=active_settings, client=client)
        return await service.crossmatch(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile,
                                        pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas,
                                        catalogs=catalogs)


async def search_object(
    name: str,
    *,
    radius_arcsec: float | None = None,
    profile: str | None = None,
    settings: Settings | None = None,
    resolver: SesameResolver | None = None,
    catalogs: list[str] | None = None,
    epoch: float | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
) -> UnifiedRecord:
    """Resolve an object name with CDS Sesame and crossmatch it.

    The resolver's answer is the search target (:func:`crossmatch.resolved_search_target`):
    its position at the resolver epoch (J2000 for SIMBAD) with its proper motion and parallax,
    its position and motion errors as the target uncertainty, and the resolved object itself,
    so the catalogue row of the resolved name is the target's identity. A requested ``epoch``
    moves the resolved position there with the requested (or the resolver's) proper motion.
    """
    active_settings = settings or Settings()
    async with httpx.AsyncClient(timeout=active_settings.request_timeout_seconds, follow_redirects=True) as client:
        active_resolver = resolver or SesameResolver(client, endpoint=active_settings.resolver_endpoint)
        resolved = await active_resolver.resolve(name)
        service = build_service(settings=active_settings, client=client)
        return await crossmatch_resolved(service, resolved, radius_arcsec=radius_arcsec, profile=profile,
                                         catalogs=catalogs, epoch=epoch, pm_ra_masyr=pm_ra_masyr,
                                         pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas)


async def crossmatch_resolved(
    service: Any,
    resolved: Any,
    *,
    radius_arcsec: float | None = None,
    profile: str | None = None,
    catalogs: list[str] | None = None,
    epoch: float | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
) -> UnifiedRecord:
    """The basic crossmatch of a resolver answer, every in-radius row kept: what
    ``astrosearch search --name`` runs (:func:`search_object`), shared with provenance's
    ``run_basic_search`` so manifests of such searches are built from the same code."""
    spec = resolved_search_target(resolved, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                  parallax_mas=parallax_mas)
    result = await service.crossmatch(
        spec["ra"],
        spec["dec"],
        radius_arcsec=radius_arcsec,
        epoch=spec["epoch"],
        profile=profile,
        pm_ra_masyr=spec["pm_ra_masyr"],
        pm_dec_masyr=spec["pm_dec_masyr"],
        pm_source=spec["pm_source"],
        parallax_mas=spec["parallax_mas"],
        catalogs=catalogs,
        **resolution_kwargs(spec, resolved),
    )
    merge_resolution(result, spec)
    result.resolved_object = resolved.as_dict()
    result.provenance["resolver"] = resolved.resolver
    return result


def catalog_definitions(*, settings: Settings | None = None) -> dict[str, Any]:
    """Configured catalog definitions (embedded + user-registered VizieR tables)."""
    return build_registry(settings=settings).catalogs


# ---------------------------------------------------------------------------
# Built-In Offline Verification Engine
# ---------------------------------------------------------------------------


def run_verification() -> bool:
    """Run the offline self-test suite (no network): core maths, parsers, registry, exports,
    the REST API with every feature router mounted, and the CLI registration."""
    print("=" * 70)
    print("AstroSearch Built-In Verification Suite (Offline)")
    print("=" * 70)
    passed = 0
    total = 0

    def check(name: str, test_fn: Callable[[], None]) -> None:
        nonlocal passed, total
        total += 1
        print(f"[{total:02d}] Testing {name:.<50} ", end="", flush=True)
        try:
            test_fn()
            print("PASSED")
            passed += 1
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    # 1. Coordinate Validation & Normalization
    def test_coords() -> None:
        t = validate_target(361.0, 10.0)
        assert t.ra == 1.0 and t.dec == 10.0
        try:
            validate_target(10.0, 95.0)
            raise AssertionError("Should have rejected dec > 90")
        except InvalidCoordinateError:
            pass

    check("Coordinate normalization and bounds validation", test_coords)

    # 2. Angular Separation Math
    def test_sep() -> None:
        sep = angular_separation_arcsec(Target(0.0, 0.0), Target(0.0, 1.0))
        assert 3590 < sep < 3610

    check("Angular separation calculation (arcseconds)", test_sep)

    # 3. Probabilistic Match Scoring
    def test_scoring() -> None:
        score_close = match_score(0.2, positional_error_arcsec=1.0)
        score_far = match_score(5.0, positional_error_arcsec=1.0)
        assert 0.0 <= score_far < score_close <= 1.0

    check("Probabilistic Gaussian match scoring", test_scoring)

    # 4. Normalization of Astronomical Field Aliases
    def test_norm() -> None:
        record = {
            "RAJ2000": 12.5,
            "DEJ2000": -4.2,
            "designation": "J0001",
            "plx_value": 15.2,
            "sp_type": "G2V",
        }
        res = normalize_source_record(record)
        assert res["ra"] == 12.5 and res["dec"] == -4.2
        assert res["source_id"] == "J0001" and res["parallax"] == 15.2
        assert res["spectral_type"] == "G2V"

    check("Astrometric field normalization (aliases)", test_norm)

    # 5. IPAC ASCII Table Parser (IRSA Gator)
    def test_ipac() -> None:
        payload = (
            "\\fixlen = T\n"
            "\\RowsRetrieved = 1\n"
            "| ra        | dec       | designation |\n"
            "| double    | double    | char        |\n"
            "|           |           |             |\n"
            " 10.123456   -5.654321   J001\n"
        )
        rows = parse_ipac_records(payload)
        assert len(rows) == 1 and rows[0]["designation"] == "J001"
        assert abs(float(rows[0]["ra"]) - 10.123456) < 1e-5

    check("IPAC table parser (IRSA Gator)", test_ipac)

    # 6. Embedded Catalog Registry
    def test_registry() -> None:
        reg = CatalogRegistry()
        enabled = reg.enabled_catalogs()
        assert len(enabled) >= 15
        assert "gaia_dr3" in enabled
        assert "twomass_psc" in enabled
        assert "allwise" in enabled
        assert "sdss" in enabled

    check("Embedded catalog registry loading", test_registry)

    # 7. CDS Sesame XML Parsing
    def test_sesame() -> None:
        # Structure of a real Sesame -oxp answer (nested pm/z elements, Resolver name).
        xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
        <Sesame><Target option="SNV"><name>Barnard's star</name>
          <Resolver name="Sc=Simbad (CDS, via client/server)">
            <otype>BY*</otype><jradeg>269.45207696</jradeg><jdedeg>4.69336497</jdedeg>
            <pm><v>10393.349</v><pmRA>-801.551</pmRA><pmDE>10362.394</pmDE></pm>
            <Vel><v>-110.11</v></Vel><plx><v>546.9759</v></plx>
            <oname>NAME Barnard's star</oname>
          </Resolver></Target></Sesame>
        """
        res = SesameResolver.parse_response("Barnard's star", xml_payload)
        assert res.canonical_name == "NAME Barnard's star"
        assert abs(res.ra_deg - 269.45207696) < 1e-7
        assert res.object_type == "BY*"
        assert res.pm_ra_masyr == -801.551 and res.pm_dec_masyr == 10362.394
        assert res.epoch == 2000.0 and res.resolver_metadata["parallax_mas"] == 546.9759

    check("CDS Sesame XML resolver parser", test_sesame)

    # 8. Advanced Query DSL & Multi-Constraint Filters
    def test_filters() -> None:
        q = AdvancedQuery.from_dict({
            "ra": 10.0,
            "dec": 5.0,
            "radius_arcsec": 10.0,
            "search_mode": "shell",
            "min_radius_arcsec": 2.0,
            "object_types": ["star"],
            "spatial_constraints": {"radius_zones": [{"min_arcsec": 2.0, "max_arcsec": 8.0}]},
        })
        source_in = {
            "ra": 10.001,
            "dec": 5.0,
            "separation_arcsec": 3.0,
            "physical": {"object_type": "star", "parallax": 10.0},
        }
        assert q.apply_filters(source_in)

        # Fails min radius for shell
        source_too_close = {**source_in, "separation_arcsec": 1.0}
        assert not q.apply_filters(source_too_close)

        # Cylinder distance bounds
        q.search_mode = "cylinder"
        q.min_distance_pc = 90.0
        q.max_distance_pc = 110.0
        assert q.apply_filters(source_in)  # 1000 / 10 = 100 pc -> in range
        q.max_distance_pc = 95.0
        assert not q.apply_filters(source_in)

    check("Advanced query filters (shell, cylinder, zones)", test_filters)

    # 9. EndpointGuard Circuit Breaker & Rate Limiter
    def test_guard() -> None:
        guard = EndpointGuard(requests_per_second=1000, failure_threshold=2, recovery_seconds=10)
        assert guard.state == "closed"
        asyncio.run(guard.fail())
        assert guard.state == "closed"
        asyncio.run(guard.fail())
        assert guard.state == "open"
        asyncio.run(guard.succeed())
        assert guard.state == "closed"

    check("EndpointGuard rate limiter and circuit breaker", test_guard)

    # 10. Streaming Dataset Multi-Format Export (JSON, CSV, Parquet, FITS)
    def test_dataset_export() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_source = {
                "catalog": "gaia_dr3",
                "source_id": "gaia-001",
                "ra": 187.25,
                "dec": 2.05,
                "confidence": 0.98,
                "match_probability": 0.97,
                "target_probability": 0.98,
                "separation_arcsec": 0.5,
                "physical": {"object_type": "star"},
            }
            for fmt in ("json", "csv", "parquet", "fits"):
                out_path = Path(tmpdir) / f"test.{fmt}"
                with DatasetWriter(out_path, fmt) as writer:
                    writer.write(sample_source)
                assert out_path.exists()
                assert out_path.stat().st_size > 0

    check("Streaming dataset exports (JSON, CSV, Parquet, FITS)", test_dataset_export)

    # 11. FastAPI REST API Route Suite
    def test_api_routes() -> None:
        from fastapi.testclient import TestClient

        from api import app

        with TestClient(app) as client:
            # Health
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            assert res.json()["status"] == "healthy"

            # Catalogs
            res = client.get("/api/v1/catalogs")
            assert res.status_code == 200
            assert "gaia_dr3" in res.json()

            # Specific catalog
            res = client.get("/api/v1/catalogs/gaia_dr3")
            assert res.status_code == 200
            assert res.json()["name"] == "gaia_dr3"

            # Saved Queries CRUD
            post_q = client.post("/api/v1/queries", json={"name": "M87-query", "query": {"ra": 187.7, "dec": 12.39}})
            assert post_q.status_code == 201
            query_id = post_q.json()["id"]

            list_q = client.get("/api/v1/queries")
            assert list_q.status_code == 200
            assert any(q["id"] == query_id for q in list_q.json())

            del_q = client.delete(f"/api/v1/queries/{query_id}")
            assert del_q.status_code == 204

            # Stats & Monitoring
            assert client.get("/api/v1/stats").status_code == 200
            assert client.get("/api/v1/monitoring").status_code == 200

    check("FastAPI REST endpoints and lifecycle", test_api_routes)

    # 12. Every feature router and the web UI are mounted
    def test_feature_routes() -> None:
        from api import route_table

        paths = {path for path, _methods in route_table()}
        expected = {
            "/", "/api/v1/search/stream", "/api/v1/batch/crossmatch", "/api/v1/skycache/status",
            "/api/v1/vizier/search", "/api/v1/sed", "/api/v1/lightcurves", "/api/v1/cutouts",
            "/api/v1/ai/query", "/api/v1/provenance/manifest", "/vo/tap/sync", "/api/v1/alerts",
        }
        missing = expected - paths
        assert not missing, f"routes not mounted: {sorted(missing)}"

    check("Feature routers and web UI mounted", test_feature_routes)

    # 13. Every feature CLI is registered
    def test_cli_commands() -> None:
        commands = {name for action in build_parser()._actions if isinstance(action, argparse._SubParsersAction)
                    for name in action.choices}
        expected = {"serve", "search", "dataset", "catalogs", "benchmark", "verify", "stream", "xmatch-calibrate",
                    "batch", "mirror", "skycache", "vizier", "sed", "lightcurve", "solar-system", "cutout", "ask",
                    "explain", "replay", "cite", "manifest", "vo", "alerts"}
        missing = expected - commands
        assert not missing, f"subcommands not registered: {sorted(missing)}"

    check("Feature CLI subcommands registered", test_cli_commands)

    print("=" * 70)
    print(f"Verification Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    return passed == total


# ---------------------------------------------------------------------------
# Command Line Interface (CLI)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The full ``astrosearch`` argument parser: the core commands plus every feature module's
    (imports all of them; the console script builds only what the command needs, cli.main)."""
    return cli.build_parser(full=True)


def _catalog_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--catalogs must name at least one catalog")
    return names


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.name and (args.ra is None or args.dec is None):
        print("Error: Specify either --name or both --ra and --dec.", file=sys.stderr)
        return 1

    async def run_search() -> UnifiedRecord:
        catalogs = _catalog_list(args.catalogs)
        if args.name:
            if args.format == "summary":  # never mixed into JSON output
                print(f"Resolving '{args.name}' via CDS Sesame...")
            return await search_object(args.name, radius_arcsec=args.radius, profile=args.profile, catalogs=catalogs,
                                       epoch=args.epoch, pm_ra_masyr=args.pm_ra, pm_dec_masyr=args.pm_dec,
                                       parallax_mas=args.parallax)
        return await crossmatch(args.ra, args.dec, radius_arcsec=args.radius, profile=args.profile, catalogs=catalogs,
                                epoch=args.epoch, pm_ra_masyr=args.pm_ra, pm_dec_masyr=args.pm_dec,
                                parallax_mas=args.parallax)

    try:
        res = asyncio.run(run_search())
    except (AstroSearchError, ValueError) as exc:
        # An unresolvable name or an invalid profile/coordinate: a one-line
        # error, not a traceback (and not a 'catalogs failed' message).
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    outage = every_catalog_failed(res)
    if args.format == "json":
        print(json.dumps(streaming.jsonable(res.as_dict()), indent=2))
    else:
        print("\n--- Crossmatch Summary ---")
        print(f"Target: RA={res.target['ra']:.6f}, DEC={res.target['dec']:.6f}")
        print(f"Catalogs Queried: {res.catalogs_queried}")
        print(f"Physical Objects Grouped: {len(res.crossmatch_groups)}")
        target_group = next((g for g in res.crossmatch_groups if g.get("contains_target")), None)
        if target_group:
            members = ", ".join(f"{m['catalog']}:{m['source_id']} (P={m.get('target_probability')})"
                                for m in target_group.get("members", []))
            print(f"Target counterparts: {members}")
        for wave, sources in res.counterparts.items():
            print(f"  [{wave.upper()}] {len(sources)} counterpart(s)")
        if res.failures:
            print(f"Failures ({len(res.failures)}): {[f['catalog'] for f in res.failures]}")
    if outage:
        # Exit 1: a script must tell a total outage from an empty sky (exit 0, no rows).
        print(f"Error: every queried catalog failed ({len(res.failures)} failure(s)); no data was retrieved.",
              file=sys.stderr)
        return 1
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    targets_file = Path(args.targets)
    if not targets_file.exists():
        print(f"Error: Targets file '{args.targets}' not found.", file=sys.stderr)
        return 1
    try:
        targets_data = json.loads(targets_file.read_text(encoding="utf-8"))
        if not isinstance(targets_data, list):
            raise ValueError("the targets file must hold a JSON array of {ra, dec[, epoch]} objects")
        catalogs = _catalog_list(args.catalogs)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    async def do_dataset() -> dict[str, Any]:
        settings = Settings()
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True) as client:
            service = build_service(settings=settings, client=client)
            engine = DatasetEngine(registry=service.registry, service=service)
            if args.output:  # checked before any query runs (the CLI may write anywhere)
                engine.check_export_path(args.output, args.format, any_path=True)
            print(f"Building dataset '{args.name}' across {len(targets_data)} targets...")
            return await engine.create_dataset(
                name=args.name,
                profile=args.profile,
                radius_arcsec=args.radius,
                catalogs=catalogs,
                count_threshold=args.count_threshold,
                min_confidence=args.min_confidence,
                output_format=args.format,
                export_path=args.output,
                targets=targets_data,
                any_export_path=True,
            )

    try:
        meta = asyncio.run(do_dataset())
    except (AstroSearchError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Dataset generated successfully: {meta['export_path']}")
    print(f"Total sources written: {meta['total_sources']} (method: {meta.get('method')})")
    if meta.get("failures"):
        print(f"Catalog failures: {len(meta['failures'])}")
    return 0


def _cmd_catalogs(args: argparse.Namespace) -> int:
    reg = build_registry()
    if args.name:
        try:
            cat = reg.get(args.name)
        except KeyError:
            print(f"Catalog '{args.name}' not found.", file=sys.stderr)
            return 1
        print(json.dumps(cat.as_dict(), indent=2))
        return 0
    print(f"{'Catalog':<28} {'Provider':<16} {'Wavelength':<14} {'Enabled'}")
    print("-" * 70)
    for name, cat in reg.catalogs.items():
        print(f"{name:<28} {cat.provider:<16} {cat.wavelength:<14} {cat.enabled}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    print(f"Benchmarking streaming {args.format.upper()} export with {args.rows} rows...")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / f"bench.{args.format}"
        sample = {
            "catalog": "gaia_dr3",
            "source_id": "gaia-bench",
            "ra": 187.25,
            "dec": 2.05,
            "separation_arcsec": 0.5,
            "confidence": 0.95,
            "match_probability": 0.95,
            "target_probability": 0.95,
            "epoch": 2016.0,
            "positional_error_arcsec": 0.05,
            "physical": {"object_type": "star", "parallax": 12.3},
            "data": {"phot_g_mean_mag": 15.2},
            "metadata": {"wavelength": "optical"},
            "provenance": {"provider": "tap"},
            "links": {},
        }
        start = time.perf_counter()
        with DatasetWriter(out_file, args.format) as writer:
            for _ in range(args.rows):
                writer.write(sample)
        elapsed = max(time.perf_counter() - start, 1e-9)
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"Wrote {args.rows:,} rows in {elapsed:.2f}s ({args.rows/elapsed:,.0f} rows/s)")
        print(f"Output size: {size_mb:.2f} MB ({size_mb/elapsed:.2f} MB/s)")
    return 0


run_handler = cli.run_handler


def main(argv: list[str] | None = None) -> None:
    """The ``astrosearch`` command line (see :func:`cli.main`)."""
    cli.main(argv)


if __name__ == "__main__":
    main()
