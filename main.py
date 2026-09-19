"""Master programmatic facade, command-line interface (CLI), and built-in offline verification suite."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from crossmatch import (
    AdvancedQuery,
    CrossmatchService,
    QueryValidator,
    angular_separation_arcsec,
    match_score,
    match_target,
)
from datasets import DatasetEngine, DatasetWriter
from models import (
    CatalogDefinition,
    CatalogRegistry,
    CatalogSource,
    InvalidCoordinateError,
    Settings,
    Target,
    UnifiedRecord,
    normalize_source_record,
    parse_ipac_records,
    parse_json_records,
    validate_target,
)
from providers import (
    EndpointGuard,
    MASTProvider,
    SesameResolver,
    provider_map,
)

# ---------------------------------------------------------------------------
# Public Programmatic API
# ---------------------------------------------------------------------------


def build_service(
    *,
    settings: Settings | None = None,
    registry_path: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> CrossmatchService:
    """Build a configured CrossmatchService instance."""
    active_settings = settings or Settings()
    registry = CatalogRegistry(registry_path or active_settings.catalog_registry_path)
    providers = provider_map(
        client,
        timeout=active_settings.request_timeout_seconds,
        max_response_bytes=active_settings.max_response_bytes,
    )
    return CrossmatchService(
        registry,
        providers,
        radius_arcsec=active_settings.default_radius_arcsec,
        timeout=active_settings.request_timeout_seconds,
    )


async def crossmatch(
    ra: float | str,
    dec: float | str,
    *,
    radius_arcsec: float | None = None,
    epoch: float | None = None,
    profile: str | None = None,
    settings: Settings | None = None,
) -> UnifiedRecord:
    """Execute a single coordinate crossmatch using a fresh client session."""
    active_settings = settings or Settings()
    async with httpx.AsyncClient(timeout=active_settings.request_timeout_seconds, follow_redirects=True) as client:
        service = build_service(settings=active_settings, client=client)
        return await service.crossmatch(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile)


async def search_object(
    name: str,
    *,
    radius_arcsec: float | None = None,
    profile: str | None = None,
    settings: Settings | None = None,
    resolver: SesameResolver | None = None,
) -> UnifiedRecord:
    """Resolve an astronomical object name via CDS Sesame and execute the crossmatch pipeline."""
    active_settings = settings or Settings()
    async with httpx.AsyncClient(timeout=active_settings.request_timeout_seconds, follow_redirects=True) as client:
        active_resolver = resolver or SesameResolver(client, endpoint=active_settings.resolver_endpoint)
        resolved = await active_resolver.resolve(name)
        target = validate_target(resolved.ra_deg, resolved.dec_deg, epoch=resolved.epoch)
        service = build_service(settings=active_settings, client=client)
        result = await service.crossmatch(
            target.ra,
            target.dec,
            radius_arcsec=radius_arcsec,
            epoch=target.epoch,
            profile=profile,
        )
        result.resolved_object = resolved.as_dict()
        result.provenance["resolver"] = resolved.resolver
        return result


def catalog_definitions(*, settings: Settings | None = None) -> dict[str, Any]:
    """Return dictionary of configured astronomical catalog definitions."""
    active_settings = settings or Settings()
    return CatalogRegistry(active_settings.catalog_registry_path).catalogs


# ---------------------------------------------------------------------------
# Built-In Offline Verification Engine
# ---------------------------------------------------------------------------


def run_verification() -> bool:
    """Run comprehensive offline test and verification suite across all 5 monoliths."""
    print("=" * 70)
    print("AstroSearch Built-In Verification Suite (Offline)")
    print("=" * 70)
    passed = 0
    total = 0

    def check(name: str, test_fn):
        nonlocal passed, total
        total += 1
        print(f"[{total:02d}] Testing {name:.<50} ", end="", flush=True)
        try:
            test_fn()
            print("PASSED")
            passed += 1
        except Exception as exc:
            print(f"FAILED: {exc}")

    # 1. Coordinate Validation & Normalization
    def test_coords():
        t = validate_target(361.0, 10.0)
        assert t.ra == 1.0 and t.dec == 10.0
        try:
            validate_target(10.0, 95.0)
            raise AssertionError("Should have rejected dec > 90")
        except InvalidCoordinateError:
            pass

    check("Coordinate normalization and bounds validation", test_coords)

    # 2. Angular Separation Math
    def test_sep():
        sep = angular_separation_arcsec(Target(0.0, 0.0), Target(0.0, 1.0))
        assert 3590 < sep < 3610

    check("Angular separation calculation (arcseconds)", test_sep)

    # 3. Probabilistic Match Scoring
    def test_scoring():
        score_close = match_score(0.2, positional_error_arcsec=1.0)
        score_far = match_score(5.0, positional_error_arcsec=1.0)
        assert 0.0 <= score_far < score_close <= 1.0

    check("Probabilistic Gaussian match scoring", test_scoring)

    # 4. Normalization of Astronomical Field Aliases
    def test_norm():
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
    def test_ipac():
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

    # 6. Embedded 19-Catalog Registry
    def test_registry():
        reg = CatalogRegistry()
        enabled = reg.enabled_catalogs()
        assert len(enabled) >= 15
        assert "gaia_dr3" in enabled
        assert "twomass_psc" in enabled
        assert "allwise" in enabled
        assert "sdss" in enabled

    check("Embedded 19-catalog registry loading", test_registry)

    # 7. CDS Sesame XML Parsing
    def test_sesame():
        xml_payload = """
        <Sesame>
          <Result>
            <oname>M 87</oname>
            <alias>NGC 4486</alias>
            <alias>Virgo A</alias>
            <jradeg>187.705930</jradeg>
            <jdedeg>12.391123</jdedeg>
            <otyp>G</otyp>
            <z_value>0.00428</z_value>
          </Result>
        </Sesame>
        """
        res = SesameResolver.parse_response("M87", xml_payload)
        assert res.canonical_name == "M 87"
        assert abs(res.ra_deg - 187.705930) < 1e-5
        assert res.object_type == "G"
        assert res.redshift == 0.00428

    check("CDS Sesame XML resolver parser", test_sesame)

    # 8. Advanced Query DSL & Multi-Constraint Filters
    def test_filters():
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
    def test_guard():
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
    def test_dataset_export():
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_source = {
                "catalog": "gaia_dr3",
                "source_id": "gaia-001",
                "ra": 187.25,
                "dec": 2.05,
                "confidence": 0.98,
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
    def test_api_routes():
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

    print("=" * 70)
    print(f"Verification Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    return passed == total


# ---------------------------------------------------------------------------
# Command Line Interface (CLI)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrosearch",
        description="AstroSearch: High-performance astronomical catalog cross-matching and dataset engine.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch the FastAPI REST server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Execute single astronomical search")
    search_parser.add_argument("--ra", type=float, help="Right ascension in degrees [0, 360)")
    search_parser.add_argument("--dec", type=float, help="Declination in degrees [-90, 90]")
    search_parser.add_argument("--name", type=str, help="Astronomical object name to resolve (e.g. M87, Vega)")
    search_parser.add_argument("--radius", type=float, default=3.0, help="Search radius in arcseconds (default: 3.0)")
    search_parser.add_argument("--profile", type=str, help="Catalog profile: optical, infrared, radio, xray, etc.")
    search_parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")

    # Command: dataset
    dataset_parser = subparsers.add_parser("dataset", help="Generate a filtered crossmatched dataset")
    dataset_parser.add_argument("--name", required=True, help="Dataset identification name")
    dataset_parser.add_argument("--profile", required=True, help="Catalog profile (e.g. stellar, optical)")
    dataset_parser.add_argument("--radius", type=float, default=5.0, help="Search radius in arcseconds")
    dataset_parser.add_argument("--targets", required=True, help="Path to JSON file containing array of targets")
    dataset_parser.add_argument("--format", choices=["parquet", "csv", "json", "fits"], default="parquet")
    dataset_parser.add_argument("--output", help="Explicit path to write the dataset export")

    # Command: catalogs
    catalogs_parser = subparsers.add_parser("catalogs", help="List and inspect configured catalog definitions")
    catalogs_parser.add_argument("--name", help="Specific catalog to inspect")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark streaming dataset export performance")
    bench_parser.add_argument("--rows", type=int, default=50000, help="Number of synthetic records to stream")
    bench_parser.add_argument("--format", choices=["parquet", "csv", "json", "fits"], default="parquet")

    # Command: verify
    subparsers.add_parser("verify", help="Run comprehensive offline self-test and verification suite")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print(f"Starting AstroSearch REST API on http://{args.host}:{args.port}")
        uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "search":
        if not args.name and (args.ra is None or args.dec is None):
            print("Error: Specify either --name or both --ra and --dec.")
            sys.exit(1)

        async def do_search():
            if args.name:
                print(f"Resolving '{args.name}' via CDS Sesame...")
                res = await search_object(args.name, radius_arcsec=args.radius, profile=args.profile)
            else:
                res = await crossmatch(args.ra, args.dec, radius_arcsec=args.radius, profile=args.profile)

            if args.format == "json":
                print(json.dumps(res.as_dict(), indent=2))
            else:
                print("\n--- Crossmatch Summary ---")
                print(f"Target: RA={res.target['ra']:.6f}, DEC={res.target['dec']:.6f}")
                print(f"Catalogs Queried: {res.catalogs_queried}")
                print(f"Physical Objects Grouped: {len(res.crossmatch_groups)}")
                for wave, sources in res.counterparts.items():
                    print(f"  [{wave.upper()}] {len(sources)} counterpart(s)")
                if res.failures:
                    print(f"Failures ({len(res.failures)}): {[f['catalog'] for f in res.failures]}")

        asyncio.run(do_search())

    elif args.command == "dataset":
        targets_file = Path(args.targets)
        if not targets_file.exists():
            print(f"Error: Targets file '{args.targets}' not found.")
            sys.exit(1)
        targets_data = json.loads(targets_file.read_text(encoding="utf-8"))

        async def do_dataset():
            engine = DatasetEngine()
            print(f"Building dataset '{args.name}' across {len(targets_data)} targets...")
            meta = await engine.create_dataset(
                name=args.name,
                profile=args.profile,
                radius_arcsec=args.radius,
                output_format=args.format,
                export_path=args.output,
                targets=targets_data,
            )
            print(f"Dataset generated successfully: {meta['export_path']}")
            print(f"Total sources written: {meta['total_sources']}")

        asyncio.run(do_dataset())

    elif args.command == "catalogs":
        reg = CatalogRegistry()
        if args.name:
            try:
                cat = reg.get(args.name)
                print(json.dumps(cat.as_dict(), indent=2))
            except KeyError:
                print(f"Catalog '{args.name}' not found.")
                sys.exit(1)
        else:
            print(f"{'Catalog':<24} {'Provider':<16} {'Wavelength':<14} {'Enabled'}")
            print("-" * 65)
            for name, cat in reg.catalogs.items():
                print(f"{name:<24} {cat.provider:<16} {cat.wavelength:<14} {cat.enabled}")

    elif args.command == "benchmark":
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
            elapsed = time.perf_counter() - start
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"Wrote {args.rows:,} rows in {elapsed:.2f}s ({args.rows/elapsed:,.0f} rows/s)")
            print(f"Output size: {size_mb:.2f} MB ({size_mb/elapsed:.2f} MB/s)")

    elif args.command == "verify":
        success = run_verification()
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
