# AstroSearch

**AstroSearch** is a high-performance Python backend system for cross-matching sky coordinates and astronomical object identities across major public astronomical survey archives (Gaia, SIMBAD, NED, 2MASS, AllWISE, Pan-STARRS, SDSS, FIRST, NVSS, Chandra, XMM, etc.), applying astrophysical filters, and generating streaming datasets in JSON, CSV, Parquet, and FITS formats.

The entire codebase is organized into **6 production-grade monolithic scripts**, providing complete architectural clarity, maximum execution speed, and self-contained operation.

---

## 🏛️ Architecture: The 6 Monolithic Scripts

```
AstroSearch/
├── models.py         # 1. Models, Astrometry, Parsers & Embedded 19-Catalog Registry
├── providers.py      # 2. Archive Adapters (TAP, Gator, MAST, SDSS, HEASARC), Sesame & Caching
├── crossmatch.py     # 3. Query DSL, Proper-Motion Propagation, DSU Grouping & Matching Engine
├── datasets.py       # 4. Streaming Dataset Exports (JSON/CSV/Parquet/FITS), Storage & Jobs
├── api.py            # 5. Production FastAPI REST Service (Auth, Quotas, Metrics, 15 Endpoints)
└── main.py           # 6. Master Programmatic Facade, Unified CLI & Built-in Verification
```

1. **[models.py](models.py)**: Dataclasses (`Target`, `CatalogSource`, `UnifiedRecord`), Astropy spherical coordinate normalization, field normalizers mapping 30+ column aliases, multi-format response parsers (VOTable, IPAC ASCII, CSV, JSON), runtime settings, and the complete embedded 19-catalog registry.
2. **[providers.py](providers.py)**: Async HTTP archive adapters for TAP/ADQL, IRSA Gator, MAST, SDSS, and HEASARC Xamin, CDS Sesame name resolver, `EndpointGuard` rate limiter & circuit breaker, and hybrid in-memory / Redis `CacheManager`.
3. **[crossmatch.py](crossmatch.py)**: `AdvancedQuery` specification, `QueryValidator`, `QueryBuilder`, Astropy proper-motion epoch propagation, probabilistic Gaussian match scoring, adaptive radius density scaling, multi-wavelength Disjoint-Set Union (DSU) counterpart clustering, and `CrossmatchService`.
4. **[datasets.py](datasets.py)**: High-throughput streaming `DatasetWriter` (`json`, `csv`, `parquet`, `fits`), `DatasetEngine` (multi-target execution, deduplication, detection thresholds), `MetadataStore` (SQLite/PostgreSQL), `ObjectStore` (S3/MinIO), and asynchronous worker jobs.
5. **[api.py](api.py)**: Full FastAPI REST API with Pydantic request/response schemas, API key and JWT bearer authentication, sliding-window `RequestQuota`, Prometheus metrics (`/api/v1/monitoring/metrics`), structured JSON logging, and 15+ REST endpoints.
6. **[main.py](main.py)**: High-level Python facade (`crossmatch`, `search_object`, `build_service`), comprehensive unified CLI (`serve`, `search`, `dataset`, `catalogs`, `benchmark`, `verify`), and a built-in offline test suite.

---

## 🚀 Installation

Requires **Python 3.12+**.

```bash
# Clone and enter workspace
git clone https://github.com/your-org/AstroSearch.git
cd AstroSearch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt  # Or: pip install .
```

### Key Dependencies
- `astropy>=6.0.0` (Coordinate frames, astrometric transformations, IPAC tables, FITS)
- `fastapi>=0.110.0` & `uvicorn>=0.29.0` (REST API service)
- `pydantic>=2.7.0` (Data validation and schema contracts)
- `httpx>=0.27.0` (Async HTTP archive client)
- `pyarrow>=15.0.0` (Streaming Parquet export with zstd compression)
- `structlog` & `prometheus-client` (Structured observability and metrics)
- Optional: `redis`, `rq`, `psycopg`, `boto3` (Distributed workers, PostgreSQL, and S3 storage)

---

## 💻 Python Library Quickstart

### 1. Crossmatch by Coordinates
```python
import asyncio
from main import crossmatch

async def main():
    # Crossmatch near Virgo A (M87) with a 5.0 arcsecond radius
    result = await crossmatch(187.705930, 12.391123, radius_arcsec=5.0)
    print(f"Catalogs queried: {result.catalogs_queried}")
    print(f"Physical objects clustered: {len(result.crossmatch_groups)}")
    for wavelength, sources in result.counterparts.items():
        print(f"  [{wavelength.upper()}] {len(sources)} detection(s)")

asyncio.run(main())
```

### 2. Crossmatch by Astronomical Object Name (CDS Sesame)
```python
import asyncio
from main import search_object

async def main():
    # Resolves object name to canonical coordinates, then crossmatches
    result = await search_object("M87", radius_arcsec=5.0, profile="optical")
    print("Resolved Name:", result.resolved_object["canonical_name"])
    print("Coordinates:", result.target["ra"], result.target["dec"])
    print("Matches:", len(result.provenance["matches"]))

asyncio.run(main())
```

### 3. Advanced Query with Physical and Spatial Filters
```python
import asyncio
from crossmatch import AdvancedQuery, CrossmatchService
from main import build_service

async def main():
    service = build_service()
    query = AdvancedQuery.from_dict({
        "ra": 187.705930,
        "dec": 12.391123,
        "radius_arcsec": 15.0,
        "object_types": ["galaxy", "quasar"],
        "min_confidence": 0.8,
        "search_mode": "cone",
        "proper_motion": True,
        "adaptive_radius": True,
    })
    result = await service.crossmatch(187.705930, 12.391123, query=query)
    print("Effective Radius:", result.provenance["effective_radius_arcsec"])

asyncio.run(main())
```

---

## 🛠️ Command-Line Interface (CLI)

`main.py` provides a unified operational command-line interface:

### Start the REST API Server
```bash
python main.py serve --host 127.0.0.1 --port 8000
```
API Documentation will be live at `http://127.0.0.1:8000/api/docs`.

### Search by Sky Position
```bash
python main.py search --ra 187.27792 --dec 2.05239 --radius 3.0 --profile optical
```

### Search by Object Name
```bash
python main.py search --name "M87" --radius 5.0 --format json
```

### Inspect Available Catalogs
```bash
python main.py catalogs
python main.py catalogs --name gaia_dr3
```

### Export Benchmark
```bash
python main.py benchmark --rows 50000 --format parquet
```

### Run Built-in Offline Verification Suite
```bash
python main.py verify
```

### Test Suite
```bash
python -m pytest -q            # offline: replays real archive responses from tests/fixtures (no network)
python -m pytest -q -m live    # live canaries: 3C 273, M87, HD 106785, HD 209458 against every catalog
python tests/fixture_io.py record          # re-record fixtures from the live archives
python scripts/catalog_census.py 10        # sources per catalog within 10" for 3C 273 and M87
```
Live tests skip only when an archive is unreachable (network error, timeout, HTTP 5xx);
query errors, parse failures, or a missing known object fail the run.

### Per-catalog result status
`catalog_results[<catalog>]["status"]` is `success` (rows inside the requested radius),
`empty` (valid query, none inside the radius: e.g. outside the survey footprint) or
`failed` (with `error_type` and the archive's own `message`). Row counts (`row_count` = rows
inside the radius, the nearest `max_rows`; `returned_count` = rows in `sources` after an
AdvancedQuery/REST confidence or type filter; `raw_row_count` = `row_count` + `pad_row_count`
+ `excess_row_count` + `dropped_rows`), elapsed milliseconds, truncation, `warnings` and
citations are recorded per catalog in `provenance["catalog_stats"]` /
`provenance["citations"]`; all warnings are also collected in `provenance["warnings"]`.
`truncated` is true when the archive held more rows than were fetched or more than `max_rows`
lie inside the radius, and a warning then states how much of the radius the kept rows cover.
Every `CatalogSource.positional_error_arcsec` is a 1-sigma circular error in arcsec,
converted from each catalog's native convention (mas, degrees, seconds of time, 95%/90%
ellipses, pixels) as declared by the registry's `pos_error` spec; `CatalogSource.epoch`
is the Julian year of the catalog position (None when unknown, e.g. NED positions whose
`pos_bibcode` is not a known survey). Rows observed at an unrecorded time within a known
span -- Chandra CSC master sources (1999.5-2022.0), NVSS, LoTSS, 1RXS, and NED positions
copied from 2MASS or WISE -- have `epoch` None and `epoch_range` (earliest, latest); a
moving target is matched against its closest approach during that span
(`epoch_propagation` = `target_pm_span`).

### Epochs and proper motion
Pass `epoch` (Julian year of the coordinates) and, when known, `pm_ra_masyr`/`pm_dec_masyr`
(`crossmatch(...)`, `AdvancedQuery`, or the REST `SearchRequest`). Name searches use the
resolver's epoch (J2000 for SIMBAD) and proper motion automatically (extragalactic objects
are treated as stationary). With an epoch, each catalog cone follows the target to that
catalog's epoch (declared by `epoch` or `parameters.epoch_range`): with a proper motion the
cone is re-centred on the target's path; without one it is widened by the largest plausible
motion (`EPOCH_PAD_MAX_PM_ARCSEC_PER_YR`, default 10.5"/yr, capped at `EPOCH_PAD_MAX_ARCSEC`,
default 300", fetching up to `EPOCH_PAD_MAX_ROWS`, default 2000, rows). Rows are compared after
propagation; rows fetched only because of the widening are returned as `pad_sources`, never
counted. When the target's proper motion is not given it is adopted from the nearest matching
row that has one (e.g. Gaia), recorded in `provenance["target_proper_motion"]` (`source`:
`input`, `resolver`, `adopted`, or `extragalactic` when the nearest identified row is a
galaxy/QSO, which is then treated as stationary), so catalogs without proper motions (2MASS,
AllWISE, PS1, CSC) can be matched too. Every fetched row -- in radius beyond `max_rows`, or
pad -- is kept until this final re-split, so adopting the motion can never lose a counterpart.
Without an epoch, positions are compared as given and cones are exact.

Adoption is refused (with a provenance warning) when a candidate with a different motion lies
within 2 x the nearest separation + 0.5" (crowded fields such as the S-stars around Sgr A*).
Rows without their own proper motion in catalogs that publish them (Gaia 2-parameter
solutions, SIMBAD positions without pm) are compared at their catalog positions
(`epoch_propagation` = `stationary`, with `target_pm_separation_arcsec` for reference); only
rows of catalogs without proper motions are moved with the target's motion. Pass
`parallax_mas` (name searches take it from the resolver; otherwise a significant parallax is
adopted with the motion) to remove the annual parallax from single-epoch positions (2MASS,
SDSS, FIRST, VLASS, 2RXS: `target_pm_parallax`; Proxima's 2MASS row goes from 0.71" to
0.05"); rows where it cannot be removed get it as extra uncertainty in their confidence.
5XMM `time`/`end_time` are the span of the detection stack, so 5XMM rows spanning more than
0.1 yr have an `epoch_range`, not a midpoint epoch. SIMBAD rows without a proper motion keep
the epoch of their original measurement (from `coo_bibcode`, else the span 1990-2025).

### Resilience settings
`PROVIDER_REQUESTS_PER_SECOND`, `PROVIDER_FAILURE_THRESHOLD`, `PROVIDER_RECOVERY_SECONDS` and
`PROVIDER_PROBE_TIMEOUT_SECONDS` tune the per-endpoint rate limiter and circuit breaker (a
cancelled or timed-out request always counts as a failure, and a stuck recovery probe expires).
HTTP 429/408 raise `RateLimitedError` (a `CatalogUnavailableError`, so fallbacks apply).
`CATALOG_REGISTRY_STRICT=true` makes the API/CLI refuse to start with an invalid registry
(problems are always logged); an unparseable registry file raises `RegistryError`.
`API_SEARCH_CACHE_TTL_SECONDS` (default 300, 0 disables) caches only searches in which every
catalog answered.
Catalog timeouts come from the registry (60-90 s); `CATALOG_TIMEOUT_CAP_SECONDS` (or an
explicitly set `REQUEST_TIMEOUT_SECONDS`) caps them. A fallback archive gets the rest of the
catalog's budget (at least min(timeout, 20 s)); when both fail, the error names both causes
and `catalog_stats[...]["fallback"]` records the primary's error. The in-process response
cache is a bounded LRU (`PROVIDER_CACHE_MAX_ENTRIES`, default 512;
`PROVIDER_CACHE_MAX_BYTES`, default 64 MB). Per-catalog outcomes are exported as
`astrosearch_catalog_queries_total{catalog,status}` (`status` gets `+fallback` when a fallback
was used). An unknown profile is rejected (HTTP 422 / CLI error) and an unresolvable object
name returns HTTP 404.

---

## 🌐 HTTP REST API

The FastAPI service in `api.py` exposes the full REST API:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status and timestamp |
| `GET` | `/api/v1/catalogs` | List all 19 configured astronomical catalogs |
| `GET` | `/api/v1/catalogs/{name}` | Retrieve catalog parameters and table info |
| `POST` | `/api/v1/search` | Single coordinate or object name crossmatch |
| `POST` | `/api/v1/search/batch` | Concurrency-limited batch crossmatch |
| `POST` | `/api/v1/datasets/create` | Submit asynchronous dataset creation (HTTP 202) |
| `GET` | `/api/v1/datasets` | List all created datasets |
| `GET` | `/api/v1/datasets/{id}` | Inspect dataset generation status & metadata |
| `GET` | `/api/v1/datasets/{id}/export` | Download dataset (JSON, CSV, Parquet, FITS) |
| `DELETE` | `/api/v1/datasets/{id}` | Delete dataset and local/S3 artifacts |
| `GET` | `/api/v1/queries` | List saved queries |
| `POST` | `/api/v1/queries` | Save query definition |
| `DELETE` | `/api/v1/queries/{id}` | Delete saved query |
| `GET` | `/api/v1/stats` | System metrics (datasets, exported sources) |
| `GET` | `/api/v1/monitoring` | Provider circuit breaker states |
| `GET` | `/api/v1/monitoring/metrics` | Prometheus scrape endpoint |

For detailed API payload specifications, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## 📄 License
MIT License.
