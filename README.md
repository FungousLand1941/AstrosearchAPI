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
