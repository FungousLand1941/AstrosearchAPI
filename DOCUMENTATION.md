# AstroSearch Technical Documentation

This is the reference for AstroSearch 0.3: its architecture, the science behind each feature,
every HTTP endpoint and CLI command, the configuration, and how it is tested. The README has
the short introduction.

## Contents

1. [Architecture](#1-architecture)
2. [Catalog registry](#2-catalog-registry)
3. [Crossmatch engine](#3-crossmatch-engine)
4. [Feature modules](#4-feature-modules)
5. [Datasets](#5-datasets)
6. [REST API reference](#6-rest-api-reference)
7. [Command-line reference](#7-command-line-reference)
8. [Security, limits and observability](#8-security-limits-and-observability)
9. [Configuration](#9-configuration)
10. [Known limitations](#10-known-limitations)
11. [Testing](#11-testing)
12. [References](#12-references)

---

## 1. Architecture

### Modules

| Module | Role |
|---|---|
| `models.py` | Dataclasses (`Target`, `CatalogSource`, `UnifiedRecord`, `ResolvedObject`), coordinate validation and propagation, response parsers (VOTable, IPAC, CSV, JSON), positional-error conventions, `Settings`, the embedded catalog registry. |
| `providers.py` | Archive adapters (`TapProvider`, `IrsaGatorProvider`, `MASTProvider`, `SDSSProvider`, HEASARC), the CDS Sesame resolver, `EndpointGuard` (rate limiter + circuit breaker), `CacheManager` (in-process LRU + optional Redis). |
| `astrometry.py` | Bayesian N-way association (pure Python/numpy): Bayes factors, priors, target association, Monte-Carlo calibration. |
| `crossmatch.py` | `AdvancedQuery`, query planning and execution, epoch propagation, `CrossmatchService` (`crossmatch`, `crossmatch_stream`, `crossmatch_many`). |
| `streaming.py` | Server-sent-event search stream. |
| `batch.py` | Batch crossmatch through TAP uploads, CDS XMatch and paced cones. |
| `skycache.py` | Local HATS/Parquet mirrors of archive regions with MOC coverage. |
| `vizier.py` | VizieR discovery, table description and registration as catalogs. |
| `sed.py` | SEDs, classification and redshift. |
| `timedomain.py` | Light curves (ZTF, NEOWISE, Gaia, TESS), variability, periods, solar-system checks. |
| `imaging.py` | HiPS cutouts through CDS hips2fits, multi-survey stacks, the web UI mount. |
| `ai.py` | Claude: natural-language query compilation and cited object explanations. |
| `provenance.py` | Reproducibility manifests, replay diffs, citations. |
| `vo_server.py` | IVOA Simple Cone Search, TAP/ADQL, UWS, VOSI. |
| `alerts.py` | ALeRCE / Fink alert ingestion and enrichment. |
| `datasets.py` | Dataset generation, exports, metadata store, object store, worker jobs. |
| `api.py` | The FastAPI application: core endpoints, middleware, every feature router, the web UI. |
| `main.py` | Service factories, the Python facade, the `astrosearch` CLI and `verify`. |
| `web/` | The single-page UI (`index.html`, `app.js`, `styles.css`). |

Every feature module exposes `router` (a FastAPI `APIRouter`) and `register_cli(subparsers)`
(its subcommands set `handler`, a function returning the exit status). `api.py` includes
every router; `main.py` registers every CLI.

### Request flow

```
client (browser UI, curl, TOPCAT, notebooks)
   |
   v
UIStaticMiddleware (serves /, /index.html, /app.js, /styles.css)
CORS middleware
request_metrics middleware: request id, authentication, quota, body limit, strict JSON, metrics
   |
   +-- core endpoints (api.py) ---------+
   +-- feature routers ----------------+--> app.state.service (MeteredCrossmatchService)
                                        |      registry: embedded + user VizieR catalogs
                                        |      providers: MirrorFirstProvider(archive adapter, sky cache)
                                        |           -> archives over one shared httpx.AsyncClient
                                        +--> module-specific upstreams (Sesame, SVO, IRSA, MAST,
                                             hips2fits, SkyBoT, ALeRCE, Fink, Anthropic, ...)
```

### Shared application state

The API lifespan builds one of each and publishes them on `app.state`, so every router uses the
same objects:

| Attribute | Object |
|---|---|
| `settings` | `models.Settings` |
| `client` | one `httpx.AsyncClient` for every upstream request |
| `registry` | `main.build_registry()`: embedded catalogs + the user registry YAML (`vizier.load_registry`) |
| `providers` | `main.build_providers()`: archive adapters on `client`, wrapped in `MirrorFirstProvider` |
| `service` | `MeteredCrossmatchService` (a `CrossmatchService` that records Prometheus catalog metrics) over `registry` and `providers` |
| `skycache` | the `SkyCache` store the providers read (`SKYCACHE_PATH`) |
| `engine`, `metadata` | `DatasetEngine` over `service`/`registry`, and its `MetadataStore` |
| `sed_filters` | one `sed.FilterCatalog` (SVO filter cache) |

`registry` is the service's own object, so a catalog registered with
`POST /api/v1/vizier/register` is queried by searches, datasets and batches at once. Routers
create further objects lazily (the alert store, the UWS job manager, the Anthropic client);
on shutdown the lifespan closes the Anthropic client (`ai.aclose_state_clients`), cancels UWS
jobs and closes their client (`vo_server.aclose_vo_state`), closes the alerts router's client,
and removes every object.

### Service factories (`main.py`)

- `build_registry(settings=None, registry_path=None)` - `vizier.load_registry(path)` plus
  `startup_check()` (problems are logged; `CATALOG_REGISTRY_STRICT=true` raises).
- `build_providers(settings=None, client=None, store=None)` - `providers.provider_map(...)` with
  the request timeout and response-size cap; unless `SKYCACHE_ENABLED=false`, each adapter is
  wrapped in `MirrorFirstProvider`.
- `build_service(settings=None, registry_path=None, client=None, registry=None, providers=None,
  service_class=CrossmatchService)` - the service; feature modules call it when they run
  outside the API.
- `crossmatch(ra, dec, ...)`, `search_object(name, ...)` - one-call searches on a fresh client.
- `api_search(service, fields, resolver)` and `crossmatch_resolved(service, resolved, ...)` - the
  searches of `POST /api/v1/search` and `astrosearch search --name`; provenance manifests and
  replays of such searches run the same functions, so their records hash identically.

`MirrorFirstProvider` answers a cone from the sky cache when the catalog is mirrored there
(falling back to the archive when the cone is not fully covered, the mirror is stale or its
definition changed) and sends every other catalog straight to the archive. It reads through to
the archive adapter's attributes (`client`, `guards`, `cache`), so batch pacing, provenance
replays and the monitoring endpoint see the same endpoint guards.

---

## 2. Catalog registry

### Built-in catalogs

| Name | Provider | Regime | Table | Archive host | Enabled | Profiles |
|---|---|---|---|---|---|---|
| `gaia_dr3` | tap | optical | `gaiadr3.gaia_source` | gea.esac.esa.int | yes | full, optical, stellar |
| `simbad` | tap | multi | `basic` (+ `allfluxes`) | simbad.cds.unistra.fr | yes | full, optical, stellar, identity |
| `ned` | tap | extragalactic | `NEDTAP.objdir` | ned.ipac.caltech.edu | yes | full, optical, extragalactic, identity |
| `exoplanet_archive` | tap | exoplanet | `pscomppars` | exoplanetarchive.ipac.caltech.edu | yes | full, exoplanet, stellar |
| `twomass_psc` | tap (IRSA Gator fallback) | infrared | `fp_psc` | irsa.ipac.caltech.edu | yes | full, infrared, stellar |
| `allwise` | tap (IRSA Gator fallback) | infrared | `allwise_p3as_psd` | irsa.ipac.caltech.edu | yes | full, infrared, stellar |
| `panstarrs_dr2` | mast | optical | mean object catalog | catalogs.mast.stsci.edu | yes | full, optical, extragalactic |
| `sdss` | sdss | optical | `PhotoPrimary` | skyserver.sdss.org | yes | full, optical, extragalactic, spectroscopy |
| `first` | tap | radio | `first` | heasarc.gsfc.nasa.gov | yes | full, radio, extragalactic |
| `nvss` | tap | radio | `nvss` | heasarc.gsfc.nasa.gov | yes | full, radio, extragalactic |
| `vlass` | tap | radio | `J/ApJ/914/42/table5` | tapvizier.cds.unistra.fr | yes | full, radio, extragalactic |
| `lotss` | tap | radio | `J/A+A/707/A198/lotssdr3` | tapvizier.cds.unistra.fr | yes | full, radio, extragalactic |
| `rosat` | tap | xray | `rass2rxs` | heasarc.gsfc.nasa.gov | yes | full, xray, high-energy |
| `chandra` | tap | xray | `csc` | heasarc.gsfc.nasa.gov | yes | full, xray, high-energy |
| `xmm` | tap | xray | `xmmssc` | heasarc.gsfc.nasa.gov | yes | full, xray, high-energy |
| `vizier_2mass_reference` | tap | infrared | `II/246/out` | tapvizier.cds.unistra.fr | no | full, infrared, stellar |
| `lotss_dr2` | tap | radio | `J/A+A/659/A1/catalog` | tapvizier.cds.unistra.fr | no | full, radio, extragalactic |
| `rosat_bsc` | tap | xray | `rassbsc` | heasarc.gsfc.nasa.gov | no | full, xray, high-energy |

Each definition declares its columns, row limit, timeout, positional-error convention
(`pos_error`: units, 1-sigma / 90% / 95% radius or ellipse), epoch (a fixed Julian year, a
per-row column, or an `epoch_range` span), fallback archive and citation.
`GET /api/v1/catalogs` returns every definition.

### User catalogs

`CATALOG_REGISTRY_PATH` (default `~/.astrosearch/catalogs.yaml`) is the user registry. A file
written by `vizier add` / `POST /api/v1/vizier/register` starts with `extends: embedded`: the
current embedded catalogs are merged under its entries, so built-ins are never lost or
shadowed by stale copies. A file without `extends` is read as a complete registry. The API and
the CLI load it at startup (`main.build_registry`).

---

## 3. Crossmatch engine

### Targets, epochs and proper motion

- RA is normalised to [0, 360) degrees and Dec checked against [-90, 90]; coordinates may be
  given as numbers or as the text typed (decimal or sexagesimal). The rounding of text
  coordinates (`187.278` is known to 0.001 deg) widens the target's positional uncertainty.
- `epoch` is the Julian year of the target position. With it, each catalog cone follows the
  target to that catalog's epoch: with a proper motion the cone is re-centred on the target's
  path; without one it is widened by the largest plausible motion
  (`EPOCH_PAD_MAX_PM_ARCSEC_PER_YR`, default 10.5"/yr, at most `EPOCH_PAD_MAX_ARCSEC`, default
  300", fetching up to `EPOCH_PAD_MAX_ROWS`, default 2000, rows). Rows fetched only because of
  the widening are returned as `pad_sources`, never counted.
- Name searches use the resolver's answer (`crossmatch.resolved_search_target`): the SIMBAD
  position at its epoch (J2000), its proper motion and parallax, their errors as the target
  uncertainty, and the resolved object itself, whose catalogue row becomes the target's
  identity. A requested epoch moves the resolved position there with the requested (else the
  resolver's) proper motion; an object without known motion cannot be moved (HTTP 422).
  Extragalactic objects are stationary.
- Without a given proper motion, one is adopted from the nearest matching row that has one
  (e.g. Gaia) unless a candidate with a different motion lies within 2 x the nearest separation
  + 0.5" (crowded fields); `provenance.target_proper_motion.source` is `input`, `resolver`,
  `adopted` or `extragalactic`. A significant parallax (>= 5 sigma) is adopted with it and
  removes the annual parallax from single-epoch positions (2MASS, SDSS, FIRST, VLASS, 2RXS).
- Without an epoch, positions are compared as given; the target's epoch is taken as unknown
  between J2000 and J2016 (`astrometry.UNDATED_TARGET_EPOCHS`).
- Rows observed at an unrecorded time within a known span (Chandra CSC master sources, NVSS,
  LoTSS, 1RXS, NED positions copied from 2MASS or WISE, 5XMM detection stacks) carry
  `epoch_range`; a moving target is matched against its closest approach during that span.
- Every row's positional error is converted to a 1-sigma circular (or elliptical) error in
  arcsec from its catalog's native convention, then grown by its proper-motion uncertainty
  over the epoch difference (Gaia proper-motion errors are estimated as 1.4 x the position
  error per year, the DR3 ratio).

### Bayesian association

Every row in the cone gets a posterior probability of being the target's counterpart, and all
rows are partitioned into physical objects (`astrometry.py`):

- **Bayes factor** (Budavari & Szalay 2008). For n detections with covariances C_i,
  B = 2^(n-1) prod_i |W_i|^(1/2) / |W|^(1/2) exp(-chi2/2), with W_i = C_i^-1, W = sum W_i and
  chi2 about the precision-weighted mean; error ellipses use the elliptical generalisation of
  Pineau et al. (2017). Covariances are mapped to a common tangent plane.
- **Heavy tails.** Positions are a mixture (1 - eps) N(C) + eps N(kappa^2 C) with eps = 0.06,
  kappa = 5, measured on 1929 isolated Gaia-CRF3 quasars against AllWISE, so a genuine
  counterpart 5-10 sigma off keeps a small, non-negligible probability.
- **Target association** (NWAY, Salvato et al. 2018). The target is the primary catalogue;
  each association (per catalogue: no counterpart or one candidate) has weight
  W = B(target + a) prod_k [c_k / (1 - c_k)] / N_k x L(C \ a) / L(C), where N_k is the
  catalogue's local source density expressed as a whole-sky count and c_k the prior that the
  target has a counterpart there. L is the partition function of the other rows as field
  objects that may be seen by several catalogues together (a neighbouring star appears in Gaia,
  2MASS and WISE at once), which removes the independent-field assumption of plain NWAY.
- **Priors.** c_k depends on the target class (star, extragalactic, extended, unknown)
  inferred from the resolver's identity, identity rows and counterparts: a star's radio or
  X-ray counterpart is rare, a quasar's is not. The row of an identity catalogue (SIMBAD, NED)
  that is the target - the resolved name's row, or a row within 1 mas of the searched
  position - gets identity prior odds. An extended target (cluster, nebula, remnant) gets the
  scatter of its catalogued centres added to its position, and compact rows inside it the
  point-source prior.
- **Densities** come from published sky counts and density maps; in crowded fields (the Harris
  globular clusters and the LMC, SMC, M31 and M33 cores) and when a small cone is far denser
  than the map predicts, a density probe re-queries the catalogue over a wider cone.
- **Search.** Associations are enumerated exactly up to 20,000 hypotheses, else by beam search
  (`provenance.association.exact` says which).
- `astrosearch xmatch-calibrate` checks completeness, purity and calibration of the posteriors
  on simulated fields.

Results: `Match.confidence` / `provenance.matches[].confidence` is the target posterior;
`crossmatch_groups` are the most probable partition (target group first, `contains_target`,
`match_flag` best/secondary, `match_probability`, `p_any`, `log10_bayes_factor`, alternatives);
members carry `match_probability`, `target_probability`, `coincident_with` (duplicate listings of
one object share a representative) and `position_at_epoch`. `provenance.association` records the
configuration, priors, densities, target class, identity rows and notes.

### AdvancedQuery filters

`AdvancedQuery` (the REST `SearchRequest`, datasets, AI-compiled queries) filters the rows after
association:

| Field | Meaning |
|---|---|
| `min_confidence` | minimum target posterior (REST default 0.5) |
| `object_types`, `spectral_types`, `morphology` | row classifications (`star`, `galaxy`, `quasar`, `nebula`, `star_cluster` or raw codes) |
| `search_mode` | `cone`; `shell` (with `min_radius_arcsec`); `cylinder` (`min_distance_pc`/`max_distance_pc` from parallax or `distance_pc`) |
| `spatial_constraints` | `radius_zones` ([{min_arcsec, max_arcsec}]) and `exclude_polygons` ([[ra, dec], ...], RA wrap handled) |
| `time_period` | observation-year window (`start_year`, `end_year`) |
| `adaptive_radius` | shrink the radius to max(1", 1.5 x median separation of the five nearest rows) |
| `max_results` | rows per catalog |
| `catalogs`, `profiles` | catalogs to query |

### Per-catalog results

`catalog_results[<catalog>].status` is `success` (rows inside the radius), `empty` (valid query,
no row inside the radius) or `failed` (with `error_type` and the archive's message; also listed
in `failures`). `provenance.catalog_stats` holds per catalog `row_count` (rows inside the radius,
the nearest `max_rows`), `returned_count` (after the query's filters), `raw_row_count`,
`elapsed_ms`, `truncated`, `warnings`, the fallback used, `source_density_deg2`, and
`provenance.citations` the citation of each catalog. A catalog answered from the sky cache
carries `meta.skycache` and each of its sources `provenance.skycache`.

### Resilience

Each endpoint has an `EndpointGuard`: a rate limiter (`PROVIDER_REQUESTS_PER_SECOND`) and a
circuit breaker (`PROVIDER_FAILURE_THRESHOLD` consecutive failures open it for
`PROVIDER_RECOVERY_SECONDS`; a stuck probe expires after `PROVIDER_PROBE_TIMEOUT_SECONDS`).
HTTP 429/408 raise `RateLimitedError`. Transient failures are retried; a fallback archive
(IRSA Gator for 2MASS and AllWISE) gets the rest of the catalog's time budget. Catalog timeouts
come from the registry (60-90 s) and are capped by `CATALOG_TIMEOUT_CAP_SECONDS` (or an
explicitly set `REQUEST_TIMEOUT_SECONDS`). Successful responses are cached in-process (bounded
LRU) and, with `REDIS_URL`, in Redis; Redis calls from async code run in worker threads.

---

## 4. Feature modules

### 4.1 Streaming (`streaming.py`)

`GET /api/v1/search/stream` runs `CrossmatchService.crossmatch_stream` and sends server-sent
events (WHATWG framing: `event:`, `id:`, one JSON `data:` line, a blank line):
`start` (target, planned catalogs, resolved object), one `catalog` per archive in completion
order (status, count, elapsed_ms, rows), one `group` per object of the final record, then `done`
with the full record. `: keepalive` comments are sent every `SSE_KEEPALIVE_SECONDS` (15). Input
errors are answered before the stream starts (422; 404 unknown name; 503 resolver down); a
later failure is an `error` event with `status` and `detail`. A client disconnect cancels the
archive queries.

### 4.2 Batch crossmatch (`batch.py`)

Up to `BATCH_MAX_TARGETS` (100,000) targets against up to `BATCH_MAX_CATALOGS` (20) catalogs,
with one request per catalog per chunk:

- **upload**: IVOA TAP 1.1 table upload joined server-side with ADQL `CONTAINS` per target cone
  (SIMBAD, VizieR, HEASARC, IRSA);
- **xmatch**: the CDS XMatch service against any VizieR table (`gaia_dr3` through
  `vizier:I/355/gaiadr3`, and any `vizier:<table>` catalog), at most 180" per pair;
- **cone**: paced per-target cone searches for archives without uploads (NED, Exoplanet
  Archive, Pan-STARRS, SDSS), at most `BATCH_MAX_CONE_TARGETS` (5000) targets.

Every target has its own epoch-aware cone (identical to the single-object search) and its rows
go through `CrossmatchService.finalize`, so `confidence` is the same posterior a single search
gives. Answers that overflow (`MAXREC`, `QUERY_STATUS=OVERFLOW`, `BATCH_MAX_RESPONSE_BYTES`)
split their chunk. Failed (target, catalog) pairs are reported (`status: failed` rows,
`X-Batch-Failed-Targets`), never confused with "no counterpart". CPU work runs in worker-thread
slices (`BATCH_CPU_SLICE_SECONDS`) so the event loop keeps serving.

### 4.3 Sky cache (`skycache.py`)

`mirror_region` downloads every row of a catalog in a cone or a set of HEALPix pixels through
the same adapters (primary archive only), deduplicates by position and stores a HATS catalog
(HEALPix-partitioned Parquet, readable by `lsdb`/`hats`) with an IVOA MOC 2.0 coverage map
(`skycache.json` and `coverage_moc.fits`). Only rows inside the completely fetched coverage are
stored; each covered cell holds one archive snapshot, so a refresh never duplicates a source.
`LocalProvider` answers a cone only when the cone (after the same epoch widening) lies inside
the coverage and the catalog definition is unchanged, returning `CatalogSource` objects identical
to the archive's; otherwise `CoverageError` sends the query to the archive. Mirrors are limited
to `SKYCACHE_MAX_RADIUS_DEG` (1 deg) and `SKYCACHE_MAX_QUERIES` (256) per call; tiles hold
`SKYCACHE_TILE_ROWS` (10,000) rows before they split. `SKYCACHE_MAX_AGE_DAYS` makes older data
count as not covered.

### 4.4 VizieR catalogs (`vizier.py`)

Keyword/UCD/wavelength search over VizieR's ASU metadata and TAP_SCHEMA (optionally the IVOA
RegTAP registry), table description (columns with units, UCDs and datatypes, row count,
bibcode/DOI, frame, position/identifier/error/epoch columns), and registration: a registry entry
for the VizieR TAP service with the positional-error convention read from the column
descriptions (e.g. 2SXPS `Err90`: 90% Rayleigh radius, sigma = r90 / 2.146). Choices the metadata
does not settle are recorded as assumptions and can be overridden (`overrides`: `pos_error`,
`epoch`, `systematic_arcsec`, ...). Entries are written under an inter-process lock.
`VIZIER_ASU_URL` can point at a mirror (e.g. `http://vizier.nao.ac.jp/viz-bin/votable`).

### 4.5 SED, classification and redshift (`sed.py`)

The crossmatch (or a given record) gives one row per catalog near the target; every photometric
measurement (Gaia G/BP/RP, 2MASS, AllWISE, Pan-STARRS1, SDSS, GALEX, SIMBAD V, FIRST, NVSS,
VLASS, LoTSS, ROSAT, Chandra, XMM-Newton) is converted to a flux density in Jy with errors and
nu F_nu. Filter wavelengths and Vega zero points come from the SVO Filter Profile Service
(validated, cached, with embedded fallbacks); AB magnitudes use 3631 Jy (Oke & Gunn 1983); SDSS
asinh magnitudes follow Lupton et al. (1999); X-ray band fluxes assume a photon index 2 power
law. Points flagged by their catalogs (saturation, quality flags, BP/RP excess) carry
`quality_warning`. Redshifts come from SDSS SpecObj/Photoz, NED and SIMBAD with their technique;
the classification (star / qso / galaxy / agn) is a transparent weighted-evidence heuristic
(`classification.evidence`), not a calibrated probability.

### 4.6 Time domain (`timedomain.py`)

Light curves from ZTF (IRSA; Masci et al. 2019), NEOWISE-R (IRSA), Gaia DR3 epoch photometry
(DataLink) and TESS SPOC 2-min (MAST), on one BMJD_TDB axis (Eastman et al. 2010). A target
epoch, proper motion and parallax (given, or from name resolution) move each survey's cone to
its epoch. Variability metrics follow Sokolovsky et al. (2017) (chi2, robust amplitude, von
Neumann eta, Stetson J, excess variance) with per-survey error models calibrated on constant
stars; periods use the generalised Lomb-Scargle periodogram (Zechmeister & Kuerster 2009;
VanderPlas 2018) with Baluev (2008) false-alarm probabilities, a multi-band periodogram
(VanderPlas & Ivezic 2015), a period-doubling test and multi-harmonic refinement with
Montgomery & O'Donoghue (1999) errors. `GET /api/v1/solar-system` lists known bodies in a cone
from IMCCE SkyBoT (Berthier et al. 2006). Results are cached (`TIMEDOMAIN_CACHE_TTL_SECONDS`,
key includes `TIMEDOMAIN_CACHE_VERSION`).

### 4.7 Imaging and the web UI (`imaging.py`, `web/`)

Cutouts are rendered by CDS hips2fits from HiPS surveys (Fernique et al. 2015) with failover
between its two endpoints; blank images are checked against the survey's MOC before they are
reported or cached as "no data" (`X-Cutout-Blank`). `GET /api/v1/cutouts/stack` plans a radio to
X-ray panel stack for the surveys whose footprint covers the target, each panel centred on the
target's position at the survey's epoch. FITS pixel units and calibration status come from the
HiPS registry. The web UI is served at `/` (exact file routes that never shadow `/api` or `/vo`);
it searches through the SSE stream and loads the SED, light curves (with the target's epoch,
proper motion and parallax), cutouts, solar-system check, citations and AI explanation panels.

### 4.8 IVOA services (`vo_server.py`)

- Simple Cone Search 1.03: `GET /vo/scs?RA=&DEC=&SR=` (degrees), VOTable of crossmatched rows.
- TAP 1.1 with an in-process ADQL 2.1 subset (single-table SELECT, TOP, WHERE with geometry
  CONTAINS/DISTANCE, ORDER BY, OFFSET, aggregates; no joins/GROUP BY/uploads) over
  `astrosearch.matches` (rows computed live; every query must constrain position),
  `astrosearch.catalogs` and `TAP_SCHEMA`; sync and UWS 1.1 async jobs.
- VOSI availability, capabilities (TAPRegExt), tables; DALI examples.
Result sizes are bounded (`VO_MAX_RESULT_BYTES`, `VO_TAP_HARD_MAXREC`); incomplete cones return
`QUERY_STATUS=OVERFLOW`. Clients: TOPCAT, Aladin, pyvo, astroquery.

### 4.9 Alerts (`alerts.py`)

Polls ALeRCE (ZTF), Fink/ZTF and Fink/LSST with complete windowed ingestion (`limit + 1`
requests detect truncation; unfetched windows become a backlog), stores alerts in the metadata
database (`ALERTS_DATABASE_URL` overrides), and enriches each alert with a crossmatch (Gaia DR3,
SIMBAD, NED within 2") and a host-galaxy search (60"), with host distances from Cosmicflows-4
(Tully et al. 2023) or the Planck 2018 Hubble flow. `POST /api/v1/alerts/poll` returns after
ingest and crossmatches in a background task (`crossmatch_mode=background`; follow
`crossmatch_status`), or synchronously with `crossmatch_mode=wait` (limit <= 25).

### 4.10 Provenance and citations (`provenance.py`)

`build_manifest` records the software version and commit, query time, input target and options,
the exact request sent to each catalog, data releases, row counts, a `query_hash` (what was
asked) and a `content_hash` (the scientific payload, canonical JSON following RFC 8785 number
formatting, volatile fields left out). `replay_manifest` re-runs the query and diffs the science
(added/removed/moved/changed rows, status and release changes); `identical` means equal content
hashes and every catalog compared. `citations_for` returns acknowledgements and BibTeX for the
catalogs and services used (bibcodes verified against NASA ADS).

### 4.11 AI (`ai.py`)

`compile_query` has Claude translate a request ("quasars near M87 with radio emission") into a
validated `AdvancedQuery`, a plan and optionally one ADQL query, through a strict tool schema
whose enums come from the live registry; every submission is re-validated and errors are fed
back for up to two corrections. Coordinates never come from the model: names are resolved with
Sesame and typed coordinates parsed with astropy. `explain_object` gathers SIMBAD facts,
bibliography and a crossmatch summary, and has Claude write an explanation that may only use
those facts, with `[n]` citations mapped to bibcodes; derived distances (Planck18 Hubble flow,
parallax inversion) are computed in Python. Credentials: `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`); without them the endpoints answer 503 and the CLI exits 3.

---

## 5. Datasets

`POST /api/v1/datasets/create` (or `astrosearch dataset`) crossmatches a list of targets and
streams the filtered rows to JSON, CSV, Parquet (zstd, 1000-row batches) or FITS.

- **Batch mode.** Lists of at least `DATASET_BATCH_MIN_TARGETS` (default 50; 0 disables) targets
  are fetched with the batch engine (one upload/XMatch request per catalog and chunk) and
  associated per target by `CrossmatchService.finalize`; the result is identical to per-target
  searches. Targets inside a crowded stellar system (Harris globular clusters, LMC/SMC/M31/M33
  cores) are searched one by one, because only a single-object search runs the density probe
  those fields need. Targets whose batch queries failed are searched one by one; when the batch
  cannot run (radius beyond 180", too many cone-only targets or catalogs), every target is. The
  metadata records `method` (`batch`, `per-target` or `batch+per-target`) and `batch`
  (strategies, request count, fallback targets, crowded targets and regions, or the error).
- **Filtering.** Each Bayesian group of a target's cone contributes its members with
  `confidence >= min_confidence` that pass the query filters and `filters`
  (`min_<field>`/`max_<field>`/exact values), when at least `count_threshold` of them do. A row
  found for several targets is written once.
- **Columns.** `catalog`, `source_id`, JSON-encoded `physical`, `data`, `metadata`,
  `provenance`, `links`; `group_id`, `match_flag`, `coincident_with`; `ra`, `dec`,
  `separation_arcsec`, `confidence` (= `target_probability`), `epoch`,
  `positional_error_arcsec`, `match_probability`, `target_probability`,
  `group_match_probability`, `p_any`, `target_ra`, `target_dec`; `target_index`;
  `contains_target`. Probabilities the association does not define are null (NaN in FITS).
- **Jobs.** With `REDIS_URL` jobs go to a Redis RQ queue (`rq worker datasets`); otherwise they
  run as background tasks of the API process with the API's service and registry.
- **Storage.** Metadata in SQLite (`DATASET_STORAGE_PATH/metadata.sqlite3`) or PostgreSQL
  (`DATABASE_URL`); exports on disk or in S3/MinIO (`S3_BUCKET`, `S3_ENDPOINT_URL`).

---

## 6. REST API reference

Interactive documentation: `/api/docs` (Swagger) and `/api/redoc`; the OpenAPI document is
`/api/openapi.json`. Unless noted, errors are JSON `{"detail": ...}`: 401 unauthenticated, 413
body too large, 422 invalid input (including NaN/Infinity numbers in JSON bodies), 429 quota
exceeded, 502 upstream failure, 503 unavailable dependency.

### Core (`api.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness (no authentication, no quota). |
| GET | `/api/v1/catalogs` | Every catalog definition (embedded + registered). |
| GET | `/api/v1/catalogs/{catalog_name}` | One definition; 404 if unknown. |
| POST | `/api/v1/search` | Crossmatch by `ra`/`dec` or `name` (`SearchRequest`: radius_arcsec, profile, catalogs, epoch, pm_ra_masyr, pm_dec_masyr, parallax_mas, min_confidence, filters). 404 unresolvable name, 422 invalid input, 502 search failure. |
| POST | `/api/v1/search/batch` | Up to `API_MAX_BATCH_SIZE` searches concurrently (`max_concurrent`); errors per item. |
| POST | `/api/v1/datasets/create` | Submit a dataset job: 202 with `Location`; 413 over `API_MAX_TARGETS`; 422 invalid. |
| GET | `/api/v1/datasets` | Datasets with their status. |
| GET | `/api/v1/datasets/{dataset_name}` | Metadata and status (`queued`, `running`, `completed`, `failed`). |
| GET | `/api/v1/datasets/{dataset_name}/export` | The export file; 409 while not completed. |
| DELETE | `/api/v1/datasets/{dataset_name}` | Delete record and file; 409 while running. |
| GET / POST | `/api/v1/queries` | List / save a search query. |
| DELETE | `/api/v1/queries/{query_id}` | Delete a saved query. |
| GET | `/api/v1/stats` | Dataset, exported-source and saved-query counts. |
| GET | `/api/v1/monitoring` | Circuit-breaker states, catalog count, sky-cache catalogs. |
| GET | `/api/v1/monitoring/metrics` | Prometheus exposition. |

### Streaming

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/search/stream` | SSE crossmatch. Query: ra, dec (as typed) or name, radius_arcsec (<= 3600), profile, catalogs (comma-separated), epoch, pm_ra_masyr, pm_dec_masyr, parallax_mas, target_uncertainty_arcsec, target_pm_error_masyr, completeness, target_class. |

### Batch

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/batch/strategies` | Strategy, endpoint and chunk size per catalog; defaults and limits. |
| POST | `/api/v1/batch/crossmatch` | Body: JSON `{targets, catalogs, radius_arcsec, strategies, nearest_only, include_data, format}`, a CSV table, or multipart `file`. Options also as query parameters. `format`: json, rows, csv, parquet. Headers `X-Batch-Request-Count`, `X-Batch-Wall-Time-S`, `X-Batch-Failed-Targets`. 413 over `BATCH_MAX_UPLOAD_BYTES`; 422 invalid targets/catalogs; 502 every catalog failed. |

### Sky cache

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/skycache/status` | Store root and per-catalog rows, partitions, coverage (area, MOC), mirror log. |
| POST | `/api/v1/skycache/mirror` | `{catalog, ra, dec, radius_deg}` or `{catalog, healpix_order, healpix_pixels}` (+ `tile_rows`): the mirror report. 404 unknown catalog, 422 invalid/too large region, 502 nothing fetched. |
| GET | `/api/v1/skycache/cone` | Local-only cone (`catalog, ra, dec, radius_arcsec`); 409 when not fully covered. |
| DELETE | `/api/v1/skycache/{catalog}` | Remove a mirrored catalog; 404 if absent. |

### VizieR

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/vizier/search` | `q`, `ucd`, `wavelength`, `max_catalogs`, `max_tables`, `registry`: catalogs and ranked tables. |
| GET | `/api/v1/vizier/catalog/{identifier}` | Table (`kind: table`) or catalogue (`kind: catalog`) description; 404, 422, 502 (`Retry-After` on transient failures). |
| POST | `/api/v1/vizier/register` | `{table_id, name?, overrides?, replace?}`: saved to the user registry and attached to the running service; 409 name taken, 422 not registrable. |
| GET | `/api/v1/vizier/registered` | User catalogs and whether the service knows them. |

### SED

| Method | Path | Description |
|---|---|---|
| GET / POST | `/api/v1/sed` | `ra, dec, radius_arcsec` or `name`: points, classification, redshift, members, filters, notes. 422 invalid/unknown name, 502 upstream. |

### Time domain

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/lightcurves` | `ra, dec` or `name`; `epoch`, `pm_ra_masyr`, `pm_dec_masyr`, `parallax_mas`, `radius_arcsec` (<= 60), `surveys` (ztf, neowise, gaia, tess), `include_flagged`, `period`, `min_period_days`, `max_period_days`, `oversampling`, `ztf_collection`, `tess_max_sectors`, `tess_bin_minutes`, `neowise_binned`. 404 unknown name, 502 all surveys failed. |
| GET | `/api/v1/solar-system` | `ra, dec, radius_arcsec, epoch_mjd, observer, max_position_error_arcsec`: SkyBoT objects. |

### Imaging

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/cutouts` | Image bytes (png, jpg, fits) for `ra, dec` or `name`, `survey`, `fov_arcmin`, `width`, `height`, `projection`, `stretch`, `cmap`, `min_cut`, `max_cut`. Headers `X-Cutout-*` (survey, HiPS, cache, pixel scale, CDELT, projection, coverage, blank, units, calibration, degraded). |
| GET | `/api/v1/cutouts/surveys` | Survey list (radio to X-ray) with pixel units and epochs. |
| GET | `/api/v1/cutouts/stack` | Panel plan with footprint check and proper-motion offsets. |

### AI

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/ai/query` | `{text, max_retries, verify_adql}`: advanced_query, plan, explanation, adql, scope, resolved_objects, attempts. 422 invalid/uncompilable, 503 no credentials, 502 upstream. |
| POST | `/api/v1/ai/explain` | `{name | ra, dec [, epoch], facts_only, include_crossmatch, radius_arcsec, crossmatch_radius_arcsec, max_references}`: summary, citations, facts. |

### Provenance

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/provenance/manifest` | `{record}` or the `SearchRequest` fields, plus `include_rows`, `include_record`, `live_release_lookup`: `{manifest, record}`. |
| POST | `/api/v1/provenance/replay` | `{manifest, position_tolerance_arcsec, sigma_fraction, numeric_rtol, confidence_tolerance, use_cache}`: identical, content_hash_equal, diff, explanation, new_manifest. |
| GET | `/api/v1/citations` | `catalogs=a,b`, `format=json|bibtex` (`X-Unknown-Citations`). |
| GET | `/api/v1/citations/sources` | Every citable catalog/service. |

### Virtual Observatory

| Method | Path | Description |
|---|---|---|
| GET | `/vo/scs` | Simple Cone Search (`RA, DEC, SR, VERB, CATALOGS`). |
| GET | `/vo/tap` | TAP service root. |
| GET | `/vo/tap/availability` | VOSI availability (no authentication, no quota). |
| GET | `/vo/tap/capabilities`, `/vo/tap/tables`, `/vo/tap/tables/{table_name}`, `/vo/tap/examples` | VOSI capabilities, tableset, one table, DALI examples. |
| GET / POST | `/vo/tap/sync` | Synchronous ADQL (`REQUEST=doQuery, LANG=ADQL, QUERY, MAXREC, RESPONSEFORMAT`). |
| GET / POST | `/vo/tap/async` | UWS job list / create (303 to the job; `PHASE=RUN` starts it). |
| GET / POST / DELETE | `/vo/tap/async/{job_id}` | Job document (`WAIT` blocks) / `ACTION=DELETE` / delete. |
| GET / POST | `/vo/tap/async/{job_id}/phase`, `/parameters`, `/executionduration`, `/destruction` | Job properties. |
| GET | `/vo/tap/async/{job_id}/results`, `/results/result`, `/error`, `/quote`, `/owner` | Results, result document, error document, quote, owner. |

VO errors are DALI error VOTables; request bodies over `vo_server.MAX_REQUEST_BYTES` get 413.

### Alerts

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/alerts` | Stored alerts: `since_mjd, until_mjd, limit, broker, classification, only_new, crossmatch_status`. |
| GET | `/api/v1/alerts/brokers` | Supported brokers and filters. |
| POST | `/api/v1/alerts/poll` | `{broker, since_mjd, until_mjd, limit, crossmatch, class_name, classifier, mjd_field, crossmatch_mode}`. |
| GET | `/api/v1/alerts/{alert_id}` | One alert with its enrichment. |
| POST | `/api/v1/alerts/{alert_id}/crossmatch` | Re-run its crossmatch; 409 without a crossmatch service. |

### Web UI

`GET /`, `/index.html`, `/app.js`, `/styles.css` serve the UI (not part of the OpenAPI document).

---

## 7. Command-line reference

`astrosearch <command> --help` shows every option. Exit status 0 means success.

| Command | Purpose | Exit statuses |
|---|---|---|
| `serve [--host --port --reload]` | Run the API and UI with uvicorn. | |
| `search (--name N | --ra --dec) [--radius --profile --catalogs --epoch --pm-ra --pm-dec --parallax --format json|summary]` | One crossmatch; `--format json` prints only JSON. | 1 error |
| `dataset --name --profile --targets FILE [--radius --catalogs --min-confidence --count-threshold --format --output]` | Build a dataset from a JSON target list. | 1 missing file, 2 invalid input |
| `catalogs [--name]` | List or show catalog definitions (embedded + registered). | 1 unknown |
| `benchmark [--rows --format]` | Export throughput benchmark. | |
| `verify` | Offline self-test (13 checks, including every router and CLI). | 1 failed check |
| `stream (--ra --dec | --name) [--radius --catalogs --epoch --pm-ra --pm-dec --parallax --target-sigma --format jsonl|sse --sources --record]` | Stream a crossmatch. | 1 error event, 2 invalid input |
| `xmatch-calibrate [--fields --seed --radius --threshold --completeness ...]` | Monte-Carlo calibration of the association. | |
| `batch --targets FILE [--catalogs --radius --out --format --strategy --nearest --no-data]` | Batch crossmatch. | 1 upstream, 2 invalid input |
| `mirror --catalog (--ra --dec --radius-deg | --order --pixels) [--store --tile-rows --timeout --json]` | Mirror a region into the sky cache. | 1 tiles failed, 2 no region |
| `skycache status|cone|delete [--store ...]` | Inspect, query or delete the sky cache. | cone: 1 not covered, 2 unknown catalog |
| `vizier search|describe|add|list` | VizieR discovery and registration. | 1 error, 2 usage |
| `sed (--name | --ra --dec) [--radius --plot FILE --json]` | SED, classification, redshift. | 1 error, 2 missing arguments |
| `lightcurve (--ra --dec | --name) [--radius --surveys --epoch --pm-ra --pm-dec --parallax --include-flagged --no-period --format]` | Light curves and periods. | 1 upstream, 2 invalid input |
| `solar-system --ra --dec [--epoch-mjd --radius --observer --format]` | Known bodies in a cone. | 1 upstream, 2 invalid input |
| `cutout (--ra --dec | --name) --out FILE [--fov --survey --format --width --height --projection --stretch --cmap --no-cache --allow-degraded --list-surveys]` | Save a cutout. | 1 upstream, 2 bad arguments, 3 degraded |
| `ask "question" [--run --limit --verify-adql --max-retries --json]` | Compile a natural-language query with Claude. | 1 upstream, 2 bad input, 3 no credentials |
| `explain (--name | --ra --dec) [--facts-only --no-crossmatch --json ...]` | Cited object explanation. | as `ask` |
| `manifest (--record FILE | --ra --dec | --name) [--radius --catalogs --no-live-release -o FILE]` | Provenance manifest. | 3 archives unreachable |
| `replay MANIFEST [--out --tolerance --json]` | Re-run and diff a manifest. | 1 differs, 2 bad input, 3 unreachable, 4 same hash with catalogs failed in both runs |
| `cite [--catalogs --from FILE --bibtex FILE --verify --json]` | Acknowledgements and BibTeX. | 2 nothing known |
| `vo cone|adql|tables` | Run VO queries locally. | |
| `alerts poll|watch|list|show|crossmatch [--db URL ...]` | Alert ingestion and inspection. | 1 broker failure / not found, 2 invalid input, 130 interrupted |

---

## 8. Security, limits and observability

- **Authentication** (`api.authenticate`): API keys (`API_KEYS=k1,k2`, sent as `X-API-Key` or
  `Authorization: Bearer`) and JWT bearer tokens (`JWT_SECRET` HS256 or `JWT_PUBLIC_KEY` RS256,
  `exp` and `sub` required, optional `JWT_ISSUER`/`JWT_AUDIENCE`). `REQUIRE_API_KEY=true` refuses
  to serve (503) until one of them is configured. Without any, requests are identified by
  client address.
- **Scope.** Authentication, the per-identity quota (`API_RATE_LIMIT_PER_MINUTE`, Redis-backed
  when `REDIS_URL` is set) and the body limit apply to every route of every router except
  `/api/v1/health`, `/vo/tap/availability` and the UI files (`UI_PUBLIC=false` puts the UI behind
  authentication too).
- **Body limits.** `MAX_REQUEST_BYTES` (1 MB) for every body, except the batch routes (streamed
  up to `BATCH_MAX_UPLOAD_BYTES`, 50 MB) and the VO services (their own limit and DALI error
  documents).
- **Strict JSON.** A JSON body containing NaN, Infinity or an overflowing number (`1e400`) is a
  422 on every route; validation errors never echo non-finite values (so they stay valid JSON).
- **CORS.** `CORS_ORIGINS` (comma-separated; empty or `*` = any origin). Credentials
  (`CORS_ALLOW_CREDENTIALS`, default true) are only allowed with explicit origins. The
  `X-Request-ID`, `X-Batch-*`, `X-Cutout-*`, `X-Unknown-Citations`, `Retry-After` and `Location`
  headers are exposed.
- **Metrics** (`/api/v1/monitoring/metrics`): `astrosearch_requests_total{method,endpoint,status_code}`,
  `astrosearch_request_latency_seconds`, `astrosearch_active_requests`,
  `astrosearch_catalog_queries_total{catalog,status}` (status `success`, `empty` or `failed`,
  with `+fallback` when a fallback archive was used) and `astrosearch_catalog_query_seconds`.
  Catalog metrics are recorded for every record the API's service finalizes: searches, streams,
  VO queries, per-target dataset searches, SEDs and AI explanations (batch runs and alert
  enrichment use their own derived services and are not counted).
- **Logs** are structured JSON (structlog) with the request id and the authenticated identity.

---

## 9. Configuration

Settings are read from the environment (see `.env.example`).

### Core and API

| Variable | Default | Meaning |
|---|---|---|
| `DEFAULT_RADIUS_ARCSEC` | 3.0 | Default search radius. |
| `REQUEST_TIMEOUT_SECONDS` | 30 | HTTP timeout of archive requests (also caps catalog timeouts when set explicitly). |
| `CATALOG_TIMEOUT_CAP_SECONDS` | none | Upper bound of every catalog's own timeout. |
| `MAX_RESPONSE_BYTES` | 10000000 | Largest archive response read. |
| `SESAME_ENDPOINT` | `https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV` | Name resolver. |
| `CATALOG_REGISTRY_PATH` | `~/.astrosearch/catalogs.yaml` | User catalog registry. |
| `CATALOG_REGISTRY_STRICT` | false | Refuse to start with registry problems. |
| `LOG_LEVEL` | INFO | Logging level. |
| `API_KEYS`, `JWT_SECRET`, `JWT_PUBLIC_KEY`, `JWT_ISSUER`, `JWT_AUDIENCE`, `REQUIRE_API_KEY` | none / false | Authentication (section 8). |
| `API_RATE_LIMIT_PER_MINUTE` | 60 | Requests per identity per minute (0 disables). |
| `MAX_REQUEST_BYTES` | 1048576 | Request body limit. |
| `API_MAX_BATCH_SIZE` | 100 | Searches per `/api/v1/search/batch`. |
| `API_MAX_TARGETS` | 10000 | Targets per dataset request. |
| `API_SEARCH_CACHE_TTL_SECONDS` | 300 | Search cache (complete records only; 0 disables). |
| `CORS_ORIGINS`, `CORS_ALLOW_CREDENTIALS` | any origin, true | CORS (section 8). |
| `UI_PUBLIC` | true | Serve the UI without authentication. |
| `ASTROSEARCH_WEB_DIR` | `web/` | Directory holding the UI files. |
| `REDIS_URL` | none | Response cache, quotas and dataset job queue. |

### Archive access

| Variable | Default | Meaning |
|---|---|---|
| `PROVIDER_REQUESTS_PER_SECOND` | 5 | Requests per second per endpoint. |
| `PROVIDER_FAILURE_THRESHOLD` | 5 | Failures that open a circuit. |
| `PROVIDER_RECOVERY_SECONDS` | 30 | Open-circuit duration. |
| `PROVIDER_PROBE_TIMEOUT_SECONDS` | 180 | Expiry of a half-open probe. |
| `PROVIDER_CACHE_TTL_SECONDS` | 600 | Response cache lifetime. |
| `PROVIDER_CACHE_MAX_ENTRIES`, `PROVIDER_CACHE_MAX_BYTES` | 512, 64 MB | In-process cache bounds. |
| `EPOCH_PAD_MAX_PM_ARCSEC_PER_YR`, `EPOCH_PAD_MAX_ARCSEC`, `EPOCH_PAD_MAX_ROWS` | 10.5, 300, 2000 | Epoch widening without a proper motion. |

### Feature modules

| Variable | Default | Meaning |
|---|---|---|
| `SSE_KEEPALIVE_SECONDS` | 15 | Stream keepalive interval. |
| `BATCH_MAX_TARGETS` / `BATCH_MAX_CATALOGS` / `BATCH_MAX_CONE_TARGETS` | 100000 / 20 / 5000 | Batch limits. |
| `BATCH_MAX_UPLOAD_BYTES` / `BATCH_MAX_RESPONSE_BYTES` | 50 MB / 256 MB | Batch body and answer limits. |
| `BATCH_CONE_CONCURRENCY` / `BATCH_CHUNK_CONCURRENCY` / `BATCH_ENDPOINT_CONCURRENCY` | 8 / 2 / 2 | Batch concurrency. |
| `BATCH_UPLOAD_TIMEOUT_SECONDS` / `BATCH_CATALOG_BUDGET_SECONDS` / `BATCH_CPU_SLICE_SECONDS` | 300 / 1800 / 0.1 | Batch timing. |
| `BATCH_CHUNK_SIMBAD`, `_VIZIER`, `_IRSA`, `_HEASARC`, `_XMATCH` | 5000, 5000, 2000, 2000, 20000 | Targets per upload chunk (for cones up to 10"). |
| `DATASET_BATCH_MIN_TARGETS` | 50 | Datasets with at least this many targets use the batch engine (0 disables). |
| `DATASET_STORAGE_PATH`, `DATABASE_URL`, `S3_BUCKET`, `S3_ENDPOINT_URL` | `datasets`, SQLite, none | Dataset storage. |
| `SKYCACHE_ENABLED` | true | Answer mirrored catalogs from the sky cache. |
| `SKYCACHE_PATH` | `~/.astrosearch/skycache` | Sky cache root. |
| `SKYCACHE_TILE_ROWS` / `SKYCACHE_PARTITION_ROWS` | 10000 / 200000 | Mirror tile and HATS partition sizes. |
| `SKYCACHE_MAX_RADIUS_DEG` / `SKYCACHE_MAX_QUERIES` / `SKYCACHE_MAX_AGE_DAYS` | 1.0 / 256 / none | Mirror limits and data age. |
| `VIZIER_ASU_URL`, `VIZIER_TAP_URL`, `REGTAP_URL` | CDS / GAVO | VizieR and registry endpoints. |
| `VIZIER_TIMEOUT_SECONDS`, `VIZIER_ROUTE_DEADLINE_SECONDS`, `VIZIER_DESCRIBE_CACHE_TTL_SECONDS` | 90, 150, 21600 | VizieR timing and cache. |
| `ASTROSEARCH_SED_OFFLINE_FILTERS` | false | Use only the embedded SVO filter values. |
| `ASTROSEARCH_SVO_FPS_URL`, `ASTROSEARCH_CACHE_DIR` | SVO, `~/.cache/astrosearch` | SVO endpoint and filter cache. |
| `ASTROSEARCH_SED_SUPPLEMENTARY_DEADLINE_SECONDS` | module default | Budget of SED supplementary lookups. |
| `TIMEDOMAIN_CACHE_TTL_SECONDS` | `PROVIDER_CACHE_TTL_SECONDS` or 3600 | Light-curve cache. |
| `TIMEDOMAIN_ZTF_DEADLINE_S`, `_NEOWISE_`, `_GAIA_`, `_TESS_`, `_SKYBOT_`, `_HORIZONS_`, `_RESOLVER_` | 480, 360, 240, 480, 300, 120, 60 | Per-service deadlines. |
| `TIMEDOMAIN_PERIOD_SEARCH_BUDGET_S`, `TIMEDOMAIN_ANALYSIS_WORKERS`, `TIMEDOMAIN_REDIS_RETRY_AFTER_S` | 60, 2, 30 | Analysis budget, workers, Redis circuit breaker. |
| `CUTOUT_CACHE_DIR`, `CUTOUT_CACHE_TTL_SECONDS`, `CUTOUT_CACHE_MAX_BYTES` | `<tmp>/astrosearch-cutouts`, 7 days, 512 MB | Cutout cache. |
| `VO_MAX_RADIUS_DEG`, `VO_CATALOG_ROW_LIMIT`, `VO_MAX_CONE_ROWS` | 0.05, 2000, 3000 | VO cone bounds. |
| `VO_TAP_DEFAULT_MAXREC`, `VO_TAP_HARD_MAXREC`, `VO_MAX_RESULT_BYTES`, `VO_ADQL_EVAL_SECONDS` | 10000, 100000, 16000000, 30 | TAP limits. |
| `VO_UWS_EXECUTION_DURATION`, `VO_UWS_MAX_JOBS`, `VO_UWS_RETENTION_SECONDS`, `VO_UWS_MAX_WAIT_SECONDS`, `VO_UWS_MAX_STORED_BYTES` | 600, 1000, 86400, 60, 256000000 | UWS jobs. |
| `ALERTS_DATABASE_URL` | the dataset metadata database | Alert store. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | none | Claude credentials. |
| `ASTROSEARCH_AI_MODEL`, `ASTROSEARCH_AI_EFFORT`, `ASTROSEARCH_AI_MAX_TOKENS`, `ASTROSEARCH_AI_TIMEOUT`, `ASTROSEARCH_AI_FALLBACKS` | claude-opus-5, high, 16000, 300, default | Claude usage. |
| `ADS_API_TOKEN` | none | NASA ADS citation ranking for explanations. |
| `ASTROSEARCH_GIT_COMMIT` | detected | Commit recorded in manifests. |

---

## 10. Known limitations

- **Association.** Proper motions are propagated linearly (Barnard's star curvature, about
  0.1-0.16" over 16 years, is absorbed, not modelled). Above 20,000 hypotheses a beam search
  replaces exact enumeration. Default completeness priors are the same for all objects of a
  class. The crowded-region list is the Harris globular clusters plus the LMC, SMC, M31 and M33
  cores. Blended Pan-STARRS positions in crowded fields may stay unmatched. The extended-object
  centre scatter uses identity-row errors and the observed SIMBAD/NED spread, not object sizes.
  `EXTENDED_POINT_COMPLETENESS` (1%) also applies to compact X-ray/radio sources, so a sparse
  catalogue's single compact source at an extended target's centre can stay below 0.5.
  Rows spanning an epoch range are classified in-radius by closest approach and weighted by
  the marginal likelihood.
- **Streaming.** Counts in `catalog` events come from the first-pass split and can differ from
  the final record for fast movers whose motion is adopted later.
- **Batch.** The batch engine does not run density probes: `finalize` records the probe as
  `not_run` with a warning and uses the conservative cone-only density for significant Poisson
  excesses; crowded stellar systems stay map-based there (datasets route them per target).
  XMatch takes one distance per request and at most 180"; generic `vizier:<table>`
  catalogs cannot fall back to cones; cone-only catalogs are limited to 5000 targets; the whole
  request body is read into memory (up to 50 MB).
- **Sky cache.** Only the tap, irsa_gator, mast and sdss providers can be mirrored (truncation
  must be detectable); the catalog is rewritten on every mirror (regional mirrors, not all-sky
  imports); write locking is per process for in-memory state; `POST /mirror` is synchronous.
- **VizieR.** Positional-error conventions are inferred from column descriptions (recorded as
  assumptions); tables with only galactic or B1950 positions cannot be registered.
- **SED.** The classification is a weighted heuristic; Cen A comes out `galaxy` by tie-break;
  X-ray fluxes assume photon index 2 and N_H = 3e20 cm^-2 for ROSAT.
- **Time domain.** TESS covers SPOC 2-min targets only; period errors are optimistic for
  non-sinusoidal light curves; Gaia-only damped random walks are adopted as periodic at about 5%.
- **Imaging.** The hips2fits colormap allow-list reflects the service on 2026-09-28; partly blank
  JPEG cutouts report no coverage fraction.
- **VO.** The UWS store is in memory; the ADQL subset has no joins, GROUP BY or uploads.
- **Alerts.** Fink paging stops after 20 requests per poll (the backlog continues on later
  polls); background enrichment is deduplicated within one process only.
- **AI.** Needs Anthropic credentials; object-type filters need SIMBAD's `otypedef`.

---

## 11. Testing

### Suites

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q              # offline (default: -m 'not live')
OPENBLAS_NUM_THREADS=1 python -m pytest -q -m live      # live, against the real archives
```

- **Offline** tests replay recorded archive answers with respx (`tests/fixture_io.py`,
  `tests/fixtures/`); a request that differs from the recording fails instead of being served a
  stale answer. The session uses temporary stores (`tests/conftest.py` sets `SKYCACHE_PATH`,
  `CATALOG_REGISTRY_PATH` and `DATASET_STORAGE_PATH`), so local mirrors or registered catalogs
  never change the results.
- **Live** tests (`@pytest.mark.live`) run the same code against the archives with polite pacing
  (5 requests/s). They skip only on network errors and HTTP 5xx; wrong answers fail.
- **Integration** tests cover the assembled system:
  - `tests/test_integration_api.py`: mounted routers and OpenAPI, shared state and shutdown,
    authentication/quota/body-limit/strict-JSON middleware, CORS, the vizier-merged registry,
    the sky cache in front of the archives, catalog metrics, resolver answers, Redis off the
    event loop;
  - `tests/test_integration_e2e.py`: the real application under uvicorn on a free port, every
    route of every module and the UI, with recorded upstream answers;
  - `tests/test_integration_live.py`: the same server against the real archives (`-m live`);
  - `tests/test_integration_cli.py`: `--help` of every subcommand and offline runs of each
    command;
  - `tests/test_integration_datasets.py`: batch-mode and per-target datasets give the same
    counterparts and posteriors; exports carry the probabilities;
  - `tests/test_integration_ui.py`: the UI's endpoints exist and light-curve requests carry the
    target's motion.
- `astrosearch verify` is a 13-check offline self-test for installations.

### Recording fixtures

Recorders hit the live services politely; re-record one case at a time on small machines.

| Fixtures | Command |
|---|---|
| Catalog cones (canaries, epoch targets, fallbacks) | `python tests/fixture_io.py record [target] [catalog]` |
| Matching (crowded fields, named searches) | `python tests/test_matching_fixtures.py record [target]` / `record-search <key>` |
| Batch | `python tests/test_batch_fixtures.py record [case] [catalog]` |
| Sky cache | `python tests/test_skycache_live.py record [catalog ...] [--mirror-only]` |
| VizieR | `python tests/test_vizier.py record [case ...]` |
| SED | `python tests/test_sed.py record [svo] [canary] [targets | <key> ...] [supplementary-only]` |
| Time domain | `python tests/test_timedomain_live.py record [case ...]` |
| Imaging | `python tests/fixtures/imaging/record_imaging.py [scenario ...]` |
| AI | `python tests/fixtures/ai/record_ai_fixtures.py` |
| Provenance (ADS, DOI) | `python tests/fixtures/provenance/record_fixtures.py [case ...]` |
| VO | `python tests/test_vo_server.py record` |
| Alerts | `python tests/fixtures/alerts/record_alerts.py [famous|enrich|variables|backlog|validation|round3]` |

Recorded bodies have IPv4 addresses redacted (`fixture_io.redact`).

---

## 12. References

- Baluev R. V. 2008, MNRAS 385, 1279 (periodogram false-alarm probabilities)
- Berthier J. et al. 2006, ASP Conf. Ser. 351, 367 (SkyBoT)
- Boch T., Pineau F.-X., Derriere S. 2012, ASP Conf. Ser. 461, 291 (CDS XMatch)
- Budavari T., Szalay A. S. 2008, ApJ 679, 301 (Bayesian cross-identification)
- Dowler P. et al. 2019, IVOA Recommendation: Table Access Protocol 1.1
- Eastman J., Siverd R., Gaudi B. S. 2010, PASP 122, 935 (BJD_TDB)
- Fernique P. et al. 2015, A&A 578, A114 (HiPS); Fernique P. et al. 2022, IVOA MOC 2.0
- Gorski K. M. et al. 2005, ApJ 622, 759 (HEALPix)
- Lupton R. H., Gunn J. E., Szalay A. S. 1999, AJ 118, 1406 (asinh magnitudes)
- Masci F. J. et al. 2019, PASP 131, 018003 (ZTF)
- Montgomery M. H., O'Donoghue D. 1999, Delta Scuti Star Newsletter 13, 28 (period errors)
- Ochsenbein F., Bauer P., Marcout J. 2000, A&AS 143, 23 (VizieR)
- Oke J. B., Gunn J. E. 1983, ApJ 266, 713 (AB magnitudes)
- Pineau F.-X. et al. 2017, A&A 597, A89 (probabilistic multi-catalogue cross-match)
- Planck Collaboration 2020, A&A 641, A6 (cosmological parameters)
- Rodrigo C., Solano E. 2020, SVO Filter Profile Service
- Salvato M. et al. 2018, MNRAS 473, 4937 (NWAY)
- Sokolovsky K. V. et al. 2017, MNRAS 464, 274 (variability indices)
- Tully R. B. et al. 2023, ApJ 944, 94 (Cosmicflows-4)
- VanderPlas J. T. 2018, ApJS 236, 16; VanderPlas J. T., Ivezic Z. 2015, ApJ 812, 18 (Lomb-Scargle)
- Wenger M. et al. 2000, A&AS 143, 9 (SIMBAD)
- Williams R. et al. 2008, IVOA Recommendation: Simple Cone Search 1.03
- Zechmeister M., Kuerster M. 2009, A&A 496, 577 (generalised Lomb-Scargle)

Catalog-specific references and acknowledgements are returned by `GET /api/v1/citations`
and `astrosearch cite`.
