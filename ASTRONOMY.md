# Astronomical summaries and MIT CSAIL Mantis

This extension adds catalog-grounded object and extrasolar-system summaries, a full NASA Exoplanet Archive snapshot, SIMBAD host cross-references, typed Mantis imports, and refresh workflows. It uses **MIT CSAIL Mantis**, at https://mantis.csail.mit.edu. No language-model API key is required.

## Install and summarize

Use Python 3.12 or later in a virtual environment:

```sh
pip install -e '.[dev]'
python -m astronomy_pipeline summarize TRAPPIST-1
python -m astronomy_pipeline summarize 'TRAPPIST-1 b'
python -m astronomy_pipeline summarize-object M87 --profile optical
python main.py serve
```

The planet-name command expands to the host's cataloged planetary system. Host and planet identifiers are exact, case-sensitive archive queries; unknown aliases return an explicit not-found response. The archive's Gaia, HD, HIP, and TIC host identifiers can also be used.

Existing search and dataset endpoints remain available. The new endpoints use the same authentication and request quotas:

```sh
curl -X POST http://127.0.0.1:8000/api/v1/summaries/system \
  -H 'Content-Type: application/json' -d '{"name":"TRAPPIST-1"}'
curl -X POST http://127.0.0.1:8000/api/v1/summaries/object \
  -H 'Content-Type: application/json' -d '{"name":"M87","profile":"optical"}'
```

`system` returns narrative text, planet measurements, units, asymmetric uncertainties, limit flags, publication references, archive update dates, raw records, and query provenance. `object` summarizes the existing multi-catalog search evidence and reports failed providers and candidate counterparts. Interactive API documentation is at `/api/docs`.

System errors: 404 for no matching archive entry, 422 for invalid input, 502 for an upstream catalog failure. An upstream failure is never reported as an empty successful catalog.

## Build the complete dataset

```sh
python -m astronomy_pipeline sync --state-dir astronomy-data
```

This fetches all rows in `pscomppars` and resolves the host identifiers against SIMBAD in batches of 100. The implementation queries the live column schema, requests a complete bounded table, and verifies the row count and uniqueness. NASA's service can apply `TOP` before ordering, so name-keyset pagination is deliberately avoided. Snapshots are limited to 100,000 planets and 64 MiB per TAP response; exceeding either limit fails explicitly. This is a complete exoplanet catalog and its relevant SIMBAD host subset, **not a mirror of all SIMBAD objects**.

Files under `astronomy-data`:

| File | Purpose |
|---|---|
| `latest.json` | Last successfully committed snapshot, counts, changes, and query provenance |
| `last-check.json` | Last successful check, including unchanged checks |
| `snapshots/<hash>/catalog.json` | Original planet records and SIMBAD identity candidates |
| `snapshots/<hash>/summaries.json` | System and planet summaries |
| `snapshots/<hash>/mantis.csv` | Typed map input containing planets, host systems, and SIMBAD objects |
| `snapshots/<hash>/relationships.json` | Explicit planet–host and host–SIMBAD relationships |
| `snapshots/<hash>/manifest.json` | Immutable run metadata and provenance |
| `publications.json` | Publication checkpoints; pending means the server outcome needs reconciliation |
| `mantis-targets.json` | Saved destination space for successive publications |

Snapshots are staged and promoted atomically. Hashes exclude retrieval timestamps but include source data and cross-references. An unchanged catalog does not create a second snapshot. Changes include added, updated, and removed planet names and changes to SIMBAD associations. Old snapshots remain available for audit. A process lock prevents overlapping sync/publication operations; after a crash, confirm the process has stopped before removing `.sync.lock`.

## Create the Mantis space

Install the MIT CSAIL CLI and authenticate locally. Never commit or paste credentials into the repository:

```sh
npm install -g mantisai-cli@3.7.0
mantis setup
mantis use get_space_context
python -m astronomy_pipeline publish --state-dir astronomy-data --dry-run
python -m astronomy_pipeline publish --state-dir astronomy-data
```

The first publication creates a space called **Astronomical Catalogs** using the CLI's `--private` option. The current server reports this visibility as `unlisted`; consult Mantis sharing controls before treating a link as restricted access. Subsequent changed snapshots are new versioned maps in that same space. To use an existing destination, pass its real UUID with `--space-id`. The publisher does not change the current Mantis space/thread selection.

The publisher checks MCP context first. If that interface is unavailable but a REST space-list request authenticates successfully, it proceeds through the official REST-backed map creation command. A transient MCP rejection therefore does not require replacing a valid API key.

The CLI submits asynchronous map creation. `submitted` means Mantis accepted the job, not that embedding is finished. Open the returned space URL and inspect the map once it finishes. A submitted snapshot is not uploaded again. A timeout or ambiguous response leaves a pending checkpoint and blocks automatic retries: inspect Mantis and reconcile `publications.json` before retrying, so a lost response cannot create duplicate maps. Save the existing map identifiers when the upload succeeded; remove a pending entry only after confirming that no corresponding map exists.

The map includes a semantic `summary`, categorical source/kind/host/status fields, numeric positions and measurements, update dates, and source links. `host_id` connects planet records to their host system for filtering. Full relationships and matched aliases are in `relationships.json`; native Mantis graph edges are not automatically created. Spatial proximity in Mantis reflects semantic similarity, not angular sky separation or physical association.

Useful map explorations include filtering `kind=planet`, grouping by `discovery_method`, examining `host_ambiguous` associations, and filtering all records for a particular `host_id`. Inspect the map's actual dimensions before issuing Mantis filters.

## Refresh as databases change

For a continuously running local worker:

```sh
python -m astronomy_pipeline watch --state-dir astronomy-data --interval-hours 24 --publish
```

The first cycle runs immediately. Later cycles refetch both sources, capture additions/deletions/measurement changes, and publish only snapshots without an existing submitted checkpoint. All publications stay in the saved space. Failures are logged and retried on the next interval; the last successful snapshot remains available. Keep this process running under your chosen service manager; stop it with Ctrl+C. The code does not install a system service automatically.

`.github/workflows/astronomy.yml` runs offline checks on pushes and pull requests, and downloads fresh catalog snapshots daily at 06:23 UTC or on manual dispatch. It retains artifacts for 30 days and restores prior snapshot state from Actions cache for change detection. Cache eviction can lose historical comparison state; download artifacts for durable archival needs. The hosted workflow does not upload to Mantis or store Mantis credentials. The local authenticated watcher handles that step. GitHub scheduling starts only after the workflow is on the repository's default branch with Actions enabled.

## Scientific interpretation

- A planet, its host system, and a SIMBAD object are separate records with separate IDs.
- SIMBAD associations use exact identifiers across hostname/Gaia/HD/HIP/TIC aliases. Conflicting identifiers produce `ambiguous`, with every candidate and matching alias preserved. Missing identifiers produce `unmatched`; there is no automatic nearest-neighbor fallback that could silently join an unrelated star.
- `pscomppars` combines parameters from different publications. It is useful for catalog exploration but need not describe a single self-consistent physical solution. Per-field references remain in the structured summaries.
- Mass provenance is retained, including `Msini`; it must not be treated as a measured true mass. Limit flags and missing uncertainties are explicit in the text and JSON. Mantis numeric columns contain the reported values; consult the summary/raw evidence before interpreting limits or doing statistical analysis.
- A temperature or radius does not establish habitability. Unavailable values remain unknown.
- Full-table requests are not a transactional freeze of the remote databases. Count and uniqueness checks catch truncation and many concurrent changes, but a same-count edit during collection can still produce a mixed-time snapshot. Query timestamps and response hashes document the actual retrievals.

## Verification

```sh
python -m pytest tests -q
python -m ruff check astronomy.py astronomy_pipeline.py tests
python main.py verify
```

Offline tests cover TAP errors and nulls, exact large identifiers, conflicting aliases, source measurement limits/errors, unchanged snapshots, deletion detection, failure preservation, process locking, uncertain publication outcomes, API validation, and authentication. Live archive calls are separate from offline tests.

## Sources

- [NASA Exoplanet Archive TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [Planetary Systems and Composite Parameters definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)
- [SIMBAD TAP service](https://simbad.cds.unistra.fr/simbad/sim-tap)
- [MIT CSAIL Mantis](https://mantis.csail.mit.edu)
- [MIT CSAIL Mantis CLI](https://github.com/KellisLab/mantis-cli)

Credit the source archives and original publications when using their data in research. Response hashes and query provenance in each snapshot support reproducibility.
