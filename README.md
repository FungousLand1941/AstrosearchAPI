# AstroSearch

AstroSearch crossmatches a sky position or object name against public astronomical archives
(Gaia DR3, SIMBAD, NED, 2MASS, AllWISE, Pan-STARRS, SDSS, FIRST, NVSS, VLASS, LoTSS, ROSAT,
Chandra, XMM-Newton, the NASA Exoplanet Archive, and any VizieR table you register), and gives
every catalogue row a Bayesian probability of being the target's counterpart. Around that engine
it provides:

- a REST API (FastAPI) with a single-page web UI at `/`;
- server-sent-event streaming of results as each archive answers;
- batch crossmatching of up to 100,000 targets through TAP uploads and CDS XMatch;
- a local HATS/Parquet sky cache that answers mirrored regions without the network;
- VizieR discovery and one-call registration of any VizieR table as a new catalogue;
- SEDs with object classification and redshift, multi-survey light curves with variability and
  period analysis, solar-system object checks, and multi-survey image cutouts;
- IVOA services (Simple Cone Search, TAP/ADQL with UWS async jobs, VOSI) for TOPCAT, Aladin,
  pyvo and astroquery;
- live transient-alert ingestion (ALeRCE, Fink) with automatic crossmatch enrichment;
- reproducibility manifests, replay diffs and verified citations;
- natural-language queries and cited object explanations with Claude;
- dataset generation (JSON, CSV, Parquet, FITS) with the match probabilities in every row.

The full reference (architecture, every endpoint, every CLI command, configuration, the science
methods with citations, testing) is in [DOCUMENTATION.md](DOCUMENTATION.md).

## Installation

Python 3.12 or later.

```bash
git clone <this repository> AstroSearch && cd AstroSearch
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"           # runtime + test dependencies
# optional extras: ".[plot]" (SED plots), ".[skycache-hats]" (remote HATS catalogs via lsdb),
#                  ".[storage]" (PostgreSQL, S3), ".[worker]" (Redis RQ dataset workers)
```

This installs the `astrosearch` command. The web UI lives in `web/`; an editable install or a
checkout serves it directly, a wheel installs it under `share/astrosearch/web` of its install
scheme (a virtual environment, `--user` or `--prefix`), and `ASTROSEARCH_WEB_DIR` points the
server at another copy.

**Install AstroSearch in its own virtual environment.** Its modules are installed as top-level
modules with generic names (`api`, `main`, `cli`, `models`, `datasets`, `batch`, `ai`, ...), so
next to another distribution with the same module name (for example Hugging Face `datasets`) one
of the two is shadowed and the CLI and API fail to start (`ImportError: cannot import name
'MetadataStore' from 'datasets'`). A script named `main.py` or `api.py` in the working directory
shadows them too; `astrosearch verify` names every shadowed module and the file it resolves to.
Moving the modules into an `astrosearch` package is planned (DOCUMENTATION.md, section 10).

The dependency floors are the lowest versions that install on Python 3.12 and pass the suite:
astropy 8 (the first release whose fast Lomb-Scargle gives the same powers on sub-grids of a
frequency grid, which the chunked period search relies on) and therefore numpy 2. Check them with
`uv pip install --resolution lowest-direct -e ".[dev]"` in a fresh environment.

Check the installation (offline, no network):

```bash
astrosearch verify
```

## Quick start

Run the API and the web UI:

```bash
astrosearch serve --host 127.0.0.1 --port 8000
# UI:  http://127.0.0.1:8000/          API docs: http://127.0.0.1:8000/api/docs
```

Search from the command line:

```bash
astrosearch search --name "3C 273" --radius 10
astrosearch search --ra 187.2779154 --dec 2.0523883 --radius 10 --catalogs gaia_dr3,simbad,nvss --format json
astrosearch stream --name "Barnard's star" --radius 5            # one JSON line per event
astrosearch batch --targets targets.csv --catalogs gaia_dr3,simbad --radius 3 --out matches.parquet
astrosearch sed --name "3C 273"
astrosearch lightcurve --name "RR Lyr" --surveys ztf,gaia
astrosearch cutout --name M87 --survey dss2 --fov 5 --out m87.png
astrosearch vizier search "Swift 2SXPS"
astrosearch vizier add IX/58/2sxps --name swift_2sxps
astrosearch mirror --catalog gaia_dr3 --ra 187.2779 --dec 2.0524 --radius-deg 0.2
astrosearch alerts poll --broker alerce --limit 20
```

`astrosearch --help` lists every command; `astrosearch <command> --help` documents each one.
Searches are limited to a cone of `API_MAX_RADIUS_ARCSEC` (default 1800" = 30'); `dataset
--catalogs` must belong to `--profile`, and its `--output` may be any unused path (over REST,
`output_path` must stay inside `DATASET_STORAGE_PATH`). VizieR-hosted catalogs (`vlass`, `lotss`,
tables added with `vizier add`) are batch-matched through CDS XMatch by default, and a registered
table is cited with its own paper.

Search over HTTP:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/search \
     -H 'content-type: application/json' \
     -d '{"name": "3C 273", "radius_arcsec": 10, "catalogs": ["gaia_dr3", "simbad", "nvss"]}'
curl -N "http://127.0.0.1:8000/api/v1/search/stream?name=3C%20273&radius_arcsec=10"
```

When the name resolver (CDS Sesame) is down, both answer 503 with `Retry-After: 30` (502 for an
unusable resolver answer, 404 for an unknown name).

From Python:

```python
import asyncio
from main import crossmatch, search_object

record = asyncio.run(search_object("3C 273", radius_arcsec=10.0))
target = next(g for g in record.crossmatch_groups if g["contains_target"])
for member in target["members"]:
    print(member["catalog"], member["source_id"], member["target_probability"])
```

Each match's `confidence` is the posterior probability that the row is the target's
counterpart (Budavari & Szalay 2008; Salvato et al. 2018); `crossmatch_groups` are the most
probable partition of all rows in the cone into physical objects, the target's group first.

## Testing

```bash
python -m pytest -q                 # offline suite: replays recorded archive answers (no network)
python -m pytest -q -m live         # live suite: the same code against the real archives
python -m pytest -q tests/test_integration_e2e.py   # the real server under uvicorn, every route
```

Set `OPENBLAS_NUM_THREADS=1` on small machines. Live tests skip only when an archive or the
name resolver is unreachable (network error, timeout, HTTP 5xx or 429; `tests/live_policy.py`);
a parse error, a non-network catalog failure or an HTTP 500 from the API fails them. The Claude
tests (`tests/test_ai_live.py`, `tests/test_ai_round3_live.py`) need `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`) and are skipped without it; so far the Claude features have been
verified offline only, against a mocked SDK. Recording new fixtures is described in
[DOCUMENTATION.md](DOCUMENTATION.md#11-testing).

## License

MIT License.
