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
checkout serves it directly, a wheel installs it under `<prefix>/share/astrosearch/web`, and
`ASTROSEARCH_WEB_DIR` points the server at another copy.

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

Search over HTTP:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/search \
     -H 'content-type: application/json' \
     -d '{"name": "3C 273", "radius_arcsec": 10, "catalogs": ["gaia_dr3", "simbad", "nvss"]}'
curl -N "http://127.0.0.1:8000/api/v1/search/stream?name=3C%20273&radius_arcsec=10"
```

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

Set `OPENBLAS_NUM_THREADS=1` on small machines. Live tests skip only on network errors and HTTP
5xx answers from the archives. Recording new fixtures is described in
[DOCUMENTATION.md](DOCUMENTATION.md#11-testing).

## License

MIT License.
