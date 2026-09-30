"""Recording and replay of real batch-crossmatch exchanges (TAP uploads, CDS XMatch) -- helpers, no tests.

Multipart requests carry a random boundary, so they are matched on their parsed form fields (every field,
including the uploaded target table, must be identical) instead of the raw body. Other requests (cone
fallbacks) use the strict signature matching of ``fixture_io``.

Record (live network; one upload/XMatch request per catalog and radius group):

    .venv/Scripts/python.exe tests/test_batch_fixtures.py record            # every case
    .venv/Scripts/python.exe tests/test_batch_fixtures.py record canary     # one case
    .venv/Scripts/python.exe tests/test_batch_fixtures.py record canary gaia_dr3   # one catalog of a case
    .venv/Scripts/python.exe tests/test_batch_fixtures.py record j2000_tables vizier:I/322A/out

Stored as ``tests/fixtures/batch/<case>/<catalog>.json`` plus ``<catalog>.<n>.body``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import HTTP as HTTP_POLICY
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fixture_io import FIXTURES, TARGETS, FixtureMismatch, redact, request_signature  # noqa: E402

BATCH_FIXTURES = FIXTURES / "batch"

# The canary targets of fixture_io (no epoch), so batch answers can be compared offline with the
# recorded per-target cone searches in tests/fixtures/<target>/<catalog>.json (radius 10").
CANARY_TARGETS = [{"id": key, "ra": ra, "dec": dec} for key, (ra, dec) in TARGETS.items()]
CANARY_RADIUS = 10.0
UPLOAD_CATALOGS = ["gaia_dr3", "simbad", "twomass_psc", "allwise", "first", "nvss", "vlass", "lotss", "rosat",
                   "chandra", "xmm"]

# Vega (SIMBAD: ICRS J2000 279.23473479 +38.78368896, pm +200.94 +286.23 mas/yr, plx 130.23 mas) and a
# Vega-adjacent position 60" north (nothing of Vega's may match there within 5").
VEGA = {"id": "Vega", "ra": 279.23473479, "dec": 38.78368896, "epoch": 2000.0,
        "pm_ra_masyr": 200.94, "pm_dec_masyr": 286.23}
VEGA_ADJACENT = {"id": "Vega+60N", "ra": 279.23473479, "dec": 38.78368896 + 60.0 / 3600.0}
VEGA_CATALOGS = ["gaia_dr3", "simbad", "twomass_psc", "allwise"]

# VizieR tables through CDS XMatch (the spec's 2MASS, AllWISE, PS1, SDSS tables, and the Gaia DR3 table
# requested by its VizieR name) for 3C 273 and HD 209458.
GENERIC_TARGETS = [{"id": key, "ra": TARGETS[key][0], "dec": TARGETS[key][1]} for key in ("3c273", "hd209458")]
GENERIC_CATALOGS = ["vizier:II/246/out", "vizier:II/328/allwise", "vizier:II/349/ps1", "vizier:V/154/sdss16",
                    "vizier:I/355/gaiadr3"]

# Eight targets 0.5" from HD 209458 (canary position) at position angles 0, 45, ..., 315 deg: SIMBAD lists
# the host star and its planet 'HD 209458b' at coordinates differing by ~1e-17 deg, so the nearest identity
# must not depend on float noise.
HD209458_OFFSETS = [
    {"id": f"HD209458+0.5@{pa}", "ra": TARGETS["hd209458"][0] + 0.5 / 3600.0 * math.sin(math.radians(pa))
     / math.cos(math.radians(TARGETS["hd209458"][1])),
     "dec": TARGETS["hd209458"][1] + 0.5 / 3600.0 * math.cos(math.radians(pa))}
    for pa in range(0, 360, 45)
]

# Bright high-proper-motion stars at their SIMBAD ICRS J2000 positions (sim-tap basic, 2026-09-28), matched
# against generic VizieR tables whose meta.main positions are NOT what CDS XMatch matches on (Hipparcos
# RAICRS/RArad are at J1991.25; XMatch matches VizieR's computed _RAJ2000/_DEJ2000 at J2000), SIMBAD and
# 2MASS. Every star twice: as given (no epoch, the default batch input) and at epoch 2000 with its SIMBAD proper
# motion ("@pm" ids).
BRIGHT_STARS: dict[str, tuple[float, float, float, float]] = {
    "Barnard": (269.4520769586187, 4.693364966576667, -801.551, 10362.394),
    "61 Cyg A": (316.7247482895925, 38.74941731943694, 4164.209, 3249.614),
    "Arcturus": (213.915300294925, 19.1824091615312, -1093.39, -2000.06),
    "Sirius": (101.28715533333335, -16.71611586111111, -546.01, -1223.07),
    "Procyon": (114.82549790798149, 5.224987557059477, -714.59, -1036.8),
    "Altair": (297.69582729638694, 8.868321196436963, 536.23, 385.29),
    "HD 209458": (330.79488644388, 18.88431927594, 29.766, -17.976),
}
HIPPARCOS_TARGETS = (
    [{"id": name, "ra": ra, "dec": dec} for name, (ra, dec, _pa, _pd) in BRIGHT_STARS.items()]
    + [{"id": f"{name}@pm", "ra": ra, "dec": dec, "epoch": 2000.0, "pm_ra_masyr": pa, "pm_dec_masyr": pd}
       for name, (ra, dec, pa, pd) in BRIGHT_STARS.items()]
)
HIPPARCOS_CATALOGS = ["vizier:I/239/hip_main", "vizier:I/311/hip2", "vizier:I/259/tyc2", "simbad", "twomass_psc"]

# The same survey requested under its registry name and as a VizieR view must get the same posterior: 3C 273,
# HD 209458 and a star in the core of M13 (2MASS 16414162+3627415, SIMBAD M 13 centre 16 41 41.634 +36 27 40.75).
DENSITY_TARGETS = [{"id": key, "ra": TARGETS[key][0], "dec": TARGETS[key][1]} for key in ("3c273", "hd209458")] + [
    {"id": "M13 core", "ra": 250.423475, "dec": 36.461319}]
DENSITY_CATALOGS = ["twomass_psc", "vizier:II/246/out", "gaia_dr3", "vizier:I/355/gaiadr3"]
DENSITY_RADIUS = 3.0

# A target whose epoch-widened cone exceeds the CDS XMatch limit (180"): 3C 286 (SIMBAD J2000
# 13 31 08.288 +30 30 32.96) at epoch 2024 without a proper motion needs 5" + 10.5"/yr x 26.6 yr against 2MASS
# (1997.4-2001.2) -- capped at 180". Its 2MASS counterpart is 13310829+3030331.
WIDE_EPOCH_TARGETS = [{"id": "3C 286@2024", "ra": 202.7845329, "dec": 30.5091550, "epoch": 2024.0}]

# Tycho-2 identifiers: a 170" cone on the h Persei (NGC 869) core holds several Tycho-2 stars whose Num (number
# of observations, UCD meta.id) coincide; their identifiers are TYC1-TYC2-TYC3.
TYCHO_FIELD_TARGETS = [{"id": "h Per", "ra": 34.7417, "dec": 57.1350}]

# High-proper-motion and planet-host stars at their SIMBAD ICRS J2000 positions with SIMBAD proper motions
# (mas/yr; sim-tap basic, 2026-09-29), for the astrometric VizieR tables whose own meta.main position is at J2000
# (UCAC4 I/322A, URAT1 I/329, USNO-B1.0 I/284, NOMAD I/297, Gaia DR2 I/345, EDR3 I/350) and for Hipparcos /
# Tycho-2 / Gaia DR2 with targets at epoch 2016.
ASTROMETRIC_STARS: dict[str, tuple[float, float, float, float]] = {
    "Barnard": (269.4520769586187, 4.693364966576667, -801.551, 10362.394),
    "61 Cyg A": (316.7247482895925, 38.74941731943694, 4164.209, 3249.614),
    "Kapteyn": (77.91912433145708, -45.01843381524333, 6491.223, -5708.614),
    "Groombridge 1830": (178.2448639115646, 37.718681698090286, 4002.655, -5817.8),
    "Lacaille 9352": (346.4668157737879, -35.85307088473306, 6765.995, 1330.285),
    "tau Cet": (26.01701307163417, -15.93747989102139, -1721.728, 854.963),
    "51 Peg": (344.36658535524, 20.768832511140005, 207.328, 61.164),
    "HD 189733": (300.18213726402, 22.71085365096, -3.208, -250.323),
    "HD 80606": (140.65657001363, 50.60373203519, 56.022, 10.331),
    "Polaris": (37.954560670189856, 89.26410896994187, 44.48, -11.85),
}
# As given by a user with J2000 positions and no epoch (the default batch input).
J2000_TABLE_TARGETS = [{"id": name, "ra": ra, "dec": dec} for name, (ra, dec, _pa, _pd) in ASTROMETRIC_STARS.items()]
J2000_TABLE_CATALOGS = ["vizier:I/322A/out", "vizier:I/329/urat1", "vizier:I/284/out", "vizier:I/297/out",
                        "vizier:I/345/gaia2", "vizier:I/350/gaiaedr3", "vizier:I/317/sample"]


def _at_2016(name: str) -> dict[str, Any]:
    from models import propagate_radec

    ra, dec, pa, pd = ASTROMETRIC_STARS[name]
    ra16, dec16 = propagate_radec(ra, dec, pa, pd, 2000.0, 2016.0)
    return {"id": name, "ra": ra16, "dec": dec16, "epoch": 2016.0, "pm_ra_masyr": pa, "pm_dec_masyr": pd}


# The same stars at epoch 2016.0 (SIMBAD J2000 positions moved with their proper motions) with the proper motion.
DATED_2016_TARGETS = [_at_2016(name) for name in ASTROMETRIC_STARS]
DATED_2016_CATALOGS = ["vizier:I/311/hip2", "vizier:I/239/hip_main", "vizier:I/259/tyc2", "vizier:I/345/gaia2"]
# 3C 273 (SIMBAD J2000) in a 20" cone: several USNO-B1.0 / NOMAD sources whose 13-character ids XMatch declares
# as char arraysize 12.
USNOB_FIELD_TARGETS = [{"id": "3C 273", "ra": 187.27791594049, "dec": 2.05238823055}]

CASES: dict[str, dict[str, Any]] = {
    "canary": {"targets": CANARY_TARGETS, "catalogs": UPLOAD_CATALOGS, "radius": CANARY_RADIUS},
    "vega": {"targets": [VEGA, VEGA_ADJACENT], "catalogs": VEGA_CATALOGS, "radius": 5.0},
    "generic": {"targets": GENERIC_TARGETS, "catalogs": GENERIC_CATALOGS, "radius": 5.0},
    "hd209458_offsets": {"targets": HD209458_OFFSETS, "catalogs": ["simbad"], "radius": 3.0},
    "hipparcos": {"targets": HIPPARCOS_TARGETS, "catalogs": HIPPARCOS_CATALOGS, "radius": 5.0},
    "density": {"targets": DENSITY_TARGETS, "catalogs": DENSITY_CATALOGS, "radius": DENSITY_RADIUS},
    "wide_epoch": {"targets": WIDE_EPOCH_TARGETS, "catalogs": ["vizier:II/246/out", "twomass_psc"], "radius": 5.0},
    "tycho_field": {"targets": TYCHO_FIELD_TARGETS, "catalogs": ["vizier:I/259/tyc2"], "radius": 170.0},
    "j2000_tables": {"targets": J2000_TABLE_TARGETS, "catalogs": J2000_TABLE_CATALOGS, "radius": 5.0},
    "dated_2016": {"targets": DATED_2016_TARGETS, "catalogs": DATED_2016_CATALOGS, "radius": 3.0},
    "usnob_field": {"targets": USNOB_FIELD_TARGETS, "catalogs": ["vizier:I/284/out", "vizier:I/297/out"],
                    "radius": 20.0},
}


def safe_name(catalog: str) -> str:
    return catalog.replace(":", "_").replace("/", "_")


def multipart_fields(content_type: str, body: bytes) -> dict[str, str]:
    """Parsed multipart/form-data fields (file parts included as text), independent of the boundary."""
    message = BytesParser(policy=HTTP_POLICY).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[str(name)] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
    return fields


@dataclass
class BatchExchange:
    method: str
    url: str
    request_content_type: str
    fields: dict[str, str] | None
    request_body: str
    status_code: int
    content_type: str
    content: bytes

    @property
    def url_base(self) -> str:
        parts = urlsplit(self.url)
        return f"{parts.scheme}://{parts.netloc}{parts.path}"


def load_batch_exchanges(case: str, catalogs: list[str] | None = None) -> list[BatchExchange]:
    folder = BATCH_FIXTURES / case
    names = {safe_name(c) for c in catalogs} if catalogs is not None else None
    out: list[BatchExchange] = []
    for meta_path in sorted(folder.glob("*.json")):
        if names is not None and meta_path.stem not in names:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(meta["exchanges"]):
            out.append(BatchExchange(
                method=item["method"], url=item["url"], request_content_type=item.get("request_content_type", ""),
                fields=item.get("fields"), request_body=item.get("request_body", ""), status_code=item["status_code"],
                content_type=item["content_type"], content=(folder / f"{meta_path.stem}.{idx}.body").read_bytes(),
            ))
    return out


def request_body(request: httpx.Request) -> bytes:
    """The request body (multipart bodies are streams that must be read first)."""
    try:
        return request.content
    except httpx.RequestNotRead:
        return request.read()


def batch_replay_side_effect(exchanges: list[BatchExchange], cone_exchanges: list[Any] | None = None):
    """respx side effect: multipart requests matched on their fields, others via fixture_io signatures."""
    from fixture_io import replay_side_effect

    cone_handler = replay_side_effect(cone_exchanges) if cone_exchanges else None

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            sent = multipart_fields(ctype, request_body(request))
            candidates = [e for e in exchanges if e.method == request.method and e.url_base == base and e.fields == sent]
            if not candidates:
                recorded = [e.fields for e in exchanges if e.url_base == base]
                raise FixtureMismatch(f"multipart request to {base} matches no recording.\nsent: {sent}\n"
                                      f"recorded: {recorded[-1] if recorded else 'none'}")
            chosen = candidates[-1]
            return httpx.Response(chosen.status_code, headers={"content-type": chosen.content_type}, content=chosen.content)
        if cone_handler is not None:
            return cone_handler(request)
        sent_sig = request_signature(request.url.query, request_body(request) or b"")
        candidates = [e for e in exchanges if e.method == request.method and e.url_base == base
                      and request_signature(urlsplit(e.url).query, e.request_body) == sent_sig]
        if not candidates:
            raise FixtureMismatch(f"request {request.method} {base} matches no recording: {sent_sig}")
        chosen = candidates[-1]
        return httpx.Response(chosen.status_code, headers={"content-type": chosen.content_type}, content=chosen.content)

    return handler


async def record_case(case: str, catalogs: list[str] | None = None) -> None:
    """Record one case: a separate client per catalog so exchanges are filed by catalog."""
    from batch import BatchCrossmatcher

    spec = CASES[case]
    folder = BATCH_FIXTURES / case
    folder.mkdir(parents=True, exist_ok=True)
    for catalog in catalogs or spec["catalogs"]:
        log: list[tuple[httpx.Request, httpx.Response]] = []

        async def hook(response: httpx.Response, log: list = log) -> None:
            await response.aread()
            await response.request.aread()
            log.append((response.request, response))

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
            engine = BatchCrossmatcher(client=client, fallback_to_cone=False)
            result = await engine.run(spec["targets"], [catalog], radius_arcsec=spec["radius"])
        run = result.runs[catalog]
        good = [(req, resp) for req, resp in log if resp.status_code == 200]
        if run.errors or not good:
            print(f"{case:<8} {catalog:<22} NOT SAVED: {run.errors}")
            continue
        name = safe_name(catalog)
        for old in folder.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, (req, resp) in enumerate(good):
            ctype = req.headers.get("content-type", "")
            body = req.content or b""
            (folder / f"{name}.{idx}.body").write_bytes(redact(resp.content))
            exchanges.append({
                "method": req.method, "url": str(req.url), "request_content_type": ctype.split(";")[0],
                "fields": multipart_fields(ctype, body) if ctype.startswith("multipart/form-data") else None,
                "request_body": "" if ctype.startswith("multipart/form-data") else body.decode("utf-8", "replace"),
                "status_code": resp.status_code, "content_type": resp.headers.get("content-type", ""),
            })
        meta = {"case": case, "catalog": catalog, "radius_arcsec": spec["radius"], "targets": spec["targets"],
                "strategy": run.strategy, "exchanges": exchanges}
        (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{case:<8} {catalog:<22} {run.strategy:<7} {run.requests} request(s), {run.rows_returned} rows, "
              f"{run.total_matches} matches")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "record":
        print(__doc__)
        return
    cases = [a for a in args[1:] if a in CASES] or list(CASES)
    catalogs = [a for a in args[1:] if a not in CASES] or None
    for case in cases:
        asyncio.run(record_case(case, catalogs))


if __name__ == "__main__":
    main()
