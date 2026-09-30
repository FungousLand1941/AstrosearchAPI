"""Record real hips2fits / MocServer exchanges for the offline imaging tests.

    .venv/Scripts/python.exe tests/fixtures/imaging/record_imaging.py              # every scenario
    .venv/Scripts/python.exe tests/fixtures/imaging/record_imaging.py NAME [NAME]  # selected scenarios

Each scenario is stored in the fixture_io layout (``<scenario>.json`` manifest with
the request method, URL, status and content type + ``<scenario>.<n>.body`` raw
bytes), so :func:`fixture_io.load_exchanges` / :func:`fixture_io.replay_side_effect`
replay them strictly (same URL base and query parameters).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from imaging import CutoutCache, CutoutRequest, CutoutService, ImagingError

# 3C 273 and M87 (SIMBAD ICRS J2000), and an empty southern field outside SDSS.
C3C273 = (187.2779154, 2.0523883)
M87 = (187.7059308, 12.3911233)
SOUTH = (100.0, -70.0)

CUTOUTS: dict[str, dict[str, Any]] = {
    "dss2_3c273_png": {"ra": C3C273[0], "dec": C3C273[1], "survey": "dss2", "fov_arcmin": 2.0, "width": 128, "height": 128, "format": "png"},
    "nvss_3c273_fits": {"ra": C3C273[0], "dec": C3C273[1], "survey": "nvss", "fov_arcmin": 6.0, "width": 64, "height": 64, "format": "fits"},
    "panstarrs_m87_jpg": {"ra": M87[0], "dec": M87[1], "survey": "panstarrs", "fov_arcmin": 3.0, "width": 200, "height": 100,
                              "format": "jpg"},
    "sdss_south_png": {"ra": SOUTH[0], "dec": SOUTH[1], "survey": "sdss", "fov_arcmin": 6.0, "width": 64, "height": 64, "format": "png"},
    "unknown_hips": {"ra": C3C273[0], "dec": C3C273[1], "survey": "BOGUS/P/nothing", "fov_arcmin": 2.0, "width": 64, "height": 64,
                         "format": "png"},
    "vlass_3c273_png": {"ra": C3C273[0], "dec": C3C273[1], "survey": "vlass", "fov_arcmin": 2.0, "width": 64, "height": 64,
                            "format": "png"},
    # VLASS where its MOC has data (the Crab): alasky HTTP 500, then a blank PNG from alaskybis.
    "vlass_crab_png": {"ra": 83.63308, "dec": 22.0145, "survey": "vlass", "fov_arcmin": 2.0, "width": 64, "height": 64,
                       "format": "png"},
    # A colour HiPS as FITS (8-bit RGBA display cube) and its single-band science companion (float pixels).
    "2mass_color_3c273_fits": {"ra": C3C273[0], "dec": C3C273[1], "survey": "2mass", "fov_arcmin": 2.0, "width": 32,
                                   "height": 32, "format": "fits"},
    # An ad-hoc (uncatalogued) colour HiPS as FITS: labelled from the returned content, not the key.
    "mellinger_3c273_fits": {"ra": C3C273[0], "dec": C3C273[1], "survey": "CDS/P/Mellinger/color", "fov_arcmin": 60.0,
                             "width": 32, "height": 32, "format": "fits"},
    "2mass_k_3c273_fits": {"ra": C3C273[0], "dec": C3C273[1], "survey": "2mass_k", "fov_arcmin": 2.0, "width": 32,
                               "height": 32, "format": "fits"},
    # JPEG blanks. VLASS at the Crab: alasky HTTP 500, alaskybis a uniformly white JPEG, the MOC overlaps,
    # and the same cutout as PNG from alaskybis is fully transparent: a rendering failure.
    "vlass_crab_jpg": {"ra": 83.63308, "dec": 22.0145, "survey": "vlass", "fov_arcmin": 2.0, "width": 64, "height": 64,
                       "format": "jpg"},
    # Pan-STARRS1 never observed (100, -70): a uniformly white JPEG that the PS1 MOC confirms as "no data".
    "ps1_south_jpg": {"ra": SOUTH[0], "dec": SOUTH[1], "survey": "panstarrs", "fov_arcmin": 6.0, "width": 64,
                      "height": 64, "format": "jpg"},
    # A flat but real field: RASS counts are all 0 in 12", rendered uniformly white with cmap=Greys (like a
    # blank); the PNG's alpha plane shows data pixels, so it is kept as data.
    "rass_flat_greys_jpg": {"ra": 187.0, "dec": 40.0, "survey": "rass", "fov_arcmin": 0.2, "width": 32, "height": 32,
                            "format": "jpg", "cmap": "Greys"},
    # A colour HiPS FITS where the first single-band companion (DR10 r) has no data but g does.
    "legacy_gap_color_fits": {"ra": 345.0, "dec": -35.0, "survey": "legacy", "fov_arcmin": 1.0, "width": 16,
                              "height": 16, "format": "fits"},
}
#: Barnard's star, SIMBAD ICRS J2000 position (moves 10.4"/yr; proper motion given to the stack).
BARNARD = (269.45207696, 4.69336497)
#: A DESI Legacy Surveys DR10 field where the colour HiPS and g/i/z have data but DR10 r does not.
LEGACY_GAP = (345.0, -35.0)
STACKS: dict[str, tuple[float, float]] = {"coverage_3c273": C3C273, "coverage_south": SOUTH,
                                          "coverage_legacy_gap": LEGACY_GAP, "coverage_barnard": BARNARD}
#: Cutouts from one hips2fits endpoint only: (request spec, endpoints). VLASS at the Crab: the MOC covers
#: the position, but alaskybis answers a blank PNG (and alasky HTTP 500): a rendering failure.
CRAB = (83.63308, 22.0145)
SINGLE_ENDPOINT_CUTOUTS: dict[str, tuple[dict[str, Any], list[str]]] = {
    "vlass_crab_png_bis": ({"ra": CRAB[0], "dec": CRAB[1], "survey": "vlass", "fov_arcmin": 2.0, "width": 64,
                            "height": 64, "format": "png"},
                           ["https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"]),
}
#: Explicit-survey stacks: (ra, dec, surveys). Ad-hoc HiPS IDs are described from the MocServer.
EXPLICIT_STACKS: dict[str, tuple[float, float, list[str]]] = {
    "stack_explicit_3c273": (C3C273[0], C3C273[1], ["rass", "2mass", "vlass", "dss2", "CDS/P/Mellinger/color"]),
}


def save(name: str, log: list[tuple[httpx.Request, httpx.Response]], extra: dict[str, Any]) -> None:
    for old in HERE.glob(f"{name}.*.body"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (HERE / f"{name}.{idx}.body").write_bytes(response.content)
        exchanges.append({
            "method": request.method,
            "url": str(request.url),
            "request_body": "",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "match": [],
        })
    (HERE / f"{name}.json").write_text(json.dumps({**extra, "exchanges": exchanges}, indent=2), encoding="utf-8")


async def record_one(name: str, runner, endpoints: list[str] | None = None) -> None:
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        service = CutoutService(client, cache=CutoutCache(HERE / ".nocache", ttl_seconds=0), endpoints=endpoints)
        try:
            outcome = await runner(service)
        except ImagingError as exc:
            outcome = f"{type(exc).__name__}: {exc}"
    save(name, log, {"scenario": name, "recorded_outcome": outcome})
    print(f"{name:<22} {[r.status_code for _, r in log]} {outcome}")


async def main(selected: set[str]) -> None:
    def wanted(name: str) -> bool:
        return not selected or name in selected

    for name, spec in CUTOUTS.items():
        if not wanted(name):
            continue

        async def run(service: CutoutService, spec: dict[str, Any] = spec) -> str:
            cutout = await service.cutout(CutoutRequest(**spec))
            return (f"{cutout.width}x{cutout.height} coverage={cutout.coverage_fraction} blank={cutout.blank} "
                    f"degraded={cutout.degraded} science={cutout.science.key if cutout.science else None}")

        await record_one(name, run)
    for name, (spec, endpoints) in SINGLE_ENDPOINT_CUTOUTS.items():
        if not wanted(name):
            continue

        async def run_single(service: CutoutService, spec: dict[str, Any] = spec) -> str:
            cutout = await service.cutout(CutoutRequest(**spec))
            return (f"{cutout.width}x{cutout.height} coverage={cutout.coverage_fraction} blank={cutout.blank} "
                    f"degraded={cutout.degraded} science={cutout.science.key if cutout.science else None}")

        await record_one(name, run_single, endpoints)
    for name, (ra, dec) in STACKS.items():
        if not wanted(name):
            continue

        async def run_stack(service: CutoutService, ra: float = ra, dec: float = dec) -> str:
            panels, coverage = await service.plan_stack(ra, dec)
            return (f"checked={coverage.checked} panels="
                    f"{[(p.survey, p.fits_survey, p.fits_note) for p in panels]}")

        await record_one(name, run_stack)
    for name, (ra, dec, surveys) in EXPLICIT_STACKS.items():
        if not wanted(name):
            continue

        async def run_explicit(service: CutoutService, ra: float = ra, dec: float = dec,
                               surveys: list[str] = surveys) -> str:
            panels, coverage = await service.plan_stack(ra, dec, surveys)
            return (f"checked={coverage.checked} panels="
                    f"{[(p.survey, p.regime, p.wavelength, p.in_coverage) for p in panels]}")

        await record_one(name, run_explicit)


if __name__ == "__main__":
    asyncio.run(main(set(sys.argv[1:])))
