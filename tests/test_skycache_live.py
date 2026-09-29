"""LIVE tests of the local sky mirror against the real archives (run with ``-m live``).

Mirrors 0.2 deg around 3C 273 from Gaia DR3 (ESA Gaia TAP) and 2MASS PSC (IRSA TAP), then
checks that LocalProvider answers several cones inside the region exactly like the remote
adapters (same ids, positions within 1 mas) and much faster.

Also the fixture recorder for the offline replay tests (tests/test_skycache_replay.py)::

    .venv/Scripts/python.exe tests/test_skycache_live.py record

stores every real exchange of the mirror queries and the remote cone queries under
``tests/fixtures/skycache/3c273_mirror`` and ``tests/fixtures/skycache/3c273_cones`` in the
``fixture_io`` format.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fixture_io import FIXTURES, TARGETS, redact

from models import CatalogRegistry, CatalogUnavailableError, QueryTimeoutError, validate_target
from providers import CacheManager, provider_map
from skycache import CoverageError, HatsRemoteProvider, LocalProvider, MirrorError, SkyCache, lsdb_available, mirror_region

MIRROR_CATALOGS = ["gaia_dr3", "twomass_psc"]
CENTER = TARGETS["3c273"]  # SIMBAD ICRS J2000 position of 3C 273
MIRROR_RADIUS_DEG = 0.2
# (label, ra, dec, radius_arcsec, target epoch). All lie well inside the 0.2 deg mirror;
# 'epoch2000' is widened by the unknown-proper-motion pad (10.5"/yr x epoch gap).
CONES = [
    ("center_10", CENTER[0], CENTER[1], 10.0, None),
    ("center_90", CENTER[0], CENTER[1], 90.0, None),
    ("north_30", CENTER[0], CENTER[1] + 0.08, 30.0, None),
    ("east_45", CENTER[0] + 0.1, CENTER[1] - 0.05, 45.0, None),
    ("sw_120", CENTER[0] - 0.07, CENTER[1] - 0.06, 120.0, None),
    ("epoch2000", CENTER[0], CENTER[1], 5.0, 2000.0),
]
# Crosses the edge of the mirrored cone: must raise CoverageError.
PARTIAL_CONE = (CENTER[0], CENTER[1] + 0.19, 120.0)

MIRROR_SET = "skycache/3c273_mirror"
CONES_SET = "skycache/3c273_cones"

# Astrophysical truth for 3C 273 (Gaia DR3 source; 2MASS PSC designation encodes
# RA 12h29m06.69s, Dec +02d03'08.5" of the quasar; Skrutskie et al. 2006).
GAIA_3C273 = "3700386905605055360"
TWOMASS_3C273 = "12290669+0203085"


def cone_target(ra, dec, epoch):
    return validate_target(ra, dec, epoch=epoch)


def _network_skip(exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (httpx.HTTPError, CatalogUnavailableError, QueryTimeoutError)) or any(
            tok in text for tok in ("CatalogUnavailableError", "QueryTimeoutError", "HTTP 5", "503", "502", "504",
                                    "ConnectError", "timed out")):
        pytest.skip(f"archive unavailable: {text[:300]}")
    raise exc


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _hooked_client(log: list):
    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    return httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]})


def _save(folder: Path, catalog: str, log: list, extra: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{catalog}.*.body"):
        old.unlink()
    exchanges = []
    for idx, (req, resp) in enumerate(log):
        (folder / f"{catalog}.{idx}.body").write_bytes(redact(resp.content))
        exchanges.append({
            "method": req.method, "url": str(req.url),
            "request_body": req.content.decode("utf-8", "replace") if req.content else "",
            "status_code": resp.status_code, "content_type": resp.headers.get("content-type", ""), "match": [],
        })
    (folder / f"{catalog}.json").write_text(json.dumps({"catalog": catalog, **extra, "exchanges": exchanges}, indent=2),
                                            encoding="utf-8")


async def record() -> None:
    registry = CatalogRegistry()
    for name in MIRROR_CATALOGS:
        definition = registry.get(name)
        with tempfile.TemporaryDirectory() as tmp:
            log: list = []
            async with _hooked_client(log) as client:
                providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
                report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=MIRROR_RADIUS_DEG,
                                             store=SkyCache(tmp), providers=providers, timeout=120.0)
            _save(FIXTURES / MIRROR_SET, name, log, {"ra": CENTER[0], "dec": CENTER[1],
                                                      "radius_deg": MIRROR_RADIUS_DEG, "report": report.as_dict()})
            print(f"mirror {name}: {report.queries} queries, {report.rows_total} rows, providers {report.providers_used}")
        log = []
        async with _hooked_client(log) as client:
            provider = provider_map(client, timeout=120.0, cache=CacheManager(None))[definition.provider]
            for label, ra, dec, radius, epoch in CONES:
                result = await provider.query(definition, cone_target(ra, dec, epoch), radius)
                print(f"cone {name} {label}: {len(result)} rows")
        _save(FIXTURES / CONES_SET, name, log, {"cones": CONES})


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def compare_results(local, remote, *, catalog: str) -> None:
    """Same sources in the same order; positions within 1 mas; same raw data and metadata."""
    assert [s.source_id for s in local] == [s.source_id for s in remote], catalog
    for a, b in zip(local, remote):
        sep_mas = math.hypot((a.ra - b.ra) * math.cos(math.radians(b.dec)), a.dec - b.dec) * 3.6e6
        assert sep_mas < 1.0
        assert a.epoch == b.epoch and a.positional_error_arcsec == b.positional_error_arcsec
        assert a.proper_motion_ra_masyr == b.proper_motion_ra_masyr and a.epoch_range == b.epoch_range
        assert a.metadata == b.metadata
        da, db = dict(a.data), dict(b.data)
        units = {c["name"]: c.get("unit") for c in remote.meta.get("columns") or []}
        for name in ("match_dist", "dist", "angle"):
            if name in db:
                # Server DISTANCE() vs local haversine. IRSA prints arcsec with 6 decimals, so
                # agreement is required to 1 uas (expressed in the column's own unit).
                tol_arcsec = 1.5e-6
                tol = tol_arcsec / 3600.0 if units.get(name) == "deg" else tol_arcsec
                assert math.isclose(da.pop(name), db.pop(name), rel_tol=0.0, abs_tol=tol), name
        assert da == db
    for key in ("status", "row_count", "truncated", "pad_row_count", "excess_row_count", "requested_radius_arcsec",
                "query_radius_arcsec", "warnings"):
        assert local.meta[key] == remote.meta[key], key


@pytest.mark.live
@pytest.mark.parametrize("catalog", MIRROR_CATALOGS)
def test_live_mirror_3c273_local_equals_remote(catalog, tmp_path):
    async def scenario():
        registry = CatalogRegistry()
        definition = registry.get(catalog)
        store = SkyCache(tmp_path / "store")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
            report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=MIRROR_RADIUS_DEG,
                                         store=store, providers=providers, timeout=120.0)
            assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
            local = LocalProvider(store)
            remote = providers[definition.provider]
            timings = []
            for label, ra, dec, radius, epoch in CONES:
                target = cone_target(ra, dec, epoch)
                t0 = time.perf_counter()
                want = await remote.query(definition, target, radius)
                remote_ms = (time.perf_counter() - t0) * 1000.0
                local_runs = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    got = await local.query(definition, target, radius)
                    local_runs.append((time.perf_counter() - t0) * 1000.0)
                compare_results(got, want, catalog=f"{catalog}/{label}")
                timings.append((label, len(got), remote_ms, local_runs[0], statistics.median(local_runs[1:])))
            with pytest.raises(CoverageError):
                await local.query(definition, validate_target(*PARTIAL_CONE[:2]), PARTIAL_CONE[2])
            return report, timings, local, definition

    try:
        report, timings, local, definition = asyncio.run(scenario())
    except MirrorError as exc:
        _network_skip(exc)
    except (httpx.HTTPError, CatalogUnavailableError, QueryTimeoutError) as exc:
        _network_skip(exc)
    print(f"\n{catalog}: mirrored {report.rows_total} rows in {report.queries} queries ({report.elapsed_s:.1f} s)")
    for label, n, remote_ms, cold_ms, warm_ms in timings:
        print(f"  {label:<10} {n:>4} rows  remote {remote_ms:8.1f} ms  local first {cold_ms:6.2f} ms  warm {warm_ms:6.2f} ms")
        assert warm_ms < 50.0, (label, warm_ms)
        assert warm_ms < remote_ms
    # Astrophysical truth: 3C 273 itself is the nearest source to its SIMBAD position.
    nearest = asyncio.run(local.query(definition, validate_target(*CENTER), 2.0))
    if catalog == "gaia_dr3":
        assert nearest[0].source_id == GAIA_3C273
        assert 12.5 < nearest[0].data["phot_g_mean_mag"] < 13.3  # the quasar, V ~ 12.9
    else:
        assert nearest[0].source_id == TWOMASS_3C273
        assert 9.5 < nearest[0].data["k_m"] < 10.5  # K ~ 10 (Skrutskie et al. 2006 PSC)
    assert nearest[0].metadata["query_separation_arcsec"] < 0.5


@pytest.mark.live
@pytest.mark.skipif(not lsdb_available(), reason="lsdb not installed")
def test_live_remote_hats_gaia_equals_gaia_tap():
    """Gaia DR3 cone from the public HATS catalog (lsdb) == the ESA Gaia TAP answer."""

    async def scenario():
        definition = CatalogRegistry().get("gaia_dr3")
        target = validate_target(*CENTER)
        hats = await HatsRemoteProvider().query(definition, target, 20.0)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            tap = await provider_map(client, timeout=120.0, cache=CacheManager(None))["tap"].query(definition, target, 20.0)
        return hats, tap

    try:
        hats, tap = asyncio.run(scenario())
    except Exception as exc:  # noqa: BLE001 - fsspec/aiohttp/dask errors; re-raised unless network
        _network_skip(exc)
    assert [s.source_id for s in hats] == [s.source_id for s in tap]
    assert hats[0].source_id == GAIA_3C273
    for a, b in zip(hats, tap):
        assert abs(a.ra - b.ra) * 3.6e6 < 1.0 and abs(a.dec - b.dec) * 3.6e6 < 1.0
        assert a.epoch == b.epoch == 2016.0
        assert math.isclose(a.positional_error_arcsec, b.positional_error_arcsec, rel_tol=1e-6)  # float32 in HATS


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        asyncio.run(record())
    else:
        print(__doc__)
