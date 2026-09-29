"""LIVE tests of the local sky mirror against the real archives (run with ``-m live``).

Mirrors 0.2 deg around 3C 273 from Gaia DR3 (ESA Gaia TAP), 2MASS PSC (IRSA TAP), Pan-STARRS
DR2 (MAST catalogs API) and SDSS DR18 (SkyServer SqlSearch), then checks that LocalProvider
answers several cones inside the region exactly like the remote adapters (same ids, positions
within 1 mas, the archives' own distance columns within float/print precision) and faster:
the local lookup takes < 50 ms even for a max_rows-sized (truncated) cone.

Also the fixture recorder for the offline replay tests (tests/test_skycache_replay.py)::

    .venv/Scripts/python.exe tests/test_skycache_live.py record [catalog ...] [--mirror-only]

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

import astropy.units as u
import httpx
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fixture_io import FIXTURES, TARGETS, redact

from models import CatalogRegistry, CatalogUnavailableError, QueryTimeoutError, validate_target
from providers import CacheManager, provider_map
from skycache import (
    CoverageError,
    HatsRemoteProvider,
    LocalProvider,
    MirrorError,
    SkyCache,
    _chord_distance,
    _query_relative_values,
    chord_angle_deg,
    lsdb_available,
    mirror_region,
)

MIRROR_CATALOGS = ["gaia_dr3", "twomass_psc", "panstarrs_dr2", "sdss"]
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
    # More than max_rows (200) Gaia DR3 sources: a truncated TOP 200 (+1 probe) answer.
    ("wide_600", CENTER[0], CENTER[1], 600.0, None),
]
# Answers converting up to SMALL_ANSWER archive rows (returned + epoch-pad + excess rows) must
# take < 50 ms warm in total. Each row's conversion
# (providers' normalize_source_record, identical to the remote path) costs ~0.5 ms, so for
# larger answers (up to the max_rows-sized cone) the local lookup itself is held to 50 ms and
# the total to less than the archive's answer time.
SMALL_ANSWER = 50
# Crosses the edge of the mirrored cone: must raise CoverageError.
PARTIAL_CONE = (CENTER[0], CENTER[1] + 0.19, 120.0)

MIRROR_SET = "skycache/3c273_mirror"
CONES_SET = "skycache/3c273_cones"

# Astrophysical truth for 3C 273 (Gaia DR3 source; 2MASS PSC designation encodes
# RA 12h29m06.69s, Dec +02d03'08.5" of the quasar; Skrutskie et al. 2006).
GAIA_3C273 = "3700386905605055360"
TWOMASS_3C273 = "12290669+0203085"
PS1_3C273 = "110461872779253351"  # PSO J187.2779+02.0524
SDSS_3C273 = "1237651735760142397"  # DR18 PhotoPrimary


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


async def record(catalogs: list[str] | None = None, *, cones: bool = True) -> None:
    registry = CatalogRegistry()
    for name in catalogs or MIRROR_CATALOGS:
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
        if not cones:
            continue
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


# Archive-computed distances vs the separation computed here (arcsec): IRSA prints 6 decimals
# and the adapters send centres with 9 decimals (<= 1.8 uas). IRSA TAP's DISTANCE() is the chord
# 2 sin(theta/2), not the arc (skycache._chord_distance); the others return the arc. Local values
# are recomputed (haversine / chord) to float precision.
ARCHIVE_DIST_ABS_ARCSEC = 3e-6
LOCAL_DIST_ABS_ARCSEC = 1e-8


def compare_results(local, remote, *, catalog: str) -> None:
    """Same sources in the same order; positions within 1 mas; same raw data and metadata.

    The archive's query-relative distance columns (TAP match_dist, Gator dist/angle, MAST
    distance, SDSS dist_arcsec) must be present locally; local values and the archive's must both
    equal the separation from the cone centre computed here (astropy Vincenty; the chord for IRSA
    TAP) -- local ones to float precision, the archive's to its printing precision.
    """
    assert [s.source_id for s in local] == [s.source_id for s in remote], catalog
    definition = CatalogRegistry().get(catalog.split("/")[0])
    center = remote.meta["cone_center"]
    units = {name: unit for name, _v, unit in _query_relative_values(definition, definition.provider, center[0],
                                                                      center[1], np.zeros(1), np.zeros(1))}
    for a, b in zip(local, remote):
        sep_mas = math.hypot((a.ra - b.ra) * math.cos(math.radians(b.dec)), a.dec - b.dec) * 3.6e6
        assert sep_mas < 1.0
        assert a.epoch == b.epoch and a.positional_error_arcsec == b.positional_error_arcsec
        assert a.proper_motion_ra_masyr == b.proper_motion_ra_masyr and a.epoch_range == b.epoch_range
        assert a.metadata == b.metadata
        da, db = dict(a.data), dict(b.data)
        arc = SkyCoord(center[0] * u.deg, center[1] * u.deg).separation(SkyCoord(b.ra * u.deg, b.dec * u.deg))
        chord = definition.provider == "tap" and _chord_distance(definition)
        exact = float(chord_angle_deg(arc.deg)) * 3600.0 if chord else arc.arcsec
        for name in ("match_dist", "dist", "angle", "distance", "dist_arcsec"):
            if name in db:
                assert name in da, f"{catalog}: query-relative column {name!r} missing locally"
                local_value, remote_value = da.pop(name), db.pop(name)
                if name == "angle":  # Gator position angle (deg), printed with 6 decimals
                    assert math.isclose(local_value, remote_value, rel_tol=0.0, abs_tol=1.5e-6), name
                    continue
                scale = 3600.0 if units[name] == "deg" else 1.0
                assert math.isclose(local_value * scale, exact, rel_tol=0.0, abs_tol=LOCAL_DIST_ABS_ARCSEC), name
                assert math.isclose(remote_value * scale, exact, rel_tol=0.0, abs_tol=ARCHIVE_DIST_ABS_ARCSEC), (
                    name, remote_value, exact)
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
                local_runs, lookups = [], []
                for _ in range(5):
                    t0 = time.perf_counter()
                    got = await local.query(definition, target, radius)
                    local_runs.append((time.perf_counter() - t0) * 1000.0)
                    lookups.append(got.meta["skycache"]["lookup_ms"])
                compare_results(got, want, catalog=f"{catalog}/{label}")
                timings.append((label, len(got), remote_ms, local_runs[0], statistics.median(local_runs[1:]),
                                statistics.median(lookups[1:]), got.meta["raw_row_count"]))
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
    for label, n, remote_ms, cold_ms, warm_ms, lookup_ms, raw_rows in timings:
        print(f"  {label:<10} {n:>4} rows ({raw_rows:>3} converted)  remote {remote_ms:8.1f} ms  "
              f"local first {cold_ms:6.2f} ms  warm {warm_ms:6.2f} ms (lookup {lookup_ms:5.2f} ms)")
        if raw_rows <= SMALL_ANSWER:
            assert warm_ms < 50.0, (label, raw_rows, warm_ms)
        assert lookup_ms < 50.0, (label, lookup_ms)
        assert warm_ms < remote_ms, (label, warm_ms, remote_ms)
    if catalog == "gaia_dr3":
        assert next(n for label, n, *_ in timings if label == "wide_600") == 200  # truncated at max_rows
    # Astrophysical truth: 3C 273 itself is the nearest source to its SIMBAD position.
    nearest = asyncio.run(local.query(definition, validate_target(*CENTER), 2.0))
    if catalog == "gaia_dr3":
        assert nearest[0].source_id == GAIA_3C273
        assert 12.5 < nearest[0].data["phot_g_mean_mag"] < 13.3  # the quasar, V ~ 12.9
    elif catalog == "twomass_psc":
        assert nearest[0].source_id == TWOMASS_3C273
        assert 9.5 < nearest[0].data["k_m"] < 10.5  # K ~ 10 (Skrutskie et al. 2006 PSC)
    elif catalog == "panstarrs_dr2":
        assert nearest[0].source_id == PS1_3C273
        assert 12.3 < nearest[0].data["rMeanPSFMag"] < 13.3  # r_PSF ~ 12.8
    else:
        assert nearest[0].source_id == SDSS_3C273
        assert 12.3 < nearest[0].data["psfMag_r"] < 13.3  # r_PSF ~ 12.7
    assert nearest[0].metadata["query_separation_arcsec"] < 0.5


@pytest.mark.live
def test_live_overlapping_refresh_keeps_each_source_once_and_lsdb_reads_match_the_archive_in_coverage(tmp_path):
    """Two overlapping Gaia DR3 mirrors (the second refreshes part of the first): every source
    once, only covered rows stored, local cones across both equal ESA's answer, and an lsdb
    cone at the coverage edge returns exactly the archive's rows inside the coverage."""
    radius = 0.05
    second = (CENTER[0] + 0.06 / math.cos(math.radians(CENTER[1])), CENTER[1])
    # 45" inside A's west edge: a 150" cone there straddles the coverage edge (and its uncovered rim).
    edge = (CENTER[0] - (radius - 45.0 / 3600.0) / math.cos(math.radians(CENTER[1])), CENTER[1])

    async def scenario():
        definition = CatalogRegistry().get("gaia_dr3")
        store = SkyCache(tmp_path / "store")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
            reports = [await mirror_region(definition, ra=ra, dec=dec, radius_deg=radius, store=store,
                                           providers=providers, timeout=120.0) for ra, dec in (CENTER, second)]
            remote = providers["tap"]
            overlap = ((CENTER[0] + second[0]) / 2.0, CENTER[1])
            want_overlap = await remote.query(definition, validate_target(*overlap), 60.0)
            want_edge = await remote.query(definition, validate_target(*edge), 150.0)
        return definition, store, reports, overlap, want_overlap, want_edge

    try:
        definition, store, reports, overlap, want_overlap, want_edge = asyncio.run(scenario())
    except MirrorError as exc:
        _network_skip(exc)
    except (httpx.HTTPError, CatalogUnavailableError, QueryTimeoutError) as exc:
        _network_skip(exc)
    assert all(r.tiles_failed == 0 and r.region_covered_fraction == 1.0 for r in reports)
    assert reports[1].rows_replaced > 0 and reports[1].rows_outside_coverage > 0
    rows = store.load_rows("gaia_dr3")
    ids = [r.source_id for r in rows]
    assert len(ids) == len(set(ids)) == reports[1].rows_total
    coverage = store.coverage("gaia_dr3")
    assert coverage.contains_points([r.ra for r in rows], [r.dec for r in rows]).all()
    got = asyncio.run(LocalProvider(store).query(definition, validate_target(*overlap), 60.0))
    assert got.meta["skycache"]["hit"] is True and len(got) >= 3
    compare_results(got, want_overlap, catalog="gaia_dr3/overlap")
    if lsdb_available():
        import lsdb

        frame = lsdb.open_catalog(store.catalog_path("gaia_dr3"),
                                  search_filter=lsdb.ConeSearch(ra=edge[0], dec=edge[1], radius_arcsec=150.0)).compute()
        in_cov = coverage.contains_points([s.ra for s in want_edge], [s.dec for s in want_edge])
        assert not all(in_cov) and any(in_cov)  # the cone straddles the coverage edge
        assert sorted(frame["source_id"].astype(str).tolist()) == sorted(
            s.source_id for s, inside in zip(want_edge, in_cov) if inside)
        assert not store.covers("gaia_dr3", edge[0], edge[1], 150.0).covered


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
        # float32 columns come out as the TAP service prints them (not widened to float64 noise digits)...
        for name in GAIA_FLOAT32_COLUMNS:
            assert a.data[name] == b.data[name], (name, a.data[name], b.data[name])
        assert a.positional_error_arcsec == b.positional_error_arcsec
        # ...float64 columns are each service's own doubles, which differ in the last digits (measured up to
        # ~3e-15 relative, e.g. parallax -0.0159502908342347 vs -0.015950290834234743 printed by TAP).
        for name, value in b.data.items():
            if isinstance(value, float) and name not in GAIA_FLOAT32_COLUMNS and name != "match_dist":
                assert math.isclose(a.data[name], value, rel_tol=1e-14, abs_tol=0.0), name
            elif not isinstance(value, float):
                assert a.data[name] == value, name


# Gaia DR3 gaia_source columns stored as float32 (single precision) among the registry's columns.
GAIA_FLOAT32_COLUMNS = ("ra_error", "dec_error", "ra_dec_corr", "parallax_error", "phot_g_mean_mag",
                        "phot_bp_mean_mag", "phot_rp_mean_mag", "ruwe")


@pytest.mark.live
def test_live_simbad_output_limit_and_maxrec_through_the_mirror_adapter():
    """SIMBAD caps JSON answers at its default output limit (50000 rows) unless MAXREC asks for more:
    mirror requests send MAXREC = TOP, and the limit is read from the service's capabilities."""
    from skycache import _mirror_adapters, _tap_output_limits

    async def scenario():
        definition = CatalogRegistry().get("simbad")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
            mirror_tap = _mirror_adapters(providers)["tap"]
            limits = await _tap_output_limits(mirror_tap, definition, 60.0)
            adql = "SELECT TOP 50002 oid FROM basic"
            counts = {}
            for label, adapter in (("service", providers["tap"]), ("mirror", mirror_tap)):
                form = adapter.build_request(definition, adql)
                response = await client.post(definition.endpoint, data=form)
                response.raise_for_status()
                counts[label] = (len(response.json()["data"]), form.get("MAXREC"))
            return limits, counts

    try:
        limits, counts = asyncio.run(scenario())
    except (httpx.HTTPError, CatalogUnavailableError, QueryTimeoutError) as exc:
        _network_skip(exc)
    assert 50000 in limits
    assert counts["service"] == (50000, None)  # silently capped: no status in JSON
    assert counts["mirror"] == (50002, "50002")


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        names = [a for a in sys.argv[2:] if not a.startswith("--")]
        asyncio.run(record(names or None, cones="--mirror-only" not in sys.argv))
    else:
        print(__doc__)
