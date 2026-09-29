"""Offline regression tests of the sky mirror (skycache.py): third adversarial review.

Covers: refreshes after the archive revised positions (each covered cell holds the rows of
one snapshot; sources moved across the edge of the refreshed cells are kept once), rows
outside the coverage never stored (HATS/lsdb readers and the coverage MOC FITS file),
mirrors running off the caller's event loop (dense tiles through the real MAST adapter,
cancellation), the archive's own error in a breaker give-up, explicit request timeouts,
CLI read errors of a damaged store, and the hierarchical inside-cone pixel search.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import threading
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest
import respx
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import shared_ssl_context
from test_skycache import (
    SynthCatalog,
    SyntheticTap,
    assert_identical,
    brute_force,
    cap_points,
    covered_rows,
    run,
    stored_ids,
    synth_definition,
)
from test_skycache_archives import CENTER, MastArchive, Sky, assert_local_equals_remote

import skycache
from models import DEFAULT_CATALOGS, CatalogRegistry, catalog_from_dict, validate_target
from providers import CacheManager, provider_map
from skycache import (
    LocalProvider,
    MirrorError,
    Moc,
    SkyCache,
    angular_sep_deg,
    healpix29,
    mirror_region,
    pixel_geometry,
    pixels_inside_cone,
)


@pytest.fixture
def synth():
    return synth_definition()[0]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(skycache, "RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(skycache, "REPLACE_DELAY_S", 0.0)
    monkeypatch.setattr(skycache, "READ_RETRY_DELAY_S", 0.0)
    monkeypatch.setenv("PROVIDER_REQUESTS_PER_SECOND", "10000")


def _mirror(store, provider, definition, **kwargs):
    return run(mirror_region(definition, store=store, providers={"tap": provider}, **kwargs))


def _layers(store, name="synth"):
    """Coverage layers, newest first: [(retrieved_at, Moc)]."""
    return [(layer["retrieved_at"], Moc.from_json(layer)) for layer in store.metadata(name)["coverage_ages"]]


def _positions_by_id(store, name="synth"):
    out: dict[str, list[tuple[float, float]]] = {}
    for row in store.load_rows(name):
        out.setdefault(row.source_id, []).append((row.data["ra"], row.data["dec"]))
    return out


def _offset(ra, dec, arcsec, angle_deg):
    """A position ``arcsec`` from (ra, dec) towards position angle ``angle_deg`` (small offsets)."""
    a = math.radians(angle_deg)
    d = arcsec / 3600.0
    return (ra + d * math.sin(a) / math.cos(math.radians(dec))) % 360.0, dec + d * math.cos(a)


# ---------------------------------------------------------------------------
# Refreshes after the archive revised positions: one snapshot per cell, each source once
# ---------------------------------------------------------------------------


def test_refresh_after_upstream_moves_stores_every_source_once_from_the_snapshot_of_its_cell(tmp_path, synth):
    """The reviewed failures (dup_subrefresh / dup_moved): an astrometric update between two
    mirrors must never leave a source stored twice (or lose it) in covered cells."""
    # Sources placed on both sides of the edge of the cells the refresh B will cover: 6 in the
    # rim of B's cone (outside its cells) that the archive later moves 25" inward, and 6 in its
    # cells that move outward into the rim.
    b_centre, b_radius = (10.02, 0.01), 0.05
    cov_order = skycache.order_for_resolution(b_radius * skycache.COVERAGE_RES_FRACTION, hi=skycache.MAX_COVERAGE_ORDER)
    b_cells = Moc.from_cone(*b_centre, b_radius, cov_order)  # what the refresh will cover
    starts, ends = [], []
    for angle in np.arange(0.0, 360.0, 7.0):
        rim = _offset(*b_centre, b_radius * 3600.0 - 1.0, angle)
        deep = _offset(*b_centre, b_radius * 3600.0 - 26.0, angle)
        if b_cells.contains_points([rim[0]], [rim[1]])[0] or not b_cells.contains_points([deep[0]], [deep[1]])[0]:
            continue
        if len(starts) < 6:  # moves in
            starts.append(rim)
            ends.append(deep)
        elif len(starts) < 12:  # moves out
            starts.append(deep)
            ends.append(rim)
    assert len(starts) == 12
    base_ra, base_dec = cap_points(10.0, 0.0, 0.35, 7000, seed=5)
    cat = SynthCatalog(np.append(base_ra, [p[0] for p in starts]), np.append(base_dec, [p[1] for p in starts]), seed=1)
    store = SkyCache(tmp_path / "store", partition_rows=400)
    first = _mirror(store, SyntheticTap(cat), synth, ra=10.0, dec=0.0, radius_deg=0.2, tile_rows=5000)
    assert first.queries == 1 and first.region_covered_fraction == 1.0
    old = copy.deepcopy(cat)

    # Upstream update: every source moves 0.2" in RA, and the 12 above cross the edge.
    cat.ra = (cat.ra + 0.2 / 3600.0) % 360.0
    movers = list(range(cat.ra.size - 12, cat.ra.size))
    for i, (ra, dec) in zip(movers, ends):
        cat.ra[i], cat.dec[i] = ra, dec
    moved_in, moved_out = movers[:6], movers[6:]

    b = _mirror(store, SyntheticTap(cat), synth, ra=b_centre[0], dec=b_centre[1], radius_deg=b_radius, tile_rows=5000)
    assert b.queries == 1 and b.rows_moved >= 12
    # C: an adjacent region overlapping A's east edge, split into tiles whose cones overhang their pixels.
    c = _mirror(store, SyntheticTap(cat), synth, ra=10.26, dec=0.0, radius_deg=0.08, tile_rows=100)
    assert c.queries > 10 and c.tiles_failed == 0 and c.rows_outside_coverage > 0

    layers = _layers(store)
    assert len(layers) == 3
    new_cells = layers[0][1] | layers[1][1]  # B and C: after the update
    old_only = layers[2][1]  # A's cells covered by neither
    assert (new_cells & old_only).empty and (new_cells | old_only) == store.coverage("synth")
    assert b_cells.difference(new_cells).empty
    cur_new = new_cells.contains_points(cat.ra, cat.dec)
    old_new = new_cells.contains_points(old.ra, old.dec)
    cur_old = old_only.contains_points(cat.ra, cat.dec)
    old_old = old_only.contains_points(old.ra, old.dec)
    expect_current = cur_new | (old_new & cur_old)  # refreshed cells, and sources that moved out of them
    expect_old = old_old & ~cur_new  # untouched cells keep A's snapshot (sources moved in excluded)
    assert (cur_new & old_old).sum() >= 6 and (old_new & cur_old).sum() >= 6

    positions = _positions_by_id(store)
    ids = np.asarray(cat.ids).astype(str)
    assert all(len(p) == 1 for p in positions.values()), [k for k, p in positions.items() if len(p) > 1][:5]
    assert set(positions) == set(ids[expect_current | expect_old])
    for i in np.flatnonzero(expect_current):
        assert positions[ids[i]] == [(float(cat.ra[i]), float(cat.dec[i]))]
    for i in np.flatnonzero(expect_old):
        assert positions[ids[i]] == [(float(old.ra[i]), float(old.dec[i]))]
    for i in moved_in + moved_out:  # a small cone around the source's new position: once, as the archive has it
        got = run(LocalProvider(store).query(synth, validate_target(float(cat.ra[i]), float(cat.dec[i])), 1.0))
        assert got.meta["skycache"]["hit"] is True and [s.source_id for s in got] == [ids[i]]
    # Cones inside one snapshot equal that snapshot's archive answer (and brute force).
    for (ra, dec, radius), archive in (((10.02, 0.01, 90.0), cat), ((10.28, 0.0, 120.0), cat),
                                       ((9.88, -0.02, 150.0), old)):
        target = validate_target(ra, dec)
        got = run(LocalProvider(store).query(synth, target, radius))
        assert_identical(got, run(SyntheticTap(archive).query(synth, target, radius)))
        inside, edge = brute_force(archive, ra, dec, radius)
        assert inside <= {int(s.source_id) for s in got} <= inside | edge


def _edge_crossing(order: int, p: int, q: int, shift_arcsec: float = 0.0):
    """Two positions 0.2" apart on either side of the edge between adjacent pixels p and q
    (in p, in q), where the line joining the pixel centres (both moved ``shift_arcsec``
    north) crosses it."""
    ras, decs, _r = pixel_geometry(order, [p, q])
    start = _offset(ras[0], decs[0], shift_arcsec, 0.0)
    end = _offset(ras[1], decs[1], shift_arcsec, 0.0)
    lon = np.linspace(start[0], end[0], 20001)
    lat = np.linspace(start[1], end[1], 20001)
    pix = healpix29(lon, lat) >> (2 * (29 - order))
    k = int(np.flatnonzero(pix == q)[0])
    assert pix[k - 1] == p
    mid = ((lon[k - 1] + lon[k]) / 2.0, (lat[k - 1] + lat[k]) / 2.0)
    pa = float(skycache.position_angle_deg(start[0], start[1], [end[0]], [end[1]])[0])
    a = _offset(*mid, 0.1, pa + 180.0)
    b = _offset(*mid, 0.1, pa)
    assert healpix29([a[0]], [a[1]])[0] >> (2 * (29 - order)) == p
    assert healpix29([b[0]], [b[1]])[0] >> (2 * (29 - order)) == q
    return a, b


def test_sources_moved_across_the_edge_of_adjacent_pixel_mirrors_are_kept_once(tmp_path, synth):
    """moved_pixels: pixel p mirrored, then its neighbour q whose enclosing cone reaches into p."""
    order = 9
    p = int(healpix29([150.0], [20.0])[0] >> (2 * (29 - order)))
    q = p ^ 1  # NESTED siblings 0 and 1 share an edge
    ras, decs, _r = pixel_geometry(order, [p, q])
    base_ra, base_dec = cap_points(float(ras.mean()), float(decs.mean()), 0.3, 4000, seed=5)
    into_q, out_of_q = _edge_crossing(order, p, q), _edge_crossing(order, p, q, shift_arcsec=60.0)
    extra = [into_q[0], out_of_q[1]]  # x starts in p (moves into q); y starts in q (moves into p)
    cat = SynthCatalog(np.append(base_ra, [e[0] for e in extra]), np.append(base_dec, [e[1] for e in extra]), seed=2)
    x, y = cat.ra.size - 2, cat.ra.size - 1
    store = SkyCache(tmp_path / "store")
    tap = SyntheticTap(cat)
    _mirror(store, tap, synth, healpix_order=order, healpix_pixels=[p, q], tile_rows=100_000)
    assert Counter(stored_ids(store))[str(cat.ids[x])] == 1 and Counter(stored_ids(store))[str(cat.ids[y])] == 1

    # Upstream: x moves from p into q, y from q into p (0.2"); q alone is mirrored again.
    cat.ra[x], cat.dec[x] = into_q[1]
    cat.ra[y], cat.dec[y] = out_of_q[0]
    report = _mirror(store, tap, synth, healpix_order=order, healpix_pixels=[q], tile_rows=100_000)
    assert report.rows_moved == 2
    positions = _positions_by_id(store)
    assert positions[str(cat.ids[x])] == [(float(cat.ra[x]), float(cat.dec[x]))]  # dropped from p, stored in q
    assert positions[str(cat.ids[y])] == [(float(cat.ra[y]), float(cat.dec[y]))]  # kept, now in p's cells
    ids = stored_ids(store)
    assert len(ids) == len(set(ids))
    for i in (x, y):
        target = validate_target(float(cat.ra[i]), float(cat.dec[i]))
        got = run(LocalProvider(store).query(synth, target, 2.0))
        assert got.meta["skycache"]["hit"] is True
        assert_identical(got, run(tap.query(synth, target, 2.0)))


def test_a_move_is_recognised_only_with_archive_ids(tmp_path, synth):
    """Rows without archive ids (numbered per answer) are never paired: a refresh keeps the
    cell snapshots as they are rather than guess."""
    from test_skycache_hardening import IdlessTap

    params = {k: v for k, v in synth.parameters.items() if k != "id_field"}
    params["columns"] = [c for c in params["columns"] if c != "source_id"]
    definition = replace(synth, parameters=params)
    order = 9
    p = int(healpix29([150.0], [20.0])[0] >> (2 * (29 - order)))
    q = p ^ 1
    ras, decs, _r = pixel_geometry(order, [p, q])
    a, b = _edge_crossing(order, p, q)
    base_ra, base_dec = cap_points(float(ras.mean()), float(decs.mean()), 0.3, 1500, seed=7)
    cat = SynthCatalog(np.append(base_ra, a[0]), np.append(base_dec, a[1]))
    store = SkyCache(tmp_path / "store")
    tap = IdlessTap(cat)
    _mirror(store, tap, definition, healpix_order=order, healpix_pixels=[p], tile_rows=100_000)
    cat.ra[-1], cat.dec[-1] = b
    report = _mirror(store, tap, definition, healpix_order=order, healpix_pixels=[q], tile_rows=100_000)
    assert report.rows_moved == 0
    positions = sorted((r.data["ra"], r.data["dec"]) for r in store.load_rows("synth"))
    assert (float(a[0]), float(a[1])) in positions and (float(b[0]), float(b[1])) in positions


def test_deletion_in_a_cell_the_refresh_did_not_cover_keeps_that_cells_snapshot(tmp_path, synth):
    """A refresh replaces the rows of the cells it covers; a cell outside them stays a
    consistent copy of the time its coverage age records (mixing snapshots inside a cell is
    what stored moved sources twice). Refreshing that cell applies the deletion."""
    cat = SynthCatalog(*cap_points(40.0, 5.0, 0.3, 3000, seed=9))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=40.0, dec=5.0, radius_deg=0.2, tile_rows=5000)
    cov_order = skycache.order_for_resolution(0.05 * skycache.COVERAGE_RES_FRACTION, hi=skycache.MAX_COVERAGE_ORDER)
    b_cells = Moc.from_cone(40.0, 5.0, 0.05, cov_order)
    sep = angular_sep_deg(40.0, 5.0, cat.ra, cat.dec)
    rim = next(int(i) for i in np.argsort(-sep) if sep[i] < 0.05 and not b_cells.contains_points([cat.ra[i]], [cat.dec[i]])[0])
    inner = int(np.argmin(sep))
    cat.alive[[rim, inner]] = False
    _mirror(store, SyntheticTap(cat), synth, ra=40.0, dec=5.0, radius_deg=0.05, tile_rows=5000)
    ids = set(stored_ids(store))
    assert str(cat.ids[inner]) not in ids and str(cat.ids[rim]) in ids
    rim_age = [when for when, moc in _layers(store) if moc.contains_points([cat.ra[rim]], [cat.dec[rim]])[0]]
    assert rim_age == [store.metadata("synth")["mirrors"][0]["retrieved_at"]]  # labelled with its snapshot
    _mirror(store, SyntheticTap(cat), synth, ra=40.0, dec=5.0, radius_deg=0.2, tile_rows=5000)
    assert str(cat.ids[rim]) not in set(stored_ids(store))


# ---------------------------------------------------------------------------
# Only covered rows are stored: HATS readers see exactly the archive inside the coverage
# ---------------------------------------------------------------------------


def test_rows_outside_the_coverage_are_never_stored_and_lsdb_reads_match_the_archive_there(tmp_path, synth):
    cat = SynthCatalog(*cap_points(10.0, 0.0, 0.3, 12000, seed=5), seed=1)
    store = SkyCache(tmp_path / "store", partition_rows=500)
    report = _mirror(store, SyntheticTap(cat), synth, ra=10.0, dec=0.0, radius_deg=0.1, tile_rows=100)
    assert report.queries > 10 and report.tiles_failed == 0 and report.rows_outside_coverage > 0
    assert report.rows_fetched == report.rows_stored + report.rows_outside_coverage + report.rows_deduplicated
    coverage = store.coverage("synth")
    covered = covered_rows(store, cat)
    rows = store.load_rows("synth")
    assert coverage.contains_points([r.ra for r in rows], [r.dec for r in rows]).all()
    assert sorted(r.source_id for r in rows) == sorted(np.asarray(cat.ids).astype(str)[covered])

    # The coverage is published as an IVOA MOC 2.0 FITS file next to the HATS tree.
    mocpy = pytest.importorskip("mocpy")
    moc = mocpy.MOC.from_fits(store.catalog_path("synth") / skycache.COVERAGE_MOC_FILE)
    ranges = moc.to_depth29_ranges() if callable(moc.to_depth29_ranges) else moc.to_depth29_ranges
    assert np.array_equal(np.asarray(ranges, dtype=np.int64), coverage.ranges)
    status = store.status()["catalogs"][0]
    assert status["coverage_moc_file"].endswith(skycache.COVERAGE_MOC_FILE)

    # An lsdb cone straddling the coverage edge: exactly the archive's rows inside the coverage.
    lsdb = pytest.importorskip("lsdb")
    outside = np.flatnonzero(~covered & (angular_sep_deg(10.0, 0.0, cat.ra, cat.dec) < 0.105))
    probe = int(outside[0])
    ra, dec = float(cat.ra[probe]), float(cat.dec[probe])
    assert not store.covers("synth", ra, dec, 30.0).covered
    frame = lsdb.open_catalog(store.catalog_path("synth"),
                              search_filter=lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=30.0)).compute()
    inside, edge = brute_force(cat, ra, dec, 30.0)
    in_cov = set(np.asarray(cat.ids)[covered].tolist())
    got = {int(v) for v in frame["_sc_source_id"].tolist()}
    assert inside & in_cov <= got <= (inside | edge) & in_cov
    assert (inside - in_cov) and not (got - in_cov)  # the uncovered part is absent, never partial rows


# ---------------------------------------------------------------------------
# Mirrors run off the caller's event loop
# ---------------------------------------------------------------------------


def _ps1():
    return catalog_from_dict("panstarrs_dr2", copy.deepcopy(DEFAULT_CATALOGS["panstarrs_dr2"]))


def test_dense_tiles_through_the_real_adapter_never_stall_the_callers_loop(tmp_path):
    """loopblock / dense_timeouts: converting a ~4000-row answer took ~3 s ON the caller's loop."""
    definition = _ps1()
    archive = MastArchive(Sky(9000, seed=4))
    hook_threads: set[int] = set()
    gaps: list[float] = []

    async def on_request(request):
        hook_threads.add(threading.get_ident())

    async def scenario():
        stop = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=archive)
            async with httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0,
                                         event_hooks={"request": [on_request]}) as client:
                providers = provider_map(client, cache=CacheManager(None))
                beat = asyncio.create_task(heartbeat())
                try:
                    report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.12,
                                                 store=SkyCache(tmp_path / "store"), providers=providers,
                                                 tile_rows=6000)
                finally:
                    stop.set()
                    await beat
                return report

    report = run(scenario())
    assert report.queries == 1 and report.tiles_failed == 0 and report.rows_fetched > 3500
    assert max(gaps) < 0.5, max(gaps)
    assert hook_threads == {threading.main_thread().ident}  # requests were sent on the caller's loop/client
    store = SkyCache(tmp_path / "store")
    got = run(LocalProvider(store).query(definition, validate_target(*CENTER), 72.0))
    remote_archive = MastArchive(Sky(9000, seed=4))

    async def remote():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=remote_archive)
            async with httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0) as client:
                return await provider_map(client, cache=CacheManager(None))["mast"].query(
                    definition, validate_target(*CENTER), 72.0)

    assert_local_equals_remote(got, run(remote()))


class SlowTap(SyntheticTap):
    def __init__(self, cat, delay):
        super().__init__(cat)
        self.delay = delay

    async def query(self, catalog, target, radius_arcsec):
        await asyncio.sleep(self.delay)
        return await super().query(catalog, target, radius_arcsec)


def test_cancelling_a_mirror_stops_it_and_writes_nothing(tmp_path, synth):
    cat = SynthCatalog(*cap_points(20.0, 20.0, 0.3, 3000, seed=3))
    tap = SlowTap(cat, 0.1)
    store = SkyCache(tmp_path / "store")

    async def scenario():
        task = asyncio.create_task(mirror_region(synth, ra=20.0, dec=20.0, radius_deg=0.2, store=store,
                                                 providers={"tap": tap}, tile_rows=50))
        await asyncio.sleep(0.6)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        calls = len(tap.calls)
        await asyncio.sleep(0.5)
        return calls

    calls = run(scenario())
    assert calls > 1 and len(tap.calls) == calls  # nothing kept running after the cancellation
    assert not store.has("synth") and store.swap_dirs() == []


def test_cancelling_a_mirror_cancels_its_request_on_the_callers_loop(tmp_path):
    definition = _ps1()
    state = {"started": 0, "cancelled": 0}

    async def handler(request):
        state["started"] += 1
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            state["cancelled"] += 1
            raise
        return httpx.Response(503)

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=handler)
            async with httpx.AsyncClient(verify=shared_ssl_context(), timeout=60.0) as client:
                providers = provider_map(client, cache=CacheManager(None))
                task = asyncio.create_task(mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.05,
                                                         store=SkyCache(tmp_path / "store"), providers=providers))
                while not state["started"]:
                    await asyncio.sleep(0.01)
                started = time.perf_counter()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                return time.perf_counter() - started

    elapsed = run(scenario())
    assert state == {"started": 1, "cancelled": 1} and elapsed < 5.0


# ---------------------------------------------------------------------------
# A breaker give-up names the archive's own failure (was flaky: 5 of 11 runs)
# ---------------------------------------------------------------------------


def test_breaker_give_up_always_carries_the_archives_last_error(tmp_path, monkeypatch):
    monkeypatch.setattr(skycache, "MIRROR_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(skycache, "MIRROR_RECOVERY_S", 30.0)  # stays open: every waiting tile gives up
    monkeypatch.setattr(skycache, "CIRCUIT_WAIT_LIMIT_S", 0.3)
    monkeypatch.setattr(skycache, "CIRCUIT_POLL_S", 0.05)
    definition = _ps1()
    archive = MastArchive(Sky(1500, seed=4))
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return archive(request) if calls["n"] == 1 else httpx.Response(503, text="archive down for maintenance")

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=handler)
            async with httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0) as client:
                providers = provider_map(client, cache=CacheManager(None))
                return await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.12,
                                           store=SkyCache(tmp_path / "store"), providers=providers, tile_rows=120)

    for _ in range(3):  # deterministic, not a race
        calls["n"] = 0
        with pytest.raises(MirrorError) as err:
            run(scenario())
        refused = [w for w in str(err.value).split("; order-") if "circuit is open" in w]
        assert refused and all("did not recover within 0.3 s; last archive error:" in w and "HTTP 503" in w
                               for w in refused), refused[:3]


# ---------------------------------------------------------------------------
# mirror_region(timeout=...) bounds every archive request
# ---------------------------------------------------------------------------


def _slow_mast(delay):
    archive = MastArchive(Sky(1500, seed=4))

    async def handler(request):
        await asyncio.sleep(delay)
        return archive(request)

    return handler


def _mast_run(handler, store, **kwargs):
    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=handler)
            async with httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0) as client:
                providers = provider_map(client, cache=CacheManager(None))
                return await mirror_region(_ps1(), ra=CENTER[0], dec=CENTER[1], radius_deg=0.05, store=store,
                                           providers=providers, tile_rows=5000, **kwargs)

    return run(scenario())


def test_explicit_timeout_bounds_requests_of_catalogs_with_longer_registry_timeouts(tmp_path):
    assert _ps1().timeout_seconds == 60.0
    started = time.perf_counter()
    with pytest.raises(MirrorError, match="time budget|timed out"):
        _mast_run(_slow_mast(2.0), SkyCache(tmp_path / "a"), timeout=0.5, max_queries=1)
    assert time.perf_counter() - started < 1.9  # the registry's 60 s did not apply
    report = _mast_run(_slow_mast(1.0), SkyCache(tmp_path / "b"), max_queries=1)  # default: registry timeout
    assert report.queries == 1 and report.tiles_complete == 1


def test_mirror_route_bounds_requests_like_the_services_searches(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYCACHE_MAX_QUERIES", "1")
    _definition, entry = synth_definition()
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(yaml.safe_dump({"catalogs": {"synth": entry}}), encoding="utf-8")
    registry = CatalogRegistry(registry_file)
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    app = FastAPI()
    app.include_router(skycache.router)
    app.state.skycache = SkyCache(tmp_path / "store")
    app.state.registry = registry
    app.state.service = SimpleNamespace(providers={"tap": SlowTap(cat, 1.5)}, registry=registry,
                                        executor=SimpleNamespace(timeout_cap=0.3))
    started = time.perf_counter()
    response = TestClient(app).post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                                     "radius_deg": 0.05})
    assert response.status_code == 502 and "timed out" in response.json()["detail"]
    assert time.perf_counter() - started < 1.4


# ---------------------------------------------------------------------------
# CLI: a damaged store is a read error (exit 1), not an unknown catalog (exit 2)
# ---------------------------------------------------------------------------


def test_cli_cone_reports_a_damaged_store_as_a_read_error(tmp_path, capsys, synth, monkeypatch):
    _definition, entry = synth_definition()
    cat = SynthCatalog(*cap_points(60.0, -30.0, 0.2, 800, seed=43))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=60.0, dec=-30.0, radius_deg=0.15)
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(yaml.safe_dump({"catalogs": {"synth": entry}}), encoding="utf-8")
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(registry_file))
    for leaf in store.catalog_path("synth").rglob("Npix=*.parquet"):  # a leaf rewritten without the spatial index
        table = pq.read_table(leaf)
        pq.write_table(table.drop_columns([skycache.SPATIAL_INDEX_COLUMN]).replace_schema_metadata(table.schema.metadata),
                       leaf)
    parser = argparse.ArgumentParser()
    skycache.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["skycache", "cone", "--catalog", "synth", "--ra", "60", "--dec", "-30",
                              "--radius-arcsec", "30", "--store", str(store.root)])
    assert args.handler(args) == 1
    out = capsys.readouterr().out
    assert "the local store could not be read: KeyError" in out and "unknown catalog" not in out
    args = parser.parse_args(["skycache", "cone", "--catalog", "nope", "--ra", "60", "--dec", "-30",
                              "--radius-arcsec", "30", "--store", str(store.root)])
    assert args.handler(args) == 2 and "unknown catalog 'nope'" in capsys.readouterr().out


def test_mirror_cli_timeout_option(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    skycache.register_cli(parser.add_subparsers(dest="command"))
    seen = {}

    async def fake_mirror(catalog, **kwargs):
        seen.update(kwargs)
        raise MirrorError("stop")

    monkeypatch.setattr(skycache, "mirror_region", fake_mirror)
    monkeypatch.delenv("CATALOG_TIMEOUT_CAP_SECONDS", raising=False)
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    base = ["mirror", "--catalog", "gaia_dr3", "--ra", "1", "--dec", "1", "--radius-deg", "0.1", "--store", str(tmp_path)]
    args = parser.parse_args(base)
    assert args.handler(args) == 1 and seen["timeout"] is None  # registry timeouts govern
    args = parser.parse_args(base + ["--timeout", "12.5"])
    assert args.handler(args) == 1 and seen["timeout"] == 12.5
    monkeypatch.setenv("CATALOG_TIMEOUT_CAP_SECONDS", "20")
    args = parser.parse_args(base)
    assert args.handler(args) == 1 and seen["timeout"] == 20.0


# ---------------------------------------------------------------------------
# Coverage pixels are found hierarchically (one long C call held the GIL for ~0.5 s)
# ---------------------------------------------------------------------------


def test_hierarchical_inside_cone_pixels_equal_the_direct_test():
    rng = np.random.default_rng(11)
    cases = [(0.0, 0.0, 0.2), (359.99, 0.0, 0.4), (10.0, 89.95, 0.3), (200.0, -89.9, 0.25), (187.28, 2.05, 0.2)]
    cases += [(float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
               float(10 ** rng.uniform(-2.5, 0))) for _ in range(25)]
    for ra, dec, radius in cases:
        order = skycache.order_for_resolution(radius * skycache.COVERAGE_RES_FRACTION, hi=skycache.MAX_COVERAGE_ORDER)
        candidates = skycache.cone_pixels(ra, dec, radius, order)
        direct = candidates[skycache._inside_mask(ra, dec, radius, order, candidates)]
        assert np.array_equal(pixels_inside_cone(ra, dec, radius, order), direct), (ra, dec, radius)
        assert Moc.from_cone(ra, dec, radius, order) == Moc.from_pixels(order, direct)
    started = time.perf_counter()
    Moc.from_cone(187.28, 2.05, 1.0, skycache.order_for_resolution(1.0 / 32, hi=skycache.MAX_COVERAGE_ORDER))
    assert time.perf_counter() - started < 0.4  # the direct search: ~0.5 s for a 0.2 deg cone
