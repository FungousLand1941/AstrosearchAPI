"""Offline tests of the local sky mirror (skycache.py) on synthetic catalogs.

A synthetic TAP-like provider answers cones from an in-memory catalog with the archive
semantics the real adapters rely on (TOP row_limit + 1 nearest rows, DISTANCE in deg) and
converts rows with the providers' own ``_sources``. Mirrors are made through the real
mirror pipeline, and local answers are compared with the synthetic "remote" answers and
with brute-force separations computed independently with astropy (Vincenty formula).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.coordinates import SkyCoord
from fastapi import FastAPI
from fastapi.testclient import TestClient

import skycache
from models import DEFAULT_CATALOGS, CatalogRegistry, ColumnMeta, catalog_from_dict, plan_cone, validate_target
from providers import _HTTPProvider
from skycache import (
    CoverageError,
    LocalProvider,
    MirrorError,
    MirrorInputError,
    Moc,
    SkyCache,
    SkyCacheProvider,
    angular_sep_deg,
    cone_pixels,
    healpix29,
    mirror_region,
    partition_orders,
    pixel_ranges,
    pixels_inside_cone,
)

SYNTH_ENDPOINT = "https://synthetic.invalid/tap/sync"


def synth_definition(name: str = "synth", max_rows: int = 200):
    """A Gaia-DR3-like registry definition (epoch column, mas errors, proper motions)."""
    entry = copy.deepcopy(DEFAULT_CATALOGS["gaia_dr3"])
    entry["endpoint"] = SYNTH_ENDPOINT
    entry["table"] = "synthetic.source"
    entry["max_rows"] = max_rows
    entry["parameters"]["columns"] = entry["parameters"]["columns"] + ["flag", "tags"]
    entry["citation"] = "synthetic catalog for tests"
    return catalog_from_dict(name, entry), entry


COLUMNS = [
    ColumnMeta("source_id", None, "meta.id;meta.main", "long"),
    ColumnMeta("ra", "deg", "pos.eq.ra;meta.main", "double"),
    ColumnMeta("dec", "deg", "pos.eq.dec;meta.main", "double"),
    ColumnMeta("ra_error", "mas", "stat.error;pos.eq.ra", "float"),
    ColumnMeta("dec_error", "mas", "stat.error;pos.eq.dec", "float"),
    ColumnMeta("ref_epoch", "yr", "meta.ref;time.epoch", "double"),
    ColumnMeta("parallax", "mas", "pos.parallax.trig", "double"),
    ColumnMeta("pmra", "mas.yr**-1", "pos.pm;pos.eq.ra", "double"),
    ColumnMeta("pmdec", "mas.yr**-1", "pos.pm;pos.eq.dec", "double"),
    ColumnMeta("phot_g_mean_mag", "mag", "phot.mag;em.opt", "float"),
    ColumnMeta("flag", None, "meta.code", "char"),
    ColumnMeta("tags", None, "meta.note", "char"),
    ColumnMeta("match_dist", "deg", None, "double"),
]


class SynthCatalog:
    """Columns of a synthetic catalog (row i has source_id ids[i])."""

    def __init__(self, ra, dec, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.ra = np.mod(np.asarray(ra, dtype=float), 360.0)
        self.dec = np.asarray(dec, dtype=float)
        n = self.ra.size
        self.ids = (4_000_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 7919).tolist()
        self.ra_error = rng.uniform(0.01, 2.0, n).round(4).tolist()
        self.dec_error = rng.uniform(0.01, 2.0, n).round(4).tolist()
        self.pmra = [None if i % 7 == 0 else float(v) for i, v in enumerate(rng.normal(0, 20, n).round(3))]
        self.pmdec = [None if i % 7 == 0 else float(v) for i, v in enumerate(rng.normal(0, 20, n).round(3))]
        self.parallax = [None if i % 5 == 0 else float(v) for i, v in enumerate(rng.uniform(0, 5, n).round(4))]
        self.mag = [None if i % 11 == 0 else float(v) for i, v in enumerate(rng.uniform(8, 21, n).round(4))]
        # Mixed-type and list columns (must round-trip exactly through Parquet).
        self.flag = [(i % 3) if i % 2 else f"F{i % 5}" for i in range(n)]
        self.tags = [None if i % 4 == 0 else [i % 3, "x", 0.5] for i in range(n)]
        self.alive = np.ones(n, dtype=bool)

    def row(self, i: int, match_dist: float) -> dict:
        return {
            "source_id": self.ids[i], "ra": float(self.ra[i]), "dec": float(self.dec[i]),
            "ra_error": self.ra_error[i], "dec_error": self.dec_error[i], "ra_dec_corr": 0.1, "ref_epoch": 2016.0,
            "parallax": self.parallax[i], "parallax_error": 0.05, "pmra": self.pmra[i], "pmdec": self.pmdec[i],
            "phot_g_mean_mag": self.mag[i], "phot_bp_mean_mag": None, "phot_rp_mean_mag": None, "ruwe": 1.0,
            "astrometric_params_solved": 31 if self.pmra[i] is not None else 3,
            "flag": self.flag[i], "tags": self.tags[i], "match_dist": match_dist,
        }


class SyntheticTap(_HTTPProvider):
    """A TAP archive stand-in with the adapters' cone semantics (TOP row_limit+1, nearest first)."""

    provider_name = "tap"

    def __init__(self, cat: SynthCatalog) -> None:
        self.cat = cat
        self.client = None
        self.timeout = 30.0
        self.max_response_bytes = 0
        self.guards = {}
        self.cache = None
        self.calls: list[tuple[float, float, float, int]] = []

    async def query(self, catalog, target, radius_arcsec):
        cone = plan_cone(catalog, target, radius_arcsec)
        self.calls.append((cone.ra, cone.dec, cone.radius_arcsec, cone.row_limit))
        sep = angular_sep_deg(cone.ra, cone.dec, self.cat.ra, self.cat.dec)
        idx = np.flatnonzero((sep <= cone.radius_arcsec / 3600.0) & self.cat.alive)
        idx = idx[np.lexsort((idx, sep[idx]))][: cone.row_limit + 1]
        rows = [self.cat.row(int(i), float(sep[i])) for i in idx]
        return self._sources(
            catalog, rows, radius_arcsec, SYNTH_ENDPOINT, {"QUERY": "synthetic"}, columns=COLUMNS, target=target,
            cone=cone, meta={"query": "synthetic", "endpoint": SYNTH_ENDPOINT, "http_method": "POST", "format": "json",
                             "truncated": False, "query_status": None, "cached": False,
                             "columns": [c.as_dict() for c in COLUMNS]},
        )


def cap_points(ra0: float, dec0: float, radius_deg: float, n: int, seed: int):
    """Uniform random points in a spherical cap (exact, valid across RA=0 and the poles)."""
    rng = np.random.default_rng(seed)
    cos_r = math.cos(math.radians(radius_deg))
    z = rng.uniform(cos_r, 1.0, n)
    phi = rng.uniform(0, 2 * math.pi, n)
    s = np.sqrt(1 - z * z)
    local = np.stack([s * np.cos(phi), s * np.sin(phi), z])
    a, d = math.radians(ra0), math.radians(dec0)
    # Rotate the north-pole cap to (ra0, dec0).
    ry = np.array([[math.sin(d), 0, math.cos(d)], [0, 1, 0], [-math.cos(d), 0, math.sin(d)]])
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    x, y, zz = rz @ ry @ local
    return np.degrees(np.arctan2(y, x)) % 360.0, np.degrees(np.arcsin(np.clip(zz, -1, 1)))


def brute_force(cat: SynthCatalog, ra: float, dec: float, radius_arcsec: float) -> tuple[set, set]:
    """(ids surely inside, ids within 1e-6 arcsec of the edge) by astropy's Vincenty separation."""
    sep = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(cat.ra * u.deg, cat.dec * u.deg)).arcsec
    ids = np.asarray(cat.ids)
    alive = cat.alive
    inside = set(ids[(sep < radius_arcsec - 1e-6) & alive].tolist())
    edge = set(ids[(np.abs(sep - radius_arcsec) <= 1e-6) & alive].tolist())
    return inside, edge


def assert_identical(local, remote) -> None:
    """Local and remote QueryResults hold the same CatalogSources (provenance aside)."""
    assert [s.source_id for s in local] == [s.source_id for s in remote]
    for a, b in zip(local, remote):
        da, db = a.as_dict(), b.as_dict()
        pa_, pb_ = da.pop("provenance"), db.pop("provenance")
        # The query-relative distance is recomputed locally: equal to float rounding (numpy vector vs scalar paths).
        ma, mb = da["data"].pop("match_dist"), db["data"].pop("match_dist")
        assert math.isclose(ma, mb, rel_tol=1e-12, abs_tol=1e-12)  # deg: 3.6e-9 arcsec
        assert da == db
        assert pa_["provider"] == "skycache" and pb_["provider"] == "tap"
        assert pa_["source_id"] == pb_["source_id"] and pa_["search_radius_arcsec"] == pb_["search_radius_arcsec"]
    for key in ("status", "row_count", "raw_row_count", "truncated", "archive_truncated", "excess_row_count",
                "pad_row_count", "requested_radius_arcsec", "query_radius_arcsec", "cone", "warnings", "citation",
                "row_limit", "max_rows", "dropped_rows", "filtered_rows", "epoch_span"):
        assert local.meta[key] == remote.meta[key], key
    for key in ("pad_sources", "excess_sources"):
        assert [s.source_id for s in local.meta[key]] == [s.source_id for s in remote.meta[key]], key
    assert set(local.meta) >= set(remote.meta)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def synth():
    definition, entry = synth_definition()
    return definition, entry


# ---------------------------------------------------------------------------
# HEALPix / MOC / partitioning primitives
# ---------------------------------------------------------------------------


def test_healpix29_matches_hats_spatial_index():
    """astropy-healpix NESTED order-29 indices equal hats' (cdshealpix-based) _healpix_29."""
    hats_si = pytest.importorskip("hats.pixel_math.spatial_index")
    rng = np.random.default_rng(3)
    ra = np.concatenate([rng.uniform(0, 360, 20000), [0.0, 359.9999999999, 180.0, 0.0, 45.0]])
    dec = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 20000))), [90.0, -90.0, 0.0, -89.9999999, 41.8103149]])
    assert np.array_equal(healpix29(ra, dec), hats_si.compute_spatial_index(ra, dec))


def test_moc_ranges_union_contains_and_ascii():
    whole0 = Moc.from_pixels(0, [0])
    assert whole0.ranges.tolist() == [[0, 4**29]]
    children = Moc.from_pixels(1, [0, 1, 2, 3])
    assert children == whole0
    assert children.to_ascii() == "0/0"
    moc = Moc.from_pixels(3, [5, 6, 7, 20]) | Moc.from_pixels(4, [4 * 21 + 1])
    assert moc.to_ascii() == "3/5-7 20 4/85"
    assert moc.contains_pixels(5, [5 * 16, 7 * 16 + 15, (4 * 21 + 1) * 4]).all()
    assert not moc.contains_pixels(3, [4, 21]).any()
    assert not moc.contains_pixels(2, [1]).any()  # pixel 1 at order 2 = 4..7 at order 3, 4 missing
    assert math.isclose(Moc.from_pixels(0, range(12)).sky_fraction, 1.0)
    assert math.isclose(Moc.from_pixels(0, range(12)).area_deg2, 41252.96, rel_tol=1e-6)
    assert Moc.from_json(moc.as_json()) == moc
    empty = Moc()
    assert empty.empty and empty.to_ascii() == "" and not empty.contains_pixels(0, [0]).any()


def test_pixels_inside_cone_are_really_inside_and_cone_pixels_inclusive():
    """Conservative containment and inclusive overlap, verified by dense sub-pixel sampling."""
    from astropy_healpix import HEALPix

    rng = np.random.default_rng(11)
    for _ in range(25):
        order = int(rng.integers(6, 13))
        ra0, dec0 = rng.uniform(0, 360), float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        radius = rng.uniform(2, 8) * skycache.pixel_resolution_deg(order)
        inside = pixels_inside_cone(ra0, dec0, radius, order)
        over = set(cone_pixels(ra0, dec0, radius, order).tolist())
        fine = HEALPix(2 ** (order + 6), order="nested")
        for p in inside.tolist()[:40]:
            lon, lat = fine.healpix_to_lonlat(np.arange(p * 4**6, (p + 1) * 4**6))
            assert angular_sep_deg(ra0, dec0, lon.deg, lat.deg).max() <= radius
        # Random points inside the cone (up to its very edge) all fall in returned overlap pixels.
        pts = cap_points(ra0, dec0, radius, 3000, seed=int(rng.integers(1e6)))
        owners = healpix29(*pts) >> (2 * (29 - order))
        assert set(owners.tolist()) <= over
        assert set(inside.tolist()) <= over


def test_partition_orders_threshold_and_disjoint():
    ra, dec = cap_points(10.0, 20.0, 1.0, 5000, seed=2)
    h = np.sort(healpix29(ra, dec))
    orders, pixels = partition_orders(h, threshold=300)
    assert (orders >= 0).all()
    assert np.array_equal(h >> (2 * (29 - orders.astype(np.int64))), pixels)
    parts = {}
    for k, p in zip(orders.tolist(), pixels.tolist()):
        parts[(k, p)] = parts.get((k, p), 0) + 1
    assert max(parts.values()) <= 300 and len({k for k, _ in parts}) > 1
    ranges = np.concatenate([pixel_ranges(k, [p]) for k, p in sorted(parts, key=lambda t: t[1] << 2 * (29 - t[0]))])
    assert (ranges[1:, 0] >= ranges[:-1, 1]).all()  # disjoint


# ---------------------------------------------------------------------------
# Mirror -> local cone == remote cone == brute force
# ---------------------------------------------------------------------------


def _mirror(store, cat, definition, **kwargs):
    provider = SyntheticTap(cat)
    report = run(mirror_region(definition, store=store, providers={"tap": provider}, **kwargs))
    return report, provider


def _compare_cones(store, cat, definition, cones, *, epoch=None):
    local = LocalProvider(store)
    remote = SyntheticTap(cat)
    for ra, dec, radius in cones:
        target = validate_target(ra, dec, epoch=epoch)
        got = run(local.query(definition, target, radius))
        want = run(remote.query(definition, target, radius))
        assert_identical(got, want)
        if epoch is None:
            inside, edge = brute_force(cat, ra, dec, radius)
            ids = {int(s.source_id) for s in got}
            if len(got) < definition.max_rows:
                assert inside <= ids <= inside | edge
        assert got.meta["skycache"]["hit"] is True


def test_mirror_cone_across_ra_zero_matches_remote_and_brute_force(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(0.02, 0.0, 0.45, 4000, seed=5)
    ra = np.concatenate([ra, [0.0, 359.9999999, 0.0000001, 0.3, 359.7]])
    dec = np.concatenate([dec, [0.0, 0.0001, -0.0001, 0.05, -0.05]])
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store", partition_rows=500)
    report, _provider = _mirror(store, cat, definition, ra=0.02, dec=0.0, radius_deg=0.4)
    assert report.queries == 1 and report.tiles_failed == 0
    assert report.region_covered_fraction > 0.95
    sep = angular_sep_deg(0.02, 0.0, cat.ra, cat.dec)
    assert report.rows_total == int((sep <= 0.4).sum())
    cones = [(0.0, 0.0, 60.0), (359.95, 0.01, 200.0), (0.05, -0.05, 400.0), (359.99, 0.0, 5.0), (0.02, 0.0, 1200.0)]
    _compare_cones(store, cat, definition, cones)
    result = store.cone_search("synth", 0.0, 0.0, 720.0)
    assert result.ra.max() > 359.0 and result.ra.min() < 1.0  # both sides of RA = 0
    assert result.partitions_read >= 2


def test_mirror_poles(tmp_path, synth):
    definition, _ = synth
    ra_n, dec_n = cap_points(0.0, 90.0, 0.4, 3000, seed=7)
    ra_s, dec_s = cap_points(0.0, -90.0, 0.4, 3000, seed=8)
    cat = SynthCatalog(np.concatenate([ra_n, ra_s, [0.0, 123.0]]), np.concatenate([dec_n, dec_s, [90.0, -90.0]]))
    store = SkyCache(tmp_path / "store", partition_rows=400)
    # North: a cone centred on the pole. South: the HEALPix pixels touching the pole.
    _mirror(store, cat, definition, ra=0.0, dec=90.0, radius_deg=0.35)
    south = cone_pixels(0.0, -90.0, 0.25, 8)
    report, _ = _mirror(store, cat, definition, healpix_order=8, healpix_pixels=south.tolist())
    assert report.region_covered_fraction == 1.0
    cones = [(0.0, 90.0, 300.0), (200.0, 89.95, 400.0), (45.0, 89.99, 30.0),
             (0.0, -90.0, 300.0), (300.0, -89.93, 250.0), (90.0, -89.999, 20.0)]
    _compare_cones(store, cat, definition, cones)
    assert store.cone_search("synth", 0.0, 90.0, 1.0).source_ids  # the source exactly at the pole


def test_pixel_boundary_sources_and_corner_cones(tmp_path, synth):
    """Sources exactly on partition-pixel edges/corners and cones centred on corners."""
    from astropy_healpix import HEALPix

    definition, _ = synth
    base_ra, base_dec = cap_points(150.0, 30.0, 0.3, 3000, seed=9)
    order = 9
    pix = cone_pixels(150.0, 30.0, 0.2, order)
    lon, lat = HEALPix(2**order, order="nested").boundaries_lonlat(pix, step=4)
    edge_ra, edge_dec = lon.deg.ravel(), lat.deg.ravel()
    cat = SynthCatalog(np.concatenate([base_ra, edge_ra]), np.concatenate([base_dec, edge_dec]))
    store = SkyCache(tmp_path / "store", partition_rows=60)  # many small partitions at high orders
    _mirror(store, cat, definition, ra=150.0, dec=30.0, radius_deg=0.28)
    meta = store.metadata("synth")
    assert len({k for k, _p, _n in meta["partitions"]}) >= 2
    corners = HEALPix(2**order, order="nested").boundaries_lonlat(pix[:6], step=1)
    cones = [(float(a) % 360, float(b), r) for a, b, r in zip(corners[0].deg[:, 0], corners[1].deg[:, 0],
                                                            [3.0, 30.0, 90.0, 150.0, 0.5, 10.0])]
    _compare_cones(store, cat, definition, cones)
    # Every stored row sits in the partition its _healpix_29 says.
    for k, p, _n in meta["partitions"]:
        part = store._partition(store._state("synth"), k, p)
        assert np.all(part.h29 >> (2 * (29 - k)) == p) and np.all(np.diff(part.h29) >= 0)
        assert np.array_equal(part.h29, healpix29(part.ra, part.dec))


def test_truncated_region_is_tiled_and_complete(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(80.0, -20.0, 0.3, 3000, seed=13)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store")
    report, provider = _mirror(store, cat, definition, ra=80.0, dec=-20.0, radius_deg=0.2, tile_rows=150)
    assert report.queries > 5 and report.tiles_failed == 0
    assert any("split into" in w for w in report.warnings)
    assert report.region_covered_fraction == 1.0
    ids = [r.source_id for r in store.load_rows("synth")]
    assert len(ids) == len(set(ids))  # overlapping tile cones deduplicated
    # Every row inside the covered area is stored.
    covered = store.coverage("synth").contains_points(cat.ra, cat.dec)
    assert set(np.asarray(cat.ids)[covered].astype(str).tolist()) <= set(ids)
    # All tile cones respected the row limit and were issued with TOP 150 + 1.
    assert all(call[3] == 150 for call in provider.calls)
    _compare_cones(store, cat, definition, [(80.0, -20.0, 400.0), (80.1, -19.9, 120.0), (79.85, -20.05, 30.0)])


def test_epoch_widened_and_row_limited_cones_match_remote(tmp_path):
    definition, _ = synth_definition(max_rows=25)
    ra, dec = cap_points(210.0, 45.0, 0.3, 4000, seed=17)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store")
    _mirror(store, cat, definition, ra=210.0, dec=45.0, radius_deg=0.25, tile_rows=5000)
    # Target epoch 2000 without proper motion: cone widened by 10.5"/yr * 16 yr (pad rows).
    _compare_cones(store, cat, definition, [(210.0, 45.0, 3.0), (210.05, 45.02, 10.0)], epoch=2000.0)
    got = run(LocalProvider(store).query(definition, validate_target(210.0, 45.0, epoch=2000.0), 3.0))
    assert got.meta["cone"]["mode"] == "unknown_pm_pad" and got.meta["pad_row_count"] > 0
    # More than max_rows rows in the cone: same truncation, excess rows and warnings.
    _compare_cones(store, cat, definition, [(210.0, 45.0, 200.0)])
    big = run(LocalProvider(store).query(definition, validate_target(210.0, 45.0), 200.0))
    assert big.meta["truncated"] and len(big) == 25


def test_coverage_errors_and_local_first_fallback(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(30.0, 10.0, 0.3, 2000, seed=21)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store")
    local = LocalProvider(store)
    with pytest.raises(CoverageError) as missing:
        run(local.query(definition, validate_target(30.0, 10.0), 5.0))
    assert missing.value.reason == "not_mirrored"
    _mirror(store, cat, definition, ra=30.0, dec=10.0, radius_deg=0.1)
    with pytest.raises(CoverageError) as partial:
        run(local.query(definition, validate_target(30.0, 10.1), 60.0))
    assert partial.value.reason == "partially_covered" and 0.0 < partial.value.covered_fraction < 1.0
    with pytest.raises(CoverageError) as outside:
        run(local.query(definition, validate_target(31.0, 10.0), 60.0))
    assert outside.value.reason == "not_covered" and outside.value.covered_fraction == 0.0
    # The epoch-widened cone (pad 168") pokes out of the 0.1 deg mirror although 5" would fit.
    with pytest.raises(CoverageError):
        run(local.query(definition, validate_target(30.0, 10.07, epoch=2000.0), 5.0))
    assert store.covers("synth", 30.0, 10.07, 5.0).covered
    # Local-first wrapper: remote answer + skycache miss note when not covered, local when covered.
    remote = SyntheticTap(cat)
    wrapped = SkyCacheProvider(remote, local)
    miss = run(wrapped.query(definition, validate_target(30.0, 10.1), 60.0))
    assert miss.meta["skycache"]["hit"] is False and len(remote.calls) == 1
    assert_identical_ids = [s.source_id for s in miss] == [s.source_id for s in run(remote.query(
        definition, validate_target(30.0, 10.1), 60.0))]
    assert assert_identical_ids
    hit = run(wrapped.query(definition, validate_target(30.0, 10.0), 60.0))
    assert hit.meta["skycache"]["hit"] is True and len(remote.calls) == 2
    wrapped_map = skycache.wrap_providers({"tap": remote}, store)
    assert isinstance(wrapped_map["tap"], SkyCacheProvider)


def test_remirror_dedupes_and_replaces_rows(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(300.0, -40.0, 0.4, 3000, seed=23)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store")
    _mirror(store, cat, definition, ra=300.0, dec=-40.0, radius_deg=0.2)
    second, _ = _mirror(store, cat, definition, ra=300.1, dec=-40.0, radius_deg=0.2)  # overlaps the first
    ids = [r.source_id for r in store.load_rows("synth")]
    assert len(ids) == len(set(ids))
    union = (angular_sep_deg(300.0, -40.0, cat.ra, cat.dec) <= 0.2) | (angular_sep_deg(300.1, -40.0, cat.ra, cat.dec) <= 0.2)
    assert len(ids) == int(union.sum())
    assert second.rows_replaced > 0
    # A source deleted upstream disappears when its area is mirrored again.
    victim = int(np.flatnonzero(angular_sep_deg(300.0, -40.0, cat.ra, cat.dec) < 0.05)[0])
    cat.alive[victim] = False
    _mirror(store, cat, definition, ra=300.0, dec=-40.0, radius_deg=0.2)
    assert str(cat.ids[victim]) not in {r.source_id for r in store.load_rows("synth")}
    assert len(store.metadata("synth")["mirrors"]) == 3


def test_raw_values_round_trip_exactly(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(5.0, 5.0, 0.1, 500, seed=29)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store")
    _mirror(store, cat, definition, ra=5.0, dec=5.0, radius_deg=0.1)
    meta = store.metadata("synth")
    assert set(meta["json_columns"]) == {"flag", "tags"}
    by_id = {str(i): k for k, i in enumerate(cat.ids)}
    rows = store.load_rows("synth")
    assert rows
    for row in rows:
        want = cat.row(by_id[row.source_id], 0.0)
        want.pop("match_dist")
        assert row.data == want
        assert list(row.data) == list(want)  # column order preserved
        assert type(row.data["source_id"]) is int and type(row.data["flag"]) is type(want["flag"])


def test_hats_layout_properties_and_lsdb(tmp_path, synth):
    definition, _ = synth
    ra, dec = cap_points(120.0, -5.0, 0.3, 3000, seed=31)
    cat = SynthCatalog(ra, dec)
    store = SkyCache(tmp_path / "store", partition_rows=300)
    _mirror(store, cat, definition, ra=120.0, dec=-5.0, radius_deg=0.25)
    root = store.catalog_path("synth")
    info = (root / "partition_info.csv").read_text().splitlines()
    assert info[0] == "Norder,Npix"
    listed = {tuple(map(int, line.split(","))) for line in info[1:]}
    files = {p.relative_to(root / "dataset").as_posix() for p in (root / "dataset").rglob("Npix=*.parquet")}
    assert files == {f"Norder={k}/Dir={(p // 10000) * 10000}/Npix={p}.parquet" for k, p in listed}
    import pyarrow.parquet as pq

    schema = pq.read_schema(root / "dataset" / "_common_metadata")
    assert schema.names[0] == "_healpix_29" and "Norder" not in schema.names
    assert schema.field("_sc_ra").metadata[b"unit"] == b"deg"
    assert schema.field("ra_error").metadata[b"ucd"] == b"stat.error;pos.eq.ra"
    md = pq.read_metadata(root / "dataset" / "_metadata")
    assert md.num_rows == store.metadata("synth")["rows"] and md.num_row_groups == len(listed)

    hats = pytest.importorskip("hats")
    props = hats.catalog.dataset.table_properties.TableProperties.read_from_dir(root)
    assert props.catalog_name == "synth" and props.catalog_type == "object"
    assert props.ra_column == "_sc_ra" and props.healpix_column == "_healpix_29" and props.healpix_order == 29
    assert props.total_rows == md.num_rows
    catalog = hats.read_hats(root)
    assert {(p.order, p.pixel) for p in catalog.get_healpix_pixels()} == listed

    lsdb = pytest.importorskip("lsdb")
    frame = lsdb.open_catalog(root).cone_search(ra=120.02, dec=-5.01, radius_arcsec=300.0).compute()
    ours = store.cone_search("synth", 120.02, -5.01, 300.0)
    assert sorted(frame["_sc_source_id"].tolist()) == sorted(ours.source_ids)
    assert len(ours) > 50


def test_mirror_input_validation(tmp_path, synth):
    definition, _ = synth
    store = SkyCache(tmp_path / "store")
    cat = SynthCatalog(*cap_points(1.0, 1.0, 0.1, 10, seed=1))
    with pytest.raises(MirrorInputError):
        _mirror(store, cat, definition, ra=1.0, dec=1.0, radius_deg=5.0)  # > 1 deg default limit
    with pytest.raises(MirrorInputError):
        _mirror(store, cat, definition, ra=1.0, dec=1.0)
    with pytest.raises(MirrorInputError):
        _mirror(store, cat, definition, healpix_order=3, healpix_pixels=[12 * 4**3])
    xamin = copy.deepcopy(definition)
    xamin.provider = "heasarc_xamin"
    with pytest.raises(MirrorInputError, match="truncation"):
        _mirror(store, cat, xamin, ra=1.0, dec=1.0, radius_deg=0.1)
    with pytest.raises(MirrorInputError):
        store.catalog_path("../evil")


def test_mirror_reports_upstream_failure(tmp_path, synth):
    from models import CatalogUnavailableError

    definition, _ = synth

    class Down(SyntheticTap):
        async def query(self, catalog, target, radius_arcsec):
            raise CatalogUnavailableError("archive down (HTTP 503)")

    store = SkyCache(tmp_path / "store")
    with pytest.raises(MirrorError, match="archive down"):
        run(mirror_region(definition, ra=1.0, dec=1.0, radius_deg=0.1, store=store,
                          providers={"tap": Down(SynthCatalog([1.0], [1.0]))}))
    assert not store.has("synth")


# ---------------------------------------------------------------------------
# Router and CLI
# ---------------------------------------------------------------------------


def _app(tmp_path, cat):
    _definition, entry = synth_definition()
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(yaml.safe_dump({"catalogs": {"synth": entry}}), encoding="utf-8")
    registry = CatalogRegistry(registry_file)
    app = FastAPI()
    app.include_router(skycache.router)
    app.state.skycache = SkyCache(tmp_path / "store")
    app.state.registry = registry
    app.state.service = SimpleNamespace(providers={"tap": SyntheticTap(cat)}, registry=registry)
    return app


def test_router_status_mirror_cone_delete(tmp_path):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 1500, seed=41))
    client = TestClient(_app(tmp_path, cat))
    assert client.get("/api/v1/skycache/status").json()["catalogs"] == []
    response = client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0, "radius_deg": 0.15})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows_total"] == int((angular_sep_deg(45.0, 45.0, cat.ra, cat.dec) <= 0.15).sum())
    status = client.get("/api/v1/skycache/status").json()
    assert status["catalogs"][0]["catalog"] == "synth" and status["catalogs"][0]["coverage_area_deg2"] > 0.05
    cone = client.get("/api/v1/skycache/cone", params={"catalog": "synth", "ra": 45.0, "dec": 45.0, "radius_arcsec": 60})
    assert cone.status_code == 200
    inside, edge = brute_force(cat, 45.0, 45.0, 60.0)
    assert inside <= {int(s["source_id"]) for s in cone.json()["sources"]} <= inside | edge
    outside = client.get("/api/v1/skycache/cone", params={"catalog": "synth", "ra": 46.0, "dec": 45.0, "radius_arcsec": 60})
    assert outside.status_code == 409 and outside.json()["detail"]["reason"] == "not_covered"
    pix = client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "healpix_order": 12,
                                                        "healpix_pixels": cone_pixels(45.0, 45.0, 0.01, 12).tolist()})
    assert pix.status_code == 200 and pix.json()["region"]["type"] == "healpix"
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 1.0}).status_code == 422
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "nope", "ra": 1.0, "dec": 1.0,
                                                         "radius_deg": 0.1}).status_code == 404
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 1.0, "dec": 1.0,
                                                         "radius_deg": 5.0}).status_code == 422
    assert client.delete("/api/v1/skycache/synth").json() == {"catalog": "synth", "deleted": True}
    assert client.delete("/api/v1/skycache/synth").status_code == 404


def _parser():
    parser = argparse.ArgumentParser()
    skycache.register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_cli_status_cone_delete(tmp_path, capsys, synth, monkeypatch):
    definition, entry = synth
    cat = SynthCatalog(*cap_points(60.0, -30.0, 0.2, 800, seed=43))
    store = SkyCache(tmp_path / "store")
    _mirror(store, cat, definition, ra=60.0, dec=-30.0, radius_deg=0.15)
    args = _parser().parse_args(["skycache", "status", "--store", str(store.root), "--json"])
    assert args.handler(args) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["catalogs"][0]["rows"] == store.metadata("synth")["rows"]
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(yaml.safe_dump({"catalogs": {"synth": entry}}), encoding="utf-8")
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(registry_file))
    args = _parser().parse_args(["skycache", "cone", "--catalog", "synth", "--ra", "60", "--dec", "-30",
                                 "--radius-arcsec", "90", "--store", str(store.root), "--json"])
    assert args.handler(args) == 0
    got = {int(s["source_id"]) for s in json.loads(capsys.readouterr().out)}
    inside, edge = brute_force(cat, 60.0, -30.0, 90.0)
    assert inside <= got <= inside | edge
    args = _parser().parse_args(["skycache", "cone", "--catalog", "synth", "--ra", "61", "--dec", "-30",
                                 "--radius-arcsec", "90", "--store", str(store.root)])
    assert args.handler(args) == 1 and "0.0% inside the mirrored coverage" in capsys.readouterr().out
    args = _parser().parse_args(["skycache", "delete", "--catalog", "synth", "--store", str(store.root)])
    assert args.handler(args) == 0 and not store.has("synth")
    args = _parser().parse_args(["mirror", "--catalog", "gaia_dr3", "--store", str(store.root)])
    assert args.handler(args) == 2  # no region given
