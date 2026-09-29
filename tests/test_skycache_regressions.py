"""Offline regression tests of the sky mirror (skycache.py) for reviewed failure modes.

Uses the synthetic TAP archive of tests/test_skycache.py (TOP row_limit + 1 nearest rows,
DISTANCE in degrees, rows converted by the providers' own ``_sources``).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_skycache import (
    COLUMNS,
    SynthCatalog,
    SyntheticTap,
    angular_sep_deg,
    assert_identical,
    cap_points,
    covered_rows,
    run,
    synth_definition,
)

import skycache
from crossmatch import QueryExecutor
from models import CatalogQueryError, CatalogRegistry, CatalogUnavailableError, QueryPlan, validate_target
from skycache import (
    CoverageError,
    LocalProvider,
    MirrorError,
    MirrorInputError,
    MirrorStoreError,
    Moc,
    SkyCache,
    SkyCacheProvider,
    StoredRow,
    definition_fingerprint,
    mirror_region,
    wrap_providers,
)


@pytest.fixture
def synth():
    return synth_definition()[0]


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(skycache, "RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(skycache, "REPLACE_DELAY_S", 0.0)


def _mirror(store, provider, definition, **kwargs):
    providers = provider if isinstance(provider, dict) else {"tap": provider}
    return run(mirror_region(definition, store=store, providers=providers, **kwargs))


def _stored_ids(store, name="synth"):
    return [r.source_id for r in store.load_rows(name)]


class _Reg:
    def __init__(self, definition):
        self.catalogs = {definition.name: definition}


# ---------------------------------------------------------------------------
# Fallback archives are never mixed into a mirror
# ---------------------------------------------------------------------------


class GatorLike(SyntheticTap):
    """A fallback archive for the same catalog with differently shaped rows (like IRSA Gator)."""

    provider_name = "irsa_gator"

    async def query(self, catalog, target, radius_arcsec):
        res = await super().query(catalog, target, radius_arcsec)
        for s in [*res, *(res.meta.get("excess_sources") or []), *(res.meta.get("pad_sources") or [])]:
            s.data["clon"] = "05h20m00.00s"
        return res


class FirstAttemptDown(SyntheticTap):
    """The primary archive fails (HTTP 503) the first time each distinct cone is requested."""

    def __init__(self, cat):
        super().__init__(cat)
        self.seen: set = set()
        self.failures = 0

    async def query(self, catalog, target, radius_arcsec):
        key = (round(target.ra, 9), round(target.dec, 9), round(radius_arcsec, 6))
        if key not in self.seen:
            self.seen.add(key)
            self.failures += 1
            raise CatalogUnavailableError("TAP query failed: HTTP 503")
        return await super().query(catalog, target, radius_arcsec)


def _with_fallback(definition):
    return replace(definition, parameters={**definition.parameters,
                                           "fallback": {"provider": "irsa_gator",
                                                        "endpoint": "https://synthetic.invalid/gator"}})


def test_flaky_primary_is_retried_never_filled_from_fallback(tmp_path, synth):
    definition = _with_fallback(synth)
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.3, 3000, seed=13))
    primary, fallback = FirstAttemptDown(cat), GatorLike(cat)
    store = SkyCache(tmp_path / "store")
    report = _mirror(store, {"tap": primary, "irsa_gator": fallback}, definition,
                     ra=80.0, dec=-20.0, radius_deg=0.2, tile_rows=150)
    assert fallback.calls == []  # the fallback archive is never asked
    assert primary.failures > 5 and report.tiles_retried == primary.failures
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert report.providers_used == ["tap"] and store.metadata("synth")["row_providers"] == ["tap"]
    ids = _stored_ids(store)
    assert len(ids) == len(set(ids))
    assert all("clon" not in r.data and r.provider == "tap" for r in store.load_rows("synth"))
    covered = store.coverage("synth").contains_points(cat.ra, cat.dec)
    assert set(np.asarray(cat.ids)[covered].astype(str).tolist()) <= set(ids)
    local = LocalProvider(store)
    for ra, dec, radius in [(80.0, -20.0, 400.0), (80.05, -19.95, 60.0)]:
        target = validate_target(ra, dec)
        got = run(local.query(definition, target, radius))
        assert_identical(got, run(SyntheticTap(cat).query(definition, target, radius)))
        assert len({s.source_id for s in got}) == len(got)


def test_primary_down_tiles_fail_without_fallback_rows(tmp_path, synth):
    definition = _with_fallback(synth)
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.3, 3000, seed=13))

    class EastDown(SyntheticTap):
        async def query(self, catalog, target, radius_arcsec):
            if target.ra > 80.05:
                raise CatalogUnavailableError("TAP query failed: HTTP 503")
            return await super().query(catalog, target, radius_arcsec)

    fallback = GatorLike(cat)
    store = SkyCache(tmp_path / "store")
    report = _mirror(store, {"tap": EastDown(cat), "irsa_gator": fallback}, definition,
                     ra=80.0, dec=-20.0, radius_deg=0.2, tile_rows=150)
    assert fallback.calls == [] and report.tiles_failed > 0 and report.region_covered_fraction < 1.0
    assert report.tiles_retried == report.tiles_failed * skycache.UNAVAILABLE_RETRIES
    assert any("CatalogUnavailableError" in w for w in report.warnings)
    with pytest.raises(CoverageError):
        run(LocalProvider(store).query(definition, validate_target(80.15, -20.0), 60.0))


def test_stored_fallback_rows_are_refused(tmp_path, synth):
    """Stores written before fallback results were refused: cones touching such rows go remote."""
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    rows = store.load_rows("synth")
    near = angular_sep_deg(30.0, 10.0, [r.ra for r in rows], [r.dec for r in rows]) < 0.02
    for row, flag in zip(rows, near):
        if flag:
            row.provider = "irsa_gator"
    meta = store.metadata("synth")
    store.write(synth, rows, store.coverage("synth"), mirrors=meta["mirrors"])
    assert store.status()["catalogs"][0]["row_providers"] == ["irsa_gator", "tap"]
    with pytest.raises(CoverageError) as err:
        run(LocalProvider(store).query(synth, validate_target(30.0, 10.0), 30.0))
    assert err.value.reason == "fallback_rows"
    remote = SyntheticTap(cat)
    res = run(SkyCacheProvider(remote, LocalProvider(store)).query(synth, validate_target(30.0, 10.0), 30.0))
    assert res.meta["skycache"] == {"hit": False, "reason": "fallback_rows", "covered_fraction": 1.0}
    assert len(remote.calls) == 1
    # A cone containing only primary rows is still answered locally.
    far = run(LocalProvider(store).query(synth, validate_target(30.08, 10.0), 30.0))
    assert far.meta["skycache"]["hit"] is True


# ---------------------------------------------------------------------------
# Deduplication by position (several rows at one position from one request survive)
# ---------------------------------------------------------------------------


def test_overlapping_tiles_dedupe_by_position_keeping_multi_row_objects(tmp_path, synth):
    ra, dec = cap_points(210.0, 45.0, 0.3, 3000, seed=17)
    cat = SynthCatalog(ra, dec)
    # 40 objects listed twice by the archive (e.g. one row per spectrum): same id and position.
    dup_of = {i: i + 1500 for i in range(0, 400, 10)}
    for src, dup in dup_of.items():
        cat.ids[dup] = cat.ids[src]
        cat.ra[dup], cat.dec[dup] = cat.ra[src], cat.dec[src]
        cat.mag[dup] = 30.0 + src  # a different row for the same object
    store = SkyCache(tmp_path / "store", partition_rows=300)
    report = _mirror(store, SyntheticTap(cat), synth, ra=210.0, dec=45.0, radius_deg=0.2, tile_rows=100)
    assert report.queries > 10 and report.rows_deduplicated > 0
    ids = _stored_ids(store)
    counts = {i: ids.count(i) for i in set(ids)}
    inside = angular_sep_deg(210.0, 45.0, cat.ra, cat.dec) <= 0.2
    # Tile cones also fetch rows just outside the region: every stored multi-row object has both rows.
    multi = {str(cat.ids[s]) for s in dup_of} & set(ids)
    assert {str(cat.ids[s]) for s in dup_of if inside[s]} <= multi
    assert multi and all(counts[i] == 2 for i in multi)
    assert all(c == 1 for i, c in counts.items() if i not in multi)
    assert report.rows_stored == len(ids)
    remote = SyntheticTap(cat)
    for src in list(dup_of)[:6]:
        if inside[src] and angular_sep_deg(210.0, 45.0, cat.ra[src], cat.dec[src]) < 0.15:
            target = validate_target(float(cat.ra[src]), float(cat.dec[src]))
            assert_identical(run(LocalProvider(store).query(synth, target, 20.0)),
                             run(remote.query(synth, target, 20.0)))


# ---------------------------------------------------------------------------
# Local cones: cost independent of cone density, exact tie handling
# ---------------------------------------------------------------------------


def test_cone_search_converts_only_the_rows_it_returns(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(100.0, 10.0, 0.12, 20000, seed=3))
    store = SkyCache(tmp_path / "store", partition_rows=4000)
    _mirror(store, SyntheticTap(cat), synth, ra=100.0, dec=10.0, radius_deg=0.1, tile_rows=30000)
    calls = {"n": 0}
    real = SkyCache._raw_row

    def counting(state, record, colset):
        calls["n"] += 1
        return real(state, record, colset)

    monkeypatch.setattr(SkyCache, "_raw_row", staticmethod(counting))
    full = store.cone_search("synth", 100.0, 10.0, 300.0)
    in_cone = int((angular_sep_deg(100.0, 10.0, cat.ra, cat.dec) <= 300.0 / 3600.0).sum())
    assert len(full) == full.rows_in_cone == in_cone > 5000 and calls["n"] == in_cone
    calls["n"] = 0
    top = store.cone_search("synth", 100.0, 10.0, 300.0, limit=11)
    assert calls["n"] == 11 and top.rows_in_cone == in_cone
    assert top.source_ids == full.source_ids[:11]
    assert np.array_equal(top.separation_arcsec, full.separation_arcsec[:11])
    # LocalProvider (TOP 201) on the dense cone converts 201 rows and equals the archive.
    calls["n"] = 0
    target = validate_target(100.0, 10.0)
    got = run(LocalProvider(store).query(synth, target, 300.0))
    assert calls["n"] == synth.max_rows + 1 and got.meta["skycache"]["rows_in_cone"] == in_cone
    assert_identical(got, run(SyntheticTap(cat).query(synth, target, 300.0)))


def test_cone_search_limit_cuts_through_ties_by_source_id(tmp_path, synth):
    """Groups of rows at identical positions: every limit gives the (separation, id)-sorted prefix."""
    rng = np.random.default_rng(8)
    rows = []
    for group in range(40):
        g_ra, g_dec = 45.0 + 0.0004 * (group + 1), 45.0  # 40 distinct separations
        for member in rng.permutation(5):
            sid = str(900_000 + member * 1000 + group)
            rows.append(StoredRow(sid, g_ra, g_dec, {"source_id": int(sid), "ra": g_ra, "dec": g_dec},
                                  provider="tap", retrieved_at="2026-01-01T00:00:00+00:00", columns=COLUMNS[:3]))
    store = SkyCache(tmp_path / "store", partition_rows=30)
    store.write(synth, rows, Moc.from_cone(45.0, 45.0, 0.1, 14))
    sep = angular_sep_deg(45.0, 45.0, [r.ra for r in rows], [r.dec for r in rows])
    expected = [rows[i].source_id for i in sorted(range(len(rows)), key=lambda i: (sep[i], rows[i].source_id))]
    full = store.cone_search("synth", 45.0, 45.0, 120.0)
    assert full.source_ids == expected
    for limit in range(len(rows) + 3):
        got = store.cone_search("synth", 45.0, 45.0, 120.0, limit=limit)
        assert got.source_ids == expected[:limit], limit
        assert len(got.rows) == len(got.colsets) == len(got.ra) == min(limit, len(rows))


# ---------------------------------------------------------------------------
# Definition changes invalidate the mirror
# ---------------------------------------------------------------------------


def test_definition_change_invalidates_local_answers(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    target = validate_target(30.0, 10.0)
    local = LocalProvider(store)
    params = synth.parameters
    changed = {
        "parameters.columns": replace(synth, parameters={**params, "columns": ["source_id", "ra", "dec"]}),
        "parameters.where": replace(synth, parameters={**params, "where": "phot_g_mean_mag < 12"}),
        "table": replace(synth, table="synthetic.other"),
        "endpoint": replace(synth, endpoint="https://elsewhere.invalid/tap"),
        "parameters.exclude_values": replace(synth, parameters={**params, "exclude_values": {"flag": ["F1"]}}),
        "parameters.null_sentinels": replace(synth, parameters={**params, "null_sentinels": [-999]}),
        "parameters.id_field": replace(synth, parameters={**params, "id_field": "designation"}),
    }
    for field_name, definition in changed.items():
        with pytest.raises(CoverageError) as err:
            run(local.query(definition, target, 60.0))
        assert err.value.reason == "definition_changed" and field_name in str(err.value), field_name
    # Fields applied at query time only (from the definition given) do not invalidate.
    for definition in (replace(synth, max_rows=7), replace(synth, citation="other", timeout_seconds=3.0),
                       replace(synth, pos_error={"columns": ["ra_error"], "units": "mas", "kind": "sigma"}),
                       replace(synth, parameters={**params, "epoch_range": [2015.0, 2017.0]}),
                       replace(synth, parameters={**params, "fallback": {"provider": "irsa_gator"}})):
        assert definition_fingerprint(definition) == definition_fingerprint(synth)
        assert_identical(run(local.query(definition, target, 60.0)),
                         run(SyntheticTap(cat).query(definition, target, 60.0)))
    # The local-first wrapper answers from the archive when the definition changed.
    remote = SyntheticTap(cat)
    res = run(SkyCacheProvider(remote, local).query(changed["parameters.where"], target, 60.0))
    assert res.meta["skycache"]["hit"] is False and res.meta["skycache"]["reason"] == "definition_changed"
    assert len(remote.calls) == 1
    # Status: compared with a registry holding the changed definition.
    status = store.status(SimpleNamespace(get=lambda name: changed["parameters.where"]))
    entry = status["catalogs"][0]
    assert entry["definition_current"] is False and entry["definition_changes"] == ["parameters.where"]
    assert store.status(SimpleNamespace(get=lambda name: synth))["catalogs"][0]["definition_current"] is True


def test_remirror_with_changed_definition_starts_a_fresh_store(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.3, 2000, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.2)
    new_def = replace(synth, parameters={**synth.parameters, "where": "ruwe < 1.4"})
    report = _mirror(store, SyntheticTap(cat), new_def, ra=30.1, dec=10.0, radius_deg=0.1)
    assert any("definition changed" in w and "parameters.where" in w for w in report.warnings)
    in_new = angular_sep_deg(30.1, 10.0, cat.ra, cat.dec) <= 0.1
    covered = covered_rows(store, cat)
    assert not covered[~in_new].any()  # nothing of the discarded copy's region is left
    assert sorted(_stored_ids(store)) == sorted(str(i) for i in np.asarray(cat.ids)[covered])
    assert report.rows_fetched == int(in_new.sum())
    assert len(store.metadata("synth")["mirrors"]) == 1
    assert not store.covers("synth", 29.9, 10.0, 10.0).covered  # only the new region is covered
    got = run(LocalProvider(store).query(new_def, validate_target(30.1, 10.0), 60.0))
    assert got.meta["skycache"]["hit"] is True
    with pytest.raises(CoverageError):
        run(LocalProvider(store).query(synth, validate_target(30.1, 10.0), 60.0))


def test_stores_without_recorded_fingerprint_are_checked_against_their_definition(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    path = store.catalog_path("synth") / "skycache.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("definition_signature")
    meta.pop("definition_fingerprint")
    path.write_text(json.dumps(meta), encoding="utf-8")
    fresh = SkyCache(store.root)
    assert run(LocalProvider(fresh).query(synth, validate_target(30.0, 10.0), 30.0)).meta["skycache"]["hit"]
    with pytest.raises(CoverageError, match="parameters.where"):
        run(LocalProvider(fresh).query(replace(synth, parameters={**synth.parameters, "where": "x"}),
                                       validate_target(30.0, 10.0), 30.0))


# ---------------------------------------------------------------------------
# Local queries never block the event loop
# ---------------------------------------------------------------------------


def test_local_query_runs_in_a_worker_thread_and_times_out(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 300, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    seen = {}
    real = LocalProvider.query_sync

    def slow(self, catalog, target, radius_arcsec):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        time.sleep(1.0)
        return real(self, catalog, target, radius_arcsec)

    monkeypatch.setattr(LocalProvider, "query_sync", slow)
    definition = replace(synth, timeout_seconds=0.2)

    async def scenario():
        gaps: list[float] = []
        stop = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        executor = QueryExecutor({"tap": LocalProvider(store)}, registry=_Reg(definition))
        t0 = time.perf_counter()
        ok, failures = await executor.execute([QueryPlan("synth", "tap", None, {}, 30.0, "optical")],
                                              validate_target(30.0, 10.0))
        elapsed = time.perf_counter() - t0
        stop.set()
        await beat
        return ok, failures, elapsed, max(gaps)

    ok, failures, elapsed, max_gap = run(scenario())
    assert ok == [] and [f.error_type for f in failures] == ["QueryTimeoutError"]
    assert elapsed < 0.8 and max_gap < 0.15
    assert seen["on_loop"] is False


def test_cone_route_runs_query_off_the_event_loop(tmp_path, monkeypatch):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    client = TestClient(_app(tmp_path, cat))
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                         "radius_deg": 0.1}).status_code == 200
    seen = {}
    real = LocalProvider.query_sync

    def spy(self, *args):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        seen["thread"] = threading.current_thread().name
        return real(self, *args)

    monkeypatch.setattr(LocalProvider, "query_sync", spy)
    response = client.get("/api/v1/skycache/cone", params={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                           "radius_arcsec": 60})
    assert response.status_code == 200 and seen["on_loop"] is False


# ---------------------------------------------------------------------------
# A damaged store falls back to the archive; stale reader state reloads
# ---------------------------------------------------------------------------


def test_wrapper_falls_back_to_remote_when_the_store_is_damaged(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    target = validate_target(30.0, 10.0)
    want = [s.source_id for s in run(SyntheticTap(cat).query(synth, target, 60.0))]
    for part in store.catalog_path("synth").rglob("Npix=*.parquet"):
        part.write_bytes(part.read_bytes()[:100])  # truncated leaf files
    remote = SyntheticTap(cat)
    wrapped = SkyCacheProvider(remote, LocalProvider(SkyCache(store.root)))
    res = run(wrapped.query(synth, target, 60.0))
    assert [s.source_id for s in res] == want and len(remote.calls) == 1
    assert res.meta["skycache"]["hit"] is False and res.meta["skycache"]["reason"] == "local_error"
    assert "Parquet" in res.meta["skycache"]["error"]
    # A malformed skycache.json too.
    (store.catalog_path("synth") / "skycache.json").write_text("{not json", encoding="utf-8")
    res = run(SkyCacheProvider(remote, LocalProvider(SkyCache(store.root))).query(synth, target, 60.0))
    assert [s.source_id for s in res] == want and res.meta["skycache"]["reason"] == "local_error"


@pytest.mark.parametrize("layout", ["same_files", "other_files"])
def test_reader_holding_state_from_before_a_swap_reloads(tmp_path, synth, monkeypatch, layout):
    """A reader's cached state predates a rewrite of the tree: it must never mix versions."""
    ra, dec = cap_points(30.0, 10.0, 0.2, 1500, seed=1)
    old_cat = SynthCatalog(ra, dec, seed=1)
    new_cat = SynthCatalog(ra, dec, seed=2)  # same positions, other values
    root = tmp_path / "store"
    _mirror(SkyCache(root, partition_rows=200), SyntheticTap(old_cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    reader = SkyCache(root, partition_rows=200)
    stale = reader._state("synth")  # metadata read, no partition loaded yet
    writer = SkyCache(root, partition_rows=200 if layout == "same_files" else 90)
    _mirror(writer, SyntheticTap(new_cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    stale_part = stale.partitions[0]
    with pytest.raises(skycache._StaleStateError):
        reader._read_partition(stale, stale_part[0], stale_part[1])
    # The reader's stamp check races with the swap: it sees the old stamp once.
    real_stamp = SkyCache._stamp
    calls = {"n": 0}

    def racing(path):
        calls["n"] += 1
        return stale.stamp if calls["n"] == 1 else real_stamp(path)

    monkeypatch.setattr(SkyCache, "_stamp", staticmethod(racing))
    result = reader.cone_search("synth", 30.0, 10.0, 200.0)
    monkeypatch.setattr(SkyCache, "_stamp", staticmethod(real_stamp))
    assert calls["n"] >= 2
    by_id = {str(i): k for k, i in enumerate(new_cat.ids)}
    assert result.rows and all(row["phot_g_mean_mag"] == new_cat.mag[by_id[sid]]
                               for row, sid in zip(result.rows, result.source_ids))
    assert result.catalog_meta["version"] == writer.metadata("synth")["version"]


# ---------------------------------------------------------------------------
# Directory swap failures, leftover swap directories, status robustness
# ---------------------------------------------------------------------------


def _replace_failing(monkeypatch, predicate):
    """Make os.replace raise a Windows-style PermissionError when ``predicate`` holds.

    Returns (attempts counter, restore function).
    """
    real = os.replace
    attempts = {"n": 0}

    def replace_(src, dst):
        if predicate(Path(src), Path(dst), attempts):
            attempts["n"] += 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(src, dst)

    monkeypatch.setattr(skycache.os, "replace", replace_)
    return attempts, lambda: monkeypatch.setattr(skycache.os, "replace", real)


def test_failed_swap_restores_the_previous_catalog(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    before = sorted(_stored_ids(store))
    version = store.metadata("synth")["version"]
    attempts, restore = _replace_failing(monkeypatch, lambda src, dst, _a: src.name.startswith(".synth.tmp-"))
    with pytest.raises(MirrorStoreError, match="PermissionError"):
        _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    restore()
    assert attempts["n"] == skycache.REPLACE_ATTEMPTS  # retried with backoff before giving up
    assert store.catalogs() == ["synth"] and store.swap_dirs() == []
    assert sorted(_stored_ids(store)) == before and store.metadata("synth")["version"] == version


def test_transient_swap_errors_are_retried(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    attempts, restore = _replace_failing(monkeypatch,
                                         lambda src, dst, a: src.name.startswith(".synth.tmp-") and a["n"] < 2)
    report = _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    restore()
    assert attempts["n"] == 2 and report.rows_total > 0 and store.swap_dirs() == []
    assert sorted(_stored_ids(store)) == sorted(str(i) for i in np.asarray(cat.ids)[covered_rows(store, cat)])


def test_backup_left_by_a_failed_restore_is_recovered(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    before = sorted(_stored_ids(store))
    # Both moving the new tree in and moving the backup back fail.
    _attempts, restore = _replace_failing(monkeypatch, lambda src, dst, _a: dst.name == "synth")
    with pytest.raises(MirrorStoreError):
        _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    restore()
    assert not store.has("synth") and [d["kind"] for d in store.swap_dirs()] == ["old"]
    status = store.status()  # hidden swap directories never break status
    assert status["catalogs"] == [] and status["orphaned_swap_dirs"][0]["kind"] == "old"
    reopened = SkyCache(store.root)  # repair on open restores the backup
    assert reopened.has("synth") and sorted(_stored_ids(reopened)) == before and reopened.swap_dirs() == []


def test_swap_directories_are_hidden_from_catalogs_and_status(tmp_path, synth):
    import shutil

    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    app = _app(tmp_path, cat)
    client = TestClient(app)
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                         "radius_deg": 0.1}).status_code == 200
    store = app.state.skycache
    young = store.root / ".synth.tmp-0123abcd"
    stale = store.root / ".synth.tmp-89abcdef"
    for leftover in (young, stale):
        shutil.copytree(store.catalog_path("synth"), leftover)  # holds a skycache.json
    old = time.time() - 2 * skycache.STALE_SWAP_SECONDS
    os.utime(stale, (old, old))
    assert store.catalogs() == ["synth"]
    response = client.get("/api/v1/skycache/status")
    assert response.status_code == 200
    body = response.json()
    assert [c["catalog"] for c in body["catalogs"]] == ["synth"] and body["catalogs"][0]["definition_current"] is True
    assert sorted(Path(d["path"]).name for d in body["orphaned_swap_dirs"]) == [young.name, stale.name]
    done = store.repair()
    assert done["removed"] == [str(stale)] and young.exists() and not stale.exists()


def test_unreadable_catalog_metadata_is_reported_not_raised(tmp_path, synth):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    app = _app(tmp_path, cat)
    client = TestClient(app)
    client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0, "radius_deg": 0.1})
    (app.state.skycache.catalog_path("synth") / "skycache.json").write_text("{broken", encoding="utf-8")
    status = client.get("/api/v1/skycache/status")
    assert status.status_code == 200 and "JSONDecodeError" in status.json()["catalogs"][0]["error"]
    cone = client.get("/api/v1/skycache/cone", params={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                       "radius_arcsec": 30})
    assert cone.status_code == 503 and "could not be read" in cone.json()["detail"]


def test_mirror_route_reports_store_write_failures_as_503(tmp_path, monkeypatch):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    client = TestClient(_app(tmp_path, cat))
    _replace_failing(monkeypatch, lambda src, dst, _a: src.name.startswith(".synth.tmp-"))
    response = client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                            "radius_deg": 0.1})
    assert response.status_code == 503 and "could not write the local store" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Merging at the Arrow level
# ---------------------------------------------------------------------------


def test_remirror_merges_without_materializing_old_rows(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(300.0, -40.0, 0.4, 3000, seed=23))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=300.0, dec=-40.0, radius_deg=0.2)

    def forbidden(self, catalog):
        raise AssertionError("mirror merge must not convert the store to Python rows")

    real_load_rows = SkyCache.load_rows
    monkeypatch.setattr(SkyCache, "load_rows", forbidden)
    second = _mirror(store, SyntheticTap(cat), synth, ra=300.1, dec=-40.0, radius_deg=0.2)
    monkeypatch.setattr(SkyCache, "load_rows", real_load_rows)
    assert second.rows_replaced > 0
    ids = _stored_ids(store)
    assert len(ids) == len(set(ids))
    assert sorted(ids) == sorted(str(i) for i in np.asarray(cat.ids)[covered_rows(store, cat)])
    by_id = {str(i): k for k, i in enumerate(cat.ids)}
    for row in store.load_rows("synth"):  # every value round-trips exactly after the merge
        want = cat.row(by_id[row.source_id], 0.0)
        want.pop("match_dist")
        assert row.data == want and list(row.data) == list(want)


def test_merging_batches_with_conflicting_column_types_round_trips(tmp_path, synth):
    """A column typed differently by two mirrors becomes JSON-encoded; values stay exact."""
    def rows(values, start):
        out = []
        for k, value in enumerate(values):
            ra = 10.0 + 0.001 * (start + k)
            data = {"source_id": start + k, "ra": ra, "dec": 5.0, "v": value}
            if k % 2:
                data["extra"] = [k, "x"]  # a second column set
            out.append(StoredRow(str(start + k), ra, 5.0, data, provider="tap", retrieved_at="t", columns=COLUMNS[:3]))
        return out

    first = rows([1, 2, None, 4], 0)  # int64
    second = rows(["a", None, "c", "d"], 100)  # string
    third = rows([1.5, None, 2.5, 3.0], 200)  # float64
    store = SkyCache(tmp_path / "store")
    coverage = Moc.from_cone(10.2, 5.0, 1.0, 10)
    batch = skycache._unify_batches([skycache._batch_from_rows(first), skycache._batch_from_rows(second),
                                     skycache._batch_from_rows(third)])
    assert "v" in batch.json_columns and "extra" in batch.json_columns
    store.write_batch(synth, batch, coverage)
    got = {r.source_id: r.data for r in store.load_rows("synth")}
    for row in first + second + third:
        assert got[row.source_id] == row.data and list(got[row.source_id]) == list(row.data)
        assert type(got[row.source_id]["v"]) is type(row.data["v"])
    # Same encoding as writing all rows at once.
    direct = skycache._batch_from_rows(first + second + third)
    assert direct.json_columns == batch.json_columns


# ---------------------------------------------------------------------------
# Tile splitting: timeouts, response-size limit, budget, MAX_TILE_ORDER
# ---------------------------------------------------------------------------


def test_timed_out_root_cone_is_split_into_tiles(tmp_path, synth):
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.2, 1500, seed=13))

    class SlowForWideCones(SyntheticTap):
        async def query(self, catalog, target, radius_arcsec):
            if radius_arcsec > 200.0:
                await asyncio.sleep(5.0)
            return await super().query(catalog, target, radius_arcsec)

    definition = replace(synth, timeout_seconds=0.3)
    store = SkyCache(tmp_path / "store")
    report = _mirror(store, SlowForWideCones(cat), definition, ra=80.0, dec=-20.0, radius_deg=0.1)
    assert report.tiles_split_timeout == 1 and report.tiles_failed == 0
    assert any("timed out" in w and "split into" in w for w in report.warnings)
    assert report.region_covered_fraction == 1.0
    covered = store.coverage("synth").contains_points(cat.ra, cat.dec)
    assert set(np.asarray(cat.ids)[covered].astype(str).tolist()) <= set(_stored_ids(store))


def test_tiles_that_keep_timing_out_fail_after_bounded_splits(tmp_path, synth):
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.05, 50, seed=13))

    class AlwaysSlow(SyntheticTap):
        requests = 0

        async def query(self, catalog, target, radius_arcsec):
            AlwaysSlow.requests += 1
            await asyncio.sleep(5.0)

    definition = replace(synth, timeout_seconds=0.05)
    store = SkyCache(tmp_path / "store")
    with pytest.raises(MirrorError, match="consecutive timeout splits"):
        _mirror(store, AlwaysSlow(cat), definition, ra=80.0, dec=-20.0, radius_deg=0.01, concurrency=64,
                max_queries=5000)
    order = skycache.order_for_resolution(0.005, hi=skycache.MAX_TILE_ORDER)
    tiles = skycache.cone_pixels(80.0, -20.0, 0.01, order).size
    assert not store.has("synth") and tiles > 0
    # The root cone, then TIMEOUT_SPLIT_DEPTH levels of splits (x1, x4, x16 per root tile), no more.
    assert AlwaysSlow.requests == 1 + tiles * sum(4**k for k in range(skycache.TIMEOUT_SPLIT_DEPTH))


def test_response_size_limit_splits_tiles(tmp_path, synth):
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.3, 3000, seed=13))

    class SizeGuard(SyntheticTap):
        async def query(self, catalog, target, radius_arcsec):
            res = await super().query(catalog, target, radius_arcsec)
            if len(res) + len(res.meta["excess_sources"]) > 120:
                raise CatalogQueryError("TAP response exceeded 10000 byte limit.")
            return res

    store = SkyCache(tmp_path / "store")
    report = _mirror(store, SizeGuard(cat), synth, ra=80.0, dec=-20.0, radius_deg=0.2, tile_rows=5000)
    assert any("response size limit" in w for w in report.warnings)
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    inside = angular_sep_deg(80.0, -20.0, cat.ra, cat.dec) <= 0.2
    assert set(np.asarray(cat.ids)[inside].astype(str).tolist()) <= set(_stored_ids(store))


def test_query_budget_is_enforced(tmp_path, synth):
    cat = SynthCatalog(*cap_points(80.0, -20.0, 0.3, 3000, seed=13))
    provider = SyntheticTap(cat)
    store = SkyCache(tmp_path / "store")
    report = _mirror(store, provider, synth, ra=80.0, dec=-20.0, radius_deg=0.2, tile_rows=300, max_queries=3,
                     concurrency=1)
    assert report.queries == 3 == len(provider.calls)
    assert report.tiles_complete == 2 and report.tiles_failed > 0
    assert any("stopped after 3 archive queries" in w for w in report.warnings)
    assert 0.0 < report.region_covered_fraction < 0.5
    with pytest.raises(CoverageError):
        run(LocalProvider(store).query(synth, validate_target(80.0, -20.0), 600.0))


def test_tile_truncated_at_max_order_is_reported_and_left_uncovered(tmp_path, synth):
    ra, dec = cap_points(80.0, -20.0, 0.01, 30, seed=13)
    clump = (80.001, -20.001)
    cat = SynthCatalog(np.concatenate([ra, [clump[0]] * 12]), np.concatenate([dec, [clump[1]] * 12]))
    store = SkyCache(tmp_path / "store")
    report = _mirror(store, SyntheticTap(cat), synth, ra=80.0, dec=-20.0, radius_deg=0.008, tile_rows=5,
                     max_queries=5000)
    failures = [w for w in report.warnings if w.startswith("order-")]
    assert report.tiles_failed == len(failures) >= 1 and report.tiles_complete > 10
    assert all(w.startswith(f"order-{skycache.MAX_TILE_ORDER} ") and w.endswith(
        f"still truncated at order {skycache.MAX_TILE_ORDER}") for w in failures)
    assert not any("stopped after" in w for w in report.warnings)
    assert not store.covers("synth", *clump, 0.1).covered
    remote = SyntheticTap(cat)
    res = run(SkyCacheProvider(remote, LocalProvider(store)).query(synth, validate_target(*clump), 1.0))
    assert res.meta["skycache"]["hit"] is False and len(res) == 12


# ---------------------------------------------------------------------------
# Provenance and data age
# ---------------------------------------------------------------------------


def test_local_provenance_reports_archive_retrieval_time(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    mirrored_at = store.metadata("synth")["mirrors"][-1]["retrieved_at"]
    time.sleep(0.01)
    res = run(LocalProvider(store).query(synth, validate_target(30.0, 10.0), 90.0))
    assert len(res) > 3
    for src in [*res, *res.meta["excess_sources"]]:
        assert src.provenance["retrieved_at"] == mirrored_at
        assert src.provenance["served_at"] > mirrored_at
        assert src.provenance["skycache"]["archive_provider"] == "tap"
        assert src.provenance["skycache"]["mirrored_from"] == synth.endpoint
    info = res.meta["skycache"]
    assert info["row_providers"] == ["tap"] and info["last_mirror_at"] == mirrored_at
    assert info["retrieved_at"] == {"oldest": mirrored_at, "newest": mirrored_at}
    assert store.status()["catalogs"][0]["retrieved_at"] == {"oldest": mirrored_at, "newest": mirrored_at}
    # Maximum age: argument, catalog parameter (not part of the fingerprint) and environment.
    target = validate_target(30.0, 10.0)
    with pytest.raises(CoverageError) as err:
        LocalProvider(store, max_age_days=1e-9).query_sync(synth, target, 30.0)
    assert err.value.reason == "stale"
    aged = replace(synth, parameters={**synth.parameters, "skycache_max_age_days": 1e-9})
    with pytest.raises(CoverageError) as err:
        LocalProvider(store).query_sync(aged, target, 30.0)
    assert err.value.reason == "stale"
    monkeypatch.setenv(skycache.ENV_MAX_AGE_DAYS, "1e-9")
    remote = SyntheticTap(cat)
    res = run(SkyCacheProvider(remote, LocalProvider(store)).query(synth, target, 30.0))
    assert res.meta["skycache"]["reason"] == "stale" and len(remote.calls) == 1
    monkeypatch.setenv(skycache.ENV_MAX_AGE_DAYS, "365")
    assert LocalProvider(store).query_sync(synth, target, 30.0).meta["skycache"]["hit"] is True
    # An empty cone is judged by the age of the mirror that last covered it (here the only one);
    # a region mirrored again is judged by the new mirror (test_skycache_hardening).
    empty = LocalProvider(store, max_age_days=1e-9)
    with pytest.raises(CoverageError, match="days old") as err:
        empty.query_sync(synth, validate_target(30.0, 10.14), 0.001)
    assert err.value.covered_fraction == 1.0


# ---------------------------------------------------------------------------
# Refreshing through wrapped providers reaches the archive
# ---------------------------------------------------------------------------


def test_refresh_through_wrapped_providers_reaches_the_archive(tmp_path, synth):
    cat = SynthCatalog(*cap_points(300.0, -40.0, 0.3, 2000, seed=23))
    remote = SyntheticTap(cat)
    store = SkyCache(tmp_path / "store")
    _mirror(store, remote, synth, ra=300.0, dec=-40.0, radius_deg=0.2)
    victim = int(np.flatnonzero(angular_sep_deg(300.0, -40.0, cat.ra, cat.dec) < 0.03)[0])
    cat.alive[victim] = False  # deleted upstream
    calls_before = len(remote.calls)
    report = _mirror(store, wrap_providers({"tap": remote}, store), synth, ra=300.0, dec=-40.0, radius_deg=0.1)
    assert len(remote.calls) == calls_before + report.queries and report.queries >= 1
    assert str(cat.ids[victim]) not in set(_stored_ids(store))
    got = run(LocalProvider(store).query(synth, validate_target(float(cat.ra[victim]), float(cat.dec[victim])), 1.0))
    assert str(cat.ids[victim]) not in {s.source_id for s in got}
    with pytest.raises(MirrorInputError, match="LocalProvider"):
        _mirror(store, {"tap": LocalProvider(store)}, synth, ra=300.0, dec=-40.0, radius_deg=0.1)


# ---------------------------------------------------------------------------
# Router helpers
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



# ---------------------------------------------------------------------------
# Hierarchical cone pixels (large cones at fine coverage orders)
# ---------------------------------------------------------------------------


def test_cone_moc_is_inclusive_and_never_wider_than_the_pixel_cone_search():
    from astropy_healpix import HEALPix

    rng = np.random.default_rng(12)
    for case in range(30):
        order = int(rng.integers(5, 15))
        ra0 = float(rng.uniform(0, 360))
        dec0 = [89.995, -89.995, 0.0][case % 3] if case < 6 else float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        radius = float(rng.uniform(0.6, 30.0)) * skycache.pixel_resolution_deg(order)
        moc = skycache.cone_moc(ra0, dec0, radius, order)
        searched = skycache.cone_pixels(ra0, dec0, radius, order)
        assert (moc & Moc.from_pixels(order, searched)) == moc  # a subset of astropy-healpix's inclusive search
        # ...that still holds every pixel really touching the cone (dense sub-pixel sampling).
        dropped = searched[~moc.contains_pixels(order, searched)]
        fine = HEALPix(2 ** (order + 5), order="nested")
        for p in dropped.tolist():
            lon, lat = fine.healpix_to_lonlat(np.arange(p * 4**5, (p + 1) * 4**5))
            assert angular_sep_deg(ra0, dec0, lon.deg, lat.deg).min() > radius, (case, p)
        points = cap_points(ra0, dec0, radius, 3000, seed=case)
        assert moc.contains_points(*points).all(), case
