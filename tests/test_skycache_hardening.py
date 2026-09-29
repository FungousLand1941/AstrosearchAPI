"""Offline regression tests of the sky mirror (skycache.py): second adversarial review.

Covers: TAP services that silently cap JSON answers (MAXREC and VOSI output limits), row
identity by position (shared ids, catalogs without ids), definition fingerprints of
planner/builder plans (local-first crossmatch), archives that vary a column's case, HATS
float32 values, the mirror's own circuit breakers and HTTP cache, transient read errors
while merging, cross-process locking of swaps and repairs, deleting catalogs with
leftover backups, per-region data age, and HEALPix pixel-list requests.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import respx
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import offline_client
from test_skycache import (
    COLUMNS,
    SynthCatalog,
    SyntheticTap,
    assert_identical,
    brute_force,
    cap_points,
    covered_rows,
    run,
    synth_definition,
)
from test_skycache_archives import CENTER, MastArchive, Sky, assert_local_equals_remote

import skycache
from crossmatch import AdvancedQuery, QueryBuilder, QueryExecutor, QueryPlanner
from models import DEFAULT_CATALOGS, CatalogRegistry, ColumnMeta, catalog_from_dict, plan_cone, validate_target
from providers import CacheManager, provider_map
from skycache import (
    CoverageError,
    HatsRemoteProvider,
    LocalProvider,
    MirrorError,
    MirrorInputError,
    MirrorStoreError,
    Moc,
    SkyCache,
    SkyCacheProvider,
    StoredRow,
    angular_sep_deg,
    definition_fingerprint,
    mirror_region,
)

ROOT = Path(__file__).resolve().parents[1]


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
    providers = provider if isinstance(provider, dict) else {"tap": provider}
    return run(mirror_region(definition, store=store, providers=providers, **kwargs))


def _positions(rows):
    """The archive's own (raw) ra/dec of stored rows (canonical positions may differ by an ulp)."""
    return sorted((r.data["ra"], r.data["dec"]) for r in rows)


# ---------------------------------------------------------------------------
# TAP services that silently cap JSON answers (SIMBAD: 50000 rows by default)
# ---------------------------------------------------------------------------

CAPPED_HOST = "capped.invalid"
CAPPED_ENDPOINT = f"https://{CAPPED_HOST}/tap/sync"
_CIRCLE = re.compile(r"CIRCLE\('ICRS', ([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")
_CAPABILITIES = """<?xml version="1.0" encoding="UTF-8"?>
<vosi:capabilities xmlns:vosi="http://www.ivoa.net/xml/VOSICapabilities/v1.0"
    xmlns:tr="http://www.ivoa.net/xml/TAPRegExt/v1.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <capability standardID="ivo://ivoa.net/std/TAP" xsi:type="tr:TableAccess">
    <language><name>ADQL</name></language>
    <outputLimit>
      <default unit="row">{default}</default>
      <hard unit="row">{hard}</hard>
    </outputLimit>
    <uploadLimit><hard unit="byte">1000000</hard></uploadLimit>
  </capability>
</vosi:capabilities>"""


class CappedTapService:
    """A TAP service whose JSON answers stop at ``default_limit`` rows unless MAXREC asks for
    more (up to ``hard_limit``), like SIMBAD; with ``honours_maxrec=False`` it always stops
    there. JSON carries no QUERY_STATUS, so a capped answer looks complete."""

    def __init__(self, cat: SynthCatalog, *, default_limit: int, hard_limit: int, honours_maxrec: bool = True,
                 publishes_limits: bool = True) -> None:
        self.cat = cat
        self.default_limit = default_limit
        self.hard_limit = hard_limit
        self.honours_maxrec = honours_maxrec
        self.publishes_limits = publishes_limits
        self.forms: list[dict[str, str]] = []
        self.capability_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/capabilities"):
            self.capability_requests += 1
            if not self.publishes_limits:
                return httpx.Response(404, text="not found")
            return httpx.Response(200, text=_CAPABILITIES.format(default=self.default_limit, hard=self.hard_limit),
                                  headers={"content-type": "text/xml"})
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.forms.append(form)
        query = form["QUERY"]
        top = int(re.match(r"SELECT TOP (\d+)", query).group(1))
        ra0, dec0, radius = (float(v) for v in _CIRCLE.search(query).groups())
        cap = self.default_limit
        if self.honours_maxrec and "MAXREC" in form:
            cap = min(int(form["MAXREC"]), self.hard_limit)
        sep = angular_sep_deg(ra0, dec0, self.cat.ra, self.cat.dec)
        idx = np.flatnonzero(sep <= radius)
        idx = idx[np.lexsort((idx, sep[idx]))][: min(top, cap)]
        names = list(self.cat.row(0, 0.0))
        known = {c.name: c for c in COLUMNS}
        metadata = [{"name": n, "datatype": known[n].datatype if n in known else "double",
                     "unit": known[n].unit if n in known else None, "ucd": known[n].ucd if n in known else None}
                    for n in names]
        rows = [self.cat.row(int(i), float(sep[i])) for i in idx]
        return httpx.Response(200, json={"metadata": metadata, "data": [[r[n] for n in names] for r in rows]})


def _capped_definition(synth):
    return replace(synth, endpoint=CAPPED_ENDPOINT)


def _mirror_capped(service, definition, store, **kwargs):
    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.route(host=CAPPED_HOST).mock(side_effect=service)
            async with offline_client() as client:
                providers = provider_map(client, cache=CacheManager(None))
                return await mirror_region(definition, store=store, providers=providers, **kwargs)

    return run(scenario())


def _assert_local_cones_complete(store, definition, cat, cones):
    """Every archive row of each (covered) cone is in the store; LocalProvider answers it locally."""
    for ra, dec, radius in cones:
        inside, edge = brute_force(cat, ra, dec, radius)
        ids = {int(i) for i in store.cone_search(definition.name, ra, dec, radius, definition=definition).source_ids}
        assert inside <= ids <= inside | edge, (ra, dec, radius)
        got = run(LocalProvider(store).query(definition, validate_target(ra, dec), radius))
        assert got.meta["skycache"]["hit"] is True


def test_mirror_sends_maxrec_so_a_service_default_limit_cannot_cap_a_tile(tmp_path, synth):
    cat = SynthCatalog(*cap_points(150.0, 30.0, 0.25, 1500, seed=31))
    service = CappedTapService(cat, default_limit=300, hard_limit=100_000)
    definition = _capped_definition(synth)
    store = SkyCache(tmp_path / "store")
    report = _mirror_capped(service, definition, store, ra=150.0, dec=30.0, radius_deg=0.2, tile_rows=2000)
    in_region = int((angular_sep_deg(150.0, 30.0, cat.ra, cat.dec) <= 0.2).sum())
    assert in_region > 300  # more than the service's default limit
    assert report.queries == 1 and report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert service.forms[0]["MAXREC"] == "2001" and "SELECT TOP 2001 " in service.forms[0]["QUERY"]
    assert report.rows_fetched == in_region
    sep = angular_sep_deg(150.0, 30.0, cat.ra, cat.dec)
    covered = covered_rows(store, cat)
    assert covered[sep <= 0.2 * (1 - 3 * skycache.COVERAGE_RES_FRACTION)].all() and not covered[sep > 0.2].any()
    assert report.rows_total == int(covered.sum())
    _assert_local_cones_complete(store, definition, cat, [(150.0, 30.0, 600.0), (150.1, 29.9, 120.0),
                                                          (149.85, 30.05, 90.0)])


def test_service_that_ignores_maxrec_is_caught_by_its_published_output_limit(tmp_path, synth):
    """The reviewed failure: an answer silently capped below TOP N+1 must not count as complete."""
    cat = SynthCatalog(*cap_points(150.0, 30.0, 0.25, 1500, seed=31))
    service = CappedTapService(cat, default_limit=300, hard_limit=100_000, honours_maxrec=False)
    definition = _capped_definition(synth)
    store = SkyCache(tmp_path / "store")
    report = _mirror_capped(service, definition, store, ra=150.0, dec=30.0, radius_deg=0.2, tile_rows=2000)
    assert service.capability_requests == 1
    assert report.queries > 1 and report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert any("capped at the service's output limit of 300 rows" in w and "split into" in w
               for w in report.warnings)
    in_region = angular_sep_deg(150.0, 30.0, cat.ra, cat.dec) <= 0.2
    stored = {int(r.source_id) for r in store.load_rows("synth")}
    assert set(np.asarray(cat.ids)[in_region].tolist()) <= stored
    _assert_local_cones_complete(store, definition, cat, [(150.0, 30.0, 600.0), (150.1, 29.9, 120.0),
                                                          (150.18, 30.0, 30.0)])


def test_output_limits_parsed_from_vosi_capabilities():
    simbad = ("<capabilities><capability><outputLimit>\n <default unit=\"row\">50000</default>\n"
              " <hard unit=\"row\">2000000</hard>\n</outputLimit></capability></capabilities>")
    assert skycache._parse_output_limits(simbad) == {50000, 2000000}
    ned = "<cap><tr:outputLimit><tr:default unit='row'>1000000</tr:default></tr:outputLimit></cap>"
    assert skycache._parse_output_limits(ned) == {1000000}
    by_bytes = "<outputLimit><default unit=\"byte\">1000</default><hard>7</hard></outputLimit>"
    assert skycache._parse_output_limits(by_bytes) == {7}  # byte limits are not row counts
    assert skycache._parse_output_limits("") == set()


# ---------------------------------------------------------------------------
# Row identity is the position: shared ids and catalogs without ids
# ---------------------------------------------------------------------------


def test_distinct_sources_sharing_an_id_survive_a_refresh_reaching_only_one(tmp_path, synth):
    """Like 5XMM J113010.7-043132: two sources, one name, 1.04" apart."""
    ra, dec = cap_points(172.5, -4.5, 0.1, 400, seed=5)
    p1, p2 = (172.544867, -4.525809), (172.544579, -4.525778)
    cat = SynthCatalog(np.append(ra, [p1[0], p2[0]]), np.append(dec, [p1[1], p2[1]]))
    cat.ids[-1] = cat.ids[-2]
    shared = str(cat.ids[-1])
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=p2[0], dec=p2[1], radius_deg=0.02)
    assert [r.source_id for r in store.load_rows("synth")].count(shared) == 2
    # A cone that reaches the first source only (P1 at 72.0", P2 at ~73.0").
    centre = (p1[0] + 72.0 / 3600.0 / math.cos(math.radians(p1[1])), p1[1])
    assert angular_sep_deg(*centre, [p1[0], p2[0]], [p1[1], p2[1]]).tolist()[1] * 3600.0 > 72.9
    report = _mirror(store, SyntheticTap(cat), synth, ra=centre[0], dec=centre[1], radius_deg=72.5 / 3600.0)
    assert report.rows_replaced >= 1
    rows = store.load_rows("synth")
    assert sorted((r.ra, r.dec) for r in rows if r.source_id == shared) == sorted([p1, p2])
    target = validate_target(*p2)
    got = run(LocalProvider(store).query(synth, target, 0.5))
    assert got.meta["skycache"]["hit"] is True and [s.source_id for s in got] == [shared]
    assert_identical(got, run(SyntheticTap(cat).query(synth, target, 0.5)))


def test_overlapping_tiles_keep_every_source_of_a_shared_id(tmp_path, synth):
    ra, dec = cap_points(210.0, 45.0, 0.3, 3000, seed=17)
    cat = SynthCatalog(ra, dec)
    rng = np.random.default_rng(3)
    pairs = {}
    for src in range(0, 600, 10):  # 60 distinct sources given the id of another source 1-5" away
        dup = src + 1500
        angle = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(1.0, 5.0) / 3600.0
        cat.ra[dup] = cat.ra[src] + offset * math.cos(angle) / math.cos(math.radians(cat.dec[src]))
        cat.dec[dup] = cat.dec[src] + offset * math.sin(angle)
        cat.ids[dup] = cat.ids[src]
        pairs[src] = dup
    store = SkyCache(tmp_path / "store", partition_rows=300)
    report = _mirror(store, SyntheticTap(cat), synth, ra=210.0, dec=45.0, radius_deg=0.2, tile_rows=100)
    assert report.queries > 10 and report.rows_deduplicated > 0 and report.region_covered_fraction == 1.0
    rows = store.load_rows("synth")
    assert len(_positions(rows)) == len(set(_positions(rows)))  # no row stored twice
    covered = store.coverage("synth").contains_points(cat.ra, cat.dec)
    stored = set(_positions(rows))
    assert all((float(cat.ra[i]), float(cat.dec[i])) in stored for i in np.flatnonzero(covered))
    remote = SyntheticTap(cat)
    checked = 0
    for src, dup in pairs.items():
        if covered[src] and covered[dup] and angular_sep_deg(210.0, 45.0, cat.ra[src], cat.dec[src]) < 0.17:
            target = validate_target(float(cat.ra[dup]), float(cat.dec[dup]))
            assert_identical(run(LocalProvider(store).query(synth, target, 8.0)), run(remote.query(synth, target, 8.0)))
            checked += 1
    assert checked >= 10


IDLESS_COLUMNS = [c for c in COLUMNS if c.name != "source_id"]


class IdlessTap(SyntheticTap):
    """An archive whose rows have no identifier column (``_sources`` numbers them per answer)."""

    async def query(self, catalog, target, radius_arcsec):
        cone = plan_cone(catalog, target, radius_arcsec)
        self.calls.append((cone.ra, cone.dec, cone.radius_arcsec, cone.row_limit))
        sep = angular_sep_deg(cone.ra, cone.dec, self.cat.ra, self.cat.dec)
        idx = np.flatnonzero((sep <= cone.radius_arcsec / 3600.0) & self.cat.alive)
        idx = idx[np.lexsort((idx, sep[idx]))][: cone.row_limit + 1]
        rows = []
        for i in idx:
            row = self.cat.row(int(i), float(sep[i]))
            row.pop("source_id")
            rows.append(row)
        return self._sources(
            catalog, rows, radius_arcsec, "https://synthetic.invalid/tap/sync", {"QUERY": "synthetic"},
            columns=IDLESS_COLUMNS, target=target, cone=cone,
            meta={"query": "synthetic", "endpoint": "https://synthetic.invalid/tap/sync", "http_method": "POST",
                  "format": "json", "truncated": False, "query_status": None, "cached": False,
                  "columns": [c.as_dict() for c in IDLESS_COLUMNS]},
        )


def test_catalog_without_archive_ids_is_mirrored_completely(tmp_path, synth):
    params = {k: v for k, v in synth.parameters.items() if k != "id_field"}
    params["columns"] = [c for c in params["columns"] if c != "source_id"]
    definition = replace(synth, parameters=params)
    cat = SynthCatalog(*cap_points(45.0, -30.0, 0.3, 2500, seed=8))
    store = SkyCache(tmp_path / "store", partition_rows=400)
    report = _mirror(store, IdlessTap(cat), definition, ra=45.0, dec=-30.0, radius_deg=0.2, tile_rows=100)
    assert report.queries > 20 and report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    rows = store.load_rows("synth")
    assert all(re.fullmatch(r"synth-\d+", r.source_id) for r in rows)  # really no archive ids
    assert len(_positions(rows)) == len(set(_positions(rows)))
    covered = store.coverage("synth").contains_points(cat.ra, cat.dec)
    stored = set(_positions(rows))
    assert all((float(cat.ra[i]), float(cat.dec[i])) in stored for i in np.flatnonzero(covered))
    mirrored_at = store.metadata("synth")["mirrors"][-1]["retrieved_at"]
    remote = IdlessTap(cat)
    for ra, dec, radius in [(45.0, -30.0, 90.0), (45.1, -30.05, 120.0), (44.9, -29.95, 100.0)]:
        target = validate_target(ra, dec)
        got = run(LocalProvider(store).query(definition, target, radius))
        want = run(remote.query(definition, target, radius))
        assert len(got) > 5 and got.meta["skycache"]["hit"] is True
        assert_identical(got, want)
        assert all(s.provenance["retrieved_at"] == mirrored_at for s in got)


# ---------------------------------------------------------------------------
# Definitions built by QueryPlanner / QueryBuilder plans match the mirrored one
# ---------------------------------------------------------------------------


def test_planner_and_builder_definitions_have_the_registry_fingerprint():
    registry = CatalogRegistry()
    executor = QueryExecutor({}, registry=registry)
    query = AdvancedQuery(target=validate_target(187.2779, 2.0524), radius_arcsec=10.0, object_types=["QSO"],
                          spectral_types=["K5V"], count_threshold=3, time_period={"start_year": 2000})
    plans = QueryPlanner(registry).plan(10.0) + QueryBuilder(registry).build(query)
    assert {p.catalog for p in plans} == set(registry.enabled_catalogs())
    for plan in plans:
        base = registry.get(plan.catalog)
        executed = executor.definition_for(plan)
        assert "table" in executed.parameters and "catalog" in executed.parameters  # the plan's copies
        assert definition_fingerprint(executed) == definition_fingerprint(base), plan.catalog
    # A parameters.table that differs from the table would change the executed query: still detected.
    base = registry.get("gaia_dr3")
    odd = replace(base, parameters={**base.parameters, "table": "gaiadr3.other"})
    assert definition_fingerprint(odd) != definition_fingerprint(base)
    assert skycache._signature_changes(skycache.definition_signature(base), skycache.definition_signature(odd)) \
        == ["parameters.table"]


def test_local_first_executor_hits_with_planner_and_builder_plans(tmp_path, synth):
    """CrossmatchService's own plans (QueryPlanner / QueryBuilder) are answered from the store."""
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.15)
    registry = _registry(tmp_path)

    class Offline(SyntheticTap):
        async def query(self, catalog, target, radius_arcsec):
            raise AssertionError("the archive must not be queried for a covered cone")

    executor = QueryExecutor(skycache.wrap_providers({"tap": Offline(cat)}, store), registry=registry)
    target = validate_target(30.0, 10.0)
    query = AdvancedQuery(target=target, radius_arcsec=30.0, object_types=["Star"], time_period={"start_year": 2000})
    plans = QueryPlanner(registry).plan(30.0) + QueryBuilder(registry).build(query)
    assert len(plans) == 2 and all("table" in p.parameters for p in plans)
    successes, failures = run(executor.execute(plans, target))
    assert failures == [] and len(successes) == 2
    for _name, result in successes:
        assert result.meta["skycache"]["hit"] is True and len(result) > 0
    assert store.status(registry)["catalogs"][0]["definition_current"] is True


def test_definition_change_message_names_keys_present_on_one_side_only(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 300, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    with_none = replace(synth, parameters={**synth.parameters, "where": None})
    with pytest.raises(CoverageError, match=r"parameters\.where") as err:
        LocalProvider(store).query_sync(with_none, validate_target(30.0, 10.0), 10.0)
    assert err.value.reason == "definition_changed"


# ---------------------------------------------------------------------------
# Archives that vary the case of a column name (MAST PS1: objName / ObjName)
# ---------------------------------------------------------------------------


class FlippingMast(MastArchive):
    """MAST's PS1 API labels objName as 'objName' or 'ObjName' from one request to the next."""

    def __init__(self, sky, *, always: str | None = None) -> None:
        super().__init__(sky)
        self.always = always

    def __call__(self, request):
        response = super().__call__(request)
        spelling = self.always or ("ObjName" if self.calls % 2 else "objName")
        body = json.loads(response.content)
        for info in body["info"]:
            if info["name"] == "objName":
                info["name"] = spelling
        return httpx.Response(200, json=body)


def _mast_mirror(handler, definition, store, *, radius_deg=0.12, tile_rows=120, cache=None, **kwargs):
    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=handler)
            async with offline_client() as client:
                providers = provider_map(client, cache=cache or CacheManager(None), timeout=kwargs.get("timeout", 30.0))
                report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=radius_deg,
                                             store=store, providers=providers, tile_rows=tile_rows, **kwargs)
                return report, providers

    return run(scenario())


def _ps1():
    return catalog_from_dict("panstarrs_dr2", copy.deepcopy(DEFAULT_CATALOGS["panstarrs_dr2"]))


def _mast_remote(sky_seed, definition, ra, dec, radius):
    archive = MastArchive(Sky(1500, seed=sky_seed))

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=archive)
            async with offline_client() as client:
                provider = provider_map(client, cache=CacheManager(None))["mast"]
                return await provider.query(definition, validate_target(ra, dec), radius)

    return run(scenario())


def test_column_spelled_two_ways_by_the_archive_gets_one_spelling(tmp_path):
    definition = _ps1()
    store = SkyCache(tmp_path / "store", partition_rows=250)
    report, _ = _mast_mirror(FlippingMast(Sky(1500, seed=4)), definition, store)
    assert report.tiles_failed == 0 and report.queries > 5
    colsets = store.metadata("panstarrs_dr2")["colsets"]
    assert all("objName" in c and "ObjName" not in c for c in colsets)
    assert "ObjName" not in {c.name for c in store.column_meta("panstarrs_dr2")}
    local = LocalProvider(store)
    for ra, dec, radius in [(150.25, 20.5, 144.0), (150.28, 20.47, 72.0)]:
        got = run(local.query(definition, validate_target(ra, dec), radius))
        assert len(got) > 10
        assert all("objName" in s.data and "ObjName" not in s.data for s in [*got, *got.meta["excess_sources"]])
        assert {c["name"] for c in got.meta["columns"]} >= {"objName"} and \
            "ObjName" not in {c["name"] for c in got.meta["columns"]}
        # Equal to an archive answer that used the definition's spelling.
        assert_local_equals_remote(got, _mast_remote(4, definition, ra, dec, radius))


def test_consistent_archive_spelling_is_kept_and_legacy_mixed_stores_are_unified(tmp_path, synth):
    rows = []
    for k in range(40):
        ra = 20.0 + 0.0005 * k
        key = "flag" if k % 2 else "FLAG"
        rows.append(StoredRow(str(1000 + k), ra, 5.0, {"source_id": 1000 + k, "ra": ra, "dec": 5.0, key: f"F{k}"},
                              provider="tap", retrieved_at="2026-01-01T00:00:00+00:00",
                              columns=[*COLUMNS[:3], ColumnMeta(key, None, "meta.code", "char")]))
    store = SkyCache(tmp_path / "mixed")
    store.write(synth, rows, Moc.from_cone(20.01, 5.0, 0.1, 14))  # a store written before unification
    got = LocalProvider(store).query_sync(synth, validate_target(20.01, 5.0), 60.0)
    assert len(got) == 40 and all(list(s.data).count("flag") == 1 and "FLAG" not in s.data for s in got)
    assert sorted(s.data["flag"] for s in got) == sorted(f"F{k}" for k in range(40))
    # An archive that always spells it 'FLAG' keeps 'FLAG' (its remote answers do too).
    same = [replace(r, data={("FLAG" if k == "flag" else k): v for k, v in r.data.items()},
                    columns=[*COLUMNS[:3], ColumnMeta("FLAG", None, "meta.code", "char")]) for r in rows]
    store2 = SkyCache(tmp_path / "consistent")
    store2.write(synth, same, Moc.from_cone(20.01, 5.0, 0.1, 14))
    got = LocalProvider(store2).query_sync(synth, validate_target(20.01, 5.0), 60.0)
    assert all("FLAG" in s.data and "flag" not in s.data for s in got)


def test_merge_unifies_spellings_across_mirrors(tmp_path):
    definition = _ps1()
    store = SkyCache(tmp_path / "store", partition_rows=250)
    _mast_mirror(FlippingMast(Sky(1500, seed=4), always="ObjName"), definition, store, radius_deg=0.05,
                 tile_rows=5000)
    colsets = store.metadata("panstarrs_dr2")["colsets"]
    assert len(colsets) == 1 and "ObjName" in colsets[0]  # one consistent mirror: kept as returned
    _mast_mirror(FlippingMast(Sky(1500, seed=4), always="objName"), definition, store, radius_deg=0.05,
                 tile_rows=5000)
    meta = store.metadata("panstarrs_dr2")
    assert all("ObjName" not in c for c in meta["colsets"]) and "ObjName" not in meta["json_columns"]
    names = [r.data.get("objName") for r in store.load_rows("panstarrs_dr2")]
    assert names and all(isinstance(n, str) and n.startswith("PSO J") for n in names)


# ---------------------------------------------------------------------------
# HATS float32 columns come out as the archives print them
# ---------------------------------------------------------------------------


def test_hats_float32_values_equal_the_archive_representation(monkeypatch):
    definition = CatalogRegistry().get("gaia_dr3")
    # Gaia DR3 604917526275831040 (M67): float32 columns as the Gaia TAP service prints them.
    printed = {"ra_error": 0.012019248, "dec_error": 0.00826429, "parallax_error": 0.018420544,
               "phot_g_mean_mag": 17.3276}
    frame = pd.DataFrame({
        "source_id": np.array([604917526275831040], dtype=np.int64),
        "ra": np.array([132.846], dtype=np.float64), "dec": np.array([11.8123], dtype=np.float64),
        "ra_error": np.array([printed["ra_error"]], dtype=np.float32),  # numpy float32
        "dec_error": pd.array([printed["dec_error"]], dtype="Float32"),  # pandas nullable float32
        "parallax_error": pd.Series(pa.array([printed["parallax_error"]], type=pa.float32()),
                                    dtype=pd.ArrowDtype(pa.float32())),  # pyarrow-backed float32
        "phot_g_mean_mag": np.array([printed["phot_g_mean_mag"]], dtype=np.float32),
        "ref_epoch": np.array([2016.0]),
    })
    assert float(np.float32(printed["dec_error"])) != printed["dec_error"]  # widening would change the value
    monkeypatch.setattr(skycache, "lsdb_available", lambda: True)
    provider = HatsRemoteProvider({"gaia_dr3": "memory://gaia"})
    monkeypatch.setattr(provider, "_fetch", lambda url, columns, ra, dec, radius: (frame, "ra", "dec"))
    target = validate_target(132.846, 11.8123)
    got = run(provider.query(definition, target, 5.0))
    assert len(got) == 1
    for name, value in printed.items():
        assert got[0].data[name] == value, name
    # The same CatalogSource as from a row holding the archive's printed values.
    row = {"source_id": 604917526275831040, "ra": 132.846, "dec": 11.8123, **printed, "ref_epoch": 2016.0,
           "match_dist": got[0].data["match_dist"]}
    want = provider._sources(definition, [row], 5.0, "memory://gaia", {}, target=target,
                             cone=plan_cone(definition, target, 5.0),
                             field_map=provider._field_map(definition, default_id=skycache._default_id(definition)))
    assert got[0].data == want[0].data and got[0].metadata == want[0].metadata
    assert got[0].positional_error_arcsec == want[0].positional_error_arcsec
    assert skycache._plain(np.float32(0.018420543521642685)) == 0.018420544
    assert skycache._plain(np.array([np.float32(0.5), np.float32(0.1)])) == [0.5, 0.1]
    assert skycache._plain(np.float64(0.018420543521642685)) == 0.018420543521642685  # float64 untouched
    assert skycache._plain(0.018420543521642685, float32=False) == 0.018420543521642685
    assert skycache._plain(np.float32("nan")) is None and skycache._plain(pd.NA) is None


# ---------------------------------------------------------------------------
# The mirror's own circuit breakers: outages and slow archives (real MAST adapter)
# ---------------------------------------------------------------------------


@pytest.fixture
def quick_breaker(monkeypatch):
    monkeypatch.setattr(skycache, "MIRROR_RECOVERY_S", 0.3)
    monkeypatch.setattr(skycache, "CIRCUIT_POLL_S", 0.05)
    monkeypatch.setattr(skycache, "RETRY_DELAY_S", 0.05)


def test_short_archive_outage_is_ridden_out_through_the_real_adapter(tmp_path, quick_breaker):
    definition = _ps1()
    archive = MastArchive(Sky(1500, seed=4))
    state = {"http": 0, "failed": 0}

    def handler(request):
        state["http"] += 1
        if 2 <= state["http"] <= 31:  # 30 HTTP 503s right after the root cone (10 failed requests)
            state["failed"] += 1
            return httpx.Response(503, text="Service Unavailable")
        return archive(request)

    store = SkyCache(tmp_path / "store", partition_rows=250)
    report, providers = _mast_mirror(handler, definition, store)
    assert state["failed"] == 30
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0, report.warnings
    assert report.tiles_retried > 0
    assert providers["mast"].guards == {}  # the service's own breakers were never used
    got = run(LocalProvider(store).query(definition, validate_target(*CENTER), 72.0))
    assert_local_equals_remote(got, _mast_remote(4, definition, CENTER[0], CENTER[1], 72.0))


def test_open_service_circuit_does_not_block_a_mirror(tmp_path, quick_breaker):
    definition = _ps1()
    archive = MastArchive(Sky(1500, seed=4))

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="catalogs.mast.stsci.edu").mock(side_effect=archive)
            async with offline_client() as client:
                providers = provider_map(client, cache=CacheManager(None))
                guard = providers["mast"]._guard(definition.endpoint)
                for _ in range(guard.failure_threshold):
                    guard.record_failure()  # live crossmatch traffic tripped the service's breaker
                assert guard.state == "open"
                report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.12,
                                             store=SkyCache(tmp_path / "store"), providers=providers, tile_rows=120)
                return report, guard.state

    report, state = run(scenario())
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert state == "open"  # and the mirror did not touch it


def test_slow_archive_is_mirrored_through_timeout_splits(tmp_path, quick_breaker, monkeypatch):
    monkeypatch.setattr(skycache, "CIRCUIT_WAIT_LIMIT_S", 30.0)
    definition = _ps1()
    definition.timeout_seconds = 0.4
    archive = MastArchive(Sky(1500, seed=4))

    async def handler(request):
        if float(request.url.params["radius"]) > 0.03:  # wide cones are slow
            await asyncio.sleep(2.0)
        return archive(request)

    store = SkyCache(tmp_path / "store", partition_rows=250)
    report, _ = _mast_mirror(handler, definition, store, timeout=0.4)
    assert report.tiles_split_timeout > skycache.MIRROR_FAILURE_THRESHOLD  # enough timeouts to open the breaker
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0, report.warnings[:5]
    got = run(LocalProvider(store).query(definition, validate_target(*CENTER), 72.0))
    assert got.meta["skycache"]["hit"] is True


def test_archive_that_dies_mid_mirror_fails_in_bounded_time_and_requests(tmp_path, quick_breaker, monkeypatch):
    monkeypatch.setattr(skycache, "CIRCUIT_WAIT_LIMIT_S", 1.0)
    definition = _ps1()
    archive = MastArchive(Sky(1500, seed=4))
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return archive(request) if calls["n"] == 1 else httpx.Response(503, text="down")

    started = time.perf_counter()
    with pytest.raises(MirrorError, match="did not recover within 1 s") as err:
        _mast_mirror(handler, definition, SkyCache(tmp_path / "store"))
    assert "HTTP 503" in str(err.value)
    assert time.perf_counter() - started < 60.0
    tiles = skycache.cone_pixels(CENTER[0], CENTER[1], 0.12,
                                 skycache.order_for_resolution(0.06, hi=skycache.MAX_TILE_ORDER)).size
    # Without the breaker, every tile would be tried 1 + UNAVAILABLE_RETRIES times (3 HTTP attempts each).
    assert calls["n"] < 0.5 * tiles * 3 * (skycache.UNAVAILABLE_RETRIES + 1)


# ---------------------------------------------------------------------------
# Mirrors never use the HTTP response cache
# ---------------------------------------------------------------------------


def test_refresh_reaches_the_archive_despite_a_shared_response_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    definition = _ps1()
    archive = MastArchive(Sky(1500, seed=4))
    shared = CacheManager(None)  # as main.build_service shares one between all adapters
    store = SkyCache(tmp_path / "store")
    _mast_mirror(archive, definition, store, radius_deg=0.05, tile_rows=5000, cache=shared)
    calls = archive.calls
    stored = {r.source_id for r in store.load_rows("panstarrs_dr2")}
    victim = next(i for i, oid in enumerate(archive.obj_id) if str(oid) in stored)
    archive.ndet[victim] = 0  # removed upstream (filtered by nDetections > 1)
    second, _ = _mast_mirror(archive, definition, store, radius_deg=0.05, tile_rows=5000, cache=shared)
    assert archive.calls == calls + second.queries and second.queries >= 1
    assert str(archive.obj_id[victim]) not in {r.source_id for r in store.load_rows("panstarrs_dr2")}
    assert shared.entry_count == 0  # tile responses never enter the service's cache


# ---------------------------------------------------------------------------
# Reading the previous copy while merging: transient errors never discard it
# ---------------------------------------------------------------------------


def _flaky_reads(monkeypatch, fail):
    real = skycache.pq.read_table
    state = {"n": 0}

    def read_table(path, *args, **kwargs):
        state["n"] += 1
        if fail(state["n"], str(path)):
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(skycache.pq, "read_table", read_table)
    return state


def test_transient_read_error_while_merging_keeps_the_previous_copy(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.6, 5000, seed=1))
    store = SkyCache(tmp_path / "store", partition_rows=300)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.3, tile_rows=100_000)
    before = store.metadata("synth")
    assert len(before["partitions"]) > 3
    _flaky_reads(monkeypatch, lambda n, _path: n == 2)  # one leaf momentarily locked
    report = _mirror(store, SyntheticTap(cat), synth, ra=30.25, dec=10.0, radius_deg=0.05)
    assert not any("damaged" in w or "replaced" in w for w in report.warnings)
    after = store.metadata("synth")
    assert after["rows"] == int(covered_rows(store, cat).sum()) and len(after["mirrors"]) == 2
    assert after["coverage"]["area_deg2"] >= before["coverage"]["area_deg2"]
    assert store.covers("synth", 30.0, 10.0, 60.0).covered


def test_persistent_read_error_aborts_the_mirror_and_changes_nothing(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.4, 2000, seed=1))
    store = SkyCache(tmp_path / "store", partition_rows=300)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.2, tile_rows=100_000)
    before = store.metadata("synth")
    state = _flaky_reads(monkeypatch, lambda _n, path: "Npix=" in path)
    with pytest.raises(MirrorStoreError, match="could not be read after 3 attempts.*nothing was changed"):
        _mirror(store, SyntheticTap(cat), synth, ra=30.15, dec=10.0, radius_deg=0.05)
    assert state["n"] >= skycache.READ_ATTEMPTS
    monkeypatch.undo()
    after = SkyCache(store.root).metadata("synth")
    assert after["version"] == before["version"] and after["rows"] == before["rows"]


def test_damaged_previous_copy_is_replaced_with_a_warning(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.4, 2000, seed=1))
    store = SkyCache(tmp_path / "store", partition_rows=300)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.2, tile_rows=100_000)
    for part in store.catalog_path("synth").rglob("Npix=*.parquet"):
        part.write_bytes(part.read_bytes()[:100])  # truncated leaf files: provably damaged
    report = _mirror(store, SyntheticTap(cat), synth, ra=30.15, dec=10.0, radius_deg=0.05)
    assert any("damaged" in w and "replaced by this mirror" in w for w in report.warnings)
    assert store.coverage("synth").area_deg2 < 0.01  # only the new cone: the damaged copy's coverage is gone
    assert store.metadata("synth")["rows"] == int(covered_rows(store, cat).sum()) > 0
    assert report.rows_fetched == int((angular_sep_deg(30.15, 10.0, cat.ra, cat.dec) <= 0.05).sum())


# ---------------------------------------------------------------------------
# Cross-process locking: opening the store never "repairs" a swap in progress
# ---------------------------------------------------------------------------


def _python(code: str, **kwargs) -> subprocess.CompletedProcess:
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
                          check=True, **kwargs)


def test_another_process_opening_the_store_during_a_swap_retry_changes_nothing(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    root = tmp_path / "store"
    store = SkyCache(root)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    real = os.replace
    seen: dict = {}

    def replace_(src, dst):
        s, d = Path(src), Path(dst)
        if s.name.startswith(".synth.tmp-") and d.name == "synth" and not seen:
            # While the writer backs off, a second process (CLI, another API worker) opens the store.
            out = _python("import json, skycache\n"
                          f"s = skycache.SkyCache(r'{root}')\n"
                          "print(json.dumps({'catalogs': s.catalogs(), 'repair': s.repair()}))")
            seen.update(json.loads(out.stdout.strip().splitlines()[-1]))
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(src, dst)

    monkeypatch.setattr(skycache.os, "replace", replace_)
    report = _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    monkeypatch.setattr(skycache.os, "replace", real)
    assert seen["repair"]["restored"] == [] and seen["repair"]["busy"]  # left alone: the swap was in progress
    reopened = SkyCache(root)
    assert report.rows_total == int(covered_rows(reopened, cat).sum()) > 0
    assert reopened.catalogs() == ["synth"] and reopened.swap_dirs() == []
    assert len(reopened.load_rows("synth")) == report.rows_total


def test_catalog_lock_excludes_other_processes(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    script = ("import sys, skycache\nfrom pathlib import Path\n"
              f"with skycache._exclusive(Path(r'{root}') / 'synth'):\n"
              "    print('locked', flush=True)\n    sys.stdin.readline()\n")
    holder = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)})
    try:
        assert holder.stdout.readline().strip() == "locked"
        with skycache._exclusive(root / "synth", wait=0) as acquired:
            assert acquired is False
        with pytest.raises(MirrorStoreError, match="locked by another process"), skycache._exclusive(root / "synth", wait=0.2):
            pass
        with skycache._exclusive(root / "other", wait=0) as acquired:  # other catalogs are not locked
            assert acquired is True
    finally:
        holder.communicate("\n", timeout=60)
    with skycache._exclusive(root / "synth", wait=0) as acquired:
        assert acquired is True
        with skycache._exclusive(root / "synth", wait=0) as again:  # re-entrant in the holding thread
            assert again is True
        seen = {}
        other = threading.Thread(target=lambda: seen.update(acquired=_try_lock(root / "synth")))
        other.start()
        other.join()
        assert seen["acquired"] is False  # another thread of this process waits its turn too


def _try_lock(path):
    with skycache._exclusive(path, wait=0) as acquired:
        return acquired


def test_failed_swap_never_deletes_a_tree_it_did_not_write(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 500, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    real = os.replace

    def replace_(src, dst):
        s, d = Path(src), Path(dst)
        if s.name.startswith(".synth.tmp-") and d.name == "synth":
            if not d.exists():  # someone else puts a catalog there meanwhile
                backup = next(p for p in d.parent.iterdir() if p.name.startswith(".synth.old-"))
                shutil.copytree(backup, d)
                meta = json.loads((d / "skycache.json").read_text(encoding="utf-8"))
                meta["version"] = "foreign"
                (d / "skycache.json").write_text(json.dumps(meta), encoding="utf-8")
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(skycache.os, "replace", replace_)
    with pytest.raises(MirrorStoreError):
        _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    monkeypatch.setattr(skycache.os, "replace", real)
    assert json.loads((store.catalog_path("synth") / "skycache.json").read_text(encoding="utf-8"))["version"] \
        == "foreign"
    assert [d["kind"] for d in store.swap_dirs()] == ["old"]  # our backup is kept, not restored over it


# ---------------------------------------------------------------------------
# delete() removes leftover backups: a deleted catalog never comes back
# ---------------------------------------------------------------------------


def test_deleted_catalog_is_not_resurrected_from_a_leftover_backup(tmp_path, synth):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    # A restorable backup left by an earlier swap whose removal failed (a file handle on Windows).
    shutil.copytree(store.catalog_path("synth"), store.root / ".synth.old-0123abcd")
    assert store.delete("synth") is True and store.catalogs() == []
    reopened = SkyCache(store.root)
    assert reopened.catalogs() == [] and reopened.swap_dirs() == [] and not reopened.has("synth")
    # Only a leftover backup (the catalog already gone): cleaned too, reported as not mirrored.
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    shutil.move(str(store.catalog_path("synth")), str(store.root / ".synth.old-89abcdef"))
    assert store.delete("synth") is False and store.swap_dirs() == []
    assert SkyCache(store.root).catalogs() == []


def _patch_old_tree_removal(monkeypatch, *, stubborn: bool):
    """Removing '.synth.old-*' trees does nothing (a handle on a file); ``stubborn``: nor can they be renamed."""
    real_rmtree, real_replace = shutil.rmtree, os.replace

    def rmtree(path, ignore_errors=False, **kwargs):
        if Path(path).name.startswith(".synth.old-"):
            if ignore_errors:
                return None
            raise PermissionError(13, "in use")
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    def replace_(src, dst):
        if stubborn and Path(src).name.startswith(".synth.old-"):
            raise PermissionError(13, "in use")
        return real_replace(src, dst)

    monkeypatch.setattr(skycache.shutil, "rmtree", rmtree)
    monkeypatch.setattr(skycache.os, "replace", replace_)
    return lambda: (monkeypatch.setattr(skycache.shutil, "rmtree", real_rmtree),
                    monkeypatch.setattr(skycache.os, "replace", real_replace))


@pytest.mark.parametrize("stubborn", [False, True])
def test_backup_a_swap_cannot_remove_is_never_restorable(tmp_path, synth, monkeypatch, stubborn):
    """The reviewed scenario: _swap's rmtree(backup) fails, then the catalog is deleted and the store reopened."""
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.2, 800, seed=1))
    store = SkyCache(tmp_path / "store")
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    restore = _patch_old_tree_removal(monkeypatch, stubborn=stubborn)
    report = _mirror(store, SyntheticTap(cat), synth, ra=30.05, dec=10.0, radius_deg=0.1)
    leftover = [Path(d["path"]) for d in store.swap_dirs()]
    if stubborn:  # cannot even be renamed: at least stripped of its skycache.json
        assert leftover and not any((p / "skycache.json").exists() for p in leftover)
    else:  # renamed to a '.del-' tree and removed
        assert leftover == []
    assert store.metadata("synth")["rows"] == report.rows_total
    assert store.delete("synth") is True
    restore()
    reopened = SkyCache(store.root)
    assert reopened.catalogs() == [] and not reopened.has("synth") and reopened.swap_dirs() == []


def test_delete_route_reports_undeletable_catalogs_as_503(tmp_path, monkeypatch):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 500, seed=41))
    client = TestClient(_app(tmp_path, cat))
    assert client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "ra": 45.0, "dec": 45.0,
                                                         "radius_deg": 0.1}).status_code == 200
    monkeypatch.setattr(skycache.os, "replace", lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "in use")))
    monkeypatch.setattr(skycache.shutil, "rmtree", lambda path, ignore_errors=False, **kw: None)
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(PermissionError(13, "x")))
    response = client.delete("/api/v1/skycache/synth")
    assert response.status_code == 503 and "could not delete" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Data age: judged by the mirror that last covered the cone
# ---------------------------------------------------------------------------


class _TenDaysAgo(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) - timedelta(days=10)


def _empty_point(store, cat, ra0, dec0, rmin, rmax):
    """A covered position with no row within 5" (an empty cone)."""
    rng = np.random.default_rng(1)
    for _ in range(5000):
        r = rng.uniform(rmin, rmax)
        a = rng.uniform(0, 2 * math.pi)
        ra, dec = ra0 + r * math.cos(a) / math.cos(math.radians(dec0)), dec0 + r * math.sin(a)
        if angular_sep_deg(ra, dec, cat.ra, cat.dec).min() * 3600.0 > 6.0 and store.covers("synth", ra, dec, 5.0).covered:
            return ra, dec
    raise AssertionError("no empty covered cone found")


def test_refreshed_region_is_fresh_even_where_it_has_no_rows(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.25, 1500, seed=1))
    store = SkyCache(tmp_path / "store")
    monkeypatch.setattr(skycache, "datetime", _TenDaysAgo)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.2)
    monkeypatch.setattr(skycache, "datetime", datetime)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)  # refreshed today
    ages = store.metadata("synth")["coverage_ages"]
    assert len(ages) == 2 and ages[0]["retrieved_at"] > ages[1]["retrieved_at"]
    local = LocalProvider(store, max_age_days=5.0)
    fresh_empty = _empty_point(store, cat, 30.0, 10.0, 0.0, 0.08)
    got = local.query_sync(synth, validate_target(*fresh_empty), 5.0)
    assert len(got) == 0 and got.meta["skycache"]["hit"] is True
    assert local.query_sync(synth, validate_target(30.0, 10.0), 60.0).meta["skycache"]["hit"] is True
    # The part only the old mirror covered is stale, rows or not.
    old_empty = _empty_point(store, cat, 30.0, 10.0, 0.12, 0.18)
    with pytest.raises(CoverageError, match="10.0 days old") as err:
        local.query_sync(synth, validate_target(*old_empty), 5.0)
    assert err.value.reason == "stale" and err.value.covered_fraction == 1.0
    with pytest.raises(CoverageError, match="days old"):
        local.query_sync(synth, validate_target(30.0, 10.15), 60.0)
    # A cone straddling both parts is as old as its oldest part.
    with pytest.raises(CoverageError, match="days old"):
        local.query_sync(synth, validate_target(30.0, 10.1), 60.0)
    remote = SyntheticTap(cat)
    res = run(SkyCacheProvider(remote, local).query(synth, validate_target(*old_empty), 5.0))
    assert res.meta["skycache"] == {"hit": False, "reason": "stale", "covered_fraction": 1.0}


def test_legacy_store_without_coverage_ages(tmp_path, synth, monkeypatch):
    cat = SynthCatalog(*cap_points(30.0, 10.0, 0.25, 1500, seed=1))
    store = SkyCache(tmp_path / "store")
    monkeypatch.setattr(skycache, "datetime", _TenDaysAgo)
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.2)
    monkeypatch.setattr(skycache, "datetime", datetime)
    path = store.catalog_path("synth") / "skycache.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("coverage_ages")
    path.write_text(json.dumps(meta), encoding="utf-8")
    store = SkyCache(store.root)
    local = LocalProvider(store, max_age_days=5.0)
    empty = _empty_point(store, cat, 30.0, 10.0, 0.0, 0.08)
    with pytest.raises(CoverageError, match="days old"):  # judged by the catalog's oldest mirror
        local.query_sync(synth, validate_target(*empty), 5.0)
    # The next mirror records ages: the old coverage keeps the old date, the refresh is new.
    _mirror(store, SyntheticTap(cat), synth, ra=30.0, dec=10.0, radius_deg=0.1)
    ages = store.metadata("synth")["coverage_ages"]
    assert len(ages) == 2 and ages[1]["retrieved_at"] == meta["mirrors"][0]["retrieved_at"]
    assert local.query_sync(synth, validate_target(*empty), 5.0).meta["skycache"]["hit"] is True


def test_moc_difference():
    a = Moc([[0, 10], [20, 30], [40, 50]])
    b = Moc([[5, 25], [45, 60]])
    assert (a - b).ranges.tolist() == [[0, 5], [25, 30], [40, 45]]
    assert (a - Moc()).ranges.tolist() == a.ranges.tolist() and (Moc() - a).empty
    assert (a - a).empty and (b - a).ranges.tolist() == [[10, 20], [50, 60]]
    rng = np.random.default_rng(4)
    for _ in range(50):
        x = Moc(np.sort(rng.integers(0, 1000, (8, 2)), axis=1))
        y = Moc(np.sort(rng.integers(0, 1000, (8, 2)), axis=1))
        d = x - y
        assert (d & y).empty and (d | (x & y)) == x


# ---------------------------------------------------------------------------
# HEALPix pixel lists: bounded by the query budget, geometry off the event loop
# ---------------------------------------------------------------------------


def test_pixel_list_longer_than_the_query_budget_is_rejected(tmp_path, synth):
    cat = SynthCatalog(*cap_points(45.0, 45.0, 0.2, 200, seed=41))
    store = SkyCache(tmp_path / "store")
    with pytest.raises(MirrorInputError, match="more than the limit of 20"):
        _mirror(store, SyntheticTap(cat), synth, healpix_order=13, healpix_pixels=list(range(21)), max_queries=20)
    client = TestClient(_app(tmp_path, cat))
    response = client.post("/api/v1/skycache/mirror", json={"catalog": "synth", "healpix_order": 13,
                                                            "healpix_pixels": list(range(1000))})
    assert response.status_code == 422 and "archive queries" in response.json()["detail"]


def test_pixel_geometry_runs_off_the_event_loop_and_the_loop_stays_responsive(tmp_path, synth, monkeypatch):
    order = 12
    first = int(skycache.healpix29([45.0], [45.0])[0] >> (2 * (skycache.SPATIAL_INDEX_ORDER - order)))
    pixels = list(range(first, first + 200))
    ras, decs, _r = skycache.pixel_geometry(order, pixels)
    cat = SynthCatalog(*cap_points(float(ras.mean()), float(decs.mean()), 1.0, 3000, seed=2))
    real = skycache.pixel_geometry
    threads = []

    def spy(order_, pixels_):
        threads.append((len(np.atleast_1d(pixels_)), threading.current_thread() is threading.main_thread()))
        return real(order_, pixels_)

    monkeypatch.setattr(skycache, "pixel_geometry", spy)
    gaps = []

    async def scenario():
        stop = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        try:
            return await mirror_region(synth, healpix_order=order, healpix_pixels=pixels,
                                       store=SkyCache(tmp_path / "store"), providers={"tap": SyntheticTap(cat)},
                                       max_queries=256, max_radius_deg=5.0)
        finally:
            stop.set()
            await beat

    report = run(scenario())
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert threads[0] == (200, False)  # all root pixels at once, in a worker thread
    assert max(gaps) < 0.5, max(gaps)


# ---------------------------------------------------------------------------
# Registry / router helpers
# ---------------------------------------------------------------------------


def _registry(tmp_path):
    _definition, entry = synth_definition()
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(yaml.safe_dump({"catalogs": {"synth": entry}}), encoding="utf-8")
    return CatalogRegistry(registry_file)


def _app(tmp_path, cat):
    registry = _registry(tmp_path)
    app = FastAPI()
    app.include_router(skycache.router)
    app.state.skycache = SkyCache(tmp_path / "store")
    app.state.registry = registry
    app.state.service = SimpleNamespace(providers={"tap": SyntheticTap(cat)}, registry=registry)
    return app
