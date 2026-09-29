"""Offline replay of real archive responses through the sky mirror (no network).

Fixtures (recorded with ``tests/test_skycache_live.py record``): the Gaia DR3 (ESA TAP) and
2MASS PSC (IRSA TAP) answers to the 0.2 deg mirror query around 3C 273, and the remote
answers to six cones inside it. Replay is strict (fixture_io): a mirror or cone request that
differs from the recorded one fails instead of being served a stale response.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client
from test_skycache_live import (
    CENTER,
    CONES,
    CONES_SET,
    GAIA_3C273,
    MIRROR_CATALOGS,
    MIRROR_RADIUS_DEG,
    MIRROR_SET,
    PARTIAL_CONE,
    TWOMASS_3C273,
    compare_results,
    cone_target,
)

import skycache
from crossmatch import QueryExecutor
from models import CatalogRegistry, QueryPlan, validate_target
from providers import CacheManager, provider_map
from skycache import CoverageError, LocalProvider, SkyCache, mirror_region, wrap_providers


def replay_router(fixture_set: str, catalog: str) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route().mock(side_effect=replay_side_effect(load_exchanges(fixture_set, [catalog])))
    return router


async def mirror_from_fixtures(catalog: str, store: SkyCache):
    definition = CatalogRegistry().get(catalog)
    with replay_router(MIRROR_SET, catalog):
        async with offline_client() as client:
            providers = provider_map(client, cache=CacheManager(None))
            return await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=MIRROR_RADIUS_DEG,
                                       store=store, providers=providers, timeout=120.0)


async def remote_cones(catalog: str):
    definition = CatalogRegistry().get(catalog)
    out = {}
    with replay_router(CONES_SET, catalog):
        async with offline_client() as client:
            provider = provider_map(client, cache=CacheManager(None))[definition.provider]
            for label, ra, dec, radius, epoch in CONES:
                out[label] = await provider.query(definition, cone_target(ra, dec, epoch), radius)
    return out


@pytest.fixture(scope="module")
def mirrored(tmp_path_factory):
    store = SkyCache(tmp_path_factory.mktemp("skycache_replay") / "store")
    reports = {name: asyncio.run(mirror_from_fixtures(name, store)) for name in MIRROR_CATALOGS}
    return store, reports


@pytest.mark.parametrize("catalog", MIRROR_CATALOGS)
def test_replayed_mirror_local_equals_remote(mirrored, catalog):
    store, reports = mirrored
    report = reports[catalog]
    assert report.queries == 1 and report.tiles_failed == 0 and report.providers_used == ["tap"]
    assert report.region_covered_fraction == 1.0
    # Real row counts of the recorded region (Gaia DR3 COUNT(*) = 511 within 0.2 deg).
    assert report.rows_total == {"gaia_dr3": 511, "twomass_psc": 166}[catalog]
    definition = CatalogRegistry().get(catalog)
    remote = asyncio.run(remote_cones(catalog))
    local = LocalProvider(store)
    for label, ra, dec, radius, epoch in CONES:
        got = asyncio.run(local.query(definition, cone_target(ra, dec, epoch), radius))
        compare_results(got, remote[label], catalog=f"{catalog}/{label}")
        assert got.meta["skycache"]["hit"] is True
    counts = {label: len(remote[label]) for label, *_ in CONES}
    assert counts["center_90"] >= 5 and counts["sw_120"] >= 5
    with pytest.raises(CoverageError) as err:
        asyncio.run(local.query(definition, validate_target(*PARTIAL_CONE[:2]), PARTIAL_CONE[2]))
    assert err.value.reason == "partially_covered"


def test_replayed_mirror_astrophysical_truth(mirrored):
    store, _ = mirrored
    registry = CatalogRegistry()
    gaia = asyncio.run(LocalProvider(store).query(registry.get("gaia_dr3"), validate_target(*CENTER), 2.0))
    assert gaia[0].source_id == GAIA_3C273
    assert 12.5 < gaia[0].data["phot_g_mean_mag"] < 13.3  # 3C 273, V ~ 12.9
    assert gaia[0].epoch == 2016.0 and gaia[0].metadata["query_separation_arcsec"] < 0.1
    # A quasar: Gaia's parallax and proper motion are consistent with zero.
    assert abs(gaia[0].data["parallax"]) < 5 * gaia[0].data["parallax_error"]
    tm = asyncio.run(LocalProvider(store).query(registry.get("twomass_psc"), validate_target(*CENTER), 2.0))
    assert tm[0].source_id == TWOMASS_3C273
    assert 9.5 < tm[0].data["k_m"] < 10.5 and 1997.4 <= tm[0].epoch <= 2001.2
    # Units/UCDs of the archive columns are kept in the store.
    cols = {c.name: c for c in store.column_meta("gaia_dr3")}
    assert cols["ra_error"].unit == "mas" and cols["pmra"].ucd.startswith("pos.pm")
    cols = {c.name: c for c in store.column_meta("twomass_psc")}
    assert cols["ra"].unit == "deg"


def test_wrapped_providers_answer_executor_from_store_without_network(mirrored):
    """QueryExecutor with wrap_providers(): covered cones never touch the network."""
    store, _ = mirrored
    registry = CatalogRegistry()

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True):  # any HTTP request would fail
            async with offline_client() as client:
                providers = wrap_providers(provider_map(client, cache=CacheManager(None)), store)
                executor = QueryExecutor(providers, registry=registry)
                plans = [QueryPlan(n, registry.get(n).provider, registry.get(n).endpoint, {}, 30.0,
                                   registry.get(n).wavelength) for n in MIRROR_CATALOGS]
                return await executor.execute(plans, validate_target(*CENTER))

    successes, failures = asyncio.run(scenario())
    assert failures == []
    by_name = dict(successes)
    assert by_name["gaia_dr3"][0].source_id == GAIA_3C273
    assert by_name["twomass_psc"][0].source_id == TWOMASS_3C273
    assert by_name["gaia_dr3"].meta["skycache"]["hit"] is True


def test_replayed_store_opens_with_lsdb(mirrored):
    store, _ = mirrored
    lsdb = pytest.importorskip("lsdb")
    frame = lsdb.open_catalog(store.catalog_path("gaia_dr3")).cone_search(
        ra=CENTER[0], dec=CENTER[1], radius_arcsec=300.0).compute()
    ours = store.cone_search("gaia_dr3", CENTER[0], CENTER[1], 300.0)
    assert sorted(frame["source_id"].astype(str).tolist()) == sorted(ours.source_ids)
    assert GAIA_3C273 in ours.source_ids


def test_cli_mirror_from_replay(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    skycache.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["mirror", "--catalog", "gaia_dr3", "--ra", str(CENTER[0]), "--dec", str(CENTER[1]),
                              "--radius-deg", str(MIRROR_RADIUS_DEG), "--store", str(tmp_path), "--json"])
    with replay_router(MIRROR_SET, "gaia_dr3"):
        assert args.handler(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows_total"] == 511 and report["region_covered_fraction"] == 1.0
    args = parser.parse_args(["skycache", "cone", "--catalog", "gaia_dr3", "--ra", str(CENTER[0]),
                              "--dec", str(CENTER[1]), "--radius-arcsec", "2", "--store", str(tmp_path)])
    assert args.handler(args) == 0
    assert GAIA_3C273 in capsys.readouterr().out
