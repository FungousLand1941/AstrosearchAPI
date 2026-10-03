"""Test helpers for replaying recorded archive responses (import from tests)."""

from __future__ import annotations

import functools
import ssl

import httpx
import respx
from fixture_io import load_exchanges, replay_side_effect, target_for

from crossmatch import CrossmatchService, QueryExecutor
from models import CatalogRegistry, QueryPlan, Target
from providers import CacheManager, provider_map


@functools.lru_cache(maxsize=1)
def shared_ssl_context() -> ssl.SSLContext:
    """One SSL context for all offline clients (creating one costs ~0.6 s on Windows)."""
    return ssl.create_default_context()


def offline_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0)


async def run_catalog(
    name: str,
    target_key: str,
    *,
    radius_arcsec: float = 10.0,
    registry: CatalogRegistry | None = None,
):
    """Query one registry catalog against recorded fixtures; returns (sources|None, failure|None)."""
    reg = registry or CatalogRegistry()
    catalog = reg.get(name)
    exchanges = load_exchanges(target_key, [name])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            executor = QueryExecutor(provider_map(client, cache=CacheManager(None)), registry=reg)
            plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, radius_arcsec, catalog.wavelength)
            successes, failures = await executor.execute([plan], target_for(target_key))
    return (successes[0][1] if successes else None), (failures[0] if failures else None)


def target_of(key: str) -> Target:
    return target_for(key)


def make_service(client: httpx.AsyncClient, registry: CatalogRegistry | None = None, **kwargs) -> CrossmatchService:
    reg = registry or CatalogRegistry()
    return CrossmatchService(reg, provider_map(client, cache=CacheManager(None)), **kwargs)
