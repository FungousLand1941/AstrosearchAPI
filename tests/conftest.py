"""Shared pytest fixtures: offline replay of recorded archive responses."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
import respx

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Deterministic, fast, network-free defaults for the offline suite.
os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "1000"
os.environ.pop("REDIS_URL", None)

from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client

from models import CatalogRegistry
from providers import CacheManager, EndpointGuard, provider_map


@pytest.fixture
def registry() -> CatalogRegistry:
    return CatalogRegistry()


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with offline_client() as client:
        yield client


@pytest.fixture
def providers(http_client: httpx.AsyncClient):
    guards: dict[str, EndpointGuard] = {}
    return provider_map(http_client, timeout=30.0, guards=guards, cache=CacheManager(None))


@pytest.fixture
def replay() -> Callable[..., respx.MockRouter]:
    """Return a context-manager factory that serves recorded fixtures for a target."""

    def factory(target: str, catalogs: list[str] | None = None) -> respx.MockRouter:
        router = respx.mock(assert_all_called=False, assert_all_mocked=True)
        router.route().mock(side_effect=replay_side_effect(load_exchanges(target, catalogs)))
        return router

    return factory
