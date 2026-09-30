"""Shared pytest fixtures: offline replay of recorded archive responses, and a hermetic environment.

Every test session gets its own sky cache, user catalog registry and dataset store (temporary
directories), so a developer's ~/.astrosearch mirrors or registered VizieR tables never change
what the offline suite sees. Offline runs pace archive adapters at 1000 requests/s (they only
talk to respx); ``-m live`` runs keep the polite production pacing of 5 requests/s.
"""

from __future__ import annotations

import atexit
import gc
import os
import shutil
import sys
import tempfile
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

# Per-session stores: nothing is read from or written to ~/.astrosearch or ./datasets.
_SESSION_DIR = Path(tempfile.mkdtemp(prefix="astrosearch-tests-"))
atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)
os.environ["SKYCACHE_PATH"] = str(_SESSION_DIR / "skycache")
os.environ["CATALOG_REGISTRY_PATH"] = str(_SESSION_DIR / "catalogs.yaml")
os.environ["DATASET_STORAGE_PATH"] = str(_SESSION_DIR / "datasets")

from fixture_io import load_exchanges, replay_side_effect
from helpers import offline_client

from models import CatalogRegistry
from providers import CacheManager, EndpointGuard, provider_map


def _selects_live(markexpr: str) -> bool:
    expression = " ".join(markexpr.split())
    return "live" in expression and "not live" not in expression


def pytest_collection_finish(session: pytest.Session) -> None:
    """Move everything alive after collection (the imported science stack, the collected test
    modules and their module-level data) out of the cyclic garbage collector's reach. These
    objects live for the whole session anyway; without this, every full collection later in
    the run re-scans them, and those stop-the-world pauses grow with the suite (they pushed the
    event-loop stall measured by the batch timing tests past its 0.5 s budget). Objects created
    by the tests themselves are collected as usual."""
    gc.collect()
    gc.freeze()


def pytest_configure(config: pytest.Config) -> None:
    # A live run talks to real archives: keep their request pacing (PROVIDER_REQUESTS_PER_SECOND
    # default 5) instead of the offline 1000 requests/s.
    if _selects_live(config.getoption("markexpr") or ""):
        os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "5"


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
