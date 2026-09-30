"""Final-review regressions for sed.py (offline).

A SED call without a caller-supplied client used to build its private httpx client with httpx's default SSL setup,
``ssl.create_default_context(cafile=certifi.where())``, synchronously on the event loop for every call (~0.7-1.3 s on
Windows, several seconds under load). A caller's ``asyncio.wait_for`` could not cancel the call until that finished, so
'cancelling a SED' took client-build time + the timeout (test_sed_round5's cancellation test failed at 1.5 s and more).
Private clients now share one process-wide SSL context, built off the event loop, and are always closed, even when
the call is cancelled.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Any

import httpx
import pytest
import respx
from test_sed import replay_record
from test_sed_round3 import offline_filters
from test_sed_round5 import SESAME, _replay_3c273_except_sdss

import sed


@pytest.fixture(autouse=True)
def _fresh_backoff() -> Any:
    sed.default_backoff().clear()
    yield
    sed.default_backoff().clear()


async def _hang(request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(3)
    raise httpx.ReadTimeout("late", request=request)


def _lookups() -> list[str]:
    names = (t.get_coro().__qualname__ for t in asyncio.all_tasks() if not t.done())
    return [n for n in names if n.startswith("fetch_") or n == "FilterCatalog._fetch"]


def _tracked_clients(monkeypatch: pytest.MonkeyPatch) -> list[httpx.AsyncClient]:
    created: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    class Tracked(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", Tracked)
    return created


async def test_slow_ssl_setup_does_not_delay_cancelling_an_owned_sed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SSL context of a private client is built off the event loop: wait_for(..., 0.5) cancels at ~0.5 s even
    when building it takes 1.5 s (it used to block the loop, so the cancellation came after it)."""
    rec = await replay_record("3c273")
    real = ssl.create_default_context

    def slow(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        time.sleep(1.5)
        return real(*args, **kwargs)

    sed._ssl_context.cache_clear()
    monkeypatch.setattr(ssl, "create_default_context", slow)
    created = _tracked_clients(monkeypatch)
    try:
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.route().mock(side_effect=_replay_3c273_except_sdss(_hang))
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(sed.sed_from_record(rec, filters=offline_filters(), deadline_seconds=20.0),
                                       0.5)
            assert time.monotonic() - started < 1.2
            assert _lookups() == []
            assert [c for c in created if not c.is_closed] == []
            await asyncio.sleep(1.3)  # let the context build finish (it is cached for the next call)
    finally:
        monkeypatch.undo()
    assert sed._ssl_context.cache_info().currsize == 1


async def test_cancelled_owned_sed_closes_its_client_and_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = await replay_record("3c273")
    sed._ssl_context()  # warm: this test is about the cancellation itself
    created = _tracked_clients(monkeypatch)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=_replay_3c273_except_sdss(_hang))
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sed.sed_from_record(rec, filters=offline_filters(), deadline_seconds=20.0), 0.5)
        assert time.monotonic() - started < 1.0
        assert _lookups() == []
        assert created, "sed_from_record made no private client?"
        assert [c for c in created if not c.is_closed] == []


async def test_owned_clients_reuse_one_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """No private client rebuilds an SSL context (httpx's default builds one from certifi per client)."""
    context = sed._ssl_context()
    builds: list[Any] = []
    real = ssl.create_default_context

    def counting(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        builds.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", counting)
    created = _tracked_clients(monkeypatch)
    payload = (SESAME / "t8main.xml").read_bytes()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/").mock(return_value=httpx.Response(200, content=payload))
        first = await sed.resolve_name("2MASSI J0415195-093506")
        second = await sed.resolve_name("2MASSI J0415195-093506")
    assert first["ra"] == second["ra"]
    assert builds == []
    assert len(created) == 2 and all(c.is_closed for c in created)
    assert all(c._transport._pool._ssl_context is context for c in created)  # type: ignore[attr-defined]
