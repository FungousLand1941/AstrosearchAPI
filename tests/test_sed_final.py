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
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import load_exchanges, replay_side_effect
from test_sed import FakeService, make_app, replay_record
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


# --- one target rule and one set of messages on GET/POST /api/v1/sed and the CLI (main.check_search_target) ---


@pytest.mark.parametrize("params, detail", [
    ({"name": "M87", "ra": 187.7, "dec": 12.4}, "NAME_AND_COORDINATES"),
    ({"name": "M87", "ra": 187.7}, "NAME_AND_COORDINATES"),
    ({"name": "M87", "dec": 12.4}, "NAME_AND_COORDINATES"),
    ({}, "NAME_OR_COORDINATES"),
    ({"name": ""}, "NAME_OR_COORDINATES"),
    ({"name": "   "}, "NAME_OR_COORDINATES"),
    ({"name": "  ", "ra": 187.7}, "NAME_OR_COORDINATES"),  # a blank name is no name: ra alone is not a target
])
def test_sed_routes_answer_the_shared_target_messages(tmp_path: Path, params: dict[str, Any], detail: str) -> None:
    """Finding: /api/v1/sed answered 'give either name or ra/dec, not both' (GET) and a pydantic error list
    (POST) where every other search route answers main.NAME_AND_COORDINATES as a plain detail."""
    from fastapi.testclient import TestClient

    import main

    service = FakeService({})
    with TestClient(make_app(service, tmp_path)) as client:
        for res in (client.get("/api/v1/sed", params=params), client.post("/api/v1/sed", json=params)):
            assert res.status_code == 422, (params, res.text)
            assert res.json() == {"detail": getattr(main, detail)}, (params, res.text)
    assert service.calls == []


@pytest.mark.parametrize("method, blank", [("GET", ""), ("GET", "   "), ("POST", ""), ("POST", " \t")])
def test_sed_routes_treat_a_blank_name_next_to_coordinates_as_no_name(tmp_path: Path, method: str, blank: str) -> None:
    """Finding: an empty ``name=`` beside ra/dec was a 422 on /api/v1/sed but 'no name' on /search and
    /cutouts. One rule now: a blank name is no name, so ra/dec are searched and nothing is resolved."""
    from fastapi.testclient import TestClient

    rec = asyncio.run(replay_record("3c273"))
    target = {"name": blank, "ra": 187.2779154, "dec": 2.0523883}
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("sed/3c273") + load_exchanges("sed/svo")))
        service = FakeService(rec)
        with TestClient(make_app(service, tmp_path)) as client:
            res = (client.get("/api/v1/sed", params=target) if method == "GET"
                   else client.post("/api/v1/sed", json=target))
        sesame = [c for c in router.calls if "sesame" in c.request.url.path]
    assert res.status_code == 200, res.text
    assert sesame == []
    assert [(ra, dec, kw.get("epoch")) for ra, dec, kw in service.calls] == [(187.2779154, 2.0523883, None)]
    assert not (res.json()["target"].get("name") or "").strip()


def test_sed_cli_uses_the_shared_target_rule(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    import main

    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))
    seen: list[Any] = []

    async def fake_build(ra: Any, dec: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append((ra, dec, kwargs["name"]))
        return {"target": {"ra": ra, "dec": dec}, "points": [], "classification": {}, "redshift": {}}

    monkeypatch.setattr(sed, "build_sed", fake_build)
    args = parser.parse_args(["sed", "--name", "M87", "--ra", "187.7", "--dec", "12.4"])
    assert args.handler(args) == 2
    assert main.NAME_AND_COORDINATES in capsys.readouterr().out
    args = parser.parse_args(["sed", "--name", "  ", "--ra", "187.7"])
    assert args.handler(args) == 2
    assert main.NAME_OR_COORDINATES in capsys.readouterr().out
    args = parser.parse_args(["sed", "--name", " ", "--ra", "187.7", "--dec", "12.4", "--json"])
    assert args.handler(args) == 0
    assert seen == [(187.7, 12.4, None)]
