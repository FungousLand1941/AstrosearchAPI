"""Regression tests for the final review of batch.py.

* VizieR-hosted registry catalogs (VLASS, LoTSS, a table registered with ``vizier add``) are matched through
  CDS XMatch by default (TAPVizieR table uploads stalled for 2 x 300 s per split level, live), an explicit
  ``xmatch`` strategy is accepted for them, and a table XMatch does not serve (LoTSS-DR3: HTTP 400 from its
  column list) goes to the upload strategy with a warning -- or fails with the reason when xmatch was asked for.
  A transient failure of that column list (5xx/429, an HTML maintenance page, a reset) is retried with backoff;
  when it persists the xmatch join is tried anyway with one warning, and nothing is cached.
* A small batch whose upload join stalls goes to per-target cone searches after one short attempt
  (``BATCH_FAST_FALLBACK_SECONDS``), and the stalls open the upload circuit so later batches skip the upload.
* The batch route answers non-finite JSON numbers (NaN, Infinity, 1e400) with 422, not 500.
* ``astrosearch batch`` / the batch route find the catalogs registered with ``vizier add``.

XMatch answers and column lists were recorded live on 2026-09-29 (tests/fixtures/final).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import batch
import vizier
from batch import BatchCrossmatcher, BatchError
from models import CatalogRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "final"
M87 = {"id": "M87", "ra": 187.7059308, "dec": 12.3911233}
C3C273 = {"id": "3C 273", "ra": 187.2779154, "dec": 2.0523883}
REGISTERED = "vizier_i_345_gaia2"


@pytest.fixture(autouse=True)
def fresh_xmatch_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-process cache of XMatch column lists starts empty in every test."""
    monkeypatch.setattr(batch, "_XMATCH_COLUMNS", {})


def _registry_with_registered() -> CatalogRegistry:
    registry = CatalogRegistry()
    data = json.loads((FIXTURES / "registered_i_345_gaia2.json").read_text(encoding="utf-8"))
    vizier.attach_definition(registry, data["name"], data["entry"])
    return registry


class Archive:
    """Offline CDS XMatch (recorded answers) and TAPVizieR whose table uploads never answer."""

    def __init__(self, *, upload_stalls: bool = True) -> None:
        self.requests: list[httpx.Request] = []
        self.upload_stalls = upload_stalls

    def uploads(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST" and b"TAP_UPLOAD" in r.content]

    def xmatch_joins(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST" and r.url.host == "cdsxmatch.u-strasbg.fr"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if "xmatch/api/v1/sync/tables" in url:
            table = request.url.params.get("tabName", "")
            if "lotssdr3" in table:  # recorded: HTTP 400, the table is not in the service
                return httpx.Response(400, content=(FIXTURES / "xmatch_columns_lotssdr3.json").read_bytes())
            name = "vlass" if "914/42" in table else "i_345_gaia2"
            return httpx.Response(200, content=(FIXTURES / f"xmatch_columns_{name}.json").read_bytes(),
                                  headers={"content-type": "application/json"})
        if request.url.host == "cdsxmatch.u-strasbg.fr":
            body = request.content
            name = "vlass" if b"J/ApJ/914/42/table5" in body else "i_345_gaia2"
            return httpx.Response(200, content=(FIXTURES / f"xmatch_{name}.xml").read_bytes(),
                                  headers={"content-type": "text/xml"})
        stalled_upload = request.method == "POST" and b"TAP_UPLOAD" in request.content and self.upload_stalls
        if "tapvizier" in url and stalled_upload:
            raise httpx.ReadTimeout("no answer", request=request)
        if "tapvizier" in url:  # per-target cone searches: empty TAP answers
            return httpx.Response(200, json={"metadata": [{"name": "Source"}], "data": []})
        raise AssertionError(f"unexpected request {request.method} {url}")


async def _run(archive: Archive, targets, catalogs, *, registry=None, strategies=None, radius=5.0,
               **kwargs: Any) -> batch.BatchResult:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=archive)
        async with httpx.AsyncClient() as client:
            engine = BatchCrossmatcher(client=client, registry=registry, **kwargs)
            engine.retry_backoff_seconds = 0.0
            return await engine.run(targets, catalogs, radius_arcsec=radius, strategies=strategies)


# ---------------------------------------------------------------------------
# Finding: VizieR-hosted registry catalogs go through CDS XMatch
# ---------------------------------------------------------------------------


def test_vizier_hosted_registry_catalogs_default_to_xmatch() -> None:
    engine = BatchCrossmatcher(registry=_registry_with_registered())
    for name, table in (("vlass", "vizier:J/ApJ/914/42/table5"), ("lotss", "vizier:J/A+A/707/A198/lotssdr3"),
                        (REGISTERED, "vizier:I/345/gaia2")):
        assert engine.default_strategy(name) == "xmatch", name
        info = engine.strategy_table()[name]
        assert info["endpoint"] == batch.XMATCH_ENDPOINT and info["vizier_table"] == table
    # The other TAP archives keep their uploads; TAPVizieR uploads stay available on request.
    assert engine.default_strategy("nvss") == "upload"
    assert engine._resolve(["vlass"], {"vlass": "upload"}) == [("vlass", "upload")]
    assert engine._resolve(["vlass", REGISTERED], {"vlass": "xmatch", REGISTERED: "xmatch"}) == [
        ("vlass", "xmatch"), (REGISTERED, "xmatch")]
    assert {"vlass", "lotss"} <= set(engine.default_catalogs())


def test_vlass_batch_is_answered_by_xmatch_without_a_tapvizier_upload() -> None:
    """Live: vlass made 10 requests with 5 retries over 1210 s through TAPVizieR uploads; XMatch answered the
    same join in 3.8 s (M87 = VLASS J123049.43+122328.3 at 0.18")."""
    archive = Archive()
    started = time.monotonic()
    result = asyncio.run(_run(archive, [M87, C3C273], ["vlass"]))
    assert time.monotonic() - started < 30.0
    run = result.runs["vlass"]
    assert run.strategy == "xmatch" and run.endpoint == batch.XMATCH_ENDPOINT
    assert not run.errors and run.failed_targets == 0 and run.fallback_targets == 0
    assert archive.uploads() == [] and len(archive.xmatch_joins()) == 1
    assert b"J/ApJ/914/42/table5" in archive.xmatch_joins()[0].content
    [match] = result.target_matches("M87")["vlass"]
    assert match["source_id"] == "J123049.43+122328.3"
    assert match["separation_arcsec"] == pytest.approx(0.18, abs=0.01) and match["confidence"] > 0.99
    assert result.target_matches("3C 273").get("vlass", []) == []


def test_registered_vizier_catalog_is_matched_by_xmatch() -> None:
    """POST /api/v1/batch/crossmatch with [vizier_i_345_gaia2] hung for 200-400 s on TAPVizieR retries; an
    explicit xmatch strategy was refused ('no CDS XMatch (VizieR) view is defined')."""
    registry = _registry_with_registered()
    for strategies in (None, {REGISTERED: "xmatch"}):
        archive = Archive()
        result = asyncio.run(_run(archive, [M87, C3C273], [REGISTERED], registry=registry, strategies=strategies))
        run = result.runs[REGISTERED]
        assert run.strategy == "xmatch" and not run.errors and run.failed_targets == 0, run
        assert archive.uploads() == []
        assert b"I/345/gaia2" in archive.xmatch_joins()[0].content
        assert result.target_matches("M87")[REGISTERED][0]["source_id"] == "3907709439453756032"
        assert result.target_matches("3C 273")[REGISTERED][0]["source_id"] == "3700386905605055360"


def test_table_not_served_by_xmatch_falls_back_to_upload_or_fails_with_the_reason() -> None:
    """LoTSS-DR3 (J/A+A/707/A198) is not in the XMatch service (HTTP 400 from its column list, live)."""
    archive = Archive(upload_stalls=False)
    result = asyncio.run(_run(archive, [M87], ["lotss"], fast_fallback_seconds=1.0))
    run = result.runs["lotss"]
    assert run.strategy == "upload" and archive.xmatch_joins() == [] and len(archive.uploads()) == 1
    assert any("CDS XMatch does not serve vizier:J/A+A/707/A198/lotssdr3" in w for w in run.warnings)
    # Known now: later plans pick the upload strategy directly.
    assert BatchCrossmatcher(registry=CatalogRegistry()).default_strategy("lotss") == "upload"

    archive = Archive()
    batch._XMATCH_COLUMNS.clear()
    result = asyncio.run(_run(archive, [M87], ["lotss"], strategies={"lotss": "xmatch"}))
    run = result.runs["lotss"]
    assert run.failed_targets == 1 and archive.uploads() == [] and archive.xmatch_joins() == []
    assert "the xmatch strategy was requested, but CDS XMatch does not serve" in run.errors[0]


# ---------------------------------------------------------------------------
# Finding: a transient failure of the XMatch column list is not a verdict on the table
# ---------------------------------------------------------------------------


class FlakyColumns(Archive):
    """The XMatch column-list endpoint fails the first ``failures`` probes with ``failure()``."""

    def __init__(self, failures: int, failure, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failures = failures
        self.failure = failure

    def column_probes(self) -> list[httpx.Request]:
        return [r for r in self.requests if "xmatch/api/v1/sync/tables" in str(r.url)]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if "xmatch/api/v1/sync/tables" in str(request.url) and self.failures > 0:
            self.failures -= 1
            self.requests.append(request)
            return self.failure(request)
        return super().__call__(request)


def _http_503(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="Service Unavailable", headers={"retry-after": "0"})


def _maintenance_page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html><body>XMatch is under maintenance</body></html>",
                          headers={"content-type": "text/html"})


def _connection_reset(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadError("connection reset by peer", request=request)


@pytest.mark.parametrize("failure", [_http_503, _maintenance_page, _connection_reset])
def test_transient_column_list_failure_is_retried_then_cached(failure) -> None:
    """Two failed probes, then the column list: the explicit xmatch run succeeds (it failed every target with
    'its table list could not be read (503 ...)' before) and the answer is cached for the next batch."""
    registry = _registry_with_registered()
    archive = FlakyColumns(batch.XMATCH_COLUMNS_ATTEMPTS - 1, failure)
    result = asyncio.run(_run(archive, [M87, C3C273], [REGISTERED], registry=registry,
                              strategies={REGISTERED: "xmatch"}))
    run = result.runs[REGISTERED]
    assert run.strategy == "xmatch" and not run.errors and run.failed_targets == 0 and result.failures == {}
    assert not any("column list" in w for w in run.warnings), run.warnings
    assert len(archive.column_probes()) == batch.XMATCH_COLUMNS_ATTEMPTS
    assert len(archive.xmatch_joins()) == 1
    assert result.target_matches("M87")[REGISTERED][0]["source_id"] == "3907709439453756032"
    assert "vizier:I/345/gaia2" in batch._XMATCH_COLUMNS

    again = FlakyColumns(0, failure)
    result = asyncio.run(_run(again, [M87], [REGISTERED], registry=registry, strategies={REGISTERED: "xmatch"}))
    assert again.column_probes() == [] and result.runs[REGISTERED].failed_targets == 0


@pytest.mark.parametrize("failure", [_http_503, _maintenance_page, _connection_reset])
@pytest.mark.parametrize("explicit", [True, False])
def test_persistent_column_list_failure_still_tries_the_xmatch_join(failure, explicit: bool) -> None:
    """The column list is only a metadata probe: while it keeps failing, the (explicit or default) xmatch run is
    tried anyway with one catalog-level warning -- no per-target failures, no switch to a TAPVizieR upload --
    and the failure is not cached as 'not served'."""
    registry = _registry_with_registered()
    archive = FlakyColumns(10**6, failure)
    result = asyncio.run(_run(archive, [M87, C3C273], [REGISTERED], registry=registry,
                              strategies={REGISTERED: "xmatch"} if explicit else None))
    run = result.runs[REGISTERED]
    assert run.strategy == "xmatch" and not run.errors and run.failed_targets == 0 and result.failures == {}
    assert archive.uploads() == [] and len(archive.xmatch_joins()) == 1
    assert len(archive.column_probes()) == batch.XMATCH_COLUMNS_ATTEMPTS
    [warning] = [w for w in run.warnings if "column list" in w]
    assert warning.startswith(f"{REGISTERED}: the CDS XMatch column list of vizier:I/345/gaia2 could not be read")
    assert "tried without checking" in warning
    assert result.target_matches("M87")[REGISTERED][0]["source_id"] == "3907709439453756032"
    assert batch._XMATCH_COLUMNS == {}
    assert not batch.known_unusable_view(BatchCrossmatcher(registry=registry).xmatch_view(REGISTERED))


def test_column_list_retries_honour_retry_after_with_a_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(batch.asyncio, "sleep", fake_sleep)
    answers = iter([httpx.Response(429, headers={"retry-after": "2"}),
                    httpx.Response(502, headers={"retry-after": "3600"}),
                    httpx.Response(200, json={"metadata": [{"name": "RA_ICRS"}, {"name": "DE_ICRS"}]})])

    async def probe() -> frozenset[str] | None:
        with respx.mock() as router:
            router.get(batch.XMATCH_TABLES_ENDPOINT).mock(side_effect=lambda request: next(answers))
            async with httpx.AsyncClient() as client:
                return await batch.xmatch_table_columns(client, "vizier:I/345/gaia2", backoff_seconds=0.5)

    assert asyncio.run(probe()) == frozenset({"RA_ICRS", "DE_ICRS"})
    assert delays == [2.0, batch.XMATCH_COLUMNS_MAX_RETRY_AFTER_SECONDS]


@pytest.mark.parametrize("status", [403, 404])
def test_non_transient_column_list_error_is_not_retried_or_cached(status: int) -> None:
    calls: list[httpx.Request] = []

    async def probe() -> None:
        with respx.mock() as router:
            router.get(batch.XMATCH_TABLES_ENDPOINT).mock(
                side_effect=lambda request: calls.append(request) or httpx.Response(status))
            async with httpx.AsyncClient() as client:
                await batch.xmatch_table_columns(client, "vizier:I/345/gaia2", backoff_seconds=0.0)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(probe())
    assert len(calls) == 1 and batch._XMATCH_COLUMNS == {}


def test_column_list_timeout_is_not_retried() -> None:
    calls: list[httpx.Request] = []

    def stall(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectTimeout("no answer", request=request)

    async def probe() -> None:
        with respx.mock() as router:
            router.get(batch.XMATCH_TABLES_ENDPOINT).mock(side_effect=stall)
            async with httpx.AsyncClient() as client:
                await batch.xmatch_table_columns(client, "vizier:I/345/gaia2", backoff_seconds=0.0)

    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(probe())
    assert len(calls) == 1 and batch._XMATCH_COLUMNS == {}


# ---------------------------------------------------------------------------
# Finding: a stalled upload endpoint costs one short attempt, then cones; the circuit opens
# ---------------------------------------------------------------------------


def test_stalled_upload_of_a_small_batch_goes_to_cones_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live: `astrosearch batch --catalogs vlass` (1 target) took 603 s: 2 x 300 s ReadTimeouts, then a cone."""
    monkeypatch.setenv("PROVIDER_FAILURE_THRESHOLD", "2")

    async def scenario() -> list[tuple[float, batch.BatchResult, int]]:
        archive = Archive()
        out = []
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=archive)
            async with httpx.AsyncClient() as client:
                engine = BatchCrossmatcher(client=client, fast_fallback_seconds=1.0, upload_timeout=300.0)
                engine.retry_backoff_seconds = 0.0
                for _ in range(3):
                    before = len(archive.uploads())
                    started = time.monotonic()
                    result = await engine.run([M87], ["vlass"], radius_arcsec=5.0, strategies={"vlass": "upload"})
                    out.append((time.monotonic() - started, result, len(archive.uploads()) - before))
        return out

    runs = asyncio.run(scenario())
    for elapsed, result, _uploads in runs:
        run = result.runs["vlass"]
        assert elapsed < 20.0, elapsed  # not 2 x BATCH_UPLOAD_TIMEOUT_SECONDS
        assert run.fallback_targets == 1 and run.failed_targets == 0 and result.failures == {}
    # One upload attempt per batch while the circuit is closed; with the threshold reached the third batch
    # meets the open circuit and goes straight to the cone search.
    assert [uploads for _, _, uploads in runs] == [1, 1, 0]
    assert "circuit is open" in runs[2][1].runs["vlass"].errors[0]


def test_large_batches_keep_the_full_upload_timeout() -> None:
    engine = BatchCrossmatcher(fast_fallback_targets=2, fast_fallback_seconds=1.0)
    assert engine.fast_fallback_targets == 2 and engine.fast_fallback_seconds == 1.0
    assert BatchCrossmatcher(fast_fallback_targets=0).fast_fallback_targets == 0  # disabled


# ---------------------------------------------------------------------------
# Finding: non-finite JSON numbers are a 422 on the batch route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
@pytest.mark.parametrize("where", ["radius", "target"])
def test_batch_route_answers_non_finite_numbers_with_422(token: str, where: str) -> None:
    app = FastAPI()
    app.include_router(batch.router)
    if where == "radius":
        body = f'{{"targets": [{{"id": "a", "ra": 1, "dec": 1}}], "catalogs": ["simbad"], "radius_arcsec": {token}}}'
    else:
        body = f'{{"targets": [{{"id": "a", "ra": {token}, "dec": 1}}], "catalogs": ["simbad"]}}'
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=AssertionError("no upstream request expected"))
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/batch/crossmatch", content=body,
                                   headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "finite" in json.dumps(detail)


# ---------------------------------------------------------------------------
# The CLI / route engine reads the registry `vizier add` writes
# ---------------------------------------------------------------------------


def test_default_engine_includes_catalogs_registered_with_vizier_add(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "catalogs.yaml"
    data = json.loads((FIXTURES / "registered_i_345_gaia2.json").read_text(encoding="utf-8"))
    vizier.save_definition(data["name"], data["entry"], path=path)
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(path))
    engine = BatchCrossmatcher()
    assert REGISTERED in engine.registry.catalogs and "gaia_dr3" in engine.registry.catalogs
    assert engine.default_strategy(REGISTERED) == "xmatch"
    with pytest.raises(BatchError, match="unknown catalog"):
        engine._resolve(["vizier_no_such_table"], None)


# ---------------------------------------------------------------------------
# Finding: owned HTTP clients reload the CA bundle on the event loop
# ---------------------------------------------------------------------------


def test_owned_clients_share_one_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """BatchCrossmatcher.run without a client, SesameResolver without one and the AI fallback
    client all use providers.shared_ssl_context: the CA bundle is loaded once per process and CA
    location (off the event loop the first time), not on the loop for every client. That it
    honours SSL_CERT_FILE / SSL_CERT_DIR like httpx is covered in tests/test_ssl_context.py."""
    import ssl

    import ai
    import providers
    from providers import SesameResolver

    loads: list[int] = []
    real = ssl.create_default_context

    def counting(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        loads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", counting)
    providers.shared_ssl_context.cache_clear()
    created: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def recording(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        assert kwargs.get("verify") is providers.shared_ssl_context(), "an owned client builds its own SSL context"
        client = real_client(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(providers.httpx, "AsyncClient", recording)

    async def scenario() -> None:
        for _ in range(3):
            engine = BatchCrossmatcher(registry=CatalogRegistry())
            engine.retry_backoff_seconds = 0.0
            with respx.mock(assert_all_called=False) as router:
                router.route().mock(side_effect=Archive())
                await engine.run([{"ra": 10.0, "dec": 20.0}], ["vlass"], radius_arcsec=5.0,
                                 strategies={"vlass": "cone"})
            resolver = SesameResolver()
            assert resolver.client is None  # nothing is built (nor any CA bundle loaded) at construction
            async with respx.mock(assert_all_called=False) as router:
                router.route().respond(503)
                with pytest.raises(Exception):  # noqa: B017 - only the client it builds matters here
                    await resolver.resolve("M31")
            assert resolver.client is not None
            await resolver.client.aclose()
            async with ai._state_http_client(object()):
                pass
            async with ai._http_client_factory():
                pass

    asyncio.run(scenario())
    assert len(created) == 12  # batch, resolver, AI state fallback, AI CLI factory, three times over
    assert sum(loads) == 1, f"the CA bundle was loaded {sum(loads)} times"
    providers.shared_ssl_context.cache_clear()


# ---------------------------------------------------------------------------
# Finding: /api/v1/batch/crossmatch wrapped an out-of-range RA (387.2 -> 27.2) instead of refusing it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ra, dec, message", [
    (387.2, 2.0, "RA must be within [0, 360)"), (360.0, 2.0, "RA must be within [0, 360)"),
    (-10.0, 2.0, "RA must be within [0, 360)"), ("-1e-14", 0, "RA must be within [0, 360)"),
    (187.2, 90.5, "DEC must be within [-90, 90]"), (187.2, -91, "DEC must be within [-90, 90]"),
])
def test_batch_route_refuses_out_of_range_coordinates_per_target(ra: Any, dec: Any, message: str) -> None:
    """A mistyped RA such as 387.2 for 187.2 searched another part of the sky and answered 200. The target is
    refused (422 naming it, as a bad Dec or epoch is, and as POST /api/v1/search refuses ra 400), never wrapped."""
    app = FastAPI()
    app.include_router(batch.router)
    body = {"targets": [{"id": "ok", "ra": 10.0, "dec": 1.0}, {"id": "typo", "ra": ra, "dec": dec}],
            "catalogs": ["simbad"]}
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=AssertionError("no upstream request expected"))
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/batch/crossmatch", json=body)
    assert response.status_code == 422, response.text
    detail = json.dumps(response.json()["detail"])
    assert "'typo'" in detail and message in detail, detail
    with pytest.raises(BatchError, match=r"target 'typo': " + message.replace("[", r"\[").replace(")", r"\)")):
        batch.parse_targets([{"id": "typo", "ra": ra, "dec": dec}])


def test_batch_csv_upload_refuses_out_of_range_ra() -> None:
    with pytest.raises(BatchError, match=r"target '2': RA must be within \[0, 360\) degrees \(got 387\.2\)"):
        batch.read_targets_csv(b"ra,dec\n187.2,2\n387.2,2\n")
    assert batch.parse_targets([{"ra": 0, "dec": -90}, {"ra": 359.9999999, "dec": 90}])[1].target.ra == 359.9999999
