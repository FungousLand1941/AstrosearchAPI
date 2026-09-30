"""Regression tests for the final review of batch.py.

* VizieR-hosted registry catalogs (VLASS, LoTSS, a table registered with ``vizier add``) are matched through
  CDS XMatch by default (TAPVizieR table uploads stalled for 2 x 300 s per split level, live), an explicit
  ``xmatch`` strategy is accepted for them, and a table XMatch does not serve (LoTSS-DR3: HTTP 400 from its
  column list) goes to the upload strategy with a warning -- or fails with the reason when xmatch was asked for.
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
