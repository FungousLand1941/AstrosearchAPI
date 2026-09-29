"""Regression tests for the adversarial review of vizier.py (offline).

Each test names the defect it pins down. Upstream behaviour is replayed from the recordings in
tests/fixtures/vizier (see tests/test_vizier.py) or produced by a scripted httpx transport for
failure modes (timeouts, 5xx, rejected queries, malformed answers, slow siblings).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import unquote_plus

import httpx
import pytest
import respx
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fixture_io import replay_side_effect
from test_vizier import _app, _exchanges, replay_describe, replaying

import vizier
from models import DEFAULT_CATALOGS, K95_1D, CatalogRegistry, compute_positional_error

PYTHON = sys.executable


@pytest.fixture(autouse=True)
def _fresh_describe_cache():
    vizier.clear_describe_cache()
    yield
    vizier.clear_describe_cache()


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(vizier, "RETRY_BACKOFF_SECONDS", 0.0)


def col(name: str, unit: str | None, ucd: str | None, description: str = "", datatype: str = "DOUBLE",
        principal: bool = False) -> vizier.ColumnInfo:
    return vizier.ColumnInfo(name, unit, ucd, datatype, description, principal)


# ---------------------------------------------------------------------------
# Scripted transport
# ---------------------------------------------------------------------------


Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class ScriptedTransport(httpx.AsyncBaseTransport):
    """Answers every request with ``handler`` and records requests and completed responses."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.completed: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        response = await self.handler(request)
        self.completed.append(request)
        return response


def kind_of(request: httpx.Request) -> str:
    body = (request.content or b"").decode("utf-8", "replace")
    if "reg.g-vo.org" in str(request.url):
        return "regtap"
    if "TAP_SCHEMA.tables" in body:
        return "tables"
    if "TAP_SCHEMA.columns" in body:
        return "columns"
    if "-meta.all" in str(request.url):
        return "asu_table"
    if "votable" in request.url.path:
        return "asu_catalog"
    return "tap_other"


def replay_handler(*cases: str, overrides: dict[str, Handler] | None = None) -> Handler:
    replay = replay_side_effect(_exchanges(*cases))

    async def handler(request: httpx.Request) -> httpx.Response:
        special = (overrides or {}).get(kind_of(request))
        if special is not None:
            return await special(request)
        return replay(request)

    return handler


def scripted_client(handler: Handler) -> tuple[httpx.AsyncClient, ScriptedTransport]:
    transport = ScriptedTransport(handler)
    return httpx.AsyncClient(transport=transport, timeout=30.0), transport


async def status(code: int, text: str = "error", content_type: str = "text/plain") -> httpx.Response:
    return httpx.Response(code, text=text, headers={"content-type": content_type})


QUERY_ERROR_VOTABLE = (
    '<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">'
    '<INFO name="QUERY_STATUS" value="ERROR">Incorrect ADQL query: Encountered "("</INFO></RESOURCE></VOTABLE>'
)


# ---------------------------------------------------------------------------
# Issue: registrations wrote a partial registry the core read as complete (18 built-ins lost)
# ---------------------------------------------------------------------------


async def test_registration_into_catalog_registry_path_keeps_every_builtin_for_the_core(tmp_path, monkeypatch):
    path = tmp_path / "core_registry.yaml"
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(path))
    with replaying("describe_2sxps"):
        async with httpx.AsyncClient() as client:
            registration = await vizier.register_table("IX/58/2sxps", client=client)  # default path = env var
    assert Path(registration.path) == path
    expected = set(DEFAULT_CATALOGS) | {"vizier_ix_58_2sxps"}
    assert set(CatalogRegistry(path).catalogs) == expected  # what main/api/ai/batch build
    import main

    service = main.build_service(registry_path=str(path))
    assert set(service.registry.catalogs) == expected
    assert not [p for p in CatalogRegistry(path).validate() if p.startswith("vizier_ix_58_2sxps")]
    assert set(vizier.load_registry(path).catalogs) == expected
    assert list(vizier.list_registered(path)) == ["vizier_ix_58_2sxps"]


def test_embedded_copies_are_ignored_by_the_merged_registry_and_refreshed_on_save(tmp_path):
    path = tmp_path / "catalogs.yaml"
    stale = json.loads(json.dumps(DEFAULT_CATALOGS["gaia_dr3"]))
    stale["description"] = "STALE COPY"
    user = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    path.write_text(yaml.safe_dump({"extends": "embedded", "embedded_copies": ["gaia_dr3"],
                                    "catalogs": {"gaia_dr3": stale, "my_radio": user}}), encoding="utf-8")
    merged = vizier.load_registry(path)
    assert merged.get("gaia_dr3").description == DEFAULT_CATALOGS["gaia_dr3"]["description"]
    assert set(merged.catalogs) == set(DEFAULT_CATALOGS) | {"my_radio"}
    assert list(vizier.list_registered(path)) == ["my_radio"]
    vizier.save_definition("my_radio_2", user, path=path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["catalogs"]["gaia_dr3"]["description"] == DEFAULT_CATALOGS["gaia_dr3"]["description"]
    assert set(document["embedded_copies"]) == set(DEFAULT_CATALOGS)
    assert set(CatalogRegistry(path).catalogs) == set(DEFAULT_CATALOGS) | {"my_radio", "my_radio_2"}


def test_previous_partial_format_is_migrated_on_the_next_save(tmp_path):
    path = tmp_path / "catalogs.yaml"
    user = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    path.write_text(yaml.safe_dump({"extends": "embedded", "catalogs": {"old_user": user}}), encoding="utf-8")
    assert list(vizier.list_registered(path)) == ["old_user"]
    vizier.save_definition("new_user", user, path=path)
    assert set(CatalogRegistry(path).catalogs) == set(DEFAULT_CATALOGS) | {"old_user", "new_user"}
    assert set(vizier.list_registered(path)) == {"old_user", "new_user"}


def test_hand_written_registry_with_comments_is_never_rewritten(tmp_path):
    path = tmp_path / "custom.yaml"
    entry = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    text = "# my curated registry -- keep these notes\n" + yaml.safe_dump({"catalogs": {"only_vlass": entry}})
    path.write_text(text, encoding="utf-8")
    with pytest.raises(vizier.VizierRegistrationError, match="comments"):
        vizier.save_definition("vizier_x", entry, path=path)
    assert path.read_text(encoding="utf-8") == text  # untouched, comments kept


def test_hand_written_registry_without_comments_stays_a_complete_registry(tmp_path, caplog):
    path = tmp_path / "custom.yaml"
    entry = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    path.write_text(yaml.safe_dump({"catalogs": {"only_vlass": entry}}), encoding="utf-8")
    with caplog.at_level("WARNING", logger="astrosearch.vizier"):
        vizier.save_definition("vizier_x", entry, path=path)
    assert any("complete catalog registry" in r.getMessage() for r in caplog.records)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "extends" not in document
    assert set(CatalogRegistry(path).catalogs) == {"only_vlass", "vizier_x"}
    assert set(vizier.load_registry(path).catalogs) == {"only_vlass", "vizier_x"}


# ---------------------------------------------------------------------------
# Issue: unlocked read-modify-write lost concurrent registrations; non-mapping entries crashed
# ---------------------------------------------------------------------------


_WRITER = textwrap.dedent("""
    import json, sys, time
    sys.path.insert(0, {root!r})
    import vizier
    from models import DEFAULT_CATALOGS
    original = vizier._read_document
    def slow_read(target):
        result = original(target)
        time.sleep(0.05)  # widen the read-modify-write window
        return result
    vizier._read_document = slow_read
    entry = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    for i in range(8):
        vizier.save_definition(f"{{sys.argv[2]}}_{{i}}", entry, path=sys.argv[1])
""")


def test_concurrent_registrations_from_two_processes_are_all_kept(tmp_path):
    path = tmp_path / "catalogs.yaml"
    script = tmp_path / "writer.py"
    script.write_text(_WRITER.format(root=str(ROOT)), encoding="utf-8")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    procs = [subprocess.Popen([PYTHON, str(script), str(path), prefix], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) for prefix in ("user_a", "user_b")]
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode("utf-8", "replace")
    names = set(vizier.list_registered(path))
    assert names == {f"{p}_{i}" for p in ("user_a", "user_b") for i in range(8)}


def test_lock_contention_times_out_with_a_conflict(tmp_path, monkeypatch):
    path = tmp_path / "catalogs.yaml"
    script = tmp_path / "holder.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        import vizier
        from pathlib import Path
        with vizier._registry_lock(Path(sys.argv[1])):
            print("locked", flush=True)
            time.sleep(3)
    """), encoding="utf-8")
    holder = subprocess.Popen([PYTHON, str(script), str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == b"locked"
        monkeypatch.setattr(vizier, "REGISTRY_LOCK_TIMEOUT_SECONDS", 0.3)
        with pytest.raises(vizier.VizierConflictError, match="locked"):
            vizier.save_definition("x_user", json.loads(json.dumps(DEFAULT_CATALOGS["vlass"])), path=path)
    finally:
        holder.communicate(timeout=30)


def test_non_mapping_entries_are_reported_not_crashing(tmp_path, capsys):
    path = tmp_path / "catalogs.yaml"
    good = json.loads(json.dumps(DEFAULT_CATALOGS["vlass"]))
    path.write_text(yaml.safe_dump({"extends": "embedded", "catalogs": {"broken": 1, "fine": good}}), encoding="utf-8")
    assert list(vizier.list_registered(path)) == ["fine"]
    assert vizier.registered_entries(path) == ({"fine": good}, ["broken"])
    client = TestClient(_app(tmp_path))
    response = client.get("/api/v1/vizier/registered")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1 and body["invalid"] == ["broken"] and body["catalogs"][0]["name"] == "fine"
    parser_args = ["list", "--registry-path", str(path)]
    assert vizier.main(parser_args) == 0
    out = capsys.readouterr().out
    assert "fine" in out and "entry 'broken' is not a mapping" in out
    assert set(vizier.load_registry(path).catalogs) == set(DEFAULT_CATALOGS) | {"fine"}


# ---------------------------------------------------------------------------
# Issue: malformed overrides gave HTTP 500 / tracebacks or were saved silently
# ---------------------------------------------------------------------------


BAD_OVERRIDES = [
    {"max_rows": "abc"},
    {"max_rows": 0},
    {"max_rows": True},
    {"max_rows": 12.5},
    {"timeout_seconds": "fast"},
    {"timeout_seconds": -1},
    {"pos_error": "Err90"},
    {"pos_error": {"columns": "Err90"}},
    {"pos_error": {"columns": ["Err90"], "divisor": "two"}},
    {"epoch_range": 5, "epoch": None},
    {"epoch_range": [2020, 2010]},
    {"epoch": True},
    {"epoch": 3000},
    {"epoch": {"span_columns": "MJD0"}},
    {"epoch_format": "unix"},
    {"profiles": "xray"},
    {"profiles": [1, 2]},
    {"extra_columns": "Err90"},
    {"enabled": "false"},
    {"enabled": 0},
    {"wavelength": 5},
    {"wavelength": "neutrino"},
    {"systematic_arcsec": -1},
    {"systematic_arcsec": "0.5"},
    {"description": 3},
    {"id_column": ["a"]},
    {"endpoint": "https://evil.example/tap"},
]


@pytest.mark.parametrize("overrides", BAD_OVERRIDES, ids=[json.dumps(o) for o in BAD_OVERRIDES])
async def test_bad_overrides_are_rejected_before_any_request(overrides, tmp_path):
    async def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected, got {request.url}")

    client, transport = scripted_client(fail)
    async with client:
        with pytest.raises(vizier.VizierInputError):
            await vizier.register_table("IX/58/2sxps", overrides=overrides, path=tmp_path / "r.yaml", client=client)
    assert transport.requests == []
    assert not (tmp_path / "r.yaml").exists()


def test_bad_overrides_are_422_from_the_api(tmp_path):
    client = TestClient(_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for overrides in ({"max_rows": "abc"}, {"timeout_seconds": "fast"}, {"pos_error": "Err90"},
                          {"epoch_range": 5, "epoch": None}, {"profiles": "xray"}, {"extra_columns": "Err90"},
                          {"enabled": "false"}):
            response = client.post("/api/v1/vizier/register", json={"table_id": "IX/58/2sxps", "overrides": overrides})
            assert response.status_code == 422, (overrides, response.text)
        assert router.calls.call_count == 0
    assert not (tmp_path / "catalogs.yaml").exists()


async def test_valid_overrides_are_normalised_into_the_entry():
    desc = await replay_describe("IX/58/2sxps")
    _, entry = vizier.build_definition(desc, overrides={
        "enabled": False, "profiles": ["xray", "mine"], "max_rows": 50.0, "timeout_seconds": 30,
        "wavelength": "X-ray", "epoch": 2010, "epoch_range": None, "extra_columns": ["IAUName"],
    })
    assert entry["enabled"] is False and entry["profiles"] == ["xray", "mine"]
    assert entry["max_rows"] == 50 and isinstance(entry["max_rows"], int) and entry["timeout_seconds"] == 30.0
    assert entry["wavelength"] == "xray" and entry["epoch"] == 2010.0
    assert '"IAUName"' in entry["parameters"]["columns"]


def test_cli_rejects_non_object_or_badly_typed_overrides_without_network(tmp_path, capsys):
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        assert vizier.main(["add", "IX/58/2sxps", "--registry-path", str(tmp_path / "r.yaml"),
                            "--overrides", "[1, 2]"]) == 2
        assert "must be a JSON object" in capsys.readouterr().out
        assert vizier.main(["add", "IX/58/2sxps", "--registry-path", str(tmp_path / "r.yaml"),
                            "--overrides", '{"max_rows": "abc"}']) == 2
        assert "max_rows" in capsys.readouterr().out
        assert router.calls.call_count == 0


async def test_builtin_or_malformed_names_are_refused_before_describe(tmp_path):
    async def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client, transport = scripted_client(fail)
    async with client:
        with pytest.raises(vizier.VizierRegistrationError, match="built-in"):
            await vizier.register_table("IX/58/2sxps", name="gaia_dr3", path=tmp_path / "r.yaml", client=client)
        with pytest.raises(vizier.VizierInputError, match="Invalid catalog name"):
            await vizier.register_table("IX/58/2sxps", name="Bad Name!", path=tmp_path / "r.yaml", client=client)
    assert transport.requests == []


# ---------------------------------------------------------------------------
# Issue: describe() left sibling requests running; optional ASU metadata was fatal
# ---------------------------------------------------------------------------


async def test_failed_columns_query_cancels_the_slow_asu_siblings():
    async def columns_rejected(request: httpx.Request) -> httpx.Response:
        return await status(400, QUERY_ERROR_VOTABLE, "application/x-votable+xml")

    async def slow_asu(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.5)
        return await status(200, "<VOTABLE/>", "text/xml")

    handler = replay_handler("describe_2sxps", overrides={"columns": columns_rejected, "asu_catalog": slow_asu,
                                                           "asu_table": slow_asu})
    client, transport = scripted_client(handler)
    async with client:
        start = time.monotonic()
        with pytest.raises(vizier.VizierUpstreamError) as info:
            await vizier.describe("IX/58/2sxps", client=client)
        elapsed = time.monotonic() - start
        await asyncio.sleep(2.0)  # an orphan would complete (or retry) during this time
    assert elapsed < 1.0
    assert not info.value.transient  # the service answered: a rejected query, not an outage
    asu = [r for r in transport.requests if kind_of(r).startswith("asu")]
    assert len(asu) == 2 and not [r for r in transport.completed if kind_of(r).startswith("asu")]


async def test_catalogue_metadata_outage_degrades_to_assumptions_and_is_not_cached(no_backoff):
    async def down(request: httpx.Request) -> httpx.Response:
        return await status(503, "maintenance")

    handler = replay_handler("describe_2sxps", overrides={"asu_catalog": down, "regtap": down})
    client, transport = scripted_client(handler)
    async with client:
        desc = await vizier.describe_table("IX/58/2sxps", client=client)
        assert desc.registrable and not desc.complete
        assert desc.pos_error["kind"] == "radius90" and desc.catalog.bibcode is None
        assert any(a.startswith("catalogue: VizieR ASU catalogue metadata unavailable") for a in desc.assumptions)
        assert any(a.startswith("citation: IVOA RegTAP citation/waveband lookup unavailable") for a in desc.assumptions)
        before = len(transport.requests)
        await vizier.describe_table("IX/58/2sxps", client=client)
        assert len(transport.requests) > before  # an incomplete description is not cached


async def test_table_metadata_outage_is_a_problem_that_blocks_registration(no_backoff):
    async def down(request: httpx.Request) -> httpx.Response:
        return await status(503, "maintenance")

    client, _transport = scripted_client(replay_handler("describe_2sxps", overrides={"asu_table": down}))
    async with client:
        desc = await vizier.describe_table("IX/58/2sxps", client=client)
    assert not desc.registrable and any("table metadata unavailable" in p for p in desc.problems)
    assert desc.pos_error["kind"] == "radius90"  # the rest of the description is still there
    # An outage is transient: registration fails with an upstream error (HTTP 502), not 422.
    assert desc.upstream_problems and set(desc.upstream_problems) <= set(desc.problems)
    with pytest.raises(vizier.VizierUpstreamError, match="COOSYS epoch") as info:
        vizier.build_definition(desc)
    assert info.value.transient


# ---------------------------------------------------------------------------
# Issue: error classification was untested (retries, transient flags, ASU >= 400, parse errors)
# ---------------------------------------------------------------------------


async def test_timeout_is_a_transient_upstream_error_and_502():
    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    client, _transport = scripted_client(timeout)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="timed out") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert info.value.transient
    assert vizier._http_error(info.value).status_code == 502


async def test_5xx_is_retried_then_succeeds(no_backoff):
    calls = {"tables": 0}
    replay = replay_handler("describe_2sxps")

    async def flaky(request: httpx.Request) -> httpx.Response:
        if kind_of(request) == "tables":
            calls["tables"] += 1
            if calls["tables"] < vizier.HTTP_ATTEMPTS:
                return await status(503, "busy")
        return await replay(request)

    client, _transport = scripted_client(flaky)
    async with client:
        desc = await vizier.describe("IX/58/2sxps", client=client)
    assert calls["tables"] == vizier.HTTP_ATTEMPTS and desc.table_id == "IX/58/2sxps"


async def test_persistent_5xx_is_transient_after_all_attempts(no_backoff):
    async def down(request: httpx.Request) -> httpx.Response:
        return await status(502, "bad gateway")

    client, transport = scripted_client(down)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="unavailable") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert info.value.transient and len(transport.requests) == vizier.HTTP_ATTEMPTS


async def test_rejected_adql_is_not_transient():
    async def rejected(request: httpx.Request) -> httpx.Response:
        return await status(400, QUERY_ERROR_VOTABLE, "application/x-votable+xml")

    client, transport = scripted_client(rejected)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="rejected the query") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert not info.value.transient and len(transport.requests) == 1  # a 4xx is never retried
    # QUERY_STATUS=ERROR inside an HTTP 200 answer is a rejection too.

    async def error_200(request: httpx.Request) -> httpx.Response:
        return await status(200, QUERY_ERROR_VOTABLE, "application/x-votable+xml")

    client, _ = scripted_client(error_200)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="Incorrect ADQL") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert not info.value.transient


async def test_asu_http_error_and_malformed_xml_are_upstream_errors():
    async def asu_404(request: httpx.Request) -> httpx.Response:
        return await status(404, "no such service")

    async def asu_garbage(request: httpx.Request) -> httpx.Response:
        return await status(200, "<VOTABLE><RESOURCE>", "text/xml")

    for special, message in ((asu_404, "HTTP 404"), (asu_garbage, "malformed XML")):
        client, _ = scripted_client(replay_handler("describe_catalog_vii250", overrides={"asu_catalog": special}))
        async with client:
            with pytest.raises(vizier.VizierUpstreamError, match=message) as info:
                await vizier.describe("VII/250", client=client)  # a catalogue listing needs the ASU metadata
        assert not info.value.transient


def test_malformed_asu_answer_is_502_from_the_api(tmp_path):
    client = TestClient(_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith=vizier.VIZIER_ASU_URL).mock(
            return_value=httpx.Response(200, text="<VOTABLE", headers={"content-type": "text/xml"}))
        router.post(vizier.VIZIER_TAP_URL).mock(side_effect=replay_side_effect(_exchanges("describe_catalog_vii250")))
        response = client.get("/api/v1/vizier/catalog/VII/250")
    assert response.status_code == 502 and "malformed XML" in response.json()["detail"]


async def test_malformed_tap_json_is_an_upstream_error():
    async def garbage(request: httpx.Request) -> httpx.Response:
        return await status(200, "{not json", "application/json;charset=UTF-8")

    client, _ = scripted_client(garbage)
    async with client:
        with pytest.raises(vizier.VizierUpstreamError, match="could not be parsed") as info:
            await vizier.describe("IX/58/2sxps", client=client)
    assert not info.value.transient


async def test_registry_outage_during_search_is_a_warning(no_backoff):
    async def down(request: httpx.Request) -> httpx.Response:
        return await status(503, "maintenance")

    client, _ = scripted_client(replay_handler("search_vlass_registry", overrides={"regtap": down}))
    async with client:
        result = await vizier.search_catalogs("VLASS", include_registry=True, client=client)
    assert "J/ApJS/255/30/comp" in [t.table_id for t in result.tables]
    assert result.registry == []
    assert any(w.startswith("IVOA registry search failed") for w in result.warnings)


async def test_description_search_outage_keeps_the_asu_results(no_backoff):
    replay = replay_handler("search_gaia_dr3")
    # The recorded prefix lookup covered the ASU catalogues plus the description matches; it is a
    # superset of the rows asked for here (rows of catalogues not asked for are ignored).
    prefix = next(e for e in _exchanges("search_gaia_dr3") if "table_name+LIKE" in e.request_body)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.content or b""
        if b"description+LIKE" in body:
            return await status(503, "maintenance")
        if b"table_name+LIKE" in body:
            return httpx.Response(200, content=prefix.content, headers={"content-type": prefix.content_type})
        return await replay(request)

    client, _ = scripted_client(handler)
    async with client:
        result = await vizier.search_catalogs("Gaia DR3", client=client)
    assert result.tables[0].table_id == "I/355/gaiadr3"
    assert any(w.startswith("TAP_SCHEMA description search failed") for w in result.warnings)


def test_vizier_sql_error_dumps_become_one_warning():
    from xml.etree import ElementTree

    root = ElementTree.fromstring(
        '<VOTABLE><INFO name="Error" value="ERROR:  syntax error at or near &quot;COMMIT&quot;"/>'
        '<INFO name="Error" value="LINE 1:  SELECT catid FROM METAcat WHERE"/><INFO name="Error" value="^"/>'
        '<INFO name="Warning" value="List of 572 matching catalogues truncated to 500"/></VOTABLE>')
    warnings, total, truncated = vizier._asu_messages(root)
    assert warnings == ["VizieR reported an internal database error; the catalogue list may be incomplete."]
    assert total == 572 and truncated


# ---------------------------------------------------------------------------
# Issue: 'python vizier.py search ...' (the documented standalone form) failed
# ---------------------------------------------------------------------------


def test_standalone_invocation_accepts_top_level_subcommands(capsys):
    with replaying("describe_2sxps"):
        assert vizier.main(["describe", "IX/58/2sxps"]) == 0
    assert "Err90" in capsys.readouterr().out
    vizier.clear_describe_cache()
    with replaying("describe_2sxps"):
        assert vizier.main(["vizier", "describe", "IX/58/2sxps"]) == 0  # the prefixed form still works
    assert vizier.main([]) == 2


def test_standalone_script_parses_search():
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    help_run = subprocess.run([PYTHON, str(ROOT / "vizier.py"), "search", "--help"], capture_output=True, text=True,
                              env=env, timeout=120, check=False)
    assert help_run.returncode == 0 and "--wavelength" in help_run.stdout
    # A bad wavelength fails validation before any request: exit code 2 (usage), no traceback.
    bad = subprocess.run([PYTHON, str(ROOT / "vizier.py"), "search", "Gaia DR3", "--wavelength", "neutrino"],
                         capture_output=True, text=True, env=env, timeout=120, check=False)
    assert bad.returncode == 2 and "Unknown wavelength" in bad.stdout and "Traceback" not in bad.stderr


# ---------------------------------------------------------------------------
# Issue: lower-case VizieR ids gave 404
# ---------------------------------------------------------------------------


async def test_lower_case_identifier_is_resolved_to_the_canonical_table():
    desc = await replay_describe("ix/58/2sxps")
    assert isinstance(desc, vizier.TableDescription)
    assert desc.table_id == "IX/58/2sxps" and desc.pos_error["kind"] == "radius90"
    assert desc.catalog.bibcode == "2020ApJS..247...54E"


# ---------------------------------------------------------------------------
# Issue: no caching, no overall deadline, journal-level prefixes downloaded 22k tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ident", ["J/A+A", "J/A+A/707", "j/apj"])
async def test_journal_level_prefix_is_refused_without_a_query(ident):
    async def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client, transport = scripted_client(fail)
    async with client:
        with pytest.raises(vizier.VizierInputError, match="journal-level prefix"):
            await vizier.describe(ident, client=client)
    assert transport.requests == []


def test_journal_level_prefix_is_422_from_the_api(tmp_path):
    client = TestClient(_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True):
        response = client.get("/api/v1/vizier/catalog/J/A+A")
    assert response.status_code == 422 and "journal-level" in response.json()["detail"]


async def test_prefix_matching_too_many_tables_is_refused():
    rows = [[f'"IX/99/t{i}"', "t", 1] for i in range(vizier.MAX_TABLES_PER_CATALOG + 1)]
    payload = json.dumps({"metadata": [{"name": "table_name", "datatype": "char"},
                                       {"name": "description", "datatype": "char"},
                                       {"name": "nrows", "datatype": "long"}], "data": rows})

    async def many(request: httpx.Request) -> httpx.Response:
        assert f"TOP {vizier.MAX_TABLES_PER_CATALOG + 1} " in unquote_plus(request.content.decode())
        return await status(200, payload, "application/json;charset=UTF-8")

    client, _ = scripted_client(many)
    async with client:
        with pytest.raises(vizier.VizierInputError, match="more than 1000"):
            await vizier.describe("IX/99", client=client)


async def test_describe_results_are_cached_and_the_cache_can_be_cleared(monkeypatch):
    replay = replay_handler("describe_2sxps")
    client, transport = scripted_client(replay)
    async with client:
        first = await vizier.describe("IX/58/2sxps", client=client)
        count = len(transport.requests)
        second = await vizier.describe_table("IX/58/2sxps", client=client)
        assert len(transport.requests) == count  # served from the cache
        assert second.as_dict() == first.as_dict() and second is not first
        second.assumptions.append("mutated by a caller")
        third = await vizier.describe("IX/58/2sxps", client=client)
        assert "mutated by a caller" not in third.assumptions  # callers get copies
        vizier.clear_describe_cache()
        await vizier.describe("IX/58/2sxps", client=client)
        assert len(transport.requests) == 2 * count
        monkeypatch.setattr(vizier, "DESCRIBE_CACHE_TTL_SECONDS", 0.0)  # disabled
        await vizier.describe("IX/58/2sxps", client=client)
        assert len(transport.requests) == 3 * count


def test_route_deadline_answers_504(tmp_path, monkeypatch):
    monkeypatch.setattr(vizier, "ROUTE_DEADLINE_SECONDS", 0.3)

    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return await status(200)

    app: FastAPI = _app(tmp_path)
    app.state.client = httpx.AsyncClient(transport=ScriptedTransport(hang))
    client = TestClient(app)
    start = time.monotonic()
    response = client.get("/api/v1/vizier/catalog/IX/58/2sxps")
    assert response.status_code == 504 and "did not finish" in response.json()["detail"]
    assert time.monotonic() - start < 3.0


# ---------------------------------------------------------------------------
# Issue (critical): wrong position epochs (PPMXL/UCAC4/USNO-B1, VSX, Tycho-2)
# ---------------------------------------------------------------------------


async def test_ppmxl_epoch_is_the_coosys_j2000_not_the_mean_epoch_column():
    desc = await replay_describe("I/317/sample")
    assert desc.epoch == 2000.0 and "COOSYS" in (desc.epoch_source or "")
    assert desc.derived_columns == {}
    _, entry = vizier.build_definition(desc)
    assert entry["epoch"] == 2000.0 and "single_epoch_positions" not in entry["parameters"]


def test_explicit_ep_in_ra_description_beats_generic_epoch_columns():
    # UCAC4 / USNO-B1: 'Ep=J2000' in the RA description, a per-row EpRA / Epoch column beside it.
    ra = col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension (FK5, Equinox=J2000.0) (Ep=J2000)")
    de = col("DEJ2000", "deg", "pos.eq.dec;meta.main", "Declination (FK5, Equinox=J2000.0) (Ep=J2000)")
    cols = [ra, de, col("EpRA", "yr", "time.epoch", "Central epoch for mean RA"),
            col("EpDE", "yr", "time.epoch", "Central epoch for mean DE")]
    got = vizier.detect_epoch(cols, ra, {}, [], dec=de)
    assert got["epoch"] == 2000.0 and "Ep=J2000" in got["source"]


def test_event_times_are_never_position_epochs():
    ra = col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension (J2000)")
    for description in ("? Epoch of maximum or minimum (HJD)", "Time of periastron (JD)", "T0 of transit (BJD)"):
        cols = [ra, col("Epoch", "d", "time.epoch", description)]
        got = vizier.detect_epoch(cols, ra, {}, [])
        assert got["epoch"] is None and any("event time" in n for n in got["notes"]), description


async def test_vsx_epoch_of_maximum_light_is_ignored():
    desc = await replay_describe("B/vsx/vsx")
    assert desc.epoch is None and not desc.single_epoch_positions
    assert any("Epoch" in a and "event time" in a for a in desc.assumptions)


async def test_tycho2_observed_position_uses_offset_epoch_composite_id_and_described_errors():
    desc = await replay_describe("I/259/tyc2")
    assert (desc.ra_column, desc.dec_column) == ("RA(ICRS)", "DE(ICRS)")
    assert desc.epoch == "EpRA-1990_jyear" and desc.epoch_format == "jyear"
    assert desc.derived_columns["EpRA-1990_jyear"] == '"EpRA-1990" + 1990'
    assert desc.epoch_range == [1990.81, 1992.13]  # '[0.81,2.13] epoch-1990 of RAdeg'
    assert desc.id_column == "TYC1-TYC2-TYC3"
    assert desc.derived_columns["TYC1-TYC2-TYC3"] == """"TYC1" || '-' || "TYC2" || '-' || "TYC3\""""
    assert desc.pos_error == {"columns": ["e_RAdeg", "e_DEdeg"], "units": ["mas", "mas"], "kind": "sigma"}
    _, entry = vizier.build_definition(desc)
    columns = entry["parameters"]["columns"]
    assert """"TYC1" || '-' || "TYC2" || '-' || "TYC3" AS "TYC1-TYC2-TYC3\"""" in columns
    assert '"Num"' not in columns[:6]


def test_mean_epoch_of_another_position_column_is_rejected():
    ra = col("RA", "deg", "pos.eq.ra;meta.main", "Observed right ascension")
    de = col("DE", "deg", "pos.eq.dec;meta.main", "Observed declination")
    other = col("RAm", "deg", "pos.eq.ra", "Mean right ascension (J2000)")
    cols = [ra, de, other, col("EpRAm", "yr", "time.epoch", "mean epoch of RAm")]
    got = vizier.detect_epoch(cols, ra, {}, [], dec=de)
    assert got["epoch"] is None and any("not of the chosen position" in n for n in got["notes"])


def test_two_unrelated_position_epochs_are_ambiguous():
    ra = col("RA", "deg", "pos.eq.ra;meta.main", "Right ascension")
    cols = [ra, col("EpOpt", "yr", "time.epoch", "Epoch of the optical observation"),
            col("EpIR", "yr", "time.epoch", "Epoch of the infrared observation")]
    got = vizier.detect_epoch(cols, ra, {}, [])
    assert got["epoch"] is None and any("ambiguous" in n for n in got["notes"])


# ---------------------------------------------------------------------------
# Issue (critical): src.redshift columns in km/s mapped to the canonical redshift
# ---------------------------------------------------------------------------


async def test_recession_velocity_is_not_a_redshift():
    desc = await replay_describe("J/ApJ/956/51/table4")
    assert "redshift" not in desc.field_map
    assert any(a.startswith("redshift: cz") and "km/s" in a for a in desc.assumptions)
    _, entry = vizier.build_definition(desc)
    selected = {c.strip('"') for c in entry["parameters"]["columns"]}
    assert "cz" not in selected  # a guarded canonical column is only selected when chosen


def test_dimensionless_redshift_is_preferred_over_velocity():
    cols = [col("cz", "km/s", "src.redshift", "Recession velocity"), col("z", None, "src.redshift", "Redshift"),
            col("zv", "---", "src.redshift", "Redshift (dimensionless)")]
    mapping, notes = vizier.detect_field_map_notes(cols)
    assert mapping == {"redshift": "z"} and len(notes) == 1 and "cz" in notes[0]


# ---------------------------------------------------------------------------
# Issue: '3{sigma}' markup read as 1 sigma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [
    ("Error in RAdeg (3{sigma})", ("sigma", 3.0)),
    ("Estimated 2{sigma} X-ray positional uncertainty", ("sigma", 2.0)),
    ("[0.3/2.3] 2{sigma} X-ray positional uncertainty", ("sigma", 2.0)),
    ("positional error (3-{sigma})", ("sigma", 3.0)),
    ("1{sigma} error on position", ("sigma", 1.0)),
    ("Position error, 90% confidence", ("percent", 90.0)),
    ("Mean error on position", None),
])
def test_confidence_reads_vizier_markup(text, expected):
    assert vizier._confidence(text) == expected


async def test_cxogbs_three_sigma_errors_are_divided_by_three():
    desc = await replay_describe("J/ApJS/210/18/cxogbs")
    assert desc.pos_error == {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": ["arcsec", "arcsec"], "kind": "sigma",
                              "divisor": 3.0}
    sigma, _ = compute_positional_error({"e_RAJ2000": 1.5, "e_DEJ2000": 1.5}, desc.pos_error)
    assert sigma == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Issue: UCDs with capital letters found nothing
# ---------------------------------------------------------------------------


def test_ucd_casing_is_kept_or_canonicalised():
    assert vizier._validate_ucd("src.spType") == "src.spType"
    assert vizier._validate_ucd("src.sptype") == "src.spType"
    assert vizier._validate_ucd("pos.errorellipse;meta.main") == "pos.errorEllipse;meta.main"
    assert vizier._validate_ucd("phot.flux.density;em.ir.4-8um") == "phot.flux.density;em.IR.4-8um"
    # LIKE fragments stop before the first capital (TAPVizieR's LIKE is case-sensitive, no LOWER()).
    assert vizier._like_prefix("src.spType") == "src.sp"
    assert vizier._like_prefix("pos.errorEllipse") == "pos.error"
    assert vizier._like_prefix("em.IR") == "em."
    assert vizier._like_prefix("phot.mag") == "phot.mag"
    assert vizier.ucd_matches("src.sptype", "src.spType") and vizier.ucd_matches("SRC.SPTYPE", "src.spType")


# ---------------------------------------------------------------------------
# Issue: identifier heuristic picked counts ('Num' = number of positions)
# ---------------------------------------------------------------------------


def test_identifier_skips_counts_and_nullable_ids_and_falls_back_to_recno():
    base = [col("RA", "deg", "pos.eq.ra;meta.main"), col("DE", "deg", "pos.eq.dec;meta.main")]
    num = col("Num", None, "meta.id", "[2,36]? Number of positions used", "SMALLINT")
    hip = col("HIP", None, "meta.id", "[1,120404]? Hipparcos number", "INTEGER")
    recno = col("recno", None, "meta.record", "Record number assigned by the VizieR team", "INTEGER")
    name, derived, notes = vizier.detect_identifier(base + [num, hip, recno])
    assert name == "recno" and derived == {} and "recno" in notes[0]
    name, _, notes = vizier.detect_identifier(base + [num, col("Seq", None, "meta.id", "Sequential number", "INTEGER")])
    assert name == "Seq" and "assumed to be a unique identifier" in notes[0]
    name, _, notes = vizier.detect_identifier(base + [num, hip])
    assert name is None and "numbered" in notes[0]
    # A non-nullable count ('Number of detections', no '?') is not an identifier either.
    nobs = col("Nobs", None, "meta.id", "Number of detections", "SMALLINT")
    name, _, _ = vizier.detect_identifier(base + [nobs, col("Seq", None, "meta.id", "Sequential number", "INTEGER")])
    assert name == "Seq"
    single = vizier.detect_identifier(base + [col("Name", None, "meta.id.part;meta.main", "Name", "CHAR(10)")])
    assert single[0] == "Name"


# ---------------------------------------------------------------------------
# Issue: per-axis errors of another position (AllWISE e_RA_pm) were used
# ---------------------------------------------------------------------------


async def test_allwise_uses_the_error_ellipse_of_the_chosen_position():
    desc = await replay_describe("II/328/allwise")
    assert (desc.ra_column, desc.dec_column) == ("RAJ2000", "DEJ2000")
    assert desc.pos_error == {"columns": ["eeMaj", "eeMin", "eePA"], "units": ["arcsec", "arcsec", "deg"],
                              "kind": "ellipse"}
    assert any("e_RA_pm/e_DE_pm" in a and "not of the chosen position" in a for a in desc.assumptions)


def test_per_axis_pair_of_another_position_is_only_a_last_resort():
    ra = col("RAJ2000", "deg", "pos.eq.ra;meta.main", "Right ascension")
    de = col("DEJ2000", "deg", "pos.eq.dec;meta.main", "Declination")
    others = [col("RA_pm", "deg", "pos.eq.ra", "RA of the motion fit"), col("DE_pm", "deg", "pos.eq.dec", "Dec fit"),
              col("e_RA_pm", "arcsec", "stat.error;pos.eq.ra", "Error on RA_pm"),
              col("e_DE_pm", "arcsec", "stat.error;pos.eq.dec", "Error on DE_pm")]
    spec, _, notes = vizier.detect_pos_error([ra, de, *others], ra, de)
    assert spec["columns"] == ["e_RA_pm", "e_DE_pm"]
    assert any("used for lack of any other positional error" in n for n in notes)


# ---------------------------------------------------------------------------
# Issue: CSC ellipses registered with the 2-D 95% conversion
# ---------------------------------------------------------------------------


def test_per_axis_convention_from_bibcode_or_title():
    spec = {"columns": ["a", "b", "pa"], "units": ["arcsec", "arcsec", "deg"], "kind": "ellipse95"}
    notes = ["pos_error: '95% confidence' ellipse a read as the 2-D 95% contour ..."]
    by_bibcode = vizier.CatalogInfo(catalog_id="J/ApJS/189/37", bibcode="2010ApJS..189...37E")
    out, new_notes = vizier.apply_error_conventions(spec, notes, by_bibcode, "J/ApJS/189/37/csc")
    assert out["kind"] == "ellipse95axis" and not any("2-D" in n for n in new_notes)
    by_title = vizier.CatalogInfo(catalog_id="IX/57", title="The Chandra Source Catalog, Release 2.0 (Evans+, 2020)")
    assert vizier.apply_error_conventions(spec, notes, by_title, "IX/57/csc2master")[0]["kind"] == "ellipse95axis"
    other = vizier.CatalogInfo(catalog_id="IX/99", title="Some other X-ray catalogue")
    assert vizier.apply_error_conventions(spec, notes, other, "IX/99/t")[0]["kind"] == "ellipse95"
    sigma, _ = compute_positional_error({"a": 1.96, "b": 1.96, "pa": 0}, {**spec, "kind": "ellipse95axis"})
    assert sigma == pytest.approx(1.96 / K95_1D)


# ---------------------------------------------------------------------------
# Issue: citations lacked author and bibcode although RegTAP has them
# ---------------------------------------------------------------------------


async def test_citation_falls_back_to_the_ivoa_registry():
    desc = await replay_describe("VIII/65/nvss")
    assert desc.catalog.bibcode == "1998AJ....115.1693C"
    assert desc.citation.startswith("Condon et al. 1998, AJ 115, 1693 (1998AJ....115.1693C); VizieR VIII/65")
    assert any("from the IVOA registry" in a for a in desc.assumptions)
    # NVSS e_RAJ2000 is in seconds of time.
    assert desc.pos_error["units"] == ["s_ra", "arcsec"]


def test_first_author_formatting():
    assert vizier._first_author("Monet D.G.; Levine S.E.; Casian B.; et al.") == "Monet et al."
    assert vizier._first_author("Condon J.J.") == "Condon"
    assert vizier._first_author("") is None
    assert "ivo://cds.vizier/viii/65" in vizier._regtap_citation_adql("VIII/65")


# ---------------------------------------------------------------------------
# Issue: RA in seconds of time accepted as degrees
# ---------------------------------------------------------------------------


async def test_ra_seconds_column_is_skipped_for_the_degree_position():
    desc = await replay_describe("J/A+A/657/A4/stars")
    assert (desc.ra_column, desc.dec_column) == ("RAJ2000", "DEJ2000")
    assert any("RAS" in a and "not degrees" in a for a in desc.assumptions)


def test_table_with_only_sexagesimal_parts_is_not_registrable():
    cols = [col("RAh", "h", "pos.eq.ra;meta.main", "Right ascension (hours)", "SMALLINT"),
            col("RAs", "s", "pos.eq.ra;meta.main", "Right ascension (seconds)"),
            col("DEd", "deg", "pos.eq.dec;meta.main", "Declination (degrees)", "SMALLINT"),
            col("DEs", "arcsec", "pos.eq.dec;meta.main", "Declination (arcsec)")]
    desc = vizier.analyse_table("X/2/t", vizier.CatalogInfo(catalog_id="X/2"), {"nrows": 1}, cols)
    assert desc.ra_column is None and not desc.registrable
    assert "RAs (unit 's', not degrees)" in desc.problems[0] and "sexagesimal part" in desc.problems[0]


# ---------------------------------------------------------------------------
# Issue: radio catalogues registered with purely statistical (500x too small) errors
# ---------------------------------------------------------------------------


def test_core_table_systematic_is_inherited_and_unknown_radio_surveys_are_flagged():
    spec = {"columns": ["e_RAJ2000", "e_DEJ2000"], "units": ["deg", "deg"], "kind": "sigma"}
    info = vizier.CatalogInfo(catalog_id="J/ApJ/914/42", title="VLASS epoch 1 catalogue (Bruzewski+, 2021)")
    out, notes = vizier.apply_error_conventions(spec, [], info, "J/ApJ/914/42/table5")
    assert out["systematic_arcsec"] == DEFAULT_CATALOGS["vlass"]["pos_error"]["systematic_arcsec"]
    assert "core 'vlass' entry for the same VizieR table" in notes[0]
    lotss = vizier.CatalogInfo(catalog_id="J/A+A/999/A1", title="LoTSS deep fields (Someone+, 2030)")
    assert vizier.apply_error_conventions(spec, [], lotss, "J/A+A/999/A1/cat")[0]["systematic_arcsec"] == 0.2
    unknown = vizier.CatalogInfo(catalog_id="J/X/1/2", wavelengths=["Radio"], title="A radio catalogue")
    cols = [col("RAJ2000", "deg", "pos.eq.ra;meta.main"), col("DEJ2000", "deg", "pos.eq.dec;meta.main"),
            col("e_RAJ2000", "arcsec", "stat.error;pos.eq.ra", "1-sigma error"),
            col("e_DEJ2000", "arcsec", "stat.error;pos.eq.dec", "1-sigma error")]
    desc = vizier.analyse_table("J/X/1/2/t", unknown, {"nrows": 1}, cols)
    assert "systematic_arcsec" not in desc.pos_error
    assert any("statistical only" in a and "systematic" in a for a in desc.assumptions)
