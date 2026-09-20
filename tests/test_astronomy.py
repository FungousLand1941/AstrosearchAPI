import csv
from xml.sax.saxutils import escape

import httpx
import pytest

from astronomy import (
    ArchiveError,
    ExoplanetArchive,
    TapClient,
    entity_id,
    literal,
    match_hosts,
    object_summary,
    parse_tap,
    planet_summary,
    system_summary,
)
from astronomy_pipeline import commit_snapshot, load_json, mantis_command, map_rows, publish, relationships, state_lock


def planet(name="Example b", host="Example", **extra):
    return {"pl_name": name, "hostname": host, "ra": 10.0, "dec": 20.0,
            "pl_orbper": 5.0, "pl_orbpererr1": 0.1, "pl_orbpererr2": -0.2,
            "pl_orbperlim": 0, "pl_orbper_reflink": "https://example.org/paper", **extra}


def no_match(host="Example"):
    return {host: {"status": "unmatched", "method": "exact_identifier", "candidates": []}}


def votable(fields, rows, status="OK"):
    field_xml = "".join(f'<FIELD name="{name}" datatype="{kind}" arraysize="*"/>' if kind == "char"
                        else f'<FIELD name="{name}" datatype="{kind}"/>' for name, kind in fields)
    row_xml = "".join("<TR>" + "".join(f"<TD>{escape(str(v)) if v is not None else ''}</TD>" for v in row) + "</TR>" for row in rows)
    return (f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">'
            f'<INFO name="QUERY_STATUS" value="OK"/><TABLE>{field_xml}<DATA><TABLEDATA>{row_xml}</TABLEDATA></DATA>'
            f'</TABLE><INFO name="QUERY_STATUS" value="{status}"/></RESOURCE></VOTABLE>').encode()


def test_parse_nulls_precision_and_identifiers():
    data = votable([("oid", "long"), ("value", "double"), ("name", "char")], [[99999999999999999, None, "A & B"]])
    row = parse_tap(data)[0]
    assert row == {"oid": 99999999999999999, "value": None, "name": "A & B"}


@pytest.mark.parametrize("status", ["ERROR", "OVERFLOW"])
def test_rejects_http_200_tap_errors(status):
    with pytest.raises(ArchiveError):
        parse_tap(votable([("n", "int")], [[1]], status))


def test_rejects_non_votable():
    with pytest.raises(ArchiveError):
        parse_tap(b"<html>maintenance</html>")


def test_summary_preserves_limits_errors_zero_and_mass_type():
    p = planet(pl_rade=0, pl_radelim=1, pl_bmassprov="Msini", pl_controv_flag=1)
    result = planet_summary(p)
    assert "upper limit 0 Earth radii" in result["text"]
    assert "upper error 0.1, lower error -0.2" in result["text"]
    assert "Msini" in result["text"]
    assert any("controversial" in w for w in result["warnings"])
    assert result["measurements"][0]["reference"] == "https://example.org/paper"
    assert not any(m["field"] == "sy_dist" for m in result["measurements"])


def test_distinct_star_and_planet_ids():
    assert entity_id("host", "A") != entity_id("host", "a")
    assert entity_id("host", "Example") != entity_id("planet", "Example")
    assert literal("O'Brien") == "'O''Brien'"
    with pytest.raises(ValueError):
        literal("hello\n")


@pytest.mark.asyncio
async def test_complete_snapshot_and_name_escaping():
    queries = []
    class FakeTap:
        async def query(self, endpoint, query, **kwargs):
            queries.append(query)
            if "TAP_SCHEMA" in query:
                return [{"column_name": k} for k in planet()]
            if "count(*)" in query:
                return [{"n": 2}]
            assert "TOP" not in query
            assert kwargs["maxrec"] == 3
            return [planet(), planet("Example c")]
    rows = await ExoplanetArchive(FakeTap()).fetch(name="O'Brien")
    assert len(rows) == 2
    assert any("hostname='O''Brien'" in q for q in queries)
    assert "ORDER BY pl_name" in queries[-1]


@pytest.mark.asyncio
async def test_count_mismatch_rejected():
    class FakeTap:
        async def query(self, endpoint, query, **kwargs):
            if "TAP_SCHEMA" in query:
                return [{"column_name": k} for k in planet()]
            return [{"n": 2}] if "count(*)" in query else [planet()]
    with pytest.raises(ArchiveError, match="changed or truncated"):
        await ExoplanetArchive(FakeTap()).fetch()


@pytest.mark.asyncio
async def test_identity_ambiguity_and_multiple_planets():
    class FakeTap:
        async def query(self, endpoint, query, **kwargs):
            return [{"id": "Example", "oid": 1, "main_id": "Star A"},
                    {"id": "Gaia DR3 123", "oid": 2, "main_id": "Star B"}]
    rows = [planet(gaia_dr3_id="Gaia DR3 123"), planet("Example c"), planet("Other b", "Other")]
    matches = await match_hosts(FakeTap(), rows)
    assert matches["Example"]["status"] == "ambiguous"
    assert len(matches["Example"]["candidates"]) == 2
    assert matches["Other"]["status"] == "unmatched"
    summary = system_summary(rows, matches)
    assert summary["systems"][0]["planet_count"] == 2
    edges = relationships(summary)
    assert sum(e["relation"] == "orbits_host" for e in edges) == 3
    assert sum(e["relation"] == "ambiguous_candidate" for e in edges) == 2
    assert len({r["catalog_id"] for r in map_rows(summary)}) == 7


@pytest.mark.asyncio
async def test_consistent_aliases_count_as_one_match():
    class FakeTap:
        async def query(self, endpoint, query, **kwargs):
            return [{"id": name, "oid": 1, "main_id": "Star A"} for name in ["Example", "Gaia DR3 123"]]
    result = await match_hosts(FakeTap(), [planet(gaia_dr3_id="Gaia DR3 123")])
    assert result["Example"]["status"] == "matched"
    assert len(result["Example"]["candidates"][0]["matched_aliases"]) == 2


@pytest.mark.asyncio
async def test_transient_http_retry(monkeypatch):
    calls = []
    async def sleep(_):
        pass
    monkeypatch.setattr("astronomy.asyncio.sleep", sleep)
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, content=votable([("n", "int")], [[2]]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await TapClient(client, delay=0).query("https://example.org/tap", "select count(*)") == [{"n": 2}]
    assert len(calls) == 2


def test_snapshots_idempotent_updates_deletes_and_csv(tmp_path):
    first = commit_snapshot(tmp_path, [planet(), planet("Example c")], no_match(), [])
    assert first["changed"]
    assert not commit_snapshot(tmp_path, [planet("Example c"), planet()], no_match(), [{"time": "different"}])["changed"]
    second = commit_snapshot(tmp_path, [planet(pl_orbper=6)], no_match(), [])
    assert second["changes"]["updated"] == ["Example b"]
    assert second["changes"]["removed"] == ["Example c"]
    assert (tmp_path / "snapshots" / first["fingerprint"] / "catalog.json").exists()
    with (tmp_path / "snapshots" / second["fingerprint"] / "mantis.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "id" not in rows[0]  # Mantis reserves its own internal point identifier.
    assert len({r["catalog_id"] for r in rows}) == 2
    assert {r["kind"] for r in rows} == {"planet", "host_system"}
    with pytest.raises(ArchiveError):
        commit_snapshot(tmp_path, [], {}, [])
    assert load_json(tmp_path / "latest.json")["fingerprint"] == second["fingerprint"]


def test_failed_snapshot_preserves_latest(tmp_path, monkeypatch):
    first = commit_snapshot(tmp_path, [planet()], no_match(), [])
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("astronomy_pipeline.map_rows", fail)
    with pytest.raises(OSError):
        commit_snapshot(tmp_path, [planet(pl_orbper=8)], no_match(), [])
    assert load_json(tmp_path / "latest.json")["fingerprint"] == first["fingerprint"]


def test_state_lock(tmp_path):
    with state_lock(tmp_path), pytest.raises(RuntimeError, match="Another"), state_lock(tmp_path):
        pass
    assert not (tmp_path / ".sync.lock").exists()


def test_private_mantis_command(tmp_path):
    argv = mantis_command(tmp_path / "with spaces.csv", "abc")
    assert "--private" in argv and "--no-activate" in argv
    assert "summary" in argv and "ra_deg,dec_deg,planet_count,period_days,radius_earth,mass_earth,distance_pc" in argv
    with pytest.raises(ValueError):
        mantis_command(tmp_path / "data.csv", "abc", space_id="not-a-uuid")


def test_publication_does_not_repeat_uncertain_write(tmp_path, monkeypatch):
    import subprocess
    commit_snapshot(tmp_path, [planet()], no_match(), [])
    calls = []
    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0 if args[0] == "use" else 1, stdout="")
    monkeypatch.setattr("astronomy_pipeline.run_mantis", invoke)
    with pytest.raises(RuntimeError, match="uncertain"):
        publish(tmp_path)
    with pytest.raises(RuntimeError, match="uncertain"):
        publish(tmp_path)
    assert len(calls) == 2


def test_publication_reuses_saved_space_and_marks_only_submitted(tmp_path, monkeypatch):
    import json
    import subprocess
    sid = "6aa1d8cf-9d4c-497b-8bed-a6337b11f4a2"
    calls = []
    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="Creation started\n" + json.dumps({"space_id": sid, "map_id": "test-map"}))
    monkeypatch.setattr("astronomy_pipeline.run_mantis", invoke)
    commit_snapshot(tmp_path, [planet()], no_match(), [])
    assert publish(tmp_path)["status"] == "submitted"
    assert publish(tmp_path)["status"] == "submitted"
    assert len(calls) == 2
    checkpoint = load_json(tmp_path / "publications.json")
    next(iter(checkpoint.values()))["status"] = "complete"
    (tmp_path / "publications.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    assert publish(tmp_path)["status"] == "complete"
    assert len(calls) == 2
    commit_snapshot(tmp_path, [planet(pl_orbper=9)], no_match(), [])
    assert publish(tmp_path)["status"] == "submitted"
    assert "--space-id" in calls[-1] and sid in calls[-1]
    assert "--private" not in calls[-1]


def test_publication_can_use_valid_rest_when_mcp_is_unavailable(tmp_path, monkeypatch):
    import json
    import subprocess
    calls = []
    def invoke(args, **kwargs):
        calls.append(args)
        body = json.dumps({"space_id": "6aa1d8cf-9d4c-497b-8bed-a6337b11f4a2", "map_id": "test-map"})
        return subprocess.CompletedProcess(args, 1 if args[0] == "use" else 0, stdout=body)
    monkeypatch.setattr("astronomy_pipeline.run_mantis", invoke)
    commit_snapshot(tmp_path, [planet()], no_match(), [])
    assert publish(tmp_path)["status"] == "submitted"
    assert [c[0] for c in calls] == ["use", "spaces", "create"]


def test_object_summary_preserves_failures_and_candidates():
    result = object_summary({"target": {"ra": 1, "dec": 2}, "counterparts": {"optical": [{"source": "test"}]},
                             "catalogs_queried": 2, "failures": [{"catalog": "simbad"}]})
    assert result["status"] == "partial"
    assert "1 candidate counterparts" in result["text"]
    assert result["failures"][0]["catalog"] == "simbad"
