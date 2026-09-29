"""Regression tests for the provenance review (round 3).

Covers: replays of catalogs served by a declared fallback archive (IRSA Gator for
2MASS/AllWISE when IRSA TAP is down) in both directions; catalogs that failed in both
runs are "not compared", never "reproduced"; live release descriptions are attached only
when read at query time (a stored record keeps the label, marked unverified); requests of
empty catalogs come from the ADQL the provider recorded; CPU work off the event loop and
the fast hashing paths; CLI registry path and definition-checked release/citation labels;
502 / exit 3 for a manifest of a total outage; the search fields equal api.SearchRequest;
TESS/GALEX/unWISE citations and partial citation requests; official NEOWISE, HEASARC and
NED acknowledgements; a stricter BibTeX validator; the manifest schema in ``cite --from``.

Archive traffic is replayed from recorded fixtures: tests/fixtures/3c273 (IRSA TAP),
tests/fixtures/fallback/3c273 (IRSA Gator, recorded while IRSA TAP was forced to 503) and
tests/fixtures/provenance/heasarc_tables (HEASARC TAP_SCHEMA.tables).
"""

from __future__ import annotations

import asyncio
import copy
import html
import io
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import TARGETS, Exchange, load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_provenance import _app, _crossmatch_3c273, _parser, _source, synthetic_record
from test_provenance_regressions import _with_tap_schema

import provenance as P
from models import DEFAULT_CATALOGS, IRSA_GATOR, IRSA_TAP, CatalogRegistry

RA_3C273, DEC_3C273 = TARGETS["3c273"]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TWOMASS_3C273 = "12290669+0203085"  # 2MASS PSC counterpart of 3C 273 (Ks = 9.98)


@pytest.fixture(autouse=True)
def _fresh_release_cache():
    P._RELEASE_CACHE.clear()
    yield
    P._RELEASE_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _irsa(*, tap_down: bool = False, gator_down: bool = False):
    """Recorded 3C 273 archives (IRSA TAP and IRSA Gator both recorded), with IRSA TAP
    and/or Gator forced to answer HTTP 503. Returns (handler, call counts)."""
    strict = replay_side_effect(load_exchanges("3c273") + load_exchanges("fallback/3c273"))
    calls = {"tap": 0, "gator": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(IRSA_TAP):
            calls["tap"] += 1
            if tap_down:
                return httpx.Response(503, text="IRSA TAP outage")
        elif url.startswith(IRSA_GATOR):
            calls["gator"] += 1
            if gator_down:
                return httpx.Response(503, text="IRSA Gator outage")
        return strict(request)

    return handler, calls


async def _search(handler) -> tuple[Any, Any]:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=handler)
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
    return record, service


async def _replay(manifest: Any, handler, **kwargs: Any) -> P.ReplayResult:
    kwargs.setdefault("live_release_lookup", False)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=handler)
        async with offline_client() as client:
            return await P.replay_manifest(manifest, make_service(client), **kwargs)


def _no_registry_blame(result: P.ReplayResult) -> None:
    assert not any("the registry definition or the request-building software changed" in line
                   for line in result.explanation), result.explanation


# ---------------------------------------------------------------------------
# Declared fallback archives (major)
# ---------------------------------------------------------------------------


async def test_catalogs_served_by_the_fallback_are_replayed_through_the_same_archive() -> None:
    handler, _ = _irsa(tap_down=True)
    record, service = await _search(handler)
    manifest = P.build_manifest(record, registry=service.registry)
    for name in ("twomass_psc", "allwise"):
        req = manifest.catalogs[name]
        assert req.provider == "irsa_gator" and req.status == "success"
        assert req.fallback["provider"] == "irsa_gator" and "HTTP 503" in req.fallback["reason"]
        assert req.request_verified is True  # the Gator request as sent
        # The primary request that failed is recorded too.
        assert req.fallback["primary_request"]["endpoint"] == IRSA_TAP
        assert req.fallback["primary_request"]["query"].startswith("SELECT TOP 201")
        assert P.served_by(req) == "fallback irsa_gator"
    assert [s["source_id"] for s in manifest.science["catalogs"]["twomass_psc"]["sources"]] == [TWOMASS_3C273]

    # Replayed on a normal day (IRSA TAP up): the fallback archive is asked first again.
    normal, calls = _irsa()
    result = await _replay(json.loads(manifest.to_json()), normal)
    assert result.identical and result.equivalent_within_tolerance, result.explanation
    assert calls == {"tap": 0, "gator": 2}
    assert result.diff["requests"] == {} and result.diff["served_by"] == {}
    assert result.explanation[0] == "Content hash identical: every archive returned the same science."
    again = result.new_manifest.catalogs["twomass_psc"]
    assert again.provider == "irsa_gator" and "asked first" in again.fallback["reason"]
    assert result.new_manifest.query_hash == manifest.query_hash


async def test_a_primary_outage_during_replay_is_attributed_to_the_fallback_archive() -> None:
    normal, _ = _irsa()
    record, service = await _search(normal)
    manifest = P.build_manifest(record, registry=service.registry)
    assert P.served_by(manifest.catalogs["twomass_psc"]) == "primary"
    down, calls = _irsa(tap_down=True)
    result = await _replay(manifest, down)
    assert calls["tap"] >= 2 and calls["gator"] == 2  # (the provider retries a 503)
    assert not result.identical and result.content_hash_equal is False  # other columns, other number format ...
    assert result.equivalent_within_tolerance, result.explanation  # ... but the same science
    assert result.diff["served_by"] == {"allwise": {"old": "primary", "new": "fallback irsa_gator"},
                                        "twomass_psc": {"old": "primary", "new": "fallback irsa_gator"}}
    tm = result.diff["catalogs"]["twomass_psc"]
    assert set(tm) == {"columns"}  # no added/removed/moved/changed rows
    assert "match_dist" in tm["columns"]["only_old"] and {"dist", "angle", "j_h"} <= set(tm["columns"]["only_new"])
    assert result.diff["totals"]["changed"] == 0
    assert result.diff["requests"]["twomass_psc"]["served_by"] == {"old": "primary", "new": "fallback irsa_gator"}
    text = "\n".join(result.explanation)
    assert "twomass_psc: the original run was served by the primary archive (tap https://irsa.ipac.caltech.edu/TAP/sync)" \
        in text
    assert "the replay by the declared fallback archive (irsa_gator" in text and "HTTP 503" in text
    assert "science is equivalent" in text
    _no_registry_blame(result)


async def test_a_fallback_that_fails_on_replay_hands_over_to_the_primary_and_is_explained() -> None:
    down, _ = _irsa(tap_down=True)
    record, service = await _search(down)
    manifest = P.build_manifest(record, registry=service.registry)
    gator_down, calls = _irsa(gator_down=True)
    result = await _replay(manifest, gator_down)
    assert calls["gator"] >= 2 and calls["tap"] == 2  # Gator first (as in the original run), then the primary
    assert result.diff["served_by"]["twomass_psc"] == {"old": "fallback irsa_gator", "new": "primary"}
    assert result.new_manifest.catalogs["twomass_psc"].fallback is None
    assert result.new_manifest.catalogs["twomass_psc"].provider == "tap"
    assert result.equivalent_within_tolerance, result.explanation
    text = "\n".join(result.explanation)
    assert "twomass_psc: the declared fallback archive that served the original run failed in the replay" in text
    _no_registry_blame(result)


async def test_a_changed_definition_is_still_reported_when_another_archive_answered() -> None:
    """The request the original archive would get today is rebuilt from today's registry:
    a real definition change is not hidden behind the change of archive service."""
    normal, _ = _irsa()
    record, service = await _search(normal)
    manifest = P.build_manifest(record, registry=service.registry)
    registry = CatalogRegistry()
    tm = registry._catalogs["twomass_psc"]
    assert tm.parameters.get("format") != "csv"
    # A TAP-only change (the Gator request does not use it): FORMAT=csv instead of json.
    registry._catalogs["twomass_psc"] = replace(tm, parameters={**tm.parameters, "format": "csv"})
    down, _ = _irsa(tap_down=True)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=down)
        async with offline_client() as client:
            result = await P.replay_manifest(manifest, make_service(client, registry=registry),
                                             live_release_lookup=False)
    assert result.diff["served_by"]["twomass_psc"] == {"old": "primary", "new": "fallback irsa_gator"}
    text = "\n".join(result.explanation)
    assert ("twomass_psc: in addition, the request the archive that answered the original run would be sent now "
            "differs from the manifest's") in text
    assert "allwise: in addition" not in text  # unchanged definition: only the service changed


def test_float32_noise_is_tolerated_only_between_different_archive_services() -> None:
    def science(j_m: float, extra: dict[str, Any]) -> dict[str, Any]:
        rec = synthetic_record()
        rec["catalog_results"] = {"twomass_psc": {"status": "success", "row_count": 1, "sources": [
            _source("twomass_psc", TWOMASS_3C273, 187.2779, 2.0524, {"j_m": j_m, "err_maj": 0.17, **extra})]}}
        rec["failures"], rec["provenance"]["matches"], rec["crossmatch_groups"] = [], [], []
        return P.science_payload(rec)

    tap = science(11.765999794, {"match_dist": 0.066363})  # VOTable float32
    gator = science(11.766, {"dist": 0.083441, "angle": 262.56})  # decimal text
    same_service = P.diff_science(tap, gator)
    assert not same_service["equivalent"]  # same service: every difference counts
    across = P.diff_science(tap, gator, transport_changed=["twomass_psc"])
    assert across["equivalent"] and set(across["catalogs"]["twomass_psc"]) == {"columns"}
    real = P.diff_science(tap, science(11.866, {"dist": 0.083441}), transport_changed=["twomass_psc"])
    assert not real["equivalent"]
    assert real["catalogs"]["twomass_psc"]["changed"][0]["fields"]["data.j_m"]["new"] == 11.866


# ---------------------------------------------------------------------------
# Catalogs that failed in both runs (major)
# ---------------------------------------------------------------------------


def _both_fail_exchanges() -> list[Exchange]:
    error_body = (FIXTURES / "errors" / "heasarc_bad_column.0.body").read_bytes()
    return [replace(e, content=error_body, content_type="text/xml") if e.catalog in ("gaia_dr3", "chandra") else e
            for e in load_exchanges("3c273")]


def test_catalogs_failing_in_both_runs_are_not_reported_as_reproduced(tmp_path: Path, capsys) -> None:
    exchanges = _both_fail_exchanges()
    record, service = asyncio.run(_crossmatch_3c273(exchanges))
    manifest = P.build_manifest(record, registry=service.registry)
    assert manifest.catalogs["gaia_dr3"].status == "failed" and manifest.catalogs["chandra"].status == "failed"
    result = asyncio.run(_replay(manifest, replay_side_effect(exchanges)))
    assert result.content_hash_equal is True
    assert result.identical is False and result.equivalent_within_tolerance is False
    assert [c["catalog"] for c in result.not_compared] == ["chandra", "gaia_dr3"]
    assert result.not_compared[0]["old_error_type"] == result.not_compared[0]["new_error_type"] == "CatalogQueryError"
    assert result.diff["not_compared"] == ["chandra", "gaia_dr3"]
    assert result.explanation[0].startswith("Content hash identical, but 2 catalog(s) failed in both runs")
    assert "gaia_dr3 (original CatalogQueryError, replay CatalogQueryError)" in result.explanation[0]
    assert not any("every archive returned the same science" in line for line in result.explanation)
    body = result.as_dict()
    assert body["identical"] is False and body["content_hash_equal"] is True and len(body["not_compared"]) == 2
    # CLI: a distinct exit code (4), not "identical" (0).
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    args = _parser().parse_args(["replay", str(path)])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_with_tap_schema(exchanges))
        assert args.handler(args) == 4
    out = capsys.readouterr().out
    assert "Identical: no" in out and "Not compared (failed in both runs): chandra, gaia_dr3" in out


# ---------------------------------------------------------------------------
# Data release of stored records (major)
# ---------------------------------------------------------------------------


def _aged(record: Any, stamp: str) -> dict[str, Any]:
    rec = record.as_dict()
    for res in rec["catalog_results"].values():
        for row in [*(res.get("sources") or []), *(res.get("pad_sources") or [])]:
            row["provenance"]["retrieved_at"] = stamp
    return rec


def _xmm_with_revised_flux() -> list[Exchange]:
    """3C 273 fixtures with the 5XMM EPIC flux of 5XMM J122906.6+020308 revised by 50%."""
    from astropy.io.votable import parse

    out = []
    for ex in load_exchanges("3c273"):
        if ex.catalog == "xmm":
            votable = parse(io.BytesIO(ex.content))
            table = votable.get_first_table()
            assert [str(n) for n in table.array["name"]] == ["5XMM J122906.6+020308"]  # the 3C 273 source
            table.array["ep_flux"][0] *= 1.5
            buf = io.BytesIO()
            votable.to_xml(buf, tabledata_format="binary")  # as served (TABLEDATA would apply FIELD precision)
            ex = replace(ex, content=buf.getvalue())
        out.append(ex)
    return out


async def test_a_stored_record_does_not_get_todays_release_label() -> None:
    record, service = await _crossmatch_3c273()
    exchanges = load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_with_tap_schema(exchanges))
        async with offline_client() as client:
            releases = await P.live_releases(client, service.registry, list(service.registry.enabled_catalogs()))
    assert releases["xmm"]["looked_up_at"]
    old = _aged(record, "2026-01-15T10:00:00+00:00")  # saved in the 4XMM era
    manifest = P.build_manifest(old, registry=service.registry, releases=releases)
    xmm = manifest.catalogs["xmm"]
    assert xmm.release == P.CATALOG_RELEASES["xmm"]["release"]  # not today's "5XMM-DR15"
    assert xmm.release_source.startswith(P.UNVERIFIED_RELEASE_PREFIX)
    assert "5XMM-DR15" in xmm.release_source and "2026-01-15T10:00:00+00:00" in xmm.release_source
    assert manifest.query_time == "2026-01-15T10:00:00+00:00"
    assert manifest.query_time_source == "earliest row retrieved_at of the record"
    assert manifest.created_at > manifest.query_time
    parsed = P.ProvenanceManifest.from_json(manifest.to_json())
    assert (parsed.query_time, parsed.query_time_source) == (manifest.query_time, manifest.query_time_source)
    assert parsed.catalogs["xmm"].release_source == xmm.release_source
    assert manifest.catalogs["gaia_dr3"].release_source == "label"
    # The same record manifested at query time gets the live description.
    fresh = P.build_manifest(record, registry=service.registry, releases=releases)
    assert fresh.catalogs["xmm"].release.startswith("XMM-Newton Serendipitous Source Catalog: 5XMM-DR15")
    assert fresh.catalogs["xmm"].release_source.startswith("live:")
    # A search run for the manifest: the time is known even when no catalog returned a row.
    empty = synthetic_record()
    empty["catalog_results"] = {"xmm": {"sources": [], "status": "empty", "row_count": 0}}
    empty["provenance"]["catalogs_planned"], empty["failures"] = ["xmm"], []
    unknown = P.build_manifest(empty, registry=service.registry, releases=releases)
    assert unknown.query_time is None and "unknown time" in unknown.catalogs["xmm"].release_source
    now = P.build_manifest(empty, registry=service.registry, releases=releases, query_time=datetime.now(UTC))
    assert now.catalogs["xmm"].release_source.startswith("live:")
    assert now.query_time_source == "the search run for this manifest"

    # Replay: the difference is not blamed on "a continuously updated database" alone.
    changed = _xmm_with_revised_flux()
    result = await _replay(manifest, _with_tap_schema(changed), live_release_lookup=True)
    assert result.diff["releases"] == {}  # an unverified label is never compared as a release change
    assert set(result.diff["catalogs"]["xmm"]["changed"][0]["fields"]) == {"data.ep_flux"}
    text = "\n".join(result.explanation)
    assert "xmm: the data release of the original query is not known" in text
    assert ("xmm: 1 changed -- the release the original query saw is unknown and this table changes release in "
            "place") in text


# ---------------------------------------------------------------------------
# Requests of empty catalogs come from the recorded ADQL (minor)
# ---------------------------------------------------------------------------


async def test_empty_catalog_requests_come_from_the_recorded_adql_and_expose_registry_changes() -> None:
    record, service = await _crossmatch_3c273()
    rec = record.as_dict()
    sent = rec["catalog_results"]["exoplanet_archive"]["query"]
    assert rec["catalog_results"]["exoplanet_archive"]["status"] == "empty" and "gaia_dr3_id, " in sent
    older = sent.replace("gaia_dr3_id, ", "")  # as an earlier registry sent it
    rec["catalog_results"]["exoplanet_archive"]["query"] = older
    manifest = P.build_manifest(rec, registry=service.registry)
    req = manifest.catalogs["exoplanet_archive"]
    assert req.request_source == "recorded (result metadata)"
    assert req.request_verified is False  # flagged, not silently stored next to a reconstruction
    assert req.query == older and req.parameters["QUERY"] == older
    result = await _replay(manifest, replay_side_effect(load_exchanges("3c273")))
    assert set(result.diff["requests"]) == {"exoplanet_archive"}
    assert result.diff["requests"]["exoplanet_archive"]["query"]["old"] == older
    assert any(line.startswith("exoplanet_archive: the request sent differs from the manifest (parameters, query)")
               for line in result.explanation)


# ---------------------------------------------------------------------------
# CPU work off the event loop, fast hashing paths (major)
# ---------------------------------------------------------------------------


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_manifest_and_replay_cpu_work_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    seen: list[tuple[str, bool]] = []
    for name in ("build_manifest", "diff_science", "verify_manifest", "record_to_dict"):
        real = getattr(P, name)

        def spy(*args: Any, _real=real, _name=name, **kwargs: Any) -> Any:
            if args and args[0]:  # replay_manifest validates its tolerances with an empty diff on the loop
                seen.append((_name, _on_event_loop()))
            return _real(*args, **kwargs)

        monkeypatch.setattr(P, name, spy)
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=_with_tap_schema(load_exchanges("3c273")))
        with TestClient(_app(make_service(offline_client()))) as client:
            made = client.post("/api/v1/provenance/manifest", json={"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0})
            assert made.status_code == 200, made.text
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": made.json()["manifest"]})
            assert replayed.status_code == 200 and replayed.json()["identical"] is True, replayed.text
    names = {n for n, _ in seen}
    assert {"build_manifest", "diff_science", "verify_manifest", "record_to_dict"} <= names
    assert [n for n, on_loop in seen if on_loop] == [], seen


def test_fast_hash_paths_equal_the_reference_definition() -> None:
    rec = synthetic_record()
    rec["catalog_results"]["gaia_dr3"]["sources"].append(
        _source("gaia_dr3", "111", 150.0000003, 2.0, {"phot_g_mean_mag": 13.0, "ruwe": 1.2}))  # repeated id
    rec["catalog_results"]["gaia_dr3"]["sources"][0]["data"]["source_id"] = 3700386905605055360  # > 2**53
    science = P.science_payload(rec)
    rows = science["catalogs"]["gaia_dr3"]["sources"]
    assert [r["source_id"] for r in rows] == ["111", "111", "222"]
    assert rows == sorted(rows, key=lambda r: P._source_order({k: v for k, v in r.items() if k != "row_hash"}))
    for row in rows:
        assert row["row_hash"] == P.row_hash(row)
    reference = copy.deepcopy(json.loads(P.canonical_json(science)))
    for item in reference["catalogs"].values():
        item["sources"] = [{**P._hash_form_row(s), "row_hash": s["row_hash"]} for s in item["sources"]]
    assert P.content_hash(rec) == P.content_hash(science) == "sha256:" + P.sha256_hex(reference)
    changed = copy.deepcopy(rec)
    changed["catalog_results"]["gaia_dr3"]["sources"][1]["data"]["ruwe"] = 1.5
    new = P.science_payload(changed)
    assert P.diff_science(science, new, verified_row_hashes=True) == P.diff_science(science, new)


# ---------------------------------------------------------------------------
# CLI registry path; releases and citations checked against the definition (minor)
# ---------------------------------------------------------------------------


def _sdss_dr19_registry(tmp_path: Path) -> Path:
    catalogs = copy.deepcopy(DEFAULT_CATALOGS)
    catalogs["sdss"]["endpoint"] = catalogs["sdss"]["endpoint"].replace("/dr18/", "/dr19/")
    path = tmp_path / "registry.yaml"
    path.write_text(json.dumps({"catalogs": catalogs}), encoding="utf-8")  # JSON is YAML
    return path


def _record_with_empty_sdss() -> dict[str, Any]:
    rec = synthetic_record()
    rec["catalog_results"]["sdss"] = {"sources": [], "status": "empty", "row_count": 0, "truncated": False}
    rec["provenance"]["catalogs_planned"].append("sdss")
    return rec


def test_cli_uses_the_deployment_registry_and_labels_follow_the_definition(tmp_path: Path, monkeypatch, capsys) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(_record_with_empty_sdss()), encoding="utf-8")
    out = tmp_path / "manifest.json"
    args = _parser().parse_args(["manifest", "--record", str(record_path), "--no-live-release", "-o", str(out)])
    assert args.handler(args) == 0
    default = P.ProvenanceManifest.from_json(out.read_text(encoding="utf-8")).catalogs["sdss"]
    assert "/dr18/" in default.endpoint and default.release == "SDSS DR18 (SkyServer dr18 context)"
    assert default.citation_source.startswith("curated")

    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(_sdss_dr19_registry(tmp_path)))
    assert args.handler(args) == 0
    custom = P.ProvenanceManifest.from_json(out.read_text(encoding="utf-8")).catalogs["sdss"]
    assert "/dr19/" in custom.endpoint  # the request the deployment really sends
    # The registry's own text with the service it names, not the curated "SDSS DR18" label.
    assert custom.release != P.CATALOG_RELEASES["sdss"]["release"] and "/dr19/" in custom.release
    assert custom.release_source.startswith("registry (registry definition differs from the curated one")
    assert custom.living_database is None and custom.citation_source == "registry (unverified)"
    capsys.readouterr()
    cite = _parser().parse_args(["cite", "--catalogs", "sdss", "--json"])
    assert cite.handler(cite) == 0
    ack = json.loads(capsys.readouterr().out)["acknowledgements"][0]
    assert ack["references"] == [] and "differs from the service/table the curated entry" in ack["note"]


# ---------------------------------------------------------------------------
# Total outage: 502 / exit 3 instead of an all-failed manifest (minor)
# ---------------------------------------------------------------------------


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("simulated outage", request=request)


def test_a_search_manifest_during_a_total_outage_is_an_upstream_error(tmp_path: Path, capsys) -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=_connect_error)
        with TestClient(_app(make_service(offline_client()))) as client:
            res = client.post("/api/v1/provenance/manifest", json={"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 5})
        assert res.status_code == 502, res.text
        assert "every catalog failed to reach its archive" in res.json()["detail"]
        args = _parser().parse_args(["manifest", "--ra", str(RA_3C273), "--dec", str(DEC_3C273), "--radius", "5",
                                     "--no-live-release", "-o", str(tmp_path / "m.json")])
        assert args.handler(args) == 3
    assert "every catalog failed to reach its archive" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()


# ---------------------------------------------------------------------------
# Search fields equal api.SearchRequest (minor)
# ---------------------------------------------------------------------------


def test_search_fields_equal_api_search_request() -> None:
    import api

    ours, theirs = P.SearchFields.model_fields, api.SearchRequest.model_fields
    assert list(ours) == list(theirs)
    for name in theirs:
        assert ours[name].annotation == theirs[name].annotation, name
        assert ours[name].default == theirs[name].default, name
        assert repr(ours[name].metadata) == repr(theirs[name].metadata), name
    assert P.SearchFields.model_config.get("extra") == api.SearchRequest.model_config.get("extra") == "forbid"
    assert P.SEARCH_FIELDS == tuple(theirs)
    # Accepted as /api/v1/search accepts them (the crossmatch normalizes ra and checks the radius).
    for body in ({"ra": 360.0, "dec": 0.0}, {"ra": 1.0, "dec": 1.0, "radius_arcsec": 7200.0}):
        api.SearchRequest.model_validate(body)
        assert P.ManifestRequest.model_validate(body).search_fields()["ra"] == body["ra"]


# ---------------------------------------------------------------------------
# Citations for the other features; partial requests (minor)
# ---------------------------------------------------------------------------


def test_light_curve_sed_and_imaging_surveys_can_be_cited(capsys) -> None:
    from fastapi.testclient import TestClient

    bundle = P.citations_for(["ztf", "neowise", "gaia", "tess", "galex", "unwise"])
    keys = {r.key for r in bundle.references}
    assert {"2015JATIS...1a4003R", "2016SPIE.9913E..3EJ", "2005ApJ...619L...1M", "2007ApJS..173..682M",
            "2017ApJS..230...24B", "2014AJ....147..108L", "2017AJ....154..161M", "2022RNAAS...6..188M"} <= keys
    P.parse_bibtex(bundle.bibtex)
    assert "TESS mission is provided by the NASA Explorer Program" in bundle.acknowledgement_text
    assert "observations made with the Galaxy Evolution Explorer" in bundle.acknowledgement_text
    args = _parser().parse_args(["cite", "--catalogs", "gaia_dr3,simbad,ztf,tess"])
    assert args.handler(args) == 0
    assert "Ricker et al. 2015" in capsys.readouterr().out
    with TestClient(_app()) as client:
        res = client.get("/api/v1/citations", params={"catalogs": "ztf,neowise,gaia,tess"})
        assert res.status_code == 200 and res.json()["unknown"] == []
        assert [a["key"] for a in res.json()["acknowledgements"]] == ["ztf", "irsa", "neowise", "gaia_dr3", "tess"]
        partial = client.get("/api/v1/citations", params={"catalogs": "tess,kepler", "format": "bibtex"})
        assert partial.status_code == 200 and partial.headers["x-unknown-citations"] == "kepler"
        assert "@article{2015JATIS...1a4003R," in partial.text
    warn = _parser().parse_args(["cite", "--catalogs", "tess,kepler"])
    assert warn.handler(warn) == 0
    assert "no citation entry for ['kepler']" in capsys.readouterr().err


def test_official_neowise_heasarc_and_ned_acknowledgements() -> None:
    neowise = P.CITATION_ENTRIES["neowise"]
    assert "University of California, Los Angeles" in neowise.acknowledgement
    assert "University of Arizona" not in neowise.acknowledgement
    assert neowise.source_url == "https://wise2.ipac.caltech.edu/docs/release/neowise/"
    heasarc = P.CITATION_ENTRIES["heasarc"]
    assert heasarc.acknowledgement.startswith("This research has made use of data, software and/or web tools obtained "
                                              "from the High Energy Astrophysics Science Archive Research Center")
    assert heasarc.source_url == "https://heasarc.gsfc.nasa.gov/docs/faq.html"
    ned = P.CITATION_ENTRIES["ned"]
    assert "(NED)" not in ned.acknowledgement
    assert ned.source_url == "https://ned.ipac.caltech.edu/Documents/Overview/Acknowledgments"


# ---------------------------------------------------------------------------
# BibTeX validator and cite --from (minor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "@article{k1,\n author = {A, B}\n title = {T}\n journal = {J}\n year = {2000}\n}",
    "@article{k1, author = {A} title = {T}, journal = {J}, year = {2000}}",
    '@misc{k2, title = "T" year = 2000}',
])
def test_bibtex_validator_requires_commas_between_fields(text: str) -> None:
    with pytest.raises(P.BibTeXError, match="expected ','"):
        P.parse_bibtex(text)


def test_bibtex_validator_still_accepts_a_trailing_comma() -> None:
    entries = P.parse_bibtex("@article{k1,\n author = {A},\n title = {T},\n journal = {J},\n year = {2000},\n}")
    assert entries[0].fields["year"] == "2000"


def test_cite_from_a_manifest_of_another_schema_is_a_schema_error(tmp_path: Path, capsys) -> None:
    data = P.build_manifest(synthetic_record()).as_dict()
    data["schema"] = "astrosearch.provenance.manifest/2"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    args = _parser().parse_args(["cite", "--from", str(path)])
    assert args.handler(args) == 2
    err = capsys.readouterr().err
    assert "unsupported manifest schema 'astrosearch.provenance.manifest/2'" in err
    with pytest.raises(P.ManifestError, match="unsupported manifest schema"):
        P.catalogs_in(data)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


def _page_text(url: str) -> str | Exception:
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 astrosearch"})
        response.raise_for_status()
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        return exc
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", response.text)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))


@pytest.mark.live
@pytest.mark.parametrize(("key", "url"), [
    ("neowise", "https://wise2.ipac.caltech.edu/docs/release/neowise/"),
    ("heasarc", "https://heasarc.gsfc.nasa.gov/docs/faq.html"),
    ("ned", "https://ned.ipac.caltech.edu/Documents/Overview/Acknowledgments"),
    ("tess", "https://archive.stsci.edu/publishing/mission-acknowledgements"),
])
def test_live_acknowledgements_are_the_providers_wording(key: str, url: str) -> None:
    page = _page_text(url)
    if isinstance(page, Exception):
        if isinstance(page, httpx.HTTPStatusError) and page.response.status_code < 500 and page.response.status_code != 429:
            raise page
        pytest.skip(f"{url} unreachable: {page}")
    assert P.CITATION_ENTRIES[key].source_url == url
    assert P.CITATION_ENTRIES[key].acknowledgement in page


@pytest.mark.live
def test_live_galex_acknowledgement_is_the_mast_legacy_template() -> None:
    page = _page_text("https://archive.stsci.edu/publishing/mission-acknowledgements")
    if isinstance(page, Exception):
        pytest.skip(f"MAST unreachable: {page}")
    ours = P.CITATION_ENTRIES["galex"].acknowledgement
    assert ours.replace("the Galaxy Evolution Explorer,", "the mission ,") in page  # the placeholder, tags removed
    assert "Galaxy Evolution Explorer" in page.split("where the")[-1]  # listed among the legacy missions


@pytest.mark.live
def test_live_new_references_exist_with_the_recorded_metadata() -> None:
    keys = ["2015JATIS...1a4003R", "2016SPIE.9913E..3EJ", "2005ApJ...619L...1M", "2007ApJS..173..682M",
            "2017ApJS..230...24B", "2014AJ....147..108L", "2017AJ....154..161M", "2022RNAAS...6..188M"]

    async def run() -> list[P.ReferenceCheck]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await P.verify_references(client, [P.REFERENCES[k] for k in keys])

    checks = asyncio.run(run())
    if all(c.undecided for c in checks):
        pytest.skip(f"ADS / doi.org unreachable: {checks[0].problems}")
    assert not [c.key for c in checks if c.undecided], [(c.key, c.problems) for c in checks]
    for check in checks:
        assert check.ok and check.metadata_matches is True, (check.key, check.problems)
        assert check.doi_from_ads == P.REFERENCES[check.key].doi, check.key


@pytest.mark.live
def test_live_3c273_replay_while_irsa_tap_is_bypassed_matches_the_fallback_archive() -> None:
    """A real 3C 273 search whose 2MASS/AllWISE rows come from IRSA Gator (IRSA TAP made
    unreachable), replayed against the real archives: the replay asks Gator again and
    reproduces the 2MASS and AllWISE science (their rows are compared like with like)."""
    import os

    from crossmatch import CrossmatchService
    from providers import CacheManager, provider_map

    os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "2"

    async def run():
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            registry = CatalogRegistry()
            names = ["twomass_psc", "allwise", "gaia_dr3"]
            pinned = P._PinnedRegistry(registry, names)
            # An RFC 2606 host never resolves: IRSA TAP is "down" (ConnectError -> fallback).
            broken = {n: replace(d, endpoint="https://irsa-tap.invalid/TAP/sync")
                      if n in ("twomass_psc", "allwise") else d for n, d in pinned.catalogs.items()}
            original_registry = P._PinnedRegistry(registry, names, broken)
            service = CrossmatchService(original_registry, provider_map(client, timeout=120.0, cache=CacheManager(None)),
                                        radius_arcsec=10.0, timeout=150.0)
            record = await service.crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
            manifest = P.build_manifest(record, registry=registry)
            replay_service = CrossmatchService(P._PinnedRegistry(registry, names),
                                               provider_map(client, timeout=120.0, cache=CacheManager(None)),
                                               radius_arcsec=10.0, timeout=150.0)
            try:
                result = await P.replay_manifest(manifest, replay_service, client=client)
            except P.ReplayUnavailableError as exc:
                return manifest, None, str(exc)
            return manifest, result, None

    manifest, result, down = asyncio.run(run())
    if down:
        pytest.skip(f"archives unreachable: {down}")
    tm = manifest.catalogs["twomass_psc"]
    if tm.status == "failed":
        if tm.error_type in P.UNREACHABLE_ERRORS:
            pytest.skip(f"IRSA Gator unreachable: {tm.error_message}")
        raise AssertionError(tm.error_message)
    assert tm.provider == "irsa_gator" and P.served_by(tm) == "fallback irsa_gator"
    assert TWOMASS_3C273 in [s["source_id"] for s in manifest.science["catalogs"]["twomass_psc"]["sources"]]
    assert result is not None
    failed_now = {c["catalog"] for c in result.not_compared}
    for name in ("twomass_psc", "allwise"):
        if name in failed_now or (result.diff["catalogs"].get(name, {}).get("status") or {}).get("new") == "failed":
            pytest.skip(f"{name}: IRSA Gator failed during the replay")
        assert P.served_by(result.new_manifest.catalogs[name]) == "fallback irsa_gator", name
        assert name not in result.diff["catalogs"], (name, result.diff["catalogs"].get(name), result.explanation)
        assert name not in result.diff["requests"], result.diff["requests"]
