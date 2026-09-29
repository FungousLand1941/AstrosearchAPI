"""Offline tests for provenance.py: hashing, manifests, replay diffs, citations, router and CLI.

Archive traffic is replayed from the recorded 3C 273 fixtures (tests/fixtures/3c273);
NASA ADS link-gateway and doi.org answers from tests/fixtures/provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fixture_io import TARGETS, Exchange, load_exchanges, replay_side_effect, request_signature
from helpers import make_service, offline_client

import provenance as P
from models import DEFAULT_CATALOGS, CatalogRegistry, UnifiedRecord, validate_target

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "provenance"
RA_3C273, DEC_3C273 = TARGETS["3c273"]
GAIA_3C273 = "3700386905605055360"


@pytest.fixture(autouse=True)
def _fresh_release_cache():
    """Live release descriptions are cached per process; every test starts empty."""
    P._RELEASE_CACHE.clear()
    yield
    P._RELEASE_CACHE.clear()


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------


def _source(catalog: str, sid: str, ra: float, dec: float, data: dict[str, Any], *, stamp: str = "2026-01-01T00:00:00+00:00",
            elapsed: float = 12.3) -> dict[str, Any]:
    return {
        "catalog": catalog, "source_id": sid, "ra": ra, "dec": dec, "positional_error_arcsec": 0.05,
        "data": data, "epoch": 2016.0, "proper_motion_ra_masyr": None, "proper_motion_dec_masyr": None,
        "metadata": {"wavelength": "optical", "query_separation_arcsec": 0.01, "elapsed_ms": elapsed, "links": {}},
        "provenance": {"catalog": catalog, "provider": "tap", "source_id": sid, "retrieved_at": stamp,
                       "endpoint": "https://example.org/tap/sync", "search_radius_arcsec": 5.0,
                       "query_parameters": {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": f"SELECT * FROM {catalog}"}},
    }


def synthetic_record(*, stamp: str = "2026-01-01T00:00:00+00:00", elapsed: float = 12.3) -> dict[str, Any]:
    gaia = [
        _source("gaia_dr3", "111", 150.0000001, 2.0, {"phot_g_mean_mag": 12.3456789, "ruwe": 1.01}, stamp=stamp, elapsed=elapsed),
        _source("gaia_dr3", "222", 150.0010000, 2.0005, {"phot_g_mean_mag": 18.5, "ruwe": 0.99}, stamp=stamp, elapsed=elapsed),
    ]
    simbad = [_source("simbad", "HD 1", 150.0000002, 2.0000001, {"otype": "*", "V": 9.1}, stamp=stamp, elapsed=elapsed)]
    return {
        "target": {"ra": 150.0, "dec": 2.0, "frame": "icrs", "epoch": None, "pm_ra_masyr": None, "pm_dec_masyr": None},
        "catalogs_queried": 3,
        "catalog_results": {
            "gaia_dr3": {"sources": gaia, "status": "success", "row_count": 2, "elapsed_ms": elapsed, "truncated": False,
                         "query": "SELECT * FROM gaia_dr3", "endpoint": "https://example.org/tap/sync", "warnings": [f"w{elapsed}"]},
            "simbad": {"sources": simbad, "status": "success", "row_count": 1, "elapsed_ms": elapsed * 2, "truncated": False,
                       "query": "SELECT * FROM simbad", "endpoint": "https://example.org/tap/sync"},
            "nvss": {"sources": [], "status": "failed", "row_count": 0, "elapsed_ms": elapsed, "truncated": False,
                     "error_type": "QueryTimeoutError", "message": f"timed out after {elapsed}s"},
        },
        "counterparts": {},
        "failures": [{"catalog": "nvss", "status": "failed", "error_type": "QueryTimeoutError",
                      "message": f"timed out after {elapsed}s", "elapsed_ms": elapsed}],
        "provenance": {
            "query_radius_arcsec": 5.0, "effective_radius_arcsec": 5.0, "target_epoch": None, "target_proper_motion": None,
            "warnings": [f"generated at {stamp}"], "profile": None, "advanced_query": None,
            "catalogs_planned": ["gaia_dr3", "simbad", "nvss"], "catalog_stats": {}, "citations": {},
            "matches": [
                {"catalog": "gaia_dr3", "source_id": "111", "separation_arcsec": 0.00036, "confidence": 0.99},
                {"catalog": "simbad", "source_id": "HD 1", "separation_arcsec": 0.00081, "confidence": 0.98},
            ],
        },
        "crossmatch_groups": [{"group_id": "object-1", "catalogs": ["gaia_dr3", "simbad"], "wavelengths": ["optical"],
                               "members": [{"catalog": "gaia_dr3", "source_id": "111"}, {"catalog": "simbad", "source_id": "HD 1"}]}],
        "resolved_object": None,
    }


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_json_is_sorted_compact_and_float_rounded() -> None:
    assert P.canonical_json({"b": 1, "a": 0.1 + 0.2, "c": [True, None, "x"]}) == '{"a":0.3,"b":1,"c":[true,null,"x"]}'
    # Last-bit float noise does not change the canonical form; a real change does.
    assert P.canonical_json(187.2779154) == P.canonical_json(187.2779154 + 1e-13)
    assert P.canonical_json(187.2779154) != P.canonical_json(187.2779164)


def test_canonicalize_numpy_nan_negative_zero_and_sets() -> None:
    value = {"i": np.int64(7), "f": np.float32(0.5), "b": np.bool_(True), "arr": np.array([1.0, 2.0]),
             "nan": float("nan"), "inf": float("-inf"), "negzero": -0.0, "set": {3, 1, 2}, 5: "int-key"}
    out = P.canonicalize(value)
    assert out == {"i": 7, "f": 0.5, "b": True, "arr": [1.0, 2.0], "nan": "NaN", "inf": "-Infinity", "negzero": 0.0,
                   "set": [1, 2, 3], "5": "int-key"}
    json.loads(P.canonical_json(value))  # strict JSON (no NaN literals)


# ---------------------------------------------------------------------------
# Content hash stability
# ---------------------------------------------------------------------------


def test_same_science_with_different_timestamps_and_timings_has_same_hash() -> None:
    a = synthetic_record(stamp="2026-01-01T00:00:00+00:00", elapsed=12.3)
    b = synthetic_record(stamp="2031-07-04T12:34:56+00:00", elapsed=987.6)
    b["catalog_results"]["gaia_dr3"]["sources"][0]["metadata"]["links"] = {"SIMBAD": "https://x"}
    b["resolved_object"] = {"query": "anything", "resolver": "Sesame"}
    assert P.content_hash(a) == P.content_hash(b)
    assert P.content_hash(a).startswith("sha256:") and len(P.content_hash(a)) == len("sha256:") + 64


def test_row_and_catalog_order_do_not_change_hash() -> None:
    a = synthetic_record()
    b = synthetic_record()
    b["catalog_results"] = dict(reversed(list(b["catalog_results"].items())))
    b["catalog_results"]["gaia_dr3"]["sources"].reverse()
    b["provenance"]["matches"].reverse()
    b["crossmatch_groups"][0]["members"].reverse()
    assert P.content_hash(a) == P.content_hash(b)


@pytest.mark.parametrize("mutate", [
    lambda r: r["catalog_results"]["gaia_dr3"]["sources"][0]["data"].__setitem__("phot_g_mean_mag", 12.3456790),
    lambda r: r["catalog_results"]["gaia_dr3"]["sources"][0].__setitem__("ra", 150.0000001 + 1e-6),  # 3.6 mas
    lambda r: r["catalog_results"]["gaia_dr3"]["sources"].pop(),
    lambda r: r["catalog_results"]["simbad"]["sources"][0].__setitem__("source_id", "HD 2"),
    lambda r: r["catalog_results"]["nvss"].update(status="empty", error_type=None) or r["failures"].clear(),
    lambda r: r["provenance"]["matches"].pop(),
    lambda r: r["target"].__setitem__("dec", 2.001),
])
def test_changed_science_changes_hash(mutate) -> None:
    a = synthetic_record()
    b = synthetic_record()
    mutate(b)
    assert P.content_hash(a) != P.content_hash(b)


def test_unified_record_object_and_its_dict_hash_identically() -> None:
    from models import CatalogSource

    rec = synthetic_record()
    obj_results = {}
    for name, res in rec["catalog_results"].items():
        res = dict(res)
        res["sources"] = [CatalogSource(**{k: s[k] for k in ("catalog", "source_id", "ra", "dec", "positional_error_arcsec",
                                                              "data", "metadata", "provenance", "epoch",
                                                              "proper_motion_ra_masyr", "proper_motion_dec_masyr")})
                          for s in res["sources"]]
        obj_results[name] = res
    obj = UnifiedRecord(target=rec["target"], catalogs_queried=3, catalog_results=obj_results, counterparts={},
                        failures=rec["failures"], provenance=rec["provenance"], crossmatch_groups=rec["crossmatch_groups"])
    assert P.content_hash(obj) == P.content_hash(rec) == P.content_hash(obj.as_dict())


def test_science_payload_excludes_volatile_fields() -> None:
    text = P.canonical_json(P.science_payload(synthetic_record()))
    for volatile in ("retrieved_at", "elapsed_ms", "timed out after", "generated at", "query_separation_arcsec"):
        assert volatile not in text
    assert '"error_type":"QueryTimeoutError"' in text


# ---------------------------------------------------------------------------
# Manifest round trip & integrity
# ---------------------------------------------------------------------------


def test_manifest_round_trip_and_integrity() -> None:
    m = P.build_manifest(synthetic_record(), created_at="2026-09-28T00:00:00+00:00")
    text = m.to_json()
    again = P.ProvenanceManifest.from_json(text)
    assert again.as_dict() == m.as_dict()
    assert json.loads(again.to_json()) == json.loads(text)
    assert P.verify_manifest(again) == {"content_hash_ok": True, "query_hash_ok": True, "row_hashes_ok": True}
    assert m.row_counts == {"gaia_dr3": 2, "nvss": 0, "simbad": 1}
    assert m.catalogs["nvss"].status == "failed" and m.catalogs["nvss"].error_type == "QueryTimeoutError"
    assert m.catalogs["gaia_dr3"].request_source == "recorded"
    assert m.catalogs["gaia_dr3"].parameters["QUERY"] == "SELECT * FROM gaia_dr3"
    assert m.catalogs["gaia_dr3"].release.startswith("Gaia DR3") and m.catalogs["gaia_dr3"].living_database is False
    assert m.catalogs["simbad"].living_database is True
    assert m.software["name"] == "astrosearch" and m.software["version"]
    assert m.software["dependencies"]["astropy"]
    assert m.created_at == "2026-09-28T00:00:00+00:00"


def test_manifest_tampering_is_detected() -> None:
    data = P.build_manifest(synthetic_record()).as_dict()
    tampered = copy.deepcopy(data)
    tampered["science"]["catalogs"]["gaia_dr3"]["sources"][0]["data"]["ruwe"] = 5.0
    check = P.verify_manifest(tampered)
    assert check["content_hash_ok"] is False and check["row_hashes_ok"] is False
    tampered = copy.deepcopy(data)
    tampered["catalogs"]["gaia_dr3"]["parameters"]["QUERY"] = "SELECT 1"
    assert P.verify_manifest(tampered)["query_hash_ok"] is False


def test_compact_manifest_keeps_digests_not_rows() -> None:
    m = P.build_manifest(synthetic_record(), include_rows=False)
    assert m.compact
    row = m.science["catalogs"]["gaia_dr3"]["sources"][0]
    assert set(row) == {"source_id", "ra", "dec", "positional_error_arcsec", "row_hash"}
    assert row["positional_error_arcsec"] == 0.05  # used for the sigma-scaled "moved" threshold
    assert m.content_hash == P.content_hash(synthetic_record())
    assert P.verify_manifest(m)["content_hash_ok"] is None


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda d: d.__setitem__("schema", "other/1"), "unsupported manifest schema"),
    (lambda d: d.pop("content_hash"), "missing"),
    (lambda d: d.__setitem__("radius_arcsec", -1), "radius"),
    (lambda d: d["science"].__setitem__("schema", "x"), "science payload"),
    (lambda d: d.__setitem__("catalogs", []), "catalogs"),
])
def test_invalid_manifests_are_rejected(mutate, message: str) -> None:
    data = P.build_manifest(synthetic_record()).as_dict()
    mutate(data)
    with pytest.raises(P.ManifestError, match=message):
        P.ProvenanceManifest.from_dict(data)
    with pytest.raises(P.ManifestError):
        P.ProvenanceManifest.from_json("{not json")


def test_input_target_does_not_take_an_adopted_proper_motion_as_input() -> None:
    rec = synthetic_record()
    rec["target"].update(epoch=2000.0, pm_ra_masyr=-798.6, pm_dec_masyr=10328.1)
    rec["provenance"]["target_proper_motion"] = {"pm_ra_masyr": -798.6, "pm_dec_masyr": 10328.1, "source": "adopted",
                                                 "catalog": "gaia_dr3", "source_id": "111"}
    given = P.input_target(rec)
    assert given.epoch == 2000.0 and given.proper_motion is None
    rec["provenance"]["target_proper_motion"]["source"] = "input"
    assert P.input_target(rec).proper_motion == (-798.6, 10328.1)


def test_input_target_of_an_advanced_query_without_proper_motion_handling() -> None:
    rec = synthetic_record()
    rec["provenance"]["advanced_query"] = {
        "target": {"ra": 150.0, "dec": 2.0, "frame": "icrs", "epoch": 2000.0, "pm_ra_masyr": 5.0, "pm_dec_masyr": 6.0},
        "radius_arcsec": 5.0, "proper_motion": False,
    }
    given = P.input_target(rec)
    assert given.epoch is None and given.proper_motion is None  # as CrossmatchService ran it
    rec["provenance"]["advanced_query"]["proper_motion"] = True
    assert P.input_target(rec).proper_motion == (5.0, 6.0)


def test_build_manifest_rejects_records_without_target_or_radius() -> None:
    rec = synthetic_record()
    rec["target"] = {}
    with pytest.raises(P.ManifestError):
        P.build_manifest(rec)
    rec = synthetic_record()
    del rec["provenance"]["query_radius_arcsec"]
    with pytest.raises(P.ManifestError):
        P.build_manifest(rec)


# ---------------------------------------------------------------------------
# Science diff (synthetic)
# ---------------------------------------------------------------------------


def test_diff_reports_added_removed_moved_changed_and_status() -> None:
    old = P.science_payload(synthetic_record())
    rec = synthetic_record()
    gaia = rec["catalog_results"]["gaia_dr3"]["sources"]
    gaia[0]["ra"] += 10.0 / 3600.0 / 1000.0          # 10 mas east (cos dec ~ 1)
    gaia[1]["data"]["phot_g_mean_mag"] = 18.6
    rec["catalog_results"]["simbad"]["sources"].append(_source("simbad", "HD 3", 150.001, 2.001, {"otype": "G"}))
    rec["catalog_results"]["simbad"]["row_count"] = 2
    rec["catalog_results"]["nvss"] = {"sources": [], "status": "empty", "row_count": 0, "truncated": False}
    rec["failures"] = []
    new = P.science_payload(rec)
    diff = P.diff_science(old, new)
    assert diff["equivalent"] is False
    moved = diff["catalogs"]["gaia_dr3"]["moved"]
    assert [m["source_id"] for m in moved] == ["111"]
    assert moved[0]["separation_arcsec"] == pytest.approx(0.010, abs=2e-5)
    changed = diff["catalogs"]["gaia_dr3"]["changed"]
    assert changed == [{"source_id": "222", "fields": {"data.phot_g_mean_mag": {"old": 18.5, "new": 18.6}}}]
    assert diff["catalogs"]["simbad"]["added"][0]["source_id"] == "HD 3"
    assert diff["catalogs"]["simbad"]["row_count"] == {"old": 1, "new": 2}
    assert diff["catalogs"]["nvss"]["status"]["old"] == "failed" and diff["catalogs"]["nvss"]["status"]["new"] == "empty"
    assert diff["totals"] == {"added": 1, "removed": 0, "moved": 1, "changed": 1, "status_changed": 1, "matches_changed": 0}
    back = P.diff_science(new, old)
    assert back["catalogs"]["simbad"]["removed"][0]["source_id"] == "HD 3"


def test_diff_position_tolerance() -> None:
    old = P.science_payload(synthetic_record())
    rec = synthetic_record()
    rec["catalog_results"]["gaia_dr3"]["sources"][0]["ra"] += 0.5 / 3600.0 / 1000.0  # 0.5 mas
    new = P.science_payload(rec)
    assert P.content_hash(old) != P.content_hash(new)  # the hash sees every change ...
    within = P.diff_science(old, new, position_tolerance_arcsec=1e-3)
    assert within["equivalent"] is True  # ... the diff tolerates sub-mas motion
    strict = P.diff_science(old, new, position_tolerance_arcsec=1e-4)
    assert strict["catalogs"]["gaia_dr3"]["moved"][0]["source_id"] == "111"
    with pytest.raises(ValueError):
        P.diff_science(old, new, position_tolerance_arcsec=-1)


def test_diff_catalog_set_matches_groups_and_compact_rows() -> None:
    old = P.science_payload(synthetic_record())
    rec = synthetic_record()
    del rec["catalog_results"]["simbad"]
    rec["provenance"]["matches"] = rec["provenance"]["matches"][:1]
    rec["crossmatch_groups"][0]["members"] = rec["crossmatch_groups"][0]["members"][:1]
    rec["catalog_results"]["gaia_dr3"]["sources"][1]["data"]["ruwe"] = 2.5
    new = P.science_payload(rec)
    diff = P.diff_science(old, new)
    assert diff["catalogs_removed"] == ["simbad"]
    assert diff["matches"]["removed"] == ["simbad:HD 1"]
    assert diff["groups"]["removed"] == [["gaia_dr3:111", "simbad:HD 1"]]
    compact = P.diff_science(P.compact_science(old), P.compact_science(new))
    change = compact["catalogs"]["gaia_dr3"]["changed"][0]
    assert change["source_id"] == "222" and change["fields"] is None and change["old_row_hash"] != change["new_row_hash"]


# ---------------------------------------------------------------------------
# Real archive responses (3C 273 fixtures)
# ---------------------------------------------------------------------------


async def _crossmatch_3c273(exchanges: list[Exchange] | None = None):
    ex = exchanges if exchanges is not None else load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(ex))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
    return record, service


async def test_manifest_of_real_3c273_result_records_exact_requests() -> None:
    record, service = await _crossmatch_3c273()
    m = P.build_manifest(record, registry=service.registry)
    assert set(m.catalogs) == set(service.registry.enabled_catalogs())
    # Known objects in the scientific payload.
    gaia_ids = [s["source_id"] for s in m.science["catalogs"]["gaia_dr3"]["sources"]]
    assert gaia_ids == [GAIA_3C273]
    assert "3C 273" in [s["source_id"] for s in m.science["catalogs"]["simbad"]["sources"]]
    assert m.row_counts["ned"] == 21
    assert m.catalogs["exoplanet_archive"].status == "empty" and m.catalogs["vlass"].status == "empty"
    for name, req in m.catalogs.items():
        if req.request_source in ("recorded", "recorded (result metadata)"):
            assert req.request_verified is True, f"{name}: reconstruction differs from the recorded request"
        else:
            assert req.request_source == "reconstructed", name
        assert req.parameters, name
    assert m.catalogs["gaia_dr3"].query.startswith("SELECT TOP 201 source_id")
    assert m.catalogs["gaia_dr3"].endpoint == "https://gea.esac.esa.int/tap-server/tap/sync"
    assert m.catalogs["panstarrs_dr2"].http_method == "GET" and m.catalogs["gaia_dr3"].http_method == "POST"
    assert m.catalogs["sdss"].query.startswith("SELECT TOP") and "fGetNearbyObjEq" in m.catalogs["sdss"].query
    assert P.verify_manifest(m) == {"content_hash_ok": True, "query_hash_ok": True, "row_hashes_ok": True}


@pytest.mark.parametrize("catalog", ["exoplanet_archive", "vlass"])
async def test_reconstructed_requests_equal_the_recorded_archive_requests(catalog: str) -> None:
    """Empty results leave no row provenance: the request is taken from the ADQL the provider
    recorded in the result metadata, checked against the registry, and equals what was sent."""
    record, service = await _crossmatch_3c273()
    req = P.build_manifest(record, registry=service.registry).catalogs[catalog]
    assert req.request_source == "recorded (result metadata)" and req.request_verified is True
    assert req.query == record.catalog_results[catalog]["query"]
    recorded = [e for e in load_exchanges("3c273", [catalog])][-1]
    assert recorded.url_base == req.endpoint
    sent_form = httpx.QueryParams(req.parameters)
    assert request_signature("", str(sent_form)) == recorded.signature


async def test_two_runs_of_the_same_query_give_the_same_hashes() -> None:
    first, service = await _crossmatch_3c273()
    second, _ = await _crossmatch_3c273()
    m1 = P.build_manifest(first, registry=service.registry, created_at="2026-09-28T00:00:00+00:00")
    m2 = P.build_manifest(second, registry=service.registry, created_at="2031-01-01T12:00:00+00:00")
    assert m1.created_at != m2.created_at  # volatile fields differ ...
    assert m1.content_hash == m2.content_hash  # ... the science and the question do not
    assert m1.query_hash == m2.query_hash


async def test_offline_replay_of_3c273_manifest_is_identical() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as client:
            result = await P.replay_manifest(json.loads(manifest.to_json()), make_service(client))
    assert result.identical and result.equivalent_within_tolerance
    assert result.new_content_hash == manifest.content_hash
    assert result.diff["totals"] == {"added": 0, "removed": 0, "moved": 0, "changed": 0, "status_changed": 0,
                                     "matches_changed": 0}
    assert result.diff["requests"] == {}
    assert result.explanation[0].startswith("Content hash identical")
    assert result.new_manifest.query_hash == manifest.query_hash


def _modified_exchanges() -> list[Exchange]:
    """3C 273 fixtures with Gaia's position shifted 36 mas and one NED row removed."""
    out = []
    for ex in load_exchanges("3c273"):
        if ex.catalog == "gaia_dr3":
            payload = json.loads(ex.content)
            cols = [c["name"] for c in payload["metadata"]]
            payload["data"][0][cols.index("ra")] += 1e-5
            ex = replace(ex, content=json.dumps(payload).encode())
        elif ex.catalog == "ned":
            payload = json.loads(ex.content)
            payload["data"] = payload["data"][:-1]
            ex = replace(ex, content=json.dumps(payload).encode())
        out.append(ex)
    return out


async def test_replay_reports_moved_and_removed_sources_when_archives_change() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(_modified_exchanges()))
        async with offline_client() as client:
            result = await P.replay_manifest(manifest, make_service(client))
    assert not result.identical and not result.equivalent_within_tolerance
    gaia = result.diff["catalogs"]["gaia_dr3"]
    assert gaia["moved"][0]["source_id"] == GAIA_3C273
    assert gaia["moved"][0]["separation_arcsec"] == pytest.approx(0.036, abs=0.001)
    assert set(gaia["moved"][0]["fields"]) == {"data.ra"}  # the archive's own RA column moved with it
    assert "changed" not in gaia
    ned = result.diff["catalogs"]["ned"]
    assert len(ned["removed"]) == 1 and ned["row_count"] == {"old": 21, "new": 20}
    text = " ".join(result.explanation)
    assert "gaia_dr3: 1 moved -- a fixed data release" in text
    assert "ned: 1 removed -- a continuously updated database" in text
    assert result.diff["requests"] == {}  # same requests, different answers


async def test_replay_reports_a_catalog_that_fails_during_replay() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    error_body = (Path(__file__).parent / "fixtures" / "errors" / "heasarc_bad_column.0.body").read_bytes()
    exchanges = [replace(e, content=error_body, content_type="text/xml") if e.catalog == "chandra" else e
                 for e in load_exchanges("3c273")]
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            result = await P.replay_manifest(manifest, make_service(client))
    status = result.diff["catalogs"]["chandra"]["status"]
    assert status["old"] == "success" and status["new"] == "failed" and status["new_error_type"] == "CatalogQueryError"
    assert any(line.startswith("chandra: failed during replay (CatalogQueryError, query error)") for line in result.explanation)
    assert result.diff["catalogs"]["chandra"]["removed"][0]["source_id"] == "2CXO J122906.6+020308"
    assert not any(line.startswith("chandra: 1 removed") for line in result.explanation)  # caused by the failure


async def test_replay_of_an_edited_manifest_reports_the_integrity_failure() -> None:
    record, service = await _crossmatch_3c273()
    data = P.build_manifest(record, registry=service.registry).as_dict()
    data["science"]["catalogs"]["gaia_dr3"]["sources"][0]["data"]["phot_g_mean_mag"] = 99.0
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as client:
            result = await P.replay_manifest(data, make_service(client))
    # The archives still give the science the stored hash describes ...
    assert result.identical
    # ... but the edited payload is exposed by the integrity check and the row diff.
    assert result.explanation[0].startswith("Manifest integrity check failed")
    change = result.diff["catalogs"]["gaia_dr3"]["changed"][0]["fields"]["data.phot_g_mean_mag"]
    assert change["old"] == 99.0 and change["new"] == pytest.approx(12.85, abs=0.05)


async def test_replay_raises_when_every_archive_is_unreachable() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated outage", request=request)

    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=timeout)
        async with offline_client() as client:
            with pytest.raises(P.ReplayUnavailableError, match="every catalog failed"):
                await P.replay_manifest(manifest, make_service(client))


async def test_replay_pins_the_manifest_catalog_set_even_if_the_registry_changed() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    registry = CatalogRegistry()
    registry._catalogs["gaia_dr3"] = replace(registry._catalogs["gaia_dr3"], enabled=False)
    registry._catalogs["rosat_bsc"] = replace(registry._catalogs["rosat_bsc"], enabled=True)
    del registry._catalogs["xmm"]
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as client:
            result = await P.replay_manifest(manifest, make_service(client, registry=registry))
    assert result.missing_catalogs == ["xmm"]
    assert "gaia_dr3" in result.new_manifest.catalogs and "rosat_bsc" not in result.new_manifest.catalogs
    assert result.diff["catalogs_removed"] == ["xmm"]
    assert "xmm" in " ".join(result.explanation)


async def test_advanced_query_manifest_replays_through_the_advanced_path() -> None:
    from crossmatch import AdvancedQuery

    ex = load_exchanges("3c273", ["gaia_dr3", "simbad", "ned"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(ex))
        async with offline_client() as client:
            service = make_service(client)
            query = AdvancedQuery.from_dict({"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0,
                                             "catalogs": ["gaia_dr3", "simbad", "ned"], "min_confidence": 0.0})
            record = await service.crossmatch(RA_3C273, DEC_3C273, query=query)
            manifest = P.build_manifest(record, registry=service.registry)
            assert manifest.query["mode"] == "advanced"
            assert sorted(manifest.catalogs) == ["gaia_dr3", "ned", "simbad"]
            result = await P.replay_manifest(manifest, service)
    assert result.identical


# ---------------------------------------------------------------------------
# Citations & BibTeX
# ---------------------------------------------------------------------------

# ADS bibcode: YYYY JJJJJ VVVV M PPPP A (19 characters; arXiv ids fill volume/qualifier/page with digits).
BIBCODE = re.compile(r"^\d{4}[A-Za-z&.]{5}[A-Za-z0-9.]{4}[A-Za-z0-9.][A-Za-z0-9.]{4}[A-Z.]$")


def test_every_bibcode_is_well_formed_and_every_entry_resolves() -> None:
    for ref in P.REFERENCES.values():
        if ref.bibcode:
            assert len(ref.bibcode) == 19 and BIBCODE.match(ref.bibcode), ref.bibcode
            assert ref.key == ref.bibcode
            assert str(ref.year) == ref.bibcode[:4]
        else:
            assert ref.doi, f"{ref.key}: a reference needs a bibcode or a DOI"
    for entry in P.CITATION_ENTRIES.values():
        assert entry.acknowledgement.strip(), entry.key
        for key in entry.references:
            assert key in P.REFERENCES, f"{entry.key} cites unknown reference {key}"
    for alias, target in P.CITATION_ALIASES.items():
        assert target in P.CITATION_ENTRIES, alias


def test_every_registry_catalog_has_a_citation_and_release() -> None:
    for name in DEFAULT_CATALOGS:
        assert name in P.CITATION_ENTRIES, name
        assert P.CITATION_ENTRIES[name].references, name
        assert name in P.CATALOG_RELEASES, name


def test_required_catalog_references_are_cited() -> None:
    expected = {
        "gaia_dr3": "2023A&A...674A...1G", "simbad": "2000A&AS..143....9W", "twomass_psc": "2006AJ....131.1163S",
        "allwise": "2010AJ....140.1868W", "panstarrs_dr2": "2020ApJS..251....7F", "sdss": "2023ApJS..267...44A",
        "first": "1995ApJ...450..559B", "nvss": "1998AJ....115.1693C", "vlass": "2020PASP..132c5001L",
        "lotss_dr2": "2022A&A...659A...1S", "rosat": "2016A&A...588A.103B", "chandra": "2010ApJS..189...37E",
        "xmm": "2020A&A...641A.136W", "exoplanet_archive": "2013PASP..125..989A", "cds_xmatch": "2020ASPC..522..125P",
        "astropy": "2022ApJ...935..167A", "skybot": "2006ASPC..351..367B", "ztf": "2019PASP..131a8002B",
        "alerce": "2021AJ....161..242F", "svo_fps": "2012ivoa.rept.1015R", "hips": "2015A&A...578A.114F",
    }
    for key, bibcode in expected.items():
        assert bibcode in P.CITATION_ENTRIES[key].references, (key, bibcode)
    assert "2016arXiv161205560C" in P.CITATION_ENTRIES["panstarrs_dr2"].references
    assert "NED_DOI_2019" in P.CITATION_ENTRIES["ned"].references


def test_citations_add_hosting_archives_and_dedupe_acknowledgements() -> None:
    bundle = P.citations_for(["gaia", "FIRST", "nvss", "chandra", "vlass", "twomass_psc"], CatalogRegistry())
    assert bundle.keys == ["gaia_dr3", "first", "heasarc", "nvss", "chandra", "vlass", "vizier", "twomass_psc", "irsa"]
    text = bundle.acknowledgement_text
    assert text.count("High Energy Astrophysics Science Archive Research Center") == 1
    assert text.count("Associated Universities, Inc.") == 1
    assert "VizieR catalogue access tool" in text and "Infrared Science Archive" in text
    keys = [r.key for r in bundle.references]
    assert len(keys) == len(set(keys))
    entries = P.parse_bibtex(bundle.bibtex)
    assert [e.key for e in entries] == keys
    gaia = next(e for e in entries if e.key == "2023A&A...674A...1G")
    assert gaia.fields["doi"] == "10.1051/0004-6361/202243940" and gaia.fields["year"] == "2023"
    assert gaia.fields["journal"] == "Astronomy \\& Astrophysics"
    first_ack = next(a for a in bundle.as_dict()["acknowledgements"] if a["key"] == "first")
    assert first_ack["registry_citation"].startswith("Becker, White & Helfand 1995")


def test_unknown_citation_names_are_rejected_with_the_known_list() -> None:
    with pytest.raises(P.UnknownCitationError) as info:
        P.citations_for(["gaia_dr3", "not_a_catalog"])
    assert "not_a_catalog" in str(info.value) and "gaia_dr3" in str(info.value)


def test_all_references_produce_valid_bibtex() -> None:
    text = P.to_bibtex(P.REFERENCES.values())
    entries = P.parse_bibtex(text)
    assert len(entries) == len(P.REFERENCES)
    by_key = {e.key: e for e in entries}
    assert by_key["2016A&A...588A.103B"].fields["author"].startswith('Boller, Th. and Freyberg, M. J. and Tr{\\"u}mper, J.')
    assert by_key["2016arXiv161205560C"].fields["eprint"] == "1612.05560"
    assert by_key["2012ASPC..461..291B"].entry_type == "inproceedings"
    assert by_key["2013wise.rept....1C"].fields["institution"] == "IPAC/Caltech"
    assert by_key["NED_DOI_2019"].fields["doi"] == "10.26132/NED1"
    assert by_key["2023A&A...674A...1G"].fields["author"].endswith("and others")
    assert by_key["2023A&A...674A...1G"].fields["adsurl"] == "https://ui.adsabs.harvard.edu/abs/2023A%26A...674A...1G/abstract"


@pytest.mark.parametrize(("text", "message"), [
    ("@article{k1, author={A}, title={T}, journal={J}, year={2020}", "unbalanced|not closed"),
    ("@article{k1, author={A}, title={T {x}, journal={J}, year={2020}}", "unbalanced"),
    ("@article{k1, author={A}, title={T}, year={2020}}", "journal"),
    ("@article{k1, author={A}, title={T}, journal={A & A}, year={2020}}", "unescaped '&'"),
    ("@misc{k1, title={T}}\n@misc{k1, title={U}}", "duplicate citation key"),
    ("@article{k1, author={A}, title={T}, journal={J}, year={20}}", "four-digit"),
    ("@article{k1, author={A}, author={B}, title={T}, journal={J}, year={2020}}", "repeated"),
    ("@article{author={A}, title={T}}", "no citation key"),
    ("just text", "no BibTeX entries"),
])
def test_bibtex_validator_rejects_broken_entries(text: str, message: str) -> None:
    with pytest.raises(P.BibTeXError, match=message):
        P.parse_bibtex(text)


def test_bibtex_validator_accepts_quotes_numbers_concatenation_and_comments() -> None:
    text = ('@comment{ignored}\n@string{aj = "AJ"}\n'
            '@article{2006AJ....131.1163S, author = "Skrutskie, M. F. and others", title = "{The 2MASS}",'
            ' journal = aj # " (journal)", year = 2006, volume = 131, pages = "1163--1183",'
            ' adsurl = {https://ui.adsabs.harvard.edu/abs/2006AJ....131.1163S/abstract}}')
    (entry,) = P.parse_bibtex(text)
    assert entry.fields["year"] == "2006" and entry.fields["journal"] == "aj (journal)"


# ---------------------------------------------------------------------------
# Reference verification (recorded ADS gateway / doi.org answers)
# ---------------------------------------------------------------------------


def _ads_replay(name: str):
    meta = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    table = {}
    for idx, ex in enumerate(meta["exchanges"]):
        table[(ex["method"], ex["url"])] = (ex["status_code"], ex["headers"], (FIXTURES / f"{name}.{idx}.body").read_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url))
        if key not in table:
            raise AssertionError(f"unrecorded request {key}")
        status, headers, body = table[key]
        return httpx.Response(status, headers=headers, content=body)

    return meta, handler


@pytest.mark.parametrize(("name", "reference"), [
    ("gaia_dr3", "2023A&A...674A...1G"), ("first_becker1995", "1995ApJ...450..559B"),
    ("allwise_expsup", "2013wise.rept....1C"), ("ned_dataset_doi", "NED_DOI_2019"),
])
async def test_verify_reference_against_recorded_ads_and_doi_answers(name: str, reference: str) -> None:
    _meta, handler = _ads_replay(name)
    with respx.mock(assert_all_called=True) as router:
        router.route().mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            check = await P.verify_reference(client, P.REFERENCES[reference])
    assert check.ok and check.exists, check.problems
    if name == "gaia_dr3":
        assert check.link_type == "PUB_HTML" and check.doi_matches is True and check.metadata_matches is True
        assert check.metadata["volume"] == "674" and check.metadata["year"] == 2023
    if name == "first_becker1995":
        assert check.link_type == "EPRINT_HTML"  # no publisher link: ADS points to SIMBAD
        assert check.metadata_matches is True  # DOI 10.1086/176166 metadata agrees
    if name == "allwise_expsup":
        assert check.doi_from_ads is None and "allwise/expsup" in check.resolved_url


async def test_verify_reference_detects_unknown_bibcode_and_wrong_metadata() -> None:
    _meta, handler = _ads_replay("nonexistent")
    fake = P.Reference(key="2099XXX.....1....1Z", entry_type="article", authors=("Nobody, N.",), title="Does not exist",
                       year=2099, bibcode="2099XXX.....1....1Z", journal="None", volume="1", pages="1")
    with respx.mock(assert_all_called=True) as router:
        router.route().mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            check = await P.verify_reference(client, fake)
    # ADS answers 404 to every link type both for an unknown bibcode and for a real one
    # without links: undecided, never "verified".
    assert check.exists is None and not check.ok
    assert "existence undecided" in check.problems[0]

    _, handler = _ads_replay("gaia_dr3")
    wrong = replace(P.REFERENCES["2023A&A...674A...1G"], volume="675")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            check = await P.verify_reference(client, wrong)
    assert check.exists is True and check.metadata_matches is False and not check.ok
    assert "volume 674 != 675" in check.problems


async def test_verify_reference_is_undecided_when_ads_is_down() -> None:
    with respx.mock() as router:
        router.route().mock(return_value=httpx.Response(503))
        async with httpx.AsyncClient() as client:
            check = await P.verify_reference(client, P.REFERENCES["2000A&AS..143....9W"])
    assert check.exists is None and not check.ok and "HTTP 503" in check.problems[0]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _app(service=None):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(P.router)
    if service is not None:
        app.state.service = service
        app.state.registry = service.registry
    return app


def test_citations_route_json_bibtex_and_errors() -> None:
    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        ok = client.get("/api/v1/citations", params={"catalogs": "gaia_dr3,simbad,hips"})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert set(body) >= {"acknowledgements", "bibtex", "acknowledgement_text", "references", "keys"}
        assert [a["key"] for a in body["acknowledgements"]] == ["gaia_dr3", "simbad", "hips"]
        assert P.parse_bibtex(body["bibtex"])
        bib = client.get("/api/v1/citations", params={"catalogs": "nvss", "format": "bibtex"})
        assert bib.status_code == 200 and bib.headers["content-type"].startswith("application/x-bibtex")
        assert "@article{1998AJ....115.1693C," in bib.text
        # A name without an entry does not cost the others their citations; it is reported.
        partial = client.get("/api/v1/citations", params={"catalogs": "gaia_dr3,nope"})
        assert partial.status_code == 200 and partial.json()["unknown"] == ["nope"]
        assert [a["key"] for a in partial.json()["acknowledgements"]] == ["gaia_dr3"]
        bad = client.get("/api/v1/citations", params={"catalogs": "nope,nada"})
        assert bad.status_code == 422 and "nope" in bad.json()["detail"]
        assert client.get("/api/v1/citations", params={"catalogs": " , "}).status_code == 422
        assert client.get("/api/v1/citations").status_code == 422
        sources = client.get("/api/v1/citations/sources").json()
        assert {s["key"] for s in sources["sources"]} == set(P.CITATION_ENTRIES)


def _heasarc_tap_schema(request: httpx.Request) -> httpx.Response:
    """Recorded HEASARC TAP_SCHEMA.tables answer (the manifest's live release lookup)."""
    assert b"TAP_SCHEMA.tables" in request.content, request
    return httpx.Response(200, content=(FIXTURES / "heasarc_tables.0.body").read_bytes(),
                          headers={"content-type": "text/xml"})


def test_manifest_route_from_record_and_validation() -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router, TestClient(_app()) as client:
        router.route(host="testserver").pass_through()
        tap = router.route(host="heasarc.gsfc.nasa.gov").mock(side_effect=_heasarc_tap_schema)
        fresh = synthetic_record(stamp=datetime.now(UTC).isoformat())  # retrieved just now
        res = client.post("/api/v1/provenance/manifest", json={"record": fresh, "include_record": True})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["manifest"]["content_hash"] == P.content_hash(synthetic_record())
        assert body["record"]["target"]["ra"] == 150.0
        # The synthetic record queried HEASARC's nvss: its release was read live, at query time.
        assert tap.call_count == 1
        assert body["manifest"]["catalogs"]["nvss"]["release"] == "NRAO VLA Sky Survey Catalog (HEASARC table nvss)"
        assert body["manifest"]["catalogs"]["nvss"]["release_source"].startswith("live:")
        assert body["manifest"]["query_time"] == fresh["catalog_results"]["gaia_dr3"]["sources"][0]["provenance"]["retrieved_at"]
        # A record retrieved months ago: today's description may not be the release it saw.
        old = client.post("/api/v1/provenance/manifest", json={"record": synthetic_record()}).json()["manifest"]
        assert old["catalogs"]["nvss"]["release"] == P.CATALOG_RELEASES["nvss"]["release"]
        assert old["catalogs"]["nvss"]["release_source"].startswith(P.UNVERIFIED_RELEASE_PREFIX)
        assert old["query_time"] == "2026-01-01T00:00:00+00:00" and old["created_at"] > old["query_time"]
        assert client.post("/api/v1/provenance/manifest", json={}).status_code == 422
        assert client.post("/api/v1/provenance/manifest", json={"record": {"target": {}}}).status_code == 422
        assert client.post("/api/v1/provenance/manifest", json={"record": {"target": {"ra": 1, "dec": 1},
                                                                            "catalog_results": []}}).status_code == 422
        # Search fields are validated as /api/v1/search validates them (ra is normalized there, dec is not).
        assert client.post("/api/v1/provenance/manifest", json={"ra": 10, "dec": 100}).status_code == 422


def test_manifest_and_replay_routes_on_recorded_archives() -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        service = make_service(offline_client())
        with TestClient(_app(service)) as client:
            made = client.post("/api/v1/provenance/manifest", json={"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0})
            assert made.status_code == 200, made.text
            manifest = made.json()["manifest"]
            assert manifest["science"]["catalogs"]["gaia_dr3"]["sources"][0]["source_id"] == GAIA_3C273
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": manifest})
            assert replayed.status_code == 200, replayed.text
            body = replayed.json()
            assert body["identical"] is True and body["diff"]["equivalent"] is True
            assert body["new_manifest"]["content_hash"] == manifest["content_hash"]
            bad = client.post("/api/v1/provenance/replay", json={"manifest": {"schema": "nope"}})
            assert bad.status_code == 422


def test_replay_route_returns_502_when_archives_are_down() -> None:
    from fastapi.testclient import TestClient

    record = synthetic_record()
    record["provenance"]["catalogs_planned"] = ["gaia_dr3"]
    record["catalog_results"] = {"gaia_dr3": record["catalog_results"]["gaia_dr3"]}
    record["failures"] = []
    manifest = P.build_manifest(record).as_dict()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated outage", request=request)

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=timeout)
        with TestClient(_app(make_service(offline_client()))) as client:
            res = client.post("/api/v1/provenance/replay", json={"manifest": manifest})
    assert res.status_code == 502 and "every catalog failed" in res.json()["detail"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    P.register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_register_cli_adds_subcommands_with_handlers() -> None:
    parser = _parser()
    assert parser.parse_args(["replay", "m.json"]).handler is P.replay_command
    assert parser.parse_args(["cite", "--catalogs", "gaia_dr3"]).handler is P.cite_command
    assert parser.parse_args(["manifest", "--record", "r.json"]).handler is P.manifest_command


def test_cli_cite_writes_valid_bibtex(tmp_path: Path, capsys) -> None:
    out = tmp_path / "refs.bib"
    args = _parser().parse_args(["cite", "--catalogs", "gaia_dr3,allwise,lotss", "--bibtex", str(out)])
    assert args.handler(args) == 0
    printed = capsys.readouterr().out
    assert "European Space Agency (ESA) mission Gaia" in printed and "Infrared Science Archive" in printed
    keys = [e.key for e in P.parse_bibtex(out.read_text(encoding="utf-8"))]
    assert "2023A&A...674A...1G" in keys and "2026A&A...707A.198S" in keys and "IRSA_AllWISE_DOI" in keys
    bad = _parser().parse_args(["cite", "--catalogs", "nope"])
    assert bad.handler(bad) == 2
    empty = _parser().parse_args(["cite"])
    assert empty.handler(empty) == 2


def test_cli_manifest_from_record_and_cite_from_manifest(tmp_path: Path, capsys) -> None:
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps(synthetic_record()), encoding="utf-8")
    out = tmp_path / "manifest.json"
    args = _parser().parse_args(["manifest", "--record", str(rec), "-o", str(out)])
    with respx.mock(assert_all_called=True) as router:
        router.route(host="heasarc.gsfc.nasa.gov").mock(side_effect=_heasarc_tap_schema)
        assert args.handler(args) == 0
    assert "content hash: sha256:" in capsys.readouterr().out
    manifest = P.ProvenanceManifest.from_json(out.read_text(encoding="utf-8"))
    assert manifest.content_hash == P.content_hash(synthetic_record())
    assert P.catalogs_in(manifest) == ["gaia_dr3", "simbad"]  # the failed nvss is not cited
    cite = _parser().parse_args(["cite", "--from", str(out), "--json"])
    assert cite.handler(cite) == 0
    bundle = json.loads(capsys.readouterr().out)
    assert bundle["keys"] == ["gaia_dr3", "simbad"]
    assert {"2023A&A...674A...1G", "2000A&AS..143....9W"} <= {r["key"] for r in bundle["references"]}


def test_cli_replay_exit_codes(tmp_path: Path, capsys) -> None:
    import asyncio

    record, service = asyncio.run(_crossmatch_3c273())
    path = tmp_path / "manifest.json"
    path.write_text(P.build_manifest(record, registry=service.registry).to_json(), encoding="utf-8")
    new_path = tmp_path / "new.json"
    args = _parser().parse_args(["replay", str(path), "--out", str(new_path)])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        assert args.handler(args) == 0
    assert "Identical: yes" in capsys.readouterr().out
    assert P.ProvenanceManifest.from_json(new_path.read_text(encoding="utf-8")).content_hash == \
        json.loads(path.read_text(encoding="utf-8"))["content_hash"]
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(_modified_exchanges()))
        assert args.handler(args) == 1
    assert "Identical: no" in capsys.readouterr().out
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": "x"}', encoding="utf-8")
    bad = _parser().parse_args(["replay", str(broken)])
    assert bad.handler(bad) == 2


def test_validate_target_helper_used_by_manifests_accepts_record_targets() -> None:
    target = validate_target(RA_3C273, DEC_3C273)
    req = P.reconstruct_request(CatalogRegistry().get("gaia_dr3"), target, 10.0)
    assert req["http_method"] == "POST" and req["parameters"]["FORMAT"] == "json"
    assert "CIRCLE('ICRS', 187.277915400, 2.052388300, 0.0027777778)" in req["query"]
