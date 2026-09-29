"""Regression tests for the provenance review (round 1).

Covers: name-resolved searches (resolver proper motion) manifest and replay exactly;
full-precision input targets; JavaScript number semantics (integers beyond 2**53);
match separation/confidence diffs; sigma-scaled "moved" threshold and angular RA/Dec
columns; tolerant comparison of the proper-motion origin; malformed manifests (422 /
exit code 2 before any network request); a pinned content hash; request-change and
cache-bypass detection; live release descriptions; citations texts; CLI encodings.

Archive traffic is replayed from recorded fixtures: tests/fixtures/barnard_*,
tests/fixtures/3c273, tests/fixtures/provenance/m87_resolver (M87 as a name search),
tests/fixtures/provenance/heasarc_tables (HEASARC TAP_SCHEMA.tables) and the ADS
link-gateway answers in tests/fixtures/provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import EPOCH_TARGETS, Exchange, load_exchanges, replay_side_effect, request_signature
from helpers import make_service, offline_client
from test_provenance import (
    DEC_3C273,
    GAIA_3C273,
    RA_3C273,
    _ads_replay,
    _app,
    _crossmatch_3c273,
    _modified_exchanges,
    _parser,
    _source,
    synthetic_record,
)

import main
import provenance as P
from models import CatalogRegistry
from providers import SesameResolver

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROV = FIXTURES / "provenance"
M87_FIXTURE = "provenance/m87_resolver"
SDSS_3C273_OBJID = "1237651735760142397"


@pytest.fixture(autouse=True)
def _fresh_release_cache():
    P._RELEASE_CACHE.clear()
    yield
    P._RELEASE_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixtureResolver:
    """A SesameResolver stand-in answering from a recorded Sesame XML file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def resolve(self, name: str):
        return SesameResolver.parse_response(name, self.path.read_text(encoding="utf-8"))


async def _name_search(name: str, sesame_xml: Path, exchanges: list[Exchange]):
    """``main.search_object`` (the CLI ``search --name`` / notebook path) on recorded archives."""
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        return await main.search_object(name, radius_arcsec=10.0, resolver=_FixtureResolver(sesame_xml))


async def _replay(manifest: Any, exchanges: list[Exchange], *, registry: CatalogRegistry | None = None, handler=None,
                  **kwargs: Any) -> P.ReplayResult:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=handler or replay_side_effect(exchanges))
        async with offline_client() as client:
            return await P.replay_manifest(manifest, make_service(client, registry=registry), **kwargs)


def _js_number(value: Any) -> Any:
    """What Python reads back after ``JSON.stringify(JSON.parse(text))`` in a browser.

    Every JSON number becomes an IEEE-754 double. ECMAScript writes a whole value below
    1e21 without fraction or exponent (2000.0 -> 2000, -0 -> 0, 3700386905605055360 ->
    3700386905605055500: shortest round-trip digits padded with zeros), which Python
    reads as an int; anything else in shortest round-trip form, read as a float.
    (Round 1 returned ``float(value)`` for floats, so 2000.0 stayed 2000.0 and the
    simulation hid the bug; test_provenance_round2 also runs the real ``node``.)
    """
    if isinstance(value, dict):
        return {k: _js_number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_number(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    double = float(value)
    if double.is_integer() and abs(double) < 1e21:
        return int(Decimal(repr(double)))
    return double


def _js_round_trip(value: Any) -> Any:
    from test_provenance_round2 import js_round_trip  # real node when installed

    return js_round_trip(value) if shutil.which("node") else _js_number(json.loads(json.dumps(value)))


def _assert_requests_as_sent(manifest: P.ProvenanceManifest, exchanges: list[Exchange]) -> None:
    """Recorded requests equal their reconstruction; requests taken from the result metadata
    (empty TAP/SDSS results: the ADQL/SQL the provider recorded) and reconstructed ones
    equal what was sent."""
    for name, req in manifest.catalogs.items():
        assert req.parameters, name
        if req.request_source == "recorded":
            assert req.request_verified is True, f"{name}: recorded request differs from the manifest's reconstruction"
        else:
            assert req.request_source in ("reconstructed", "recorded (result metadata)"), name
            if req.request_source == "recorded (result metadata)":
                assert req.request_verified is True, f"{name}: recorded ADQL differs from the registry's"
            sent = [e for e in exchanges if e.catalog == name][-1]
            assert sent.url_base == req.endpoint, name
            assert request_signature("", str(httpx.QueryParams(req.parameters))) == sent.signature, name


# ---------------------------------------------------------------------------
# Proper-motion origin (critical: name-resolved basic searches)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("source", "given"), [("input", True), ("resolver", True), ("adopted", False),
                                               ("extragalactic", False)])
def test_input_target_keeps_only_proper_motions_given_with_the_query(source: str, given: bool) -> None:
    rec = synthetic_record()
    rec["target"].update(epoch=2000.0, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394)
    rec["provenance"]["target_proper_motion"] = {"pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394, "source": source}
    target = P.input_target(rec)
    assert target.epoch == 2000.0
    assert (target.proper_motion == (-801.551, 10362.394)) is given
    assert (target.proper_motion is None) is not given
    assert P.given_pm_source(rec) == (source if given else None)
    assert P.build_manifest(rec).query["pm_source"] == (source if given else None)


async def test_barnards_star_name_search_manifests_the_requests_sent_and_replays_identically() -> None:
    """main.search_object('Barnard's star'): Sesame J2000 position + proper motion (source 'resolver')."""
    exchanges = load_exchanges("barnard_j2000_pm")
    record = await _name_search("Barnard's star", FIXTURES / "sesame" / "barnards_star.xml", exchanges)
    assert not record.failures
    pm = record.provenance["target_proper_motion"]
    assert {k: pm[k] for k in ("pm_ra_masyr", "pm_dec_masyr", "source")} == {
        "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394, "source": "resolver"}
    # The star really is found: Gaia DR3 4472832130942575872 moves 10.39"/yr.
    nearest = min((m for m in record.provenance["matches"] if m["catalog"] == "gaia_dr3"),
                  key=lambda m: m["separation_arcsec"])
    assert nearest["source_id"] == "4472832130942575872"
    assert nearest["separation_arcsec"] < 0.1  # followed to Gaia's J2016 with the resolver motion
    manifest = P.build_manifest(record, registry=CatalogRegistry())
    assert {k: manifest.input_target[k] for k in ("ra", "dec", "frame", "epoch", "pm_ra_masyr", "pm_dec_masyr")} == {
        "ra": 269.45207696, "dec": 4.69336497, "frame": "icrs", "epoch": 2000.0, "pm_ra_masyr": -801.551,
        "pm_dec_masyr": 10362.394}
    assert manifest.query["pm_source"] == "resolver" and manifest.query["resolver"]
    _assert_requests_as_sent(manifest, exchanges)
    result = await _replay(json.loads(manifest.to_json()), exchanges)
    assert result.identical, result.explanation
    assert result.diff["requests"] == {} and result.diff["other"] == {}
    assert result.new_manifest.science["target_proper_motion"]["source"] == "resolver"
    assert result.new_manifest.query_hash == manifest.query_hash
    assert result.explanation[0].startswith("Content hash identical")


async def test_galaxy_name_search_keeps_the_resolver_zero_proper_motion() -> None:
    """main.search_object('M87'): extragalactic, so the resolver gives pm = (0, 0) with source 'resolver'.

    Before the fix the manifest dropped it, and replay adopted 'extragalactic' instead,
    re-sending different epoch cones.
    """
    exchanges = load_exchanges(M87_FIXTURE)
    record = await _name_search("M87", PROV / "m87_resolver" / "sesame.xml", exchanges)
    assert not record.failures
    pm = record.provenance["target_proper_motion"]
    assert pm["source"] == "resolver" and (pm["pm_ra_masyr"], pm["pm_dec_masyr"]) == (0.0, 0.0)
    assert record.target["epoch"] == 2000.0
    # M87 truths: SIMBAD identifies it (an AGN, Virgo A); the Chandra catalog has its core/jet sources.
    simbad_ids = [s.source_id for s in record.catalog_results["simbad"]["sources"]]
    assert "M 87" in simbad_ids and "NAME M87 Jet" in simbad_ids
    assert record.catalog_results["chandra"]["row_count"] >= 1
    manifest = P.build_manifest(record, registry=CatalogRegistry())
    assert manifest.input_target["pm_ra_masyr"] == 0.0 and manifest.input_target["pm_dec_masyr"] == 0.0
    assert manifest.query["pm_source"] == "resolver"
    _assert_requests_as_sent(manifest, exchanges)
    result = await _replay(json.loads(manifest.to_json()), exchanges)
    assert result.identical, result.explanation
    assert result.diff["requests"] == {}
    assert result.new_manifest.science["target_proper_motion"]["source"] == "resolver"


def test_manifest_route_name_search_records_the_resolver_origin_and_replays_identically() -> None:
    """POST /provenance/manifest {name}: _run_search labels the Sesame motion 'resolver'."""
    from fastapi.testclient import TestClient

    sesame_xml = (FIXTURES / "sesame" / "barnards_star.xml").read_text(encoding="utf-8")
    exchanges = load_exchanges("barnard_j2000_pm")
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=httpx.Response(200, text=sesame_xml, headers={"content-type": "text/xml"}))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(_app(make_service(offline_client()))) as client:
            made = client.post("/api/v1/provenance/manifest", json={"name": "Barnard's star", "radius_arcsec": 10.0})
            assert made.status_code == 200, made.text
            manifest = made.json()["manifest"]
            assert manifest["query"]["pm_source"] == "resolver" and manifest["query"]["resolver"]
            assert manifest["input_target"]["pm_dec_masyr"] == 10362.394
            assert manifest["resolved_object"]["ra_deg"] == 269.45207696
            assert all(c["request_verified"] is not False for c in manifest["catalogs"].values())
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": manifest})
            assert replayed.status_code == 200, replayed.text
            body = replayed.json()
    assert body["identical"] is True, body["explanation"]
    assert body["diff"]["requests"] == {} and body["diff"]["other"] == {}


async def test_adopted_proper_motion_is_readopted_on_replay() -> None:
    """No proper motion given (barnard_j2000): Gaia's is adopted in both runs."""
    spec = EPOCH_TARGETS["barnard_j2000"]
    exchanges = load_exchanges("barnard_j2000")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(spec["ra"], spec["dec"], radius_arcsec=10.0, epoch=spec["epoch"])
    assert record.provenance["target_proper_motion"]["source"] == "adopted"
    manifest = P.build_manifest(record, registry=service.registry)
    assert manifest.input_target["pm_ra_masyr"] is None and manifest.query["pm_source"] is None
    _assert_requests_as_sent(manifest, exchanges)
    result = await _replay(manifest.as_dict(), exchanges)
    assert result.identical, result.explanation


# ---------------------------------------------------------------------------
# Full-precision input (major: 12-digit rounding of the replay position)
# ---------------------------------------------------------------------------


async def test_gaia_epoch_position_with_17_digits_replays_identically() -> None:
    spec = EPOCH_TARGETS["barnard_gaia2016"]  # Gaia DR3 position at J2016.0, 17 significant digits
    exchanges = load_exchanges("barnard_gaia2016")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(spec["ra"], spec["dec"], radius_arcsec=10.0, epoch=spec["epoch"])
    manifest = P.build_manifest(record, registry=service.registry)
    data = json.loads(manifest.to_json())
    assert data["input_target"]["ra"] == spec["ra"] and data["input_target"]["dec"] == spec["dec"]
    result = await _replay(data, exchanges)
    assert result.identical, result.explanation
    assert result.diff["other"] == {} and result.explanation[0].startswith("Content hash identical")


async def test_high_precision_coordinates_replay_identically() -> None:
    ra, dec = 187.2779154000123, 2.0523883000077  # e.g. a position copied from a catalog row
    exchanges = load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(ra, dec, radius_arcsec=10.0)
    manifest = P.build_manifest(record, registry=service.registry)
    data = json.loads(manifest.to_json())
    assert (data["input_target"]["ra"], data["input_target"]["dec"]) == (ra, dec)
    assert P.ProvenanceManifest.from_dict(data).input_target["ra"] == ra
    result = await _replay(data, exchanges)
    assert result.identical, result.explanation
    assert not any("below the diff tolerances" in line for line in result.explanation)


def test_other_changes_use_the_numeric_tolerances() -> None:
    old = P.science_payload(synthetic_record())
    rec = synthetic_record()
    rec["provenance"]["target_proper_motion"] = {"pm_ra_masyr": -801.55, "pm_dec_masyr": 10362.39, "source": "adopted",
                                                 "catalog": "gaia_dr3", "source_id": "111", "separation_arcsec": 0.0}
    base = P.science_payload(rec)
    rec["provenance"]["target_proper_motion"]["separation_arcsec"] = 1.57e-06  # 1.6 micro-arcsec
    noise = P.science_payload(rec)
    assert P.diff_science(base, noise)["other"] == {}
    rec["provenance"]["target_proper_motion"]["source"] = "resolver"
    changed = P.diff_science(base, P.science_payload(rec))
    assert set(changed["other"]) == {"target_proper_motion"} and not changed["equivalent"]
    assert set(P.diff_science(old, base)["other"]) == {"target_proper_motion"}


# ---------------------------------------------------------------------------
# JavaScript number semantics (major: integers beyond 2**53)
# ---------------------------------------------------------------------------


def test_canonical_form_writes_integers_beyond_2_53_as_strings() -> None:
    assert P.canonicalize(3700386905605055360) == "3700386905605055360"
    assert P.canonicalize(2**53 - 1) == 2**53 - 1 and P.canonicalize(-(2**53)) == str(-(2**53))
    # JSON.stringify(JSON.parse("3700386905605055360")) == "3700386905605055500" (shortest round trip)
    assert P.js_number_value(3700386905605055360) == P.js_number_value(3700386905605055500) == "3700386905605055488"
    assert P.js_number_value(3700386905605055000) != P.js_number_value(3700386905605055360)  # a different double
    assert P.js_number_value("3700386905605055360") == "3700386905605055488"
    assert P.js_number_value("123") == "123" and P.js_number_value(True) is True


async def test_records_and_manifests_survive_javascript_number_semantics() -> None:
    record, service = await _crossmatch_3c273()
    server = record.as_dict()
    browser = _js_round_trip(server)
    gaia_row = browser["catalog_results"]["gaia_dr3"]["sources"][0]
    assert gaia_row["data"]["source_id"] != int(GAIA_3C273)  # precision really lost in the browser
    assert gaia_row["source_id"] == GAIA_3C273  # the row identity is an exact string
    assert P.content_hash(browser) == P.content_hash(server)
    m_server = P.build_manifest(server, registry=service.registry)
    m_browser = P.build_manifest(browser, registry=service.registry)
    assert m_browser.content_hash == m_server.content_hash
    # The manifest itself is JavaScript-safe: big integers are decimal strings.
    data = json.loads(m_server.to_json())
    sci = data["science"]["catalogs"]
    assert sci["gaia_dr3"]["sources"][0]["data"]["source_id"] == GAIA_3C273
    assert sci["sdss"]["sources"][0]["data"]["objid"] == SDSS_3C273_OBJID  # column names case-folded
    assert _js_round_trip(data) == data
    # UI flow: browser record -> manifest -> browser -> replay (no false tamper alarm).
    replay_input = _js_round_trip(json.loads(m_browser.to_json()))
    assert P.verify_manifest(replay_input) == {"content_hash_ok": True, "query_hash_ok": True, "row_hashes_ok": True}
    result = await _replay(replay_input, load_exchanges("3c273"))
    assert result.identical, result.explanation
    assert not any("integrity check failed" in line for line in result.explanation)


def test_ui_flow_through_the_routes_with_javascript_numbers() -> None:
    from fastapi.testclient import TestClient

    record = json.loads(json.dumps(__import__("asyncio").run(_crossmatch_3c273())[0].as_dict()))
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        service = make_service(offline_client())
        with TestClient(_app(service)) as client:
            made = client.post("/api/v1/provenance/manifest", json={"record": _js_round_trip(record),
                                                                    "live_release_lookup": False})
            assert made.status_code == 200, made.text
            manifest = made.json()["manifest"]
            assert manifest["content_hash"] == P.content_hash(record)
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": _js_round_trip(manifest)})
            assert replayed.status_code == 200, replayed.text
            body = replayed.json()
    assert body["identical"] is True and body["equivalent_within_tolerance"] is True
    assert not any("integrity check failed" in line for line in body["explanation"])


# ---------------------------------------------------------------------------
# Pinned hash format
# ---------------------------------------------------------------------------

# If one of these changes, every issued manifest's content hash changes: bump
# SCIENCE_SCHEMA (and MANIFEST_SCHEMA) and update the constants deliberately.
# science/3 (round 2): whole-valued floats as integers, target parallax and origin.
GOLDEN_SYNTHETIC_HASH = "sha256:aa1c775658dac00b3e409f6992d64722e3c68b0bfe0e4a74814d1050ef80c0b9"
GOLDEN_3C273_HASH = "sha256:2544a79ae20707273284a552893631da0197e7ebea7de6b4b2f6cbf69d843713"


def test_content_hash_format_is_pinned() -> None:
    assert P.SCIENCE_SCHEMA == "astrosearch.provenance.science/3"
    assert P.MANIFEST_SCHEMA == "astrosearch.provenance.manifest/3"
    assert P.content_hash(synthetic_record()) == GOLDEN_SYNTHETIC_HASH
    # A real 3C 273 record (all enabled catalogs, 10", recorded 2026) frozen as JSON.
    record = json.loads((PROV / "record_3c273.json").read_text(encoding="utf-8"))
    assert P.content_hash(record) == GOLDEN_3C273_HASH
    assert P.content_hash(_js_round_trip(record)) == GOLDEN_3C273_HASH
    reversed_rows = copy.deepcopy(record)
    for res in reversed_rows["catalog_results"].values():
        res["sources"] = list(reversed(res.get("sources") or []))
    assert P.content_hash(reversed_rows) == GOLDEN_3C273_HASH  # archive row order is not science


# ---------------------------------------------------------------------------
# Diff: matches, sigma-scaled moves, angular columns
# ---------------------------------------------------------------------------


def test_archive_column_name_case_is_not_science() -> None:
    """MAST answered 'objName' and, a minute later, 'ObjName' for the same Pan-STARRS rows."""
    a = synthetic_record()
    b = synthetic_record()
    a["catalog_results"]["gaia_dr3"]["sources"][0]["data"]["objName"] = "PSO J150.0000+02.0000"
    b["catalog_results"]["gaia_dr3"]["sources"][0]["data"]["ObjName"] = "PSO J150.0000+02.0000"
    assert P.content_hash(a) == P.content_hash(b)
    assert P.diff_science(P.science_payload(a), P.science_payload(b))["catalogs"] == {}
    # Columns that differ only in case within one row are both kept (never merged).
    c = synthetic_record()
    c["catalog_results"]["gaia_dr3"]["sources"][0]["data"].update(Jmag=10.0, jmag=11.0)
    row = P.science_payload(c)["catalogs"]["gaia_dr3"]["sources"][0]
    assert row["data"]["Jmag"] == 10.0 and row["data"]["jmag"] == 11.0


def test_diff_reports_match_separation_and_confidence_changes() -> None:
    old = P.science_payload(synthetic_record())
    rec = synthetic_record()
    rec["provenance"]["matches"][0].update(confidence=0.10, separation_arcsec=4.5)
    new = P.science_payload(rec)
    assert P.content_hash(old) != P.content_hash(new)
    diff = P.diff_science(old, new)
    assert diff["equivalent"] is False and diff["totals"]["matches_changed"] == 1
    assert diff["matches"]["changed"] == [{"match": "gaia_dr3:111", "separation_arcsec": {"old": 0.00036, "new": 4.5},
                                           "confidence": {"old": 0.99, "new": 0.1}}]
    m_old = P.build_manifest(synthetic_record())
    m_new = P.build_manifest(rec)
    notes = P._explain(m_old, m_new, diff, {}, [])
    assert any("changed separation or confidence" in n for n in notes)
    assert not any("below the diff tolerances" in n for n in notes)
    # Float noise in the scores stays equivalent.
    rec = synthetic_record()
    rec["provenance"]["matches"][0].update(confidence=0.99 + 1e-9, separation_arcsec=0.00036 + 1e-8)
    assert P.diff_science(old, P.science_payload(rec))["equivalent"] is True


def _row_at(ra: float, shift_mas: float, sigma_arcsec: float) -> dict[str, Any]:
    rec = synthetic_record()
    src = rec["catalog_results"]["gaia_dr3"]["sources"][0]
    dec = src["dec"]
    src["ra"] = ra + shift_mas / 3.6e6 / math.cos(math.radians(dec))
    src["positional_error_arcsec"] = sigma_arcsec
    src["data"]["ra"] = src["ra"]  # the archive's own RA column moves with the row
    src["data"]["dec"] = dec
    return rec


@pytest.mark.parametrize("ra", [0.01, 180.0, 350.0])
def test_moves_are_judged_against_the_rows_positional_error_at_any_ra(ra: float) -> None:
    # A Gaia-like row (sigma 0.02 mas) moving 0.5 mas moved by 25 sigma: reported.
    old = P.science_payload(_row_at(ra, 0.0, 2e-5))
    new = P.science_payload(_row_at(ra, 0.5, 2e-5))
    diff = P.diff_science(old, new)
    moved = diff["catalogs"]["gaia_dr3"]["moved"]
    assert [m["source_id"] for m in moved] == ["111"] and moved[0]["separation_arcsec"] == pytest.approx(5e-4, rel=1e-3)
    assert moved[0]["threshold_arcsec"] == pytest.approx(2e-6)
    # A row with a 50 mas error moving 0.5 mas is within tolerance -- including its RA
    # column, whatever the RA (relative tolerances used to flag RA ~ 0 only).
    old = P.science_payload(_row_at(ra, 0.0, 0.05))
    new = P.science_payload(_row_at(ra, 0.5, 0.05))
    assert P.content_hash(old) != P.content_hash(new)
    assert P.diff_science(old, new)["equivalent"] is True


def test_explanation_does_not_blame_the_archive_for_rows_missing_after_an_original_failure() -> None:
    old_rec = synthetic_record()
    new_rec = synthetic_record()
    new_rec["catalog_results"]["nvss"] = {"sources": [_source("nvss", "NVSS J1", 150.0, 2.0, {"S1.4": 12.0})],
                                          "status": "success", "row_count": 1, "truncated": False}
    new_rec["failures"] = []
    old, new = P.build_manifest(old_rec), P.build_manifest(new_rec)
    diff = P.diff_science(old.science, new.science)
    assert diff["catalogs"]["nvss"]["added"][0]["source_id"] == "NVSS J1"
    notes = P._explain(old, new, diff, {}, [])
    assert any(n.startswith("nvss: failed in the original run (QueryTimeoutError) but answered now") for n in notes)
    assert not any(n.startswith("nvss: 1 added") for n in notes)


# ---------------------------------------------------------------------------
# Malformed manifests: ManifestError / 422 / exit 2 before any network request
# ---------------------------------------------------------------------------


def _manifest_dict() -> dict[str, Any]:
    return P.build_manifest(synthetic_record()).as_dict()


MALFORMED: list[tuple[str, Any]] = [
    ("catalog entry is a list", lambda d: d["catalogs"].__setitem__("gaia_dr3", [])),
    ("catalog entry is a string", lambda d: d["catalogs"].__setitem__("gaia_dr3", "x")),
    ("catalog row_count is text", lambda d: d["catalogs"]["gaia_dr3"].__setitem__("row_count", "many")),
    ("catalog parameters is a list", lambda d: d["catalogs"]["gaia_dr3"].__setitem__("parameters", [1])),
    ("software is a string", lambda d: d.__setitem__("software", "astrosearch")),
    ("software is null", lambda d: d.__setitem__("software", None)),
    ("input_target is empty", lambda d: d.__setitem__("input_target", {})),
    ("input_target is a string", lambda d: d.__setitem__("input_target", "3C 273")),
    ("input_target ra is text", lambda d: d["input_target"].__setitem__("ra", "12h")),
    ("input_target ra out of range", lambda d: d["input_target"].__setitem__("ra", 400.0)),
    ("input_target dec is NaN", lambda d: d["input_target"].__setitem__("dec", float("nan"))),
    ("query is a string", lambda d: d.__setitem__("query", "cone")),
    ("query is null", lambda d: d.__setitem__("query", None)),
    ("query mode unknown", lambda d: d["query"].__setitem__("mode", "magic")),
    ("advanced query invalid", lambda d: d["query"].update(mode="advanced", advanced_query={"target": 5})),
    # Round 2: type errors inside a structurally valid advanced query (were HTTP 500).
    *[(f"advanced query {key}={value!r}",
       lambda d, key=key, value=value: d["query"].update(mode="advanced", advanced_query={
           "target": {"ra": 150.0, "dec": 2.0}, "radius_arcsec": 5.0, key: value}))
      for key, value in (("time_period", "2020"), ("spatial_constraints", "x"),
                         ("spatial_constraints", {"exclude_polygons": 5}), ("spatial_constraints", {"radius_zones": 3}),
                         ("object_types", 5), ("metadata", [1]))],
    ("resolved_object is a list", lambda d: d.__setitem__("resolved_object", [1, 2])),
    ("created_at is a number", lambda d: d.__setitem__("created_at", 5)),
    ("science catalogs is a list", lambda d: d["science"].__setitem__("catalogs", [])),
    ("science sources is a string", lambda d: d["science"]["catalogs"]["gaia_dr3"].__setitem__("sources", "rows")),
    ("science source is an int", lambda d: d["science"]["catalogs"]["gaia_dr3"]["sources"].append(5)),
    ("science matches is a string", lambda d: d["science"].__setitem__("matches", "none")),
    ("science groups are ints", lambda d: d["science"].__setitem__("groups", [1, 2])),
    ("science target is a list", lambda d: d["science"].__setitem__("target", [150.0, 2.0])),
]


@pytest.mark.parametrize(("label", "mutate"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_manifests_raise_manifest_error(label: str, mutate) -> None:
    data = _manifest_dict()
    mutate(data)
    with pytest.raises(P.ManifestError):
        P.ProvenanceManifest.from_dict(data)


def test_replay_route_rejects_malformed_manifests_with_422_before_querying_archives() -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        archive = router.route()
        archive.mock(side_effect=AssertionError("an archive was queried for a malformed manifest"))
        with TestClient(_app(make_service(offline_client())), raise_server_exceptions=False) as client:
            for label, mutate in MALFORMED:
                data = json.loads(json.dumps(_manifest_dict()))
                mutate(data)
                res = client.post("/api/v1/provenance/replay", content=json.dumps({"manifest": data}),
                                  headers={"content-type": "application/json"})
                assert res.status_code == 422, (label, res.status_code, res.text)
            # Catalogs unknown to the registry: a bad request, not an upstream outage.
            data = _manifest_dict()
            data["catalogs"] = {f"nope_{k}": {**v, "catalog": f"nope_{k}"} for k, v in data["catalogs"].items()}
            res = client.post("/api/v1/provenance/replay", json={"manifest": data})
            assert res.status_code == 422 and "none of the manifest's catalogs" in res.json()["detail"]
            bad_tol = client.post("/api/v1/provenance/replay", json={"manifest": _manifest_dict(), "sigma_fraction": -1})
            assert bad_tol.status_code == 422
        assert archive.call_count == 0


def test_manifest_route_rejects_non_finite_radius_and_maps_resolver_errors() -> None:
    from fastapi.testclient import TestClient

    record = synthetic_record()
    record["provenance"]["query_radius_arcsec"] = float("inf")
    with pytest.raises(P.ManifestError, match="finite"):
        P.build_manifest(record)
    not_found = ('<?xml version="1.0" encoding="UTF-8"?><Sesame><Target option="SNV"><name>NoSuchObjectXYZ123</name>'
                 '<INFO>*** Nothing found ***</INFO></Target></Sesame>')
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        sesame = router.route(host="cds.unistra.fr")
        with TestClient(_app(make_service(offline_client())), raise_server_exceptions=False) as client:
            res = client.post("/api/v1/provenance/manifest", content=json.dumps({"record": record}),
                              headers={"content-type": "application/json"})
            assert res.status_code == 422, res.text
            sesame.mock(return_value=httpx.Response(200, text=not_found, headers={"content-type": "text/xml"}))
            res = client.post("/api/v1/provenance/manifest", json={"name": "NoSuchObjectXYZ123"})
            assert res.status_code == 422 and "No coordinates found" in res.json()["detail"]
            sesame.mock(return_value=httpx.Response(200, text="<html>maintenance</html", headers={"content-type": "text/html"}))
            res = client.post("/api/v1/provenance/manifest", json={"name": "M87"})
            assert res.status_code == 502 and "could not be parsed" in res.json()["detail"]
            sesame.mock(side_effect=httpx.ConnectError("down"))
            res = client.post("/api/v1/provenance/manifest", json={"name": "M87"})
            assert res.status_code == 502


# ---------------------------------------------------------------------------
# Replay: request changes, cache bypass
# ---------------------------------------------------------------------------


async def test_a_changed_registry_definition_shows_up_as_a_request_change() -> None:
    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    registry = CatalogRegistry()
    registry._catalogs["gaia_dr3"] = replace(registry._catalogs["gaia_dr3"], max_rows=100)  # TOP 201 -> TOP 101
    exchanges = load_exchanges("3c273")
    strict = replay_side_effect(exchanges)
    loose = replay_side_effect([e for e in exchanges if e.catalog == "gaia_dr3"], strict=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return loose(request) if request.url.host == "gea.esac.esa.int" else strict(request)

    result = await _replay(manifest, exchanges, registry=registry, handler=handler)
    change = result.diff["requests"]["gaia_dr3"]
    assert set(change) == {"query", "parameters"}
    assert change["query"]["old"].startswith("SELECT TOP 201 ") and change["query"]["new"].startswith("SELECT TOP 101 ")
    assert set(result.diff["requests"]) == {"gaia_dr3"}
    assert any(line.startswith("gaia_dr3: the request sent differs from the manifest") for line in result.explanation)


async def test_replay_requeries_the_archives_unless_use_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    record, _ = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=CatalogRegistry())
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    async with offline_client() as client:
        stale = make_service(client)
        with respx.mock(assert_all_called=False) as router:  # the service's cache now holds changed answers
            router.route().mock(side_effect=replay_side_effect(_modified_exchanges()))
            await stale.crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
            fresh = await P.replay_manifest(manifest, stale, live_release_lookup=False)
            cached = await P.replay_manifest(manifest, stale, use_cache=True, live_release_lookup=False)
    assert fresh.identical, fresh.explanation
    assert not cached.identical
    assert cached.diff["catalogs"]["gaia_dr3"]["moved"][0]["source_id"] == GAIA_3C273


# ---------------------------------------------------------------------------
# Data releases read live from HEASARC TAP_SCHEMA.tables
# ---------------------------------------------------------------------------


def _heasarc_tables_body(replace_text: tuple[str, str] | None = None) -> bytes:
    body = (PROV / "heasarc_tables.0.body").read_bytes()
    if replace_text is None:
        return body
    import io

    from astropy.io.votable import parse
    from astropy.io.votable.tree import VOTableFile

    table = parse(io.BytesIO(body)).get_first_table().to_table()
    table["description"] = [str(d).replace(*replace_text) for d in table["description"]]
    out = io.BytesIO()
    VOTableFile.from_table(table).to_xml(out)
    return out.getvalue()


def _with_tap_schema(exchanges: list[Exchange], body: bytes | None = None, status: int = 200):
    strict = replay_side_effect(exchanges)

    def handler(request: httpx.Request) -> httpx.Response:
        if b"TAP_SCHEMA.tables" in (request.content or b""):
            return httpx.Response(status, content=body if body is not None else _heasarc_tables_body(),
                                  headers={"content-type": "text/xml"})
        return strict(request)

    return handler


async def test_manifest_records_the_release_heasarc_serves_and_replay_attributes_a_release_change() -> None:
    record, service = await _crossmatch_3c273()
    exchanges = load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_with_tap_schema(exchanges))
        async with offline_client() as client:
            releases = await P.live_releases(client, service.registry, list(service.registry.enabled_catalogs()))
    manifest = P.build_manifest(record, registry=service.registry, releases=releases)
    xmm = manifest.catalogs["xmm"]
    assert xmm.release == "XMM-Newton Serendipitous Source Catalog: 5XMM-DR15 Version (HEASARC table xmmssc)"
    assert xmm.release_source.startswith("live: https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync TAP_SCHEMA.tables")
    assert manifest.catalogs["chandra"].release.startswith("Chandra Source Catalog, v2.1.1")
    assert manifest.catalogs["first"].release.startswith("Faint Images of the Radio Sky")  # listed as '"first"'
    assert manifest.catalogs["gaia_dr3"].release_source == "label"  # not a HEASARC table
    # Same release on replay: nothing to report.
    same = await _replay(manifest, exchanges, handler=_with_tap_schema(exchanges))
    assert same.identical and same.diff["releases"] == {}
    # HEASARC moved xmmssc to a new data release in place.
    P._RELEASE_CACHE.clear()
    newer = _heasarc_tables_body(("5XMM-DR15", "5XMM-DR16"))
    moved_on = await _replay(manifest, exchanges, handler=_with_tap_schema(exchanges, newer))
    assert moved_on.diff["releases"] == {"xmm": {
        "old": "XMM-Newton Serendipitous Source Catalog: 5XMM-DR15 Version (HEASARC table xmmssc)",
        "new": "XMM-Newton Serendipitous Source Catalog: 5XMM-DR16 Version (HEASARC table xmmssc)"}}
    assert any(line.startswith("xmm: the archive's data release changed") for line in moved_on.explanation)


async def test_release_lookup_failure_keeps_the_label_and_says_why() -> None:
    registry = CatalogRegistry()
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(503))
        async with offline_client() as client:
            releases = await P.live_releases(client, registry, ["xmm", "gaia_dr3"])
    assert releases == {"xmm": {"source": "unavailable: HTTPStatusError"}}
    record = synthetic_record()
    record["provenance"]["catalogs_planned"] = ["xmm"]
    record["catalog_results"] = {"xmm": {"sources": [], "status": "empty", "row_count": 0, "truncated": False}}
    record["failures"] = []
    xmm = P.build_manifest(record, registry=registry, releases=releases).catalogs["xmm"]
    assert xmm.release == P.CATALOG_RELEASES["xmm"]["release"]
    assert xmm.release_source == "label (unavailable: HTTPStatusError)"


# ---------------------------------------------------------------------------
# Software environment
# ---------------------------------------------------------------------------


def test_git_state_counts_untracked_python_files_as_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"})

    git("init", "-q")
    (tmp_path / "core.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "core.py")
    git("commit", "-q", "-m", "init")
    clean = P.git_state_at(tmp_path)
    assert clean["commit"] and clean["dirty"] is False and clean["untracked_python_files"] == 0
    (tmp_path / "notes.txt").write_text("not code\n", encoding="utf-8")
    assert P.git_state_at(tmp_path)["dirty"] is False
    (tmp_path / "feature.py").write_text("y = 2\n", encoding="utf-8")
    state = P.git_state_at(tmp_path)
    assert state["dirty"] is True and state["untracked_python_files"] == 1
    assert P.git_state_at(tmp_path / "missing-dir")["commit"] is None


async def test_software_info_is_cached_and_identifies_the_code() -> None:
    info = P.software_info()
    assert info == await P.software_info_async()
    digests = info["code_sha256"]
    assert set(digests) == set(P.CORE_MODULES) and all(len(v or "") == 16 for v in digests.values())
    info["git"] = "mutated"
    assert P.software_info()["git"] != "mutated"  # callers get a copy


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_official_acknowledgements_are_complete() -> None:
    ps1 = P.CITATION_ENTRIES["panstarrs_dr2"].acknowledgement
    assert ps1.startswith("The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive")
    assert "Grant No. NNX08AR22G" in ps1 and "AST-1238877" in ps1 and ps1.endswith("Gordon and Betty Moore Foundation.")
    sdss = P.CITATION_ENTRIES["sdss"].acknowledgement
    for phase in ("Funding for the SDSS and SDSS-II has been provided", "Funding for SDSS-III has been provided",
                  "Funding for the Sloan Digital Sky Survey IV", "Funding for the Sloan Digital Sky Survey V"):
        assert phase in sdss, phase
    assert "Center for High-Performance Computing at the University of Utah" in sdss
    assert "Universidad Nacional Autónoma de México" in sdss
    for key in ("lotss", "lotss_dr2"):
        entry = P.CITATION_ENTRIES[key]
        assert entry.acknowledgement.startswith("LOFAR data products were provided by the LOFAR Surveys Key Science "
                                                "project (LSKSP; https://lofar-surveys.org/)")
        assert "LOFAR (van Haarlem et al. 2013) is the Low Frequency Array" in entry.acknowledgement
        assert "Jülich Supercomputing Centre (JSC)" in entry.acknowledgement  # credits-page text
        assert "2013A&A...556A...2V" in entry.references
    chandra = P.CITATION_ENTRIES["chandra"]
    assert "CXC_CSC2_DOI" in chandra.references and P.REFERENCES["CXC_CSC2_DOI"].doi == "10.25574/csc2"
    assert "2020A&A...641A.137T" in P.CITATION_ENTRIES["xmm"].references
    h2f = P.citations_for(["cutouts"])
    assert h2f.keys == ["hips2fits"]
    assert h2f.acknowledgements[0]["text"] == ("This research made use of hips2fits,\\footnote{https://alasky.cds.unistra.fr/"
                                               "hips-image-services/hips2fits} a service provided by CDS.")
    assert "generic" in (P.CITATION_ENTRIES["hips"].note or "")


def test_dedupe_keeps_initials_and_abbreviations_inside_sentences() -> None:
    bundle = P.citations_for(["sdss", "panstarrs_dr2"])
    text = bundle.acknowledgement_text
    assert "Alfred P. Sloan Foundation, the Participating Institutions, the National Science Foundation, the U.S. " \
           "Department of Energy, the National Aeronautics" in text
    assert "under Grant No. NNX08AR22G issued through" in text
    # The Utah sentence shared by SDSS-IV and SDSS-V is given once.
    assert text.count("SDSS acknowledges support and resources from the Center for High-Performance Computing") == 1


def test_acknowledgements_carry_the_catalog_label_the_ui_reads() -> None:
    bundle = P.citations_for(["gaia_dr3", "hips2fits"], CatalogRegistry()).as_dict()
    assert [(a["catalog"], a["key"]) for a in bundle["acknowledgements"]] == [("gaia_dr3", "gaia_dr3"),
                                                                              ("hips2fits", "hips2fits")]


def test_bibtex_doi_is_not_latex_escaped() -> None:
    entries = {e.key: e for e in P.parse_bibtex(P.to_bibtex([P.REFERENCES["1991ASSL..171...89H"]]))}
    assert entries["1991ASSL..171...89H"].fields["doi"] == "10.1007/978-94-011-3250-3_10"
    for key in ("2013A&A...556A...2V", "2020A&A...641A.137T", "CXC_CSC2_DOI"):
        (entry,) = P.parse_bibtex(P.to_bibtex([P.REFERENCES[key]]))
        assert entry.fields["doi"] == P.REFERENCES[key].doi


async def test_a_real_bibcode_without_full_text_is_verified_through_other_ads_links() -> None:
    meta, handler = _ads_replay("apass_no_fulltext")
    ref = P.Reference(key="2015AAS...22533616H", entry_type="misc", authors=("Henden, A. A.",),
                      title="APASS - The Latest Data Release", year=2015, bibcode="2015AAS...22533616H")
    with respx.mock(assert_all_called=True) as router:
        router.route().mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            check = await P.verify_reference(client, ref)
    assert check.exists is True and check.ok and check.link_type == "ASSOCIATED", check.problems
    # Every full-text type was 404 first (recorded exchanges).
    assert [ex["status_code"] for ex in meta["exchanges"][:5]] == [404] * 5


# ---------------------------------------------------------------------------
# CLI robustness
# ---------------------------------------------------------------------------

_CLI = ("import argparse, sys; import provenance as P; p = argparse.ArgumentParser(); "
        "P.register_cli(p.add_subparsers(dest='command')); a = p.parse_args(sys.argv[1:]); sys.exit(a.handler(a))")


def _run_cli(*args: str, encoding: str = "cp1252") -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-c", _CLI, *args], cwd=ROOT, env=env, capture_output=True, timeout=120,
                          check=False)


def test_cli_cite_output_survives_a_cp1252_stdout() -> None:
    done = _run_cli("cite", "--catalogs", "astropy", "--json")
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    bundle = json.loads(done.stdout.decode("cp1252"))
    assert "Sipőcz, B. M." in bundle["references"][1]["authors"]
    text = _run_cli("cite", "--catalogs", "lotss,sdss")
    assert text.returncode == 0, text.stderr.decode("utf-8", "replace")
    out = text.stdout.decode("cp1252")
    assert "Jülich Supercomputing Centre" in out and "@article{2013A&A...556A...2V," in out


def test_cli_replay_prints_the_result_before_an_unwritable_out_path(tmp_path: Path, capsys) -> None:
    record, service = __import__("asyncio").run(_crossmatch_3c273())
    path = tmp_path / "manifest.json"
    path.write_text(P.build_manifest(record, registry=service.registry).to_json(), encoding="utf-8")
    args = _parser().parse_args(["replay", str(path), "--out", str(tmp_path / "no" / "such" / "dir" / "new.json")])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        assert args.handler(args) == 2
    captured = capsys.readouterr()
    assert "Identical: yes" in captured.out and "cannot write" in captured.err


def test_cli_rejects_bad_manifests_names_and_files_with_exit_code_2(tmp_path: Path, capsys) -> None:
    data = _manifest_dict()
    data["input_target"] = {}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    args = _parser().parse_args(["replay", str(bad)])
    assert args.handler(args) == 2
    assert "Invalid manifest" in capsys.readouterr().err
    missing = _parser().parse_args(["replay", str(tmp_path / "missing.json")])
    assert missing.handler(missing) == 2
    not_found = ('<?xml version="1.0" encoding="UTF-8"?><Sesame><Target option="SNV"><name>NoSuchObjectXYZ123</name>'
                 '<INFO>*** Nothing found ***</INFO></Target></Sesame>')
    with respx.mock(assert_all_called=False) as router:
        router.route(host="cds.unistra.fr").mock(return_value=httpx.Response(200, text=not_found))
        args = _parser().parse_args(["manifest", "--name", "NoSuchObjectXYZ123"])
        assert args.handler(args) == 2
    assert "No coordinates found" in capsys.readouterr().err
    with respx.mock(assert_all_called=False) as router:
        router.route(host="cds.unistra.fr").mock(side_effect=httpx.ConnectError("resolver down"))
        args = _parser().parse_args(["manifest", "--name", "M87"])
        assert args.handler(args) == 3
    unwritable = _parser().parse_args(["cite", "--catalogs", "gaia_dr3", "--bibtex", str(tmp_path / "no" / "x.bib")])
    assert unwritable.handler(unwritable) == 2
    not_json = tmp_path / "x.json"
    not_json.write_text("{", encoding="utf-8")
    from_bad = _parser().parse_args(["cite", "--from", str(not_json)])
    assert from_bad.handler(from_bad) == 2


def test_parser_help_mentions_every_subcommand() -> None:
    parser = argparse.ArgumentParser(prog="astrosearch")
    sub = parser.add_subparsers(dest="command")
    P.register_cli(sub)
    assert {"replay", "cite", "manifest"} <= set(sub.choices)
    assert parser.parse_args(["manifest", "--name", "M87", "--no-live-release"]).no_live_release is True
