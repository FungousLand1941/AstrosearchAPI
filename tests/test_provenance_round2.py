"""Regression tests for the provenance review (round 2).

Covers: ECMAScript number form of canonical JSON checked with a real ``node`` round
trip (no false tamper alarms, identical replays after a browser); target parallax
origin (a parallax adopted from a catalog row is part of the answer and is adopted
again on replay; a given one is part of the question; both are in the content hash
and the diff); ``POST /provenance/manifest`` searches run exactly as
``POST /api/v1/search`` (resolver parallax, min_confidence 0.5) and the CLI as
``astrosearch search``; replay of advanced queries whose catalogs are outside their
profiles; malformed advanced queries (422 / exit 2 before any archive request); the
deployment's execution limits during replay; no leaked HTTP clients; a non-JSON
doi.org answer; citing only catalogs that returned rows; verified curated catalog
citations (Chandra DOI); official ZTF and Astropy acknowledgements; malformed records.

Archive traffic is replayed from recorded fixtures (tests/fixtures/3c273,
barnard_j2000_pm, provenance/m87_resolver, sesame/*.xml). Live checks at the end are
marked ``live``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fixture_io import EPOCH_TARGETS, TARGETS, load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_provenance import _app, _crossmatch_3c273, _parser, _source, synthetic_record

import main
import provenance as P
from crossmatch import AdvancedQuery
from models import CatalogRegistry
from providers import CacheManager, SesameResolver

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROV = FIXTURES / "provenance"
SESAME = FIXTURES / "sesame"
RA_3C273, DEC_3C273 = TARGETS["3c273"]
BARNARD = EPOCH_TARGETS["barnard_j2000_pm"]
BARNARD_GAIA = "4472832130942575872"
BARNARD_2MASS = "17574849+0441405"
# Gaia DR3 parallax of Barnard's star (SIMBAD quotes the same value via Sesame).
BARNARD_PARALLAX_MAS = 546.9759
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node (JavaScript engine) not installed")


@pytest.fixture(autouse=True)
def _fresh_release_cache():
    P._RELEASE_CACHE.clear()
    yield
    P._RELEASE_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def js_number(value: Any) -> Any:
    """What Python reads back after ``JSON.parse`` + ``JSON.stringify`` in a browser.

    Every JSON number becomes an IEEE-754 double. ECMAScript's Number::toString writes a
    whole value below 1e21 without a fraction or exponent (2000.0 -> "2000", -0 -> "0",
    3700386905605055360 -> "3700386905605055500": the shortest round-trip digits padded
    with zeros), which Python's json then reads as an *int*; other values are written
    in the shortest round-trip form, which Python reads as a float.
    """
    if isinstance(value, dict):
        return {k: js_number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [js_number(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    double = float(value)
    if double.is_integer() and abs(double) < 1e21:
        return int(Decimal(repr(double)))
    return double


def js_round_trip(value: Any) -> Any:
    """Browser round trip: with ``node`` when installed (the real thing), else simulated."""
    text = json.dumps(value)
    if NODE is not None:
        done = subprocess.run(
            [NODE, "-e", ("let s='';process.stdin.on('data',d=>s+=d).on('end',()=>"
                          "process.stdout.write(JSON.stringify(JSON.parse(s))))")],
            input=text.encode("utf-8"), capture_output=True, check=True, timeout=60)
        return json.loads(done.stdout.decode("utf-8"))
    return js_number(json.loads(text))


def _sesame(path: Path) -> httpx.Response:
    return httpx.Response(200, text=path.read_text(encoding="utf-8"), headers={"content-type": "text/xml"})


class _FixtureResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def resolve(self, name: str):
        return SesameResolver.parse_response(name, self.path.read_text(encoding="utf-8"))


async def _replay(manifest: Any, exchanges, *, service_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> P.ReplayResult:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            return await P.replay_manifest(manifest, make_service(client, **(service_kwargs or {})),
                                           live_release_lookup=False, **kwargs)


async def _barnard(**kwargs: Any):
    """Barnard's star at J2000 with the SIMBAD proper motion given (recorded archives)."""
    exchanges = load_exchanges("barnard_j2000_pm")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(BARNARD["ra"], BARNARD["dec"], radius_arcsec=10.0, epoch=BARNARD["epoch"],
                                              pm_ra_masyr=BARNARD["pm_ra_masyr"], pm_dec_masyr=BARNARD["pm_dec_masyr"],
                                              **kwargs)
    return record, service, exchanges


# ---------------------------------------------------------------------------
# JavaScript numbers (major: whole-valued floats)
# ---------------------------------------------------------------------------


def test_canonical_numbers_follow_ecmascript() -> None:
    assert P.canonicalize(2000.0) == 2000 and isinstance(P.canonicalize(2000.0), int)
    assert P.canonicalize(-0.0) == 0 and P.canonicalize(0.0) == 0 and P.exact(-0.0) == 0
    assert P.canonicalize(2.5) == 2.5 and P.exact(187.2779154000123) == 187.2779154000123
    assert P.canonical_json({"epoch": 2000.0, "pm": [0.0, -0.0]}) == P.canonical_json({"epoch": 2000, "pm": [0, 0]})
    # Whole floats beyond 2**53 follow the big-integer rule (decimal string of the value).
    assert P.canonicalize(1e21) == "1000000000000000000000" == P.canonicalize(10**21)
    assert P.exact(float(2**53)) == str(2**53)
    assert P.canonicalize(float(2**53)) == 9007199254740000  # 12 significant digits first
    assert P.canonicalize(float("nan")) == "NaN" and P.canonicalize(float("-inf")) == "-Infinity"
    # Rounding to 12 significant digits can make a value whole.
    assert P.canonicalize(1999.9999999999998) == 2000
    # The simulation used when node is absent agrees with ECMAScript on the cases that matter.
    assert js_number([2000.0, -0.0, 0.5, 3700386905605055360, 1e21]) == [2000, 0, 0.5, 3700386905605055500, 1e21]


def test_query_hash_compares_numbers_by_value() -> None:
    manifest = P.build_manifest(synthetic_record())
    as_ints = json.loads(json.dumps(manifest.as_dict()))
    as_ints["input_target"] = {k: (int(v) if isinstance(v, float) and v.is_integer() else v)
                               for k, v in as_ints["input_target"].items()}
    as_floats = copy.deepcopy(as_ints)
    as_floats["input_target"]["ra"] = 150.0
    as_floats["radius_arcsec"] = 5.0
    for data in (as_ints, as_floats):
        assert P.verify_manifest(data) == {"content_hash_ok": True, "query_hash_ok": True, "row_hashes_ok": True}


@needs_node
def test_golden_3c273_record_hash_survives_a_real_node_round_trip() -> None:
    record = json.loads((PROV / "record_3c273.json").read_text(encoding="utf-8"))
    browser = js_round_trip(record)
    # The failing value of round 1: "error_ellipse_angle": 0.0 came back as 0.
    assert browser != record
    assert P.content_hash(browser) == P.content_hash(record) == GOLDEN_3C273_HASH
    manifest = json.loads(P.build_manifest(record).to_json())
    assert js_round_trip(manifest) == manifest  # a manifest is a fixed point of a browser round trip
    assert P.verify_manifest(js_round_trip(manifest)) == {"content_hash_ok": True, "query_hash_ok": True,
                                                          "row_hashes_ok": True}


GOLDEN_3C273_HASH = "sha256:2544a79ae20707273284a552893631da0197e7ebea7de6b4b2f6cbf69d843713"


@needs_node
async def test_m87_manifest_with_whole_epoch_and_zero_motion_survives_the_browser() -> None:
    """Round 1: epoch 2000.0 and pm 0.0 turned into ints and every integrity check failed."""
    exchanges = load_exchanges("provenance/m87_resolver")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        record = await main.search_object("M87", radius_arcsec=10.0,
                                          resolver=_FixtureResolver(PROV / "m87_resolver" / "sesame.xml"))
    manifest = json.loads(P.build_manifest(record, registry=CatalogRegistry()).to_json())
    assert manifest["input_target"]["epoch"] == 2000 and manifest["input_target"]["pm_ra_masyr"] == 0
    browser = js_round_trip(manifest)
    assert P.verify_manifest(browser) == {"content_hash_ok": True, "query_hash_ok": True, "row_hashes_ok": True}
    result = await _replay(browser, exchanges)
    assert result.identical, result.explanation
    assert not any("integrity check failed" in line for line in result.explanation)


@needs_node
def test_ui_flow_record_and_manifest_through_node_and_the_routes() -> None:
    """/api/v1/search-like record -> browser -> POST manifest {record} -> browser -> POST replay."""
    from fastapi.testclient import TestClient

    exchanges = load_exchanges("barnard_j2000_pm")
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=_sesame(SESAME / "barnards_star.xml"))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        service = make_service(offline_client())
        with TestClient(_app(service)) as client:
            record = asyncio.run(P.run_api_search(service, {"name": "Barnard's star", "radius_arcsec": 10.0},
                                                  resolver=_FixtureResolver(SESAME / "barnards_star.xml"))).as_dict()
            server_hash = P.content_hash(record)
            browser_record = js_round_trip(json.loads(json.dumps(P.exact(record))))
            made = client.post("/api/v1/provenance/manifest", json={"record": browser_record, "live_release_lookup": False})
            assert made.status_code == 200, made.text
            manifest = made.json()["manifest"]
            assert manifest["content_hash"] == server_hash
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": js_round_trip(manifest)})
            assert replayed.status_code == 200, replayed.text
            body = replayed.json()
    assert body["identical"] is True, body["explanation"]
    assert body["equivalent_within_tolerance"] is True and body["diff"]["other"] == {}
    assert not any("integrity check failed" in line for line in body["explanation"])


# ---------------------------------------------------------------------------
# Target parallax: origin, replay, hash, diff (major)
# ---------------------------------------------------------------------------


async def test_an_adopted_parallax_is_answer_not_question_and_is_readopted_on_replay() -> None:
    """Barnard's star at J2000, proper motion given, no parallax: the crossmatch adopts Gaia's."""
    record, service, exchanges = await _barnard()
    adopted = record.provenance["target_parallax"]
    # The nearest row whose motion agrees: SIMBAD's Barnard's star (a living database).
    assert adopted["source"] == "adopted" and adopted["catalog"] == "simbad"
    assert adopted["parallax_mas"] == pytest.approx(BARNARD_PARALLAX_MAS, abs=1e-3)
    manifest = P.build_manifest(record, registry=service.registry)
    assert manifest.input_target["parallax_mas"] is None  # not part of the question
    assert manifest.query["parallax_source"] is None and manifest.query["pm_source"] == "input"
    assert manifest.science["target_parallax"]["source"] == "adopted"
    assert manifest.science["target"]["parallax_mas"] == pytest.approx(BARNARD_PARALLAX_MAS, abs=1e-3)
    calls: list[dict[str, Any]] = []
    original = P.CrossmatchService.crossmatch

    async def spy(self, *args: Any, **kwargs: Any):
        calls.append(kwargs)
        return await original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(P.CrossmatchService, "crossmatch", spy)
        result = await _replay(json.loads(manifest.to_json()), exchanges)
    assert "parallax_mas" not in calls[0]  # the replay asks the original question ...
    assert result.new_manifest.science["target_parallax"]["source"] == "adopted"  # ... and adopts again
    assert result.identical, result.explanation
    assert result.diff["other"] == {}


async def test_a_given_parallax_is_question_and_is_replayed_as_given() -> None:
    record, service, exchanges = await _barnard(parallax_mas=BARNARD_PARALLAX_MAS)
    assert record.provenance["target_parallax"] == {"parallax_mas": BARNARD_PARALLAX_MAS, "source": "input"}
    manifest = P.build_manifest(record, registry=service.registry)
    assert manifest.input_target["parallax_mas"] == BARNARD_PARALLAX_MAS
    assert manifest.query["parallax_source"] == "input"
    result = await _replay(json.loads(manifest.to_json()), exchanges)
    assert result.identical, result.explanation
    assert result.new_manifest.science["target_parallax"] == {"parallax_mas": BARNARD_PARALLAX_MAS, "source": "input"}
    # The question differs from the adopted-parallax run, and so does the query hash.
    adopted, _, _ = await _barnard()
    assert P.build_manifest(adopted, registry=service.registry).query_hash != manifest.query_hash


@pytest.mark.parametrize(("info", "expected"), [
    ({"parallax_mas": 546.9759, "source": "input"}, "input"),
    ({"parallax_mas": 546.9759, "source": "resolver"}, "resolver"),
    ({"parallax_mas": 546.9759, "source": "adopted", "catalog": "simbad"}, None),
    (None, None),
])
def test_given_parallax_source_and_input_target(info: dict[str, Any] | None, expected: str | None) -> None:
    rec = synthetic_record()
    rec["target"].update(epoch=2000.0, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394,
                         parallax_mas=(info or {}).get("parallax_mas"))
    rec["provenance"]["target_proper_motion"] = {"pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394, "source": "input"}
    rec["provenance"]["target_parallax"] = info
    assert P.given_parallax_source(rec) == expected
    target = P.input_target(rec)
    assert target.parallax_mas == (546.9759 if expected else None)
    assert P.build_manifest(rec).query["parallax_source"] == expected


def test_a_given_parallax_is_kept_even_when_the_proper_motion_was_adopted() -> None:
    rec = synthetic_record()
    rec["target"].update(epoch=2000.0, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394, parallax_mas=546.9759)
    rec["provenance"]["target_proper_motion"] = {"pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394, "source": "adopted"}
    rec["provenance"]["target_parallax"] = {"parallax_mas": 546.9759, "source": "input"}
    target = P.input_target(rec)
    assert target.proper_motion is None and target.parallax_mas == 546.9759


def test_parallax_and_its_origin_are_science() -> None:
    base = synthetic_record()
    base["target"]["parallax_mas"] = None
    base["provenance"]["target_parallax"] = None
    with_plx = copy.deepcopy(base)
    with_plx["target"]["parallax_mas"] = 546.98
    with_plx["provenance"]["target_parallax"] = {"parallax_mas": 546.98, "source": "adopted", "catalog": "simbad",
                                                 "source_id": "NAME Barnard's star", "separation_arcsec": 0.0}
    assert P.content_hash(base) != P.content_hash(with_plx)
    diff = P.diff_science(P.science_payload(base), P.science_payload(with_plx))
    assert diff["equivalent"] is False
    assert set(diff["other"]) == {"target_parallax"} and diff["target"]["parallax_mas"] == {"old": None, "new": 546.98}
    revised = copy.deepcopy(with_plx)  # SIMBAD revised the parallax it serves
    revised["target"]["parallax_mas"] = 547.45
    revised["provenance"]["target_parallax"]["parallax_mas"] = 547.45
    diff = P.diff_science(P.science_payload(with_plx), P.science_payload(revised))
    assert diff["other"]["target_parallax"]["new"]["parallax_mas"] == 547.45
    notes = P._explain(P.build_manifest(with_plx), P.build_manifest(revised), diff, {}, [])
    assert any(n.startswith("Target parallax changed: adopted 546.98 -> adopted 547.45 mas") for n in notes), notes


# ---------------------------------------------------------------------------
# The manifest route searches as /api/v1/search; the CLI as `astrosearch search` (critical)
# ---------------------------------------------------------------------------


async def _api_search(monkeypatch: pytest.MonkeyPatch, service, client: httpx.AsyncClient, **fields: Any) -> dict[str, Any]:
    """api._search (the body of POST /api/v1/search) with this service and client."""
    import api

    monkeypatch.setattr(api.app.state, "service", service, raising=False)
    monkeypatch.setattr(api.app.state, "client", client, raising=False)
    monkeypatch.setattr(api, "cache", CacheManager(None))
    monkeypatch.setenv("API_SEARCH_CACHE_TTL_SECONDS", "0")
    return await api._search(api.SearchRequest(**fields))


def _route_manifest(service, body: dict[str, Any], exchanges, sesame: Path | None = None) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        if sesame is not None:
            router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(return_value=_sesame(sesame))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(_app(service)) as client:
            made = client.post("/api/v1/provenance/manifest", json={**body, "live_release_lookup": False,
                                                                     "include_record": True})
    assert made.status_code == 200, made.text
    return made.json()


async def test_route_name_search_equals_api_search_and_keeps_the_resolver_parallax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 2 critical: {name: Barnard's star, catalogs: [twomass_psc]} dropped the Sesame parallax.

    Without it the real 2MASS counterpart (17574849+0441405, J2000.0 epoch 1998.4) sat
    at 0.3" with confidence 0.007 and fell below /api/v1/search's min_confidence 0.5.
    """
    exchanges = load_exchanges("barnard_j2000_pm")
    fields = {"name": "Barnard's star", "radius_arcsec": 10.0, "catalogs": ["twomass_psc", "allwise"]}
    with respx.mock(assert_all_called=False) as router:
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=_sesame(SESAME / "barnards_star.xml"))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            api_record = await _api_search(monkeypatch, service, client, **fields)
    body = _route_manifest(make_service(offline_client()), fields, exchanges, SESAME / "barnards_star.xml")
    manifest = body["manifest"]
    assert manifest["content_hash"] == P.content_hash(api_record)
    assert manifest["query"]["mode"] == "advanced" and manifest["query"]["advanced_query"]["min_confidence"] == 0.5
    assert manifest["input_target"]["parallax_mas"] == BARNARD_PARALLAX_MAS  # Sesame's, labelled as such
    assert manifest["query"]["parallax_source"] == "resolver" and manifest["query"]["pm_source"] == "resolver"
    assert manifest["science"]["target_parallax"] == {"parallax_mas": BARNARD_PARALLAX_MAS, "source": "resolver"}
    assert manifest["query"]["resolver"]
    twomass = {m["source_id"]: m for m in manifest["science"]["matches"] if m["catalog"] == "twomass_psc"}
    assert twomass[BARNARD_2MASS]["separation_arcsec"] < 0.05 and twomass[BARNARD_2MASS]["confidence"] > 0.9
    assert any(m["catalog"] == "allwise" and m["confidence"] >= 0.5 for m in manifest["science"]["matches"])


async def test_route_coordinate_search_equals_api_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """3C 273 at 10": the route used to keep every in-radius row (36) where /search keeps 11."""
    exchanges = load_exchanges("3c273")
    fields = {"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0}
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            api_record = await _api_search(monkeypatch, make_service(client), client, **fields)
    body = _route_manifest(make_service(offline_client()), fields, exchanges)
    assert body["manifest"]["content_hash"] == P.content_hash(api_record)
    rows = sum(len(c["sources"]) for c in body["manifest"]["science"]["catalogs"].values())
    assert rows == sum(len(r["sources"]) for r in api_record["catalog_results"].values())
    # Other /api/v1/search fields are accepted and change the search the same way.
    fields = {**fields, "min_confidence": 0.0, "catalogs": ["gaia_dr3", "simbad"], "max_results": 1}
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            api_record = await _api_search(monkeypatch, make_service(client), client, **fields)
    body = _route_manifest(make_service(offline_client()), fields, exchanges)
    assert body["manifest"]["content_hash"] == P.content_hash(api_record)
    assert sorted(body["manifest"]["catalogs"]) == ["gaia_dr3", "simbad"]


def test_route_rejects_a_record_together_with_search_fields() -> None:
    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        res = client.post("/api/v1/provenance/manifest", json={"record": synthetic_record(), "ra": 1.0, "dec": 1.0})
        assert res.status_code == 422 and "not both" in res.json()["detail"]
        assert client.post("/api/v1/provenance/manifest", json={"ra": 1.0, "dec": 1.0, "min_confidence": 2}).status_code == 422


def test_cli_manifest_name_search_equals_astrosearch_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`manifest --name` runs what `astrosearch search --name` (main.search_object) runs, parallax included."""
    exchanges = load_exchanges("barnard_j2000_pm")
    with respx.mock(assert_all_called=False) as router:
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=_sesame(SESAME / "barnards_star.xml"))
        router.route().mock(side_effect=replay_side_effect(exchanges))
        record = asyncio.run(main.search_object("Barnard's star", radius_arcsec=10.0))
        out = tmp_path / "m.json"
        args = _parser().parse_args(["manifest", "--name", "Barnard's star", "--radius", "10", "--no-live-release",
                                     "-o", str(out)])
        assert args.handler(args) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["content_hash"] == P.content_hash(record)
    assert manifest["query"]["mode"] == "basic"
    assert manifest["input_target"]["parallax_mas"] == BARNARD_PARALLAX_MAS
    assert manifest["query"]["parallax_source"] == "resolver"


def test_cli_manifest_with_catalogs_searches_as_the_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exchanges = load_exchanges("barnard_j2000_pm")
    with respx.mock(assert_all_called=False) as router:
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=_sesame(SESAME / "barnards_star.xml"))
        router.route().mock(side_effect=replay_side_effect(exchanges))

        async def api_record() -> dict[str, Any]:
            async with offline_client() as client:
                return await _api_search(monkeypatch, make_service(client), client, name="Barnard's star",
                                         radius_arcsec=10.0, catalogs=["twomass_psc"])

        expected = asyncio.run(api_record())
        out = tmp_path / "m.json"
        args = _parser().parse_args(["manifest", "--name", "Barnard's star", "--radius", "10", "--catalogs", "twomass_psc",
                                     "--no-live-release", "-o", str(out)])
        assert args.handler(args) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["content_hash"] == P.content_hash(expected)
    assert manifest["query"]["mode"] == "advanced"


# ---------------------------------------------------------------------------
# Replay of advanced queries (major)
# ---------------------------------------------------------------------------


async def test_advanced_manifest_with_catalogs_outside_its_profiles_replays() -> None:
    """{profiles: [optical], catalogs: [gaia_dr3, nvss]} silently planned gaia_dr3 only (nvss is radio); the
    crossmatch now refuses such a query (a catalog asked for by name is never dropped silently), and the manifest
    of the query it asks for instead ({profiles: [optical], catalogs: [gaia_dr3]}) replays identically."""
    exchanges = load_exchanges("3c273")
    fields = {"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0, "profiles": ["optical"], "min_confidence": 0.5}
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            with pytest.raises(ValueError, match="nvss are not in profile 'optical'"):
                await service.crossmatch(RA_3C273, DEC_3C273,
                                         query=AdvancedQuery.from_dict({**fields, "catalogs": ["gaia_dr3", "nvss"]}))
            query = AdvancedQuery.from_dict({**fields, "catalogs": ["gaia_dr3"]})
            record = await service.crossmatch(RA_3C273, DEC_3C273, query=query)
    manifest = P.build_manifest(record, registry=service.registry)
    assert sorted(manifest.catalogs) == ["gaia_dr3"]
    result = await _replay(json.loads(manifest.to_json()), exchanges)
    assert result.identical, result.explanation
    # CLI replay of the same manifest: exit 0 (identical), not 2.


def _advanced_manifest() -> dict[str, Any]:
    data = P.build_manifest(synthetic_record()).as_dict()
    data["query"].update(mode="advanced", advanced_query={"target": {"ra": 150.0, "dec": 2.0}, "radius_arcsec": 5.0,
                                                           "catalogs": ["gaia_dr3", "simbad"]})
    return data


BAD_ADVANCED: list[tuple[str, str, Any]] = [
    ("time_period is text", "time_period", "2020"),
    ("spatial_constraints is text", "spatial_constraints", "x"),
    ("exclude_polygons is a number", "spatial_constraints", {"exclude_polygons": 5}),
    ("radius_zones is a number", "spatial_constraints", {"radius_zones": 3}),
    ("radius zone is a list", "spatial_constraints", {"radius_zones": [[0, 1]]}),
    ("object_types is a number", "object_types", 5),
    ("object_types holds numbers", "object_types", [5]),
    ("metadata is a list", "metadata", [1]),
    ("metadata pm_source is a number", "metadata", {"pm_source": 7}),
    ("filters is a list", "filters", ["x"]),
    ("catalogs is a string", "catalogs", "gaia_dr3"),
    ("min_confidence out of range", "min_confidence", 3.0),
    ("time_period reversed", "time_period", {"start": 2020, "end": 2010}),
]


@pytest.mark.parametrize(("label", "key", "value"), BAD_ADVANCED, ids=[b[0] for b in BAD_ADVANCED])
def test_malformed_advanced_queries_are_manifest_errors(label: str, key: str, value: Any) -> None:
    data = _advanced_manifest()
    P.ProvenanceManifest.from_dict(data)  # the base is valid
    data["query"]["advanced_query"][key] = value
    with pytest.raises(P.ManifestError):
        P.ProvenanceManifest.from_dict(data)


def test_replay_route_and_cli_reject_malformed_advanced_queries_before_any_archive(tmp_path: Path, capsys) -> None:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        archive = router.route()
        archive.mock(side_effect=AssertionError("an archive was queried for a malformed manifest"))
        with TestClient(_app(make_service(offline_client())), raise_server_exceptions=False) as client:
            for label, key, value in BAD_ADVANCED:
                data = _advanced_manifest()
                data["query"]["advanced_query"][key] = value
                res = client.post("/api/v1/provenance/replay", json={"manifest": data})
                assert res.status_code == 422, (label, res.status_code, res.text)
            # Valid structure, but a catalog the registry cannot plan: still 422, no request.
            data = _advanced_manifest()
            data["query"]["advanced_query"]["profiles"] = ["no-such-profile"]
            res = client.post("/api/v1/provenance/replay", json={"manifest": data})
            assert res.status_code == 422, res.text
        data = _advanced_manifest()
        data["query"]["advanced_query"]["metadata"] = [1]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        args = _parser().parse_args(["replay", str(bad)])
        assert args.handler(args) == 2
        assert archive.call_count == 0
    assert "Invalid manifest" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Replay execution limits; HTTP client hygiene (minor)
# ---------------------------------------------------------------------------


async def test_replay_uses_the_deployment_timeout_cap_and_response_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from providers import provider_map

    record, service = await _crossmatch_3c273()
    manifest = P.build_manifest(record, registry=service.registry)
    seen: dict[str, Any] = {}
    real_provider_map, real_service = P.provider_map, P.CrossmatchService

    def spy_map(*args: Any, **kwargs: Any):
        seen["max_response_bytes"] = kwargs.get("max_response_bytes")
        return real_provider_map(*args, **kwargs)

    class SpyService(real_service):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen["timeout_cap"] = kwargs.get("timeout_cap")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(P, "provider_map", spy_map)
    monkeypatch.setattr(P, "CrossmatchService", SpyService)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as client:
            deployment = real_service(CatalogRegistry(), provider_map(client, cache=CacheManager(None),
                                                                      max_response_bytes=25_000_000),
                                      timeout=45.0, timeout_cap=20.0)
            result = await P.replay_manifest(manifest, deployment, live_release_lookup=False)
    assert result.identical, result.explanation
    assert seen == {"max_response_bytes": 25_000_000, "timeout_cap": 20.0}


def test_routes_without_app_state_close_every_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare app (no app.state service/client): one manifest {name} and one replay request."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    created: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    class Tracked(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", Tracked)
    monkeypatch.setenv("CATALOG_REGISTRY_STRICT", "false")
    app = FastAPI()
    app.include_router(P.router)
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").mock(
            return_value=_sesame(SESAME / "3c273.xml"))
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        with TestClient(app) as client:
            made = client.post("/api/v1/provenance/manifest", json={"name": "3C 273", "radius_arcsec": 10.0,
                                                                     "live_release_lookup": False})
            assert made.status_code == 200, made.text
            replayed = client.post("/api/v1/provenance/replay", json={"manifest": made.json()["manifest"]})
            assert replayed.status_code == 200, replayed.text
    assert created, "the routes made no HTTP client at all?"
    assert [c for c in created if not c.is_closed] == []


# ---------------------------------------------------------------------------
# Citations (minor)
# ---------------------------------------------------------------------------


async def test_a_non_json_doi_org_answer_is_undecided_not_a_crash() -> None:
    ref = P.REFERENCES["2006AJ....131.1163S"]
    dataset = P.REFERENCES["IRSA_2MASS_PSC_DOI"]
    html = httpx.Response(200, text="<html><body>DOI landing page</body></html>", headers={"content-type": "text/html"})
    with respx.mock(assert_all_called=False) as router:
        router.route(host="ui.adsabs.harvard.edu").mock(
            return_value=httpx.Response(302, headers={"location": f"https://doi.org/{ref.doi}"}))
        router.route(host="doi.org").mock(return_value=html)
        async with httpx.AsyncClient() as client:
            checks = await P.verify_references(client, [ref, dataset], delay_seconds=0)
    article, data = checks
    assert article.exists is True and article.metadata_matches is None and article.undecided
    assert any(p.startswith("unparsable doi.org answer") for p in article.problems)
    assert data.exists is None and data.undecided


def _record_with_empty_catalogs() -> dict[str, Any]:
    rec = synthetic_record()
    rec["catalog_results"]["chandra"] = {"sources": [], "status": "empty", "row_count": 0, "truncated": False}
    rec["catalog_results"]["exoplanet_archive"] = {"sources": [], "status": "empty", "row_count": 0, "truncated": False}
    rec["provenance"]["catalogs_planned"] += ["chandra", "exoplanet_archive"]
    return rec


def test_only_catalogs_that_returned_rows_are_cited(tmp_path: Path, capsys) -> None:
    rec = _record_with_empty_catalogs()
    manifest = P.build_manifest(rec)
    for obj in (rec, manifest, manifest.as_dict()):
        assert P.catalogs_in(obj) == ["gaia_dr3", "simbad"]
        assert P.catalogs_in(obj, include_empty=True) == ["chandra", "exoplanet_archive", "gaia_dr3", "simbad"]
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    args = _parser().parse_args(["cite", "--from", str(path), "--json"])
    assert args.handler(args) == 0
    keys = json.loads(capsys.readouterr().out)["keys"]
    assert keys == ["gaia_dr3", "simbad"]
    args = _parser().parse_args(["cite", "--from", str(path), "--include-empty"])
    assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert "Chandra Source Catalog" in out and "NASA Exoplanet Archive" in out


def test_manifest_catalog_citations_come_from_the_verified_references() -> None:
    rec = _record_with_empty_catalogs()
    chandra = P.build_manifest(rec, registry=CatalogRegistry()).catalogs["chandra"]
    assert chandra.citation_source.startswith("curated")
    assert "doi:10.25574/csc2" in chandra.citation and "csc2.1" not in chandra.citation
    assert "2024ApJS..274...22E" in chandra.citation
    # The registry's free text is passed on only as such, and a DOI it names that is not a
    # verified reference is listed (doi.org answers 404 for 10.25574/csc2.1).
    assert P.unverified_dois("Evans et al. 2010, ApJS 189, 37; DOI 10.25574/csc2.1") == ["10.25574/csc2.1"]
    assert P.unverified_dois("DOI 10.25574/csc2 and 10.1086/498708.") == []
    ack = P.citations_for(["chandra"], CatalogRegistry()).acknowledgements[0]
    assert ack["registry_citation_verified"] is False
    assert ack["registry_citation_unverified_dois"] == P.unverified_dois(ack["registry_citation"])


def test_ztf_and_astropy_acknowledgements_are_the_official_wording() -> None:
    ztf = P.CITATION_ENTRIES["ztf"]
    assert ztf.acknowledgement.startswith(
        "Based on observations obtained with the Samuel Oschin Telescope 48-inch and the 60-inch Telescope at the "
        "Palomar Observatory as part of the Zwicky Transient Facility project. ZTF is supported by the National "
        "Science Foundation under Grants No. AST-1440341 and AST-2034437 and a collaboration including current partners")
    assert ztf.acknowledgement.endswith("Operations are conducted by COO, IPAC, and UW.")
    assert ztf.source_url.endswith("/ztf_release_notes_dr24.pdf")
    assert ztf.references[0] == "2019PASP..131a8003M"  # the paper the release notes ask for
    assert P.citations_for(["ztf"]).keys == ["ztf", "irsa"]  # IRSA serves the data
    astropy = P.CITATION_ENTRIES["astropy"].acknowledgement
    # astropy.org/acknowledging.html has a space between "Astropy:" and "\footnote".
    assert astropy.startswith("This work made use of Astropy: \\footnote{https://www.astropy.org} a community-developed "
                              "core Python package and an ecosystem of tools and resources for astronomy")
    bib_keys = {e.key for e in P.parse_bibtex(P.citations_for(["astropy"]).bibtex)}
    cited = astropy.split("\\citep{", 1)[1].split("}", 1)[0].split(", ")
    assert set(cited) == bib_keys  # every \citep key exists in the BibTeX written alongside


# ---------------------------------------------------------------------------
# Malformed records (minor)
# ---------------------------------------------------------------------------


BAD_RECORDS: list[tuple[str, Any]] = [
    ("catalogs_planned is a string", lambda r: r["provenance"].__setitem__("catalogs_planned", "gaia_dr3")),
    ("catalogs_planned holds numbers", lambda r: r["provenance"].__setitem__("catalogs_planned", [1])),
    ("catalog_results is a list", lambda r: r.__setitem__("catalog_results", [])),
    ("catalog result is a string", lambda r: r["catalog_results"].__setitem__("gaia_dr3", "rows")),
    ("sources is a string", lambda r: r["catalog_results"]["gaia_dr3"].__setitem__("sources", "rows")),
    ("matches is an object", lambda r: r["provenance"].__setitem__("matches", {"a": 1})),
    ("failures holds strings", lambda r: r.__setitem__("failures", ["nvss"])),
    ("target_parallax is a number", lambda r: r["provenance"].__setitem__("target_parallax", 5)),
    ("resolved_object is a list", lambda r: r.__setitem__("resolved_object", ["M87"])),
]


@pytest.mark.parametrize(("label", "mutate"), BAD_RECORDS, ids=[b[0] for b in BAD_RECORDS])
def test_malformed_records_are_rejected(label: str, mutate) -> None:
    from fastapi.testclient import TestClient

    rec = synthetic_record()
    mutate(rec)
    with pytest.raises(P.ManifestError):
        P.build_manifest(rec)
    with TestClient(_app()) as client:
        res = client.post("/api/v1/provenance/manifest", json={"record": rec, "live_release_lookup": False})
    assert res.status_code == 422, res.text


def test_old_schema_manifests_are_refused_with_the_expected_schema() -> None:
    data = P.build_manifest(synthetic_record()).as_dict()
    data["schema"] = "astrosearch.provenance.manifest/2"
    with pytest.raises(P.ManifestError, match="manifest/3"):
        P.ProvenanceManifest.from_dict(data)


def test_empty_catalog_rows_count_uses_sources_when_row_count_is_missing() -> None:
    rec = synthetic_record()
    rec["catalog_results"]["simbad"].pop("row_count")
    assert P.catalogs_in(rec) == ["gaia_dr3", "simbad"]
    rec["catalog_results"]["simbad"]["sources"] = []
    assert P.catalogs_in(rec) == ["gaia_dr3"]
    assert _source  # shared helper stays importable for the round-1 tests


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


def _unreachable(record: Any) -> list[str]:
    return [f["catalog"] for f in record.failures if f.get("error_type") in P.UNREACHABLE_ERRORS]


@pytest.mark.live
def test_live_adopted_parallax_is_readopted_on_replay() -> None:
    """Barnard's star at J2000 with its proper motion and no parallax, live (Gaia DR3 + SIMBAD + 2MASS)."""

    async def run():
        os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "2"
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            service = main.build_service(client=client)
            query = {"ra": BARNARD["ra"], "dec": BARNARD["dec"], "radius_arcsec": 10.0, "epoch": 2000.0,
                     "pm_ra_masyr": BARNARD["pm_ra_masyr"], "pm_dec_masyr": BARNARD["pm_dec_masyr"]}
            registry = P._PinnedRegistry(service.registry, ["gaia_dr3", "simbad", "twomass_psc"])
            pinned = P.CrossmatchService(registry, service.providers, timeout=60.0)
            record = await pinned.crossmatch(query["ra"], query["dec"], radius_arcsec=10.0, epoch=2000.0,
                                             pm_ra_masyr=query["pm_ra_masyr"], pm_dec_masyr=query["pm_dec_masyr"])
            if _unreachable(record):
                return record, None, None
            manifest = P.build_manifest(record, registry=registry)
            result = await P.replay_manifest(json.loads(manifest.to_json()), service, client=client,
                                             live_release_lookup=False)
            return record, manifest, result

    record, manifest, result = asyncio.run(run())
    if manifest is None:
        pytest.skip(f"archives unreachable: {_unreachable(record)}")
    plx = record.provenance["target_parallax"]
    assert plx["source"] == "adopted" and plx["parallax_mas"] == pytest.approx(546.98, abs=0.1)  # Gaia DR3 / SIMBAD
    assert manifest.input_target["parallax_mas"] is None and manifest.query["parallax_source"] is None
    assert result.new_manifest.science["target_parallax"]["source"] == "adopted"
    assert result.new_manifest.science["target_parallax"]["parallax_mas"] == pytest.approx(546.98, abs=0.1)
    # 2MASS 17574849+0441405 (epoch 1998.4) is Barnard's star once proper motion and parallax are removed.
    twomass = [m for m in record.provenance["matches"] if m["catalog"] == "twomass_psc"]
    assert any(m["source_id"] == BARNARD_2MASS and m["separation_arcsec"] < 0.2 for m in twomass), twomass
    if not result.identical:
        changed = set(result.diff["catalogs"]) | set(result.diff["other"])
        assert changed <= {"simbad", "target_parallax"}, result.explanation  # SIMBAD is a living database


@pytest.mark.live
def test_live_route_name_manifest_equals_api_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """61 Cyg A, 5": POST /provenance/manifest {name} and /api/v1/search give the same science."""
    from fastapi.testclient import TestClient

    import api

    fields = {"name": "61 Cyg A", "radius_arcsec": 5.0, "catalogs": ["gaia_dr3", "twomass_psc", "simbad"]}

    async def api_record() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            return await _api_search(monkeypatch, main.build_service(client=client), client, **fields)

    expected = asyncio.run(api_record())
    if [f for f in expected["failures"] if f.get("error_type") in P.UNREACHABLE_ERRORS]:
        pytest.skip(f"archives unreachable: {expected['failures']}")
    app = __import__("fastapi").FastAPI()
    app.include_router(P.router)
    with TestClient(app) as client:
        made = client.post("/api/v1/provenance/manifest", json={**fields, "live_release_lookup": False})
    if made.status_code == 502:
        pytest.skip(f"upstream unavailable: {made.text}")
    assert made.status_code == 200, made.text
    manifest = made.json()["manifest"]
    # 61 Cyg A: parallax 286 mas (Gaia DR3 285.99), from the resolver, as /api/v1/search uses it.
    assert manifest["query"]["parallax_source"] == "resolver"
    assert 280.0 < manifest["input_target"]["parallax_mas"] < 292.0
    assert expected["provenance"]["target_parallax"]["source"] == "resolver"
    if manifest["content_hash"] != P.content_hash(expected):
        diff = P.diff_science(P.science_payload(expected), manifest["science"])
        changed = set(diff["catalogs"])
        assert changed <= {"simbad"}, diff  # only the living database may change between the two calls
    assert api  # imported for _api_search


@pytest.mark.live
@needs_node
def test_live_3c273_manifest_survives_a_node_round_trip() -> None:
    async def run():
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            service = main.build_service(client=client)
            return await P.run_api_search(service, {"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0,
                                                    "catalogs": ["gaia_dr3", "simbad", "sdss", "twomass_psc"]})

    record = asyncio.run(run())
    if _unreachable(record):
        pytest.skip(f"archives unreachable: {_unreachable(record)}")
    assert "3C 273" in [s.source_id for s in record.catalog_results["simbad"]["sources"]]
    server = record.as_dict()
    manifest = json.loads(P.build_manifest(server, registry=CatalogRegistry()).to_json())
    assert P.content_hash(js_round_trip(json.loads(json.dumps(P.exact(server))))) == manifest["content_hash"]
    assert P.verify_manifest(js_round_trip(manifest)) == {"content_hash_ok": True, "query_hash_ok": True,
                                                          "row_hashes_ok": True}


@pytest.mark.live
def test_live_chandra_dois_and_astropy_acknowledgement() -> None:
    async def run():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                good = await P.fetch_csl(client, "10.25574/csc2")
                bad = await P.fetch_csl(client, "10.25574/csc2.1")
                page = await client.get("https://www.astropy.org/acknowledging.html")
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                return exc
            return good, bad, page

    out = asyncio.run(run())
    if isinstance(out, Exception):
        pytest.skip(f"doi.org / astropy.org unreachable: {out}")
    good, bad, page = out
    assert good is not None and "Chandra" in str(good.get("title"))  # the CSC Release 2 series DOI
    assert bad is None  # the registry's DOI is not registered
    import html
    import re

    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", page.text)))
    # Word for word up to the \citep, whose keys are ours (see the entry's note).
    ours = P.CITATION_ENTRIES["astropy"].acknowledgement.split(" \\citep{", 1)[0]
    assert ours.startswith("This work made use of Astropy: \\footnote{https://www.astropy.org}")
    assert ours + " \\citep{astropy:2013, astropy:2018, astropy:2022}." in text
