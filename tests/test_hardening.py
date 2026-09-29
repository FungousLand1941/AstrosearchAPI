"""Regression tests for the round-2 review: caching of encoded responses, transport-error
classification, request pacing, truncation warnings, null sentinels, unit parsing,
positional-error specs, archive error text, and API filtered counts.

Everything goes through the real provider/executor/service stack with httpx
MockTransport or respx; nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import gzip
import itertools
import json
import math
import time
import warnings

import httpx
import pytest
import respx
from fixture_io import FIXTURES, TARGETS, load_exchanges, replay_side_effect

from crossmatch import CrossmatchService, QueryExecutor
from models import (
    DEFAULT_CATALOGS,
    CatalogQueryError,
    CatalogRegistry,
    CatalogUnavailableError,
    QueryPlan,
    angle_to_deg,
    arcsec_per_unit,
    catalog_from_dict,
    compute_positional_error,
    haversine_arcsec,
    pm_to_masyr,
    propagate_radec,
    validate_catalog_definition,
    validate_target,
    votable_query_status,
)
from providers import CacheManager, IRSAGatorProvider, TapProvider, provider_map

TARGET = validate_target(187.2779154, 2.0523883)
PRIMARY = "https://primary.example/tap/sync"
SECONDARY = "https://secondary.example/tap/sync"


def tap_def(name: str = "t", endpoint: str = PRIMARY, **extra):
    entry = {
        "provider": "tap", "wavelength": "optical", "endpoint": endpoint, "table": "t", "epoch": 2000.0,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0,
    }
    parameters = extra.pop("parameters", None)
    entry.update(extra)
    if parameters:
        entry["parameters"] = {**entry["parameters"], **parameters}
    return catalog_from_dict(name, entry)


def tap_payload(rows: list[list]) -> dict:
    return {"metadata": [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}, {"name": "match_dist"}],
            "data": rows}


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Provider cache: gzip-encoded responses replay correctly
# ---------------------------------------------------------------------------


async def test_gzip_response_is_served_from_cache_on_the_second_query(monkeypatch) -> None:
    # VizieR TAP and the Exoplanet Archive answer 'content-encoding: gzip'. The cached
    # body is the DECODED content, so replaying the encoding header broke every repeat.
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    body = gzip.compress(json.dumps(tap_payload([["A", 187.2779154, 2.0523883, 0.0]])).encode())
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "application/json", "content-encoding": "gzip"})

    cache = CacheManager(None)
    async with mock_client(handler) as client:
        provider = TapProvider(client, cache=cache)
        first = await provider.query(tap_def(), TARGET, 5.0)
        second = await provider.query(tap_def(), TARGET, 5.0)
    assert len(calls) == 1
    (_, entry, _size), = cache._local_cache.values()
    assert entry["headers"] == {"content-type": "application/json"}  # no content-encoding/length stored
    assert first.meta["cached"] is False and second.meta["cached"] is True
    assert [s.source_id for s in second] == [s.source_id for s in first] == ["A"]


async def test_legacy_cache_entry_with_encoding_header_still_replays(monkeypatch) -> None:
    # Entries written before the fix (Redis survives restarts) still carry the header.
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    cache = CacheManager(None)
    provider = TapProvider(mock_client(lambda r: httpx.Response(500)), cache=cache)
    catalog = tap_def()
    from models import plan_cone

    cone = plan_cone(catalog, TARGET, 5.0)
    form = provider.build_request(catalog, provider.build_adql(catalog, TARGET, 5.0, cone))
    key = cache.make_key("provider", "POST", PRIMARY, None, form)
    import base64

    cache.set(key, {"status": 200, "headers": {"content-type": "application/json", "content-encoding": "gzip"},
                    "content": base64.b64encode(json.dumps(tap_payload([["B", 187.2779154, 2.0523883, 0.0]])).encode()).decode()})
    result = await provider.query(catalog, TARGET, 5.0)
    assert result.meta["cached"] is True and result[0].source_id == "B"


def test_cache_manager_ttl_zero_or_negative_stores_nothing() -> None:
    cache = CacheManager(None)
    cache.set("k0", {"v": 1}, ttl=0)
    cache.set("kn", {"v": 1}, ttl=-5)
    assert cache._local_cache == {}  # not stored at all (Redis would reject setex with ttl <= 0)
    assert cache.get("k0") is None and cache.get("kn") is None
    cache.set("k1", {"v": 1}, ttl=60)
    assert cache.get("k1") == {"v": 1}


# ---------------------------------------------------------------------------
# Transport errors outside the retry/timeout tuples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", [httpx.DecodingError, httpx.TooManyRedirects, httpx.UnsupportedProtocol,
                                      httpx.LocalProtocolError])
async def test_other_httpx_errors_are_catalog_unavailable(exc_type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("boom", request=request)

    guards: dict = {}
    async with mock_client(handler) as client:
        provider = TapProvider(client, guards=guards, cache=CacheManager(None))
        with pytest.raises(CatalogUnavailableError, match=exc_type.__name__):
            await provider.query(tap_def(), TARGET, 5.0)
    assert guards[PRIMARY]._failures == 1  # counted by the circuit breaker exactly once


async def test_decoding_error_triggers_the_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary.example":
            raise httpx.DecodingError("Error -3 while decompressing data", request=request)
        return httpx.Response(200, json=tap_payload([["FB", 187.2779154, 2.0523883, 0.0]]))

    catalog = tap_def("fb", parameters={"fallback": {"provider": "tap", "endpoint": SECONDARY}})
    reg = CatalogRegistry()
    reg._catalogs = {"fb": catalog}
    async with mock_client(handler) as client:
        executor = QueryExecutor(provider_map(client, cache=CacheManager(None)), registry=reg)
        plan = QueryPlan("fb", "tap", PRIMARY, {}, 5.0, "optical")
        successes, failures = await executor.execute([plan], TARGET)
    assert failures == []
    result = successes[0][1]
    assert result[0].source_id == "FB"
    assert "DecodingError" in result.meta["fallback"]["reason"]


# ---------------------------------------------------------------------------
# Request pacing through the provider stack
# ---------------------------------------------------------------------------


async def test_requests_to_one_endpoint_are_paced(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_REQUESTS_PER_SECOND", "10")
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(time.monotonic())
        return httpx.Response(200, json=tap_payload([["A", 187.2779154, 2.0523883, 0.0]]))

    async with mock_client(handler) as client:
        provider = TapProvider(client, cache=CacheManager(None))
        await asyncio.gather(*(provider.query(tap_def(), TARGET, r) for r in (1.0, 2.0, 3.0, 4.0, 5.0)))
    starts.sort()
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    # 10 req/s -> request starts ~0.1 s apart (Windows monotonic ticks are ~16 ms; unpaced gaps are ~0).
    assert len(starts) == 5 and min(gaps) >= 0.06, gaps
    assert starts[-1] - starts[0] >= 0.35


# ---------------------------------------------------------------------------
# Row handling: RA wrap, truncation warnings, Gator error text, Oracle ACOS retry
# ---------------------------------------------------------------------------


def test_ra_is_wrapped_into_0_360() -> None:
    provider = TapProvider(mock_client(lambda r: httpx.Response(500)), cache=CacheManager(None))
    rows = [{"id": "neg", "ra": -0.0001, "dec": 0.0}, {"id": "over", "ra": 360.0002, "dec": 0.0},
            {"id": "tiny", "ra": -1e-14, "dec": 0.0}]
    result = provider._sources(tap_def(), rows, 5.0, PRIMARY, {}, target=validate_target(0.0, 0.0))
    by_id = {s.source_id: s.ra for s in result}
    assert by_id["neg"] == pytest.approx(359.9999)
    assert by_id["over"] == pytest.approx(0.0002)
    assert by_id["tiny"] == 0.0
    assert all(0.0 <= ra < 360.0 for ra in by_id.values())


async def test_truncated_crowded_field_warns_and_reports_the_covered_radius() -> None:
    # 4 rows for max_rows=3 (TOP 4 probe): only the nearest 3 are kept and a warning says
    # how much of the requested cone they cover.
    rows = [[f"S{i}", 187.2779154 + i * 0.5 / 3600.0, 2.0523883, i * 0.5] for i in range(4)]
    reg = CatalogRegistry()
    reg._catalogs = {"crowd": tap_def("crowd", max_rows=3)}
    async with mock_client(lambda r: httpx.Response(200, json=tap_payload(rows))) as client:
        record = await CrossmatchService(reg, provider_map(client, cache=CacheManager(None))).crossmatch(
            TARGET.ra, TARGET.dec, radius_arcsec=30.0)
    res = record.catalog_results["crowd"]
    assert res["truncated"] is True and res["row_count"] == 3
    assert any("truncated at the 3 nearest rows" in w and "1.00 of the requested 30" in w for w in res["warnings"])
    assert any("truncated at the 3 nearest rows" in w for w in record.provenance["warnings"])


async def test_untruncated_result_has_no_truncation_warning() -> None:
    rows = [[f"S{i}", 187.2779154 + i * 0.5 / 3600.0, 2.0523883, i * 0.5] for i in range(3)]
    reg = CatalogRegistry()
    reg._catalogs = {"crowd": tap_def("crowd", max_rows=3)}
    async with mock_client(lambda r: httpx.Response(200, json=tap_payload(rows))) as client:
        record = await CrossmatchService(reg, provider_map(client, cache=CacheManager(None))).crossmatch(
            TARGET.ra, TARGET.dec, radius_arcsec=30.0)
    assert record.catalog_results["crowd"]["truncated"] is False
    assert record.provenance["warnings"] == []


async def test_gator_error_text_in_http_200_is_a_query_error() -> None:
    # Live Gator answers HTTP 200 text/plain for a bad column.
    text = '[struct stat="ERROR", msg="nosuchcol: column not found in catalog fp_psc"]\n'
    catalog = catalog_from_dict("g", {**DEFAULT_CATALOGS["twomass_psc"], "provider": "irsa_gator",
                                      "endpoint": "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"})
    async with mock_client(lambda r: httpx.Response(200, text=text, headers={"content-type": "text/plain"})) as client:
        with pytest.raises(CatalogQueryError, match="IRSA Gator error.*column not found"):
            await IRSAGatorProvider(client, cache=CacheManager(None)).query(catalog, TARGET, 5.0)


async def test_oracle_acos_domain_error_is_retried_with_a_nudged_centre() -> None:
    # NASA Exoplanet Archive (Oracle) failed live for Barnard's star with pm given:
    # 'ORA-01428: argument '1.00000000000000010296' is out of range'.
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(httpx.QueryParams(request.content.decode())["QUERY"])
        if len(sent) == 1:
            return httpx.Response(400, text="ORA-01428: argument '1.00000000000000010296455773204304534' is out of range",
                                  headers={"content-type": "text/plain"})
        return httpx.Response(200, json=tap_payload([["P", 187.2779154, 2.0523883, 0.0]]))

    async with mock_client(handler) as client:
        result = await TapProvider(client, cache=CacheManager(None)).query(tap_def(), TARGET, 5.0)
    assert result[0].source_id == "P" and len(sent) == 2
    import re

    circles = [re.search(r"CIRCLE\('ICRS', ([\d.]+), ([\d.-]+), ([\d.]+)\)", q).groups() for q in sent]
    (ra0, dec0, r0), (ra1, dec1, r1) = [tuple(map(float, c)) for c in circles]
    assert haversine_arcsec(ra0, dec0, ra1, dec1) == pytest.approx(0.05, abs=1e-3)
    assert (r1 - r0) * 3600.0 == pytest.approx(0.05, abs=1e-3)


# ---------------------------------------------------------------------------
# Units, positional-error specs, archive error text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["hourangle", "hour", "hr", "h", "[h]"])
def test_hour_units_are_15_degrees(unit: str) -> None:
    assert angle_to_deg(1.0, unit) == pytest.approx(15.0)


def test_unknown_words_are_not_units() -> None:
    # VOUnit's silent parser invents a custom unit for any word; it must not be used.
    assert arcsec_per_unit("foo") is None and arcsec_per_unit("hour") is None
    assert arcsec_per_unit("hourangle") == pytest.approx(54000.0)
    assert pm_to_masyr(1.0, '"/yr') == pytest.approx(1000.0)
    assert pm_to_masyr(1.0, "arcsec/a") == pytest.approx(1000.0)
    assert pm_to_masyr(2.0, "mas/yr") == 2.0


def test_divisor_applies_to_every_kind() -> None:
    sigma, details = compute_positional_error({"e": 1.0}, {"columns": ["e"], "units": "arcsec", "kind": "sigma", "divisor": 2})
    assert sigma == pytest.approx(0.5) and details["divisor"] == 2
    spec = {"columns": ["maj", "min", "peak", "rms"], "units": ["arcsec", "arcsec", None, None], "kind": "first"}
    base, _ = compute_positional_error({"maj": 6.0, "min": 5.0, "peak": 100.25, "rms": 0.2}, spec)
    halved, _ = compute_positional_error({"maj": 6.0, "min": 5.0, "peak": 100.25, "rms": 0.2},
                                         {**spec, "divisor": 1.6448536269514722 * 2})
    assert halved == pytest.approx(base / 2)


def test_scale_is_applied_after_the_systematic() -> None:
    spec = {"columns": ["x", "y"], "units": 45.0, "kind": "sigma", "systematic_arcsec": 3.0, "scale": 1.22}
    sigma, details = compute_positional_error({"x": 0.4, "y": 0.4}, spec)  # 18" statistical
    assert sigma == pytest.approx(1.22 * math.hypot(18.0, 3.0))  # 22.3", Freund et al. 2022 Eq. 2
    assert details["scale"] == 1.22
    bad = catalog_from_dict("x", {"provider": "tap", "wavelength": "x", "table": "t", "epoch": 2000.0,
                                  "parameters": {"columns": ["x", "ra", "dec"]},
                                  "pos_error": {"columns": ["x"], "units": "arcsec", "kind": "sigma", "scale": 0}})
    assert any("scale" in p for p in validate_catalog_definition(bad))


def test_simbad_quality_fallbacks_lie_inside_the_documented_ranges() -> None:
    # SIMBAD guide: A mas-level (Hipparcos/Gaia with pm), B 0.01-0.1", C 0.1-1", D 1-10", E >= 10".
    values = DEFAULT_CATALOGS["simbad"]["pos_error"]["fallback_values"]
    ranges = {"A": (0.0, 0.01), "B": (0.01, 0.1), "C": (0.1, 1.0), "D": (1.0, 10.0), "E": (10.0, 1e9)}
    for grade, (lo, hi) in ranges.items():
        assert lo <= values[grade] <= hi, grade


def test_votable_error_message_is_unescaped() -> None:
    status, message = votable_query_status((FIXTURES / "errors" / "heasarc_bad_column.0.body").read_bytes())
    assert status == "ERROR"
    assert 'column "hard_hs" does not exist' in message and "CIRCLE('ICRS'" in message
    assert "&quot;" not in message and "&apos;" not in message
    _, message = votable_query_status(b'<VOTABLE><INFO name="QUERY_STATUS" value="ERROR">ORA-00904: '
                                      b'&quot;NOSUCHCOL&quot;: invalid identifier</INFO></VOTABLE>')
    assert message == 'ORA-00904: "NOSUCHCOL": invalid identifier'


def test_linear_propagation_limits_are_documented_correctly() -> None:
    # propagate_radec's docstring quotes these rigorous-minus-linear offsets for Barnard's star.
    from astropy import units as u
    from astropy.coordinates import Distance, SkyCoord
    from astropy.time import Time

    ra, dec, pm = 269.44850252543836, 4.739420051112412, (-801.551, 10362.394)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        star = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=Distance(parallax=546.9759 * u.mas),
                        pm_ra_cosdec=pm[0] * u.mas / u.yr, pm_dec=pm[1] * u.mas / u.yr,
                        radial_velocity=-110.11 * u.km / u.s, obstime=Time(2016.0, format="jyear"))
        for epoch, expected in ((2000.0, 0.16), (1995.0, 0.28), (1991.0, 0.40)):
            rigorous = star.apply_space_motion(new_obstime=Time(epoch, format="jyear"))
            linear = propagate_radec(ra, dec, *pm, 2016.0, epoch)
            assert haversine_arcsec(rigorous.ra.deg, rigorous.dec.deg, *linear) == pytest.approx(expected, abs=0.01)
    assert "0.16" in propagate_radec.__doc__ and "well under a milliarcsecond" not in propagate_radec.__doc__


# ---------------------------------------------------------------------------
# API: counts before and after AdvancedQuery filters
# ---------------------------------------------------------------------------


def test_api_reports_rows_and_returned_count_separately() -> None:
    from fastapi.testclient import TestClient

    from api import app

    exchanges = load_exchanges("3c273", ["first", "gaia_dr3", "panstarrs_dr2"])
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(app) as client:
            response = client.post("/api/v1/search", json={
                "ra": TARGETS["3c273"][0], "dec": TARGETS["3c273"][1], "radius_arcsec": 10.0,
                "catalogs": ["first", "gaia_dr3", "panstarrs_dr2"], "min_confidence": 0.5,
            })
    assert response.status_code == 200, response.text
    results = response.json()["catalog_results"]
    # Pan-STARRS has three rows inside 10": 3C 273 itself (0.007") and two unrelated
    # objects at 6.9" and 8.8" whose posterior of being 3C 273 is ~0: all three are counted
    # in row_count, only the counterpart passes the 0.5 confidence cut.
    ps1 = results["panstarrs_dr2"]
    assert ps1["status"] == "success" and ps1["row_count"] == 3
    assert ps1["returned_count"] == len(ps1["sources"]) == 1
    assert ps1["sources"][0]["source_id"] == "110461872779253351"
    # Confidence is the Bayesian posterior (it used to be a Gaussian score): FIRST's 3C 273
    # core, 0.64" away with a 0.18" fit error and a resolved 6.5" x 5.5" structure, is the
    # radio counterpart (posterior > 0.99; the Gaussian score was 0.002 and cut it).
    first = results["first"]
    assert first["status"] == "success" and first["row_count"] == first["returned_count"] == 1
    gaia = results["gaia_dr3"]
    assert gaia["returned_count"] == len(gaia["sources"]) == 1
    matches = {m["catalog"]: m for m in response.json()["provenance"]["matches"]}
    assert matches["first"]["confidence"] > 0.99 and matches["gaia_dr3"]["confidence"] > 0.99
