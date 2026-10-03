"""Live tests of the Bayesian matching engine and the SSE stream against the real archives.

Run with ``-m live``. Assertions are astrophysical facts: 3C 273 is one object from the
radio to X-rays; Barnard's star, which moved ~3' between 2MASS and Gaia, is one object
once proper motion is applied; a crowded bulge field holds many distinct stars. Tests
skip only when an archive is unreachable (network error / HTTP 5xx / timeout).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import httpx
import pytest
from httpx_sse import ServerSentEvent
from live_policy import NETWORK_ERRORS, skip_on_network_failures

pytestmark = pytest.mark.live

THREE_C_273 = (187.2779154, 2.0523883)
# Radio, optical, infrared and X-ray catalogues that detect 3C 273.
SPEC_3C273 = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2", "sdss", "first", "nvss", "rosat", "chandra", "xmm"]
# Barnard's star: SIMBAD J2000 position and the Gaia DR3 proper motion.
BARNARD = {"ra": 269.45207696, "dec": 4.69336497, "epoch": 2000.0, "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394}
BULGE = (272.0, -27.0)


def parse_sse(lines: Iterable[str]) -> Iterator[ServerSentEvent]:
    """The events of an SSE stream, interpreted as the WHATWG spec says.

    Comment lines (``: keepalive``, sent by the server after SSE_KEEPALIVE_SECONDS without an event)
    are ignored, and a blank line dispatches nothing while the data buffer is empty. httpx_sse's
    decoder instead dispatches an empty 'message' event carrying the last id for the blank line after a
    comment, so under load a keepalive shows up as a spurious event with a repeated id.
    """
    event_type, data, last_id, retry = "", None, "", None
    for line in lines:
        if not line:
            if data is not None:  # the spec's "data buffer is not empty": at least one data field
                yield ServerSentEvent(event=event_type or "message", data="\n".join(data), id=last_id, retry=retry)
            event_type, data, retry = "", None, None  # the last event id persists across events
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_type = value
        elif field == "data":
            data = [*(data or []), value]
        elif field == "id" and "\0" not in value:
            last_id = value
        elif field == "retry" and value.isdigit():
            retry = int(value)


def sse_events(source) -> Iterator[ServerSentEvent]:
    """The spec-interpreted events of an ``httpx_sse`` EventSource (see :func:`parse_sse`)."""
    return parse_sse(source.response.iter_lines())


def test_parse_sse_ignores_keepalive_comments() -> None:
    raw = ["id: 1", "event: start", 'data: {"a": 1}', "", ": keepalive", "", "id: 2", "event: catalog",
           "data: x", "data: y", "", ": keepalive", "", "id: 3", "event: done", "data: {}", ""]
    events = list(parse_sse(raw))
    assert [(e.event, e.id, e.data) for e in events] == [
        ("start", "1", '{"a": 1}'), ("catalog", "2", "x\ny"), ("done", "3", "{}")]


def skip_on_network_catalog_events(catalog_events: list[dict]) -> None:
    """The streamed 'catalog' events' counterpart of :func:`skip_on_network_failures`."""
    failed = [c for c in catalog_events if c["status"] == "failed"]
    down = [f"{c['catalog']}: {c['error_type']}" for c in failed if c.get("error_type") in NETWORK_ERRORS]
    if down:
        pytest.skip("archive unreachable: " + "; ".join(down))
    assert not failed, failed


async def live_service(client: httpx.AsyncClient):
    from main import build_service

    return build_service(client=client)


async def test_live_3c273_is_one_object_across_the_spectrum() -> None:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(*THREE_C_273, radius_arcsec=10.0)
    skip_on_network_failures(record, SPEC_3C273)
    group = record.crossmatch_groups[0]
    assert group["contains_target"] and group["match_flag"] == "best"
    assert group["p_any"] > 0.95
    assert set(SPEC_3C273) <= set(group["catalogs"]), group["catalogs"]
    members = {m["catalog"]: m for m in group["members"] if m["coincident_with"] is None}
    assert len(members) == len([m for m in group["members"] if m["coincident_with"] is None])
    for name in SPEC_3C273:
        assert members[name]["match_probability"] > 0.95, (name, members[name]["match_probability"])
        assert members[name]["confidence"] > 0.95
    # The identity rows: SIMBAD '3C 273' and NED '3C 273' (a quasar at z = 0.158).
    assert members["simbad"]["source_id"] == "3C 273"
    assert members["ned"]["physical"]["redshift"] == pytest.approx(0.158, abs=0.002)
    # No other group claims the quasar.
    assert all(not g["contains_target"] for g in record.crossmatch_groups[1:])


async def test_live_barnards_star_grouped_despite_three_arcminutes_of_motion() -> None:
    catalogs = ["gaia_dr3", "twomass_psc", "allwise", "simbad"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(BARNARD["ra"], BARNARD["dec"], radius_arcsec=10.0, epoch=BARNARD["epoch"],
                                          pm_ra_masyr=BARNARD["pm_ra_masyr"], pm_dec_masyr=BARNARD["pm_dec_masyr"],
                                          catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    group = record.crossmatch_groups[0]
    assert group["contains_target"] and set(catalogs) <= set(group["catalogs"])
    assert group["p_any"] > 0.95
    members = {m["catalog"]: m for m in group["members"] if m["coincident_with"] is None}
    for name in catalogs:
        assert members[name]["match_probability"] > 0.95, (name, members[name]["match_probability"])
    assert members["gaia_dr3"]["source_id"] == "4472832130942575872"
    assert members["simbad"]["source_id"] == "NAME Barnard's star"
    assert members["twomass_psc"]["source_id"] == "17574849+0441405"
    # Gaia saw the star 16 yr after J2000: 166" from the J2000 position before propagation.
    assert members["gaia_dr3"]["metadata"]["query_separation_arcsec"] > 150.0
    assert members["gaia_dr3"]["separation_arcsec"] < 0.1


async def test_live_crowded_bulge_field_gives_distinct_objects() -> None:
    catalogs = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(*BULGE, radius_arcsec=20.0, catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    assert record.catalog_results["gaia_dr3"]["row_count"] >= 60
    groups = record.crossmatch_groups
    multi = [g for g in groups if sum(1 for m in g["members"] if m["coincident_with"] is None) >= 2]
    assert len(multi) >= 20
    for group in groups:
        independent = [m["catalog"] for m in group["members"] if m["coincident_with"] is None]
        assert len(independent) == len(set(independent)), group["catalogs"]  # no double-catalogue members
    ids = [(m["catalog"], m["source_id"]) for g in groups for m in g["members"]]
    assert len(ids) == len(set(ids))


def _stream_app():
    from fastapi import FastAPI

    import streaming

    app = FastAPI()
    app.include_router(streaming.router)  # no app.state.service: the route builds its own
    return app


def test_live_sse_stream_for_3c273() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    catalogs = ["gaia_dr3", "simbad", "twomass_psc", "first", "nvss", "rosat"]
    arrivals: list[tuple[float, str]] = []
    with TestClient(_stream_app()) as client:
        started = time.perf_counter()
        with connect_sse(client, "GET", "/api/v1/search/stream", params={
                "ra": THREE_C_273[0], "dec": THREE_C_273[1], "radius_arcsec": 10.0,
                "catalogs": ",".join(catalogs)}, timeout=180.0) as source:
            assert source.response.status_code == 200
            events = []
            for event in sse_events(source):
                arrivals.append((time.perf_counter() - started, event.event))
                events.append(event)
    kinds = [e.event for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("catalog") == len(catalogs)
    assert [int(e.id) for e in events] == list(range(1, len(events) + 1))
    catalog_events = [e.json() for e in events if e.event == "catalog"]
    skip_on_network_catalog_events(catalog_events)
    record = events[-1].json()["record"]
    group = record["crossmatch_groups"][0]
    assert group["contains_target"] and set(catalogs) <= set(group["catalogs"])
    first_catalog = next(t for t, kind in arrivals if kind == "catalog")
    assert first_catalog < arrivals[-1][0]
    # Every catalogue event carries rows and timing.
    for data in catalog_events:
        assert data["count"] == len(data["sources"]) >= 1 and data["elapsed_ms"] > 0


def test_live_sse_stream_by_name() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    with TestClient(_stream_app()) as client, connect_sse(client, "GET", "/api/v1/search/stream",
                     params={"name": "3C 273", "radius_arcsec": 5.0, "catalogs": "gaia_dr3,simbad"},
                     timeout=180.0) as source:
        if source.response.status_code >= 500:
            pytest.skip(f"upstream error {source.response.status_code}")
        assert source.response.status_code == 200
        events = list(sse_events(source))
    start = events[0].json()
    assert start["resolved_object"]["canonical_name"].replace(" ", "") == "3C273"
    assert start["target"]["ra"] == pytest.approx(THREE_C_273[0], abs=1e-4)
    record = events[-1].json()["record"]
    skip_on_network_failures(record, ["gaia_dr3", "simbad"])
    group = record["crossmatch_groups"][0]
    assert group["contains_target"] and {"gaia_dr3", "simbad"} <= set(group["catalogs"])
    # The resolver's own position error (errRAmas/errDEmas) replaced the default target sigma.
    assert record["provenance"]["association"]["target_sigma_arcsec"] < 0.01


# ---------------------------------------------------------------------------
# Regressions of the round-1 review (live archives)
# ---------------------------------------------------------------------------


async def test_live_undated_simbad_coordinates_identify_tau_ceti_through_the_search_filter() -> None:
    from crossmatch import AdvancedQuery

    ra, dec = 26.01701307, -15.93747989  # SIMBAD J2000, no epoch given
    catalogs = ["simbad", "twomass_psc"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        query = AdvancedQuery.from_dict({"ra": ra, "dec": dec, "radius_arcsec": 10.0, "catalogs": catalogs,
                                         "min_confidence": 0.5})
        record = await service.crossmatch(ra, dec, query=query)
    skip_on_network_failures(record, catalogs)
    confidences = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}
    assert confidences.get("* tau Cet", 0.0) > 0.95, confidences
    assert record.catalog_results["simbad"]["returned_count"] >= 1


async def test_live_gaia_field_star_does_not_take_an_unrelated_nvss_source() -> None:
    # Gaia DR3 629559574718166784 (G = 15.7, parallax 1.04 mas) and NVSS J100535+215127
    # 5.2" away (major axis an upper limit).
    catalogs = ["gaia_dr3", "nvss"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(151.3953787449629, 21.85844960158126, radius_arcsec=35.0, epoch=2016.0,
                                          catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    rows = {m["source_id"]: m for g in record.crossmatch_groups for m in g["members"]}
    assert rows["629559574718166784"]["target_probability"] > 0.99
    assert rows["NVSS J100535+215127"]["target_probability"] < 0.01
    assert record.provenance["association"]["target_class"]["class"] == "star"


async def test_live_baade_window_density_at_the_default_radius() -> None:
    from models import parse_json_records

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(272.0, -27.0, radius_arcsec=3.0, catalogs=["gaia_dr3"],
                                          target_uncertainty_arcsec=0.1)
        skip_on_network_failures(record, ["gaia_dr3"])
        query = ("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                 "CIRCLE('ICRS', 272.0, -27.0, 0.0166666667))")
        try:
            response = await client.post("https://gea.esac.esa.int/tap-server/tap/sync",
                                         data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query})
        except httpx.HTTPError as exc:
            pytest.skip(f"Gaia archive unreachable: {exc}")
        if response.status_code >= 500:
            pytest.skip(f"Gaia archive error {response.status_code}")
    count = parse_json_records(response.content)[0]["n"]
    import math

    truth = count / (math.pi * (1 / 60.0) ** 2)  # per deg^2 within 60"
    density = record.provenance["association"]["densities"]["gaia_dr3"]["density_per_deg2"]
    assert truth / 2.0 < density < truth * 2.0, (density, truth)


async def test_live_quasar_allwise_counterparts_several_sigma_off_are_identified() -> None:
    catalogs = ["allwise", "gaia_dr3", "simbad"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(194.04652741, -5.78931254, radius_arcsec=5.0, catalogs=catalogs)  # 3C 279
    skip_on_network_failures(record, catalogs)
    allwise = [m for m in record.provenance["matches"] if m["catalog"] == "allwise"]
    assert allwise and allwise[0]["confidence"] > 0.9  # J125611.15-054721.7, 0.24" off (was 6e-5)


async def test_live_rounded_3c273_coordinates_identify_the_quasar() -> None:
    catalogs = ["simbad", "gaia_dr3", "panstarrs_dr2"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        # The coordinates as typed (text): their rounding sets the target uncertainty.
        record = await service.crossmatch("187.278", "2.052", radius_arcsec=10.0, catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    confidences = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}
    assert confidences["3C 273"] > 0.95 and confidences["3700386905605055360"] > 0.95
    assert record.provenance["association"]["target_sigma_source"] == "coordinate_precision"


def test_live_sse_stream_kruger60a_by_name_finds_its_two_parameter_gaia_row() -> None:
    from fastapi.testclient import TestClient
    from httpx_sse import connect_sse

    with TestClient(_stream_app()) as client, connect_sse(
            client, "GET", "/api/v1/search/stream",
            params={"name": "GJ 860 A", "radius_arcsec": 16.0, "catalogs": "simbad,gaia_dr3"},
            timeout=180.0) as source:
        if source.response.status_code >= 500:
            pytest.skip(f"upstream error {source.response.status_code}")
        assert source.response.status_code == 200
        events = list(sse_events(source))
    start, record = events[0].json(), events[-1].json()["record"]
    skip_on_network_failures(record, ["simbad", "gaia_dr3"])
    from providers import SesameResolver

    resolver = (start["resolved_object"]["resolver_metadata"] or {})
    if SesameResolver.answer_kind(resolver) != "simbad":
        # SIMBAD did not answer Sesame (even when retried alone): the VizieR fallback is an
        # undated catalogue position 50" from Kruger 60 A. An upstream outage, not a regression
        # -- but the record must say so.
        assert any("not SIMBAD" in w for w in record["provenance"]["warnings"]), record["provenance"]["warnings"]
        pytest.skip(f"SIMBAD unavailable through Sesame: answered by {resolver.get('resolver_name')} "
                    f"(retry: {resolver.get('simbad_retry')})")
    rows = {m["source_id"]: m for g in record["crossmatch_groups"] for m in g["members"]}
    assert rows["2007876324466455040"]["target_probability"] > 0.9  # Kruger 60 A (was 0.0)


# ---------------------------------------------------------------------------
# Regressions of the round-2 review (live archives)
# ---------------------------------------------------------------------------


async def _live_stream(name: str, radius: float, catalogs: list[str]) -> dict:
    from models import ResolverUnavailableError

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        service = await live_service(client)
        try:
            events = [e async for e in service.crossmatch_stream(name=name, radius_arcsec=radius, catalogs=catalogs)]
        except ResolverUnavailableError as exc:  # the name resolver is an archive too
            pytest.skip(f"Sesame unreachable: {exc}")
    record = events[-1]["data"]["record"]
    skip_on_network_failures(record, catalogs)
    return record


def _rows(record: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in (m for g in record["crossmatch_groups"] for m in g["members"]):
        if m["source_id"] not in out or m["catalog"] != "ned":
            out[m["source_id"]] = m
    return out


async def test_live_named_3c273_keeps_its_radio_and_x_ray_counterparts() -> None:
    record = await _live_stream("3C 273", 5.0, ["simbad", "gaia_dr3", "first", "rosat", "chandra", "xmm"])
    assert record["provenance"]["association"]["target_class"]["class"] == "extragalactic"
    assert record["provenance"]["target_parallax"] is None
    probability = {m["catalog"]: m["target_probability"] for g in record["crossmatch_groups"] for m in g["members"]
                   if g["contains_target"]}
    for name in ("first", "rosat", "chandra", "xmm", "simbad", "gaia_dr3"):
        assert probability.get(name, 0.0) > 0.95, (name, probability)


async def test_live_named_x_ray_binary_keeps_its_radio_counterparts() -> None:
    record = await _live_stream("SS 433", 5.0, ["simbad", "nvss", "vlass"])
    assert record["provenance"]["association"]["target_class"]["class"] == "unknown"
    probability = {m["catalog"]: m["target_probability"] for g in record["crossmatch_groups"] for m in g["members"]
                   if g["contains_target"]}
    assert probability.get("nvss", 0.0) > 0.95 and probability.get("vlass", 0.0) > 0.95, probability


async def test_live_named_crab_nebula_keeps_its_identity_row() -> None:
    record = await _live_stream("Crab Nebula", 10.0, ["simbad", "gaia_dr3", "twomass_psc"])
    assert record["provenance"]["association"]["target_class"]["class"] == "extended"
    rows = _rows(record)
    assert rows["M 1"]["target_probability"] > 0.9
    assert all(m["target_probability"] < 0.1 for m in rows.values() if m["catalog"] in ("gaia_dr3", "twomass_psc"))


async def test_live_m13_core_posterior_does_not_depend_on_the_radius() -> None:
    records = []
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        service = await live_service(client)
        for radius in (3.0, 30.0):
            record = await service.crossmatch(250.4237, 36.4614, radius_arcsec=radius, catalogs=["gaia_dr3"],
                                              target_uncertainty_arcsec=0.5)
            skip_on_network_failures(record, ["gaia_dr3"])
            records.append(record)
    small, large = (r.provenance["association"]["densities"]["gaia_dr3"] for r in records)
    if (small.get("probe") or {}).get("status") == "failed":
        error = small["probe"]["error"]
        if any(kind in error for kind in NETWORK_ERRORS):
            pytest.skip(f"density probe unreachable: {error}")
        raise AssertionError(error)
    assert small["density_per_deg2"] > 1.0e6 and small["density_per_deg2"] == pytest.approx(
        large["density_per_deg2"], rel=0.2)
    p = [{m["source_id"]: m["confidence"] for m in r.provenance["matches"]} for r in records]
    common = set(p[0]) & set(p[1])
    assert common and max(abs(p[0][k] - p[1][k]) for k in common) < 0.05


async def test_live_undated_fast_mover_warns_about_rows_outside_the_cone() -> None:
    catalogs = ["simbad", "twomass_psc"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(53.232685, -9.458261, radius_arcsec=3.0, catalogs=catalogs)  # eps Eri J2000
    skip_on_network_failures(record, catalogs)
    assert any(w.startswith("Undated target") and "epoch=2000" in w for w in record.provenance["warnings"])
    twomass = [m for m in record.provenance["matches"] if m["catalog"] == "twomass_psc"]
    assert twomass and twomass[0]["confidence"] > 0.95


async def test_live_sexagesimal_3c273_as_typed_identifies_the_quasar() -> None:
    catalogs = ["simbad", "gaia_dr3", "panstarrs_dr2"]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch("12 29 07", "+02 03 09", radius_arcsec=10.0, catalogs=catalogs)
    skip_on_network_failures(record, catalogs)
    confidences = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}
    assert confidences["3C 273"] > 0.9 and confidences["3700386905605055360"] > 0.9
    assert confidences.get("[CME2001] 3C 273 1", 0.0) < 0.1


# ---------------------------------------------------------------------------
# Regressions of the round-3 review (live archives)
# ---------------------------------------------------------------------------


async def test_live_named_coma_cluster_is_its_identity() -> None:
    record = await _live_stream("Coma Cluster", 20.0, ["simbad", "ned", "gaia_dr3", "chandra"])
    rows = {(m["catalog"], m["source_id"]): m for g in record["crossmatch_groups"] for m in g["members"]}
    assert rows[("simbad", "ACO 1656")]["target_probability"] > 0.99  # was 0.09
    assert all(m["target_probability"] < 0.1 for (catalog, _), m in rows.items() if catalog in ("gaia_dr3", "chandra"))
    group = next(g for g in record["crossmatch_groups"] if g["contains_target"])
    assert set(group["catalogs"]) <= {"simbad", "ned"}


async def test_live_named_cas_a_simbad_and_ned_rows_are_one_object() -> None:
    record = await _live_stream("Cas A", 20.0, ["simbad", "ned"])
    rows = {(m["catalog"], m["source_id"]): m for g in record["crossmatch_groups"] for m in g["members"]}
    assert rows[("simbad", "NAME Cas A")]["target_probability"] > 0.99
    assert rows[("ned", "Cas A")]["target_probability"] > 0.99  # was 0.001


async def test_live_proxima_xmm_detections_along_its_track_are_its_counterparts() -> None:
    record = await _live_stream("Proxima Centauri", 10.0, ["simbad", "xmm"])
    xmm = [m for g in record["crossmatch_groups"] for m in g["members"] if m["catalog"] == "xmm"
           and m["target_probability"] > 0.1]
    assert len(xmm) >= 3 and all(m["target_probability"] > 0.8 for m in xmm)  # were 0.37 / 0.36 / 0.27


async def test_live_47_tuc_core_is_probed_at_3_arcsec() -> None:
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        service = await live_service(client)
        record = await service.crossmatch(6.0236, -72.0813, radius_arcsec=3.0, catalogs=["gaia_dr3"])
    skip_on_network_failures(record, ["gaia_dr3"])
    probe = record.provenance["association"]["densities"]["gaia_dr3"]["probe"]
    if probe["status"] == "failed" and any(kind in probe["error"] for kind in NETWORK_ERRORS):
        pytest.skip(f"density probe unreachable: {probe['error']}")
    assert probe["status"] == "success" and probe["reason"] == "crowded region NGC 104"
    assert record.provenance["association"]["densities"]["gaia_dr3"]["density_per_deg2"] > 3.0e5
