"""Offline tests for alerts.py: recorded ALeRCE / Fink (ZTF, LSST) answers and crossmatch queries.

Fixtures are real responses recorded by ``tests/fixtures/alerts/record_alerts.py`` and
replayed strictly (the code must send the recorded URL, query and body). A few logic
tests (paging, flag rules, store semantics) use small synthetic inputs; they say so.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self
from urllib.parse import unquote_plus

import httpx
import pytest
import respx
from astropy import units as u
from astropy.coordinates import SkyCoord
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client

import alerts
from alerts import (
    FINK_ZTF_CLASS_SCORES,
    HYPERLEDA,
    NJY_AB_ZERO_POINT_MAG,
    AlerceBroker,
    Alert,
    AlertEnricher,
    AlertEnrichment,
    AlertService,
    AlertStore,
    BrokerError,
    FetchResult,
    FinkLSSTBroker,
    FinkZTFBroker,
    HostSearchRegistry,
    PollRequest,
    d25_ellipse,
    directional_light_radius,
    fetch_alerts,
    group_host_candidates,
    is_transient_designation,
    mjd_to_iso,
    negative_from_isdiffpos,
    njy_to_ab_mag,
    position_angle_deg,
    projected_offset_kpc,
    register_cli,
    router,
    tai_mjd_to_utc,
    utc_mjd_to_tai,
)
from datasets import MetadataStore
from models import UnifiedRecord

FIX = Path(__file__).resolve().parent / "fixtures" / "alerts"
BROKER_SCENARIOS = {"alerce": "alerce", "fink": "fink_ztf", "fink_lsst": "fink_lsst"}


def meta(name: str) -> dict[str, Any]:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def body(name: str, idx: int) -> Any:
    return json.loads((FIX / f"{name}.{idx}.body").read_bytes())


def params(name: str) -> dict[str, Any]:
    return meta(name)["params"]


class Replay:
    """Strict respx replay of alert fixtures that also counts the requests served."""

    def __init__(self, *names: str) -> None:
        self.calls: list[httpx.Request] = []
        handler = replay_side_effect(load_exchanges("alerts", list(names)))

        def side_effect(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return handler(request)

        self.router = respx.mock(assert_all_called=False, assert_all_mocked=True)
        self.router.route().mock(side_effect=side_effect)

    def __enter__(self) -> Self:
        self.router.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.router.__exit__(*exc)


@pytest.fixture(autouse=True)
def _fresh_class_lists() -> None:
    """Each test sees the brokers' class lists afresh (they are cached for hours in production)."""
    alerts.clear_class_list_cache()


@pytest.fixture
def store(tmp_path: Path) -> AlertStore:
    return AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'alerts.sqlite3').as_posix()}"))


async def fetch_recorded(broker: str) -> FetchResult:
    p = params(BROKER_SCENARIOS[broker])
    with Replay(BROKER_SCENARIOS[broker]):
        async with offline_client() as client:
            return await fetch_alerts(client, broker, since_mjd=p["since_mjd"], until_mjd=p["until_mjd"],
                                      limit=p["limit"], options=p["options"])


def recorded_alerts(scenario: str) -> list[Alert]:
    return [Alert.from_dict(a) for a in params(scenario)["alerts"]]


def famous_alert(name: str) -> Alert:
    return next(a for a in recorded_alerts("xmatch_famous") if a.object_id == name)


def names_of(host: dict[str, Any]) -> set[str]:
    return {host["name"], *(a.split(":", 1)[1] for a in host["aliases"])}


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_ab_zero_point_matches_astropy() -> None:
    assert NJY_AB_ZERO_POINT_MAG == pytest.approx(31.4, abs=1e-12)
    for flux in (1.0, 3021.5986, 1e6):
        mag, err = njy_to_ab_mag(flux, flux / 10.0)
        assert mag == pytest.approx((flux * u.nJy).to(u.ABmag).value, abs=1e-6)
        assert err == pytest.approx(2.5 / math.log(10) / 10.0)
    assert njy_to_ab_mag(3631e9)[0] == pytest.approx(0.0, abs=1e-3)  # 3631 Jy ~ AB 0
    assert njy_to_ab_mag(-5.0) == (None, None)
    assert njy_to_ab_mag(None) == (None, None)


def test_time_conversions() -> None:
    assert mjd_to_iso(51544.5) == "2000-01-01 12:00:00"
    assert mjd_to_iso(61311.3123) == "2026-09-28 07:29:42"
    assert mjd_to_iso(61311.3123, round_up=True) == "2026-09-28 07:29:43"
    assert mjd_to_iso(61311.5, round_up=True) == mjd_to_iso(61311.5) == "2026-09-28 12:00:00"
    # TAI - UTC = 37 s since 2017-01-01.
    assert (61235.4127394704 - tai_mjd_to_utc(61235.4127394704)) * 86400.0 == pytest.approx(37.0, abs=1e-4)
    assert (utc_mjd_to_tai(61300.0) - 61300.0) * 86400.0 == pytest.approx(37.0, abs=1e-4)


def test_isdiffpos_sign() -> None:
    # ZTF alert packets: 't'/'1' positive subtraction, 'f'/'0' negative; ALeRCE uses 1 / -1.
    assert [negative_from_isdiffpos(v) for v in ("t", "1", 1, True)] == [False] * 4
    assert [negative_from_isdiffpos(v) for v in ("f", "0", -1, 0, False)] == [True] * 5
    assert negative_from_isdiffpos(None) is None and negative_from_isdiffpos("x") is None


# ---------------------------------------------------------------------------
# Broker parsing (recorded answers)
# ---------------------------------------------------------------------------


async def test_alerce_fetch_parses_objects_last_detection_and_truncation() -> None:
    p = params("alerce")
    result = await fetch_recorded("alerce")
    objects = body("alerce", 0)["items"]
    # limit + 1 objects requested: a 4th object exists, so the window is reported as truncated.
    assert p["limit"] == 3 and len(objects) == 4 and result.truncated and result.warnings == []
    found = result.alerts
    assert [a.object_id for a in found] == [o["oid"] for o in objects[:3]]
    assert result.boundary_mjd == objects[2]["firstmjd"] and objects[3]["firstmjd"] <= result.boundary_mjd
    for idx, (alert, obj) in enumerate(zip(found, objects, strict=False), start=1):
        assert alert.broker == "alerce" and alert.survey == "ztf" and alert.object_id.startswith("ZTF")
        assert (alert.ra, alert.dec) == (obj["meanra"], obj["meandec"])
        assert alert.mjd == obj["lastmjd"] and alert.first_mjd == obj["firstmjd"]
        # firstmjd window of the query (stamp classifier: brand-new objects).
        assert p["since_mjd"] <= alert.first_mjd <= p["until_mjd"] and alert.first_mjd <= alert.mjd
        assert alert.classification == "SN" and alert.extra["classifier"] == "stamp_classifier"
        assert 0.0 < alert.probability <= 1.0
        assert alert.url == f"https://alerce.online/object/{alert.object_id}"
        # Independent check of the photometry: the latest detection of /detections.
        latest = max(body("alerce", idx), key=lambda d: d["mjd"])
        assert alert.magpsf == latest["magpsf"] and alert.magpsf_err == latest["sigmapsf"]
        assert alert.band == {1: "g", 2: "r", 3: "i"}[latest["fid"]]
        assert alert.is_negative is (latest["isdiffpos"] == -1)
        assert alert.extra["candid"] == str(latest["candid"]) and alert.mjd == pytest.approx(latest["mjd"], abs=1e-3)
        assert 12.0 < alert.magpsf < 22.5  # ZTF single-epoch depth ~20.5-21


def test_alerce_detections_keep_the_sign_of_the_latest_detection() -> None:
    # Synthetic detections: the latest one is a negative subtraction (isdiffpos -1).
    alert = Alert("alerce", "ZTF18abablcx", 1.0, 1.0, 61309.19, None, None, None, None, "")
    detections = [
        {"mjd": 61306.2, "fid": 1, "magpsf": 18.41, "sigmapsf": 0.1, "isdiffpos": 1, "candid": 1},
        {"mjd": 61309.19, "fid": 2, "magpsf": 18.40, "sigmapsf": 0.11, "isdiffpos": -1, "candid": 2},
    ]
    AlerceBroker.apply_detections(alert, detections)
    assert (alert.magpsf, alert.band, alert.is_negative) == (18.40, "r", True)
    assert alert.extra["n_negative_detections"] == 1 and alert.extra["bands"]["g"]["last_is_negative"] is False
    with pytest.raises(BrokerError):
        AlerceBroker.apply_detections(alert, {"message": "Internal Server Error"})


async def test_fink_ztf_fetch_applies_fink_sn_definition() -> None:
    p = params("fink_ztf")
    result = await fetch_recorded("fink")
    rows = body("fink_ztf", 0)
    # n = limit + 1 = 4 rows, 4 objects: the newest 3 are kept and the window is truncated.
    assert len(rows) == 4 == len({r["i:objectId"] for r in rows}) and result.truncated
    found = result.alerts
    assert [a.object_id for a in found] == [r["i:objectId"] for r in rows[:3]]
    assert result.boundary_mjd == pytest.approx(rows[2]["i:jd"] - 2400000.5)
    for alert in found:
        row = next(r for r in rows if r["i:objectId"] == alert.object_id)
        assert alert.mjd == pytest.approx(row["i:jd"] - 2400000.5, abs=1e-9)
        assert p["since_mjd"] <= alert.mjd <= p["until_mjd"]
        assert alert.first_mjd == pytest.approx(row["i:jdstarthist"] - 2400000.5) and alert.first_mjd <= alert.mjd
        assert alert.classification == "SN candidate"
        # Fink's SN-candidate filter requires snn_snia_vs_nonia > 0.5 or snn_sn_vs_all > 0.5 ...
        assert max(row[c] for c in FINK_ZTF_CLASS_SCORES["SN candidate"]) > 0.5
        # ... but P(SN) is SuperNNova's SN-vs-all score; the Ia-vs-non-Ia score (a subclass score
        # among SNe) is kept in extra['scores'] only.
        assert alert.probability == row["d:snn_sn_vs_all"] and alert.extra["probability_score"] == "snn_sn_vs_all"
        assert alert.extra["scores"]["snn_snia_vs_nonia"] == row["d:snn_snia_vs_nonia"]
        if alert.object_id == "ZTF26abtnjsl":  # selected by its Ia-vs-non-Ia score alone
            assert alert.probability == pytest.approx(0.2504, abs=1e-4)
            assert row["d:snn_snia_vs_nonia"] == pytest.approx(0.755, abs=1e-3)
        assert alert.band == {1: "g", 2: "r", 3: "i"}[row["i:fid"]]
        assert alert.magpsf == row["i:magpsf"] and alert.magpsf_err == row["i:sigmapsf"]
        assert alert.is_negative is (row["i:isdiffpos"] == "f")
        assert alert.extra["candid"] == str(row["i:candid"])
        assert alert.url == f"https://ztf.fink-portal.org/{alert.object_id}"
    # The TNS classification (d:tns) and the Mangrove host hint are carried through; the TNS
    # class is 'tns_type' (as for Fink/LSST), 'tns' is reserved for TNS names.
    snia = next(a for a in found if a.extra["tns_type"] == "SN Ia")
    assert snia.extra["mangrove_hyperleda_name"] == "PGC010173"
    assert all("tns" not in a.extra for a in found)


async def test_fink_lsst_fetch_converts_flux_time_and_keeps_64bit_ids() -> None:
    p = params("fink_lsst")
    result = await fetch_recorded("fink_lsst")
    text = (FIX / "fink_lsst.0.body").read_text(encoding="utf-8")
    rows = json.loads(text)
    request = meta("fink_lsst")["exchanges"][0]["url"]
    # The UTC window is sent in TAI (Fink compares it with midpointMjdTai): +37 s.
    assert "startdate=2026-06-09+00%3A00%3A37" in request
    assert len(result.alerts) == 3 and result.truncated
    for alert in result.alerts:
        # diaObjectIds exceed 2**53: kept exactly as the digits sent by Fink.
        assert f'"r:diaObjectId":{alert.object_id}' in text and int(alert.object_id) > 2**53
        row = next(r for r in rows if str(r["r:diaObjectId"]) == alert.object_id)
        assert alert.survey == "lsst" and alert.band in {"u", "g", "r", "i", "z", "y"}
        assert alert.magpsf == pytest.approx((abs(row["r:psfFlux"]) * u.nJy).to(u.ABmag).value, abs=1e-6)
        assert alert.is_negative is bool(row["r:isNegative"]) and (row["r:psfFlux"] < 0) is alert.is_negative
        assert (row["r:midpointMjdTai"] - alert.mjd) * 86400.0 == pytest.approx(37.0, abs=1e-3)
        assert p["since_mjd"] <= alert.mjd <= p["until_mjd"]
        assert alert.dec < 35.0  # Rubin (Cerro Pachon, latitude -30.2 deg) observes the southern sky
        assert alert.classification == "SN-like" and alert.probability == row["f:clf_cats_score"]
        assert alert.extra["tag"] == "extragalactic_new_candidate"
        assert alert.extra["gaia_dr3_name"] is None  # Fink's 'Fail' placeholder
        assert alert.url == f"https://lsst.fink-portal.org/{alert.object_id}"


def test_lsst_negative_flux_keeps_a_magnitude_and_the_sign() -> None:
    # Synthetic row: a negative difference flux gets |flux| magnitude and is_negative True (ZTF convention).
    row = {"r:diaObjectId": 170657518965489670, "r:ra": 10.0, "r:dec": -30.0, "r:midpointMjdTai": 61235.4,
           "r:band": "r", "r:psfFlux": -2000.0, "r:psfFluxErr": 100.0, "r:isNegative": True}
    (alert,), _ = FinkLSSTBroker.parse_tags([row], "extragalactic_new_candidate")
    assert alert.is_negative is True and alert.magpsf == pytest.approx(31.4 - 2.5 * math.log10(2000.0))


@pytest.mark.parametrize(("scenario", "broker", "match"), [
    ("invalid_alerce", "alerce", "(?i)unknown"), ("invalid_fink_ztf", "fink", "(?i)unknown"),
    ("invalid_fink_lsst", "fink_lsst", "not a valid tag"),
    ("unsupported_fink_lsst", "fink_lsst", "only available from the Livestream"),
])
async def test_unknown_or_unsupported_class_is_a_value_error(scenario: str, broker: str, match: str) -> None:
    p = params(scenario)
    with Replay(scenario):
        async with offline_client() as client:
            with pytest.raises(ValueError, match=match):
                await fetch_alerts(client, broker, since_mjd=p["since_mjd"], until_mjd=p["until_mjd"], limit=2,
                                   options=p["options"])


async def test_fink_class_validation_accepts_bare_simbad_names() -> None:
    classes = body("invalid_fink_ztf", 1)  # the real /classes answer
    assert "(SIMBAD) RRLyrae" in {c for g in classes.values() for c in g} and "RRLyrae" not in classes
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(200, json=[]))
        mock.get("https://api.ztf.fink-portal.org/api/v1/classes").mock(return_value=httpx.Response(200, json=classes))
        async with offline_client() as client:
            quiet = await FinkZTFBroker().fetch(client, since_mjd=61311.0, until_mjd=61311.2, class_name="RRLyrae")
            assert quiet.alerts == [] and not quiet.truncated
            await FinkZTFBroker().fetch(client, since_mjd=61311.0, until_mjd=61311.2, class_name="(SIMBAD) RRLyrae")
            with pytest.raises(ValueError, match="Unknown Fink/ZTF class"):
                await FinkZTFBroker().fetch(client, since_mjd=61311.0, until_mjd=61311.2, class_name="RRLyraeX")


async def test_broker_upstream_failures_raise_broker_error() -> None:
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get(url__startswith="https://api.alerce.online/").mock(return_value=httpx.Response(503, text="maintenance"))
        mock.get(url__startswith="https://api.ztf.fink-portal.org/").mock(return_value=httpx.Response(200, text="<html>"))
        mock.get(url__startswith="https://api.lsst.fink-portal.org/").mock(side_effect=httpx.ConnectTimeout("timed out"))
        async with offline_client() as client:
            with pytest.raises(BrokerError) as err:
                await AlerceBroker().fetch(client, since_mjd=61300.0, until_mjd=61301.0)
            assert err.value.status_code == 503 and err.value.unreachable
            with pytest.raises(BrokerError, match="non-JSON") as err:
                await FinkZTFBroker().fetch(client, since_mjd=61300.0, until_mjd=61301.0)
            assert not err.value.unreachable  # a broken answer is a failure, not an outage
            with pytest.raises(BrokerError, match="network error") as err:
                await FinkLSSTBroker().fetch(client, since_mjd=61300.0, until_mjd=61301.0)
            assert err.value.unreachable
    for parse in (lambda: AlerceBroker.parse_objects({"message": "Internal Server Error"}),
                  lambda: AlerceBroker.parse_objects({"results": []}),
                  lambda: FinkZTFBroker.parse_latests({"message": "Internal Server Error"}, "SN candidate"),
                  lambda: FinkLSSTBroker.parse_tags({"detail": "x"}, "t")):
        with pytest.raises(BrokerError) as err:
            parse()
        assert not err.value.unreachable and err.value.status_code is None


def test_parsers_skip_rows_without_position() -> None:
    alerts_, warnings = FinkZTFBroker.parse_latests(
        [{"i:objectId": "ZTF00aaaaaaa", "i:ra": None, "i:dec": 10.0, "i:jd": 2461300.5}], "SN candidate"
    )
    assert alerts_ == [] and len(warnings) == 1
    alerts_, warnings = AlerceBroker.parse_objects({"items": [{"oid": "ZTF00aaaaaab", "meanra": 400.0, "meandec": 0.0,
                                                                "lastmjd": 61300.0}]})
    assert alerts_ == [] and len(warnings) == 1


def test_unsupported_options_and_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        AlerceBroker().object_params(since_mjd=1, until_mjd=2, limit=1, mjd_field="jd")
    with pytest.raises(ValueError):
        alerts.get_broker("antares")
    with pytest.raises(ValueError):
        asyncio.run(fetch_alerts(offline_client(), "fink", since_mjd=1, until_mjd=2, options={"classifier": "x"}))
    for bad in (0, alerts.MAX_POLL_LIMIT + 1):
        with pytest.raises(ValueError, match="limit"):
            asyncio.run(fetch_alerts(offline_client(), "fink", since_mjd=1, until_mjd=2, limit=bad))


def _fink_row(oid: str, jd: float) -> dict[str, Any]:
    return {"i:objectId": oid, "i:ra": 10.0, "i:dec": 20.0, "i:jd": jd, "i:magpsf": 19.0, "i:fid": 2, "i:isdiffpos": "t",
            "d:snn_snia_vs_nonia": 0.9, "d:snn_sn_vs_all": 0.8}


async def test_fink_paging_walks_back_in_time() -> None:
    """Synthetic /latests pages: repeated alerts of one object make Fink page back until limit+1 objects."""
    jd0 = 2461306.8
    pages = {
        # stopdate -> rows (newest first); A has 3 alerts, so 3 rows hold only 1 object.
        mjd_to_iso(61306.35, round_up=True): [_fink_row("A", jd0 + 0.003), _fink_row("A", jd0 + 0.002),
                                              _fink_row("A", jd0 + 0.001)],
        mjd_to_iso(jd0 + 0.001 - 2400000.5, round_up=True): [_fink_row("A", jd0 + 0.001), _fink_row("B", jd0 - 0.01),
                                                             _fink_row("C", jd0 - 0.02)],
    }
    sent: list[dict[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        sent.append(q)
        return httpx.Response(200, json=pages[q["stopdate"]][: int(q["n"])])

    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(side_effect=answer)
        async with offline_client() as client:
            result = await FinkZTFBroker().fetch(client, since_mjd=61306.0, until_mjd=61306.35, limit=2)
    assert [q["n"] for q in sent] == ["3", "3"] and result.requests == 2
    assert [a.object_id for a in result.alerts] == ["A", "B"] and result.truncated
    assert result.alerts[0].mjd == pytest.approx(jd0 + 0.003 - 2400000.5)  # A's newest alert
    assert result.boundary_mjd == pytest.approx(jd0 - 0.01 - 2400000.5)  # B: the oldest kept


async def test_fink_paging_grows_n_when_one_second_holds_a_full_page() -> None:
    """Synthetic: 4 alerts of one object in the same second; n doubles instead of looping forever."""
    jd = 2461306.8
    same = [_fink_row("A", jd)] * 4 + [_fink_row("B", jd - 0.01)]
    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        n = int(request.url.params["n"])
        sent.append(request.url.params["n"])
        return httpx.Response(200, json=same[:n])

    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(side_effect=answer)
        async with offline_client() as client:
            result = await FinkZTFBroker().fetch(client, since_mjd=61306.0, until_mjd=61306.4, limit=1)
    assert sent == ["2", "2", "4", "8"]
    assert [a.object_id for a in result.alerts] == ["A"] and result.truncated  # B exists beyond the limit


# ---------------------------------------------------------------------------
# Host geometry and naming
# ---------------------------------------------------------------------------


def test_d25_geometry() -> None:
    # HyperLEDA logD25 in log10(0.1 arcmin): M31 (PGC 2557) logD25 = 3.30 -> D25 = 199.5' -> a = 5986".
    a, b = d25_ellipse(3.30, 0.45)
    assert a == pytest.approx(0.1 * 10**3.30 * 60 / 2) and b == pytest.approx(a / 10**0.45)
    assert d25_ellipse(None, 0.1) is None and d25_ellipse(1.0, None) == (30.0, 30.0)
    assert directional_light_radius(10.0, 5.0, 30.0, 30.0) == pytest.approx(10.0)  # along the major axis
    assert directional_light_radius(10.0, 5.0, 30.0, 120.0) == pytest.approx(5.0)  # along the minor axis
    assert directional_light_radius(10.0, 5.0, 30.0, 210.0) == pytest.approx(10.0)
    assert directional_light_radius(10.0, 5.0, None, 0.0) == pytest.approx(math.sqrt(50.0))
    # Position angle (N through E) agrees with astropy.
    for ra2, dec2 in ((10.0, 21.0), (11.0, 20.0), (9.5, 19.3)):
        expected = SkyCoord(10.0, 20.0, unit="deg").position_angle(SkyCoord(ra2, dec2, unit="deg")).deg
        assert position_angle_deg(10.0, 20.0, ra2, dec2) == pytest.approx(expected, abs=1e-9)


def test_transient_designations() -> None:
    for name in ("SN 2018cow", "AT 2019dsg", "AT2017gfo", "SN 1987A", "ASASSN-14li", "GrW 170817", "GW170817",
                 "GRB 170817A", "ZTF19aapreis", "Gaia16aye", "PS1-10jh", "ATLAS19abc", "iPTF16fnl", "FRB 20180916B"):
        assert is_transient_designation(name), name
    for name in ("ZTF J1901+1458", "ASASSN-V J123456.78+123456.7", "NGC 4993", "M 31", "Messier 031",
                 "2MASX J20570298+1412165", "SN", "", None, "Gaia DR3 1220110705972528512", "SN 1994I HOST"):
        assert not is_transient_designation(name), name
    # SIMBAD spells survey names with a space ('ATLAS 17kol'), TNS without: both are designations.
    for name in ("ATLAS 17kol", "ATLAS17kol", "Gaia 16aye", "PS 15cey", "ZTF 18aayefwp"):
        assert is_transient_designation(name), name


def test_simbad_type_sets_follow_otypedef() -> None:
    # Codes and labels (Fink's cross-match emits labels, incl. '_Candidate' ones; verified live).
    for label in ("CataclyV*_Candidate", "RRLyrae_Candidate", "EclBin_Candidate", "Mira_Candidate",
                  "LongPeriodV*_Candidate", "XrayBin_Candidate", "Symbiotic*_Candidate", "CV?", "RRLyrae", "HXB"):
        assert label in alerts.SIMBAD_VARIABLE_TYPES and label in alerts.SIMBAD_STAR_TYPES, label
    for label in ("WhiteDwarf_Candidate", "BrownD*_Candidate", "TTauri*_Candidate", "WR*", "PN", "*"):
        assert label in alerts.SIMBAD_STAR_TYPES, label
    for label in ("SN*", "Supernova", "GWE", "Pl", "HH", "G", "Galaxy_Candidate"):
        assert label not in alerts.SIMBAD_STAR_TYPES, label
    assert {"SN*", "SN?", "ev", "GWE", "gB"} <= alerts.SIMBAD_TRANSIENT_TYPES
    assert set(alerts.SIMBAD_HOST_OTYPES) <= alerts.SIMBAD_GALAXY_TYPES
    assert {"GinPair", "PairG", "Seyfert2", "LensedG"} <= alerts.SIMBAD_GALAXY_TYPES
    assert not alerts.SIMBAD_GALAXY_TYPES & alerts.SIMBAD_STAR_TYPES


def test_ned_types_and_catalogued_cvs_with_transient_names() -> None:
    def entry(catalog: str, name: str, otype: str | None) -> dict[str, Any]:
        return {"catalog": catalog, "source_id": name, "object_type": otype}

    for otype in ("WR*", "Psr", "Red*", "Blue*", "Flare*", "exG*", "!*", "!V*"):
        assert alerts._is_stellar(entry("ned", "X", otype)), otype
    assert alerts._is_variable(entry("ned", "X", "Flare*")) and alerts._is_variable(entry("ned", "X", "!Nova"))
    assert alerts._is_extragalactic_star(entry("ned", "X", "exG*")) and not alerts._is_extragalactic_star(entry("ned", "X", "*"))
    assert alerts._is_ned_galactic_star(entry("ned", "X", "!*")) and not alerts._is_ned_galactic_star(entry("ned", "X", "*"))
    # A transient-style name is the transient itself unless the catalogue types it as a star/variable.
    assert alerts._is_transient(entry("simbad", "SN 2018cow", "SN*"))
    assert alerts._is_transient(entry("ned", "AT 2019dsg", "G"))
    assert alerts._is_transient(entry("simbad", "ZTF19aapreis", None))
    for cv in (entry("simbad", "MASTER OT J073857.06+182648.2", "CV?"), entry("simbad", "ZTF18aayefwp", "CV*"),
               entry("simbad", "PNV J00444033+4113068", "No*"), entry("simbad", "ATLAS 17kol", "CV?"),
               entry("ned", "AT 2016dah", "Nova")):
        assert not alerts._is_transient(cv), cv


def test_group_host_candidates_merges_aliases_and_prefers_catalogue_names() -> None:
    gal = [
        {"catalog": "ned", "source_id": "AT 2019dsg", "ra": 10.0, "dec": 0.0, "separation_arcsec": 0.3, "object_type": "G",
         "redshift": None},
        {"catalog": "simbad", "source_id": "2MASX J00400000+0000001", "ra": 10.0 + 1.0 / 3600, "dec": 0.0,
         "separation_arcsec": 0.5, "object_type": "G", "redshift": 0.02},
        {"catalog": "ned", "source_id": "B", "ra": 10.01, "dec": 0.0, "separation_arcsec": 30.0, "object_type": "G?",
         "redshift": None},
    ]
    groups = group_host_candidates(gal)
    assert [g["source_id"] for g in groups] == ["2MASX J00400000+0000001", "B"]
    assert groups[0]["aliases"] == ["ned:AT 2019dsg"] and groups[0]["redshift"] == 0.02
    assert groups[0]["separation_arcsec"] == 0.3  # the group's nearest entry
    assert groups[0]["redshift_source"] == "simbad:2MASX J00400000+0000001" and groups[0]["projected_offset_kpc"] > 0
    assert groups[1]["projected_offset_kpc"] is None
    assert not groups[0]["transient_named"] and not groups[1]["transient_named"]


def test_host_names_avoid_transient_labels_and_subcomponents() -> None:
    """NED names M51 'SN 1994I HOST' and IC 10's stars 'IC 0010:[..]': major catalogue names win."""
    m51 = [
        {"catalog": "ned", "source_id": "SN 1994I HOST", "ra": 202.4696, "dec": 47.1952, "separation_arcsec": 156.0,
         "object_type": "G", "redshift": 0.00157},
        {"catalog": "ned", "source_id": "IC 0010:[GTA2012] 31-B", "ra": 202.4696, "dec": 47.1952 + 0.5 / 3600,
         "separation_arcsec": 156.1, "object_type": "G", "redshift": None},
        {"catalog": "simbad", "source_id": "M  51", "ra": 202.4696, "dec": 47.1952 + 1.0 / 3600,
         "separation_arcsec": 156.2, "object_type": "Sy2", "redshift": 0.0015},
    ]
    (grp,) = group_host_candidates(m51)
    assert grp["source_id"] == "M  51" and "ned:SN 1994I HOST" in grp["aliases"]
    assert grp["redshift"] == 0.0015 and grp["redshift_source"] == "simbad:M  51"
    # Without a major-catalogue name, any other name still beats the transient-host label.
    (grp,) = group_host_candidates([m51[0], {**m51[1], "source_id": "WISEA J132952.71+471142.6"}])
    assert grp["source_id"] == "WISEA J132952.71+471142.6" and grp["redshift"] == 0.00157
    # A galaxy known only under a transient's name is flagged.
    (only,) = group_host_candidates([{"catalog": "ned", "source_id": "AT 2017abr", "ra": 1.0, "dec": 1.0,
                                      "separation_arcsec": 0.3, "object_type": "G", "redshift": 0.2069}])
    assert only["transient_named"] is True


def test_projected_offset_distances() -> None:
    from astropy.cosmology import Planck18

    # z = 0.05: angular-diameter distance (not the luminosity distance, (1+z)^2 = 1.10 larger) of a flat
    # LCDM with the Planck 2018 densities and the Cosmicflows-4 H0 = 74.6 km/s/Mpc.
    d_a = Planck18.clone(H0=74.6).angular_diameter_distance(0.05).to(u.kpc).value
    assert projected_offset_kpc(10.0, 0.05) == pytest.approx(d_a * math.radians(10.0 / 3600.0), rel=1e-6)
    # Below cz = 1500 km/s a redshift distance is meaningless (Local Group: M31 at -300 km/s, the LMC at +262 km/s).
    assert projected_offset_kpc(4133.7, 0.000875) is None and projected_offset_kpc(100.0, -0.001) is None
    assert projected_offset_kpc(100.0, None) is None
    # With a Cosmicflows-4 distance modulus: LMC DM 18.469 -> 49.3 kpc; SN 1987A 4133.7" away -> 0.99 kpc
    # (Pietrzynski et al. 2019: 49.59 kpc -> 0.994 kpc).
    assert projected_offset_kpc(4133.7, 0.000875, (18.469, 0.07)) == pytest.approx(0.988, abs=0.005)
    # CF4 is used for z < 0.01 only; beyond, the Hubble flow (z = 0.0141, AT 2018cow's host).
    assert projected_offset_kpc(5.0, 0.0141, (35.0, 0.1)) == pytest.approx(projected_offset_kpc(5.0, 0.0141), rel=1e-9)
    near = alerts.host_distance(0.006)
    assert near["method"] == "hubble_flow" and near["fractional_uncertainty"] == pytest.approx(300 / (0.006 * 299792.458))
    assert near["hubble_constant_kms_mpc"] == pytest.approx(74.6)


def test_hyperleda_pa_is_precessed_from_b1950() -> None:
    # HyperLEDA PA is the "Adopted 1950-position angle": the ICRS PA differs by ~0.278 deg sin(RA) / cos(Dec).
    ras, decs = [148.968, 10.6846, 90.0, 270.0, 0.0], [69.679, 41.269, 0.0, 0.0, 0.0]
    out = alerts.pa_b1950_to_icrs(ras, decs, [65.0, 35.0, 10.0, 10.0, 10.0])
    for ra, dec, pa in zip(ras, decs, out, strict=True):
        approx_shift = 0.2783 * math.sin(math.radians(ra)) / math.cos(math.radians(dec))
        assert pa - (65.0 if ra == 148.968 else 35.0 if ra == 10.6846 else 10.0) == pytest.approx(approx_shift, abs=0.01)
    assert alerts.pa_b1950_to_icrs([1.0], [2.0], [None]) == [None]


def test_host_search_uses_server_side_galaxy_filters() -> None:
    from models import CatalogRegistry, validate_target
    from providers import TapProvider

    registry = HostSearchRegistry(CatalogRegistry())
    target = validate_target(148.925583, 69.673889)
    simbad = TapProvider().build_adql(registry.get("simbad"), target, 60.0)
    ned = TapProvider().build_adql(registry.get("ned"), target, 60.0)
    assert "AND (b.otype IN ('G', 'G?', 'AGN'" in simbad and "'SBG'" in simbad and "'HII'" not in simbad
    assert "AND (prefphytype IN ('G', 'GPair', 'GTrpl', 'G_Lens', 'QSO'))" in ned
    assert registry.get(HYPERLEDA).table == '"VII/237/pgc"'
    assert registry.get(alerts.COSMICFLOWS4).table == '"J/ApJ/944/94/table2"'
    assert "where" not in CatalogRegistry().get("simbad").parameters  # the base registry is untouched
    # The counterpart search asks Gaia DR3 for the proper-motion errors as well.
    gaia = TapProvider().build_adql(alerts.MatchSearchRegistry(CatalogRegistry()).get("gaia_dr3"), target, 2.0)
    assert all(c in gaia for c in ("pmra_error", "pmdec_error", "pmra_pmdec_corr", "parallax_error", "ruwe"))
    assert "pmra_error" not in TapProvider().build_adql(CatalogRegistry().get("gaia_dr3"), target, 2.0)
    # Every PGC galaxy type is searched (VII/237 OType 'G ', 'GM', 'M '): M86 and IC 10 are type 'M'.
    bodies = [unquote_plus(e["request_body"]) for e in meta("xmatch_famous")["exchanges"]]
    d25 = [b for b in bodies if '"VII/237/pgc"' in b]
    assert d25 and all("(OType LIKE 'G%' OR OType LIKE 'M%')" in b for b in d25)


# ---------------------------------------------------------------------------
# Enrichment truths (recorded Gaia DR3 / SIMBAD / NED / HyperLEDA answers)
# ---------------------------------------------------------------------------


async def enrich_recorded(scenario: str) -> dict[str, AlertEnrichment]:
    out = {}
    with Replay(scenario):
        async with offline_client() as client:
            enricher = AlertEnricher(make_service(client))
            for alert in recorded_alerts(scenario):
                out[alert.object_id] = await enricher.enrich(alert)
    return out


@pytest.fixture(scope="module")
def famous() -> dict[str, AlertEnrichment]:
    return asyncio.run(enrich_recorded("xmatch_famous"))


def test_t_crb_is_a_known_variable_galactic_star(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["T_CrB"]
    assert res.status == "done" and res.catalog_status == {"gaia_dr3": "success", "simbad": "success", "ned": "success"}
    assert res.known_star is True and res.known_variable is True and res.is_new is False
    gaia = next(c for c in res.counterparts if c["catalog"] == "gaia_dr3")
    # Gaia DR3 parallax of T CrB ~1.09 mas (distance ~0.9 kpc), measured at ~40 sigma. Its RUWE
    # (1.64) reflects the binary, not a spurious parallax: outside any galaxy it still counts.
    assert gaia["parallax_mas"] == pytest.approx(1.09, abs=0.05) and gaia["parallax_over_error"] > 30
    assert gaia["ruwe"] > 1.4 and any("Galactic star" in e and "Gaia DR3" in e for e in res.evidence)
    simbad = next(c for c in res.counterparts if c["catalog"] == "simbad")
    assert simbad["source_id"] == "V* T CrB" and simbad["object_type"] in alerts.SIMBAD_VARIABLE_TYPES
    # A star has no extragalactic host; the background galaxies stay listed as candidates.
    assert res.host is None and res.host_status == "not_applicable_star" and res.host_candidates


def test_at2018cow_host_is_cgcg_137_068(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["AT2018cow"]
    assert "SN 2018cow" in res.transient_designations and len(set(res.transient_designations)) == len(res.transient_designations)
    assert res.known_star is False and res.known_variable is False and res.status == "done"
    host = res.host
    assert res.host_status == "found" and host is not None and res.host_search_complete is True
    assert {"CGCG 137-068", "Z 137-68"} <= names_of(host)  # NED and SIMBAD entries of the same galaxy merged
    # Inside CGCG 137-068's D25 ellipse (HyperLEDA PGC 57660).
    assert host["method"] == "d25_ellipse" and host["pgc"] == 57660 and host["d_dlr"] < 1.0
    # Host redshift z = 0.0141 and a ~1.5-1.7 kpc projected offset (Prentice et al. 2018: 1.7 kpc).
    assert host["redshift"] == pytest.approx(0.0141, abs=0.0003)
    assert 4.0 < host["separation_arcsec"] < 7.0
    assert 1.2 < host["projected_offset_kpc"] < 2.0


def test_m31n_2008_12a_is_an_extragalactic_nova_in_m31(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["M31N_2008-12a"]
    # The recurrent nova is catalogued (SIMBAD No*, NED Nova): stellar, variable, not new ...
    assert res.stellar_counterpart is True and res.known_variable is True and res.is_new is False
    assert any(c["object_type"] in {"No*", "Nova"} for c in res.counterparts)
    # ... but it is in M31, not a Galactic star, and M31 is its host (0.8 deg away, inside the D25 ellipse).
    assert res.known_star is False
    host = res.host
    assert host is not None and res.host_status == "found" and host["method"] == "d25_ellipse"
    assert names_of(host) & {"Messier 031", "M 31"} and host["pgc"] == 2557
    assert 2900 < host["separation_arcsec"] < 2970 and host["d_dlr"] < 1.0
    assert host["redshift"] == pytest.approx(-0.001, abs=0.0002)  # M31 approaches at ~300 km/s
    # No Hubble-flow distance for a blueshift: the Cosmicflows-4 distance of M31 (DM 24.37, 0.75 Mpc,
    # cf. 0.76 Mpc from Cepheids/TRGB) gives the projected offset instead: 2940" -> ~10.6 kpc.
    assert host["distance_method"] == "cosmicflows4" and host["distance_modulus"] == pytest.approx(24.37, abs=0.05)
    assert host["projected_offset_kpc"] == pytest.approx(
        10 ** (host["distance_modulus"] / 5 - 2) * math.radians(host["separation_arcsec"] / 3600), rel=1e-3)
    assert 10.0 < host["projected_offset_kpc"] < 11.5
    assert any("extragalactic star" in e for e in res.evidence)
    assert not any("Galactic star" in e and "not a Galactic" not in e for e in res.evidence)


def test_m82_nucleus_keeps_its_host(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["M82_nucleus"]
    # An X-ray binary inside M82 lies within 1": stellar and variable, but extragalactic.
    assert any(c["object_type"] == "HXB" for c in res.counterparts)
    assert res.stellar_counterpart is True and res.known_star is False
    host = res.host
    assert host is not None and res.host_status == "found" and names_of(host) & {"M 82", "Messier 082"}
    assert host["method"] == "d25_ellipse" and host["pgc"] == 28655 and host["d_dlr"] < 0.1
    assert host["redshift"] == pytest.approx(0.0007, abs=0.0002)


def test_sn2014j_host_is_m82_not_a_fragment(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["SN2014J"]
    assert "SN 2014J" in res.transient_designations and res.known_star is False
    host = res.host
    assert host is not None and names_of(host) & {"M 82", "Messier 082"}
    # 58" from M82's centre but well inside its D25 ellipse (a = 321"), while an SDSS fragment of
    # M82 is nearer (26"): the D25 ellipse, not the nearest centre, identifies the host.
    assert host["method"] == "d25_ellipse" and 55.0 < host["separation_arcsec"] < 60.0 and host["d_dlr"] < 0.5
    assert res.host_candidates[0]["separation_arcsec"] < 30.0 and res.host_candidates[0]["source_id"] != host["name"]
    assert res.host_search_complete is True


def test_at2017gfo_designations_and_ngc4993(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["AT2017gfo"]
    # SIMBAD 'GrW 170817' (type GWE) and NED 'AT 2017gfo' are the event itself, not prior sources.
    assert {"GrW 170817", "AT 2017gfo"} <= set(res.transient_designations)
    assert not any(is_transient_designation(c["source_id"]) for c in res.counterparts)
    host = res.host
    assert host is not None and "NGC 4993" in names_of(host) and host["method"] == "d25_ellipse"
    # NGC 4993 at z = 0.0098; projected offset ~2.1 kpc (Levan et al. 2017, ApJL 848, L28).
    assert host["redshift"] == pytest.approx(0.0098, abs=0.0003)
    assert 1.8 < host["projected_offset_kpc"] < 2.4


def test_at2019dsg_host_is_named_after_the_galaxy(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["AT2019dsg"]
    host = res.host
    assert host is not None and not is_transient_designation(host["name"])
    assert "AT 2019dsg" in names_of(host)  # NED's host entry under the transient's name, kept as an alias
    assert host["redshift"] == pytest.approx(0.0512, abs=0.001)  # Stein et al. 2021: z = 0.0512
    assert "AT 2019dsg" in res.transient_designations
    # Projected offset at the Hubble-flow angular-diameter distance (Planck 2018 densities, CF4 H0 = 74.6;
    # to 1%; the luminosity distance
    # would be (1+z)^2 = 10.5% larger) of the CMB-frame redshift: the Sun moves at 369.82 km/s towards
    # Galactic (264.021, 48.253) (Planck 2020), 146 deg from AT 2019dsg, so cz_CMB < cz_helio by ~307 km/s.
    from astropy.cosmology import Planck18

    apex = SkyCoord(l=264.021 * u.deg, b=48.253 * u.deg, frame="galactic")
    v_r = 369.82 * math.cos(apex.separation(SkyCoord(host["ra"] * u.deg, host["dec"] * u.deg)).rad)
    z_cmb = (1 + host["redshift"]) / (1 - v_r / 299792.458) - 1
    assert -320 < v_r < -290 and host["redshift_cmb"] == pytest.approx(z_cmb, rel=1e-6)
    assert host["velocity_frame"] == "cmb"
    cosmo = Planck18.clone(H0=74.6)
    d_a = cosmo.comoving_transverse_distance(z_cmb).to(u.kpc).value / (1 + host["redshift"])
    expected = d_a * math.radians(host["separation_arcsec"] / 3600.0)
    assert host["projected_offset_kpc"] == pytest.approx(expected, rel=1e-6)
    assert host["distance_method"] == "hubble_flow" and host["hubble_constant_kms_mpc"] == pytest.approx(74.6)
    # Heliocentric would be 1.9% too far.
    helio = cosmo.angular_diameter_distance(host["redshift"]).to(u.kpc).value * math.radians(host["separation_arcsec"] / 3600.0)
    assert helio / host["projected_offset_kpc"] == pytest.approx(1.019, abs=0.004)


def test_ic10_x1_is_in_ic10_a_pgc_multiple_system(famous: dict[str, AlertEnrichment]) -> None:
    """IC 10 is PGC type 'M': the D25 search must see it; IC 10 X-1 is then an extragalactic star."""
    res = famous["IC10_X-1"]
    assert res.status == "done" and res.stellar_counterpart is True and res.known_variable is True
    assert res.known_star is False
    # NED's Wolf-Rayet counterparts (type WR*) count as stellar.
    assert any(c["catalog"] == "ned" and c["object_type"] == "WR*" for c in res.counterparts)
    assert any("NED IC 0010:[CDA2003] 17-B (type WR*)" in e for e in res.evidence)
    host = res.host
    assert host is not None and host["method"] == "d25_ellipse" and host["pgc"] == 1305
    # HyperLEDA names the host (NED/SIMBAD's nearest entry lies 31" off the D25 centre).
    assert host["name"] == "IC 10" and 0.25 < host["d_dlr"] < 0.4
    # IC 10 at 0.78 Mpc (CF4 DM 24.45): 62.7" -> 0.24 kpc.
    assert host["distance_method"] == "cosmicflows4" and host["projected_offset_kpc"] == pytest.approx(0.236, abs=0.01)


def test_m86_disk_point_gets_m86(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: M86 (PGC 40653, type 'M') was invisible to the D25 search and a 48" dwarf was the host."""
    res = famous["M86_disk"]
    host = res.host
    assert host is not None and names_of(host) & {"M 86", "Messier 086"} and host["pgc"] == 40653
    assert host["method"] == "d25_ellipse" and host["d_dlr"] == pytest.approx(0.60, abs=0.03)
    assert res.host_search_complete is True and res.is_new is True
    # M86 is blueshifted (-244 km/s) and has no Cosmicflows-4 distance of its own: it takes the CF4 distance of
    # its group, the Virgo cluster (Tully 2015 nest 100002, dominant galaxy M49 = PGC 41220, DMzp 31.048 ->
    # 16.2 Mpc; SBF distance of M86: 16.8 Mpc, Mei et al. 2007), uncertain by ~9% (Virgo's depth R2t 1.44 Mpc).
    assert host["distance_method"] == "cosmicflows4_group" and host["distance_modulus"] == pytest.approx(31.048, abs=0.01)
    assert host["group"]["nest"] == 100002 and host["group"]["pgc1"] == 41220 and host["group"]["sigma_v_kms"] == 670
    assert host["distance_mpc"] == pytest.approx(16.2, abs=0.1)
    assert host["projected_offset_kpc"] == pytest.approx(
        host["distance_mpc"] * 1000 * math.radians(host["separation_arcsec"] / 3600), rel=1e-6)
    assert 11.0 < host["projected_offset_kpc"] < 12.5
    assert 0.07 < host["distance_uncertainty_fraction"] < 0.12
    assert any("of its group 100002" in e for e in res.evidence)


def test_sn2006gy_host_ngc1260_by_d25(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["SN2006gy"]
    host = res.host
    assert "SN 2006gy" in res.transient_designations
    assert host is not None and "NGC 1260" in names_of(host) and host["pgc"] == 12219 and host["method"] == "d25_ellipse"
    assert host["redshift"] == pytest.approx(0.0192, abs=0.0003)  # NGC 1260, cz = 5750 km/s


@pytest.mark.parametrize(("name", "criterion"), [
    ("M31_fg_star", "secure even projected"),  # G = 12.2, 75-sigma parallax, RUWE 1.8
    ("M31_fg_star2", "secure even projected"),  # 60-sigma parallax, RUWE 5.3
    ("M101_fg_star", "proper motion"),  # 3.7-sigma parallax but 16.9 mas/yr at 200 sigma
])
def test_foreground_stars_on_nearby_galaxies_are_galactic(famous: dict[str, AlertEnrichment], name: str,
                                                          criterion: str) -> None:
    res = famous[name]
    assert res.status == "done" and res.known_star is True
    assert res.host is None and res.host_status == "not_applicable_star"
    assert any(criterion in e and "Galactic star" in e for e in res.evidence), res.evidence
    assert not any("extragalactic star" in e for e in res.evidence)
    gaia = next(c for c in res.counterparts if c["catalog"] == "gaia_dr3")
    if name == "M101_fg_star":
        assert gaia["parallax_over_error"] < 5 and gaia["pm_masyr"] == pytest.approx(16.9, abs=0.1)
        assert gaia["pm_over_error"] > 100
        # G = 17.65 at M101's distance modulus (29.15) would be M_G = -11.5.
        assert any("M_G = -11.5" in e for e in res.evidence)


def test_sn1987a_offset_uses_the_lmc_distance(famous: dict[str, AlertEnrichment]) -> None:
    res = famous["SN1987A"]
    host = res.host
    assert host is not None and names_of(host) & {"NAME LMC", "Large Magellanic Cloud"} and host["pgc"] == 17223
    # 4134" at the CF4 LMC distance (DM 18.47, 49.3 kpc) = 0.99 kpc (Hubble flow at z = 0.0009 gave 77.6 kpc).
    assert host["distance_method"] == "cosmicflows4" and host["projected_offset_kpc"] == pytest.approx(0.99, abs=0.02)
    # The progenitor Sk -69 202 (NED 'CPD-69 00402') is an LMC star; LMC field stars (pm ~1.5-2 mas/yr) stay LMC stars.
    assert res.known_star is False and res.stellar_counterpart is True
    assert "SN 1987A" in res.transient_designations


@pytest.mark.parametrize(("name", "host_name", "label"), [("SN2011dh", "M 51", "SN 1994I HOST"),
                                                          ("SN1993J", "M 81", "SN 1993J HOST")])
def test_host_is_not_named_after_another_transient(famous: dict[str, AlertEnrichment], name: str, host_name: str,
                                                   label: str) -> None:
    res = famous[name]
    host = res.host
    assert host is not None and host["name"] == host_name and host["method"] == "d25_ellipse"
    assert f"ned:{label}" in host["aliases"]
    if name == "SN2011dh":  # M51 at the CF4 distance (DM 29.61, 8.3 Mpc): 156" -> 6.3 kpc
        assert host["distance_method"] == "cosmicflows4" and host["projected_offset_kpc"] == pytest.approx(6.3, abs=0.2)
    else:  # M81: blueshifted, no CF4 distance of its own -> the CF4 distance of the M81 group (3.6 Mpc; TRGB 3.6 Mpc)
        assert host["distance_method"] == "cosmicflows4_group" and host["group"]["pgc1"] == 28630
        assert host["distance_mpc"] == pytest.approx(3.63, abs=0.05)
        assert host["projected_offset_kpc"] == pytest.approx(2.94, abs=0.05)


@pytest.mark.parametrize("name", ["ZTF18aayefwp", "ZTF18aaawtyh"])
def test_catalogued_cvs_with_transient_names_are_known_variables(famous: dict[str, AlertEnrichment], name: str) -> None:
    """Regression: SIMBAD CVs named 'ZTF18aayefwp' (CV*) / 'MASTER OT J073857.06+182648.2' (CV?) were
    dropped as 'the transient itself', so repeat outbursts looked new and not variable."""
    res = famous[name]
    cv_name = {"ZTF18aayefwp": "ZTF18aayefwp", "ZTF18aaawtyh": "MASTER OT J073857.06+182648.2"}[name]
    cv = next(c for c in res.counterparts if c["source_id"] == cv_name)
    assert cv["object_type"] in {"CV*", "CV?"} and cv_name not in res.transient_designations
    assert res.known_variable is True and res.is_new is False and res.stellar_counterpart is True
    assert res.known_star is True and res.host is None  # a Galactic CV has no host galaxy
    assert any("transient-style designation" in e and cv_name in e for e in res.evidence)


async def test_broker_alerts_enrichment_flags() -> None:
    res = await enrich_recorded("xmatch_fink_ztf")
    fink_alerts = {a.object_id: a for a in recorded_alerts("xmatch_fink_ztf")}
    # The TNS SN Ia ZTF26abpbqtk lies in CGCG 414-037 (z = 0.0213); Fink's Mangrove match gives a
    # luminosity distance of ~100 Mpc for its host, i.e. z ~ 0.022 in Planck 2018 cosmology.
    snia = res["ZTF26abpbqtk"]
    assert snia.host is not None and snia.host["redshift"] == pytest.approx(0.0213, abs=0.001)
    assert names_of(snia.host) & {"CGCG 414-037", "Z 414-37"}
    from astropy.cosmology import Planck18, z_at_value

    z_mangrove = z_at_value(Planck18.luminosity_distance, fink_alerts["ZTF26abpbqtk"].extra["mangrove_lum_dist"] * u.Mpc).value
    assert snia.host["redshift"] == pytest.approx(z_mangrove, rel=0.1)
    # ZTF26abuuisj: our D25 host is the galaxy Fink's Mangrove catalogue names (NGC1030).
    assert fink_alerts["ZTF26abuuisj"].extra["mangrove_hyperleda_name"] == "NGC1030"
    assert "NGC 1030" in {n.replace("NGC1030", "NGC 1030") for n in names_of(res["ZTF26abuuisj"].host)}
    for result in res.values():
        assert result.status == "done" and result.known_star is False
        assert result.is_new == (not result.counterparts)
    lsst = await enrich_recorded("xmatch_fink_lsst")
    for result in lsst.values():
        assert result.status == "done" and result.is_new == (not result.counterparts)
        assert result.host_status in {"found", "none_within_radius"} and result.host_search_complete is True


async def fink_variable_alerts() -> dict[str, Alert]:
    """Real Fink/ZTF '(TNS) CV' and 'RRLyrae' rows (broker SIMBAD labels and Gaia DR3 parallaxes)."""
    p = params("fink_variables")
    found: dict[str, Alert] = {}
    with Replay("fink_variables"):
        async with offline_client() as client:
            for cls in p["classes"]:
                res = await fetch_alerts(client, "fink", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"],
                                         limit=p["limit"], options={"class_name": cls})
                found.update({a.object_id: a for a in res.alerts})
    return found


async def test_fink_broker_crossmatch_evidence() -> None:
    """Recorded Fink rows: Fink emits current SIMBAD labels ('CataclyV*_Candidate', 'RRLyrae') and
    Gaia DR3 parallaxes; with nothing else known they alone make the alert a variable / Galactic star."""
    found = await fink_variable_alerts()
    cv = found["ZTF18acclkfa"]
    assert cv.extra["simbad_otype"] == "CataclyV*_Candidate" and cv.extra["tns_type"] == "CV"
    assert cv.extra["gaia_dr3_name"] == "Gaia DR3 3126747736361227008"
    nothing = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats))  # every archive answers: no source
    res = await nothing.enrich(cv)
    assert res.known_variable is True and res.stellar_counterpart is True and res.known_star is True
    assert any("fink SIMBAD xmatch (type CataclyV*_Candidate)" in e for e in res.evidence)
    # Its Fink Gaia parallax 1.374 +/- 0.225 mas (6.1 sigma) is Galactic evidence of its own.
    assert any("fink Gaia DR3 xmatch Gaia DR3 3126747736361227008: parallax 1.374" in e and "-> Galactic star" in e
               for e in res.evidence)
    # The SIMBAD label alone (without Gaia's variability flag) makes it a known variable.
    no_flag = Alert.from_dict({**cv.as_dict(), "extra": {**cv.extra, "gaia_var_flag": 0}})
    res = await nothing.enrich(no_flag)
    assert res.known_variable is True
    assert any(e.startswith("known variable: ") and "type CataclyV*_Candidate" in e for e in res.evidence)
    rr = found["ZTF18aaxdjur"]
    assert rr.extra["simbad_otype"] == "RRLyrae" and rr.extra["gaia_var_flag"] == 1
    res = await nothing.enrich(Alert.from_dict({**rr.as_dict(), "extra": {**rr.extra, "gaia_var_flag": 0}}))
    assert res.known_variable is True
    assert any(e.startswith("known variable: ") and "type RRLyrae" in e for e in res.evidence)
    # Without the SIMBAD label, the broker parallax (14.8 sigma) alone makes it a Galactic star.
    bare = Alert.from_dict({**rr.as_dict(), "extra": {**rr.extra, "simbad_otype": None, "gaia_var_flag": 0}})
    res = await nothing.enrich(bare)
    assert res.known_star is True and res.stellar_counterpart is False and res.known_variable is False
    assert any("parallax 0.374" in e and "-> Galactic star" in e for e in res.evidence)


async def test_fink_variables_real_enrichment() -> None:
    res = await enrich_recorded("xmatch_fink_variables")
    assert set(res) == {"ZTF19aaxvjli", "ZTF18acclkfa", "ZTF18aaxdjur", "ZTF18adqfyiv"}
    for oid in ("ZTF18acclkfa", "ZTF18aaxdjur", "ZTF18adqfyiv"):
        assert res[oid].status == "done" and res[oid].known_variable is True and res[oid].known_star is True, oid
        assert res[oid].is_new is False and res[oid].host is None
    # ZTF19aaxvjli (a TNS CV) has no catalogued counterpart and no Fink cross-match: nothing marks it.
    assert res["ZTF19aaxvjli"].is_new is True and res["ZTF19aaxvjli"].known_variable is False


# ---------------------------------------------------------------------------
# Flag rules and partial failures (synthetic crossmatch answers)
# ---------------------------------------------------------------------------


def fake_service(answer: Any) -> Any:
    """A CrossmatchService stand-in whose crossmatch() returns answer(ra, dec, catalogs)."""
    real = make_service(offline_client())

    class Fake:
        registry = real.registry
        providers: ClassVar[dict[str, Any]] = {}
        executor = SimpleNamespace(timeout=30.0)

        async def crossmatch(self, ra, dec, *, query, **kwargs):
            return answer(ra, dec, list(query.catalogs))

    return Fake()


def record_of(ra: float, dec: float, catalogs: list[str], *, failed: dict[str, str] | None = None,
              sources: dict[str, list[dict[str, Any]]] | None = None) -> UnifiedRecord:
    failed = failed or {}
    results: dict[str, Any] = {}
    counterparts: dict[str, list[dict[str, Any]]] = {}
    for name in catalogs:
        if name in failed:
            results[name] = {"sources": [], "status": "failed", "truncated": False}
            continue
        rows = (sources or {}).get(name, [])
        results[name] = {"sources": rows, "status": "success" if rows else "empty", "truncated": False, "warnings": []}
        for row in rows:
            counterparts.setdefault("optical", []).append({"catalog": name, **row})
    failures = [{"catalog": n, "status": "failed", "error_type": t, "message": "down", "elapsed_ms": 1.0}
                for n, t in failed.items() if n in catalogs]
    return UnifiedRecord(target={"ra": ra, "dec": dec}, catalogs_queried=len(catalogs), catalog_results=results,
                         counterparts=counterparts, failures=failures, provenance={})


def enricher_with(answer: Any, d25: tuple[list[dict[str, Any]], dict[str, Any] | None, bool] = ([], None, False),
                  cf4: dict[int, tuple[float, float]] | Exception | None = None,
                  groups: dict[int, dict[str, Any]] | Exception | None = None) -> AlertEnricher:
    """AlertEnricher whose archive answers are synthetic: ``answer(ra, dec, catalogs)`` for the cones, ``d25``
    for HyperLEDA, ``cf4`` for the Cosmicflows-4 distances and ``groups`` for the Tully (2015) group lookup
    (by PGC; a missing PGC is in no group, an exception makes the lookup fail)."""
    enricher = AlertEnricher(fake_service(answer))
    # The counterpart, host cone and identity lookups all use the same fake.
    enricher.match_service = enricher.host_service = enricher.service

    async def fake_d25(alert: Alert):
        return d25

    async def fake_cf4(pgc: int, ra: float, dec: float):
        if isinstance(cf4, Exception):
            raise cf4
        return (cf4 or {}).get(pgc)

    async def fake_group(pgc: int, ra: float, dec: float):
        if isinstance(groups, Exception):
            raise groups
        return (groups or {}).get(pgc)

    enricher._d25 = fake_d25  # type: ignore[method-assign]
    enricher._cf4_distance = fake_cf4  # type: ignore[method-assign]
    enricher._cf4_group = fake_group  # type: ignore[method-assign]
    return enricher


ALERT = Alert("fink", "ZTF26aaaaaaa", 150.0, 2.0, 61306.3, 19.0, "r", "SN candidate", 0.9, "")


async def test_partial_failure_leaves_dependent_flags_unknown() -> None:
    # Gaia DR3 times out; SIMBAD/NED answer with nothing at the position.
    res = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed={"gaia_dr3": "QueryTimeoutError"})
                              ).enrich(ALERT)
    assert res.status == "partial" and "gaia_dr3: QueryTimeoutError" in (res.error or "")
    assert res.known_star is None  # the parallax catalogue did not answer
    assert res.known_variable is False  # SIMBAD answered
    assert res.is_new is None  # not every catalogue answered
    # SIMBAD fails instead: the variable flag is unknown.
    res = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed={"simbad": "CatalogUnavailableError"})
                              ).enrich(ALERT)
    assert res.status == "partial" and res.known_variable is None and res.known_star is None and res.is_new is None
    # A counterpart found by the catalogues that answered settles is_new = False.
    star_row = {"source_id": "V* X", "ra": 150.0, "dec": 2.0, "separation_arcsec": 0.3, "data": {"otype": "RR*"}}
    res = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed={"gaia_dr3": "QueryTimeoutError"},
                                                              sources={"simbad": [star_row]})).enrich(ALERT)
    assert res.is_new is False and res.known_variable is True and res.stellar_counterpart is True
    assert res.known_star is True  # a catalogued star not projected on any galaxy
    # The D25 search fails as well: the Galactic nature of the star is unknown.
    res = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, sources={"simbad": [star_row]}),
                              d25=([], {"catalog": HYPERLEDA, "error_type": "QueryTimeoutError"}, False)).enrich(ALERT)
    assert res.known_star is None and res.status == "partial" and res.host_status == "incomplete"


def _d25_galaxy(ra: float, dec: float, alert: Alert, a: float) -> dict[str, Any]:
    sep = alerts.haversine_arcsec(ra, dec, alert.ra, alert.dec)
    return {"catalog": HYPERLEDA, "source_id": "PGC 1", "pgc": 1, "hyperleda_names": ["NGC1"], "ra": ra, "dec": dec,
            "separation_arcsec": sep, "semi_major_arcsec": a, "semi_minor_arcsec": a, "pa_deg": None, "dlr_arcsec": a,
            "d_dlr": sep / a}


@pytest.mark.parametrize(("z", "expected"), [(0.002, False), (0.05, True), (None, None)])
async def test_stellar_counterpart_inside_a_galaxy(z: float | None, expected: bool | None) -> None:
    """Synthetic: a SIMBAD star at the alert, inside a galaxy's D25 ellipse whose redshift is z."""
    star = {"source_id": "[X] 1", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.2, "data": {"otype": "*"}}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 20 / 3600, "dec": ALERT.dec, "separation_arcsec": 20.0,
              "data": {"otype": "G", "rvz_redshift": z}}

    def answer(ra, dec, cats):
        if "gaia_dr3" in cats:
            return record_of(ra, dec, cats, sources={"simbad": [star]})
        return record_of(ra, dec, cats, sources={"simbad": [galaxy]})

    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 60.0)], None, False)
    res = await enricher_with(answer, d25).enrich(ALERT)
    assert res.status == "done" and res.stellar_counterpart is True and res.known_star is expected
    if expected is True:  # a foreground star: no host
        assert res.host is None and res.host_status == "not_applicable_star"
    else:
        assert res.host is not None and res.host["name"] == "NGC 1" and res.host["method"] == "d25_ellipse"


async def test_gaia_parallax_needs_good_ruwe_inside_a_galaxy() -> None:
    """Synthetic: a 6-sigma Gaia parallax with RUWE 2.5 inside a galaxy is not taken as Galactic."""
    gaia = {"source_id": "1", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1,
            "data": {"parallax": 0.6, "parallax_error": 0.1, "ruwe": 2.5}}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.03}}

    def answer(ra, dec, cats):
        return record_of(ra, dec, cats, sources={"gaia_dr3": [gaia]} if "gaia_dr3" in cats else {"simbad": [galaxy]})

    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)
    inside = await enricher_with(answer, d25).enrich(ALERT)
    assert inside.status == "done" and inside.known_star is False and inside.host is not None
    assert any("parallax not used" in e for e in inside.evidence)
    outside = await enricher_with(answer).enrich(ALERT)  # no D25 galaxy: a binary star's RUWE is no reason to doubt
    assert outside.known_star is True and outside.host is None

    # Regression: Fink's own Gaia xmatch of the same 6-sigma source (no RUWE/G) must not override that verdict.
    fink = Alert.from_dict({**ALERT.as_dict(), "extra": {"gaia_parallax_mas": 0.6, "gaia_parallax_error_mas": 0.1,
                                                         "gaia_dr3_name": "Gaia DR3 1"}})
    same = await enricher_with(answer, d25).enrich(fink)
    assert same.known_star is False and same.host is not None and same.host_status == "found"
    assert any("judged from the Gaia DR3 row" in e for e in same.evidence)
    assert not any("-> Galactic star" in e for e in same.evidence)
    # ... nor when the counterpart search did not return that source.
    other = Alert.from_dict({**fink.as_dict(), "extra": {**fink.extra, "gaia_dr3_name": "Gaia DR3 999"}})
    alone = await enricher_with(answer, d25).enrich(other)
    assert alone.known_star is False and alone.host is not None
    assert any("broker gives no RUWE/G" in e for e in alone.evidence)
    # A >= 10 sigma broker parallax is secure even inside the galaxy; outside galaxies 5 sigma suffices.
    strong = Alert.from_dict({**other.as_dict(), "extra": {**other.extra, "gaia_parallax_error_mas": 0.05}})
    assert (await enricher_with(answer, d25).enrich(strong)).known_star is True
    assert (await enricher_with(answer).enrich(other)).known_star is True


@pytest.mark.parametrize(("data", "expected", "why"), [
    ({"parallax": 0.6, "parallax_error": 0.1, "ruwe": 2.5, "phot_g_mean_mag": 18.2}, True, "G < 19"),
    ({"parallax": 0.6, "parallax_error": 0.1, "ruwe": 1.1, "phot_g_mean_mag": 20.5}, True, "RUWE < 1.4"),
    ({"parallax": 1.2, "parallax_error": 0.1, "ruwe": 5.3, "phot_g_mean_mag": 20.5}, True, "secure even projected"),
    ({"parallax": 0.6, "parallax_error": 0.1, "ruwe": 2.5, "phot_g_mean_mag": 20.5}, False, "parallax not used"),
    # Proper motion: 5 mas/yr at 50 sigma is far above what any galaxy moves (0.16 mas/yr at 1 Mpc).
    ({"parallax": 0.05, "parallax_error": 0.1, "pmra": 3.0, "pmdec": 4.0, "pmra_error": 0.1, "pmdec_error": 0.1,
      "pmra_pmdec_corr": 0.2, "phot_g_mean_mag": 19.5}, True, "proper motion"),
    # ... but not at 3 sigma, nor 0.1 mas/yr (below the 0.16 mas/yr limit at 1 Mpc) even at 10 sigma.
    ({"pmra": 0.3, "pmdec": 0.4, "pmra_error": 0.1, "pmdec_error": 0.2, "phot_g_mean_mag": 20.0}, False, None),
    ({"pmra": 0.06, "pmdec": 0.08, "pmra_error": 0.006, "pmdec_error": 0.008, "phot_g_mean_mag": 18.0}, False, None),
])
async def test_gaia_foreground_criteria_inside_a_galaxy(data: dict[str, Any], expected: bool, why: str | None) -> None:
    """Synthetic: a Gaia source inside the D25 ellipse of a galaxy at 1 Mpc (CF4 DM 25.0, z = 0.0005)."""
    gaia = {"source_id": "7", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1, "data": data}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.0005}}

    def answer(ra, dec, cats):
        return record_of(ra, dec, cats, sources={"gaia_dr3": [gaia]} if "gaia_dr3" in cats else {"simbad": [galaxy]})

    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)
    res = await enricher_with(answer, d25, cf4={1: (25.0, 0.05)}).enrich(ALERT)
    assert res.status == "done" and res.known_star is expected
    if why is not None:
        assert any(why in e for e in res.evidence), res.evidence
    if expected:
        assert res.host is None and res.host_status == "not_applicable_star"
    else:
        assert res.host is not None and res.host["distance_method"] == "cosmicflows4"
        assert res.host["distance_mpc"] == pytest.approx(1.0 / 1.0005**2, rel=1e-6)  # D_A = D_L / (1+z)^2


async def test_bright_catalogued_star_on_a_galaxy_is_foreground() -> None:
    """Synthetic: a SIMBAD star with G = 14 on a galaxy at DM 29 (M_G = -15) is a foreground star;
    the same G without a stellar type (a nucleus or cluster could be that bright) decides nothing."""
    gaia = {"source_id": "8", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1,
            "data": {"phot_g_mean_mag": 14.0}}
    star = {"source_id": "[X] 2", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.2, "data": {"otype": "*"}}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.001}}
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)

    def answer_with(sources):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources=sources if "gaia_dr3" in cats else {"simbad": [galaxy]})

    res = await enricher_with(answer_with({"gaia_dr3": [gaia], "simbad": [star]}), d25, cf4={1: (29.0, 0.1)}).enrich(ALERT)
    assert res.status == "done" and res.known_star is True and any("M_G = -15.0" in e for e in res.evidence)
    res = await enricher_with(answer_with({"gaia_dr3": [gaia]}), d25, cf4={1: (29.0, 0.1)}).enrich(ALERT)
    assert res.known_star is False and res.host is not None


async def test_ned_extragalactic_and_galactic_star_types() -> None:
    def answer_for(otype):
        row = {"source_id": "S1", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.3, "data": {"prefphytype": otype}}
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={"ned": [row]} if "gaia_dr3" in cats else {})

    exg = await enricher_with(answer_for("exG*")).enrich(ALERT)  # not projected on any galaxy, yet extragalactic
    assert exg.known_star is False and exg.stellar_counterpart is True
    assert any("extragalactic star" in e for e in exg.evidence)
    mw = await enricher_with(answer_for("!*")).enrich(ALERT)
    assert mw.known_star is True and any("Milky Way object" in e for e in mw.evidence)
    flare = await enricher_with(answer_for("Flare*")).enrich(ALERT)
    assert flare.known_variable is True and flare.known_star is True


async def test_truncated_host_cone_makes_the_search_incomplete() -> None:
    def answer(ra, dec, cats):
        rec = record_of(ra, dec, cats)
        if "gaia_dr3" not in cats:
            rec.catalog_results["simbad"]["truncated"] = True
            rec.catalog_results["simbad"]["warnings"] = ["simbad: truncated at the 200 nearest rows"]
        return rec

    res = await enricher_with(answer).enrich(ALERT)
    assert res.host is None and res.host_status == "incomplete" and res.host_search_complete is False
    assert any("truncated" in e for e in res.evidence)


async def test_enrichment_failure_is_recorded_not_raised() -> None:
    def boom(ra, dec, cats):
        raise RuntimeError("archive exploded")

    down_d25 = ([], {"catalog": HYPERLEDA, "error_type": "RuntimeError"}, False)
    res = await enricher_with(boom, down_d25).enrich(recorded_alerts("xmatch_alerce")[0])
    assert res.status == "failed" and "archive exploded" in (res.error or "")
    assert res.host_status == "failed" and res.is_new is None and res.known_star is None


async def test_truncated_d25_or_failed_identity_makes_the_host_search_incomplete() -> None:
    """Synthetic: the alert lies inside a galaxy found only by HyperLEDA (outside the 60" cone)."""
    far = _d25_galaxy(ALERT.ra + 300 / 3600, ALERT.dec, ALERT, 600.0)
    empty = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats), ([far], None, False))
    ok = await empty.enrich(ALERT)
    assert ok.host is not None and ok.host["method"] == "d25_ellipse" and ok.host_search_complete is True
    truncated = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats), ([far], None, True)).enrich(ALERT)
    assert truncated.host is not None and truncated.host_search_complete is False
    assert any("D25 search truncated" in e for e in truncated.evidence)
    calls = {"n": 0}

    def identity_down(ra, dec, cats):
        calls["n"] += 1
        if abs(ra - far["ra"]) < 1e-9:  # the identity lookup at the galaxy's centre
            raise RuntimeError("NED down")
        return record_of(ra, dec, cats)

    res = await enricher_with(identity_down, ([far], None, False)).enrich(ALERT)
    assert res.host is not None and res.host["name"] == "PGC 1" and res.host_search_complete is False
    assert res.status == "partial" and any(f["catalog"] == "host_identity" for f in res.failures)
    # A failed Cosmicflows-4 lookup leaves the host without an offset and the enrichment partial.
    nocf4 = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats), ([far], None, False),
                                cf4=RuntimeError("VizieR down")).enrich(ALERT)
    assert nocf4.status == "partial" and nocf4.host["projected_offset_kpc"] is None
    assert any(f["catalog"] == alerts.COSMICFLOWS4 for f in nocf4.failures)
    assert any("Cosmicflows-4 lookup failed" in e for e in nocf4.evidence)


async def test_galaxy_named_only_by_a_transient_designation_is_not_adopted() -> None:
    """Synthetic (the NED situation of ZTF18aaawtyh): NED's only galaxy at the position is 'AT 2017abr'."""
    at = {"source_id": "AT 2017abr", "ra": ALERT.ra, "dec": ALERT.dec + 0.2 / 3600, "separation_arcsec": 0.2,
          "data": {"prefphytype": "G", "z": 0.206904}}
    # A z = 0.08 galaxy 4" away (0.7 typical light radii of 8 kpc; chance coincidence 1 - exp(-(4/60)^2) = 0.004:
    # the entry named after a transient is not a galaxy of the cone).
    far_galaxy = {"source_id": "WISEA J100000.00+020004.0", "ra": ALERT.ra, "dec": ALERT.dec + 4 / 3600,
                  "separation_arcsec": 4.0, "data": {"prefphytype": "G", "z": 0.08}}

    def answer_with(host_rows):
        def answer(ra, dec, cats):
            if "gaia_dr3" in cats:
                return record_of(ra, dec, cats, sources={"ned": [at]})
            return record_of(ra, dec, cats, sources={"ned": host_rows})
        return answer

    only = await enricher_with(answer_with([at])).enrich(ALERT)
    assert "AT 2017abr" in only.transient_designations
    assert only.host is None and only.host_status == "ambiguous_transient_entry"
    assert any("named only by a transient designation" in e for e in only.evidence)
    other = await enricher_with(answer_with([at, far_galaxy])).enrich(ALERT)
    assert other.host is not None and other.host["name"] == "WISEA J100000.00+020004.0" and other.status == "done"
    assert other.host["method"] == "nearest" and other.host["p_chance"] == pytest.approx(1 - math.exp(-((4 / 60) ** 2)))
    # 30" away the same galaxy is 9 typical light radii off: not a host (Gupta et al. 2016: d_DLR < 4 second-moment
    # radii ~ 2 D25 radii), and the host search says galaxies were found but none is associated.
    far = {**far_galaxy, "source_id": "WISEA J100000.00+020030.0", "dec": ALERT.dec + 30 / 3600, "separation_arcsec": 30.0}
    distant = await enricher_with(answer_with([at, far])).enrich(ALERT)
    assert distant.host is None and distant.host_status == "unassociated" and distant.host_search_complete is True
    assert any("WISEA J100000.00+020030.0 at 30.0\" not adopted" in e and "typical light radii" in e
               for e in distant.evidence), distant.evidence


async def test_all_catalogs_failing_marks_enrichment_failed(store: AlertStore) -> None:
    down = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed=dict.fromkeys(cats, "CatalogUnavailableError")),
                         d25=([], {"catalog": HYPERLEDA, "error_type": "CatalogUnavailableError"}, False))
    alert = recorded_alerts("xmatch_alerce")[0]
    res = await down.enrich(alert)
    assert res.status == "failed" and "CatalogUnavailableError" in (res.error or "") and res.exception is None
    assert res.is_new is None and res.known_star is None and res.host_status == "failed"
    store.upsert(alert)
    svc = AlertService(store, offline_client(), down)
    await svc.enrich_alert(alert)
    assert store.crossmatch_status(alert.alert_id) == "failed"


async def test_enricher_exception_is_stored_with_its_type(store: AlertStore) -> None:
    enricher = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats))

    def broken(*args: Any, **kwargs: Any) -> None:
        raise TypeError("injected bug")

    enricher._flags = broken  # type: ignore[method-assign]
    store.upsert(ALERT)
    enrichment, stored = await AlertService(store, offline_client(), enricher).enrich_alert(ALERT)
    assert enrichment.status == "failed" and enrichment.exception == "TypeError: injected bug" and stored == "stored"
    assert store.get(ALERT.alert_id)["last_crossmatch_error"] == "TypeError: injected bug"


async def test_failed_recrossmatch_keeps_the_previous_enrichment(store: AlertStore) -> None:
    galaxy = {"source_id": "CGCG 414-037", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.0213}}
    good = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats
                                                         else {"simbad": [galaxy]}))
    down = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed=dict.fromkeys(cats, "CatalogUnavailableError")))
    store.upsert(ALERT)
    await AlertService(store, offline_client(), good).enrich_alert(ALERT)
    before = store.get(ALERT.alert_id)
    assert before["crossmatch_status"] == "done" and before["host_name"] == "CGCG 414-037" and before["is_new"] is True
    enrichment, stored = await AlertService(store, offline_client(), down).enrich_alert(ALERT)
    assert enrichment.status == "failed" and stored == "kept_previous"
    after = store.get(ALERT.alert_id)
    for key in ("crossmatch_status", "host_name", "host_redshift", "is_new", "known_star", "enrichment"):
        assert after[key] == before[key], key
    # An outage (every archive unreachable) is recorded, but is not counted as an attempt.
    assert "CatalogUnavailableError" in after["last_crossmatch_error"]
    assert (after["crossmatch_attempts"], after["crossmatch_outages"]) == (1, 1)
    # A partial re-run does not replace a complete one either.
    partial = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed={"gaia_dr3": "QueryTimeoutError"}))
    assert (await AlertService(store, offline_client(), partial).enrich_alert(ALERT))[1] == "kept_previous"
    assert store.get(ALERT.alert_id)["enrichment"] == before["enrichment"]


async def test_partial_rows_are_retried_by_later_polls(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    gaia_down = {"value": True}
    enricher = enricher_with(lambda ra, dec, cats: record_of(
        ra, dec, cats, failed={"gaia_dr3": "QueryTimeoutError"} if gaia_down["value"] else {}))

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert.from_dict(ALERT.as_dict())] if since_mjd < 61306.5 else [])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    now = {"t": 61307.0}
    svc = AlertService(store, offline_client(), enricher, clock=lambda: now["t"])
    first = await svc.poll("fink", since_mjd=61306.0)
    assert (first.inserted, first.crossmatch_partial, first.retried) == (1, 1, 0)
    row = store.get(ALERT.alert_id)
    assert row["crossmatch_status"] == "partial"
    # A Gaia timeout is an outage: not an attempt, but the retry backs off (5 minutes after the first).
    assert (row["crossmatch_attempts"], row["crossmatch_outages"]) == (0, 1)
    assert row["next_crossmatch_mjd"] == pytest.approx(61307.0 + alerts.RETRY_BACKOFF_SECONDS / 86400)
    gaia_down["value"] = False
    assert (await svc.poll("fink", since_mjd=61306.6)).retried == 0  # a poll a moment later: not due yet
    now["t"] += 600 / 86400
    # A window without the alert: the retry sweep re-enriches the stored partial row.
    second = await svc.poll("fink", since_mjd=61306.6)
    assert (second.fetched, second.retried, second.crossmatched) == (0, 1, 1)
    row = store.get(ALERT.alert_id)
    assert row["crossmatch_status"] == "done" and row["known_star"] is False and row["is_new"] is True
    assert (row["crossmatch_attempts"], row["crossmatch_outages"]) == (1, 1) and row["last_crossmatch_error"] is None
    assert row["next_crossmatch_mjd"] is None
    assert store.incomplete(broker="fink", now_mjd=now["t"] + 10) == []  # done rows are not retried


async def test_outage_attempts_back_off_and_do_not_use_up_the_attempt_cap(store: AlertStore,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: every watch cycle re-crossmatched the alerts of the overlap window while NED timed out, so a
    30-minute outage used up MAX_CROSSMATCH_ATTEMPTS (in 25 minutes) and the rows stayed partial forever."""
    ned = {"down": True}
    calls = {"n": 0}

    def answer(ra, dec, cats):
        calls["n"] += 1
        return record_of(ra, dec, cats, failed={"ned": "QueryTimeoutError"} if ned["down"] else {})

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert.from_dict(ALERT.as_dict())])  # the alert is in every window

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    now = {"t": 61306.35}
    svc = AlertService(store, offline_client(), enricher_with(answer), clock=lambda: now["t"])
    results = []
    for cycle in range(24):  # 2 h of watch cycles 5 minutes apart; NED is back after 30 minutes
        ned["down"] = cycle < 6
        results.append(await svc.poll("fink", limit=5))
        now["t"] += 300 / 86400
    row = store.get(ALERT.alert_id)
    assert row["crossmatch_status"] == "done" and row["crossmatch_attempts"] == 1
    # Attempts at 0, 5, 15 and 35 minutes (backoff 5, 10, 20 min): the 35-minute one, after the outage, succeeds.
    assert [r.crossmatch_partial + r.crossmatched for r in results[:8]] == [1, 1, 0, 1, 0, 0, 0, 1]
    assert row["crossmatch_outages"] == 3 and calls["n"] == 2 * 4
    assert all(r.crossmatch_capped == 0 for r in results)
    assert sum(r.crossmatch_backoff for r in results) == 4
    assert any("retry backs off" in w for w in results[2].warnings)


async def test_partial_rows_stop_being_retried_after_max_attempts(store: AlertStore,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete crossmatch that is not an outage (NED answers an error) is retried at most
    MAX_CROSSMATCH_ATTEMPTS times by the polls, then once a day by the sweep."""
    calls = {"n": 0}

    def ned_broken(ra, dec, cats):
        calls["n"] += 1
        return record_of(ra, dec, cats, failed={"ned": "CatalogQueryError"})

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert.from_dict(ALERT.as_dict())])  # the alert is in every window

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    now = {"t": 61306.35}
    svc = AlertService(store, offline_client(), enricher_with(ned_broken), clock=lambda: now["t"])
    results = []
    for _ in range(8):  # polls 2 hours apart: every backoff (at most 80 minutes before the cap) has passed
        results.append(await svc.poll("fink", limit=5))
        now["t"] += 2 / 24
    row = store.get(ALERT.alert_id)
    cap = alerts.MAX_CROSSMATCH_ATTEMPTS
    assert row["crossmatch_status"] == "partial" and row["crossmatch_attempts"] == cap and row["crossmatch_outages"] == 0
    assert calls["n"] == 2 * cap  # counterpart + host cone per attempt
    assert [r.crossmatch_partial for r in results] == [1] * cap + [0] * (8 - cap)
    assert all(r.crossmatch_capped == 1 and any("attempts already made" in w for w in r.warnings) for r in results[cap:])
    # The retry sweep is capped as well -- for a day after the last attempt.
    assert store.incomplete(broker="fink", now_mjd=now["t"]) == []
    last = row["next_crossmatch_mjd"] - alerts.CAPPED_RETRY_DAYS
    assert [a.alert_id for a in store.incomplete(broker="fink", now_mjd=last + 1.01)] == [ALERT.alert_id]
    # A moved alert (new position) starts again at once.
    moved = Alert.from_dict({**ALERT.as_dict(), "mjd": ALERT.mjd + 1.0, "dec": ALERT.dec + 5.0 / 3600})

    async def fetch_moved(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert.from_dict(moved.as_dict())])

    monkeypatch.setattr(alerts, "fetch_alerts", fetch_moved)
    again = await svc.poll("fink", limit=5)
    assert again.updated == 1 and again.crossmatch_partial == 1 and store.get(ALERT.alert_id)["crossmatch_attempts"] == 1


async def test_retry_sweep_and_bare_id_lookup_are_scoped(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two brokers' rows of the same ZTF object: the sweep retries only the polled broker's rows and
    a bare object id returns the newest row."""
    enricher = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats))
    fink_row = Alert.from_dict({**ALERT.as_dict(), "mjd": 61306.3})
    alerce_row = Alert.from_dict({**ALERT.as_dict(), "broker": "alerce", "mjd": 61306.9})
    store.upsert(fink_row)
    store.upsert(alerce_row)
    assert store.get(ALERT.object_id)["id"] == alerce_row.alert_id  # newest first
    assert [a.alert_id for a in store.incomplete(broker="fink")] == [fink_row.alert_id]

    async def nothing(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([])

    monkeypatch.setattr(alerts, "fetch_alerts", nothing)
    res = await AlertService(store, offline_client(), enricher, clock=lambda: 61307.0).poll("fink", since_mjd=61306.5)
    assert res.retried == 1 and res.crossmatched == 1
    assert store.crossmatch_status(fink_row.alert_id) == "done"
    assert store.crossmatch_status(alerce_row.alert_id) == "pending"


async def test_concurrent_crossmatch_of_one_alert_runs_once(store: AlertStore) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    class Slow:
        match_radius_arcsec, host_radius_arcsec, catalogs = 2.0, 60.0, ["gaia_dr3"]

        async def enrich(self, alert: Alert) -> AlertEnrichment:
            calls["n"] += 1
            started.set()
            await release.wait()
            return AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"])

    store.upsert(ALERT)
    svc = AlertService(store, offline_client(), Slow())  # type: ignore[arg-type]
    first = asyncio.create_task(svc.crossmatch_alerts([ALERT]))
    await started.wait()
    second = await asyncio.wait_for(svc.crossmatch_alerts([ALERT, ALERT]), timeout=10)  # never waits on the first
    release.set()
    assert second == {"done": 0, "partial": 0, "failed": 0, "skipped_in_flight": 2}
    assert (await first)["done"] == 1 and calls["n"] == 1


# ---------------------------------------------------------------------------
# Store: dedupe, idempotency, older detections, migration
# ---------------------------------------------------------------------------


def test_store_upsert_is_idempotent_and_tracks_updates(store: AlertStore) -> None:
    alert = recorded_alerts("xmatch_alerce")[0]
    assert store.upsert(alert) == "inserted"
    assert store.upsert(alert) == "unchanged"
    assert store.count() == 1
    newer = Alert.from_dict({**alert.as_dict(), "mjd": alert.mjd + 1.0, "magpsf": 17.5})
    assert store.upsert(newer) == "updated"
    row = store.get(alert.alert_id)
    assert row["mjd"] == alert.mjd + 1.0 and row["magpsf"] == 17.5 and row["n_updates"] == 1
    assert row["crossmatch_status"] == "pending"
    # Bare object id lookup and filters.
    assert store.get(alert.object_id)["id"] == alert.alert_id
    assert store.latest_mjd("alerce") == alert.mjd + 1.0 and store.latest_mjd("fink") is None
    assert store.list(broker="fink") == [] and len(store.list(since_mjd=alert.mjd)) == 1
    assert store.list(since_mjd=alert.mjd + 2.0) == []
    assert [r["id"] for r in store.get_many([alert.alert_id, "alerce:nope", alert.alert_id])] == [alert.alert_id]


def test_store_older_alert_never_overwrites_a_newer_one(store: AlertStore) -> None:
    """Regression: a backfill poll returning an older Fink alert must not mix two detections."""
    base = {"broker": "fink", "object_id": "ZTF26abtnjsl", "ra": 39.6342382, "dec": 3.9232084, "classification": "SN candidate",
            "url": "", "first_mjd": 61300.0}
    newer = Alert.from_dict({**base, "mjd": 61310.4, "magpsf": 18.1, "band": "r", "probability": 0.91, "is_negative": False,
                             "extra": {"candid": "NEW"}})
    older = Alert.from_dict({**base, "mjd": 61305.3, "magpsf": 19.9, "band": "g", "probability": 0.62, "is_negative": True,
                             "ra": 39.6342390, "extra": {"candid": "OLD"}})
    assert store.upsert(newer) == "inserted"
    assert store.upsert(older) == "unchanged"
    row = store.get(newer.alert_id)
    assert (row["mjd"], row["magpsf"], row["band"], row["probability"], row["is_negative"], row["ra"]) == (
        61310.4, 18.1, "r", 0.91, False, 39.6342382)
    assert row["extra"] == {"candid": "NEW"} and row["n_updates"] == 0
    # An older alert that reveals an earlier first detection only extends first_mjd.
    earliest = Alert.from_dict({**older.as_dict(), "first_mjd": 61290.0})
    assert store.upsert(earliest) == "updated"
    row = store.get(newer.alert_id)
    assert row["first_mjd"] == 61290.0 and row["magpsf"] == 18.1 and row["extra"] == {"candid": "NEW"}
    # The same detection re-scored (same mjd) updates the classification only as data of that detection.
    rescored = Alert.from_dict({**newer.as_dict(), "first_mjd": 61290.0, "probability": 0.95})
    assert store.upsert(rescored) == "updated" and store.get(newer.alert_id)["probability"] == 0.95


def test_store_requeues_crossmatch_when_position_moves(store: AlertStore) -> None:
    alert = recorded_alerts("xmatch_alerce")[0]
    store.upsert(alert)
    enrichment = AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0,
                                 catalogs=["gaia_dr3"], is_new=True, known_star=False)
    store.set_enrichment(alert.alert_id, enrichment)
    row = store.get(alert.alert_id)
    assert row["crossmatch_status"] == "done" and row["is_new"] is True and row["known_star"] is False
    assert store.list(only_new=True)[0]["id"] == alert.alert_id and store.list(only_new=False) == []
    small = Alert.from_dict({**alert.as_dict(), "mjd": alert.mjd + 0.1, "dec": alert.dec + 0.5 / 3600})
    assert store.upsert(small) == "updated" and store.crossmatch_status(alert.alert_id) == "done"
    moved = Alert.from_dict({**alert.as_dict(), "mjd": alert.mjd + 0.2, "dec": alert.dec + 3.0 / 3600})
    assert store.upsert(moved) == "updated" and store.crossmatch_status(alert.alert_id) == "pending"
    assert store.get(alert.alert_id)["crossmatch_attempts"] == 0


def test_store_migrates_a_first_release_table(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE alerts (id TEXT PRIMARY KEY, broker TEXT NOT NULL, object_id TEXT NOT NULL, survey TEXT, "
        "ra DOUBLE PRECISION NOT NULL, dec DOUBLE PRECISION NOT NULL, mjd DOUBLE PRECISION NOT NULL, first_mjd DOUBLE PRECISION, "
        "magpsf DOUBLE PRECISION, magpsf_err DOUBLE PRECISION, band TEXT, classification TEXT, probability DOUBLE PRECISION, "
        "url TEXT, extra_json TEXT, crossmatch_status TEXT NOT NULL, enrichment_json TEXT, is_new INTEGER, known_star INTEGER, "
        "known_variable INTEGER, host_name TEXT, host_separation_arcsec DOUBLE PRECISION, host_redshift DOUBLE PRECISION, "
        "first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL, n_updates INTEGER NOT NULL DEFAULT 0, UNIQUE (broker, object_id))"
    )
    conn.execute("INSERT INTO alerts VALUES ('fink:ZTF1', 'fink', 'ZTF1', 'ztf', 1.0, 2.0, 61300.0, NULL, 19.0, NULL, 'r', "
                 "'SN candidate', 0.9, '', '{}', 'done', NULL, 1, 0, 0, NULL, NULL, NULL, 't', 't', 0)")
    conn.commit()
    conn.close()
    migrated = AlertStore(MetadataStore(f"sqlite:///{path.as_posix()}"))
    row = migrated.get("fink:ZTF1")
    assert row["is_negative"] is None and row["crossmatch_attempts"] == 0 and row["last_crossmatch_error"] is None
    assert row["is_new"] is True and row["magpsf"] == 19.0


# ---------------------------------------------------------------------------
# Service: poll -> persist -> crossmatch; windows, backlog, watch loop
# ---------------------------------------------------------------------------


async def test_poll_persists_crossmatches_and_repoll_is_idempotent(store: AlertStore) -> None:
    p = params("alerce")
    with Replay("alerce", "xmatch_alerce") as replay:
        async with offline_client() as client:
            svc = AlertService(store, client, AlertEnricher(make_service(client)))
            first = await svc.poll("alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"], limit=p["limit"],
                                   options=p["options"])
            assert (first.fetched, first.inserted, first.crossmatched, first.crossmatch_failed) == (3, 3, 3, 0)
            assert first.new_alert_ids == first.alert_ids and first.window == "explicit"
            assert first.truncated and "window truncated at limit=3" in first.warnings[0]
            assert f"until_mjd={first.boundary_mjd:.6f}" in first.warnings[0]
            n_calls = len(replay.calls)
            # objects + 3 classifier versions + 3 detections; 3 alerts x (3 counterpart + 2 host-cone + 1 HyperLEDA)
            # queries, plus the Cosmicflows-4 galaxy and group lookups for the host of ZTF26abxsxww (a PGC galaxy
            # of unknown redshift, not in CF4 nor in a Tully 2015 group).
            assert n_calls == 7 + 20 == 7 + len(meta("xmatch_alerce")["exchanges"])
            second = await svc.poll("alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"], limit=p["limit"],
                                    options=p["options"])
            assert (second.inserted, second.updated, second.unchanged, second.crossmatched, second.retried) == (0, 0, 3, 0, 0)
            assert len(replay.calls) == n_calls + 7  # broker only: nothing is crossmatched twice
    rows = store.list()
    assert len(rows) == 3 and all(r["crossmatch_status"] == "done" and r["crossmatch_attempts"] == 1 for r in rows)
    new_one = store.get("alerce:ZTF26abxsxww")
    assert new_one["is_new"] is True and new_one["enrichment"]["host"]["name"] == "LEDA 1136493"
    # ZTF26abxsysn sits on a common-proper-motion pair of Gaia DR3 stars (9.8 mas/yr at 116 and 85
    # sigma): a Galactic star, although NED types the blended WISE source there as a galaxy.
    old_one = store.get("alerce:ZTF26abxsysn")
    assert old_one["is_new"] is False and old_one["known_star"] is True and old_one["host_name"] is None
    assert old_one["enrichment"]["host_status"] == "not_applicable_star"
    assert any("proper motion 9.79 mas/yr" in e for e in old_one["enrichment"]["evidence"])
    assert all(r["is_negative"] is False for r in rows)


async def test_poll_window_defaults_to_cursor(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    windows: list[tuple[float, float]] = []
    recorded = AlerceBroker.parse_objects(body("alerce", 0))[0][:3]

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        windows.append((since_mjd, until_mjd))
        return FetchResult([Alert.from_dict(a.as_dict()) for a in recorded])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    svc = AlertService(store, offline_client(), None, clock=lambda: 61311.0, lookback_days=2.0, overlap_days=0.5)
    first = await svc.poll("alerce", crossmatch=False)
    second = await svc.poll("alerce", crossmatch=False)
    # First poll: lookback; then the stream cursor (end of the last complete window) minus the overlap.
    assert windows == [(61309.0, 61311.0), (61310.5, 61311.0)]
    assert (first.window, second.window) == ("new", "new") and first.backlog is None
    # Another filter is another stream: without its own cursor it starts from the broker's latest alert.
    await svc.poll("alerce", crossmatch=False, options={"class_name": "AGN"})
    assert windows[-1] == (pytest.approx(max(a.mjd for a in recorded) - 0.5), 61311.0)
    with pytest.raises(ValueError):
        await svc.poll("alerce", since_mjd=61312.0, crossmatch=False)
    with pytest.raises(RuntimeError, match="enricher"):
        await AlertService(AlertStore(store.metadata), offline_client(), None, clock=lambda: 61311.0).poll(
            "fink", crossmatch=True)


@pytest.mark.parametrize("scenario", ["fink_backlog", "alerce_backlog"])
async def test_truncated_window_is_ingested_completely_via_the_backlog(store: AlertStore, scenario: str) -> None:
    """Recorded: limit 2 on a real window of 5-12 objects; default-window polls walk the backlog to the end."""
    p = params(scenario)
    with Replay(scenario):
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: p["until_mjd"], lookback_days=p["lookback_days"],
                               overlap_days=0.25)
            results = []
            for _ in range(len(p["polls"])):
                results.append(await svc.poll(p["broker"], limit=2, crossmatch=False, options=p["options"]))
            reference = await fetch_alerts(client, p["broker"], since_mjd=results[0].since_mjd, until_mjd=p["until_mjd"],
                                           limit=100, options=p["options"])
    assert [r.window for r in results] == [q["window"] for q in p["polls"]]
    assert results[0].window == "new" and results[0].truncated and results[0].backlog is not None
    assert results[1].window == "backlog" and results[-1].window == "new" and results[-1].backlog is None
    assert all(r.fetched <= 2 for r in results)
    # Every object of the window is stored exactly once, although each poll kept only 2.
    reference_ids = sorted(a.alert_id for a in reference.alerts)
    assert not reference.truncated and len(reference_ids) >= 5 and reference_ids == p["reference_ids"]
    assert sorted(r["id"] for r in store.list(limit=100)) == reference_ids
    assert sum(r.inserted for r in results) == len(reference_ids)
    # Backlog windows only shrink: each ends at the boundary of the poll before.
    for prev, cur in itertools.pairwise(results):
        if cur.window == "backlog":
            assert cur.until_mjd == pytest.approx(prev.backlog["until_mjd"]) and cur.until_mjd <= prev.until_mjd


async def test_truncated_overlap_does_not_rewalk_covered_time(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic: after a complete poll, a later truncated window whose cut lies in the re-covered overlap leaves no backlog."""
    answers = [FetchResult([]), FetchResult([Alert("fink", "ZTF1", 1.0, 1.0, 61310.8, 19.0, "r", "SN candidate", 0.9, "")],
                                            truncated=True, boundary_mjd=61310.8)]

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return answers.pop(0)

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    now = {"t": 61310.9}
    svc = AlertService(store, offline_client(), None, clock=lambda: now["t"], overlap_days=1.0)
    await svc.poll("fink", limit=1, crossmatch=False)  # complete: cursor at 61310.9
    now["t"] = 61311.0
    res = await svc.poll("fink", limit=1, crossmatch=False)  # [61309.9, 61311.0] cut at 61310.8 < 61310.9
    assert res.truncated and res.backlog is None
    assert any("re-covered overlap" in w for w in res.warnings)
    now["t"] = 61311.5
    answers.append(FetchResult([Alert("fink", "ZTF2", 1.0, 1.0, 61311.4, 19.0, "r", "SN candidate", 0.9, "")],
                               truncated=True, boundary_mjd=61311.4))
    res = await svc.poll("fink", limit=1, crossmatch=False)  # the cut lies after the last resume point (61311.0)
    assert res.backlog == {"since_mjd": 61311.0, "until_mjd": 61311.4}


async def test_backlog_whose_newest_second_is_overfull_steps_past_it(store: AlertStore,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic: more than `limit` objects at the backlog's newest instant must not repeat the same window forever."""
    windows: list[tuple[float, float]] = []
    edge = 61310.5

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        windows.append((since_mjd, until_mjd))
        if len(windows) == 1:  # the first window is cut at `edge`
            return FetchResult([Alert("fink", "ZTF1", 1.0, 1.0, 61310.8, 19.0, "r", "SN candidate", 0.9, "")],
                               truncated=True, boundary_mjd=edge)
        if len(windows) == 2:  # the backlog [.., edge] is cut again at its own top second
            return FetchResult([Alert("fink", "ZTF2", 1.0, 1.0, edge, 19.0, "r", "SN candidate", 0.9, "")],
                               truncated=True, boundary_mjd=edge)
        return FetchResult([])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    svc = AlertService(store, offline_client(), None, clock=lambda: 61311.0, lookback_days=1.0)
    first = await svc.poll("fink", limit=1, crossmatch=False)
    second = await svc.poll("fink", limit=1, crossmatch=False)
    third = await svc.poll("fink", limit=1, crossmatch=False)
    assert first.backlog == {"since_mjd": 61310.0, "until_mjd": edge}
    assert second.window == "backlog" and second.until_mjd == edge
    assert second.backlog["until_mjd"] == pytest.approx(edge - 1.0 / 86400.0, abs=1e-9)
    assert any("rest of that second is skipped" in w for w in second.warnings)
    assert third.window == "backlog" and third.until_mjd == pytest.approx(edge - 1.0 / 86400.0, abs=1e-9)
    assert third.backlog is None


async def test_fink_paging_boundary_after_max_pages_is_the_oldest_row_reached(store: AlertStore) -> None:
    """Regression: one object with 200 alerts in the window hid 3 older objects behind a boundary at
    its newest alert; the walk now pages faster (n doubles) and resumes from the oldest row reached."""
    until = 61311.0
    rows = [("A", until - 0.001 * (i + 1)) for i in range(200)] + [(o, until - 0.3 - 0.01 * k) for k, o in enumerate("BCD")]
    sent: list[int] = []

    def iso_to_mjd(text: str) -> float:
        from datetime import datetime

        stamp = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=alerts.UTC)
        return (stamp - alerts.MJD_EPOCH).total_seconds() / 86400.0

    def latests(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        lo, hi, n = iso_to_mjd(q["startdate"]), iso_to_mjd(q["stopdate"]), int(q["n"])
        sent.append(n)
        sel = sorted([r for r in rows if lo <= r[1] <= hi], key=lambda r: -r[1])[:n]
        return httpx.Response(200, json=[_fink_row(o, m + alerts.JD_MJD_OFFSET) for o, m in sel])

    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(side_effect=latests)
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: until, lookback_days=1.0)
            results = [await svc.poll("fink", limit=2, crossmatch=False) for _ in range(4)]
            capped = await FinkZTFBroker().fetch(client, since_mjd=until - 1.0, until_mjd=until, limit=2)
    assert sorted(r["object_id"] for r in store.list(limit=10)) == ["A", "B", "C", "D"]
    assert sent[:4] == [3, 3, 6, 12]  # n doubles on pages that bring no new object
    assert [r.window for r in results[:3]] == ["new", "backlog", "backlog"] and results[2].backlog is None
    assert [a.object_id for a in capped.alerts] == ["A", "B"] and capped.truncated
    assert capped.boundary_mjd == pytest.approx(until - 0.3, abs=1e-6)  # C exists beyond the limit
    # With MAX_PAGES too small to reach B, the boundary is where paging stopped, not A's newest alert.
    old_max = alerts.MAX_PAGES
    try:
        alerts.MAX_PAGES = 3
        with respx.mock(assert_all_mocked=True) as mock:
            mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(side_effect=latests)
            async with offline_client() as client:
                short = await FinkZTFBroker().fetch(client, since_mjd=until - 1.0, until_mjd=until, limit=2)
    finally:
        alerts.MAX_PAGES = old_max
    assert [a.object_id for a in short.alerts] == ["A"] and short.truncated
    # 3 + 3 + 6 rows of A were read: the oldest row reached is A's 10th alert.
    assert short.boundary_mjd == pytest.approx(until - 0.010, abs=1e-6)
    assert any("stopped after 3 requests" in w for w in short.warnings)


async def test_watch_loop_iterations_errors_and_cancellation(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = AlerceBroker.parse_objects(body("alerce", 0))[0][:3]
    calls: list[str] = []

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        calls.append(broker)
        if broker == "fink":
            raise BrokerError("fink", "HTTP 503 from https://api.ztf.fink-portal.org/api/v1/latests", 503, unreachable=True)
        return FetchResult([Alert.from_dict(a.as_dict()) for a in recorded])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    svc = AlertService(store, offline_client(), None, clock=lambda: 61311.0)
    seen: list[alerts.PollResult] = []
    results = await svc.watch(["alerce", "fink"], interval_seconds=0.01, crossmatch=False, iterations=2, on_result=seen.append)
    assert calls == ["alerce", "fink", "alerce", "fink"] and results == seen
    assert results[0].inserted == 3 and results[2].unchanged == 3
    assert results[1].error and "503" in results[1].error and results[1].since_mjd is None
    assert store.count() == 3

    # Graceful cancellation mid-wait: CancelledError propagates, everything polled is committed.
    first = asyncio.Event()
    task = asyncio.create_task(svc.watch(["alerce"], interval_seconds=3600, crossmatch=False,
                                         on_result=lambda r: first.set()))
    await asyncio.wait_for(first.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.count() == 3
    # stop_event ends the loop cleanly.
    stop = asyncio.Event()
    stop.set()
    assert await svc.watch(["alerce"], interval_seconds=1, crossmatch=False, stop_event=stop) == []
    for bad in ({"brokers": ["nope"]}, {"brokers": ["alerce"], "interval_seconds": 0},
                {"brokers": ["alerce"], "limit": 0}, {"brokers": []}):
        with pytest.raises(ValueError):
            await svc.watch(**{"interval_seconds": 1, **bad})


# ---------------------------------------------------------------------------
# REST router
# ---------------------------------------------------------------------------


@pytest.fixture
def api(store: AlertStore) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.alert_store = store
    return TestClient(app)


def test_router_poll_list_get_and_recrossmatch(api: TestClient) -> None:
    p = params("alerce")
    payload = {"broker": "alerce", "since_mjd": p["since_mjd"], "until_mjd": p["until_mjd"], "limit": p["limit"],
               "class_name": "SN", "classifier": "stamp_classifier", "mjd_field": "firstmjd"}
    with Replay("alerce", "xmatch_alerce"):
        # Default: the poll answers after ingest; the crossmatch runs as a background task.
        res = api.post("/api/v1/alerts/poll", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert (data["fetched"], data["inserted"], data["crossmatched"], data["truncated"]) == (3, 3, 0, True)
        assert data["crossmatch_deferred"] is True and data["crossmatch_queued"] == 3
        assert [a["id"] for a in data["alerts"]] == data["alert_ids"]
        assert all(a["crossmatch_status"] == "pending" and a["is_negative"] is False for a in data["alerts"])
        # The background task (run by the server after the answer) stored the enrichments.
        listed = api.get("/api/v1/alerts").json()["alerts"]
        assert len(listed) == 3 and all(a["crossmatch_status"] == "done" and a["enrichment"]["status"] == "done"
                                        for a in listed)
        again = api.post("/api/v1/alerts/poll", json=payload).json()
        assert (again["inserted"], again["unchanged"], again["crossmatched"], again["crossmatch_queued"]) == (0, 3, 0, 0)
        rerun = api.post("/api/v1/alerts/alerce:ZTF26abxsxww/crossmatch")
        assert rerun.status_code == 200 and rerun.json()["is_new"] is True and rerun.json()["crossmatch_attempts"] == 2

    listing = api.get("/api/v1/alerts", params={"since_mjd": p["since_mjd"], "limit": 2}).json()
    assert listing["count"] == 2 and listing["alerts"][0]["mjd"] >= listing["alerts"][1]["mjd"]
    assert api.get("/api/v1/alerts", params={"only_new": True}).json()["count"] == 1
    assert api.get("/api/v1/alerts", params={"crossmatch_status": "partial"}).json()["count"] == 0
    one = api.get("/api/v1/alerts/alerce:ZTF26abxsysn")
    assert one.status_code == 200 and one.json()["object_id"] == "ZTF26abxsysn"
    assert api.get("/api/v1/alerts/ZTF26abxsysn").json()["id"] == "alerce:ZTF26abxsysn"
    assert api.get("/api/v1/alerts/alerce:ZTF00nothere").status_code == 404
    assert api.post("/api/v1/alerts/alerce:ZTF00nothere/crossmatch").status_code == 404
    brokers = api.get("/api/v1/alerts/brokers").json()
    assert [b["name"] for b in brokers] == ["alerce", "fink", "fink_lsst"]


def test_router_validation_and_upstream_errors(api: TestClient) -> None:
    assert api.post("/api/v1/alerts/poll", json={"broker": "antares"}).status_code == 422
    assert api.post("/api/v1/alerts/poll", json={"broker": "alerce", "limit": 0}).status_code == 422
    assert api.post("/api/v1/alerts/poll", json={"broker": "fink", "classifier": "lc_classifier"}).status_code == 422
    big_wait = api.post("/api/v1/alerts/poll", json={"broker": "fink", "limit": 26, "crossmatch_mode": "wait"})
    assert big_wait.status_code == 422 and "background" in big_wait.json()["detail"]
    assert api.post("/api/v1/alerts/poll", json={"broker": "fink", "crossmatch_mode": "later"}).status_code == 422
    bad_window = api.post("/api/v1/alerts/poll", json={"broker": "fink", "since_mjd": 61300.0, "until_mjd": 61299.0,
                                                        "crossmatch": False})
    assert bad_window.status_code == 422 and "earlier" in bad_window.json()["detail"]
    assert api.get("/api/v1/alerts", params={"limit": 0}).status_code == 422
    assert api.get("/api/v1/alerts", params={"broker": "nope"}).status_code == 422
    for scenario, text in (("invalid_fink_lsst", "not a valid tag"), ("unsupported_fink_lsst", "Livestream")):
        with Replay(scenario):
            p = params(scenario)
            res = api.post("/api/v1/alerts/poll", json={"broker": "fink_lsst", "since_mjd": p["since_mjd"],
                                                         "until_mjd": p["until_mjd"], "limit": 2,
                                                         "class_name": p["options"]["class_name"], "crossmatch": False})
            assert res.status_code == 422 and text in res.json()["detail"]
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get(url__startswith="https://api.ztf.fink-portal.org/").mock(return_value=httpx.Response(503, text="down"))
        res = api.post("/api/v1/alerts/poll", json={"broker": "fink", "since_mjd": 61300.0, "until_mjd": 61301.0,
                                                     "crossmatch": False})
        assert res.status_code == 502 and "upstream" in res.json()["detail"]


def test_router_poll_without_body_uses_defaults(api: TestClient) -> None:
    classifiers = body("invalid_alerce", 1)  # the real /classifiers/ answer
    with respx.mock(assert_all_mocked=True) as mock:
        objects = mock.get("https://api.alerce.online/ztf/v1/objects/").mock(
            return_value=httpx.Response(200, json={"items": [], "total": None, "has_next": False}))
        mock.get("https://api.alerce.online/ztf/v1/classifiers/").mock(return_value=httpx.Response(200, json=classifiers))
        res = api.post("/api/v1/alerts/poll")
        assert res.status_code == 200, res.text
        data = res.json()
        assert (data["broker"], data["window"], data["fetched"], data["since_mjd"] < data["until_mjd"]) == (
            "alerce", "new", 0, True)
        sent = dict(objects.calls[0].request.url.params.multi_items())
        assert sent["classifier"] == "stamp_classifier" and sent["class"] == "SN" and sent["page_size"] == "21"
        assert api.post("/api/v1/alerts/poll", json={}).status_code == 200


def test_router_shared_service_without_enricher_falls_back(store: AlertStore) -> None:
    p = params("alerce")
    app = FastAPI()
    app.include_router(router)
    with Replay("alerce", "xmatch_alerce"):
        client = offline_client()
        app.state.alert_service = AlertService(store, client, None)
        app.state.service = make_service(client)
        api = TestClient(app)
        res = api.post("/api/v1/alerts/poll", json={"broker": "alerce", "since_mjd": p["since_mjd"],
                                                     "until_mjd": p["until_mjd"], "limit": 3, "crossmatch": True,
                                                     "crossmatch_mode": "wait"})
        assert res.status_code == 200, res.text
        assert res.json()["crossmatched"] == 3 and res.json()["crossmatch_deferred"] is False
        assert all(a["crossmatch_status"] == "done" for a in res.json()["alerts"])
        assert api.post("/api/v1/alerts/alerce:ZTF26abxsxww/crossmatch").status_code == 200


def test_router_recrossmatch_failure_is_502_and_keeps_data(api: TestClient, store: AlertStore) -> None:
    good = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats))
    store.upsert(ALERT)
    asyncio.run(AlertService(store, offline_client(), good).enrich_alert(ALERT))
    before = store.get(ALERT.alert_id)
    down = enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats, failed=dict.fromkeys(cats, "CatalogUnavailableError")))
    api.app.state.alert_service = AlertService(store, offline_client(), down)
    res = api.post(f"/api/v1/alerts/{ALERT.alert_id}/crossmatch")
    assert res.status_code == 502 and "previous enrichment was kept" in res.json()["detail"]
    after = store.get(ALERT.alert_id)
    assert after["enrichment"] == before["enrichment"] and after["crossmatch_status"] == "done"


def test_router_honours_alerts_database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alerts_db = f"sqlite:///{(tmp_path / 'alerts_only.sqlite3').as_posix()}"
    AlertStore(MetadataStore(alerts_db)).upsert(ALERT)
    monkeypatch.setenv("ALERTS_DATABASE_URL", alerts_db)
    app = FastAPI()
    app.include_router(router)
    app.state.metadata = MetadataStore(f"sqlite:///{(tmp_path / 'datasets.sqlite3').as_posix()}")
    listing = TestClient(app).get("/api/v1/alerts").json()
    assert [a["id"] for a in listing["alerts"]] == [ALERT.alert_id]


def test_poll_request_options() -> None:
    assert PollRequest(broker="alerce", class_name="SNIa", classifier="lc_classifier").options() == {
        "class_name": "SNIa", "classifier": "lc_classifier"}
    assert PollRequest(broker="fink_lsst", class_name="in_tns").options() == {"class_name": "in_tns"}
    with pytest.raises(ValueError):
        PollRequest(broker="fink", mjd_field="lastmjd").options()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_cli_poll_list_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = f"sqlite:///{(tmp_path / 'cli.sqlite3').as_posix()}"
    p = params("alerce")
    args = cli(["alerts", "poll", "--broker", "alerce", "--since-mjd", str(p["since_mjd"]), "--until-mjd",
                str(p["until_mjd"]), "--limit", "3", "--class", "SN", "--classifier", "stamp_classifier",
                "--mjd-field", "firstmjd", "--db", db, "--format", "json"])
    with Replay("alerce", "xmatch_alerce"):
        assert args.handler(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["inserted"], out["crossmatched"], out["truncated"]) == (3, 3, True)
    assert {a["id"] for a in out["alerts"]} == {"alerce:ZTF26abxsysn", "alerce:ZTF26abxsxww", "alerce:ZTF26abxsyjc"}

    args = cli(["alerts", "list", "--db", db])
    assert args.handler(args) == 0
    text = capsys.readouterr().out
    assert text.startswith("3 alert(s)") and "alerce:ZTF26abxsxww" in text and "is_new" in text

    args = cli(["alerts", "show", "alerce:ZTF26abxsxww", "--db", db])
    assert args.handler(args) == 0
    assert json.loads(capsys.readouterr().out)["enrichment"]["host"]["name"] == "LEDA 1136493"
    args = cli(["alerts", "show", "nope", "--db", db])
    assert args.handler(args) == 1

    args = cli(["alerts", "poll", "--broker", "fink", "--classifier", "x", "--db", db])
    assert args.handler(args) == 2
    assert cli(["alerts"]).handler(cli(["alerts"])) == 2


def test_cli_poll_reports_broker_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = f"sqlite:///{(tmp_path / 'cli.sqlite3').as_posix()}"
    args = cli(["alerts", "poll", "--broker", "fink_lsst", "--since-mjd", "61300", "--until-mjd", "61301",
                "--no-crossmatch", "--db", db])
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get(url__startswith="https://api.lsst.fink-portal.org/").mock(return_value=httpx.Response(500, text="boom"))
        assert args.handler(args) == 1
    assert "HTTP 500" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["alerts", "poll", "--radius", "0"], ["alerts", "poll", "--host-radius", "-5"], ["alerts", "watch", "--radius", "-1"],
    ["alerts", "poll", "--limit", "0"], ["alerts", "poll", "--limit", "501"], ["alerts", "watch", "--interval", "0"],
    ["alerts", "watch", "--iterations", "0"], ["alerts", "poll", "--radius", "nan"],
])
def test_cli_rejects_invalid_numbers(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli(argv)
    assert exc.value.code == 2 and "Traceback" not in capsys.readouterr().err


def test_cli_watch_runs_iterations(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = FinkZTFBroker.parse_latests(body("fink_ztf", 0), "SN candidate")[0]

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert.from_dict(a.as_dict()) for a in recorded])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    db = f"sqlite:///{(tmp_path / 'watch.sqlite3').as_posix()}"
    args = cli(["alerts", "watch", "--broker", "fink", "--interval", "0.01", "--iterations", "2", "--no-crossmatch",
                "--db", db])
    assert args.handler(args) == 0
    captured = capsys.readouterr()
    assert "Watching fink" in captured.err and "Watching" not in captured.out  # banner on stderr
    assert "new 4" in captured.out and "unchanged 4" in captured.out
    bad = cli(["alerts", "watch", "--broker", "nope", "--iterations", "1", "--db", db])
    assert bad.handler(bad) == 2


def test_cli_watch_json_is_strict_json_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        if broker == "fink":
            raise BrokerError("fink", "HTTP 503", 503, unreachable=True)
        return FetchResult([Alert("alerce", "ZTF26abxsxww", 126.9, -0.7, 61309.5, 18.85, "r", "SN", 0.6, "")])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    db = f"sqlite:///{(tmp_path / 'watch.sqlite3').as_posix()}"
    args = cli(["alerts", "watch", "--broker", "alerce,fink", "--interval", "0.01", "--iterations", "1",
                "--no-crossmatch", "--format", "json", "--db", db])
    assert args.handler(args) == 0
    lines = capsys.readouterr().out.strip().splitlines()

    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    docs = [json.loads(line, parse_constant=reject) for line in lines]
    assert [d["broker"] for d in docs] == ["alerce", "fink"]
    assert docs[0]["alerts"][0]["id"] == "alerce:ZTF26abxsxww"
    assert docs[1]["error"] and docs[1]["since_mjd"] is None


def test_cli_watch_interrupted_exits_130(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    async def cancelled(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise asyncio.CancelledError

    monkeypatch.setattr(AlertService, "watch", cancelled)
    args = cli(["alerts", "watch", "--no-crossmatch", "--db", f"sqlite:///{(tmp_path / 'w.sqlite3').as_posix()}"])
    assert args.handler(args) == 130
    assert "cancelled" in capsys.readouterr().err


@pytest.mark.parametrize("db", ["bogus://x", "sqlite:///Z:/no/such/drive/x.sqlite3"])
@pytest.mark.parametrize("argv", [["alerts", "list"], ["alerts", "show", "foo"], ["alerts", "poll", "--no-crossmatch"],
                                  ["alerts", "watch", "--no-crossmatch", "--iterations", "1"]])
def test_cli_bad_database_exits_2_without_traceback(argv: list[str], db: str, capsys: pytest.CaptureFixture[str]) -> None:
    args = cli([*argv, "--db", db])
    assert args.handler(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "Traceback" not in err


def test_cli_prints_negative_detections_as_marked_magnitudes(capsys: pytest.CaptureFixture[str]) -> None:
    row = {"id": "fink:ZTF1", "mjd": 61300.5, "ra": 10.0, "dec": 20.0, "band": "r", "magpsf": 18.45, "is_negative": True,
           "classification": "SN candidate", "probability": 0.9, "crossmatch_status": "done"}
    alerts._print_row(row)
    out = capsys.readouterr().out
    assert "18.45(neg)" in out and "-18.45" not in out
    alerts._print_row({**row, "is_negative": False})
    assert " 18.45 " in capsys.readouterr().out


async def test_class_lists_are_cached() -> None:
    classes = body("invalid_fink_ztf", 1)
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(200, json=[]))
        route = mock.get("https://api.ztf.fink-portal.org/api/v1/classes").mock(return_value=httpx.Response(200, json=classes))
        async with offline_client() as client:
            first = await FinkZTFBroker().fetch(client, since_mjd=61311.0, until_mjd=61311.2, class_name="SN candidate")
            second = await FinkZTFBroker().fetch(client, since_mjd=61311.2, until_mjd=61311.4, class_name="SN candidate")
            with pytest.raises(ValueError):
                await FinkZTFBroker().fetch(client, since_mjd=61311.2, until_mjd=61311.4, class_name="Nope")
    assert route.call_count == 1 and (first.requests, second.requests) == (2, 1)


async def test_fink_lsst_paging_boundary_is_utc() -> None:
    """Synthetic /tags pages of one diaObject: the MAX_PAGES boundary (a midpointMjdTai row time) is
    reported in UTC like every alert time (TAI - UTC = 37 s)."""
    pages = iter([[61235.40, 61235.39, 61235.38], [61235.37, 61235.36, 61235.35]])

    def tags(request: httpx.Request) -> httpx.Response:
        times = next(pages)[: int(request.url.params["n"])]
        return httpx.Response(200, json=[{"r:diaObjectId": 170657518965489670, "r:ra": 10.0, "r:dec": -30.0,
                                          "r:midpointMjdTai": t, "r:band": "r", "r:psfFlux": 2000.0} for t in times])

    old_max = alerts.MAX_PAGES
    try:
        alerts.MAX_PAGES = 2
        with respx.mock(assert_all_mocked=True) as mock:
            mock.get("https://api.lsst.fink-portal.org/api/v1/tags").mock(side_effect=tags)
            async with offline_client() as client:
                res = await FinkLSSTBroker().fetch(client, since_mjd=61235.0, until_mjd=61235.5, limit=2)
    finally:
        alerts.MAX_PAGES = old_max
    assert len(res.alerts) == 1 and res.truncated
    assert (61235.35 - res.boundary_mjd) * 86400.0 == pytest.approx(37.0, abs=1e-3)


async def test_synthetic_hosts_take_the_cosmicflows4_group_distance_or_report_its_failure() -> None:
    """Synthetic: a PGC host at z = 0.002 without a CF4 distance of its own takes its Tully (2015) group's CF4
    distance; a failed group lookup (a VizieR timeout) leaves the enrichment partial, with that error type."""
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 20 / 3600, "dec": ALERT.dec, "separation_arcsec": 20.0,
              "data": {"otype": "G", "rvz_redshift": 0.002}}
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 60.0)], None, False)

    def answer(ra, dec, cats):
        return record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats else {"simbad": [galaxy]})

    group = {"nest": 100002, "pgc1": 41220, "n_members": 5.0, "sigma_v_kms": 670.0, "r2t_mpc": 1.44, "dm": 31.048,
             "e_dm": 0.008, "v3k_kms": 1479.0}
    res = await enricher_with(answer, d25, groups={1: group}).enrich(ALERT)
    assert res.status == "done" and res.host is not None and res.host["distance_method"] == "cosmicflows4_group"
    assert res.host["distance_mpc"] == pytest.approx(10 ** (31.048 / 5 - 5) / 1.002 ** 2, rel=1e-9)
    assert res.host["group"]["nest"] == 100002 and any("of its group 100002" in e for e in res.evidence)
    down = await enricher_with(answer, d25, groups=alerts.CatalogLookupError("QueryTimeoutError", "VizieR")).enrich(ALERT)
    assert down.status == "partial" and down.host["projected_offset_kpc"] is None
    (failure,) = down.failures
    assert failure["catalog"] == alerts.COSMICFLOWS4_GROUPS and failure["error_type"] == "QueryTimeoutError"
    assert down.outage  # only an unreachable service: retried with a backoff, not counted as an attempt
