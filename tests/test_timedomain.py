"""Offline tests for timedomain.py: recorded-fixture replay, statistics, router and CLI.

Fixtures in ``tests/fixtures/timedomain`` are real upstream responses recorded with
``tests/test_timedomain_live.py record``; replay is strict (the outgoing request's
query/form parameters must equal the recorded ones).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixture_io import FixtureMismatch, request_signature
from test_timedomain_live import (
    BARNARD_GAIA_ID,
    BARNARD_NAME,
    CERES_EPOCH_MJD,
    EA_PERIOD_DAYS,
    EW_PERIOD_DAYS,
    GAPS_QUIET_ID,
    QSO_3C273,
    QUIET_STAR,
    RR_LYR,
    RR_LYR_PERIOD_DAYS,
    RRAB,
    RRAB_PERIOD_DAYS,
    VESTA_EPOCH_MJD,
    case_barnard_name,
    case_ceres,
    case_ea,
    case_ew,
    case_gaps_quiet,
    case_persistence_std,
    case_quiet,
    case_rrab_neowise_wide,
    case_tess_61cyg,
    case_vesta,
)

import timedomain as td

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "timedomain"


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def load_case(name: str) -> list[dict[str, Any]]:
    meta = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for idx, item in enumerate(meta["exchanges"]):
        item = dict(item)
        item["content"] = gzip.decompress((FIXTURES / f"{name}.{idx}.body.gz").read_bytes())
        parts = httpx.URL(item["url"])
        item["url_base"] = f"{parts.scheme}://{parts.host}{parts.path}"
        item["signature"] = request_signature(parts.query, item.get("request_body", ""))
        out.append(item)
    return out


def replay_handler(*names: str):
    exchanges = [e for n in names for e in load_case(n)]

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        sent = request_signature(request.url.query, request.content or b"")
        exact = [e for e in exchanges if e["method"] == request.method and e["url_base"] == base and e["signature"] == sent]
        if not exact:
            raise FixtureMismatch(f"no recording for {request.method} {base} {sent}")
        e = exact[-1]
        headers = {"content-type": e["content_type"]} if e["content_type"] else {}
        if e.get("location"):
            headers["location"] = e["location"]
        return httpx.Response(e["status_code"], headers=headers, content=e["content"])

    return handler


def replay(*names: str) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route().mock(side_effect=replay_handler(*names))
    return router


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=30.0)


def within(value: float, truth: float, rel: float) -> bool:
    return abs(value - truth) / truth < rel


@pytest.fixture(scope="module")
def rrab() -> td.LightCurveResult:
    import asyncio

    async def go() -> td.LightCurveResult:
        with replay("rrab_css_j132708"):
            async with client() as c:
                return await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=c, use_cache=False)

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# Replay: light curves of real objects
# ---------------------------------------------------------------------------


def test_rrab_ztf_period_and_variability(rrab: td.LightCurveResult) -> None:
    assert rrab.failures == []
    keys = {s.key for s in rrab.series}
    assert {"ztf:g", "ztf:r", "neowise:W1", "neowise:W2", "gaia:G", "gaia:BP", "gaia:RP"} <= keys
    best = rrab.period_search.best
    assert best is not None and best.reliable
    assert within(best.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    for key in ("ztf:g", "ztf:r"):
        per = next(p for p in rrab.period_search.per_series if p.series == key)
        assert within(per.best_period_days, RRAB_PERIOD_DAYS, 0.01)
        assert per.false_alarm_probability < 1e-10
        m = rrab.variability[key]
        assert m.is_variable is True and m.n >= 50
        assert m.significance_sigma > 5 and m.intrinsic_scatter > m.noise_rms
    mb = rrab.period_search.multiband
    assert mb is not None and within(mb.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    assert set(mb.members) >= {"ztf:g", "ztf:r"}


def test_rrab_ztf_series_contents(rrab: td.LightCurveResult) -> None:
    g = next(s for s in rrab.series if s.key == "ztf:g")
    assert g.unit == "mag" and g.time_system == "BMJD_TDB"
    assert all(p.flag == 0 for p in g.points)  # flagged epochs dropped by default
    assert g.n_total == len(g.points) + g.n_rejected
    assert g.source_ids and all(s.isdigit() for s in g.source_ids)  # ZTF object IDs
    assert g.metadata["source_separation_arcsec"] < 0.5
    mjds = [p.mjd for p in g.points]
    assert mjds == sorted(mjds) and 58190 < mjds[0] < 59000  # ZTF public survey started 2018-03
    m = rrab.variability["ztf:g"]
    assert 15.0 < m.weighted_mean < 15.8  # Chen+2020 mean g = 15.309


def test_rrab_gaia_and_neowise(rrab: td.LightCurveResult) -> None:
    g = next(s for s in rrab.series if s.key == "gaia:G")
    assert g.source_ids == ["Gaia DR3 1476301553808173824"]
    assert g.n_total >= 100 and g.n_rejected < 10
    per = next(p for p in rrab.period_search.per_series if p.series == "gaia:G")
    assert within(per.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    assert 14.9 < rrab.variability["gaia:G"].weighted_mean < 15.5
    w1 = next(s for s in rrab.series if s.key == "neowise:W1")
    assert len(w1.points) >= 10 and all(p.n and p.n >= 1 for p in w1.points)
    assert sum(p.n for p in w1.points) == w1.metadata["n_exposures_used"]
    assert w1.points[-1].mjd - w1.points[0].mjd > 8 * 365.25
    assert not any(p.series.startswith("neowise") for p in rrab.period_search.per_series)


def test_rrab_as_dict_contract(rrab: td.LightCurveResult) -> None:
    out = json.loads(json.dumps(rrab.as_dict()))
    assert set(out) >= {"target", "series", "variability", "period", "failures", "provenance"}
    s = out["series"][0]
    assert set(s) >= {"survey", "band", "unit", "points"}
    assert set(s["points"][0]) >= {"mjd", "value", "error", "flag"}
    assert set(out["variability"]) == {"per_series", "summary"}
    assert out["variability"]["summary"]["is_variable"] is True
    period = out["period"]
    assert set(period) >= {"best_period_days", "power", "false_alarm_probability", "series"}
    assert period["series"].startswith(("ztf:", "gaia:"))
    ztf_prov = out["provenance"]["ztf"]
    assert "CIRCLE('ICRS', 201.7846706, 38.7450683, 0.00083333)" in ztf_prov["objects_query"]
    assert ztf_prov["objects_table"].startswith("ztf_objects_dr")
    assert ["COLLECTION", ztf_prov["collection"]] in ztf_prov["lightcurve_params"]


def test_ztf_bmjd_matches_irsa_hjd() -> None:
    """Our BJD_TDB from the ZTF UTC 'mjd' (exposure start + exptime/2) vs IRSA's HJD(UTC):
    offset = TT-UTC (69.184 s since 2017) + TDB-TT (ms) + barycentric-heliocentric (< ~4 s)."""
    body = next(e for e in load_case("rrab_css_j132708") if "nph_light_curves" in e["url"])["content"].decode()
    rows = td._csv_rows(body)[:200]
    mjd = np.array([float(r["mjd"]) + float(r["exptime"]) / 2 / 86400 for r in rows])
    hmjd = np.array([float(r["hjd"]) for r in rows]) - td.MJD_JD_OFFSET
    bmjd = td.utc_mjd_to_bmjd_tdb(mjd, *RRAB)
    diff_s = (bmjd - hmjd) * 86400.0
    assert np.all(diff_s > 64.0) and np.all(diff_s < 75.0), (diff_s.min(), diff_s.max())
    assert np.all(np.abs(bmjd - mjd) * 86400 < 520)  # light-travel time across 1 au is 499 s


def test_gaia_time_conversion_tcb_to_tdb() -> None:
    t = np.array([1680.7515868085763])  # first G transit of the RRab star
    bmjd = td.gaia_time_to_bmjd_tdb(t)
    offset_s = (bmjd[0] - (t[0] + 55197.0)) * 86400.0
    # IAU 2006 Resolution B3: TDB = TCB - L_B (JD_TCB - T0) 86400 s + TDB0,
    # L_B = 1.550519768e-8, T0 = 2443144.5003725, TDB0 = -6.55e-5 s.
    jd_tcb = t[0] + 2455197.5
    expected = -1.550519768e-8 * (jd_tcb - 2443144.5003725) * 86400.0 - 6.55e-5
    assert offset_s == pytest.approx(expected, abs=1e-3)


def test_3c273_is_variable_in_ztf() -> None:
    import asyncio

    async def go() -> td.LightCurveResult:
        with replay("qso_3c273"):
            async with client() as c:
                return await td.get_lightcurves(*QSO_3C273, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=c, use_cache=False)

    result = asyncio.run(go())
    assert result.failures == []
    ztf = {k: m for k, m in result.variability.items() if k.startswith("ztf:") and m.n >= 20}
    assert ztf and any(m.is_variable for m in ztf.values())
    w1 = result.variability["neowise:W1"]
    assert 7.5 < w1.weighted_mean < 9.0


def run_case(fixture: str, case) -> Any:
    import asyncio

    async def go() -> Any:
        with replay(fixture):
            async with client() as c:
                return await case(c)

    return asyncio.run(go())


def test_quiet_standard_star_not_variable() -> None:
    """Default surveys (ztf, neowise, gaia): the review found the NEOWISE path never tested on a constant star."""
    result = run_case("quiet_s82_standard", case_quiet)
    assert result.failures == [] and result.target["surveys"] == ["ztf", "neowise", "gaia"]
    for key in ("ztf:g", "ztf:r", "neowise:W1", "neowise:W2"):
        m = result.variability[key]
        assert m.n >= 15 and m.is_variable is False, (key, m.evidence)
    assert 0.6 < result.variability["ztf:g"].chi2_dof < 1.6  # calibrated ZTF error model (was 0.39-0.57)
    assert 15.6 < result.variability["ztf:r"].weighted_mean < 16.1
    assert any("no published epoch photometry" in n for n in result.notes)
    assert result.period_search.best is None
    assert result.as_dict()["period"] is None
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_quiet_result_is_strict_json_and_cli_json_parses_strictly(capsys: pytest.CaptureFixture[str]) -> None:
    """Review: significance_sigma was -inf for chi2 < dof; CLI printed '-Infinity'."""
    result = run_case("quiet_s82_standard", case_quiet)
    body = result.as_dict()
    text = json.dumps(body, allow_nan=False)
    sig = [m["significance_sigma"] for m in body["variability"]["per_series"].values()]
    assert all(v is None or math.isfinite(v) for v in sig) and any(v is not None and v < 0 for v in sig)
    assert "Infinity" not in text
    args = make_parser().parse_args(["lightcurve", "--ra", str(QUIET_STAR[0]), "--dec", str(QUIET_STAR[1]),
                                     "--format", "json"])
    with replay("quiet_s82_standard"):
        assert args.handler(args) == 0
    out = capsys.readouterr().out
    parsed = json.loads(out, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert parsed["variability"]["summary"]["is_variable"] is False


def test_eclipsing_binary_ea_variable_with_orbital_period() -> None:
    """Review: the MAD veto declared this Chen+2020 EA constant and its period was halved."""
    result = run_case("ea_ztfj0140", case_ea)
    assert result.failures == []
    for band in ("ztf:g", "ztf:r"):
        m = result.variability[band]
        assert m.is_variable is True and m.decision_basis == "chi2" and m.amplitude_5_95 > 0.5
    best = result.period_search.best
    assert best is not None and best.series.startswith("ztf:")
    assert within(best.best_period_days, EA_PERIOD_DAYS, 0.01)
    assert best.harmonic_test["doubled"] is True
    assert within(best.alternative_period_days, EA_PERIOD_DAYS / 2, 0.01)
    assert best.period_error_days is not None and best.period_error_days < 1e-3
    body = result.as_dict()
    assert body["period"]["harmonic_test"]["doubled"] is True
    assert body["variability"]["summary"]["is_variable"] is True


def test_eclipsing_binary_ew_orbital_period() -> None:
    result = run_case("ew_ztfj0300", case_ew)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, EW_PERIOD_DAYS, 0.01), best
    assert within(best.lomb_scargle_period_days, EW_PERIOD_DAYS / 2, 0.01)
    assert all(result.variability[k].is_variable for k in ("ztf:g", "ztf:r"))


def test_gaia_gaps_constant_source_not_variable() -> None:
    """Review: pipeline errors without a floor flagged 6/6 constant GAPS sources (this one had chi2/dof 9.9)."""
    result = run_case("gaps_quiet", case_gaps_quiet)
    assert result.provenance["gaia"]["source_id"] == GAPS_QUIET_ID
    g = result.variability["gaia:G"]
    scale, floor = td.gaia_g_error_model(14.02)
    assert (g.error_scale, g.error_floor) == pytest.approx((scale, floor), abs=1e-3)
    assert g.is_variable is False and g.chi2_dof < 2.0, g.evidence
    assert result.variability["gaia:BP"].is_variable is None and result.variability["gaia:RP"].is_variable is None
    series = next(s for s in result.series if s.key == "gaia:G")
    assert series.metadata["phot_variable_flag"] == "NOT_AVAILABLE"
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_neowise_persistence_standard_not_variable() -> None:
    """Review: persistence-contaminated visits made this Ivezic+2007 standard 'strongly variable' in W1."""
    result = run_case("persistence_std", case_persistence_std)
    assert result.failures == []
    w1 = next(s for s in result.series if s.key == "neowise:W1")
    assert w1.metadata["n_visits_rejected"] >= 5
    values = [p.value for p in w1.points]
    assert max(values) - min(values) < 0.1  # the 0.25-mag bright contaminated visits are gone
    for key in ("neowise:W1", "neowise:W2", "ztf:g", "ztf:r"):
        assert result.variability[key].is_variable is False, (key, result.variability[key].evidence)
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_neowise_wide_cone_matches_narrow_cone() -> None:
    """Review: with a 60-arcsec cone, 27 frames adopted a neighbour 3-59 arcsec away."""
    wide = run_case("rrab_neowise_wide", case_rrab_neowise_wide)
    narrow_body = next(e for e in load_case("rrab_css_j132708") if "neowiser_p1bs_psd" in e["request_body"])["content"]
    narrow = td.parse_neowise_csv(narrow_body.decode(), *RRAB)
    for band in ("W1", "W2"):
        a = next(s for s in wide.series if s.band == band)
        b = next(s for s in narrow.series if s.band == band)
        assert a.metadata["n_exposures_used"] == b.metadata["n_exposures_used"]
        assert [round(p.value, 9) for p in a.points] == [round(p.value, 9) for p in b.points]
    assert any("neighbours" in n for n in wide.notes)


def test_tess_two_target_cone_uses_nearest_tic_with_pdcsap() -> None:
    """Review: 61 Cyg A and B (30.7 arcsec apart) were merged into one series."""
    result = run_case("tess_61cyg", case_tess_61cyg)
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 165602000"] and tess.metadata["tic_id"] == "165602000"
    assert sorted(tess.metadata["sectors"]) == [82, 83]
    assert tess.metadata["flux_columns"] == {"82": "PDCSAP_FLUX", "83": "PDCSAP_FLUX"}
    assert set(result.provenance["tess"]["other_tics_in_cone"]) == {"165602023", "1960499855"}
    assert result.provenance["tess"]["other_tics_in_cone"]["165602023"] == pytest.approx(30.7, abs=0.1)
    assert any("other SPOC target" in n for n in result.notes)
    assert abs(np.median([p.value for p in tess.points]) - 1.0) < 0.01
    assert all(p.n and p.n <= 6 for p in tess.points)
    files = result.provenance["tess"]["files"]
    assert len(files) == 2 and all("0000000165602000" in f for f in files)


def test_barnard_by_name_propagates_proper_motion() -> None:
    """Review: the J2000 position found no Gaia DR3 source (Barnard's star moved 166 arcsec by J2016)."""
    result = run_case("barnard_name", case_barnard_name)
    assert result.target["name"] == BARNARD_NAME and result.target["epoch"] == 2000.0
    assert result.target["resolver"]["pm_dec_masyr"] == pytest.approx(10362.394, abs=1)
    gaia_cone = result.target["survey_positions"]["gaia"]
    assert gaia_cone["dec"] == pytest.approx(4.73942, abs=2e-4)  # Gaia DR3 J2016.0 Dec 4.739420
    assert result.provenance["gaia"]["source_id"] == BARNARD_GAIA_ID
    assert result.provenance["gaia"]["separation_arcsec"] < 0.5
    assert any("propagated" in n and "J2016" in n for n in result.notes)


def test_rr_lyr_tess_period() -> None:
    import asyncio

    async def go() -> td.LightCurveResult:
        with replay("rr_lyr_tess"):
            async with client() as c:
                return await td.get_lightcurves(*RR_LYR, radius_arcsec=3.0, surveys="tess", client=c,
                                                tess_max_sectors=1, use_cache=False)

    result = asyncio.run(go())
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.unit == "flux" and tess.source_ids == ["TIC 159717514"]
    assert tess.metadata["sectors"] == [41]
    assert abs(np.median([p.value for p in tess.points]) - 1.0) < 0.1
    counts = [p.n for p in tess.points]
    # 10-min bins of 2-min cadences: 5 per bin except at gaps / bin-edge jitter.
    assert all(1 <= c <= 6 for c in counts) and np.median(counts) == 5
    per = next(p for p in result.period_search.per_series if p.series == "tess:TESS")
    assert within(per.best_period_days, RR_LYR_PERIOD_DAYS, 0.01)
    # Refined (multi-harmonic) period: the grid step of one 26-d sector is 0.42 % of f.
    grid_step = per.best_period_days**2 * (per.max_frequency - per.min_frequency) / (per.n_frequencies - 1)
    assert per.period_error_days is not None and per.period_error_days < 0.1 * grid_step
    assert per.harmonic_test["doubled"] is False
    assert result.variability["tess:TESS"].is_variable is True
    # The TIC lists Tmag 16.6 for RR Lyr (V = 7.2), so PDC over-subtracts "crowding" and
    # PDCSAP goes negative; the SAP fallback keeps the flux physical.
    assert tess.metadata["flux_columns"] == {"41": "SAP_FLUX"}
    assert any("PDCSAP flux has non-positive values" in n for n in result.notes)
    values = np.array([p.value for p in tess.points])
    assert values.min() > 0 and 1.3 < values.max() / values.min() < 2.5  # RRab pulsation in the TESS band


def test_name_resolution_then_gaia() -> None:
    import asyncio

    async def go() -> tuple[float, float, td.LightCurveResult]:
        with replay("name_3c273_gaia"):
            async with client() as c:
                ra, dec, _resolved = await td.resolve_name("3C 273", c)
                return ra, dec, await td.get_lightcurves(ra, dec, radius_arcsec=3.0, surveys="gaia", client=c,
                                                         name="3C 273", use_cache=False)

    ra, dec, result = asyncio.run(go())
    assert abs(ra - QSO_3C273[0]) < 1e-5 and abs(dec - QSO_3C273[1]) < 1e-5
    assert result.provenance["gaia"]["source_id"] == "3700386905605055360"
    assert result.target["name"] == "3C 273"


def test_ztf_upstream_error_is_reported() -> None:
    import asyncio

    async def go() -> None:
        with replay("ztf_bad_collection"):
            async with client() as c:
                await td.fetch_ztf(c, *QUIET_STAR, 3.0, collection="ztf_dr999")

    with pytest.raises(td.UpstreamServiceError) as info:
        asyncio.run(go())
    assert info.value.service == "ztf" and not info.value.retryable
    # IRSA TAP reports the unknown objects table as HTTP 200 + QUERY_STATUS=ERROR.
    assert "unknown table: ztf_objects_dr999" in info.value.message


def test_ztf_neighbour_exclusion() -> None:
    body = next(e for e in load_case("rrab_css_j132708") if "nph_light_curves" in e["url"])["content"].decode()
    rows = td._csv_rows(body)
    fake = dict(rows[0])
    fake.update(oid="999999999999999", ra=f"{RRAB[0] + 2.5 / 3600 / math.cos(math.radians(RRAB[1])):.7f}",
                dec=f"{RRAB[1]:.7f}", mag="18.5")
    lines = [",".join(rows[0].keys())] + [",".join(r.values()) for r in rows] + [",".join(fake.values())]
    result = td.parse_ztf_csv("\n".join(lines), *RRAB)
    assert all("999999999999999" not in s.source_ids for s in result.series)
    assert any("neighbouring" in n for n in result.notes)


def test_empty_ztf_answer() -> None:
    result = td.parse_ztf_csv("oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec\n", *RRAB)
    assert result.series == [] and result.notes


# ---------------------------------------------------------------------------
# Replay: solar system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,case,number,name,epoch", [
    ("vesta_2023", case_vesta, 4, "Vesta", VESTA_EPOCH_MJD),
    ("ceres_2024", case_ceres, 1, "Ceres", CERES_EPOCH_MJD),
])
def test_skybot_matches_horizons(fixture: str, case, number: int, name: str, epoch: float) -> None:
    import asyncio

    async def go():
        with replay(fixture):
            async with client() as c:
                return await case(c)

    eph, sky = asyncio.run(go())
    assert name in eph.target and abs(eph.epoch_mjd - epoch) < 1e-6
    assert sky.epoch_jd == pytest.approx(epoch + 2400000.5)
    obj = next(o for o in sky.objects if o.number == number)
    assert obj.name == name and obj.type == "asteroid" and obj.object_class.startswith("MB")
    assert obj.separation_arcsec < 5.0
    assert abs(obj.v_mag - eph.v_mag) < 0.6
    assert sky.objects == sorted(sky.objects, key=lambda o: o.separation_arcsec)
    assert obj.heliocentric_distance_au and 2.0 < obj.heliocentric_distance_au < 3.1  # main belt


def test_horizons_vesta_known_position() -> None:
    """JPL Horizons, (4) Vesta, 2023-02-25 00:00 UT, geocentric astrometric ICRF (checked live)."""
    import asyncio

    async def go():
        with replay("vesta_2023"):
            async with client() as c:
                return await td.horizons_ephemeris("4;", VESTA_EPOCH_MJD, client=c)

    point = asyncio.run(go())[0]
    assert point.ra == pytest.approx(8.68394, abs=1e-5) and point.dec == pytest.approx(-2.11113, abs=1e-5)
    assert point.v_mag == pytest.approx(8.354, abs=1e-3)


def test_skybot_empty_field() -> None:
    import asyncio

    async def go():
        with replay("skybot_empty"):
            async with client() as c:
                return await td.solar_system_objects(180.0, 80.0, epoch_mjd=VESTA_EPOCH_MJD, radius_arcsec=30.0, client=c)

    result = asyncio.run(go())
    assert result.objects == [] and result.provenance["http_status"] == 204


@pytest.mark.parametrize("kwargs", [
    {"epoch_mjd": 1.0e6}, {"radius_arcsec": 0.0}, {"radius_arcsec": 40000.0}, {"observer": "bad code"},
    {"max_position_error_arcsec": -1.0},
])
async def test_skybot_input_validation(kwargs: dict[str, Any]) -> None:
    base = {"epoch_mjd": VESTA_EPOCH_MJD}
    base.update(kwargs)
    with pytest.raises(td.InvalidCoordinateError):
        await td.solar_system_objects(10.0, 10.0, **base)


def test_horizons_error_payload() -> None:
    import asyncio

    async def go():
        with respx.mock(assert_all_mocked=True) as router:
            router.get(td.HORIZONS_API_URL).mock(return_value=httpx.Response(
                200, json={"error": "No matches found.", "signature": {"version": "1.2"}}))
            async with client() as c:
                await td.horizons_ephemeris("NoSuchBody;", 60000.0, client=c)

    with pytest.raises(td.UpstreamServiceError, match="No matches found"):
        asyncio.run(go())


# ---------------------------------------------------------------------------
# Failure handling & cache
# ---------------------------------------------------------------------------


async def test_partial_failure_keeps_other_surveys() -> None:
    handler = replay_handler("rrab_css_j132708")

    def side_effect(request: httpx.Request) -> httpx.Response:
        if "nph_light_curves" in request.url.path:
            return httpx.Response(503, text="Service Unavailable")
        return handler(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,gaia", client=c, use_cache=False)
    assert [f["survey"] for f in result.failures] == ["ztf"]
    assert result.failures[0]["retryable"] is True and result.failures[0]["status_code"] == 503
    assert {s.survey for s in result.series} == {"gaia"}
    assert within(result.period_search.best.best_period_days, RRAB_PERIOD_DAYS, 0.01)


async def test_all_surveys_failing_raises() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=httpx.ConnectTimeout("timed out"))
        async with client() as c:
            with pytest.raises(td.UpstreamServiceError) as info:
                await td.get_lightcurves(*RRAB, surveys="ztf,neowise", client=c, use_cache=False)
    assert info.value.retryable and "all requested surveys failed" in str(info.value)


async def test_tap_error_inside_http_200() -> None:
    votable = (b'<?xml version="1.0"?><VOTABLE><RESOURCE type="results"><INFO name="QUERY_STATUS" value="ERROR">'
               b"Table neowiser_p1bs_psd not found</INFO></RESOURCE></VOTABLE>")
    with respx.mock(assert_all_mocked=True) as router:
        router.post(td.IRSA_TAP_SYNC_URL).mock(return_value=httpx.Response(200, content=votable,
                                                                           headers={"content-type": "text/xml"}))
        async with client() as c:
            with pytest.raises(td.UpstreamServiceError, match="not found"):
                await td.fetch_neowise(c, *RRAB, 3.0)


async def test_cache_serves_second_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "60")
    monkeypatch.setattr(td, "_cache", None)
    with replay("name_3c273_gaia"):
        async with client() as c:
            ra, dec, _ = await td.resolve_name("3C 273", c)
            first = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=httpx.ConnectError("offline"))
        async with client() as c:
            second = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
    assert not route.called
    assert second.provenance["cache"] == "hit"
    assert [s.as_dict() for s in second.series] == [s.as_dict() for s in first.series]


async def test_input_validation() -> None:
    with pytest.raises(td.InvalidCoordinateError):
        await td.get_lightcurves(10.0, 95.0)
    with pytest.raises(td.InvalidCoordinateError):
        await td.get_lightcurves(10.0, 10.0, radius_arcsec=0.0)
    with pytest.raises(ValueError):
        await td.get_lightcurves(10.0, 10.0, surveys="ztf,hubble")
    with pytest.raises(ValueError):
        await td.get_lightcurves(10.0, 10.0, min_period_days=1.0, max_period_days=0.5)
    with pytest.raises(ValueError):
        await td.get_lightcurves(10.0, 10.0, ztf_collection="dr23; DROP TABLE")


async def test_malformed_upstream_answer_is_a_survey_failure() -> None:
    handler = replay_handler("rrab_css_j132708")

    def side_effect(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gea.esac.esa.int" and request.url.path.endswith("/data"):
            return httpx.Response(200, text="<html>maintenance page</html>", headers={"content-type": "text/html"})
        return handler(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,gaia", client=c, use_cache=False)
    gaia = [f for f in result.failures if f["survey"] == "gaia"]
    assert gaia and "DataLink returned no CSV" in gaia[0]["error"]
    assert {s.survey for s in result.series} == {"ztf"}


def test_parse_surveys() -> None:
    assert td.parse_surveys(None) == td.DEFAULT_SURVEYS
    assert td.parse_surveys("ZTF, gaia,ztf") == ("ztf", "gaia")
    with pytest.raises(ValueError):
        td.parse_surveys("ztf,foo")
    with pytest.raises(ValueError):
        td.parse_surveys(" , ")


# ---------------------------------------------------------------------------
# Statistics on synthetic data with known answers
# ---------------------------------------------------------------------------


def _ztf_like_times(rng: np.random.Generator, n: int, years: float = 5.0) -> np.ndarray:
    """Nightly-ish irregular sampling: one epoch per night (random hour), seasonal gaps."""
    nights = np.sort(rng.choice(int(years * 365), size=n, replace=False)).astype(float)
    season = (nights % 365) < 250
    t = nights[season] + rng.uniform(0.15, 0.45, size=int(season.sum()))
    return 58200.0 + t


def test_constant_star_statistics() -> None:
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1000, 800))
    dy = np.full_like(t, 0.02)
    y = 16.0 + rng.normal(0, 0.02, t.size)
    m = td.variability_metrics(t, y, dy)
    assert m.is_variable is False
    assert 0.85 < m.chi2_dof < 1.15
    assert 1.8 < m.von_neumann_eta < 2.2  # E[eta] = 2 for independent noise (von Neumann 1941)
    assert abs(m.excess_variance) < 3 * m.excess_variance_error
    assert m.amplitude_5_95 == pytest.approx(2 * 1.6449 * 0.02, rel=0.15)
    assert abs(m.weighted_mean - 16.0) < 0.005


def test_sinusoid_statistics() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 1000, 600))
    amp = 0.3
    y = 15.0 + amp * np.sin(2 * np.pi * t / 0.63) + rng.normal(0, 0.02, t.size)
    m = td.variability_metrics(t, y, np.full_like(t, 0.02))
    assert m.is_variable is True and m.significance_sigma > 30
    assert m.amplitude_5_95 == pytest.approx(2 * amp * math.sin(0.45 * math.pi), rel=0.08)
    assert m.intrinsic_scatter > 5 * m.noise_rms


def test_excess_variance_of_flux_sinusoid() -> None:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 500, 2000))
    y = 1.0 + 0.1 * np.sin(2 * np.pi * t / 3.7) + rng.normal(0, 0.01, t.size)
    m = td.variability_metrics(t, y, np.full_like(t, 0.01), unit="flux")
    # F_var of a sinusoid of amplitude A about mean 1 is A / sqrt(2) (Vaughan et al. 2003).
    assert m.fractional_variability == pytest.approx(0.1 / math.sqrt(2), rel=0.05)


def test_red_noise_has_small_eta_and_large_stetson_j() -> None:
    rng = np.random.default_rng(4)
    t = np.repeat(np.arange(300.0), 2) + np.tile([0.0, 0.02], 300)  # nightly pairs ~30 min apart
    walk = np.cumsum(rng.normal(0, 0.03, 300))
    y = 17.0 + np.repeat(walk, 2) + rng.normal(0, 0.01, t.size)
    m = td.variability_metrics(t, y, np.full_like(t, 0.01))
    assert m.von_neumann_eta < 0.5
    assert m.stetson_n_pairs == 300 and m.stetson_j > 5
    assert m.is_variable is True


def test_outliers_do_not_make_a_star_variable() -> None:
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 1000, 1500))
    y = 16.0 + rng.normal(0, 0.02, t.size)
    y[[100, 700, 1200]] += [1.5, -2.0, 1.0]
    raw_chi2 = float(np.sum(((y - np.mean(y)) / 0.02) ** 2))
    assert td.stats.chi2.sf(raw_chi2, t.size - 1) < td.FIVE_SIGMA_PVALUE  # chi^2 of all epochs is fooled
    m = td.variability_metrics(t, y, np.full_like(t, 0.02))
    assert m.n_outliers_excluded == 3  # the 3 isolated epochs are set aside ...
    assert m.is_variable is False and m.chi2_dof < 1.2  # ... and the star is constant


def test_stetson_j_without_pairs_is_none() -> None:
    t = np.arange(20.0) * 5
    j, pairs = td.stetson_j(t, np.ones(20), np.ones(20))
    assert j is None and pairs == 0


def test_too_few_points_is_undetermined() -> None:
    m = td.variability_metrics([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [0.01, 0.01, 0.01])
    assert m.n == 3 and m.is_variable is None


def test_lomb_scargle_recovers_period_from_sparse_sampling() -> None:
    rng = np.random.default_rng(6)
    t = _ztf_like_times(rng, 700)
    period = 0.6297706
    y = 15.3 + 0.35 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.03, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.03), series="synthetic")
    assert res is not None and res.reliable
    assert within(res.best_period_days, period, 1e-3)
    assert res.false_alarm_probability < 1e-20
    grid_step = (res.max_frequency - res.min_frequency) / (res.n_frequencies - 1)
    assert grid_step == pytest.approx(1 / (td.DEFAULT_OVERSAMPLING * res.baseline_days), rel=1e-3)


def test_lomb_scargle_noise_is_not_significant() -> None:
    rng = np.random.default_rng(7)
    t = _ztf_like_times(rng, 300)
    res = td.lomb_scargle_period(t, rng.normal(0, 0.02, t.size), np.full_like(t, 0.02))
    assert res is not None and res.false_alarm_probability > td.PERIOD_FAP_THRESHOLD and not res.reliable


def test_multiband_combines_bands() -> None:
    rng = np.random.default_rng(8)
    period = 0.4
    series = []
    for band, offset, amp in (("g", 15.5, 0.25), ("r", 15.2, 0.18)):
        t = _ztf_like_times(rng, 60)  # too sparse to be robust alone, fine combined
        y = offset + amp * np.sin(2 * np.pi * t / period + 0.3) + rng.normal(0, 0.03, t.size)
        pts = [td.LightCurvePoint(float(a), float(b), 0.03) for a, b in zip(t, y)]
        series.append(td.LightCurveSeries("ztf", band, "mag", pts, "AB"))
    mb = td.multiband_period(series)
    assert mb is not None and within(mb.best_period_days, period, 1e-3)
    assert mb.members == ["ztf:g", "ztf:r"] and mb.false_alarm_probability is None


def test_frequency_grid_cap() -> None:
    freq, os_used = td.frequency_grid(3000.0, min_period_days=0.01, oversampling=10, max_frequencies=1_000_000)
    assert len(freq) == 1_000_000 and td.MIN_OVERSAMPLING <= os_used < 10
    with pytest.raises(ValueError, match="samples per peak"):  # would drop below 2 samples per peak
        td.frequency_grid(3000.0, min_period_days=0.01, oversampling=10, max_frequencies=100_000)
    with pytest.raises(ValueError):
        td.frequency_grid(10.0, min_period_days=20.0)


def test_neowise_single_exposures_reveal_rrab_variability() -> None:
    """Visit means average over the RRab's 0.63 d cycle; single exposures keep it."""
    body = next(e for e in load_case("rrab_css_j132708") if "neowiser_p1bs_psd" in e["request_body"])["content"].decode()
    binned = td.parse_neowise_csv(body, *RRAB)
    single = td.parse_neowise_csv(body, *RRAB, binned=False)
    w1_bin = next(s for s in binned.series if s.band == "W1")
    w1_exp = next(s for s in single.series if s.band == "W1")
    assert len(w1_exp.points) == w1_bin.metadata["n_exposures_used"] == sum(p.n for p in w1_bin.points)
    assert w1_exp.metadata["n_visits"] == len(w1_bin.points)
    assert all(p.n >= td.NEOWISE_MIN_VISIT_EXPOSURES for p in w1_bin.points)
    m_bin, m_exp = td.series_variability(w1_bin), td.series_variability(w1_exp)
    assert m_bin.is_variable is False
    assert m_exp.is_variable is True and m_exp.amplitude_5_95 > 3 * m_bin.amplitude_5_95


def test_bin_visits() -> None:
    t = [100.0, 100.1, 100.2, 280.0, 280.05]
    y = [10.0, 10.2, 10.1, 11.0, 11.0]
    dy = [0.01, 0.01, 0.01, 0.05, 0.05]
    bins = td.bin_visits(t, y, dy)
    assert [b.n for b in bins] == [3, 2]
    assert bins[0].value == pytest.approx(10.1) and bins[0].mjd == pytest.approx(100.1)
    # Intra-visit scatter (sample std 0.1 -> SEM 0.0577) exceeds the formal error (0.01/sqrt(3)) and wins.
    assert bins[0].error == pytest.approx(0.1 / math.sqrt(3))
    assert bins[0].error > 5 * 0.01 / math.sqrt(3)
    # Identical values: SEM = 0, so the formal weighted-mean error is kept.
    assert bins[1].error == pytest.approx(0.05 / math.sqrt(2))


def test_bin_uniform() -> None:
    t = np.arange(0, 1, 2 / 1440)  # 2-min cadence for a day
    pts = td.bin_uniform(t, np.ones_like(t), np.full_like(t, 0.01), 10 / 1440)
    assert len(pts) == 144 and all(p.n == 5 for p in pts)
    assert pts[0].error == pytest.approx(0.01 / math.sqrt(5))


def test_sigma_from_tiny_pvalues_is_monotonic() -> None:
    values = [td._sigma_from_log_sf(lp) for lp in (-10.0, -100.0, -699.0, -701.0, -5000.0)]
    assert math.isfinite(td.chi2_significance(1e7, 500)) and td.chi2_significance(1e7, 500) > values[-1]
    assert values == sorted(values)
    assert values[0] == pytest.approx(float(td.stats.norm.isf(math.exp(-10.0))), rel=1e-9)
    assert values[3] == pytest.approx(values[2], rel=0.01)  # asymptotic branch joins smoothly


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(td.router)
    return app


def test_router_lightcurves() -> None:
    with replay("rrab_css_j132708"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"ra": RRAB[0], "dec": RRAB[1], "radius_arcsec": 3,
                                                            "surveys": "ztf,neowise,gaia"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert within(body["period"]["best_period_days"], RRAB_PERIOD_DAYS, 0.01)
    assert body["variability"]["per_series"]["ztf:r"]["is_variable"] is True
    point = body["series"][0]["points"][0]
    assert set(point) >= {"mjd", "value", "error", "flag"}


def test_router_lightcurves_by_name() -> None:
    with replay("name_3c273_gaia"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"name": "3C 273", "surveys": "gaia"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"]["name"] == "3C 273"
    assert {s["key"] for s in body["series"]} >= {"gaia:G"}


@pytest.mark.parametrize("params", [
    {"ra": 10.0},  # no dec
    {},
    {"ra": 10.0, "dec": 95.0},
    {"ra": 10.0, "dec": 10.0, "radius_arcsec": 0},
    {"ra": 10.0, "dec": 10.0, "radius_arcsec": 61},
    {"ra": 10.0, "dec": 10.0, "surveys": "ztf,kepler"},
    {"ra": 10.0, "dec": 10.0, "min_period_days": 1.0, "max_period_days": 0.5},
    {"ra": "abc", "dec": 10.0},
])
def test_router_lightcurves_422(params: dict[str, Any]) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params=params)
        assert not route.called
    assert response.status_code == 422, response.text


def test_router_lightcurves_502() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(500, text="Internal Server Error"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"ra": 10.0, "dec": 10.0, "surveys": "ztf,neowise"})
    assert response.status_code == 502
    assert "all requested surveys failed" in response.json()["detail"]


def test_router_uses_shared_client() -> None:
    app = make_app()
    shared = httpx.AsyncClient(follow_redirects=True)
    app.state.client = shared
    with replay("vesta_2023"), TestClient(app) as http:
        eph = next(e for e in load_case("vesta_2023") if "horizons" in e["url"])
        result = json.loads(eph["content"])["result"]
        point = td.parse_horizons_result(result)[0]
        response = http.get("/api/v1/solar-system", params={"ra": f"{point.ra:.7f}", "dec": f"{point.dec:.7f}",
                                                             "epoch_mjd": VESTA_EPOCH_MJD, "radius_arcsec": 600})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["objects"][0]["name"] == "Vesta" and body["objects"][0]["number"] == 4
    assert body["objects"][0]["separation_arcsec"] < 5
    assert set(body["objects"][0]) >= {"name", "type", "ra", "dec", "separation_arcsec", "v_mag"}


@pytest.mark.parametrize("params", [
    {"ra": 10.0, "dec": 10.0, "epoch_mjd": 1e6},
    {"ra": 10.0, "dec": 10.0, "observer": "toolong"},
    {"ra": 10.0, "dec": 100.0},
    {"ra": 10.0},
    {"ra": 10.0, "dec": 10.0, "radius_arcsec": 50000},
])
def test_router_solar_system_422(params: dict[str, Any]) -> None:
    with TestClient(make_app()) as http:
        response = http.get("/api/v1/solar-system", params=params)
    assert response.status_code == 422, response.text


def test_router_solar_system_502() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.get(td.SKYBOT_CONESEARCH_URL).mock(return_value=httpx.Response(503, text="maintenance"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/solar-system", params={"ra": 10.0, "dec": 10.0, "epoch_mjd": 60000})
    assert response.status_code == 502 and "skybot" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    td.register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_cli_lightcurve_summary(capsys: pytest.CaptureFixture[str]) -> None:
    args = make_parser().parse_args(["lightcurve", "--ra", str(RRAB[0]), "--dec", str(RRAB[1]),
                                     "--surveys", "ztf,neowise,gaia"])
    with replay("rrab_css_j132708"):
        code = args.handler(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "[ztf:r" in out and "VARIABLE" in out
    period_line = next(line for line in out.splitlines() if line.startswith("Period:"))
    assert within(float(period_line.split()[1]), RRAB_PERIOD_DAYS, 0.01)


def test_cli_solar_system_json(capsys: pytest.CaptureFixture[str]) -> None:
    args = make_parser().parse_args(["solar-system", "--ra", "180", "--dec", "80", "--epoch-mjd", str(VESTA_EPOCH_MJD),
                                     "--radius", "30", "--format", "json"])
    with replay("skybot_empty"):
        code = args.handler(args)
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["objects"] == [] and body["epoch_mjd"] == VESTA_EPOCH_MJD


def test_cli_errors(capsys: pytest.CaptureFixture[str]) -> None:
    parser = make_parser()
    args = parser.parse_args(["lightcurve", "--ra", "10"])
    assert args.handler(args) == 2
    args = parser.parse_args(["solar-system", "--ra", "10", "--dec", "10", "--epoch-mjd", "1e7"])
    assert args.handler(args) == 2
    assert "Error" in capsys.readouterr().err
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(502, text="bad gateway"))
        args = parser.parse_args(["lightcurve", "--ra", "10", "--dec", "10", "--surveys", "gaia"])
        assert args.handler(args) == 1


def test_chunked_power_equals_single_grid() -> None:
    from astropy.timeseries import LombScargle

    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0, 300, 400))
    y = np.sin(2 * np.pi * t / 1.7) + rng.normal(0, 0.3, t.size)
    ls = LombScargle(t, y, np.full_like(t, 0.3))
    freq, _ = td.frequency_grid(300.0)
    assert np.allclose(td.ls_power(ls, freq, chunk=1000), td.ls_power(ls, freq, chunk=10**7), atol=1e-8)


def test_diurnal_alias_detection() -> None:
    assert td.diurnal_alias(1.0 / 0.99745, 2660.0) == "sidereal day"  # the quiet star's ZTF g peak
    assert td.diurnal_alias(2.0, 2660.0) == "2 x solar day"
    assert td.diurnal_alias(1.0 / RRAB_PERIOD_DAYS, 2660.0) is None
    rng = np.random.default_rng(10)
    t = _ztf_like_times(rng, 400)
    y = 0.05 * np.cos(2 * np.pi * td.SIDEREAL_DAY_FREQUENCY * t) + rng.normal(0, 0.01, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.01), ground_based=True)
    assert res.false_alarm_probability < td.PERIOD_FAP_THRESHOLD and not res.reliable
    assert any("sidereal day" in w for w in res.warnings)


# Shape of a real SkyBoT worker crash answer observed live (HTTP 200, backtrace truncated here).
SKYBOT_CRASH = {"flag": -1, "ticket": 187756412085009412, "status": 200,
                "message": "SkyBoT asteroid conesearch -> numIntAstromJ2000_ failed\n#21  0x4051b0 in _start\n"}


async def test_skybot_worker_crash_is_retried() -> None:
    real = replay_handler("skybot_empty")
    calls = {"n": 0}

    def side_effect(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=SKYBOT_CRASH)
        return real(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            result = await td.solar_system_objects(180.0, 80.0, epoch_mjd=VESTA_EPOCH_MJD, radius_arcsec=30.0, client=c)
    assert calls["n"] == 2 and result.objects == []


def test_skybot_persistent_crash_is_502() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.get(td.SKYBOT_CONESEARCH_URL).mock(return_value=httpx.Response(200, json=SKYBOT_CRASH))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/solar-system", params={"ra": 10.0, "dec": 10.0, "epoch_mjd": 60000})
    assert response.status_code == 502
    assert "service error (flag -1)" in response.json()["detail"]
