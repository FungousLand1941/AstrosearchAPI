"""Offline regression tests for the round-3 adversarial review of timedomain.py.

One section per review finding (numbered as in the review). Synthetic data where the
defect is a property of the algorithm; recorded upstream answers (``tests/fixtures/
timedomain``, see ``tests/test_timedomain_live.py record``) where it needs real data.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import warnings
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from astropy.utils import iers
from fastapi.testclient import TestClient
from test_timedomain import client, load_case, make_app, make_parser, replay, replay_handler, run_case, within
from test_timedomain_live import (
    ALGOL_PERIOD_DAYS,
    CM_DRA_PERIOD_DAYS,
    EA2_PERIOD_DAYS,
    QSO_3C273,
    RRAB,
    V350_LYR_PERIOD_DAYS,
    V445_LYR_PERIOD_DAYS,
    case_cm_dra_ztf,
    case_ea2,
    case_name_3c273_gaia,
    case_oj287_name,
    case_qso_j0747,
    case_rr_lyr_ztf10,
    case_tau_cet_tess,
    case_v350_lyr_gaia,
    case_v445_lyr_ztf,
)

import timedomain as td

HTML = "<!DOCTYPE html><html><head><title>Maintenance</title></head><body><h1>Service down for maintenance</h1></body></html>"


def mag_series(survey: str, band: str, t: np.ndarray, y: np.ndarray, err: float | np.ndarray, *,
               unit: str = "mag", **meta: Any) -> td.LightCurveSeries:
    errs = np.broadcast_to(np.asarray(err, dtype=float), np.shape(t))
    pts = [td.LightCurvePoint(float(a), float(b), float(e)) for a, b, e in zip(t, y, errs)]
    return td.LightCurveSeries(survey, band, unit, pts, "test", metadata=dict(meta))


def ztf_like_times(rng: np.random.Generator, n: int, years: float = 6.0) -> np.ndarray:
    nights = np.sort(rng.choice(int(years * 365), size=n, replace=False)).astype(float)
    season = (nights % 365) < 250
    return 58200.0 + nights[season] + rng.uniform(0.15, 0.45, size=int(season.sum()))


def gaia_like_times(rng: np.random.Generator, n_visits: int = 25, baseline: float = 1030.0) -> np.ndarray:
    """Gaia scanning: visits of 1-3 transits 106.5 min / 6 h apart, visits clustered in time."""
    starts = np.sort(rng.uniform(56870.0, 56870.0 + baseline, n_visits))
    t = []
    for s in starts:
        for k in range(int(rng.integers(1, 4))):
            t.append(s + k * 0.25 + (106.5 / 1440.0 if rng.random() < 0.5 else 0.0))
    return np.sort(np.array(t))


def eclipsing_binary(t: np.ndarray, period: float, *, depth1: float, depth2: float, width: float,
                     base: float = 15.0) -> np.ndarray:
    phase = (t / period) % 1.0
    d1 = np.minimum(phase, 1.0 - phase)
    d2 = np.abs(phase - 0.5)
    return base + depth1 * np.exp(-0.5 * (d1 / width) ** 2) + depth2 * np.exp(-0.5 * (d2 / width) ** 2)


def rrab_mags(t: np.ndarray, period: float, base: float = 15.0) -> np.ndarray:
    phase = 2 * np.pi * t / period
    return base + 0.35 * np.sin(phase) + 0.12 * np.sin(2 * phase + 1.0) + 0.05 * np.sin(3 * phase + 2.0)


def damped_random_walk(rng: np.random.Generator, t: np.ndarray, *, tau: float, sigma: float) -> np.ndarray:
    """Ornstein-Uhlenbeck (DRW) light curve, the standard quasar variability model (Kelly et al. 2009)."""
    out = np.empty_like(t)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, len(t)):
        a = math.exp(-(t[i] - t[i - 1]) / tau)
        out[i] = a * out[i - 1] + rng.normal(0.0, sigma * math.sqrt(1.0 - a * a))
    return out


# ---------------------------------------------------------------------------
# 1. Identity: the nearest object in the cone is not the target (ZTF, TESS, Gaia)
# ---------------------------------------------------------------------------


def test_rr_lyr_saturated_in_ztf_gets_no_neighbour_light_curve() -> None:
    """Review: a 10" cone around RR Lyr returned the 17.9-mag neighbour 7.9" away with a 0.361-d 'period'."""
    result = run_case("rr_lyr_ztf10", case_rr_lyr_ztf10)
    assert result.failures == []
    assert [s for s in result.series if s.survey == "ztf"] == []
    assert result.period_search is None or result.period_search.best is None
    prov = result.provenance["ztf"]
    assert prov["match_radius_arcsec"] == td.ZTF_MATCH_ARCSEC
    assert prov["neighbour_object_ids"]["728116300014796"] == pytest.approx(7.9, abs=0.1)
    assert any("no object in" in n and "no ZTF light curve for the target" in n for n in result.notes)
    assert any("728116300014796 (g, 7.9\"" in n for n in result.notes)


def test_tess_lone_spoc_target_40_arcsec_away_is_not_the_target() -> None:
    obs = [{"obs_id": f"tess2024249191853-s0083-{159717514:016d}-0280-s", "obsid": 1, "s_ra": 10.0,
            "s_dec": 10.0 + 40.0 / 3600}]
    tic, chosen, others = td.select_tess_observations(obs, 10.0, 10.0, max_sectors=2)
    assert tic is None and chosen == []
    assert others == {"159717514": pytest.approx(40.0, abs=0.01)}


async def test_fetch_tess_reports_distant_spoc_target_without_downloading() -> None:
    obs = {"obs_id": f"tess2024249191853-s0083-{159717514:016d}-0280-s", "obsid": 1, "s_ra": 10.0,
           "s_dec": 10.0 + 40.0 / 3600}
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        query = router.post(td.MAST_INVOKE_URL).mock(
            return_value=httpx.Response(200, json={"status": "COMPLETE", "data": [obs]}))
        download = router.get(td.MAST_DOWNLOAD_URL).mock(side_effect=AssertionError("no download of another star"))
        async with client() as c:
            result = await td.fetch_tess(c, 10.0, 10.0, 60.0)
    assert result.series == [] and query.call_count == 1 and not download.called
    assert any("no SPOC 2-min target within 2\"" in n for n in result.notes)
    assert any("TIC 159717514 (40\")" in n for n in result.notes)
    assert result.provenance["tic_id"] is None and result.provenance["match_radius_arcsec"] == td.TESS_MATCH_ARCSEC


async def test_gaia_nearest_source_outside_match_tolerance_is_not_adopted() -> None:
    cone = ("source_id,ra,dec,phot_g_mean_mag,has_epoch_photometry,phot_variable_flag,dist\n"
            f"111,10.0,{10.0 + 5.0 / 3600:.7f},17.9,true,VARIABLE,{5.0 / 3600:.8f}\n"
            f"222,10.0,{10.0 - 9.0 / 3600:.7f},18.4,false,NOT_AVAILABLE,{9.0 / 3600:.8f}\n")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.post(td.GAIA_TAP_SYNC_URL).mock(return_value=httpx.Response(200, text=cone))
        datalink = router.get(td.GAIA_DATALINK_URL).mock(side_effect=AssertionError("no DataLink for a neighbour"))
        async with client() as c:
            result = await td.fetch_gaia(c, 10.0, 10.0, 60.0)
    assert result.series == [] and not datalink.called
    assert "no source within 1\"" in result.notes[0] and "111 (5.0\")" in result.notes[0]
    assert result.provenance["neighbour_source_ids"] == {"111": pytest.approx(5.0, abs=0.01),
                                                         "222": pytest.approx(9.0, abs=0.01)}


# ---------------------------------------------------------------------------
# 2. Outlier removal must never drop short series or clustered (real) events
# ---------------------------------------------------------------------------


def test_five_epochs_with_three_deviant_keep_all_epochs() -> None:
    """Review (a): N = 5 with 3 wild points: 3 were excluded (60 % of the data) and dof = 1."""
    m = td.variability_metrics(np.arange(5.0), [15.0, 15.0, 18.0, 19.0, 17.0], np.full(5, 0.01))
    assert m.n_outliers_excluded == 0 and m.dof == 4 and m.is_variable is True


def test_neowise_dimming_event_of_three_consecutive_visits_is_variable() -> None:
    """Review (b): 20 visits, 3 consecutive faded by 1.5 mag: chi2/dof 0.76, not variable."""
    rng = np.random.default_rng(31)
    t = 56700.0 + np.arange(20) * 182.6
    y = 13.0 + rng.normal(0, 0.01, t.size)
    y[8:11] += 1.5
    m = td.series_variability(mag_series("neowise", "W1", t, y, 0.01))
    assert m.n_outliers_excluded == 0 and m.is_variable is True and m.chi2_dof > 100


def test_gaia_detached_eclipsing_binary_with_three_in_eclipse_transits_is_variable() -> None:
    """Review (c): 40 G transits, 3 in eclipse (0.6-0.8 mag): chi2/dof 0.61, not variable."""
    rng = np.random.default_rng(32)
    t = np.sort(rng.uniform(56900, 57900, 40))
    y = 15.0 + rng.normal(0, 0.003, t.size)
    y[[5, 17, 30]] += [0.6, 0.7, 0.8]
    m = td.series_variability(mag_series("gaia", "G", t, y, 0.002, phot_g_mean_mag=15.0))
    assert m.n_outliers_excluded == 0 and m.is_variable is True


def test_ztf_eclipses_seen_in_both_filters_of_a_night_are_not_outliers() -> None:
    """Review (d): P = 27.3 d detached EB, 0.6-mag primary eclipse in 5-13 of ~1000 epochs (<= 1 %): flagged
    variable in 3 of 20 realisations. ZTF observes a field in g and r in the same night (Bellm et al. 2019);
    an eclipse lasting hours dims both, so these epochs are not isolated artefacts."""
    flagged = 0
    for k in range(20):
        rng = np.random.default_rng(100 + k)
        tg = ztf_like_times(rng, 1500)
        tr = tg + rng.uniform(0.02, 0.1, tg.size)
        yg = eclipsing_binary(tg, 27.3, depth1=0.6, depth2=0.0, width=0.0025, base=16.0) + rng.normal(0, 0.015, tg.size)
        yr = eclipsing_binary(tr, 27.3, depth1=0.6, depth2=0.0, width=0.0025, base=16.0) + rng.normal(0, 0.015, tr.size)
        g, r = mag_series("ztf", "g", tg, yg, 0.013), mag_series("ztf", "r", tr, yr, 0.013)
        m = td.series_variability(g, companions=[r])
        in_eclipse = int(np.sum(yg - 16.0 > 0.1))
        # only eclipse-edge epochs whose r visit fell outside the eclipse can still look isolated
        assert m.n_outliers_excluded < in_eclipse / 2, (k, m.n_outliers_excluded, in_eclipse)
        flagged += bool(m.is_variable)
    assert flagged == 20


def test_companions_of_another_survey_do_not_protect_artefacts() -> None:
    rng = np.random.default_rng(33)
    t = np.sort(rng.uniform(58300, 60300, 1200))
    y = 16.0 + rng.normal(0, 0.015, t.size)
    y[[100, 600, 1100]] += 1.0  # isolated single-exposure artefacts
    other = mag_series("gaia", "G", t[[100, 600, 1100]] + 0.01, np.full(3, 17.0), 0.01, phot_g_mean_mag=16.0)
    m = td.series_variability(mag_series("ztf", "g", t, y, 0.013), companions=[other])
    assert m.n_outliers_excluded == 3 and m.is_variable is False


# ---------------------------------------------------------------------------
# 3. No process-global state is touched (warnings filters, astropy IERS configuration)
# ---------------------------------------------------------------------------


async def test_concurrent_lightcurves_leave_warnings_and_iers_configuration_untouched() -> None:
    filters_before = list(warnings.filters)
    auto_before = iers.conf.auto_download
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_handler("rrab_css_j132708", "qso_3c273"))
        async with client() as c:
            a, b = await asyncio.gather(
                td.get_lightcurves(*RRAB, surveys="ztf,neowise,gaia", client=c, period=False, use_cache=False),
                td.get_lightcurves(*QSO_3C273, surveys="ztf,neowise,gaia", client=c, period=False, use_cache=False))
    assert a.failures == [] and b.failures == [] and a.series and b.series
    assert list(warnings.filters) == filters_before
    assert iers.conf.auto_download == auto_before
    with pytest.warns(UserWarning, match="still visible"):
        warnings.warn("still visible", UserWarning, stacklevel=1)


def test_time_conversions_in_many_threads_do_not_change_global_state() -> None:
    filters_before = list(warnings.filters)
    auto_before = iers.conf.auto_download
    mjd = np.linspace(58000.0, 64000.0, 2000)  # up to 2033: beyond the leap-second table, no warning either
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(20):
                td.utc_mjd_to_bmjd_tdb(mjd, 201.78, 38.74)
                td.gaia_time_to_bmjd_tdb(mjd - 55197.0)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning emitted by a conversion would raise in the worker
        threads = [threading.Thread(target=work) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert errors == []
    assert list(warnings.filters) == filters_before and iers.conf.auto_download == auto_before


def test_bmjd_conversion_matches_astropy_barycentric_correction() -> None:
    """The ERFA-based conversion reproduces astropy Time + light_travel_time (geocentre) to < 1 ms."""
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    mjd = np.array([57000.123, 58500.5, 60400.25])
    ra, dec = 201.7846706, 38.7450683
    ours = td.utc_mjd_to_bmjd_tdb(mjd, ra, dec)
    with iers.conf.set_temp("auto_download", False), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        times = Time(mjd, format="mjd", scale="utc", location=EarthLocation.from_geocentric(0, 0, 0, unit="m"))
        ltt = times.light_travel_time(SkyCoord(ra, dec, unit="deg"), kind="barycentric", ephemeris="builtin")
        expected = (times.tdb + ltt).mjd
    assert np.all(np.abs(ours - expected) * 86400.0 < 1e-3)


# ---------------------------------------------------------------------------
# 4. Empty / HTML HTTP-200 answers are failures, never "no source", and are never cached
# ---------------------------------------------------------------------------


def _target_service(request: httpx.Request, which: str) -> bool:
    if which == "irsa_tap":
        return request.url.host == "irsa.ipac.caltech.edu" and request.url.path.endswith("/TAP/sync")
    if which == "gaia_tap":
        return request.url.host == "gea.esac.esa.int" and request.url.path.endswith("/tap/sync")
    if which == "ztf_lc":
        return "nph_light_curves" in request.url.path
    raise AssertionError(which)


@pytest.mark.parametrize("body", [HTML, "", "  \n"], ids=["html", "empty", "blank"])
@pytest.mark.parametrize("which,failed", [("irsa_tap", {"ztf", "neowise"}), ("gaia_tap", {"gaia"}),
                                          ("ztf_lc", {"ztf"})])
async def test_unusable_http_200_is_a_retryable_survey_failure(which: str, failed: set[str], body: str) -> None:
    real = replay_handler("rrab_css_j132708")

    def side_effect(request: httpx.Request) -> httpx.Response:
        if _target_service(request, which):
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})
        return real(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, surveys="ztf,neowise,gaia", client=c, period=False,
                                              use_cache=False)
    assert {f["survey"] for f in result.failures} == failed
    assert all(f["retryable"] for f in result.failures)
    assert all(("HTML" in f["error"]) or ("empty answer" in f["error"]) for f in result.failures)
    assert {s.survey for s in result.series} == {"ztf", "neowise", "gaia"} - failed
    empty_claims = ("no source in the cone", "no single-exposure detections", "no light-curve points", "no object in")
    assert not any(claim in n for n in result.notes for claim in empty_claims)


async def test_csv_without_the_expected_columns_is_a_failure() -> None:
    real = replay_handler("rrab_css_j132708")

    def side_effect(request: httpx.Request) -> httpx.Response:
        if _target_service(request, "gaia_tap"):
            return httpx.Response(200, text="message\nquota exceeded\n", headers={"content-type": "text/csv"})
        return real(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, surveys="gaia,neowise", client=c, period=False, use_cache=False)
    (failure,) = result.failures
    assert failure["survey"] == "gaia" and failure["retryable"] and "lacks column" in failure["error"]


async def test_failed_upstream_answer_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: HTML-200 answers were cached as 'no source' for the TTL, hiding 120 Gaia epochs for an hour."""
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setattr(td, "_cache", None)
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(200, text=HTML, headers={"content-type": "text/html"}))
        async with client() as c:
            with pytest.raises(td.UpstreamServiceError) as info:
                await td.get_lightcurves(*RRAB, surveys="neowise,gaia", client=c, period=False)
    assert info.value.retryable
    with replay("rrab_css_j132708"):
        async with client() as c:
            healthy = await td.get_lightcurves(*RRAB, surveys="neowise,gaia", client=c, period=False)
    assert healthy.failures == [] and {s.survey for s in healthy.series} == {"neowise", "gaia"}
    assert "cache" not in healthy.provenance["gaia"] and "cache" not in healthy.provenance["neowise"]
    # ... while a good answer is cached and served.
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=httpx.ConnectError("offline"))
        async with client() as c:
            again = await td.get_lightcurves(*RRAB, surveys="neowise,gaia", client=c, period=False)
    assert not route.called and again.provenance["gaia"]["cache"] == "hit"


# ---------------------------------------------------------------------------
# 5. TESS error model: quiet bright stars are not variable
# ---------------------------------------------------------------------------


def test_tess_error_model_has_a_calibrated_relative_floor() -> None:
    s = mag_series("tess", "TESS", np.arange(10.0), np.ones(10), 1e-5, unit="flux")
    assert td.series_error_model(s) == td.ERROR_MODELS[("tess", "TESS")] == (1.0, 1.0e-4)


def test_tess_granulation_level_scatter_is_not_variable_but_a_pulsator_is() -> None:
    """tau Cet: 5854 10-min points, pipeline errors 14 ppm, rms 91 ppm (granulation, PDC residuals): was
    VARIABLE with chi2/dof 44."""
    rng = np.random.default_rng(35)
    t = np.arange(0.0, 26.0, 10 / 1440)
    red = np.convolve(rng.normal(0, 1, t.size + 30), np.ones(30) / math.sqrt(30), mode="valid")[: t.size]
    y = 1.0 + 60e-6 * red + rng.normal(0, 60e-6, t.size)
    quiet = td.series_variability(mag_series("tess", "TESS", t, y, 14e-6, unit="flux"))
    assert quiet.is_variable is False and quiet.chi2_dof < 1.5, quiet.evidence
    y2 = y + 0.002 * np.sin(2 * np.pi * t / 0.8)  # 2-mmag semi-amplitude pulsation
    assert td.series_variability(mag_series("tess", "TESS", t, y2, 14e-6, unit="flux")).is_variable is True


def test_tau_ceti_tess_sector_is_not_variable() -> None:
    result = run_case("tau_cet_tess", case_tau_cet_tess)
    assert result.failures == []
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 419015728"] and len(tess.points) > 3000
    assert tess.metadata["error_floor_relative_flux"] == 1.0e-4
    m = result.variability["tess:TESS"]
    assert m.is_variable is False and m.chi2_dof < 1.5, m.evidence
    assert result.summary_is_variable() is False
    assert result.period_search.best is None


# ---------------------------------------------------------------------------
# 6. Malformed upstream payloads are 502 (not 422/404)
# ---------------------------------------------------------------------------

GOOD_SKYBOT_ROW = {"Num": "4", "Name": "Vesta", "RA (hms)": "00 34 12.3456", "DEC (dms)": "+05 00 00.00",
                   "Class": "MB>Inner", "VMag (mag)": "7.5"}
BAD_SKYBOT_ROW = {"Num": "", "Name": "2031 XX", "RA (hms)": "00 34 --.--", "DEC (dms)": "+05 00 00.00",
                  "Class": "MB>Inner"}


def test_skybot_malformed_row_is_dropped_and_noted() -> None:
    notes: list[str] = []
    objects = td.parse_skybot_json([GOOD_SKYBOT_ROW, BAD_SKYBOT_ROW], 8.55, 5.0, notes=notes)
    assert [o.name for o in objects] == ["Vesta"]
    assert notes == ["SkyBoT: 1 object(s) with unparseable positions were dropped."]
    with pytest.raises(td.UpstreamServiceError, match="could be parsed"):
        td.parse_skybot_json([BAD_SKYBOT_ROW], 8.55, 5.0)


def test_router_skybot_partial_garbage_is_200_with_note_and_total_garbage_is_502() -> None:
    params = {"ra": 8.55, "dec": 5.0, "epoch_mjd": 60000, "radius_arcsec": 600}
    with respx.mock(assert_all_mocked=True) as router:
        router.get(td.SKYBOT_CONESEARCH_URL).mock(return_value=httpx.Response(200, json=[GOOD_SKYBOT_ROW,
                                                                                          BAD_SKYBOT_ROW]))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/solar-system", params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [o["name"] for o in body["objects"]] == ["Vesta"] and "1 object(s) with unparseable" in body["notes"][0]
    with respx.mock(assert_all_mocked=True) as router:
        router.get(td.SKYBOT_CONESEARCH_URL).mock(return_value=httpx.Response(200, json=[BAD_SKYBOT_ROW]))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/solar-system", params=params)
    assert response.status_code == 502 and "skybot" in response.json()["detail"].lower()


@pytest.mark.parametrize("body", [HTML, "", '<?xml version="1.0"?><html><body>maintenance</body></html>',
                                  '<?xml version="1.0" encoding="UTF-8"?>\n<Sesame>\n<Target option="SNV">\n<name>RR'],
                         ids=["html", "empty", "xhtml", "truncated"])
def test_router_sesame_unusable_http_200_is_502_not_404(body: str) -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/html"}))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"name": "RR Lyr"})
    assert response.status_code == 502, response.text
    assert "resolver unavailable" in response.json()["detail"]


async def test_resolve_target_accepts_a_real_sesame_answer() -> None:
    with replay("name_3c273_gaia"):
        async with client() as c:
            target, record = await td.resolve_target("3C 273", c)
    assert abs(target.ra - QSO_3C273[0]) < 1e-5 and record["canonical_name"]


# ---------------------------------------------------------------------------
# 7. Tests that can catch their regressions
# ---------------------------------------------------------------------------


def test_router_sends_every_upstream_request_through_the_shared_client() -> None:
    """Review: respx mocked every client, so returning None from _state_client went unnoticed."""
    seen: list[str] = []

    async def hook(request: httpx.Request) -> None:
        seen.append(request.url.host)

    app = make_app()
    app.state.client = httpx.AsyncClient(follow_redirects=True, event_hooks={"request": [hook]})
    with replay("rrab_css_j132708"), TestClient(app) as http:
        response = http.get("/api/v1/lightcurves", params={"ra": RRAB[0], "dec": RRAB[1],
                                                            "surveys": "ztf,neowise,gaia", "period": "false"})
    assert response.status_code == 200, response.text
    assert len(seen) == len(load_case("rrab_css_j132708"))
    assert set(seen) == {"irsa.ipac.caltech.edu", "gea.esac.esa.int"}


def test_neowise_times_are_bmjd_tdb_of_the_exposures() -> None:
    """Review: removing the NEOWISE UTC -> BMJD_TDB conversion broke no test."""
    body = next(e for e in load_case("rrab_css_j132708") if "neowiser_p1bs_psd" in e["request_body"])["content"]
    result = td.parse_neowise_csv(body.decode(), *RRAB, binned=False)
    w1 = next(s for s in result.series if s.band == "W1")
    rows = td._csv_rows(body.decode())
    utc = np.array(sorted({float(r["mjd"]) for r in rows if r["w1mpro"]}))
    expected = td.utc_mjd_to_bmjd_tdb(utc, *RRAB)
    got = np.array([p.mjd for p in w1.points])
    # every exposure epoch is a converted UTC epoch (69.2 s TT-UTC + TDB-TT + Roemer delay), not the raw UTC
    assert len(got) > 300
    assert np.all(np.min(np.abs(got[:, None] - expected[None, :]), axis=1) < 1e-9)
    assert np.all(np.min(np.abs(got[:, None] - utc[None, :]), axis=1) > 1e-7)
    assert np.ptp(expected - utc) * 86400.0 > 30.0  # the barycentric (Roemer) term varies over the year


def test_include_flagged_returns_flagged_epochs_of_every_survey_but_statistics_ignore_them() -> None:
    async def go() -> tuple[td.LightCurveResult, td.LightCurveResult]:
        with replay("rrab_css_j132708"):
            async with client() as c:
                flagged = await td.get_lightcurves(*RRAB, surveys="ztf,neowise,gaia", client=c, period=False,
                                                   include_flagged=True, use_cache=False)
                clean = await td.get_lightcurves(*RRAB, surveys="ztf,neowise,gaia", client=c, period=False,
                                                 use_cache=False)
        return flagged, clean

    flagged, clean = asyncio.run(go())
    by_key = {s.key: s for s in flagged.series}
    for key in ("ztf:g", "ztf:r", "neowise:W1", "neowise:W2", "gaia:BP", "gaia:RP"):
        s = by_key[key]
        bad = [p for p in s.points if p.flag != 0]
        assert bad, key
        assert flagged.variability[key].n == clean.variability[key].n  # statistics use flag-0 epochs only
    assert {p.flag for p in by_key["neowise:W1"].points} <= {0, 1, 2}
    assert by_key["neowise:W1"].metadata["flagged_points"].startswith("single rejected exposures")
    assert len([p for p in by_key["ztf:g"].points if p.flag]) == by_key["ztf:g"].n_rejected


def test_tess_include_flagged_returns_quality_flagged_cadences() -> None:
    t = 60000.0 + np.arange(200) * (2 / 1440)
    quality = np.zeros(200, dtype=np.int64)
    quality[[10, 11, 50]] = [4, 4, 128]
    lc = {"t": t, "pdcsap": np.full(200, 1000.0), "pdcsap_err": np.full(200, 1.0), "sap": np.full(200, 1000.0),
          "sap_err": np.full(200, 1.0), "quality": quality,
          "info": {"sector": 1, "crowdsap": 1.0, "tess_mag": 8.0}}
    points, _meta, _notes, total, rejected = td._assemble_tess([(1, "uri", lc)], 10.0, include_flagged=True)
    flagged = [p for p in points if p.flag]
    assert [p.flag for p in flagged] == [4, 4, 128] and total == 200 and rejected == 3
    good, *_ = td._assemble_tess([(1, "uri", lc)], 10.0)
    assert all(p.flag == 0 for p in good) and sum(p.n for p in good) == 197
    unbinned, *_ = td._assemble_tess([(1, "uri", lc)], 0.0)
    assert len(unbinned) == 197 and all(p.n is None for p in unbinned)


def test_cli_lightcurve_by_name_resolves_and_prints_multiband_caveats(capsys: pytest.CaptureFixture[str]) -> None:
    """Review: the CLI --name path had no offline test, and the multi-band peak of 3C 273 (1.4 cycles, no FAP)
    was printed after 'no significant period' without its caveats."""
    args = make_parser().parse_args(["lightcurve", "--name", "3C 273", "--surveys", "gaia"])
    with replay("name_3c273_gaia"):
        code = args.handler(args)
    out = capsys.readouterr().out
    assert code == 0, out
    assert "[gaia:G" in out and "Period: no significant period" in out
    # Round 4: Gaia BP/RP (not decisive) no longer enter the multi-band sum, so a Gaia-only
    # query has no multi-band peak at all (the 3C 273 one was a BP-dominated 1.4-cycle peak).
    assert not any(line.startswith("Multi-band peak") for line in out.splitlines()), out
    # A multi-band sum over decisive bands (ZTF g/r/i of the blazar OJ 287, --name path, default
    # surveys) is printed with its caveats and, without an adopted period, as not a detection.
    args = make_parser().parse_args(["lightcurve", "--name", "OJ 287"])
    with replay("oj287_name"):
        code = args.handler(args)
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Period: no significant period" in out
    mb_line = next(line for line in out.splitlines() if line.startswith("Multi-band peak"))
    assert "ztf:g" in mb_line and "gaia:" not in mb_line
    assert "not a detection" in mb_line and "no analytic false-alarm probability" in mb_line


def test_cli_tess_options_are_validated_before_any_request(capsys: pytest.CaptureFixture[str]) -> None:
    parser = make_parser()
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        for extra in (["--tess-sectors", "100"], ["--tess-sectors", "0"], ["--tess-bin-minutes", "-1"],
                      ["--tess-bin-minutes", "0", "--tess-sectors", "5"]):
            args = parser.parse_args(["lightcurve", "--ra", "10", "--dec", "10", "--surveys", "tess", *extra])
            assert args.handler(args) == 2, extra
        assert not route.called
    assert "tess_" in capsys.readouterr().err


def test_bin_uniform_scales_linearly() -> None:
    """Complexity instead of a wall-clock limit: 10x the points must cost far less than 100x the time."""
    rng = np.random.default_rng(24)

    def cost(n: int) -> float:
        t = np.arange(n) * (2 / 1440)
        y = 1 + rng.normal(0, 0.001, n)
        best = math.inf
        for _ in range(3):
            started = time.perf_counter()
            td.bin_uniform(t, y, np.full(n, 0.001), 10 / 1440)
            best = min(best, time.perf_counter() - started)
        return best

    assert cost(180_000) / cost(18_000) < 30.0


def test_skybot_parser_converts_positions_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = td._sexagesimal_coords

    def counting(items: Any) -> Any:
        calls.append(len(items))
        return original(items)

    monkeypatch.setattr(td, "_sexagesimal_coords", counting)
    payload = [dict(GOOD_SKYBOT_ROW, Name=f"A{i}", **{"RA (hms)": f"00 34 {12 + i * 1e-3:07.4f}"}) for i in range(500)]
    assert len(td.parse_skybot_json(payload, 8.55, 5.0)) == 500
    assert calls == [500]  # vectorised: one conversion for all rows


# ---------------------------------------------------------------------------
# 8. CPU-bound work runs off the event loop
# ---------------------------------------------------------------------------


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_router_serialises_the_response_in_a_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    on_loop: list[bool] = []
    original = td.LightCurveResult.as_dict

    def spy(self: td.LightCurveResult) -> dict[str, Any]:
        on_loop.append(_on_event_loop())
        return original(self)

    monkeypatch.setattr(td.LightCurveResult, "as_dict", spy)
    with replay("rrab_css_j132708"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"ra": RRAB[0], "dec": RRAB[1], "surveys": "ztf,gaia"})
    assert response.status_code == 200 and on_loop == [False]
    body = response.json()
    assert body["period"]["noise_continuum"] is not None  # the extended period fields are exposed


async def test_parsers_and_cache_serialisation_run_in_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "60")
    monkeypatch.setattr(td, "_cache", None)
    on_loop: dict[str, list[bool]] = {}
    for name in ("parse_ztf_csv", "parse_neowise_csv", "parse_gaia_epoch_csv"):
        original = getattr(td, name)

        def spy(*args: Any, _name: str = name, _orig: Any = original, **kwargs: Any) -> Any:
            on_loop.setdefault(_name, []).append(_on_event_loop())
            return _orig(*args, **kwargs)

        monkeypatch.setattr(td, name, spy)
    original_dict = td.SurveyResult.as_dict

    def dict_spy(self: td.SurveyResult) -> dict[str, Any]:
        on_loop.setdefault("SurveyResult.as_dict", []).append(_on_event_loop())
        return original_dict(self)

    monkeypatch.setattr(td.SurveyResult, "as_dict", dict_spy)
    with replay("rrab_css_j132708"):
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, surveys="ztf,neowise,gaia", client=c, period=False)
    assert set(on_loop) == {"parse_ztf_csv", "parse_neowise_csv", "parse_gaia_epoch_csv", "SurveyResult.as_dict"}
    assert not any(any(v) for v in on_loop.values()), on_loop
    assert result.provenance["ztf"]["rows"] > 100 and result.provenance["neowise"]["rows"] > 100


# ---------------------------------------------------------------------------
# 9 / 17. Proper motion without an epoch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{"pm_ra_masyr": 500, "pm_dec_masyr": 500}, {"pm_ra_masyr": 1000}])
def test_router_rejects_proper_motion_without_epoch(params: dict[str, Any]) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"ra": 10, "dec": 10, "surveys": "gaia", **params})
        assert not route.called
    assert response.status_code == 422 and "epoch" in response.json()["detail"]


async def test_core_rejects_proper_motion_without_epoch() -> None:
    with pytest.raises(td.InvalidCoordinateError, match="epoch"):
        await td.get_lightcurves(10.0, 10.0, pm_ra_masyr=1000.0, pm_dec_masyr=0.0)


# ---------------------------------------------------------------------------
# 10. Bounded work: survey deadlines and TESS limits in the core API
# ---------------------------------------------------------------------------


async def test_survey_deadline_bounds_a_hanging_service(monkeypatch: pytest.MonkeyPatch) -> None:
    async def hang(*_args: Any, **_kwargs: Any) -> td.SurveyResult:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    monkeypatch.setattr(td, "fetch_gaia", hang)
    monkeypatch.setitem(td.SURVEY_DEADLINES, "gaia", 0.2)
    started = time.perf_counter()
    with replay("rrab_css_j132708"):
        async with client() as c:
            result = await td.get_lightcurves(*RRAB, surveys="neowise,gaia", client=c, period=False, use_cache=False)
    assert time.perf_counter() - started < 30
    (failure,) = result.failures
    assert failure["survey"] == "gaia" and failure["retryable"] and "deadline" in failure["error"]
    assert {s.survey for s in result.series} == {"neowise"}


@pytest.mark.parametrize("kwargs", [{"tess_max_sectors": 100}, {"tess_max_sectors": 0}, {"tess_bin_minutes": -1.0},
                                    {"tess_bin_minutes": float("nan")}, {"tess_bin_minutes": 0.0, "tess_max_sectors": 4}])
async def test_core_api_bounds_tess_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="tess_"):
        await td.get_lightcurves(10.0, 10.0, surveys="tess", **kwargs)


# ---------------------------------------------------------------------------
# 12. Eclipsing binaries: orbital period, not P/2
# ---------------------------------------------------------------------------


def test_equal_minima_eclipsing_binary_reports_orbital_period() -> None:
    """Review: equal-depth EB, P = 1.5 d, 900 random epochs -> 0.75 d with no warning and no alternative."""
    rng = np.random.default_rng(40)
    period = 1.5
    t = np.sort(rng.uniform(0, 1500, 900))
    y = eclipsing_binary(t, period, depth1=0.5, depth2=0.5, width=0.03) + rng.normal(0, 0.01, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.01))
    assert res is not None and res.reliable
    assert abs(res.best_period_days - period) / period < 1e-4
    assert abs(res.lomb_scargle_period_days - period / 2) / (period / 2) < 1e-3
    assert res.harmonic_test["doubled_by"] == "eclipse_shape"
    assert res.alternative_period_days == pytest.approx(period / 2, rel=1e-4)
    assert res.period_error_days is not None and abs(res.best_period_days - period) < 5 * res.period_error_days


def test_period_error_of_doubled_equal_minima_binary_uses_the_harmonics() -> None:
    """With equal minima the fundamental of the orbital period has ~zero amplitude; an error from A_1 alone
    blew up (CM Dra: +-0.028 d for a period good to 4e-5 d)."""
    rng = np.random.default_rng(41)
    t = np.sort(rng.uniform(0, 1500, 900))
    y = eclipsing_binary(t, 1.5, depth1=0.5, depth2=0.5, width=0.03) + rng.normal(0, 0.01, t.size)
    f, sigma_f, _k = td.refine_period(t, y, np.full_like(t, 0.01), 1 / 1.5, 0.5 / 1500, max_harmonics=20)
    _f_half, sigma_half, _k2 = td.refine_period(t, y, np.full_like(t, 0.01), 2 / 1.5, 0.5 / 1500, max_harmonics=20)
    assert sigma_f is not None and sigma_half is not None
    # same data, same information: sigma(f_orb) = sigma(2 f_orb) / 2
    assert sigma_f == pytest.approx(sigma_half / 2, rel=0.3)
    assert abs(f - 1 / 1.5) < 5 * sigma_f


def test_cm_dra_ztf_reports_the_orbital_period() -> None:
    """Review: CM Dra (VSX 1.26838965 d, equal minima) came out at 0.634274 d (doubled only from the LS P/4)."""
    result = run_case("cm_dra_ztf", case_cm_dra_ztf)
    g = next(s for s in result.series if s.key == "ztf:g")
    assert len(g.points) > 300
    best = result.period_search.best
    assert best is not None and best.series == "ztf:g"
    assert abs(best.best_period_days - CM_DRA_PERIOD_DAYS) < 5 * best.period_error_days
    assert best.period_error_days < 1e-3
    assert best.harmonic_test["n_doublings"] == 2
    assert best.alternative_period_days == pytest.approx(CM_DRA_PERIOD_DAYS / 2, rel=1e-3)


def test_chen2020_ea_with_equal_minima_reports_the_orbital_period() -> None:
    """Review: ZTFJ004818.49+625831.1 (1.2128074 d; Gaia EB 1.21293) came out at 0.606465 d, reliable."""
    result = run_case("ea_ztfj0048", case_ea2)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, EA2_PERIOD_DAYS, 1e-3)
    assert best.alternative_period_days == pytest.approx(EA2_PERIOD_DAYS / 2, rel=1e-3)
    assert any("eclipsing binary" in w for w in best.warnings)


def test_eclipse_shape_ignores_bin_noise_bumps() -> None:
    rng = np.random.default_rng(42)
    period = 0.634
    t = np.sort(rng.uniform(0, 2300, 420))
    y = eclipsing_binary(t, period, depth1=0.6, depth2=0.0, width=0.02, base=13.5)
    y = y + rng.normal(0, 0.02, t.size) + 0.02 * damped_random_walk(rng, t, tau=3.0, sigma=1.0)  # spots/flares
    shape = td.eclipse_shape(t, y, np.full_like(t, 0.012), 1 / period)
    assert shape["n_dips"] == 1 and shape["eclipse_like"] is True


# ---------------------------------------------------------------------------
# 13. Red-noise sources (quasars, blazars) get no period
# ---------------------------------------------------------------------------


def test_gaia_scanning_law_frequencies_are_aliases() -> None:
    assert td.gaia_scanning_alias(12.0005, 1030.0) == "3 x 6-h spin"  # OJ 287's spurious 0.08333-d 'period'
    assert td.gaia_scanning_alias(1440.0 / 106.5, 1030.0) == "1 x 106.5-min field-of-view separation"
    assert td.gaia_scanning_alias(1 / 63.12, 1030.0) == "1 x 63-d precession"
    assert td.gaia_scanning_alias(1 / 0.5942378, 1030.0) is None


def test_damped_random_walk_quasars_get_no_period_from_gaia() -> None:
    adopted = []
    for k in range(12):
        rng = np.random.default_rng(300 + k)
        t = gaia_like_times(rng)
        y = 18.0 + damped_random_walk(rng, t, tau=200.0, sigma=0.25) + rng.normal(0, 0.01, t.size)
        s = mag_series("gaia", "G", t, y, 0.008, phot_g_mean_mag=18.0)
        metrics = {s.key: td.series_variability(s)}
        search = td.search_periods([s], variability=metrics)
        if search.best is not None:
            adopted.append((k, search.best.best_period_days))
    assert adopted == []


def test_damped_random_walk_quasars_get_no_period_from_ztf() -> None:
    adopted = []
    for k in range(6):
        rng = np.random.default_rng(400 + k)
        series = []
        base = ztf_like_times(rng, 600)
        drw = damped_random_walk(rng, base, tau=150.0, sigma=0.15)
        for band, shift, scale in (("g", 0.0, 1.0), ("r", 0.03, 0.8)):
            t = base + shift
            series.append(mag_series("ztf", band, t, 18.5 + scale * drw + rng.normal(0, 0.02, t.size), 0.018))
        metrics = {s.key: td.series_variability(s) for s in series}
        search = td.search_periods(series, variability=metrics)
        if search.best is not None:
            adopted.append((k, search.best.best_period_days))
    assert adopted == []


def test_qso_j0747_has_no_adopted_period() -> None:
    """Review: SDSS J074756.99+454527.7 adopted a 232.41 +- 1.3 d 'period' from ZTF g."""
    result = run_case("qso_j0747_ztf", case_qso_j0747)
    assert result.failures == [] and any(result.variability[k].is_variable for k in ("ztf:g", "ztf:r"))
    assert result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "ztf:g")
    assert not g.reliable


def test_blazar_oj287_has_no_period() -> None:
    """Review: OJ 287 (ztf, neowise, gaia) adopted 0.08333 d (3rd harmonic of Gaia's 6-h spin), FAP 2.6e-6."""
    result = run_case("oj287_name", case_oj287_name)
    assert result.failures == [] and result.summary_is_variable() is True
    assert result.period_search.best is None and result.as_dict()["period"] is None
    g = next(p for p in result.period_search.per_series if p.series == "gaia:G")
    assert not g.reliable
    if abs(g.best_frequency_per_day - 12.0) < 0.01:
        assert any("Gaia scanning-law frequency (3 x 6-h spin)" in w for w in g.warnings)


def test_3c273_gaia_has_no_period() -> None:
    result = run_case("name_3c273_gaia", case_name_3c273_gaia)
    assert result.variability["gaia:G"].is_variable is True
    assert result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "gaia:G")
    assert g.best_period_days <= 1.0 / g.min_frequency * (1 + 1e-9)  # refinement stays inside the searched range


def test_red_noise_continuum_is_never_extrapolated_wildly() -> None:
    """A 2-3-cycle peak leaves one short side of the [f/3, 3f] window; the power-law fit extrapolated from it
    gave continua of 1e9-1e90 (seen for tau Cet in TESS and 3C 273/OJ 287 in Gaia BP)."""
    rng = np.random.default_rng(43)
    t = gaia_like_times(rng)
    y = 17.0 + damped_random_walk(rng, t, tau=300.0, sigma=0.3)
    sigma = np.full_like(t, 0.01)
    baseline = float(t.max() - t.min())
    out = td.red_noise_fap(t, y, sigma, 1.2 / baseline, 1, min_frequency=1 / baseline, max_frequency=20.0)
    assert 0 < out["continuum"] < 1e6 and math.isfinite(out["slope"])


# ---------------------------------------------------------------------------
# 15. Gaia BP/RP never drive the period ladder or the periodic evidence
# ---------------------------------------------------------------------------


def test_gaia_rp_cannot_double_the_g_period() -> None:
    """Review: V350 Lyr (RRab, 0.5942 d) came out at 1.188 d because RP alone had delta BIC +21."""
    rng = np.random.default_rng(44)
    period = 0.5942378
    t = gaia_like_times(rng, n_visits=30)
    g = mag_series("gaia", "G", t, rrab_mags(t, period) + rng.normal(0, 0.01, t.size), 0.004, phot_g_mean_mag=15.0)
    # RP: the same star plus neighbour contamination alternating from cycle to cycle (a 2P signal)
    contamination = 0.25 * np.cos(np.pi * t / period)
    rp = mag_series("gaia", "RP", t, rrab_mags(t, period, 14.5) + contamination + rng.normal(0, 0.01, t.size), 0.004)
    metrics = {s.key: td.series_variability(s) for s in (g, rp)}
    search = td.search_periods([g, rp], variability=metrics)
    best = search.best
    assert best is not None and best.series == "gaia:G"
    assert abs(best.best_period_days - period) / period < 1e-3
    assert best.harmonic_test["doubled"] is False
    assert len(best.harmonic_test["per_band_delta_bic"]) == 1  # decided on G only


def test_gaia_bp_cannot_confirm_a_periodic_upgrade() -> None:
    per = [td.PeriodogramResult("gaia:G", 60, 1000.0, 0.5, 2.0, 0.5, 1e-12, 0.001, 20.0, 1000, 5.0, "m",
                                peaks=[{"period_days": 0.5, "frequency_per_day": 2.0, "power": 0.5}], reliable=True,
                                lomb_scargle_period_days=0.5),
           td.PeriodogramResult("gaia:BP", 60, 1000.0, 0.5, 2.0, 0.5, 1e-12, 0.001, 20.0, 1000, 5.0, "m",
                                peaks=[{"period_days": 0.5, "frequency_per_day": 2.0, "power": 0.5}], reliable=True,
                                lomb_scargle_period_days=0.5)]
    m = td.VariabilityMetrics(n=60, unit="mag", is_variable=False, significance_sigma=8.0)
    td._apply_periodic_evidence(per, {"gaia:G": m})
    assert m.is_variable is False  # BP (window photometry) is not an independent confirmation


def test_v350_lyr_gaia_only_period_is_not_doubled() -> None:
    result = run_case("v350_lyr_gaia", case_v350_lyr_gaia)
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G"
    assert within(best.best_period_days, V350_LYR_PERIOD_DAYS, 1e-3)
    assert best.harmonic_test["doubled"] is False


# ---------------------------------------------------------------------------
# 16. Cycle-count ambiguity across long gaps; daily aliases between bands
# ---------------------------------------------------------------------------


def test_sectors_years_apart_widen_the_period_error_to_the_alias_spacing() -> None:
    """Review: Algol with sectors 58 and 85 (754 d apart): 2.861672 +- 0.000300 d, 19 sigma off."""
    rng = np.random.default_rng(45)
    seg = np.arange(0.0, 26.0, 30 / 1440)
    t = np.concatenate([59880.0 + seg, 59880.0 + 754.0 + seg])
    phase = (t / ALGOL_PERIOD_DAYS) % 1.0
    d1 = np.minimum(phase, 1 - phase)
    y = 1.0 - 0.7 * np.exp(-0.5 * (d1 / 0.02) ** 2) - 0.05 * np.exp(-0.5 * ((phase - 0.5) / 0.02) ** 2)
    y = y + rng.normal(0, 3e-4, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 3e-4), unit="flux", min_period_days=0.5)
    assert res is not None
    comb = res.harmonic_test.get("alias_comb")
    assert comb is not None and comb["gap_days"] > 700
    spacing_days = comb["spacing"] * res.best_period_days**2
    assert res.period_error_days >= 0.5 * spacing_days
    assert any("cycle count across the gap is ambiguous" in w for w in res.warnings)
    assert abs(res.best_period_days - ALGOL_PERIOD_DAYS) < 3 * res.period_error_days


def test_daily_alias_of_another_bands_peak_is_marked_unreliable() -> None:
    """Review: V445 Lyr (0.513075 d): ztf:g was 'reliable' at 0.2530346 d = f_true + 2 c/d, missed because
    diurnal_alias only checks n x 1 c/d itself."""
    rng = np.random.default_rng(46)
    period = V445_LYR_PERIOD_DAYS
    grams = {}
    per = []
    for band, n in (("g", 60), ("r", 90)):
        t = ztf_like_times(rng, n * 2)[:n]
        y = rrab_mags(t, period) + rng.normal(0, 0.02, t.size)
        sigma = np.full_like(t, 0.02)
        freq, used = td.frequency_grid(float(t.max() - t.min()))
        gram = td._periodogram(t, y, sigma, freq, used, series=f"ztf:{band}")
        f = 1 / period + (2 * td.SIDEREAL_DAY_FREQUENCY if band == "g" else 0.0)  # g picked the alias
        gram.result.best_frequency_per_day = f
        gram.result.best_period_days = 1 / f
        gram.result.lomb_scargle_period_days = 1 / f
        gram.result.n_harmonics = 3
        gram.result.reliable = True
        grams[gram.result.series] = gram
        per.append(gram.result)
    td._flag_cross_band_aliases(per, grams)
    g, r = per
    assert g.reliable is False and any("daily alias of the ztf:r peak" in w for w in g.warnings)
    assert r.reliable is True


def test_v445_lyr_adopts_the_true_period() -> None:
    result = run_case("v445_lyr_ztf", case_v445_lyr_ztf)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, V445_LYR_PERIOD_DAYS, 1e-3)
    g = next(p for p in result.period_search.per_series if p.series == "ztf:g")
    assert not (g.reliable and not within(g.best_period_days, V445_LYR_PERIOD_DAYS, 1e-2))


# ---------------------------------------------------------------------------
# Refinement stays inside the searched frequency range
# ---------------------------------------------------------------------------


def test_refinement_never_leaves_the_searched_range() -> None:
    """3C 273 Gaia G: the 727-d Lomb-Scargle peak was 'refined' to 1115 d, longer than the 1040-d baseline."""
    rng = np.random.default_rng(47)
    t = gaia_like_times(rng)
    y = 17.0 + damped_random_walk(rng, t, tau=400.0, sigma=0.2)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.01))
    assert res is not None and res.best_frequency_per_day >= res.min_frequency * (1 - 1e-9)
    f, _sf, _k = td.refine_period(t, y, np.full_like(t, 0.01), 1.2 / 1000, 5 / 1000, bounds=(1 / 1000, 20.0))
    assert f >= 1 / 1000 * (1 - 1e-9)
