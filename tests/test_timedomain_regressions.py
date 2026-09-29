"""Offline regression tests for the time-domain review fixes (synthetic data and unit behaviour).

Each test names the defect it guards against. Real-data replays of the same defects live in
``tests/test_timedomain.py`` (fixtures recorded by ``tests/test_timedomain_live.py record``).
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import timedomain as td
from models import validate_target

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ztf_like_times(rng: np.random.Generator, n: int, years: float = 6.0) -> np.ndarray:
    """One epoch per night at a random hour, with seasonal gaps (ground-based sampling)."""
    nights = np.sort(rng.choice(int(years * 365), size=n, replace=False)).astype(float)
    season = (nights % 365) < 250
    return 58200.0 + nights[season] + rng.uniform(0.15, 0.45, size=int(season.sum()))


def eclipsing_binary_mags(t: np.ndarray, period: float, *, depth1: float, depth2: float, width: float = 0.05,
                          base: float = 15.0) -> np.ndarray:
    """Detached EB: flat out of eclipse, Gaussian-shaped primary (phase 0) and secondary (0.5) minima."""
    phase = (t / period) % 1.0
    d1 = np.minimum(np.abs(phase), np.abs(phase - 1.0))
    d2 = np.abs(phase - 0.5)
    return base + depth1 * np.exp(-0.5 * (d1 / width) ** 2) + depth2 * np.exp(-0.5 * (d2 / width) ** 2)


def mag_series(survey: str, band: str, t: np.ndarray, y: np.ndarray, err: float, **meta: Any) -> td.LightCurveSeries:
    pts = [td.LightCurvePoint(float(a), float(b), err) for a, b in zip(t, y)]
    return td.LightCurveSeries(survey, band, "mag", pts, "AB", metadata=dict(meta))


# ---------------------------------------------------------------------------
# Variability decision (review: MAD veto hid eclipsing binaries; floors mis-calibrated)
# ---------------------------------------------------------------------------


def test_detached_eclipsing_binary_is_variable() -> None:
    """1500 epochs, 1.0 mag primary eclipse (~10 % of the phase), 0.02 mag noise: the old MAD rule said constant."""
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 2000, 1500))
    y = eclipsing_binary_mags(t, 1.8083775, depth1=1.0, depth2=0.3, width=0.02) + rng.normal(0, 0.02, t.size)
    m = td.variability_metrics(t, y, np.full_like(t, 0.02))
    assert m.mad_std < 0.03  # the out-of-eclipse majority is flat ...
    assert m.is_variable is True and m.decision_basis == "chi2"  # ... but the source is variable
    assert m.n_outliers_excluded == 0  # eclipse points are far too many to be "outliers"
    assert m.chi2_dof > 50


def test_few_isolated_outliers_still_do_not_make_a_star_variable() -> None:
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 1000, 1500))
    y = 16.0 + rng.normal(0, 0.02, t.size)
    y[[100, 700, 1200]] += [1.5, -2.0, 1.0]
    m = td.variability_metrics(t, y, np.full_like(t, 0.02))
    assert m.n_outliers_excluded == 3 and m.is_variable is False
    assert any("isolated" in e for e in m.evidence)


def test_isolated_outlier_allowance_scales_with_n() -> None:
    y = np.zeros(1000)
    sigma = np.full(1000, 0.01)
    spread10 = np.linspace(0, 999, 10).astype(int)  # isolated in time order
    y[spread10] = 1.0  # 1 % of 1000 -> allowed
    assert td.isolated_outliers(y, sigma).sum() == 10
    y = np.zeros(1000)
    y[np.linspace(0, 999, 11).astype(int)] = 1.0  # 11 > floor(1 % of 1000) -> a property of the source
    assert td.isolated_outliers(y, sigma).sum() == 0
    # Review (round 2): extreme epochs adjacent in time are an event, never "isolated outliers".
    y = np.zeros(1000)
    y[:10] = 1.0
    assert td.isolated_outliers(y, sigma).sum() == 0


def test_error_model_is_applied_and_reported() -> None:
    rng = np.random.default_rng(12)
    t = np.sort(rng.uniform(0, 1000, 800))
    # Scatter 0.016 mag with pipeline errors of 0.012: the ZTF g model (x1.15 (+) 4 mmag) explains it.
    y = 16.0 + rng.normal(0, math.hypot(1.15 * 0.012, 0.004), t.size)
    series = mag_series("ztf", "g", t, y, 0.012)
    m = td.series_variability(series)
    assert (m.error_scale, m.error_floor) == td.ERROR_MODELS[("ztf", "g")]
    assert 0.85 < m.chi2_dof < 1.15 and m.is_variable is False


def test_ztf_floor_no_longer_hides_low_amplitude_periodic_variables() -> None:
    """Review: 0.03 mag semi-amplitude, 0.015 mag errors, 700 epochs/band gave is_variable False and no period."""
    rng = np.random.default_rng(13)
    period = 0.2718
    series = []
    for band in ("g", "r"):
        t = ztf_like_times(rng, 900)
        y = 16.0 + 0.03 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.015, t.size)
        series.append(mag_series("ztf", band, t, y, 0.015))
    metrics = {s.key: td.series_variability(s) for s in series}
    search = td.search_periods(series, variability=metrics)
    for key, m in metrics.items():
        assert m.is_variable is True, (key, m.evidence)
    assert search.best is not None and abs(search.best.best_period_days - period) / period < 1e-3


def test_periodic_evidence_upgrades_sub_noise_periodic_variable() -> None:
    """Variance below the noise (chi2/dof < 2) but a >> significant, cross-band periodic signal."""
    rng = np.random.default_rng(14)
    period = 1.3578034  # BY Dra star ZTFJ155741.59+364734.1 (Chen et al. 2020), amplitude ~0.04
    series = []
    for band in ("g", "r"):
        t = ztf_like_times(rng, 1600)
        y = 16.0 + 0.012 * np.sin(2 * np.pi * t / period + 0.4) + rng.normal(0, 0.012, t.size)
        series.append(mag_series("ztf", band, t, y, 0.011))
    metrics = {s.key: td.series_variability(s) for s in series}
    assert all(m.chi2_dof < td.VARIABLE_CHI2_DOF and m.is_variable is False for m in metrics.values())
    search = td.search_periods(series, variability=metrics)
    assert all(m.is_variable is True and m.decision_basis == "periodic" for m in metrics.values())
    assert search.best is not None and abs(search.best.best_period_days - period) / period < 1e-3


def test_constant_star_with_sidereal_systematic_is_not_upgraded() -> None:
    rng = np.random.default_rng(15)
    series = []
    for band in ("g", "r"):
        t = ztf_like_times(rng, 900)
        y = 16.0 + 0.01 * np.cos(2 * np.pi * td.SIDEREAL_DAY_FREQUENCY * t) + rng.normal(0, 0.012, t.size)
        series.append(mag_series("ztf", band, t, y, 0.011))
    metrics = {s.key: td.series_variability(s) for s in series}
    search = td.search_periods(series, variability=metrics)
    assert all(m.is_variable is False for m in metrics.values())
    assert search.best is None


def test_gaia_g_error_model_uses_evans_table_and_floor() -> None:
    scale, floor = td.gaia_g_error_model(19.0)
    assert scale == 1.2 and floor == pytest.approx(0.0052)  # Evans et al. 2023 Table 2 at G = 19.0
    assert td.gaia_g_error_model(14.0)[1] == 0.003  # Table 2 (~1 mmag) raised to the 3 mmag floor
    assert td.gaia_g_error_model(None) == (1.2, 0.003)


def test_gaia_bp_rp_are_not_decisive() -> None:
    rng = np.random.default_rng(16)
    t = np.sort(rng.uniform(56900, 57800, 40))
    y = 15.0 + rng.normal(0, 0.05, t.size)
    bp = mag_series("gaia", "BP", t, y, 0.005)
    m = td.series_variability(bp)
    assert m.chi2_dof > 10 and m.is_variable is None
    assert any("not used for the variability decision" in e for e in m.evidence)


def test_summary_is_none_when_no_series_is_decided() -> None:
    few = td.variability_metrics([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [0.01, 0.01, 0.01])
    result = td.LightCurveResult({"ra": 0, "dec": 0}, [], {"ztf:g": few}, None)
    assert few.is_variable is None
    assert result.summary_is_variable() is None and result.as_dict()["variability"]["summary"]["is_variable"] is None
    quiet = td.variability_metrics(np.arange(50.0), np.full(50, 10.0) + np.tile([0.01, -0.01], 25), np.full(50, 0.01))
    result = td.LightCurveResult({"ra": 0, "dec": 0}, [], {"ztf:g": few, "ztf:r": quiet}, None)
    assert result.summary_is_variable() is False


# ---------------------------------------------------------------------------
# chi2 significance and strict JSON (review: -inf for chi2 << dof)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi2,dof", [(200.0, 500), (1.0, 5000), (0.0, 30), (500.0, 500), (1e7, 500), (3.0, 1)])
def test_chi2_significance_is_finite_and_signed(chi2: float, dof: int) -> None:
    z = td.chi2_significance(chi2, dof)
    assert math.isfinite(z)
    if chi2 < dof - 3 * math.sqrt(2 * dof):
        assert z < 0
    if chi2 > dof + 3 * math.sqrt(2 * dof):
        assert z > 0


def test_chi2_significance_matches_exact_inversion() -> None:
    from scipy import stats
    for chi2, dof in ((520.0, 500), (480.0, 500), (225.3, 580)):
        expected = (stats.norm.isf(stats.chi2.sf(chi2, dof)) if chi2 > dof
                    else -stats.norm.isf(stats.chi2.cdf(chi2, dof)))  # lower tail inverted directly
        assert td.chi2_significance(chi2, dof) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    values = [td.chi2_significance(c, 500) for c in (100.0, 300.0, 500.0, 700.0, 5000.0, 1e7)]
    assert values == sorted(values)


def test_result_as_dict_is_strict_json() -> None:
    rng = np.random.default_rng(17)
    t = np.sort(rng.uniform(0, 1000, 600))
    y = 16.0 + rng.normal(0, 0.005, t.size)  # errors overestimated 4x -> chi2 << dof
    s = mag_series("ztf", "r", t, y, 0.02)
    m = td.series_variability(s)
    assert m.significance_sigma is not None and m.significance_sigma < -10
    result = td.LightCurveResult({"ra": 1.0, "dec": 2.0}, [s], {s.key: m}, td.search_periods([s], variability={s.key: m}))
    text = json.dumps(result.as_dict(), allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text
    metrics = td.VariabilityMetrics(n=3, unit="mag", chi2=float("inf"), significance_sigma=float("-inf"))
    assert metrics.as_dict()["chi2"] is None and metrics.as_dict()["significance_sigma"] is None


# ---------------------------------------------------------------------------
# Periods: doubling, refinement, grid limits, aliases
# ---------------------------------------------------------------------------


def test_eclipsing_binary_period_is_doubled() -> None:
    """Unequal minima: the single-harmonic peak lies at P/2; the 2P test must recover P."""
    rng = np.random.default_rng(18)
    period = 2.9458406
    series = []
    for band, d2 in (("g", 0.35), ("r", 0.30)):
        t = ztf_like_times(rng, 1000)
        y = eclipsing_binary_mags(t, period, depth1=0.9, depth2=d2, width=0.04) + rng.normal(0, 0.02, t.size)
        series.append(mag_series("ztf", band, t, y, 0.02))
    metrics = {s.key: td.series_variability(s) for s in series}
    search = td.search_periods(series, variability=metrics)
    best = search.best
    assert best is not None and best.harmonic_test["doubled"] is True
    assert abs(best.best_period_days - period) / period < 1e-3
    assert abs(best.lomb_scargle_period_days - period / 2) / (period / 2) < 1e-2
    assert best.alternative_period_days == pytest.approx(period / 2, rel=1e-2)


def test_pulsator_period_is_not_doubled() -> None:
    rng = np.random.default_rng(19)
    period = 0.6297706
    t = ztf_like_times(rng, 700)
    phase = 2 * np.pi * t / period
    y = 15.3 + 0.35 * np.sin(phase) + 0.12 * np.sin(2 * phase + 1.0) + 0.05 * np.sin(3 * phase + 2.0)
    y = y + rng.normal(0, 0.02, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.02), ground_based=True)
    assert res is not None and res.reliable and res.harmonic_test["doubled"] is False
    assert abs(res.best_period_days - period) / period < 1e-4
    assert res.n_harmonics >= 3


def test_refinement_beats_grid_quantisation_on_short_baseline() -> None:
    """Review: one TESS sector (26 d) quantised P = 0.5668 d to +-0.2 %."""
    rng = np.random.default_rng(20)
    period = 0.566788
    t = np.arange(0.0, 26.0, 10 / 1440)
    phase = 2 * np.pi * t / period
    y = 1.0 + 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase) + rng.normal(0, 0.002, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.002))
    grid_step_period = period**2 * (res.max_frequency - res.min_frequency) / (res.n_frequencies - 1)
    assert abs(res.best_period_days - period) < 0.05 * grid_step_period
    assert res.period_error_days is not None and abs(res.best_period_days - period) < 5 * res.period_error_days + 1e-6


def test_frequency_grid_refuses_undersampled_grids() -> None:
    with pytest.raises(ValueError, match="samples per peak"):
        td.frequency_grid(2700.0, min_period_days=0.0004)
    freq, used = td.frequency_grid(2700.0, min_period_days=0.005)
    assert td.MIN_OVERSAMPLING <= used < td.DEFAULT_OVERSAMPLING and len(freq) == td.MAX_FREQUENCIES
    with pytest.raises(ValueError):
        td.frequency_grid(100.0, min_period_days=float("inf"))


def test_reduced_oversampling_is_reported() -> None:
    rng = np.random.default_rng(21)
    t = np.sort(rng.uniform(0, 3000, 200))
    res = td.lomb_scargle_period(t, rng.normal(0, 0.01, t.size), np.full_like(t, 0.01), min_period_days=0.005)
    assert any("samples per peak" in w for w in res.warnings)


def test_search_periods_records_skipped_grids() -> None:
    rng = np.random.default_rng(22)
    t = np.sort(rng.uniform(0, 3000, 100))
    s = mag_series("ztf", "g", t, 16 + rng.normal(0, 0.01, t.size), 0.01)
    search = td.search_periods([s], min_period_days=0.0004)
    assert search.per_series == [] and any("period search skipped" in n for n in search.notes)


def test_annual_and_diurnal_side_lobes_are_aliases() -> None:
    year = 1 / 365.25
    assert td.diurnal_alias(td.SIDEREAL_DAY_FREQUENCY * 2 + year, 2700.0) == "2 x sidereal day + 1/yr"
    assert td.diurnal_alias(year, 2700.0) == "annual"
    assert td.diurnal_alias(1.0 - 2 * year, 2700.0) == "solar day - 2/yr"
    assert td.diurnal_alias(1 / 0.62974, 2700.0) is None


# ---------------------------------------------------------------------------
# Router / core validation
# ---------------------------------------------------------------------------


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(td.router)
    return app


@pytest.mark.parametrize("params", [
    {"ra": 10.0, "dec": 10.0, "min_period_days": 0.0004},
    {"ra": 10.0, "dec": 10.0, "min_period_days": "inf"},
    {"ra": 10.0, "dec": 10.0, "max_period_days": "inf"},
    {"name": "   "},
])
def test_router_rejects_before_any_upstream_call(params: dict[str, Any]) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params=params)
        assert not route.called
    assert response.status_code == 422, response.text


async def test_core_rejects_non_finite_period_bounds() -> None:
    with pytest.raises(ValueError):
        await td.get_lightcurves(10.0, 10.0, min_period_days=float("inf"))
    with pytest.raises(ValueError):
        await td.get_lightcurves(10.0, 10.0, min_period_days=0.001)


def test_router_resolver_outage_is_502_and_unknown_name_404() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(
            return_value=httpx.Response(500, text="Internal Server Error"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"name": "M 31"})
    assert response.status_code == 502 and "resolver unavailable" in response.json()["detail"]
    empty = ('<?xml version="1.0"?><Sesame><Target option="SNV"><name>nosuchthing123</name>'
             "<INFO>*** Nothing found ***</INFO></Target></Sesame>")
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(
            return_value=httpx.Response(200, text=empty, headers={"content-type": "text/xml"}))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"name": "nosuchthing123"})
    assert response.status_code == 404


@pytest.mark.parametrize("value", ["inf", "1e9", "-1"])
def test_router_solar_system_rejects_bad_position_error_filter(value: str) -> None:
    with TestClient(make_app()) as http:
        response = http.get("/api/v1/solar-system", params={"ra": 10.0, "dec": 10.0, "epoch_mjd": 60000,
                                                             "max_position_error_arcsec": value})
    assert response.status_code == 422


async def test_core_rejects_infinite_position_error_filter() -> None:
    with pytest.raises(td.InvalidCoordinateError):
        await td.solar_system_objects(10.0, 10.0, epoch_mjd=60000.0, max_position_error_arcsec=float("inf"))


# ---------------------------------------------------------------------------
# Upstream handling: TAP transient faults
# ---------------------------------------------------------------------------

TRANSIENT = (b'<?xml version="1.0"?><VOTABLE><RESOURCE type="results"><INFO name="QUERY_STATUS" value="ERROR">'
             b"TransientFault: INTERNAL_SERVER_ERROR: ORA-24459: OCISessionGet() timed out waiting for pool to "
             b"create new connections</INFO></RESOURCE></VOTABLE>")


async def test_tap_transient_fault_in_http_200_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "RETRY_PAUSE_SECONDS", 0.0)
    calls = {"n": 0}

    def side_effect(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, content=TRANSIENT, headers={"content-type": "text/xml"})
        return httpx.Response(200, text="ra,dec,mjd\n", headers={"content-type": "text/csv"})

    with respx.mock(assert_all_mocked=True) as router:
        router.post(td.IRSA_TAP_SYNC_URL).mock(side_effect=side_effect)
        async with httpx.AsyncClient() as c:
            result = await td.fetch_neowise(c, 10.0, 10.0, 3.0)
    assert calls["n"] == 2 and result.series == []


async def test_persistent_tap_transient_fault_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "RETRY_PAUSE_SECONDS", 0.0)
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(td.IRSA_TAP_SYNC_URL).mock(
            return_value=httpx.Response(200, content=TRANSIENT, headers={"content-type": "text/xml"}))
        async with httpx.AsyncClient() as c:
            with pytest.raises(td.UpstreamServiceError) as info:
                await td.fetch_neowise(c, 10.0, 10.0, 3.0)
    assert route.call_count == 2 and info.value.retryable is True


# ---------------------------------------------------------------------------
# NEOWISE: visit acceptance and neighbour exclusion
# ---------------------------------------------------------------------------

NEO_HEADER = ["ra", "dec", "mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro", "qual_frame", "qi_fact", "saa_sep",
              "moon_masked", "cc_flags", "nb", "na", "w1rchi2", "w2rchi2"]


def neo_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=NEO_HEADER)
    writer.writeheader()
    for row in rows:
        base = {"ra": 10.0, "dec": 0.0, "w1sigmpro": 0.03, "w2mpro": 12.5, "w2sigmpro": 0.05, "qual_frame": 10,
                "qi_fact": 1.0, "saa_sep": 50, "moon_masked": "00", "cc_flags": "0000", "nb": 1, "na": 0,
                "w1rchi2": 1.0, "w2rchi2": 1.0}
        base.update(row)
        writer.writerow(base)
    return buf.getvalue()


def test_neowise_drops_visits_where_most_exposures_fail() -> None:
    """Review: persistence-flagged visits left 1-4 biased survivors that made a standard star 'variable'."""
    rows = []
    for v, start in enumerate((57000.0, 57180.0, 57360.0, 57540.0)):
        contaminated = v % 2 == 1
        for i in range(12):
            flagged = contaminated and i >= 2  # 2 of 12 survive the cc_flags cut
            rows.append({"mjd": start + 0.066 * i, "w1mpro": 12.80 if contaminated else 13.06,
                         "cc_flags": "PP00" if flagged else "0000"})
    result = td.parse_neowise_csv(neo_csv(rows), 10.0, 0.0)
    w1 = next(s for s in result.series if s.band == "W1")
    assert len(w1.points) == 2 and all(abs(p.value - 13.06) < 1e-9 for p in w1.points)
    assert w1.metadata["n_visits_rejected"] == 2
    assert any("visit(s) dropped" in n for n in result.notes)


def test_neowise_never_adopts_a_neighbour_in_a_wide_cone() -> None:
    """Review: in frames without the target, the nearest detection 3-59 arcsec away was adopted."""
    rows = []
    for i in range(10):
        mjd = 57000.0 + 0.066 * i
        rows.append({"mjd": mjd, "w1mpro": 13.9})
        rows.append({"mjd": mjd, "w1mpro": 15.5, "dec": 20.0 / 3600})  # neighbour 20" north, same frame
    for i in range(3):  # frames where only the neighbour was detected
        rows.append({"mjd": 57001.0 + 0.066 * i, "w1mpro": 15.5, "dec": 20.0 / 3600})
    result = td.parse_neowise_csv(neo_csv(rows), 10.0, 0.0, binned=False)
    w1 = next(s for s in result.series if s.band == "W1")
    assert len(w1.points) == 10 and all(p.value == 13.9 for p in w1.points)
    assert any("3 frame(s)" in n for n in result.notes)


def test_neowise_matches_high_proper_motion_target_per_epoch() -> None:
    target = validate_target(10.0, 0.0, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=1000.0)  # 1 "/yr north
    rows = []
    for i, mjd in enumerate((57000.0, 57000.1, 57000.2, 60000.0, 60000.1, 60000.2)):
        jy = td.jyear_from_mjd(mjd)
        rows.append({"mjd": mjd, "w1mpro": 13.0, "dec": (jy - 2000.0) / 3600})
    static = td.parse_neowise_csv(neo_csv(rows), 10.0, 0.0, binned=False)
    moving = td.parse_neowise_csv(neo_csv(rows), 10.0, 0.0, binned=False, target=target)
    assert static.series == []  # 15-24" from the J2000 position: nothing matched
    assert len(next(s for s in moving.series if s.band == "W1").points) == 6


def test_bin_visits_uses_intra_visit_scatter_when_it_exceeds_formal_error() -> None:
    t = [100.0, 100.1, 100.2, 100.3]
    y = [10.0, 10.2, 9.8, 10.0]
    dy = [0.01] * 4
    (b,) = td.bin_visits(t, y, dy)
    formal = 0.01 / 2
    sem = float(np.std(y, ddof=1) / 2)
    assert sem > 10 * formal and b.error == pytest.approx(sem)


# ---------------------------------------------------------------------------
# ZTF: anchor selection and per-field zero points
# ---------------------------------------------------------------------------


def test_ztf_identity_is_a_match_tolerance_not_the_nearest_object() -> None:
    """Review (round 2): the nearest object with data was adopted at any distance in the cone
    (RR Lyr, saturated in ZTF, got a 17.9-mag neighbour 7.9" away as its light curve)."""
    rows = [
        {"oid": "1", "ra": f"{269.452077 + 4.15 / 3600:.7f}", "dec": "4.693365", "filtercode": "zg", "ngoodobs": "0"},
        {"oid": "2", "ra": "269.452077", "dec": f"{4.693365 + 9.5 / 3600:.7f}", "filtercode": "zg", "ngoodobs": "209"},
        {"oid": "3", "ra": "269.452077", "dec": f"{4.693365 + 9.5 / 3600:.7f}", "filtercode": "zr", "ngoodobs": "150"},
    ]
    same, others = td.select_ztf_objects(rows, 269.452077, 4.693365)
    assert same == [] and [r["oid"] for r in others] == ["1", "2", "3"]  # nothing within 1.5": no target
    # A 4.15" offset in the RA *coordinate* is 4.15" x cos(dec) on the sky.
    assert float(others[0]["_sep"]) == pytest.approx(4.15 * math.cos(math.radians(4.693365)), abs=0.002)
    # The target's own object (0.2") without good epochs is still the target; a neighbour with data is not.
    rows = [{"oid": "7", "ra": "269.452077", "dec": f"{4.693365 + 0.2 / 3600:.7f}", "filtercode": "zg", "ngoodobs": "0"},
            {"oid": "8", "ra": "269.452077", "dec": f"{4.693365 + 25 / 3600:.7f}", "filtercode": "zg", "ngoodobs": "400"}]
    same, others = td.select_ztf_objects(rows, 269.452077, 4.693365)
    assert [r["oid"] for r in same] == ["7"] and [r["oid"] for r in others] == ["8"]


async def test_ztf_saturated_target_returns_no_series_and_names_neighbours() -> None:
    objects = ("oid,ra,dec,filtercode,field,ccdid,qid,ngoodobs\n"
               f"7,269.452077,{4.693365 + 0.2 / 3600:.7f},zg,1,1,1,0\n"
               f"8,269.452077,{4.693365 + 25 / 3600:.7f},zg,1,1,1,400\n")
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        tap = router.post(td.IRSA_TAP_SYNC_URL).mock(return_value=httpx.Response(200, text=objects))
        lc = router.get(td.ZTF_LIGHTCURVE_URL).mock(side_effect=AssertionError("no light curve of a neighbour"))
        async with httpx.AsyncClient() as c:
            result = await td.fetch_ztf(c, 269.452077, 4.693365, 30.0, collection="ztf_dr24")
    assert tap.called and not lc.called
    assert result.series == []
    assert any("no good epochs" in n and "7 (g)" in n for n in result.notes)
    assert any("8 (g, 25.0\"" in n for n in result.notes)
    assert result.provenance["target_object_ids"] == ["7"]


def test_ztf_track_association_for_high_proper_motion_star() -> None:
    """Review (round 2): CM Dra (1.6 "/yr) lost its own g object (5.45" from the J2021.6 cone
    centre, but on its J2018.2 track position) as a 'neighbour'."""
    target = validate_target(248.58470944, 57.1623247, epoch=2000.0, pm_ra_masyr=-1113.797, pm_dec_masyr=1180.977)
    ra18, dec18 = td.position_at(target, 2018.2)
    ra23, dec23 = td.position_at(target, 2023.0)
    ra_mid, dec_mid, _r, _n = td.survey_cone(target, "ztf", 3.0)
    rows = [{"oid": "g2018", "ra": f"{ra18:.7f}", "dec": f"{dec18:.7f}", "filtercode": "zg", "ngoodobs": "407"},
            {"oid": "r2023", "ra": f"{ra23:.7f}", "dec": f"{dec23:.7f}", "filtercode": "zr", "ngoodobs": "300"},
            {"oid": "star", "ra": f"{ra_mid:.7f}", "dec": f"{dec_mid + 6 / 3600:.7f}", "filtercode": "zg",
             "ngoodobs": "500"}]
    assert td.haversine_arcsec(ra_mid, dec_mid, ra18, dec18) > 5.0  # far from the mid-survey cone centre
    same, others = td.select_ztf_objects(rows, target.ra, target.dec, target=target)
    assert {r["oid"] for r in same} == {"g2018", "r2023"} and [r["oid"] for r in others] == ["star"]


def test_ztf_field_offsets_align_constant_star_but_not_sparse_variable() -> None:
    rng = np.random.default_rng(23)
    t1 = np.sort(rng.uniform(58300, 60900, 400))
    t2 = np.sort(rng.uniform(58300, 60900, 120))
    pts = [(t, 15.40 + e, "a") for t, e in zip(t1, rng.normal(0, 0.012, t1.size))]
    pts += [(t, 15.376 + e, "b") for t, e in zip(t2, rng.normal(0, 0.012, t2.size))]
    offsets = td.ztf_field_offsets(pts)
    assert offsets["b"] == pytest.approx(0.024, abs=0.004)
    # A 1-mag-amplitude variable: a 25-epoch oid's median is phase noise, not calibration.
    t3 = np.sort(rng.uniform(58300, 60900, 25))
    var = [(t, 15.4 + 0.5 * np.sin(2 * np.pi * t / 0.39), "a") for t in t1]
    var += [(t, 15.4 + 0.5 * np.sin(2 * np.pi * t / 0.39), "c") for t in t3]
    assert td.ztf_field_offsets(var) == {}


# ---------------------------------------------------------------------------
# TESS: one star only; O(n) binning
# ---------------------------------------------------------------------------


def test_tess_observation_selection_keeps_one_tic() -> None:
    def obs(sector: int, tic: int, ra: float, dec: float) -> dict[str, Any]:
        return {"obs_id": f"tess2024249191853-s{sector:04d}-{tic:016d}-0280-s", "obsid": sector * 1000 + tic % 997,
                "s_ra": ra, "s_dec": dec}

    a, b = (316.72475, 38.74942), (316.72475, 38.74942 + 30.7 / 3600)
    data = [obs(83, 165602000, *a), obs(83, 165602023, *b), obs(82, 165602000, *a), obs(82, 165602023, *b),
            obs(56, 165602000, *a), obs(15, 165602000, *a), {"obs_id": "hlsp_qlp_tess_ffi", "s_ra": a[0], "s_dec": a[1]}]
    tic, chosen, others = td.select_tess_observations(data, *a, max_sectors=3)
    assert tic == "165602000" and [s for s, _ in chosen] == [83, 82, 56]
    assert others == {"165602023": pytest.approx(30.7, abs=0.05)}
    tic_b, chosen_b, _ = td.select_tess_observations(data, *b, max_sectors=10)
    assert tic_b == "165602023" and [s for s, _ in chosen_b] == [83, 82]


def test_bin_uniform_is_linear_time_and_correct() -> None:
    n = 180_000  # 10 sectors of 2-min cadence
    t = np.arange(n) * (2 / 1440)
    rng = np.random.default_rng(24)
    y = 1 + rng.normal(0, 0.001, n)
    # Linear scaling is asserted without a wall-clock limit in
    # test_timedomain_review.py::test_bin_uniform_scales_linearly (limits were flaky on loaded machines).
    pts = td.bin_uniform(t, y, np.full(n, 0.001), 10 / 1440)
    assert len(pts) == n // 5 and all(p.n == 5 for p in pts[:100])
    assert pts[0].value == pytest.approx(float(np.mean(y[:5])))
    assert pts[0].error == pytest.approx(0.001 / math.sqrt(5))


# ---------------------------------------------------------------------------
# SkyBoT parser: vectorised
# ---------------------------------------------------------------------------


def test_skybot_parser_is_vectorised_and_consistent() -> None:
    payload = []
    for i in range(3000):
        ra_h = 1.0 + i * 1e-5
        h = int(ra_h)
        m = int((ra_h - h) * 60)
        sec = ((ra_h - h) * 60 - m) * 60
        payload.append({"Num": str(i + 1), "Name": f"A{i}", "RA (hms)": f"{h:02d} {m:02d} {sec:07.4f}",
                        "DEC (dms)": "+05 00 00.00", "Class": "MB>Middle", "VMag (mag)": "18.0"})
    # Vectorisation (one position conversion for all rows) is asserted without a wall-clock limit in
    # test_timedomain_review.py::test_skybot_parser_converts_positions_in_one_call.
    objects = td.parse_skybot_json(payload, 15.0, 5.0)
    assert len(objects) == 3000 and objects[0].name == "A0" and objects[0].separation_arcsec < 0.01
    assert objects[-1].ra == pytest.approx((1.0 + 2999e-5) * 15.0, abs=1e-7)
