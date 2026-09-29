"""Offline regression tests for the round-4 adversarial review of timedomain.py.

One section per review finding (numbered as in the review). Synthetic data where the
defect is a property of the algorithm; recorded upstream answers (``tests/fixtures/
timedomain``, recorded with ``tests/test_timedomain_live.py record``) where it needs real data.
"""

from __future__ import annotations

import asyncio
import io
import math
import time
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient
from test_timedomain import client, load_case, make_app, make_parser, replay, replay_handler, run_case, within
from test_timedomain_live import (
    BARNARD_NAME,
    GAIA_RRAB_EPSL_PERIOD_DAYS,
    GAIA_RRAB_ID,
    GAIA_RRAB_PERIOD_DAYS,
    GAIA_SPARSE_EB_ID,
    LPV_PERIOD_DAYS,
    RRAB,
    RRC_BLAZHKO_PERIOD_DAYS,
    V354_LYR_PERIOD_DAYS,
    WASP12_PERIOD_DAYS,
    case_barnard_ztf,
    case_ea2,
    case_ew,
    case_gaia_agn_coherent,
    case_gaia_rrab,
    case_gaia_rrab_epsl,
    case_gaia_sparse_eb,
    case_groombridge1830_ztf,
    case_lpv_ztf,
    case_pg1323_086b_ztf,
    case_rr_lyr_tess,
    case_rrc_blazhko_ztf,
    case_v354_lyr_ztf,
    case_wasp12_tess,
)
from test_timedomain_review import _on_event_loop, gaia_like_times, mag_series, ztf_like_times

import timedomain as td
from models import validate_target

HTML = "<!DOCTYPE html><html><head><title>Maintenance</title></head><body><h1>Service down</h1></body></html>"


def rrab_fixture_times() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Real ZTF g/r epochs and pipeline errors of CSS J132708.3+384442 (recorded fixture)."""
    exchange = next(e for e in load_case("rrab_css_j132708") if "nph_light_curves" in e["url"])
    result = td.parse_ztf_csv(exchange["content"].decode(), *RRAB)
    out = {}
    for s in result.series:
        if s.band in ("g", "r"):
            t, _y, dy = s.good_arrays()
            out[s.band] = (t, dy)
    return out


# ---------------------------------------------------------------------------
# 1. ZTF identity of moving targets: epoch-by-epoch, one object per reference image
# ---------------------------------------------------------------------------


def _ztf_csv(rows: list[dict[str, Any]]) -> str:
    cols = ["oid", "mjd", "mag", "magerr", "catflags", "filtercode", "ra", "dec", "chi", "sharp", "field", "ccdid",
            "qid", "exptime"]
    out = io.StringIO()
    out.write(",".join(cols) + "\n")
    for r in rows:
        out.write(",".join(str(r[c]) for c in cols) + "\n")
    return out.getvalue()


def test_barnard_star_ztf_track_crossings_are_not_the_target() -> None:
    """Review: Barnard's star (r ~ 8.5, saturated) got 'light curves' of r ~ 20.5 stars lying on its 80" track."""
    result = run_case("barnard_ztf", case_barnard_ztf)
    assert result.failures == []
    assert [s for s in result.series if s.survey == "ztf"] == []
    provenance = result.provenance["ztf"]
    assert {"485209300035121", "485209300052229"} <= set(provenance["candidate_object_ids"])
    assert provenance["target_object_ids"] == []
    note = next(n for n in result.notes if "are not the target" in n)
    assert "485209300035121" in note and "485209300052229" in note
    assert any("no ZTF light curve for the target" in n for n in result.notes)
    assert result.period_search is None or result.period_search.best is None


def test_groombridge_1830_ztf_has_no_neighbour_light_curve() -> None:
    """Review: Groombridge 1830 (V = 6.4) got the 15.8-mag r light curve of object 714201200026226, which its
    track crosses at J2020.08."""
    result = run_case("groombridge1830_ztf", case_groombridge1830_ztf)
    assert result.failures == [] and [s for s in result.series if s.survey == "ztf"] == []
    assert "714201200026226" in result.provenance["ztf"]["candidate_object_ids"]
    assert any("714201200026226" in n and "not the target" in n for n in result.notes)


def test_ztf_epochs_must_follow_the_moving_target_and_one_object_per_image() -> None:
    target = validate_target(150.0, 20.0, epoch=2000.0, pm_ra_masyr=3000.0, pm_dec_masyr=-4000.0)
    rng = np.random.default_rng(7)
    mjd = np.sort(rng.uniform(58300.0, 60600.0, 120))
    years = 2000.0 + (mjd - 51544.5) / 365.25
    tra, tdec = td.track_positions(target, years)
    cross_ra, cross_dec = td.position_at(target, 2021.0)  # a star the track crosses in 2021.0
    rows = []
    for i, (m, a, d) in enumerate(zip(mjd, tra, tdec)):
        common = {"catflags": 0, "filtercode": "zg", "chi": 0.8, "sharp": 0.0, "field": 500, "ccdid": "0x3",
                  "qid": "0x1", "exptime": 30}
        jitter = rng.normal(0, 0.1 / 3600.0, 2)
        rows.append({"oid": "own", "mjd": m, "mag": 15.0 + rng.normal(0, 0.01), "magerr": 0.01,
                     "ra": a + jitter[0] / math.cos(math.radians(d)), "dec": d + jitter[1], **common})
        rows.append({"oid": "crossed", "mjd": m, "mag": 18.0 + rng.normal(0, 0.05), "magerr": 0.05,
                     "ra": cross_ra, "dec": cross_dec, **common})
        if i % 3 == 0:  # a second object of the same image measuring the same exposures (round-5 review:
            # real twins hold disjoint later epochs and are kept, see test_timedomain_round5; an exposure
            # measured twice is used once)
            rows.append({"oid": "twin", "mjd": m, "mag": 15.0, "magerr": 0.01, "ra": a, "dec": d, **common})
    result = td.parse_ztf_csv(_ztf_csv(rows), float(tra.mean()), float(tdec.mean()), oids=["own", "crossed", "twin"],
                              target=target)
    (g,) = result.series
    assert g.source_ids == ["own"] and len(g.points) == 120
    assert abs(float(np.median([p.value for p in g.points])) - 15.0) < 0.01
    assert g.metadata["source_separation_arcsec"] < 0.3
    assert any("crossed" in n and "not the target" in n for n in result.notes)
    assert any("40 exposure(s) measured by two object IDs" in n for n in result.notes)


def test_stationary_target_keeps_the_nearest_object_of_each_reference_image() -> None:
    rows = [{"oid": "near", "ra": "10.0000000", "dec": "20.0000000", "filtercode": "zr", "field": "600",
             "ccdid": "5", "qid": "2", "ngoodobs": "300"},
            {"oid": "blend", "ra": "10.0000000", "dec": "20.0003000", "filtercode": "zr", "field": "600",
             "ccdid": "5", "qid": "2", "ngoodobs": "280"},  # 1.08" away, same image
            {"oid": "other_field", "ra": "10.0000000", "dec": "20.0001000", "filtercode": "zr", "field": "1600",
             "ccdid": "9", "qid": "1", "ngoodobs": "50"}]
    same, others = td.select_ztf_objects(rows, 10.0, 20.0)
    assert [r["oid"] for r in same] == ["near", "other_field"]
    assert [r["oid"] for r in others] == ["blend"]


def test_vectorised_track_matches_sampled_track() -> None:
    target = validate_target(269.45207696, 4.69336497, epoch=2000.0, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394)
    years = np.linspace(2017.9, 2025.6, 50)
    ras, decs = td.track_positions(target, years)
    for y, a, d in zip(years, ras, decs):
        ra, dec = td.position_at(target, float(y))
        assert td.haversine_arcsec(ra, dec, a, d) < 1e-6
    rng = np.random.default_rng(3)
    obj_ra = 269.4485 + rng.uniform(-40, 40, 200) / 3600.0
    obj_dec = 4.742 + rng.uniform(-60, 60, 200) / 3600.0
    fast = td.track_separations_arcsec(target, obj_ra, obj_dec, td.ZTF_POSITION_EPOCH_RANGE)
    fine = np.arange(2017.9, 2025.6 + 1e-9, 0.001)
    tra, tdec = td.track_positions(target, fine)
    brute = [float(np.min(td.haversine_arcsec_array(tra, tdec, a, d))) for a, d in zip(obj_ra, obj_dec)]
    assert np.allclose(fast, brute, atol=0.02)


# 10 (left over from issue 8): ZTF track matching must not block the event loop.
def test_ztf_track_selection_of_a_crowded_cone_is_fast() -> None:
    target = validate_target(300.0, 35.0, epoch=2000.0, pm_ra_masyr=300.0, pm_dec_masyr=150.0)
    rng = np.random.default_rng(11)
    rows = [{"oid": str(i), "ra": f"{300.0 + rng.uniform(-60, 60) / 3600 / math.cos(math.radians(35)):.7f}",
             "dec": f"{35.0 + rng.uniform(-60, 60) / 3600:.7f}", "filtercode": "zr", "field": "700", "ccdid": "1",
             "qid": str(i % 4 + 1), "ngoodobs": "100"} for i in range(3000)]
    started = time.perf_counter()
    same, others = td.select_ztf_objects(rows, target.ra, target.dec, target=target)
    elapsed = time.perf_counter() - started
    assert len(same) + len(others) == 3000
    assert elapsed < 1.0, elapsed  # the per-row Python track sampling took ~13 s for 3000 rows


async def test_ztf_object_selection_runs_in_a_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    on_loop: list[bool] = []
    original = td.select_ztf_objects

    def spy(*args: Any, **kwargs: Any) -> Any:
        on_loop.append(_on_event_loop())
        return original(*args, **kwargs)

    monkeypatch.setattr(td, "select_ztf_objects", spy)
    with replay("barnard_ztf"):
        async with client() as c:
            await td.get_lightcurves_by_name(BARNARD_NAME, client=c, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False, period=False)
    assert on_loop == [False]


# ---------------------------------------------------------------------------
# 2. Parity test: cluster-robust; modulated pulsators are not doubled
# ---------------------------------------------------------------------------


def test_v354_lyr_blazhko_rrab_period_is_not_doubled() -> None:
    """Review: V354 Lyr (Kepler Blazhko RRab, 0.5617 d) was adopted at 4.4934 d (8 x P), 'unequal minima'."""
    result = run_case("v354_lyr_ztf", case_v354_lyr_ztf)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, V354_LYR_PERIOD_DAYS, 1e-3), best.best_period_days
    assert best.harmonic_test["doubled"] is False
    assert not any("unequal minima" in w for w in best.warnings)
    for p in result.period_search.per_series:
        if p.reliable:
            assert within(p.best_period_days, V354_LYR_PERIOD_DAYS, 1e-3), (p.series, p.best_period_days)


def test_blazhko_rrc_period_is_not_doubled() -> None:
    result = run_case("rrc_blazhko_ztf", case_rrc_blazhko_ztf)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, RRC_BLAZHKO_PERIOD_DAYS, 1e-4)
    assert best.harmonic_test["doubled"] is False


def _blazhko_rrab(t: np.ndarray, period: float, *, amp_mod: float, phase_mod: float, blazhko_days: float) -> np.ndarray:
    ph = 2 * np.pi * t / period + phase_mod * np.sin(2 * np.pi * t / blazhko_days)
    a = 0.35 * (1 + amp_mod * np.sin(2 * np.pi * t / blazhko_days + 1.0))
    return 15.5 + a * (np.sin(ph) + 0.45 * np.sin(2 * ph + 0.6) + 0.25 * np.sin(3 * ph + 1.4) + 0.12 * np.sin(4 * ph + 2.3))


def test_parity_test_ignores_blazhko_modulation_on_real_ztf_sampling() -> None:
    """Review: 30 % / 0.3-rad Blazhko modulation at 60 d on real ZTF epochs doubled 10/10 realisations 8x; the
    old per-epoch ANOVA treated same-cycle epochs as independent."""
    times = rrab_fixture_times()
    period = 0.5617
    doubled = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        bands = []
        for band, (t, dy) in times.items():
            scale, floor = td.ERROR_MODELS[("ztf", band)]
            sigma = np.sqrt((scale * dy) ** 2 + floor**2)
            y = _blazhko_rrab(t, period, amp_mod=0.3, phase_mod=0.3, blazhko_days=60.0) + rng.normal(0, sigma)
            bands.append((t, y, sigma))
        result = td.parity_test(bands, 1.0 / period)
        doubled += int(result["sigma"] >= td.PARITY_SIGMA and result["effect"] >= td.PARITY_MIN_EFFECT)
    assert doubled == 0


def test_blazhko_rrab_search_keeps_the_pulsation_period() -> None:
    times = rrab_fixture_times()
    period = 0.5617
    for seed in range(2):
        rng = np.random.default_rng(100 + seed)
        series = []
        for band, (t, dy) in times.items():
            scale, floor = td.ERROR_MODELS[("ztf", band)]
            y = (_blazhko_rrab(t, period, amp_mod=0.3, phase_mod=0.3, blazhko_days=60.0)
                 + rng.normal(0, np.sqrt((scale * dy) ** 2 + floor**2)))
            series.append(mag_series("ztf", band, t, y, dy))
        metrics = {s.key: td.series_variability(s) for s in series}
        best = td.search_periods(series, variability=metrics).best
        assert best is not None and within(best.best_period_days, period, 1e-3), best and best.best_period_days
        assert best.harmonic_test["doubled"] is False


def test_parity_is_vetoed_by_strongly_negative_odd_harmonic_bic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A significant even/odd difference that the odd harmonics of f/2 reject (delta BIC < -10) is modulation,
    not unequal minima: no doubling."""
    rng = np.random.default_rng(5)
    t = ztf_like_times(rng, 900)
    y = 15.0 + 0.3 * np.sin(2 * np.pi * t / 0.4) + rng.normal(0, 0.01, t.size)
    s = mag_series("ztf", "r", t, y, 0.01)
    monkeypatch.setattr(td, "parity_test", lambda *_a, **_k: {"sigma": 9.0, "effect": 0.2, "n_bins": 40})
    result = td.search_periods([s], variability={s.key: td.series_variability(s)}).per_series[0]
    step = result.harmonic_test["ladder"][0]
    assert step["delta_bic"] < -td.HARMONIC_BIC_THRESHOLD and step["parity_vetoed_by_bic"] is True
    assert step["doubled"] is False and result.harmonic_test["doubled"] is False
    assert within(result.best_period_days, 0.4, 1e-4)


def test_eclipsing_binaries_are_still_doubled_by_the_ladder() -> None:
    for fixture, case, period in (("ew_ztfj0300", case_ew, 0.3858416),):
        best = run_case(fixture, case).period_search.best
        assert best is not None and within(best.best_period_days, period, 1e-3)
        assert best.harmonic_test["doubled_by"] == "parity"


# ---------------------------------------------------------------------------
# 3. Phase coherence: no double-counted error scale; halves fitted with the adopted harmonics
# ---------------------------------------------------------------------------


def test_long_period_variable_is_adopted() -> None:
    """Review: ZTFJ125600.42+164558.5 (LPV, Gaia DR3 155.5 d) had r-band FAP 3e-59 but was rejected as red noise:
    the error-bar scale was applied twice (halves at 0.96/0.78 sigma instead of ~15)."""
    result = run_case("lpv_ztf", case_lpv_ztf)
    best = result.period_search.best
    assert best is not None, [p.warnings for p in result.period_search.per_series]
    assert within(best.best_period_days, LPV_PERIOD_DAYS, 0.02) and best.reliable
    coherence = best.harmonic_test["phase_coherence"]
    assert coherence["coherent"] is True and min(coherence["amplitude_sigma_halves"]) > 5
    assert best.period_error_days is not None and best.period_error_days < 0.01 * best.best_period_days


def test_gaia_only_rrab_is_adopted() -> None:
    """Review: 4 of 7 Gaia DR3 vari_rrlyrae RRab (FAP down to 1e-86, period = pf to 1e-5) were rejected as
    'not the same in the two halves'."""
    result = run_case("gaia_rrab", case_gaia_rrab)
    assert result.provenance["gaia"]["source_id"] == GAIA_RRAB_ID
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G"
    assert within(best.best_period_days, GAIA_RRAB_PERIOD_DAYS, 2e-5)
    assert best.harmonic_test["phase_coherence"]["coherent"] is True


def test_gaia_rrab_with_a_phase_gap_in_one_half_is_coherent() -> None:
    """Half of the 86 transits lie in Gaia's first 6 days: that half's light curve, with the adopted 12 harmonics,
    is unconstrained across its 0.22 phase gap; the half templates are limited to 1/(2 x gap) harmonics."""
    best = run_case("gaia_rrab_epsl", case_gaia_rrab_epsl).period_search.best
    assert best is not None and within(best.best_period_days, GAIA_RRAB_EPSL_PERIOD_DAYS, 2e-5)
    assert best.harmonic_test["phase_coherence"]["coherent"] is True


def test_sparse_gaia_eclipsing_binary_gets_no_chance_period() -> None:
    """With the error scale no longer counted twice, the coherence test alone let chance frequencies aligning the
    few in-eclipse transits of sparse Gaia EBs through (0.0524 d for this 2.02-d binary, Gaussian FAP 6e-5): the
    outlier-robust rank FAP rejects them."""
    result = run_case("gaia_sparse_eb", case_gaia_sparse_eb)
    assert result.provenance["gaia"]["source_id"] == GAIA_SPARSE_EB_ID
    assert result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "gaia:G")
    assert not g.reliable and g.robust_false_alarm_probability >= td.PERIOD_FAP_THRESHOLD
    assert any("few extreme epochs" in w for w in g.warnings)


def test_unconfirmable_red_noise_peak_needs_a_stronger_fap() -> None:
    """The corrected coherence test (no double-counted error scale) passed the 6.68-d red-noise peak of the Gaia
    vari_agn source 6378923189372478592 (FAP 0.006, both halves agree): a period that no other band can confirm
    needs FAP < 1e-3."""
    result = run_case("gaia_agn_coherent", case_gaia_agn_coherent)
    assert result.provenance["gaia"]["source_id"] == "6378923189372478592"
    assert result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "gaia:G")
    assert not g.reliable and any("required without confirmation" in w for w in g.warnings)


def test_rank_fap_keeps_pulsations_and_discounts_outlier_driven_peaks() -> None:
    from astropy.timeseries import LombScargle

    kw = {"min_frequency": 1e-3, "max_frequency": 20.0}
    rng = np.random.default_rng(21)
    t = gaia_like_times(rng, n_visits=40)
    phase = 2 * np.pi * t / 0.37
    rrab = 16 + 0.3 * (np.sin(phase) + 0.45 * np.sin(2 * phase + 0.6) + 0.25 * np.sin(3 * phase + 1.4))
    rrab += rng.normal(0, 0.01, t.size)
    assert td.rank_fap(t, rrab, 1 / 0.37, 3, **kw) < 1e-10
    # Rank-based: a monotonic transform of the magnitudes (e.g. to flux) does not change it.
    assert td.rank_fap(t, 10 ** (-0.4 * rrab), 1 / 0.37, 3, **kw) == pytest.approx(td.rank_fap(t, rrab, 1 / 0.37, 3, **kw))
    # Constant star plus five random deep epochs: the best periodogram peak aligns the outliers; the
    # rank FAP gives it no significance, less than the Gaussian FAP does.
    for seed in (0, 2):
        rng = np.random.default_rng(seed)
        t = gaia_like_times(rng, n_visits=15)
        y = 16 + rng.normal(0, 0.01, t.size)
        idx = rng.choice(t.size, 5, replace=False)
        y[idx] += rng.uniform(0.2, 0.5, 5)
        grid = np.linspace(1e-3, 20.0, 400_000)
        f = float(grid[np.argmax(LombScargle(t, y, 0.01).power(grid))])
        _log_p, gaussian = td.multiharmonic_fap(t, y, np.full(t.size, 0.01), f, 1, **kw)
        robust = td.rank_fap(t, y, f, 1, **kw)
        assert robust > 0.9 and robust > gaussian


@pytest.mark.parametrize("error_scale", [1.0, 5.0])
def test_coherence_verdict_is_invariant_to_the_error_bar_scale(error_scale: float) -> None:
    """Review: a strict sinusoid in white noise at k x the stated errors (fixed S/N) was adopted 20/20 for k = 1-3
    but only 3/20 for k = 5."""
    adopted = 0
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        t = gaia_like_times(rng, n_visits=40)
        amp = 15.0 * error_scale * 0.01 / math.sqrt(t.size / 2)  # S/N fixed relative to the true noise
        y = 17.0 + amp * np.sin(2 * np.pi * t / 0.37 + 1.0) + rng.normal(0, error_scale * 0.01, t.size)
        s = mag_series("gaia", "G", t, y, 0.01)
        search = td.search_periods([s], variability={s.key: td.series_variability(s)})
        adopted += int(search.best is not None and within(search.best.best_period_days, 0.37, 1e-3))
    assert adopted == 6


def test_modulated_but_strong_signal_is_coherent() -> None:
    """RR Lyr in one TESS sector: a 39-d Blazhko cycle changes the amplitude between the halves, but a >1000-sigma
    signal changed by ~10 % is periodic, not red noise."""
    result = run_case("rr_lyr_tess", case_rr_lyr_tess)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, 0.566788, 0.01)
    coherence = best.harmonic_test["phase_coherence"]
    assert coherence["coherent"] is True and coherence["relative_change"] <= td.COHERENCE_MAX_RELATIVE_CHANGE


# ---------------------------------------------------------------------------
# 4. Transits are not doubled by the eclipse-shape rule; the CLI names the doubling reason
# ---------------------------------------------------------------------------


def test_wasp12_transit_period_is_not_doubled() -> None:
    """Review: WASP-12 b (P = 1.0914 d, 1.5 % deep in TESS) was adopted at 2.1829 d, 'unequal minima' in the CLI."""
    result = run_case("wasp12_tess", case_wasp12_tess)
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, WASP12_PERIOD_DAYS, 1e-4), best.best_period_days
    assert best.harmonic_test["doubled"] is False and best.harmonic_test["shape"]["transit_like"] is True
    assert best.alternative_period_days is not None and within(best.alternative_period_days, 2 * WASP12_PERIOD_DAYS, 1e-4)
    assert any("transit-like" in w for w in best.warnings)
    summary = td.format_lightcurve_summary(result)
    assert "unequal minima" not in summary and "transit-like single dip" in summary


def test_synthetic_shallow_transit_keeps_its_period_but_a_deep_equal_eclipse_is_doubled() -> None:
    rng = np.random.default_rng(5)
    t = 60000.0 + np.arange(0.0, 27.0, 10.0 / 1440.0)
    t = t[(t - 60000.0 < 13.2) | (t - 60000.0 > 14.2)]
    for depth, expected in ((0.01, 2.2), (0.3, 4.4)):
        phase = ((t - 60000.3) / 2.2 + 0.5) % 1.0 - 0.5
        y = 1.0 - depth * (np.abs(phase * 2.2) < 1.25 / 24.0) + rng.normal(0, 4e-4, t.size)
        s = mag_series("tess", "TESS", t, y, 4e-4, unit="flux")
        best = td.search_periods([s], variability={s.key: td.series_variability(s)}).best
        assert best is not None and within(best.best_period_days, expected, 1e-3), (depth, best.best_period_days)


def test_cli_prints_the_doubling_reason() -> None:
    eclipse = td.format_lightcurve_summary(run_case("ea_ztfj0048", case_ea2))
    assert "one narrow dip per cycle, eclipsing binary with equal minima assumed" in eclipse
    assert "unequal minima" not in eclipse.split("period note")[0]
    parity = td.format_lightcurve_summary(run_case("ew_ztfj0300", case_ew))
    assert "unequal minima (consecutive cycles differ)" in parity


# ---------------------------------------------------------------------------
# 5. Period errors account for correlated residuals
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, t: np.ndarray, tau: float, amp: float) -> np.ndarray:
    out = np.empty_like(t)
    value, last = rng.normal(0, amp), t[0]
    for i, ti in enumerate(t):
        rho = math.exp(-(ti - last) / tau)
        value = rho * value + math.sqrt(1 - rho**2) * rng.normal(0, amp)
        out[i], last = value, ti
    return out


def test_period_errors_are_calibrated_by_independent_halves() -> None:
    """Review: across 36 ZTF series the two halves' periods differed by a median 3.5 quoted sigma (Gaussian:
    0.67), 56 % beyond 3 sigma. Known-truth signals with white, red and Blazhko-modulated residuals must give
    two-halves z-scores of roughly unit scale."""
    t, dy = rrab_fixture_times()["r"]
    scale, floor = td.ERROR_MODELS[("ztf", "r")]
    sigma = np.sqrt((scale * dy) ** 2 + floor**2)
    period = 0.37

    def pulsator(tt: np.ndarray, amp_mod: float = 0.0, phase_mod: float = 0.0) -> np.ndarray:
        ph = 2 * np.pi * tt / period + phase_mod * np.sin(2 * np.pi * tt / 60.0)
        return 15.0 + 0.1 * (1 + amp_mod * np.sin(2 * np.pi * tt / 60.0 + 1.0)) * (np.sin(ph) + 0.3 * np.sin(2 * ph + 0.4))

    scenarios = {
        "white noise, errors 3x understated": lambda rng: pulsator(t) + rng.normal(0, 3 * sigma),
        "red noise (tau 30 d)": lambda rng: pulsator(t) + rng.normal(0, sigma) + _ar1(rng, t, 30.0, 0.03),
        "Blazhko 30 % / 0.3 rad / 60 d": lambda rng: pulsator(t, 0.3, 0.3) + rng.normal(0, sigma),
    }
    mid = t.size // 2
    for name, make in scenarios.items():
        zs = []
        for seed in range(6):
            y = make(np.random.default_rng(seed))
            halves = [td.lomb_scargle_period(t[sl], y[sl], dy[sl], survey="ztf", ground_based=True,
                                             min_period_days=0.9 * period, max_period_days=1.1 * period)
                      for sl in (slice(0, mid), slice(mid, None))]
            assert all(h is not None and h.period_error_days for h in halves), name
            zs.append(abs(halves[0].best_period_days - halves[1].best_period_days)
                      / math.hypot(halves[0].period_error_days, halves[1].period_error_days))
        assert np.median(zs) < 1.3 and max(zs) < 3.5, (name, np.round(zs, 2))


def test_block_frequency_error_matches_white_noise_bound_and_grows_with_red_noise() -> None:
    rng = np.random.default_rng(12)
    t = ztf_like_times(rng, 1500)
    f = 1 / 0.37
    white = 15 + 0.1 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.02, t.size)
    sigma = np.full(t.size, 0.02)
    _f, crb, _k = td.refine_period(t, white, sigma, f, 1e-4, n_harmonics=1)
    block = td.block_frequency_error(t, white, sigma, f, 1)
    assert block is not None and 0.6 < block / crb < 2.0
    # A slowly wandering phase (0.3 rad over 300 d): residuals correlated for months.
    wander = 15 + 0.1 * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * t / 300.0)) + rng.normal(0, 0.02, t.size)
    _f, crb_wander, _k = td.refine_period(t, wander, sigma, f, 1e-4, n_harmonics=1)
    block_wander = td.block_frequency_error(t, wander, sigma, f, 1)
    assert block_wander is not None and block_wander > 2.0 * crb_wander


def test_adopted_period_error_includes_the_correlated_noise_inflation() -> None:
    """V354 Lyr (Blazhko): the quoted error must exceed the white-noise Cramer-Rao bound of the same fit."""
    result = run_case("v354_lyr_ztf", case_v354_lyr_ztf)
    best = result.period_search.best
    assert best.error_inflation is not None and best.error_inflation > 1.0
    series = next(s for s in result.series if s.key == best.series)
    (_s, t, y, sigma), = td._usable_arrays([series])
    _f, white_bound, _k = td.refine_period(t, y, sigma, best.best_frequency_per_day, 1e-6,
                                           n_harmonics=best.n_harmonics)
    assert best.period_error_days > 2.0 * white_bound / best.best_frequency_per_day**2


# ---------------------------------------------------------------------------
# 6. PSF-fit outliers of catflags = 0 epochs
# ---------------------------------------------------------------------------


def test_landolt_standard_with_bad_psf_fits_is_not_variable() -> None:
    """Review: PG1323-086B (V = 13.41) was VARIABLE in ZTF r (chi2/dof 86) from seven 0.5-1.7-mag-faint catflags = 0
    epochs with PSF chi 2.3-9.1 and |sharp| up to 2.1."""
    result = run_case("pg1323_086b_ztf", case_pg1323_086b_ztf)
    for band in ("ztf:g", "ztf:r", "ztf:i"):
        m = result.variability[band]
        assert m.n > 50 and m.is_variable is False and m.chi2_dof < 2.0, (band, m.evidence)
    assert any("outlying PSF fit" in n for n in result.notes)
    assert result.summary_is_variable() is False


def test_psf_outliers_are_relative_to_the_object() -> None:
    rng = np.random.default_rng(4)
    chi = rng.lognormal(math.log(0.7), 0.3, 400)
    sharp = rng.normal(0.0, 0.03, 400)
    chi[[5, 6]] = [5.0, 9.0]
    sharp[[7, 8]] = [1.2, -0.9]
    assert set(np.flatnonzero(td.ztf_psf_outliers(chi, sharp))) == {5, 6, 7, 8}
    # A bright star (chi ~ 2.5 everywhere) and an extended source (sharp ~ 0.6) keep all their epochs.
    assert not td.ztf_psf_outliers(rng.lognormal(math.log(2.5), 0.3, 400), rng.normal(0.0, 0.03, 400)).any()
    assert not td.ztf_psf_outliers(rng.lognormal(math.log(0.7), 0.2, 400), 0.6 + rng.normal(0, 0.05, 400)).any()


def test_psf_flagged_epochs_are_returned_with_their_flag() -> None:
    rows = [{"oid": "a", "mjd": 59000.0 + i, "mag": 15.0 if i != 3 else 16.2, "magerr": 0.01, "catflags": 0,
             "filtercode": "zr", "ra": 10.0, "dec": 20.0, "chi": 0.7 if i != 3 else 6.0, "sharp": 0.01,
             "field": 1, "ccdid": 1, "qid": 1, "exptime": 30} for i in range(40)]
    text = _ztf_csv(rows)
    default = td.parse_ztf_csv(text, 10.0, 20.0)
    assert len(default.series[0].points) == 39 and default.series[0].n_rejected == 1
    flagged = td.parse_ztf_csv(text, 10.0, 20.0, include_flagged=True)
    assert [p.flag for p in flagged.series[0].points if p.flag] == [td.ZTF_PSF_FLAG]


# ---------------------------------------------------------------------------
# 7. Multi-band periodogram: decisive bands only; Gaia scanning-law check
# ---------------------------------------------------------------------------


def test_gaia_bp_rp_do_not_enter_the_multiband_sum(capsys: pytest.CaptureFixture[str]) -> None:
    """Review: BP (chi2_0 3.3e6 vs G 4.1e5) dominated the Gaia multi-band sum, printing 8.77 d for a 0.46-d RRab
    and the 3rd harmonic of the 6-h spin for another."""
    result = run_case("gaia_rrab", case_gaia_rrab)
    assert result.period_search.multiband is None
    assert "Multi-band peak" not in td.format_lightcurve_summary(result)
    assert td.multiband_period([s for s in result.series if s.survey == "gaia"]) is None


def test_multiband_flags_gaia_scanning_law_peaks() -> None:
    rng = np.random.default_rng(9)
    t = gaia_like_times(rng, n_visits=60)
    grams = []
    freq = np.linspace(1 / 500.0, 20.0, 60000)
    for key in ("gaia:G", "gaia:G2"):  # two decisive series with a signal at the 63.12-d precession period
        y = 17 + 0.05 * np.sin(2 * np.pi * t / 63.12) + rng.normal(0, 0.01, t.size)
        grams.append(td._periodogram(t, y, np.full(t.size, 0.01), freq, 5.0, series=key))
    multi = td._combine_multiband(grams, freq, 5.0, n_points=2 * t.size, baseline=float(t.max() - t.min()),
                                  ground_based=False, survey="gaia")
    assert multi is not None and any("scanning-law" in w for w in multi.warnings)
    bp = td._periodogram(t, 17 + rng.normal(0, 0.3, t.size), np.full(t.size, 0.01), freq, 5.0, series="gaia:BP")
    assert td._combine_multiband([grams[0], bp], freq, 5.0, n_points=2 * t.size, baseline=1000.0,
                                 ground_based=False, survey="gaia") is None


# ---------------------------------------------------------------------------
# 8. TESS: sub-cadence bins count as unbinned; bounded points and analysis cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sectors,bin_minutes", [(10, 1.0), (10, 0.01), (4, 0.0), (4, 2.0), (10, 5.0)])
async def test_core_api_bounds_the_number_of_tess_points(sectors: int, bin_minutes: float) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with pytest.raises(ValueError, match="tess_bin_minutes"):
            await td.get_lightcurves(10.0, 10.0, surveys="tess", tess_max_sectors=sectors, tess_bin_minutes=bin_minutes)
        assert not route.called


@pytest.mark.parametrize("sectors,bin_minutes", [(10, 10.0), (10, 6.7), (3, 0.0), (3, 1.0), (1, 0.0)])
def test_allowed_tess_options_stay_within_the_point_budget(sectors: int, bin_minutes: float) -> None:
    td._validate_tess_options(sectors, bin_minutes)
    assert td.tess_points_estimate(sectors, bin_minutes) <= td.MAX_TESS_POINTS


@pytest.mark.parametrize("params", [{"tess_max_sectors": 10, "tess_bin_minutes": 1},
                                    {"tess_max_sectors": 10, "tess_bin_minutes": 0.01}])
def test_router_rejects_sub_cadence_tess_bins_with_many_sectors(params: dict[str, Any]) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"ra": 10, "dec": 10, "surveys": "tess", **params})
        assert not route.called
    assert response.status_code == 422 and "tess_bin_minutes" in response.json()["detail"]


def test_cli_rejects_sub_cadence_tess_bins_with_many_sectors(capsys: pytest.CaptureFixture[str]) -> None:
    parser = make_parser()
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        for extra in (["--tess-sectors", "10", "--tess-bin-minutes", "1"],
                      ["--tess-sectors", "10", "--tess-bin-minutes", "0.01"]):
            args = parser.parse_args(["lightcurve", "--ra", "10", "--dec", "10", "--surveys", "tess", *extra])
            assert args.handler(args) == 2, extra
        assert not route.called
    assert "tess_bin_minutes" in capsys.readouterr().err


def test_period_search_of_three_unbinned_sectors_is_bounded() -> None:
    rng = np.random.default_rng(2)
    # Three consecutive unbinned 2-min sectors (59,184 cadences).
    t = np.concatenate([start + np.arange(0.0, 27.4, 2.0 / 1440.0) for start in (60000.0, 60027.4, 60054.8)])
    y = 1.0 + 0.001 * np.sin(2 * np.pi * t / 1.3) + rng.normal(0, 3e-4, t.size)
    s = mag_series("tess", "TESS", t, y, 3e-4, unit="flux")
    started = time.perf_counter()
    search = td.search_periods([s], variability={s.key: td.series_variability(s)})
    elapsed = time.perf_counter() - started
    assert any("period search on" in n and "bins" in n for n in search.notes), search.notes
    assert search.per_series[0].n <= td.PERIOD_SEARCH_MAX_POINTS
    assert search.best is not None and within(search.best.best_period_days, 1.3, 1e-3)
    assert elapsed < 30.0, elapsed  # 47 s before (search on all 59,000 cadences)
    assert len(s.points) == t.size  # the returned series itself is not binned


# ---------------------------------------------------------------------------
# 9. /lightcurves with a name: epoch and proper motion come with the resolver's coordinates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params,needle", [
    ({"epoch": 2016}, "epoch"),
    ({"epoch": 2016, "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394}, "epoch"),
    ({"pm_ra_masyr": -801.551}, "together"),
])
def test_router_rejects_motion_options_that_contradict_resolved_coordinates(params: dict[str, Any], needle: str) -> None:
    """Review: name + epoch=2016 kept Sesame's J2000 coordinates but labelled them J2016, dropping the proper
    motion: the Gaia cone was 166" off Barnard's star."""
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"name": BARNARD_NAME, "surveys": "gaia", **params})
        assert not route.called
    assert response.status_code == 422 and needle in response.json()["detail"]


def test_router_name_with_proper_motion_uses_the_resolver_epoch() -> None:
    """Review: name + pm_ra/pm_dec (no epoch) was a 422 'epoch required' although the resolver supplies it."""
    with replay("barnard_name"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"name": BARNARD_NAME, "surveys": "gaia", "radius_arcsec": 3,
                                                           "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"]["epoch"] == 2000.0 and body["target"]["pm_dec_masyr"] == pytest.approx(10362.394)
    assert body["provenance"]["gaia"]["source_id"] == "4472832130942575872"


async def test_core_by_name_rejects_an_epoch_before_any_request() -> None:
    """Review: get_lightcurves_by_name(..., epoch=2016) raised TypeError after the Sesame request."""
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        with pytest.raises(ValueError, match="epoch"):
            await td.get_lightcurves_by_name(BARNARD_NAME, epoch=2016.0, surveys="gaia")
        with pytest.raises(ValueError, match="together"):
            await td.get_lightcurves_by_name(BARNARD_NAME, pm_ra_masyr=5.0, surveys="gaia")
        assert not route.called


async def test_core_by_name_accepts_a_proper_motion_override() -> None:
    with replay("barnard_name"):
        async with client() as c:
            result = await td.get_lightcurves_by_name(BARNARD_NAME, client=c, radius_arcsec=3.0, surveys="gaia",
                                                      use_cache=False, pm_ra_masyr=-801.551, pm_dec_masyr=10362.394)
    assert result.target["epoch"] == 2000.0 and result.provenance["gaia"]["source_id"] == "4472832130942575872"


# ---------------------------------------------------------------------------
# 11. Unusable TESS/MAST HTTP-200 answers are retryable failures without raw exception text
# ---------------------------------------------------------------------------


def _real_fits() -> bytes:
    exchange = next(e for e in load_case("tau_cet_tess") if e["url"].startswith("https://stpubdata"))
    return exchange["content"]


@pytest.mark.parametrize("body,content_type,needle", [
    (HTML.encode(), "text/html", "not a FITS file"),
    (b"", "application/fits", "empty"),
    ("truncated", "application/fits", "truncated"),
    ("block-truncated", "application/fits", "truncated"),
], ids=["html", "empty", "truncated", "block-truncated"])
async def test_unusable_tess_download_is_a_retryable_failure(body: Any, content_type: str, needle: str) -> None:
    """Review: HTML / empty / truncated downloads gave non-retryable 'OSError: No SIMPLE card ...' or
    'TypeError: buffer is too small ...' failures."""
    real = replay_handler("tau_cet_tess")
    fits_bytes = _real_fits()
    if body == "truncated":
        content = fits_bytes[: len(fits_bytes) - 1000]
    elif body == "block-truncated":  # cut at a 2880-byte block boundary inside the table
        content = fits_bytes[: (len(fits_bytes) // 2 // td.FITS_BLOCK_BYTES) * td.FITS_BLOCK_BYTES]
    else:
        content = body

    def side_effect(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("stpubdata"):
            return httpx.Response(200, content=content, headers={"content-type": content_type})
        return real(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            with pytest.raises(td.UpstreamServiceError) as info:
                await td.get_lightcurves_by_name("tau Cet", client=c, radius_arcsec=3.0, surveys="tess",
                                                 tess_max_sectors=1, use_cache=False)
    assert info.value.retryable and needle in str(info.value)
    for raw in ("OSError", "TypeError", "SIMPLE card", "buffer is too small"):
        assert raw not in str(info.value)


async def test_non_json_mast_answer_is_retryable() -> None:
    real = replay_handler("tau_cet_tess")

    def side_effect(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mast.stsci.edu" and request.url.path.endswith("/invoke"):
            return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
        return real(request)

    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            with pytest.raises(td.UpstreamServiceError) as info:
                await td.get_lightcurves_by_name("tau Cet", client=c, radius_arcsec=3.0, surveys="tess",
                                                 tess_max_sectors=1, use_cache=False)
    assert info.value.retryable and "non-JSON" in str(info.value)


def test_fits_without_light_curve_columns_is_an_upstream_failure() -> None:
    from astropy.io import fits

    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([fits.Column("X", "D", array=np.zeros(3))])]
                 ).writeto(buffer)
    with pytest.raises(td.UpstreamServiceError) as info:
        td.parse_tess_lc_fits(buffer.getvalue())
    assert info.value.retryable and "lacks column" in info.value.message


# ---------------------------------------------------------------------------
# 12. Whole-call deadlines of SkyBoT, Horizons and the name resolver
# ---------------------------------------------------------------------------


async def _slow(_request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(10.0)
    return httpx.Response(200, json=[])


async def test_skybot_deadline_bounds_a_slow_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: TIMEDOMAIN_SKYBOT_DEADLINE_S was defined but never used (0.2 s setting, 2-s answer: success)."""
    monkeypatch.setattr(td, "SKYBOT_DEADLINE_S", 0.3)
    # (a request cancelled by the deadline is not recorded as a completed call)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.route().mock(side_effect=_slow)
        started = time.perf_counter()
        with pytest.raises(td.UpstreamServiceError) as info:
            await td.solar_system_objects(10.0, 10.0, epoch_mjd=60000.0)
    assert time.perf_counter() - started < 5.0
    assert info.value.service == "skybot" and info.value.retryable and "deadline" in info.value.message


def test_router_solar_system_deadline_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "SKYBOT_DEADLINE_S", 0.3)
    # (a request cancelled by the deadline is not recorded as a completed call)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.route().mock(side_effect=_slow)
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/solar-system", params={"ra": 10, "dec": 10, "epoch_mjd": 60000})
    assert response.status_code == 502 and "deadline" in response.json()["detail"]


async def test_horizons_deadline_bounds_a_slow_service() -> None:
    # (a request cancelled by the deadline is not recorded as a completed call)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.route().mock(side_effect=_slow)
        started = time.perf_counter()
        with pytest.raises(td.UpstreamServiceError) as info:
            await td.horizons_ephemeris("4;", 60000.0, deadline_s=0.3)
    assert time.perf_counter() - started < 5.0 and info.value.retryable and "deadline" in info.value.message


def test_router_name_resolution_deadline_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "RESOLVER_DEADLINE_S", 0.3)
    # (a request cancelled by the deadline is not recorded as a completed call)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.route().mock(side_effect=_slow)
        with TestClient(make_app()) as http:
            response = http.get("/api/v1/lightcurves", params={"name": "RR Lyr", "surveys": "gaia"})
    assert response.status_code == 502 and "deadline" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 13. Cache reads/writes (Redis included) run off the event loop; Redis has socket timeouts
# ---------------------------------------------------------------------------


async def test_cache_io_runs_in_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    class RecordingCache(td.CacheManager):
        def get(self, key: str) -> Any:
            calls.append(("get", _on_event_loop()))
            return super().get(key)

        def set(self, key: str, value: Any, ttl: int = 3600) -> None:
            calls.append(("set", _on_event_loop()))
            super().set(key, value, ttl=ttl)

    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "60")
    monkeypatch.setattr(td, "_cache", RecordingCache(None))
    ra, dec = RRAB
    with replay("rrab_css_j132708"):
        async with client() as c:
            first = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
            second = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
    assert second.provenance.get("cache") == "hit" and first.provenance.get("cache") is None
    assert [name for name, _ in calls] == ["get", "set", "get"]
    assert not any(on_loop for _, on_loop in calls), calls


def test_redis_client_has_socket_timeouts() -> None:
    pytest.importorskip("redis")
    cache = td._TimeDomainCache("redis://127.0.0.1:9/0")  # nothing listens on port 9
    assert cache._redis is not None
    kwargs = cache._redis.connection_pool.connection_kwargs
    assert kwargs["socket_connect_timeout"] == td.REDIS_SOCKET_TIMEOUT_S
    assert kwargs["socket_timeout"] == td.REDIS_SOCKET_TIMEOUT_S
    started = time.perf_counter()
    assert cache.get("missing") is None  # an unreachable Redis falls back to the local cache
    assert time.perf_counter() - started < 3 * td.REDIS_SOCKET_TIMEOUT_S + 1.0
