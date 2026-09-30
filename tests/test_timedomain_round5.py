"""Offline regression tests for the round-5 adversarial review of timedomain.py.

One section per review finding (numbered as in the review). Recorded upstream answers
(``tests/fixtures/timedomain``, recorded with ``tests/test_timedomain_live.py record``) where
the defect needs real data; simulations on real survey timestamps where it is statistical.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from functools import cache
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient
from test_timedomain import client, load_case, make_app, make_parser, replay, replay_handler, run_case, within
from test_timedomain_live import (
    CASES,
    CYG61_A,
    DSCT_PERIOD_DAYS,
    GAIA_CEPHEIDS,
    GAIA_EB_PARITY,
    GAIA_EW_ID,
    GAIA_EW_PERIOD_DAYS,
    MIRA_206_PERIOD_DAYS,
    MIRA_320_PERIOD_DAYS,
    RRAB,
    TOI_519_PERIOD_DAYS,
    TOI_5205_PERIOD_DAYS,
    case_cm_dra_ztf,
)
from test_timedomain_review import damped_random_walk, eclipsing_binary, mag_series

import timedomain as td
from models import validate_target


def _gaia_case(prefix: str, source_id: str) -> td.LightCurveResult:
    return run_case(f"{prefix}_{source_id}", CASES[f"{prefix}_{source_id}"])


@cache
def _gaia_g_times(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Real Gaia DR3 G epochs and errors of a recorded source."""
    result = run_case(case, CASES[case])
    t, _y, dy = next(s for s in result.series if s.key == "gaia:G").good_arrays()
    return t, dy


# ---------------------------------------------------------------------------
# 1. Sparse (Gaia) eclipsing binaries keep their orbital period: the secondary eclipse counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", list(GAIA_EB_PARITY))
def test_sparse_gaia_eclipsing_binaries_are_not_doubled_past_their_orbital_period(source_id: str) -> None:
    """Review: all 5 'parity+eclipse_shape' results among 42 random Gaia DR3 EBs were exactly 2x the VSX period
    (e.g. 1974933547345251968: 1.408904 d vs 0.704444 d), reliable: 10-bin profile differences (0.04-0.05 mag)
    were taken as the noise, hiding 0.09-mag secondary eclipses."""
    truth = GAIA_EB_PARITY[source_id][2]
    result = _gaia_case("gaia_eb", source_id)
    assert result.provenance["gaia"]["source_id"] == source_id
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G"
    assert within(best.best_period_days, truth, 1e-4), (best.best_period_days, truth)
    assert "eclipse_shape" not in str(best.harmonic_test["doubled_by"])
    shape = best.harmonic_test["shape"]
    assert shape["n_dips"] >= 2 and shape["eclipse_like"] is False


def test_asassn_v_j2159_secondary_eclipse_is_detected_epoch_by_epoch() -> None:
    """Gaia DR3 1974933547345251968 folded at its 0.7044-d period: 9 epochs ~0.09 mag low half a cycle from the
    primary, errors ~3 mmag -- a secondary eclipse, so the fold is two-dipped and the period is not doubled."""
    _ra, _dec, period = GAIA_EB_PARITY["1974933547345251968"]
    result = _gaia_case("gaia_eb", "1974933547345251968")
    s = next(s for s in result.series if s.key == "gaia:G")
    (_, t, y, sigma), = td._usable_arrays([s])
    shape = td.eclipse_shape(t, y, sigma, 1.0 / period)
    assert shape["n_dips"] == 2 and shape["eclipse_like"] is False
    # The noise of a bin median is that of the epochs about the profile (~mmag), not the 10-bin
    # profile's first differences (0.036 mag in the review), so the binned profile shows the secondary;
    # the epoch-level search finds it too.
    assert shape["bin_noise"] < 0.01
    phase = ((t - t.min()) / period) % 1.0
    prof = td._phase_profile(phase, y, 10)
    level = float(np.median(prof))
    secondary = td._secondary_eclipse(phase, y, sigma, prof, level=level, depth=float(prof.max() - level))
    assert secondary is not None and secondary["detected"] is True
    assert abs(secondary["phase"] - 0.5) <= td.SECONDARY_WINDOW and secondary["depth"] > 0.05
    assert secondary["sigma"] >= td.SECONDARY_SIGMA


def test_synthetic_algol_on_real_gaia_times_keeps_its_period() -> None:
    """Review: an EA with a 0.35-mag primary, a 0.09-mag secondary and 4 mmag noise on the real Gaia times came
    out at 2P or 4P in 11 of 20 runs."""
    t, _dy = _gaia_g_times("gaia_eb_1974933547345251968")
    period = GAIA_EB_PARITY["1974933547345251968"][2]
    rng = np.random.default_rng(12)
    outcomes = []
    for _ in range(20):
        shift = rng.uniform(0.0, period)
        y = eclipsing_binary(t + shift, period, depth1=0.35, depth2=0.09, width=0.03) + rng.normal(0, 0.004, t.size)
        s = mag_series("gaia", "G", t, y, 0.004)
        search = td.search_periods([s], variability={s.key: td.series_variability(s)})
        outcomes.append(None if search.best is None else search.best.best_period_days / period)
    assert all(r is not None and abs(r - 1.0) < 1e-3 for r in outcomes), outcomes


def test_equal_minima_binaries_are_still_doubled() -> None:
    """The fix must not stop genuine equal-minima doubling: no secondary half a cycle away at P_orb/2."""
    rng = np.random.default_rng(40)
    t = np.sort(rng.uniform(0, 1500, 900))
    y = eclipsing_binary(t, 1.5, depth1=0.5, depth2=0.5, width=0.03) + rng.normal(0, 0.01, t.size)
    res = td.lomb_scargle_period(t, y, np.full_like(t, 0.01))
    assert res is not None and res.harmonic_test["doubled_by"] == "eclipse_shape"
    assert within(res.best_period_days, 1.5, 1e-4)
    assert res.harmonic_test["shape"]["secondary"]["detected"] is False


# ---------------------------------------------------------------------------
# 2. Red noise aliased by the ground-based window gets no period
# ---------------------------------------------------------------------------


@cache
def _mrk817() -> td.LightCurveResult:
    return run_case("mrk817_ztf", CASES["mrk817_ztf"])


def test_mrk817_stochastic_seyfert_gets_no_ztf_period() -> None:
    """Review: get_lightcurves_by_name('Mrk 817', surveys='ztf') adopted ztf:g P = 277.27 d, reliable, FAP 2e-5:
    the fitted harmonics left holes in the residual periodogram that pulled the continuum down (543x -> ~950x)."""
    result = _mrk817()
    assert result.period_search is not None and result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "ztf:g")
    assert not g.reliable and g.false_alarm_probability >= td.PERIOD_FAP_THRESHOLD
    assert g.noise_continuum > 800


def test_red_noise_continuum_is_not_pulled_down_by_the_fitted_harmonics() -> None:
    result = _mrk817()
    s = next(s for s in result.series if s.key == "ztf:g")
    (_, t, y, sigma), = td._usable_arrays([s])
    kw = {"min_frequency": 1.0 / float(t.max() - t.min()), "max_frequency": 20.0}
    one = td.red_noise_fap(t, y, sigma, 1 / 277.27, 1, **kw)["local_continuum"]
    four = td.red_noise_fap(t, y, sigma, 1 / 277.27, 4, **kw)["local_continuum"]
    assert four > 0.6 * one, (one, four)  # 533 vs 1296 with the holes in the fit (review)


def _drw_on_mrk817_times(seed: int, tau: float) -> list[td.LightCurveSeries]:
    """A damped random walk (sigma 0.15 mag) sampled at the real ZTF g/r epochs of Mrk 817, the same walk in
    both bands (AGN variability is correlated across bands), plus the calibrated per-epoch noise."""
    rng = np.random.default_rng(seed)
    bands = {s.band: s for s in _mrk817().series if s.band in ("g", "r")}
    t_g, _y, e_g = bands["g"].good_arrays()
    t_r, _y, e_r = bands["r"].good_arrays()
    t_all = np.concatenate([t_g, t_r])
    order = np.argsort(t_all)
    walk = np.empty_like(t_all)
    walk[order] = damped_random_walk(rng, t_all[order], tau=tau, sigma=0.15)
    out = []
    for band, t, e, w in (("g", t_g, e_g, walk[:len(t_g)]), ("r", t_r, e_r, walk[len(t_g):])):
        scale, floor = td.ERROR_MODELS[("ztf", band)]
        y = 15.0 + w + rng.normal(0.0, np.sqrt((scale * e) ** 2 + floor**2))
        out.append(mag_series("ztf", band, t, y, e))
    return out


def test_damped_random_walks_on_real_ztf_times_get_no_diurnal_alias_period() -> None:
    """Review: 170 DRWs (tau 50-1000 d) on the real ZTF g/r times of Mrk 817 gave 9 adopted periods, seven of them
    ~1 d with FAPs 2.6e-9 to 2.8e-15 (e.g. 0.999228 d, 2.1/T from 1 c/d), one eclipse-doubled to 1.9967 d. Red
    noise aliased by the 1-c/d window is now part of the noise continuum. 60 walks: none near the daily
    comb of window frequencies, and at most one adoption at all (the nominal 1 % false-alarm level of two
    series)."""
    centres = [c for c in td.window_alias_centres("ztf", ground_based=True) if c > 0.5]  # the daily comb
    adopted = []
    for k in range(60):
        series = _drw_on_mrk817_times(3000 + k, (50.0, 100.0, 200.0, 500.0, 1000.0)[k % 5])
        search = td.search_periods(series, variability={s.key: td.series_variability(s) for s in series},
                                   min_period_days=0.2)
        if search.best is not None:
            adopted.append((k, search.best.series, search.best.best_period_days))
    near_window = [a for a in adopted
                   if any(abs(j / a[2] - c) <= td.RED_NOISE_ALIAS_MAX_OFFSET for c in centres for j in (1, 2))]
    assert near_window == [], near_window
    assert len(adopted) <= 1, adopted


def test_diurnal_red_noise_alias_is_assessed_against_the_aliased_continuum() -> None:
    """Seed 3000 (tau 50 d): the r band peaks at 0.99831 d, 4.6/T from 1 c/d, outside the +-2/T diurnal flag; its
    FAP was 1.5e-5 against the local continuum. The 1-c/d window carries the walk's low-frequency power there."""
    series = _drw_on_mrk817_times(3000, 50.0)
    r = next(s for s in series if s.band == "r")
    (_, t, y, sigma), = td._usable_arrays([r])
    f = 1 / 0.9983128
    kw = {"min_frequency": 1.0 / float(t.max() - t.min()), "max_frequency": 20.0}
    plain = td.red_noise_fap(t, y, sigma, f, 3, **kw)
    aliased = td.red_noise_fap(t, y, sigma, f, 3, alias_centres=td.window_alias_centres("ztf", ground_based=True), **kw)
    assert plain["alias"] is None and aliased["alias"] is not None
    assert aliased["continuum"] > 5 * plain["continuum"]
    assert plain["fap"] < 1e-12 and aliased["fap"] > 1e9 * plain["fap"]
    # End to end the peak lies in the aliased-red-noise band: not adopted.
    search = td.search_periods(series, variability={s.key: td.series_variability(s) for s in series},
                               min_period_days=0.2)
    result = next(p for p in search.per_series if p.series == "ztf:r")
    assert not result.reliable and result.harmonic_test["aliased_red_noise"] is not None


def test_single_dip_at_a_diurnal_alias_is_not_doubled_from_the_ground() -> None:
    """Review: 'never let the eclipse-shape rule double a peak at 1 c/d' (a DRW's 0.99838-d alias became a
    1.9967-d 'eclipsing binary with equal minima'). From space the same fold is doubled."""
    rng = np.random.default_rng(21)
    nights = np.sort(rng.choice(np.arange(2600), 900, replace=False)).astype(float)
    t = 58200.0 + nights + rng.uniform(0.1, 0.5, nights.size)  # nightly epochs
    y = eclipsing_binary(t, 0.5, depth1=0.4, depth2=0.4, width=0.03) + rng.normal(0, 0.01, t.size)
    ground = td.lomb_scargle_period(t, y, np.full(t.size, 0.01), ground_based=True, survey="ztf", min_period_days=0.2)
    space = td.lomb_scargle_period(t, y, np.full(t.size, 0.01), min_period_days=0.2)
    # The Lomb-Scargle peak (a harmonic of the narrow dips) lies at n c/d, a diurnal sampling alias.
    assert td.diurnal_alias(1.0 / ground.lomb_scargle_period_days, ground.baseline_days) is not None
    assert ground is not None and space is not None
    assert "eclipse_shape" not in str(ground.harmonic_test["doubled_by"]) and not ground.reliable
    assert "sampling alias" in ground.harmonic_test["shape"].get("not_doubled", "")
    assert space.harmonic_test["doubled_by"] == "eclipse_shape"


# ---------------------------------------------------------------------------
# 3. ZTF moving targets: every object on the track at its own epochs is the target
# ---------------------------------------------------------------------------


def test_gj3622_keeps_its_later_ztf_object() -> None:
    """Review: GJ 3622 r kept 367206400029887 but dropped 367206400017938, which holds the star's 75 good 2023-2025
    epochs (0.24" from the track; 29 % of the r data)."""
    result = run_case("gj3622_ztf", CASES["gj3622_ztf"])
    r = next(s for s in result.series if s.key == "ztf:r")
    assert set(r.source_ids) == {"367206400029887", "367206400017938"}
    good = [p for p in r.points if p.flag == 0]
    late = [p for p in good if p.mjd > 60200.0]
    assert len(good) >= 250 and len(late) >= 70
    assert not any("one star is one object per reference image" in n for n in result.notes)
    assert "367206400017938" in result.provenance["ztf"]["target_object_ids"]


def _ztf_rows(target: Any, mjd: np.ndarray, oid: str, rng: np.random.Generator, *, field: int = 500) -> list[dict]:
    years = 2000.0 + (mjd - 51544.5) / 365.25
    tra, tdec = td.track_positions(target, years)
    rows = []
    for m, a, d in zip(mjd, tra, tdec):
        jitter = rng.normal(0, 0.1 / 3600.0, 2)
        rows.append({"oid": oid, "mjd": m, "mag": 15.0 + rng.normal(0, 0.01), "magerr": 0.01, "catflags": 0,
                     "filtercode": "zg", "ra": a + jitter[0] / math.cos(math.radians(d)), "dec": d + jitter[1],
                     "chi": 0.8, "sharp": 0.0, "field": field, "ccdid": "0x3", "qid": "0x1", "exptime": 30})
    return rows


def _csv(rows: list[dict]) -> str:
    cols = ["oid", "mjd", "mag", "magerr", "catflags", "filtercode", "ra", "dec", "chi", "sharp", "field", "ccdid",
            "qid", "exptime"]
    return ",".join(cols) + "\n" + "".join(",".join(str(r[c]) for c in cols) + "\n" for r in rows)


def test_moving_target_objects_with_disjoint_epochs_in_one_image_are_all_kept() -> None:
    target = validate_target(150.0, 20.0, epoch=2000.0, pm_ra_masyr=3000.0, pm_dec_masyr=-4000.0)
    rng = np.random.default_rng(7)
    early = np.sort(rng.uniform(58300.0, 59500.0, 90))
    late = np.sort(rng.uniform(59600.0, 60600.0, 40))  # a new object ID further along the track
    rows = _ztf_rows(target, early, "first", rng) + _ztf_rows(target, late, "later", rng)
    tra, tdec = td.track_positions(target, np.array([2021.0]))
    result = td.parse_ztf_csv(_csv(rows), float(tra[0]), float(tdec[0]), oids=["first", "later"], target=target)
    (g,) = result.series
    assert g.source_ids == ["first", "later"] and len(g.points) == 130


def test_an_exposure_measured_by_two_target_objects_is_used_once() -> None:
    target = validate_target(150.0, 20.0, epoch=2000.0, pm_ra_masyr=3000.0, pm_dec_masyr=-4000.0)
    rng = np.random.default_rng(8)
    mjd = np.sort(rng.uniform(58300.0, 60600.0, 60))
    rows = _ztf_rows(target, mjd, "main", rng) + _ztf_rows(target, mjd[::4], "twin", rng)
    tra, tdec = td.track_positions(target, np.array([2021.0]))
    result = td.parse_ztf_csv(_csv(rows), float(tra[0]), float(tdec[0]), oids=["main", "twin"], target=target)
    (g,) = result.series
    assert len(g.points) == 60 and g.source_ids == ["main"]
    assert any("15 exposure(s) measured by two object IDs" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 4. Peaks at annual / Gaia scanning-law frequencies: strong, coherent signals are kept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case,truth,window", [("mira_320_ztf", MIRA_320_PERIOD_DAYS, "annual"),
                                               ("mira_206_ztf", MIRA_206_PERIOD_DAYS, "semi-annual")])
def test_ztf_miras_near_the_annual_window_are_adopted(case: str, truth: float, window: str) -> None:
    """Review: Miras ZTFJ192918.06+221823.1 (Gaia LPV 325 d; ZTF 321.5/322.5 d, FAP ~0) and ZTFJ180131.33-073403.9
    (Gaia 206 d; ZTF 200.6 d, 4-5 mag) were rejected as the 'annual'/'semi-annual sampling alias'."""
    result = run_case(case, CASES[case])
    best = result.period_search.best
    assert best is not None and best.reliable
    assert within(best.best_period_days, truth, 0.03), best.best_period_days  # Gaia LPV periods: +-13.6 / 1.9 d
    assert any(f"{window} sampling alias" in w and "real signal" in w for w in best.warnings)


def test_small_annual_signal_stays_a_sampling_alias() -> None:
    """A coherent, significant 1-yr pattern of a few 10 mmag is what seasonal calibration systematics look like:
    still vetoed (semi-amplitude < WINDOW_ALIAS_MIN_AMPLITUDE_MAG); a 0.5-mag one is kept."""
    rng = np.random.default_rng(3)
    nights = np.arange(0.0, 2700.0, 2.0)
    nights = nights[(nights % 365.25) < 240]
    t = 58200.0 + nights + rng.uniform(0.15, 0.45, nights.size)
    for amp, kept in ((0.02, False), (0.5, True)):
        y = 15.0 + amp * np.sin(2 * np.pi * t / td.DAYS_PER_JULIAN_YEAR) + rng.normal(0, 0.01, t.size)
        res = td.lomb_scargle_period(t, y, np.full(t.size, 0.01), ground_based=True, survey="ztf",
                                     min_period_days=2.0)
        assert res is not None and within(res.best_period_days, td.DAYS_PER_JULIAN_YEAR, 0.02), res.best_period_days
        assert res.reliable is kept, (amp, res.warnings)
        assert any("annual sampling alias" in w for w in res.warnings)


def test_gaia_cepheid_at_a_precession_harmonic_is_adopted() -> None:
    """Review: the 12.315-d Gaia DR3 T2CEP 4041710426226283776 (pf 12.3122 d) was flagged at 5 x the 63-d
    precession frequency whatever its strength."""
    source_id = "4041710426226283776"
    best = _gaia_case("gaia_cep", source_id).period_search.best
    assert best is not None and within(best.best_period_days, GAIA_CEPHEIDS[source_id][2], 1e-3)
    assert any("scanning-law" in w and "real signal" in w for w in best.warnings)


# ---------------------------------------------------------------------------
# 5. Gaia-only strictly periodic pulsators: template uncertainty, sparse effective N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_id", ["2049531563002338432", "4655167172163949184", "4661366184375995520",
                                       "2686850313960864256"])
def test_gaia_only_cepheids_are_adopted(source_id: str) -> None:
    """Review: 7 of 20 random Gaia DR3 Cepheids got no period although it matched pf in 6 (FAPs down to 1e-32):
    'typical of red noise' from phase_coherence, or FAP = 1 whenever N_eff < 2K + 2."""
    truth = GAIA_CEPHEIDS[source_id][2]
    result = _gaia_case("gaia_cep", source_id)
    assert result.provenance["gaia"]["source_id"] == source_id
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G" and within(best.best_period_days, truth, 1e-4)
    assert best.false_alarm_probability < td.UNCONFIRMED_PERIOD_FAP


def test_sparse_cepheid_with_few_effective_points_uses_the_continuum() -> None:
    """2686850313960864256 (15.57 d, 37 transits): the K-harmonic misfit, identical within a visit, left N_eff 4-6;
    the F test then had < 1 residual degree of freedom and FAP was 1. The continuum FAP is used, and the signal
    must repeat in both halves at >= 5 sigma."""
    best = _gaia_case("gaia_cep", "2686850313960864256").period_search.best
    assert best is not None and best.harmonic_test["continuum_only"] is True
    assert best.effective_n < 2 * (2 * best.n_harmonics + 1)
    coherence = best.harmonic_test["phase_coherence"]
    assert coherence["coherent"] is True and min(coherence["amplitude_sigma_halves"]) >= td.CONTINUUM_ONLY_HALF_SIGMA


def test_gaia_contact_binary_with_correlated_misfit_is_adopted() -> None:
    """Review: Gaia EW 5690870910317297408 (0.3209272 d, correct) had FAP = 1 (rank FAP 3e-17)."""
    result = run_case("gaia_ew_5690870910317297408", CASES["gaia_ew_5690870910317297408"])
    assert result.provenance["gaia"]["source_id"] == GAIA_EW_ID
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, GAIA_EW_PERIOD_DAYS, 1e-4)


def _sawtooth(phase: np.ndarray) -> np.ndarray:
    phase = phase % 1.0
    return np.where(phase < 0.15, phase / 0.15, 1.0 - (phase - 0.15) / 0.85)


def test_strictly_periodic_sawtooth_on_real_gaia_cepheid_times_is_adopted() -> None:
    """Review: a strictly periodic 0.8-mag saw-tooth with 3 mmag noise on the real Gaia times of 4 Cepheids was
    declared incoherent in 27 % of the runs (up to 8 of 12 for one sampling). End to end, >= 90 % are adopted
    now (the rest: an alias peak chosen by the periodogram, or halves below 5 sigma with a continuum-only FAP)."""
    rng = np.random.default_rng(5)
    adopted = total = 0
    for source_id in ("2049531563002338432", "4655167172163949184", "4661366184375995520", "2686850313960864256"):
        t, _dy = _gaia_g_times(f"gaia_cep_{source_id}")
        period = GAIA_CEPHEIDS[source_id][2]
        for _ in range(12):
            y = 15.0 - 0.8 * _sawtooth(t / period + rng.uniform()) + rng.normal(0, 0.003, t.size)
            s = mag_series("gaia", "G", t, y, 0.003)
            search = td.search_periods([s], variability={s.key: td.series_variability(s)})
            total += 1
            adopted += int(search.best is not None and within(search.best.best_period_days, period, 1e-3))
    assert adopted >= 0.85 * total, (adopted, total)


def test_phase_coherence_propagates_the_template_uncertainty() -> None:
    """A sparse half's template is uncertain between its epochs: its Monte Carlo spread widens the comparison,
    so the saw-tooth's misfit is not read as a phase change (28 of 48 incoherent before, see the review)."""
    rng = np.random.default_rng(5)
    incoherent = 0
    for source_id in ("2049531563002338432", "4655167172163949184", "4661366184375995520", "2686850313960864256"):
        t, _dy = _gaia_g_times(f"gaia_cep_{source_id}")
        period = GAIA_CEPHEIDS[source_id][2]
        for _ in range(12):
            y = 15.0 - 0.8 * _sawtooth(t / period + rng.uniform()) + rng.normal(0, 0.003, t.size)
            s = np.full(t.size, 0.003)
            k = td.best_n_harmonics(t, y, s, 1 / period, max_harmonics=td.MAX_HARMONICS)
            coherence = td.phase_coherence(t, y, s, 1 / period, n_harmonics=k)
            incoherent += int(coherence is not None and not coherence["coherent"])
    assert incoherent <= 16, incoherent


def test_damped_random_walks_on_real_gaia_times_rarely_get_a_period() -> None:
    """The template-uncertainty and continuum fallbacks must not open the door to red noise: DRWs sampled like the
    recorded Gaia sources are adopted no more often than before this round (4 of 72 for these seeds: FAPs 1e-4 to
    1e-3 near the Gaia spin frequencies), and never with a continuum-only FAP."""
    cases = [f"gaia_eb_{k}" for k in GAIA_EB_PARITY] + [f"gaia_cep_{k}" for k in GAIA_CEPHEIDS]
    rng = np.random.default_rng(7)
    adopted = []
    for _rep in range(4):
        for case in cases:
            t, _dy = _gaia_g_times(case)
            tau = float(rng.choice([5.0, 20.0, 50.0, 100.0, 300.0]))
            y = 17.0 + damped_random_walk(rng, t, tau=tau, sigma=0.25) + rng.normal(0, 0.005, t.size)
            s = mag_series("gaia", "G", t, y, 0.005)
            search = td.search_periods([s], variability={s.key: td.series_variability(s)})
            if search.best is not None:
                adopted.append((case, tau, search.best.best_period_days, search.best.harmonic_test["continuum_only"]))
    assert len(adopted) <= 3, adopted  # 40 walks
    assert not any(a[3] for a in adopted), adopted


# ---------------------------------------------------------------------------
# 6. Giant planets transiting M dwarfs: a single deep dip is not doubled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case,truth", [("toi519_tess", TOI_519_PERIOD_DAYS), ("toi5205_tess", TOI_5205_PERIOD_DAYS)])
def test_giant_planets_transiting_m_dwarfs_keep_their_period(case: str, truth: float) -> None:
    """Review: TOI-519 (10.3 % deep) came out at 2.5304661 d and TOI-5205 (6.8 %) at 3.261263 d, both reliable."""
    result = run_case(case, CASES[case])
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.metadata["tic_radius_rsun"] < 0.6
    best = result.period_search.best
    assert best is not None and abs(best.best_period_days - truth) < max(5 * best.period_error_days, 1e-4 * truth)
    shape = best.harmonic_test["shape"]
    assert "eclipse_shape" not in str(best.harmonic_test["doubled_by"]) and shape["transit_like"] is True
    assert shape["depth_relative_flux"] >= td.ECLIPSE_MIN_DOUBLING_DEPTH  # deeper than the old 5 % limit
    assert best.alternative_period_days == pytest.approx(2 * truth, rel=1e-3)


def test_planet_depth_limit_follows_the_host_radius() -> None:
    assert td.planet_max_depth(None) is None
    assert td.planet_max_depth(1.0) == pytest.approx((2 * 0.10276) ** 2, rel=1e-6)  # 4.2 % for a Sun
    assert td.planet_max_depth(0.36) == td.PLANET_MAX_DEPTH  # capped at 25 %


def test_deep_single_dip_on_a_sunlike_host_is_still_an_equal_minima_binary() -> None:
    rng = np.random.default_rng(9)
    t = np.sort(np.concatenate([60000 + np.arange(0, 27, 10 / 1440), 60700 + np.arange(0, 27, 10 / 1440)]))
    flux = 10 ** (-0.4 * (eclipsing_binary(t, 1.2, depth1=0.35, depth2=0.35, width=0.02) - 15.0))
    y = flux + rng.normal(0, 5e-4, t.size)
    s = mag_series("tess", "TESS", t, y, 5e-4, unit="flux", tic_radius_rsun=1.0)
    search = td.search_periods([s], variability={s.key: td.series_variability(s)})
    best = search.best
    assert best is not None and within(best.best_period_days, 1.2, 1e-3)
    assert best.harmonic_test["doubled_by"] == "eclipse_shape"


# ---------------------------------------------------------------------------
# 7. The resolver's parallax reaches the ZTF per-epoch tolerance (and the cache key)
# ---------------------------------------------------------------------------


def test_resolver_parallax_widens_the_ztf_tolerance() -> None:
    """Review: target.parallax_mas was always None, so the tolerance stayed 1.5" (Teegarden's Star, GJ 3622 lost
    epochs lying 1.5-1.8" from the track)."""
    result = run_case("gj3622_ztf", CASES["gj3622_ztf"])
    assert result.target["parallax_mas"] == pytest.approx(219.3302, abs=0.1)
    assert any("more than 1.7\" from the target's position" in n for n in result.notes)


async def test_cache_key_includes_the_parallax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "60")
    monkeypatch.setattr(td, "_cache", td.CacheManager(None))
    keys: list[str] = []
    original = td._cache_get

    def spy(cache_: Any, key: str) -> Any:
        keys.append(key)
        return original(cache_, key)

    monkeypatch.setattr(td, "_cache_get", spy)
    with respx.mock(assert_all_mocked=True) as router:
        router.route().mock(return_value=httpx.Response(503))
        async with client() as c:
            for plx in (None, 200.0):
                target = validate_target(10.0, 10.0, epoch=2000.0, pm_ra_masyr=1000.0, pm_dec_masyr=0.0,
                                         parallax_mas=plx)
                with pytest.raises(td.UpstreamServiceError):
                    await td.fetch_survey(c, "neowise", 10.0, 10.0, 3.0, target=target)
    assert len(keys) == 2 and keys[0] != keys[1]


def test_router_passes_the_resolver_parallax() -> None:
    with replay("gj3622_ztf"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"name": "GJ 3622", "surveys": "ztf", "radius_arcsec": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"]["parallax_mas"] == pytest.approx(219.3302, abs=0.1)
    assert "367206400017938" in body["provenance"]["ztf"]["target_object_ids"]


# ---------------------------------------------------------------------------
# 8. Low-amplitude periodic variables with chi^2/dof < 1 are upgraded on their period
# ---------------------------------------------------------------------------


def test_delta_scuti_with_chi2_below_one_is_variable_and_periodic() -> None:
    """Review: ZTFJ210655.21+462600.4 (P = 0.0743561 d) was found at FAP 2e-27 in g (confirmed in r), but chi^2/dof
    0.83 / 0.59 kept every series 'not variable' and no period was adopted."""
    result = run_case("dsct_ztf", CASES["dsct_ztf"])
    g = result.variability["ztf:g"]
    assert g.chi2_dof < 1.0 and g.is_variable is True and g.decision_basis == "periodic"
    best = result.period_search.best
    assert best is not None and best.series == "ztf:g" and within(best.best_period_days, DSCT_PERIOD_DAYS, 1e-4)


# ---------------------------------------------------------------------------
# 9. A Redis outage costs one timeout, not a global lock held across I/O
# ---------------------------------------------------------------------------


class _SlowRedis:
    """Redis stand-in whose every call blocks for ``delay`` seconds and then fails (an unreachable server)."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0
        self.threads: set[str] = set()
        self.lock = threading.Lock()

    def _fail(self, *_args: Any) -> None:
        with self.lock:
            self.calls += 1
            self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        raise ConnectionError("Redis unreachable")

    get = setex = delete = _fail


async def test_redis_outage_does_not_serialise_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: with REDIS_URL set and Redis down, every cache call took 2 s under one global lock: 8 concurrent
    requests took 67-98 s each, and an unrelated asyncio.to_thread call waited 10 s."""
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "600")
    cache = td._TimeDomainCache(None)
    cache._redis = _SlowRedis(1.0)
    monkeypatch.setattr(td, "_cache", cache)

    async def one() -> td.LightCurveResult:
        async with client() as c:
            return await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=c,
                                            period=False)

    async def unrelated() -> float:
        await asyncio.sleep(0.2)
        started = time.perf_counter()
        await asyncio.to_thread(lambda: None)
        return time.perf_counter() - started

    with replay("rrab_css_j132708"):
        started = time.perf_counter()
        results = await asyncio.gather(*(one() for _ in range(6)), unrelated())
        elapsed = time.perf_counter() - started
    lightcurves, wait = results[:-1], results[-1]
    assert all(r.failures == [] and len(r.series) >= 5 for r in lightcurves)
    # Cache I/O runs on its own executor: the default one (10 s wait in the review) stays free.
    assert all(name.startswith("timedomain-cache") for name in cache._redis.threads), cache._redis.threads
    assert wait < 3.0, wait
    assert elapsed < 20.0, elapsed  # ~100 s with the lock held across Redis I/O
    # Circuit breaker: after the first failures Redis is skipped instead of costing each call 1 s.
    assert cache._redis.calls <= td.CACHE_IO_WORKERS + 2, cache._redis.calls


def test_local_cache_works_while_redis_is_down() -> None:
    cache = td._TimeDomainCache(None)
    cache._redis = _SlowRedis(0.05)
    cache.set("k", {"format": "x", "content": "{}"}, ttl=60)
    started = time.perf_counter()
    assert cache.get("k") == {"format": "x", "content": "{}"}  # local LRU first
    assert cache.get("other") is None
    assert cache.get("other2") is None
    # The failed setex opened the circuit breaker: the gets did not touch Redis.
    assert time.perf_counter() - started < 1.0 and cache._redis.calls == 1


# ---------------------------------------------------------------------------
# 10. Cached results carry a version; undecodable entries are misses
# ---------------------------------------------------------------------------


async def test_entries_of_an_older_cache_version_are_not_served(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: the key had no version, so a deploy kept serving the previous code's Barnard's star 'light curves'
    (background stars on its track) for the whole TTL."""
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "600")
    shared = td.CacheManager(None)  # e.g. Redis, shared by the deploys
    monkeypatch.setattr(td, "_cache", shared)
    ra, dec = RRAB
    monkeypatch.setattr(td, "TIMEDOMAIN_CACHE_VERSION", "previous-deploy")
    with replay("rrab_css_j132708"):
        async with client() as c:
            old = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
            assert (await td.fetch_survey(c, "gaia", ra, dec, 3.0)).provenance.get("cache") == "hit"
    assert old.provenance.get("cache") is None and shared.entry_count == 1
    monkeypatch.setattr(td, "TIMEDOMAIN_CACHE_VERSION", "this-deploy")
    with replay("rrab_css_j132708"):
        async with client() as c:
            new = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
    assert new.provenance.get("cache") is None and new.series  # fetched again, the old entry not served
    assert shared.entry_count == 2


@pytest.mark.parametrize("entry", [{"content": "{\"survey\": \"gaia\"}", "format": "timedomain.SurveyResult/json"},
                                   {"content": "not json", "format": "timedomain.SurveyResult/json"},
                                   {"content": "{}", "format": "some-other-format"}, ["not", "a", "dict"]])
async def test_undecodable_cache_entries_are_misses_and_deleted(monkeypatch: pytest.MonkeyPatch, entry: Any) -> None:
    """Review: a schema change made SurveyResult.from_dict raise KeyError inside fetch_survey, failing the survey
    as non-retryable for the TTL."""
    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "600")
    shared = td.CacheManager(None)
    monkeypatch.setattr(td, "_cache", shared)
    ra, dec = RRAB
    motion_key = td.CacheManager.make_key("timedomain", td.TIMEDOMAIN_CACHE_VERSION, "gaia", round(ra, 7),
                                          round(dec, 7), round(3.0, 4), False, None, 2, 10.0, True, None)
    shared.set(motion_key, entry, ttl=600)
    with replay("rrab_css_j132708"):
        async with client() as c:
            result = await td.fetch_survey(c, "gaia", ra, dec, 3.0)
    assert result.series and result.provenance.get("cache") is None
    cached = shared.get(motion_key)
    assert isinstance(cached, dict) and cached["format"] == td.CACHE_FORMAT  # replaced by the fresh result


# ---------------------------------------------------------------------------
# 11. TESS: a failed sector does not discard the others
# ---------------------------------------------------------------------------


async def _tess_61cyg_with(side_effect: Any, monkeypatch: pytest.MonkeyPatch, *, use_cache: bool = False) -> Any:
    monkeypatch.setattr(td, "RETRY_PAUSE_SECONDS", 0.0)
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.route().mock(side_effect=side_effect)
        async with client() as c:
            return await td.get_lightcurves(*CYG61_A, radius_arcsec=45.0, surveys="tess", client=c,
                                            tess_max_sectors=2, use_cache=use_cache, period=False)


async def test_one_failed_tess_sector_keeps_the_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: with the sector-82 download answering 503, sector 83 (downloaded fine) was discarded as well."""
    real = replay_handler("tess_61cyg")

    def side(request: httpx.Request) -> httpx.Response:
        if "Download/file" in request.url.path and "s0082" in str(request.url):
            return httpx.Response(503, text="Service Unavailable")
        return real(request)

    monkeypatch.setenv("TIMEDOMAIN_CACHE_TTL_SECONDS", "600")
    monkeypatch.setattr(td, "_cache", td.CacheManager(None))
    result = await _tess_61cyg_with(side, monkeypatch, use_cache=True)
    assert result.failures == []
    (tess,) = result.series
    assert tess.metadata["sectors"] == [83]
    provenance = result.provenance["tess"]
    assert provenance["partial"] is True and list(provenance["sectors_failed"]) == ["82"]
    assert any("sector(s) 82" in n and "partial result" in n for n in result.notes)
    assert td._cache.entry_count == 0  # a partial answer is not cached


async def test_all_tess_sectors_failing_is_a_survey_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    real = replay_handler("tess_61cyg")

    def side(request: httpx.Request) -> httpx.Response:
        if "Download/file" in request.url.path:
            return httpx.Response(503, text="Service Unavailable")
        return real(request)

    with pytest.raises(td.UpstreamServiceError) as info:
        await _tess_61cyg_with(side, monkeypatch)
    assert info.value.retryable


async def test_tess_sector_without_a_light_curve_product_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    real = replay_handler("tess_61cyg")
    exchanges = load_case("tess_61cyg")
    product_calls = [e for e in exchanges if "Mast.Caom.Products" in (e.get("request_body") or "")]
    assert len(product_calls) == 2

    def side(request: httpx.Request) -> httpx.Response:
        response = real(request)
        if b"Mast.Caom.Products" in (request.content or b""):
            payload = response.json()  # the products answer lists no _lc.fits
            payload["data"] = [p for p in payload["data"] if not str(p.get("productFilename")).endswith("_lc.fits")]
            return httpx.Response(200, json=payload)
        return response

    result = await _tess_61cyg_with(side, monkeypatch)
    assert result.series == [] and result.failures == []
    assert any("list no SPOC light-curve (_lc.fits) product" in n for n in result.notes)
    assert any("none of the selected sectors lists a light-curve product" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 12. TESS: the point budget holds on the actual data
# ---------------------------------------------------------------------------


def test_tess_point_budget_is_enforced_on_the_actual_cadences(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: sector 97 of tau Cet spans 54.6 d (28,889 good cadences) against the 19,728 estimated for one
    sector; three such unbinned sectors would return 1.5-2x the budget."""
    monkeypatch.setattr(td, "MAX_TESS_POINTS", 20_000)

    async def case(c: httpx.AsyncClient) -> td.LightCurveResult:
        return await td.get_lightcurves_by_name("tau Cet", client=c, radius_arcsec=3.0, surveys="tess",
                                                tess_max_sectors=1, tess_bin_minutes=0.0, use_cache=False,
                                                period=False)

    result = run_case("tau_cet_tess", case)
    (tess,) = result.series
    assert len(tess.points) <= 20_000
    assert tess.metadata["bin_minutes"] > 1.5 * td.TESS_CADENCE_MINUTES  # 2 cadences
    assert any("exceed the 20,000-point limit" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 13. The period analysis has its own bounded executor and a time budget
# ---------------------------------------------------------------------------


async def test_analysis_runs_on_its_own_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[str] = []
    original = td.search_periods

    def spy(*args: Any, **kwargs: Any) -> td.PeriodSearch:
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(td, "search_periods", spy)
    with replay("rrab_css_j132708"):
        async with client() as c:
            await td.get_lightcurves(*RRAB, surveys="gaia", client=c, use_cache=False)
    assert len(threads) == 1 and threads[0].startswith("timedomain-analysis")


def test_period_search_stops_at_its_time_budget() -> None:
    """Review: min_period_days = 0.01 and oversampling = 50 took 63 s for 6 series, unbounded."""
    rng = np.random.default_rng(4)
    series = [mag_series("ztf", band, t, 15 + 0.3 * np.sin(2 * np.pi * t / 0.63) + rng.normal(0, 0.02, t.size), 0.02)
              for band, t in (("g", np.sort(rng.uniform(58200, 60600, 600))),
                              ("r", np.sort(rng.uniform(58200, 60600, 600))),
                              ("i", np.sort(rng.uniform(58200, 60600, 300))))]
    started = time.perf_counter()
    search = td.search_periods(series, time_budget_s=0.0)
    assert time.perf_counter() - started < 30.0
    assert len(search.per_series) == 1  # the lead series only
    assert any("time budget" in n and "ztf:r" in n and "ztf:i" in n for n in search.notes)


# ---------------------------------------------------------------------------
# 14. The CLI passes epoch and proper motion with --ra/--dec
# ---------------------------------------------------------------------------


def test_cli_ra_dec_with_epoch_and_proper_motion_follows_a_fast_star(capsys: pytest.CaptureFixture[str]) -> None:
    """Review: 'lightcurve' had no --epoch/--pm-* options, so CM Dra at its J2000 position (1.6 "/yr) got no ZTF
    series from --ra/--dec; with the resolver's motion it has 409 g epochs."""
    exchange = next(e for e in load_case("cm_dra_ztf") if "Sesame" in e["url"] or "sesame" in e["url"])
    import re as _re

    text = exchange["content"].decode()
    ra = float(_re.search(r"<jradeg>([^<]+)</jradeg>", text).group(1))
    dec = float(_re.search(r"<jdedeg>([^<]+)</jdedeg>", text).group(1))
    pm_ra = float(_re.search(r"<pmRA>([^<]+)</pmRA>", text).group(1))
    pm_dec = float(_re.search(r"<pmDE>([^<]+)</pmDE>", text).group(1))
    args = make_parser().parse_args(["lightcurve", "--ra", repr(ra), "--dec", repr(dec), "--epoch", "2000",
                                     "--pm-ra", repr(pm_ra), "--pm-dec", repr(pm_dec), "--surveys", "ztf",
                                     "--format", "json"])
    with replay("cm_dra_ztf"):
        code = args.handler(args)
    out = capsys.readouterr().out
    assert code == 0
    body = __import__("json").loads(out)
    assert body["target"]["pm_ra_masyr"] == pytest.approx(pm_ra) and body["target"]["epoch"] == 2000.0
    g = next(s for s in body["series"] if s["band"] == "g")
    assert sum(1 for p in g["points"] if p["flag"] == 0) >= 400


@pytest.mark.parametrize("argv,needle", [(["--name", "CM Dra", "--epoch", "2016"], "epoch"),
                                         (["--ra", "10", "--dec", "10", "--pm-ra", "5", "--pm-dec", "5"], "--epoch")])
def test_cli_rejects_inconsistent_motion_options(argv: list[str], needle: str,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        route = router.route().mock(side_effect=AssertionError("no upstream call expected"))
        args = make_parser().parse_args(["lightcurve", *argv, "--surveys", "ztf"])
        assert args.handler(args) == 2
        assert not route.called
    assert needle in capsys.readouterr().err


def test_cli_by_name_uses_the_resolver_motion() -> None:
    result = run_case("cm_dra_ztf", case_cm_dra_ztf)
    assert result.target["epoch"] is not None and result.target["pm_ra_masyr"] is not None


# ---------------------------------------------------------------------------
# 15. Same-image duplicates are not described as lying beyond the tolerance
# ---------------------------------------------------------------------------


async def test_same_image_duplicate_gets_its_own_note() -> None:
    """Review: 'blend' 1.08" from the target in the target object's image was listed as a neighbour 'more than
    1.5\\" of the target'."""
    objects = ("oid,ra,dec,filtercode,field,ccdid,qid,ngoodobs\n"
               "near,10.0000000,20.0000000,zr,600,5,2,300\n"
               "blend,10.0000000,20.0003000,zr,600,5,2,280\n"
               "far,10.0000000,20.0020000,zr,600,5,2,100\n")
    lc = "oid,mjd,mag,magerr,catflags,filtercode,ra,dec,chi,sharp,field,ccdid,qid,exptime\n" + "".join(
        f"near,{59000 + i},15.0,0.01,0,zr,10.0,20.0,1.0,0.0,600,5,2,30\n" for i in range(30))
    with respx.mock(assert_all_mocked=True) as router:
        router.post(td.IRSA_TAP_SYNC_URL).mock(return_value=httpx.Response(200, text=objects))
        router.get(td.ZTF_LIGHTCURVE_URL).mock(return_value=httpx.Response(200, text=lc))
        async with httpx.AsyncClient() as c:
            result = await td.fetch_ztf(c, 10.0, 20.0, 10.0, collection="ztf_dr24")
    beyond = next(n for n in result.notes if "more than 1.5" in n)
    assert "far" in beyond and "blend" not in beyond
    same = next(n for n in result.notes if "one star is one object per reference image" in n)
    assert "blend" in same and "far" not in same
    assert result.provenance["same_image_object_ids"] == {"blend": "near"}
