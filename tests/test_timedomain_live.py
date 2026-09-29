"""Live truth tests for timedomain.py (run with ``pytest -m live``) and the fixture recorder.

Targets and catalogued truths:

* CSS J132708.3+384442 = ZTF J132708.32+384442.3 = Gaia DR3 1476301553808173824,
  an RRab star with P = 0.62974 d (AAVSO VSX; Catalina/Drake et al. 2014),
  0.6297706 d (Chen et al. 2020, ApJS 249, 18, ZTF periodic variables) and
  pf = 0.62974 d (Gaia DR3 vari_rrlyrae, Clementini et al. 2023).
* RR Lyr (TIC 159717514; TIC position from MAST): P = 0.566788 d (Kolenberg et al.
  2011, MNRAS 411, 878, Kepler photometry).
* 3C 273: optically variable quasar (e.g. ZTF/ASAS-SN monitoring; Soldi et al. 2008).
* SDSS Stripe 82 standard star at (10.742019, +1.126138), r = 15.817 with rms 0.006
  mag over 11 SDSS epochs (Ivezic et al. 2007, AJ 134, 973, catalog J/AJ/134/973);
  Gaia DR3 2549372572635491072, not in VSX: a photometrically quiet star.
* (4) Vesta and (1) Ceres positions from JPL Horizons (live) must be matched by SkyBoT.
* Eclipsing binaries from Chen et al. (2020, ApJS 249, 18; J/ApJS/249/18/table2, checked
  live): ZTFJ014057.73+582241.5, type EA, P = 1.8083775 d (rAmp 1.075 mag), and
  ZTFJ030001.97+010340.8, type EW (W UMa), P = 0.3858416 d. Both have their
  single-harmonic Lomb-Scargle peak at P/2; the pipeline must report the orbital period.
* Gaia DR3 387837952011193088 (GAPS field, G = 14.02, phot_variable_flag NOT_AVAILABLE):
  a source with GAPS epoch photometry (Evans et al. 2023, A&A 674, A4) that was flagged
  variable (G chi^2/dof = 9.9) with the uncalibrated pipeline errors.
* SDSS Stripe 82 standard at (321.620483, -0.892541), r = 14.467, rchi2 0.1 over 10
  epochs (Ivezic et al. 2007), whose NEOWISE visits alternate with latent-image
  (persistence) contaminated ones: constant in all default surveys.
* 61 Cyg A (TIC 165602000; 61 Cyg B = TIC 165602023 lies 30.7" away): a 45" cone holds
  three SPOC 2-min targets; only the nearest may be used.
* Barnard's star by name: the Sesame J2000 position and proper motion (-801.551,
  10362.394 mas/yr) must be propagated to J2016.0 to find Gaia DR3 4472832130942575872.

Only network errors, timeouts and HTTP 5xx skip; wrong physics fails.

Recording fixtures for the offline suite (polite: one pass, a few minutes):

    .venv/Scripts/python.exe tests/test_timedomain_live.py record [case ...]
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timedomain as td

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "timedomain"

RRAB = (201.7846706228748, 38.74506826403494)  # Gaia DR3 1476301553808173824 (J2016.0; pm negligible here)
RRAB_PERIOD_DAYS = 0.62974
RR_LYR = (291.366301347666, 42.7843585093162)  # TIC 159717514
RR_LYR_PERIOD_DAYS = 0.566788
QSO_3C273 = (187.2779154, 2.0523883)  # SIMBAD J2000
QUIET_STAR = (10.742019, 1.126138)  # Ivezic et al. 2007 standard
VESTA_EPOCH_MJD = 60000.0  # 2023-02-25 00:00 UTC
CERES_EPOCH_MJD = 60400.0  # 2024-03-30 00:00 UTC
EA_STAR = (25.24055, 58.3782)  # ZTFJ014057.73+582241.5 (Chen et al. 2020)
EA_PERIOD_DAYS = 1.8083775
EW_STAR = (45.00823, 1.06134)  # ZTFJ030001.97+010340.8 (Chen et al. 2020)
EW_PERIOD_DAYS = 0.3858416
GAPS_QUIET = (9.572303162281901, 44.2737627337779)  # Gaia DR3 387837952011193088 (J2016.0)
GAPS_QUIET_ID = "387837952011193088"
PERSISTENCE_STD = (321.620483, -0.892541)  # Ivezic et al. 2007 standard, r = 14.467
CYG61_A = (316.72475, 38.74942)  # 61 Cyg A (TIC 165602000)
BARNARD_NAME = "Barnard" + chr(39) + "s star"
BARNARD_GAIA_ID = "4472832130942575872"


# ---------------------------------------------------------------------------
# Cases shared by the live tests and the fixture recorder
# ---------------------------------------------------------------------------


async def case_rrab(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=client, use_cache=False)


async def case_3c273(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*QSO_3C273, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=client, use_cache=False)


async def case_quiet(client: httpx.AsyncClient) -> td.LightCurveResult:
    # Default survey set (ztf, neowise, gaia).
    return await td.get_lightcurves(*QUIET_STAR, radius_arcsec=3.0, client=client, use_cache=False)


async def case_rr_lyr_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RR_LYR, radius_arcsec=3.0, surveys="tess", client=client, tess_max_sectors=1,
                                    use_cache=False)


async def case_ea(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*EA_STAR, radius_arcsec=3.0, surveys="ztf,gaia", client=client, use_cache=False)


async def case_ew(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*EW_STAR, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_gaps_quiet(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAPS_QUIET, radius_arcsec=1.0, surveys="gaia", client=client, use_cache=False)


async def case_persistence_std(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*PERSISTENCE_STD, radius_arcsec=3.0, client=client, use_cache=False)


async def case_tess_61cyg(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*CYG61_A, radius_arcsec=45.0, surveys="tess", client=client, tess_max_sectors=2,
                                    use_cache=False)


async def case_barnard_name(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(BARNARD_NAME, client=client, radius_arcsec=3.0, surveys="gaia",
                                            use_cache=False)


async def case_rrab_neowise_wide(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RRAB, radius_arcsec=60.0, surveys="neowise", client=client, period=False,
                                    use_cache=False)


async def _asteroid_case(client: httpx.AsyncClient, command: str, epoch: float) -> tuple[td.EphemerisPoint, td.SolarSystemResult]:
    eph = (await td.horizons_ephemeris(command, epoch, client=client))[0]
    sky = await td.solar_system_objects(eph.ra, eph.dec, epoch_mjd=epoch, radius_arcsec=600.0, client=client)
    return eph, sky


async def case_vesta(client: httpx.AsyncClient):
    return await _asteroid_case(client, "4;", VESTA_EPOCH_MJD)


async def case_ceres(client: httpx.AsyncClient):
    return await _asteroid_case(client, "1;", CERES_EPOCH_MJD)


async def case_skybot_empty(client: httpx.AsyncClient) -> td.SolarSystemResult:
    # Near the north ecliptic pole region (dec +80): no catalogued body within 30".
    return await td.solar_system_objects(180.0, 80.0, epoch_mjd=VESTA_EPOCH_MJD, radius_arcsec=30.0, client=client)


async def case_name_3c273_gaia(client: httpx.AsyncClient) -> td.LightCurveResult:
    ra, dec, _resolved = await td.resolve_name("3C 273", client)
    return await td.get_lightcurves(ra, dec, radius_arcsec=3.0, surveys="gaia", client=client, name="3C 273",
                                    use_cache=False)


async def case_ztf_bad_collection(client: httpx.AsyncClient) -> Any:
    try:
        return await td.fetch_ztf(client, *QUIET_STAR, 3.0, collection="ztf_dr999")
    except td.UpstreamServiceError as exc:
        return exc


CASES: dict[str, Callable[[httpx.AsyncClient], Awaitable[Any]]] = {
    "rrab_css_j132708": case_rrab,
    "qso_3c273": case_3c273,
    "quiet_s82_standard": case_quiet,
    "rr_lyr_tess": case_rr_lyr_tess,
    "vesta_2023": case_vesta,
    "ceres_2024": case_ceres,
    "skybot_empty": case_skybot_empty,
    "ztf_bad_collection": case_ztf_bad_collection,
    "name_3c273_gaia": case_name_3c273_gaia,
    "ea_ztfj0140": case_ea,
    "ew_ztfj0300": case_ew,
    "gaps_quiet": case_gaps_quiet,
    "persistence_std": case_persistence_std,
    "tess_61cyg": case_tess_61cyg,
    "barnard_name": case_barnard_name,
    "rrab_neowise_wide": case_rrab_neowise_wide,
}


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def _run(case: Callable[[httpx.AsyncClient], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
            return await case(client)
    return asyncio.run(go())


def _skip_if_unreachable(exc: td.UpstreamServiceError) -> None:
    if exc.retryable:
        pytest.skip(f"upstream unreachable: {exc}")
    raise exc


def _live(case: Callable[[httpx.AsyncClient], Awaitable[Any]]) -> Any:
    try:
        return _run(case)
    except td.UpstreamServiceError as exc:
        _skip_if_unreachable(exc)


def _require_survey(result: td.LightCurveResult, survey: str) -> None:
    for failure in result.failures:
        if failure["survey"] == survey:
            if failure["retryable"]:
                pytest.skip(f"{survey} unreachable: {failure['error']}")
            pytest.fail(f"{survey} failed: {failure['error']}")


@pytest.fixture(scope="module")
def rrab_result() -> td.LightCurveResult:
    return _live(case_rrab)


@pytest.fixture(scope="module")
def qso_result() -> td.LightCurveResult:
    return _live(case_3c273)


pytestmark = pytest.mark.live
MIN_N_QUIET = 20


def within(value: float, truth: float, rel: float) -> bool:
    return abs(value - truth) / truth < rel


def test_live_rrab_ztf_period_within_one_percent(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "ztf")
    ztf = [p for p in rrab_result.period_search.per_series if p.series.startswith("ztf:")]
    assert ztf, "no ZTF periodogram"
    best_ztf = min(ztf, key=lambda p: p.false_alarm_probability)
    assert within(best_ztf.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    assert best_ztf.false_alarm_probability < 1e-10 and best_ztf.reliable
    assert best_ztf.harmonic_test["doubled"] is False  # a pulsator: no period doubling
    best = rrab_result.period_search.best
    assert best is not None and within(best.best_period_days, RRAB_PERIOD_DAYS, 0.001)
    for band in ("ztf:g", "ztf:r"):
        m = rrab_result.variability[band]
        assert m.n >= 50 and m.is_variable is True
        assert 0.3 < m.amplitude_5_95 < 1.5  # RRab V amplitudes 0.5-1.3 mag; Chen+2020 gAmp 0.775 (half-range)


def test_live_rrab_gaia_epoch_photometry(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "gaia")
    keys = {s.key for s in rrab_result.series}
    assert {"gaia:G", "gaia:BP", "gaia:RP"} <= keys
    g = next(s for s in rrab_result.series if s.key == "gaia:G")
    assert g.source_ids == ["Gaia DR3 1476301553808173824"]
    m = rrab_result.variability["gaia:G"]
    assert m.n >= 50 and m.is_variable is True
    # Intensity-averaged catalogue G = 15.116; a magnitude average of a pulsator is a bit fainter.
    assert 14.9 < m.weighted_mean < 15.5
    gaia_p = next(p for p in rrab_result.period_search.per_series if p.series == "gaia:G")
    assert within(gaia_p.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    # Gaia times lie within the DR3 epoch-photometry window (2014-07-25 .. 2017-05-28).
    t, _, _ = g.good_arrays()
    assert 56863 < t.min() and t.max() < 57902


def test_live_rrab_neowise_visits(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "neowise")
    w1 = next(s for s in rrab_result.series if s.key == "neowise:W1")
    assert len(w1.points) >= 10
    assert all(p.n and p.n >= 3 for p in w1.points)
    span = w1.points[-1].mjd - w1.points[0].mjd
    assert span > 8 * 365.25  # NEOWISE-R: Dec 2013 - Aug 2024
    assert 12.0 < rrab_result.variability["neowise:W1"].weighted_mean < 15.5


def test_live_3c273_variable_in_ztf(qso_result: td.LightCurveResult) -> None:
    _require_survey(qso_result, "ztf")
    ztf = {k: m for k, m in qso_result.variability.items() if k.startswith("ztf:") and m.n >= 20}
    assert ztf, "3C 273 has no ZTF light curve"
    assert any(m.is_variable for m in ztf.values()), {k: m.evidence for k, m in ztf.items()}
    assert qso_result.as_dict()["variability"]["summary"]["is_variable"] is True


def test_live_3c273_neowise(qso_result: td.LightCurveResult) -> None:
    _require_survey(qso_result, "neowise")
    w1 = next(s for s in qso_result.series if s.key == "neowise:W1")
    assert len(w1.points) >= 15
    # AllWISE W1 of 3C 273 is ~8.2 mag (Vega); NEOWISE per-visit means cluster near it.
    assert 7.5 < qso_result.variability["neowise:W1"].weighted_mean < 9.0


def test_live_name_resolution_gaia_source() -> None:
    result = _live(case_name_3c273_gaia)
    _require_survey(result, "gaia")
    # 3C 273 = Gaia DR3 3700386905605055360 (see tests/test_live_canary.py).
    assert result.provenance["gaia"]["source_id"] == "3700386905605055360"
    assert result.provenance["gaia"]["separation_arcsec"] < 0.1
    assert result.target["name"] == "3C 273"


def test_live_quiet_standard_star_not_variable() -> None:
    result = _live(case_quiet)
    for survey in ("ztf", "neowise", "gaia"):
        _require_survey(result, survey)
    ztf = {k: m for k, m in result.variability.items() if k.startswith("ztf:")}
    assert {"ztf:g", "ztf:r"} <= set(ztf)
    assert {"neowise:W1", "neowise:W2"} <= set(result.variability)
    for key, m in result.variability.items():
        if m.n >= MIN_N_QUIET or key.startswith("neowise:"):
            assert m.is_variable is not True, (key, m.evidence)
    assert 15.6 < result.variability["ztf:r"].weighted_mean < 16.1  # SDSS r = 15.817
    assert not any(k.startswith("gaia:") for k in result.variability)
    assert any("no published epoch photometry" in n for n in result.notes)
    assert result.period_search.best is None
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False
    json.dumps(result.as_dict(), allow_nan=False)  # strict RFC 8259 JSON


def test_live_eclipsing_binary_ea_variable_with_orbital_period() -> None:
    result = _live(case_ea)
    _require_survey(result, "ztf")
    for band in ("ztf:g", "ztf:r"):
        m = result.variability[band]
        assert m.n >= 300 and m.is_variable is True, m.evidence
        assert m.amplitude_5_95 > 0.5  # Chen+2020 rAmp 1.075 mag
    best = result.period_search.best
    assert best is not None and best.series.startswith("ztf:")
    assert within(best.best_period_days, EA_PERIOD_DAYS, 0.01), best
    assert best.harmonic_test["doubled"] is True
    assert result.as_dict()["variability"]["summary"]["is_variable"] is True


def test_live_eclipsing_binary_ew_orbital_period() -> None:
    result = _live(case_ew)
    _require_survey(result, "ztf")
    assert all(result.variability[b].is_variable for b in ("ztf:g", "ztf:r"))
    best = result.period_search.best
    assert best is not None
    # The W UMa minima differ enough in ZTF (O'Connell effect) for the 2P test to decide.
    assert within(best.best_period_days, EW_PERIOD_DAYS, 0.01), (best.best_period_days, best.harmonic_test)
    assert best.alternative_period_days is not None and within(best.alternative_period_days, EW_PERIOD_DAYS / 2, 0.01)


def test_live_gaia_gaps_constant_source_not_variable() -> None:
    result = _live(case_gaps_quiet)
    _require_survey(result, "gaia")
    assert result.provenance["gaia"]["source_id"] == GAPS_QUIET_ID
    g = result.variability["gaia:G"]
    assert g.n >= 20 and g.is_variable is False, g.evidence
    assert result.variability["gaia:BP"].is_variable is None  # BP/RP are not decisive
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_live_neowise_persistence_standard_not_variable() -> None:
    result = _live(case_persistence_std)
    for survey in ("ztf", "neowise"):
        _require_survey(result, survey)
    for key in ("neowise:W1", "neowise:W2"):
        m = result.variability[key]
        assert m.n >= 8 and m.is_variable is False, (key, m.evidence)
    w1 = next(s for s in result.series if s.key == "neowise:W1")
    assert w1.metadata["n_visits_rejected"] >= 3  # latent-image visits dropped
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_live_tess_uses_only_the_nearest_spoc_target() -> None:
    result = _live(case_tess_61cyg)
    _require_survey(result, "tess")
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 165602000"]  # 61 Cyg A only
    assert "165602023" in result.provenance["tess"]["other_tics_in_cone"]  # 61 Cyg B, 30.7"
    assert len(set(tess.metadata["sectors"])) == len(tess.metadata["sectors"]) == 2
    assert set(tess.metadata["flux_columns"].values()) == {"PDCSAP_FLUX"}


def test_live_barnard_by_name_follows_proper_motion() -> None:
    result = _live(case_barnard_name)
    _require_survey(result, "gaia")
    assert result.target["pm_dec_masyr"] == pytest.approx(10362.394, abs=1)
    assert result.provenance["gaia"]["source_id"] == BARNARD_GAIA_ID
    assert result.provenance["gaia"]["separation_arcsec"] < 0.5
    assert any("propagated" in n for n in result.notes)


def test_live_neowise_wide_cone_does_not_adopt_neighbours(rrab_result: td.LightCurveResult) -> None:
    wide = _live(case_rrab_neowise_wide)
    _require_survey(wide, "neowise")
    _require_survey(rrab_result, "neowise")
    w1_wide = next(s for s in wide.series if s.key == "neowise:W1")
    w1_narrow = next(s for s in rrab_result.series if s.key == "neowise:W1")
    assert w1_wide.metadata["n_exposures_used"] == w1_narrow.metadata["n_exposures_used"]
    assert [round(p.value, 6) for p in w1_wide.points] == [round(p.value, 6) for p in w1_narrow.points]


def test_live_rr_lyr_tess_period() -> None:
    result = _live(case_rr_lyr_tess)
    _require_survey(result, "tess")
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 159717514"]
    p = next(p for p in result.period_search.per_series if p.series == "tess:TESS")
    assert within(p.best_period_days, RR_LYR_PERIOD_DAYS, 0.01)
    assert p.period_error_days is not None and p.period_error_days < 0.01 * RR_LYR_PERIOD_DAYS
    assert min(pt.value for pt in tess.points) > 0  # physical (SAP fallback when PDCSAP is over-corrected)
    assert result.variability["tess:TESS"].is_variable is True


@pytest.mark.parametrize("case,number,name", [(case_vesta, 4, "Vesta"), (case_ceres, 1, "Ceres")])
def test_live_skybot_matches_horizons(case, number: int, name: str) -> None:
    eph, sky = _live(case)
    assert name in eph.target
    match = [o for o in sky.objects if o.number == number]
    assert match, [o.name for o in sky.objects]
    obj = match[0]
    assert obj.name == name and obj.type == "asteroid"
    # Independent ephemerides (JPL DE441/SB441 vs IMCCE INPOP) agree to ~arcsec.
    assert obj.separation_arcsec < 5.0
    assert eph.v_mag is not None and obj.v_mag is not None and abs(eph.v_mag - obj.v_mag) < 0.6
    assert sky.objects[0].number == number  # the brightest body sits at the cone centre


def test_live_skybot_empty_field() -> None:
    result = _live(case_skybot_empty)
    assert result.objects == []
    assert result.provenance["http_status"] == 204


def test_live_ztf_bad_collection_is_reported() -> None:
    outcome = _run(case_ztf_bad_collection)
    assert isinstance(outcome, td.UpstreamServiceError), outcome
    if outcome.retryable:
        pytest.skip(f"ZTF unreachable: {outcome}")
    assert outcome.service == "ztf" and "ztf_objects_dr999" in outcome.message


# ---------------------------------------------------------------------------
# Fixture recorder
# ---------------------------------------------------------------------------


async def _record_case(name: str) -> str:
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        try:
            outcome = await CASES[name](client)
            summary = type(outcome).__name__
            if isinstance(outcome, td.LightCurveResult) and outcome.failures:
                summary += f" FAILURES {outcome.failures}"
        except Exception as exc:  # noqa: BLE001 - recorded anyway: the offline suite asserts on it
            summary = f"raised {type(exc).__name__}: {exc}"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for old in FIXTURES.glob(f"{name}.*.body.gz"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (FIXTURES / f"{name}.{idx}.body.gz").write_bytes(gzip.compress(response.content, mtime=0))
        try:
            body = request.content.decode("utf-8", "replace") if request.content else ""
        except httpx.RequestNotRead:  # redirected follow-up requests carry no body
            body = ""
        exchanges.append({
            "method": request.method, "url": str(request.url), "request_body": body,
            "status_code": response.status_code, "content_type": response.headers.get("content-type", ""),
            "location": response.headers.get("location", ""),
        })
    (FIXTURES / f"{name}.json").write_text(json.dumps({"case": name, "exchanges": exchanges}, indent=2), encoding="utf-8")
    return f"{name:<22} {len(exchanges)} exchange(s); {summary}"


async def record(names: list[str]) -> None:
    for line in await asyncio.gather(*(_record_case(n) for n in names)):
        print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "record":
        print(__doc__)
        raise SystemExit(0)
    asyncio.run(record(args[1:] or list(CASES)))
