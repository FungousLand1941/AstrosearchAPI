"""NEOWISE saturation and solar-system coordinate range (review: running-app).

* Saturated NEOWISE single-exposure photometry (W1 < 8, W2 < 7 mag, or saturated pixels
  in the fit; NEOWISE Explanatory Supplement sec. II.1.c) made Barnard's star (W1 ~ 4.5)
  "variable in neowise:W1" (chi^2/dof 13.5, 1.9 mag amplitude). Saturated points are now
  flagged 3, returned for plotting and kept out of the variability decision and the
  period search, and the evidence says so.
* /api/v1/solar-system and ``astrosearch solar-system`` wrapped RA=400 to 40 while every
  other route rejects it; they now answer 422 / exit 2.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_timedomain import load_case, make_app, make_parser, replay, run_case
from test_timedomain_live import RRAB, case_3c273, case_barnard_neowise

import timedomain as td

SAT = td.NEOWISE_SATURATED_FLAG


# ---------------------------------------------------------------------------
# Recorded: Barnard's star (saturated), 3C 273 (just unsaturated), RRab (faint variable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def barnard() -> td.LightCurveResult:
    return run_case("barnard_neowise", case_barnard_neowise)


def test_saturation_limits_follow_the_explanatory_supplement() -> None:
    # "brighter than the saturation limits of W1<8 and W2<7 mag" (NEOWISE ES sec. II.1.c)
    assert td.NEOWISE_SATURATION_MAG == {"W1": 8.0, "W2": 7.0}
    assert {"w1sat", "w2sat"} <= set(td.NEOWISE_COLUMNS)
    assert "w1sat" in td.neowise_adql(10.0, 10.0, 3.0)


def test_barnard_neowise_is_not_variable_from_saturated_points(barnard: td.LightCurveResult) -> None:
    assert barnard.failures == []
    summary = barnard.as_dict()["variability"]["summary"]
    assert summary["is_variable"] is not True and summary["variable_series"] == []
    for key, limit in (("neowise:W1", 8.0), ("neowise:W2", 7.0)):
        series = next(s for s in barnard.series if s.key == key)
        # every visit is saturated (Barnard's star: W1 ~ 4.5, W2 ~ 4.0): all points are
        # kept for plotting, all flagged, none enters the statistics
        assert len(series.points) >= 15
        assert all(p.flag == SAT for p in series.points)
        assert float(sorted(p.value for p in series.points)[len(series.points) // 2]) < limit
        assert series.metadata["n_visits_saturated"] == len(series.points)
        assert series.metadata["saturation_limit_mag"] == limit
        m = barnard.variability[key]
        assert m.n == 0 and m.is_variable is None and m.chi2_dof is None
        assert any("saturated" in e and "not assessed" in e for e in m.evidence), m.evidence
    assert sum("saturation limit" in n and "flag 3" in n for n in barnard.notes) == 2, barnard.notes
    period = barnard.period_search
    assert period is None or (period.best is None and not any(p.series.startswith("neowise")
                                                               for p in period.per_series))


def test_barnard_cli_and_json_say_not_assessed(barnard: td.LightCurveResult) -> None:
    text = td.format_lightcurve_summary(barnard)
    lines = [line for line in text.splitlines() if line.strip().startswith("[neowise:")]
    assert len(lines) == 2 and all("not assessed (saturated)" in line for line in lines), text
    assert "VARIABLE" not in text
    body = barnard.as_dict()
    w1 = next(s for s in body["series"] if s["key"] == "neowise:W1")
    assert {p["flag"] for p in w1["points"]} == {SAT}
    assert "flag 3" in w1["metadata"]["saturated_points"]


def test_barnard_route_returns_saturated_points_without_verdict() -> None:
    with replay("barnard_neowise"), TestClient(make_app()) as http:
        response = http.get("/api/v1/lightcurves", params={"name": "Barnard's star", "radius_arcsec": 3,
                                                           "surveys": "neowise"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variability"]["summary"]["variable_series"] == []
    assert body["variability"]["summary"]["is_variable"] is None
    for s in body["series"]:
        assert s["points"] and all(p["flag"] == SAT for p in s["points"])


def test_3c273_w1_near_the_limit_is_still_assessed() -> None:
    result = run_case("qso_3c273", case_3c273)
    w1 = next(s for s in result.series if s.key == "neowise:W1")
    # 3C 273: W1 ~ 8.48 (w1sat = 0 throughout), just fainter than the W1 < 8 limit
    assert w1.metadata["n_visits_saturated"] == 0 and w1.metadata["n_exposures_saturated"] == 0
    assert all(p.flag == 0 for p in w1.points)
    m = result.variability["neowise:W1"]
    assert m.is_variable is not None and m.n >= 15 and 8.0 < m.weighted_mean < 9.0
    # W2 ~ 7.5 (> 7): decided on its unsaturated visits; the few exposures with saturated
    # pixels (w2sat > 0) are flagged, not averaged in
    w2 = next(s for s in result.series if s.key == "neowise:W2")
    assert w2.metadata["n_visits_saturated"] == 0
    assert result.variability["neowise:W2"].is_variable is not None
    assert w2.metadata["n_exposures_saturated"] >= 1
    assert any(p.flag == SAT for p in w2.points)  # binned per visit, flagged
    assert all(p.flag in (0, SAT) for p in w2.points)


def test_faint_variable_rrab_single_exposures_still_variable() -> None:
    body = next(e for e in load_case("rrab_css_j132708") if "neowiser_p1bs_psd" in e["request_body"])["content"].decode()
    single = td.parse_neowise_csv(body, *RRAB, binned=False)
    w1 = next(s for s in single.series if s.key == "neowise:W1")
    assert w1.metadata["n_visits_saturated"] == 0 and not any(p.flag == SAT for p in w1.points)
    assert td.series_variability(w1).is_variable is True


# ---------------------------------------------------------------------------
# Synthetic single exposures: the rules themselves
# ---------------------------------------------------------------------------

COLUMNS = td.NEOWISE_COLUMNS


def _csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    out.write(",".join(COLUMNS) + "\n")
    base = {"ra": 10.0, "dec": 10.0, "w1sigmpro": 0.02, "w2sigmpro": 0.02, "qual_frame": 10, "qi_fact": 1.0,
            "saa_sep": 100, "moon_masked": "00", "cc_flags": "0000", "nb": 1, "na": 0, "w1rchi2": 1.0,
            "w2rchi2": 1.0, "w1sat": 0.0, "w2sat": 0.0}
    for row in rows:
        full = {**base, **row}
        out.write(",".join(str(full.get(c, "")) for c in COLUMNS) + "\n")
    return out.getvalue()


def _visits(mags: list[float], *, per_visit: int = 6, w1sat: dict[tuple[int, int], float] | None = None,
            scatter: float = 0.0) -> str:
    rows = []
    for v, mag in enumerate(mags):
        for k in range(per_visit):
            wiggle = scatter * (1 if (v + k) % 2 else -1)
            rows.append({"mjd": 57000.0 + 180.0 * v + 0.1 * k, "w1mpro": mag + wiggle, "w2mpro": mag + 2.0,
                         "w1sat": (w1sat or {}).get((v, k), 0.0)})
    return _csv(rows)


def test_saturated_visit_is_flagged_and_excluded() -> None:
    # visits 0-4 at 9.0 mag (fine), visits 5-6 at 7.0 mag (W1 < 8: saturated)
    result = td.parse_neowise_csv(_visits([9.0] * 5 + [7.0, 7.2]), 10.0, 10.0)
    w1 = next(s for s in result.series if s.band == "W1")
    good = [p for p in w1.points if p.flag == 0]
    sat = [p for p in w1.points if p.flag == SAT]
    assert len(good) == 5 and len(sat) == 2 and all(p.value < 8.0 for p in sat)
    assert w1.metadata["n_visits"] == 5 and w1.metadata["n_visits_saturated"] == 2
    m = td.series_variability(w1)
    # the 2 mag jump is saturated photometry, not variability
    assert m.n == 5 and m.is_variable is False
    assert any("2 saturated point(s)" in e for e in m.evidence)


def test_saturated_pixels_flag_single_exposures_or_whole_visits() -> None:
    # 9.0 mag everywhere; visit 1 has one exposure with saturated pixels (flagged alone),
    # visit 2 has 4 of 6 (fewer than 3 unsaturated left: whole visit flagged)
    sat = {(1, 0): 0.05, **{(2, k): 0.1 for k in range(4)}}
    result = td.parse_neowise_csv(_visits([9.0] * 6, w1sat=sat), 10.0, 10.0, binned=False)
    w1 = next(s for s in result.series if s.band == "W1")
    flagged = [p for p in w1.points if p.flag == SAT]
    assert len(flagged) == 1 + 6
    assert w1.metadata["n_exposures_saturated"] == 7 and w1.metadata["n_visits_saturated"] == 1
    assert w1.metadata["n_exposures_used"] == 6 * 6 - 7
    # W2 (w2sat = 0, 11 mag) is untouched
    w2 = next(s for s in result.series if s.band == "W2")
    assert not any(p.flag == SAT for p in w2.points)
    assert any("further exposure(s) with saturated pixels (w1sat > 0)" in n for n in result.notes), result.notes


def test_all_saturated_band_is_not_assessed_and_scatter_is_ignored() -> None:
    result = td.parse_neowise_csv(_visits([4.0, 5.5, 3.5, 5.0, 4.2, 3.9], scatter=0.3), 10.0, 10.0)
    w1 = next(s for s in result.series if s.band == "W1")
    assert len(w1.points) == 6 and all(p.flag == SAT for p in w1.points)
    m = td.series_variability(w1)
    assert m.is_variable is None and m.n == 0
    assert any("no unsaturated" in n for n in result.notes)
    assert not any("none of the" in n for n in result.notes)  # they passed the quality cuts
    search = td.search_periods([w1])
    assert search.best is None


# ---------------------------------------------------------------------------
# Solar system: out-of-range coordinates are refused, never wrapped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"ra": 400.0, "dec": 0.0},
    {"ra": 360.0, "dec": 0.0},
    {"ra": -0.5, "dec": 0.0},
    {"ra": 10.0, "dec": 90.5},
    {"ra": 10.0, "dec": -91.0},
])
def test_router_solar_system_rejects_out_of_range_coordinates(params: dict[str, float]) -> None:
    with TestClient(make_app()) as http:  # no network: validation comes first
        response = http.get("/api/v1/solar-system", params={**params, "epoch_mjd": 60000})
    assert response.status_code == 422, response.text
    detail = json.dumps(response.json()["detail"]).lower()
    assert ("ra" in detail and "360" in detail) or "dec" in detail


@pytest.mark.parametrize("argv", [["--ra", "400", "--dec", "0"], ["--ra", "-1", "--dec", "0"],
                                  ["--ra", "10", "--dec", "95"]])
def test_cli_solar_system_rejects_out_of_range_coordinates(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    args = make_parser().parse_args(["solar-system", *argv, "--epoch-mjd", "60000"])
    assert args.handler(args) == 2
    captured = capsys.readouterr()
    assert "Error" in captured.err and captured.out == ""


def test_solar_system_accepts_the_edges_of_the_range() -> None:
    import asyncio

    import httpx
    import respx

    # RA 0 and 359.999 and Dec +-90 are valid: they reach SkyBoT (mocked) unchanged.
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["-ra"])
        return httpx.Response(200, json={"data": []}, headers={"content-type": "application/json"})

    async def go(ra: float, dec: float) -> None:
        with respx.mock(assert_all_mocked=True) as router:
            router.get(td.SKYBOT_CONESEARCH_URL).mock(side_effect=handler)
            try:
                await td.solar_system_objects(ra, dec, epoch_mjd=60000.0)
            except td.UpstreamServiceError:
                pass  # only the coordinate validation matters here

    for ra, dec in ((0.0, 90.0), (359.999, -90.0)):
        asyncio.run(go(ra, dec))
    assert [float(v) for v in seen] == [0.0, 359.999]
