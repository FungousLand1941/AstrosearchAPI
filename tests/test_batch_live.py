"""Live tests for batch.py against the real archives (run with ``-m live``; add ``-s`` to see the benchmark).

* 200 random-ish targets that include 3C 273, M87, HD 209458, Vega (J2000 + proper motion) and Vega-adjacent
  positions, matched with Gaia DR3 (CDS XMatch), SIMBAD (TAP upload), 2MASS PSC / AllWISE (IRSA TAP upload)
  and FIRST / NVSS (HEASARC TAP upload): the known objects must be identified correctly in every catalog
  (NVSS included: 3C 273 and M87 carry a per-target 10" radius because their NVSS counterparts lie 5.6" and
  6.5" away -- NVSS positions of extended sources are uncertain by arcseconds), with one request per catalog
  chunk (plus transient retries).
* The batch answers must equal per-target cone searches for a sample of 10 targets (same ids, separations
  within 0.01 arcsec), with matches compared in every catalog.
* Benchmark: 1000 targets in batch mode versus the cone approach. The cone side is measured on 100 of the
  targets; the report gives the measured 100-target speedup (batch and cones both run on the same 100
  targets) next to the linear extrapolation to 1000 targets (the cone path is paced per endpoint, so its
  cost is linear in the target count) and quotes the range. Set ``BATCH_LIVE_FULL_CONE=1`` to measure the
  cone side on all 1000 targets instead (~6000 paced requests, ~20 minutes).

Only network failures and HTTP 5xx lead to a skip; every astrophysical assertion is strict.
"""

from __future__ import annotations

import asyncio
import math
import os
import time

import numpy as np
import pytest
from live_policy import network_text

from batch import BatchCrossmatcher, BatchResult, format_report

pytestmark = pytest.mark.live

CATALOGS = ["gaia_dr3", "simbad", "twomass_psc", "allwise", "first", "nvss"]
RADIUS = 5.0

# SIMBAD ICRS J2000 positions; HD 209458 at the NASA Exoplanet Archive J2015.5 position (pm ~ 30 mas/yr).
KNOWN = [
    {"id": "3C 273", "ra": 187.2779154, "dec": 2.0523883, "radius_arcsec": 10.0},
    {"id": "M87", "ra": 187.7059308, "dec": 12.3911233, "radius_arcsec": 10.0},
    {"id": "HD 209458", "ra": 330.79502, "dec": 18.88432},
    {"id": "Vega", "ra": 279.23473479, "dec": 38.78368896, "epoch": 2000.0, "pm_ra_masyr": 200.94, "pm_dec_masyr": 286.23},
    {"id": "Vega+60N", "ra": 279.23473479, "dec": 38.78368896 + 60.0 / 3600.0},
    {"id": "Vega+90E", "ra": 279.23473479 + 90.0 / 3600.0 / math.cos(math.radians(38.78368896)), "dec": 38.78368896},
]

# Nearest-match identities (radius 5"; 10" for 3C 273 and M87), from the archives' own designations.
EXPECTED = {
    ("3C 273", "gaia_dr3"): "3700386905605055360",
    ("3C 273", "simbad"): "3C 273",
    ("3C 273", "twomass_psc"): "12290669+0203085",
    ("3C 273", "allwise"): "J122906.69+020308.6",
    ("3C 273", "first"): "FIRST J122906.7+020308",
    ("3C 273", "nvss"): "NVSS J122906+020305",
    ("M87", "simbad"): "M 87",
    ("M87", "gaia_dr3"): "3907709439453756032",
    ("M87", "twomass_psc"): "12304942+1223278",
    ("M87", "allwise"): "J123049.43+122328.0",
    ("M87", "first"): "FIRST J123049.3+122323",
    ("M87", "nvss"): "NVSS J123049+122321",
    ("HD 209458", "gaia_dr3"): "1779546757669063552",
    ("HD 209458", "simbad"): "HD 209458",
    ("HD 209458", "twomass_psc"): "22031077+1853036",
    ("HD 209458", "allwise"): "J220310.79+185303.3",
    ("Vega", "simbad"): "* alf Lyr",
    ("Vega", "twomass_psc"): "18365633+3847012",
    ("Vega", "allwise"): "J183656.51+384704.4",
}


@pytest.fixture(scope="module", autouse=True)
def polite_pacing():
    """Production pacing (5 requests/s per endpoint) instead of the offline suite's 1000/s (tests/conftest.py)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROVIDER_REQUESTS_PER_SECOND", "5")
        yield


def random_targets(n: int, seed: int) -> list[dict]:
    """Deterministic 'random-ish' positions: half uniform on the sky above Dec -30, half in the FIRST/SDSS area."""
    rng = np.random.default_rng(seed)
    half = n // 2
    ra1 = rng.uniform(0.0, 360.0, half)
    dec1 = np.degrees(np.arcsin(rng.uniform(-0.5, 1.0, half)))
    ra2 = rng.uniform(120.0, 240.0, n - half)
    dec2 = np.degrees(np.arcsin(rng.uniform(0.0, math.sin(math.radians(60.0)), n - half)))
    ras = np.concatenate([ra1, ra2])
    decs = np.concatenate([dec1, dec2])
    return [{"id": f"r{seed}-{i:04d}", "ra": float(a), "dec": float(d)} for i, (a, d) in enumerate(zip(ras, decs))]


def unavailable(result: BatchResult) -> list[str]:
    """Catalog runs that could not be completed because an archive was down (network errors / 5xx)."""
    down = []
    for name, run in result.runs.items():
        if run.failed_targets or run.fallback_targets:
            text = " ".join(run.errors)
            if any(k in text for k in ("Unavailable", "Timeout", "HTTP 5", "circuit", "ConnectError", "ReadError")):
                down.append(f"{name}: {text[:300]}")
    return down


def run_batch(targets, catalogs, *, strategies=None, fallback=True) -> BatchResult:
    engine = BatchCrossmatcher(fallback_to_cone=fallback)
    return asyncio.run(engine.run(targets, catalogs, radius_arcsec=RADIUS, strategies=strategies))


@pytest.fixture(scope="module")
def batch200() -> BatchResult:
    targets = KNOWN + random_targets(200 - len(KNOWN), seed=20260928)
    result = run_batch(targets, CATALOGS, fallback=False)
    down = unavailable(result)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    print("\n" + format_report(result))
    return result


def test_live_200_targets_known_objects(batch200):
    result = batch200
    assert len(result.targets) == 200
    for (target, catalog), expected in EXPECTED.items():
        found = [m["source_id"] for m in result.target_matches(target).get(catalog, [])]
        assert found and found[0] == expected, (target, catalog, found)
    # Vega (G ~ 0) is not in Gaia DR3; nothing of Vega's lies 60" north or 90" east of it.
    assert result.target_matches("Vega").get("gaia_dr3", []) == []
    for adjacent in ("Vega+60N", "Vega+90E"):
        assert "* alf Lyr" not in [m["source_id"] for m in result.target_matches(adjacent).get("simbad", [])]
    # Gaia DR3 source of 3C 273 at the SIMBAD position: < 10 mas.
    assert result.target_matches("3C 273")["gaia_dr3"][0]["separation_arcsec"] < 0.01
    # 2MASS Vega is matched after moving the 1999.3 detection with Vega's proper motion (given) and parallax
    # (adopted from the SIMBAD row, as a single-object search does).
    assert result.target_matches("Vega")["twomass_psc"][0]["epoch_propagation"] == "target_pm_parallax"
    plx = result.target_association("Vega")["target_parallax"]
    assert plx["source"] == "adopted" and plx["catalog"] == "simbad" and abs(plx["parallax_mas"] - 130.23) < 0.5
    # NVSS: the 3C 273 core and M87 (both extended) at 5.6" and 6.5".
    assert 5.0 < result.target_matches("3C 273")["nvss"][0]["separation_arcsec"] < 6.0
    assert 6.0 < result.target_matches("M87")["nvss"][0]["separation_arcsec"] < 7.0
    # One request per catalog chunk: 200 targets fit in one chunk. Gaia's XMatch radii (5", 10" and Vega's
    # 8.6" proper-motion cone) share one radius bucket. Retries are transient connection resets.
    for name, run in result.runs.items():
        assert run.strategy == ("xmatch" if name == "gaia_dr3" else "upload"), name
        assert run.chunks == 1, name
        assert run.requests == run.chunks + run.retries, name
        assert not run.errors, (name, run.errors)
    # Canonical fields on every match.
    for item in result.as_dict(include_data=False)["targets"]:
        own_radius = item["radius_arcsec"] or RADIUS
        for name, matches in item["matches"].items():
            for m in matches:
                assert m["separation_arcsec"] <= own_radius + 1e-6
                assert m["source_id"] and math.isfinite(m["ra"]) and math.isfinite(m["dec"])
                if name in ("gaia_dr3", "twomass_psc", "allwise"):
                    assert m["positional_error_arcsec"] is not None and m["epoch"] is not None


def test_live_batch_agrees_with_cone_searches(batch200):
    """10 targets: the batch rows equal per-target cone searches (ids; separations within 0.01")."""
    matched_random = [t for t in batch200.targets[len(KNOWN):] if sum(len(v) for v in batch200.target_matches(t.id).values())]
    sample = [t for t in batch200.targets if t.id in {"3C 273", "M87", "HD 209458", "Vega", "Vega+60N"}]
    sample += matched_random[:10 - len(sample)]
    assert len(sample) == 10
    cone = run_batch(sample, CATALOGS, strategies={c: "cone" for c in CATALOGS})
    down = unavailable(cone)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    compared = dict.fromkeys(CATALOGS, 0)
    for item in sample:
        up = batch200.target_matches(item.id)
        cs = cone.target_matches(item.id)
        for catalog in CATALOGS:
            a = {m["source_id"]: m["separation_arcsec"] for m in up.get(catalog, [])}
            b = {m["source_id"]: m["separation_arcsec"] for m in cs.get(catalog, [])}
            assert set(a) == set(b), (item.id, catalog, sorted(a), sorted(b))
            for sid in a:
                assert abs(a[sid] - b[sid]) < 0.01, (item.id, catalog, sid, a[sid], b[sid])
                compared[catalog] += 1
    # Every catalog's upload join (NVSS and FIRST included) is compared on real matches, not 0 against 0.
    assert all(compared.values()), compared
    assert sum(compared.values()) >= 20
    assert cone.request_count >= len(sample) * len(CATALOGS)


def _same_answers(bulk: BatchResult, cone: BatchResult, ids: list[str]) -> None:
    for target_id in ids:
        for catalog in CATALOGS:
            a = {m["source_id"]: m["separation_arcsec"] for m in bulk.target_matches(target_id).get(catalog, [])}
            b = {m["source_id"]: m["separation_arcsec"] for m in cone.target_matches(target_id).get(catalog, [])}
            assert set(a) == set(b), (target_id, catalog, sorted(a), sorted(b))
            assert all(abs(a[s] - b[s]) < 0.01 for s in a), (target_id, catalog)


def _timed(targets, **kwargs) -> tuple[BatchResult, float]:
    started = time.perf_counter()
    result = run_batch(targets, CATALOGS, **kwargs)
    wall = time.perf_counter() - started
    down = unavailable(result)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    return result, wall


def test_live_benchmark_1000_targets_vs_cone():
    """1000 targets: batch vs per-target cone searches.

    Measured both ways on the same 100 targets (the lower end of the quoted speedup range) and, for 1000
    targets, batch measured against the cone side extrapolated linearly from those 100 targets -- or measured
    on all 1000 with BATCH_LIVE_FULL_CONE=1.
    """
    targets = KNOWN + random_targets(1000 - len(KNOWN), seed=4)
    full_cone = os.getenv("BATCH_LIVE_FULL_CONE", "").strip().lower() in {"1", "true", "yes"}
    sample = targets[:100]
    cone_strategies = {c: "cone" for c in CATALOGS}

    bulk, bulk_wall = _timed(targets, fallback=False)
    bulk100, bulk100_wall = _timed(sample, fallback=False)
    cone100, cone100_wall = _timed(sample, strategies=cone_strategies)
    if full_cone:
        cone_all, cone_all_wall = _timed(targets, strategies=cone_strategies)
        cone_requests_1000, cone_wall_1000, how = float(cone_all.request_count), cone_all_wall, "measured"
    else:
        scale = len(targets) / len(sample)
        cone_requests_1000, cone_wall_1000 = cone100.request_count * scale, cone100_wall * scale
        how = "linear extrapolation from the 100 measured"

    speed100 = cone100_wall / bulk100_wall
    speed1000 = cone_wall_1000 / bulk_wall
    report = [
        "", format_report(bulk), format_report(bulk100), format_report(cone100),
        f"BENCHMARK {len(CATALOGS)} catalogs, radius {RADIUS:g}\" (10\" for 3C 273, M87):",
        (f"  100 targets, both measured : batch {bulk100.request_count} requests / {bulk100_wall:.1f} s, "
         f"cone {cone100.request_count} requests / {cone100_wall:.1f} s -> "
         f"{cone100.request_count / bulk100.request_count:.0f}x fewer requests, {speed100:.1f}x faster"),
        (f"  1000 targets               : batch {bulk.request_count} requests / {bulk_wall:.1f} s (measured), "
         f"cone ~{cone_requests_1000:.0f} requests / ~{cone_wall_1000:.0f} s ({how}) -> "
         f"{cone_requests_1000 / bulk.request_count:.0f}x fewer requests, {speed1000:.1f}x faster"),
        (f"  wall-time speedup range    : {min(speed100, speed1000):.1f}x .. {max(speed100, speed1000):.1f}x "
         "(depends on archive load and connection resets; one run is not a stable single figure)"),
    ]
    print("\n".join(report))

    # The same answers for the 100 targets measured both ways (and for all 1000 when measured).
    _same_answers(bulk, cone100, [t["id"] for t in sample])
    _same_answers(bulk100, cone100, [t["id"] for t in sample])
    if full_cone:
        _same_answers(bulk, cone_all, [t["id"] for t in targets])
    for result in (bulk, bulk100):
        for name, run in result.runs.items():
            assert run.chunks == 1 and run.requests == 1 + run.retries, (name, run.chunks, run.requests)
    assert cone100.request_count >= len(sample) * len(CATALOGS)
    assert speed100 > 2.0 and speed1000 > 3.0


# ---------------------------------------------------------------------------
# Round 2: generic VizieR tables, view priors, capped epoch-widened cones
# ---------------------------------------------------------------------------


def _live(targets, catalogs, radius) -> BatchResult:
    result = asyncio.run(BatchCrossmatcher(fallback_to_cone=False).run(targets, catalogs, radius_arcsec=radius))
    down = unavailable(result)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    return result


def test_live_hipparcos_and_tycho2_rows_at_xmatch_positions():
    """Bright high-proper-motion stars at their SIMBAD J2000 positions (no epoch): every star is matched in
    Hipparcos (XMatch matches VizieR's J2000 _RAJ2000; the J1991.25 meta.main columns dropped 6 of 7), and
    Tycho-2 identifiers are TYC1-TYC2-TYC3."""
    from test_batch_fixtures import BRIGHT_STARS

    targets = [{"id": name, "ra": ra, "dec": dec} for name, (ra, dec, _pa, _pd) in BRIGHT_STARS.items()]
    result = _live(targets, ["vizier:I/239/hip_main", "vizier:I/259/tyc2"], 5.0)
    for name in BRIGHT_STARS:
        [match] = result.target_matches(name)["vizier:I/239/hip_main"]
        assert match["separation_arcsec"] < 0.2 and match["epoch"] == 2000.0, (name, match)
    assert result.target_matches("HD 209458")["vizier:I/239/hip_main"][0]["separation_arcsec"] < 0.02
    tycho = {name: [m["source_id"] for m in result.target_matches(name).get("vizier:I/259/tyc2", [])]
             for name in ("Barnard", "HD 209458")}
    assert tycho == {"Barnard": ["425-2502-1"], "HD 209458": ["1688-1821-1"]}


def test_live_vizier_view_posterior_equals_its_survey():
    """The same 2MASS rows requested as twomass_psc (IRSA upload) and as vizier:II/246/out (CDS XMatch) get the
    same posterior (the view used to get a generic density prior)."""
    from test_batch_fixtures import DENSITY_RADIUS, DENSITY_TARGETS

    survey = _live(DENSITY_TARGETS, ["twomass_psc"], DENSITY_RADIUS)
    view = _live(DENSITY_TARGETS, ["vizier:II/246/out"], DENSITY_RADIUS)
    compared = 0
    for item in DENSITY_TARGETS:
        a = {m["source_id"]: m["confidence"] for m in survey.target_matches(item["id"]).get("twomass_psc", [])}
        b = {m["source_id"]: m["confidence"] for m in view.target_matches(item["id"]).get("vizier:II/246/out", [])}
        assert set(a) == set(b), (item["id"], a, b)
        for sid, value in a.items():
            assert abs(round(value * 1e6) - round(b[sid] * 1e6)) <= 1, (item["id"], sid, value, b[sid])
            compared += 1
    assert compared >= 3


def test_live_epoch_widened_view_cone_is_capped_and_matched():
    """3C 286 at epoch 2024 (no proper motion) against 2MASS via VizieR: the 284" epoch-widened cone is capped at
    XMatch's 180" limit and finds 13310829+3030331 (it failed as 'request failed' before)."""
    from test_batch_fixtures import WIDE_EPOCH_TARGETS

    result = _live(WIDE_EPOCH_TARGETS, ["vizier:II/246/out"], 5.0)
    [match] = result.target_matches("3C 286@2024")["vizier:II/246/out"]
    assert match["source_id"] == "13310829+3030331" and match["separation_arcsec"] < 0.3
    run = result.runs["vizier:II/246/out"]
    assert run.failed_targets == 0 and any("capped at the XMatch limit" in w for w in run.warnings)


# ---------------------------------------------------------------------------
# Round 3: J2000 generic tables, complete identifiers, dated targets, Gaia DR2/EDR3 ids
# ---------------------------------------------------------------------------


def test_live_j2000_tables_date_high_pm_stars_at_j2000():
    """Undated SIMBAD J2000 positions vs UCAC4, URAT1 and USNO-B1.0: the true counterparts are dated 2000.0 (the
    tables' mean observation epochs re-dated them: Barnard's UCAC4 row had posterior 0.0067), identifiers are
    complete, and UCAC4/URAT1 rows carry XMatch's radec_err as positional error."""
    from test_batch_fixtures import J2000_TABLE_TARGETS

    catalogs = ["vizier:I/322A/out", "vizier:I/329/urat1", "vizier:I/284/out"]
    result = _live(J2000_TABLE_TARGETS, catalogs, 5.0)
    ucac4 = {"Barnard": "474-068224", "Kapteyn": "225-005836", "Groombridge 1830": "639-046031",
             "61 Cyg A": "644-101660"}
    for name, ucac in ucac4.items():
        matches = result.target_matches(name)["vizier:I/322A/out"]
        match = matches[0]
        assert match["source_id"] == ucac and match["epoch"] == 2000.0, (name, match)
        assert match["positional_error_arcsec"] is not None
        if name == "61 Cyg A":  # shares the posterior with UCAC4 644-101659 (2.47" away, same motion)
            other = next(m for m in matches if m["source_id"] == "644-101659")
            assert match["confidence"] > 0.5 > other["confidence"] and match["confidence"] + other["confidence"] > 0.99
        else:
            assert match["confidence"] > 0.99, (name, match["confidence"])
    hd = next(m for m in result.target_matches("HD 80606")["vizier:I/329/urat1"] if m["pm_ra_masyr"] is not None)
    assert hd["epoch"] == 2000.0 and hd["separation_arcsec"] < 0.3 and hd["positional_error_arcsec"] is not None
    usno = [m["source_id"] for t in result.targets for m in result.target_matches(t.id).get("vizier:I/284/out", [])]
    assert usno and all(len(i) == 13 for i in usno), usno


def test_live_usno_b1_ids_are_complete_and_unique():
    from test_batch_fixtures import USNOB_FIELD_TARGETS

    result = _live(USNOB_FIELD_TARGETS, ["vizier:I/284/out", "vizier:I/297/out"], 20.0)
    for catalog in ("vizier:I/284/out", "vizier:I/297/out"):
        ids = [m["source_id"] for m in result.target_matches("3C 273")[catalog]]
        assert len(ids) >= 3 and len(set(ids)) == len(ids) and all(len(i) == 13 for i in ids), (catalog, ids)
    assert result.target_matches("3C 273")["vizier:I/284/out"][0]["source_id"] == "0920-00258572"
    assert result.target_matches("3C 273")["vizier:I/297/out"][0]["source_id"] == "0920-00260612"


def test_live_dated_targets_find_hipparcos_and_tycho2_rows():
    """Targets at epoch 2016.0 with their SIMBAD proper motions: every star is found in Hipparcos 2 (HD 189733
    and 51 Peg, and the six fast movers, had no row at 3" when the cone was left at the 2016 position)."""
    from test_batch_fixtures import DATED_2016_TARGETS

    hip = {"Barnard": "87937", "61 Cyg A": "104214", "Kapteyn": "24186", "Groombridge 1830": "57939",
           "Lacaille 9352": "114046", "tau Cet": "8102", "51 Peg": "113357", "HD 189733": "98505"}
    result = _live(DATED_2016_TARGETS, ["vizier:I/311/hip2", "vizier:I/259/tyc2"], 3.0)
    for name, number in hip.items():
        matches = result.target_matches(name).get("vizier:I/311/hip2", [])
        assert [m["source_id"] for m in matches] == [number], (name, matches)
        # XMatch's Hipparcos columns carry no positional or proper-motion errors: the association's error model
        # for the 16-year propagation keeps these posteriors near 0.79 (see test_batch_round3).
        assert matches[0]["confidence"] > 0.75 and matches[0]["epoch"] == 2000.0
    for name in ("HD 189733", "51 Peg"):
        assert result.target_matches(name).get("vizier:I/259/tyc2"), name


def test_live_gaia_dr2_keeps_source_id():
    from test_batch_fixtures import J2000_TABLE_TARGETS

    targets = [t for t in J2000_TABLE_TARGETS if t["id"] in {"Barnard", "61 Cyg A"}]
    result = _live(targets, ["vizier:I/345/gaia2", "vizier:I/350/gaiaedr3"], 5.0)
    for catalog, cyg_a in (("vizier:I/345/gaia2", "1872046574983507456"),
                           ("vizier:I/350/gaiaedr3", "1872046609345556480")):
        barnard = result.target_matches("Barnard")[catalog][0]
        assert barnard["source_id"] == "4472832130942575872" and barnard["epoch"] == 2000.0, (catalog, barnard)
        assert result.target_matches("61 Cyg A")[catalog][0]["source_id"] == cyg_a


# ---------------------------------------------------------------------------
# VizieR-hosted radio catalogs and a registered VizieR table in batch mode (final review): these went through
# TAPVizieR table uploads, which stalled for 2 x 300 s per split level (20 minutes for 2 targets, live).
# ---------------------------------------------------------------------------


def _timed(targets, catalogs, radius, *, registry=None) -> tuple[BatchResult, float]:
    started = time.monotonic()
    result = asyncio.run(BatchCrossmatcher(registry=registry).run(targets, catalogs, radius_arcsec=radius))
    return result, time.monotonic() - started


def test_live_vlass_batch_uses_xmatch_and_answers_quickly():
    targets = [{"id": "M87", "ra": 187.7059308, "dec": 12.3911233}, {"id": "3C 273", "ra": 187.2779154, "dec": 2.0523883}]
    result, elapsed = _timed(targets, ["vlass"], 5.0)
    down = unavailable(result)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    run = result.runs["vlass"]
    assert run.strategy == "xmatch" and not run.errors and run.fallback_targets == 0, run
    assert elapsed < 120.0, elapsed
    [m87] = result.target_matches("M87")["vlass"]
    assert m87["source_id"] == "J123049.43+122328.3" and m87["separation_arcsec"] < 0.5


def test_live_lotss_batch_finishes_within_the_fast_fallback_bound():
    """LoTSS-DR3 is not in the XMatch service: TAPVizieR upload, with the small-batch fast fallback to cones."""
    targets = [{"id": "M87", "ra": 187.7059308, "dec": 12.3911233}]
    result, elapsed = _timed(targets, ["lotss"], 5.0)
    run = result.runs["lotss"]
    assert elapsed < 240.0, elapsed  # 2 x BATCH_FAST_FALLBACK_SECONDS + the cone search, never 2 x 300 s
    non_network = [e for e in run.errors if not network_text(e)]
    assert not non_network, run.errors
    if result.failures:
        pytest.skip(f"archive unavailable: {result.failures}")


def test_live_registered_vizier_table_batch_uses_xmatch():
    import json
    from pathlib import Path

    import vizier
    from models import CatalogRegistry

    data = json.loads((Path(__file__).parent / "fixtures" / "final" / "registered_i_345_gaia2.json")
                      .read_text(encoding="utf-8"))
    registry = CatalogRegistry()
    vizier.attach_definition(registry, data["name"], data["entry"])
    targets = [{"id": "M87", "ra": 187.7059308, "dec": 12.3911233}, {"id": "3C 273", "ra": 187.2779154, "dec": 2.0523883}]
    result, elapsed = _timed(targets, [data["name"]], 5.0, registry=registry)
    down = unavailable(result)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))
    run = result.runs[data["name"]]
    assert run.strategy == "xmatch" and not run.errors, run
    assert elapsed < 120.0, elapsed
    assert result.target_matches("3C 273")[data["name"]][0]["source_id"] == "3700386905605055360"
    assert result.target_matches("M87")[data["name"]][0]["source_id"] == "3907709439453756032"
