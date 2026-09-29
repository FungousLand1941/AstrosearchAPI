"""Live tests for batch.py against the real archives (run with ``-m live``; add ``-s`` to see the benchmark).

* 200 random-ish targets that include 3C 273, M87, HD 209458, Vega (J2000 + proper motion) and Vega-adjacent
  positions, matched with Gaia DR3 (CDS XMatch), SIMBAD (TAP upload), 2MASS PSC / AllWISE (IRSA TAP upload)
  and FIRST / NVSS (HEASARC TAP upload): the known objects must be identified correctly, with one request
  per catalog (per XMatch radius group, plus transient retries).
* The batch answers must equal per-target cone searches for a sample of 10 targets (same ids, separations
  within 0.01 arcsec).
* Benchmark: 1000 targets in batch mode versus the cone approach (measured on 100 of the targets,
  extrapolated linearly -- the cone path is paced per endpoint, so its cost is linear in the target count).

Only network failures and HTTP 5xx lead to a skip; every astrophysical assertion is strict.
"""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pytest

from batch import BatchCrossmatcher, BatchResult, format_report

pytestmark = pytest.mark.live

CATALOGS = ["gaia_dr3", "simbad", "twomass_psc", "allwise", "first", "nvss"]
RADIUS = 5.0

# SIMBAD ICRS J2000 positions; HD 209458 at the NASA Exoplanet Archive J2015.5 position (pm ~ 30 mas/yr).
KNOWN = [
    {"id": "3C 273", "ra": 187.2779154, "dec": 2.0523883},
    {"id": "M87", "ra": 187.7059308, "dec": 12.3911233},
    {"id": "HD 209458", "ra": 330.79502, "dec": 18.88432},
    {"id": "Vega", "ra": 279.23473479, "dec": 38.78368896, "epoch": 2000.0, "pm_ra_masyr": 200.94, "pm_dec_masyr": 286.23},
    {"id": "Vega+60N", "ra": 279.23473479, "dec": 38.78368896 + 60.0 / 3600.0},
    {"id": "Vega+90E", "ra": 279.23473479 + 90.0 / 3600.0 / math.cos(math.radians(38.78368896)), "dec": 38.78368896},
]

# Nearest-match identities (radius 5"), from the archives' own designations.
EXPECTED = {
    ("3C 273", "gaia_dr3"): "3700386905605055360",
    ("3C 273", "simbad"): "3C 273",
    ("3C 273", "twomass_psc"): "12290669+0203085",
    ("3C 273", "allwise"): "J122906.69+020308.6",
    ("3C 273", "first"): "FIRST J122906.7+020308",
    ("M87", "simbad"): "M 87",
    ("M87", "gaia_dr3"): "3907709439453756032",
    ("M87", "twomass_psc"): "12304942+1223278",
    ("M87", "allwise"): "J123049.43+122328.0",
    ("M87", "first"): "FIRST J123049.3+122323",
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
    # 2MASS Vega is matched after moving the 1999.3 detection with Vega's proper motion.
    assert result.target_matches("Vega")["twomass_psc"][0]["epoch_propagation"] == "target_pm"
    # One request per catalog chunk: 200 targets fit in one chunk; Gaia needs one XMatch request per cone
    # radius (plain 5" and Vega's epoch-widened cone). Retries are transient connection resets.
    for name, run in result.runs.items():
        assert run.strategy == ("xmatch" if name == "gaia_dr3" else "upload"), name
        assert run.chunks == (2 if name == "gaia_dr3" else 1), name
        assert run.requests == run.chunks + run.retries, name
        assert not run.errors, (name, run.errors)
    # Canonical fields on every match.
    for item in result.as_dict(include_data=False)["targets"]:
        for name, matches in item["matches"].items():
            for m in matches:
                assert m["separation_arcsec"] <= RADIUS + 1e-6
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
    compared = 0
    for item in sample:
        up = batch200.target_matches(item.id)
        cs = cone.target_matches(item.id)
        for catalog in CATALOGS:
            a = {m["source_id"]: m["separation_arcsec"] for m in up.get(catalog, [])}
            b = {m["source_id"]: m["separation_arcsec"] for m in cs.get(catalog, [])}
            assert set(a) == set(b), (item.id, catalog, sorted(a), sorted(b))
            for sid in a:
                assert abs(a[sid] - b[sid]) < 0.01, (item.id, catalog, sid, a[sid], b[sid])
                compared += 1
    assert compared >= 20
    assert cone.request_count >= len(sample) * len(CATALOGS)


def test_live_benchmark_1000_targets_vs_cone():
    targets = KNOWN + random_targets(1000 - len(KNOWN), seed=4)
    started = time.perf_counter()
    bulk = run_batch(targets, CATALOGS, fallback=False)
    bulk_wall = time.perf_counter() - started
    down = unavailable(bulk)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))

    sample = targets[:100]
    started = time.perf_counter()
    cone = run_batch(sample, CATALOGS, strategies={c: "cone" for c in CATALOGS})
    cone_wall = time.perf_counter() - started
    down = unavailable(cone)
    if down:
        pytest.skip("archive unavailable: " + "; ".join(down))

    scale = len(targets) / len(sample)
    cone_requests_1000 = cone.request_count * scale
    cone_wall_1000 = cone_wall * scale
    report = [
        "", format_report(bulk), format_report(cone),
        f"BENCHMARK 1000 targets x {len(CATALOGS)} catalogs, radius {RADIUS:g}\":",
        f"  batch : {bulk.request_count} requests, {bulk_wall:.1f} s wall",
        f"  cone  : {cone.request_count} requests, {cone_wall:.1f} s for {len(sample)} targets -> "
        f"~{cone_requests_1000:.0f} requests, ~{cone_wall_1000:.0f} s for {len(targets)} (linear extrapolation)",
        f"  speedup: {cone_requests_1000 / bulk.request_count:.0f}x fewer requests, {cone_wall_1000 / bulk_wall:.1f}x faster",
    ]
    print("\n".join(report))

    # The same answers for the 100 targets measured both ways.
    for item in sample:
        for catalog in CATALOGS:
            a = {m["source_id"]: m["separation_arcsec"] for m in bulk.target_matches(item["id"]).get(catalog, [])}
            b = {m["source_id"]: m["separation_arcsec"] for m in cone.target_matches(item["id"]).get(catalog, [])}
            assert set(a) == set(b), (item["id"], catalog)
            assert all(abs(a[s] - b[s]) < 0.01 for s in a)
    assert bulk.request_count <= len(CATALOGS) + 1 + sum(r.retries for r in bulk.runs.values())
    assert cone.request_count >= len(sample) * len(CATALOGS)
    assert cone_wall_1000 / bulk_wall > 3.0
