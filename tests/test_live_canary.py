"""Live canary tests against the real public archives (run with ``pytest -m live``).

Each canary target is queried once per catalog (small 10 arcsec cones, rate
limited). A test only *skips* when the archive is unreachable (network error,
timeout, HTTP 5xx); parse failures, query errors, or a missing known object fail.
Only catalogs that genuinely contain an object are asserted as hits; coverage
holes (e.g. M87 in SDSS/LoTSS, 3C 273 in VLASS) are asserted only as not-failed.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from typing import Any

import httpx
import pytest
from fixture_io import TARGETS

from crossmatch import CrossmatchService, QueryExecutor
from models import (
    CatalogRegistry,
    QueryPlan,
    catalog_from_dict,
    haversine_arcsec,
    propagate_radec,
    resolved_target,
    validate_target,
)
from providers import CacheManager, SesameResolver, provider_map

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", autouse=True)
def polite_request_rate():
    """Pace live requests at 2/s for this module only (restored afterwards, so later
    offline tests in the same session keep the conftest's fast setting)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROVIDER_REQUESTS_PER_SECOND", "2")
        yield

RADIUS_ARCSEC = 10.0
UNREACHABLE = {"CatalogUnavailableError", "QueryTimeoutError"}

# (target, catalog) -> (expected source_id of the nearest source, max separation arcsec)
EXPECTED: dict[tuple[str, str], tuple[str, float]] = {
    ("3c273", "gaia_dr3"): ("3700386905605055360", 0.05),
    ("3c273", "simbad"): ("3C 273", 0.05),
    ("3c273", "ned"): ("3C 273", 0.05),
    ("3c273", "twomass_psc"): ("12290669+0203085", 0.3),
    ("3c273", "vizier_2mass_reference"): ("12290669+0203085", 0.3),
    ("3c273", "allwise"): ("J122906.69+020308.6", 0.3),
    ("3c273", "panstarrs_dr2"): ("110461872779253351", 0.1),
    ("3c273", "sdss"): ("1237651735760142397", 0.2),
    ("3c273", "first"): ("FIRST J122906.7+020308", 1.5),
    ("3c273", "nvss"): ("NVSS J122906+020305", 8.0),  # 45" beam blends core + jet
    ("3c273", "lotss"): ("ILTJ122906.32+020304.4", 8.0),  # DR3 only; jet dominates at 144 MHz
    ("3c273", "rosat"): ("2RXS J122906.6+020309", 3.0),
    ("3c273", "rosat_bsc"): ("1RXS J122906.5+020311", 6.0),
    ("3c273", "chandra"): ("2CXO J122906.6+020308", 1.0),
    ("3c273", "xmm"): ("5XMM J122906.6+020308", 1.0),
    ("m87", "gaia_dr3"): ("3907709439453756032", 0.1),
    ("m87", "simbad"): ("M 87", 0.05),
    ("m87", "ned"): ("Messier 087", 0.05),
    ("m87", "twomass_psc"): ("12304942+1223278", 0.5),
    ("m87", "allwise"): ("J123049.43+122328.0", 0.5),
    ("m87", "panstarrs_dr2"): ("122861877059169881", 0.3),
    ("m87", "first"): ("FIRST J123049.3+122323", 6.0),  # extended: lobe component
    ("m87", "nvss"): ("NVSS J123049+122321", 8.0),
    ("m87", "vlass"): ("J123049.43+122328.3", 1.0),
    ("m87", "rosat"): ("2RXS J123049.9+122323", 10.0),  # cluster-core emission
    ("m87", "chandra"): ("2CXO J123049.4+122327", 1.0),
    ("m87", "xmm"): ("5XMM J123049.3+122330", 4.0),
    ("hd106785", "gaia_dr3"): ("3908454259797359232", 0.1),
    ("hd106785", "simbad"): ("HD 106785", 0.05),
    ("hd106785", "twomass_psc"): ("12164896+1232209", 0.3),
    ("hd106785", "allwise"): ("J121648.96+123221.0", 0.5),
    ("hd209458", "exoplanet_archive"): ("HD 209458 b", 1.0),
    ("hd209458", "simbad"): ("HD 209458", 1.0),
    ("hd209458", "gaia_dr3"): ("1779546757669063552", 1.0),
    ("hd209458", "twomass_psc"): ("22031077+1853036", 1.0),
    ("hd209458", "allwise"): ("J220310.79+185303.3", 1.0),
}

_RESULTS: dict[str, dict[str, Any]] = {}


async def _query_all(target_key: str) -> dict[str, Any]:
    registry = CatalogRegistry()
    names = sorted({c for (t, c) in EXPECTED if t == target_key} | set(registry.enabled_catalogs()))
    plans = [
        QueryPlan(n, registry.get(n).provider, registry.get(n).endpoint, {}, RADIUS_ARCSEC, registry.get(n).wavelength)
        for n in names
    ]
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        executor = QueryExecutor(provider_map(client, timeout=120.0, cache=CacheManager(None)), timeout=150.0, registry=registry)
        successes, failures = await executor.execute(plans, validate_target(*TARGETS[target_key]))
    return {"ok": dict(successes), "failed": {f.catalog: f for f in failures}}


def results_for(target_key: str) -> dict[str, Any]:
    if target_key not in _RESULTS:
        _RESULTS[target_key] = asyncio.run(_query_all(target_key))
    return _RESULTS[target_key]


@pytest.mark.parametrize(("target", "catalog"), sorted(EXPECTED))
def test_canary_object_found(target: str, catalog: str) -> None:
    results = results_for(target)
    failure = results["failed"].get(catalog)
    if failure is not None:
        if failure.error_type in UNREACHABLE:
            pytest.skip(f"{catalog} unreachable: {failure.error_type}: {failure.message}")
        pytest.fail(f"{catalog} failed: {failure.error_type}: {failure.message}")
    sources = results["ok"][catalog]
    fallback = sources.meta.get("fallback")
    if fallback is not None:
        # The primary archive (e.g. IRSA TAP) failed and the fallback answered: this canary
        # checks the primary, so report the outage instead of passing on the fallback.
        pytest.skip(f"{catalog}: primary archive unavailable, answered by fallback "
                    f"{fallback.get('provider')} ({fallback.get('reason')})")
    expected_id, max_sep = EXPECTED[(target, catalog)]
    assert sources, f"{catalog} returned no rows for {target} (status={sources.meta.get('status')})"
    nearest = sources[0]
    sep = nearest.metadata["query_separation_arcsec"]
    # Rows coincident with the nearest (e.g. SIMBAD host star + planets at 0") are ties;
    # the expected object must be among them (ties are also ordered star-first).
    tied = [s.source_id for s in sources if s.metadata["query_separation_arcsec"] - sep < 1e-3]
    assert expected_id in tied, f"nearest is {tied} at {sep:.3f}\""
    assert nearest.source_id == expected_id, f"tie order put {nearest.source_id} first"
    assert sep <= max_sep
    assert nearest.positional_error_arcsec is not None
    # NED epoch only when pos_bibcode is a known survey; CSC/NVSS/LoTSS/1RXS rows have an epoch span.
    assert nearest.epoch is not None or nearest.epoch_range is not None or catalog == "ned"
    assert all(s.metadata["query_separation_arcsec"] <= RADIUS_ARCSEC + 1e-6 for s in sources)


def _ok(target: str, catalog: str):
    results = results_for(target)
    failure = results["failed"].get(catalog)
    if failure is not None:
        if failure.error_type in UNREACHABLE:
            pytest.skip(f"{catalog} unreachable: {failure.error_type}")
        pytest.fail(f"{catalog} failed: {failure.error_type}: {failure.message}")
    return results["ok"][catalog]


def test_proper_motions_survive_the_query() -> None:
    # Guards against a query regression silently dropping pm columns.
    gaia = _ok("hd209458", "gaia_dr3")[0]
    assert gaia.proper_motion_ra_masyr == pytest.approx(29.0, abs=3) and gaia.proper_motion_dec_masyr is not None
    simbad = _ok("hd209458", "simbad")[0]
    assert simbad.proper_motion_ra_masyr is not None and simbad.proper_motion_dec_masyr is not None
    planet = _ok("hd209458", "exoplanet_archive")[0]
    assert planet.data.get("sy_pmra") is not None and planet.proper_motion_ra_masyr == pytest.approx(planet.data["sy_pmra"])
    assert planet.epoch == 2015.5


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_no_catalog_fails_for_canary(target: str) -> None:
    results = results_for(target)
    hard = {n: f"{f.error_type}: {f.message}" for n, f in results["failed"].items() if f.error_type not in UNREACHABLE}
    assert hard == {}, hard
    unreachable = [n for n, f in results["failed"].items() if f.error_type in UNREACHABLE]
    for name, sources in results["ok"].items():
        assert sources.meta["status"] in {"success", "empty"}, name
        assert sources.meta["row_count"] == len(sources)
    if unreachable:
        pytest.skip(f"unreachable archives (not failures): {unreachable}")


def _run(coro):
    try:
        return asyncio.run(coro)
    except Exception as exc:  # network trouble only
        if "HTTP 5" in str(exc) or isinstance(exc.__cause__, httpx.HTTPError) or isinstance(exc, httpx.HTTPError):
            pytest.skip(f"archive unreachable: {exc}")
        raise


async def _crossmatch(catalogs: list[str], ra: float, dec: float, radius: float, **kwargs):
    registry = CatalogRegistry()
    registry._catalogs = {k: v for k, v in registry._catalogs.items() if k in catalogs}
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = CrossmatchService(registry, provider_map(client, timeout=120.0, cache=CacheManager(None)), timeout=150.0)
        return await service.crossmatch(ra, dec, radius_arcsec=radius, **kwargs)


def _hits(record, catalog: str) -> list[str]:
    result = record.catalog_results[catalog]
    if result["status"] == "failed":
        if result["error_type"] in UNREACHABLE:
            pytest.skip(f"{catalog} unreachable: {result['message']}")
        pytest.fail(f"{catalog} failed: {result['error_type']}: {result['message']}")
    return [s.source_id for s in result["sources"]]


BARNARD_GAIA = (269.44850252543836, 4.739420051112412)  # Gaia DR3 4472832130942575872, J2016.0
BARNARD_SIMBAD = (269.45207696, 4.69336497)  # J2000


def test_barnard_with_epoch_2016_finds_2mass_simbad_allwise() -> None:
    record = _run(_crossmatch(["gaia_dr3", "simbad", "twomass_psc", "allwise", "exoplanet_archive"],
                              *BARNARD_GAIA, 10.0, epoch=2016.0))
    assert _hits(record, "gaia_dr3")[0] == "4472832130942575872"
    assert _hits(record, "simbad")[0] == "NAME Barnard's star"
    assert _hits(record, "twomass_psc")[0] == "17574849+0441405"
    assert _hits(record, "allwise")[0] == "J175747.94+044323.8"
    assert set(_hits(record, "exoplanet_archive")) >= {"Barnard b"}


def test_barnard_at_simbad_j2000_finds_gaia_not_a_field_star() -> None:
    record = _run(_crossmatch(["gaia_dr3", "exoplanet_archive"], *BARNARD_SIMBAD, 10.0, epoch=2000.0))
    assert _hits(record, "gaia_dr3")[0] == "4472832130942575872"
    assert "Barnard b" in _hits(record, "exoplanet_archive")


def test_proxima_planets_found_from_simbad_j2000_name_resolution() -> None:
    async def run():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resolved = await SesameResolver(client).resolve("Proxima Cen")
        target = resolved_target(resolved)
        assert target.epoch == 2000.0 and target.proper_motion is not None
        with_pm = await _crossmatch(["exoplanet_archive"], target.ra, target.dec, 5.0, epoch=target.epoch,
                                    pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr)
        without_pm = await _crossmatch(["exoplanet_archive"], target.ra, target.dec, 5.0, epoch=target.epoch)
        return with_pm, without_pm

    with_pm, without_pm = _run(run())
    for record in (with_pm, without_pm):  # 59.8" from J2000 to J2015.5
        assert "Proxima Cen b" in _hits(record, "exoplanet_archive")
        assert record.catalog_results["exoplanet_archive"]["status"] == "success"


def test_sesame_real_structure_barnard() -> None:
    async def resolve():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            return await SesameResolver(client).resolve("Barnard's star")

    resolved = _run(resolve())
    assert resolved.pm_dec_masyr == pytest.approx(10362.4, abs=5)
    assert resolved.pm_ra_masyr == pytest.approx(-801.6, abs=5)
    assert resolved.epoch == 2000.0
    assert resolved.resolver_metadata["parallax_mas"] == pytest.approx(547.0, abs=2)


# ---------------------------------------------------------------------------
# Positional-error calibration against Gaia (catches unit / factor mistakes)
# ---------------------------------------------------------------------------

# Rayleigh median of offset / sigma for a 2-D Gaussian with per-axis sigma.
RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))  # 1.1774


async def _calibration(catalog_name: str, ra: float, dec: float, radius: float, max_rows: int = 1000) -> list[float]:
    registry = CatalogRegistry()

    def widened(name: str, rows: int):
        base = registry.get(name)
        return catalog_from_dict(name, {**base.as_dict(), "max_rows": rows})

    target = validate_target(ra, dec)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
        cat = widened(catalog_name, max_rows)
        gaia_def = widened("gaia_dr3", 5000)
        rows = await providers[cat.provider].query(cat, target, radius)
        gaia = await providers["tap"].query(gaia_def, target, radius + 3.0)
    ratios: list[float] = []
    for src in rows:
        # Rows without a per-source date (CSC) are compared at mid-span: field stars move
        # a few mas/yr, negligible against their arcsec-level X-ray errors.
        epoch = src.epoch if src.epoch is not None else (sum(src.epoch_range) / 2 if src.epoch_range else None)
        if src.positional_error_arcsec is None or epoch is None:
            continue
        best = None
        for g in gaia:
            if g.proper_motion_ra_masyr is None or g.positional_error_arcsec is None:
                continue
            gra, gdec = propagate_radec(g.ra, g.dec, g.proper_motion_ra_masyr, g.proper_motion_dec_masyr, g.epoch, epoch)
            sep = haversine_arcsec(src.ra, src.dec, gra, gdec)
            if best is None or sep < best[0]:
                best = (sep, g)
        if best is None:
            continue
        sep, g = best
        sigma = math.hypot(src.positional_error_arcsec, g.positional_error_arcsec)
        if sep <= max(1.0, 4.0 * sigma):  # likely the same object
            ratios.append(sep / sigma)
    return ratios


# Observed medians of offset/sigma against Gaia (2026-09; honest 1-sigma errors give
# RAYLEIGH_MEDIAN = 1.18). Values != 1.18 are the catalogs' own (documented) conservative
# or optimistic errors in these fields; what the test pins is that OUR conversion keeps
# reproducing them. The band is x/÷ 1.35 around the recorded median (sampling scatter
# of a median of 15-200 pairs is ~10-20%), so a factor-of-sqrt(2) (1.41), 95%-vs-1-sigma
# (1.96/2.45) or unit (x60, x1000) mistake fails.
CALIBRATION_MEDIANS: dict[str, tuple[tuple[float, float], float, float]] = {
    "panstarrs_dr2": ((132.8250, 11.8000), 90.0, 0.481),  # M67 (PS1 errors conservative for bright stars)
    "twomass_psc": ((132.8250, 11.8000), 90.0, 1.33),  # M67
    "allwise": ((132.8250, 11.8000), 90.0, 1.84),  # M67 (sigra/sigdec optimistic for bright stars)
    "sdss": ((132.8250, 11.8000), 90.0, 0.824),  # M67
    "chandra": ((83.8186, -5.3897), 120.0, 0.296),  # ONC (CSC errors conservative in crowded fields)
    "xmm": ((83.8186, -5.3897), 120.0, 1.78),  # ONC
}
CALIBRATION_BAND = 1.35
# Not calibrated here: FIRST/NVSS/VLASS/LoTSS (too few Gaia counterparts -- mostly
# quasars -- in any field small enough for a polite query) and 2RXS (1 pair in the ONC).


@pytest.mark.parametrize("catalog", sorted(CALIBRATION_MEDIANS))
def test_positional_errors_are_calibrated_against_gaia(catalog: str) -> None:
    field, radius, expected = CALIBRATION_MEDIANS[catalog]
    ratios = _run(_calibration(catalog, *field, radius))
    if len(ratios) < 10:
        pytest.skip(f"only {len(ratios)} {catalog}-Gaia pairs")
    median = statistics.median(ratios)
    assert expected / CALIBRATION_BAND <= median <= expected * CALIBRATION_BAND, (
        f"{catalog}: median offset/sigma {median:.3f} over {len(ratios)} pairs, recorded {expected}"
    )


def test_sesame_resolves_m87() -> None:
    async def resolve():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            return await SesameResolver(client).resolve("M87")

    try:
        resolved = asyncio.run(resolve())
    except Exception as exc:  # network trouble only
        if "HTTP 5" in str(exc) or isinstance(exc.__cause__, httpx.HTTPError):
            pytest.skip(f"Sesame unreachable: {exc}")
        raise
    assert abs(resolved.ra_deg - 187.7059308) < 1e-3
    assert abs(resolved.dec_deg - 12.3911233) < 1e-3


# ---------------------------------------------------------------------------
# Round-2 regressions against the live archives
# ---------------------------------------------------------------------------


def test_barnard_every_catalog_same_result_with_given_or_adopted_motion() -> None:
    from test_epoch_cones import BARNARD_EMPTY, BARNARD_HITS, BARNARD_PM

    catalogs = sorted(set(BARNARD_HITS) | BARNARD_EMPTY)
    given = _run(_crossmatch(catalogs, *BARNARD_SIMBAD, 10.0, epoch=2000.0,
                             pm_ra_masyr=BARNARD_PM[0], pm_dec_masyr=BARNARD_PM[1]))
    adopted = _run(_crossmatch(catalogs, *BARNARD_SIMBAD, 10.0, epoch=2000.0))
    for record in (given, adopted):
        for name, expected in BARNARD_HITS.items():
            assert expected <= set(_hits(record, name)), name
    for name in catalogs:
        if "failed" in (given.catalog_results[name]["status"], adopted.catalog_results[name]["status"]):
            continue
        assert set(_hits(given, name)) == set(_hits(adopted, name)), name


@pytest.mark.parametrize(("catalog", "ra", "dec", "pm", "expected"), [
    ("chandra", 217.42894222, -62.67949019, (-3781.741, 769.465), "2CXO J142942.7-624046"),  # Proxima Cen
    ("lotss", 177.74050219, 48.37737811, (-1545.069, -962.724), "ILTJ115055.51+482224.3"),  # GJ 1151
])
def test_high_pm_star_in_catalog_without_row_epochs(catalog, ra, dec, pm, expected) -> None:
    record = _run(_crossmatch([catalog], ra, dec, 5.0, epoch=2000.0, pm_ra_masyr=pm[0], pm_dec_masyr=pm[1]))
    assert expected in _hits(record, catalog)


def test_provider_cache_replays_gzip_archives(monkeypatch) -> None:
    # VizieR TAP and the Exoplanet Archive gzip their answers; the second identical query
    # must come from the cache and parse.
    monkeypatch.setenv("PROVIDER_CACHE_TTL_SECONDS", "600")
    monkeypatch.setenv("PROVIDER_REQUESTS_PER_SECOND", "2")

    async def twice():
        registry = CatalogRegistry()
        cache = CacheManager(None)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            providers = provider_map(client, timeout=120.0, cache=cache)
            out = []
            for name, target in (("vlass", TARGETS["m87"]), ("exoplanet_archive", TARGETS["hd209458"])):
                cat = registry.get(name)
                first = await providers[cat.provider].query(cat, validate_target(*target), 10.0)
                second = await providers[cat.provider].query(cat, validate_target(*target), 10.0)
                out.append((name, first, second))
            return out

    for name, first, second in _run(twice()):
        assert first.meta["cached"] is False and second.meta["cached"] is True, name
        assert [s.source_id for s in second] == [s.source_id for s in first] and len(first) > 0, name


def test_sesame_3c273_is_stationary() -> None:
    async def resolve():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            return await SesameResolver(client).resolve("3C 273")

    resolved = _run(resolve())
    assert resolved.resolver_metadata["extragalactic"] is True
    assert resolved_target(resolved).proper_motion == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Round-3 regressions against the live archives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["twomass_psc", "allwise"])
def test_gator_fallback_live_matches_irsa_tap(name: str) -> None:
    """The Gator fallback must keep working: force IRSA TAP 'down' and compare with TAP."""
    from models import IRSA_TAP

    async def run():
        registry = CatalogRegistry()
        catalog = registry.get(name)
        plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, RADIUS_ARCSEC, catalog.wavelength)
        real = httpx.AsyncHTTPTransport()

        class PrimaryDown(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if str(request.url).startswith(IRSA_TAP):
                    return httpx.Response(503, text="simulated outage", request=request)
                return await real.handle_async_request(request)

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, transport=PrimaryDown()) as client:
            executor = QueryExecutor(provider_map(client, timeout=120.0, cache=CacheManager(None)), timeout=150.0,
                                     registry=registry)
            fb_ok, fb_failed = await executor.execute([plan], validate_target(*TARGETS["3c273"]))
        return fb_ok, fb_failed

    fb_ok, fb_failed = _run(run())
    if fb_failed:
        failure = fb_failed[0]
        if failure.error_type in UNREACHABLE:
            pytest.skip(f"Gator unreachable: {failure.message}")
        pytest.fail(f"Gator fallback failed: {failure.error_type}: {failure.message}")
    (_, gator), = fb_ok
    assert gator.meta["fallback"]["provider"] == "irsa_gator"
    tap = _ok("3c273", name)
    if tap.meta.get("fallback"):
        pytest.skip("IRSA TAP itself is down; nothing to compare with")
    assert [s.source_id for s in gator] == [s.source_id for s in tap]
    for g, t in zip(gator, tap):
        assert g.metadata["query_separation_arcsec"] == pytest.approx(t.metadata["query_separation_arcsec"], abs=1e-3)
        assert g.positional_error_arcsec == pytest.approx(t.positional_error_arcsec, rel=1e-6)
        assert g.epoch == pytest.approx(t.epoch, abs=1e-6)


def _resolved(name: str):
    async def resolve():
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            return await SesameResolver(client).resolve(name)

    return resolved_target(_run(resolve()))


def test_proxima_5xmm_2mass_and_gaia_from_name_resolution() -> None:
    # 5XMM stack spans (not midpoints), 768 mas parallax for 2MASS, Gaia 2-parameter
    # field stars kept at their own positions -- as search_object('Proxima Cen') runs.
    target = _resolved("Proxima Cen")
    assert target.parallax_mas == pytest.approx(768.07, abs=0.5)
    record = _run(_crossmatch(["xmm", "twomass_psc", "gaia_dr3"], target.ra, target.dec, 3.0, epoch=target.epoch,
                              pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr,
                              parallax_mas=target.parallax_mas, pm_source="resolver"))
    xmm = set(_hits(record, "xmm"))
    assert {"5XMM J142933.5-624032", "5XMM J142941.9-624044", "5XMM J142937.8-624040"} <= xmm
    twomass = record.catalog_results["twomass_psc"]["sources"]
    assert twomass and twomass[0].source_id == "14294291-6240465"
    assert twomass[0].metadata["epoch_separation_arcsec"] < 0.15
    matched_gaia = {m["source_id"] for m in record.provenance["matches"] if m["catalog"] == "gaia_dr3"}
    assert "5853498713190524032" not in matched_gaia  # 2-parameter star 61" from Proxima at J2000


def test_kapteyn_5xmm_detections_are_not_dropped() -> None:
    target = _resolved("Kapteyn's star")
    record = _run(_crossmatch(["xmm"], target.ra, target.dec, 3.0, epoch=target.epoch,
                              pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr))
    hits = set(_hits(record, "xmm"))
    assert {"5XMM J051152.5-450257", "5XMM J051153.5-450307"} <= hits
    for src in record.catalog_results["xmm"]["sources"]:
        if src.source_id in {"5XMM J051152.5-450257", "5XMM J051153.5-450307"}:
            assert src.metadata["epoch_separation_arcsec"] < 1.0, src.source_id


def test_vlass_redundant_duplicate_removed_live() -> None:
    record = _run(_crossmatch(["vlass"], 43.8634477, 5.9886937, 5.0))
    result = record.catalog_results["vlass"]
    ids = _hits(record, "vlass")
    assert ids == ["J025527.22+055919.5"] and result["row_count"] == 1
    assert result["sources"][0].data["Flag"] == 1


def test_sgr_a_star_adoption_is_refused_in_the_crowded_field() -> None:
    record = _run(_crossmatch(["simbad"], 266.4168, -29.0078, 30.0, epoch=2016.0))
    if record.catalog_results["simbad"]["status"] == "failed":
        _hits(record, "simbad")  # skips/fails with the archive's error
    assert record.provenance["target_proper_motion"] is None
    assert any("not adopted" in w and "ambiguous" in w for w in record.provenance["warnings"])
