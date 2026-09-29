"""Epoch-aware cones, pad rows, proper-motion propagation/adoption, and the Sesame resolver.

High proper-motion regression targets are Barnard's star (10.39"/yr) recorded with a
target epoch but *no* proper motion (tests/fixtures/barnard_*), so every catalog
cone is widened and the pad rows must never be reported as results.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fixture_io import (
    EPOCH_TARGET_CATALOG_SETS,
    FIXTURES,
    TARGETS,
    load_exchanges,
    replay_side_effect,
    target_for,
    target_radius,
)
from helpers import make_service, offline_client, run_catalog
from test_matching_fixtures import assert_replayed_exactly

from crossmatch import AdvancedQuery, CrossmatchService, _adopt_proper_motion, match_target
from models import (
    CatalogRegistry,
    CatalogSource,
    catalog_epoch_range,
    catalog_from_dict,
    closest_span_epoch,
    epoch_separation_arcsec,
    haversine_arcsec,
    plan_cone,
    propagate_radec,
    resolved_target,
    validate_target,
)
from providers import CacheManager, SesameResolver, provider_map

BARNARD_GAIA_ID = "4472832130942575872"
BARNARD_PM = (-801.551, 10362.394)  # SIMBAD / Gaia DR3, mas/yr


def hd209458_b_position() -> tuple[float, float]:
    """pscomppars (J2015.5) position of HD 209458 b from the recorded response."""
    from models import parse_json_records

    row = parse_json_records((FIXTURES / "hd209458" / "exoplanet_archive.0.body").read_bytes())[0]
    return float(row["ra"]), float(row["dec"])


def epoch_registry(key: str = "barnard_j2000") -> CatalogRegistry:
    reg = CatalogRegistry()
    reg._catalogs = {k: v for k, v in reg._catalogs.items() if k in EPOCH_TARGET_CATALOG_SETS[key]}
    return reg


async def crossmatch_fixture(key: str, **kwargs):
    """Crossmatch a recorded epoch target (with its proper motion when the fixture has one)."""
    target = target_for(key)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(key)))
        async with offline_client() as client:
            record = await make_service(client, epoch_registry(key)).crossmatch(
                target.ra, target.dec, radius_arcsec=target_radius(key), epoch=target.epoch,
                pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr, **kwargs)
    assert_replayed_exactly(record)
    return record


# ---------------------------------------------------------------------------
# Propagation helpers (checked against archive-published positions)
# ---------------------------------------------------------------------------


def test_propagation_matches_exoplanet_archive_j2015_5_position() -> None:
    # Gaia DR3 (J2016.0) propagated to J2015.5 must land on pscomppars' Barnard b
    # position (269.4486144, 4.7379808), which the archive derives the same way.
    ra, dec = propagate_radec(269.44850252543836, 4.739420051112412, *BARNARD_PM, 2016.0, 2015.5)
    assert haversine_arcsec(ra, dec, 269.4486144, 4.7379808) < 0.005
    # ... and SIMBAD J2000 -> J2016.0 lands on Gaia DR3 within a few mas.
    ra, dec = propagate_radec(269.45207696, 4.69336497, *BARNARD_PM, 2000.0, 2016.0)
    assert haversine_arcsec(ra, dec, 269.44850252543836, 4.739420051112412) < 0.01


def test_propagation_is_stable_near_the_pole() -> None:
    ra, dec = propagate_radec(10.0, 89.9999, 0.0, 1000.0, 2000.0, 2010.0)  # 10" north across the pole
    assert -90.0 <= dec <= 90.0 and 0.0 <= ra < 360.0
    assert haversine_arcsec(10.0, 89.9999, ra, dec) == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Cone planning
# ---------------------------------------------------------------------------


def test_no_epoch_means_exact_cone(registry: CatalogRegistry) -> None:
    plan = plan_cone(registry.get("exoplanet_archive"), validate_target(10.0, 20.0), 5.0)
    assert (plan.radius_arcsec, plan.pad_arcsec, plan.mode, plan.row_limit) == (5.0, 0.0, "none", 200)


def test_unknown_proper_motion_widens_by_max_pm_times_gap(registry: CatalogRegistry, monkeypatch) -> None:
    target = validate_target(10.0, 20.0, epoch=2000.0)
    plan = plan_cone(registry.get("exoplanet_archive"), target, 5.0)  # J2015.5
    assert plan.mode == "unknown_pm_pad"
    assert plan.pad_arcsec == pytest.approx(10.5 * 15.5)  # covers Barnard's star (10.39"/yr)
    assert plan.row_limit == 2000  # a fast mover can be beyond the 200 nearest rows
    twomass = plan_cone(registry.get("twomass_psc"), target, 5.0)
    assert catalog_epoch_range(registry.get("twomass_psc")) == (1997.4, 2001.2)
    assert twomass.pad_arcsec == pytest.approx(10.5 * 2.6)
    monkeypatch.setenv("EPOCH_PAD_MAX_ARCSEC", "60")
    capped = plan_cone(registry.get("exoplanet_archive"), target, 5.0)
    assert capped.pad_arcsec == 60.0 and "capped" in capped.warnings[0]


def test_known_proper_motion_moves_cone_and_covers_the_epoch_span(registry: CatalogRegistry) -> None:
    target = validate_target(269.45207696, 4.69336497, epoch=2000.0, pm_ra_masyr=BARNARD_PM[0], pm_dec_masyr=BARNARD_PM[1])
    gaia_j2016 = (269.44850252543836, 4.739420051112412)
    # Gaia has per-row proper motions: the cone must enclose the target at J2016 (5")
    # AND the target's field-star neighbours at J2000 (5" + 0.1"/yr x 16 yr).
    single = plan_cone(registry.get("gaia_dr3"), target, 5.0)
    assert single.mode == "target_pm"
    assert haversine_arcsec(single.ra, single.dec, *gaia_j2016) + 5.0 <= single.radius_arcsec + 1e-3
    assert haversine_arcsec(single.ra, single.dec, target.ra, target.dec) + 5.0 + 1.6 <= single.radius_arcsec + 1e-3
    assert single.radius_arcsec == pytest.approx(5.0 + (166.2 + 1.6) / 2, abs=0.2)  # minimal enclosing circle
    # 2MASS has no proper motions: centred on the track, padded by half the 1997.4-2001.2 path.
    span = plan_cone(registry.get("twomass_psc"), target, 5.0)
    assert span.pad_arcsec == pytest.approx(10.3933 * (2001.2 - 1997.4) / 2, rel=1e-3)
    track_mid = propagate_radec(target.ra, target.dec, *BARNARD_PM, 2000.0, 1999.3)
    assert haversine_arcsec(span.ra, span.dec, *track_mid) < 0.01


def test_multi_year_catalogs_without_row_epochs_are_padded_over_their_span(registry: CatalogRegistry) -> None:
    # CSC master sources, NVSS, LoTSS and 1RXS have no per-source date: the epoch is the
    # survey span, never a single guessed year (CSC used to be pinned to 2010.0).
    spans = {"chandra": (1999.5, 2022.0), "nvss": (1993.7, 1997.3), "lotss": (2014.3, 2024.5),
             "lotss_dr2": (2014.3, 2020.2), "rosat_bsc": (1990.5, 1991.1)}
    target = validate_target(217.42894222, -62.67949019, epoch=2000.0, pm_ra_masyr=-3781.741, pm_dec_masyr=769.465)
    for name, span in spans.items():
        catalog = registry.get(name)
        assert catalog.epoch is None and catalog_epoch_range(catalog) == span, name
        plan = plan_cone(catalog, target, 5.0)
        assert plan.pad_arcsec == pytest.approx(3.8592 * (span[1] - span[0]) / 2, rel=1e-3), name
    unknown_pm = plan_cone(registry.get("chandra"), validate_target(10.0, 20.0, epoch=2000.0), 10.0)
    assert unknown_pm.pad_arcsec == pytest.approx(10.5 * 22.0)  # |2022.0 - 2000|


def test_catalog_without_epoch_warns(registry: CatalogRegistry) -> None:
    catalog = catalog_from_dict("x", {"provider": "tap", "wavelength": "x", "table": "t", "epoch": "obs_col",
                                      "parameters": {"columns": ["id", "ra", "dec", "obs_col"]}})
    plan = plan_cone(catalog, validate_target(1.0, 2.0, epoch=2016.0), 5.0)
    assert plan.mode == "epoch_unknown" and "not epoch-propagated" in plan.warnings[0]


# ---------------------------------------------------------------------------
# Pad rows never count as results
# ---------------------------------------------------------------------------


def _exoplanet_route(router: respx.MockRouter) -> respx.Route:
    body = (FIXTURES / "hd209458" / "exoplanet_archive.0.body").read_bytes()
    return router.post("https://exoplanetarchive.ipac.caltech.edu/TAP/sync").respond(
        200, content=body, headers={"content-type": "application/json"})


@pytest.mark.parametrize("use_query", [False, True])
async def test_out_of_radius_rows_are_not_success(use_query: bool) -> None:
    # Reviewer repro: a 5" search 25" north of HD 209458. Even when the archive
    # returns the planet (here via a static pad), it lies 25" away: status 'empty'.
    reg = CatalogRegistry()
    base = reg.get("exoplanet_archive")
    padded = catalog_from_dict("exoplanet_archive", {**base.as_dict(), "parameters": {**base.parameters, "radius_pad_arcsec": 30.0}})
    reg._catalogs = {"exoplanet_archive": padded}
    planet_ra, planet_dec = hd209458_b_position()
    ra, dec = planet_ra, planet_dec + 25.0 / 3600.0
    with respx.mock(assert_all_called=True) as router:
        _exoplanet_route(router)
        async with offline_client() as client:
            service = CrossmatchService(reg, provider_map(client, cache=CacheManager(None)))
            query = AdvancedQuery.from_dict({"ra": ra, "dec": dec, "radius_arcsec": 5.0, "min_confidence": 0.0}) if use_query else None
            record = await service.crossmatch(ra, dec, radius_arcsec=5.0, query=query)
    result = record.catalog_results["exoplanet_archive"]
    assert result["status"] == "empty" and result["row_count"] == 0 and result["matched_count"] == 0
    assert result["sources"] == [] and result["pad_row_count"] == 1
    assert record.provenance["catalog_stats"]["exoplanet_archive"]["status"] == "empty"
    if not use_query:
        pad = result["pad_sources"][0]
        assert pad.source_id == "HD 209458 b" and pad.metadata["outside_radius"] is True
        assert pad.metadata["query_separation_arcsec"] == pytest.approx(25.0, abs=0.05)


async def test_rows_outside_a_tiny_radius_are_trimmed_even_without_pad() -> None:
    # The recorded planet is 0.28" from the fixture target: a 0.1" search is empty.
    with respx.mock(assert_all_called=True) as router:
        _exoplanet_route(router)
        async with offline_client() as client:
            record = await make_service(client).crossmatch(*TARGETS["hd209458"], radius_arcsec=0.1, profile="exoplanet")
    assert record.catalog_results["exoplanet_archive"]["status"] == "empty"
    assert record.catalog_results["exoplanet_archive"]["pad_row_count"] == 1


async def test_epoch_incomplete_flag_when_rows_cannot_be_checked() -> None:
    # Catalog without proper motions, target epoch 16 yr earlier, unknown target pm:
    # a row 100" away could be the target -> 'empty' but flagged incomplete.
    catalog = catalog_from_dict("nopm", {
        "provider": "tap", "wavelength": "x", "endpoint": "https://nopm.example/tap", "table": "t", "epoch": 2016.0,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"}, "timeout_seconds": 10.0,
    })
    reg = CatalogRegistry()
    reg._catalogs = {"nopm": catalog}
    payload = {"metadata": [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}],
               "data": [["far", 10.0, 20.0 + 100.0 / 3600.0]]}
    with respx.mock() as router:
        route = router.post("https://nopm.example/tap").respond(json=payload)
        async with offline_client() as client:
            record = await make_service(client, reg).crossmatch(10.0, 20.0, radius_arcsec=5.0, epoch=2000.0)
    sent = httpx.QueryParams(route.calls.last.request.content.decode())["QUERY"]
    assert f"{(5.0 + 10.5 * 16.0) / 3600.0:.10f}" in sent  # widened for the 16 yr gap
    result = record.catalog_results["nopm"]
    assert result["status"] == "empty" and result["epoch_incomplete"] is True
    assert "could not be epoch-checked" in result["warnings"][0]
    assert any("could not be epoch-checked" in w for w in record.provenance["warnings"])


# ---------------------------------------------------------------------------
# Barnard's star: recorded high proper-motion fixtures
# ---------------------------------------------------------------------------


# Barnard's star in every enabled catalog (independently identified: each row lies on
# the Gaia DR3 track at its own epoch). Catalog -> ids that must be in the results.
BARNARD_HITS: dict[str, set[str]] = {
    "gaia_dr3": {BARNARD_GAIA_ID},
    "simbad": {"NAME Barnard's star"},
    # NED copies both the AllWISE and the 2MASS position (the 2MASS one used to be
    # propagated from a guessed 1999.3 and dropped at 11.7").
    "ned": {"WISEA J175747.94+044323.8", "2MASS J17574849+0441405"},
    "exoplanet_archive": {"Barnard b", "Barnard c", "Barnard d", "Barnard e"},
    "twomass_psc": {"17574849+0441405"},
    "allwise": {"J175747.94+044323.8"},
    # PS1 detections lie 105-160" from J2000: beyond the 200 statically nearest rows,
    # so they were cut before the proper motion was adopted.
    "panstarrs_dr2": {"113672694490149466"},
    "sdss": {"1237668573088843078"},
    "rosat": {"2RXS J175749.5+043954"},
    # CSC saw Barnard's star in 2019.45; with the old fixed epoch 2010.0 it was missed.
    "chandra": {"2CXO J175747.4+044457"},
}
BARNARD_EMPTY = {"first", "nvss", "vlass", "lotss", "xmm"}
BARNARD_KEYS = ["barnard_j2000", "barnard_j2000_pm", "barnard_gaia2016"]


@pytest.mark.parametrize("key", BARNARD_KEYS)
async def test_barnard_found_in_every_catalog_that_observed_it(key: str) -> None:
    record = await crossmatch_fixture(key)
    results = record.catalog_results
    assert record.failures == []
    assert set(results) == set(BARNARD_HITS) | BARNARD_EMPTY
    for name, expected in BARNARD_HITS.items():
        ids = {s.source_id for s in results[name]["sources"]}
        assert expected <= ids, (name, ids)
        assert results[name]["status"] == "success", name
        for src in results[name]["sources"]:
            if src.source_id in expected:
                assert src.metadata["epoch_separation_arcsec"] < 1.0 or name == "rosat", (name, src.source_id)
    for name in BARNARD_EMPTY:
        assert results[name]["status"] == "empty", name
    for name, res in results.items():
        assert res["row_count"] == len(res["sources"])
        assert all(s.metadata["epoch_separation_arcsec"] <= 10.0 + 1e-6 for s in res["sources"])
        # Every fetched row is accounted for: nothing is dropped silently.
        assert res["raw_row_count"] == (res["row_count"] + res["pad_row_count"] + res["excess_row_count"]
                                        + res["dropped_rows"] + res["filtered_rows"]), name
    pm = record.provenance["target_proper_motion"]
    assert pm["pm_dec_masyr"] == pytest.approx(10362.394, abs=1)
    assert pm["source"] == ("input" if key.endswith("_pm") else "adopted")
    # Groups are now Bayesian associations (one row per catalogue per object) instead of
    # single-linkage chains of everything within the radius. The target group holds
    # Barnard's star in every catalogue whose row agrees with the track within its
    # quoted errors, each with a posterior > 0.99.
    target_group = record.crossmatch_groups[0]
    assert target_group["contains_target"] and target_group["match_flag"] == "best"
    assert target_group["p_any"] > 0.99
    independent = [m for m in target_group["members"] if m["coincident_with"] is None]
    assert len({m["catalog"] for m in independent}) == len(independent)
    secure = {"gaia_dr3", "simbad", "twomass_psc", "allwise", "exoplanet_archive", "chandra", "rosat"}
    assert secure <= set(target_group["catalogs"])
    for member in independent:
        # A representative stands for the rows listed with it: Pan-STARRS split the fast mover
        # into six objects along its track (0.5-1.5" from it at the target epoch), which are
        # one listing of the star now -- the known hit among them.
        listed = {member["source_id"]} | {m["source_id"] for m in target_group["members"]
                                          if m["coincident_with"] == member["source_id"]}
        assert listed & BARNARD_HITS[member["catalog"]], member["source_id"]
        # (SIMBAD and the Exoplanet Archive list the planets at the star's position.)
        assert len(listed) == 1 or member["catalog"] in ("panstarrs_dr2", "ned", "simbad", "exoplanet_archive"), listed
        if member["catalog"] in secure:
            assert member["match_probability"] > 0.99 and member["confidence"] > 0.99, member["catalog"]
    ps1 = [m for m in target_group["members"] if m["catalog"] == "panstarrs_dr2"]
    assert len(ps1) == 6 and len({m["target_probability"] for m in ps1}) == 1  # was one of them, the rest apart
    # NED lists the star twice (its WISEA and 2MASS entries, measured ~10 years and 110"
    # apart on its track): both are the star, one listing of the other -- they no longer
    # split the posterior (was 0.5 each, one of them outside the target group). NED gives
    # only the survey spans as their epochs (1997.4-2001.2: 40" of Barnard's track), so the
    # likelihood is marginalised over the span and the posterior stays below a dated row's.
    ned = [m for g in record.crossmatch_groups for m in g["members"] if m["catalog"] == "ned"
           and m["source_id"] in BARNARD_HITS["ned"]]
    assert len(ned) == 2 and all(m["target_probability"] > 0.75 for m in ned), ned
    assert ned[0]["target_probability"] == ned[1]["target_probability"]
    assert {m["coincident_with"] for m in ned} - {None} <= BARNARD_HITS["ned"]
    assert sum(m["coincident_with"] is not None for m in ned) == 1
    assert all(m in target_group["members"] for m in ned)
    # SDSS saw the star saturated (r = 11.1, clean = 0): its centroid lies 0.51" off the
    # track, 6 sigma of its quoted 0.079" error. It IS Barnard's star (BARNARD_HITS): the
    # heavy-tailed error model (6% of positions 5x off, measured on Gaia-CRF3 quasars)
    # identifies it -- a Gaussian model gave P < 0.01 -- but below a well-measured row.
    sdss = next(m for g in record.crossmatch_groups for m in g["members"]
                if m["catalog"] == "sdss" and m["source_id"] in BARNARD_HITS["sdss"])
    assert 0.5 < sdss["target_probability"] < 0.99 and sdss["data"]["psfMag_r"] < 14.0


async def test_barnard_results_do_not_depend_on_whether_the_motion_was_given_or_adopted() -> None:
    given = await crossmatch_fixture("barnard_j2000_pm")
    adopted = await crossmatch_fixture("barnard_j2000")
    for name in given.catalog_results:
        in_given = {s.source_id for s in given.catalog_results[name]["sources"]}
        in_adopted = {s.source_id for s in adopted.catalog_results[name]["sources"]}
        assert in_given == in_adopted, (name, in_given ^ in_adopted)


async def test_barnard_panstarrs_counterpart_survives_adoption_at_provider_level() -> None:
    # Provider level (no adoption yet): PS1 rows have no proper motion and the target's is
    # unknown, so the Barnard detections are pad rows -- all of them must be kept (not cut
    # to the 200 nearest) so the service can re-split them after adopting Gaia's motion.
    sources, failure = await run_catalog("panstarrs_dr2", "barnard_j2000")
    assert failure is None and sources.meta["cone"]["mode"] == "unknown_pm_pad"
    assert sources.meta["raw_row_count"] > 1000
    assert sources.meta["pad_row_count"] + sources.meta["row_count"] == sources.meta["raw_row_count"]
    assert any(s.source_id == "113672694490149466" for s in sources.meta["pad_sources"])


async def test_barnard_adopted_motion_reorders_rows_nearest_first() -> None:
    # Provider order is by static distance from J2000; after adoption, the service must
    # re-sort by epoch-corrected separation.
    record = await crossmatch_fixture("barnard_j2000")
    for name, res in record.catalog_results.items():
        seps = [s.metadata["epoch_separation_arcsec"] for s in res["sources"]]
        assert seps == sorted(seps), name
    ps1 = record.catalog_results["panstarrs_dr2"]["sources"]
    assert ps1[0].source_id == "113672694490149466"
    assert ps1[0].metadata["query_separation_arcsec"] > 100  # far in static distance, nearest after propagation


async def test_ned_2mass_position_is_matched_over_the_2mass_span() -> None:
    record = await crossmatch_fixture("barnard_j2000_pm")
    ned = {s.source_id: s for s in record.catalog_results["ned"]["sources"]}
    row = ned["2MASS J17574849+0441405"]
    assert row.epoch is None and row.epoch_range == (1997.4, 2001.2)
    assert row.metadata["epoch_propagation"] == "target_pm_span"
    assert row.metadata["epoch_separation_arcsec"] < 0.5  # 11.7" with the old point epoch 1999.3
    target = target_for("barnard_j2000_pm")
    t = closest_span_epoch(row.ra, row.dec, target.ra, target.dec, 2000.0, BARNARD_PM, row.epoch_range)
    assert t == pytest.approx(2000.40, abs=0.05)  # 2MASS PSC jdate of this star: 2000.405


@pytest.mark.parametrize(("key", "catalog", "source_id", "old_epoch"), [
    # Proxima Cen from its SIMBAD J2000 position + pm (as a name search supplies them).
    ("proxima_j2000_pm", "chandra", "2CXO J142942.7-624046", 2010.0),
    # GJ 1151 (LOFAR-detected M dwarf) in LoTSS-DR3.
    ("gj1151_j2000_pm", "lotss", "ILTJ115055.51+482224.3", 2018.0),
])
async def test_high_pm_star_found_in_catalog_without_row_epochs(key: str, catalog: str, source_id: str, old_epoch: float) -> None:
    record = await crossmatch_fixture(key)
    res = record.catalog_results[catalog]
    assert res["status"] == "success" and res["warnings"] == []
    src = next(s for s in res["sources"] if s.source_id == source_id)
    assert src.metadata["epoch_propagation"] == "target_pm_span"
    assert src.metadata["epoch_separation_arcsec"] < 1.0
    # The old single fixed epoch put it far outside the 5" radius.
    target = target_for(key)
    pinned = CatalogSource(**{**src.as_dict(), "epoch": old_epoch, "epoch_range": None})
    assert epoch_separation_arcsec(target, pinned)[0] > 5.0


async def test_pad_rows_beyond_max_rows_are_kept_until_the_motion_is_adopted() -> None:
    # Regression: epoch without pm, catalog without pm, more pad rows than max_rows, and
    # the counterpart beyond the Nth nearest. It must be found once the pm is adopted.
    pm = (0.0, 8000.0)  # 8"/yr north
    target_epoch, cat_epoch = 2000.0, 2016.0
    tra, tdec = 150.0, 10.0
    cra, cdec = propagate_radec(tra, tdec, *pm, target_epoch, cat_epoch)  # 128" north
    nopm = catalog_from_dict("nopm", {
        "provider": "tap", "wavelength": "x", "endpoint": "https://nopm.example/tap", "table": "t", "epoch": cat_epoch,
        "parameters": {"columns": ["id", "ra", "dec"], "id_field": "id", "format": "json"}, "max_rows": 5,
        "timeout_seconds": 10.0,
    })
    pmcat = catalog_from_dict("pmcat", {
        "provider": "tap", "wavelength": "x", "endpoint": "https://pmcat.example/tap", "table": "g", "epoch": cat_epoch,
        "parameters": {"columns": ["id", "ra", "dec", "pmra", "pmdec"], "id_field": "id", "format": "json"},
        "timeout_seconds": 10.0,
    })
    reg = CatalogRegistry()
    reg._catalogs = {"nopm": nopm, "pmcat": pmcat}
    meta = [{"name": "id"}, {"name": "ra", "unit": "deg"}, {"name": "dec", "unit": "deg"}]
    field = [[f"field{i}", tra + (20.0 + i) / 3600.0, tdec] for i in range(12)]  # 20-31" east: nearer than the star
    nopm_rows = field + [["star", cra, cdec]]
    pm_payload = {"metadata": meta + [{"name": "pmra", "unit": "mas/yr"}, {"name": "pmdec", "unit": "mas/yr"}],
                  "data": [["gstar", cra, cdec, pm[0], pm[1]]]}
    with respx.mock() as router:
        router.post("https://nopm.example/tap").respond(json={"metadata": meta, "data": nopm_rows})
        router.post("https://pmcat.example/tap").respond(json=pm_payload)
        async with offline_client() as client:
            record = await make_service(client, reg).crossmatch(tra, tdec, radius_arcsec=5.0, epoch=target_epoch)
    assert record.provenance["target_proper_motion"]["source_id"] == "gstar"
    res = record.catalog_results["nopm"]
    assert [s.source_id for s in res["sources"]] == ["star"]
    assert res["status"] == "success" and res["epoch_incomplete"] is False
    assert res["pad_row_count"] == 12 and len(res["pad_sources"]) == 5 and res["pad_sources_truncated"] is True
    assert res["raw_row_count"] == res["row_count"] + res["pad_row_count"] + res["excess_row_count"]


def test_adoption_treats_extragalactic_identities_as_stationary() -> None:
    target = validate_target(187.7059308, 12.3911233, epoch=2000.0)

    def row(cat, sid, **kw):
        physical = kw.pop("physical", {})
        return CatalogSource(cat, sid, 187.7059308, 12.3911233, 0.01, kw.pop("data", {}),
                             {"physical": physical}, {}, epoch=kw.pop("epoch", 2000.0), **kw)

    # M87: SIMBAD lists a Gaia proper motion of the nucleus (-8.0, 10.7 mas/yr) -- noise.
    simbad = row("simbad", "M 87", physical={"object_type": "AGN", "redshift": 0.00428},
                 proper_motion_ra_masyr=-8.029, proper_motion_dec_masyr=10.734)
    gaia = row("gaia_dr3", "3907709439453756032", epoch=2016.0, data={"parallax": 0.2, "parallax_error": 0.4},
               proper_motion_ra_masyr=-3.0, proper_motion_dec_masyr=1.0)
    result = _adopt_proper_motion(target, [("simbad", [simbad]), ("gaia_dr3", [gaia])], 10.0)
    assert result is not None
    stationary, origin = result
    assert stationary.proper_motion == (0.0, 0.0)
    assert origin["source"] == "extragalactic" and origin["source_id"] == "M 87"
    # Without the identity row, the Gaia nucleus motion (insignificant parallax, 3 mas/yr) is not adopted.
    assert _adopt_proper_motion(target, [("gaia_dr3", [gaia])], 10.0) is None
    # A star (Barnard: SIMBAD rvz_redshift -0.00037 is a radial velocity) is not extragalactic.
    star = row("simbad", "NAME Barnard's star", physical={"object_type": "BY*", "redshift": -0.000367},
               proper_motion_ra_masyr=BARNARD_PM[0], proper_motion_dec_masyr=BARNARD_PM[1])
    adopted, origin = _adopt_proper_motion(target, [("simbad", [star])], 10.0)
    assert adopted.proper_motion == BARNARD_PM and origin["source"] == "adopted"


async def test_barnard_at_gaia_j2016_with_epoch_finds_2mass_allwise_simbad() -> None:
    record = await crossmatch_fixture("barnard_gaia2016")
    results = record.catalog_results
    assert results["gaia_dr3"]["sources"][0].source_id == BARNARD_GAIA_ID
    assert results["simbad"]["sources"][0].source_id == "NAME Barnard's star"  # SIMBAD J2000, 166" away
    assert results["twomass_psc"]["sources"][0].source_id == "17574849+0441405"  # epoch 2000.4
    assert results["allwise"]["sources"][0].source_id == "J175747.94+044323.8"
    # 2MASS saw Barnard's star at jdate 2000.405: its apparent position is displaced by the
    # 547 mas annual parallax. Removing it (parallax adopted with Gaia's motion) brings the
    # 2MASS row from 0.315" to within its 0.06" error of the Gaia track.
    twomass = results["twomass_psc"]["sources"][0]
    assert twomass.metadata["epoch_propagation"] == "target_pm_parallax"
    assert twomass.metadata["epoch_separation_arcsec"] < 0.05
    assert record.provenance["target_proper_motion"]["catalog"] == "gaia_dr3"
    assert record.provenance["target_parallax"]["parallax_mas"] == pytest.approx(546.976, abs=0.01)
    assert record.provenance["target_parallax"]["source"] == "adopted"
    assert record.target["epoch"] == 2016.0 and record.target["parallax_mas"] == pytest.approx(546.976, abs=0.01)
    # AllWISE positions average epochs half a year apart: no parallax removal.
    assert results["allwise"]["sources"][0].metadata["epoch_propagation"] == "target_pm"


async def test_barnard_explicit_proper_motion_is_used_and_reported() -> None:
    # With the proper motion given, the same recorded rows match (the cones differ,
    # so this checks the matching layer via the providers' pad rows).
    record = await crossmatch_fixture("barnard_gaia2016")
    base = record.catalog_results["twomass_psc"]["sources"][0]
    target = validate_target(269.44850252543836, 4.739420051112412, epoch=2016.0,
                             pm_ra_masyr=BARNARD_PM[0], pm_dec_masyr=BARNARD_PM[1])
    matches = match_target(target, [base], 1.0)
    assert matches and matches[0].separation_arcsec < 0.5


async def test_per_catalog_pad_rows_without_service() -> None:
    # Provider level: AllWISE has no proper motions and the target's is unknown, so
    # Barnard's AllWISE row (109" away at J2010.4) is a pad row, never a result.
    sources, failure = await run_catalog("allwise", "barnard_j2000")
    assert failure is None
    assert sources.meta["status"] == "empty" and sources.meta["row_count"] == 0
    assert sources.meta["pad_row_count"] > 0
    assert any(s.source_id == "J175747.94+044323.8" for s in sources.meta["pad_sources"])
    assert sources.meta["cone"]["mode"] == "unknown_pm_pad"


def test_match_target_uses_source_proper_motion() -> None:
    src = CatalogSource("gaia_dr3", "g", 269.44850252543836, 4.739420051112412, 1e-5, {}, {"wavelength": "optical"}, {},
                        epoch=2016.0, proper_motion_ra_masyr=BARNARD_PM[0], proper_motion_dec_masyr=BARNARD_PM[1])
    at_j2000 = validate_target(269.45207696, 4.69336497, epoch=2000.0)
    assert match_target(at_j2000, [src], 1.0)[0].separation_arcsec < 0.01
    assert match_target(validate_target(269.45207696, 4.69336497), [src], 10.0) == []  # no epoch: compared as given


# ---------------------------------------------------------------------------
# Sesame (recorded real -oxp responses)
# ---------------------------------------------------------------------------


def _sesame(name: str) -> str:
    return (FIXTURES / "sesame" / f"{name}.xml").read_text(encoding="utf-8")


def test_sesame_real_response_barnard_has_pm_dec_epoch_and_parallax() -> None:
    res = SesameResolver.parse_response("Barnard's star", _sesame("barnards_star"))
    assert res.canonical_name == "NAME Barnard's star"
    assert res.pm_ra_masyr == -801.551 and res.pm_dec_masyr == 10362.394
    assert res.epoch == 2000.0  # SIMBAD positions are epoch J2000
    assert res.redshift is None and res.object_type == "BY*"
    assert res.resolver_metadata["parallax_mas"] == 546.9759
    assert res.resolver_metadata["radial_velocity_kms"] == -110.11
    target = resolved_target(res)
    assert target.epoch == 2000.0 and target.proper_motion == (-801.551, 10362.394)


def test_sesame_real_response_3c273_is_extragalactic_despite_a_noise_parallax() -> None:
    # SIMBAD lists Gaia's parallax (0.0108 mas) and pm (-0.023, 0.098) for the quasar.
    res = SesameResolver.parse_response("3C 273", _sesame("3c273"))
    assert res.object_type == "BLL" and res.redshift == pytest.approx(0.1576, abs=1e-3)
    assert res.resolver_metadata["parallax_mas"] == pytest.approx(0.0108, abs=1e-3)
    assert res.resolver_metadata["extragalactic"] is True
    assert res.resolver_metadata["proper_motion_applicable"] is False
    assert resolved_target(res).proper_motion == (0.0, 0.0)


def test_sesame_real_response_m87_redshift_and_stationary() -> None:
    res = SesameResolver.parse_response("M87", _sesame("m87"))
    assert res.canonical_name == "M 87"
    assert res.redshift == pytest.approx(0.0042)
    assert res.object_type == "AGN"
    assert res.resolver_metadata["extragalactic"] is True
    assert res.resolver_metadata["proper_motion_applicable"] is False  # Gaia pm of a nucleus is noise
    assert resolved_target(res).proper_motion == (0.0, 0.0)


@respx.mock
async def test_sesame_http_5xx_raises_resolution_error() -> None:
    from models import ObjectResolutionError

    respx.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(503, text="down")
    async with offline_client() as client:
        with pytest.raises(ObjectResolutionError, match="HTTP 503"):
            await SesameResolver(client).resolve("M87")


def test_api_name_search_passes_resolver_epoch_and_proper_motion(monkeypatch) -> None:
    import api

    captured: dict = {}

    class FakeService:
        registry = CatalogRegistry()

        async def crossmatch(self, ra, dec, *, query=None, **kwargs):
            captured["target"] = query.target
            captured["query"] = query
            from models import UnifiedRecord
            return UnifiedRecord({"ra": ra, "dec": dec}, 0, {}, {}, [], {})

    monkeypatch.setattr(api, "get_service", lambda: FakeService())
    monkeypatch.setattr(api, "cache", CacheManager(None))
    with respx.mock(assert_all_called=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
            200, text=_sesame("barnards_star"), headers={"content-type": "text/xml"})
        import asyncio
        asyncio.run(api._search(api.SearchRequest(name="Barnard's star", radius_arcsec=5.0)))
    target = captured["target"]
    assert target.epoch == 2000.0 and target.pm_dec_masyr == 10362.394
    assert captured["query"].metadata["pm_source"] == "resolver"


async def test_resolver_motion_is_reported_as_resolver_not_input() -> None:
    record = await crossmatch_fixture("proxima_j2000_pm", pm_source="resolver")
    assert record.provenance["target_proper_motion"]["source"] == "resolver"
    record = await crossmatch_fixture("proxima_j2000_pm")
    assert record.provenance["target_proper_motion"]["source"] == "input"


def test_recorded_epoch_fixtures_document_their_epoch() -> None:
    meta = json.loads((FIXTURES / "barnard_j2000" / "gaia_dr3.json").read_text(encoding="utf-8"))
    assert meta["epoch"] == 2000.0
    assert "TOP 2001" in meta["exchanges"][0]["request_body"].replace("+", " ")
