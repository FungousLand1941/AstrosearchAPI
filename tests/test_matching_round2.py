"""Regression tests for the adversarial review of the Bayesian matching engine (round 2).

Whole searches recorded end to end (``tests/test_matching_fixtures.py record-search``: every
request, density probes and the Sesame answer included) are replayed strictly; synthetic
rows check the arithmetic.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import fixture_io
from test_matching_fixtures import (
    RECORDED_SEARCHES,
    recorded_resolver,
    replay_crossmatch,
    replay_search,
)

import astrometry
from astrometry import (
    AssociationConfig,
    Detection,
    associate,
    completeness_prior,
    detection_covariance,
    source_detection,
)
from crossmatch import (
    AdvancedQuery,
    CrossmatchService,
    Match,
    _class_from_association,
    coordinate_sigma_arcsec,
    identity_kind,
    is_binary_component,
    parse_target_coordinates,
)
from models import CatalogRegistry, CatalogSource, offset_radec, resolved_target, validate_target
from providers import QueryResult


def members(record: dict) -> dict[str, dict]:
    """Rows by source id (NED lists other catalogues' designations: the survey's own row wins)."""
    out: dict[str, dict] = {}
    for m in (m for g in record["crossmatch_groups"] for m in g["members"]):
        if m["source_id"] not in out or m["catalog"] != "ned":
            out[m["source_id"]] = m
    return out


def target_group(record: dict) -> dict | None:
    return next((g for g in record["crossmatch_groups"] if g["contains_target"]), None)


def row(catalog: str, ra: float, dec: float, err: float | None, **kw) -> CatalogSource:
    metadata = kw.pop("metadata", {})
    data = kw.pop("data", {})
    return CatalogSource(catalog, kw.pop("source_id", "x"), ra, dec, err, data, metadata, {}, **kw)


class StaticProvider:
    """Serves fixed rows to every catalogue (by the row's catalogue name)."""

    def __init__(self, rows: list[CatalogSource]) -> None:
        self.rows = rows

    async def query(self, catalog, target, radius_arcsec):
        return QueryResult([r for r in self.rows if r.catalog == catalog.name], {"max_rows": 400, "row_limit": 400})


def static_service(rows: list[CatalogSource], names: list[str]) -> CrossmatchService:
    registry = CatalogRegistry()
    registry._catalogs = {n: registry.get(n) for n in names}
    provider = StaticProvider(rows)
    return CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})


async def search_endpoint_emulation(key: str, *, min_confidence: float = 0.5):
    """What POST /api/v1/search does for a name (api._search): the resolver's target model,
    an AdvancedQuery with the default min_confidence, the search radius and catalogues of the
    recorded search -- and no resolver identity passed to the engine."""
    spec = RECORDED_SEARCHES[key]
    obj = recorded_resolver(key).obj
    t = resolved_target(obj)
    query = AdvancedQuery.from_dict({
        "ra": t.ra, "dec": t.dec, "epoch": t.epoch, "pm_ra_masyr": t.pm_ra_masyr, "pm_dec_masyr": t.pm_dec_masyr,
        "parallax_mas": t.parallax_mas, "radius_arcsec": spec["radius_arcsec"], "min_confidence": min_confidence,
        "catalogs": spec["catalogs"], "metadata": {"pm_source": "resolver" if t.proper_motion is not None else None}})
    return (await replay_search(key, query=query, catalogs=None, radius_arcsec=None, ra=t.ra, dec=t.dec))["unified"]



# ---------------------------------------------------------------------------
# Issue: named extragalactic targets became 'star' through a noise parallax
# ---------------------------------------------------------------------------


async def test_named_3c273_is_extragalactic_and_keeps_its_radio_and_xray_counterparts() -> None:
    result = await replay_search("named_3c273")
    record = result["record"]
    assoc = record["provenance"]["association"]
    # The resolver says BLL: extragalactic before any parallax rule; no SIMBAD noise
    # parallax (0.011 mas) is adopted (it made the class 'star' and the priors 1e-7).
    assert assoc["target_class"]["class"] == "extragalactic"
    assert record["provenance"]["target_parallax"] is None
    assert set(assoc["completeness"].values()) == {0.5}
    rows = members(record)
    for sid in ("2CXO J122906.6+020308", "FIRST J122906.7+020308", "2RXS J122906.6+020309", "5XMM J122906.6+020308",
                "3C 273", "3700386905605055360"):
        assert rows[sid]["target_probability"] > 0.99, sid  # were 0.0079, 0.0014, 0.0039, 0.0093
    assert rows["3C 273"]["target_identity"]  # the resolved name's SIMBAD row
    group = target_group(record)
    assert {"chandra", "first", "rosat", "xmm", "simbad", "gaia_dr3"} <= set(group["catalogs"])


async def test_search_endpoint_keeps_3c273s_counterparts_without_the_resolver_identity() -> None:
    # POST /search passes the resolver's pm (0, 0) and no identity: the SIMBAD row's type
    # (BLL) decides, and its noise parallax is refused.
    record = await search_endpoint_emulation("named_3c273")
    assert record.provenance["association"]["target_class"]["class"] == "extragalactic"
    assert record.provenance["target_parallax"] is None
    returned = {n: {s.source_id for s in r["sources"]} for n, r in record.catalog_results.items()}
    assert returned["first"] == {"FIRST J122906.7+020308"}
    assert returned["rosat"] == {"2RXS J122906.6+020309"}
    assert returned["chandra"] == {"2CXO J122906.6+020308"}
    assert returned["xmm"] == {"5XMM J122906.6+020308"}


def test_noise_parallaxes_of_extragalactic_rows_are_never_adopted() -> None:
    from crossmatch import _adopt_parallax

    target = validate_target(187.2779154, 2.0523883, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
    quasar = row("simbad", 187.2779154, 2.0523883, 0.01, source_id="3C 273", epoch=2000.0,
                 proper_motion_ra_masyr=-0.1, proper_motion_dec_masyr=0.0,
                 metadata={"physical": {"object_type": "BLL", "parallax": 0.0108, "redshift": 0.158}})
    assert _adopt_parallax(target, [("simbad", QueryResult([quasar], {}))], 5.0) is None
    # Without an error, a parallax counts only from 10 mas (NGC 4151's 0.216 mas is noise) ...
    galaxy_like = dataclasses.replace(quasar, metadata={"physical": {"object_type": "*", "parallax": 0.216}})
    assert _adopt_parallax(target, [("simbad", QueryResult([galaxy_like], {}))], 5.0) is None
    # ... and with one, from 5 sigma.
    gaia = row("gaia_dr3", 187.2779154, 2.0523883, 0.0003, source_id="g", epoch=2016.0, proper_motion_ra_masyr=0.1,
               proper_motion_dec_masyr=0.0, data={"parallax": 0.0108, "parallax_error": 0.03})
    assert _adopt_parallax(target, [("gaia_dr3", QueryResult([gaia], {}))], 5.0) is None
    star = dataclasses.replace(gaia, data={"parallax": 12.0, "parallax_error": 0.03})
    assert _adopt_parallax(target, [("gaia_dr3", QueryResult([star], {}))], 5.0)[0].parallax_mas == 12.0


# ---------------------------------------------------------------------------
# Issue: a parallax forced class 'star' before the identity type was looked at
# ---------------------------------------------------------------------------


async def test_named_pulsar_keeps_its_radio_counterparts() -> None:
    record = (await replay_search("named_psr_b0950"))["record"]
    assoc = record["provenance"]["association"]
    assert assoc["target_class"]["class"] == "unknown" and assoc["target_class"]["object_type"] == "Psr"
    assert record["provenance"]["target_parallax"]["parallax_mas"] == pytest.approx(7.9)  # it still has one
    rows = members(record)
    for sid in ("NVSS J095309+075536", "FIRST J095309.2+075535", "J095309.29+075536.9"):
        assert rows[sid]["target_probability"] > 0.99, sid  # were 0.024, 0.00013, 0.009


async def test_named_x_ray_binary_keeps_its_radio_and_x_ray_counterparts() -> None:
    record = (await replay_search("named_ss433"))["record"]
    assert record["provenance"]["association"]["target_class"]["class"] == "unknown"
    rows = members(record)
    for sid in ("J191149.57+045858.0", "NVSS J191149+045858", "2RXS J191149.7+045858", "5XMM J191149.6+045857"):
        assert rows[sid]["target_probability"] > 0.99, sid  # were 0.006, 0.008, 0.032, 0.49
    # The same rows through POST /search (the resolver's parallax passed as input).
    unified = await search_endpoint_emulation("named_ss433")
    returned = {n: [s.source_id for s in r["sources"]] for n, r in unified.catalog_results.items()}
    assert returned["vlass"] == ["J191149.57+045858.0"] and returned["nvss"] == ["NVSS J191149+045858"]
    assert returned["rosat"] == ["2RXS J191149.7+045858"] and returned["xmm"] == ["5XMM J191149.6+045857"]


async def test_wolf_rayet_star_gets_the_default_priors_not_the_quiet_star_fraction() -> None:
    record = (await replay_search("named_wr147"))["record"]
    cls = record["provenance"]["association"]["target_class"]
    assert cls["class"] == "unknown" and cls["object_type"] == "WR*"
    rows = members(record)
    for sid in ("J203643.64+402107.6", "2CXO J203643.6+402107", "5XMM J203643.6+402107"):
        assert rows[sid]["target_probability"] > 0.99, sid  # were 0.0025, 0.085, 0.17


def test_identity_kinds() -> None:
    assert identity_kind("QSO") == identity_kind("BLL") == identity_kind("Sy1") == "extragalactic"
    assert identity_kind("GlC") == identity_kind("SNR") == identity_kind("!*Cl") == identity_kind("ClG") == "extended"
    for otype in ("Psr", "LXB", "HXB", "XB*", "WR*", "s*b", "Sy*", "PN", "Be*", "TT*", "RS*", "BY*", "Sy?", "XrayS"):
        assert identity_kind(otype) == "emitter", otype
    for otype in ("*", "PM*", "**", "V*", "LP?", "WD*", "RG*", "Pl"):
        assert identity_kind(otype) == "star", otype
    # An O star or a Wolf-Rayet spectral type makes an ordinary stellar type an emitter.
    assert identity_kind("*", "O9.5Iab") == identity_kind("**", "WC7+O5") == "emitter"
    assert identity_kind("*", "G2V") == "star"
    assert identity_kind(None) is None and identity_kind("Opt") is None


def test_completeness_prior_of_emitters_and_extended_targets() -> None:
    assert completeness_prior("nvss", "extended") == 0.5  # radio: the extended emission itself
    assert completeness_prior("gaia_dr3", "extended") == astrometry.EXTENDED_POINT_COMPLETENESS
    assert completeness_prior("simbad", "extended") == 0.5


# ---------------------------------------------------------------------------
# Issue: a NED type pre-empted SIMBAD's (Sco X-1: NED 'V*', SIMBAD 'LXB')
# ---------------------------------------------------------------------------


def test_simbad_high_energy_type_decides_over_ned_star_type_in_either_order() -> None:
    target = (244.97945528, -15.64028269)
    simbad = row("simbad", *target, 0.01, source_id="V* V818 Sco", metadata={"physical": {"object_type": "LXB"}})
    ned = row("ned", *target, 0.05, source_id="V0818 Sco", metadata={"physical": {"object_type": "V*"}})
    for rows in ([simbad, ned], [ned, simbad]):
        dets = [Detection(r.catalog, r.ra, r.dec, (1e-4, 0.0, 1e-4)) for r in rows]
        result = associate(dets, {r.catalog: 1e3 for r in rows}, target=target)
        info = _class_from_association([Match(r.catalog, r, 0.0, 0.0) for r in rows], result)
        assert info["class"] == "unknown" and info["catalog"] == "simbad", info
    # Any identity row of a decisive type decides, even when SIMBAD's is an ordinary star.
    star = dataclasses.replace(simbad, metadata={"physical": {"object_type": "*"}})
    xray = dataclasses.replace(ned, metadata={"physical": {"object_type": "XrayS"}})
    dets = [Detection(r.catalog, r.ra, r.dec, (1e-4, 0.0, 1e-4)) for r in (star, xray)]
    result = associate(dets, {"simbad": 1e3, "ned": 1e3}, target=target)
    assert _class_from_association([Match(r.catalog, r, 0.0, 0.0) for r in (star, xray)], result)["class"] == "unknown"


async def test_sco_x1_by_coordinates_keeps_its_radio_counterparts() -> None:
    record = (await replay_search("scox1_coordinates"))["record"]
    cls = record["provenance"]["association"]["target_class"]
    assert cls["class"] == "unknown" and cls["catalog"] == "simbad"  # SIMBAD's LXB, not NED's V*
    rows = members(record)
    assert rows["J161955.04-153824.7"]["target_probability"] > 0.99  # was 0.055
    assert rows["NVSS J161955-153824"]["target_probability"] > 0.99  # was 0.00077


# ---------------------------------------------------------------------------
# Issue: named extended objects lost their identity row to a neighbouring star
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "identity"), [("named_crab", "M 1"), ("named_m42", "M 42"), ("named_m13", "M 13")])
async def test_named_extended_object_keeps_its_identity_row(key: str, identity: str) -> None:
    record = (await replay_search(key))["record"]
    assoc = record["provenance"]["association"]
    assert assoc["target_class"]["class"] == "extended"
    rows = members(record)
    assert rows[identity]["target_probability"] > 0.9, rows[identity]["target_probability"]  # were 0.0, 0.0, 1e-6
    assert rows[identity]["target_identity"]
    group = target_group(record)
    assert identity in {m["source_id"] for m in group["members"]}
    # No star, field source or cluster member is the counterpart of the object.
    assert {m["catalog"] for m in group["members"]} <= {"simbad", "ned"}
    point = [m for m in rows.values() if m["catalog"] in ("gaia_dr3", "twomass_psc")]
    assert point and all(m["target_probability"] < 0.1 for m in point)


async def test_crab_star_near_the_cone_edge_is_one_field_object_not_four_coincidences() -> None:
    # The 2MASS J05343217+2200560 star, 8.7" from the Crab in a 10" cone, is listed by Gaia,
    # 2MASS, SIMBAD and NED: near the cone edge its rows were independent coincidences
    # (a 4-way boost of ~(B/N)^3) and took the target group with P = 0.83.
    record = (await replay_search("named_crab"))["record"]
    rows = members(record)
    for sid in ("3403818176867563264", "05343217+2200560", "2MASS J05343217+2200560"):
        assert rows[sid]["target_probability"] < 0.01, sid


async def test_m13_by_coordinates_through_the_search_endpoint_returns_m13() -> None:
    record = await search_endpoint_emulation("named_m13")
    assert record.provenance["association"]["target_class"]["class"] == "extended"
    returned = {n: [s.source_id for s in r["sources"]] for n, r in record.catalog_results.items()}
    assert "M 13" in returned["simbad"] and "NGC 6205 576" not in returned["simbad"]  # was the member star
    assert returned["gaia_dr3"] == []


async def test_star_near_an_extended_objects_centre_is_still_the_star() -> None:
    # The HVC row 5.5" from a Gaia star (E quality) is not the star's identity and does not
    # make the target 'extended' (the star's own row coincides better).
    record = await replay_crossmatch("hvc_star")
    assoc = record.provenance["association"]
    assert assoc["target_class"]["class"] != "extended"
    assert assoc["identity_rows"] == []


# ---------------------------------------------------------------------------
# Issue: the 1.8-deg density map cannot see globular-cluster cores
# ---------------------------------------------------------------------------


async def test_m13_core_posterior_does_not_depend_on_the_search_radius() -> None:
    small = (await replay_search("m13_core_3arcsec", target_uncertainty_arcsec=0.5))["record"]
    large = (await replay_search("m13_core_30arcsec", target_uncertainty_arcsec=0.5))["record"]
    d_small = small["provenance"]["association"]["densities"]["gaia_dr3"]
    d_large = large["provenance"]["association"]["densities"]["gaia_dr3"]
    # The 3" cone held 7 Gaia rows where the order-5 map expects 0.15: probed over 30".
    assert d_small["probe"]["status"] == "success" and d_small["probe"]["radius_arcsec"] == 30.0
    assert d_small["density_per_deg2"] > 1.0e6  # Gaia 30" COUNT: 2.86e6; the 3" estimate was 1.2e5
    assert d_small["density_per_deg2"] == pytest.approx(d_large["density_per_deg2"], rel=0.05)
    p_small = members(small)["1328057905737084928"]["target_probability"]
    p_large = members(large)["1328057905737084928"]["target_probability"]
    assert abs(p_small - p_large) < 0.02, (p_small, p_large)  # were 0.67 and 0.22


def test_density_probe_trigger_is_a_poisson_excess_over_the_map() -> None:
    from crossmatch import SearchContext, _needs_density_probe

    target = validate_target(150.0, 20.0)
    ctx = SearchContext(target, [], 3.0, None, None, None, 0.1, None)
    sparse = QueryResult([row("gaia_dr3", *offset_radec(150.0, 20.0, 1.0 + i * 0.3, 0.0), 0.001, source_id=str(i))
                          for i in range(2)], {"query_radius_arcsec": 3.0})
    crowded = QueryResult([row("gaia_dr3", *offset_radec(150.0, 20.0, 0.5 + i * 0.3, 0.2), 0.001, source_id=str(i))
                           for i in range(7)], {"query_radius_arcsec": 3.0})
    assert not _needs_density_probe("gaia_dr3", sparse, ctx, CatalogRegistry())
    assert _needs_density_probe("gaia_dr3", crowded, ctx, CatalogRegistry())
    assert not _needs_density_probe("simbad", crowded, ctx, CatalogRegistry())  # literature entries of the target
    truncated = QueryResult(list(crowded), {"query_radius_arcsec": 3.0, "archive_truncated": True})
    assert not _needs_density_probe("gaia_dr3", truncated, ctx, CatalogRegistry())


async def test_a_failed_density_probe_falls_back_to_the_searched_cone() -> None:
    class Provider:
        def __init__(self) -> None:
            self.radii: list[float] = []

        async def query(self, catalog, target, radius_arcsec):
            self.radii.append(radius_arcsec)
            if radius_arcsec > 10:
                raise RuntimeError("probe refused")
            rows = [row("gaia_dr3", *offset_radec(target.ra, target.dec, 0.5 + i * 0.3, 0.2), 0.001, source_id=str(i),
                        metadata={"wavelength": "optical"}) for i in range(7)]
            return QueryResult(rows, {"max_rows": 200, "row_limit": 200, "query_radius_arcsec": radius_arcsec})

    registry = CatalogRegistry()
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3")}
    provider = Provider()
    record = await CrossmatchService(registry, {"tap": provider}).crossmatch(150.0, 20.0, radius_arcsec=3.0)
    assert provider.radii == [3.0, 30.0]
    info = record.provenance["association"]["densities"]["gaia_dr3"]
    assert info["probe"]["status"] == "failed" and "probe refused" in info["probe"]["error"]
    assert info["radius_arcsec"] == 3.0 and record.failures == []


# ---------------------------------------------------------------------------
# Issue: undated fast movers (cones not widened, pm-less rows not moved)
# ---------------------------------------------------------------------------


async def test_undated_fast_mover_warns_that_its_rows_at_other_epochs_may_be_outside_the_cone() -> None:
    record = (await replay_search("epseri_undated"))["record"]
    assert record["target"]["epoch"] is None
    warnings = [w for w in record["provenance"]["warnings"] if w.startswith("Undated target")]
    assert len(warnings) == 1 and "15.6 arcsec" in warnings[0] and "epoch=2000" in warnings[0]
    motion = record["provenance"]["target_proper_motion"]
    assert motion["source"] == "adopted" and motion["catalog"] == "simbad"
    # The 2MASS row (1998) moves with eps Eri's motion: the target at 0.99".
    twomass = members(record)["03325591-0927298"]
    assert twomass["position_at_epoch"]["propagation"] == "target_pm_undated"
    assert twomass["target_probability"] > 0.99


async def test_undated_identity_motion_places_the_pm_less_2mass_row() -> None:
    record = (await replay_search("gj15a_undated"))["record"]
    twomass = members(record)["00182256+4401222"]
    assert twomass["separation_arcsec"] > 3.0
    assert twomass["target_probability"] > 0.99  # was 0.045


def test_undated_target_motion_gives_pm_less_rows_one_placement_per_epoch() -> None:
    target = validate_target(10.0, 20.0, pm_ra_masyr=0.0, pm_dec_masyr=1000.0)
    tm = row("twomass_psc", 10.0, 20.0 - 1.0 / 3600.0, 0.07, epoch=1999.0)
    det, info = source_detection(tm, target)
    assert info["propagation"] == "target_pm_undated" and info["epoch_hypotheses"] == [2000.0, 2016.0]
    north = [(p[1] - 20.0) * 3600.0 for p in det.epoch_placements]
    assert north == pytest.approx([0.0, 16.0], abs=1e-6)  # moved from 1999 to J2000 and J2016


# ---------------------------------------------------------------------------
# Issue: AllWISE rows of bright stars (no w1mjdmean; saturated positions)
# ---------------------------------------------------------------------------


async def test_aldebaran_allwise_row_without_an_epoch_is_identified() -> None:
    record = (await replay_search("named_aldebaran"))["record"]
    allwise = members(record)["J043555.27+163031.2"]
    assert allwise["epoch"] is None and allwise["epoch_range"] == [2010.0, 2011.2]  # the survey span
    assert allwise["position_at_epoch"]["propagation"] == "target_pm_span"
    assert "+saturation" in allwise["position_at_epoch"]["covariance_shape"]
    assert allwise["target_probability"] > 0.99  # was 0.0 at 2.26" with sigma 0.02"


def test_missing_row_epoch_falls_back_to_the_catalogue_span() -> None:
    from models import resolve_epoch_range

    allwise = CatalogRegistry().get("allwise")
    assert resolve_epoch_range({"w1mjdmean": None}, allwise) == (2010.0, 2011.2)
    unknown = row("custom", 1.0, 1.0, 0.1)
    assert astrometry._epoch_gap(unknown, 2016.0) == pytest.approx(26.0)  # UNKNOWN_EPOCH_RANGE, never 0


@pytest.mark.parametrize(("w1", "offset", "sigra", "minimum"), [(-1.23, 0.82, 0.0019, 0.8), (-2.7, 1.44, 0.0052, 0.5)])
def test_saturated_allwise_star_offset_by_an_arcsecond_is_still_identified(w1, offset, sigra, minimum) -> None:
    # Antares (W1 = -1.23, 0.82" from its Hipparcos position, sigra 1.9 mas) and a
    # Betelgeuse-like case (1.44", about the 98th percentile of the offsets measured at
    # W1 < 0): formal errors of a few mas, positions off by arcseconds.
    target = validate_target(247.35191542, -26.43200261, epoch=2010.5, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
    ra, dec = offset_radec(target.ra, target.dec, offset, 0.0)

    def p_of(data):
        wise = row("allwise", ra, dec, sigra, source_id="w", epoch=2010.5, data=data,
                   metadata={"positional_error": {"sigma_ra_arcsec": sigra, "sigma_dec_arcsec": sigra}})
        det, _ = source_detection(wise, target)
        result = associate([det], {"allwise": 3e4}, target=(target.ra, target.dec))
        return float(result.target_probability[0])

    assert p_of({"w1mpro": w1}) > minimum  # was 0.0
    assert p_of({"w1mpro": 12.0}) < 1e-3  # an unsaturated source that far off is not the star
    cov, shape = detection_covariance(row("allwise", 0.0, 0.0, sigra, data={"w1mpro": w1}))
    assert shape.endswith("+saturation") and math.sqrt(cov[0]) > 0.2


def test_brightest_2mass_stars_get_their_measured_scatter() -> None:
    # Arcturus (K = -2.9): its 2MASS row lies 0.78" from SIMBAD's J2000 position moved to the
    # 2MASS epoch; stars brighter than K = -1 scatter to 1.0" (90th percentile).
    bright, shape = detection_covariance(row("twomass_psc", 0.0, 0.0, 0.29, data={"k_m": -2.9}))
    faint, faint_shape = detection_covariance(row("twomass_psc", 0.0, 0.0, 0.29, data={"k_m": 0.5}))
    assert shape.endswith("+saturation") and not faint_shape.endswith("+saturation")
    assert math.sqrt(bright[0]) > 0.45 > math.sqrt(faint[0])


# ---------------------------------------------------------------------------
# Issue: stellar priors for registered radio catalogues and partial overrides
# ---------------------------------------------------------------------------


async def test_partial_completeness_override_keeps_the_class_priors_of_the_other_catalogues() -> None:
    key = "nvss_star_629559574718166784"
    for extra in ({}, {"target_class": "star"}):
        record = await replay_crossmatch(key, completeness={"gaia_dr3": 0.6}, **extra)
        assoc = record.provenance["association"]
        assert assoc["target_class"]["class"] == "star"
        assert assoc["completeness"]["gaia_dr3"] == 0.6 and assoc["completeness"]["nvss"] < 0.01
        assert assoc["completeness_override"] == {"gaia_dr3": 0.6}
        nvss = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}["NVSS J100535+215127"]
        assert nvss < 0.01, nvss  # was 0.9937
    # A scalar is a full override.
    record = await replay_crossmatch(key, completeness=0.5)
    assert record.provenance["association"]["completeness"] == 0.5


async def test_registered_radio_catalogue_gets_the_radio_stellar_fraction() -> None:
    registry = CatalogRegistry()
    nvss = registry.get("nvss")
    registry._catalogs = {"gaia_dr3": registry.get("gaia_dr3"), "tgss_like": dataclasses.replace(nvss, name="tgss_like")}
    record = await replay_crossmatch("nvss_star_629559574718166784", registry=registry,
                                     catalogs=["gaia_dr3", "tgss_like"])
    assoc = record.provenance["association"]
    assert assoc["target_class"]["class"] == "star" and assoc["completeness"]["tgss_like"] < 0.01  # was 0.5
    p = {m["source_id"]: m["confidence"] for m in record.provenance["matches"]}["NVSS J100535+215127"]
    assert p < 0.1  # was 0.9877
    # A definition may declare its own table.
    declared = [[0.63, 0.2], [125.3, 0.4]]
    assert completeness_prior("tgss_like", "star", 1.0, wavelength="radio", fractions=declared) > 0.2
    assert completeness_prior("erass_like", "star", 1.0, wavelength="xray") == completeness_prior("rosat", "star", 1.0)


# ---------------------------------------------------------------------------
# Issue: coordinate precision from float repr; sexagesimal input
# ---------------------------------------------------------------------------


def _star_at(ra: float, dec: float) -> CatalogSource:
    return row("gaia_dr3", *offset_radec(ra, dec, 0.02, 0.0), 0.0003, source_id="star", epoch=2016.0,
               proper_motion_ra_masyr=2.0, proper_motion_dec_masyr=-3.0, data={"parallax": 5.0, "parallax_error": 0.05},
               metadata={"wavelength": "optical", "catalog_has_proper_motions": True})


async def test_round_float_coordinates_are_exact() -> None:
    service = static_service([_star_at(150.5, 2.2)], ["gaia_dr3"])
    for ra, dec in ((150.5, 2.2), ("150.500000", "2.200000"), (150.500001, 2.200001)):
        record = await service.crossmatch(ra, dec, radius_arcsec=3.0)
        assoc = record.provenance["association"]
        assert assoc["target_sigma_arcsec"] == pytest.approx(0.1), (ra, dec)  # was 103.9"
        assert record.provenance["matches"][0]["confidence"] > 0.99, (ra, dec)  # was 0.047
    # The same through the POST /search query path (JSON floats, min_confidence 0.5).
    query = AdvancedQuery.from_dict({"ra": 150.5, "dec": 2.2, "radius_arcsec": 3.0, "min_confidence": 0.5})
    record = await service.crossmatch(150.5, 2.2, query=query)
    assert record.catalog_results["gaia_dr3"]["returned_count"] == 1


def test_round_floats_and_zero_padded_text_keep_the_star_through_the_sse_route() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import streaming

    app = FastAPI()
    app.include_router(streaming.router)
    app.state.service = static_service([_star_at(150.5, 2.2)], ["gaia_dr3"])
    with TestClient(app) as client:
        response = client.get("/api/v1/search/stream", params={"ra": "150.500000", "dec": "2.200000",
                                                                "radius_arcsec": 3.0})
        assert response.status_code == 200
        done = [line for line in response.text.splitlines() if line.startswith("data:")][-1]
        record = json.loads(done[5:])["record"]
        assert record["provenance"]["association"]["target_sigma_source"] == "default"
        assert record["provenance"]["matches"][0]["confidence"] > 0.99
        assert client.get("/api/v1/search/stream", params={"ra": "400", "dec": "2"}).status_code == 422
        assert client.get("/api/v1/search/stream", params={"ra": "abc", "dec": "2"}).status_code == 422
        assert client.get("/api/v1/search/stream", params={"ra": "0e400", "dec": "2"}).status_code == 422
        response = client.get("/api/v1/search/stream", params={"ra": "150.5", "dec": "2.2",
                                                                "target_uncertainty_arcsec": 1e300})
        assert response.status_code == 422


def test_coordinate_text_rounding_decimal_and_sexagesimal() -> None:
    ra, dec, sigma = parse_target_coordinates("12 29 07", "+02 03 09")
    assert ra == 15 * (12 + 29 / 60 + 7 / 3600) and dec == 2 + 3 / 60 + 9 / 3600
    s_ra = 15.0 * math.cos(math.radians(dec)) / math.sqrt(12.0)
    s_dec = 1.0 / math.sqrt(12.0)
    assert sigma == pytest.approx(math.sqrt((s_ra**2 + s_dec**2) / 2.0))
    assert parse_target_coordinates("12:29:06.70", "+02:03:08.6")[2] < 0.1
    assert parse_target_coordinates("12h29m07s", "-02d03m09s")[1] == pytest.approx(-(2 + 3 / 60 + 9 / 3600))
    assert parse_target_coordinates("187d16m45s", "2 03 09")[0] == pytest.approx(187 + 16 / 60 + 45 / 3600)
    # The web UI's float conversion of '12 29 07 +02 03 09' keeps its rounding signature ...
    assert coordinate_sigma_arcsec(15 * (12 + 29 / 60 + 7 / 3600), 2 + 3 / 60 + 9 / 3600) == pytest.approx(sigma)
    # ... which exact floats never carry.
    assert coordinate_sigma_arcsec(187.2779154, 2.0523883) == 0.0


async def test_ui_parsed_sexagesimal_3c273_keeps_the_quasar_and_not_the_jet_knot() -> None:
    for coords in ({}, {"ra": "12 29 07", "dec": "+02 03 09"}):
        record = (await replay_search("3c273_sexagesimal", **coords))["record"]
        assoc = record["provenance"]["association"]
        assert assoc["target_sigma_source"] == "coordinate_precision"
        east, north = assoc["target_sigma_axes"]
        assert east == pytest.approx(math.hypot(0.1, 15.0 * math.cos(math.radians(2.0525)) / math.sqrt(12.0)), rel=1e-3)
        assert north == pytest.approx(math.hypot(0.1, 1.0 / math.sqrt(12.0)), rel=1e-3)
        rows = members(record)
        for sid in ("3C 273", "3700386905605055360", "110461872779253351"):
            assert rows[sid]["target_probability"] > 0.9, sid  # were 0.0 (sigma 0.1")
        assert rows["[CME2001] 3C 273 1"]["target_probability"] < 0.1  # was 0.9957


# ---------------------------------------------------------------------------
# Issue: malformed rows, unbounded target uncertainty
# ---------------------------------------------------------------------------


RA0, DEC0 = 150.123456, 20.654321


def _fuzz(cat, sid, dx, err, data=None, meta=None, **kw):
    ra, dec = offset_radec(RA0, DEC0, dx, 0.0)
    return CatalogSource(cat, sid, ra, dec, err, data or {}, {"wavelength": "x", **(meta or {})}, {}, **kw)


FUZZ = {
    "err_huge": ([_fuzz("gaia_dr3", "a", 0.1, 1e200)], False),
    "err_inf": ([_fuzz("gaia_dr3", "a", 0.1, float("inf"))], True),
    "ellipse_huge": ([_fuzz("allwise", "a", 0.1, 0.1, meta={"positional_error": {"ellipse_1sigma_arcsec": {
        "major": 1e200, "minor": 1e200, "pa_deg": 30}}})], False),
    "nvss_huge": ([_fuzz("nvss", "a", 1.0, 1.0, data={"major_axis": 1e10, "minor_axis": 1e10, "position_angle": 10})],
                  False),
    "pm_nan": ([_fuzz("gaia_dr3", "a", 0.1, 0.1, epoch=2016.0, proper_motion_ra_masyr=float("nan"),
                      proper_motion_dec_masyr=0.0)], True),
    "pm_huge": ([_fuzz("gaia_dr3", "a", 0.1, 0.1, epoch=2016.0, proper_motion_ra_masyr=1e12,
                       proper_motion_dec_masyr=0.0)], True),
    "epoch_nan": ([_fuzz("gaia_dr3", "a", 0.1, 0.1, epoch=float("nan"), proper_motion_ra_masyr=100.0,
                         proper_motion_dec_masyr=0.0)], True),
}


@pytest.mark.parametrize("case", sorted(FUZZ))
async def test_one_malformed_row_never_aborts_the_association(case: str) -> None:
    rows, dropped = FUZZ[case]
    good = _fuzz("twomass_psc", "good", 0.05, 0.07, epoch=1999.0)
    names = sorted({r.catalog for r in rows} | {"twomass_psc"})
    service = static_service([*rows, good], names)
    for epoch in (None, 2010.0):
        record = await service.crossmatch(RA0, DEC0, radius_arcsec=5.0, epoch=epoch)
        confidences = [m["confidence"] for m in record.provenance["matches"]]
        assert all(isinstance(c, float) and math.isfinite(c) and 0.0 <= c <= 1.0 for c in confidences), confidences
        assert "good" in {m["source_id"] for m in record.provenance["matches"]}
        bad = [w for w in record.provenance["warnings"] if "invalid astrometry" in w]
        assert bool(bad) == dropped, record.provenance["warnings"]
        assert record.catalog_results[rows[0].catalog]["invalid_rows"] == (1 if dropped else 0)


async def test_target_uncertainty_has_an_upper_bound() -> None:
    service = static_service([], ["gaia_dr3"])
    with pytest.raises(ValueError, match="at most 3600"):
        await service.crossmatch(RA0, DEC0, target_uncertainty_arcsec=1e300)
    with pytest.raises(ValueError, match="too coarse"):
        service.prepare("1e5", "10")
    with pytest.raises(ValueError, match="no usable precision"):
        service.prepare("0e400", "10")


def test_binary_component_names_are_not_constellations() -> None:
    from types import SimpleNamespace

    def component(name: str) -> bool:
        return is_binary_component(SimpleNamespace(object_type="*", canonical_name=name, query=name))

    for name in ("* alf PsA", "V* R CrB", "* alf TrA", "V* R CrA", "Sgr A", "NAME Sgr A", "Cas A", "Proxima Cen"):
        assert not component(name), name  # were components (30 mas/yr motion allowance)
    for name in ("* 61 Cyg A", "HD 239960A", "GJ 860 A", "Wolf 1069 B", "alf Cen A", "BD+56 2783A"):
        assert component(name), name


# ---------------------------------------------------------------------------
# Issue: CPU-bound association on the event loop; cost of the field corrections
# ---------------------------------------------------------------------------


class Crowd:
    """A crowded synthetic sky: n objects in the cone, each seen by 70% of the catalogues."""

    ERR: ClassVar[dict[str, float]] = {"gaia_dr3": 0.02, "twomass_psc": 0.08, "allwise": 0.15, "panstarrs_dr2": 0.03,
                                       "simbad": 0.05, "sdss": 0.05}

    def __init__(self, n_obj: int, seed: int = 1) -> None:
        self.n_obj, self.seed = n_obj, seed

    async def query(self, catalog, target, radius_arcsec):
        import random

        rng = random.Random(self.seed)
        objs = [(radius_arcsec * math.sqrt(rng.random()), rng.random() * 2 * math.pi) for _ in range(self.n_obj)]
        rng2 = random.Random(catalog.name)
        err = self.ERR.get(catalog.name, 0.1)
        out = []
        for i, (r, th) in enumerate(objs):
            if rng2.random() > 0.7:
                continue
            x, y = r * math.cos(th) + rng2.gauss(0, err), r * math.sin(th) + rng2.gauss(0, err)
            ra, dec = offset_radec(target.ra, target.dec, x, y)
            out.append((math.hypot(x, y), CatalogSource(catalog.name, f"{catalog.name}-{i}", ra, dec, err, {},
                                                         {"wavelength": catalog.wavelength}, {})))
        out.sort(key=lambda t: t[0])
        rows = [s for _, s in out[:catalog.max_rows]]
        return QueryResult(rows, {"max_rows": catalog.max_rows, "row_limit": catalog.max_rows,
                                  "archive_truncated": len(out) > catalog.max_rows})


def crowd_service(n_obj: int, catalogs: list[str]) -> CrossmatchService:
    registry = CatalogRegistry()
    registry._catalogs = {n: registry.get(n) for n in catalogs}
    provider = Crowd(n_obj)
    return CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})


SIX = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2", "simbad", "sdss"]


async def test_crowded_multi_catalogue_association_keeps_the_event_loop_responsive() -> None:
    service = crowd_service(400, SIX)
    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    started = time.perf_counter()
    record = await service.crossmatch(272.01234567, -27.01234567, radius_arcsec=30.0)
    elapsed = time.perf_counter() - started
    done.set()
    await beat
    assert record.provenance["association"]["states"] >= 1000  # the beam-search regime
    assert max(gaps) < 0.5, max(gaps)  # the association runs in a worker thread (was 11.9 s)
    assert elapsed < 20.0, elapsed


def test_sse_route_answers_other_requests_while_associating() -> None:
    import httpx
    from fastapi import FastAPI

    import streaming

    app = FastAPI()
    app.include_router(streaming.router)

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    app.state.service = crowd_service(400, SIX)

    async def run() -> tuple[float, int]:
        latencies: list[float] = []
        done = asyncio.Event()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=None) as http:
            async def pinger() -> None:
                while not done.is_set():
                    t = time.perf_counter()
                    await http.get("/ping")
                    latencies.append(time.perf_counter() - t)
                    await asyncio.sleep(0.02)

            task = asyncio.create_task(pinger())
            await asyncio.sleep(0.1)
            response = await http.get("/api/v1/search/stream", params={"ra": "272.01234567", "dec": "-27.01234567",
                                                                       "radius_arcsec": 30})
            done.set()
            await task
        return max(latencies), response.text.count("event: done")

    worst, dones = asyncio.run(run())
    assert dones == 1
    assert worst < 0.5, worst


def test_budgeted_field_corrections_match_the_exact_ones() -> None:
    # Crowded 15-catalogue-like case: the beam is full (20,000 states) and greedy objects
    # exist; the most probable associations get the full correction, the rest the
    # block-by-block estimate. The posteriors agree with the all-exact computation.
    cats = [astrometry.SimCatalog(f"c{k}", s, d, 0.5) for k, (s, d) in
            enumerate([(0.1, 40.0), (0.2, 30.0), (0.3, 30.0), (0.5, 20.0), (0.8, 10.0), (1.2, 8.0)])]
    rng = np.random.default_rng(3)
    dets, _, target = astrometry.simulate_field(rng, cats, radius_arcsec=12.0, target_sigma_arcsec=0.5)
    dens = {c.name: c.density_arcmin2 * 3600 for c in cats}
    cfg = AssociationConfig(target_sigma_arcsec=0.5, completeness=0.5)
    budget = associate(dets, dens, target=target, config=cfg, cone=(150.0, 20.0, 12.0))
    original = astrometry._corrected_weights
    try:
        astrometry._corrected_weights = lambda fp, nodes, dc, lw, **_: original(fp, nodes, dc, lw, max_exact=10**9)
        exact = associate(dets, dens, target=target, config=cfg, cone=(150.0, 20.0, 12.0))
    finally:
        astrometry._corrected_weights = original
    assert np.max(np.abs(budget.target_probability - exact.target_probability)) < 1e-3
    assert abs(budget.p_any - exact.p_any) < 1e-3


# ---------------------------------------------------------------------------
# Issue: calibration only on the engine's own generative model
# ---------------------------------------------------------------------------


def _calibration(detection: str, object_density: float | None, fields: int = 1500, seed: int = 11):
    cats = [astrometry.SimCatalog("a", 0.5, 7.5, 0.5), astrometry.SimCatalog("b", 1.0, 5.0, 0.5),
            astrometry.SimCatalog("c", 1.5, 3.75, 0.5), astrometry.SimCatalog("d", 2.0, 2.5, 0.5)]
    dens = {c.name: c.density_arcmin2 * 3600 for c in cats}
    cfg = AssociationConfig(target_sigma_arcsec=1.0, completeness={c.name: 0.5 for c in cats})
    rng = np.random.default_rng(seed)
    probs, truth = [], []
    for _ in range(fields):
        dets, true, target = astrometry.simulate_field(rng, cats, radius_arcsec=20.0, target_sigma_arcsec=1.0,
                                                       detection=detection, object_density_arcmin2=object_density)
        result = associate(dets, dens, target=target, config=cfg, cone=(150.0, 20.0, 20.0))
        probs.extend(result.target_probability.tolist())
        truth.extend(t == 0 for t in true)
    p, t = np.asarray(probs), np.asarray(truth)
    edges = np.quantile(p[p > 0.02], np.linspace(0, 1, 7))
    bins = []
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & (p <= hi)
        bins.append((float(p[m].mean()), float(t[m].mean()), int(m.sum())))
    return bins, p, t


def test_posteriors_on_misspecified_skies() -> None:
    # Nested detection (an object seen by a shallow catalogue is seen by the deeper ones), as
    # in real skies, with the marginal densities unchanged: every sextile within 3 binomial
    # sigma (measured |true - mean| <= 0.01; the review's run without cone-edge objects
    # found the middle sextile overconfident at z = -3.1).
    bins, p, t = _calibration("nested", None)
    for mean, true, count in bins:
        assert abs(true - mean) <= 3.0 * math.sqrt(mean * (1 - mean) / count), (mean, true, count)
    assert p.sum() == pytest.approx(t.sum(), rel=0.03)
    # Nested detection AND twice the object density the model assumes (22.5 vs 11.25 per
    # arcmin^2): within 0.03 of the true fraction in every sextile (measured <= 0.02, the
    # posteriors on the conservative side where they differ significantly).
    bins, _, _ = _calibration("nested", 3.0 * 7.5)
    for mean, true, count in bins:
        assert abs(true - mean) <= 0.03, (mean, true, count)
        if abs(true - mean) > 3.0 * math.sqrt(mean * (1 - mean) / count):
            assert true > mean, (mean, true, count)  # never significantly overconfident


# ---------------------------------------------------------------------------
# Issue: strict fixture replay did not fail the regression tests
# ---------------------------------------------------------------------------


async def test_an_altered_request_fails_the_replay() -> None:
    registry = CatalogRegistry()
    for name in ("gaia_dr3", "twomass_psc"):
        registry._catalogs[name] = dataclasses.replace(registry.get(name), max_rows=199)
    with pytest.raises(fixture_io.FixtureMismatch, match="gaia_dr3"):
        await replay_crossmatch("barnard_undated", registry=registry)
    record = await replay_crossmatch("barnard_undated", registry=registry, allow_mismatch=True)
    assert {(f["catalog"], f["error_type"]) for f in record.failures} == {
        ("gaia_dr3", "FixtureMismatch"), ("twomass_psc", "FixtureMismatch")}
