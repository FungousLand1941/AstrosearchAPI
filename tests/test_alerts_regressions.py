"""Offline regression tests for alerts.py (review round 3).

Recorded fixtures (``tests/fixtures/alerts``, recorded live by ``record_alerts.py``):

* ``xmatch_famous``: Gaia DR3 / SIMBAD / NED / HyperLEDA / Cosmicflows-4 answers for galaxy nuclei
  whose Gaia 5-parameter solutions have spurious, formally significant proper motions (M87,
  NGC 4395, M106, the Seyfert 1 nuclei NGC 3783, NGC 7469, NGC 6814, NGC 3516, NGC 4151,
  ASASSN-14ko's host), SNe on star clusters (SN 2020oi, SN 2004dj), NGC 3115, M83, SN 2009ip,
  hosts outside their D25 ellipse (SN 2023bee, SN 2018aoz), Virgo members without their own CF4
  distance (M100: SN 2006X) and the blazar 3C 273;
* ``alerce_duplicates``: ALeRCE ``lc_classifier`` AGN rows repeated once per classifier version.

The remaining tests use small synthetic inputs and say so.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from astropy import units as u
from astropy.coordinates import SkyCoord
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import make_service, offline_client
from test_alerts import (
    ALERT,
    Replay,
    _d25_galaxy,
    body,
    enrich_recorded,
    enricher_with,
    meta,
    names_of,
    params,
    record_of,
)

import alerts
from alerts import (
    AlerceBroker,
    Alert,
    AlertEnricher,
    AlertEnrichment,
    AlertService,
    AlertStore,
    FinkLSSTBroker,
    FinkZTFBroker,
    fetch_alerts,
    router,
)
from datasets import MetadataStore


@pytest.fixture(autouse=True)
def _fresh_class_lists() -> None:
    alerts.clear_class_list_cache()


@pytest.fixture
def store(tmp_path: Path) -> AlertStore:
    return AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'alerts.sqlite3').as_posix()}"))


@pytest.fixture(scope="module")
def famous() -> dict[str, AlertEnrichment]:
    return asyncio.run(enrich_recorded("xmatch_famous"))


def gaia_rows(res: AlertEnrichment) -> list[dict[str, Any]]:
    return [c for c in res.counterparts if c["catalog"] == "gaia_dr3"]


# ---------------------------------------------------------------------------
# Galaxy nuclei, AGN and clusters are not Galactic stars (recorded)
# ---------------------------------------------------------------------------

# name: (host names, HyperLEDA PGC, why the nucleus' Gaia astrometry is rejected, its pm significance)
NUCLEI: dict[str, tuple[set[str], int, str, float]] = {
    "M87_nucleus": ({"M 87", "Messier 087"}, 41361, "parallax -3.4 sigma", 12.0),
    "NGC4395_nucleus": ({"NGC 4395"}, 40596, "parallax -4.1 sigma", 17.3),
    "NGC4258_nucleus": ({"M 106", "Messier 106"}, 39600, "Gaia DSC P(galaxy) + P(quasar) = 1.000", 7.4),
    "NGC3783_nucleus": ({"NGC 3783"}, 36101, "parallax -4.7 sigma", 14.5),
    "NGC7469_nucleus": ({"NGC 7469"}, 70348, "SIMBAD NGC 7469 (type Sy1) at 0.09\" with RUWE 1.49", 10.0),
    "NGC6814_nucleus": ({"NGC 6814"}, 63545, "with excess-noise significance 6.8", 8.9),
    "NGC3516_nucleus": ({"NGC 3516"}, 33623, "parallax -10.2 sigma", 9.6),
    "NGC4151_nucleus": ({"NGC 4151"}, 38739, "with RUWE 3.02, excess-noise significance 402.7", 4.0),
    "ASASSN-14ko": ({"ESO 253-3", "ESO 253- G 003"}, 17260, "Gaia DSC P(galaxy) + P(quasar) = 1.000", 9.8),
}


@pytest.mark.parametrize("name", sorted(NUCLEI))
def test_galaxy_nuclei_keep_their_host_and_are_not_galactic_stars(famous: dict[str, AlertEnrichment], name: str) -> None:
    """Regression: a >= 5 sigma Gaia proper motion of a galaxy nucleus made it a 'Galactic star' and dropped its host."""
    host_names, pgc, why, pm_sigma = NUCLEI[name]
    res = famous[name]
    assert res.status == "done" and res.known_star is False and res.known_agn is True
    assert res.host is not None and res.host_status == "found" and res.host["method"] == "d25_ellipse"
    assert names_of(res.host) & host_names and res.host["pgc"] == pgc
    assert not any("-> Galactic" in e for e in res.evidence), res.evidence
    nucleus = min(gaia_rows(res), key=lambda c: c["separation_arcsec"])
    # The spurious proper motion is really there, formally significant ...
    assert nucleus["pm_over_error"] == pytest.approx(pm_sigma, abs=0.3)
    # ... and rejected for the stated reason.
    assert any(f"Gaia DR3 {nucleus['source_id']}" in e and "not a star" in e and why in e for e in res.evidence), res.evidence


def test_seyfert1_nuclei_with_stellar_dsc_are_caught_by_coincidence_and_excess_noise(
        famous: dict[str, AlertEnrichment]) -> None:
    """NGC 7469 and NGC 6814: Gaia DSC calls the bright nucleus a star (P ~ 1) and the parallax is not
    negative; only the coincident Seyfert entry plus a poor single-star fit marks it as a nucleus."""
    for name in ("NGC7469_nucleus", "NGC6814_nucleus"):
        nucleus = min(gaia_rows(famous[name]), key=lambda c: c["separation_arcsec"])
        assert nucleus["dsc_p_extragalactic"] < 0.05 and nucleus["parallax_over_error"] > -3
        assert nucleus["astrometric_excess_noise_sig"] > alerts.GAIA_EXCESS_NOISE_SIG_MAX
    assert min(gaia_rows(famous["NGC6814_nucleus"]), key=lambda c: c["separation_arcsec"])["ruwe"] < 1.4


@pytest.mark.parametrize(("name", "host_names", "pgc"), [
    ("SN2020oi", {"Messier 100", "M 100"}, 40153),  # on a compact cluster: Gaia pm 4.1 mas/yr at 5.9 sigma
    ("SN2004dj", {"NGC 2403"}, 21396),  # on the cluster Sandage 96: 6.1 mas/yr at 6.2 sigma, RUWE 3.3
])
def test_supernovae_on_star_clusters_keep_their_host(famous: dict[str, AlertEnrichment], name: str, host_names: set[str],
                                                     pgc: int) -> None:
    res = famous[name]
    assert res.known_star is False and res.host is not None and names_of(res.host) & host_names
    assert res.host["pgc"] == pgc
    cluster = min(gaia_rows(res), key=lambda c: c["separation_arcsec"])
    assert cluster["pm_over_error"] > 5 and cluster["in_galaxy_candidates"] is True
    assert cluster["dsc_p_extragalactic"] > 0.99
    assert any(f"Gaia DR3 {cluster['source_id']}" in e and "not a star" in e for e in res.evidence)


def test_ngc3115_nucleus_is_not_a_foreground_star_through_unrelated_lmxbs(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: extragalactic LMXBs within 2" made the nucleus (G = 15.8, M_G = -14.1) 'a catalogued star'."""
    res = famous["NGC3115_nucleus"]
    assert res.known_star is False and res.stellar_counterpart is True and res.known_variable is True
    assert res.host is not None and "NGC 3115" in names_of(res.host) and res.host["distance_method"] == "cosmicflows4"
    assert any(c["object_type"] == "LXB" for c in res.counterparts)
    assert not any("M_G =" in e and "-> Galactic" in e for e in res.evidence)
    assert any("an extragalactic star, not a Galactic one" in e for e in res.evidence)


def test_foreground_star_luminosity_needs_the_star_itself(famous: dict[str, AlertEnrichment]) -> None:
    """M101's foreground star: the NED stellar entry 0.24" away *is* the Gaia source (point-like, RUWE 1.04)."""
    res = famous["M101_fg_star"]
    gaia = gaia_rows(res)[0]
    assert gaia["ruwe"] < 1.4 and gaia["astrometric_excess_noise_sig"] < 2 and gaia["dsc_p_extragalactic"] < 0.01
    assert any("= NED WISEA J140336.04+542636.4 (type *)" in e and "M_G = -11.5" in e for e in res.evidence)


def test_m83_host_is_named_m83_not_a_fibre_entry(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: 3" from M83's nucleus the host was '6dFGS gJ133700.5-295200', measured to the fibre position."""
    res = famous["M83_near_nucleus"]
    host = res.host
    assert host is not None and host["name"] in {"M 83", "Messier 083"} and host["pgc"] == 48082
    assert not host["name"].startswith("6dFGS")
    assert host["separation_arcsec"] < 3.0  # to the NED/SIMBAD nucleus, not the fibre 6.3" away
    assert host["distance_method"] == "cosmicflows4" and host["distance_mpc"] == pytest.approx(4.77, abs=0.05)


def test_messier_names_match_hyperleda_ngc_names() -> None:
    assert "NGC5236" in alerts._name_keys("Messier 083") and "NGC5236" in alerts._name_keys("M  83")
    assert alerts._name_keys("M 102") == {"M102"}  # disputed identification: no NGC equivalent
    group = {"source_id": "Messier 083", "aliases": ["simbad:M 83"]}
    assert alerts._names_hyperleda_galaxy(group, {"hyperleda_names": ["NGC5236", "ESO444-81"]})


# ---------------------------------------------------------------------------
# Hosts outside the D25 ellipse; stellar types near nearby galaxies (recorded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "host_name", "pgc", "d_dlr", "dm"), [
    ("SN2023bee", "NGC 2708", 25097, 1.48, 32.965),  # 135" from NGC 2708 (a = 95")
    ("SN2018aoz", "NGC 3923", 37061, 1.45, 31.69),  # 224" from NGC 3923 (a = 203"); Ni et al. 2022
])
def test_hosts_outside_the_d25_ellipse_by_dlr(famous: dict[str, AlertEnrichment], name: str, host_name: str, pgc: int,
                                              d_dlr: float, dm: float) -> None:
    """Regression: only containing D25 ellipses were searched, so these hosts gave 'none_within_radius'."""
    res = famous[name]
    host = res.host
    assert res.host_status == "found" and res.host_search_complete is True
    assert host is not None and host_name in names_of(host) and host["pgc"] == pgc
    assert host["method"] == "dlr_outside_d25" and host["d_dlr"] == pytest.approx(d_dlr, abs=0.03)
    assert 1.0 < host["d_dlr"] <= alerts.DLR_HOST_MAX
    assert host["distance_method"] == "cosmicflows4" and host["distance_modulus"] == pytest.approx(dm, abs=0.01)
    assert host["projected_offset_kpc"] == pytest.approx(
        host["distance_mpc"] * 1000 * math.radians(host["separation_arcsec"] / 3600), rel=1e-6)
    assert any("outside every D25 ellipse" in e and "Gupta et al. 2016" in e for e in res.evidence)


def test_sn2009ip_impostor_is_not_declared_a_galactic_star(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: SN 2009ip (SIMBAD s*b, its LBV progenitor) just outside NGC 7259's D25 ellipse was a 'Galactic star'."""
    res = famous["SN2009ip"]
    assert res.stellar_counterpart is True and res.known_variable is True
    assert res.known_star is None  # no parallax/proper-motion evidence: unknown, not Galactic
    host = res.host
    assert host is not None and "NGC 7259" in names_of(host) and host["method"] == "dlr_outside_d25"
    assert host["d_dlr"] == pytest.approx(1.16, abs=0.02) and host["redshift"] == pytest.approx(0.00596, abs=0.0002)
    assert any("Galactic nature not established" in e and "NGC 7259" in e for e in res.evidence)
    assert not any("a Galactic star" in e and "not projected" in e for e in res.evidence)


# ---------------------------------------------------------------------------
# Distances: CF4 group distances and CMB-frame redshifts (recorded + unit)
# ---------------------------------------------------------------------------


def test_m100_takes_the_virgo_group_distance(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: M100 (no CF4 distance of its own) got a 23 Mpc Hubble-flow distance from z = 0.00524;
    Virgo's CF4 group distance is 16.2 Mpc (M100 Cepheids: ~15-16 Mpc)."""
    for name in ("SN2006X", "SN2020oi"):
        host = famous[name].host
        assert host is not None and host["pgc"] == 40153 and host["distance_method"] == "cosmicflows4_group"
        assert host["group"]["nest"] == 100002 and host["group"]["pgc1"] == 41220  # Virgo, dominant galaxy M49
        assert 15.5 < host["distance_mpc"] < 16.5
    host = famous["SN2006X"].host
    # 47.8" -> 3.7 kpc (the Hubble-flow distance gave 5.35 kpc).
    assert host["projected_offset_kpc"] == pytest.approx(3.72, abs=0.05)
    # Uncertainty: CF4's group e_DM (0.008 mag) and Virgo's depth (R2t = 1.44 Mpc / 16.2 Mpc = 9%).
    assert host["distance_uncertainty_fraction"] == pytest.approx(math.hypot(math.log(10) / 5 * 0.008, 1.44 / 16.2), rel=0.02)


def test_host_distance_group_and_cmb_frame() -> None:
    virgo = {"dm": 31.048, "e_dm": 0.008, "r2t_mpc": 1.44, "sigma_v_kms": 670.0, "v3k_kms": 1479.0}
    grp = alerts.host_distance(0.00524, None, ra=185.7287, dec=15.8223, group=virgo)
    assert grp["method"] == "cosmicflows4_group" and grp["angular_diameter_mpc"] == pytest.approx(16.2 / 1.00524**2, rel=2e-3)
    # The galaxy's own CF4 distance wins over its group's.
    own = alerts.host_distance(0.00524, (30.9, 0.05), group=virgo)
    assert own["method"] == "cosmicflows4" and own["distance_modulus"] == 30.9
    # Above z = 0.01 the Hubble flow uses the group's CMB velocity (free of the intra-group dispersion) ...
    far = alerts.host_distance(0.0231, None, ra=194.95, dec=27.98,
                               group={"dm": None, "sigma_v_kms": 1000.0, "v3k_kms": 7194.0})
    from astropy.cosmology import Planck18

    assert far["velocity_frame"] == "cmb_group" and far["redshift_cmb"] == pytest.approx(7194.0 / 299792.458)
    assert far["angular_diameter_mpc"] == pytest.approx(
        Planck18.comoving_transverse_distance(7194.0 / 299792.458).value / 1.0231, rel=1e-9)
    assert far["fractional_uncertainty"] == pytest.approx(300.0 / 7194.0)
    # ... else the galaxy's own redshift in the CMB frame, with the group's dispersion as its uncertainty.
    member = alerts.host_distance(0.0231, None, ra=194.95, dec=27.98, group={"dm": None, "sigma_v_kms": 1000.0})
    assert member["velocity_frame"] == "cmb" and member["peculiar_velocity_kms"] == 1000.0
    # The solar dipole: +369.82 km/s towards the apex, -369.82 km/s away from it.
    apex = SkyCoord(l=264.021 * u.deg, b=48.253 * u.deg, frame="galactic").icrs
    assert alerts.cmb_dipole_velocity_kms(apex.ra.deg, apex.dec.deg) == pytest.approx(369.82, abs=1e-6)
    anti = SkyCoord(l=84.021 * u.deg, b=-48.253 * u.deg, frame="galactic").icrs
    assert alerts.cmb_dipole_velocity_kms(anti.ra.deg, anti.dec.deg) == pytest.approx(-369.82, abs=1e-6)
    assert alerts.cmb_redshift(0.01, apex.ra.deg, apex.dec.deg) == pytest.approx(1.01 / (1 - 369.82 / 299792.458) - 1)
    # Without a position, a heliocentric redshift is used as before.
    assert alerts.host_distance(0.05)["velocity_frame"] == "heliocentric"


# ---------------------------------------------------------------------------
# AGN / blazars (recorded + synthetic)
# ---------------------------------------------------------------------------


def test_blazar_is_a_known_variable_and_agn(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: 3C 273 (SIMBAD BLL) had known_variable False."""
    res = famous["3C273"]
    assert res.known_variable is True and res.known_agn is True and res.known_star is False
    assert any("a blazar, variable by definition" in e for e in res.evidence)


def test_seyfert_is_flagged_as_agn(famous: dict[str, AlertEnrichment]) -> None:
    """NGC 4151 (Sy1): not a variable by type, but flagged as a catalogued AGN (the main alert contaminant)."""
    res = famous["NGC4151_nucleus"]
    assert res.known_agn is True and res.known_variable is False
    assert any("coincident with a catalogued AGN/QSO" in e and "NGC 4151 (type Sy1)" in e for e in res.evidence)
    assert famous["SN2023bee"].known_agn is False


async def test_broker_blazar_label_is_variable_and_agn() -> None:
    """Synthetic: a Fink row whose SIMBAD cross-match is 'BLLac' and whose broker parallax would otherwise count."""
    blazar = Alert.from_dict({**ALERT.as_dict(), "extra": {"simbad_otype": "BLLac", "gaia_parallax_mas": 1.0,
                                                           "gaia_parallax_error_mas": 0.1, "gaia_dr3_name": "Gaia DR3 5"}})
    res = await enricher_with(lambda ra, dec, cats: record_of(ra, dec, cats)).enrich(blazar)
    assert res.known_variable is True and res.known_agn is True
    assert res.known_star is False  # a 10-sigma broker parallax of a blazar is not used
    assert any("broker's SIMBAD cross-match is a galaxy/AGN" in e for e in res.evidence)


# ---------------------------------------------------------------------------
# Gaia source-quality gates (synthetic rows inside a galaxy at 1 Mpc)
# ---------------------------------------------------------------------------

POINT = {"parallax": 0.05, "parallax_error": 0.1, "pmra": 3.0, "pmdec": 4.0, "pmra_error": 0.1, "pmdec_error": 0.1,
         "pmra_pmdec_corr": 0.0, "phot_g_mean_mag": 19.5, "ruwe": 1.0, "astrometric_excess_noise_sig": 0.3,
         "in_galaxy_candidates": False, "classprob_dsc_combmod_star": 0.99, "classprob_dsc_combmod_galaxy": 0.005,
         "classprob_dsc_combmod_quasar": 0.005}


async def _inside_galaxy(data: dict[str, Any], extra_sources: list[dict[str, Any]] | None = None) -> AlertEnrichment:
    gaia = {"source_id": "7", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1, "data": data}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.0005}}

    def answer(ra, dec, cats):
        if "gaia_dr3" in cats:
            return record_of(ra, dec, cats, sources={"gaia_dr3": [gaia], "simbad": list(extra_sources or [])})
        return record_of(ra, dec, cats, sources={"simbad": [galaxy]})

    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)
    return await enricher_with(answer, d25, cf4={1: (25.0, 0.05)}).enrich(ALERT)


@pytest.mark.parametrize(("change", "expected", "why"), [
    ({}, True, "-> Galactic star"),  # a well-behaved point source: 5 mas/yr at 35 sigma is Galactic
    ({"ruwe": 2.0}, False, "not a well-behaved point source (RUWE 2.00)"),
    ({"astrometric_excess_noise_sig": 5.0}, False, "excess-noise significance 5.0"),
    ({"in_galaxy_candidates": True}, False, "a Gaia DR3 galaxy candidate"),
    ({"classprob_dsc_combmod_star": 0.1, "classprob_dsc_combmod_galaxy": 0.9}, False, "Gaia DSC P(galaxy) + P(quasar)"),
    ({"parallax": -0.4}, False, "parallax -4.0 sigma: negative"),
])
async def test_proper_motion_needs_a_point_source(change: dict[str, Any], expected: bool, why: str) -> None:
    res = await _inside_galaxy({**POINT, **change})
    assert res.known_star is expected
    assert any(why in e for e in res.evidence), res.evidence
    assert (res.host is None) is expected


async def test_coincident_galaxy_entry_vetoes_only_a_poorly_fitted_source() -> None:
    """Synthetic: a SIMBAD Sy1 entry at the Gaia position. With excess noise (a nucleus) the astrometry is
    not used; a well-fitted star blended with a catalogued galaxy (ZTF26abxsysn's case) keeps its evidence."""
    sy1 = {"source_id": "NGC 9", "ra": ALERT.ra, "dec": ALERT.dec + 0.3 / 3600, "separation_arcsec": 0.3,
           "data": {"otype": "Sy1"}}
    nucleus = await _inside_galaxy({**POINT, "astrometric_excess_noise_sig": 8.0}, [sy1])
    assert nucleus.known_star is False and nucleus.known_agn is True
    assert any("SIMBAD NGC 9 (type Sy1) at 0.30\" with excess-noise significance 8.0" in e for e in nucleus.evidence)
    blended = await _inside_galaxy(dict(POINT), [sy1])
    assert blended.known_star is True


async def test_luminosity_rule_needs_the_stellar_entry_to_be_the_gaia_source() -> None:
    """Synthetic: G = 14 at DM 29 (M_G = -15). A SIMBAD star 1.2" away (another source) proves nothing;
    at the Gaia position (after the J2000 -> J2016 drift) it does."""
    gaia = {"phot_g_mean_mag": 14.0, "ruwe": 1.0, "astrometric_excess_noise_sig": 0.2}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 5 / 3600, "dec": ALERT.dec, "separation_arcsec": 5.0,
              "data": {"otype": "G", "rvz_redshift": 0.001}}
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)

    def with_star(offset_arcsec: float):
        star = {"source_id": "[X] 2", "ra": ALERT.ra, "dec": ALERT.dec + offset_arcsec / 3600,
                "separation_arcsec": offset_arcsec, "data": {"otype": "*"}}
        row = {"source_id": "8", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.0, "data": gaia}
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={"gaia_dr3": [row], "simbad": [star]}
                                               if "gaia_dr3" in cats else {"simbad": [galaxy]})

    other = await enricher_with(with_star(1.2), d25, cf4={1: (29.0, 0.1)}).enrich(ALERT)
    assert other.known_star is False and other.host is not None
    assert any("no stellar-type entry is this Gaia source" in e for e in other.evidence)
    same = await enricher_with(with_star(0.2), d25, cf4={1: (29.0, 0.1)}).enrich(ALERT)
    assert same.known_star is True and any("= SIMBAD [X] 2 (type *)" in e and "M_G = -15.0" in e for e in same.evidence)


# ---------------------------------------------------------------------------
# DLR host association rules (synthetic)
# ---------------------------------------------------------------------------


def _outer_galaxy(d_dlr: float, a: float = 60.0) -> dict[str, Any]:
    """A round HyperLEDA galaxy of semi-major axis ``a`` whose centre is d_dlr * a north of ALERT."""
    return _d25_galaxy(ALERT.ra, ALERT.dec + d_dlr * a / 3600, ALERT, a)


async def test_outer_dlr_host_is_blocked_by_a_nearer_unsized_galaxy() -> None:
    outer = _outer_galaxy(2.0)
    small = {"source_id": "WISEA J100000.00+020003.0", "ra": ALERT.ra, "dec": ALERT.dec - 3 / 3600,
             "separation_arcsec": 3.0, "data": {"prefphytype": "G", "z": 0.05}}

    def answer_with(rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats else {"ned": rows})

    alone = await enricher_with(answer_with([]), ([outer], None, False)).enrich(ALERT)
    assert alone.host is not None and alone.host["method"] == "dlr_outside_d25" and alone.host["pgc"] == 1
    assert alone.host["d_dlr"] == pytest.approx(2.0, rel=1e-3)
    # A galaxy without a D25 size 3" away (nearer than NGC 1's light radius, 60") may be the host.
    blocked = await enricher_with(answer_with([small]), ([outer], None, False)).enrich(ALERT)
    assert blocked.host is not None and blocked.host["name"] == small["source_id"] and blocked.host["method"] == "nearest"
    assert any("not adopted" in e and "may be the host" in e for e in blocked.evidence)
    # An unsized entry *inside* the D25 galaxy's ellipse is a part of it: no block.
    part = {**small, "source_id": "NGC 1:[X] 7", "ra": ALERT.ra, "dec": ALERT.dec + 100 / 3600, "separation_arcsec": 100.0}
    far = await enricher_with(answer_with([part]), ([_outer_galaxy(1.5, 80.0)], None, False)).enrich(ALERT)
    assert far.host is not None and far.host["method"] == "dlr_outside_d25"
    # Beyond d_DLR = 4 no D25 association is made.
    none = await enricher_with(answer_with([]), ([_outer_galaxy(4.5)], None, False)).enrich(ALERT)
    assert none.host is None and none.host_status == "none_within_radius"


async def test_stellar_counterpart_near_a_nearby_galaxy_is_unknown_not_galactic() -> None:
    """Synthetic (the SN 2009ip situation): a SIMBAD stellar-type source outside every D25 ellipse."""
    star = {"source_id": "[X] 3", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1, "data": {"otype": "s*b"}}

    def answer_with(host_rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={"simbad": [star]} if "gaia_dr3" in cats
                                               else {"simbad": host_rows})

    local = {"source_id": "NGC 2", "ra": ALERT.ra + 40 / 3600, "dec": ALERT.dec, "separation_arcsec": 40.0,
             "data": {"otype": "G", "rvz_redshift": 0.006}}
    near_local = await enricher_with(answer_with([local])).enrich(ALERT)  # in the host cone, |z| < 0.01
    assert near_local.known_star is None and near_local.host is not None
    assert any("Galactic nature not established" in e for e in near_local.evidence)
    distant = {**local, "data": {"otype": "G", "rvz_redshift": 0.08}}
    assert (await enricher_with(answer_with([distant])).enrich(ALERT)).known_star is True
    assert (await enricher_with(answer_with([])).enrich(ALERT)).known_star is True


# ---------------------------------------------------------------------------
# Brokers: Fink SN probability, ALeRCE duplicates, missing photometry, malformed values
# ---------------------------------------------------------------------------


def test_fink_sn_candidate_probability_is_sn_vs_all() -> None:
    """Regression: max(snn_snia_vs_nonia, snn_sn_vs_all) reported the Ia-vs-non-Ia score as P(SN)."""
    row = {"i:objectId": "ZTF26x", "i:ra": 1.0, "i:dec": 2.0, "i:jd": 2461306.8, "i:fid": 1,
           "d:snn_snia_vs_nonia": 0.75, "d:snn_sn_vs_all": 0.25}
    (alert,), _ = FinkZTFBroker.parse_latests([row], "SN candidate")
    assert alert.probability == 0.25 and alert.extra["scores"] == {"snn_snia_vs_nonia": 0.75, "snn_sn_vs_all": 0.25}
    (early,), _ = FinkZTFBroker.parse_latests([{**row, "d:rf_snia_vs_nonia": 0.6}], "Early SN Ia candidate")
    assert early.probability == 0.6
    (other,), _ = FinkZTFBroker.parse_latests([row], "(TNS) SN Ia")
    assert other.probability is None  # a filter/crossmatch-defined class has no single probability


async def test_alerce_rows_repeated_per_classifier_version_are_deduplicated(store: AlertStore) -> None:
    """Recorded lc_classifier AGN answer: 16 rows but 11 objects on the first page."""
    p = params("alerce_duplicates")
    first_page = body("alerce_duplicates", 0)["items"]
    assert len(first_page) == p["limit"] + 1 > len({i["oid"] for i in first_page})
    with Replay("alerce_duplicates") as replay:
        async with offline_client() as client:
            result = await fetch_alerts(client, "alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"],
                                        limit=p["limit"], options=p["options"])
            n_fetch = len(replay.calls)
            ids = [a.object_id for a in result.alerts]
            assert len(ids) == len(set(ids)) == p["limit"] and ids == p["alert_ids"] and result.warnings == []
            # Another page was read to find limit + 1 distinct objects.
            assert sum("/objects/?" in str(c.url) for c in replay.calls) == 2 and result.truncated
            repeated = {a.object_id: a for a in result.alerts if len(a.extra.get("classifier_rows") or []) > 1}
            assert set(repeated) == set(p["repeated"]) and repeated
            actadei = repeated["ZTF18actadei"]
            # Its two lc_classifier versions: hierarchical_rf_1.1.0 (0.84372) and lc_classifier_1.1.13 (0.497556).
            assert actadei.extra["classifier_versions"] == {"hierarchical_rf_1.1.0": 0.84372,
                                                            "lc_classifier_1.1.13": 0.497556}
            assert actadei.extra["classifier_version"] == "lc_classifier_1.1.13"
            assert actadei.probability == 0.497556 and actadei.extra["classifier_choice"] == "newest_version"
            svc = AlertService(store, client, None, clock=lambda: p["until_mjd"])
            polls = [await svc.poll("alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"], limit=p["limit"],
                                    crossmatch=False, options=p["options"]) for _ in range(3)]
            assert len(replay.calls) == 4 * n_fetch
    assert (polls[0].fetched, polls[0].inserted, polls[0].updated) == (p["limit"], p["limit"], 0)
    for later in polls[1:]:  # idempotent: nothing is 'reclassified' by another version's row
        assert (later.inserted, later.updated, later.unchanged) == (0, 0, p["limit"])
    assert len(polls[0].alert_ids) == len(set(polls[0].alert_ids))
    assert all(r["n_updates"] == 0 for r in store.list(limit=100))


def test_alerce_version_key_orders_numeric_versions() -> None:
    key = AlerceBroker.version_key
    assert key("lc_classifier_1.1.13") > key("hierarchical_rf_1.1.0")
    assert key("stamp_classifier_1.0.4") > key("stamp_classifier_1.0.0") and key("1.0.10") > key("1.0.9")
    rows = AlerceBroker.parse_objects({"items": [
        {"oid": "ZTF1", "meanra": 1.0, "meandec": 1.0, "lastmjd": 61300.0, "class": "SN", "probability": 0.4},
        {"oid": "ZTF1", "meanra": 1.0, "meandec": 1.0, "lastmjd": 61300.0, "class": "SN", "probability": 0.7},
    ]})[0]
    assert len(rows) == 1 and rows[0].probability == 0.7 and rows[0].extra["classifier_choice"] == "max_probability"


def _alerce_mock(mock: respx.MockRouter, detections: list[httpx.Response]) -> None:
    item = {"oid": "ZTF26aaaaaab", "meanra": 150.0, "meandec": 2.0, "firstmjd": 61305.0, "lastmjd": 61306.3,
            "class": "SN", "probability": 0.9, "classifier": "stamp_classifier"}
    mock.get(url__startswith="https://api.alerce.online/ztf/v1/objects/?").mock(
        return_value=httpx.Response(200, json={"items": [item]}))
    mock.get("https://api.alerce.online/ztf/v1/objects/ZTF26aaaaaab/detections").mock(side_effect=detections)


DETECTION = [{"mjd": 61306.3, "magpsf": 19.1, "sigmapsf": 0.1, "fid": 2, "isdiffpos": "t", "candid": 123}]


async def test_photometry_missed_by_a_failed_detections_request_is_filled_later(store: AlertStore) -> None:
    """Synthetic (respx): /detections answers 503, then 200, then 503 again."""
    answers = [httpx.Response(503, text="busy"), httpx.Response(200, json=DETECTION), httpx.Response(503, text="busy")]
    with respx.mock(assert_all_mocked=True) as mock:
        _alerce_mock(mock, answers)
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: 61307.0)
            kwargs = {"since_mjd": 61306.0, "until_mjd": 61307.0, "crossmatch": False}
            first = await svc.poll("alerce", **kwargs)
            assert first.inserted == 1 and any("no photometry for ZTF26aaaaaab" in w for w in first.warnings)
            assert store.get("alerce:ZTF26aaaaaab")["magpsf"] is None
            second = await svc.poll("alerce", **kwargs)
            assert (second.updated, second.unchanged) == (1, 0) and second.warnings == []
            row = store.get("alerce:ZTF26aaaaaab")
            assert (row["magpsf"], row["band"], row["is_negative"], row["extra"]["candid"]) == (19.1, "r", False, "123")
            third = await svc.poll("alerce", **kwargs)  # the photometry request fails again: nothing is lost
            assert third.unchanged == 1
    row = store.get("alerce:ZTF26aaaaaab")
    assert row["magpsf"] == 19.1 and row["magpsf_err"] == 0.1 and row["n_updates"] == 1


async def test_reclassified_repoll_without_photometry_keeps_the_stored_photometry(store: AlertStore) -> None:
    alert = Alert("alerce", "ZTF26aaaaaac", 1.0, 2.0, 61306.3, 19.1, "r", "SN", 0.9, "", magpsf_err=0.1,
                  extra={"candid": "5"}, is_negative=False)
    store.upsert(alert)
    again = Alert("alerce", "ZTF26aaaaaac", 1.0, 2.0, 61306.3, None, None, "SN", 0.8, "", extra={})
    assert store.upsert(again) == "updated"
    row = store.get(alert.alert_id)
    assert (row["magpsf"], row["magpsf_err"], row["band"], row["probability"], row["extra"]["candid"]) == \
        (19.1, 0.1, "r", 0.8, "5")


def test_router_survives_malformed_upstream_values(store: AlertStore) -> None:
    """Regression: a Fink 'i:fid' of 'g' or an ALeRCE fid NaN answered 422 with Python internals and
    dropped the whole poll."""
    app = FastAPI()
    app.include_router(router)
    app.state.alert_store = store
    api = TestClient(app)
    fink_rows = [{"i:objectId": "ZTF26aaaaaad", "i:ra": 10.0, "i:dec": 5.0, "i:jd": 2461306.8, "i:magpsf": 19.0,
                  "i:fid": "g", "i:isdiffpos": "t", "d:snn_sn_vs_all": 0.9},
                 {"i:objectId": "ZTF26aaaaaae", "i:ra": 11.0, "i:dec": 5.0, "i:jd": 2461306.7, "i:magpsf": 18.0,
                  "i:fid": 1, "i:isdiffpos": "t", "d:snn_sn_vs_all": 0.8}]
    items = [{"oid": f"ZTF26aaaaaa{c}", "meanra": 150.0 + i, "meandec": 2.0, "firstmjd": 61306.1, "lastmjd": 61306.3,
              "class": "SN", "probability": 0.9} for i, c in enumerate("fg")]
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(200, json=fink_rows))
        mock.get(url__startswith="https://api.alerce.online/ztf/v1/objects/?").mock(
            return_value=httpx.Response(200, json={"items": items}))
        mock.get("https://api.alerce.online/ztf/v1/objects/ZTF26aaaaaaf/detections").mock(
            return_value=httpx.Response(200, text='[{"mjd": 61306.3, "magpsf": 19.2, "fid": NaN, "isdiffpos": "t"}]',
                                        headers={"content-type": "application/json"}))
        mock.get("https://api.alerce.online/ztf/v1/objects/ZTF26aaaaaag/detections").mock(
            return_value=httpx.Response(200, json=DETECTION))
        fink = api.post("/api/v1/alerts/poll", json={"broker": "fink", "since_mjd": 61306.0, "until_mjd": 61307.0,
                                                     "crossmatch": False})
        assert fink.status_code == 200, fink.text
        by_id = {a["object_id"]: a for a in fink.json()["alerts"]}
        assert by_id["ZTF26aaaaaad"]["band"] is None and by_id["ZTF26aaaaaae"]["band"] == "g"
        assert any("malformed filter id 'g'" in w for w in fink.json()["warnings"])
        alerce = api.post("/api/v1/alerts/poll", json={"broker": "alerce", "since_mjd": 61306.0, "until_mjd": 61307.0,
                                                       "crossmatch": False})
        assert alerce.status_code == 200, alerce.text
        rows = {a["object_id"]: a for a in alerce.json()["alerts"]}
        assert set(rows) == {"ZTF26aaaaaaf", "ZTF26aaaaaag"}  # the other alert is stored too
        assert rows["ZTF26aaaaaaf"]["band"] is None and rows["ZTF26aaaaaaf"]["magpsf"] == 19.2
        assert rows["ZTF26aaaaaaf"]["extra"]["malformed_fid"] == "nan" and rows["ZTF26aaaaaag"]["band"] == "r"


def test_malformed_rows_are_skipped_with_a_warning() -> None:
    """Synthetic: a row whose values raise inside the parser is skipped, the others are kept."""
    class Boom(dict):
        def get(self, key, default=None):
            if key == "i:jdstarthist":
                raise TypeError("unsupported operand")
            return super().get(key, default)

    good = {"i:objectId": "ZTF26b", "i:ra": 1.0, "i:dec": 2.0, "i:jd": 2461306.8, "i:fid": 2}
    found, warnings = FinkZTFBroker.parse_latests([Boom({**good, "i:objectId": "ZTF26a"}), good], "SN candidate")
    assert [a.object_id for a in found] == ["ZTF26b"] and any("malformed /latests row" in w for w in warnings)
    lsst = {"r:diaObjectId": 1, "r:ra": 1.0, "r:dec": 2.0, "r:midpointMjdTai": 61200.5, "r:psfFlux": 1000.0,
            "f:clf_cats_class": float("nan")}
    (row,), _ = FinkLSSTBroker.parse_tags([lsst], "t")
    assert row.extra["cats_class"] is None and row.classification == "t"


async def test_watch_records_an_unexpected_broker_exception_and_continues(store: AlertStore) -> None:
    svc = AlertService(store, offline_client(), None, clock=lambda: 61311.0)
    calls: list[str] = []

    async def poll(name: str, **kwargs: Any) -> alerts.PollResult:
        calls.append(name)
        if name == "fink" and len(calls) < 3:
            raise ValueError("invalid literal for int() with base 10: 'g'")
        return alerts.PollResult(broker=name, since_mjd=61310.0, until_mjd=61311.0)

    svc.poll = poll  # type: ignore[method-assign]
    results = await svc.watch(["fink", "alerce"], interval_seconds=0.01, crossmatch=False, iterations=2)
    assert calls == ["fink", "alerce", "fink", "alerce"]
    assert results[0].error == "unexpected ValueError: invalid literal for int() with base 10: 'g'"
    assert results[1].error is None and results[2].error is None


# ---------------------------------------------------------------------------
# Service: one enrichment per alert at a time; executor limits are inherited
# ---------------------------------------------------------------------------


class SlowEnricher:
    match_radius_arcsec, host_radius_arcsec, catalogs = 2.0, 60.0, ["gaia_dr3"]

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def enrich(self, alert: Alert) -> AlertEnrichment:
        self.calls.append(alert.alert_id)
        self.started.set()
        await self.release.wait()
        return AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"],
                               known_star=False)


async def test_recrossmatch_joins_a_running_background_enrichment(store: AlertStore) -> None:
    """Regression: POST /{id}/crossmatch (enrich_alert) ran concurrently with a background poll's crossmatch."""
    enricher = SlowEnricher()
    store.upsert(ALERT)
    svc = AlertService(store, offline_client(), enricher)  # type: ignore[arg-type]
    background = asyncio.create_task(svc.crossmatch_in_background([ALERT]))
    await enricher.started.wait()
    assert svc.in_flight == {ALERT.alert_id}
    joined = asyncio.create_task(svc.enrich_alert(ALERT))
    await asyncio.sleep(0.05)
    enricher.release.set()
    enrichment, stored = await asyncio.wait_for(joined, timeout=10)
    await background
    assert enricher.calls == [ALERT.alert_id] and stored == "stored" and enrichment.status == "done"
    assert store.get(ALERT.alert_id)["crossmatch_attempts"] == 1 and svc.in_flight == frozenset()
    # Cancelling a joined caller does not cancel the shared enrichment.
    enricher2 = SlowEnricher()
    svc2 = AlertService(store, offline_client(), enricher2)  # type: ignore[arg-type]
    first = asyncio.create_task(svc2.enrich_alert(ALERT))
    await enricher2.started.wait()
    second = asyncio.create_task(svc2.enrich_alert(ALERT))
    await asyncio.sleep(0.01)
    second.cancel()
    enricher2.release.set()
    assert (await first)[0].status == "done" and enricher2.calls == [ALERT.alert_id]
    assert store.get(ALERT.alert_id)["crossmatch_attempts"] == 2


def test_router_recrossmatch_joins_a_running_enrichment(store: AlertStore) -> None:
    """Synthetic: two concurrent POST /{id}/crossmatch requests run one enrichment."""
    calls: list[str] = []

    class Enricher:
        match_radius_arcsec, host_radius_arcsec, catalogs = 2.0, 60.0, ["gaia_dr3"]

        async def enrich(self, alert: Alert) -> AlertEnrichment:
            calls.append(alert.alert_id)
            await asyncio.sleep(0.2)
            return AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"])

    store.upsert(ALERT)
    app = FastAPI()
    app.include_router(router)
    app.state.alert_service = AlertService(store, offline_client(), Enricher())  # type: ignore[arg-type]

    async def both() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            answers = await asyncio.gather(*(client.post(f"/api/v1/alerts/{ALERT.alert_id}/crossmatch") for _ in range(2)))
        return [a.status_code for a in answers]

    assert asyncio.run(both()) == [200, 200]
    assert calls == [ALERT.alert_id] and store.get(ALERT.alert_id)["crossmatch_attempts"] == 1


def test_enricher_inherits_the_service_executor_limits() -> None:
    """Regression: the derived match/host services dropped timeout_cap (CATALOG_TIMEOUT_CAP_SECONDS)."""
    from astrometry import AssociationConfig

    config = AssociationConfig()
    service = make_service(offline_client(), timeout=20.0, timeout_cap=5.0, association_config=config, max_concurrency=2)
    enricher = AlertEnricher(service)
    for derived in (enricher.match_service, enricher.host_service):
        assert derived.executor.timeout_cap == 5.0 and derived.executor.timeout == 20.0
        assert derived.association_config is config and derived.max_concurrency == 2
    # Gaia DR3 (90 s) and HyperLEDA (60 s) are capped at 5 s.
    gaia = enricher.match_service.registry.get("gaia_dr3")
    leda = enricher.host_service.registry.get(alerts.HYPERLEDA)
    assert enricher.match_service.executor.catalog_limit(gaia) == 5.0
    assert enricher.host_service.executor.catalog_limit(leda) == 5.0


def test_match_search_fetches_the_gaia_quality_columns() -> None:
    from models import CatalogRegistry, validate_target
    from providers import TapProvider

    adql = TapProvider().build_adql(alerts.MatchSearchRegistry(CatalogRegistry()).get("gaia_dr3"),
                                    validate_target(1.0, 2.0), 2.0)
    for column in alerts.GAIA_QUALITY_COLUMNS:
        assert column in adql
    exchanges = meta("xmatch_famous")["exchanges"]
    assert any("astrometric_excess_noise_sig" in e["request_body"] for e in exchanges)


async def test_known_agn_is_stored_and_listed(store: AlertStore) -> None:
    store.upsert(ALERT)
    enrichment = AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["simbad"],
                                 known_agn=True, known_variable=False, known_star=False, is_new=False)
    assert store.set_enrichment(ALERT.alert_id, enrichment) == "stored"
    row = store.get(ALERT.alert_id)
    assert row["known_agn"] is True and row["enrichment"]["known_agn"] is True
    assert json.loads(json.dumps(row, default=str))["known_agn"] is True
