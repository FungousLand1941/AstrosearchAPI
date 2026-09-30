"""Offline regression tests for alerts.py (review round 3).

Recorded fixtures (``tests/fixtures/alerts``, recorded live by ``record_alerts.py``):

* ``xmatch_famous``: Gaia DR3 / SIMBAD / NED / HyperLEDA / Cosmicflows-4 answers for galaxy nuclei
  whose Gaia 5-parameter solutions have spurious, formally significant proper motions (M87,
  NGC 4395, M106, the Seyfert 1 nuclei NGC 3783, NGC 7469, NGC 6814, NGC 3516, NGC 4151,
  ASASSN-14ko's host), SNe on star clusters (SN 2020oi, SN 2004dj), NGC 3115, M83, SN 2009ip,
  hosts outside their D25 ellipse (SN 2023bee, SN 2018aoz), Virgo members without their own CF4
  distance (M100: SN 2006X) and the blazar 3C 273;
* ``alerce_duplicates``: ALeRCE ``lc_classifier`` AGN rows repeated once per classifier version.
* ``xmatch_final``: the Gaia DR3 / SIMBAD / NED / HyperLEDA answers at fink:ZTF26abuxdqd, a transient on a
  slowly moving, high-confidence Gaia star far from every galaxy.

The remaining tests use small synthetic inputs and say so.
"""

from __future__ import annotations

import argparse
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
    FetchResult,
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
        Planck18.clone(H0=74.6).comoving_transverse_distance(7194.0 / 299792.458).value / 1.0231, rel=1e-9)
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


def test_hubble_flow_distances_are_on_the_cosmicflows4_scale() -> None:
    """Regression: CF4 distances (H0 = 74.6 zero point) were used below z = 0.01 and Planck 2018 (H0 = 67.66)
    above it, so a host's distance and offset jumped by ~10% where the method changed."""
    # A galaxy on the Hubble flow at cz_CMB = 2990 km/s: its CF4 distance is ~cz / 74.6 = 40.1 Mpc.
    z = 0.00997
    dm_cf4 = 5 * math.log10(299792.458 * z / 74.6 * 1e5)
    below = alerts.host_distance(z, (dm_cf4, 0.05))
    above = alerts.host_distance(z + 0.00006)  # just above z = 0.01: the Hubble flow (heliocentric, no position)
    assert below["method"] == "cosmicflows4" and above["method"] == "hubble_flow"
    assert above["hubble_constant_kms_mpc"] == pytest.approx(74.6)
    # Continuous to the redshift step and the (1+z) terms (< 1.5%), not the 10% H0 jump.
    assert above["angular_diameter_mpc"] / below["angular_diameter_mpc"] == pytest.approx(1.0, abs=0.015)
    from astropy.cosmology import Planck18

    planck = Planck18.angular_diameter_distance(z + 0.00006).value
    assert planck / above["angular_diameter_mpc"] == pytest.approx(74.6 / 67.66, rel=0.002)


# ---------------------------------------------------------------------------
# AGN / blazars (recorded + synthetic)
# ---------------------------------------------------------------------------


def test_blazar_is_a_known_variable_and_agn(famous: dict[str, AlertEnrichment]) -> None:
    """Regression: 3C 273 (SIMBAD BLL) had known_variable False."""
    res = famous["3C273"]
    assert res.known_variable is True and res.known_agn is True and res.known_star is False
    assert any("a blazar, variable by definition" in e for e in res.evidence)
    # Review round 3: its host is 3C 273 itself (z = 0.158), not the z = 0.0053 dwarf 10.8" away.
    host = res.host
    assert host is not None and host["name"] == "3C 273" and host["method"] == "agn_nucleus"
    assert host["redshift"] == pytest.approx(0.1576, abs=1e-4) and host["distance_mpc"] > 400
    assert res.transient_redshift == host["redshift"]


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


# The same 5 mas/yr proper motion measured at 10 sigma: significant, but not decisive (< PM_SNR_DECISIVE).
MARGINAL = {**POINT, "pmra_error": 0.4, "pmdec_error": 0.6}
DSC_GALAXY = {"classprob_dsc_combmod_star": 0.1, "classprob_dsc_combmod_galaxy": 0.9}


@pytest.mark.parametrize(("data", "expected", "why"), [
    (POINT, True, "-> Galactic star"),  # a well-behaved point source: 5 mas/yr at 50 sigma is Galactic
    (MARGINAL, True, "-> Galactic star"),  # ... and at 10 sigma
    # A poor single-star fit (a nucleus, a cluster -- or a binary): a < 20 sigma proper motion is not used ...
    ({**MARGINAL, "ruwe": 2.0}, False, "not a well-behaved point source (RUWE 2.00)"),
    ({**MARGINAL, "astrometric_excess_noise_sig": 5.0}, False, "excess-noise significance 5.0"),
    # ... a >= 20 sigma one is real (a binary's; the spurious ones of nuclei and clusters stay < 18 sigma).
    ({**POINT, "ruwe": 2.0}, True, "real although the single-star model fits it poorly (RUWE 2.00; a binary)"),
    # Gaia's extragalactic classification vetoes marginal astrometry ...
    ({**MARGINAL, "in_galaxy_candidates": True}, False, "a Gaia DR3 galaxy candidate"),
    ({**MARGINAL, **DSC_GALAXY}, False, "Gaia DSC P(galaxy) + P(quasar)"),
    # ... and that of a poorly fitted source, however significant ...
    ({**POINT, **DSC_GALAXY, "ruwe": 2.0}, False, "Gaia DSC P(galaxy) + P(quasar) = 0.905"),
    # ... but not the decisive astrometry of a well-fitted point source (DSC's classes have a low purity).
    ({**POINT, "in_galaxy_candidates": True}, True, "decisive astrometry (DSC's galaxy/quasar classes have a low purity"),
    ({**POINT, **DSC_GALAXY}, True, "-> Galactic star"),
    ({**POINT, "parallax": -0.4}, False, "parallax -4.0 sigma: negative"),
])
async def test_proper_motion_needs_a_point_source_or_decisive_significance(data: dict[str, Any], expected: bool,
                                                                          why: str) -> None:
    """Synthetic: inside a galaxy at 1 Mpc (proper-motion limit 0.16 mas/yr)."""
    res = await _inside_galaxy(data)
    assert res.status == "done"
    assert res.known_star is expected
    assert any(why in e for e in res.evidence), res.evidence
    assert (res.host is None) is expected


async def _alone(data: dict[str, Any], ra: float = ALERT.ra, dec: float = ALERT.dec,
                 extra_sources: list[dict[str, Any]] | None = None) -> AlertEnrichment:
    """Synthetic: a Gaia source at the alert with no galaxy in the host cone nor any D25 ellipse."""
    alert = Alert.from_dict({**ALERT.as_dict(), "ra": ra, "dec": dec})
    gaia = {"source_id": "9", "ra": ra, "dec": dec, "separation_arcsec": 0.1, "data": data}

    def answer(qra, qdec, cats):
        if "gaia_dr3" in cats:
            return record_of(qra, qdec, cats, sources={"gaia_dr3": [gaia], "simbad": list(extra_sources or [])})
        return record_of(qra, qdec, cats)

    return await enricher_with(answer).enrich(alert)


# A well-behaved point source moving 2.70 mas/yr at 18 sigma (the Galactic star at l = 48, b = -10 that the LMC-distance
# limit of 3.19 mas/yr left 'not a star'), without a significant parallax.
SLOW = {**POINT, "pmra": 1.62, "pmdec": 2.16, "pmra_error": 0.15, "pmdec_error": 0.15, "classprob_dsc_combmod_star": 0.5,
        "classprob_dsc_combmod_galaxy": 0.25, "classprob_dsc_combmod_quasar": 0.25}
QUIET = {**POINT, "pmra": 0.05, "pmdec": 0.0, "pmra_error": 0.1, "pmdec_error": 0.1}  # no significant motion


async def test_significant_motion_far_from_every_galaxy_is_galactic_and_probable_stars_are_unknown() -> None:
    """Synthetic. Far from every galaxy that could hold stars (no host, no D25 ellipse, far from the Magellanic
    Clouds, M31, M33 and the Local Group dwarfs) any significant motion of a well-behaved point source is a
    Galactic star's. Near the LMC the same slow motion is below the 750 km/s limit: the source then stays a
    probable star of unknown nature (None) -- as does a well-behaved point source that Gaia's DSC calls a star
    without decisive astrometry -- unless Gaia classifies it as extragalactic, its parallax is negative or a
    catalogued AGN lies on it (a BL Lac nucleus is a well-fitted 'star' for DSC: PKS 2155-304)."""
    isolated = await _alone(SLOW)
    assert isolated.status == "done" and isolated.known_star is True and isolated.host_status == "not_applicable_star"
    assert any("2.70 mas/yr (18 sigma), with no galaxy associated" in e and "-> Galactic star" in e
               for e in isolated.evidence), isolated.evidence
    near_lmc = await _alone(SLOW, ra=80.0, dec=-68.0)
    assert near_lmc.known_star is None
    assert any("well-behaved point source" in e and "2.70 mas/yr proper motion (18 sigma)" in e
               and "Galactic nature not established" in e for e in near_lmc.evidence), near_lmc.evidence
    probable = await _alone(QUIET)
    assert probable.known_star is None
    assert any("DSC classifies as a star (P = 0.990)" in e and "Galactic nature not established" in e
               for e in probable.evidence), probable.evidence
    for data in ({**QUIET, **DSC_GALAXY}, {**QUIET, "in_galaxy_candidates": True}, {**QUIET, "parallax": -0.4}):
        res = await _alone(data)
        assert res.status == "done" and res.known_star is False, data
        assert not any("Galactic nature not established" in e for e in res.evidence)
    bllac = {"source_id": "PKS X", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.0,
             "data": {"otype": "BLL", "rvz_redshift": 0.116}}
    nucleus = await _alone(QUIET, extra_sources=[bllac])
    assert nucleus.known_star is False and nucleus.known_agn is True and nucleus.known_variable is True


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
    outer = _outer_galaxy(1.8)
    small = {"source_id": "WISEA J100000.00+020003.0", "ra": ALERT.ra, "dec": ALERT.dec - 3 / 3600,
             "separation_arcsec": 3.0, "data": {"prefphytype": "G", "z": 0.05}}

    def answer_with(rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats else {"ned": rows})

    alone = await enricher_with(answer_with([]), ([outer], None, False)).enrich(ALERT)
    assert alone.status == "done" and alone.host is not None and alone.host["method"] == "dlr_outside_d25"
    assert alone.host["pgc"] == 1 and alone.host["d_dlr"] == pytest.approx(1.8, rel=1e-3)
    # A galaxy without a D25 size 3" away (nearer than NGC 1's light radius, 60") may be the host.
    blocked = await enricher_with(answer_with([small]), ([outer], None, False)).enrich(ALERT)
    assert blocked.host is not None and blocked.host["name"] == small["source_id"] and blocked.host["method"] == "nearest"
    assert any("not adopted" in e and "may be the host" in e for e in blocked.evidence)
    # An unsized entry *inside* the D25 galaxy's ellipse is a part of it: no block.
    part = {**small, "source_id": "NGC 1:[X] 7", "ra": ALERT.ra, "dec": ALERT.dec + 100 / 3600, "separation_arcsec": 100.0}
    far = await enricher_with(answer_with([part]), ([_outer_galaxy(1.5, 80.0)], None, False)).enrich(ALERT)
    assert far.host is not None and far.host["method"] == "dlr_outside_d25"
    # Beyond d_DLR = 2 (Gupta et al.'s 4 second-moment radii) no D25 association is made: up to 4 D25 radii the
    # galaxy is reported as a possible association ('unassociated'), beyond that not at all.
    possible = await enricher_with(answer_with([]), ([_outer_galaxy(3.0)], None, False)).enrich(ALERT)
    assert possible.status == "done" and possible.host is None and possible.host_status == "unassociated"
    assert any("possible association, not adopted: PGC 1" in e and "d_DLR = 3.00" in e for e in possible.evidence)
    none = await enricher_with(answer_with([]), ([_outer_galaxy(4.5)], None, False)).enrich(ALERT)
    assert none.status == "done" and none.host is None and none.host_status == "none_within_radius"


async def test_stellar_counterpart_near_a_nearby_galaxy_is_unknown_not_galactic() -> None:
    """Synthetic (the SN 2009ip situation): a SIMBAD stellar-type source outside every D25 ellipse."""
    star = {"source_id": "[X] 3", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1, "data": {"otype": "s*b"}}

    def answer_with(host_rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={"simbad": [star]} if "gaia_dr3" in cats
                                               else {"simbad": host_rows})

    local = {"source_id": "NGC 2", "ra": ALERT.ra + 40 / 3600, "dec": ALERT.dec, "separation_arcsec": 40.0,
             "data": {"otype": "G", "rvz_redshift": 0.006}}
    near_local = await enricher_with(answer_with([local])).enrich(ALERT)  # in the host cone, |z| < 0.01
    assert near_local.status == "done" and near_local.known_star is None
    assert any("Galactic nature not established" in e for e in near_local.evidence)
    # NGC 2 (no D25 size) 40" away is not adopted as host (review round 3): it lies within a typical 8 kpc light
    # radius at z = 0.006, but its chance-coincidence probability is 0.36.
    assert near_local.host is None and near_local.host_status == "unassociated"
    assert any("NGC 2" in e and "does not make it the host" in e for e in near_local.evidence)
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
    """Recorded lc_classifier AGN answer: 16 rows but 11 objects on the first page; every kept object is
    classified by its newest classifier version (/probabilities)."""
    p = params("alerce_duplicates")
    first_page = body("alerce_duplicates", 0)["items"]
    assert len(first_page) == p["limit"] + 1 > len({i["oid"] for i in first_page})
    # Two objects of the window are AGN only in a superseded version (one /objects/ row each): the newest,
    # lc_classifier_1.1.13, ranks CV/Nova and Blazar first. They are dropped, with a warning.
    dropped = {"ZTF18acusetg": "CV/Nova", "ZTF21aaqvvvt": "Blazar"}
    assert all(sum(i["oid"] == oid for i in first_page) == 1 for oid in dropped)
    with Replay("alerce_duplicates") as replay:
        async with offline_client() as client:
            result = await fetch_alerts(client, "alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"],
                                        limit=p["limit"], options=p["options"])
            n_fetch = len(replay.calls)
            ids = [a.object_id for a in result.alerts]
            assert len(ids) == len(set(ids)) == p["limit"] - len(dropped) and ids == p["alert_ids"]
            assert not set(dropped) & set(ids)
            (warning,) = result.warnings
            assert "2 object(s) dropped" in warning and all(f"{oid} ({cls} " in warning for oid, cls in dropped.items())
            # Another page was read to find limit + 1 distinct objects, by keyset: it ends at the exact MJD of
            # the first page's last row (rows of one exposure come in a different order on every request).
            pages = [c for c in replay.calls if "/objects/?" in str(c.url)]
            assert len(pages) == 2 and result.truncated
            assert pages[1].url.params.get_list("lastmjd")[1] == repr(min(i["lastmjd"] for i in first_page))
            assert "page" not in pages[1].url.params
            # One /probabilities request per kept object (dropped ones included), one /detections per stored one.
            probability_calls = [c for c in replay.calls if c.url.path.endswith("/probabilities")]
            assert len(probability_calls) == p["limit"] and result.requests == 2 + p["limit"] + len(ids)
            assert {a.object_id: a.probability for a in result.alerts} == p["probabilities"]
            assert all(a.extra["classifier_choice"] == "newest_version" for a in result.alerts)
            assert all(a.extra["classifier_version"] == "lc_classifier_1.1.13" for a in result.alerts)
            repeated = {a.object_id: a for a in result.alerts if len(a.extra.get("classifier_rows") or []) > 1}
            assert set(repeated) == set(p["repeated"]) and repeated
            actadei = repeated["ZTF18actadei"]
            # Its two lc_classifier versions: hierarchical_rf_1.1.0 (0.84372) and lc_classifier_1.1.13 (0.497556).
            assert actadei.extra["classifier_versions"] == {"hierarchical_rf_1.1.0": 0.84372,
                                                            "lc_classifier_1.1.13": 0.497556}
            assert actadei.probability == 0.497556
            svc = AlertService(store, client, None, clock=lambda: p["until_mjd"])
            polls = [await svc.poll("alerce", since_mjd=p["since_mjd"], until_mjd=p["until_mjd"], limit=p["limit"],
                                    crossmatch=False, options=p["options"]) for _ in range(3)]
            assert len(replay.calls) == 4 * n_fetch
    n = len(ids)
    assert (polls[0].fetched, polls[0].inserted, polls[0].updated) == (n, n, 0)
    for later in polls[1:]:  # idempotent: nothing is 'reclassified' by another version's row
        assert (later.inserted, later.updated, later.unchanged) == (0, 0, n)
    assert len(polls[0].alert_ids) == len(set(polls[0].alert_ids))
    assert all(r["n_updates"] == 0 for r in store.list(limit=100))


class ShufflingAlerce:
    """Synthetic ALeRCE /objects/ server: rows sorted by lastmjd DESC, rows of one MJD (one exposure) in a
    new random order on every request -- as the live API does (ZTF18abnrdci's two version rows were read
    on one poll and split by a page offset on the next, flipping its stored probability).

    ``objects``: oid -> (lastmjd, probabilities of the versions ranking AGN first, oldest version first);
    ``newest_class``: oid -> (class, probability) ranked first by the newest version instead (the object's
    AGN row then comes from the older version only); ``probability_status``: HTTP status of /probabilities.
    """

    VERSIONS = ("hierarchical_rf_1.1.0", "lc_classifier_1.1.13")

    def __init__(self, objects: dict[str, tuple[float, tuple[float, ...]]], seed: int = 1,
                 newest_class: dict[str, tuple[str, float]] | None = None) -> None:
        self.objects = objects
        self.newest_class = newest_class or {}
        self.random = __import__("random").Random(seed)
        self.pages: list[httpx.URL] = []
        self.probability_calls: list[str] = []
        self.probability_status = 200

    def rows(self) -> list[dict[str, Any]]:
        return [{"oid": oid, "meanra": 10.0 + i, "meandec": 5.0, "firstmjd": mjd - 30.0, "lastmjd": mjd, "class": "AGN",
                 "probability": prob, "classifier": "lc_classifier"}
                for i, (oid, (mjd, probs)) in enumerate(self.objects.items()) for prob in probs]

    def objects_page(self, request: httpx.Request) -> httpx.Response:
        self.pages.append(request.url)
        lo, hi = (float(v) for v in request.url.params.get_list("lastmjd"))
        size, page = int(request.url.params["page_size"]), int(request.url.params.get("page", "1"))
        rows = [r for r in self.rows() if lo <= r["lastmjd"] <= hi]
        self.random.shuffle(rows)
        rows.sort(key=lambda r: -r["lastmjd"])  # stable: ties keep the shuffled order
        return httpx.Response(200, json={"items": rows[(page - 1) * size:page * size]})

    def probabilities(self, request: httpx.Request) -> httpx.Response:
        oid = request.url.path.split("/")[-2]
        self.probability_calls.append(oid)
        if self.probability_status != 200:
            return httpx.Response(self.probability_status, text="Service Unavailable")
        _, probs = self.objects[oid]
        if oid in self.newest_class:
            cls, prob = self.newest_class[oid]
            rows = [{"classifier_name": "lc_classifier", "classifier_version": self.VERSIONS[0], "class_name": "AGN",
                     "probability": probs[0], "ranking": 1},
                    {"classifier_name": "lc_classifier", "classifier_version": self.VERSIONS[1], "class_name": cls,
                     "probability": prob, "ranking": 1},
                    {"classifier_name": "lc_classifier", "classifier_version": self.VERSIONS[1], "class_name": "AGN",
                     "probability": 0.1, "ranking": 2}]
            return httpx.Response(200, json=rows)
        versions = self.VERSIONS[-len(probs):] if len(probs) < len(self.VERSIONS) else self.VERSIONS
        return httpx.Response(200, json=[{"classifier_name": "lc_classifier", "classifier_version": v, "class_name": "AGN",
                                          "probability": p, "ranking": 1} for v, p in zip(versions, probs, strict=True)])

    def mount(self, mock: respx.MockRouter) -> None:
        base = "https://api.alerce.online/ztf/v1/objects/"
        mock.get(url__startswith=base + "?").mock(side_effect=self.objects_page)
        mock.get(url__regex=r".*/objects/[^/]+/probabilities.*").mock(side_effect=self.probabilities)
        mock.get(url__regex=r".*/objects/[^/]+/detections$").mock(return_value=httpx.Response(200, json=[]))


AGN_OPTIONS = {"classifier": "lc_classifier", "class_name": "AGN", "mjd_field": "lastmjd"}


async def test_alerce_probabilities_do_not_depend_on_the_order_of_tied_rows(store: AlertStore) -> None:
    """Synthetic: an exposure's rows (MJD 61311.5) straddle the first page; with page offsets an object's
    second version row was sometimes skipped or read twice, so its probability flipped between the two
    versions from poll to poll ('updated' without a new detection). Keyset paging reads the tie again."""
    server = ShufflingAlerce({
        "ZTF26aaaaaaa": (61311.9, (0.7,)),
        "ZTF26aaaaaab": (61311.5, (0.9, 0.3)),  # versions: hierarchical_rf 0.9, lc_classifier (newest) 0.3
        "ZTF26aaaaaac": (61311.5, (0.6,)),
        "ZTF26aaaaaad": (61311.5, (0.8, 0.4)),
        "ZTF26aaaaaae": (61311.5, (0.55,)),
        "ZTF26aaaaaaf": (61311.2, (0.95, 0.35)),
        "ZTF26aaaaaag": (61311.1, (0.5,)),
        "ZTF26aaaaaah": (61311.0, (0.5,)),
    })
    window = {"since_mjd": 61310.0, "until_mjd": 61312.0, "limit": 5, "options": AGN_OPTIONS, "crossmatch": False}
    newest = {"ZTF26aaaaaaa": 0.7, "ZTF26aaaaaab": 0.3, "ZTF26aaaaaac": 0.6, "ZTF26aaaaaad": 0.4, "ZTF26aaaaaae": 0.55}
    with respx.mock(assert_all_mocked=True) as mock:
        server.mount(mock)
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: 61312.0)
            polls = [await svc.poll("alerce", **window) for _ in range(8)]
    assert all(sorted(p.alert_ids) == sorted(f"alerce:{oid}" for oid in newest) for p in polls)
    assert all(p.truncated and p.warnings[1:] == [] for p in polls)
    assert (polls[0].inserted, polls[0].updated) == (5, 0)
    for later in polls[1:]:
        assert (later.inserted, later.updated, later.unchanged) == (0, 0, 5)
    rows = {r["object_id"]: r for r in store.list(limit=10)}
    assert {oid: r["probability"] for oid, r in rows.items()} == newest
    assert all(r["n_updates"] == 0 for r in rows.values())
    assert rows["ZTF26aaaaaab"]["extra"]["classifier_version"] == "lc_classifier_1.1.13"
    # The pages after the first end at the exact MJD of the previous page's last row, not at a page offset
    # (the second request re-reads the exposure at 61311.5; the third steps through it by offset).
    uppers = [url.params.get_list("lastmjd")[1] for url in server.pages[:3]]
    assert uppers == ["61312.000000", "61311.5", "61311.5"]
    assert [url.params.get("page") for url in server.pages[:3]] == [None, None, "2"]


async def test_alerce_keyset_pages_through_a_tie_larger_than_a_page(store: AlertStore) -> None:
    """Synthetic: 4 objects with 2 version rows each share one MJD (8 rows, more than a page of limit + 1 = 6):
    a page of the tie holds at most 4 < 6 distinct objects, so the walk must step through the tie by page
    offsets to reach the older objects. Regression: the former test's single-row objects filled the first
    page with distinct objects, so the offset path never ran."""
    tied = {f"ZTF26aaaab{c}a": (61311.5, (0.9, 0.3)) for c in "abcd"}
    server = ShufflingAlerce(tied | {"ZTF26aaaacaa": (61311.4, (0.8,)), "ZTF26aaaadaa": (61311.3, (0.7,))}, seed=3)
    with respx.mock(assert_all_mocked=True) as mock:
        server.mount(mock)
        async with offline_client() as client:
            result = await fetch_alerts(client, "alerce", since_mjd=61310.0, until_mjd=61312.0, limit=5, options=AGN_OPTIONS)
    requests = [(url.params.get_list("lastmjd")[1], url.params.get("page")) for url in server.pages]
    assert requests == [("61312.000000", None), ("61311.5", None), ("61311.5", "2")]
    ids = [a.object_id for a in result.alerts]
    assert len(ids) == len(set(ids)) == 5 and set(tied) <= set(ids) and "ZTF26aaaacaa" in ids
    assert result.truncated and result.boundary_mjd == 61311.4 and result.warnings == []
    # Every tied object was seen with both version rows and took its newest version's probability.
    for oid in tied:
        alert = next(a for a in result.alerts if a.object_id == oid)
        assert alert.probability == 0.3 and alert.extra["classifier_versions"] == {"hierarchical_rf_1.1.0": 0.9,
                                                                                   "lc_classifier_1.1.13": 0.3}


async def test_alerce_object_whose_newest_version_ranks_another_class_is_dropped(store: AlertStore) -> None:
    """Synthetic: ZTF26aaaaaab's only AGN row comes from hierarchical_rf_1.1.0; the newest version ranks Blazar
    first. Regression: such single-row objects were stored as 'AGN' with the obsolete probability, silently
    (live: 8 of 30 lc_classifier SNIa objects of a lastmjd window, e.g. ZTF18abvvwjv, an LPV at 0.659)."""
    server = ShufflingAlerce({"ZTF26aaaaaaa": (61311.9, (0.7,)), "ZTF26aaaaaab": (61311.8, (0.85,)),
                              "ZTF26aaaaaac": (61311.7, (0.6, 0.5))}, newest_class={"ZTF26aaaaaab": ("Blazar", 0.62)})
    with respx.mock(assert_all_mocked=True) as mock:
        server.mount(mock)
        async with offline_client() as client:
            result = await fetch_alerts(client, "alerce", since_mjd=61310.0, until_mjd=61312.0, limit=5, options=AGN_OPTIONS)
            svc = AlertService(store, client, None, clock=lambda: 61312.0)
            poll = await svc.poll("alerce", since_mjd=61310.0, until_mjd=61312.0, limit=5, options=AGN_OPTIONS,
                                  crossmatch=False)
    assert [a.object_id for a in result.alerts] == ["ZTF26aaaaaaa", "ZTF26aaaaaac"]
    assert sorted(server.probability_calls[:3]) == ["ZTF26aaaaaaa", "ZTF26aaaaaab", "ZTF26aaaaaac"]  # all resolved
    (warning,) = result.warnings
    assert "1 object(s) dropped" in warning and "ZTF26aaaaaab (Blazar 0.62 in lc_classifier_1.1.13)" in warning
    assert poll.inserted == 2 and store.get("alerce:ZTF26aaaaaab") is None
    single = store.get("alerce:ZTF26aaaaaaa")  # a single row, resolved too
    assert single["extra"]["classifier_choice"] == "newest_version" and single["extra"]["classifier_version"] == \
        "lc_classifier_1.1.13"


async def test_alerce_failed_version_lookup_on_a_repoll_keeps_the_stored_probability(store: AlertStore) -> None:
    """Synthetic: /probabilities answers 200, then 503, then 200 on three polls of the same detections.
    Regression: the 503 poll stored the highest (obsolete) row probability as a reclassification and the next
    poll flipped it back (2 'updates' without a new detection)."""
    server = ShufflingAlerce({"ZTF26aaaaaab": (61311.5, (0.9, 0.3)), "ZTF26aaaaaac": (61311.4, (0.6,))})
    window = {"since_mjd": 61310.0, "until_mjd": 61312.0, "limit": 5, "options": AGN_OPTIONS, "crossmatch": False}
    with respx.mock(assert_all_mocked=True) as mock:
        server.mount(mock)
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: 61312.0)
            first = await svc.poll("alerce", **window)
            server.probability_status = 503
            outage = await svc.poll("alerce", **window)
            server.probability_status = 200
            after = await svc.poll("alerce", **window)
    assert first.inserted == 2 and first.warnings == []
    assert (outage.updated, outage.unchanged) == (0, 2)
    assert sum("classifier versions unavailable" in w for w in outage.warnings) == 2
    assert (after.updated, after.unchanged) == (0, 2)
    row = store.get("alerce:ZTF26aaaaaab")
    assert row["probability"] == 0.3 and row["n_updates"] == 0
    assert row["extra"]["classifier_choice"] == "newest_version" and row["extra"]["classifier_version"] == "lc_classifier_1.1.13"
    # A newer detection whose lookup fails is stored as it came (the stored version belongs to an older detection).
    newer = Alert.from_dict({**Alert.from_dict(row).as_dict(), "mjd": 61311.6, "probability": 0.9,
                             "extra": {"classifier_choice": "unresolved"}})
    assert store.upsert(newer) == "updated" and store.get("alerce:ZTF26aaaaaab")["probability"] == 0.9


def test_alerce_version_key_orders_numeric_versions() -> None:
    key = AlerceBroker.version_key
    assert key("lc_classifier_1.1.13") > key("hierarchical_rf_1.1.0")
    assert key("stamp_classifier_1.0.4") > key("stamp_classifier_1.0.0") and key("1.0.10") > key("1.0.9")
    rows = AlerceBroker.parse_objects({"items": [
        {"oid": "ZTF1", "meanra": 1.0, "meandec": 1.0, "lastmjd": 61300.0, "class": "SN", "probability": 0.4},
        {"oid": "ZTF1", "meanra": 1.0, "meandec": 1.0, "lastmjd": 61300.0, "class": "SN", "probability": 0.7},
    ]})[0]
    assert len(rows) == 1 and rows[0].probability == 0.7 and rows[0].extra["classifier_choice"] == "max_probability"


def stamp_probabilities(mock: respx.MockRouter, probability: float = 0.9) -> None:
    """Every object's /probabilities: stamp_classifier_1.0.4 ranks SN first."""
    mock.get(url__regex=r".*/objects/[^/]+/probabilities.*").mock(return_value=httpx.Response(200, json=[
        {"classifier_name": "stamp_classifier", "classifier_version": "stamp_classifier_1.0.4", "class_name": "SN",
         "probability": probability, "ranking": 1},
        {"classifier_name": "stamp_classifier", "classifier_version": "stamp_classifier_1.0.4", "class_name": "AGN",
         "probability": 0.05, "ranking": 2}]))


def _alerce_mock(mock: respx.MockRouter, detections: list[httpx.Response]) -> None:
    item = {"oid": "ZTF26aaaaaab", "meanra": 150.0, "meandec": 2.0, "firstmjd": 61305.0, "lastmjd": 61306.3,
            "class": "SN", "probability": 0.9, "classifier": "stamp_classifier"}
    mock.get(url__startswith="https://api.alerce.online/ztf/v1/objects/?").mock(
        return_value=httpx.Response(200, json={"items": [item]}))
    mock.get("https://api.alerce.online/ztf/v1/objects/ZTF26aaaaaab/detections").mock(side_effect=detections)
    stamp_probabilities(mock)


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
        stamp_probabilities(mock)
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


# ---------------------------------------------------------------------------
# Review round 2 (recorded: xmatch_review) -- Galactic-star verdicts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def review() -> dict[str, AlertEnrichment]:
    return asyncio.run(enrich_recorded("xmatch_review"))


@pytest.mark.parametrize(("name", "criterion"), [
    ("WD1647493723250083584", "parallax 9.186 +/- 0.092 mas (99.8 sigma"),  # WDJ153053.31+690231.98, DSC ext 0.985
    ("UV_Per", "proper motion 33.65 mas/yr (446 sigma)"),  # the dwarf nova, DSC ext 0.954
    ("M31_LF82d", "12.9 sigma"),  # [LF82] d on M31: a Gaia galaxy candidate, DSC ext 0.71
    ("M31GC_J004237", "proper motion 14.68 mas/yr (208 sigma)"),  # a Gaia galaxy candidate on M31
])
def test_decisive_astrometry_outweighs_gaia_extragalactic_classification(review: dict[str, AlertEnrichment], name: str,
                                                                        criterion: str) -> None:
    """Regression: Gaia DSC P(galaxy) + P(quasar) > 0.5 or galaxy-candidate membership discarded the 100-sigma
    parallax of a white dwarf (and the astrometry of 32% of SIMBAD's CVs): known_star False."""
    res = review[name]
    gaia = min((c for c in res.counterparts if c["catalog"] == "gaia_dr3"), key=lambda c: c["separation_arcsec"])
    # The regression's condition: Gaia classifies the source as extragalactic, yet it is a well-fitted point source.
    assert gaia["dsc_p_extragalactic"] > 0.5 or gaia["in_galaxy_candidates"] is True
    assert gaia["ruwe"] < 1.4 and gaia["astrometric_excess_noise_sig"] <= 2.0
    assert gaia["pm_over_error"] >= alerts.PM_SNR_DECISIVE or gaia["parallax_over_error"] >= alerts.PARALLAX_SNR_SECURE
    assert res.status == "done" and res.known_star is True
    assert res.host is None and res.host_status == "not_applicable_star"
    assert any(criterion in e and "-> Galactic star" in e for e in res.evidence), res.evidence
    assert any("decisive astrometry" in e for e in res.evidence)
    assert not any("looks like a galaxy nucleus" in e or "is not a star's" in e for e in res.evidence), res.evidence


@pytest.mark.parametrize(("name", "simbad_id", "m_g"), [
    ("M31_WWV2004_LPV", "[WWV2004] J0043124+404639", -13.3),  # LP*, G = 11.04
    ("M31_GSC02805-02180", "GSC 02805-02180", -11.4),
    ("M31_MLV92_187476", "[MLV92] 187476", -10.7),
    ("M31_GPM11.17+41.19", "GPM 11.174412+41.194938", -10.5),
])
def test_bright_catalogued_stars_on_m31_are_foreground_whatever_their_excess_noise(
        review: dict[str, AlertEnrichment], name: str, simbad_id: str, m_g: float) -> None:
    """Regression: the M_G < -10 test required a well-behaved point source, and bright stars on M31's disc have
    huge Gaia excess noise: 4 of 4 were 'extragalactic stars' in M31."""
    res = review[name]
    gaia = min((c for c in res.counterparts if c["catalog"] == "gaia_dr3"), key=lambda c: c["separation_arcsec"])
    assert gaia["astrometric_excess_noise_sig"] > 100  # not a well-behaved point source
    assert res.status == "done" and res.known_star is True and res.stellar_counterpart is True
    assert res.host is None and res.host_status == "not_applicable_star"
    assert any(f"= SIMBAD {simbad_id}" in e and f"M_G = {m_g}" in e and "-> Galactic foreground star" in e
               for e in res.evidence), res.evidence
    assert not any("an extragalactic star" in e for e in res.evidence)


def test_binary_near_ngc5078_is_galactic_and_ngc5078_has_its_current_size(review: dict[str, AlertEnrichment]) -> None:
    """Regression: HyperLEDA 2003 gave NGC 5078 logD25 2.71 (51'; RC3 1.60, current HyperLEDA 1.409), so the
    eclipsing binary Gaia DR3 6189441739218449664, 5.9' away, was 'an extragalactic star in NGC 5078'."""
    res = review["EB_6189441739218449664"]
    ngc5078 = next(g for g in res.d25_galaxies if g["pgc"] == 46490)
    assert ngc5078["log_d25_2003"] == pytest.approx(2.71) and ngc5078["log_d25"] == pytest.approx(1.409)
    assert ngc5078["semi_major_arcsec"] == pytest.approx(3.0 * 10 ** 1.409) and ngc5078["d_dlr"] > alerts.DLR_SEARCH_MAX
    assert res.known_star is True and res.host is None and res.host_status == "not_applicable_star"
    # RUWE 5.1 (a binary), yet an 11 mas/yr proper motion at 31 sigma is real.
    assert any("proper motion 11.12 mas/yr (31 sigma)" in e and "real although the single-star model fits it poorly" in e
               for e in res.evidence), res.evidence
    far = review["NGC5078_40arcmin"]
    assert far.host is None and all(g["d_dlr"] > alerts.DLR_SEARCH_MAX for g in far.d25_galaxies if g["pgc"] == 46490)
    assert alerts.HYPERLEDA_D25_CORRECTIONS[46490] == (1.409, 0.62) and alerts.HYPERLEDA_D25_CORRECTIONS[29220][0] == 0.992


@pytest.mark.parametrize("name", ["SN2002gn", "SN2018aks"])
def test_ned_point_source_without_gaia_does_not_make_a_supernova_galactic(review: dict[str, AlertEnrichment],
                                                                         name: str) -> None:
    """Regression: a NED '*' ("star or point source") entry at the SN inside a z >= 0.01 galaxy made it a
    'Galactic foreground star' and dropped its host; there is no Gaia source (it is the SN's own old SDSS
    detection, or a compact knot)."""
    res = review[name]
    assert any(c["catalog"] == "ned" and c["object_type"] == "*" for c in res.counterparts)
    assert not any(c["catalog"] == "gaia_dr3" for c in res.counterparts)
    assert res.known_star is None and res.stellar_counterpart is True
    assert res.host is not None and res.host_status == "found" and res.host["method"] == "d25_ellipse"
    assert res.host["redshift"] == pytest.approx(res.transient_redshift, abs=0.01)
    assert any("NED type '*' (star or point source)" in e and "no Gaia DR3 point source" in e for e in res.evidence)


async def test_ned_point_source_with_a_gaia_point_source_is_a_foreground_star() -> None:
    """Synthetic: the same NED '*' entry inside a z = 0.05 galaxy, with (or without) a Gaia source at its position."""
    point = {"source_id": "SDSS J100000.00+020000.0", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1,
             "data": {"prefphytype": "*"}}
    gaia = {"source_id": "9", "ra": ALERT.ra, "dec": ALERT.dec + 0.1 / 3600, "separation_arcsec": 0.1,
            "data": {"phot_g_mean_mag": 20.5, "ruwe": 1.0, "astrometric_excess_noise_sig": 0.0,
                     "classprob_dsc_combmod_star": 0.99, "classprob_dsc_combmod_galaxy": 0.005,
                     "classprob_dsc_combmod_quasar": 0.005}}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 10 / 3600, "dec": ALERT.dec, "separation_arcsec": 10.0,
              "data": {"otype": "G", "rvz_redshift": 0.05}}
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)

    def answer_with(rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources=rows if "gaia_dr3" in cats else {"simbad": [galaxy]})

    bare = await enricher_with(answer_with({"ned": [point]}), d25).enrich(ALERT)
    assert bare.status == "done" and bare.known_star is None and bare.host is not None
    with_gaia = await enricher_with(answer_with({"ned": [point], "gaia_dr3": [gaia]}), d25).enrich(ALERT)
    assert with_gaia.status == "done" and with_gaia.known_star is True and with_gaia.host is None
    # G = 20.5 at the galaxy's distance modulus (36.4) would be M_G = -15.9: the persistent point source is a star.
    assert any("= NED SDSS J100000.00+020000.0 (type *)" in e and "Galactic foreground star" in e for e in with_gaia.evidence)


# ---------------------------------------------------------------------------
# Review round 2 (recorded) -- host association
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "host_name", "foreground_pgc", "z"), [
    ("PTF11dws", "PS1-11acn HOST", 39600, 0.15),  # on M106, 0.8" from its z = 0.15 host
    ("PTF10hv", "WISEA J140356.54+542726.5", 50063, 0.0518),  # on M101
    ("SN2008hz", "WISEA J004319.36+421017.7", 2557, 0.072),  # on M31; a nearer galaxy of unknown z is 6.9" away
])
def test_background_supernovae_are_not_hosted_by_the_foreground_giant(review: dict[str, AlertEnrichment], name: str,
                                                                     host_name: str, foreground_pgc: int,
                                                                     z: float) -> None:
    """Regression: inside M106/M101/M31's D25 ellipse the giant won over a catalogued galaxy at the SN's own
    redshift 0.8-9" away: the host distance (and any luminosity) was wrong by a factor 40-400."""
    res = review[name]
    assert res.status == "done" and res.host_status == "found" and res.known_star is False
    host = res.host
    assert host is not None and host["name"] == host_name and host["method"] == "nearest"
    assert host["redshift"] == pytest.approx(z, abs=0.001) and host["distance_method"] == "hubble_flow"
    assert host["distance_mpc"] > 150 and host["projected_offset_kpc"] < 15
    # The giant is recorded as a foreground projection, not the host.
    giant = next(g for g in res.d25_galaxies if g["pgc"] == foreground_pgc)
    assert giant["d_dlr"] <= 1.0 and res.transient_redshift is not None
    assert any(f"PGC {foreground_pgc}" in e and "foreground/background projection" in e for e in res.evidence)


@pytest.mark.parametrize("name", ["iPTF15dql", "PTF11paw", "iPTF15dhn", "PTF12gix"])
def test_background_supernovae_near_m31_do_not_get_m31(review: dict[str, AlertEnrichment], name: str) -> None:
    """Regression: SNe at z = 0.11-0.19 up to 3.3 D25 radii from M31 got M31 (0.75 Mpc) as host."""
    res = review[name]
    assert res.status == "done" and res.transient_redshift > 0.1
    assert res.host is None or res.host["pgc"] != 2557
    assert not any("[dlr_outside_d25]" in e or ("PGC 2557" in e and "inside the D25" in e) for e in res.evidence)
    if name == "iPTF15dql":  # 3.26 D25 radii: beyond the association limit (Gupta's 4 second-moment radii ~ 2 D25)
        assert res.host_status == "unassociated"
        assert any("possible association, not adopted: PGC 2557" in e and "d_DLR = 3.26" in e for e in res.evidence)
    if name in {"PTF11paw", "iPTF15dhn"}:  # within 2 D25 radii, but at the SN's redshift M31 is a foreground galaxy
        assert res.host_status == "unassociated"
        assert any("PGC 2557" in e and "foreground/background projection" in e for e in res.evidence)


def test_sn2016bam_keeps_ngc2445_not_a_quasar(review: dict[str, AlertEnrichment]) -> None:
    """Regression: a z = 2.06 QSO 15.5" away blocked NGC 2445 (d_DLR 1.03) and became the host (1.6 Gpc, 120 kpc)."""
    res = review["SN2016bam"]
    host = res.host
    assert host is not None and host["name"] == "NGC 2445" and host["method"] == "dlr_outside_d25"
    assert host["d_dlr"] == pytest.approx(1.03, abs=0.01) and host["redshift"] == pytest.approx(0.01335, abs=1e-4)
    assert res.transient_redshift == pytest.approx(0.0135, abs=1e-4) and host["projected_offset_kpc"] < 15
    qso = next(g for g in res.host_candidates if g["object_type"] == "QSO")
    assert qso["redshift"] > 2 and qso["d_dlr_estimated"] > alerts.DLR_HOST_MAX
    # The Tully (2015) group entry (a pair, 16" away) is not a host candidate either.
    assert any("[T2015] nest 102796 (type PaG" in e and "not a host candidate" in e for e in res.evidence)


def test_ptf12lz_gets_no_distant_galaxy_or_quasar_as_host(review: dict[str, AlertEnrichment]) -> None:
    """Regression: M33 (d_DLR 3.04) was blocked by a z = 0.336 galaxy 48" away, which became the host at 216 kpc."""
    res = review["PTF12lz"]
    assert res.host is None and res.host_status == "unassociated" and res.host_search_complete is True
    assert any("WISEA J013717.15+300802.3 at 48.2\" not adopted" in e for e in res.evidence)
    assert any("WISEA J013713.14+300734.5 (type QSO" in e and "not a host candidate" in e for e in res.evidence)
    assert any("possible association, not adopted: PGC 5818" in e for e in res.evidence)


def test_random_blank_positions_rarely_get_a_host(review: dict[str, AlertEnrichment]) -> None:
    """Regression: 12 of 20 random positions at |b| > 30 deg got host_status 'found' (the nearest catalogued galaxy
    up to 52.8" away), one the Ursa Minor dwarf 33' away."""
    rand = {k: v for k, v in review.items() if k.startswith("rand")}
    assert len(rand) == 20 and all(r.status == "done" and r.host_search_complete for r in rand.values())
    found = {k: r for k, r in rand.items() if r.host is not None}
    assert 0 < len(found) <= 3
    for r in found.values():
        host = r.host
        assert host["method"] == "nearest" and host["separation_arcsec"] < 15.0
        assert host["p_chance"] <= alerts.P_CHANCE_MAX
    # Galaxies within 60" were rejected as likely chance alignments, not ignored.
    assert sum(r.host_status == "unassociated" for r in rand.values()) >= 5
    ursa_minor = next(g for g in rand["rand6"].d25_galaxies if g["pgc"] == 54074)
    assert ursa_minor["log_d25_2003"] == pytest.approx(2.54) and ursa_minor["d_dlr"] > 100
    assert rand["rand6"].host is None


# ---------------------------------------------------------------------------
# Host association rules (synthetic)
# ---------------------------------------------------------------------------


def _foreground_giant(z_giant: float = 0.0015) -> tuple[dict[str, Any], dict[str, Any]]:
    """A HyperLEDA giant (a = 600") whose centre is 200" north of ALERT, and its NED identity."""
    giant = _d25_galaxy(ALERT.ra, ALERT.dec + 200 / 3600, ALERT, 600.0)
    identity = {"source_id": "NGC 1", "ra": giant["ra"], "dec": giant["dec"], "separation_arcsec": 200.0,
                "data": {"prefphytype": "G", "z": z_giant}}
    return giant, identity


async def test_background_galaxy_on_a_giant_wins_without_the_transient_redshift() -> None:
    """Synthetic (a live alert: no catalogued redshift): 0.8" from a z = 0.15 galaxy on a nearby giant's disc
    (d_DLR 0.33). The alert lies within the background galaxy's light (~0.3 typical light radii): its host."""
    giant, identity = _foreground_giant()
    background = {"source_id": "SDSS J100000.00+020000.8", "ra": ALERT.ra, "dec": ALERT.dec + 0.8 / 3600,
                  "separation_arcsec": 0.8, "data": {"prefphytype": "G", "z": 0.15}}

    def answer_with(rows):
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats else {"ned": rows})

    res = await enricher_with(answer_with([background, identity]), ([giant], None, False)).enrich(ALERT)
    assert res.status == "done" and res.transient_redshift is None
    assert res.host is not None and res.host["name"] == background["source_id"] and res.host["method"] == "nearest"
    assert res.host["d_dlr_estimated"] == pytest.approx(0.8 / alerts.typical_light_radius_arcsec(0.15, ALERT.ra, ALERT.dec))
    assert any("PGC 1" in e and "different redshift" in e and "may be the host" in e for e in res.evidence)
    # 20" away the background galaxy is ~5 typical radii off: the giant keeps the alert.
    far = {**background, "dec": ALERT.dec + 20 / 3600, "separation_arcsec": 20.0}
    kept = await enricher_with(answer_with([far, identity]), ([giant], None, False)).enrich(ALERT)
    assert kept.host is not None and kept.host["pgc"] == 1 and kept.host["method"] == "d25_ellipse"


async def test_transient_redshift_from_the_broker_rejects_a_foreground_host() -> None:
    """Synthetic: Fink/LSST's TNS redshift (f:xm_tns_redshift) of the alert is 0.1: the z = 0.0015 giant it is
    projected on is not its host; without that redshift it is."""
    giant, identity = _foreground_giant()

    def answer(ra, dec, cats):
        return record_of(ra, dec, cats, sources={} if "gaia_dr3" in cats else {"ned": [identity]})

    lsst = Alert.from_dict({**ALERT.as_dict(), "broker": "fink_lsst", "extra": {"tns": "SN 2026abc", "tns_redshift": 0.1}})
    res = await enricher_with(answer, ([giant], None, False)).enrich(lsst)
    assert res.transient_redshift == 0.1 and "TNS" in res.transient_redshift_source
    assert res.host is None and res.host_status == "unassociated"
    assert any("foreground/background projection" in e for e in res.evidence)
    plain = await enricher_with(answer, ([giant], None, False)).enrich(ALERT)
    assert plain.host is not None and plain.host["pgc"] == 1


def test_same_redshift_and_host_types() -> None:
    assert alerts._same_redshift(0.0795, 0.072)  # SN 2008hz and its host: 2250 km/s < 3000 x 1.08
    assert not alerts._same_redshift(0.0518, 0.000811)  # PTF 10hv vs M101
    assert not alerts._is_host_type("simbad", "QSO") and not alerts._is_host_type("ned", "QSO")
    assert not alerts._is_host_type("simbad", "PaG") and not alerts._is_host_type("ned", "GPair")
    assert alerts._is_host_type("simbad", "Sy1") and alerts._is_host_type("ned", "G")
    # A group is plausible when one of its entries is a single galaxy (NED 'G' + SIMBAD 'QSO' of one object).
    assert alerts._is_plausible_host({"member_types": [("simbad", "QSO"), ("ned", "G")]})
    assert not alerts._is_plausible_host({"member_types": [("simbad", "QSO"), ("ned", "QSO")]})


# ---------------------------------------------------------------------------
# Review round 2 -- service: moved alerts, priority re-crossmatch, shared concurrency, watch memory
# ---------------------------------------------------------------------------


class GatedEnricher:
    """Synthetic enricher: records the positions it enriches; enrichments wait for ``gate`` when ``slow``."""

    match_radius_arcsec, host_radius_arcsec, catalogs = 2.0, 60.0, ["gaia_dr3"]

    def __init__(self, *, slow: bool = True, delay: float = 0.0) -> None:
        self.positions: list[tuple[str, float]] = []
        self.gate = asyncio.Event()
        self.slow = slow
        self.delay = delay
        self.active = 0
        self.peak = 0

    async def enrich(self, alert: Alert) -> AlertEnrichment:
        self.positions.append((alert.alert_id, alert.dec))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.slow and len(self.positions) == 1:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"],
                               is_new=True, ra=alert.ra, dec=alert.dec, evidence=[f"computed at dec={alert.dec:.6f}"])


async def test_moved_alert_is_crossmatched_again_after_a_stale_in_flight_enrichment(
        store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: poll 2 moved the alert by 3" while poll 1's background enrichment (of the old position) was
    running; that enrichment then stored 'done' and the new position was never crossmatched."""
    old = Alert("fink", "ZTF26race", 150.0, 2.0, 61306.30, 19.0, "r", "SN candidate", 0.9, "")
    new = Alert("fink", "ZTF26race", 150.0, 2.0 + 3.0 / 3600, 61306.35, 18.9, "r", "SN candidate", 0.9, "")
    answers = [FetchResult([old]), FetchResult([new]), FetchResult([new]), FetchResult([new])]

    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return answers.pop(0)

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    enricher = GatedEnricher()
    svc = AlertService(store, offline_client(), enricher, clock=lambda: 61307.0)  # type: ignore[arg-type]
    _, due1 = await svc.ingest("fink", since_mjd=61306.0, limit=5)
    background = asyncio.create_task(svc.crossmatch_in_background(due1))
    await asyncio.sleep(0.05)
    moved, due2 = await svc.ingest("fink", since_mjd=61306.0, limit=5)  # the alert moved while being enriched
    assert moved.updated == 1 and store.crossmatch_status(old.alert_id) == "pending"
    assert (await svc.crossmatch_alerts(due2))["skipped_in_flight"] == 1
    enricher.gate.set()
    await background
    row = store.get(old.alert_id)
    # The old position's enrichment is kept as data but the row stays pending, attempts untouched.
    assert row["crossmatch_status"] == "pending" and row["crossmatch_attempts"] == 0
    assert row["enrichment"]["dec"] == pytest.approx(2.0) and "previous position" in row["last_crossmatch_error"]
    _, due3 = await svc.ingest("fink", since_mjd=61306.0, limit=5)
    assert [a.dec for a in due3] == [pytest.approx(new.dec)]
    assert (await svc.crossmatch_alerts(due3))["done"] == 1
    row = store.get(old.alert_id)
    assert row["crossmatch_status"] == "done" and row["enrichment"]["dec"] == pytest.approx(new.dec)
    assert [d for _, d in enricher.positions] == [pytest.approx(2.0), pytest.approx(new.dec)]
    _, due4 = await svc.ingest("fink", since_mjd=61306.0, limit=5)
    assert due4 == []


async def test_recrossmatch_of_a_moved_alert_waits_for_the_old_enrichment_and_runs_again(store: AlertStore) -> None:
    """Synthetic: POST /{id}/crossmatch (enrich_alert) for the new position while the old one is enriched."""
    old = Alert("fink", "ZTF26move", 150.0, 2.0, 61306.30, 19.0, "r", "SN candidate", 0.9, "")
    store.upsert(old)
    enricher = GatedEnricher()
    svc = AlertService(store, offline_client(), enricher)  # type: ignore[arg-type]
    background = asyncio.create_task(svc.crossmatch_in_background([old]))
    await asyncio.sleep(0.02)
    new = Alert.from_dict({**old.as_dict(), "mjd": 61306.4, "dec": 2.0 + 3.0 / 3600})
    store.upsert(new)
    request = asyncio.create_task(svc.enrich_alert(new))
    await asyncio.sleep(0.02)
    enricher.gate.set()
    enrichment, stored = await asyncio.wait_for(request, timeout=10)
    await background
    assert stored == "stored" and enrichment.dec == pytest.approx(new.dec)
    assert store.get(old.alert_id)["crossmatch_status"] == "done"
    assert [d for _, d in enricher.positions] == [pytest.approx(2.0), pytest.approx(new.dec)]


def test_router_recrossmatch_skips_the_queue_of_a_background_batch(store: AlertStore) -> None:
    """Regression: POST /{id}/crossmatch of an alert queued last in a background batch waited for the whole
    batch (60 alerts, concurrency 3: ~20 enrichments; live, a 500-alert batch made the request hang ~33 min)."""
    batch = [Alert("alerce", f"ZTF26b{i:06d}", 150.0 + i * 0.01, 2.0, 61306.3, 19.0, "r", "SN", 0.9, "") for i in range(60)]
    store.upsert_many(batch)
    enricher = GatedEnricher(slow=False, delay=0.1)
    svc = AlertService(store, offline_client(), enricher)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(router)
    app.state.alert_service = svc

    async def scenario() -> tuple[int, int, int]:
        background = asyncio.create_task(svc.crossmatch_in_background(batch))
        await asyncio.sleep(0.02)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            answer = await client.post(f"/api/v1/alerts/{batch[-1].alert_id}/crossmatch")
        done_before = sum(1 for r in store.list(limit=100) if r["crossmatch_status"] == "done")
        await background
        return answer.status_code, done_before, enricher.peak

    status, done_before, peak = asyncio.run(scenario())
    assert status == 200
    # The request ran at once: when it answered, only the first few batch enrichments had finished.
    assert done_before <= 10
    assert peak <= svc.concurrency + 1  # the batch's 3 slots plus the request that skipped the queue
    assert all(r["crossmatch_attempts"] == 1 for r in store.list(limit=100))


async def test_overlapping_batches_share_the_service_concurrency(store: AlertStore) -> None:
    """Regression: every crossmatch_alerts call created its own semaphore: 4 overlapping background batches ran
    12 enrichments at once with concurrency 3."""
    batches = [[Alert("alerce", f"ZTF26c{k}{i:05d}", 10.0 + i * 0.01, 2.0, 61306.3, 19.0, "r", "SN", 0.9, "")
                for i in range(9)] for k in range(4)]
    for b in batches:
        store.upsert_many(b)
    enricher = GatedEnricher(slow=False, delay=0.02)
    svc = AlertService(store, offline_client(), enricher, concurrency=3)  # type: ignore[arg-type]
    await asyncio.gather(*(svc.crossmatch_in_background(b) for b in batches))
    assert enricher.peak == 3 and len(enricher.positions) == 36
    assert all(r["crossmatch_status"] == "done" for r in store.list(limit=100))


async def test_watch_without_iterations_keeps_a_bounded_history(store: AlertStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: `alerts watch` (iterations None) kept every PollResult for the life of the daemon."""
    async def fake_fetch(client, broker, *, since_mjd, until_mjd, limit=20, options=None):
        return FetchResult([Alert("fink", "ZTF26w", 1.0, 1.0, 61306.3, 19.0, "r", "SN candidate", 0.9, "")])

    monkeypatch.setattr(alerts, "fetch_alerts", fake_fetch)
    svc = AlertService(store, offline_client(), None, clock=lambda: 61307.0)
    stop = asyncio.Event()
    seen: list[alerts.PollResult] = []

    def on_result(res: alerts.PollResult) -> None:
        seen.append(res)
        if len(seen) >= 3 * alerts.WATCH_RESULTS_KEPT:
            stop.set()

    kept = await svc.watch(["fink"], interval_seconds=0.001, crossmatch=False, stop_event=stop, on_result=on_result)
    assert len(seen) == 3 * alerts.WATCH_RESULTS_KEPT and len(kept) == alerts.WATCH_RESULTS_KEPT
    assert kept == seen[-alerts.WATCH_RESULTS_KEPT:]
    # With a number of iterations every result is returned.
    assert len(await svc.watch(["fink"], interval_seconds=0.001, crossmatch=False, iterations=15)) == 15


# ---------------------------------------------------------------------------
# Review round 2 -- CLI: MJD range, non-finite passthrough values, `alerts crossmatch`
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    alerts.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.mark.parametrize("argv", [
    ["alerts", "poll", "--broker", "fink", "--no-crossmatch", "--since-mjd", "3000000", "--until-mjd", "3000001"],
    ["alerts", "poll", "--broker", "fink_lsst", "--no-crossmatch", "--since-mjd", "-700000"],
    ["alerts", "poll", "--until-mjd", "nan"],
    ["alerts", "list", "--since-mjd", "nan"],
    ["alerts", "list", "--since-mjd", "39999"],
])
def test_cli_rejects_out_of_range_mjds_with_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Regression: an MJD outside the datetime range raised an uncaught OverflowError (a traceback)."""
    with pytest.raises(SystemExit) as exc:
        _cli(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "an MJD must lie between 40000 and 100000" in err and "Traceback" not in err


def test_non_finite_passthrough_values_are_stored_as_null(store: AlertStore, tmp_path: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    """Regression: an upstream NaN in a raw passthrough field (ALeRCE isdiffpos, Fink i:ndethist) was stored as a
    NaN token, and `alerts list/show --format json` then raised ValueError (the API wrote null)."""
    db = f"sqlite:///{(tmp_path / 'nan.sqlite3').as_posix()}"
    nan_store = AlertStore(MetadataStore(db))
    app = FastAPI()
    app.include_router(router)
    app.state.alert_store = nan_store
    api = TestClient(app)
    items = [{"oid": "ZTF26aaaaaaf", "meanra": 150.0, "meandec": 2.0, "firstmjd": 61306.1, "lastmjd": 61306.3,
              "class": "SN", "probability": 0.9, "ndet": float("nan"), "stellar": float("nan")}]
    fink_rows = [{"i:objectId": "ZTF26aaaaaad", "i:ra": 10.0, "i:dec": 5.0, "i:jd": 2461306.8, "i:magpsf": 19.0,
                  "i:fid": 1, "i:isdiffpos": "t", "d:snn_sn_vs_all": 0.9, "i:ndethist": float("nan"),
                  "d:gaiaVarFlag": float("nan"), "d:roid": float("inf")}]
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get(url__startswith="https://api.alerce.online/ztf/v1/objects/?").mock(return_value=httpx.Response(
            200, text=json.dumps({"items": items}), headers={"content-type": "application/json"}))
        mock.get("https://api.alerce.online/ztf/v1/objects/ZTF26aaaaaaf/detections").mock(return_value=httpx.Response(
            200, text='[{"mjd": 61306.3, "magpsf": 19.2, "fid": 1, "isdiffpos": NaN, "candid": 7}]',
            headers={"content-type": "application/json"}))
        stamp_probabilities(mock)
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(
            200, text=json.dumps(fink_rows), headers={"content-type": "application/json"}))
        for broker in ("alerce", "fink"):
            answer = api.post("/api/v1/alerts/poll", json={"broker": broker, "since_mjd": 61306.0, "until_mjd": 61307.0,
                                                           "crossmatch": False})
            assert answer.status_code == 200, answer.text
    alerce_row, fink_row = nan_store.get("alerce:ZTF26aaaaaaf"), nan_store.get("fink:ZTF26aaaaaad")
    assert alerce_row["extra"]["isdiffpos"] is None and alerce_row["extra"]["ndet"] is None
    assert alerce_row["extra"]["stellar"] is None and alerce_row["is_negative"] is None
    assert (fink_row["extra"]["ndethist"], fink_row["extra"]["gaia_var_flag"], fink_row["extra"]["roid"]) == (None, None, None)
    for argv in (["alerts", "list", "--db", db, "--format", "json"], ["alerts", "show", "fink:ZTF26aaaaaad", "--db", db]):
        args = _cli(argv)
        assert args.handler(args) == 0
        out = capsys.readouterr().out
        assert "NaN" not in out and "Infinity" not in out
        json.loads(out)
    # A row stored with a NaN token before this fix is still printed as strict JSON.
    with nan_store._conn() as conn:
        conn.execute("UPDATE alerts SET extra_json = ? WHERE id = ?", ('{"ndethist": NaN}', "fink:ZTF26aaaaaad"))
    args = _cli(["alerts", "show", "fink:ZTF26aaaaaad", "--db", db])
    assert args.handler(args) == 0 and json.loads(capsys.readouterr().out)["extra"] == {"ndethist": None}


def test_cli_crossmatch_reruns_one_alert_or_the_incomplete_ones(store: AlertStore, tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """`alerts crossmatch <id>` / `--incomplete`: run now, whatever the retry schedule or attempt cap."""
    db = f"sqlite:///{(tmp_path / 'x.sqlite3').as_posix()}"
    cli_store = AlertStore(MetadataStore(db))
    capped = Alert.from_dict({**ALERT.as_dict(), "object_id": "ZTF26capped"})
    other = Alert.from_dict({**ALERT.as_dict(), "object_id": "ZTF26other", "ra": 151.0})
    cli_store.upsert_many([capped, other])
    broken = AlertEnrichment(status="partial", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"],
                             failures=[{"catalog": "ned", "error_type": "CatalogQueryError"}], ra=capped.ra, dec=capped.dec)
    for _ in range(alerts.MAX_CROSSMATCH_ATTEMPTS):
        cli_store.set_enrichment(capped.alert_id, broken, now_mjd=61306.5)
    assert cli_store.incomplete(now_mjd=61306.6) == [Alert.from_dict(cli_store.get(other.alert_id))]
    built: list[AlertService] = []

    def fake_build(client, *, store, crossmatch=True, match_radius_arcsec=2.0, host_radius_arcsec=60.0, **kwargs):
        svc = AlertService(store, client, GatedEnricher(slow=False))  # type: ignore[arg-type]
        built.append(svc)
        return svc

    monkeypatch.setattr(alerts, "build_alert_service", fake_build)
    args = _cli(["alerts", "crossmatch", capped.alert_id, "--db", db])
    assert args.handler(args) == 0
    assert "crossmatched 1 alert(s): done 1" in capsys.readouterr().out
    assert cli_store.get(capped.alert_id)["crossmatch_status"] == "done"
    args = _cli(["alerts", "crossmatch", "--incomplete", "--db", db, "--format", "json"])
    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["crossmatched"] == 1 and [r["id"] for r in payload["alerts"]] == [other.alert_id]
    assert payload["counts"]["done"] == 1
    for argv, code in ((["alerts", "crossmatch", "--db", db], 2), (["alerts", "crossmatch", "fink:nope", "--db", db], 1)):
        args = _cli(argv)
        assert args.handler(args) == code
    assert "give an alert id or --incomplete" in capsys.readouterr().err


def test_transient_on_a_slowly_moving_gaia_star_far_from_galaxies_is_a_known_star() -> None:
    """Regression (recorded, live POST /alerts/poll): fink:ZTF26abuxdqd at l = 48.0, b = -10.3 lies 0.35" from Gaia DR3
    4298320600309152000 (G = 18.6, DSC P(star) = 0.999996, RUWE 0.97, no excess noise) moving 2.70 mas/yr at 18 sigma.
    No SIMBAD/NED entry has a stellar type. The LMC-distance limit (3.2 mas/yr) was applied far from the Magellanic
    Clouds, so the alert came back known_star=False with no evidence at all. Far from every galaxy that could hold
    stars, a significant motion of a well-behaved point source is a Galactic star's."""
    res = asyncio.run(enrich_recorded("xmatch_final"))["ZTF26abuxdqd"]
    assert res.status == "done" and res.catalog_status == {"gaia_dr3": "success", "simbad": "empty", "ned": "success"}
    star = min(gaia_rows(res), key=lambda c: c["separation_arcsec"])
    assert star["source_id"] == "4298320600309152000" and star["separation_arcsec"] < 0.5
    assert star["dsc_p_star"] > 0.999 and star["ruwe"] < 1.0 and star["astrometric_excess_noise_sig"] == 0.0
    assert star["pm_masyr"] == pytest.approx(2.70, abs=0.01) and star["pm_over_error"] == pytest.approx(18.2, abs=0.1)
    assert star["pm_masyr"] < alerts.PM_MAX_UNKNOWN_DISTANCE  # below the LMC-distance limit: that limit must not apply
    assert alerts.near_star_forming_galaxy(299.0526, 8.4242) is None and alerts.local_group_dwarf_at(299.0526, 8.4242) is None
    assert res.known_star is True and res.host is None and res.host_status == "not_applicable_star"
    assert res.evidence, "a known_star answer must carry its evidence"
    assert any("Gaia DR3 4298320600309152000: proper motion 2.70 mas/yr (18 sigma)" in e and "-> Galactic star" in e
               for e in res.evidence), res.evidence
    # No catalogue gives it a stellar type (NED lists only the WISE source): stellar_counterpart reports exactly that.
    assert res.stellar_counterpart is False
