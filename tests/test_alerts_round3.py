"""Offline regression tests for alerts.py (review round 3).

Recorded fixture ``tests/fixtures/alerts/xmatch_round3`` (``record_alerts.py round3``): Gaia DR3 / SIMBAD / NED /
HyperLEDA / Cosmicflows-4 answers for blazars and QSOs (3C 273, BL Lac, PKS 2155-304, Mrk 421, 3C 279,
PG 1553+113), galaxy nuclei with a SIMBAD '*' duplicate entry (LEDA 1798300, 2MASS J09480288+1319111,
Pul -3 1020035, 2MASS J14075302+5336515) and a real foreground star on a galaxy (MGC 97030), blank points
near a z = 0.005 dwarf, and the Sculptor RR Lyrae star EV* SclG V0214. The other tests use small
synthetic inputs and say so.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import make_service, offline_client
from test_alerts import ALERT, Replay, _d25_galaxy, enrich_recorded, enricher_with, record_of, recorded_alerts

import alerts
from alerts import (
    Alert,
    AlertEnricher,
    AlertEnrichment,
    AlertService,
    AlertStore,
    BrokerError,
    FinkZTFBroker,
    PollRequest,
    fetch_alerts,
    normalize_options,
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
def round3() -> dict[str, AlertEnrichment]:
    return asyncio.run(enrich_recorded("xmatch_round3"))


def candidate(res: AlertEnrichment, name: str) -> dict[str, Any]:
    return next(g for g in res.host_candidates if g["source_id"] == name)


async def recorded_host_cone(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The recorded host cone (every galaxy group, annotated by the host choice) and D25 galaxies of a position."""
    alert = next(a for a in recorded_alerts("xmatch_round3") if a.object_id == name)
    with Replay("xmatch_round3"):
        async with offline_client() as client:
            enricher = AlertEnricher(make_service(client))
            record = await enricher._crossmatch(enricher.host_service, alert.ra, alert.dec, enricher.host_catalogs,
                                                enricher.host_radius_arcsec)
            d25, failure, _ = await enricher._d25(alert)
            assert failure is None and not record.failures
            groups = alerts.group_host_candidates(alerts._galaxies(record))
            result = AlertEnrichment(status="done", match_radius_arcsec=enricher.match_radius_arcsec,
                                     host_radius_arcsec=enricher.host_radius_arcsec, catalogs=list(enricher.catalogs))
            await enricher._choose_host(alert, result, groups, d25)
    return groups, d25


# ---------------------------------------------------------------------------
# AGN / QSO / blazar alerts: the host is the active galaxy itself (recorded)
# ---------------------------------------------------------------------------

# name: (host name, catalogued redshift, a neighbouring galaxy that must not be the host)
AGN_NUCLEI: dict[str, tuple[str, float, str | None]] = {
    "3C273": ("3C 273", 0.15757, "[RGG2003] new galaxy"),  # the z = 0.0053 dwarf 10.8" away was the host
    "BL_Lac": ("NAME BL Lac", 0.06550, "2MASX J22024434+4216304"),  # 15.4" away, P_cc 0.12 (was the host)
    "PKS2155-304": ("PKS J2158-3013", 0.116, "[FGM91] G1"),  # a companion 4.4" away at the same redshift
    "3C279": ("3C 279", 0.53542, None),  # was 'unassociated'
    "PG1553+113": ("PG 1553+113", 0.36, None),  # was 'unassociated'
}


@pytest.mark.parametrize("name", sorted(AGN_NUCLEI))
def test_agn_alerts_are_hosted_by_their_own_active_galaxy(round3: dict[str, AlertEnrichment], name: str) -> None:
    """Regression: the non-host exclusion of quasars/blazars also removed the AGN at the alert position, so a
    blazar got an unrelated neighbour at another redshift as host (3C 273 -> a z = 0.0053 dwarf at 25.8 Mpc),
    or none."""
    host_name, z, neighbour = AGN_NUCLEI[name]
    res = round3[name]
    host = res.host
    assert res.status == "done" and res.host_status == "found" and res.known_agn is True and res.known_star is False
    assert host is not None and host["name"] == host_name and host["method"] == "agn_nucleus"
    assert host["separation_arcsec"] < 0.5 and host["object_type"] in alerts.SIMBAD_AGN_TYPES
    assert host["redshift"] == pytest.approx(z, abs=5e-4) and host["distance_method"] == "hubble_flow"
    # Its redshift is the transient's: galaxies at another redshift cannot be adopted.
    assert res.transient_redshift == host["redshift"] and "AGN/QSO at the alert position" in res.transient_redshift_source
    assert any("its own active nucleus" in e and "[agn_nucleus]" not in e for e in res.evidence)
    if neighbour is not None:
        other = candidate(res, neighbour)
        assert other["source_id"] != host["name"]
        if name == "3C273":
            assert other["redshift_consistent"] is False and other["redshift"] == pytest.approx(0.0053)


def test_blazar_with_its_own_d25_galaxy_keeps_the_d25_host(round3: dict[str, AlertEnrichment]) -> None:
    """Mrk 421 is PGC 33452: the D25 rule adopts it (its SIMBAD BLL entry is the D25 galaxy's identity), at the
    AGN's own redshift -- the same galaxy as the 'agn_nucleus' rule, sized."""
    res = round3["Mrk421"]
    host = res.host
    assert host is not None and host["method"] == "d25_ellipse" and host["pgc"] == 33452
    assert host["object_type"] == "BLL" and host["separation_arcsec"] < 0.5
    assert res.transient_redshift == host["redshift"] == pytest.approx(0.030)


def test_agn_nucleus_is_only_the_agn_at_the_alert_position() -> None:
    """Synthetic: a QSO (z = 0.2) at the alert and a galaxy 4" away (z = 0.02, P_cc < 0.1). At the QSO the host
    is the QSO; with the QSO 15" away (SN 2016bam's case) it is not a host and the galaxy is."""
    galaxy = {"source_id": "SDSS J100000.27+020000.0", "ra": ALERT.ra + 4 / 3600, "dec": ALERT.dec,
              "separation_arcsec": 4.0, "data": {"otype": "G", "rvz_redshift": 0.02}}

    def qso_at(sep: float) -> dict[str, Any]:
        return {"source_id": "QSO J1", "ra": ALERT.ra, "dec": ALERT.dec + sep / 3600, "separation_arcsec": sep,
                "data": {"otype": "QSO", "rvz_redshift": 0.2}}

    def answer_with(qso: dict[str, Any]):
        near = [qso] if qso["separation_arcsec"] <= 2.0 else []
        return lambda ra, dec, cats: record_of(ra, dec, cats, sources={"simbad": near} if "gaia_dr3" in cats
                                               else {"simbad": [qso, galaxy]})

    on_qso = asyncio.run(enricher_with(answer_with(qso_at(0.2))).enrich(ALERT))
    assert on_qso.host is not None and on_qso.host["name"] == "QSO J1" and on_qso.host["method"] == "agn_nucleus"
    assert on_qso.transient_redshift == 0.2 and on_qso.known_agn is True
    assert any("SDSS J100000.27+020000.0" in g["source_id"] and g["redshift_consistent"] is False
               for g in on_qso.host_candidates)
    off_qso = asyncio.run(enricher_with(answer_with(qso_at(15.0))).enrich(ALERT))
    assert off_qso.host is not None and off_qso.host["name"] == galaxy["source_id"] and off_qso.host["method"] == "nearest"
    assert off_qso.transient_redshift is None and off_qso.known_agn is False
    assert any(g["source_id"] == "QSO J1" for g in off_qso.host_candidates)


# ---------------------------------------------------------------------------
# SIMBAD / NED '*' entries of galaxy nuclei (recorded + synthetic)
# ---------------------------------------------------------------------------

# name: (the '*' entry, the host's redshift)
STAR_DUPLICATES: dict[str, tuple[str, float]] = {
    "LEDA1798300": ("LEDA 1798300", 0.02748),
    "2MASS_J09480288+1319111": ("2MASS J09480288+1319111", 0.08094),
    "Pul-3_1020035": ("Pul -3 1020035", 0.03506),
    "2MASS_J14075302+5336515": ("2MASS J14075302+5336515", 0.07812),
}


@pytest.mark.parametrize("name", sorted(STAR_DUPLICATES))
def test_simbad_star_entry_of_a_galaxy_nucleus_is_not_a_galactic_star(round3: dict[str, AlertEnrichment],
                                                                      name: str) -> None:
    """Regression: a SIMBAD '*' entry on a z = 0.03-0.08 galaxy nucleus made a nuclear transient a 'Galactic
    foreground star' (host_status 'not_applicable_star'); the Gaia source there is extragalactic and extended."""
    star_id, z = STAR_DUPLICATES[name]
    res = round3[name]
    star = next(c for c in res.counterparts if c["source_id"] == star_id)
    assert star["catalog"] == "simbad" and star["object_type"] == "*"
    gaia = min((c for c in res.counterparts if c["catalog"] == "gaia_dr3"), key=lambda c: c["separation_arcsec"])
    assert ((gaia["dsc_p_extragalactic"] or 0) > 0.5 or gaia["in_galaxy_candidates"]) and \
        gaia["astrometric_excess_noise_sig"] > 20
    assert res.status == "done" and res.known_star is False and res.stellar_counterpart is False
    assert res.host is not None and res.host_status == "found" and res.host["redshift"] == pytest.approx(z, abs=1e-4)
    assert res.host["separation_arcsec"] < 0.2
    assert any(f"SIMBAD {star_id} (type *)" in e and "another catalogue entry of the galaxy's nucleus" in e
               for e in res.evidence)
    assert not any("Galactic foreground star" in e or "-> Galactic" in e for e in res.evidence)


def test_real_foreground_star_on_a_galaxy_entry_stays_galactic(round3: dict[str, AlertEnrichment]) -> None:
    """MGC 97030 ('*') lies 0.12" from a galaxy entry, but a well-behaved Gaia point source moving 9.6 mas/yr
    (34 sigma) confirms the star: it stays a Galactic star."""
    res = round3["MGC97030"]
    assert res.known_star is True and res.stellar_counterpart is True and res.host_status == "not_applicable_star"
    assert not any("another catalogue entry of the galaxy's nucleus" in e for e in res.evidence)
    assert any("proper motion 9.59 mas/yr (34 sigma)" in e and "-> Galactic star" in e for e in res.evidence)


def _nucleus_with_star(gaia_data: dict[str, Any] | None, star_type: str = "*", catalog: str = "simbad"
                       ) -> AlertEnrichment:
    """Synthetic: a generic stellar entry 0.1" from a SIMBAD galaxy entry (z = 0.05, inside its D25 ellipse), with
    an optional Gaia DR3 source at the stellar entry's position."""
    field = "otype" if catalog == "simbad" else "prefphytype"
    star = {"source_id": "[X] 5", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.0, "data": {field: star_type}}
    galaxy = {"source_id": "SDSS J100000.00+020000.1", "ra": ALERT.ra, "dec": ALERT.dec + 0.1 / 3600,
              "separation_arcsec": 0.1, "data": {"otype": "G", "rvz_redshift": 0.05}}
    rows: dict[str, list[dict[str, Any]]] = {catalog: [star], "simbad": [*([star] if catalog == "simbad" else []), galaxy]}
    if gaia_data is not None:
        rows["gaia_dr3"] = [{"source_id": "11", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.0,
                             "data": gaia_data}]
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 10.0)], None, False)
    answer = (lambda ra, dec, cats: record_of(ra, dec, cats, sources=rows if "gaia_dr3" in cats
                                              else {"simbad": [galaxy]}))
    return asyncio.run(enricher_with(answer, d25).enrich(ALERT))


GAIA_STAR = {"phot_g_mean_mag": 20.0, "ruwe": 1.0, "astrometric_excess_noise_sig": 0.5, "parallax": 0.1,
             "parallax_error": 0.2, "classprob_dsc_combmod_star": 0.99, "classprob_dsc_combmod_galaxy": 0.005,
             "classprob_dsc_combmod_quasar": 0.005}
GAIA_NUCLEUS = {**GAIA_STAR, "astrometric_excess_noise_sig": 40.0, "classprob_dsc_combmod_star": 0.01,
                "classprob_dsc_combmod_galaxy": 0.98}


@pytest.mark.parametrize(("gaia", "expected", "duplicate"), [
    (None, False, True),  # no Gaia source confirms a star: the nucleus' own entry
    (GAIA_NUCLEUS, False, True),  # the Gaia source is the extended, extragalactic nucleus
    ({**GAIA_STAR, "astrometric_excess_noise_sig": 8.0}, False, True),  # not a well-behaved point source
    (GAIA_STAR, True, False),  # a well-behaved stellar point source: a foreground star blended with the galaxy
])
def test_generic_star_entry_on_a_galaxy_needs_a_gaia_point_source(gaia: dict[str, Any] | None, expected: bool,
                                                                  duplicate: bool) -> None:
    res = _nucleus_with_star(gaia)
    assert res.status == "done" and res.known_star is expected and res.stellar_counterpart is not duplicate
    assert any("another catalogue entry of the galaxy's nucleus" in e for e in res.evidence) is duplicate
    assert (res.host is not None) is not expected
    if expected:
        assert any("too distant for individually catalogued stars" in e for e in res.evidence)


def test_curated_stellar_types_and_ned_galactic_stars_are_never_nucleus_duplicates() -> None:
    """Synthetic: an RR Lyrae (SIMBAD RR*) or a NED '!*' (Milky Way star) entry on a galaxy entry stays stellar."""
    rr = _nucleus_with_star(None, "RR*")
    assert rr.stellar_counterpart is True and not any("nucleus" in e and "not taken as a star" in e for e in rr.evidence)
    milky_way = _nucleus_with_star(None, "!*", catalog="ned")
    assert milky_way.known_star is True and any("NED marks it as a Milky Way object" in e for e in milky_way.evidence)
    ned_point = _nucleus_with_star(None, "*", catalog="ned")
    assert ned_point.known_star is False and ned_point.stellar_counterpart is False


def test_ned_point_source_on_an_extragalactic_gaia_source_is_not_confirmed() -> None:
    """Synthetic (mutation M24): a NED '*' inside a z = 0.05 galaxy on a Gaia source that Gaia's DSC calls
    extragalactic (a compact knot of the host) is not confirmed as a star (None); on a stellar one it is (True)."""
    point = {"source_id": "SDSS J100000.00+020000.0", "ra": ALERT.ra, "dec": ALERT.dec, "separation_arcsec": 0.1,
             "data": {"prefphytype": "*"}}
    galaxy = {"source_id": "NGC 1", "ra": ALERT.ra + 10 / 3600, "dec": ALERT.dec, "separation_arcsec": 10.0,
              "data": {"otype": "G", "rvz_redshift": 0.05}}
    d25 = ([_d25_galaxy(galaxy["ra"], galaxy["dec"], ALERT, 30.0)], None, False)

    def run(gaia_data: dict[str, Any]) -> AlertEnrichment:
        gaia = {"source_id": "9", "ra": ALERT.ra, "dec": ALERT.dec + 0.1 / 3600, "separation_arcsec": 0.1,
                "data": gaia_data}
        rows = {"ned": [point], "gaia_dr3": [gaia]}
        return asyncio.run(enricher_with(lambda ra, dec, cats: record_of(
            ra, dec, cats, sources=rows if "gaia_dr3" in cats else {"simbad": [galaxy]}), d25).enrich(ALERT))

    knot = run({**GAIA_STAR, "classprob_dsc_combmod_star": 0.3, "classprob_dsc_combmod_galaxy": 0.7})
    assert knot.known_star is None and knot.host is not None
    assert any("no Gaia DR3 point source" in e for e in knot.evidence)
    # G = 20 at the host's distance would be M_G = -16.7, but the NED '*' is not confirmed as a star there.
    assert any("its only stellar-type entry is NED type '*'" in e and "luminosity not used" in e for e in knot.evidence)
    assert run({**GAIA_STAR, "in_galaxy_candidates": True}).known_star is None
    assert run(dict(GAIA_STAR)).known_star is True


# ---------------------------------------------------------------------------
# Typical light radius is no evidence of association (recorded + synthetic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["blank_near_dwarf", "blank_near_dwarf2"])
def test_blank_point_near_a_low_z_dwarf_gets_no_host(round3: dict[str, AlertEnrichment], name: str) -> None:
    """Regression: 45-50" from a z = 0.0053 dwarf (R25 ~1-2 kpc) the fixed 8 kpc 'typical light radius' made it
    the host with P_cc = 0.9997."""
    res = round3[name]
    assert res.status == "done" and res.host is None and res.host_status == "unassociated" and res.is_new is True
    assert res.host_search_complete is True
    (why,) = [e for e in res.evidence if "simbad:[RGG2003] new galaxy" in e]
    assert "not adopted: chance-coincidence probability 1.00 > 0.1" in why
    assert "within a typical 8 kpc light radius" in why and "does not make it the host" in why
    groups, _ = asyncio.run(recorded_host_cone(name))
    dwarf = next(g for g in groups if g["source_id"] == "[RGG2003] new galaxy")
    assert dwarf["redshift"] == pytest.approx(0.0053) and dwarf["d_dlr_estimated"] < 1.0 and dwarf["p_chance"] > 0.99


def test_blank_point_does_not_give_a_cone_galaxy_another_galaxys_pgc(round3: dict[str, AlertEnrichment]) -> None:
    """Regression: the dwarf 9.2" from PGC 41121 (= 3C 273, 60.4" from the alert, outside the host cone) was
    tagged PGC 41121, and 3C 273's Cosmicflows-4 / group identity was looked up for it."""
    res = round3["blank_near_dwarf"]
    three_c_273 = round3["3C273"].host  # at 3C 273 itself (0" from PGC 41121) the tag is right
    assert three_c_273["pgc"] == 41121
    groups, d25 = asyncio.run(recorded_host_cone("blank_near_dwarf"))
    # HyperLEDA's centre of PGC 41121 is 59.0" away, but its catalogue entry (3C 273) 60.4": outside the cone.
    pgc41121 = next(g for g in d25 if g["pgc"] == 41121)
    assert pgc41121["d_dlr"] is None and "3C273" in pgc41121["hyperleda_names"]
    assert alerts.haversine_arcsec(res.ra, res.dec, three_c_273["ra"], three_c_273["dec"]) > res.host_radius_arcsec
    dwarf = next(g for g in groups if g["source_id"] == "[RGG2003] new galaxy")
    to_centre = alerts.haversine_arcsec(dwarf["ra"], dwarf["dec"], pgc41121["ra"], pgc41121["dec"])
    assert to_centre < alerts.D25_ALIAS_MIN_ARCSEC  # the old rule tagged it: within 10" of the centre
    assert pgc41121["separation_arcsec"] + to_centre > res.host_radius_arcsec  # but the cone may miss a nearer entry
    assert dwarf.get("pgc") is None and all(g.get("pgc") != 41121 for g in groups)


def test_d25_alias_needs_the_whole_neighbourhood_in_the_cone() -> None:
    """Synthetic: an unnamed cone galaxy 9" from a HyperLEDA centre takes its PGC only when every entry nearer to
    that centre would be inside the host cone (or it carries one of the HyperLEDA names)."""
    def run(centre_sep: float, name: str) -> AlertEnrichment:
        pgc = _d25_galaxy(ALERT.ra, ALERT.dec + centre_sep / 3600, ALERT, 5.0)
        pgc.update({"semi_major_arcsec": None, "semi_minor_arcsec": None, "dlr_arcsec": None, "d_dlr": None,
                    "hyperleda_names": ["3C273"]})
        entry = {"source_id": name, "ra": ALERT.ra, "dec": ALERT.dec + (centre_sep - 9.0) / 3600,
                 "separation_arcsec": centre_sep - 9.0, "data": {"otype": "G", "rvz_redshift": 0.0053}}
        return asyncio.run(enricher_with(lambda ra, dec, cats: record_of(
            ra, dec, cats, sources={} if "gaia_dr3" in cats else {"simbad": [entry]}), ([pgc], None, False)).enrich(ALERT))

    edge = run(55.0, "[X] dwarf")  # 55 + 9 > 60: the galaxy's own entry may lie outside the cone
    assert edge.host_candidates[0].get("pgc") is None
    inside = run(30.0, "[X] dwarf")  # 30 + 9 <= 60: the cone holds every nearer entry
    assert inside.host_candidates[0].get("pgc") == 1
    named = run(55.0, "3C 273")  # carries a HyperLEDA name: tagged wherever it is
    assert named.host_candidates[0].get("pgc") == 1


# ---------------------------------------------------------------------------
# Local Group dwarf members (recorded + synthetic)
# ---------------------------------------------------------------------------


def test_sculptor_rr_lyrae_is_an_extragalactic_star(round3: dict[str, AlertEnrichment]) -> None:
    """Regression: Sculptor's current D25 (34") made its RR Lyrae star 12' from the centre 'not projected on any
    HyperLEDA galaxy ... a Galactic star'. Membership uses its stellar extent; host association keeps the D25."""
    res = round3["SclG_V0214"]
    assert res.status == "done" and res.stellar_counterpart is True and res.known_variable is True
    assert res.known_star is False and res.host is None
    assert any("Local Group dwarf Sculptor" in e and "one of its stars" in e for e in res.evidence)
    assert not any("a Galactic star" in e for e in res.evidence)
    sculptor = next(g for g in res.d25_galaxies if g["pgc"] == 3589)
    assert sculptor["log_d25"] == pytest.approx(1.056) and sculptor["d_dlr"] > 15  # the current D25 is kept


def test_local_group_dwarf_extent_geometry() -> None:
    _, ra, dec, r_h, ell, pa, dm = next(d for d in alerts.LOCAL_GROUP_DWARFS if d[0] == "Sculptor")
    assert (r_h, ell, pa, dm) == (11.3, 0.32, 99.0, 19.67)  # McConnachie (2012)
    assert alerts.local_group_dwarf_at(ra, dec)[:2] == ("Sculptor", 0.0)
    # 12' from the centre (EV* SclG V0214): ~1.1 r_h.
    found = alerts.local_group_dwarf_at(14.83046939535, -33.6108759036)
    assert found is not None and found[0] == "Sculptor" and 1.0 < found[1] < 1.2
    # Along the minor axis (PA 9 deg) the radius is (1 - e) r_h: 30' there is 4.0 r_h, along the major axis 2.65.
    north = alerts.local_group_dwarf_at(ra, dec + 30 / 60)
    east = alerts.local_group_dwarf_at(ra + 30 / 60 / math.cos(math.radians(dec)), dec)
    assert north is not None and east is not None and north[1] > east[1]
    assert alerts.local_group_dwarf_at(ra, dec + 3.0) is None  # 16 r_h: beyond the stellar extent
    assert all(d[0] not in {"Sagittarius dSph", "LMC", "SMC", "Andromeda", "Triangulum"} for d in alerts.LOCAL_GROUP_DWARFS)


@pytest.mark.parametrize(("offset_rh", "expected", "why"), [
    (1.0, False, "one of its stars"),
    (4.5, None, "it may be one of its stars"),
    (9.0, True, "not projected on any HyperLEDA galaxy"),
])
def test_stellar_counterpart_near_a_local_group_dwarf(offset_rh: float, expected: bool | None, why: str) -> None:
    """Synthetic: a SIMBAD RR Lyrae star along Fornax's major axis (r_h = 16.6', PA 41), no galaxy nearby."""
    _, ra, dec, r_h, *_ = next(d for d in alerts.LOCAL_GROUP_DWARFS if d[0] == "Fornax")
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    at = SkyCoord(ra * u.deg, dec * u.deg).directional_offset_by(41 * u.deg, offset_rh * r_h * u.arcmin)
    alert = Alert("manual", "fornax_star", float(at.ra.deg), float(at.dec.deg), 61306.3, None, None, None, None, "")
    star = {"source_id": "[X] RR", "ra": alert.ra, "dec": alert.dec, "separation_arcsec": 0.1, "data": {"otype": "RR*"}}
    res = asyncio.run(enricher_with(lambda r, d, cats: record_of(r, d, cats, sources={"simbad": [star]} if "gaia_dr3"
                                                                 in cats else {})).enrich(alert))
    assert res.status == "done" and res.known_star is expected
    assert any(why in e for e in res.evidence), res.evidence


# ---------------------------------------------------------------------------
# Cluster types
# ---------------------------------------------------------------------------


def test_simbad_cluster_candidates_are_clusters() -> None:
    """Regression: the current SIMBAD codes 'Cl?' (Cluster*_Candidate, e.g. M31's PHAT candidates [JSD2012] PC)
    and 'As?' (Association_Candidate) were not clusters, so the cluster vetoes did not protect them."""
    for code in ("Cl?", "As?", "Cl*", "GlC", "Gl?", "OpC", "As*", "St*", "MGr", "Cluster*_Candidate",
                 "Association_Candidate", "C?*"):
        assert code in alerts.SIMBAD_CLUSTER_TYPES and alerts._is_cluster({"catalog": "simbad", "object_type": code})
    assert not alerts._is_cluster({"catalog": "simbad", "object_type": "G"})
    gaia = {"catalog": "gaia_dr3", "ra": 10.681653, "dec": 41.210579}
    pc = {"catalog": "simbad", "source_id": "[JSD2012] PC 1432", "object_type": "Cl?", "ra": 10.681653,
          "dec": 41.210579 + 0.3 / 3600}
    found = alerts._coincident_extended(gaia, [pc])
    assert found is not None and found[0]["source_id"] == "[JSD2012] PC 1432" and found[1] == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# ALeRCE: superseded objects and classifier-version metadata of stored rows (synthetic)
# ---------------------------------------------------------------------------

ROW = {"oid": "ZTF18abvvwjv", "meanra": 10.0, "meandec": 20.0, "lastmjd": 61300.5, "firstmjd": 61200.0,
       "class": "SNIa", "probability": 0.41, "classifier": "lc_classifier"}
SNIA_OPTIONS = {"classifier": "lc_classifier", "class_name": "SNIa", "mjd_field": "lastmjd"}
DETECTIONS = [{"mjd": 61300.5, "magpsf": 18.0, "sigmapsf": 0.05, "fid": 2, "isdiffpos": "t", "candid": "1"}]


def _versions(newest_class: str, newest_probability: float) -> list[dict[str, Any]]:
    return [{"classifier_name": "lc_classifier", "classifier_version": "hierarchical_rf_1.1.0", "class_name": "SNIa",
             "probability": 0.41, "ranking": 1},
            {"classifier_name": "lc_classifier", "classifier_version": "lc_classifier_1.1.13",
             "class_name": newest_class, "probability": newest_probability, "ranking": 1}]


async def _poll_alerce(svc: AlertService, status: int, versions: list[dict[str, Any]]) -> alerts.PollResult:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        mock.get("https://api.alerce.online/ztf/v1/objects/").mock(return_value=httpx.Response(200, json={"items": [ROW]}))
        mock.get(url__regex=r".*/probabilities.*").mock(
            return_value=httpx.Response(status, json=versions if status == 200 else {"detail": "busy"}))
        mock.get(url__regex=r".*/detections$").mock(return_value=httpx.Response(200, json=DETECTIONS))
        return await svc.poll("alerce", since_mjd=61300.0, until_mjd=61301.0, limit=5, crossmatch=False,
                              options=SNIA_OPTIONS)


async def test_row_stored_during_a_lookup_outage_is_reclassified_when_superseded(store: AlertStore) -> None:
    """Regression: poll 1 (/probabilities 503) stored ZTF18abvvwjv as 'SNIa 0.41 unresolved'; poll 2 found that the
    newest version ranks LPV first and dropped the object before the store saw it, so the row stayed SNIa for ever
    and class filters kept returning it."""
    async with offline_client() as client:
        svc = AlertService(store, client, None)
        first = await _poll_alerce(svc, 503, [])
        assert first.inserted == 1 and any("classifier versions unavailable" in w for w in first.warnings)
        assert store.get("alerce:ZTF18abvvwjv")["extra"]["classifier_choice"] == "unresolved"
        second = await _poll_alerce(svc, 200, _versions("LPV", 0.659))
        assert (second.fetched, second.inserted, second.updated, second.superseded) == (0, 0, 0, 1)
        assert any("1 object(s) dropped" in w for w in second.warnings)
        assert any("1 stored alert(s) reclassified" in w and "alerce:ZTF18abvvwjv" in w for w in second.warnings)
        row = store.get("alerce:ZTF18abvvwjv")
        assert (row["classification"], row["probability"], row["n_updates"]) == ("LPV", 0.659, 1)
        assert row["extra"]["classifier_choice"] == "superseded"
        assert row["extra"]["classifier_version"] == "lc_classifier_1.1.13"
        assert row["extra"]["superseded_class"] == {"class": "SNIa", "probability": 0.41}
        assert row["extra"]["candid"] == "1" and row["magpsf"] == 18.0  # the detection is left as it is
        assert store.list(classification="SNIa") == []
        assert [r["id"] for r in store.list(classification="LPV")] == ["alerce:ZTF18abvvwjv"]
        # Idempotent; and another failed lookup of the same detection keeps the superseded mark.
        third = await _poll_alerce(svc, 200, _versions("LPV", 0.659))
        assert third.superseded == 0 and store.get("alerce:ZTF18abvvwjv")["n_updates"] == 1
        outage = await _poll_alerce(svc, 503, [])
        assert (outage.fetched, outage.updated, outage.unchanged) == (1, 0, 1)
        row = store.get("alerce:ZTF18abvvwjv")
        assert row["classification"] == "LPV" and row["extra"]["classifier_choice"] == "superseded"
        # ALeRCE ranks it SNIa again: the next poll fetches it and reclassifies the row back.
        back = await _poll_alerce(svc, 200, _versions("SNIa", 0.7))
        assert back.updated == 1 and store.get("alerce:ZTF18abvvwjv")["classification"] == "SNIa"
        assert "superseded_class" not in store.get("alerce:ZTF18abvvwjv")["extra"]


async def test_row_stored_during_a_lookup_outage_gets_its_version_when_resolved(store: AlertStore) -> None:
    """Regression: when the newest version ranks the same class with the same probability, the row stored during
    the outage kept classifier_choice 'unresolved' and no classifier_version ('updated 0')."""
    async with offline_client() as client:
        svc = AlertService(store, client, None)
        await _poll_alerce(svc, 503, [])
        resolved = await _poll_alerce(svc, 200, _versions("SNIa", 0.41))
        assert (resolved.updated, resolved.unchanged, resolved.superseded) == (1, 0, 0)
        row = store.get("alerce:ZTF18abvvwjv")
        assert (row["classification"], row["probability"]) == ("SNIa", 0.41)
        assert row["extra"]["classifier_choice"] == "newest_version"
        assert row["extra"]["classifier_version"] == "lc_classifier_1.1.13"
        again = await _poll_alerce(svc, 200, _versions("SNIa", 0.41))
        assert (again.updated, again.unchanged) == (0, 1)


def test_mark_superseded_never_inserts(store: AlertStore) -> None:
    ghost = Alert("alerce", "ZTF26zzzzzzz", 1.0, 2.0, 61300.0, None, None, "SNIa", 0.4, "",
                  extra={"newest_version_class": {"class": "LPV", "probability": 0.6, "classifier_version": "v2"}})
    assert store.mark_superseded([ghost]) == [] and store.count() == 0


def test_same_detection_with_a_new_candid_is_updated(store: AlertStore) -> None:
    """Mutation M25: a re-poll of the same MJD with another candid (another packet of the same instant) refreshes
    the photometry; the same candid changes nothing."""
    base = Alert("alerce", "ZTF26aaaaaax", 150.0, 2.0, 61306.3, 19.1, "r", "SN", 0.9, "", extra={"candid": "100"})
    assert store.upsert(base) == "inserted"
    assert store.upsert(Alert.from_dict({**base.as_dict(), "magpsf": 19.1})) == "unchanged"
    other = Alert.from_dict({**base.as_dict(), "magpsf": 18.7, "band": "g", "extra": {"candid": "101"}})
    assert store.upsert(other) == "updated"
    row = store.get(base.alert_id)
    assert (row["magpsf"], row["band"], row["extra"]["candid"], row["n_updates"]) == (18.7, "g", "101", 1)


# ---------------------------------------------------------------------------
# set_enrichment: a concurrent poll between the read and the write (synthetic)
# ---------------------------------------------------------------------------


class RacingStore(AlertStore):
    """An AlertStore whose set_enrichment lets another thread commit right after its SELECT (the race window)."""

    hook: Any = None

    @contextlib.contextmanager
    def _conn(self) -> Iterator[Any]:
        with super()._conn() as conn:
            outer = self

            class Proxy:
                def execute(self, sql: str, *args: Any) -> Any:
                    cur = conn.execute(sql, *args)
                    if outer.hook is not None and sql.lstrip().upper().startswith("SELECT CROSSMATCH_STATUS, ENRICHMENT_JSON"):
                        hook, outer.hook = outer.hook, None
                        row = cur.fetchone()
                        thread = threading.Thread(target=hook)
                        thread.start()
                        thread.join()
                        return type("Cursor", (), {"fetchone": lambda _self: row})()
                    return cur

                def __getattr__(self, name: str) -> Any:
                    return getattr(conn, name)

            yield Proxy()


@pytest.mark.parametrize(("shift_arcsec", "outcome", "status"), [(3.0, "stale_position", "pending"),
                                                                 (0.3, "stored", "done")])
def test_set_enrichment_rechecks_a_row_changed_after_its_read(tmp_path: Path, shift_arcsec: float, outcome: str,
                                                              status: str) -> None:
    """Regression: an upsert moving the alert 3" between set_enrichment's SELECT and UPDATE was overwritten with
    'done' and the flags of the old position (never re-crossmatched). A small move (< 1") still stores."""
    url = f"sqlite:///{(tmp_path / 'race.sqlite3').as_posix()}"
    racing, other = RacingStore(MetadataStore(url)), AlertStore(MetadataStore(url))
    alert = Alert("fink", "ZTF1", 10.0, 20.0, 61300.0, 19.0, "r", "SN candidate", 0.9, "")
    moved = Alert("fink", "ZTF1", 10.0, 20.0 + shift_arcsec / 3600, 61300.5, 18.9, "r", "SN candidate", 0.9, "")
    racing.upsert(alert)
    seen: list[str] = []
    racing.hook = lambda: seen.append(other.upsert(moved))
    enrichment = AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=["gaia_dr3"],
                                 ra=alert.ra, dec=alert.dec, is_new=True)
    assert racing.set_enrichment(alert.alert_id, enrichment) == outcome
    assert seen == ["updated"]
    row = other.get(alert.alert_id)
    assert row["dec"] == pytest.approx(moved.dec) and row["crossmatch_status"] == status
    assert row["crossmatch_attempts"] == (1 if status == "done" else 0)


# ---------------------------------------------------------------------------
# Blank class names; class-list outages; until-only windows (synthetic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broker", ["alerce", "fink", "fink_lsst"])
def test_blank_class_names_are_rejected(broker: str) -> None:
    """Regression: class_name '' disabled ALeRCE's class filter and made every object 'superseded' (0 alerts)."""
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="must not be blank"):
            normalize_options(broker, {"class_name": blank})
    assert normalize_options(broker, {"class_name": " SN "})["class_name"] == "SN"
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_options("alerce", {"classifier": ""})
    for blank in ("", "  "):
        with pytest.raises(ValueError):
            PollRequest(class_name=blank)
        with pytest.raises(ValueError):
            PollRequest(classifier=blank)


def test_router_and_cli_reject_blank_class_names(store: AlertStore, capsys: pytest.CaptureFixture[str]) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.alert_store = store
    with TestClient(app) as client:
        answer = client.post("/api/v1/alerts/poll", json={"class_name": "", "crossmatch": False})
    assert answer.status_code == 422
    parser = argparse.ArgumentParser(prog="astrosearch")
    alerts.register_cli(parser.add_subparsers(dest="command"))
    for option in ("--class", "--classifier"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["alerts", "poll", option, " "])
        assert exc.value.code == 2 and "must not be blank" in capsys.readouterr().err


CLASSIFIERS = [{"classifier_name": "stamp_classifier", "classifier_version": "stamp_classifier_1.0.4",
                "classes": ["SN", "AGN", "VS", "asteroid", "bogus"]}]


async def test_empty_window_survives_a_class_list_outage() -> None:
    """Regression: an empty ALeRCE/Fink window whose class-list check failed (503) turned the broker's valid empty
    answer into a failed poll (HTTP 502 / exit 1). An unknown class is still an error once the list answers."""
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.alerce.online/ztf/v1/objects/").mock(return_value=httpx.Response(200, json={"items": []}))
        classifiers = mock.get("https://api.alerce.online/ztf/v1/classifiers/").mock(
            side_effect=[httpx.Response(503, text="down"), httpx.Response(200, json=CLASSIFIERS)])
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(200, json=[]))
        mock.get("https://api.ztf.fink-portal.org/api/v1/classes").mock(return_value=httpx.Response(503, text="down"))
        async with offline_client() as client:
            quiet = await fetch_alerts(client, "alerce", since_mjd=61300.0, until_mjd=61301.0, limit=5)
            assert quiet.alerts == [] and quiet.requests == 2
            assert any("could not be checked" in w and "HTTP 503" in w for w in quiet.warnings)
            with pytest.raises(ValueError, match="Unknown ALeRCE class 'Nope'"):
                await fetch_alerts(client, "alerce", since_mjd=61300.0, until_mjd=61301.0, limit=5,
                                   options={"class_name": "Nope"})
            fink = await FinkZTFBroker().fetch(client, since_mjd=61300.0, until_mjd=61301.0, class_name="SN candidate")
            assert fink.alerts == [] and any("fink: empty window; the class could not be checked" in w
                                             for w in fink.warnings)
    assert classifiers.call_count == 2


PARTIAL_CLASSIFIERS = [{"classifier_name": "lc_classifier", "classes": ["SNIa"]}]


@pytest.mark.parametrize("bad", [[], {"detail": "oops"}, PARTIAL_CLASSIFIERS])
async def test_unusable_class_lists_are_not_cached(bad: Any) -> None:
    """Regression (and mutation M5): a valid-but-empty (or malformed, or partial) class list was cached for 6 h,
    so every later empty window failed with "known: []". It is used once and asked for again: an empty or
    malformed one only adds a warning; one listing other classifiers but not the one polled is an error then."""
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.alerce.online/ztf/v1/objects/").mock(return_value=httpx.Response(200, json={"items": []}))
        route = mock.get("https://api.alerce.online/ztf/v1/classifiers/").mock(
            side_effect=[httpx.Response(200, json=bad), httpx.Response(200, json=CLASSIFIERS)])
        async with offline_client() as client:
            if bad is PARTIAL_CLASSIFIERS:
                with pytest.raises(ValueError, match=r"Unknown ALeRCE classifier 'stamp_classifier'; known: \['lc_"):
                    await fetch_alerts(client, "alerce", since_mjd=61300.0, until_mjd=61300.5, limit=5)
            else:
                first = await fetch_alerts(client, "alerce", since_mjd=61300.0, until_mjd=61300.5, limit=5)
                assert first.alerts == [] and len(first.warnings) == 1 and "no classifier listed" in first.warnings[0]
            second = await fetch_alerts(client, "alerce", since_mjd=61300.5, until_mjd=61301.0, limit=5)
            assert second.warnings == [] and second.requests == 2
            third = await fetch_alerts(client, "alerce", since_mjd=61301.0, until_mjd=61301.5, limit=5)
            assert third.requests == 1  # the valid list is cached
    assert route.call_count == 2


def test_until_only_window_uses_the_lookback_when_the_cursor_is_later(store: AlertStore) -> None:
    """Regression: until_mjd alone with stored alerts newer than it gave an 86-second window, silently."""
    store.upsert(Alert("fink", "ZTF1", 10.0, 20.0, 61311.0, 19.0, "r", "SN candidate", 0.9, ""))
    svc = AlertService(store, httpx.AsyncClient(), None, clock=lambda: 61312.0, lookback_days=2.0)
    options = normalize_options("fink", None)
    window = svc.plan_window("fink", options, None, 61300.0)
    assert (window.since, window.until, window.kind) == (61298.0, 61300.0, "explicit")
    assert len(window.warnings) == 1 and "lookback_days = 2 d" in window.warnings[0]
    # A cursor-derived start before until_mjd is kept, without a warning.
    ok = svc.plan_window("fink", options, None, 61311.5)
    assert (ok.since, ok.until, ok.warnings) == (61310.0, 61311.5, [])
    # The default window (no until) is unchanged.
    new = svc.plan_window("fink", options, None, None)
    assert (new.since, new.until, new.kind, new.warnings) == (61310.0, 61312.0, "new", [])


async def test_until_only_poll_reports_the_window_change(store: AlertStore) -> None:
    store.upsert(Alert("fink", "ZTF1", 10.0, 20.0, 61311.0, 19.0, "r", "SN candidate", 0.9, ""))
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.ztf.fink-portal.org/api/v1/latests").mock(return_value=httpx.Response(200, json=[]))
        mock.get("https://api.ztf.fink-portal.org/api/v1/classes").mock(
            return_value=httpx.Response(200, json={"Fink classifiers": ["SN candidate"]}))
        async with offline_client() as client:
            svc = AlertService(store, client, None, clock=lambda: 61312.0)
            result = await svc.poll("fink", until_mjd=61300.0, crossmatch=False)
    assert (result.since_mjd, result.until_mjd) == (61299.0, 61300.0)
    assert any("give since_mjd to choose it" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Mutation-testing gaps: semaphore slot on cancellation, strict JSON
# ---------------------------------------------------------------------------


async def test_cancelled_claim_never_leaks_a_semaphore_slot() -> None:
    """Mutation M7: a claim waiting in _acquire_or_bypass is cancelled at every step around the moment a slot is
    released to it; the slot acquired meanwhile must be given back (a leaked one blocks batches for ever)."""
    for steps in range(8):
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        waiter = asyncio.ensure_future(alerts._acquire_or_bypass(semaphore, asyncio.Event()))
        for _ in range(3):
            await asyncio.sleep(0)
        semaphore.release()
        for _ in range(steps):
            await asyncio.sleep(0)
        waiter.cancel()
        try:
            held = await waiter
        except asyncio.CancelledError:
            held = False
        if held:  # it acquired the slot before the cancellation arrived: its owner releases it
            semaphore.release()
        assert not semaphore.locked(), steps
        await asyncio.wait_for(semaphore.acquire(), timeout=1.0)


def test_stored_values_are_strict_json(store: AlertStore) -> None:
    """Mutation M13: a non-finite value reaching the store (an enrichment or extra built in code, not parsed) is
    stored as null, never as a NaN token that strict JSON readers reject."""
    assert alerts._dumps({"a": math.nan, "b": [math.inf, 1.0], "c": {"d": -math.inf}}) == \
        '{"a": null, "b": [null, 1.0], "c": {"d": null}}'
    alert = Alert("fink", "ZTF2", 10.0, 20.0, 61300.0, 19.0, "r", "SN candidate", 0.9, "", extra={"score": math.nan})
    store.upsert(alert)
    enrichment = AlertEnrichment(status="done", match_radius_arcsec=2.0, host_radius_arcsec=60.0, catalogs=[],
                                 ra=alert.ra, dec=alert.dec, elapsed_ms=math.inf)
    assert store.set_enrichment(alert.alert_id, enrichment) == "stored"
    row = store.get(alert.alert_id)
    assert row["extra"] == {"score": None} and row["enrichment"]["elapsed_ms"] is None
    json.loads(json.dumps(row), parse_constant=lambda token: pytest.fail(f"non-strict JSON token {token}"))


def test_broker_error_class_is_unchanged_for_real_failures() -> None:
    """A non-empty window never validates the class; an outage of the objects endpoint itself still fails."""
    with respx.mock(assert_all_mocked=True) as mock:
        mock.get("https://api.alerce.online/ztf/v1/objects/").mock(return_value=httpx.Response(503, text="down"))

        async def run() -> None:
            async with offline_client() as client:
                await fetch_alerts(client, "alerce", since_mjd=61300.0, until_mjd=61301.0, limit=5)

        with pytest.raises(BrokerError) as err:
            asyncio.run(run())
    assert err.value.unreachable
