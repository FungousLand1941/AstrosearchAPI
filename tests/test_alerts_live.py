"""Live tests for alerts.py against the real ALeRCE / Fink brokers and archives (``pytest -m live``).

A test only skips when a service is unreachable (network error, timeout, HTTP 5xx/429:
``BrokerError.unreachable`` or a catalog failure of an UNREACHABLE_ERROR_TYPES type). Empty
answers, parse failures, enrichment exceptions and wrong astrophysics fail.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from astropy import units as u

from alerts import (
    C_KMS,
    FINK_ZTF_CLASS_SCORES,
    UNREACHABLE_ERROR_TYPES,
    AlerceBroker,
    Alert,
    AlertEnricher,
    AlertEnrichment,
    AlertService,
    AlertStore,
    BrokerError,
    FetchResult,
    build_alert_service,
    fetch_alerts,
    now_mjd,
)
from datasets import MetadataStore

pytestmark = pytest.mark.live

# SIMBAD ICRS (J2000) positions, queried 2026-09-28.
T_CRB = (239.87567594413002, 25.92017038415)
AT2018COW = (244.00105833333333, 22.268083333333333)
M31N_2008_12A = (11.370375, 41.902806)
M82_NUCLEUS = (148.96969, 69.67938)
SN2014J = (148.925583, 69.673889)
AT2017GFO = (197.450375, -23.381481)
# Fink/LSST data exist from 2026 (last processed night so far 2026-07-14).
LSST_SINCE_MJD = 61200.0
ALERCE_SN = {"classifier": "stamp_classifier", "class_name": "SN", "mjd_field": "firstmjd"}


def skip_if_unreachable(exc: BrokerError, what: str) -> None:
    if exc.unreachable:
        pytest.skip(f"{what} unreachable: {exc}")


SUPERSEDED_WARNING = re.compile(r"^alerce: \d+ object\(s\) dropped: their newest \S+ version ranks another class")


async def live_fetch(broker: str, since: float, until: float, limit: int, options: dict[str, Any], *,
                     superseded_ok: bool = False) -> FetchResult:
    """A live fetch that must answer without warnings -- except, with ``superseded_ok``, ALeRCE's report of
    objects dropped because their newest classifier version ranks another class first (live streams
    classified by superseded versions: lc_classifier lastmjd windows)."""
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        try:
            result = await fetch_alerts(client, broker, since_mjd=since, until_mjd=until, limit=limit, options=options)
        except BrokerError as exc:
            skip_if_unreachable(exc, broker)
            raise
    unexpected = [w for w in result.warnings if not (superseded_ok and SUPERSEDED_WARNING.match(w))]
    assert unexpected == [], unexpected
    return result


def check_common(alert: Alert, broker: str, since: float, until: float) -> None:
    assert alert.broker == broker and alert.object_id
    assert 0.0 <= alert.ra < 360.0 and -90.0 <= alert.dec <= 90.0
    assert since - 1e-6 <= alert.mjd <= until + 1e-6
    assert alert.url.startswith("https://") and alert.object_id in alert.url
    if alert.magpsf is not None:
        assert 10.0 < alert.magpsf < 26.0
    if alert.probability is not None:
        assert 0.0 <= alert.probability <= 1.0


async def test_alerce_recent_sn_candidates_live() -> None:
    until = now_mjd()
    since = until - 14.0
    result = await live_fetch("alerce", since, until, 5, ALERCE_SN)
    assert result.alerts, "ALeRCE returned no stamp-classifier SN candidates first detected in the last 14 days"
    for alert in result.alerts:
        assert alert.object_id.startswith("ZTF") and alert.survey == "ztf"
        assert since <= alert.first_mjd <= alert.mjd <= until + 1e-6
        check_common(alert, "alerce", since, until)
        assert alert.classification == "SN"
        assert alert.band in {"g", "r", "i"} and 12.0 < alert.magpsf < 22.5  # ZTF alert depth ~20.5-21
        assert alert.is_negative in (True, False) and alert.magpsf_err is not None and 0 < alert.magpsf_err < 1.0
        assert alert.dec > -35.0  # Palomar (latitude +33.4 deg) cannot reach the far south
    if result.truncated:
        assert result.boundary_mjd == result.alerts[-1].first_mjd


async def test_fink_ztf_recent_sn_candidates_live() -> None:
    until = now_mjd()
    since = until - 14.0
    result = await live_fetch("fink", since, until, 5, {"class_name": "SN candidate"})
    assert result.alerts, "Fink returned no 'SN candidate' alerts in the last 14 days"
    for alert in result.alerts:
        check_common(alert, "fink", since, until)
        assert alert.object_id.startswith("ZTF") and alert.band in {"g", "r", "i"}
        # Fink SN candidates need snn_snia_vs_nonia > 0.5 or snn_sn_vs_all > 0.5; P(SN) is snn_sn_vs_all.
        scores = alert.extra["scores"]
        assert set(scores) >= {c.split(":")[1] for c in FINK_ZTF_CLASS_SCORES["SN candidate"]}
        assert max(scores["snn_snia_vs_nonia"] or 0.0, scores["snn_sn_vs_all"] or 0.0) > 0.5
        assert alert.probability == scores["snn_sn_vs_all"]
        assert alert.first_mjd <= alert.mjd and alert.dec > -35.0
        assert alert.is_negative is (alert.extra["isdiffpos"] in {"f", "0"})


async def test_fink_ztf_accepts_bare_simbad_class_live() -> None:
    until = now_mjd()
    since = until - 30.0
    bare = await live_fetch("fink", since, until, 3, {"class_name": "RRLyrae"})
    prefixed = await live_fetch("fink", since, until, 3, {"class_name": "(SIMBAD) RRLyrae"})
    assert bare.alerts, "Fink returned no RR Lyrae alerts in the last 30 days"
    # Both spellings select the same alerts.
    assert [a.object_id for a in bare.alerts] == [a.object_id for a in prefixed.alerts]
    for alert in bare.alerts:
        check_common(alert, "fink", since, until)
        assert alert.is_negative in (True, False)


async def test_fink_lsst_alerts_live() -> None:
    until = now_mjd()
    result = await live_fetch("fink_lsst", LSST_SINCE_MJD, until, 5, {"class_name": "extragalactic_new_candidate"})
    assert result.alerts, "Fink/LSST returned no extragalactic_new_candidate alerts since MJD 61200"
    for alert in result.alerts:
        check_common(alert, "fink_lsst", LSST_SINCE_MJD, until)
        assert alert.survey == "lsst" and int(alert.object_id) > 0 and alert.object_id.isdigit()
        assert alert.band in {"u", "g", "r", "i", "z", "y"}
        assert alert.dec < 35.0  # Rubin at Cerro Pachon (latitude -30.2 deg)
        flux = alert.extra["psf_flux_njy"]
        assert flux is not None and alert.is_negative is (flux < 0)
        assert alert.magpsf == pytest.approx((abs(flux) * u.nJy).to(u.ABmag).value, abs=1e-6)
        assert (alert.extra["midpoint_mjd_tai"] - alert.mjd) * 86400.0 == pytest.approx(37.0, abs=1e-3)


async def test_fink_lsst_tag_without_api_support_is_invalid_input_live() -> None:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        with pytest.raises(ValueError, match="Livestream"):
            try:
                await fetch_alerts(client, "fink_lsst", since_mjd=61234.0, until_mjd=61236.0, limit=2,
                                   options={"class_name": "uniform_sample"})
            except BrokerError as exc:
                skip_if_unreachable(exc, "fink_lsst")
                raise


async def pick_window(broker: str, options: dict[str, Any], limit: int) -> tuple[float, float, list[str]]:
    """A recent real window holding more than `limit` (and at most 15) objects: (since, until, ids).

    Windows end just after one of the recent alerts -- the newest first, then earlier ones: a
    broker's newest alert can follow a gap of several days (ZTF weather or maintenance; Fink SN
    candidates on 2026-09-29: 5.8 days), so no window ending there holds enough objects. The
    14-day listing picks the candidate windows (only where it is complete); each is then
    fetched on its own and must hold the objects without truncation.
    """
    now = now_mjd()
    recent = await live_fetch(broker, now - 14.0, now, 100, options)
    if not recent.alerts:
        pytest.fail(f"{broker}: no alerts in the last 14 days to build a window from")
    times = sorted((a.first_mjd if broker == "alerce" else a.mjd for a in recent.alerts), reverse=True)
    # A truncated listing holds the newest objects only: windows reaching further back are not counted.
    complete_from = times[-1] if recent.truncated else now - 14.0
    fetched = 0
    for anchor in dict.fromkeys(times):
        end = anchor + 0.001
        for days in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
            if end - days < complete_from:
                break
            if not limit < sum(1 for t in times if end - days <= t <= end) <= 15:
                continue
            found = await live_fetch(broker, end - days, end, 100, options)
            if limit < len(found.alerts) <= 15 and not found.truncated:
                return end - days, end, sorted(a.alert_id for a in found.alerts)
            fetched += 1
            if fetched >= 8:  # polite: the listing and the broker disagree repeatedly
                pytest.fail(f"{broker}: {fetched} candidate windows held a different number of objects when fetched")
    pytest.fail(f"{broker}: no window with {limit + 1}-15 objects in the last 14 days")


@pytest.mark.parametrize(("broker", "options"), [("fink", {"class_name": "SN candidate"}), ("alerce", ALERCE_SN)])
async def test_overfull_window_is_ingested_completely_live(tmp_path: Path, broker: str, options: dict[str, Any]) -> None:
    """Regression: a window with more alerts than `limit` is fully ingested over successive default polls."""
    since, until, reference = await pick_window(broker, options, 3)
    store = AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'backlog.sqlite3').as_posix()}"))
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        svc = AlertService(store, client, None, clock=lambda: until, lookback_days=until - since, overlap_days=0.1)
        results = []
        try:
            for _ in range(10):
                res = await svc.poll(broker, limit=3, crossmatch=False, options=options)
                results.append(res)
                if res.window == "new" and res.backlog is None and len(results) > 1:
                    break
        except BrokerError as exc:
            skip_if_unreachable(exc, broker)
            raise
    assert results[0].truncated and results[0].backlog is not None and results[1].window == "backlog"
    stored = sorted(r["id"] for r in store.list(limit=100))
    assert set(reference) <= set(stored), f"missing {sorted(set(reference) - set(stored))}"
    assert all(r.fetched <= 3 for r in results)


def unreachable_only(failures: list[dict[str, Any]]) -> bool:
    return bool(failures) and all(f.get("error_type") in UNREACHABLE_ERROR_TYPES for f in failures)


async def test_poll_and_crossmatch_one_real_alert_live(tmp_path: Path) -> None:
    store = AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'live.sqlite3').as_posix()}"))
    until = now_mjd()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        svc = build_alert_service(client, store=store)
        try:
            result = await svc.poll("alerce", since_mjd=until - 14.0, until_mjd=until, limit=1)
        except BrokerError as exc:
            skip_if_unreachable(exc, "ALeRCE")
            raise
        assert result.fetched == 1 and result.inserted == 1
        row = store.get(result.alert_ids[0])
        assert row is not None and row["crossmatch_status"] in {"done", "partial", "failed"}
        enrichment = row["enrichment"]
        assert enrichment["exception"] is None, f"enrichment raised: {enrichment['exception']}"
        if row["crossmatch_status"] != "done":
            if unreachable_only(enrichment["failures"]):
                pytest.skip(f"archives unreachable: {enrichment['failures']}")
            pytest.fail(f"crossmatch {row['crossmatch_status']} without an outage: {enrichment['failures']}")
        assert set(enrichment["catalog_status"]) == {"gaia_dr3", "simbad", "ned"}
        assert enrichment["is_new"] in (True, False)
        assert enrichment["is_new"] == (not enrichment["counterparts"])
        assert enrichment["host_status"] in {"found", "none_within_radius", "unassociated", "not_applicable_star",
                                             "ambiguous_transient_entry"}
        assert enrichment["host_search_complete"] is True
        # Re-polling the same window is idempotent.
        again = await svc.poll("alerce", since_mjd=result.since_mjd, until_mjd=result.until_mjd, limit=1)
        assert again.inserted == 0 and again.crossmatched == 0 and again.retried == 0 and store.count() == 1


async def enrich_live(ra: float, dec: float, name: str) -> AlertEnrichment:
    from main import build_service

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        res = await AlertEnricher(build_service(client=client)).enrich(
            Alert("manual", name, ra, dec, now_mjd(), None, None, None, None, "", survey="none"))
    down = [f for f in res.failures if f.get("error_type") in UNREACHABLE_ERROR_TYPES]
    if down:
        pytest.skip(f"archives unreachable: {sorted({f['catalog'] for f in down})}")
    assert not res.failures, res.failures
    assert res.status == "done" and res.exception is None
    return res


def names_of(host: dict[str, Any]) -> set[str]:
    return {host["name"], *(a.split(":", 1)[1] for a in host["aliases"])}


async def test_t_crb_enrichment_live() -> None:
    res = await enrich_live(*T_CRB, "T_CrB")
    assert res.known_star is True and res.known_variable is True and res.is_new is False
    gaia = next(c for c in res.counterparts if c["catalog"] == "gaia_dr3")
    assert gaia["parallax_mas"] == pytest.approx(1.09, abs=0.05) and gaia["parallax_over_error"] > 30
    assert res.host is None and res.host_status == "not_applicable_star"


async def test_at2018cow_host_enrichment_live() -> None:
    res = await enrich_live(*AT2018COW, "AT2018cow")
    assert "SN 2018cow" in res.transient_designations
    assert res.known_star is False
    host = res.host
    assert host is not None and res.host_status == "found"
    assert "CGCG 137-068" in names_of(host) or "Z 137-68" in names_of(host)
    assert host["redshift"] == pytest.approx(0.0141, abs=0.0003)
    assert 4.0 < host["separation_arcsec"] < 7.0 and 1.2 < host["projected_offset_kpc"] < 2.0
    # z = 0.0141 > 0.01: a Hubble-flow distance on the Cosmicflows-4 scale (H0 = 74.6), ~56 Mpc for
    # cz_CMB ~ 4.2e3 km/s (Planck's H0 = 67.66 would give ~62 Mpc, a 10% jump from CF4 distances).
    assert host["distance_method"] == "hubble_flow" and host["hubble_constant_kms_mpc"] == pytest.approx(74.6)
    assert host["distance_mpc"] == pytest.approx(C_KMS * host["redshift_cmb"] / 74.6 / (1 + host["redshift"]), rel=0.03)


async def test_m31n_2008_12a_is_extragalactic_live() -> None:
    res = await enrich_live(*M31N_2008_12A, "M31N_2008-12a")
    assert res.stellar_counterpart is True and res.known_variable is True
    assert res.known_star is False  # a nova in M31, not a Galactic star
    host = res.host
    assert host is not None and host["method"] == "d25_ellipse" and names_of(host) & {"Messier 031", "M 31"}
    assert host["d_dlr"] < 1.0 and host["redshift"] == pytest.approx(-0.001, abs=0.0002)


async def test_m82_nucleus_keeps_m82_as_host_live() -> None:
    res = await enrich_live(*M82_NUCLEUS, "M82_nucleus")
    assert res.known_star is False and res.host_status == "found"
    assert names_of(res.host) & {"M 82", "Messier 082"} and res.host["d_dlr"] < 0.1


async def test_sn2014j_host_is_m82_live() -> None:
    res = await enrich_live(*SN2014J, "SN2014J")
    assert "SN 2014J" in res.transient_designations and res.host_search_complete is True
    host = res.host
    assert host is not None and names_of(host) & {"M 82", "Messier 082"}
    assert host["method"] == "d25_ellipse" and 55.0 < host["separation_arcsec"] < 60.0


async def test_at2017gfo_host_is_ngc4993_live() -> None:
    res = await enrich_live(*AT2017GFO, "AT2017gfo")
    assert {"GrW 170817", "AT 2017gfo"} <= set(res.transient_designations)
    host = res.host
    assert host is not None and "NGC 4993" in names_of(host)
    assert host["redshift"] == pytest.approx(0.0098, abs=0.0003) and 1.8 < host["projected_offset_kpc"] < 2.4


# Positions: SIMBAD ICRS (queried 2026-09-28) or Gaia DR3 (J2016) for the foreground stars.
IC10_X1 = (5.120971269999999, 59.28104471)
M86_DISK = (186.55042, 12.98722)  # 150" N of M86's nucleus, inside its D25 ellipse
SN1987A = (83.86661833333334, -69.26975372222223)
SN2011DH = (202.521273125, 47.16970075)
ZTF18AAYEFWP = (306.5039620530862, 33.66233011833)
FOREGROUND_STARS = {
    "Gaia DR3 381261910408440576 (M31)": (10.6395038, 41.2639317),
    "Gaia DR3 369244286970444416 (M31)": (10.7444782, 40.9837978),
    "Gaia DR3 1609299644238883584 (M101)": (210.9002279, 54.4433872),
}


async def test_ic10_x1_is_extragalactic_live() -> None:
    res = await enrich_live(*IC10_X1, "IC10_X-1")
    assert res.known_star is False and res.stellar_counterpart is True and res.known_variable is True
    host = res.host
    assert host is not None and host["pgc"] == 1305 and host["method"] == "d25_ellipse" and host["d_dlr"] < 1.0


async def test_m86_disk_host_is_m86_live() -> None:
    res = await enrich_live(*M86_DISK, "M86_disk")
    host = res.host
    assert host is not None and host["pgc"] == 40653 and host["method"] == "d25_ellipse"
    assert names_of(host) & {"M 86", "Messier 086"} and res.host_search_complete is True


@pytest.mark.parametrize("name", sorted(FOREGROUND_STARS))
async def test_foreground_stars_on_galaxies_are_galactic_live(name: str) -> None:
    res = await enrich_live(*FOREGROUND_STARS[name], name)
    assert res.known_star is True and res.host is None and res.host_status == "not_applicable_star"


async def test_sn1987a_projected_offset_live() -> None:
    res = await enrich_live(*SN1987A, "SN1987A")
    host = res.host
    assert host is not None and host["pgc"] == 17223
    # 1.15 deg from the LMC centre at ~49.5 kpc: ~1.0 kpc (not 77.6 kpc from the Hubble flow).
    assert host["projected_offset_kpc"] == pytest.approx(1.0, abs=0.05)


async def test_sn2011dh_host_named_m51_live() -> None:
    res = await enrich_live(*SN2011DH, "SN2011dh")
    host = res.host
    assert host is not None and host["name"] in {"M 51", "M  51", "MESSIER 051", "Messier 051", "NGC 5194"}
    assert "HOST" not in host["name"] and 5.5 < host["projected_offset_kpc"] < 7.5


async def test_catalogued_cv_with_ztf_name_live() -> None:
    res = await enrich_live(*ZTF18AAYEFWP, "ZTF18aayefwp")
    assert res.known_variable is True and res.is_new is False and res.stellar_counterpart is True
    assert "ZTF18aayefwp" not in res.transient_designations


async def test_rr_lyrae_alerts_are_known_variables_live() -> None:
    """Fink 'RRLyrae' alerts (SIMBAD RR Lyrae cross-match) must come out as known variables."""
    from main import build_service

    until = now_mjd()
    found = await live_fetch("fink", until - 30.0, until, 2, {"class_name": "RRLyrae"})
    assert found.alerts, "Fink returned no RR Lyrae alerts in the last 30 days"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        enricher = AlertEnricher(build_service(client=client))
        for alert in found.alerts:
            assert alert.extra["simbad_otype"] == "RRLyrae"
            res = await enricher.enrich(alert)
            down = [f for f in res.failures if f.get("error_type") in UNREACHABLE_ERROR_TYPES]
            if down:
                pytest.skip(f"archives unreachable: {sorted({f['catalog'] for f in down})}")
            assert res.exception is None and res.known_variable is True, res.evidence
            assert res.stellar_counterpart is True and res.is_new is False


# ---------------------------------------------------------------------------
# Review round 3: galaxy nuclei, DLR hosts, group distances, AGN, ALeRCE duplicates
# ---------------------------------------------------------------------------

# SIMBAD ICRS positions (queried 2026-09-28).
NUCLEI_LIVE = {
    "M87": ((187.70593076725, 12.391123246083334), {"M 87", "Messier 087"}),
    "NGC 3783": ((174.7571236746, -37.73861378972), {"NGC 3783"}),
    "NGC 7469": ((345.8151, 8.8739), {"NGC 7469"}),
    "NGC 4395": ((186.45359712911997, 33.54686115781999), {"NGC 4395"}),
    # Further Seyfert nuclei with spurious >= 5 sigma Gaia proper motions (NGC 1566 10 sigma, NGC 3227
    # 5.7 sigma, M58 8.0 sigma, NGC 5033 7.8 sigma; live sweep of 26 Seyferts and 58 HyperLEDA galaxy
    # centres, 2026-09-28: none is flagged as a Galactic star).
    "NGC 1566": ((65.00165353052, -54.93795130799), {"NGC 1566"}),
    "NGC 3227": ((155.87740214554, 19.86507839075), {"NGC 3227"}),
    "NGC 4579": ((189.43165612500002, 11.818090000000002), {"M 58", "Messier 058", "NGC 4579"}),
    "NGC 5033": ((198.36472916666668, 36.593650000000004), {"NGC 5033"}),
}


@pytest.mark.parametrize("name", sorted(NUCLEI_LIVE))
async def test_galaxy_nuclei_are_not_galactic_stars_live(name: str) -> None:
    """Real Gaia DR3 nuclei with spurious >= 10 sigma proper motions keep their host galaxy."""
    position, host_names = NUCLEI_LIVE[name]
    res = await enrich_live(*position, name)
    assert res.known_star is False and res.known_agn is True
    assert res.host is not None and res.host["method"] == "d25_ellipse" and names_of(res.host) & host_names
    nucleus = min((c for c in res.counterparts if c["catalog"] == "gaia_dr3"), key=lambda c: c["separation_arcsec"])
    assert nucleus["pm_over_error"] > 5  # the spurious proper motion is there ...
    assert any(f"Gaia DR3 {nucleus['source_id']}" in e and "not a star" in e for e in res.evidence)  # ... and rejected


async def test_sn2009ip_is_not_a_galactic_star_live() -> None:
    res = await enrich_live(335.7844166666667, -28.947888888888887, "SN2009ip")
    assert res.stellar_counterpart is True and res.known_star is None
    assert res.host is not None and "NGC 7259" in names_of(res.host) and res.host["method"] == "dlr_outside_d25"
    assert 1.0 < res.host["d_dlr"] < 1.3


@pytest.mark.parametrize(("name", "position", "host_name"), [
    ("SN2023bee", (134.04841666666667, -3.3255694444444446), "NGC 2708"),
    ("SN2018aoz", (177.758, -28.744099999999996), "NGC 3923"),
])
async def test_hosts_outside_the_d25_ellipse_live(name: str, position: tuple[float, float], host_name: str) -> None:
    res = await enrich_live(*position, name)
    host = res.host
    assert res.host_status == "found" and host is not None and host_name in names_of(host)
    assert host["method"] == "dlr_outside_d25" and 1.0 < host["d_dlr"] < 2.0
    assert host["distance_method"] == "cosmicflows4" and 15.0 < host["projected_offset_kpc"] < 35.0


async def test_m100_uses_the_virgo_group_distance_live() -> None:
    res = await enrich_live(185.72471, 15.80888, "SN2006X")
    host = res.host
    assert host is not None and host["pgc"] == 40153 and host["distance_method"] == "cosmicflows4_group"
    assert 15.0 < host["distance_mpc"] < 17.5  # Virgo (M100 Cepheids: ~15-16 Mpc), not 23 Mpc from z = 0.0052
    assert 3.3 < host["projected_offset_kpc"] < 4.1


async def test_m83_host_is_named_m83_live() -> None:
    res = await enrich_live(204.25383, -29.864927777777778, "M83_near_nucleus")
    assert res.host is not None and res.host["name"] in {"M 83", "M  83", "Messier 083"} and res.host["pgc"] == 48082


async def test_blazar_is_known_variable_and_agn_live() -> None:
    res = await enrich_live(187.27791594049, 2.05238823055, "3C273")
    assert res.known_variable is True and res.known_agn is True and res.known_star is False


# ---------------------------------------------------------------------------
# Review round 3 (live): AGN hosts, '*' entries of galaxy nuclei, light radii, Local Group dwarfs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "position", "z", "not_host"), [
    ("3C 273", (187.27791594049, 2.05238823055), 0.158, "[RGG2003] new galaxy"),  # a z = 0.0053 dwarf 10.8" away
    ("BL Lac", (330.68038064033993, 42.27777206022), 0.0655, "2MASX J22024434+4216304"),
    ("PKS 2155-304", (329.71693843745, -30.225588457719997), 0.116, "[FGM91] G1"),
])
async def test_agn_alerts_are_hosted_by_their_own_active_galaxy_live(name: str, position: tuple[float, float], z: float,
                                                                     not_host: str) -> None:
    res = await enrich_live(*position, name)
    host = res.host
    assert res.known_agn is True and host is not None and host["method"] == "agn_nucleus"
    assert host["separation_arcsec"] < 1.0 and host["redshift"] == pytest.approx(z, abs=0.002)
    assert host["name"] != not_host and not_host not in names_of(host)
    assert res.transient_redshift == pytest.approx(z, abs=0.002)


@pytest.mark.parametrize(("name", "position"), [
    ("LEDA 1798300", (192.96265666372997, 27.118380852369995)),
    ("2MASS J09480288+1319111", (147.01196897129, 13.319751523210002)),
])
async def test_simbad_star_entries_of_galaxy_nuclei_are_not_galactic_stars_live(name: str,
                                                                                position: tuple[float, float]) -> None:
    res = await enrich_live(*position, name)
    assert any(c["catalog"] == "simbad" and c["source_id"] == name and c["object_type"] == "*" for c in res.counterparts)
    assert res.known_star is False and res.host is not None and res.host_status == "found"
    assert res.host["redshift"] is not None and res.host["redshift"] > 0.02
    assert any(name in e and "another catalogue entry of the galaxy's nucleus" in e for e in res.evidence)


async def test_blank_point_near_a_low_redshift_dwarf_gets_no_host_live() -> None:
    res = await enrich_live(187.26112, 2.05167, "blank 50\" from [RGG2003] new galaxy")
    assert res.host is None and res.host_status == "unassociated"
    assert any("[RGG2003] new galaxy" in e and "does not make it the host" in e for e in res.evidence)


async def test_sculptor_rr_lyrae_is_an_extragalactic_star_live() -> None:
    res = await enrich_live(14.83046939535, -33.6108759036, "EV* SclG V0214")
    assert res.known_variable is True and res.stellar_counterpart is True and res.known_star is False
    assert any("Local Group dwarf Sculptor" in e for e in res.evidence)


async def test_alerce_rows_per_classifier_version_are_deduplicated_live(tmp_path: Path) -> None:
    """ALeRCE lc_classifier AGN answers repeat objects once per classifier version: ids must be unique,
    repeated objects must take their newest version, and re-polling must change nothing."""
    until = now_mjd()
    options = {"classifier": "lc_classifier", "class_name": "AGN", "mjd_field": "lastmjd"}
    result = await live_fetch("alerce", until - 3.0, until, 15, options, superseded_ok=True)
    assert result.alerts, "ALeRCE returned no lc_classifier AGN objects detected in the last 3 days"
    ids = [a.object_id for a in result.alerts]
    assert len(ids) == len(set(ids))
    for alert in result.alerts:
        check_common(alert, "alerce", until - 3.0 - 1e-6, until)
        # Every stored object -- repeated or not -- is classified by its newest classifier version.
        assert alert.classification == "AGN" and alert.extra["classifier_choice"] == "newest_version"
        assert alert.probability == alert.extra["classifier_versions"][alert.extra["classifier_version"]]
    store = AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'dups.sqlite3').as_posix()}"))
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        svc = AlertService(store, client, None)
        try:
            first = await svc.poll("alerce", since_mjd=until - 3.0, until_mjd=until, limit=15, crossmatch=False,
                                   options=options)
            before = {r["id"]: r for r in store.get_many(first.alert_ids)}
            second = await svc.poll("alerce", since_mjd=until - 3.0, until_mjd=until, limit=15, crossmatch=False,
                                    options=options)
        except BrokerError as exc:
            skip_if_unreachable(exc, "ALeRCE")
            raise
    assert len(first.alert_ids) == len(set(first.alert_ids)) == first.inserted
    assert len(second.alert_ids) == len(set(second.alert_ids)) == second.fetched
    # ALeRCE is live: an object detected again between the polls leaves the fixed window (its lastmjd
    # passes `until`) and the next object enters it, and a late-processed detection inside the window
    # updates its object. A same-detection update is legitimate when its cause is visible -- ALeRCE re-scored
    # the object (the stored probability is its newest version's), its photometry arrived or changed (a
    # /detections request failed in poll 1, or a new candid), or poll 1's version lookup failed for it --
    # but a repeated classifier-version row must never "reclassify" an object (the bug made n_updates reach
    # 5 per repeated object, flipping between the versions' probabilities).
    assert second.inserted == len(set(second.alert_ids) - set(first.alert_ids))
    rows = {r["id"]: r for r in store.get_many(second.alert_ids)}
    for alert_id in set(second.alert_ids) & set(first.alert_ids):
        row, old = rows[alert_id], before[alert_id]
        extra, old_extra = row["extra"], old["extra"]
        oid = alert_id.split(":", 1)[1]
        warned = any(oid in w for w in first.warnings + second.warnings)
        if not warned:  # both version lookups answered: the stored probability is the newest version's
            assert row["classification"] == "AGN" and extra["classifier_choice"] == "newest_version"
            assert row["probability"] == extra["classifier_versions"][extra["classifier_version"]], (alert_id, extra)
        if row["n_updates"] == 0 or row["mjd"] > old["mjd"]:
            continue
        rescored = extra.get("classifier_choice") == "newest_version" and (
            row["probability"] != old["probability"] or extra.get("classifier_version") != old_extra.get("classifier_version"))
        rephotometered = old["magpsf"] is None or extra.get("candid") != old_extra.get("candid")
        assert rescored or rephotometered or warned, (alert_id, old, row)
    assert second.updated == sum(1 for i in set(second.alert_ids) & set(first.alert_ids) if rows[i]["n_updates"] > 0)



# ---------------------------------------------------------------------------
# Review round 2 (live): Galactic-star verdicts, host association, ALeRCE classifier versions
# ---------------------------------------------------------------------------

# Gaia DR3 (J2016) positions.
DECISIVE_STARS = {  # Gaia DSC-extragalactic or galaxy candidates, with decisive astrometry
    "WD 1647493723250083584": (232.72187939379668, 69.04262554294488),
    "UV Per": (32.53473412720054, 57.189166068564425),
    "[LF82] d (on M31)": (10.675847289712872, 41.26275007396792),
}
BRIGHT_M31_STARS = {  # SIMBAD stars on M31's disc with huge Gaia excess noise, M_G < -10 at M31's distance
    "[WWV2004] J0043124+404639": (10.801861658055273, 40.77813034081048),
    "GSC 02805-02180": (10.645509790548862, 41.34538678030782),
}


@pytest.mark.parametrize("name", sorted(DECISIVE_STARS))
async def test_decisive_astrometry_outweighs_gaia_dsc_live(name: str) -> None:
    res = await enrich_live(*DECISIVE_STARS[name], name)
    gaia = min((c for c in res.counterparts if c["catalog"] == "gaia_dr3"), key=lambda c: c["separation_arcsec"])
    assert (gaia["dsc_p_extragalactic"] or 0.0) > 0.5 or gaia["in_galaxy_candidates"] is True
    assert res.known_star is True and res.host is None and res.host_status == "not_applicable_star"
    assert not any("looks like a galaxy nucleus" in e for e in res.evidence)


@pytest.mark.parametrize("name", sorted(BRIGHT_M31_STARS))
async def test_bright_stars_on_m31_are_foreground_live(name: str) -> None:
    res = await enrich_live(*BRIGHT_M31_STARS[name], name)
    assert res.known_star is True and res.host is None
    assert any("M_G = " in e and "-> Galactic foreground star" in e for e in res.evidence), res.evidence


async def test_binary_near_ngc5078_is_galactic_live() -> None:
    res = await enrich_live(199.85616010770747, -27.374244054046667, "EB 6189441739218449664")
    assert res.known_star is True and res.host is None
    ngc5078 = next((g for g in res.d25_galaxies if g["pgc"] == 46490), None)
    assert ngc5078 is None or ngc5078["d_dlr"] > 2.0  # its current D25 (2.6'), not the 2003 one (51')


@pytest.mark.parametrize(("name", "position", "wrong_pgc"), [
    ("PTF 10hv", (210.98408333333336, 54.45863888888889), 50063),  # z = 0.052 SN on M101
    ("PTF 11dws", (184.7785, 47.355805555555555), 39600),  # z = 0.15 SN on M106
    ("SN 2008hz", (10.827583333333333, 42.170611111111114), 2557),  # z = 0.0795 SN on M31
])
async def test_background_supernovae_on_nearby_giants_live(name: str, position: tuple[float, float],
                                                           wrong_pgc: int) -> None:
    res = await enrich_live(*position, name)
    assert res.transient_redshift is not None and res.transient_redshift > 0.04
    assert res.host is None or res.host.get("pgc") != wrong_pgc
    if res.host is not None:
        assert res.host["redshift"] is None or abs(res.host["redshift"] - res.transient_redshift) < 0.02


async def test_sn2016bam_host_is_ngc2445_live() -> None:
    res = await enrich_live(116.71966666666667, 39.02272222222222, "SN 2016bam")
    assert res.host is not None and "NGC 2445" in names_of(res.host) and res.host["object_type"] != "QSO"
    assert res.host["projected_offset_kpc"] < 20


async def test_ned_point_source_does_not_make_sn2002gn_galactic_live() -> None:
    res = await enrich_live(29.22733333333333, -1.113611111111111, "SN 2002gn")
    assert res.known_star is not True and res.host is not None and res.host_status == "found"


async def test_random_blank_positions_rarely_get_a_host_live() -> None:
    """Random positions at |b| > 30 deg with no transient: an adopted host must pass the association criteria
    (a likely chance alignment is 'unassociated'), and most positions get none."""
    from astropy.coordinates import SkyCoord

    rng = random.Random(29)
    positions: list[tuple[float, float]] = []
    while len(positions) < 6:
        ra, dec = rng.uniform(0.0, 360.0), math.degrees(math.asin(rng.uniform(-0.5, 1.0)))
        if abs(SkyCoord(ra * u.deg, dec * u.deg).galactic.b.deg) > 30.0:
            positions.append((ra, dec))
    found = 0
    for i, (ra, dec) in enumerate(positions):
        res = await enrich_live(ra, dec, f"random{i}")
        if res.host is not None:
            found += 1
            host = res.host
            assert host["method"] in {"nearest", "d25_ellipse", "dlr_outside_d25", "agn_nucleus"}
            if host["method"] == "nearest":  # no transient redshift: only a small chance coincidence counts
                assert host["p_chance"] <= 0.1, host
            elif host["method"] == "agn_nucleus":
                assert host["separation_arcsec"] <= res.match_radius_arcsec and res.known_agn is True
            else:
                assert host["d_dlr"] <= 2.0
        else:
            assert res.host_status in {"none_within_radius", "unassociated", "not_applicable_star"}
    assert found <= 3


async def test_alerce_objects_are_classified_by_their_newest_version_live() -> None:
    """lc_classifier SNIa objects by lastmjd: every object kept is SNIa in its newest classifier version;
    objects whose AGN/SNIa row came from a superseded version are dropped (and reported)."""
    until = now_mjd()
    options = {"classifier": "lc_classifier", "class_name": "SNIa", "mjd_field": "lastmjd"}
    result = await live_fetch("alerce", until - 3.0, until, 20, options, superseded_ok=True)
    assert result.alerts, "ALeRCE returned no lc_classifier SNIa objects detected in the last 3 days"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for alert in result.alerts[:5]:  # independent check of the newest version's top class
            assert alert.classification == "SNIa" and alert.extra["classifier_choice"] == "newest_version"
            answer = await client.get(f"https://api.alerce.online/ztf/v1/objects/{alert.object_id}/probabilities",
                                      params={"classifier": "lc_classifier"})
            if answer.status_code >= 500:
                pytest.skip(f"ALeRCE unreachable: HTTP {answer.status_code}")
            first = [p for p in answer.json() if p["ranking"] == 1]
            newest = max(first, key=lambda p: AlerceBroker.version_key(p["classifier_version"]))
            assert newest["class_name"] == "SNIa" and newest["classifier_version"] == alert.extra["classifier_version"]
