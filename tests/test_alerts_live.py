"""Live tests for alerts.py against the real ALeRCE / Fink brokers and archives (``pytest -m live``).

A test only skips when a service is unreachable (network error, timeout, HTTP 5xx/429:
``BrokerError.unreachable`` or a catalog failure of an UNREACHABLE_ERROR_TYPES type). Empty
answers, parse failures, enrichment exceptions and wrong astrophysics fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from astropy import units as u

from alerts import (
    FINK_ZTF_CLASS_SCORES,
    UNREACHABLE_ERROR_TYPES,
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


async def live_fetch(broker: str, since: float, until: float, limit: int, options: dict[str, Any]) -> FetchResult:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        try:
            result = await fetch_alerts(client, broker, since_mjd=since, until_mjd=until, limit=limit, options=options)
        except BrokerError as exc:
            skip_if_unreachable(exc, broker)
            raise
    assert result.warnings == [], result.warnings
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
    """A recent real window holding more than `limit` (and at most 15) objects: (since, until, ids)."""
    now = now_mjd()
    recent = await live_fetch(broker, now - 14.0, now, 100, options)
    if not recent.alerts:
        pytest.fail(f"{broker}: no alerts in the last 14 days to build a window from")
    end = max(a.first_mjd if broker == "alerce" else a.mjd for a in recent.alerts) + 0.001
    for days in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        found = await live_fetch(broker, end - days, end, 100, options)
        if limit < len(found.alerts) <= 15 and not found.truncated:
            return end - days, end, sorted(a.alert_id for a in found.alerts)
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
        assert enrichment["host_status"] in {"found", "none_within_radius", "not_applicable_star", "ambiguous_transient_entry"}
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


async def test_alerce_rows_per_classifier_version_are_deduplicated_live(tmp_path: Path) -> None:
    """ALeRCE lc_classifier AGN answers repeat objects once per classifier version: ids must be unique,
    repeated objects must take their newest version, and re-polling must change nothing."""
    until = now_mjd()
    options = {"classifier": "lc_classifier", "class_name": "AGN", "mjd_field": "lastmjd"}
    result = await live_fetch("alerce", until - 3.0, until, 15, options)
    assert result.alerts, "ALeRCE returned no lc_classifier AGN objects detected in the last 3 days"
    ids = [a.object_id for a in result.alerts]
    assert len(ids) == len(set(ids))
    for alert in result.alerts:
        check_common(alert, "alerce", until - 3.0 - 1e-6, until)
        if len(alert.extra.get("classifier_rows") or []) > 1:
            assert alert.extra["classifier_choice"] == "newest_version"
            assert alert.probability == alert.extra["classifier_versions"][alert.extra["classifier_version"]]
    store = AlertStore(MetadataStore(f"sqlite:///{(tmp_path / 'dups.sqlite3').as_posix()}"))
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        svc = AlertService(store, client, None)
        try:
            first = await svc.poll("alerce", since_mjd=until - 3.0, until_mjd=until, limit=15, crossmatch=False,
                                   options=options)
            second = await svc.poll("alerce", since_mjd=until - 3.0, until_mjd=until, limit=15, crossmatch=False,
                                    options=options)
        except BrokerError as exc:
            skip_if_unreachable(exc, "ALeRCE")
            raise
    assert len(first.alert_ids) == len(set(first.alert_ids)) == first.inserted
    # Objects detected again between the polls legitimately update; nothing else changes.
    assert second.inserted <= 1 and second.unchanged >= len(second.alert_ids) - 2

