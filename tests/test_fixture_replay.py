"""Offline replay of real archive responses recorded for canary targets.

Every catalog is queried through the real provider code; respx serves the bytes
recorded from the live archive (tests/fixtures). Assertions check that the known
object comes out with the right identifier, separation, 1-sigma error, and epoch.
"""

from __future__ import annotations

import math

import pytest
import respx
from fixture_io import TARGETS, load_exchanges, replay_side_effect
from helpers import make_service, offline_client, run_catalog

# (target, catalog) -> (source_id, separation_arcsec, sep_tolerance)
EXPECTED_HITS: dict[tuple[str, str], tuple[str, float, float]] = {
    ("3c273", "gaia_dr3"): ("3700386905605055360", 0.0021, 0.002),
    ("3c273", "simbad"): ("3C 273", 0.002, 0.002),
    ("3c273", "ned"): ("3C 273", 0.0001, 0.002),
    ("3c273", "vizier_2mass_reference"): ("12290669+0203085", 0.066, 0.01),
    ("3c273", "twomass_psc"): ("12290669+0203085", 0.066, 0.01),
    ("3c273", "allwise"): ("J122906.69+020308.6", 0.036, 0.01),
    ("3c273", "panstarrs_dr2"): ("110461872779253351", 0.0067, 0.005),
    ("3c273", "sdss"): ("1237651735760142397", 0.033, 0.01),
    ("3c273", "first"): ("FIRST J122906.7+020308", 0.64, 0.05),
    ("3c273", "nvss"): ("NVSS J122906+020305", 5.58, 0.05),
    ("3c273", "lotss"): ("ILTJ122906.32+020304.4", 7.0, 0.05),
    ("3c273", "rosat"): ("2RXS J122906.6+020309", 1.34, 0.05),
    ("3c273", "rosat_bsc"): ("1RXS J122906.5+020311", 3.86, 0.05),
    ("3c273", "chandra"): ("2CXO J122906.6+020308", 0.24, 0.02),
    ("3c273", "xmm"): ("5XMM J122906.6+020308", 0.29, 0.02),
    ("m87", "gaia_dr3"): ("3907709439453756032", 0.0158, 0.003),
    ("m87", "simbad"): ("M 87", 0.0002, 0.002),
    ("m87", "ned"): ("Messier 087", 0.0002, 0.002),
    ("m87", "twomass_psc"): ("12304942+1223278", 0.173, 0.01),
    ("m87", "allwise"): ("J123049.43+122328.0", 0.1195, 0.01),
    ("m87", "panstarrs_dr2"): ("122861877059169881", 0.082, 0.01),
    ("m87", "first"): ("FIRST J123049.3+122323", 4.69, 0.05),
    ("m87", "nvss"): ("NVSS J123049+122321", 6.47, 0.05),
    ("m87", "vlass"): ("J123049.43+122328.3", 0.18, 0.02),
    ("m87", "rosat"): ("2RXS J123049.9+122323", 8.62, 0.05),
    ("m87", "chandra"): ("2CXO J123049.4+122327", 0.26, 0.02),
    ("m87", "xmm"): ("5XMM J123049.3+122330", 2.54, 0.05),
    ("hd209458", "exoplanet_archive"): ("HD 209458 b", 0.28, 0.02),
    ("hd209458", "simbad"): ("HD 209458", 0.455, 0.02),
    ("hd209458", "gaia_dr3"): ("1779546757669063552", 0.291, 0.02),
    ("hd209458", "twomass_psc"): ("22031077+1853036", 0.387, 0.02),
    ("hd106785", "simbad"): ("HD 106785", 0.0, 0.002),
    ("hd106785", "ned"): ("HD 106785", 0.12, 0.02),
    ("hd106785", "gaia_dr3"): ("3908454259797359232", 0.028, 0.005),
    ("hd106785", "twomass_psc"): ("12164896+1232209", 0.056, 0.01),
    ("hd106785", "allwise"): ("J121648.96+123221.0", 0.12, 0.01),
}

# Valid queries that legitimately return nothing (coverage holes / not in catalog).
EXPECTED_EMPTY = [
    ("3c273", "exoplanet_archive"),
    ("3c273", "vlass"),  # too bright / excluded from VLASS QL catalogs
    ("3c273", "lotss_dr2"),  # outside DR2 footprint
    ("m87", "sdss"),  # hole in SDSS photometry around the giant galaxy
    ("m87", "lotss"),  # LoTSS-DR3 hole around Virgo A
    ("m87", "rosat_bsc"),
    ("m87", "exoplanet_archive"),
    ("hd106785", "exoplanet_archive"),
    ("hd106785", "chandra"),
]


@pytest.mark.parametrize(("target", "catalog"), sorted(EXPECTED_HITS))
async def test_known_object_recovered_from_recorded_response(target: str, catalog: str) -> None:
    sources, failure = await run_catalog(catalog, target)
    assert failure is None, failure
    assert sources, f"{catalog} returned no rows for {target}"
    assert sources.meta["status"] == "success"
    expected_id, expected_sep, tol = EXPECTED_HITS[(target, catalog)]
    nearest = sources[0]
    assert nearest.source_id == expected_id
    sep = nearest.metadata["query_separation_arcsec"]
    assert sep == pytest.approx(expected_sep, abs=tol)
    # nearest-first ordering (rows within 0.1 mas are ties, ordered deterministically)
    seps = [round(s.metadata["query_separation_arcsec"] / 1e-4) for s in sources]
    assert seps == sorted(seps)
    assert all(s.metadata["query_separation_arcsec"] <= 10.0 + 1e-6 for s in sources)  # nothing outside the cone
    assert nearest.positional_error_arcsec is not None and nearest.positional_error_arcsec >= 0
    if catalog == "ned":
        # NED positions carry the epoch of the survey they were copied from (via
        # pos_bibcode); a VLBI or literature position has no single epoch -> None.
        assert nearest.epoch is None or 1985 < nearest.epoch < 2030
    elif nearest.epoch is None:
        # CSC, NVSS, LoTSS, 1RXS: no per-source date, but a known observing span.
        lo, hi = nearest.epoch_range
        assert 1985 < lo <= hi < 2030
    else:
        assert 1985 < nearest.epoch < 2030 and nearest.epoch_range is None
    assert nearest.metadata["citation"]


@pytest.mark.parametrize(("target", "catalog"), EXPECTED_EMPTY)
async def test_legitimately_empty_results_are_empty_not_failed(target: str, catalog: str) -> None:
    sources, failure = await run_catalog(catalog, target)
    assert failure is None
    assert list(sources) == []
    assert sources.meta["status"] == "empty"
    assert sources.meta["row_count"] == 0


async def test_gaia_errors_epoch_and_proper_motion() -> None:
    sources, _ = await run_catalog("gaia_dr3", "3c273")
    src = sources[0]
    raw = src.data
    assert raw["ra_error"] == pytest.approx(0.0181, abs=5e-4)  # mas in the archive
    expected = math.sqrt((raw["ra_error"] ** 2 + raw["dec_error"] ** 2) / 2) / 1000.0
    assert src.positional_error_arcsec == pytest.approx(expected)
    assert src.metadata["positional_error"]["sigma_ra_arcsec"] == pytest.approx(raw["ra_error"] / 1000)
    assert src.epoch == 2016.0
    assert src.proper_motion_ra_masyr == pytest.approx(raw["pmra"])
    assert src.source_id == str(raw["source_id"]) == "3700386905605055360"


async def test_simbad_errors_are_mas_ellipse_equal_to_gaia() -> None:
    simbad, _ = await run_catalog("simbad", "3c273")
    gaia, _ = await run_catalog("gaia_dr3", "3c273")
    # SIMBAD copies Gaia EDR3's J2016.0 errors for its J2000.0 (propagated) position; the
    # 16 yr of proper-motion error is added: pm_error ~ 1.4 x position error per yr (Gaia
    # DR3 upper decile), i.e. 0.0181 mas x 1.4 x 16 = 0.41 mas >> the 0.018 mas at J2016.
    details = simbad[0].metadata["positional_error"]
    growth = details["epoch_growth"]
    assert growth["source_epoch"] == 2016.0 and growth["to_epoch"] == 2000.0
    major = simbad[0].data["coo_err_maj"] / 1000.0
    assert growth["term_arcsec"] == pytest.approx(major * 1.4 * 16.0, rel=1e-9)
    assert simbad[0].positional_error_arcsec == pytest.approx(
        math.hypot(gaia[0].positional_error_arcsec, growth["term_arcsec"]), rel=2e-2)
    assert simbad[0].positional_error_arcsec > 20 * gaia[0].positional_error_arcsec
    assert simbad[0].epoch == 2000.0
    assert simbad[0].metadata["physical"]["object_type"] == "BLL"
    assert simbad[0].metadata["physical"]["redshift"] == pytest.approx(0.1576, abs=1e-3)


async def test_ned_95pct_ellipse_converted_to_sigma() -> None:
    sources, _ = await run_catalog("ned", "3c273")
    src = sources[0]
    assert src.metadata["physical"]["object_type"] == "QSO"
    assert src.metadata["physical"]["redshift"] == pytest.approx(0.158339, abs=1e-5)
    assert src.epoch is None  # 1995AJ....110..880J (VLBI) has no survey epoch


@pytest.mark.parametrize(("target", "allwise_sigra", "allwise_sigdec"), [
    ("hd209458", 0.0349, 0.0341),  # AllWISE J220310.79+185303.3
    ("hd106785", 0.0338, 0.0322),  # AllWISE J121648.96+123221.0
])
async def test_ned_sigma_equals_the_source_catalogs_own_sigma(target, allwise_sigra, allwise_sigdec) -> None:
    # Independent check: NED copies AllWISE positions for these stars (pos_bibcode
    # 2013wise.rept....1C). Its 1-sigma error must equal AllWISE's documented 1-sigma
    # sigra/sigdec (NED stores 2.5x those values; the old /2.4477 was 2% off).
    ned, _ = await run_catalog("ned", target)
    row = next(s for s in ned if s.data["pos_bibcode"] == "2013wise.rept....1C")
    expected = math.sqrt((allwise_sigra**2 + allwise_sigdec**2) / 2)
    assert row.positional_error_arcsec == pytest.approx(expected, rel=1e-6)
    allwise, _ = await run_catalog("allwise", target)
    assert row.positional_error_arcsec == pytest.approx(allwise[0].positional_error_arcsec, rel=1e-6)
    # AllWISE via pos_bibcode: observed once at an unrecorded date in 2010.0-2011.2 (not 2000.0,
    # and not a guessed mid-survey 2010.5).
    assert row.epoch is None and row.epoch_range == (2010.0, 2011.2)
    assert row.metadata["positional_error"]["units"][2] == "deg"  # uncposa is an angle


async def test_ned_epoch_is_unknown_for_unmapped_bibcodes() -> None:
    ned, _ = await run_catalog("ned", "m87")
    by_bib = {s.data["pos_bibcode"]: s.epoch for s in ned}
    assert by_bib["2009ApJ...703...42P"] is None
    assert all(e is None or 1985 < e < 2030 for e in by_bib.values())


async def test_twomass_ellipse_and_julian_date_epoch() -> None:
    sources, _ = await run_catalog("twomass_psc", "3c273")
    src = sources[0]
    assert src.positional_error_arcsec == pytest.approx(math.sqrt((0.17**2 + 0.08**2) / 2), abs=1e-6)
    assert src.epoch == pytest.approx(2000.150, abs=1e-3)  # jdate 2451599.855


async def test_chandra_nvss_xmm_rosat_error_conversions() -> None:
    # Expected 1-sigma values are written from each producer's documentation for the
    # recorded 3C 273 rows -- not from the implementation's constants.
    chandra, _ = await run_catalog("chandra", "3c273")
    assert chandra[0].data["error_ellipse_r0"] == pytest.approx(0.6478, abs=1e-4)
    # CXC: r0/r1 are per-axis 95% intervals (1.96 sigma), systematic already included.
    assert chandra[0].positional_error_arcsec == pytest.approx(0.6478 / 1.96, abs=2e-4)  # 0.3305"
    assert chandra[0].metadata["positional_error"]["units"] == ["arcsec", "arcsec", "deg"]
    # CSC master sources have no per-source date: the epoch is the CSC observing span.
    assert chandra[0].epoch is None and chandra[0].epoch_range == (1999.5, 2022.0)

    nvss, _ = await run_catalog("nvss", "3c273")
    # ra_error 0.03 s of time at Dec +2.0514 -> 0.03*15*cos(Dec) = 0.44971"; dec_error 0.6".
    assert nvss[0].metadata["positional_error"]["sigma_ra_arcsec"] == pytest.approx(0.44971, abs=1e-4)
    assert nvss[0].positional_error_arcsec == pytest.approx(math.sqrt((0.44971**2 + 0.6**2) / 2), abs=1e-4)

    assert nvss[0].epoch is None and nvss[0].epoch_range == (1993.7, 1997.3)

    xmm, _ = await run_catalog("xmm", "3c273")
    # Statistical error_radius is negligible for this 4e8-likelihood source: the
    # 0.88" 5XMM systematic dominates.
    assert xmm[0].positional_error_arcsec == pytest.approx(0.88, abs=1e-4)
    # time/end_time are the first (MJD 51708.38, 2000.42) and last (MJD 60317.42, 2024.0)
    # observation of the stack: the position was measured at an unknown time in that span,
    # so the epoch is the span (closest approach for movers) -- not its midpoint 2012.23.
    assert xmm[0].epoch is None
    assert xmm[0].epoch_range == pytest.approx((2000.0 + (51708.38 - 51544.5) / 365.25,
                                                2000.0 + (60317.42 - 51544.5) / 365.25), abs=1e-3)

    rosat, _ = await run_catalog("rosat", "3c273")
    # 2RXS pixel errors 0.015277/0.015316 pix x 45"/pix (rms 0.6883"); Freund et al. 2022
    # (A&A 664, A105) Eq. 2: sigma = 1.22 sqrt(0.6883^2 + 3^2) = 3.755" (not hypot(0.69, 5) = 5.05").
    assert rosat[0].positional_error_arcsec == pytest.approx(1.22 * math.hypot(0.68834, 3.0), abs=1e-3)
    assert rosat[0].positional_error_arcsec == pytest.approx(3.755, abs=2e-3)
    assert rosat[0].epoch == pytest.approx(1991.0, abs=0.1)


async def test_sdss_lotss_first_error_conversions() -> None:
    # Pins the per-catalog systematics/formulas against the recorded rows.
    sdss, _ = await run_catalog("sdss", "3c273")
    # raErr/decErr arcsec (0.05562/0.06821) (+) the 0.04" SDSS astrometric systematic.
    rms = math.sqrt((0.0556168269075737**2 + 0.0682123259099677**2) / 2)
    assert sdss[0].positional_error_arcsec == pytest.approx(math.hypot(rms, 0.04), abs=1e-6)
    assert sdss[0].positional_error_arcsec == pytest.approx(0.07398, abs=1e-4)
    assert sdss[0].epoch == pytest.approx(2000.0 + (51668 - 51544.5) / 365.25, abs=1e-6)  # MJD 51668

    lotss, _ = await run_catalog("lotss", "3c273")
    row = lotss[0]
    rms = math.sqrt((row.data["e_RAJ2000"] ** 2 + row.data["e_DEJ2000"] ** 2) / 2)
    assert row.positional_error_arcsec == pytest.approx(math.hypot(rms, 0.2), abs=1e-6)  # 0.2" LoTSS systematic
    assert row.positional_error_arcsec > rms
    assert row.epoch is None and row.epoch_range == (2014.3, 2024.5)

    first, _ = await run_catalog("first", "3c273")
    # FIRST (White et al. 1997): 90% per-axis error = size * (1/SNR + 1/20), SNR = (peak - 0.25)/rms;
    # 6.5" x 5.53", peak 35558.44 mJy, rms 6.196 mJy -> 1-sigma = eps90 / 1.645.
    snr = (35558.44 - 0.25) / 6.196
    eps = [axis * (1 / snr + 1 / 20) / 1.6448536269514722 for axis in (6.5, 5.53)]
    assert first[0].positional_error_arcsec == pytest.approx(math.sqrt((eps[0] ** 2 + eps[1] ** 2) / 2), abs=1e-6)
    assert first[0].positional_error_arcsec == pytest.approx(0.18408, abs=1e-4)
    assert first[0].epoch == pytest.approx(2000.0 + (51039.2 - 51544.5) / 365.25, abs=1e-4)


async def test_null_sentinels_are_removed_from_source_data() -> None:
    # PS1 writes -999 for "no measurement": it must be None in the exported data too,
    # so a 'brighter than 12 mag' dataset filter does not accept a missing magnitude.
    from datasets import DatasetEngine

    sources, _ = await run_catalog("panstarrs_dr2", "m87")
    nucleus = next(s for s in sources if s.source_id == "122861877059169881")
    assert nucleus.data["iMeanKronMag"] is None and nucleus.data["gMeanPSFMagErr"] is None
    for src in sources:
        assert all(v not in (-999, -999.0, "NULL") for v in src.data.values() if not isinstance(v, bool))
    flat = {"catalog": "panstarrs_dr2", "source_id": nucleus.source_id, "data": nucleus.data, "physical": {}}
    assert DatasetEngine._passes_filter(flat, "max_iMeanKronMag", 12) is False


async def test_vlass_degree_errors_and_jd_epoch() -> None:
    sources, _ = await run_catalog("vlass", "m87")
    src = sources[0]
    e = math.sqrt((src.data["e_RAJ2000"] ** 2 + src.data["e_DEJ2000"] ** 2) / 2) * 3600.0
    assert src.positional_error_arcsec == pytest.approx(math.hypot(e, 0.5))
    assert src.epoch == pytest.approx(2019.29, abs=0.01)  # observed 2019-04-17


async def test_panstarrs_filters_and_sentinels() -> None:
    sources, _ = await run_catalog("panstarrs_dr2", "3c273")
    assert len(sources) == 3  # nDetections > 1 removes the artefacts around the QSO
    assert all(s.data["nDetections"] > 1 for s in sources)
    src = sources[0]
    assert src.epoch == pytest.approx(2011.743, abs=1e-3)
    assert src.positional_error_arcsec == pytest.approx(math.hypot(
        math.sqrt((src.data["raMeanErr"] ** 2 + src.data["decMeanErr"] ** 2) / 2), 0.015))


async def test_exoplanet_archive_epoch_and_default_error() -> None:
    sources, _ = await run_catalog("exoplanet_archive", "hd209458")
    src = sources[0]
    assert src.epoch == 2015.5
    assert src.positional_error_arcsec == 0.1
    assert src.metadata["positional_error"]["source"] == "default"
    assert src.data["hostname"] == "HD 209458"
    assert src.proper_motion_ra_masyr is not None  # sy_pmra alias


async def test_full_crossmatch_replay_3c273_status_and_provenance() -> None:
    exchanges = load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            record = await service.crossmatch(*TARGETS["3c273"], radius_arcsec=10.0)

    assert record.failures == []
    assert record.catalogs_queried == 15
    statuses = {name: res["status"] for name, res in record.catalog_results.items()}
    assert statuses.pop("exoplanet_archive") == "empty"
    assert statuses.pop("vlass") == "empty"
    assert set(statuses.values()) == {"success"}
    for res in record.catalog_results.values():
        assert res["row_count"] == len(res["sources"])
        assert isinstance(res["elapsed_ms"], float)
        assert res["citation"]
    stats = record.provenance["catalog_stats"]
    assert stats["gaia_dr3"]["row_count"] == 1 and stats["ned"]["row_count"] == 21
    assert set(record.provenance["citations"]) == set(record.catalog_results)
    matched = {m["catalog"] for m in record.provenance["matches"] if m["separation_arcsec"] < 1.0}
    assert {"gaia_dr3", "simbad", "ned", "twomass_psc", "allwise", "panstarrs_dr2", "sdss", "chandra", "xmm", "first"} <= matched


def test_api_search_endpoint_serializes_replayed_results() -> None:
    from fastapi.testclient import TestClient

    from api import app

    exchanges = load_exchanges("3c273")
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=replay_side_effect(exchanges))
        with TestClient(app) as client:
            response = client.post("/api/v1/search", json={
                "ra": TARGETS["3c273"][0], "dec": TARGETS["3c273"][1], "radius_arcsec": 10.0,
                "catalogs": ["gaia_dr3", "ned", "vlass", "chandra"], "min_confidence": 0.0,
            })
    assert response.status_code == 200, response.text
    body = response.json()
    statuses = {name: res["status"] for name, res in body["catalog_results"].items()}
    assert statuses == {"gaia_dr3": "success", "ned": "success", "vlass": "empty", "chandra": "success"}
    gaia = body["catalog_results"]["gaia_dr3"]["sources"][0]
    assert gaia["source_id"] == "3700386905605055360"
    assert body["provenance"]["catalog_stats"]["vlass"]["row_count"] == 0
