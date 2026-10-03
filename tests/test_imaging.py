"""Imaging: hips2fits cutouts, survey catalogue, disk cache, stacks, router and CLI.

Offline tests replay real hips2fits / MocServer exchanges recorded by
``tests/fixtures/imaging/record_imaging.py`` (strict request matching through
``fixture_io.replay_side_effect``). Live tests (``-m live``) hit CDS and assert
astrophysical truth for 3C 273, M87 and survey footprints.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixture_io import FIXTURES, load_exchanges, replay_side_effect
from helpers import offline_client

import imaging
from imaging import (
    SURVEYS,
    CutoutCache,
    CutoutRequest,
    CutoutService,
    CutoutUpstreamError,
    CutoutValidationError,
    UnknownSurveyError,
    coverage_fraction,
    get_survey,
    list_surveys,
    probe_image,
)

IMG = FIXTURES / "imaging"
C3C273 = (187.2779154, 2.0523883)
M87 = (187.7059308, 12.3911233)
SOUTH = (100.0, -70.0)
SESAME_3C273 = (FIXTURES / "sesame" / "3c273.xml").read_text(encoding="utf-8")


def body(name: str, idx: int = 0) -> bytes:
    return (IMG / f"{name}.{idx}.body").read_bytes()


def replay_router(*scenarios: str, passthrough_testserver: bool = False, sesame: str = SESAME_3C273) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    if passthrough_testserver:
        router.route(host="testserver").pass_through()
    router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").respond(
        200, text=sesame, headers={"content-type": "text/xml"})
    router.route().mock(side_effect=replay_side_effect(load_exchanges("imaging", list(scenarios))))
    return router


@pytest.fixture
def cache(tmp_path: Path) -> CutoutCache:
    return CutoutCache(tmp_path / "cutouts", ttl_seconds=3600, max_bytes=10 * 1024 * 1024)


def spec(name: str) -> CutoutRequest:
    specs = {
        "dss2_3c273_png": {"ra": C3C273[0], "dec": C3C273[1], "survey": "dss2", "fov_arcmin": 2.0, "width": 128, "height": 128, "format": "png"},
        "nvss_3c273_fits": {"ra": C3C273[0], "dec": C3C273[1], "survey": "nvss", "fov_arcmin": 6.0, "width": 64, "height": 64, "format": "fits"},
        "panstarrs_m87_jpg": {"ra": M87[0], "dec": M87[1], "survey": "panstarrs", "fov_arcmin": 3.0, "width": 200, "height": 100,
                                  "format": "jpg"},
        "sdss_south_png": {"ra": SOUTH[0], "dec": SOUTH[1], "survey": "sdss", "fov_arcmin": 6.0, "width": 64, "height": 64, "format": "png"},
        "unknown_hips": {"ra": C3C273[0], "dec": C3C273[1], "survey": "BOGUS/P/nothing", "fov_arcmin": 2.0, "width": 64, "height": 64,
                             "format": "png"},
        "vlass_3c273_png": {"ra": C3C273[0], "dec": C3C273[1], "survey": "vlass", "fov_arcmin": 2.0, "width": 64, "height": 64,
                                "format": "png"},
        "vlass_crab_png": {"ra": 83.63308, "dec": 22.0145, "survey": "vlass", "fov_arcmin": 2.0, "width": 64, "height": 64,
                               "format": "png"},
    }
    return CutoutRequest(**specs[name])


# ---------------------------------------------------------------------------
# Survey catalogue
# ---------------------------------------------------------------------------


def test_survey_catalogue_is_consistent() -> None:
    ids = [s.hips_id for s in SURVEYS.values()]
    assert len(ids) == len(set(ids))
    for key, survey in SURVEYS.items():
        assert key == survey.key
        assert imaging._HIPS_ID_RE.match(survey.hips_id), survey.hips_id
        assert 0 < survey.em_min_m <= survey.em_max_m, key
        assert survey.regime in imaging.REGIME_ORDER
        assert survey.coverage in ("full", "moc"), key  # every catalogued survey has a usable MOC
        # A colour composite names a single-band companion (FITS survey pixel values) in the same regime.
        if survey.color and survey.science is not None:
            science = SURVEYS[survey.science]
            assert not science.color and science.science is None, key
            assert science.regime == survey.regime, key
            # Within the composite's band (20 % slack: the registry gives DSS2 colour 400-600 nm
            # but DSS2 red 640-658 nm).
            assert survey.em_min_m / 1.2 <= science.wavelength_m <= survey.em_max_m * 1.2, key
        if not survey.color:
            assert survey.science is None and imaging.science_survey(survey) is survey
        # Alternates are further single-band bands of the same composite.
        for alt in survey.science_alternates:
            assert survey.color and not SURVEYS[alt].color and SURVEYS[alt].regime == survey.regime, (key, alt)
        # Every single-band survey says what its FITS pixels are; a colour composite does not.
        if survey.color:
            assert survey.pixel_units is None and not survey.calibrated, key
        else:
            assert survey.pixel_units and survey.pixel_note, key
        if survey.epoch_span is not None:
            # DSS2 blue (SERC-J) plates, part of the DSS2 colour composite, start in 1975.
            assert 1970 < survey.epoch_span[0] <= survey.epoch_span[1] < 2026, key
    # Only the Chandra RGB (PNG tiles only) lacks a single-band FITS companion.
    assert [k for k, s in SURVEYS.items() if s.color and s.science is None] == ["chandra"]
    assert {SURVEYS[k].science for k in ("2mass", "allwise", "panstarrs", "xmm")} == {
        "2mass_k", "allwise_w1", "panstarrs_g", "xmm_eb2"}
    listed = list_surveys()
    waves = [s["wavelength_m"] for s in listed]
    assert waves == sorted(waves, reverse=True), "surveys must be listed radio -> X-ray"
    assert listed[0]["regime"] == "radio" and listed[-1]["regime"] == "xray"
    # Every regime the UI stacks is represented.
    assert {s.regime for s in SURVEYS.values()} == set(imaging.REGIME_ORDER)


def test_wavelength_labels_follow_physics() -> None:
    # NVSS em = 0.21413747 m -> c / lambda = 1.40 GHz.
    assert SURVEYS["nvss"].wavelength_label == "1.4 GHz"
    assert SURVEYS["nvss"].frequency_hz == pytest.approx(1.4e9, rel=1e-4)
    # TGSS ADR: the "150 MHz" survey is centred on 147.5 MHz (registry em 2.03143-2.03363 m;
    # Intema et al. 2017); the nominal name stays in the label only.
    assert SURVEYS["tgss"].wavelength_label == "147.5 MHz"
    assert SURVEYS["tgss"].frequency_hz == pytest.approx(147.5e6, rel=2e-4)
    assert SURVEYS["tgss"].label == "TGSS ADR 150 MHz"
    # RASS em 5.166e-10..1.2398e-8 m <-> 0.1..2.4 keV (E = hc / lambda).
    assert SURVEYS["rass"].wavelength_label == "0.1–2.4 keV"
    assert SURVEYS["erosita"].wavelength_label == "0.2–2.3 keV"
    assert SURVEYS["2mass"].wavelength_label == "1.15–2.3 µm"
    assert SURVEYS["galex"].wavelength_label == "134–283 nm"
    assert SURVEYS["dss2"].as_dict()["bib_url"] == "https://ui.adsabs.harvard.edu/abs/1996ASPC..101...88L/abstract"
    # The unit follows the upper limit: 0.39-1.02 um, never "390-1.02e+03 nm".
    assert SURVEYS["legacy"].wavelength_label == "0.39–1.02 µm"
    assert all("e+" not in s["wavelength"] and "e-" not in s["wavelength"] for s in list_surveys())
    # XMM PN band 2 is 0.5-1 keV (registry em limits are swapped; E = hc / lambda).
    assert SURVEYS["xmm_eb2"].wavelength_label == "0.5–1 keV"


def test_adhoc_survey_without_wavelength_is_json_safe() -> None:
    adhoc = get_survey("CDS/P/Mellinger/color")
    assert adhoc.wavelength_label == "unknown" and not adhoc.has_wavelength
    data = adhoc.as_dict()
    assert data["wavelength"] == "unknown"
    assert data["wavelength_m"] is None and data["frequency_hz"] is None and data["em_min_m"] is None
    json.dumps(data, allow_nan=False)  # NaN would make the API answer invalid JSON


def test_regime_for_wavelength() -> None:
    assert imaging.regime_for_wavelength(0.21) == "radio"
    assert imaging.regime_for_wavelength(2.2e-6) == "infrared"
    assert imaging.regime_for_wavelength(5.5e-7) == "optical"
    assert imaging.regime_for_wavelength(2.3e-7) == "uv"
    assert imaging.regime_for_wavelength(1.24e-9) == "xray"  # 1 keV
    assert imaging.regime_for_wavelength(1.24e-12) == "gamma"  # 1 MeV
    assert imaging.regime_for_wavelength(math.nan) == "unknown"

def test_get_survey_by_key_id_and_adhoc() -> None:
    assert get_survey("DSS2").hips_id == "CDS/P/DSS2/color"
    assert get_survey("cds/p/nvss").key == "nvss"
    adhoc = get_survey("CDS/P/Mellinger/color")
    assert adhoc.hips_id == "CDS/P/Mellinger/color" and adhoc.regime == "unknown"
    with pytest.raises(UnknownSurveyError, match="FIRST is not published as a HiPS"):
        get_survey("first")
    with pytest.raises(UnknownSurveyError, match="Unknown survey"):
        get_survey("not a survey")
    with pytest.raises(UnknownSurveyError):
        get_survey("")


def test_local_coverage_rules() -> None:
    # VLASS's HiPS does not fill the survey's dec > -40 footprint: its MOC decides (no local shortcut).
    assert SURVEYS["vlass"].coverage == "moc" and SURVEYS["vlass"].covered(2.05) is None
    assert SURVEYS["2mass"].covered(-89.0) is True
    assert SURVEYS["nvss"].covered(0.0) is None  # needs the MocServer
    limited = imaging.HipsSurvey("x", "X/P/y", "x", "radio", 0.1, 0.1, None, False, "dec>-40")
    assert limited.covered(2.05) is True and limited.covered(-45.0) is False


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("changes, message", [
    ({"ra": 360.0}, "ra must be"),
    ({"ra": -0.1}, "ra must be"),
    ({"dec": 90.5}, "dec must be"),
    ({"ra": math.nan}, "finite"),
    ({"fov_arcmin": 0.0}, "fov_arcmin"),
    ({"fov_arcmin": 21601.0, "projection": "MOL"}, r"fov_arcmin must be in \[0.05, 21600\]"),
    ({"fov_arcmin": 10800.0}, "< 10800 for TAN"),
    ({"fov_arcmin": 10801.0, "projection": "SIN"}, "<= 10800 for SIN"),
    ({"fov_arcmin": 21600.0, "projection": "STG"}, "< 21600 for STG"),
    ({"min_cut": "200%"}, r"min_cut percentile must be in \[0, 100\]"),
    ({"max_cut": "-1%"}, "max_cut percentile"),
    ({"min_cut": "5", "max_cut": "1"}, "min_cut must not exceed max_cut"),
    ({"min_cut": "99%", "max_cut": "1%"}, "min_cut must not exceed max_cut"),
    # A lone percentile is compared with hips2fits' default for the other cut (0.5 % / 99.5 %):
    # these gave HTTP 500 live (NVSS at the Crab).
    ({"min_cut": "99.6%"}, r"min_cut must be at most the hips2fits default max_cut \(default 99.5%\)"),
    ({"min_cut": "100%"}, "min_cut must be at most"),
    ({"min_cut": "1E2%"}, "min_cut must be at most"),
    ({"max_cut": "0.4%"}, r"max_cut must be at least the hips2fits default min_cut \(default 0.5%\)"),
    ({"width": 4}, "width"),
    ({"height": 5000}, "height"),
    ({"width": True}, "width"),
    ({"width": 12.5}, "width"),
    ({"format": "bmp"}, "format"),
    ({"projection": "XYZ"}, "projection"),
    ({"stretch": "cubic"}, "stretch"),
    ({"cmap": "viridis;rm"}, "cmap"),
    ({"min_cut": "abc"}, "min_cut"),
    ({"format": "fits", "stretch": "asinh"}, "only to png/jpg"),
    ({"coordsys": "fk4"}, "coordsys"),
    ({"survey": "first"}, "FIRST"),
])
def test_cutout_request_validation(changes: dict, message: str) -> None:
    base = {"ra": 10.0, "dec": 10.0, "fov_arcmin": 5.0, "survey": "dss2"}
    with pytest.raises(CutoutValidationError, match=message):
        CutoutRequest(**{**base, **changes})


@pytest.mark.parametrize("changes", [
    # hips2fits renders these (verified live): fov is the full width, so the whole sky is 360 deg.
    {"projection": "MOL", "fov_arcmin": 21600.0},
    {"projection": "AIT", "fov_arcmin": 21600.0},
    {"projection": "CAR", "fov_arcmin": 21600.0},
    {"projection": "ZEA", "fov_arcmin": 21600.0},
    {"projection": "ARC", "fov_arcmin": 21600.0},
    {"projection": "TAN", "fov_arcmin": 7200.0},
    {"projection": "SIN", "fov_arcmin": 10800.0},
    {"projection": "STG", "fov_arcmin": 18000.0},
    {"min_cut": "0.5%", "max_cut": "99.5%"},
    {"min_cut": "0%", "max_cut": "100%"},
    {"min_cut": "-3.5", "max_cut": "1e3"},
    {"min_cut": "50%", "max_cut": "10"},  # different kinds cannot be compared
    # Rendered by hips2fits (HTTP 200, verified live): equal cuts and lone cuts up to the other default.
    {"min_cut": "3", "max_cut": "3"},
    {"min_cut": "5%", "max_cut": "5%"},
    {"min_cut": "99.5%"},
    {"max_cut": "0.5%"},
])
def test_cutout_request_accepts_valid_wide_fields_and_cuts(changes: dict) -> None:
    request = CutoutRequest(**{"ra": 10.0, "dec": 10.0, "survey": "dss2", **changes})
    params = request.params()
    assert float(params["fov"]) * 60 == pytest.approx(request.fov_arcmin)


def test_cutout_request_params_use_documented_hips2fits_names() -> None:
    req = CutoutRequest(ra=187.2779154, dec=2.0523883, fov_arcmin=6.0, survey="nvss", width=300, height=200,
                        format="JPEG", projection="tan", stretch="asinh", cmap="viridis", min_cut="0.5%", max_cut="99.5%",
                        rotation_angle=30.0)
    assert req.format == "jpg" and req.projection == "TAN"
    assert req.params() == {
        "hips": "CDS/P/NVSS", "width": "300", "height": "200", "fov": "0.1", "projection": "TAN",
        "ra": "187.2779154", "dec": "2.0523883", "format": "jpg", "rotation_angle": "30.0",
        "stretch": "asinh", "cmap": "viridis", "min_cut": "0.5%", "max_cut": "99.5%",
    }
    assert req.pixel_scale_arcsec == pytest.approx(6.0 * 60 / 300)
    assert req.media_type == "image/jpeg"
    assert req.filename() == "cutout_nvss_187.27792_+2.05239_6arcmin.jpg"
    assert len(req.cache_key()) == 64
    assert req.cache_key() != CutoutRequest(ra=187.2779154, dec=2.0523883, fov_arcmin=6.0, survey="nvss").cache_key()


# ---------------------------------------------------------------------------
# Header probing & coverage (recorded bytes)
# ---------------------------------------------------------------------------


def test_probe_image_reads_real_headers() -> None:
    assert probe_image(body("dss2_3c273_png")) == imaging.ImageInfo("png", 128, 128, 4)
    jpg = probe_image(body("panstarrs_m87_jpg"))
    assert (jpg.format, jpg.width, jpg.height) == ("jpg", 200, 100)
    fits_info = probe_image(body("nvss_3c273_fits"))
    assert (fits_info.format, fits_info.width, fits_info.height, fits_info.planes) == ("fits", 64, 64, 1)
    with pytest.raises(CutoutUpstreamError):
        probe_image(b'{"title": "500 Internal Server Error"}')
    with pytest.raises(CutoutUpstreamError):
        probe_image(b"\xff\xd8\xff\xd9")


def test_coverage_fraction_detects_blank_hips2fits_output() -> None:
    assert coverage_fraction(body("dss2_3c273_png"), "png") == 1.0
    # SDSS never observed (100, -70): hips2fits answers a fully transparent image.
    assert coverage_fraction(body("sdss_south_png"), "png") == 0.0
    assert coverage_fraction(body("nvss_3c273_fits"), "fits") == 1.0
    assert coverage_fraction(body("panstarrs_m87_jpg"), "jpg") is None


def test_recorded_nvss_cutout_shows_3c273_radio_core() -> None:
    """3C 273 is among the brightest 1.4 GHz sources: tens of Jy/beam at its optical position."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(io.BytesIO(body("nvss_3c273_fits"))) as hdul:
        data, header = hdul[0].data, hdul[0].header
    peak = np.unravel_index(np.nanargmax(data), data.shape)
    assert data[peak] > 20.0  # Jy/beam; NVSS catalogues ~55 Jy integrated for 3C 273
    position = WCS(header).pixel_to_world(peak[1], peak[0])
    assert position.separation(SkyCoord(*C3C273, unit="deg")).arcsec < 15.0


# ---------------------------------------------------------------------------
# Service: fetch, mirrors, errors, cache
# ---------------------------------------------------------------------------


async def test_cutout_fetch_then_cache_hit(cache: CutoutCache) -> None:
    with replay_router("dss2_3c273_png") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            first = await service.cutout(spec("dss2_3c273_png"))
            second = await service.cutout(spec("dss2_3c273_png"))
        calls = router.calls.call_count
    assert calls == 1, "the second request must be served from the disk cache"
    assert first.content == body("dss2_3c273_png") == second.content
    assert (first.width, first.height, first.media_type) == (128, 128, "image/png")
    assert not first.cached and second.cached
    assert first.endpoint == imaging.HIPS2FITS_URLS[0]
    assert "hips=CDS%2FP%2FDSS2%2Fcolor" in first.source_url
    headers = second.headers()
    assert headers["X-Cutout-Cache"] == "hit" and headers["X-Cutout-Survey"] == "dss2"
    assert headers["X-Cutout-Pixel-Scale-Arcsec"] == "0.9375"
    assert first.coverage_fraction == 1.0 and headers["X-Cutout-Coverage"] == "1.0000"
    assert headers["X-Cutout-Pixels"] == "display" and headers["Cache-Control"] == "public, max-age=86400"
    assert cache.size_bytes() == len(first.content)


async def test_fits_and_jpeg_cutouts(cache: CutoutCache) -> None:
    with replay_router("nvss_3c273_fits", "panstarrs_m87_jpg"):
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            fits_cut = await service.cutout(spec("nvss_3c273_fits"))
            jpg_cut = await service.cutout(spec("panstarrs_m87_jpg"))
    assert fits_cut.media_type == "application/fits" and fits_cut.coverage_fraction == 1.0
    assert (jpg_cut.width, jpg_cut.height, jpg_cut.media_type) == (200, 100, "image/jpeg")
    assert jpg_cut.coverage_fraction is None


async def test_mirror_blank_after_primary_5xx_is_degraded_and_not_cached(cache: CutoutCache) -> None:
    """Recorded at the Crab: VLASS gave HTTP 500 on alasky and a fully transparent image on
    alaskybis, while the VLASS MOC overlaps the field.

    That blank image is a rendering failure, not a statement about the footprint: it is
    flagged, served with no-store, and never cached (so a recovered primary is used next time).
    """
    with replay_router("vlass_crab_png") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(spec("vlass_crab_png"))
            again = await service.cutout(spec("vlass_crab_png"))
        hosts = [(call.request.url.host, call.request.url.path.split("/")[1]) for call in router.calls]
    assert hosts == [("alasky.cds.unistra.fr", "hips-image-services"), ("alaskybis.cds.unistra.fr", "hips-image-services"),
                     ("alasky.cds.unistra.fr", "MocServer")] * 2, "the blank answer must not be cached"
    assert cut.endpoint == imaging.HIPS2FITS_URLS[1]
    assert cut.coverage_fraction == 0.0 and not again.cached
    assert cut.degraded and "alaskybis.cds.unistra.fr" in cut.degraded and "HTTP 500" in cut.degraded
    headers = cut.headers()
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Cutout-Degraded"].startswith("blank image from mirror alaskybis.cds.unistra.fr after ")
    assert "\n" not in headers["X-Cutout-Degraded"]
    assert cache.size_bytes() == 0


async def test_mirror_blank_where_the_moc_has_no_data_is_a_real_no_data(cache: CutoutCache) -> None:
    """Recorded at 3C 273: the same alasky HTTP 500 + blank alaskybis PNG, but the VLASS Quick Look
    MOC does not reach the 2' field at all, so the blank image is the true answer (cached)."""
    with replay_router("vlass_3c273_png") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(spec("vlass_3c273_png"))
            again = await service.cutout(spec("vlass_3c273_png"))
        moc = router.calls[-1].request.url.params
        calls = router.calls.call_count
    assert calls == 3 and again.cached
    # The footprint check covers the whole field: the circle circumscribing the 2' x 2' image.
    assert float(moc["SR"]) == pytest.approx(2.0 / 120.0 * math.sqrt(2.0))
    assert cut.coverage_fraction == 0.0 and cut.degraded is None and cut.cacheable
    assert cut.headers()["Cache-Control"] == "public, max-age=86400"


async def test_blank_image_from_primary_is_cached_as_no_data(cache: CutoutCache) -> None:
    """Outside the SDSS footprint the primary answers a blank image and the SDSS MOC confirms
    the position is not covered (recorded MocServer answer): a real 'no data', cached."""
    with replay_router("sdss_south_png") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(spec("sdss_south_png"))
            again = await service.cutout(spec("sdss_south_png"))
        hosts = [(call.request.url.host, call.request.url.path) for call in router.calls]
        moc = router.calls[1].request.url.params
    assert hosts == [("alasky.cds.unistra.fr", "/hips-image-services/hips2fits"),
                     ("alasky.cds.unistra.fr", "/MocServer/query")], "one image + one footprint check, then the cache"
    assert moc["expr"] == "ID=CDS/P/SDSS9/color" and moc["RA"] == "100.0" and moc["DEC"] == "-70.0"
    assert float(moc["SR"]) == pytest.approx(6.0 / 120.0 * math.sqrt(2.0))  # the whole 6' field
    assert cut.coverage_fraction == 0.0 and cut.degraded is None and cut.cacheable
    assert again.cached and again.coverage_fraction == 0.0
    assert "X-Cutout-Degraded" not in cut.headers() and cut.headers()["Cache-Control"] == "public, max-age=86400"
    assert cache.size_bytes() == len(cut.content)


async def test_blank_image_where_the_moc_has_data_is_degraded_not_cached(cache: CutoutCache) -> None:
    """Recorded with alaskybis only (HIPS2FITS_URLS override): VLASS at the Crab comes back fully
    transparent although the VLASS MOC overlaps the field. That is a rendering failure, not 'no data'."""
    request = CutoutRequest(ra=83.63308, dec=22.0145, survey="vlass", fov_arcmin=2.0, width=64, height=64)
    with replay_router("vlass_crab_png_bis") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache, endpoints=[imaging.HIPS2FITS_URLS[1]])
            cut = await service.cutout(request)
            again = await service.cutout(request)
        paths = [call.request.url.path for call in router.calls]
    assert paths == ["/hips-image-services/hips2fits", "/MocServer/query"] * 2, "the blank image must not be cached"
    assert cut.coverage_fraction == 0.0 and not again.cached
    assert cut.degraded == ("blank image from alaskybis.cds.unistra.fr although the VLASS 3 GHz footprint "
                            "(CDS MocServer MOC) overlaps this field")
    headers = cut.headers()
    assert headers["Cache-Control"] == "no-store" and headers["X-Cutout-Degraded"].startswith("blank image from")
    assert cache.size_bytes() == 0


async def test_blank_image_is_not_cached_when_the_footprint_cannot_be_checked(cache: CutoutCache) -> None:
    """MocServer down: a blank image may be real or a rendering failure. The mirror is tried too;
    when it is blank as well, the image is returned neither cached nor flagged degraded, but marked
    ``X-Cutout-Blank: unconfirmed`` (never presented as a confirmed 'no data')."""
    blank = body("vlass_crab_png_bis")
    request = CutoutRequest(ra=83.63308, dec=22.0145, survey="vlass", fov_arcmin=2.0, width=64, height=64)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        images = [router.get(url__startswith=url).respond(200, content=blank, headers={"content-type": "image/png"})
                  for url in imaging.HIPS2FITS_URLS]
        mocs = [router.get(url__startswith=url).respond(503) for url in imaging.MOCSERVER_URLS]
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
    assert [r.call_count for r in images] == [1, 1], "a blank that the MOC cannot confirm fails over to the mirror"
    assert [r.call_count for r in mocs] == [1, 1], "the footprint is asked once per cutout, on each MocServer"
    assert cut.endpoint == imaging.HIPS2FITS_URLS[0]
    assert cut.coverage_fraction == 0.0 and cut.degraded is None and not cut.cacheable
    assert cut.blank == "unconfirmed"
    headers = cut.headers()
    assert headers["Cache-Control"] == "no-store" and headers["X-Cutout-Blank"] == "unconfirmed"
    assert "X-Cutout-Degraded" not in headers
    assert cache.size_bytes() == 0


async def test_unknown_hips_is_a_validation_error(cache: CutoutCache) -> None:
    with replay_router("unknown_hips"):
        async with offline_client() as client:
            with pytest.raises(UnknownSurveyError, match="BOGUS/P/nothing"):
                await CutoutService(client, cache=cache).cutout(spec("unknown_hips"))


async def test_all_mirrors_down_raises_upstream_error(cache: CutoutCache) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(503, json={"title": "Service Unavailable"})
        router.get(url__startswith=imaging.HIPS2FITS_URLS[1]).mock(side_effect=httpx.ConnectTimeout("timed out"))
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError) as info:
                await CutoutService(client, cache=cache).cutout(spec("dss2_3c273_png"))
    message = str(info.value)
    assert "alasky.cds.unistra.fr: HTTP 503 Service Unavailable" in message
    assert "alaskybis.cds.unistra.fr: ConnectTimeout" in message


def _png(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (40, 80, 120, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


TRUNCATED_PNG = body("dss2_3c273_png")[:400]  # valid 128x128 header, pixel stream cut short


def _corrupt_png() -> bytes:
    """The real DSS2 PNG with its compressed pixel stream scrambled but the IEND trailer intact."""
    data = bytearray(body("dss2_3c273_png"))
    start = data.index(b"IDAT") + 40
    data[start:start + 200] = bytes(200)
    return bytes(data)
# case: (status, body, content type, error fragment, requested size)
BAD_ANSWERS = {
    "json_200": (200, b'{"title": "odd"}', "application/json", "not a PNG, JPEG or FITS", 64),
    "html_maintenance_200": (200, b"<html><body>Maintenance</body></html>", "text/html", "not a PNG, JPEG or FITS", 64),
    "wrong_size": (200, body("dss2_3c273_png"), "image/png", "expected 64x64", 64),
    "truncated_png": (200, TRUNCATED_PNG, "image/png", r"truncated PNG \(no IEND chunk\)", 128),
    "corrupt_png_pixels": (200, _corrupt_png(), "image/png", "could not be decoded", 128),
    # Binary junk under an error status is described, never pasted into the message.
    "binary_500": (500, b"\x00\x01\xffBINARY" * 20, "application/octet-stream",
                   "HTTP 500 180 bytes of application/octet-stream", 64),
    "rate_limited_429": (429, b'{"title": "Too Many Requests"}', "application/json", "HTTP 429", 64),
    "not_found_404": (404, b"<html>Not Found</html>", "text/html", "HTTP 404", 64),
}


@pytest.mark.parametrize("case", sorted(BAD_ANSWERS))
async def test_unusable_answers_on_every_mirror_are_rejected(cache: CutoutCache, case: str) -> None:
    status, content, content_type, message, size = BAD_ANSWERS[case]
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=size, height=size)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.HIPS2FITS_URLS:
            router.get(url__startswith=url).respond(status, content=content, headers={"content-type": content_type})
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError, match=message) as info:
                await CutoutService(client, cache=cache).cutout(request)
    assert "alasky.cds.unistra.fr" in str(info.value) and "alaskybis.cds.unistra.fr" in str(info.value)
    assert cache.size_bytes() == 0, "rejected bodies must never be cached"


@pytest.mark.parametrize("case", sorted(BAD_ANSWERS))
async def test_mirror_serves_the_cutout_when_the_primary_answer_is_unusable(cache: CutoutCache, case: str) -> None:
    """A maintenance page, a truncated image, 429 or a bare 404 from alasky is retried on alaskybis."""
    status, content, content_type, _, size = BAD_ANSWERS[case]
    good = body("dss2_3c273_png") if size == 128 else _png(size, size)
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=size, height=size)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        primary = router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(
            status, content=content, headers={"content-type": content_type})
        mirror = router.get(url__startswith=imaging.HIPS2FITS_URLS[1]).respond(
            200, content=good, headers={"content-type": "image/png"})
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
    assert primary.called and mirror.called
    assert cut.endpoint == imaging.HIPS2FITS_URLS[1] and cut.content == good
    assert cut.degraded is None  # the mirror's image has data
    assert cache.size_bytes() == len(good)


async def test_client_errors_other_than_retryable_ones_are_final(cache: CutoutCache) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(
            400, json={"title": "Bad Request", "description": "bad projection"})
        mirror = router.get(url__startswith=imaging.HIPS2FITS_URLS[1]).respond(500)
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError, match=r"rejected the request \(HTTP 400\): Bad Request - bad projection"):
                await CutoutService(client, cache=cache).cutout(spec("dss2_3c273_png"))
    assert not mirror.called


async def test_blocking_work_runs_off_the_event_loop(cache: CutoutCache, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixel decoding and disk I/O run in worker threads: the loop keeps serving other tasks."""
    import asyncio

    def slow_coverage(content: bytes, fmt: str) -> float:
        time.sleep(0.4)  # stands in for decoding a 4096 x 4096 PNG (~0.7 s measured)
        return 1.0

    monkeypatch.setattr(imaging, "coverage_fraction", slow_coverage)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    with replay_router("dss2_3c273_png"):
        async with offline_client() as client:
            task = asyncio.create_task(ticker())
            started = time.perf_counter()
            await CutoutService(client, cache=cache).cutout(spec("dss2_3c273_png"))
            elapsed = time.perf_counter() - started
            task.cancel()
    assert elapsed >= 0.4
    assert ticks >= 15, f"event loop stalled: only {ticks} ticks in {elapsed:.2f} s"


# ---------------------------------------------------------------------------
# Colour composites vs single-band FITS (recorded bytes)
# ---------------------------------------------------------------------------


def test_colour_fits_is_a_display_cube_and_science_fits_holds_pixel_values() -> None:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(io.BytesIO(body("2mass_color_3c273_fits"))) as hdul:
        cube = hdul[0].data
    # hips2fits' FITS of a colour HiPS: 4 planes (RGBA) of 8-bit display values.
    assert cube.shape == (4, 32, 32) and cube.dtype.kind == "u" and cube.dtype.itemsize == 1
    with fits.open(io.BytesIO(body("2mass_k_3c273_fits"))) as hdul:
        data, header = hdul[0].data, hdul[0].header
    assert data.shape == (32, 32) and data.dtype.kind == "f"
    assert np.isfinite(data).all()
    # 3C 273 (Ks ~ 10 mag) is the brightest source at the centre of the 2' Ks field.
    peak = np.unravel_index(np.nanargmax(data), data.shape)
    sep = WCS(header).pixel_to_world(peak[1], peak[0]).separation(SkyCoord(*C3C273, unit="deg")).arcsec
    assert sep < 8.0
    colour = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass", fov_arcmin=2.0, width=32, height=32,
                           format="fits")
    science = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass_k", fov_arcmin=2.0, width=32, height=32,
                            format="fits")
    assert colour.pixel_kind == "rgb-preview" and science.pixel_kind == "survey"
    assert CutoutRequest(ra=1.0, dec=1.0, survey="2mass").pixel_kind == "display"


async def test_colour_fits_headers_point_to_the_science_survey(cache: CutoutCache) -> None:
    with replay_router("2mass_color_3c273_fits", "2mass_k_3c273_fits"):
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            colour = await service.cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass", fov_arcmin=2.0,
                                                        width=32, height=32, format="fits"))
            science = await service.cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass_k",
                                                         fov_arcmin=2.0, width=32, height=32, format="fits"))
    assert colour.headers()["X-Cutout-Pixels"] == "rgb-preview"
    assert colour.headers()["X-Cutout-Science-Survey"] == "2mass_k"
    assert science.headers()["X-Cutout-Pixels"] == "survey" and "X-Cutout-Science-Survey" not in science.headers()


def test_cache_expiry_prune_disable_and_clear(tmp_path: Path) -> None:
    store = CutoutCache(tmp_path, ttl_seconds=100, max_bytes=2500)
    keys = [f"{i:064x}" for i in range(3)]
    for i, key in enumerate(keys):
        store.put(key, bytes([i]) * 1000, {"n": i})
        os.utime(tmp_path / f"{key}.bin", (time.time() + i, time.time() + i))
    # 3000 bytes > 2500: the least recently used entry (keys[0]) was pruned.
    assert store.get(keys[0]) is None
    content, meta = store.get(keys[2])
    assert content == bytes([2]) * 1000 and meta["n"] == 2 and meta["bytes"] == 1000
    meta_path = tmp_path / f"{keys[2]}.json"
    stale = json.loads(meta_path.read_text())
    stale["stored_at"] = time.time() - 101
    meta_path.write_text(json.dumps(stale))
    assert store.get(keys[2]) is None, "entries older than the TTL are misses"
    (tmp_path / f"{keys[1]}.bin").write_bytes(b"truncated")
    assert store.get(keys[1]) is None, "a size mismatch (partial write) is a miss"
    assert store.clear() == 2
    with pytest.raises(ValueError):
        store.get("../../etc/passwd")
    off = CutoutCache(tmp_path / "off", ttl_seconds=0)
    off.put(keys[0], b"x", {})
    assert off.get(keys[0]) is None and not (tmp_path / "off").exists()


def test_cache_prunes_only_when_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No directory scan per write: size is tracked; prune on overflow or every N writes."""
    store = CutoutCache(tmp_path, ttl_seconds=3600, max_bytes=10_000)
    calls = 0
    original = CutoutCache.prune

    def counting_prune(self: CutoutCache) -> int:
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CutoutCache, "prune", counting_prune)
    for i in range(10):
        store.put(f"{i:064x}", b"x" * 1000, {})
    assert calls == 0, "writes up to max_bytes must not scan the cache directory"
    store.put(f"{10:064x}", b"x" * 1000, {})  # 11_000 > 10_000
    assert calls == 1 and store.size_bytes() <= 10_000
    store.put(f"{10:064x}", b"y" * 1000, {})  # rewriting an entry does not double-count it
    assert calls == 1
    monkeypatch.setattr(imaging, "CACHE_PRUNE_EVERY_PUTS", 3)
    sweep = CutoutCache(tmp_path / "sweep", ttl_seconds=3600, max_bytes=10**9)
    for i in range(3):
        sweep.put(f"{i:064x}", b"z", {})
    assert calls == 2, "expired entries are still swept periodically"


def test_shared_cache_is_one_instance_per_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "a"))
    first = CutoutCache.shared()
    assert CutoutCache.shared() is first
    assert CutoutService(None).cache is first
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "b"))
    assert CutoutCache.shared() is not first


def test_cache_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "60")
    monkeypatch.setenv("CUTOUT_CACHE_MAX_BYTES", "1234")
    store = CutoutCache.from_env()
    assert (store.directory, store.ttl_seconds, store.max_bytes) == (tmp_path / "c", 60.0, 1234)


# ---------------------------------------------------------------------------
# Multi-wavelength stack & coverage
# ---------------------------------------------------------------------------


async def test_stack_for_3c273_orders_radio_to_xray(cache: CutoutCache) -> None:
    with replay_router("coverage_3c273"):
        async with offline_client() as client:
            panels, coverage = await CutoutService(client, cache=cache).plan_stack(*C3C273)
    assert coverage.checked and coverage.error is None
    assert [p.survey for p in panels] == ["tgss", "nvss", "allwise", "2mass", "panstarrs", "galex", "erosita", "xmm"]
    waves = [p.wavelength_m for p in panels]
    assert waves == sorted(waves, reverse=True)
    assert all(p.in_coverage is True for p in panels)
    # 3C 273 (l = 290 deg) lies in the German eROSITA half of the sky (l > 180 deg).
    assert coverage.covered["erosita"] is True
    # Full-sky surveys are decided without asking the MocServer.
    assert coverage.covered["allwise"] is True


async def test_stack_far_south_respects_survey_footprints(cache: CutoutCache) -> None:
    with replay_router("coverage_south"):
        async with offline_client() as client:
            panels, coverage = await CutoutService(client, cache=cache).plan_stack(*SOUTH)
    chosen = [p.survey for p in panels]
    # NVSS stops at dec -40, TGSS at -53, SDSS/Pan-STARRS are northern: none at dec -70.
    for absent in ("nvss", "tgss", "panstarrs", "sdss"):
        assert coverage.covered[absent] is False
        assert absent not in chosen
    assert chosen[0] == "racs_low"  # ASKAP (southern) radio survey fills the radio slot
    assert "allwise" in chosen and "2mass" in chosen and "legacy" in chosen


async def test_stack_without_mocserver_uses_first_choices(cache: CutoutCache) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith=imaging.MOCSERVER_URLS[0]).respond(500)
        router.get(url__startswith=imaging.MOCSERVER_URLS[1]).respond(502)
        async with offline_client() as client:
            panels, coverage = await CutoutService(client, cache=cache).plan_stack(*C3C273)
    assert not coverage.checked and "MocServer unavailable" in coverage.error
    # Unknown footprints: a slot prefers a full-sky survey (DSS2, RASS), else its first choice.
    assert [p.survey for p in panels] == ["tgss", "nvss", "allwise", "2mass", "dss2", "galex", "rass", "xmm"]
    assert {p.survey: p.in_coverage for p in panels}["allwise"] is True
    assert {p.survey: p.in_coverage for p in panels}["nvss"] is None


async def test_stack_with_explicit_surveys_is_wavelength_ordered(cache: CutoutCache) -> None:
    """Recorded: the ad-hoc Mellinger HiPS is described from its MocServer record; VLASS's MOC is queried."""
    with replay_router("stack_explicit_3c273") as router:
        async with offline_client() as client:
            panels, coverage = await CutoutService(client, cache=cache).plan_stack(
                *C3C273, surveys=["rass", "2mass", "vlass", "dss2", "CDS/P/Mellinger/color", "dss2"])
        assert router.calls.call_count == 2  # one registry description + one footprint query
    assert coverage.checked
    assert [p.survey for p in panels] == ["vlass", "2mass", "CDS/P/Mellinger/color", "dss2", "rass"]
    by_key = {p.survey: p for p in panels}
    # The VLASS Quick Look HiPS has no data at 3C 273 (MocServer), although the survey's
    # nominal footprint (dec > -40 deg) contains it.
    assert by_key["vlass"].in_coverage is False
    mellinger = by_key["CDS/P/Mellinger/color"]
    assert (mellinger.label, mellinger.regime, mellinger.wavelength) == (
        "Mellinger color optical survey", "optical", "400–800 nm")
    assert mellinger.color and mellinger.fits_survey is None  # JPEG tiles only: no single-band FITS
    assert mellinger.in_coverage is True
    assert by_key["2mass"].fits_survey == "2mass_k" and by_key["rass"].fits_survey == "rass"
    assert all(p.wavelength_m is not None for p in panels)


async def test_describe_hips_tolerates_registry_failures(cache: CutoutCache) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.get(url__startswith=imaging.MOCSERVER_URLS[0]).respond(500)
        router.get(url__startswith=imaging.MOCSERVER_URLS[1]).mock(side_effect=httpx.ConnectError("down"))
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            assert await service.describe_hips(["CDS/P/Mellinger/color"]) == {}
            panels, coverage = await service.plan_stack(*C3C273, surveys=["CDS/P/Mellinger/color"])
    assert not coverage.checked
    assert panels[0].wavelength == "unknown" and panels[0].wavelength_m is None and panels[0].regime == "unknown"


def test_survey_from_registry_record_sorts_swapped_limits() -> None:
    survey = imaging._survey_from_record({"ID": "xcatdb/P/XMM/PN/eb2", "obs_title": "XMM PN eb2", "em_min": "24.8e-10",
                                          "em_max": "12.4e-10", "bib_reference": "Submitted",
                                          "hips_tile_format": "png fits"})
    assert survey is not None
    assert (survey.em_min_m, survey.em_max_m) == (1.24e-9, 2.48e-9)
    assert survey.regime == "xray" and survey.bib_reference is None and not survey.color
    spitzer = imaging._survey_from_record({"ID": "CDS/P/SPITZER/IRAC1", "em_min": "3.1296e-06", "em_max": "3.9614e-06",
                                           "bib_reference": ["2003PASP..115..953B", "2009PASP..121..213C"],
                                           "hips_tile_format": "jpeg fits"})
    assert spitzer.bib_reference == "2003PASP..115..953B" and spitzer.regime == "infrared"
    assert imaging._survey_from_record({"ID": "not an id"}) is None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def make_app(tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(imaging.router)
    app.state.cutout_cache = CutoutCache(tmp_path / "api-cache", ttl_seconds=3600)
    return app


def test_router_serves_png_and_fits(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("dss2_3c273_png", "nvss_3c273_fits", passthrough_testserver=True):
        png = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2.0,
                                                    "survey": "dss2", "width": 128, "height": 128})
        fits_resp = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 6.0,
                                                          "survey": "nvss", "width": 64, "height": 64, "format": "fits"})
    assert png.status_code == 200, png.text
    assert png.headers["content-type"] == "image/png"
    assert png.content == body("dss2_3c273_png")
    assert png.headers["x-cutout-hips"] == "CDS/P/DSS2/color"
    assert png.headers["content-disposition"].startswith('inline; filename="cutout_dss2_')
    assert fits_resp.status_code == 200
    assert fits_resp.headers["content-type"] == "application/fits"
    assert fits_resp.content.startswith(b"SIMPLE  =")


def test_router_resolves_names_with_sesame(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="cds.unistra.fr").respond(200, text=SESAME_3C273)
        moc = router.get(url__startswith=imaging.MOCSERVER_URLS[0]).respond(
            200, content=body("coverage_3c273"), headers={"content-type": "application/json"})
        resp = client.get("/api/v1/cutouts/stack", params={"name": "3C 273", "fov_arcmin": 2.5, "size": 200})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Sesame (SIMBAD) J2000 position of 3C 273, used for the footprint query and the panel URLs.
    assert data["target"]["ra"] == pytest.approx(187.27791594) and data["target"]["canonical_name"] == "3C 273"
    assert moc.calls.last.request.url.params["RA"] == "187.27791594"
    assert all("ra=187.27791594" in p["url"] and "width=200" in p["url"] for p in data["panels"])
    # SIMBAD lists pm -0.02/+0.10 mas/yr for this quasar (Gaia noise): the resolver marks it not
    # applicable, so the panels are not moved.
    assert data["proper_motion"] is None and all(p["offset_arcsec"] == 0.0 for p in data["panels"])


@pytest.mark.parametrize("params, status, fragment", [
    ({"ra": 400, "dec": 0}, 422, None),
    ({"dec": 0}, 422, "Either name or both ra and dec"),
    ({"ra": 10, "dec": 10, "survey": "first"}, 422, "FIRST is not published"),
    ({"ra": 10, "dec": 10, "stretch": "cubic"}, 422, "stretch must be"),
    ({"ra": 10, "dec": 10, "format": "gif"}, 422, None),
    ({"ra": 10, "dec": 10, "fov_arcmin": 0}, 422, None),
    # hips2fits answers these with HTTP 500, which would read as an outage: rejected here.
    ({"ra": 83.63308, "dec": 22.0145, "min_cut": "200%"}, 422, "min_cut percentile must be in [0, 100]"),
    ({"ra": 83.63308, "dec": 22.0145, "min_cut": "5", "max_cut": "1"}, 422, "min_cut must not exceed max_cut"),
    ({"ra": 10, "dec": 10, "fov_arcmin": 10800}, 422, "< 10800 for TAN"),
    ({"ra": 10, "dec": 10, "fov_arcmin": 21601, "projection": "MOL"}, 422, None),
])
def test_router_rejects_bad_input(tmp_path: Path, params: dict, status: int, fragment: str | None) -> None:
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        resp = client.get("/api/v1/cutouts", params=params)
    assert resp.status_code == status
    if fragment:
        assert fragment in resp.json()["detail"]


def test_router_unknown_hips_is_422_and_outage_is_502(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("unknown_hips", passthrough_testserver=True):
        unknown = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2.0,
                                                        "survey": "BOGUS/P/nothing", "width": 64, "height": 64})
    assert unknown.status_code == 422 and "Could not find a HiPS" in unknown.json()["detail"]
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        router.get(url__startswith="https://alasky").respond(500, json={"title": "500 Internal Server Error"})
        down = client.get("/api/v1/cutouts", params={"ra": 10, "dec": 10})
    assert down.status_code == 502
    assert "hips2fits unavailable" in down.json()["detail"] and "HTTP 500" in down.json()["detail"]


def test_router_accepts_all_sky_projections(tmp_path: Path) -> None:
    """MOL at fov = 360 deg is a valid hips2fits request (its own HI4PI example is 360 deg wide)."""
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        upstream = router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(
            200, content=_png(128, 64), headers={"content-type": "image/png"})
        resp = client.get("/api/v1/cutouts", params={"ra": 0, "dec": 0, "fov_arcmin": 21600, "projection": "MOL",
                                                     "width": 128, "height": 64, "survey": "dss2"})
    assert resp.status_code == 200, resp.text
    sent = upstream.calls.last.request.url.params
    assert (sent["projection"], float(sent["fov"])) == ("MOL", 360.0)


def test_router_colour_fits_is_labelled_and_degraded_is_not_cacheable(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("2mass_color_3c273_fits", "vlass_crab_png", passthrough_testserver=True):
        colour = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2.0,
                                                       "survey": "2mass", "width": 32, "height": 32, "format": "fits"})
        vlass = client.get("/api/v1/cutouts", params={"ra": 83.63308, "dec": 22.0145, "fov_arcmin": 2.0,
                                                      "survey": "vlass", "width": 64, "height": 64})
    assert colour.status_code == 200
    assert colour.headers["x-cutout-pixels"] == "rgb-preview" and colour.headers["x-cutout-science-survey"] == "2mass_k"
    assert vlass.status_code == 200
    assert vlass.headers["x-cutout-coverage"] == "0.0000"
    assert vlass.headers["x-cutout-degraded"].startswith("blank image from mirror")
    assert vlass.headers["cache-control"] == "no-store"


def test_router_surveys_and_stack(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    surveys = client.get("/api/v1/cutouts/surveys")
    assert surveys.status_code == 200
    listed = surveys.json()
    assert {"nvss", "vlass", "allwise", "2mass", "panstarrs", "sdss", "dss2", "galex", "rass", "xmm"} <= {s["key"] for s in listed}
    assert all({"hips_id", "label", "wavelength", "wavelength_m", "regime", "pixel_units", "calibrated",
                "epoch_span"} <= set(s) for s in listed)
    by_key = {s["key"]: s for s in listed}
    # What FITS pixels are, per survey (registry + survey papers): never "calibrated" for plate densities or DN.
    assert (by_key["dss2_red"]["pixel_units"], by_key["dss2_red"]["calibrated"]) == ("photographic density", False)
    assert (by_key["2mass_k"]["pixel_units"], by_key["2mass_k"]["calibrated"]) == ("DN", False)
    assert (by_key["nvss"]["pixel_units"], by_key["nvss"]["calibrated"]) == ("Jy/beam", True)
    assert by_key["legacy"]["science_alternates"] == ["legacy_g", "legacy_z", "legacy_i"]
    assert by_key["2mass"]["epoch_span"] == [1997.41, 2001.09]
    with replay_router("coverage_3c273", passthrough_testserver=True):
        resp = client.get("/api/v1/cutouts/stack", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 3})
    assert resp.status_code == 200, resp.text
    stack = resp.json()
    assert stack["coverage_checked"] and stack["size_px"] == 256 and stack["format"] == "png"
    panels = stack["panels"]
    assert [p["survey"] for p in panels][:2] == ["tgss", "nvss"] and panels[-1]["regime"] == "xray"
    for panel in panels:
        assert {"survey", "label", "wavelength", "url", "fits_url", "fits_survey", "color"} <= set(panel)
        # Root-relative: valid behind a TLS-terminating proxy, whatever Host/scheme uvicorn saw.
        assert panel["url"].startswith("/api/v1/cutouts?")
        url = httpx.URL(panel["url"])
        assert url.host == "" and url.path == "/api/v1/cutouts"
        assert url.params["survey"] == panel["survey"] and url.params["fov_arcmin"] == "3.0"
        assert url.params["ra"] == "187.2779154" and url.params["width"] == url.params["height"] == "256"
        assert url.params["format"] == "png"
        fits_url = httpx.URL(panel["fits_url"])
        assert fits_url.params["format"] == "fits" and fits_url.params["survey"] == panel["fits_survey"]
    units = {p["survey"]: (p["fits_pixel_units"], p["fits_calibrated"]) for p in panels}
    assert units["nvss"] == ("Jy/beam", True) and units["2mass"] == ("DN", False)
    assert units["panstarrs"] == ("counts", False) and units["xmm"] == ("unknown", False)
    assert stack["proper_motion"] is None and all(p["epoch"] is None for p in panels)
    fits_of = {p["survey"]: p["fits_survey"] for p in panels}
    # Colour composites link to a single-band survey that covers the position; single-band ones to themselves.
    assert fits_of == {"tgss": "tgss", "nvss": "nvss", "allwise": "allwise_w1", "2mass": "2mass_k",
                       "panstarrs": "panstarrs_g", "galex": "galex_nuv", "erosita": "erosita", "xmm": "xmm_eb2"}


def test_router_stack_urls_follow_the_root_path(tmp_path: Path) -> None:
    """Served under a sub-path (root_path), panel URLs carry it, still without scheme or host."""
    client = TestClient(make_app(tmp_path), root_path="/astro")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        for url in imaging.MOCSERVER_URLS:
            router.get(url__startswith=url).respond(503)
        resp = client.get("/api/v1/cutouts/stack", params={"ra": C3C273[0], "dec": C3C273[1], "surveys": "dss2,chandra"})
    assert resp.status_code == 200, resp.text
    # DSS2 is full-sky; Chandra's footprint stays unknown here (MocServer down) but the panel is listed.
    assert resp.json()["coverage_checked"] is False
    panels = {p["survey"]: p for p in resp.json()["panels"]}
    assert panels["dss2"]["url"].startswith("/astro/api/v1/cutouts?")
    assert panels["dss2"]["fits_url"].startswith("/astro/api/v1/cutouts?") and "survey=dss2_red" in panels["dss2"]["fits_url"]
    # DSS2 red FITS are photographic plate densities: linked, but never called calibrated.
    assert (panels["dss2"]["fits_pixel_units"], panels["dss2"]["fits_calibrated"]) == ("photographic density", False)
    assert panels["chandra"]["fits_url"] is None and panels["chandra"]["fits_survey"] is None


def test_router_stack_with_adhoc_hips_is_strict_json(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("stack_explicit_3c273", passthrough_testserver=True):
        resp = client.get("/api/v1/cutouts/stack", params={
            "ra": C3C273[0], "dec": C3C273[1], "surveys": "rass,2mass,vlass,dss2,CDS/P/Mellinger/color"})
    assert resp.status_code == 200, resp.text

    def no_nan(token: str) -> None:
        raise AssertionError(f"non-JSON number {token} in the response")

    data = json.loads(resp.text, parse_constant=no_nan)
    mellinger = next(p for p in data["panels"] if p["survey"] == "CDS/P/Mellinger/color")
    assert "nan" not in mellinger["wavelength"] and mellinger["wavelength"] == "400–800 nm"


def test_stack_panel_url_round_trips_through_the_cutout_route(tmp_path: Path) -> None:
    """A panel URL, requested verbatim, reaches hips2fits with exactly the recorded parameters."""
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        stack = client.get("/api/v1/cutouts/stack", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2.0,
                                                            "surveys": "dss2", "size": 128}).json()
    url = httpx.URL(stack["panels"][0]["url"])
    with replay_router("dss2_3c273_png", passthrough_testserver=True):
        resp = client.get(url.path, params=dict(url.params))
    assert resp.status_code == 200, resp.text
    assert resp.content == body("dss2_3c273_png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    imaging.register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_cli_downloads_fits_with_format_from_extension(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "cli-cache"))
    out = tmp_path / "nvss.fits"
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "6",
                                    "--survey", "nvss", "--width", "64", "--height", "64", "--out", str(out)])
    assert args.handler is imaging.cli_cutout
    with replay_router("nvss_3c273_fits"):
        code = args.handler(args)
    assert code == 0
    assert out.read_bytes() == body("nvss_3c273_fits")
    printed = capsys.readouterr().out
    assert "NVSS 1.4 GHz 64x64 FITS" in printed and "data coverage 100.0%" in printed


def test_cli_unwritable_output_is_a_clean_error(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory")
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "6",
                                    "--survey", "nvss", "--width", "64", "--height", "64",
                                    "--out", str(blocker / "x.fits")])
    with replay_router("nvss_3c273_fits"):
        code = args.handler(args)
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: cannot write ") and "Traceback" not in err


def test_cli_warns_that_colour_fits_is_not_survey_data(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "2",
                                    "--survey", "2mass", "--width", "32", "--height", "32",
                                    "--out", str(tmp_path / "c.fits")])
    with replay_router("2mass_color_3c273_fits"):
        assert args.handler(args) == 0
    err = capsys.readouterr().err
    assert "colour composite" in err and "--survey 2mass_k" in err


def test_cli_resolves_name(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    out = tmp_path / "dss.png"
    args = cli_parser().parse_args(["cutout", "--name", "3C 273", "--fov", "2", "--survey", "dss2",
                                    "--width", "128", "--height", "128", "--out", str(out)])
    captured: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=body("dss2_3c273_png"), headers={"content-type": "image/png"})

    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="cds.unistra.fr").respond(200, text=SESAME_3C273)
        router.route(host="alasky.cds.unistra.fr").mock(side_effect=answer)
        code = args.handler(args)
    assert code == 0 and out.exists()
    assert "Resolved '3C 273' -> RA=187.277916" in capsys.readouterr().out
    assert captured[0].url.params["ra"] == "187.27791594"


def test_cli_errors_and_listing(tmp_path: Path, capsys) -> None:
    parser = cli_parser()
    no_out = parser.parse_args(["cutout", "--ra", "1", "--dec", "1"])
    assert no_out.handler(no_out) == 2
    assert "--out is required" in capsys.readouterr().err
    no_pos = parser.parse_args(["cutout", "--out", str(tmp_path / "x.png")])
    assert no_pos.handler(no_pos) == 2
    # Invalid arguments (including an unknown survey) are usage errors: exit code 2.
    bad = parser.parse_args(["cutout", "--ra", "1", "--dec", "1", "--survey", "first", "--out", str(tmp_path / "x.png")])
    assert bad.handler(bad) == 2
    assert "FIRST is not published" in capsys.readouterr().err
    out_of_range = parser.parse_args(["cutout", "--ra", "400", "--dec", "0", "--out", str(tmp_path / "x.png")])
    assert out_of_range.handler(out_of_range) == 2
    assert "ra must be in [0, 360)" in capsys.readouterr().err
    listing = parser.parse_args(["cutout", "--list-surveys"])
    assert listing.handler(listing) == 0
    text = capsys.readouterr().out
    assert "CDS/P/NVSS" in text and "first" in text and "unavailable" in text
    assert not (tmp_path / "x.png").exists()


# ---------------------------------------------------------------------------
# Regressions: truncated bodies, cut diagnosis, error labels, MocServer error kinds
# ---------------------------------------------------------------------------

JPEG = body("panstarrs_m87_jpg")


def _jpeg_request() -> CutoutRequest:
    return spec("panstarrs_m87_jpg")


def test_completeness_checks_need_no_decoder() -> None:
    """IEND / EOI / FITS data length are checked structurally, before any pixel decoding."""
    png, fits_body = body("dss2_3c273_png"), body("nvss_3c273_fits")
    for content in (png, JPEG, fits_body, body("2mass_color_3c273_fits")):
        imaging.check_complete(content, probe_image(content))
    with pytest.raises(CutoutUpstreamError, match="truncated PNG"):
        imaging.check_complete(png[:-12], probe_image(png))
    # The JPEG comment block puts SOF0 far into the file: a body cut after it still has a valid header.
    cut_jpeg = JPEG[:int(len(JPEG) * 0.9)]
    assert probe_image(cut_jpeg) == probe_image(JPEG)
    with pytest.raises(CutoutUpstreamError, match="truncated JPEG"):
        imaging.check_complete(cut_jpeg, probe_image(cut_jpeg))
    header_and_half = fits_body[:2880 + 64 * 64 * 4 // 2]
    with pytest.raises(CutoutUpstreamError, match=r"truncated FITS \(\d+ of 19264 bytes\)"):
        imaging.check_complete(header_and_half, probe_image(header_and_half))
    with pytest.raises(CutoutUpstreamError, match="could not be decoded"):
        coverage_fraction(JPEG[:3300], "jpg")  # Pillow refuses truncated scans too


async def test_truncated_jpeg_is_retried_on_the_mirror_and_never_cached(cache: CutoutCache) -> None:
    truncated = JPEG[:int(len(JPEG) * 0.9)]
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        primary = router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(
            200, content=truncated, headers={"content-type": "image/jpeg"})
        mirror = router.get(url__startswith=imaging.HIPS2FITS_URLS[1]).respond(
            200, content=JPEG, headers={"content-type": "image/jpeg"})
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(_jpeg_request())
    assert primary.called and mirror.called
    assert cut.endpoint == imaging.HIPS2FITS_URLS[1] and cut.content == JPEG
    assert cache.size_bytes() == len(JPEG)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.HIPS2FITS_URLS:
            router.get(url__startswith=url).respond(200, content=truncated, headers={"content-type": "image/jpeg"})
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError, match="truncated JPEG") as info:
                await CutoutService(client, cache=CutoutCache(cache.directory / "x", ttl_seconds=60)).cutout(
                    _jpeg_request())
    assert str(info.value).startswith("hips2fits returned no usable image: ")


@pytest.mark.parametrize("case, prefix", [
    ("json_200", "hips2fits returned no usable image: "),
    ("truncated_png", "hips2fits returned no usable image: "),
    ("rate_limited_429", "hips2fits unavailable: "),
    ("binary_500", "hips2fits unavailable: "),
])
async def test_failure_messages_separate_outages_from_bad_answers(cache: CutoutCache, case: str, prefix: str) -> None:
    status, content, content_type, _, size = BAD_ANSWERS[case]
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=size, height=size)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.HIPS2FITS_URLS:
            router.get(url__startswith=url).respond(status, content=content, headers={"content-type": content_type})
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError) as info:
                await CutoutService(client, cache=cache).cutout(request)
    assert str(info.value).startswith(prefix)
    assert "BINARY" not in str(info.value) and "\x00" not in str(info.value)


def test_error_detail_never_embeds_binary_bodies() -> None:
    fits_as_text = httpx.Response(500, content=body("nvss_3c273_fits")[:600], headers={"content-type": "image/fits"})
    assert imaging._detail(fits_as_text) == "600 bytes of image/fits"
    unlabelled = httpx.Response(500, content=b"\xff\xd8\xff\xe0 junk")
    assert imaging._detail(unlabelled) == "9 bytes of unlabelled content"
    assert imaging._detail(httpx.Response(500, json={"title": "500 Internal Server Error"})) == "500 Internal Server Error"
    assert imaging._detail(httpx.Response(503, text="Service\x07 down", headers={"content-type": "text/plain"})) == "Service? down"


def _cut_router(router: respx.MockRouter, without_cuts: int) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if "min_cut" in request.url.params or "max_cut" in request.url.params:
            return httpx.Response(500, json={"title": "500 Internal Server Error"})
        return httpx.Response(without_cuts, content=body("dss2_3c273_png") if without_cuts == 200 else b"",
                              headers={"content-type": "image/png"})

    for url in imaging.HIPS2FITS_URLS:
        router.get(url__startswith=url).mock(side_effect=answer)


async def test_cuts_that_hips2fits_cannot_render_are_a_validation_error(cache: CutoutCache) -> None:
    """min_cut=1% with max_cut=0.02 passes local checks (different kinds) but gives HTTP 500 on both
    mirrors (verified live); the same cutout without cuts renders, so the cuts are the problem."""
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=128, height=128,
                            min_cut="1%", max_cut="0.02")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        _cut_router(router, 200)
        async with offline_client() as client:
            with pytest.raises(CutoutValidationError, match=r"cannot render this cutout with min_cut=1%, max_cut=0\.02"):
                await CutoutService(client, cache=cache).cutout(request)
        sent = [dict(call.request.url.params) for call in router.calls]
    assert [("min_cut" in p) for p in sent] == [True, True, False]  # both mirrors, then one probe without cuts
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        _cut_router(router, 500)  # a real outage: the probe fails too
        async with offline_client() as client:
            with pytest.raises(CutoutUpstreamError, match="^hips2fits unavailable: "):
                await CutoutService(client, cache=cache).cutout(request)


def test_router_answers_422_for_unrenderable_cuts(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        _cut_router(router, 200)
        resp = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2,
                                                     "width": 128, "height": 128, "min_cut": "1%", "max_cut": "0.02"})
    assert resp.status_code == 422, resp.text
    assert "cannot render this cutout" in resp.json()["detail"] and "unavailable" not in resp.json()["detail"]


@pytest.mark.parametrize("answer, kind, outage", [
    (lambda: httpx.Response(200, json={"ids": ["CDS/P/NVSS"]}), "parse", False),  # a format change, not an outage
    (lambda: httpx.Response(200, text="<html>maintenance</html>", headers={"content-type": "text/html"}), "parse", False),
    (lambda: httpx.Response(400, text="bad expr"), "http4xx", False),
    (lambda: httpx.Response(503), "http5xx", True),
])
async def test_coverage_reports_why_the_footprint_is_unknown(cache: CutoutCache, answer, kind: str, outage: bool) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.MOCSERVER_URLS:
            router.get(url__startswith=url).mock(side_effect=lambda request: answer())
        async with offline_client() as client:
            result = await CutoutService(client, cache=cache).coverage(*C3C273, [SURVEYS["nvss"], SURVEYS["2mass"]])
    assert not result.checked and result.covered == {"nvss": None, "2mass": True}
    assert result.error_kind == kind and result.outage is outage
    assert result.error.startswith("MocServer unavailable: " if outage else "MocServer answer unusable: ")


async def test_coverage_transport_failure_is_an_outage(cache: CutoutCache) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.MOCSERVER_URLS:
            router.get(url__startswith=url).mock(side_effect=httpx.ConnectError("refused"))
        async with offline_client() as client:
            result = await CutoutService(client, cache=cache).coverage(*C3C273, [SURVEYS["nvss"]])
    assert result.error_kind == "transport" and result.outage


# ---------------------------------------------------------------------------
# Regressions: pixel labelling from content, FITS pixel units
# ---------------------------------------------------------------------------


async def test_adhoc_colour_hips_fits_is_labelled_from_its_content(cache: CutoutCache) -> None:
    """CDS/P/Mellinger/color is not catalogued (get_survey assumes single-band), but hips2fits returns
    an 8-bit RGBA cube for it: the cutout must say 'rgb-preview', not 'survey'."""
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="CDS/P/Mellinger/color", fov_arcmin=60.0,
                            width=32, height=32, format="fits")
    assert request.pixel_kind == "survey"  # what the key alone predicts
    with replay_router("mellinger_3c273_fits"):
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(request)
            hit = await service.cutout(request)
    assert cut.info == imaging.ImageInfo("fits", 32, 32, 4, 8)
    assert cut.pixel_kind == "rgb-preview" and hit.cached and hit.pixel_kind == "rgb-preview"
    headers = cut.headers()
    assert headers["X-Cutout-Pixels"] == "rgb-preview"
    assert "X-Cutout-Pixel-Units" not in headers and "X-Cutout-Science-Survey" not in headers


def test_router_labels_survey_fits_with_pixel_units(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("nvss_3c273_fits", "2mass_k_3c273_fits", "mellinger_3c273_fits", passthrough_testserver=True):
        nvss = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 6.0,
                                                     "survey": "nvss", "width": 64, "height": 64, "format": "fits"})
        ks = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 2.0,
                                                   "survey": "2mass_k", "width": 32, "height": 32, "format": "fits"})
        mellinger = client.get("/api/v1/cutouts", params={"ra": C3C273[0], "dec": C3C273[1], "fov_arcmin": 60.0,
                                                          "survey": "CDS/P/Mellinger/color", "width": 32, "height": 32,
                                                          "format": "fits"})
    assert (nvss.headers["x-cutout-pixels"], nvss.headers["x-cutout-pixel-units"],
            nvss.headers["x-cutout-calibrated"]) == ("survey", "Jy/beam", "true")
    # 2MASS Ks pixels are background-subtracted DN: no zero point in the header, so not calibrated.
    assert (ks.headers["x-cutout-pixel-units"], ks.headers["x-cutout-calibrated"]) == ("DN", "false")
    assert mellinger.status_code == 200 and mellinger.headers["x-cutout-pixels"] == "rgb-preview"


def test_cli_warns_for_adhoc_colour_fits(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "60",
                                    "--survey", "CDS/P/Mellinger/color", "--width", "32", "--height", "32",
                                    "--out", str(tmp_path / "m.fits")])
    with replay_router("mellinger_3c273_fits"):
        assert args.handler(args) == 0
    err = capsys.readouterr().err
    assert "colour composite" in err and "8-bit RGBA display planes" in err


# ---------------------------------------------------------------------------
# Regressions: FITS companions follow their own footprint
# ---------------------------------------------------------------------------


def test_fits_companion_falls_back_within_the_survey() -> None:
    legacy = SURVEYS["legacy"]
    assert imaging.fits_companion(legacy, {"legacy_r": True})[0].key == "legacy_r"
    pick, note = imaging.fits_companion(legacy, {"legacy_r": False, "legacy_g": True, "legacy_z": True})
    assert pick.key == "legacy_g" and note == "Legacy Surveys DR10 r has no data here; Legacy Surveys DR10 g is used instead."
    pick, note = imaging.fits_companion(legacy, {"legacy_r": False, "legacy_g": False, "legacy_z": False, "legacy_i": False})
    assert pick is None and note.startswith("No single-band FITS survey of Legacy Surveys DR10 covers this position")
    assert imaging.fits_companion(legacy, {})[0].key == "legacy_r"  # footprints unknown: first choice
    assert imaging.fits_companion(SURVEYS["2mass"], {"2mass_k": None})[0].key == "2mass_k"  # full sky
    pick, note = imaging.fits_companion(SURVEYS["chandra"], {})
    assert pick is None and "no single-band FITS HiPS" in note


async def test_stack_checks_the_fits_companion_footprint(cache: CutoutCache) -> None:
    """Recorded at (345, -35): the MocServer lists DR10 colour, g, i and z but not DR10 r there."""
    with replay_router("coverage_legacy_gap") as router:
        async with offline_client() as client:
            panels, coverage = await CutoutService(client, cache=cache).plan_stack(345.0, -35.0)
        expr = router.calls.last.request.url.params["expr"]
    # Companions are part of the one footprint query.
    assert "ID=CDS/P/DESI-Legacy-Surveys/DR10/r" in expr and "ID=CDS/P/GALEXGR6_7/NUV" in expr
    assert coverage.covered["legacy"] is True and coverage.covered["legacy_r"] is False
    assert coverage.covered["legacy_g"] is True
    legacy = next(p for p in panels if p.survey == "legacy")
    assert (legacy.fits_survey, legacy.fits_pixel_units, legacy.fits_calibrated) == ("legacy_g", "nanomaggies", True)
    assert legacy.fits_note == "Legacy Surveys DR10 r has no data here; Legacy Surveys DR10 g is used instead."


def test_router_stack_links_a_covered_fits_band(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("coverage_legacy_gap", passthrough_testserver=True):
        resp = client.get("/api/v1/cutouts/stack", params={"ra": 345.0, "dec": -35.0})
    assert resp.status_code == 200, resp.text
    legacy = next(p for p in resp.json()["panels"] if p["survey"] == "legacy")
    assert httpx.URL(legacy["fits_url"]).params["survey"] == "legacy_g"
    assert legacy["fits_label"] == "Legacy Surveys DR10 g" and legacy["fits_note"]


# ---------------------------------------------------------------------------
# Regressions: proper motion (Barnard's star)
# ---------------------------------------------------------------------------

BARNARD = (269.45207696, 4.69336497)  # SIMBAD ICRS J2000
BARNARD_PM = (-801.551, 10362.394)  # SIMBAD, mas/yr
SESAME_BARNARD = (FIXTURES / "sesame" / "barnards_star.xml").read_text(encoding="utf-8")


def test_panel_centre_moves_the_target_to_the_survey_epoch() -> None:
    from astropy.coordinates import SkyCoord

    motion = imaging.ProperMotion(*BARNARD_PM, 2000.0)
    ps1 = SURVEYS["panstarrs"]
    centre = imaging.panel_centre(ps1, *BARNARD, motion, fov_arcmin=3.0)
    # PS1 3pi: registry t_min/t_max MJD 54999.5-56896.2 = 2009.46-2014.65; the centre is the mid-epoch.
    assert ps1.epoch_span == (pytest.approx(2009.46, abs=0.01), pytest.approx(2014.65, abs=0.01))
    assert centre.epoch == pytest.approx(sum(ps1.epoch_span) / 2)
    rate = math.hypot(*BARNARD_PM) / 1000.0  # 10.39"/yr
    assert centre.offset_arcsec == pytest.approx(rate * (centre.epoch - 2000.0), rel=1e-3)
    moved = SkyCoord(centre.ra, centre.dec, unit="deg").separation(SkyCoord(*BARNARD, unit="deg")).arcsec
    assert moved == pytest.approx(centre.offset_arcsec, rel=2e-3)  # ~125": outside a 3' field's half-width
    assert centre.dec > BARNARD[1]  # Barnard's star moves north
    assert centre.note is None  # +-27" over 2009-2014 fits in a 90" half-field
    tight = imaging.panel_centre(ps1, *BARNARD, motion, fov_arcmin=0.5)
    assert tight.note and "may lie outside the field" in tight.note
    unknown = imaging.panel_centre(SURVEYS["chandra"], *BARNARD, motion, fov_arcmin=3.0)
    assert (unknown.ra, unknown.dec, unknown.epoch) == (*BARNARD, None) and "Survey epoch unknown" in unknown.note
    # unWISE neo8 coadds exposures of 2010.0-2011.1 and 2014.0-2022.0 (Meisner et al. 2022): the panel is
    # centred at 2016.0, 166" north-ish of J2000, and the +-62" spread fits the 90" half-field.
    for key in ("unwise", "unwise_w1"):
        neo8 = imaging.panel_centre(SURVEYS[key], *BARNARD, motion, fov_arcmin=3.0)
        assert neo8.epoch == 2016.0 and neo8.offset_arcsec == pytest.approx(rate * 16.0, rel=1e-3)
        assert neo8.spread_arcsec == pytest.approx(rate * 6.0, rel=1e-3) and neo8.note is None
    # DSS2 colour combines blue plates from 1975 with red plates to 1999 (registry 1975.0-1999.0): the
    # +-125" spread exceeds a 3' field's half-width, so the panel says so; DSS2 red alone spans 1984-1999.
    dss2 = imaging.panel_centre(SURVEYS["dss2"], *BARNARD, motion, fov_arcmin=3.0)
    assert SURVEYS["dss2"].epoch_span == (pytest.approx(1975.0, abs=0.01), pytest.approx(1999.0, abs=0.01))
    assert dss2.epoch == pytest.approx(1987.0, abs=0.01) and dss2.spread_arcsec == pytest.approx(rate * 12.0, rel=2e-3)
    assert dss2.note and "may lie outside the field" in dss2.note
    assert SURVEYS["dss2_red"].epoch_span == (pytest.approx(1984.1, abs=0.1), pytest.approx(1999.0, abs=0.01))
    assert imaging.panel_centre(ps1, *BARNARD, None) == imaging.PanelCentre(*BARNARD)
    with pytest.raises(CutoutValidationError):
        imaging.ProperMotion(30000.0, 0.0)


def _expected_centre(survey: str) -> tuple[float, float]:
    from models import propagate_radec

    return propagate_radec(*BARNARD, *BARNARD_PM, 2000.0, SURVEYS[survey].mean_epoch)


def test_router_stack_follows_proper_motion(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("coverage_barnard", passthrough_testserver=True):
        resp = client.get("/api/v1/cutouts/stack", params={
            "ra": BARNARD[0], "dec": BARNARD[1], "pm_ra_masyr": BARNARD_PM[0], "pm_dec_masyr": BARNARD_PM[1]})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["proper_motion"] == {"pm_ra_masyr": BARNARD_PM[0], "pm_dec_masyr": BARNARD_PM[1], "epoch": 2000.0}
    panels = {p["survey"]: p for p in data["panels"]}
    assert {"panstarrs", "2mass", "allwise", "rass"} <= set(panels)
    for key in ("panstarrs", "2mass", "allwise", "rass"):
        panel = panels[key]
        ra, dec = _expected_centre(key)
        assert (panel["center_ra"], panel["center_dec"]) == (pytest.approx(ra, abs=1e-9), pytest.approx(dec, abs=1e-9))
        for link in (panel["url"], panel["fits_url"]):
            params = httpx.URL(link).params
            assert float(params["ra"]) == pytest.approx(ra, abs=1e-9) and float(params["dec"]) == pytest.approx(dec, abs=1e-9)
    # AllWISE (2010.6): ~110" north of the J2000 position; RASS (1990.8): ~96" south of it.
    assert panels["allwise"]["offset_arcsec"] == pytest.approx(110.2, abs=0.5)
    assert panels["rass"]["center_dec"] < BARNARD[1] < panels["allwise"]["center_dec"]


def test_router_stack_takes_proper_motion_from_sesame(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with replay_router("coverage_barnard", passthrough_testserver=True, sesame=SESAME_BARNARD):
        resp = client.get("/api/v1/cutouts/stack", params={"name": "Barnard's star"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["proper_motion"] == {"pm_ra_masyr": BARNARD_PM[0], "pm_dec_masyr": BARNARD_PM[1], "epoch": 2000.0}
    ps1 = next(p for p in data["panels"] if p["survey"] == "panstarrs")
    assert ps1["center_ra"] == pytest.approx(_expected_centre("panstarrs")[0], abs=1e-9)
    assert ps1["center_dec"] == pytest.approx(_expected_centre("panstarrs")[1], abs=1e-9)


def test_router_stack_rejects_half_a_proper_motion(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    resp = client.get("/api/v1/cutouts/stack", params={"ra": 10, "dec": 10, "pm_ra_masyr": 5})
    assert resp.status_code == 422 and "given together" in resp.json()["detail"]
    assert client.get("/api/v1/cutouts/stack", params={"ra": 10, "dec": 10, "pm_ra_masyr": 5, "pm_dec_masyr": 1,
                                                       "epoch": 1700}).status_code == 422


# ---------------------------------------------------------------------------
# Regressions: CLI exit codes
# ---------------------------------------------------------------------------


def test_cli_degraded_rendering_exits_3_and_saves_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    out = tmp_path / "vlass.png"
    argv = ["cutout", "--ra", "83.63308", "--dec", "22.0145", "--fov", "2", "--survey", "vlass",
            "--width", "64", "--height", "64", "--out", str(out)]
    args = cli_parser().parse_args(argv)
    with replay_router("vlass_crab_png"):  # alasky HTTP 500, then a blank PNG from alaskybis; the MOC has data
        assert args.handler(args) == imaging.EXIT_DEGRADED == 3
    assert not out.exists()
    assert "rendering degraded, nothing saved" in capsys.readouterr().err
    allowed = cli_parser().parse_args(argv + ["--allow-degraded"])
    with replay_router("vlass_crab_png"):
        assert allowed.handler(allowed) == 0
    assert out.read_bytes() == body("vlass_crab_png", 1)
    assert "Warning: rendering degraded" in capsys.readouterr().err


def test_cli_rejects_an_unknown_output_extension(tmp_path: Path, capsys) -> None:
    for name in ("x.tiff", "noext"):
        args = cli_parser().parse_args(["cutout", "--ra", "1", "--dec", "1", "--out", str(tmp_path / name)])
        assert args.handler(args) == 2
        assert "cannot tell the image format" in capsys.readouterr().err
        assert not (tmp_path / name).exists()


def test_cli_reports_fits_pixel_units(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "2",
                                    "--survey", "2mass_k", "--width", "32", "--height", "32",
                                    "--out", str(tmp_path / "k.fits")])
    with replay_router("2mass_k_3c273_fits"):
        assert args.handler(args) == 0
    assert "FITS pixels: DN (not flux-calibrated)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Live: real hips2fits / MocServer answers and astrophysical truth
# ---------------------------------------------------------------------------


CANARY = {"hips": "CDS/P/DSS2/color", "width": "32", "height": "32", "fov": "0.05", "projection": "TAN",
          "ra": "187.2779154", "dec": "2.0523883", "format": "png"}


async def hips2fits_is_down() -> bool:
    """True only when a known-good request also fails on every endpoint.

    hips2fits answers malformed parameters with HTTP 500 too, so a 5xx on our own
    request is an outage only if this canary fails as well; otherwise the test fails.
    """
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for url in imaging.HIPS2FITS_URLS:
            try:
                response = await client.get(url, params=CANARY)
            except httpx.HTTPError:
                continue
            if response.status_code == 200 and response.content.startswith(b"\x89PNG"):
                return False
    return True


async def live_cutout(request: CutoutRequest, tmp_path: Path, **service_options) -> imaging.Cutout:
    """Fetch live; skip only on a real outage (transport errors / 5xx *and* a failing canary).

    An unusable answer ("hips2fits returned no usable image") is a failure, never a skip.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            service = CutoutService(client, cache=CutoutCache(tmp_path, ttl_seconds=0), **service_options)
            return await service.cutout(request)
    except CutoutUpstreamError as exc:
        if str(exc).startswith("hips2fits unavailable") and await hips2fits_is_down():
            pytest.skip(f"hips2fits unreachable (canary failed too): {exc}")
        raise


def skip_if_mocserver_down(*results: imaging.CoverageResult) -> None:
    """Skip on a MocServer outage only; an answer that does not parse or a refused query fails."""
    for result in results:
        if not result.checked:
            if result.outage:
                pytest.skip(result.error)
            pytest.fail(f"MocServer answered but the footprint is unknown ({result.error_kind}): {result.error}")


@pytest.mark.live
async def test_live_dss2_png_decodes_with_requested_size(tmp_path: Path) -> None:
    from PIL import Image

    cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=3.0,
                                          width=256, height=192), tmp_path)
    image = Image.open(io.BytesIO(cut.content))
    image.load()
    assert image.format == "PNG" and image.size == (256, 192)
    pixels = np.asarray(image.convert("RGB"), dtype=float)
    assert cut.coverage_fraction == 1.0
    assert pixels.std() > 5, "a real DSS2 field is not a flat image"
    # 3C 273 (V ~ 12.9) is the brightest thing near the field centre.
    lum = pixels.mean(axis=2)
    cy, cx = np.array(lum.shape) // 2
    assert lum[cy - 6:cy + 6, cx - 6:cx + 6].max() >= np.percentile(lum, 99)


@pytest.mark.live
async def test_live_nvss_fits_finds_3c273_at_tens_of_jansky(tmp_path: Path) -> None:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="nvss", fov_arcmin=8.0,
                                          width=96, height=96, format="fits"), tmp_path)
    with fits.open(io.BytesIO(cut.content)) as hdul:
        data, header = hdul[0].data, hdul[0].header
    assert data.shape == (96, 96)
    peak = np.unravel_index(np.nanargmax(data), data.shape)
    assert data[peak] > 20.0
    sep = WCS(header).pixel_to_world(peak[1], peak[0]).separation(SkyCoord(*C3C273, unit="deg")).arcsec
    assert sep < 15.0
    # The 45" NVSS beam is much narrower than the 8' field: the median sky is ~0 Jy/beam.
    assert abs(float(np.nanmedian(data))) < 0.05


@pytest.mark.live
async def test_live_panstarrs_jpeg_non_square(tmp_path: Path) -> None:
    from PIL import Image

    cut = await live_cutout(CutoutRequest(ra=M87[0], dec=M87[1], survey="panstarrs", fov_arcmin=4.0,
                                          width=300, height=150, format="jpg"), tmp_path)
    image = Image.open(io.BytesIO(cut.content))
    image.load()
    assert image.format == "JPEG" and image.size == (300, 150)
    # M87's bright core sits at the centre of the frame.
    lum = np.asarray(image.convert("L"), dtype=float)
    assert lum[65:85, 140:160].mean() > np.median(lum) + 30


@pytest.mark.live
async def test_live_blank_outside_sdss_footprint(tmp_path: Path) -> None:
    cut = await live_cutout(CutoutRequest(ra=SOUTH[0], dec=SOUTH[1], survey="sdss", fov_arcmin=4.0,
                                          width=64, height=64, format="fits"), tmp_path)
    assert cut.coverage_fraction == 0.0


@pytest.mark.live
async def test_live_every_default_stack_survey_renders_at_3c273(tmp_path: Path) -> None:
    """Each HiPS the stack can pick is known to hips2fits and returns data at 3C 273 when covered."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        panels, coverage = await CutoutService(client, cache=CutoutCache(tmp_path, ttl_seconds=0)).plan_stack(*C3C273)
    skip_if_mocserver_down(coverage)
    assert [p.survey for p in panels] == ["tgss", "nvss", "allwise", "2mass", "panstarrs", "galex", "erosita", "xmm"]
    for panel in panels:
        cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey=panel.survey, fov_arcmin=3.0,
                                              width=64, height=64, format="png"), tmp_path)
        assert (cut.width, cut.height) == (64, 64)
        assert cut.coverage_fraction is not None and cut.coverage_fraction > 0.5, panel.survey


@pytest.mark.live
async def test_live_single_band_fits_companions_hold_survey_pixels(tmp_path: Path) -> None:
    """The FITS link of every colour panel at 3C 273 is a single-band float image, not an RGB cube.

    These are survey pixel values in the units each survey declares (``pixel_units``), not
    necessarily flux-calibrated: hips2fits writes no BUNIT or zero point.
    """
    from astropy.io import fits

    for key in ("allwise", "2mass", "panstarrs", "galex", "xmm", "sdss", "legacy", "unwise"):
        science = imaging.science_survey(SURVEYS[key])
        assert science is not None and not science.color, key
        cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey=science.key, fov_arcmin=3.0,
                                              width=48, height=48, format="fits"), tmp_path)
        with fits.open(io.BytesIO(cut.content)) as hdul:
            data = hdul[0].data
        assert data.shape == (48, 48) and data.dtype.kind == "f", (key, data.shape, data.dtype)
        assert cut.coverage_fraction is not None and cut.coverage_fraction > 0.9, key
        finite = data[np.isfinite(data)]
        assert finite.max() > np.median(finite), key  # a real field, not a constant plane
    colour = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass", fov_arcmin=3.0,
                                             width=48, height=48, format="fits"), tmp_path)
    with fits.open(io.BytesIO(colour.content)) as hdul:
        cube = hdul[0].data
    assert cube.shape == (4, 48, 48) and cube.dtype.itemsize == 1  # the display cube the UI no longer offers
    assert colour.pixel_kind == "rgb-preview"
    # DSS2 red: 16-bit integer plate densities (thousands), which is why it is not called calibrated.
    red = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2_red", fov_arcmin=3.0,
                                          width=48, height=48, format="fits"), tmp_path)
    with fits.open(io.BytesIO(red.content)) as hdul:
        plate, header = hdul[0].data, hdul[0].header
    assert header["BITPIX"] == 16 and plate.dtype.kind == "i" and np.median(plate) > 500
    assert "BUNIT" not in header and not SURVEYS["dss2_red"].calibrated


@pytest.mark.live
async def test_live_2mass_ks_peak_is_3c273(tmp_path: Path) -> None:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="2mass_k", fov_arcmin=2.0,
                                          width=64, height=64, format="fits"), tmp_path)
    with fits.open(io.BytesIO(cut.content)) as hdul:
        data, header = hdul[0].data, hdul[0].header
    peak = np.unravel_index(np.nanargmax(data), data.shape)
    sep = WCS(header).pixel_to_world(peak[1], peak[0]).separation(SkyCoord(*C3C273, unit="deg")).arcsec
    assert sep < 5.0  # 2MASS pixels are 1"; the quasar is the brightest Ks source in the field


@pytest.mark.live
async def test_live_all_sky_mollweide(tmp_path: Path) -> None:
    """fov = 360 deg in MOL renders the whole sky (the limit the old validation rejected)."""
    from PIL import Image

    cut = await live_cutout(CutoutRequest(ra=0.0, dec=0.0, survey="dss2", fov_arcmin=21600.0, projection="MOL",
                                          width=128, height=64, format="png"), tmp_path)
    image = Image.open(io.BytesIO(cut.content))
    image.load()
    assert image.size == (128, 64)
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    # A Mollweide ellipse fills pi/4 of its bounding box; the corners are outside the sky.
    assert 0.70 < (alpha > 0).mean() < 0.86
    assert alpha[0, 0] == 0 and alpha[32, 64] > 0


@pytest.mark.live
async def test_live_vlass_footprint_comes_from_its_moc() -> None:
    """MocServer answers spatial queries for the VLASS Quick Look HiPS: no data at 3C 273, data at the Crab."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        service = CutoutService(client, cache=CutoutCache(Path("."), ttl_seconds=0))
        at_3c273 = await service.coverage(*C3C273, [SURVEYS["vlass"]])
        at_crab = await service.coverage(83.63308, 22.0145, [SURVEYS["vlass"]])
    skip_if_mocserver_down(at_3c273, at_crab)
    assert at_3c273.covered == {"vlass": False}
    assert at_crab.covered == {"vlass": True}


@pytest.mark.live
async def test_live_describe_adhoc_hips() -> None:
    ids = ["CDS/P/Mellinger/color", "CDS/P/Fermi/color", "BOGUS/P/nothing"]
    async with httpx.AsyncClient(timeout=60.0) as client:
        described = await CutoutService(client, cache=CutoutCache(Path("."), ttl_seconds=0)).describe_hips(ids)
        if not described:
            # describe_hips swallows failures by design; only a real outage may skip this test.
            try:
                raw = await client.get(imaging.MOCSERVER_URLS[0], params={
                    "expr": "||".join(f"ID={h}" for h in ids), "get": "record", "fmt": "json"})
            except httpx.HTTPError as exc:
                pytest.skip(f"MocServer unreachable: {exc}")
            if raw.status_code >= 500:
                pytest.skip(f"MocServer HTTP {raw.status_code}")
            pytest.fail(f"MocServer answered HTTP {raw.status_code} but nothing was parsed: {raw.text[:300]}")
    assert described["CDS/P/Mellinger/color"].regime == "optical"
    assert described["CDS/P/Mellinger/color"].bib_reference == "2009PASP..121.1180M"
    assert described["CDS/P/Fermi/color"].regime == "gamma"  # Fermi-LAT: GeV photons
    assert "BOGUS/P/nothing" not in described


@pytest.mark.live
def test_live_router_rejects_bad_cuts_before_hips2fits(tmp_path: Path) -> None:
    """These parameters make hips2fits answer HTTP 500; the API answers 422 without calling it."""
    client = TestClient(make_app(tmp_path))
    for params in ({"min_cut": "200%"}, {"min_cut": "5", "max_cut": "1"}):
        resp = client.get("/api/v1/cutouts", params={"ra": 83.63308, "dec": 22.0145, "fov_arcmin": 3, "width": 64,
                                                     "height": 64, **params})
        assert resp.status_code == 422, resp.text


@pytest.mark.live
async def test_live_mocserver_footprints() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        service = CutoutService(client, cache=CutoutCache(Path("."), ttl_seconds=0))
        south = await service.coverage(*SOUTH, [SURVEYS["nvss"], SURVEYS["racs_low"], SURVEYS["sdss"], SURVEYS["2mass"]])
        north = await service.coverage(*M87, [SURVEYS["nvss"], SURVEYS["sdss"], SURVEYS["chandra"]])
    skip_if_mocserver_down(south, north)
    assert south.covered == {"nvss": False, "racs_low": True, "sdss": False, "2mass": True}
    # M87 is in SDSS and NVSS, and one of the most observed Chandra targets.
    assert north.covered == {"nvss": True, "sdss": True, "chandra": True}


@pytest.mark.live
def test_live_router_end_to_end(tmp_path: Path) -> None:
    from PIL import Image

    client = TestClient(make_app(tmp_path))
    resp = client.get("/api/v1/cutouts", params={"name": "M87", "fov_arcmin": 2, "survey": "2mass", "width": 128,
                                                 "height": 128})
    if resp.status_code == 502 and "hips2fits unavailable" in resp.text and asyncio.run(hips2fits_is_down()):
        pytest.skip(resp.text)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-cutout-coverage"] == "1.0000" and resp.headers["x-cutout-pixels"] == "display"
    image = Image.open(io.BytesIO(resp.content))
    assert image.size == (128, 128)
    assert resp.headers["x-cutout-cache"] == "miss"
    again = client.get("/api/v1/cutouts", params={"name": "M87", "fov_arcmin": 2, "survey": "2mass", "width": 128,
                                                  "height": 128})
    assert again.headers["x-cutout-cache"] == "hit" and again.content == resp.content


@pytest.mark.live
async def test_live_registry_matches_the_survey_catalogue() -> None:
    """Every catalogued wavelength range and FITS companion agrees with the CDS MocServer registry."""
    ids = [s.hips_id for s in SURVEYS.values()] + ["astron.nl/P/tgssadr"]
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(imaging.MOCSERVER_URLS[0], params={
                "expr": "||".join(f"ID={h}" for h in ids), "get": "record", "fmt": "json",
                "fields": "ID,em_min,em_max,hips_tile_format,moc_sky_fraction,hips_bunit"})
    except httpx.HTTPError as exc:
        pytest.skip(f"MocServer unreachable: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"MocServer HTTP {response.status_code}")
    records = {r["ID"]: r for r in response.json()}
    assert set(ids) <= set(records), f"not in the registry: {set(ids) - set(records)}"
    tgss = records["astron.nl/P/tgssadr"]
    assert (float(tgss["em_min"]), float(tgss["em_max"])) == imaging._TGSS_EM
    assert 299792458 / math.sqrt(imaging._TGSS_EM[0] * imaging._TGSS_EM[1]) == pytest.approx(147.5e6, rel=1e-3)
    for key, survey in SURVEYS.items():
        record = records[survey.hips_id]
        if "em_min" in record and key != "tgss":
            lo, hi = sorted((float(record["em_min"]), float(record["em_max"])))
            assert (survey.em_min_m, survey.em_max_m) == (pytest.approx(lo, rel=1e-4), pytest.approx(hi, rel=1e-4)), key
        if not survey.color:  # single-band surveys (FITS companions) must have FITS tiles
            assert "fits" in str(record.get("hips_tile_format")).split(), key
        if survey.coverage == "full":
            assert float(record["moc_sky_fraction"]) == 1.0, key
    # GALEX's registry hips_bunit strings are what the module documents (and deliberately does not use:
    # the pixels are count rates, see test_live_galex_pixels_follow_the_morrissey_zero_points).
    for key, bunit in imaging.GALEX_REGISTRY_BUNITS.items():
        assert records[SURVEYS[key].hips_id]["hips_bunit"] == bunit, key
        assert bunit not in SURVEYS[key].pixel_note and SURVEYS[key].pixel_units == "counts/s", key
    # Legacy DR10 r covers much less sky than the colour HiPS: the reason for the FITS fallbacks.
    assert float(records[SURVEYS["legacy_r"].hips_id]["moc_sky_fraction"]) < float(
        records[SURVEYS["legacy"].hips_id]["moc_sky_fraction"]) - 0.1


@pytest.mark.live
async def test_live_legacy_gap_links_a_band_with_data(tmp_path: Path) -> None:
    """At (345, -35) the DR10 colour HiPS has data but DR10 r does not: the FITS link uses g, which has data."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        panels, coverage = await CutoutService(client, cache=CutoutCache(tmp_path, ttl_seconds=0)).plan_stack(345.0, -35.0)
    skip_if_mocserver_down(coverage)
    assert coverage.covered["legacy"] is True and coverage.covered["legacy_r"] is False
    legacy = next(p for p in panels if p.survey == "legacy")
    assert legacy.fits_survey in ("legacy_g", "legacy_z", "legacy_i") and legacy.fits_note
    fits_cut = await live_cutout(CutoutRequest(ra=345.0, dec=-35.0, survey=legacy.fits_survey, fov_arcmin=3.0,
                                               width=48, height=48, format="fits"), tmp_path)
    assert fits_cut.coverage_fraction is not None and fits_cut.coverage_fraction > 0.9
    # The r band really is blank there, and the MOC confirms it: a genuine (cacheable) 'no data'.
    r_cut = await live_cutout(CutoutRequest(ra=345.0, dec=-35.0, survey="legacy_r", fov_arcmin=3.0,
                                            width=48, height=48, format="fits"), tmp_path)
    assert r_cut.coverage_fraction == 0.0 and r_cut.degraded is None and r_cut.cacheable


def _aperture_peak(image: np.ndarray, wcs, position, radius_px: float = 4.0) -> float:
    """Brightest pixel within ``radius_px`` of a sky position (asserts the position is on the image)."""
    x, y = wcs.world_to_pixel(position)
    rows, cols = np.mgrid[:image.shape[0], :image.shape[1]]
    inside = (cols - float(x)) ** 2 + (rows - float(y)) ** 2 <= radius_px ** 2
    assert inside.any(), "position outside the panel"
    return float(np.nanmax(image[inside]))


@pytest.mark.live
def test_live_barnards_star_panels_follow_its_proper_motion(tmp_path: Path) -> None:
    """Barnard's star (10.4"/yr north) lies ~125" from its J2000 position in Pan-STARRS1 (2009-2014)
    and ~110" in AllWISE (2010-2011): the panels are centred where the star actually is.

    Checked on 6' single-band FITS panels that contain both positions: the star's light sits at
    the panel centre, while the J2000 position is empty sky. (Measured 2026-09-28: PS1 g peak
    1526 counts at the centre vs 10 at J2000; AllWISE W1 1184 DN vs 1.2.)
    """
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    client = TestClient(make_app(tmp_path))
    resp = client.get("/api/v1/cutouts/stack", params={"name": "Barnard's star", "fov_arcmin": 6, "size": 180,
                                                       "surveys": "panstarrs,allwise"})
    if resp.status_code == 502:
        pytest.skip(resp.text)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["proper_motion"]["pm_dec_masyr"] == pytest.approx(10362.394, abs=5.0)
    j2000 = SkyCoord(data["target"]["ra"], data["target"]["dec"], unit="deg")
    assert {p["survey"] for p in data["panels"]} == {"panstarrs", "allwise"}
    for panel in data["panels"]:
        centre = SkyCoord(panel["center_ra"], panel["center_dec"], unit="deg")
        assert centre.separation(j2000).arcsec > 90.0, panel["survey"]  # outside a 3' field around J2000
        assert panel["fits_survey"] in ("panstarrs_g", "allwise_w1")
        fits_resp = client.get(panel["fits_url"])
        if fits_resp.status_code == 502 and asyncio.run(hips2fits_is_down()):
            pytest.skip(fits_resp.text)
        assert fits_resp.status_code == 200, fits_resp.text
        with fits.open(io.BytesIO(fits_resp.content)) as hdul:
            image, wcs = hdul[0].data.astype(float), WCS(hdul[0].header)
        at_centre, at_j2000 = _aperture_peak(image, wcs, centre), _aperture_peak(image, wcs, j2000)
        background = abs(float(np.nanmedian(image)))
        assert at_centre > 10 * max(at_j2000, background, 1.0), (panel["survey"], at_centre, at_j2000, background)


@pytest.mark.live
def test_live_router_cut_diagnosis(tmp_path: Path) -> None:
    """Lone percentiles past the other default are rejected locally; a crossing value/percentile pair
    makes hips2fits answer 500, which the API reports as 422 (bad cuts), not as an outage."""
    client = TestClient(make_app(tmp_path))
    base = {"ra": 83.63308, "dec": 22.0145, "fov_arcmin": 3, "survey": "nvss", "width": 64, "height": 64}
    for params in ({"min_cut": "99.6%"}, {"max_cut": "0.4%"}, {"min_cut": "100%"}, {"min_cut": "1E2%"}):
        resp = client.get("/api/v1/cutouts", params={**base, **params})
        assert resp.status_code == 422, (params, resp.text)
    for params in ({"min_cut": "99.5%"}, {"max_cut": "0.5%"}, {"min_cut": "5%", "max_cut": "5%"}):
        resp = client.get("/api/v1/cutouts", params={**base, **params})
        if resp.status_code == 502 and asyncio.run(hips2fits_is_down()):
            pytest.skip(resp.text)
        assert resp.status_code == 200, (params, resp.text)
    crossing = client.get("/api/v1/cutouts", params={**base, "min_cut": "1%", "max_cut": "0.02"})
    if crossing.status_code == 502 and asyncio.run(hips2fits_is_down()):
        pytest.skip(crossing.text)
    assert crossing.status_code == 422, crossing.text
    assert "cannot render this cutout" in crossing.json()["detail"]


@pytest.mark.live
async def test_live_vlass_blank_where_its_moc_has_data_is_never_cached(tmp_path: Path) -> None:
    """With only alaskybis (as HIPS2FITS_URLS may configure), VLASS at the Crab: either real pixels,
    or a blank image flagged degraded (its MOC covers the Crab) and kept out of the cache."""
    cache = CutoutCache(tmp_path / "c", ttl_seconds=3600)
    request = CutoutRequest(ra=83.63308, dec=22.0145, survey="vlass", fov_arcmin=2.0, width=64, height=64)
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            cut = await CutoutService(client, cache=cache, endpoints=[imaging.HIPS2FITS_URLS[1]]).cutout(request)
    except CutoutUpstreamError as exc:
        if str(exc).startswith("hips2fits unavailable") and await hips2fits_is_down():
            pytest.skip(str(exc))
        raise
    assert cut.coverage_fraction is not None
    if cut.coverage_fraction == 0.0:
        assert cut.degraded and "overlaps this field" in cut.degraded
        assert cut.headers()["Cache-Control"] == "no-store" and cache.size_bytes() == 0
    else:
        assert cut.degraded is None and cache.size_bytes() == len(cut.content)


@pytest.mark.live
async def test_live_adhoc_colour_hips_fits_is_labelled_rgb_preview(tmp_path: Path) -> None:
    cut = await live_cutout(CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="CDS/P/Mellinger/color",
                                          fov_arcmin=60.0, width=32, height=32, format="fits"), tmp_path)
    assert cut.info is not None and (cut.info.planes, cut.info.bitpix) == (4, 8)
    assert cut.pixel_kind == "rgb-preview" and cut.headers()["X-Cutout-Pixels"] == "rgb-preview"
