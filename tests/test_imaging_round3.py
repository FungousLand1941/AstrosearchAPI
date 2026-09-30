"""Imaging regressions (round 3): blank JPEGs, blank failover, --no-cache, colormaps, pixel scale,
colour-FITS companions, cache metadata, Sesame failures, unconfirmed blanks and survey metadata.

Offline tests replay real hips2fits / MocServer / Sesame answers recorded on 2026-09-28
(``tests/fixtures/imaging/record_imaging.py``); live tests (``-m live``) check the same
behaviour against CDS.
"""

from __future__ import annotations

import asyncio
import io
import math
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient
from fixture_io import FIXTURES
from helpers import offline_client
from live_policy import api_ok
from test_imaging import (
    C3C273,
    _png,
    body,
    cli_parser,
    hips2fits_is_down,
    live_cutout,
    make_app,
    replay_router,
)

import imaging
from imaging import SURVEYS, CutoutCache, CutoutRequest, CutoutService, CutoutValidationError

CRAB = (83.63308, 22.0145)
SOUTH = (100.0, -70.0)
LEGACY_GAP = (345.0, -35.0)
SESAME_UNKNOWN = (FIXTURES / "imaging" / "sesame_unknown.xml").read_text(encoding="utf-8")


@pytest.fixture
def cache(tmp_path: Path) -> CutoutCache:
    return CutoutCache(tmp_path / "cutouts", ttl_seconds=3600, max_bytes=10 * 1024 * 1024)


def vlass_crab_jpg() -> CutoutRequest:
    return CutoutRequest(ra=CRAB[0], dec=CRAB[1], survey="vlass", fov_arcmin=2.0, width=64, height=64, format="jpg")


def ps1_south_jpg() -> CutoutRequest:
    return CutoutRequest(ra=SOUTH[0], dec=SOUTH[1], survey="panstarrs", fov_arcmin=6.0, width=64, height=64,
                         format="jpg")


def rass_flat_jpg() -> CutoutRequest:
    return CutoutRequest(ra=187.0, dec=40.0, survey="rass", fov_arcmin=0.2, width=32, height=32, format="jpg",
                         cmap="Greys")


def legacy_gap_fits() -> CutoutRequest:
    return CutoutRequest(ra=LEGACY_GAP[0], dec=LEGACY_GAP[1], survey="legacy", fov_arcmin=1.0, width=16, height=16,
                         format="fits")


def calls(router: respx.MockRouter) -> list[tuple[str, str, str | None]]:
    return [(c.request.url.host, c.request.url.path.split("/")[1], c.request.url.params.get("format"))
            for c in router.calls]


HIPS, MOC = "hips-image-services", "MocServer"
ALASKY, BIS = "alasky.cds.unistra.fr", "alaskybis.cds.unistra.fr"


# ---------------------------------------------------------------------------
# Blank JPEGs (hips2fits renders "no data" and failures as uniform white JPEGs)
# ---------------------------------------------------------------------------


def test_uniform_jpeg_detection_on_recorded_images() -> None:
    white = body("vlass_crab_jpg", 1)
    pixels = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(white)))
    assert pixels.min() == pixels.max() == 255  # the recorded VLASS failure: every RGB value 255
    assert imaging.is_uniform_raster(white)
    assert imaging.is_uniform_raster(body("ps1_south_jpg", 0))
    assert imaging.is_uniform_raster(body("rass_flat_greys_jpg", 0))  # real data, flat: uniform too
    assert not imaging.is_uniform_raster(body("panstarrs_m87_jpg"))
    assert not imaging.is_uniform_raster(body("dss2_3c273_png"))
    # The JPEG alone cannot tell no-data from data: coverage stays unknown at this level.
    assert imaging.coverage_fraction(white, "jpg") is None


async def test_blank_jpeg_rendering_failure_is_degraded_and_never_cached(cache: CutoutCache) -> None:
    """Recorded VLASS at the Crab as JPEG: alasky HTTP 500, alaskybis a uniformly white JPEG; the VLASS
    MOC overlaps the field and the same cutout as PNG from alaskybis is fully transparent."""
    with replay_router("vlass_crab_jpg") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(vlass_crab_jpg())
            again = await service.cutout(vlass_crab_jpg())
        sequence = calls(router)
    one = [(ALASKY, HIPS, "jpg"), (BIS, HIPS, "jpg"), (ALASKY, MOC, None), (BIS, HIPS, "png")]
    assert sequence == one * 2, "the blank JPEG must not be served from the cache"
    assert not again.cached
    assert cut.endpoint == imaging.HIPS2FITS_URLS[1] and cut.coverage_fraction == 0.0
    assert cut.blank == "rendering-failure" and not cut.cacheable
    assert cut.degraded.startswith("blank (uniform) JPEG from mirror alaskybis.cds.unistra.fr after "
                                   "alasky.cds.unistra.fr: HTTP 500")
    assert cut.degraded.endswith("the VLASS 3 GHz footprint (CDS MocServer MOC) overlaps this field")
    headers = cut.headers()
    assert headers["Cache-Control"] == "no-store" and headers["X-Cutout-Blank"] == "rendering-failure"
    assert headers["X-Cutout-Coverage"] == "0.0000" and "X-Cutout-Degraded" in headers
    assert cache.size_bytes() == 0


def test_router_blank_jpeg_is_flagged_and_never_cached(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    params = {"ra": CRAB[0], "dec": CRAB[1], "fov_arcmin": 2.0, "survey": "vlass", "width": 64, "height": 64,
              "format": "jpg"}
    with replay_router("vlass_crab_jpg", passthrough_testserver=True):
        first = client.get("/api/v1/cutouts", params=params)
        second = client.get("/api/v1/cutouts", params=params)
    for resp in (first, second):
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
        assert resp.headers["x-cutout-cache"] == "miss" and resp.headers["cache-control"] == "no-store"
        assert resp.headers["x-cutout-blank"] == "rendering-failure" and resp.headers["x-cutout-degraded"]
    assert not any((tmp_path / "api-cache").glob("*.bin"))


def test_cli_blank_jpeg_exits_3_and_caches_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "cli-cache"))
    out = tmp_path / "crab_vlass.jpg"
    argv = ["cutout", "--ra", str(CRAB[0]), "--dec", str(CRAB[1]), "--fov", "2", "--survey", "vlass",
            "--width", "64", "--height", "64", "--out", str(out)]
    for _ in range(2):  # a second run must not find a cached copy either
        args = cli_parser().parse_args(argv)
        with replay_router("vlass_crab_jpg"):
            assert args.handler(args) == imaging.EXIT_DEGRADED
        assert "rendering degraded, nothing saved" in capsys.readouterr().err
    assert not out.exists()
    assert not any((tmp_path / "cli-cache").glob("*.bin"))


async def test_uniform_jpeg_outside_the_footprint_is_a_confirmed_no_data(cache: CutoutCache) -> None:
    """PS1 never observed (100, -70): a white JPEG whose MOC check confirms 'no data' (no PNG needed)."""
    with replay_router("ps1_south_jpg") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(ps1_south_jpg())
            again = await service.cutout(ps1_south_jpg())
        sequence = calls(router)
    assert sequence == [(ALASKY, HIPS, "jpg"), (ALASKY, MOC, None)]
    assert cut.blank == "no-data" and cut.coverage_fraction == 0.0 and cut.degraded is None and cut.cacheable
    assert cut.headers()["X-Cutout-Blank"] == "no-data" and cut.headers()["Cache-Control"] == "public, max-age=86400"
    assert again.cached and again.blank == "no-data" and again.coverage_fraction == 0.0


async def test_flat_but_real_jpeg_is_kept_as_data(cache: CutoutCache) -> None:
    """RASS counts are all 0 in this 12" field; with cmap=Greys the JPEG is as white as a blank. RASS is
    all-sky (no MOC query), and the PNG's alpha plane shows every pixel carries data."""
    with replay_router("rass_flat_greys_jpg") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(rass_flat_jpg())
            again = await service.cutout(rass_flat_jpg())
        sequence = calls(router)
    assert sequence == [(ALASKY, HIPS, "jpg"), (ALASKY, HIPS, "png")]
    assert imaging.is_uniform_raster(cut.content)
    assert cut.coverage_fraction == 1.0 and cut.blank is None and cut.degraded is None and cut.cacheable
    assert again.cached and again.coverage_fraction == 1.0 and again.blank is None


async def test_uniform_jpeg_whose_png_check_fails_is_not_trusted(cache: CutoutCache) -> None:
    white = body("vlass_crab_jpg", 1)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        for url in imaging.HIPS2FITS_URLS:
            router.get(url__startswith=url, params__contains={"format": "jpg"}).respond(
                200, content=white, headers={"content-type": "image/jpeg"})
            router.get(url__startswith=url, params__contains={"format": "png"}).respond(503)
        router.get(url__startswith=imaging.MOCSERVER_URLS[0]).respond(
            200, content=body("vlass_crab_png", 2), headers={"content-type": "application/json"})
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(vlass_crab_jpg())
    assert cut.coverage_fraction is None and cut.blank == "rendering-failure" and cut.degraded
    assert cut.degraded.startswith("blank (uniform) JPEG from alasky.cds.unistra.fr and alaskybis.cds.unistra.fr "
                                   "although the VLASS 3 GHz footprint")
    assert cache.size_bytes() == 0


# ---------------------------------------------------------------------------
# Blank failover to the mirror
# ---------------------------------------------------------------------------


def _blank_then(mirror_status: int, mirror_body: bytes | None, *, moc: bytes | int) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.get(url__startswith=imaging.HIPS2FITS_URLS[0]).respond(
        200, content=body("vlass_crab_png_bis"), headers={"content-type": "image/png"})
    router.get(url__startswith=imaging.HIPS2FITS_URLS[1]).respond(
        mirror_status, content=mirror_body or b"", headers={"content-type": "image/png"})
    for url in imaging.MOCSERVER_URLS:
        if isinstance(moc, int):
            router.get(url__startswith=url).respond(moc)
        else:
            router.get(url__startswith=url).respond(200, content=moc, headers={"content-type": "application/json"})
    return router


async def test_blank_primary_fails_over_to_a_good_mirror(cache: CutoutCache) -> None:
    good = _png(64, 64)
    request = CutoutRequest(ra=CRAB[0], dec=CRAB[1], survey="vlass", fov_arcmin=2.0, width=64, height=64)
    with _blank_then(200, good, moc=body("vlass_crab_png", 2)) as router:  # the MOC lists VLASS here
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
        sequence = calls(router)
    assert sequence == [(ALASKY, HIPS, "png"), (ALASKY, MOC, None), (BIS, HIPS, "png")]
    assert cut.endpoint == imaging.HIPS2FITS_URLS[1] and cut.content == good
    assert cut.coverage_fraction == 1.0 and cut.degraded is None and cut.blank is None and cut.cacheable
    assert cache.size_bytes() == len(good)


async def test_blank_on_both_endpoints_where_the_moc_has_data_is_degraded(cache: CutoutCache) -> None:
    request = CutoutRequest(ra=CRAB[0], dec=CRAB[1], survey="vlass", fov_arcmin=2.0, width=64, height=64)
    with _blank_then(200, body("vlass_crab_png_bis"), moc=body("vlass_crab_png", 2)) as router:
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
        sequence = calls(router)
    assert sequence == [(ALASKY, HIPS, "png"), (ALASKY, MOC, None), (BIS, HIPS, "png")], "one MOC query per cutout"
    assert cut.endpoint == imaging.HIPS2FITS_URLS[0] and cut.blank == "rendering-failure"
    assert cut.degraded == ("blank image from alasky.cds.unistra.fr and alaskybis.cds.unistra.fr although the "
                            "VLASS 3 GHz footprint (CDS MocServer MOC) overlaps this field")
    assert cache.size_bytes() == 0


async def test_blank_primary_then_failing_mirror_is_degraded(cache: CutoutCache) -> None:
    request = CutoutRequest(ra=CRAB[0], dec=CRAB[1], survey="vlass", fov_arcmin=2.0, width=64, height=64)
    with _blank_then(503, None, moc=body("vlass_crab_png", 2)):
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
    assert cut.endpoint == imaging.HIPS2FITS_URLS[0] and cut.blank == "rendering-failure"
    assert cut.degraded.startswith("blank image from alasky.cds.unistra.fr after alaskybis.cds.unistra.fr: HTTP 503")
    assert cache.size_bytes() == 0


# ---------------------------------------------------------------------------
# Unconfirmed blanks are never presented as "no data"
# ---------------------------------------------------------------------------


def test_cli_and_router_word_unconfirmed_blanks_as_unconfirmed(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "cli-cache"))
    blank = body("vlass_crab_png_bis")
    out = tmp_path / "v.png"
    args = cli_parser().parse_args(["cutout", "--ra", str(CRAB[0]), "--dec", str(CRAB[1]), "--fov", "2",
                                    "--survey", "vlass", "--width", "64", "--height", "64", "--out", str(out)])
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        for url in imaging.HIPS2FITS_URLS:
            router.get(url__startswith=url).respond(200, content=blank, headers={"content-type": "image/png"})
        for url in imaging.MOCSERVER_URLS:
            router.get(url__startswith=url).respond(503)
        assert args.handler(args) == 0  # the image may be right: saved, but not called 'no data'
        resp = client.get("/api/v1/cutouts", params={"ra": CRAB[0], "dec": CRAB[1], "fov_arcmin": 2, "survey": "vlass",
                                                     "width": 64, "height": 64})
    err = capsys.readouterr().err
    assert "footprint could not be checked" in err and "has no data at this position" not in err
    assert out.read_bytes() == blank and not any((tmp_path / "cli-cache").glob("*.bin"))
    assert resp.status_code == 200 and resp.headers["x-cutout-blank"] == "unconfirmed"
    assert resp.headers["cache-control"] == "no-store" and "x-cutout-degraded" not in resp.headers


def test_cli_confirmed_no_data_says_so(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    args = cli_parser().parse_args(["cutout", "--ra", "100", "--dec", "-70", "--fov", "6", "--survey", "panstarrs",
                                    "--width", "64", "--height", "64", "--out", str(tmp_path / "s.jpg")])
    with replay_router("ps1_south_jpg"):
        assert args.handler(args) == 0
    captured = capsys.readouterr()
    assert "data coverage 0.0%" in captured.out
    assert "the survey has no data at this position" in captured.err and "MOC does not reach" in captured.err


# ---------------------------------------------------------------------------
# --no-cache / use_cache=False
# ---------------------------------------------------------------------------


async def test_use_cache_false_neither_reads_nor_writes(cache: CutoutCache) -> None:
    with replay_router("dss2_3c273_png") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=128, height=128)
            fresh = await service.cutout(request, use_cache=False)
            assert cache.size_bytes() == 0, "use_cache=False must not write the cache"
            await service.cutout(request)  # fills the cache
            bypass = await service.cutout(request, use_cache=False)
        count = router.calls.call_count
    assert not fresh.cached and not bypass.cached and count == 3


def test_cli_no_cache_leaves_the_cache_directory_empty(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache2"
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(cache_dir))
    args = cli_parser().parse_args(["cutout", "--ra", str(C3C273[0]), "--dec", str(C3C273[1]), "--fov", "2",
                                    "--survey", "dss2", "--width", "128", "--height", "128", "--no-cache",
                                    "--out", str(tmp_path / "d.png")])
    with replay_router("dss2_3c273_png"):
        assert args.handler(args) == 0
    assert (tmp_path / "d.png").exists()
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["foobar", "Viridis", "okabe_ito", "viridis;rm", "VIRIDIS_r"])
def test_unknown_colormaps_are_rejected(name: str) -> None:
    with pytest.raises(CutoutValidationError, match="not a colormap hips2fits renders"):
        CutoutRequest(ra=10.0, dec=10.0, survey="nvss", cmap=name)


@pytest.mark.parametrize("name", ["viridis", "viridis_r", "Greys_r", "inferno", "cubehelix", "twilight_shifted_r"])
def test_verified_colormaps_are_accepted(name: str) -> None:
    assert CutoutRequest(ra=10.0, dec=10.0, survey="nvss", cmap=name).params()["cmap"] == name


def test_router_rejects_an_unknown_colormap(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        hips = router.route(host__regex=r"^alasky(bis)?\.cds\.unistra\.fr$").respond(500)
        resp = client.get("/api/v1/cutouts", params={"ra": 10, "dec": 10, "survey": "nvss", "cmap": "foobar"})
    assert resp.status_code == 422 and "foobar" in resp.json()["detail"]
    assert not hips.called, "rejected before any upstream request"


# ---------------------------------------------------------------------------
# Pixel scale and CDELT
# ---------------------------------------------------------------------------


# (projection, fov deg, width, height, CDELT1 magnitude hips2fits returned live on 2026-09-28)
LIVE_CDELT = [
    ("TAN", 0.1, 300, 150, 3.333e-4), ("SIN", 0.1, 300, 150, 3.333e-4), ("CAR", 360.0, 360, 180, 1.0),
    ("MOL", 360.0, 400, 200, 0.81028), ("AIT", 360.0, 256, 256, 1.26607), ("TAN", 100.0, 40, 20, 3.41412),
    ("STG", 100.0, 40, 20, 2.67175), ("ZEA", 360.0, 40, 20, 5.12469), ("TSC", 1.0, 40, 20, 0.0196355),
    ("XPH", 100.0, 40, 20, 3.23897), ("AIT", 100.0, 40, 20, 2.48022), ("HPX", 360.0, 40, 20, 9.0),
]


@pytest.mark.parametrize("projection, fov, width, height, cdelt", LIVE_CDELT)
def test_cdelt_matches_what_hips2fits_writes(projection: str, fov: float, width: int, height: int,
                                             cdelt: float) -> None:
    request = CutoutRequest(ra=10.0, dec=20.0, survey="rass", fov_arcmin=fov * 60.0, width=width, height=height,
                            projection=projection)
    assert request.cdelt_deg == pytest.approx(cdelt, rel=2e-4)


def test_pixel_scale_is_the_angular_size_at_the_image_centre() -> None:
    def req(projection: str, fov_deg: float, width: int, height: int) -> CutoutRequest:
        return CutoutRequest(ra=10.0, dec=20.0, survey="rass", fov_arcmin=fov_deg * 60.0, width=width,
                             height=height, projection=projection)

    # Small fields: the nominal fov / size.
    assert req("TAN", 2 / 60, 128, 128).pixel_scale_arcsec == pytest.approx(0.9375, rel=1e-6)
    # Equal-area all-sky projections: CDELT (2 sqrt 2 / pi = 0.9003 of the nominal value at fov 360).
    for projection, width, height in (("MOL", 400, 200), ("AIT", 256, 256)):
        r = req(projection, 360.0, width, height)
        assert r.pixel_scale_arcsec == pytest.approx(r.cdelt_deg * 3600.0, rel=1e-6)
        assert r.pixel_scale_arcsec / r.nominal_pixel_scale_arcsec == pytest.approx(2 * math.sqrt(2) / math.pi,
                                                                                   rel=1e-4)
    # A 100 deg TAN field: pixels at the centre are tan(50 deg)/(50 deg in rad) = 1.366 x the mean.
    wide = req("TAN", 100.0, 40, 20)
    assert wide.pixel_scale_arcsec / wide.nominal_pixel_scale_arcsec == pytest.approx(
        math.tan(math.radians(50)) / math.radians(50), rel=1e-5)
    # TSC: CDELT is in cube-face units (pi/4 per degree at the centre), the angular scale is nominal.
    tsc = req("TSC", 1.0, 40, 20)
    assert tsc.pixel_scale_arcsec == pytest.approx(tsc.nominal_pixel_scale_arcsec, rel=1e-4)


def test_headers_carry_centre_scale_and_cdelt() -> None:
    request = CutoutRequest(ra=10.0, dec=20.0, survey="rass", fov_arcmin=21600.0, width=400, height=200,
                            projection="MOL")
    cut = imaging.Cutout(request, b"", "image/png", 400, 200, "u", imaging.HIPS2FITS_URLS[0], False, "t")
    headers = cut.headers()
    assert headers["X-Cutout-Pixel-Scale-Arcsec"] == "2917.02"  # 0.81028 deg, not the nominal 3240
    assert headers["X-Cutout-Cdelt-Deg"] == "0.810285" and headers["X-Cutout-Projection"] == "MOL"
    summary = cut.summary()
    assert summary["nominal_pixel_scale_arcsec"] == 3240.0 and summary["cdelt_deg"] == pytest.approx(0.8102847)


# ---------------------------------------------------------------------------
# Colour FITS: the companion named in headers and the CLI covers the field
# ---------------------------------------------------------------------------


async def test_colour_fits_names_a_companion_with_data(cache: CutoutCache) -> None:
    """(345, -35): Legacy DR10 colour has data but DR10 r does not; g does (recorded MocServer answer)."""
    with replay_router("legacy_gap_color_fits") as router:
        async with offline_client() as client:
            service = CutoutService(client, cache=cache)
            cut = await service.cutout(legacy_gap_fits())
            again = await service.cutout(legacy_gap_fits())
        count = router.calls.call_count
    assert count == 2, "one image + one companion footprint query; the cache hit keeps the answer"
    for item in (cut, again):
        headers = item.headers()
        assert headers["X-Cutout-Pixels"] == "rgb-preview" and headers["X-Cutout-Science-Survey"] == "legacy_g"
        assert "Legacy Surveys DR10 r has no data here" in headers["X-Cutout-Science-Note"]
    assert again.cached


def test_cli_colour_fits_hint_names_the_band_with_data(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_TTL_SECONDS", "0")
    args = cli_parser().parse_args(["cutout", "--ra", "345", "--dec", "-35", "--fov", "1", "--survey", "legacy",
                                    "--width", "16", "--height", "16", "--out", str(tmp_path / "l.fits")])
    with replay_router("legacy_gap_color_fits"):
        assert args.handler(args) == 0
    err = capsys.readouterr().err
    assert "--survey legacy_g" in err and "--survey legacy_r" not in err


def test_colour_fits_without_a_covering_companion_names_none() -> None:
    cut = imaging.Cutout(legacy_gap_fits(), b"", "application/fits", 16, 16, "u", "e", False, "t",
                         info=imaging.ImageInfo("fits", 16, 16, 4, 8))
    cut.science, cut.science_note = imaging.fits_companion(SURVEYS["legacy"], {
        "legacy_r": False, "legacy_g": False, "legacy_z": False, "legacy_i": False})
    cut.science_checked = True
    headers = cut.headers()
    assert "X-Cutout-Science-Survey" not in headers and "No single-band FITS survey" in headers["X-Cutout-Science-Note"]


# ---------------------------------------------------------------------------
# Cache entries with bad metadata are misses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", [
    {"source_url": "x"},
    {"width": "128", "height": 128, "source_url": "x", "endpoint": "e", "fetched_at": "t"},
    {"width": 128, "height": 128, "source_url": "x", "endpoint": None, "fetched_at": "t"},
    {"width": 64, "height": 64, "source_url": "x", "endpoint": "e", "fetched_at": "t"},
    {"width": 128, "height": 128, "source_url": "x", "endpoint": "e", "fetched_at": "t", "coverage_fraction": "1"},
    {"width": 128, "height": 128, "source_url": "x", "endpoint": "e", "fetched_at": "t", "blank": "maybe"},
])
async def test_cache_entries_with_bad_metadata_are_misses(cache: CutoutCache, meta: dict) -> None:
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=128, height=128)
    cache.put(request.cache_key(), body("dss2_3c273_png"), meta)
    with replay_router("dss2_3c273_png") as router:
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
        assert router.calls.call_count == 1
    assert not cut.cached and cut.content == body("dss2_3c273_png")
    # The refetch replaced the bad entry with a complete one.
    content, fresh = cache.get(request.cache_key())
    assert content == body("dss2_3c273_png")
    assert fresh["width"] == 128 and fresh["endpoint"] == imaging.HIPS2FITS_URLS[0]


async def test_cache_entry_with_other_image_bytes_is_a_miss(cache: CutoutCache) -> None:
    request = CutoutRequest(ra=C3C273[0], dec=C3C273[1], survey="dss2", fov_arcmin=2.0, width=128, height=128)
    good_meta = {"width": 128, "height": 128, "source_url": "x", "endpoint": "e", "fetched_at": "t"}
    cache.put(request.cache_key(), _png(64, 64), good_meta)
    with replay_router("dss2_3c273_png"):
        async with offline_client() as client:
            cut = await CutoutService(client, cache=cache).cutout(request)
    assert not cut.cached and (cut.width, cut.height) == (128, 128)


# ---------------------------------------------------------------------------
# Sesame failures (models.resolution_failure_status): unknown name 404, resolver outage 503 + Retry-After,
# unusable answer 502
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route, status, answer, expected, fragment", [
    # Sesame down (HTTP 5xx): the name may be valid, so 503 + Retry-After.
    ("/api/v1/cutouts", 503, "Service Unavailable", 503, "request failed"),
    ("/api/v1/cutouts/stack", 503, "Service Unavailable", 503, "request failed"),
    # Sesame answered, but not with a usable document: 502.
    ("/api/v1/cutouts", 200, "<html><body>Maintenance <b>tonight</body></html>", 502, "could not be parsed"),
    ("/api/v1/cutouts/stack", 200, '<?xml version="1.0"?><Sesame><Target><Resolver', 502, "could not be parsed"),
])
def test_router_sesame_outage_is_503_and_unusable_answer_502(tmp_path: Path, route: str, status: int, answer: str,
                                                             expected: int, fragment: str) -> None:
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        router.route(host="cds.unistra.fr").respond(status, text=answer, headers={"content-type": "text/html"})
        resp = client.get(route, params={"name": "3C 273"})
    assert resp.status_code == expected, resp.text
    assert fragment in resp.json()["detail"]
    if expected == 503:
        assert resp.headers["retry-after"] == "30"
    else:
        assert "retry-after" not in resp.headers


@pytest.mark.parametrize("route", ["/api/v1/cutouts", "/api/v1/cutouts/stack"])
def test_router_unknown_name_is_404(tmp_path: Path, route: str) -> None:
    """The recorded Sesame answer for an unknown name is well-formed XML without coordinates."""
    client = TestClient(make_app(tmp_path))
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="testserver").pass_through()
        router.route(host="cds.unistra.fr").respond(200, text=SESAME_UNKNOWN, headers={"content-type": "text/plain"})
        resp = client.get(route, params={"name": "NoSuchObjectQzx42"})
    assert resp.status_code == 404 and "No coordinates found" in resp.json()["detail"]


def test_name_classification_helper() -> None:
    from xml.etree.ElementTree import ParseError

    from models import ObjectResolutionError

    def wrapped(cause: BaseException, text: str) -> ObjectResolutionError:
        exc = ObjectResolutionError(text)
        exc.__cause__ = cause
        return exc

    assert imaging.name_unknown_to_sesame(wrapped(ValueError("No coordinates found for object 'x'."), "parsed"))
    assert imaging.name_unknown_to_sesame(ObjectResolutionError("Object name must not be empty."))
    assert not imaging.name_unknown_to_sesame(wrapped(ParseError("junk"), "Sesame response could not be parsed"))
    assert not imaging.name_unknown_to_sesame(ObjectResolutionError("Sesame request failed: HTTP 503"))
    assert not imaging.name_unknown_to_sesame(wrapped(TypeError("odd"), "Sesame response could not be parsed"))


# ---------------------------------------------------------------------------
# Survey metadata
# ---------------------------------------------------------------------------


def test_galex_pixels_are_count_rates_with_published_zero_points() -> None:
    for key, zero_point in (("galex_nuv", "20.08"), ("galex_fuv", "18.82")):
        survey = SURVEYS[key]
        assert survey.pixel_units == "counts/s" and survey.calibrated is False
        assert f"m_AB = {zero_point} - 2.5 log10(cps)" in survey.pixel_note
        assert "2007ApJS..173..682M" in survey.pixel_note and "MJy/sr'" not in survey.pixel_note
    # Uncalibrated pixels are labelled as such in the stack's FITS links and the CLI.
    panel = imaging._panel("uv", SURVEYS["galex"], {"galex_nuv": True}, imaging.PanelCentre(0.0, 0.0))
    assert panel.fits_survey == "galex_nuv" and panel.fits_calibrated is False


def test_unwise_and_dss2_epochs_follow_the_data() -> None:
    assert SURVEYS["unwise"].epoch_span == SURVEYS["unwise_w1"].epoch_span == (2010.0, 2022.0)
    assert SURVEYS["unwise_w1"].mean_epoch == 2016.0
    assert SURVEYS["dss2"].mean_epoch == pytest.approx(1987.0, abs=0.01)
    assert SURVEYS["dss2_red"].mean_epoch == pytest.approx(1991.5, abs=0.05)
    assert "different epochs" in SURVEYS["dss2"].note


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


async def _live(request: CutoutRequest, cache: CutoutCache, **options) -> imaging.Cutout:
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            return await CutoutService(client, cache=cache, **options).cutout(request)
    except imaging.CutoutUpstreamError as exc:
        if str(exc).startswith("hips2fits unavailable") and await hips2fits_is_down():
            pytest.skip(str(exc))
        raise


def _skip_if_moc_unknown(cut: imaging.Cutout) -> None:
    if cut.blank == "unconfirmed":
        pytest.skip("MocServer unreachable: blank could not be judged")


@pytest.mark.live
async def test_live_vlass_crab_jpeg_is_never_a_silent_blank(tmp_path: Path) -> None:
    """VLASS at the Crab as JPEG: either real pixels (the Crab is a ~1 kJy radio source, so the image
    cannot be flat), or a blank flagged as a rendering failure and kept out of the cache."""
    cache = CutoutCache(tmp_path / "c", ttl_seconds=3600)
    cut = await _live(vlass_crab_jpg(), cache)
    _skip_if_moc_unknown(cut)
    if imaging.is_uniform_raster(cut.content):
        assert cut.blank == "rendering-failure" and cut.degraded and "overlaps this field" in cut.degraded
        assert cut.headers()["Cache-Control"] == "no-store" and cache.size_bytes() == 0
    else:
        assert cut.blank is None and cut.degraded is None and cache.size_bytes() == len(cut.content)


@pytest.mark.live
def test_live_cli_vlass_crab_jpeg_is_not_saved_when_blank(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUTOUT_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "crab.jpg"
    args = cli_parser().parse_args(["cutout", "--ra", str(CRAB[0]), "--dec", str(CRAB[1]), "--fov", "3",
                                    "--survey", "vlass", "--out", str(out)])
    code = args.handler(args)
    captured = capsys.readouterr()
    if code == 1 and asyncio.run(hips2fits_is_down()):
        pytest.skip(captured.err)
    if code == imaging.EXIT_DEGRADED:
        assert not out.exists() and not any((tmp_path / "cache").glob("*.bin"))
    else:
        assert code == 0 and "footprint could not be checked" not in captured.err
        from PIL import Image

        pixels = np.asarray(Image.open(out))
        assert pixels.max() > pixels.min(), "a saved VLASS image of the Crab must show structure"


@pytest.mark.live
async def test_live_ps1_south_jpeg_is_a_confirmed_no_data(tmp_path: Path) -> None:
    cache = CutoutCache(tmp_path / "c", ttl_seconds=3600)
    cut = await _live(ps1_south_jpg(), cache)
    _skip_if_moc_unknown(cut)
    assert imaging.is_uniform_raster(cut.content)  # hips2fits' white 'no data' JPEG
    assert cut.blank == "no-data" and cut.coverage_fraction == 0.0 and cut.degraded is None
    assert cache.size_bytes() == len(cut.content)


@pytest.mark.live
async def test_live_flat_rass_field_is_data_not_blank(tmp_path: Path) -> None:
    """ROSAT's all-sky survey has ~1 photon per arcmin^2 at high latitude: a 12" field at (187, +40)
    holds none, and hips2fits renders it flat; the PNG check keeps it as data."""
    cut = await _live(rass_flat_jpg(), CutoutCache(tmp_path / "c", ttl_seconds=0))
    assert imaging.is_uniform_raster(cut.content)
    assert cut.coverage_fraction == 1.0 and cut.blank is None and cut.degraded is None


@pytest.mark.live
@pytest.mark.parametrize("projection, fov, width, height", [
    ("MOL", 360.0, 400, 200), ("AIT", 360.0, 256, 256), ("TAN", 100.0, 40, 20), ("TAN", 0.1, 300, 150),
])
async def test_live_cdelt_and_centre_scale_match_the_fits_header(tmp_path: Path, projection: str, fov: float,
                                                                 width: int, height: int) -> None:
    request = CutoutRequest(ra=10.0, dec=20.0, survey="rass", fov_arcmin=fov * 60.0, width=width, height=height,
                            projection=projection, format="fits")
    cut = await live_cutout(request, tmp_path)
    from astropy.io import fits

    header = fits.getheader(io.BytesIO(cut.content))
    assert abs(header["CDELT1"]) == pytest.approx(request.cdelt_deg, rel=1e-4)
    assert abs(header["CDELT2"]) == pytest.approx(request.cdelt_deg, rel=1e-4)
    assert float(cut.headers()["X-Cutout-Cdelt-Deg"]) == pytest.approx(abs(header["CDELT1"]), rel=1e-5)
    if projection in ("MOL", "AIT"):  # equal-area: the centre scale is CDELT
        assert request.pixel_scale_arcsec == pytest.approx(abs(header["CDELT1"]) * 3600, rel=1e-4)


@pytest.mark.live
async def test_live_verified_colormap_changes_the_rendering(tmp_path: Path) -> None:
    base = {"ra": C3C273[0], "dec": C3C273[1], "survey": "nvss", "fov_arcmin": 6.0, "width": 24, "height": 24}
    plain = await live_cutout(CutoutRequest(**base), tmp_path)
    viridis = await live_cutout(CutoutRequest(**base, cmap="viridis"), tmp_path)
    reversed_grey = await live_cutout(CutoutRequest(**base, cmap="Greys_r"), tmp_path)
    assert viridis.content != plain.content  # a known name is applied
    assert reversed_grey.content == plain.content  # Greys_r is hips2fits' default


@pytest.mark.live
async def test_live_colour_fits_names_the_legacy_band_with_data(tmp_path: Path) -> None:
    cut = await _live(legacy_gap_fits(), CutoutCache(tmp_path / "c", ttl_seconds=0))
    assert cut.pixel_kind == "rgb-preview"
    headers = cut.headers()
    assert headers["X-Cutout-Science-Survey"] in ("legacy_g", "legacy_z", "legacy_i")
    assert "Legacy Surveys DR10 r has no data here" in headers["X-Cutout-Science-Note"]


@pytest.mark.live
def test_live_unknown_name_is_404(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path))
    resp = client.get("/api/v1/cutouts/stack", params={"name": "NoSuchObjectQzx42"})
    # A resolver outage (503 naming a network error) skips; a 502 (unusable answer) or any other status fails.
    body = api_ok(resp, 404)
    assert "No coordinates found" in body["detail"]


@pytest.mark.live
def test_live_barnards_star_is_in_the_unwise_panel(tmp_path: Path) -> None:
    """unWISE neo8 (2010-2022) panels are centred at 2016.0, 166" from J2000: Barnard's star (W1 ~ 4.5)
    must be the brightest source and near the centre of a 3' panel (J2000 lies outside it)."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    client = TestClient(make_app(tmp_path))
    resp = client.get("/api/v1/cutouts/stack", params={"name": "Barnard's star", "fov_arcmin": 3, "size": 90,
                                                       "surveys": "unwise_w1"})
    if resp.status_code == 502:
        pytest.skip(resp.text)
    assert resp.status_code == 200, resp.text
    panel = resp.json()["panels"][0]
    assert panel["epoch"] == 2016.0 and panel["note"] is None
    fits_resp = client.get(panel["fits_url"])
    if fits_resp.status_code == 502 and asyncio.run(hips2fits_is_down()):
        pytest.skip(fits_resp.text)
    assert fits_resp.status_code == 200, fits_resp.text
    with fits.open(io.BytesIO(fits_resp.content)) as hdul:
        image, wcs = hdul[0].data.astype(float), WCS(hdul[0].header)
    peak = np.unravel_index(np.nanargmax(image), image.shape)
    star = wcs.pixel_to_world(peak[1], peak[0])
    centre = SkyCoord(panel["center_ra"], panel["center_dec"], unit="deg")
    # The coadd weights the eight NEOWISE years (2014-2022) over 2010, so the star sits a little north
    # of the 2016.0 centre (measured ~2017.3), well inside the 90" half-field.
    assert star.separation(centre).arcsec < 40.0


@pytest.mark.live
async def test_live_galex_pixels_follow_the_morrissey_zero_points(tmp_path: Path) -> None:
    """Aperture photometry of GALEX NUV hips2fits pixels (counts/s per native 1.5" pixel) against the
    GUVcat AIS catalogue (VizieR II/335/galex_ais) reproduces m_AB = 20.08 - 2.5 log10(cps) within 0.3 mag."""
    from astropy.io import fits
    from astropy.wcs import WCS

    ra0, dec0 = C3C273
    query = {"-source": "II/335/galex_ais", "-c": f"{ra0} {dec0:+}", "-c.rm": "4.8", "-out": "RAJ2000,DEJ2000,NUVmag",
             "-out.max": "50", "NUVmag": "13..20.5"}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            answer = await client.get("https://vizier.cds.unistra.fr/viz-bin/asu-tsv", params=query)
            answer.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"VizieR unavailable: {exc}")
    rows = [line.split("\t") for line in answer.text.splitlines()
            if line and not line.startswith("#") and line[0].isdigit()]
    sources = [(float(r[0]), float(r[1]), float(r[2])) for r in rows if len(r) >= 3 and r[2].strip()]
    assert sources, "GUVcat lists NUV sources around 3C 273"
    cut = await live_cutout(CutoutRequest(ra=ra0, dec=dec0, survey="galex_nuv", fov_arcmin=12.0, width=480,
                                          height=480, format="fits"), tmp_path)
    with fits.open(io.BytesIO(cut.content)) as hdul:
        image, wcs = hdul[0].data.astype(float), WCS(hdul[0].header)
    assert abs(hdul[0].header["CDELT1"]) * 3600 == pytest.approx(1.5, rel=1e-3)  # native GALEX pixels
    rows_, cols_ = np.mgrid[:image.shape[0], :image.shape[1]]
    zero_points = []
    for ra, dec, mag in sources:
        x, y = wcs.world_to_pixel_values(ra, dec)
        if not (20 < x < 460 and 20 < y < 460):
            continue
        r = np.hypot(cols_ - x, rows_ - y)
        background = np.nanmedian(image[(r > 9) & (r < 14)])
        flux = np.nansum(image[r < 6] - background)
        if flux > 0:
            zero_points.append(mag + 2.5 * math.log10(flux))
    assert zero_points
    assert float(np.median(zero_points)) == pytest.approx(20.08, abs=0.3)


def test_cli_pixel_scale_text_is_readable() -> None:
    assert imaging._angle_text(0.3515625) == '0.352"'
    assert imaging._angle_text(225.0) == "3.75'"
    assert imaging._angle_text(2917.0248) == "48.6'"
    assert imaging._angle_text(4557.85) == "1.27 deg"
