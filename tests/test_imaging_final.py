"""Final-review regressions of the imaging router: /cutouts?name= follows the resolver's proper
motion (as /cutouts/stack does), and name-resolution failures use the shared statuses."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixture_io import FIXTURES

import imaging
from imaging import SURVEYS, CutoutCache

BARNARD = (269.45207696, 4.69336497)  # SIMBAD ICRS J2000
BARNARD_PM = (-801.551, 10362.394)  # SIMBAD, mas/yr
SESAME_BARNARD = (FIXTURES / "sesame" / "barnards_star.xml").read_text(encoding="utf-8")
SESAME_UNKNOWN = (FIXTURES / "imaging" / "sesame_unknown.xml").read_text(encoding="utf-8")
JPEG = (FIXTURES / "imaging" / "panstarrs_m87_jpg.0.body").read_bytes()


def make_app(tmp_path) -> FastAPI:
    app = FastAPI()
    app.include_router(imaging.router)
    app.state.cutout_cache = CutoutCache(tmp_path / "api-cache", ttl_seconds=3600)
    return app


def _router(sesame: dict) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route(host="testserver").pass_through()
    router.route(host="cds.unistra.fr", path__startswith="/cgi-bin/nph-sesame").respond(**sesame)
    for url in imaging.HIPS2FITS_URLS:
        router.get(url__startswith=url).respond(200, content=JPEG, headers={"content-type": "image/jpeg"})
    for url in imaging.MOCSERVER_URLS:
        router.get(url__startswith=url).respond(503)
    return router


def test_cutout_by_name_is_centred_on_the_survey_epoch_position(tmp_path) -> None:
    """Finding: /cutouts?name=Barnard's star was centred on the J2000 position whatever the survey
    epoch (the star moves 10.4"/yr: ~125" off-centre in PanSTARRS)."""
    from models import propagate_radec

    client = TestClient(make_app(tmp_path))
    with _router({"status_code": 200, "text": SESAME_BARNARD, "headers": {"content-type": "text/xml"}}) as router:
        response = client.get("/api/v1/cutouts", params={"name": "Barnard's star", "survey": "panstarrs",
                                                         "format": "jpg", "fov_arcmin": 4, "width": 200, "height": 100})
        sent = [call.request for call in router.calls if call.request.url.host != "cds.unistra.fr"
                and "hips" in str(call.request.url.params)]
    assert response.status_code == 200, response.text
    epoch = SURVEYS["panstarrs"].mean_epoch
    ra, dec = propagate_radec(*BARNARD, *BARNARD_PM, 2000.0, epoch)
    params = httpx.URL(str(sent[0].url)).params
    assert float(params["ra"]) == pytest.approx(ra, abs=1e-6) and float(params["dec"]) == pytest.approx(dec, abs=1e-6)
    assert float(response.headers["x-cutout-centre-epoch"]) == pytest.approx(epoch, abs=1e-3)
    assert float(response.headers["x-cutout-centre-offset-arcsec"]) == pytest.approx(125, abs=5)


@pytest.mark.parametrize(("sesame", "status"), [
    ({"status_code": 200, "text": SESAME_UNKNOWN, "headers": {"content-type": "text/xml"}}, 404),
    ({"status_code": 503}, 503),
])
def test_cutout_name_failures_use_the_shared_statuses(tmp_path, sesame: dict, status: int) -> None:
    """Finding: an unresolvable name was a 422 on /cutouts but a 404 on /search."""
    client = TestClient(make_app(tmp_path))
    with _router(sesame):
        response = client.get("/api/v1/cutouts", params={"name": "NoSuchObjectQzx42"})
    assert response.status_code == status, response.text
    if status == 503:
        assert response.headers["retry-after"] == "30"
