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


# --- a name together with coordinates is refused (main.check_search_target), never dropped ---

@pytest.mark.parametrize("route", ["/api/v1/cutouts", "/api/v1/cutouts/stack"])
@pytest.mark.parametrize("coordinates", [{"ra": 1, "dec": 2}, {"ra": 1}, {"dec": 2}])
def test_cutout_routes_refuse_a_name_with_coordinates(tmp_path, route: str, coordinates: dict) -> None:
    """Finding: ?name=M87&ra=1&dec=2 rendered (and planned) the field at 1,2, silently ignoring
    the name. The project rule is a 422 before anything is resolved or fetched."""
    from main import NAME_AND_COORDINATES

    client = TestClient(make_app(tmp_path))
    with _router({"status_code": 200, "text": SESAME_BARNARD, "headers": {"content-type": "text/xml"}}) as router:
        response = client.get(route, params={"name": "M87", **coordinates})
        upstream = [call for call in router.calls if call.request.url.host != "testserver"]
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == NAME_AND_COORDINATES
    assert upstream == []  # neither Sesame nor hips2fits/MocServer was asked


@pytest.mark.parametrize("route", ["/api/v1/cutouts", "/api/v1/cutouts/stack"])
def test_cutout_routes_accept_coordinates_with_an_empty_name(tmp_path, route: str) -> None:
    """``name=`` (an empty form field) is no name, as on every search route: ra/dec are used."""
    client = TestClient(make_app(tmp_path))
    with _router({"status_code": 503}) as router:
        params = {"name": "", "ra": BARNARD[0], "dec": BARNARD[1], "format": "jpg"}
        if route == "/api/v1/cutouts":  # the recorded JPEG is 200x100
            params.update(survey="panstarrs", width=200, height=100)
        response = client.get(route, params=params)
        sesame = [call for call in router.calls if call.request.url.host == "cds.unistra.fr"
                  and "sesame" in call.request.url.path]
    assert response.status_code == 200, response.text
    assert sesame == []


def _cli_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="astrosearch")
    imaging.register_cli(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.parametrize("coordinates", [["--ra", "1", "--dec", "2"], ["--ra", "1"], ["--dec", "2"]])
def test_cli_cutout_refuses_a_name_with_coordinates(tmp_path, capsys, coordinates: list[str]) -> None:
    from main import NAME_AND_COORDINATES

    out = tmp_path / "m87.png"
    args = _cli_parser().parse_args(["cutout", "--name", "M87", *coordinates, "--out", str(out)])
    with respx.mock(assert_all_mocked=True) as router:  # any request would raise
        assert args.handler(args) == 2
    assert router.calls.call_count == 0 and not out.exists()
    assert NAME_AND_COORDINATES in capsys.readouterr().err


def test_cli_cutout_refuses_a_name_with_coordinates_through_main_parser(tmp_path, capsys) -> None:
    """The full ``astrosearch`` parser dispatches to the same check (exit 2)."""
    import main

    out = tmp_path / "m87.png"
    args = main.build_parser().parse_args(["cutout", "--name", "M87", "--ra", "1", "--dec", "2", "--out", str(out)])
    assert args.handler(args) == 2 and not out.exists()
    assert main.NAME_AND_COORDINATES in capsys.readouterr().err


def test_cli_cutout_without_a_full_target_is_exit_2(tmp_path, capsys) -> None:
    args = _cli_parser().parse_args(["cutout", "--ra", "1", "--out", str(tmp_path / "x.png")])
    assert args.handler(args) == 2
    assert "Either name or both ra and dec are required" in capsys.readouterr().err


def test_owned_clients_reuse_one_ssl_context(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The clients the router opens itself (no app.state.client: Sesame, then hips2fits) share one
    cached SSL context; httpx's default builds one from certifi per client (~0.5 s on Windows)."""
    import ssl

    context = imaging._ssl_context()
    builds: list[object] = []
    real_context = ssl.create_default_context

    def counting(*args, **kwargs):
        builds.append(args)
        return real_context(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", counting)
    created: list[httpx.AsyncClient] = []
    real_new = imaging._new_client

    def tracked(timeout: float) -> httpx.AsyncClient:
        created.append(real_new(timeout))
        return created[-1]

    monkeypatch.setattr(imaging, "_new_client", tracked)
    client = TestClient(make_app(tmp_path))
    with _router({"status_code": 200, "text": SESAME_BARNARD, "headers": {"content-type": "text/xml"}}):
        response = client.get("/api/v1/cutouts", params={"name": "Barnard's star", "survey": "panstarrs",
                                                         "format": "jpg", "fov_arcmin": 4, "width": 200, "height": 100})
    assert response.status_code == 200, response.text
    assert builds == []
    assert len(created) >= 2 and all(c.is_closed for c in created)
    assert all(c._transport._pool._ssl_context is context for c in created)  # type: ignore[attr-defined]
