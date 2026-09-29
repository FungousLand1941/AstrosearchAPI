"""The web UI (web/app.js) against the integrated API: every endpoint it calls is served by
api.app, and the light-curve request carries the target's epoch, proper motion and parallax
(the cutout stack already did; without them a fast star's ZTF/NEOWISE/Gaia cones miss it)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_imaging_ui import API_CONTRACT, NODE, endpoints_in_app_js, run_node

import api


def _shape(path: str) -> str:
    """A path with its parameters unnamed ('/api/v1/alerts/{alert_id}' -> '/api/v1/alerts/{}')."""
    return re.sub(r"\{[^}]*\}", "{}", path)


def test_every_endpoint_the_ui_calls_is_served_by_the_app() -> None:
    served = {(method, path) for path, methods in api.route_table() for method in methods}
    for name, (method, path) in endpoints_in_app_js().items():
        assert (method, path) in served, f"{name}: {method} {path} is not served by api.app"
    shapes = {(method, _shape(path)) for method, path in served}
    missing = {(method, path) for method, path in API_CONTRACT if (method, _shape(path)) not in shapes}
    assert not missing, sorted(missing)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_light_curve_query_carries_the_targets_motion(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
out.moving = m.lightcurveParams({ ra: 269.45207696, dec: 4.69336497, pm_ra_masyr: -801.551, pm_dec_masyr: 10362.394,
                                  epoch: 2000, parallax_mas: 546.9759 }, 10);
out.bare = m.lightcurveParams({ ra: 187.2779154, dec: 2.0523883 }, 3);
out.undated = m.lightcurveParams({ ra: 10, dec: 20, pm_ra_masyr: 5, pm_dec_masyr: 5, parallax_mas: -1 }, 5);
out.info = m.targetInfo({ ra: '269.45207696', dec: '4.69336497', pm_ra_masyr: -801.551, pm_dec_masyr: 10362.394,
                          epoch: 2000, parallax_mas: 546.9759, frame: 'icrs' });
""")
    assert out["moving"] == {"ra": 269.45207696, "dec": 4.69336497, "radius_arcsec": 10, "surveys": "ztf,neowise,gaia",
                             "epoch": 2000, "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394, "parallax_mas": 546.9759}
    assert out["bare"] == {"ra": 187.2779154, "dec": 2.0523883, "radius_arcsec": 3, "surveys": "ztf,neowise,gaia"}
    # A proper motion without the epoch of the position cannot be applied; a negative parallax is not sent.
    assert out["undated"] == {"ra": 10, "dec": 20, "radius_arcsec": 5, "surveys": "ztf,neowise,gaia"}
    assert out["info"]["parallax_mas"] == 546.9759 and out["info"]["epoch"] == 2000


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_known_target_sends_its_motion_to_the_light_curve_service(tmp_path: Path) -> None:
    """A search record's target (Barnard's star at J2000 with its SIMBAD motion and parallax): the
    light-curve request the UI then sends follows the star."""
    out = run_node(tmp_path, r"""
const body = dom.installDom('http://ui.test/');
for (const id of ['sed-panel', 'lc-panel', 'cutout-panel', 'sso-panel']) body.append(dom.element('div', id));
const fov = dom.element('select', 'cutout-fov');
fov.value = '3';
body.append(fov);
const fetched = [];
globalThis.fetch = async (url) => { fetched.push(new URL(String(url))); return dom.json({ detail: 'Not Found' }, 404); };
m.state.query = { name: "Barnard's star", radius: 10 };
m.state.target = null;
m.onTargetKnown(m.targetInfo({ ra: 269.45207696, dec: 4.69336497, epoch: 2000, pm_ra_masyr: -801.551,
                               pm_dec_masyr: 10362.394, parallax_mas: 546.9759 }), new AbortController().signal);
for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0));
const lc = fetched.find((u) => u.pathname === '/api/v1/lightcurves');
out.lc = lc ? Object.fromEntries(lc.searchParams) : null;
""")
    assert out["lc"] is not None
    assert {k: out["lc"][k] for k in ("epoch", "pm_ra_masyr", "pm_dec_masyr", "parallax_mas")} == {
        "epoch": "2000", "pm_ra_masyr": "-801.551", "pm_dec_masyr": "10362.394", "parallax_mas": "546.9759"}
    assert float(out["lc"]["ra"]) == pytest.approx(269.45207696) and out["lc"]["surveys"] == "ztf,neowise,gaia"
