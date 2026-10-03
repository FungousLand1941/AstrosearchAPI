"""Round-6 live truths for sed.py (run with ``pytest -m live``).

Asserted against the real archives: Hercules A (3C 348) is at z = 0.155 (NED and SIMBAD agree), never at its SDSS
photo-z 0.134; Cygnus A and NGC 1275 at the default 10 arcsec radius are AGN in resolved hosts, not quasars; the
Local Volume Seyfert NGC 4395 keeps its cz ~ 320 km/s redshift; PKS 1510-089 is never reported at NED's z = 0.0068 as
reliable even while SIMBAD is backed off; Gl 229B (a T7 dwarf) does not get its M1V primary's Chandra flux; Procyon B's
blended X-rays are not AGN evidence. Tests skip only when a service is unreachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_sed_live import build, require_catalogs, require_lookups
from test_sed_round3_live import evidence, used

import sed

pytestmark = pytest.mark.live


async def test_hercules_a_is_at_its_spectroscopic_redshift(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Hercules A")
    require_catalogs(result, ["ned", "simbad"])
    z = result["redshift"]
    assert z["value"] == pytest.approx(0.155, abs=0.002), z
    assert z["source"] in {"ned", "simbad", "sdss_specobj", "sdss"} and z["kind"] != "photo", z
    assert z["reliable"] is True and not z.get("conflict"), z


@pytest.mark.parametrize("name", ["Cygnus A", "NGC 1275"])
async def test_blazar_typed_radio_galaxies_are_not_quasars_at_the_default_radius(tmp_path: Path, name: str) -> None:
    result = await build(tmp_path, name=name)  # default 10 arcsec: no lobe evidence
    require_catalogs(result, ["simbad", "ned"])
    label = result["classification"]["label"]
    assert label in {"agn", "galaxy"}, result["classification"]
    line = next((t for t in result["classification"]["evidence"] if t.startswith("SIMBAD object type 'Bla'")), None)
    if line is not None:  # SIMBAD still types it a blazar: the extended host must have capped it
        assert "nucleus is not quasar-luminous" in line, line


async def test_ngc4395_keeps_its_local_volume_redshift(tmp_path: Path) -> None:
    result = await build(tmp_path, name="NGC 4395")
    require_catalogs(result, ["ned", "simbad"])
    z = result["redshift"]
    assert z["value"] == pytest.approx(0.00107, abs=0.0002), z  # cz ~ 320 km/s
    assert z["reliable"] is True and not z.get("conflict"), z
    assert result["classification"]["label"] in {"galaxy", "agn"}, result["classification"]


async def test_pks_1510_089_with_simbad_backed_off(tmp_path: Path) -> None:
    """The degraded case the round-5 live test skipped: SIMBAD TAP skipped by the back-off."""
    backoff = sed.default_backoff()
    backoff.record_failure(sed._host(sed.SIMBAD_TAP_URL), "ReadTimeout after 60 s (test)", 600.0)
    try:
        result = await build(tmp_path, name="PKS 1510-089")
    finally:
        backoff.clear()
    require_catalogs(result, ["ned", "simbad"])
    assert any(n.startswith("simbad lookup failed: skipped") for n in result["notes"]), result["notes"]
    z = result["redshift"]
    assert not (z["value"] is not None and z["value"] < 0.02 and z["reliable"] is True), z
    if z["value"] is not None and z["value"] < 0.02:
        assert z["conflict"] is True and "cross-check incomplete" in z["note"], z
    assert "below the quasar luminosity threshold" not in evidence(result)
    assert result["classification"]["label"] == "qso", result["classification"]


async def test_gl229b_does_not_get_its_primarys_chandra_flux(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Gl 229B")
    require_catalogs(result, ["chandra", "gaia_dr3", "simbad"])
    chandra = [p for p in result["points"] if p["catalog"] == "chandra"]
    for point in chandra:
        assert point["quality_warning"], point
        assert any("not attributed to the target" in w for w in point["warnings"]), point["warnings"]
    assert not any(e.startswith("log(fX/fV)") and "(Chandra ACIS" in e for e in result["classification"]["evidence"])
    assert result["classification"]["label"] == "star"


async def test_procyon_b_xrays_are_not_agn_evidence(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Procyon B")
    require_catalogs(result, ["simbad"])
    require_lookups(result, ["simbad"])
    detail = [d for d in result["classification"]["evidence_detail"] if d["text"].startswith("log(fX/fV)")]
    for line in detail:
        assert line["weights"] == {} and "not diagnostic" in line["text"], line
    assert result["classification"]["label"] == "star"
    gaia = used(result, "gaia_dr3")
    assert gaia is None or gaia["separation_arcsec"] <= gaia["tolerance_arcsec"]
