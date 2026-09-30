"""Round-5 live truths for sed.py (run with ``pytest -m live``).

Asserted against the real archives: name resolution returns the object's identifiers, so the T8 dwarf queried by
SIMBAD's main identifier or its AllWISE designation keeps its own 2MASS row (J = 15.695) and Scholz's star its Gaia
DR3 row; PKS 1510-089 is reported at z = 0.36 (or with an explicit redshift conflict), never at NED's z = 0.0068 as
reliable; Proxima Cen's 5XMM stack fluxes are flagged; SIMBAD quality-E radial velocities (Sco X-1, NGC 7027) are
unreliable while Sirius' quality-A velocity is a reliable spectrum; the dusty starbursts Arp 220 and Mrk 231 are not
called radio-loud AGN from their radio/optical ratio. Tests skip only when a service is unreachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from test_sed_live import build, require_catalogs, require_lookups
from test_sed_round3_live import bands, evidence, used

import sed
from models import ObjectResolutionError

pytestmark = pytest.mark.live


async def test_sesame_resolution_lists_aliases() -> None:
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            info = await sed.resolve_name("Sirius", client)
    except ObjectResolutionError as exc:
        if str(exc).startswith("Sesame request failed"):
            pytest.skip(f"Sesame unreachable: {exc}")
        raise
    aliases = info["resolved"]["aliases"]
    assert "HD 48915" in aliases and any(a.startswith(("Gaia DR3 ", "2MASS J")) for a in aliases), aliases


@pytest.mark.parametrize("name", ["2MASSI J0415195-093506", "WISEA J041521.26-093500.4"])
async def test_t8_dwarf_by_any_name_keeps_its_2mass_row(tmp_path: Path, name: str) -> None:
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["twomass_psc", "simbad"])
    tmass = used(result, "twomass_psc")
    assert tmass is not None and tmass["source_id"] == "04151954-0935066", result["members"]
    assert tmass["designation"] == "2MASS J04151954-0935066"
    assert bands(result, "2MASS") == {"J", "H", "Ks"}
    j = next(p for p in result["points"] if p["facility"] == "2MASS" and p["band"] == "J")
    assert j["magnitude"] == pytest.approx(15.695, abs=1e-3)
    assert result["classification"]["label"] == "star"


async def test_scholzs_star_keeps_its_gaia_row(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Scholz's star")
    require_catalogs(result, ["gaia_dr3", "simbad"])
    gaia = used(result, "gaia_dr3")
    assert gaia is not None and gaia["source_id"] == "3048443305671969152", result["members"]
    assert gaia["designation"] == "Gaia DR3 3048443305671969152"
    assert bands(result, "Gaia") == {"G", "BP", "RP"}
    assert result["classification"]["label"] == "star"


async def test_pks_1510_089_is_not_reported_at_neds_z(tmp_path: Path) -> None:
    result = await build(tmp_path, name="PKS 1510-089")
    require_catalogs(result, ["ned", "simbad"])
    # No require_lookups(['simbad']) (round-6 review): a failed SIMBAD TAP lookup is exactly the case to check -- the
    # SIMBAD row's z = 0.356 still takes part in the cross-check, so NED's z = 0.0068 is never reported as reliable.
    z = result["redshift"]
    if z.get("conflict"):
        assert z["reliable"] is False and len(z["discordant"]) >= 2, z
    else:
        assert z["value"] == pytest.approx(0.36, abs=0.01) and z["reliable"] is True, z
    assert not (z["value"] < 0.02 and z["reliable"] is True), z
    assert result["classification"]["label"] == "qso", result["classification"]
    assert "below the quasar luminosity threshold" not in evidence(result)


async def test_proxima_stacked_xmm_flux_is_flagged(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Proxima Centauri", radius_arcsec=5.0)
    require_catalogs(result, ["xmm"])
    xmm = [p for p in result["points"] if p["catalog"] == "xmm"]
    assert xmm, result["members"]
    assert all(any("moving source in a stacked catalogue" in w for w in p["warnings"]) for p in xmm), xmm
    assert not any(e.startswith("log(fX/fV)") and "XMM-Newton" in e for e in result["classification"]["evidence"])


@pytest.mark.parametrize(("name", "kind", "reliable"), [
    ("Sco X-1", "spec", False),  # rvz_qual E (1995A&AS..114..269D), null rvz_nature
    ("NGC 7027", "spec", False),  # rvz_qual E (1953GCRV)
    ("Sirius", "spec", True),  # rvz_qual A
])
async def test_simbad_radial_velocity_quality(tmp_path: Path, name: str, kind: str, reliable: bool) -> None:
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["simbad"])
    require_lookups(result, ["simbad"])
    simbad = next((c for c in result["redshift"]["candidates"] if c["source"] == "simbad"), None)
    assert simbad is not None, result["redshift"]
    assert simbad["rvz_type"] == "v" and simbad["nature"] is None, simbad  # a radial velocity, rvz_nature null
    assert (simbad["kind"], simbad["reliable"]) == (kind, reliable), simbad


@pytest.mark.parametrize("name", ["Arp 220", "Mrk 231"])
async def test_dusty_starbursts_are_not_radio_loud_agn_by_radio_optical_ratio(tmp_path: Path, name: str) -> None:
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["allwise", "nvss", "first"])
    detail: list[dict[str, Any]] = [d for d in result["classification"]["evidence_detail"]
                                    if d["text"].startswith("radio loudness")]
    assert detail, result["classification"]["evidence"]
    line = detail[0]
    if " > 1 (" in line["text"]:  # R = 1.51 (Arp 220), 1.47 (Mrk 231) in the review
        assert "q22 = log10(F_W4/F_1.4GHz)" in line["text"] and ">= 0.5" in line["text"], line
        assert line["weights"] == {}
