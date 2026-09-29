"""Round-3 live truths for sed.py (run with ``pytest -m live``).

Asserted against the real archives: high proper-motion stars keep the survey photometry the crossmatch put in other
groups (Groombridge 1830 2MASS, Kapteyn's star AllWISE, Luhman 16 2MASS + AllWISE); 51 Peg's SIMBAD member is the
star (not '* 51 Peg b') with its Doppler z of about -1.1e-4; M87 keeps its Gaia nucleus and 3C 273 its NED quasar row;
the supernova SN 1998bw is not a star and its host redshift is not declared erroneous; the SIMBAD otypedef table
matches the embedded copy; AD Leo's Gaia DSC stellar probability counts the binary class; NGC 4151 W3 = 3.2 and
Polaris W3 = 0.6 are valid profile-fit measurements; an SDSS quasar with a NED '*' row is still a quasar. Tests skip
only when a service is unreachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from test_sed_live import build, require_catalogs, require_lookups

import sed

pytestmark = pytest.mark.live


def used(result: dict[str, Any], catalog: str) -> dict[str, Any] | None:
    return next((m for m in result["members"] if m["catalog"] == catalog and m["used"]), None)


def bands(result: dict[str, Any], facility: str) -> set[str]:
    return {p["band"] for p in result["points"] if p["facility"].startswith(facility)}


def evidence(result: dict[str, Any]) -> str:
    return " | ".join(result["classification"]["evidence"])


async def test_groombridge_1830_has_2mass_photometry(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Groombridge 1830")
    require_catalogs(result, ["twomass_psc", "gaia_dr3", "simbad"])
    tmass = used(result, "twomass_psc")
    assert tmass is not None and tmass["source_id"] == "11525880+3743060", result["members"]
    assert bands(result, "2MASS") == {"J", "H", "Ks"}
    assert result["classification"]["label"] == "star"


async def test_kapteyns_star_has_allwise_photometry_and_the_host_row(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Kapteyn's star")
    require_catalogs(result, ["allwise", "simbad"])
    wise = used(result, "allwise")
    assert wise is not None and wise["source_id"] == "J051146.81-450204.5", result["members"]
    assert bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    simbad = used(result, "simbad")
    assert simbad is not None and simbad["source_id"] == "HD 33793"
    assert result["classification"]["label"] == "star"


async def test_luhman_16_has_2mass_and_allwise_photometry(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Luhman 16")
    require_catalogs(result, ["twomass_psc", "allwise", "simbad"])
    assert bands(result, "2MASS") == {"J", "H", "Ks"} and bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    assert len(result["points"]) > 3
    assert result["classification"]["label"] == "star"


async def test_51_peg_member_is_the_star_with_its_doppler_redshift(tmp_path: Path) -> None:
    result = await build(tmp_path, name="51 Peg")
    require_catalogs(result, ["simbad"])
    require_lookups(result, ["simbad"])
    simbad = used(result, "simbad")
    assert simbad is not None and simbad["source_id"] == "* 51 Peg"
    z = result["redshift"]
    # 51 Peg: radial velocity -33.2 km/s -> z = -1.1e-4 (a Doppler shift, not a cosmological redshift).
    assert z["value"] == pytest.approx(-1.1e-4, abs=2e-5) and z["source"] == "simbad" and "Doppler" in z["note"]
    assert result["classification"]["label"] == "star"
    assert "galaxy morphology" not in evidence(result)


async def test_m87_keeps_its_gaia_nucleus(tmp_path: Path) -> None:
    result = await build(tmp_path, name="M87")
    require_catalogs(result, ["gaia_dr3", "ned", "simbad"])
    gaia = used(result, "gaia_dr3")
    assert gaia is not None and gaia["separation_arcsec"] < 0.5, result["members"]
    assert bands(result, "Gaia") == {"G", "BP", "RP"}
    assert used(result, "ned") is not None
    assert result["classification"]["label"] in {"galaxy", "agn"}
    assert result["redshift"]["value"] == pytest.approx(0.0043, abs=5e-4)


async def test_3c273_keeps_its_ned_quasar_row(tmp_path: Path) -> None:
    result = await build(tmp_path, name="3C 273")
    require_catalogs(result, ["ned", "gaia_dr3"])
    ned = used(result, "ned")
    assert ned is not None and ned["source_id"] == "3C 273"
    assert used(result, "gaia_dr3") is not None
    assert result["classification"]["label"] == "qso"
    assert result["redshift"]["value"] == pytest.approx(0.158, abs=0.001)


async def test_sn_1998bw_is_not_a_star(tmp_path: Path) -> None:
    result = await build(tmp_path, name="SN 1998bw")
    require_catalogs(result, ["simbad"])
    assert result["classification"]["label"] != "star", result["classification"]
    assert "transient" in evidence(result)
    z = result["redshift"]
    # Host galaxy ESO 184-G82: z = 0.0085 (Tinney et al. 1998).
    assert z["value"] == pytest.approx(0.0085, abs=3e-4) and not z.get("conflict")


async def test_simbad_otypedef_matches_the_embedded_table() -> None:
    query = "SELECT otype, path, is_candidate FROM otypedef"
    try:
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            response = await client.post(sed.SIMBAD_TAP_URL, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
                                                                   "QUERY": query})
    except httpx.TransportError as exc:
        pytest.skip(f"SIMBAD unreachable: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"SIMBAD HTTP {response.status_code}")
    rows = {r[0]: (r[1], bool(r[2])) for r in response.json()["data"]}
    assert rows == sed.SIMBAD_OTYPEDEF


async def test_ad_leo_dsc_counts_the_binary_class(tmp_path: Path) -> None:
    result = await build(tmp_path, name="AD Leo")
    require_catalogs(result, ["gaia_dr3"])
    require_lookups(result, ["gaia"])
    assert result["gaia_extra"]["classprob_dsc_combmod_binarystar"] > 0.99
    line = next(t for t in result["classification"]["evidence"] if "DSC-Combmod" in t)
    assert "-> star = 1.000" in line
    assert result["classification"]["label"] == "star"


@pytest.mark.parametrize(("name", "w3_max"), [("NGC 4151", 4.0), ("Polaris", 1.5)])
async def test_bright_w3_profile_fit_magnitudes_are_valid(tmp_path: Path, name: str, w3_max: float) -> None:
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["allwise"])
    w3 = next(p for p in result["points"] if p["facility"] == "WISE" and p["band"] == "W3")
    assert w3["magnitude"] < w3_max and not w3["is_upper_limit"]
    assert not any("saturation" in w or "VI.3.d" in w for w in w3["warnings"]), w3["warnings"]


async def test_sdss_quasar_with_a_ned_star_row_is_a_quasar(tmp_path: Path) -> None:
    # SDSS DR18 quasar z = 2.38; NED lists WISEA J100058.51+030252.7 there with type '*' and z = 2.374.
    result = await build(tmp_path, ra=150.24369, dec=3.04813, radius_arcsec=3.0)
    require_catalogs(result, ["ned"])
    text = evidence(result)
    assert "NED preferred type '*' -> star" not in text
    ned_lines = [t for t in result["classification"]["evidence"] if t.startswith("NED preferred type")]
    assert not any(t.endswith("-> star") for t in ned_lines), ned_lines
    ned = used(result, "ned")
    for line in ned_lines:
        if line.startswith("NED preferred type '*'"):
            # NED still types the row '*' with a quasar redshift: the stellar type is reported as ignored, not counted.
            assert ned is not None and f"ignored: the same NED row ({ned['source_id']})" in line, line
            assert "internally inconsistent NED entry" in line
    assert result["classification"]["label"] == "qso", result["classification"]
    assert result["redshift"]["value"] == pytest.approx(2.37, abs=0.02)
