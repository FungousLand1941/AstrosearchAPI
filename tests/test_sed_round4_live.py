"""Fix-round live truths for sed.py (run with ``pytest -m live``).

Asserted against the real archives: the T8 dwarf 2MASS J04151954-0935066 keeps its own 2MASS row (identified by its
designation although SIMBAD's epoch-mislabelled position puts it 2.5 arcsec away) and its AllWISE photometry, and its
methane W1-W2 colour is not taken for an AGN; SN 2006gy is its host galaxy NGC 1260 at z = 0.019, not a star or a
quasar; the SIMBAD 'Sy?' (symbiotic star candidate) 2MASS J06390585+0946508 gets star, not AGN, evidence; the Einstein
Cross Q2237+030 gets a label consistent with the adopted redshift. Tests skip only when a service is unreachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_sed_live import build, require_catalogs
from test_sed_round3_live import bands, evidence, used

pytestmark = pytest.mark.live


async def test_t8_dwarf_has_its_own_2mass_row(tmp_path: Path) -> None:
    result = await build(tmp_path, name="2MASS J04151954-0935066")
    require_catalogs(result, ["twomass_psc", "allwise", "simbad"])
    tmass = used(result, "twomass_psc")
    assert tmass is not None and tmass["source_id"] == "04151954-0935066", result["members"]
    assert tmass["designation"] == "2MASS J04151954-0935066"
    assert bands(result, "2MASS") == {"J", "H", "Ks"} and bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    line = next(t for t in result["classification"]["evidence"] if t.startswith("WISE W1-W2"))
    assert "brown dwarfs" in line and "not applied" in line
    assert result["classification"]["label"] == "star"


async def test_sn_2006gy_is_its_host_galaxy(tmp_path: Path) -> None:
    result = await build(tmp_path, name="SN 2006gy")
    require_catalogs(result, ["simbad", "ned"])
    assert "SIMBAD object type 'SN*'" in evidence(result) and "transient" in evidence(result)
    assert result["classification"]["label"] == "galaxy", result["classification"]
    # NGC 1260: cz = 5750 km/s (z = 0.0192).
    assert result["redshift"]["value"] == pytest.approx(0.0192, abs=5e-4) and result["redshift"]["reliable"] is True


async def test_symbiotic_star_candidate_is_star_evidence(tmp_path: Path) -> None:
    result = await build(tmp_path, name="2MASS J06390585+0946508")
    require_catalogs(result, ["simbad"])
    detail = next(d for d in result["classification"]["evidence_detail"] if d["text"].startswith("SIMBAD object type"))
    assert "'Sy?'" in detail["text"] and "-> star" in detail["text"], detail
    assert detail["weights"] == {"star": 1.5}
    assert result["classification"]["label"] == "star"


async def test_einstein_cross_label_agrees_with_its_redshift(tmp_path: Path) -> None:
    # Q2237+030: quasar images at z = 1.695 lensed by the bulge of the galaxy CGCG 378-015 at z = 0.0394.
    result = await build(tmp_path, name="Q2237+030")
    require_catalogs(result, ["simbad", "ned"])
    label, z = result["classification"]["label"], result["redshift"]["value"]
    assert label in {"galaxy", "qso"}, result["classification"]
    if label == "galaxy":
        assert z == pytest.approx(0.0394, abs=0.002), result["redshift"]
    else:
        assert z == pytest.approx(1.695, abs=0.01), result["redshift"]
