"""Fix-round regressions for sed.py (offline), on top of tests/test_sed_round3.py.

1. Identification by designation: a 2MASS/AllWISE/Gaia DR3 row whose designation is a name of the object (Sesame
   query, main identifier, aliases, or the chosen SIMBAD/NED entry) is used even beyond the positional tolerance. The
   T8 dwarf 2MASS J04151954-0935066 lost its own 2MASS row: SIMBAD lists it at its 2MASS epoch-1998.9 position
   labelled J2000, so after epoch propagation (2.26 arcsec/yr) the row comes out 2.53 arcsec away (> 1.5 arcsec).
2. A name together with coordinates is rejected by the library entry point and the CLI too (not only the router).
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest
from test_sed_regressions import passport
from test_sed_round3 import GAIA_ROW, TMASS_ROW, WISE_ROW, bands, record, row, sed_of, used

import sed


@pytest.fixture(autouse=True)
def _fresh_backoff() -> Any:
    sed.default_backoff().clear()
    yield
    sed.default_backoff().clear()


# ---------------------------------------------------------------------------
# 1. Identification by designation
# ---------------------------------------------------------------------------


def test_catalog_designations() -> None:
    assert sed.catalog_designations("twomass_psc", "04151954-0935066") == {"2MASS J04151954-0935066",
                                                                          "2MASS 04151954-0935066"}
    assert sed.catalog_designations("allwise", "J041521.26-093500.4") == {"WISEA J041521.26-093500.4"}
    assert sed.catalog_designations("gaia_dr3", "4034171629042489088") == {"GAIA DR3 4034171629042489088"}
    assert sed.catalog_designations("sdss", "1237664819285852192") == set()  # objIDs are not designations
    assert sed.catalog_designations("twomass_psc", "  ") == set()


def test_designated_row_beyond_tolerance_is_used_and_reported() -> None:
    resolved = {"query": "2MASS J04151954-0935066", "canonical_name": "2MASSI J0415195-093506", "aliases": []}
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("simbad", "2MASSI J0415195-093506", 0.0, {"otype": "BD*"})]}
    moved = {"group_id": "object-3", "members": [row("twomass_psc", "04151954-0935066", 2.53, TMASS_ROW, 0.11),
                                                row("allwise", "J041521.26-093500.4", 3.4, WISE_ROW)]}
    result = sed_of(record(target, moved, resolved=resolved))
    tmass = used(result, "twomass_psc")
    assert tmass["designation"] == "2MASS J04151954-0935066" and tmass["tolerance_arcsec"] < 2.53
    assert "identified by designation: '2MASS J04151954-0935066' is a name of the object" in tmass["reason"]
    assert "its designation identifies it as the target" in tmass["reason"]
    assert bands(result, "2MASS") == {"J", "H", "Ks"}
    # Without a name match the same offset row stays out (3.4" > the 3" AllWISE tolerance).
    wise = next(m for m in result["members"] if m["catalog"] == "allwise")
    assert not wise["used"] and "designation" not in wise and not bands(result, "WISE")
    note = next(n for n in result["notes"] if n.startswith("members beyond the positional tolerance"))
    assert "twomass_psc 04151954-0935066 (2.53 arcsec > 1.50 arcsec)" in note
    # A coordinate query has no resolved name and the SIMBAD main identifier is not the 2MASS designation.
    anonymous = sed_of(record(target, moved))
    assert not next(m for m in anonymous["members"] if m["catalog"] == "twomass_psc")["used"]
    assert not any(n.startswith("members beyond the positional tolerance") for n in anonymous["notes"])


def test_designation_from_the_chosen_ned_entry_and_aliases() -> None:
    """Sesame aliases are names of the object (used beyond tolerance). The chosen NED entry's name (e.g.
    '2MASS J10491891-5319100' for Luhman 16) is itself a positional match: it only picks among rows within tolerance.
    A NED row that was NOT chosen (a sub-component) never names anything."""
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("ned", "2MASS J10491891-5319100", 0.0, {"prefphytype": "*"}),
        row("ned", "WISEA J104915.52-531906.1", 2.5, {"prefphytype": "IrS"})]}
    beyond = {"group_id": "object-2", "members": [
        row("twomass_psc", "10491891-5319100", 1.9, TMASS_ROW, 0.06),
        row("allwise", "J104915.52-531906.1", 3.2, WISE_ROW),
        row("gaia_dr3", "5353625852001928960", 1.4, GAIA_ROW, 0.01)]}
    result = sed_of(record(target, beyond))
    for catalog in ("twomass_psc", "allwise", "gaia_dr3"):
        member = next(m for m in result["members"] if m["catalog"] == catalog)
        assert not member["used"] and "designation" not in member, member
    # Within tolerance the NED entry's designation wins over a nearer row of the target group.
    near = {"group_id": "object-1", "contains_target": True, "members": [
        row("ned", "2MASS J10491891-5319100", 0.0, {"prefphytype": "*"}),
        row("twomass_psc", "10491880-5319090", 0.4, dict(TMASS_ROW, j_m=16.0), 0.06)]}
    other = {"group_id": "object-2", "members": [row("twomass_psc", "10491891-5319100", 1.2, TMASS_ROW, 0.06)]}
    picked = used(sed_of(record(near, other)), "twomass_psc")
    assert picked["source_id"] == "10491891-5319100" and picked["designation"] == "2MASS J10491891-5319100"
    assert "is the name of the chosen SIMBAD/NED entry; 1.200 arcsec <= tolerance" in picked["reason"]
    # A Sesame alias is authoritative: the Gaia row 1.4" away is the object.
    aliased = sed_of(record(target, beyond, resolved={"query": "Luhman 16", "canonical_name": "NAME Luhman 16",
                                                      "aliases": ["Gaia DR3 5353625852001928960"]}))
    gaia = used(aliased, "gaia_dr3")
    assert gaia["designation"] == "Gaia DR3 5353625852001928960"
    assert "is a name of the object resolved by Sesame; 1.400 arcsec > tolerance 1.00 arcsec" in gaia["reason"]
    assert bands(aliased, "Gaia") == {"G", "BP", "RP"}


def test_designated_row_wins_over_a_nearer_row() -> None:
    resolved = {"query": "2MASS J11525880+3743060", "canonical_name": "HD 103095", "aliases": []}
    target = {"group_id": "object-1", "contains_target": True, "members": [
        row("twomass_psc", "11525871+3743065", 0.3, dict(TMASS_ROW, j_m=15.0), 0.06)]}
    other = {"group_id": "object-2", "members": [row("twomass_psc", "11525880+3743060", 0.6, TMASS_ROW, 0.06)]}
    result = sed_of(record(target, other, resolved=resolved))
    tmass = used(result, "twomass_psc")
    assert tmass["source_id"] == "11525880+3743060" and "<= tolerance" in tmass["reason"]
    note = next(n for n in result["notes"] if "11525871+3743065" in n)
    assert "also within tolerance but not used" in note and "11525880+3743060 (0.60 arcsec) was chosen" in note


def test_t8_dwarf_keeps_its_own_2mass_row(tmp_path_factory: pytest.TempPathFactory) -> None:
    result = passport("t8dwarf", tmp_path_factory)
    tmass = used(result, "twomass_psc")
    assert tmass["source_id"] == "04151954-0935066" and tmass["designation"] == "2MASS J04151954-0935066"
    assert tmass["separation_arcsec"] == pytest.approx(2.529, abs=0.01) and tmass["tolerance_arcsec"] == 1.5
    assert bands(result, "2MASS") == {"J", "H", "Ks"} and bands(result, "WISE") == {"W1", "W2", "W3", "W4"}
    j = next(p for p in result["points"] if p["facility"] == "2MASS" and p["band"] == "J")
    assert j["magnitude"] == pytest.approx(15.695, abs=1e-3)
    # The epoch-shifted PS1 row (2.87") has no designation match and stays out.
    assert not next(m for m in result["members"] if m["catalog"] == "panstarrs_dr2")["used"]
    assert result["classification"]["label"] == "star"


# ---------------------------------------------------------------------------
# 2. Name together with coordinates
# ---------------------------------------------------------------------------


def test_build_sed_rejects_a_name_with_coordinates() -> None:
    class NeverCalled:
        async def crossmatch(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("no crossmatch for an ambiguous target")

    for kwargs in ({"ra": 187.7, "dec": 12.4, "name": "M87"}, {"ra": 187.7, "name": "M87"}, {"ra": 187.7},
                   {"dec": 12.4}, {}):
        with pytest.raises(sed.SEDInputError):
            asyncio.run(sed.build_sed(service=NeverCalled(), supplementary=False, **kwargs))


def test_cli_rejects_a_name_with_coordinates(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    sed.register_cli(parser.add_subparsers(dest="command"))

    async def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("build_sed must not run")

    monkeypatch.setattr(sed, "build_sed", never)
    args = parser.parse_args(["sed", "--name", "M87", "--ra", "187.7", "--dec", "12.4"])
    assert args.handler(args) == 2
    assert "not both" in capsys.readouterr().out
