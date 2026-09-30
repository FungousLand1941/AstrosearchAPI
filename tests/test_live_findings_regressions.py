"""Offline regressions for conditions first seen in the live suite.

* CDS Sesame answering a name from its VizieR-local fallback (an undated position without motion):
  the light-curve result says so, as the crossmatch search does, instead of silently returning no series;
* the MAST catalogs API returning Pan-STARRS objID as a string in one answer and a number in another:
  both become one int, so a sky-cache mirror and the archive give the same record and content hash.
"""

from __future__ import annotations

import main
import providers
import timedomain as td

VIZIER_LOCAL = {"query": "GJ 3622", "canonical_name": "GJ 3622",
                "resolver_metadata": {"resolver_name": "Vl=VizieR (local)",
                                      "simbad_retry": {"error": "Nothing found"}}}
SIMBAD = {"query": "GJ 3622", "resolver_metadata": {"resolver_name": "Sc=Simbad (via url)"}}


def test_light_curves_warn_when_sesame_answered_from_vizier() -> None:
    warning = td.resolver_fallback_warning(VIZIER_LOCAL)
    assert warning and "not SIMBAD" in warning and "Vl=VizieR (local)" in warning and "GJ 3622" in warning
    assert "moving object" in warning
    assert td.resolver_fallback_warning(SIMBAD) is None
    assert td.resolver_fallback_warning({"resolver_metadata": {}}) is None


def test_panstarrs_objid_is_one_integer_whatever_the_archive_sends() -> None:
    catalog = main.build_registry().get("panstarrs_dr2")
    names = providers._integer_id_columns(catalog)
    assert "objID" in names
    as_text = {"objID": "110561872774570857", "raMean": 187.27, "decMean": 2.05}
    as_number = {"objID": 110561872774570857, "raMean": 187.27, "decMean": 2.05}
    providers._integer_ids(as_text, names)
    providers._integer_ids(as_number, names)
    assert as_text == as_number and isinstance(as_text["objID"], int)
    # Not an integer: left alone.
    odd = {"objID": "PSO J1"}
    providers._integer_ids(odd, names)
    assert odd == {"objID": "PSO J1"}
