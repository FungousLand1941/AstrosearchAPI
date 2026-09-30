"""Live regression tests for the round-3 review of ai.py (real SIMBAD, Sesame, NED and archives).

Run with:  .venv/Scripts/python.exe -m pytest -m live tests/test_ai_round3_live.py

Only transport failures and HTTP 5xx skip; wrong or empty answers fail. Claude tests
also skip when the Anthropic SDK resolves no credentials.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from astropy.cosmology import Planck18
from test_ai import FakeAnthropic, reply, submission, tool_use
from test_ai_live import needs_claude, network

import ai
from models import CatalogRegistry, haversine_arcsec

pytestmark = pytest.mark.live

C3C273_ICRS = (187.27791594, 2.05238823)


@pytest.fixture
async def live_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        yield client


def _q(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["quantity"]: e for e in entries}


async def _live_facts(client: httpx.AsyncClient, name: str) -> ai.ObjectFacts:
    return await network(ai.gather_facts(http_client=client, name=name, include_crossmatch=False, max_references=2))


@pytest.mark.parametrize(("name", "otype", "min_pc", "max_pc"), [
    ("NGC 7293", "PN", 190, 210),  # Helix Nebula, Gaia ~200 pc
    ("Mira", "Mi*", 80, 105),  # Hipparcos 10.91 mas
    ("47 Tuc", "GlC", 4000, 4700),  # Gaia EDR3 cluster parallax ~0.23 mas
    ("M 13", "GlC", 7000, 9000),
    ("NGC 6543", "PN", 1250, 1500),  # Cat's Eye 0.73 mas
])
async def test_live_galactic_objects_with_stray_galaxy_types_keep_parallax_distances(
    live_client: httpx.AsyncClient, name: str, otype: str, min_pc: float, max_pc: float
) -> None:
    facts = await _live_facts(live_client, name)
    assert facts.identity["otype"] == otype
    parallax = _q(facts.measurements)["parallax"]
    assert "extragalactic" not in (parallax.get("note") or "")
    dist = _q(facts.derived)["parallax_distance"]
    assert min_pc < dist["value"] < max_pc
    assert not any("withheld" in w for w in facts.warnings)


async def test_live_ton618_distance_error_reflects_redshift_precision(live_client: httpx.AsyncClient) -> None:
    facts = await _live_facts(live_client, "Ton 618")
    z = _q(facts.measurements)["redshift"]
    assert z["value"] == pytest.approx(2.2, abs=0.05)
    derived = _q(facts.derived)
    if "error" not in z:  # SIMBAD states z = 2.2 without an error (checked 2026-09)
        assert derived["redshift_cmb_frame"]["error_inferred"] is True
        dl = derived["luminosity_distance"]
        assert dl["plus_error"] >= 200 and dl["minus_error"] >= 200
        z_better = 2.219
        assert abs((1 + z_better) * Planck18.comoving_transverse_distance(z_better).value - dl["value"]) < dl["plus_error"]
    assert "morphological_type" not in _q(facts.measurements) or any(
        c.isalnum() for c in _q(facts.measurements)["morphological_type"]["value"])


@pytest.mark.parametrize("name", ["NGC 6086", "NGC 315"])
async def test_live_galaxy_velocities_become_cz_over_c(live_client: httpx.AsyncClient, name: str) -> None:
    facts = await _live_facts(live_client, name)
    measured = _q(facts.measurements)
    velocity = measured["radial_velocity"]["value"]
    assert velocity > 3000  # both are catalogued as velocities (rvz_type v)
    assert measured["redshift"]["value"] == pytest.approx(velocity / ai.SPEED_OF_LIGHT_KMS, abs=2e-6)
    derived = _q(facts.derived)
    assert derived["luminosity_distance"]["plus_error"] > 3  # peculiar velocity included


async def test_live_pleiades_dr2_parallax_systematics(live_client: httpx.AsyncClient) -> None:
    facts = await _live_facts(live_client, "Pleiades")
    dist = _q(facts.derived)["parallax_distance"]
    assert 130 < dist["value"] < 140
    if dist.get("gaia_release") == "Gaia DR2":  # the 2018 Gaia DR2 HRD cluster parallax (checked 2026-09)
        assert "below the 0.04 mas" in dist["note"] and "-0.029 mas" in dist["note"]
    else:
        assert dist.get("gaia_release"), dist  # a Gaia-based cluster parallax in any case


async def test_live_otypedef_quasar_and_star_expansion(live_client: httpx.AsyncClient) -> None:
    ai._OTYPE_TREES.clear()
    tree = await network(ai.simbad_otype_tree(live_client))
    quasar, errors = ai.expand_object_types(["quasar"], ["simbad", "ned"], tree)
    assert not errors and {"QSO", "Q?", "Bla", "Bz?", "BLL", "BL?"} <= set(quasar)
    star, errors = ai.expand_object_types(["star"], ["simbad"], tree)
    assert not errors and {"PM*", "Mi*", "HXB", "WD*"} <= set(star) and "PN" not in star


def _counterpart_ids(record: dict[str, Any], catalog: str) -> list[str]:
    return [" ".join(str(s["source_id"]).split()) for sources in record["counterparts"].values() for s in sources
            if s["catalog"] == catalog]


@pytest.mark.parametrize(("request_text", "name", "types", "catalogs", "radius", "catalog", "expected_id"), [
    ("quasars near 3C 273 in SIMBAD and NED", "3C 273", ["quasar"], ["simbad", "ned"], 30.0, "simbad", "3C 273"),
    ("quasars near 3C 273 in SIMBAD and NED", "3C 273", ["quasar"], ["simbad", "ned"], 30.0, "ned", "3C 273"),
    ("stars within 2 arcmin of Proxima Centauri", "Proxima Centauri", ["star"], ["simbad"], 120.0, "simbad",
     "NAME Proxima Centauri"),
])
async def test_live_compiled_type_filter_keeps_the_real_object(
    live_client: httpx.AsyncClient, request_text: str, name: str, types: list[str], catalogs: list[str], radius: float,
    catalog: str, expected_id: str,
) -> None:
    """Compile (scripted Claude, live Sesame + otypedef) and execute against the live archives."""
    from main import build_service

    fake = FakeAnthropic(reply(tool_use("submit_query", submission(
        target={"name": name, "coordinates": None}, radius_arcsec=radius, catalogs=catalogs, object_types=types))))
    service = build_service(client=live_client)
    registry = getattr(service, "registry", None) or CatalogRegistry()
    compiled = await network(ai.compile_query(request_text, anthropic_client=fake, registry=registry,
                                              http_client=live_client, settings=ai.AISettings()))
    result = await network(ai.execute_compiled_query(compiled, service=service, http_client=live_client,
                                                     registry=registry))
    record = result["record"]
    failed = [f for f in record.get("failures") or [] if f.get("catalog") == catalog]
    if failed:
        kind = f"{failed[0].get('error_type')} {failed[0].get('message')}".lower()
        if any(k in kind for k in ("timeout", "unavailable", "connect", "network", "http 5", "status 5")):
            pytest.skip(f"{catalog} unavailable (transport/5xx): {failed[0]}")
    assert not failed, failed
    assert expected_id in _counterpart_ids(record, catalog), _counterpart_ids(record, catalog)


@needs_claude
async def test_live_claude_b1950_designation_does_not_shift_the_position(live_client: httpx.AsyncClient) -> None:
    text = "radio sources within 60 arcsec of 12h29m06.7s +02d03m09s, i.e. the quasar with B1950 designation 1226+023"
    compiled = await network(ai.compile_query(text, anthropic_client=ai.build_anthropic_client(),
                                              registry=CatalogRegistry(), http_client=live_client))
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *C3C273_ICRS) < 60.0


@needs_claude
async def test_live_claude_distance_limited_xray_request_is_executable(live_client: httpx.AsyncClient) -> None:
    compiled = await network(ai.compile_query(
        "X-ray emitting stars closer than 20 pc within 30 arcmin of 12h30m49.42s +12d23m28.0s",
        anthropic_client=ai.build_anthropic_client(), registry=CatalogRegistry(), http_client=live_client))
    q = compiled.advanced_query
    if q is not None and q["search_mode"] == "cylinder":
        assert set(q["catalogs"]) <= {"gaia_dr3", "simbad"} and q["catalogs"]
    if q is not None and q["search_mode"] != "cylinder":
        assert q["min_distance_pc"] is None and q["max_distance_pc"] is None
    if compiled.adql and compiled.scope == "cone":
        assert "CIRCLE('ICRS', 187.705" in compiled.adql
