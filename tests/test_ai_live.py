"""Live tests for ai.py against CDS Sesame, SIMBAD TAP, the catalog archives and (when credentials exist) Claude.

Run with:  .venv/Scripts/python.exe -m pytest -m live tests/test_ai_live.py

Only transport failures and HTTP 5xx skip; wrong or empty answers fail. Claude tests
additionally skip when the Anthropic SDK resolves no credentials (API key, auth token,
`ant auth login` profile or Workload Identity Federation).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable
from typing import Any

import httpx
import pytest

import ai
from models import CatalogRegistry, ObjectResolutionError, haversine_arcsec
from providers import SesameResolver

pytestmark = pytest.mark.live

needs_claude = pytest.mark.skipif(not ai.anthropic_configured(), reason="no Anthropic credentials resolved by the SDK")


@pytest.fixture
async def live_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        yield client


async def network[T](awaitable: Awaitable[T]) -> T:
    """Await; skip only on transport errors or upstream 5xx (never on wrong answers)."""
    try:
        return await awaitable
    except httpx.TransportError as exc:
        pytest.skip(f"network unavailable: {exc!r}")
    except ai.UpstreamServiceError as exc:
        text = str(exc)
        if "unreachable" in text or re.search(r"HTTP 5\d\d", text):
            pytest.skip(f"upstream unavailable: {text}")
        raise
    except ObjectResolutionError as exc:
        if str(exc).startswith("Sesame request failed"):
            pytest.skip(f"Sesame unavailable: {exc}")
        raise


def _quantities(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["quantity"]: e for e in entries}


async def test_live_simbad_bibliography_3c273(live_client: httpx.AsyncClient) -> None:
    resolved = await network(SesameResolver(live_client).resolve("3C 273"))
    oid = int(resolved.resolver_metadata["raw_fields"]["oid"][0])
    biblio = await network(ai.fetch_simbad_bibliography(live_client, oid, limit=10))
    for key in ("focused", "recent", "foundational"):
        assert len(biblio[key]) == 10, key
        for entry in biblio[key]:
            assert ai.BIBCODE_RE.fullmatch(entry["bibcode"]), entry["bibcode"]
            assert entry["title"]
    foundational = {e["bibcode"]: e for e in biblio["foundational"]}
    # Schmidt (1963) identified the redshift of 3C 273; Hazard et al. (1963) the lunar-occultation position.
    assert foundational["1963Natur.197.1040S"]["title"].startswith("3C 273: a star-like object with large red-shift")
    assert "1963Natur.197.1037H" in foundational
    assert biblio["recent"][0]["year"] >= 2025
    # Only papers about 3C 273: SIMBAD's ref_flag puts it in the title (bit 1) or abstract (bit 2).
    for key in ("focused", "recent"):
        assert all(e["object_in_title"] or e["object_in_abstract"] for e in biblio[key]), key
    freqs = [e["simbad_obj_freq"] for e in biblio["focused"]]
    assert freqs == sorted(freqs, reverse=True) and freqs[-1] >= 10
    focused = {e["bibcode"] for e in biblio["focused"]}
    assert not focused & {e["bibcode"] for e in biblio["recent"]}  # distinct lists


async def test_live_facts_3c273(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name="3C 273", include_crossmatch=False, max_references=5))
    assert facts.identity["main_id"] == "3C 273"
    assert facts.object["simbad_oid"] == 1940765
    assert "QSO" in facts.identity["all_otypes"]
    assert facts.identity["simbad_reference_count"] > 5000
    measured = _quantities(facts.measurements)
    # SIMBAD's adopted value is z = 0.15756751 +- 0.0005 from BASS DR2 (2022ApJS..261....2K), consistent
    # with Schmidt's (1963) z = 0.158.
    redshift = measured["redshift"]
    assert redshift["value"] == pytest.approx(0.1576, abs=0.001) and redshift["nature"] == "spectroscopic"
    assert "radial_velocity" not in measured  # no relativistic Doppler 'velocity' for a cosmological redshift
    derived = _quantities(facts.derived)
    assert 740 < derived["luminosity_distance"]["value"] < 820
    assert "parallax_distance" not in derived
    photometry = {p["band"]: p for p in facts.photometry}
    assert 12.0 < photometry["G"]["magnitude"] < 13.5  # Gaia G of the ~13th-magnitude quasar
    v_ref = facts.sources[photometry["V"]["ref"] - 1]
    assert v_ref.bibcode and v_ref.n == photometry["V"]["ref"]  # every magnitude carries its reference
    sources = {s.bibcode: s for s in facts.sources if s.bibcode}
    assert sources["2020A&A...641A...6P"].title == "Planck 2018 results. VI. Cosmological parameters."
    assert "2020A&A...641A...1P" in sources  # the CMB dipole used for the CMB-frame redshift
    assert all(s.title for s in facts.sources if s.bibcode)  # every bibcode got its SIMBAD title


async def test_live_facts_m87_by_coordinates(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, ra=187.70593077, dec=12.39112325,
                                          include_crossmatch=False, max_references=3))
    assert facts.identity["main_id"] == "M 87"
    redshift = _quantities(facts.measurements)["redshift"]
    assert redshift["value"] == pytest.approx(0.0042, abs=0.0005)
    assert "luminosity_distance" not in _quantities(facts.derived)
    mpc = [d["value"] for d in facts.distances if d["unit"] == "Mpc"]
    assert mpc and 10 < sorted(mpc)[len(mpc) // 2] < 25  # Virgo cluster, ~16.5 Mpc


@pytest.mark.parametrize("name", ["NGC 4639", "ESO 140-43", "NGC 4486A"])
async def test_live_galaxies_get_no_parallax_distance(live_client: httpx.AsyncClient, name: str) -> None:
    """Gaia parallaxes of galaxy nuclei (NGC 4639: 2.65 +- 0.47 mas) are spurious; distances are Mpc."""
    facts = await network(ai.gather_facts(http_client=live_client, name=name, include_crossmatch=False, max_references=1))
    assert ai.simbad_type_is_extragalactic(facts.identity["otype"], facts.identity["otype_path"])
    assert "parallax_distance" not in _quantities(facts.derived)
    for derived in facts.derived:
        if derived["unit"] == "Mpc":
            assert derived["value"] > 5


async def test_live_galactic_photometric_redshift_gets_no_cosmology(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name="HM Cnc", include_crossmatch=False, max_references=1))
    assert facts.identity["otype"] == "XB*"  # a Galactic double white dwarf binary
    assert not [d for d in facts.derived if d["unit"] in {"Mpc", "Gyr"}]
    z = _quantities(facts.measurements).get("photometric_redshift")
    assert z is None or z["nature"] == "photometric"


async def test_live_star_keeps_its_parallax_distance(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name="Barnard's star", include_crossmatch=False,
                                          max_references=1))
    assert _quantities(facts.derived)["parallax_distance"]["value"] == pytest.approx(1.83, abs=0.01)
    assert _quantities(facts.measurements)["radial_velocity"]["value"] == pytest.approx(-110, abs=2)


async def test_live_explain_by_gaia_epoch_position_finds_barnards_star(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, ra=269.44850252543836, dec=4.739420051112412,
                                          epoch=2016.0, include_crossmatch=False, max_references=1))
    assert facts.identity["main_id"] == "NAME Barnard's star"
    assert facts.query["simbad_match_separation_arcsec"] < 0.5


async def test_live_crossmatch_summary_respects_footprints(live_client: httpx.AsyncClient) -> None:
    """Centaurus A (Dec -43) lies outside NVSS/VLASS/Pan-STARRS: never reported as undetected there."""
    from main import build_service

    facts = await network(ai.gather_facts(http_client=live_client, name="Centaurus A", service=build_service(client=live_client),
                                          max_references=1))
    xm = facts.crossmatch
    assert xm is not None, facts.warnings
    outside = {d["catalog"] for d in xm["outside_coverage"]}
    assert {"nvss", "vlass", "panstarrs_dr2"} <= outside | {f["catalog"] for f in xm["failures"]}
    assert not outside & {d["catalog"] for d in xm["no_source_within_radius"]}
    assert "chandra" in {d["catalog"] for d in xm["detections"]}  # the X-ray nucleus


async def test_live_adql_verification(live_client: httpx.AsyncClient) -> None:
    simbad = CatalogRegistry().get("simbad")
    good = "SELECT TOP 3 b.main_id AS main_id FROM basic AS b WHERE b.plx_value > 500 ORDER BY main_id"
    assert await network(ai.verify_adql_remote(simbad, good, live_client)) is None
    problem = await network(ai.verify_adql_remote(simbad, "SELECT TOP 3 b.nonexistent_col FROM basic AS b", live_client))
    assert problem is not None and "nonexistent_col" in problem


@needs_claude
async def test_live_claude_compiles_radio_quasars_near_m87(live_client: httpx.AsyncClient) -> None:
    registry = CatalogRegistry()
    client = ai.build_anthropic_client()
    compiled = await network(ai.compile_query(
        "quasars near M87 with radio emission", anthropic_client=client, registry=registry, http_client=live_client,
    ))
    assert compiled.scope == "cone" and compiled.advanced_query is not None
    q = compiled.advanced_query
    assert haversine_arcsec(q["target"]["ra"], q["target"]["dec"], 187.70593077, 12.39112325) < 1.0
    assert q["metadata"]["target_source"] == "sesame"
    enabled = registry.enabled_catalogs()
    radio = {n for n, c in enabled.items() if c.wavelength == "radio"}
    selected = set(ai.selected_catalogs(registry, q["catalogs"] or [], q["profiles"] or []))
    assert selected & radio, "radio emission needs the radio catalogs"
    if q["object_types"]:  # a type filter may only be used when every selected catalog reports a type
        assert selected <= set(ai.typed_catalogs(registry, "object_type"))
    assert compiled.plan and compiled.explanation


@needs_claude
async def test_live_claude_typed_coordinates_are_parsed_not_computed(live_client: httpx.AsyncClient) -> None:
    compiled = await network(ai.compile_query(
        "radio sources within 60 arcsec of 12h30m49.42s +12d23m28.0s", anthropic_client=ai.build_anthropic_client(),
        registry=CatalogRegistry(), http_client=live_client,
    ))
    q = compiled.advanced_query
    assert q is not None and q["metadata"]["target_source"] == "user_coordinates"
    assert haversine_arcsec(q["target"]["ra"], q["target"]["dec"], 187.705917, 12.391111) < 0.5


@needs_claude
async def test_live_claude_all_sky_request_gets_verified_adql(live_client: httpx.AsyncClient) -> None:
    registry = CatalogRegistry()
    compiled = await network(ai.compile_query(
        "X-ray bright M dwarfs within 20 pc", anthropic_client=ai.build_anthropic_client(), registry=registry,
        http_client=live_client, verify_adql=True,
    ))
    assert compiled.scope == "all_sky" and compiled.advanced_query is None
    assert compiled.adql is not None and compiled.adql_catalog in ai.tap_catalogs(registry)
    assert ai.validate_adql(compiled.adql_catalog, compiled.adql, registry) == []
    adql = compiled.adql.lower()
    assert re.search(r"(plx_value|parallax)\s*>=?\s*50", adql), "within 20 pc means parallax >= 50 mas"
    assert "'m" in adql, "an M spectral-type constraint is expected"
    result = await network(ai.execute_compiled_query(compiled, http_client=live_client, registry=registry, row_limit=50))
    assert result["kind"] == "adql" and result["rows"], "the verified ADQL should return nearby M dwarfs"
    for row in result["rows"]:
        for key, value in row.items():
            if value is None:
                continue
            if key.lower() in {"plx_value", "parallax"}:
                assert float(value) >= 50.0
            if key.lower() == "sp_type":
                assert str(value).lstrip("sd").upper().startswith("M"), value


@needs_claude
async def test_live_claude_explains_3c273_with_real_citations(live_client: httpx.AsyncClient) -> None:
    result = await network(ai.explain_object(
        http_client=live_client, name="3C 273", anthropic_client=ai.build_anthropic_client(), include_crossmatch=False,
        max_references=5,
    ))
    assert result.summary and len(result.summary.split()) >= 60
    assert len(result.citations) >= 2
    # Every [n] left in the text maps to a returned citation, and every citation is used.
    markers = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", result.summary) for n in group.split(",")}
    assert markers == {c.n for c in result.citations}
    leftover = [m for m in re.findall(r"\[[^\]]*\d[^\]]*\]", result.summary) if not re.fullmatch(r"\[[\d,\s]+\]", m)]
    assert not leftover, leftover
    cited = [c for c in result.citations if c.bibcode]
    assert cited and all(c.url and c.url.startswith("https://ui.adsabs.harvard.edu/abs/") for c in cited)
    assert re.search(r"0\.15[78]", result.summary), "the SIMBAD redshift should be reported"
    assert not any("matches no provided source" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Regression truths (review round 2)
# ---------------------------------------------------------------------------


async def test_live_proxima_focused_papers_are_about_proxima(live_client: httpx.AsyncClient) -> None:
    """Passing mentions (obj_freq 1, ref_flag 128) must not be listed as papers studying Proxima Cen."""
    facts = await network(ai.gather_facts(http_client=live_client, name="Proxima Centauri", include_crossmatch=False,
                                          max_references=6))
    assert facts.object["simbad_oid"] == 3379714
    for key in ("focused", "recent"):
        entries = facts.bibliography[key]
        assert entries, key
        assert all(e["object_in_title"] or e["object_in_abstract"] for e in entries), key
    titled = [e for e in facts.bibliography["focused"] if e["object_in_title"]]
    assert titled and all(re.search(r"proxima", e["title"], re.IGNORECASE) for e in titled)


async def test_live_m1_answered_by_ned_is_the_crab_nebula(live_client: httpx.AsyncClient) -> None:
    """Sesame's NED answer for M 1 is the pulsar's position; SIMBAD's identifier table still gives the SNR."""
    resolver = SesameResolver(live_client, endpoint="https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/N")
    facts = await network(ai.gather_facts(http_client=live_client, name="M 1", include_crossmatch=False,
                                          max_references=1, resolver=resolver))
    assert facts.identity["main_id"] == "M 1" and facts.identity["otype"] == "SNR"
    assert facts.object["simbad_oid"] == 795871
    assert "V* CM Tau" not in facts.identity["main_id"]


async def test_live_ngc4993_hubble_flow_error_includes_peculiar_velocity(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name="NGC 4993", include_crossmatch=False,
                                          max_references=1))
    dl = _quantities(facts.derived)["luminosity_distance"]
    assert 45 < dl["value"] < 52 and dl["plus_error"] >= 3.5 and dl["minus_error"] >= 3.5
    # GW170817 host: the SBF distance (~40.7 Mpc) is consistent within the honest Hubble-flow error.
    mpc = [d for d in facts.distances if d["unit"] == "Mpc"]
    assert any(abs(d["value"] - dl["value"]) < 2.5 * dl["minus_error"] for d in mpc)
    assert all(d["minus_error"] is None or d["minus_error"] >= 0 for d in facts.distances)


@pytest.mark.parametrize("name", ["NGC 6621", "NGC 4807"])
async def test_live_candidate_and_pair_galaxies_get_cosmology(live_client: httpx.AsyncClient, name: str) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name=name, include_crossmatch=False,
                                          max_references=1))
    assert facts.identity["otype_path"].startswith("G")
    assert not any("not one of the galaxy/AGN classes" in w for w in facts.warnings)
    assert 80 < _quantities(facts.derived)["luminosity_distance"]["value"] < 125  # z ~ 0.02


async def test_live_cyg_x1_parallax_distance_states_gaia_zero_point(live_client: httpx.AsyncClient) -> None:
    """Cyg X-1 (HXB; a stray 'BL?' candidate type) keeps its parallax distance, ~2.2 kpc, with errors and the ZP note."""
    facts = await network(ai.gather_facts(http_client=live_client, name="Cyg X-1", include_crossmatch=False,
                                          max_references=1))
    dist = _quantities(facts.derived)["parallax_distance"]
    assert 2000 < dist["value"] < 2500 and 50 < dist["minus_error"] < dist["plus_error"] < 150
    assert "zero-point" in dist["note"] and "larger than the formal parallax error" in dist["note"]


async def test_live_betelgeuse_parallax_distance_errors(live_client: httpx.AsyncClient) -> None:
    facts = await network(ai.gather_facts(http_client=live_client, name="Betelgeuse", include_crossmatch=False,
                                          max_references=1))
    dist = _quantities(facts.derived)["parallax_distance"]
    assert 140 < dist["value"] < 170  # Hipparcos 6.55 +- 0.83 mas
    assert dist["plus_error"] > dist["minus_error"] > 10


async def test_live_gn_z11_lone_field_rows_are_not_associated(live_client: httpx.AsyncClient) -> None:
    from main import build_service

    facts = await network(ai.gather_facts(http_client=live_client, name="GN-z11", service=build_service(client=live_client),
                                          max_references=1))
    xm = facts.crossmatch
    assert xm is not None, facts.warnings
    for entry in xm["detections"] + xm["ambiguous"]:
        assert entry["separation_arcsec"] <= entry["association_radius_arcsec"]
        # P = 0 only for a row exactly at the position (a lone row gets null, never 0).
        assert entry["chance_coincidence_probability"] != 0.0 or entry["separation_arcsec"] == 0.0
    assert "unlikely to be a chance coincidence" not in str(xm)
    assert not {"multi", "extragalactic", "exoplanet"} & set(xm["wavelengths_detected"])


@needs_claude
async def test_live_claude_compact_designation_is_parsed(live_client: httpx.AsyncClient) -> None:
    compiled = await network(ai.compile_query(
        "radio sources within 60 arcsec of J123049.42+122328.0", anthropic_client=ai.build_anthropic_client(),
        registry=CatalogRegistry(), http_client=live_client,
    ))
    q = compiled.advanced_query
    assert q is not None and q["metadata"]["target_source"] == "user_coordinates"
    assert haversine_arcsec(q["target"]["ra"], q["target"]["dec"], 187.705917, 12.391111) < 0.5
