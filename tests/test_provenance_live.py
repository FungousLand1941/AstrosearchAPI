"""Live provenance tests (run with ``pytest -m live``).

* A real 3C 273 crossmatch (10 arcsec, every enabled catalog) is manifested and then
  replayed against the live archives: the content hash must be identical, or every
  difference must be explained by a continuously updated database.
* Name searches through ``main.search_object`` (the CLI ``search --name`` path, whose
  Sesame proper motion is labelled "resolver") of Barnard's star (10.4"/yr) and of the
  galaxy M87 (pm = 0 from the resolver) are manifested and replayed: every recorded
  request must equal its reconstruction and the replay must send the same requests.
* HEASARC's live TAP_SCHEMA descriptions identify the data releases queried.
* Every bibcode/DOI in provenance.REFERENCES must exist at NASA ADS / doi.org with the
  recorded volume, first page and year.

Only unreachable services (network errors, timeouts, HTTP 5xx/429) skip; wrong or empty
answers fail.
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
from fixture_io import TARGETS

import main
import provenance as P
from crossmatch import CrossmatchService
from models import CatalogRegistry
from providers import CacheManager, provider_map

pytestmark = pytest.mark.live

RA_3C273, DEC_3C273 = TARGETS["3c273"]


def _fresh_service(client: httpx.AsyncClient) -> CrossmatchService:
    # No response cache: both runs really ask the archives.
    return CrossmatchService(CatalogRegistry(), provider_map(client, timeout=120.0, cache=CacheManager(None)),
                             radius_arcsec=10.0, timeout=150.0)


async def _live_manifest_and_replay():
    os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "2"  # polite to shared endpoints
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        service = _fresh_service(client)
        record = await service.crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
        manifest = P.build_manifest(record, registry=service.registry)
        unreachable = [f for f in record.failures if f.get("error_type") in P.UNREACHABLE_ERRORS]
        if unreachable:
            return manifest, None, unreachable
        try:
            result = await P.replay_manifest(manifest, _fresh_service(client), client=client)
        except P.ReplayUnavailableError as exc:
            return manifest, None, [{"catalog": "*", "error_type": str(exc)}]
    return manifest, result, []


def test_live_3c273_replay_reproduces_the_content_hash() -> None:
    manifest, result, unreachable = asyncio.run(_live_manifest_and_replay())
    if unreachable:
        pytest.skip(f"archives unreachable: {unreachable}")
    # Real astrophysical content of the manifest.
    science = manifest.science["catalogs"]
    assert [s["source_id"] for s in science["gaia_dr3"]["sources"]] == ["3700386905605055360"]
    assert "3C 273" in [s["source_id"] for s in science["simbad"]["sources"]]
    assert "3C 273" in [s["source_id"] for s in science["ned"]["sources"]]
    assert "2CXO J122906.6+020308" in [s["source_id"] for s in science["chandra"]["sources"]]
    assert science["exoplanet_archive"]["status"] == "empty"  # a quasar hosts no known planet
    # Every recorded request equals what the providers' builders produce.
    for name, req in manifest.catalogs.items():
        assert req.parameters, name
        if req.request_source == "recorded":
            assert req.request_verified is True, name
    assert P.verify_manifest(manifest)["content_hash_ok"] is True

    # A catalog that failed with a query error (not an outage) is a bug, not a skip.
    broken = {n: r.error_type for n, r in manifest.catalogs.items() if r.status == "failed"}
    assert not broken, f"catalogs failed in the original run: {broken}"
    assert result is not None
    failed_now = [n for n, e in result.diff["catalogs"].items() if (e.get("status") or {}).get("new") == "failed"]
    if failed_now and all(
        result.diff["catalogs"][n]["status"].get("new_error_type") in P.UNREACHABLE_ERRORS for n in failed_now
    ):
        pytest.skip(f"archives unreachable during replay: {failed_now}")
    # Nothing may pass as "reproduced" without having been compared.
    assert result.not_compared == [], result.not_compared
    assert result.diff["requests"] == {}, result.diff["requests"]
    if not result.identical:
        # Only continuously updated databases may legitimately change within minutes.
        changed = set(result.diff["catalogs"])
        living = {n for n in changed if manifest.catalogs[n].living_database}
        assert changed == living, f"fixed data releases changed between runs: {changed - living}; {result.explanation}"
        assert result.explanation and any("continuously updated" in line for line in result.explanation)
    else:
        assert result.new_content_hash == manifest.content_hash
        assert result.explanation[0].startswith("Content hash identical")


def test_live_every_reference_exists_with_the_recorded_metadata() -> None:
    async def run() -> list[P.ReferenceCheck]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await P.verify_references(client, P.REFERENCES.values())

    checks = asyncio.run(run())
    undecided = [c.key for c in checks if c.exists is None]
    if undecided and len(undecided) == len(checks):
        pytest.skip(f"ADS / doi.org unreachable: {checks[0].problems}")
    failures = {c.key: c.problems for c in checks if c.exists is not None and not c.ok}
    assert not failures, failures
    assert not undecided, f"undecided (service errors): {undecided}"
    by_key = {c.key: c for c in checks}
    # Spot checks of what ADS returns for the headline catalog papers.
    assert by_key["2023A&A...674A...1G"].doi_from_ads == "10.1051/0004-6361/202243940"
    assert by_key["2006AJ....131.1163S"].metadata["volume"] == "131"
    assert by_key["2013PASP..125..989A"].metadata_matches is True


async def _live_name_search_and_replay(name: str):
    os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "2"
    P._RELEASE_CACHE.clear()
    record = await main.search_object(name, radius_arcsec=5.0)
    manifest = P.build_manifest(record, registry=CatalogRegistry())
    unreachable = [f for f in record.failures if f.get("error_type") in P.UNREACHABLE_ERRORS]
    if unreachable:
        return record, manifest, None, unreachable
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        try:
            result = await P.replay_manifest(json.loads(manifest.to_json()), _fresh_service(client), client=client)
        except P.ReplayUnavailableError as exc:
            return record, manifest, None, [{"catalog": "*", "error_type": str(exc)}]
    return record, manifest, result, []


def _check_name_search_replay(manifest: P.ProvenanceManifest, result: P.ReplayResult) -> None:
    for catalog, req in manifest.catalogs.items():
        assert req.parameters, catalog
        if req.request_source == "recorded":
            assert req.request_verified is True, f"{catalog}: the manifest does not describe the request sent"
    failed_now = [n for n, e in result.diff["catalogs"].items() if (e.get("status") or {}).get("new") == "failed"]
    if failed_now and all(result.diff["catalogs"][n]["status"].get("new_error_type") in P.UNREACHABLE_ERRORS
                          for n in failed_now):
        pytest.skip(f"archives unreachable during replay: {failed_now}")
    assert result.diff["requests"] == {}, result.diff["requests"]
    assert result.diff["other"] == {}, result.diff["other"]  # same proper-motion origin and radius
    assert result.new_manifest.science["target_proper_motion"]["source"] == "resolver"
    if not result.identical:
        changed = set(result.diff["catalogs"])
        living = {n for n in changed if manifest.catalogs[n].living_database}
        assert changed == living, f"fixed data releases changed between runs: {changed - living}; {result.explanation}"


def test_live_barnards_star_name_search_replays_with_the_resolver_proper_motion() -> None:
    record, manifest, result, unreachable = asyncio.run(_live_name_search_and_replay("Barnard's star"))
    if unreachable:
        pytest.skip(f"archives unreachable: {unreachable}")
    pm = record.provenance["target_proper_motion"]
    assert pm["source"] == "resolver"
    assert pm["pm_dec_masyr"] == pytest.approx(10362.4, abs=1.0)  # SIMBAD / Gaia DR3: +10.36"/yr in Dec
    assert manifest.query["pm_source"] == "resolver"
    assert manifest.input_target["pm_dec_masyr"] == pytest.approx(10362.4, abs=1.0)
    # Gaia DR3 4472832130942575872 is Barnard's star, found by following it to J2016.
    nearest = min((m for m in record.provenance["matches"] if m["catalog"] == "gaia_dr3"),
                  key=lambda m: m["separation_arcsec"])
    assert nearest["source_id"] == "4472832130942575872" and nearest["separation_arcsec"] < 0.5
    assert result is not None
    _check_name_search_replay(manifest, result)


def test_live_m87_name_search_replays_with_the_resolver_zero_proper_motion() -> None:
    record, manifest, result, unreachable = asyncio.run(_live_name_search_and_replay("M87"))
    if unreachable:
        pytest.skip(f"archives unreachable: {unreachable}")
    pm = record.provenance["target_proper_motion"]
    assert pm["source"] == "resolver" and (pm["pm_ra_masyr"], pm["pm_dec_masyr"]) == (0.0, 0.0)  # a galaxy
    assert "M 87" in [s.source_id for s in record.catalog_results["simbad"]["sources"]]
    assert manifest.input_target["pm_ra_masyr"] == 0.0 and manifest.query["pm_source"] == "resolver"
    assert result is not None
    _check_name_search_replay(manifest, result)


def test_live_heasarc_release_descriptions() -> None:
    async def run() -> dict[str, dict[str, str]]:
        P._RELEASE_CACHE.clear()
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await P.live_releases(client, CatalogRegistry(), ["xmm", "chandra", "first", "nvss", "gaia_dr3"])

    releases = asyncio.run(run())
    if all(r.get("source", "").startswith("unavailable") for r in releases.values()):
        pytest.skip(f"HEASARC TAP unreachable: {releases}")
    assert set(releases) == {"xmm", "chandra", "first", "nvss"}  # gaia_dr3 is not a HEASARC table
    assert "XMM-Newton Serendipitous Source Catalog" in releases["xmm"]["release"]
    assert "DR" in releases["xmm"]["release"]
    assert releases["chandra"]["release"].startswith("Chandra Source Catalog, v2")
    assert "FIRST" in releases["first"]["release"]
    assert all(r["source"].startswith("live: https://heasarc.gsfc.nasa.gov/") for r in releases.values())


def test_live_registered_gaia_dr2_citation_resolves_to_its_paper() -> None:
    """A table registered with `vizier add` (I/345/gaia2) is cited with its own paper, resolved through ADS and
    doi.org (final review: its BibTeX held only the VizieR entries)."""
    import vizier

    citation = ("Gaia Collaboration 2018, A&A 616, A1 (2018A&A...616A...1G); VizieR I/345 (I/345/gaia2), "
                "DOI 10.26093/cds/vizier.1345")
    refs, notes = asyncio.run(vizier.citation_references(citation))
    if notes and all(("not resolved" in n) for n in notes):
        pytest.skip(f"ADS/doi.org unreachable: {notes}")
    assert notes == [], notes
    [ref] = refs
    assert ref["bibcode"] == "2018A&A...616A...1G" and ref["doi"] == "10.1051/0004-6361/201833051"
    assert ref["verified"] is True and ref["entry_type"] == "article" and "Gaia Data Release 2" in ref["title"]
    stored = P.reference_from_registry(ref)
    assert stored is not None
    P.parse_bibtex(P.reference_to_bibtex(stored))
