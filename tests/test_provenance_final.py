"""Regression tests for the final review of provenance.py (and the citation part of vizier.py).

* Citations/BibTeX of a user-registered VizieR catalog name the table's own paper (Gaia DR2 for I/345/gaia2),
  from the references ``vizier add`` resolved through ADS/doi.org and stored in the registry entry, or -- for
  an entry registered before references were stored -- parsed from its citation text (unverified) and resolved
  by ``cite --verify`` / ``/api/v1/citations?verify=true``. The VizieR acknowledgement appears once.
  ``astrosearch cite`` reads the registry ``vizier add`` writes (default ~/.astrosearch/catalogs.yaml).
* Search radii: /api/v1/search, its batch items, saved queries and search manifests share
  ``provenance.SearchFields``, which refuses a cone above API_MAX_RADIUS_ARCSEC (default 1800").

The registry entry is the one ``POST /api/v1/vizier/register {"table_id": "I/345/gaia2"}`` wrote live
(tests/fixtures/final/registered_i_345_gaia2.json).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import provenance as P
import vizier

FIXTURE = Path(__file__).parent / "fixtures" / "final" / "registered_i_345_gaia2.json"
GAIA_DR2 = "2018A&A...616A...1G"
NAME = "vizier_i_345_gaia2"


def _registered() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _gaia_dr2_reference() -> P.Reference:
    stored = _registered()["entry"]["parameters"][P.REGISTRY_REFERENCES_KEY][0]
    ref = P.reference_from_registry(stored)
    assert ref is not None
    return ref


@pytest.fixture()
def registry_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The user registry with the live-registered I/345/gaia2 entry (references stored)."""
    path = tmp_path / "catalogs.yaml"
    data = _registered()
    vizier.save_definition(data["name"], data["entry"], path=path)
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(path))
    return path


@pytest.fixture()
def legacy_registry_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The same entry registered before `vizier add` stored its references: citation text only."""
    path = tmp_path / "catalogs.yaml"
    data = _registered()
    entry = copy.deepcopy(data["entry"])
    entry["parameters"].pop(P.REGISTRY_REFERENCES_KEY)
    vizier.save_definition(data["name"], entry, path=path)
    monkeypatch.setenv("CATALOG_REGISTRY_PATH", str(path))
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    P.register_cli(parser.add_subparsers(dest="command"))
    return parser


def _app(registry) -> FastAPI:
    app = FastAPI()
    app.include_router(P.router)
    app.state.registry = registry
    return app


def _no_network(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - must not be called
    raise AssertionError("no ADS/doi.org request expected")


# ---------------------------------------------------------------------------
# Citations of a registered VizieR catalog
# ---------------------------------------------------------------------------


def test_registered_catalog_cites_its_own_paper(registry_file: Path) -> None:
    bundle = P.citations_for([NAME], P._deployment_registry(), strict=False)
    assert bundle.unknown == []
    refs = {ref.key: ref for ref in bundle.references}
    assert GAIA_DR2 in refs and refs[GAIA_DR2].verified
    assert refs[GAIA_DR2].doi == "10.1051/0004-6361/201833051"
    [ack] = [a for a in bundle.acknowledgements if a["key"] == NAME]
    assert ack["bibcodes"] == [GAIA_DR2] and ack["references_verified"] is True
    # The VizieR acknowledgement once (the curated 'vizier' entry), not also in the registry's wording.
    assert bundle.acknowledgement_text.count("VizieR catalogue access tool") == 1
    entries = {entry.key: entry for entry in P.parse_bibtex(bundle.bibtex)}
    assert entries[GAIA_DR2].entry_type == "article"
    assert entries[GAIA_DR2].fields["doi"] == "10.1051/0004-6361/201833051"


def test_cite_cli_reads_the_vizier_add_registry(registry_file: Path, tmp_path: Path, capsys, monkeypatch) -> None:
    """``astrosearch cite`` read models.CatalogRegistry(CATALOG_REGISTRY_PATH): with the default path
    (~/.astrosearch/catalogs.yaml, where `vizier add` writes) a registered table was unknown."""
    monkeypatch.setattr(P, "lookup_reference", _no_network)
    out = tmp_path / "cite.bib"
    args = _parser().parse_args(["cite", "--catalogs", NAME, "--bibtex", str(out)])
    assert args.handler(args) == 0
    text = out.read_text(encoding="utf-8")
    assert f"@article{{{GAIA_DR2}," in text and "@article{2000A&AS..143...23O," in text
    assert GAIA_DR2 in capsys.readouterr().out  # the References list names the paper

    # The default location (no CATALOG_REGISTRY_PATH): the vizier user registry in the home directory.
    home = tmp_path / "home"
    monkeypatch.delenv("CATALOG_REGISTRY_PATH")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    data = _registered()
    vizier.save_definition(data["name"], data["entry"])
    assert (home / ".astrosearch" / "catalogs.yaml").exists()
    assert NAME in P._deployment_registry().catalogs


def test_citations_route_bibtex_names_the_registered_paper(registry_file: Path) -> None:
    with TestClient(_app(P._deployment_registry())) as client:
        bib = client.get("/api/v1/citations", params={"catalogs": f"gaia_dr3,{NAME}", "format": "bibtex"})
        data = client.get("/api/v1/citations", params={"catalogs": NAME}).json()
    assert bib.status_code == 200 and f"@article{{{GAIA_DR2}," in bib.text
    assert "X-Unverified-References" not in bib.headers
    [ack] = [a for a in data["acknowledgements"] if a["key"] == NAME]
    assert ack["bibcodes"] == [GAIA_DR2] and ack["references"]
    assert data["unverified_references"] == []


def test_legacy_entry_is_cited_unverified_then_resolved_by_verify(legacy_registry_file: Path, tmp_path: Path,
                                                                 monkeypatch, capsys) -> None:
    registry = P._deployment_registry()
    bundle = P.citations_for([NAME], registry, strict=False)
    [paper] = [ref for ref in bundle.references if ref.bibcode == GAIA_DR2]
    assert not paper.verified and paper.entry_type == "misc"
    assert f"@misc{{{GAIA_DR2}," in bundle.bibtex  # cited even offline, marked unverified

    calls: list[str] = []

    async def lookup(_client: httpx.AsyncClient, bibcode: str) -> P.Reference | None:
        calls.append(bibcode)
        return _gaia_dr2_reference() if bibcode == GAIA_DR2 else None

    monkeypatch.setattr(P, "lookup_reference", lookup)
    with TestClient(_app(registry)) as client:
        plain = client.get("/api/v1/citations", params={"catalogs": NAME, "format": "bibtex"})
        verified = client.get("/api/v1/citations", params={"catalogs": NAME, "format": "bibtex", "verify": "true"})
    assert plain.headers["X-Unverified-References"] == GAIA_DR2
    assert verified.status_code == 200 and "X-Unverified-References" not in verified.headers
    assert f"@article{{{GAIA_DR2}," in verified.text and "Gaia Data Release 2" in verified.text

    checked: list[str] = []

    async def verify(_client: httpx.AsyncClient, references, **_kwargs) -> list[P.ReferenceCheck]:
        """--verify then checks every reference against ADS; offline, each is reported found."""
        checked.extend(ref.key for ref in references)
        return [P.ReferenceCheck(key=ref.key, bibcode=ref.bibcode, exists=True, doi_matches=True)
                for ref in references]

    monkeypatch.setattr(P, "verify_references", verify)
    out = tmp_path / "cite.bib"
    args = _parser().parse_args(["cite", "--catalogs", NAME, "--verify", "--bibtex", str(out)])
    code = args.handler(args)
    text = out.read_text(encoding="utf-8")
    assert f"@article{{{GAIA_DR2}," in text and f"@misc{{{GAIA_DR2}," not in text, (code, capsys.readouterr())
    assert GAIA_DR2 in checked and calls.count(GAIA_DR2) == 2


def test_vizier_add_stores_the_resolved_paper(monkeypatch) -> None:
    """``vizier add`` resolves the bibcodes of the registry citation through ADS/doi.org and stores them."""
    citation = _registered()["entry"]["citation"]

    async def lookup(_client: httpx.AsyncClient, bibcode: str) -> P.Reference | None:
        return _gaia_dr2_reference() if bibcode == GAIA_DR2 else None

    monkeypatch.setattr(P, "lookup_reference", lookup)

    async def run() -> tuple[list[dict[str, Any]], list[str]]:
        async with httpx.AsyncClient() as client:
            return await vizier.citation_references(citation, client=client)

    refs, notes = asyncio.run(run())
    assert notes == [] and [r["bibcode"] for r in refs] == [GAIA_DR2] and refs[0]["verified"] is True

    async def down(_client: httpx.AsyncClient, bibcode: str) -> P.Reference | None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(P, "lookup_reference", down)
    refs, notes = asyncio.run(run())
    assert refs == [] and len(notes) == 1 and GAIA_DR2 in notes[0] and "unverified" in notes[0]


@pytest.mark.parametrize("error", [RuntimeError("transport broke"), AssertionError("no recording for ADS"),
                                   httpx.ReadError("reset")])
def test_vizier_add_survives_an_unexpected_transport_error(error: Exception) -> None:
    """The paper lookup is optional: a transport raising something other than httpx.HTTPError (a
    respx side effect without a recording, a broken custom transport) leaves the paper unverified
    instead of failing ``vizier add``."""
    citation = _registered()["entry"]["citation"]

    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    async def run() -> tuple[list[dict[str, Any]], list[str]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await vizier.citation_references(citation, client=client)

    refs, notes = asyncio.run(run())
    assert refs == [] and len(notes) == 1 and GAIA_DR2 in notes[0] and "unverified" in notes[0]
    if not isinstance(error, httpx.HTTPError):  # an httpx error is already an unanswered lookup (provenance)
        assert type(error).__name__ in notes[0]


# ---------------------------------------------------------------------------
# Search radius limit (API_MAX_RADIUS_ARCSEC)
# ---------------------------------------------------------------------------


def test_search_fields_refuse_a_radius_above_the_configured_limit(monkeypatch) -> None:
    """A 100000" (28 degree) cone was sent to every archive by POST /api/v1/search (and manifests)."""
    import api

    for model in (api.SearchRequest, P.ManifestRequest, P.SearchFields):
        with pytest.raises(ValidationError, match="API_MAX_RADIUS_ARCSEC|less than or equal"):
            model.model_validate({"ra": 200.0, "dec": -30.0, "radius_arcsec": 100000.0})
        with pytest.raises(ValidationError, match="API_MAX_RADIUS_ARCSEC"):
            model.model_validate({"ra": 200.0, "dec": -30.0, "radius_arcsec": 1801.0})
        assert model.model_validate({"ra": 200.0, "dec": -30.0, "radius_arcsec": 1800.0}).radius_arcsec == 1800.0
    monkeypatch.setenv("API_MAX_RADIUS_ARCSEC", "60")
    with pytest.raises(ValidationError, match="60 arcsec"):
        api.SearchRequest.model_validate({"ra": 200.0, "dec": -30.0, "radius_arcsec": 61.0})
    with pytest.raises(ValueError, match="API_MAX_RADIUS_ARCSEC"):
        asyncio.run(P.run_basic_search(object(), ra=1.0, dec=1.0, radius_arcsec=61.0))  # before any request


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_run_basic_search_treats_a_blank_name_as_no_name(blank: str) -> None:
    """Regression: run_basic_search passed the raw name on after the shared check, so a
    whitespace name beside ra/dec went to the resolver and the coordinates were ignored
    ('manifest --name "   " --ra 10 --dec 11' exited 2 with 'Object name must not be empty')."""

    class Service:
        calls: list[tuple[float, float]] = []

        async def crossmatch(self, ra, dec, **_kwargs):
            self.calls.append((ra, dec))
            return "record"

    class NoResolver:
        async def resolve(self, name):  # pragma: no cover - reaching it is the bug
            raise AssertionError(f"resolver called for blank name {name!r}")

    service = Service()
    result = asyncio.run(P.run_basic_search(service, name=blank, ra=10.0, dec=11.0, radius_arcsec=5.0,
                                            resolver=NoResolver()))
    assert result == "record" and service.calls == [(10.0, 11.0)]


def test_search_radius_checks_share_one_implementation(monkeypatch) -> None:
    """main.check_search_radius (search, stream, saved queries) and provenance.check_radius_limit
    (manifests, replays) are one check against models.Settings().max_radius_arcsec: same limit, same message."""
    import main
    import models

    assert main.check_search_radius is models.check_search_radius
    monkeypatch.setenv("API_MAX_RADIUS_ARCSEC", "2400")
    messages = []
    for check in (main.check_search_radius, P.check_radius_limit):
        check(None)
        check(2400.0)
        with pytest.raises(ValueError, match="API_MAX_RADIUS_ARCSEC") as raised:
            check(2401.0)
        messages.append(str(raised.value))
    assert messages[0] == messages[1] == ("radius_arcsec 2401 exceeds the largest search radius, 2400 arcsec "
                                          "(API_MAX_RADIUS_ARCSEC)")


class _UnusedService:
    """A replay refused before any archive is queried never reaches the service."""

    @property
    def registry(self) -> Any:  # pragma: no cover - must not be called
        raise AssertionError("the replay must be refused before the registry or any archive is used")


@pytest.mark.parametrize(("where", "radius", "limit"), [
    ("radius_arcsec", 7200.0, None),  # above the static 3600" ceiling
    ("radius_arcsec", 1801.0, None),  # above the default API_MAX_RADIUS_ARCSEC (1800")
    ("radius_arcsec", 61.0, 60.0),  # above the deployment's configured limit
    ("advanced_query", 7200.0, None),
    ("advanced_query", 1801.0, None),
])
def test_replay_refuses_a_manifest_radius_above_the_limit(where: str, radius: float, limit: float | None) -> None:
    """Finding: a manifest edited to radius_arcsec 7200 was replayed (200) with a 2-degree cone
    to every archive. The replay now checks the 3600" ceiling and API_MAX_RADIUS_ARCSEC (422)."""
    from test_provenance import synthetic_record

    from models import Settings

    data = P.build_manifest(synthetic_record()).as_dict()
    if where == "radius_arcsec":
        data["radius_arcsec"] = radius
    else:
        data["query"]["mode"] = "advanced"
        data["query"]["advanced_query"] = {"target": {"ra": 150.0, "dec": 2.0}, "radius_arcsec": radius}
    app = _app(None)
    app.state.service = _UnusedService()
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(_no_network))
    if limit is not None:
        app.state.settings = Settings(API_MAX_RADIUS_ARCSEC=limit)
    with TestClient(app) as client:
        response = client.post("/api/v1/provenance/replay", json={"manifest": data})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "radius_arcsec" in detail and ("API_MAX_RADIUS_ARCSEC" in detail or "3600" in detail), detail
    with pytest.raises(P.ManifestError, match="radius_arcsec"):
        asyncio.run(P.replay_manifest(data, _UnusedService(), client=object(),
                                      settings=Settings(API_MAX_RADIUS_ARCSEC=limit or 1800.0)))


def test_manifest_route_refuses_a_wide_cone_before_searching() -> None:
    with TestClient(_app(None)) as client:
        response = client.post("/api/v1/provenance/manifest",
                               json={"ra": 200.0, "dec": -30.0, "radius_arcsec": 100000.0})
    assert response.status_code == 422, response.text
