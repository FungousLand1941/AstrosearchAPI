"""Regression test for the final review of vizier.py: ``vizier add`` stores the table's own paper.

A registered table (2SXPS, IX/58/2sxps, from the recorded VizieR metadata) is registered with the ADS/doi.org
lookup answering offline; the entry keeps the resolved reference, and ``provenance.citations_for`` cites it
as a verified BibTeX entry (before, a registered table's BibTeX held only the VizieR entries).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_vizier import _client, replaying

import provenance as P
import vizier

SXPS = "2020ApJS..247...54E"


def test_register_table_stores_and_cites_the_resolved_paper(tmp_path: Path, monkeypatch) -> None:
    asked: list[str] = []

    async def lookup(_client: httpx.AsyncClient, bibcode: str) -> P.Reference | None:
        asked.append(bibcode)
        return P.Reference(key=bibcode, entry_type="article", authors=("Evans, P. A.",), more_authors=True,
                           title="2SXPS: An Improved and Expanded Swift X-Ray Telescope Point-source Catalog",
                           year=2020, bibcode=bibcode, journal="The Astrophysical Journal Supplement Series",
                           volume="247", pages="54", doi="10.3847/1538-4365/ab7db9")

    monkeypatch.setattr(P, "lookup_reference", lookup)

    async def register() -> vizier.Registration:
        with replaying("describe_2sxps"):
            async with _client() as client:
                return await vizier.register_table("IX/58/2sxps", path=tmp_path / "c.yaml", client=client)

    registration = asyncio.run(register())
    assert SXPS in (registration.entry.get("citation") or "") and asked == [SXPS]
    [stored] = registration.entry["parameters"][P.REGISTRY_REFERENCES_KEY]
    assert stored["bibcode"] == SXPS and stored["verified"] is True

    registry = vizier.load_registry(tmp_path / "c.yaml")
    bundle = P.citations_for([registration.name], registry, strict=False)
    [paper] = [ref for ref in bundle.references if ref.bibcode == SXPS]
    assert paper.verified and paper.doi == "10.3847/1538-4365/ab7db9"
    assert f"@article{{{SXPS}," in bundle.bibtex
    assert bundle.acknowledgement_text.count("VizieR catalogue access tool") == 1
