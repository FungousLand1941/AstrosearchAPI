"""Record real Sesame / SIMBAD TAP exchanges used by the offline ai.py tests.

    .venv/Scripts/python.exe tests/fixtures/ai/record_ai_fixtures.py

Each set is stored in the tests/fixture_io.py layout (``<set>/traffic.json`` plus
``traffic.<n>.body``) so ``load_exchanges("ai/<set>")`` and ``replay_side_effect``
replay it strictly (a request that differs from the recording fails the test).
Polite: a handful of small queries per set, run sequentially per set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"

from fixture_io import redact

import ai
from models import (
    CatalogRegistry,
    ObjectResolutionError,
    ResolvedObject,
    resolved_target,
)
from providers import SesameResolver

M87 = (187.70593077, 12.39112325)  # SIMBAD ICRS J2000 position of M 87 (via Sesame)

GOOD_ADQL = (
    "SELECT TOP 5 b.main_id AS main_id, b.plx_value AS plx_value, b.sp_type AS sp_type FROM basic AS b "
    "WHERE b.plx_value > 500 ORDER BY plx_value DESC"
)
BAD_ADQL = "SELECT TOP 5 b.main_id, b.no_such_column FROM basic AS b WHERE b.plx_value > 500"


async def _record(name: str, work: Callable[[httpx.AsyncClient], Awaitable[Any]]) -> None:
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        outcome = await work(client)
    folder = HERE / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("traffic.*.body"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (folder / f"traffic.{idx}.body").write_bytes(redact(response.content))
        exchanges.append({
            "method": request.method,
            "url": str(request.url),
            "request_body": request.content.decode("utf-8", "replace") if request.content else "",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "match": [],
        })
    (folder / "traffic.json").write_text(json.dumps({"set": name, "exchanges": exchanges}, indent=2), encoding="utf-8")
    print(f"{name:<22} {len(exchanges)} exchange(s)  {outcome}")


async def explain_3c273(client: httpx.AsyncClient) -> str:
    facts = await ai.gather_facts(http_client=client, name="3C 273", include_crossmatch=False, max_references=8)
    return f"main_id={facts.identity.get('main_id')} sources={len(facts.sources)}"


async def explain_m87_coords(client: httpx.AsyncClient) -> str:
    facts = await ai.gather_facts(http_client=client, ra=M87[0], dec=M87[1], include_crossmatch=False, max_references=3)
    return f"main_id={facts.identity.get('main_id')}"


async def sesame_names(client: httpx.AsyncClient) -> str:
    resolver = SesameResolver(client)
    out = []
    for name in ("M87", "3C 273", "Qzxv Nonexistent Object 999"):
        try:
            out.append((await resolver.resolve(name)).canonical_name)
        except ObjectResolutionError as exc:
            out.append(f"error: {exc}")
    ai._OTYPE_TREES.clear()
    tree = await ai.simbad_otype_tree(client)  # object-type vocabulary for compiled type filters
    out.append(f"otypedef: {len(tree)} types")
    return "; ".join(str(o) for o in out)


async def adql_checks(client: httpx.AsyncClient) -> str:
    simbad = CatalogRegistry().get("simbad")
    good = await ai.verify_adql_remote(simbad, GOOD_ADQL, client)
    bad = await ai.verify_adql_remote(simbad, BAD_ADQL, client)
    compiled = ai.CompiledQuery(
        text="nearest stars", scope="all_sky", advanced_query=None, plan=["run"], explanation="x", adql=GOOD_ADQL,
        adql_catalog="simbad", adql_endpoint=simbad.endpoint, resolved_objects=[], attempts=1, validation_history=[],
        model="recording",
    )
    rows = await ai.execute_compiled_query(compiled, http_client=client, registry=CatalogRegistry(), row_limit=5)
    return f"good={good!r} bad={bad!r} rows={[r['main_id'] for r in rows['rows']]}"


# Objects whose SIMBAD data exercise the physical guards of the derived quantities.
BARNARD_GAIA2016 = (269.44850252543836, 4.739420051112412)  # Gaia DR3 position of Barnard's star at J2016.0
BLANK_SKY = (150.1234, -30.4567)


def _facts_set(**kwargs: Any) -> Callable[[httpx.AsyncClient], Awaitable[str]]:
    async def work(client: httpx.AsyncClient) -> str:
        facts = await ai.gather_facts(http_client=client, include_crossmatch=False, max_references=2, **kwargs)
        return f"main_id={facts.identity.get('main_id')} derived={[d['quantity'] for d in facts.derived]}"

    return work


# CDS Sesame restricted to NED (option N): the answer carries no SIMBAD oid, as when the
# default SNV query is answered from Sesame's NED cache. NED's "M 1" position is the
# Crab Pulsar's (05:34:31.94 +22:00:52.1), 10.7 arcsec from SIMBAD's nebula centre.
SESAME_NED_ONLY = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/N"
NED_M1 = (83.633107, 22.014486)
UNKNOWN_NED_NAME = "Qzxv NED-only Object 1"


def ned_only_resolver(client: httpx.AsyncClient) -> SesameResolver:
    return SesameResolver(client, endpoint=SESAME_NED_ONLY)


class UnknownNameNedResolver:
    """A NED-only answer for a name SIMBAD does not list (NED's M 1 position)."""

    async def resolve(self, name: str) -> ResolvedObject:
        return ResolvedObject(query=name, canonical_name=name, ra_deg=NED_M1[0], dec_deg=NED_M1[1], aliases=[],
                              object_type="SNR", redshift=None, pm_ra_masyr=None, pm_dec_masyr=None, epoch=None,
                              resolver="cds_sesame", resolver_metadata={"resolver_name": "N=NED", "raw_fields": {}})


async def explain_m1_ned(client: httpx.AsyncClient) -> str:
    facts = await ai.gather_facts(http_client=client, name="M 1", include_crossmatch=False, max_references=2,
                                  resolver=ned_only_resolver(client))
    return f"main_id={facts.identity.get('main_id')} otype={facts.identity.get('otype')}"


async def explain_ned_unknown_name(client: httpx.AsyncClient) -> str:
    facts = await ai.gather_facts(http_client=client, name=UNKNOWN_NED_NAME, include_crossmatch=False,
                                  max_references=2, resolver=UnknownNameNedResolver())  # type: ignore[arg-type]
    return f"main_id={facts.identity.get('main_id')} warnings={facts.warnings}"


SETS: dict[str, Callable[[httpx.AsyncClient], Awaitable[Any]]] = {
    "explain_3c273": explain_3c273,
    "explain_m87_coords": explain_m87_coords,
    "sesame": sesame_names,
    "adql": adql_checks,
    # Seyfert 1 galaxy with a spurious Gaia parallax (2.65 +- 0.47 mas) and a measured velocity.
    "explain_ngc4639": _facts_set(name="NGC 4639"),
    # Seyfert 1 with spectroscopic z = 0.0139 and a 6-sigma spurious parallax.
    "explain_eso140_43": _facts_set(name="ESO 140-43"),
    # Galactic double white dwarf (otype XB*) with a photometric, quality-E "redshift" of 2.19.
    "explain_hmcnc": _facts_set(name="HM Cnc"),
    # Barnard's star searched at its Gaia DR3 J2016.0 position (moved ~166 arcsec since J2000).
    "explain_barnard_2016": _facts_set(ra=BARNARD_GAIA2016[0], dec=BARNARD_GAIA2016[1], epoch=2016.0),
    # A high-latitude position with no SIMBAD object within 5 arcsec.
    "explain_blank_sky": _facts_set(ra=BLANK_SKY[0], dec=BLANK_SKY[1]),
    # "M 1" answered by NED only (no SIMBAD oid): identified through SIMBAD's ident table.
    "explain_m1_ned": explain_m1_ned,
    # A NED-only name SIMBAD does not list: nearest-object fallback with an identification warning.
    "explain_ned_unknown_name": explain_ned_unknown_name,
    # GW170817 host (Sy2, z_cmb ~ 0.0108; SBF distance 40.7 +- 1.4 Mpc): peculiar velocity dominates the error.
    "explain_ngc4993": _facts_set(name="NGC 4993"),
    # Main SIMBAD type AG? (AGN candidate, path G > AGN): still a galaxy for cosmology.
    "explain_ngc6621": _facts_set(name="NGC 6621"),
    # Galaxy in a pair (GiP, path G > GiP).
    "explain_ngc4807": _facts_set(name="NGC 4807"),
    # Nearest star: focused bibliography must name it in title/abstract; Gaia DR3 parallax.
    "explain_proxima": _facts_set(name="Proxima Centauri"),
    # Galactic objects whose SIMBAD otypes include a stray confirmed "G": parallax distances stay.
    "explain_helix": _facts_set(name="NGC 7293"),  # PN, Gaia plx 5.01 +- 0.04 mas
    "explain_mira": _facts_set(name="Mira"),  # Mi*, plx 10.91 +- 1.22 mas
    "explain_47tuc": _facts_set(name="47 Tuc"),  # GlC, Gaia-EDR3-based plx 0.232 +- 0.009 mas
    # Blazar with z = 2.2 stated to one decimal and no error (rvz_redshift_prec 1).
    "explain_ton618": _facts_set(name="Ton 618"),
    # LINER galaxy whose redshift is stored as a velocity (rvz_type v, 9581 +- 5 km/s).
    "explain_ngc6086": _facts_set(name="NGC 6086"),
    # Cluster parallax 7.364 +- 0.005 mas from the Gaia DR2 HRD paper (2018A&A...616A..10G).
    "explain_pleiades": _facts_set(name="Pleiades"),
    # HXB with a stray candidate "BL?"; Gaia DR3 parallax whose zero point exceeds the formal error.
    "explain_cygx1": _facts_set(name="Cyg X-1"),
    # Hipparcos parallax with parallax/error ~ 8: clearly asymmetric distance errors.
    "explain_betelgeuse": _facts_set(name="Betelgeuse"),
}

# Real crossmatch records (UnifiedRecord.as_dict(), trimmed to what summarize_crossmatch
# reads) for positions where footprints and chance coincidences matter.
CROSSMATCH_RECORDS: dict[str, str] = {
    "centaurus_a": "Centaurus A",  # Dec -43: outside NVSS/VLASS/Pan-STARRS footprints
    "ngc4151": "NGC 4151",  # ROSAT 2RXS counterpart ~6 arcsec from the nucleus
    "sgr_a_star": "Sgr A*",  # Galactic Centre: dense 2MASS / Pan-STARRS fields
    "gn_z11": "GN-z11",  # z ~ 10.6 galaxy: lone unrelated field rows 25-27 arcsec away
    "vega": "Vega",  # too bright for Gaia DR3: the only Gaia row is an unrelated star ~25 arcsec away
}


def _trim_record(record: dict[str, Any]) -> dict[str, Any]:
    keep_meta = ("wavelength", "citation")
    counterparts = {
        wave: [
            {
                "catalog": src["catalog"], "source_id": src["source_id"], "separation_arcsec": src["separation_arcsec"],
                "positional_error_arcsec": src.get("positional_error_arcsec"),
                "metadata": {k: (src.get("metadata") or {}).get(k) for k in keep_meta},
                "data": src.get("data") or {},
            }
            for src in sources
        ]
        for wave, sources in (record.get("counterparts") or {}).items()
    }
    results = {
        name: {k: v for k, v in info.items() if k not in {"sources", "pad_sources", "query", "warnings"}}
        for name, info in (record.get("catalog_results") or {}).items()
    }
    provenance = record.get("provenance") or {}
    return {
        "target": record.get("target"),
        "catalogs_queried": record.get("catalogs_queried"),
        "catalog_results": results,
        "counterparts": counterparts,
        "provenance": {"query_radius_arcsec": provenance.get("query_radius_arcsec")},
    }


async def record_crossmatch(key: str, name: str) -> None:
    from main import build_service

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resolved = await SesameResolver(client).resolve(name)
        target = resolved_target(resolved)
        record = await build_service(client=client).crossmatch(
            target.ra, target.dec, radius_arcsec=ai.DEFAULT_CROSSMATCH_RADIUS_ARCSEC, epoch=target.epoch,
            pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr,
        )
    folder = HERE / "crossmatch_records"
    folder.mkdir(parents=True, exist_ok=True)
    data = _trim_record(record.as_dict())
    (folder / f"{key}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"crossmatch {key:<14} {len(data['catalog_results'])} catalogs")


async def main(names: list[str]) -> None:
    for name in names or list(SETS) + list(CROSSMATCH_RECORDS):
        if name in CROSSMATCH_RECORDS:
            await record_crossmatch(name, CROSSMATCH_RECORDS[name])
        else:
            await _record(name, SETS[name])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
