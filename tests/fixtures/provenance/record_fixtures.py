"""Record real NASA ADS link-gateway and doi.org responses for the offline provenance tests.

    .venv/Scripts/python.exe tests/fixtures/provenance/record_fixtures.py            # every case
    .venv/Scripts/python.exe tests/fixtures/provenance/record_fixtures.py nonexistent heasarc_tables

Each checked reference is stored as ``<name>.json`` (every HTTP exchange made by
:func:`provenance.verify_reference`, including redirect hops) plus the raw bodies
``<name>.<n>.body``. Polite: a handful of requests, one reference at a time.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance import REFERENCES, Reference, verify_reference

# name -> reference to check (the last one is a deliberately non-existent bibcode).
CASES: dict[str, Reference] = {
    "gaia_dr3": REFERENCES["2023A&A...674A...1G"],        # ADS -> DOI, CSL metadata
    "first_becker1995": REFERENCES["1995ApJ...450..559B"],  # PUB_HTML 404, EPRINT_HTML redirect
    "allwise_expsup": REFERENCES["2013wise.rept....1C"],   # redirect to a non-DOI page
    "ned_dataset_doi": REFERENCES["NED_DOI_2019"],          # dataset DOI (DataCite)
    # A real AAS meeting abstract (APASS DR9, cited through SIMBAD) without any full-text
    # link: existence is proven by the ASSOCIATED/DATA link types.
    "apass_no_fulltext": Reference(key="2015AAS...22533616H", entry_type="misc", authors=("Henden, A. A.",),
                                   title="APASS - The Latest Data Release", year=2015, bibcode="2015AAS...22533616H"),
    "nonexistent": Reference(key="2099XXX.....1....1Z", entry_type="article", authors=("Nobody, N.",),
                             title="Does not exist", year=2099, bibcode="2099XXX.....1....1Z", journal="None",
                             volume="1", pages="1"),
}


_IPV4 = re.compile(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HEASARC_TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
M87_RADIUS_ARCSEC = 10.0


async def record_m87_resolver() -> None:
    """M87 as a name search: live Sesame answer + every enabled catalog queried for the
    resolved target (J2000 position, pm = (0, 0) because M87 is extragalactic), stored
    in the fixture_io layout under m87_resolver/ (one <catalog>.json per catalog)."""
    import os

    from crossmatch import QueryExecutor
    from models import CatalogRegistry, QueryPlan, resolved_target
    from providers import CacheManager, SesameResolver, provider_map

    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    folder = HERE / "m87_resolver"
    folder.mkdir(exist_ok=True)
    registry = CatalogRegistry()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        sesame = SesameResolver(client)
        response = await client.get(f"{sesame.endpoint}?M87", headers={"Accept": "application/xml, text/xml"})
        response.raise_for_status()
        (folder / "sesame.xml").write_bytes(response.content)
        target = resolved_target(SesameResolver.parse_response("M87", response.text))
    print("m87 resolver target", target)
    for name in registry.enabled_catalogs():
        catalog = registry.get(name)
        log: list[tuple[httpx.Request, httpx.Response]] = []

        async def hook(resp: httpx.Response, log: list = log) -> None:
            await resp.aread()
            log.append((resp.request, resp))

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
            executor = QueryExecutor(provider_map(client, timeout=120.0, cache=CacheManager(None)), timeout=150.0,
                                     registry=registry)
            plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, M87_RADIUS_ARCSEC, catalog.wavelength)
            successes, failures = await executor.execute([plan], target)
        for old in folder.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, (request, resp) in enumerate(log):
            (folder / f"{name}.{idx}.body").write_bytes(_IPV4.sub(b"203.0.113.1", resp.content))  # as fixture_io.redact
            exchanges.append({"method": request.method, "url": str(request.url),
                              "request_body": request.content.decode("utf-8", "replace") if request.content else "",
                              "status_code": resp.status_code, "content_type": resp.headers.get("content-type", ""),
                              "match": []})
        meta = {"catalog": name, "target": "M87 (Sesame)", "ra": target.ra, "dec": target.dec, "epoch": target.epoch,
                "pm_ra_masyr": target.pm_ra_masyr, "pm_dec_masyr": target.pm_dec_masyr,
                "radius_arcsec": M87_RADIUS_ARCSEC, "exchanges": exchanges}
        (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        outcome = f"FAILED {failures[0].error_type}" if failures else f"{len(successes[0][1])} rows"
        print(f"m87_resolver {name:<20} {outcome}")
        await asyncio.sleep(0.3)


async def record_heasarc_tables() -> None:
    """HEASARC TAP_SCHEMA.tables descriptions of the tables in the catalog registry."""
    from models import CatalogRegistry
    from provenance import _RELEASE_CACHE, _heasarc_tables, live_releases

    registry = CatalogRegistry()
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    _RELEASE_CACHE.clear()
    async with httpx.AsyncClient(timeout=60.0, event_hooks={"response": [hook]}) as client:
        releases = await live_releases(client, registry, list(_heasarc_tables(registry, registry.catalogs)))
    (request, response), = log
    (HERE / "heasarc_tables.0.body").write_bytes(response.content)
    meta = {"reference": "HEASARC TAP_SCHEMA.tables", "recorded_releases": releases, "exchanges": [{
        "method": request.method, "url": str(request.url), "request_body": request.content.decode("utf-8"),
        "status_code": response.status_code,
        "headers": {k: v for k, v in response.headers.items() if k.lower() == "content-type"},
    }]}
    (HERE / "heasarc_tables.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("heasarc_tables", releases)


async def record(names: list[str] | None = None) -> None:
    if not names or "heasarc_tables" in names:
        await record_heasarc_tables()
    if not names or "m87_resolver" in names:
        await record_m87_resolver()
    for name, ref in CASES.items():
        if names and name not in names:
            continue
        log: list[tuple[httpx.Request, httpx.Response]] = []

        async def hook(response: httpx.Response, log: list = log) -> None:
            await response.aread()
            log.append((response.request, response))

        async with httpx.AsyncClient(timeout=60.0, event_hooks={"response": [hook]}) as client:
            check = await verify_reference(client, ref)
        for old in HERE.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, (request, response) in enumerate(log):
            (HERE / f"{name}.{idx}.body").write_bytes(response.content)
            exchanges.append({
                "method": request.method,
                "url": str(request.url),
                "accept": request.headers.get("accept", ""),
                "status_code": response.status_code,
                "headers": {k: v for k, v in response.headers.items() if k.lower() in {"location", "content-type"}},
            })
        meta = {"reference": ref.key, "recorded_check": check.as_dict(), "exchanges": exchanges}
        (HERE / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name:<18} ok={check.ok} exists={check.exists} link={check.link_type} problems={check.problems}")
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(record(sys.argv[1:] or None))
