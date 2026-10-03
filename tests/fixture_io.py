"""Record real archive responses to tests/fixtures and replay them offline via respx.

Recording (live network, polite: one cone query per catalog per target):

    .venv/Scripts/python.exe tests/fixture_io.py record            # all targets
    .venv/Scripts/python.exe tests/fixture_io.py record 3c273      # one target
    .venv/Scripts/python.exe tests/fixture_io.py record proxima_j2000_pm xmm   # one catalog
    .venv/Scripts/python.exe tests/fixture_io.py record fallbacks  # Gator with IRSA TAP down

Each exchange is stored as ``fixtures/<target>/<catalog>.json`` (request method,
URL, form body, status, content type, match token) plus the raw response bytes in
``fixtures/<target>/<catalog>.<n>.body``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Canary targets (ICRS degrees). Positions from SIMBAD (J2000) except HD 209458,
# whose position is the Exoplanet Archive's J2015.5 value (pm ~ 30 mas/yr).
TARGETS: dict[str, tuple[float, float]] = {
    "3c273": (187.2779154, 2.0523883),
    "m87": (187.7059308, 12.3911233),
    "hd209458": (330.79502, 18.88432),
    "hd106785": (184.20402414769, 12.53916028867),
}
RECORD_RADIUS_ARCSEC = 10.0

# High proper-motion targets recorded WITH a target epoch, so the cones are
# epoch-widened (proper motion not given) or follow the target (proper motion given).
# Barnard's star moves 10.39"/yr, Proxima Cen 3.86"/yr, GJ 1151 1.82"/yr.
BARNARD_PM = (-801.551, 10362.394)  # SIMBAD / Gaia DR3, mas/yr
EPOCH_TARGETS: dict[str, dict[str, Any]] = {
    # SIMBAD position (ICRS, epoch J2000), proper motion unknown (adopted from Gaia).
    "barnard_j2000": {"ra": 269.45207696, "dec": 4.69336497, "epoch": 2000.0},
    # The same position with the SIMBAD proper motion given (e.g. from name resolution).
    "barnard_j2000_pm": {"ra": 269.45207696, "dec": 4.69336497, "epoch": 2000.0,
                         "pm_ra_masyr": BARNARD_PM[0], "pm_dec_masyr": BARNARD_PM[1]},
    # Gaia DR3 4472832130942575872 at its reference epoch J2016.0.
    "barnard_gaia2016": {"ra": 269.44850252543836, "dec": 4.739420051112412, "epoch": 2016.0},
    # SIMBAD J2000 positions and proper motions (as Sesame supplies them for a name search).
    "proxima_j2000_pm": {"ra": 217.42894222, "dec": -62.67949019, "epoch": 2000.0,
                         "pm_ra_masyr": -3781.741, "pm_dec_masyr": 769.465, "radius_arcsec": 5.0},
    "gj1151_j2000_pm": {"ra": 177.74050219, "dec": 48.37737811, "epoch": 2000.0,
                        "pm_ra_masyr": -1545.069, "pm_dec_masyr": -962.724, "radius_arcsec": 5.0},
    # Not an epoch target: a VLASS source with a 'Redundant' (Flag 2) duplicate row.
    "vlass_duplicate": {"ra": 43.8634477, "dec": 5.9886937, "radius_arcsec": 5.0},
}
# Every enabled catalog is recorded for the Barnard targets (so each catalog's epoch
# handling is exercised with and without a known proper motion).
BARNARD_CATALOGS = [
    "gaia_dr3", "simbad", "ned", "exoplanet_archive", "twomass_psc", "allwise", "panstarrs_dr2", "sdss",
    "first", "nvss", "vlass", "lotss", "rosat", "chandra", "xmm",
]
EPOCH_TARGET_CATALOGS = BARNARD_CATALOGS
EPOCH_TARGET_CATALOG_SETS: dict[str, list[str]] = {
    "barnard_j2000": BARNARD_CATALOGS,
    "barnard_j2000_pm": BARNARD_CATALOGS,
    "barnard_gaia2016": BARNARD_CATALOGS,
    # CSC: 2CXO J142942.7-624046 is Proxima at ~2000.35. 5XMM: three unique sources are
    # Proxima (2001.8, 2009.2, 2017.1) in a stack spanning 2001.61-2018.19. 2MASS
    # 14294291-6240465 (2000.194) needs the 768 mas parallax. Gaia: 2-parameter field
    # stars near Proxima's 2016 position must not be moved with Proxima's motion.
    "proxima_j2000_pm": ["chandra", "xmm", "twomass_psc", "gaia_dr3"],
    "gj1151_j2000_pm": ["lotss"],  # LoTSS-DR3: ILTJ115055.51+482224.3 is GJ 1151 at ~2014.4
    "vlass_duplicate": ["vlass"],
}
# Primary-archive outages replayed from real fallback responses: the primary (IRSA TAP)
# is forced to answer 503 while recording, so only the fallback (Gator) exchange is kept.
# Stored under fixtures/fallback/<target>/<catalog>.json.
FALLBACK_RECORDINGS: dict[str, list[str]] = {"3c273": ["twomass_psc", "allwise"]}


def target_radius(key: str) -> float:
    return float(EPOCH_TARGETS.get(key, {}).get("radius_arcsec", RECORD_RADIUS_ARCSEC))


def target_for(key: str):
    """Validated Target for a canary (TARGETS) or epoch (EPOCH_TARGETS) fixture key."""
    from models import validate_target

    if key in EPOCH_TARGETS:
        spec = EPOCH_TARGETS[key]
        return validate_target(spec["ra"], spec["dec"], epoch=spec.get("epoch"),
                               pm_ra_masyr=spec.get("pm_ra_masyr"), pm_dec_masyr=spec.get("pm_dec_masyr"))
    return validate_target(*TARGETS[key])


def target_coords(key: str) -> tuple[float, float]:
    if key in EPOCH_TARGETS:
        return EPOCH_TARGETS[key]["ra"], EPOCH_TARGETS[key]["dec"]
    return TARGETS[key]


@dataclass
class Exchange:
    """One recorded HTTP request/response pair."""

    catalog: str
    method: str
    url: str
    request_body: str
    status_code: int
    content_type: str
    content: bytes
    match: list[str]

    @property
    def url_base(self) -> str:
        parts = urlsplit(self.url)
        return f"{parts.scheme}://{parts.netloc}{parts.path}"

    @property
    def signature(self) -> dict[str, list[str]]:
        """Normalized request parameters: URL query plus form body (order-insensitive)."""
        return request_signature(urlsplit(self.url).query, self.request_body)


def request_signature(query: str | bytes, body: str | bytes) -> dict[str, list[str]]:
    """Merge URL query parameters and a urlencoded form body into one comparable dict."""
    if isinstance(query, bytes):
        query = query.decode("utf-8", "replace")
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    merged: dict[str, list[str]] = {}
    for part in (query or "", body or ""):
        for key, values in parse_qs(part, keep_blank_values=True).items():
            merged.setdefault(key, []).extend(values)
    return {k: sorted(v) for k, v in sorted(merged.items())}


class FixtureMismatch(AssertionError):
    """The code sent a request that differs from every recorded one (query regression)."""


def match_tokens(catalog: Any) -> list[str]:
    """Strings that identify a catalog's request among others sent to a shared endpoint."""
    tokens: list[str] = []
    if catalog.table:
        tokens.append(f"FROM {catalog.table} ")
    if catalog.catalog:
        tokens.append(f"catalog={catalog.catalog}")
    return tokens


_IPV4 = re.compile(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def redact(content: bytes) -> bytes:
    """Replace IPv4 addresses (services echo the caller's IP) with a documentation address."""
    return _IPV4.sub(b"203.0.113.1", content)


def _request_text(request: httpx.Request) -> str:
    body = request.content.decode("utf-8", "replace") if request.content else ""
    return unquote_plus(str(request.url)) + "\n" + unquote_plus(body)


# ---------------------------------------------------------------------------
# Loading & replay
# ---------------------------------------------------------------------------


def load_exchanges(target: str, catalogs: list[str] | None = None) -> list[Exchange]:
    folder = FIXTURES / target
    exchanges: list[Exchange] = []
    for meta_path in sorted(folder.glob("*.json")):
        catalog = meta_path.stem
        if catalogs is not None and catalog not in catalogs:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(meta["exchanges"]):
            exchanges.append(
                Exchange(
                    catalog=catalog,
                    method=item["method"],
                    url=item["url"],
                    request_body=item.get("request_body", ""),
                    status_code=item["status_code"],
                    content_type=item["content_type"],
                    content=(folder / f"{catalog}.{idx}.body").read_bytes(),
                    match=item.get("match", []),
                )
            )
    return exchanges


def replay_side_effect(exchanges: list[Exchange], *, strict: bool = True):
    """Build a respx side effect that answers each request from matching recordings.

    ``strict`` (default): the outgoing request must equal a recorded one -- same
    method, URL base, query parameters and form body (ADQL, columns, radius, table,
    TOP N ...). A mismatch raises :class:`FixtureMismatch`, so a query regression
    can never be served a stale recording. ``strict=False`` matches on method + URL
    base only (for hand-built error fixtures whose exact request is irrelevant).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        candidates = [e for e in exchanges if e.method == request.method and e.url_base == base]
        if strict:
            sent = request_signature(request.url.query, request.content or b"")
            exact = [e for e in candidates if e.signature == sent]
            if not exact:
                recorded = [e.signature for e in candidates]
                raise FixtureMismatch(
                    f"request does not match any recording for {request.method} {base}.\n"
                    f"sent:     {sent}\nrecorded: {recorded[-1] if recorded else 'none'}"
                )
            candidates = exact
        elif len(candidates) > 1:
            text = _request_text(request)
            narrowed = [e for e in candidates if e.match and any(tok in text for tok in e.match)]
            candidates = narrowed or candidates
        # Prefer the last recorded exchange for a catalog (after any server retries).
        if not candidates:
            return httpx.Response(599, text=f"no fixture for {request.method} {base}")
        chosen = candidates[-1]
        return httpx.Response(chosen.status_code, headers={"content-type": chosen.content_type}, content=chosen.content)

    return handler


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


async def _record_catalog(name: str, target_key: str, ra: float, dec: float) -> tuple[str, list[dict[str, Any]], str]:
    from crossmatch import QueryExecutor
    from models import CatalogRegistry, QueryPlan
    from providers import CacheManager, provider_map

    registry = CatalogRegistry()
    catalog = registry.get(name)
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    outcome = "ok"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        providers = provider_map(client, timeout=120.0, cache=CacheManager(None))
        executor = QueryExecutor(providers, timeout=150.0, registry=registry)
        plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, target_radius(target_key), catalog.wavelength)
        successes, failures = await executor.execute([plan], target_for(target_key))
        if failures:
            outcome = f"FAILED {failures[0].error_type}: {failures[0].message}"
        else:
            outcome = f"{len(successes[0][1])} rows"

    records: list[dict[str, Any]] = []
    for request, response in log:
        records.append(
            {
                "method": request.method,
                "url": str(request.url),
                "request_body": request.content.decode("utf-8", "replace") if request.content else "",
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "match": match_tokens(catalog),
                "_content": response.content,
            }
        )
    return name, records, outcome


def _save(target_key: str, name: str, records: list[dict[str, Any]], *, coords_key: str | None = None) -> None:
    folder = FIXTURES / target_key
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{name}.*.body"):
        old.unlink()
    exchanges = []
    for idx, record in enumerate(records):
        (folder / f"{name}.{idx}.body").write_bytes(redact(record.pop("_content")))
        exchanges.append(record)
    key = coords_key or target_key
    ra, dec = target_coords(key)
    spec = EPOCH_TARGETS.get(key, {})
    meta = {"catalog": name, "target": key, "ra": ra, "dec": dec, "epoch": spec.get("epoch"),
            "pm_ra_masyr": spec.get("pm_ra_masyr"), "pm_dec_masyr": spec.get("pm_dec_masyr"),
            "radius_arcsec": target_radius(key), "exchanges": exchanges}
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


async def record(targets: list[str], catalogs: list[str] | None = None) -> None:
    from models import CatalogRegistry

    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    for target_key in targets:
        names = catalogs or (EPOCH_TARGET_CATALOG_SETS[target_key] if target_key in EPOCH_TARGETS else list(CatalogRegistry().catalogs))
        ra, dec = target_coords(target_key)
        results = await asyncio.gather(*(_record_catalog(n, target_key, ra, dec) for n in names))
        for name, records, outcome in results:
            if records:
                _save(target_key, name, records)
            print(f"{target_key:<10} {name:<24} {outcome}")


class _PrimaryDown(httpx.AsyncBaseTransport):
    """Answers 503 for the primary endpoint; forwards (and logs) everything else."""

    def __init__(self, primary: str) -> None:
        self.primary = primary
        self.real = httpx.AsyncHTTPTransport()
        self.log: list[tuple[httpx.Request, httpx.Response]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(self.primary):
            return httpx.Response(503, text="simulated primary outage", request=request)
        response = await self.real.handle_async_request(request)
        await response.aread()
        self.log.append((request, response))
        return response


async def record_fallbacks() -> None:
    """Record real fallback (Gator) responses with the primary IRSA TAP forced to fail."""
    from crossmatch import QueryExecutor
    from models import IRSA_TAP, CatalogRegistry, QueryPlan
    from providers import CacheManager, provider_map

    registry = CatalogRegistry()
    for target_key, names in FALLBACK_RECORDINGS.items():
        for name in names:
            catalog = registry.get(name)
            transport = _PrimaryDown(IRSA_TAP)
            log = transport.log
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, transport=transport) as client:
                executor = QueryExecutor(provider_map(client, timeout=120.0, cache=CacheManager(None)), timeout=150.0,
                                         registry=registry)
                plan = QueryPlan(name, catalog.provider, catalog.endpoint, {}, target_radius(target_key), catalog.wavelength)
                successes, failures = await executor.execute([plan], target_for(target_key))
            outcome = (f"FAILED {failures[0].error_type}: {failures[0].message}" if failures
                       else f"{len(successes[0][1])} rows via {successes[0][1].meta.get('fallback')}")
            records = [{
                "method": req.method, "url": str(req.url),
                "request_body": req.content.decode("utf-8", "replace") if req.content else "",
                "status_code": resp.status_code, "content_type": resp.headers.get("content-type", ""),
                "match": match_tokens(catalog), "_content": resp.content,
            } for req, resp in log]
            if records and not failures:
                _save(f"fallback/{target_key}", name, records, coords_key=target_key)
            print(f"fallback   {target_key:<10} {name:<24} {outcome}")


async def record_errors() -> None:
    """Record real error responses used by the status/regression tests."""
    folder = FIXTURES / "errors"
    folder.mkdir(parents=True, exist_ok=True)
    ra, dec = TARGETS["3c273"]
    cone = f"CIRCLE('ICRS', {ra}, {dec}, 0.0027777778)"
    cases = {
        # Bug 4: GET with '...)) = 1' in the query string is blocked by Cloudflare (403 HTML).
        "exoplanet_get_403": ("GET", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync", {
            "query": f"select pl_name, ra, dec from ps where contains(point('icrs', ra, dec), {cone.lower()}) = 1",
            "format": "json"}, None),
        # HEASARC reports SQL errors as HTTP 200 + QUERY_STATUS=ERROR.
        "heasarc_bad_column": ("POST", "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync", None, {
            "REQUEST": "doQuery", "LANG": "ADQL",
            "QUERY": f"SELECT TOP 5 name, ra, dec, hard_hs FROM csc WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {cone})"}),
        # CDS services report ADQL errors as HTTP 400 + VOTable QUERY_STATUS=ERROR.
        "vizier_unquoted_table": ("POST", "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync", None, {
            "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
            "QUERY": f"SELECT TOP 5 2MASS, RAJ2000, DEJ2000 FROM II/246/out WHERE CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), {cone}) = 1"}),
        # SDSS SkyServer SQL error -> HTTP 500 with JSON ErrorMessage.
        "sdss_sql_error": ("GET", "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch", {
            "cmd": "SELECT TOP 1 nosuchcolumn FROM PhotoPrimary", "format": "json"}, None),
        # Bug 5: legacy ConeSearchService applies 'sr' in ARCMIN (10" = 0.1667').
        "sdss_conesearch_3c273": ("GET", "https://skyserver.sdss.org/dr18/SkyServerWS/ConeSearch/ConeSearchService", {
            "ra": str(ra), "dec": str(dec), "sr": f"{10.0 / 60.0:.10f}", "format": "csv"}, None),
        # Bug 3: Xamin with position= and radius in ARCMIN returns the right sky area.
        "xamin_nvss_position": ("GET", "https://heasarc.gsfc.nasa.gov/xamin/query", {
            "table": "nvss", "position": f"{ra},{dec}", "radius": f"{10.0 / 60.0:.8f}", "format": "json"}, None),
    }
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for key, (method, url, params, data) in cases.items():
            try:
                response = await client.request(method, url, params=params, data=data)
            except httpx.HTTPError as exc:
                print(f"errors     {key:<24} network error {exc!r}")
                continue
            (folder / f"{key}.0.body").write_bytes(redact(response.content))
            meta = {"exchanges": [{
                "method": method, "url": str(response.request.url),
                "request_body": response.request.content.decode("utf-8", "replace") if response.request.content else "",
                "status_code": response.status_code, "content_type": response.headers.get("content-type", ""),
                "match": [],
            }]}
            (folder / f"{key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"errors     {key:<24} HTTP {response.status_code} {response.headers.get('content-type', '')} {len(response.content)} bytes")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "record":
        print(__doc__)
        return
    rest = args[1:]
    if rest == ["errors"]:
        asyncio.run(record_errors())
        return
    if rest == ["fallbacks"]:
        asyncio.run(record_fallbacks())
        return
    known = set(TARGETS) | set(EPOCH_TARGETS)
    targets = [t for t in rest if t in known] or list(TARGETS) + list(EPOCH_TARGETS)
    catalogs = [c for c in rest if c not in known] or None
    asyncio.run(record(targets, catalogs))
    if not rest:
        asyncio.run(record_errors())


if __name__ == "__main__":
    main()
