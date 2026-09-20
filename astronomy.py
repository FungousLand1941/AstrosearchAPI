"""Evidence-based astronomical summaries and NASA Exoplanet Archive / SIMBAD ingestion.

No generative model is needed: every statement is derived from preserved source data.
TAP VOTables carry QUERY_STATUS, which must be checked even on HTTP 200 responses.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
from astropy.io.votable import parse_single_table

NEA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
NEA_DOCS = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
SCHEMA_VERSION = 2
MEASURES = {
    "pl_orbper": ("orbital period", "days"),
    "pl_orbsmax": ("semi-major axis", "AU"),
    "pl_rade": ("radius", "Earth radii"),
    "pl_bmasse": ("reported mass parameter", "Earth masses"),
    "pl_orbeccen": ("orbital eccentricity", ""),
    "pl_eqt": ("equilibrium temperature", "K"),
    "st_teff": ("host effective temperature", "K"),
    "st_mass": ("host mass", "solar masses"),
    "st_rad": ("host radius", "solar radii"),
    "sy_dist": ("system distance", "pc"),
}
ALIAS_COLUMNS = ("hostname", "gaia_dr3_id", "gaia_dr2_id", "hd_name", "hip_name", "tic_id")
BASE_COLUMNS = [
    "pl_name", *ALIAS_COLUMNS, "ra", "dec", "sy_snum", "sy_pnum", "cb_flag",
    "discoverymethod", "disc_year", "disc_facility", "disc_refname", "pl_controv_flag",
    "pl_bmassprov", "rowupdate",
]


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def entity_id(kind: str, name: str) -> str:
    # Preserve case: binary star A and planet a must never silently collapse.
    return f"{kind}:{quote(name.strip(), safe='')}"


def literal(value: str) -> str:
    if not value or len(value) > 300 or any(ord(c) < 32 for c in value):
        raise ValueError("Object names must contain 1–300 printable characters")
    return "'" + value.replace("'", "''") + "'"


class ArchiveError(RuntimeError):
    """An upstream response is unavailable, invalid, or incomplete."""


def parse_tap(content: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(content)
        statuses = [e.attrib.get("value", "").upper() for e in root.iter()
                    if e.tag.split("}")[-1] == "INFO" and e.attrib.get("name") == "QUERY_STATUS"]
        if not statuses or any(s != "OK" for s in statuses):
            raise ArchiveError(f"TAP result rejected: QUERY_STATUS={statuses}")
        table = parse_single_table(io.BytesIO(content)).to_table()
        rows = []
        for row in table:
            clean = {}
            for key in table.colnames:
                value = row[key]
                if np.ma.is_masked(value):
                    value = None
                elif isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                clean[key] = value
            rows.append(clean)
        return rows
    except ArchiveError:
        raise
    except Exception as exc:
        raise ArchiveError("Invalid TAP VOTable response") from exc


class TapClient:
    def __init__(self, client: httpx.AsyncClient, *, delay: float = 0.2):
        self.client = client
        self.delay = delay
        self.provenance: list[dict[str, Any]] = []

    async def query(self, endpoint: str, query: str, *, maxrec: int = 20000) -> list[dict[str, Any]]:
        for attempt in range(4):
            try:
                async with self.client.stream("POST", endpoint, data={
                    "REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query,
                    "FORMAT": "votable/td" if endpoint == SIMBAD_TAP else "votable", "MAXREC": str(maxrec),
                }, timeout=120) as response:
                    response.raise_for_status()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > 64 * 1024 * 1024:
                            raise ArchiveError("TAP response exceeded 64 MiB safety limit")
                rows = parse_tap(bytes(data))
                self.provenance.append({"endpoint": endpoint, "query": query, "retrieved_at": utcnow(),
                                        "rows": len(rows), "response_sha256": hashlib.sha256(data).hexdigest()})
                if self.delay:
                    await asyncio.sleep(self.delay)
                return rows
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in (429, 500, 502, 503, 504)
                if not retryable or attempt == 3:
                    raise ArchiveError(f"TAP request failed at {endpoint}") from exc
                await asyncio.sleep(2 ** attempt)
        raise AssertionError("unreachable")


class ExoplanetArchive:
    def __init__(self, tap: TapClient):
        self.tap = tap

    async def fetch(self, *, name: str | None = None) -> list[dict[str, Any]]:
        schema = await self.tap.query(NEA_TAP, "SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='pscomppars'")
        available = {r["column_name"].lower() for r in schema}
        desired = BASE_COLUMNS + [m + suffix for m in MEASURES for suffix in ("", "err1", "err2", "lim", "_reflink")]
        if not {"pl_name", "hostname", "ra", "dec"} <= available:
            raise ArchiveError("Exoplanet Archive schema lacks required identity columns")
        columns = [c for c in desired if c in available]
        conditions = []
        if name is not None:
            value = literal(name.strip())
            conditions.append("(" + " OR ".join(f"{c}={value}" for c in ("pl_name", *ALIAS_COLUMNS) if c in available) + ")")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        expected = await self.tap.query(NEA_TAP, "SELECT count(*) AS n FROM pscomppars" + where)
        total = int(expected[0]["n"])
        # This service can apply TOP before ORDER BY. Keyset paging would silently
        # skip records. Request the bounded complete catalog and verify its count.
        if total > 100000:
            raise ArchiveError("Catalog exceeds the 100,000-row synchronous snapshot limit")
        result = await self.tap.query(NEA_TAP, f"SELECT {','.join(columns)} FROM pscomppars{where} ORDER BY pl_name", maxrec=total + 1)
        if len(result) != total or len({r["pl_name"] for r in result}) != total:
            raise ArchiveError(f"Archive changed or truncated during download: expected {total}, received {len(result)}, "
                               f"unique {len({r['pl_name'] for r in result})}; retry the snapshot")
        return result

    async def system(self, name: str) -> list[dict[str, Any]]:
        rows = await self.fetch(name=name)
        hosts = sorted({r["hostname"] for r in rows})
        if len(hosts) == 1 and any(r["pl_name"] == name for r in rows):
            return await self.fetch(name=hosts[0])
        return rows


def aliases(planets: list[dict[str, Any]]) -> list[str]:
    return sorted({str(p[c]).strip() for p in planets for c in ALIAS_COLUMNS if p.get(c)})


async def match_hosts(tap: TapClient, planets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    hosts = defaultdict(list)
    for p in planets:
        hosts[p["hostname"]].append(p)
    names = aliases(planets)
    matched = defaultdict(dict)
    for offset in range(0, len(names), 100):
        batch = names[offset:offset + 100]
        query = ("SELECT i.id,b.oid,b.main_id,b.ra,b.dec,b.otype,b.sp_type,b.plx_value,b.pmra,b.pmdec "
                 "FROM ident AS i JOIN basic AS b ON i.oidref=b.oid WHERE i.id IN (" +
                 ",".join(literal(n) for n in batch) + ")")
        rows = await tap.query(SIMBAD_TAP, query)
        for row in rows:
            matched[re.sub(r"\s+", " ", row["id"].strip())][str(row["oid"])] = row
    result = {}
    for host, members in sorted(hosts.items()):
        candidates = {}
        for alias in aliases(members):
            for oid, row in matched[re.sub(r"\s+", " ", alias)].items():
                if oid not in candidates:
                    candidates[oid] = {"record": {k: v for k, v in row.items() if k != "id"}, "matched_aliases": []}
                candidates[oid]["matched_aliases"].append(alias)
        result[host] = {"status": "matched" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched",
                        "method": "exact_identifier", "candidates": sorted(candidates.values(), key=lambda c: str(c["record"]["oid"]))}
    return result


def measurement(row: dict[str, Any], field: str) -> dict[str, Any] | None:
    value = row.get(field)
    if value is None:
        return None
    label, unit = MEASURES[field]
    return {"field": field, "label": label, "value": value, "unit": unit,
            "error_plus": row.get(field + "err1"), "error_minus": row.get(field + "err2"),
            "limit": row.get(field + "lim"), "reference": row.get(field + "_reflink")}


def planet_summary(row: dict[str, Any]) -> dict[str, Any]:
    measures = [m for f in MEASURES if (m := measurement(row, f)) is not None]
    sentences = [f"{row['pl_name']} is listed in the NASA Exoplanet Archive around {row['hostname']}."]
    if row.get("discoverymethod"):
        year = f" in {int(row['disc_year'])}" if row.get("disc_year") else ""
        sentences.append(f"Discovery method: {row['discoverymethod']}{year}.")
    for m in measures:
        bound = {1: "upper limit ", -1: "lower limit "}.get(m["limit"], "")
        errors = ""
        if m["error_plus"] is not None or m["error_minus"] is not None:
            plus = str(m["error_plus"]) if m["error_plus"] is not None else "unknown"
            minus = str(m["error_minus"]) if m["error_minus"] is not None else "unknown"
            errors = f" (upper error {plus}, lower error {minus})"
        sentences.append(f"{m['label'].capitalize()}: {bound}{m['value']} {m['unit']}{errors}.")
    if row.get("pl_bmassprov"):
        sentences.append(f"Mass parameter provenance: {row['pl_bmassprov']}.")
    warnings = ["Composite parameters may come from different publications and need not be a self-consistent solution.",
                "Missing measurements are unknown; equilibrium temperature alone does not establish habitability."]
    if row.get("pl_controv_flag") == 1:
        warnings.append("The archive flags this planet's confirmation as controversial.")
    return {"id": entity_id("planet", row["pl_name"]), "name": row["pl_name"], "host": row["hostname"],
            "text": " ".join(sentences), "measurements": measures, "warnings": warnings,
            "source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/" + quote(row["pl_name"], safe=""),
            "source_updated_at": row.get("rowupdate"), "discovery_reference": row.get("disc_refname"), "record": row}


def system_summary(planets: list[dict[str, Any]], matches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for p in planets:
        groups[p["hostname"]].append(p)
    systems = []
    for name, members in sorted(groups.items()):
        crossref = matches.get(name, {"status": "unavailable", "candidates": []})
        summaries = [planet_summary(p) for p in sorted(members, key=lambda p: p["pl_name"])]
        text = f"{name}: {len(members)} planet records in this selection. "
        text += f"SIMBAD host cross-reference: {crossref['status']}."
        if crossref["status"] == "matched":
            star = crossref["candidates"][0]["record"]
            text += f" SIMBAD identifier: {star['main_id']}; object type: {star.get('otype') or 'unknown'}."
            if star.get("sp_type"):
                text += f" Spectral type: {star['sp_type']}."
        systems.append({"id": entity_id("host", name), "name": name, "text": text,
                        "planet_count": len(members), "planets": summaries, "simbad": crossref,
                        "warnings": ["Host matches do not identify planets as the same object as their stars.",
                                     "Identifier matches are catalog associations, not independent physical confirmation."]})
    return {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "systems": systems}


def object_summary(result: dict[str, Any]) -> dict[str, Any]:
    resolved = result.get("resolved_object") or {}
    name = resolved.get("canonical_name") or "Coordinate target"
    target = result.get("target", {})
    counterparts = result.get("counterparts", {})
    counts = {wave: len(rows) for wave, rows in counterparts.items()}
    failures = result.get("failures", [])
    text = f"{name}: RA {target.get('ra', 'unknown')} deg, Dec {target.get('dec', 'unknown')} deg. "
    text += f"Queried {result.get('catalogs_queried', 0)} catalogs; returned {sum(counts.values())} candidate counterparts. "
    text += f"{len(failures)} catalog queries failed."
    candidates = []
    for wave, rows in counterparts.items():
        for row in rows:
            physical = row.get("physical") or {}
            candidate = {"catalog": row.get("catalog"), "source_id": row.get("source_id"), "wavelength": wave,
                         "separation_arcsec": row.get("separation_arcsec"), "match_score": row.get("confidence"),
                         "object_type": physical.get("object_type"), "spectral_type": physical.get("spectral_type"),
                         "links": row.get("links", {}), "provenance": row.get("provenance", {})}
            candidates.append(candidate)
    for candidate in candidates[:5]:
        text += f" Candidate {candidate['source_id'] or 'unnamed'} in {candidate['catalog'] or 'unknown catalog'}"
        if candidate["separation_arcsec"] is not None:
            text += f" lies {candidate['separation_arcsec']:.3g} arcsec from the query position"
        if candidate["object_type"]:
            text += f"; catalog object type {candidate['object_type']}"
        if candidate["spectral_type"]:
            text += f"; spectral type {candidate['spectral_type']}"
        text += "."
    return {"schema_version": SCHEMA_VERSION, "name": name, "text": text, "counts_by_wavelength": counts,
            "candidates": candidates,
            "status": "partial" if failures else "complete", "failures": failures,
            "warnings": ["Cone-search counterparts are candidates; proximity does not establish object identity.",
                         "No detections in queried catalogs does not prove an object is absent."],
            "evidence": result}


async def summarize_system(client: httpx.AsyncClient, name: str) -> dict[str, Any]:
    tap = TapClient(client)
    planets = await ExoplanetArchive(tap).system(name)
    if not planets:
        raise LookupError(f"No Exoplanet Archive system found for {name}")
    matches = await match_hosts(tap, planets)
    result = system_summary(planets, matches)
    result["provenance"] = tap.provenance
    return result
