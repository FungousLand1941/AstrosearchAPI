"""Atomic catalog snapshots, typed MIT CSAIL Mantis exports, and resumable publication."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from astronomy import (
    SCHEMA_VERSION,
    ArchiveError,
    ExoplanetArchive,
    TapClient,
    canonical_json,
    digest,
    entity_id,
    match_hosts,
    object_summary,
    summarize_system,
    system_summary,
    utcnow,
)

MAP_FIELDS = ["catalog_id", "title", "kind", "host_id", "summary", "source", "source_url", "match_status",
              "ra_deg", "dec_deg", "planet_count", "discovery_method", "period_days", "radius_earth",
              "mass_earth", "distance_pc", "source_updated_at"]


@contextmanager
def state_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".sync.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Another sync/publication holds .sync.lock; after a crashed run, verify no job is running before removing it") from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def write_json(path: Path, value: Any):
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temp.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def map_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows, seen_simbad = [], set()
    for system in summary["systems"]:
        members = system["planets"]
        first = members[0]["record"]
        crossref = system["simbad"]
        rows.append({"catalog_id": system["id"], "title": system["name"], "kind": "host_system",
                     "host_id": system["id"], "summary": system["text"], "source": "NASA Exoplanet Archive + SIMBAD",
                     "source_url": "https://exoplanetarchive.ipac.caltech.edu/overview/" + quote(system["name"], safe=""),
                     "match_status": crossref["status"], "planet_count": len(members),
                     "ra_deg": first.get("ra"), "dec_deg": first.get("dec"), "distance_pc": first.get("sy_dist")})
        for p in members:
            raw = p["record"]
            rows.append({"catalog_id": p["id"], "title": p["name"], "kind": "planet", "host_id": system["id"],
                         "summary": p["text"] + " " + " ".join(p["warnings"]), "source": "NASA Exoplanet Archive",
                         "source_url": p["source_url"], "source_updated_at": p["source_updated_at"],
                         "match_status": "host_" + crossref["status"], "ra_deg": raw.get("ra"), "dec_deg": raw.get("dec"),
                         "discovery_method": raw.get("discoverymethod"), "period_days": raw.get("pl_orbper"),
                         "radius_earth": raw.get("pl_rade"), "mass_earth": raw.get("pl_bmasse"), "distance_pc": raw.get("sy_dist")})
        for candidate in crossref["candidates"]:
            star = candidate["record"]
            sid = entity_id("simbad", str(star["oid"]))
            if sid in seen_simbad:
                continue
            seen_simbad.add(sid)
            rows.append({"catalog_id": sid, "title": star["main_id"], "kind": "simbad_object", "source": "SIMBAD",
                         "summary": f"{star['main_id']}. SIMBAD type {star.get('otype') or 'unknown'}. "
                                    f"Spectral type {star.get('sp_type') or 'unknown'}. "
                                    "Identifier cross-reference candidate for exoplanet host records; see relationships.json.",
                         "source_url": "https://simbad.cds.unistra.fr/simbad/sim-id?Ident=" + quote(star["main_id"], safe=""),
                         "match_status": "candidate", "ra_deg": star.get("ra"), "dec_deg": star.get("dec")})
    return sorted(rows, key=lambda r: r["catalog_id"])


def relationships(summary: dict[str, Any]) -> list[dict[str, Any]]:
    edges = []
    for s in summary["systems"]:
        for p in s["planets"]:
            edges.append({"from": p["id"], "to": s["id"], "relation": "orbits_host", "source": "NASA Exoplanet Archive"})
        for candidate in s["simbad"]["candidates"]:
            edges.append({"from": s["id"], "to": entity_id("simbad", str(candidate["record"]["oid"])),
                          "relation": "identifier_match" if s["simbad"]["status"] == "matched" else "ambiguous_candidate",
                          "source": "SIMBAD ident", "aliases": candidate["matched_aliases"]})
    return edges


def commit_snapshot(root: Path, planets: list[dict[str, Any]], matches: dict[str, Any], provenance: list[dict[str, Any]]) -> dict[str, Any]:
    if not planets:
        raise ArchiveError("Refusing to replace a catalog snapshot with an empty result")
    planets = sorted(planets, key=lambda p: p["pl_name"])
    payload = {"schema_version": SCHEMA_VERSION, "planets": planets, "matches": matches}
    fingerprint = digest(payload)
    previous = load_json(root / "latest.json")
    if previous and previous["fingerprint"] == fingerprint:
        write_json(root / "last-check.json", {"checked_at": utcnow(), "changed": False, "fingerprint": fingerprint,
                                             "provenance": provenance})
        return {**previous, "changed": False}
    prior = load_json(root / "snapshots" / previous["fingerprint"] / "catalog.json", {}) if previous else {}
    old = {p["pl_name"]: digest(p) for p in prior.get("planets", [])}
    new = {p["pl_name"]: digest(p) for p in planets}
    changes = {"added": sorted(new.keys() - old.keys()), "removed": sorted(old.keys() - new.keys()),
               "updated": sorted(k for k in new.keys() & old.keys() if old[k] != new[k]),
               "simbad_changed": prior.get("matches") != matches}
    final = root / "snapshots" / fingerprint
    stage = root / "snapshots" / (".staging-" + uuid.uuid4().hex)
    summary = system_summary(planets, matches)
    rows = map_rows(summary)
    manifest = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "created_at": utcnow(),
                "planet_count": len(planets), "host_count": len(matches), "map_rows": len(rows),
                "match_counts": dict(Counter(m["status"] for m in matches.values())), "changes": changes,
                "provenance": provenance}
    try:
        stage.mkdir(parents=True)
        for filename, data in {"catalog.json": payload, "summaries.json": summary,
                               "relationships.json": relationships(summary), "manifest.json": manifest}.items():
            write_json(stage / filename, data)
        with (stage / "mantis.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MAP_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if final.exists():
            shutil.rmtree(stage)
        else:
            stage.rename(final)
        write_json(root / "latest.json", manifest)
        write_json(root / "last-check.json", {"checked_at": utcnow(), "changed": True, "fingerprint": fingerprint})
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**manifest, "changed": True}


async def sync(root: Path) -> dict[str, Any]:
    with state_lock(root):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tap = TapClient(client)
            planets = await ExoplanetArchive(tap).fetch()
            matches = await match_hosts(tap, planets)
            return commit_snapshot(root, planets, matches, tap.provenance)


def mantis_command(csv_path: Path, fingerprint: str, *, space_id: str | None = None, space_name: str = "Astronomical Catalogs") -> list[str]:
    command = ["mantis", "create", "map", str(csv_path), "--space-mode", "existing" if space_id else "new"]
    if space_id:
        command.extend(["--space-id", str(uuid.UUID(space_id))])
    else:
        command.extend(["--space-name", space_name, "--private"])
    command.extend(["--map-name", f"Exoplanets and SIMBAD {fingerprint[:12]}", "--title-column", "title",
                    "--semantic-column", "summary", "--categoric-column", "catalog_id,kind,host_id,source,match_status,discovery_method",
                    "--numeric-column", "ra_deg,dec_deg,planet_count,period_days,radius_earth,mass_earth,distance_pc",
                    "--date-column", "source_updated_at", "--links-column", "source_url", "--no-activate"])
    return command


def run_mantis(arguments: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    executable = shutil.which("mantis")
    if not executable:
        raise RuntimeError("MIT CSAIL Mantis CLI is missing; install mantisai-cli and run mantis setup")
    # On Windows invoke the installed Node entry point directly, avoiding cmd.exe quoting.
    if os.name == "nt":
        script = Path(executable).parent / "node_modules" / "mantisai-cli" / "dist" / "mantis.js"
        node = shutil.which("node")
        if not node or not script.exists():
            raise RuntimeError("Cannot locate the installed Mantis Node entry point")
        argv = [node, str(script), *arguments]
    else:
        argv = [executable, *arguments]
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False)


def publish(root: Path, *, space_id: str | None = None, space_name: str = "Astronomical Catalogs") -> dict[str, Any]:
    with state_lock(root):
        latest = load_json(root / "latest.json")
        if not latest:
            raise RuntimeError("Run sync before publishing")
        fingerprint = latest["fingerprint"]
        path = root / "publications.json"
        publications = load_json(path, {})
        target = space_id or "new:" + space_name
        key = target + ":" + fingerprint
        if key in publications:
            saved = publications[key]
            if saved["status"] not in {"submitted", "complete"}:
                raise RuntimeError("Previous publication has an uncertain outcome. Inspect Mantis before reconciling publications.json; automatic retry could duplicate a map")
            return saved
        context = run_mantis(["use", "get_space_context"], timeout=60)
        if context.returncode:
            # MCP and REST have separate authentication paths. Map creation uses
            # REST, so an unavailable MCP session must not block a valid REST key.
            rest = run_mantis(["spaces", "list", "--filter", space_name], timeout=60)
            if rest.returncode:
                raise RuntimeError("Mantis authentication failed for both MCP and REST. Reconnect locally with mantis setup")
        targets_path = root / "mantis-targets.json"
        targets = load_json(targets_path, {})
        actual_space = space_id or targets.get(target)
        command = mantis_command(root / "snapshots" / fingerprint / "mantis.csv", fingerprint, space_id=actual_space, space_name=space_name)
        publications[key] = {"status": "pending", "started_at": utcnow(), "fingerprint": fingerprint}
        write_json(path, publications)
        response = run_mantis(command[1:], timeout=1800)
        if response.returncode:
            raise RuntimeError("Mantis publication failed or has an uncertain outcome. Inspect the space before retrying")
        # The CLI submits asynchronous embedding; an exit code alone is not evidence
        # that the map finished building. Preserve server identifiers for inspection.
        decoded = None
        for offset, character in enumerate(response.stdout):
            if character == "{":
                try:
                    value, _ = json.JSONDecoder().raw_decode(response.stdout[offset:])
                    if isinstance(value, dict) and value.get("space_id") and value.get("map_id"):
                        decoded = value
                except ValueError:
                    pass
        if decoded is None:
            raise RuntimeError("Mantis accepted the request but its identifiers could not be parsed; reconcile the pending publication before retrying")
        targets[target] = str(uuid.UUID(decoded["space_id"]))
        write_json(targets_path, targets)
        publications[key].update({"status": "submitted", "submitted_at": utcnow(), "result": decoded,
                                  "space_url": "https://mantis.csail.mit.edu/space/" + targets[target]})
        write_json(path, publications)
        return publications[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summarize = sub.add_parser("summarize", help="Summarize an extrasolar host or planet and its system")
    summarize.add_argument("name")
    general = sub.add_parser("summarize-object", help="Summarize a general astronomical object using catalog search")
    general.add_argument("name")
    general.add_argument("--profile", default="optical")
    synchronize = sub.add_parser("sync", help="Download complete exoplanet catalog and cross-reference SIMBAD hosts")
    synchronize.add_argument("--state-dir", type=Path, default=Path("astronomy-data"))
    publication = sub.add_parser("publish", help="Publish the latest snapshot as a private Mantis map")
    publication.add_argument("--state-dir", type=Path, default=Path("astronomy-data"))
    publication.add_argument("--space-id")
    publication.add_argument("--space-name", default="Astronomical Catalogs")
    publication.add_argument("--dry-run", action="store_true")
    watch = sub.add_parser("watch", help="Continuously refresh snapshots; optionally publish changed maps")
    watch.add_argument("--state-dir", type=Path, default=Path("astronomy-data"))
    watch.add_argument("--interval-hours", type=float, default=24)
    watch.add_argument("--publish", action="store_true")
    watch.add_argument("--space-id")
    arguments = parser.parse_args()
    try:
        if arguments.command == "summarize":
            async def run():
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    return await summarize_system(client, arguments.name)
            result = asyncio.run(run())
        elif arguments.command == "summarize-object":
            from main import search_object
            result = object_summary(asyncio.run(search_object(arguments.name, profile=arguments.profile)).as_dict())
        elif arguments.command == "sync":
            result = asyncio.run(sync(arguments.state_dir.resolve()))
        elif arguments.command == "watch":
            if not 1 <= arguments.interval_hours <= 8760:
                raise ValueError("interval-hours must be between 1 and 8760")
            while True:
                try:
                    result = asyncio.run(sync(arguments.state_dir.resolve()))
                    if arguments.publish:
                        publication_result = publish(arguments.state_dir.resolve(), space_id=arguments.space_id)
                        result["publication"] = publication_result
                    print(json.dumps({"checked_at": utcnow(), "changed": result["changed"],
                                      "fingerprint": result["fingerprint"]}), flush=True)
                except (ArchiveError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    print(json.dumps({"checked_at": utcnow(), "error": str(exc)}), file=sys.stderr, flush=True)
                time.sleep(arguments.interval_hours * 3600)
        elif arguments.dry_run:
            root = arguments.state_dir.resolve()
            latest = load_json(root / "latest.json")
            if not latest:
                raise RuntimeError("Run sync first")
            fingerprint = latest["fingerprint"]
            result = {"argv": mantis_command(root / "snapshots" / fingerprint / "mantis.csv", fingerprint,
                                             space_id=arguments.space_id, space_name=arguments.space_name)}
        else:
            result = publish(arguments.state_dir.resolve(), space_id=arguments.space_id, space_name=arguments.space_name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ArchiveError, LookupError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
