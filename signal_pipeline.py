"""Immutable telescope-signal batches and Mantis-ready vector exports."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from astronomy import digest, utcnow
from astronomy_pipeline import load_json, state_lock, write_json
from representations import cross_reference_observation, mantis_signal_row

SIGNAL_FIELDS = [
    "catalog_id", "title", "kind", "object_id", "modality", "instrument", "band", "triage_status", "summary",
    "ra_deg", "dec_deg", "sample_count", "good_fraction", "nearest_representation_distance", "representation",
    "representation_version", "observed_at", "source_url", "provenance",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} of {path} is not an object")
            records.append(value)
    return records


def _commit_signal_batch(root: Path, observations: list[dict[str, Any]], references: list[dict[str, Any]], *,
                         radius_arcsec: float = 2.0, representation_threshold: float = 0.25,
                         anomaly_threshold: float = 0.45) -> dict[str, Any]:
    """Validate, deduplicate, represent, triage, and atomically commit a delivery."""
    if not observations:
        raise ValueError("refusing to commit an empty signal delivery")
    by_id: dict[str, dict[str, Any]] = {}
    for observation in observations:
        observation_id = str(observation.get("observation_id") or "").strip()
        if not observation_id:
            raise ValueError("every observation requires observation_id")
        if observation_id in by_id and digest(by_id[observation_id]) != digest(observation):
            raise ValueError(f"conflicting duplicate observation_id: {observation_id}")
        by_id[observation_id] = observation
    ordered = [by_id[key] for key in sorted(by_id)]
    results = [cross_reference_observation(
        observation, references, radius_arcsec=radius_arcsec,
        representation_threshold=representation_threshold, anomaly_threshold=anomaly_threshold,
    ) for observation in ordered]
    payload = {
        "schema_version": 1, "representation_version": results[0]["representation"]["version"],
        "observations": ordered, "reference_fingerprint": digest(references),
        "thresholds": {"radius_arcsec": radius_arcsec, "representation_distance": representation_threshold,
                       "anomaly_distance": anomaly_threshold},
    }
    fingerprint = digest(payload)
    previous = load_json(root / "latest-signals.json")
    if previous and previous["fingerprint"] == fingerprint:
        write_json(root / "last-signal-check.json", {"checked_at": utcnow(), "changed": False, "fingerprint": fingerprint})
        return {**previous, "changed": False}
    rows = [mantis_signal_row(observation, result) for observation, result in zip(ordered, results, strict=True)]
    final = root / "signal-snapshots" / fingerprint
    stage = root / "signal-snapshots" / (".staging-" + uuid.uuid4().hex)
    manifest = {
        "schema_version": 1, "fingerprint": fingerprint, "created_at": utcnow(), "observation_count": len(ordered),
        "reference_count": len(references), "status_counts": dict(Counter(result["status"] for result in results)),
        "representation_version": results[0]["representation"]["version"],
    }
    try:
        stage.mkdir(parents=True)
        write_json(stage / "delivery.json", payload)
        write_json(stage / "cross-reference-results.json", results)
        write_json(stage / "manifest.json", manifest)
        with (stage / "mantis-signals.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if final.exists():
            shutil.rmtree(stage)
        else:
            stage.rename(final)
        write_json(root / "latest-signals.json", manifest)
        write_json(root / "last-signal-check.json", {"checked_at": utcnow(), "changed": True, "fingerprint": fingerprint})
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**manifest, "changed": True, "snapshot_path": str(final)}


def commit_signal_batch(root: Path, observations: list[dict[str, Any]], references: list[dict[str, Any]], *,
                        radius_arcsec: float = 2.0, representation_threshold: float = 0.25,
                        anomaly_threshold: float = 0.45) -> dict[str, Any]:
    """Commit one delivery while excluding concurrent catalog or signal writers."""
    with state_lock(root):
        return _commit_signal_batch(
            root, observations, references, radius_arcsec=radius_arcsec,
            representation_threshold=representation_threshold, anomaly_threshold=anomaly_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="JSON Lines telescope delivery")
    parser.add_argument("--references", type=Path, required=True, help="JSON array of catalog/reference objects")
    parser.add_argument("--state-dir", type=Path, default=Path("astronomy-data"))
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--representation-threshold", type=float, default=0.25)
    parser.add_argument("--anomaly-threshold", type=float, default=0.45)
    arguments = parser.parse_args()
    references = json.loads(arguments.references.read_text(encoding="utf-8"))
    if not isinstance(references, list):
        raise SystemExit("references must be a JSON array")
    result = commit_signal_batch(
        arguments.state_dir.resolve(), load_jsonl(arguments.observations), references,
        radius_arcsec=arguments.radius_arcsec, representation_threshold=arguments.representation_threshold,
        anomaly_threshold=arguments.anomaly_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
