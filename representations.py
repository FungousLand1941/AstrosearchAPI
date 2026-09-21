"""Deterministic representations and candidate triage for astronomical signals.

The transparent, versioned baseline supports expert review. An anomaly is never
reported as a discovery without detector checks and independent follow-up.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

REPRESENTATION_VERSION = "astrosearch-signal-v1"
VECTOR_SIZE = 48
MIN_SAMPLES = 8


class RepresentationError(ValueError):
    """The observation cannot be represented without inventing data."""


@dataclass(frozen=True)
class SignalRepresentation:
    observation_id: str
    object_id: str | None
    modality: Literal["light_curve", "spectrum"]
    vector: list[float]
    sample_count: int
    input_sample_count: int
    good_fraction: float
    axis_min: float
    axis_max: float
    value_median: float
    value_scale: float
    version: str = REPRESENTATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RepresentationError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise RepresentationError(f"{name} must be finite")
    return number


def _prepare(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    axis = np.asarray(observation.get("axis", []), dtype=float)
    values = np.asarray(observation.get("values", []), dtype=float)
    if axis.ndim != 1 or values.ndim != 1 or len(axis) != len(values):
        raise RepresentationError("axis and values must be one-dimensional arrays of equal length")
    input_count = len(axis)
    if input_count < MIN_SAMPLES:
        raise RepresentationError(f"at least {MIN_SAMPLES} samples are required")
    mask = np.isfinite(axis) & np.isfinite(values)
    quality = observation.get("quality")
    if quality is not None:
        flags = np.asarray(quality)
        if flags.ndim != 1 or len(flags) != input_count:
            raise RepresentationError("quality must have one entry per sample")
        mask &= flags == 0
    errors = observation.get("uncertainties")
    clean_errors = None
    if errors is not None:
        clean_errors = np.asarray(errors, dtype=float)
        if clean_errors.ndim != 1 or len(clean_errors) != input_count:
            raise RepresentationError("uncertainties must have one entry per sample")
        mask &= np.isfinite(clean_errors) & (clean_errors > 0)
    axis, values = axis[mask], values[mask]
    clean_errors = clean_errors[mask] if clean_errors is not None else None
    if len(axis) < MIN_SAMPLES:
        raise RepresentationError(f"fewer than {MIN_SAMPLES} usable samples remain after quality filtering")
    order = np.argsort(axis, kind="stable")
    axis, values = axis[order], values[order]
    clean_errors = clean_errors[order] if clean_errors is not None else None
    if axis[-1] <= axis[0]:
        raise RepresentationError("axis must span more than one coordinate")
    return axis, values, clean_errors, input_count


def _shape_bins(axis: np.ndarray, values: np.ndarray, count: int = 32) -> np.ndarray:
    scaled = (axis - axis[0]) / (axis[-1] - axis[0])
    edges = np.linspace(0.0, 1.0, count + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    result = np.empty(count, dtype=float)
    for index in range(count):
        upper = scaled <= edges[index + 1] if index == count - 1 else scaled < edges[index + 1]
        selected = values[(scaled >= edges[index]) & upper]
        result[index] = np.median(selected) if len(selected) else np.nan
    missing = np.isnan(result)
    result[missing] = np.interp(centers[missing], centers[~missing], result[~missing])
    return result


def represent_signal(observation: dict[str, Any]) -> SignalRepresentation:
    """Convert a light curve or spectrum to a fixed, interpretable vector."""
    modality = observation.get("modality")
    if modality not in {"light_curve", "spectrum"}:
        raise RepresentationError("modality must be light_curve or spectrum")
    observation_id = str(observation.get("observation_id") or "").strip()
    if not observation_id:
        raise RepresentationError("observation_id is required")
    axis, values, errors, input_count = _prepare(observation)
    physical_axis_min, physical_axis_max = float(axis[0]), float(axis[-1])
    if modality == "spectrum" and np.all(axis > 0):
        axis = np.log(axis)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        raise RepresentationError("signal has no measurable variation")
    normalized = (values - median) / scale
    weights = None if errors is None else 1.0 / np.square(errors / scale)
    x = (axis - axis.mean()) / (axis[-1] - axis[0])
    slope = float(np.polyfit(x, normalized, 1, w=None if weights is None else np.sqrt(weights))[0])
    centered = normalized - normalized.mean()
    std = float(np.std(normalized))
    skew = float(np.mean(centered ** 3) / std ** 3) if std else 0.0
    kurtosis = float(np.mean(centered ** 4) / std ** 4 - 3) if std else 0.0
    derivative = np.diff(normalized) / np.maximum(np.diff(axis), np.finfo(float).eps)
    stats = np.asarray([
        float(np.mean(normalized)), std, float(np.min(normalized)), float(np.max(normalized)),
        *[float(v) for v in np.quantile(normalized, [0.05, 0.25, 0.75, 0.95])],
        skew, kurtosis, float(np.median(np.abs(derivative))), slope,
    ])
    correlations = []
    for lag in (1, 2, 4, 8):
        if len(normalized) <= lag or np.std(normalized[:-lag]) == 0 or np.std(normalized[lag:]) == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]))
    vector = np.nan_to_num(np.concatenate([stats, _shape_bins(axis, normalized), correlations]))
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return SignalRepresentation(
        observation_id=observation_id,
        object_id=str(observation["object_id"]).strip() if observation.get("object_id") else None,
        modality=modality,
        vector=[float(v) for v in vector],
        sample_count=len(axis), input_sample_count=input_count, good_fraction=len(axis) / input_count,
        axis_min=physical_axis_min, axis_max=physical_axis_max, value_median=median, value_scale=scale,
    )


def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    haversine = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((a2 - a1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(haversine)))) * 3600


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise RepresentationError("reference vectors must match the observation vector length")
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise RepresentationError("reference vectors must contain finite, non-zero values")
    return float(np.clip(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 2))


def cross_reference_observation(observation: dict[str, Any], references: list[dict[str, Any]], *,
                                radius_arcsec: float = 2.0, representation_threshold: float = 0.25,
                                anomaly_threshold: float = 0.45) -> dict[str, Any]:
    """Rank known counterparts and gate an observation for human novelty review."""
    if radius_arcsec <= 0 or not 0 <= representation_threshold <= 2 or not 0 <= anomaly_threshold <= 2:
        raise RepresentationError("invalid cross-reference threshold")
    representation = represent_signal(observation)
    ra, dec = _finite_number(observation.get("ra_deg"), "ra_deg"), _finite_number(observation.get("dec_deg"), "dec_deg")
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise RepresentationError("coordinates must satisfy 0 <= RA < 360 and -90 <= Dec <= 90")
    candidates, nearest = [], None
    for reference in references:
        separation = angular_separation_arcsec(ra, dec, _finite_number(reference.get("ra_deg"), "reference ra_deg"),
                                               _finite_number(reference.get("dec_deg"), "reference dec_deg"))
        distance = None
        compatible = (reference.get("modality") == representation.modality and
                      reference.get("representation_version") == representation.version)
        if reference.get("representation") is not None and compatible:
            distance = cosine_distance(representation.vector, reference["representation"])
            nearest = distance if nearest is None else min(nearest, distance)
        if separation <= radius_arcsec:
            candidates.append({"catalog_id": str(reference.get("catalog_id") or ""), "title": reference.get("title"),
                               "separation_arcsec": separation, "representation_distance": distance,
                               "position_match": True,
                               "representation_compatible": compatible,
                               "representation_match": distance is not None and distance <= representation_threshold})
    candidates.sort(key=lambda item: (item["separation_arcsec"], item["representation_distance"] or 3.0))
    supported = [candidate for candidate in candidates if candidate["representation_match"]]
    if len(supported) == 1:
        status = "matched"
    elif len(supported) > 1:
        status = "ambiguous"
    elif candidates:
        status = "position_only_review"
    elif nearest is None:
        status = "insufficient_reference_coverage"
    elif nearest >= anomaly_threshold and representation.good_fraction >= 0.8:
        status = "novelty_review_candidate"
    else:
        status = "unmatched"
    result = {
        "schema_version": 1, "status": status, "discovery_claim": False,
        "representation": representation.as_dict(), "coordinates": {"frame": "ICRS", "ra_deg": ra, "dec_deg": dec},
        "thresholds": {"radius_arcsec": radius_arcsec, "representation_distance": representation_threshold,
                       "anomaly_distance": anomaly_threshold},
        "nearest_representation_distance": nearest, "candidates": candidates,
        "required_follow_up": [
            "inspect pixels or detector-level data for artifacts and blending",
            "cross-match at the observation epoch with proper motion and solar-system ephemerides",
            "seek independent repeat observations or another instrument",
            "obtain domain-expert review before describing the source as a discovery",
        ],
    }
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def mantis_signal_row(observation: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rep = result["representation"]
    return {
        "catalog_id": "observation:" + rep["observation_id"], "title": observation.get("title") or rep["observation_id"],
        "kind": "astronomical_signal", "object_id": rep["object_id"] or "", "modality": rep["modality"],
        "instrument": observation.get("instrument") or "", "band": observation.get("band") or "",
        "triage_status": result["status"],
        "summary": f"{rep['modality']} observation with {rep['sample_count']} usable samples; triage status {result['status']}.",
        "ra_deg": result["coordinates"]["ra_deg"], "dec_deg": result["coordinates"]["dec_deg"],
        "sample_count": rep["sample_count"], "good_fraction": rep["good_fraction"],
        "nearest_representation_distance": result["nearest_representation_distance"],
        "representation": json.dumps(rep["vector"], separators=(",", ":")), "representation_version": rep["version"],
        "observed_at": observation.get("observed_at") or "", "source_url": observation.get("source_url") or "",
        "provenance": json.dumps(observation.get("provenance") or {}, sort_keys=True, separators=(",", ":")),
    }
