import csv
import json

import numpy as np
import pytest

from representations import VECTOR_SIZE, RepresentationError, cross_reference_observation, mantis_signal_row, represent_signal
from signal_pipeline import commit_signal_batch


def observation(**changes):
    axis = np.linspace(0, 10, 64)
    result = {
        "observation_id": "tess-sector-1-tic-42", "object_id": "TIC 42", "modality": "light_curve",
        "ra_deg": 10.0, "dec_deg": 20.0, "axis": axis.tolist(),
        "values": (1 + 0.01 * np.sin(axis)).tolist(), "uncertainties": [0.001] * len(axis),
        "quality": [0] * len(axis), "instrument": "TESS", "band": "TESS",
        "provenance": {"archive": "MAST", "product": "example_lc.fits"},
    }
    result.update(changes)
    return result


def reference(vector, **changes):
    result = {"catalog_id": "known:1", "ra_deg": 10, "dec_deg": 20, "representation": vector,
              "modality": "light_curve", "representation_version": "astrosearch-signal-v1"}
    result.update(changes)
    return result


def test_representation_is_fixed_deterministic_and_quality_aware():
    item = observation()
    item["quality"][3] = 1
    first = represent_signal(item)
    second = represent_signal(item)
    assert first.vector == second.vector
    assert len(first.vector) == VECTOR_SIZE
    assert first.sample_count == 63
    assert first.good_fraction == 63 / 64
    assert np.linalg.norm(first.vector) == pytest.approx(1)


def test_spectrum_uses_physical_axis_bounds_and_rejects_bad_inputs():
    spectrum = observation(modality="spectrum", axis=np.linspace(1, 5, 64).tolist())
    represented = represent_signal(spectrum)
    assert represented.axis_min == 1
    assert represented.axis_max == 5
    with pytest.raises(RepresentationError, match="usable samples"):
        represent_signal(observation(quality=[1] * 64))
    with pytest.raises(RepresentationError, match="coordinates"):
        cross_reference_observation(observation(ra_deg=400), [])


def test_position_and_representation_are_separate_evidence():
    item = observation()
    vector = represent_signal(item).vector
    matched = cross_reference_observation(item, [reference(vector, title="Known")])
    assert matched["status"] == "matched"
    assert not matched["discovery_claim"]
    position_only = cross_reference_observation(item, [{"catalog_id": "known:1", "ra_deg": 10.0, "dec_deg": 20.0}])
    assert position_only["status"] == "position_only_review"


def test_novelty_requires_reference_coverage_and_quality():
    item = observation()
    assert cross_reference_observation(item, [])["status"] == "insufficient_reference_coverage"
    orthogonal = [0.0] * VECTOR_SIZE
    orthogonal[0] = 1.0
    result = cross_reference_observation(item, [reference(orthogonal, catalog_id="far", ra_deg=30, dec_deg=40)],
                                         anomaly_threshold=0.2)
    assert result["status"] == "novelty_review_candidate"
    assert "domain-expert review" in result["required_follow_up"][-1]


def test_mantis_row_and_atomic_idempotent_batch(tmp_path):
    item = observation()
    known = reference(represent_signal(item).vector)
    result = cross_reference_observation(item, [known])
    row = mantis_signal_row(item, result)
    assert len(json.loads(row["representation"])) == VECTOR_SIZE
    first = commit_signal_batch(tmp_path, [item, item], [known])
    second = commit_signal_batch(tmp_path, [item], [known])
    assert first["changed"] and not second["changed"]
    csv_path = tmp_path / "signal-snapshots" / first["fingerprint"] / "mantis-signals.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["triage_status"] == "matched"
    conflict = observation(values=(np.linspace(0, 1, 64) ** 2).tolist())
    with pytest.raises(ValueError, match="conflicting duplicate"):
        commit_signal_batch(tmp_path, [item, conflict], [known])
