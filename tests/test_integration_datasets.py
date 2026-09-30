"""Dataset creation through the batch engine and per target, and the exported match probabilities.

Offline: the batch path replays the recorded TAP-upload answers of the batch canary fixture
(tests/fixtures/batch/canary), the per-target path the recorded cone searches of the same four
canary targets (tests/fixtures/<target>/), so both modes see the same archive rows.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import respx
from astropy.table import Table
from fixture_io import TARGETS, load_exchanges, replay_side_effect
from helpers import make_service, offline_client
from test_batch_fixtures import CANARY_RADIUS, batch_replay_side_effect, load_batch_exchanges

import batch
import datasets
from datasets import EXPORT_COLUMNS, DatasetEngine, DatasetWriter

CATALOGS = ["gaia_dr3", "simbad"]
TARGET_LIST = [{"ra": ra, "dec": dec} for ra, dec in TARGETS.values()]
PROBABILITY_COLUMNS = {"match_probability", "target_probability", "group_match_probability", "p_any",
                       "contains_target", "match_flag", "group_id", "target_index"}


async def _create(tmp_path: Path, *, min_batch_targets: int, handler: Any, fmt: str = "parquet",
                  targets: list[dict[str, Any]] | None = None, radius: float = CANARY_RADIUS,
                  catalogs: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=handler)
        async with offline_client() as client:
            engine = DatasetEngine(storage_path=str(tmp_path), service=make_service(client),
                                   min_batch_targets=min_batch_targets)
            return await engine.create_dataset(name="canary", profile="full", radius_arcsec=radius,
                                               catalogs=catalogs or CATALOGS, targets=targets or TARGET_LIST,
                                               output_format=fmt, **kwargs)


def _cone_handler():
    exchanges = [e for key in TARGETS for e in load_exchanges(key, CATALOGS)]
    return replay_side_effect(exchanges)


def _rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    return pq.read_table(meta["export_path"]).to_pylist()


def _target_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str, str], float]:
    return {(r["target_index"], r["catalog"], r["source_id"]): r["target_probability"]
            for r in rows if r["contains_target"]}


@pytest.fixture(scope="module")
def both_modes(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_dir, single_dir = tmp_path_factory.mktemp("batch"), tmp_path_factory.mktemp("single")
    uploads = batch_replay_side_effect(load_batch_exchanges("canary", CATALOGS))
    batch_meta = asyncio.run(_create(batch_dir, min_batch_targets=2, handler=uploads))
    single_meta = asyncio.run(_create(single_dir, min_batch_targets=0, handler=_cone_handler()))
    return batch_meta, single_meta


def test_long_target_lists_use_the_batch_engine(both_modes) -> None:
    batch_meta, single_meta = both_modes
    assert batch_meta["method"] == "batch" and batch_meta["status"] == "completed"
    info = batch_meta["batch"]
    assert info["strategies"] == {"gaia_dr3": "xmatch", "simbad": "upload"}  # CDS XMatch, SIMBAD TAP upload
    assert info["fallback_targets"] == [] and "error" not in info
    # One XMatch / upload request per catalog for four targets (per target: one cone per target and catalog).
    assert info["request_count"] == 2
    assert single_meta["method"] == "per-target" and single_meta["batch"] is None
    assert batch_meta["targets"] == single_meta["targets"] == 4


def test_batch_and_per_target_datasets_find_the_same_counterparts(both_modes) -> None:
    batch_meta, single_meta = both_modes
    batch_rows, single_rows = _rows(batch_meta), _rows(single_meta)
    assert batch_rows and single_rows
    by_batch, by_single = _target_rows(batch_rows), _target_rows(single_rows)
    # Every target's counterparts (the rows of its target group) are the same rows in both modes ...
    assert set(by_batch) == set(by_single)
    assert {index for index, _, _ in by_batch} == {0, 1, 2, 3}
    # ... with the same Bayesian posterior (the batch engine runs CrossmatchService.finalize per target).
    for key, probability in by_single.items():
        assert by_batch[key] == pytest.approx(probability, abs=1e-6), key
    # 3C 273's SIMBAD identity is a confident counterpart.
    assert max(p for (i, c, s), p in by_batch.items() if i == 0 and c == "simbad") > 0.9


def test_exports_carry_the_association_probabilities(both_modes) -> None:
    batch_meta, _ = both_modes
    table = pq.read_table(batch_meta["export_path"])
    assert list(table.schema.names) == list(EXPORT_COLUMNS) == batch_meta["columns"]
    assert PROBABILITY_COLUMNS <= set(table.schema.names)
    types = {field.name: str(field.type) for field in table.schema}
    assert types["match_probability"] == types["target_probability"] == "double"
    assert types["target_index"] == "int64" and types["contains_target"] == "bool"
    for row in table.to_pylist():
        assert 0.0 <= row["confidence"] <= 1.0 and row["confidence"] == row["target_probability"]
        # Probabilities the association does not define (a lone field row's membership, p_any of a
        # group without the target) are null, never a made-up number.
        for name in ("match_probability", "group_match_probability", "p_any"):
            assert row[name] is None or 0.0 <= row[name] <= 1.0, (name, row)
        assert row["group_id"].startswith("object-") and row["target_index"] in range(4)
        assert (row["target_ra"], row["target_dec"]) == (TARGET_LIST[row["target_index"]]["ra"],
                                                         TARGET_LIST[row["target_index"]]["dec"])
        if row["contains_target"]:
            assert row["match_flag"] in {"best", "secondary"}
            assert row["match_probability"] is not None and row["p_any"] is not None


def test_min_confidence_and_count_threshold_apply_in_batch_mode(tmp_path: Path) -> None:
    uploads = batch_replay_side_effect(load_batch_exchanges("canary", CATALOGS))
    meta = asyncio.run(_create(tmp_path, min_batch_targets=2, handler=uploads, fmt="csv", min_confidence=0.5,
                               count_threshold=2))
    assert meta["method"] == "batch"
    with open(meta["export_path"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and list(rows[0]) == list(EXPORT_COLUMNS)
    assert all(float(r["confidence"]) >= 0.5 for r in rows)
    groups: dict[tuple[str, str], int] = {}
    for r in rows:
        groups[(r["target_index"], r["group_id"])] = groups.get((r["target_index"], r["group_id"]), 0) + 1
    assert min(groups.values()) >= 2


def test_batch_limits_fall_back_to_per_target_searches(tmp_path: Path) -> None:
    # A radius beyond the CDS XMatch limit (180"): the batch cannot run, every target is searched alone.
    wide = batch.MAX_RADIUS_ARCSEC + 1.0
    calls: list[str] = []

    def refuse(request):
        calls.append(str(request.url))
        raise AssertionError("no batch request expected")

    engine_meta = asyncio.run(_run_with_fake_service(tmp_path, radius=wide, handler=refuse))
    assert engine_meta["method"] == "per-target"
    assert "exceeds the batch limit" in engine_meta["batch"]["error"]
    assert calls == []


async def _run_with_fake_service(tmp_path: Path, *, radius: float, handler: Any) -> dict[str, Any]:
    service = _RecordingService()
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=handler)
        engine = DatasetEngine(storage_path=str(tmp_path), service=service, min_batch_targets=2)  # type: ignore[arg-type]
        meta = await engine.create_dataset(name="wide", profile="full", radius_arcsec=radius, catalogs=CATALOGS,
                                           targets=TARGET_LIST, output_format="json")
    assert service.calls == [(t["ra"], t["dec"]) for t in TARGET_LIST]
    return meta


class _RecordingService:
    """A per-target service returning one group per target (member = the target itself)."""

    def __init__(self) -> None:
        from models import CatalogRegistry

        self.registry = CatalogRegistry()
        self.providers: dict[str, Any] = {}
        self.calls: list[tuple[float, float]] = []

    async def crossmatch(self, ra: float, dec: float, *, query: Any = None, **kwargs: Any):
        from models import UnifiedRecord

        self.calls.append((ra, dec))
        member = {"catalog": "simbad", "source_id": f"single-{len(self.calls)}", "ra": ra, "dec": dec,
                  "confidence": 0.9, "match_probability": 0.9, "target_probability": 0.9, "separation_arcsec": 0.0,
                  "physical": {}, "data": {}, "metadata": {}}
        group = {"group_id": "object-1", "members": [member], "contains_target": True, "match_flag": "best",
                 "match_probability": 0.9, "p_any": 0.95}
        return UnifiedRecord({"ra": ra, "dec": dec}, 1, {}, {}, [], {}, [group])


class _PartialBatchResult:
    """What BatchCrossmatcher.run returns when the queries of target '1' failed."""

    def __init__(self) -> None:
        self.runs = {"simbad": type("Run", (), {"strategy": "upload"})()}
        self.request_count = 1
        self.wall_time_s = 0.01
        self.groups = {i: [{"group_id": "object-1", "contains_target": True, "match_flag": "best",
                            "match_probability": 0.8, "p_any": 0.9,
                            "members": [{"catalog": "simbad", "source_id": f"batch-{i}", "ra": 1.0, "dec": 1.0,
                                         "confidence": 0.8, "match_probability": 0.8, "target_probability": 0.8,
                                         "separation_arcsec": 0.1, "physical": {}, "data": {}, "metadata": {}}]}]
                       for i in (0, 2, 3)}

    def failures_by_id(self) -> dict[str, dict[str, str]]:
        return {"1": {"simbad": "HTTP 503 from the upload service"}}


def test_targets_whose_batch_queries_failed_are_searched_one_by_one(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(self, targets, catalogs, *, radius_arcsec=3.0, strategies=None, nearest_only=False):
        captured.update(targets=targets, catalogs=catalogs, radius=radius_arcsec, keep_groups=self.keep_groups)
        return _PartialBatchResult()

    monkeypatch.setattr(batch.BatchCrossmatcher, "run", fake_run)
    service = _RecordingService()
    engine = DatasetEngine(storage_path=str(tmp_path), service=service, min_batch_targets=2)  # type: ignore[arg-type]
    meta = asyncio.run(engine.create_dataset(name="partial", profile="full", radius_arcsec=5.0,
                                             catalogs=["simbad"], targets=TARGET_LIST, output_format="json"))
    assert captured["keep_groups"] is True and captured["catalogs"] == ["simbad"] and captured["radius"] == 5.0
    assert [t["id"] for t in captured["targets"]] == ["0", "1", "2", "3"]
    assert meta["method"] == "batch+per-target" and meta["batch"]["fallback_targets"] == [1]
    assert service.calls == [(TARGET_LIST[1]["ra"], TARGET_LIST[1]["dec"])]
    rows = json.loads(Path(meta["export_path"]).read_text(encoding="utf-8"))
    # Target order is kept: the per-target result of target 1 sits between the batch results.
    assert [(r["target_index"], r["source_id"]) for r in rows] == [(0, "batch-0"), (1, "single-1"), (2, "batch-2"),
                                                                   (3, "batch-3")]


def test_short_target_lists_stay_per_target(tmp_path: Path, monkeypatch) -> None:
    async def must_not_run(*args, **kwargs):
        raise AssertionError("batch engine used for a short list")

    monkeypatch.setattr(batch.BatchCrossmatcher, "run", must_not_run)
    monkeypatch.setenv("DATASET_BATCH_MIN_TARGETS", "10")
    service = _RecordingService()
    engine = DatasetEngine(storage_path=str(tmp_path), service=service)  # type: ignore[arg-type]
    assert engine.min_batch_targets == 10
    meta = asyncio.run(engine.create_dataset(name="short", profile="full", radius_arcsec=5.0, catalogs=["simbad"],
                                             targets=TARGET_LIST, output_format="csv"))
    assert meta["method"] == "per-target" and len(service.calls) == 4


def test_batch_min_targets_setting(monkeypatch) -> None:
    monkeypatch.delenv("DATASET_BATCH_MIN_TARGETS", raising=False)
    assert datasets.batch_min_targets() == 50
    monkeypatch.setenv("DATASET_BATCH_MIN_TARGETS", "0")
    assert datasets.batch_min_targets() == 0
    monkeypatch.setenv("DATASET_BATCH_MIN_TARGETS", "many")
    assert datasets.batch_min_targets() == 50


@pytest.mark.parametrize("fmt", ["json", "csv", "parquet", "fits"])
def test_writer_exports_every_column_with_missing_values(tmp_path: Path, fmt: str) -> None:
    rows = [
        {"catalog": "gaia_dr3", "source_id": "1", "ra": 10.0, "dec": 20.0, "confidence": 0.97,
         "match_probability": 0.95, "target_probability": 0.97, "group_match_probability": 0.9, "p_any": 0.99,
         "contains_target": True, "match_flag": "best", "group_id": "object-1", "target_index": 0,
         "target_ra": 10.0, "target_dec": 20.0, "physical": {"parallax": 1.2}},
        # A row without association fields (and a non-finite number) must still export.
        {"catalog": "simbad", "source_id": "M 87", "ra": 187.7, "dec": 12.4, "confidence": float("nan")},
    ]
    path = tmp_path / f"out.{fmt}"
    with DatasetWriter(path, fmt) as writer:
        for row in rows:
            writer.write(row)
    assert writer.count == 2
    if fmt == "parquet":
        read = pq.read_table(path).to_pylist()
        assert read[0]["contains_target"] is True and read[0]["target_index"] == 0
        assert read[1]["contains_target"] is None and read[1]["confidence"] is None
    elif fmt == "csv":
        with open(path, newline="", encoding="utf-8") as handle:
            read = list(csv.DictReader(handle))
        assert list(read[0]) == list(EXPORT_COLUMNS)
        assert read[0]["match_probability"] == "0.95" and read[1]["group_id"] == ""
    elif fmt == "fits":
        table = Table.read(path)
        assert list(table.colnames) == list(EXPORT_COLUMNS)
        assert bool(table["contains_target"][0]) is True and int(table["target_index"][1]) == -1
        assert table["match_flag"][0] == "best"
    else:
        read = json.loads(path.read_text(encoding="utf-8"))
        assert read[0]["match_probability"] == 0.95 and read[1]["source_id"] == "M 87"


def test_targets_in_crowded_stellar_systems_are_searched_one_by_one(tmp_path: Path, monkeypatch) -> None:
    """A batch cannot run the density probes a single search makes in crowded fields (47 Tuc = NGC 104):
    such targets are searched per target, the others still go through the batch."""
    captured: dict[str, Any] = {}

    async def fake_run(self, targets, catalogs, *, radius_arcsec=3.0, strategies=None, nearest_only=False):
        captured["ids"] = [t["id"] for t in targets]
        result = _PartialBatchResult()
        template = result.groups[0][0]
        result.groups = {position: [{**template, "members": [{**template["members"][0], "source_id": f"batch-{t['id']}"}]}]
                         for position, t in enumerate(targets)}
        result.failures_by_id = lambda: {}  # type: ignore[method-assign]
        return result

    monkeypatch.setattr(batch.BatchCrossmatcher, "run", fake_run)
    service = _RecordingService()
    targets = [*TARGET_LIST[:2], {"ra": 6.0236, "dec": -72.0813}, *TARGET_LIST[2:]]
    engine = DatasetEngine(storage_path=str(tmp_path), service=service, min_batch_targets=2)  # type: ignore[arg-type]
    meta = asyncio.run(engine.create_dataset(name="tuc", profile="full", radius_arcsec=3.0, catalogs=["simbad"],
                                             targets=targets, output_format="json"))
    assert captured["ids"] == ["0", "1", "3", "4"]
    assert service.calls == [(6.0236, -72.0813)]
    assert meta["method"] == "batch+per-target"
    assert meta["batch"]["crowded_targets"] == 1 and meta["batch"]["crowded_regions"] == ["NGC 104"]
    rows = json.loads(Path(meta["export_path"]).read_text(encoding="utf-8"))
    assert [(r["target_index"], r["source_id"]) for r in rows] == [
        (0, "batch-0"), (1, "batch-1"), (2, "single-1"), (3, "batch-3"), (4, "batch-4")]


def test_crowded_target_screen_agrees_with_astrometry() -> None:
    import random

    from astrometry import crowded_region

    rng = random.Random(7)
    targets = [{"ra": rng.uniform(0, 360), "dec": rng.uniform(-90, 90)} for _ in range(5000)]
    targets += [{"ra": 80.9 + rng.uniform(-3, 3), "dec": -69.7 + rng.uniform(-3, 3)} for _ in range(500)]  # LMC
    targets += [{"ra": 6.0217 + rng.uniform(-0.2, 0.2), "dec": -72.0808 + rng.uniform(-0.2, 0.2)} for _ in range(200)]
    expected = {i: crowded_region(t["ra"], t["dec"]) for i, t in enumerate(targets)}
    assert datasets.crowded_targets(targets, chunk=777) == {i: r for i, r in expected.items() if r is not None}
    assert datasets.crowded_targets([]) == {}
