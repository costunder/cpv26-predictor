from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cpv26.data.kbo_graph_dataset import GraphDay
from cpv26.data.kbo_temporal_archive import _sample_fingerprint
from cpv26.training import kbo_runner
from cpv26.training.kbo_runner import KBOTrainingConfig
from cpv26.training.kbo_temporal_preflight import (
    TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
    build_temporal_execution_plan,
)


class _Dataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = {
            "dataset_version": 7,
            "graph_schema": "temporal_v7",
            "fingerprint": "d" * 64,
            "sampling_policy_fingerprint": "p" * 64,
            "temporal_batching": {
                "max_nodes_per_batch": 100,
                "max_edges_per_batch": 100,
                "max_days_per_batch": 8,
            },
        }
        self._days = (
            date(2023, 4, 1),
            date(2023, 4, 2),
            date(2024, 4, 1),
            date(2025, 4, 1),
        )

    def days(self) -> tuple[date, ...]:
        return self._days

    def load_day(self, day: date | str) -> Any:
        raise AssertionError(f"test planning fixture must not materialize {day}")


def _config() -> KBOTrainingConfig:
    return KBOTrainingConfig(
        device="cuda:0",
        batch_days=8,
        max_pa_per_day=0,
        max_edges_per_route_per_day=0,
        train_seasons=(2023,),
        validation_season=2024,
        test_season=2025,
        chronological=True,
        activation_checkpointing=True,
    )


def _runtime() -> dict[str, Any]:
    return {
        "device": "cuda:0",
        "gpu_name": "test GPU",
        "total_memory_bytes": 10 * 2**30,
        "compute_capability": [8, 0],
        "torch_version": "2.test",
        "cuda_runtime": "12.test",
        "precision": "bf16",
    }


def _write_index(directory: Path, days: tuple[date, ...]) -> None:
    rows = []
    for selected in days[:-1]:
        rows.append(
            {
                "day": selected.isoformat(),
                "sample_nodes": {"player": 10, "team": 2, "game": 3},
                "sample_edges": {"event": 20},
                "sample_fingerprint": hashlib.sha256(
                    selected.isoformat().encode()
                ).hexdigest(),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    (directory / "sample_index.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sample_fingerprint_scope": "all_materialized_arrays_v2",
                "dataset_fingerprint": "d" * 64,
                "sampling_policy_fingerprint": "p" * 64,
                "days": rows,
                "fingerprint": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def _write_passed_report(path: Path, dataset: _Dataset, config: KBOTrainingConfig) -> None:
    plan = build_temporal_execution_plan(dataset, config)
    total_memory = int(_runtime()["total_memory_bytes"])
    measurements = []
    for index, row in enumerate(plan["ordered_batches"]):
        reserved = (3 * 2**30) + index
        prefetched_next = None
        if row["prefetch_next"]:
            following = plan["ordered_batches"][index + 1]
            prefetched_next = {
                "split": following["split"],
                "dates": following["dates"],
                "nodes": following["nodes"],
                "edges": following["edges"],
                "oversize_single_day": following["oversize_single_day"],
            }
        measurements.append(
            {
                "actual_batch_index": index,
                "split": row["split"],
                "split_batch_index": row["split_batch_index"],
                "dates": row["dates"],
                "nodes": row["nodes"],
                "edges": row["edges"],
                "oversize_single_day": row["oversize_single_day"],
                "peak_allocated_bytes": 2 * 2**30,
                "peak_reserved_bytes": reserved,
                "total_memory_bytes": total_memory,
                "peak_reserved_fraction": reserved / total_memory,
                "prefetch_depth": 1,
                "prefetched_next_batch": prefetched_next,
            }
        )
    peak = max(row["peak_reserved_fraction"] for row in measurements)
    loader_autotune = {
        "status": "measured",
        "selected": {
            **plan["loader_runtime"],
            "graph_days_per_second": 10.0,
            "host_memory_safe": True,
            "host_memory_safety": {"status": "passed"},
        },
    }
    resources = {
        "host_inventory": {},
        "interval": {},
        "input_tensor_shapes_first_actual_batch": {},
        "stage_seconds": {},
        "throughput": {},
        "steady_cuda_allocated_bytes": {},
        "steady_cuda_reserved_bytes": {},
        "peak_cuda_allocated_bytes": 2 * 2**30,
        "peak_cuda_reserved_bytes": 3 * 2**30,
        "physical_batching": plan["physical_batching"],
    }
    candidate = {
        "protocol": "temporal_v7_cuda_budget_plan",
        "protocol_version": TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
        "status": "passed",
        "selected_for_training": True,
        "max_reserved_fraction": 0.85,
        "configuration": asdict(config),
        "runtime": _runtime(),
        "all_actual_batches_measured": True,
        "completed_actual_batch_count": len(measurements),
        "peak_reserved_fraction": peak,
        "measurements": measurements,
        "execution_plan": plan,
        "loader_autotune": loader_autotune,
        "resource_measurements": resources,
    }
    kbo_runner._atomic_json(
        path,
        {
            "protocol": "temporal_v7_cuda_budget_plan_adaptive",
            "protocol_version": TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
            "status": "passed",
            "selected_for_training": True,
            "max_reserved_fraction": 0.85,
            "selected_attempt": 1,
            "attempts": [candidate],
            "execution_plan": plan,
            "peak_reserved_fraction": peak,
            "all_actual_batches_measured": True,
        },
    )


def test_passed_plan_binds_both_variants_to_identical_batches(tmp_path: Path) -> None:
    dataset = _Dataset(tmp_path)
    config = _config()
    _write_index(tmp_path, dataset.days())
    report = tmp_path / "preflight.json"
    _write_passed_report(report, dataset, config)

    full = kbo_runner._load_temporal_execution(
        dataset, config, report, runtime=_runtime()
    )
    node = kbo_runner._load_temporal_execution(
        dataset,
        replace(config, route_schedule="node_only"),
        report,
        runtime=_runtime(),
    )

    assert full.plan_fingerprint == node.plan_fingerprint
    assert full.report_sha256 == node.report_sha256
    assert [row["dates"] for row in full.rows] == [row["dates"] for row in node.rows]
    assert all("2025" not in day for row in full.rows for day in row["dates"])


def test_plan_runtime_or_topology_tampering_is_rejected(tmp_path: Path) -> None:
    dataset = _Dataset(tmp_path)
    config = _config()
    _write_index(tmp_path, dataset.days())
    report = tmp_path / "preflight.json"
    _write_passed_report(report, dataset, config)
    document = json.loads(report.read_text(encoding="utf-8"))
    document["execution_plan"]["ordered_batches"][0]["dates"][0] = "2025-04-01"
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        kbo_runner._load_temporal_execution(
            dataset, config, report, runtime=_runtime()
        )


def test_selected_plan_must_equal_the_measured_attempt(tmp_path: Path) -> None:
    dataset = _Dataset(tmp_path)
    config = _config()
    _write_index(tmp_path, dataset.days())
    report = tmp_path / "preflight.json"
    _write_passed_report(report, dataset, config)
    document = json.loads(report.read_text(encoding="utf-8"))
    plan = document["execution_plan"]
    plan["budgets"]["max_nodes"] = 99
    core = {
        key: plan[key]
        for key in (
            "dataset_fingerprint",
            "sampling_policy_fingerprint",
            "budgets",
            "batching_basis",
            "fixed_day_count_cap",
            "prefetch_depth",
            "loader_runtime",
            "physical_batching",
            "ordered_batches",
        )
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":"))
    plan["plan_fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    plan["variant_plan_fingerprints"] = {
        "full": plan["plan_fingerprint"],
        "node_only": plan["plan_fingerprint"],
    }
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its measured attempt"):
        kbo_runner._load_temporal_execution(
            dataset, config, report, runtime=_runtime()
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("actual_batch_index", 9, "measurement differs"),
        ("prefetch_depth", 2, "measurement differs"),
        ("peak_reserved_bytes", 9 * 2**30, "byte/fraction evidence"),
    ),
)
def test_selected_measurement_evidence_is_strictly_validated(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    dataset = _Dataset(tmp_path)
    config = _config()
    _write_index(tmp_path, dataset.days())
    report = tmp_path / "preflight.json"
    _write_passed_report(report, dataset, config)
    document = json.loads(report.read_text(encoding="utf-8"))
    document["attempts"][0]["measurements"][0][field] = value
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        kbo_runner._load_temporal_execution(
            dataset, config, report, runtime=_runtime()
        )


def test_selected_peak_and_limit_must_match_measurements(tmp_path: Path) -> None:
    dataset = _Dataset(tmp_path)
    config = _config()
    _write_index(tmp_path, dataset.days())
    report = tmp_path / "preflight.json"
    _write_passed_report(report, dataset, config)
    document = json.loads(report.read_text(encoding="utf-8"))
    document["peak_reserved_fraction"] = 0.84
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive-report peak differs"):
        kbo_runner._load_temporal_execution(
            dataset, config, report, runtime=_runtime()
        )


def test_prefetch_segments_follow_selected_plan_barriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        {
            "day_ids": [f"2023-04-0{index}"],
            "node_features": {"player": np.zeros((2, 1), dtype=np.float32)},
            "routes": [SimpleNamespace(num_edges=3)],
        }
        for index in range(1, 5)
    ]
    rows = [
        {
            "dates": batch["day_ids"],
            "nodes": 2,
            "edges": 3,
            "prefetch_next": value,
        }
        for batch, value in zip(batches, (True, False, False, False), strict=True)
    ]
    events: list[tuple[str, int]] = []

    def fake_prefetch(source: Any, *_args: Any, **_kwargs: Any) -> Any:
        values = list(source)
        events.append(("prefetch", len(values)))
        yield from values

    class _Stream:
        def synchronize(self) -> None:
            events.append(("barrier", 0))

    class _Cuda:
        @staticmethod
        def current_stream(_device: Any) -> _Stream:
            return _Stream()

    fake_torch = SimpleNamespace(cuda=_Cuda())

    monkeypatch.setattr(kbo_runner, "prefetch_batches", fake_prefetch)
    monkeypatch.setattr(kbo_runner, "require_torch", lambda: (fake_torch, None))
    observed = list(kbo_runner._planned_device_batches(batches, rows, "cuda:0"))

    assert observed == batches
    assert events == [
        ("prefetch", 2),
        ("barrier", 0),
        ("prefetch", 1),
        ("barrier", 0),
        ("prefetch", 1),
    ]


def test_day_dataset_verifies_each_materialized_sample_before_collate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_day = date(2023, 4, 1)
    graph = GraphDay(
        day=selected_day,
        player_ids=("p1",),
        team_ids=("t1",),
        game_ids=("g1",),
        arrays={"player_features": np.asarray([[1.0]], dtype=np.float32)},
    )

    class _Loaded:
        manifest: dict[str, Any] = {}

        @staticmethod
        def days() -> tuple[date, ...]:
            return (selected_day,)

        @staticmethod
        def load_day(_day: date | str) -> GraphDay:
            return graph

    open_calls: list[dict[str, Any]] = []

    def open_dataset(_path: Any, **kwargs: Any) -> _Loaded:
        open_calls.append(kwargs)
        return _Loaded()

    monkeypatch.setattr(kbo_runner, "open_kbo_graph_dataset", open_dataset)
    expected = _sample_fingerprint(graph)
    verified = kbo_runner._DayDataset(
        tmp_path,
        (selected_day,),
        expected_sample_fingerprints={selected_day: expected},
        label_year_ceiling=2024,
    )
    assert verified[0] is graph
    assert open_calls == [{"label_year_ceiling": 2024}]

    rejected = kbo_runner._DayDataset(
        tmp_path,
        (selected_day,),
        expected_sample_fingerprints={selected_day: "0" * 64},
    )
    with pytest.raises(RuntimeError, match="differs from the selected CUDA plan"):
        rejected[0]
