from __future__ import annotations

import gc
import hashlib
import json
import weakref
from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cpv26.training import kbo_runner
from cpv26.training import kbo_temporal_preflight as preflight
from cpv26.training.kbo_runner import KBOTrainingConfig
from cpv26.training.kbo_temporal_preflight import (
    build_temporal_execution_plan,
    run_adaptive_temporal_cuda_preflight,
    validate_temporal_execution_plan,
)


class _Dataset:
    def __init__(self, directory: Path, days: list[date]) -> None:
        self.directory = directory
        self._days = tuple(days)
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

    def days(self) -> tuple[date, ...]:
        return self._days

    def load_day(self, day: date | str) -> Any:
        raise AssertionError(f"planner must not load graph payloads: {day}")


def _config(batch_days: int = 8) -> KBOTrainingConfig:
    return KBOTrainingConfig(
        device="cuda:0",
        batch_days=batch_days,
        max_pa_per_day=0,
        max_edges_per_route_per_day=0,
        train_seasons=(2023,),
        validation_season=2024,
        test_season=2025,
        chronological=True,
    )


def _write_index(
    directory: Path,
    rows: list[tuple[date, int, int]],
) -> None:
    encoded_rows: list[dict[str, Any]] = []
    for day, nodes, edges in rows:
        encoded_rows.append(
            {
                "day": day.isoformat(),
                "sample_nodes": {"player": nodes, "team": 0, "game": 0},
                "sample_edges": {"event": edges},
                "sample_fingerprint": hashlib.sha256(day.isoformat().encode()).hexdigest(),
            }
        )
    encoded = json.dumps(encoded_rows, sort_keys=True, separators=(",", ":"))
    (directory / "sample_index.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sample_fingerprint_scope": "all_materialized_arrays_v2",
                "dataset_fingerprint": "d" * 64,
                "sampling_policy_fingerprint": "p" * 64,
                "days": encoded_rows,
                "fingerprint": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_plan_uses_only_node_edge_budgets_and_is_identical_for_variants(
    tmp_path: Path,
) -> None:
    days = [date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3), date(2024, 4, 1)]
    _write_index(
        tmp_path,
        [
            (days[0], 60, 20),
            (days[1], 40, 90),
            (days[2], 10, 10),
            (days[3], 25, 25),
        ],
    )
    dataset = _Dataset(tmp_path, days)

    plan = build_temporal_execution_plan(dataset, _config(batch_days=8))
    other_batch_days = build_temporal_execution_plan(dataset, _config(batch_days=1))

    assert [row["dates"] for row in plan["ordered_batches"]] == [
        [days[0].isoformat()],
        [days[1].isoformat(), days[2].isoformat()],
        [days[3].isoformat()],
    ]
    assert plan["fixed_day_count_cap"] is False
    assert plan["plan_fingerprint"] == other_batch_days["plan_fingerprint"]
    assert set(plan["variant_plan_fingerprints"].values()) == {
        plan["plan_fingerprint"]
    }


def test_oversize_single_day_is_retained_and_marked_for_cuda_gate(tmp_path: Path) -> None:
    days = [date(2023, 4, 1), date(2023, 4, 2), date(2024, 4, 1)]
    _write_index(
        tmp_path,
        [(days[0], 30, 30), (days[1], 120, 80), (days[2], 20, 20)],
    )
    plan = build_temporal_execution_plan(_Dataset(tmp_path, days), _config())

    assert plan["oversize_single_day_batches"] == 1
    oversize = [row for row in plan["ordered_batches"] if row["oversize_single_day"]]
    assert len(oversize) == 1
    assert oversize[0]["dates"] == [days[1].isoformat()]
    assert all(row["prefetch_next"] is False for row in plan["ordered_batches"])
    assert sorted(
        value for row in plan["ordered_batches"] for value in row["dates"]
    ) == sorted(day.isoformat() for day in days)


def test_only_cuda_passed_untampered_plan_can_feed_either_variant(tmp_path: Path) -> None:
    days = [date(2023, 4, 1), date(2024, 4, 1)]
    _write_index(tmp_path, [(day, 20, 20) for day in days])
    plan = build_temporal_execution_plan(_Dataset(tmp_path, days), _config())
    report = {
        "status": "passed",
        "selected_for_training": True,
        "execution_plan": plan,
    }

    full = validate_temporal_execution_plan(
        report, dataset_fingerprint="d" * 64, variant="full"
    )
    node = validate_temporal_execution_plan(
        report, dataset_fingerprint="d" * 64, variant="node_only"
    )
    assert full == node

    failed = dict(report, status="failed_memory_threshold", selected_for_training=False)
    with pytest.raises(ValueError, match="did not pass"):
        validate_temporal_execution_plan(
            failed, dataset_fingerprint="d" * 64, variant="full"
        )
    plan["ordered_batches"][0]["dates"] = [date(2023, 4, 2).isoformat()]
    with pytest.raises(ValueError, match="fingerprint"):
        validate_temporal_execution_plan(
            report, dataset_fingerprint="d" * 64, variant="full"
        )


def test_plan_rejects_sampling_caps_or_nonchronological_execution(tmp_path: Path) -> None:
    days = [date(2023, 4, 1), date(2024, 4, 1)]
    _write_index(tmp_path, [(day, 20, 20) for day in days])
    dataset = _Dataset(tmp_path, days)
    with pytest.raises(ValueError, match="chronological"):
        build_temporal_execution_plan(dataset, replace(_config(), chronological=False))
    with pytest.raises(ValueError, match="uncapped"):
        build_temporal_execution_plan(dataset, replace(_config(), max_pa_per_day=1))


def test_adaptive_gate_halves_budgets_and_selects_first_complete_safe_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = [date(2023, 4, day) for day in range(1, 5)] + [date(2024, 4, 1)]
    _write_index(tmp_path, [(day, 30, 30) for day in days])
    dataset = _Dataset(tmp_path, days)
    attempted: list[tuple[int, int]] = []
    failed_exception_markers: list[weakref.ReferenceType[Any]] = []

    class _ExceptionMarker:
        pass

    def fake_candidate(
        selected: Any,
        config: KBOTrainingConfig,
        *,
        output: Path,
        max_nodes: int,
        max_edges: int,
        **_: Any,
    ) -> dict[str, Any]:
        if failed_exception_markers:
            gc.collect()
            assert failed_exception_markers[0]() is None
        attempted.append((max_nodes, max_edges))
        plan = build_temporal_execution_plan(
            selected, config, max_nodes=max_nodes, max_edges=max_edges
        )
        if max_nodes > 50:
            report = {
                "status": "failed_memory_threshold",
                "selected_for_training": False,
                "execution_plan": plan,
                "failed_batch": {
                    **plan["ordered_batches"][0],
                    "peak_reserved_fraction": 0.91,
                },
            }
            kbo_runner._atomic_json(output, report)
            marker = _ExceptionMarker()
            failed_exception_markers.append(weakref.ref(marker))
            error = RuntimeError("synthetic threshold failure")
            error.marker = marker  # type: ignore[attr-defined]
            raise error
        report = {
            "status": "passed",
            "selected_for_training": True,
            "execution_plan": plan,
            "peak_reserved_fraction": 0.72,
            "all_actual_batches_measured": True,
        }
        kbo_runner._atomic_json(output, report)
        return report

    monkeypatch.setattr(preflight, "run_temporal_cuda_preflight", fake_candidate)
    output = tmp_path / "adaptive.json"
    result = run_adaptive_temporal_cuda_preflight(
        dataset, _config(), output=output, max_nodes=100, max_edges=100
    )

    assert attempted == [(100, 100), (50, 50)]
    assert result["status"] == "passed" and result["selected_attempt"] == 2
    assert result["execution_plan"]["budgets"] == {"max_nodes": 50, "max_edges": 50}
    assert len(result["attempts"]) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob(".*.attempt-*.json"))


def test_compute_cuda_oom_is_audited_retryable_and_closes_device_iterator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = [date(2023, 4, 1), date(2024, 4, 1)]
    _write_index(tmp_path, [(day, 20, 20) for day in days])
    dataset = _Dataset(tmp_path, days)
    device = SimpleNamespace(type="cuda")
    cleanup = {"empty": 0, "reset": 0, "sync": 0}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            cleanup["empty"] += 1

        @staticmethod
        def reset_peak_memory_stats(_device: Any) -> None:
            cleanup["reset"] += 1

        @staticmethod
        def synchronize(_device: Any) -> None:
            cleanup["sync"] += 1

    class _Scaler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _Torch:
        float16 = object()
        cuda = _Cuda()
        amp = SimpleNamespace(GradScaler=_Scaler)

        @staticmethod
        def manual_seed(_seed: int) -> None:
            return None

        @staticmethod
        def autocast(*_args: Any, **_kwargs: Any) -> Any:
            return nullcontext()

    class _Model:
        def __init__(self, _config: Any) -> None:
            pass

        def to(self, _device: Any) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, _batch: Any) -> Any:
            raise RuntimeError("CUDA out of memory during synthetic forward")

    class _Optimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

    class _TrackedIterator:
        def __init__(self) -> None:
            self.closed = False
            self.produced = False

        def __iter__(self) -> _TrackedIterator:
            return self

        def __next__(self) -> dict[str, Any]:
            if self.produced:
                raise StopIteration
            self.produced = True
            return {
                "node_features": {"player": SimpleNamespace(shape=(20, 1))},
                "routes": [SimpleNamespace(num_edges=20)],
            }

        def close(self) -> None:
            self.closed = True

    tracked = _TrackedIterator()
    fake_torch = _Torch()
    monkeypatch.setattr(
        kbo_runner,
        "_device_and_precision",
        lambda *_args, **_kwargs: (device, None, {"device": "cuda:0"}),
    )
    monkeypatch.setattr(preflight, "require_torch", lambda: (fake_torch, None))
    monkeypatch.setattr(kbo_runner, "_model_config", lambda *_args: object())
    monkeypatch.setattr(preflight, "KBORelGNNModel", _Model)
    monkeypatch.setattr(preflight, "make_adamw", lambda *_args, **_kwargs: _Optimizer())
    monkeypatch.setattr(preflight, "_planned_device_batches", lambda *_args: tracked)
    output = tmp_path / "compute-oom.json"

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        preflight.run_temporal_cuda_preflight(dataset, _config(), output=output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed_cuda_oom"
    assert saved["failure_stage"] == "forward"
    assert saved["failed_batch"]["dates"] == [days[0].isoformat()]
    assert preflight._retryable_memory_failure(saved) == (True, False)
    assert tracked.closed is True
    assert cleanup["empty"] == 2
    assert cleanup["reset"] == 2
    assert cleanup["sync"] == 1


def test_compute_cuda_oom_on_oversize_singleton_is_hard_failure() -> None:
    report = {
        "status": "failed_cuda_oom",
        "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"},
        "failed_cuda_window": [{"oversize_single_day": True}],
    }

    assert preflight._retryable_memory_failure(report) == (True, True)


def test_adaptive_gate_hard_fails_isolated_oversize_day_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = [date(2023, 4, 1), date(2024, 4, 1)]
    _write_index(tmp_path, [(days[0], 120, 80), (days[1], 20, 20)])
    dataset = _Dataset(tmp_path, days)
    attempts = 0

    def fake_candidate(
        selected: Any,
        config: KBOTrainingConfig,
        *,
        output: Path,
        max_nodes: int,
        max_edges: int,
        **_: Any,
    ) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        plan = build_temporal_execution_plan(
            selected, config, max_nodes=max_nodes, max_edges=max_edges
        )
        failed = next(row for row in plan["ordered_batches"] if row["oversize_single_day"])
        report = {
            "status": "failed_memory_threshold",
            "selected_for_training": False,
            "execution_plan": plan,
            "failed_batch": {**failed, "peak_reserved_fraction": 0.9},
        }
        kbo_runner._atomic_json(output, report)
        raise RuntimeError("synthetic oversize threshold failure")

    monkeypatch.setattr(preflight, "run_temporal_cuda_preflight", fake_candidate)
    output = tmp_path / "adaptive.json"
    with pytest.raises(RuntimeError, match="isolated single-day graph"):
        run_adaptive_temporal_cuda_preflight(
            dataset, _config(), output=output, max_nodes=100, max_edges=100
        )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert attempts == 1
    assert saved["status"] == "failed_oversize_single_day"
    assert saved["selected_for_training"] is False
