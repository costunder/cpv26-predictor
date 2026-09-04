from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import cpv26.training.kbo_runner as runner_module
import cpv26.training.kbo_scale_preflight as preflight
from cpv26.data.kbo_graph_dataset import GraphDay, KBOGraphDataset
from cpv26.training.kbo_runner import KBOTrainingConfig
from cpv26.training.kbo_scale_preflight import (
    _enforce_reserved_fraction,
    audit_kbo_scale_workload,
)


def _day(
    value: str,
    *,
    players: int = 2,
    teams: int = 2,
    edges: int = 0,
    matches: int = 1,
    live_hit: int = 0,
    pa: int = 0,
    box_pa: int = 0,
    box_pitch: int = 0,
) -> GraphDay:
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "player_features": np.zeros((players, 2), dtype=np.float32),
        "team_features": np.zeros((teams, 2), dtype=np.float32),
        "player_batting_features": np.zeros((players, 1), dtype=np.float32),
        "player_pitching_features": np.zeros((players, 1), dtype=np.float32),
        "match_home_team_index": np.zeros(matches, dtype=np.int64),
        "live_hit_player_index": np.zeros(live_hit, dtype=np.int64),
        "pa_batter_index": np.zeros(pa, dtype=np.int64),
        "box_pa_player_index": np.zeros(box_pa, dtype=np.int64),
        "box_pitch_player_index": np.zeros(box_pitch, dtype=np.int64),
    }
    route = "batter_pa_pitcher"
    arrays[f"{route}__source_index"] = np.zeros(edges, dtype=np.int64)
    arrays[f"{route}__destination_index"] = np.zeros(edges, dtype=np.int64)
    arrays[f"{route}__event_features"] = np.zeros((edges, 1), dtype=np.float32)
    arrays[f"{route}__event_age_seconds"] = np.zeros(edges, dtype=np.float32)
    arrays[f"{route}__publication_delay_seconds"] = np.zeros(edges, dtype=np.float32)
    arrays[f"{route}__weights"] = np.ones(edges, dtype=np.float32)
    return GraphDay(
        date.fromisoformat(value),
        tuple(f"p{index}" for index in range(players)),
        tuple(f"t{index}" for index in range(teams)),
        arrays,
    )


class _SpyDataset(KBOGraphDataset):
    def __init__(self, graphs: dict[date, GraphDay]) -> None:
        self.directory = Path("/in-memory/kbo")
        self.manifest = {"fingerprint": "fixture", "days": []}
        self._graphs = graphs
        self.loaded: list[date] = []

    def days(self) -> tuple[date, ...]:
        return tuple(sorted(self._graphs))

    def load_day(self, day: date | str) -> GraphDay:
        selected = date.fromisoformat(day) if isinstance(day, str) else day
        self.loaded.append(selected)
        if selected.year == 2025:
            raise AssertionError("held-out test GraphDay was loaded")
        return self._graphs[selected]


def _config(**overrides: Any) -> KBOTrainingConfig:
    values: dict[str, Any] = {
        "device": "cpu",
        "amp": "off",
        "workers": 0,
        "batch_days": 2,
        "hidden_dim": 8,
        "layers": 2,
        "heads": 2,
        "max_pa_per_day": 0,
        "max_edges_per_route_per_day": 0,
        "train_seasons": (2023,),
        "validation_season": 2024,
        "test_season": 2025,
        "chronological": True,
    }
    values.update(overrides)
    return KBOTrainingConfig(**values)


def test_audit_selects_an_actual_non_overlapping_chronological_batch() -> None:
    graphs = {
        date(2023, 4, 1): _day("2023-04-01", edges=1),
        date(2023, 4, 2): _day("2023-04-02", edges=10),
        date(2023, 4, 3): _day("2023-04-03", edges=9),
        date(2023, 4, 4): _day("2023-04-04", edges=1),
        date(2024, 4, 1): _day("2024-04-01", edges=1),
        date(2025, 4, 1): _day("2025-04-01", edges=100_000),
    }
    dataset = _SpyDataset(graphs)

    report = audit_kbo_scale_workload(dataset, _config())

    # The sliding pair 04-02/04-03 is larger, but it is not an actual batch in
    # the sorted loader.  Preflight must select from loader-aligned chunks.
    assert report["splits"]["train"]["worst_batch"]["dates"] == [
        "2023-04-01",
        "2023-04-02",
    ]
    assert report["splits"]["train"]["chronological_batches"] == 2
    assert [batch["dates"] for batch in report["splits"]["train"]["batches"]] == [
        ["2023-04-01", "2023-04-02"],
        ["2023-04-03", "2023-04-04"],
    ]
    assert report["batching"] == {
        "train": "chronological_non_overlapping",
        "validation": "chronological_non_overlapping",
    }
    json.dumps(report, allow_nan=False)


def test_audit_applies_real_collate_caps_and_never_loads_test_graph() -> None:
    graphs = {
        date(2023, 4, 1): _day("2023-04-01", edges=9, pa=7),
        date(2024, 4, 1): _day("2024-04-01", edges=9, pa=7),
        date(2025, 4, 1): _day("2025-04-01", edges=999, pa=999),
    }
    dataset = _SpyDataset(graphs)
    report = audit_kbo_scale_workload(
        dataset,
        _config(
            batch_days=1,
            max_pa_per_day=2,
            max_edges_per_route_per_day=3,
        ),
    )

    train = report["splits"]["train"]["days"][0]
    validation = report["splits"]["validation"]["days"][0]
    assert train["route_edges"] == 9
    assert train["effective_route_edges"] == 3
    assert train["query_counts"]["pa"] == 7
    assert train["effective_query_counts"]["pa"] == 2
    assert validation["effective_query_counts"]["pa"] == 7
    assert dataset.loaded == [date(2023, 4, 1), date(2024, 4, 1)]
    assert report["splits"]["test"] == {
        "season": 2025,
        "dates_listed_from_manifest": 1,
        "date_start": "2025-04-01",
        "date_end": "2025-04-01",
        "graph_days_loaded": False,
        "labels_loaded": False,
        "sealed": True,
    }


def test_shuffled_audit_uses_explicit_conservative_top_k_upper_bound() -> None:
    graphs = {
        date(2023, 4, 1): _day("2023-04-01", edges=1),
        date(2023, 4, 2): _day("2023-04-02", edges=20),
        date(2023, 4, 3): _day("2023-04-03", edges=2),
        date(2023, 4, 4): _day("2023-04-04", edges=19),
        date(2024, 4, 1): _day("2024-04-01"),
        date(2025, 4, 1): _day("2025-04-01"),
    }

    report = audit_kbo_scale_workload(_SpyDataset(graphs), _config(chronological=False))

    worst = report["splits"]["train"]["worst_batch"]
    assert worst["dates"] == ["2023-04-02", "2023-04-04"]
    assert worst["selection_kind"] == "conservative_top_k_day_upper_bound"
    assert report["batching"] == {
        "train": "conservative_top_k_upper_bound_for_shuffled_loader",
        "validation": "chronological_non_overlapping",
    }
    assert (
        report["splits"]["validation"]["worst_batch"]["selection_kind"]
        == "chronological_non_overlapping_batch"
    )


def test_audit_dedupes_component_maxima_without_dropping_query_shapes() -> None:
    graphs = {
        date(2023, 4, 1): _day("2023-04-01", edges=50),
        date(2023, 4, 2): _day("2023-04-02", players=100),
        date(2023, 4, 3): _day("2023-04-03", matches=100),
        date(2023, 4, 4): _day("2023-04-04", live_hit=100),
        date(2023, 4, 5): _day("2023-04-05", pa=100),
        date(2023, 4, 6): _day("2023-04-06", box_pa=100),
        date(2023, 4, 7): _day("2023-04-07", box_pitch=100),
        date(2024, 4, 1): _day("2024-04-01"),
        date(2025, 4, 1): _day("2025-04-01"),
    }

    report = audit_kbo_scale_workload(_SpyDataset(graphs), _config(batch_days=1))

    selected = {
        dimension: candidate["dates"]
        for candidate in report["candidate_batches"]
        if candidate["split"] == "train"
        for dimension in candidate["selection_dimensions"]
    }
    assert selected["route_edges"] == ["2023-04-01"]
    assert selected["nodes"] == ["2023-04-02"]
    assert selected["query:match"] == ["2023-04-03"]
    assert selected["query:live_hit"] == ["2023-04-04"]
    assert selected["query:pa"] == ["2023-04-05"]
    assert selected["query:box_pa"] == ["2023-04-06"]
    assert selected["query:box_pitch"] == ["2023-04-07"]
    assert report["candidate_batch_count"] <= len(
        report["candidate_selection_dimensions"]
    ) * 2


def _measured_batch(
    phase: str,
    index: int,
    batch: dict[str, Any],
    *,
    peak: int,
) -> dict[str, Any]:
    state = {
        "parameter_entries": 2,
        "tensor_values": 4,
        "total_tensor_bytes": 32,
        "cuda_tensor_bytes": 32,
    }
    return {
        "status": "completed",
        "phase": phase,
        "actual_batch_index": index,
        "split": batch["split"],
        "split_batch_index": batch["split_batch_index"],
        "dates": batch["dates"],
        "diagnostic_selection_dimensions": batch["diagnostic_selection_dimensions"],
        "estimated_work_units": batch["estimated_work_units"],
        "parameter_count": 20,
        "forward_verified": True,
        "backward_verified": True,
        "adamw_step_verified": True,
        "optimizer_state_before": state,
        "optimizer_state_after": state,
        "optimizer_state_entries_after_step": 2,
        "loss": 1.0,
        "cpu_batch_counts": {"days": 1},
        "gpu_batch_counts": {"days": 1},
        "batch_count_transfer_verified": True,
        "allocated_before_bytes": 20,
        "reserved_before_bytes": 30,
        "peak_allocated_bytes": peak - 5,
        "peak_reserved_bytes": peak,
        "total_memory_bytes": 100,
        "free_memory_before_bytes": 70,
        "free_memory_bytes": 100 - peak,
        "peak_reserved_fraction": peak / 100,
        "headroom_bytes": 100 - peak,
    }


def test_run_warms_materializes_then_measures_every_actual_batch_and_saves_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _SpyDataset(
        {
            date(2023, 4, 1): _day("2023-04-01", edges=3),
            date(2024, 4, 1): _day("2024-04-01", live_hit=4),
            date(2025, 4, 1): _day("2025-04-01"),
        }
    )
    config = _config(device="cuda:0", amp="auto", batch_days=1)
    calls: list[tuple[str, int, int, int | None, bool, bool]] = []

    class _Parameter:
        @staticmethod
        def numel() -> int:
            return 10

    class _Model:
        def train(self) -> None:
            pass

        def parameters(self) -> list[_Parameter]:
            return [_Parameter(), _Parameter()]

    class _Cuda:
        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            pass

    class _Torch:
        float16 = object()
        cuda = _Cuda()
        amp = SimpleNamespace(GradScaler=lambda *_args, **_kwargs: object())

        @staticmethod
        def manual_seed(_seed: int) -> None:
            pass

    monkeypatch.setattr(
        runner_module,
        "_device_and_precision",
        lambda *_: (SimpleNamespace(type="cuda"), None, {"device": "cuda:0"}),
    )
    monkeypatch.setattr(runner_module, "_model_config", lambda *_: object())
    monkeypatch.setattr(preflight, "require_torch", lambda: (_Torch(), object()))
    monkeypatch.setattr(preflight, "KBORelGNNModel", lambda _: _Model())

    persistent_optimizer = object()

    def measure(
        _selected: Any,
        batch: dict[str, Any],
        *,
        phase: str,
        actual_batch_index: int,
        model: Any,
        optimizer: Any | None,
        clear_allocator_cache_before: bool = True,
        clear_allocator_cache_after: bool = True,
        **_: Any,
    ) -> tuple[Any, dict[str, Any]]:
        calls.append(
            (
                phase,
                actual_batch_index,
                id(model),
                None if optimizer is None else id(optimizer),
                clear_allocator_cache_before,
                clear_allocator_cache_after,
            )
        )
        resolved = persistent_optimizer if optimizer is None else optimizer
        peak = 90 if phase == "warmup" else 50 + actual_batch_index * 20
        return resolved, _measured_batch(
            phase, actual_batch_index, batch, peak=peak
        )

    monkeypatch.setattr(preflight, "_measure_persistent_batch", measure)
    output = tmp_path / "preflight.json"
    progress_messages: list[str] = []

    report = preflight.run_kbo_scale_preflight(
        dataset,
        config,
        output=output,
        max_reserved_fraction=0.95,
        progress=progress_messages.append,
    )

    assert [(phase, index) for phase, index, *_ in calls] == [
        ("warmup", 0),
        ("materialization", 0),
        ("materialization", 1),
        ("steady_state", 0),
        ("steady_state", 1),
    ]
    assert len({model_id for _, _, model_id, *_ in calls}) == 1
    assert calls[0][3] is None
    assert {optimizer_id for _, _, _, optimizer_id, *_ in calls[1:]} == {
        id(persistent_optimizer)
    }
    assert [(before, after) for *_, before, after in calls] == [
        (True, True),
        (True, True),
        (True, True),
        (True, False),
        (False, False),
    ]
    assert report["planned_actual_batch_count"] == 2
    assert report["completed_materialization_batch_count"] == 2
    assert report["completed_actual_batch_count"] == 2
    assert report["winning_batch"]["phase"] == "warmup"
    assert report["execution"]["peak_reserved_bytes"] == 90
    assert report["execution"]["evaluated_batch_count"] == 2
    assert report["execution"]["materialization_steps"] == 2
    assert report["execution"]["first_batch_repeated_after_warmup"] is True
    assert report["execution"]["all_actual_batches_evaluated"] is True
    assert report["execution"]["steady_state_allocator_cache_cleared_once_before_pass"] is True
    assert report["execution"]["steady_state_allocator_cache_retained_between_batches"] is True
    assert report["execution"]["steady_state_cumulative_peak_reserved_bytes"] == 70
    assert report["execution"]["steady_state_cumulative_peak_reserved_fraction"] == 0.7
    assert [
        row["steady_state_cumulative_peak_reserved_bytes"]
        for row in report["evaluated_batches"]
    ] == [50, 70]
    assert report["evaluated_batches"][0][
        "allocator_cache_retained_from_previous_steady_batch"
    ] is False
    assert report["evaluated_batches"][1][
        "allocator_cache_retained_from_previous_steady_batch"
    ] is True
    assert report["measurement_policy"]["allocator_between_batches"] == (
        "steady_state_cache_retained_after_single_initial_empty_cache"
    )
    assert report["execution"]["final_optimizer_state"]["cuda_tensor_bytes"] == 32
    assert report["status"] == "passed"
    assert any("planned actual batches=2" in message for message in progress_messages)
    assert any("materialization: 2/2" in message for message in progress_messages)
    assert any("steady-state: 2/2" in message for message in progress_messages)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert all(value.year != 2025 for value in dataset.loaded)

    failure_output = tmp_path / "failed.json"

    def fail_measure(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if kwargs["phase"] == "materialization" and kwargs["actual_batch_index"] == 1:
            cause = RuntimeError("CUDA out of memory")
            failed = {
                "status": "failed",
                "phase": "materialization",
                "actual_batch_index": 1,
                "split": "validation",
                "dates": ["2024-04-01"],
                "error": {"type": "RuntimeError", "message": str(cause)},
            }
            raise preflight._BatchMeasurementFailure(failed, cause)
        return measure(*args, **kwargs)

    monkeypatch.setattr(preflight, "_measure_persistent_batch", fail_measure)
    with pytest.raises(RuntimeError, match="actual train/validation batch"):
        preflight.run_kbo_scale_preflight(
            dataset,
            config,
            output=failure_output,
            max_reserved_fraction=0.95,
            progress=lambda _: None,
        )
    failed_report = json.loads(failure_output.read_text(encoding="utf-8"))
    assert failed_report["status"] == "execution_failed"
    assert failed_report["failed_batch"]["error"]["message"] == "CUDA out of memory"
    assert len(failed_report["materialization_batches"]) == 1

    growth_output = tmp_path / "growth.json"

    def grow_during_steady(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        optimizer, measured = measure(*args, **kwargs)
        if kwargs["phase"] == "steady_state" and kwargs["actual_batch_index"] == 1:
            measured["optimizer_state_after"] = {
                **measured["optimizer_state_after"],
                "parameter_entries": 3,
                "cuda_tensor_bytes": 48,
            }
        return optimizer, measured

    monkeypatch.setattr(preflight, "_measure_persistent_batch", grow_during_steady)
    with pytest.raises(RuntimeError, match="actual train/validation batch"):
        preflight.run_kbo_scale_preflight(
            dataset,
            config,
            output=growth_output,
            max_reserved_fraction=0.95,
            progress=lambda _: None,
        )
    growth_report = json.loads(growth_output.read_text(encoding="utf-8"))
    assert growth_report["failed_batch"]["status"] == "failed_optimizer_state_growth"
    assert "did not establish complete coverage" in growth_report["execution_error"]["message"]


def test_persistent_batch_helper_audits_transfer_and_retains_adamw_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Scalar:
        def __float__(self) -> float:
            return 1.25

        def backward(self) -> None:
            events.append("backward")

        def detach(self) -> _Scalar:
            return self

        def cpu(self) -> _Scalar:
            return self

    class _Scaler:
        def __init__(self, *_: Any, enabled: bool) -> None:
            self.enabled = enabled

        def scale(self, loss: _Scalar) -> _Scalar:
            return loss

        def unscale_(self, _optimizer: Any) -> None:
            events.append("unscale")

        def step(self, optimizer: Any) -> None:
            optimizer.step()

        def update(self) -> None:
            events.append("scaler_update")

        def get_scale(self) -> float:
            return 1.0

        def is_enabled(self) -> bool:
            return self.enabled

    class _Cuda:
        def empty_cache(self) -> None:
            events.append("empty_cache")

        def synchronize(self, _device: Any) -> None:
            events.append("synchronize")

        def mem_get_info(self, _device: Any) -> tuple[int, int]:
            return 40, 100

        def memory_allocated(self, _device: Any) -> int:
            return 0

        def memory_reserved(self, _device: Any) -> int:
            return 0

        def reset_peak_memory_stats(self, _device: Any) -> None:
            events.append("reset_peak")

        def manual_seed_all(self, _seed: int) -> None:
            events.append("cuda_seed")

        def max_memory_allocated(self, _device: Any) -> int:
            return 55

        def max_memory_reserved(self, _device: Any) -> int:
            return 60

    class _Torch:
        float16 = object()
        cuda = _Cuda()
        amp = SimpleNamespace(GradScaler=_Scaler)

        @staticmethod
        def manual_seed(_seed: int) -> None:
            events.append("seed")

        @staticmethod
        def autocast(*_: Any, **__: Any) -> Any:
            return nullcontext()

        @staticmethod
        def isfinite(_value: Any) -> bool:
            return True

        @staticmethod
        def is_tensor(value: Any) -> bool:
            return isinstance(value, _StateTensor)

    class _StateTensor:
        device = SimpleNamespace(type="cuda")

        @staticmethod
        def numel() -> int:
            return 2

        @staticmethod
        def element_size() -> int:
            return 4

    class _Parameter:
        @staticmethod
        def numel() -> int:
            return 10

    class _Model:
        def parameters(self) -> list[_Parameter]:
            return [_Parameter(), _Parameter()]

        def to(self, _device: Any) -> None:
            events.append("model_to")

        def train(self) -> None:
            events.append("train")

        def __call__(self, _batch: Any) -> dict[str, Any]:
            events.append("forward")
            return {"output": True}

    class _Optimizer:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def step(self) -> None:
            events.append("adamw_step")
            self.state.setdefault("parameter", {"exp_avg": _StateTensor()})

    class _Vector:
        shape = (2, 1)

        @staticmethod
        def numel() -> int:
            return 2

    batch: dict[str, Any] = {
        "node_features": {"player": _Vector(), "team": _Vector()},
        "routes": [SimpleNamespace(route_name="route", num_edges=3)],
        "day_ids": ("2023-04-01",),
        **{name: _Vector() for name in preflight._QUERY_INDEX_NAMES.values()},
    }
    scalar = _Scalar()
    monkeypatch.setattr(preflight, "require_torch", lambda: (_Torch(), object()))
    monkeypatch.setattr(preflight, "KBORelGNNModel", lambda _: _Model())
    monkeypatch.setattr(preflight, "make_adamw", lambda *_, **__: _Optimizer())
    monkeypatch.setattr(preflight, "move_batch", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(preflight, "_loss", lambda *_: {"loss": scalar})
    monkeypatch.setattr(preflight, "collate_kbo_day_graphs", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(runner_module, "_prepare_graph_batch", lambda value, _: value)
    monkeypatch.setattr(
        runner_module,
        "_clip_gradient_norms",
        lambda *_: {"shared": scalar},
    )

    dataset = _SpyDataset(
        {
            date(2023, 4, 1): _day("2023-04-01"),
            date(2024, 4, 1): _day("2024-04-01"),
            date(2025, 4, 1): _day("2025-04-01"),
        }
    )
    batch_spec = {
        "split": "train",
        "split_batch_index": 0,
        "dates": ["2023-04-01"],
        "diagnostic_selection_dimensions": ["rough_cost"],
        "estimated_work_units": 10,
        "node_counts": {"player": 2, "team": 2},
        "effective_route_edge_counts": {"route": 3},
        "effective_query_counts": dict.fromkeys(preflight._QUERY_INDEX_NAMES, 2),
    }
    model = _Model()
    optimizer, warmup = preflight._measure_persistent_batch(
        dataset,
        batch_spec,
        config=_config(device="cuda:0", amp="off"),
        model=model,
        optimizer=None,
        scaler=_Scaler("cuda", enabled=False),
        device=SimpleNamespace(type="cuda"),
        dtype=None,
        parameter_count=20,
        phase="warmup",
        actual_batch_index=0,
    )
    _, steady = preflight._measure_persistent_batch(
        dataset,
        batch_spec,
        config=_config(device="cuda:0", amp="off"),
        model=model,
        optimizer=optimizer,
        scaler=_Scaler("cuda", enabled=False),
        device=SimpleNamespace(type="cuda"),
        dtype=None,
        parameter_count=20,
        phase="steady_state",
        actual_batch_index=0,
        clear_allocator_cache_before=False,
        clear_allocator_cache_after=False,
    )

    assert warmup["parameter_count"] == 20
    assert warmup["peak_reserved_fraction"] == 0.6
    assert warmup["optimizer_state_before"]["parameter_entries"] == 0
    assert warmup["optimizer_state_after"]["cuda_tensor_bytes"] == 8
    assert steady["optimizer_state_before"] == warmup["optimizer_state_after"]
    assert steady["batch_count_transfer_verified"] is True
    assert steady["gpu_batch_counts"]["route_edges"] == 3
    assert {"forward", "backward", "adamw_step"} <= set(events)
    assert events.count("forward") == 2
    assert events.count("backward") == 2
    assert events.count("adamw_step") == 2
    assert events.count("empty_cache") == 2
    assert warmup["allocator_cache_cleared_before_batch"] is True
    assert warmup["allocator_cache_cleared_after_batch"] is True
    assert steady["allocator_cache_cleared_before_batch"] is False
    assert steady["allocator_cache_cleared_after_batch"] is False


def test_run_rejects_accumulation_and_writes_failure_without_loading_graphs(
    tmp_path: Path,
) -> None:
    dataset = _SpyDataset(
        {
            date(2023, 4, 1): _day("2023-04-01"),
            date(2024, 4, 1): _day("2024-04-01"),
            date(2025, 4, 1): _day("2025-04-01"),
        }
    )
    output = tmp_path / "rejected.json"

    with pytest.raises(ValueError, match="accumulate_steps=1"):
        preflight.run_kbo_scale_preflight(
            dataset,
            _config(device="cuda:0", accumulate_steps=2),
            output=output,
        )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "execution_failed"
    assert saved["execution_error"]["message"] == (
        "KBO scale preflight requires accumulate_steps=1"
    )
    assert dataset.loaded == []


def test_reserved_fraction_threshold_writes_failed_report_before_raising(
    tmp_path: Path,
) -> None:
    report: dict[str, Any] = {
        "execution": {
            "peak_reserved_fraction": 0.86,
            "peak_reserved_bytes": 86,
            "total_memory_bytes": 100,
        }
    }
    output = tmp_path / "preflight.json"

    with pytest.raises(RuntimeError, match="exceeds max_reserved_fraction 85.000%"):
        _enforce_reserved_fraction(report, max_reserved_fraction=0.85, output=output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed_memory_threshold"
    assert saved["memory_safety"] == {
        "passed": False,
        "max_reserved_fraction": 0.85,
        "peak_reserved_fraction": 0.86,
        "threshold_reserved_bytes": 85,
        "headroom_to_threshold_bytes": -1,
    }


def test_reserved_fraction_threshold_is_inclusive_and_validated(tmp_path: Path) -> None:
    report: dict[str, Any] = {
        "execution": {
            "peak_reserved_fraction": 0.85,
            "peak_reserved_bytes": 85,
            "total_memory_bytes": 100,
        }
    }
    output = tmp_path / "preflight.json"

    returned = _enforce_reserved_fraction(
        report, max_reserved_fraction=0.85, output=output
    )

    assert returned is report
    assert returned["status"] == "passed"
    assert json.loads(output.read_text(encoding="utf-8"))["memory_safety"]["passed"] is True
    with pytest.raises(ValueError, match="must be finite and in"):
        _enforce_reserved_fraction(report, max_reserved_fraction=float("nan"))
    with pytest.raises(ValueError, match="must be finite and in"):
        _enforce_reserved_fraction(report, max_reserved_fraction=True)
