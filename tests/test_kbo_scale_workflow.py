from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_kbo_scale_comparison import _reports

from cpv26.training import kbo_scale_workflow as workflow
from cpv26.training.kbo_runner import KBOTrainingConfig
from cpv26.training.kbo_scale_preflight import SCALE_PREFLIGHT_PROTOCOL_VERSION


def _valid_baseline() -> dict[str, Any]:
    baseline, _ = _reports()
    config = KBOTrainingConfig.from_dict(baseline["training_config"])
    policies = workflow.capacity._variant_protocols(config)
    baseline["variant_policies"] = policies
    for variant in ("full", "node_only"):
        baseline["runs"]["expanded_128x3"][variant]["variant_policy"] = policies[
            variant
        ]
    return baseline


def _baseline_path(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    baseline = _valid_baseline()
    path = tmp_path / "capacity_comparison_report.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    return path, baseline


def test_completed_baseline_validator_accepts_6c4658b_schema() -> None:
    baseline = _valid_baseline()

    config, lineage, runtime = workflow._validate_completed_baseline(baseline)

    assert (config.hidden_dim, config.layers, config.heads) == (128, 3, 4)
    assert config.activation_checkpointing is False
    assert config.compact_kbo_channels is False
    assert config.seed == lineage["seed"] == 2026
    assert runtime == baseline["runtime_signature"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(status="running"), "not completed"),
        (
            lambda report: report.update(
                test_used_for_training_selection_or_comparison=True
            ),
            "held-out test stayed sealed",
        ),
        (
            lambda report: report["runs"]["expanded_128x3"]["full"].update(
                test_used_during_training=True
            ),
            "test stayed sealed",
        ),
        (
            lambda report: report["training_config"].update(seed=2027),
            "requires seed=2026",
        ),
        (
            lambda report: report["training_config"].pop("dropout"),
            "training_config schema differs",
        ),
        (
            lambda report: report["initialization_audit"].update(
                all_variants_equal=False
            ),
            "initialization audit",
        ),
    ],
)
def test_completed_baseline_validator_rejects_unsealed_or_drifted_sources(
    mutate: Any, message: str
) -> None:
    baseline = _valid_baseline()
    mutated = copy.deepcopy(baseline)
    mutate(mutated)

    with pytest.raises(ValueError, match=message):
        workflow._validate_completed_baseline(mutated)


def test_completed_baseline_validator_rejects_smoke_or_capped_source() -> None:
    smoke = _valid_baseline()
    smoke["training_config"]["max_days_per_split"] = 1
    with pytest.raises(ValueError, match="smoke/subset"):
        workflow._validate_completed_baseline(smoke)

    capped = _valid_baseline()
    capped["training_config"]["max_edges_per_route_per_day"] = 100
    with pytest.raises(ValueError, match="uncapped"):
        workflow._validate_completed_baseline(capped)


def test_prepare_derives_only_the_controlled_capacity_and_execution_change(
    tmp_path: Path, monkeypatch: Any
) -> None:
    baseline_path, baseline = _baseline_path(tmp_path)
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    class FakeDataset:
        manifest = {
            "fingerprint": baseline["dataset_fingerprint"],
            "dataset_version": 5,
        }

        def __init__(self, directory: Path) -> None:
            assert directory == dataset_directory.resolve()

    class FakeParameter:
        def numel(self) -> int:
            return 26_140_772

    class FakeModel:
        def __init__(self, _config: Any) -> None:
            pass

        def parameters(self) -> list[FakeParameter]:
            return [FakeParameter()]

    monkeypatch.setattr(workflow, "KBOGraphDataset", FakeDataset)
    monkeypatch.setattr(workflow, "KBORelGNNModel", FakeModel)
    monkeypatch.setattr(workflow.runner, "_model_config", lambda *_args: object())
    monkeypatch.setattr(
        workflow.matched,
        "_split_day_fingerprint",
        lambda _dataset, _config: (
            baseline["split_day_fingerprint"],
            baseline["split_days"],
        ),
    )
    monkeypatch.setattr(
        workflow.matched,
        "_runtime_signature",
        lambda _config: baseline["runtime_signature"],
    )
    output = tmp_path / "runs" / "candidate"

    plan = workflow.prepare_kbo_scale_training(
        baseline_path,
        dataset_directory,
        output,
        device="cuda:0",
    )

    source = KBOTrainingConfig.from_dict(baseline["training_config"])
    actual = asdict(plan.config)
    expected = asdict(source)
    expected.update(
        hidden_dim=256,
        layers=3,
        heads=8,
        activation_checkpointing=True,
        compact_kbo_channels=False,
    )
    assert actual == expected
    assert plan.preflight_report == output.parent / "candidate.scale_preflight.json"
    assert plan.preflight_report.parent == output.parent
    assert plan.preflight_report.parent != output
    assert plan.candidate_report.parent == output.resolve()
    assert plan.scale_report.parent == output.resolve()
    assert plan.expected_parameter_count == 26_140_772


def test_prepare_rejects_device_string_drift_before_opening_dataset(
    tmp_path: Path, monkeypatch: Any
) -> None:
    baseline_path, _ = _baseline_path(tmp_path)
    opened = False

    class UnexpectedDataset:
        def __init__(self, _directory: Path) -> None:
            nonlocal opened
            opened = True

    monkeypatch.setattr(workflow, "KBOGraphDataset", UnexpectedDataset)

    with pytest.raises(ValueError, match="must exactly match"):
        workflow.prepare_kbo_scale_training(
            baseline_path,
            tmp_path,
            tmp_path / "candidate",
            device="cuda:1",
        )
    assert opened is False


def test_prepare_rejects_vnext_dataset_before_split_or_model(
    tmp_path: Path, monkeypatch: Any
) -> None:
    baseline_path, baseline = _baseline_path(tmp_path)
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    class VNextDataset:
        manifest = {
            "fingerprint": baseline["dataset_fingerprint"],
            "dataset_version": 6,
        }

        def __init__(self, _directory: Path) -> None:
            pass

    monkeypatch.setattr(workflow, "KBOGraphDataset", VNextDataset)
    with pytest.raises(ValueError, match="requires v5"):
        workflow.prepare_kbo_scale_training(
            baseline_path, dataset_directory, tmp_path / "candidate"
        )


def _plan(tmp_path: Path) -> workflow.KBOScaleTrainingPlan:
    baseline_path = tmp_path / "capacity_comparison_report.json"
    baseline_path.write_text(json.dumps({"dataset_fingerprint": "fingerprint"}))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "candidate"
    config = replace(
        KBOTrainingConfig(),
        device="cuda:0",
        epochs=30,
        batch_days=8,
        hidden_dim=256,
        layers=3,
        heads=8,
        workers=0,
        accumulate_steps=1,
        max_pa_per_day=0,
        max_edges_per_route_per_day=0,
        patience=0,
        train_seasons=tuple(range(2001, 2025)),
        validation_season=2025,
        test_season=2026,
        chronological=True,
        activation_checkpointing=True,
        compact_kbo_channels=False,
    )
    return workflow.KBOScaleTrainingPlan(
        baseline_report=baseline_path.resolve(),
        dataset_directory=dataset.resolve(),
        output_directory=output.resolve(),
        preflight_report=(tmp_path / "candidate.scale_preflight.json").resolve(),
        candidate_report=(output / "full_node_comparison_report.json").resolve(),
        scale_report=(output / "scale_comparison_report.json").resolve(),
        config=config,
        runtime_signature={
            "device": "cuda:0",
            "gpu_name": "A100",
            "total_memory_bytes": 10_000,
            "compute_capability": [8, 0],
            "torch_version": "2.5.1+cu121",
            "cuda_runtime": "12.1",
            "precision": "bf16",
        },
        expected_parameter_count=26_140_772,
        dataset_fingerprint="fingerprint",
        baseline_report_sha256=workflow.sha256_file(baseline_path),
    )


def _passed_preflight(plan: workflow.KBOScaleTrainingPlan) -> dict[str, Any]:
    test_seal = {
        "season": plan.config.test_season,
        "graph_days_loaded": False,
        "labels_loaded": False,
        "sealed": True,
    }
    return {
        "status": "passed",
        "protocol_version": SCALE_PREFLIGHT_PROTOCOL_VERSION,
        "candidate_config": asdict(plan.config),
        "measurement_policy": {
            "model_and_optimizer": "one_fresh_instance_persistent_across_all_steps",
            "passes": [
                "first_batch_optimizer_warmup",
                "all_actual_batches_optimizer_state_materialization",
                "all_actual_batches_steady_state_measurement",
            ],
            "allocator_between_batches": (
                "steady_state_cache_retained_after_single_initial_empty_cache"
            ),
            "warmup_and_materialization_allocator": (
                "isolated_empty_cache_before_and_after_each_batch"
            ),
        },
        "runtime": dict(plan.runtime_signature),
        "workload_audit": {
            "dataset_fingerprint": "fingerprint",
            "candidate_config": asdict(plan.config),
            "splits": {"test": dict(test_seal)},
        },
        "held_out_test": test_seal,
        "planned_actual_batch_count": 2,
        "completed_actual_batch_count": 2,
        "completed_materialization_batch_count": 2,
        "materialization_batches": [{}, {}],
        "evaluated_batches": [{}, {}],
        "execution": {
            "all_actual_batches_evaluated": True,
            "optimizer_state_locked_before_steady_state": True,
            "steady_state_allocator_cache_cleared_once_before_pass": True,
            "steady_state_allocator_cache_retained_between_batches": True,
            "steady_state_cumulative_peak_reserved_bytes": 7_000,
            "steady_state_cumulative_peak_reserved_fraction": 0.7,
            "overall_peak_includes_warmup": True,
            "overall_peak_includes_materialization_pass": True,
            "warmup_steps": 1,
            "materialization_steps": 2,
            "steady_state_steps": 2,
            "evaluated_batch_count": 2,
            "parameter_count": plan.expected_parameter_count,
            "peak_reserved_bytes": 7_000,
            "total_memory_bytes": 10_000,
            "peak_reserved_fraction": 0.7,
            "headroom_bytes": 3_000,
        },
        "memory_safety": {
            "passed": True,
            "max_reserved_fraction": 0.85,
            "peak_reserved_fraction": 0.7,
            "threshold_reserved_bytes": 8_500,
            "headroom_to_threshold_bytes": 1_500,
        },
    }


def test_workflow_consumes_persisted_gate_before_pair_and_report(
    tmp_path: Path, monkeypatch: Any
) -> None:
    plan = _plan(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        workflow,
        "prepare_kbo_scale_training",
        lambda *_args, **_kwargs: plan,
    )

    from cpv26.training import kbo_scale_preflight

    def fake_preflight(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("preflight")
        report = _passed_preflight(plan)
        Path(kwargs["output"]).write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_pair(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("pair")
        assert kwargs["config"] is plan.config
        plan.output_directory.mkdir()
        plan.candidate_report.write_text("{}", encoding="utf-8")
        return {}

    def fake_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("compare")
        assert args == (plan.baseline_report, plan.candidate_report)
        assert kwargs["output_path"] is None
        return {
            "status": "completed",
            "protocol": "single_seed_validation_scale_comparison",
            "protocol_version": 1,
        }

    monkeypatch.setattr(kbo_scale_preflight, "run_kbo_scale_preflight", fake_preflight)
    monkeypatch.setattr(
        workflow.capacity, "train_kbo_full_node_comparison", fake_pair
    )
    monkeypatch.setattr(workflow.scale, "compare_kbo_scale_reports", fake_compare)

    report = workflow.train_kbo_scale_workflow(
        plan.baseline_report,
        plan.dataset_directory,
        plan.output_directory,
    )

    assert events == ["preflight", "pair", "compare"]
    assert report["status"] == "completed"
    assert report["preflight"]["execution"]["parameter_count"] == 26_140_772
    persisted_scale = json.loads(plan.scale_report.read_text(encoding="utf-8"))
    assert persisted_scale["preflight_gate"]["status"] == "passed"
    assert persisted_scale["preflight_gate"]["all_actual_batches_evaluated"] is True
    assert persisted_scale["protocol"] == workflow.SCALE_TRAINING_WORKFLOW_PROTOCOL
    assert persisted_scale["comparison_protocol"]["protocol_version"] == 1


def test_workflow_rejects_persisted_preflight_drift_before_pair(
    tmp_path: Path, monkeypatch: Any
) -> None:
    plan = _plan(tmp_path)
    monkeypatch.setattr(
        workflow,
        "prepare_kbo_scale_training",
        lambda *_args, **_kwargs: plan,
    )
    from cpv26.training import kbo_scale_preflight

    def fake_preflight(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        returned = _passed_preflight(plan)
        persisted = copy.deepcopy(returned)
        persisted["candidate_config"]["seed"] = 7
        Path(kwargs["output"]).write_text(json.dumps(persisted), encoding="utf-8")
        return returned

    pair_called = False

    def unexpected_pair(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal pair_called
        pair_called = True
        return {}

    monkeypatch.setattr(kbo_scale_preflight, "run_kbo_scale_preflight", fake_preflight)
    monkeypatch.setattr(
        workflow.capacity, "train_kbo_full_node_comparison", unexpected_pair
    )

    with pytest.raises(ValueError, match="persisted scale preflight report differs"):
        workflow.train_kbo_scale_workflow(
            plan.baseline_report,
            plan.dataset_directory,
            plan.output_directory,
        )
    assert pair_called is False


def test_workflow_rejects_preflight_changed_during_pair(
    tmp_path: Path, monkeypatch: Any
) -> None:
    plan = _plan(tmp_path)
    monkeypatch.setattr(
        workflow, "prepare_kbo_scale_training", lambda *_args, **_kwargs: plan
    )
    from cpv26.training import kbo_scale_preflight

    def fake_preflight(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        report = _passed_preflight(plan)
        Path(kwargs["output"]).write_text(json.dumps(report), encoding="utf-8")
        return report

    def mutating_pair(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        plan.preflight_report.write_text("{}", encoding="utf-8")
        return {}

    compared = False

    def unexpected_compare(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal compared
        compared = True
        return {}

    monkeypatch.setattr(kbo_scale_preflight, "run_kbo_scale_preflight", fake_preflight)
    monkeypatch.setattr(
        workflow.capacity, "train_kbo_full_node_comparison", mutating_pair
    )
    monkeypatch.setattr(workflow.scale, "compare_kbo_scale_reports", unexpected_compare)

    with pytest.raises(ValueError, match="preflight report changed"):
        workflow.train_kbo_scale_workflow(
            plan.baseline_report, plan.dataset_directory, plan.output_directory
        )
    assert compared is False
