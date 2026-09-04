from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from cpv26.training import kbo_capacity_comparison as capacity
from cpv26.training import kbo_temporal_preflight as temporal_preflight
from cpv26.training import kbo_temporal_workflow as workflow


class _Dataset:
    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        self.directory = directory
        self.manifest = manifest
        self.loaded_days: list[str] = []

    def load_day(self, day: Any) -> Any:
        self.loaded_days.append(str(day))
        raise AssertionError("workflow must not materialize graph samples")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _archive(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    directory = tmp_path / "temporal-v7"
    days = [f"{year}-04-01" for year in range(2001, 2027)]
    manifest: dict[str, Any] = {
        "dataset_version": 7,
        "graph_schema": "temporal_v7",
        "fingerprint": "a" * 64,
        "sampling_policy": {
            "lookback_days": 365,
            "max_games_per_seed_team": 160,
            "max_games_per_player": 48,
            "max_historical_games_total": 160,
        },
        "sampling_policy_fingerprint": "b" * 64,
        "node_feature_dims": {"player": 4, "team": 8, "game": 4},
        "route_feature_dims": {
            "batter_game_event": 6,
            "pitcher_game_event": 6,
            "team_game_event": 4,
            "batter_pa_pitcher_event": 17,
        },
        "archive_policy": {"daily_graph_files": False},
        "temporal_batching": {
            "max_nodes_per_batch": 100_000,
            "max_edges_per_batch": 200_000,
            "max_days_per_batch": 8,
        },
        "days": [
            {"day": day, "query_games": 1, "label_references": 3} for day in days
        ],
    }
    entries = [
        {
            "day": day,
            "sample_nodes": {"player": 100 + index, "team": 10, "game": 81},
            "sample_edges": {
                "batter_game_event": 1_200 + index,
                "pitcher_game_event": 400 + index,
                "team_game_event": 162,
                "batter_pa_pitcher_event": 3_000 + index,
            },
            "sample_fingerprint": f"{index + 1:064x}",
        }
        for index, day in enumerate(days[:-1])
    ]
    sample_index = {
        "schema_version": 2,
        "sample_fingerprint_scope": "all_materialized_arrays_v2",
        "dataset_fingerprint": manifest["fingerprint"],
        "sampling_policy": manifest["sampling_policy"],
        "sampling_policy_fingerprint": manifest["sampling_policy_fingerprint"],
        "label_year_ceiling": 2025,
        "held_out_labels_loaded": False,
        "days": entries,
        "fingerprint": workflow._json_sha256(entries),
    }
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "sample_index.json", sample_index)
    return directory, manifest, entries


def _child_report(
    plan: workflow.KBOTemporalWorkflowPlan,
    *,
    test_used: bool = False,
) -> dict[str, Any]:
    child_directory = plan.output_directory / workflow.TEMPORAL_CHILD_DIRECTORY
    runtime = {
        "device": "cuda:0",
        "gpu_name": "test GPU",
        "total_memory_bytes": 10 * 2**30,
        "compute_capability": [8, 0],
        "torch_version": "2.test",
        "cuda_runtime": "12.test",
        "precision": "bf16",
    }
    runs: dict[str, Any] = {}
    for variant in workflow.TEMPORAL_VARIANTS:
        run_directory = child_directory / f"seed-{plan.config.seed}" / variant
        training = {
            "status": "completed",
            "dataset_fingerprint": "a" * 64,
            "held_out_test_season": 2026,
            "test_used_during_training": test_used,
            "runtime": runtime,
            "peak_cuda_allocated_bytes": 4 * 2**30,
            "peak_cuda_reserved_bytes": 6 * 2**30,
        }
        _write_json(run_directory / "training_report.json", training)
        runs[variant] = {
            "run_directory": str(run_directory.resolve()),
            "validation_selection_loss": 4.5 if variant == "full" else 4.54,
            "test_used_during_training": test_used,
        }
    loader_fingerprint = "c" * 64
    preflight_path = plan.preflight_report_path
    assert preflight_path is not None
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    return {
        "status": "completed",
        "protocol": capacity.FULL_NODE_COMPARISON_PROTOCOL,
        "dataset_fingerprint": "a" * 64,
        "seed": plan.config.seed,
        "variants": list(workflow.TEMPORAL_VARIANTS),
        "training_config": json.loads(json.dumps(asdict(plan.config))),
        "capacity": dict(workflow.TEMPORAL_CAPACITY),
        "held_out_test_season": 2026,
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "smoke_test_only": False,
        "runtime_signature": runtime,
        "loader_lineage": {
            "dataset_fingerprint": "a" * 64,
            "variant_fingerprints": {
                variant: loader_fingerprint for variant in workflow.TEMPORAL_VARIANTS
            },
            "all_non_route_settings_equal": True,
        },
        "runs": runs,
        "validation_selection_comparison": {
            "lower_is_better": True,
            "full": 4.5,
            "node_only": 4.54,
            "node_only_minus_full": 0.04,
        },
        "temporal_execution_attestation": {
            "all_variants_exact_plan": True,
            "plan_fingerprint": preflight["execution_plan"]["plan_fingerprint"],
            "preflight_report_sha256": preflight_sha,
        },
    }


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch,
    plan: workflow.KBOTemporalWorkflowPlan,
) -> None:
    def run(*_args: Any, output: Path, **_kwargs: Any) -> dict[str, Any]:
        selected_plan = {
            "plan_fingerprint": "e" * 64,
            "budgets": {"max_nodes": 100_000, "max_edges": 200_000},
            "actual_batch_count": 25,
        }
        report = {
            "status": "passed",
            "selected_for_training": True,
            "max_reserved_fraction": 0.85,
            "selected_attempt": 1,
            "execution_plan": selected_plan,
            "peak_reserved_fraction": 0.72,
            "all_actual_batches_measured": True,
        }
        _write_json(output, report)
        return report

    monkeypatch.setattr(
        temporal_preflight, "run_adaptive_temporal_cuda_preflight", run
    )


def test_plan_rejects_multi_seed_extra_variants_and_nonproduction_config(
    tmp_path: Path,
) -> None:
    dataset, _, _ = _archive(tmp_path)
    base = workflow._default_training_config()
    with pytest.raises(ValueError, match="exactly one configured seed"):
        workflow.KBOTemporalWorkflowPlan(
            dataset,
            tmp_path / "run-a",
            config=base,
            seeds=(base.seed, base.seed + 1),
        )
    with pytest.raises(ValueError, match="exactly full and node_only"):
        workflow.KBOTemporalWorkflowPlan(
            dataset,
            tmp_path / "run-b",
            config=base,
            variants=("full", "node_only", "rewired"),
        )
    with pytest.raises(ValueError, match="fixed production contract"):
        workflow.KBOTemporalWorkflowPlan(
            dataset,
            tmp_path / "run-c",
            config=replace(base, hidden_dim=128),
        )
    with pytest.raises(ValueError, match="explicit CUDA"):
        workflow.KBOTemporalWorkflowPlan(
            dataset,
            tmp_path / "run-d",
            config=replace(base, device="cpu", amp="off"),
        )


def test_workflow_validates_all_topologies_and_runs_one_matched_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, entries = _archive(tmp_path)
    dataset = _Dataset(directory, manifest)
    plan = workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "workflow")
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    _patch_preflight(monkeypatch, plan)

    def train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, "kwargs": kwargs})
        report = _child_report(plan)
        _write_json(
            plan.output_directory
            / workflow.TEMPORAL_CHILD_DIRECTORY
            / "full_node_comparison_report.json",
            report,
        )
        return report

    monkeypatch.setattr(capacity, "train_kbo_full_node_comparison", train)
    report = workflow.run_kbo_temporal_workflow(plan, progress=lambda _: None)

    assert report["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["kwargs"]["config"] == plan.config
    assert calls[0]["kwargs"]["temporal_preflight_report"] == plan.preflight_report_path
    assert calls[0]["args"] == (
        plan.dataset_directory,
        plan.output_directory / workflow.TEMPORAL_CHILD_DIRECTORY,
    )
    assert dataset.loaded_days == []
    assert report["held_out_test"] == {
        "season": 2026,
        "labels_loaded_by_workflow": False,
        "sample_loaded_by_workflow": False,
        "used_for_training_selection_or_comparison": False,
        "sealed": True,
    }
    lineage = report["sample_fingerprint_lineage"]
    assert lineage["splits"]["train"]["days"] == 24
    assert lineage["splits"]["validation"]["days"] == 1
    assert len(set(lineage["variant_fingerprints"].values())) == 1
    assert report["topology_size_quantiles"]["sample_count"] == len(entries)
    assert report["gpu_runtime"]["all_variants_same_runtime"] is True
    assert report["validation_selection_comparison"]["node_only_minus_full"] == 0.04


def test_workflow_rejects_missing_or_held_out_topology_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, _ = _archive(tmp_path)
    dataset = _Dataset(directory, manifest)
    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    monkeypatch.setattr(
        capacity,
        "train_kbo_full_node_comparison",
        lambda *_args, **_kwargs: pytest.fail("training must not start"),
    )
    index_path = directory / "sample_index.json"
    sample_index = json.loads(index_path.read_text(encoding="utf-8"))
    sample_index["days"].pop()
    sample_index["fingerprint"] = workflow._json_sha256(sample_index["days"])
    _write_json(index_path, sample_index)
    with pytest.raises(ValueError, match="every and only train/validation day"):
        workflow.run_kbo_temporal_workflow(
            workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "missing")
        )

    directory, manifest, _ = _archive(tmp_path / "second")
    dataset = _Dataset(directory, manifest)
    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    index_path = directory / "sample_index.json"
    sample_index = json.loads(index_path.read_text(encoding="utf-8"))
    test_entry = {
        **sample_index["days"][-1],
        "day": "2026-04-01",
        "sample_fingerprint": "f" * 64,
    }
    sample_index["days"].append(test_entry)
    sample_index["fingerprint"] = workflow._json_sha256(sample_index["days"])
    _write_json(index_path, sample_index)
    with pytest.raises(ValueError, match="held-out test sample"):
        workflow.run_kbo_temporal_workflow(
            workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "held-out")
        )


def test_workflow_marks_report_failed_when_child_exposes_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, _ = _archive(tmp_path)
    dataset = _Dataset(directory, manifest)
    plan = workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "workflow")
    _patch_preflight(monkeypatch, plan)
    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    monkeypatch.setattr(
        capacity,
        "train_kbo_full_node_comparison",
        lambda *_args, **_kwargs: _child_report(plan, test_used=True),
    )

    with pytest.raises(ValueError, match="opened or mislabeled held-out test"):
        workflow.run_kbo_temporal_workflow(plan, progress=lambda _: None)
    saved = json.loads(
        (plan.output_directory / workflow.TEMPORAL_WORKFLOW_REPORT).read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == "failed"
    assert saved["held_out_test"]["sealed"] is True


def test_workflow_resume_binds_requested_memory_limit_into_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, _ = _archive(tmp_path)
    dataset = _Dataset(directory, manifest)
    plan = workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "workflow")
    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    _patch_preflight(monkeypatch, plan)
    monkeypatch.setattr(
        capacity,
        "train_kbo_full_node_comparison",
        lambda *_args, **_kwargs: _child_report(plan),
    )
    workflow.run_kbo_temporal_workflow(plan, progress=lambda _: None)

    changed = workflow.KBOTemporalWorkflowPlan(
        directory,
        plan.output_directory,
        max_reserved_fraction=0.5,
    )
    with pytest.raises(ValueError, match="resume changed"):
        workflow.run_kbo_temporal_workflow(changed, progress=lambda _: None)


def test_workflow_rejects_reused_preflight_with_a_different_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, _ = _archive(tmp_path)
    dataset = _Dataset(directory, manifest)
    plan = workflow.KBOTemporalWorkflowPlan(directory, tmp_path / "workflow")
    monkeypatch.setattr(workflow, "open_kbo_graph_dataset", lambda _: dataset)
    _patch_preflight(monkeypatch, plan)
    monkeypatch.setattr(
        capacity,
        "train_kbo_full_node_comparison",
        lambda *_args, **_kwargs: _child_report(plan),
    )
    workflow.run_kbo_temporal_workflow(plan, progress=lambda _: None)

    assert plan.preflight_report_path is not None
    preflight = json.loads(plan.preflight_report_path.read_text(encoding="utf-8"))
    preflight["max_reserved_fraction"] = 0.5
    _write_json(plan.preflight_report_path, preflight)
    with pytest.raises(ValueError, match="memory limit differs"):
        workflow.run_kbo_temporal_workflow(plan, progress=lambda _: None)
