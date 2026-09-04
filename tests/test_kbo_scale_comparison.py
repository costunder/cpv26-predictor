from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cpv26.training.kbo_capacity_comparison import (
    CAPACITY_COMPARISON_PROTOCOL,
    FULL_NODE_COMPARISON_PROTOCOL,
)
from cpv26.training.kbo_scale_comparison import compare_kbo_scale_reports

VARIANTS = ("full", "node_only")


def _training_config(*, candidate: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "device": "cuda:0",
        "epochs": 30,
        "batch_days": 8,
        "hidden_dim": 256 if candidate else 128,
        "layers": 3,
        "heads": 8 if candidate else 4,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "amp": "bf16",
        "workers": 0,
        "accumulate_steps": 1,
        "gradient_clip": 1.0,
        "max_pa_per_day": 0,
        "max_edges_per_route_per_day": 0,
        "patience": 0,
        "seed": 2026,
        "match_weight": 1.0,
        "live_hit_weight": 1.0,
        "pa_weight": 0.2,
        "run_weight": 0.1,
        "box_pa_weight": 0.2,
        "box_pitch_weight": 0.1,
        "selection_target": "auto",
        "box_gradient_mode": "auto",
        "max_days_per_split": None,
        "train_seasons": list(range(2001, 2025)),
        "validation_season": 2025,
        "test_season": 2026,
        "chronological": True,
        "route_message_normalization": "none",
        "route_schedule": "full",
        "route_edge_chunk_size": 0,
        "graph_control": "intact",
        "graph_control_seed": 2026,
        "compact_kbo_channels": True,
    }
    if candidate:
        config["activation_checkpointing"] = True
    return config


def _variant_policies() -> dict[str, Any]:
    return {
        "full": {
            "route_message_normalization": "none",
            "route_schedule": "full",
            "graph_control": "intact",
            "resolved_route_schedule": [["route"]] * 3,
        },
        "node_only": {
            "route_message_normalization": "none",
            "route_schedule": "node_only",
            "graph_control": "intact",
            "resolved_route_schedule": [[], [], []],
        },
    }


def _loader(config: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "dataset_fingerprint": "dataset-fingerprint",
        "split_day_fingerprint": "split-fingerprint",
        "split_days": {
            "train": ["2001-04-05", "2024-10-01"],
            "validation": ["2025-03-22", "2025-10-01"],
        },
        "seed": 2026,
        "chronological": True,
        "batch_days": config["batch_days"],
        "workers": 0,
        "accumulate_steps": 1,
        "max_days_per_split": None,
        "max_pa_per_day": 0,
        "max_edges_per_route_per_day": 0,
        "graph_control": "intact",
        "graph_control_seed": 2026,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    value.update(
        fingerprint=fingerprint,
        variant_fingerprints={variant: fingerprint for variant in VARIANTS},
        all_non_route_settings_equal=True,
    )
    return value


def _run(
    *,
    loss: float,
    parameters: int,
    attempted_steps: int,
    policy: dict[str, Any],
    initialization_hash: str,
    shared_initialization_hash: str,
) -> dict[str, Any]:
    node_only = policy["route_schedule"] == "node_only"
    architecture = {
        "variant": "node_only" if node_only else "relational",
        "relational_message_passing_enabled": not node_only,
    }
    return {
        "completed_epochs": 30,
        "attempted_optimizer_steps": attempted_steps,
        "optimizer_steps": attempted_steps,
        "skipped_optimizer_steps": 0,
        "parameter_count": parameters,
        "trainable_parameter_count": parameters,
        "parameter_contract": {
            "parameter_count": parameters,
            "trainable_parameter_count": parameters,
            "optimizer_covers_all_trainable": True,
            "architecture": architecture,
        },
        "initial_model_state_sha256": initialization_hash,
        "shared_parameter_initialization_sha256": shared_initialization_hash,
        "architecture": architecture,
        "validation_selection_loss": loss,
        "validation_metrics": {
            "selection_target": "weighted",
            "loss_sample_counts": {
                "match": 4200,
                "live_hit": 15000,
                "pa": 650000,
            },
        },
        "variant_policy": policy,
        "test_used_during_training": False,
        "smoke_test_only": False,
    }


def _reports() -> tuple[dict[str, Any], dict[str, Any]]:
    policies = _variant_policies()
    baseline_config = _training_config(candidate=False)
    candidate_config = _training_config(candidate=True)
    baseline_runs = {
        "full": _run(
            loss=4.50,
            parameters=1_000_000,
            attempted_steps=900,
            policy=policies["full"],
            initialization_hash="b" * 64,
            shared_initialization_hash="1" * 64,
        ),
        "node_only": _run(
            loss=4.55,
            parameters=650_000,
            attempted_steps=900,
            policy=policies["node_only"],
            initialization_hash="a" * 64,
            shared_initialization_hash="1" * 64,
        ),
    }
    candidate_runs = {
        "full": _run(
            loss=4.40,
            parameters=3_000_000,
            attempted_steps=900,
            policy=policies["full"],
            initialization_hash="c" * 64,
            shared_initialization_hash="2" * 64,
        ),
        "node_only": _run(
            loss=4.46,
            parameters=1_800_000,
            attempted_steps=900,
            policy=policies["node_only"],
            initialization_hash="f" * 64,
            shared_initialization_hash="2" * 64,
        ),
    }
    common = {
        "status": "completed",
        "protocol_version": 2,
        "dataset_fingerprint": "dataset-fingerprint",
        "split_day_fingerprint": "split-fingerprint",
        "split_days": {
            "train": ["2001-04-05", "2024-10-01"],
            "validation": ["2025-03-22", "2025-10-01"],
        },
        "training_seasons": list(range(2001, 2025)),
        "validation_season": 2025,
        "held_out_test_season": 2026,
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "smoke_test_only": False,
        "seed": 2026,
        "variants": list(VARIANTS),
        "runtime_signature": {
            "torch_version": "2.5.1+cu121",
            "cuda_runtime": "12.1",
            "precision": "bf16",
            "compute_capability": [8, 0],
        },
        "variant_policies": policies,
    }
    baseline = {
        **copy.deepcopy(common),
        "protocol": CAPACITY_COMPARISON_PROTOCOL,
        "training_config": baseline_config,
        "expanded_capacity": {"hidden_dim": 128, "layers": 3},
        "initialization_audit": {
            "seed": 2026,
            "all_shared_parameters_equal": True,
            "shared_parameter_initialization_sha256": "1" * 64,
            "variants": {
                "full": {
                    "initial_model_state_sha256": "b" * 64,
                    "parameter_count": 1_000_000,
                    "trainable_parameter_count": 1_000_000,
                    "shared_parameter_initialization_sha256": "1" * 64,
                },
                "node_only": {
                    "initial_model_state_sha256": "a" * 64,
                    "parameter_count": 650_000,
                    "trainable_parameter_count": 650_000,
                    "shared_parameter_initialization_sha256": "1" * 64,
                },
            },
        },
        "loader_lineage": {
            **_loader(baseline_config),
            "capacities": ["baseline_64x2", "expanded_128x3"],
            "all_capacities_equal": True,
        },
        "runs": {
            "baseline_64x2": {},
            "expanded_128x3": baseline_runs,
        },
        "parameter_count_audit": {
            "baseline_64x2": {"full": 300_000, "node_only": 200_000},
            "expanded_128x3": {"full": 1_000_000, "node_only": 650_000},
            "expanded_128x3_trainable": {
                "full": 1_000_000,
                "node_only": 650_000,
            },
            "increase_by_variant": {"full": 700_000, "node_only": 450_000},
            "variant_counts_intentionally_distinct": True,
        },
        "budget_audit": {
            "seed": 2026,
            "all_runs_equal": True,
            "completed_epochs": {
                "baseline_64x2": {"full": 30, "node_only": 30},
                "expanded_128x3": {"full": 30, "node_only": 30},
            },
            "attempted_optimizer_steps": {
                "baseline_64x2": {"full": 900, "node_only": 900},
                "expanded_128x3": {"full": 900, "node_only": 900},
            },
        },
        "validation_selection_comparison": {
            "lower_is_better": True,
            "baseline_64x2": {
                "full": 4.6,
                "node_only": 4.7,
                "node_only_minus_full": 0.1,
            },
            "expanded_128x3": {
                "full": 4.50,
                "node_only": 4.55,
                "node_only_minus_full": 0.05,
            },
        },
    }
    candidate = {
        **copy.deepcopy(common),
        "protocol": FULL_NODE_COMPARISON_PROTOCOL,
        "training_config": candidate_config,
        "capacity": {"hidden_dim": 256, "layers": 3},
        "initialization_audit": {
            "seed": 2026,
            "all_shared_parameters_equal": True,
            "shared_parameter_initialization_sha256": "2" * 64,
            "variants": {
                "full": {
                    "initial_model_state_sha256": "c" * 64,
                    "parameter_count": 3_000_000,
                    "trainable_parameter_count": 3_000_000,
                    "shared_parameter_initialization_sha256": "2" * 64,
                },
                "node_only": {
                    "initial_model_state_sha256": "f" * 64,
                    "parameter_count": 1_800_000,
                    "trainable_parameter_count": 1_800_000,
                    "shared_parameter_initialization_sha256": "2" * 64,
                },
            },
        },
        "loader_lineage": _loader(candidate_config),
        "runs": candidate_runs,
        "parameter_count_audit": {
            "variant_parameter_counts": {
                "full": 3_000_000,
                "node_only": 1_800_000,
            },
            "variant_trainable_parameter_counts": {
                "full": 3_000_000,
                "node_only": 1_800_000,
            },
            "variant_counts_intentionally_distinct": True,
        },
        "budget_audit": {
            "all_variants_equal": True,
            "completed_epochs": {"full": 30, "node_only": 30},
            "attempted_optimizer_steps": {"full": 900, "node_only": 900},
        },
        "validation_sample_count_audit": {
            "available": True,
            "all_variants_equal": True,
            "loss_sample_counts": {
                "match": 4200,
                "live_hit": 15000,
                "pa": 650000,
            },
        },
        "validation_selection_comparison": {
            "lower_is_better": True,
            "full": 4.40,
            "node_only": 4.46,
            "node_only_minus_full": 0.06,
        },
    }
    return baseline, candidate


def _write_reports(
    tmp_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[Path, Path]:
    baseline_path = tmp_path / "capacity_comparison_report.json"
    candidate_path = tmp_path / "full_node_comparison_report.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return baseline_path, candidate_path


def test_scale_comparison_validates_lineage_and_writes_atomic_report(
    tmp_path: Path,
) -> None:
    baseline, candidate = _reports()
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)
    output = tmp_path / "nested" / "scale_comparison_report.json"

    report = compare_kbo_scale_reports(baseline_path, candidate_path, output)

    comparison = report["validation_selection_comparison"]
    assert report["status"] == "completed"
    assert report["selection_split"] == "validation"
    assert report["test_used_for_training_selection_or_comparison"] is False
    assert report["capacities"] == {
        "baseline_128x3": {"hidden_dim": 128, "layers": 3, "heads": 4},
        "candidate_256x3": {"hidden_dim": 256, "layers": 3, "heads": 8},
    }
    assert comparison["candidate_minus_baseline"]["full"] == pytest.approx(-0.1)
    assert comparison[
        "dependency_gap_change_256x3_minus_128x3"
    ] == pytest.approx(0.01)
    assert report["parameter_count_audit"][
        "candidate_to_baseline_ratio_by_variant"
    ] == {"full": 3.0, "node_only": pytest.approx(1_800_000 / 650_000)}
    checkpoint_audit = report["runtime_audit"]["permitted_execution_differences"][
        "activation_checkpointing"
    ]
    assert checkpoint_audit == {
        "baseline_128x3": False,
        "candidate_256x3": True,
        "equal": False,
        "comparison_allowed": True,
        "baseline_legacy_missing_normalized_to_false": True,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not list(output.parent.glob(".*.part"))


def test_batch_days_and_optimizer_step_differences_are_rejected(tmp_path: Path) -> None:
    baseline, candidate = _reports()
    candidate["training_config"]["batch_days"] = 4
    candidate["loader_lineage"] = _loader(candidate["training_config"])
    candidate["runs"]["full"]["attempted_optimizer_steps"] = 1800
    candidate["runs"]["node_only"]["attempted_optimizer_steps"] = 1800
    candidate["budget_audit"]["attempted_optimizer_steps"] = {
        "full": 1800,
        "node_only": 1800,
    }
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match="training_config.batch_days differ"):
        compare_kbo_scale_reports(baseline_path, candidate_path)

    baseline, candidate = _reports()
    for run in candidate["runs"].values():
        run["attempted_optimizer_steps"] = 899
        run["optimizer_steps"] = 899
    candidate["budget_audit"]["attempted_optimizer_steps"] = {
        "full": 899,
        "node_only": 899,
    }
    (tmp_path / "step-drift").mkdir()
    baseline_path, candidate_path = _write_reports(
        tmp_path / "step-drift", baseline, candidate
    )
    with pytest.raises(ValueError, match="optimizer-step budgets differ"):
        compare_kbo_scale_reports(baseline_path, candidate_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda _base, cand: cand.update(status="failed"), "not completed"),
        (
            lambda _base, cand: cand.update(dataset_fingerprint="other"),
            "dataset_fingerprint differ",
        ),
        (
            lambda _base, cand: cand.update(validation_season=2024),
            "split/seed lineage disagrees",
        ),
        (
            lambda _base, cand: cand.update(
                test_used_for_training_selection_or_comparison=True
            ),
            "held-out test stayed sealed",
        ),
        (
            lambda _base, cand: cand["training_config"].update(match_weight=2.0),
            "training_config.match_weight differ",
        ),
        (
            lambda _base, cand: cand["training_config"].update(max_pa_per_day=10),
            "training_config.max_pa_per_day differ",
        ),
        (
            lambda _base, cand: cand["variant_policies"]["full"].update(
                graph_control="changed"
            ),
            "variant graph policies differ",
        ),
        (
            lambda _base, cand: cand["runtime_signature"].update(precision="fp16"),
            "runtime signatures differ",
        ),
        (
            lambda _base, cand: cand["training_config"].update(new_optimizer=True),
            "unaccounted fields",
        ),
    ],
)
def test_scale_comparison_rejects_lineage_or_policy_drift(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    baseline, candidate = _reports()
    mutation(baseline, candidate)
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match=match):
        compare_kbo_scale_reports(baseline_path, candidate_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda _base, cand: cand["training_config"].update(hidden_dim=512),
            "must use hidden_dim=256",
        ),
        (
            lambda _base, cand: cand["training_config"].update(heads=4),
            "must use heads=8",
        ),
        (
            lambda base, _cand: base["training_config"].update(heads=8),
            "must use heads=4",
        ),
        (
            lambda _base, cand: cand["runs"]["full"].update(
                test_used_during_training=True
            ),
            "test stayed sealed",
        ),
        (
            lambda _base, cand: cand["runs"]["node_only"].update(
                completed_epochs=29
            ),
            "configured epochs",
        ),
        (
            lambda _base, cand: cand["runs"]["node_only"].update(
                attempted_optimizer_steps=899,
                optimizer_steps=899,
            ),
            "training budgets differ",
        ),
        (
            lambda _base, cand: cand["runs"]["full"]["validation_metrics"][
                "loss_sample_counts"
            ].update(match=4199),
            "candidate full/node_only validation sample counts differ",
        ),
        (
            lambda _base, cand: cand["runs"]["full"].update(
                validation_selection_loss=float("nan")
            ),
            "finite number",
        ),
        (
            lambda _base, cand: cand["validation_selection_comparison"].update(
                full=4.0
            ),
            "disagrees with its runs",
        ),
        (
            lambda _base, cand: cand["parameter_count_audit"].update(
                variant_parameter_counts={"full": 999, "node_only": 1_800_000}
            ),
            "parameter audit disagrees",
        ),
    ],
)
def test_scale_comparison_rejects_run_or_capacity_contract_drift(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    baseline, candidate = _reports()
    mutation(baseline, candidate)
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match=match):
        compare_kbo_scale_reports(baseline_path, candidate_path)


def test_candidate_must_explicitly_declare_activation_checkpointing(
    tmp_path: Path,
) -> None:
    baseline, candidate = _reports()
    del candidate["training_config"]["activation_checkpointing"]
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match="activation_checkpointing is missing"):
        compare_kbo_scale_reports(baseline_path, candidate_path)


@pytest.mark.parametrize("value", [False, "true"])
def test_scale_comparison_rejects_compact_or_invalid_architecture_flag(
    tmp_path: Path, value: Any
) -> None:
    baseline, candidate = _reports()
    candidate["training_config"]["compact_kbo_channels"] = value
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match="compact_kbo_channels"):
        compare_kbo_scale_reports(baseline_path, candidate_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda candidate: candidate["runs"]["node_only"].update(
            initial_model_state_sha256="d" * 64
        ),
        lambda candidate: candidate["initialization_audit"]["variants"][
            "node_only"
        ].update(initial_model_state_sha256="d" * 64),
        lambda candidate: candidate["initialization_audit"].update(seed=2027),
    ],
)
def test_scale_comparison_rejects_initialization_lineage_drift(
    tmp_path: Path, mutation: Any
) -> None:
    baseline, candidate = _reports()
    mutation(candidate)
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match="initialization"):
        compare_kbo_scale_reports(baseline_path, candidate_path)


def test_output_must_not_overwrite_source_report(tmp_path: Path) -> None:
    baseline, candidate = _reports()
    baseline_path, candidate_path = _write_reports(tmp_path, baseline, candidate)

    with pytest.raises(ValueError, match="must not overwrite"):
        compare_kbo_scale_reports(baseline_path, candidate_path, baseline_path)
