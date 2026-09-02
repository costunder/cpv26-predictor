from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.training import kbo_capacity_comparison as capacity
from cpv26.training import kbo_matched_ablation as matched
from cpv26.training import kbo_runner as runner

DATASET_FINGERPRINT = "d" * 64
SPLIT_FINGERPRINT = "s" * 64
SEED = 71


def _baseline_config() -> runner.KBOTrainingConfig:
    return runner.KBOTrainingConfig(
        device="cpu",
        amp="off",
        workers=0,
        epochs=2,
        batch_days=2,
        hidden_dim=64,
        layers=2,
        heads=4,
        dropout=0.0,
        patience=0,
        max_pa_per_day=0,
        max_edges_per_route_per_day=0,
        seed=SEED,
        graph_control_seed=991,
        train_seasons=(2023,),
        validation_season=2024,
        test_season=2025,
    )


def _training_report(
    config: runner.KBOTrainingConfig,
    checkpoint: Path,
    *,
    loss: float,
    parameter_count: int = 100,
    initial_hash: str = "baseline-initial",
) -> dict[str, Any]:
    history = [
        {"epoch": 1, "validation": {"selection_loss": loss + 0.1}},
        {"epoch": 2, "validation": {"selection_loss": loss}},
    ]
    return {
        "status": "completed",
        "configuration": asdict(config),
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "test_used_during_training": False,
        "completed_epochs": 2,
        "best_epoch": 2,
        "best_validation_loss": loss,
        "optimizer_steps": 10,
        "skipped_optimizer_steps": 0,
        "attempted_optimizer_steps": 10,
        "parameter_count": parameter_count,
        "initial_model_state_sha256": initial_hash,
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "last_checkpoint_sha256": "unused-in-this-fixture",
        "history": history,
    }


def _write_baseline(tmp_path: Path) -> tuple[Path, Path, runner.KBOTrainingConfig]:
    graph = tmp_path / "graph"
    graph.mkdir()
    suite = tmp_path / "baseline"
    suite.mkdir()
    base = _baseline_config()
    runs: dict[str, dict[str, Any]] = {}
    for variant, loss in (("full", 1.0), ("node_only", 1.2)):
        run = suite / f"seed-{SEED}" / variant
        run.mkdir(parents=True)
        checkpoint = run / "best.pt"
        checkpoint.write_bytes(f"baseline-{variant}".encode("ascii"))
        config = matched._variant_config(base, variant, SEED)
        training = _training_report(config, checkpoint, loss=loss)
        (run / "training_report.json").write_text(
            json.dumps(training), encoding="utf-8"
        )
        runs[variant] = {
            "run_directory": str(run),
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": sha256_file(checkpoint),
            "best_epoch": 2,
            "completed_epochs": 2,
            "attempted_optimizer_steps": 10,
            "parameter_count": 100,
            "initial_model_state_sha256": "baseline-initial",
            "route_message_normalization": config.route_message_normalization,
            "route_schedule_preset": config.route_schedule,
            "graph_control": runner._graph_control_report(config),
            "validation_selection_loss": loss,
            "validation_metrics": {"selection_loss": loss},
            "test_used_during_training": False,
        }
    manifest_config = asdict(base)
    for field in (
        "seed",
        "route_message_normalization",
        "route_schedule",
        "graph_control",
    ):
        manifest_config.pop(field)
    suite_manifest = {
        "protocol_version": matched.MATCHED_ABLATION_PROTOCOL_VERSION,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_day_fingerprint": SPLIT_FINGERPRINT,
        "seeds": [SEED],
        "variants": list(capacity.CAPACITY_COMPARISON_VARIANTS),
        "base_training_config": manifest_config,
        "test_policy": "held_out_metadata_only_never_loaded_or_evaluated",
        "runtime_signature": {"device": "cpu", "precision": "off"},
    }
    (suite / "suite_config.json").write_text(
        json.dumps(suite_manifest), encoding="utf-8"
    )
    report = {
        "status": "completed",
        "protocol": "matched_from_scratch_validation_graph_ablation",
        "protocol_version": matched.MATCHED_ABLATION_PROTOCOL_VERSION,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_day_fingerprint": SPLIT_FINGERPRINT,
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "held_out_test_season": 2025,
        "training_seasons": [2023],
        "validation_season": 2024,
        "split_days": {"train": ["2023-04-01"], "validation": ["2024-04-01"]},
        "seeds": [SEED],
        "variants": list(capacity.CAPACITY_COMPARISON_VARIANTS),
        "base_training_config": asdict(base),
        "runtime_signature": {"device": "cpu", "precision": "off"},
        "runs": {str(SEED): runs},
    }
    (suite / "matched_retraining_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return graph, suite, base


class _FakeDataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = {"fingerprint": DATASET_FINGERPRINT}


def _patch_protocol_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capacity, "KBOGraphDataset", _FakeDataset)
    monkeypatch.setattr(
        matched,
        "_split_day_fingerprint",
        lambda dataset, config: (
            SPLIT_FINGERPRINT,
            {"train": ["2023-04-01"], "validation": ["2024-04-01"]},
        ),
    )
    monkeypatch.setattr(
        matched,
        "_runtime_signature",
        lambda config: {"device": config.device, "precision": config.amp},
    )
    monkeypatch.setattr(
        capacity,
        "_two_variant_initialization_audit",
        lambda dataset, config: {
            "seed": config.seed,
            "all_variants_equal": True,
            "initial_model_state_sha256": "expanded-initial",
            "parameter_count": 400,
            "variants": {
                variant: {
                    "initial_model_state_sha256": "expanded-initial",
                    "parameter_count": 400,
                }
                for variant in capacity.CAPACITY_COMPARISON_VARIANTS
            },
        },
    )


def _patch_candidate_runs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_variant: str | None = None,
) -> tuple[list[str], list[str]]:
    trained: list[str] = []
    evaluated: list[str] = []

    def fake_train(
        dataset: Any,
        dataset_directory: Path,
        run_directory: Path,
        config: runner.KBOTrainingConfig,
        initialization: dict[str, Any],
        progress: Any,
    ) -> dict[str, Any]:
        variant = "node_only" if config.route_schedule == "node_only" else "full"
        trained.append(variant)
        if variant == fail_variant:
            raise RuntimeError("simulated candidate interruption")
        run_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = run_directory / "best.pt"
        checkpoint.write_bytes(f"expanded-{variant}".encode("ascii"))
        (run_directory / "last.pt").write_bytes(b"resumable")
        loss = 0.8 if variant == "full" else 0.9
        training = _training_report(
            config,
            checkpoint,
            loss=loss,
            parameter_count=400,
            initial_hash="expanded-initial",
        )
        (run_directory / "training_report.json").write_text(
            json.dumps(training), encoding="utf-8"
        )
        return training

    def fake_evaluate(
        run_directory: Path,
        dataset_directory: Path,
        config: runner.KBOTrainingConfig,
    ) -> dict[str, Any]:
        variant = run_directory.name
        evaluated.append(variant)
        loss = 0.8 if variant == "full" else 0.9
        output = run_directory / "matched_validation" / "fixture"
        output.mkdir(parents=True, exist_ok=True)
        return {
            "split": "validation",
            "output_directory": str(output),
            "metrics": {"selection_loss": loss},
        }

    monkeypatch.setattr(matched, "_train_or_resume_child", fake_train)
    monkeypatch.setattr(matched, "_verify_initialization_lineage", lambda *a, **k: None)
    monkeypatch.setattr(matched, "_reevaluate_best_on_validation", fake_evaluate)
    return trained, evaluated


def test_candidate_config_allows_only_the_128x3_capacity_change() -> None:
    baseline = _baseline_config()
    candidate = replace(baseline, hidden_dim=128, layers=3)
    capacity._validate_candidate_config(candidate, baseline)

    with pytest.raises(ValueError, match="unexpected fields: batch_days"):
        capacity._validate_candidate_config(replace(candidate, batch_days=1), baseline)
    with pytest.raises(ValueError, match="unexpected fields: hidden_dim"):
        capacity._validate_candidate_config(replace(candidate, hidden_dim=256), baseline)
    assert "seeds" not in inspect.signature(capacity.train_kbo_capacity_comparison).parameters


def test_baseline_requires_exactly_one_completed_seed_and_sealed_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seeds"] = [SEED, SEED + 1]
    report["runs"][str(SEED + 1)] = report["runs"][str(SEED)]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one training seed"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            progress=lambda _: None,
        )

    report["seeds"] = [SEED]
    report["runs"].pop(str(SEED + 1))
    report["test_used_for_training_selection_or_comparison"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out test stayed sealed"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            progress=lambda _: None,
        )


def test_trains_only_128x3_full_and_node_only_and_reuses_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch)
    output = tmp_path / "capacity"

    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        output,
        config=replace(base, hidden_dim=128, layers=3),
        progress=lambda _: None,
    )

    assert report["status"] == "completed"
    assert trained == evaluated == ["full", "node_only"]
    assert report["seed"] == SEED
    assert report["selection_split"] == "validation"
    assert report["test_used_for_training_selection_or_comparison"] is False
    assert tuple(report["runs"]["baseline_64x2"]) == ("full", "node_only")
    assert tuple(report["runs"]["expanded_128x3"]) == ("full", "node_only")
    assert report["budget_audit"]["all_runs_equal"] is True
    assert report["parameter_count_audit"] == {
        "baseline_64x2": 100,
        "expanded_128x3": 400,
        "increase": 300,
        "within_capacity_variants_equal": True,
    }
    comparison = report["validation_selection_comparison"]
    assert comparison["baseline_64x2"]["node_only_minus_full"] == pytest.approx(0.2)
    assert comparison["expanded_128x3"]["node_only_minus_full"] == pytest.approx(0.1)
    assert comparison["dependency_gap_change_128x3_minus_64x2"] == pytest.approx(-0.1)
    saved = json.loads(
        (output / capacity.CAPACITY_COMPARISON_REPORT).read_text(encoding="utf-8")
    )
    assert saved == report


def test_resume_rejects_any_baseline_lineage_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _patch_candidate_runs(monkeypatch)
    output = tmp_path / "capacity"
    config = replace(base, hidden_dim=128, layers=3)
    capacity.train_kbo_capacity_comparison(
        graph, suite, output, config=config, progress=lambda _: None
    )

    report_path = suite / "matched_retraining_report.json"
    baseline_report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline_report["lineage_note"] = "content changed after comparison"
    report_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline lineage"):
        capacity.train_kbo_capacity_comparison(
            graph, suite, output, config=config, progress=lambda _: None
        )


def test_capacity_rejects_a_runtime_change_from_reused_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    monkeypatch.setattr(
        matched,
        "_runtime_signature",
        lambda config: {"device": "different-gpu", "precision": config.amp},
    )

    with pytest.raises(ValueError, match="candidate runtime differs"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "capacity",
            config=replace(base, hidden_dim=128, layers=3),
            progress=lambda _: None,
        )


def test_interruption_atomically_marks_report_failed_with_completed_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch, fail_variant="node_only")
    output = tmp_path / "capacity"

    with pytest.raises(RuntimeError, match="simulated candidate interruption"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            output,
            config=replace(base, hidden_dim=128, layers=3),
            progress=lambda _: None,
        )

    saved = json.loads(
        (output / capacity.CAPACITY_COMPARISON_REPORT).read_text(encoding="utf-8")
    )
    assert saved["status"] == "failed"
    assert tuple(saved["runs"]["expanded_128x3"]) == ("full",)
    assert trained == ["full", "node_only"]
    assert evaluated == ["full"]


def test_generic_full_node_runner_trains_exactly_two_conditions_and_seals_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = tmp_path / "graph"
    graph.mkdir()
    _patch_protocol_runtime(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch)
    config = _baseline_config()
    output = tmp_path / "full-node"

    report = capacity.train_kbo_full_node_comparison(
        graph,
        output,
        config=config,
        progress=lambda _: None,
    )

    assert report["status"] == "completed"
    assert report["protocol"] == capacity.FULL_NODE_COMPARISON_PROTOCOL
    assert report["seed"] == SEED
    assert report["variants"] == ["full", "node_only"]
    assert tuple(report["runs"]) == ("full", "node_only")
    assert trained == evaluated == ["full", "node_only"]
    assert report["selection_split"] == "validation"
    assert report["test_used_for_training_selection_or_comparison"] is False
    assert report["smoke_test_only"] is False
    assert report["initialization_audit"]["all_variants_equal"] is True
    assert report["budget_audit"]["all_variants_equal"] is True
    assert report["loader_lineage"]["all_non_route_settings_equal"] is True
    assert report["validation_selection_comparison"][
        "node_only_minus_full"
    ] == pytest.approx(0.1)
    assert report["runs"]["full"]["selection_loss_delta_vs_full"] == 0.0
    assert report["runs"]["node_only"][
        "selection_loss_delta_vs_full"
    ] == pytest.approx(0.1)
    assert "seeds" not in inspect.signature(
        capacity.train_kbo_full_node_comparison
    ).parameters
    assert (output / f"seed-{SEED}" / "full" / "best.pt").is_file()
    assert (output / f"seed-{SEED}" / "node_only" / "best.pt").is_file()
    saved = json.loads(
        (output / capacity.FULL_NODE_COMPARISON_REPORT).read_text(encoding="utf-8")
    )
    assert saved == report

    with pytest.raises(ValueError, match="fairness settings"):
        capacity.train_kbo_full_node_comparison(
            graph,
            output,
            config=replace(config, dropout=0.1),
            progress=lambda _: None,
        )


def test_generic_full_node_runner_requires_fixed_budget_and_marks_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _baseline_config()
    with pytest.raises(ValueError, match="patience=0"):
        capacity.train_kbo_full_node_comparison(
            tmp_path / "unused",
            tmp_path / "unused-output",
            config=replace(config, patience=1),
            progress=lambda _: None,
        )
    with pytest.raises(ValueError, match="route_schedule=full"):
        capacity.train_kbo_full_node_comparison(
            tmp_path / "unused",
            tmp_path / "unused-output",
            config=replace(config, route_schedule="core"),
            progress=lambda _: None,
        )

    graph = tmp_path / "graph"
    graph.mkdir()
    _patch_protocol_runtime(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch, fail_variant="node_only")
    output = tmp_path / "full-node"
    with pytest.raises(RuntimeError, match="simulated candidate interruption"):
        capacity.train_kbo_full_node_comparison(
            graph,
            output,
            config=config,
            progress=lambda _: None,
        )
    saved = json.loads(
        (output / capacity.FULL_NODE_COMPARISON_REPORT).read_text(encoding="utf-8")
    )
    assert saved["status"] == "failed"
    assert tuple(saved["runs"]) == ("full",)
    assert trained == ["full", "node_only"]
    assert evaluated == ["full"]
