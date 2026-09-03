from __future__ import annotations

import inspect
import json
import shutil
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
        {
            "epoch": 1,
            "global_step": 5,
            "skipped_optimizer_steps": 0,
            "validation": {"selection_loss": loss + 0.1},
        },
        {
            "epoch": 2,
            "global_step": 10,
            "skipped_optimizer_steps": 0,
            "validation": {"selection_loss": loss},
        },
    ]
    return {
        "status": "completed",
        "configuration": asdict(config),
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "graph_control": runner._graph_control_report(config),
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
        "last_checkpoint_sha256": sha256_file(checkpoint.with_name("last.pt")),
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
        (run / "last.pt").write_bytes(f"baseline-{variant}-last".encode("ascii"))
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
        lambda dataset, config: _initialization_fixture(config),
    )


def _initialization_fixture(config: runner.KBOTrainingConfig) -> dict[str, Any]:
    initial_hash = "baseline-initial" if config.hidden_dim == 64 else "expanded-initial"
    parameter_count = 100 if config.hidden_dim == 64 else 400
    return {
        "seed": config.seed,
        "all_variants_equal": True,
        "initial_model_state_sha256": initial_hash,
        "parameter_count": parameter_count,
        "variants": {
            variant: {
                "initial_model_state_sha256": initial_hash,
                "parameter_count": parameter_count,
            }
            for variant in capacity.CAPACITY_COMPARISON_VARIANTS
        },
    }


def _orphan_selected_seed(suite: Path, base: runner.KBOTrainingConfig) -> None:
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report["runs"] = {}
    report["initialization_audit"] = {str(SEED): _initialization_fixture(base)}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    for variant, loss in (("full", 1.0), ("node_only", 1.2)):
        run = suite / f"seed-{SEED}" / variant
        checkpoint_hash = sha256_file(run / "best.pt")
        output = run / "matched_validation" / checkpoint_hash[:16]
        evaluation = _validation_report_fixture(
            output,
            checkpoint_hash=checkpoint_hash,
            config=matched._variant_config(base, variant, SEED),
            loss=loss,
        )
        (output / "metrics.json").write_text(json.dumps(evaluation), encoding="utf-8")


def _prediction_fixture(
    output: Path,
    *,
    loss: float,
    null_tasks: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {"selection_loss": loss}
    artifacts: dict[str, Any] = {}
    for task in capacity.RECOVERY_PREDICTION_TASKS:
        if task in null_tasks:
            metrics[task] = None
            continue
        count_field = (
            "samples" if task in {"match", "live_hit", "pa"} else "player_game_queries"
        )
        metrics[task] = {"fixture_metric": loss, count_field: 1}
        target = output / f"{task}_predictions.parquet"
        target.write_bytes(f"{task}-predictions".encode("ascii"))
        artifacts[task] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "rows": 1,
        }
    return metrics, artifacts


def _validation_report_fixture(
    output: Path,
    *,
    checkpoint_hash: str,
    config: runner.KBOTrainingConfig,
    loss: float,
    null_tasks: tuple[str, ...] = (),
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    metrics, prediction_artifacts = _prediction_fixture(
        output,
        loss=loss,
        null_tasks=null_tasks,
    )
    return {
        "split": "validation",
        "checkpoint_sha256": checkpoint_hash,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "training_seasons": list(config.train_seasons),
        "validation_season": config.validation_season,
        "held_out_test_season": config.test_season,
        "graph_control": runner._graph_control_report(config),
        "metrics": metrics,
        "prediction_artifacts": prediction_artifacts,
        "output_directory": str(output),
    }


def _patch_orphan_checkpoint_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    def checkpoint_state(path: Path) -> dict[str, Any]:
        training = json.loads(
            (path.parent / "training_report.json").read_text(encoding="utf-8")
        )
        epoch = (
            int(training["best_epoch"])
            if path.name == "best.pt"
            else int(training["completed_epochs"])
        )
        return {
            "epoch": epoch,
            "best_epoch": training["best_epoch"],
            "best_score": training["best_validation_loss"],
            "global_step": training["optimizer_steps"],
            "skipped_optimizer_steps": training["skipped_optimizer_steps"],
            "history": training["history"][:epoch],
        }

    monkeypatch.setattr(
        capacity.runner,
        "_read_checkpoint",
        checkpoint_state,
    )
    monkeypatch.setattr(matched, "_validate_child_checkpoint", lambda *a, **k: None)


def _remove_orphan_validation_cache(suite: Path) -> None:
    for variant in capacity.CAPACITY_COMPARISON_VARIANTS:
        shutil.rmtree(suite / f"seed-{SEED}" / variant / "matched_validation")


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


def test_multi_seed_baseline_requires_explicit_selection_and_sealed_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seeds"] = [SEED, SEED + 1]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = suite / "suite_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seeds"] = [SEED, SEED + 1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="declares multiple seeds"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            progress=lambda _: None,
        )

    report["test_used_for_training_selection_or_comparison"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out test stayed sealed"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_running_baseline_is_rejected_even_when_selected_seed_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "running"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="still running"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_failed_partial_suite_requires_both_selected_seed_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report["runs"][str(SEED)].pop("node_only")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="no completed node_only child"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "output",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_failed_suite_recovers_complete_orphan_children_without_baseline_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    original_report = sha256_file(suite / "matched_retraining_report.json")
    original_manifest = sha256_file(suite / "suite_config.json")
    _patch_orphan_checkpoint_validation(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch)

    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        tmp_path / "capacity",
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )

    assert report["status"] == "completed"
    assert trained == evaluated == ["full", "node_only"]
    baseline_runs = report["runs"]["baseline_64x2"]
    assert {
        run["baseline_record_source"] for run in baseline_runs.values()
    } == {"recovered_from_complete_local_child_artifacts"}
    assert all(
        child["recovered"]
        for child in report["baseline_suite_lineage"]["children"].values()
    )
    assert sha256_file(suite / "matched_retraining_report.json") == original_report
    assert sha256_file(suite / "suite_config.json") == original_manifest
    assert not (tmp_path / "capacity" / "baseline-validation-recovery").exists()


def test_orphan_recovery_reports_child_availability_without_fallback_from_partial_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    (suite / f"seed-{SEED}" / "node_only" / "training_report.json").unlink()

    with pytest.raises(
        ValueError,
        match=r"cannot be recovered: .*node_only/training_report.json.*node_only=partial",
    ):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "capacity",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_orphan_recovery_allows_missing_last_but_rejects_present_bad_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    last_checkpoint = suite / f"seed-{SEED}" / "node_only" / "last.pt"
    last_checkpoint.unlink()
    _patch_orphan_checkpoint_validation(monkeypatch)
    _patch_candidate_runs(monkeypatch)
    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        tmp_path / "missing-last",
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )
    assert report["status"] == "completed"
    assert (
        report["baseline_suite_lineage"]["children"]["node_only"]["recovered"]
        is True
    )
    assert "last_checkpoint" not in report["runs"]["baseline_64x2"]["node_only"]

    last_checkpoint.write_bytes(b"restored-but-different")
    with pytest.raises(ValueError, match="last checkpoint hash differs"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "wrong-last-hash",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_recovered_validation_accepts_null_tasks_only_without_artifacts(
    tmp_path: Path,
) -> None:
    config = matched._variant_config(_baseline_config(), "full", SEED)
    checkpoint_hash = "c" * 64
    output = tmp_path / "validation"
    report = _validation_report_fixture(
        output,
        checkpoint_hash=checkpoint_hash,
        config=config,
        loss=1.0,
        null_tasks=("live_hit",),
    )
    report["metrics"]["box_pa"] = None

    validated = capacity._validated_recovery_evaluation(
        output / "metrics.json",
        report=report,
        checkpoint_hash=checkpoint_hash,
        dataset_fingerprint=DATASET_FINGERPRINT,
        config=config,
        context="fixture",
    )

    assert set(validated["prediction_artifacts"]) == {
        "match",
        "pa",
        "box_pa",
        "box_pitch",
    }


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "missing required prediction artifacts"),
        ("zero_rows", "match artifact lineage is malformed"),
        ("row_mismatch", "match artifact row count differs"),
        ("external_path", "match artifact path is not canonical"),
        ("bad_sha", "match artifact hash differs"),
    ),
)
def test_recovered_validation_rejects_incomplete_prediction_artifact_lineage(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    config = matched._variant_config(_baseline_config(), "full", SEED)
    checkpoint_hash = "c" * 64
    output = tmp_path / corruption
    report = _validation_report_fixture(
        output,
        checkpoint_hash=checkpoint_hash,
        config=config,
        loss=1.0,
    )
    artifacts = report["prediction_artifacts"]
    if corruption == "missing":
        artifacts.pop("match")
    elif corruption == "zero_rows":
        artifacts["match"]["rows"] = 0
    elif corruption == "row_mismatch":
        artifacts["match"]["rows"] = 2
    elif corruption == "external_path":
        external = tmp_path / "external.parquet"
        external.write_bytes(b"match-predictions")
        artifacts["match"]["path"] = str(external)
    else:
        artifacts["match"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match=message):
        capacity._validated_recovery_evaluation(
            output / "metrics.json",
            report=report,
            checkpoint_hash=checkpoint_hash,
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=config,
            context="fixture",
        )


def test_existing_manifest_never_writes_missing_orphan_validation_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    _remove_orphan_validation_cache(suite)
    _patch_orphan_checkpoint_validation(monkeypatch)
    output = tmp_path / "capacity"
    output.mkdir()
    manifest = output / capacity.CAPACITY_COMPARISON_MANIFEST
    manifest.write_text(json.dumps({"stale": True}), encoding="utf-8")
    original_manifest_hash = sha256_file(manifest)
    evaluations: list[Path] = []

    def forbidden_evaluation(checkpoint: Path, **kwargs: Any) -> dict[str, Any]:
        evaluations.append(checkpoint)
        raise AssertionError("existing-manifest recovery must be read-only")

    monkeypatch.setattr(capacity.runner, "evaluate_kbo_relgnn", forbidden_evaluation)

    with pytest.raises(ValueError, match="refusing to modify"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            output,
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )

    assert evaluations == []
    assert sha256_file(manifest) == original_manifest_hash
    assert not (output / "baseline-validation-recovery").exists()


def test_fresh_output_may_create_missing_orphan_validation_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    _remove_orphan_validation_cache(suite)
    _patch_orphan_checkpoint_validation(monkeypatch)
    trained, candidate_evaluated = _patch_candidate_runs(monkeypatch)
    baseline_evaluated: list[str] = []

    def fake_baseline_evaluation(
        checkpoint: Path,
        *,
        split: str,
        output_directory: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert split == "validation"
        variant = checkpoint.parent.name
        baseline_evaluated.append(variant)
        loss = 1.0 if variant == "full" else 1.2
        return _validation_report_fixture(
            output_directory,
            checkpoint_hash=sha256_file(checkpoint),
            config=matched._variant_config(base, variant, SEED),
            loss=loss,
        )

    monkeypatch.setattr(capacity.runner, "evaluate_kbo_relgnn", fake_baseline_evaluation)
    output = tmp_path / "capacity"

    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        output,
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )

    assert report["status"] == "completed"
    assert baseline_evaluated == ["full", "node_only"]
    assert trained == candidate_evaluated == ["full", "node_only"]
    assert (output / capacity.CAPACITY_COMPARISON_MANIFEST).is_file()
    assert (output / "baseline-validation-recovery").is_dir()


def test_selected_orphan_seed_recovers_when_another_seed_record_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    other_seed = SEED + 1
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seeds"] = [other_seed, SEED]
    report["runs"] = {str(other_seed): {"full": {"survived": True}}}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = suite / "suite_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seeds"] = [other_seed, SEED]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _patch_orphan_checkpoint_validation(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch)

    result = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        tmp_path / "capacity",
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )

    assert result["seed"] == SEED
    assert trained == evaluated == ["full", "node_only"]
    assert all(
        child["recovered"]
        for child in result["baseline_suite_lineage"]["children"].values()
    )


def test_null_selected_seed_record_is_malformed_and_never_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runs"] = {str(SEED): None}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="selected-seed run record is malformed"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "capacity",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_orphan_recovery_records_top_level_drift_and_rejects_test_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    report_path = suite / "matched_retraining_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["initialization_audit"][str(SEED)]["initial_model_state_sha256"] = "changed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _patch_orphan_checkpoint_validation(monkeypatch)
    _patch_candidate_runs(monkeypatch)
    recovered = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        tmp_path / "changed-init",
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )
    initialization = recovered["baseline_suite_lineage"][
        "orphan_initialization_audit"
    ]
    assert (
        initialization["top_level_snapshot"]["hash_matches_child_consensus"]
        is False
    )

    _orphan_selected_seed(suite, base)
    checkpoint = suite / f"seed-{SEED}" / "full" / "best.pt"
    metrics_path = (
        checkpoint.parent
        / "matched_validation"
        / sha256_file(checkpoint)[:16]
        / "metrics.json"
    )
    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation["split"] = "test"
    metrics_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="not validation-only"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "test-evaluation",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_orphan_recovery_allows_current_initialization_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)

    def drifted_initialization(dataset: Any, config: runner.KBOTrainingConfig) -> dict[str, Any]:
        value = _initialization_fixture(config)
        if config.hidden_dim == 64:
            value["initial_model_state_sha256"] = "current-code-initial"
            for variant in capacity.CAPACITY_COMPARISON_VARIANTS:
                value["variants"][variant][
                    "initial_model_state_sha256"
                ] = "current-code-initial"
        return value

    monkeypatch.setattr(
        capacity,
        "_two_variant_initialization_audit",
        drifted_initialization,
    )
    _patch_orphan_checkpoint_validation(monkeypatch)
    trained, evaluated = _patch_candidate_runs(monkeypatch)

    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        tmp_path / "capacity",
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
        progress=lambda _: None,
    )

    audit = report["baseline_suite_lineage"]["orphan_initialization_audit"]
    assert audit["authority"] == "completed_full_and_node_only_child_consensus"
    assert audit["current_reproduction"]["hash_matches_child_consensus"] is False
    assert audit["current_reproduction"][
        "parameter_count_matches_child_consensus"
    ] is True
    assert trained == evaluated == ["full", "node_only"]


def test_orphan_recovery_rejects_child_initialization_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)
    training_path = suite / f"seed-{SEED}" / "node_only" / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["initial_model_state_sha256"] = "different-child-initial"
    training_path.write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(ValueError, match="do not share historical initialization consensus"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "capacity",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_orphan_recovery_rejects_current_parameter_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    _patch_protocol_runtime(monkeypatch)
    _orphan_selected_seed(suite, base)

    def incompatible_initialization(
        dataset: Any, config: runner.KBOTrainingConfig
    ) -> dict[str, Any]:
        value = _initialization_fixture(config)
        if config.hidden_dim == 64:
            value["parameter_count"] = 101
            for variant in capacity.CAPACITY_COMPARISON_VARIANTS:
                value["variants"][variant]["parameter_count"] = 101
        return value

    monkeypatch.setattr(
        capacity,
        "_two_variant_initialization_audit",
        incompatible_initialization,
    )
    with pytest.raises(ValueError, match="parameter count is incompatible"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            tmp_path / "capacity",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
            progress=lambda _: None,
        )


def test_capacity_output_cannot_be_inside_baseline_suite(tmp_path: Path) -> None:
    graph, suite, base = _write_baseline(tmp_path)
    with pytest.raises(ValueError, match="outside the baseline suite"):
        capacity.train_kbo_capacity_comparison(
            graph,
            suite,
            suite / "child-output",
            config=replace(base, hidden_dim=128, layers=3),
            baseline_seed=SEED,
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
    report_path = suite / "matched_retraining_report.json"
    baseline_report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline_report["status"] = "failed"
    baseline_report["seeds"] = [SEED, SEED + 1]
    report_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    manifest_path = suite / "suite_config.json"
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_manifest["seeds"] = [SEED, SEED + 1]
    manifest_path.write_text(json.dumps(baseline_manifest), encoding="utf-8")

    report = capacity.train_kbo_capacity_comparison(
        graph,
        suite,
        output,
        config=replace(base, hidden_dim=128, layers=3),
        baseline_seed=SEED,
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
