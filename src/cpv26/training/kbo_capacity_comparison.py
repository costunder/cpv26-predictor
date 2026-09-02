"""Single-seed 64x2 versus 128x3 RelGNN capacity comparison.

The completed 64x2 ``full`` and ``node_only`` children are reused from an
existing matched-ablation suite.  Only the two 128x3 children are trained.
Every comparison remains validation-only; the held-out test split is metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from cpv26.data.kbo_graph_dataset import KBOGraphDataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import KBORelGNNModel
from cpv26.training import kbo_matched_ablation as matched
from cpv26.training import kbo_runner as runner

CAPACITY_COMPARISON_PROTOCOL = "single_seed_validation_capacity_comparison"
CAPACITY_COMPARISON_PROTOCOL_VERSION = 1
CAPACITY_COMPARISON_VARIANTS = ("full", "node_only")
BASELINE_CAPACITY = {"hidden_dim": 64, "layers": 2}
EXPANDED_CAPACITY = {"hidden_dim": 128, "layers": 3}
CAPACITY_COMPARISON_REPORT = "capacity_comparison_report.json"
CAPACITY_COMPARISON_MANIFEST = "capacity_comparison_config.json"
FULL_NODE_COMPARISON_PROTOCOL = "single_seed_validation_full_node_comparison"
FULL_NODE_COMPARISON_PROTOCOL_VERSION = 1
FULL_NODE_COMPARISON_REPORT = "full_node_comparison_report.json"
FULL_NODE_COMPARISON_MANIFEST = "full_node_comparison_config.json"


@dataclass(frozen=True)
class _BaselineSuite:
    seed: int
    config: runner.KBOTrainingConfig
    runs: dict[str, dict[str, Any]]
    lineage: dict[str, Any]
    split_day_fingerprint: str
    runtime_signature: dict[str, Any]


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _single_seed(report: Mapping[str, Any]) -> int:
    seeds = report.get("seeds")
    runs = report.get("runs")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 1
        or isinstance(seeds[0], bool)
        or not isinstance(seeds[0], int)
        or seeds[0] < 0
        or not isinstance(runs, Mapping)
        or list(runs) != [str(seeds[0])]
    ):
        raise ValueError("baseline matched suite must contain exactly one training seed")
    return int(seeds[0])


def _baseline_run_directory(
    suite_directory: Path,
    seed: int,
    variant: str,
    child: Mapping[str, Any],
) -> Path:
    local = suite_directory / f"seed-{seed}" / variant
    if (local / "training_report.json").is_file():
        return local
    saved = child.get("run_directory")
    if not isinstance(saved, str) or not saved:
        raise ValueError(f"baseline {seed}/{variant} has no run-directory lineage")
    resolved = Path(saved).expanduser().resolve()
    if not (resolved / "training_report.json").is_file():
        raise FileNotFoundError(
            f"baseline {seed}/{variant} training report is unavailable: {resolved}"
        )
    return resolved


def _require_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _validate_baseline_child(
    *,
    suite_directory: Path,
    seed: int,
    variant: str,
    child: Mapping[str, Any],
    dataset_fingerprint: str,
    expected_config: runner.KBOTrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = f"baseline {seed}/{variant}"
    run_directory = _baseline_run_directory(suite_directory, seed, variant, child)
    training_path = run_directory / "training_report.json"
    training = _load_json(training_path)
    matched._validate_child_report(training, expected_config)
    matched._training_history_summary(training, context=context)
    if training.get("status") != "completed":
        raise ValueError(f"{context} is not completed")
    if training.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError(f"{context} uses a different dataset fingerprint")
    if int(training.get("completed_epochs", -1)) != expected_config.epochs:
        raise ValueError(f"{context} did not receive the complete fixed epoch budget")

    for field in (
        "completed_epochs",
        "attempted_optimizer_steps",
        "parameter_count",
        "initial_model_state_sha256",
    ):
        if child.get(field) != training.get(field):
            raise ValueError(f"{context} report and training lineage differ at {field}")
    if child.get("test_used_during_training") is not False:
        raise ValueError(f"{context} does not prove that held-out test stayed sealed")
    if (
        child.get("route_message_normalization")
        != expected_config.route_message_normalization
        or child.get("route_schedule_preset") != expected_config.route_schedule
    ):
        raise ValueError(f"{context} route-policy lineage is mislabeled")
    child_control = child.get("graph_control")
    if (
        not isinstance(child_control, Mapping)
        or child_control.get("mode") != expected_config.graph_control
    ):
        raise ValueError(f"{context} graph-control lineage is mislabeled")
    if int(child.get("best_epoch", -1)) != int(training.get("best_epoch", -2)):
        raise ValueError(f"{context} best-epoch lineage differs")

    best_checkpoint = run_directory / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"{context} best checkpoint is unavailable")
    best_hash = sha256_file(best_checkpoint)
    if (
        child.get("best_checkpoint_sha256") != best_hash
        or training.get("best_checkpoint_sha256") != best_hash
    ):
        raise ValueError(f"{context} best-checkpoint lineage is stale")
    selection_loss = _require_number(
        child.get("validation_selection_loss"),
        context=f"{context} validation selection loss",
    )
    metrics = child.get("validation_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{context} validation metrics are unavailable")
    metric_selection = _require_number(
        metrics.get("selection_loss"),
        context=f"{context} metric selection loss",
    )
    if not math.isclose(selection_loss, metric_selection, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{context} validation selection-loss lineage differs")

    normalized = _plain(child)
    normalized.update(
        run_directory=str(run_directory),
        best_checkpoint=str(best_checkpoint),
        best_checkpoint_sha256=best_hash,
    )
    lineage = {
        "run_directory": str(run_directory),
        "training_report": str(training_path),
        "training_report_sha256": sha256_file(training_path),
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": best_hash,
    }
    return normalized, lineage


def _validate_baseline_suite(
    dataset: KBOGraphDataset,
    dataset_directory: Path,
    suite_directory: Path,
) -> _BaselineSuite:
    report_path = suite_directory / "matched_retraining_report.json"
    manifest_path = suite_directory / "suite_config.json"
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    if report.get("status") != "completed":
        raise ValueError("baseline matched suite must be completed")
    if report.get("protocol") != "matched_from_scratch_validation_graph_ablation":
        raise ValueError("baseline is not a matched validation graph-ablation suite")
    if report.get("protocol_version") not in {
        None,
        matched.MATCHED_ABLATION_PROTOCOL_VERSION,
    }:
        raise ValueError("baseline matched-suite protocol version is unsupported")
    if report.get("selection_split") != "validation":
        raise ValueError("baseline matched suite did not select on validation")
    if report.get("test_used_for_training_selection_or_comparison") is not False:
        raise ValueError("baseline matched suite does not prove that held-out test stayed sealed")

    fingerprint = dataset.manifest["fingerprint"]
    if (
        report.get("dataset_fingerprint") != fingerprint
        or manifest.get("dataset_fingerprint") != fingerprint
    ):
        raise ValueError("baseline suite and requested graph dataset fingerprints differ")
    seed = _single_seed(report)
    if manifest.get("seeds") != [seed]:
        raise ValueError("baseline suite manifest does not have the same single seed")
    report_variants = report.get("variants")
    manifest_variants = manifest.get("variants")
    if not isinstance(report_variants, list) or not set(
        CAPACITY_COMPARISON_VARIANTS
    ).issubset(set(report_variants)):
        raise ValueError("baseline suite is missing full or node_only")
    if not isinstance(manifest_variants, list) or not set(
        CAPACITY_COMPARISON_VARIANTS
    ).issubset(set(manifest_variants)):
        raise ValueError("baseline suite manifest is missing full or node_only")

    raw_config = report.get("base_training_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("baseline suite has no base training configuration")
    config = replace(runner.KBOTrainingConfig.from_dict(raw_config), seed=seed)
    if config.hidden_dim != BASELINE_CAPACITY["hidden_dim"] or config.layers != BASELINE_CAPACITY[
        "layers"
    ]:
        raise ValueError("baseline matched suite must use the completed 64x2 model")
    if config.patience != 0:
        raise ValueError("baseline suite must use patience=0 for a fixed epoch budget")
    expected_base_policy = {
        "route_message_normalization": "none",
        "route_schedule": "full",
        "graph_control": "intact",
    }
    for field, policy_value in expected_base_policy.items():
        if getattr(config, field) != policy_value:
            raise ValueError(f"baseline base configuration must use {field}={policy_value}")

    expected_manifest_config = asdict(config)
    for field in (
        "seed",
        "route_message_normalization",
        "route_schedule",
        "graph_control",
    ):
        expected_manifest_config.pop(field)
    if manifest.get("base_training_config") != _plain(expected_manifest_config):
        raise ValueError("baseline suite manifest and report training configurations differ")
    if manifest.get("test_policy") != "held_out_metadata_only_never_loaded_or_evaluated":
        raise ValueError("baseline suite manifest does not seal held-out test")
    report_runtime = report.get("runtime_signature")
    manifest_runtime = manifest.get("runtime_signature")
    if not isinstance(report_runtime, Mapping) or not isinstance(
        manifest_runtime, Mapping
    ):
        raise ValueError("baseline suite runtime signature is unavailable")
    if _plain(report_runtime) != _plain(manifest_runtime):
        raise ValueError("baseline suite runtime signature lineage differs")

    actual_split_fingerprint, actual_split_days = matched._split_day_fingerprint(dataset, config)
    if (
        report.get("split_day_fingerprint") != actual_split_fingerprint
        or manifest.get("split_day_fingerprint") != actual_split_fingerprint
    ):
        raise ValueError("baseline suite train/validation split fingerprint differs")
    if report.get("split_days") != actual_split_days:
        raise ValueError("baseline suite train/validation day lineage differs")
    if (
        report.get("training_seasons") != list(config.train_seasons)
        or report.get("validation_season") != config.validation_season
    ):
        raise ValueError("baseline suite season lineage differs")
    if report.get("held_out_test_season") != config.test_season:
        raise ValueError("baseline held-out test-season lineage differs")

    raw_runs = report.get("runs")
    assert isinstance(raw_runs, Mapping)
    per_seed = raw_runs[str(seed)]
    if not isinstance(per_seed, Mapping):
        raise ValueError("baseline single-seed run record is malformed")
    runs: dict[str, dict[str, Any]] = {}
    child_lineage: dict[str, Any] = {}
    for variant in CAPACITY_COMPARISON_VARIANTS:
        child = per_seed.get(variant)
        if not isinstance(child, Mapping):
            raise ValueError(f"baseline suite has no completed {variant} child")
        expected_child = matched._variant_config(config, variant, seed)
        runs[variant], child_lineage[variant] = _validate_baseline_child(
            suite_directory=suite_directory,
            seed=seed,
            variant=variant,
            child=child,
            dataset_fingerprint=fingerprint,
            expected_config=expected_child,
        )

    completed = {int(run["completed_epochs"]) for run in runs.values()}
    attempts = {int(run["attempted_optimizer_steps"]) for run in runs.values()}
    counts = {int(run["parameter_count"]) for run in runs.values()}
    initial_states = {str(run["initial_model_state_sha256"]) for run in runs.values()}
    if len(completed) != 1 or len(attempts) != 1:
        raise ValueError("baseline full and node_only did not receive the same fixed budget")
    if len(counts) != 1 or len(initial_states) != 1:
        raise ValueError("baseline full and node_only do not have matched initialization")

    lineage = {
        "suite_directory": str(suite_directory),
        "dataset_directory": str(dataset_directory),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "suite_manifest": str(manifest_path),
        "suite_manifest_sha256": sha256_file(manifest_path),
        "seed": seed,
        "children": child_lineage,
    }
    return _BaselineSuite(
        seed=seed,
        config=config,
        runs=runs,
        lineage=lineage,
        split_day_fingerprint=actual_split_fingerprint,
        runtime_signature=_plain(report_runtime),
    )


def _validate_candidate_config(
    config: runner.KBOTrainingConfig,
    baseline: runner.KBOTrainingConfig,
) -> None:
    expected = replace(
        baseline,
        hidden_dim=EXPANDED_CAPACITY["hidden_dim"],
        layers=EXPANDED_CAPACITY["layers"],
    )
    actual_values = _plain(asdict(config))
    expected_values = _plain(asdict(expected))
    if actual_values == expected_values:
        return
    differences = sorted(
        key for key in expected_values if actual_values.get(key) != expected_values[key]
    )
    joined = ", ".join(differences) if differences else "unknown fields"
    raise ValueError(
        "capacity comparison must inherit the baseline seed, split, budget, optimizer, "
        f"loss, sampling, and runtime settings; unexpected fields: {joined}"
    )


def _variant_protocols(config: runner.KBOTrainingConfig) -> dict[str, dict[str, Any]]:
    protocols: dict[str, dict[str, Any]] = {}
    for variant in CAPACITY_COMPARISON_VARIANTS:
        variant_config = matched._variant_config(config, variant, config.seed)
        schedule = runner._resolved_route_schedule(variant_config)
        protocols[variant] = {
            "route_message_normalization": variant_config.route_message_normalization,
            "route_schedule": variant_config.route_schedule,
            "graph_control": variant_config.graph_control,
            "resolved_route_schedule": (
                None if schedule is None else [list(layer) for layer in schedule]
            ),
            "graph_control_protocol": runner._graph_control_report(variant_config),
        }
    return protocols


def _loader_lineage(
    *,
    dataset_fingerprint: str,
    split_day_fingerprint: str,
    split_days: Mapping[str, Any],
    config: runner.KBOTrainingConfig,
) -> dict[str, Any]:
    protocol = {
        "dataset_fingerprint": dataset_fingerprint,
        "split_day_fingerprint": split_day_fingerprint,
        "split_days": _plain(split_days),
        "seed": config.seed,
        "chronological": config.chronological,
        "batch_days": config.batch_days,
        "workers": config.workers,
        "accumulate_steps": config.accumulate_steps,
        "max_days_per_split": config.max_days_per_split,
        "max_pa_per_day": config.max_pa_per_day,
        "max_edges_per_route_per_day": config.max_edges_per_route_per_day,
        "graph_control": "intact",
        "graph_control_seed": config.graph_control_seed,
    }
    encoded = json.dumps(
        _plain(protocol), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return {
        **protocol,
        "fingerprint": fingerprint,
        "variant_fingerprints": {
            variant: fingerprint for variant in CAPACITY_COMPARISON_VARIANTS
        },
        "all_non_route_settings_equal": True,
    }


def _two_variant_initialization_audit(
    dataset: KBOGraphDataset,
    config: runner.KBOTrainingConfig,
) -> dict[str, Any]:
    """Audit exactly full/node_only without constructing unrelated ablation models."""

    torch, _ = require_torch()
    variants: dict[str, dict[str, Any]] = {}
    reference_hash: str | None = None
    reference_count: int | None = None
    for variant in CAPACITY_COMPARISON_VARIANTS:
        variant_config = matched._variant_config(config, variant, config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        model = KBORelGNNModel(runner._model_config(dataset, variant_config))
        state_hash = runner._model_state_sha256(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        variants[variant] = {
            "initial_model_state_sha256": state_hash,
            "parameter_count": parameter_count,
        }
        if reference_hash is None:
            reference_hash = state_hash
            reference_count = parameter_count
        elif state_hash != reference_hash or parameter_count != reference_count:
            raise ValueError(
                "full and node_only do not share identical initialization and parameter count"
            )
        del model
    assert reference_hash is not None and reference_count is not None
    return {
        "seed": config.seed,
        "all_variants_equal": True,
        "initial_model_state_sha256": reference_hash,
        "parameter_count": reference_count,
        "variants": variants,
    }


def _comparison_manifest(
    *,
    dataset_directory: Path,
    dataset_fingerprint: str,
    baseline: _BaselineSuite,
    config: runner.KBOTrainingConfig,
    runtime_signature: Mapping[str, Any],
    variant_protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol": CAPACITY_COMPARISON_PROTOCOL,
        "protocol_version": CAPACITY_COMPARISON_PROTOCOL_VERSION,
        "dataset_directory": str(dataset_directory),
        "dataset_fingerprint": dataset_fingerprint,
        "split_day_fingerprint": baseline.split_day_fingerprint,
        "seed": baseline.seed,
        "variants": list(CAPACITY_COMPARISON_VARIANTS),
        "baseline_capacity": dict(BASELINE_CAPACITY),
        "expanded_capacity": dict(EXPANDED_CAPACITY),
        "baseline_suite_lineage": _plain(baseline.lineage),
        "baseline_runtime_signature": _plain(baseline.runtime_signature),
        "training_config": _plain(asdict(config)),
        "runtime_signature": _plain(runtime_signature),
        "variant_policies": _plain(variant_protocols),
        "test_policy": "held_out_metadata_only_never_loaded_or_evaluated",
    }


def _validate_or_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if not path.exists():
        runner._atomic_json(path, manifest)
        return
    saved = _load_json(path)
    if saved != _plain(manifest):
        raise ValueError(
            "capacity comparison resume changes its dataset, baseline lineage, or fairness settings"
        )


def _matched_variant_run(
    *,
    dataset: KBOGraphDataset,
    dataset_directory: Path,
    output: Path,
    run_group: str,
    label: str,
    variant: str,
    config: runner.KBOTrainingConfig,
    initialization: Mapping[str, Any],
    variant_protocol: Mapping[str, Any],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    run_directory = output / run_group / variant
    prefix = f"[{label}/{variant}] "

    def child_progress(message: str) -> None:
        progress(prefix + message)

    training = matched._train_or_resume_child(
        dataset,
        dataset_directory,
        run_directory,
        config,
        initialization,
        child_progress,
    )
    matched._verify_initialization_lineage(
        run_directory,
        training,
        dataset,
        config,
        initialization,
    )
    validation = matched._reevaluate_best_on_validation(run_directory, dataset_directory, config)
    if validation.get("split") != "validation":
        raise ValueError("capacity comparison received a non-validation evaluation")
    if training.get("test_used_during_training") is not False:
        raise ValueError("capacity comparison child does not prove that test stayed sealed")
    expected_smoke = config.max_days_per_split is not None
    saved_smoke = training.get("smoke_test_only")
    if saved_smoke is not None and saved_smoke is not expected_smoke:
        raise ValueError("capacity comparison child smoke-test lineage is mislabeled")
    metrics = validation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("capacity comparison validation metrics are unavailable")
    selection_loss = _require_number(
        metrics.get("selection_loss"),
        context=f"{label}/{variant} validation selection loss",
    )
    history_summary = matched._training_history_summary(
        training, context=f"{label}/{variant}"
    )
    return {
        "run_directory": str(run_directory),
        "best_checkpoint": str(run_directory / "best.pt"),
        "best_checkpoint_sha256": sha256_file(run_directory / "best.pt"),
        "best_epoch": int(training["best_epoch"]),
        "completed_epochs": int(training["completed_epochs"]),
        "optimizer_steps": int(training["optimizer_steps"]),
        "skipped_optimizer_steps": int(training["skipped_optimizer_steps"]),
        "attempted_optimizer_steps": int(training["attempted_optimizer_steps"]),
        "parameter_count": int(training["parameter_count"]),
        "initial_model_state_sha256": str(training["initial_model_state_sha256"]),
        "graph_control": runner._graph_control_report(config),
        "route_message_normalization": config.route_message_normalization,
        "route_schedule_preset": config.route_schedule,
        "resolved_route_schedule": variant_protocol["resolved_route_schedule"],
        "variant_policy": _plain(variant_protocol),
        "validation_selection_loss": selection_loss,
        **history_summary,
        "final_minus_best_selection_loss": (
            float(history_summary["final_validation_selection_loss"]) - selection_loss
        ),
        "validation_metrics": _plain(metrics),
        "validation_output_directory": str(validation["output_directory"]),
        "test_used_during_training": False,
        "smoke_test_only": expected_smoke,
    }


def _selection_comparison(
    baseline_runs: Mapping[str, Mapping[str, Any]],
    expanded_runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {
        variant: _require_number(
            baseline_runs[variant].get("validation_selection_loss"),
            context=f"64x2/{variant} validation selection loss",
        )
        for variant in CAPACITY_COMPARISON_VARIANTS
    }
    expanded = {
        variant: _require_number(
            expanded_runs[variant].get("validation_selection_loss"),
            context=f"128x3/{variant} validation selection loss",
        )
        for variant in CAPACITY_COMPARISON_VARIANTS
    }
    baseline_gap = baseline["node_only"] - baseline["full"]
    expanded_gap = expanded["node_only"] - expanded["full"]
    return {
        "lower_is_better": True,
        "baseline_64x2": {
            "full": baseline["full"],
            "node_only": baseline["node_only"],
            "node_only_minus_full": baseline_gap,
        },
        "expanded_128x3": {
            "full": expanded["full"],
            "node_only": expanded["node_only"],
            "node_only_minus_full": expanded_gap,
        },
        "expanded_minus_baseline": {
            variant: expanded[variant] - baseline[variant]
            for variant in CAPACITY_COMPARISON_VARIANTS
        },
        "dependency_gap_change_128x3_minus_64x2": expanded_gap - baseline_gap,
        "interpretation": (
            "positive node_only_minus_full favors relational messages on validation; "
            "this single-seed comparison does not establish stability"
        ),
    }


def _budget_audit(
    baseline_runs: Mapping[str, Mapping[str, Any]],
    expanded_runs: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    groups = {
        "baseline_64x2": baseline_runs,
        "expanded_128x3": expanded_runs,
    }
    completed: dict[str, dict[str, int]] = {}
    attempted: dict[str, dict[str, int]] = {}
    for name, runs in groups.items():
        completed[name] = {
            variant: int(runs[variant]["completed_epochs"])
            for variant in CAPACITY_COMPARISON_VARIANTS
        }
        attempted[name] = {
            variant: int(runs[variant]["attempted_optimizer_steps"])
            for variant in CAPACITY_COMPARISON_VARIANTS
        }
    all_completed = set(completed["baseline_64x2"].values()) | set(
        completed["expanded_128x3"].values()
    )
    all_attempted = set(attempted["baseline_64x2"].values()) | set(
        attempted["expanded_128x3"].values()
    )
    if len(all_completed) != 1 or len(all_attempted) != 1:
        raise ValueError(
            "64x2 and 128x3 full/node_only runs did not receive the same fixed budget"
        )
    return {
        "seed": seed,
        "all_runs_equal": True,
        "completed_epochs": completed,
        "attempted_optimizer_steps": attempted,
    }


def _validate_full_node_config(config: runner.KBOTrainingConfig) -> None:
    if config.patience != 0:
        raise ValueError(
            "full/node comparison requires patience=0 so both conditions receive the "
            "same fixed epoch budget"
        )
    expected = {
        "route_message_normalization": "none",
        "route_schedule": "full",
        "graph_control": "intact",
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"full/node base configuration must use {field}={value}")


def _full_node_manifest(
    *,
    dataset_directory: Path,
    dataset_fingerprint: str,
    split_day_fingerprint: str,
    config: runner.KBOTrainingConfig,
    runtime_signature: Mapping[str, Any],
    variant_protocols: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol": FULL_NODE_COMPARISON_PROTOCOL,
        "protocol_version": FULL_NODE_COMPARISON_PROTOCOL_VERSION,
        "dataset_directory": str(dataset_directory),
        "dataset_fingerprint": dataset_fingerprint,
        "split_day_fingerprint": split_day_fingerprint,
        "seed": config.seed,
        "variants": list(CAPACITY_COMPARISON_VARIANTS),
        "training_config": _plain(asdict(config)),
        "runtime_signature": _plain(runtime_signature),
        "variant_policies": _plain(variant_protocols),
        "test_policy": "held_out_metadata_only_never_loaded_or_evaluated",
    }


def _full_node_budget_audit(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    completed = {
        variant: int(runs[variant]["completed_epochs"])
        for variant in CAPACITY_COMPARISON_VARIANTS
    }
    attempted = {
        variant: int(runs[variant]["attempted_optimizer_steps"])
        for variant in CAPACITY_COMPARISON_VARIANTS
    }
    if len(set(completed.values())) != 1 or len(set(attempted.values())) != 1:
        raise ValueError("full and node_only did not receive the same fixed training budget")
    return {
        "all_variants_equal": True,
        "completed_epochs": completed,
        "attempted_optimizer_steps": attempted,
    }


def _validation_count_audit(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int] | None] = {}
    for variant in CAPACITY_COMPARISON_VARIANTS:
        metrics = runs[variant].get("validation_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{variant} validation metrics are unavailable")
        raw = metrics.get("loss_sample_counts")
        if raw is None:
            counts[variant] = None
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"{variant} validation loss-sample counts are malformed")
        normalized: dict[str, int] = {}
        for task, value in raw.items():
            if (
                not isinstance(task, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{variant} validation loss-sample counts are malformed")
            normalized[task] = value
        counts[variant] = normalized
    available = [counts[variant] is not None for variant in CAPACITY_COMPARISON_VARIANTS]
    if any(available) and not all(available):
        raise ValueError("full and node_only have inconsistent validation count schemas")
    if all(available) and counts["full"] != counts["node_only"]:
        raise ValueError("full and node_only validation sample counts differ")
    return {
        "available": all(available),
        "all_variants_equal": all(available),
        "loss_sample_counts": counts["full"] if all(available) else None,
    }


def _full_node_selection_comparison(
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    full = _require_number(
        runs["full"].get("validation_selection_loss"),
        context="full validation selection loss",
    )
    node_only = _require_number(
        runs["node_only"].get("validation_selection_loss"),
        context="node_only validation selection loss",
    )
    return {
        "lower_is_better": True,
        "full": full,
        "node_only": node_only,
        "node_only_minus_full": node_only - full,
        "interpretation": (
            "positive node_only_minus_full favors relational messages on validation; "
            "this single-seed comparison does not establish stability"
        ),
    }


def train_kbo_full_node_comparison(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    config: runner.KBOTrainingConfig,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train exactly full and node_only once, comparing validation only."""

    _validate_full_node_config(config)
    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    split_fingerprint, split_days = matched._split_day_fingerprint(dataset, config)
    runtime_signature = matched._runtime_signature(config)
    variant_protocols = _variant_protocols(config)
    manifest = _full_node_manifest(
        dataset_directory=directory,
        dataset_fingerprint=dataset.manifest["fingerprint"],
        split_day_fingerprint=split_fingerprint,
        config=config,
        runtime_signature=runtime_signature,
        variant_protocols=variant_protocols,
    )
    manifest_path = output / FULL_NODE_COMPARISON_MANIFEST
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(
            "full/node output directory is non-empty and has no comparison manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(manifest_path, manifest)

    initialization = _two_variant_initialization_audit(dataset, config)
    report: dict[str, Any] = {
        "status": "running",
        "protocol": FULL_NODE_COMPARISON_PROTOCOL,
        "protocol_version": FULL_NODE_COMPARISON_PROTOCOL_VERSION,
        "output_directory": str(output),
        "dataset_directory": str(directory),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "split_day_fingerprint": split_fingerprint,
        "split_days": split_days,
        "training_seasons": list(config.train_seasons),
        "validation_season": config.validation_season,
        "held_out_test_season": config.test_season,
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "smoke_test_only": config.max_days_per_split is not None,
        "seed": config.seed,
        "variants": list(CAPACITY_COMPARISON_VARIANTS),
        "capacity": {"hidden_dim": config.hidden_dim, "layers": config.layers},
        "runtime_signature": runtime_signature,
        "training_config": _plain(asdict(config)),
        "variant_policies": variant_protocols,
        "initialization_audit": _plain(initialization),
        "loader_lineage": _loader_lineage(
            dataset_fingerprint=dataset.manifest["fingerprint"],
            split_day_fingerprint=split_fingerprint,
            split_days=split_days,
            config=config,
        ),
        "runs": {},
        "limitations": [
            "This is one fixed training seed and cannot establish seed stability.",
            "node_only retains graph-derived node and role features while removing messages.",
            "Validation selects checkpoints and compares conditions; held-out test is "
            "never loaded.",
            "No multi-seed or additional ablation variants are implemented by this protocol.",
        ],
    }
    report_path = output / FULL_NODE_COMPARISON_REPORT
    runner._atomic_json(report_path, report)

    try:
        runs: dict[str, dict[str, Any]] = {}
        for variant in CAPACITY_COMPARISON_VARIANTS:
            variant_config = matched._variant_config(config, variant, config.seed)
            runs[variant] = _matched_variant_run(
                dataset=dataset,
                dataset_directory=directory,
                output=output,
                run_group=f"seed-{config.seed}",
                label=str(config.seed),
                variant=variant,
                config=variant_config,
                initialization=initialization,
                variant_protocol=variant_protocols[variant],
                progress=progress,
            )
            report["runs"] = runs
            runner._atomic_json(report_path, report)

        parameter_counts = {int(run["parameter_count"]) for run in runs.values()}
        initial_states = {
            str(run["initial_model_state_sha256"]) for run in runs.values()
        }
        if len(parameter_counts) != 1 or len(initial_states) != 1:
            raise ValueError("full and node_only do not have matched initialization")
        full_loss = float(runs["full"]["validation_selection_loss"])
        for run in runs.values():
            run["selection_loss_delta_vs_full"] = (
                float(run["validation_selection_loss"]) - full_loss
            )
        report["parameter_count_audit"] = {
            "all_variants_equal": True,
            "parameter_count": next(iter(parameter_counts)),
        }
        report["budget_audit"] = _full_node_budget_audit(runs)
        report["validation_sample_count_audit"] = _validation_count_audit(runs)
        report["validation_selection_comparison"] = _full_node_selection_comparison(runs)
        report["status"] = "completed"
        runner._atomic_json(report_path, report)
    except Exception:
        report["status"] = "failed"
        runner._atomic_json(report_path, report)
        raise
    return report


def train_kbo_capacity_comparison(
    dataset_directory: str | Path,
    baseline_suite_directory: str | Path,
    output_directory: str | Path,
    *,
    config: runner.KBOTrainingConfig,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Reuse one 64x2 suite and train only 128x3 full/node_only on validation."""

    directory = Path(dataset_directory).expanduser().resolve()
    baseline_directory = Path(baseline_suite_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output == baseline_directory:
        raise ValueError("capacity output directory must differ from the baseline suite")
    dataset = KBOGraphDataset(directory)
    baseline = _validate_baseline_suite(dataset, directory, baseline_directory)
    _validate_candidate_config(config, baseline.config)
    split_fingerprint, split_days = matched._split_day_fingerprint(dataset, config)
    if split_fingerprint != baseline.split_day_fingerprint:
        raise ValueError("128x3 candidate uses a different train/validation split")

    runtime_signature = matched._runtime_signature(config)
    if _plain(runtime_signature) != baseline.runtime_signature:
        raise ValueError(
            "128x3 candidate runtime differs from the reused 64x2 baseline runtime"
        )
    variant_protocols = _variant_protocols(config)
    manifest = _comparison_manifest(
        dataset_directory=directory,
        dataset_fingerprint=dataset.manifest["fingerprint"],
        baseline=baseline,
        config=config,
        runtime_signature=runtime_signature,
        variant_protocols=variant_protocols,
    )
    manifest_path = output / CAPACITY_COMPARISON_MANIFEST
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(
            "capacity output directory is non-empty and has no comparison manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(manifest_path, manifest)

    initialization = _two_variant_initialization_audit(dataset, config)
    report: dict[str, Any] = {
        "status": "running",
        "protocol": CAPACITY_COMPARISON_PROTOCOL,
        "protocol_version": CAPACITY_COMPARISON_PROTOCOL_VERSION,
        "output_directory": str(output),
        "dataset_directory": str(directory),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "split_day_fingerprint": split_fingerprint,
        "split_days": split_days,
        "training_seasons": list(config.train_seasons),
        "validation_season": config.validation_season,
        "held_out_test_season": config.test_season,
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "smoke_test_only": config.max_days_per_split is not None,
        "seed": baseline.seed,
        "variants": list(CAPACITY_COMPARISON_VARIANTS),
        "baseline_capacity": dict(BASELINE_CAPACITY),
        "expanded_capacity": dict(EXPANDED_CAPACITY),
        "baseline_suite_lineage": _plain(baseline.lineage),
        "baseline_runtime_signature": _plain(baseline.runtime_signature),
        "runtime_signature": runtime_signature,
        "training_config": _plain(asdict(config)),
        "variant_policies": variant_protocols,
        "initialization_audit": _plain(initialization),
        "loader_lineage": {
            **_loader_lineage(
                dataset_fingerprint=dataset.manifest["fingerprint"],
                split_day_fingerprint=split_fingerprint,
                split_days=split_days,
                config=config,
            ),
            "capacities": ["baseline_64x2", "expanded_128x3"],
            "all_capacities_equal": True,
        },
        "runs": {
            "baseline_64x2": _plain(baseline.runs),
            "expanded_128x3": {},
        },
        "limitations": [
            "This is one fixed training seed and cannot establish seed stability.",
            "The baseline 64x2 children are reused and are not retrained.",
            "Validation selects checkpoints and compares capacities; held-out test is "
            "never loaded.",
            "No multi-seed expansion is implemented by this protocol.",
        ],
    }
    report_path = output / CAPACITY_COMPARISON_REPORT
    runner._atomic_json(report_path, report)

    try:
        expanded_runs: dict[str, dict[str, Any]] = {}
        for variant in CAPACITY_COMPARISON_VARIANTS:
            variant_config = matched._variant_config(config, variant, baseline.seed)
            expanded_runs[variant] = _matched_variant_run(
                dataset=dataset,
                dataset_directory=directory,
                output=output,
                run_group="expanded-128x3",
                label="128x3",
                variant=variant,
                config=variant_config,
                initialization=initialization,
                variant_protocol=variant_protocols[variant],
                progress=progress,
            )
            report["runs"]["expanded_128x3"] = expanded_runs
            runner._atomic_json(report_path, report)

        baseline_parameter_count = int(baseline.runs["full"]["parameter_count"])
        expanded_counts = {int(run["parameter_count"]) for run in expanded_runs.values()}
        expanded_states = {
            str(run["initial_model_state_sha256"]) for run in expanded_runs.values()
        }
        if len(expanded_counts) != 1 or len(expanded_states) != 1:
            raise ValueError("128x3 full and node_only do not have matched initialization")
        expanded_parameter_count = next(iter(expanded_counts))
        if expanded_parameter_count <= baseline_parameter_count:
            raise ValueError("128x3 model does not have more parameters than the 64x2 baseline")

        report["parameter_count_audit"] = {
            "baseline_64x2": baseline_parameter_count,
            "expanded_128x3": expanded_parameter_count,
            "increase": expanded_parameter_count - baseline_parameter_count,
            "within_capacity_variants_equal": True,
        }
        report["budget_audit"] = _budget_audit(
            baseline.runs,
            expanded_runs,
            seed=baseline.seed,
        )
        report["validation_selection_comparison"] = _selection_comparison(
            baseline.runs,
            expanded_runs,
        )
        report["status"] = "completed"
        runner._atomic_json(report_path, report)
    except Exception:
        report["status"] = "failed"
        runner._atomic_json(report_path, report)
        raise
    return report


__all__ = [
    "BASELINE_CAPACITY",
    "CAPACITY_COMPARISON_MANIFEST",
    "CAPACITY_COMPARISON_PROTOCOL",
    "CAPACITY_COMPARISON_PROTOCOL_VERSION",
    "CAPACITY_COMPARISON_REPORT",
    "CAPACITY_COMPARISON_VARIANTS",
    "EXPANDED_CAPACITY",
    "FULL_NODE_COMPARISON_MANIFEST",
    "FULL_NODE_COMPARISON_PROTOCOL",
    "FULL_NODE_COMPARISON_PROTOCOL_VERSION",
    "FULL_NODE_COMPARISON_REPORT",
    "train_kbo_capacity_comparison",
    "train_kbo_full_node_comparison",
]
