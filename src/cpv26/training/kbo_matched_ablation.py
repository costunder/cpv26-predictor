"""Matched-from-scratch graph ablations selected and compared on validation only."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from cpv26.data.kbo_graph_dataset import KBOGraphDataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import KBORelGNNConfig, KBORelGNNModel
from cpv26.training import kbo_runner as runner

MATCHED_GRAPH_VARIANTS = (
    "full",
    "normalized",
    "staged",
    "core",
    "node_only",
    "rewired",
)
MATCHED_ABLATION_PROTOCOL_VERSION = 2
_LOSS_TASKS = ("match", "live_hit", "pa", "run", "box_pa", "box_pitch")
_NAMED_CONTRASTS = (
    {
        "name": "normalization",
        "candidate": "normalized",
        "reference": "full",
        "hypothesis": "route-message layer normalization with the full route schedule",
    },
    {
        "name": "staged_schedule",
        "candidate": "staged",
        "reference": "normalized",
        "hypothesis": "staged route schedule at fixed layer normalization",
    },
    {
        "name": "core_pruning",
        "candidate": "core",
        "reference": "normalized",
        "hypothesis": "core route pruning at fixed layer normalization",
    },
    {
        "name": "remove_messages",
        "candidate": "node_only",
        "reference": "full",
        "hypothesis": "removing relational messages from the raw full model",
    },
    {
        "name": "permute_endpoints",
        "candidate": "rewired",
        "reference": "full",
        "hypothesis": "permuting endpoint identities in the raw full model",
    },
)
_RUNTIME_SIGNATURE_FIELDS = (
    "device",
    "gpu_name",
    "total_memory_bytes",
    "compute_capability",
    "torch_version",
    "cuda_runtime",
    "precision",
)
_VARIANT_POLICIES: dict[str, dict[str, str]] = {
    "full": {
        "route_message_normalization": "none",
        "route_schedule": "full",
        "graph_control": "intact",
    },
    "normalized": {
        "route_message_normalization": "layer_norm",
        "route_schedule": "full",
        "graph_control": "intact",
    },
    "staged": {
        "route_message_normalization": "layer_norm",
        "route_schedule": "staged",
        "graph_control": "intact",
    },
    "core": {
        "route_message_normalization": "layer_norm",
        "route_schedule": "core",
        "graph_control": "intact",
    },
    "node_only": {
        "route_message_normalization": "none",
        "route_schedule": "node_only",
        "graph_control": "intact",
    },
    "rewired": {
        "route_message_normalization": "none",
        "route_schedule": "full",
        "graph_control": "permuted_endpoints",
    },
}


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))


def _variant_config(
    base: runner.KBOTrainingConfig, variant: str, seed: int
) -> runner.KBOTrainingConfig:
    try:
        policy = _VARIANT_POLICIES[variant]
    except KeyError as exc:
        raise ValueError(f"unknown matched graph variant: {variant}") from exc
    return replace(
        base,
        seed=seed,
        route_message_normalization=policy["route_message_normalization"],
        route_schedule=policy["route_schedule"],
        graph_control=policy["graph_control"],
    )


def _initialization_audit(
    dataset: KBOGraphDataset,
    base: runner.KBOTrainingConfig,
    seed: int,
) -> dict[str, Any]:
    torch, _ = require_torch()
    node_only_config = _variant_config(base, "node_only", seed)
    torch.manual_seed(seed)
    random.seed(seed)
    node_only_model: Any = KBORelGNNModel(
        runner._model_config(dataset, node_only_config)
    )
    shared_parameter_names = tuple(sorted(dict(node_only_model.named_parameters())))
    del node_only_model

    variants: dict[str, Any] = {}
    shared_hash: str | None = None
    for variant in MATCHED_GRAPH_VARIANTS:
        config = _variant_config(base, variant, seed)
        torch.manual_seed(seed)
        random.seed(seed)
        model: Any = KBORelGNNModel(runner._model_config(dataset, config))
        state_hash = runner._model_state_sha256(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        variant_shared_hash = runner._parameter_state_sha256(
            model, shared_parameter_names
        )
        variants[variant] = {
            "initial_model_state_sha256": state_hash,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "shared_parameter_initialization_sha256": variant_shared_hash,
            "architecture": model.architecture_contract(),
        }
        if shared_hash is None:
            shared_hash = variant_shared_hash
        elif variant_shared_hash != shared_hash:
            raise ValueError(
                "matched variants do not share identical common-parameter initialization: "
                f"seed={seed}, variant={variant}"
            )
        del model
    assert shared_hash is not None
    return {
        "seed": seed,
        "comparison_basis": (
            "same seed and identical initialization for every parameter shared by all "
            "variants; variant-specific relational capacity is reported, not padded"
        ),
        "all_shared_parameters_equal": True,
        "shared_parameter_tensors": len(shared_parameter_names),
        "shared_parameter_initialization_sha256": shared_hash,
        "variant_architectures_intentionally_distinct": True,
        "variants": variants,
    }


def _variant_protocols(
    base: runner.KBOTrainingConfig,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variant in MATCHED_GRAPH_VARIANTS:
        config = _variant_config(base, variant, base.seed)
        schedule = runner._resolved_route_schedule(config)
        result[variant] = {
            **_VARIANT_POLICIES[variant],
            "resolved_route_schedule": (
                None if schedule is None else [list(layer) for layer in schedule]
            ),
            "graph_control_protocol": runner._graph_control_report(config),
        }
    return result


def _split_day_fingerprint(
    dataset: KBOGraphDataset, config: runner.KBOTrainingConfig
) -> tuple[str, dict[str, list[str]]]:
    splits = runner._split_days(dataset, config)
    selected = {
        name: [day.isoformat() for day in splits[name]] for name in ("train", "validation")
    }
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest(), selected


def _runtime_signature(config: runner.KBOTrainingConfig) -> dict[str, Any]:
    """Resolve the numerical runtime without binding the suite to a MIG UUID."""

    _, _, runtime = runner._device_and_precision(config.device, config.amp)
    return {name: runtime.get(name) for name in _RUNTIME_SIGNATURE_FIELDS}


def _suite_manifest(
    dataset: KBOGraphDataset,
    directory: Path,
    base: runner.KBOTrainingConfig,
    seeds: tuple[int, ...],
    split_fingerprint: str,
    variant_protocols: Mapping[str, Mapping[str, Any]],
    runtime_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    configuration = asdict(base)
    for key in (
        "seed",
        "route_message_normalization",
        "route_schedule",
        "graph_control",
    ):
        configuration.pop(key)
    return {
        "protocol_version": MATCHED_ABLATION_PROTOCOL_VERSION,
        "dataset_directory": str(directory),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "split_day_fingerprint": split_fingerprint,
        "runtime_signature": _plain(
            runtime_signature if runtime_signature is not None else _runtime_signature(base)
        ),
        "seeds": list(seeds),
        "variants": list(MATCHED_GRAPH_VARIANTS),
        "variant_policies": _plain(variant_protocols),
        "base_training_config": configuration,
        "test_policy": "held_out_metadata_only_never_loaded_or_evaluated",
    }


def _normalized_legacy_execution_fields(
    value: Any, *, context: str
) -> dict[str, Any]:
    """Fill only execution switches absent from pre-scale artifacts."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} has no training configuration")
    normalized = dict(_plain(value))
    normalized.setdefault("activation_checkpointing", False)
    normalized.setdefault("compact_kbo_channels", False)
    normalized.setdefault("route_edge_chunk_size", 0)
    return normalized


def _normalized_manifest_training_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("matched suite manifest has no base training configuration")
    normalized = _normalized_legacy_execution_fields(
        value, context="matched suite manifest"
    )
    projected_fields = {
        "seed",
        "route_message_normalization",
        "route_schedule",
        "graph_control",
    }
    present_fields = projected_fields.intersection(normalized)
    if present_fields and present_fields != projected_fields:
        raise ValueError(
            "matched suite training configuration has a partial projected-policy field set"
        )
    for key in present_fields:
        normalized.pop(key)
    return dict(_plain(normalized))


def _validate_or_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if not path.exists():
        runner._atomic_json(path, manifest)
        return
    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    left, right = dict(saved), _plain(manifest)
    left["base_training_config"] = _normalized_manifest_training_config(
        left.get("base_training_config")
    )
    right["base_training_config"] = _normalized_manifest_training_config(
        right.get("base_training_config")
    )
    old_seeds = list(left.pop("seeds"))
    new_seeds = list(right.pop("seeds"))
    if new_seeds[: len(old_seeds)] != old_seeds:
        raise ValueError(
            "matched suite seeds can only be appended without removal or reordering"
        )
    old_epochs = int(left["base_training_config"].pop("epochs"))
    new_epochs = int(right["base_training_config"].pop("epochs"))
    if left != right:
        raise ValueError("matched suite resume changes a fairness setting other than epochs")
    if new_epochs < old_epochs:
        raise ValueError("matched suite target epochs cannot decrease")
    if new_epochs != old_epochs or new_seeds != old_seeds:
        runner._atomic_json(path, manifest)


def _validate_child_report(
    report: Mapping[str, Any], expected: runner.KBOTrainingConfig
) -> None:
    raw_actual = report.get("configuration")
    if not isinstance(raw_actual, Mapping):
        raise ValueError("existing matched child has no training configuration")
    actual = _normalized_legacy_execution_fields(
        raw_actual, context="existing matched child"
    )
    expected_values = _plain(asdict(expected))
    actual_epochs = int(actual.pop("epochs"))
    expected_epochs = int(expected_values.pop("epochs"))
    if actual != expected_values:
        raise ValueError("existing matched child run differs from the suite fairness settings")
    if actual_epochs > expected_epochs:
        raise ValueError("existing matched child run exceeds the requested target epochs")
    if report.get("test_used_during_training") is not False:
        raise ValueError("matched child report does not prove that test was held out")
    contract = report.get("parameter_contract")
    observed_gradient_coverage = report.get(
        "all_epochs_trainable_parameters_received_gradient"
    )
    if (
        not isinstance(contract, Mapping)
        or contract.get("optimizer_covers_all_trainable") is not True
        or not isinstance(observed_gradient_coverage, bool)
        or int(report.get("trainable_parameter_count", -1))
        != int(contract.get("trainable_parameter_count", -2))
    ):
        raise ValueError(
            "matched child does not prove optimizer coverage and observed gradient auditing"
        )


def _validate_child_checkpoint(
    state: Mapping[str, Any],
    *,
    dataset: KBOGraphDataset,
    expected: runner.KBOTrainingConfig,
    initialization: Mapping[str, Any],
) -> None:
    if state.get("dataset_fingerprint") != dataset.manifest["fingerprint"]:
        raise ValueError("interrupted matched child uses a different graph dataset")
    actual = _plain(asdict(runner.KBOTrainingConfig.from_dict(state["training_config"])))
    expected_values = _plain(asdict(expected))
    actual_epochs = int(actual.pop("epochs"))
    expected_epochs = int(expected_values.pop("epochs"))
    if actual != expected_values:
        raise ValueError(
            "interrupted matched child changes a fairness setting other than epochs"
        )
    if actual_epochs > expected_epochs or int(state["epoch"]) > expected_epochs:
        raise ValueError("interrupted matched child exceeds the requested target epochs")
    runner._validate_checkpoint_graph_control(state, expected)
    model_config = runner._model_config(dataset, expected).to_dict()
    if KBORelGNNConfig(**state["model_config"]).to_dict() != model_config:
        raise ValueError("interrupted matched child has a different model configuration")
    if (
        state.get("initial_model_state_sha256")
        != initialization["initial_model_state_sha256"]
        or int(state.get("parameter_count", -1)) != int(initialization["parameter_count"])
        or int(state.get("trainable_parameter_count", -1))
        != int(initialization["trainable_parameter_count"])
    ):
        raise ValueError("interrupted matched child has different initialization lineage")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _train_or_resume_child(
    dataset: KBOGraphDataset,
    dataset_directory: Path,
    run_directory: Path,
    config: runner.KBOTrainingConfig,
    initialization: Mapping[str, Any],
    progress: Callable[[str], None],
    temporal_preflight_report: str | Path | None = None,
) -> dict[str, Any]:
    training_report_path = run_directory / "training_report.json"
    previous: dict[str, Any] | None = None
    if training_report_path.is_file():
        previous = _load_json(training_report_path)
        _validate_child_report(previous, config)
        if int(previous["completed_epochs"]) >= config.epochs:
            return previous
    last_checkpoint = run_directory / "last.pt"
    resume = last_checkpoint if last_checkpoint.is_file() else None
    if resume is not None:
        _validate_child_checkpoint(
            runner._read_checkpoint(resume),
            dataset=dataset,
            expected=config,
            initialization=initialization,
        )
    if run_directory.exists() and any(run_directory.iterdir()) and resume is None:
        raise FileExistsError(
            f"incomplete matched child has no resumable last.pt: {run_directory}"
        )
    return runner.train_kbo_relgnn(
        dataset_directory,
        run_directory,
        config=config,
        resume=resume,
        temporal_preflight_report=temporal_preflight_report,
        progress=progress,
    )


def _reevaluate_best_on_validation(
    run_directory: Path,
    dataset_directory: Path,
    config: runner.KBOTrainingConfig,
    temporal_preflight_report: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = run_directory / "best.pt"
    checkpoint_hash = sha256_file(checkpoint)
    output = run_directory / "matched_validation" / checkpoint_hash[:16]
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        report = _load_json(metrics_path)
        if (
            report.get("split") != "validation"
            or report.get("checkpoint_sha256") != checkpoint_hash
        ):
            raise ValueError("cached matched validation report has different lineage")
        return report
    if output.exists():
        raise FileExistsError(f"partial matched validation output is not reusable: {output}")
    # Keep this sibling name shorter than the final hash directory so nested
    # atomic artifact names remain below legacy Windows path limits in tests.
    temporary = output.with_name(f".tmp-{uuid4().hex[:8]}")
    report = runner.evaluate_kbo_relgnn(
        checkpoint,
        dataset_directory=dataset_directory,
        split="validation",
        device=config.device,
        amp=config.amp,
        batch_days=config.batch_days,
        workers=config.workers,
        output_directory=temporary,
        temporal_preflight_report=temporal_preflight_report,
    )
    report["output_directory"] = str(output)
    for artifact in report.get("prediction_artifacts", {}).values():
        artifact["path"] = str(output / Path(artifact["path"]).name)
    runner._atomic_json(temporary / "metrics.json", report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return report


def _verify_initialization_lineage(
    run_directory: Path,
    training: Mapping[str, Any],
    dataset: KBOGraphDataset,
    config: runner.KBOTrainingConfig,
    expected: Mapping[str, Any],
) -> None:
    expected_hash = expected["initial_model_state_sha256"]
    expected_count = int(expected["parameter_count"])
    if (
        training.get("initial_model_state_sha256") != expected_hash
        or int(training.get("parameter_count", -1)) != expected_count
        or int(training.get("trainable_parameter_count", -1))
        != int(expected["trainable_parameter_count"])
    ):
        raise ValueError("matched training report does not match the audited initialization")
    for name, report_hash_key in (
        ("best.pt", "best_checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
    ):
        path = run_directory / name
        actual_hash = sha256_file(path)
        if training.get(report_hash_key) != actual_hash:
            raise ValueError(f"matched training report has a stale {name} hash")
        checkpoint = runner._read_checkpoint(path)
        _validate_child_checkpoint(
            checkpoint,
            dataset=dataset,
            expected=config,
            initialization=expected,
        )


def _aggregate_values(
    values: Sequence[float],
    deltas: Sequence[float],
    core_deltas: Sequence[float] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seeds": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "paired_delta_vs_full_mean": statistics.fmean(deltas),
        "paired_delta_vs_full_population_std": statistics.pstdev(deltas),
    }
    if core_deltas is not None:
        result.update(
            paired_delta_vs_core_mean=statistics.fmean(core_deltas),
            paired_delta_vs_core_population_std=statistics.pstdev(core_deltas),
        )
    return result


def _numeric_metric(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variant: str,
    path: Sequence[str],
) -> list[float] | None:
    values: list[float] = []
    unavailable: list[str] = []
    for seed, per_seed in runs.items():
        current: Any = per_seed[variant]
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is None:
            unavailable.append(seed)
            continue
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            joined = ".".join(path)
            raise ValueError(f"matched metric {variant}.{joined} is not numeric")
        values.append(float(current))
    if unavailable and values:
        joined = ".".join(path)
        raise ValueError(
            f"matched metric {variant}.{joined} is unavailable for only some seeds: "
            f"{', '.join(unavailable)}"
        )
    if unavailable:
        return None
    return values


def _paired_metric(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variant: str,
    path: Sequence[str],
) -> dict[str, Any] | None:
    values = _numeric_metric(runs, variant, path)
    full = _numeric_metric(runs, "full", path)
    core = _numeric_metric(runs, "core", path)
    available = (values is not None, full is not None, core is not None)
    if not any(available):
        return None
    if not all(available):
        joined = ".".join(path)
        raise ValueError(f"matched variants have inconsistent metric schema at {joined}")
    assert values is not None and full is not None and core is not None
    return _aggregate_values(
        values,
        [value - reference for value, reference in zip(values, full, strict=True)],
        [value - reference for value, reference in zip(values, core, strict=True)],
    )


def _aggregate_runs(runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_fields = (
        "log_loss",
        "accuracy",
        "expected_calibration_error",
        "brier_score",
    )
    for variant in MATCHED_GRAPH_VARIANTS:
        selection = _paired_metric(runs, variant, ("validation_selection_loss",))
        if selection is None:
            raise ValueError(f"matched run {variant!r} has no validation selection loss")
        task_metrics: dict[str, Any] = {}
        for task in ("match", "live_hit", "pa"):
            task_result: dict[str, Any] = {}
            for field in metric_fields:
                aggregate = _paired_metric(
                    runs,
                    variant,
                    ("validation_metrics", task, field),
                )
                if aggregate is not None:
                    task_result[field] = aggregate
            if task == "live_hit":
                for field in (
                    "joint_nll",
                    "observed_nll",
                    "partial_pa_nll",
                    "expected_hits_lower_bound_mae",
                    "expected_pa_lower_bound_mae",
                ):
                    aggregate = _paired_metric(
                        runs,
                        variant,
                        ("validation_metrics", task, field),
                    )
                    if aggregate is not None:
                        task_result[field] = aggregate
            task_metrics[task] = task_result

        validation_losses: dict[str, Any] = {}
        loss_sample_counts: dict[str, Any] = {}
        weighted_contributions: dict[str, Any] = {}
        for task in _LOSS_TASKS:
            aggregate = _paired_metric(
                runs,
                variant,
                ("validation_metrics", "losses", task),
            )
            if aggregate is not None:
                validation_losses[task] = aggregate
            count = _paired_metric(
                runs,
                variant,
                ("validation_metrics", "loss_sample_counts", task),
            )
            if count is not None:
                loss_sample_counts[task] = count
            contribution = _paired_metric(
                runs,
                variant,
                ("validation_metrics", "weighted_loss_contributions", task),
            )
            if contribution is not None:
                weighted_contributions[task] = contribution

        checkpoint_selection: dict[str, Any] = {}
        for name, path in (
            ("best_epoch", ("best_epoch",)),
            (
                "final_validation_selection_loss",
                ("final_validation_selection_loss",),
            ),
            (
                "final_minus_best_selection_loss",
                ("final_minus_best_selection_loss",),
            ),
            (
                "last_five_validation_selection_loss_mean",
                ("last_five_validation_selection_loss_mean",),
            ),
        ):
            aggregate = _paired_metric(runs, variant, path)
            if aggregate is not None:
                checkpoint_selection[name] = aggregate

        result[variant] = {
            "validation_selection_loss": selection,
            "validation_losses": validation_losses,
            "validation_loss_sample_counts": loss_sample_counts,
            "validation_weighted_loss_contributions": weighted_contributions,
            "validation_metrics": task_metrics,
            "checkpoint_selection": checkpoint_selection,
            "parameter_count": next(iter(runs.values()))[variant]["parameter_count"],
        }
    return result


def _training_history_summary(
    training: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if training.get("status") != "completed":
        raise ValueError(f"{context}: training report is not completed")
    history = training.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{context}: completed training history is unavailable")
    rows: list[tuple[int, float]] = []
    validation_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ValueError(f"{context}: history row {index} is malformed")
        epoch = row.get("epoch")
        validation = row.get("validation")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not isinstance(validation, Mapping)
            or isinstance(validation.get("selection_loss"), bool)
            or not isinstance(validation.get("selection_loss"), (int, float))
        ):
            raise ValueError(f"{context}: history row {index} has no epoch selection loss")
        rows.append((epoch, float(validation["selection_loss"])))
        validation_rows.append(validation)
    if [epoch for epoch, _ in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{context}: training history epochs are not contiguous")
    if int(training.get("completed_epochs", -1)) != len(rows):
        raise ValueError(f"{context}: completed epoch count differs from history")
    history_best_epoch, history_best_loss = min(rows, key=lambda item: item[1])
    if int(training.get("best_epoch", -1)) != history_best_epoch:
        raise ValueError(f"{context}: best epoch differs from the history argmin")
    reported_best = training.get("best_validation_loss")
    if (
        isinstance(reported_best, bool)
        or not isinstance(reported_best, (int, float))
        or not math.isclose(float(reported_best), history_best_loss, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError(f"{context}: best validation loss differs from the history argmin")
    tail = [loss for _, loss in rows[-5:]]
    result: dict[str, Any] = {
        "history_best_epoch": history_best_epoch,
        "history_best_validation_selection_loss": history_best_loss,
        "final_epoch": rows[-1][0],
        "final_validation_selection_loss": rows[-1][1],
        "last_five_validation_selection_loss_mean": statistics.fmean(tail),
    }
    for source_field, final_field, tail_field in (
        (
            "losses",
            "final_validation_losses",
            "last_five_validation_losses_mean",
        ),
        (
            "weighted_loss_contributions",
            "final_validation_weighted_loss_contributions",
            "last_five_validation_weighted_loss_contributions_mean",
        ),
    ):
        available = [isinstance(row.get(source_field), Mapping) for row in validation_rows]
        if not any(available):
            continue
        if not all(available):
            raise ValueError(f"{context}: history has a partial {source_field} schema")
        task_rows: list[dict[str, float]] = []
        for index, validation in enumerate(validation_rows):
            values = validation[source_field]
            assert isinstance(values, Mapping)
            task_row: dict[str, float] = {}
            for task in _LOSS_TASKS:
                value = values.get(task)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{context}: history row {index} has invalid {source_field}.{task}"
                    )
                task_row[task] = float(value)
            task_rows.append(task_row)
        result[final_field] = task_rows[-1]
        result[tail_field] = {
            task: statistics.fmean(row[task] for row in task_rows[-5:])
            for task in _LOSS_TASKS
        }
    if all(
        isinstance(row.get("losses"), Mapping)
        and isinstance(row.get("weighted_loss_contributions"), Mapping)
        for row in validation_rows
    ):
        for index, validation in enumerate(validation_rows):
            losses = validation["losses"]
            contributions = validation["weighted_loss_contributions"]
            assert isinstance(losses, Mapping) and isinstance(contributions, Mapping)
            weighted_total = validation.get("weighted_multitask_loss")
            target = validation.get("selection_target")
            selection = validation["selection_loss"]
            if (
                isinstance(weighted_total, bool)
                or not isinstance(weighted_total, (int, float))
                or target not in {"weighted", "match"}
                or not math.isclose(
                    float(weighted_total),
                    math.fsum(float(contributions[task]) for task in _LOSS_TASKS),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(f"{context}: history row {index} has inconsistent task totals")
            expected_selection = (
                float(weighted_total) if target == "weighted" else float(losses["match"])
            )
            if not math.isclose(
                float(selection), expected_selection, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"{context}: history row {index} has inconsistent selection target"
                )
    return result


def _contrast_metric(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate: str,
    reference: str,
    path: Sequence[str],
    *,
    lower_is_better: bool,
) -> dict[str, Any] | None:
    candidate_values = _numeric_metric(runs, candidate, path)
    reference_values = _numeric_metric(runs, reference, path)
    if candidate_values is None and reference_values is None:
        return None
    if candidate_values is None or reference_values is None:
        joined = ".".join(path)
        raise ValueError(
            f"contrast {candidate}-{reference} has inconsistent metric schema at {joined}"
        )
    deltas = [
        value - baseline
        for value, baseline in zip(candidate_values, reference_values, strict=True)
    ]
    delta_mean = statistics.fmean(deltas)
    return {
        "seeds": len(deltas),
        "candidate": candidate,
        "reference": reference,
        "delta_mean": delta_mean,
        "population_std": statistics.pstdev(deltas) if len(deltas) > 1 else None,
        "lower_is_better": lower_is_better,
        "improvement_mean": -delta_mean if lower_is_better else delta_mean,
    }


def _aggregate_named_contrasts(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    prediction_paths = {
        "match_log_loss": (("validation_metrics", "match", "log_loss"), True),
        "match_accuracy": (("validation_metrics", "match", "accuracy"), False),
        "live_hit_marginal_log_loss": (
            ("validation_metrics", "live_hit", "log_loss"),
            True,
        ),
        "live_hit_joint_nll": (
            ("validation_metrics", "live_hit", "joint_nll"),
            True,
        ),
        "live_hit_observed_nll": (
            ("validation_metrics", "live_hit", "observed_nll"),
            True,
        ),
        "pa_log_loss": (("validation_metrics", "pa", "log_loss"), True),
        "pa_accuracy": (("validation_metrics", "pa", "accuracy"), False),
    }
    for definition in _NAMED_CONTRASTS:
        name = definition["name"]
        candidate = definition["candidate"]
        reference = definition["reference"]
        selection = _contrast_metric(
            runs,
            candidate,
            reference,
            ("validation_selection_loss",),
            lower_is_better=True,
        )
        assert selection is not None
        checkpoint: dict[str, Any] = {}
        for field in (
            "final_validation_selection_loss",
            "last_five_validation_selection_loss_mean",
        ):
            value = _contrast_metric(
                runs,
                candidate,
                reference,
                (field,),
                lower_is_better=True,
            )
            if value is not None:
                checkpoint[field] = value
        task_losses: dict[str, Any] = {}
        contributions: dict[str, Any] = {}
        for task in _LOSS_TASKS:
            loss = _contrast_metric(
                runs,
                candidate,
                reference,
                ("validation_metrics", "losses", task),
                lower_is_better=True,
            )
            if loss is not None:
                task_losses[task] = loss
            contribution = _contrast_metric(
                runs,
                candidate,
                reference,
                ("validation_metrics", "weighted_loss_contributions", task),
                lower_is_better=True,
            )
            if contribution is not None:
                contributions[task] = contribution
        checkpoint_task_deltas: dict[str, Any] = {}
        for view, loss_field, contribution_field in (
            (
                "final",
                "final_validation_losses",
                "final_validation_weighted_loss_contributions",
            ),
            (
                "last_five",
                "last_five_validation_losses_mean",
                "last_five_validation_weighted_loss_contributions_mean",
            ),
        ):
            view_losses: dict[str, Any] = {}
            view_contributions: dict[str, Any] = {}
            for task in _LOSS_TASKS:
                loss = _contrast_metric(
                    runs,
                    candidate,
                    reference,
                    (loss_field, task),
                    lower_is_better=True,
                )
                if loss is not None:
                    view_losses[task] = loss
                contribution = _contrast_metric(
                    runs,
                    candidate,
                    reference,
                    (contribution_field, task),
                    lower_is_better=True,
                )
                if contribution is not None:
                    view_contributions[task] = contribution
            if view_losses or view_contributions:
                checkpoint_task_deltas[view] = {
                    "validation_loss_deltas": view_losses,
                    "weighted_contribution_deltas": view_contributions,
                }
        prediction_metrics: dict[str, Any] = {}
        for metric, (path, lower_is_better) in prediction_paths.items():
            value = _contrast_metric(
                runs,
                candidate,
                reference,
                path,
                lower_is_better=lower_is_better,
            )
            if value is not None:
                prediction_metrics[metric] = value
        result[name] = {
            **definition,
            "validation_selection_loss": selection,
            "checkpoint_selection": checkpoint,
            "validation_loss_deltas": task_losses,
            "weighted_contribution_deltas": contributions,
            "checkpoint_task_deltas": checkpoint_task_deltas,
            "prediction_metric_deltas": prediction_metrics,
        }
    return result


def _validate_validation_metric_contracts(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, list[str]]:
    selection_targets: set[str] = set()
    warnings: list[str] = []
    zero_count_tasks: set[str] = set()
    for seed, per_seed in runs.items():
        reference_counts: dict[str, int] | None = None
        for variant in MATCHED_GRAPH_VARIANTS:
            child = per_seed[variant]
            metrics = child.get("validation_metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{seed}/{variant}: validation metrics are unavailable")
            target = metrics.get("selection_target")
            if target not in {"weighted", "match"}:
                raise ValueError(f"{seed}/{variant}: unknown validation selection target")
            selection_targets.add(str(target))
            losses = metrics.get("losses")
            contributions = metrics.get("weighted_loss_contributions")
            raw_counts = metrics.get("loss_sample_counts")
            if (
                not isinstance(losses, Mapping)
                or not isinstance(contributions, Mapping)
                or not isinstance(raw_counts, Mapping)
            ):
                raise ValueError(f"{seed}/{variant}: task loss decomposition is incomplete")
            counts: dict[str, int] = {}
            contribution_values: list[float] = []
            for task in _LOSS_TASKS:
                loss = losses.get(task)
                contribution = contributions.get(task)
                count = raw_counts.get(task)
                if (
                    isinstance(loss, bool)
                    or not isinstance(loss, (int, float))
                    or isinstance(contribution, bool)
                    or not isinstance(contribution, (int, float))
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    raise ValueError(f"{seed}/{variant}: invalid {task} loss decomposition")
                counts[task] = count
                contribution_values.append(float(contribution))
                if count == 0:
                    zero_count_tasks.add(task)
            if reference_counts is None:
                reference_counts = counts
            elif counts != reference_counts:
                raise ValueError(
                    f"{seed}/{variant}: validation task sample counts differ from full"
                )
            selection = metrics.get("selection_loss")
            child_selection = child.get("validation_selection_loss")
            if (
                isinstance(selection, bool)
                or not isinstance(selection, (int, float))
                or isinstance(child_selection, bool)
                or not isinstance(child_selection, (int, float))
                or not math.isclose(
                    float(selection), float(child_selection), rel_tol=1e-9, abs_tol=1e-9
                )
            ):
                raise ValueError(f"{seed}/{variant}: saved validation selection loss differs")
            expected = (
                metrics.get("weighted_multitask_loss")
                if target == "weighted"
                else losses.get("match")
            )
            weighted_total = metrics.get("weighted_multitask_loss")
            if (
                isinstance(weighted_total, bool)
                or not isinstance(weighted_total, (int, float))
                or not math.isclose(
                    float(weighted_total),
                    math.fsum(contribution_values),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    f"{seed}/{variant}: weighted task contributions do not sum to total"
                )
            if (
                isinstance(expected, bool)
                or not isinstance(expected, (int, float))
                or not math.isclose(
                    float(selection), float(expected), rel_tol=1e-9, abs_tol=1e-9
                )
            ):
                raise ValueError(f"{seed}/{variant}: selection target does not match its loss")
    if len(selection_targets) != 1:
        raise ValueError("matched variants use different validation selection targets")
    for task in _LOSS_TASKS:
        if task in zero_count_tasks:
            warnings.append(
                f"{task}: zero validation labels; its numeric zero loss is not a measured score"
            )
    return next(iter(selection_targets)), warnings


def analyze_matched_graph_ablations(suite_directory: str | Path) -> dict[str, Any]:
    """Audit and decompose a completed validation-only suite from saved reports."""

    output = Path(suite_directory).expanduser().resolve()
    report_path = output / "matched_retraining_report.json"
    report = _load_json(report_path)
    if report.get("status") != "completed":
        raise ValueError(f"matched suite is not completed: {report.get('status')!r}")
    if report.get("protocol") != "matched_from_scratch_validation_graph_ablation":
        raise ValueError("report is not a matched validation graph-ablation suite")
    protocol_version = report.get("protocol_version")
    if protocol_version not in {1, MATCHED_ABLATION_PROTOCOL_VERSION}:
        raise ValueError("matched suite protocol version is unsupported")
    if report.get("selection_split") != "validation":
        raise ValueError("matched suite did not use validation for selection")
    if report.get("test_used_for_training_selection_or_comparison") is not False:
        raise ValueError("matched suite does not prove that held-out test stayed sealed")
    raw_runs = report.get("runs")
    if not isinstance(raw_runs, Mapping) or not raw_runs:
        raise ValueError("matched suite report has no completed runs")
    runs = _plain(raw_runs)
    warnings: list[str] = []
    if protocol_version == 1:
        warnings.append(
            "legacy protocol v1 has no variant-specific trainable-parameter audit; "
            "metrics are analyzed read-only and are not promoted to the v2 capacity contract"
        )
    for seed, per_seed in runs.items():
        if not isinstance(per_seed, Mapping):
            raise ValueError(f"matched suite seed {seed!r} is not an object")
        if set(per_seed) != set(MATCHED_GRAPH_VARIANTS):
            raise ValueError(f"matched suite seed {seed!r} has incomplete variants")
        for variant in MATCHED_GRAPH_VARIANTS:
            child = per_seed[variant]
            if not isinstance(child, dict):
                raise ValueError(f"matched child {seed}/{variant} is not an object")
            saved_run_directory = Path(child["run_directory"])
            local_run_directory = output / f"seed-{seed}" / variant
            run_directory = (
                local_run_directory
                if (local_run_directory / "training_report.json").is_file()
                else saved_run_directory
            )
            training_path = run_directory / "training_report.json"
            training = _load_json(training_path)
            context = f"{seed}/{variant}"
            expected_policy = _VARIANT_POLICIES[variant]
            training_config = training.get("configuration")
            child_policy = child.get("variant_policy")
            if not isinstance(training_config, Mapping) or not isinstance(
                child_policy, Mapping
            ):
                raise ValueError(f"{context}: variant policy lineage is unavailable")
            for policy_field, child_field, training_field in (
                (
                    "route_message_normalization",
                    "route_message_normalization",
                    "route_message_normalization",
                ),
                ("route_schedule", "route_schedule_preset", "route_schedule"),
            ):
                expected_value = expected_policy[policy_field]
                if (
                    child.get(child_field) != expected_value
                    or training_config.get(training_field) != expected_value
                    or child_policy.get(policy_field) != expected_value
                ):
                    raise ValueError(f"{context}: {policy_field} is mislabeled")
            expected_control = expected_policy["graph_control"]
            child_control = child.get("graph_control")
            training_control = training.get("graph_control")
            if (
                not isinstance(child_control, Mapping)
                or not isinstance(training_control, Mapping)
                or child_control.get("mode") != expected_control
                or training_control.get("mode") != expected_control
                or training_config.get("graph_control") != expected_control
                or child_policy.get("graph_control") != expected_control
            ):
                raise ValueError(f"{context}: graph control is mislabeled")
            summary = _training_history_summary(training, context=context)
            if int(child.get("best_epoch", -1)) != int(training["best_epoch"]):
                raise ValueError(f"{context}: matched and training best epochs differ")
            if (
                "completed_epochs" in child
                and int(child["completed_epochs"]) != int(training["completed_epochs"])
            ):
                raise ValueError(f"{context}: matched and training epoch counts differ")
            for hash_field in ("best_checkpoint_sha256", "last_checkpoint_sha256"):
                if (
                    hash_field in child
                    and hash_field in training
                    and child[hash_field] != training[hash_field]
                ):
                    raise ValueError(f"{context}: {hash_field} lineage differs")
            if child.get("test_used_during_training") is not False or training.get(
                "test_used_during_training"
            ) is not False:
                raise ValueError(f"{context}: training report does not keep test sealed")
            reevaluated = float(child["validation_selection_loss"])
            history_best = float(summary["history_best_validation_selection_loss"])
            if not math.isclose(reevaluated, history_best, rel_tol=1e-6, abs_tol=1e-6):
                warnings.append(
                    f"{context}: best.pt re-evaluation ({reevaluated:.6f}) differs from "
                    f"the training-history minimum ({history_best:.6f})"
                )
            child.update(summary)
            child["final_minus_best_selection_loss"] = (
                float(summary["final_validation_selection_loss"]) - reevaluated
            )
    selection_target, contract_warnings = _validate_validation_metric_contracts(runs)
    warnings.extend(contract_warnings)
    if len(runs) == 1:
        warnings.append(
            "one training seed is present; a population standard deviation of zero is "
            "arithmetic only, not evidence of stability"
        )
    warnings.append(
        "best epochs and variants were compared on the same validation split; compare "
        "best, final, and last-five-epoch contrasts before attributing a route effect"
    )
    return {
        "report_path": str(report_path),
        "protocol": report.get("protocol"),
        "protocol_version": protocol_version,
        "selection_split": report.get("selection_split"),
        "held_out_test_season": report.get("held_out_test_season"),
        "test_used_for_training_selection_or_comparison": report.get(
            "test_used_for_training_selection_or_comparison"
        ),
        "seeds": list(runs),
        "variants": list(MATCHED_GRAPH_VARIANTS),
        "selection_target": selection_target,
        "aggregate": _aggregate_runs(runs),
        "named_contrasts": _aggregate_named_contrasts(runs),
        "warnings": warnings,
    }


def train_matched_graph_ablations(
    dataset_directory: str | Path,
    suite_directory: str | Path,
    *,
    base_config: runner.KBOTrainingConfig,
    seeds: Sequence[int],
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train all graph variants from matched seeds without evaluating the test split."""

    selected_seeds = tuple(seeds)
    if (
        not selected_seeds
        or len(set(selected_seeds)) != len(selected_seeds)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in selected_seeds
        )
    ):
        raise ValueError(
            "matched seeds must be a non-empty sequence of unique non-negative integers"
        )
    if base_config.patience != 0:
        raise ValueError(
            "matched ablations require patience=0 so every variant gets the same epoch budget"
        )

    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(suite_directory).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    split_fingerprint, split_days = _split_day_fingerprint(dataset, base_config)
    variant_protocols = _variant_protocols(base_config)
    runtime_signature = _runtime_signature(base_config)
    manifest = _suite_manifest(
        dataset,
        directory,
        base_config,
        selected_seeds,
        split_fingerprint,
        variant_protocols,
        runtime_signature,
    )
    if output.exists() and any(output.iterdir()) and not (output / "suite_config.json").is_file():
        raise FileExistsError("matched suite directory is non-empty and has no suite_config.json")
    output.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(output / "suite_config.json", manifest)

    initialization = {
        str(seed): _initialization_audit(dataset, base_config, seed) for seed in selected_seeds
    }
    report: dict[str, Any] = {
        "status": "running",
        "protocol": "matched_from_scratch_validation_graph_ablation",
        "protocol_version": MATCHED_ABLATION_PROTOCOL_VERSION,
        "suite_directory": str(output),
        "dataset_directory": str(directory),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "split_day_fingerprint": split_fingerprint,
        "runtime_signature": runtime_signature,
        "split_days": split_days,
        "training_seasons": list(base_config.train_seasons),
        "validation_season": base_config.validation_season,
        "held_out_test_season": base_config.test_season,
        "test_used_for_training_selection_or_comparison": False,
        "selection_split": "validation",
        "seeds": list(selected_seeds),
        "variants": list(MATCHED_GRAPH_VARIANTS),
        "variant_policies": variant_protocols,
        "base_training_config": _plain(asdict(base_config)),
        "initialization_audit": initialization,
        "runs": {},
        "limitations": [
            "node_only retains graph-derived node/role features and removes relational "
            "messages only.",
            "node_only has no relation-attention, route-gate, relational updater, or "
            "relational-normalization parameters; variant-specific parameter counts are reported.",
            "All parameters common to the variants share identical seeded initialization; "
            "data, split, epochs, and optimizer-attempt budgets remain matched.",
            "Validation selects checkpoints and compares variants; the held-out test is "
            "never loaded.",
            "Population standard deviation across the requested training seeds is descriptive.",
        ],
    }
    runner._atomic_json(output / "matched_retraining_report.json", report)

    try:
        for seed in selected_seeds:
            seed_runs: dict[str, Any] = {}
            full_loss: float | None = None
            for variant in MATCHED_GRAPH_VARIANTS:
                config = _variant_config(base_config, variant, seed)
                variant_initialization = initialization[str(seed)]["variants"][variant]
                run_directory = output / f"seed-{seed}" / variant
                prefix = f"[{seed}/{variant}] "

                def child_progress(message: str, *, prefix: str = prefix) -> None:
                    progress(prefix + message)

                training = _train_or_resume_child(
                    dataset,
                    directory,
                    run_directory,
                    config,
                    variant_initialization,
                    child_progress,
                )
                _verify_initialization_lineage(
                    run_directory,
                    training,
                    dataset,
                    config,
                    variant_initialization,
                )
                validation = _reevaluate_best_on_validation(run_directory, directory, config)
                if validation.get("split") != "validation":
                    raise ValueError("matched suite received a non-validation evaluation")
                selection_loss = float(validation["metrics"]["selection_loss"])
                history_summary = _training_history_summary(
                    training, context=f"{seed}/{variant}"
                )
                if full_loss is None:
                    if variant != "full":
                        raise AssertionError("full must be the first matched variant")
                    full_loss = selection_loss
                seed_runs[variant] = {
                    "run_directory": str(run_directory),
                    "best_checkpoint": str(run_directory / "best.pt"),
                    "best_checkpoint_sha256": sha256_file(run_directory / "best.pt"),
                    "best_epoch": int(training["best_epoch"]),
                    "completed_epochs": int(training["completed_epochs"]),
                    "optimizer_steps": int(training["optimizer_steps"]),
                    "skipped_optimizer_steps": int(training["skipped_optimizer_steps"]),
                    "attempted_optimizer_steps": int(training["attempted_optimizer_steps"]),
                    "parameter_count": int(training["parameter_count"]),
                    "trainable_parameter_count": int(
                        training["trainable_parameter_count"]
                    ),
                    "parameter_contract": training["parameter_contract"],
                    "initial_model_state_sha256": str(
                        training["initial_model_state_sha256"]
                    ),
                    "shared_parameter_initialization_sha256": variant_initialization[
                        "shared_parameter_initialization_sha256"
                    ],
                    "architecture": variant_initialization["architecture"],
                    "graph_control": runner._graph_control_report(config),
                    "route_message_normalization": config.route_message_normalization,
                    "route_schedule_preset": config.route_schedule,
                    "resolved_route_schedule": variant_protocols[variant][
                        "resolved_route_schedule"
                    ],
                    "variant_policy": variant_protocols[variant],
                    "validation_selection_loss": selection_loss,
                    "selection_loss_delta_vs_full": selection_loss - full_loss,
                    **history_summary,
                    "final_minus_best_selection_loss": (
                        float(history_summary["final_validation_selection_loss"])
                        - selection_loss
                    ),
                    "validation_metrics": validation["metrics"],
                    "validation_output_directory": validation["output_directory"],
                    "test_used_during_training": training["test_used_during_training"],
                }
                report["runs"][str(seed)] = seed_runs
                runner._atomic_json(output / "matched_retraining_report.json", report)
            attempted_steps = {
                int(run["attempted_optimizer_steps"]) for run in seed_runs.values()
            }
            completed_epochs = {int(run["completed_epochs"]) for run in seed_runs.values()}
            if len(attempted_steps) != 1 or len(completed_epochs) != 1:
                raise ValueError(
                    "matched variants did not receive the same epoch and optimizer-attempt budget"
                )
        report["aggregate"] = _aggregate_runs(report["runs"])
        report["status"] = "completed"
        runner._atomic_json(output / "matched_retraining_report.json", report)
    except Exception:
        report["status"] = "failed"
        runner._atomic_json(output / "matched_retraining_report.json", report)
        raise
    return report


__all__ = [
    "MATCHED_GRAPH_VARIANTS",
    "analyze_matched_graph_ablations",
    "train_matched_graph_ablations",
]
