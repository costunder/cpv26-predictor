"""Matched-from-scratch graph ablations selected and compared on validation only."""

from __future__ import annotations

import hashlib
import json
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
MATCHED_ABLATION_PROTOCOL_VERSION = 1
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
    variants: dict[str, Any] = {}
    reference_hash: str | None = None
    reference_count: int | None = None
    for variant in MATCHED_GRAPH_VARIANTS:
        config = _variant_config(base, variant, seed)
        torch.manual_seed(seed)
        random.seed(seed)
        model = KBORelGNNModel(runner._model_config(dataset, config))
        state_hash = runner._model_state_sha256(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        variants[variant] = {
            "initial_model_state_sha256": state_hash,
            "parameter_count": parameter_count,
        }
        if reference_hash is None:
            reference_hash, reference_count = state_hash, parameter_count
        elif state_hash != reference_hash or parameter_count != reference_count:
            raise ValueError(
                "matched variants do not share identical initialization and parameter count: "
                f"seed={seed}, variant={variant}"
            )
        del model
    assert reference_hash is not None and reference_count is not None
    return {
        "seed": seed,
        "all_variants_equal": True,
        "initial_model_state_sha256": reference_hash,
        "parameter_count": reference_count,
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


def _validate_or_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if not path.exists():
        runner._atomic_json(path, manifest)
        return
    with path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    left, right = dict(saved), _plain(manifest)
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
    actual = dict(report["configuration"])
    expected_values = _plain(asdict(expected))
    actual_epochs = int(actual.pop("epochs"))
    expected_epochs = int(expected_values.pop("epochs"))
    if actual != expected_values:
        raise ValueError("existing matched child run differs from the suite fairness settings")
    if actual_epochs > expected_epochs:
        raise ValueError("existing matched child run exceeds the requested target epochs")
    if report.get("test_used_during_training") is not False:
        raise ValueError("matched child report does not prove that test was held out")


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
        progress=progress,
    )


def _reevaluate_best_on_validation(
    run_directory: Path,
    dataset_directory: Path,
    config: runner.KBOTrainingConfig,
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


def _aggregate_values(values: Sequence[float], deltas: Sequence[float]) -> dict[str, Any]:
    return {
        "seeds": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "paired_delta_vs_full_mean": statistics.fmean(deltas),
        "paired_delta_vs_full_population_std": statistics.pstdev(deltas),
    }


def _aggregate_runs(runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_fields = (
        "log_loss",
        "accuracy",
        "expected_calibration_error",
        "brier_score",
    )
    for variant in MATCHED_GRAPH_VARIANTS:
        losses = [
            float(per_seed[variant]["validation_selection_loss"])
            for per_seed in runs.values()
        ]
        deltas = [
            float(per_seed[variant]["selection_loss_delta_vs_full"])
            for per_seed in runs.values()
        ]
        task_metrics: dict[str, Any] = {}
        for task in ("match", "live_hit", "pa"):
            task_result: dict[str, Any] = {}
            for field in metric_fields:
                values = [
                    float(per_seed[variant]["validation_metrics"][task][field])
                    for per_seed in runs.values()
                ]
                metric_deltas = [
                    float(per_seed[variant]["validation_metrics"][task][field])
                    - float(per_seed["full"]["validation_metrics"][task][field])
                    for per_seed in runs.values()
                ]
                task_result[field] = _aggregate_values(values, metric_deltas)
            task_metrics[task] = task_result
        result[variant] = {
            "validation_selection_loss": _aggregate_values(losses, deltas),
            "validation_metrics": task_metrics,
            "parameter_count": next(iter(runs.values()))[variant]["parameter_count"],
        }
    return result


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
            "node_only keeps the unused route parameters, so equal parameter count is a "
            "budget-control property rather than proof that every parameter receives gradients.",
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
                run_directory = output / f"seed-{seed}" / variant
                prefix = f"[{seed}/{variant}] "

                def child_progress(message: str, *, prefix: str = prefix) -> None:
                    progress(prefix + message)

                training = _train_or_resume_child(
                    dataset,
                    directory,
                    run_directory,
                    config,
                    initialization[str(seed)],
                    child_progress,
                )
                _verify_initialization_lineage(
                    run_directory,
                    training,
                    dataset,
                    config,
                    initialization[str(seed)],
                )
                validation = _reevaluate_best_on_validation(run_directory, directory, config)
                if validation.get("split") != "validation":
                    raise ValueError("matched suite received a non-validation evaluation")
                selection_loss = float(validation["metrics"]["selection_loss"])
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
                    "parameter_count": initialization[str(seed)]["parameter_count"],
                    "initial_model_state_sha256": initialization[str(seed)][
                        "initial_model_state_sha256"
                    ],
                    "graph_control": runner._graph_control_report(config),
                    "route_message_normalization": config.route_message_normalization,
                    "route_schedule_preset": config.route_schedule,
                    "resolved_route_schedule": variant_protocols[variant][
                        "resolved_route_schedule"
                    ],
                    "variant_policy": variant_protocols[variant],
                    "validation_selection_loss": selection_loss,
                    "selection_loss_delta_vs_full": selection_loss - full_loss,
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
    "train_matched_graph_ablations",
]
