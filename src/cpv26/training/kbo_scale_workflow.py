"""One-command, fail-closed 128x3 to 256x3 RelGNN scale workflow.

The workflow deliberately derives its training configuration from a completed
capacity-comparison report.  Users cannot restate the split, seed, optimizer,
sampling, or objective on the command line and accidentally create an
incomparable candidate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from cpv26.data.kbo_graph_dataset import GRAPH_DATASET_VERSION, KBOGraphDataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.models.kbo_relgnn import KBORelGNNModel
from cpv26.training import kbo_capacity_comparison as capacity
from cpv26.training import kbo_matched_ablation as matched
from cpv26.training import kbo_runner as runner
from cpv26.training import kbo_scale_comparison as scale

SCALE_CANDIDATE_CAPACITY = {"hidden_dim": 256, "layers": 3, "heads": 8}
SCALE_TRAINING_WORKFLOW_PROTOCOL = "production_single_seed_scale_workflow"
SCALE_TRAINING_WORKFLOW_PROTOCOL_VERSION = 1
SCALE_TRAINING_WORKFLOW_REPORT = "scale_workflow_report.json"
_LEGACY_EXECUTION_FIELDS = {"activation_checkpointing", "compact_kbo_channels"}


@dataclass(frozen=True)
class KBOScaleTrainingPlan:
    """Validated paths and the only candidate configuration allowed to run."""

    baseline_report: Path
    dataset_directory: Path
    output_directory: Path
    preflight_report: Path
    candidate_report: Path
    scale_report: Path
    config: runner.KBOTrainingConfig
    runtime_signature: dict[str, Any]
    expected_parameter_count: int
    dataset_fingerprint: str
    baseline_report_sha256: str


def _load_hashed_report(
    path: str | Path, *, context: str
) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} report does not exist: {resolved}")
    try:
        payload = resolved.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} report is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} report must be a JSON object")
    return resolved, value, hashlib.sha256(payload).hexdigest()


def _validate_completed_baseline(
    report: Mapping[str, Any],
) -> tuple[runner.KBOTrainingConfig, dict[str, Any], dict[str, Any]]:
    """Validate the completed 128x3 source with the scale report's contracts."""

    context = "baseline capacity"
    scale._require_completed_protocol(
        report,
        context=context,
        protocol=capacity.CAPACITY_COMPARISON_PROTOCOL,
    )
    raw_config = scale._mapping(
        report.get("training_config"), context="baseline training_config"
    )
    known_fields = {field.name for field in fields(runner.KBOTrainingConfig)}
    raw_fields = set(raw_config)
    missing = known_fields.difference(raw_fields).difference(_LEGACY_EXECUTION_FIELDS)
    unexpected = raw_fields.difference(known_fields)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise ValueError(
            "baseline training_config schema differs from the production contract: "
            + "; ".join(details)
        )
    normalized_config = dict(raw_config)
    for field_name in _LEGACY_EXECUTION_FIELDS:
        normalized_config.setdefault(field_name, False)
    base = runner.KBOTrainingConfig.from_dict(normalized_config)
    scale._capacity(
        report,
        raw_config,
        context=context,
        report_field="expanded_capacity",
        expected={"hidden_dim": 128, "layers": 3, "heads": 4},
    )
    capacity._validate_full_node_config(base)
    if not base.compact_kbo_channels:
        raise ValueError(
            "baseline capacity uses the legacy dense-channel architecture with "
            "inactive trainable parameters; rerun the compliant baseline with "
            "compact_kbo_channels=True"
        )
    expected_train_seasons = tuple(range(2001, 2025))
    if (
        base.train_seasons != expected_train_seasons
        or base.validation_season != 2025
        or base.test_season != 2026
        or not base.chronological
    ):
        raise ValueError(
            "production scale workflow requires chronological train=2001..2024, "
            "validation=2025, held-out test=2026"
        )
    if base.max_days_per_split is not None:
        raise ValueError("production scale workflow rejects smoke/subset baselines")
    if base.max_pa_per_day != 0 or base.max_edges_per_route_per_day != 0:
        raise ValueError("production scale workflow requires uncapped PA and route edges")
    if (
        base.seed != 2026
        or base.epochs != 30
        or base.batch_days != 8
        or base.accumulate_steps != 1
    ):
        raise ValueError(
            "production scale workflow requires seed=2026, epochs=30, "
            "batch_days=8, and accumulate_steps=1"
        )

    lineage = scale._lineage(report, raw_config, context=context)
    policies = scale._variant_policies(report, context=context)
    expected_policies = capacity._variant_protocols(base)
    if scale._plain(policies) != scale._plain(expected_policies):
        raise ValueError("baseline capacity variant policies disagree with training_config")
    scale._validate_loader_lineage(report, raw_config, lineage, context=context)

    configured_epochs = scale._integer(
        raw_config.get("epochs"),
        context="baseline training_config.epochs",
        minimum=1,
    )
    runs = scale._runs(
        report,
        context="baseline 128x3",
        run_group="expanded_128x3",
        configured_epochs=configured_epochs,
        variant_policies=policies,
    )
    scale._validate_initialization_audit(
        report,
        runs,
        expected_seed=lineage["seed"],
        context="baseline 128x3",
    )
    scale._validate_source_comparison(
        report,
        runs,
        context="baseline 128x3",
        comparison_group="expanded_128x3",
    )
    scale._validate_parameter_audit(
        report,
        runs,
        context="baseline 128x3",
        audit_field="expanded_128x3",
    )
    scale._validate_budget_audit(
        report,
        runs,
        context="baseline 128x3",
        budget_group="expanded_128x3",
    )
    counts = runs["full"]["validation_loss_sample_counts"]
    if counts != runs["node_only"]["validation_loss_sample_counts"]:
        raise ValueError("baseline full/node_only validation sample counts differ")
    scale._validate_root_sample_audit(report, counts, context="baseline 128x3")
    selection_targets = {
        runs[variant]["validation_selection_target"]
        for variant in capacity.CAPACITY_COMPARISON_VARIANTS
    }
    if len(selection_targets) != 1:
        raise ValueError("baseline full/node_only validation selection targets differ")

    runtime_signature = dict(
        scale._mapping(
            report.get("runtime_signature"), context="baseline runtime_signature"
        )
    )
    if not runtime_signature:
        raise ValueError("baseline runtime_signature must not be empty")
    return base, lineage, runtime_signature


def prepare_kbo_scale_training(
    baseline_report: str | Path,
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    device: str | None = None,
) -> KBOScaleTrainingPlan:
    """Validate all reusable lineage before preflight or training writes anything."""

    baseline_path, report, baseline_sha256 = _load_hashed_report(
        baseline_report, context="baseline capacity"
    )
    base, lineage, baseline_runtime = _validate_completed_baseline(report)
    if device is not None and device != base.device:
        raise ValueError(
            "--device must exactly match baseline training_config.device "
            f"({base.device!r}); runtime drift is not allowed"
        )

    candidate = replace(
        base,
        device=base.device if device is None else device,
        hidden_dim=SCALE_CANDIDATE_CAPACITY["hidden_dim"],
        layers=SCALE_CANDIDATE_CAPACITY["layers"],
        heads=SCALE_CANDIDATE_CAPACITY["heads"],
        activation_checkpointing=True,
        compact_kbo_channels=True,
    )
    # This also rejects a source that did not use the fixed-budget two-condition
    # contract (patience=0, intact graph, full base schedule).
    capacity._validate_full_node_config(candidate)

    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"graph dataset directory does not exist: {directory}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"scale output must be a directory: {output}")
    if output == directory or output in directory.parents or directory in output.parents:
        raise ValueError("scale output and graph dataset directories must not overlap")
    if output in baseline_path.parents:
        raise ValueError("scale output must not contain the baseline source report")

    dataset = KBOGraphDataset(directory)
    if dataset.manifest.get("dataset_version") != GRAPH_DATASET_VERSION:
        raise ValueError(
            f"production scale workflow requires v5 dataset_version={GRAPH_DATASET_VERSION}; "
            "graph-vNext is a separate architecture comparison"
        )
    dataset_fingerprint = dataset.manifest.get("fingerprint")
    if dataset_fingerprint != lineage["dataset_fingerprint"]:
        raise ValueError("selected graph dataset fingerprint differs from baseline capacity")
    split_fingerprint, split_days = matched._split_day_fingerprint(dataset, candidate)
    if split_fingerprint != lineage["split_day_fingerprint"]:
        raise ValueError("selected graph dataset produces a different train/validation split")
    if scale._plain(split_days) != scale._plain(lineage["split_days"]):
        raise ValueError("selected graph dataset split days differ from baseline capacity")

    current_runtime = matched._runtime_signature(candidate)
    if scale._plain(current_runtime) != scale._plain(baseline_runtime):
        raise ValueError(
            "current numerical runtime differs from baseline capacity runtime_signature"
        )

    # If this is a resume, reject a stale or manually configured candidate before
    # spending time on the all-batch CUDA preflight.  A new/empty output remains
    # untouched until preflight passes.
    manifest_path = output / capacity.FULL_NODE_COMPARISON_MANIFEST
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(
                "scale output directory is non-empty and has no full/node manifest"
            )
        expected_manifest = capacity._full_node_manifest(
            dataset_directory=directory,
            dataset_fingerprint=str(dataset_fingerprint),
            split_day_fingerprint=split_fingerprint,
            config=candidate,
            runtime_signature=current_runtime,
            variant_protocols=capacity._variant_protocols(candidate),
        )
        capacity._validate_or_write_manifest(manifest_path, expected_manifest)

    model = KBORelGNNModel(runner._model_config(dataset, candidate))
    expected_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    del model

    raw_preflight = output.parent / f"{output.name}.scale_preflight.json"
    if raw_preflight.is_symlink():
        raise ValueError("scale preflight report path must not be a symbolic link")
    preflight = raw_preflight.resolve()
    candidate_report = output / capacity.FULL_NODE_COMPARISON_REPORT
    scale_report = output / SCALE_TRAINING_WORKFLOW_REPORT
    if candidate_report.is_symlink() or scale_report.is_symlink():
        raise ValueError("scale output reports must not be symbolic links")
    if preflight == baseline_path:
        raise ValueError("derived preflight report would overwrite the baseline report")
    if sha256_file(baseline_path) != baseline_sha256:
        raise ValueError("baseline report changed during scale plan validation")
    return KBOScaleTrainingPlan(
        baseline_report=baseline_path,
        dataset_directory=directory,
        output_directory=output,
        preflight_report=preflight,
        candidate_report=candidate_report,
        scale_report=scale_report,
        config=candidate,
        runtime_signature=dict(current_runtime),
        expected_parameter_count=expected_parameter_count,
        dataset_fingerprint=str(dataset_fingerprint),
        baseline_report_sha256=baseline_sha256,
    )


def _validate_preflight_gate(
    plan: KBOScaleTrainingPlan,
    returned: Mapping[str, Any],
    *,
    max_reserved_fraction: float,
) -> tuple[dict[str, Any], str]:
    """Consume the persisted all-batch CUDA gate before pair training starts."""

    from cpv26.training.kbo_scale_preflight import SCALE_PREFLIGHT_PROTOCOL_VERSION

    saved_path, saved, saved_sha256 = _load_hashed_report(
        plan.preflight_report, context="scale preflight"
    )
    if saved_path != plan.preflight_report:
        raise ValueError("scale preflight report path changed unexpectedly")
    if scale._plain(saved) != scale._plain(returned):
        raise ValueError("persisted scale preflight report differs from returned report")
    if saved.get("status") != "passed":
        raise ValueError("scale preflight report is not passed")
    if saved.get("protocol_version") != SCALE_PREFLIGHT_PROTOCOL_VERSION:
        raise ValueError("scale preflight protocol_version is unsupported")
    expected_config = scale._plain(asdict(plan.config))
    if scale._plain(saved.get("candidate_config")) != expected_config:
        raise ValueError("scale preflight candidate_config differs from derived config")
    policy = scale._mapping(
        saved.get("measurement_policy"), context="scale preflight measurement_policy"
    )
    expected_passes = [
        "first_batch_optimizer_warmup",
        "all_actual_batches_optimizer_state_materialization",
        "all_actual_batches_steady_state_measurement",
    ]
    if (
        policy.get("model_and_optimizer")
        != "one_fresh_instance_persistent_across_all_steps"
        or policy.get("passes") != expected_passes
        or policy.get("allocator_between_batches")
        != "steady_state_cache_retained_after_single_initial_empty_cache"
        or policy.get("warmup_and_materialization_allocator")
        != "isolated_empty_cache_before_and_after_each_batch"
    ):
        raise ValueError("scale preflight measurement policy is not the all-batch protocol")

    workload = scale._mapping(
        saved.get("workload_audit"), context="scale preflight workload_audit"
    )
    _, baseline = scale._load_report(
        plan.baseline_report, context="baseline capacity"
    )
    if workload.get("dataset_fingerprint") != baseline.get("dataset_fingerprint"):
        raise ValueError("scale preflight dataset fingerprint differs from baseline")
    if scale._plain(workload.get("candidate_config")) != expected_config:
        raise ValueError("scale preflight workload config differs from derived config")

    expected_test = {
        "season": plan.config.test_season,
        "graph_days_loaded": False,
        "labels_loaded": False,
        "sealed": True,
    }
    held_out = scale._mapping(
        saved.get("held_out_test"), context="scale preflight held_out_test"
    )
    if scale._plain(held_out) != scale._plain(expected_test):
        raise ValueError("scale preflight does not prove that held-out test stayed sealed")
    splits = scale._mapping(
        workload.get("splits"), context="scale preflight workload splits"
    )
    test_split = scale._mapping(
        splits.get("test"), context="scale preflight workload test split"
    )
    for field, expected in expected_test.items():
        if test_split.get(field) != expected:
            raise ValueError("scale preflight workload audit opened held-out test data")

    runtime = scale._mapping(saved.get("runtime"), context="scale preflight runtime")
    for field, expected in plan.runtime_signature.items():
        if scale._plain(runtime.get(field)) != scale._plain(expected):
            raise ValueError(f"scale preflight runtime.{field} differs from baseline")

    count_fields = (
        "planned_actual_batch_count",
        "completed_actual_batch_count",
        "completed_materialization_batch_count",
    )
    counts = [
        scale._integer(saved.get(field), context=f"scale preflight {field}", minimum=1)
        for field in count_fields
    ]
    evaluated = saved.get("evaluated_batches")
    materialized = saved.get("materialization_batches")
    if not isinstance(evaluated, list) or not isinstance(materialized, list):
        raise ValueError("scale preflight batch measurements are malformed")
    if len(set((*counts, len(evaluated), len(materialized)))) != 1:
        raise ValueError("scale preflight did not measure every actual batch in both passes")
    planned = counts[0]

    execution = scale._mapping(
        saved.get("execution"), context="scale preflight execution"
    )
    if execution.get("all_actual_batches_evaluated") is not True:
        raise ValueError("scale preflight did not attest all actual batches")
    if execution.get("optimizer_state_locked_before_steady_state") is not True:
        raise ValueError("scale preflight did not lock AdamW state before measurement")
    if (
        execution.get("steady_state_allocator_cache_cleared_once_before_pass")
        is not True
        or execution.get("steady_state_allocator_cache_retained_between_batches")
        is not True
    ):
        raise ValueError("scale preflight did not retain the steady-state CUDA allocator")
    if execution.get("overall_peak_includes_warmup") is not True or execution.get(
        "overall_peak_includes_materialization_pass"
    ) is not True:
        raise ValueError("scale preflight peak does not cover all measurement phases")
    expected_steps = {
        "warmup_steps": 1,
        "materialization_steps": planned,
        "steady_state_steps": planned,
    }
    for field, expected in expected_steps.items():
        if scale._integer(
            execution.get(field), context=f"scale preflight execution.{field}", minimum=1
        ) != expected:
            raise ValueError("scale preflight phase step counts are inconsistent")
    if scale._integer(
        execution.get("evaluated_batch_count"),
        context="scale preflight execution.evaluated_batch_count",
        minimum=1,
    ) != planned:
        raise ValueError("scale preflight execution batch count is inconsistent")
    if scale._integer(
        execution.get("parameter_count"),
        context="scale preflight execution.parameter_count",
        minimum=1,
    ) != plan.expected_parameter_count:
        raise ValueError("scale preflight measured a different model parameter count")

    memory = scale._mapping(
        saved.get("memory_safety"), context="scale preflight memory_safety"
    )
    limit = scale._number(
        memory.get("max_reserved_fraction"),
        context="scale preflight max_reserved_fraction",
    )
    observed = scale._number(
        memory.get("peak_reserved_fraction"),
        context="scale preflight peak_reserved_fraction",
    )
    if memory.get("passed") is not True or observed > limit:
        raise ValueError("scale preflight memory threshold did not pass")
    if limit != float(max_reserved_fraction):
        raise ValueError("scale preflight memory threshold differs from requested limit")
    peak = scale._integer(
        execution.get("peak_reserved_bytes"),
        context="scale preflight execution.peak_reserved_bytes",
        minimum=0,
    )
    total = scale._integer(
        execution.get("total_memory_bytes"),
        context="scale preflight execution.total_memory_bytes",
        minimum=1,
    )
    execution_fraction = scale._number(
        execution.get("peak_reserved_fraction"),
        context="scale preflight execution.peak_reserved_fraction",
    )
    threshold = scale._integer(
        memory.get("threshold_reserved_bytes"),
        context="scale preflight memory_safety.threshold_reserved_bytes",
        minimum=0,
    )
    if not (
        peak <= total
        and execution_fraction == peak / total
        and observed == execution_fraction
        and threshold == int(total * limit)
        and memory.get("headroom_to_threshold_bytes") == threshold - peak
        and execution.get("headroom_bytes") == total - peak
    ):
        raise ValueError("scale preflight memory measurements are internally inconsistent")
    return dict(saved), saved_sha256


def train_kbo_scale_workflow(
    baseline_report: str | Path,
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    device: str | None = None,
    max_reserved_fraction: float = 0.85,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Preflight every real batch, train one pair, then compare validation."""

    plan = prepare_kbo_scale_training(
        baseline_report,
        dataset_directory,
        output_directory,
        device=device,
    )
    # Imported only after fail-fast lineage validation, before candidate writes.
    from cpv26.training.kbo_scale_preflight import run_kbo_scale_preflight

    progress(f"Scale lineage validated from {plan.baseline_report}")
    returned_preflight = run_kbo_scale_preflight(
        plan.dataset_directory,
        plan.config,
        output=plan.preflight_report,
        max_reserved_fraction=max_reserved_fraction,
        progress=progress,
    )
    preflight_report, preflight_sha256 = _validate_preflight_gate(
        plan,
        returned_preflight,
        max_reserved_fraction=max_reserved_fraction,
    )
    if sha256_file(plan.baseline_report) != plan.baseline_report_sha256:
        raise ValueError("baseline capacity report changed during scale preflight")
    progress(f"All-batch CUDA preflight passed: {plan.preflight_report}")
    pair_report = capacity.train_kbo_full_node_comparison(
        plan.dataset_directory,
        plan.output_directory,
        config=plan.config,
        progress=progress,
    )
    if sha256_file(plan.preflight_report) != preflight_sha256:
        raise ValueError("scale preflight report changed while pair training ran")
    if sha256_file(plan.baseline_report) != plan.baseline_report_sha256:
        raise ValueError("baseline capacity report changed while pair training ran")
    candidate_path, persisted_pair, candidate_sha256 = _load_hashed_report(
        plan.candidate_report, context="candidate scale"
    )
    if candidate_path != plan.candidate_report:
        raise ValueError("candidate scale report path changed unexpectedly")
    if scale._plain(persisted_pair) != scale._plain(pair_report):
        raise ValueError("persisted candidate report differs from returned pair report")
    comparison_report = scale.compare_kbo_scale_reports(
        plan.baseline_report,
        plan.candidate_report,
        output_path=None,
    )
    if sha256_file(plan.preflight_report) != preflight_sha256:
        raise ValueError("scale preflight report changed during final comparison")
    if sha256_file(plan.baseline_report) != plan.baseline_report_sha256:
        raise ValueError("baseline capacity report changed during final comparison")
    if sha256_file(plan.candidate_report) != candidate_sha256:
        raise ValueError("candidate scale report changed during final comparison")
    config_fingerprint = hashlib.sha256(
        json.dumps(
            scale._plain(asdict(plan.config)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    comparison_protocol = {
        "protocol": comparison_report["protocol"],
        "protocol_version": comparison_report["protocol_version"],
    }
    comparison_report = {
        **comparison_report,
        "protocol": SCALE_TRAINING_WORKFLOW_PROTOCOL,
        "protocol_version": SCALE_TRAINING_WORKFLOW_PROTOCOL_VERSION,
        "comparison_protocol": comparison_protocol,
        "preflight_gate": {
            "status": "passed",
            "report": str(plan.preflight_report),
            "report_sha256": preflight_sha256,
            "protocol_version": preflight_report["protocol_version"],
            "candidate_config_sha256": config_fingerprint,
            "parameter_count": plan.expected_parameter_count,
            "max_reserved_fraction": float(max_reserved_fraction),
            "peak_reserved_bytes": preflight_report["execution"][
                "peak_reserved_bytes"
            ],
            "total_memory_bytes": preflight_report["execution"][
                "total_memory_bytes"
            ],
            "peak_reserved_fraction": preflight_report["execution"][
                "peak_reserved_fraction"
            ],
            "all_actual_batches_evaluated": True,
            "held_out_test_sealed": True,
            "source_report_sha256": {
                "baseline_128x3": plan.baseline_report_sha256,
                "candidate_256x3": candidate_sha256,
            },
        },
    }
    runner._atomic_json(plan.scale_report, comparison_report)
    return {
        "status": "completed",
        "baseline_report": str(plan.baseline_report),
        "dataset_directory": str(plan.dataset_directory),
        "output_directory": str(plan.output_directory),
        "preflight_report": str(plan.preflight_report),
        "candidate_report": str(plan.candidate_report),
        "scale_report": str(plan.scale_report),
        "candidate_config": asdict(plan.config),
        "preflight": preflight_report,
        "pair": pair_report,
        "comparison": comparison_report,
    }


__all__ = [
    "KBOScaleTrainingPlan",
    "SCALE_CANDIDATE_CAPACITY",
    "SCALE_TRAINING_WORKFLOW_PROTOCOL",
    "SCALE_TRAINING_WORKFLOW_PROTOCOL_VERSION",
    "SCALE_TRAINING_WORKFLOW_REPORT",
    "prepare_kbo_scale_training",
    "train_kbo_scale_workflow",
]
