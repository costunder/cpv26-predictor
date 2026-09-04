"""Fail-closed validation comparison for 128x3 and 256x3 RelGNN runs.

This module does not train or evaluate a model.  It consumes the completed
validation-only reports written by :mod:`kbo_capacity_comparison` and verifies
that their experimental lineage is comparable before calculating scale deltas.
The held-out test split must remain sealed in both source reports.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from cpv26.training.kbo_capacity_comparison import (
    CAPACITY_COMPARISON_PROTOCOL,
    FULL_NODE_COMPARISON_PROTOCOL,
)

SCALE_COMPARISON_PROTOCOL = "single_seed_validation_scale_comparison"
SCALE_COMPARISON_PROTOCOL_VERSION = 2
SCALE_COMPARISON_REPORT = "scale_comparison_report.json"

_SUPPORTED_SOURCE_PROTOCOL_VERSION = 2
_VARIANTS = ("full", "node_only")
_BASELINE_CAPACITY = {"hidden_dim": 128, "layers": 3, "heads": 4}
_CANDIDATE_CAPACITY = {"hidden_dim": 256, "layers": 3, "heads": 8}
_LOSS_WEIGHT_FIELDS = (
    "match_weight",
    "live_hit_weight",
    "pa_weight",
    "run_weight",
    "box_pa_weight",
    "box_pitch_weight",
)
_OBJECTIVE_FIELDS = (
    *_LOSS_WEIGHT_FIELDS,
    "selection_target",
    "box_gradient_mode",
)
_GRAPH_CONFIG_FIELDS = (
    "route_message_normalization",
    "route_schedule",
    "route_edge_chunk_size",
    "graph_control",
    "graph_control_seed",
)
_SAMPLING_FIELDS = (
    "max_days_per_split",
    "max_pa_per_day",
    "max_edges_per_route_per_day",
)
_CAPACITY_FIELDS = frozenset({"hidden_dim", "layers", "heads"})
_PERMITTED_EXECUTION_DIFFERENCES = frozenset({"activation_checkpointing"})


def _plain(value: Any) -> Any:
    """Return a detached, strictly JSON-serializable representation."""

    return json.loads(
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    )


def _load_report(path: str | Path, *, context: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} report does not exist: {resolved}")
    try:
        with resolved.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} report is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} report must be a JSON object")
    return resolved, value


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, *, context: str) -> str:
    encoded = _nonempty_string(value, context=context)
    if len(encoded) != 64 or any(
        character not in "0123456789abcdef" for character in encoded
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return encoded


def _integer(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return int(value)


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _seasons(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    seasons = [_integer(item, context=f"{context} item", minimum=1) for item in value]
    if len(set(seasons)) != len(seasons):
        raise ValueError(f"{context} contains duplicate seasons")
    return seasons


def _require_completed_protocol(
    report: Mapping[str, Any],
    *,
    context: str,
    protocol: str,
) -> None:
    if report.get("status") != "completed":
        raise ValueError(f"{context} report is not completed")
    if report.get("protocol") != protocol:
        raise ValueError(f"{context} report has an unsupported protocol")
    if report.get("protocol_version") != _SUPPORTED_SOURCE_PROTOCOL_VERSION:
        raise ValueError(f"{context} report has an unsupported protocol version")
    if report.get("variants") != list(_VARIANTS):
        raise ValueError(f"{context} report must contain exactly full and node_only")


def _require_equal(
    baseline: Any,
    candidate: Any,
    *,
    context: str,
) -> Any:
    if _plain(baseline) != _plain(candidate):
        raise ValueError(f"baseline and candidate {context} differ")
    return _plain(baseline)


def _capacity(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    context: str,
    report_field: str,
    expected: Mapping[str, int],
) -> dict[str, int]:
    declared = _mapping(report.get(report_field), context=f"{context} {report_field}")
    normalized: dict[str, int] = {}
    for field, expected_value in expected.items():
        config_value = _integer(
            config.get(field), context=f"{context} training_config.{field}", minimum=1
        )
        if config_value != expected_value:
            raise ValueError(
                f"{context} must use {field}={expected_value}, got {config_value}"
            )
        if field in declared:
            declared_value = _integer(
                declared[field], context=f"{context} {report_field}.{field}", minimum=1
            )
            if declared_value != config_value:
                raise ValueError(
                    f"{context} {report_field}.{field} disagrees with training_config"
                )
        elif field in {"hidden_dim", "layers"}:
            raise ValueError(f"{context} {report_field}.{field} is missing")
        normalized[field] = config_value
    return normalized


def _activation_checkpointing(
    config: Mapping[str, Any], *, context: str, legacy_default: bool
) -> bool:
    if "activation_checkpointing" not in config:
        if legacy_default:
            return False
        raise ValueError(f"{context} training_config.activation_checkpointing is missing")
    return _boolean(
        config["activation_checkpointing"],
        context=f"{context} training_config.activation_checkpointing",
    )


def _validate_config_keys(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Reject unaccounted configuration drift, including future unknown fields."""

    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    allowed_one_sided = {"activation_checkpointing", "compact_kbo_channels"}
    one_sided = baseline_keys ^ candidate_keys
    if one_sided - allowed_one_sided:
        names = ", ".join(sorted(one_sided - allowed_one_sided))
        raise ValueError(f"training_config schema differs at unaccounted fields: {names}")

    ignored = _CAPACITY_FIELDS | _PERMITTED_EXECUTION_DIFFERENCES
    for field in sorted((baseline_keys & candidate_keys) - ignored):
        if _plain(baseline[field]) != _plain(candidate[field]):
            raise ValueError(f"baseline and candidate training_config.{field} differ")


def _lineage(
    report: Mapping[str, Any], config: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    dataset_fingerprint = _nonempty_string(
        report.get("dataset_fingerprint"), context=f"{context} dataset_fingerprint"
    )
    split_fingerprint = _nonempty_string(
        report.get("split_day_fingerprint"),
        context=f"{context} split_day_fingerprint",
    )
    seed = _integer(report.get("seed"), context=f"{context} seed", minimum=0)
    training_seasons = _seasons(
        report.get("training_seasons"), context=f"{context} training_seasons"
    )
    validation_season = _integer(
        report.get("validation_season"), context=f"{context} validation_season", minimum=1
    )
    test_season = _integer(
        report.get("held_out_test_season"),
        context=f"{context} held_out_test_season",
        minimum=1,
    )
    if report.get("selection_split") != "validation":
        raise ValueError(f"{context} comparison is not validation-only")
    if report.get("test_used_for_training_selection_or_comparison") is not False:
        raise ValueError(f"{context} does not prove that the held-out test stayed sealed")
    if report.get("smoke_test_only") is not (config.get("max_days_per_split") is not None):
        raise ValueError(f"{context} smoke-test declaration is inconsistent")

    config_seed = _integer(config.get("seed"), context=f"{context} training_config.seed", minimum=0)
    config_training_seasons = _seasons(
        config.get("train_seasons"), context=f"{context} training_config.train_seasons"
    )
    config_validation = _integer(
        config.get("validation_season"),
        context=f"{context} training_config.validation_season",
        minimum=1,
    )
    config_test = _integer(
        config.get("test_season"), context=f"{context} training_config.test_season", minimum=1
    )
    if (
        seed != config_seed
        or training_seasons != config_training_seasons
        or validation_season != config_validation
        or test_season != config_test
    ):
        raise ValueError(f"{context} split/seed lineage disagrees with training_config")

    split_days = _mapping(report.get("split_days"), context=f"{context} split_days")
    if set(split_days) != {"train", "validation"}:
        raise ValueError(f"{context} split_days must contain exactly train and validation")
    for split in ("train", "validation"):
        raw_days = split_days[split]
        if not isinstance(raw_days, list) or not all(
            isinstance(day, str) and day for day in raw_days
        ):
            raise ValueError(f"{context} split_days.{split} is malformed")

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "split_day_fingerprint": split_fingerprint,
        "split_days": _plain(split_days),
        "seed": seed,
        "training_seasons": training_seasons,
        "validation_season": validation_season,
        "held_out_test_season": test_season,
    }


def _variant_policies(report: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    policies = _mapping(report.get("variant_policies"), context=f"{context} variant_policies")
    if set(policies) != set(_VARIANTS):
        raise ValueError(f"{context} variant_policies must contain exactly full and node_only")
    normalized: dict[str, Any] = {}
    for variant in _VARIANTS:
        normalized[variant] = _plain(
            _mapping(policies[variant], context=f"{context} {variant} policy")
        )
    return normalized


def _loss_sample_counts(run: Mapping[str, Any], *, context: str) -> dict[str, int]:
    metrics = _mapping(run.get("validation_metrics"), context=f"{context} validation_metrics")
    if metrics.get("selection_target") is None:
        raise ValueError(f"{context} validation selection_target is missing")
    raw = _mapping(
        metrics.get("loss_sample_counts"),
        context=f"{context} validation loss_sample_counts",
    )
    if not raw:
        raise ValueError(f"{context} validation loss_sample_counts is empty")
    counts: dict[str, int] = {}
    for task, value in raw.items():
        if not isinstance(task, str) or not task:
            raise ValueError(f"{context} validation loss_sample_counts has an invalid task")
        counts[task] = _integer(
            value,
            context=f"{context} validation loss_sample_counts.{task}",
            minimum=0,
        )
    return counts


def _runs(
    report: Mapping[str, Any],
    *,
    context: str,
    run_group: str | None,
    configured_epochs: int,
    variant_policies: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    outer = _mapping(report.get("runs"), context=f"{context} runs")
    raw_runs = (
        _mapping(outer.get(run_group), context=f"{context} runs.{run_group}")
        if run_group is not None
        else outer
    )
    if set(raw_runs) != set(_VARIANTS):
        raise ValueError(f"{context} runs must contain exactly full and node_only")

    normalized: dict[str, dict[str, Any]] = {}
    for variant in _VARIANTS:
        run = _mapping(raw_runs[variant], context=f"{context} {variant} run")
        if run.get("test_used_during_training") is not False:
            raise ValueError(f"{context}/{variant} does not prove that test stayed sealed")
        if "smoke_test_only" not in run or run.get("smoke_test_only") is not report.get(
            "smoke_test_only"
        ):
            raise ValueError(f"{context}/{variant} smoke-test lineage differs")
        run_policy = _mapping(
            run.get("variant_policy"), context=f"{context}/{variant} variant_policy"
        )
        if _plain(run_policy) != _plain(variant_policies[variant]):
            raise ValueError(f"{context}/{variant} graph policy lineage differs")

        completed_epochs = _integer(
            run.get("completed_epochs"),
            context=f"{context}/{variant} completed_epochs",
            minimum=1,
        )
        if completed_epochs != configured_epochs:
            raise ValueError(f"{context}/{variant} did not complete the configured epochs")
        attempted_steps = _integer(
            run.get("attempted_optimizer_steps"),
            context=f"{context}/{variant} attempted_optimizer_steps",
            minimum=1,
        )
        optimizer_steps = _integer(
            run.get("optimizer_steps"),
            context=f"{context}/{variant} optimizer_steps",
            minimum=0,
        )
        skipped_optimizer_steps = _integer(
            run.get("skipped_optimizer_steps"),
            context=f"{context}/{variant} skipped_optimizer_steps",
            minimum=0,
        )
        if optimizer_steps + skipped_optimizer_steps != attempted_steps:
            raise ValueError(
                f"{context}/{variant} optimizer-step accounting is inconsistent"
            )
        parameter_count = _integer(
            run.get("parameter_count"),
            context=f"{context}/{variant} parameter_count",
            minimum=1,
        )
        trainable_parameter_count = _integer(
            run.get("trainable_parameter_count"),
            context=f"{context}/{variant} trainable_parameter_count",
            minimum=1,
        )
        parameter_contract = _mapping(
            run.get("parameter_contract"),
            context=f"{context}/{variant} parameter_contract",
        )
        if (
            parameter_contract.get("optimizer_covers_all_trainable") is not True
            or parameter_contract.get("parameter_count") != parameter_count
            or parameter_contract.get("trainable_parameter_count")
            != trainable_parameter_count
        ):
            raise ValueError(
                f"{context}/{variant} does not prove exact optimizer parameter coverage"
            )
        loss = _number(
            run.get("validation_selection_loss"),
            context=f"{context}/{variant} validation_selection_loss",
        )
        counts = _loss_sample_counts(run, context=f"{context}/{variant}")
        selection_target = _nonempty_string(
            _mapping(
                run.get("validation_metrics"),
                context=f"{context}/{variant} validation_metrics",
            ).get("selection_target"),
            context=f"{context}/{variant} validation selection_target",
        )
        normalized[variant] = {
            "validation_selection_loss": loss,
            "validation_loss_sample_counts": counts,
            "validation_selection_target": selection_target,
            "completed_epochs": completed_epochs,
            "attempted_optimizer_steps": attempted_steps,
            "optimizer_steps": optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_contract": _plain(parameter_contract),
            "shared_parameter_initialization_sha256": _sha256(
                run.get("shared_parameter_initialization_sha256"),
                context=f"{context}/{variant} shared initialization hash",
            ),
            "architecture": _plain(
                _mapping(
                    run.get("architecture"),
                    context=f"{context}/{variant} architecture",
                )
            ),
            "initial_model_state_sha256": _sha256(
                run.get("initial_model_state_sha256"),
                context=f"{context}/{variant} initial_model_state_sha256",
            ),
        }
    return normalized


def _validate_initialization_audit(
    report: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    *,
    expected_seed: int,
    context: str,
) -> dict[str, Any]:
    audit = _mapping(
        report.get("initialization_audit"),
        context=f"{context} initialization_audit",
    )
    audit_seed = _integer(
        audit.get("seed"), context=f"{context} initialization seed", minimum=0
    )
    if audit_seed != expected_seed:
        raise ValueError(f"{context} initialization seed differs from report seed")
    if audit.get("all_shared_parameters_equal") is not True:
        raise ValueError(
            f"{context} initialization audit does not prove shared-parameter equality"
        )
    legacy_all_variants_equal = audit.get("all_variants_equal")
    if (
        legacy_all_variants_equal is not None
        and legacy_all_variants_equal is not True
    ):
        raise ValueError(
            f"{context} initialization audit contains a failed legacy equality claim"
        )
    shared_hash = _sha256(
        audit.get("shared_parameter_initialization_sha256"),
        context=f"{context} shared initialization hash",
    )
    variants = _mapping(
        audit.get("variants"), context=f"{context} initialization variants"
    )
    if set(variants) != set(_VARIANTS):
        raise ValueError(
            f"{context} initialization audit must contain exactly full and node_only"
        )
    for variant in _VARIANTS:
        variant_audit = _mapping(
            variants[variant], context=f"{context} {variant} initialization"
        )
        variant_hash = _sha256(
            variant_audit.get("initial_model_state_sha256"),
            context=f"{context} {variant} initialization hash",
        )
        variant_count = _integer(
            variant_audit.get("parameter_count"),
            context=f"{context} {variant} initialization parameter_count",
            minimum=1,
        )
        trainable_count = _integer(
            variant_audit.get("trainable_parameter_count"),
            context=f"{context} {variant} trainable parameter count",
            minimum=1,
        )
        variant_shared_hash = _sha256(
            variant_audit.get("shared_parameter_initialization_sha256"),
            context=f"{context} {variant} shared initialization hash",
        )
        if variant_shared_hash != shared_hash:
            raise ValueError(f"{context} shared parameter initialization differs")
        if runs[variant]["initial_model_state_sha256"] != variant_hash:
            raise ValueError(
                f"{context}/{variant} saved run initialization disagrees with audit"
            )
        if (
            runs[variant]["parameter_count"] != variant_count
            or runs[variant]["trainable_parameter_count"] != trainable_count
            or runs[variant]["shared_parameter_initialization_sha256"] != shared_hash
        ):
            raise ValueError(
                f"{context}/{variant} saved run parameter contract disagrees with "
                "initialization audit"
            )
    return {
        "seed": expected_seed,
        "all_shared_parameters_equal": True,
        "shared_parameter_initialization_sha256": shared_hash,
        "variant_parameter_counts": {
            variant: int(runs[variant]["parameter_count"]) for variant in _VARIANTS
        },
    }


def _validate_source_comparison(
    report: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
    comparison_group: str | None,
) -> None:
    comparison = _mapping(
        report.get("validation_selection_comparison"),
        context=f"{context} validation_selection_comparison",
    )
    if comparison.get("lower_is_better") is not True:
        raise ValueError(f"{context} does not declare lower validation loss as better")
    source = (
        _mapping(
            comparison.get(comparison_group),
            context=f"{context} comparison.{comparison_group}",
        )
        if comparison_group is not None
        else comparison
    )
    expected_full = float(runs["full"]["validation_selection_loss"])
    expected_node = float(runs["node_only"]["validation_selection_loss"])
    declared_full = _number(source.get("full"), context=f"{context} comparison full")
    declared_node = _number(
        source.get("node_only"), context=f"{context} comparison node_only"
    )
    declared_gap = _number(
        source.get("node_only_minus_full"), context=f"{context} comparison gap"
    )
    if not (
        math.isclose(declared_full, expected_full, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(declared_node, expected_node, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            declared_gap, expected_node - expected_full, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(f"{context} validation comparison disagrees with its runs")


def _validate_parameter_audit(
    report: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
    audit_field: str,
) -> dict[str, int]:
    counts = {variant: int(runs[variant]["parameter_count"]) for variant in _VARIANTS}
    if counts["full"] <= counts["node_only"]:
        raise ValueError(f"{context} node_only did not remove relational capacity")
    audit = _mapping(
        report.get("parameter_count_audit"), context=f"{context} parameter_count_audit"
    )
    if audit_field == "expanded_128x3":
        declared_raw = audit.get("expanded_128x3")
    else:
        declared_raw = audit.get("variant_parameter_counts")
    declared = _mapping(
        declared_raw,
        context=f"{context} parameter_count_audit.{audit_field}",
    )
    declared_counts = {
        variant: _integer(
            declared.get(variant),
            context=f"{context} parameter_count_audit.{audit_field}.{variant}",
            minimum=1,
        )
        for variant in _VARIANTS
    }
    if declared_counts != counts:
        raise ValueError(f"{context} parameter audit disagrees with its runs")
    return counts


def _validate_budget_audit(
    report: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
    budget_group: str | None,
) -> dict[str, dict[str, int]]:
    completed = {variant: int(runs[variant]["completed_epochs"]) for variant in _VARIANTS}
    attempted = {
        variant: int(runs[variant]["attempted_optimizer_steps"])
        for variant in _VARIANTS
    }
    successful = {
        variant: int(runs[variant]["optimizer_steps"]) for variant in _VARIANTS
    }
    skipped = {
        variant: int(runs[variant]["skipped_optimizer_steps"])
        for variant in _VARIANTS
    }
    if any(
        len(set(values.values())) != 1
        for values in (completed, attempted, successful, skipped)
    ):
        raise ValueError(f"{context} full and node_only training budgets differ")

    audit = _mapping(report.get("budget_audit"), context=f"{context} budget_audit")
    equality_field = "all_runs_equal" if budget_group is not None else "all_variants_equal"
    if audit.get(equality_field) is not True:
        raise ValueError(f"{context} budget audit does not prove equality")
    declared_completed = _mapping(
        audit.get("completed_epochs"), context=f"{context} completed epoch audit"
    )
    declared_attempted = _mapping(
        audit.get("attempted_optimizer_steps"),
        context=f"{context} optimizer-step audit",
    )
    if budget_group is not None:
        declared_completed = _mapping(
            declared_completed.get(budget_group),
            context=f"{context} completed epoch audit.{budget_group}",
        )
        declared_attempted = _mapping(
            declared_attempted.get(budget_group),
            context=f"{context} optimizer-step audit.{budget_group}",
        )
    if _plain(declared_completed) != completed or _plain(declared_attempted) != attempted:
        raise ValueError(f"{context} budget audit disagrees with its runs")
    return {
        "completed_epochs": completed,
        "attempted_optimizer_steps": attempted,
        "optimizer_steps": successful,
        "skipped_optimizer_steps": skipped,
    }


def _validate_root_sample_audit(
    report: Mapping[str, Any], counts: Mapping[str, int], *, context: str
) -> None:
    """Validate a root audit when present; run metrics remain mandatory either way."""

    if "validation_sample_count_audit" not in report:
        return
    audit = _mapping(
        report["validation_sample_count_audit"],
        context=f"{context} validation_sample_count_audit",
    )
    if audit.get("available") is not True or audit.get("all_variants_equal") is not True:
        raise ValueError(f"{context} validation sample audit does not prove equality")
    declared = _mapping(
        audit.get("loss_sample_counts"),
        context=f"{context} validation sample audit counts",
    )
    if _plain(declared) != _plain(counts):
        raise ValueError(f"{context} validation sample audit disagrees with its runs")


def _loader_fingerprint(lineage: Mapping[str, Any]) -> str:
    fields = (
        "dataset_fingerprint",
        "split_day_fingerprint",
        "split_days",
        "seed",
        "chronological",
        "batch_days",
        "workers",
        "accumulate_steps",
        "max_days_per_split",
        "max_pa_per_day",
        "max_edges_per_route_per_day",
        "graph_control",
        "graph_control_seed",
    )
    protocol = {field: lineage[field] for field in fields}
    encoded = json.dumps(
        _plain(protocol), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_loader_lineage(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    context: str,
) -> None:
    loader = _mapping(report.get("loader_lineage"), context=f"{context} loader_lineage")
    expected = {
        "dataset_fingerprint": lineage["dataset_fingerprint"],
        "split_day_fingerprint": lineage["split_day_fingerprint"],
        "split_days": lineage["split_days"],
        "seed": lineage["seed"],
        "chronological": config.get("chronological"),
        "batch_days": config.get("batch_days"),
        "workers": config.get("workers"),
        "accumulate_steps": config.get("accumulate_steps"),
        "max_days_per_split": config.get("max_days_per_split"),
        "max_pa_per_day": config.get("max_pa_per_day"),
        "max_edges_per_route_per_day": config.get("max_edges_per_route_per_day"),
        "graph_control": config.get("graph_control"),
        "graph_control_seed": config.get("graph_control_seed"),
    }
    for field, value in expected.items():
        if field not in loader or _plain(loader[field]) != _plain(value):
            raise ValueError(f"{context} loader_lineage.{field} is inconsistent")
    fingerprint = _nonempty_string(
        loader.get("fingerprint"), context=f"{context} loader fingerprint"
    )
    if fingerprint != _loader_fingerprint(loader):
        raise ValueError(f"{context} loader fingerprint is invalid")
    variant_fingerprints = _mapping(
        loader.get("variant_fingerprints"),
        context=f"{context} loader variant_fingerprints",
    )
    if variant_fingerprints != {variant: fingerprint for variant in _VARIANTS}:
        raise ValueError(f"{context} loader variant fingerprints differ")
    if loader.get("all_non_route_settings_equal") is not True:
        raise ValueError(f"{context} loader lineage does not prove variant equality")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        partial.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def compare_kbo_scale_reports(
    baseline_report_path: str | Path,
    candidate_report_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare completed 128x3 and 256x3 pairs without opening the test split.

    ``baseline_report_path`` must be a completed capacity-comparison report; its
    ``expanded_128x3`` children are used.  ``candidate_report_path`` must be a
    completed full/node report for a 256-hidden, three-layer, eight-head model.
    Any unapproved lineage drift raises :class:`ValueError` before output is
    written.
    """

    baseline_path, baseline = _load_report(
        baseline_report_path, context="baseline capacity"
    )
    candidate_path, candidate = _load_report(
        candidate_report_path, context="candidate scale"
    )
    if baseline_path == candidate_path:
        raise ValueError("baseline and candidate report paths must differ")

    _require_completed_protocol(
        baseline,
        context="baseline capacity",
        protocol=CAPACITY_COMPARISON_PROTOCOL,
    )
    _require_completed_protocol(
        candidate,
        context="candidate scale",
        protocol=FULL_NODE_COMPARISON_PROTOCOL,
    )

    baseline_config = _mapping(
        baseline.get("training_config"), context="baseline training_config"
    )
    candidate_config = _mapping(
        candidate.get("training_config"), context="candidate training_config"
    )
    _validate_config_keys(baseline_config, candidate_config)
    baseline_capacity = _capacity(
        baseline,
        baseline_config,
        context="baseline capacity",
        report_field="expanded_capacity",
        expected=_BASELINE_CAPACITY,
    )
    candidate_capacity = _capacity(
        candidate,
        candidate_config,
        context="candidate scale",
        report_field="capacity",
        expected=_CANDIDATE_CAPACITY,
    )
    baseline_activation = _activation_checkpointing(
        baseline_config, context="baseline", legacy_default=True
    )
    candidate_activation = _activation_checkpointing(
        candidate_config, context="candidate", legacy_default=False
    )
    baseline_compact = baseline_config.get("compact_kbo_channels", False)
    candidate_compact = candidate_config.get("compact_kbo_channels", False)
    if not isinstance(baseline_compact, bool) or not isinstance(candidate_compact, bool):
        raise ValueError("compact_kbo_channels must be boolean when declared")
    if baseline_compact is not True or candidate_compact is not True:
        raise ValueError(
            "compact_kbo_channels must be true for both runs: production scale "
            "comparisons reject legacy dense channels with inactive trainable parameters"
        )

    baseline_epochs = _integer(
        baseline_config.get("epochs"), context="baseline training_config.epochs", minimum=1
    )
    candidate_epochs = _integer(
        candidate_config.get("epochs"), context="candidate training_config.epochs", minimum=1
    )
    if baseline_epochs != candidate_epochs:
        raise ValueError("baseline and candidate configured epochs differ")

    baseline_lineage = _lineage(
        baseline, baseline_config, context="baseline capacity"
    )
    candidate_lineage = _lineage(
        candidate, candidate_config, context="candidate scale"
    )
    for field in (
        "dataset_fingerprint",
        "split_day_fingerprint",
        "split_days",
        "seed",
        "training_seasons",
        "validation_season",
        "held_out_test_season",
    ):
        _require_equal(
            baseline_lineage[field], candidate_lineage[field], context=field
        )

    baseline_policies = _variant_policies(baseline, context="baseline capacity")
    candidate_policies = _variant_policies(candidate, context="candidate scale")
    graph_policy = _require_equal(
        baseline_policies, candidate_policies, context="variant graph policies"
    )
    _validate_loader_lineage(
        baseline,
        baseline_config,
        baseline_lineage,
        context="baseline capacity",
    )
    _validate_loader_lineage(
        candidate,
        candidate_config,
        candidate_lineage,
        context="candidate scale",
    )

    baseline_runtime = _mapping(
        baseline.get("runtime_signature"), context="baseline runtime_signature"
    )
    candidate_runtime = _mapping(
        candidate.get("runtime_signature"), context="candidate runtime_signature"
    )
    if not baseline_runtime or not candidate_runtime:
        raise ValueError("runtime signatures must not be empty")
    runtime_signature = _require_equal(
        baseline_runtime, candidate_runtime, context="runtime signatures"
    )

    baseline_runs = _runs(
        baseline,
        context="baseline 128x3",
        run_group="expanded_128x3",
        configured_epochs=baseline_epochs,
        variant_policies=baseline_policies,
    )
    candidate_runs = _runs(
        candidate,
        context="candidate 256x3",
        run_group=None,
        configured_epochs=candidate_epochs,
        variant_policies=candidate_policies,
    )
    baseline_initialization = _validate_initialization_audit(
        baseline,
        baseline_runs,
        expected_seed=baseline_lineage["seed"],
        context="baseline 128x3",
    )
    candidate_initialization = _validate_initialization_audit(
        candidate,
        candidate_runs,
        expected_seed=candidate_lineage["seed"],
        context="candidate 256x3",
    )
    _validate_source_comparison(
        baseline,
        baseline_runs,
        context="baseline 128x3",
        comparison_group="expanded_128x3",
    )
    _validate_source_comparison(
        candidate,
        candidate_runs,
        context="candidate 256x3",
        comparison_group=None,
    )

    baseline_parameter_counts = _validate_parameter_audit(
        baseline,
        baseline_runs,
        context="baseline 128x3",
        audit_field="expanded_128x3",
    )
    candidate_parameter_counts = _validate_parameter_audit(
        candidate,
        candidate_runs,
        context="candidate 256x3",
        audit_field="parameter_count",
    )
    for variant in _VARIANTS:
        if candidate_parameter_counts[variant] <= baseline_parameter_counts[variant]:
            raise ValueError(
                f"256x3 {variant} does not have more parameters than 128x3 {variant}"
            )

    baseline_budget = _validate_budget_audit(
        baseline,
        baseline_runs,
        context="baseline 128x3",
        budget_group="expanded_128x3",
    )
    candidate_budget = _validate_budget_audit(
        candidate,
        candidate_runs,
        context="candidate 256x3",
        budget_group=None,
    )
    if (
        baseline_budget["attempted_optimizer_steps"]["full"]
        != candidate_budget["attempted_optimizer_steps"]["full"]
    ):
        raise ValueError(
            "baseline and candidate attempted optimizer-step budgets differ"
        )
    for field in ("optimizer_steps", "skipped_optimizer_steps"):
        if baseline_budget[field]["full"] != candidate_budget[field]["full"]:
            raise ValueError(f"baseline and candidate {field} differ")

    baseline_counts = baseline_runs["full"]["validation_loss_sample_counts"]
    candidate_counts = candidate_runs["full"]["validation_loss_sample_counts"]
    if baseline_counts != baseline_runs["node_only"]["validation_loss_sample_counts"]:
        raise ValueError("baseline full/node_only validation sample counts differ")
    if candidate_counts != candidate_runs["node_only"]["validation_loss_sample_counts"]:
        raise ValueError("candidate full/node_only validation sample counts differ")
    if baseline_counts != candidate_counts:
        raise ValueError("baseline and candidate validation sample counts differ")
    _validate_root_sample_audit(
        baseline, baseline_counts, context="baseline 128x3"
    )
    _validate_root_sample_audit(
        candidate, candidate_counts, context="candidate 256x3"
    )

    baseline_targets = {
        baseline_runs[variant]["validation_selection_target"] for variant in _VARIANTS
    }
    candidate_targets = {
        candidate_runs[variant]["validation_selection_target"] for variant in _VARIANTS
    }
    if len(baseline_targets) != 1 or baseline_targets != candidate_targets:
        raise ValueError("baseline and candidate validation selection targets differ")

    objective = {
        field: _plain(baseline_config[field])
        for field in _OBJECTIVE_FIELDS
        if field in baseline_config
    }
    if set(objective) != set(_OBJECTIVE_FIELDS):
        missing = ", ".join(sorted(set(_OBJECTIVE_FIELDS) - set(objective)))
        raise ValueError(f"training objective fields are missing: {missing}")
    objective["resolved_validation_selection_target"] = next(iter(baseline_targets))

    graph_config = {
        field: _plain(baseline_config[field])
        for field in _GRAPH_CONFIG_FIELDS
        if field in baseline_config
    }
    if set(graph_config) != set(_GRAPH_CONFIG_FIELDS):
        missing = ", ".join(sorted(set(_GRAPH_CONFIG_FIELDS) - set(graph_config)))
        raise ValueError(f"graph policy configuration fields are missing: {missing}")
    graph_config["variant_policies"] = graph_policy

    sampling_limits = {
        field: _plain(baseline_config[field])
        for field in _SAMPLING_FIELDS
        if field in baseline_config
    }
    if set(sampling_limits) != set(_SAMPLING_FIELDS):
        missing = ", ".join(sorted(set(_SAMPLING_FIELDS) - set(sampling_limits)))
        raise ValueError(f"sampling-limit fields are missing: {missing}")

    baseline_losses = {
        variant: float(baseline_runs[variant]["validation_selection_loss"])
        for variant in _VARIANTS
    }
    candidate_losses = {
        variant: float(candidate_runs[variant]["validation_selection_loss"])
        for variant in _VARIANTS
    }
    baseline_gap = baseline_losses["node_only"] - baseline_losses["full"]
    candidate_gap = candidate_losses["node_only"] - candidate_losses["full"]

    execution_differences = {
        "activation_checkpointing": {
            "baseline_128x3": baseline_activation,
            "candidate_256x3": candidate_activation,
            "equal": baseline_activation == candidate_activation,
            "comparison_allowed": True,
            "baseline_legacy_missing_normalized_to_false": (
                "activation_checkpointing" not in baseline_config
            ),
        },
    }

    report: dict[str, Any] = {
        "status": "completed",
        "protocol": SCALE_COMPARISON_PROTOCOL,
        "protocol_version": SCALE_COMPARISON_PROTOCOL_VERSION,
        "source_reports": {
            "baseline_128x3": str(baseline_path),
            "candidate_256x3": str(candidate_path),
        },
        "dataset_fingerprint": baseline_lineage["dataset_fingerprint"],
        "split_day_fingerprint": baseline_lineage["split_day_fingerprint"],
        "split_days": baseline_lineage["split_days"],
        "seed": baseline_lineage["seed"],
        "training_seasons": baseline_lineage["training_seasons"],
        "validation_season": baseline_lineage["validation_season"],
        "held_out_test_season": baseline_lineage["held_out_test_season"],
        "selection_split": "validation",
        "test_used_for_training_selection_or_comparison": False,
        "capacities": {
            "baseline_128x3": baseline_capacity,
            "candidate_256x3": candidate_capacity,
        },
        "training_objective": objective,
        "graph_policy": graph_config,
        "sampling_limits": sampling_limits,
        "runtime_audit": {
            "runtime_signatures_equal": True,
            "runtime_signature": runtime_signature,
            "permitted_execution_differences": execution_differences,
        },
        "validation_selection_comparison": {
            "lower_is_better": True,
            "baseline_128x3": {
                **baseline_losses,
                "node_only_minus_full": baseline_gap,
            },
            "candidate_256x3": {
                **candidate_losses,
                "node_only_minus_full": candidate_gap,
            },
            "candidate_minus_baseline": {
                variant: candidate_losses[variant] - baseline_losses[variant]
                for variant in _VARIANTS
            },
            "dependency_gap_change_256x3_minus_128x3": candidate_gap
            - baseline_gap,
            "interpretation": (
                "negative candidate_minus_baseline favors 256x3 on validation; "
                "positive node_only_minus_full favors relational messages"
            ),
        },
        "parameter_count_audit": {
            "baseline_128x3": baseline_parameter_counts,
            "candidate_256x3": candidate_parameter_counts,
            "increase_by_variant": {
                variant: candidate_parameter_counts[variant]
                - baseline_parameter_counts[variant]
                for variant in _VARIANTS
            },
            "candidate_to_baseline_ratio_by_variant": {
                variant: candidate_parameter_counts[variant]
                / baseline_parameter_counts[variant]
                for variant in _VARIANTS
            },
            "variant_counts_intentionally_distinct": True,
        },
        "initialization_audit": {
            "within_capacity_shared_parameters_equal": True,
            "baseline_128x3": baseline_initialization,
            "candidate_256x3": candidate_initialization,
        },
        "budget_audit": {
            "configured_epochs": baseline_epochs,
            "epochs_equal_across_capacities": True,
            "within_capacity_variants_equal": True,
            "baseline_128x3": baseline_budget,
            "candidate_256x3": candidate_budget,
            "optimizer_steps_equal_across_capacities": True,
        },
        "validation_sample_count_audit": {
            "available": True,
            "all_runs_equal": True,
            "loss_sample_counts": _plain(baseline_counts),
        },
        "limitations": [
            "This is one fixed training seed and cannot establish seed stability.",
            "Validation compares capacities; the held-out test is never loaded or evaluated.",
            "Batch size, accumulation, and attempted optimizer-step budget are required "
            "to match across capacities.",
            "Activation checkpointing is an execution-memory policy difference; exact "
            "floating-point trajectory identity is not claimed.",
            "The candidate changes hidden width and attention-head count together, so their "
            "individual effects are not identified.",
            "node_only retains graph-derived node and role features while removing messages.",
        ],
    }
    report = _plain(report)

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination in {baseline_path, candidate_path}:
            raise ValueError("output_path must not overwrite a source report")
        _atomic_json(destination, report)
    return report


__all__ = [
    "SCALE_COMPARISON_PROTOCOL",
    "SCALE_COMPARISON_PROTOCOL_VERSION",
    "SCALE_COMPARISON_REPORT",
    "compare_kbo_scale_reports",
]
