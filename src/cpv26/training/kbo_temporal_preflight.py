"""CUDA-measured execution-plan gate for temporal-v7 KBO training.

The plan is based on node and edge totals, never a fixed number of dates.  A
sample that exceeds either budget is retained as a singleton and measured; it
is never silently dropped.  Only a plan whose complete train/validation pass
stays within the configured CUDA reservation threshold may be selected.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from cpv26.data.kbo_dataset_loader import KBOGraphDatasetLike, open_kbo_graph_dataset
from cpv26.data.kbo_temporal_archive import _sample_fingerprint
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import KBORelGNNModel, collate_kbo_day_graphs
from cpv26.training import kbo_runner as runner
from cpv26.training.batch_transfer import prefetch_batches
from cpv26.training.kbo_temporal_batching import (
    TemporalSampleSize,
    load_temporal_sample_sizes,
)
from cpv26.training.optimizer_state import make_adamw

TEMPORAL_PREFLIGHT_PROTOCOL = "temporal_v7_cuda_budget_plan"
TEMPORAL_PREFLIGHT_PROTOCOL_VERSION = 1
DEFAULT_MAX_RESERVED_FRACTION = 0.85
TEMPORAL_VARIANTS = ("full", "node_only")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _reserved_limit(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_reserved_fraction must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError("max_reserved_fraction must be finite and in (0, 1]")
    return result


def _pack_split(
    split: str,
    days: Sequence[date],
    sizes: Mapping[date, TemporalSampleSize],
    *,
    max_nodes: int,
    max_edges: int,
) -> list[dict[str, Any]]:
    """Consecutively pack one split using topology budgets only."""

    result: list[dict[str, Any]] = []
    current: list[TemporalSampleSize] = []
    nodes = 0
    edges = 0

    def flush() -> None:
        nonlocal current, nodes, edges
        if not current:
            return
        oversize = len(current) == 1 and (
            current[0].nodes > max_nodes or current[0].edges > max_edges
        )
        result.append(
            {
                "split": split,
                "split_batch_index": len(result),
                "dates": [sample.day.isoformat() for sample in current],
                "sample_fingerprints": [sample.fingerprint for sample in current],
                "nodes": nodes,
                "edges": edges,
                "oversize_single_day": oversize,
            }
        )
        current, nodes, edges = [], 0, 0

    for day in days:
        try:
            sample = sizes[day]
        except KeyError as exc:
            raise ValueError(f"temporal sample index is missing {day.isoformat()}") from exc
        if current and (
            nodes + sample.nodes > max_nodes or edges + sample.edges > max_edges
        ):
            flush()
        current.append(sample)
        nodes += sample.nodes
        edges += sample.edges
        if sample.nodes > max_nodes or sample.edges > max_edges:
            flush()
    flush()
    return result


def build_temporal_execution_plan(
    dataset: KBOGraphDatasetLike | str | Path,
    config: runner.KBOTrainingConfig,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """Build an explicit, variant-shared train/validation batch plan."""

    selected = (
        dataset
        if not isinstance(dataset, (str, Path))
        else open_kbo_graph_dataset(dataset)
    )
    manifest = selected.manifest
    if manifest.get("dataset_version") != 7 or manifest.get("graph_schema") != "temporal_v7":
        raise ValueError("temporal execution planning requires dataset_version=7 temporal_v7")
    if not config.chronological:
        raise ValueError("temporal execution planning requires chronological=True")
    if config.max_pa_per_day or config.max_edges_per_route_per_day:
        raise ValueError("temporal execution planning requires uncapped graph samples")
    batching = manifest.get("temporal_batching")
    if not isinstance(batching, Mapping):
        raise ValueError("temporal_v7 manifest is missing its batching recommendation")
    node_budget = _positive_int(
        max_nodes if max_nodes is not None else batching.get("max_nodes_per_batch"),
        "max_nodes",
    )
    edge_budget = _positive_int(
        max_edges if max_edges is not None else batching.get("max_edges_per_batch"),
        "max_edges",
    )
    dataset_fingerprint = str(manifest.get("fingerprint"))
    policy_fingerprint = str(manifest.get("sampling_policy_fingerprint"))
    sizes = load_temporal_sample_sizes(
        selected.directory,
        dataset_fingerprint=dataset_fingerprint,
        sampling_policy_fingerprint=policy_fingerprint,
    )
    splits = runner._split_days(selected, config)
    batches = [
        *_pack_split(
            "train", splits["train"], sizes, max_nodes=node_budget, max_edges=edge_budget
        ),
        *_pack_split(
            "validation",
            splits["validation"],
            sizes,
            max_nodes=node_budget,
            max_edges=edge_budget,
        ),
    ]
    if not batches:
        raise ValueError("temporal execution plan has no train/validation batches")
    for index, row in enumerate(batches):
        next_row = batches[index + 1] if index + 1 < len(batches) else None
        row["prefetch_next"] = bool(
            next_row is not None
            and next_row["split"] == row["split"]
            and not row["oversize_single_day"]
            and not next_row["oversize_single_day"]
        )
    plan_core = {
        "dataset_fingerprint": dataset_fingerprint,
        "sampling_policy_fingerprint": policy_fingerprint,
        "budgets": {"max_nodes": node_budget, "max_edges": edge_budget},
        "batching_basis": "node_and_edge_totals_only",
        "fixed_day_count_cap": False,
        "prefetch_depth": 1,
        "ordered_batches": batches,
    }
    fingerprint = _sha256_json(plan_core)
    return {
        **plan_core,
        "plan_fingerprint": fingerprint,
        "variant_plan_fingerprints": {variant: fingerprint for variant in TEMPORAL_VARIANTS},
        "variants_share_exact_plan": True,
        "actual_batch_count": len(batches),
        "train_batch_count": sum(row["split"] == "train" for row in batches),
        "validation_batch_count": sum(row["split"] == "validation" for row in batches),
        "oversize_single_day_batches": sum(row["oversize_single_day"] for row in batches),
        "held_out_test": {
            "season": config.test_season,
            "graph_days_loaded": False,
            "labels_loaded": False,
            "sealed": True,
        },
    }


def validate_temporal_execution_plan(
    plan: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    variant: str,
) -> tuple[tuple[date, ...], ...]:
    """Validate a selected plan before either variant constructs a loader."""

    if variant not in TEMPORAL_VARIANTS:
        raise ValueError("temporal execution plan variant must be full or node_only")
    if plan.get("status") != "passed" or plan.get("selected_for_training") is not True:
        raise ValueError("temporal execution plan did not pass its CUDA memory gate")
    raw = plan.get("execution_plan")
    if not isinstance(raw, Mapping) or raw.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("temporal execution plan belongs to a different dataset")
    fingerprint = raw.get("plan_fingerprint")
    variants = raw.get("variant_plan_fingerprints")
    if (
        not isinstance(fingerprint, str)
        or not isinstance(variants, Mapping)
        or set(variants) != set(TEMPORAL_VARIANTS)
        or any(value != fingerprint for value in variants.values())
    ):
        raise ValueError("temporal full/node_only execution plans are not identical")
    core = {
        key: raw[key]
        for key in (
            "dataset_fingerprint",
            "sampling_policy_fingerprint",
            "budgets",
            "batching_basis",
            "fixed_day_count_cap",
            "prefetch_depth",
            "ordered_batches",
        )
    }
    if _sha256_json(core) != fingerprint:
        raise ValueError("temporal execution plan fingerprint is inconsistent")
    ordered = raw.get("ordered_batches")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("temporal execution plan has no ordered batches")
    return tuple(
        tuple(date.fromisoformat(str(value)) for value in row["dates"])
        for row in ordered
    )


def _cpu_batches(
    dataset: KBOGraphDatasetLike,
    rows: Sequence[Mapping[str, Any]],
    config: runner.KBOTrainingConfig,
) -> Iterator[Mapping[str, Any]]:
    for row in rows:
        days = [dataset.load_day(value) for value in row["dates"]]
        observed_fingerprints = [_sample_fingerprint(graph) for graph in days]
        if observed_fingerprints != list(row["sample_fingerprints"]):
            raise RuntimeError(
                "materialized temporal sample differs from its validation-bounded index"
            )
        batch = collate_kbo_day_graphs(
            days,
            device="cpu",
            max_pa_per_day=None,
            max_edges_per_route_per_day=None,
            seed=config.seed,
        )
        yield runner._prepare_graph_batch(batch, config)


def _planned_device_batches(
    dataset: KBOGraphDatasetLike,
    plan: Mapping[str, Any],
    config: runner.KBOTrainingConfig,
    device: Any,
) -> Iterator[Mapping[str, Any]]:
    """Honor prefetch barriers at split boundaries and oversize singletons."""

    rows = plan["ordered_batches"]
    start = 0
    while start < len(rows):
        end = start
        while end + 1 < len(rows) and rows[end]["prefetch_next"]:
            end += 1
        segment = rows[start : end + 1]
        yield from prefetch_batches(
            _cpu_batches(dataset, segment, config), device, mover=runner._move
        )
        start = end + 1


def _observed_batch_counts(batch: Mapping[str, Any]) -> tuple[int, int]:
    nodes = sum(int(value.shape[0]) for value in batch["node_features"].values())
    edges = sum(int(route.num_edges) for route in batch["routes"])
    return nodes, edges


def _is_cuda_oom(exc: BaseException) -> bool:
    """Recognize torch CUDA OOMs without binding this module to one torch ABI."""

    name = type(exc).__name__.replace("_", "").lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message


def _failure_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
) -> list[dict[str, Any]]:
    return [
        {
            "actual_batch_index": start_index + offset,
            "split": row["split"],
            "dates": row["dates"],
            "nodes": row["nodes"],
            "edges": row["edges"],
            "oversize_single_day": row["oversize_single_day"],
        }
        for offset, row in enumerate(rows)
    ]


def run_temporal_cuda_preflight(
    dataset: KBOGraphDatasetLike | str | Path,
    config: runner.KBOTrainingConfig,
    *,
    output: str | Path,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_FRACTION,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure every planned batch with full RelGNN and production prefetch."""

    limit = _reserved_limit(max_reserved_fraction)
    if limit > DEFAULT_MAX_RESERVED_FRACTION:
        raise ValueError(
            "temporal production preflight cannot exceed the 85% CUDA reservation limit"
        )
    selected = (
        dataset
        if not isinstance(dataset, (str, Path))
        else open_kbo_graph_dataset(dataset)
    )
    plan = build_temporal_execution_plan(
        selected, config, max_nodes=max_nodes, max_edges=max_edges
    )
    report: dict[str, Any] = {
        "protocol": TEMPORAL_PREFLIGHT_PROTOCOL,
        "protocol_version": TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "selected_for_training": False,
        "max_reserved_fraction": limit,
        "configuration": asdict(config),
        "execution_plan": plan,
        "measurements": [],
        "measurement_policy": {
            "variant": "full",
            "reason": "full executes every relation; node_only shares the exact plan",
            "optimizer": "persistent AdamW with state materialized by the first step",
            "prefetch": "one next CUDA batch resident on a dedicated copy stream",
            "coverage": "every actual train and validation batch unless a failure stops early",
        },
    }
    destination = Path(output).expanduser().resolve()
    runner._atomic_json(destination, report)
    torch: Any = None
    device: Any = None
    model: Any = None
    optimizer: Any = None
    scaler: Any = None
    device_batches: Any = None
    batch: Any = None
    losses: Any = None
    norms: Any = None
    try:
        device, dtype, runtime = runner._device_and_precision(config.device, config.amp)
        if device.type != "cuda":
            raise ValueError("temporal production preflight requires an explicit CUDA device")
        report["runtime"] = runtime
        torch, _ = require_torch()
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        gc.collect()
        torch.cuda.empty_cache()
        model = KBORelGNNModel(runner._model_config(selected, config))
        model.to(device)
        model.train()
        optimizer = make_adamw(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and dtype == torch.float16
        )
        optimizer.zero_grad(set_to_none=True)
        device_batches = iter(
            _planned_device_batches(selected, plan, config, device)
        )
        total_batches = int(plan["actual_batch_count"])
        for index, batch_spec in enumerate(plan["ordered_batches"]):
            torch.cuda.reset_peak_memory_stats(device)
            prefetch_window = [batch_spec]
            if batch_spec["prefetch_next"]:
                prefetch_window.append(plan["ordered_batches"][index + 1])
            try:
                batch = next(device_batches)
            except Exception as exc:
                window = _failure_window(prefetch_window, start_index=index)
                report.update(
                    status=(
                        "failed_cuda_oom"
                        if _is_cuda_oom(exc)
                        else "failed_during_prefetch_or_transfer"
                    ),
                    all_actual_batches_measured=False,
                    early_stop_safe=True,
                    failure_stage="prefetch_or_transfer",
                    failed_cuda_window=window if _is_cuda_oom(exc) else None,
                    failed_prefetch_window=window,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                runner._atomic_json(destination, report)
                raise
            failure_stage = "batch_validation"
            try:
                observed_nodes, observed_edges = _observed_batch_counts(batch)
                if (observed_nodes, observed_edges) != (
                    int(batch_spec["nodes"]),
                    int(batch_spec["edges"]),
                ):
                    raise RuntimeError("collated temporal batch differs from its indexed plan")
                failure_stage = "forward"
                with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                    losses = runner._losses(model(batch), batch, config)
                if not bool(torch.isfinite(losses["loss"])):
                    raise FloatingPointError("temporal preflight produced a non-finite loss")
                failure_stage = "backward"
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                norms = runner._clip_gradient_norms(model, config.gradient_clip)
                if not all(
                    math.isfinite(float(value.detach().cpu())) for value in norms.values()
                ):
                    raise FloatingPointError("temporal preflight produced non-finite gradients")
                failure_stage = "optimizer_step"
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled() and float(scaler.get_scale()) < previous_scale:
                    raise FloatingPointError("temporal preflight skipped an FP16 optimizer step")
                failure_stage = "cuda_synchronize_and_measure"
                torch.cuda.synchronize(device)
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                _, total_memory = torch.cuda.mem_get_info(device)
                fraction = peak_reserved / int(total_memory)
            except Exception as exc:
                if _is_cuda_oom(exc):
                    window = _failure_window(prefetch_window, start_index=index)
                    report.update(
                        status="failed_cuda_oom",
                        all_actual_batches_measured=False,
                        early_stop_safe=True,
                        failure_stage=failure_stage,
                        failed_batch=window[0],
                        failed_cuda_window=window,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                    runner._atomic_json(destination, report)
                raise
            measurement = {
                "actual_batch_index": index,
                "split": batch_spec["split"],
                "split_batch_index": batch_spec["split_batch_index"],
                "dates": batch_spec["dates"],
                "nodes": observed_nodes,
                "edges": observed_edges,
                "oversize_single_day": batch_spec["oversize_single_day"],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "total_memory_bytes": int(total_memory),
                "peak_reserved_fraction": fraction,
                "prefetch_depth": 1,
                "prefetched_next_batch": (
                    {
                        "split": prefetch_window[1]["split"],
                        "dates": prefetch_window[1]["dates"],
                        "nodes": prefetch_window[1]["nodes"],
                        "edges": prefetch_window[1]["edges"],
                        "oversize_single_day": prefetch_window[1][
                            "oversize_single_day"
                        ],
                    }
                    if len(prefetch_window) == 2
                    else None
                ),
            }
            report["measurements"].append(measurement)
            report["completed_actual_batch_count"] = index + 1
            if (index + 1) % 10 == 0 or index + 1 == total_batches:
                progress(
                    f"temporal preflight {index + 1}/{total_batches}; "
                    f"max reserved={fraction:.1%}"
                )
            losses = None
            batch = None
            if fraction > limit:
                report["status"] = "failed_memory_threshold"
                report["all_actual_batches_measured"] = False
                report["early_stop_safe"] = True
                report["failed_batch"] = measurement
                report["failure_reason"] = (
                    "oversize_single_day_exceeds_cuda_limit"
                    if measurement["oversize_single_day"]
                    or (
                        measurement["prefetched_next_batch"] is not None
                        and measurement["prefetched_next_batch"]["oversize_single_day"]
                    )
                    else "planned_batch_exceeds_cuda_limit_reduce_node_edge_budgets"
                )
                runner._atomic_json(destination, report)
                raise RuntimeError(
                    "temporal CUDA preflight rejected the plan: "
                    f"batch {index} reserved {fraction:.3%}, above {limit:.3%}"
                )
        try:
            next(device_batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError("temporal preflight produced more batches than its plan")
        peak = max(row["peak_reserved_fraction"] for row in report["measurements"])
        report.update(
            status="passed",
            selected_for_training=True,
            all_actual_batches_measured=True,
            peak_reserved_fraction=peak,
            completed_actual_batch_count=total_batches,
        )
        runner._atomic_json(destination, report)
        return report
    except Exception as exc:
        if report.get("status") == "running":
            report["status"] = "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            runner._atomic_json(destination, report)
        raise
    finally:
        close = getattr(device_batches, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
        if optimizer is not None:
            with suppress(Exception):
                optimizer.zero_grad(set_to_none=True)
        device_batches = None
        batch = None
        losses = None
        norms = None
        scaler = None
        model = None
        optimizer = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            if device is not None:
                with suppress(Exception):
                    torch.cuda.synchronize(device)
            gc.collect()
            with suppress(Exception):
                torch.cuda.empty_cache()
            if device is not None:
                with suppress(Exception):
                    torch.cuda.reset_peak_memory_stats(device)


def _retryable_memory_failure(report: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (retryable, hard_oversize_singleton_failure)."""

    status = report.get("status")
    if status == "failed_memory_threshold":
        failed = report.get("failed_batch")
        hard = isinstance(failed, Mapping) and failed.get("oversize_single_day") is True
        return True, hard
    if status not in {"failed_during_prefetch_or_transfer", "failed_cuda_oom"}:
        return False, False
    error = report.get("error")
    if not isinstance(error, Mapping):
        return False, False
    text = f"{error.get('type', '')}: {error.get('message', '')}".lower()
    oom = "outofmemory" in text.replace("_", "") or "out of memory" in text
    window = report.get("failed_cuda_window", report.get("failed_prefetch_window"))
    hard = bool(
        oom
        and isinstance(window, list)
        and any(
            isinstance(row, Mapping) and row.get("oversize_single_day") is True
            for row in window
        )
    )
    return oom, hard


def _batch_grouping_fingerprint(plan: Mapping[str, Any]) -> str:
    return _sha256_json(
        [
            {
                "split": row["split"],
                "dates": row["dates"],
                "prefetch_next": row["prefetch_next"],
            }
            for row in plan["ordered_batches"]
        ]
    )


def run_adaptive_temporal_cuda_preflight(
    dataset: KBOGraphDatasetLike | str | Path,
    config: runner.KBOTrainingConfig,
    *,
    output: str | Path,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_FRACTION,
    max_attempts: int = 12,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Try successively smaller topology budgets and select the first safe plan.

    Each measured attempt has a fresh model, optimizer, allocator cleanup, and
    exact plan.  Threshold/OOM failures retry at half the node and edge budgets.
    An isolated oversize singleton that itself fails is a hard stop because no
    topology-only batch plan can split it without changing graph semantics.
    """

    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= 32
    ):
        raise ValueError("max_attempts must be an integer between 1 and 32")
    limit = _reserved_limit(max_reserved_fraction)
    if limit > DEFAULT_MAX_RESERVED_FRACTION:
        raise ValueError(
            "temporal production preflight cannot exceed the 85% CUDA reservation limit"
        )
    selected = (
        dataset
        if not isinstance(dataset, (str, Path))
        else open_kbo_graph_dataset(dataset)
    )
    initial = build_temporal_execution_plan(
        selected, config, max_nodes=max_nodes, max_edges=max_edges
    )
    node_budget = int(initial["budgets"]["max_nodes"])
    edge_budget = int(initial["budgets"]["max_edges"])
    destination = Path(output).expanduser().resolve()
    final: dict[str, Any] = {
        "protocol": f"{TEMPORAL_PREFLIGHT_PROTOCOL}_adaptive",
        "protocol_version": TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "selected_for_training": False,
        "max_reserved_fraction": limit,
        "budget_search": "largest_configured_budget_then_halve_nodes_and_edges",
        "fixed_day_count_cap": False,
        "attempts": [],
    }
    runner._atomic_json(destination, final)
    measured_groupings: set[str] = set()
    last_error: dict[str, str] | None = None
    for attempt_index in range(max_attempts):
        candidate_plan = build_temporal_execution_plan(
            selected,
            config,
            max_nodes=node_budget,
            max_edges=edge_budget,
        )
        grouping = _batch_grouping_fingerprint(candidate_plan)
        if grouping in measured_groupings:
            final["attempts"].append(
                {
                    "attempt": attempt_index + 1,
                    "status": "skipped_duplicate_batch_grouping",
                    "budgets": candidate_plan["budgets"],
                    "batch_grouping_fingerprint": grouping,
                }
            )
        else:
            measured_groupings.add(grouping)
            attempt_path = destination.with_name(
                f".{destination.name}.attempt-{attempt_index + 1}.json"
            )
            candidate: dict[str, Any] | None = None
            failure: dict[str, str] | None = None
            progress(
                f"temporal preflight attempt {attempt_index + 1}/{max_attempts}: "
                f"nodes<={node_budget:,}, edges<={edge_budget:,}"
            )
            try:
                candidate = run_temporal_cuda_preflight(
                    selected,
                    config,
                    output=attempt_path,
                    max_nodes=node_budget,
                    max_edges=edge_budget,
                    max_reserved_fraction=limit,
                    progress=progress,
                )
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
                last_error = dict(failure)
                if attempt_path.is_file():
                    loaded = json.loads(attempt_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        candidate = loaded
            finally:
                attempt_path.unlink(missing_ok=True)
            if candidate is None:
                final.update(
                    status="failed_internal",
                    error=(
                        failure
                        if failure is not None
                        else {
                            "type": "RuntimeError",
                            "message": "candidate attempt produced no report",
                        }
                    ),
                )
                runner._atomic_json(destination, final)
                raise RuntimeError(
                    "temporal preflight attempt produced no auditable report"
                ) from None
            final["attempts"].append(candidate)
            runner._atomic_json(destination, final)
            if candidate.get("status") == "passed":
                final.update(
                    status="passed",
                    selected_for_training=True,
                    selected_attempt=attempt_index + 1,
                    execution_plan=candidate["execution_plan"],
                    peak_reserved_fraction=candidate["peak_reserved_fraction"],
                    all_actual_batches_measured=True,
                )
                runner._atomic_json(destination, final)
                return final
            retryable, hard_oversize = _retryable_memory_failure(candidate)
            if hard_oversize:
                final.update(
                    status="failed_oversize_single_day",
                    hard_failure=True,
                    failure_reason=(
                        "an isolated single-day graph exceeds the CUDA reservation limit; "
                        "lower the immutable temporal sampling policy"
                    ),
                )
                runner._atomic_json(destination, final)
                raise RuntimeError(final["failure_reason"]) from None
            if not retryable:
                final.update(
                    status="failed_internal",
                    hard_failure=True,
                    error=candidate.get("error"),
                )
                runner._atomic_json(destination, final)
                if failure is not None:
                    raise RuntimeError(
                        f"{failure['type']}: {failure['message']}"
                    ) from None
                raise RuntimeError("temporal preflight failed for a non-memory reason")

        if node_budget == 1 and edge_budget == 1:
            break
        node_budget = max(1, node_budget // 2)
        edge_budget = max(1, edge_budget // 2)

    final.update(
        status="failed_no_safe_budget",
        hard_failure=True,
        failure_reason=(
            "no node/edge budget produced a complete plan below the CUDA reservation limit"
        ),
    )
    if last_error is not None:
        final["last_error"] = last_error
    runner._atomic_json(destination, final)
    raise RuntimeError(final["failure_reason"]) from None


__all__ = [
    "DEFAULT_MAX_RESERVED_FRACTION",
    "TEMPORAL_PREFLIGHT_PROTOCOL",
    "TEMPORAL_PREFLIGHT_PROTOCOL_VERSION",
    "build_temporal_execution_plan",
    "run_adaptive_temporal_cuda_preflight",
    "run_temporal_cuda_preflight",
    "validate_temporal_execution_plan",
]
