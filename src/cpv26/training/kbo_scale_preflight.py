"""All-batch, two-pass CUDA memory preflight for production KBO RelGNNs.

The held-out test split is deliberately limited to manifest dates.  Neither this
module's audit nor its CUDA execution path loads a test ``GraphDay`` or a test
label payload.
"""

from __future__ import annotations

import gc
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from cpv26.data.kbo_graph_dataset import GraphDay, KBOGraphDataset
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import (
    KBORelGNNModel,
    collate_kbo_day_graphs,
    kbo_multitask_loss,
)
from cpv26.training.batch_transfer import move_batch
from cpv26.training.optimizer_state import make_adamw

from . import kbo_runner as runner

SCALE_PREFLIGHT_PROTOCOL_VERSION = 3
DEFAULT_MAX_RESERVED_FRACTION = 0.85
_QUERY_INDEX_NAMES = {
    "match": "match_home_team_index",
    "live_hit": "live_hit_player_index",
    "pa": "pa_batter_index",
    "box_pa": "box_pa_player_index",
    "box_pitch": "box_pitch_player_index",
}


def _dataset(value: KBOGraphDataset | str | Path) -> KBOGraphDataset:
    return value if isinstance(value, KBOGraphDataset) else KBOGraphDataset(value)


def _axis_zero(value: Any, name: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError(f"{name} must have at least one dimension")
    count = int(shape[0])
    if count < 0:
        raise ValueError(f"{name} has an invalid leading dimension")
    return count


def _optional_axis_zero(day: GraphDay, name: str) -> int:
    try:
        value = getattr(day, name)
    except AttributeError:
        return 0
    return _axis_zero(value, name)


def _sum_counts(rows: Sequence[Mapping[str, int]], name: str) -> dict[str, int]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}


def _day_workload(
    graph: GraphDay,
    config: runner.KBOTrainingConfig,
    *,
    training: bool,
) -> dict[str, Any]:
    node_counts = {
        name: _axis_zero(values, f"{name} node features")
        for name, values in graph.node_features.items()
    }
    route_edge_counts = {
        name: _axis_zero(values["source_index"], f"{name} source indices")
        for name, values in graph.routes.items()
    }
    edge_cap = config.max_edges_per_route_per_day or None
    effective_route_edge_counts = {
        name: min(count, edge_cap) if edge_cap is not None else count
        for name, count in route_edge_counts.items()
    }
    query_counts = {
        task: _optional_axis_zero(graph, index_name)
        for task, index_name in _QUERY_INDEX_NAMES.items()
    }
    effective_query_counts = dict(query_counts)
    pa_cap = config.max_pa_per_day or None
    if training and pa_cap is not None:
        effective_query_counts["pa"] = min(effective_query_counts["pa"], pa_cap)

    nodes = sum(node_counts.values())
    route_edges = sum(route_edge_counts.values())
    effective_route_edges = sum(effective_route_edge_counts.values())
    queries = sum(query_counts.values())
    effective_queries = sum(effective_query_counts.values())
    graph_state_work_units = (
        config.hidden_dim * config.layers * (nodes + effective_route_edges)
    )
    query_work_units = config.hidden_dim * effective_queries
    return {
        "date": graph.day.isoformat(),
        "node_counts": node_counts,
        "nodes": nodes,
        "route_edge_counts": route_edge_counts,
        "route_edges": route_edges,
        "effective_route_edge_counts": effective_route_edge_counts,
        "effective_route_edges": effective_route_edges,
        "query_counts": query_counts,
        "queries": queries,
        "effective_query_counts": effective_query_counts,
        "effective_queries": effective_queries,
        "estimated_work_units": graph_state_work_units + query_work_units,
        "estimated_graph_state_work_units": graph_state_work_units,
        "estimated_query_work_units": query_work_units,
    }


def _batch_workload(
    days: Sequence[Mapping[str, Any]],
    *,
    selection_kind: str,
    window_index: int | None,
) -> dict[str, Any]:
    if not days:
        raise ValueError("cannot summarize an empty workload batch")
    node_counts = _sum_counts([row["node_counts"] for row in days], "node_counts")
    route_edge_counts = _sum_counts(
        [row["route_edge_counts"] for row in days], "route_edge_counts"
    )
    effective_route_edge_counts = _sum_counts(
        [row["effective_route_edge_counts"] for row in days],
        "effective_route_edge_counts",
    )
    query_counts = _sum_counts([row["query_counts"] for row in days], "query_counts")
    effective_query_counts = _sum_counts(
        [row["effective_query_counts"] for row in days], "effective_query_counts"
    )
    result: dict[str, Any] = {
        "selection_kind": selection_kind,
        "window_index": window_index,
        "dates": [str(row["date"]) for row in days],
        "days": len(days),
        "node_counts": node_counts,
        "nodes": sum(node_counts.values()),
        "route_edge_counts": route_edge_counts,
        "route_edges": sum(route_edge_counts.values()),
        "effective_route_edge_counts": effective_route_edge_counts,
        "effective_route_edges": sum(effective_route_edge_counts.values()),
        "query_counts": query_counts,
        "queries": sum(query_counts.values()),
        "effective_query_counts": effective_query_counts,
        "effective_queries": sum(effective_query_counts.values()),
        "estimated_work_units": sum(int(row["estimated_work_units"]) for row in days),
        "estimated_graph_state_work_units": sum(
            int(row["estimated_graph_state_work_units"]) for row in days
        ),
        "estimated_query_work_units": sum(
            int(row["estimated_query_work_units"]) for row in days
        ),
    }
    return result


def _workload_order(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["estimated_work_units"]),
        int(row["effective_route_edges"]),
        int(row["nodes"]),
        int(row["effective_queries"]),
    )


def _candidate_order(
    row: Mapping[str, Any], *, dimension: str
) -> tuple[int, int, int, int, int]:
    if dimension == "rough_cost":
        value = int(row["estimated_work_units"])
    elif dimension == "route_edges":
        value = int(row["effective_route_edges"])
    elif dimension == "nodes":
        value = int(row["nodes"])
    elif dimension == "queries":
        value = int(row["effective_queries"])
    elif dimension.startswith("route:"):
        value = int(row["effective_route_edge_counts"].get(dimension[6:], 0))
    elif dimension.startswith("node:"):
        value = int(row["node_counts"].get(dimension[5:], 0))
    else:
        task = dimension.removeprefix("query:")
        value = int(row["effective_query_counts"][task])
    return value, *_workload_order(row)


def _candidate_dimensions(
    batches: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    route_names = sorted(
        {
            str(name)
            for batch in batches
            for name in batch["effective_route_edge_counts"]
        }
    )
    node_names = sorted(
        {str(name) for batch in batches for name in batch["node_counts"]}
    )
    return (
        "rough_cost",
        "route_edges",
        "nodes",
        "queries",
        *(f"route:{name}" for name in route_names),
        *(f"node:{name}" for name in node_names),
        *(f"query:{task}" for task in _QUERY_INDEX_NAMES),
    )


def _candidate_batches(
    batches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe batches maximizing independent activation-size dimensions."""

    dimensions = _candidate_dimensions(batches)
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    for dimension in dimensions:
        best = max(
            batches,
            key=partial(_candidate_order, dimension=dimension),
        )
        identity = tuple(str(value) for value in best["dates"])
        if identity not in selected:
            selected[identity] = {**best, "selection_dimensions": []}
        selected[identity]["selection_dimensions"].append(dimension)
    return list(selected.values())


def _split_workload(
    rows: Sequence[Mapping[str, Any]],
    config: runner.KBOTrainingConfig,
    *,
    training: bool,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("scale preflight split has no graph days")
    chronological_batches = [
        _batch_workload(
            rows[offset : offset + config.batch_days],
            selection_kind="chronological_non_overlapping_batch",
            window_index=offset // config.batch_days,
        )
        for offset in range(0, len(rows), config.batch_days)
    ]
    if config.chronological or not training:
        candidates = chronological_batches
        selection_note = (
            "actual sorted, non-overlapping batches used by the loader"
        )
    else:
        # A shuffled epoch can combine arbitrary dates.  The top-k individual
        # days form a conservative upper bound without pretending a particular
        # epoch permutation was audited.
        top_days = sorted(rows, key=_workload_order, reverse=True)[: config.batch_days]
        candidates = [
            _batch_workload(
                sorted(top_days, key=lambda row: str(row["date"])),
                selection_kind="conservative_top_k_day_upper_bound",
                window_index=None,
            )
        ]
        selection_note = (
            "conservative top-k-day upper bound; not asserted to be one realized shuffled batch"
        )
    worst = max(candidates, key=_workload_order)
    return {
        "days_scanned": len(rows),
        "chronological_batches": len(chronological_batches),
        "selection_note": selection_note,
        "worst_batch": worst,
        "candidate_batches": _candidate_batches(candidates),
        "batches": chronological_batches,
        "days": list(rows),
    }


def audit_kbo_scale_workload(
    dataset: KBOGraphDataset | str | Path,
    config: runner.KBOTrainingConfig,
) -> dict[str, Any]:
    """Scan train/validation graphs and select the most expensive candidate batch.

    Test dates are obtained from the manifest-backed split only so the report can
    prove their graph and label payloads remained sealed.
    """

    selected = _dataset(dataset)
    splits = runner._split_days(selected, config)
    audited: dict[str, Any] = {}
    for split_name, training in (("train", True), ("validation", False)):
        rows = [
            _day_workload(selected.load_day(day), config, training=training)
            for day in splits[split_name]
        ]
        audited[split_name] = _split_workload(rows, config, training=training)

    worst_candidates = []
    cuda_candidates: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        candidate = dict(audited[split_name]["worst_batch"])
        candidate["split"] = split_name
        worst_candidates.append(candidate)
        for raw_candidate in audited[split_name]["candidate_batches"]:
            cuda_candidates.append({**raw_candidate, "split": split_name})
    selected_worst = max(worst_candidates, key=_workload_order)
    test_days = splits["test"]
    audited["test"] = {
        "season": config.test_season,
        "dates_listed_from_manifest": len(test_days),
        "date_start": min(test_days).isoformat() if test_days else None,
        "date_end": max(test_days).isoformat() if test_days else None,
        "graph_days_loaded": False,
        "labels_loaded": False,
        "sealed": True,
    }
    return {
        "protocol_version": SCALE_PREFLIGHT_PROTOCOL_VERSION,
        "dataset_directory": str(getattr(selected, "directory", "<in-memory>")),
        "dataset_fingerprint": selected.manifest.get("fingerprint"),
        "candidate_config": asdict(config),
        "batching": {
            "train": (
                "chronological_non_overlapping"
                if config.chronological
                else "conservative_top_k_upper_bound_for_shuffled_loader"
            ),
            "validation": "chronological_non_overlapping",
        },
        "cost_model": {
            "formula": (
                "hidden_dim * layers * (nodes + effective_route_edges) "
                "+ hidden_dim * effective_queries"
            ),
            "purpose": (
                "diagnostic ranking only; the production CUDA gate measures every actual batch"
            ),
            "edge_cap_per_route_per_day": config.max_edges_per_route_per_day or None,
            "training_pa_cap_per_day": config.max_pa_per_day or None,
            "validation_pa_cap_per_day": None,
        },
        "splits": audited,
        "selected_worst_batch": selected_worst,
        "candidate_batches": cuda_candidates,
        "candidate_batch_count": len(cuda_candidates),
        "candidate_selection_dimensions": list(
            dict.fromkeys(
                dimension
                for candidate in cuda_candidates
                for dimension in candidate["selection_dimensions"]
            )
        ),
        "held_out_test_policy": (
            "test dates may be counted from manifest metadata; test GraphDay and labels are sealed"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        partial.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _validate_max_reserved_fraction(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("max_reserved_fraction must be finite and in (0, 1]")
    return float(value)


def _enforce_reserved_fraction(
    report: dict[str, Any],
    *,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_FRACTION,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Annotate/write a measured report, then reject an unsafe reservation peak."""

    limit = _validate_max_reserved_fraction(max_reserved_fraction)
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("preflight report is missing execution memory measurements")
    observed = execution.get("peak_reserved_fraction")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError("preflight report has no numeric peak_reserved_fraction")
    fraction = float(observed)
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("peak_reserved_fraction must be finite and non-negative")
    passed = fraction <= limit
    total = int(execution.get("total_memory_bytes", 0))
    peak = int(execution.get("peak_reserved_bytes", 0))
    report["memory_safety"] = {
        "passed": passed,
        "max_reserved_fraction": limit,
        "peak_reserved_fraction": fraction,
        "threshold_reserved_bytes": int(total * limit),
        "headroom_to_threshold_bytes": int(total * limit) - peak,
    }
    report["status"] = "passed" if passed else "failed_memory_threshold"
    if output is not None:
        _atomic_json(Path(output), report)
    if not passed:
        destination = f"; report written to {Path(output).expanduser().resolve()}" if output else ""
        raise RuntimeError(
            "KBO scale preflight rejected the candidate: peak CUDA reserved fraction "
            f"{fraction:.3%} exceeds max_reserved_fraction {limit:.3%}{destination}"
        )
    return report


def _batch_counts(batch: Mapping[str, Any]) -> dict[str, Any]:
    node_counts = {
        name: int(values.shape[0]) for name, values in batch["node_features"].items()
    }
    route_edge_counts = {
        str(route.route_name): int(route.num_edges) for route in batch["routes"]
    }
    query_counts = {
        task: int(batch[index_name].numel())
        for task, index_name in _QUERY_INDEX_NAMES.items()
    }
    return {
        "days": len(batch["day_ids"]),
        "dates": list(batch["day_ids"]),
        "node_counts": node_counts,
        "nodes": sum(node_counts.values()),
        "route_edge_counts": route_edge_counts,
        "route_edges": sum(route_edge_counts.values()),
        "query_counts": query_counts,
        "queries": sum(query_counts.values()),
    }


def _validate_batch_matches_audit(
    batch: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    counts = _batch_counts(batch)
    expected = {
        "dates": [str(value) for value in candidate["dates"]],
        "node_counts": {
            str(name): int(value) for name, value in candidate["node_counts"].items()
        },
        "route_edge_counts": {
            str(name): int(value)
            for name, value in candidate["effective_route_edge_counts"].items()
        },
        "query_counts": {
            str(name): int(value)
            for name, value in candidate["effective_query_counts"].items()
        },
    }
    for field, expected_value in expected.items():
        if counts[field] != expected_value:
            raise RuntimeError(f"{context} batch {field} differs from its workload audit")
    return counts


def _loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: runner.KBOTrainingConfig,
) -> dict[str, Any]:
    return kbo_multitask_loss(
        output,
        batch,
        match_weight=config.match_weight,
        live_hit_weight=config.live_hit_weight,
        pa_weight=config.pa_weight,
        run_weight=config.run_weight,
        box_pa_weight=config.box_pa_weight,
        box_pitch_weight=config.box_pitch_weight,
    )


def _complete_training_step(
    model: Any,
    optimizer: Any,
    scaler: Any,
    batch: Mapping[str, Any],
    *,
    config: runner.KBOTrainingConfig,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Execute and verify one full optimizer step."""

    torch, _ = require_torch()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
        losses = _loss(model(batch), batch, config)
    if not bool(torch.isfinite(losses["loss"])):
        raise FloatingPointError("KBO scale preflight produced a non-finite loss")
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    gradient_norms = runner._clip_gradient_norms(model, config.gradient_clip)
    if not all(
        math.isfinite(float(value.detach().cpu())) for value in gradient_norms.values()
    ):
        raise FloatingPointError("KBO scale preflight produced non-finite gradients")
    previous_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if scaler.is_enabled() and float(scaler.get_scale()) < previous_scale:
        raise FloatingPointError("KBO scale preflight AdamW step was skipped by GradScaler")
    return losses


def _actual_batches(workload: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostic_dimensions = {
        (str(candidate["split"]), tuple(str(day) for day in candidate["dates"])): list(
            candidate["selection_dimensions"]
        )
        for candidate in workload["candidate_batches"]
    }
    batches: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        for split_index, raw in enumerate(workload["splits"][split_name]["batches"]):
            dates = tuple(str(day) for day in raw["dates"])
            batches.append(
                {
                    **raw,
                    "split": split_name,
                    "split_batch_index": split_index,
                    "diagnostic_selection_dimensions": diagnostic_dimensions.get(
                        (split_name, dates), []
                    ),
                }
            )
    return batches


def _validated_dates(
    batch_spec: Mapping[str, Any], config: runner.KBOTrainingConfig
) -> list[date]:
    split_name = str(batch_spec["split"])
    if split_name == "train":
        expected_years = set(config.train_seasons)
    elif split_name == "validation":
        expected_years = {config.validation_season}
    else:
        raise RuntimeError("preflight batch escaped the train/validation split")
    values = [date.fromisoformat(str(value)) for value in batch_spec["dates"]]
    if not values or any(
        value.year not in expected_years or value.year == config.test_season
        for value in values
    ):
        raise RuntimeError("preflight batch escaped the train/validation split")
    return values


class _BatchMeasurementFailure(RuntimeError):
    def __init__(self, record: dict[str, Any], cause: Exception) -> None:
        super().__init__(str(cause))
        self.record = record
        self.cause = cause


def _failure_memory_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    try:
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        return {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": int(total),
            "free_memory_bytes": int(free),
            "peak_reserved_fraction": peak_reserved / int(total),
            "headroom_bytes": int(total) - peak_reserved,
        }
    except Exception:
        return {}


def _optimizer_state_audit(optimizer: Any | None, torch: Any) -> dict[str, int]:
    if optimizer is None:
        return {
            "parameter_entries": 0,
            "tensor_values": 0,
            "total_tensor_bytes": 0,
            "cuda_tensor_bytes": 0,
        }
    tensor_values = 0
    total_bytes = 0
    cuda_bytes = 0
    seen: set[int] = set()

    def visit(value: Any) -> None:
        nonlocal tensor_values, total_bytes, cuda_bytes
        if torch.is_tensor(value):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            size = int(value.numel()) * int(value.element_size())
            tensor_values += 1
            total_bytes += size
            if value.device.type == "cuda":
                cuda_bytes += size
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(optimizer.state)
    return {
        "parameter_entries": len(optimizer.state),
        "tensor_values": tensor_values,
        "total_tensor_bytes": total_bytes,
        "cuda_tensor_bytes": cuda_bytes,
    }


def _measure_persistent_batch(
    selected: KBOGraphDataset,
    batch_spec: Mapping[str, Any],
    *,
    config: runner.KBOTrainingConfig,
    model: Any,
    optimizer: Any | None,
    scaler: Any,
    device: Any,
    dtype: Any,
    parameter_count: int,
    phase: str,
    actual_batch_index: int,
    clear_allocator_cache_before: bool = True,
    clear_allocator_cache_after: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Measure one batch while retaining model, AdamW, and optional allocator state."""

    torch, _ = require_torch()
    dates = _validated_dates(batch_spec, config)
    record: dict[str, Any] = {
        "status": "running",
        "phase": phase,
        "actual_batch_index": actual_batch_index,
        "split": str(batch_spec["split"]),
        "split_batch_index": int(batch_spec["split_batch_index"]),
        "dates": [value.isoformat() for value in dates],
        "diagnostic_selection_dimensions": list(
            batch_spec["diagnostic_selection_dimensions"]
        ),
        "estimated_work_units": int(batch_spec["estimated_work_units"]),
        "parameter_count": parameter_count,
        "allocator_cache_cleared_before_batch": clear_allocator_cache_before,
        "allocator_cache_cleared_after_batch": clear_allocator_cache_after,
    }
    batch_days: Any = None
    cpu_batch: Any = None
    gpu_batch: Any = None
    losses: Any = None
    try:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if clear_allocator_cache_before:
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        free_before, total_before = torch.cuda.mem_get_info(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        optimizer_state_before = _optimizer_state_audit(optimizer, torch)

        batch_days = [selected.load_day(value) for value in dates]
        cpu_batch = collate_kbo_day_graphs(
            batch_days,
            device="cpu",
            max_pa_per_day=(
                config.max_pa_per_day if batch_spec["split"] == "train" else None
            ),
            max_edges_per_route_per_day=config.max_edges_per_route_per_day,
            seed=config.seed,
        )
        cpu_batch = dict(runner._prepare_graph_batch(cpu_batch, config))
        cpu_counts = _validate_batch_matches_audit(
            cpu_batch, batch_spec, context="collated CPU"
        )
        if optimizer is None:
            model.to(device)
            optimizer = make_adamw(
                model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        gpu_batch = move_batch(cpu_batch, device, packed=True)
        gpu_counts = _batch_counts(gpu_batch)
        if gpu_counts != cpu_counts:
            raise RuntimeError("CPU/GPU batch counts differ after device transfer")
        losses = _complete_training_step(
            model,
            optimizer,
            scaler,
            gpu_batch,
            config=config,
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize(device)
        free_after, total_after = torch.cuda.mem_get_info(device)
        if int(total_before) != int(total_after):
            raise RuntimeError("CUDA total memory changed during KBO scale preflight")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        total = int(total_after)
        optimizer_state_after = _optimizer_state_audit(optimizer, torch)
        record.update(
            status="completed",
            forward_verified=True,
            backward_verified=True,
            adamw_step_verified=True,
            optimizer_state_materialized_before_measurement=(
                optimizer_state_before["parameter_entries"] > 0
            ),
            optimizer_state_before=optimizer_state_before,
            optimizer_state_after=optimizer_state_after,
            optimizer_state_entries_after_step=optimizer_state_after["parameter_entries"],
            loss=float(losses["loss"].detach().cpu()),
            cpu_batch_counts=cpu_counts,
            gpu_batch_counts=gpu_counts,
            batch_count_transfer_verified=True,
            allocated_before_bytes=allocated_before,
            reserved_before_bytes=reserved_before,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            total_memory_bytes=total,
            free_memory_before_bytes=int(free_before),
            free_memory_bytes=int(free_after),
            peak_reserved_fraction=peak_reserved / total,
            headroom_bytes=total - peak_reserved,
        )
        return optimizer, record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if cpu_batch is not None:
            record["cpu_batch_counts"] = _batch_counts(cpu_batch)
        if gpu_batch is not None:
            record["gpu_batch_counts"] = _batch_counts(gpu_batch)
        record["optimizer_state_after_failure"] = _optimizer_state_audit(
            optimizer, torch
        )
        record.update(_failure_memory_snapshot(torch, device))
        raise _BatchMeasurementFailure(record, exc) from exc
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        losses = None
        gpu_batch = None
        cpu_batch = None
        batch_days = None
        gc.collect()
        if clear_allocator_cache_after:
            torch.cuda.empty_cache()


def _overall_execution(
    warmup: Mapping[str, Any],
    materialization: Sequence[Mapping[str, Any]],
    evaluated: Sequence[Mapping[str, Any]],
    *,
    parameter_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measurements = [warmup, *materialization, *evaluated]
    totals = {int(row["total_memory_bytes"]) for row in measurements}
    if len(totals) != 1:
        raise RuntimeError("CUDA total memory changed across scale preflight batches")
    winner = max(
        measurements,
        key=lambda row: (
            int(row["peak_reserved_bytes"]),
            int(row["peak_allocated_bytes"]),
        ),
    )
    total = totals.pop()
    peak_reserved = max(int(row["peak_reserved_bytes"]) for row in measurements)
    peak_allocated = max(int(row["peak_allocated_bytes"]) for row in measurements)
    steady_state_cumulative_peak = int(
        evaluated[-1]["steady_state_cumulative_peak_reserved_bytes"]
    )
    observed_steady_state_peak = max(
        int(row["peak_reserved_bytes"]) for row in evaluated
    )
    if steady_state_cumulative_peak != observed_steady_state_peak:
        raise RuntimeError("steady-state cumulative CUDA reservation audit is inconsistent")
    execution = {
        "forward_verified": True,
        "backward_verified": True,
        "adamw_step_verified": True,
        "parameter_count": parameter_count,
        "batch_counts": winner["gpu_batch_counts"],
        "allocated_before_bytes": int(winner["allocated_before_bytes"]),
        "reserved_before_bytes": int(winner["reserved_before_bytes"]),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_memory_bytes": total,
        "free_memory_before_bytes": min(
            int(row["free_memory_before_bytes"]) for row in measurements
        ),
        "free_memory_bytes": min(int(row["free_memory_bytes"]) for row in measurements),
        "peak_reserved_fraction": peak_reserved / total,
        "headroom_bytes": total - peak_reserved,
        "warmup_steps": 1,
        "materialization_steps": len(materialization),
        "steady_state_steps": len(evaluated),
        "evaluated_batch_count": len(evaluated),
        "all_actual_batches_evaluated": True,
        "overall_peak_includes_warmup": True,
        "overall_peak_includes_materialization_pass": True,
        "first_batch_repeated_after_warmup": (
            bool(materialization)
            and bool(evaluated)
            and warmup["dates"] == materialization[0]["dates"] == evaluated[0]["dates"]
        ),
        "final_optimizer_state": evaluated[-1]["optimizer_state_after"],
        "optimizer_state_locked_before_steady_state": True,
        "steady_state_allocator_cache_cleared_once_before_pass": True,
        "steady_state_allocator_cache_retained_between_batches": True,
        "steady_state_cumulative_peak_reserved_bytes": steady_state_cumulative_peak,
        "steady_state_cumulative_peak_reserved_fraction": (
            steady_state_cumulative_peak / total
        ),
    }
    winning_batch = {
        "criterion": "maximum_observed_peak_reserved_bytes_then_allocated_bytes",
        "phase": winner["phase"],
        "actual_batch_index": winner["actual_batch_index"],
        "split": winner["split"],
        "split_batch_index": winner["split_batch_index"],
        "dates": winner["dates"],
        "diagnostic_selection_dimensions": winner["diagnostic_selection_dimensions"],
    }
    return execution, winning_batch


def _progress_update(
    progress: Callable[[str], None],
    *,
    phase: str,
    completed: int,
    total: int,
    measurements: Sequence[Mapping[str, Any]],
) -> None:
    if completed % 10 and completed != total:
        return
    peak = max(int(row["peak_reserved_bytes"]) for row in measurements)
    memory_total = int(measurements[0]["total_memory_bytes"])
    progress(
        f"scale preflight {phase}: {completed}/{total} actual batches; "
        f"observed max reserved={peak / 2**30:.2f} GiB ({peak / memory_total:.1%})"
    )


def run_kbo_scale_preflight(
    dataset: KBOGraphDataset | str | Path,
    config: runner.KBOTrainingConfig,
    *,
    output: str | Path | None = None,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_FRACTION,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure warmup and every actual train/validation batch with persistent AdamW."""

    report: dict[str, Any] = {
        "protocol_version": SCALE_PREFLIGHT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "candidate_config": asdict(config),
        "measurement_policy": {
            "model_and_optimizer": "one_fresh_instance_persistent_across_all_steps",
            "passes": [
                "first_batch_optimizer_warmup",
                "all_actual_batches_optimizer_state_materialization",
                "all_actual_batches_steady_state_measurement",
            ],
            "allocator_between_batches": (
                "steady_state_cache_retained_after_single_initial_empty_cache"
            ),
            "warmup_and_materialization_allocator": (
                "isolated_empty_cache_before_and_after_each_batch"
            ),
            "gate_scope": (
                "long-running steady-state caching-allocator high-water with persistent "
                "model/final optimizer state; isolated warmup/materialization peaks included"
            ),
        },
        "materialization_batches": [],
        "evaluated_batches": [],
        "held_out_test": {
            "season": config.test_season,
            "graph_days_loaded": False,
            "labels_loaded": False,
            "sealed": True,
        },
    }
    try:
        limit = _validate_max_reserved_fraction(max_reserved_fraction)
        if config.accumulate_steps != 1:
            raise ValueError("KBO scale preflight requires accumulate_steps=1")
        if not config.chronological:
            raise ValueError("KBO scale preflight requires chronological=True")
        selected = _dataset(dataset)
        device, dtype, runtime = runner._device_and_precision(config.device, config.amp)
        if device.type != "cuda":
            raise ValueError("KBO scale preflight requires an explicit CUDA device")
        report["runtime"] = runtime
        workload = audit_kbo_scale_workload(selected, config)
        report["workload_audit"] = workload
        actual_batches = _actual_batches(workload)
        if not actual_batches:
            raise RuntimeError("KBO scale preflight found no actual batches to execute")
        report["planned_actual_batch_count"] = len(actual_batches)
        progress(
            f"Scale preflight planned actual batches={len(actual_batches)}; "
            "one optimizer warmup plus materialization and steady-state passes"
        )

        model_config = runner._model_config(selected, config)
        torch, _ = require_torch()
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        model: Any = KBORelGNNModel(model_config)
        model.train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        optimizer: Any | None = None

        optimizer, warmup = _measure_persistent_batch(
            selected,
            actual_batches[0],
            config=config,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            dtype=dtype,
            parameter_count=parameter_count,
            phase="warmup",
            actual_batch_index=0,
        )
        report["warmup"] = warmup
        _progress_update(
            progress,
            phase="warmup",
            completed=1,
            total=1,
            measurements=[warmup],
        )
        report["status"] = "materializing_optimizer_state"
        for actual_batch_index, batch_spec in enumerate(actual_batches):
            optimizer, measured = _measure_persistent_batch(
                selected,
                batch_spec,
                config=config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                dtype=dtype,
                parameter_count=parameter_count,
                phase="materialization",
                actual_batch_index=actual_batch_index,
            )
            report["materialization_batches"].append(measured)
            report["completed_materialization_batch_count"] = actual_batch_index + 1
            _progress_update(
                progress,
                phase="materialization",
                completed=actual_batch_index + 1,
                total=len(actual_batches),
                measurements=[warmup, *report["materialization_batches"]],
            )

        materialized_state = report["materialization_batches"][-1][
            "optimizer_state_after"
        ]
        report["optimizer_state_after_materialization"] = materialized_state
        report["status"] = "measuring_all_actual_batches_steady_state"
        steady_state_cumulative_peak = 0
        for actual_batch_index, batch_spec in enumerate(actual_batches):
            optimizer, measured = _measure_persistent_batch(
                selected,
                batch_spec,
                config=config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                dtype=dtype,
                parameter_count=parameter_count,
                phase="steady_state",
                actual_batch_index=actual_batch_index,
                clear_allocator_cache_before=actual_batch_index == 0,
                clear_allocator_cache_after=False,
            )
            if (
                measured["optimizer_state_before"] != materialized_state
                or measured["optimizer_state_after"] != materialized_state
            ):
                cause = RuntimeError(
                    "AdamW optimizer state grew during the steady-state pass; "
                    "the materialization pass did not establish complete coverage"
                )
                measured["status"] = "failed_optimizer_state_growth"
                measured["error"] = {
                    "type": type(cause).__name__,
                    "message": str(cause),
                }
                raise _BatchMeasurementFailure(measured, cause)
            steady_state_cumulative_peak = max(
                steady_state_cumulative_peak,
                int(measured["peak_reserved_bytes"]),
            )
            measured["allocator_cache_retained_from_previous_steady_batch"] = (
                actual_batch_index > 0
            )
            measured["steady_state_cumulative_peak_reserved_bytes"] = (
                steady_state_cumulative_peak
            )
            measured["steady_state_cumulative_peak_reserved_fraction"] = (
                steady_state_cumulative_peak / int(measured["total_memory_bytes"])
            )
            report["evaluated_batches"].append(measured)
            report["completed_actual_batch_count"] = actual_batch_index + 1
            _progress_update(
                progress,
                phase="steady-state",
                completed=actual_batch_index + 1,
                total=len(actual_batches),
                measurements=[
                    warmup,
                    *report["materialization_batches"],
                    *report["evaluated_batches"],
                ],
            )

        execution, winning_batch = _overall_execution(
            warmup,
            report["materialization_batches"],
            report["evaluated_batches"],
            parameter_count=parameter_count,
        )
        report["execution"] = execution
        report["winning_batch"] = winning_batch
    except _BatchMeasurementFailure as exc:
        report["status"] = "execution_failed"
        report["failed_batch"] = exc.record
        report["execution_error"] = {
            "type": type(exc.cause).__name__,
            "message": str(exc.cause),
        }
        if output is not None:
            _atomic_json(Path(output), report)
        raise RuntimeError(
            "KBO scale preflight failed while measuring an actual train/validation batch"
        ) from exc.cause
    except Exception as exc:
        report["status"] = "execution_failed"
        report["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if output is not None:
            _atomic_json(Path(output), report)
        raise

    return _enforce_reserved_fraction(
        report, max_reserved_fraction=limit, output=output
    )


__all__ = [
    "DEFAULT_MAX_RESERVED_FRACTION",
    "SCALE_PREFLIGHT_PROTOCOL_VERSION",
    "audit_kbo_scale_workload",
    "run_kbo_scale_preflight",
]
