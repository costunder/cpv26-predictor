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
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from cpv26.data.kbo_dataset_loader import KBOGraphDatasetLike, open_kbo_graph_dataset
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import KBORelGNNModel
from cpv26.training import kbo_runner as runner
from cpv26.training.kbo_temporal_batching import (
    TemporalSampleSize,
    load_temporal_sample_sizes,
)
from cpv26.training.optimizer_state import make_adamw
from cpv26.training.resource_telemetry import (
    allowed_cpu_ids,
    host_resource_inventory,
    numeric_distribution,
    resource_snapshot,
    resource_snapshot_with_children,
    summarize_resource_interval,
    tensor_shape_manifest,
)

TEMPORAL_PREFLIGHT_PROTOCOL = "temporal_v7_cuda_budget_plan"
TEMPORAL_PREFLIGHT_PROTOCOL_VERSION = 2
DEFAULT_MAX_RESERVED_FRACTION = 0.85
DEFAULT_MAX_HOST_MEMORY_USED_FRACTION = 0.85
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


def _open_preflight_dataset(
    dataset: KBOGraphDatasetLike | str | Path,
    config: runner.KBOTrainingConfig,
) -> KBOGraphDatasetLike:
    selected = (
        dataset
        if not isinstance(dataset, (str, Path))
        else open_kbo_graph_dataset(
            dataset, label_year_ceiling=config.validation_season
        )
    )
    label_year_ceiling = getattr(selected, "label_year_ceiling", None)
    if (
        selected.manifest.get("graph_schema") == "temporal_v7"
        and hasattr(selected, "label_year_ceiling")
        and label_year_ceiling != config.validation_season
    ):
        raise ValueError(
            "temporal preflight dataset must seal labels after the validation season"
        )
    return selected


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
    loader_workers: int | None = None,
    loader_prefetch_factor: int | None = None,
) -> dict[str, Any]:
    """Build an explicit, variant-shared train/validation batch plan."""

    selected = _open_preflight_dataset(dataset, config)
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
    workers = config.workers if loader_workers is None else loader_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("loader_workers must be a non-negative integer")
    prefetch_factor = (
        None
        if workers == 0
        else _positive_int(
            2 if loader_prefetch_factor is None else loader_prefetch_factor,
            "loader_prefetch_factor",
        )
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
    graph_days = [len(row["dates"]) for row in batches]
    effective: list[int] = []
    for split in ("train", "validation"):
        split_rows = [row for row in batches if row["split"] == split]
        for start in range(0, len(split_rows), config.accumulate_steps):
            effective.append(
                sum(
                    len(row["dates"])
                    for row in split_rows[start : start + config.accumulate_steps]
                )
            )
    physical_batching = {
        "unit": "graph_days",
        "physical_graph_days": numeric_distribution(graph_days),
        "effective_graph_days_per_optimizer_step": numeric_distribution(effective),
        "gradient_accumulation_steps": config.accumulate_steps,
        "data_parallel_workers": 1,
        "formula": (
            "effective graph-days = sum(dynamic physical graph-days in accumulation "
            "group) * data-parallel workers"
        ),
        "no_graph_or_event_dropped_by_batching": True,
    }
    plan_core = {
        "dataset_fingerprint": dataset_fingerprint,
        "sampling_policy_fingerprint": policy_fingerprint,
        "budgets": {"max_nodes": node_budget, "max_edges": edge_budget},
        "batching_basis": "node_and_edge_totals_only",
        "fixed_day_count_cap": False,
        "prefetch_depth": 1,
        "loader_runtime": {
            "workers": workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": workers > 0,
            "loader_instances": 2,
            "simultaneous_worker_pools": 2 if workers > 0 else 0,
            "total_worker_processes": workers * 2,
            "packed_transfers": True,
            "pin_memory": False,
        },
        "physical_batching": physical_batching,
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


def _loader_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Select a topology/time-stratified profiling window, not a training subset."""

    total = len(rows)
    if total < 1:
        raise ValueError("loader calibration requires at least one planned batch")
    target = min(total, max(8, math.ceil(math.sqrt(total))))
    if target == total:
        return tuple(rows)
    chronological = {
        round(index * (total - 1) / max(1, target // 2 - 1))
        for index in range(target // 2)
    }
    largest = sorted(
        range(total),
        key=lambda index: (int(rows[index]["nodes"]) + int(rows[index]["edges"]), index),
        reverse=True,
    )
    selected = set(chronological)
    for index in largest:
        if len(selected) >= target:
            break
        selected.add(index)
    return tuple(rows[index] for index in sorted(selected))


def _worker_candidates(allowed_cpus: int, calibration_batches: int) -> tuple[int, ...]:
    maximum = max(1, min(allowed_cpus, calibration_batches))
    candidates = {0, 1, maximum}
    value = 2
    while value < maximum:
        candidates.add(value)
        value *= 2
    return tuple(sorted(candidates))


def _loader_worker_pids(iterator: Any) -> tuple[int, ...]:
    return tuple(
        int(worker.pid)
        for worker in getattr(iterator, "_workers", ())
        if getattr(worker, "pid", None) is not None
    )


def _shutdown_loader_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        with suppress(Exception):
            shutdown()


def _host_memory_safety(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    physical = inventory.get("physical_ram_bytes")
    cgroup_limit = inventory.get("cgroup_memory_limit_bytes")
    limits = [
        int(value)
        for value in (physical, cgroup_limit)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    effective_limit = min(limits) if limits else None
    cgroup_current = after.get("cgroup_memory_current_bytes")
    available = after.get("available_ram_bytes")
    cgroup_is_effective = (
        isinstance(cgroup_limit, int)
        and not isinstance(cgroup_limit, bool)
        and effective_limit == cgroup_limit
    )
    if cgroup_is_effective:
        if isinstance(cgroup_current, int):
            used = cgroup_current
            source = "cgroup_current_over_effective_limit"
        else:
            # MemAvailable is host-wide and cannot prove safety inside a tighter
            # cgroup.  Fail closed when the authoritative current usage is absent.
            used = None
            source = "cgroup_current_unavailable"
    elif isinstance(effective_limit, int) and isinstance(available, int):
        used = max(0, effective_limit - min(effective_limit, available))
        source = "system_memavailable_over_effective_limit"
    else:
        used = None
        source = "unavailable"
    fraction = (
        float(used) / float(effective_limit)
        if isinstance(used, int) and isinstance(effective_limit, int)
        else None
    )
    safe = fraction is not None and fraction <= DEFAULT_MAX_HOST_MEMORY_USED_FRACTION
    return {
        "status": "passed" if safe else "rejected",
        "safe_for_selection": safe,
        "measurement_source": source,
        "max_used_fraction": DEFAULT_MAX_HOST_MEMORY_USED_FRACTION,
        "effective_limit_bytes": effective_limit,
        "used_bytes_after_simultaneous_residency": used,
        "used_fraction_after_simultaneous_residency": fraction,
        "available_ram_bytes_before": before.get("available_ram_bytes"),
        "available_ram_bytes_after": available,
        "main_process_current_rss_bytes_before": before.get(
            "process_current_rss_bytes"
        ),
        "main_process_current_rss_bytes": after.get("process_current_rss_bytes"),
        "loader_child_process_count": after.get("child_process_count"),
        "loader_child_rss_bytes": after.get("child_process_rss_bytes"),
        "loader_child_rss_by_pid": after.get("child_process_rss_by_pid"),
    }


def _measure_loader_candidate(
    dataset: KBOGraphDatasetLike,
    config: runner.KBOTrainingConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    prefetch_factor: int | None,
) -> dict[str, Any]:
    torch, _ = require_torch()
    cpu = torch.device("cpu")
    before = resource_snapshot(torch, cpu)
    inventory = host_resource_inventory(torch, cpu, dataset_directory=dataset.directory)
    loaders: list[Any] = []
    iterators: list[Any] = []
    waits: list[float] = []
    steady_indices: list[int] = []
    first_shapes: dict[str, Any] | None = None
    measured_graph_days = measured_nodes = measured_edges = 0
    try:
        split_rows = {
            split: tuple(row for row in rows if row["split"] == split)
            for split in ("train", "validation")
        }
        if any(not selected_rows for selected_rows in split_rows.values()):
            raise RuntimeError(
                "loader autotune calibration must include train and validation batches"
            )
        for split, selected_rows in split_rows.items():
            days = [
                date.fromisoformat(str(value))
                for row in selected_rows
                for value in row["dates"]
            ]
            loader = runner._loader(
                dataset.directory,
                days,
                config,
                epoch=0,
                training=split == "train",
                planned_rows=selected_rows,
                workers_override=workers,
                prefetch_factor_override=prefetch_factor,
                persistent_workers_override=workers > 0,
            )
            loaders.append(loader)
            iterators.append(iter(loader))

        flat_index = 0
        for selected_rows, iterator in zip(split_rows.values(), iterators, strict=True):
            for split_index, row in enumerate(selected_rows):
                started = time.perf_counter()
                batch = next(iterator)
                waits.append(time.perf_counter() - started)
                observed_nodes, observed_edges = _observed_batch_counts(batch)
                if (observed_nodes, observed_edges) != (
                    int(row["nodes"]),
                    int(row["edges"]),
                ):
                    raise RuntimeError(
                        "loader autotune batch differs from its execution plan"
                    )
                if first_shapes is None:
                    first_shapes = tensor_shape_manifest(batch, torch)
                if split_index > 0:
                    steady_indices.append(flat_index)
                    measured_graph_days += len(row["dates"])
                    measured_nodes += observed_nodes
                    measured_edges += observed_edges
                flat_index += 1
                del batch
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("loader autotune produced more batches than planned")
        if not steady_indices:
            steady_indices = list(range(len(waits)))
            measured_graph_days = sum(len(row["dates"]) for row in rows)
            measured_nodes = sum(int(row["nodes"]) for row in rows)
            measured_edges = sum(int(row["edges"]) for row in rows)
        worker_pids = tuple(
            pid for iterator in iterators for pid in _loader_worker_pids(iterator)
        )
        after = resource_snapshot_with_children(
            torch, cpu, child_pids=worker_pids
        )
        memory_safety = _host_memory_safety(before, after, inventory)
        steady_wait = sum(waits[index] for index in steady_indices)
        if steady_wait <= 0:
            raise RuntimeError("loader autotune observed a non-positive measured duration")
        return {
            "status": "measured",
            "eligible_for_selection": memory_safety["safe_for_selection"],
            "host_memory_safe": memory_safety["safe_for_selection"],
            "host_memory_safety": memory_safety,
            "workers": workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers_during_measurement": workers > 0,
            "production_persistent_workers": workers > 0,
            "loader_instances": 2,
            "simultaneous_worker_pools": 2 if workers > 0 else 0,
            "expected_worker_processes": workers * 2,
            "observed_worker_processes": len(worker_pids),
            "calibration_batch_count": len(rows),
            "startup_first_batch_seconds": waits[0],
            "startup_batch_seconds_by_split": [
                waits[0], waits[len(split_rows["train"])]
            ],
            "steady_state_wait_seconds": steady_wait,
            "steady_state_graph_days": measured_graph_days,
            "steady_state_nodes": measured_nodes,
            "steady_state_edges": measured_edges,
            "graph_days_per_second": measured_graph_days / steady_wait,
            "nodes_per_second": measured_nodes / steady_wait,
            "edges_per_second": measured_edges / steady_wait,
            "input_tensor_shapes": first_shapes,
            "resources": summarize_resource_interval(
                before,
                after,
                allowed_cpu_count=len(allowed_cpu_ids()),
            ),
        }
    finally:
        for iterator in iterators:
            _shutdown_loader_iterator(iterator)
        iterators.clear()
        loaders.clear()
        gc.collect()


def autotune_temporal_loader(
    dataset: KBOGraphDatasetLike,
    config: runner.KBOTrainingConfig,
    plan: Mapping[str, Any],
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure worker/prefetch candidates and select actual loader throughput."""

    rows = plan.get("ordered_batches")
    if not isinstance(rows, list) or not rows:
        raise ValueError("loader autotune requires a non-empty execution plan")
    calibration = _loader_calibration_rows(rows)
    cpu_count = len(allowed_cpu_ids())
    settings = [
        (workers, prefetch)
        for workers in _worker_candidates(cpu_count, len(calibration))
        for prefetch in ((None,) if workers == 0 else (1, 2, 4))
    ]
    measurements: list[dict[str, Any]] = []
    per_worker_rss_estimates: list[float] = []
    non_loader_used_bytes: int | None = None
    effective_memory_limit: int | None = None
    for workers, prefetch in settings:
        projected_fraction = None
        if (
            workers > 0
            and per_worker_rss_estimates
            and non_loader_used_bytes is not None
            and effective_memory_limit is not None
        ):
            projected_used = non_loader_used_bytes + math.ceil(
                max(per_worker_rss_estimates) * workers * 2
            )
            projected_fraction = projected_used / effective_memory_limit
        if (
            projected_fraction is not None
            and projected_fraction > DEFAULT_MAX_HOST_MEMORY_USED_FRACTION
        ):
            result = {
                "status": "skipped_projected_host_memory",
                "workers": workers,
                "prefetch_factor": prefetch,
                "eligible_for_selection": False,
                "host_memory_safe": False,
                "projected_used_fraction": projected_fraction,
                "max_used_fraction": DEFAULT_MAX_HOST_MEMORY_USED_FRACTION,
                "projection_basis": (
                    "largest measured per-worker RSS times both production loader pools"
                ),
            }
        else:
            try:
                result = _measure_loader_candidate(
                    dataset,
                    config,
                    calibration,
                    workers=workers,
                    prefetch_factor=prefetch,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "workers": workers,
                    "prefetch_factor": prefetch,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
        memory = result.get("host_memory_safety")
        if isinstance(memory, Mapping):
            limit = memory.get("effective_limit_bytes")
            used = memory.get("used_bytes_after_simultaneous_residency")
            child_rss = memory.get("loader_child_rss_bytes")
            main_before = memory.get("main_process_current_rss_bytes_before")
            main_after = memory.get("main_process_current_rss_bytes")
            if isinstance(limit, int) and limit > 0:
                effective_memory_limit = limit
            if workers > 0 and isinstance(child_rss, int) and child_rss > 0:
                per_worker_rss_estimates.append(child_rss / (workers * 2))
                if isinstance(used, int):
                    non_loader_used_bytes = max(0, used - child_rss)
            elif (
                workers == 0
                and isinstance(main_before, int)
                and isinstance(main_after, int)
                and main_after > main_before
            ):
                loader_rss = main_after - main_before
                per_worker_rss_estimates.append(loader_rss / 2)
                if isinstance(used, int):
                    non_loader_used_bytes = max(0, used - loader_rss)
        measurements.append(result)
        throughput = result.get("graph_days_per_second")
        progress(
            "loader autotune "
            f"workers={workers}, prefetch={prefetch}: "
            f"{throughput:.2f} graph-days/s"
            if isinstance(throughput, (int, float))
            else f"loader autotune workers={workers}, prefetch={prefetch}: failed"
        )
    passed = [
        row
        for row in measurements
        if row["status"] == "measured" and row.get("host_memory_safe") is not False
    ]
    if not passed:
        raise RuntimeError(
            "every measured DataLoader candidate failed or exceeded host-memory safety"
        )
    selected = max(
        passed,
        key=lambda row: (
            float(row["graph_days_per_second"]),
            -int(row["workers"]),
            -int(row["prefetch_factor"] or 0),
        ),
    )
    torch, _ = require_torch()
    device, _, _ = runner._device_and_precision(config.device, config.amp)
    return {
        "status": "measured",
        "selection_metric": "maximum steady-state graph-days per second",
        "host_memory_gate": {
            "required": True,
            "max_used_fraction": DEFAULT_MAX_HOST_MEMORY_USED_FRACTION,
            "selected_status": selected.get("host_memory_safety", {}).get("status"),
        },
        "selection_changes_transport_only": True,
        "training_plan_coverage_unchanged": True,
        "held_out_test_loaded": False,
        "calibration_policy": (
            "sqrt(batch_count), minimum eight where available; half time-stratified "
            "and remainder largest topology; first batch excluded as worker startup"
        ),
        "calibration_batch_count": len(calibration),
        "total_planned_batch_count": len(rows),
        "calibration_plan_indices": [rows.index(row) for row in calibration],
        "host_resources": host_resource_inventory(
            torch, device, dataset_directory=dataset.directory
        ),
        "candidates": measurements,
        "selected": {
            "workers": int(selected["workers"]),
            "prefetch_factor": selected["prefetch_factor"],
            "persistent_workers": int(selected["workers"]) > 0,
            "loader_instances": 2,
            "simultaneous_worker_pools": (
                2 if int(selected["workers"]) > 0 else 0
            ),
            "total_worker_processes": int(selected["workers"]) * 2,
            "packed_transfers": True,
            "pin_memory": False,
            "graph_days_per_second": selected["graph_days_per_second"],
            "host_memory_safe": selected.get("host_memory_safe", True),
            "host_memory_safety": selected.get("host_memory_safety"),
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
            "loader_runtime",
            "physical_batching",
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


def _planned_device_batches(
    dataset: KBOGraphDatasetLike,
    plan: Mapping[str, Any],
    config: runner.KBOTrainingConfig,
    device: Any,
    *,
    observer: Callable[[str, Any], None] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Use the two simultaneously resident production loader pools."""

    rows = plan["ordered_batches"]
    loader_runtime = plan["loader_runtime"]
    loaders: list[Any] = []
    iterators: list[Any] = []
    try:
        for split in ("train", "validation"):
            selected_rows = tuple(row for row in rows if row["split"] == split)
            days = [
                date.fromisoformat(str(value))
                for row in selected_rows
                for value in row["dates"]
            ]
            loader = runner._loader(
                dataset.directory,
                days,
                config,
                epoch=0,
                training=split == "train",
                planned_rows=selected_rows,
                workers_override=int(loader_runtime["workers"]),
                prefetch_factor_override=loader_runtime["prefetch_factor"],
                persistent_workers_override=bool(loader_runtime["persistent_workers"]),
            )
            loaders.append(loader)
            iterators.append(iter(loader))
        for split, iterator in zip(("train", "validation"), iterators, strict=True):
            selected_rows = tuple(row for row in rows if row["split"] == split)
            yield from runner._planned_device_batches(
                iterator, selected_rows, device, observer=observer
            )
    finally:
        for iterator in iterators:
            _shutdown_loader_iterator(iterator)
        iterators.clear()
        loaders.clear()


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
    loader_autotune: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure every planned batch with full RelGNN and production prefetch."""

    limit = _reserved_limit(max_reserved_fraction)
    if limit > DEFAULT_MAX_RESERVED_FRACTION:
        raise ValueError(
            "temporal production preflight cannot exceed the 85% CUDA reservation limit"
        )
    selected = _open_preflight_dataset(dataset, config)
    base_plan = build_temporal_execution_plan(
        selected, config, max_nodes=max_nodes, max_edges=max_edges
    )
    tuning = (
        autotune_temporal_loader(selected, config, base_plan, progress=progress)
        if loader_autotune is None
        else dict(loader_autotune)
    )
    selected_loader = tuning.get("selected")
    if (
        tuning.get("status") != "measured"
        or not isinstance(selected_loader, Mapping)
        or isinstance(selected_loader.get("workers"), bool)
        or not isinstance(selected_loader.get("workers"), int)
        or selected_loader.get("host_memory_safe") is not True
    ):
        raise ValueError(
            "temporal preflight requires a measured host-memory-safe loader autotune result"
        )
    plan = build_temporal_execution_plan(
        selected,
        config,
        max_nodes=max_nodes,
        max_edges=max_edges,
        loader_workers=int(selected_loader["workers"]),
        loader_prefetch_factor=selected_loader.get("prefetch_factor"),
    )
    dataset_label_year_ceiling = getattr(selected, "label_year_ceiling", None)
    report: dict[str, Any] = {
        "protocol": TEMPORAL_PREFLIGHT_PROTOCOL,
        "protocol_version": TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "selected_for_training": False,
        "max_reserved_fraction": limit,
        "configuration": asdict(config),
        "execution_plan": plan,
        "loader_autotune": tuning,
        "held_out_test_access": {
            "season": config.test_season,
            "label_year_ceiling": dataset_label_year_ceiling,
            "runtime_ceiling_verified": (
                dataset_label_year_ceiling == config.validation_season
                if hasattr(selected, "label_year_ceiling")
                else None
            ),
            "graph_samples_loaded": False,
            "labels_loaded": False,
        },
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
    resource_start: Mapping[str, Any] | None = None
    first_input_shapes: dict[str, Any] | None = None
    transfer_host_seconds = {
        "source_wait_seconds": 0.0,
        "h2d_host_dispatch_seconds": 0.0,
    }
    transfer_cuda_events: list[tuple[Any, Any]] = []

    def observe_transfer(name: str, value: Any) -> None:
        if name in transfer_host_seconds:
            transfer_host_seconds[name] += float(value)
        elif name == "h2d_cuda_event":
            start_event, end_event = value
            transfer_cuda_events.append((start_event, end_event))

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
        model_config = runner._model_config(selected, config)
        model = KBORelGNNModel(model_config)
        model.to(device)
        model.train()
        report["model_execution"] = {
            "configuration": model_config.to_dict(),
            "architecture_contract": model.architecture_contract(),
            "route_edge_chunk_size_configured": config.route_edge_chunk_size,
            "route_edge_chunking_is_lossless": True,
            "nodes_edges_events_dropped": 0,
        }
        runner._atomic_json(destination, report)
        optimizer = make_adamw(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and dtype == torch.float16
        )
        optimizer.zero_grad(set_to_none=True)
        resource_start = resource_snapshot(torch, device)
        device_batches = iter(
            _planned_device_batches(
                selected,
                plan,
                config,
                device,
                observer=observe_transfer,
            )
        )
        total_batches = int(plan["actual_batch_count"])
        for index, batch_spec in enumerate(plan["ordered_batches"]):
            batch_wall_started = time.perf_counter()
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
                if first_input_shapes is None:
                    first_input_shapes = tensor_shape_manifest(batch, torch)
                stage_events: dict[str, tuple[Any, Any]] = {}
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
                failure_stage = "forward"
                with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                    losses = runner._losses(model(batch), batch, config)
                forward_end.record()
                stage_events["forward_and_loss"] = (forward_start, forward_end)
                if not bool(torch.isfinite(losses["loss"])):
                    raise FloatingPointError("temporal preflight produced a non-finite loss")
                failure_stage = "backward"
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                backward_start.record()
                scaler.scale(losses["loss"]).backward()
                backward_end.record()
                stage_events["backward"] = (backward_start, backward_end)
                scaler.unscale_(optimizer)
                norms = runner._clip_gradient_norms(model, config.gradient_clip)
                if not all(
                    math.isfinite(float(value.detach().cpu())) for value in norms.values()
                ):
                    raise FloatingPointError("temporal preflight produced non-finite gradients")
                failure_stage = "optimizer_step"
                optimizer_start = torch.cuda.Event(enable_timing=True)
                optimizer_end = torch.cuda.Event(enable_timing=True)
                optimizer_start.record()
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_end.record()
                stage_events["optimizer_step"] = (optimizer_start, optimizer_end)
                if scaler.is_enabled() and float(scaler.get_scale()) < previous_scale:
                    raise FloatingPointError("temporal preflight skipped an FP16 optimizer step")
                failure_stage = "cuda_synchronize_and_measure"
                torch.cuda.synchronize(device)
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                _, total_memory = torch.cuda.mem_get_info(device)
                fraction = peak_reserved / int(total_memory)
                stage_seconds = {
                    name: float(start.elapsed_time(end)) / 1000.0
                    for name, (start, end) in stage_events.items()
                }
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
                "steady_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "steady_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "stage_seconds": stage_seconds,
                "collate_seconds": float(
                    batch.get("_runtime_telemetry", {}).get("collate_seconds", 0.0)
                ),
                "end_to_end_batch_wall_seconds": (
                    time.perf_counter() - batch_wall_started
                ),
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
        resource_end = resource_snapshot(torch, device)
        interval = summarize_resource_interval(
            resource_start,
            resource_end,
            allowed_cpu_count=len(allowed_cpu_ids()),
        )
        elapsed = float(interval["wall_seconds"])
        graph_days = sum(len(row["dates"]) for row in plan["ordered_batches"])
        total_nodes = sum(int(row["nodes"]) for row in plan["ordered_batches"])
        total_edges = sum(int(row["edges"]) for row in plan["ordered_batches"])
        h2d_seconds = sum(
            float(start.elapsed_time(end)) / 1000.0
            for start, end in transfer_cuda_events
        )
        report["resource_measurements"] = {
            "host_inventory": host_resource_inventory(
                torch, device, dataset_directory=selected.directory
            ),
            "interval": interval,
            "input_tensor_shapes_first_actual_batch": first_input_shapes,
            "stage_seconds": {
                **transfer_host_seconds,
                "h2d_cuda_seconds": h2d_seconds,
                **{
                    name: sum(
                        float(row["stage_seconds"][name])
                        for row in report["measurements"]
                    )
                    for name in ("forward_and_loss", "backward", "optimizer_step")
                },
                "collate_worker_seconds": sum(
                    float(row["collate_seconds"]) for row in report["measurements"]
                ),
            },
            "end_to_end_batch_wall_seconds": numeric_distribution(
                [
                    row["end_to_end_batch_wall_seconds"]
                    for row in report["measurements"]
                ]
            ),
            "throughput": {
                "graph_days_per_second": graph_days / elapsed if elapsed else None,
                "nodes_per_second": total_nodes / elapsed if elapsed else None,
                "edges_per_second": total_edges / elapsed if elapsed else None,
            },
            "steady_cuda_allocated_bytes": numeric_distribution(
                [row["steady_allocated_bytes"] for row in report["measurements"]]
            ),
            "steady_cuda_reserved_bytes": numeric_distribution(
                [row["steady_reserved_bytes"] for row in report["measurements"]]
            ),
            "peak_cuda_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in report["measurements"]
            ),
            "peak_cuda_reserved_bytes": max(
                row["peak_reserved_bytes"] for row in report["measurements"]
            ),
            "physical_batching": plan["physical_batching"],
        }
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
    selected = _open_preflight_dataset(dataset, config)
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
