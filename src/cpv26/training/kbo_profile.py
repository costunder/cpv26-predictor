"""Bounded, isolated timing experiments; never resume or write a training run.

All optimizer updates affect a private model copy. This is a diagnostic replay,
not training/evaluation, and must not compete with a live job on the same GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cpv26.data.kbo_graph_dataset import KBOGraphDataset
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import KBORelGNNConfig, KBORelGNNModel
from cpv26.training.batch_transfer import move_batch
from cpv26.training.kbo_runner import (
    KBOTrainingConfig,
    _clip_gradient_norms,
    _counts,
    _device_and_precision,
    _loader,
    _losses,
    _model_config,
    _move,
    _read_checkpoint,
    _runtime_memory,
)
from cpv26.training.optimizer_state import make_adamw


def select_windows(
    manifest: Mapping[str, Any], train_seasons: Sequence[int], count: int,
) -> dict[str, list[date]]:
    """Choose mid-season windows, separately for box-only and PA training data."""
    if count < 1:
        raise ValueError("count must be positive")
    groups: dict[str, list[date]] = defaultdict(list)
    for row in manifest["days"]:
        day = date.fromisoformat(row["day"])
        if day.year in train_seasons:
            groups["with_pa" if row["pa_queries"] > 0 else "box_only"].append(day)
    result: dict[str, list[date]] = {}
    for name, dates in sorted(groups.items()):
        latest = max(day.year for day in dates)
        dates = sorted(day for day in dates if day.year == latest)
        if len(dates) < count:
            raise ValueError(f"{name}/{latest}: need {count} days, found {len(dates)}")
        start = (len(dates) - count) // 2
        result[name] = dates[start : start + count]
    if not result:
        raise ValueError("no training dates to profile")
    return result


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def summarize_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Use a union, NOT a sum of overlapping CUDA kernels, within measured steps.

    This trace fraction is NOT DCGM GRACT/SM occupancy. Profiling itself adds
    overhead. Missing CUPTI/device events are unavailable, never zero activity.
    """
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    step_events = [e for e in events if e.get("name") == "cpv26/profile_step"]
    windows = _merge_intervals([(e["ts"], e["ts"] + e["dur"]) for e in step_events])
    wall_us = sum(end - start for start, end in windows)
    cuda_intervals: list[tuple[float, float]] = []
    cpu: dict[str, dict[str, Any]] = {}
    for event in events:
        start, end = float(event["ts"]), float(event["ts"] + event["dur"])
        overlaps = [(max(start, a), min(end, b)) for a, b in windows if start < b and end > a]
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}:
            cuda_intervals.extend(overlaps)
        elif event.get("cat") == "cpu_op" and overlaps:
            name = str(event["name"])
            record = cpu.setdefault(name, {"name": name, "calls": 0, "inclusive_cpu_ms": 0.0})
            record["calls"] += 1
            record["inclusive_cpu_ms"] += sum(b - a for a, b in overlaps) / 1000
    active = sum(b - a for a, b in _merge_intervals(cuda_intervals))
    available = wall_us > 0 and any(
        e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"} for e in events
    )
    return {
        "profiled_step_count": len(step_events),
        "step_wall_ms": wall_us / 1000,
        "cuda_active_ms": active / 1000 if available else None,
        "cuda_active_fraction": active / wall_us if available else None,
        "top_cpu_ops": sorted(cpu.values(), key=lambda r: -r["inclusive_cpu_ms"])[:20],
        "interpretation": (
            "CUDA event interval union / profiled step wall time; not DCGM utilization. "
            "CPU op times are inclusive and must not be added together. "
            "No device events means unavailable, not 0%."
        ),
    }


@contextmanager
def _stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    torch, _ = require_torch()
    started = time.perf_counter()
    with torch.profiler.record_function(f"cpv26/{name}"):
        yield
    timings[name] = time.perf_counter() - started


def _sync(device: Any) -> None:
    torch, _ = require_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _new_session(
    state: Mapping[str, Any], config: KBOTrainingConfig, device: Any,
    *, optimized: bool = True,
) -> Any:
    torch, _ = require_torch()
    torch.manual_seed(config.seed)
    model: Any = KBORelGNNModel(KBORelGNNConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.backbone.set_execution_optimization(optimized)
    model.to(device)
    model.train()
    optimizer = make_adamw(
        model, learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        checkpoint=state, clone_state=True,
    )
    _, dtype, _ = _device_and_precision(str(device), config.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    if scaler.is_enabled() and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, scaler, dtype


def _step(
    session: Any, batch: Any, config: KBOTrainingConfig, device: Any,
    *, index: int, statistics_enabled: bool,
) -> dict[str, float]:
    """Replay the runner's update with safety checks; optionally omit statistics ONLY."""
    torch, _ = require_torch()
    model, optimizer, scaler, dtype = session
    timings: dict[str, float] = {}
    with (
        _stage("forward_and_loss", timings),
        torch.autocast(device.type, enabled=dtype is not None, dtype=dtype),
    ):
        losses = _losses(model(batch), batch, config)
    with _stage("loss_finite_host_read", timings):
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError("non-finite diagnostic loss")
    with _stage("backward", timings):
        scaler.scale(losses["loss"]).backward()
    with _stage("statistics_host_reads", timings):
        if statistics_enabled:
            counts = _counts(batch, include_boxscore=model.config.include_boxscore_heads)
            for name, count in counts.items():
                float(losses[f"{name}_loss"].detach().cpu()) * count
    with _stage("clip_and_norm_host_reads", timings):
        scaler.unscale_(optimizer)
        norms = _clip_gradient_norms(model, config.gradient_clip)
        finite = all(math.isfinite(float(norm.detach().cpu())) for norm in norms.values())
        if not scaler.is_enabled() and not finite:
            raise FloatingPointError("non-finite diagnostic gradients")
    with _stage("optimizer", timings):
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        skipped = scaler.get_scale() < previous_scale
    with _stage("progress_host_read", timings):
        if statistics_enabled and (index + 1) % 10 == 0:
            float(losses["loss"].detach().cpu())
    timings["skipped_optimizer_steps"] = float(skipped)
    return timings


def _measure(
    session: Any, batches: Any, config: KBOTrainingConfig, device: Any,
    *, warmup: int, steps: int, resident: bool, statistics_enabled: bool,
    packed_transfers: bool = True,
) -> dict[str, Any]:
    iterator_started = time.perf_counter()
    iterator = iter(batches)
    iterator_startup = time.perf_counter() - iterator_started
    rows: list[dict[str, float]] = []
    first_batch_wait = 0.0
    started = 0.0
    for index in range(warmup + steps):
        if index == warmup:
            _sync(device)
            started = time.perf_counter()
        row: dict[str, float] = {}
        with _stage("next_batch_host_wait", row):
            raw = next(iterator)
        if index == 0:
            first_batch_wait = row["next_batch_host_wait"]
        with _stage("h2d_host_dispatch", row):
            batch = raw if resident else move_batch(raw, device, packed=packed_transfers)
        row.update(_step(
            session, batch, config, device, index=index,
            statistics_enabled=statistics_enabled,
        ))
        if index >= warmup:
            rows.append(row)
    drain = time.perf_counter()
    _sync(device)
    elapsed = time.perf_counter() - started
    final_drain = time.perf_counter() - drain
    return {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "milliseconds_per_batch": elapsed * 1000 / steps,
        "training_days_per_second": steps * config.batch_days / elapsed,
        "iterator_startup_seconds_excluded": iterator_startup,
        "first_batch_wait_seconds_excluded": first_batch_wait,
        "final_device_drain_seconds": final_drain,
        "skipped_optimizer_steps": int(sum(row["skipped_optimizer_steps"] for row in rows)),
        "host_stage_mean_ms": {
            key: statistics.mean(row[key] for row in rows) * 1000 for key in rows[0]
            if key != "skipped_optimizer_steps"
        },
        "host_stage_meaning": (
            "Host wall time, including waits where scalar reads synchronize. "
            "Backward kernels may be charged to a later host-read stage. "
            "These are not isolated GPU kernel durations."
        ),
    }


def _trace(
    session: Any, batches: Sequence[Any], config: KBOTrainingConfig, device: Any,
    path: Path, steps: int,
) -> dict[str, Any]:
    torch, _ = require_torch()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # Warm the private model/optimizer before enabling profiling.
    _step(session, batches[0], config, device, index=0, statistics_enabled=True)
    _sync(device)
    with torch.profiler.profile(activities=activities) as profiler:
        for index, batch in enumerate(batches[:steps]):
            with torch.profiler.record_function("cpv26/profile_step"):
                _step(session, batch, config, device, index=index, statistics_enabled=True)
                _sync(device)
            profiler.step()
    profiler.export_chrome_trace(str(path))
    with path.open(encoding="utf-8") as handle:
        return summarize_trace(json.load(handle))


def _comparisons(cases: Mapping[str, Any]) -> dict[str, Any]:
    stream = cases["stream"]["milliseconds_per_batch"]
    resident = cases["resident"]["milliseconds_per_batch"]
    lean = cases["resident_no_statistics"]["milliseconds_per_batch"]
    return {
        "removing_loader_and_transfers_speedup": stream / resident,
        "removing_statistics_host_reads_speedup": resident / lean,
        "interpretation": (
            "Ratios above 1 mean the isolated replay became faster. Small differences "
            "can be noise; repeat before attributing a cause. Resident mode removes "
            "BOTH input loading and H2D, not one alone. No-statistics still checks "
            "finite losses/gradients and performs identical clipping/optimizer updates."
        ),
    }


def _compare_optimizations(
    state: Mapping[str, Any], directory: Path, days: Sequence[date],
    config: KBOTrainingConfig, device: Any, *, warmup: int, steps: int,
    repeats: int, progress: Callable[[str], None],
) -> dict[str, Any]:
    """Alternate reference/optimized private replays; never reuse updated weights."""
    samples: dict[str, list[dict[str, Any]]] = {"reference": [], "optimized": []}
    execution_order: list[list[str]] = []
    for repeat in range(repeats):
        order = ["reference", "optimized"] if repeat % 2 == 0 else ["optimized", "reference"]
        execution_order.append(order)
        for name in order:
            optimized = name == "optimized"
            session = _new_session(state, config, device, optimized=optimized)
            loader = _loader(directory, days, config, epoch=0, training=True)
            result = _measure(
                session, loader, config, device, warmup=warmup, steps=steps,
                resident=False, statistics_enabled=True, packed_transfers=optimized,
            )
            samples[name].append(result)
            progress(
                f"comparison {repeat + 1}/{repeats} {name}: "
                f"{result['milliseconds_per_batch']:.3f} ms/batch"
            )
            del session, loader
            gc.collect()
    reference = statistics.median(row["milliseconds_per_batch"] for row in samples["reference"])
    optimized_ms = statistics.median(row["milliseconds_per_batch"] for row in samples["optimized"])
    return {
        "repeats": repeats, "execution_order": execution_order,
        "reference_median_ms_per_batch": reference,
        "optimized_median_ms_per_batch": optimized_ms,
        "speedup": reference / optimized_ms,
        "time_reduction_percent": 100 * (1 - optimized_ms / reference),
        "samples": samples,
        "reference": {"shared_route_context": False, "packed_transfers": False},
        "optimized": {"shared_route_context": True, "packed_transfers": True},
        "interpretation": (
            "Same checkpoint, seed, dates, batch size, workers, precision and optimizer. "
            "Both modes load real input batches and collect the same statistics. "
            "Every replay uses a new private model/optimizer copy. "
            "Ratio of median batch times, not GPU utilization. Startup/warmup excluded; "
            "filesystem cache and host scheduling are uncontrolled. Small changes can be noise."
        ),
    }


def _host_runtime() -> dict[str, Any]:
    torch, _ = require_torch()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        with suppress(OSError):
            affinity = sorted(os.sched_getaffinity(0))
    return {
        "logical_cpu_count": os.cpu_count(), "allowed_cpu_affinity": affinity,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "note": "Affinity is not a measurement of CPU utilization or cgroup CPU quota.",
    }


def profile_run(
    run_directory: str | Path, *, dataset_directory: str | Path | None = None,
    output_directory: str | Path | None = None, device: str = "cuda:0",
    steps: int = 12, warmup: int = 3, trace_steps: int = 3,
    batch_days: int | None = None, workers: int | None = None,
    device_idle: bool = False, compare_optimizations: bool = False, repeats: int = 3,
    progress: Callable[[str], None] = print,
) -> Path:
    """Replay a few TRAINING days. Checkpoint/config/cache are read-only inputs."""
    if min(steps, warmup) < 1 or not 0 <= trace_steps <= steps:
        raise ValueError("steps/warmup must be positive; trace_steps must be 0..steps")
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    if device.startswith("cuda") and not device_idle:
        raise ValueError(
            "Wait until training on this MIG has finished, then use --device-idle. "
            "Concurrent jobs invalidate timings; this tool never stops another process."
        )
    run = Path(run_directory).expanduser().resolve()
    with (run / "config.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    directory = Path(dataset_directory or saved["dataset_directory"]).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    if saved["dataset_fingerprint"] != dataset.manifest["fingerprint"]:
        raise ValueError("run/dataset fingerprint mismatch")
    torch, _ = require_torch()
    state = _read_checkpoint(run / "last.pt")
    if state["dataset_fingerprint"] != dataset.manifest["fingerprint"]:
        raise ValueError("checkpoint/dataset fingerprint mismatch")
    # config.json is not rewritten on resume; the checkpoint has the actual last settings.
    config = KBOTrainingConfig.from_dict(state["training_config"])
    config = replace(
        config, device=device, amp="off" if device == "cpu" else config.amp,
        batch_days=config.batch_days if batch_days is None else batch_days,
        workers=config.workers if workers is None else workers, chronological=True,
    )
    if config.accumulate_steps != 1:
        raise ValueError("this diagnostic currently requires accumulate_steps=1")
    windows = select_windows(
        dataset.manifest, config.train_seasons, (steps + warmup) * config.batch_days,
    )
    output = Path(output_directory or (
        Path("var/reports") / ("relgnn_profile_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ) + "_" + uuid4().hex[:8])
    )).expanduser().resolve()
    if output in (run, directory) or run in output.parents or directory in output.parents:
        raise ValueError("diagnostic output must be outside the run and graph dataset")
    if output.exists():
        raise FileExistsError("diagnostic output already exists; choose a new directory")
    selected, _, runtime = _device_and_precision(device, config.amp)
    expected = _model_config(dataset, config).to_dict()
    if KBORelGNNConfig(**state["model_config"]).to_dict() != expected:
        raise ValueError("checkpoint model configuration mismatch")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running", "diagnostic_only": True,
        "profile_schema_version": 2,
        "mode": "optimization_comparison" if compare_optimizations else "bottleneck_diagnostic",
        "host_runtime": _host_runtime(),
        "run_directory": str(run), "checkpoint": str(run / "last.pt"),
        "checkpoint_epoch": state["epoch"], "runtime": runtime,
        "configuration": asdict(config), "original_training": state["training_config"],
        "device_idle_asserted_by_user": device_idle,
        "windows": {},
        "limitations": [
            "Private checkpoint replay, not a measurement of the existing training PID.",
            "Only training dates; no validation/test evaluation or checkpoint writes.",
            "Filesystem cache state is uncontrolled; chronological windows, not full epochs.",
            "No-statistics is a diagnostic experiment, not an installed training fix.",
            "Trace event activity is not DCGM utilization or theoretical compute efficiency.",
            "Compare only while no other workload is running on the same MIG/device.",
        ],
    }
    try:
        for name, days in windows.items():
            progress(f"{name}: {days[0]}..{days[-1]}, batch_days={config.batch_days}")
            window: dict[str, Any] = {
                "days": [day.isoformat() for day in days], "cases": {},
            }
            report["windows"][name] = window
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
            if compare_optimizations:
                comparison = _compare_optimizations(
                    state, directory, days, config, selected, warmup=warmup,
                    steps=steps, repeats=repeats, progress=progress,
                )
                window["optimization_comparison"] = comparison
                progress("Optimization comparison: " + json.dumps({
                    "window": name,
                    **{key: comparison[key] for key in (
                        "reference_median_ms_per_batch", "optimized_median_ms_per_batch",
                        "speedup", "time_reduction_percent",
                    )},
                }, ensure_ascii=False))
                if trace_steps:
                    loader = _loader(directory, days, config, epoch=0, training=True)
                    resident = [_move(batch, selected) for batch in loader]
                    _sync(selected)
                    del loader
                    session = _new_session(state, config, selected)
                    try:
                        window["trace"] = _trace(
                            session, resident[warmup:], config, selected,
                            output / f"{name}_trace.json", trace_steps,
                        )
                    except RuntimeError as exc:
                        window["trace"] = {"unavailable": str(exc)}
                    del session, resident
                window.update(_runtime_memory(selected))
                gc.collect()
                continue
            session = _new_session(state, config, selected)
            loader = _loader(directory, days, config, epoch=0, training=True)
            window["cases"]["stream"] = _measure(
                session, loader, config, selected, warmup=warmup, steps=steps,
                resident=False, statistics_enabled=True,
            )
            del session, loader
            # Preload exactly the SAME batches, using the same collator/seed/limits.
            loader = _loader(directory, days, config, epoch=0, training=True)
            resident = [_move(batch, selected) for batch in loader]
            _sync(selected)
            del loader
            for case, statistics_enabled in (("resident", True), ("resident_no_statistics", False)):
                session = _new_session(state, config, selected)
                window["cases"][case] = _measure(
                    session, resident, config, selected, warmup=warmup, steps=steps,
                    resident=True, statistics_enabled=statistics_enabled,
                )
                del session
            window["comparisons"] = _comparisons(window["cases"])
            if trace_steps:
                session = _new_session(state, config, selected)
                try:
                    window["trace"] = _trace(
                        session, resident[warmup:], config, selected,
                        output / f"{name}_trace.json", trace_steps,
                    )
                except RuntimeError as exc:
                    window["trace"] = {"unavailable": str(exc)}
                del session
            window.update(_runtime_memory(selected))
            progress(json.dumps({"window": name, **window["comparisons"]}, ensure_ascii=False))
            del resident
            gc.collect()
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # The directory was exclusively created above. Never overwrite training artifacts.
        with (output / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        progress(f"Diagnostic report: {output / 'report.json'}")
    return output / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-idle", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--batch-days", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--compare-optimizations", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    profile_run(
        args.run_dir, dataset_directory=args.dataset, output_directory=args.output,
        device=args.device, device_idle=args.device_idle, steps=args.steps,
        warmup=args.warmup, trace_steps=args.trace_steps, batch_days=args.batch_days,
        workers=args.workers, compare_optimizations=args.compare_optimizations,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
