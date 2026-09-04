"""Read-only, best-effort telemetry for production training."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any


def _meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    with suppress(OSError, UnicodeError, ValueError):
        for line in path.read_text(encoding="ascii").splitlines():
            name, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields:
                scale = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
                values[name] = int(fields[0]) * scale
    return values


def _positive_file_integer(path: Path) -> int | None:
    with suppress(OSError, UnicodeError, ValueError):
        raw = path.read_text(encoding="ascii").strip()
        if raw != "max" and int(raw) > 0:
            return int(raw)
    return None


def _cgroup_memory_limit() -> int | None:
    paths = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    values = [value for path in paths if (value := _positive_file_integer(path))]
    physical = _meminfo().get("MemTotal")
    plausible = [value for value in values if physical is None or value <= physical * 2]
    return min(plausible) if plausible else None


def _cgroup_memory_current() -> int | None:
    paths = (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    values = [value for path in paths if (value := _positive_file_integer(path))]
    return max(values) if values else None


def _process_current_rss_bytes(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return None
    with suppress(OSError, UnicodeError, ValueError):
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                return int(fields[1]) * 1024
    return None


def allowed_cpu_ids() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        with suppress(OSError):
            return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def host_resource_inventory(
    torch: Any, device: Any, *, dataset_directory: str | Path | None = None
) -> dict[str, Any]:
    """Describe only resources visible to this process; change no settings."""

    affinity = allowed_cpu_ids()
    memory = _meminfo()
    storage: dict[str, int] | None = None
    if dataset_directory is not None:
        statvfs = getattr(os, "statvfs", None)
        with suppress(OSError):
            if callable(statvfs):
                stat = statvfs(Path(dataset_directory).expanduser().resolve())
                storage = {
                    "block_size_bytes": int(stat.f_frsize),
                    "total_bytes": int(stat.f_blocks * stat.f_frsize),
                    "available_bytes": int(stat.f_bavail * stat.f_frsize),
                }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_name: str | None = None
    if getattr(device, "type", None) == "cuda":
        with suppress(Exception):
            gpu_name = str(torch.cuda.get_device_properties(device).name)
    scheduler_names = (
        "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        "PBS_JOBID", "NSLOTS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    )
    return {
        "logical_cpu_count": os.cpu_count(),
        "allowed_cpu_ids": list(affinity),
        "allowed_cpu_count": len(affinity),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "physical_ram_bytes": memory.get("MemTotal"),
        "available_ram_bytes_at_start": memory.get("MemAvailable"),
        "cgroup_memory_limit_bytes": _cgroup_memory_limit(),
        "storage": storage,
        "visible_gpu_count": (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        ),
        "selected_gpu_name": gpu_name,
        "mig_partition_visible": bool(
            (gpu_name is not None and "mig" in gpu_name.lower())
            or (
                visible is not None
                and any(item.strip().startswith("MIG-") for item in visible.split(","))
            )
        ),
        "scheduler_environment": {
            name.lower(): os.environ.get(name) for name in scheduler_names
        },
        "measurement_notes": {
            "affinity": "OS affinity, not a claim of exclusive cores",
            "ram": "/proc and cgroup counters when available; null is unavailable",
            "mig": "inferred from CUDA name or visible-device identifier",
        },
    }


def _process_peak_rss_bytes() -> int | None:
    try:
        resource: Any = importlib.import_module("resource")
    except ImportError:
        return None
    with suppress(Exception):
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    return None


def _proc_cpu_totals() -> tuple[int, int] | None:
    path = Path("/proc/stat")
    if not path.is_file():
        return None
    with suppress(OSError, UnicodeError, ValueError):
        fields = path.read_text(encoding="ascii").splitlines()[0].split()
        if fields and fields[0] == "cpu":
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle
    return None


def _gpu_utilization(torch: Any, device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gpu_utilization_percent": None,
        "gpu_memory_utilization_percent": None,
    }
    if getattr(device, "type", None) != "cuda":
        return result
    utilization = getattr(torch.cuda, "utilization", None)
    memory_usage = getattr(torch.cuda, "memory_usage", None)
    if callable(utilization):
        with suppress(Exception):
            result["gpu_utilization_percent"] = float(utilization(device))
    if callable(memory_usage):
        with suppress(Exception):
            result["gpu_memory_utilization_percent"] = float(memory_usage(device))
    return result


def resource_snapshot(torch: Any, device: Any) -> dict[str, Any]:
    """Capture cheap point-in-time counters around a measured interval."""

    return resource_snapshot_with_children(torch, device)


def resource_snapshot_with_children(
    torch: Any,
    device: Any,
    *,
    child_pids: Sequence[int] = (),
) -> dict[str, Any]:
    """Capture counters including explicitly owned DataLoader worker RSS."""

    memory = _meminfo()
    child_rss = {
        str(pid): value
        for pid in sorted(set(int(item) for item in child_pids))
        if (value := _process_current_rss_bytes(pid)) is not None
    }
    cuda: dict[str, int | None] = {
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if getattr(device, "type", None) == "cuda":
        cuda = {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return {
        "wall_time": time.perf_counter(),
        "process_cpu_seconds": time.process_time(),
        "system_cpu_totals": _proc_cpu_totals(),
        "available_ram_bytes": memory.get("MemAvailable"),
        "process_current_rss_bytes": _process_current_rss_bytes(os.getpid()),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "child_process_count": len(child_rss),
        "child_process_rss_bytes": sum(child_rss.values()),
        "child_process_rss_by_pid": child_rss,
        "cgroup_memory_current_bytes": _cgroup_memory_current(),
        **cuda,
        **_gpu_utilization(torch, device),
    }


def summarize_resource_interval(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    allowed_cpu_count: int,
) -> dict[str, Any]:
    wall = max(0.0, float(end["wall_time"]) - float(start["wall_time"]))
    process_cpu = max(
        0.0,
        float(end["process_cpu_seconds"]) - float(start["process_cpu_seconds"]),
    )
    normalized = (
        100.0 * process_cpu / wall / max(1, allowed_cpu_count) if wall > 0 else None
    )
    system_percent: float | None = None
    before_cpu = start.get("system_cpu_totals")
    after_cpu = end.get("system_cpu_totals")
    if isinstance(before_cpu, Sequence) and isinstance(after_cpu, Sequence):
        total = int(after_cpu[0]) - int(before_cpu[0])
        idle = int(after_cpu[1]) - int(before_cpu[1])
        if total > 0:
            system_percent = 100.0 * (total - idle) / total
    gpu_values = [
        float(value)
        for value in (
            start.get("gpu_utilization_percent"),
            end.get("gpu_utilization_percent"),
        )
        if isinstance(value, (int, float))
    ]
    gpu_memory_values = [
        float(value)
        for value in (
            start.get("gpu_memory_utilization_percent"),
            end.get("gpu_memory_utilization_percent"),
        )
        if isinstance(value, (int, float))
    ]
    return {
        "wall_seconds": wall,
        "process_cpu_seconds": process_cpu,
        "process_cpu_utilization_percent_of_allowed_capacity": normalized,
        "system_cpu_utilization_percent": system_percent,
        "available_ram_bytes_start": start.get("available_ram_bytes"),
        "available_ram_bytes_end": end.get("available_ram_bytes"),
        "process_current_rss_bytes": end.get("process_current_rss_bytes"),
        "process_peak_rss_bytes": end.get("process_peak_rss_bytes"),
        "child_process_count": end.get("child_process_count"),
        "child_process_rss_bytes": end.get("child_process_rss_bytes"),
        "child_process_rss_by_pid": end.get("child_process_rss_by_pid"),
        "cgroup_memory_current_bytes_start": start.get("cgroup_memory_current_bytes"),
        "cgroup_memory_current_bytes_end": end.get("cgroup_memory_current_bytes"),
        "gpu_utilization_percent_snapshot_mean": (
            sum(gpu_values) / len(gpu_values) if gpu_values else None
        ),
        "gpu_memory_utilization_percent_snapshot_mean": (
            sum(gpu_memory_values) / len(gpu_memory_values)
            if gpu_memory_values
            else None
        ),
        "utilization_note": (
            "GPU values are sparse vendor-counter snapshots when PyTorch exposes them; "
            "null means unavailable, not zero. Process CPU is normalized by affinity."
        ),
    }


def tensor_shape_manifest(value: Any, torch: Any) -> dict[str, dict[str, Any]]:
    """Flatten tensor paths into auditable dtype/device/shape records."""

    result: dict[str, dict[str, Any]] = {}

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            result[path] = {
                "shape": [int(size) for size in item.shape],
                "dtype": str(item.dtype),
                "device": str(item.device),
                "numel": int(item.numel()),
            }
        elif isinstance(item, Mapping):
            for name, child in item.items():
                visit(child, f"{path}.{name}" if path else str(name))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif hasattr(item, "__dataclass_fields__") and not isinstance(item, type):
            for name in item.__dataclass_fields__:
                visit(getattr(item, name), f"{path}.{name}" if path else str(name))

    visit(value, "")
    return result


def numeric_distribution(
    values: Sequence[int | float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "p50": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": median,
        "maximum": ordered[-1],
    }


__all__ = [
    "allowed_cpu_ids",
    "host_resource_inventory",
    "numeric_distribution",
    "resource_snapshot",
    "resource_snapshot_with_children",
    "summarize_resource_interval",
    "tensor_shape_manifest",
]
