from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cpv26.training import resource_telemetry as telemetry


def test_numeric_distribution_reports_physical_range_without_rounding() -> None:
    assert telemetry.numeric_distribution([1, 2, 8, 9]) == {
        "count": 4,
        "minimum": 1.0,
        "mean": 5.0,
        "p50": 5.0,
        "maximum": 9.0,
    }
    assert telemetry.numeric_distribution([])["count"] == 0


def test_resource_interval_distinguishes_unavailable_gpu_from_zero() -> None:
    result = telemetry.summarize_resource_interval(
        {
            "wall_time": 10.0,
            "process_cpu_seconds": 3.0,
            "system_cpu_totals": (100, 20),
            "available_ram_bytes": 900,
            "process_peak_rss_bytes": 100,
            "gpu_utilization_percent": None,
            "gpu_memory_utilization_percent": None,
        },
        {
            "wall_time": 12.0,
            "process_cpu_seconds": 5.0,
            "system_cpu_totals": (200, 40),
            "available_ram_bytes": 700,
            "process_peak_rss_bytes": 250,
            "gpu_utilization_percent": None,
            "gpu_memory_utilization_percent": None,
        },
        allowed_cpu_count=4,
    )

    assert result["wall_seconds"] == 2.0
    assert result["process_cpu_utilization_percent_of_allowed_capacity"] == 25.0
    assert result["system_cpu_utilization_percent"] == 80.0
    assert result["available_ram_bytes_start"] == 900
    assert result["available_ram_bytes_end"] == 700
    assert result["process_peak_rss_bytes"] == 250
    assert result["gpu_utilization_percent_snapshot_mean"] is None


def test_host_inventory_reports_only_process_visible_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry, "allowed_cpu_ids", lambda: (2, 3, 6))
    monkeypatch.setattr(
        telemetry,
        "_meminfo",
        lambda: {"MemTotal": 1_000, "MemAvailable": 400},
    )
    monkeypatch.setattr(telemetry, "_cgroup_memory_limit", lambda: 800)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test-allocation")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _device: SimpleNamespace(name="NVIDIA MIG test"),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        get_num_threads=lambda: 3,
        get_num_interop_threads=lambda: 1,
    )

    result = telemetry.host_resource_inventory(
        torch, SimpleNamespace(type="cuda"), dataset_directory=None
    )

    assert result["allowed_cpu_ids"] == [2, 3, 6]
    assert result["allowed_cpu_count"] == 3
    assert result["physical_ram_bytes"] == 1_000
    assert result["available_ram_bytes_at_start"] == 400
    assert result["cgroup_memory_limit_bytes"] == 800
    assert result["visible_gpu_count"] == 1
    assert result["mig_partition_visible"] is True


def test_tensor_shape_manifest_flattens_nested_tensor_paths() -> None:
    torch = pytest.importorskip("torch")
    batch: dict[str, Any] = {
        "node_features": {"player": torch.zeros((3, 4))},
        "labels": [torch.ones(2, dtype=torch.int64)],
        "metadata": "unchanged",
    }

    result = telemetry.tensor_shape_manifest(batch, torch)

    assert result["node_features.player"]["shape"] == [3, 4]
    assert result["node_features.player"]["numel"] == 12
    assert result["labels[0]"]["shape"] == [2]
    assert all("metadata" not in key for key in result)
