"""Pure helper tests: no model execution, CUDA requirement, or user data writes."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from cpv26.training import kbo_profile
from cpv26.training.kbo_profile import select_windows, summarize_trace


def _days(year: int, count: int, *, pa_queries: int, spacing: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "day": (date(year, 7, 1) + timedelta(days=index * spacing)).isoformat(),
            "pa_queries": pa_queries,
        }
        for index in range(count)
    ]


def _event(name: str, ts: float, dur: float, *, cat: str = "cpu_op") -> dict[str, Any]:
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "cat": cat}


def _step(ts: float, dur: float) -> dict[str, Any]:
    return _event("cpv26/profile_step", ts, dur, cat="user_annotation")


def test_windows_choose_each_groups_latest_training_year_and_central_dates() -> None:
    rows = (
        _days(2001, 11, pa_queries=0)
        + _days(2022, 9, pa_queries=0)
        + _days(2023, 11, pa_queries=100)
        + _days(2024, 9, pa_queries=200)
        + _days(2025, 13, pa_queries=0)
        + _days(2026, 13, pa_queries=300)
    )
    # Input ordering cannot change chronological window selection.
    manifest = {"days": list(reversed(rows))}
    result = select_windows(manifest, (2001, 2022, 2023, 2024), 3)

    assert result == {
        "box_only": [date(2022, 7, day) for day in (4, 5, 6)],
        "with_pa": [date(2024, 7, day) for day in (4, 5, 6)],
    }
    assert all(day.year not in (2025, 2026) for window in result.values() for day in window)


def test_windows_are_contiguous_available_dates_not_invented_calendar_dates() -> None:
    manifest = {
        "days": _days(2022, 9, pa_queries=0, spacing=2) + _days(2024, 9, pa_queries=1, spacing=3)
    }
    result = select_windows(manifest, (2022, 2024), 3)
    assert result["box_only"] == [date(2022, 7, day) for day in (7, 9, 11)]
    assert result["with_pa"] == [date(2024, 7, day) for day in (10, 13, 16)]


def test_both_groups_can_use_the_same_training_year_without_mixing_rows() -> None:
    first = _days(2024, 5, pa_queries=0)
    second = [{"day": date(2024, 8, day).isoformat(), "pa_queries": 1} for day in range(1, 6)]
    result = select_windows({"days": first + second}, (2024,), 3)
    assert result["box_only"] == [date(2024, 7, day) for day in (2, 3, 4)]
    assert result["with_pa"] == [date(2024, 8, day) for day in (2, 3, 4)]


@pytest.mark.parametrize("count", [0, -1])
def test_windows_require_positive_count(count: int) -> None:
    manifest = {"days": _days(2022, 5, pa_queries=0) + _days(2024, 5, pa_queries=1)}
    with pytest.raises(ValueError):
        select_windows(manifest, (2022, 2024), count)


@pytest.mark.parametrize("short_group", ["box_only", "with_pa"])
def test_windows_fail_if_latest_selected_year_is_short_instead_of_using_older_year(
    short_group: str,
) -> None:
    rows = (
        _days(2001, 9, pa_queries=0)
        + _days(2022, 2 if short_group == "box_only" else 9, pa_queries=0)
        + _days(2023, 9, pa_queries=1)
        + _days(2024, 2 if short_group == "with_pa" else 9, pa_queries=1)
    )
    with pytest.raises(ValueError):
        select_windows({"days": rows}, (2001, 2022, 2023, 2024), 3)


@pytest.mark.parametrize("pa_queries", [0, 1])
def test_missing_training_group_is_not_filled_from_held_out_year(pa_queries: int) -> None:
    rows = _days(2024, 7, pa_queries=pa_queries) + _days(2025, 7, pa_queries=1 - pa_queries)
    result = select_windows({"days": rows}, (2024,), 3)
    expected_group = "with_pa" if pa_queries else "box_only"
    assert result == {expected_group: [date(2024, 7, day) for day in (3, 4, 5)]}
    assert all(day.year == 2024 for window in result.values() for day in window)


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_cuda_profile_requires_idle_confirmation_before_path_or_torch_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str,
) -> None:
    run = tmp_path / "run"
    dataset = tmp_path / "graph"
    output = tmp_path / "diagnostic"
    run.mkdir()
    dataset.mkdir()
    sources = {
        run / "config.json": b'{"unchanged": "config"}',
        run / "last.pt": b"unchanged checkpoint bytes",
        dataset / "manifest.json": b'{"unchanged": "manifest"}',
    }
    for path, content in sources.items():
        path.write_bytes(content)
    original_paths = set(tmp_path.rglob("*"))

    def forbidden_access(*args: object, **kwargs: object) -> Any:
        pytest.fail("idle-device guard must run before filesystem or Torch access")

    with monkeypatch.context() as guard:
        guard.setattr(kbo_profile, "Path", forbidden_access)
        guard.setattr(kbo_profile, "require_torch", forbidden_access)
        with pytest.raises(ValueError, match="--device-idle"):
            kbo_profile.profile_run(
                run, dataset_directory=dataset, output_directory=output,
                device=device, device_idle=False,
            )

    assert not output.exists()
    assert set(tmp_path.rglob("*")) == original_paths
    assert {path: path.read_bytes() for path in sources} == sources


def test_trace_clips_cuda_events_to_step_and_unions_overlapping_categories() -> None:
    trace = {
        "traceEvents": [
            _step(1000, 4000),
            _event("before_kernel", 500, 1000, cat="kernel"),
            _event("kernel_a", 2000, 2000, cat="kernel"),
            _event("copy", 2500, 2000, cat="gpu_memcpy"),
            _event("clear", 4400, 1600, cat="gpu_memset"),
            _event("aten::cpu_only", 1000, 4000),
        ]
    }
    result = summarize_trace(trace)
    assert result["profiled_step_count"] == 1
    assert result["step_wall_ms"] == pytest.approx(4)
    assert result["cuda_active_ms"] == pytest.approx(3.5)
    assert result["cuda_active_fraction"] == pytest.approx(0.875)
    assert "top_cpu_ops" in result


def test_trace_does_not_count_idle_gap_between_profile_steps() -> None:
    result = summarize_trace(
        {
            "traceEvents": [
                _step(1000, 2000),
                _step(5000, 3000),
                _event("kernel", 0, 10000, cat="kernel"),
            ]
        }
    )
    assert result["profiled_step_count"] == 2
    assert result["step_wall_ms"] == pytest.approx(5)
    assert result["cuda_active_ms"] == pytest.approx(5)
    assert result["cuda_active_fraction"] == pytest.approx(1)


def test_trace_uses_union_of_overlapping_steps_as_wall_denominator() -> None:
    result = summarize_trace(
        {
            "traceEvents": [
                _step(1000, 4000),
                _step(3000, 3000),
                _event("kernel", 2000, 2000, cat="kernel"),
            ]
        }
    )
    assert result["profiled_step_count"] == 2
    assert result["step_wall_ms"] == pytest.approx(5)
    assert result["cuda_active_ms"] == pytest.approx(2)
    assert result["cuda_active_fraction"] == pytest.approx(0.4)


def test_concurrent_cuda_streams_cannot_produce_activity_above_one() -> None:
    first = {**_event("kernel_a", 0, 10000, cat="kernel"), "tid": 1}
    second = {**_event("kernel_b", 0, 10000, cat="kernel"), "tid": 2}
    result = summarize_trace({"traceEvents": [_step(0, 10000), first, second]})
    assert result["cuda_active_ms"] == pytest.approx(10)
    assert result["cuda_active_fraction"] == pytest.approx(1)


def test_trace_microseconds_become_milliseconds_without_integer_truncation() -> None:
    result = summarize_trace(
        {"traceEvents": [_step(0, 2500), _event("kernel", 1000.25, 249.5, cat="kernel")]}
    )
    assert result["step_wall_ms"] == pytest.approx(2.5)
    assert result["cuda_active_ms"] == pytest.approx(0.2495)
    assert result["cuda_active_fraction"] == pytest.approx(0.0998)


def test_cpu_only_trace_does_not_claim_measured_zero_gpu_activity() -> None:
    result = summarize_trace({"traceEvents": [_step(0, 2500), _event("aten::add", 100, 200)]})
    assert result["profiled_step_count"] == 1
    assert result["step_wall_ms"] == pytest.approx(2.5)
    assert result["cuda_active_ms"] is None
    assert result["cuda_active_fraction"] is None


def test_cpu_cuda_launch_calls_are_not_counted_as_device_kernel_activity() -> None:
    result = summarize_trace(
        {"traceEvents": [_step(0, 2500), _event("cudaLaunchKernel", 0, 2500, cat="cuda_runtime")]}
    )
    assert result["step_wall_ms"] == pytest.approx(2.5)
    assert result["cuda_active_ms"] is None
    assert result["cuda_active_fraction"] is None


def test_only_complete_duration_events_define_steps_and_device_intervals() -> None:
    result = summarize_trace(
        {
            "traceEvents": [
                _step(0, 1000),
                {**_step(0, 100000), "ph": "B"},
                {**_event("instant", 0, 1000, cat="kernel"), "ph": "i"},
            ]
        }
    )
    assert result["profiled_step_count"] == 1
    assert result["step_wall_ms"] == pytest.approx(1)
    assert result["cuda_active_ms"] is None


def test_cuda_events_outside_measured_steps_mean_zero_measured_overlap() -> None:
    result = summarize_trace(
        {"traceEvents": [_step(1000, 1000), _event("warmup_kernel", 0, 500, cat="kernel")]}
    )
    assert result["cuda_active_ms"] == pytest.approx(0)
    assert result["cuda_active_fraction"] == pytest.approx(0)


@pytest.mark.parametrize("events", [[], [_event("kernel", 0, 1000, cat="kernel")]])
def test_trace_without_profile_steps_cannot_report_step_gpu_utilization(
    events: list[dict[str, Any]],
) -> None:
    result = summarize_trace({"traceEvents": events})
    assert result["profiled_step_count"] == 0
    assert result["step_wall_ms"] == pytest.approx(0)
    assert result["cuda_active_ms"] is None
    assert result["cuda_active_fraction"] is None
