from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from cpv26.cli import app


def _runtime(tmp_path: Path, monkeypatch: Any) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    database = runtime / "cpv26.duckdb"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"fixture")
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))
    return runtime, database


def test_graph_build_forwards_vnext_schema(tmp_path: Path, monkeypatch: Any) -> None:
    from cpv26.data import kbo_graph_dataset

    runtime, _ = _runtime(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    class _Dataset:
        manifest = {"fingerprint": "f" * 64}

        @staticmethod
        def days() -> tuple[int, ...]:
            return (1, 2)

    def fake_build(*args: Any, **kwargs: Any) -> _Dataset:
        calls.append({"args": args, **kwargs})
        return _Dataset()

    monkeypatch.setattr(kbo_graph_dataset, "build_kbo_graph_dataset", fake_build)
    output = runtime / "datasets" / "kbo_graph_vnext"
    result = CliRunner().invoke(
        app,
        [
            "kbo-graph-build",
            "--graph-schema",
            "vnext",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["graph_schema"] == "vnext"
    assert calls[0]["args"][1] == output.resolve()


def test_graph_audit_writes_json_report(tmp_path: Path, monkeypatch: Any) -> None:
    from cpv26.data import kbo_graph_audit

    _runtime(tmp_path, monkeypatch)
    report = {
        "totals": {
            "days": 3,
            "node_occurrences": {"player": 12, "team": 6},
            "history_compression": {"unique_edge_occurrences": 20},
        }
    }
    monkeypatch.setattr(
        kbo_graph_audit, "audit_kbo_graph_dataset", lambda *_, **__: report
    )
    output = tmp_path / "audit.json"
    result = CliRunner().invoke(
        app,
        [
            "kbo-graph-audit",
            "--end-date",
            "2024-12-31",
            "--dataset",
            str(tmp_path / "graph"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "unique edge occurrences: 20" in result.output


def test_temporal_run_builds_indexes_and_starts_one_fixed_pair(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.data import kbo_temporal_archive
    from cpv26.training import kbo_temporal_workflow

    runtime, database = _runtime(tmp_path, monkeypatch)
    calls: dict[str, Any] = {}

    class _Archive:
        pass

    def fake_archive(*args: Any, **kwargs: Any) -> _Archive:
        calls["archive"] = (args, kwargs)
        return _Archive()

    def fake_index(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["index"] = (args, kwargs)
        return {}

    def fake_workflow(plan: Any, **kwargs: Any) -> dict[str, Any]:
        calls["plan"] = plan
        return {
            "validation_selection_comparison": {
                "full": 4.5,
                "node_only": 4.6,
                "node_only_minus_full": 0.1,
            }
        }

    monkeypatch.setattr(kbo_temporal_archive, "build_kbo_temporal_archive", fake_archive)
    monkeypatch.setattr(
        kbo_temporal_archive, "build_kbo_temporal_sample_index", fake_index
    )
    monkeypatch.setattr(kbo_temporal_workflow, "run_kbo_temporal_workflow", fake_workflow)
    dataset = runtime / "datasets" / "temporal"
    output = runtime / "runs" / "temporal"
    result = CliRunner().invoke(
        app,
        [
            "relgnn-temporal-run",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--workers",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["archive"][0] == (database.resolve(), dataset.resolve())
    assert calls["index"][1]["label_year_ceiling"] == 2025
    config = calls["plan"].config
    assert (config.hidden_dim, config.layers, config.heads) == (256, 3, 8)
    assert config.train_seasons == tuple(range(2001, 2025))
    assert config.validation_season == 2025
    assert config.test_season == 2026
    assert config.seed == 2026
    assert config.chronological is True
    assert config.activation_checkpointing is True
    assert config.workers == 4
    assert "2026 test labels and samples were not opened" in result.output


def test_capacity_command_reuses_baseline_and_requests_only_128x3(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_capacity_comparison
    from cpv26.training.kbo_runner import KBOTrainingConfig

    _runtime(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    base = KBOTrainingConfig(
        device="cpu", amp="off", workers=0, hidden_dim=64, layers=2, seed=17
    )
    (baseline / "matched_retraining_report.json").write_text(
        json.dumps({"seeds": [17, 23], "base_training_config": asdict(base)}),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {
            "validation_selection_comparison": {
                "baseline_64x2": {"node_only_minus_full": 0.1},
                "expanded_128x3": {"node_only_minus_full": 0.2},
            }
        }

    monkeypatch.setattr(
        kbo_capacity_comparison, "train_kbo_capacity_comparison", fake_compare
    )
    dataset, output = tmp_path / "graph", tmp_path / "capacity"
    result = CliRunner().invoke(
        app,
        [
            "relgnn-capacity-compare",
            "--baseline-suite",
            str(baseline),
            "--baseline-seed",
            "17",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["args"] == (dataset.resolve(), baseline.resolve(), output.resolve())
    assert calls[0]["baseline_seed"] == 17
    config = calls[0]["config"]
    assert (config.hidden_dim, config.layers, config.seed) == (128, 3, 17)


def test_pair_command_trains_exactly_the_two_condition_protocol(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_capacity_comparison

    _runtime(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    def fake_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {"validation_selection_comparison": {"node_only_minus_full": 0.125}}

    monkeypatch.setattr(
        kbo_capacity_comparison, "train_kbo_full_node_comparison", fake_pair
    )
    dataset, output = tmp_path / "vnext", tmp_path / "pair"
    result = CliRunner().invoke(
        app,
        [
            "relgnn-pair-train",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--amp",
            "off",
            "--workers",
            "0",
            "--seed",
            "23",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["args"] == (dataset.resolve(), output.resolve())
    config = calls[0]["config"]
    assert (config.hidden_dim, config.layers, config.seed) == (128, 3, 23)
    assert config.patience == 0
    assert config.route_schedule == "full"
    assert config.graph_control == "intact"


def test_pair_command_forwards_production_scale_execution_options(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_capacity_comparison

    _runtime(tmp_path, monkeypatch)
    calls: list[Any] = []

    def fake_pair(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["config"])
        return {"validation_selection_comparison": {"node_only_minus_full": 0.0}}

    monkeypatch.setattr(
        kbo_capacity_comparison, "train_kbo_full_node_comparison", fake_pair
    )
    result = CliRunner().invoke(
        app,
        [
            "relgnn-pair-train",
            "--dataset",
            str(tmp_path / "graph"),
            "--output",
            str(tmp_path / "pair"),
            "--device",
            "cpu",
            "--amp",
            "off",
            "--hidden-dim",
            "256",
            "--layers",
            "3",
            "--heads",
            "8",
            "--activation-checkpointing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    config = calls[0]
    assert (config.hidden_dim, config.layers, config.heads) == (256, 3, 8)
    assert config.activation_checkpointing is True
    assert config.compact_kbo_channels is False


def test_scale_preflight_uses_production_defaults_and_writes_requested_report(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_scale_preflight

    _runtime(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    def fake_preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {
            "execution": {
                "parameter_count": 26_140_772,
                "peak_reserved_bytes": 5 * 2**30,
                "peak_reserved_fraction": 0.5,
            }
        }

    monkeypatch.setattr(kbo_scale_preflight, "run_kbo_scale_preflight", fake_preflight)
    dataset = tmp_path / "graph"
    output = tmp_path / "preflight.json"
    result = CliRunner().invoke(
        app,
        [
            "relgnn-scale-preflight",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--seed",
            "2026",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["args"][0] == dataset.resolve()
    assert calls[0]["output"] == output.resolve()
    assert calls[0]["max_reserved_fraction"] == 0.85
    config = calls[0]["args"][1]
    assert (config.hidden_dim, config.layers, config.heads) == (256, 3, 8)
    assert config.batch_days == 8
    assert config.accumulate_steps == 1
    assert config.activation_checkpointing is True
    assert config.compact_kbo_channels is False
    assert config.chronological is True
    assert config.train_seasons == tuple(range(2001, 2025))
    assert (config.validation_season, config.test_season) == (2025, 2026)


def test_scale_report_forwards_source_reports_and_destination(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_scale_comparison

    _runtime(tmp_path, monkeypatch)
    baseline = tmp_path / "capacity_comparison_report.json"
    candidate = tmp_path / "candidate" / "full_node_comparison_report.json"
    candidate.parent.mkdir()
    baseline.write_text("{}", encoding="utf-8")
    candidate.write_text("{}", encoding="utf-8")
    output = tmp_path / "scale.json"
    calls: list[dict[str, Any]] = []

    def fake_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {
            "validation_selection_comparison": {
                "candidate_minus_baseline": {"full": -0.01},
                "dependency_gap_change_256x3_minus_128x3": 0.02,
            }
        }

    monkeypatch.setattr(kbo_scale_comparison, "compare_kbo_scale_reports", fake_compare)
    result = CliRunner().invoke(
        app,
        [
            "relgnn-scale-report",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "args": (baseline, candidate),
            "output_path": output.resolve(),
        }
    ]


def test_scale_train_uses_one_fail_closed_workflow_command(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cpv26.training import kbo_scale_workflow

    _runtime(tmp_path, monkeypatch)
    baseline = tmp_path / "capacity_comparison_report.json"
    baseline.write_text("{}", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "candidate"
    calls: list[dict[str, Any]] = []

    def fake_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        return {
            "output_directory": str(output),
            "preflight_report": str(tmp_path / "preflight.json"),
            "scale_report": str(output / "scale_comparison_report.json"),
            "comparison": {
                "validation_selection_comparison": {
                    "candidate_minus_baseline": {"full": -0.01},
                    "dependency_gap_change_256x3_minus_128x3": 0.02,
                }
            },
        }

    monkeypatch.setattr(kbo_scale_workflow, "train_kbo_scale_workflow", fake_workflow)
    result = CliRunner().invoke(
        app,
        [
            "relgnn-scale-train",
            "--baseline-report",
            str(baseline),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--max-reserved-fraction",
            "0.8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["args"] == (baseline, dataset, output)
    assert calls[0]["device"] == "cuda:0"
    assert calls[0]["max_reserved_fraction"] == 0.8
    assert callable(calls[0]["progress"])


def test_new_graph_commands_have_help() -> None:
    runner = CliRunner()
    for command in (
        "kbo-graph-audit",
        "relgnn-capacity-compare",
        "relgnn-pair-train",
        "relgnn-scale-preflight",
        "relgnn-scale-report",
        "relgnn-scale-train",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
