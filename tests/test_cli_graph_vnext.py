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


def test_new_graph_commands_have_help() -> None:
    runner = CliRunner()
    for command in (
        "kbo-graph-audit",
        "relgnn-capacity-compare",
        "relgnn-pair-train",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
