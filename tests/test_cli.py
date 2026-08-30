import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch
from typer.testing import CliRunner

import cpv26.cli as cli_module
from cpv26.cli import app
from cpv26.data import SCHEMA_VERSION


def test_database_cli_lifecycle(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    database = runtime / "cpv26.duckdb"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))

    runner = CliRunner()
    initialised = runner.invoke(app, ["db-init"])
    assert initialised.exit_code == 0, initialised.output
    assert database.is_file()
    assert "Database ready" in initialised.output
    assert f"schema={SCHEMA_VERSION}" in initialised.output

    checked = runner.invoke(app, ["db-check"])
    assert checked.exit_code == 0, checked.output
    assert "Database schema and references are current" in checked.output
    assert f"version {SCHEMA_VERSION}" in checked.output

    shown = runner.invoke(app, ["show-config"])
    assert shown.exit_code == 0, shown.output
    assert "database_path" in shown.output
    assert "Asia/Seoul" in shown.output


def test_kbo_commands_are_available_without_loading_catboost() -> None:
    runner = CliRunner()
    for command in (
        "kbo-fetch",
        "kbo-import",
        "kbo-match-evaluate",
        "kbo-live-hit-evaluate",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output


def test_kbo_fetch_uses_runtime_directory_and_requested_year(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    calls: list[tuple[Path, tuple[int, ...]]] = []

    def fake_download(destination: Path, *, years: Any) -> tuple[Path, ...]:
        selected = tuple(years)
        calls.append((destination, selected))
        return tuple(destination / f"kbo_pbp_{year}.parquet" for year in selected)

    monkeypatch.setattr(cli_module, "download_kbo_playbyplay", fake_download)
    result = CliRunner().invoke(app, ["kbo-fetch", "--year", "2026"])
    assert result.exit_code == 0, result.output
    assert calls == [(runtime / "datasets" / "kbo_playbyplay" / "v0", (2026,))]
    assert "kbo_pbp_2026.parquet" in result.output


def test_kbo_import_explains_missing_source(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CPV26_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("CPV26_DB_PATH", str(tmp_path / "runtime" / "cpv26.duckdb"))
    result = CliRunner().invoke(app, ["kbo-import"])
    assert result.exit_code == 1
    assert "KBO source file not found" in result.output
    assert "cpv26 kbo-fetch" in result.output


@pytest.mark.parametrize(
    ("command", "evaluator", "task"),
    [
        ("kbo-match-evaluate", "evaluate_fixed_season_catboost_json", "kbo_match_baseline"),
        (
            "kbo-live-hit-evaluate",
            "evaluate_live_hit_fixed_season_catboost_json",
            "kbo_live_hit_baseline",
        ),
    ],
)
def test_evaluation_reruns_preserve_models_and_archived_reports(
    tmp_path: Path, monkeypatch: MonkeyPatch, command: str, evaluator: str, task: str
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(runtime / "cpv26.duckdb"))
    runner = CliRunner()
    assert runner.invoke(app, ["db-init"]).exit_code == 0

    def fake_evaluate(rows: Any, **options: Any) -> str:
        directory: Path = options["model_output_directory"]
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "test_2025.cbm"
        model.write_bytes(str(options["catboost_parameters"]["iterations"]).encode())
        return json.dumps(
            {
                "folds": [
                    {
                        "name": "test_2025",
                        "model_path": str(model),
                        "train_games": 2,
                        "evaluation_games": 1,
                        "train_player_games": 2,
                        "evaluation_player_games": 1,
                        "metrics": {"log_loss": 0.5, "accuracy": 0.6},
                        "prior_baseline": {"metrics": {"log_loss": 0.7}},
                    }
                ]
            }
        )

    monkeypatch.setattr(cli_module, evaluator, fake_evaluate)
    first = runner.invoke(app, [command, "--iterations", "2"])
    assert first.exit_code == 0, first.output
    first_report = json.loads((runtime / "reports" / f"{task}.json").read_text())
    second_report_path = runtime / "reports" / "comparison.json"
    second = runner.invoke(
        app, [command, "--iterations", "3", "--report", str(second_report_path)]
    )
    assert second.exit_code == 0, second.output
    second_report = json.loads(second_report_path.read_text())
    assert first_report["model_directory"] != second_report["model_directory"]
    for report, expected in ((first_report, b"2"), (second_report, b"3")):
        fold = report["folds"][0]
        assert Path(fold["model_path"]).read_bytes() == expected
        assert fold["model_sha256"] == hashlib.sha256(expected).hexdigest()
        archive = Path(report["model_directory"]) / "evaluation.json"
        assert json.loads(archive.read_text()) == report
    assert not list((runtime / "reports").glob("*.part"))
