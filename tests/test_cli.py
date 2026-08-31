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
        "gpu-check",
        "kbo-graph-build",
        "relgnn-train",
        "relgnn-evaluate",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("options", "train_seasons", "validation_season", "test_season", "chronological"),
    [
        ([], (2023,), 2024, 2025, False),
        (
            [
                "--train-start-year", "2000", "--train-end-year", "2024",
                "--validation-year", "2025", "--test-year", "2026", "--chronological",
            ],
            tuple(range(2000, 2025)),
            2025,
            2026,
            True,
        ),
    ],
)
def test_relgnn_train_forwards_seasons_and_date_order_without_gpu(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    options: list[str],
    train_seasons: tuple[int, ...],
    validation_season: int,
    test_season: int,
    chronological: bool,
) -> None:
    from cpv26.training import kbo_runner

    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    calls: list[tuple[Path, Path, kbo_runner.KBOTrainingConfig]] = []

    def fake_train(
        directory: Path, output: Path, *, config: kbo_runner.KBOTrainingConfig, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((directory, output, config))
        assert kwargs["resume"] is None
        return {"completed_epochs": 17, "best_epoch": 11, "smoke_test_only": False}

    monkeypatch.setattr(kbo_runner, "train_kbo_relgnn", fake_train)
    result = CliRunner().invoke(app, ["relgnn-train", *options])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    directory, output, config = calls[0]
    assert directory == (runtime / "datasets" / "kbo_graph").resolve()
    assert output.parent == (runtime / "runs" / "relgnn").resolve()
    assert config.train_seasons == train_seasons
    assert config.validation_season == validation_season
    assert config.test_season == test_season
    assert config.chronological is chronological
    assert config.device == "cuda:0"
    assert config.epochs == 30
    assert config.batch_days == 2
    assert f"{test_season} test was not used" in result.output
    assert "Epochs: 17; best: 11" in result.output
    assert not runtime.exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ["--train-start-year", "2024", "--train-end-year", "2023"],
            "must not exceed",
        ),
        (["--train-end-year", "2024"], "precede validation"),
        (["--validation-year", "2023"], "precede validation"),
        (["--test-year", "2024"], "precede validation"),
    ],
)
def test_relgnn_train_rejects_invalid_splits_before_training(
    tmp_path: Path, monkeypatch: MonkeyPatch, options: list[str], message: str
) -> None:
    from cpv26.training import kbo_runner

    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    calls: list[bool] = []

    def unexpected_train(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        pytest.fail("Invalid season splits must not reach model training")

    monkeypatch.setattr(kbo_runner, "train_kbo_relgnn", unexpected_train)
    result = CliRunner().invoke(app, ["relgnn-train", *options])
    assert result.exit_code != 0
    assert message in result.output
    assert not calls
    assert not runtime.exists()


@pytest.mark.parametrize(
    "option", ["--train-start-year", "--train-end-year", "--validation-year", "--test-year"]
)
@pytest.mark.parametrize("year", ["0", "10000"])
def test_relgnn_train_rejects_years_outside_date_range(option: str, year: str) -> None:
    result = CliRunner().invoke(app, ["relgnn-train", option, year])
    assert result.exit_code == 2
    assert "1<=x<=9999" in result.output


def test_relgnn_season_help_describes_checkpoint_splits_and_epoch_date_order(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    train = runner.invoke(app, ["relgnn-train", "--help"], terminal_width=160)
    assert train.exit_code == 0, train.output
    for option in (
        "--train-start-year", "--train-end-year", "--validation-year",
        "--test-year", "--chronological",
    ):
        assert option in train.output
    assert "within each epoch" in train.output
    assert "not streaming" in train.output
    evaluate = runner.invoke(app, ["relgnn-evaluate", "--help"], terminal_width=160)
    assert evaluate.exit_code == 0, evaluate.output
    assert "years come from the checkpoint" in evaluate.output
    assert "test (2025)" not in evaluate.output


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
    second = runner.invoke(app, [command, "--iterations", "3", "--report", str(second_report_path)])
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
