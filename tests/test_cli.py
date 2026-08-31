import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch
from rich.text import Text
from typer import rich_utils
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
        "kbo-history-fetch",
        "kbo-history-import",
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
                "--train-start-year", "2001", "--train-end-year", "2024",
                "--validation-year", "2025", "--test-year", "2026", "--chronological",
            ],
            tuple(range(2001, 2025)),
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
    assert config.max_pa_per_day == 0
    assert config.max_edges_per_route_per_day == 0
    assert config.box_pa_weight == 0.2
    assert config.box_pitch_weight == 0.1
    assert config.selection_target == "auto"
    assert config.box_gradient_mode == "auto"
    assert f"{test_season} test was not used" in result.output
    assert "Epochs: 17; best: 11" in result.output
    assert not runtime.exists()


@pytest.mark.parametrize(
    ("option", "value", "field_name"),
    [("--selection-target", "match", "selection_target"),
     ("--box-gradient-mode", "head_only", "box_gradient_mode")],
)
def test_relgnn_train_explicit_policy_options(
    tmp_path: Path, monkeypatch: MonkeyPatch, option: str, value: str, field_name: str,
) -> None:
    from cpv26.training import kbo_runner

    monkeypatch.setenv("CPV26_HOME", str(tmp_path / "runtime"))

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert getattr(kwargs["config"], field_name) == value
        return {"completed_epochs": 1, "best_epoch": 1, "smoke_test_only": False}

    monkeypatch.setattr(kbo_runner, "train_kbo_relgnn", fake_train)
    result = CliRunner().invoke(app, ["relgnn-train", option, value])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("option", ["--selection-target", "--box-gradient-mode"])
def test_relgnn_train_rejects_invalid_policy_before_training(option: str) -> None:
    result = CliRunner().invoke(app, ["relgnn-train", option, "invalid"])
    assert result.exit_code != 0
    assert "must be" in result.output


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


@pytest.mark.parametrize("force_color", [False, True])
def test_relgnn_season_help_describes_checkpoint_splits_and_epoch_date_order(
    monkeypatch: MonkeyPatch,
    force_color: bool,
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", force_color)
    runner = CliRunner()
    train = runner.invoke(app, ["relgnn-train", "--help"], terminal_width=160)
    assert train.exit_code == 0, train.output
    train_text = Text.from_ansi(train.output).plain
    for option in (
        "--train-start-year", "--train-end-year", "--validation-year",
        "--test-year", "--chronological",
    ):
        assert option in train_text
    assert "within each epoch" in train_text
    assert "not streaming" in train_text
    evaluate = runner.invoke(app, ["relgnn-evaluate", "--help"], terminal_width=160)
    assert evaluate.exit_code == 0, evaluate.output
    evaluate_text = Text.from_ansi(evaluate.output).plain
    assert "years come from the checkpoint" in evaluate_text
    assert "test (2025)" not in evaluate_text


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


@pytest.mark.parametrize("years", [(2001, 2022), (2001, 2001), (2021, 2022)])
def test_kbo_history_fetch_uses_inclusive_year_range_and_destination(
    tmp_path: Path, monkeypatch: MonkeyPatch, years: tuple[int, int]
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    calls: list[tuple[Path, tuple[int, ...]]] = []

    def fake_download(destination: Path, *, years: Any) -> tuple[Path, ...]:
        calls.append((destination, tuple(years)))
        return (destination / "verified_archive.json",)

    monkeypatch.setattr(cli_module, "download_kbo_history", fake_download)
    options = []
    destination = runtime / "datasets" / "kbo_history"
    if years != (2001, 2022):
        destination = tmp_path / "chosen-archive"
        options = [
            "--start-year", str(years[0]), "--end-year", str(years[1]),
            "--destination", str(destination),
        ]
    result = CliRunner().invoke(app, ["kbo-history-fetch", *options])
    assert result.exit_code == 0, result.output
    assert calls == [(destination.resolve(), tuple(range(years[0], years[1] + 1)))]
    assert "KBO history ready" in result.output
    assert "verified_archive.json" in result.output
    assert not runtime.exists()


@pytest.mark.parametrize("command", ["kbo-history-fetch", "kbo-history-import"])
@pytest.mark.parametrize(
    "options",
    [
        ["--start-year", "2000"],
        ["--end-year", "2023"],
        ["--start-year", "2022", "--end-year", "2001"],
    ],
)
def test_kbo_history_commands_reject_unavailable_or_reversed_years_before_writes(
    tmp_path: Path, monkeypatch: MonkeyPatch, command: str, options: list[str]
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(runtime / "cpv26.duckdb"))
    result = CliRunner().invoke(app, [command, *options])
    assert result.exit_code == 2
    assert not runtime.exists()


def test_kbo_history_import_missing_archive_does_not_create_database(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(runtime / "cpv26.duckdb"))
    result = CliRunner().invoke(app, ["kbo-history-import"])
    assert result.exit_code == 1
    assert "KBO history source file not found" in result.output
    assert "cpv26 kbo-history-fetch" in result.output
    assert not runtime.exists()


@pytest.mark.parametrize("default_paths", [False, True])
def test_kbo_history_import_forwards_years_checks_references_and_writes_coverage_report(
    tmp_path: Path, monkeypatch: MonkeyPatch, default_paths: bool
) -> None:
    runtime = tmp_path / "runtime"
    database = runtime / "cpv26.duckdb"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))
    directory = runtime / "datasets" / "kbo_history" if default_paths else tmp_path / "archives"
    output = (
        runtime / "reports" / "kbo_history_import.json"
        if default_paths else tmp_path / "report.json"
    )
    years = tuple(range(2001, 2023)) if default_paths else (2001, 2002)
    directory.mkdir(parents=True)
    for artifact in cli_module.select_history_artifacts(cli_module.KBO_HISTORY_FILES, years):
        (directory / artifact.filename).write_text("{}", encoding="utf-8")
    calls: list[tuple[Path, tuple[int, ...]]] = []
    checks: list[str] = []
    report = {
        "games": 532,
        "season_coverage": [
            {"year": 2001, "games": 532, "date_start": "2001-04-05", "date_end": "2001-10-04",
             "batter_rows": 13000, "pitcher_rows": 4000, "hit_labels": 12990,
             "verified_batting_outcomes": 41000}
        ],
        "inserted_rows": {"game": 532},
        "total_rows": {"game": 532},
    }

    def fake_import(store: Any, source: Path, *, years: Any, progress: Any) -> dict[str, Any]:
        assert callable(progress)
        assert store.connection.execute("SELECT count(*) FROM game").fetchone() == (0,)
        calls.append((source, tuple(years)))
        return report

    monkeypatch.setattr(cli_module, "import_kbo_history", fake_import)
    for name in ("assert_referential_integrity", "assert_composite_referential_integrity"):
        original = getattr(cli_module.DuckDBStore, name)

        def checked(self: Any, _name: str = name, _original: Any = original) -> None:
            checks.append(_name)
            _original(self)

        monkeypatch.setattr(cli_module.DuckDBStore, name, checked)
    options = [] if default_paths else [
        "--start-year", "2001", "--end-year", "2002",
        "--source-dir", str(directory), "--report", str(output),
    ]
    result = CliRunner().invoke(app, ["kbo-history-import", *options])
    assert result.exit_code == 0, result.output
    assert calls == [(directory.resolve(), years)]
    assert checks == ["assert_referential_integrity", "assert_composite_referential_integrity"]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "Historical games: 532" in result.output
    assert "partial player records retained" in result.output
    assert not list(output.parent.glob("*.part"))


@pytest.mark.parametrize("target", ["database", "source"])
def test_kbo_history_import_report_cannot_overwrite_database_or_source(
    tmp_path: Path, monkeypatch: MonkeyPatch, target: str
) -> None:
    runtime = tmp_path / "runtime"
    database = runtime / "cpv26.duckdb"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))
    output = (
        database if target == "database" else runtime / "datasets" / "kbo_history" / "SOURCE.json"
    )
    result = CliRunner().invoke(app, ["kbo-history-import", "--report", str(output)])
    assert result.exit_code == 2
    assert "must not overwrite" in result.output
    assert not runtime.exists()


def test_kbo_history_fetch_failure_is_reported_without_traceback(monkeypatch: MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("archive SHA-256 mismatch")

    monkeypatch.setattr(cli_module, "download_kbo_history", fail)
    result = CliRunner().invoke(app, ["kbo-history-fetch"])
    assert result.exit_code == 1
    assert "KBO history download failed" in result.output
    assert "archive SHA-256 mismatch" in result.output


def test_kbo_history_import_failure_preserves_previous_report(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(runtime / "cpv26.duckdb"))
    directory = runtime / "datasets" / "kbo_history"
    directory.mkdir(parents=True)
    for artifact in cli_module.select_history_artifacts(cli_module.KBO_HISTORY_FILES, (2001,)):
        (directory / artifact.filename).write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text('{"previous": true}\n', encoding="utf-8")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("existing canonical score conflict")

    monkeypatch.setattr(cli_module, "import_kbo_history", fail)
    result = CliRunner().invoke(
        app, ["kbo-history-import", "--end-year", "2001", "--report", str(report)]
    )
    assert result.exit_code == 1
    assert "KBO history import failed" in result.output
    assert "existing canonical score conflict" in result.output
    assert json.loads(report.read_text(encoding="utf-8")) == {"previous": True}
    assert not list(tmp_path.rglob("*.part"))


def test_json_report_failed_replace_preserves_previous_report_and_removes_partial(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("report replacement unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement unavailable"):
        cli_module._write_json_report(output, {"updated": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [output]


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
