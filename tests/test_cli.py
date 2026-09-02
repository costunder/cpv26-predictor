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
        "relgnn-ablation-train",
        "relgnn-ablation-report",
        "relgnn-evaluate",
        "relgnn-graph-diagnose",
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
    assert config.seed == 2026
    assert config.route_message_normalization == "none"
    assert config.route_schedule == "full"
    assert config.graph_control == "intact"
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

    diagnose = runner.invoke(app, ["relgnn-graph-diagnose", "--help"], terminal_width=160)
    assert diagnose.exit_code == 0, diagnose.output
    diagnose_text = Text.from_ansi(diagnose.output).plain
    for option in (
        "--checkpoint", "--dataset", "--split", "--device", "--amp", "--batch-days",
        "--workers", "--seed", "--max-days", "--output",
    ):
        assert option in diagnose_text
    assert "validation (default)" in diagnose_text
    assert "fixed checkpoint" in diagnose_text

    ablation = runner.invoke(app, ["relgnn-ablation-train", "--help"], terminal_width=200)
    assert ablation.exit_code == 0, ablation.output
    ablation_text = Text.from_ansi(ablation.output).plain
    for option in (
        "--suite-dir",
        "--seed",
        "--graph-control-seed",
        "--validation-year",
        "--test-year",
    ):
        assert option in ablation_text
    assert "validation only" in ablation_text
    assert "never loads or evaluates" in ablation_text

    ablation_report = runner.invoke(
        app, ["relgnn-ablation-report", "--help"], terminal_width=200
    )
    assert ablation_report.exit_code == 0, ablation_report.output
    report_text = Text.from_ansi(ablation_report.output).plain
    assert "--suite-dir" in report_text
    assert "saved suite/training JSON only" in report_text


def test_relgnn_ablation_report_decomposes_saved_validation_without_training(
    tmp_path: Path,
) -> None:
    from cpv26.training import kbo_matched_ablation

    suite = tmp_path / "suite"
    runs: dict[str, dict[str, Any]] = {"2026": {}}
    for index, variant in enumerate(kbo_matched_ablation.MATCHED_GRAPH_VARIANTS):
        offset = index / 100
        policy = kbo_matched_ablation._VARIANT_POLICIES[variant]
        run = suite / "seed-2026" / variant
        run.mkdir(parents=True)
        best_epoch = 10 + index
        history = []
        for epoch in range(1, 31):
            epoch_loss = (
                4.0 + offset
                if epoch == best_epoch
                else 4.5 + offset + abs(epoch - best_epoch) / 1000
            )
            epoch_contributions = {
                task: epoch_loss / 6
                for task in ("match", "live_hit", "pa", "run", "box_pa", "box_pitch")
            }
            history.append(
                {
                    "epoch": epoch,
                    "validation": {
                        "selection_loss": epoch_loss,
                        "losses": epoch_contributions,
                        "weighted_loss_contributions": epoch_contributions,
                        "weighted_multitask_loss": epoch_loss,
                        "selection_target": "weighted",
                    },
                }
            )
        (run / "training_report.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "completed_epochs": 30,
                    "best_epoch": best_epoch,
                    "best_validation_loss": 4.0 + offset,
                    "test_used_during_training": False,
                    "configuration": {
                        "route_message_normalization": policy[
                            "route_message_normalization"
                        ],
                        "route_schedule": policy["route_schedule"],
                        "graph_control": policy["graph_control"],
                    },
                    "graph_control": {"mode": policy["graph_control"]},
                    "history": history,
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "selection_loss": 4.0 + offset,
            "losses": {
                "match": 0.8 + offset,
                "live_hit": 2.4 + offset,
                "pa": 1.5 + offset,
                "run": 5.0 + offset,
                "box_pa": 1.4 + offset,
                "box_pitch": 3.0 + offset,
            },
            "weighted_loss_contributions": {
                "match": 0.8 + offset / 6,
                "live_hit": 2.0 + offset / 6,
                "pa": 0.3 + offset / 6,
                "run": 0.4 + offset / 6,
                "box_pa": 0.2 + offset / 6,
                "box_pitch": 0.3 + offset / 6,
            },
            "loss_sample_counts": {
                "match": 470,
                "live_hit": 10717,
                "pa": 37102,
                "run": 470,
                "box_pa": 10717,
                "box_pitch": 940,
            },
            "weighted_multitask_loss": 4.0 + offset,
            "selection_target": "weighted",
            "match": {
                "log_loss": 0.8 + offset,
                "accuracy": 0.5,
                "expected_calibration_error": 0.05,
                "brier_score": 0.5,
            },
            "live_hit": {
                "log_loss": 0.66 + offset,
                "accuracy": 0.6,
                "expected_calibration_error": 0.02,
                "brier_score": 0.47,
                "joint_nll": 2.4 + offset,
                "observed_nll": 2.4 + offset,
                "expected_hits_lower_bound_mae": 0.7,
                "expected_pa_lower_bound_mae": 0.9,
            },
            "pa": {
                "log_loss": 1.5 + offset,
                "accuracy": 0.44,
                "expected_calibration_error": 0.02,
                "brier_score": 0.72,
            },
        }
        runs["2026"][variant] = {
            "run_directory": str(tmp_path / "old-machine-suite" / variant),
            "best_epoch": best_epoch,
            "completed_epochs": 30,
            "parameter_count": 1234,
            "validation_selection_loss": metrics["selection_loss"],
            "selection_loss_delta_vs_full": offset,
            "validation_metrics": metrics,
            "test_used_during_training": False,
            "route_message_normalization": policy["route_message_normalization"],
            "route_schedule_preset": policy["route_schedule"],
            "graph_control": {"mode": policy["graph_control"]},
            "variant_policy": dict(policy),
        }
    report = {
        "status": "completed",
        "protocol": "matched_from_scratch_validation_graph_ablation",
        "protocol_version": 1,
        "selection_split": "validation",
        "held_out_test_season": 2026,
        "test_used_for_training_selection_or_comparison": False,
        "runs": runs,
    }
    report_path = suite / "matched_retraining_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    before = report_path.read_bytes()

    result = CliRunner().invoke(
        app,
        ["relgnn-ablation-report", "--suite-dir", str(suite)],
        terminal_width=240,
    )

    assert result.exit_code == 0, result.output
    plain = Text.from_ansi(result.output).plain
    assert "Matched validation checkpoint selection" in plain
    assert "Predefined matched contrasts" in plain
    assert "Weighted task contribution deltas" in plain
    assert "Raw task-loss deltas by matched contrast" in plain
    assert "Core-normalized weighted task attribution by checkpoint view" in plain
    assert "last_five:" in plain
    assert "Validation prediction metrics" in plain
    assert "core-normalized" in plain
    assert "+0.020000" in plain
    assert "box_pitch" in plain
    assert "joint_NLL" in plain
    assert "observed_NLL" in plain
    assert "not evidence of stability" in plain
    assert "test was not loaded or evaluated" in plain
    assert report_path.read_bytes() == before

    corrupted = json.loads(before)
    corrupted["runs"]["2026"]["full"]["validation_metrics"][
        "weighted_loss_contributions"
    ]["match"] += 1
    report_path.write_text(json.dumps(corrupted), encoding="utf-8")
    invalid_total = CliRunner().invoke(
        app,
        ["relgnn-ablation-report", "--suite-dir", str(suite)],
    )
    assert invalid_total.exit_code == 1
    invalid_text = Text.from_ansi(invalid_total.output).plain
    assert "weighted task contributions" in invalid_text
    assert "total" in invalid_text

    report_path.write_bytes(before)
    corrupted = json.loads(before)
    corrupted["runs"]["2026"]["core"]["route_schedule_preset"] = "full"
    report_path.write_text(json.dumps(corrupted), encoding="utf-8")
    mislabeled = CliRunner().invoke(
        app,
        ["relgnn-ablation-report", "--suite-dir", str(suite)],
    )
    assert mislabeled.exit_code == 1
    assert "route_schedule is mislabeled" in Text.from_ansi(mislabeled.output).plain


def test_relgnn_graph_diagnose_forwards_options_and_prints_compact_deltas(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    from cpv26.training import kbo_graph_diagnostic

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    dataset = tmp_path / "graphs"
    output = tmp_path / "diagnostic"
    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_diagnose(path: Path, **options: Any) -> dict[str, Any]:
        calls.append((path, options))
        return {
            "output_directory": str(output),
            "conditions": {
                "intact": {
                    "metrics": {"selection_loss": 4.0},
                    "metric_delta_vs_intact": {"selection_loss": 0.0},
                    "prediction_sensitivity_vs_intact": {},
                },
                "no_routes": {
                    "metrics": {"selection_loss": 4.25},
                    "metric_delta_vs_intact": {"selection_loss": 0.25},
                    "prediction_sensitivity_vs_intact": {
                        "match": {"mean_total_variation": 0.01},
                        "live_hit": {"mean_total_variation": 0.02},
                        "pa": {"mean_total_variation": 0.03},
                    },
                },
            },
        }

    monkeypatch.setattr(kbo_graph_diagnostic, "diagnose_kbo_graph_dependence", fake_diagnose)
    result = CliRunner().invoke(
        app,
        [
            "relgnn-graph-diagnose",
            "--checkpoint", str(checkpoint),
            "--dataset", str(dataset),
            "--device", "cpu",
            "--amp", "off",
            "--batch-days", "3",
            "--workers", "0",
            "--seed", "7",
            "--max-days", "5",
            "--output", str(output),
        ],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    path, options = calls[0]
    assert path == checkpoint
    assert options == {
        "dataset_directory": dataset,
        "split": "validation",
        "device": "cpu",
        "amp": "off",
        "batch_days": 3,
        "workers": 0,
        "seed": 7,
        "max_days": 5,
        "output_directory": output,
    }
    plain = Text.from_ansi(result.output).plain
    assert "no_routes" in plain
    assert "+0.250000" in plain
    assert "0.010000 / 0.020000 / 0.030000" in plain
    assert output.name in plain
    assert "report.json" in plain
    assert "matched retraining" in plain
    assert "Limited-date smoke diagnostic" in plain


def test_relgnn_ablation_train_forwards_repeated_seeds_and_prints_protocol(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    from cpv26.training import kbo_matched_ablation

    runtime = tmp_path / "runtime"
    dataset = tmp_path / "graph"
    output = tmp_path / "suite"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    calls: list[dict[str, Any]] = []

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        aggregate = {
            variant: {
                "validation_selection_loss": {
                    "mean": 4.0 + index / 10,
                    "population_std": 0.01,
                    "paired_delta_vs_full_mean": index / 10,
                },
                "parameter_count": 1234,
            }
            for index, variant in enumerate(kbo_matched_ablation.MATCHED_GRAPH_VARIANTS)
        }
        return {"aggregate": aggregate}

    monkeypatch.setattr(kbo_matched_ablation, "train_matched_graph_ablations", fake_train)
    result = CliRunner().invoke(
        app,
        [
            "relgnn-ablation-train",
            "--dataset", str(dataset),
            "--suite-dir", str(output),
            "--train-start-year", "2001",
            "--train-end-year", "2024",
            "--validation-year", "2025",
            "--test-year", "2026",
            "--device", "cpu",
            "--amp", "off",
            "--workers", "0",
            "--epochs", "3",
            "--seed", "11",
            "--seed", "12",
            "--graph-control-seed", "91",
        ],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == (dataset.resolve(), output.resolve())
    assert call["seeds"] == [11, 12]
    config = call["base_config"]
    assert config.seed == 11
    assert config.graph_control_seed == 91
    assert config.patience == 0
    assert config.train_seasons == tuple(range(2001, 2025))
    assert config.validation_season == 2025
    assert config.test_season == 2026
    plain = Text.from_ansi(result.output).plain
    compact = " ".join(plain.split())
    assert "12 runs" in plain
    assert "much longer than one run" in compact
    assert "metadata only" in plain
    assert "validation only" in compact
    assert "matched_retraining_report.json" in plain


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


@pytest.mark.parametrize("target", ["database", "source", "source_directory"])
def test_kbo_import_report_cannot_overwrite_database_or_source(
    tmp_path: Path, monkeypatch: MonkeyPatch, target: str,
) -> None:
    runtime = tmp_path / "runtime"
    sources = runtime / "datasets" / "kbo_playbyplay" / "v0"
    sources.mkdir(parents=True)
    database = runtime / "cpv26.duckdb"
    source = sources / "kbo_pbp_2023.parquet"
    database.write_bytes(b"preserve database")
    source.write_bytes(b"preserve source")
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))
    output = {"database": database, "source": source, "source_directory": sources}[target]
    result = CliRunner().invoke(app, ["kbo-import", "--year", "2023", "--report", str(output)])
    assert result.exit_code != 0
    assert "must not overwrite" in result.output
    assert database.read_bytes() == b"preserve database"
    assert source.read_bytes() == b"preserve source"


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
