from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pytest

import cpv26.training.kbo_runner as runner_module
from cpv26.data.kbo_graph_dataset import build_kbo_graph_dataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.training.kbo_runner import (
    KBOTrainingConfig,
    _float64_probabilities,
    _loader,
    _resume_compatible,
    _split_days,
    _split_summary,
    check_gpu,
    evaluate_kbo_relgnn,
    train_kbo_relgnn,
)


def test_probability_roundoff_is_repaired_but_invalid_probabilities_rejected() -> None:
    rounded = np.asarray([[0.2, 0.3, 0.5000002]], dtype=np.float32)
    normalized = _float64_probabilities(rounded)
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized.sum(axis=1), [1.0], rtol=0, atol=1e-15)
    for malformed in ([[0.2, 0.3, 0.6]], [[-0.1, 0.1, 1.0]], [[float("nan"), 0.5, 0.5]]):
        with pytest.raises(FloatingPointError):
            _float64_probabilities(np.asarray(malformed))


@pytest.fixture
def graph_directory(tmp_path: Path) -> Path:
    """Small real-schema rows, generated only in pytest's temporary directory."""
    database = tmp_path / "canonical.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE source_revision (
                source_revision_id VARCHAR, source_name VARCHAR, source_locator VARCHAR,
                content_sha256 VARCHAR, metadata_json VARCHAR, event_at TIMESTAMPTZ,
                available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ,
                valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
            );
            CREATE TABLE game (
                game_id VARCHAR, scheduled_start TIMESTAMPTZ, home_team_id VARCHAR,
                away_team_id VARCHAR, game_status VARCHAR, home_score INTEGER, away_score INTEGER,
                source_revision_id VARCHAR, event_at TIMESTAMPTZ, available_at TIMESTAMPTZ,
                ingested_at TIMESTAMPTZ, valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
            );
            CREATE TABLE observed_plate_appearance (
                observed_pa_row_id VARCHAR, plate_appearance_id VARCHAR, game_id VARCHAR,
                inning INTEGER, half_inning VARCHAR, batter_id VARCHAR, pitcher_id VARCHAR,
                batting_team_id VARCHAR, fielding_team_id VARCHAR, home_score_before INTEGER,
                away_score_before INTEGER, outs_before INTEGER, runners_before VARCHAR,
                outcome VARCHAR, is_at_bat BOOLEAN, is_hit BOOLEAN, total_bases INTEGER,
                source_revision_id VARCHAR, event_at TIMESTAMPTZ, available_at TIMESTAMPTZ,
                ingested_at TIMESTAMPTZ, valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
            )
        """)
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        connection.execute(
            "INSERT INTO source_revision VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["fixture", "pytest", "local", "a" * 64, "{}", start, start, start, start, None],
        )
        for year in (2023, 2024, 2025, 2026):
            for day in (1, 2):
                start = datetime(year, 4, day, tzinfo=timezone(timedelta(hours=9)))
                event = start + timedelta(hours=23, minutes=59, seconds=59)
                available = start + timedelta(days=1)
                game_id = f"{year}-{day}"
                connection.execute(
                    "INSERT INTO game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        game_id,
                        start,
                        "H",
                        "A",
                        "final",
                        day,
                        1,
                        "fixture",
                        event,
                        available,
                        available,
                        event,
                        None,
                    ],
                )
                for number, (batter, pitcher, offense, defense, outcome) in enumerate(
                    (
                        ("b1", "p1", "H", "A", "single"),
                        ("b2", "p2", "A", "H", "strikeout"),
                    )
                ):
                    identity = f"{game_id}-pa{number}"
                    connection.execute(
                        "INSERT INTO observed_plate_appearance VALUES ("
                        + ",".join(["?"] * 23)
                        + ")",
                        [
                            identity,
                            identity,
                            game_id,
                            1,
                            "top" if offense == "A" else "bottom",
                            batter,
                            pitcher,
                            offense,
                            defense,
                            0,
                            0,
                            0,
                            "000",
                            outcome,
                            True,
                            outcome == "single",
                            int(outcome == "single"),
                            "fixture",
                            event,
                            available,
                            available,
                            event,
                            None,
                        ],
                    )
    output = tmp_path / "graphs"
    build_kbo_graph_dataset(database, output)
    return output


def _config(epochs: int = 1, **overrides: Any) -> KBOTrainingConfig:
    return KBOTrainingConfig(
        device="cpu",
        amp="off",
        workers=0,
        epochs=epochs,
        batch_days=1,
        hidden_dim=8,
        layers=1,
        heads=2,
        dropout=0.0,
        **overrides,
    )


def test_cuda_is_required_unless_cpu_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        check_gpu("cuda:0")
    assert check_gpu("cpu", amp="off")["forward_backward_verified"]
    with pytest.raises(ValueError, match="amp must"):
        check_gpu("cpu", amp="invalid")
    with pytest.raises(ValueError, match="CPU validation"):
        check_gpu("cpu", amp="fp16")


def test_real_graph_train_resume_and_test_artifacts(graph_directory: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    run = tmp_path / "training"
    first = train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    assert first["completed_epochs"] == 1
    assert not first["test_used_during_training"]
    assert first["best_checkpoint_sha256"] == sha256_file(run / "best.pt")
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert state["optimizer"]["state"]
    assert state["model"]
    assert state["history"][0]["training_samples"]["match"] == 2
    assert state["history"][0]["training_samples"]["live_hit"] == 4
    resumed = train_kbo_relgnn(
        graph_directory,
        run,
        config=_config(epochs=2),
        resume=run / "last.pt",
        progress=lambda _: None,
    )
    assert resumed["completed_epochs"] == 2
    assert len((run / "history.jsonl").read_text().splitlines()) == 2
    report = evaluate_kbo_relgnn(
        run / "best.pt",
        split="test",
        device="cpu",
        amp="off",
        workers=0,
    )
    assert report["date_start"] == "2025-04-01"
    assert report["metrics"]["match"]["samples"] == 2
    assert report["metrics"]["live_hit"]["samples"] == 4
    assert report["metrics"]["pa"]["samples"] == 4
    for artifact in report["prediction_artifacts"].values():
        path = Path(artifact["path"])
        assert path.is_file() and artifact["sha256"] == sha256_file(path)
        with duckdb.connect() as connection:
            assert (
                connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[
                    0
                ]
                == artifact["rows"]
            )
    repeated = evaluate_kbo_relgnn(
        run / "best.pt",
        split="test",
        device="cpu",
        amp="off",
        workers=0,
    )
    assert report["metrics"] == repeated["metrics"]
    assert report["output_directory"] != repeated["output_directory"]


@pytest.mark.parametrize("chronological", [False, True])
def test_epoch_resume_matches_uninterrupted_training(
    graph_directory: Path, tmp_path: Path, chronological: bool
) -> None:
    torch = pytest.importorskip("torch")
    first = tmp_path / "resumed"
    full = tmp_path / "uninterrupted"
    train_kbo_relgnn(
        graph_directory, first, config=_config(chronological=chronological), progress=lambda _: None
    )
    train_kbo_relgnn(
        graph_directory,
        first,
        config=_config(epochs=2, chronological=chronological),
        resume=first / "last.pt",
        progress=lambda _: None,
    )
    train_kbo_relgnn(
        graph_directory,
        full,
        config=_config(epochs=2, chronological=chronological),
        progress=lambda _: None,
    )
    resumed = torch.load(first / "last.pt", map_location="cpu", weights_only=True)
    uninterrupted = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    for key, tensor in resumed["model"].items():
        torch.testing.assert_close(tensor, uninterrupted["model"][key], rtol=0, atol=0)


def test_resume_refuses_wrong_lineage_and_overwrites(graph_directory: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run = tmp_path / "run"
    train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    with pytest.raises(FileExistsError, match="not empty"):
        train_kbo_relgnn(graph_directory, run, config=_config())
    with pytest.raises(ValueError, match="last.pt"):
        train_kbo_relgnn(graph_directory, run, config=_config(2), resume=run / "best.pt")
    with pytest.raises(ValueError, match="learning_rate"):
        train_kbo_relgnn(
            graph_directory,
            run,
            config=replace(_config(2), learning_rate=0.1),
            resume=run / "last.pt",
        )
    manifest = graph_directory / "manifest.json"
    changed = json.loads(manifest.read_text())
    changed["fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_kbo_relgnn(run / "best.pt", device="cpu", amp="off", workers=0)


def test_worker_process_graph_loading(graph_directory: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    report = train_kbo_relgnn(
        graph_directory,
        tmp_path / "worker-training",
        config=replace(_config(), workers=1),
        progress=lambda _: None,
    )
    assert report["completed_epochs"] == 1


def test_training_config_disallows_invalid_splits_and_empty_tasks() -> None:
    with pytest.raises(ValueError, match="held-out"):
        KBOTrainingConfig(train_seasons=(2025,))
    with pytest.raises(ValueError, match="both match"):
        KBOTrainingConfig(live_hit_weight=0)
    with pytest.raises(ValueError, match="divisible"):
        KBOTrainingConfig(hidden_dim=63, heads=4)


@pytest.mark.parametrize("years", [(2023, 2023), (2024, 2023), (False,), (0,), (10000,)])
def test_training_config_rejects_invalid_years(years: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="season"):
        KBOTrainingConfig(train_seasons=years, validation_season=2025, test_season=2026)


def test_training_config_accepts_historical_years_and_reads_legacy_checkpoints() -> None:
    expanded = KBOTrainingConfig(
        train_seasons=tuple(range(2000, 2024)), chronological=True
    )
    assert expanded.train_seasons[0] == 2000
    assert KBOTrainingConfig.from_dict(asdict(expanded)) == expanded
    legacy = asdict(KBOTrainingConfig())
    del legacy["chronological"]
    del legacy["box_pa_weight"]
    del legacy["box_pitch_weight"]
    assert KBOTrainingConfig.from_dict(legacy).chronological is False
    assert KBOTrainingConfig.from_dict(legacy).box_pa_weight == 0.2
    assert KBOTrainingConfig.from_dict(legacy).box_pitch_weight == 0.1
    _resume_compatible({"training_config": legacy, "epoch": 1}, KBOTrainingConfig(epochs=2))
    with pytest.raises(ValueError, match="chronological"):
        _resume_compatible(
            {"training_config": legacy, "epoch": 1},
            KBOTrainingConfig(epochs=2, chronological=True),
        )


def test_training_config_supports_all_records_without_silent_sampling() -> None:
    config = KBOTrainingConfig(max_pa_per_day=0, max_edges_per_route_per_day=0)
    assert config.max_pa_per_day == config.max_edges_per_route_per_day == 0
    for key in ("max_pa_per_day", "max_edges_per_route_per_day"):
        for value in (-1, True, 1.5):
            with pytest.raises(ValueError, match="non-negative"):
                KBOTrainingConfig(**{key: value})
    for key in ("box_pa_weight", "box_pitch_weight"):
        with pytest.raises(ValueError, match="non-negative"):
            KBOTrainingConfig(**{key: -1})


def test_model_config_enables_boxscore_features_only_for_v3_graphs(graph_directory: Path) -> None:
    dataset = runner_module.KBOGraphDataset(graph_directory)
    current = runner_module._model_config(dataset, KBOTrainingConfig())
    assert current.include_boxscore_heads is True
    assert current.box_batting_feature_dim == 19
    assert current.box_pitching_feature_dim == 21
    legacy = SimpleNamespace(manifest={**dataset.manifest, "dataset_version": 2})
    old = runner_module._model_config(legacy, KBOTrainingConfig())
    assert old.include_boxscore_heads is False
    assert old.node_feature_dims == current.node_feature_dims
    assert old.role_feature_dims == current.role_feature_dims


def test_nullable_prediction_export_does_not_replace_unknown_labels_with_zero(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial.parquet"
    runner_module._write_prediction_parquet(output, [
        {"query_id": "first", "observed_pa": None, "observed_pa_lower_bound": 3,
         "observed_pitches_thrown": None, "expected_hits_lower_bound": 1.2},
        {"query_id": "second", "observed_pa": 4, "observed_pa_lower_bound": 4,
         "observed_pitches_thrown": None, "expected_hits_lower_bound": 1.5},
    ])
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT observed_pa, observed_pitches_thrown FROM read_parquet(?) ORDER BY query_id",
            [str(output)],
        ).fetchall()
        types = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(output)]
        ).fetchall()
    assert rows == [(None, None), (4, None)]
    assert {row[0]: row[1] for row in types}["observed_pa"] == "BIGINT"
    assert {row[0]: row[1] for row in types}["observed_pitches_thrown"] == "BIGINT"


def test_v2_checkpoint_without_boxscore_fields_still_resumes_and_evaluates(
    graph_directory: Path, tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = graph_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run = tmp_path / "legacy-run"
    train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    for key in ("include_boxscore_heads", "box_batting_feature_dim", "box_pitching_feature_dim"):
        state["model_config"].pop(key)
    for key in ("box_pa_weight", "box_pitch_weight"):
        state["training_config"].pop(key)
    torch.save(state, run / "last.pt")
    resumed = train_kbo_relgnn(
        graph_directory, run, config=_config(epochs=2), resume=run / "last.pt",
        progress=lambda _: None,
    )
    assert resumed["completed_epochs"] == 2
    assert set(resumed["history"][-1]["training_samples"]) == {"match", "live_hit", "pa", "run"}
    report = evaluate_kbo_relgnn(run / "best.pt", device="cpu", amp="off", workers=0)
    assert set(report["metrics"]["losses"]) == {"match", "live_hit", "pa", "run"}


def test_multi_year_splits_are_complete_ordered_and_disjoint(graph_directory: Path) -> None:
    dataset = runner_module.KBOGraphDataset(graph_directory)
    config = KBOTrainingConfig(
        train_seasons=(2023, 2024), validation_season=2025, test_season=2026, chronological=True
    )
    splits = _split_days(dataset, config)
    assert splits["train"] == (
        date(2023, 4, 1), date(2023, 4, 2), date(2024, 4, 1), date(2024, 4, 2)
    )
    assert {day.year for day in splits["validation"]} == {2025}
    assert {day.year for day in splits["test"]} == {2026}
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    summary = _split_summary(dataset, splits)
    assert summary["train"]["games"] == 4
    assert summary["train"]["live_hit_queries"] == 8
    assert summary["train"]["pa_queries"] == 8
    assert summary["train"]["date_start"] == "2023-04-01"
    assert summary["train"]["date_end"] == "2024-04-02"
    assert summary["test"]["days"] == 2


def test_requested_training_or_validation_year_cannot_be_silently_skipped(
    graph_directory: Path,
) -> None:
    dataset = runner_module.KBOGraphDataset(graph_directory)
    with pytest.raises(ValueError, match="training seasons: 2022"):
        _split_days(dataset, KBOTrainingConfig(train_seasons=(2022, 2023)))
    with pytest.raises(ValueError, match="validation seasons: 2027"):
        _split_days(dataset, KBOTrainingConfig(validation_season=2027, test_season=2028))
    # A future test season need not be available to train; evaluation checks it separately.
    splits = _split_days(dataset, KBOTrainingConfig(test_season=2027))
    assert splits["test"] == ()


@pytest.mark.parametrize(
    ("chronological", "training", "expected_shuffle"),
    [(True, True, False), (False, True, True), (False, False, False)],
)
def test_loader_orders_training_dates_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chronological: bool,
    training: bool,
    expected_shuffle: bool,
) -> None:
    class Generator:
        def manual_seed(self, value: int) -> Generator:
            self.seed = value
            return self

    def capture_loader(dataset: Any, **options: Any) -> Any:
        return {"dataset": dataset, **options}

    fake_torch = SimpleNamespace(
        Generator=Generator,
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=capture_loader)),
    )
    monkeypatch.setattr(runner_module, "require_torch", lambda: (fake_torch, None))
    days = (date(2024, 4, 2), date(2023, 4, 1), date(2024, 4, 1))
    config = _config(chronological=chronological)
    loader = _loader(tmp_path, days, config, epoch=3, training=training)
    assert loader["shuffle"] is expected_shuffle
    assert loader["generator"].seed == config.seed + 3
    expected_days = days if expected_shuffle else tuple(sorted(days))
    assert loader["dataset"].selected_days == expected_days


def test_multi_year_chronological_training_and_2026_evaluation(
    graph_directory: Path, tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    run = tmp_path / "multi-year"
    config = _config(
        train_seasons=(2023, 2024), validation_season=2025, test_season=2026,
        chronological=True,
    )
    progress: list[str] = []
    report = train_kbo_relgnn(graph_directory, run, config=config, progress=progress.append)
    assert report["training_order"] == "chronological"
    assert report["training_seasons"] == [2023, 2024]
    assert report["validation_season"] == 2025
    assert report["held_out_test_season"] == 2026
    assert report["split_summary"]["train"]["games"] == 4
    assert report["history"][0]["training_samples"]["match"] == 4
    assert report["history"][0]["validation"]["match"]["samples"] == 2
    assert any("2026 test is not used" in line for line in progress)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert state["training_config"]["chronological"] is True
    evaluation = evaluate_kbo_relgnn(
        run / "best.pt", split="test", device="cpu", amp="off", workers=0
    )
    assert evaluation["date_start"] == "2026-04-01"
    assert evaluation["date_end"] == "2026-04-02"
    assert evaluation["held_out_test_season"] == 2026
    assert evaluation["metrics"]["match"]["samples"] == 2
