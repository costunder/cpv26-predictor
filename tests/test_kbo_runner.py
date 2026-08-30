from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from cpv26.data.kbo_graph_dataset import build_kbo_graph_dataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.training.kbo_runner import (
    KBOTrainingConfig,
    _float64_probabilities,
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
        for year in (2023, 2024, 2025):
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


def test_epoch_resume_matches_uninterrupted_training(graph_directory: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    first = tmp_path / "resumed"
    full = tmp_path / "uninterrupted"
    train_kbo_relgnn(graph_directory, first, config=_config(), progress=lambda _: None)
    train_kbo_relgnn(
        graph_directory,
        first,
        config=_config(epochs=2),
        resume=first / "last.pt",
        progress=lambda _: None,
    )
    train_kbo_relgnn(graph_directory, full, config=_config(epochs=2), progress=lambda _: None)
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
