from __future__ import annotations

import gc
import json
import weakref
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
from cpv26.training.kbo_graph_diagnostic import diagnose_kbo_graph_dependence
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


def test_route_schedule_presets_resolve_exact_directions() -> None:
    full = (
        "batter_pa_pitcher__forward",
        "batter_pa_pitcher__reverse",
        "batter_participation_team__forward",
        "batter_participation_team__reverse",
        "pitcher_participation_team__forward",
        "pitcher_participation_team__reverse",
        "home_team_game_away_team__forward",
        "home_team_game_away_team__reverse",
    )
    staged_first = (
        "batter_pa_pitcher__forward",
        "batter_pa_pitcher__reverse",
        "batter_participation_team__forward",
        "pitcher_participation_team__reverse",
        "home_team_game_away_team__forward",
        "home_team_game_away_team__reverse",
    )
    core = (
        "batter_pa_pitcher__forward",
        "batter_pa_pitcher__reverse",
        "batter_participation_team__forward",
        "batter_participation_team__reverse",
    )
    assert runner_module._resolved_route_schedule(
        KBOTrainingConfig(layers=2, route_schedule="full")
    ) is None
    assert full == runner_module._FULL_ROUTE_GATE_KEYS
    assert runner_module._resolved_route_schedule(
        KBOTrainingConfig(layers=2, route_schedule="staged")
    ) == (staged_first, core)
    assert runner_module._resolved_route_schedule(
        KBOTrainingConfig(layers=2, route_schedule="core")
    ) == (core, core)
    assert runner_module._resolved_route_schedule(
        KBOTrainingConfig(layers=2, route_schedule="node_only")
    ) == ((), ())


def test_matched_suite_manifest_allows_seed_append_only(tmp_path: Path) -> None:
    from cpv26.training.kbo_matched_ablation import _validate_or_write_manifest

    def manifest(seeds: list[int]) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "seeds": seeds,
            "base_training_config": {"epochs": 3, "batch_days": 8},
        }

    path = tmp_path / "suite_config.json"
    _validate_or_write_manifest(path, manifest([17]))
    _validate_or_write_manifest(path, manifest([17, 29]))
    assert json.loads(path.read_text(encoding="utf-8"))["seeds"] == [17, 29]
    for invalid in ([29], [29, 17], [17]):
        with pytest.raises(ValueError, match="appended"):
            _validate_or_write_manifest(path, manifest(invalid))


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


def test_training_releases_each_device_batch_before_the_next_transfer(
    graph_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference release changes lifetime only, never numerical/report results."""

    torch = pytest.importorskip("torch")
    config = _config()
    reference = train_kbo_relgnn(
        graph_directory,
        tmp_path / "reference-lifetime",
        config=config,
        progress=lambda _: None,
    )
    original_move = runner_module._move
    original_losses = runner_module._losses
    previous_batch_sentinel: weakref.ReferenceType[Any] | None = None
    previous_output: weakref.ReferenceType[Any] | None = None
    transfer_count = 0

    def tracked_move(value: Any, device: Any) -> Any:
        nonlocal previous_batch_sentinel, transfer_count
        gc.collect()
        if previous_batch_sentinel is not None:
            assert previous_batch_sentinel() is None, (
                "the previous device batch still overlaps the next transfer"
            )
        if previous_output is not None:
            assert previous_output() is None, (
                "the previous model output still overlaps the next transfer"
            )
        moved = dict(original_move(value, device))
        sentinel = torch.empty(1, device=device)
        moved["_device_batch_lifetime_sentinel"] = sentinel
        previous_batch_sentinel = weakref.ref(sentinel)
        transfer_count += 1
        return moved

    def tracked_losses(outputs: Any, batch: Any, options: Any) -> Any:
        nonlocal previous_output
        previous_output = weakref.ref(outputs["match_logits"])
        return original_losses(outputs, batch, options)

    monkeypatch.setattr(runner_module, "_move", tracked_move)
    monkeypatch.setattr(runner_module, "_losses", tracked_losses)
    observed = train_kbo_relgnn(
        graph_directory,
        tmp_path / "observed-lifetime",
        config=config,
        progress=lambda _: None,
    )
    gc.collect()

    assert transfer_count == 4  # two train batches followed by two validation batches
    assert previous_batch_sentinel is not None and previous_batch_sentinel() is None
    assert previous_output is not None and previous_output() is None
    assert observed["initial_model_state_sha256"] == reference[
        "initial_model_state_sha256"
    ]
    assert observed["parameter_count"] == reference["parameter_count"]
    assert observed["optimizer_steps"] == reference["optimizer_steps"]
    assert observed["best_validation_loss"] == reference["best_validation_loss"]
    assert observed["history"][0]["train_losses"] == reference["history"][0][
        "train_losses"
    ]
    assert observed["history"][0]["validation"] == reference["history"][0][
        "validation"
    ]


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


@pytest.mark.parametrize("compare_optimizations", [False, True])
def test_isolated_profile_preserves_run_and_data(
    graph_directory: Path, tmp_path: Path, compare_optimizations: bool,
) -> None:
    pytest.importorskip("torch")
    from cpv26.training.kbo_profile import profile_run

    run = tmp_path / "original-run"
    train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    # Resume settings live in the checkpoint, not the old config.json.
    saved = json.loads((run / "config.json").read_text())
    saved["training"]["batch_days"] = 99
    (run / "config.json").write_text(json.dumps(saved))
    protected = [
        path for root in (run, graph_directory) for path in root.rglob("*") if path.is_file()
    ]
    original_hashes = {path: sha256_file(path) for path in protected}
    report_path = profile_run(
        run, output_directory=tmp_path / "diagnostic", device="cpu",
        warmup=1, steps=1, trace_steps=0, workers=0, progress=lambda _: None,
        compare_optimizations=compare_optimizations, repeats=2,
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["diagnostic_only"] is True
    assert report["configuration"]["batch_days"] == 1
    assert report["checkpoint_epoch"] == 1
    assert report["profile_schema_version"] == 2
    assert report["host_runtime"]["torch_num_threads"] > 0
    for window in report["windows"].values():
        assert all(day.startswith("2023-") for day in window["days"])
        if compare_optimizations:
            comparison = window["optimization_comparison"]
            assert comparison["repeats"] == 2
            assert comparison["execution_order"] == [
                ["reference", "optimized"], ["optimized", "reference"],
            ]
            assert comparison["speedup"] > 0
            assert len(comparison["samples"]["reference"]) == 2
            assert len(comparison["samples"]["optimized"]) == 2
        else:
            assert set(window["cases"]) == {"stream", "resident", "resident_no_statistics"}
            for case in window["cases"].values():
                assert case["steps"] == 1 and case["milliseconds_per_batch"] > 0
            assert window["cases"]["resident_no_statistics"]["host_stage_mean_ms"]
    assert original_hashes == {path: sha256_file(path) for path in protected}
    for forbidden in (run, run / "new-output", graph_directory / "new-output", report_path.parent):
        with pytest.raises((ValueError, FileExistsError)):
            profile_run(
                run, output_directory=forbidden, device="cpu",
                warmup=1, steps=1, trace_steps=0, workers=0, progress=lambda _: None,
            )


def test_profile_optimizer_copy_does_not_alias_checkpoint(
    graph_directory: Path, tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from cpv26.training.kbo_profile import _new_session, _step

    run = tmp_path / "optimizer-source"
    config = _config()
    train_kbo_relgnn(graph_directory, run, config=config, progress=lambda _: None)
    state = runner_module._read_checkpoint(run / "last.pt")
    snapshots = {
        (key, name): value.clone()
        for key, values in state["optimizer"]["state"].items()
        for name, value in values.items() if torch.is_tensor(value)
    }
    device = torch.device("cpu")
    session = _new_session(state, config, device)
    batch = next(iter(_loader(
        graph_directory, [date(2023, 4, 2)], config, epoch=0, training=True,
    )))
    _step(session, batch, config, device, index=0, statistics_enabled=True)
    assert snapshots
    for (key, name), original in snapshots.items():
        assert torch.equal(original, state["optimizer"]["state"][key][name])


def test_real_graph_train_resume_and_test_artifacts(graph_directory: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    run = tmp_path / "training"
    first = train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    assert first["completed_epochs"] == 1
    assert not first["test_used_during_training"]
    assert first["best_checkpoint_sha256"] == sha256_file(run / "best.pt")
    assert len(first["initial_model_state_sha256"]) == 64
    assert first["parameter_count"] > 0
    assert first["attempted_optimizer_steps"] == (
        first["optimizer_steps"] + first["skipped_optimizer_steps"]
    )
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert state["initial_model_state_sha256"] == first["initial_model_state_sha256"]
    assert state["parameter_count"] == first["parameter_count"]
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


def test_graph_dependence_diagnostic_runs_all_conditions_on_a_real_checkpoint(
    graph_directory: Path, tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    run = tmp_path / "diagnostic-training"
    train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    checkpoint = run / "best.pt"
    checkpoint_hash = sha256_file(checkpoint)
    standard = evaluate_kbo_relgnn(
        checkpoint,
        split="validation",
        device="cpu",
        amp="off",
        batch_days=2,
        workers=0,
        output_directory=tmp_path / "standard-evaluation",
    )

    output = tmp_path / "graph-diagnostic"
    report = diagnose_kbo_graph_dependence(
        checkpoint,
        dataset_directory=graph_directory,
        split="validation",
        device="cpu",
        amp="off",
        batch_days=2,
        workers=0,
        seed=17,
        output_directory=output,
    )

    assert tuple(report["conditions"]) == (
        "intact",
        "no_routes",
        "permuted_endpoints",
        "permuted_edge_attributes",
        "without_batter_pa_pitcher",
        "without_batter_participation_team",
        "without_pitcher_participation_team",
        "without_home_team_game_away_team",
    )
    intact = report["conditions"]["intact"]
    assert intact["metrics"] == standard["metrics"]
    assert intact["metric_delta_vs_intact"]["selection_loss"] == pytest.approx(0.0)
    assert intact["prediction_sensitivity_vs_intact"]["match"][
        "mean_total_variation"
    ] == pytest.approx(0.0)
    assert intact["internal_diagnostics"]["attention"]["by_layer_route_direction"]
    assert all(
        condition["internal_diagnostics"] is None
        for name, condition in report["conditions"].items()
        if name != "intact"
    )
    assert report["conditions"]["no_routes"]["transform"]["edges_after"] == 0
    assert report["conditions"]["permuted_endpoints"]["transform"][
        "effective_changes"
    ] > 0
    assert report["checkpoint_sha256"] == checkpoint_hash == sha256_file(checkpoint)
    assert report["split"] == "validation"
    assert not report["smoke_test_only"]
    assert report["internal_diagnostics_scope"] == "intact condition only"
    assert "binary hit/no-hit marginal" in report["prediction_sensitivity_definitions"][
        "live_hit"
    ]
    assert any("multiple intervention seeds" in item for item in report["limitations"])
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_sha256"] == checkpoint_hash
    assert saved["dataset_fingerprint"] == report["dataset_fingerprint"]


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


def test_cuda_rng_resume_restores_selected_device_across_visible_count_changes() -> None:
    restored: list[tuple[Any, Any]] = []

    def device(value: str) -> Any:
        kind, separator, raw_index = value.partition(":")
        return SimpleNamespace(type=kind, index=int(raw_index) if separator else None)

    torch = SimpleNamespace(
        device=device,
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            set_rng_state=lambda value, device=None: restored.append((value, device)),
        ),
    )
    current = torch.device("cuda:0")
    selected = object()
    runner_module._restore_cuda_rng_state(
        torch,
        {
            "selected_cuda_device": "cuda:6",
            "selected_cuda_rng_state": selected,
            "cuda_rng_states": [object()] * 8,
            "training_config": asdict(KBOTrainingConfig(device="cuda:6")),
        },
        current,
    )
    assert len(restored) == 1
    assert restored[0][0] is selected and restored[0][1] is current

    restored.clear()
    legacy = [object() for _ in range(8)]
    runner_module._restore_cuda_rng_state(
        torch,
        {
            "cuda_rng_states": legacy,
            "training_config": asdict(KBOTrainingConfig(device="cuda:6")),
        },
        current,
    )
    assert len(restored) == 1
    assert restored[0][0] is legacy[6] and restored[0][1] is current


def test_cuda_rng_resume_rejects_missing_state_but_cpu_resume_remains_compatible() -> None:
    restored: list[Any] = []

    def device(value: str) -> Any:
        kind, separator, raw_index = value.partition(":")
        return SimpleNamespace(type=kind, index=int(raw_index) if separator else None)

    torch = SimpleNamespace(
        device=device,
        cuda=SimpleNamespace(
            set_rng_state=lambda value, device=None: restored.append((value, device))
        ),
    )
    current = torch.device("cuda:0")
    with pytest.raises(ValueError, match="selected-device RNG state"):
        runner_module._restore_cuda_rng_state(
            torch,
            {"selected_cuda_rng_state": None},
            current,
        )
    with pytest.raises(ValueError, match="no restorable CUDA RNG state"):
        runner_module._restore_cuda_rng_state(
            torch,
            {"cuda_rng_states": [], "training_config": asdict(KBOTrainingConfig())},
            current,
        )
    with pytest.raises(ValueError, match="configured device"):
        runner_module._restore_cuda_rng_state(
            torch,
            {
                "cuda_rng_states": [object()],
                "training_config": asdict(KBOTrainingConfig(device="cpu")),
            },
            current,
        )
    runner_module._restore_cuda_rng_state(torch, {}, torch.device("cpu"))
    assert not restored


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


@pytest.mark.filterwarnings("error:.*multi-threaded.*fork.*:DeprecationWarning")
def test_worker_process_graph_loading(
    graph_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_loader = runner_module._loader
    loader_modes: list[bool] = []

    def checked_loader(*args: Any, **kwargs: Any) -> Any:
        loader = original_loader(*args, **kwargs)
        assert loader.num_workers == 2
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        loader_modes.append(kwargs["training"])
        return loader

    monkeypatch.setattr(runner_module, "_loader", checked_loader)
    run = tmp_path / "worker-training"
    report = train_kbo_relgnn(
        graph_directory,
        run,
        config=replace(_config(), workers=2),
        progress=lambda _: None,
    )
    assert report["completed_epochs"] == 1
    assert loader_modes == [True, False]  # Train and validation both spawned.
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert state["training_config"]["workers"] == 2
    assert state["history"][0]["training_samples"]["match"] == 2
    saved_training = json.loads((run / "training_report.json").read_text())
    assert saved_training["completed_epochs"] == 1

    evaluated = evaluate_kbo_relgnn(
        run / "best.pt", split="test", device="cpu", amp="off", workers=2,
    )
    assert loader_modes == [True, False, False]
    assert evaluated["metrics"]["match"]["samples"] == 2
    assert evaluated["metrics"]["live_hit"]["samples"] == 4
    saved_metrics = json.loads(
        (Path(evaluated["output_directory"]) / "metrics.json").read_text()
    )
    assert saved_metrics["metrics"] == evaluated["metrics"]
    for artifact in evaluated["prediction_artifacts"].values():
        path = Path(artifact["path"])
        assert artifact["sha256"] == sha256_file(path)
        with duckdb.connect() as connection:
            row = connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()
        assert row is not None and row[0] == artifact["rows"]


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
    del legacy["selection_target"]
    del legacy["box_gradient_mode"]
    assert KBOTrainingConfig.from_dict(legacy).chronological is False
    assert KBOTrainingConfig.from_dict(legacy).box_pa_weight == 0.2
    assert KBOTrainingConfig.from_dict(legacy).box_pitch_weight == 0.1
    assert runner_module._training_policies(KBOTrainingConfig.from_dict(legacy)) == {
        "selection_target": "weighted", "box_gradient_mode": "shared",
    }
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
    for version in (2, 3, 4):
        selected = SimpleNamespace(manifest={**dataset.manifest, "dataset_version": version})
        config = runner_module._model_config(selected, KBOTrainingConfig())
        assert config.include_boxscore_heads is (version >= 3)
        assert config.box_gradient_mode == "shared"
    isolated = runner_module._model_config(
        dataset, KBOTrainingConfig(box_gradient_mode="head_only")
    )
    assert isolated.box_gradient_mode == "head_only"


def test_selection_and_gradient_policies_are_independent_explicit_controls() -> None:
    assert runner_module._training_policies(KBOTrainingConfig(selection_target="match")) == {
        "selection_target": "match", "box_gradient_mode": "shared",
    }
    assert runner_module._training_policies(KBOTrainingConfig(box_gradient_mode="head_only")) == {
        "selection_target": "weighted", "box_gradient_mode": "head_only",
    }
    for field in ("selection_target", "box_gradient_mode"):
        with pytest.raises(ValueError, match=field):
            KBOTrainingConfig(**{field: "invalid"})
        changed = KBOTrainingConfig(
            **{field: "match" if field == "selection_target" else "head_only"}
        )
        with pytest.raises(ValueError, match=field):
            _resume_compatible(
                {"training_config": asdict(KBOTrainingConfig()), "epoch": 1}, changed
            )


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


@pytest.mark.parametrize("version", [2, 3])
def test_legacy_checkpoint_without_new_policy_fields_still_resumes_and_evaluates(
    graph_directory: Path, tmp_path: Path, version: int,
) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = graph_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_version"] = version
    # Older PBP-only caches had no unified box histories or aggregate queries.
    for entry in manifest["days"]:
        path = graph_directory / entry["file"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                key: archive[key] for key in archive.files
                if not key.startswith(("box_", "player_box_", "team_box_"))
                and key != "live_hit_pa_min"
            }
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        entry["sha256"] = sha256_file(path)
        for key in tuple(entry):
            if key.startswith("box_"):
                del entry[key]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run = tmp_path / "legacy-run"
    train_kbo_relgnn(graph_directory, run, config=_config(), progress=lambda _: None)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    state.pop("selected_cuda_device")
    state.pop("selected_cuda_rng_state")
    state["model_config"].pop("box_gradient_mode")
    if version == 2:
        for key in (
            "include_boxscore_heads", "box_batting_feature_dim", "box_pitching_feature_dim",
        ):
            state["model_config"].pop(key)
    for key in ("box_pa_weight", "box_pitch_weight", "selection_target", "box_gradient_mode"):
        state["training_config"].pop(key)
    torch.save(state, run / "last.pt")
    resumed = train_kbo_relgnn(
        graph_directory, run, config=_config(epochs=2), resume=run / "last.pt",
        progress=lambda _: None,
    )
    assert resumed["completed_epochs"] == 2
    tasks = {"match", "live_hit", "pa", "run"} | (
        {"box_pa", "box_pitch"} if version >= 3 else set()
    )
    assert set(resumed["history"][-1]["training_samples"]) == tasks
    assert resumed["training_policies"]["selection_target"] == "weighted"
    assert resumed["training_policies"]["box_gradient_mode"] == "shared"
    report = evaluate_kbo_relgnn(run / "best.pt", device="cpu", amp="off", workers=0)
    assert set(report["metrics"]["losses"]) == tasks


def test_cross_era_graph_inputs_train_and_evaluate_with_audited_optional_policies(
    graph_directory: Path, tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from test_kbo_history_graph import _historical_database

    database = tmp_path / "cross-era.duckdb"
    _historical_database(database)
    with duckdb.connect(str(database)) as combined, duckdb.connect(
        str(graph_directory.parent / "canonical.duckdb"), read_only=True
    ) as recent:
        for table in ("source_revision", "game", "observed_plate_appearance"):
            rows = recent.execute(f"SELECT * FROM {table}").fetchall()
            combined.executemany(
                f"INSERT INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows,
            )
    directory = tmp_path / "cross-era-graphs"
    dataset = build_kbo_graph_dataset(database, directory)
    assert dataset.manifest["dataset_version"] >= 4
    options = _config(
        epochs=2, train_seasons=(2001, 2023, 2024), validation_season=2025, test_season=2026,
        chronological=True, max_pa_per_day=0, max_edges_per_route_per_day=0,
        selection_target="match", box_gradient_mode="head_only",
    )
    model_config = runner_module._model_config(dataset, options)
    model = runner_module.KBORelGNNModel(model_config)
    for day in ("2001-04-02", "2023-04-02"):
        graph = dataset.load_day(day)
        assert graph.team_box_batting_features.any()
        assert graph.box_pa_counts.size or graph.box_pitch_mask.any()
        batch = runner_module.collate_kbo_day_graphs([graph])
        batch["team_box_batting_features"].requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = runner_module._losses(model(batch), batch, options)["match_loss"]
        loss.backward()
        gradient = batch["team_box_batting_features"].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    run = tmp_path / "cross-era-run"
    training = train_kbo_relgnn(directory, run, config=options, progress=lambda _: None)
    policy = training["training_policies"]
    assert policy["selection_target"] == "match"
    assert policy["box_gradient_mode"] == "head_only"
    assert policy["gradient_clipping"] == "primary_and_box_heads_separately"
    for record in training["history"]:
        assert record["training_samples"]["box_pa"] > 0
        assert record["training_samples"]["box_pitch"] > 0
        assert record["gradient_audit"]["primary"]["max_finite_preclip_norm"] > 0
        assert record["gradient_audit"]["box_heads"]["max_finite_preclip_norm"] > 0
        validation = record["validation"]
        assert validation["selection_loss"] == validation["losses"]["match"]
        assert validation["loss_sample_counts"]["box_pa"] > 0
        assert validation["loss_sample_counts"]["box_pitch"] > 0
    report = evaluate_kbo_relgnn(run / "best.pt", device="cpu", amp="off", workers=0)
    metrics = report["metrics"]
    assert metrics["selection_target"] == "match"
    assert metrics["selection_loss"] == metrics["losses"]["match"]
    assert metrics["weighted_multitask_loss"] == pytest.approx(
        sum(metrics["weighted_loss_contributions"].values())
    )
    assert metrics["box_pa"]["observed_outcomes"] == metrics["loss_sample_counts"]["box_pa"]
    assert metrics["box_pitch"]["observed_counts"] == metrics["loss_sample_counts"]["box_pitch"]


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


@pytest.mark.parametrize("manifest_version", [2, 3, 4, 5])
def test_split_summary_counts_actual_modern_box_targets(
    graph_directory: Path, manifest_version: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = runner_module.KBOGraphDataset(graph_directory)
    config = KBOTrainingConfig(
        train_seasons=(2023, 2024), validation_season=2025, test_season=2026,
    )
    splits = _split_days(dataset, config)
    keys = ("box_pa_queries", "box_pa_outcomes", "box_pitch_queries", "box_pitch_observed_counts")
    expected: dict[str, dict[str, int]] = {}
    for split, days in splits.items():
        totals = dict.fromkeys(keys, 0)
        for day in days:
            graph = dataset.load_day(day)
            totals["box_pa_queries"] += len(graph.box_pa_counts)
            totals["box_pa_outcomes"] += int(graph.box_pa_counts.sum())
            totals["box_pitch_queries"] += len(graph.box_pitch_mask)
            totals["box_pitch_observed_counts"] += int(graph.box_pitch_mask.sum())
        assert all(value > 0 for value in totals.values())
        expected[split] = totals
    dataset.manifest["dataset_version"] = manifest_version
    if manifest_version < 5:
        for entry in dataset.manifest["days"]:
            entry.update(dict.fromkeys(keys, 0))
    else:
        def unexpected_load(day: object) -> None:
            raise AssertionError("v5 coverage must use complete manifest counts")

        monkeypatch.setattr(dataset, "load_day", unexpected_load)
    result = _split_summary(dataset, {**splits, "empty": ()})
    for split, totals in expected.items():
        assert {key: result[split][key] for key in keys} == totals
        assert result[split]["box_coverage_source"] == (
            "manifest_all_sources" if manifest_version >= 5 else "graph_arrays_legacy_manifest"
        )
    assert all(result["empty"][key] == 0 for key in keys)


@pytest.mark.parametrize(
    ("chronological", "training", "expected_shuffle"),
    [(True, True, False), (False, True, True), (False, False, False)],
)
@pytest.mark.parametrize("workers", [0, 2])
def test_loader_orders_training_dates_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chronological: bool,
    training: bool,
    expected_shuffle: bool,
    workers: int,
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
    config = replace(_config(chronological=chronological), workers=workers)
    loader = _loader(tmp_path, days, config, epoch=3, training=training)
    assert loader["shuffle"] is expected_shuffle
    assert loader["generator"].seed == config.seed + 3
    assert loader["num_workers"] == workers
    assert loader["multiprocessing_context"] == ("spawn" if workers else None)
    assert loader["batch_size"] == config.batch_days
    assert loader["pin_memory"] is False
    assert loader["collate_fn"].func is runner_module._collate_loader_days
    assert loader["collate_fn"].keywords == {
        "config": config,
        "epoch": 3,
        "training": training,
    }
    expected_days = days if expected_shuffle else tuple(sorted(days))
    assert loader["dataset"].selected_days == expected_days


@pytest.mark.parametrize("manifest_version", [2, 3, 4, 5])
def test_training_never_loads_held_out_test_graphs_for_any_dataset_version(
    graph_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_version: int,
) -> None:
    pytest.importorskip("torch")
    dataset = runner_module.KBOGraphDataset(graph_directory)
    dataset.manifest["dataset_version"] = manifest_version
    original_load = dataset.load_day
    loaded: list[date] = []

    def tracked_load(day: date | str) -> Any:
        selected = date.fromisoformat(day) if isinstance(day, str) else day
        loaded.append(selected)
        return original_load(day)

    monkeypatch.setattr(dataset, "load_day", tracked_load)
    monkeypatch.setattr(runner_module, "KBOGraphDataset", lambda _: dataset)

    class StopAfterSplitSummary(Exception):
        pass

    def stop_before_model(*args: Any, **kwargs: Any) -> None:
        raise StopAfterSplitSummary

    monkeypatch.setattr(runner_module, "KBORelGNNModel", stop_before_model)
    with pytest.raises(StopAfterSplitSummary):
        train_kbo_relgnn(
            graph_directory,
            tmp_path / f"sealed-test-v{manifest_version}",
            config=_config(),
            progress=lambda _: None,
        )
    assert all(day.year != 2025 for day in loaded)
    assert runner_module._sealed_split_summary(
        (date(2025, 4, 1), date(2025, 4, 2))
    ) == {
        "seasons": [2025],
        "days": 2,
        "date_start": "2025-04-01",
        "date_end": "2025-04-02",
        "labels_or_graphs_loaded": False,
    }


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
