from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pytest

import cpv26.training.kbo_matched_ablation as matched
import cpv26.training.kbo_runner as runner
from cpv26.data.kbo_graph_dataset import KBOGraphDataset, build_kbo_graph_dataset
from cpv26.data.kbo_playbyplay import sha256_file

torch = pytest.importorskip("torch")


@pytest.fixture
def graph_directory(tmp_path: Path) -> Path:
    """Build a tiny real graph cache with train, validation, and sealed test years."""

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
                away_team_id VARCHAR, game_status VARCHAR, home_score INTEGER,
                away_score INTEGER, source_revision_id VARCHAR, event_at TIMESTAMPTZ,
                available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ,
                valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
            );
            CREATE TABLE observed_plate_appearance (
                observed_pa_row_id VARCHAR, plate_appearance_id VARCHAR, game_id VARCHAR,
                inning INTEGER, half_inning VARCHAR, batter_id VARCHAR, pitcher_id VARCHAR,
                batting_team_id VARCHAR, fielding_team_id VARCHAR,
                home_score_before INTEGER, away_score_before INTEGER, outs_before INTEGER,
                runners_before VARCHAR, outcome VARCHAR, is_at_bat BOOLEAN, is_hit BOOLEAN,
                total_bases INTEGER, source_revision_id VARCHAR, event_at TIMESTAMPTZ,
                available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ,
                valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
            )
        """)
        origin = datetime(2023, 1, 1, tzinfo=timezone.utc)
        connection.execute(
            "INSERT INTO source_revision VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["fixture", "pytest", "local", "a" * 64, "{}", origin, origin, origin, origin, None],
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
                appearances = (
                    ("b1", "p1", "H", "A", "single"),
                    ("b2", "p2", "A", "H", "strikeout"),
                )
                for number, (batter, pitcher, offense, defense, outcome) in enumerate(
                    appearances
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


def _config(epochs: int = 1, **overrides: Any) -> runner.KBOTrainingConfig:
    options: dict[str, Any] = {
        "device": "cpu",
        "amp": "off",
        "workers": 0,
        "epochs": epochs,
        "batch_days": 2,
        "hidden_dim": 8,
        "layers": 1,
        "heads": 2,
        "dropout": 0.0,
        "patience": 0,
        "max_pa_per_day": 0,
        "max_edges_per_route_per_day": 0,
        "seed": 17,
        "graph_control_seed": 991,
        "train_seasons": (2023,),
        "validation_season": 2024,
        "test_season": 2025,
    }
    options.update(overrides)
    return runner.KBOTrainingConfig(**options)


def test_six_variant_policies_resolve_to_exact_direction_schedules() -> None:
    base = _config(layers=3)
    protocols = matched._variant_protocols(base)
    core = [
        "batter_pa_pitcher__forward",
        "batter_pa_pitcher__reverse",
        "batter_participation_team__forward",
        "batter_participation_team__reverse",
    ]
    staged_first = [
        "batter_pa_pitcher__forward",
        "batter_pa_pitcher__reverse",
        "batter_participation_team__forward",
        "pitcher_participation_team__reverse",
        "home_team_game_away_team__forward",
        "home_team_game_away_team__reverse",
    ]
    assert matched.MATCHED_GRAPH_VARIANTS == (
        "full",
        "normalized",
        "staged",
        "core",
        "node_only",
        "rewired",
    )
    assert {
        name: (
            item["route_message_normalization"],
            item["route_schedule"],
            item["graph_control"],
        )
        for name, item in protocols.items()
    } == {
        "full": ("none", "full", "intact"),
        "normalized": ("layer_norm", "full", "intact"),
        "staged": ("layer_norm", "staged", "intact"),
        "core": ("layer_norm", "core", "intact"),
        "node_only": ("none", "node_only", "intact"),
        "rewired": ("none", "full", "permuted_endpoints"),
    }
    assert protocols["full"]["resolved_route_schedule"] is None
    assert protocols["normalized"]["resolved_route_schedule"] is None
    assert protocols["rewired"]["resolved_route_schedule"] is None
    assert protocols["staged"]["resolved_route_schedule"] == [
        staged_first,
        core,
        core,
    ]
    assert protocols["core"]["resolved_route_schedule"] == [core, core, core]
    assert protocols["node_only"]["resolved_route_schedule"] == [[], [], []]


def test_same_seed_audit_matches_shared_initialization_and_reports_variant_capacity(
    graph_directory: Path,
) -> None:
    dataset = KBOGraphDataset(graph_directory)
    audit = matched._initialization_audit(dataset, _config(layers=2), 731)
    variants = audit["variants"]

    assert audit["all_shared_parameters_equal"] is True
    assert {
        item["shared_parameter_initialization_sha256"] for item in variants.values()
    } == {audit["shared_parameter_initialization_sha256"]}
    assert variants["full"]["parameter_count"] > variants["node_only"][
        "parameter_count"
    ]
    assert variants["node_only"]["architecture"]["effective_relational_layers"] == 0
    assert variants["node_only"]["architecture"]["relational_parameter_count"] == 0
    assert variants["full"]["architecture"]["effective_relational_layers"] == 2


def test_split_fingerprint_is_exact_and_excludes_held_out_test(
    graph_directory: Path,
) -> None:
    dataset = KBOGraphDataset(graph_directory)
    fingerprint, days = matched._split_day_fingerprint(dataset, _config())
    encoded = json.dumps(days, sort_keys=True, separators=(",", ":")).encode("ascii")

    assert fingerprint == hashlib.sha256(encoded).hexdigest()
    assert set(days) == {"train", "validation"}
    assert all(value.startswith("2023-") for value in days["train"])
    assert all(value.startswith("2024-") for value in days["validation"])
    later_test_fingerprint, later_test_days = matched._split_day_fingerprint(
        dataset,
        replace(_config(), test_season=2026),
    )
    assert later_test_fingerprint == fingerprint
    assert later_test_days == days


def test_runner_loader_applies_epoch_invariant_rewired_control(
    graph_directory: Path,
) -> None:
    dataset = KBOGraphDataset(graph_directory)
    intact_config = _config(chronological=True)
    rewired_config = replace(
        intact_config,
        graph_control="permuted_endpoints",
        graph_control_seed=413,
    )
    days = runner._split_days(dataset, intact_config)["train"]
    intact = next(
        iter(runner._loader(graph_directory, days, intact_config, epoch=0, training=True))
    )
    first = next(
        iter(runner._loader(graph_directory, days, rewired_config, epoch=0, training=True))
    )
    later = next(
        iter(runner._loader(graph_directory, days, rewired_config, epoch=9, training=True))
    )

    for key in ("day_ids", "match_query_ids", "live_hit_query_ids", "pa_query_ids"):
        assert first[key] == intact[key] == later[key]
    for key in ("match_targets", "live_hit_pa", "live_hit_hits", "pa_targets"):
        torch.testing.assert_close(first[key], intact[key], rtol=0, atol=0)
        torch.testing.assert_close(first[key], later[key], rtol=0, atol=0)
    changed = False
    for original, transformed, repeated in zip(
        intact["routes"], first["routes"], later["routes"], strict=True
    ):
        assert original.route_name == transformed.route_name == repeated.route_name
        torch.testing.assert_close(
            transformed.source_index, repeated.source_index, rtol=0, atol=0
        )
        torch.testing.assert_close(
            transformed.destination_index,
            repeated.destination_index,
            rtol=0,
            atol=0,
        )
        changed = changed or not torch.equal(
            original.source_index, transformed.source_index
        ) or not torch.equal(original.destination_index, transformed.destination_index)
    assert changed


def test_suite_manifest_resume_allows_only_epoch_extension(
    graph_directory: Path,
    tmp_path: Path,
) -> None:
    dataset = KBOGraphDataset(graph_directory)
    # Missing compact_kbo_channels denotes the legacy dense architecture.
    base = replace(_config(), compact_kbo_channels=False)
    split_fingerprint, _ = matched._split_day_fingerprint(dataset, base)

    def manifest(config: runner.KBOTrainingConfig) -> dict[str, Any]:
        return matched._suite_manifest(
            dataset,
            graph_directory.resolve(),
            config,
            (17, 29),
            split_fingerprint,
            matched._variant_protocols(config),
        )

    path = tmp_path / "suite_config.json"
    original = manifest(base)
    matched._validate_or_write_manifest(path, original)
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["base_training_config"].pop("activation_checkpointing")
    legacy["base_training_config"].pop("compact_kbo_channels")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    matched._validate_or_write_manifest(path, original)
    extended = manifest(replace(base, epochs=3))
    matched._validate_or_write_manifest(path, extended)
    assert json.loads(path.read_text(encoding="utf-8"))["base_training_config"][
        "epochs"
    ] == 3
    with pytest.raises(ValueError, match="cannot decrease"):
        matched._validate_or_write_manifest(path, manifest(replace(base, epochs=2)))
    with pytest.raises(ValueError, match="fairness setting"):
        matched._validate_or_write_manifest(
            path,
            manifest(replace(base, epochs=4, batch_days=1)),
        )

    changed_runtime = manifest(replace(base, epochs=4))
    changed_runtime["runtime_signature"]["torch_version"] = "different-runtime"
    with pytest.raises(ValueError, match="fairness setting"):
        matched._validate_or_write_manifest(path, changed_runtime)


def test_child_report_normalizes_legacy_missing_execution_fields() -> None:
    expected = matched._variant_config(
        replace(_config(epochs=2), compact_kbo_channels=False), "full", 37
    )
    configuration = asdict(expected)
    configuration.pop("activation_checkpointing")
    configuration.pop("compact_kbo_channels")
    report = {
        "configuration": configuration,
        "completed_epochs": 2,
        "test_used_during_training": False,
        "trainable_parameter_count": 123,
        "parameter_contract": {
            "trainable_parameter_count": 123,
            "optimizer_covers_all_trainable": True,
        },
        "all_epochs_trainable_parameters_received_gradient": True,
    }

    matched._validate_child_report(report, expected)

    truncated = copy.deepcopy(report)
    truncated["configuration"].pop("dropout")
    with pytest.raises(ValueError, match="fairness settings"):
        matched._validate_child_report(truncated, expected)


def test_interrupted_child_checkpoint_rejects_fairness_and_control_mismatch(
    graph_directory: Path,
) -> None:
    dataset = KBOGraphDataset(graph_directory)
    expected = matched._variant_config(_config(epochs=2), "rewired", 37)
    audit = matched._initialization_audit(dataset, expected, 37)
    initialization = audit["variants"]["rewired"]
    state: dict[str, Any] = {
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "training_config": asdict(replace(expected, epochs=1)),
        "graph_control": runner._graph_control_report(expected),
        "model_config": runner._model_config(dataset, expected).to_dict(),
        "initial_model_state_sha256": initialization["initial_model_state_sha256"],
        "parameter_count": initialization["parameter_count"],
        "trainable_parameter_count": initialization["trainable_parameter_count"],
        "epoch": 1,
    }
    matched._validate_child_checkpoint(
        state,
        dataset=dataset,
        expected=expected,
        initialization=initialization,
    )

    changed_config = copy.deepcopy(state)
    changed_config["training_config"]["batch_days"] = 1
    with pytest.raises(ValueError, match="fairness setting"):
        matched._validate_child_checkpoint(
            changed_config,
            dataset=dataset,
            expected=expected,
            initialization=initialization,
        )

    changed_control = copy.deepcopy(state)
    changed_control["graph_control"] = runner._graph_control_report(
        replace(expected, graph_control_seed=expected.graph_control_seed + 1)
    )
    with pytest.raises(ValueError, match="graph-control protocol"):
        matched._validate_child_checkpoint(
            changed_control,
            dataset=dataset,
            expected=expected,
            initialization=initialization,
        )


def _metric_block(base: float, offset: float) -> dict[str, float]:
    return {
        "log_loss": base + offset,
        "accuracy": 0.5 + offset,
        "expected_calibration_error": 0.1 + offset,
        "brier_score": 0.4 + offset,
    }


def test_aggregate_reports_seed_paired_deltas_against_full() -> None:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for seed_index, seed in enumerate(("11", "29")):
        per_seed: dict[str, dict[str, Any]] = {}
        full_loss = 1.0 + seed_index
        for variant_index, variant in enumerate(matched.MATCHED_GRAPH_VARIANTS):
            offset = variant_index / 10
            per_seed[variant] = {
                "validation_selection_loss": full_loss + offset,
                "selection_loss_delta_vs_full": offset,
                "best_epoch": 10 + variant_index,
                "final_validation_selection_loss": full_loss + offset + 0.25,
                "final_minus_best_selection_loss": 0.25,
                "validation_metrics": {
                    **{
                        task: _metric_block(0.8 + seed_index, offset)
                        for task in ("match", "live_hit", "pa")
                    },
                    "losses": {
                        "match": 0.8 + seed_index + offset,
                        "live_hit": 2.0 + seed_index + offset,
                        "pa": 1.5 + seed_index + offset,
                        "run": 4.0 + seed_index + offset,
                        "box_pa": 1.2 + seed_index + offset,
                        "box_pitch": 3.0 + seed_index + offset,
                    },
                    "weighted_loss_contributions": {
                        "match": 0.8 + seed_index + offset,
                        "live_hit": 2.0 + seed_index + offset,
                        "pa": 0.3 + offset / 5,
                        "run": 0.4 + offset / 10,
                        "box_pa": 0.24 + offset / 5,
                        "box_pitch": 0.3 + offset / 10,
                    },
                },
                "parameter_count": 1234,
            }
            per_seed[variant]["validation_metrics"]["live_hit"].update(
                joint_nll=2.0 + seed_index + offset,
                expected_hits_lower_bound_mae=0.6 + offset,
            )
        runs[seed] = per_seed

    report = matched._aggregate_runs(runs)
    assert report["full"]["validation_selection_loss"][
        "paired_delta_vs_full_mean"
    ] == pytest.approx(0.0)
    normalized = report["normalized"]["validation_selection_loss"]
    assert normalized["mean"] == pytest.approx(1.6)
    assert normalized["population_std"] == pytest.approx(0.5)
    assert normalized["paired_delta_vs_full_mean"] == pytest.approx(0.1)
    assert normalized["paired_delta_vs_full_population_std"] == pytest.approx(0.0)
    assert normalized["paired_delta_vs_core_mean"] == pytest.approx(-0.2)
    assert report["core"]["validation_metrics"]["match"]["accuracy"][
        "paired_delta_vs_full_mean"
    ] == pytest.approx(0.3)
    assert report["normalized"]["validation_losses"]["run"]["mean"] == pytest.approx(4.6)
    assert report["normalized"]["validation_losses"]["run"][
        "paired_delta_vs_core_mean"
    ] == pytest.approx(-0.2)
    assert report["normalized"]["validation_weighted_loss_contributions"]["pa"][
        "mean"
    ] == pytest.approx(0.32)
    assert report["normalized"]["validation_metrics"]["live_hit"]["joint_nll"][
        "paired_delta_vs_full_mean"
    ] == pytest.approx(0.1)
    assert report["normalized"]["checkpoint_selection"]["best_epoch"][
        "mean"
    ] == pytest.approx(11.0)
    assert report["normalized"]["checkpoint_selection"][
        "final_minus_best_selection_loss"
    ]["mean"] == pytest.approx(0.25)
    contrasts = matched._aggregate_named_contrasts(runs)
    assert contrasts["normalization"]["validation_selection_loss"][
        "delta_mean"
    ] == pytest.approx(0.1)
    assert contrasts["core_pruning"]["validation_selection_loss"][
        "delta_mean"
    ] == pytest.approx(0.2)
    assert contrasts["remove_messages"]["validation_selection_loss"][
        "delta_mean"
    ] == pytest.approx(0.4)
    assert contrasts["core_pruning"]["validation_selection_loss"][
        "population_std"
    ] == pytest.approx(0.0)
    one_seed = matched._aggregate_named_contrasts({"11": runs["11"]})
    assert one_seed["core_pruning"]["validation_selection_loss"][
        "population_std"
    ] is None


@pytest.mark.parametrize(
    ("selection_split", "test_sealed", "message"),
    [
        ("test", False, "did not use validation"),
        ("validation", True, "does not prove"),
    ],
)
def test_saved_ablation_analysis_fails_closed_before_reporting_test_policy(
    tmp_path: Path,
    selection_split: str,
    test_sealed: bool,
    message: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "matched_retraining_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol": "matched_from_scratch_validation_graph_ablation",
                "protocol_version": matched.MATCHED_ABLATION_PROTOCOL_VERSION,
                "selection_split": selection_split,
                "test_used_for_training_selection_or_comparison": test_sealed,
                "runs": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        matched.analyze_matched_graph_ablations(suite)


def test_validation_reevaluation_commits_only_a_complete_hash_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"matched-checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    final = run / "matched_validation" / checkpoint_hash[:16]

    def fail_mid_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        temporary = Path(kwargs["output_directory"])
        temporary.mkdir(parents=True)
        (temporary / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "evaluate_kbo_relgnn", fail_mid_evaluation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        matched._reevaluate_best_on_validation(run, tmp_path / "graph", _config())
    assert not final.exists()

    def complete_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        temporary = Path(kwargs["output_directory"])
        temporary.mkdir(parents=True)
        prediction = temporary / "match_predictions.parquet"
        prediction.write_bytes(b"predictions")
        return {
            "split": "validation",
            "checkpoint_sha256": checkpoint_hash,
            "output_directory": str(temporary),
            "prediction_artifacts": {
                "match": {
                    "path": str(prediction),
                    "sha256": sha256_file(prediction),
                    "rows": 1,
                }
            },
            "metrics": {"selection_loss": 1.0},
        }

    monkeypatch.setattr(runner, "evaluate_kbo_relgnn", complete_evaluation)
    report = matched._reevaluate_best_on_validation(run, tmp_path / "graph", _config())
    assert report["output_directory"] == str(final)
    assert report["prediction_artifacts"]["match"]["path"] == str(
        final / "match_predictions.parquet"
    )
    assert (final / "metrics.json").is_file()
    saved = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    assert saved == report


def test_real_tiny_suite_is_matched_resumable_and_never_loads_or_evaluates_test(
    graph_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated_splits: list[str] = []
    loaded_years: list[int] = []
    original_evaluate = runner.evaluate_kbo_relgnn
    original_load_day = KBOGraphDataset.load_day

    def recording_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluated_splits.append(str(kwargs.get("split")))
        return original_evaluate(*args, **kwargs)

    def recording_load_day(self: KBOGraphDataset, day: Any) -> Any:
        value = day if hasattr(day, "year") else datetime.fromisoformat(str(day))
        loaded_years.append(int(value.year))
        return original_load_day(self, day)

    monkeypatch.setattr(runner, "evaluate_kbo_relgnn", recording_evaluate)
    monkeypatch.setattr(KBOGraphDataset, "load_day", recording_load_day)
    output = tmp_path / "matched"
    config = _config(dropout=0.2)
    report = matched.train_matched_graph_ablations(
        graph_directory,
        output,
        base_config=config,
        seeds=(17,),
        progress=lambda _: None,
    )

    assert report["status"] == "completed"
    assert report["selection_split"] == "validation"
    assert report["test_used_for_training_selection_or_comparison"] is False
    assert evaluated_splits == ["validation"] * len(matched.MATCHED_GRAPH_VARIANTS)
    assert loaded_years and set(loaded_years) == {2023, 2024}
    runs = report["runs"]["17"]
    assert tuple(runs) == matched.MATCHED_GRAPH_VARIANTS
    # Pruned ablations intentionally have different complete state dictionaries;
    # fairness is bit-identical initialization of the parameters shared by all.
    assert len(
        {
            run["shared_parameter_initialization_sha256"]
            for run in runs.values()
        }
    ) == 1
    assert len({run["parameter_count"] for run in runs.values()}) > 1
    assert runs["full"]["parameter_count"] > runs["node_only"]["parameter_count"]
    assert len({run["attempted_optimizer_steps"] for run in runs.values()}) == 1
    assert len({run["completed_epochs"] for run in runs.values()}) == 1
    assert all(run["test_used_during_training"] is False for run in runs.values())
    checkpoint_hashes = {
        variant: sha256_file(Path(run["best_checkpoint"]))
        for variant, run in runs.items()
    }

    resumed = matched.train_matched_graph_ablations(
        graph_directory,
        output,
        base_config=config,
        seeds=(17,),
        progress=lambda _: None,
    )
    assert evaluated_splits == ["validation"] * len(matched.MATCHED_GRAPH_VARIANTS)
    assert resumed["aggregate"] == report["aggregate"]
    assert {
        variant: sha256_file(Path(run["best_checkpoint"]))
        for variant, run in resumed["runs"]["17"].items()
    } == checkpoint_hashes

    extended_config = replace(config, epochs=2)
    extended = matched.train_matched_graph_ablations(
        graph_directory,
        output,
        base_config=extended_config,
        seeds=(17,),
        progress=lambda _: None,
    )
    uninterrupted = matched.train_matched_graph_ablations(
        graph_directory,
        tmp_path / "uninterrupted",
        base_config=extended_config,
        seeds=(17,),
        progress=lambda _: None,
    )
    assert evaluated_splits and set(evaluated_splits) == {"validation"}
    assert set(loaded_years) == {2023, 2024}
    for variant in matched.MATCHED_GRAPH_VARIANTS:
        resumed_state = runner._read_checkpoint(
            Path(extended["runs"]["17"][variant]["run_directory"]) / "last.pt"
        )
        uninterrupted_state = runner._read_checkpoint(
            Path(uninterrupted["runs"]["17"][variant]["run_directory"]) / "last.pt"
        )
        assert resumed_state["epoch"] == uninterrupted_state["epoch"] == 2
        assert resumed_state["global_step"] == uninterrupted_state["global_step"]
        assert resumed_state["model"].keys() == uninterrupted_state["model"].keys()
        for key, tensor in resumed_state["model"].items():
            torch.testing.assert_close(
                tensor,
                uninterrupted_state["model"][key],
                rtol=0,
                atol=0,
            )
