from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any

import duckdb
import numpy as np
import pytest

from cpv26.pipelines.kbo_live_hit_baseline import (
    ANY_HIT,
    LIVE_HIT_CANONICAL_SQL,
    LIVE_HIT_FEATURE_NAMES,
    LIVE_HIT_LABELS,
    NO_HIT,
    CanonicalPlateAppearanceRow,
    LiveHitFeatureDataset,
    PlateAppearanceInputRow,
    build_live_hit_fixed_season_splits,
    build_pregame_live_hit_dataset,
    canonicalize_plate_appearance_rows,
    evaluate_live_hit_fixed_season_catboost,
    evaluate_live_hit_fixed_season_catboost_json,
    reduce_player_game_live_hit_targets,
)


def _pa(
    pa_id: str,
    game: str,
    day: str,
    player: str = "p1",
    outcome: str = "strikeout",
    team: str = "A",
    opponent: str = "B",
) -> tuple[str, str, str, str, str, str, str]:
    return pa_id, game, day, player, team, opponent, outcome


def _row_index(dataset: LiveHitFeatureDataset, game: str, player: str = "p1") -> int:
    return list(zip(dataset.game_ids, dataset.player_ids, strict=True)).index((game, player))


def _feature(dataset: LiveHitFeatureDataset, game: str, name: str, player: str = "p1") -> float:
    return float(
        dataset.features[_row_index(dataset, game, player), dataset.feature_names.index(name)]
    )


def test_current_game_hit_and_pa_count_never_enter_its_features() -> None:
    prior = [_pa("old-1", "prior", "2023-04-01", outcome="single")]
    one_hit = build_pregame_live_hit_dataset(
        [*prior, _pa("now-1", "current", "2023-04-02", outcome="home_run")]
    )
    many_outs = build_pregame_live_hit_dataset(
        [*prior, *[_pa(f"now-{number}", "current", "2023-04-02") for number in range(1, 7)]]
    )

    index = _row_index(one_hit, "current")
    np.testing.assert_array_equal(one_hit.features[index], many_outs.features[index])
    assert one_hit.targets[index] == ANY_HIT
    assert many_outs.targets[index] == NO_HIT
    assert one_hit.plate_appearances[index] == 1
    assert many_outs.plate_appearances[index] == 6
    assert _feature(one_hit, "current", "player_pa_before") == 1
    assert "plate_appearances" not in LIVE_HIT_FEATURE_NAMES
    assert "hits" not in LIVE_HIT_FEATURE_NAMES


def test_same_date_doubleheaders_share_pre_date_history_and_update_next_date() -> None:
    original = [
        _pa("a", "dh-1", "2023-04-01", outcome="single"),
        _pa("b", "dh-2", "2023-04-01"),
        _pa("c", "next", "2023-04-02"),
    ]
    changed = [
        _pa("a", "dh-1", "2023-04-01"),
        _pa("a-extra", "dh-1", "2023-04-01"),
        *original[1:],
    ]
    first = build_pregame_live_hit_dataset(original)
    second = build_pregame_live_hit_dataset(changed)

    for game in ("dh-1", "dh-2"):
        np.testing.assert_array_equal(
            first.features[_row_index(first, game)], second.features[_row_index(second, game)]
        )
        assert _feature(first, game, "player_games_before") == 0
        assert _feature(first, game, "batting_team_games_before") == 0
        assert _feature(first, game, "opponent_pa_faced_before") == 0
        assert _feature(first, game, "player_vs_opponent_pa_before") == 0
    assert _feature(first, "next", "player_games_before") == 2
    assert _feature(first, "next", "player_hits_before") == 1
    assert _feature(second, "next", "player_hits_before") == 0
    assert _feature(first, "next", "batting_team_games_before") == 2


def test_future_rows_do_not_change_earlier_features_or_targets() -> None:
    earlier = [
        _pa("a", "g1", "2023-04-01", outcome="single"),
        _pa("b", "g2", "2023-04-02"),
    ]
    original = build_pregame_live_hit_dataset(earlier)
    extended = build_pregame_live_hit_dataset(
        [*earlier, _pa("future", "g3", "2026-04-01", outcome="home_run")]
    )

    np.testing.assert_array_equal(original.features, extended.features[:2])
    np.testing.assert_array_equal(original.targets, extended.targets[:2])


def test_player_team_defense_and_smoothed_opponent_relation_use_prior_games() -> None:
    dataset = build_pregame_live_hit_dataset(
        [
            _pa("1", "g1", "2023-04-01", outcome="single"),
            _pa("2", "g1", "2023-04-01"),
            _pa("3", "g1", "2023-04-01", player="p2", outcome="double"),
            _pa("4", "g1", "2023-04-01", player="p3", team="B", opponent="A"),
            _pa("5", "g2", "2023-04-02", opponent="C"),
            _pa("6", "g3", "2023-04-03"),
        ],
        rolling_games=1,
        hit_prior_rate=0.25,
        hit_prior_pa=2,
        relation_prior_pa=4,
    )

    assert _feature(dataset, "g2", "player_games_before") == 1
    assert _feature(dataset, "g2", "player_pa_before") == 2
    assert _feature(dataset, "g2", "player_career_hit_rate_smoothed") == pytest.approx(1.5 / 4)
    assert _feature(dataset, "g2", "batting_team_games_before") == 1
    assert _feature(dataset, "g2", "batting_team_pa_before") == 3
    assert _feature(dataset, "g2", "batting_team_hits_before") == 2
    assert _feature(dataset, "g2", "opponent_pa_faced_before") == 0
    assert _feature(dataset, "g2", "player_vs_opponent_hit_rate_smoothed") == 0.25
    assert _feature(dataset, "g3", "opponent_pa_faced_before") == 3
    assert _feature(dataset, "g3", "opponent_hits_allowed_before") == 2
    assert _feature(dataset, "g3", "player_vs_opponent_pa_before") == 2
    assert _feature(dataset, "g3", "player_vs_opponent_hit_rate_smoothed") == pytest.approx(2 / 6)
    assert _feature(dataset, "g3", "player_recent_pa_before") == 1
    assert _feature(dataset, "g3", "player_recent_hits_before") == 0
    assert _feature(dataset, "g3", "batting_team_recent_games_before") == 1


def test_reduction_counts_pa_not_ab_and_accepts_mapping_and_joined_tuples() -> None:
    rows: list[PlateAppearanceInputRow] = [
        {
            "pa_id": "1",
            "game_id": "g1",
            "game_date": "2023-04-01",
            "player_id": "p1",
            "player_name": "Batter",
            "team_id": "A",
            "fielding_team_id": "B",
            "outcome": "walk",
        },
        ("2", "g1", date(2023, 4, 1), "p1", "Batter", "A", "Alpha", "B", "Beta", "single", True),
        _pa("3", "g1", "2023-04-01", player="p2"),
    ]
    targets = reduce_player_game_live_hit_targets(rows)

    assert len(targets) == 2
    assert targets[0].player_name == "Batter"
    assert targets[0].plate_appearances == 2
    assert targets[0].hits == 1
    assert targets[0].target_class == ANY_HIT
    assert targets[1].target_class == NO_HIT


def test_timezone_is_kbo_local_and_input_order_does_not_change_features() -> None:
    canonical = CanonicalPlateAppearanceRow("a", "g1", date(2023, 4, 1), "p1", "A", "B", "single")
    rows: list[PlateAppearanceInputRow] = [
        {
            "plate_appearance_id": "b",
            "game_id": "g2",
            "batter_id": "p1",
            "batting_team_id": "A",
            "opponent_team_id": "B",
            "outcome": "strikeout",
            "scheduled_start": datetime(2023, 4, 1, 15, 0, tzinfo=timezone.utc),
        },
        canonical,
    ]
    normal = build_pregame_live_hit_dataset(rows)
    reversed_rows = build_pregame_live_hit_dataset(reversed(rows))
    assert normal.game_dates == (date(2023, 4, 1), date(2023, 4, 2))
    np.testing.assert_array_equal(normal.features, reversed_rows.features)
    assert _feature(normal, "g2", "player_games_before") == 1


def test_fixed_season_splits_are_separate_and_exclude_2026() -> None:
    dataset = build_pregame_live_hit_dataset(_three_season_rows())
    validation, test = build_live_hit_fixed_season_splits(dataset)

    assert set(dataset.seasons[validation.train_indices]) == {2023}
    assert set(dataset.seasons[validation.evaluation_indices]) == {2024}
    assert set(dataset.seasons[test.train_indices]) == {2023, 2024}
    assert set(dataset.seasons[test.evaluation_indices]) == {2025}
    with pytest.raises(ValueError, match="read-only"):
        dataset.features[0, 0] = 10
    with pytest.raises(ValueError, match="read-only"):
        dataset.plate_appearances[0] = 10


def test_evaluation_retrains_aligns_binary_classes_and_discloses_appearance_condition() -> None:
    training_sizes: list[int] = []

    class FakeClassifier:
        @property
        def classes_(self) -> np.ndarray[Any, np.dtype[np.int64]]:
            return np.asarray([ANY_HIT, NO_HIT], dtype=np.int64)

        def fit(self, features: Any, targets: Any) -> FakeClassifier:
            training_sizes.append(len(targets))
            assert np.asarray(features).shape[1] == len(LIVE_HIT_FEATURE_NAMES)
            assert set(np.asarray(targets).tolist()) == {NO_HIT, ANY_HIT}
            return self

        def predict_proba(self, features: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
            return np.tile([0.7, 0.3], (len(features), 1))

    payload = evaluate_live_hit_fixed_season_catboost_json(
        _three_season_rows(), model_factory=FakeClassifier, n_calibration_bins=3
    )
    result = json.loads(payload)

    assert payload.endswith("\n")
    assert result["class_order"] == list(LIVE_HIT_LABELS)
    assert result["population"] == "player_game_with_at_least_one_observed_plate_appearance"
    assert (
        result["feature_generation"]["current_game_pa_and_hits"]
        == "target_metadata_only_not_features"
    )
    assert "separate model" in result["limitations"][0]
    assert training_sizes == [2, 4]
    assert [fold["name"] for fold in result["folds"]] == ["validation_2024", "test_2025"]
    for fold in result["folds"]:
        assert fold["metrics"]["accuracy"] == 0.5
        assert fold["metrics"]["log_loss"] == pytest.approx((-np.log(0.7) - np.log(0.3)) / 2)


def test_input_and_dataset_validation_reject_ambiguous_or_inconsistent_rows() -> None:
    single = _pa("one", "g1", "2023-04-01", outcome="single")
    with pytest.raises(ValueError, match="duplicate plate_appearance_id"):
        canonicalize_plate_appearance_rows([single, single])
    with pytest.raises(ValueError, match="is_hit disagrees"):
        canonicalize_plate_appearance_rows([(*single, False)])
    with pytest.raises(ValueError, match="inconsistent date or teams"):
        canonicalize_plate_appearance_rows([single, _pa("two", "g1", "2023-04-02")])
    with pytest.raises(ValueError, match="conflicting teams"):
        reduce_player_game_live_hit_targets(
            [single, _pa("two", "g1", "2023-04-01", team="B", opponent="A")]
        )
    dataset = build_pregame_live_hit_dataset([single])
    with pytest.raises(ValueError, match="at least one observed PA"):
        replace(dataset, plate_appearances=np.asarray([0]))
    with pytest.raises(ValueError, match="requires both hit classes"):
        evaluate_live_hit_fixed_season_catboost(
            [_pa(str(year), str(year), f"{year}-04-01") for year in (2023, 2024, 2025)]
        )


@pytest.mark.parametrize("closed_table", [None, "game", "pa", "player", "team"])
@pytest.mark.parametrize("latest_game_status", ["final", "cancelled"])
def test_canonical_sql_selects_latest_revisions_before_state_filters(
    closed_table: str | None, latest_game_status: str
) -> None:
    connection = duckdb.connect()
    # Minimal canonical schema columns used by the query; two revisions each.
    connection.execute(
        "CREATE TABLE source_revision (source_revision_id VARCHAR, source_name VARCHAR, "
        "metadata_json JSON, ingested_at TIMESTAMPTZ)"
    )
    temporal = (
        "available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ, "
        "valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ, source_revision_id VARCHAR"
    )
    connection.execute(
        "CREATE TABLE observed_plate_appearance "
        "(observed_pa_row_id VARCHAR, plate_appearance_id VARCHAR, "
        f"game_id VARCHAR, batter_id VARCHAR, batting_team_id VARCHAR, fielding_team_id VARCHAR, "
        "outcome VARCHAR, is_hit BOOLEAN, sequence_in_game INTEGER, "
        f"event_subsequence INTEGER, {temporal})"
    )
    connection.execute(
        f"CREATE TABLE game (game_row_id VARCHAR, game_id VARCHAR, scheduled_start TIMESTAMPTZ, "
        f"game_status VARCHAR, {temporal})"
    )
    connection.execute(
        "CREATE TABLE player (player_row_id VARCHAR, player_id VARCHAR, "
        f"display_name VARCHAR, {temporal})"
    )
    connection.execute(
        f"CREATE TABLE team (team_row_id VARCHAR, team_id VARCHAR, team_name VARCHAR, {temporal})"
    )
    for revision in (1, 2):
        timestamp = f"2023-04-0{revision} 00:00:00+09"

        times: dict[str, list[str | None]] = {}
        for table in ("pa", "game", "player", "team"):
            valid_to = "2023-05-01 00:00:00+09" if revision == 2 and closed_table == table else None
            times[table] = [timestamp, timestamp, timestamp, valid_to]

        connection.execute(
            "INSERT INTO observed_plate_appearance "
            "VALUES (?, 'pa', 'g', 'p', 'A', 'B', ?, ?, 1, 0, ?, ?, ?, ?, 'fixture')",
            [
                str(revision),
                "single" if revision == 2 else "strikeout",
                revision == 2,
                *times["pa"],
            ],
        )
        connection.execute(
            "INSERT INTO game VALUES (?, 'g', '2023-04-01 00:00:00+09', ?, ?, ?, ?, ?, 'fixture')",
            [str(revision), latest_game_status if revision == 2 else "final", *times["game"]],
        )
        connection.execute(
            "INSERT INTO player VALUES (?, 'p', ?, ?, ?, ?, ?, 'fixture')",
            [str(revision), f"Player {revision}", *times["player"]],
        )
        for team in ("A", "B"):
            connection.execute(
                "INSERT INTO team VALUES (?, ?, ?, ?, ?, ?, ?, 'fixture')",
                [f"{team}{revision}", team, f"{team} {revision}", *times["team"]],
            )
    try:
        rows = connection.execute(LIVE_HIT_CANONICAL_SQL).fetchall()
    finally:
        connection.close()

    if closed_table in {"game", "pa"} or latest_game_status != "final":
        assert rows == []
        return
    assert len(rows) == 1
    assert rows[0][2] == date(2023, 4, 1)
    assert rows[0][4] == (None if closed_table == "player" else "Player 2")
    assert rows[0][6] == (None if closed_table == "team" else "A 2")
    assert rows[0][8] == (None if closed_table == "team" else "B 2")
    assert rows[0][-2:] == ("single", True)
    assert build_pregame_live_hit_dataset(rows).targets.tolist() == [ANY_HIT]


def _three_season_rows() -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    for year in (2023, 2024, 2025, 2026):
        rows.extend(
            [
                _pa(f"{year}-1", f"{year}-g1", f"{year}-04-01", outcome="single"),
                _pa(f"{year}-2", f"{year}-g2", f"{year}-04-02"),
            ]
        )
    return rows
