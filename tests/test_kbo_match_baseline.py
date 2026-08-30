from __future__ import annotations

import json
from datetime import date
from typing import Any

import duckdb
import numpy as np
import pytest

from cpv26.pipelines.kbo_match_baseline import (
    DRAW,
    HOME_LOSS,
    HOME_RESULT_LABELS,
    HOME_WIN,
    MATCH_CANONICAL_SQL,
    MATCH_FEATURE_NAMES,
    build_fixed_season_splits,
    build_pregame_match_dataset,
    canonicalize_game_rows,
    evaluate_fixed_season_catboost_json,
)


def test_current_result_does_not_change_its_pregame_features() -> None:
    history = [("prior", "2023-04-01", "A", "B", 3, 1)]
    home_win = build_pregame_match_dataset([*history, ("target", "2023-04-02", "A", "B", 10, 0)])
    away_win = build_pregame_match_dataset([*history, ("target", "2023-04-02", "A", "B", 0, 10)])

    np.testing.assert_allclose(home_win.features[1], away_win.features[1])
    assert home_win.targets.tolist() == [HOME_WIN, HOME_WIN]
    assert away_win.targets.tolist() == [HOME_WIN, HOME_LOSS]
    home_games = MATCH_FEATURE_NAMES.index("home_games_before")
    elo_difference = MATCH_FEATURE_NAMES.index("elo_difference_before")
    assert home_win.features[1, home_games] == 1.0
    assert home_win.features[1, elo_difference] > 0.0


def test_same_day_rows_are_features_before_any_same_day_result() -> None:
    dataset = build_pregame_match_dataset(
        [
            ("g1", "2023-04-01", "A", "B", 5, 0),
            ("g2", "2023-04-01", "A", "C", 0, 5),
            ("g3", "2023-04-02", "A", "D", 2, 1),
        ]
    )
    home_games = MATCH_FEATURE_NAMES.index("home_games_before")
    home_elo = MATCH_FEATURE_NAMES.index("home_elo_before")

    assert dataset.features[0, home_games] == 0.0
    assert dataset.features[1, home_games] == 0.0
    assert dataset.features[0, home_elo] == 1_500.0
    assert dataset.features[1, home_elo] == 1_500.0
    assert dataset.features[2, home_games] == 2.0


def test_canonicalizer_accepts_mapping_aliases_and_duckdb_tuples() -> None:
    games = canonicalize_game_rows(
        [
            (date(2024, 4, 2), "A", "B", 2, 2),
            {
                "game_pk": "naver-1",
                "game_date": "2024-04-01",
                "home_team": "C",
                "away_team": "D",
                "home_score": 1,
                "away_score": 3,
            },
        ]
    )

    assert [game.game_date for game in games] == [date(2024, 4, 1), date(2024, 4, 2)]
    assert games[0].game_id == "naver-1"
    assert games[0].target_class == HOME_LOSS
    assert games[1].target_class == DRAW
    assert games[1].game_id == "2024-04-02:B@A"


def test_fixed_season_splits_use_required_expanding_boundaries() -> None:
    dataset = build_pregame_match_dataset(_three_season_rows())

    validation, test = build_fixed_season_splits(dataset)

    assert set(dataset.seasons[validation.train_indices]) == {2023}
    assert set(dataset.seasons[validation.evaluation_indices]) == {2024}
    assert set(dataset.seasons[test.train_indices]) == {2023, 2024}
    assert set(dataset.seasons[test.evaluation_indices]) == {2025}
    with pytest.raises(ValueError, match="read-only"):
        validation.train_indices[0] = 99


def test_catboost_evaluation_contract_retrains_and_returns_strict_json() -> None:
    fitted_targets: list[list[int]] = []

    class FakeClassifier:
        @property
        def classes_(self) -> np.ndarray[Any, np.dtype[np.int64]]:
            # Deliberately non-canonical to verify probability-column alignment.
            return np.asarray([HOME_WIN, HOME_LOSS, DRAW], dtype=np.int64)

        def fit(self, features: Any, targets: Any) -> FakeClassifier:
            assert np.asarray(features).ndim == 2
            fitted_targets.append(np.asarray(targets, dtype=np.int64).tolist())
            return self

        def predict_proba(self, features: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
            rows = len(features)
            # Columns correspond to classes_ = W, L, D.
            return np.tile(np.asarray([0.55, 0.35, 0.10]), (rows, 1))

    payload = evaluate_fixed_season_catboost_json(
        _three_season_rows(),
        model_factory=FakeClassifier,
        n_calibration_bins=3,
    )
    result = json.loads(payload)

    assert payload.endswith("\n")
    assert result["class_order"] == list(HOME_RESULT_LABELS)
    assert [fold["name"] for fold in result["folds"]] == [
        "validation_2024",
        "test_2025",
    ]
    assert [fold["train_seasons"] for fold in result["folds"]] == [
        [2023],
        [2023, 2024],
    ]
    assert len(fitted_targets) == 2
    assert len(fitted_targets[1]) > len(fitted_targets[0])
    assert set(fitted_targets[0]) == {HOME_LOSS, DRAW, HOME_WIN}
    for fold in result["folds"]:
        assert set(fold["metrics"]) == {
            "accuracy",
            "brier_score",
            "expected_calibration_error",
            "log_loss",
        }
        assert fold["prior_baseline"]["method"] == "training_class_prevalence_laplace_1"
        assert sum(fold["prior_baseline"]["probabilities"]) == pytest.approx(1.0)


def test_completed_game_validation_rejects_missing_score_and_duplicate_id() -> None:
    with pytest.raises(ValueError, match="cannot be null"):
        canonicalize_game_rows(
            [
                {
                    "game_date": "2024-04-01",
                    "home_team_id": "A",
                    "away_team_id": "B",
                    "home_score": None,
                    "away_score": 1,
                }
            ]
        )
    with pytest.raises(ValueError, match="duplicate game_id"):
        canonicalize_game_rows(
            [
                ("same", "2024-04-01", "A", "B", 1, 0),
                ("same", "2024-04-02", "B", "A", 0, 1),
            ]
        )


@pytest.mark.parametrize(
    ("latest_status", "latest_home_score", "latest_season", "latest_valid_to"),
    [
        ("cancelled", 3, 2025, None),
        ("final", None, 2025, None),
        ("final", 3, 2026, None),
        ("final", 3, 2025, "2026-02-01 00:00:00+00"),
    ],
)
def test_match_sql_never_resurrects_old_final_after_latest_revision_filter(
    latest_status: str,
    latest_home_score: int | None,
    latest_season: int,
    latest_valid_to: str | None,
) -> None:
    with duckdb.connect() as connection:
        _create_match_query_table(connection)
        connection.execute(
            "INSERT INTO game VALUES "
            "('old', 'game', 2025, '2025-04-01 00:00:00+09', 'A', 'B', 3, 1, 'final', "
            "'2025-04-02 00:00:00+09', '2025-04-02 00:00:00+09', "
            "'2025-04-02 00:00:00+09', NULL)"
        )
        connection.execute(
            "INSERT INTO game VALUES "
            "('new', 'game', ?, ?, 'A', 'B', ?, 1, ?, "
            "'2026-01-01 00:00:00+09', '2026-01-01 00:00:00+09', "
            "'2026-01-01 00:00:00+09', ?)",
            [
                latest_season,
                f"{latest_season}-04-01 00:00:00+09",
                latest_home_score,
                latest_status,
                latest_valid_to,
            ],
        )
        assert connection.execute(MATCH_CANONICAL_SQL).fetchall() == []


def test_match_sql_keeps_new_final_after_old_cancelled_revision() -> None:
    with duckdb.connect() as connection:
        _create_match_query_table(connection)
        for row_id, timestamp, status, home_score in (
            ("old", "2025-04-01 00:00:00+09", "cancelled", None),
            ("new", "2025-04-02 00:00:00+09", "final", 5),
        ):
            connection.execute(
                "INSERT INTO game VALUES "
                "(?, 'game', 2025, '2025-04-01 00:00:00+09', 'A', 'B', ?, 2, ?, ?, ?, ?, NULL)",
                [row_id, home_score, status, timestamp, timestamp, timestamp],
            )
        assert connection.execute(MATCH_CANONICAL_SQL).fetchall() == [
            ("game", date(2025, 4, 1), "A", "B", 5, 2)
        ]


def _create_match_query_table(connection: Any) -> None:
    connection.execute(
        "CREATE TABLE game (game_row_id VARCHAR, game_id VARCHAR, season INTEGER, "
        "scheduled_start TIMESTAMPTZ, home_team_id VARCHAR, away_team_id VARCHAR, "
        "home_score INTEGER, away_score INTEGER, game_status VARCHAR, "
        "available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ, "
        "valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ)"
    )


def _three_season_rows() -> list[tuple[str, str, str, str, int, int]]:
    rows: list[tuple[str, str, str, str, int, int]] = []
    outcomes = ((3, 1), (2, 2), (1, 3), (4, 0), (0, 4), (1, 1))
    for season in (2023, 2024, 2025):
        for day, (home_score, away_score) in enumerate(outcomes, start=1):
            rows.append(
                (
                    f"{season}-{day}",
                    f"{season}-04-{day:02d}",
                    "A" if day % 2 else "B",
                    "B" if day % 2 else "A",
                    home_score,
                    away_score,
                )
            )
    return rows
