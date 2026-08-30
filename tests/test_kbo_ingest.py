from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from cpv26.data import DuckDBStore
from cpv26.data.kbo_ingest import KBOIngestError, import_kbo_playbyplay

PITCH_COLUMNS = (
    "game_pk VARCHAR, game_date VARCHAR, home_team VARCHAR, away_team VARCHAR, "
    "inning INTEGER, inning_topbot VARCHAR, at_bat_number INTEGER, pitch_number INTEGER, "
    "batter VARCHAR, pitcher VARCHAR, batter_name VARCHAR, pitcher_name VARCHAR, "
    "outs_when_up INTEGER, on_1b VARCHAR, on_2b VARCHAR, on_3b VARCHAR, "
    "home_score INTEGER, away_score INTEGER, stand VARCHAR, events VARCHAR, "
    "post_home_score INTEGER, post_away_score INTEGER, post_outs INTEGER, "
    "runs_scored INTEGER, post_on_1b VARCHAR, post_on_2b VARCHAR, post_on_3b VARCHAR, "
    "strikes INTEGER"
)


def _pitch(
    *,
    at_bat_number: int,
    pitch_number: int,
    inning_topbot: str,
    batter: str,
    batter_name: str,
    events: str | None,
    outs_before: int,
    post_outs: int,
    home_score: int,
    away_score: int,
    post_home_score: int,
    post_away_score: int,
    on_1b: str | None = None,
    post_on_1b: str | None = None,
    strikes: int = 0,
    runs_scored: int | None = None,
) -> tuple[object, ...]:
    return (
        "20230401AAHH02023",
        "2023-04-01",
        "HH",
        "AA",
        1,
        inning_topbot,
        at_bat_number,
        pitch_number,
        batter,
        "p1" if inning_topbot == "top" else "p2",
        batter_name,
        "Pitcher One" if inning_topbot == "top" else "Pitcher Two",
        outs_before,
        on_1b,
        None,
        None,
        home_score,
        away_score,
        "L",
        events,
        post_home_score,
        post_away_score,
        post_outs,
        (
            post_home_score + post_away_score - home_score - away_score
            if runs_scored is None
            else runs_scored
        ),
        post_on_1b,
        None,
        None,
        strikes,
    )


def _write_source(path: Path) -> None:
    rows = [
        _pitch(
            at_bat_number=1,
            pitch_number=1,
            inning_topbot="top",
            batter="b1",
            batter_name="Batter One",
            events="single",
            outs_before=0,
            post_outs=0,
            home_score=0,
            away_score=0,
            post_home_score=0,
            post_away_score=0,
            post_on_1b="b1",
        ),
        _pitch(
            at_bat_number=1,
            pitch_number=2,
            inning_topbot="top",
            batter="b1",
            batter_name="Batter One",
            events="single",
            outs_before=0,
            post_outs=0,
            home_score=0,
            away_score=0,
            post_home_score=0,
            post_away_score=0,
            post_on_1b="b1",
        ),
        _pitch(
            at_bat_number=2,
            pitch_number=1,
            inning_topbot="top",
            batter="b2",
            batter_name="Interrupted Batter",
            events=None,
            outs_before=2,
            post_outs=3,
            home_score=0,
            away_score=0,
            post_home_score=0,
            post_away_score=0,
            on_1b="b1",
        ),
        _pitch(
            at_bat_number=3,
            pitch_number=1,
            inning_topbot="bot",
            batter="b3",
            batter_name="Home Batter",
            events="home_run",
            outs_before=0,
            post_outs=0,
            home_score=0,
            away_score=0,
            post_home_score=1,
            post_away_score=0,
        ),
    ]
    _write_rows(path, rows)


def _write_rows(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = duckdb.connect()
    connection.execute(f"CREATE TABLE pitches ({PITCH_COLUMNS})")
    connection.executemany(
        f"INSERT INTO pitches VALUES ({', '.join('?' for _ in range(28))})",
        rows,
    )
    connection.execute("COPY pitches TO ? (FORMAT PARQUET)", [str(path)])
    connection.close()


def test_import_reduces_pitches_preserves_transitions_and_is_retry_safe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kbo_pbp_2023.parquet"
    _write_source(source)
    observed_at = datetime(2026, 8, 30, 1, 2, 3, tzinfo=timezone.utc)

    with DuckDBStore() as store:
        report = import_kbo_playbyplay(
            store,
            [source],
            revision="test-revision",
            ingested_at=observed_at,
        )

        assert report.completed_plate_appearances == 2
        assert report.unlabelled_plate_appearances == 1
        assert report.unlabelled_runs == 0
        assert not report.as_dict()["simulator_ready"]
        assert report.inserted_rows == {
            "source_revision": 1,
            "team": 2,
            "player": 5,
            "game": 1,
            "team_game": 2,
            "observed_plate_appearance": 2,
        }
        plate_appearances = store.connection.execute(
            """
            SELECT outcome, runners_before, runners_after, is_hit, total_bases,
                   available_at, ingested_at
            FROM observed_plate_appearance
            ORDER BY sequence_in_game
            """
        ).fetchall()
        assert plate_appearances[0][:5] == ("single", "000", "100", True, 1)
        assert plate_appearances[1][:5] == ("home_run", "000", "000", True, 4)
        assert plate_appearances[0][5] == datetime(2023, 4, 1, 15, 0, tzinfo=timezone.utc)
        assert plate_appearances[0][6] == observed_at

        team_games = store.connection.execute(
            """
            SELECT team_id, runs, hits, errors, result
            FROM team_game
            ORDER BY team_id
            """
        ).fetchall()
        assert team_games == [
            ("kbo-team:AA", 0, 1, None, "loss"),
            ("kbo-team:HH", 1, 1, None, "win"),
        ]
        store.assert_referential_integrity()
        store.assert_composite_referential_integrity()
        assert (
            store.connection.execute(
                "SELECT count(*) FROM player WHERE bats IS NOT NULL"
            ).fetchone()[0]
            == 0
        )

        repeated = import_kbo_playbyplay(
            store,
            [source],
            revision="test-revision",
            ingested_at=observed_at,
        )
        assert all(value == 0 for value in repeated.inserted_rows.values())


def test_import_credits_substitution_hits_and_two_strike_strikeouts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kbo_pbp_2023.parquet"
    rows: list[tuple[object, ...]] = []
    for pa_number, outcome in ((1, "single"), (2, "strikeout")):
        for pitch_number, batter, strikes in ((1, "original", 1), (2, "substitute", 2)):
            rows.append(
                _pitch(
                    at_bat_number=pa_number,
                    pitch_number=pitch_number,
                    inning_topbot="top",
                    batter=batter,
                    batter_name=batter,
                    events=outcome,
                    outs_before=2,
                    post_outs=3 if outcome == "strikeout" else 2,
                    home_score=0,
                    away_score=0,
                    post_home_score=0,
                    post_away_score=0,
                    post_on_1b="stranded",
                    strikes=strikes,
                )
            )
    _write_rows(source, rows)
    with DuckDBStore() as store:
        import_kbo_playbyplay(store, [source], revision="test-revision")
        observed = store.connection.execute(
            """
            SELECT batter_id, outcome, runners_after
            FROM observed_plate_appearance
            ORDER BY sequence_in_game
            """
        ).fetchall()
        assert observed == [
            ("kbo-player:substitute", "single", "100"),
            ("kbo-player:original", "strikeout", "000"),
        ]


def test_import_preserves_null_boundaries_and_reports_inconsistent_source(tmp_path: Path) -> None:
    source = tmp_path / "kbo_pbp_2023.parquet"
    rows = [
        _pitch(
            at_bat_number=1,
            pitch_number=pitch_number,
            inning_topbot="top",
            batter="b1",
            batter_name="Batter One",
            events="field_out",
            outs_before=0,
            post_outs=1,
            home_score=0,
            away_score=0,
            post_home_score=0,
            post_away_score=0,
            on_1b=None if pitch_number == 1 else "runner",
            post_on_1b="runner" if pitch_number == 1 else None,
            runs_scored=1,
        )
        for pitch_number in (1, 2)
    ]
    rows.append(
        _pitch(
            at_bat_number=3,
            pitch_number=1,
            inning_topbot="bot",
            batter="b2",
            batter_name="Home Batter",
            events="home_run",
            outs_before=0,
            post_outs=0,
            home_score=0,
            away_score=0,
            post_home_score=1,
            post_away_score=0,
        )
    )
    _write_rows(source, rows)
    with DuckDBStore() as store:
        report = import_kbo_playbyplay(store, [source], revision="test-revision")
        assert report.invalid_score_transitions == 1
        assert report.unreconciled_score_games == 1
        assert report.source_unallocated_runs == -1
        assert report.source_sequence_gaps == 1
        transitions = store.connection.execute(
            """
            SELECT runners_before, runners_after, transition_complete
            FROM observed_plate_appearance ORDER BY sequence_in_game
            """
        ).fetchall()
        assert transitions == [("000", "000", False), ("000", "000", True)]


def test_import_rejects_a_parquet_file_with_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "kbo_pbp_2023.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT 'game' AS game_pk, '2023-04-01' AS game_date) TO ? (FORMAT PARQUET)",
        [str(source)],
    )
    connection.close()

    with DuckDBStore() as store, pytest.raises(KBOIngestError, match="missing required"):
        import_kbo_playbyplay(store, [source], revision="test-revision")
