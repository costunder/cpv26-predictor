from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cpv26.data import DuckDBStore
from cpv26.data.kbo_history_ingest import import_kbo_history, parse_historical_game
from cpv26.data.kbo_history_source import KBOHistoryArtifact


def _contents(away_score: int = 6, home_score: int = 11) -> dict[str, Any]:
    return {
        "scoreboard": [
            {"팀": "LG", "승패": "패", "R": away_score, "H": 13, "E": 0},
            {"팀": "SK", "승패": "승", "R": home_score, "H": 16, "E": 1},
        ]
    }


def _artifact(
    tmp_path: Path, payload: Any = None, *, format: str = "game_map"
) -> KBOHistoryArtifact:
    path = tmp_path / "kbo_history_2001.json"
    if payload is None:
        payload = {"20010405_LGSK0": _contents()}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(encoded)
    return KBOHistoryArtifact(
        year=2001,
        filename=path.name,
        sha256=hashlib.sha256(encoded).hexdigest(),
        url=path.as_uri(),
        game_count=len(payload),
        format=format,
    )


def test_history_import_is_atomic_idempotent_and_does_not_create_player_events(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    now = datetime(2026, 8, 31, tzinfo=timezone.utc)
    with DuckDBStore(tmp_path / "canonical.duckdb") as store:
        report = import_kbo_history(store, tmp_path, artifacts=(artifact,), ingested_at=now)
        assert report["games"] == 1
        assert report["season_coverage"][0]["year"] == 2001
        assert report["inserted_rows"] == {
            "source_revision": 2,
            "team": 2,
            "game": 1,
            "team_game": 2,
            "historical_boxscore": 0,
            "historical_game_detail": 1,
        }
        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (0,)
        assert store.connection.execute(
            "SELECT count(*) FROM observed_plate_appearance"
        ).fetchone() == (0,)
        row = store.connection.execute(
            "SELECT game_id, season, home_team_id, away_team_id, home_score, away_score, "
            "available_at, ingested_at, scheduled_start FROM game"
        ).fetchone()
        assert row[:6] == ("kbo-game:20010405LGSK02001", 2001, "kbo-team:SK", "kbo-team:LG", 11, 6)
        assert row[6] == datetime(2001, 4, 5, 15, tzinfo=timezone.utc)
        assert row[7] == now
        assert row[8] == datetime(2001, 4, 4, 15, tzinfo=timezone.utc)
        assert store.connection.execute(
            "SELECT result FROM team_game ORDER BY is_home"
        ).fetchall() == [
            ("loss",),
            ("win",),
        ]
        retried = import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert all(value == 0 for value in retried["inserted_rows"].values())
        store.assert_composite_referential_integrity()


def test_exact_archive_duplicates_are_counted_but_not_repeated(tmp_path: Path) -> None:
    row = {"id": "20010405_LGSK0", "contents": _contents()}
    artifact = _artifact(tmp_path, [row, row], format="game_list")
    with DuckDBStore() as store:
        report = import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert report["games"] == 1
        assert report["season_coverage"][0]["duplicate_records"] == 1


def test_conflicting_duplicates_fail_before_any_rows_are_written(tmp_path: Path) -> None:
    rows = [
        {"id": "20010405_LGSK0", "contents": _contents()},
        {"id": "20010405_LGSK0", "contents": _contents(home_score=12)},
    ]
    artifact = _artifact(tmp_path, rows, format="game_list")
    with DuckDBStore() as store:
        with pytest.raises(ValueError, match="conflicting"):
            import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (0,)


@pytest.mark.parametrize("failure", ["checksum", "count", "year", "missing"])
def test_bad_archives_leave_database_unchanged(tmp_path: Path, failure: str) -> None:
    artifact = _artifact(tmp_path)
    if failure == "checksum":
        artifact = replace(artifact, sha256="0" * 64)
    elif failure == "count":
        artifact = replace(artifact, game_count=2)
    elif failure == "year":
        artifact = replace(artifact, year=2002)
    else:
        (tmp_path / artifact.filename).unlink()
    with DuckDBStore() as store:
        with pytest.raises((ValueError, FileNotFoundError)):
            import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert store.connection.execute("SELECT count(*) FROM game").fetchone() == (0,)


@pytest.mark.parametrize("change", ["score", "result", "team", "order", "sides", "bool"])
def test_rejects_invalid_or_unfinished_scoreboards(change: str) -> None:
    contents = _contents()
    board = contents["scoreboard"]
    if change == "score":
        board[0]["R"] = "-"
    elif change == "result":
        board[0]["승패"] = "승"
    elif change == "team":
        board[0]["팀"] = "롯데"
    elif change == "order":
        board.reverse()
    elif change == "sides":
        board.pop()
    else:
        board[0]["R"] = True
    with pytest.raises(ValueError):
        parse_historical_game("20010405_LGSK0", contents, 2001)


def test_blank_archived_heroes_display_name_keeps_original_game_identity() -> None:
    contents = _contents()
    contents["scoreboard"][0]["팀"] = ""
    game = parse_historical_game("20090405_WOSK0", contents, 2009)
    assert game.away == "WO"
    assert game.home == "SK"
    assert game.away_score == 6
    assert game.home_score == 11


def test_tie_breaker_is_retained_with_its_actual_game_type(tmp_path: Path) -> None:
    game = {
        "scoreboard": [
            {"팀": "KT", "승패": "승", "R": 1},
            {"팀": "삼성", "승패": "패", "R": 0},
        ]
    }
    regular = {
        "scoreboard": [
            {"팀": "LG", "승패": "무", "R": 0},
            {"팀": "SSG", "승패": "무", "R": 0},
        ]
    }
    artifact = _artifact(tmp_path, {"20211030_LGSK0": regular, "20211031_KTSS0": game})
    artifact = replace(artifact, year=2021)
    with DuckDBStore() as store:
        report = import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert report["games"] == 2
        assert report["season_coverage"][0]["regular_games"] == 1
        assert report["season_coverage"][0]["non_regular_games"] == 1
        assert report["files"][0]["retained_non_regular_game_ids"] == ["20211031_KTSS0"]
        assert store.connection.execute(
            "SELECT game_type FROM game WHERE game_id='kbo-game:20211031KTSS02021'"
        ).fetchone() == ("tiebreaker",)


def test_existing_score_conflict_rolls_back_the_entire_import(tmp_path: Path) -> None:
    first = _artifact(tmp_path)
    with DuckDBStore() as store:
        import_kbo_history(store, tmp_path, artifacts=(first,))
        changed = _artifact(tmp_path, {"20010405_LGSK0": _contents(home_score=12)})
        with pytest.raises(ValueError, match="existing canonical score conflict"):
            import_kbo_history(store, tmp_path, artifacts=(changed,))
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (2,)
        assert store.connection.execute("SELECT home_score FROM game").fetchone() == (11,)


def test_duplicate_map_keys_are_not_silently_lost(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    text = '{"20010405_LGSK0": {}, "20010405_LGSK0": {}}'
    path = tmp_path / artifact.filename
    path.write_text(text, encoding="utf-8")
    artifact = replace(artifact, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with DuckDBStore() as store, pytest.raises(ValueError, match="duplicate JSON object key"):
        import_kbo_history(store, tmp_path, artifacts=(artifact,))
