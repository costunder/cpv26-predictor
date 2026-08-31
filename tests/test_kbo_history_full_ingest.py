"""Historical player-box import tests using bounded, source-shaped records."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import cpv26.data.kbo_history_ingest as ingest_module
from cpv26.data import DuckDBStore
from cpv26.data.kbo_graph_dataset import build_kbo_graph_dataset
from cpv26.data.kbo_history_ingest import import_kbo_history
from cpv26.data.kbo_history_source import KBOHistoryArtifact

_KEY = "20010405_LGSK0"
_NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)
_IMPORTED_TABLES = (
    "source_revision", "team", "game", "team_game", "historical_boxscore",
    "historical_game_detail",
)


def _source_game() -> dict[str, Any]:
    # The first batter's notation follows the actual 2001 archive's schema;
    # other reduced rows exercise incomplete/zero/name-collision cases.
    return {
        "scoreboard": [
            {"팀": "LG", "승패": "패", "R": 0, "H": 3, "E": 0, "1": "0"},
            {"팀": "SK", "승패": "승", "R": 1, "H": 1, "E": 0, "1": "1"},
        ],
        "away_batter": [
            {
                "선수명": "이병규", "포지션": "중", "1": "투땅", "2": "2땅",
                "3": 0, "4": 0, "5": "중안", "6": "중안", "7": 0, "8": "우중안",
                "9": 0, "타수": 5, "안타": 3, "득점": 0, "타점": 0, "타율": 0.6,
                "추가원천필드": "preserve exactly",
            },
            {
                "선수명": "이병규", "포지션": "타", "1": "unknown-code", "타수": 1,
                "안타": 0, "득점": 0, "타점": 0, "타율": 0,
            },
            {
                "선수명": "대주자", "포지션": "주", "1": 0, "타수": 0,
                "안타": 0, "득점": 0, "타점": 0,
            },
            {"선수명": "데이터가 없습니다.", "1": "데이터가 없습니다."},
            "unparsed source row",
        ],
        "home_batter": [
            {"선수명": "이병규", "1": "좌안", "타수": 1, "안타": 1, "득점": 1, "타점": 1},
        ],
        "away_pitcher": [
            {
                "선수명": "투수", "등판": "선발", "이닝": "0", "타자": 1,
                "투구수": 4, "타수": 1, "피안타": 1, "홈런": 0, "4사구": 0,
                "삼진": 0, "실점": 1, "자책": 1, "승": "0", "패": "1",
                "세": "0", "평균자책점": "-", "원천추가값": None,
            },
        ],
        "home_pitcher": [
            {
                "선수명": "투수", "등판": "선발", "이닝": "1", "타자": 6,
                "투구수": 24, "타수": 6, "피안타": 3, "홈런": 0, "4사구": 0,
                "삼진": 0, "실점": 0, "자책": 0, "승": "1", "패": "0",
                "세": "0", "평균자책점": 0,
            },
        ],
        "ETC_info": {"관중": "12,345", "경기시간": "2:34", "unknown_meta": [0, None]},
        "new_source_top_field": {"nested": ["raw value", 0, None]},
    }


def _artifact(
    directory: Path, payload: Any, *, filename: str = "kbo_history_2001.json",
    format: str = "game_map",
) -> KBOHistoryArtifact:
    path = directory / filename
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    return KBOHistoryArtifact(
        year=2001, filename=filename, sha256=hashlib.sha256(data).hexdigest(),
        url=path.as_uri(), game_count=len(payload), format=format,
    )


def _boxes(store: DuckDBStore) -> list[dict[str, Any]]:
    cursor = store.connection.execute(
        "SELECT * FROM historical_boxscore ORDER BY role, side, row_index"
    )
    columns = [column[0] for column in cursor.description]
    records = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    for record in records:
        for field in ("stats_json", "raw_json", "quality_json"):
            record[field] = json.loads(record[field])
    return records


def _snapshot(store: DuckDBStore, tables: tuple[str, ...]) -> dict[str, Any]:
    return {
        table: store.connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        for table in tables
    }


def test_all_partial_rows_and_raw_source_fields_survive_import(tmp_path: Path) -> None:
    raw = _source_game()
    artifact = _artifact(tmp_path, {_KEY: raw})
    with DuckDBStore() as store:
        report = import_kbo_history(store, tmp_path, artifacts=(artifact,), ingested_at=_NOW)
        rows = _boxes(store)
        assert len(rows) == 8
        batting = [row for row in rows if row["role"] == "batting"]
        assert len(batting) == 6
        assert [row["raw_json"] for row in batting[:4]] == raw["away_batter"][:4]
        assert batting[4]["raw_json"] == {"unparsed_value": "unparsed source row"}
        assert batting[5]["raw_json"] == raw["home_batter"][0]
        assert batting[0]["stats_json"]["at_bats"] == 5
        assert batting[0]["stats_json"]["hits"] == 3
        assert batting[0]["stats_json"]["hits_verified"] is True
        assert batting[0]["stats_json"]["plate_appearances"] is None
        assert batting[1]["stats_json"]["hits"] == 0
        assert "unknown_inning_token:1:unknown-code" in batting[1]["quality_json"]
        assert batting[2]["stats_json"]["at_bats"] == 0
        assert batting[3]["display_name"] is None
        assert "player_display_name_missing" in batting[3]["quality_json"]
        assert "row_not_object" in batting[4]["quality_json"]
        assert batting[5]["stats_json"]["plate_appearances"] == 1
        assert batting[5]["stats_json"]["counts_verified"] is True
        assert rows[6]["stats_json"]["outs"] == 0
        assert rows[6]["stats_json"]["era"] is None
        assert rows[6]["raw_json"]["평균자책점"] == "-"
        assert rows[6]["stats_json"]["wins"] == 0
        assert all(row["event_at"] < row["available_at"] for row in rows)
        assert all(row["ingested_at"] == _NOW for row in rows)
        coverage = report["season_coverage"][0]
        assert coverage["batter_rows"] == 6 and coverage["pitcher_rows"] == 2
        assert coverage["known_pa_hit_labels"] == 1
        assert coverage["partial_pa_hit_labels"] == 2
        assert coverage["verified_batting_outcomes"] == 1
        store.assert_referential_integrity()
        store.assert_composite_referential_integrity()


def test_game_detail_preserves_all_non_player_source_fields(tmp_path: Path) -> None:
    raw = _source_game()
    artifact = _artifact(tmp_path, {_KEY: raw})
    with DuckDBStore() as store:
        import_kbo_history(store, tmp_path, artifacts=(artifact,))
        metadata = json.loads(store.connection.execute(
            "SELECT metadata_json FROM historical_game_detail"
        ).fetchone()[0])
        assert metadata["ETC_info"] == raw["ETC_info"]
        assert metadata["new_source_top_field"] == raw["new_source_top_field"]
        assert metadata["scoreboard"] == raw["scoreboard"]
        assert set(metadata) == {"ETC_info", "scoreboard", "new_source_top_field"}


def test_same_names_remain_distinct_source_observations_not_canonical_players(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, {_KEY: _source_game()})
    with DuckDBStore() as store:
        import_kbo_history(store, tmp_path, artifacts=(artifact,))
        rows = _boxes(store)
        same_name = [row for row in rows if row["display_name"] == "이병규"]
        assert len(same_name) == 3
        assert len({row["observation_id"] for row in same_name}) == 3
        assert len({row["player_id"] for row in same_name}) == 3
        assert {row["team_id"] for row in same_name} == {"kbo-team:LG", "kbo-team:SK"}
        assert all(row["identity_status"] == "source_observation" for row in rows)
        assert all(row["player_id"].startswith("kbo-box-observation:") for row in rows)
        assert len({row["player_id"] for row in rows}) == len(rows)
        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (0,)
        assert store.connection.execute(
            "SELECT count(*) FROM observed_plate_appearance"
        ).fetchone() == (0,)


def test_full_player_import_is_idempotent_without_changing_provenance(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, {_KEY: _source_game()})
    with DuckDBStore() as store:
        import_kbo_history(store, tmp_path, artifacts=(artifact,), ingested_at=_NOW)
        before = _snapshot(store, _IMPORTED_TABLES)
        retry = import_kbo_history(
            store, tmp_path, artifacts=(artifact,), ingested_at=_NOW + timedelta(days=1),
        )
        assert all(count == 0 for count in retry["inserted_rows"].values())
        assert _snapshot(store, _IMPORTED_TABLES) == before


def test_equal_full_payload_duplicates_are_imported_once(tmp_path: Path) -> None:
    game = {"id": _KEY, "contents": _source_game()}
    artifact = _artifact(tmp_path, [game, copy.deepcopy(game)], format="game_list")
    with DuckDBStore() as store:
        report = import_kbo_history(store, tmp_path, artifacts=(artifact,))
        assert report["season_coverage"][0]["duplicate_records"] == 1
        assert report["season_coverage"][0]["batter_rows"] == 6
        assert len(_boxes(store)) == 8
        assert store.connection.execute(
            "SELECT count(*) FROM historical_game_detail"
        ).fetchone() == (1,)


@pytest.mark.parametrize("different_file", [False, True])
def test_equal_scores_but_conflicting_raw_boxes_roll_back_every_table(
    tmp_path: Path, different_file: bool,
) -> None:
    original = _source_game()
    changed = copy.deepcopy(original)
    # Finals still match; even an unnormalized source-field conflict is not ignored.
    changed["away_batter"][0]["추가원천필드"] = "different raw evidence"
    if different_file:
        artifacts = (
            _artifact(tmp_path, {_KEY: original}, filename="history_2001_a.json"),
            _artifact(tmp_path, {_KEY: changed}, filename="history_2001_b.json"),
        )
    else:
        artifacts = (_artifact(tmp_path, [
            {"id": _KEY, "contents": original}, {"id": _KEY, "contents": changed},
        ], format="game_list"),)
    with DuckDBStore() as store:
        with pytest.raises(ValueError, match="conflicting historical box-score payloads"):
            import_kbo_history(store, tmp_path, artifacts=artifacts)
        assert _snapshot(store, _IMPORTED_TABLES) == dict.fromkeys(_IMPORTED_TABLES, [])


def test_existing_core_v1_import_is_upgraded_without_rewriting_core_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path, {_KEY: _source_game()})
    core_tables = ("source_revision", "team", "game", "team_game")
    with DuckDBStore() as store:
        # Reproduce the original score-only adapter's writes, using its unchanged
        # v1 IDs/metadata, without its newly added second-pass boxscore adapter.
        with monkeypatch.context() as patch:
            patch.setattr(ingest_module, "_import_boxscores", lambda *_args, **_kwargs: {})
            import_kbo_history(store, tmp_path, artifacts=(artifact,), ingested_at=_NOW)
        original = _snapshot(store, core_tables)
        assert len(original["source_revision"]) == 1
        assert store.connection.execute(
            "SELECT count(*) FROM historical_boxscore"
        ).fetchone() == (0,)
        upgraded = import_kbo_history(
            store, tmp_path, artifacts=(artifact,), ingested_at=_NOW + timedelta(days=1),
        )
        assert upgraded["inserted_rows"]["historical_boxscore"] == 8
        assert upgraded["inserted_rows"]["historical_game_detail"] == 1
        assert upgraded["inserted_rows"]["source_revision"] == 1
        for table in ("team", "game", "team_game"):
            assert upgraded["inserted_rows"][table] == 0
            assert _snapshot(store, (table,))[table] == original[table]
        source_rows = _snapshot(store, ("source_revision",))["source_revision"]
        assert original["source_revision"][0] in source_rows
        assert {json.loads(row[4])["adapter_version"] for row in source_rows} == {1, 2}
        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (0,)


def test_full_import_graph_has_partial_hit_targets_without_fabricated_pa_events(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, {_KEY: _source_game()})
    database = tmp_path / "canonical.duckdb"
    with DuckDBStore(database) as store:
        import_kbo_history(store, tmp_path, artifacts=(artifact,), ingested_at=_NOW)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    graph = dataset.load_day("2001-04-05")
    assert graph.arrays["match_targets"].size == 1
    assert graph.arrays["pa_targets"].size == 0
    assert graph.arrays["pa_context"].shape == (0, 10)
    assert sorted(graph.arrays["live_hit_pa"].tolist()) == [-1, -1, 1]
    assert sorted(graph.arrays["live_hit_pa_min"].tolist()) == [1, 1, 5]
    assert sorted(graph.arrays["live_hit_hits"].tolist()) == [0, 1, 3]
    assert graph.arrays["box_pa_counts"].shape == (1, 10)
    assert graph.arrays["box_pa_counts"].sum() == 1
    assert graph.arrays["box_pitch_targets"].shape == (2, 10)
    assert graph.arrays["player_box_batting_features"].sum() == 0
    assert graph.arrays["player_box_pitching_features"].sum() == 0


def test_zero_pa_and_interference_rows_keep_distinct_opportunity_semantics(tmp_path: Path) -> None:
    raw = _source_game()
    zero = raw["away_batter"][2]
    interference = {
        "선수명": "타격방해", "1": "타방", "타수": 0, "안타": 0, "득점": 0, "타점": 0,
    }
    raw["away_batter"] = [raw["away_batter"][0], zero, interference]
    raw["home_pitcher"][0]["타수"] = 5
    artifact = _artifact(tmp_path, {_KEY: raw})
    database = tmp_path / "zero-and-interference.duckdb"
    with DuckDBStore(database) as store:
        import_kbo_history(store, tmp_path, artifacts=(artifact,))
        batting = [row for row in _boxes(store) if row["role"] == "batting"]
        assert batting[1]["stats_json"]["plate_appearances"] == 0
        assert batting[1]["stats_json"]["counts_verified"] is True
        assert batting[2]["stats_json"]["plate_appearances"] == 1
        assert batting[2]["stats_json"]["catcher_interference"] == 1
        assert sum(batting[2]["stats_json"]["outcome_counts"]) == 0
        assert store.connection.execute(
            "SELECT count(*) FROM observed_plate_appearance"
        ).fetchone() == (0,)
    graph = build_kbo_graph_dataset(database, tmp_path / "zero-graph").load_day("2001-04-05")
    assert sorted(graph.arrays["live_hit_pa"].tolist()) == [1, 1, 5]
    assert sorted(graph.arrays["live_hit_hits"].tolist()) == [0, 1, 3]
    assert graph.arrays["box_pa_counts"].shape == (2, 10)
    assert graph.arrays["box_pa_counts"].sum() == 6
