from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest

import cpv26.data.kbo_graph_dataset as graph_module
from cpv26.data.kbo_graph_dataset import KBOGraphDataset, build_kbo_graph_dataset

_KST = ZoneInfo("Asia/Seoul")


def _database(path: Path, *, season: int = 2023, include_pas: bool = True) -> None:
    connection = duckdb.connect(str(path))
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
    at = datetime(season, 4, 1, tzinfo=_KST)
    connection.execute(
        "INSERT INTO source_revision VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "source1",
            "local-test-fixture",
            "file://fixture",
            "a" * 64,
            json.dumps({"dataset_revision": "pinned-test", "adapter_version": 1}),
            at,
            at,
            at,
            at,
            None,
        ],
    )
    for number in (1, 2, 3):
        _game(connection, f"g{number}", number, season=season)
    if include_pas:
        _pa(connection, "pa1", "g1", 1, "old-batter", "old-pitcher", "single", season=season)
        _pa(connection, "pa2", "g2", 2, "new-batter", "new-pitcher", "home_run", season=season)
        _pa(connection, "pa3", "g2", 2, "old-batter", "old-pitcher", "strikeout", season=season)
        _pa(connection, "pa4", "g3", 3, "old-batter", "old-pitcher", "walk", season=season)
    connection.close()


def _game(
    connection: duckdb.DuckDBPyConnection, game_id: str, day: int, *, season: int = 2023
) -> None:
    start = datetime(season, 4, day, tzinfo=_KST)
    event = start + timedelta(hours=23, minutes=59, seconds=59)
    connection.execute(
        "INSERT INTO game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            game_id,
            start,
            "home",
            "away",
            "final",
            day,
            1,
            "source1",
            event,
            start + timedelta(days=1),
            start,
            event,
            None,
        ],
    )


def _pa(
    connection: duckdb.DuckDBPyConnection,
    pa_id: str,
    game_id: str,
    day: int,
    batter: str,
    pitcher: str,
    outcome: str,
    *,
    row_id: str | None = None,
    available_at: datetime | None = None,
    season: int = 2023,
) -> None:
    start = datetime(season, 4, day, tzinfo=_KST)
    event = start + timedelta(hours=23, minutes=59, seconds=59)
    hit = outcome in ("single", "double", "triple", "home_run")
    connection.execute(
        "INSERT INTO observed_plate_appearance VALUES (" + ",".join(["?"] * 23) + ")",
        [
            row_id or pa_id,
            pa_id,
            game_id,
            1,
            "top",
            batter,
            pitcher,
            "away",
            "home",
            0,
            0,
            0,
            "000",
            outcome,
            outcome not in ("walk", "hit_by_pitch"),
            hit,
            {"single": 1, "double": 2, "triple": 3, "home_run": 4}.get(outcome, 0),
            "source1",
            event,
            available_at or start + timedelta(days=1),
            start,
            event,
            None,
        ],
    )


def test_daily_cutoff_isolated_newcomers_and_safe_npz(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    assert dataset.days() == (date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3))
    first = dataset.load_day(date(2023, 4, 1))
    assert not first.node_features["player"].any()
    assert not first.node_features["team"].any()
    assert all(len(route["source_index"]) == 0 for route in first.routes.values())
    second = dataset.load_day("2023-04-02")
    assert second.day_id == "2023-04-02"
    assert second.role_features["batting"].shape == (4, 8)
    old = second.player_ids.index("old-batter")
    new = second.player_ids.index("new-batter")
    assert second.role_features["batting"][old, 2] == 1
    assert not second.role_features["batting"][new].any()
    assert second.match_targets.tolist() == [2]
    assert second.match_runs.tolist() == [[2, 1]]
    assert second.pa_targets.tolist() == [5, 0]
    assert second.live_hit_pa.tolist() == [1, 1]
    assert second.live_hit_hits.tolist() == [1, 0]
    route = second.routes["batter_pa_pitcher"]
    assert route["source_index"].tolist() == [old]
    assert route["destination_index"].tolist() == [second.player_ids.index("old-pitcher")]
    assert route["event_features"][0, 1] == 1
    for relation in second.routes.values():
        assert np.all(relation["event_age_seconds"] >= relation["publication_delay_seconds"])
    assert dataset.manifest["node_feature_dims"] == {"player": 4, "team": 8}
    assert dataset.manifest["source_provenance"][0]["metadata"]["adapter_version"] == 1
    for day in dataset.days():
        graph = dataset.load_day(day)
        for array in graph.arrays.values():
            assert not array.dtype.hasobject
            if np.issubdtype(array.dtype, np.number):
                assert np.isfinite(array).all()


@pytest.mark.parametrize("status,score", [("cancelled", 1), ("final", None)])
def test_game_correction_without_final_score_does_not_resurrect_prior_final(
    tmp_path: Path, status: str, score: int | None
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "INSERT INTO game SELECT game_id,scheduled_start,home_team_id,away_team_id,"
            "?, ?, away_score,source_revision_id,event_at,"
            "TIMESTAMPTZ '2023-04-02 12:00:00+09',ingested_at,"
            "TIMESTAMPTZ '2023-04-02 12:00:00+09',NULL FROM game WHERE game_id='g1'",
            [status, score],
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    assert date(2023, 4, 1) not in dataset.days()
    early = dataset.load_day("2023-04-02")
    later = dataset.load_day("2023-04-03")
    # Before publication the prior final was still available. Afterwards only
    # g2, not the withdrawn/cancelled g1 final, contributes to team game totals.
    assert early.team_features[early.team_ids.index("home"), 0] > 0
    assert later.team_features[later.team_ids.index("home"), 0] == pytest.approx(
        np.log1p(1) / np.log1p(90)
    )
    assert len(later.routes["home_team_game_away_team"]["source_index"]) == 1


def test_2001_game_only_graph_preserves_scores_without_fabricating_player_labels(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001, include_pas=False)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    assert dataset.days() == (date(2001, 4, 1), date(2001, 4, 2), date(2001, 4, 3))
    for day in dataset.days():
        graph = dataset.load_day(day)
        assert graph.player_ids == ()
        assert graph.node_features["player"].shape == (0, 4)
        assert graph.role_features["batting"].shape == (0, 8)
        assert graph.role_features["pitching"].shape == (0, 8)
        assert graph.pa_context.shape == (0, 10)
        assert graph.pa_targets.shape == (0,)
        assert graph.live_hit_pa.shape == (0,)
        assert graph.live_hit_hits.shape == (0,)
        assert len(graph.match_targets) == 1
        for name, route in graph.routes.items():
            if name != "home_team_game_away_team":
                assert len(route["source_index"]) == 0
    first = dataset.load_day("2001-04-01")
    assert not first.node_features["team"].any()
    assert not first.routes["home_team_game_away_team"]["source_index"].size
    second = dataset.load_day("2001-04-02")
    assert second.match_targets.tolist() == [2]
    assert second.match_runs.tolist() == [[2, 1]]
    home = second.team_ids.index("home")
    np.testing.assert_allclose(second.node_features["team"][home, 1:6], [0, 1, 0.1, 0.1, 0])
    assert len(second.routes["home_team_game_away_team"]["source_index"]) == 1
    expected_archive = {
        "box_batting_rows": 0,
        "box_pitching_rows": 0,
        "box_pa_queries": 0,
        "box_pitch_queries": 0,
        "box_live_hit_queries": 0,
        "box_live_hit_unknown_pa_queries": 0,
        "box_pa_outcomes": 0,
        "box_pitch_observed_counts": 0,
        "box_target_missing_reasons": {},
    }
    assert dataset.manifest["season_coverage"] == [
        {
            "season": 2001,
            "days": 3,
            "date_start": "2001-04-01",
            "date_end": "2001-04-03",
            "games": 3,
            "games_with_pa": 0,
            "games_with_boxscore": 0,
            "boxscore_only_games": 0,
            "game_only_games": 3,
            "observed_completed_pa": 0,
            "live_hit_queries": 0,
            "pa_queries": 0,
            "pa_derived_batting_queries": 0,
            "pa_derived_pitching_queries": 0,
            "live_hit_unknown_pa_queries": 0,
            "raw_archive_boxscore": expected_archive,
            "box_batting_rows": 0,
            "box_pitching_rows": 0,
            "box_pa_queries": 0,
            "box_pitch_queries": 0,
            "box_live_hit_queries": 0,
            "box_live_hit_unknown_pa_queries": 0,
            "box_pa_outcomes": 0,
            "box_pitch_observed_counts": 0,
            "box_target_missing_reasons": {},
        }
    ]


def test_game_only_same_day_scores_and_future_pa_do_not_enter_historical_features(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001, include_pas=False)
    before = build_kbo_graph_dataset(database, tmp_path / "before").load_day("2001-04-02")
    connection = duckdb.connect(str(database))
    connection.execute("UPDATE game SET home_score = 30 WHERE game_id IN ('g2', 'g3')")
    _game(connection, "g2-doubleheader", 2, season=2001)
    _game(connection, "g2023", 1)
    _pa(connection, "future-pa", "g2023", 1, "future-batter", "future-pitcher", "double")
    connection.close()
    dataset = build_kbo_graph_dataset(database, tmp_path / "after")
    after = dataset.load_day("2001-04-02")
    assert before.player_ids == after.player_ids == ()
    assert before.team_ids == after.team_ids
    for key in before.arrays:
        if key.endswith("features") or "__" in key:
            np.testing.assert_array_equal(before.arrays[key], after.arrays[key])
    assert after.match_runs.tolist() == [[30, 1], [2, 1]]
    early, recent = dataset.manifest["season_coverage"]
    assert early["season"] == 2001
    assert early["games"] == early["game_only_games"] == 4
    assert early["games_with_pa"] == early["observed_completed_pa"] == 0
    assert recent["season"] == 2023
    assert recent["games"] == recent["games_with_pa"] == 1
    assert recent["game_only_games"] == 0
    assert recent["observed_completed_pa"] == recent["pa_queries"] == 1
    assert recent["live_hit_queries"] == 1


def test_same_day_results_and_lineup_do_not_change_graph_features(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    before = build_kbo_graph_dataset(database, tmp_path / "before").load_day("2023-04-02")
    connection = duckdb.connect(str(database))
    connection.execute("UPDATE game SET home_score = 30 WHERE game_id = 'g2'")
    connection.execute("""
        UPDATE observed_plate_appearance SET is_hit = false, total_bases = 0,
            outcome = 'strikeout', home_score_before = 19 WHERE game_id = 'g2'
    """)
    # A second same-day game must not turn the first game's actual lineup into history.
    _game(connection, "g2-doubleheader", 2)
    _pa(connection, "pa5", "g2-doubleheader", 2, "new-batter", "new-pitcher", "single")
    connection.close()
    after = build_kbo_graph_dataset(database, tmp_path / "after").load_day("2023-04-02")
    assert before.player_ids == after.player_ids
    assert before.team_ids == after.team_ids
    for key in before.arrays:
        if key.endswith("features") or "__" in key:
            np.testing.assert_array_equal(before.arrays[key], after.arrays[key])
    assert before.live_hit_hits.tolist() != after.live_hit_hits.tolist()


def test_vnext_adds_game_resolution_without_current_player_game_leakage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    legacy = build_kbo_graph_dataset(database, tmp_path / "legacy")
    dataset = build_kbo_graph_dataset(
        database, tmp_path / "vnext", graph_schema="vnext"
    )

    assert legacy.manifest["dataset_version"] == 5
    assert "graph_schema" not in legacy.manifest
    assert legacy.manifest["node_feature_dims"] == {"player": 4, "team": 8}
    assert dataset.manifest["dataset_version"] == 6
    assert dataset.manifest["graph_schema"] == "vnext"
    assert dataset.manifest["node_feature_dims"] == {"player": 4, "team": 8, "game": 4}

    second = dataset.load_day("2023-04-02")
    assert set(second.game_ids) == {"g1", "g2"}
    assert second.node_features["game"][second.game_ids.index("g1")].tolist()[:2] == [0, 1]
    assert second.node_features["game"][second.game_ids.index("g2")].tolist()[:2] == [1, 0]
    assert second.match_game_index.tolist() == [second.game_ids.index("g2")]

    for route_name in ("batter_game_participation", "pitcher_game_participation"):
        route = second.routes[route_name]
        destinations = {second.game_ids[index] for index in route["destination_index"]}
        assert destinations == {"g1"}
    context = second.routes["team_game_context"]
    assert len(context["source_index"]) == 4
    current = context["event_features"][:, 0].astype(bool)
    assert current.sum() == 2
    assert set(context["event_age_seconds"][current]) == {0}
    for route in second.routes.values():
        assert np.all(route["event_age_seconds"] >= route["publication_delay_seconds"])

    third = dataset.load_day("2023-04-03")
    route = third.routes["batter_game_participation"]
    old = third.player_ids.index("old-batter")
    old_games = {
        third.game_ids[destination]
        for source, destination in zip(
            route["source_index"], route["destination_index"], strict=True
        )
        if source == old
    }
    assert old_games == {"g1", "g2"}


def test_vnext_inputs_are_invariant_to_current_day_results(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    before = build_kbo_graph_dataset(
        database, tmp_path / "before-vnext", graph_schema="vnext"
    ).load_day("2023-04-02")
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE game SET home_score = 30 WHERE game_id = 'g2'")
        connection.execute("""
            UPDATE observed_plate_appearance SET is_hit = false, total_bases = 0,
                outcome = 'strikeout', home_score_before = 19 WHERE game_id = 'g2'
        """)
    after = build_kbo_graph_dataset(
        database, tmp_path / "after-vnext", graph_schema="vnext"
    ).load_day("2023-04-02")

    assert before.player_ids == after.player_ids
    assert before.team_ids == after.team_ids
    assert before.game_ids == after.game_ids
    for key in before.arrays:
        if key.endswith("features") or "__" in key or key.endswith("_game_index"):
            np.testing.assert_array_equal(before.arrays[key], after.arrays[key])
    assert before.match_runs.tolist() != after.match_runs.tolist()
    assert before.live_hit_hits.tolist() != after.live_hit_hits.tolist()


def test_vnext_current_doubleheaders_have_distinct_known_time_features(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "g2-doubleheader", 2)
        connection.execute("""
            UPDATE game
            SET scheduled_start = TIMESTAMPTZ '2023-04-02 18:30:00+09'
            WHERE game_id = 'g2-doubleheader'
        """)
    graph = build_kbo_graph_dataset(
        database, tmp_path / "vnext-doubleheader", graph_schema="vnext"
    ).load_day("2023-04-02")

    assert set(graph.match_query_ids.tolist()) == {"g2", "g2-doubleheader"}
    by_game = {
        game_id: graph.game_features[graph.game_ids.index(game_id), -1]
        for game_id in ("g2", "g2-doubleheader")
    }
    assert by_game["g2"] == 0
    assert by_game["g2-doubleheader"] == pytest.approx(18.5 / 24)
    assert len(set(graph.match_game_index.tolist())) == 2


def test_vnext_manifest_contract_is_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    directory = tmp_path / "vnext"
    build_kbo_graph_dataset(database, directory, graph_schema="vnext")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("graph_schema")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="graph_schema=vnext"):
        KBOGraphDataset(directory)


def test_window_expiry_and_range_filter_keep_original_history(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    dataset = build_kbo_graph_dataset(
        database,
        tmp_path / "graph",
        rolling_days=1,
        start_day="2023-04-03",
        end_day="2023-04-03",
    )
    graph = dataset.load_day("2023-04-03")
    assert dataset.days() == (date(2023, 4, 3),)
    old = graph.player_ids.index("old-batter")
    # Apr 1 hit expired; Apr 2 strikeout remains despite start_day filtering.
    assert graph.role_features["batting"][old, 2] == 0
    assert graph.role_features["batting"][old, 5] == 1
    assert graph.node_features["team"][graph.team_ids.index("home"), 3] == pytest.approx(0.2)


def test_publication_and_revision_validity_cutoffs(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    connection = duckdb.connect(str(database))
    _pa(
        connection,
        "pa1",
        "g1",
        1,
        "old-batter",
        "old-pitcher",
        "strikeout",
        row_id="pa1-correction",
        available_at=datetime(2023, 4, 2, 12, tzinfo=_KST),
    )
    connection.execute("""
        UPDATE observed_plate_appearance SET valid_from = TIMESTAMPTZ '2023-04-02 12:00:00+09'
        WHERE observed_pa_row_id = 'pa1-correction'
    """)
    connection.execute("""
        UPDATE observed_plate_appearance SET valid_to = TIMESTAMPTZ '2023-04-02 12:00:00+09'
        WHERE observed_pa_row_id = 'pa1'
    """)
    connection.close()
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    second, third = dataset.load_day("2023-04-02"), dataset.load_day("2023-04-03")
    assert second.role_features["batting"][second.player_ids.index("old-batter"), 2] == 1
    assert third.role_features["batting"][third.player_ids.index("old-batter"), 2] == 0
    # Latest snapshot targets can be corrected, but correction cannot enter Apr 2 features.
    assert dataset.load_day("2023-04-01").live_hit_hits.tolist() == [0]


def test_cache_incremental_reuse_integrity_and_corruption_rebuild(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "graph"
    first = build_kbo_graph_dataset(database, output, end_day="2023-04-02")
    fingerprint = first.manifest["fingerprint"]
    cached_file = output / first.manifest["days"][0]["file"]
    original_time = cached_file.stat().st_mtime_ns
    same = build_kbo_graph_dataset(database, output, end_day="2023-04-02")
    assert same.manifest["fingerprint"] == fingerprint
    assert same.manifest["cache_reused_days"] == 2
    assert cached_file.stat().st_mtime_ns == original_time
    extended = build_kbo_graph_dataset(database, output)
    assert extended.manifest["cache_reused_days"] == 2
    assert extended.manifest["cache_built_days"] == 1
    # Tests intentionally corrupt only their own temporary artifact.
    cached_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        extended.load_day("2023-04-01")
    repaired = build_kbo_graph_dataset(database, output)
    assert repaired.manifest["cache_reused_days"] == 2
    assert repaired.manifest["cache_built_days"] == 1
    assert KBOGraphDataset(output).load_day("2023-04-01").day == date(2023, 4, 1)
    assert not list(output.rglob("*.part"))


def test_coverage_metadata_upgrade_reuses_legacy_graph_arrays_and_fingerprint(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "graph"
    first = build_kbo_graph_dataset(database, output)
    paths = [output / entry["file"] for entry in first.manifest["days"]]
    original_times = [path.stat().st_mtime_ns for path in paths]
    for path in paths:
        sidecar = path.with_suffix(".json")
        entry = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in ("games_with_pa", "game_only_games", "observed_completed_pa"):
            entry.pop(key)
        sidecar.write_text(json.dumps(entry), encoding="utf-8")
    updated = build_kbo_graph_dataset(database, output)
    assert updated.manifest["fingerprint"] == first.manifest["fingerprint"]
    assert updated.manifest["cache_reused_days"] == 3
    assert updated.manifest["cache_built_days"] == 0
    assert [path.stat().st_mtime_ns for path in paths] == original_times
    assert updated.manifest["season_coverage"] == first.manifest["season_coverage"]
    assert updated.manifest["season_coverage"][0]["games_with_pa"] == 3
    assert updated.manifest["season_coverage"][0]["game_only_games"] == 0
    assert updated.manifest["season_coverage"][0]["observed_completed_pa"] == 4
    assert updated.manifest["days"] == first.manifest["days"]


def test_reject_bad_options_and_unknown_day(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with pytest.raises(ValueError, match="positive"):
        build_kbo_graph_dataset(database, tmp_path / "graph", rolling_days=0)
    with pytest.raises(ValueError, match="timezone"):
        build_kbo_graph_dataset(database, tmp_path / "graph", knowledge_at=datetime(2023, 4, 1))
    with pytest.raises(ValueError, match="after"):
        build_kbo_graph_dataset(
            database, tmp_path / "graph", start_day="2024-01-01", end_day="2023-01-01"
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    with pytest.raises(KeyError, match="not present"):
        dataset.load_day("2025-01-01")


def test_appended_future_games_reuse_earlier_graphs(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "graph"
    first = build_kbo_graph_dataset(database, output)
    connection = duckdb.connect(str(database))
    _game(connection, "g4", 4)
    _pa(connection, "pa5", "g4", 4, "future-batter", "future-pitcher", "double")
    connection.close()
    second = build_kbo_graph_dataset(database, output)
    assert second.manifest["cache_reused_days"] == 3
    assert second.manifest["cache_built_days"] == 1
    assert second.manifest["fingerprint"] != first.manifest["fingerprint"]
    assert second.manifest["days"][:3] == first.manifest["days"]
    assert "future-batter" not in second.load_day("2023-04-03").player_ids


def test_knowledge_snapshot_and_conditional_pa10_exclusions(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    connection = duckdb.connect(str(database))
    _pa(connection, "interference", "g1", 1, "rare-batter", "old-pitcher", "catcher_interference")
    connection.close()
    dataset = build_kbo_graph_dataset(
        database,
        tmp_path / "graph",
        knowledge_at=datetime(2023, 4, 2, tzinfo=_KST),
    )
    assert dataset.days() == (date(2023, 4, 1),)
    graph = dataset.load_day("2023-04-01")
    assert graph.live_hit_pa.tolist() == [1, 1]
    assert len(graph.pa_targets) == 1
    assert dataset.manifest["label_quality"]["pa10_excluded_catcher_interference"] == 1
    assert dataset.manifest["label_quality"]["unlabelled_source_pa"] is None


def test_incomplete_transition_masks_only_pa_score_context(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    connection = duckdb.connect(str(database))
    connection.execute(
        "ALTER TABLE observed_plate_appearance ADD COLUMN transition_complete BOOLEAN DEFAULT true"
    )
    connection.execute("""
        UPDATE observed_plate_appearance SET home_score_before = 7, away_score_before = 3
        WHERE observed_pa_row_id = 'pa2'
    """)
    connection.close()
    before = build_kbo_graph_dataset(database, tmp_path / "before")
    connection = duckdb.connect(str(database))
    connection.execute("""
        UPDATE observed_plate_appearance SET transition_complete = false
        WHERE observed_pa_row_id = 'pa2'
    """)
    connection.close()
    after = build_kbo_graph_dataset(database, tmp_path / "after")
    assert after.manifest["dataset_version"] == 5
    assert after.manifest["pa_incomplete_transition_context"] == "mask_pre_scores_unknown"
    assert after.manifest["fingerprint"] != before.manifest["fingerprint"]
    quality = after.manifest["label_quality"]
    assert quality["incomplete_pa_transitions"] == 1
    assert quality["pa_context_scores_masked_incomplete_transition"] == 1
    assert (
        quality["observed_completed_pa"]
        == before.manifest["label_quality"]["observed_completed_pa"]
    )
    for day in after.days():
        prior, masked = before.load_day(day), after.load_day(day)
        assert prior.player_ids == masked.player_ids
        assert prior.team_ids == masked.team_ids
        # Includes WDL, LiveHit, PA labels, all nodes, and next-day historical edges.
        for name in prior.arrays:
            if name != "pa_context":
                np.testing.assert_array_equal(prior.arrays[name], masked.arrays[name])
    prior, masked = before.load_day("2023-04-02"), after.load_day("2023-04-02")
    index = list(masked.pa_query_ids).index("pa2")
    np.testing.assert_allclose(prior.pa_context[index, 6:], [0.7, 0.3, 0, 0])
    np.testing.assert_array_equal(masked.pa_context[index, 6:], [0, 0, 1, 1])
    np.testing.assert_array_equal(prior.pa_context[index, :6], masked.pa_context[index, :6])
    other = list(masked.pa_query_ids).index("pa3")
    np.testing.assert_array_equal(prior.pa_context[other], masked.pa_context[other])


def test_dataset_policy_version_invalidates_cached_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "graph"
    monkeypatch.setattr(graph_module, "GRAPH_DATASET_VERSION", 1)
    prior = build_kbo_graph_dataset(database, output)
    monkeypatch.setattr(graph_module, "GRAPH_DATASET_VERSION", 2)
    with pytest.raises(ValueError, match="unsupported"):
        KBOGraphDataset(output)
    current = build_kbo_graph_dataset(database, output)
    assert current.manifest["cache_reused_days"] == 0
    assert current.manifest["cache_built_days"] == 3
    assert current.manifest["config_fingerprint"] != prior.manifest["config_fingerprint"]
    assert current.manifest["fingerprint"] != prior.manifest["fingerprint"]
