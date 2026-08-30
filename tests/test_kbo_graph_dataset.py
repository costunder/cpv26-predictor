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


def _database(path: Path) -> None:
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
    at = datetime(2023, 4, 1, tzinfo=_KST)
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
        _game(connection, f"g{number}", number)
    _pa(connection, "pa1", "g1", 1, "old-batter", "old-pitcher", "single")
    _pa(connection, "pa2", "g2", 2, "new-batter", "new-pitcher", "home_run")
    _pa(connection, "pa3", "g2", 2, "old-batter", "old-pitcher", "strikeout")
    _pa(connection, "pa4", "g3", 3, "old-batter", "old-pitcher", "walk")
    connection.close()


def _game(connection: duckdb.DuckDBPyConnection, game_id: str, day: int) -> None:
    start = datetime(2023, 4, day, tzinfo=_KST)
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
) -> None:
    start = datetime(2023, 4, day, tzinfo=_KST)
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
    assert dataset.days() == (date(2023, 4, 1), date(2023, 4, 2))
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
    assert after.manifest["dataset_version"] == 2
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
