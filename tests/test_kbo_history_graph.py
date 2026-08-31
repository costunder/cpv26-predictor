from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np

import cpv26.data.kbo_graph_dataset as graph_module
from cpv26.data.kbo_graph_dataset import KBOGraphDataset, build_kbo_graph_dataset
from cpv26.data.kbo_player_identity import historical_player_prior
from cpv26.data.schema_v5 import V5_DDL
from test_kbo_graph_dataset import _database

_KST = ZoneInfo("Asia/Seoul")


def _batting(*, known_pa: bool = True, ab: int = 3, hits: int = 1) -> dict[str, Any]:
    return {
        "at_bats": ab,
        "hits": hits,
        "runs": 0,
        "rbi": 1,
        "plate_appearances": 4 if known_pa else None,
        "outcome_counts": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        "counts_verified": known_pa,
        "hits_verified": True,
    }


def _pitching(*, partial: bool = False) -> dict[str, Any]:
    return {
        "batters_faced": 4,
        "outs": None if partial else 3,
        "pitches": None if partial else 12,
        "at_bats": 3,
        "hits": 1,
        "home_runs": 0,
        "walks_hbp": 1,
        "strikeouts": 1,
        "runs": 0,
        "earned_runs": 0,
    }


def _box(
    connection: duckdb.DuckDBPyConnection,
    observation: str,
    day: int,
    role: str,
    stats: dict[str, Any],
    *,
    available_at: datetime | None = None,
) -> None:
    start = datetime(2001, 4, day, tzinfo=_KST)
    event = start + timedelta(hours=23, minutes=59, seconds=59)
    team, opponent = ("away", "home") if role == "batting" else ("home", "away")
    row = {
        "boxscore_row_id": observation,
        "observation_id": observation,
        "game_id": f"g{day}",
        "team_game_id": f"g{day}:{team}",
        "team_id": team,
        "opponent_team_id": opponent,
        "role": role,
        "side": "away" if role == "batting" else "home",
        "player_id": f"observed:{observation}",
        "identity_status": "source_observation",
        "display_name": "동명이인",
        "row_index": 0,
        "stats_json": json.dumps(stats),
        "raw_json": "{}",
        "quality_json": json.dumps({"reasons": ["identity_unresolved"]}),
        "source_revision_id": "source1",
        "event_at": event,
        "available_at": available_at or start + timedelta(days=1),
        "ingested_at": start,
        "valid_from": event,
        "valid_to": None,
    }
    connection.execute(
        "INSERT INTO historical_boxscore ("
        + ",".join(row)
        + ") VALUES ("
        + ",".join("?" for _ in row)
        + ")",
        list(row.values()),
    )


def _historical_database(path: Path) -> None:
    _database(path, season=2001, include_pas=False)
    with duckdb.connect(str(path)) as connection:
        connection.execute(V5_DDL[0])
        _box(connection, "bat1", 1, "batting", _batting())
        _box(connection, "pitch1", 1, "pitching", _pitching())
        _box(connection, "bat2", 2, "batting", _batting(known_pa=False, ab=2))
        _box(connection, "zero-ab", 2, "batting", _batting(known_pa=False, ab=0, hits=0))
        _box(connection, "pitch2", 2, "pitching", _pitching(partial=True))
        _box(connection, "bat3", 3, "batting", _batting())


def test_boxscore_targets_use_actual_counts_and_missing_pa_lower_bounds(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    first, second = dataset.load_day("2001-04-01"), dataset.load_day("2001-04-02")

    assert dataset.manifest["dataset_version"] == 5
    assert dataset.manifest["boxscore_feature_dims"] == {"batting": 19, "pitching": 21}
    assert first.live_hit_pa.tolist() == [4]
    assert first.live_hit_pa_min.tolist() == [4]
    assert first.live_hit_hits.tolist() == [1]
    assert first.box_pa_counts.tolist() == [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0]]
    assert first.box_pitch_targets.tolist() == [[4, 3, 12, 3, 1, 0, 1, 1, 0, 0]]
    assert first.box_pitch_mask.all()
    assert second.live_hit_pa.tolist() == [-1]
    assert second.live_hit_pa_min.tolist() == [2]
    assert second.live_hit_hits.tolist() == [1]
    assert second.box_pa_counts.shape == (0, 10)
    assert second.box_pitch_mask.tolist() == [
        [True, False, False, True, True, True, True, True, True, True]
    ]
    assert first.pa_targets.size == second.pa_targets.size == 0  # No fictional ordered PA.
    coverage = dataset.manifest["season_coverage"][0]
    assert coverage["box_batting_rows"] == 4
    assert coverage["box_pitching_rows"] == 2
    assert coverage["box_live_hit_queries"] == 3
    assert coverage["box_live_hit_unknown_pa_queries"] == 1
    assert coverage["box_pa_outcomes"] == 8
    assert coverage["box_pitch_observed_counts"] == 18
    assert coverage["box_target_missing_reasons"]["live_hit_no_observed_appearance"] == 1


def test_all_past_rows_contribute_priors_without_merging_observation_ids_or_current_roster(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    first, second = dataset.load_day("2001-04-01"), dataset.load_day("2001-04-02")
    assert not first.player_box_batting_features.any()
    assert not first.player_box_pitching_features.any()
    assert not first.team_box_batting_features.any()
    assert all(not route["source_index"].size for route in first.routes.values())

    assert "observed:bat1" not in second.player_ids  # Not retained as a guessed career identity.
    assert "observed:bat2" in second.player_ids
    identity = historical_player_prior("away", "batting", "동명이인")
    assert identity.identity_status == "source_name_team_cohort"
    proxy = identity.prior_id
    assert proxy in second.player_ids
    query = second.player_ids.index("observed:bat2")
    prior = second.player_ids.index(proxy)
    np.testing.assert_array_equal(
        second.player_box_batting_features[query], second.player_box_batting_features[prior]
    )
    assert second.player_box_batting_features[query, 0] == np.float32(
        math.log1p(3) / math.log1p(500)
    )
    # Complete observed outcome totals can populate the same legacy feature
    # layout, but remain explicitly uncertain cohort history, not a career ID.
    np.testing.assert_array_equal(
        second.role_features["batting"][query], second.role_features["batting"][prior]
    )
    np.testing.assert_allclose(
        second.role_features["batting"][query, :7],
        [math.log1p(4) / math.log1p(500), 3 / 4, 1 / 4, 1 / 16, 1 / 4, 1 / 4, 0],
    )
    route = second.routes["batter_participation_team"]
    assert route["source_index"].tolist() == [prior]
    assert query not in route["source_index"]  # Current actual roster cannot influence team nodes.
    np.testing.assert_allclose(
        route["event_features"][0, :5],
        [math.log1p(4) / math.log1p(100), 1 / 4, 1 / 4, 1 / 4, 1 / 16],
    )
    assert np.all(route["event_age_seconds"] >= route["publication_delay_seconds"])


def test_changing_same_day_box_stats_changes_labels_not_features(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    before = build_kbo_graph_dataset(database, tmp_path / "before")
    with duckdb.connect(str(database)) as connection:
        changed = _batting(known_pa=False, ab=100, hits=30)
        connection.execute(
            "UPDATE historical_boxscore SET stats_json=? WHERE observation_id='bat2'",
            [json.dumps(changed)],
        )
    after = build_kbo_graph_dataset(database, tmp_path / "after")
    first = before.load_day("2001-04-02")
    second = after.load_day("2001-04-02")
    for name in first.arrays:
        if "features" in name or "__" in name:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    assert first.live_hit_hits.tolist() == [1]
    assert second.live_hit_hits.tolist() == [30]
    assert not np.array_equal(
        before.load_day("2001-04-03").team_box_batting_features,
        after.load_day("2001-04-03").team_box_batting_features,
    )


def test_late_publication_is_excluded_from_past_priors(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE historical_boxscore SET available_at=? "
            "WHERE observation_id IN ('bat1','pitch1')",
            [datetime(2001, 4, 2, 12, tzinfo=_KST)],
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    second = dataset.load_day("2001-04-02")
    assert not second.team_box_batting_features.any()
    assert not second.team_box_pitching_features.any()
    assert not second.routes["batter_participation_team"]["source_index"].size
    assert dataset.load_day("2001-04-03").team_box_batting_features.any()


def test_unknown_fields_have_observation_masks_not_fake_zeros(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph", rolling_days=1)
    third = dataset.load_day("2001-04-03")
    home = third.team_ids.index("home")
    features = third.team_box_pitching_features[home]
    assert features[1] == features[2] == 0  # Unknown outs/pitches sums.
    assert (
        features[11] == features[12] == 0
    )  # Explicitly zero reporting counts, not observed zeros.
    assert features[15] > 0  # Home runs = 0 was actually reported.
    assert features[5] == 0


def test_unverified_hits_do_not_create_live_hit_targets_but_rows_remain_in_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    with duckdb.connect(str(database)) as connection:
        stats = _batting(known_pa=False, ab=2)
        stats["hits_verified"] = False
        connection.execute(
            "UPDATE historical_boxscore SET stats_json=? WHERE observation_id='bat2'",
            [json.dumps(stats)],
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph", rolling_days=1)
    assert dataset.load_day("2001-04-02").live_hit_pa.size == 0
    assert dataset.manifest["season_coverage"][0]["box_batting_rows"] == 4
    third = dataset.load_day("2001-04-03")
    prior = third.team_box_batting_features[third.team_ids.index("away")]
    assert prior[0] > 0  # The independently recorded AB remains usable.
    assert prior[1] == 0  # The inconsistent hit count does not enter history.
    assert prior[10] == np.float32(math.log1p(1) / math.log1p(500))  # Only zero-AB row reports H.


def test_contradictory_pitching_fields_are_masked_identically_in_targets_and_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    with duckdb.connect(str(database)) as connection:
        stats = _pitching()
        stats.update(hits=6, home_runs=8, earned_runs=2, strikeouts=5)
        connection.execute(
            "UPDATE historical_boxscore SET stats_json=? WHERE observation_id='pitch1'",
            [json.dumps(stats)],
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    first, second = dataset.load_day("2001-04-01"), dataset.load_day("2001-04-02")
    expected = [False, True, True, False, False, False, True, False, False, False]
    assert first.box_pitch_mask.tolist() == [expected]
    assert first.box_pitch_targets.tolist() == [[0, 3, 12, 0, 0, 0, 1, 0, 0, 0]]
    prior = second.team_box_pitching_features[second.team_ids.index("home")]
    assert (prior[10:20] > 0).tolist() == expected
    assert dataset.manifest["season_coverage"][0]["box_pitching_rows"] == 2
    reasons = dataset.manifest["season_coverage"][0]["box_target_missing_reasons"]
    assert reasons["unusable_field:pitching:hits"] == 1
    assert reasons["unusable_field:pitching:home_runs"] == 1
    assert reasons["unusable_field:pitching:batters_faced"] == 1


def test_version_two_graph_cache_loads_with_empty_boxscore_defaults(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "graph"
    dataset = build_kbo_graph_dataset(database, output)
    manifest = dataset.manifest
    manifest["dataset_version"] = 2
    entry = manifest["days"][0]
    path = output / entry["file"]
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            key: source[key]
            for key in source.files
            if "box_" not in key and key != "live_hit_pa_min"
        }
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    graph = KBOGraphDataset(output).load_day(entry["day"])
    assert graph.box_pa_counts.shape == graph.box_pitch_targets.shape == (0, 10)
    assert graph.player_box_batting_features.shape == (len(graph.player_ids), 19)
    assert not graph.player_box_batting_features.any()
    np.testing.assert_array_equal(graph.live_hit_pa, graph.live_hit_pa_min)


def test_streamed_records_drop_unused_payload_only_after_full_provenance_hash(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    records, _, digest_before = graph_module._read_records(database, None)
    boxes = [record for record in records if record.kind.startswith("box_")]
    assert len(boxes) == 6
    assert all("raw_json" not in record.data for record in boxes)
    # The name is now a used cohort key; full raw payload remains hash-only.
    assert all(record.data["display_name"] == "동명이인" for record in boxes)
    before = build_kbo_graph_dataset(database, tmp_path / "before")
    with duckdb.connect(str(database)) as connection:
        # This unused source payload must remain part of provenance, even though
        # it need not consume memory in every retained graph record.
        connection.execute(
            "UPDATE historical_boxscore SET raw_json=? WHERE observation_id='bat1'",
            [json.dumps({"unused_original_source_field": "changed"})],
        )
        streamed = list(
            graph_module._iter_dicts(
                connection,
                "SELECT observation_id FROM historical_boxscore ORDER BY observation_id",
                chunk_size=2,
            )
        )
        assert len(streamed) == 6
    _, _, digest_after = graph_module._read_records(database, None)
    after = build_kbo_graph_dataset(database, tmp_path / "after")
    assert digest_after != digest_before
    assert after.manifest["fingerprint"] != before.manifest["fingerprint"]
    for day in before.days():
        old, new = before.load_day(day), after.load_day(day)
        for name in old.arrays:
            np.testing.assert_array_equal(old.arrays[name], new.arrays[name])


def test_all_missing_placeholder_is_audited_without_false_participation_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001, include_pas=False)
    with duckdb.connect(str(database)) as connection:
        connection.execute(V5_DDL[0])
        _box(connection, "missing-placeholder", 1, "batting", {})
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    assert dataset.manifest["boxscore_history_policy"] == "common_player_game_observed_fields_v3"
    first, second = dataset.load_day("2001-04-01"), dataset.load_day("2001-04-02")
    assert first.live_hit_pa.size == first.box_pa_counts.size == 0
    assert second.player_ids == ()
    assert not second.team_box_batting_features.any()
    assert not second.routes["batter_participation_team"]["source_index"].size
    assert dataset.manifest["season_coverage"][0]["box_batting_rows"] == 1
    assert (
        dataset.manifest["season_coverage"][0]["box_target_missing_reasons"][
            "live_hit_missing_or_unverified_hits"
        ]
        == 1
    )


def test_reported_zero_is_not_treated_as_an_all_missing_placeholder(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001, include_pas=False)
    with duckdb.connect(str(database)) as connection:
        connection.execute(V5_DDL[0])
        _box(connection, "zero-ab", 1, "batting", {"at_bats": 0})
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    second = dataset.load_day("2001-04-02")
    prior = second.team_box_batting_features[second.team_ids.index("away")]
    assert prior[0] == 0
    assert prior[9] > 0  # The AB field was actually reported, although its value is zero.
    assert prior[-1] > 0
    assert len(second.routes["batter_participation_team"]["source_index"]) == 1
