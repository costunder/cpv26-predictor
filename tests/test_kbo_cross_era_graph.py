"""Regression contracts for source-neutral historical/modern graph features."""

from __future__ import annotations

import json
import math
import runpy
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest

from cpv26.data.kbo_graph_dataset import build_kbo_graph_dataset
from cpv26.data.kbo_player_identity import historical_player_prior
from cpv26.data.schema_v5 import V5_DDL
from test_kbo_graph_dataset import _database, _pa
from test_kbo_history_graph import _box, _historical_database

_KST = ZoneInfo("Asia/Seoul")


def _encoded(value: float) -> float:
    return math.log1p(value) / math.log1p(500)


def test_modern_observed_pa_populates_the_same_four_box_feature_blocks(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    graph = dataset.load_day("2023-04-02")
    batter = graph.player_ids.index("old-batter")
    pitcher = graph.player_ids.index("old-pitcher")
    away, home = graph.team_ids.index("away"), graph.team_ids.index("home")

    assert graph.player_box_batting_features[batter, 1] == pytest.approx(_encoded(1))
    assert graph.player_box_pitching_features[pitcher, 0] == pytest.approx(_encoded(1))
    assert graph.team_box_batting_features[away, 1] == pytest.approx(_encoded(1))
    assert graph.team_box_pitching_features[home, 0] == pytest.approx(_encoded(1))
    assert graph.player_box_batting_features.shape[1] == 19
    assert graph.player_box_pitching_features.shape[1] == 21


def test_pa_summary_reporting_counts_use_player_game_units_not_pa_rows(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "pa1-walk", "g1", 1, "old-batter", "old-pitcher", "walk")
    graph = build_kbo_graph_dataset(database, tmp_path / "graph").load_day("2023-04-02")
    batter = graph.player_box_batting_features[graph.player_ids.index("old-batter")]
    pitcher = graph.player_box_pitching_features[graph.player_ids.index("old-pitcher")]

    assert batter[0] == pytest.approx(_encoded(1))  # One observed AB, plus one walk.
    assert batter[4] == pytest.approx(_encoded(2))  # Two observed completed PAs.
    assert batter[9] == pytest.approx(_encoded(1))  # AB appears in one player-game summary.
    assert batter[13] == pytest.approx(_encoded(1))  # PA appears in one summary, not two.
    assert pitcher[0] == pytest.approx(_encoded(2))
    assert pitcher[10] == pytest.approx(_encoded(1))


def test_pa_derived_missing_official_fields_are_not_reported_as_zero(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    graph = build_kbo_graph_dataset(database, tmp_path / "graph").load_day("2023-04-02")
    batter = graph.player_box_batting_features[graph.player_ids.index("old-batter")]
    pitcher = graph.player_box_pitching_features[graph.player_ids.index("old-pitcher")]

    assert batter[11] == batter[12] == 0  # No credited player R or RBI from team runs_scored.
    assert pitcher[11] == pitcher[12] == 0  # Complete pitcher outs/pitches are not available.
    assert pitcher[18] == pitcher[19] == 0  # No official pitcher R/ER allocation from terminal PA.
    assert pitcher[10] > 0  # BF, unlike those fields, is actually observed in this summary.
    assert batter[9] > 0


def test_same_team_players_keep_distinct_personal_pa_histories(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "pa1-other", "g1", 1, "other-batter", "old-pitcher", "strikeout")
    graph = build_kbo_graph_dataset(database, tmp_path / "graph").load_day("2023-04-02")
    first = graph.player_box_batting_features[graph.player_ids.index("old-batter")]
    second = graph.player_box_batting_features[graph.player_ids.index("other-batter")]

    assert first[1] > 0
    assert second[1] == 0
    assert first[10] == second[10] == pytest.approx(_encoded(1))
    assert not np.array_equal(first, second)
    team = graph.team_box_batting_features[graph.team_ids.index("away")]
    assert team[9] == pytest.approx(_encoded(2))  # Two distinct player-game observations.


def test_modern_same_day_changes_affect_labels_but_not_shared_input_blocks(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    before = build_kbo_graph_dataset(database, tmp_path / "before")
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE observed_plate_appearance SET outcome='walk',is_hit=false,"
            "is_at_bat=false,total_bases=0 WHERE observed_pa_row_id='pa2'"
        )
    after = build_kbo_graph_dataset(database, tmp_path / "after")
    original, changed = before.load_day("2023-04-02"), after.load_day("2023-04-02")

    for name in original.arrays:
        if "features" in name or "__" in name:
            np.testing.assert_array_equal(original.arrays[name], changed.arrays[name])
    assert not np.array_equal(
        before.load_day("2023-04-03").team_box_batting_features,
        after.load_day("2023-04-03").team_box_batting_features,
    )


def test_late_pa_cannot_enter_summary_before_its_publication_cutoff(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _pa(
            connection,
            "late-pa",
            "g1",
            1,
            "old-batter",
            "old-pitcher",
            "home_run",
            available_at=datetime(2023, 4, 2, 12, tzinfo=_KST),
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    second, third = dataset.load_day("2023-04-02"), dataset.load_day("2023-04-03")
    early = second.player_box_batting_features[second.player_ids.index("old-batter")]
    later = third.player_box_batting_features[third.player_ids.index("old-batter")]

    # Deferring the whole player-game group until all components are known is
    # also valid; including the unpublished second hit is not.
    assert early[1] <= _encoded(1) + 1e-7
    assert early[6] == 0  # The late home run has not been published.
    assert later[1] == pytest.approx(_encoded(2))
    assert later[6] == pytest.approx(_encoded(1))


def test_verified_historical_box_values_populate_legacy_role_inputs_too(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    graph = build_kbo_graph_dataset(database, tmp_path / "graph").load_day("2001-04-02")
    batter = graph.player_ids.index(historical_player_prior("away", "batting", "동명이인").prior_id)
    pitcher = graph.player_ids.index(
        historical_player_prior("home", "pitching", "동명이인").prior_id
    )

    assert graph.player_batting_features[batter, 0] == pytest.approx(_encoded(4))
    assert graph.player_batting_features[batter, 1] == pytest.approx(3 / 4)
    assert graph.player_batting_features[batter, 2] == pytest.approx(1 / 4)
    # Official pitching box scores do not report total bases. The complete
    # seven-count legacy role representation stays unavailable, rather than
    # claiming zero TB; separately masked box fields still carry actual BF/H.
    assert not graph.player_pitching_features[pitcher].any()
    assert graph.player_box_pitching_features[pitcher, 0] == pytest.approx(_encoded(4))
    assert graph.player_box_pitching_features[pitcher, 4] == pytest.approx(_encoded(1))


def test_overlapping_box_and_pa_sources_do_not_double_count_team_history(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001)
    with duckdb.connect(str(database)) as connection:
        connection.execute(V5_DDL[0])
        _box(
            connection,
            "overlap-box",
            1,
            "batting",
            {
                "at_bats": 1,
                "hits": 1,
                "plate_appearances": 1,
                "outcome_counts": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                "counts_verified": True,
                "hits_verified": True,
            },
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    graph = dataset.load_day("2001-04-02")
    team = graph.team_box_batting_features[graph.team_ids.index("away")]

    assert team[0] == pytest.approx(_encoded(1))
    assert team[1] == pytest.approx(_encoded(1))
    assert team[9] == pytest.approx(_encoded(1))
    # Unresolved archive observations cannot duplicate the same team's modern
    # player-game supervision simply by using a different observation ID.
    assert len(dataset.load_day("2001-04-01").live_hit_query_ids) == 1


def test_one_revised_pa_rebuilds_its_group_without_losing_other_pas(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "pa1-walk", "g1", 1, "old-batter", "old-pitcher", "walk")
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
        connection.execute(
            "UPDATE observed_plate_appearance "
            "SET valid_from=TIMESTAMPTZ '2023-04-02 12:00:00+09' "
            "WHERE observed_pa_row_id='pa1-correction'"
        )
        connection.execute(
            "UPDATE observed_plate_appearance "
            "SET valid_to=TIMESTAMPTZ '2023-04-02 12:00:00+09' "
            "WHERE observed_pa_row_id='pa1'"
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    early, later = dataset.load_day("2023-04-02"), dataset.load_day("2023-04-03")
    first = early.player_box_batting_features[early.player_ids.index("old-batter")]
    revised = later.player_box_batting_features[later.player_ids.index("old-batter")]

    assert first[1] == pytest.approx(_encoded(1))
    assert first[4] == pytest.approx(_encoded(2))
    assert first[9] == first[13] == pytest.approx(_encoded(1))
    assert revised[1] == 0  # The prior single is corrected, not retained beside its revision.
    assert revised[4] == pytest.approx(_encoded(3))  # Day-one K+walk plus day-two K.
    assert revised[7] == pytest.approx(_encoded(1))  # The unrevised walk is not lost.
    assert revised[9] == revised[13] == pytest.approx(_encoded(2))  # Two player-games.


def test_expired_pas_remove_their_entire_player_game_summary(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph", rolling_days=1)
    early, later = dataset.load_day("2023-04-02"), dataset.load_day("2023-04-03")
    first = early.player_box_batting_features[early.player_ids.index("old-batter")]
    recent = later.player_box_batting_features[later.player_ids.index("old-batter")]
    team = later.team_box_batting_features[later.team_ids.index("away")]

    assert first[1] == pytest.approx(_encoded(1))
    assert recent[1] == 0  # Day-one single expired; only day-two strikeout remains.
    assert recent[4] == recent[9] == recent[13] == pytest.approx(_encoded(1))
    assert team[0] == team[9] == pytest.approx(_encoded(2))  # Day-two HR and K batters.
    assert team[1] == pytest.approx(_encoded(1))


def test_historical_different_names_have_distinct_priors_within_one_team(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2001, include_pas=False)
    hit = {
        "at_bats": 1,
        "hits": 1,
        "plate_appearances": 1,
        "outcome_counts": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "counts_verified": True,
        "hits_verified": True,
    }
    out = {
        "at_bats": 1,
        "hits": 0,
        "plate_appearances": 1,
        "outcome_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "counts_verified": True,
        "hits_verified": True,
    }
    with duckdb.connect(str(database)) as connection:
        connection.execute(V5_DDL[0])
        for observation, day, stats in (
            ("name-a-past", 1, hit),
            ("name-b-past", 1, out),
            ("name-a-query", 2, out),
            ("name-b-query", 2, hit),
            ("name-a-second-query", 2, hit),
        ):
            _box(connection, observation, day, "batting", stats)
        connection.execute(
            "UPDATE historical_boxscore SET display_name=CASE "
            "WHEN observation_id LIKE 'name-a-%' THEN '타자갑' ELSE '타자을' END"
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    graph = dataset.load_day("2001-04-02")
    first = graph.player_ids.index("observed:name-a-query")
    second = graph.player_ids.index("observed:name-b-query")
    same_name = graph.player_ids.index("observed:name-a-second-query")

    assert graph.player_box_batting_features[first, 1] == pytest.approx(_encoded(1))
    assert graph.player_box_batting_features[second, 1] == 0
    assert not np.array_equal(
        graph.player_box_batting_features[first], graph.player_box_batting_features[second]
    )
    assert first != same_name  # Sharing a cohort must not merge source observation/query IDs.
    np.testing.assert_array_equal(
        graph.player_box_batting_features[first], graph.player_box_batting_features[same_name]
    )
    assert len(graph.live_hit_query_ids) == len(set(graph.live_hit_query_ids)) == 3
    assert historical_player_prior("away", "batting", "타자갑").identity_status == (
        "source_name_team_cohort"
    )
    assert "cohort" in dataset.manifest["policies"]["boxscore_identity"].lower()


def test_missing_historical_name_uses_an_explicit_team_fallback_not_a_person_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _historical_database(database)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE historical_boxscore SET display_name=NULL WHERE observation_id='bat2'"
        )
    graph = build_kbo_graph_dataset(database, tmp_path / "graph").load_day("2001-04-02")
    player = graph.player_ids.index("observed:bat2")
    team = graph.team_ids.index("away")
    np.testing.assert_array_equal(
        graph.player_box_batting_features[player], graph.team_box_batting_features[team]
    )
    assert historical_player_prior("away", "batting", None).identity_status == "team_role_fallback"


def test_actual_modern_aggregate_targets_are_counted_separately_from_raw_archive(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    for entry in dataset.manifest["days"]:
        graph = dataset.load_day(entry["day"])
        assert entry["box_pa_queries"] == len(graph.box_pa_counts)
        assert entry["box_pa_outcomes"] == int(graph.box_pa_counts.sum())
        assert entry["box_pitch_queries"] == len(graph.box_pitch_targets)
        assert entry["box_pitch_observed_counts"] == int(graph.box_pitch_mask.sum())
        assert entry["raw_archive_boxscore"]["box_pa_queries"] == 0
        assert entry["raw_archive_boxscore"]["box_pitch_queries"] == 0
    season = dataset.manifest["season_coverage"][0]
    assert season["box_pa_queries"] == season["pa_derived_batting_queries"] == 4
    assert season["box_pa_outcomes"] == 4
    assert season["box_pitch_queries"] == season["pa_derived_pitching_queries"] == 4
    assert season["box_pitch_observed_counts"] == 24
    assert dataset.manifest["label_quality"]["training_targets"]["box_pa_queries"] == 4
    assert dataset.manifest["label_quality"]["historical_boxscore"]["box_pa_queries"] == 0


def test_ci_only_pa_group_is_not_counted_as_emitted_batting_histogram(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "ci", "g1", 1, "ci-only-batter", "old-pitcher", "catcher_interference")
        connection.execute(
            "UPDATE observed_plate_appearance SET is_at_bat=false WHERE plate_appearance_id='ci'"
        )
    dataset = build_kbo_graph_dataset(database, tmp_path / "graph")
    first = dataset.manifest["days"][0]
    graph = dataset.load_day(first["day"])
    assert first["live_hit_queries"] == 2
    assert first["pa_derived_batting_queries"] == first["box_pa_queries"] == 1
    assert first["box_pa_queries"] == len(graph.box_pa_counts)


@pytest.mark.parametrize("target", ["manifest", "npz", "directory"])
def test_audit_cli_cannot_overwrite_its_graph_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    directory = tmp_path / "graph"
    dataset = build_kbo_graph_dataset(database, directory)
    manifest = directory / "manifest.json"
    npz = directory / dataset.manifest["days"][0]["file"]
    before = (manifest.read_bytes(), npz.read_bytes())
    output = {"manifest": manifest, "npz": npz, "directory": directory}[target]
    script = Path(__file__).parents[1] / "scripts" / "audit_cross_era_graph.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(directory), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert (manifest.read_bytes(), npz.read_bytes()) == before


def test_audit_cli_checks_all_source_counts_and_writes_only_external_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    directory = tmp_path / "graph"
    build_kbo_graph_dataset(database, directory)
    output = tmp_path / "audit.json"
    script = Path(__file__).parents[1] / "scripts" / "audit_cross_era_graph.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(directory), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["validation"]["target_count_mismatches"] == 0
    assert report["training_targets_from_arrays"]["box_pa_queries"] == 4
