from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from cpv26.data import (
    V26_CAPTURE_PHASES,
    DatasetContractError,
    GameSample,
    PlateAppearanceRow,
    PlayerGameBattingRow,
    RunnerEventMarker,
    TeamGameRow,
    TemporalRole,
    TransitionHalf,
    TransitionRow,
    V26CaptureRow,
    V26EligibilityRow,
    V26RuleVersionRow,
    V26SlateRow,
    WeatherExperiment,
    WeatherFeatureRow,
    WeatherSourceKind,
    assert_box_score_consistency,
    assert_simulator_ready_transitions,
    assert_v26_capture_consistency,
    audit_box_score_consistency,
    audit_pa_transitions,
    audit_player_game_batting,
    audit_team_scores,
    audit_v26_capture_consistency,
    audit_weather_usage,
    build_expanding_temporal_split,
)

UTC = timezone.utc


def test_expanding_split_preserves_game_groups_and_declares_live_role() -> None:
    rows = [
        GameSample("pa-2018-a", "game-2018", 2018),
        GameSample("pa-2018-b", "game-2018", 2018),
        GameSample("pa-2022", "game-2022", 2022),
        GameSample("pa-2023-a", "game-2023", 2023),
        GameSample("horizon-2023", "game-2023", 2023),
        GameSample("pa-2024", "game-2024", 2024),
        GameSample("pa-2025", "game-2025", 2025),
        GameSample("candidate-2026", "game-2026", 2026),
    ]

    split = build_expanding_temporal_split(rows)

    assert split.assignment_for("game-2018").row_ids == ("pa-2018-a", "pa-2018-b")
    assert split.assignment_for("game-2023").role is TemporalRole.MODEL_SELECTION
    assert split.rows_for_role(TemporalRole.MODEL_SELECTION) == (
        "horizon-2023",
        "pa-2023-a",
    )
    assert split.games_for_role(TemporalRole.LIVE) == ("game-2026",)

    selection = split.fold_for(TemporalRole.MODEL_SELECTION)
    calibration = split.fold_for(TemporalRole.CALIBRATION)
    holdout = split.fold_for(TemporalRole.HOLDOUT)
    live = split.fold_for(TemporalRole.LIVE)
    assert selection.training_seasons == tuple(range(2018, 2023))
    assert selection.training_game_ids == ("game-2018", "game-2022")
    assert selection.target_game_ids == ("game-2023",)
    assert calibration.training_game_ids == ("game-2018", "game-2022", "game-2023")
    assert holdout.training_game_ids == (
        "game-2018",
        "game-2022",
        "game-2023",
        "game-2024",
    )
    assert live.training_game_ids == (
        "game-2018",
        "game-2022",
        "game-2023",
        "game-2024",
        "game-2025",
    )
    assert not set(live.training_game_ids).intersection(live.target_game_ids)


def test_expanding_split_rejects_conflicting_game_seasons() -> None:
    rows = [
        GameSample("row-a", "double-header-safe-id", 2024),
        GameSample("row-b", "double-header-safe-id", 2025),
    ]

    with pytest.raises(DatasetContractError) as error:
        build_expanding_temporal_split(rows)

    assert [violation.code for violation in error.value.violations] == [
        "GAME_SEASON_CONFLICT"
    ]


def test_expanding_split_rejects_duplicate_rows_and_undeclared_seasons() -> None:
    rows = [
        GameSample("same-row", "old-game", 2017),
        GameSample("same-row", "future-game", 2027),
    ]

    with pytest.raises(DatasetContractError) as error:
        build_expanding_temporal_split(rows)

    assert {violation.code for violation in error.value.violations} == {
        "DUPLICATE_SAMPLE_ROW_ID",
        "SEASON_OUTSIDE_POLICY",
    }


def _consistent_box_score() -> tuple[
    list[PlateAppearanceRow],
    list[PlayerGameBattingRow],
    list[TeamGameRow],
]:
    appearances = [
        PlateAppearanceRow("pa-1", "game-1", "player-a", "team-a", True, True, 1, 0),
        PlateAppearanceRow("pa-2", "game-1", "player-a", "team-a", True, False, 0, 0),
        PlateAppearanceRow("pa-3", "game-1", "player-b", "team-a", True, True, 4, 2),
        PlateAppearanceRow("pa-4", "game-1", "player-c", "team-b", False, False, 0, 1),
    ]
    batting = [
        PlayerGameBattingRow(
            "bat-a",
            "game-1",
            "player-a",
            "team-a",
            plate_appearances=2,
            at_bats=2,
            hits=1,
            singles=1,
            doubles=0,
            triples=0,
            home_runs=0,
            started=True,
            batting_order=1,
        ),
        PlayerGameBattingRow(
            "bat-b",
            "game-1",
            "player-b",
            "team-a",
            plate_appearances=1,
            at_bats=1,
            hits=1,
            singles=0,
            doubles=0,
            triples=0,
            home_runs=1,
        ),
        PlayerGameBattingRow(
            "bat-c",
            "game-1",
            "player-c",
            "team-b",
            plate_appearances=1,
            at_bats=0,
            hits=0,
            singles=0,
            doubles=0,
            triples=0,
            home_runs=0,
        ),
    ]
    teams = [
        TeamGameRow("tg-a", "game-1", "team-a", "team-b", True, 2),
        TeamGameRow("tg-b", "game-1", "team-b", "team-a", False, 1),
    ]
    return appearances, batting, teams


def test_box_score_contract_accepts_consistent_pa_player_and_team_totals() -> None:
    appearances, batting, teams = _consistent_box_score()

    report = audit_box_score_consistency(appearances, batting, teams)

    assert report.ok
    assert report.violations == ()
    assert_box_score_consistency(appearances, batting, teams)


def test_player_game_audit_reports_missing_summary_and_hit_type_drift() -> None:
    appearances, batting, _ = _consistent_box_score()
    changed_appearances = [replace(appearances[0], total_bases=2), *appearances[1:]]
    changed_batting = batting[:-1]

    report = audit_player_game_batting(changed_appearances, changed_batting)
    codes = {violation.code for violation in report.violations}

    assert "PLAYER_GAME_SINGLES_MISMATCH" in codes
    assert "PLAYER_GAME_DOUBLES_MISMATCH" in codes
    assert "PLAYER_GAME_BATTING_MISSING" in codes
    with pytest.raises(DatasetContractError):
        report.raise_if_invalid()


def test_team_score_audit_reports_event_score_and_pair_failures() -> None:
    appearances, _, teams = _consistent_box_score()
    broken_teams = [replace(teams[0], runs=3)]

    report = audit_team_scores(appearances, broken_teams)
    codes = {violation.code for violation in report.violations}

    assert codes >= {
        "TEAM_SCORE_MISMATCH",
        "TEAM_GAME_MISSING",
        "TEAM_GAME_PAIR_INCOMPLETE",
    }


def _complete_transition(
    plate_appearance_id: str,
    sequence_in_game: int,
    *,
    inning: int = 1,
    half_inning: TransitionHalf = TransitionHalf.TOP,
    home_score_before: int = 0,
    away_score_before: int = 0,
    outs_before: int = 0,
    runners_before: str = "000",
    outs_added: int = 0,
    runners_after: str = "100",
    home_score_after: int = 0,
    away_score_after: int = 0,
    runs_scored: int = 0,
) -> TransitionRow:
    return TransitionRow(
        plate_appearance_id=plate_appearance_id,
        game_id="game-transition",
        sequence_in_game=sequence_in_game,
        event_subsequence=0,
        inning=inning,
        half_inning=half_inning,
        home_score_before=home_score_before,
        away_score_before=away_score_before,
        outs_before=outs_before,
        runners_before=runners_before,
        outs_added=outs_added,
        runners_after=runners_after,
        home_score_after=home_score_after,
        away_score_after=away_score_after,
        runs_scored=runs_scored,
        transition_complete=True,
    )


def test_legacy_transition_is_allowed_but_not_simulator_ready() -> None:
    legacy = TransitionRow(
        plate_appearance_id="legacy-pa",
        game_id="game-transition",
        sequence_in_game=1,
        event_subsequence=0,
        inning=1,
        half_inning=TransitionHalf.TOP,
        home_score_before=None,
        away_score_before=None,
        outs_before=0,
        runners_before="000",
        outs_added=None,
        runners_after=None,
        home_score_after=None,
        away_score_after=None,
        runs_scored=0,
        transition_complete=False,
    )

    assert audit_pa_transitions([legacy]).ok
    ready = audit_pa_transitions([legacy], simulator_ready=True)
    assert {violation.code for violation in ready.violations} == {
        "TRANSITION_INCOMPLETE_FOR_SIMULATOR"
    }
    with pytest.raises(DatasetContractError):
        assert_simulator_ready_transitions([legacy])


def test_complete_pa_transitions_preserve_simple_next_pa_state() -> None:
    first = _complete_transition("pa-1", 1)
    second = _complete_transition(
        "pa-2",
        2,
        runners_before="100",
        outs_added=1,
        runners_after="100",
    )

    report = audit_pa_transitions([second, first], simulator_ready=True)

    assert report.ok
    assert_simulator_ready_transitions([second, first])


def test_pa_continuity_skips_pair_with_explicit_runner_event() -> None:
    first = _complete_transition("pa-1", 1)
    changed_pre_state = _complete_transition(
        "pa-2",
        2,
        runners_before="010",
        runners_after="010",
    )
    without_marker = audit_pa_transitions([first, changed_pre_state])
    assert "TRANSITION_NEXT_RUNNERS_MISMATCH" in {
        violation.code for violation in without_marker.violations
    }

    marker = RunnerEventMarker(
        "runner-event-1",
        "game-transition",
        sequence_in_game=1,
        event_subsequence=1,
    )
    assert audit_pa_transitions(
        [first, changed_pre_state],
        runner_events=[marker],
        simulator_ready=True,
    ).ok


def test_pa_transition_validates_score_outs_runners_and_half_rollover() -> None:
    invalid = _complete_transition(
        "pa-invalid",
        1,
        home_score_before=0,
        away_score_before=0,
        outs_before=2,
        outs_added=1,
        runners_after="100",
        home_score_after=1,
        away_score_after=0,
        runs_scored=1,
    )
    overflow = replace(invalid, plate_appearance_id="pa-overflow", outs_added=2)
    codes = {
        violation.code
        for violation in audit_pa_transitions([invalid, overflow]).violations
    }
    assert codes >= {
        "TRANSITION_SCORE_DELTA_MISMATCH",
        "TRANSITION_RUNNERS_AFTER_THREE_OUTS",
        "TRANSITION_OUTS_OVERFLOW",
    }

    top_end = _complete_transition(
        "pa-top-end",
        10,
        outs_before=2,
        runners_before="100",
        outs_added=1,
        runners_after="000",
    )
    bottom_start = _complete_transition(
        "pa-bottom-start",
        11,
        half_inning=TransitionHalf.BOTTOM,
    )
    assert audit_pa_transitions([top_end, bottom_start], simulator_ready=True).ok


def _consistent_v26_capture_contract() -> tuple[
    list[V26SlateRow],
    list[V26RuleVersionRow],
    list[V26EligibilityRow],
    list[V26CaptureRow],
]:
    lock_at = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
    slates = [
        V26SlateRow(
            slate_id="slate-1",
            rule_version="rules-v1",
            live_card_version="cards-v1",
            position_eligibility_snapshot_id="elig-v1",
            lock_at=lock_at,
        )
    ]
    rules = [V26RuleVersionRow("rules-v1")]
    eligibility = [
        V26EligibilityRow(
            eligibility_row_id="elig-row-1",
            slate_id="slate-1",
            position_eligibility_snapshot_id="elig-v1",
            live_card_version="cards-v1",
            player_id="player-1",
            position="OF",
            is_eligible=True,
        )
    ]
    captures = [
        V26CaptureRow(
            capture_row_id=f"capture-{phase.value}",
            selection_snapshot_id=f"snapshot-{phase.value}",
            slate_id="slate-1",
            phase=phase,
            rule_version="rules-v1",
            live_card_version="cards-v1",
            position_eligibility_snapshot_id="elig-v1",
            player_id="player-1",
            position="OF",
            captured_at=lock_at - timedelta(hours=4 - index),
        )
        for index, phase in enumerate(V26_CAPTURE_PHASES)
    ]
    return slates, rules, eligibility, captures


def test_v26_capture_contract_accepts_four_ordered_horizons() -> None:
    slates, rules, eligibility, captures = _consistent_v26_capture_contract()

    report = audit_v26_capture_consistency(slates, rules, eligibility, captures)

    assert report.ok
    assert_v26_capture_consistency(slates, rules, eligibility, captures)


def test_v26_capture_contract_reports_missing_horizon() -> None:
    slates, rules, eligibility, captures = _consistent_v26_capture_contract()

    report = audit_v26_capture_consistency(slates, rules, eligibility, captures[:-1])
    codes = {violation.code for violation in report.violations}

    assert codes >= {
        "V26_SLATE_CAPTURE_PHASES_INCOMPLETE",
        "V26_ELIGIBLE_CAPTURE_PHASES_INCOMPLETE",
    }


def test_v26_capture_contract_cross_checks_slate_rule_card_and_eligibility() -> None:
    slates, _rules, eligibility, captures = _consistent_v26_capture_contract()
    broken = replace(
        captures[0],
        rule_version="wrong-rule",
        live_card_version="wrong-card",
        position_eligibility_snapshot_id="wrong-eligibility",
        captured_at=slates[0].lock_at + timedelta(minutes=1),
    )

    report = audit_v26_capture_consistency(
        slates,
        [],
        eligibility,
        [broken, *captures[1:]],
    )
    codes = {violation.code for violation in report.violations}

    assert codes >= {
        "V26_SLATE_RULE_MISSING",
        "V26_CAPTURE_RULE_MISMATCH",
        "V26_CAPTURE_CARD_MISMATCH",
        "V26_CAPTURE_ELIGIBILITY_SNAPSHOT_MISMATCH",
        "V26_CAPTURE_ELIGIBILITY_MISSING",
        "V26_CAPTURE_AFTER_LOCK",
    }


def test_v26_capture_contract_rejects_ineligible_and_out_of_order_capture() -> None:
    slates, rules, eligibility, captures = _consistent_v26_capture_contract()
    ineligible = [replace(eligibility[0], is_eligible=False)]
    out_of_order = [
        *captures[:-1],
        replace(captures[-1], captured_at=captures[0].captured_at - timedelta(minutes=1)),
    ]

    report = audit_v26_capture_consistency(
        slates,
        rules,
        ineligible,
        out_of_order,
    )
    codes = {violation.code for violation in report.violations}

    assert "V26_CAPTURE_PLAYER_INELIGIBLE" in codes
    assert "V26_CAPTURE_PHASE_ORDER_INVALID" in codes


def test_v26_capture_phase_must_be_explicit() -> None:
    lock_at = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)

    with pytest.raises(TypeError, match="explicit V26CapturePhase"):
        V26CaptureRow(
            capture_row_id="capture-unspecified",
            selection_snapshot_id="snapshot-unspecified",
            slate_id="slate-1",
            phase="unspecified",  # type: ignore[arg-type]
            rule_version="rules-v1",
            live_card_version="cards-v1",
            position_eligibility_snapshot_id="elig-v1",
            player_id="player-1",
            position="OF",
            captured_at=lock_at,
        )


def _weather_times() -> tuple[datetime, datetime, datetime, datetime]:
    issued_at = datetime(2025, 7, 1, 6, 0, tzinfo=UTC)
    available_at = issued_at + timedelta(minutes=5)
    cutoff_at = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
    target_at = datetime(2025, 7, 1, 18, 30, tzinfo=UTC)
    return issued_at, available_at, cutoff_at, target_at


def test_forecast_weather_contract_requires_pre_cutoff_revision() -> None:
    issued_at, available_at, cutoff_at, target_at = _weather_times()
    safe = WeatherFeatureRow(
        "weather-1",
        "game-1",
        "temperature",
        WeatherSourceKind.FORECAST,
        target_at,
        available_at,
        issued_at=issued_at,
    )

    assert audit_weather_usage(
        [safe],
        cutoff_by_game={"game-1": cutoff_at},
        experiment=WeatherExperiment.FORECAST,
    ).ok

    leaked = replace(
        safe,
        weather_feature_id="weather-leaked",
        issued_at=cutoff_at + timedelta(minutes=1),
        available_at=cutoff_at + timedelta(minutes=2),
    )
    report = audit_weather_usage(
        [leaked],
        cutoff_by_game={"game-1": cutoff_at},
        experiment=WeatherExperiment.FORECAST,
    )
    assert {violation.code for violation in report.violations} >= {
        "FORECAST_ISSUED_AFTER_CUTOFF",
        "FORECAST_AVAILABLE_AFTER_CUTOFF",
    }


def test_observed_weather_requires_explicit_oracle_experiment() -> None:
    _, _, cutoff_at, target_at = _weather_times()
    observed_at = target_at + timedelta(minutes=30)
    observation = WeatherFeatureRow(
        "observation-1",
        "game-1",
        "temperature",
        WeatherSourceKind.OBSERVATION,
        target_at,
        observed_at + timedelta(minutes=5),
        observed_at=observed_at,
    )

    ordinary = audit_weather_usage(
        [observation],
        cutoff_by_game={"game-1": cutoff_at},
        experiment=WeatherExperiment.FORECAST,
    )
    assert {violation.code for violation in ordinary.violations} == {
        "ORACLE_WEATHER_IN_FORECAST_DATASET"
    }
    assert audit_weather_usage(
        [observation],
        cutoff_by_game={"game-1": cutoff_at},
        experiment=WeatherExperiment.ORACLE_WEATHER,
    ).ok


def test_oracle_dataset_rejects_mixed_forecast_rows() -> None:
    issued_at, available_at, cutoff_at, target_at = _weather_times()
    forecast = WeatherFeatureRow(
        "forecast-1",
        "game-1",
        "wind_speed",
        WeatherSourceKind.FORECAST,
        target_at,
        available_at,
        issued_at=issued_at,
    )

    report = audit_weather_usage(
        [forecast],
        cutoff_by_game={"game-1": cutoff_at},
        experiment=WeatherExperiment.ORACLE_WEATHER,
    )

    assert {violation.code for violation in report.violations} == {
        "FORECAST_IN_ORACLE_DATASET"
    }


def test_weather_contract_rejects_naive_timestamps() -> None:
    _, available_at, _, target_at = _weather_times()

    with pytest.raises(ValueError, match="issued_at must include timezone"):
        WeatherFeatureRow(
            "forecast-naive",
            "game-1",
            "humidity",
            WeatherSourceKind.FORECAST,
            target_at,
            available_at,
            issued_at=datetime(2025, 7, 1, 6, 0),
        )
