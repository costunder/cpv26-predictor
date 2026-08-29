from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from cpv26.optimization import (
    GamePredictionMarket,
    MatchPickOption,
    MatchPredictionObjective,
    MatchPredictionOptimizer,
)
from cpv26.simulation import (
    BaseRunners,
    GameEndReason,
    GameSimulationSpec,
    GameSimulator,
    GameState,
    HalfInning,
    PitchingPlan,
    RunnerAdvancementRates,
    StateTransitionEngine,
    StaticPlateAppearanceProbabilityModel,
    TeamLineup,
    TerminalPlateAppearanceEvent,
)


def _simulation_spec() -> GameSimulationSpec:
    return GameSimulationSpec(
        prediction_run_id="run-2026-08-29",
        cutoff_at=datetime(2026, 8, 29, 8, tzinfo=timezone.utc),
        game_id="game-1",
        away_lineup=TeamLineup(
            "away",
            tuple(f"away-{index}" for index in range(1, 10)),
        ),
        home_lineup=TeamLineup(
            "home",
            tuple(f"home-{index}" for index in range(1, 10)),
        ),
        away_pitching_plan=PitchingPlan("away-starter"),
        home_pitching_plan=PitchingPlan("home-starter"),
        regulation_innings=2,
        max_innings=2,
    )


def test_seeded_sequential_simulation_preserves_batch_invariants() -> None:
    probability_model = StaticPlateAppearanceProbabilityModel(
        {
            TerminalPlateAppearanceEvent.SINGLE: 0.20,
            TerminalPlateAppearanceEvent.WALK: 0.10,
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT: 0.70,
        }
    )
    assert sum(probability_model.probabilities.values()) == pytest.approx(1.0)

    simulator = GameSimulator(
        probability_model,
        record_plate_appearances=True,
    )
    first = simulator.simulate_many(_simulation_spec(), 6, seed=2026)
    repeated = simulator.simulate_many(_simulation_spec(), 6, seed=2026)

    assert len(first.samples) == 6
    assert (
        first.away_win_probability
        + first.home_win_probability
        + first.tie_probability
    ) == pytest.approx(1.0)
    assert [
        (sample.away_score, sample.home_score, sample.player_hits)
        for sample in first.samples
    ] == [
        (sample.away_score, sample.home_score, sample.player_hits)
        for sample in repeated.samples
    ]

    for sample in first.samples:
        assert sample.away_score >= 0
        assert sample.home_score >= 0
        assert sample.final_inning == 2
        assert tuple(
            appearance.plate_appearance_number
            for appearance in sample.plate_appearances
        ) == tuple(range(1, len(sample.plate_appearances) + 1))
        for appearance in sample.plate_appearances:
            assert 0 <= appearance.state_after.outs <= 2
            assert (
                appearance.state_after.away_score
                >= appearance.context.state.away_score
            )
            assert (
                appearance.state_after.home_score
                >= appearance.context.state.home_score
            )


def test_match_optimizer_accounts_for_all_correct_bonus() -> None:
    markets = (
        GamePredictionMarket(
            "game-1",
            (
                MatchPickOption("favorite-1", "Favorite 1", 0.70, 1.0),
                MatchPickOption("underdog-1", "Underdog 1", 0.30, 3.0),
            ),
        ),
        GamePredictionMarket(
            "game-2",
            (
                MatchPickOption("favorite-2", "Favorite 2", 0.70, 1.0),
                MatchPickOption("underdog-2", "Underdog 2", 0.30, 3.0),
            ),
        ),
    )
    optimizer = MatchPredictionOptimizer()

    points_only = optimizer.optimize(markets)[0]
    all_correct = optimizer.optimize(
        markets,
        objective=MatchPredictionObjective(all_correct_bonus_points=10.0),
    )[0]

    assert tuple(option.option_id for _, option in points_only.picks) == (
        "underdog-1",
        "underdog-2",
    )
    assert tuple(option.option_id for _, option in all_correct.picks) == (
        "favorite-1",
        "favorite-2",
    )
    assert all_correct.all_correct_probability == pytest.approx(0.49)


def _walk_off_outcome(
    event: TerminalPlateAppearanceEvent,
    bases: BaseRunners,
    *,
    away_score: int = 0,
    home_score: int = 0,
    outs: int = 0,
):
    initial_state = GameState(
        inning=9,
        half=HalfInning.BOTTOM,
        outs=outs,
        bases=bases,
        away_score=away_score,
        home_score=home_score,
    )
    spec = replace(
        _simulation_spec(),
        regulation_innings=9,
        max_innings=12,
        initial_state=initial_state,
    )
    transition_engine = StateTransitionEngine(
        RunnerAdvancementRates(
            runner_on_second_scores_on_single=1.0,
            runner_on_first_reaches_third_on_single=1.0,
            runner_on_first_scores_on_double=1.0,
            runner_on_second_scores_on_error=1.0,
            runner_on_first_reaches_third_on_error=1.0,
        )
    )
    simulator = GameSimulator(
        StaticPlateAppearanceProbabilityModel({event: 1.0}),
        transition_engine=transition_engine,
        record_plate_appearances=True,
    )
    return initial_state, simulator.simulate(spec, seed=7)


@pytest.mark.parametrize(
    "event,bases,official_event",
    (
        (
            TerminalPlateAppearanceEvent.SINGLE,
            BaseRunners(second="runner-2", third="runner-3"),
            TerminalPlateAppearanceEvent.SINGLE,
        ),
        (
            TerminalPlateAppearanceEvent.DOUBLE,
            BaseRunners("runner-1", "runner-2", "runner-3"),
            TerminalPlateAppearanceEvent.SINGLE,
        ),
        (
            TerminalPlateAppearanceEvent.WALK,
            BaseRunners("runner-1", "runner-2", "runner-3"),
            TerminalPlateAppearanceEvent.WALK,
        ),
        (
            TerminalPlateAppearanceEvent.HIT_BY_PITCH,
            BaseRunners("runner-1", "runner-2", "runner-3"),
            TerminalPlateAppearanceEvent.HIT_BY_PITCH,
        ),
        (
            TerminalPlateAppearanceEvent.REACHED_ON_ERROR,
            BaseRunners(second="runner-2", third="runner-3"),
            TerminalPlateAppearanceEvent.REACHED_ON_ERROR,
        ),
    ),
)
def test_non_home_run_walk_off_credits_only_winning_run(
    event: TerminalPlateAppearanceEvent,
    bases: BaseRunners,
    official_event: TerminalPlateAppearanceEvent,
) -> None:
    initial, outcome = _walk_off_outcome(event, bases)

    assert outcome.end_reason is GameEndReason.WALK_OFF
    assert outcome.home_score == 1
    assert len(outcome.plate_appearances) == 1
    appearance = outcome.plate_appearances[0]
    assert appearance.runs_scored == outcome.home_score - initial.home_score
    assert appearance.state_after == outcome.final_state
    assert outcome.final_state.bases.is_empty
    assert appearance.sampled_event is event
    assert appearance.applied_event is official_event
    assert appearance.credited_total_bases == official_event.hit_value
    assert sum(outcome.player_hits.values()) == int(official_event.is_hit)


def test_walk_off_from_one_run_deficit_credits_two_runs() -> None:
    initial, outcome = _walk_off_outcome(
        TerminalPlateAppearanceEvent.SINGLE,
        BaseRunners(second="runner-2", third="runner-3"),
        away_score=1,
    )

    assert outcome.home_score == 2
    assert outcome.plate_appearances[0].runs_scored == 2
    assert outcome.home_score - initial.home_score == 2


def test_two_out_walk_off_keeps_out_state_and_caps_runs() -> None:
    _, outcome = _walk_off_outcome(
        TerminalPlateAppearanceEvent.DOUBLE,
        BaseRunners(second="runner-2", third="runner-3"),
        outs=2,
    )

    appearance = outcome.plate_appearances[0]
    assert outcome.home_score == 1
    assert appearance.context.state.outs == 2
    assert appearance.state_after.outs == 2
    assert appearance.runs_scored == 1


def test_walk_off_home_run_credits_every_run() -> None:
    _, outcome = _walk_off_outcome(
        TerminalPlateAppearanceEvent.HOME_RUN,
        BaseRunners("runner-1", "runner-2", "runner-3"),
    )

    assert outcome.home_score == 4
    assert outcome.plate_appearances[0].runs_scored == 4
    assert outcome.plate_appearances[0].credited_total_bases == 4


@pytest.mark.parametrize(
    "event,bases,away_score,official_event,official_total_bases",
    (
        (
            TerminalPlateAppearanceEvent.DOUBLE,
            BaseRunners(second="runner-2"),
            0,
            TerminalPlateAppearanceEvent.DOUBLE,
            2,
        ),
        (
            TerminalPlateAppearanceEvent.TRIPLE,
            BaseRunners(first="runner-1"),
            0,
            TerminalPlateAppearanceEvent.TRIPLE,
            3,
        ),
        (
            TerminalPlateAppearanceEvent.TRIPLE,
            BaseRunners(second="runner-2", third="runner-3"),
            1,
            TerminalPlateAppearanceEvent.DOUBLE,
            2,
        ),
        (
            TerminalPlateAppearanceEvent.DOUBLE,
            BaseRunners("runner-1", "runner-2", "runner-3"),
            1,
            TerminalPlateAppearanceEvent.DOUBLE,
            2,
        ),
    ),
)
def test_walk_off_hit_credit_uses_winning_runner_starting_base(
    event: TerminalPlateAppearanceEvent,
    bases: BaseRunners,
    away_score: int,
    official_event: TerminalPlateAppearanceEvent,
    official_total_bases: int,
) -> None:
    _, outcome = _walk_off_outcome(event, bases, away_score=away_score)

    appearance = outcome.plate_appearances[0]
    assert appearance.sampled_event is event
    assert appearance.applied_event is official_event
    assert appearance.credited_total_bases == official_total_bases
    assert appearance.state_after.bases.is_empty


def test_simulation_rejects_initial_state_that_is_already_a_walk_off() -> None:
    completed = GameState(
        inning=9,
        half=HalfInning.BOTTOM,
        away_score=0,
        home_score=1,
    )

    with pytest.raises(ValueError, match="completed walk-off"):
        replace(
            _simulation_spec(),
            regulation_innings=9,
            max_innings=12,
            initial_state=completed,
        )
