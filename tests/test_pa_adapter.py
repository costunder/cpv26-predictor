from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from random import Random

import pytest

from cpv26.simulation import (
    NEURAL_PA_OUTCOME_TO_INDEX,
    NEURAL_PA_OUTCOMES,
    TERMINAL_TO_NEURAL_TARGET,
    AdaptedPlateAppearanceProbabilityModel,
    BaseRunners,
    FuturePlateAppearanceContext,
    GameState,
    NeuralTerminalAdapterConfig,
    NeuralTerminalProbabilityAdapter,
    ObservedPlateAppearance,
    PlateAppearanceProbabilityModel,
    StateTransitionEngine,
    TerminalPlateAppearanceEvent,
    neural_training_target,
    neural_training_target_index,
)

CUTOFF = datetime(2026, 8, 29, 9, tzinfo=timezone.utc)
EMPTY_BASES = BaseRunners()


def _observed_records() -> tuple[ObservedPlateAppearance, ...]:
    events = (
        TerminalPlateAppearanceEvent.WALK,
        TerminalPlateAppearanceEvent.WALK,
        TerminalPlateAppearanceEvent.WALK,
        TerminalPlateAppearanceEvent.HIT_BY_PITCH,
        TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE,
        TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
        TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
        TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
        TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
        TerminalPlateAppearanceEvent.DOUBLE_PLAY,
        TerminalPlateAppearanceEvent.DOUBLE_PLAY,
        TerminalPlateAppearanceEvent.FIELDERS_CHOICE,
    )
    records = []
    for index, event in enumerate(events):
        is_batted_ball_split = event in {
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
            TerminalPlateAppearanceEvent.DOUBLE_PLAY,
            TerminalPlateAppearanceEvent.FIELDERS_CHOICE,
        }
        records.append(
            ObservedPlateAppearance(
                game_id="historical-game",
                plate_appearance_id=f"pa-{index}",
                event_at=CUTOFF - timedelta(days=2),
                available_at=CUTOFF - timedelta(days=1),
                batter_id=f"batter-{index % 3}",
                pitcher_id=f"pitcher-{index % 2}",
                batter_team_id="batting-team",
                pitcher_team_id="fielding-team",
                event=event,
                outs_before=0 if is_batted_ball_split else None,
                bases_before=(
                    BaseRunners(first="historical-runner")
                    if is_batted_ball_split
                    else None
                ),
            )
        )
    return tuple(records)


def _context(*, bases: BaseRunners = EMPTY_BASES, outs: int = 0) -> FuturePlateAppearanceContext:
    return FuturePlateAppearanceContext(
        prediction_run_id="prediction-run",
        cutoff_at=CUTOFF,
        game_id="future-game",
        plate_appearance_number=1,
        batter_id="future-batter",
        pitcher_id="future-pitcher",
        catcher_id="future-catcher",
        batter_team_id="batting-team",
        pitcher_team_id="fielding-team",
        batting_order_index=0,
        state=GameState(outs=outs, bases=bases),
    )


def _neural_weights(**overrides: float) -> dict[str, float]:
    weights = dict.fromkeys(NEURAL_PA_OUTCOMES, 0.0)
    weights.update(overrides)
    return weights


def test_fold_estimated_adapter_splits_neural_categories_and_normalizes() -> None:
    config = NeuralTerminalAdapterConfig.estimate(
        _observed_records(),
        cutoff_at=CUTOFF,
        source_fold_id="train-fold-1",
    )
    adapter = NeuralTerminalProbabilityAdapter(config)
    probabilities = adapter.adapt(
        _neural_weights(walk_or_hbp=0.4, ball_in_play_out=0.6),
        _context(bases=BaseRunners(first="runner")),
    )

    assert config.records_used == len(_observed_records())
    assert config.source_fold_id == "train-fold-1"
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[TerminalPlateAppearanceEvent.WALK] > probabilities[
        TerminalPlateAppearanceEvent.HIT_BY_PITCH
    ]
    assert probabilities[TerminalPlateAppearanceEvent.DOUBLE_PLAY] > 0.0
    assert probabilities[TerminalPlateAppearanceEvent.FIELDERS_CHOICE] > 0.0
    assert probabilities[TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE] > 0.0
    assert set(probabilities) == set(TerminalPlateAppearanceEvent)


def test_every_terminal_event_has_one_explicit_neural_training_policy() -> None:
    assert set(TERMINAL_TO_NEURAL_TARGET) == set(TerminalPlateAppearanceEvent)
    assert {
        target for target in TERMINAL_TO_NEURAL_TARGET.values() if target is not None
    } == set(NEURAL_PA_OUTCOMES)
    assert neural_training_target(TerminalPlateAppearanceEvent.HIT_BY_PITCH) == "walk_or_hbp"
    assert neural_training_target(TerminalPlateAppearanceEvent.DOUBLE_PLAY) == "ball_in_play_out"
    assert neural_training_target(TerminalPlateAppearanceEvent.FIELDERS_CHOICE) == (
        "ball_in_play_out"
    )
    assert neural_training_target(TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE) is None
    for event in TerminalPlateAppearanceEvent:
        label = neural_training_target(event)
        target_index = neural_training_target_index(event)
        if label is None:
            assert target_index is None
        else:
            assert target_index == NEURAL_PA_OUTCOME_TO_INDEX[label]
            assert NEURAL_PA_OUTCOMES[target_index] == label


def test_adapter_estimation_uses_latest_known_revision_without_event_resurrection() -> None:
    base = _observed_records()[0]
    records = list(_observed_records())
    records.extend(
        (
            replace(
                base,
                plate_appearance_id="late-publication",
                event=TerminalPlateAppearanceEvent.HIT_BY_PITCH,
                available_at=CUTOFF + timedelta(seconds=1),
            ),
            replace(
                base,
                plate_appearance_id="at-cutoff",
                event_at=CUTOFF,
                available_at=CUTOFF,
            ),
            replace(
                base,
                plate_appearance_id="corrected-event-time",
                event=TerminalPlateAppearanceEvent.WALK,
                available_at=CUTOFF - timedelta(days=2),
            ),
            replace(
                base,
                plate_appearance_id="corrected-event-time",
                event=TerminalPlateAppearanceEvent.HIT_BY_PITCH,
                event_at=CUTOFF,
                available_at=CUTOFF,
            ),
        )
    )

    config = NeuralTerminalAdapterConfig.estimate(
        records,
        cutoff_at=CUTOFF,
        source_fold_id="strict-cutoff-fold",
    )

    assert config.records_used == len(_observed_records())


def test_adapter_rejects_configuration_trained_after_prediction_cutoff() -> None:
    config = NeuralTerminalAdapterConfig.estimate(
        _observed_records(),
        cutoff_at=CUTOFF,
        source_fold_id="future-config-fold",
    )
    context = replace(_context(), cutoff_at=CUTOFF - timedelta(seconds=1))

    with pytest.raises(ValueError, match="training cutoff is later"):
        NeuralTerminalProbabilityAdapter(config).adapt(
            _neural_weights(strikeout=1.0),
            context,
        )


def test_adapter_legalizes_sacrifice_hit_without_runners() -> None:
    config = NeuralTerminalAdapterConfig.estimate(
        _observed_records(),
        cutoff_at=CUTOFF,
        source_fold_id="train-fold-1",
    )
    probabilities = NeuralTerminalProbabilityAdapter(config).adapt(
        _neural_weights(sacrifice_hit=1.0),
        _context(),
    )

    assert probabilities[TerminalPlateAppearanceEvent.SACRIFICE_BUNT] == 0.0
    assert probabilities[TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT] > 0.0
    assert sum(probabilities.values()) == pytest.approx(1.0)

    transition = StateTransitionEngine().apply(
        GameState(),
        TerminalPlateAppearanceEvent.SACRIFICE_BUNT,
        "batter",
        9,
        Random(1),
    )
    assert transition.applied_event is TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
    assert transition.state.outs == 1


def test_sacrifice_fly_requires_runner_on_third_and_fewer_than_two_outs() -> None:
    config = NeuralTerminalAdapterConfig.estimate(
        _observed_records(),
        cutoff_at=CUTOFF,
        source_fold_id="train-fold-1",
    )
    adapter = NeuralTerminalProbabilityAdapter(config)

    legal = adapter.adapt(
        _neural_weights(sacrifice_fly=1.0),
        _context(bases=BaseRunners(third="runner"), outs=1),
    )
    no_runner = adapter.adapt(
        _neural_weights(sacrifice_fly=1.0),
        _context(),
    )
    two_outs = adapter.adapt(
        _neural_weights(sacrifice_fly=1.0),
        _context(bases=BaseRunners(third="runner"), outs=2),
    )

    assert legal[TerminalPlateAppearanceEvent.SACRIFICE_FLY] > 0.0
    assert no_runner[TerminalPlateAppearanceEvent.SACRIFICE_FLY] == 0.0
    assert two_outs[TerminalPlateAppearanceEvent.SACRIFICE_FLY] == 0.0


def test_adapted_model_satisfies_probability_model_protocol() -> None:
    config = NeuralTerminalAdapterConfig.estimate(
        _observed_records(),
        cutoff_at=CUTOFF,
        source_fold_id="train-fold-1",
    )

    class Source:
        def predict_neural_proba(
            self,
            context: FuturePlateAppearanceContext,
        ) -> dict[str, float]:
            assert context.game_id == "future-game"
            return _neural_weights(strikeout=1.0)

    model = AdaptedPlateAppearanceProbabilityModel(
        Source(),
        NeuralTerminalProbabilityAdapter(config),
    )

    assert isinstance(model, PlateAppearanceProbabilityModel)
    probabilities = model.predict_proba(_context(outs=2))
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[TerminalPlateAppearanceEvent.STRIKEOUT] > 0.0
