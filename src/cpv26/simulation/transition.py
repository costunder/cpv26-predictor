"""Terminal plate-appearance outcomes and baseball state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .events import TerminalPlateAppearanceEvent
from .state import BaseRunners, GameState


@dataclass(frozen=True, slots=True)
class RunnerAdvancementRates:
    """Empirical advancement rates for outcomes that do not fix every base."""

    runner_on_second_scores_on_single: float = 0.62
    runner_on_first_reaches_third_on_single: float = 0.28
    runner_on_first_scores_on_double: float = 0.45
    runner_on_second_scores_on_error: float = 0.55
    runner_on_first_reaches_third_on_error: float = 0.25
    runner_on_second_reaches_third_on_sacrifice_fly: float = 0.35

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: GameState
    sampled_event: TerminalPlateAppearanceEvent
    applied_event: TerminalPlateAppearanceEvent
    runs_scored: int
    outs_recorded: int
    half_inning_ended: bool
    completed_inning: int


class StateTransitionEngine:
    """Advance an immutable :class:`GameState` by one terminal event."""

    def __init__(self, advancement_rates: RunnerAdvancementRates | None = None) -> None:
        self.advancement_rates = advancement_rates or RunnerAdvancementRates()

    def apply(
        self,
        state: GameState,
        sampled_event: TerminalPlateAppearanceEvent,
        batter_id: str,
        lineup_size: int,
        rng: Random,
    ) -> TransitionResult:
        if not batter_id:
            raise ValueError("batter_id must not be empty")
        if batter_id in state.bases.as_tuple():
            raise ValueError("the current batter is already on base")

        event = self._resolve_illegal_event(state, sampled_event)
        bases, runs, outs = self._advance_runners(state, event, batter_id, rng)
        total_outs = state.outs + outs
        ended = total_outs >= 3

        advanced = state.with_bases_and_outs(
            BaseRunners() if ended else bases,
            total_outs,
            runs_scored=runs,
        ).advance_batter(lineup_size)
        if ended:
            advanced = advanced.next_half_inning()

        return TransitionResult(
            state=advanced,
            sampled_event=sampled_event,
            applied_event=event,
            runs_scored=runs,
            outs_recorded=outs,
            half_inning_ended=ended,
            completed_inning=state.inning,
        )

    @staticmethod
    def _resolve_illegal_event(
        state: GameState,
        event: TerminalPlateAppearanceEvent,
    ) -> TerminalPlateAppearanceEvent:
        if event is TerminalPlateAppearanceEvent.DOUBLE_PLAY and (
            state.outs >= 2 or state.bases.first is None
        ):
            return TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
        if event is TerminalPlateAppearanceEvent.SACRIFICE_FLY and (
            state.outs >= 2 or state.bases.third is None
        ):
            return TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
        if event is TerminalPlateAppearanceEvent.SACRIFICE_BUNT and (
            state.outs >= 2 or state.bases.is_empty
        ):
            return TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
        if event is TerminalPlateAppearanceEvent.FIELDERS_CHOICE and state.bases.is_empty:
            return TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
        return event

    def _advance_runners(
        self,
        state: GameState,
        event: TerminalPlateAppearanceEvent,
        batter_id: str,
        rng: Random,
    ) -> tuple[BaseRunners, int, int]:
        bases = state.bases
        if event is TerminalPlateAppearanceEvent.SINGLE:
            return self._single(
                bases,
                batter_id,
                rng,
                self.advancement_rates.runner_on_second_scores_on_single,
                self.advancement_rates.runner_on_first_reaches_third_on_single,
            )
        if event is TerminalPlateAppearanceEvent.DOUBLE:
            runs = int(bases.third is not None) + int(bases.second is not None)
            new_third: str | None = None
            if bases.first is not None:
                if rng.random() < self.advancement_rates.runner_on_first_scores_on_double:
                    runs += 1
                else:
                    new_third = bases.first
            return BaseRunners(second=batter_id, third=new_third), runs, 0
        if event is TerminalPlateAppearanceEvent.TRIPLE:
            return BaseRunners(third=batter_id), bases.count, 0
        if event is TerminalPlateAppearanceEvent.HOME_RUN:
            return BaseRunners(), bases.count + 1, 0
        if event in {
            TerminalPlateAppearanceEvent.WALK,
            TerminalPlateAppearanceEvent.HIT_BY_PITCH,
            TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE,
        }:
            return self._forced_advance(bases, batter_id)
        if event in {
            TerminalPlateAppearanceEvent.STRIKEOUT,
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
        }:
            return bases, 0, 1
        if event is TerminalPlateAppearanceEvent.DOUBLE_PLAY:
            if state.outs == 0:
                return (
                    BaseRunners(third=bases.second),
                    int(bases.third is not None),
                    2,
                )
            return BaseRunners(), 0, 2
        if event is TerminalPlateAppearanceEvent.SACRIFICE_FLY:
            new_third = (
                bases.second
                if (
                    bases.second is not None
                    and rng.random()
                    < self.advancement_rates.runner_on_second_reaches_third_on_sacrifice_fly
                )
                else None
            )
            new_second = None if new_third is not None else bases.second
            return (
                BaseRunners(first=bases.first, second=new_second, third=new_third),
                1,
                1,
            )
        if event is TerminalPlateAppearanceEvent.SACRIFICE_BUNT:
            return (
                BaseRunners(second=bases.first, third=bases.second),
                int(bases.third is not None),
                1,
            )
        if event is TerminalPlateAppearanceEvent.REACHED_ON_ERROR:
            return self._single(
                bases,
                batter_id,
                rng,
                self.advancement_rates.runner_on_second_scores_on_error,
                self.advancement_rates.runner_on_first_reaches_third_on_error,
            )
        if event is TerminalPlateAppearanceEvent.FIELDERS_CHOICE:
            if bases.first is not None:
                return BaseRunners(batter_id, bases.second, bases.third), 0, 1
            if bases.second is not None:
                return BaseRunners(batter_id, None, bases.third), 0, 1
            return BaseRunners(first=batter_id), 0, 1
        raise ValueError(f"unsupported terminal event: {event}")

    @staticmethod
    def _forced_advance(
        bases: BaseRunners,
        batter_id: str,
    ) -> tuple[BaseRunners, int, int]:
        if bases.first is None:
            return BaseRunners(batter_id, bases.second, bases.third), 0, 0
        if bases.second is None:
            return BaseRunners(batter_id, bases.first, bases.third), 0, 0
        runs = int(bases.third is not None)
        return BaseRunners(batter_id, bases.first, bases.second), runs, 0

    @staticmethod
    def _single(
        bases: BaseRunners,
        batter_id: str,
        rng: Random,
        second_scores_rate: float,
        first_to_third_rate: float,
    ) -> tuple[BaseRunners, int, int]:
        runs = int(bases.third is not None)
        new_third: str | None = None

        if bases.second is not None:
            if rng.random() < second_scores_rate:
                runs += 1
            else:
                new_third = bases.second

        new_second: str | None = None
        if bases.first is not None:
            if new_third is None and rng.random() < first_to_third_rate:
                new_third = bases.first
            else:
                new_second = bases.first

        return BaseRunners(batter_id, new_second, new_third), runs, 0
