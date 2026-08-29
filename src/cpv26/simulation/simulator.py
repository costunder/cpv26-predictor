"""Sequential Monte Carlo baseball game simulation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from random import Random
from statistics import fmean
from typing import Any

from cpv26.domain import utc_datetime

from .events import (
    FuturePlateAppearanceContext,
    SimulatedPlateAppearance,
    TerminalPlateAppearanceEvent,
)
from .probability import (
    PlateAppearanceProbabilityModel,
    normalize_event_probabilities,
    sample_terminal_event,
)
from .state import BaseRunners, GameState, HalfInning
from .transition import StateTransitionEngine, TransitionResult


@dataclass(frozen=True, slots=True)
class TeamLineup:
    team_id: str
    batter_ids: tuple[str, ...]
    catcher_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "batter_ids", tuple(self.batter_ids))
        if not self.team_id:
            raise ValueError("team_id must not be empty")
        if not self.batter_ids:
            raise ValueError("a lineup must contain at least one batter")
        if any(not batter_id for batter_id in self.batter_ids):
            raise ValueError("batter identifiers must not be empty")
        if len(set(self.batter_ids)) != len(self.batter_ids):
            raise ValueError("a lineup cannot contain the same player twice")


@dataclass(frozen=True, slots=True)
class PitchingPlan:
    """A reproducible starter/bullpen usage plan for one simulation path."""

    starter_id: str
    reliever_ids: tuple[str, ...] = ()
    starter_max_batters_faced: int = 24
    starter_through_inning: int = 6
    reliever_max_batters_faced: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "reliever_ids", tuple(self.reliever_ids))
        if not self.starter_id:
            raise ValueError("starter_id must not be empty")
        if any(not pitcher_id for pitcher_id in self.reliever_ids):
            raise ValueError("reliever identifiers must not be empty")
        if self.starter_id in self.reliever_ids:
            raise ValueError("the starter cannot also appear in the bullpen")
        if len(set(self.reliever_ids)) != len(self.reliever_ids):
            raise ValueError("a reliever cannot appear twice in a pitching plan")
        if self.starter_max_batters_faced < 1:
            raise ValueError("starter_max_batters_faced must be positive")
        if self.starter_through_inning < 1:
            raise ValueError("starter_through_inning must be positive")
        if self.reliever_max_batters_faced < 1:
            raise ValueError("reliever_max_batters_faced must be positive")


class _PitchingUsage:
    def __init__(self, plan: PitchingPlan) -> None:
        self.plan = plan
        self.reliever_index = -1
        self.current_batters_faced = 0

    def pitcher_for(self, inning: int) -> str:
        starter_due_for_removal = (
            self.reliever_index == -1
            and self.plan.reliever_ids
            and (
                self.current_batters_faced >= self.plan.starter_max_batters_faced
                or inning > self.plan.starter_through_inning
            )
        )
        reliever_due_for_removal = (
            self.reliever_index >= 0
            and self.reliever_index < len(self.plan.reliever_ids) - 1
            and self.current_batters_faced >= self.plan.reliever_max_batters_faced
        )
        if starter_due_for_removal:
            self.reliever_index = 0
            self.current_batters_faced = 0
        elif reliever_due_for_removal:
            self.reliever_index += 1
            self.current_batters_faced = 0

        if self.reliever_index == -1:
            return self.plan.starter_id
        return self.plan.reliever_ids[self.reliever_index]

    def record_plate_appearance(self) -> None:
        self.current_batters_faced += 1


@dataclass(frozen=True, slots=True)
class GameSimulationSpec:
    prediction_run_id: str
    cutoff_at: datetime
    game_id: str
    away_lineup: TeamLineup
    home_lineup: TeamLineup
    away_pitching_plan: PitchingPlan
    home_pitching_plan: PitchingPlan
    regulation_innings: int = 9
    max_innings: int = 12
    initial_state: GameState = GameState()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.prediction_run_id:
            raise ValueError("prediction_run_id must not be empty")
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.away_lineup.team_id == self.home_lineup.team_id:
            raise ValueError("home and away teams must be different")
        if self.regulation_innings < 1:
            raise ValueError("regulation_innings must be positive")
        if self.max_innings < self.regulation_innings:
            raise ValueError("max_innings cannot precede regulation innings")
        if self.initial_state.inning > self.max_innings:
            raise ValueError("initial state is beyond max_innings")
        if (
            self.initial_state.half is HalfInning.BOTTOM
            and self.initial_state.inning >= self.regulation_innings
            and self.initial_state.home_score > self.initial_state.away_score
        ):
            raise ValueError("initial state already represents a completed walk-off game")
        if self.initial_state.away_batting_index >= len(self.away_lineup.batter_ids):
            raise ValueError("away batting index is outside the lineup")
        if self.initial_state.home_batting_index >= len(self.home_lineup.batter_ids):
            raise ValueError("home batting index is outside the lineup")
        if set(self.away_lineup.batter_ids) & set(self.home_lineup.batter_ids):
            raise ValueError("the same player cannot appear for both teams")
        object.__setattr__(
            self,
            "cutoff_at",
            utc_datetime(self.cutoff_at, field_name="cutoff_at"),
        )


class GameEndReason(str, Enum):
    WALK_OFF = "walk_off"
    HOME_LEAD_AFTER_TOP = "home_lead_after_top"
    COMPLETED_INNINGS = "completed_innings"
    MAX_INNINGS_TIE = "max_innings_tie"


@dataclass(frozen=True, slots=True)
class SimulationOutcome:
    simulation_id: int
    game_id: str
    away_team_id: str
    home_team_id: str
    away_score: int
    home_score: int
    final_inning: int
    end_reason: GameEndReason
    player_hits: Mapping[str, int]
    plate_appearances: tuple[SimulatedPlateAppearance, ...]
    final_state: GameState

    def __post_init__(self) -> None:
        object.__setattr__(self, "player_hits", dict(self.player_hits))

    @property
    def winner_team_id(self) -> str | None:
        if self.away_score > self.home_score:
            return self.away_team_id
        if self.home_score > self.away_score:
            return self.home_team_id
        return None


@dataclass(frozen=True, slots=True)
class HitScenario:
    """One joint draw of player hit counts used by Live Hit optimization."""

    hits_by_player: Mapping[str, int]
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits_by_player", dict(self.hits_by_player))
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("scenario weight must be finite and positive")
        for player_id, hits in self.hits_by_player.items():
            if not player_id:
                raise ValueError("scenario player identifiers must not be empty")
            if isinstance(hits, bool) or not isinstance(hits, int) or hits < 0:
                raise ValueError("hit counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class SimulationBatch:
    game_id: str
    samples: tuple[SimulationOutcome, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("a simulation batch must contain at least one sample")
        if any(sample.game_id != self.game_id for sample in self.samples):
            raise ValueError("all samples in a batch must belong to the same game")

    @property
    def away_win_probability(self) -> float:
        return fmean(sample.away_score > sample.home_score for sample in self.samples)

    @property
    def home_win_probability(self) -> float:
        return fmean(sample.home_score > sample.away_score for sample in self.samples)

    @property
    def tie_probability(self) -> float:
        return fmean(sample.home_score == sample.away_score for sample in self.samples)

    @property
    def expected_away_score(self) -> float:
        return fmean(sample.away_score for sample in self.samples)

    @property
    def expected_home_score(self) -> float:
        return fmean(sample.home_score for sample in self.samples)

    def expected_hits(self, player_id: str) -> float:
        return fmean(sample.player_hits.get(player_id, 0) for sample in self.samples)

    def hit_probability(self, player_id: str) -> float:
        return fmean(sample.player_hits.get(player_id, 0) >= 1 for sample in self.samples)

    def joint_hit_probability(self, player_ids: Iterable[str]) -> float:
        selected = tuple(player_ids)
        if not selected:
            return 1.0
        return fmean(
            all(sample.player_hits.get(player_id, 0) >= 1 for player_id in selected)
            for sample in self.samples
        )

    def to_hit_scenarios(self) -> tuple[HitScenario, ...]:
        weight = 1.0 / len(self.samples)
        return tuple(HitScenario(sample.player_hits, weight=weight) for sample in self.samples)


class SimulationLimitError(RuntimeError):
    """Raised when a sampled path cannot terminate within its safety bound."""


class GameSimulator:
    """Simulate full games one terminal plate appearance at a time."""

    def __init__(
        self,
        probability_model: PlateAppearanceProbabilityModel,
        *,
        transition_engine: StateTransitionEngine | None = None,
        max_plate_appearances: int = 1_000,
        record_plate_appearances: bool = False,
    ) -> None:
        if max_plate_appearances < 1:
            raise ValueError("max_plate_appearances must be positive")
        self.probability_model = probability_model
        self.transition_engine = transition_engine or StateTransitionEngine()
        self.max_plate_appearances = max_plate_appearances
        self.record_plate_appearances = record_plate_appearances

    def simulate(
        self,
        spec: GameSimulationSpec,
        *,
        seed: int | None = None,
        simulation_id: int = 0,
    ) -> SimulationOutcome:
        rng = Random(seed)
        state = spec.initial_state
        away_pitching = _PitchingUsage(spec.away_pitching_plan)
        home_pitching = _PitchingUsage(spec.home_pitching_plan)
        player_hits = {
            player_id: 0
            for player_id in (
                *spec.away_lineup.batter_ids,
                *spec.home_lineup.batter_ids,
            )
        }
        records: list[SimulatedPlateAppearance] = []

        for pa_number in range(1, self.max_plate_appearances + 1):
            state_before = state
            if state.half is HalfInning.TOP:
                offense = spec.away_lineup
                defense = spec.home_lineup
                pitching_usage = home_pitching
            else:
                offense = spec.home_lineup
                defense = spec.away_lineup
                pitching_usage = away_pitching

            order_index = state.batting_index
            batter_id = offense.batter_ids[order_index]
            pitcher_id = pitching_usage.pitcher_for(state.inning)
            context = FuturePlateAppearanceContext(
                prediction_run_id=spec.prediction_run_id,
                cutoff_at=spec.cutoff_at,
                game_id=spec.game_id,
                plate_appearance_number=pa_number,
                batter_id=batter_id,
                pitcher_id=pitcher_id,
                catcher_id=defense.catcher_id,
                batter_team_id=offense.team_id,
                pitcher_team_id=defense.team_id,
                batting_order_index=order_index,
                state=state_before,
                metadata=spec.metadata,
            )
            probabilities = normalize_event_probabilities(
                self.probability_model.predict_proba(context),
                context,
            )
            sampled_event = sample_terminal_event(probabilities, rng)
            transition = self.transition_engine.apply(
                state_before,
                sampled_event,
                batter_id,
                len(offense.batter_ids),
                rng,
            )
            transition = self._cap_non_home_run_walk_off(
                spec,
                state_before,
                transition,
            )
            pitching_usage.record_plate_appearance()
            state = transition.state

            if transition.applied_event.is_hit:
                player_hits[batter_id] += 1
            if self.record_plate_appearances:
                records.append(
                    SimulatedPlateAppearance(
                        simulation_id=simulation_id,
                        plate_appearance_number=pa_number,
                        context=context,
                        sampled_event=sampled_event,
                        applied_event=transition.applied_event,
                        state_after=state,
                        runs_scored=transition.runs_scored,
                        outs_recorded=transition.outs_recorded,
                    )
                )

            end_reason = self._end_reason(spec, state_before, state, transition.half_inning_ended)
            if end_reason is not None:
                return SimulationOutcome(
                    simulation_id=simulation_id,
                    game_id=spec.game_id,
                    away_team_id=spec.away_lineup.team_id,
                    home_team_id=spec.home_lineup.team_id,
                    away_score=state.away_score,
                    home_score=state.home_score,
                    final_inning=state_before.inning,
                    end_reason=end_reason,
                    player_hits=dict(player_hits),
                    plate_appearances=tuple(records),
                    final_state=state,
                )

        raise SimulationLimitError(
            f"game {spec.game_id} exceeded {self.max_plate_appearances} plate appearances"
        )

    def simulate_many(
        self,
        spec: GameSimulationSpec,
        simulations: int,
        *,
        seed: int | None = None,
    ) -> SimulationBatch:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        seed_generator = Random(seed)
        samples = tuple(
            self.simulate(
                spec,
                seed=seed_generator.getrandbits(64),
                simulation_id=index,
            )
            for index in range(simulations)
        )
        return SimulationBatch(spec.game_id, samples)

    @staticmethod
    def _cap_non_home_run_walk_off(
        spec: GameSimulationSpec,
        state_before: GameState,
        transition: TransitionResult,
    ) -> TransitionResult:
        if (
            state_before.half is not HalfInning.BOTTOM
            or state_before.inning < spec.regulation_innings
            or transition.applied_event is TerminalPlateAppearanceEvent.HOME_RUN
            or transition.state.home_score <= transition.state.away_score
        ):
            return transition

        winning_score = state_before.away_score + 1
        credited_runs = winning_score - state_before.home_score
        if credited_runs <= 0:
            return transition
        counted_runs = min(transition.runs_scored, credited_runs)
        official_event = GameSimulator._official_walk_off_event(
            state_before,
            transition,
            counted_runs,
        )
        # A walk-off outcome is terminal, so the post-play bases are deliberately
        # cleared instead of exposing the full, uncapped advancement generated
        # before the winning run ended the game. ``applied_event`` carries the
        # scorer-credited hit value; ``sampled_event`` retains the physical draw.
        corrected_state = replace(
            transition.state,
            bases=BaseRunners(),
            home_score=state_before.home_score + counted_runs,
        )
        return replace(
            transition,
            state=corrected_state,
            applied_event=official_event,
            runs_scored=counted_runs,
        )

    @staticmethod
    def _official_walk_off_event(
        state_before: GameState,
        transition: TransitionResult,
        counted_runs: int,
    ) -> TerminalPlateAppearanceEvent:
        """Apply official non-HR game-ending hit credit.

        A batter receives no more bases than the winning runner advanced from
        their starting base. The sampled event remains available separately on
        ``SimulatedPlateAppearance`` for model diagnostics.
        """

        event = transition.applied_event
        if not event.is_hit or event is TerminalPlateAppearanceEvent.HOME_RUN:
            return event

        surviving_runners = {
            runner for runner in transition.state.bases.as_tuple() if runner is not None
        }
        scoring_origins = tuple(
            base_number
            for base_number, runner in (
                (3, state_before.bases.third),
                (2, state_before.bases.second),
                (1, state_before.bases.first),
            )
            if runner is not None and runner not in surviving_runners
        )
        if not 1 <= counted_runs <= len(scoring_origins):
            raise RuntimeError("walk-off hit transition lost scoring-runner lineage")

        winning_runner_origin = scoring_origins[counted_runs - 1]
        credited_bases = min(event.hit_value, 4 - winning_runner_origin)
        return {
            1: TerminalPlateAppearanceEvent.SINGLE,
            2: TerminalPlateAppearanceEvent.DOUBLE,
            3: TerminalPlateAppearanceEvent.TRIPLE,
        }[credited_bases]

    @staticmethod
    def _end_reason(
        spec: GameSimulationSpec,
        state_before: GameState,
        state_after: GameState,
        half_inning_ended: bool,
    ) -> GameEndReason | None:
        inning = state_before.inning
        half = state_before.half

        if (
            half is HalfInning.BOTTOM
            and inning >= spec.regulation_innings
            and state_before.home_score <= state_before.away_score
            and state_after.home_score > state_after.away_score
        ):
            return GameEndReason.WALK_OFF
        if not half_inning_ended or inning < spec.regulation_innings:
            return None
        if half is HalfInning.TOP and state_after.home_score > state_after.away_score:
            return GameEndReason.HOME_LEAD_AFTER_TOP
        if half is HalfInning.BOTTOM:
            if state_after.home_score != state_after.away_score:
                return GameEndReason.COMPLETED_INNINGS
            if inning >= spec.max_innings:
                return GameEndReason.MAX_INNINGS_TIE
        return None


def combine_hit_scenarios(
    batches: Iterable[SimulationBatch],
    *,
    scenario_count: int | None = None,
    seed: int | None = None,
) -> tuple[HitScenario, ...]:
    """Combine independent games while preserving within-game hit dependence."""

    batch_tuple = tuple(batches)
    if not batch_tuple:
        raise ValueError("at least one simulation batch is required")
    if len(batch_tuple) == 1 and scenario_count is None:
        return batch_tuple[0].to_hit_scenarios()
    if scenario_count is None:
        scenario_count = max(len(batch.samples) for batch in batch_tuple)
    if scenario_count < 1:
        raise ValueError("scenario_count must be positive")

    rng = Random(seed)
    scenarios: list[HitScenario] = []
    for _ in range(scenario_count):
        combined: dict[str, int] = {}
        for batch in batch_tuple:
            sample = batch.samples[rng.randrange(len(batch.samples))]
            for player_id, hits in sample.player_hits.items():
                combined[player_id] = combined.get(player_id, 0) + hits
        scenarios.append(HitScenario(combined, weight=1.0 / scenario_count))
    return tuple(scenarios)
