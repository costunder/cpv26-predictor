"""Adapt neural PA categories to state-conditional simulator outcomes."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from cpv26.domain import utc_datetime

from .events import (
    FuturePlateAppearanceContext,
    ObservedPlateAppearance,
    TerminalPlateAppearanceEvent,
)
from .probability import EventProbabilities, normalize_event_probabilities
from .state import GameState

NEURAL_PA_OUTCOMES: tuple[str, ...] = (
    "strikeout",
    "walk_or_hbp",
    "single",
    "double",
    "triple",
    "home_run",
    "ball_in_play_out",
    "reached_on_error",
    "sacrifice_hit",
    "sacrifice_fly",
)

NEURAL_PA_OUTCOME_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(NEURAL_PA_OUTCOMES)}
)

TERMINAL_TO_NEURAL_TARGET: Mapping[TerminalPlateAppearanceEvent, str | None] = (
    MappingProxyType(
        {
            TerminalPlateAppearanceEvent.STRIKEOUT: "strikeout",
            TerminalPlateAppearanceEvent.WALK: "walk_or_hbp",
            TerminalPlateAppearanceEvent.HIT_BY_PITCH: "walk_or_hbp",
            TerminalPlateAppearanceEvent.SINGLE: "single",
            TerminalPlateAppearanceEvent.DOUBLE: "double",
            TerminalPlateAppearanceEvent.TRIPLE: "triple",
            TerminalPlateAppearanceEvent.HOME_RUN: "home_run",
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT: "ball_in_play_out",
            TerminalPlateAppearanceEvent.DOUBLE_PLAY: "ball_in_play_out",
            TerminalPlateAppearanceEvent.FIELDERS_CHOICE: "ball_in_play_out",
            TerminalPlateAppearanceEvent.REACHED_ON_ERROR: "reached_on_error",
            TerminalPlateAppearanceEvent.SACRIFICE_BUNT: "sacrifice_hit",
            TerminalPlateAppearanceEvent.SACRIFICE_FLY: "sacrifice_fly",
            # Catcher interference is estimated as a separate rare-event rate by the
            # adapter, so it must not be folded into the decoder's ten-way target.
            TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE: None,
        }
    )
)


def neural_training_target(event: TerminalPlateAppearanceEvent) -> str | None:
    """Map one simulator terminal event to the neural decoder target.

    ``None`` means that the record trains the adapter's rare-event component,
    not the ten-way decoder. This explicit contract prevents data loaders from
    assigning HBP, double plays, fielder's choices, or catcher interference
    differently across training jobs.
    """

    return TERMINAL_TO_NEURAL_TARGET[event]


def neural_training_target_index(event: TerminalPlateAppearanceEvent) -> int | None:
    """Encode one terminal event for ``PATargets.outcome_index``.

    Catcher interference returns ``None`` and must be excluded from the
    decoder cross-entropy batch while remaining available to adapter-rate
    estimation.
    """

    label = neural_training_target(event)
    return None if label is None else NEURAL_PA_OUTCOME_TO_INDEX[label]


@dataclass(frozen=True, slots=True)
class BattedBallOutSplit:
    """Conditional shares within the neural ``ball_in_play_out`` category."""

    ordinary_out: float
    double_play: float
    fielders_choice: float

    def __post_init__(self) -> None:
        values = (self.ordinary_out, self.double_play, self.fielders_choice)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("batted-ball split values must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("batted-ball split values must sum to one")


def terminal_state_bucket(state: GameState) -> str:
    """Return the stable key used to estimate conditional BIP transitions."""

    return (
        f"outs={state.outs}|first={int(state.bases.first is not None)}"
        f"|runners={int(not state.bases.is_empty)}"
    )


def _observed_state_bucket(record: ObservedPlateAppearance) -> str | None:
    if record.outs_before is None or record.bases_before is None:
        return None
    state = GameState(outs=record.outs_before, bases=record.bases_before)
    return terminal_state_bucket(state)


@dataclass(frozen=True, slots=True)
class NeuralTerminalAdapterConfig:
    """Train/fold-derived conditional decomposition of neural PA categories."""

    training_cutoff_at: datetime
    source_fold_id: str
    records_used: int
    hit_by_pitch_share: float
    catcher_interference_rate: float
    global_batted_ball_split: BattedBallOutSplit
    batted_ball_split_by_state: Mapping[str, BattedBallOutSplit] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_cutoff_at",
            utc_datetime(self.training_cutoff_at, field_name="training_cutoff_at"),
        )
        object.__setattr__(
            self,
            "batted_ball_split_by_state",
            dict(self.batted_ball_split_by_state),
        )
        if not self.source_fold_id:
            raise ValueError("source_fold_id must identify the training fold")
        if self.records_used < 1:
            raise ValueError("records_used must be positive")
        if not 0.0 <= self.hit_by_pitch_share <= 1.0:
            raise ValueError("hit_by_pitch_share must be between zero and one")
        if not 0.0 <= self.catcher_interference_rate < 1.0:
            raise ValueError("catcher_interference_rate must be in [0, 1)")

    @classmethod
    def estimate(
        cls,
        records: Iterable[ObservedPlateAppearance],
        *,
        cutoff_at: datetime,
        source_fold_id: str,
        pseudocount: float = 0.5,
        state_prior_strength: float = 20.0,
    ) -> NeuralTerminalAdapterConfig:
        """Estimate every ambiguous split using only records known at cutoff."""

        cutoff = utc_datetime(cutoff_at, field_name="cutoff_at")
        if not math.isfinite(pseudocount) or pseudocount <= 0.0:
            raise ValueError("pseudocount must be finite and positive")
        if not math.isfinite(state_prior_strength) or state_prior_strength <= 0.0:
            raise ValueError("state_prior_strength must be finite and positive")

        known_revisions = tuple(record for record in records if record.available_at <= cutoff)
        latest_by_plate_appearance: dict[str, ObservedPlateAppearance] = {}
        for record in known_revisions:
            previous = latest_by_plate_appearance.get(record.plate_appearance_id)
            if previous is None or record.available_at > previous.available_at:
                latest_by_plate_appearance[record.plate_appearance_id] = record
            elif record.available_at == previous.available_at and record != previous:
                raise ValueError(
                    "ambiguous plate-appearance revisions share available_at: "
                    f"{record.plate_appearance_id}"
                )
        # Event-time filtering follows correction selection so a newer revision
        # that moves an event to/after the cutoff cannot resurrect an older row.
        usable = tuple(
            record
            for record in latest_by_plate_appearance.values()
            if record.event_at < cutoff
        )
        if not usable:
            raise ValueError("no observed plate appearances are available at cutoff")

        event_counts = Counter(record.event for record in usable)
        walk_hbp_total = (
            event_counts[TerminalPlateAppearanceEvent.WALK]
            + event_counts[TerminalPlateAppearanceEvent.HIT_BY_PITCH]
        )
        hbp_share = (
            event_counts[TerminalPlateAppearanceEvent.HIT_BY_PITCH] + pseudocount
        ) / (walk_hbp_total + 2.0 * pseudocount)
        ci_rate = (
            event_counts[TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE]
            + pseudocount
        ) / (len(usable) + 2.0 * pseudocount)

        split_events = (
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
            TerminalPlateAppearanceEvent.DOUBLE_PLAY,
            TerminalPlateAppearanceEvent.FIELDERS_CHOICE,
        )
        global_counts = Counter(
            record.event for record in usable if record.event in split_events
        )
        global_split = cls._smoothed_split(global_counts, pseudocount)

        bucket_counts: defaultdict[
            str, Counter[TerminalPlateAppearanceEvent]
        ] = defaultdict(Counter)
        for record in usable:
            if record.event not in split_events:
                continue
            bucket = _observed_state_bucket(record)
            if bucket is not None:
                bucket_counts[bucket][record.event] += 1

        by_state = {
            bucket: cls._hierarchical_split(
                counts,
                global_split,
                state_prior_strength,
            )
            for bucket, counts in bucket_counts.items()
        }
        return cls(
            training_cutoff_at=cutoff,
            source_fold_id=source_fold_id,
            records_used=len(usable),
            hit_by_pitch_share=hbp_share,
            catcher_interference_rate=ci_rate,
            global_batted_ball_split=global_split,
            batted_ball_split_by_state=by_state,
        )

    @staticmethod
    def _smoothed_split(
        counts: Mapping[TerminalPlateAppearanceEvent, int],
        pseudocount: float,
    ) -> BattedBallOutSplit:
        events = (
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
            TerminalPlateAppearanceEvent.DOUBLE_PLAY,
            TerminalPlateAppearanceEvent.FIELDERS_CHOICE,
        )
        denominator = sum(counts.get(event, 0) for event in events) + 3 * pseudocount
        shares = tuple(
            (counts.get(event, 0) + pseudocount) / denominator for event in events
        )
        return BattedBallOutSplit(*shares)

    @staticmethod
    def _hierarchical_split(
        counts: Mapping[TerminalPlateAppearanceEvent, int],
        prior: BattedBallOutSplit,
        prior_strength: float,
    ) -> BattedBallOutSplit:
        events = (
            TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT,
            TerminalPlateAppearanceEvent.DOUBLE_PLAY,
            TerminalPlateAppearanceEvent.FIELDERS_CHOICE,
        )
        prior_shares = (
            prior.ordinary_out,
            prior.double_play,
            prior.fielders_choice,
        )
        denominator = sum(counts.get(event, 0) for event in events) + prior_strength
        shares = tuple(
            (counts.get(event, 0) + prior_strength * prior_share) / denominator
            for event, prior_share in zip(events, prior_shares, strict=True)
        )
        return BattedBallOutSplit(*shares)


class NeuralTerminalProbabilityAdapter:
    """Convert the 10 neural categories into 14 legal terminal outcomes."""

    def __init__(self, config: NeuralTerminalAdapterConfig) -> None:
        self.config = config

    def adapt(
        self,
        neural_probabilities: Mapping[str, float],
        context: FuturePlateAppearanceContext,
    ) -> dict[TerminalPlateAppearanceEvent, float]:
        if self.config.training_cutoff_at > context.cutoff_at:
            raise ValueError(
                "adapter training cutoff is later than the prediction cutoff"
            )
        labels = set(neural_probabilities)
        required = set(NEURAL_PA_OUTCOMES)
        if labels != required:
            raise ValueError(
                "neural outcome labels differ from the adapter contract; "
                f"missing={sorted(required - labels)}, extra={sorted(labels - required)}"
            )
        weights = {label: float(neural_probabilities[label]) for label in NEURAL_PA_OUTCOMES}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("neural probabilities must be finite and non-negative")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("neural probabilities must have positive total mass")
        neural = {label: value / total for label, value in weights.items()}

        ci_rate = self.config.catcher_interference_rate
        scale = 1.0 - ci_rate
        hbp_share = self.config.hit_by_pitch_share
        walk_mass = scale * neural["walk_or_hbp"]
        batted_ball_mass = scale * neural["ball_in_play_out"]
        split = self.config.batted_ball_split_by_state.get(
            terminal_state_bucket(context.state),
            self.config.global_batted_ball_split,
        )

        terminal: dict[TerminalPlateAppearanceEvent, float] = {
            event: 0.0 for event in TerminalPlateAppearanceEvent
        }
        terminal[TerminalPlateAppearanceEvent.STRIKEOUT] = scale * neural["strikeout"]
        terminal[TerminalPlateAppearanceEvent.WALK] = walk_mass * (1.0 - hbp_share)
        terminal[TerminalPlateAppearanceEvent.HIT_BY_PITCH] = walk_mass * hbp_share
        terminal[TerminalPlateAppearanceEvent.SINGLE] = scale * neural["single"]
        terminal[TerminalPlateAppearanceEvent.DOUBLE] = scale * neural["double"]
        terminal[TerminalPlateAppearanceEvent.TRIPLE] = scale * neural["triple"]
        terminal[TerminalPlateAppearanceEvent.HOME_RUN] = scale * neural["home_run"]
        terminal[TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT] = (
            batted_ball_mass * split.ordinary_out
        )
        terminal[TerminalPlateAppearanceEvent.DOUBLE_PLAY] = (
            batted_ball_mass * split.double_play
        )
        terminal[TerminalPlateAppearanceEvent.FIELDERS_CHOICE] = (
            batted_ball_mass * split.fielders_choice
        )
        terminal[TerminalPlateAppearanceEvent.REACHED_ON_ERROR] = (
            scale * neural["reached_on_error"]
        )
        terminal[TerminalPlateAppearanceEvent.SACRIFICE_BUNT] = (
            scale * neural["sacrifice_hit"]
        )
        terminal[TerminalPlateAppearanceEvent.SACRIFICE_FLY] = (
            scale * neural["sacrifice_fly"]
        )
        terminal[TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE] = ci_rate
        return normalize_event_probabilities(terminal, context)


@runtime_checkable
class NeuralPlateAppearanceProbabilitySource(Protocol):
    """Application adapter that supplies neural probabilities for a context."""

    def predict_neural_proba(
        self,
        context: FuturePlateAppearanceContext,
    ) -> Mapping[str, float]: ...


class AdaptedPlateAppearanceProbabilityModel:
    """PlateAppearanceProbabilityModel-compatible neural/simulator bridge."""

    def __init__(
        self,
        source: NeuralPlateAppearanceProbabilitySource,
        adapter: NeuralTerminalProbabilityAdapter,
    ) -> None:
        self.source = source
        self.adapter = adapter

    def predict_proba(
        self,
        context: FuturePlateAppearanceContext,
    ) -> EventProbabilities:
        return self.adapter.adapt(self.source.predict_neural_proba(context), context)
