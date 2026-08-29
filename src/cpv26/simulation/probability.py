"""Probability-model contract and leakage-safe empirical implementation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import Protocol, runtime_checkable

from cpv26.domain import utc_datetime

from .events import (
    FuturePlateAppearanceContext,
    ObservedPlateAppearance,
    TerminalPlateAppearanceEvent,
)

EventProbabilities = Mapping[TerminalPlateAppearanceEvent, float]


@runtime_checkable
class PlateAppearanceProbabilityModel(Protocol):
    """A model that scores a future context without receiving its outcome."""

    def predict_proba(
        self,
        context: FuturePlateAppearanceContext,
    ) -> EventProbabilities:
        """Return non-negative weights or probabilities for terminal events."""

        ...


DEFAULT_EVENT_PRIOR: dict[TerminalPlateAppearanceEvent, float] = {
    TerminalPlateAppearanceEvent.SINGLE: 0.155,
    TerminalPlateAppearanceEvent.DOUBLE: 0.045,
    TerminalPlateAppearanceEvent.TRIPLE: 0.004,
    TerminalPlateAppearanceEvent.HOME_RUN: 0.030,
    TerminalPlateAppearanceEvent.WALK: 0.085,
    TerminalPlateAppearanceEvent.HIT_BY_PITCH: 0.011,
    TerminalPlateAppearanceEvent.STRIKEOUT: 0.190,
    TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT: 0.380,
    TerminalPlateAppearanceEvent.DOUBLE_PLAY: 0.022,
    TerminalPlateAppearanceEvent.SACRIFICE_FLY: 0.015,
    TerminalPlateAppearanceEvent.SACRIFICE_BUNT: 0.007,
    TerminalPlateAppearanceEvent.REACHED_ON_ERROR: 0.013,
    TerminalPlateAppearanceEvent.FIELDERS_CHOICE: 0.025,
    TerminalPlateAppearanceEvent.CATCHER_INTERFERENCE: 0.001,
}


def normalize_event_probabilities(
    weights: EventProbabilities,
    context: FuturePlateAppearanceContext | None = None,
) -> dict[TerminalPlateAppearanceEvent, float]:
    """Validate, legalize for the current state, and normalize event weights."""

    normalized_weights = {
        event: float(weights.get(event, 0.0)) for event in TerminalPlateAppearanceEvent
    }
    for event, value in normalized_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid probability weight for {event.value}: {value}")

    if context is not None:
        state = context.state
        illegal: list[TerminalPlateAppearanceEvent] = []
        if state.outs >= 2 or state.bases.first is None:
            illegal.append(TerminalPlateAppearanceEvent.DOUBLE_PLAY)
        if state.outs >= 2 or state.bases.third is None:
            illegal.append(TerminalPlateAppearanceEvent.SACRIFICE_FLY)
        if state.outs >= 2 or state.bases.is_empty:
            illegal.append(TerminalPlateAppearanceEvent.SACRIFICE_BUNT)
        if state.bases.is_empty:
            illegal.append(TerminalPlateAppearanceEvent.FIELDERS_CHOICE)

        transferred = sum(normalized_weights[event] for event in illegal)
        for event in illegal:
            normalized_weights[event] = 0.0
        normalized_weights[TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT] += transferred

    total = sum(normalized_weights.values())
    if total <= 0.0:
        raise ValueError("the probability model returned no positive event weight")
    return {event: value / total for event, value in normalized_weights.items()}


def sample_terminal_event(
    probabilities: EventProbabilities,
    rng: Random,
) -> TerminalPlateAppearanceEvent:
    """Sample from a normalized distribution using a caller-owned RNG."""

    threshold = rng.random()
    cumulative = 0.0
    last_event = TerminalPlateAppearanceEvent.BALL_IN_PLAY_OUT
    for event in TerminalPlateAppearanceEvent:
        last_event = event
        cumulative += probabilities.get(event, 0.0)
        if threshold < cumulative:
            return event
    return last_event


@dataclass(frozen=True, slots=True)
class StaticPlateAppearanceProbabilityModel:
    """A deterministic distribution provider, useful for calibrated baselines."""

    probabilities: EventProbabilities

    def __post_init__(self) -> None:
        normalized = normalize_event_probabilities(self.probabilities)
        object.__setattr__(self, "probabilities", normalized)

    def predict_proba(
        self,
        context: FuturePlateAppearanceContext,
    ) -> EventProbabilities:
        return self.probabilities


class EmpiricalPlateAppearanceProbabilityModel:
    """Hierarchically smoothed batter/pitcher/matchup event probabilities.

    This baseline is intentionally fitted from ``ObservedPlateAppearance``
    instances whose timestamps precede ``cutoff_at``.  It is useful both as a
    standalone model and as a deterministic simulator smoke-test model.
    """

    def __init__(
        self,
        *,
        cutoff_at: datetime,
        league_counts: Counter[TerminalPlateAppearanceEvent],
        batter_counts: Mapping[str, Counter[TerminalPlateAppearanceEvent]],
        pitcher_counts: Mapping[str, Counter[TerminalPlateAppearanceEvent]],
        matchup_counts: Mapping[tuple[str, str], Counter[TerminalPlateAppearanceEvent]],
        prior: EventProbabilities,
        league_prior_strength: float,
        player_prior_strength: float,
        matchup_prior_strength: float,
    ) -> None:
        self.cutoff_at = utc_datetime(cutoff_at, field_name="cutoff_at")
        self._league_counts = league_counts
        self._batter_counts = dict(batter_counts)
        self._pitcher_counts = dict(pitcher_counts)
        self._matchup_counts = dict(matchup_counts)
        self._prior = normalize_event_probabilities(prior)
        self._league_prior_strength = league_prior_strength
        self._player_prior_strength = player_prior_strength
        self._matchup_prior_strength = matchup_prior_strength

    @classmethod
    def fit(
        cls,
        records: Iterable[ObservedPlateAppearance],
        *,
        cutoff_at: datetime,
        prior: EventProbabilities | None = None,
        league_prior_strength: float = 200.0,
        player_prior_strength: float = 80.0,
        matchup_prior_strength: float = 30.0,
    ) -> EmpiricalPlateAppearanceProbabilityModel:
        cutoff_at = utc_datetime(cutoff_at, field_name="cutoff_at")
        for name, value in {
            "league_prior_strength": league_prior_strength,
            "player_prior_strength": player_prior_strength,
            "matchup_prior_strength": matchup_prior_strength,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        league: Counter[TerminalPlateAppearanceEvent] = Counter()
        batters: defaultdict[str, Counter[TerminalPlateAppearanceEvent]] = defaultdict(Counter)
        pitchers: defaultdict[str, Counter[TerminalPlateAppearanceEvent]] = defaultdict(Counter)
        matchups: defaultdict[tuple[str, str], Counter[TerminalPlateAppearanceEvent]] = defaultdict(
            Counter
        )

        for record in records:
            if record.event_at >= cutoff_at or record.available_at > cutoff_at:
                continue
            league[record.event] += 1
            batters[record.batter_id][record.event] += 1
            pitchers[record.pitcher_id][record.event] += 1
            matchups[(record.batter_id, record.pitcher_id)][record.event] += 1

        return cls(
            cutoff_at=cutoff_at,
            league_counts=league,
            batter_counts=batters,
            pitcher_counts=pitchers,
            matchup_counts=matchups,
            prior=prior or DEFAULT_EVENT_PRIOR,
            league_prior_strength=league_prior_strength,
            player_prior_strength=player_prior_strength,
            matchup_prior_strength=matchup_prior_strength,
        )

    def predict_proba(
        self,
        context: FuturePlateAppearanceContext,
    ) -> EventProbabilities:
        if self.cutoff_at > context.cutoff_at:
            raise ValueError(
                "model cutoff is later than the prediction cutoff and would leak "
                "future observations"
            )

        league = self._posterior(
            self._league_counts,
            self._prior,
            self._league_prior_strength,
        )
        batter = self._posterior(
            self._batter_counts.get(context.batter_id, Counter()),
            league,
            self._player_prior_strength,
        )
        pitcher = self._posterior(
            self._pitcher_counts.get(context.pitcher_id, Counter()),
            league,
            self._player_prior_strength,
        )
        player_prior = {
            event: 0.5 * batter[event] + 0.5 * pitcher[event]
            for event in TerminalPlateAppearanceEvent
        }
        return self._posterior(
            self._matchup_counts.get((context.batter_id, context.pitcher_id), Counter()),
            player_prior,
            self._matchup_prior_strength,
        )

    @staticmethod
    def _posterior(
        counts: Mapping[TerminalPlateAppearanceEvent, int],
        prior: EventProbabilities,
        prior_strength: float,
    ) -> dict[TerminalPlateAppearanceEvent, float]:
        total = sum(counts.values()) + prior_strength
        return {
            event: (counts.get(event, 0) + prior_strength * prior[event]) / total
            for event in TerminalPlateAppearanceEvent
        }
