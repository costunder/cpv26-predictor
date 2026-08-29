"""Plate-appearance records used by training and forward simulation.

Observed records deliberately carry an outcome and an observation timestamp. A
future context never carries either field, which makes it difficult to leak a
target into a model through the simulator API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from cpv26.domain import utc_datetime

from .state import BaseRunners, GameState


class TerminalPlateAppearanceEvent(str, Enum):
    """Mutually exclusive terminal outcomes understood by the state engine."""

    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"
    WALK = "walk"
    HIT_BY_PITCH = "hit_by_pitch"
    STRIKEOUT = "strikeout"
    BALL_IN_PLAY_OUT = "ball_in_play_out"
    DOUBLE_PLAY = "double_play"
    SACRIFICE_FLY = "sacrifice_fly"
    SACRIFICE_BUNT = "sacrifice_bunt"
    REACHED_ON_ERROR = "reached_on_error"
    FIELDERS_CHOICE = "fielders_choice"
    CATCHER_INTERFERENCE = "catcher_interference"

    @property
    def is_hit(self) -> bool:
        return self in {
            self.SINGLE,
            self.DOUBLE,
            self.TRIPLE,
            self.HOME_RUN,
        }

    @property
    def hit_value(self) -> int:
        return {
            self.SINGLE: 1,
            self.DOUBLE: 2,
            self.TRIPLE: 3,
            self.HOME_RUN: 4,
        }.get(self, 0)


@dataclass(frozen=True, slots=True)
class ObservedPlateAppearance:
    """An immutable, historical plate appearance available for model fitting."""

    game_id: str
    plate_appearance_id: str
    event_at: datetime
    available_at: datetime
    batter_id: str
    pitcher_id: str
    batter_team_id: str
    pitcher_team_id: str
    event: TerminalPlateAppearanceEvent
    catcher_id: str | None = None
    inning: int | None = None
    outs_before: int | None = None
    bases_before: BaseRunners | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        for name in (
            "game_id",
            "plate_appearance_id",
            "batter_id",
            "pitcher_id",
            "batter_team_id",
            "pitcher_team_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.inning is not None and self.inning < 1:
            raise ValueError("inning must be at least 1")
        if self.outs_before is not None and not 0 <= self.outs_before <= 2:
            raise ValueError("outs_before must be between 0 and 2")
        object.__setattr__(
            self,
            "event_at",
            utc_datetime(self.event_at, field_name="event_at"),
        )
        object.__setattr__(
            self,
            "available_at",
            utc_datetime(self.available_at, field_name="available_at"),
        )
        if self.available_at < self.event_at:
            raise ValueError("available_at cannot precede event_at")


@dataclass(frozen=True, slots=True)
class FuturePlateAppearanceContext:
    """Information known immediately before a simulated plate appearance."""

    prediction_run_id: str
    cutoff_at: datetime
    game_id: str
    plate_appearance_number: int
    batter_id: str
    pitcher_id: str
    catcher_id: str | None
    batter_team_id: str
    pitcher_team_id: str
    batting_order_index: int
    state: GameState
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.prediction_run_id:
            raise ValueError("prediction_run_id must not be empty")
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.plate_appearance_number < 1:
            raise ValueError("plate_appearance_number must be at least 1")
        if self.batting_order_index < 0:
            raise ValueError("batting_order_index must not be negative")
        if self.batter_team_id == self.pitcher_team_id:
            raise ValueError("batter and pitcher teams must be different")
        object.__setattr__(
            self,
            "cutoff_at",
            utc_datetime(self.cutoff_at, field_name="cutoff_at"),
        )


@dataclass(frozen=True, slots=True)
class SimulatedPlateAppearance:
    """A sampled future outcome, kept separate from historical observations."""

    simulation_id: int
    plate_appearance_number: int
    context: FuturePlateAppearanceContext
    sampled_event: TerminalPlateAppearanceEvent
    applied_event: TerminalPlateAppearanceEvent
    state_after: GameState
    runs_scored: int
    outs_recorded: int

    @property
    def credited_hit(self) -> bool:
        return self.applied_event.is_hit

    @property
    def credited_total_bases(self) -> int:
        """Official total bases credited for this simulated appearance."""

        return self.applied_event.hit_value
