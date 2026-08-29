"""Immutable baseball game state used by the sequential simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class HalfInning(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"

    @property
    def opposite(self) -> HalfInning:
        return self.BOTTOM if self is self.TOP else self.TOP


@dataclass(frozen=True, slots=True)
class BaseRunners:
    """Runner identifiers on first, second, and third base."""

    first: str | None = None
    second: str | None = None
    third: str | None = None

    def __post_init__(self) -> None:
        occupied = [runner for runner in self.as_tuple() if runner is not None]
        if len(occupied) != len(set(occupied)):
            raise ValueError("a runner cannot occupy more than one base")

    def as_tuple(self) -> tuple[str | None, str | None, str | None]:
        return self.first, self.second, self.third

    @property
    def count(self) -> int:
        return sum(runner is not None for runner in self.as_tuple())

    @property
    def is_empty(self) -> bool:
        return self.count == 0


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete state required to advance a game by one plate appearance."""

    inning: int = 1
    half: HalfInning = HalfInning.TOP
    outs: int = 0
    bases: BaseRunners = BaseRunners()
    away_score: int = 0
    home_score: int = 0
    away_batting_index: int = 0
    home_batting_index: int = 0

    def __post_init__(self) -> None:
        if self.inning < 1:
            raise ValueError("inning must be at least 1")
        if not 0 <= self.outs <= 2:
            raise ValueError("outs must be between 0 and 2")
        if self.away_score < 0 or self.home_score < 0:
            raise ValueError("scores must not be negative")
        if self.away_batting_index < 0 or self.home_batting_index < 0:
            raise ValueError("batting indexes must not be negative")

    @classmethod
    def initial(cls) -> GameState:
        return cls()

    @property
    def batting_index(self) -> int:
        return self.away_batting_index if self.half is HalfInning.TOP else self.home_batting_index

    @property
    def batting_score(self) -> int:
        return self.away_score if self.half is HalfInning.TOP else self.home_score

    @property
    def fielding_score(self) -> int:
        return self.home_score if self.half is HalfInning.TOP else self.away_score

    def with_bases_and_outs(
        self,
        bases: BaseRunners,
        outs: int,
        *,
        runs_scored: int = 0,
    ) -> GameState:
        if not 0 <= outs <= 3:
            raise ValueError("intermediate outs must be between 0 and 3")
        if runs_scored < 0:
            raise ValueError("runs_scored must not be negative")
        if self.half is HalfInning.TOP:
            return replace(
                self,
                bases=bases,
                outs=min(outs, 2),
                away_score=self.away_score + runs_scored,
            )
        return replace(
            self,
            bases=bases,
            outs=min(outs, 2),
            home_score=self.home_score + runs_scored,
        )

    def advance_batter(self, lineup_size: int) -> GameState:
        if lineup_size < 1:
            raise ValueError("lineup_size must be positive")
        if self.half is HalfInning.TOP:
            return replace(
                self,
                away_batting_index=(self.away_batting_index + 1) % lineup_size,
            )
        return replace(
            self,
            home_batting_index=(self.home_batting_index + 1) % lineup_size,
        )

    def next_half_inning(self) -> GameState:
        if self.half is HalfInning.TOP:
            return replace(
                self,
                half=HalfInning.BOTTOM,
                outs=0,
                bases=BaseRunners(),
            )
        return replace(
            self,
            inning=self.inning + 1,
            half=HalfInning.TOP,
            outs=0,
            bases=BaseRunners(),
        )
