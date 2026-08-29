"""Exhaustive expected-utility optimization for V26 match predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class MatchPickOption:
    """One selectable outcome and its model probability/reward."""

    option_id: str
    label: str
    probability: float
    reward_points: float

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("option_id must not be empty")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be finite and between 0 and 1")
        if not math.isfinite(self.reward_points) or self.reward_points < 0.0:
            raise ValueError("reward_points must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GamePredictionMarket:
    game_id: str
    options: tuple[MatchPickOption, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if len(self.options) < 2:
            raise ValueError("a market must expose at least two outcomes")
        option_ids = [option.option_id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("market option identifiers must be unique")
        if sum(option.probability for option in self.options) <= 0.0:
            raise ValueError("a market must have positive total probability")

    def normalized_options(self) -> tuple[tuple[MatchPickOption, float], ...]:
        total = sum(option.probability for option in self.options)
        return tuple((option, option.probability / total) for option in self.options)


@dataclass(frozen=True, slots=True)
class MatchPredictionObjective:
    """Point rules and optional mean-variance risk preference."""

    all_correct_bonus_points: float = 0.0
    risk_aversion: float = 0.0
    minimum_all_correct_probability: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.all_correct_bonus_points) or self.all_correct_bonus_points < 0.0:
            raise ValueError("all_correct_bonus_points must be finite and non-negative")
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and non-negative")
        if not 0.0 <= self.minimum_all_correct_probability <= 1.0:
            raise ValueError("minimum_all_correct_probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MatchPredictionRecommendation:
    picks: tuple[tuple[str, MatchPickOption], ...]
    expected_points: float
    point_variance: float
    all_correct_probability: float
    expected_correct_picks: float
    expected_utility: float

    @property
    def pick_by_game(self) -> dict[str, MatchPickOption]:
        return dict(self.picks)

    @property
    def point_standard_deviation(self) -> float:
        return math.sqrt(self.point_variance)


class MatchPredictionOptimizer:
    """Enumerate every daily pick combination and rank exact utility."""

    def optimize(
        self,
        markets: tuple[GamePredictionMarket, ...],
        *,
        objective: MatchPredictionObjective | None = None,
        top_k: int = 1,
    ) -> tuple[MatchPredictionRecommendation, ...]:
        if not markets:
            raise ValueError("at least one game market is required")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        game_ids = [market.game_id for market in markets]
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("game identifiers must be unique")

        rules = objective or MatchPredictionObjective()
        normalized = tuple(market.normalized_options() for market in markets)
        recommendations: list[MatchPredictionRecommendation] = []

        for picked in product(*(market.options for market in markets)):
            chosen_probabilities = tuple(
                next(
                    probability
                    for option, probability in normalized[index]
                    if option.option_id == selected.option_id
                )
                for index, selected in enumerate(picked)
            )
            all_correct_probability = math.prod(chosen_probabilities)
            if all_correct_probability < rules.minimum_all_correct_probability:
                continue

            expected_points, variance = self._point_moments(
                normalized,
                picked,
                rules.all_correct_bonus_points,
            )
            utility = expected_points - rules.risk_aversion * variance
            recommendations.append(
                MatchPredictionRecommendation(
                    picks=tuple(
                        (market.game_id, selection)
                        for market, selection in zip(markets, picked, strict=True)
                    ),
                    expected_points=expected_points,
                    point_variance=variance,
                    all_correct_probability=all_correct_probability,
                    expected_correct_picks=sum(chosen_probabilities),
                    expected_utility=utility,
                )
            )

        recommendations.sort(
            key=lambda recommendation: (
                recommendation.expected_utility,
                recommendation.expected_points,
                recommendation.all_correct_probability,
                tuple(option.option_id for _, option in recommendation.picks),
            ),
            reverse=True,
        )
        return tuple(recommendations[:top_k])

    @staticmethod
    def _point_moments(
        normalized_markets: tuple[tuple[tuple[MatchPickOption, float], ...], ...],
        picked: tuple[MatchPickOption, ...],
        all_correct_bonus: float,
    ) -> tuple[float, float]:
        first_moment = 0.0
        second_moment = 0.0
        for realized in product(*normalized_markets):
            probability = math.prod(probability for _, probability in realized)
            correct = tuple(
                actual.option_id == selected.option_id
                for (actual, _), selected in zip(realized, picked, strict=True)
            )
            points = sum(
                selected.reward_points
                for selected, is_correct in zip(picked, correct, strict=True)
                if is_correct
            )
            if all(correct):
                points += all_correct_bonus
            first_moment += probability * points
            second_moment += probability * points * points
        variance = max(0.0, second_moment - first_moment * first_moment)
        return first_moment, variance
