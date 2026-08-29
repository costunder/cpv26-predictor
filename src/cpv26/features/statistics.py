"""Dependency-light statistical primitives for leakage-safe feature creation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from cpv26.domain import utc_datetime


@dataclass(frozen=True, slots=True)
class CountRate:
    """A rate that retains the evidence used to compute it."""

    numerator: float
    denominator: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.numerator) or not math.isfinite(self.denominator):
            raise ValueError("numerator and denominator must be finite")
        if self.numerator < 0.0 or self.denominator < 0.0:
            raise ValueError("numerator and denominator cannot be negative")

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator > 0.0 else None

    def value_or(self, default: float) -> float:
        value = self.value
        return default if value is None else value

    def as_features(self, prefix: str) -> dict[str, float | None]:
        return {
            f"{prefix}_numerator": self.numerator,
            f"{prefix}_denominator": self.denominator,
            f"{prefix}_rate": self.value,
        }

    def __add__(self, other: CountRate) -> CountRate:
        if not isinstance(other, CountRate):
            return NotImplemented
        return CountRate(
            self.numerator + other.numerator,
            self.denominator + other.denominator,
        )


@dataclass(frozen=True, slots=True)
class BetaPrior:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta):
            raise ValueError("beta prior parameters must be finite")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("beta prior parameters must be positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    @classmethod
    def from_mean_strength(cls, mean: float, strength: float) -> BetaPrior:
        if not 0.0 < mean < 1.0:
            raise ValueError("prior mean must be strictly between zero and one")
        if strength <= 0.0 or not math.isfinite(strength):
            raise ValueError("prior strength must be positive and finite")
        return cls(mean * strength, (1.0 - mean) * strength)


@dataclass(frozen=True, slots=True)
class BetaPosterior:
    alpha: float
    beta: float
    observed: CountRate
    prior: BetaPrior

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1.0))

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    @property
    def data_weight(self) -> float:
        denominator = self.observed.denominator + self.prior.strength
        return self.observed.denominator / denominator

    def as_features(self, prefix: str) -> dict[str, float | None]:
        features = self.observed.as_features(prefix)
        features.update(
            {
                f"{prefix}_shrunk": self.mean,
                f"{prefix}_posterior_sd": self.standard_deviation,
                f"{prefix}_data_weight": self.data_weight,
            }
        )
        return features


def aggregate_count_rates(observations: Iterable[CountRate]) -> CountRate:
    numerator = 0.0
    denominator = 0.0
    for observation in observations:
        numerator += observation.numerator
        denominator += observation.denominator
    return CountRate(numerator, denominator)


def fit_beta_prior(
    observations: Iterable[CountRate],
    *,
    minimum_strength: float = 2.0,
    maximum_strength: float = 10_000.0,
    fallback_mean: float = 0.5,
) -> BetaPrior:
    """Estimate a beta prior from a cohort using method-of-moments EB.

    The between-player rate variance is corrected for average binomial sampling
    variance.  Degenerate cohorts fall back to a strongly pooled prior rather
    than producing infinite or negative concentration.
    """

    if minimum_strength <= 0.0 or maximum_strength < minimum_strength:
        raise ValueError("invalid prior strength bounds")
    cohort = [item for item in observations if item.denominator > 0.0]
    if not cohort:
        return BetaPrior.from_mean_strength(fallback_mean, minimum_strength)
    for item in cohort:
        _validate_binomial_count(item)
    pooled = aggregate_count_rates(cohort)
    mean = pooled.numerator / pooled.denominator
    epsilon = 1e-9
    mean = min(max(mean, epsilon), 1.0 - epsilon)
    if len(cohort) == 1:
        return BetaPrior.from_mean_strength(mean, minimum_strength)

    rates = [item.numerator / item.denominator for item in cohort]
    weights = [item.denominator for item in cohort]
    weight_total = sum(weights)
    squared_weight_total = sum(weight * weight for weight in weights)
    effective_denominator = weight_total - squared_weight_total / weight_total
    if effective_denominator <= 0.0:
        observed_variance = 0.0
    else:
        observed_variance = (
            sum(weight * (rate - mean) ** 2 for weight, rate in zip(weights, rates, strict=True))
            / effective_denominator
        )
    average_sampling_variance = (
        sum(
            weight * mean * (1.0 - mean) / item.denominator
            for weight, item in zip(weights, cohort, strict=True)
        )
        / weight_total
    )
    between_variance = max(observed_variance - average_sampling_variance, 0.0)
    if between_variance <= epsilon:
        strength = maximum_strength
    else:
        strength = mean * (1.0 - mean) / between_variance - 1.0
        strength = min(max(strength, minimum_strength), maximum_strength)
    return BetaPrior.from_mean_strength(mean, strength)


def empirical_bayes_rate(observed: CountRate, prior: BetaPrior) -> BetaPosterior:
    """Return the exact beta-binomial posterior for one observed proportion."""

    _validate_binomial_count(observed)
    return BetaPosterior(
        alpha=prior.alpha + observed.numerator,
        beta=prior.beta + observed.denominator - observed.numerator,
        observed=observed,
        prior=prior,
    )


def empirical_bayes_shrinkage(
    observations: Sequence[CountRate],
    *,
    prior: BetaPrior | None = None,
    minimum_strength: float = 2.0,
    maximum_strength: float = 10_000.0,
) -> tuple[BetaPrior, tuple[BetaPosterior, ...]]:
    """Fit a cohort prior when needed, then shrink every supplied rate."""

    fitted = prior or fit_beta_prior(
        observations,
        minimum_strength=minimum_strength,
        maximum_strength=maximum_strength,
    )
    return fitted, tuple(empirical_bayes_rate(item, fitted) for item in observations)


def ewma(
    values: Iterable[float | None],
    *,
    alpha: float | None = None,
    span: float | None = None,
    half_life: float | None = None,
    min_periods: int = 1,
    adjust: bool = False,
) -> tuple[float | None, ...]:
    """Compute an exponentially weighted moving average without pandas."""

    smoothing = _resolve_alpha(alpha=alpha, span=span, half_life=half_life)
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    result: list[float | None] = []
    state: float | None = None
    weighted_sum = 0.0
    total_weight = 0.0
    count = 0
    decay = 1.0 - smoothing
    for raw_value in values:
        if raw_value is None:
            result.append(state if state is not None and count >= min_periods else None)
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("EWMA values must be finite or None")
        count += 1
        if adjust:
            weighted_sum = value + decay * weighted_sum
            total_weight = 1.0 + decay * total_weight
            state = weighted_sum / total_weight
        elif state is None:
            state = value
        else:
            state = smoothing * value + decay * state
        result.append(state if count >= min_periods else None)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TimedValue:
    occurred_at: datetime
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurred_at",
            utc_datetime(self.occurred_at, field_name="occurred_at"),
        )
        if not math.isfinite(self.value):
            raise ValueError("timed value must be finite")


def time_decay_ewma(
    observations: Iterable[TimedValue],
    *,
    as_of: datetime,
    half_life: timedelta,
) -> float | None:
    """Return an irregular-time EWMA evaluated at ``as_of``.

    Each observation receives weight ``2 ** (-age / half_life)``.  This direct
    weighted formulation is stable for irregular baseball schedules and makes
    postponed games and off-days decay correctly.
    """

    point = utc_datetime(as_of, field_name="as_of")
    half_life_seconds = half_life.total_seconds()
    if half_life_seconds <= 0.0:
        raise ValueError("half_life must be positive")
    weighted_values: list[tuple[float, float]] = []
    for observation in observations:
        if observation.occurred_at >= point:
            raise ValueError("EWMA observations must occur before as_of")
        age_seconds = (point - observation.occurred_at).total_seconds()
        log_weight = -math.log(2.0) * age_seconds / half_life_seconds
        weighted_values.append((log_weight, observation.value))
    if not weighted_values:
        return None
    maximum_log_weight = max(item[0] for item in weighted_values)
    weighted_sum = 0.0
    total_weight = 0.0
    for log_weight, value in weighted_values:
        weight = math.exp(log_weight - maximum_log_weight)
        weighted_sum += weight * value
        total_weight += weight
    return weighted_sum / total_weight


def rolling_count_rate(
    observations: Iterable[CountRate | None],
    *,
    window: int,
    min_periods: int = 1,
) -> tuple[CountRate | None, ...]:
    """Aggregate rolling numerators and denominators without averaging rates."""

    if window < 1:
        raise ValueError("window must be positive")
    if min_periods < 1 or min_periods > window:
        raise ValueError("min_periods must be between one and window")
    queue: deque[CountRate | None] = deque()
    numerator = 0.0
    denominator = 0.0
    present = 0
    output: list[CountRate | None] = []
    for observation in observations:
        queue.append(observation)
        if observation is not None:
            numerator += observation.numerator
            denominator += observation.denominator
            present += 1
        if len(queue) > window:
            removed = queue.popleft()
            if removed is not None:
                numerator -= removed.numerator
                denominator -= removed.denominator
                present -= 1
        if abs(numerator) < 1e-12:
            numerator = 0.0
        if abs(denominator) < 1e-12:
            denominator = 0.0
        output.append(CountRate(numerator, denominator) if present >= min_periods else None)
    return tuple(output)


def rolling_sum(
    values: Iterable[float | None],
    *,
    window: int,
    min_periods: int = 1,
) -> tuple[float | None, ...]:
    """Compute a fixed-observation rolling sum while ignoring missing values."""

    if window < 1:
        raise ValueError("window must be positive")
    if min_periods < 1 or min_periods > window:
        raise ValueError("min_periods must be between one and window")
    queue: deque[float | None] = deque()
    total = 0.0
    present = 0
    output: list[float | None] = []
    for raw_value in values:
        value: float | None
        if raw_value is None:
            value = None
        else:
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("rolling values must be finite or None")
        queue.append(value)
        if value is not None:
            total += value
            present += 1
        if len(queue) > window:
            removed = queue.popleft()
            if removed is not None:
                total -= removed
                present -= 1
        if abs(total) < 1e-12:
            total = 0.0
        output.append(total if present >= min_periods else None)
    return tuple(output)


def rolling_mean(
    values: Iterable[float | None],
    *,
    window: int,
    min_periods: int = 1,
) -> tuple[float | None, ...]:
    materialised = tuple(values)
    sums = rolling_sum(materialised, window=window, min_periods=min_periods)
    counts = rolling_sum(
        (None if value is None else 1.0 for value in materialised),
        window=window,
        min_periods=1,
    )
    output: list[float | None] = []
    for total, count in zip(sums, counts, strict=True):
        if total is None or count is None or count == 0.0:
            output.append(None)
        else:
            output.append(total / count)
    return tuple(output)


def _resolve_alpha(
    *,
    alpha: float | None,
    span: float | None,
    half_life: float | None,
) -> float:
    supplied = sum(value is not None for value in (alpha, span, half_life))
    if supplied != 1:
        raise ValueError("provide exactly one of alpha, span or half_life")
    if alpha is not None:
        smoothing = float(alpha)
    elif span is not None:
        if span < 1.0:
            raise ValueError("span must be at least one")
        smoothing = 2.0 / (float(span) + 1.0)
    else:
        assert half_life is not None
        if half_life <= 0.0:
            raise ValueError("half_life must be positive")
        smoothing = 1.0 - math.exp(math.log(0.5) / float(half_life))
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return smoothing


def _validate_binomial_count(observed: CountRate) -> None:
    if observed.numerator > observed.denominator:
        raise ValueError("binomial numerator cannot exceed denominator")
