"""Point-in-time batting feature construction from observed plate appearances."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from cpv26.domain import utc_datetime, utc_isoformat

from .statistics import (
    BetaPrior,
    CountRate,
    TimedValue,
    empirical_bayes_rate,
    fit_beta_prior,
    time_decay_ewma,
)

_STRIKEOUT_OUTCOMES = {"K", "SO", "STRIKEOUT", "STRIKEOUT_LOOKING"}
_WALK_OUTCOMES = {"BB", "IBB", "WALK", "INTENTIONAL_WALK"}
_HBP_OUTCOMES = {"HBP", "HIT_BY_PITCH"}


@dataclass(frozen=True, slots=True)
class BattingFeatureConfig:
    rolling_plate_appearances: tuple[int, ...] = (20, 50, 100)
    ewma_half_life_days: tuple[float, ...] = (7.0, 14.0, 30.0)
    prior_minimum_strength: float = 25.0
    prior_maximum_strength: float = 2_000.0

    def __post_init__(self) -> None:
        if not self.rolling_plate_appearances:
            raise ValueError("at least one rolling window is required")
        if any(window < 1 for window in self.rolling_plate_appearances):
            raise ValueError("rolling windows must be positive")
        if len(set(self.rolling_plate_appearances)) != len(self.rolling_plate_appearances):
            raise ValueError("rolling windows must be unique")
        if any(days <= 0.0 for days in self.ewma_half_life_days):
            raise ValueError("EWMA half-lives must be positive")


@dataclass(frozen=True, slots=True)
class BatterFeatureVector:
    player_id: str
    cutoff_at: str
    numerators: Mapping[str, float]
    denominators: Mapping[str, float]
    features: Mapping[str, float | int | str | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerators", dict(self.numerators))
        object.__setattr__(self, "denominators", dict(self.denominators))
        object.__setattr__(self, "features", dict(self.features))


@dataclass(frozen=True, slots=True)
class _PlateAppearance:
    plate_appearance_id: str
    batter_id: str
    event_at: datetime
    outcome: str
    is_at_bat: bool
    is_hit: bool
    total_bases: int


def build_batter_features(
    rows: Iterable[Mapping[str, Any]],
    *,
    cutoff_at: datetime,
    knowledge_at: datetime | None = None,
    config: BattingFeatureConfig | None = None,
    hit_rate_prior: BetaPrior | None = None,
) -> dict[str, BatterFeatureVector]:
    """Build one feature vector per batter using facts strictly before cutoff.

    If temporal audit columns are present, they are validated. This means a raw
    non-snapshotted query fails loudly instead of leaking a late correction into
    historical training data.
    """

    settings = config or BattingFeatureConfig()
    cutoff = utc_datetime(cutoff_at, field_name="cutoff_at")
    knowledge = utc_datetime(knowledge_at or cutoff, field_name="knowledge_at")
    if knowledge < cutoff:
        raise ValueError("knowledge_at cannot precede cutoff_at")
    grouped: dict[str, list[_PlateAppearance]] = defaultdict(list)
    seen_ids: set[str] = set()
    for raw in rows:
        plate_appearance = _parse_plate_appearance(raw, cutoff_at=cutoff, knowledge_at=knowledge)
        if plate_appearance.plate_appearance_id in seen_ids:
            raise ValueError(
                "duplicate plate_appearance_id after point-in-time selection: "
                f"{plate_appearance.plate_appearance_id}"
            )
        seen_ids.add(plate_appearance.plate_appearance_id)
        grouped[plate_appearance.batter_id].append(plate_appearance)
    for appearances in grouped.values():
        appearances.sort(key=lambda item: (item.event_at, item.plate_appearance_id))

    hit_counts = [_count_rates(appearances)["hit_rate"] for appearances in grouped.values()]
    prior = hit_rate_prior or fit_beta_prior(
        hit_counts,
        minimum_strength=settings.prior_minimum_strength,
        maximum_strength=settings.prior_maximum_strength,
        fallback_mean=0.25,
    )
    vectors: dict[str, BatterFeatureVector] = {}
    for player_id, appearances in sorted(grouped.items()):
        full_counts = _count_rates(appearances)
        numerators = {name: rate.numerator for name, rate in full_counts.items()}
        denominators = {name: rate.denominator for name, rate in full_counts.items()}
        features: dict[str, float | int | str | None] = {
            "cutoff_at": utc_isoformat(cutoff),
            "plate_appearances": len(appearances),
            "first_observed_at": utc_isoformat(appearances[0].event_at),
            "last_observed_at": utc_isoformat(appearances[-1].event_at),
        }
        for name, rate in full_counts.items():
            features.update(rate.as_features(f"career_{name}"))
        features.update(
            empirical_bayes_rate(full_counts["hit_rate"], prior).as_features("career_hit_rate_eb")
        )
        features["hit_rate_prior_mean"] = prior.mean
        features["hit_rate_prior_strength"] = prior.strength

        for window in sorted(settings.rolling_plate_appearances):
            window_counts = _count_rates(appearances[-window:])
            for name, rate in window_counts.items():
                features.update(rate.as_features(f"last_{window}_pa_{name}"))

        for days in sorted(settings.ewma_half_life_days):
            label = _format_days(days)
            for feature_name, timed_values in _timed_indicators(appearances).items():
                features[f"{feature_name}_ewma_{label}d"] = time_decay_ewma(
                    timed_values,
                    as_of=cutoff,
                    half_life=timedelta(days=days),
                )
        vectors[player_id] = BatterFeatureVector(
            player_id=player_id,
            cutoff_at=utc_isoformat(cutoff),
            numerators=numerators,
            denominators=denominators,
            features=features,
        )
    return vectors


def _parse_plate_appearance(
    row: Mapping[str, Any], *, cutoff_at: datetime, knowledge_at: datetime
) -> _PlateAppearance:
    required = {
        "plate_appearance_id",
        "batter_id",
        "event_at",
        "outcome",
        "is_at_bat",
        "is_hit",
        "total_bases",
    }
    missing = required - set(row)
    if missing:
        raise ValueError("plate appearance lacks fields: " + ", ".join(sorted(missing)))
    event_at = utc_datetime(row["event_at"], field_name="event_at")
    if event_at >= cutoff_at:
        raise ValueError("plate appearance event_at must be before cutoff_at")
    if "available_at" in row:
        available_at = utc_datetime(row["available_at"], field_name="available_at")
        if available_at > cutoff_at:
            raise ValueError("plate appearance was not available at cutoff_at")
    if "ingested_at" in row:
        ingested_at = utc_datetime(row["ingested_at"], field_name="ingested_at")
        if ingested_at > knowledge_at:
            raise ValueError("plate appearance was not ingested at knowledge_at")
    total_bases = int(row["total_bases"])
    is_hit = bool(row["is_hit"])
    if not 0 <= total_bases <= 4:
        raise ValueError("total_bases must be between zero and four")
    if is_hit != (total_bases > 0):
        raise ValueError("is_hit and total_bases are inconsistent")
    return _PlateAppearance(
        plate_appearance_id=str(row["plate_appearance_id"]),
        batter_id=str(row["batter_id"]),
        event_at=event_at,
        outcome=str(row["outcome"]).strip().upper(),
        is_at_bat=bool(row["is_at_bat"]),
        is_hit=is_hit,
        total_bases=total_bases,
    )


def _count_rates(
    appearances: Sequence[_PlateAppearance],
) -> dict[str, CountRate]:
    plate_appearances = float(len(appearances))
    at_bats = float(sum(item.is_at_bat for item in appearances))
    hits = float(sum(item.is_hit for item in appearances))
    home_runs = float(sum(item.is_hit and item.total_bases == 4 for item in appearances))
    extra_base_hits = float(sum(item.is_hit and item.total_bases >= 2 for item in appearances))
    strikeouts = float(sum(item.outcome in _STRIKEOUT_OUTCOMES for item in appearances))
    walks = float(sum(item.outcome in _WALK_OUTCOMES for item in appearances))
    hit_by_pitch = float(sum(item.outcome in _HBP_OUTCOMES for item in appearances))
    total_bases = float(sum(item.total_bases for item in appearances))
    return {
        "hit_rate": CountRate(hits, at_bats),
        "home_run_rate": CountRate(home_runs, at_bats),
        "extra_base_hit_rate": CountRate(extra_base_hits, at_bats),
        "strikeout_rate": CountRate(strikeouts, plate_appearances),
        "walk_rate": CountRate(walks, plate_appearances),
        "hit_by_pitch_rate": CountRate(hit_by_pitch, plate_appearances),
        "total_bases_per_at_bat": CountRate(total_bases, at_bats),
    }


def _timed_indicators(
    appearances: Sequence[_PlateAppearance],
) -> dict[str, list[TimedValue]]:
    return {
        "hit": [TimedValue(item.event_at, float(item.is_hit)) for item in appearances],
        "home_run": [
            TimedValue(item.event_at, float(item.is_hit and item.total_bases == 4))
            for item in appearances
        ],
        "strikeout": [
            TimedValue(item.event_at, float(item.outcome in _STRIKEOUT_OUTCOMES))
            for item in appearances
        ],
        "walk": [
            TimedValue(item.event_at, float(item.outcome in _WALK_OUTCOMES)) for item in appearances
        ],
    }


def _format_days(days: float) -> str:
    return str(int(days)) if float(days).is_integer() else str(days).replace(".", "p")
