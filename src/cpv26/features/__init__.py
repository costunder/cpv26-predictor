"""Statistical and baseball-specific point-in-time feature builders."""

from .batting import (
    BatterFeatureVector,
    BattingFeatureConfig,
    build_batter_features,
)
from .state import TeamFeatureVector, player_state_rows, team_state_rows
from .statistics import (
    BetaPosterior,
    BetaPrior,
    CountRate,
    TimedValue,
    aggregate_count_rates,
    empirical_bayes_rate,
    empirical_bayes_shrinkage,
    ewma,
    fit_beta_prior,
    rolling_count_rate,
    rolling_mean,
    rolling_sum,
    time_decay_ewma,
)

__all__ = [
    "BatterFeatureVector",
    "BattingFeatureConfig",
    "BetaPosterior",
    "BetaPrior",
    "CountRate",
    "TeamFeatureVector",
    "TimedValue",
    "aggregate_count_rates",
    "build_batter_features",
    "empirical_bayes_rate",
    "empirical_bayes_shrinkage",
    "ewma",
    "fit_beta_prior",
    "player_state_rows",
    "rolling_count_rate",
    "rolling_mean",
    "rolling_sum",
    "team_state_rows",
    "time_decay_ewma",
]
