"""Causal Live Hit baseline conditional on a player recording at least one PA.

The unit of prediction is a player-game, not a team-game or a single plate
appearance. This historical experiment estimates ``P(any hit | appeared)``.
It does not estimate the chance that a candidate appears, and it must not be
presented as an unconditional pregame candidate-ranking model.

Use the complete canonical PA population for team offense/defense features.
All rows on one KBO calendar date share the same pre-date history; even a
doubleheader's first result is unavailable to its second game's features.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from cpv26.data.kbo_source_snapshots import source_snapshot_filter_sql
from cpv26.evaluation import evaluate_probabilities
from cpv26.models.baseline import DEFAULT_CATBOOST_PARAMETERS, CatBoostClassifierBaseline

from .kbo_match_baseline import (
    FIXED_SEASON_EVALUATIONS,
    FixedSeasonEvaluation,
    SeasonDatasetSplit,
    render_evaluation_json,
)

NO_HIT = 0
ANY_HIT = 1
LIVE_HIT_LABELS: tuple[str, str] = ("no_hit", "hit")
_HIT_OUTCOMES = frozenset({"single", "double", "triple", "home_run"})
_NON_HIT_OUTCOMES = frozenset(
    {
        "walk",
        "hit_by_pitch",
        "strikeout",
        "ball_in_play_out",
        "double_play",
        "sacrifice_fly",
        "sacrifice_bunt",
        "reached_on_error",
        "fielders_choice",
        "catcher_interference",
    }
)

LIVE_HIT_FEATURE_NAMES: tuple[str, ...] = (
    "player_games_before",
    "player_pa_before",
    "player_hits_before",
    "player_career_hit_rate_smoothed",
    "player_career_any_hit_rate_smoothed",
    "player_recent_games_before",
    "player_recent_pa_before",
    "player_recent_hits_before",
    "player_recent_hit_rate_smoothed",
    "player_recent_any_hit_rate_smoothed",
    "batting_team_games_before",
    "batting_team_pa_before",
    "batting_team_hits_before",
    "batting_team_offense_hit_rate_smoothed",
    "batting_team_recent_games_before",
    "batting_team_recent_hit_rate_smoothed",
    "opponent_fielding_games_before",
    "opponent_pa_faced_before",
    "opponent_hits_allowed_before",
    "opponent_hits_allowed_rate_smoothed",
    "opponent_recent_games_before",
    "opponent_recent_hits_allowed_rate_smoothed",
    "player_vs_opponent_pa_before",
    "player_vs_opponent_hits_before",
    "player_vs_opponent_hit_rate_smoothed",
)

# Current canonical revisions are intended for a retrospective experiment.
# Historical publication timestamps are not reconstructed by this query.
# Validity/state filters follow revision selection so closed newer revisions
# cannot resurrect older, still-open rows.
LIVE_HIT_CANONICAL_SQL = f"""
WITH latest_pa AS (
    SELECT * FROM observed_plate_appearance
    WHERE {source_snapshot_filter_sql()}
    QUALIFY row_number() OVER (
        PARTITION BY plate_appearance_id
        ORDER BY available_at DESC, ingested_at DESC, valid_from DESC, observed_pa_row_id DESC
    ) = 1
), latest_game AS (
    SELECT * FROM game
    WHERE {source_snapshot_filter_sql()}
    QUALIFY row_number() OVER (
        PARTITION BY game_id
        ORDER BY available_at DESC, ingested_at DESC, valid_from DESC, game_row_id DESC
    ) = 1
), latest_player AS (
    SELECT * FROM player
    WHERE {source_snapshot_filter_sql()}
    QUALIFY row_number() OVER (
        PARTITION BY player_id
        ORDER BY available_at DESC, ingested_at DESC, valid_from DESC, player_row_id DESC
    ) = 1
), latest_team AS (
    SELECT * FROM team
    WHERE {source_snapshot_filter_sql()}
    QUALIFY row_number() OVER (
        PARTITION BY team_id
        ORDER BY available_at DESC, ingested_at DESC, valid_from DESC, team_row_id DESC
    ) = 1
)
SELECT
    pa.plate_appearance_id,
    pa.game_id,
    CAST(g.scheduled_start AT TIME ZONE 'Asia/Seoul' AS DATE) AS game_date,
    pa.batter_id,
    p.display_name AS batter_name,
    pa.batting_team_id,
    bt.team_name AS batting_team_name,
    pa.fielding_team_id AS opponent_team_id,
    ft.team_name AS opponent_team_name,
    pa.outcome,
    pa.is_hit
FROM latest_pa AS pa
JOIN latest_game AS g ON g.game_id = pa.game_id AND g.valid_to IS NULL
LEFT JOIN latest_player AS p ON p.player_id = pa.batter_id AND p.valid_to IS NULL
LEFT JOIN latest_team AS bt ON bt.team_id = pa.batting_team_id AND bt.valid_to IS NULL
LEFT JOIN latest_team AS ft ON ft.team_id = pa.fielding_team_id AND ft.valid_to IS NULL
WHERE pa.valid_to IS NULL AND g.game_status = 'final'
ORDER BY game_date, pa.game_id, pa.sequence_in_game, pa.event_subsequence
"""


@dataclass(frozen=True, slots=True)
class CanonicalPlateAppearanceRow:
    """Canonical PA joined to its KBO-local game date and optional names."""

    plate_appearance_id: str
    game_id: str
    game_date: date
    batter_id: str
    batting_team_id: str
    opponent_team_id: str
    outcome: str
    is_hit: bool | None = None
    batter_name: str | None = None
    batting_team_name: str | None = None
    opponent_team_name: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "plate_appearance_id",
            "game_id",
            "batter_id",
            "batting_team_id",
            "opponent_team_id",
            "outcome",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.batting_team_id == self.opponent_team_id:
            raise ValueError("batting and opponent teams must be different")
        object.__setattr__(self, "game_date", _coerce_game_date(self.game_date))
        known_hit = self.outcome in _HIT_OUTCOMES
        known_outcome = known_hit or self.outcome in _NON_HIT_OUTCOMES
        if self.is_hit is None:
            if not known_outcome:
                raise ValueError(f"unknown outcome without is_hit: {self.outcome!r}")
            hit = known_hit
        else:
            if not isinstance(self.is_hit, (bool, np.bool_)):
                raise TypeError("is_hit must be a boolean or None")
            hit = bool(self.is_hit)
            if known_outcome and hit != known_hit:
                raise ValueError("is_hit disagrees with canonical outcome")
        object.__setattr__(self, "is_hit", hit)
        for name in ("batter_name", "batting_team_name", "opponent_team_name"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")


PlateAppearanceInputRow = CanonicalPlateAppearanceRow | Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True, slots=True)
class PlayerGameLiveHitRow:
    """Observed player-game target; absent players deliberately have no row."""

    game_id: str
    game_date: date
    player_id: str
    team_id: str
    opponent_team_id: str
    plate_appearances: int
    hits: int
    player_name: str | None = None

    @property
    def target_class(self) -> int:
        return ANY_HIT if self.hits > 0 else NO_HIT


@dataclass(frozen=True, slots=True)
class LiveHitFeatureDataset:
    """Read-only pregame features plus separate observed target metadata."""

    game_ids: tuple[str, ...]
    game_dates: tuple[date, ...]
    player_ids: tuple[str, ...]
    team_ids: tuple[str, ...]
    opponent_team_ids: tuple[str, ...]
    player_names: tuple[str | None, ...]
    seasons: NDArray[np.int64]
    features: NDArray[np.float64]
    targets: NDArray[np.int64]
    plate_appearances: NDArray[np.int64]
    hits: NDArray[np.int64]
    feature_names: tuple[str, ...] = LIVE_HIT_FEATURE_NAMES

    def __post_init__(self) -> None:
        identity_names = (
            "game_ids",
            "game_dates",
            "player_ids",
            "team_ids",
            "opponent_team_ids",
            "player_names",
        )
        for name in identity_names:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.game_ids)
        if count == 0:
            raise ValueError("Live Hit dataset cannot be empty")
        if any(len(getattr(self, name)) != count for name in identity_names):
            raise ValueError("Live Hit identity columns must have equal length")
        if len(set(zip(self.game_ids, self.player_ids, strict=True))) != count:
            raise ValueError("player-game identities must be unique")
        if any(
            left > right for left, right in zip(self.game_dates, self.game_dates[1:], strict=False)
        ):
            raise ValueError("Live Hit dataset must be chronological")
        feature_names = tuple(self.feature_names)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        for name in ("seasons", "targets", "plate_appearances", "hits"):
            values = np.asarray(getattr(self, name), dtype=np.int64).copy()
            if values.shape != (count,):
                raise ValueError(f"{name} must contain one value per player-game")
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        features = np.asarray(self.features, dtype=np.float64).copy()
        if features.shape != (count, len(feature_names)) or not np.all(np.isfinite(features)):
            raise ValueError("features must be a finite player-game by feature matrix")
        if np.any(self.plate_appearances < 1):
            raise ValueError("Live Hit population requires at least one observed PA")
        if np.any(self.hits < 0) or np.any(self.hits > self.plate_appearances):
            raise ValueError("hits must lie between zero and observed PA count")
        if not np.array_equal(self.targets, (self.hits > 0).astype(np.int64)):
            raise ValueError("targets must indicate at least one observed hit")
        if not np.array_equal(self.seasons, [item.year for item in self.game_dates]):
            raise ValueError("seasons must agree with KBO-local game dates")
        features.flags.writeable = False
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", feature_names)

    @property
    def row_count(self) -> int:
        return len(self.game_ids)


@dataclass(frozen=True, slots=True)
class _Counts:
    pa: int
    hits: int


@dataclass(slots=True)
class _History:
    recent: deque[_Counts]
    games: int = 0
    pa: int = 0
    hits: int = 0
    hit_games: int = 0

    def record(self, counts: _Counts) -> None:
        self.games += 1
        self.pa += counts.pa
        self.hits += counts.hits
        self.hit_games += int(counts.hits > 0)
        self.recent.append(counts)

    @property
    def recent_pa(self) -> int:
        return sum(item.pa for item in self.recent)

    @property
    def recent_hits(self) -> int:
        return sum(item.hits for item in self.recent)


@dataclass(frozen=True, slots=True)
class _Priors:
    hit_rate: float
    pa: float
    any_hit_rate: float
    games: float
    relation_pa: float

    def hit(self, hits: int, pa: int, *, relation: bool = False) -> float:
        strength = self.relation_pa if relation else self.pa
        return (hits + self.hit_rate * strength) / (pa + strength)

    def any_hit(self, hit_games: int, games: int) -> float:
        return (hit_games + self.any_hit_rate * self.games) / (games + self.games)


@dataclass(slots=True)
class _Histories:
    rolling_games: int
    players: dict[str, _History] = field(default_factory=dict)
    offense: dict[str, _History] = field(default_factory=dict)
    defense: dict[str, _History] = field(default_factory=dict)
    relations: dict[tuple[str, str], _Counts] = field(default_factory=dict)

    def get(self, collection: dict[str, _History], key: str) -> _History:
        if key not in collection:
            collection[key] = _History(recent=deque(maxlen=self.rolling_games))
        return collection[key]


class _ProbabilityClassifier(Protocol):
    @property
    def classes_(self) -> NDArray[Any]: ...

    def fit(self, features: Any, targets: Any) -> Any: ...

    def predict_proba(self, features: Any) -> Any: ...


def canonicalize_plate_appearance_rows(
    rows: Iterable[PlateAppearanceInputRow],
) -> tuple[CanonicalPlateAppearanceRow, ...]:
    """Accept canonical objects, mappings, or seven/eight/eleven-value tuples.

    Short tuple order: ``(pa_id, game_id, game_date, batter_id, batting_team_id,
    opponent_team_id, outcome[, is_hit])``. Eleven-value tuples follow
    :data:`LIVE_HIT_CANONICAL_SQL`, including the three display-name columns.
    Mapping rows also accept ``player_id``, ``team_id``, ``fielding_team_id``,
    and ``scheduled_start`` aliases. Duplicate PA IDs are rejected, not counted.
    """

    materialized: list[CanonicalPlateAppearanceRow] = []
    seen: set[str] = set()
    game_contexts: dict[str, tuple[date, frozenset[str]]] = {}
    for raw in rows:
        row = _canonical_pa(raw)
        if row.plate_appearance_id in seen:
            raise ValueError(f"duplicate plate_appearance_id: {row.plate_appearance_id}")
        seen.add(row.plate_appearance_id)
        context = (row.game_date, frozenset((row.batting_team_id, row.opponent_team_id)))
        previous = game_contexts.setdefault(row.game_id, context)
        if previous != context:
            raise ValueError(f"inconsistent date or teams for game_id: {row.game_id}")
        materialized.append(row)
    if not materialized:
        raise ValueError("at least one observed plate appearance is required")
    return tuple(sorted(materialized, key=lambda row: (row.game_date, row.game_id)))


def reduce_player_game_live_hit_targets(
    rows: Iterable[PlateAppearanceInputRow],
) -> tuple[PlayerGameLiveHitRow, ...]:
    """Reduce terminal PAs to appeared-player game labels and PA-count metadata."""

    grouped: dict[tuple[str, str], list[CanonicalPlateAppearanceRow]] = {}
    for row in canonicalize_plate_appearance_rows(rows):
        grouped.setdefault((row.game_id, row.batter_id), []).append(row)
    results: list[PlayerGameLiveHitRow] = []
    for (_, player_id), appearances in grouped.items():
        first = appearances[0]
        if any(item.batting_team_id != first.batting_team_id for item in appearances):
            raise ValueError(f"player {player_id} has conflicting teams in {first.game_id}")
        results.append(
            PlayerGameLiveHitRow(
                game_id=first.game_id,
                game_date=first.game_date,
                player_id=player_id,
                team_id=first.batting_team_id,
                opponent_team_id=first.opponent_team_id,
                plate_appearances=len(appearances),
                hits=sum(bool(item.is_hit) for item in appearances),
                player_name=first.batter_name,
            )
        )
    return tuple(sorted(results, key=lambda row: (row.game_date, row.game_id, row.player_id)))


def build_pregame_live_hit_dataset(
    rows: Iterable[PlateAppearanceInputRow],
    *,
    rolling_games: int = 10,
    hit_prior_rate: float = 0.25,
    hit_prior_pa: float = 20.0,
    any_hit_prior_rate: float = 0.5,
    any_hit_prior_games: float = 4.0,
    relation_prior_pa: float = 40.0,
) -> LiveHitFeatureDataset:
    """Build player/team/relation features using exclusively earlier dates.

    Career rates cover the supplied history, not unseen years. Smoothing uses
    fixed, configurable priors, never full-dataset or future-season averages.
    Current PA count and hit count are retained only in target metadata.
    """

    if isinstance(rolling_games, bool) or not isinstance(rolling_games, int) or rolling_games < 1:
        raise ValueError("rolling_games must be a positive integer")
    for name, rate in (
        ("hit_prior_rate", hit_prior_rate),
        ("any_hit_prior_rate", any_hit_prior_rate),
    ):
        if not math.isfinite(rate) or not 0.0 < rate < 1.0:
            raise ValueError(f"{name} must be finite and strictly between zero and one")
    for name, strength in (
        ("hit_prior_pa", hit_prior_pa),
        ("any_hit_prior_games", any_hit_prior_games),
        ("relation_prior_pa", relation_prior_pa),
    ):
        if not math.isfinite(strength) or strength <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    priors = _Priors(
        hit_prior_rate, hit_prior_pa, any_hit_prior_rate, any_hit_prior_games, relation_prior_pa
    )
    observations = reduce_player_game_live_hit_targets(rows)
    histories = _Histories(rolling_games)
    features: list[tuple[float, ...]] = []
    position = 0
    while position < len(observations):
        date_end = position + 1
        current_date = observations[position].game_date
        while date_end < len(observations) and observations[date_end].game_date == current_date:
            date_end += 1
        daily = observations[position:date_end]
        # Capture every player's features before observing any result today.
        for row in daily:
            features.append(_pregame_features(row, histories, priors))
        _record_date(daily, histories)
        position = date_end
    return LiveHitFeatureDataset(
        game_ids=tuple(row.game_id for row in observations),
        game_dates=tuple(row.game_date for row in observations),
        player_ids=tuple(row.player_id for row in observations),
        team_ids=tuple(row.team_id for row in observations),
        opponent_team_ids=tuple(row.opponent_team_id for row in observations),
        player_names=tuple(row.player_name for row in observations),
        seasons=np.asarray([row.game_date.year for row in observations], dtype=np.int64),
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray([row.target_class for row in observations], dtype=np.int64),
        plate_appearances=np.asarray(
            [row.plate_appearances for row in observations], dtype=np.int64
        ),
        hits=np.asarray([row.hits for row in observations], dtype=np.int64),
    )


def build_live_hit_fixed_season_splits(
    dataset: LiveHitFeatureDataset,
    *,
    specifications: Sequence[FixedSeasonEvaluation] = FIXED_SEASON_EVALUATIONS,
) -> tuple[SeasonDatasetSplit, ...]:
    """Use 2023→2024 validation and refitted 2023-24→2025 testing."""

    if not specifications:
        raise ValueError("at least one fixed-season evaluation is required")
    splits: list[SeasonDatasetSplit] = []
    for spec in specifications:
        train = np.flatnonzero(np.isin(dataset.seasons, spec.train_seasons)).astype(np.int64)
        evaluation = np.flatnonzero(dataset.seasons == spec.evaluation_season).astype(np.int64)
        if train.size == 0 or evaluation.size == 0:
            raise ValueError(f"{spec.name} requires non-empty training and evaluation player-games")
        if max(dataset.game_dates[index] for index in train) >= min(
            dataset.game_dates[index] for index in evaluation
        ):
            raise ValueError(f"{spec.name} training rows must precede evaluation rows")
        splits.append(SeasonDatasetSplit(spec, train, evaluation))
    return tuple(splits)


def evaluate_live_hit_fixed_season_catboost(
    rows: Iterable[PlateAppearanceInputRow] | LiveHitFeatureDataset,
    *,
    catboost_parameters: Mapping[str, object] | None = None,
    n_calibration_bins: int = 15,
    model_factory: Callable[[], _ProbabilityClassifier] | None = None,
    model_output_directory: str | Path | None = None,
    **feature_parameters: Any,
) -> dict[str, Any]:
    """Train two independent binary models and return JSON-compatible metrics.

    CatBoost is optional until fitting. Evaluation-season history may update
    the next day's features, but never the fitted model or same-day features.
    This is sequential retrospective evaluation, not a frozen season forecast.
    """

    if n_calibration_bins < 1:
        raise ValueError("n_calibration_bins must be positive")
    if model_factory is not None and catboost_parameters is not None:
        raise ValueError("catboost_parameters cannot be combined with model_factory")
    if model_factory is not None and model_output_directory is not None:
        raise ValueError("model saving requires the default CatBoost model factory")
    if isinstance(rows, LiveHitFeatureDataset):
        if feature_parameters:
            raise ValueError("feature parameters cannot be applied to an existing dataset")
        dataset = rows
    else:
        dataset = build_pregame_live_hit_dataset(rows, **feature_parameters)

    def default_factory() -> _ProbabilityClassifier:
        parameters = dict(catboost_parameters or {})
        parameters.setdefault("loss_function", "Logloss")
        parameters.setdefault("thread_count", 1)
        return CatBoostClassifierBaseline(parameters=parameters)

    factory = model_factory or default_factory
    folds: list[dict[str, Any]] = []
    for split in build_live_hit_fixed_season_splits(dataset):
        training_targets = dataset.targets[split.train_indices]
        if set(training_targets.tolist()) != {NO_HIT, ANY_HIT}:
            raise ValueError(f"{split.specification.name} training data requires both hit classes")
        classifier = factory()
        classifier.fit(dataset.features[split.train_indices], training_targets)
        model_path: Path | None = None
        if model_output_directory is not None:
            if not isinstance(classifier, CatBoostClassifierBaseline):
                raise TypeError("model saving requires CatBoostClassifierBaseline")
            model_path = Path(model_output_directory) / f"{split.specification.name}.cbm"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.model_.save_model(str(model_path))
        probabilities = _align_probabilities(
            classifier.predict_proba(dataset.features[split.evaluation_indices]),
            classifier.classes_,
        )
        targets = dataset.targets[split.evaluation_indices]
        metrics = evaluate_probabilities(targets, probabilities, n_bins=n_calibration_bins)
        prior = (np.bincount(training_targets, minlength=2) + 1.0) / (training_targets.size + 2.0)
        prior_metrics = evaluate_probabilities(
            targets,
            np.tile(prior, (targets.size, 1)),
            n_bins=n_calibration_bins,
        )
        folds.append(
            {
                "name": split.specification.name,
                "model_path": str(model_path) if model_path is not None else None,
                "phase": split.specification.phase,
                "train_seasons": list(split.specification.train_seasons),
                "evaluation_season": split.specification.evaluation_season,
                "train_player_games": int(split.train_indices.size),
                "evaluation_player_games": int(split.evaluation_indices.size),
                "train_date_start": dataset.game_dates[int(split.train_indices[0])].isoformat(),
                "train_date_end": dataset.game_dates[int(split.train_indices[-1])].isoformat(),
                "evaluation_date_start": dataset.game_dates[
                    int(split.evaluation_indices[0])
                ].isoformat(),
                "evaluation_date_end": dataset.game_dates[
                    int(split.evaluation_indices[-1])
                ].isoformat(),
                "class_counts": {
                    "train": _class_counts(training_targets),
                    "evaluation": _class_counts(targets),
                },
                "evaluation_observed_pa": int(
                    dataset.plate_appearances[split.evaluation_indices].sum()
                ),
                "metrics": {
                    **asdict(metrics),
                    "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
                },
                "prior_baseline": {
                    "method": "training_class_prevalence_laplace_1",
                    "probabilities": prior.tolist(),
                    "metrics": {
                        **asdict(prior_metrics),
                        "accuracy": float(np.mean(np.argmax(prior) == targets)),
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "task": "kbo_player_game_any_hit_conditional_on_appearance",
        "class_order": list(LIVE_HIT_LABELS),
        "class_indices": {label: index for index, label in enumerate(LIVE_HIT_LABELS)},
        "model": {
            "name": "CatBoostClassifierBaseline",
            "parameter_overrides": dict(catboost_parameters or {}),
            "effective_parameters": {
                **DEFAULT_CATBOOST_PARAMETERS,
                "loss_function": "Logloss",
                "thread_count": 1,
                **dict(catboost_parameters or {}),
            },
        },
        "population": "player_game_with_at_least_one_observed_plate_appearance",
        "target": "at_least_one_hit_in_the_game",
        "feature_generation": {
            "causal_order": "features_before_current_date_result_updates",
            "same_day_policy": "simultaneous",
            "history_update_policy": "earlier_dates_only_including_completed_evaluation_dates",
            "current_game_pa_and_hits": "target_metadata_only_not_features",
            "feature_parameter_overrides": dict(feature_parameters),
            "feature_names": list(dataset.feature_names),
        },
        "dataset_player_games": dataset.row_count,
        "dataset_observed_pa": int(dataset.plate_appearances.sum()),
        "folds": folds,
        "limitations": [
            "Conditional on observed appearance (PA >= 1); candidate selection and appearance "
            "probabilities require a separate model and separate evaluation.",
            "Non-appearing candidates are not labeled as no-hit in this experiment.",
            "Team aggregates require the complete input PA population; source gaps bias them.",
            "Names are metadata only; career history starts at the earliest supplied season.",
            "This is not an end-to-end V26 recommendation or bonus/selection-rate optimizer.",
        ],
    }


def evaluate_live_hit_fixed_season_catboost_json(
    rows: Iterable[PlateAppearanceInputRow] | LiveHitFeatureDataset,
    **kwargs: Any,
) -> str:
    """Evaluate and serialize using the same strict JSON contract as match play."""

    return render_evaluation_json(evaluate_live_hit_fixed_season_catboost(rows, **kwargs))


def _canonical_pa(raw: PlateAppearanceInputRow) -> CanonicalPlateAppearanceRow:
    if isinstance(raw, CanonicalPlateAppearanceRow):
        return raw
    if isinstance(raw, Mapping):
        return CanonicalPlateAppearanceRow(
            plate_appearance_id=_required(raw, "plate_appearance_id", "pa_id"),
            game_id=_required(raw, "game_id"),
            game_date=_required(raw, "game_date", "scheduled_start"),
            batter_id=_required(raw, "batter_id", "player_id"),
            batting_team_id=_required(raw, "batting_team_id", "team_id"),
            opponent_team_id=_required(raw, "opponent_team_id", "fielding_team_id"),
            outcome=_required(raw, "outcome"),
            is_hit=raw.get("is_hit"),
            batter_name=raw.get("batter_name", raw.get("player_name")),
            batting_team_name=raw.get("batting_team_name", raw.get("team_name")),
            opponent_team_name=raw.get("opponent_team_name", raw.get("fielding_team_name")),
        )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("PA rows must be canonical objects, mappings, or sequences")
    if len(raw) in (7, 8):
        pa_id, game_id, game_date, batter, team, opponent, outcome = raw[:7]
        return CanonicalPlateAppearanceRow(
            pa_id,
            game_id,
            game_date,
            batter,
            team,
            opponent,
            outcome,
            is_hit=raw[7] if len(raw) == 8 else None,
        )
    if len(raw) == 11:
        (
            pa_id,
            game_id,
            game_date,
            batter,
            name,
            team,
            team_name,
            opponent,
            opp_name,
            outcome,
            hit,
        ) = raw
        return CanonicalPlateAppearanceRow(
            pa_id,
            game_id,
            game_date,
            batter,
            team,
            opponent,
            outcome,
            is_hit=hit,
            batter_name=name,
            batting_team_name=team_name,
            opponent_team_name=opp_name,
        )
    raise ValueError("positional PA rows must contain seven, eight, or eleven values")


def _required(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            if row[name] is None:
                raise ValueError(f"{name} cannot be null")
            return row[name]
    raise ValueError("PA row lacks required field: " + " or ".join(names))


def _coerce_game_date(value: Any) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(ZoneInfo("Asia/Seoul")).date()
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("game_date cannot be NaT")
        return date.fromisoformat(np.datetime_as_string(value, unit="D"))
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return _coerce_game_date(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    raise TypeError(f"unsupported game_date type: {type(value).__name__}")


def _pregame_features(
    row: PlayerGameLiveHitRow, histories: _Histories, priors: _Priors
) -> tuple[float, ...]:
    player = histories.get(histories.players, row.player_id)
    offense = histories.get(histories.offense, row.team_id)
    defense = histories.get(histories.defense, row.opponent_team_id)
    relation = histories.relations.get((row.player_id, row.opponent_team_id), _Counts(0, 0))
    return (
        float(player.games),
        float(player.pa),
        float(player.hits),
        priors.hit(player.hits, player.pa),
        priors.any_hit(player.hit_games, player.games),
        float(len(player.recent)),
        float(player.recent_pa),
        float(player.recent_hits),
        priors.hit(player.recent_hits, player.recent_pa),
        priors.any_hit(sum(item.hits > 0 for item in player.recent), len(player.recent)),
        float(offense.games),
        float(offense.pa),
        float(offense.hits),
        priors.hit(offense.hits, offense.pa),
        float(len(offense.recent)),
        priors.hit(offense.recent_hits, offense.recent_pa),
        float(defense.games),
        float(defense.pa),
        float(defense.hits),
        priors.hit(defense.hits, defense.pa),
        float(len(defense.recent)),
        priors.hit(defense.recent_hits, defense.recent_pa),
        float(relation.pa),
        float(relation.hits),
        priors.hit(relation.hits, relation.pa, relation=True),
    )


def _record_date(rows: Sequence[PlayerGameLiveHitRow], histories: _Histories) -> None:
    team_games: dict[tuple[str, str, str], _Counts] = {}
    for row in rows:
        counts = _Counts(row.plate_appearances, row.hits)
        histories.get(histories.players, row.player_id).record(counts)
        relation_key = (row.player_id, row.opponent_team_id)
        prior = histories.relations.get(relation_key, _Counts(0, 0))
        histories.relations[relation_key] = _Counts(prior.pa + counts.pa, prior.hits + counts.hits)
        team_key = (row.game_id, row.team_id, row.opponent_team_id)
        prior_team = team_games.get(team_key, _Counts(0, 0))
        team_games[team_key] = _Counts(prior_team.pa + counts.pa, prior_team.hits + counts.hits)
    # A team's recent window advances once per game, not once per batter.
    for (_, team, opponent), counts in team_games.items():
        histories.get(histories.offense, team).record(counts)
        histories.get(histories.defense, opponent).record(counts)


def _align_probabilities(probabilities: Any, classes: Any) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise RuntimeError("classifier must return two probability columns")
    if class_values.ndim != 1 or class_values.size != 2:
        raise RuntimeError("classifier must retain both hit classes")
    if set(class_values.tolist()) != {NO_HIT, ANY_HIT}:
        raise RuntimeError("classifier returned unknown or duplicate hit classes")
    aligned = values[
        :, [int(np.flatnonzero(class_values == label)[0]) for label in (NO_HIT, ANY_HIT)]
    ]
    if not np.all(np.isfinite(aligned)) or np.any(aligned < 0.0):
        raise RuntimeError("classifier returned invalid probabilities")
    totals = aligned.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-7, atol=1e-9):
        raise RuntimeError("classifier probabilities do not sum to one")
    return np.asarray(aligned / totals[:, None], dtype=np.float64)


def _class_counts(targets: NDArray[np.int64]) -> dict[str, int]:
    counts = Counter(int(value) for value in targets)
    return {label: counts[index] for index, label in enumerate(LIVE_HIT_LABELS)}


__all__ = [
    "ANY_HIT",
    "NO_HIT",
    "LIVE_HIT_CANONICAL_SQL",
    "LIVE_HIT_FEATURE_NAMES",
    "LIVE_HIT_LABELS",
    "CanonicalPlateAppearanceRow",
    "LiveHitFeatureDataset",
    "PlateAppearanceInputRow",
    "PlayerGameLiveHitRow",
    "build_live_hit_fixed_season_splits",
    "build_pregame_live_hit_dataset",
    "canonicalize_plate_appearance_rows",
    "evaluate_live_hit_fixed_season_catboost",
    "evaluate_live_hit_fixed_season_catboost_json",
    "reduce_player_game_live_hit_targets",
]
