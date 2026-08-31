"""Leakage-safe KBO match-result baseline over canonical completed games.

The feature builder is intentionally independent of a provider and DuckDB. It
accepts canonical objects, mapping rows, or positional rows returned by a
DuckDB query. Every feature for a game is captured before that game's result
updates the rolling statistics or Elo ratings.

When only a calendar date is available, all games on that date are treated as
simultaneous. This conservative rule prevents an earlier-ordered row from
leaking its result into another same-day game (including doubleheaders).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from cpv26.data.kbo_source_snapshots import source_snapshot_filter_sql
from cpv26.evaluation import evaluate_probabilities
from cpv26.models.baseline import DEFAULT_CATBOOST_PARAMETERS, CatBoostClassifierBaseline

HOME_LOSS = 0
DRAW = 1
HOME_WIN = 2
HOME_RESULT_LABELS: tuple[str, str, str] = ("L", "D", "W")

# Pick the latest revision before filtering its state or season. Filtering
# first could resurrect an older final result after a cancellation/correction.
MATCH_CANONICAL_SQL = f"""
WITH latest_game AS (
    SELECT * FROM game
    WHERE {source_snapshot_filter_sql()}
    QUALIFY row_number() OVER (
        PARTITION BY game_id
        ORDER BY available_at DESC, ingested_at DESC, valid_from DESC, game_row_id DESC
    ) = 1
)
SELECT
    game_id,
    CAST(scheduled_start AT TIME ZONE 'Asia/Seoul' AS DATE) AS game_date,
    home_team_id,
    away_team_id,
    home_score,
    away_score
FROM latest_game
WHERE valid_to IS NULL
  AND season BETWEEN 2023 AND 2025
  AND game_status = 'final'
  AND home_score IS NOT NULL
  AND away_score IS NOT NULL
ORDER BY game_date, game_id
"""

MATCH_FEATURE_NAMES: tuple[str, ...] = (
    "home_elo_before",
    "away_elo_before",
    "elo_difference_before",
    "home_elo_expected_score",
    "home_games_before",
    "away_games_before",
    "home_points_rate_before",
    "away_points_rate_before",
    "points_rate_difference_before",
    "home_draw_rate_before",
    "away_draw_rate_before",
    "home_runs_for_per_game_before",
    "away_runs_for_per_game_before",
    "home_runs_against_per_game_before",
    "away_runs_against_per_game_before",
    "home_run_differential_per_game_before",
    "away_run_differential_per_game_before",
    "run_differential_difference_before",
    "home_recent_games_before",
    "away_recent_games_before",
    "home_recent_points_rate_before",
    "away_recent_points_rate_before",
    "recent_points_rate_difference_before",
    "home_recent_run_differential_before",
    "away_recent_run_differential_before",
    "recent_run_differential_difference_before",
)


@dataclass(frozen=True, slots=True)
class CanonicalGameRow:
    """One completed game at the source-neutral match-result grain."""

    game_date: date
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    game_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_date", _coerce_game_date(self.game_date))
        for name in ("home_team_id", "away_team_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.home_team_id == self.away_team_id:
            raise ValueError("home and away teams must be different")
        object.__setattr__(self, "home_score", _coerce_score(self.home_score, "home_score"))
        object.__setattr__(self, "away_score", _coerce_score(self.away_score, "away_score"))
        if self.game_id is not None:
            if not isinstance(self.game_id, str) or not self.game_id.strip():
                raise ValueError("game_id must be a non-empty string or None")
            object.__setattr__(self, "game_id", self.game_id.strip())

    @property
    def season(self) -> int:
        return self.game_date.year

    @property
    def target_class(self) -> int:
        if self.home_score < self.away_score:
            return HOME_LOSS
        if self.home_score == self.away_score:
            return DRAW
        return HOME_WIN


GameInputRow = CanonicalGameRow | Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True, slots=True)
class MatchFeatureDataset:
    """Chronological numeric features and three-class home-result targets."""

    game_ids: tuple[str, ...]
    game_dates: tuple[date, ...]
    home_team_ids: tuple[str, ...]
    away_team_ids: tuple[str, ...]
    seasons: NDArray[np.int64]
    features: NDArray[np.float64]
    targets: NDArray[np.int64]
    feature_names: tuple[str, ...] = MATCH_FEATURE_NAMES

    def __post_init__(self) -> None:
        game_ids = tuple(self.game_ids)
        game_dates = tuple(self.game_dates)
        home_team_ids = tuple(self.home_team_ids)
        away_team_ids = tuple(self.away_team_ids)
        row_count = len(game_ids)
        if row_count == 0:
            raise ValueError("match dataset cannot be empty")
        if len(set(game_ids)) != row_count:
            raise ValueError("game_ids must be unique")
        if not (len(game_dates) == len(home_team_ids) == len(away_team_ids) == row_count):
            raise ValueError("match dataset identity columns must have equal length")
        feature_names = tuple(self.feature_names)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be non-empty and unique")

        seasons = np.asarray(self.seasons, dtype=np.int64).copy()
        features = np.asarray(self.features, dtype=np.float64).copy()
        targets = np.asarray(self.targets, dtype=np.int64).copy()
        if seasons.shape != (row_count,):
            raise ValueError("seasons must contain one value per game")
        if features.shape != (row_count, len(feature_names)):
            raise ValueError("features shape does not match rows and feature_names")
        if targets.shape != (row_count,):
            raise ValueError("targets must contain one class per game")
        if not np.all(np.isfinite(features)):
            raise ValueError("match features must be finite")
        if np.any((targets < HOME_LOSS) | (targets > HOME_WIN)):
            raise ValueError("targets must use 0=L, 1=D, 2=W")
        if any(left > right for left, right in zip(game_dates, game_dates[1:], strict=False)):
            raise ValueError("match dataset must be chronological")

        seasons.flags.writeable = False
        features.flags.writeable = False
        targets.flags.writeable = False
        object.__setattr__(self, "game_ids", game_ids)
        object.__setattr__(self, "game_dates", game_dates)
        object.__setattr__(self, "home_team_ids", home_team_ids)
        object.__setattr__(self, "away_team_ids", away_team_ids)
        object.__setattr__(self, "seasons", seasons)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "feature_names", feature_names)

    @property
    def row_count(self) -> int:
        return len(self.game_ids)


@dataclass(frozen=True, slots=True)
class FixedSeasonEvaluation:
    """One required train/evaluation season boundary."""

    name: str
    phase: str
    train_seasons: tuple[int, ...]
    evaluation_season: int

    def __post_init__(self) -> None:
        if not self.name or not self.phase:
            raise ValueError("season evaluation name and phase cannot be empty")
        train_seasons = tuple(int(value) for value in self.train_seasons)
        if not train_seasons or len(set(train_seasons)) != len(train_seasons):
            raise ValueError("train_seasons must be non-empty and unique")
        if tuple(sorted(train_seasons)) != train_seasons:
            raise ValueError("train_seasons must be chronological")
        if max(train_seasons) >= self.evaluation_season:
            raise ValueError("training seasons must precede the evaluation season")
        object.__setattr__(self, "train_seasons", train_seasons)


FIXED_SEASON_EVALUATIONS: tuple[FixedSeasonEvaluation, ...] = (
    FixedSeasonEvaluation(
        name="validation_2024",
        phase="validation",
        train_seasons=(2023,),
        evaluation_season=2024,
    ),
    FixedSeasonEvaluation(
        name="test_2025",
        phase="test",
        train_seasons=(2023, 2024),
        evaluation_season=2025,
    ),
)


@dataclass(frozen=True, slots=True)
class SeasonDatasetSplit:
    specification: FixedSeasonEvaluation
    train_indices: NDArray[np.int64]
    evaluation_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64).copy()
        evaluation = np.asarray(self.evaluation_indices, dtype=np.int64).copy()
        if train.ndim != 1 or evaluation.ndim != 1:
            raise ValueError("season split indices must be one-dimensional")
        if train.size == 0 or evaluation.size == 0:
            raise ValueError("season split train and evaluation sets cannot be empty")
        if np.intersect1d(train, evaluation).size:
            raise ValueError("season split train and evaluation rows cannot overlap")
        train.flags.writeable = False
        evaluation.flags.writeable = False
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "evaluation_indices", evaluation)


@dataclass(frozen=True, slots=True)
class _TeamGameObservation:
    result_score: float
    runs_for: int
    runs_against: int


@dataclass(slots=True)
class _TeamState:
    elo: float
    recent: deque[_TeamGameObservation]
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    runs_for: int = 0
    runs_against: int = 0

    def record(self, observation: _TeamGameObservation) -> None:
        self.games += 1
        if observation.result_score == 1.0:
            self.wins += 1
        elif observation.result_score == 0.5:
            self.draws += 1
        else:
            self.losses += 1
        self.runs_for += observation.runs_for
        self.runs_against += observation.runs_against
        self.recent.append(observation)

    @property
    def points_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.5

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def runs_for_per_game(self) -> float:
        return self.runs_for / self.games if self.games else 0.0

    @property
    def runs_against_per_game(self) -> float:
        return self.runs_against / self.games if self.games else 0.0

    @property
    def run_differential_per_game(self) -> float:
        return (self.runs_for - self.runs_against) / self.games if self.games else 0.0

    @property
    def recent_points_rate(self) -> float:
        if not self.recent:
            return 0.5
        return sum(item.result_score for item in self.recent) / len(self.recent)

    @property
    def recent_run_differential(self) -> float:
        if not self.recent:
            return 0.0
        return sum(item.runs_for - item.runs_against for item in self.recent) / len(self.recent)


class _ProbabilityClassifier(Protocol):
    @property
    def classes_(self) -> NDArray[Any]: ...

    def fit(self, features: Any, targets: Any) -> Any: ...

    def predict_proba(self, features: Any) -> Any: ...


def canonicalize_game_rows(rows: Iterable[GameInputRow]) -> tuple[CanonicalGameRow, ...]:
    """Normalize canonical objects, mappings, or DuckDB-style positional rows.

    Positional rows must be either ``(date, home, away, home_score,
    away_score)`` or ``(game_id, date, home, away, home_score, away_score)``.
    Mapping rows accept canonical schema names and the public-data aliases
    ``game_pk``, ``home_team``, and ``away_team``.
    """

    materialized: list[tuple[int, CanonicalGameRow]] = []
    generated_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for input_index, raw in enumerate(rows):
        game = _canonical_game_row(raw)
        if game.game_id is None:
            base = f"{game.game_date.isoformat()}:{game.away_team_id}@{game.home_team_id}"
            generated_counts[base] += 1
            suffix = generated_counts[base]
            game_id = base if suffix == 1 else f"{base}#{suffix}"
            game = replace(game, game_id=game_id)
        game_id = cast(str, game.game_id)
        if game_id in seen_ids:
            raise ValueError(f"duplicate game_id: {game_id}")
        seen_ids.add(game_id)
        materialized.append((input_index, game))
    if not materialized:
        raise ValueError("at least one completed game row is required")
    materialized.sort(key=lambda item: (item[1].game_date, item[0]))
    return tuple(item[1] for item in materialized)


def build_pregame_match_dataset(
    rows: Iterable[GameInputRow],
    *,
    rolling_games: int = 10,
    initial_elo: float = 1_500.0,
    elo_k_factor: float = 20.0,
    elo_home_advantage: float = 50.0,
) -> MatchFeatureDataset:
    """Build pregame rolling/Elo features, then update with each final result."""

    if rolling_games < 1:
        raise ValueError("rolling_games must be positive")
    for name, value in {
        "initial_elo": initial_elo,
        "elo_k_factor": elo_k_factor,
        "elo_home_advantage": elo_home_advantage,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if elo_k_factor <= 0.0:
        raise ValueError("elo_k_factor must be positive")

    games = canonicalize_game_rows(rows)
    states: dict[str, _TeamState] = {}

    def state(team_id: str) -> _TeamState:
        if team_id not in states:
            states[team_id] = _TeamState(
                elo=float(initial_elo),
                recent=deque(maxlen=rolling_games),
            )
        return states[team_id]

    feature_rows: list[tuple[float, ...]] = []
    targets: list[int] = []
    position = 0
    while position < len(games):
        current_date = games[position].game_date
        date_end = position
        while date_end < len(games) and games[date_end].game_date == current_date:
            date_end += 1
        daily_games = games[position:date_end]

        for game in daily_games:
            home = state(game.home_team_id)
            away = state(game.away_team_id)
            feature_rows.append(
                _pregame_features(
                    home,
                    away,
                    elo_home_advantage=elo_home_advantage,
                )
            )
            targets.append(game.target_class)

        # Compute every same-day Elo delta from the same pre-day ratings. This
        # avoids row-order leakage when only a calendar date is available.
        elo_deltas: defaultdict[str, float] = defaultdict(float)
        for game in daily_games:
            home = state(game.home_team_id)
            away = state(game.away_team_id)
            expected = _elo_expected_score(
                home.elo,
                away.elo,
                home_advantage=elo_home_advantage,
            )
            actual = _home_result_score(game.target_class)
            delta = elo_k_factor * (actual - expected)
            elo_deltas[game.home_team_id] += delta
            elo_deltas[game.away_team_id] -= delta
        for team_id, delta in elo_deltas.items():
            state(team_id).elo += delta

        for game in daily_games:
            home_score = _home_result_score(game.target_class)
            state(game.home_team_id).record(
                _TeamGameObservation(home_score, game.home_score, game.away_score)
            )
            state(game.away_team_id).record(
                _TeamGameObservation(1.0 - home_score, game.away_score, game.home_score)
            )
        position = date_end

    return MatchFeatureDataset(
        game_ids=tuple(cast(str, game.game_id) for game in games),
        game_dates=tuple(game.game_date for game in games),
        home_team_ids=tuple(game.home_team_id for game in games),
        away_team_ids=tuple(game.away_team_id for game in games),
        seasons=np.asarray([game.season for game in games], dtype=np.int64),
        features=np.asarray(feature_rows, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.int64),
    )


def build_fixed_season_splits(
    dataset: MatchFeatureDataset,
    *,
    specifications: Sequence[FixedSeasonEvaluation] = FIXED_SEASON_EVALUATIONS,
) -> tuple[SeasonDatasetSplit, ...]:
    """Materialize the required 2023→2024 and 2023-24→2025 splits."""

    specs = tuple(specifications)
    if not specs:
        raise ValueError("at least one fixed-season evaluation is required")
    splits: list[SeasonDatasetSplit] = []
    for spec in specs:
        train = np.flatnonzero(np.isin(dataset.seasons, spec.train_seasons)).astype(np.int64)
        evaluation = np.flatnonzero(dataset.seasons == spec.evaluation_season).astype(np.int64)
        if train.size == 0:
            raise ValueError(f"{spec.name} has no training games")
        if evaluation.size == 0:
            raise ValueError(f"{spec.name} has no evaluation games")
        if max(dataset.game_dates[index] for index in train) >= min(
            dataset.game_dates[index] for index in evaluation
        ):
            raise ValueError(f"{spec.name} training rows do not precede evaluation rows")
        splits.append(SeasonDatasetSplit(spec, train, evaluation))
    return tuple(splits)


def evaluate_fixed_season_catboost(
    rows: Iterable[GameInputRow] | MatchFeatureDataset,
    *,
    rolling_games: int = 10,
    initial_elo: float = 1_500.0,
    elo_k_factor: float = 20.0,
    elo_home_advantage: float = 50.0,
    catboost_parameters: Mapping[str, object] | None = None,
    n_calibration_bins: int = 15,
    model_factory: Callable[[], _ProbabilityClassifier] | None = None,
    model_output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Fit independent CatBoost models at the fixed validation/test boundaries.

    CatBoost remains optional: merely importing this module or building
    features does not import it. The dependency is first required when the
    default model factory calls :meth:`CatBoostClassifierBaseline.fit`.
    """

    if n_calibration_bins < 1:
        raise ValueError("n_calibration_bins must be positive")
    if isinstance(rows, MatchFeatureDataset):
        dataset = rows
    else:
        dataset = build_pregame_match_dataset(
            rows,
            rolling_games=rolling_games,
            initial_elo=initial_elo,
            elo_k_factor=elo_k_factor,
            elo_home_advantage=elo_home_advantage,
        )
    if model_factory is not None and catboost_parameters is not None:
        raise ValueError("catboost_parameters cannot be combined with model_factory")
    if model_factory is not None and model_output_directory is not None:
        raise ValueError("model saving requires the default CatBoost model factory")

    def default_factory() -> _ProbabilityClassifier:
        parameters = dict(catboost_parameters or {})
        parameters.setdefault("loss_function", "MultiClass")
        parameters.setdefault("thread_count", 1)
        return CatBoostClassifierBaseline(parameters=parameters)

    factory = model_factory or default_factory
    fold_results: list[dict[str, Any]] = []
    for split in build_fixed_season_splits(dataset):
        train_targets = dataset.targets[split.train_indices]
        _require_three_training_classes(train_targets, split.specification.name)
        classifier = factory()
        classifier.fit(dataset.features[split.train_indices], train_targets)
        model_path: Path | None = None
        if model_output_directory is not None:
            if not isinstance(classifier, CatBoostClassifierBaseline):
                raise TypeError("model saving requires CatBoostClassifierBaseline")
            model_path = Path(model_output_directory) / f"{split.specification.name}.cbm"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.model_.save_model(str(model_path))
        raw_probabilities = classifier.predict_proba(dataset.features[split.evaluation_indices])
        probabilities = _align_probabilities(raw_probabilities, classifier.classes_)
        evaluation_targets = dataset.targets[split.evaluation_indices]
        metrics = evaluate_probabilities(
            evaluation_targets,
            probabilities,
            n_bins=n_calibration_bins,
        )
        predictions = np.argmax(probabilities, axis=1)
        prior = (np.bincount(train_targets, minlength=3) + 1.0) / (train_targets.size + 3.0)
        prior_probabilities = np.tile(prior, (evaluation_targets.size, 1))
        prior_metrics = evaluate_probabilities(
            evaluation_targets,
            prior_probabilities,
            n_bins=n_calibration_bins,
        )
        fold_results.append(
            {
                "name": split.specification.name,
                "model_path": str(model_path) if model_path is not None else None,
                "phase": split.specification.phase,
                "train_seasons": list(split.specification.train_seasons),
                "evaluation_season": split.specification.evaluation_season,
                "train_games": int(split.train_indices.size),
                "evaluation_games": int(split.evaluation_indices.size),
                "train_date_start": min(
                    dataset.game_dates[index] for index in split.train_indices
                ).isoformat(),
                "train_date_end": max(
                    dataset.game_dates[index] for index in split.train_indices
                ).isoformat(),
                "evaluation_date_start": min(
                    dataset.game_dates[index] for index in split.evaluation_indices
                ).isoformat(),
                "evaluation_date_end": max(
                    dataset.game_dates[index] for index in split.evaluation_indices
                ).isoformat(),
                "class_counts": {
                    "train": _class_counts(train_targets),
                    "evaluation": _class_counts(evaluation_targets),
                },
                "metrics": {
                    **asdict(metrics),
                    "accuracy": float(np.mean(predictions == evaluation_targets)),
                },
                "prior_baseline": {
                    "method": "training_class_prevalence_laplace_1",
                    "probabilities": prior.tolist(),
                    "metrics": {
                        **asdict(prior_metrics),
                        "accuracy": float(np.mean(np.argmax(prior) == evaluation_targets)),
                    },
                },
            }
        )

    return {
        "schema_version": 1,
        "task": "kbo_home_result_3class",
        "class_order": list(HOME_RESULT_LABELS),
        "class_indices": {label: index for index, label in enumerate(HOME_RESULT_LABELS)},
        "model": {
            "name": "CatBoostClassifierBaseline",
            "parameter_overrides": dict(catboost_parameters or {}),
            "effective_parameters": {
                **DEFAULT_CATBOOST_PARAMETERS,
                "loss_function": "MultiClass",
                "thread_count": 1,
                **dict(catboost_parameters or {}),
            },
        },
        "feature_generation": {
            "causal_order": "features_before_current_result_update",
            "same_day_policy": "simultaneous",
            "rolling_games": rolling_games,
            "initial_elo": initial_elo,
            "elo_k_factor": elo_k_factor,
            "elo_home_advantage": elo_home_advantage,
            "feature_names": list(dataset.feature_names),
        },
        "dataset_games": dataset.row_count,
        "folds": fold_results,
    }


def render_evaluation_json(result: Mapping[str, Any], *, indent: int = 2) -> str:
    """Return deterministic, strict JSON suitable for an evaluation artifact."""

    if indent < 0:
        raise ValueError("indent cannot be negative")
    return (
        json.dumps(
            dict(result),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    )


def evaluate_fixed_season_catboost_json(
    rows: Iterable[GameInputRow] | MatchFeatureDataset,
    **kwargs: Any,
) -> str:
    """Evaluate the fixed folds and serialize the result as strict JSON."""

    return render_evaluation_json(evaluate_fixed_season_catboost(rows, **kwargs))


def _canonical_game_row(raw: GameInputRow) -> CanonicalGameRow:
    if isinstance(raw, CanonicalGameRow):
        return raw
    if isinstance(raw, Mapping):
        return CanonicalGameRow(
            game_id=_optional_mapping_value(raw, "game_id", "game_pk"),
            game_date=_required_mapping_value(raw, "game_date", "scheduled_start"),
            home_team_id=_required_mapping_value(raw, "home_team_id", "home_team"),
            away_team_id=_required_mapping_value(raw, "away_team_id", "away_team"),
            home_score=_required_mapping_value(raw, "home_score"),
            away_score=_required_mapping_value(raw, "away_score"),
        )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("game rows must be canonical objects, mappings, or sequences")
    if len(raw) == 5:
        game_date, home_team, away_team, home_score, away_score = raw
        game_id = None
    elif len(raw) == 6:
        game_id, game_date, home_team, away_team, home_score, away_score = raw
    else:
        raise ValueError("positional game rows must contain five or six values")
    return CanonicalGameRow(
        game_id=game_id,
        game_date=game_date,
        home_team_id=home_team,
        away_team_id=away_team,
        home_score=home_score,
        away_score=away_score,
    )


def _required_mapping_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            value = row[name]
            if value is None:
                raise ValueError(f"{name} cannot be null for a completed game")
            return value
    raise ValueError("game row lacks required field: " + " or ".join(names))


def _optional_mapping_value(row: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in row:
            return row[name]
    return None


def _coerce_game_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("game_date cannot be NaT")
        return date.fromisoformat(np.datetime_as_string(value, unit="D"))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("game_date cannot be empty")
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
            except ValueError as exc:
                raise ValueError(f"invalid game_date: {value!r}") from exc
    raise TypeError(f"unsupported game_date type: {type(value).__name__}")


def _coerce_score(value: Any, name: str) -> int:
    if value is None or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if numeric < 0:
        raise ValueError(f"{name} cannot be negative")
    return numeric


def _pregame_features(
    home: _TeamState,
    away: _TeamState,
    *,
    elo_home_advantage: float,
) -> tuple[float, ...]:
    elo_expected = _elo_expected_score(
        home.elo,
        away.elo,
        home_advantage=elo_home_advantage,
    )
    return (
        home.elo,
        away.elo,
        home.elo - away.elo,
        elo_expected,
        float(home.games),
        float(away.games),
        home.points_rate,
        away.points_rate,
        home.points_rate - away.points_rate,
        home.draw_rate,
        away.draw_rate,
        home.runs_for_per_game,
        away.runs_for_per_game,
        home.runs_against_per_game,
        away.runs_against_per_game,
        home.run_differential_per_game,
        away.run_differential_per_game,
        home.run_differential_per_game - away.run_differential_per_game,
        float(len(home.recent)),
        float(len(away.recent)),
        home.recent_points_rate,
        away.recent_points_rate,
        home.recent_points_rate - away.recent_points_rate,
        home.recent_run_differential,
        away.recent_run_differential,
        home.recent_run_differential - away.recent_run_differential,
    )


def _elo_expected_score(
    home_elo: float,
    away_elo: float,
    *,
    home_advantage: float,
) -> float:
    exponent = (away_elo - (home_elo + home_advantage)) / 400.0
    return float(1.0 / (1.0 + 10.0**exponent))


def _home_result_score(target_class: int) -> float:
    return (0.0, 0.5, 1.0)[target_class]


def _require_three_training_classes(targets: NDArray[np.int64], fold_name: str) -> None:
    observed = {int(value) for value in np.unique(targets)}
    missing = sorted(set(range(len(HOME_RESULT_LABELS))).difference(observed))
    if missing:
        labels = ", ".join(HOME_RESULT_LABELS[index] for index in missing)
        raise ValueError(f"{fold_name} training data lacks outcome classes: {labels}")


def _align_probabilities(probabilities: Any, classes: Any) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    if values.ndim != 2 or values.shape[0] == 0:
        raise RuntimeError("classifier returned an invalid probability matrix")
    if class_values.ndim != 1 or class_values.size != values.shape[1]:
        raise RuntimeError("classifier classes do not match probability columns")
    aligned = np.zeros((values.shape[0], len(HOME_RESULT_LABELS)), dtype=np.float64)
    seen: set[int] = set()
    for source_index, raw_class in enumerate(class_values):
        class_index = int(raw_class)
        if class_index not in range(len(HOME_RESULT_LABELS)) or class_index in seen:
            raise RuntimeError("classifier returned unknown or duplicate outcome classes")
        aligned[:, class_index] = values[:, source_index]
        seen.add(class_index)
    if seen != set(range(len(HOME_RESULT_LABELS))):
        raise RuntimeError("classifier did not retain all three outcome classes")
    if not np.all(np.isfinite(aligned)) or np.any(aligned < 0.0):
        raise RuntimeError("classifier returned invalid probabilities")
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-9):
        raise RuntimeError("classifier probabilities do not sum to one")
    normalized = np.asarray(aligned / row_sums[:, None], dtype=np.float64)
    return normalized


def _class_counts(targets: NDArray[np.int64]) -> dict[str, int]:
    counts = Counter(int(value) for value in targets)
    return {label: counts[index] for index, label in enumerate(HOME_RESULT_LABELS)}


__all__ = [
    "DRAW",
    "FIXED_SEASON_EVALUATIONS",
    "HOME_LOSS",
    "HOME_RESULT_LABELS",
    "HOME_WIN",
    "MATCH_CANONICAL_SQL",
    "MATCH_FEATURE_NAMES",
    "CanonicalGameRow",
    "FixedSeasonEvaluation",
    "MatchFeatureDataset",
    "SeasonDatasetSplit",
    "build_fixed_season_splits",
    "build_pregame_match_dataset",
    "canonicalize_game_rows",
    "evaluate_fixed_season_catboost",
    "evaluate_fixed_season_catboost_json",
    "render_evaluation_json",
]
