"""Time-ordered evaluation contracts and dependency-light probability metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

TimestampLike = date | datetime | np.datetime64


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Indices and temporal boundaries for one expanding validation fold."""

    fold_index: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    train_end: np.datetime64
    validation_start: np.datetime64
    validation_end: np.datetime64

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64).copy()
        validation = np.asarray(self.validation_indices, dtype=np.int64).copy()
        if train.ndim != 1 or validation.ndim != 1:
            raise ValueError("fold indices must be one-dimensional")
        if train.size == 0 or validation.size == 0:
            raise ValueError("each fold requires non-empty train and validation indices")
        if np.intersect1d(train, validation).size:
            raise ValueError("train and validation indices cannot overlap")
        train.flags.writeable = False
        validation.flags.writeable = False
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "validation_indices", validation)


def expanding_walk_forward_split(
    timestamps: Iterable[TimestampLike],
    *,
    min_train_periods: int,
    validation_periods: int,
    step_periods: int | None = None,
    gap_periods: int = 0,
    max_folds: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding folds over distinct, chronologically ordered periods.

    Rows sharing the same timestamp are kept together. Returned indices refer
    to the original input but are ordered chronologically inside each fold.
    ``gap_periods`` leaves complete periods between train and validation.
    """

    if min_train_periods < 1:
        raise ValueError("min_train_periods must be positive")
    if validation_periods < 1:
        raise ValueError("validation_periods must be positive")
    step = validation_periods if step_periods is None else step_periods
    if step < 1:
        raise ValueError("step_periods must be positive")
    if gap_periods < 0:
        raise ValueError("gap_periods cannot be negative")
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive")

    converted = tuple(_to_datetime64(value) for value in timestamps)
    if not converted:
        raise ValueError("timestamps cannot be empty")
    timestamp_array = np.asarray(converted, dtype="datetime64[ns]")
    order = np.asarray(np.argsort(timestamp_array, kind="stable"), dtype=np.int64)
    ordered_timestamps = timestamp_array[order]
    periods = np.unique(ordered_timestamps)
    required = min_train_periods + gap_periods + validation_periods
    if periods.size < required:
        raise ValueError(
            f"need at least {required} distinct periods, received {periods.size}"
        )

    folds: list[WalkForwardFold] = []
    train_period_count = min_train_periods
    while True:
        validation_start_position = train_period_count + gap_periods
        validation_end_position = validation_start_position + validation_periods
        if validation_end_position > periods.size:
            break
        train_end = periods[train_period_count - 1]
        validation_start = periods[validation_start_position]
        validation_end = periods[validation_end_position - 1]
        train_mask = ordered_timestamps <= train_end
        validation_mask = (ordered_timestamps >= validation_start) & (
            ordered_timestamps <= validation_end
        )
        folds.append(
            WalkForwardFold(
                fold_index=len(folds),
                train_indices=order[train_mask],
                validation_indices=order[validation_mask],
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
        train_period_count += step
    if not folds:
        raise ValueError("split configuration produced no complete validation fold")
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    log_loss: float
    brier_score: float
    expected_calibration_error: float


def multiclass_log_loss(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    labels: Sequence[object] | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Compute weighted multiclass negative log likelihood."""

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")
    target_indices, values, weights = _metric_inputs(
        targets, probabilities, sample_weight=sample_weight, labels=labels
    )
    selected = values[np.arange(values.shape[0]), target_indices]
    losses = -np.log(np.clip(selected, epsilon, 1.0))
    return _weighted_mean(losses, weights)


def multiclass_brier_score(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    labels: Sequence[object] | None = None,
) -> float:
    """Compute the multiclass Brier score using the sum over classes."""

    target_indices, values, weights = _metric_inputs(
        targets, probabilities, sample_weight=sample_weight, labels=labels
    )
    one_hot = np.zeros_like(values)
    one_hot[np.arange(values.shape[0]), target_indices] = 1.0
    row_scores = np.sum((values - one_hot) ** 2, axis=1)
    return _weighted_mean(row_scores, weights)


def expected_calibration_error(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
    sample_weight: ArrayLike | None = None,
    labels: Sequence[object] | None = None,
) -> float:
    """Compute top-label expected calibration error."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    target_indices, values, weights = _metric_inputs(
        targets, probabilities, sample_weight=sample_weight, labels=labels
    )
    confidence = np.max(values, axis=1)
    predicted = np.argmax(values, axis=1)
    correct = (predicted == target_indices).astype(np.float64)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    bin_indices = np.searchsorted(edges[1:-1], confidence, side="right")
    total_weight = float(weights.sum())
    error = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        bin_weight = float(bin_weights.sum())
        if bin_weight == 0.0:
            continue
        accuracy = _weighted_mean(correct[mask], bin_weights)
        mean_confidence = _weighted_mean(confidence[mask], bin_weights)
        error += bin_weight / total_weight * abs(accuracy - mean_confidence)
    return float(error)


def evaluate_probabilities(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
    calibration_strategy: Literal["uniform", "quantile"] = "uniform",
    sample_weight: ArrayLike | None = None,
    labels: Sequence[object] | None = None,
) -> ProbabilityMetrics:
    """Evaluate the three project-level probability metrics together."""

    return ProbabilityMetrics(
        log_loss=multiclass_log_loss(
            targets,
            probabilities,
            sample_weight=sample_weight,
            labels=labels,
        ),
        brier_score=multiclass_brier_score(
            targets,
            probabilities,
            sample_weight=sample_weight,
            labels=labels,
        ),
        expected_calibration_error=expected_calibration_error(
            targets,
            probabilities,
            n_bins=n_bins,
            strategy=calibration_strategy,
            sample_weight=sample_weight,
            labels=labels,
        ),
    )


def _metric_inputs(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    sample_weight: ArrayLike | None,
    labels: Sequence[object] | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (rows, at least two classes)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be finite and between zero and one")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("each probability row must sum to one")
    values = values / row_sums[:, None]

    # Explicit labels may mix Python types; do not coerce [1, "home"] into
    # strings before looking up the caller's original labels.
    target_values = np.asarray(targets, dtype=object if labels is not None else None)
    if target_values.ndim != 1 or target_values.shape[0] != values.shape[0]:
        raise ValueError("targets must contain one class label per probability row")
    target_indices = _target_indices(target_values, values.shape[1], labels)
    weights = _probability_weights(sample_weight, values.shape[0])
    return target_indices, values, weights


def _target_indices(
    targets: NDArray[np.generic],
    class_count: int,
    labels: Sequence[object] | None,
) -> NDArray[np.int64]:
    if labels is None:
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("non-integer targets require an explicit labels sequence")
        indices = np.asarray(targets, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= class_count):
            raise ValueError("integer targets must index probability columns")
        return indices
    label_values = tuple(labels)
    if len(label_values) != class_count or len(set(label_values)) != class_count:
        raise ValueError("labels must uniquely name every probability column")
    lookup = {label: index for index, label in enumerate(label_values)}
    mapped_indices: list[int] = []
    for target in targets:
        try:
            mapped_indices.append(lookup[target])
        except KeyError as exc:
            raise ValueError(f"unknown target label: {target!r}") from exc
    return np.asarray(mapped_indices, dtype=np.int64)


def _probability_weights(
    sample_weight: ArrayLike | None, row_count: int
) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.full(row_count, 1.0 / row_count, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != row_count:
        raise ValueError("sample_weight must contain one value per row")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    maximum = float(weights.max())
    if maximum <= 0.0:
        raise ValueError("sample_weight must contain positive total weight")
    # Both the raw sum and loss * weight can overflow for valid finite weights.
    # Scaling first preserves their ratios, including subnormal weights, and
    # keeps all downstream weighted metrics and ECE bins bounded.
    scaled = weights / maximum
    return np.asarray(scaled / scaled.sum(), dtype=np.float64)


def _weighted_mean(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _to_datetime64(value: TimestampLike) -> np.datetime64:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime timestamps must include timezone information")
        normalised = value.astimezone(timezone.utc).replace(tzinfo=None)
        result = np.datetime64(normalised.isoformat(timespec="microseconds"), "ns")
    elif isinstance(value, date):
        result = np.datetime64(value.isoformat(), "ns")
    elif isinstance(value, np.datetime64):
        result = value.astype("datetime64[ns]")
    else:
        raise TypeError(f"unsupported timestamp type: {type(value).__name__}")
    if np.isnat(result):
        raise ValueError("timestamps cannot contain NaT")
    return result


__all__ = [
    "ProbabilityMetrics",
    "TimestampLike",
    "WalkForwardFold",
    "evaluate_probabilities",
    "expanding_walk_forward_split",
    "expected_calibration_error",
    "multiclass_brier_score",
    "multiclass_log_loss",
]
