"""Out-of-fold probability stacking and prediction-time calibration."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
ObjectArray: TypeAlias = NDArray[np.object_]


def _as_probabilities(
    values: ArrayLike,
    *,
    name: str,
    rows: int | None = None,
) -> FloatArray:
    probabilities: FloatArray = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must have shape [samples, classes>=2]")
    if rows is not None and probabilities.shape[0] != rows:
        raise ValueError(f"{name} has {probabilities.shape[0]} rows; expected {rows}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError(f"{name} contains invalid probabilities")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} contains a row with zero probability mass")
    probabilities = probabilities / row_sums
    return np.asarray(np.clip(probabilities, 1e-12, 1.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OOFPredictionSet:
    """Base-model probabilities generated strictly out of time-ordered folds."""

    probabilities: Mapping[str, FloatArray]
    targets: Sequence[int]
    fold_ids: Sequence[Hashable]
    information_stages: Sequence[str] | None = None
    fold_order: Sequence[Hashable] | None = None

    def __post_init__(self) -> None:
        targets: IntArray = np.asarray(self.targets, dtype=np.int64)
        fold_ids = cast(ObjectArray, np.asarray(self.fold_ids, dtype=object))
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError("targets must be a non-empty one-dimensional sequence")
        if fold_ids.shape != targets.shape:
            raise ValueError("fold_ids must have one value per target")
        if len(set(fold_ids.tolist())) < 2:
            raise ValueError("OOF predictions require at least two distinct folds")
        if not self.probabilities:
            raise ValueError("at least one base model is required")

        normalized: dict[str, FloatArray] = {}
        class_count: int | None = None
        for model_name, values in self.probabilities.items():
            if not model_name or model_name.strip() != model_name:
                raise ValueError("base model names must be non-empty and trimmed")
            array = _as_probabilities(values, name=model_name, rows=targets.size)
            if class_count is None:
                class_count = int(array.shape[1])
            elif array.shape[1] != class_count:
                raise ValueError("all base models must predict the same classes")
            array.setflags(write=False)
            normalized[model_name] = array
        assert class_count is not None
        if np.any(targets < 0) or np.any(targets >= class_count):
            raise ValueError("targets contain an out-of-range class index")

        if self.information_stages is None:
            stages = cast(ObjectArray, np.full(targets.shape, "all", dtype=object))
        else:
            stages = cast(
                ObjectArray,
                np.asarray(self.information_stages, dtype=object),
            )
            if stages.shape != targets.shape:
                raise ValueError("information_stages must have one value per target")
            if any(not str(stage) for stage in stages):
                raise ValueError("information stage names cannot be empty")
            stages = cast(
                ObjectArray,
                np.asarray([str(stage) for stage in stages], dtype=object),
            )

        observed_folds = tuple(dict.fromkeys(fold_ids.tolist()))
        if self.fold_order is None:
            ordered_folds = observed_folds
        else:
            ordered_folds = tuple(self.fold_order)
            if len(set(ordered_folds)) != len(ordered_folds):
                raise ValueError("fold_order cannot contain duplicates")
            if set(ordered_folds) != set(observed_folds):
                raise ValueError("fold_order must contain every observed fold exactly once")

        targets.setflags(write=False)
        fold_ids.setflags(write=False)
        stages.setflags(write=False)
        object.__setattr__(self, "probabilities", MappingProxyType(normalized))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "fold_ids", fold_ids)
        object.__setattr__(self, "information_stages", stages)
        object.__setattr__(self, "fold_order", ordered_folds)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.probabilities)

    @property
    def num_classes(self) -> int:
        return int(next(iter(self.probabilities.values())).shape[1])

    @property
    def num_samples(self) -> int:
        return len(self.targets)


class OOFProbabilityStacker:
    """Multinomial log-probability stacker fitted from OOF base predictions."""

    def __init__(
        self,
        *,
        l2: float = 1e-3,
        learning_rate: float = 0.03,
        max_iter: int = 2000,
        tolerance: float = 1e-7,
    ) -> None:
        if l2 < 0.0 or learning_rate <= 0.0 or max_iter <= 0 or tolerance <= 0.0:
            raise ValueError("invalid stacker optimization settings")
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.model_names_: tuple[str, ...] | None = None
        self.num_classes_: int | None = None
        self.feature_mean_: FloatArray | None = None
        self.feature_scale_: FloatArray | None = None
        self.coef_: FloatArray | None = None
        self.intercept_: FloatArray | None = None
        self.n_iter_: int = 0

    @staticmethod
    def _softmax(logits: FloatArray) -> FloatArray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return np.asarray(
            exp_values / exp_values.sum(axis=1, keepdims=True),
            dtype=np.float64,
        )

    @staticmethod
    def _features(
        probabilities: Mapping[str, FloatArray],
        model_names: Sequence[str],
        *,
        expected_classes: int | None = None,
    ) -> tuple[FloatArray, int]:
        if set(probabilities) != set(model_names):
            missing = set(model_names).difference(probabilities)
            unexpected = set(probabilities).difference(model_names)
            raise ValueError(
                f"base models differ from fitted stacker; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        rows: int | None = None
        class_count = expected_classes
        blocks: list[FloatArray] = []
        for name in model_names:
            values = _as_probabilities(probabilities[name], name=name, rows=rows)
            rows = values.shape[0]
            if class_count is None:
                class_count = int(values.shape[1])
            elif values.shape[1] != class_count:
                raise ValueError("base-model class dimensions do not match")
            blocks.append(np.log(values))
        assert class_count is not None
        return np.asarray(np.concatenate(blocks, axis=1), dtype=np.float64), class_count

    def _fit_arrays(
        self,
        probabilities: Mapping[str, FloatArray],
        targets: IntArray,
        sample_weight: FloatArray | None = None,
    ) -> OOFProbabilityStacker:
        names = tuple(probabilities)
        features, class_count = self._features(probabilities, names)
        targets = np.asarray(targets, dtype=np.int64)
        if targets.shape != (features.shape[0],):
            raise ValueError("targets must have one class index per prediction row")
        if np.any(targets < 0) or np.any(targets >= class_count):
            raise ValueError("targets contain an out-of-range class index")
        if sample_weight is None:
            weights: FloatArray = np.ones(features.shape[0], dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != targets.shape or np.any(weights < 0.0):
                raise ValueError("sample_weight must be non-negative with one value per row")
            if weights.sum() <= 0.0:
                raise ValueError("sample_weight must have positive total mass")
        weights = weights / weights.sum()

        mean: FloatArray = features.mean(axis=0)
        scale: FloatArray = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (features - mean) / scale
        one_hot: FloatArray = np.eye(class_count, dtype=np.float64)[targets]
        coefficients: FloatArray = np.zeros(
            (standardized.shape[1], class_count),
            dtype=np.float64,
        )
        priors: FloatArray = (
            np.bincount(targets, minlength=class_count).astype(np.float64) + 0.5
        )
        intercept: FloatArray = np.log(priors / priors.sum())

        first_moment_w = np.zeros_like(coefficients)
        second_moment_w = np.zeros_like(coefficients)
        first_moment_b = np.zeros_like(intercept)
        second_moment_b = np.zeros_like(intercept)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        self.n_iter_ = self.max_iter
        for step in range(1, self.max_iter + 1):
            predicted = self._softmax(standardized @ coefficients + intercept)
            error = (predicted - one_hot) * weights[:, None]
            gradient_w = standardized.T @ error + self.l2 * coefficients
            gradient_b = error.sum(axis=0)
            gradient_norm = max(
                float(np.max(np.abs(gradient_w))),
                float(np.max(np.abs(gradient_b))),
            )
            if gradient_norm < self.tolerance:
                self.n_iter_ = step
                break

            first_moment_w = beta1 * first_moment_w + (1.0 - beta1) * gradient_w
            second_moment_w = beta2 * second_moment_w + (1.0 - beta2) * gradient_w**2
            first_moment_b = beta1 * first_moment_b + (1.0 - beta1) * gradient_b
            second_moment_b = beta2 * second_moment_b + (1.0 - beta2) * gradient_b**2
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            coefficients -= (
                self.learning_rate
                * (first_moment_w / correction1)
                / (np.sqrt(second_moment_w / correction2) + epsilon)
            )
            intercept -= (
                self.learning_rate
                * (first_moment_b / correction1)
                / (np.sqrt(second_moment_b / correction2) + epsilon)
            )

        self.model_names_ = names
        self.num_classes_ = class_count
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.coef_ = coefficients
        self.intercept_ = intercept
        return self

    def fit(
        self,
        data: OOFPredictionSet,
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> OOFProbabilityStacker:
        if not isinstance(data, OOFPredictionSet):
            raise TypeError("fit expects an OOFPredictionSet with explicit fold IDs")
        weights: FloatArray | None = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        targets: IntArray = np.asarray(data.targets, dtype=np.int64)
        return self._fit_arrays(data.probabilities, targets, weights)

    def predict_proba(self, probabilities: Mapping[str, FloatArray]) -> FloatArray:
        if (
            self.model_names_ is None
            or self.num_classes_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
            or self.coef_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("stacker has not been fitted")
        features, _ = self._features(
            probabilities,
            self.model_names_,
            expected_classes=self.num_classes_,
        )
        standardized = (features - self.feature_mean_) / self.feature_scale_
        return self._softmax(standardized @ self.coef_ + self.intercept_)


class TemperatureCalibrator:
    """Single-temperature multiclass calibrator optimized for log loss."""

    def __init__(self, *, minimum: float = 0.05, maximum: float = 20.0) -> None:
        if minimum <= 0.0 or maximum <= minimum:
            raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.temperature_: float | None = None
        self.num_classes_: int | None = None

    @staticmethod
    def _apply(probabilities: FloatArray, temperature: float) -> FloatArray:
        log_values = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        log_values -= log_values.max(axis=1, keepdims=True)
        values = np.exp(log_values)
        return np.asarray(
            values / values.sum(axis=1, keepdims=True),
            dtype=np.float64,
        )

    def fit(
        self,
        probabilities: ArrayLike,
        targets: ArrayLike,
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> TemperatureCalibrator:
        values = _as_probabilities(probabilities, name="probabilities")
        target_array: IntArray = np.asarray(targets, dtype=np.int64)
        if target_array.shape != (values.shape[0],):
            raise ValueError("targets must have one value per probability row")
        if np.any(target_array < 0) or np.any(target_array >= values.shape[1]):
            raise ValueError("target class index is out of range")
        weights: FloatArray
        if sample_weight is None:
            weights = np.ones(values.shape[0], dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != target_array.shape or np.any(weights < 0.0):
                raise ValueError("invalid sample weights")
        if weights.sum() <= 0.0:
            raise ValueError("sample weights must have positive total mass")
        weights /= weights.sum()

        def objective(log_temperature: float) -> float:
            calibrated = self._apply(values, float(np.exp(log_temperature)))
            losses = -np.log(calibrated[np.arange(values.shape[0]), target_array])
            return float(np.dot(weights, losses))

        left = float(np.log(self.minimum))
        right = float(np.log(self.maximum))
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(96):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = objective(x2)
        self.temperature_ = float(np.exp((left + right) / 2.0))
        self.num_classes_ = int(values.shape[1])
        return self

    def predict_proba(self, probabilities: ArrayLike) -> FloatArray:
        if self.temperature_ is None or self.num_classes_ is None:
            raise RuntimeError("calibrator has not been fitted")
        values = _as_probabilities(probabilities, name="probabilities")
        if values.shape[1] != self.num_classes_:
            raise ValueError("probability class count differs from fitted calibrator")
        return self._apply(values, self.temperature_)


class StagewiseTemperatureCalibrator:
    """Fit calibration separately for each prediction information stage."""

    def __init__(self, *, min_samples_per_stage: int = 100) -> None:
        if min_samples_per_stage < 1:
            raise ValueError("min_samples_per_stage must be positive")
        self.min_samples_per_stage = int(min_samples_per_stage)
        self.global_: TemperatureCalibrator | None = None
        self.by_stage_: dict[str, TemperatureCalibrator] = {}

    def fit(
        self,
        probabilities: ArrayLike,
        targets: ArrayLike,
        stages: Sequence[object],
    ) -> StagewiseTemperatureCalibrator:
        values = _as_probabilities(probabilities, name="probabilities")
        target_array: IntArray = np.asarray(targets, dtype=np.int64)
        stage_values = tuple(str(stage) for stage in stages)
        stage_array = cast(
            ObjectArray,
            np.asarray(stage_values, dtype=object),
        )
        if target_array.shape != (values.shape[0],) or stage_array.shape != target_array.shape:
            raise ValueError("targets and stages must align with probability rows")
        self.global_ = TemperatureCalibrator().fit(values, target_array)
        self.by_stage_ = {}
        for stage in dict.fromkeys(stage_values):
            mask: BoolArray = np.asarray(stage_array == stage, dtype=np.bool_)
            enough_samples = int(mask.sum()) >= self.min_samples_per_stage
            has_multiple_classes = np.unique(target_array[mask]).size > 1
            if enough_samples and has_multiple_classes:
                self.by_stage_[stage] = TemperatureCalibrator().fit(
                    values[mask], target_array[mask]
                )
        return self

    def predict_proba(
        self,
        probabilities: ArrayLike,
        stages: Sequence[object],
    ) -> FloatArray:
        if self.global_ is None:
            raise RuntimeError("stagewise calibrator has not been fitted")
        values = _as_probabilities(probabilities, name="probabilities")
        stage_values = tuple(str(stage) for stage in stages)
        stage_array = cast(
            ObjectArray,
            np.asarray(stage_values, dtype=object),
        )
        if stage_array.shape != (values.shape[0],):
            raise ValueError("stages must have one value per probability row")
        result: FloatArray = np.empty_like(values)
        for stage in dict.fromkeys(stage_values):
            mask: BoolArray = np.asarray(stage_array == stage, dtype=np.bool_)
            calibrator = self.by_stage_.get(stage, self.global_)
            result[mask] = calibrator.predict_proba(values[mask])
        return result


class BinaryIsotonicCalibrator:
    """Weighted pool-adjacent-violators calibration for hit/no-hit probabilities."""

    def __init__(self) -> None:
        self.thresholds_: FloatArray | None = None
        self.values_: FloatArray | None = None

    def fit(
        self,
        probabilities: ArrayLike,
        targets: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> BinaryIsotonicCalibrator:
        scores: FloatArray = np.asarray(probabilities, dtype=np.float64)
        labels: FloatArray = np.asarray(targets, dtype=np.float64)
        if scores.ndim != 1 or labels.shape != scores.shape or scores.size == 0:
            raise ValueError("probabilities and targets must be aligned one-dimensional arrays")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("binary probabilities must be finite values in [0, 1]")
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("binary targets must be 0 or 1")
        weights: FloatArray
        if sample_weight is None:
            weights = np.ones_like(scores)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != scores.shape or np.any(weights <= 0.0):
                raise ValueError("isotonic sample weights must be positive")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        sorted_weights = weights[order]
        unique_result = np.unique(sorted_scores, return_inverse=True)
        unique_scores: FloatArray = np.asarray(unique_result[0], dtype=np.float64)
        inverse: IntArray = np.asarray(unique_result[1], dtype=np.int64)
        grouped_weights: FloatArray = np.asarray(
            np.bincount(
                inverse,
                weights=sorted_weights,
            ),
            dtype=np.float64,
        )
        grouped_positive_weights: FloatArray = np.asarray(
            np.bincount(
                inverse,
                weights=sorted_labels * sorted_weights,
            ),
            dtype=np.float64,
        )
        blocks: list[list[float]] = []
        for score, weight, positive_weight in zip(
            unique_scores,
            grouped_weights,
            grouped_positive_weights,
            strict=True,
        ):
            blocks.append(
                [float(score), float(score), float(weight), float(positive_weight)]
            )
            while len(blocks) >= 2:
                previous_mean = blocks[-2][3] / blocks[-2][2]
                current_mean = blocks[-1][3] / blocks[-1][2]
                if previous_mean <= current_mean:
                    break
                current = blocks.pop()
                previous = blocks.pop()
                blocks.append(
                    [
                        previous[0],
                        current[1],
                        previous[2] + current[2],
                        previous[3] + current[3],
                    ]
                )
        self.thresholds_ = np.asarray([block[1] for block in blocks], dtype=np.float64)
        self.values_ = np.asarray([block[3] / block[2] for block in blocks], dtype=np.float64)
        return self

    def predict(self, probabilities: ArrayLike) -> FloatArray:
        if self.thresholds_ is None or self.values_ is None:
            raise RuntimeError("isotonic calibrator has not been fitted")
        scores: FloatArray = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("binary probabilities must be finite values in [0, 1]")
        indices = np.searchsorted(self.thresholds_, scores, side="left")
        indices = np.clip(indices, 0, self.values_.size - 1)
        return np.asarray(self.values_[indices], dtype=np.float64)


class OOFStackingPipeline:
    """Expanding-window meta-model cross-fit and information-stage calibration."""

    def __init__(
        self,
        *,
        stacker: OOFProbabilityStacker | None = None,
        min_samples_per_stage: int = 100,
    ) -> None:
        self.stacker = stacker or OOFProbabilityStacker()
        self.calibrator = StagewiseTemperatureCalibrator(
            min_samples_per_stage=min_samples_per_stage
        )

    def _new_stacker(self) -> OOFProbabilityStacker:
        return OOFProbabilityStacker(
            l2=self.stacker.l2,
            learning_rate=self.stacker.learning_rate,
            max_iter=self.stacker.max_iter,
            tolerance=self.stacker.tolerance,
        )

    def fit(self, data: OOFPredictionSet) -> OOFStackingPipeline:
        ordered_folds = data.fold_order
        information_stages = data.information_stages
        if ordered_folds is None or information_stages is None:
            raise RuntimeError("OOF prediction data was not normalized")
        fold_ids = tuple(data.fold_ids)
        targets: IntArray = np.asarray(data.targets, dtype=np.int64)
        stage_array = cast(
            ObjectArray,
            np.asarray(information_stages, dtype=object),
        )
        cross_fitted: FloatArray = np.empty(
            (data.num_samples, data.num_classes),
            dtype=np.float64,
        )
        calibration_mask: BoolArray = np.zeros(data.num_samples, dtype=np.bool_)
        for fold_position, fold_id in enumerate(ordered_folds):
            if fold_position == 0:
                continue
            validation_mask: BoolArray = np.asarray(
                [candidate == fold_id for candidate in fold_ids],
                dtype=np.bool_,
            )
            previous_folds = set(ordered_folds[:fold_position])
            training_mask: BoolArray = np.asarray(
                [candidate in previous_folds for candidate in fold_ids],
                dtype=np.bool_,
            )
            fold_stacker = self._new_stacker()._fit_arrays(
                {
                    name: probabilities[training_mask]
                    for name, probabilities in data.probabilities.items()
                },
                targets[training_mask],
            )
            cross_fitted[validation_mask] = fold_stacker.predict_proba(
                {
                    name: probabilities[validation_mask]
                    for name, probabilities in data.probabilities.items()
                }
            )
            calibration_mask |= validation_mask
        self.stacker.fit(data)
        self.calibrator.fit(
            cross_fitted[calibration_mask],
            targets[calibration_mask],
            tuple(str(stage) for stage in stage_array[calibration_mask].tolist()),
        )
        return self

    def predict_proba(
        self,
        probabilities: Mapping[str, FloatArray],
        information_stages: Sequence[str],
    ) -> FloatArray:
        stacked = self.stacker.predict_proba(probabilities)
        return self.calibrator.predict_proba(stacked, information_stages)


TemporalOOFStackingPipeline = OOFStackingPipeline
