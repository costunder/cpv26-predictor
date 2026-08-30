"""Strong tabular baseline with an optional, lazily imported CatBoost runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_CATBOOST_PARAMETERS: Mapping[str, object] = MappingProxyType(
    {
        "iterations": 800,
        "depth": 7,
        "learning_rate": 0.05,
        "random_seed": 2026,
        "verbose": False,
        "allow_writing_files": False,
    }
)


class CatBoostClassifierBaseline:
    """A small sklearn-style wrapper around ``catboost.CatBoostClassifier``.

    CatBoost is imported only by :meth:`fit`, so data preparation and metric
    jobs do not require the optional ML dependency group. Binary and multiclass
    losses are selected from the observed labels unless explicitly configured.
    """

    def __init__(self, parameters: Mapping[str, object] | None = None) -> None:
        configured = dict(DEFAULT_CATBOOST_PARAMETERS)
        if parameters is not None:
            configured.update(parameters)
        self.parameters = MappingProxyType(configured)
        self._model: Any | None = None
        self._classes: NDArray[Any] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def classes_(self) -> NDArray[Any]:
        if self._classes is None:
            raise RuntimeError("CatBoost baseline has not been fitted")
        return self._classes.copy()

    @property
    def model_(self) -> Any:
        if self._model is None:
            raise RuntimeError("CatBoost baseline has not been fitted")
        return self._model

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        model = self.model_
        values = getattr(model, "feature_importances_", None)
        if values is None:
            raise RuntimeError("fitted CatBoost model has no feature importances")
        return np.asarray(values, dtype=np.float64).copy()

    def fit(
        self,
        features: Any,
        targets: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        categorical_features: Sequence[int | str] | None = None,
        eval_set: Any | None = None,
    ) -> CatBoostClassifierBaseline:
        """Fit the baseline and return ``self``.

        ``features`` intentionally remains dataframe-compatible so CatBoost can
        consume named categorical columns without adding pandas as a project
        dependency.
        """

        target_array = _validate_targets(targets)
        row_count = _row_count(features)
        if row_count != target_array.shape[0]:
            raise ValueError("features and targets must contain the same row count")
        weights = _validate_weights(sample_weight, row_count)
        classifier_type = _load_catboost_classifier()
        parameters = dict(self.parameters)
        if "loss_function" not in parameters:
            parameters["loss_function"] = (
                "Logloss" if np.unique(target_array).size == 2 else "MultiClass"
            )
        model = classifier_type(**parameters)
        fit_arguments: dict[str, Any] = {}
        if weights is not None:
            fit_arguments["sample_weight"] = weights
        if categorical_features is not None:
            fit_arguments["cat_features"] = tuple(categorical_features)
        if eval_set is not None:
            fit_arguments["eval_set"] = eval_set
        model.fit(features, target_array, **fit_arguments)
        classes = np.asarray(getattr(model, "classes_", np.unique(target_array)))
        if classes.ndim != 1 or classes.size < 2:
            raise RuntimeError("fitted CatBoost model returned invalid classes")
        self._model = model
        self._classes = classes.copy()
        return self

    def predict_proba(self, features: Any) -> NDArray[np.float64]:
        """Return a validated ``(rows, classes)`` probability matrix."""

        model = self.model_
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        expected_rows = _row_count(features)
        if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
            raise RuntimeError("CatBoost returned an invalid probability matrix")
        if self._classes is None or probabilities.shape[1] != self._classes.size:
            raise RuntimeError("CatBoost probability columns do not match fitted classes")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise RuntimeError("CatBoost returned invalid probabilities")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-9):
            raise RuntimeError("CatBoost probabilities do not sum to one")
        return probabilities


def _load_catboost_classifier() -> Any:
    try:
        module = import_module("catboost")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "CatBoost is required to fit the tabular baseline; "
            "install it with `bash scripts/setup.sh tabular` or "
            "`pip install -e '.[tabular]'`."
        ) from exc
    classifier_type = getattr(module, "CatBoostClassifier", None)
    if classifier_type is None:
        raise RuntimeError("installed catboost package does not expose CatBoostClassifier")
    return classifier_type


def _validate_targets(targets: ArrayLike) -> NDArray[Any]:
    values = np.asarray(targets)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("targets must be a one-dimensional array with at least two rows")
    if np.unique(values).size < 2:
        raise ValueError("classification targets must contain at least two classes")
    if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
        raise ValueError("numeric targets must be finite")
    return values


def _validate_weights(
    sample_weight: ArrayLike | None, row_count: int
) -> NDArray[np.float64] | None:
    if sample_weight is None:
        return None
    values = np.asarray(sample_weight, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != row_count:
        raise ValueError("sample_weight must contain one value per row")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    if values.sum() <= 0.0:
        raise ValueError("sample_weight must contain positive total weight")
    return values


def _row_count(features: Any) -> int:
    shape = getattr(features, "shape", None)
    if shape is None or len(shape) < 1:
        try:
            return len(features)
        except TypeError as exc:
            raise TypeError("features must be a sized tabular object") from exc
    rows = int(shape[0])
    if rows < 1:
        raise ValueError("features cannot be empty")
    return rows


__all__ = ["DEFAULT_CATBOOST_PARAMETERS", "CatBoostClassifierBaseline"]
