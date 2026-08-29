from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import ModuleType

import numpy as np
import pytest

import cpv26.models.baseline as baseline_module
from cpv26.evaluation import (
    evaluate_probabilities,
    expanding_walk_forward_split,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)
from cpv26.models.baseline import CatBoostClassifierBaseline

UTC = timezone.utc


def test_expanding_walk_forward_keeps_periods_together_and_expands() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        start + timedelta(days=2),
        start,
        start,
        start + timedelta(days=1),
        start + timedelta(days=3),
        start + timedelta(days=4),
    ]

    folds = expanding_walk_forward_split(
        timestamps,
        min_train_periods=2,
        validation_periods=1,
        max_folds=2,
    )

    assert len(folds) == 2
    assert folds[0].train_indices.tolist() == [1, 2, 3]
    assert folds[0].validation_indices.tolist() == [0]
    assert folds[1].train_indices.tolist() == [1, 2, 3, 0]
    assert folds[1].validation_indices.tolist() == [4]
    assert folds[0].train_end < folds[0].validation_start
    with pytest.raises(ValueError, match="read-only"):
        folds[0].train_indices[0] = 99


def test_expanding_walk_forward_honours_complete_period_gap() -> None:
    start = datetime(2026, 2, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=offset) for offset in range(6)]

    fold = expanding_walk_forward_split(
        timestamps,
        min_train_periods=2,
        validation_periods=1,
        gap_periods=1,
        max_folds=1,
    )[0]

    assert fold.train_indices.tolist() == [0, 1]
    assert fold.validation_indices.tolist() == [3]
    assert 2 not in fold.train_indices
    assert 2 not in fold.validation_indices


def test_probability_metrics_match_direct_multiclass_calculation() -> None:
    targets = np.array([0, 1, 1], dtype=np.int64)
    probabilities = np.array(
        [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]], dtype=np.float64
    )
    expected_log_loss = float(-np.mean(np.log([0.8, 0.9, 0.6])))
    one_hot = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    expected_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    expected_ece = 1.0 - float(np.mean([0.8, 0.9, 0.6]))

    metrics = evaluate_probabilities(targets, probabilities, n_bins=2)

    assert multiclass_log_loss(targets, probabilities) == pytest.approx(expected_log_loss)
    assert multiclass_brier_score(targets, probabilities) == pytest.approx(expected_brier)
    assert expected_calibration_error(
        targets, probabilities, n_bins=2
    ) == pytest.approx(expected_ece)
    assert metrics.log_loss == pytest.approx(expected_log_loss)
    assert metrics.brier_score == pytest.approx(expected_brier)
    assert metrics.expected_calibration_error == pytest.approx(expected_ece)


def test_probability_metrics_support_explicit_string_labels() -> None:
    probabilities = np.array([[0.75, 0.25], [0.2, 0.8]], dtype=np.float64)

    score = multiclass_log_loss(
        np.array(["away", "home"]),
        probabilities,
        labels=("away", "home"),
    )

    assert score == pytest.approx(float(-np.mean(np.log([0.75, 0.8]))))
    with pytest.raises(ValueError, match="sum to one"):
        multiclass_log_loss([0], [[0.8, 0.3]])


def test_catboost_baseline_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_catboost(name: str) -> ModuleType:
        assert name == "catboost"
        raise ModuleNotFoundError("catboost is not installed", name=name)

    monkeypatch.setattr(baseline_module, "import_module", missing_catboost)
    baseline = CatBoostClassifierBaseline(parameters={"iterations": 2})

    with pytest.raises(RuntimeError, match=r"pip install -e '.\[tabular\]'"):
        baseline.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))


def test_catboost_baseline_actual_fit_predict_smoke() -> None:
    pytest.importorskip("catboost")
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    targets = np.array([0, 0, 0, 1, 1, 1])
    baseline = CatBoostClassifierBaseline(
        parameters={"iterations": 5, "depth": 2, "thread_count": 1}
    ).fit(features, targets)

    probabilities = baseline.predict_proba(features)

    assert baseline.is_fitted
    assert probabilities.shape == (6, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
