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


@pytest.mark.parametrize(
    ("targets", "labels"),
    [
        (np.array(["away", "home"], dtype=object), ("away", "home")),
        (np.array([1, 2], dtype=object), (1, 2)),
        (np.array([1, "home"], dtype=object), (1, "home")),
        ([1, "home"], (1, "home")),
        (np.array([np.int64(1), np.str_("home")], dtype=object), (1, "home")),
        (np.array([None, "home"], dtype=object), (None, "home")),
        (np.array([b"away", b"home"]), (b"away", b"home")),
    ],
)
def test_probability_metrics_preserve_object_and_mixed_labels(
    targets: object, labels: tuple[object, object]
) -> None:
    probabilities = [[0.75, 0.25], [0.2, 0.8]]
    expected = evaluate_probabilities([0, 1], probabilities, n_bins=2)

    actual = evaluate_probabilities(targets, probabilities, labels=labels, n_bins=2)

    assert actual.log_loss == pytest.approx(expected.log_loss)
    assert actual.brier_score == pytest.approx(expected.brier_score)
    assert actual.expected_calibration_error == pytest.approx(expected.expected_calibration_error)


def test_probability_metrics_unknown_object_label_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown target label"):
        evaluate_probabilities(
            np.array(["unknown"], dtype=object), [[0.75, 0.25]], labels=("away", "home")
        )


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
@pytest.mark.parametrize("scale", [1e308, 1e-300, np.nextafter(0.0, 1.0) * 8])
def test_probability_metrics_are_invariant_to_extreme_finite_weight_scale(
    strategy: str, scale: float
) -> None:
    probabilities = np.array([[0.5, 0.5], [0.1, 0.9], [1.0, 0.0], [0.0, 1.0]])
    targets = np.array([0, 1, 1, 0])
    relative_weights = np.array([1.0, 0.5, 0.0, 1.0])
    weights = relative_weights * scale
    expected_log_loss = float(np.average(-np.log([0.5, 0.9, 1e-15, 1e-15]),
                                         weights=relative_weights))
    expected_brier = float(np.average([0.5, 0.02, 2.0, 2.0], weights=relative_weights))
    # One bin checks the weighted global mean for either strategy.
    expected_ece = abs((1.0 + 0.5) / 2.5 - (0.5 + 0.45 + 1.0) / 2.5)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = evaluate_probabilities(
            targets, probabilities, sample_weight=weights, n_bins=1,
            calibration_strategy=strategy,
        )

    assert actual.log_loss == pytest.approx(expected_log_loss)
    assert actual.brier_score == pytest.approx(expected_brier)
    assert actual.expected_calibration_error == pytest.approx(expected_ece)
    # Distinct bins must remain scale-invariant too, not only a global mean.
    assert expected_calibration_error(
        targets, probabilities, sample_weight=weights, n_bins=4, strategy=strategy,
    ) == pytest.approx(expected_calibration_error(
        targets, probabilities, sample_weight=relative_weights, n_bins=4, strategy=strategy,
    ))


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 1.0], [np.inf, 1.0], [np.nan, 1.0]])
def test_probability_metrics_reject_invalid_sample_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        evaluate_probabilities([0, 1], [[0.8, 0.2], [0.1, 0.9]], sample_weight=weights)


def test_catboost_baseline_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_catboost(name: str) -> ModuleType:
        assert name == "catboost"
        raise ModuleNotFoundError("catboost is not installed", name=name)

    monkeypatch.setattr(baseline_module, "import_module", missing_catboost)
    baseline = CatBoostClassifierBaseline(parameters={"iterations": 2})

    with pytest.raises(RuntimeError, match=r"conda activate cpv26.*python -m pip install"):
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
