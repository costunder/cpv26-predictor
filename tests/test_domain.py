from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from cpv26.domain import (
    UTC,
    InformationHorizon,
    PredictionRun,
    PredictionRunStatus,
    PredictionRunStatusEvent,
)


def _prediction_run() -> PredictionRun:
    korea_time = timezone(timedelta(hours=9))
    cutoff = datetime(2026, 8, 29, 17, 30, tzinfo=korea_time)
    return PredictionRun(
        prediction_run_id="20260829-game-1-lineup",
        target_game_id="20260829-game-1",
        cutoff_at=cutoff,
        knowledge_at=cutoff + timedelta(minutes=2),
        created_at=cutoff + timedelta(minutes=2),
        horizon_type=InformationHorizon.LINEUP_KNOWN,
        feature_version="feature-v1",
        model_name="relgnn-catboost-ensemble",
        model_version="model-v1",
        simulator_version="simulator-v1",
        v26_rule_version="v26-2026-08",
    )


def test_prediction_run_locks_versions_and_normalises_time() -> None:
    run = _prediction_run()

    assert run.cutoff_at.tzinfo is UTC
    assert run.cutoff_at.hour == 8
    assert run.horizon_type is InformationHorizon.LINEUP_KNOWN
    assert run.v26_rule_version == "v26-2026-08"


def test_prediction_run_rejects_future_knowledge_reversal() -> None:
    run = _prediction_run()

    with pytest.raises(ValueError, match="knowledge_at cannot be earlier"):
        replace(run, knowledge_at=run.cutoff_at - timedelta(microseconds=1))


def test_prediction_run_status_is_a_separate_immutable_event() -> None:
    run = _prediction_run()
    event = PredictionRunStatusEvent(
        prediction_run_status_event_id="status-1",
        prediction_run_id=run.prediction_run_id,
        status=PredictionRunStatus.CREATED,
        occurred_at=run.created_at,
        detail={"worker": "test"},
    )

    assert not hasattr(run, "status")
    assert event.occurred_at.tzinfo is UTC
    assert event.status is PredictionRunStatus.CREATED
