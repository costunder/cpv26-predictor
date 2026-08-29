"""Domain primitives shared by the data and modelling layers.

The project treats timestamps as facts, not incidental metadata.  Public APIs in
this module therefore reject naive datetimes and normalise every accepted value
to UTC.  That keeps a snapshot produced on a Linux training host identical to a
snapshot assembled on a developer machine in another timezone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

UTC = timezone.utc


def utc_datetime(value: datetime, *, field_name: str = "datetime") -> datetime:
    """Return ``value`` in UTC, rejecting ambiguous naive datetimes."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value.astimezone(UTC)


def utc_isoformat(value: datetime) -> str:
    """Serialise a timezone-aware timestamp in a stable ISO-8601 form."""

    return utc_datetime(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


class PredictionRunStatus(str, Enum):
    CREATED = "created"
    SNAPSHOTTED = "snapshotted"
    SCORED = "scored"
    FAILED = "failed"


class InformationHorizon(str, Enum):
    """Information state at which a prediction is locked."""

    EARLY = "early"
    STARTER_KNOWN = "starter_known"
    LINEUP_KNOWN = "lineup_known"
    NEAR_LOCK = "near_lock"


class PredictionKind(str, Enum):
    PLATE_APPEARANCE = "plate_appearance"
    PLAYER_HITS = "player_hits"
    TEAM_RUNS = "team_runs"
    GAME_OUTCOME = "game_outcome"


@dataclass(frozen=True, slots=True)
class TemporalBounds:
    """The four temporal axes carried by mutable source records.

    ``event_at`` describes when the represented event happened (and may be in
    the future for a scheduled game).  ``available_at`` is when a forecaster
    could first know the row; ``ingested_at`` is when this repository observed
    it.  The half-open validity interval is ``[valid_from, valid_to)``.
    """

    event_at: datetime
    available_at: datetime
    ingested_at: datetime
    valid_from: datetime
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("event_at", "available_at", "ingested_at", "valid_from"):
            object.__setattr__(self, name, utc_datetime(getattr(self, name), field_name=name))
        if self.valid_to is not None:
            object.__setattr__(
                self,
                "valid_to",
                utc_datetime(self.valid_to, field_name="valid_to"),
            )
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be later than valid_from")

    def is_known_at(self, cutoff_at: datetime, knowledge_at: datetime) -> bool:
        cutoff = utc_datetime(cutoff_at, field_name="cutoff_at")
        knowledge = utc_datetime(knowledge_at, field_name="knowledge_at")
        return self.available_at <= cutoff and self.ingested_at <= knowledge

    def is_valid_at(self, instant: datetime) -> bool:
        point = utc_datetime(instant, field_name="instant")
        return self.valid_from <= point and (self.valid_to is None or point < self.valid_to)


@dataclass(frozen=True, slots=True)
class PredictionRun:
    prediction_run_id: str
    target_game_id: str
    cutoff_at: datetime
    knowledge_at: datetime
    created_at: datetime
    horizon_type: InformationHorizon
    feature_version: str
    model_name: str
    model_version: str
    simulator_version: str
    v26_rule_version: str
    feature_fingerprint: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text = {
            "prediction_run_id": self.prediction_run_id,
            "target_game_id": self.target_game_id,
            "feature_version": self.feature_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "simulator_version": self.simulator_version,
            "v26_rule_version": self.v26_rule_version,
        }
        for name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        object.__setattr__(self, "cutoff_at", utc_datetime(self.cutoff_at, field_name="cutoff_at"))
        object.__setattr__(
            self,
            "knowledge_at",
            utc_datetime(self.knowledge_at, field_name="knowledge_at"),
        )
        object.__setattr__(
            self, "created_at", utc_datetime(self.created_at, field_name="created_at")
        )
        if self.knowledge_at < self.cutoff_at:
            raise ValueError("knowledge_at cannot be earlier than cutoff_at")
        if self.feature_fingerprint is not None:
            if len(self.feature_fingerprint) != 64:
                raise ValueError("feature_fingerprint must be a SHA-256 hex digest")
            try:
                int(self.feature_fingerprint, 16)
            except ValueError as exc:
                raise ValueError("feature_fingerprint must be hexadecimal") from exc
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True, slots=True)
class PredictionRunStatusEvent:
    """One immutable lifecycle transition for a prediction run."""

    prediction_run_status_event_id: str
    prediction_run_id: str
    status: PredictionRunStatus
    occurred_at: datetime
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("prediction_run_status_event_id", "prediction_run_id"):
            value = getattr(self, name)
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(self.status, PredictionRunStatus):
            raise TypeError("status must be a PredictionRunStatus")
        object.__setattr__(
            self,
            "occurred_at",
            utc_datetime(self.occurred_at, field_name="occurred_at"),
        )
        object.__setattr__(self, "detail", dict(self.detail))


@dataclass(frozen=True, slots=True)
class PlayerGameCandidate:
    """A possible future player/game assignment known at a run's cutoff."""

    candidate_id: str
    prediction_run_id: str
    game_id: str
    player_id: str
    team_id: str
    opponent_team_id: str
    role: str
    lineup_slot: int | None
    start_probability: float
    expected_plate_appearances: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_probability <= 1.0:
            raise ValueError("start_probability must be between 0 and 1")
        if self.expected_plate_appearances < 0.0:
            raise ValueError("expected_plate_appearances cannot be negative")
        if self.lineup_slot is not None and not 1 <= self.lineup_slot <= 9:
            raise ValueError("lineup_slot must be between 1 and 9")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    prediction_id: str
    prediction_run_id: str
    prediction_kind: PredictionKind
    entity_id: str
    value: float
    label: str | None = None
    distribution: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction_id.strip() or not self.entity_id.strip():
            raise ValueError("prediction_id and entity_id cannot be empty")
        probabilities = dict(self.distribution)
        if any(probability < 0.0 for probability in probabilities.values()):
            raise ValueError("distribution probabilities cannot be negative")
        if probabilities and abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("distribution probabilities must sum to one")
        object.__setattr__(self, "distribution", probabilities)
