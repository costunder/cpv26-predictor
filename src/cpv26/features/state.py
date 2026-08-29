"""Serialisation of computed feature vectors into immutable state rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cpv26.domain import utc_datetime, utc_isoformat

from .batting import BatterFeatureVector


@dataclass(frozen=True, slots=True)
class TeamFeatureVector:
    team_id: str
    cutoff_at: str
    numerators: Mapping[str, float] = field(default_factory=dict)
    denominators: Mapping[str, float] = field(default_factory=dict)
    features: Mapping[str, float | int | str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerators", dict(self.numerators))
        object.__setattr__(self, "denominators", dict(self.denominators))
        object.__setattr__(self, "features", dict(self.features))


def player_state_rows(
    vectors: Mapping[str, BatterFeatureVector],
    *,
    prediction_run_id: str,
    source_fingerprint: str,
    ingested_at: datetime,
    role: str = "batting",
) -> list[dict[str, Any]]:
    """Convert batting vectors into rows accepted by ``DuckDBStore.append``."""

    _validate_fingerprint(source_fingerprint)
    ingested = utc_datetime(ingested_at, field_name="ingested_at")
    rows: list[dict[str, Any]] = []
    for player_id, vector in sorted(vectors.items()):
        if player_id != vector.player_id:
            raise ValueError("feature mapping key does not match vector.player_id")
        cutoff = _parse_utc(vector.cutoff_at)
        row_id = _state_id(
            "player",
            prediction_run_id,
            player_id,
            role,
            vector.cutoff_at,
            source_fingerprint,
        )
        rows.append(
            {
                "player_state_snapshot_id": row_id,
                "prediction_run_id": prediction_run_id,
                "player_id": player_id,
                "role": role,
                "cutoff_at": cutoff,
                "numerator_json": vector.numerators,
                "denominator_json": vector.denominators,
                "feature_json": vector.features,
                "source_fingerprint": source_fingerprint,
                "event_at": cutoff,
                "available_at": cutoff,
                "ingested_at": ingested,
                "valid_from": cutoff,
                "valid_to": None,
            }
        )
    return rows


def team_state_rows(
    vectors: Mapping[str, TeamFeatureVector],
    *,
    prediction_run_id: str,
    source_fingerprint: str,
    ingested_at: datetime,
) -> list[dict[str, Any]]:
    """Convert team feature vectors into immutable point-in-time state rows."""

    _validate_fingerprint(source_fingerprint)
    ingested = utc_datetime(ingested_at, field_name="ingested_at")
    rows: list[dict[str, Any]] = []
    for team_id, vector in sorted(vectors.items()):
        if team_id != vector.team_id:
            raise ValueError("feature mapping key does not match vector.team_id")
        cutoff = _parse_utc(vector.cutoff_at)
        row_id = _state_id(
            "team",
            prediction_run_id,
            team_id,
            vector.cutoff_at,
            source_fingerprint,
        )
        rows.append(
            {
                "team_state_snapshot_id": row_id,
                "prediction_run_id": prediction_run_id,
                "team_id": team_id,
                "cutoff_at": cutoff,
                "numerator_json": vector.numerators,
                "denominator_json": vector.denominators,
                "feature_json": vector.features,
                "source_fingerprint": source_fingerprint,
                "event_at": cutoff,
                "available_at": cutoff,
                "ingested_at": ingested,
                "valid_from": cutoff,
                "valid_to": None,
            }
        )
    return rows


def _state_id(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_fingerprint(value: str) -> None:
    if len(value) != 64:
        raise ValueError("source_fingerprint must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("source_fingerprint must be hexadecimal") from exc


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    normalised = utc_datetime(parsed, field_name="cutoff_at")
    if utc_isoformat(normalised) != value:
        raise ValueError("cutoff_at must use canonical UTC ISO-8601 format")
    return normalised
