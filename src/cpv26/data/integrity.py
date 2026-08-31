"""Explicit reference audits for the append-only DuckDB schema.

Physical foreign keys are intentionally avoided because licensed feeds are
often staged out of order. These checks provide the stronger pre-snapshot and
pre-training contract: every non-null business reference must resolve to at
least one physical parent record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReferenceRule:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str

    @property
    def name(self) -> str:
        return (
            f"{self.child_table}.{self.child_column} -> "
            f"{self.parent_table}.{self.parent_column}"
        )


@dataclass(frozen=True, slots=True)
class ReferenceViolation:
    rule: ReferenceRule
    missing_value_count: int
    sample_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositeReferenceRule:
    child_table: str
    child_columns: tuple[str, ...]
    parent_table: str
    parent_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.child_columns or len(self.child_columns) != len(self.parent_columns):
            raise ValueError("composite reference columns must have equal non-zero length")

    @property
    def name(self) -> str:
        child = ",".join(self.child_columns)
        parent = ",".join(self.parent_columns)
        return f"{self.child_table}({child}) -> {self.parent_table}({parent})"


@dataclass(frozen=True, slots=True)
class CompositeReferenceViolation:
    rule: CompositeReferenceRule
    missing_value_count: int
    sample_values: tuple[str, ...]


class ReferentialIntegrityError(RuntimeError):
    """Raised when one or more declared business references do not resolve."""

    def __init__(self, violations: tuple[ReferenceViolation, ...]) -> None:
        self.violations = violations
        summary = "; ".join(
            f"{violation.rule.name}: {violation.missing_value_count} missing"
            for violation in violations
        )
        super().__init__(f"referential integrity violations: {summary}")


class CompositeReferentialIntegrityError(RuntimeError):
    """Raised when one or more multi-column references do not resolve."""

    def __init__(self, violations: tuple[CompositeReferenceViolation, ...]) -> None:
        self.violations = violations
        summary = "; ".join(
            f"{violation.rule.name}: {violation.missing_value_count} missing"
            for violation in violations
        )
        super().__init__(f"composite referential integrity violations: {summary}")


REFERENCE_RULES: tuple[ReferenceRule, ...] = (
    ReferenceRule("historical_boxscore", "game_id", "game", "game_id"),
    ReferenceRule("historical_boxscore", "team_id", "team", "team_id"),
    ReferenceRule("historical_boxscore", "opponent_team_id", "team", "team_id"),
    ReferenceRule(
        "historical_boxscore", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("historical_game_detail", "game_id", "game", "game_id"),
    ReferenceRule(
        "historical_game_detail", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("prediction_run", "target_game_id", "game", "game_id"),
    ReferenceRule(
        "prediction_run_status_event",
        "prediction_run_id",
        "prediction_run",
        "prediction_run_id",
    ),
    ReferenceRule(
        "v26_live_hit_rule_set",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule("v26_slate", "rule_version", "v26_live_hit_rule_set", "rule_version"),
    ReferenceRule(
        "v26_slate",
        "position_eligibility_snapshot_id",
        "v26_player_position_eligibility",
        "position_eligibility_snapshot_id",
    ),
    ReferenceRule("v26_slate", "source_revision_id", "source_revision", "source_revision_id"),
    ReferenceRule(
        "v26_player_position_eligibility", "slate_id", "v26_slate", "slate_id"
    ),
    ReferenceRule(
        "v26_player_position_eligibility",
        "player_id",
        "player",
        "player_id",
    ),
    ReferenceRule(
        "v26_player_position_eligibility",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule(
        "v26_selection_snapshot",
        "slate_id",
        "v26_slate",
        "slate_id",
    ),
    ReferenceRule(
        "v26_selection_snapshot",
        "player_id",
        "player",
        "player_id",
    ),
    ReferenceRule(
        "v26_selection_snapshot",
        "rule_version",
        "v26_live_hit_rule_set",
        "rule_version",
    ),
    ReferenceRule(
        "v26_selection_snapshot",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule(
        "user_collection_snapshot",
        "player_id",
        "player",
        "player_id",
    ),
    ReferenceRule(
        "user_collection_snapshot",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule("v26_submission", "slate_id", "v26_slate", "slate_id"),
    ReferenceRule(
        "v26_submission", "selected_player_id", "player", "player_id"
    ),
    ReferenceRule(
        "v26_submission", "selected_synergy_team_id", "team", "team_id"
    ),
    ReferenceRule(
        "v26_submission", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("player", "source_revision_id", "source_revision", "source_revision_id"),
    ReferenceRule("team", "source_revision_id", "source_revision", "source_revision_id"),
    ReferenceRule(
        "stadium", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("game", "home_team_id", "team", "team_id"),
    ReferenceRule("game", "away_team_id", "team", "team_id"),
    ReferenceRule("game", "stadium_id", "stadium", "stadium_id"),
    ReferenceRule("game", "resumed_from_game_id", "game", "game_id"),
    ReferenceRule("game", "source_revision_id", "source_revision", "source_revision_id"),
    ReferenceRule("game_status_snapshot", "game_id", "game", "game_id"),
    ReferenceRule(
        "game_status_snapshot", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("starter_announcement", "game_id", "game", "game_id"),
    ReferenceRule("starter_announcement", "team_id", "team", "team_id"),
    ReferenceRule("starter_announcement", "pitcher_id", "player", "player_id"),
    ReferenceRule(
        "starter_announcement", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("team_season", "team_id", "team", "team_id"),
    ReferenceRule(
        "team_season", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("team_game", "game_id", "game", "game_id"),
    ReferenceRule("team_game", "team_id", "team", "team_id"),
    ReferenceRule("team_game", "opponent_team_id", "team", "team_id"),
    ReferenceRule(
        "team_game", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("player_game_batting", "game_id", "game", "game_id"),
    ReferenceRule(
        "player_game_batting", "team_game_id", "team_game", "team_game_id"
    ),
    ReferenceRule("player_game_batting", "team_id", "team", "team_id"),
    ReferenceRule("player_game_batting", "player_id", "player", "player_id"),
    ReferenceRule(
        "player_game_batting", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("roster_spell", "player_id", "player", "player_id"),
    ReferenceRule("roster_spell", "team_id", "team", "team_id"),
    ReferenceRule(
        "roster_spell", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("lineup_version", "game_id", "game", "game_id"),
    ReferenceRule("lineup_version", "team_id", "team", "team_id"),
    ReferenceRule(
        "lineup_version", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule(
        "lineup_entry", "lineup_version_id", "lineup_version", "lineup_version_id"
    ),
    ReferenceRule("lineup_entry", "game_id", "game", "game_id"),
    ReferenceRule("lineup_entry", "team_id", "team", "team_id"),
    ReferenceRule("lineup_entry", "player_id", "player", "player_id"),
    ReferenceRule(
        "lineup_entry", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("observed_plate_appearance", "game_id", "game", "game_id"),
    ReferenceRule("observed_plate_appearance", "batter_id", "player", "player_id"),
    ReferenceRule("observed_plate_appearance", "pitcher_id", "player", "player_id"),
    ReferenceRule("observed_plate_appearance", "catcher_id", "player", "player_id"),
    ReferenceRule(
        "observed_plate_appearance", "batting_team_id", "team", "team_id"
    ),
    ReferenceRule(
        "observed_plate_appearance", "fielding_team_id", "team", "team_id"
    ),
    ReferenceRule(
        "observed_plate_appearance",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule("substitution_event", "game_id", "game", "game_id"),
    ReferenceRule("substitution_event", "team_id", "team", "team_id"),
    ReferenceRule(
        "substitution_event", "outgoing_player_id", "player", "player_id"
    ),
    ReferenceRule(
        "substitution_event", "incoming_player_id", "player", "player_id"
    ),
    ReferenceRule(
        "substitution_event", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("runner_event", "game_id", "game", "game_id"),
    ReferenceRule("runner_event", "batting_team_id", "team", "team_id"),
    ReferenceRule("runner_event", "fielding_team_id", "team", "team_id"),
    ReferenceRule("runner_event", "runner_id", "player", "player_id"),
    ReferenceRule("runner_event", "pitcher_id", "player", "player_id"),
    ReferenceRule("runner_event", "catcher_id", "player", "player_id"),
    ReferenceRule(
        "runner_event", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("fielding_assignment", "game_id", "game", "game_id"),
    ReferenceRule("fielding_assignment", "team_id", "team", "team_id"),
    ReferenceRule("fielding_assignment", "player_id", "player", "player_id"),
    ReferenceRule(
        "fielding_assignment", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("catcher_assignment", "game_id", "game", "game_id"),
    ReferenceRule("catcher_assignment", "team_id", "team", "team_id"),
    ReferenceRule("catcher_assignment", "catcher_id", "player", "player_id"),
    ReferenceRule("catcher_assignment", "pitcher_id", "player", "player_id"),
    ReferenceRule(
        "catcher_assignment", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule("pitching_appearance", "game_id", "game", "game_id"),
    ReferenceRule(
        "pitching_appearance", "team_game_id", "team_game", "team_game_id"
    ),
    ReferenceRule("pitching_appearance", "team_id", "team", "team_id"),
    ReferenceRule("pitching_appearance", "player_id", "player", "player_id"),
    ReferenceRule(
        "pitching_appearance", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule(
        "weather_station_version", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule(
        "stadium_weather_station_map", "stadium_id", "stadium", "stadium_id"
    ),
    ReferenceRule(
        "stadium_weather_station_map",
        "station_id",
        "weather_station_version",
        "station_id",
    ),
    ReferenceRule(
        "stadium_weather_station_map",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule(
        "weather_forecast_snapshot", "stadium_id", "stadium", "stadium_id"
    ),
    ReferenceRule(
        "weather_forecast_snapshot",
        "source_revision_id",
        "source_revision",
        "source_revision_id",
    ),
    ReferenceRule("weather_observation", "stadium_id", "stadium", "stadium_id"),
    ReferenceRule(
        "weather_observation", "station_id", "weather_station_version", "station_id"
    ),
    ReferenceRule(
        "weather_observation", "source_revision_id", "source_revision", "source_revision_id"
    ),
    ReferenceRule(
        "player_game_candidate",
        "prediction_run_id",
        "prediction_run",
        "prediction_run_id",
    ),
    ReferenceRule("player_game_candidate", "game_id", "game", "game_id"),
    ReferenceRule("player_game_candidate", "player_id", "player", "player_id"),
    ReferenceRule("player_game_candidate", "team_id", "team", "team_id"),
    ReferenceRule("player_game_candidate", "opponent_team_id", "team", "team_id"),
    ReferenceRule(
        "player_state_snapshot",
        "prediction_run_id",
        "prediction_run",
        "prediction_run_id",
    ),
    ReferenceRule("player_state_snapshot", "player_id", "player", "player_id"),
    ReferenceRule(
        "team_state_snapshot",
        "prediction_run_id",
        "prediction_run",
        "prediction_run_id",
    ),
    ReferenceRule("team_state_snapshot", "team_id", "team", "team_id"),
    ReferenceRule(
        "model_prediction", "prediction_run_id", "prediction_run", "prediction_run_id"
    ),
)


COMPOSITE_REFERENCE_RULES: tuple[CompositeReferenceRule, ...] = (
    CompositeReferenceRule(
        "historical_boxscore", ("team_game_id", "game_id", "team_id", "opponent_team_id"),
        "team_game", ("team_game_id", "game_id", "team_id", "opponent_team_id"),
    ),
    CompositeReferenceRule(
        "lineup_entry",
        ("lineup_version_id", "game_id", "team_id"),
        "lineup_version",
        ("lineup_version_id", "game_id", "team_id"),
    ),
    CompositeReferenceRule(
        "pitching_appearance",
        ("team_game_id", "game_id", "team_id"),
        "team_game",
        ("team_game_id", "game_id", "team_id"),
    ),
    CompositeReferenceRule(
        "player_game_batting",
        ("team_game_id", "game_id", "team_id"),
        "team_game",
        ("team_game_id", "game_id", "team_id"),
    ),
    CompositeReferenceRule(
        "v26_player_position_eligibility",
        ("slate_id", "position_eligibility_snapshot_id"),
        "v26_slate",
        ("slate_id", "position_eligibility_snapshot_id"),
    ),
    CompositeReferenceRule(
        "v26_selection_snapshot",
        ("slate_id", "rule_version"),
        "v26_slate",
        ("slate_id", "rule_version"),
    ),
    CompositeReferenceRule(
        "v26_selection_snapshot",
        ("slate_id", "player_id", "position"),
        "v26_player_position_eligibility",
        ("slate_id", "player_id", "position"),
    ),
    CompositeReferenceRule(
        "v26_submission",
        ("slate_id", "selected_player_id", "position"),
        "v26_player_position_eligibility",
        ("slate_id", "player_id", "position"),
    ),
)


def find_reference_violations(
    connection: Any,
    *,
    sample_limit: int = 10,
    rules: tuple[ReferenceRule, ...] = REFERENCE_RULES,
) -> tuple[ReferenceViolation, ...]:
    """Return unresolved reference values without mutating the database."""

    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    violations: list[ReferenceViolation] = []
    for rule in rules:
        rows = connection.execute(
            f"""
            WITH missing AS (
                SELECT DISTINCT child."{rule.child_column}" AS missing_value
                FROM "{rule.child_table}" AS child
                WHERE child."{rule.child_column}" IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM "{rule.parent_table}" AS parent
                      WHERE parent."{rule.parent_column}" = child."{rule.child_column}"
                  )
            )
            SELECT CAST(missing_value AS VARCHAR), count(*) OVER () AS missing_count
            FROM missing
            ORDER BY CAST(missing_value AS VARCHAR)
            LIMIT ?
            """,
            [sample_limit],
        ).fetchall()
        if rows:
            violations.append(
                ReferenceViolation(
                    rule=rule,
                    missing_value_count=int(rows[0][1]),
                    sample_values=tuple(str(row[0]) for row in rows),
                )
            )
    return tuple(violations)


def find_composite_reference_violations(
    connection: Any,
    *,
    sample_limit: int = 10,
    rules: tuple[CompositeReferenceRule, ...] = COMPOSITE_REFERENCE_RULES,
) -> tuple[CompositeReferenceViolation, ...]:
    """Return mismatched multi-column business references."""

    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    violations: list[CompositeReferenceViolation] = []
    for rule in rules:
        present = " AND ".join(
            f'child."{column}" IS NOT NULL' for column in rule.child_columns
        )
        matches = " AND ".join(
            f'parent."{parent}" = child."{child}"'
            for child, parent in zip(
                rule.child_columns,
                rule.parent_columns,
                strict=True,
            )
        )
        sample = " || '|' || ".join(
            f'CAST(child."{column}" AS VARCHAR)' for column in rule.child_columns
        )
        rows = connection.execute(
            f"""
            WITH missing AS (
                SELECT DISTINCT {sample} AS missing_value
                FROM "{rule.child_table}" AS child
                WHERE {present}
                  AND NOT EXISTS (
                      SELECT 1
                      FROM "{rule.parent_table}" AS parent
                      WHERE {matches}
                  )
            )
            SELECT missing_value, count(*) OVER () AS missing_count
            FROM missing
            ORDER BY missing_value
            LIMIT ?
            """,
            [sample_limit],
        ).fetchall()
        if rows:
            violations.append(
                CompositeReferenceViolation(
                    rule=rule,
                    missing_value_count=int(rows[0][1]),
                    sample_values=tuple(str(row[0]) for row in rows),
                )
            )
    return tuple(violations)


def assert_referential_integrity(
    connection: Any,
    *,
    sample_limit: int = 10,
) -> None:
    """Raise with all declared unresolved references."""

    violations = find_reference_violations(connection, sample_limit=sample_limit)
    if violations:
        raise ReferentialIntegrityError(violations)


def assert_composite_referential_integrity(
    connection: Any,
    *,
    sample_limit: int = 10,
) -> None:
    """Raise when any declared multi-column reference does not resolve."""

    violations = find_composite_reference_violations(
        connection,
        sample_limit=sample_limit,
    )
    if violations:
        raise CompositeReferentialIntegrityError(violations)
