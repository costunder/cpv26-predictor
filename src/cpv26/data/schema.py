"""DuckDB schema for append-only, point-in-time baseball data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .schema_v4 import (
    V4_COLUMN_UPGRADE_DDL,
    V4_DDL,
    V4_DOMAIN_TIMESTAMP_COLUMNS,
    V4_EXTRA_REQUIRED_COLUMNS,
    V4_INDEX_DDL,
    V4_REQUIRED_COLUMNS,
    V4_REQUIRED_INDEXES,
    V4_TABLE_DDL,
    V4_TABLE_SPECS,
    V4_TIMEZONE_COLUMNS,
)

SCHEMA_VERSION = 4

TEMPORAL_COLUMNS = (
    "event_at",
    "available_at",
    "ingested_at",
    "valid_from",
    "valid_to",
)

DOMAIN_TIMESTAMP_COLUMNS = tuple(sorted(V4_DOMAIN_TIMESTAMP_COLUMNS))


@dataclass(frozen=True, slots=True)
class TableDefinition:
    name: str
    row_identity: str
    natural_identity: tuple[str, ...]
    generated: bool = False


TABLE_DEFINITIONS: dict[str, TableDefinition] = {
    "source_revision": TableDefinition(
        "source_revision", "source_revision_id", ("source_revision_id",)
    ),
    "prediction_run": TableDefinition(
        "prediction_run", "prediction_run_row_id", ("prediction_run_id",), True
    ),
    "prediction_run_status_event": TableDefinition(
        "prediction_run_status_event",
        "prediction_run_status_event_id",
        ("prediction_run_status_event_id",),
        True,
    ),
    "v26_live_hit_rule_set": TableDefinition(
        "v26_live_hit_rule_set",
        "v26_live_hit_rule_set_row_id",
        ("rule_version",),
    ),
    "v26_player_position_eligibility": TableDefinition(
        "v26_player_position_eligibility",
        "v26_player_position_eligibility_row_id",
        (
            "position_eligibility_snapshot_id",
            "slate_id",
            "live_card_version",
            "player_id",
            "position",
        ),
    ),
    "v26_selection_snapshot": TableDefinition(
        "v26_selection_snapshot",
        "v26_selection_snapshot_row_id",
        ("selection_snapshot_id", "slate_id", "player_id", "position"),
    ),
    "user_collection_snapshot": TableDefinition(
        "user_collection_snapshot",
        "user_collection_snapshot_row_id",
        ("user_id", "live_card_version", "player_id"),
    ),
    "player": TableDefinition("player", "player_row_id", ("player_id",)),
    "team": TableDefinition("team", "team_row_id", ("team_id",)),
    "game": TableDefinition("game", "game_row_id", ("game_id",)),
    "team_season": TableDefinition("team_season", "team_season_row_id", ("team_season_id",)),
    "team_game": TableDefinition("team_game", "team_game_row_id", ("team_game_id",)),
    "roster_spell": TableDefinition("roster_spell", "roster_spell_row_id", ("roster_spell_id",)),
    "lineup_version": TableDefinition(
        "lineup_version", "lineup_version_row_id", ("lineup_version_id",)
    ),
    "lineup_entry": TableDefinition("lineup_entry", "lineup_entry_row_id", ("lineup_entry_id",)),
    "observed_plate_appearance": TableDefinition(
        "observed_plate_appearance",
        "observed_pa_row_id",
        ("plate_appearance_id",),
    ),
    "pitching_appearance": TableDefinition(
        "pitching_appearance",
        "pitching_appearance_row_id",
        ("pitching_appearance_id",),
    ),
    "player_game_candidate": TableDefinition(
        "player_game_candidate",
        "candidate_row_id",
        ("prediction_run_id", "candidate_id"),
        True,
    ),
    "player_state_snapshot": TableDefinition(
        "player_state_snapshot",
        "player_state_snapshot_id",
        ("prediction_run_id", "player_id", "role"),
        True,
    ),
    "team_state_snapshot": TableDefinition(
        "team_state_snapshot",
        "team_state_snapshot_id",
        ("prediction_run_id", "team_id"),
        True,
    ),
    "model_prediction": TableDefinition(
        "model_prediction", "prediction_id", ("prediction_id",), True
    ),
}

TABLE_DEFINITIONS.update(
    {
        name: TableDefinition(name, row_identity, natural_identity)
        for name, row_identity, natural_identity in V4_TABLE_SPECS
    }
)


SCHEMA_MIGRATION_DDL = """
    CREATE TABLE IF NOT EXISTS schema_migration (
        schema_version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL,
        description VARCHAR NOT NULL
    )
"""

PREDICTION_RUN_DDL = """
    CREATE TABLE IF NOT EXISTS prediction_run (
        prediction_run_row_id VARCHAR PRIMARY KEY,
        prediction_run_id VARCHAR NOT NULL UNIQUE,
        target_game_id VARCHAR NOT NULL,
        cutoff_at TIMESTAMPTZ NOT NULL,
        knowledge_at TIMESTAMPTZ NOT NULL,
        horizon_type VARCHAR NOT NULL,
        feature_version VARCHAR NOT NULL,
        model_name VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL,
        simulator_version VARCHAR NOT NULL,
        v26_rule_version VARCHAR NOT NULL,
        feature_fingerprint VARCHAR,
        config_json VARCHAR NOT NULL DEFAULT '{}',
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (horizon_type IN ('early', 'starter_known', 'lineup_known', 'near_lock')),
        CHECK (knowledge_at >= cutoff_at),
        CHECK (feature_fingerprint IS NULL OR length(feature_fingerprint) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

PREDICTION_RUN_STATUS_EVENT_DDL = """
    CREATE TABLE IF NOT EXISTS prediction_run_status_event (
        prediction_run_status_event_id VARCHAR PRIMARY KEY,
        prediction_run_id VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        detail_json VARCHAR NOT NULL DEFAULT '{}',
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (status IN ('created', 'snapshotted', 'scored', 'failed')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

V26_LIVE_HIT_RULE_SET_DDL = """
    CREATE TABLE IF NOT EXISTS v26_live_hit_rule_set (
        v26_live_hit_rule_set_row_id VARCHAR PRIMARY KEY,
        rule_version VARCHAR NOT NULL,
        position_eligibility_snapshot_id VARCHAR,
        rule_payload_json VARCHAR NOT NULL,
        provenance_kind VARCHAR NOT NULL,
        provenance_json VARCHAR NOT NULL DEFAULT '{}',
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(trim(rule_version)) > 0),
        CHECK (
            position_eligibility_snapshot_id IS NULL
            OR length(trim(position_eligibility_snapshot_id)) > 0
        ),
        CHECK (json_valid(rule_payload_json)),
        CHECK (provenance_kind IN (
            'official', 'in_game_observation', 'user_supplied', 'inferred', 'test_fixture'
        )),
        CHECK (json_valid(provenance_json)),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

V26_PLAYER_POSITION_ELIGIBILITY_DDL = """
    CREATE TABLE IF NOT EXISTS v26_player_position_eligibility (
        v26_player_position_eligibility_row_id VARCHAR PRIMARY KEY,
        position_eligibility_snapshot_id VARCHAR NOT NULL,
        slate_id VARCHAR NOT NULL,
        slate_date DATE NOT NULL,
        live_card_version VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        position VARCHAR NOT NULL,
        is_eligible BOOLEAN NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(trim(position_eligibility_snapshot_id)) > 0),
        CHECK (length(trim(slate_id)) > 0),
        CHECK (length(trim(live_card_version)) > 0),
        CHECK (length(trim(position)) > 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

V26_SELECTION_SNAPSHOT_DDL = """
    CREATE TABLE IF NOT EXISTS v26_selection_snapshot (
        v26_selection_snapshot_row_id VARCHAR PRIMARY KEY,
        selection_snapshot_id VARCHAR NOT NULL,
        slate_id VARCHAR NOT NULL,
        lock_at TIMESTAMPTZ NOT NULL,
        player_id VARCHAR NOT NULL,
        position VARCHAR NOT NULL,
        selection_rate DOUBLE NOT NULL,
        rule_version VARCHAR NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        capture_phase VARCHAR NOT NULL DEFAULT 'unspecified',
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(trim(selection_snapshot_id)) > 0),
        CHECK (length(trim(slate_id)) > 0),
        CHECK (length(trim(position)) > 0),
        CHECK (selection_rate BETWEEN 0.0 AND 1.0),
        CHECK (length(trim(rule_version)) > 0),
        CHECK (capture_phase IN (
            'unspecified', 'early', 'starter_known', 'lineup_known', 'near_lock'
        )),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

USER_COLLECTION_SNAPSHOT_DDL = """
    CREATE TABLE IF NOT EXISTS user_collection_snapshot (
        user_collection_snapshot_row_id VARCHAR PRIMARY KEY,
        collection_snapshot_id VARCHAR NOT NULL,
        user_id VARCHAR NOT NULL,
        live_card_version VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        owned BOOLEAN NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(trim(collection_snapshot_id)) > 0),
        CHECK (length(trim(user_id)) > 0),
        CHECK (length(trim(live_card_version)) > 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
"""

LIVE_HIT_DDL: tuple[str, ...] = (
    V26_LIVE_HIT_RULE_SET_DDL,
    V26_PLAYER_POSITION_ELIGIBILITY_DDL,
    V26_SELECTION_SNAPSHOT_DDL,
    USER_COLLECTION_SNAPSHOT_DDL,
    (
        "CREATE INDEX IF NOT EXISTS idx_v26_live_hit_rule_asof "
        "ON v26_live_hit_rule_set (rule_version, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_v26_position_eligibility_snapshot "
        "ON v26_player_position_eligibility "
        "(position_eligibility_snapshot_id, slate_id, live_card_version, "
        "player_id, position, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_v26_selection_snapshot "
        "ON v26_selection_snapshot "
        "(selection_snapshot_id, slate_id, player_id, position, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_user_collection_asof "
        "ON user_collection_snapshot "
        "(user_id, live_card_version, player_id, available_at, ingested_at)"
    ),
)


DDL: tuple[str, ...] = (
    SCHEMA_MIGRATION_DDL,
    """
    CREATE TABLE IF NOT EXISTS source_revision (
        source_revision_id VARCHAR PRIMARY KEY,
        source_name VARCHAR NOT NULL,
        source_locator VARCHAR,
        content_sha256 VARCHAR NOT NULL,
        metadata_json VARCHAR NOT NULL DEFAULT '{}',
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(content_sha256) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    PREDICTION_RUN_DDL,
    PREDICTION_RUN_STATUS_EVENT_DDL,
    *LIVE_HIT_DDL,
    """
    CREATE TABLE IF NOT EXISTS player (
        player_row_id VARCHAR PRIMARY KEY,
        player_id VARCHAR NOT NULL,
        display_name VARCHAR NOT NULL,
        birth_date DATE,
        bats VARCHAR,
        throws VARCHAR,
        primary_position VARCHAR,
        debut_year INTEGER,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (bats IS NULL OR bats IN ('L', 'R', 'S')),
        CHECK (throws IS NULL OR throws IN ('L', 'R')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team (
        team_row_id VARCHAR PRIMARY KEY,
        team_id VARCHAR NOT NULL,
        team_name VARCHAR NOT NULL,
        short_name VARCHAR,
        city VARCHAR,
        active_from DATE,
        active_to DATE,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (active_to IS NULL OR active_from IS NULL OR active_to >= active_from),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS game (
        game_row_id VARCHAR PRIMARY KEY,
        game_id VARCHAR NOT NULL,
        season INTEGER NOT NULL,
        game_type VARCHAR NOT NULL DEFAULT 'regular',
        scheduled_start TIMESTAMPTZ NOT NULL,
        home_team_id VARCHAR NOT NULL,
        away_team_id VARCHAR NOT NULL,
        stadium_id VARCHAR,
        doubleheader_number INTEGER,
        resumed_from_game_id VARCHAR,
        game_status VARCHAR NOT NULL,
        home_score INTEGER,
        away_score INTEGER,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (home_team_id <> away_team_id),
        CHECK (doubleheader_number IS NULL OR doubleheader_number IN (1, 2)),
        CHECK (resumed_from_game_id IS NULL OR resumed_from_game_id <> game_id),
        CHECK (home_score IS NULL OR home_score >= 0),
        CHECK (away_score IS NULL OR away_score >= 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roster_spell (
        roster_spell_row_id VARCHAR PRIMARY KEY,
        roster_spell_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        roster_status VARCHAR NOT NULL,
        uniform_number VARCHAR,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_season (
        team_season_row_id VARCHAR PRIMARY KEY,
        team_season_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        season INTEGER NOT NULL,
        league_name VARCHAR NOT NULL DEFAULT 'KBO',
        games_scheduled INTEGER,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (games_scheduled IS NULL OR games_scheduled >= 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_game (
        team_game_row_id VARCHAR PRIMARY KEY,
        team_game_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        opponent_team_id VARCHAR NOT NULL,
        is_home BOOLEAN NOT NULL,
        runs INTEGER,
        hits INTEGER,
        errors INTEGER,
        result VARCHAR,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (team_id <> opponent_team_id),
        CHECK (runs IS NULL OR runs >= 0),
        CHECK (hits IS NULL OR hits >= 0),
        CHECK (errors IS NULL OR errors >= 0),
        CHECK (result IS NULL OR result IN ('win', 'draw', 'loss', 'no_result')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineup_version (
        lineup_version_row_id VARCHAR PRIMARY KEY,
        lineup_version_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        version_number INTEGER NOT NULL,
        lineup_status VARCHAR NOT NULL,
        published_at TIMESTAMPTZ,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (version_number >= 1),
        CHECK (lineup_status IN ('projected', 'announced', 'official', 'final')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineup_entry (
        lineup_entry_row_id VARCHAR PRIMARY KEY,
        lineup_entry_id VARCHAR NOT NULL,
        lineup_version_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        batting_order INTEGER,
        fielding_position VARCHAR,
        is_starter BOOLEAN NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (batting_order IS NULL OR batting_order BETWEEN 1 AND 9),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS observed_plate_appearance (
        observed_pa_row_id VARCHAR PRIMARY KEY,
        plate_appearance_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        inning INTEGER NOT NULL,
        half_inning VARCHAR NOT NULL,
        sequence_in_game INTEGER NOT NULL,
        event_subsequence INTEGER NOT NULL DEFAULT 0,
        batter_id VARCHAR NOT NULL,
        pitcher_id VARCHAR NOT NULL,
        catcher_id VARCHAR,
        batting_team_id VARCHAR NOT NULL,
        fielding_team_id VARCHAR NOT NULL,
        home_score_before INTEGER,
        away_score_before INTEGER,
        outs_before INTEGER NOT NULL,
        runners_before VARCHAR NOT NULL,
        outs_added INTEGER,
        runners_after VARCHAR,
        home_score_after INTEGER,
        away_score_after INTEGER,
        transition_complete BOOLEAN NOT NULL DEFAULT FALSE,
        outcome VARCHAR NOT NULL,
        is_at_bat BOOLEAN NOT NULL,
        is_hit BOOLEAN NOT NULL,
        total_bases INTEGER NOT NULL,
        runs_scored INTEGER NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (inning >= 1),
        CHECK (half_inning IN ('top', 'bottom')),
        CHECK (sequence_in_game >= 1),
        CHECK (event_subsequence >= 0),
        CHECK (home_score_before IS NULL OR home_score_before >= 0),
        CHECK (away_score_before IS NULL OR away_score_before >= 0),
        CHECK (outs_before BETWEEN 0 AND 2),
        CHECK (regexp_matches(runners_before, '^[01]{3}$')),
        CHECK (outs_added IS NULL OR outs_added BETWEEN 0 AND 3),
        CHECK (runners_after IS NULL OR regexp_matches(runners_after, '^[01]{3}$')),
        CHECK (home_score_after IS NULL OR home_score_after >= 0),
        CHECK (away_score_after IS NULL OR away_score_after >= 0),
        CHECK (
            NOT transition_complete OR (
                home_score_before IS NOT NULL
                AND away_score_before IS NOT NULL
                AND outs_added IS NOT NULL
                AND runners_after IS NOT NULL
                AND home_score_after IS NOT NULL
                AND away_score_after IS NOT NULL
            )
        ),
        CHECK (total_bases BETWEEN 0 AND 4),
        CHECK (runs_scored >= 0),
        CHECK (batting_team_id <> fielding_team_id),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_game_candidate (
        candidate_row_id VARCHAR PRIMARY KEY,
        candidate_id VARCHAR NOT NULL,
        prediction_run_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        opponent_team_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL,
        lineup_slot INTEGER,
        fielding_position VARCHAR,
        start_probability DOUBLE NOT NULL,
        expected_plate_appearances DOUBLE NOT NULL,
        scenario_weight DOUBLE NOT NULL DEFAULT 1.0,
        scenario_id VARCHAR NOT NULL DEFAULT 'base',
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (team_id <> opponent_team_id),
        CHECK (lineup_slot IS NULL OR lineup_slot BETWEEN 1 AND 9),
        CHECK (start_probability BETWEEN 0.0 AND 1.0),
        CHECK (expected_plate_appearances >= 0.0),
        CHECK (scenario_weight > 0.0 AND scenario_weight <= 1.0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pitching_appearance (
        pitching_appearance_row_id VARCHAR PRIMARY KEY,
        pitching_appearance_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        appearance_role VARCHAR NOT NULL,
        sequence_in_game INTEGER NOT NULL,
        outs_recorded INTEGER NOT NULL,
        batters_faced INTEGER NOT NULL,
        pitches_thrown INTEGER,
        hits_allowed INTEGER NOT NULL,
        walks_allowed INTEGER NOT NULL,
        strikeouts INTEGER NOT NULL,
        runs_allowed INTEGER NOT NULL,
        earned_runs INTEGER NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (appearance_role IN ('starter', 'reliever')),
        CHECK (sequence_in_game >= 1),
        CHECK (outs_recorded >= 0),
        CHECK (batters_faced >= 0),
        CHECK (pitches_thrown IS NULL OR pitches_thrown >= 0),
        CHECK (hits_allowed >= 0),
        CHECK (walks_allowed >= 0),
        CHECK (strikeouts >= 0),
        CHECK (runs_allowed >= 0),
        CHECK (earned_runs >= 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_state_snapshot (
        player_state_snapshot_id VARCHAR PRIMARY KEY,
        prediction_run_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL,
        cutoff_at TIMESTAMPTZ NOT NULL,
        numerator_json VARCHAR NOT NULL,
        denominator_json VARCHAR NOT NULL,
        feature_json VARCHAR NOT NULL,
        source_fingerprint VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(source_fingerprint) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_state_snapshot (
        team_state_snapshot_id VARCHAR PRIMARY KEY,
        prediction_run_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        cutoff_at TIMESTAMPTZ NOT NULL,
        numerator_json VARCHAR NOT NULL,
        denominator_json VARCHAR NOT NULL,
        feature_json VARCHAR NOT NULL,
        source_fingerprint VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(source_fingerprint) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_prediction (
        prediction_id VARCHAR PRIMARY KEY,
        prediction_run_id VARCHAR NOT NULL,
        model_name VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL,
        prediction_kind VARCHAR NOT NULL,
        entity_id VARCHAR NOT NULL,
        game_id VARCHAR,
        target_name VARCHAR NOT NULL,
        label VARCHAR,
        value DOUBLE NOT NULL,
        distribution_json VARCHAR NOT NULL DEFAULT '{}',
        standard_error DOUBLE,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (prediction_kind IN ('plate_appearance', 'player_hits', 'team_runs', 'game_outcome')),
        CHECK (standard_error IS NULL OR standard_error >= 0.0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_source_revision_asof "
        "ON source_revision (available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_prediction_run_id "
        "ON prediction_run (prediction_run_id, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_prediction_run_status "
        "ON prediction_run_status_event (prediction_run_id, available_at, ingested_at)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_player_asof ON player (player_id, available_at, ingested_at)",
    "CREATE INDEX IF NOT EXISTS idx_team_asof ON team (team_id, available_at, ingested_at)",
    "CREATE INDEX IF NOT EXISTS idx_game_asof ON game (game_id, available_at, ingested_at)",
    (
        "CREATE INDEX IF NOT EXISTS idx_team_season_asof "
        "ON team_season (team_id, season, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_team_game_asof "
        "ON team_game (game_id, team_id, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_roster_asof "
        "ON roster_spell (player_id, team_id, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_lineup_version_game "
        "ON lineup_version (game_id, team_id, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_lineup_entry_game "
        "ON lineup_entry (game_id, team_id, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_pa_event "
        "ON observed_plate_appearance (event_at, batter_id, pitcher_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_candidate_run "
        "ON player_game_candidate (prediction_run_id, game_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_pitching_appearance_event "
        "ON pitching_appearance (event_at, player_id, team_game_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_player_snapshot_run "
        "ON player_state_snapshot (prediction_run_id, player_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_team_snapshot_run "
        "ON team_state_snapshot (prediction_run_id, team_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_prediction_run_entity "
        "ON model_prediction (prediction_run_id, prediction_kind, entity_id)"
    ),
    *V4_DDL,
)


def install_schema(connection: Any) -> None:
    """Install schema v4 or transactionally migrate a compatible v1/v2/v3 database."""

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(SCHEMA_MIGRATION_DDL)
        row = connection.execute("SELECT max(schema_version) FROM schema_migration").fetchone()
        installed = row[0] if row else None
        existing_tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if installed is None and "prediction_run" in existing_tables:
            raise RuntimeError(
                "cannot initialise an unversioned database containing prediction_run"
            )
        if installed == 1:
            _migrate_v1_to_v2(connection)
            _record_migration(
                connection,
                2,
                "immutable prediction runs and append-only status events",
            )
            installed = 2
        if installed == 2:
            _migrate_v2_to_v3(connection)
            _record_migration(
                connection,
                3,
                "point-in-time V26 Live Hit rules, eligibility, selection, and collection",
            )
            installed = 3
        if installed == 3:
            _migrate_v3_to_v4(connection)
            _record_migration(
                connection,
                4,
                "replay events, dataset reconciliation, weather, and V26 slate state",
            )
            installed = 4
        if installed is not None and installed != SCHEMA_VERSION:
            raise RuntimeError(
                "database schema version is "
                f"{installed!r}; expected 1, 2, 3, or {SCHEMA_VERSION}"
            )
        for statement in DDL:
            connection.execute(statement)
        _record_migration(
            connection,
            SCHEMA_VERSION,
            "replay events, dataset reconciliation, weather, and V26 slate state",
        )
        assert_schema_current(connection)
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def assert_schema_current(connection: Any) -> None:
    """Raise when the database was created by a newer or incomplete schema."""

    row = connection.execute("SELECT max(schema_version) FROM schema_migration").fetchone()
    installed = row[0] if row else None
    if installed != SCHEMA_VERSION:
        raise RuntimeError(f"database schema version is {installed!r}; expected {SCHEMA_VERSION}")
    actual_tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    expected_tables = {"schema_migration", *TABLE_DEFINITIONS}
    missing_tables = expected_tables - actual_tables
    if missing_tables:
        raise RuntimeError(
            "database is missing schema tables: " + ", ".join(sorted(missing_tables))
        )
    for table, definition in TABLE_DEFINITIONS.items():
        info = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        columns = {row[1] for row in info}
        missing_temporal = set(TEMPORAL_COLUMNS) - columns
        if missing_temporal:
            raise RuntimeError(
                f"table {table} is missing temporal columns: " + ", ".join(sorted(missing_temporal))
            )
        missing_identity = {definition.row_identity, *definition.natural_identity} - columns
        if missing_identity:
            raise RuntimeError(
                f"table {table} is missing identity columns: "
                + ", ".join(sorted(missing_identity))
            )
        types = {row[1]: str(row[2]).upper() for row in info}
        wrong_temporal_types = {
            column
            for column in TEMPORAL_COLUMNS
            if "TIMESTAMP WITH TIME ZONE" not in types[column]
        }
        if wrong_temporal_types:
            raise RuntimeError(
                f"table {table} has non-timezone temporal columns: "
                + ", ".join(sorted(wrong_temporal_types))
            )
    prediction_run_columns = {
        row[1] for row in connection.execute("PRAGMA table_info('prediction_run')").fetchall()
    }
    required_run_columns = {
        "prediction_run_row_id",
        "prediction_run_id",
        "target_game_id",
        "cutoff_at",
        "knowledge_at",
        "horizon_type",
        "feature_version",
        "model_name",
        "model_version",
        "simulator_version",
        "v26_rule_version",
        "feature_fingerprint",
        "config_json",
        *TEMPORAL_COLUMNS,
    }
    missing_run_columns = required_run_columns - prediction_run_columns
    if missing_run_columns or "status" in prediction_run_columns:
        detail = ", ".join(sorted(missing_run_columns)) or "legacy status column present"
        raise RuntimeError(f"prediction_run does not satisfy the v2 contract: {detail}")
    if not _prediction_run_id_is_unique(connection):
        raise RuntimeError("prediction_run.prediction_run_id must have a UNIQUE constraint")
    _assert_required_columns(
        connection,
        "prediction_run_status_event",
        {
            "prediction_run_status_event_id",
            "prediction_run_id",
            "status",
            "detail_json",
            *TEMPORAL_COLUMNS,
        },
    )
    _assert_required_columns(
        connection,
        "v26_live_hit_rule_set",
        {
            "v26_live_hit_rule_set_row_id",
            "rule_version",
            "position_eligibility_snapshot_id",
            "rule_payload_json",
            "provenance_kind",
            "provenance_json",
            "source_revision_id",
            *TEMPORAL_COLUMNS,
        },
    )
    _assert_required_columns(
        connection,
        "v26_player_position_eligibility",
        {
            "v26_player_position_eligibility_row_id",
            "position_eligibility_snapshot_id",
            "slate_id",
            "slate_date",
            "live_card_version",
            "player_id",
            "position",
            "is_eligible",
            "captured_at",
            "source_revision_id",
            *TEMPORAL_COLUMNS,
        },
    )
    _assert_required_columns(
        connection,
        "v26_selection_snapshot",
        {
            "v26_selection_snapshot_row_id",
            "selection_snapshot_id",
            "slate_id",
            "lock_at",
            "player_id",
            "position",
            "selection_rate",
            "rule_version",
            "captured_at",
            "capture_phase",
            "source_revision_id",
            *TEMPORAL_COLUMNS,
        },
    )
    _assert_required_columns(
        connection,
        "user_collection_snapshot",
        {
            "user_collection_snapshot_row_id",
            "collection_snapshot_id",
            "user_id",
            "live_card_version",
            "player_id",
            "owned",
            "captured_at",
            "source_revision_id",
            *TEMPORAL_COLUMNS,
        },
    )
    for table, required_columns in V4_EXTRA_REQUIRED_COLUMNS.items():
        _assert_required_columns(connection, table, set(required_columns))
    for table, required_columns in V4_REQUIRED_COLUMNS.items():
        _assert_required_columns(connection, table, set(required_columns))
    for table, timestamp_columns in V4_TIMEZONE_COLUMNS.items():
        _assert_timezone_columns(connection, table, set(timestamp_columns))
    _assert_required_indexes(
        connection,
        {
            "idx_v26_live_hit_rule_asof",
            "idx_v26_position_eligibility_snapshot",
            "idx_v26_selection_snapshot",
            "idx_user_collection_asof",
            *V4_REQUIRED_INDEXES,
        },
    )


def _migrate_v1_to_v2(connection: Any) -> None:
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "prediction_run" not in tables:
        raise RuntimeError("schema v1 database is missing prediction_run")
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info('prediction_run')").fetchall()
    }
    required = {
        "prediction_run_row_id",
        "prediction_run_id",
        "target_game_id",
        "cutoff_at",
        "knowledge_at",
        "horizon_type",
        "feature_version",
        "model_name",
        "model_version",
        "simulator_version",
        "v26_rule_version",
        "status",
        "feature_fingerprint",
        "config_json",
        *TEMPORAL_COLUMNS,
    }
    missing = required - columns
    if missing:
        raise RuntimeError(
            "schema v1 prediction_run cannot be migrated; missing columns: "
            + ", ".join(sorted(missing))
        )
    duplicates = connection.execute(
        """
        SELECT prediction_run_id, count(*) AS revision_count
        FROM prediction_run
        GROUP BY prediction_run_id
        HAVING count(*) > 1
        ORDER BY prediction_run_id
        """
    ).fetchall()
    if duplicates:
        identifiers = ", ".join(f"{row[0]} ({row[1]} rows)" for row in duplicates)
        raise RuntimeError(
            "schema v1 migration requires one immutable row per prediction_run_id; "
            f"duplicates found: {identifiers}"
        )
    # DuckDB refuses to rename a table while user-created indexes depend on it.
    # Drop every v1 index on this table; the canonical v2 indexes are recreated
    # from ``DDL`` after the migration finishes.
    for (index_name,) in connection.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'prediction_run'
        ORDER BY index_name
        """
    ).fetchall():
        quoted_name = str(index_name).replace('"', '""')
        connection.execute(f'DROP INDEX "{quoted_name}"')
    connection.execute("ALTER TABLE prediction_run RENAME TO prediction_run_v1_backup")
    connection.execute(PREDICTION_RUN_DDL)
    connection.execute(PREDICTION_RUN_STATUS_EVENT_DDL)
    connection.execute(
        """
        INSERT INTO prediction_run (
            prediction_run_row_id, prediction_run_id, target_game_id,
            cutoff_at, knowledge_at, horizon_type, feature_version,
            model_name, model_version, simulator_version, v26_rule_version,
            feature_fingerprint, config_json,
            event_at, available_at, ingested_at, valid_from, valid_to
        )
        SELECT
            prediction_run_row_id, prediction_run_id, target_game_id,
            cutoff_at, knowledge_at, horizon_type, feature_version,
            model_name, model_version, simulator_version, v26_rule_version,
            feature_fingerprint, config_json,
            event_at, available_at, ingested_at, valid_from, valid_to
        FROM prediction_run_v1_backup
        """
    )
    connection.execute(
        """
        INSERT INTO prediction_run_status_event (
            prediction_run_status_event_id, prediction_run_id, status, detail_json,
            event_at, available_at, ingested_at, valid_from, valid_to
        )
        SELECT
            'migration-v1-' || prediction_run_row_id,
            prediction_run_id,
            status,
            '{"migration":"v1_to_v2"}',
            event_at, available_at, ingested_at, valid_from, valid_to
        FROM prediction_run_v1_backup
        """
    )
    connection.execute("DROP TABLE prediction_run_v1_backup")


def _migrate_v2_to_v3(connection: Any) -> None:
    """Add Live Hit source-state tables without rewriting existing v2 facts."""

    for statement in LIVE_HIT_DDL:
        connection.execute(statement)


def _migrate_v3_to_v4(connection: Any) -> None:
    """Add replay/weather tables and preserve unknown legacy transition fields.

    Some historical migration fixtures contain only the tables relevant to the
    earlier migration.  Column upgrades are therefore applied only when their
    target table already exists.  The canonical ``DDL`` pass that follows the
    migration creates missing v1/v2 tables and applies the same idempotent
    upgrades afterwards.
    """

    for statement in V4_TABLE_DDL:
        connection.execute(statement)
    existing_tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    for statement in V4_COLUMN_UPGRADE_DDL:
        tokens = statement.split()
        target_table = tokens[2] if tokens[0].upper() == "ALTER" else tokens[1]
        if target_table in existing_tables:
            connection.execute(statement)
    for statement in V4_INDEX_DDL:
        connection.execute(statement)


def _record_migration(
    connection: Any,
    schema_version: int,
    description: str,
) -> None:
    connection.execute(
        """
        INSERT INTO schema_migration (schema_version, applied_at, description)
        VALUES (?, now(), ?)
        ON CONFLICT DO NOTHING
        """,
        [schema_version, description],
    )


def _prediction_run_id_is_unique(connection: Any) -> bool:
    rows = connection.execute(
        """
        SELECT constraint_type, constraint_column_names
        FROM duckdb_constraints()
        WHERE table_name = 'prediction_run'
        """
    ).fetchall()
    return any(
        row[0] in {"UNIQUE", "PRIMARY KEY"} and list(row[1]) == ["prediction_run_id"]
        for row in rows
    )


def _assert_required_columns(
    connection: Any,
    table: str,
    required_columns: set[str],
) -> None:
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    }
    missing = required_columns - columns
    if missing:
        raise RuntimeError(
            f"table {table} is missing required columns: " + ", ".join(sorted(missing))
        )


def _assert_timezone_columns(connection: Any, table: str, required_columns: set[str]) -> None:
    types = {
        row[1]: str(row[2]).upper()
        for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    }
    wrong = {
        column
        for column in required_columns
        if "TIMESTAMP WITH TIME ZONE" not in types.get(column, "")
    }
    if wrong:
        raise RuntimeError(
            f"table {table} has non-timezone timestamp columns: "
            + ", ".join(sorted(wrong))
        )


def _assert_required_indexes(connection: Any, required_indexes: set[str]) -> None:
    indexes = {
        str(row[0])
        for row in connection.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    }
    missing = required_indexes - indexes
    if missing:
        raise RuntimeError("database is missing schema indexes: " + ", ".join(sorted(missing)))


def table_names(*, include_metadata: bool = False) -> tuple[str, ...]:
    names: Iterable[str] = TABLE_DEFINITIONS
    if include_metadata:
        return ("schema_migration", *names)
    return tuple(names)
