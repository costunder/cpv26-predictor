"""Lossless partial historical box scores, without fabricated player identities."""

from __future__ import annotations

V5_TABLE_SPECS = (
    ("historical_boxscore", "boxscore_row_id", ("observation_id",)),
    ("historical_game_detail", "detail_row_id", ("game_id",)),
)

V5_DDL = (
    """
    CREATE TABLE IF NOT EXISTS historical_boxscore (
        boxscore_row_id VARCHAR PRIMARY KEY,
        observation_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        opponent_team_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL CHECK (role IN ('batting', 'pitching')),
        side VARCHAR NOT NULL CHECK (side IN ('away', 'home')),
        player_id VARCHAR NOT NULL,
        identity_status VARCHAR NOT NULL CHECK (identity_status = 'source_observation'),
        display_name VARCHAR,
        row_index INTEGER NOT NULL CHECK (row_index >= 0),
        stats_json VARCHAR NOT NULL,
        raw_json VARCHAR NOT NULL,
        quality_json VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (team_id <> opponent_team_id),
        CHECK (json_valid(stats_json) AND json_valid(raw_json) AND json_valid(quality_json)),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_game_detail (
        detail_row_id VARCHAR PRIMARY KEY,
        game_id VARCHAR NOT NULL,
        metadata_json VARCHAR NOT NULL,
        quality_json VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (json_valid(metadata_json) AND json_valid(quality_json)),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_historical_boxscore_game ON historical_boxscore "
    "(game_id, role, available_at)",
    "CREATE INDEX IF NOT EXISTS idx_historical_detail_game ON historical_game_detail "
    "(game_id, available_at)",
)

V5_REQUIRED_COLUMNS = {
    "historical_boxscore": frozenset(
        (
            "observation_id", "game_id", "team_game_id", "team_id", "opponent_team_id",
            "role", "side", "player_id", "identity_status", "display_name", "row_index",
            "stats_json", "raw_json", "quality_json", "source_revision_id",
        )
    ),
    "historical_game_detail": frozenset(
        ("game_id", "metadata_json", "quality_json", "source_revision_id")
    ),
}

V5_REQUIRED_INDEXES = frozenset(("idx_historical_boxscore_game", "idx_historical_detail_game"))
