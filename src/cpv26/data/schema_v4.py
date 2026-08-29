"""Schema-v4 additions for replay-complete baseball and V26 datasets.

This module deliberately contains no migration orchestration.  ``schema.py``
owns the transaction and applies these idempotent statements after the v3
tables exist.
"""

from __future__ import annotations

V4_TABLE_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("stadium", "stadium_row_id", ("stadium_id",)),
    (
        "game_status_snapshot",
        "game_status_snapshot_row_id",
        ("game_id",),
    ),
    (
        "starter_announcement",
        "starter_announcement_row_id",
        ("game_id", "team_id"),
    ),
    (
        "player_game_batting",
        "player_game_batting_row_id",
        ("game_id", "team_id", "player_id"),
    ),
    (
        "substitution_event",
        "substitution_event_row_id",
        ("substitution_event_id",),
    ),
    ("runner_event", "runner_event_row_id", ("runner_event_id",)),
    (
        "fielding_assignment",
        "fielding_assignment_row_id",
        ("fielding_assignment_id",),
    ),
    (
        "catcher_assignment",
        "catcher_assignment_row_id",
        ("catcher_assignment_id",),
    ),
    (
        "weather_station_version",
        "weather_station_version_row_id",
        ("station_id",),
    ),
    (
        "stadium_weather_station_map",
        "stadium_weather_station_map_row_id",
        ("stadium_id", "station_id", "map_purpose"),
    ),
    (
        "weather_forecast_snapshot",
        "weather_forecast_snapshot_row_id",
        ("provider", "stadium_id", "forecast_target_at"),
    ),
    (
        "weather_observation",
        "weather_observation_row_id",
        ("observation_source", "station_id", "observed_at"),
    ),
    ("v26_slate", "v26_slate_row_id", ("slate_id",)),
    (
        "v26_submission",
        "v26_submission_row_id",
        ("submission_id", "position"),
    ),
)

V4_COLUMN_UPGRADE_DDL: tuple[str, ...] = (
    "ALTER TABLE game ADD COLUMN IF NOT EXISTS doubleheader_number INTEGER",
    "ALTER TABLE game ADD COLUMN IF NOT EXISTS resumed_from_game_id VARCHAR",
    "ALTER TABLE lineup_version ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ",
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS home_score_before INTEGER"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS away_score_before INTEGER"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS outs_added INTEGER"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS event_subsequence INTEGER DEFAULT 0"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS runners_after VARCHAR"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS home_score_after INTEGER"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS away_score_after INTEGER"
    ),
    (
        "ALTER TABLE observed_plate_appearance "
        "ADD COLUMN IF NOT EXISTS transition_complete BOOLEAN DEFAULT FALSE"
    ),
    (
        "ALTER TABLE v26_player_position_eligibility "
        "ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ"
    ),
    (
        "UPDATE v26_player_position_eligibility "
        "SET captured_at = event_at WHERE captured_at IS NULL"
    ),
    (
        "ALTER TABLE v26_selection_snapshot "
        "ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ"
    ),
    (
        "ALTER TABLE v26_selection_snapshot "
        "ADD COLUMN IF NOT EXISTS capture_phase VARCHAR DEFAULT 'unspecified'"
    ),
    (
        "UPDATE v26_selection_snapshot "
        "SET captured_at = event_at WHERE captured_at IS NULL"
    ),
    (
        "ALTER TABLE user_collection_snapshot "
        "ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ"
    ),
    (
        "UPDATE user_collection_snapshot "
        "SET captured_at = event_at WHERE captured_at IS NULL"
    ),
)

V4_TABLE_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS stadium (
        stadium_row_id VARCHAR PRIMARY KEY,
        stadium_id VARCHAR NOT NULL,
        stadium_name VARCHAR NOT NULL,
        city VARCHAR,
        latitude DOUBLE,
        longitude DOUBLE,
        timezone_name VARCHAR NOT NULL DEFAULT 'Asia/Seoul',
        is_domed BOOLEAN NOT NULL DEFAULT FALSE,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (latitude IS NULL OR latitude BETWEEN -90.0 AND 90.0),
        CHECK (longitude IS NULL OR longitude BETWEEN -180.0 AND 180.0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS game_status_snapshot (
        game_status_snapshot_row_id VARCHAR PRIMARY KEY,
        game_status_snapshot_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        scheduled_start TIMESTAMPTZ NOT NULL,
        status_reason VARCHAR,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (status IN (
            'scheduled', 'delayed', 'in_progress', 'final', 'cancelled',
            'postponed', 'suspended', 'no_result'
        )),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS starter_announcement (
        starter_announcement_row_id VARCHAR PRIMARY KEY,
        starter_announcement_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        pitcher_id VARCHAR NOT NULL,
        announcement_status VARCHAR NOT NULL,
        announced_at TIMESTAMPTZ NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (announcement_status IN (
            'projected', 'announced', 'confirmed', 'changed', 'scratched'
        )),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_game_batting (
        player_game_batting_row_id VARCHAR PRIMARY KEY,
        player_game_batting_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        started BOOLEAN NOT NULL,
        batting_order INTEGER,
        plate_appearances INTEGER NOT NULL,
        at_bats INTEGER NOT NULL,
        hits INTEGER NOT NULL,
        singles INTEGER NOT NULL,
        doubles INTEGER NOT NULL,
        triples INTEGER NOT NULL,
        home_runs INTEGER NOT NULL,
        walks INTEGER NOT NULL,
        hit_by_pitch INTEGER NOT NULL DEFAULT 0,
        strikeouts INTEGER NOT NULL,
        runs INTEGER NOT NULL DEFAULT 0,
        runs_batted_in INTEGER NOT NULL DEFAULT 0,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (batting_order IS NULL OR batting_order BETWEEN 1 AND 9),
        CHECK (plate_appearances >= 0),
        CHECK (at_bats BETWEEN 0 AND plate_appearances),
        CHECK (hits BETWEEN 0 AND at_bats),
        CHECK (hits = singles + doubles + triples + home_runs),
        CHECK (singles >= 0 AND doubles >= 0 AND triples >= 0 AND home_runs >= 0),
        CHECK (walks >= 0 AND hit_by_pitch >= 0 AND strikeouts >= 0),
        CHECK (runs >= 0 AND runs_batted_in >= 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS substitution_event (
        substitution_event_row_id VARCHAR PRIMARY KEY,
        substitution_event_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        sequence_in_game INTEGER NOT NULL,
        event_subsequence INTEGER NOT NULL DEFAULT 0,
        inning INTEGER NOT NULL,
        half_inning VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        outgoing_player_id VARCHAR,
        incoming_player_id VARCHAR,
        substitution_role VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (sequence_in_game >= 1),
        CHECK (event_subsequence >= 0),
        CHECK (inning >= 1),
        CHECK (half_inning IN ('top', 'bottom')),
        CHECK (outgoing_player_id IS NOT NULL OR incoming_player_id IS NOT NULL),
        CHECK (
            outgoing_player_id IS NULL OR incoming_player_id IS NULL
            OR outgoing_player_id <> incoming_player_id
        ),
        CHECK (substitution_role IN (
            'pinch_hitter', 'pinch_runner', 'pitcher', 'defense', 'catcher', 'other'
        )),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runner_event (
        runner_event_row_id VARCHAR PRIMARY KEY,
        runner_event_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        sequence_in_game INTEGER NOT NULL,
        event_subsequence INTEGER NOT NULL DEFAULT 0,
        inning INTEGER NOT NULL,
        half_inning VARCHAR NOT NULL,
        batting_team_id VARCHAR NOT NULL,
        fielding_team_id VARCHAR NOT NULL,
        runner_id VARCHAR NOT NULL,
        pitcher_id VARCHAR,
        catcher_id VARCHAR,
        event_type VARCHAR NOT NULL,
        base_before INTEGER NOT NULL,
        base_after INTEGER,
        outs_added INTEGER NOT NULL,
        runs_scored INTEGER NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (sequence_in_game >= 1),
        CHECK (event_subsequence >= 0),
        CHECK (inning >= 1),
        CHECK (half_inning IN ('top', 'bottom')),
        CHECK (batting_team_id <> fielding_team_id),
        CHECK (base_before BETWEEN 1 AND 3),
        CHECK (base_after IS NULL OR base_after BETWEEN 1 AND 4),
        CHECK (outs_added BETWEEN 0 AND 1),
        CHECK (runs_scored BETWEEN 0 AND 1),
        CHECK (event_type IN (
            'stolen_base', 'caught_stealing', 'pickoff', 'wild_pitch',
            'passed_ball', 'balk', 'advance', 'defensive_indifference',
            'pinch_runner', 'other'
        )),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fielding_assignment (
        fielding_assignment_row_id VARCHAR PRIMARY KEY,
        fielding_assignment_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        fielding_position VARCHAR NOT NULL,
        sequence_start INTEGER NOT NULL,
        sequence_end INTEGER,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (sequence_start >= 1),
        CHECK (sequence_end IS NULL OR sequence_end >= sequence_start),
        CHECK (length(trim(fielding_position)) > 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS catcher_assignment (
        catcher_assignment_row_id VARCHAR PRIMARY KEY,
        catcher_assignment_id VARCHAR NOT NULL,
        game_id VARCHAR NOT NULL,
        team_id VARCHAR NOT NULL,
        catcher_id VARCHAR NOT NULL,
        pitcher_id VARCHAR,
        sequence_start INTEGER NOT NULL,
        sequence_end INTEGER,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (sequence_start >= 1),
        CHECK (sequence_end IS NULL OR sequence_end >= sequence_start),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_station_version (
        weather_station_version_row_id VARCHAR PRIMARY KEY,
        weather_station_version_id VARCHAR NOT NULL,
        station_id VARCHAR NOT NULL,
        station_name VARCHAR NOT NULL,
        station_network VARCHAR NOT NULL,
        latitude DOUBLE NOT NULL,
        longitude DOUBLE NOT NULL,
        elevation_m DOUBLE,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (station_network IN ('ASOS', 'AWS', 'OTHER')),
        CHECK (latitude BETWEEN -90.0 AND 90.0),
        CHECK (longitude BETWEEN -180.0 AND 180.0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stadium_weather_station_map (
        stadium_weather_station_map_row_id VARCHAR PRIMARY KEY,
        stadium_weather_station_map_id VARCHAR NOT NULL,
        stadium_id VARCHAR NOT NULL,
        station_id VARCHAR NOT NULL,
        map_purpose VARCHAR NOT NULL DEFAULT 'observation',
        distance_km DOUBLE NOT NULL,
        is_primary BOOLEAN NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (distance_km >= 0.0),
        CHECK (map_purpose IN ('forecast', 'observation')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_forecast_snapshot (
        weather_forecast_snapshot_row_id VARCHAR PRIMARY KEY,
        forecast_snapshot_id VARCHAR NOT NULL,
        stadium_id VARCHAR NOT NULL,
        provider VARCHAR NOT NULL,
        grid_x INTEGER,
        grid_y INTEGER,
        forecast_issued_at TIMESTAMPTZ NOT NULL,
        forecast_target_at TIMESTAMPTZ NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        temperature_c DOUBLE,
        humidity_pct DOUBLE,
        wind_speed_mps DOUBLE,
        wind_direction_deg DOUBLE,
        precipitation_probability DOUBLE,
        precipitation_type VARCHAR,
        precipitation_amount_mm DOUBLE,
        raw_response_sha256 VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (forecast_target_at >= forecast_issued_at),
        CHECK (captured_at >= forecast_issued_at),
        CHECK (humidity_pct IS NULL OR humidity_pct BETWEEN 0.0 AND 100.0),
        CHECK (wind_speed_mps IS NULL OR wind_speed_mps >= 0.0),
        CHECK (
            wind_direction_deg IS NULL OR wind_direction_deg BETWEEN 0.0 AND 360.0
        ),
        CHECK (
            precipitation_probability IS NULL
            OR precipitation_probability BETWEEN 0.0 AND 1.0
        ),
        CHECK (precipitation_amount_mm IS NULL OR precipitation_amount_mm >= 0.0),
        CHECK (length(raw_response_sha256) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_observation (
        weather_observation_row_id VARCHAR PRIMARY KEY,
        weather_observation_id VARCHAR NOT NULL,
        stadium_id VARCHAR NOT NULL,
        station_id VARCHAR NOT NULL,
        observation_source VARCHAR NOT NULL,
        observed_at TIMESTAMPTZ NOT NULL,
        temperature_c DOUBLE,
        humidity_pct DOUBLE,
        wind_speed_mps DOUBLE,
        wind_direction_deg DOUBLE,
        precipitation_amount_mm DOUBLE,
        raw_response_sha256 VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (humidity_pct IS NULL OR humidity_pct BETWEEN 0.0 AND 100.0),
        CHECK (wind_speed_mps IS NULL OR wind_speed_mps >= 0.0),
        CHECK (
            wind_direction_deg IS NULL OR wind_direction_deg BETWEEN 0.0 AND 360.0
        ),
        CHECK (precipitation_amount_mm IS NULL OR precipitation_amount_mm >= 0.0),
        CHECK (length(raw_response_sha256) = 64),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v26_slate (
        v26_slate_row_id VARCHAR PRIMARY KEY,
        slate_id VARCHAR NOT NULL,
        slate_date DATE NOT NULL,
        lock_at TIMESTAMPTZ NOT NULL,
        live_card_version VARCHAR NOT NULL,
        rule_version VARCHAR NOT NULL,
        position_eligibility_snapshot_id VARCHAR NOT NULL,
        slate_status VARCHAR NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (slate_status IN ('scheduled', 'open', 'locked', 'scored', 'cancelled')),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS v26_submission (
        v26_submission_row_id VARCHAR PRIMARY KEY,
        submission_id VARCHAR NOT NULL,
        slate_id VARCHAR NOT NULL,
        user_id VARCHAR NOT NULL,
        position VARCHAR NOT NULL,
        selected_player_id VARCHAR NOT NULL,
        selected_synergy_team_id VARCHAR NOT NULL,
        submitted_at TIMESTAMPTZ NOT NULL,
        source_revision_id VARCHAR NOT NULL,
        event_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        valid_from TIMESTAMPTZ NOT NULL,
        valid_to TIMESTAMPTZ,
        CHECK (length(trim(position)) > 0),
        CHECK (length(trim(user_id)) > 0),
        CHECK (valid_to IS NULL OR valid_to > valid_from)
    )
    """,
)

V4_INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_stadium_asof ON stadium (stadium_id, available_at)",
    (
        "CREATE INDEX IF NOT EXISTS idx_game_status_asof "
        "ON game_status_snapshot (game_id, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_starter_announcement_asof "
        "ON starter_announcement (game_id, team_id, available_at, ingested_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_player_game_batting_game "
        "ON player_game_batting (game_id, team_id, player_id, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_substitution_game_sequence "
        "ON substitution_event (game_id, sequence_in_game, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_runner_game_sequence "
        "ON runner_event (game_id, sequence_in_game, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_fielding_assignment_game "
        "ON fielding_assignment (game_id, team_id, sequence_start, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_catcher_assignment_game "
        "ON catcher_assignment (game_id, team_id, sequence_start, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_weather_station_asof "
        "ON weather_station_version (station_id, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_stadium_station_map_asof "
        "ON stadium_weather_station_map (stadium_id, station_id, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_weather_forecast_asof "
        "ON weather_forecast_snapshot "
        "(stadium_id, forecast_target_at, forecast_issued_at, available_at)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_weather_observation_event "
        "ON weather_observation (stadium_id, observed_at, available_at)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_v26_slate_asof ON v26_slate (slate_id, available_at)",
    (
        "CREATE INDEX IF NOT EXISTS idx_v26_submission_slate "
        "ON v26_submission (slate_id, user_id, submitted_at)"
    ),
)

V4_DDL: tuple[str, ...] = (*V4_TABLE_DDL, *V4_COLUMN_UPGRADE_DDL, *V4_INDEX_DDL)

V4_REQUIRED_INDEXES = frozenset(
    (
        "idx_stadium_asof",
        "idx_game_status_asof",
        "idx_starter_announcement_asof",
        "idx_player_game_batting_game",
        "idx_substitution_game_sequence",
        "idx_runner_game_sequence",
        "idx_fielding_assignment_game",
        "idx_catcher_assignment_game",
        "idx_weather_station_asof",
        "idx_stadium_station_map_asof",
        "idx_weather_forecast_asof",
        "idx_weather_observation_event",
        "idx_v26_slate_asof",
        "idx_v26_submission_slate",
    )
)

V4_EXTRA_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "game": frozenset(("doubleheader_number", "resumed_from_game_id")),
    "lineup_version": frozenset(("published_at",)),
    "observed_plate_appearance": frozenset(
        (
            "home_score_before",
            "away_score_before",
            "outs_added",
            "event_subsequence",
            "runners_after",
            "home_score_after",
            "away_score_after",
            "transition_complete",
        )
    ),
    "v26_player_position_eligibility": frozenset(("captured_at",)),
    "v26_selection_snapshot": frozenset(("captured_at", "capture_phase")),
    "user_collection_snapshot": frozenset(("captured_at",)),
}

V4_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "stadium": frozenset(
        ("stadium_id", "stadium_name", "timezone_name", "source_revision_id")
    ),
    "game_status_snapshot": frozenset(
        (
            "game_status_snapshot_id",
            "game_id",
            "status",
            "scheduled_start",
            "source_revision_id",
        )
    ),
    "starter_announcement": frozenset(
        (
            "starter_announcement_id",
            "game_id",
            "team_id",
            "pitcher_id",
            "announcement_status",
            "announced_at",
            "source_revision_id",
        )
    ),
    "player_game_batting": frozenset(
        (
            "player_game_batting_id",
            "game_id",
            "team_game_id",
            "team_id",
            "player_id",
            "plate_appearances",
            "at_bats",
            "hits",
            "singles",
            "doubles",
            "triples",
            "home_runs",
            "source_revision_id",
        )
    ),
    "substitution_event": frozenset(
        (
            "substitution_event_id",
            "game_id",
            "sequence_in_game",
            "event_subsequence",
            "team_id",
            "substitution_role",
            "source_revision_id",
        )
    ),
    "runner_event": frozenset(
        (
            "runner_event_id",
            "game_id",
            "sequence_in_game",
            "event_subsequence",
            "runner_id",
            "event_type",
            "base_before",
            "outs_added",
            "runs_scored",
            "source_revision_id",
        )
    ),
    "fielding_assignment": frozenset(
        (
            "fielding_assignment_id",
            "game_id",
            "team_id",
            "player_id",
            "fielding_position",
            "sequence_start",
            "source_revision_id",
        )
    ),
    "catcher_assignment": frozenset(
        (
            "catcher_assignment_id",
            "game_id",
            "team_id",
            "catcher_id",
            "sequence_start",
            "source_revision_id",
        )
    ),
    "weather_station_version": frozenset(
        (
            "weather_station_version_id",
            "station_id",
            "station_name",
            "station_network",
            "latitude",
            "longitude",
            "source_revision_id",
        )
    ),
    "stadium_weather_station_map": frozenset(
        (
            "stadium_weather_station_map_id",
            "stadium_id",
            "station_id",
            "map_purpose",
            "distance_km",
            "source_revision_id",
        )
    ),
    "weather_forecast_snapshot": frozenset(
        (
            "forecast_snapshot_id",
            "stadium_id",
            "provider",
            "forecast_issued_at",
            "forecast_target_at",
            "captured_at",
            "raw_response_sha256",
            "source_revision_id",
        )
    ),
    "weather_observation": frozenset(
        (
            "weather_observation_id",
            "stadium_id",
            "station_id",
            "observation_source",
            "observed_at",
            "raw_response_sha256",
            "source_revision_id",
        )
    ),
    "v26_slate": frozenset(
        (
            "slate_id",
            "slate_date",
            "lock_at",
            "live_card_version",
            "rule_version",
            "position_eligibility_snapshot_id",
            "slate_status",
            "source_revision_id",
        )
    ),
    "v26_submission": frozenset(
        (
            "submission_id",
            "slate_id",
            "user_id",
            "position",
            "selected_player_id",
            "selected_synergy_team_id",
            "submitted_at",
            "source_revision_id",
        )
    ),
}

V4_DOMAIN_TIMESTAMP_COLUMNS = frozenset(
    (
        "announced_at",
        "captured_at",
        "cutoff_at",
        "forecast_issued_at",
        "forecast_target_at",
        "knowledge_at",
        "lock_at",
        "observed_at",
        "published_at",
        "scheduled_start",
        "submitted_at",
    )
)

V4_TIMEZONE_COLUMNS: dict[str, frozenset[str]] = {
    "game": frozenset(("scheduled_start",)),
    "game_status_snapshot": frozenset(("scheduled_start",)),
    "starter_announcement": frozenset(("announced_at",)),
    "lineup_version": frozenset(("published_at",)),
    "weather_forecast_snapshot": frozenset(
        ("forecast_issued_at", "forecast_target_at", "captured_at")
    ),
    "weather_observation": frozenset(("observed_at",)),
    "v26_slate": frozenset(("lock_at",)),
    "v26_player_position_eligibility": frozenset(("captured_at",)),
    "v26_selection_snapshot": frozenset(("lock_at", "captured_at")),
    "user_collection_snapshot": frozenset(("captured_at",)),
    "v26_submission": frozenset(("submitted_at",)),
}
