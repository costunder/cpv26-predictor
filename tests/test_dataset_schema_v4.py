from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pytest

from cpv26.data import (
    DEFAULT_TABLE_SPECS,
    SCHEMA_VERSION,
    TABLE_DEFINITIONS,
    DuckDBStore,
    SnapshotBuilder,
    live_hit_snapshot_specs,
)

UTC = timezone.utc

V4_NATURAL_IDENTITIES = {
    "stadium": ("stadium_id",),
    "game_status_snapshot": ("game_id",),
    "starter_announcement": ("game_id", "team_id"),
    "player_game_batting": ("game_id", "team_id", "player_id"),
    "substitution_event": ("substitution_event_id",),
    "runner_event": ("runner_event_id",),
    "fielding_assignment": ("fielding_assignment_id",),
    "catcher_assignment": ("catcher_assignment_id",),
    "weather_station_version": ("station_id",),
    "stadium_weather_station_map": ("stadium_id", "station_id", "map_purpose"),
    "weather_forecast_snapshot": ("provider", "stadium_id", "forecast_target_at"),
    "weather_observation": ("observation_source", "station_id", "observed_at"),
    "v26_slate": ("slate_id",),
    "v26_submission": ("submission_id", "position"),
}

DOMAIN_TIMESTAMPS = {
    "game_status_snapshot": {"scheduled_start"},
    "starter_announcement": {"announced_at"},
    "weather_forecast_snapshot": {
        "forecast_issued_at",
        "forecast_target_at",
        "captured_at",
    },
    "weather_observation": {"observed_at"},
    "v26_slate": {"lock_at"},
    "v26_submission": {"submitted_at"},
}

TEMPORAL_COLUMNS = {
    "event_at",
    "available_at",
    "ingested_at",
    "valid_from",
    "valid_to",
}


def _temporal(
    at: datetime,
    *,
    available_at: datetime | None = None,
    valid_from: datetime | None = None,
) -> dict[str, datetime | None]:
    available = available_at or at
    return {
        "event_at": at,
        "available_at": available,
        "ingested_at": available,
        "valid_from": valid_from or at,
        "valid_to": None,
    }


def _type_map(store: DuckDBStore, table: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2]).upper()
        for row in store.connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    }


def test_schema_v4_fresh_install_has_fourteen_tables_keys_and_timezone_columns() -> None:
    assert SCHEMA_VERSION == 4
    assert len(V4_NATURAL_IDENTITIES) == 14

    with DuckDBStore() as store:
        installed = {
            str(row[0]) for row in store.connection.execute("SHOW TABLES").fetchall()
        }
        migration_versions = store.connection.execute(
            "SELECT schema_version FROM schema_migration ORDER BY schema_version"
        ).fetchall()

        assert set(V4_NATURAL_IDENTITIES) <= installed
        assert migration_versions == [(4,)]
        for table, natural_identity in V4_NATURAL_IDENTITIES.items():
            assert TABLE_DEFINITIONS[table].natural_identity == natural_identity
            column_types = _type_map(store, table)
            for column in TEMPORAL_COLUMNS | DOMAIN_TIMESTAMPS.get(table, set()):
                assert "TIMESTAMP WITH TIME ZONE" in column_types[column]


def test_v3_migration_adds_v4_contract_and_preserves_legacy_rows(tmp_path: Path) -> None:
    database = tmp_path / "schema-v3.duckdb"
    legacy_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    _create_v3_fixture(database, legacy_at)

    with DuckDBStore(database) as store:
        version = store.connection.execute(
            "SELECT max(schema_version) FROM schema_migration"
        ).fetchone()
        pa = store.connection.execute(
            """
            SELECT plate_appearance_id, event_subsequence, transition_complete,
                   home_score_before, away_score_before, outs_added,
                   runners_after, home_score_after, away_score_after
            FROM observed_plate_appearance
            """
        ).fetchone()
        eligibility = store.connection.execute(
            "SELECT player_id, captured_at FROM v26_player_position_eligibility"
        ).fetchone()
        selection = store.connection.execute(
            "SELECT player_id, captured_at, capture_phase FROM v26_selection_snapshot"
        ).fetchone()
        collection = store.connection.execute(
            "SELECT player_id, captured_at FROM user_collection_snapshot"
        ).fetchone()
        migration_versions = store.connection.execute(
            "SELECT schema_version FROM schema_migration ORDER BY schema_version"
        ).fetchall()

        assert version == (4,)
        assert migration_versions == [(3,), (4,)]
        assert pa == ("legacy-pa", 0, False, None, None, None, None, None, None)
        assert eligibility == ("legacy-player", legacy_at)
        assert selection == ("legacy-player", legacy_at, "unspecified")
        assert collection == ("legacy-player", legacy_at)
        assert set(V4_NATURAL_IDENTITIES) <= {
            str(row[0]) for row in store.connection.execute("SHOW TABLES").fetchall()
        }


def test_fresh_pa_transition_requires_post_state_and_canonical_runner_bitmaps() -> None:
    at = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    base = _complete_pa_row(at)

    with DuckDBStore() as store:
        store.append("observed_plate_appearance", base)

        missing_post_state = {
            **base,
            "observed_pa_row_id": "pa-row-missing-post-state",
            "plate_appearance_id": "pa-missing-post-state",
            "runners_after": None,
        }
        with pytest.raises(duckdb.ConstraintException):
            store.append("observed_plate_appearance", missing_post_state)

        invalid_before = {
            **base,
            "observed_pa_row_id": "pa-row-invalid-before",
            "plate_appearance_id": "pa-invalid-before",
            "transition_complete": False,
            "runners_before": "00X",
        }
        with pytest.raises(duckdb.ConstraintException):
            store.append("observed_plate_appearance", invalid_before)

        invalid_after = {
            **base,
            "observed_pa_row_id": "pa-row-invalid-after",
            "plate_appearance_id": "pa-invalid-after",
            "runners_after": "0X0",
        }
        with pytest.raises(duckdb.ConstraintException):
            store.append("observed_plate_appearance", invalid_after)

        assert store.connection.execute(
            "SELECT count(*) FROM observed_plate_appearance"
        ).fetchone() == (1,)


def test_status_starter_and_forecast_point_in_time_exclude_late_publication() -> None:
    first = datetime(2026, 8, 22, 8, 0, tzinfo=UTC)
    cutoff = first + timedelta(hours=2)
    late = cutoff + timedelta(minutes=5)
    after_late = late + timedelta(minutes=1)
    target = first + timedelta(hours=10)

    with DuckDBStore() as store:
        store.append(
            "game_status_snapshot",
            [
                _status_row("status-early", "scheduled", first, first),
                _status_row("status-late", "delayed", first + timedelta(hours=1), late),
            ],
        )
        store.append(
            "starter_announcement",
            [
                _starter_row("starter-early", "pitcher-a", first, first),
                _starter_row(
                    "starter-late",
                    "pitcher-b",
                    first + timedelta(hours=1),
                    late,
                ),
            ],
        )
        store.append(
            "weather_forecast_snapshot",
            [
                _forecast_row("forecast-early", 21.0, first, first, target),
                _forecast_row(
                    "forecast-late",
                    24.0,
                    first + timedelta(hours=1),
                    late,
                    target,
                ),
            ],
        )

        status_before = store.fetch_as_of(
            "game_status_snapshot", cutoff_at=cutoff, current_only=True
        )
        status_after = store.fetch_as_of(
            "game_status_snapshot", cutoff_at=after_late, current_only=True
        )
        starter_before = store.fetch_as_of(
            "starter_announcement", cutoff_at=cutoff, current_only=True
        )
        starter_after = store.fetch_as_of(
            "starter_announcement", cutoff_at=after_late, current_only=True
        )
        forecast_before = store.fetch_as_of(
            "weather_forecast_snapshot", cutoff_at=cutoff, current_only=True
        )
        forecast_after = store.fetch_as_of(
            "weather_forecast_snapshot", cutoff_at=after_late, current_only=True
        )

    assert status_before[0]["status"] == "scheduled"
    assert status_after[0]["status"] == "delayed"
    assert starter_before[0]["pitcher_id"] == "pitcher-a"
    assert starter_after[0]["pitcher_id"] == "pitcher-b"
    assert forecast_before[0]["temperature_c"] == pytest.approx(21.0)
    assert forecast_after[0]["temperature_c"] == pytest.approx(24.0)


def test_forecast_is_input_but_future_observation_is_cutoff_filtered(
    tmp_path: Path,
) -> None:
    cutoff = datetime(2026, 8, 23, 9, 0, tzinfo=UTC)
    target = cutoff + timedelta(hours=5)
    specs_by_name = {spec.name: spec for spec in DEFAULT_TABLE_SPECS}
    forecast_spec = specs_by_name["weather_forecast_snapshot"]
    observation_spec = specs_by_name["weather_observation"]

    assert forecast_spec.current_only is True
    assert forecast_spec.observed_before_cutoff is False
    assert observation_spec.observed_before_cutoff is True

    with DuckDBStore() as store:
        store.append("prediction_run", _prediction_run(cutoff))
        store.append(
            "weather_forecast_snapshot",
            _forecast_row(
                "forecast-input",
                25.0,
                cutoff - timedelta(hours=1),
                cutoff - timedelta(minutes=50),
                target,
            ),
        )
        store.append(
            "weather_observation",
            {
                "weather_observation_row_id": "observation-oracle-row",
                "weather_observation_id": "observation-oracle",
                "stadium_id": "stadium-1",
                "station_id": "station-1",
                "observation_source": "fixture",
                "observed_at": target,
                "temperature_c": 31.0,
                "humidity_pct": 60.0,
                "wind_speed_mps": 2.0,
                "wind_direction_deg": 180.0,
                "precipitation_amount_mm": 0.0,
                "raw_response_sha256": "b" * 64,
                "source_revision_id": "source-weather",
                **_temporal(target, available_at=cutoff - timedelta(minutes=1)),
            },
        )

        manifest = SnapshotBuilder(
            store,
            tmp_path,
            table_specs=(forecast_spec, observation_spec),
        ).build("weather-run", knowledge_at=cutoff)

    artifacts = {artifact.table: artifact for artifact in manifest.artifacts}
    assert artifacts["weather_forecast_snapshot"].row_count == 1
    assert artifacts["weather_observation"].row_count == 0


def test_live_hit_scope_includes_slate_and_excludes_submission_actions() -> None:
    specs = live_hit_snapshot_specs(
        user_id="user-1",
        slate_id="slate-1",
        live_card_version="live-card-1",
        rule_version="rule-1",
        position_eligibility_snapshot_id="eligibility-1",
        selection_snapshot_id="selection-1",
    )
    names = {spec.name for spec in specs}
    default_names = {spec.name for spec in DEFAULT_TABLE_SPECS}

    assert "v26_slate" in names
    assert "v26_submission" not in names
    assert "v26_submission" not in default_names
    slate_spec = next(spec for spec in specs if spec.name == "v26_slate")
    assert dict(slate_spec.filters) == {
        "slate_id": "slate-1",
        "rule_version": "rule-1",
        "live_card_version": "live-card-1",
        "position_eligibility_snapshot_id": "eligibility-1",
    }


def _complete_pa_row(at: datetime) -> dict[str, Any]:
    return {
        "observed_pa_row_id": "pa-row-complete",
        "plate_appearance_id": "pa-complete",
        "game_id": "game-1",
        "inning": 1,
        "half_inning": "top",
        "sequence_in_game": 1,
        "event_subsequence": 0,
        "batter_id": "batter-1",
        "pitcher_id": "pitcher-1",
        "catcher_id": "catcher-1",
        "batting_team_id": "team-a",
        "fielding_team_id": "team-b",
        "home_score_before": 0,
        "away_score_before": 0,
        "outs_before": 0,
        "runners_before": "000",
        "outs_added": 0,
        "runners_after": "100",
        "home_score_after": 0,
        "away_score_after": 0,
        "transition_complete": True,
        "outcome": "1B",
        "is_at_bat": True,
        "is_hit": True,
        "total_bases": 1,
        "runs_scored": 0,
        "source_revision_id": "source-pa",
        **_temporal(at),
    }


def _status_row(
    row_id: str,
    status: str,
    effective_at: datetime,
    available_at: datetime,
) -> dict[str, Any]:
    return {
        "game_status_snapshot_row_id": row_id,
        "game_status_snapshot_id": row_id,
        "game_id": "game-1",
        "status": status,
        "scheduled_start": effective_at + timedelta(hours=8),
        "status_reason": None,
        "source_revision_id": "source-status",
        **_temporal(effective_at, available_at=available_at),
    }


def _starter_row(
    row_id: str,
    pitcher_id: str,
    effective_at: datetime,
    available_at: datetime,
) -> dict[str, Any]:
    return {
        "starter_announcement_row_id": row_id,
        "starter_announcement_id": row_id,
        "game_id": "game-1",
        "team_id": "team-a",
        "pitcher_id": pitcher_id,
        "announcement_status": "announced",
        "announced_at": effective_at,
        "source_revision_id": "source-starter",
        **_temporal(effective_at, available_at=available_at),
    }


def _forecast_row(
    row_id: str,
    temperature_c: float,
    issued_at: datetime,
    available_at: datetime,
    target_at: datetime,
) -> dict[str, Any]:
    return {
        "weather_forecast_snapshot_row_id": row_id,
        "forecast_snapshot_id": row_id,
        "stadium_id": "stadium-1",
        "provider": "fixture",
        "grid_x": 60,
        "grid_y": 127,
        "forecast_issued_at": issued_at,
        "forecast_target_at": target_at,
        "captured_at": available_at,
        "temperature_c": temperature_c,
        "humidity_pct": 50.0,
        "wind_speed_mps": 2.0,
        "wind_direction_deg": 90.0,
        "precipitation_probability": 0.1,
        "precipitation_type": "none",
        "precipitation_amount_mm": 0.0,
        "raw_response_sha256": "a" * 64,
        "source_revision_id": "source-weather",
        **_temporal(issued_at, available_at=available_at),
    }


def _prediction_run(at: datetime) -> dict[str, Any]:
    return {
        "prediction_run_row_id": "weather-run-row",
        "prediction_run_id": "weather-run",
        "target_game_id": "game-1",
        "cutoff_at": at,
        "knowledge_at": at,
        "horizon_type": "early",
        "feature_version": "fixture-v1",
        "model_name": "fixture",
        "model_version": "fixture-v1",
        "simulator_version": "fixture-v1",
        "v26_rule_version": "fixture-v1",
        "feature_fingerprint": None,
        "config_json": {},
        **_temporal(at),
    }


def _create_v3_fixture(path: Path, at: datetime) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE schema_migration (
                schema_version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL,
                description VARCHAR NOT NULL
            )
            """
        )
        connection.execute("INSERT INTO schema_migration VALUES (3, ?, 'schema v3 fixture')", [at])
        connection.execute(
            """
            CREATE TABLE observed_plate_appearance (
                observed_pa_row_id VARCHAR PRIMARY KEY,
                plate_appearance_id VARCHAR NOT NULL,
                game_id VARCHAR NOT NULL,
                inning INTEGER NOT NULL,
                half_inning VARCHAR NOT NULL,
                sequence_in_game INTEGER NOT NULL,
                batter_id VARCHAR NOT NULL,
                pitcher_id VARCHAR NOT NULL,
                catcher_id VARCHAR,
                batting_team_id VARCHAR NOT NULL,
                fielding_team_id VARCHAR NOT NULL,
                outs_before INTEGER NOT NULL,
                runners_before VARCHAR NOT NULL,
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
                valid_to TIMESTAMPTZ
            )
            """
        )
        connection.execute(
            """
            INSERT INTO observed_plate_appearance VALUES (
                'legacy-pa-row', 'legacy-pa', 'legacy-game', 1, 'top', 1,
                'legacy-player', 'legacy-pitcher', NULL, 'legacy-team-a',
                'legacy-team-b', 0, '000', 'OUT', TRUE, FALSE, 0, 0,
                'legacy-source', ?, ?, ?, ?, NULL
            )
            """,
            [at, at, at, at],
        )
        _create_v3_live_hit_tables(connection)
        temporal_values = [at, at, at, at]
        connection.execute(
            """
            INSERT INTO v26_player_position_eligibility VALUES (
                'legacy-eligibility-row', 'legacy-eligibility', 'legacy-slate',
                DATE '2026-08-20', 'legacy-card', 'legacy-player', 'OF', TRUE,
                'legacy-source', ?, ?, ?, ?, NULL
            )
            """,
            temporal_values,
        )
        connection.execute(
            """
            INSERT INTO v26_selection_snapshot VALUES (
                'legacy-selection-row', 'legacy-selection', 'legacy-slate', ?,
                'legacy-player', 'OF', 0.25, 'legacy-rule', 'legacy-source',
                ?, ?, ?, ?, NULL
            )
            """,
            [at + timedelta(hours=2), *temporal_values],
        )
        connection.execute(
            """
            INSERT INTO user_collection_snapshot VALUES (
                'legacy-collection-row', 'legacy-collection', 'legacy-user',
                'legacy-card', 'legacy-player', TRUE, 'legacy-source',
                ?, ?, ?, ?, NULL
            )
            """,
            temporal_values,
        )
    finally:
        connection.close()


def _create_v3_live_hit_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE v26_player_position_eligibility (
            v26_player_position_eligibility_row_id VARCHAR PRIMARY KEY,
            position_eligibility_snapshot_id VARCHAR NOT NULL,
            slate_id VARCHAR NOT NULL,
            slate_date DATE NOT NULL,
            live_card_version VARCHAR NOT NULL,
            player_id VARCHAR NOT NULL,
            position VARCHAR NOT NULL,
            is_eligible BOOLEAN NOT NULL,
            source_revision_id VARCHAR NOT NULL,
            event_at TIMESTAMPTZ NOT NULL,
            available_at TIMESTAMPTZ NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            valid_from TIMESTAMPTZ NOT NULL,
            valid_to TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE v26_selection_snapshot (
            v26_selection_snapshot_row_id VARCHAR PRIMARY KEY,
            selection_snapshot_id VARCHAR NOT NULL,
            slate_id VARCHAR NOT NULL,
            lock_at TIMESTAMPTZ NOT NULL,
            player_id VARCHAR NOT NULL,
            position VARCHAR NOT NULL,
            selection_rate DOUBLE NOT NULL,
            rule_version VARCHAR NOT NULL,
            source_revision_id VARCHAR NOT NULL,
            event_at TIMESTAMPTZ NOT NULL,
            available_at TIMESTAMPTZ NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            valid_from TIMESTAMPTZ NOT NULL,
            valid_to TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE user_collection_snapshot (
            user_collection_snapshot_row_id VARCHAR PRIMARY KEY,
            collection_snapshot_id VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            live_card_version VARCHAR NOT NULL,
            player_id VARCHAR NOT NULL,
            owned BOOLEAN NOT NULL,
            source_revision_id VARCHAR NOT NULL,
            event_at TIMESTAMPTZ NOT NULL,
            available_at TIMESTAMPTZ NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            valid_from TIMESTAMPTZ NOT NULL,
            valid_to TIMESTAMPTZ
        )
        """
    )
