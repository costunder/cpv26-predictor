from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from cpv26.data import (
    SCHEMA_VERSION,
    AppendValidationError,
    DuckDBStore,
    ReferentialIntegrityError,
    SnapshotBuilder,
    SnapshotManifest,
    install_schema,
)
from cpv26.data.schema import assert_schema_current

UTC = timezone.utc


def _temporal(
    *,
    event_at: datetime,
    available_at: datetime,
    ingested_at: datetime | None = None,
    valid_to: datetime | None = None,
) -> dict[str, datetime | None]:
    return {
        "event_at": event_at,
        "available_at": available_at,
        "ingested_at": ingested_at or available_at,
        "valid_from": event_at,
        "valid_to": valid_to,
    }


def test_as_of_selects_latest_known_revision_and_excludes_later_publication() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    with DuckDBStore() as store:
        common = {
            "player_id": "player-7",
            "birth_date": None,
            "bats": "L",
            "throws": "R",
            "primary_position": "OF",
            "debut_year": 2021,
            "source_revision_id": "source-1",
            "event_at": cutoff - timedelta(days=30),
            "valid_from": cutoff - timedelta(days=30),
            "valid_to": None,
        }
        store.append(
            "player",
            [
                {
                    **common,
                    "player_row_id": "player-row-original",
                    "display_name": "original",
                    "available_at": cutoff - timedelta(days=3),
                    "ingested_at": cutoff - timedelta(days=3),
                },
                {
                    **common,
                    "player_row_id": "player-row-known-correction",
                    "display_name": "known correction",
                    "available_at": cutoff - timedelta(hours=2),
                    "ingested_at": cutoff - timedelta(hours=1),
                },
                {
                    **common,
                    "player_row_id": "player-row-late-correction",
                    "display_name": "late correction",
                    "available_at": cutoff + timedelta(minutes=1),
                    "ingested_at": cutoff + timedelta(minutes=2),
                },
            ],
        )

        rows = store.fetch_as_of(
            "player",
            cutoff_at=cutoff,
            knowledge_at=cutoff + timedelta(days=1),
            filters={"player_id": "player-7"},
        )

        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (3,)
        assert len(rows) == 1
        assert rows[0]["player_row_id"] == "player-row-known-correction"
        assert rows[0]["display_name"] == "known correction"
        assert rows[0]["available_at"] < cutoff


def test_current_as_of_selects_valid_business_version_before_revision_ranking() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    current_valid_from = cutoff - timedelta(days=30)
    with DuckDBStore() as store:
        common = {
            "player_id": "player-future-version",
            "birth_date": None,
            "bats": "L",
            "throws": "R",
            "primary_position": "OF",
            "debut_year": 2021,
            "source_revision_id": "source-1",
            "valid_to": None,
        }
        store.append(
            "player",
            [
                {
                    **common,
                    "player_row_id": "player-current-version",
                    "display_name": "current",
                    "event_at": current_valid_from,
                    "available_at": cutoff - timedelta(days=10),
                    "ingested_at": cutoff - timedelta(days=10),
                    "valid_from": current_valid_from,
                },
                {
                    **common,
                    "player_row_id": "player-future-version",
                    "display_name": "future",
                    "event_at": cutoff + timedelta(days=1),
                    "available_at": cutoff - timedelta(hours=1),
                    "ingested_at": cutoff - timedelta(minutes=30),
                    "valid_from": cutoff + timedelta(days=1),
                },
            ],
        )

        rows = store.fetch_as_of(
            "player",
            cutoff_at=cutoff,
            knowledge_at=cutoff,
            current_only=True,
        )

    assert len(rows) == 1
    assert rows[0]["player_row_id"] == "player-current-version"


def test_current_as_of_does_not_resurrect_revision_closed_by_correction() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    valid_from = cutoff - timedelta(days=30)
    with DuckDBStore() as store:
        common = {
            "player_id": "player-closed-version",
            "display_name": "closed",
            "birth_date": None,
            "bats": "L",
            "throws": "R",
            "primary_position": "OF",
            "debut_year": 2021,
            "source_revision_id": "source-1",
            "event_at": valid_from,
            "valid_from": valid_from,
        }
        store.append(
            "player",
            [
                {
                    **common,
                    "player_row_id": "player-open-revision",
                    "available_at": cutoff - timedelta(days=3),
                    "ingested_at": cutoff - timedelta(days=3),
                    "valid_to": None,
                },
                {
                    **common,
                    "player_row_id": "player-closing-correction",
                    "available_at": cutoff - timedelta(days=2),
                    "ingested_at": cutoff - timedelta(days=2),
                    "valid_to": cutoff - timedelta(days=1),
                },
            ],
        )

        rows = store.fetch_as_of(
            "player",
            cutoff_at=cutoff,
            knowledge_at=cutoff,
            current_only=True,
        )

    assert rows == []


def test_observed_filter_does_not_resurrect_superseded_future_event() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    common = {
        "team_game_id": "team-game-1",
        "game_id": "game-1",
        "team_id": "team-a",
        "opponent_team_id": "team-b",
        "is_home": True,
        "runs": 2,
        "hits": 5,
        "errors": 0,
        "result": "loss",
        "source_revision_id": "source-1",
        "valid_to": None,
    }
    with DuckDBStore() as store:
        store.append(
            "team_game",
            [
                {
                    **common,
                    "team_game_row_id": "team-game-observed",
                    "event_at": cutoff - timedelta(days=1),
                    "available_at": cutoff - timedelta(hours=20),
                    "ingested_at": cutoff - timedelta(hours=20),
                    "valid_from": cutoff - timedelta(days=1),
                },
                {
                    **common,
                    "team_game_row_id": "team-game-future-correction",
                    "event_at": cutoff + timedelta(days=1),
                    "available_at": cutoff - timedelta(hours=1),
                    "ingested_at": cutoff - timedelta(minutes=30),
                    "valid_from": cutoff + timedelta(days=1),
                },
            ],
        )

        rows = store.fetch_as_of(
            "team_game",
            cutoff_at=cutoff,
            knowledge_at=cutoff,
            observed_before_cutoff=True,
        )

    assert rows == []


def test_snapshot_manifest_is_deterministic_for_identical_run(
    tmp_path: Path,
) -> None:
    cutoff = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)
    knowledge_at = cutoff + timedelta(minutes=5)
    source_time = cutoff - timedelta(days=2)
    with DuckDBStore() as store:
        store.append(
            "source_revision",
            {
                "source_revision_id": "source-1",
                "source_name": "licensed-test-feed",
                "source_locator": None,
                "content_sha256": "a" * 64,
                "metadata_json": {"fixture": "point-in-time"},
                **_temporal(event_at=source_time, available_at=source_time),
            },
        )
        store.append(
            "prediction_run",
            {
                "prediction_run_row_id": "run-row-1",
                "prediction_run_id": "run-20260401",
                "target_game_id": "game-target",
                "cutoff_at": cutoff,
                "knowledge_at": knowledge_at,
                "horizon_type": "lineup_known",
                "feature_version": "1",
                "model_name": "manifest-test",
                "model_version": "1",
                "simulator_version": "1",
                "v26_rule_version": "v26-2026-04",
                "feature_fingerprint": None,
                "config_json": {"seed": 2026},
                **_temporal(
                    event_at=cutoff,
                    available_at=knowledge_at,
                    ingested_at=knowledge_at,
                ),
            },
        )
        store.append(
            "prediction_run_status_event",
            {
                "prediction_run_status_event_id": "run-status-1",
                "prediction_run_id": "run-20260401",
                "status": "created",
                "detail_json": {},
                **_temporal(
                    event_at=knowledge_at,
                    available_at=knowledge_at,
                    ingested_at=knowledge_at,
                ),
            },
        )
        store.append(
            "observed_plate_appearance",
            {
                "observed_pa_row_id": "pa-row-1",
                "plate_appearance_id": "pa-1",
                "game_id": "game-prior",
                "inning": 1,
                "half_inning": "top",
                "sequence_in_game": 1,
                "batter_id": "batter-1",
                "pitcher_id": "pitcher-1",
                "catcher_id": "catcher-1",
                "batting_team_id": "team-a",
                "fielding_team_id": "team-b",
                "outs_before": 0,
                "runners_before": "000",
                "outcome": "1B",
                "is_at_bat": True,
                "is_hit": True,
                "total_bases": 1,
                "runs_scored": 0,
                "source_revision_id": "source-1",
                **_temporal(
                    event_at=cutoff - timedelta(days=1),
                    available_at=cutoff - timedelta(hours=23),
                ),
            },
        )

        builder = SnapshotBuilder(store, tmp_path)
        first = builder.build("run-20260401", knowledge_at=knowledge_at)
        second = builder.build("run-20260401", knowledge_at=knowledge_at)

    snapshot_directory = tmp_path / "run-20260401"
    loaded = SnapshotManifest.load(snapshot_directory / "manifest.json")
    observed = next(
        artifact
        for artifact in loaded.artifacts
        if artifact.table == "observed_plate_appearance"
    )

    assert first.fingerprint == second.fingerprint == loaded.fingerprint
    assert observed.row_count == 1
    assert all(artifact.table != "prediction_run_status_event" for artifact in loaded.artifacts)
    loaded.verify(snapshot_directory)


def test_v1_database_migrates_run_and_status_through_v2_v3_to_v4(tmp_path: Path) -> None:
    database = tmp_path / "v1.duckdb"
    _create_v1_database(database, duplicate_run=False)

    with DuckDBStore(database) as store:
        columns = set(store.table_columns("prediction_run"))
        run = store.latest_prediction_run(
            "legacy-run",
            knowledge_at=datetime(2026, 4, 1, 8, 10, tzinfo=UTC),
        )
        status_rows = store.fetch_as_of(
            "prediction_run_status_event",
            cutoff_at=datetime(2026, 4, 1, 8, 10, tzinfo=UTC),
            knowledge_at=datetime(2026, 4, 1, 8, 10, tzinfo=UTC),
            filters={"prediction_run_id": "legacy-run"},
        )
        version = store.connection.execute(
            "SELECT max(schema_version) FROM schema_migration"
        ).fetchone()
        migration_versions = store.connection.execute(
            "SELECT schema_version FROM schema_migration ORDER BY schema_version"
        ).fetchall()

        assert version == (SCHEMA_VERSION,)
        assert migration_versions == [(1,), (2,), (3,), (4,), (5,)]
        assert "status" not in columns
        assert run["target_game_id"] == "legacy-game"
        assert len(status_rows) == 1
        assert status_rows[0]["status"] == "created"
        migrated_indexes = {
            row[0]
            for row in store.connection.execute(
                "SELECT index_name FROM duckdb_indexes() "
                "WHERE table_name = 'prediction_run'"
            ).fetchall()
        }
        assert "idx_prediction_run_id" in migrated_indexes

        duplicate = dict(run)
        duplicate["prediction_run_row_id"] = "replacement-row"
        with pytest.raises(AppendValidationError, match="immutable and already exists"):
            store.append("prediction_run", duplicate)


def test_v1_migration_rejects_duplicate_prediction_run_ids_transactionally(
    tmp_path: Path,
) -> None:
    database = tmp_path / "duplicate-v1.duckdb"
    _create_v1_database(database, duplicate_run=True)
    connection = duckdb.connect(str(database))
    try:
        with pytest.raises(RuntimeError, match="duplicates found: legacy-run"):
            install_schema(connection)
        version = connection.execute(
            "SELECT max(schema_version) FROM schema_migration"
        ).fetchone()
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        run_count = connection.execute("SELECT count(*) FROM prediction_run").fetchone()
    finally:
        connection.close()

    assert version == (1,)
    assert run_count == (2,)
    assert "prediction_run_v1_backup" not in tables
    assert "prediction_run_status_event" not in tables


def test_append_batch_is_atomic_when_later_row_violates_constraint() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    common = {
        "player_id": "player-atomic",
        "display_name": "Atomic Player",
        "birth_date": None,
        "throws": "R",
        "primary_position": "OF",
        "debut_year": 2021,
        "source_revision_id": "source-1",
        **_temporal(event_at=cutoff, available_at=cutoff),
    }
    with DuckDBStore() as store:
        with pytest.raises(duckdb.ConstraintException):
            store.append(
                "player",
                [
                    {**common, "player_row_id": "atomic-1", "bats": "L"},
                    {**common, "player_row_id": "atomic-2", "bats": "X"},
                ],
            )

        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (0,)


def test_failed_append_marks_explicit_transaction_for_rollback() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    common = {
        "display_name": "Transactional Player",
        "birth_date": None,
        "throws": "R",
        "primary_position": "OF",
        "debut_year": 2021,
        "source_revision_id": "source-1",
        **_temporal(event_at=cutoff, available_at=cutoff),
    }
    with DuckDBStore() as store:
        with pytest.raises(RuntimeError, match="failed append"), store.transaction():
            store.append(
                "player",
                {
                    **common,
                    "player_row_id": "outer-valid",
                    "player_id": "outer-valid",
                    "bats": "L",
                },
            )
            with pytest.raises(duckdb.ConstraintException):
                store.append(
                    "player",
                    {
                        **common,
                        "player_row_id": "outer-invalid",
                        "player_id": "outer-invalid",
                        "bats": "X",
                    },
                )

        assert store.connection.execute("SELECT count(*) FROM player").fetchone() == (0,)


def test_candidate_as_of_is_isolated_by_prediction_run() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    common = {
        "candidate_id": "candidate-shared",
        "game_id": "game-1",
        "player_id": "player-1",
        "team_id": "team-a",
        "opponent_team_id": "team-b",
        "role": "batter",
        "lineup_slot": 1,
        "fielding_position": "OF",
        "start_probability": 0.9,
        "expected_plate_appearances": 4.2,
        "scenario_weight": 1.0,
        "scenario_id": "base",
        "event_at": cutoff,
        "valid_from": cutoff,
        "valid_to": None,
    }
    with DuckDBStore() as store:
        store.append(
            "player_game_candidate",
            [
                {
                    **common,
                    "candidate_row_id": "candidate-run-1",
                    "prediction_run_id": "run-1",
                    "available_at": cutoff,
                    "ingested_at": cutoff,
                },
                {
                    **common,
                    "candidate_row_id": "candidate-run-2",
                    "prediction_run_id": "run-2",
                    "available_at": cutoff + timedelta(minutes=1),
                    "ingested_at": cutoff + timedelta(minutes=1),
                },
            ],
        )
        rows = store.fetch_as_of(
            "player_game_candidate",
            cutoff_at=cutoff + timedelta(minutes=2),
            knowledge_at=cutoff + timedelta(minutes=2),
            filters={"prediction_run_id": "run-1"},
        )

    assert [row["candidate_row_id"] for row in rows] == ["candidate-run-1"]


def test_schema_check_rejects_incomplete_status_event_contract() -> None:
    connection = duckdb.connect(":memory:")
    try:
        install_schema(connection)
        connection.execute("DROP INDEX idx_prediction_run_status")
        connection.execute("ALTER TABLE prediction_run_status_event RENAME TO status_backup")
        connection.execute(
            """
            CREATE TABLE prediction_run_status_event (
                prediction_run_status_event_id VARCHAR PRIMARY KEY,
                event_at TIMESTAMPTZ NOT NULL,
                available_at TIMESTAMPTZ NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL,
                valid_from TIMESTAMPTZ NOT NULL,
                valid_to TIMESTAMPTZ
            )
            """
        )

        with pytest.raises(RuntimeError, match="prediction_run_id, status"):
            assert_schema_current(connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("table", "index_name", "replacement_ddl", "missing_column"),
    [
        (
            "v26_live_hit_rule_set",
            "idx_v26_live_hit_rule_asof",
            """
            CREATE TABLE v26_live_hit_rule_set (
                v26_live_hit_rule_set_row_id VARCHAR PRIMARY KEY,
                rule_version VARCHAR NOT NULL,
                event_at TIMESTAMPTZ NOT NULL,
                available_at TIMESTAMPTZ NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL,
                valid_from TIMESTAMPTZ NOT NULL,
                valid_to TIMESTAMPTZ
            )
            """,
            "rule_payload_json",
        ),
        (
            "stadium",
            "idx_stadium_asof",
            """
            CREATE TABLE stadium (
                stadium_row_id VARCHAR PRIMARY KEY,
                stadium_id VARCHAR NOT NULL,
                event_at TIMESTAMPTZ NOT NULL,
                available_at TIMESTAMPTZ NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL,
                valid_from TIMESTAMPTZ NOT NULL,
                valid_to TIMESTAMPTZ
            )
            """,
            "stadium_name",
        ),
    ],
)
def test_schema_check_rejects_incomplete_versioned_table_contracts(
    table: str,
    index_name: str,
    replacement_ddl: str,
    missing_column: str,
) -> None:
    connection = duckdb.connect(":memory:")
    try:
        install_schema(connection)
        connection.execute(f'DROP INDEX "{index_name}"')
        connection.execute(f'ALTER TABLE "{table}" RENAME TO "{table}_backup"')
        connection.execute(replacement_ddl)

        with pytest.raises(RuntimeError, match=missing_column):
            assert_schema_current(connection)
    finally:
        connection.close()


def test_reference_audit_reports_and_clears_missing_parent() -> None:
    cutoff = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    with DuckDBStore() as store:
        store.append(
            "player",
            {
                "player_row_id": "orphan-player-row",
                "player_id": "orphan-player",
                "display_name": "Orphan Player",
                "birth_date": None,
                "bats": "L",
                "throws": "R",
                "primary_position": "OF",
                "debut_year": 2021,
                "source_revision_id": "missing-source",
                **_temporal(event_at=cutoff, available_at=cutoff),
            },
        )

        violations = store.reference_violations(sample_limit=3)
        source_violation = next(
            violation
            for violation in violations
            if violation.rule.child_table == "player"
            and violation.rule.child_column == "source_revision_id"
        )
        assert source_violation.missing_value_count == 1
        assert source_violation.sample_values == ("missing-source",)
        with pytest.raises(ReferentialIntegrityError, match="player.source_revision_id"):
            store.assert_referential_integrity()

        store.append(
            "source_revision",
            {
                "source_revision_id": "missing-source",
                "source_name": "late-staged-source",
                "source_locator": None,
                "content_sha256": "f" * 64,
                "metadata_json": {},
                **_temporal(event_at=cutoff, available_at=cutoff),
            },
        )

        assert store.reference_violations() == ()


def _create_v1_database(path: Path, *, duplicate_run: bool) -> None:
    connection = duckdb.connect(str(path))
    cutoff = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)
    knowledge_at = cutoff + timedelta(minutes=5)
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
        connection.execute(
            "INSERT INTO schema_migration VALUES (1, ?, 'legacy schema')",
            [cutoff],
        )
        connection.execute(
            """
            CREATE TABLE prediction_run (
                prediction_run_row_id VARCHAR PRIMARY KEY,
                prediction_run_id VARCHAR NOT NULL,
                target_game_id VARCHAR NOT NULL,
                cutoff_at TIMESTAMPTZ NOT NULL,
                knowledge_at TIMESTAMPTZ NOT NULL,
                horizon_type VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL,
                model_name VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                simulator_version VARCHAR NOT NULL,
                v26_rule_version VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                feature_fingerprint VARCHAR,
                config_json VARCHAR NOT NULL,
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
            CREATE INDEX idx_prediction_run_id
            ON prediction_run (prediction_run_id, available_at, ingested_at)
            """
        )
        row_count = 2 if duplicate_run else 1
        for index in range(row_count):
            connection.execute(
                """
                INSERT INTO prediction_run VALUES (
                    ?, 'legacy-run', 'legacy-game', ?, ?, 'lineup_known',
                    'feature-v1', 'model', 'model-v1', 'sim-v1', 'rules-v1',
                    'created', NULL, '{}', ?, ?, ?, ?, NULL
                )
                """,
                [
                    f"legacy-row-{index}",
                    cutoff,
                    knowledge_at,
                    cutoff,
                    knowledge_at,
                    knowledge_at,
                    cutoff,
                ],
            )
    finally:
        connection.close()
