from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from cpv26.data import (
    SCHEMA_VERSION,
    TABLE_DEFINITIONS,
    DuckDBStore,
    SnapshotBuilder,
    SnapshotManifest,
    live_hit_snapshot_specs,
)

UTC = timezone.utc


def _temporal(
    *,
    event_at: datetime,
    available_at: datetime,
    ingested_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> dict[str, datetime | None]:
    return {
        "event_at": event_at,
        "available_at": available_at,
        "ingested_at": ingested_at or available_at,
        "valid_from": valid_from or event_at,
        "valid_to": valid_to,
    }


def _append_source_and_player(
    store: DuckDBStore,
    *,
    at: datetime,
    player_id: str = "player-1",
) -> None:
    store.append(
        "source_revision",
        {
            "source_revision_id": "live-hit-source-1",
            "source_name": "user-supplied-live-hit-fixture",
            "source_locator": None,
            "content_sha256": "a" * 64,
            "metadata_json": {"official": False},
            **_temporal(event_at=at, available_at=at),
        },
    )
    store.append(
        "player",
        {
            "player_row_id": f"{player_id}-row",
            "player_id": player_id,
            "display_name": "Fixture Player",
            "birth_date": None,
            "bats": "L",
            "throws": "R",
            "primary_position": "OF",
            "debut_year": 2022,
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )


def _append_live_hit_inputs(
    store: DuckDBStore,
    *,
    at: datetime,
    user_id: str = "user-a",
) -> None:
    slate_date = date(2026, 8, 29)
    store.append(
        "v26_player_position_eligibility",
        {
            "v26_player_position_eligibility_row_id": "eligibility-row-1",
            "position_eligibility_snapshot_id": "eligibility-snapshot-1",
            "slate_id": "slate-20260829",
            "slate_date": slate_date,
            "live_card_version": "live-2026-week-23",
            "player_id": "player-1",
            "position": "OF",
            "is_eligible": True,
            "captured_at": at,
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )
    store.append(
        "v26_live_hit_rule_set",
        {
            "v26_live_hit_rule_set_row_id": "rule-row-1",
            "rule_version": "user-rule-v1",
            "position_eligibility_snapshot_id": "eligibility-snapshot-1",
            "rule_payload_json": {
                "mode": "weekly_hit_points",
                "hit_points": {"0": 0, "1": 1, "2": 2},
                "percentage_combination": "additive_percentage_points",
            },
            "provenance_kind": "user_supplied",
            "provenance_json": {
                "official": False,
                "note": "test contract, not an asserted V26 official rule",
            },
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )
    store.append(
        "v26_slate",
        {
            "v26_slate_row_id": "slate-row-1",
            "slate_id": "slate-20260829",
            "slate_date": slate_date,
            "lock_at": at + timedelta(hours=2),
            "live_card_version": "live-2026-week-23",
            "rule_version": "user-rule-v1",
            "position_eligibility_snapshot_id": "eligibility-snapshot-1",
            "slate_status": "open",
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )
    store.append(
        "v26_selection_snapshot",
        {
            "v26_selection_snapshot_row_id": "selection-row-1",
            "selection_snapshot_id": "selection-snapshot-1",
            "slate_id": "slate-20260829",
            "lock_at": at + timedelta(hours=2),
            "player_id": "player-1",
            "position": "OF",
            "selection_rate": 0.2,
            "rule_version": "user-rule-v1",
            "captured_at": at,
            "capture_phase": "early",
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )
    store.append(
        "user_collection_snapshot",
        {
            "user_collection_snapshot_row_id": f"collection-row-{user_id}",
            "collection_snapshot_id": f"collection-snapshot-{user_id}",
            "user_id": user_id,
            "live_card_version": "live-2026-week-23",
            "player_id": "player-1",
            "owned": True,
            "captured_at": at,
            "source_revision_id": "live-hit-source-1",
            **_temporal(event_at=at, available_at=at),
        },
    )


def test_live_hit_tables_have_explicit_point_in_time_natural_keys() -> None:
    assert SCHEMA_VERSION == 4
    assert TABLE_DEFINITIONS["v26_live_hit_rule_set"].natural_identity == (
        "rule_version",
    )
    assert TABLE_DEFINITIONS[
        "v26_player_position_eligibility"
    ].natural_identity == (
        "position_eligibility_snapshot_id",
        "slate_id",
        "live_card_version",
        "player_id",
        "position",
    )
    assert TABLE_DEFINITIONS["v26_selection_snapshot"].natural_identity == (
        "selection_snapshot_id",
        "slate_id",
        "player_id",
        "position",
    )
    assert TABLE_DEFINITIONS["user_collection_snapshot"].natural_identity == (
        "user_id",
        "live_card_version",
        "player_id",
    )

    with DuckDBStore() as store:
        for table in (
            "v26_live_hit_rule_set",
            "v26_player_position_eligibility",
            "v26_selection_snapshot",
            "user_collection_snapshot",
            "v26_slate",
        ):
            columns = set(store.table_columns(table))
            assert {
                "event_at",
                "available_at",
                "ingested_at",
                "valid_from",
                "valid_to",
            } <= columns


def test_live_hit_rows_round_trip_and_pass_reference_audit() -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    with DuckDBStore() as store:
        _append_source_and_player(store, at=at)
        _append_live_hit_inputs(store, at=at)

        rules = store.fetch_as_of(
            "v26_live_hit_rule_set",
            cutoff_at=at + timedelta(minutes=1),
            current_only=True,
            filters={"rule_version": "user-rule-v1"},
        )
        eligibility = store.fetch_as_of(
            "v26_player_position_eligibility",
            cutoff_at=at + timedelta(minutes=1),
            current_only=True,
            filters={"position_eligibility_snapshot_id": "eligibility-snapshot-1"},
        )
        selection = store.fetch_as_of(
            "v26_selection_snapshot",
            cutoff_at=at + timedelta(minutes=1),
            current_only=True,
            filters={"selection_snapshot_id": "selection-snapshot-1"},
        )
        collection = store.fetch_as_of(
            "user_collection_snapshot",
            cutoff_at=at + timedelta(minutes=1),
            current_only=True,
            filters={"user_id": "user-a"},
        )

        payload = json.loads(rules[0]["rule_payload_json"])
        provenance = json.loads(rules[0]["provenance_json"])
        assert payload["percentage_combination"] == "additive_percentage_points"
        assert provenance["official"] is False
        assert rules[0]["provenance_kind"] == "user_supplied"
        assert eligibility[0]["slate_date"] == date(2026, 8, 29)
        assert eligibility[0]["is_eligible"] is True
        assert selection[0]["selection_rate"] == pytest.approx(0.2)
        assert selection[0]["lock_at"].tzinfo is not None
        assert collection[0]["owned"] is True
        assert store.reference_violations() == ()


def test_selection_snapshot_as_of_excludes_late_rate_correction() -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    first_cutoff = at + timedelta(minutes=30)
    late_publication = at + timedelta(hours=1)
    with DuckDBStore() as store:
        _append_source_and_player(store, at=at)
        _append_live_hit_inputs(store, at=at)
        common = {
            "selection_snapshot_id": "selection-snapshot-1",
            "slate_id": "slate-20260829",
            "lock_at": at + timedelta(hours=2),
            "player_id": "player-1",
            "position": "OF",
            "rule_version": "user-rule-v1",
            "captured_at": at,
            "capture_phase": "early",
            "source_revision_id": "live-hit-source-1",
            "event_at": at,
            "valid_from": at,
            "valid_to": None,
        }
        store.append(
            "v26_selection_snapshot",
            {
                **common,
                "v26_selection_snapshot_row_id": "selection-row-late-correction",
                "selection_rate": 0.35,
                "available_at": late_publication,
                "ingested_at": late_publication + timedelta(minutes=1),
            },
        )

        before = store.fetch_as_of(
            "v26_selection_snapshot",
            cutoff_at=first_cutoff,
            knowledge_at=first_cutoff,
            filters={"selection_snapshot_id": "selection-snapshot-1"},
        )
        after = store.fetch_as_of(
            "v26_selection_snapshot",
            cutoff_at=late_publication + timedelta(minutes=2),
            knowledge_at=late_publication + timedelta(minutes=2),
            filters={"selection_snapshot_id": "selection-snapshot-1"},
        )

    assert before[0]["selection_rate"] == pytest.approx(0.2)
    assert after[0]["selection_rate"] == pytest.approx(0.35)


def test_collection_snapshot_selects_business_state_at_cutoff() -> None:
    first = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    acquired = first + timedelta(hours=1)
    with DuckDBStore() as store:
        _append_source_and_player(store, at=first)
        common = {
            "user_id": "user-a",
            "live_card_version": "live-2026-week-23",
            "player_id": "player-1",
            "source_revision_id": "live-hit-source-1",
        }
        store.append(
            "user_collection_snapshot",
            [
                {
                    **common,
                    "user_collection_snapshot_row_id": "collection-before",
                    "collection_snapshot_id": "collection-snapshot-before",
                    "owned": False,
                    "captured_at": first,
                    **_temporal(
                        event_at=first,
                        available_at=first,
                        valid_to=acquired,
                    ),
                },
                {
                    **common,
                    "user_collection_snapshot_row_id": "collection-after",
                    "collection_snapshot_id": "collection-snapshot-after",
                    "owned": True,
                    "captured_at": acquired,
                    **_temporal(event_at=acquired, available_at=acquired),
                },
            ],
        )

        before = store.fetch_as_of(
            "user_collection_snapshot",
            cutoff_at=first + timedelta(minutes=30),
            current_only=True,
            filters={"user_id": "user-a"},
        )
        after = store.fetch_as_of(
            "user_collection_snapshot",
            cutoff_at=acquired + timedelta(minutes=30),
            current_only=True,
            filters={"user_id": "user-a"},
        )

    assert before[0]["owned"] is False
    assert after[0]["owned"] is True


def test_live_hit_constraints_reject_invalid_rate_and_unclassified_provenance() -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    with DuckDBStore() as store:
        _append_source_and_player(store, at=at)
        with pytest.raises(duckdb.ConstraintException):
            store.append(
                "v26_selection_snapshot",
                {
                    "v26_selection_snapshot_row_id": "invalid-rate",
                    "selection_snapshot_id": "selection-snapshot-invalid",
                    "slate_id": "slate-20260829",
                    "lock_at": at + timedelta(hours=1),
                    "player_id": "player-1",
                    "position": "OF",
                    "selection_rate": 1.01,
                    "rule_version": "user-rule-v1",
                    "captured_at": at,
                    "capture_phase": "early",
                    "source_revision_id": "live-hit-source-1",
                    **_temporal(event_at=at, available_at=at),
                },
            )
        with pytest.raises(duckdb.ConstraintException):
            store.append(
                "v26_live_hit_rule_set",
                {
                    "v26_live_hit_rule_set_row_id": "invalid-provenance",
                    "rule_version": "unclassified-rule",
                    "position_eligibility_snapshot_id": "eligibility-snapshot-1",
                    "rule_payload_json": {},
                    "provenance_kind": "assumed_official",
                    "provenance_json": {},
                    "source_revision_id": "live-hit-source-1",
                    **_temporal(event_at=at, available_at=at),
                },
            )


def test_live_hit_snapshot_specs_scope_user_and_enter_fingerprint(tmp_path: Path) -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    cutoff = at + timedelta(minutes=30)
    with DuckDBStore() as store:
        _append_source_and_player(store, at=at)
        _append_live_hit_inputs(store, at=at, user_id="user-a")
        store.append(
            "user_collection_snapshot",
            {
                "user_collection_snapshot_row_id": "collection-row-user-b",
                "collection_snapshot_id": "collection-snapshot-user-b",
                "user_id": "user-b",
                "live_card_version": "live-2026-week-23",
                "player_id": "player-1",
                "owned": False,
                "captured_at": at,
                "source_revision_id": "live-hit-source-1",
                **_temporal(event_at=at, available_at=at),
            },
        )
        store.append(
            "prediction_run",
            {
                "prediction_run_row_id": "live-hit-run-row",
                "prediction_run_id": "live-hit-run",
                "target_game_id": "fixture-game",
                "cutoff_at": cutoff,
                "knowledge_at": cutoff,
                "horizon_type": "near_lock",
                "feature_version": "fixture-v1",
                "model_name": "fixture",
                "model_version": "fixture-v1",
                "simulator_version": "fixture-v1",
                "v26_rule_version": "user-rule-v1",
                "feature_fingerprint": None,
                "config_json": {},
                **_temporal(event_at=cutoff, available_at=cutoff),
            },
        )
        specs = live_hit_snapshot_specs(
            user_id="user-a",
            slate_id="slate-20260829",
            live_card_version="live-2026-week-23",
            rule_version="user-rule-v1",
            position_eligibility_snapshot_id="eligibility-snapshot-1",
            selection_snapshot_id="selection-snapshot-1",
        )
        manifest = SnapshotBuilder(store, tmp_path, table_specs=specs).build(
            "live-hit-run",
            knowledge_at=cutoff,
        )

    loaded = SnapshotManifest.load(tmp_path / "live-hit-run" / "manifest.json")
    loaded.verify(tmp_path / "live-hit-run")
    artifacts = {artifact.table: artifact for artifact in manifest.artifacts}
    assert set(artifacts) == {
        "v26_slate",
        "v26_live_hit_rule_set",
        "v26_player_position_eligibility",
        "v26_selection_snapshot",
        "user_collection_snapshot",
    }
    assert all(artifact.row_count == 1 for artifact in artifacts.values())
    assert artifacts["user_collection_snapshot"].filters == (
        ("user_id", "user-a"),
        ("live_card_version", "live-2026-week-23"),
    )
    assert loaded.fingerprint == manifest.fingerprint


def test_v2_database_migrates_through_live_hit_schema_to_current(tmp_path: Path) -> None:
    database = tmp_path / "v2.duckdb"
    live_hit_tables = (
        "v26_live_hit_rule_set",
        "v26_player_position_eligibility",
        "v26_selection_snapshot",
        "user_collection_snapshot",
    )
    live_hit_indexes = (
        "idx_v26_live_hit_rule_asof",
        "idx_v26_position_eligibility_snapshot",
        "idx_v26_selection_snapshot",
        "idx_user_collection_asof",
    )
    with DuckDBStore(database) as store:
        for index in live_hit_indexes:
            store.connection.execute(f'DROP INDEX "{index}"')
        for table in live_hit_tables:
            store.connection.execute(f'DROP TABLE "{table}"')
        store.connection.execute("DELETE FROM schema_migration WHERE schema_version >= 3")
        store.connection.execute(
            "INSERT INTO schema_migration VALUES (2, now(), 'schema v2 fixture')"
        )

    with DuckDBStore(database) as migrated:
        version = migrated.connection.execute(
            "SELECT max(schema_version) FROM schema_migration"
        ).fetchone()
        tables = {row[0] for row in migrated.connection.execute("SHOW TABLES").fetchall()}
        indexes = {
            row[0]
            for row in migrated.connection.execute(
                "SELECT index_name FROM duckdb_indexes()"
            ).fetchall()
        }

    assert version == (SCHEMA_VERSION,)
    assert set(live_hit_tables) <= tables
    assert set(live_hit_indexes) <= indexes
