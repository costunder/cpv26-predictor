from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from cpv26.data.kbo_source_snapshots import (
    source_snapshot_filter_sql,
    superseded_source_ids,
    superseded_source_ids_sql,
)

_FIRST = datetime(2026, 8, 1, tzinfo=timezone.utc)
_LATER = _FIRST + timedelta(days=1)


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE source_revision (source_revision_id VARCHAR, source_name VARCHAR, "
        "metadata_json JSON, ingested_at TIMESTAMPTZ)"
    )
    return connection


def _source(
    connection: duckdb.DuckDBPyConnection, revision: str, *, season: int = 2023,
    adapter: int = 1, at: datetime = _FIRST, policy: str | None = None,
    source_name: str = "slothman3878/kbo_playbyplay", source_id: str | None = None,
) -> str:
    identifier = source_id or f"hf-kbo-playbyplay:{revision}:{season}:adapter-v{adapter}"
    metadata: dict[str, str | int] = {
        "dataset_revision": revision, "season": season, "adapter_version": adapter,
    }
    if policy is not None:
        metadata["snapshot_policy"] = policy
    connection.execute(
        "INSERT INTO source_revision VALUES (?, ?, ?, ?)",
        [identifier, source_name, json.dumps(metadata), at],
    )
    return identifier


def test_known_annual_snapshot_replaces_legacy_without_touching_other_scopes() -> None:
    with _connection() as connection:
        old = _source(connection, "old")
        new = _source(connection, "new", adapter=2, at=_LATER, policy="annual_snapshot")
        _source(connection, "other-year", season=2024)
        _source(connection, "other-provider", at=_LATER, source_name="another provider")
        _source(connection, "incremental", at=_LATER, policy="incremental")
        _source(connection, "similar-name", at=_LATER, source_id="manual-source")
        assert superseded_source_ids(connection, _FIRST) == frozenset()
        assert superseded_source_ids(connection, _LATER) == frozenset({old})
        assert superseded_source_ids(connection) == frozenset({old})
        assert new not in superseded_source_ids(connection)
        assert connection.execute("SELECT count(*) FROM source_revision").fetchone() == (6,)


def test_same_ingestion_time_prefers_new_adapter_then_stable_source_id() -> None:
    with _connection() as connection:
        old_adapter = _source(connection, "zzzz", adapter=1)
        older_id = _source(connection, "aaaa", adapter=2)
        chosen = _source(connection, "bbbb", adapter=2)
        assert superseded_source_ids(connection) == frozenset({old_adapter, older_id})
        assert chosen not in superseded_source_ids(connection)


def test_bound_sql_and_record_reader_share_one_knowledge_parameter() -> None:
    with _connection() as connection:
        old = _source(connection, "old")
        new = _source(connection, "new", at=_LATER)
        sql = superseded_source_ids_sql(knowledge_bound=True)
        assert sql.count("?") == 1
        assert connection.execute(sql, [_LATER]).fetchall() == [(old,)]
        connection.execute("CREATE TABLE facts (source_revision_id VARCHAR)")
        connection.executemany(
            "INSERT INTO facts VALUES (?)", [(old,), (new,), ("manual",), (None,)]
        )
        filtered = source_snapshot_filter_sql("facts.source_revision_id", knowledge_bound=True)
        assert connection.execute(
            f"SELECT * FROM facts WHERE {filtered} ORDER BY source_revision_id", [_LATER]
        ).fetchall() == [(new,), ("manual",), (None,)]


@pytest.mark.parametrize("metadata", [{}, {"season": "invalid"}, [], None])
def test_missing_or_malformed_legacy_metadata_is_not_assumed_to_be_a_snapshot(
    metadata: object,
) -> None:
    with _connection() as connection:
        connection.execute(
            "INSERT INTO source_revision VALUES (?, ?, ?, ?)",
            ["legacy-unmanaged", "slothman3878/kbo_playbyplay", json.dumps(metadata), _FIRST],
        )
        _source(connection, "new", at=_LATER)
        assert not superseded_source_ids(connection)


def test_naive_knowledge_and_non_column_sql_identifiers_are_rejected() -> None:
    with _connection() as connection, pytest.raises(ValueError, match="timezone"):
        superseded_source_ids(connection, datetime(2026, 8, 1))
    with pytest.raises(ValueError, match="simple column"):
        source_snapshot_filter_sql("source_revision_id) OR TRUE --")
