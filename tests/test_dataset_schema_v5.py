from pathlib import Path

import duckdb
import pytest

from cpv26.data import DuckDBStore
from cpv26.data.schema import DDL, SCHEMA_MIGRATION_DDL
from cpv26.data.schema_v5 import V5_DDL


def _legacy_database(path: Path, version: int = 4) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute(SCHEMA_MIGRATION_DDL)
        for statement in DDL:
            if statement not in V5_DDL:
                connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migration VALUES (?, current_timestamp, 'legacy fixture')",
            [version],
        )
        connection.execute("""
            INSERT INTO source_revision (
                source_revision_id, source_name, source_locator, metadata_json, content_sha256,
                event_at, available_at, ingested_at, valid_from
            ) VALUES ('original', 'original-source', 'original-locator', '{"unchanged":true}',
                      repeat('a', 64),
                      '2001-04-05 00:00:00+00', '2001-04-06 00:00:00+00',
                      '2026-08-31 00:00:00+00', '2001-04-05 00:00:00+00')
        """)


def test_v4_migration_adds_lossless_tables_without_rewriting_original_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.duckdb"
    _legacy_database(path)
    with duckdb.connect(str(path)) as connection:
        before = connection.execute("SELECT * FROM source_revision").fetchall()
        assert "historical_boxscore" not in {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        }
    for _ in range(2):
        with DuckDBStore(path) as store:
            assert store.connection.execute("SELECT * FROM source_revision").fetchall() == before
            assert store.connection.execute(
                "SELECT schema_version FROM schema_migration ORDER BY schema_version"
            ).fetchall() == [(4,), (5,)]
            for table in ("historical_boxscore", "historical_game_detail"):
                assert store.connection.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
            store.assert_referential_integrity()
            store.assert_composite_referential_integrity()


def test_newer_schema_refused_without_installing_or_rewriting_tables(tmp_path: Path) -> None:
    path = tmp_path / "future.duckdb"
    _legacy_database(path, 6)
    with pytest.raises(RuntimeError, match="schema version"):
        DuckDBStore(path)
    with duckdb.connect(str(path)) as connection:
        assert "historical_boxscore" not in {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        }
        assert connection.execute("SELECT count(*) FROM source_revision").fetchone() == (1,)
