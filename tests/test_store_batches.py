from datetime import datetime, timezone
from typing import Any

import duckdb
import pytest

from cpv26.data import DuckDBStore


def _row(index: int) -> dict[str, Any]:
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return {
        "source_revision_id": f"fixture-{index}",
        "source_name": "test_fixture",
        "content_sha256": f"{index:064x}",
        "metadata_json": {"test_index": index},
        "event_at": now,
        "available_at": now,
        "ingested_at": now,
        "valid_from": now,
        "valid_to": None,
    }


def test_batched_append_preserves_normalization_and_retry_semantics() -> None:
    with DuckDBStore() as store:
        rows = [_row(index) for index in range(7)]
        assert store.append("source_revision", rows, batch_size=3) == 7
        assert store.append("source_revision", rows, batch_size=2, ignore_existing=True) == 7
        result = store.connection.execute(
            "SELECT source_revision_id, metadata_json, event_at FROM source_revision "
            "ORDER BY source_revision_id"
        ).fetchall()
        assert len(result) == 7
        assert result[4] == (
            "fixture-4",
            '{"test_index":4}',
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        )


def test_later_batch_failure_rolls_back_earlier_batches() -> None:
    with DuckDBStore() as store:
        store.append("source_revision", _row(0))
        with pytest.raises(duckdb.ConstraintException):
            store.append("source_revision", [_row(1), _row(2), _row(0)], batch_size=2)
        assert store.connection.execute(
            "SELECT source_revision_id FROM source_revision"
        ).fetchall() == [
            ("fixture-0",),
        ]


def test_batched_append_participates_in_outer_transaction() -> None:
    with DuckDBStore() as store:
        with pytest.raises(RuntimeError, match="stop"), store.transaction():
            store.append("source_revision", [_row(1), _row(2)], batch_size=2)
            raise RuntimeError("stop")
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (0,)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected_before_writing(batch_size: Any) -> None:
    with DuckDBStore() as store, pytest.raises(ValueError, match="batch_size"):
        store.append("source_revision", _row(0), batch_size=batch_size)
