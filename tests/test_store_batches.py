from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pytest

from cpv26.data import DuckDBStore
from cpv26.data.store import AppendValidationError


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


@pytest.mark.parametrize("columnar", [False, True])
def test_batched_append_preserves_normalization_and_retry_semantics(columnar: bool) -> None:
    with DuckDBStore() as store:
        rows = [_row(index) for index in range(7)]
        assert store.append("source_revision", rows, batch_size=3, columnar=columnar) == 7
        assert store.append(
            "source_revision", rows, batch_size=2, ignore_existing=True, columnar=columnar,
        ) == 7
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


@pytest.mark.parametrize("columnar", [False, True])
def test_later_batch_failure_rolls_back_earlier_batches(columnar: bool) -> None:
    with DuckDBStore() as store:
        store.append("source_revision", _row(0))
        with pytest.raises(duckdb.ConstraintException):
            store.append(
                "source_revision", [_row(1), _row(2), _row(0)], batch_size=2, columnar=columnar,
            )
        assert store.connection.execute(
            "SELECT source_revision_id FROM source_revision"
        ).fetchall() == [
            ("fixture-0",),
        ]


@pytest.mark.parametrize("columnar", [False, True])
def test_batched_append_participates_in_outer_transaction(columnar: bool) -> None:
    with DuckDBStore() as store:
        with pytest.raises(RuntimeError, match="stop"), store.transaction():
            store.append("source_revision", [_row(1), _row(2)], batch_size=2, columnar=columnar)
            raise RuntimeError("stop")
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (0,)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected_before_writing(batch_size: Any) -> None:
    with DuckDBStore() as store, pytest.raises(ValueError, match="batch_size"):
        store.append("source_revision", _row(0), batch_size=batch_size)


@pytest.mark.parametrize("columnar", [None, 0, 1, "yes"])
def test_invalid_columnar_flag_is_rejected_before_writing(columnar: Any) -> None:
    with DuckDBStore() as store, pytest.raises(ValueError, match="columnar"):
        store.append("source_revision", _row(0), columnar=columnar)


def test_columnar_handles_null_columns_json_timezone_and_omitted_defaults() -> None:
    rows = [_row(0), _row(1)]
    for row in rows:
        row.pop("metadata_json")  # Uses the DB default, not an explicit null.
        row["source_locator"] = None  # An entire typed VARCHAR[] parameter is null.
        row["event_at"] = datetime(2020, 1, 1, 9, tzinfo=timezone(timedelta(hours=9)))
    with DuckDBStore() as store:
        assert store.append("source_revision", rows, batch_size=2, columnar=True) == 2
        assert store.connection.execute(
            "SELECT source_locator, metadata_json, event_at, valid_to FROM source_revision"
        ).fetchall() == [(None, "{}", datetime(2020, 1, 1, tzinfo=timezone.utc), None)] * 2
        extra = _row(2)
        extra["metadata_json"] = {"한글": [0, None, False], "nested": {"null": None}}
        store.append("source_revision", extra, columnar=True)
        assert store.connection.execute(
            "SELECT metadata_json FROM source_revision WHERE source_revision_id = 'fixture-2'"
        ).fetchone() == ('{"nested":{"null":null},"한글":[0,null,false]}',)


@pytest.mark.parametrize("columnar", [False, True])
def test_ignore_existing_does_not_change_first_row_or_suppress_other_constraints(
    columnar: bool,
) -> None:
    first, second = _row(0), _row(0)
    second["source_name"] = "must not replace first"
    with DuckDBStore() as store:
        assert store.append(
            "source_revision", [first, second], batch_size=2,
            columnar=columnar, ignore_existing=True,
        ) == 2
        assert store.connection.execute("SELECT source_name FROM source_revision").fetchone() == (
            "test_fixture",
        )
        invalid = _row(2)
        invalid["source_name"] = None
        with pytest.raises(duckdb.ConstraintException):
            store.append(
                "source_revision", [_row(1), invalid], batch_size=1,
                columnar=columnar, ignore_existing=True,
            )
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (1,)


@pytest.mark.parametrize("failure", ["unknown", "missing", "heterogeneous", "naive", "json", "sha"])
def test_columnar_preserves_preinsert_validation(failure: str) -> None:
    rows = [_row(0), _row(1)]
    if failure == "unknown":
        for row in rows:
            row["unexpected"] = 1
    elif failure == "missing":
        for row in rows:
            row.pop("source_name")
    elif failure == "heterogeneous":
        rows[1]["source_locator"] = None
    elif failure == "naive":
        rows[1]["event_at"] = datetime(2020, 1, 1)
    elif failure == "json":
        rows[1]["metadata_json"] = "{invalid json"
    else:
        rows[1]["content_sha256"] = "not-a-digest"
    with DuckDBStore() as store:
        with pytest.raises((AppendValidationError, ValueError)):
            store.append("source_revision", rows, batch_size=2, columnar=True)
        assert store.connection.execute("SELECT count(*) FROM source_revision").fetchone() == (0,)


def test_columnar_failure_cannot_be_caught_to_commit_partial_outer_transaction() -> None:
    with DuckDBStore() as store:
        store.append("source_revision", _row(0))
        with pytest.raises(RuntimeError, match="rolled back"), store.transaction():
            store.append("source_revision", _row(1), columnar=True)
            with pytest.raises(duckdb.ConstraintException):
                store.append("source_revision", [_row(2), _row(0)], batch_size=2, columnar=True)
        assert store.connection.execute(
            "SELECT source_revision_id FROM source_revision"
        ).fetchall() == [("fixture-0",)]


def test_columnar_preserves_prediction_run_immutability_guard() -> None:
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    row = {
        "prediction_run_row_id": "physical-run", "prediction_run_id": "logical-run",
        "target_game_id": "game", "cutoff_at": now, "knowledge_at": now,
        "horizon_type": "early", "feature_version": "v1", "model_name": "test",
        "model_version": "v1", "simulator_version": "v1", "v26_rule_version": "v1",
        **{key: now for key in ("event_at", "available_at", "ingested_at", "valid_from")},
        "valid_to": None,
    }
    with DuckDBStore() as store:
        store.append("prediction_run", row, columnar=True)
        with pytest.raises(AppendValidationError, match="immutable"):
            store.append("prediction_run", row, columnar=True, ignore_existing=True)
        assert store.connection.execute("SELECT count(*) FROM prediction_run").fetchone() == (1,)


def test_columnar_respects_read_only_store_and_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "read-only.duckdb"
    with DuckDBStore(path) as store:
        assert store.append("source_revision", [], columnar=True) == 0
    with DuckDBStore(path, read_only=True) as store, pytest.raises(PermissionError):
        store.append("source_revision", _row(0), columnar=True)
