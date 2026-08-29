"""Append-only DuckDB access with point-in-time query helpers."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import duckdb

from cpv26.domain import utc_datetime

from .integrity import (
    COMPOSITE_REFERENCE_RULES,
    REFERENCE_RULES,
    CompositeReferenceViolation,
    ReferenceViolation,
    assert_composite_referential_integrity,
    assert_referential_integrity,
    find_composite_reference_violations,
    find_reference_violations,
)
from .schema import (
    DOMAIN_TIMESTAMP_COLUMNS,
    TABLE_DEFINITIONS,
    TEMPORAL_COLUMNS,
    assert_schema_current,
    install_schema,
)


class AppendValidationError(ValueError):
    """Raised before a malformed row reaches DuckDB."""


def _normalise_value(column: str, value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if column.endswith("_sha256") or column.endswith("_fingerprint"):
        if value is None:
            return None
        if not isinstance(value, str) or len(value) != 64:
            raise AppendValidationError(f"{column} must be a SHA-256 hex digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise AppendValidationError(f"{column} must be hexadecimal") from exc
        return value.lower()
    if column in TEMPORAL_COLUMNS:
        if value is None and column == "valid_to":
            return None
        if not isinstance(value, datetime):
            raise AppendValidationError(f"{column} must be a datetime")
        return utc_datetime(value, field_name=column)
    if column in DOMAIN_TIMESTAMP_COLUMNS:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise AppendValidationError(f"{column} must be a datetime")
        return utc_datetime(value, field_name=column)
    if column.endswith("_json"):
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError as exc:
                raise AppendValidationError(f"{column} is not valid JSON") from exc
            return value
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=_json_default,
            )
        except (TypeError, ValueError) as exc:
            raise AppendValidationError(f"{column} is not JSON serialisable") from exc
    return value


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


class DuckDBStore:
    """Application-level append-only facade over one DuckDB connection.

    The store deliberately has no update or delete method. Corrections arrive
    as a new physical row sharing the same natural identity and a later
    ``available_at``/``ingested_at`` pair. The raw :attr:`connection` remains
    available for read queries and DuckDB interoperability, but mutating SQL
    executed through it bypasses every append-only and prediction-run guard.
    Production callers must therefore restrict DB-file write access and route
    writes through :meth:`append`.
    """

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        read_only: bool = False,
        threads: int | None = None,
    ) -> None:
        if isinstance(database, Path):
            database.parent.mkdir(parents=True, exist_ok=True)
            database_name = str(database)
        else:
            database_name = database
            if database_name != ":memory:" and not read_only:
                Path(database_name).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.database = database_name
        self.read_only = read_only
        self._connection = duckdb.connect(database_name, read_only=read_only)
        self._connection.execute("SET TimeZone = 'UTC'")
        if threads is not None:
            if threads < 1:
                raise ValueError("threads must be positive")
            self._connection.execute(f"SET threads = {int(threads)}")
        self._lock = threading.RLock()
        self._transaction_active = False
        self._transaction_failed = False
        self._columns_cache: dict[str, tuple[str, ...]] = {}
        self._required_cache: dict[str, frozenset[str]] = {}
        if read_only:
            assert_schema_current(self._connection)
        else:
            install_schema(self._connection)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the raw connection; mutating SQL bypasses store guarantees."""

        return self._connection

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group several append calls atomically."""

        with self._lock:
            if self._transaction_active:
                raise RuntimeError("nested DuckDBStore transactions are not supported")
            self._connection.execute("BEGIN TRANSACTION")
            self._transaction_active = True
            self._transaction_failed = False
            try:
                yield
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                if self._transaction_failed:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("transaction rolled back after a failed append")
                self._connection.execute("COMMIT")
            finally:
                self._transaction_active = False
                self._transaction_failed = False

    def table_columns(self, table: str) -> tuple[str, ...]:
        self._validate_table(table)
        with self._lock:
            cached = self._columns_cache.get(table)
            if cached is not None:
                return cached
            rows = self._connection.execute(f"PRAGMA table_info('{table}')").fetchall()
            columns = tuple(row[1] for row in rows)
            required = frozenset(row[1] for row in rows if bool(row[3]) and row[4] is None)
            self._columns_cache[table] = columns
            self._required_cache[table] = required
            return columns

    def append(
        self,
        table: str,
        rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        ignore_existing: bool = False,
    ) -> int:
        """Append homogeneous rows and return the submitted row count.

        ``ignore_existing`` only suppresses primary-key duplicates, which makes
        source ingestion retry-safe without modifying the existing row.
        """

        if self.read_only:
            raise PermissionError("cannot append through a read-only store")
        self._validate_table(table)
        materialised = self._materialise_rows(rows)
        if not materialised:
            return 0
        available_columns = frozenset(self.table_columns(table))
        required_columns = self._required_cache[table]
        first_columns = tuple(materialised[0])
        if not first_columns:
            raise AppendValidationError("row cannot be empty")
        first_column_set = frozenset(first_columns)
        unknown = first_column_set - available_columns
        if unknown:
            raise AppendValidationError(f"unknown {table} columns: {', '.join(sorted(unknown))}")
        missing = required_columns - first_column_set
        if missing:
            raise AppendValidationError(
                f"missing required {table} columns: {', '.join(sorted(missing))}"
            )
        for index, row in enumerate(materialised[1:], start=1):
            if frozenset(row) != first_column_set:
                raise AppendValidationError(f"row {index} columns do not match the first row")
        self._validate_temporal_rows(table, materialised)
        placeholders = ", ".join("?" for _ in first_columns)
        column_sql = ", ".join(f'"{column}"' for column in first_columns)
        conflict_sql = " ON CONFLICT DO NOTHING" if ignore_existing else ""
        sql = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders}){conflict_sql}'
        values = [
            [_normalise_value(column, row[column]) for column in first_columns]
            for row in materialised
        ]
        with self._lock:
            owns_transaction = not self._transaction_active
            if owns_transaction:
                self._connection.execute("BEGIN TRANSACTION")
            try:
                if table == "prediction_run":
                    self._validate_new_prediction_runs(materialised)
                self._connection.executemany(sql, values)
                if owns_transaction:
                    self._connection.execute("COMMIT")
            except Exception:
                if owns_transaction:
                    self._connection.execute("ROLLBACK")
                else:
                    self._transaction_failed = True
                raise
        return len(materialised)

    def as_of_sql(
        self,
        table: str,
        *,
        cutoff_at: datetime,
        knowledge_at: datetime | None = None,
        current_only: bool = False,
        observed_before_cutoff: bool = False,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build a deterministic latest-revision query for a forecast cutoff.

        Future scheduled games are intentionally allowed.  For tables that hold
        realised outcomes, set ``observed_before_cutoff`` to prohibit future
        events.  ``knowledge_at`` is the bitemporal system-time boundary and
        defaults to the cutoff for strict live replay.
        """

        self._validate_table(table)
        columns = self.table_columns(table)
        definition = TABLE_DEFINITIONS[table]
        cutoff = utc_datetime(cutoff_at, field_name="cutoff_at")
        knowledge = utc_datetime(knowledge_at or cutoff, field_name="knowledge_at")
        if knowledge < cutoff:
            raise ValueError("knowledge_at cannot be earlier than cutoff_at")
        eligibility_clauses = ["available_at <= ?", "ingested_at <= ?"]
        parameters: list[Any] = [cutoff, knowledge]
        final_clauses = ["_pit_rank = 1"]
        if current_only:
            parameters.extend([cutoff, cutoff])
        if observed_before_cutoff:
            # Apply this after revision selection: when a correction moves an
            # event past the cutoff, the superseded earlier timestamp must not
            # make the old physical row reappear in a historical snapshot.
            final_clauses.append("event_at < ?")
            parameters.append(cutoff)
        for column, value in (filters or {}).items():
            if column not in columns:
                raise ValueError(f"unknown {table} filter column: {column}")
            if isinstance(value, (list, tuple, set, frozenset)):
                candidates = list(value)
                if not candidates:
                    final_clauses.append("FALSE")
                else:
                    final_clauses.append(f'"{column}" IN ({", ".join("?" for _ in candidates)})')
                    parameters.extend(candidates)
            elif value is None:
                final_clauses.append(f'"{column}" IS NULL')
            else:
                final_clauses.append(f'"{column}" = ?')
                parameters.append(value)
        partition_sql = ", ".join(f'"{column}"' for column in definition.natural_identity)
        projected_columns = ", ".join(f'"{column}"' for column in columns)
        row_id = definition.row_identity
        if current_only:
            sql = f"""
                WITH eligible AS (
                    SELECT {projected_columns}
                    FROM "{table}"
                    WHERE {" AND ".join(eligibility_clauses)}
                ), corrected_ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY {partition_sql}, valid_from
                        ORDER BY available_at DESC, ingested_at DESC, "{row_id}" DESC
                    ) AS _correction_rank
                    FROM eligible
                ), corrected_versions AS (
                    SELECT {projected_columns}
                    FROM corrected_ranked
                    WHERE _correction_rank = 1
                ), valid_versions AS (
                    SELECT {projected_columns}
                    FROM corrected_versions
                    WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY {partition_sql}
                        ORDER BY valid_from DESC, available_at DESC,
                                 ingested_at DESC, "{row_id}" DESC
                    ) AS _pit_rank
                    FROM valid_versions
                )
                SELECT {projected_columns}
                FROM ranked
                WHERE {" AND ".join(final_clauses)}
                ORDER BY {partition_sql}
            """
        else:
            sql = f"""
                WITH eligible AS (
                    SELECT {projected_columns}
                    FROM "{table}"
                    WHERE {" AND ".join(eligibility_clauses)}
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY {partition_sql}
                        ORDER BY available_at DESC, ingested_at DESC,
                                 valid_from DESC, "{row_id}" DESC
                    ) AS _pit_rank
                    FROM eligible
                )
                SELECT {projected_columns}
                FROM ranked
                WHERE {" AND ".join(final_clauses)}
                ORDER BY {partition_sql}
            """
        return sql, parameters

    def fetch_as_of(
        self,
        table: str,
        *,
        cutoff_at: datetime,
        knowledge_at: datetime | None = None,
        current_only: bool = False,
        observed_before_cutoff: bool = False,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sql, parameters = self.as_of_sql(
            table,
            cutoff_at=cutoff_at,
            knowledge_at=knowledge_at,
            current_only=current_only,
            observed_before_cutoff=observed_before_cutoff,
            filters=filters,
        )
        with self._lock:
            cursor = self._connection.execute(sql, parameters)
            columns = tuple(description[0] for description in cursor.description)
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def latest_prediction_run(
        self,
        prediction_run_id: str,
        *,
        knowledge_at: datetime,
    ) -> dict[str, Any]:
        knowledge = utc_datetime(knowledge_at, field_name="knowledge_at")
        rows = self.fetch_as_of(
            "prediction_run",
            cutoff_at=knowledge,
            knowledge_at=knowledge,
            filters={"prediction_run_id": prediction_run_id},
        )
        if not rows:
            raise KeyError(f"unknown prediction_run_id: {prediction_run_id}")
        return rows[0]

    def reference_violations(self, *, sample_limit: int = 10) -> tuple[ReferenceViolation, ...]:
        """Audit unresolved business references across all physical rows."""

        with self._lock:
            return find_reference_violations(
                self._connection,
                sample_limit=sample_limit,
            )

    def assert_referential_integrity(self, *, sample_limit: int = 10) -> None:
        """Fail before snapshot or training when declared references are missing."""

        with self._lock:
            assert_referential_integrity(
                self._connection,
                sample_limit=sample_limit,
            )

    def composite_reference_violations(
        self,
        *,
        sample_limit: int = 10,
    ) -> tuple[CompositeReferenceViolation, ...]:
        """Audit multi-column references across all physical rows."""

        with self._lock:
            return find_composite_reference_violations(
                self._connection,
                sample_limit=sample_limit,
            )

    def assert_composite_referential_integrity(self, *, sample_limit: int = 10) -> None:
        """Fail when a child points to the wrong game/team/slate parent tuple."""

        with self._lock:
            assert_composite_referential_integrity(
                self._connection,
                sample_limit=sample_limit,
            )

    def as_of_reference_violations(
        self,
        *,
        cutoff_at: datetime,
        knowledge_at: datetime | None = None,
        sample_limit: int = 10,
    ) -> tuple[ReferenceViolation | CompositeReferenceViolation, ...]:
        """Audit latest valid child/parent states at one replay boundary.

        This catches a parent revision that exists physically but was not yet
        available at the prediction cutoff, a case the raw reference audit is
        intentionally unable to detect.
        """

        if sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        cutoff = utc_datetime(cutoff_at, field_name="cutoff_at")
        knowledge = utc_datetime(knowledge_at or cutoff, field_name="knowledge_at")
        if knowledge < cutoff:
            raise ValueError("knowledge_at cannot be earlier than cutoff_at")
        table_cache: dict[str, list[dict[str, Any]]] = {}

        def rows(table: str) -> list[dict[str, Any]]:
            if table not in table_cache:
                table_cache[table] = self.fetch_as_of(
                    table,
                    cutoff_at=cutoff,
                    knowledge_at=knowledge,
                    current_only=True,
                )
            return table_cache[table]

        violations: list[ReferenceViolation | CompositeReferenceViolation] = []
        for reference_rule in REFERENCE_RULES:
            parent_values = {
                row[reference_rule.parent_column]
                for row in rows(reference_rule.parent_table)
                if row[reference_rule.parent_column] is not None
            }
            missing = sorted(
                {
                    str(row[reference_rule.child_column])
                    for row in rows(reference_rule.child_table)
                    if row[reference_rule.child_column] is not None
                    and row[reference_rule.child_column] not in parent_values
                }
            )
            if missing:
                violations.append(
                    ReferenceViolation(
                        rule=reference_rule,
                        missing_value_count=len(missing),
                        sample_values=tuple(missing[:sample_limit]),
                    )
                )
        for composite_rule in COMPOSITE_REFERENCE_RULES:
            parent_values = {
                tuple(row[column] for column in composite_rule.parent_columns)
                for row in rows(composite_rule.parent_table)
                if all(row[column] is not None for column in composite_rule.parent_columns)
            }
            missing_values = {
                tuple(row[column] for column in composite_rule.child_columns)
                for row in rows(composite_rule.child_table)
                if all(row[column] is not None for column in composite_rule.child_columns)
                and tuple(row[column] for column in composite_rule.child_columns)
                not in parent_values
            }
            missing = sorted("|".join(str(value) for value in item) for item in missing_values)
            if missing:
                violations.append(
                    CompositeReferenceViolation(
                        rule=composite_rule,
                        missing_value_count=len(missing),
                        sample_values=tuple(missing[:sample_limit]),
                    )
                )
        return tuple(violations)

    @staticmethod
    def _materialise_rows(
        rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(rows, Mapping) or is_dataclass(rows):
            source: Sequence[Any] = [rows]
        else:
            source = list(rows)
        materialised: list[dict[str, Any]] = []
        for row in source:
            if is_dataclass(row) and not isinstance(row, type):
                row = asdict(cast(Any, row))
            if not isinstance(row, Mapping):
                raise TypeError("append rows must be mappings or dataclass instances")
            materialised.append(dict(row))
        return materialised

    @staticmethod
    def _validate_temporal_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        for index, row in enumerate(rows):
            missing = set(TEMPORAL_COLUMNS[:-1]) - set(row)
            if missing:
                raise AppendValidationError(
                    f"{table} row {index} lacks temporal columns: {', '.join(sorted(missing))}"
                )
            for column in TEMPORAL_COLUMNS[:-1]:
                if not isinstance(row[column], datetime):
                    raise AppendValidationError(f"{column} must be a datetime")
                utc_datetime(row[column], field_name=column)
            available_at = utc_datetime(row["available_at"])
            ingested_at = utc_datetime(row["ingested_at"])
            if ingested_at < available_at:
                raise AppendValidationError("ingested_at cannot be earlier than available_at")
            valid_from = row["valid_from"]
            valid_to = row.get("valid_to")
            if valid_to is not None:
                if not isinstance(valid_to, datetime):
                    raise AppendValidationError("valid_to must be a datetime or None")
                if utc_datetime(valid_to) <= utc_datetime(valid_from):
                    raise AppendValidationError("valid_to must be later than valid_from")

    def _validate_new_prediction_runs(self, rows: Sequence[Mapping[str, Any]]) -> None:
        identifiers: list[str] = []
        for row in rows:
            value = row.get("prediction_run_id")
            if not isinstance(value, str) or not value.strip():
                raise AppendValidationError("prediction_run_id must be a non-empty string")
            identifiers.append(value)
        duplicate_batch = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        if duplicate_batch:
            raise AppendValidationError(
                "prediction_run_id is immutable and duplicated in append batch: "
                + ", ".join(duplicate_batch)
            )
        placeholders = ", ".join("?" for _ in identifiers)
        existing = self._connection.execute(
            f"""
            SELECT prediction_run_id
            FROM prediction_run
            WHERE prediction_run_id IN ({placeholders})
            ORDER BY prediction_run_id
            """,
            identifiers,
        ).fetchall()
        if existing:
            raise AppendValidationError(
                "prediction_run_id is immutable and already exists: "
                + ", ".join(str(row[0]) for row in existing)
            )

    @staticmethod
    def _validate_table(table: str) -> None:
        if table not in TABLE_DEFINITIONS:
            raise ValueError(f"unknown append-only table: {table}")
