"""Reproducible, leakage-safe Parquet snapshots for one prediction run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from cpv26.domain import UTC, utc_datetime, utc_isoformat

from .schema import SCHEMA_VERSION, TABLE_DEFINITIONS
from .store import DuckDBStore


@dataclass(frozen=True, slots=True)
class SnapshotTableSpec:
    name: str
    current_only: bool = False
    observed_before_cutoff: bool = False
    prediction_run_scoped: bool = False
    generated_for_run: bool = False
    filters: tuple[tuple[str, str | int | float | bool | None], ...] = ()


DEFAULT_TABLE_SPECS: tuple[SnapshotTableSpec, ...] = (
    SnapshotTableSpec("source_revision"),
    SnapshotTableSpec("prediction_run", prediction_run_scoped=True, generated_for_run=True),
    # Lifecycle status is operational metadata and is excluded to avoid a
    # circular fingerprint when the run is marked snapshotted after this build.
    SnapshotTableSpec("player", current_only=True),
    SnapshotTableSpec("team", current_only=True),
    SnapshotTableSpec("stadium", current_only=True),
    SnapshotTableSpec("game"),
    SnapshotTableSpec("game_status_snapshot", current_only=True),
    SnapshotTableSpec("starter_announcement", current_only=True),
    SnapshotTableSpec("team_season"),
    SnapshotTableSpec("team_game", observed_before_cutoff=True),
    SnapshotTableSpec("player_game_batting", observed_before_cutoff=True),
    SnapshotTableSpec("roster_spell"),
    SnapshotTableSpec("lineup_version"),
    SnapshotTableSpec("lineup_entry"),
    SnapshotTableSpec("observed_plate_appearance", observed_before_cutoff=True),
    SnapshotTableSpec("substitution_event", observed_before_cutoff=True),
    SnapshotTableSpec("runner_event", observed_before_cutoff=True),
    SnapshotTableSpec("fielding_assignment", observed_before_cutoff=True),
    SnapshotTableSpec("catcher_assignment", observed_before_cutoff=True),
    SnapshotTableSpec("pitching_appearance", observed_before_cutoff=True),
    SnapshotTableSpec("weather_station_version", current_only=True),
    SnapshotTableSpec("stadium_weather_station_map", current_only=True),
    SnapshotTableSpec("weather_forecast_snapshot", current_only=True),
    SnapshotTableSpec("weather_observation", observed_before_cutoff=True),
    SnapshotTableSpec(
        "player_game_candidate",
        prediction_run_scoped=True,
        generated_for_run=True,
    ),
    SnapshotTableSpec(
        "player_state_snapshot",
        prediction_run_scoped=True,
        generated_for_run=True,
    ),
    SnapshotTableSpec(
        "team_state_snapshot",
        prediction_run_scoped=True,
        generated_for_run=True,
    ),
)


def live_hit_snapshot_specs(
    *,
    user_id: str,
    slate_id: str,
    live_card_version: str,
    rule_version: str,
    position_eligibility_snapshot_id: str,
    selection_snapshot_id: str,
) -> tuple[SnapshotTableSpec, ...]:
    """Return account- and slate-scoped Live Hit input specifications.

    User collection state is deliberately excluded from the generic default
    snapshot. A prediction run has no first-class ``user_id`` column, so this
    explicit factory prevents one user's artifact from silently containing all
    other users' collection rows.
    """

    values = {
        "user_id": user_id,
        "slate_id": slate_id,
        "live_card_version": live_card_version,
        "rule_version": rule_version,
        "position_eligibility_snapshot_id": position_eligibility_snapshot_id,
        "selection_snapshot_id": selection_snapshot_id,
    }
    empty = sorted(name for name, value in values.items() if not value.strip())
    if empty:
        raise ValueError("Live Hit snapshot scope cannot be empty: " + ", ".join(empty))
    return (
        SnapshotTableSpec(
            "v26_slate",
            current_only=True,
            filters=(
                ("slate_id", slate_id),
                ("rule_version", rule_version),
                ("live_card_version", live_card_version),
                (
                    "position_eligibility_snapshot_id",
                    position_eligibility_snapshot_id,
                ),
            ),
        ),
        SnapshotTableSpec(
            "v26_live_hit_rule_set",
            current_only=True,
            filters=(("rule_version", rule_version),),
        ),
        SnapshotTableSpec(
            "v26_player_position_eligibility",
            current_only=True,
            filters=(
                ("position_eligibility_snapshot_id", position_eligibility_snapshot_id),
                ("slate_id", slate_id),
                ("live_card_version", live_card_version),
            ),
        ),
        SnapshotTableSpec(
            "v26_selection_snapshot",
            current_only=True,
            filters=(
                ("selection_snapshot_id", selection_snapshot_id),
                ("slate_id", slate_id),
                ("rule_version", rule_version),
            ),
        ),
        SnapshotTableSpec(
            "user_collection_snapshot",
            current_only=True,
            filters=(("user_id", user_id), ("live_card_version", live_card_version)),
        ),
    )


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    table: str
    path: str
    row_count: int
    columns: tuple[str, ...]
    content_sha256: str
    file_sha256: str
    current_only: bool
    observed_before_cutoff: bool
    prediction_run_scoped: bool
    generated_for_run: bool
    filters: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SnapshotArtifact:
        values = dict(data)
        values["columns"] = tuple(values["columns"])
        values["filters"] = tuple(
            (str(item[0]), item[1]) for item in values.get("filters", ())
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    schema_version: int
    prediction_run_id: str
    cutoff_at: str
    knowledge_at: str
    created_at: str
    fingerprint: str
    artifacts: tuple[SnapshotArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SnapshotManifest:
        values = dict(data)
        values["artifacts"] = tuple(
            SnapshotArtifact.from_dict(item) for item in values["artifacts"]
        )
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> SnapshotManifest:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def verify(self, snapshot_directory: str | Path) -> None:
        root = Path(snapshot_directory)
        for artifact in self.artifacts:
            file_path = root / artifact.path
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            actual = _file_sha256(file_path)
            if actual != artifact.file_sha256:
                raise ValueError(
                    f"snapshot file checksum mismatch for {artifact.table}: "
                    f"{actual} != {artifact.file_sha256}"
                )
        actual_fingerprint = _manifest_fingerprint(
            self.prediction_run_id,
            self.cutoff_at,
            self.knowledge_at,
            self.artifacts,
            schema_version=self.schema_version,
        )
        if actual_fingerprint != self.fingerprint:
            raise ValueError("snapshot manifest fingerprint is inconsistent")


class SnapshotBuilder:
    """Materialise a prediction run's point-in-time inputs as Parquet.

    Source rows must satisfy ``available_at <= cutoff_at``. Generated rows tied
    to the run (candidate scenarios and state snapshots) may be produced after
    the forecast cutoff, but must satisfy ``ingested_at <= knowledge_at`` and
    are always filtered by ``prediction_run_id``. Realised plate appearances
    additionally require ``event_at < cutoff_at``.
    """

    def __init__(
        self,
        store: DuckDBStore,
        output_root: str | Path,
        *,
        table_specs: Sequence[SnapshotTableSpec] = DEFAULT_TABLE_SPECS,
        fetch_batch_size: int = 10_000,
    ) -> None:
        if fetch_batch_size < 1:
            raise ValueError("fetch_batch_size must be positive")
        if not table_specs:
            raise ValueError("at least one table must be snapshotted")
        names = [spec.name for spec in table_specs]
        if len(names) != len(set(names)):
            raise ValueError("snapshot table names must be unique")
        for spec in table_specs:
            if spec.name not in TABLE_DEFINITIONS:
                raise ValueError(f"unknown snapshot table: {spec.name}")
            if spec.generated_for_run and not spec.prediction_run_scoped:
                raise ValueError("generated snapshot tables must be prediction-run scoped")
            if spec.generated_for_run and spec.observed_before_cutoff:
                raise ValueError("generated tables cannot be marked as observed history")
            filter_names = [name for name, _ in spec.filters]
            if len(filter_names) != len(set(filter_names)):
                raise ValueError(f"duplicate snapshot filters for {spec.name}")
            unknown_filters = set(filter_names) - set(store.table_columns(spec.name))
            if unknown_filters:
                raise ValueError(
                    f"unknown snapshot filters for {spec.name}: "
                    + ", ".join(sorted(unknown_filters))
                )
            if spec.prediction_run_scoped and "prediction_run_id" in filter_names:
                raise ValueError(
                    "prediction_run_id is supplied automatically for run-scoped tables"
                )
        self.store = store
        self.output_root = Path(output_root)
        self.table_specs = tuple(table_specs)
        self.fetch_batch_size = fetch_batch_size

    def build(
        self,
        prediction_run_id: str,
        *,
        cutoff_at: datetime | None = None,
        knowledge_at: datetime | None = None,
    ) -> SnapshotManifest:
        safe_run_id = _safe_path_component(prediction_run_id)
        lookup_at = utc_datetime(knowledge_at or datetime.now(UTC), field_name="knowledge_at")
        run = self.store.latest_prediction_run(prediction_run_id, knowledge_at=lookup_at)
        stored_cutoff = utc_datetime(run["cutoff_at"], field_name="cutoff_at")
        stored_knowledge = utc_datetime(run["knowledge_at"], field_name="knowledge_at")
        cutoff = utc_datetime(cutoff_at or stored_cutoff, field_name="cutoff_at")
        knowledge = utc_datetime(knowledge_at or stored_knowledge, field_name="knowledge_at")
        if cutoff != stored_cutoff:
            raise ValueError("cutoff_at does not match the prediction run")
        if knowledge != stored_knowledge:
            raise ValueError("knowledge_at does not match the prediction run")
        if knowledge < cutoff:
            raise ValueError("knowledge_at cannot precede cutoff_at")

        self.output_root.mkdir(parents=True, exist_ok=True)
        destination = self.output_root / safe_run_id
        temp_path = Path(tempfile.mkdtemp(prefix=f".{safe_run_id}.", dir=self.output_root))
        artifacts: list[SnapshotArtifact] = []
        try:
            with self.store.transaction():
                for spec in self.table_specs:
                    filters: dict[str, Any] = dict(spec.filters)
                    if spec.prediction_run_scoped:
                        filters["prediction_run_id"] = prediction_run_id
                    availability_boundary = knowledge if spec.generated_for_run else cutoff
                    sql, parameters = self.store.as_of_sql(
                        spec.name,
                        cutoff_at=availability_boundary,
                        knowledge_at=knowledge,
                        current_only=spec.current_only,
                        observed_before_cutoff=spec.observed_before_cutoff,
                        filters=filters,
                    )
                    row_count, columns, content_sha256 = self._query_digest(sql, parameters)
                    relative_path = f"{spec.name}.parquet"
                    parquet_path = temp_path / relative_path
                    self._copy_to_parquet(sql, parameters, parquet_path)
                    artifacts.append(
                        SnapshotArtifact(
                            table=spec.name,
                            path=relative_path,
                            row_count=row_count,
                            columns=columns,
                            content_sha256=content_sha256,
                            file_sha256=_file_sha256(parquet_path),
                            current_only=spec.current_only,
                            observed_before_cutoff=spec.observed_before_cutoff,
                            prediction_run_scoped=spec.prediction_run_scoped,
                            generated_for_run=spec.generated_for_run,
                            filters=spec.filters,
                        )
                    )
            artifact_tuple = tuple(artifacts)
            fingerprint = _manifest_fingerprint(
                prediction_run_id,
                utc_isoformat(cutoff),
                utc_isoformat(knowledge),
                artifact_tuple,
                schema_version=SCHEMA_VERSION,
            )
            manifest = SnapshotManifest(
                schema_version=SCHEMA_VERSION,
                prediction_run_id=prediction_run_id,
                cutoff_at=utc_isoformat(cutoff),
                knowledge_at=utc_isoformat(knowledge),
                created_at=utc_isoformat(datetime.now(UTC)),
                fingerprint=fingerprint,
                artifacts=artifact_tuple,
            )
            manifest_path = temp_path / "manifest.json"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    manifest.to_dict(),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
            if destination.exists():
                existing_manifest_path = destination / "manifest.json"
                if existing_manifest_path.is_file():
                    existing = SnapshotManifest.load(existing_manifest_path)
                    existing.verify(destination)
                    if existing.fingerprint == manifest.fingerprint:
                        shutil.rmtree(temp_path)
                        return existing
                raise FileExistsError(
                    f"snapshot directory already exists with different content: {destination}"
                )
            os.replace(temp_path, destination)
            manifest.verify(destination)
            return manifest
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise

    def _query_digest(
        self, sql: str, parameters: Sequence[Any]
    ) -> tuple[int, tuple[str, ...], str]:
        cursor = self.store.connection.execute(sql, parameters)
        columns = tuple(item[0] for item in cursor.description)
        column_types = tuple(str(item[1]) for item in cursor.description)
        digest = hashlib.sha256()
        digest.update(_canonical_json({"columns": columns, "types": column_types}) + b"\n")
        row_count = 0
        while True:
            rows = cursor.fetchmany(self.fetch_batch_size)
            if not rows:
                break
            for row in rows:
                digest.update(_canonical_json([_canonical_value(value) for value in row]))
                digest.update(b"\n")
                row_count += 1
        return row_count, columns, digest.hexdigest()

    def _copy_to_parquet(self, sql: str, parameters: Sequence[Any], destination: Path) -> None:
        escaped_path = str(destination).replace("'", "''")
        copy_sql = f"COPY ({sql}) TO '{escaped_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        self.store.connection.execute(copy_sql, parameters)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, datetime):
        return utc_isoformat(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return {"decimal": format(value, "f")}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return {"type": type(value).__name__, "text": str(value)}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _manifest_fingerprint(
    prediction_run_id: str,
    cutoff_at: str,
    knowledge_at: str,
    artifacts: Iterable[SnapshotArtifact],
    *,
    schema_version: int,
) -> str:
    payload = {
        "schema_version": schema_version,
        "prediction_run_id": prediction_run_id,
        "cutoff_at": cutoff_at,
        "knowledge_at": knowledge_at,
        "tables": [
            {
                "table": artifact.table,
                "row_count": artifact.row_count,
                "columns": artifact.columns,
                "content_sha256": artifact.content_sha256,
                "current_only": artifact.current_only,
                "observed_before_cutoff": artifact.observed_before_cutoff,
                "prediction_run_scoped": artifact.prediction_run_scoped,
                "generated_for_run": artifact.generated_for_run,
                **({"filters": artifact.filters} if schema_version >= 3 else {}),
            }
            for artifact in artifacts
        ],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path_component(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError("prediction_run_id cannot be empty or relative")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if any(character not in allowed for character in value):
        raise ValueError(
            "prediction_run_id may contain only letters, digits, dash, underscore and dot"
        )
    return value
