"""Immutable, season-sharded temporal KBO graph archive.

Version five and six materialize one overlapping graph file for every cutoff
day.  This module keeps each source-record version once, stores labels as
references, and derives a deterministic target-centred temporal subgraph when
a day is loaded.  Current-day results and participants are not consulted while
the topology is selected.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np

from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES

from .kbo_graph_dataset import (
    BOX_FEATURE_DIMS,
    NODE_FEATURE_NAMES,
    PA_CONTEXT_FEATURE_NAMES,
    ROLE_FEATURE_NAMES,
    VNEXT_GAME_FEATURE_NAMES,
    Array,
    GraphDay,
    _Aggregate,
    _box_features,
    _box_history_id,
    _box_label_audit,
    _common_box_records,
    _complete_legacy_box_values,
    _History,
    _json_default,
    _json_sha256,
    _label_boxes,
    _label_records,
    _queries,
    _read_records,
    _Record,
    _role_features,
    _route_features,
    _target_coverage,
    _team_features,
)

TEMPORAL_GRAPH_DATASET_VERSION = 7
TEMPORAL_GRAPH_SCHEMA = "temporal_v7"
TEMPORAL_MATERIALIZATION_CONTRACT_VERSION = 3
TEMPORAL_SAMPLE_INDEX_SCHEMA_VERSION = 2
_KST = ZoneInfo("Asia/Seoul")

_PLAYER_GAME_FEATURE_NAMES = (
    "pa_log100",
    "hit_per_pa",
    "walk_hbp_per_pa",
    "strikeout_per_pa",
    "total_bases_per_pa_div4",
    "recency",
)
_RAW_PA_FEATURE_NAMES = (
    "is_plate_appearance",
    "is_at_bat",
    "is_hit",
    "total_bases_div4",
    "is_walk_or_hit_by_pitch",
    "is_strikeout",
    "is_home_run",
    *PA_CONTEXT_FEATURE_NAMES,
)
TEMPORAL_ROUTE_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "batter_game_event": _PLAYER_GAME_FEATURE_NAMES,
    "pitcher_game_event": _PLAYER_GAME_FEATURE_NAMES,
    "team_game_event": (
        "is_current_query_game",
        "is_home_team",
        "is_away_team",
        "is_historical_game",
    ),
    "batter_pa_pitcher_event": _RAW_PA_FEATURE_NAMES,
}
TEMPORAL_ROUTE_METADATA: dict[str, dict[str, Any]] = {
    "batter_game_event": {
        "source_type": "player",
        "destination_type": "game",
        "source_role": "batting",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past player-game batting participation; never current-day participation",
    },
    "pitcher_game_event": {
        "source_type": "player",
        "destination_type": "game",
        "source_role": "pitching",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past player-game pitching participation; never current-day participation",
    },
    "team_game_event": {
        "source_type": "team",
        "destination_type": "game",
        "source_role": "shared",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past games plus score-free current matchup structure",
    },
    "batter_pa_pitcher_event": {
        "source_type": "player",
        "destination_type": "player",
        "source_role": "batting",
        "destination_role": "pitching",
        "bidirectional": True,
        "meaning": "one parallel edge per completed historical plate appearance",
    },
}
_ROUTE_ARRAY_FIELDS = (
    "source_index",
    "destination_index",
    "event_features",
    "event_age_seconds",
    "publication_delay_seconds",
    "weights",
)


@dataclass(frozen=True, slots=True)
class TemporalSamplingPolicy:
    """Bound a deterministic target-centred history expansion."""

    lookback_days: int = 365
    max_games_per_seed_team: int = 160
    max_games_per_player: int = 48
    max_historical_games_total: int = 160

    def __post_init__(self) -> None:
        for name in (
            "lookback_days",
            "max_games_per_seed_team",
            "max_games_per_player",
            "max_historical_games_total",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def fingerprint(self) -> str:
        return _json_sha256({"version": 1, **asdict(self)})

    def to_dict(self) -> dict[str, int]:
        return cast(dict[str, int], asdict(self))


@dataclass(frozen=True, slots=True)
class _QueryDescriptor:
    day: date
    game_id: str
    home_team_id: str
    away_team_id: str


@dataclass(frozen=True, slots=True)
class _SelectedTopology:
    historical_games: tuple[str, ...]
    game_records: Mapping[str, _Record]
    active_plate_appearances: tuple[_Record, ...]
    active_records: tuple[_Record, ...]


class _TemporalGraphDay(GraphDay):
    """GraphDay view that exposes version-seven route prefixes."""

    __slots__ = ()

    @property
    def routes(self) -> dict[str, dict[str, Array]]:
        return {
            route: {key: self.arrays[f"{route}__{key}"] for key in _ROUTE_ARRAY_FIELDS}
            for route in TEMPORAL_ROUTE_METADATA
        }


class KBOTemporalGraphDataset:
    """Read an immutable v7 archive and derive one cutoff-safe graph on demand."""

    def __init__(
        self,
        directory: str | Path,
        label_year_ceiling: int | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        with (self.directory / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        _validate_manifest(self.manifest)
        if label_year_ceiling is not None and (
            isinstance(label_year_ceiling, bool)
            or not isinstance(label_year_ceiling, int)
            or not 1 <= label_year_ceiling <= 9999
        ):
            raise ValueError("label_year_ceiling must be a calendar year")
        self.label_year_ceiling = label_year_ceiling
        self.policy = TemporalSamplingPolicy(**self.manifest["sampling_policy"])
        self._entries = {entry["day"]: entry for entry in self.manifest["days"]}
        self._record_shards = _index_shards(self.manifest["record_shards"])
        self._label_shards = _index_shards(self.manifest["label_shards"])
        self._query_shards = _index_shards(self.manifest["query_shards"])
        self._records_cache: dict[int, tuple[_Record, ...]] = {}
        self._labels_cache: dict[int, dict[str, tuple[str, ...]]] = {}
        self._queries_cache: dict[int, dict[str, tuple[_QueryDescriptor, ...]]] = {}
        self._stream_year: int | None = None
        self._stream_last_day: date | None = None
        self._stream_history: _History | None = None
        self._stream_record_lookup: dict[str, _Record] = {}

    def days(self) -> tuple[date, ...]:
        parsed = tuple(date.fromisoformat(key) for key in sorted(self._entries))
        return tuple(
            day
            for day in parsed
            if self.label_year_ceiling is None or day.year <= self.label_year_ceiling
        )

    def load_day(self, day: date | str) -> GraphDay:
        selected_day = date.fromisoformat(day) if isinstance(day, str) else day
        key = selected_day.isoformat()
        if key not in self._entries:
            raise KeyError(f"temporal graph day not present: {key}")
        if self.label_year_ceiling is not None and selected_day.year > self.label_year_ceiling:
            raise PermissionError(
                f"labels after {self.label_year_ceiling} are sealed by label_year_ceiling"
            )

        history, lookup = self._history_for_day(selected_day)

        # This ordering is a security boundary: descriptors are score/participant
        # free, and topology is frozen before any current-day label is resolved.
        descriptors = self._queries_for_year(selected_day.year).get(key, ())
        if not descriptors:
            raise ValueError(f"temporal archive has no query descriptor for {key}")
        topology = _select_topology(history, descriptors, self.policy)
        label_keys = self._labels_for_year(selected_day.year).get(key, ())
        try:
            labels = tuple(lookup[label_key] for label_key in label_keys)
        except KeyError as exc:
            raise ValueError(
                f"label reference is missing from the immutable record shards: {exc}"
            ) from exc
        graph = _materialize_graph(
            selected_day, history, topology, descriptors, labels, self.policy
        )
        _validate_temporal_graph(graph)
        return graph

    def _history_for_day(self, day: date) -> tuple[_History, dict[str, _Record]]:
        """Reuse one monotonic history cursor per process and calendar year."""

        rebuild = (
            self._stream_history is None
            or self._stream_year != day.year
            or (self._stream_last_day is not None and day < self._stream_last_day)
        )
        if rebuild:
            records, lookup = self._records_for_stream_year(day.year)
            self._stream_history = _History(records, self.policy.lookback_days, graph_schema="v5")
            self._stream_record_lookup = lookup
            self._stream_year = day.year
            self._stream_last_day = None
        assert self._stream_history is not None
        self._stream_history.advance(day)
        self._stream_last_day = day
        return self._stream_history, self._stream_record_lookup

    def _records_for_stream_year(self, season: int) -> tuple[list[_Record], dict[str, _Record]]:
        year_start = date(season, 1, 1)
        year_end = date(season, 12, 31)
        earliest = year_start - timedelta(days=self.policy.lookback_days)
        records: list[_Record] = []
        lookup: dict[str, _Record] = {}
        for shard_year in range(earliest.year, season + 1):
            for record in self._records_for_year(shard_year):
                if earliest <= record.day <= year_end:
                    records.append(record)
                    lookup[_record_key(record)] = record
        records.sort(key=lambda record: (record.kind, record.entity, record.rank))
        return records, lookup

    def _records_for_year(self, season: int) -> tuple[_Record, ...]:
        cached = self._records_cache.get(season)
        if cached is not None:
            return cached
        entry = self._record_shards.get(season)
        if entry is None:
            self._records_cache[season] = ()
            return ()
        arrays = _read_shard(self.directory, entry)
        records = _decode_record_shard(arrays)
        self._records_cache[season] = records
        # Persistent workers only need the current and lookback shards.  Bound
        # decoded Python objects so each worker does not accumulate 25 seasons.
        while len(self._records_cache) > 3:
            oldest = next(iter(self._records_cache))
            if oldest == season:
                break
            del self._records_cache[oldest]
        return records

    def _labels_for_year(self, season: int) -> dict[str, tuple[str, ...]]:
        cached = self._labels_cache.get(season)
        if cached is not None:
            return cached
        entry = self._label_shards.get(season)
        if entry is None:
            result: dict[str, tuple[str, ...]] = {}
        else:
            arrays = _read_shard(self.directory, entry)
            days = _string_column(arrays, "day")
            keys = _string_column(arrays, "record_key")
            if len(days) != len(keys):
                raise ValueError("label shard columns disagree")
            grouped: dict[str, list[str]] = defaultdict(list)
            for day, record_key in zip(days, keys, strict=True):
                grouped[day].append(record_key)
            result = {day: tuple(values) for day, values in grouped.items()}
        self._labels_cache[season] = result
        return result

    def _queries_for_year(self, season: int) -> dict[str, tuple[_QueryDescriptor, ...]]:
        cached = self._queries_cache.get(season)
        if cached is not None:
            return cached
        entry = self._query_shards.get(season)
        if entry is None:
            result: dict[str, tuple[_QueryDescriptor, ...]] = {}
        else:
            arrays = _read_shard(self.directory, entry)
            columns = {
                name: _string_column(arrays, name)
                for name in (
                    "day",
                    "game_id",
                    "home_team_id",
                    "away_team_id",
                )
            }
            lengths = {len(values) for values in columns.values()}
            if len(lengths) != 1:
                raise ValueError("query descriptor shard columns disagree")
            grouped: dict[str, list[_QueryDescriptor]] = defaultdict(list)
            for values in zip(*columns.values(), strict=True):
                day_id, game_id, home, away = values
                grouped[day_id].append(
                    _QueryDescriptor(
                        date.fromisoformat(day_id),
                        game_id,
                        home,
                        away,
                    )
                )
            result = {
                day: tuple(sorted(values, key=lambda item: item.game_id))
                for day, values in grouped.items()
            }
        self._queries_cache[season] = result
        return result


def build_kbo_temporal_archive(
    database_path: str | Path,
    output_dir: str | Path,
    *,
    start_day: date | str | None = None,
    end_day: date | str | None = None,
    knowledge_at: datetime | None = None,
    policy: TemporalSamplingPolicy | None = None,
) -> KBOTemporalGraphDataset:
    """Build or verify a v7 immutable primitive archive.

    Source-record versions are stored once in season shards.  Label shards hold
    only record references, and query shards hold only game/team/time topology.
    """

    sampling = policy or TemporalSamplingPolicy()
    first = _as_day(start_day)
    last = _as_day(end_day)
    if first is not None and last is not None and first > last:
        raise ValueError("start_day must not be after end_day")
    snapshot = knowledge_at or datetime.now(timezone.utc)
    if snapshot.utcoffset() is None:
        raise ValueError("knowledge_at must be timezone-aware")
    database = Path(database_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    records, provenance, source_fingerprint = _read_records(database, snapshot)
    labels_by_day = _label_records(records, snapshot)
    selected_days = tuple(
        sorted(
            day
            for day, rows in labels_by_day.items()
            if any(record.kind == "game" for record in rows)
            and (first is None or day >= first)
            and (last is None or day <= last)
        )
    )
    if not selected_days:
        raise ValueError("no final KBO games in the requested temporal archive range")
    earliest_record_day = selected_days[0] - timedelta(days=sampling.lookback_days)
    latest_record_day = selected_days[-1]
    stored_records = tuple(
        record for record in records if earliest_record_day <= record.day <= latest_record_day
    )
    record_keys = [_record_key(record) for record in stored_records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("source record versions do not have unique immutable keys")
    stored_key_set = set(record_keys)

    labels: dict[int, list[tuple[str, str]]] = defaultdict(list)
    queries: dict[int, list[_QueryDescriptor]] = defaultdict(list)
    day_entries: list[dict[str, Any]] = []
    for day in selected_days:
        rows = labels_by_day[day]
        references = [_record_key(record) for record in rows]
        if not set(references) <= stored_key_set:
            raise ValueError("selected label is absent from the stored record interval")
        labels[day.year].extend((day.isoformat(), reference) for reference in references)
        game_rows = [record for record in rows if record.kind == "game"]
        for record in game_rows:
            data = record.data
            queries[day.year].append(
                _QueryDescriptor(
                    day,
                    str(data["game_id"]),
                    str(data["home_team_id"]),
                    str(data["away_team_id"]),
                )
            )
        game_ids = {record.entity for record in game_rows}
        pa_game_ids = {
            str(record.data["game_id"]) for record in rows if record.kind == "pa"
        }
        box_game_ids = {
            str(record.data["game_id"])
            for record in rows
            if record.kind.startswith("box_")
        }
        raw_box_audit = _box_label_audit(rows)
        day_entries.append(
            {
                "day": day.isoformat(),
                "query_games": len(game_rows),
                "label_references": len(references),
                # Satisfy the established trainer summary contract without
                # materializing or duplicating a daily graph in this archive.
                "games": len(game_rows),
                "games_with_pa": len(game_ids & pa_game_ids),
                "game_only_games": len(game_ids - pa_game_ids - box_game_ids),
                "observed_completed_pa": sum(record.kind == "pa" for record in rows),
                **raw_box_audit,
                **_target_coverage(rows),
                "raw_archive_boxscore": raw_box_audit,
            }
        )

    build_config = {
        "dataset_version": TEMPORAL_GRAPH_DATASET_VERSION,
        "graph_schema": TEMPORAL_GRAPH_SCHEMA,
        "materialization_contract_version": TEMPORAL_MATERIALIZATION_CONTRACT_VERSION,
        "day_summary_contract": "trainer_split_summary_v1",
        "sampling_policy": sampling.to_dict(),
        "cutoff_timezone": "Asia/Seoul",
        "cutoff_time": "00:00:00",
        "source_fingerprint": source_fingerprint,
        "date_start": selected_days[0].isoformat(),
        "date_end": selected_days[-1].isoformat(),
        "record_keys": record_keys,
        "label_references": [item for season in sorted(labels) for item in labels[season]],
        "query_descriptors": [
            _descriptor_fingerprint_row(item)
            for season in sorted(queries)
            for item in sorted(queries[season], key=lambda value: (value.day, value.game_id))
        ],
    }
    build_fingerprint = _json_sha256(build_config)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (output / "manifest.json").is_file():
        existing = KBOTemporalGraphDataset(output)
        if existing.manifest.get("build_fingerprint") != build_fingerprint:
            raise FileExistsError("immutable temporal archive differs from the requested build")
        return existing
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("temporal archive output directory is not empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        record_entries = _write_record_shards(staging, stored_records)
        label_entries = _write_label_shards(staging, labels)
        query_entries = _write_query_shards(staging, queries)
        artifact_fingerprint = _artifact_fingerprint(
            build_fingerprint=build_fingerprint,
            record_shards=record_entries,
            label_shards=label_entries,
            query_shards=query_entries,
        )
        manifest = {
            "dataset_version": TEMPORAL_GRAPH_DATASET_VERSION,
            "graph_schema": TEMPORAL_GRAPH_SCHEMA,
            "materialization_contract_version": TEMPORAL_MATERIALIZATION_CONTRACT_VERSION,
            "day_summary_contract": "trainer_split_summary_v1",
            "fingerprint": artifact_fingerprint,
            "build_fingerprint": build_fingerprint,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "label_snapshot_at": snapshot.astimezone(timezone.utc).isoformat(),
            "date_start": selected_days[0].isoformat(),
            "date_end": selected_days[-1].isoformat(),
            "sampling_policy": sampling.to_dict(),
            "sampling_policy_fingerprint": sampling.fingerprint,
            "temporal_batching": {
                "max_edges_per_batch": 200_000,
                "max_nodes_per_batch": 100_000,
                "max_days_per_batch": 8,
                "scope": "execution recommendation; excluded from sampling fingerprint",
            },
            "source_fingerprint": source_fingerprint,
            "source_provenance": provenance,
            "record_count": len(stored_records),
            "record_shards": record_entries,
            "label_shards": label_entries,
            "query_shards": query_entries,
            "days": day_entries,
            "node_feature_dims": {
                **{key: len(value) for key, value in NODE_FEATURE_NAMES.items()},
                "game": len(VNEXT_GAME_FEATURE_NAMES),
            },
            "role_feature_dims": {key: len(value) for key, value in ROLE_FEATURE_NAMES.items()},
            "player_role_feature_dims": {
                key: len(value) for key, value in ROLE_FEATURE_NAMES.items()
            },
            "boxscore_feature_dims": BOX_FEATURE_DIMS,
            "route_feature_dims": {
                key: len(value) for key, value in TEMPORAL_ROUTE_FEATURE_NAMES.items()
            },
            "feature_names": {
                "nodes": {**NODE_FEATURE_NAMES, "game": VNEXT_GAME_FEATURE_NAMES},
                "roles": ROLE_FEATURE_NAMES,
                "routes": TEMPORAL_ROUTE_FEATURE_NAMES,
            },
            "route_metadata": TEMPORAL_ROUTE_METADATA,
            "pa_context_dim": len(PA_CONTEXT_FEATURE_NAMES),
            "pa_context_feature_names": PA_CONTEXT_FEATURE_NAMES,
            "pa_target_classes": NEURAL_PA_OUTCOMES,
            "match_target_classes": ("away_win", "draw", "home_win"),
            "archive_policy": {
                "primitive_storage": (
                    "each selected source-record version occurs in one season shard"
                ),
                "labels": "season shards contain immutable record references only",
                "query_topology": "game_id/home_team_id/away_team_id only",
                "history": (
                    "event day before cutoff; available and valid at cutoff; bounded lookback"
                ),
                "current_game": "exactly two team-game edges; no current participant or PA edges",
                "same_day": "current labels are resolved only after historical topology selection",
                "daily_graph_files": False,
            },
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return KBOTemporalGraphDataset(output)


def build_kbo_temporal_sample_index(
    dataset_or_directory: KBOTemporalGraphDataset | str | Path,
    output_path: str | Path | None = None,
    *,
    end_day: date | str | None = None,
    label_year_ceiling: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Lazily index only an explicitly unsealed date range.

    Either ``end_day`` or ``label_year_ceiling`` is required.  A caller cannot
    accidentally inspect every label year (including a held-out test season).
    """

    if end_day is None and label_year_ceiling is None:
        raise ValueError("end_day or label_year_ceiling is required to keep held-out labels sealed")
    directory = (
        dataset_or_directory.directory
        if isinstance(dataset_or_directory, KBOTemporalGraphDataset)
        else Path(dataset_or_directory)
    )
    dataset = KBOTemporalGraphDataset(directory, label_year_ceiling=label_year_ceiling)
    last = _as_day(end_day)
    entries: list[dict[str, Any]] = []
    days = tuple(day for day in dataset.days() if last is None or day <= last)
    if not days:
        raise ValueError("no unsealed temporal graph days selected for sample index")
    for index, day in enumerate(days, start=1):
        graph = dataset.load_day(day)
        node_counts = {
            "player": len(graph.player_ids),
            "team": len(graph.team_ids),
            "game": len(graph.game_ids),
        }
        edge_counts = {name: len(columns["source_index"]) for name, columns in graph.routes.items()}
        entry: dict[str, Any] = {
            "day": day.isoformat(),
            "sample_nodes": node_counts,
            "sample_edges": edge_counts,
            "sample_fingerprint": _sample_fingerprint(graph),
            "sample_node_total": sum(node_counts.values()),
            "sample_edge_total": sum(edge_counts.values()),
        }
        entries.append(entry)
        if progress is not None and (index == len(days) or index % 100 == 0):
            progress(f"temporal sample index {index}/{len(days)}")
    report = {
        "schema_version": TEMPORAL_SAMPLE_INDEX_SCHEMA_VERSION,
        "sample_fingerprint_scope": "all_materialized_arrays_v2",
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "sampling_policy": dataset.policy.to_dict(),
        "sampling_policy_fingerprint": dataset.policy.fingerprint,
        "label_year_ceiling": label_year_ceiling,
        "held_out_labels_loaded": False,
        "date_start": days[0].isoformat(),
        "date_end": days[-1].isoformat(),
        "day_count": len(days),
        "max_sample_node_total": max(entry["sample_node_total"] for entry in entries),
        "max_sample_edge_total": max(entry["sample_edge_total"] for entry in entries),
        "days": entries,
        "fingerprint": _json_sha256(entries),
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else dataset.directory / "sample_index.json"
    )
    _write_json_atomic(destination, report)
    return report


def _select_topology(
    history: _History,
    descriptors: Sequence[_QueryDescriptor],
    policy: TemporalSamplingPolicy,
) -> _SelectedTopology:
    active = tuple(history.active.values())
    game_records = dict(history.games)
    seed_teams = {
        team
        for descriptor in descriptors
        for team in (descriptor.home_team_id, descriptor.away_team_id)
    }
    games_by_team: dict[str, list[str]] = defaultdict(list)
    for game_id, record in game_records.items():
        games_by_team[str(record.data["home_team_id"])].append(game_id)
        games_by_team[str(record.data["away_team_id"])].append(game_id)

    def newest(game_ids: Sequence[str], limit: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(game_ids),
                key=lambda game_id: (game_records[game_id].event_at, game_id),
                reverse=True,
            )[:limit]
        )

    seed_games = {
        game_id
        for team in sorted(seed_teams)
        for game_id in newest(games_by_team.get(team, ()), policy.max_games_per_seed_team)
    }
    participants_by_game: dict[str, set[str]] = defaultdict(set)
    games_by_player: dict[str, set[str]] = defaultdict(set)
    active_pa: list[_Record] = []
    for record in active:
        data = record.data
        game_id = str(data.get("game_id", ""))
        if game_id not in game_records:
            continue
        players: tuple[str, ...]
        if record.kind == "pa":
            active_pa.append(record)
            players = (str(data["batter_id"]), str(data["pitcher_id"]))
        elif record.kind.startswith("box_"):
            players = (_box_history_id(data),)
        else:
            continue
        participants_by_game[game_id].update(players)
        for player in players:
            games_by_player[player].add(game_id)
    selected_players = {
        player for game_id in seed_games for player in participants_by_game.get(game_id, ())
    }
    expanded_games = set(seed_games)
    for player in sorted(selected_players):
        expanded_games.update(newest(tuple(games_by_player[player]), policy.max_games_per_player))
    bounded_games = newest(tuple(expanded_games), policy.max_historical_games_total)
    selected_games = tuple(
        sorted(
            bounded_games,
            key=lambda game_id: (game_records[game_id].event_at, game_id),
        )
    )
    selected_set = set(selected_games)
    return _SelectedTopology(
        historical_games=selected_games,
        game_records={game_id: game_records[game_id] for game_id in selected_games},
        active_plate_appearances=tuple(
            sorted(
                (record for record in active_pa if record.data["game_id"] in selected_set),
                key=lambda record: (record.event_at, record.row_id),
            )
        ),
        active_records=tuple(
            record
            for record in active
            if str(record.data.get("game_id", "")) in selected_set
            and (record.kind == "pa" or record.kind.startswith("box_"))
        ),
    )


def _materialize_graph(
    day: date,
    history: _History,
    topology: _SelectedTopology,
    descriptors: Sequence[_QueryDescriptor],
    labels: Sequence[_Record],
    policy: TemporalSamplingPolicy,
) -> GraphDay:
    cutoff = datetime.combine(day, time.min, tzinfo=_KST).timestamp()
    historical_game_ids = set(topology.historical_games)
    current_game_ids = {descriptor.game_id for descriptor in descriptors}
    if historical_game_ids & current_game_ids:
        raise ValueError("current query game appeared in historical topology")

    common = _historical_player_game_aggregates(topology.active_records)
    historical_players = {player for aggregates in common.values() for player, _ in aggregates}
    historical_players.update(
        str(record.data[field])
        for record in topology.active_plate_appearances
        for field in ("batter_id", "pitcher_id")
    )
    historical_teams = {
        str(record.data[field])
        for record in topology.game_records.values()
        for field in ("home_team_id", "away_team_id")
    }
    historical_teams.update(
        team
        for descriptor in descriptors
        for team in (descriptor.home_team_id, descriptor.away_team_id)
    )
    # Only now may current labels expose conditional PA/box query players.
    label_boxes = _label_boxes(list(labels))
    query_players = {
        str(record.data[field])
        for record in labels
        if record.kind == "pa"
        for field in ("batter_id", "pitcher_id")
    }
    query_players.update(str(record.data["player_id"]) for record in label_boxes)
    player_ids = tuple(sorted(historical_players)) + tuple(
        sorted(query_players - historical_players)
    )
    team_ids = tuple(sorted(historical_teams))
    game_ids = tuple(sorted(historical_game_ids)) + tuple(sorted(current_game_ids))
    players = {value: index for index, value in enumerate(player_ids)}
    teams = {value: index for index, value in enumerate(team_ids)}
    games = {value: index for index, value in enumerate(game_ids)}

    descriptor_by_game = {descriptor.game_id: descriptor for descriptor in descriptors}
    if len(descriptor_by_game) != len(descriptors):
        raise ValueError("duplicate current query game descriptor")
    for record in labels:
        data = record.data
        game_id = str(data["game_id"])
        descriptor = descriptor_by_game.get(game_id)
        if descriptor is None:
            raise ValueError("current label has no score-free query descriptor")
        if record.kind == "game" and (str(data["home_team_id"]), str(data["away_team_id"])) != (
            descriptor.home_team_id,
            descriptor.away_team_id,
        ):
            raise ValueError("current game label conflicts with query topology descriptor")
        if record.kind == "pa" and {
            str(data["batting_team_id"]),
            str(data["fielding_team_id"]),
        } != {descriptor.home_team_id, descriptor.away_team_id}:
            raise ValueError("current PA label teams conflict with query topology descriptor")
        if record.kind.startswith("box_") and {
            str(data["team_id"]),
            str(data["opponent_team_id"]),
        } != {descriptor.home_team_id, descriptor.away_team_id}:
            raise ValueError("current box label teams conflict with query topology descriptor")

    box_queries = {(row.data["player_id"], row.data["role"]): row.data for row in label_boxes}

    def role_history(player: str, role: str) -> _Aggregate | None:
        mapping = cast(dict[str, _Aggregate], getattr(history, role))
        aggregate = mapping.get(player)
        query = box_queries.get((player, role))
        if aggregate is None and query is not None:
            aggregate = mapping.get(_box_history_id(query))
        return aggregate

    batting = np.asarray(
        [
            _role_features(role_history(player, "batting"), cutoff, policy.lookback_days)
            for player in player_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    pitching = np.asarray(
        [
            _role_features(role_history(player, "pitching"), cutoff, policy.lookback_days)
            for player in player_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    arrays: dict[str, Array] = {
        "player_features": np.column_stack(
            (batting[:, 0], pitching[:, 0], batting[:, -1], pitching[:, -1])
        ).astype(np.float32, copy=False),
        "player_batting_features": batting,
        "player_pitching_features": pitching,
        "team_features": np.asarray(
            [
                _team_features(history.teams.get(team), cutoff, policy.lookback_days)
                for team in team_ids
            ],
            dtype=np.float32,
        ).reshape(-1, 8),
        "game_features": np.asarray(
            [
                _temporal_game_features(
                    game_id,
                    current=game_id in current_game_ids,
                    historical_record=topology.game_records.get(game_id),
                    descriptor=descriptor_by_game.get(game_id),
                    cutoff=cutoff,
                    lookback_days=policy.lookback_days,
                )
                for game_id in game_ids
            ],
            dtype=np.float32,
        ).reshape(-1, 4),
    }
    for role, dimension in BOX_FEATURE_DIMS.items():
        personal = cast(dict[str, _Aggregate], getattr(history, f"box_{role}"))
        team_prior = cast(dict[str, _Aggregate], getattr(history, f"box_team_{role}"))
        player_features = []
        for player in player_ids:
            aggregate = personal.get(player)
            query = box_queries.get((player, role))
            if aggregate is None and query is not None:
                aggregate = personal.get(_box_history_id(query)) or team_prior.get(query["team_id"])
            player_features.append(
                _box_features(aggregate, cutoff, policy.lookback_days, dimension)
            )
        arrays[f"player_box_{role}_features"] = np.asarray(
            player_features, dtype=np.float32
        ).reshape(-1, dimension)
        arrays[f"team_box_{role}_features"] = np.asarray(
            [
                _box_features(team_prior.get(team), cutoff, policy.lookback_days, dimension)
                for team in team_ids
            ],
            dtype=np.float32,
        ).reshape(-1, dimension)

    _add_player_game_routes(arrays, common, players, games, cutoff, policy.lookback_days)
    _add_raw_pa_route(arrays, topology.active_plate_appearances, players, cutoff)
    _add_team_game_route(arrays, topology.game_records, descriptors, teams, games, cutoff)
    current_games = [record for record in labels if record.kind == "game"]
    current_pas = [record for record in labels if record.kind == "pa"]
    arrays.update(_queries(current_games, current_pas, players, teams, label_boxes, games))
    return _TemporalGraphDay(day, player_ids, team_ids, arrays, game_ids)


def _historical_player_game_aggregates(
    active_records: Sequence[_Record],
) -> dict[str, dict[tuple[str, str], _Aggregate]]:
    grouped: dict[tuple[str, str, str], list[_Record]] = defaultdict(list)
    for record in active_records:
        data = record.data
        if record.kind == "pa":
            grouped[(data["game_id"], data["batting_team_id"], "batting")].append(record)
            grouped[(data["game_id"], data["fielding_team_id"], "pitching")].append(record)
        elif record.kind.startswith("box_"):
            grouped[(data["game_id"], data["team_id"], data["role"])].append(record)
    result: dict[str, dict[tuple[str, str], _Aggregate]] = {
        "batting": {},
        "pitching": {},
    }
    for (_, _, role), records in sorted(grouped.items()):
        for record in _common_box_records(records, role):
            player = _box_history_id(record.data)
            game_id = str(record.data["game_id"])
            values = _complete_legacy_box_values(record.data)
            if values is None:
                values = np.zeros(7, dtype=np.float64)
            aggregate = result[role].setdefault((player, game_id), _Aggregate())
            aggregate.update(record, values, True)
    return result


def _add_player_game_routes(
    arrays: dict[str, Array],
    common: Mapping[str, Mapping[tuple[str, str], _Aggregate]],
    players: Mapping[str, int],
    games: Mapping[str, int],
    cutoff: float,
    lookback_days: int,
) -> None:
    for role, route in (
        ("batting", "batter_game_event"),
        ("pitching", "pitcher_game_event"),
    ):
        rows = [
            (player, game_id, aggregate)
            for (player, game_id), aggregate in common[role].items()
            if player in players and game_id in games
        ]
        rows.sort(key=lambda row: (row[1], row[0]))
        prefix = route + "__"
        arrays[prefix + "source_index"] = np.asarray(
            [players[player] for player, _, _ in rows], dtype=np.int64
        )
        arrays[prefix + "destination_index"] = np.asarray(
            [games[game_id] for _, game_id, _ in rows], dtype=np.int64
        )
        arrays[prefix + "event_features"] = np.asarray(
            [_route_features(aggregate, cutoff, lookback_days, route) for _, _, aggregate in rows],
            dtype=np.float32,
        ).reshape(-1, 6)
        arrays[prefix + "event_age_seconds"] = np.asarray(
            [cutoff - aggregate.last_event for _, _, aggregate in rows], dtype=np.float32
        )
        arrays[prefix + "publication_delay_seconds"] = np.asarray(
            [max(0.0, aggregate.last_available - aggregate.last_event) for _, _, aggregate in rows],
            dtype=np.float32,
        )
        arrays[prefix + "weights"] = np.ones(len(rows), dtype=np.float32)


def _add_raw_pa_route(
    arrays: dict[str, Array],
    records: Sequence[_Record],
    players: Mapping[str, int],
    cutoff: float,
) -> None:
    rows = [
        record
        for record in records
        if record.data["batter_id"] in players and record.data["pitcher_id"] in players
    ]
    rows.sort(key=lambda record: (record.event_at, record.row_id))
    prefix = "batter_pa_pitcher_event__"
    arrays[prefix + "source_index"] = np.asarray(
        [players[str(record.data["batter_id"])] for record in rows], dtype=np.int64
    )
    arrays[prefix + "destination_index"] = np.asarray(
        [players[str(record.data["pitcher_id"])] for record in rows], dtype=np.int64
    )
    arrays[prefix + "event_features"] = np.asarray(
        [_raw_pa_features(record) for record in rows], dtype=np.float32
    ).reshape(-1, 17)
    arrays[prefix + "event_age_seconds"] = np.asarray(
        [cutoff - record.event_at.timestamp() for record in rows], dtype=np.float32
    )
    arrays[prefix + "publication_delay_seconds"] = np.asarray(
        [max(0.0, (record.available_at - record.event_at).total_seconds()) for record in rows],
        dtype=np.float32,
    )
    arrays[prefix + "weights"] = np.ones(len(rows), dtype=np.float32)


def _raw_pa_features(record: _Record) -> list[float]:
    row = record.data
    runners = str(row["runners_before"])
    if len(runners) != 3 or set(runners) - {"0", "1"}:
        raise ValueError("invalid historical runners_before bitmap")
    complete = row.get("transition_complete", True)
    home_before = row["home_score_before"] if complete else None
    away_before = row["away_score_before"] if complete else None
    values = record.values
    return [
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]) / 4.0,
        float(values[4]),
        float(values[5]),
        float(values[6]),
        float(row["inning"]) / 12.0,
        float(row["half_inning"] == "bottom"),
        float(row["outs_before"]) / 2.0,
        *[float(int(bit)) for bit in runners],
        float(home_before or 0) / 10.0,
        float(away_before or 0) / 10.0,
        float(home_before is None),
        float(away_before is None),
    ]


def _add_team_game_route(
    arrays: dict[str, Array],
    historical: Mapping[str, _Record],
    current: Sequence[_QueryDescriptor],
    teams: Mapping[str, int],
    games: Mapping[str, int],
    cutoff: float,
) -> None:
    rows: list[tuple[str, str, bool, bool, datetime | None, datetime | None]] = []
    for game_id, record in historical.items():
        rows.extend(
            (
                (
                    str(record.data["home_team_id"]),
                    game_id,
                    True,
                    False,
                    record.event_at,
                    record.available_at,
                ),
                (
                    str(record.data["away_team_id"]),
                    game_id,
                    False,
                    False,
                    record.event_at,
                    record.available_at,
                ),
            )
        )
    for descriptor in current:
        rows.extend(
            (
                (descriptor.home_team_id, descriptor.game_id, True, True, None, None),
                (descriptor.away_team_id, descriptor.game_id, False, True, None, None),
            )
        )
    rows.sort(key=lambda row: (row[1], not row[2], row[0]))
    prefix = "team_game_event__"
    arrays[prefix + "source_index"] = np.asarray([teams[team] for team, *_ in rows], dtype=np.int64)
    arrays[prefix + "destination_index"] = np.asarray(
        [games[game_id] for _, game_id, *_ in rows], dtype=np.int64
    )
    arrays[prefix + "event_features"] = np.asarray(
        [
            [float(is_current), float(home), float(not home), float(not is_current)]
            for _, _, home, is_current, _, _ in rows
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
    arrays[prefix + "event_age_seconds"] = np.asarray(
        [0.0 if event is None else cutoff - event.timestamp() for *_, event, _ in rows],
        dtype=np.float32,
    )
    arrays[prefix + "publication_delay_seconds"] = np.asarray(
        [
            0.0
            if event is None or available is None
            else max(0.0, (available - event).total_seconds())
            for *_, event, available in rows
        ],
        dtype=np.float32,
    )
    arrays[prefix + "weights"] = np.ones(len(rows), dtype=np.float32)


def _temporal_game_features(
    game_id: str,
    *,
    current: bool,
    historical_record: _Record | None,
    descriptor: _QueryDescriptor | None,
    cutoff: float,
    lookback_days: int,
) -> list[float]:
    scheduled = None
    if not current and historical_record is not None:
        value = historical_record.data.get("scheduled_start")
        scheduled = value if isinstance(value, datetime) else None
    scheduled_fraction = 0.0
    if scheduled is not None:
        local = scheduled.astimezone(_KST)
        scheduled_fraction = (local.hour * 3600 + local.minute * 60 + local.second) / 86400
    if current:
        return [1.0, 0.0, 0.0, scheduled_fraction]
    if historical_record is None:
        raise ValueError(f"historical game {game_id!r} lacks a selected record")
    age = max(0.0, cutoff - historical_record.event_at.timestamp()) / 86400.0
    return [0.0, 1.0, min(age / lookback_days, 1.0), scheduled_fraction]


def _validate_temporal_graph(graph: GraphDay) -> None:
    if (
        len(set(graph.player_ids)) != len(graph.player_ids)
        or len(set(graph.team_ids)) != len(graph.team_ids)
        or len(set(graph.game_ids)) != len(graph.game_ids)
    ):
        raise ValueError("duplicate temporal graph node IDs")
    node_sizes = {
        "player": len(graph.player_ids),
        "team": len(graph.team_ids),
        "game": len(graph.game_ids),
    }
    for name, array in graph.arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {name}")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"non-finite temporal graph array: {name}")
    for route, metadata in TEMPORAL_ROUTE_METADATA.items():
        columns = graph.routes[route]
        count = len(columns["source_index"])
        source_limit = node_sizes[str(metadata["source_type"])]
        destination_limit = node_sizes[str(metadata["destination_type"])]
        for name, limit in (
            ("source_index", source_limit),
            ("destination_index", destination_limit),
        ):
            values = columns[name]
            if values.shape != (count,) or np.any(values < 0) or np.any(values >= limit):
                raise ValueError(f"invalid temporal route {route} {name}")
        if columns["event_features"].shape != (count, len(TEMPORAL_ROUTE_FEATURE_NAMES[route])):
            raise ValueError(f"invalid temporal route features: {route}")
        if np.any(columns["event_age_seconds"] < columns["publication_delay_seconds"] - 1e-6):
            raise ValueError(f"temporal route is unavailable at cutoff: {route}")
    current_games = set(np.asarray(graph.arrays["match_game_index"], dtype=np.int64).tolist())
    for route in ("batter_game_event", "pitcher_game_event"):
        if current_games & set(graph.routes[route]["destination_index"].tolist()):
            raise ValueError("current query game has a forbidden participant edge")
    team_route = graph.routes["team_game_event"]
    features = team_route["event_features"]
    for game_index in current_games:
        selected = team_route["destination_index"] == game_index
        if (
            int(selected.sum()) != 2
            or not np.all(features[selected, 0] == 1)
            or not np.all(features[selected, 3] == 0)
        ):
            raise ValueError("current query game must have exactly two score-free team edges")


def _write_record_shards(directory: Path, records: Sequence[_Record]) -> list[dict[str, Any]]:
    grouped: dict[int, list[_Record]] = defaultdict(list)
    for record in records:
        grouped[record.day.year].append(record)
    entries = []
    for season in sorted(grouped):
        selected = sorted(
            grouped[season],
            key=lambda record: (record.day, record.kind, record.entity, record.rank),
        )
        path = directory / "records" / f"{season}.npz"
        _write_npz_atomic(path, _encode_record_shard(selected))
        entries.append(_shard_entry(directory, path, season, len(selected)))
    return entries


def _encode_record_shard(records: Sequence[_Record]) -> dict[str, Array]:
    box_width = max((len(record.box_values) for record in records), default=0)
    padded = np.zeros((len(records), box_width), dtype=np.float64)
    lengths = np.zeros(len(records), dtype=np.int16)
    for index, record in enumerate(records):
        length = len(record.box_values)
        lengths[index] = length
        if length:
            padded[index, :length] = record.box_values
    string_columns: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for name, value in (
            ("record_key", _record_key(record)),
            ("kind", record.kind),
            ("entity", record.entity),
            ("row_id", record.row_id),
            ("day", record.day.isoformat()),
            ("event_at", record.event_at.isoformat()),
            ("available_at", record.available_at.isoformat()),
            ("ingested_at", record.ingested_at.isoformat()),
            ("valid_from", record.valid_from.isoformat()),
            ("valid_to", record.valid_to.isoformat() if record.valid_to is not None else ""),
            ("source_id", record.source_id),
            ("digest", f"{record.digest:064x}"),
            (
                "data_json",
                json.dumps(
                    record.data,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                ),
            ),
        ):
            string_columns[name].append(value)
    return {
        **{name: np.asarray(values, dtype=np.str_) for name, values in string_columns.items()},
        "values": np.asarray([record.values for record in records], dtype=np.float64).reshape(
            -1, 7
        ),
        "box_values": padded,
        "box_value_lengths": lengths,
    }


def _decode_record_shard(arrays: Mapping[str, Array]) -> tuple[_Record, ...]:
    names = (
        "record_key",
        "kind",
        "entity",
        "row_id",
        "day",
        "event_at",
        "available_at",
        "ingested_at",
        "valid_from",
        "valid_to",
        "source_id",
        "digest",
        "data_json",
    )
    columns = {name: _string_column(arrays, name) for name in names}
    count = len(columns["record_key"])
    if any(len(column) != count for column in columns.values()):
        raise ValueError("record shard string columns disagree")
    values = np.asarray(arrays.get("values"))
    box_values = np.asarray(arrays.get("box_values"))
    lengths = np.asarray(arrays.get("box_value_lengths"))
    if (
        values.shape != (count, 7)
        or box_values.ndim != 2
        or box_values.shape[0] != count
        or lengths.shape != (count,)
    ):
        raise ValueError("record shard numeric columns disagree")
    records = []
    for index in range(count):
        data = json.loads(columns["data_json"][index])
        if not isinstance(data, dict):
            raise ValueError("record data_json must decode to an object")
        for name in (
            "scheduled_start",
            "event_at",
            "available_at",
            "ingested_at",
            "valid_from",
            "valid_to",
        ):
            if isinstance(data.get(name), str):
                data[name] = _parse_datetime(data[name])
        length = int(lengths[index])
        if not 0 <= length <= box_values.shape[1]:
            raise ValueError("invalid box_value_lengths entry")
        record = _Record(
            kind=columns["kind"][index],
            entity=columns["entity"][index],
            row_id=columns["row_id"][index],
            day=date.fromisoformat(columns["day"][index]),
            event_at=_parse_datetime(columns["event_at"][index]),
            available_at=_parse_datetime(columns["available_at"][index]),
            ingested_at=_parse_datetime(columns["ingested_at"][index]),
            valid_from=_parse_datetime(columns["valid_from"][index]),
            valid_to=_parse_datetime(columns["valid_to"][index])
            if columns["valid_to"][index]
            else None,
            source_id=columns["source_id"][index],
            data=data,
            digest=int(columns["digest"][index], 16),
            values=np.asarray(values[index], dtype=np.float64).copy(),
            box_values=np.asarray(box_values[index, :length], dtype=np.float64).copy(),
        )
        if _record_key(record) != columns["record_key"][index]:
            raise ValueError("record shard immutable key mismatch")
        records.append(record)
    return tuple(records)


def _write_label_shards(
    directory: Path, labels: Mapping[int, Sequence[tuple[str, str]]]
) -> list[dict[str, Any]]:
    entries = []
    for season in sorted(labels):
        rows = sorted(labels[season])
        path = directory / "labels" / f"{season}.npz"
        _write_npz_atomic(
            path,
            {
                "day": np.asarray([row[0] for row in rows], dtype=np.str_),
                "record_key": np.asarray([row[1] for row in rows], dtype=np.str_),
            },
        )
        entries.append(_shard_entry(directory, path, season, len(rows)))
    return entries


def _write_query_shards(
    directory: Path, queries: Mapping[int, Sequence[_QueryDescriptor]]
) -> list[dict[str, Any]]:
    entries = []
    for season in sorted(queries):
        rows = sorted(queries[season], key=lambda value: (value.day, value.game_id))
        path = directory / "queries" / f"{season}.npz"
        _write_npz_atomic(
            path,
            {
                "day": np.asarray([row.day.isoformat() for row in rows], dtype=np.str_),
                "game_id": np.asarray([row.game_id for row in rows], dtype=np.str_),
                "home_team_id": np.asarray([row.home_team_id for row in rows], dtype=np.str_),
                "away_team_id": np.asarray([row.away_team_id for row in rows], dtype=np.str_),
            },
        )
        entries.append(_shard_entry(directory, path, season, len(rows)))
    return entries


def _write_npz_atomic(path: Path, arrays: Mapping[str, Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise ValueError(f"pickle/object arrays are forbidden in temporal archive: {name}")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _read_shard(directory: Path, entry: Mapping[str, Any]) -> dict[str, Array]:
    path = (directory / str(entry["file"])).resolve()
    if directory not in path.parents or not path.is_file():
        raise ValueError("temporal shard path escapes or is missing from archive")
    if _sha256_file(path) != entry["sha256"]:
        raise ValueError(f"temporal shard checksum mismatch: {entry['file']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("temporal shard contains a forbidden object array")
    return arrays


def _shard_entry(directory: Path, path: Path, season: int, count: int) -> dict[str, Any]:
    return {
        "season": season,
        "file": path.relative_to(directory).as_posix(),
        "sha256": _sha256_file(path),
        "rows": count,
    }


def _index_shards(entries: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in entries:
        season = raw.get("season")
        if isinstance(season, bool) or not isinstance(season, int) or season in result:
            raise ValueError("temporal archive contains invalid or duplicate season shards")
        result[season] = dict(raw)
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("dataset_version") != TEMPORAL_GRAPH_DATASET_VERSION
        or manifest.get("graph_schema") != TEMPORAL_GRAPH_SCHEMA
    ):
        raise ValueError("unsupported temporal KBO graph archive")
    if manifest.get("day_summary_contract") != "trainer_split_summary_v1":
        raise ValueError("temporal archive lacks the trainer day-summary contract")
    if (
        manifest.get("materialization_contract_version")
        != TEMPORAL_MATERIALIZATION_CONTRACT_VERSION
    ):
        raise ValueError("temporal archive materialization contract differs")
    if set(cast(Mapping[str, Any], manifest.get("node_feature_dims", {}))) != {
        "player",
        "team",
        "game",
    }:
        raise ValueError("temporal archive must declare player/team/game nodes")
    expected_dims = {name: len(values) for name, values in TEMPORAL_ROUTE_FEATURE_NAMES.items()}
    if manifest.get("route_feature_dims") != expected_dims:
        raise ValueError("temporal archive route feature contract differs from v7")
    TemporalSamplingPolicy(**cast(dict[str, Any], manifest.get("sampling_policy", {})))
    days = manifest.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("temporal archive days must be a non-empty list")
    identifiers = [entry.get("day") for entry in days if isinstance(entry, dict)]
    if len(identifiers) != len(days) or len(set(identifiers)) != len(identifiers):
        raise ValueError("temporal archive contains invalid or duplicate days")
    required_day_counts = {
        "query_games",
        "label_references",
        "games",
        "live_hit_queries",
        "pa_queries",
        "box_pa_queries",
        "box_pa_outcomes",
        "box_pitch_queries",
        "box_pitch_observed_counts",
    }
    for entry in days:
        assert isinstance(entry, dict)
        for name in required_day_counts:
            value = entry.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"temporal archive day {name} must be a non-negative integer")
        if entry["games"] != entry["query_games"]:
            raise ValueError("temporal archive day game counts disagree")
    shard_entries: dict[str, list[Any]] = {}
    for name in ("record_shards", "label_shards", "query_shards"):
        entries = manifest.get(name)
        if not isinstance(entries, list):
            raise ValueError(f"temporal archive {name} must be a list")
        shard_entries[name] = entries
    build_fingerprint = _require_lowercase_sha256(
        manifest.get("build_fingerprint"), "temporal archive build_fingerprint"
    )
    artifact_fingerprint = _require_lowercase_sha256(
        manifest.get("fingerprint"), "temporal archive fingerprint"
    )
    expected_artifact_fingerprint = _artifact_fingerprint(
        build_fingerprint=build_fingerprint,
        record_shards=shard_entries["record_shards"],
        label_shards=shard_entries["label_shards"],
        query_shards=shard_entries["query_shards"],
    )
    if artifact_fingerprint != expected_artifact_fingerprint:
        raise ValueError("temporal archive artifact fingerprint is inconsistent")


def _record_key(record: _Record) -> str:
    return _json_sha256(
        {
            "kind": record.kind,
            "entity": record.entity,
            "row_id": record.row_id,
            "digest": f"{record.digest:064x}",
        }
    )


def _require_lowercase_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _artifact_fingerprint(
    *,
    build_fingerprint: str,
    record_shards: Sequence[Mapping[str, Any]],
    label_shards: Sequence[Mapping[str, Any]],
    query_shards: Sequence[Mapping[str, Any]],
) -> str:
    """Bind the logical build request to the exact immutable shard artifacts."""

    return _json_sha256(
        {
            "build_fingerprint": build_fingerprint,
            "record_shards": list(record_shards),
            "label_shards": list(label_shards),
            "query_shards": list(query_shards),
        }
    )


def _descriptor_fingerprint_row(value: _QueryDescriptor) -> dict[str, Any]:
    return {
        "day": value.day.isoformat(),
        "game_id": value.game_id,
        "home_team_id": value.home_team_id,
        "away_team_id": value.away_team_id,
    }


def _sample_fingerprint(graph: GraphDay) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("player_ids", graph.player_ids),
        ("team_ids", graph.team_ids),
        ("game_ids", graph.game_ids),
    ):
        digest.update(name.encode())
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    # Bind the index to every materialized numeric array, including box-role
    # features, query endpoints/context, masks, and labels.  The index is built
    # only through the explicitly unsealed validation ceiling, so this stronger
    # integrity check cannot inspect the held-out test season.
    for name in sorted(graph.arrays):
        array = np.ascontiguousarray(graph.arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _string_column(arrays: Mapping[str, Array], name: str) -> list[str]:
    raw = np.asarray(arrays.get(name))
    if raw.ndim != 1 or raw.dtype.kind not in {"U", "S"}:
        raise ValueError(f"temporal shard {name} must be a one-dimensional string column")
    return [str(value) for value in raw.tolist()]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("temporal archive datetime must be timezone-aware")
    return parsed


def _as_day(value: date | str | None) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) else value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "KBOTemporalGraphDataset",
    "TEMPORAL_GRAPH_DATASET_VERSION",
    "TEMPORAL_GRAPH_SCHEMA",
    "TEMPORAL_MATERIALIZATION_CONTRACT_VERSION",
    "TEMPORAL_SAMPLE_INDEX_SCHEMA_VERSION",
    "TEMPORAL_ROUTE_FEATURE_NAMES",
    "TEMPORAL_ROUTE_METADATA",
    "TemporalSamplingPolicy",
    "build_kbo_temporal_archive",
    "build_kbo_temporal_sample_index",
]
