"""Daily, leakage-safe NumPy graphs from the canonical KBO database.

This is a retrospective dataset, not a claim that these files were ingested in
2023. Publication/validity and event time are gated at midnight Asia/Seoul;
ingestion is gated at the requested database knowledge snapshot. No current-day
appearance is an edge. The current-day PA context belongs only to the auxiliary
PA query, never to the pregame match or conditional player-game hit model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from numpy.typing import NDArray

from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES, neural_training_target_index
from cpv26.simulation.events import TerminalPlateAppearanceEvent

GRAPH_DATASET_VERSION = 2
_KST = ZoneInfo("Asia/Seoul")
Array = NDArray[Any]

NODE_FEATURE_NAMES = {
    "player": (
        "batting_pa_log500",
        "pitching_pa_log500",
        "batting_recency",
        "pitching_recency",
    ),
    "team": (
        "games_log90",
        "win_rate",
        "draw_rate",
        "runs_per_game_div10",
        "runs_allowed_per_game_div10",
        "run_diff_per_game_div10",
        "home_rate",
        "recency",
    ),
}
ROLE_FEATURE_NAMES = {
    "batting": (
        "pa_log500",
        "ab_per_pa",
        "hit_per_pa",
        "total_bases_per_pa_div4",
        "walk_hbp_per_pa",
        "strikeout_per_pa",
        "home_run_per_pa",
        "recency",
    ),
    "pitching": (
        "pa_faced_log500",
        "ab_per_pa",
        "hit_allowed_per_pa",
        "total_bases_per_pa_div4",
        "walk_hbp_per_pa",
        "strikeout_per_pa",
        "home_run_allowed_per_pa",
        "recency",
    ),
}
_PA_ROUTE_FEATURES = (
    "pa_log100",
    "hit_per_pa",
    "walk_hbp_per_pa",
    "strikeout_per_pa",
    "total_bases_per_pa_div4",
    "recency",
)
ROUTE_FEATURE_NAMES = {
    "batter_pa_pitcher": _PA_ROUTE_FEATURES,
    "batter_participation_team": _PA_ROUTE_FEATURES,
    "pitcher_participation_team": _PA_ROUTE_FEATURES,
    "home_team_game_away_team": (
        "games_log20",
        "home_win_rate",
        "draw_rate",
        "home_runs_per_game_div10",
        "away_runs_per_game_div10",
        "recency",
    ),
}
ROUTE_METADATA = {
    "batter_pa_pitcher": {
        "source_type": "player",
        "destination_type": "player",
        "source_role": "batting",
        "destination_role": "pitching",
        "bidirectional": True,
        "meaning": "completed historical PA, terminal pitcher (not official run responsibility)",
    },
    "batter_participation_team": {
        "source_type": "player",
        "destination_type": "team",
        "source_role": "batting",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past batting appearance for team; not an authoritative roster",
    },
    "pitcher_participation_team": {
        "source_type": "player",
        "destination_type": "team",
        "source_role": "pitching",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past pitching appearance for team; not an authoritative roster",
    },
    "home_team_game_away_team": {
        "source_type": "team",
        "destination_type": "team",
        "source_role": "shared",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "historical final game, oriented historical home to away",
    },
}
PA_CONTEXT_FEATURE_NAMES = (
    "inning_div12",
    "is_bottom",
    "outs_div2",
    "runner_on_first",
    "runner_on_second",
    "runner_on_third",
    "home_score_before_div10",
    "away_score_before_div10",
    "home_score_missing",
    "away_score_missing",
)
_ROUTE_ARRAY_FIELDS = (
    "source_index",
    "destination_index",
    "event_features",
    "event_age_seconds",
    "publication_delay_seconds",
    "weights",
)


@dataclass(frozen=True, slots=True)
class GraphDay:
    """One cutoff's shared graph and separate supervised query arrays.

    ``arrays`` is the flat, pickle-free disk contract. Mapping properties are
    conveniences for the Torch-free model collator interface.
    """

    day: date
    player_ids: tuple[str, ...]
    team_ids: tuple[str, ...]
    arrays: dict[str, Array]

    @property
    def day_id(self) -> str:
        return self.day.isoformat()

    @property
    def node_features(self) -> dict[str, Array]:
        return {"player": self.arrays["player_features"], "team": self.arrays["team_features"]}

    @property
    def role_features(self) -> dict[str, Array]:
        return {role: self.arrays[f"player_{role}_features"] for role in ROLE_FEATURE_NAMES}

    @property
    def routes(self) -> dict[str, dict[str, Array]]:
        return {
            route: {key: self.arrays[f"{route}__{key}"] for key in _ROUTE_ARRAY_FIELDS}
            for route in ROUTE_METADATA
        }

    def __getattr__(self, name: str) -> Any:
        # Query names are shared with the model collator without copying arrays.
        try:
            arrays = object.__getattribute__(self, "arrays")
            return arrays[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class KBOGraphDataset:
    """Read a completed graph-cache manifest; validate each file before loading."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        with (self.directory / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        if self.manifest.get("dataset_version") != GRAPH_DATASET_VERSION:
            raise ValueError("unsupported KBO graph dataset version")
        self._entries = {entry["day"]: entry for entry in self.manifest["days"]}
        if len(self._entries) != len(self.manifest["days"]):
            raise ValueError("duplicate graph days in manifest")

    def days(self) -> tuple[date, ...]:
        return tuple(date.fromisoformat(day) for day in sorted(self._entries))

    def load_day(self, day: date | str) -> GraphDay:
        key = day.isoformat() if isinstance(day, date) else day
        if key not in self._entries:
            raise KeyError(f"graph day not present: {key}")
        entry = self._entries[key]
        path = (self.directory / entry["file"]).resolve()
        if self.directory not in path.parents:
            raise ValueError("graph file escapes dataset directory")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"graph cache checksum mismatch: {key}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        players = tuple(str(value) for value in arrays.pop("_player_ids"))
        teams = tuple(str(value) for value in arrays.pop("_team_ids"))
        graph = GraphDay(date.fromisoformat(key), players, teams, arrays)
        _validate_graph(graph)
        return graph


@dataclass(slots=True)
class _Record:
    kind: str
    entity: str
    row_id: str
    day: date
    event_at: datetime
    available_at: datetime
    ingested_at: datetime
    valid_from: datetime
    valid_to: datetime | None
    source_id: str
    data: dict[str, Any]
    digest: int = 0
    values: Array = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    @property
    def rank(self) -> tuple[datetime, datetime, datetime, str]:
        return self.valid_from, self.available_at, self.ingested_at, self.row_id


@dataclass(slots=True)
class _Aggregate:
    values: Array = field(default_factory=lambda: np.zeros(7, dtype=np.float64))
    events: dict[str, _Record] = field(default_factory=dict)
    last_event: float = 0.0
    last_available: float = 0.0

    def update(self, record: _Record, values: Array, add: bool) -> None:
        if add:
            self.values += values
            self.events[record.row_id] = record
            self.last_event = max(self.last_event, record.event_at.timestamp())
            self.last_available = max(self.last_available, record.available_at.timestamp())
        else:
            self.values -= values
            self.events.pop(record.row_id)
            if record.event_at.timestamp() == self.last_event:
                self.last_event = max(
                    (item.event_at.timestamp() for item in self.events.values()),
                    default=0.0,
                )
            if record.available_at.timestamp() == self.last_available:
                self.last_available = max(
                    (item.available_at.timestamp() for item in self.events.values()),
                    default=0.0,
                )


class _History:
    """Incremental rolling aggregates, including publication/validity revisions."""

    def __init__(self, records: list[_Record], rolling_days: int) -> None:
        self.rolling_days = rolling_days
        self.versions: dict[tuple[str, str], list[_Record]] = defaultdict(list)
        schedule: dict[date, set[tuple[str, str]]] = defaultdict(set)
        for record in records:
            key = (record.kind, record.entity)
            self.versions[key].append(record)
            activation = max(
                record.day + timedelta(days=1),
                _first_cutoff(record.available_at),
                _first_cutoff(record.valid_from),
            )
            schedule[activation].add(key)
            schedule[record.day + timedelta(days=rolling_days + 1)].add(key)
            if record.valid_to is not None:
                schedule[_first_cutoff(record.valid_to)].add(key)
        self.schedule = sorted(schedule.items())
        self.position = 0
        self.active: dict[tuple[str, str], _Record] = {}
        self.digest = 0
        self.batting: dict[str, _Aggregate] = {}
        self.pitching: dict[str, _Aggregate] = {}
        self.teams: dict[str, _Aggregate] = {}
        self.routes: dict[str, dict[tuple[str, str], _Aggregate]] = {
            route: {} for route in ROUTE_METADATA
        }

    def advance(self, day: date) -> None:
        changed: set[tuple[str, str]] = set()
        while self.position < len(self.schedule) and self.schedule[self.position][0] <= day:
            changed.update(self.schedule[self.position][1])
            self.position += 1
        cutoff = datetime.combine(day, time.min, tzinfo=_KST)
        earliest = day - timedelta(days=self.rolling_days)
        for key in sorted(changed):
            eligible = [
                record
                for record in self.versions[key]
                if earliest <= record.day < day
                and record.available_at <= cutoff
                and record.valid_from <= cutoff
                and (record.valid_to is None or cutoff < record.valid_to)
            ]
            selected = max(eligible, key=lambda item: item.rank) if eligible else None
            previous = self.active.get(key)
            if previous is selected:
                continue
            if previous is not None:
                self._update(previous, False)
                self.digest ^= previous.digest
                del self.active[key]
            if selected is not None:
                self._update(selected, True)
                self.digest ^= selected.digest
                self.active[key] = selected

    @staticmethod
    def _update_bucket(
        mapping: dict[Any, _Aggregate],
        key: Any,
        record: _Record,
        values: Array,
        add: bool,
    ) -> None:
        aggregate = mapping.get(key)
        if aggregate is None:
            aggregate = _Aggregate()
            mapping[key] = aggregate
        aggregate.update(record, values, add)
        if not aggregate.events:
            del mapping[key]

    def _update(self, record: _Record, add: bool) -> None:
        data = record.data
        update = self._update_bucket
        if record.kind == "pa":
            batter, pitcher = data["batter_id"], data["pitcher_id"]
            update(self.batting, batter, record, record.values, add)
            update(self.pitching, pitcher, record, record.values, add)
            for route, key in (
                ("batter_pa_pitcher", (batter, pitcher)),
                ("batter_participation_team", (batter, data["batting_team_id"])),
                ("pitcher_participation_team", (pitcher, data["fielding_team_id"])),
            ):
                update(self.routes[route], key, record, record.values, add)
        else:
            home, away = data["home_team_id"], data["away_team_id"]
            home_runs, away_runs = data["home_score"], data["away_score"]
            for team, own, opponent, is_home in (
                (home, home_runs, away_runs, 1),
                (away, away_runs, home_runs, 0),
            ):
                values = np.array(
                    [1, own > opponent, own == opponent, own, opponent, own - opponent, is_home],
                    dtype=np.float64,
                )
                update(self.teams, team, record, values, add)
            update(
                self.routes["home_team_game_away_team"],
                (home, away),
                record,
                record.values,
                add,
            )


def build_kbo_graph_dataset(
    database_path: str | Path,
    output_dir: str | Path,
    *,
    rolling_days: int = 90,
    start_day: date | str | None = None,
    end_day: date | str | None = None,
    knowledge_at: datetime | None = None,
) -> KBOGraphDataset:
    """Build/reuse daily graphs; date filters NEVER truncate preceding history.

    The output contains ``manifest.json`` and ``days/YYYY-MM-DD.npz``. Sidecars
    allow interrupted builds and appended source days to reuse verified caches.
    Labels are the latest final rows visible at the database knowledge snapshot.
    The default knowledge snapshot is all rows already ingested into this DB.
    """
    if isinstance(rolling_days, bool) or not isinstance(rolling_days, int) or rolling_days < 1:
        raise ValueError("rolling_days must be a positive integer")
    if knowledge_at is not None and knowledge_at.utcoffset() is None:
        raise ValueError("knowledge_at must be timezone-aware")
    first, last = _as_date(start_day), _as_date(end_day)
    if first is not None and last is not None and first > last:
        raise ValueError("start_day must not be after end_day")
    database = Path(database_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    records, sources, source_digest = _read_records(database, knowledge_at)
    labels = _label_records(records)
    selected_days = sorted(
        day
        for day, rows in labels.items()
        if any(row.kind == "game" for row in rows)
        and (first is None or day >= first)
        and (last is None or day <= last)
    )
    if not selected_days:
        raise ValueError("no final KBO games in the requested date range")
    directory = Path(output_dir).expanduser().resolve()
    (directory / "days").mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_version": GRAPH_DATASET_VERSION,
        "rolling_days": rolling_days,
        "pa_incomplete_transition_context": "mask_pre_scores_unknown",
        "cutoff_timezone": "Asia/Seoul",
        "cutoff_time": "00:00:00",
        "knowledge_at": knowledge_at.isoformat()
        if knowledge_at is not None
        else "database_snapshot",
    }
    config_fingerprint = _json_sha256(config)
    history = _History(records, rolling_days)
    entries = []
    reused = 0
    for day in selected_days:
        history.advance(day)
        input_fingerprint = _json_sha256(
            {
                "config": config_fingerprint,
                "day": day.isoformat(),
                "history_xor_sha256": f"{history.digest:064x}",
                "history_rows": len(history.active),
                "labels": [f"{record.digest:064x}" for record in labels[day]],
            }
        )
        path = directory / "days" / f"{day.isoformat()}.npz"
        sidecar = path.with_suffix(".json")
        cached = _read_valid_cache(sidecar, path, input_fingerprint)
        if cached is not None:
            entries.append(cached)
            reused += 1
            continue
        graph = _make_graph(day, history, labels[day])
        _validate_graph(graph)
        temporary = path.with_suffix(".npz.part")
        archive_arrays: dict[str, Any] = {
            "_player_ids": np.asarray(graph.player_ids, dtype=np.str_),
            "_team_ids": np.asarray(graph.team_ids, dtype=np.str_),
            **graph.arrays,
        }
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **archive_arrays)
        os.replace(temporary, path)
        entry = {
            "day": day.isoformat(),
            "file": path.relative_to(directory).as_posix(),
            "sha256": _sha256_file(path),
            "input_fingerprint": input_fingerprint,
            "players": len(graph.player_ids),
            "teams": len(graph.team_ids),
            "games": len(graph.arrays["match_targets"]),
            "live_hit_queries": len(graph.arrays["live_hit_pa"]),
            "pa_queries": len(graph.arrays["pa_targets"]),
            "history_rows": len(history.active),
        }
        _write_json_atomic(sidecar, entry)
        entries.append(entry)
    manifest = {
        **config,
        "config_fingerprint": config_fingerprint,
        "source_fingerprint": source_digest,
        "fingerprint": _json_sha256(
            {
                "config": config_fingerprint,
                "days": [(entry["day"], entry["input_fingerprint"]) for entry in entries],
            }
        ),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "days": entries,
        "cache_reused_days": reused,
        "cache_built_days": len(entries) - reused,
        "node_feature_dims": {key: len(value) for key, value in NODE_FEATURE_NAMES.items()},
        "role_feature_dims": {key: len(value) for key, value in ROLE_FEATURE_NAMES.items()},
        "player_role_feature_dims": {key: len(value) for key, value in ROLE_FEATURE_NAMES.items()},
        "route_feature_dims": {key: len(value) for key, value in ROUTE_FEATURE_NAMES.items()},
        "feature_names": {
            "nodes": NODE_FEATURE_NAMES,
            "roles": ROLE_FEATURE_NAMES,
            "routes": ROUTE_FEATURE_NAMES,
        },
        "route_metadata": ROUTE_METADATA,
        "pa_context_dim": len(PA_CONTEXT_FEATURE_NAMES),
        "pa_context_feature_names": PA_CONTEXT_FEATURE_NAMES,
        "pa_target_classes": NEURAL_PA_OUTCOMES,
        "match_target_classes": ("away_win", "draw", "home_win"),
        "source_provenance": sources,
        "label_quality": {
            "observed_completed_pa": sum(
                row.kind == "pa" for day in selected_days for row in labels[day]
            ),
            "pa10_excluded_catcher_interference": sum(
                row.kind == "pa" and row.data["outcome"] == "catcher_interference"
                for day in selected_days
                for row in labels[day]
            ),
            "incomplete_pa_transitions": sum(
                row.kind == "pa" and not row.data.get("transition_complete", True)
                for day in selected_days
                for row in labels[day]
            ),
            "pa_context_scores_masked_incomplete_transition": sum(
                row.kind == "pa"
                and not row.data.get("transition_complete", True)
                and row.data["outcome"] != "catcher_interference"
                for day in selected_days
                for row in labels[day]
            ),
            "unlabelled_source_pa": None,
            "unlabelled_source_note": "not in canonical completed-PA table; see import report",
        },
        "policies": {
            "history": "event_date < day; available_at <= cutoff; valid at cutoff; rolling window",
            "ingestion": "retrospective database knowledge snapshot, not historical live ingestion",
            "same_day": "all same-day PA/final scores are labels only, regardless of game order",
            "current_players": "isolated query-only nodes unless they have eligible prior history",
            "team_membership": "past role-specific appearances, never today's actual roster",
            "match_queries": "home and away team only; no actual current lineup or pitcher",
            "live_hit": "conditional completed observed PA >= 1; not an appearance forecast",
            "pa_context": "current pre-PA state, separate auxiliary decoder only",
            "pa_incomplete_transition": "keep PA label; mask both pre-scores to 0 + missing flags",
            "pitcher": "last observed pitcher, not official inherited-count/run responsibility",
            "pa_pair": "credited batter/terminal pitcher; substitutions may not face final pitch",
            "scaling": "fixed constants and past-only ratios; no fitted full-sample normalization",
            "unknown_labels": "unlabelled source PAs omitted; catcher interference is not PA10",
            "simulator_ready": False,
        },
    }
    _write_json_atomic(directory / "manifest.json", manifest)
    return KBOGraphDataset(directory)


def _read_records(
    database: Path,
    knowledge_at: datetime | None,
) -> tuple[list[_Record], list[dict[str, Any]], str]:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        connection.execute("SET enable_progress_bar = false")
        connection.execute("BEGIN TRANSACTION")
        sources = _fetch_dicts(
            connection, "SELECT * FROM source_revision ORDER BY source_revision_id"
        )
        source_lookup = {row["source_revision_id"]: row for row in sources}
        records: list[_Record] = []
        queries = (
            ("game", "SELECT * FROM game WHERE game_status = 'final' ORDER BY game_id, valid_from"),
            (
                "pa",
                "SELECT * FROM observed_plate_appearance ORDER BY plate_appearance_id, valid_from",
            ),
        )
        for kind, query in queries:
            for row in _fetch_dicts(connection, query):
                if knowledge_at is not None and row["ingested_at"] > knowledge_at:
                    continue
                if kind == "game" and (row["home_score"] is None or row["away_score"] is None):
                    continue
                source = source_lookup.get(row["source_revision_id"])
                if source is None:
                    raise ValueError(f"missing provenance source {row['source_revision_id']}")
                if knowledge_at is not None and source["ingested_at"] > knowledge_at:
                    continue
                event = row["event_at"].astimezone(_KST)
                game_day = (row["scheduled_start"] if kind == "game" else event).astimezone(_KST)
                record = _Record(
                    kind=kind,
                    entity=row["game_id" if kind == "game" else "plate_appearance_id"],
                    row_id=row.get("observed_pa_row_id")
                    or row.get("game_row_id")
                    or row["game_id"] + ":" + str(row["valid_from"]),
                    day=game_day.date(),
                    event_at=event,
                    available_at=row["available_at"],
                    ingested_at=row["ingested_at"],
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    source_id=row["source_revision_id"],
                    data=row,
                )
                record.digest = int(_json_sha256({"kind": kind, "row": row, "source": source}), 16)
                if kind == "pa":
                    outcome = row["outcome"]
                    record.values = np.array(
                        [
                            1,
                            row["is_at_bat"],
                            row["is_hit"],
                            row["total_bases"],
                            outcome in ("walk", "hit_by_pitch"),
                            outcome == "strikeout",
                            outcome == "home_run",
                        ],
                        dtype=np.float64,
                    )
                else:
                    home, away = row["home_score"], row["away_score"]
                    record.values = np.array(
                        [1, home > away, home == away, home, away, home - away, 1],
                        dtype=np.float64,
                    )
                records.append(record)
        records.sort(key=lambda record: (record.kind, record.entity, record.rank))
        used_sources = {record.source_id for record in records}
        provenance = [
            {
                "source_revision_id": row["source_revision_id"],
                "source_name": row["source_name"],
                "source_locator": row["source_locator"],
                "content_sha256": row["content_sha256"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in sources
            if row["source_revision_id"] in used_sources
        ]
        source_digest = _json_sha256([f"{record.digest:064x}" for record in records])
        return records, provenance, source_digest
    finally:
        connection.close()


def _fetch_dicts(connection: Any, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _label_records(records: list[_Record]) -> dict[date, list[_Record]]:
    latest: dict[tuple[str, str], _Record] = {}
    for record in records:
        key = (record.kind, record.entity)
        previous = latest.get(key)
        if previous is None or record.rank > previous.rank:
            latest[key] = record
    final_games = {row.entity for row in latest.values() if row.kind == "game"}
    labels: dict[date, list[_Record]] = defaultdict(list)
    for record in latest.values():
        if record.data["game_id"] in final_games:
            labels[record.day].append(record)
    for rows in labels.values():
        rows.sort(key=lambda item: (item.kind, item.entity, item.row_id))
    return labels


def _make_graph(day: date, history: _History, labels: list[_Record]) -> GraphDay:
    games = [row for row in labels if row.kind == "game"]
    pas = [row for row in labels if row.kind == "pa"]
    player_ids = tuple(
        sorted(
            set(history.batting)
            | set(history.pitching)
            | {row.data[key] for row in pas for key in ("batter_id", "pitcher_id")}
        )
    )
    team_ids = tuple(
        sorted(
            set(history.teams)
            | {
                key[1]
                for name in ("batter_participation_team", "pitcher_participation_team")
                for key in history.routes[name]
            }
            | {row.data[key] for row in games for key in ("home_team_id", "away_team_id")}
        )
    )
    players = {value: index for index, value in enumerate(player_ids)}
    teams = {value: index for index, value in enumerate(team_ids)}
    cutoff = datetime.combine(day, time.min, tzinfo=_KST).timestamp()
    batting = np.asarray(
        [
            _role_features(history.batting.get(player), cutoff, history.rolling_days)
            for player in player_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    pitching = np.asarray(
        [
            _role_features(history.pitching.get(player), cutoff, history.rolling_days)
            for player in player_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    arrays: dict[str, Array] = {
        "player_features": np.column_stack(
            (batting[:, 0], pitching[:, 0], batting[:, -1], pitching[:, -1])
        ),
        "player_batting_features": batting,
        "player_pitching_features": pitching,
        "team_features": np.asarray(
            [
                _team_features(history.teams.get(team), cutoff, history.rolling_days)
                for team in team_ids
            ],
            dtype=np.float32,
        ).reshape(-1, 8),
    }
    for route, aggregates in history.routes.items():
        metadata = ROUTE_METADATA[route]
        source = players if metadata["source_type"] == "player" else teams
        destination = players if metadata["destination_type"] == "player" else teams
        keys = sorted(aggregates)
        prefix = f"{route}__"
        arrays[prefix + "source_index"] = np.asarray(
            [source[key[0]] for key in keys], dtype=np.int64
        )
        arrays[prefix + "destination_index"] = np.asarray(
            [destination[key[1]] for key in keys],
            dtype=np.int64,
        )
        arrays[prefix + "event_features"] = np.asarray(
            [_route_features(aggregates[key], cutoff, history.rolling_days, route) for key in keys],
            dtype=np.float32,
        ).reshape(-1, 6)
        arrays[prefix + "event_age_seconds"] = np.asarray(
            [cutoff - aggregates[key].last_event for key in keys],
            dtype=np.float32,
        )
        arrays[prefix + "publication_delay_seconds"] = np.asarray(
            [max(0, aggregates[key].last_available - aggregates[key].last_event) for key in keys],
            dtype=np.float32,
        )
        # Count is already encoded as a feature; each aggregate is one relation.
        arrays[prefix + "weights"] = np.ones(len(keys), dtype=np.float32)
    arrays.update(_queries(games, pas, players, teams))
    return GraphDay(day, player_ids, team_ids, arrays)


def _recency(aggregate: _Aggregate, cutoff: float, rolling_days: int) -> float:
    return max(0.0, 1.0 - (cutoff - aggregate.last_event) / (rolling_days * 86400))


def _role_features(aggregate: _Aggregate | None, cutoff: float, window: int) -> list[float]:
    if aggregate is None:
        return [0.0] * 8
    pa, ab, hits, bases, walks, strikeouts, homers = aggregate.values
    return [
        math.log1p(pa) / math.log1p(500),
        ab / pa,
        hits / pa,
        bases / pa / 4,
        walks / pa,
        strikeouts / pa,
        homers / pa,
        _recency(aggregate, cutoff, window),
    ]


def _team_features(aggregate: _Aggregate | None, cutoff: float, window: int) -> list[float]:
    if aggregate is None:
        return [0.0] * 8
    games, wins, draws, own, against, difference, home = aggregate.values
    return [
        math.log1p(games) / math.log1p(90),
        wins / games,
        draws / games,
        own / games / 10,
        against / games / 10,
        difference / games / 10,
        home / games,
        _recency(aggregate, cutoff, window),
    ]


def _route_features(aggregate: _Aggregate, cutoff: float, window: int, route: str) -> list[float]:
    count, second, third, fourth, fifth, sixth, _ = aggregate.values
    if route == "home_team_game_away_team":
        return [
            math.log1p(count) / math.log1p(20),
            second / count,
            third / count,
            fourth / count / 10,
            fifth / count / 10,
            _recency(aggregate, cutoff, window),
        ]
    return [
        math.log1p(count) / math.log1p(100),
        third / count,
        fifth / count,
        sixth / count,
        fourth / count / 4,
        _recency(aggregate, cutoff, window),
    ]


def _queries(
    games: list[_Record],
    pas: list[_Record],
    players: dict[str, int],
    teams: dict[str, int],
) -> dict[str, Array]:
    match_rows = [row.data for row in games]
    live: dict[tuple[str, str], list[Any]] = {}
    pa_rows: list[dict[str, Any]] = []
    pa_targets = []
    for record in pas:
        row = record.data
        key = (row["game_id"], row["batter_id"])
        target = live.setdefault(key, [row["batting_team_id"], row["fielding_team_id"], 0, 0])
        if target[:2] != [row["batting_team_id"], row["fielding_team_id"]]:
            raise ValueError(f"conflicting team identities for player-game {key}")
        target[2] += 1
        target[3] += int(row["is_hit"])
        outcome = neural_training_target_index(TerminalPlateAppearanceEvent(row["outcome"]))
        if outcome is not None:
            pa_rows.append(row)
            pa_targets.append(outcome)
    live_keys = sorted(live)
    context = []
    for row in pa_rows:
        runners = str(row["runners_before"])
        if len(runners) != 3 or set(runners) - {"0", "1"}:
            raise ValueError("invalid canonical runners_before bitmap")
        # Terminal labels can be valid even when the source's score boundary is
        # inconsistent. Never present those before-scores as observed context.
        complete = row.get("transition_complete", True)
        home_before = row["home_score_before"] if complete else None
        away_before = row["away_score_before"] if complete else None
        context.append(
            [
                row["inning"] / 12,
                row["half_inning"] == "bottom",
                row["outs_before"] / 2,
                *[int(bit) for bit in runners],
                (home_before or 0) / 10,
                (away_before or 0) / 10,
                home_before is None,
                away_before is None,
            ]
        )
    return {
        "match_home_team_index": np.asarray(
            [teams[row["home_team_id"]] for row in match_rows],
            dtype=np.int64,
        ),
        "match_away_team_index": np.asarray(
            [teams[row["away_team_id"]] for row in match_rows],
            dtype=np.int64,
        ),
        "match_targets": np.asarray(
            [
                2
                if row["home_score"] > row["away_score"]
                else int(row["home_score"] == row["away_score"])
                for row in match_rows
            ],
            dtype=np.int64,
        ),
        "match_runs": np.asarray(
            [[row["home_score"], row["away_score"]] for row in match_rows],
            dtype=np.float32,
        ).reshape(-1, 2),
        "match_query_ids": np.asarray([row["game_id"] for row in match_rows], dtype=np.str_),
        "live_hit_player_index": np.asarray([players[key[1]] for key in live_keys], dtype=np.int64),
        "live_hit_team_index": np.asarray(
            [teams[live[key][0]] for key in live_keys], dtype=np.int64
        ),
        "live_hit_opponent_index": np.asarray(
            [teams[live[key][1]] for key in live_keys], dtype=np.int64
        ),
        "live_hit_pa": np.asarray([live[key][2] for key in live_keys], dtype=np.int64),
        "live_hit_hits": np.asarray([live[key][3] for key in live_keys], dtype=np.int64),
        "live_hit_query_ids": np.asarray(
            [f"{key[0]}|{key[1]}" for key in live_keys], dtype=np.str_
        ),
        "pa_batter_index": np.asarray(
            [players[row["batter_id"]] for row in pa_rows], dtype=np.int64
        ),
        "pa_pitcher_index": np.asarray(
            [players[row["pitcher_id"]] for row in pa_rows], dtype=np.int64
        ),
        "pa_targets": np.asarray(pa_targets, dtype=np.int64),
        "pa_context": np.asarray(context, dtype=np.float32).reshape(-1, 10),
        "pa_query_ids": np.asarray([row["plate_appearance_id"] for row in pa_rows], dtype=np.str_),
    }


def _validate_graph(graph: GraphDay) -> None:
    if len(set(graph.player_ids)) != len(graph.player_ids) or len(set(graph.team_ids)) != len(
        graph.team_ids
    ):
        raise ValueError("duplicate graph node IDs")
    for name, array in graph.arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"object/pickle array is forbidden: {name}")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"non-finite graph array: {name}")
    for name, size, dimension in (
        ("player_features", len(graph.player_ids), 4),
        ("player_batting_features", len(graph.player_ids), 8),
        ("player_pitching_features", len(graph.player_ids), 8),
        ("team_features", len(graph.team_ids), 8),
    ):
        if graph.arrays[name].shape != (size, dimension):
            raise ValueError(f"invalid node feature dimensions: {name}")
    for name, arrays in graph.routes.items():
        metadata = ROUTE_METADATA[name]
        count = len(arrays["source_index"])
        for side in ("source", "destination"):
            limit = len(
                graph.player_ids if metadata[f"{side}_type"] == "player" else graph.team_ids
            )
            index = arrays[f"{side}_index"]
            if index.shape != (count,) or np.any(index < 0) or np.any(index >= limit):
                raise ValueError(f"invalid route {side} indices: {name}")
        if arrays["event_features"].shape != (count, 6):
            raise ValueError(f"invalid route features: {name}")
        if np.any(arrays["event_age_seconds"] < arrays["publication_delay_seconds"]):
            raise ValueError(f"route includes information unavailable at cutoff: {name}")
    if np.any(graph.arrays["live_hit_pa"] < 1):
        raise ValueError("conditional LiveHit targets must have at least one observed PA")
    if np.any(graph.arrays["live_hit_hits"] > graph.arrays["live_hit_pa"]):
        raise ValueError("hits exceed observed PA")


def _first_cutoff(value: datetime) -> date:
    local = value.astimezone(_KST)
    return local.date() + timedelta(days=bool(local.time().replace(tzinfo=None) != time.min))


def _as_date(value: date | str | None) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) else value


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default
        )
        handle.write("\n")
    os.replace(temporary, path)


def _read_valid_cache(sidecar: Path, path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        with sidecar.open(encoding="utf-8") as handle:
            entry = cast(dict[str, Any], json.load(handle))
        if entry["input_fingerprint"] == fingerprint and entry["sha256"] == _sha256_file(path):
            return entry
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
