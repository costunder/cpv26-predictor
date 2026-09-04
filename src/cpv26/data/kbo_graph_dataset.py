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
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, TypeGuard, cast
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from numpy.typing import NDArray

from cpv26.data.kbo_player_identity import historical_player_prior
from cpv26.data.kbo_source_snapshots import superseded_source_ids
from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES, neural_training_target_index
from cpv26.simulation.events import TerminalPlateAppearanceEvent

GRAPH_DATASET_VERSION = 5
GRAPH_VNEXT_DATASET_VERSION = 6
GRAPH_SCHEMAS = ("v5", "vnext")
_KST = ZoneInfo("Asia/Seoul")
Array = NDArray[Any]

# Graph caches are immutable inputs during training.  Remember a successful
# verification only while the file's kernel identity is unchanged; this avoids
# re-reading every compressed archive solely to hash it again on every epoch.
# The cache is process-local by design, so it is neither persisted nor part of
# the dataset fingerprint.
_VERIFIED_GRAPH_FILES: dict[Path, tuple[str, tuple[int, int, int, int, int]]] = {}
_VERIFIED_GRAPH_FILES_LOCK = threading.Lock()
BOX_BATTING_FIELDS = (
    "at_bats",
    "hits",
    "runs",
    "rbi",
    "plate_appearances",
    "total_bases",
    "home_runs",
    "walks_hbp",
    "strikeouts",
)
BOX_PITCHING_FIELDS = (
    "batters_faced",
    "outs",
    "pitches",
    "at_bats",
    "hits",
    "home_runs",
    "walks_hbp",
    "strikeouts",
    "runs",
    "earned_runs",
)
BOX_FEATURE_DIMS = {"batting": 19, "pitching": 21}

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
VNEXT_GAME_FEATURE_NAMES = (
    "is_current_query_game",
    "is_historical_game",
    "age_days_div90",
    "scheduled_start_fraction_of_day",
)
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
VNEXT_ROUTE_FEATURE_NAMES = {
    **ROUTE_FEATURE_NAMES,
    "batter_game_participation": _PA_ROUTE_FEATURES,
    "pitcher_game_participation": _PA_ROUTE_FEATURES,
    "team_game_context": (
        "is_current_query_game",
        "is_home_team",
        "is_away_team",
        "is_historical_game",
    ),
}
VNEXT_ROUTE_METADATA = {
    **ROUTE_METADATA,
    "batter_game_participation": {
        "source_type": "player",
        "destination_type": "game",
        "source_role": "batting",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past player-game batting participation; never a current-day appearance",
    },
    "pitcher_game_participation": {
        "source_type": "player",
        "destination_type": "game",
        "source_role": "pitching",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "past player-game pitching participation; never a current-day appearance",
    },
    "team_game_context": {
        "source_type": "team",
        "destination_type": "game",
        "source_role": "shared",
        "destination_role": "shared",
        "bidirectional": True,
        "meaning": "known matchup structure for historical and current query games; no result",
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
    game_ids: tuple[str, ...] = ()

    @property
    def day_id(self) -> str:
        return self.day.isoformat()

    @property
    def node_features(self) -> dict[str, Array]:
        result = {
            "player": self.arrays["player_features"],
            "team": self.arrays["team_features"],
        }
        if "game_features" in self.arrays:
            result["game"] = self.arrays["game_features"]
        return result

    @property
    def role_features(self) -> dict[str, Array]:
        return {role: self.arrays[f"player_{role}_features"] for role in ROLE_FEATURE_NAMES}

    @property
    def routes(self) -> dict[str, dict[str, Array]]:
        known = tuple(
            route
            for route in VNEXT_ROUTE_METADATA
            if f"{route}__source_index" in self.arrays
        )
        discovered = {
            name.removesuffix("__source_index")
            for name in self.arrays
            if name.endswith("__source_index")
        }
        # Preserve the v5/v6 insertion contract, then expose versioned route
        # families without teaching this transport dataclass every new schema.
        routes = (*known, *sorted(discovered.difference(known)))
        return {
            route: {key: self.arrays[f"{route}__{key}"] for key in _ROUTE_ARRAY_FIELDS}
            for route in routes
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
        version = self.manifest.get("dataset_version")
        if version not in (
            2,
            3,
            4,
            GRAPH_DATASET_VERSION,
            GRAPH_VNEXT_DATASET_VERSION,
        ):
            raise ValueError("unsupported KBO graph dataset version")
        self.dataset_version = int(version)
        node_dims = self.manifest.get("node_feature_dims")
        route_dims = self.manifest.get("route_feature_dims")
        if self.dataset_version == GRAPH_VNEXT_DATASET_VERSION:
            if self.manifest.get("graph_schema") != "vnext":
                raise ValueError("version-six graph manifest must declare graph_schema=vnext")
            if not isinstance(node_dims, dict) or set(node_dims) != {"player", "team", "game"}:
                raise ValueError("version-six graph manifest must declare game node features")
            if not isinstance(route_dims, dict) or set(route_dims) != set(
                VNEXT_ROUTE_FEATURE_NAMES
            ):
                raise ValueError("version-six graph manifest must declare every vNext route")
        elif self.manifest.get("graph_schema") == "vnext" or (
            isinstance(node_dims, dict) and "game" in node_dims
        ):
            raise ValueError("legacy graph versions cannot declare the vNext game schema")
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
        arrays, file_identity = _read_verified_graph_archive(
            path, str(entry["sha256"]), key
        )
        has_game_ids = "_game_ids" in arrays
        if has_game_ids != (self.dataset_version == GRAPH_VNEXT_DATASET_VERSION):
            raise ValueError("graph archive game-node contract disagrees with manifest version")
        players = tuple(str(value) for value in arrays.pop("_player_ids"))
        teams = tuple(str(value) for value in arrays.pop("_team_ids"))
        games = tuple(str(value) for value in arrays.pop("_game_ids", ()))
        _add_box_defaults(arrays, len(players), len(teams))
        graph = GraphDay(date.fromisoformat(key), players, teams, arrays, games)
        _validate_graph(graph)
        _remember_verified_graph_file(path, str(entry["sha256"]), file_identity)
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
    box_values: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

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

    def __init__(
        self, records: list[_Record], rolling_days: int, *, graph_schema: str = "v5"
    ) -> None:
        self.rolling_days = rolling_days
        self.graph_schema = graph_schema
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
        self.box_batting: dict[str, _Aggregate] = {}
        self.box_pitching: dict[str, _Aggregate] = {}
        self.box_team_batting: dict[str, _Aggregate] = {}
        self.box_team_pitching: dict[str, _Aggregate] = {}
        self.games: dict[str, _Record] = {}
        # Active source versions, NOT latest retrospective labels. Rebuild only
        # changed player-game groups after publication/revision/expiry updates.
        self.box_inputs: dict[tuple[str, str, str], dict[str, _Record]] = defaultdict(dict)
        self.box_applied: dict[tuple[str, str, str], list[_Record]] = {}
        self.changed_box_groups: set[tuple[str, str, str]] = set()
        route_names = list(ROUTE_METADATA)
        if graph_schema == "vnext":
            route_names.extend(("batter_game_participation", "pitcher_game_participation"))
        self.routes: dict[str, dict[tuple[str, str], _Aggregate]] = {
            route: {} for route in route_names
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
                if record.available_at <= cutoff and record.valid_from <= cutoff
            ]
            selected = max(eligible, key=lambda item: item.rank) if eligible else None
            if selected is not None and (
                not earliest <= selected.day < day
                or (selected.valid_to is not None and cutoff >= selected.valid_to)
            ):
                selected = None
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
        for group in sorted(self.changed_box_groups):
            for record in self.box_applied.pop(group, ()):
                self._update_box(record, False)
            inputs = self.box_inputs.get(group, {})
            if inputs:
                applied = _common_box_records(list(inputs.values()), group[2])
                for record in applied:
                    self._update_box(record, True)
                self.box_applied[group] = applied
            else:
                self.box_inputs.pop(group, None)
        self.changed_box_groups.clear()

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
            aggregate = _Aggregate(values=np.zeros_like(values))
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
            if self.graph_schema == "vnext":
                for route, player in (
                    ("batter_game_participation", batter),
                    ("pitcher_game_participation", pitcher),
                ):
                    update(
                        self.routes[route],
                        (player, data["game_id"]),
                        record,
                        record.values,
                        add,
                    )
            for role, team in (
                ("batting", data["batting_team_id"]),
                ("pitching", data["fielding_team_id"]),
            ):
                self._track_box_input(record, role, team, add)
        elif record.kind.startswith("box_"):
            self._track_box_input(record, data["role"], data["team_id"], add)
        else:
            game_id = str(data["game_id"])
            if add and _has_final_score(data):
                self.games[game_id] = record
            elif not add:
                self.games.pop(game_id, None)
            if not _has_final_score(data):
                return  # A cancellation/withdrawal still supersedes the old source row.
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

    def _track_box_input(self, record: _Record, role: str, team: str, add: bool) -> None:
        key = (record.data["game_id"], team, role)
        if add:
            self.box_inputs[key][record.row_id] = record
        else:
            del self.box_inputs[key][record.row_id]
        self.changed_box_groups.add(key)

    def _update_box(self, record: _Record, add: bool) -> None:
        data = record.data
        if not record.box_values[len(record.box_values) // 2 :].any():
            return
        role, team = data["role"], data["team_id"]
        identity = _box_history_id(data)
        update = self._update_bucket
        update(getattr(self, f"box_{role}"), identity, record, record.box_values, add)
        team_values = record.box_values
        if "team_history_field_mask" in data:
            mask = np.asarray(data["team_history_field_mask"], dtype=np.float64)
            team_values = team_values * np.tile(mask, 2)
        if team_values[len(team_values) // 2 :].any():
            update(getattr(self, f"box_team_{role}"), team, record, team_values, add)
        if data.get("derived_from_pa", False):
            # PA already populated individual legacy features and actual routes.
            return
        values = _complete_legacy_box_values(data)
        if values is not None and not data.get("has_pa_history", False):
            update(getattr(self, role), identity, record, values, add)
        # The cohort is an explicitly uncertain source-name/team group, never
        # a resolved person or an invented batter-pitcher encounter.
        if not data.get("has_pa_history", False):
            route = (
                "batter_participation_team" if role == "batting" else "pitcher_participation_team"
            )
            update(
                self.routes[route],
                (identity, team),
                record,
                values if values is not None else np.zeros(7),
                add,
            )
            if self.graph_schema == "vnext":
                game_route = (
                    "batter_game_participation"
                    if role == "batting"
                    else "pitcher_game_participation"
                )
                update(
                    self.routes[game_route],
                    (identity, data["game_id"]),
                    record,
                    values if values is not None else np.zeros(7),
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
    graph_schema: str = "v5",
) -> KBOGraphDataset:
    """Build/reuse daily graphs; date filters NEVER truncate preceding history.

    The output contains ``manifest.json`` and ``days/YYYY-MM-DD.npz``. Sidecars
    allow interrupted builds and appended source days to reuse verified caches.
    Labels are the latest final rows visible at the database knowledge snapshot.
    The default knowledge snapshot is captured once at build start. Source
    snapshot selection, ingestion, publication and label validity share it.
    """
    if isinstance(rolling_days, bool) or not isinstance(rolling_days, int) or rolling_days < 1:
        raise ValueError("rolling_days must be a positive integer")
    if graph_schema not in GRAPH_SCHEMAS:
        raise ValueError(f"graph_schema must be one of: {', '.join(GRAPH_SCHEMAS)}")
    if knowledge_at is not None and knowledge_at.utcoffset() is None:
        raise ValueError("knowledge_at must be timezone-aware")
    first, last = _as_date(start_day), _as_date(end_day)
    if first is not None and last is not None and first > last:
        raise ValueError("start_day must not be after end_day")
    database = Path(database_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    label_snapshot_at = knowledge_at or datetime.now(timezone.utc)
    records, sources, source_digest = _read_records(database, label_snapshot_at)
    labels = _label_records(records, label_snapshot_at)
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
    dataset_version = (
        GRAPH_VNEXT_DATASET_VERSION if graph_schema == "vnext" else GRAPH_DATASET_VERSION
    )
    route_feature_names = (
        VNEXT_ROUTE_FEATURE_NAMES if graph_schema == "vnext" else ROUTE_FEATURE_NAMES
    )
    route_metadata = VNEXT_ROUTE_METADATA if graph_schema == "vnext" else ROUTE_METADATA
    node_feature_names = dict(NODE_FEATURE_NAMES)
    if graph_schema == "vnext":
        node_feature_names["game"] = VNEXT_GAME_FEATURE_NAMES
    config = {
        "dataset_version": dataset_version,
        **({"graph_schema": graph_schema} if graph_schema == "vnext" else {}),
        "rolling_days": rolling_days,
        "pa_incomplete_transition_context": "mask_pre_scores_unknown",
        "boxscore_history_policy": "common_player_game_observed_fields_v3",
        "boxscore_identity_policy": "unmerged_observations_name_team_role_cohorts_v1",
        "label_snapshot_policy": "latest_known_valid_entity_in_latest_annual_source_v1",
        "cutoff_timezone": "Asia/Seoul",
        "cutoff_time": "00:00:00",
        "knowledge_at": knowledge_at.isoformat()
        if knowledge_at is not None
        else "database_snapshot",
    }
    config_fingerprint = _json_sha256(config)
    history = _History(records, rolling_days, graph_schema=graph_schema)
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
        games = {row.entity for row in labels[day] if row.kind == "game"}
        pa_games = {row.data["game_id"] for row in labels[day] if row.kind == "pa"}
        box_games = {row.data["game_id"] for row in labels[day] if row.kind.startswith("box_")}
        raw_audit = _box_label_audit(labels[day])
        coverage = {
            "games_with_pa": len(games & pa_games),
            "games_with_boxscore": len(games & box_games),
            "boxscore_only_games": len((games & box_games) - pa_games),
            "game_only_games": len(games - pa_games - box_games),
            "observed_completed_pa": sum(row.kind == "pa" for row in labels[day]),
            **raw_audit,
            **_target_coverage(labels[day]),
            "raw_archive_boxscore": raw_audit,
        }
        cached = _read_valid_cache(sidecar, path, input_fingerprint)
        if cached is not None:
            # Coverage is metadata only: preserve old graph/checkpoint fingerprints.
            if any(cached.get(key) != value for key, value in coverage.items()):
                cached.update(coverage)
                _write_json_atomic(sidecar, cached)
            entries.append(cached)
            reused += 1
            continue
        graph = _make_graph(day, history, labels[day], graph_schema=graph_schema)
        _validate_graph(graph)
        temporary = path.with_suffix(".npz.part")
        archive_arrays: dict[str, Any] = {
            "_player_ids": np.asarray(graph.player_ids, dtype=np.str_),
            "_team_ids": np.asarray(graph.team_ids, dtype=np.str_),
            **(
                {"_game_ids": np.asarray(graph.game_ids, dtype=np.str_)}
                if graph_schema == "vnext"
                else {}
            ),
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
            **({"graph_game_nodes": len(graph.game_ids)} if graph_schema == "vnext" else {}),
            "games": len(graph.arrays["match_targets"]),
            "live_hit_queries": len(graph.arrays["live_hit_pa"]),
            "pa_queries": len(graph.arrays["pa_targets"]),
            "history_rows": len(history.active),
            "feature_coverage": {
                name: int(np.any(graph.arrays[name] != 0, axis=1).sum())
                for name in (
                    "player_batting_features",
                    "player_pitching_features",
                    "player_box_batting_features",
                    "player_box_pitching_features",
                    "team_box_batting_features",
                    "team_box_pitching_features",
                )
            },
            **coverage,
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
        "label_snapshot_at": label_snapshot_at.isoformat(),
        "days": entries,
        "season_coverage": _season_coverage(entries),
        "cache_reused_days": reused,
        "cache_built_days": len(entries) - reused,
        "node_feature_dims": {key: len(value) for key, value in node_feature_names.items()},
        "role_feature_dims": {key: len(value) for key, value in ROLE_FEATURE_NAMES.items()},
        "player_role_feature_dims": {key: len(value) for key, value in ROLE_FEATURE_NAMES.items()},
        "boxscore_feature_dims": BOX_FEATURE_DIMS,
        "boxscore_feature_fields": {
            "batting": BOX_BATTING_FIELDS,
            "pitching": BOX_PITCHING_FIELDS,
            "encoding": "log1p observed sum / log501, log1p observed field count / log501, recency",
        },
        "box_pitch_target_fields": BOX_PITCHING_FIELDS,
        "route_feature_dims": {key: len(value) for key, value in route_feature_names.items()},
        "feature_names": {
            "nodes": node_feature_names,
            "roles": ROLE_FEATURE_NAMES,
            "routes": route_feature_names,
        },
        "route_metadata": route_metadata,
        "pa_context_dim": len(PA_CONTEXT_FEATURE_NAMES),
        "pa_context_feature_names": PA_CONTEXT_FEATURE_NAMES,
        "pa_target_classes": NEURAL_PA_OUTCOMES,
        "match_target_classes": ("away_win", "draw", "home_win"),
        "source_provenance": sources,
        "label_quality": {
            "training_targets": {
                name: sum(entry[name] for entry in entries) for name in _TRAINING_COUNT_FIELDS
            },
            "raw_archive_boxscore": _sum_archive_audits(
                [entry["raw_archive_boxscore"] for entry in entries]
            ),
            "historical_boxscore_identity": _identity_audit(
                [row for day in selected_days for row in labels[day]]
            ),
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
            "historical_boxscore": {
                name: sum(entry["raw_archive_boxscore"][name] for entry in entries)
                for name in (
                    "box_batting_rows",
                    "box_pitching_rows",
                    "box_pa_queries",
                    "box_pitch_queries",
                    "box_live_hit_queries",
                    "box_live_hit_unknown_pa_queries",
                    "box_pa_outcomes",
                    "box_pitch_observed_counts",
                )
            },
        },
        "policies": {
            "history": "event_date < day; available_at <= cutoff; valid at cutoff; rolling window",
            "ingestion": "retrospective database knowledge snapshot, not historical live ingestion",
            "same_day": "all same-day PA/boxscore/final stats are labels only, regardless of order",
            "current_players": "isolated query-only nodes unless they have eligible prior history",
            "current_player_game_relation": (
                "none_without_timestamped_asof_lineup_or_roster_source"
                if graph_schema == "vnext"
                else "not represented"
            ),
            "game_nodes": (
                "rolling historical games plus current query games; current nodes contain no "
                "score, result, lineup, starter, or observed appearance"
                if graph_schema == "vnext"
                else "not represented"
            ),
            "team_membership": "past role-specific appearances, never today's actual roster",
            "match_queries": "home and away team only; no actual current lineup or pitcher",
            "live_hit": "conditional completed observed PA >= 1; not an appearance forecast",
            "pa_context": "current pre-PA state, separate auxiliary decoder only",
            "pa_incomplete_transition": "keep PA label; mask both pre-scores to 0 + missing flags",
            "pitcher": "last observed pitcher, not official inherited-count/run responsibility",
            "pa_pair": "credited batter/terminal pitcher; substitutions may not face final pitch",
            "scaling": "fixed constants and past-only ratios; no fitted full-sample normalization",
            "unknown_labels": "unlabelled source PAs omitted; catcher interference is not PA10",
            "game_only": "final scores supervise match/run only; no synthetic PA or LiveHit labels",
            "boxscore_identity": (
                "observation IDs stay distinct; canonical IDs only when source-verified; "
                "unresolved names use explicitly uncertain name/team/role cohort features, "
                "not individual career history; missing names fall back to team-role priors"
            ),
            "boxscore_history": (
                "box and PA sources share player-game observed sums/counts; one field count "
                "per player-game, not per PA; unknown fields remain masked; revisions selected "
                "at cutoff before aggregation; raw source records remain unchanged"
            ),
            "overlap": (
                "usable archive fields precede PA-derived team fields; independent PA player "
                "history remains; each auxiliary task/field requires usable archive evidence "
                "before suppressing PA targets; incomplete PA never verifies full box totals"
            ),
            "label_snapshot": (
                "latest known annual source snapshot, then latest available/valid entity; "
                "expired latest rows are withdrawn, not replaced by older superseded truth"
            ),
            "boxscore_pa": "verified actual outcome totals only; no invented ordered PA or pitcher",
            "boxscore_live_hit": (
                "known PA uses joint target; unknown PA with AB>=1 uses PA=-1 and lower bound AB; "
                "unknown-PA zero-AB rows do not imply an appearance"
            ),
            "boxscore_pitching": "each reported count is separately observed or masked",
            "simulator_ready": False,
        },
    }
    _write_json_atomic(directory / "manifest.json", manifest)
    return KBOGraphDataset(directory)


def _season_coverage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seasons: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        seasons[date.fromisoformat(entry["day"]).year].append(entry)
    counts = (
        "games",
        "games_with_pa",
        "games_with_boxscore",
        "boxscore_only_games",
        "game_only_games",
        "observed_completed_pa",
        "live_hit_queries",
        "pa_queries",
        "box_batting_rows",
        "box_pitching_rows",
        "box_pa_queries",
        "box_pitch_queries",
        "box_live_hit_queries",
        "box_live_hit_unknown_pa_queries",
        "box_pa_outcomes",
        "box_pitch_observed_counts",
        "pa_derived_batting_queries",
        "pa_derived_pitching_queries",
        "live_hit_unknown_pa_queries",
    )
    return [
        {
            "season": season,
            "days": len(rows),
            "date_start": min(row["day"] for row in rows),
            "date_end": max(row["day"] for row in rows),
            **{name: sum(row[name] for row in rows) for name in counts},
            "raw_archive_boxscore": _sum_archive_audits(
                [row["raw_archive_boxscore"] for row in rows]
            ),
            "box_target_missing_reasons": {
                reason: sum(row["box_target_missing_reasons"].get(reason, 0) for row in rows)
                for reason in sorted(
                    {reason for row in rows for reason in row["box_target_missing_reasons"]}
                )
            },
        }
        for season, rows in sorted(seasons.items())
    ]


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
        superseded = superseded_source_ids(connection, knowledge_at)
        records: list[_Record] = []
        queries = [
            ("game", "SELECT * FROM game ORDER BY game_id, valid_from"),
            (
                "pa",
                "SELECT * FROM observed_plate_appearance ORDER BY plate_appearance_id, valid_from",
            ),
        ]
        table_names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "historical_boxscore" in table_names:
            queries.append(
                ("box", "SELECT * FROM historical_boxscore ORDER BY observation_id, valid_from")
            )
        for kind, query in queries:
            for row in _iter_dicts(connection, query):
                if row["source_revision_id"] in superseded:
                    continue
                if knowledge_at is not None and row["ingested_at"] > knowledge_at:
                    continue
                source = source_lookup.get(row["source_revision_id"])
                if source is None:
                    raise ValueError(f"missing provenance source {row['source_revision_id']}")
                if knowledge_at is not None and source["ingested_at"] > knowledge_at:
                    continue
                event = row["event_at"].astimezone(_KST)
                game_day = (row["scheduled_start"] if kind == "game" else event).astimezone(_KST)
                record_kind = kind
                if kind == "box":
                    if row["role"] not in BOX_FEATURE_DIMS:
                        raise ValueError("invalid historical boxscore role")
                    record_kind = "box_" + row["role"]
                    for name in ("stats_json", "quality_json"):
                        if isinstance(row.get(name), str):
                            row[name] = json.loads(row[name])
                    if not isinstance(row.get("stats_json"), dict):
                        raise ValueError("historical boxscore stats must be an object")
                record = _Record(
                    kind=record_kind,
                    entity=row[
                        "game_id"
                        if kind == "game"
                        else "observation_id"
                        if kind == "box"
                        else "plate_appearance_id"
                    ],
                    row_id=row.get("boxscore_row_id")
                    or row.get("observed_pa_row_id")
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
                elif kind == "box":
                    record.box_values = _box_record_values(row)
                    # The full source row has already contributed to the digest.
                    # Keep only model/audit fields in hundreds of thousands of
                    # in-memory records, not raw JSON or inning-token objects.
                    record.data = _box_graph_data(row)
                elif _has_final_score(row):
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
                "superseded_at_knowledge_snapshot": row["source_revision_id"] in superseded,
            }
            for row in sources
            if row["source_revision_id"] in used_sources | superseded
        ]
        source_digest = _json_sha256([f"{record.digest:064x}" for record in records])
        return records, provenance, source_digest
    finally:
        connection.close()


def _fetch_dicts(connection: Any, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _iter_dicts(connection: Any, query: str, chunk_size: int = 4096) -> Iterator[dict[str, Any]]:
    """Stream records in bounded chunks without changing SQL ordering or hashes."""
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    while rows := cursor.fetchmany(chunk_size):
        for row in rows:
            yield dict(zip(names, row, strict=True))


def _box_graph_data(row: Mapping[str, Any]) -> dict[str, Any]:
    stats = row["stats_json"]
    names = (
        set(BOX_BATTING_FIELDS) | set(BOX_PITCHING_FIELDS) | {"counts_verified", "hits_verified"}
    )
    if stats.get("counts_verified"):
        names.add("outcome_counts")
    return {
        "display_name": row.get("display_name"),
        **{
            name: row[name]
            for name in (
                "game_id",
                "observation_id",
                "role",
                "player_id",
                "identity_status",
                "team_id",
                "opponent_team_id",
            )
        },
        "stats_json": {name: stats[name] for name in names if name in stats},
        "quality_json": row.get("quality_json", ()),
    }


def _label_records(
    records: list[_Record], knowledge_at: datetime | None = None
) -> dict[date, list[_Record]]:
    snapshot = knowledge_at or datetime.now(timezone.utc)
    if snapshot.utcoffset() is None:
        raise ValueError("label knowledge snapshot must be timezone-aware")
    latest: dict[tuple[str, str], _Record] = {}
    for record in records:
        if any(
            value > snapshot
            for value in (record.ingested_at, record.available_at, record.valid_from)
        ):
            continue
        key = (record.kind, record.entity)
        previous = latest.get(key)
        if previous is None or record.rank > previous.rank:
            latest[key] = record
    current = [
        record
        for record in latest.values()
        if record.valid_to is None or snapshot < record.valid_to
    ]
    final_games = {
        row.entity for row in current if row.kind == "game" and _has_final_score(row.data)
    }
    labels: dict[date, list[_Record]] = defaultdict(list)
    for record in current:
        if record.data["game_id"] in final_games:
            labels[record.day].append(record)
    for rows in labels.values():
        rows.sort(key=lambda item: (item.kind, item.entity, item.row_id))
    return labels


def _has_final_score(row: Mapping[str, Any]) -> bool:
    return (
        row.get("game_status") == "final"
        and row.get("home_score") is not None
        and row.get("away_score") is not None
    )


def _make_graph(
    day: date,
    history: _History,
    labels: list[_Record],
    *,
    graph_schema: str = "v5",
) -> GraphDay:
    vnext = graph_schema == "vnext"
    games = [row for row in labels if row.kind == "game"]
    pas = [row for row in labels if row.kind == "pa"]
    boxes = _label_boxes(labels)
    box_queries = {(row.data["player_id"], row.data["role"]): row.data for row in boxes}
    current_game_records = {row.data["game_id"]: row for row in games}
    historical_game_records = dict(history.games) if vnext else {}
    game_route_aggregates: dict[str, _Aggregate] = {}
    if vnext:
        for route in ("batter_game_participation", "pitcher_game_participation"):
            for (_, game_id), aggregate in history.routes[route].items():
                previous = game_route_aggregates.get(game_id)
                if previous is None or aggregate.last_event > previous.last_event:
                    game_route_aggregates[game_id] = aggregate
    player_ids = tuple(
        sorted(
            set(history.batting)
            | set(history.pitching)
            | set(history.box_batting)
            | set(history.box_pitching)
            | (
                {
                    key[0]
                    for route in (
                        "batter_game_participation",
                        "pitcher_game_participation",
                    )
                    for key in history.routes[route]
                }
                if vnext
                else set()
            )
            | {key[0] for key in box_queries}
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
            | (
                {
                    record.data[key]
                    for record in historical_game_records.values()
                    for key in ("home_team_id", "away_team_id")
                }
                if vnext
                else set()
            )
            | {row.data[key] for row in games for key in ("home_team_id", "away_team_id")}
        )
    )
    game_ids = (
        tuple(
            sorted(
                set(historical_game_records)
                | set(game_route_aggregates)
                | {row.data["game_id"] for row in games}
            )
        )
        if vnext
        else ()
    )
    players = {value: index for index, value in enumerate(player_ids)}
    teams = {value: index for index, value in enumerate(team_ids)}
    game_nodes = {value: index for index, value in enumerate(game_ids)}
    cutoff = datetime.combine(day, time.min, tzinfo=_KST).timestamp()

    def role_history(player: str, role: str) -> _Aggregate | None:
        mapping = getattr(history, role)
        aggregate = mapping.get(player)
        query = box_queries.get((player, role))
        if aggregate is None and query is not None:
            aggregate = mapping.get(_box_history_id(query))
        return cast(_Aggregate | None, aggregate)

    batting = np.asarray(
        [
            _role_features(role_history(player, "batting"), cutoff, history.rolling_days)
            for player in player_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 8)
    pitching = np.asarray(
        [
            _role_features(role_history(player, "pitching"), cutoff, history.rolling_days)
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
    if vnext:
        current_game_ids = {row.data["game_id"] for row in games}
        arrays["game_features"] = np.asarray(
            [
                _game_node_features(
                    game_id,
                    current=game_id in current_game_ids,
                    record=current_game_records.get(game_id)
                    or historical_game_records.get(game_id),
                    fallback=game_route_aggregates.get(game_id),
                    cutoff=cutoff,
                )
                for game_id in game_ids
            ],
            dtype=np.float32,
        ).reshape(-1, len(VNEXT_GAME_FEATURE_NAMES))
    for role, dimension in BOX_FEATURE_DIMS.items():
        personal = getattr(history, f"box_{role}")
        team_prior = getattr(history, f"box_team_{role}")
        player_features = []
        for player in player_ids:
            aggregate = personal.get(player)
            query = box_queries.get((player, role))
            if aggregate is None and query is not None:
                aggregate = personal.get(_box_history_id(query))
                if aggregate is None:
                    aggregate = team_prior.get(query["team_id"])
            player_features.append(
                _box_features(aggregate, cutoff, history.rolling_days, dimension)
            )
        arrays[f"player_box_{role}_features"] = np.asarray(
            player_features, dtype=np.float32
        ).reshape(-1, dimension)
        arrays[f"team_box_{role}_features"] = np.asarray(
            [
                _box_features(team_prior.get(team), cutoff, history.rolling_days, dimension)
                for team in team_ids
            ],
            dtype=np.float32,
        ).reshape(-1, dimension)
    route_metadata = VNEXT_ROUTE_METADATA if vnext else ROUTE_METADATA
    node_mappings = {"player": players, "team": teams, "game": game_nodes}
    for route, aggregates in history.routes.items():
        metadata = route_metadata[route]
        source = node_mappings[cast(str, metadata["source_type"])]
        destination = node_mappings[cast(str, metadata["destination_type"])]
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
        ).reshape(-1, len(VNEXT_ROUTE_FEATURE_NAMES[route]))
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
    if vnext:
        _add_team_game_context_route(
            arrays,
            current_games=games,
            historical_games=historical_game_records,
            teams=teams,
            games=game_nodes,
            cutoff=cutoff,
        )
    arrays.update(_queries(games, pas, players, teams, boxes, game_nodes if vnext else None))
    return GraphDay(day, player_ids, team_ids, arrays, game_ids)


def _game_node_features(
    game_id: str,
    *,
    current: bool,
    record: _Record | None,
    fallback: _Aggregate | None,
    cutoff: float,
) -> list[float]:
    scheduled = record.data.get("scheduled_start") if record is not None else None
    scheduled_fraction = 0.0
    if isinstance(scheduled, datetime):
        local = scheduled.astimezone(_KST)
        scheduled_fraction = (
            local.hour * 3600 + local.minute * 60 + local.second
        ) / 86400
    if current:
        return [1.0, 0.0, 0.0, scheduled_fraction]
    event_at = record.event_at.timestamp() if record is not None else None
    if event_at is None and fallback is not None:
        event_at = fallback.last_event
    age_days = max(0.0, cutoff - event_at) / 86400 if event_at is not None else 90.0
    return [0.0, 1.0, min(age_days / 90.0, 1.0), scheduled_fraction]


def _add_team_game_context_route(
    arrays: dict[str, Array],
    *,
    current_games: list[_Record],
    historical_games: dict[str, _Record],
    teams: dict[str, int],
    games: dict[str, int],
    cutoff: float,
) -> None:
    rows: list[tuple[str, str, bool, bool, _Record]] = []
    for current, records in ((False, historical_games.values()), (True, current_games)):
        for record in records:
            game_id = record.data["game_id"]
            rows.extend(
                (
                    (record.data["home_team_id"], game_id, True, current, record),
                    (record.data["away_team_id"], game_id, False, current, record),
                )
            )
    rows.sort(key=lambda row: (row[1], not row[2], row[0]))
    prefix = "team_game_context__"
    arrays[prefix + "source_index"] = np.asarray(
        [teams[team_id] for team_id, *_ in rows], dtype=np.int64
    )
    arrays[prefix + "destination_index"] = np.asarray(
        [games[game_id] for _, game_id, *_ in rows], dtype=np.int64
    )
    arrays[prefix + "event_features"] = np.asarray(
        [
            [float(current), float(home), float(not home), float(not current)]
            for _, _, home, current, _ in rows
        ],
        dtype=np.float32,
    ).reshape(-1, len(VNEXT_ROUTE_FEATURE_NAMES["team_game_context"]))
    arrays[prefix + "event_age_seconds"] = np.asarray(
        [0.0 if current else cutoff - record.event_at.timestamp() for *_, current, record in rows],
        dtype=np.float32,
    )
    arrays[prefix + "publication_delay_seconds"] = np.asarray(
        [
            0.0
            if current
            else max(0.0, record.available_at.timestamp() - record.event_at.timestamp())
            for *_, current, record in rows
        ],
        dtype=np.float32,
    )
    arrays[prefix + "weights"] = np.ones(len(rows), dtype=np.float32)


def _box_prior_id(role: str, team: str) -> str:
    return f"kbo-team-role-prior:{role}:{team}"


def _box_history_id(row: Mapping[str, Any]) -> str:
    if row.get("identity_status") == "canonical_verified":
        return str(row["player_id"])
    return historical_player_prior(
        team_id=row["team_id"], role=row["role"], display_name=row.get("display_name")
    ).prior_id


def _complete_legacy_box_values(row: Mapping[str, Any]) -> Array | None:
    stats = _box_stat_fields(row)
    fields = (
        "plate_appearances" if row["role"] == "batting" else "batters_faced",
        "at_bats",
        "hits",
        "total_bases",
        "walks_hbp",
        "strikeouts",
        "home_runs",
    )
    if not all(_observed_nonnegative(stats.get(name)) for name in fields):
        return None
    if stats[fields[0]] <= 0:
        return None
    return np.asarray([stats[name] for name in fields], dtype=np.float64)


def _aggregate_record(data: dict[str, Any], components: list[_Record]) -> _Record:
    digest = int(
        _json_sha256(
            {
                "policy": "common_player_game_v1",
                "data": data,
                "components": sorted(f"{row.digest:064x}" for row in components),
            }
        ),
        16,
    )
    return _Record(
        kind="box_" + data["role"],
        entity=data["observation_id"],
        row_id=f"pa-box:{digest:064x}",
        day=components[0].day,
        event_at=max(row.event_at for row in components),
        available_at=max(row.available_at for row in components),
        ingested_at=max(row.ingested_at for row in components),
        valid_from=max(row.valid_from for row in components),
        valid_to=None,
        source_id=components[0].source_id,
        data=data,
        digest=digest,
        box_values=_box_record_values(data),
    )


def _pa_box_records(pas: list[_Record], role: str) -> list[_Record]:
    """Observed PA totals, one field-observation per player-game, not per PA.

    Runs/RBI, pitcher outs/pitches/runs/ER cannot be reconstructed completely
    from terminal PA rows. They remain unobserved, even when all PAs are known.
    Call with already selected source versions (history or retrospective labels).
    """
    player_field = "batter_id" if role == "batting" else "pitcher_id"
    team_field = "batting_team_id" if role == "batting" else "fielding_team_id"
    opponent_field = "fielding_team_id" if role == "batting" else "batting_team_id"
    grouped: dict[tuple[str, str], list[_Record]] = defaultdict(list)
    for record in pas:
        grouped[(record.data["game_id"], record.data[player_field])].append(record)
    result = []
    for (game, player), components in sorted(grouped.items()):
        first = components[0].data
        if any(
            (row.data[team_field], row.data[opponent_field])
            != (first[team_field], first[opponent_field])
            for row in components
        ):
            raise ValueError(f"conflicting PA teams for player-game {(game, player)}")
        totals = np.sum([row.values for row in components], axis=0)
        stats: dict[str, Any] = dict(
            zip(
                (
                    "plate_appearances" if role == "batting" else "batters_faced",
                    "at_bats",
                    "hits",
                    "total_bases",
                    "walks_hbp",
                    "strikeouts",
                    "home_runs",
                ),
                (int(value) for value in totals),
                strict=True,
            )
        )
        if role == "batting":
            counts = [0] * len(NEURAL_PA_OUTCOMES)
            for row in components:
                target = neural_training_target_index(
                    TerminalPlateAppearanceEvent(row.data["outcome"])
                )
                if target is not None:
                    counts[target] += 1
            stats.update(outcome_counts=counts, counts_verified=True, hits_verified=True)
        observation = f"observed-pa-box:{role}:{game}:{player}"
        data = {
            "game_id": game,
            "observation_id": observation,
            "role": role,
            "player_id": player,
            "identity_status": "canonical_verified",
            "team_id": first[team_field],
            "opponent_team_id": first[opponent_field],
            "stats_json": stats,
            "quality_json": [],
            "derived_from_pa": True,
        }
        result.append(_aggregate_record(data, components))
    return result


def _common_box_records(records: list[_Record], role: str) -> list[_Record]:
    """Merge a single game/team/role without counting overlapping sources twice.

    Verified same-ID boxes supply reported fields first; observed PA fills only
    missing fields. Unresolved archive rows cannot be joined to a person: their
    reported team totals take precedence, while PA personal history is retained.
    """
    boxes = [row for row in records if row.kind == "box_" + role]
    derived = _pa_box_records([row for row in records if row.kind == "pa"], role)
    if not boxes:
        return derived
    by_player = {row.data["player_id"]: row for row in derived}
    result = []
    for record in boxes:
        row = record.data
        pa = (
            by_player.pop(row["player_id"], None)
            if (row.get("identity_status") == "canonical_verified")
            else None
        )
        if pa is None:
            result.append(record)
            continue
        stats = _box_stat_fields(row)
        pa_stats = pa.data["stats_json"]
        # Observed PAs can be incomplete relative to an official box score.
        # Never turn a smaller PA histogram into a verified complete box total
        # (for example H=2 beside a one-single histogram and TB=1).
        compatible = all(
            not _observed_nonnegative(stats.get(key)) or stats[key] == value
            for key, value in pa_stats.items()
            if _observed_nonnegative(value)
        )
        if compatible:
            if not _observed_nonnegative(stats.get("hits")):
                stats["hits"] = pa_stats.get("hits")
                stats["hits_verified"] = True
            if role == "batting" and not stats.get("counts_verified"):
                stats["outcome_counts"] = pa_stats["outcome_counts"]
                stats["counts_verified"] = True
            for key, value in pa_stats.items():
                if stats.get(key) is None:
                    stats[key] = value
        result.append(
            _aggregate_record({**row, "stats_json": stats, "has_pa_history": True}, [record, pa])
        )
    unresolved = [row for row in boxes if row.data.get("identity_status") != "canonical_verified"]
    team_mask = _fallback_field_mask(unresolved, list(by_player.values()), role)
    for record in by_player.values():
        record.data["team_history_field_mask"] = team_mask
        result.append(record)
    return result


def _fallback_field_mask(
    preferred: list[_Record], fallback: list[_Record], role: str
) -> list[bool]:
    """Prefer only actually observed fields, not the existence of a source row.

    An unknown identity prevents assigning archive rows to individual PA
    players. Team source precedence therefore applies field by field. Do not
    fill an unavailable official count with a smaller, contradictory PA total.
    Both sources still retain their separate personal observed history.
    """
    fields = BOX_BATTING_FIELDS if role == "batting" else BOX_PITCHING_FIELDS
    width = len(fields)
    primary = (
        np.sum([row.box_values for row in preferred], axis=0) if preferred else np.zeros(2 * width)
    )
    secondary = (
        np.sum([row.box_values for row in fallback], axis=0) if fallback else np.zeros(2 * width)
    )
    mask = primary[width:] == 0
    combined = {
        name: primary[index] if not mask[index] else secondary[index]
        for index, name in enumerate(fields)
        if primary[index + width] > 0 or secondary[index + width] > 0
    }
    constraints = (
        (
            ("hits", "at_bats"),
            ("at_bats", "plate_appearances"),
            ("hits", "total_bases"),
            ("home_runs", "hits"),
            ("strikeouts", "at_bats"),
        )
        if role == "batting"
        else (
            ("hits", "at_bats"),
            ("at_bats", "batters_faced"),
            ("home_runs", "hits"),
            ("strikeouts", "batters_faced"),
            ("earned_runs", "runs"),
        )
    )
    for numerator, denominator in constraints:
        if (
            numerator in combined
            and denominator in combined
            and combined[numerator] > combined[denominator]
        ):
            for name in (numerator, denominator):
                index = fields.index(name)
                if primary[index + width] == 0:
                    mask[index] = False
    return [bool(value) for value in mask]


def _label_boxes(labels: list[_Record]) -> list[_Record]:
    grouped: dict[tuple[str, str, str], list[_Record]] = defaultdict(list)
    for record in labels:
        row = record.data
        if record.kind == "pa":
            for role, field in (("batting", "batting_team_id"), ("pitching", "fielding_team_id")):
                grouped[(row["game_id"], row[field], role)].append(record)
        elif record.kind.startswith("box_"):
            grouped[(row["game_id"], row["team_id"], row["role"])].append(record)
    result = []
    for group, inputs in sorted(grouped.items()):
        role = group[2]
        boxes = [row for row in inputs if row.kind == "box_" + role]
        result.extend(boxes)
        derived = _pa_box_records([row for row in inputs if row.kind == "pa"], role)
        for record in derived:
            preferred = [
                row
                for row in boxes
                if row.data.get("identity_status") != "canonical_verified"
                or row.data["player_id"] == record.data["player_id"]
            ]
            data = dict(record.data)
            if role == "batting":
                data["box_pa_target_enabled"] = not any(
                    stats.get("counts_verified") and sum(stats.get("outcome_counts", ())) > 0
                    for stats in (_box_stat_fields(row.data) for row in preferred)
                )
            else:
                data["box_pitch_target_mask"] = [
                    not any(
                        _observed_nonnegative(_box_stat_fields(row.data).get(name))
                        for row in preferred
                    )
                    for name in BOX_PITCHING_FIELDS
                ]
            result.append(replace(record, data=data))
    return result


def _box_stat_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict(row["stats_json"])
    if row["role"] == "batting":
        hits, ab = stats.get("hits"), stats.get("at_bats")
        if not stats.get("hits_verified", True) or (
            _observed_nonnegative(hits) and _observed_nonnegative(ab) and hits > ab
        ):
            stats["hits"] = None
            stats["counts_verified"] = False
    else:
        # Keep the original row for provenance, but do not feed contradictory
        # scalars to either a target or a history prior. Compute the union before
        # masking so chained contradictions cannot hide another affected field.
        invalid: set[str] = set()
        for numerator, denominator in (
            ("hits", "at_bats"),
            ("home_runs", "hits"),
            ("earned_runs", "runs"),
            ("at_bats", "batters_faced"),
            ("strikeouts", "batters_faced"),
        ):
            first, second = stats.get(numerator), stats.get(denominator)
            if _observed_nonnegative(first) and _observed_nonnegative(second) and first > second:
                invalid.update((numerator, denominator))
        for name in invalid:
            stats[name] = None
    if row["role"] == "batting" and stats.get("counts_verified"):
        counts = np.asarray(stats.get("outcome_counts"), dtype=np.float64)
        if counts.shape != (10,) or np.any(counts < 0) or not np.isfinite(counts).all():
            raise ValueError("invalid verified historical PA counts")
        stats.update(
            total_bases=float(counts[2] + 2 * counts[3] + 3 * counts[4] + 4 * counts[5]),
            home_runs=float(counts[5]),
            walks_hbp=float(counts[1]),
            strikeouts=float(counts[0]),
        )
    return stats


def _observed_nonnegative(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _box_record_values(row: Mapping[str, Any]) -> Array:
    stats = _box_stat_fields(row)
    fields = BOX_BATTING_FIELDS if row["role"] == "batting" else BOX_PITCHING_FIELDS
    observed = [_observed_nonnegative(stats.get(name)) for name in fields]
    return np.asarray(
        [float(stats[name]) if known else 0.0 for name, known in zip(fields, observed, strict=True)]
        + [float(known) for known in observed],
        dtype=np.float64,
    )


def _box_features(
    aggregate: _Aggregate | None, cutoff: float, window: int, dimension: int
) -> list[float]:
    if aggregate is None:
        return [0.0] * dimension
    values = np.maximum(aggregate.values, 0)
    # Every sum is paired with the number of observations that actually report
    # that field. An unknown total is never silently converted into an observed zero.
    return [float(value) for value in np.log1p(values) / math.log1p(500)] + [
        _recency(aggregate, cutoff, window)
    ]


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
    if count <= 0:
        return [0.0] * 5 + [_recency(aggregate, cutoff, window)]
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
    boxes: list[_Record] | None = None,
    game_nodes: dict[str, int] | None = None,
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
    pa_live_keys = set(live)
    archive_live_keys: set[tuple[str, str]] = set()
    archive_live_scopes: set[tuple[str, str]] = set()
    for record in boxes or ():
        row = record.data
        if row["role"] != "batting":
            continue
        stats = _box_stat_fields(row)
        ab, hits, pa = (stats.get(name) for name in ("at_bats", "hits", "plate_appearances"))
        observed_pa = int(pa) if _observed_nonnegative(pa) else -1
        observed_ab = int(ab) if _observed_nonnegative(ab) else -1
        known_pa, known_ab = observed_pa >= 1, observed_ab >= 1
        if (
            not stats.get("hits_verified", True)
            or not _observed_nonnegative(hits)
            or not (known_pa or known_ab)
        ):
            continue
        if (known_pa and hits > observed_pa) or (observed_ab >= 0 and hits > observed_ab):
            raise ValueError("historical hits exceed observed opportunities")
        key = (row["game_id"], row["player_id"])
        if key in live and row.get("derived_from_pa", False):
            continue
        archive_live_keys.add(key)
        if row.get("identity_status") != "canonical_verified":
            archive_live_scopes.add((row["game_id"], row["team_id"]))
        live[key] = [
            row["team_id"],
            row["opponent_team_id"],
            observed_pa if known_pa else -1,
            int(hits),
            observed_pa if known_pa else observed_ab,
        ]
    for key in pa_live_keys:
        if key not in archive_live_keys and (key[0], live[key][0]) in archive_live_scopes:
            del live[key]
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
    arrays: dict[str, Array] = {
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
        "live_hit_pa_min": np.asarray(
            [live[key][4] if len(live[key]) > 4 else live[key][2] for key in live_keys],
            dtype=np.int64,
        ),
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
    if game_nodes is not None:
        arrays.update(
            {
                "match_game_index": np.asarray(
                    [game_nodes[row["game_id"]] for row in match_rows], dtype=np.int64
                ),
                "live_hit_game_index": np.asarray(
                    [game_nodes[key[0]] for key in live_keys], dtype=np.int64
                ),
                "pa_game_index": np.asarray(
                    [game_nodes[row["game_id"]] for row in pa_rows], dtype=np.int64
                ),
            }
        )
    arrays.update(_box_queries(boxes or [], players, teams, game_nodes))
    return arrays


def _box_queries(
    boxes: list[_Record],
    players: dict[str, int],
    teams: dict[str, int],
    game_nodes: dict[str, int] | None = None,
) -> dict[str, Array]:
    batting, pitching = [], []
    pa_counts, pitch_targets, pitch_masks = [], [], []
    for record in boxes:
        row, stats = record.data, _box_stat_fields(record.data)
        if row["role"] == "batting":
            if not stats.get("counts_verified") or not row.get("box_pa_target_enabled", True):
                continue
            counts = np.asarray(stats.get("outcome_counts"), dtype=np.float32)
            if counts.shape != (10,) or np.any(counts < 0) or not np.isfinite(counts).all():
                raise ValueError("invalid verified historical PA counts")
            if np.any(counts != np.floor(counts)):
                raise ValueError("historical PA outcome counts must be integers")
            if counts.sum() > 0:
                batting.append(row)
                pa_counts.append(counts)
        else:
            permitted = row.get("box_pitch_target_mask", [True] * len(BOX_PITCHING_FIELDS))
            known = [
                allowed and _observed_nonnegative(stats.get(name))
                for name, allowed in zip(BOX_PITCHING_FIELDS, permitted, strict=True)
            ]
            if any(known):
                pitching.append(row)
                pitch_targets.append(
                    [
                        float(stats[name]) if observed else 0.0
                        for name, observed in zip(BOX_PITCHING_FIELDS, known, strict=True)
                    ]
                )
                pitch_masks.append(known)
    arrays: dict[str, Array] = {}
    for prefix, rows in (("box_pa", batting), ("box_pitch", pitching)):
        arrays[f"{prefix}_player_index"] = np.asarray(
            [players[row["player_id"]] for row in rows], dtype=np.int64
        )
        arrays[f"{prefix}_team_index"] = np.asarray(
            [teams[row["team_id"]] for row in rows], dtype=np.int64
        )
        arrays[f"{prefix}_opponent_index"] = np.asarray(
            [teams[row["opponent_team_id"]] for row in rows], dtype=np.int64
        )
        arrays[f"{prefix}_query_ids"] = np.asarray(
            [row["observation_id"] for row in rows], dtype=np.str_
        )
        if game_nodes is not None:
            arrays[f"{prefix}_game_index"] = np.asarray(
                [game_nodes[row["game_id"]] for row in rows], dtype=np.int64
            )
    arrays["box_pa_counts"] = np.asarray(pa_counts, dtype=np.float32).reshape(-1, 10)
    arrays["box_pitch_targets"] = np.asarray(pitch_targets, dtype=np.float32).reshape(-1, 10)
    arrays["box_pitch_mask"] = np.asarray(pitch_masks, dtype=np.bool_).reshape(-1, 10)
    return arrays


def _add_box_defaults(arrays: dict[str, Array], players: int, teams: int) -> None:
    """In-memory defaults preserve loading existing version-two cache files."""
    arrays.setdefault("live_hit_pa_min", arrays["live_hit_pa"].copy())
    for role, dimension in BOX_FEATURE_DIMS.items():
        for kind, count in (("player", players), ("team", teams)):
            arrays.setdefault(
                f"{kind}_box_{role}_features", np.zeros((count, dimension), np.float32)
            )
    for name, value in _box_queries([], {}, {}).items():
        arrays.setdefault(name, value)


_TRAINING_COUNT_FIELDS = (
    "live_hit_queries",
    "live_hit_unknown_pa_queries",
    "pa_queries",
    "box_pa_queries",
    "box_pa_outcomes",
    "box_pitch_queries",
    "box_pitch_observed_counts",
    "pa_derived_batting_queries",
    "pa_derived_pitching_queries",
)


def _target_coverage(labels: list[_Record]) -> dict[str, int]:
    """Count emitted task arrays, including modern PA-derived aggregates."""
    games = [row for row in labels if row.kind == "game"]
    pas = [row for row in labels if row.kind == "pa"]
    boxes = _label_boxes(labels)
    players = sorted(
        {row.data["player_id"] for row in boxes}
        | {row.data[field] for row in pas for field in ("batter_id", "pitcher_id")}
    )
    teams = sorted(
        {row.data[field] for row in games for field in ("home_team_id", "away_team_id")}
        | {row.data[field] for row in pas for field in ("batting_team_id", "fielding_team_id")}
        | {row.data[field] for row in boxes for field in ("team_id", "opponent_team_id")}
    )
    arrays = _queries(
        games,
        pas,
        {key: index for index, key in enumerate(players)},
        {key: index for index, key in enumerate(teams)},
        boxes,
    )
    derived_ids = {
        row.data["observation_id"] for row in boxes if row.data.get("derived_from_pa", False)
    }
    return {
        "live_hit_queries": len(arrays["live_hit_query_ids"]),
        "live_hit_unknown_pa_queries": int(np.sum(arrays["live_hit_pa"] < 0)),
        "pa_queries": len(arrays["pa_targets"]),
        "box_pa_queries": len(arrays["box_pa_counts"]),
        "box_pa_outcomes": int(arrays["box_pa_counts"].sum()),
        "box_pitch_queries": len(arrays["box_pitch_targets"]),
        "box_pitch_observed_counts": int(arrays["box_pitch_mask"].sum()),
        "pa_derived_batting_queries": sum(key in derived_ids for key in arrays["box_pa_query_ids"]),
        "pa_derived_pitching_queries": sum(
            key in derived_ids for key in arrays["box_pitch_query_ids"]
        ),
    }


def _sum_archive_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    names = {name for audit in audits for name in audit if name != "box_target_missing_reasons"}
    reasons = {name for audit in audits for name in audit.get("box_target_missing_reasons", {})}
    return {
        **{name: sum(audit.get(name, 0) for audit in audits) for name in sorted(names)},
        "box_target_missing_reasons": {
            reason: sum(
                audit.get("box_target_missing_reasons", {}).get(reason, 0) for audit in audits
            )
            for reason in sorted(reasons)
        },
    }


def _box_label_audit(labels: list[_Record]) -> dict[str, Any]:
    batting = [record for record in labels if record.kind == "box_batting"]
    pitching = [record for record in labels if record.kind == "box_pitching"]
    reasons: dict[str, int] = defaultdict(int)
    live_known = live_unknown = pa_queries = pitch_queries = pa_outcomes = pitch_observed = 0
    for record in batting:
        stats = _box_stat_fields(record.data)
        pa, ab, hits = (stats.get(name) for name in ("plate_appearances", "at_bats", "hits"))
        if stats.get("counts_verified") and sum(stats.get("outcome_counts", ())) > 0:
            pa_queries += 1
            pa_outcomes += sum(stats["outcome_counts"])
        else:
            reasons["box_pa_unverified_or_zero_counts"] += 1
        if not stats.get("hits_verified", True) or not _observed_nonnegative(hits):
            reasons["live_hit_missing_or_unverified_hits"] += 1
        elif _observed_nonnegative(pa) and pa >= 1:
            live_known += 1
        elif _observed_nonnegative(ab) and ab >= 1:
            live_unknown += 1
        else:
            reasons["live_hit_no_observed_appearance"] += 1
    for record in pitching:
        stats = _box_stat_fields(record.data)
        observed = [_observed_nonnegative(stats.get(name)) for name in BOX_PITCHING_FIELDS]
        if any(observed):
            pitch_queries += 1
            pitch_observed += sum(observed)
        else:
            reasons["box_pitch_no_observed_count_fields"] += 1
        for name, known in zip(BOX_PITCHING_FIELDS, observed, strict=True):
            if not known:
                reasons[f"box_pitch_missing:{name}"] += 1
    for record in (*batting, *pitching):
        usable = _box_stat_fields(record.data)
        raw_stats = record.data["stats_json"]
        fields = BOX_BATTING_FIELDS if record.data["role"] == "batting" else BOX_PITCHING_FIELDS
        for name in fields:
            if _observed_nonnegative(raw_stats.get(name)) and not _observed_nonnegative(
                usable.get(name)
            ):
                reasons[f"unusable_field:{record.data['role']}:{name}"] += 1
        quality = record.data.get("quality_json", {})
        source_reasons = (
            quality.get("reasons", quality.get("quality_reasons", ()))
            if isinstance(quality, dict)
            else quality
        )
        if isinstance(source_reasons, (list, tuple)):
            for reason in source_reasons:
                reasons[f"source:{reason}"] += 1
    return {
        "box_batting_rows": len(batting),
        "box_pitching_rows": len(pitching),
        "box_pa_queries": pa_queries,
        "box_pitch_queries": pitch_queries,
        "box_live_hit_queries": live_known + live_unknown,
        "box_live_hit_unknown_pa_queries": live_unknown,
        "box_pa_outcomes": pa_outcomes,
        "box_pitch_observed_counts": pitch_observed,
        "box_target_missing_reasons": dict(sorted(reasons.items())),
    }


def _identity_audit(labels: list[_Record]) -> dict[str, int]:
    counts = {
        "source_observation_rows": 0,
        "name_team_cohort_rows": 0,
        "team_role_fallback_rows": 0,
        "canonical_verified_rows": 0,
        "same_game_cohort_collision_groups": 0,
    }
    cohorts: dict[tuple[str, str], int] = defaultdict(int)
    for record in labels:
        if not record.kind.startswith("box_"):
            continue
        row = record.data
        if row.get("identity_status") == "canonical_verified":
            counts["canonical_verified_rows"] += 1
        else:
            counts["source_observation_rows"] += 1
            prior = historical_player_prior(
                team_id=row["team_id"], role=row["role"], display_name=row.get("display_name")
            )
            named = prior.identity_status == "source_name_team_cohort"
            counts["name_team_cohort_rows" if named else "team_role_fallback_rows"] += 1
            if named:
                cohorts[(row["game_id"], prior.prior_id)] += 1
    counts["same_game_cohort_collision_groups"] = sum(count > 1 for count in cohorts.values())
    return counts


def _validate_graph(graph: GraphDay) -> None:
    if any(
        len(set(node_ids)) != len(node_ids)
        for node_ids in (graph.player_ids, graph.team_ids, graph.game_ids)
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
        *(
            (("game_features", len(graph.game_ids), len(VNEXT_GAME_FEATURE_NAMES)),)
            if "game_features" in graph.arrays
            else ()
        ),
    ):
        if graph.arrays[name].shape != (size, dimension):
            raise ValueError(f"invalid node feature dimensions: {name}")
    if bool(graph.game_ids) != ("game_features" in graph.arrays):
        raise ValueError("game node IDs/features must be present together")
    route_metadata = (
        VNEXT_ROUTE_METADATA if "game_features" in graph.arrays else ROUTE_METADATA
    )
    if set(graph.routes) != set(route_metadata):
        raise ValueError("graph archive is missing required reviewed route arrays")
    node_sizes = {
        "player": len(graph.player_ids),
        "team": len(graph.team_ids),
        "game": len(graph.game_ids),
    }
    for name, arrays in graph.routes.items():
        metadata = route_metadata[name]
        count = len(arrays["source_index"])
        for side in ("source", "destination"):
            limit = node_sizes[cast(str, metadata[f"{side}_type"])]
            route_index = arrays[f"{side}_index"]
            if (
                route_index.shape != (count,)
                or np.any(route_index < 0)
                or np.any(route_index >= limit)
            ):
                raise ValueError(f"invalid route {side} indices: {name}")
        if arrays["event_features"].shape != (
            count,
            len(VNEXT_ROUTE_FEATURE_NAMES[name]),
        ):
            raise ValueError(f"invalid route features: {name}")
        if np.any(arrays["event_age_seconds"] < arrays["publication_delay_seconds"]):
            raise ValueError(f"route includes information unavailable at cutoff: {name}")
    if "game_features" in graph.arrays:
        query_counts = {
            "match": len(graph.arrays["match_targets"]),
            "live_hit": len(graph.arrays["live_hit_pa"]),
            "pa": len(graph.arrays["pa_targets"]),
            "box_pa": len(graph.arrays["box_pa_counts"]),
            "box_pitch": len(graph.arrays["box_pitch_targets"]),
        }
        for prefix, count in query_counts.items():
            query_game_index = graph.arrays.get(f"{prefix}_game_index")
            if (
                query_game_index is None
                or query_game_index.shape != (count,)
                or np.any(query_game_index < 0)
                or np.any(query_game_index >= len(graph.game_ids))
            ):
                raise ValueError(f"invalid {prefix} game indices")
    live_pa = graph.arrays["live_hit_pa"]
    live_min = graph.arrays.get("live_hit_pa_min", live_pa)
    if (
        live_min.shape != live_pa.shape
        or np.any(live_min < 1)
        or np.any((live_pa < 1) & (live_pa != -1))
    ):
        raise ValueError(
            "conditional LiveHit targets require observed PA or a positive lower bound"
        )
    if np.any((live_pa >= 1) & (live_min != live_pa)):
        raise ValueError("known LiveHit PA must equal its lower bound")
    if np.any((live_pa >= 1) & (graph.arrays["live_hit_hits"] > live_pa)):
        raise ValueError("hits exceed observed PA")
    if np.any(graph.arrays["live_hit_hits"] < 0) or np.any(
        graph.arrays["live_hit_hits"] > live_min
    ):
        raise ValueError("hits exceed the observed PA/AB lower bound")
    for role, dimension in BOX_FEATURE_DIMS.items():
        for kind, count in (("player", len(graph.player_ids)), ("team", len(graph.team_ids))):
            name = f"{kind}_box_{role}_features"
            if name in graph.arrays and graph.arrays[name].shape != (count, dimension):
                raise ValueError(f"invalid boxscore feature dimensions: {name}")
    for prefix in ("box_pa", "box_pitch"):
        if f"{prefix}_player_index" not in graph.arrays:
            continue
        count = len(graph.arrays[f"{prefix}_player_index"])
        for field_name, limit in (
            ("player", len(graph.player_ids)),
            ("team", len(graph.team_ids)),
            ("opponent", len(graph.team_ids)),
        ):
            index = graph.arrays[f"{prefix}_{field_name}_index"]
            if index.shape != (count,) or np.any(index < 0) or np.any(index >= limit):
                raise ValueError(f"invalid {prefix} {field_name} indices")
        if graph.arrays[f"{prefix}_query_ids"].shape != (count,):
            raise ValueError(f"invalid {prefix} query IDs")
        target_name = "box_pa_counts" if prefix == "box_pa" else "box_pitch_targets"
        target = graph.arrays[target_name]
        if target.shape != (count, 10) or np.any(target < 0):
            raise ValueError(f"invalid {target_name}")
        if prefix == "box_pa" and (
            np.any(target.sum(axis=1) <= 0) or np.any(target != np.floor(target))
        ):
            raise ValueError("historical PA counts must be positive-sum integer count vectors")
        if prefix == "box_pitch":
            mask = graph.arrays["box_pitch_mask"]
            if mask.shape != (count, 10) or mask.dtype != np.bool_ or np.any(~mask.any(axis=1)):
                raise ValueError("invalid historical pitching observation mask")


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


def _open_file_identity(handle: Any) -> tuple[int, int, int, int, int]:
    """Return the metadata that must remain stable around one archive read."""

    stat = os.fstat(handle.fileno())
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _sha256_open_file(handle: Any) -> str:
    """Hash an already-open file and leave it ready for a subsequent reader."""

    digest = hashlib.sha256()
    handle.seek(0)
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _read_npz_arrays(handle: Any) -> dict[str, Array]:
    """Read every member while the integrity-checked file descriptor stays open."""

    with np.load(handle, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _read_verified_graph_archive(
    path: Path,
    expected_sha256: str,
    day_key: str,
) -> tuple[dict[str, Array], tuple[int, int, int, int, int]]:
    """Verify and load the same open file, rejecting concurrent mutations.

    A cache hit skips only the SHA-256 pass.  Parsing and the caller's complete
    graph validation still run on every load.  The descriptor identity is
    checked both before and after NumPy reads so a concurrent in-place write
    cannot turn an earlier successful hash into a TOCTOU bypass.
    """

    with path.open("rb") as handle:
        before = _open_file_identity(handle)
        with _VERIFIED_GRAPH_FILES_LOCK:
            cached = _VERIFIED_GRAPH_FILES.get(path) == (expected_sha256, before)
        if not cached and _sha256_open_file(handle) != expected_sha256:
            raise ValueError(f"graph cache checksum mismatch: {day_key}")
        arrays = _read_npz_arrays(handle)
        after = _open_file_identity(handle)
    if after != before:
        with _VERIFIED_GRAPH_FILES_LOCK:
            _VERIFIED_GRAPH_FILES.pop(path, None)
        raise ValueError(f"graph cache changed while loading: {day_key}")
    return arrays, before


def _remember_verified_graph_file(
    path: Path,
    expected_sha256: str,
    identity: tuple[int, int, int, int, int],
) -> None:
    """Publish a cache entry only after archive parsing and graph validation."""

    with _VERIFIED_GRAPH_FILES_LOCK:
        _VERIFIED_GRAPH_FILES[path] = (expected_sha256, identity)


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
