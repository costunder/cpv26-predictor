"""Bulk adapter from the public KBO play-by-play Parquet files to schema v4.

The upstream dataset contains one row per pitch.  This module deliberately
keeps acquisition separate from ingestion and reduces pitches to one observed
plate-appearance row before writing the canonical append-only tables.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .kbo_playbyplay import (
    KBO_PLAYBYPLAY_FILES,
    KBO_PLAYBYPLAY_REPOSITORY_URL,
    KBO_PLAYBYPLAY_REVISION,
    sha256_file,
)
from .kbo_source_snapshots import ANNUAL_SNAPSHOT_POLICY
from .store import DuckDBStore

DEFAULT_DATASET_REVISION = KBO_PLAYBYPLAY_REVISION
KBO_ADAPTER_VERSION = 2
DATASET_NAME = "slothman3878/kbo_playbyplay"
DATASET_PAGE = KBO_PLAYBYPLAY_REPOSITORY_URL
UPSTREAM_REPOSITORY = "https://github.com/slothman3878/kbo_pbp_naver_sports"

_FILE_PATTERN = re.compile(r"^kbo_pbp_(?P<season>20\d{2})\.parquet$")
_KST = ZoneInfo("Asia/Seoul")
_REQUIRED_COLUMNS = frozenset(
    {
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "strikes",
        "batter",
        "pitcher",
        "batter_name",
        "pitcher_name",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "home_score",
        "away_score",
        "stand",
        "events",
        "post_home_score",
        "post_away_score",
        "post_outs",
        "runs_scored",
        "post_on_1b",
        "post_on_2b",
        "post_on_3b",
    }
)
_TARGET_TABLES = (
    "source_revision",
    "team",
    "player",
    "game",
    "team_game",
    "observed_plate_appearance",
)
_TEMP_OBJECTS = (
    "_kbo_team_game_stats",
    "_kbo_game",
    "_kbo_pa",
    "_kbo_source_context",
    "_kbo_pitch",
)


class KBOIngestError(ValueError):
    """Raised when source files cannot satisfy the canonical data contract."""


@dataclass(frozen=True, slots=True)
class KBOImportFile:
    """Validated identity and audit metadata for one season file."""

    path: Path
    season: int
    content_sha256: str
    pitch_rows: int
    game_rows: int
    first_game_date: date
    last_game_date: date
    source_revision_id: str


@dataclass(frozen=True, slots=True)
class KBOImportReport:
    """Counts returned by a retry-safe play-by-play import."""

    revision: str
    files: tuple[KBOImportFile, ...]
    inserted_rows: Mapping[str, int]
    total_rows: Mapping[str, int]
    completed_plate_appearances: int
    unlabelled_plate_appearances: int
    unlabelled_runs: int
    invalid_score_transitions: int
    unreconciled_score_games: int
    source_unallocated_runs: int
    source_sequence_gaps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "adapter_version": KBO_ADAPTER_VERSION,
            "files": [
                {
                    "path": str(item.path),
                    "season": item.season,
                    "content_sha256": item.content_sha256,
                    "pitch_rows": item.pitch_rows,
                    "game_rows": item.game_rows,
                    "first_game_date": item.first_game_date.isoformat(),
                    "last_game_date": item.last_game_date.isoformat(),
                    "source_revision_id": item.source_revision_id,
                }
                for item in self.files
            ],
            "inserted_rows": dict(self.inserted_rows),
            "total_rows": dict(self.total_rows),
            "completed_plate_appearances": self.completed_plate_appearances,
            "unlabelled_plate_appearances": self.unlabelled_plate_appearances,
            "unlabelled_runs": self.unlabelled_runs,
            "invalid_score_transitions": self.invalid_score_transitions,
            "unreconciled_score_games": self.unreconciled_score_games,
            "source_unallocated_runs": self.source_unallocated_runs,
            "source_sequence_gaps": self.source_sequence_gaps,
            "simulator_ready": False,
            "quality_notes": [
                "Historical publication and start times are reconstructed from game dates.",
                "Unlabelled at-bats remain in the source Parquet but are not PA targets.",
                "Runner-only events and official fielding-error totals are not inferred.",
                "Inconsistent PA score transitions are marked transition_complete=false.",
                "This import supports baselines, not a complete sequential-event replay.",
            ],
        }


def import_kbo_playbyplay(
    store: DuckDBStore,
    parquet_files: Iterable[str | Path],
    *,
    revision: str = DEFAULT_DATASET_REVISION,
    ingested_at: datetime | None = None,
) -> KBOImportReport:
    """Import one or more annual files into the canonical v4 tables.

    Historical publication timestamps are not present upstream.  The adapter
    therefore applies a documented retrospective policy: a completed game's
    labels become available at 00:00 Asia/Seoul on the following calendar day.
    ``ingested_at`` still records when this local repository actually observed
    the pinned file, so provenance is not confused with event availability.

    Rows without a terminal ``events`` label include interrupted at-bats (for
    example, an inning-ending caught stealing) and any unclassified outcomes.
    They are counted in the report, not silently assigned an out or a hit.
    """

    if store.read_only:
        raise PermissionError("KBO import requires a writable DuckDBStore")
    if not revision.strip():
        raise KBOIngestError("revision cannot be empty")
    observed_at = ingested_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise KBOIngestError("ingested_at must include timezone information")
    observed_at = observed_at.astimezone(timezone.utc)

    paths = tuple(Path(value).expanduser().resolve() for value in parquet_files)
    if not paths:
        raise KBOIngestError("at least one Parquet file is required")
    if len(paths) != len(set(paths)):
        raise KBOIngestError("the same Parquet file was supplied more than once")

    connection = store.connection
    files = _validate_files(connection, paths, revision)
    source_ids = tuple(item.source_revision_id for item in files)
    _assert_existing_source_hashes(store, files)
    before = _target_counts(connection, source_ids)

    try:
        connection.read_parquet([str(item.path) for item in files], union_by_name=True).create_view(
            "_kbo_pitch", replace=True
        )
        _create_context(connection, files, observed_at)
        _create_reduced_tables(connection)
        _validate_reduced_tables(connection)
        plate_appearance_counts = connection.execute(
            """
            SELECT
                count(*) FILTER (WHERE events IS NOT NULL),
                count(*) FILTER (WHERE events IS NULL),
                coalesce(sum(runs_scored) FILTER (WHERE events IS NULL), 0)
            FROM _kbo_pa
            """
        ).fetchone()
        if plate_appearance_counts is None:
            raise RuntimeError("failed to count reduced KBO plate appearances")
        completed, unlabelled, unlabelled_runs = plate_appearance_counts
        score_transition_count = connection.execute(
            """
                SELECT count(*)
                FROM _kbo_pa
                WHERE events IS NOT NULL AND (
                    post_home_score - home_score_before
                        <> CASE WHEN inning_topbot = 'top' THEN 0 ELSE runs_scored END
                    OR post_away_score - away_score_before
                        <> CASE WHEN inning_topbot = 'top' THEN runs_scored ELSE 0 END
                )
                """
        ).fetchone()
        if score_transition_count is None:
            raise RuntimeError("failed to audit KBO plate-appearance score transitions")
        invalid_score_transitions = int(score_transition_count[0])
        score_reconciliation = connection.execute(
            """
            WITH pa_runs AS (
                SELECT game_pk, sum(runs_scored) AS runs_scored
                FROM _kbo_pa
                GROUP BY game_pk
            )
            SELECT
                count(*) FILTER (WHERE home_score + away_score <> runs_scored),
                coalesce(sum(home_score + away_score - runs_scored), 0)
            FROM _kbo_game
            JOIN pa_runs USING (game_pk)
            """
        ).fetchone()
        if score_reconciliation is None:
            raise RuntimeError("failed to reconcile KBO game scores")
        unreconciled_score_games, source_unallocated_runs = score_reconciliation
        sequence_gap_row = connection.execute(
            """
            WITH ordered AS (
                SELECT at_bat_number, lag(at_bat_number) OVER (
                    PARTITION BY game_pk ORDER BY at_bat_number
                ) AS previous_number
                FROM _kbo_pa
            )
            SELECT count(*)
            FROM ordered
            WHERE at_bat_number > previous_number + 1
            """
        ).fetchone()
        if sequence_gap_row is None:
            raise RuntimeError("failed to audit KBO source sequence gaps")

        with store.transaction():
            for item in files:
                event_at = _end_of_game_day(item.first_game_date)
                available_at = _next_day_available(item.first_game_date)
                store.append(
                    "source_revision",
                    {
                        "source_revision_id": item.source_revision_id,
                        "source_name": DATASET_NAME,
                        "source_locator": (
                            f"{DATASET_PAGE}/resolve/{revision}/v0/{item.path.name}"
                        ),
                        "content_sha256": item.content_sha256,
                        "metadata_json": {
                            "dataset_revision": revision,
                            "adapter_version": KBO_ADAPTER_VERSION,
                            "season": item.season,
                            "snapshot_policy": ANNUAL_SNAPSHOT_POLICY,
                            "snapshot_scope": "provider_and_season",
                            "doubleheader_policy": (
                                "validated_game_id_suffix_0_null_1_first_2_second"
                            ),
                            "license_declared_by_dataset": "CC-BY-4.0",
                            "upstream_repository": UPSTREAM_REPOSITORY,
                            "pitch_rows": item.pitch_rows,
                            "game_rows": item.game_rows,
                            "event_time_policy": "game_date_end_of_day_kst",
                            "availability_policy": "next_calendar_day_00:00_kst",
                            "scheduled_start_policy": "game_date_00:00_kst_imputed",
                            "batter_attribution_policy": (
                                "terminal_batter_except_two_strike_substitution_strikeout"
                            ),
                            "pitcher_attribution_policy": (
                                "last_observed_pitcher_not_official_run_responsibility"
                            ),
                            "retrospective_reconstruction": True,
                            "unavailable_fields": [
                                "official_team_errors",
                                "player_runs_and_rbi",
                                "authoritative_player_handedness",
                                "catcher_id",
                                "stadium_id",
                                "historical_lineup_announcements",
                            ],
                        },
                        "event_at": event_at,
                        "available_at": available_at,
                        "ingested_at": observed_at,
                        "valid_from": event_at,
                        "valid_to": None,
                    },
                    ignore_existing=True,
                )
            _insert_teams(connection)
            _insert_players(connection)
            _insert_games(connection)
            _insert_team_games(connection)
            _insert_plate_appearances(connection)
    finally:
        _drop_temp_objects(connection)

    after = _target_counts(connection, source_ids)
    inserted = {name: after[name] - before[name] for name in _TARGET_TABLES}
    return KBOImportReport(
        revision=revision,
        files=files,
        inserted_rows=inserted,
        total_rows=after,
        completed_plate_appearances=int(completed),
        unlabelled_plate_appearances=int(unlabelled),
        unlabelled_runs=int(unlabelled_runs),
        invalid_score_transitions=invalid_score_transitions,
        unreconciled_score_games=int(unreconciled_score_games),
        source_unallocated_runs=int(source_unallocated_runs),
        source_sequence_gaps=int(sequence_gap_row[0]),
    )


def _validate_files(
    connection: Any,
    paths: tuple[Path, ...],
    revision: str,
) -> tuple[KBOImportFile, ...]:
    files: list[KBOImportFile] = []
    seen_seasons: set[int] = set()
    for path in paths:
        if not path.is_file():
            raise KBOIngestError(f"Parquet file not found: {path}")
        match = _FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise KBOIngestError(f"expected file name kbo_pbp_YYYY.parquet, got: {path.name}")
        season = int(match.group("season"))
        if season in seen_seasons:
            raise KBOIngestError(f"more than one file supplied for season {season}")
        seen_seasons.add(season)

        columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
        }
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise KBOIngestError(f"{path.name} is missing required columns: {', '.join(missing)}")
        pitch_rows, game_rows, first_raw, last_raw, season_count = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT game_pk),
                min(CAST(game_date AS DATE)),
                max(CAST(game_date AS DATE)),
                count(DISTINCT year(CAST(game_date AS DATE)))
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
        if not pitch_rows or first_raw is None or last_raw is None:
            raise KBOIngestError(f"{path.name} contains no pitch rows")
        if season_count != 1 or first_raw.year != season or last_raw.year != season:
            raise KBOIngestError(
                f"{path.name} contains game dates outside declared season {season}"
            )
        content_sha256 = sha256_file(path)
        if revision == DEFAULT_DATASET_REVISION:
            expected_hashes = {item.year: item.sha256 for item in KBO_PLAYBYPLAY_FILES}
            expected = expected_hashes.get(season)
            if expected is None:
                raise KBOIngestError(f"season {season} is not part of the pinned dataset")
            if content_sha256 != expected:
                raise KBOIngestError(
                    f"SHA-256 mismatch for {path.name}: expected {expected}, got {content_sha256}"
                )
        files.append(
            KBOImportFile(
                path=path,
                season=season,
                content_sha256=content_sha256,
                pitch_rows=int(pitch_rows),
                game_rows=int(game_rows),
                first_game_date=first_raw,
                last_game_date=last_raw,
                source_revision_id=(
                    f"hf-kbo-playbyplay:{revision}:{season}:adapter-v{KBO_ADAPTER_VERSION}"
                ),
            )
        )
    return tuple(sorted(files, key=lambda item: item.season))


def _assert_existing_source_hashes(
    store: DuckDBStore,
    files: tuple[KBOImportFile, ...],
) -> None:
    for item in files:
        row = store.connection.execute(
            """
            SELECT content_sha256
            FROM source_revision
            WHERE source_revision_id = ?
            """,
            [item.source_revision_id],
        ).fetchone()
        if row is not None and str(row[0]).lower() != item.content_sha256:
            raise KBOIngestError(
                f"existing source revision has a different content hash: {item.source_revision_id}"
            )


def _create_context(
    connection: Any,
    files: tuple[KBOImportFile, ...],
    ingested_at: datetime,
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE _kbo_source_context (
            season INTEGER,
            source_revision_id VARCHAR,
            ingested_at TIMESTAMPTZ
        )
        """
    )
    connection.executemany(
        "INSERT INTO _kbo_source_context VALUES (?, ?, ?)",
        [(item.season, item.source_revision_id, ingested_at) for item in files],
    )


def _create_reduced_tables(connection: Any) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE _kbo_pa AS
        SELECT
            CAST(game_pk AS VARCHAR) AS game_pk,
            CAST(game_date AS DATE) AS game_date,
            CAST(any_value(home_team) AS VARCHAR) AS home_team,
            CAST(any_value(away_team) AS VARCHAR) AS away_team,
            CAST(at_bat_number AS INTEGER) AS at_bat_number,
            CAST(first(inning ORDER BY pitch_number) AS INTEGER) AS inning,
            CAST(first(inning_topbot ORDER BY pitch_number) AS VARCHAR) AS inning_topbot,
            CAST(CASE
                WHEN last(events ORDER BY pitch_number) = 'strikeout' THEN coalesce(
                    last(batter ORDER BY pitch_number) FILTER (WHERE strikes < 2),
                    first(batter ORDER BY pitch_number)
                )
                ELSE last(batter ORDER BY pitch_number)
            END AS VARCHAR) AS batter,
            CAST(last(pitcher ORDER BY pitch_number) AS VARCHAR) AS pitcher,
            CAST(first(outs_when_up ORDER BY pitch_number) AS INTEGER) AS outs_before,
            CAST(first(on_1b ORDER BY pitch_number) AS VARCHAR) AS on_1b,
            CAST(first(on_2b ORDER BY pitch_number) AS VARCHAR) AS on_2b,
            CAST(first(on_3b ORDER BY pitch_number) AS VARCHAR) AS on_3b,
            CAST(first(home_score ORDER BY pitch_number) AS INTEGER) AS home_score_before,
            CAST(first(away_score ORDER BY pitch_number) AS INTEGER) AS away_score_before,
            CAST(last(events ORDER BY pitch_number) AS VARCHAR) AS events,
            CAST(last(post_outs ORDER BY pitch_number) AS INTEGER) AS post_outs,
            CAST(last(post_on_1b ORDER BY pitch_number) AS VARCHAR) AS post_on_1b,
            CAST(last(post_on_2b ORDER BY pitch_number) AS VARCHAR) AS post_on_2b,
            CAST(last(post_on_3b ORDER BY pitch_number) AS VARCHAR) AS post_on_3b,
            CAST(last(post_home_score ORDER BY pitch_number) AS INTEGER) AS post_home_score,
            CAST(last(post_away_score ORDER BY pitch_number) AS INTEGER) AS post_away_score,
            CAST(coalesce(last(runs_scored ORDER BY pitch_number), 0) AS INTEGER)
                AS runs_scored
        FROM _kbo_pitch
        GROUP BY game_pk, game_date, at_bat_number
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE _kbo_game AS
        SELECT
            CAST(game_pk AS VARCHAR) AS game_pk,
            CAST(game_date AS DATE) AS game_date,
            CAST(any_value(home_team) AS VARCHAR) AS home_team,
            CAST(any_value(away_team) AS VARCHAR) AS away_team,
            CAST(max(post_home_score) AS INTEGER) AS home_score,
            CAST(max(post_away_score) AS INTEGER) AS away_score
        FROM _kbo_pitch
        GROUP BY game_pk, game_date
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE _kbo_team_game_stats AS
        WITH completed AS (
            SELECT
                game_pk,
                CASE WHEN inning_topbot = 'top' THEN away_team ELSE home_team END
                    AS batting_team,
                events
            FROM _kbo_pa
            WHERE events IS NOT NULL
        ), batting AS (
            SELECT
                game_pk,
                batting_team AS team,
                count(*) FILTER (
                    WHERE events IN ('single', 'double', 'triple', 'home_run')
                ) AS hits
            FROM completed
            GROUP BY game_pk, batting_team
        )
        SELECT
            side.game_pk,
            side.team,
            coalesce(batting.hits, 0)::INTEGER AS hits,
            NULL::INTEGER AS errors
        FROM (
            SELECT game_pk, home_team AS team FROM _kbo_game
            UNION ALL
            SELECT game_pk, away_team AS team FROM _kbo_game
        ) AS side
        LEFT JOIN batting USING (game_pk, team)
        """
    )


def _validate_reduced_tables(connection: Any) -> None:
    invalid_games = connection.execute(
        """
        SELECT game_pk
        FROM _kbo_game
        GROUP BY game_pk
        HAVING count(*) <> 1
           OR count(home_score) <> 1
           OR count(away_score) <> 1
        LIMIT 1
        """
    ).fetchone()
    if invalid_games is not None:
        raise KBOIngestError(f"ambiguous or incomplete game: {invalid_games[0]}")
    invalid_game_id = connection.execute(
        """
        SELECT game_pk FROM _kbo_game
        WHERE game_pk IS NULL
           OR NOT regexp_full_match(game_pk, '[0-9]{8}[A-Z]{4}[012][0-9]{4}')
           OR substr(game_pk, 1, 8) <> strftime(game_date, '%Y%m%d')
           OR substr(game_pk, 9, 2) <> away_team OR away_team IS NULL
           OR substr(game_pk, 11, 2) <> home_team OR home_team IS NULL
           OR substr(game_pk, 14, 4) <> strftime(game_date, '%Y')
        LIMIT 1
        """
    ).fetchone()
    if invalid_game_id is not None:
        raise KBOIngestError(f"game ID/date/team contract mismatch: {invalid_game_id[0]}")
    invalid_pa = connection.execute(
        """
        SELECT game_pk, at_bat_number, events
        FROM _kbo_pa
        WHERE events IS NOT NULL AND (
            events NOT IN (
                'field_out', 'strikeout', 'single', 'walk', 'double',
                'fielders_choice', 'home_run', 'double_play', 'hit_by_pitch',
                'field_error', 'sac_fly', 'sac_bunt', 'triple',
                'catcher_interference', 'triple_play'
            )
            OR inning_topbot NOT IN ('top', 'bot', 'bottom')
            OR inning_topbot IS NULL
            OR batter IS NULL OR pitcher IS NULL
            OR outs_before NOT BETWEEN 0 AND 2 OR outs_before IS NULL
            OR post_outs - outs_before NOT BETWEEN 0 AND 3 OR post_outs IS NULL
            OR home_score_before IS NULL OR away_score_before IS NULL
            OR post_home_score IS NULL OR post_away_score IS NULL
        )
        LIMIT 1
        """
    ).fetchone()
    if invalid_pa is not None:
        raise KBOIngestError(
            "unsupported or incomplete terminal plate appearance: "
            f"{invalid_pa[0]}:{invalid_pa[1]} ({invalid_pa[2]})"
        )


def _insert_teams(connection: Any) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO team (
            team_row_id, team_id, team_name, short_name, city,
            active_from, active_to, source_revision_id,
            event_at, available_at, ingested_at, valid_from, valid_to
        )
        WITH appearances AS (
            SELECT game_date, home_team AS team_code FROM _kbo_game
            UNION ALL
            SELECT game_date, away_team AS team_code FROM _kbo_game
        ), grouped AS (
            SELECT
                year(game_date)::INTEGER AS season,
                team_code,
                min(game_date) AS first_date
            FROM appearances
            GROUP BY year(game_date), team_code
        )
        SELECT
            source_revision_id || ':team-row:' || team_code,
            'kbo-team:' || team_code,
            team_code,
            team_code,
            NULL,
            first_date,
            NULL,
            source_revision_id,
            ((first_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            (((first_date + INTERVAL 1 DAY)::TIMESTAMP) AT TIME ZONE 'Asia/Seoul'),
            ingested_at,
            ((first_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            NULL
        FROM grouped
        JOIN _kbo_source_context USING (season)
        """
    )


def _insert_players(connection: Any) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO player (
            player_row_id, player_id, display_name, birth_date, bats, throws,
            primary_position, debut_year, source_revision_id,
            event_at, available_at, ingested_at, valid_from, valid_to
        )
        WITH appearances AS (
            SELECT
                CAST(game_date AS DATE) AS game_date,
                CAST(batter AS VARCHAR) AS player_code,
                CAST(batter_name AS VARCHAR) AS display_name,
                CAST(stand AS VARCHAR) AS stand
            FROM _kbo_pitch
            UNION ALL
            SELECT
                CAST(game_date AS DATE),
                CAST(pitcher AS VARCHAR),
                CAST(pitcher_name AS VARCHAR),
                NULL
            FROM _kbo_pitch
        ), grouped AS (
            SELECT
                year(game_date)::INTEGER AS season,
                player_code,
                coalesce(
                    first(nullif(trim(display_name), '') ORDER BY game_date), player_code
                ) AS display_name,
                min(game_date) AS first_date
            FROM appearances
            GROUP BY year(game_date), player_code
        )
        SELECT
            source_revision_id || ':player-row:' || player_code,
            'kbo-player:' || player_code,
            display_name,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            source_revision_id,
            ((first_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            (((first_date + INTERVAL 1 DAY)::TIMESTAMP) AT TIME ZONE 'Asia/Seoul'),
            ingested_at,
            ((first_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            NULL
        FROM grouped
        JOIN _kbo_source_context USING (season)
        """
    )


def _insert_games(connection: Any) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO game (
            game_row_id, game_id, season, game_type, scheduled_start,
            home_team_id, away_team_id, stadium_id, doubleheader_number,
            resumed_from_game_id, game_status, home_score, away_score,
            source_revision_id, event_at, available_at, ingested_at,
            valid_from, valid_to
        )
        SELECT
            source_revision_id || ':game-row:' || game_pk,
            'kbo-game:' || game_pk,
            year(game_date)::INTEGER,
            'regular',
            (game_date::TIMESTAMP AT TIME ZONE 'Asia/Seoul'),
            'kbo-team:' || home_team,
            'kbo-team:' || away_team,
            NULL,
            nullif(CAST(substr(game_pk, 13, 1) AS INTEGER), 0),
            NULL,
            'final',
            home_score,
            away_score,
            source_revision_id,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            (((game_date + INTERVAL 1 DAY)::TIMESTAMP) AT TIME ZONE 'Asia/Seoul'),
            ingested_at,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            NULL
        FROM _kbo_game
        JOIN _kbo_source_context ON season = year(game_date)
        """
    )


def _insert_team_games(connection: Any) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO team_game (
            team_game_row_id, team_game_id, game_id, team_id,
            opponent_team_id, is_home, runs, hits, errors, result,
            source_revision_id, event_at, available_at, ingested_at,
            valid_from, valid_to
        )
        WITH sides AS (
            SELECT
                game_pk, game_date, home_team AS team, away_team AS opponent,
                TRUE AS is_home, home_score AS runs, home_score, away_score
            FROM _kbo_game
            UNION ALL
            SELECT
                game_pk, game_date, away_team, home_team,
                FALSE, away_score, home_score, away_score
            FROM _kbo_game
        )
        SELECT
            source_revision_id || ':team-game-row:' || game_pk || ':' || team,
            'kbo-team-game:' || game_pk || ':' || team,
            'kbo-game:' || game_pk,
            'kbo-team:' || team,
            'kbo-team:' || opponent,
            is_home,
            runs,
            stats.hits,
            stats.errors,
            CASE
                WHEN home_score = away_score THEN 'draw'
                WHEN (is_home AND home_score > away_score)
                    OR (NOT is_home AND away_score > home_score) THEN 'win'
                ELSE 'loss'
            END,
            source_revision_id,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            (((game_date + INTERVAL 1 DAY)::TIMESTAMP) AT TIME ZONE 'Asia/Seoul'),
            ingested_at,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            NULL
        FROM sides
        JOIN _kbo_source_context ON season = year(game_date)
        JOIN _kbo_team_game_stats AS stats USING (game_pk, team)
        """
    )


def _insert_plate_appearances(connection: Any) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO observed_plate_appearance (
            observed_pa_row_id, plate_appearance_id, game_id, inning,
            half_inning, sequence_in_game, event_subsequence,
            batter_id, pitcher_id, catcher_id, batting_team_id, fielding_team_id,
            home_score_before, away_score_before, outs_before, runners_before,
            outs_added, runners_after, home_score_after, away_score_after,
            transition_complete, outcome, is_at_bat, is_hit, total_bases,
            runs_scored, source_revision_id, event_at, available_at,
            ingested_at, valid_from, valid_to
        )
        SELECT
            source_revision_id || ':pa-row:' || game_pk || ':' || at_bat_number,
            'kbo-pa:' || game_pk || ':' || at_bat_number,
            'kbo-game:' || game_pk,
            inning,
            CASE WHEN inning_topbot = 'top' THEN 'top' ELSE 'bottom' END,
            at_bat_number,
            0,
            'kbo-player:' || batter,
            'kbo-player:' || pitcher,
            NULL,
            'kbo-team:' || CASE
                WHEN inning_topbot = 'top' THEN away_team ELSE home_team
            END,
            'kbo-team:' || CASE
                WHEN inning_topbot = 'top' THEN home_team ELSE away_team
            END,
            home_score_before,
            away_score_before,
            outs_before,
            (CASE WHEN on_1b IS NULL THEN '0' ELSE '1' END)
                || (CASE WHEN on_2b IS NULL THEN '0' ELSE '1' END)
                || (CASE WHEN on_3b IS NULL THEN '0' ELSE '1' END),
            post_outs - outs_before,
            CASE WHEN post_outs = 3 THEN '000' ELSE
                (CASE WHEN post_on_1b IS NULL THEN '0' ELSE '1' END)
                    || (CASE WHEN post_on_2b IS NULL THEN '0' ELSE '1' END)
                    || (CASE WHEN post_on_3b IS NULL THEN '0' ELSE '1' END)
            END,
            post_home_score,
            post_away_score,
            (
                post_home_score - home_score_before
                    = CASE WHEN inning_topbot = 'top' THEN 0 ELSE runs_scored END
                AND post_away_score - away_score_before
                    = CASE WHEN inning_topbot = 'top' THEN runs_scored ELSE 0 END
            ),
            CASE events
                WHEN 'field_out' THEN 'ball_in_play_out'
                WHEN 'field_error' THEN 'reached_on_error'
                WHEN 'sac_fly' THEN 'sacrifice_fly'
                WHEN 'sac_bunt' THEN 'sacrifice_bunt'
                WHEN 'triple_play' THEN 'ball_in_play_out'
                ELSE events
            END,
            events NOT IN (
                'walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt',
                'catcher_interference'
            ),
            events IN ('single', 'double', 'triple', 'home_run'),
            CASE events
                WHEN 'single' THEN 1
                WHEN 'double' THEN 2
                WHEN 'triple' THEN 3
                WHEN 'home_run' THEN 4
                ELSE 0
            END,
            runs_scored,
            source_revision_id,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            (((game_date + INTERVAL 1 DAY)::TIMESTAMP) AT TIME ZONE 'Asia/Seoul'),
            ingested_at,
            ((game_date::TIMESTAMP + INTERVAL '23:59:59') AT TIME ZONE 'Asia/Seoul'),
            NULL
        FROM _kbo_pa
        JOIN _kbo_source_context ON season = year(game_date)
        WHERE events IS NOT NULL
        """
    )


def _target_counts(
    connection: Any,
    source_ids: tuple[str, ...],
) -> dict[str, int]:
    placeholders = ", ".join("?" for _ in source_ids)
    counts: dict[str, int] = {}
    for table in _TARGET_TABLES:
        counts[table] = int(
            connection.execute(
                f'SELECT count(*) FROM "{table}" WHERE "source_revision_id" IN ({placeholders})',
                list(source_ids),
            ).fetchone()[0]
        )
    return counts


def _drop_temp_objects(connection: Any) -> None:
    for name in _TEMP_OBJECTS:
        kind = "VIEW" if name == "_kbo_pitch" else "TABLE"
        connection.execute(f'DROP {kind} IF EXISTS "{name}"')


def _end_of_game_day(value: date) -> datetime:
    return datetime.combine(value, time(23, 59, 59), tzinfo=_KST).astimezone(timezone.utc)


def _next_day_available(value: date) -> datetime:
    return datetime.combine(value + timedelta(days=1), time.min, tzinfo=_KST).astimezone(
        timezone.utc
    )


def write_import_report(report: KBOImportReport, destination: str | Path) -> Path:
    """Persist a deterministic, human-readable import report."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


__all__ = [
    "DATASET_NAME",
    "DATASET_PAGE",
    "DEFAULT_DATASET_REVISION",
    "KBOImportFile",
    "KBOImportReport",
    "KBOIngestError",
    "import_kbo_playbyplay",
    "write_import_report",
]
