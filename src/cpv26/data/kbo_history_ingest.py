"""Import historical games and lossless, partially observed player box scores."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .kbo_history_boxscore import parse_historical_boxscore
from .kbo_history_source import (
    EXPECTED_HISTORY_GAMES,
    KBO_HISTORY_FILES,
    KBOHistoryArtifact,
    select_history_artifacts,
)
from .kbo_playbyplay import sha256_file
from .store import DuckDBStore

_KST = ZoneInfo("Asia/Seoul")
_GAME_KEY = re.compile(r"^(\d{8})_([A-Z]{2})([A-Z]{2})([012])$")
_CORE_TABLES = ("source_revision", "team", "game", "team_game")
_TABLES = (*_CORE_TABLES, "historical_boxscore", "historical_game_detail")
# The published official schedule marks this tie-breaker SR_ID=6, not regular SR_ID=0.
_NON_REGULAR = frozenset({"20211031_KTSS0"})
_TEAM_NAMES = {
    "LG": {"LG"},
    "SK": {"SK", "SSG"},
    "OB": {"OB", "두산"},
    "HT": {"해태", "KIA", "기아"},
    "HD": {"현대"},
    "LT": {"롯데"},
    "HH": {"한화"},
    "SS": {"삼성"},
    "WO": {"우리", "넥센", "키움", "히어로즈"},
    "NC": {"NC"},
    "KT": {"KT", "kt"},
}


@dataclass(frozen=True)
class HistoricalGame:
    key: str
    day: date
    away: str
    home: str
    doubleheader: int
    away_score: int
    home_score: int
    away_hits: int | None
    home_hits: int | None
    away_errors: int | None
    home_errors: int | None

    @property
    def game_pk(self) -> str:
        # Same natural identity as the NAVER/Hugging Face adapter.
        return self.key.replace("_", "") + str(self.day.year)


def _integer(value: Any, field: str, *, optional: bool = False) -> int | None:
    if optional and value in (None, "", "-"):
        return None
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"invalid historical {field}: {value!r}")
    return int(value)


def parse_historical_game(key: str, contents: Mapping[str, Any], year: int) -> HistoricalGame:
    if not isinstance(key, str) or not isinstance(contents, Mapping):
        raise ValueError("historical game must have a string ID and object contents")
    match = _GAME_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid historical game ID: {key}")
    day_text, away, home, doubleheader = match.groups()
    day = date.fromisoformat(f"{day_text[:4]}-{day_text[4:6]}-{day_text[6:]}")
    if day.year != year or away == home or away not in _TEAM_NAMES or home not in _TEAM_NAMES:
        raise ValueError(f"historical game year/team mismatch: {key}")
    board = contents.get("scoreboard")
    if not isinstance(board, list) or len(board) != 2:
        raise ValueError(f"historical final scoreboard requires two sides: {key}")
    for side, code in zip(board, (away, home), strict=True):
        if not isinstance(side, dict):
            raise ValueError(f"historical scoreboard team/order mismatch: {key}")
        name = str(side.get("팀", "")).strip()
        # Some 2009 Heroes display names are blank; the original game ID still
        # identifies both clubs and scoreboard rows retain away/home order.
        if name and name not in _TEAM_NAMES[code]:
            raise ValueError(f"historical scoreboard team/order mismatch: {key}")
    scores = [_integer(side.get("R"), f"{key} runs") for side in board]
    away_score, home_score = scores
    assert away_score is not None and home_score is not None
    expected = (
        ("무", "무")
        if away_score == home_score
        else (("승", "패") if away_score > home_score else ("패", "승"))
    )
    for side, result in zip(board, expected, strict=True):
        if side.get("승패") not in (result, "무승부" if result == "무" else result):
            raise ValueError(f"historical final score/result mismatch: {key}")
    return HistoricalGame(
        key,
        day,
        away,
        home,
        int(doubleheader),
        away_score,
        home_score,
        _integer(board[0].get("H"), "hits", optional=True),
        _integer(board[1].get("H"), "hits", optional=True),
        _integer(board[0].get("E"), "errors", optional=True),
        _integer(board[1].get("E"), "errors", optional=True),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key in history archive: {key}")
        result[key] = value
    return result


def _read_history_records(
    path: Path,
    artifact: KBOHistoryArtifact,
) -> list[tuple[str, dict[str, Any]]]:
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"historical source SHA-256 mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    if artifact.format == "game_map" and isinstance(payload, dict):
        records = list(payload.items())
    elif artifact.format == "game_list" and isinstance(payload, list):
        if any(
            not isinstance(row, dict) or "id" not in row or "contents" not in row for row in payload
        ):
            raise ValueError(f"malformed historical game list: {path.name}")
        records = [(row["id"], row["contents"]) for row in payload]
    else:
        raise ValueError(f"unexpected historical source structure: {path.name}")
    if len(records) != artifact.game_count:
        raise ValueError(f"historical source record count mismatch: {path.name}")
    if any(not isinstance(key, str) or not isinstance(value, dict) for key, value in records):
        raise ValueError(f"malformed historical game contents: {path.name}")
    return records


def read_history_artifact(
    path: Path,
    artifact: KBOHistoryArtifact,
) -> tuple[list[HistoricalGame], list[str]]:
    records = _read_history_records(path, artifact)
    games, excluded = [], []
    for key, contents in records:
        if key in _NON_REGULAR:
            excluded.append(key)
        games.append(parse_historical_game(key, contents, artifact.year))
    return games, excluded


def _timestamps(day: date, ingested_at: datetime) -> dict[str, Any]:
    event = datetime.combine(day, time(23, 59, 59), _KST)
    return {
        "event_at": event,
        "available_at": datetime.combine(day + timedelta(days=1), time(), _KST),
        "ingested_at": ingested_at,
        "valid_from": event,
        "valid_to": None,
    }


def _table_counts(store: DuckDBStore) -> dict[str, int]:
    counts = {}
    for table in _TABLES:
        row = store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert row is not None
        counts[table] = int(row[0])
    return counts


def _import_boxscores(
    store: DuckDBStore,
    root: Path,
    selected: tuple[KBOHistoryArtifact, ...],
    games: dict[str, HistoricalGame],
    now: datetime,
) -> dict[int, dict[str, Any]]:
    """Append one archive at a time, preserving raw rows and field-level masks.

    A separate source revision upgrades a v1 import without rewriting its game
    rows, source provenance or identities. Equal cross-file duplicates count once;
    differing box-score payloads fail the entire import instead of choosing one.
    """
    seen: dict[str, str] = {}
    coverage: dict[int, dict[str, Any]] = {}
    for artifact in selected:
        source_id = f"kbo-history-box:v1:{artifact.filename}:{artifact.sha256}"
        box_rows: list[dict[str, Any]] = []
        details = []
        season = coverage.setdefault(artifact.year, {
            "batter_rows": 0, "pitcher_rows": 0, "games_with_boxscores": 0,
            "hit_labels": 0, "known_pa_hit_labels": 0, "partial_pa_hit_labels": 0,
            "verified_batting_outcome_rows": 0, "verified_batting_outcomes": 0,
            "quality_reasons": Counter(), "source_fields": Counter(),
        })
        first: date | None = None
        for key, contents in _read_history_records(root / artifact.filename, artifact):
            payload_hash = hashlib.sha256(json.dumps(
                contents, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            if key in seen:
                if seen[key] != payload_hash:
                    raise ValueError(f"conflicting historical box-score payloads: {key}")
                continue
            seen[key] = payload_hash
            game = games[key]
            first = min(first, game.day) if first is not None else game.day
            parsed = parse_historical_boxscore(key, contents)
            game_id = f"kbo-game:{game.game_pk}"
            times = _timestamps(game.day, now)
            details.append({
                "detail_row_id": f"{source_id}:detail:{game.game_pk}",
                "game_id": game_id,
                "metadata_json": {
                    field: value for field, value in parsed.raw.items()
                    if field not in {"away_batter", "home_batter", "away_pitcher", "home_pitcher"}
                },
                "quality_json": list(parsed.quality_reasons),
                "source_revision_id": source_id,
                **times,
            })
            season["games_with_boxscores"] += bool(parsed.batters or parsed.pitchers)
            season["quality_reasons"].update(parsed.quality_reasons)
            for role, observations in (("batting", parsed.batters), ("pitching", parsed.pitchers)):
                season["batter_rows" if role == "batting" else "pitcher_rows"] += len(observations)
                for observation in observations:
                    stats = asdict(observation)
                    raw = stats.pop("raw")
                    reasons = stats.pop("quality_reasons")
                    season["quality_reasons"].update(reasons)
                    season["source_fields"].update(f"{role}:{field}" for field in raw)
                    team = observation.team_code
                    opponent = game.home if observation.side == "away" else game.away
                    box_rows.append({
                        "boxscore_row_id": f"{source_id}:row:{observation.observation_id}",
                        "observation_id": f"kbo-box:{observation.observation_id}",
                        "game_id": game_id,
                        "team_game_id": f"kbo-team-game:{game.game_pk}:{team}",
                        "team_id": f"kbo-team:{team}",
                        "opponent_team_id": f"kbo-team:{opponent}",
                        # This is a source observation, NOT a resolved career player ID.
                        "player_id": f"kbo-box-observation:{observation.observation_id}",
                        "identity_status": "source_observation",
                        "role": role, "side": observation.side,
                        "display_name": observation.display_name,
                        "row_index": observation.row_index,
                        "stats_json": stats, "raw_json": raw, "quality_json": reasons,
                        "source_revision_id": source_id,
                        **times,
                    })
            for batter in parsed.batters:
                positive_appearance = (
                    batter.plate_appearances is not None and batter.plate_appearances >= 1
                ) or (batter.at_bats is not None and batter.at_bats >= 1)
                if batter.hits_verified and positive_appearance:
                    season["hit_labels"] += 1
                    season["known_pa_hit_labels" if batter.plate_appearances is not None
                           else "partial_pa_hit_labels"] += 1
                outcome_total: int = sum(batter.outcome_counts)
                if batter.counts_verified and outcome_total:
                    season["verified_batting_outcome_rows"] += 1
                    season["verified_batting_outcomes"] += outcome_total
        if first is None:
            continue
        store.append("source_revision", [{
            "source_revision_id": source_id,
            "source_name": "kbo_historical_boxscore",
            "source_locator": artifact.url, "content_sha256": artifact.sha256,
            "metadata_json": {
                "adapter_version": 2, "boxscore_parser_version": 1,
                "repository_url": artifact.repository_url, "revision": artifact.revision,
                "filename": artifact.filename, "repository_license": artifact.repository_license,
                "verification_provenance": artifact.provenance,
                "identity_policy": "source observations; names are not canonical player IDs",
                "timestamp_policy": "retrospective: labels available next day 00:00 KST",
                "label_tier": "partial_player_boxscore",
            }, **_timestamps(first, now),
        }], ignore_existing=True)
        store.append("historical_game_detail", details, ignore_existing=True, batch_size=256)
        store.append("historical_boxscore", box_rows, ignore_existing=True, batch_size=256)
    for season in coverage.values():
        for field in ("quality_reasons", "source_fields"):
            season[field] = dict(sorted(season[field].items()))
    return coverage


def import_kbo_history(
    store: DuckDBStore,
    directory: str | Path,
    *,
    years: Iterable[int] | None = None,
    artifacts: tuple[KBOHistoryArtifact, ...] = KBO_HISTORY_FILES,
    ingested_at: datetime | None = None,
) -> dict[str, Any]:
    """Atomically append archived games and every available player box-score row.

    Actual source files remain unchanged. Missing identities/event contexts are
    not inferred. Partial observations retain usable fields and quality reasons.
    """
    if store.read_only:
        raise PermissionError("historical import requires a writable database")
    selected = select_history_artifacts(artifacts, years)
    selected_years = {artifact.year for artifact in selected}
    pinned_selection = all(artifact in KBO_HISTORY_FILES for artifact in selected) and (
        {artifact.filename for artifact in selected}
        == {artifact.filename for artifact in KBO_HISTORY_FILES if artifact.year in selected_years}
    )
    now = ingested_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("ingested_at must be timezone-aware")
    root = Path(directory).expanduser()
    rows: dict[str, list[dict[str, Any]]] = {table: [] for table in _TABLES}
    unique: dict[str, HistoricalGame] = {}
    report_files = []
    duplicates: Counter[int] = Counter()
    for artifact in selected:
        path = root / artifact.filename
        games, excluded = read_history_artifact(path, artifact)
        if not games:
            raise ValueError(f"no regular final games in {artifact.filename}")
        first = min(game.day for game in games)
        source_id = f"kbo-history:v1:{artifact.filename}:{artifact.sha256}"
        metadata = {
            "adapter_version": 1,
            "repository_url": artifact.repository_url,
            "revision": artifact.revision,
            "repository_license": artifact.repository_license,
            "filename": artifact.filename,
            "raw_records": artifact.game_count,
            "verification_provenance": artifact.provenance,
            "label_tier": "final_game_score_only",
            "timestamp_policy": "retrospective: final labels available next day 00:00 KST",
            "excluded_non_regular_game_ids": excluded,
        }
        rows["source_revision"].append(
            {
                "source_revision_id": source_id,
                "source_name": "kbo_historical_archive",
                "source_locator": artifact.url,
                "content_sha256": artifact.sha256,
                "metadata_json": metadata,
                **_timestamps(first, now),
            }
        )
        teams: dict[str, date] = {}
        new_games = 0
        for game in sorted(games, key=lambda item: item.key):
            previous = unique.get(game.key)
            if previous is not None:
                if previous != game:
                    raise ValueError(f"conflicting historical final records: {game.key}")
                duplicates[game.day.year] += 1
                continue
            unique[game.key] = game
            new_games += 1
            # Legacy score-only source revisions excluded this non-regular game.
            # Its new record belongs to the v2 box-score source, preserving v1 provenance.
            game_source_id = (
                f"kbo-history-box:v1:{artifact.filename}:{artifact.sha256}"
                if game.key in _NON_REGULAR else source_id
            )
            times = _timestamps(game.day, now)
            for team in (game.home, game.away):
                teams[team] = min(teams.get(team, game.day), game.day)
            game_id = f"kbo-game:{game.game_pk}"
            rows["game"].append(
                {
                    "game_row_id": f"{game_source_id}:game:{game.game_pk}",
                    "game_id": game_id,
                    "season": game.day.year,
                    "game_type": "tiebreaker" if game.key in _NON_REGULAR else "regular",
                    "scheduled_start": datetime.combine(game.day, time(), _KST),
                    "home_team_id": f"kbo-team:{game.home}",
                    "away_team_id": f"kbo-team:{game.away}",
                    "doubleheader_number": game.doubleheader or None,
                    "game_status": "final",
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "source_revision_id": game_source_id,
                    **times,
                }
            )
            for is_home in (False, True):
                team, opponent = (game.home, game.away) if is_home else (game.away, game.home)
                score, other_score = (
                    (game.home_score, game.away_score)
                    if is_home
                    else (
                        game.away_score,
                        game.home_score,
                    )
                )
                rows["team_game"].append(
                    {
                        "team_game_row_id": f"{game_source_id}:team-game:{game.game_pk}:{team}",
                        "team_game_id": f"kbo-team-game:{game.game_pk}:{team}",
                        "game_id": game_id,
                        "team_id": f"kbo-team:{team}",
                        "opponent_team_id": f"kbo-team:{opponent}",
                        "is_home": is_home,
                        "runs": score,
                        "hits": game.home_hits if is_home else game.away_hits,
                        "errors": game.home_errors if is_home else game.away_errors,
                        "result": "draw"
                        if score == other_score
                        else ("win" if score > other_score else "loss"),
                        "source_revision_id": game_source_id,
                        **times,
                    }
                )
        for team, day in sorted(teams.items()):
            rows["team"].append(
                {
                    "team_row_id": f"{source_id}:team:{team}",
                    "team_id": f"kbo-team:{team}",
                    "team_name": team,
                    "short_name": team,
                    "active_from": day,
                    "source_revision_id": source_id,
                    **_timestamps(day, now),
                }
            )
        report_files.append(
            {
                **asdict(artifact),
                "unique_games_added": new_games,
                "retained_non_regular_game_ids": excluded,
            }
        )
    coverage: list[dict[str, Any]] = []
    for year in sorted({artifact.year for artifact in selected}):
        season = [game for game in unique.values() if game.day.year == year]
        if not season:
            raise ValueError(f"no historical games for requested year {year}")
        expected = EXPECTED_HISTORY_GAMES[year] if pinned_selection else None
        regular_count = sum(game.key not in _NON_REGULAR for game in season)
        if expected is not None and regular_count != expected:
            raise ValueError(f"incomplete historical season {year}: {regular_count} != {expected}")
        coverage.append(
            {
                "year": year,
                "games": len(season),
                "regular_games": regular_count,
                "non_regular_games": len(season) - regular_count,
                "expected_regular_games": expected,
                "regular_season_complete": regular_count == expected if expected else None,
                "duplicate_records": duplicates[year],
                "date_start": min(game.day for game in season).isoformat(),
                "date_end": max(game.day for game in season).isoformat(),
                "pa_queries": 0,
                "live_hit_queries": 0,
            }
        )
    with store.transaction():
        # A second provider must not silently change an existing logical game's label.
        existing = {
            row[0]: row[1:]
            for row in store.connection.execute(
                "SELECT game_id, home_team_id, away_team_id, home_score, away_score FROM game"
            ).fetchall()
        }
        for row in rows["game"]:
            previous_facts = existing.get(row["game_id"])
            facts = (row["home_team_id"], row["away_team_id"], row["home_score"], row["away_score"])
            if previous_facts is not None and previous_facts != facts:
                raise ValueError(f"existing canonical score conflict: {row['game_id']}")
        before = _table_counts(store)
        for table in _CORE_TABLES:
            store.append(table, rows[table], ignore_existing=True, batch_size=256)
        box_coverage = _import_boxscores(store, root, selected, unique, now)
        store.assert_referential_integrity()
        store.assert_composite_referential_integrity()
        totals = _table_counts(store)
    for season_report in coverage:
        season_report.update(box_coverage.get(season_report["year"], {}))
        season_report["live_hit_queries"] = season_report.get("hit_labels", 0)
    return {
        "adapter_version": 2,
        "files": report_files,
        "season_coverage": coverage,
        "inserted_rows": {table: totals[table] - before[table] for table in _TABLES},
        "total_rows": totals,
        "games": len(unique),
        "label_tier": "partial_player_boxscore",
        "field_usage": {
            "final_scores": "match and run targets; subsequent-day team history",
            "batting_totals": "observed hit/PA labels; masked subsequent-day batting history",
            "inning_results": "verified aggregate outcome targets; raw tokens retained",
            "pitching_totals": "masked count targets; subsequent-day pitching history",
            "display_names": "display/audit only; no name-based player merge",
            "other_fields": "lossless raw rows and game metadata; not guessed as pregame inputs",
        },
        "notes": [
            "Every player row is retained, including invalid/partial rows with quality reasons.",
            "Unresolved player observations use past team-role history, not fabricated career IDs.",
            "Known hits with unknown PA train a marginal likelihood, not an invented PA count.",
            "Verified inning outcomes train aggregates, not invented full PA state/matchups.",
            "Same-day scores are labels, never prior-game features.",
            "Publication/start timestamps are reconstructed, not real-time observations.",
        ],
    }
