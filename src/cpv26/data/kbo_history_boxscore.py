"""Parse factual historical box scores, without inventing event context or identities.

The ten-way counts summarize each batter's recorded inning cells. They are not
an ordered play-by-play log: no pitcher matchup, base state or out state is
reconstructed. Invalid rows and unknown tokens remain available with reasons.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES

# The archive author's published data dictionary, not an inferred code scheme.
NUMERIC_CODE_SOURCE = (
    "https://raw.githubusercontent.com/LOPES-HUFS/KBO_data/"
    "94e72c797e07b3b72167c92258728bef599ed5fc/kbo_data/code_list.ini"
)
_GAME_KEY = re.compile(r"^(\d{8})_([A-Z]{2})([A-Z]{2})([012])$")
_FIELDER = r"[123456789유투포좌중우]+"
_NUMERIC_OUTCOMES = {
    **dict.fromkeys(range(1000, 1031), "single"),
    **dict.fromkeys(range(1100, 1124), "double"),
    **dict.fromkeys(range(1200, 1206), "triple"),
    **dict.fromkeys(range(1300, 1305), "home_run"),
    **dict.fromkeys((2000, 2100), "strikeout"),
    **dict.fromkeys((3000, 3100, 3200), "walk_or_hbp"),
    **dict.fromkeys(range(4000, 4006), "ball_in_play_out"),
    **dict.fromkeys(range(4100, 4108), "sacrifice_hit"),
    **dict.fromkeys(range(5000, 5008), "sacrifice_fly"),
    **dict.fromkeys(range(6000, 6009), "reached_on_error"),
    **dict.fromkeys(range(6100, 6108), "sacrifice_hit"),
    **dict.fromkeys(range(6200, 6205), "sacrifice_hit"),
    6300: "ball_in_play_out",
    6400: "catcher_interference",
    **dict.fromkeys(range(7000, 7009), "ball_in_play_out"),
    **dict.fromkeys(range(7100, 7109), "ball_in_play_out"),
    **dict.fromkeys(range(7200, 7225), "ball_in_play_out"),
    7226: "ball_in_play_out",
    **dict.fromkeys(range(7300, 7308), "ball_in_play_out"),
    **dict.fromkeys(range(7400, 7408), "ball_in_play_out"),
}


@dataclass(frozen=True)
class InningToken:
    inning: int
    cell_ordinal: int
    raw_token: str
    mapped_outcome: str | None


@dataclass(frozen=True)
class ParsedBatter:
    observation_id: str
    game_key: str
    season: int
    team_code: str
    side: str
    row_index: int
    display_name: str | None
    position: str | None
    at_bats: int | None
    hits: int | None
    runs: int | None
    rbi: int | None
    batting_average: float | None
    plate_appearances: int | None
    outcome_counts: tuple[int, ...]
    catcher_interference: int
    innings_tokens: tuple[InningToken, ...]
    counts_verified: bool
    hits_verified: bool
    quality_reasons: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ParsedPitcher:
    observation_id: str
    game_key: str
    season: int
    team_code: str
    side: str
    row_index: int
    display_name: str | None
    entry: str | None
    batters_faced: int | None
    outs: int | None
    pitches: int | None
    at_bats: int | None
    hits: int | None
    home_runs: int | None
    walks_hbp: int | None
    strikeouts: int | None
    runs: int | None
    earned_runs: int | None
    era: float | None
    wins: int | None
    losses: int | None
    saves: int | None
    holds: int | None
    quality_reasons: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ParsedBoxscore:
    batters: tuple[ParsedBatter, ...]
    pitchers: tuple[ParsedPitcher, ...]
    game_metadata: dict[str, Any]
    quality_reasons: tuple[str, ...]
    raw: dict[str, Any]


def _integer(value: Any, field: str, reasons: list[str], *, optional: bool = False) -> int | None:
    if value is None or value == "" or value == "-":
        if not optional:
            reasons.append(f"missing:{field}")
        return None
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        reasons.append(f"invalid_nonnegative_integer:{field}")
        return None
    return int(value)


def _rate(value: Any, field: str, reasons: list[str]) -> float | None:
    if value is None or value in ("", "-"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        reasons.append(f"invalid_nonnegative_rate:{field}")
        return None
    return number


def _text(value: Any) -> str | None:
    return None if value is None else str(value).strip() or None


def _raw_row(row: Any, reasons: list[str]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return copy.deepcopy(dict(row))
    reasons.append("row_not_object")
    return {"unparsed_value": copy.deepcopy(row)}


def map_inning_outcome(token: str) -> str | None:
    """Map only recognized source notation; unknown codes are not decomposed.

    Some 2009 cells concatenate two four-digit codes without a delimiter. They
    remain unknown, rather than assuming where the source lost a boundary.
    """
    token = token.strip().translate(str.maketrans("一二三", "123"))
    if re.fullmatch(r"\d{4,}", token):
        return _NUMERIC_OUTCOMES.get(int(token))
    exact = {
        "삼진": "strikeout", "스낫": "strikeout", "4구": "walk_or_hbp",
        "사구": "walk_or_hbp", "고4": "walk_or_hbp", "야선": "ball_in_play_out",
        "삼선": "sacrifice_hit", "삼비": "ball_in_play_out", "삼파": "ball_in_play_out",
        "타방": "catcher_interference",
    }
    if token in exact:
        return exact[token]
    suffixes = (
        ("희비", "sacrifice_fly"), (r"희(?:번|선|실)", "sacrifice_hit"),
        ("홈", "home_run"), ("안", "single"), ("2", "double"), ("3", "triple"),
        ("실", "reached_on_error"), (r"(?:땅|비|직|파|병|번|삼중)", "ball_in_play_out"),
    )
    for suffix, outcome in suffixes:
        if re.fullmatch(_FIELDER + suffix, token):
            return outcome
    return None


def _inning_tokens(raw: Mapping[str, Any]) -> tuple[InningToken, ...]:
    result = []
    fields = (key for key in raw if isinstance(key, str) and key.isdigit())
    for field in sorted(fields, key=int):
        value = raw[field]
        if not isinstance(value, bool) and value in (0, "0", "", "-", None):
            continue
        # Both '/' and the archive's literal '\/' separate multiple results.
        for ordinal, token in enumerate(re.split(r"\\?/", str(value))):
            token = token.strip()
            result.append(InningToken(int(field), ordinal, token, map_inning_outcome(token)))
    return tuple(result)


def parse_batter_boxscore(
    game_key: str, team_code: str, row_index: int, row: Any, *, side: str
) -> ParsedBatter:
    """Keep a source row even when only its AB/H totals can be used."""
    reasons: list[str] = []
    raw = _raw_row(row, reasons)
    name = _text(raw.get("선수명"))
    if not name or name == "데이터가 없습니다.":
        reasons.append("player_display_name_missing")
        name = None
    ab = _integer(raw.get("타수"), "at_bats", reasons)
    hits = _integer(raw.get("안타"), "hits", reasons)
    runs = _integer(raw.get("득점"), "runs", reasons)
    rbi = _integer(raw.get("타점"), "rbi", reasons)
    average = _rate(raw.get("타율"), "batting_average", reasons)
    if average is not None and average > 1:
        reasons.append("batting_average_exceeds_one")
        average = None
    hits_verified = ab is not None and hits is not None and hits <= ab
    if ab is not None and hits is not None and hits > ab:
        reasons.append("hits_exceed_at_bats")
    tokens = _inning_tokens(raw)
    known = all(token.mapped_outcome is not None for token in tokens)
    for token in tokens:
        if token.mapped_outcome is None:
            reasons.append(f"unknown_inning_token:{token.inning}:{token.raw_token}")
    counts = tuple(sum(t.mapped_outcome == label for t in tokens) for label in NEURAL_PA_OUTCOMES)
    interference = sum(t.mapped_outcome == "catcher_interference" for t in tokens)
    token_ab = sum(counts[i] for i in (0, 2, 3, 4, 5, 6, 7))
    token_hits = sum(counts[i] for i in (2, 3, 4, 5))
    has_innings = any(isinstance(key, str) and key.isdigit() for key in raw)
    if not has_innings:
        reasons.append("inning_results_missing")
    if ab is not None and token_ab != ab:
        reasons.append(f"inning_at_bats_mismatch:{token_ab}!={ab}")
    if hits is not None and token_hits != hits:
        reasons.append(f"inning_hits_mismatch:{token_hits}!={hits}")
    verified = known and has_innings and hits_verified and token_ab == ab and token_hits == hits
    return ParsedBatter(
        observation_id=f"{game_key}:batting:{side}:{row_index}", game_key=game_key,
        season=int(game_key[:4]), team_code=team_code, side=side, row_index=row_index,
        display_name=name, position=_text(raw.get("포지션")), at_bats=ab, hits=hits,
        runs=runs, rbi=rbi, batting_average=average,
        plate_appearances=len(tokens) if verified else None,
        outcome_counts=counts, catcher_interference=interference, innings_tokens=tokens,
        counts_verified=verified, hits_verified=hits_verified,
        quality_reasons=tuple(reasons), raw=raw,
    )


def parse_pitcher_outs(raw: Mapping[str, Any]) -> int | None:
    """Read whole innings plus thirds; decimal ERA/entry values are not innings."""
    if "이닝" not in raw and "inning" in raw:
        whole = _integer(raw.get("inning"), "inning", [])
        remainder = _integer(raw.get("restinning"), "restinning", [])
        if whole is None or remainder not in (0, 1, 2):
            return None
        return 3 * whole + remainder
    value = raw.get("이닝")
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace("\\/", "/")
    if re.fullmatch(r"\d+", text):
        return 3 * int(text)
    match = re.fullmatch(r"(?:(\d+)\s+)?([12])/3", text)
    if match:
        return 3 * int(match.group(1) or 0) + int(match.group(2))
    return None


def parse_pitcher_boxscore(
    game_key: str, team_code: str, row_index: int, row: Any, *, side: str
) -> ParsedPitcher:
    reasons: list[str] = []
    raw = _raw_row(row, reasons)
    name = _text(raw.get("선수명"))
    if not name or name == "데이터가 없습니다.":
        reasons.append("player_display_name_missing")
        name = None
    fields = {
        "batters_faced": "타자", "pitches": "투구수", "at_bats": "타수", "hits": "피안타",
        "home_runs": "홈런", "walks_hbp": "4사구", "strikeouts": "삼진", "runs": "실점",
        "earned_runs": "자책",
    }
    values = {name: _integer(raw.get(key), name, reasons) for name, key in fields.items()}
    for smaller, larger in (
        ("hits", "at_bats"), ("home_runs", "hits"), ("earned_runs", "runs"),
        ("at_bats", "batters_faced"), ("strikeouts", "batters_faced"),
    ):
        small, large = values[smaller], values[larger]
        if small is not None and large is not None and small > large:
            reasons.append(f"{smaller}_exceed_{larger}")
    outs = parse_pitcher_outs(raw)
    if outs is None:
        reasons.append("invalid_or_missing_innings_pitched")
    era = _rate(raw.get("평균자책점"), "era", reasons)
    optional = {
        "wins": ("승", "승리"), "losses": ("패", "패배"),
        "saves": ("세", "세이브"), "holds": ("홀드", "홀드"),
    }
    optional_values = {
        name: _integer(raw.get(keys[0], raw.get(keys[1])), name, reasons, optional=True)
        for name, keys in optional.items()
    }
    return ParsedPitcher(
        observation_id=f"{game_key}:pitching:{side}:{row_index}", game_key=game_key,
        season=int(game_key[:4]), team_code=team_code, side=side, row_index=row_index,
        display_name=name, entry=_text(raw.get("등판")), outs=outs, era=era,
        quality_reasons=tuple(reasons), raw=raw, **values, **optional_values,
    )


def _invalidate_counts(batters: list[ParsedBatter], reason: str) -> list[ParsedBatter]:
    return [replace(
        row, plate_appearances=None, counts_verified=False,
        quality_reasons=(*row.quality_reasons, reason),
    ) for row in batters]


def parse_historical_boxscore(game_key: str, contents: Mapping[str, Any]) -> ParsedBoxscore:
    """Parse observations and audit team PA totals against opposing pitchers' BF.

    Observation IDs distinguish source rows only. They must not be treated as
    persistent player identities, and display names must not be merged blindly.
    """
    match = _GAME_KEY.fullmatch(game_key)
    if match is None:
        raise ValueError(f"invalid historical game key: {game_key}")
    if not isinstance(contents, Mapping):
        raise TypeError("historical boxscore contents must be a mapping")
    batters: dict[str, list[ParsedBatter]] = {}
    pitchers: dict[str, list[ParsedPitcher]] = {}
    reasons: list[str] = []
    for side, team in (("away", match.group(2)), ("home", match.group(3))):
        for kind in ("batter", "pitcher"):
            field = f"{side}_{kind}"
            if not isinstance(contents.get(field, []), list):
                reasons.append(f"collection_not_list:{field}")
        braw = contents.get(f"{side}_batter", [])
        praw = contents.get(f"{side}_pitcher", [])
        if not isinstance(braw, list):
            braw = [braw]
        if not isinstance(praw, list):
            praw = [praw]
        batters[side] = [
            parse_batter_boxscore(game_key, team, i, row, side=side) for i, row in enumerate(braw)
        ]
        pitchers[side] = [
            parse_pitcher_boxscore(game_key, team, i, row, side=side) for i, row in enumerate(praw)
        ]
        if not pitchers[side]:
            reasons.append(f"missing_pitching_boxscore:{side}")
    for side, opposite, index in (("away", "home", 0), ("home", "away", 1)):
        team_batters, opposing_pitchers = batters[side], pitchers[opposite]
        if not team_batters:
            reasons.append(f"missing_batting_boxscore:{side}")
            continue
        board = contents.get("scoreboard", [])
        team_hits = None
        if isinstance(board, list) and len(board) > index and isinstance(board[index], Mapping):
            team_hits = _integer(board[index].get("H"), "team_hits", [], optional=True)
        if team_hits is not None and all(row.hits_verified for row in team_batters):
            total_hits = sum(row.hits or 0 for row in team_batters)
            if team_hits != total_hits:
                reason = f"team_hits_mismatch:{total_hits}!={team_hits}"
                reasons.append(f"{side}:{reason}")
                team_batters = [replace(row, hits_verified=False) for row in team_batters]
                team_batters = _invalidate_counts(team_batters, reason)
        complete = (
            all(row.counts_verified for row in team_batters) and bool(opposing_pitchers)
            and all(row.batters_faced is not None for row in opposing_pitchers)
        )
        if complete:
            pa = sum(row.plate_appearances or 0 for row in team_batters)
            bf = sum(row.batters_faced or 0 for row in opposing_pitchers)
            if pa != bf:
                reason = f"team_pa_batters_faced_mismatch:{pa}!={bf}"
                reasons.append(f"{side}:{reason}")
                team_batters = _invalidate_counts(team_batters, reason)
        else:
            reasons.append(f"team_pa_totals_unverified:{side}")
            team_batters = _invalidate_counts(team_batters, "team_pa_totals_unverified")
        batters[side] = team_batters
    metadata = contents.get("ETC_info", {})
    if not isinstance(metadata, Mapping):
        reasons.append("game_metadata_not_object")
        metadata = {"unparsed_value": metadata}
    return ParsedBoxscore(
        tuple((*batters["away"], *batters["home"])),
        tuple((*pitchers["away"], *pitchers["home"])),
        copy.deepcopy(dict(metadata)), tuple(reasons), copy.deepcopy(dict(contents)),
    )
