"""Small source-pattern tests; these records are not training fixtures."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from cpv26.data.kbo_history_boxscore import (
    NUMERIC_CODE_SOURCE,
    map_inning_outcome,
    parse_batter_boxscore,
    parse_historical_boxscore,
    parse_pitcher_boxscore,
    parse_pitcher_outs,
)
from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES


def _batter(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "선수명": "이병규", "포지션": "중", "1": "투땅", "2": "2땅", "3": 0,
        "4": 0, "5": "중안", "6": "중안", "7": 0, "8": "우중안", "9": 0,
        "타수": 5, "안타": 3, "득점": 1, "타점": 0, "타율": 0.6,
    }
    result.update(changes)
    return result


def _pitcher(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "선수명": "해리거", "등판": "선발", "결과": "패", "승": "0", "패": "1",
        "세": "0", "이닝": "4", "타자": 22, "투구수": 89, "타수": 20,
        "피안타": 9, "홈런": 0, "4사구": 2, "삼진": 3, "실점": 5,
        "자책": 5, "평균자책점": 11.25,
    }
    result.update(changes)
    return result


def _outcomes(counts: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(NEURAL_PA_OUTCOMES, counts, strict=True))


def _one_batter_game() -> dict[str, Any]:
    # A reduced box score that can exercise both sides' total reconciliation.
    return {
        "away_batter": [_batter()], "home_batter": [_batter()],
        "away_pitcher": [_pitcher(타자=5, 타수=5, 피안타=3)],
        "home_pitcher": [_pitcher(타자=5, 타수=5, 피안타=3)],
        "scoreboard": [{"팀": "LG", "H": 3}, {"팀": "SK", "H": 3}],
        "ETC_info": {"심판": "심판 원문", "관중": "12,345", "unknown": "kept"},
        "additional_source_field": {"value": 0},
    }


def test_2001_actual_batter_pattern_and_raw_retention() -> None:
    # 20010405_LGSK0 away row 0: the five inning cells and factual totals.
    raw = _batter()
    parsed = parse_batter_boxscore("20010405_LGSK0", "LG", 0, raw, side="away")
    assert parsed.display_name == "이병규"
    assert parsed.position == "중"
    assert (parsed.at_bats, parsed.hits, parsed.runs, parsed.rbi) == (5, 3, 1, 0)
    assert parsed.batting_average == 0.6
    assert parsed.plate_appearances == 5
    assert parsed.hits_verified and parsed.counts_verified
    assert _outcomes(parsed.outcome_counts)["single"] == 3
    assert _outcomes(parsed.outcome_counts)["ball_in_play_out"] == 2
    assert [t.inning for t in parsed.innings_tokens] == [1, 2, 5, 6, 8]
    assert parsed.raw == raw and parsed.raw is not raw
    assert parsed.quality_reasons == ()


def test_2009_actual_numeric_pattern_uses_published_dictionary() -> None:
    raw = {
        "선수명": "강동우", "포지션": "중", "1": 2000, "2": 0, "3": 1019,
        "4": 0, "5": 7104, "6": 0, "7": 4104, "8": 3000, "9": 0,
        "타수": 3, "안타": 1, "득점": 0, "타점": 0, "타율": 0.333,
        "경기날짜": "20090404", "원정팀": "한화", "홈팀": "SK", "더블헤더여부": 0,
    }
    parsed = parse_batter_boxscore("20090404_HHSK0", "HH", 0, raw, side="away")
    assert NUMERIC_CODE_SOURCE.endswith("/kbo_data/code_list.ini")
    assert parsed.plate_appearances == 5 and parsed.counts_verified
    counts = _outcomes(parsed.outcome_counts)
    assert counts["single"] == counts["strikeout"] == counts["walk_or_hbp"] == 1
    assert counts["sacrifice_hit"] == counts["ball_in_play_out"] == 1
    assert parsed.raw["경기날짜"] == "20090404"


@pytest.mark.parametrize(("token", "outcome"), [
    ("좌중2", "double"), ("二땅", "ball_in_play_out"), ("우3", "triple"),
    ("우중홈", "home_run"), ("1희번", "sacrifice_hit"), ("포희실", "sacrifice_hit"),
    ("투희선", "sacrifice_hit"), ("삼선", "sacrifice_hit"),
    ("좌희비", "sacrifice_fly"), ("유실", "reached_on_error"),
    ("야선", "ball_in_play_out"), ("투번", "ball_in_play_out"),
    ("3삼중", "ball_in_play_out"), ("스낫", "strikeout"),
    ("고4", "walk_or_hbp"), ("사구", "walk_or_hbp"), ("타방", "catcher_interference"),
    ("7226", "ball_in_play_out"), ("6400", "catcher_interference"),
    ("7225", None), ("1099", None), ("10047003", None), ("30003000", None),
    ("三", None), ("땅", None), ("unknown", None), ("", None),
])
def test_only_recognized_notation_maps(token: str, outcome: str | None) -> None:
    assert map_inning_outcome(token) == outcome


def test_multiple_results_in_one_inning_and_catcher_interference() -> None:
    raw = {
        "선수명": "기록", "1": "4구\\/ 우2 / 타방", "타수": 1, "안타": 1,
        "득점": 0, "타점": 0,
    }
    parsed = parse_batter_boxscore("20220402_HHOB0", "HH", 2, raw, side="away")
    assert parsed.plate_appearances == 3 and parsed.counts_verified
    assert parsed.catcher_interference == 1
    assert sum(parsed.outcome_counts) == 2  # CI is not fabricated into one of ten outcomes.
    assert [t.cell_ordinal for t in parsed.innings_tokens] == [0, 1, 2]


def test_unknown_inning_token_keeps_hits_but_not_guessed_pa() -> None:
    raw = _batter(**{"1": "10047003"})
    parsed = parse_batter_boxscore("20010405_LGSK0", "LG", 0, raw, side="away")
    assert parsed.plate_appearances is None and not parsed.counts_verified
    assert parsed.hits_verified and parsed.hits == 3
    assert "unknown_inning_token:1:10047003" in parsed.quality_reasons
    assert parsed.innings_tokens[0].raw_token == "10047003"
    assert parsed.raw["1"] == "10047003"


@pytest.mark.parametrize(("ab", "hits", "hits_ok"), [(4, 3, True), (5, 4, True), (2, 3, False)])
def test_inning_totals_must_match_without_discarding_factual_totals(
    ab: int, hits: int, hits_ok: bool,
) -> None:
    parsed = parse_batter_boxscore(
        "20010405_LGSK0", "LG", 0, _batter(타수=ab, 안타=hits), side="away",
    )
    assert not parsed.counts_verified and parsed.plate_appearances is None
    assert (parsed.at_bats, parsed.hits) == (ab, hits)
    assert parsed.hits_verified is hits_ok
    assert any("mismatch" in reason for reason in parsed.quality_reasons)


@pytest.mark.parametrize(("token", "pa"), [(0, 0), ("0", 0), ("4구", 1)])
def test_zero_ab_rows_not_dropped_and_walk_still_counts(token: Any, pa: int) -> None:
    raw = {"선수명": "선수", "1": token, "타수": 0, "안타": 0, "득점": 0, "타점": 0}
    parsed = parse_batter_boxscore("20220402_HHOB0", "HH", 0, raw, side="away")
    assert parsed.plate_appearances == pa
    assert parsed.counts_verified and parsed.hits_verified


def test_missing_inning_fields_are_not_assumed_zero_pa() -> None:
    raw = {"선수명": "선수", "타수": 0, "안타": 0, "득점": 0, "타점": 0}
    parsed = parse_batter_boxscore("20220402_HHOB0", "HH", 0, raw, side="away")
    assert parsed.hits_verified
    assert parsed.plate_appearances is None
    assert "inning_results_missing" in parsed.quality_reasons


@pytest.mark.parametrize("invalid", [None, "-", "", -1, True, 1.5, "bad"])
def test_invalid_batting_totals_remain_in_raw(invalid: Any) -> None:
    raw = _batter(안타=invalid)
    parsed = parse_batter_boxscore("20010405_LGSK0", "LG", 0, raw, side="away")
    assert parsed.hits is None and not parsed.hits_verified
    assert parsed.plate_appearances is None
    assert parsed.raw["안타"] == invalid


@pytest.mark.parametrize(("raw", "outs"), [
    ({"이닝": "4"}, 12), ({"이닝": 0}, 0), ({"이닝": "5 1\\/3"}, 16),
    ({"이닝": "5 2/3"}, 17), ({"이닝": "1/3"}, 1),
    ({"inning": 5, "restinning": "1"}, 16),
    ({"inning": 5, "restinning": "2"}, 17),
    ({"inning": 0, "restinning": 0}, 0),
    ({"이닝": 5.1}, None), ({"이닝": "5.2"}, None), ({"이닝": True}, None),
    ({"이닝": "-1"}, None), ({"이닝": "5 3/3"}, None),
    ({"inning": 5, "restinning": 3}, None), ({"inning": 5}, None), ({}, None),
])
def test_innings_use_explicit_thirds_not_decimal_guessing(raw: dict[str, Any], outs: int) -> None:
    assert parse_pitcher_outs(raw) == outs


def test_pitcher_all_factual_totals_and_raw_scalar_fields() -> None:
    raw = _pitcher()
    parsed = parse_pitcher_boxscore("20010405_LGSK0", "LG", 0, raw, side="away")
    assert parsed.display_name == "해리거" and parsed.entry == "선발"
    assert (parsed.batters_faced, parsed.outs, parsed.pitches) == (22, 12, 89)
    assert (parsed.at_bats, parsed.hits, parsed.home_runs, parsed.walks_hbp) == (20, 9, 0, 2)
    assert (parsed.strikeouts, parsed.runs, parsed.earned_runs) == (3, 5, 5)
    assert (parsed.wins, parsed.losses, parsed.saves, parsed.holds) == (0, 1, 0, None)
    assert parsed.era == 11.25
    assert parsed.raw == raw and parsed.quality_reasons == ()


def test_pitcher_alternate_schema_aliases_and_zero_outing() -> None:
    raw = _pitcher(타자=0, 타수=0, 피안타=0, 투구수=0, 삼진=0, 실점=0, 자책=0)
    for key in ("이닝", "승", "패", "세"):
        del raw[key]
    raw.update(inning=0, restinning=0, 승리=0, 패배=0, 세이브=0, 홀드=0, 무승부=0)
    parsed = parse_pitcher_boxscore("20090404_HHSK0", "HH", 3, raw, side="away")
    assert parsed.outs == parsed.batters_faced == parsed.pitches == 0
    assert (parsed.wins, parsed.losses, parsed.saves, parsed.holds) == (0, 0, 0, 0)
    assert parsed.raw["무승부"] == 0


def test_pitcher_stat_constraints_flag_instead_of_discarding() -> None:
    raw = _pitcher(타자=1, 타수=2, 피안타=3, 홈런=4, 삼진=2, 실점=0, 자책=1)
    parsed = parse_pitcher_boxscore("20010405_LGSK0", "LG", 0, raw, side="away")
    assert parsed.hits == 3 and parsed.home_runs == 4 and parsed.earned_runs == 1
    assert {"hits_exceed_at_bats", "home_runs_exceed_hits", "earned_runs_exceed_runs",
            "at_bats_exceed_batters_faced", "strikeouts_exceed_batters_faced"}.issubset(
        parsed.quality_reasons,
    )


def test_game_reconciles_team_totals_and_preserves_metadata_without_input_mutation() -> None:
    raw = _one_batter_game()
    before = copy.deepcopy(raw)
    parsed = parse_historical_boxscore("20010405_LGSK0", raw)
    assert len(parsed.batters) == len(parsed.pitchers) == 2
    assert all(row.counts_verified and row.hits_verified for row in parsed.batters)
    assert parsed.quality_reasons == ()
    assert parsed.game_metadata == raw["ETC_info"]
    assert parsed.raw == before and raw == before
    parsed.raw["additional_source_field"]["value"] = 4
    assert raw == before


def test_team_pa_mismatch_invalidates_counts_but_keeps_ab_hits() -> None:
    raw = _one_batter_game()
    raw["home_pitcher"][0]["타자"] = 6
    parsed = parse_historical_boxscore("20010405_LGSK0", raw)
    away, home = parsed.batters
    assert away.plate_appearances is None and not away.counts_verified
    assert away.hits_verified and (away.at_bats, away.hits) == (5, 3)
    assert home.plate_appearances == 5 and home.counts_verified
    assert "team_pa_batters_faced_mismatch:5!=6" in away.quality_reasons


def test_team_hits_mismatch_is_not_accepted() -> None:
    raw = _one_batter_game()
    raw["scoreboard"][0]["H"] = 4
    parsed = parse_historical_boxscore("20010405_LGSK0", raw)
    assert not parsed.batters[0].hits_verified
    assert not parsed.batters[0].counts_verified
    assert parsed.batters[0].hits == 3
    assert "team_hits_mismatch:3!=4" in parsed.batters[0].quality_reasons


def test_same_name_source_rows_keep_distinct_observations() -> None:
    raw = _one_batter_game()
    raw["away_batter"].append(_batter())
    raw["scoreboard"][0]["H"] = 6
    raw["home_pitcher"][0]["타자"] = 10
    parsed = parse_historical_boxscore("20010405_LGSK0", raw)
    first, second = parsed.batters[:2]
    assert first.display_name == second.display_name == "이병규"
    assert first.observation_id != second.observation_id
    assert first.observation_id.endswith(":away:0")
    assert second.observation_id.endswith(":away:1")


def test_missing_and_malformed_records_are_reported_and_retained() -> None:
    raw = {
        "away_batter": ["데이터가 없습니다.", {"선수명": "데이터가 없습니다."}],
        "home_batter": {"unexpected": 9}, "away_pitcher": None,
        "ETC_info": ["nonmapping", "metadata"],
    }
    parsed = parse_historical_boxscore("20010405_LGSK0", raw)
    assert len(parsed.batters) == 3 and len(parsed.pitchers) == 1
    assert parsed.batters[0].raw == {"unparsed_value": "데이터가 없습니다."}
    assert parsed.batters[1].raw == {"선수명": "데이터가 없습니다."}
    assert parsed.batters[2].raw == {"unexpected": 9}
    assert parsed.pitchers[0].raw == {"unparsed_value": None}
    assert all(row.display_name is None for row in parsed.batters)
    assert all(row.plate_appearances is None for row in parsed.batters)
    assert "collection_not_list:home_batter" in parsed.quality_reasons
    assert "missing_pitching_boxscore:home" in parsed.quality_reasons
    assert parsed.game_metadata == {"unparsed_value": ["nonmapping", "metadata"]}


def test_missing_boxscores_does_not_invent_observations() -> None:
    parsed = parse_historical_boxscore("20150401_HTSK0", {"scoreboard": [{"R": 0}, {"R": 1}]})
    assert parsed.batters == parsed.pitchers == ()
    assert "missing_batting_boxscore:away" in parsed.quality_reasons
    assert "missing_pitching_boxscore:home" in parsed.quality_reasons


@pytest.mark.parametrize("key", ["2022", "20220402HHOB0", "20220402_HHOB9"])
def test_invalid_game_key_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="invalid historical game key"):
        parse_historical_boxscore(key, {})
