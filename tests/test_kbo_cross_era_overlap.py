"""Overlap source precedence and point-in-time summary reconciliation."""

from __future__ import annotations

import copy
import hashlib
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from cpv26.data.kbo_graph_dataset import (
    _box_record_values,
    _common_box_records,
    _History,
    _label_boxes,
    _label_records,
    _queries,
    _Record,
)

_KST = ZoneInfo("Asia/Seoul")


def _at(day: int, hour: int = 0) -> datetime:
    return datetime(2001, 4, day, hour, tzinfo=_KST)


def _record(kind: str, row_id: str, data: dict[str, Any], available_at: datetime) -> _Record:
    return _Record(
        kind=kind,
        entity=data.get("observation_id", data.get("plate_appearance_id", row_id)),
        row_id=row_id,
        day=date(2001, 4, 1),
        event_at=_at(1, 23),
        available_at=available_at,
        ingested_at=available_at,
        valid_from=_at(1, 23),
        valid_to=None,
        source_id="source:" + kind,
        data=data,
        digest=int(hashlib.sha256(row_id.encode()).hexdigest(), 16),
    )


def _pa(*, available_at: datetime | None = None) -> _Record:
    result = _record(
        "pa",
        "pa-physical",
        {
            "game_id": "g1",
            "plate_appearance_id": "pa1",
            "batter_id": "real-batter",
            "pitcher_id": "real-pitcher",
            "batting_team_id": "away",
            "fielding_team_id": "home",
            "outcome": "single",
            "is_hit": True,
            "is_at_bat": True,
            "total_bases": 1,
            "inning": 1,
            "half_inning": "top",
            "outs_before": 0,
            "runners_before": "000",
            "home_score_before": 0,
            "away_score_before": 0,
        },
        available_at or _at(2),
    )
    result.values = np.asarray([1, 1, 1, 1, 0, 0, 0], dtype=np.float64)
    return result


def _box(
    *,
    hits: int = 1,
    available_at: datetime | None = None,
    row_id: str = "archive-physical",
    canonical: bool = False,
) -> _Record:
    data = {
        "game_id": "g1",
        "observation_id": "archive-observation",
        "role": "batting",
        "player_id": "real-batter" if canonical else "archive-query-observation",
        "identity_status": "canonical_verified" if canonical else "source_observation",
        "team_id": "away",
        "opponent_team_id": "home",
        "display_name": "원천이름",
        "stats_json": {
            "at_bats": hits,
            "hits": hits,
            "plate_appearances": hits,
            "outcome_counts": [0, 0, hits, 0, 0, 0, 0, 0, 0, 0],
            "counts_verified": True,
            "hits_verified": True,
        },
        "quality_json": [],
    }
    result = _record("box_batting", row_id, data, available_at or _at(2))
    result.box_values = _box_record_values(data)
    return result


def _query_arrays(pa: _Record, box: _Record) -> dict[str, Any]:
    return _queries(
        [],
        [pa],
        {"real-batter": 0, "real-pitcher": 1, "archive-query-observation": 2},
        {"away": 0, "home": 1},
        _label_boxes([pa, box]),
    )


def test_unresolved_overlap_uses_one_live_hit_source_and_keeps_real_pa_head() -> None:
    pa, box = _pa(), _box()
    arrays = _query_arrays(pa, box)
    assert arrays["live_hit_query_ids"].tolist() == ["g1|archive-query-observation"]
    assert arrays["live_hit_pa"].tolist() == [1]
    assert arrays["live_hit_hits"].tolist() == [1]
    assert arrays["pa_query_ids"].tolist() == ["pa1"]
    assert arrays["pa_targets"].tolist() == [2]
    assert arrays["box_pa_counts"].sum() == 1
    assert pa.data["batter_id"] == "real-batter"
    assert box.data["player_id"] == "archive-query-observation"


def test_unusable_archive_live_hit_row_does_not_suppress_real_pa_evidence() -> None:
    pa, box = _pa(), _box()
    box.data["stats_json"].update(hits=None, hits_verified=False, counts_verified=False)
    arrays = _query_arrays(pa, box)
    assert arrays["live_hit_query_ids"].tolist() == ["g1|real-batter"]
    assert arrays["live_hit_hits"].tolist() == [1]
    assert arrays["pa_query_ids"].tolist() == ["pa1"]


def test_verified_pa_fill_updates_field_verification_without_mutating_raw_source_stats() -> None:
    pa, box = _pa(), _box(canonical=True)
    box.data["stats_json"].update(
        hits=None,
        outcome_counts=None,
        hits_verified=False,
        counts_verified=False,
    )
    original = copy.deepcopy(box.data)
    merged = _common_box_records([pa, box], "batting")[0]
    stats = merged.data["stats_json"]
    assert stats["hits"] == 1 and stats["hits_verified"] is True
    assert stats["outcome_counts"] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert stats["counts_verified"] is True
    assert merged.box_values[1] == merged.box_values[10] == 1
    assert merged.box_values[9] == 1  # One field observation, not two sources.
    assert merged.data["has_pa_history"] is True
    assert box.data == original


def test_unverified_source_histogram_is_replaced_not_relabelled_as_verified() -> None:
    pa, box = _pa(), _box(canonical=True)
    unverified = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    box.data["stats_json"].update(
        hits=None,
        hits_verified=False,
        outcome_counts=unverified,
        counts_verified=False,
    )
    original = copy.deepcopy(box.data)
    merged = _common_box_records([pa, box], "batting")[0]
    assert merged.data["stats_json"]["hits"] == 1
    assert merged.data["stats_json"]["hits_verified"] is True
    assert merged.data["stats_json"]["outcome_counts"] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert merged.data["stats_json"]["counts_verified"] is True
    assert merged.box_values[1] == merged.box_values[10] == 1
    assert box.data == original


def test_future_archive_overlap_cannot_replace_already_available_pa_history_early() -> None:
    pa, late = _pa(), _box(hits=2, available_at=_at(2, 12))
    history = _History([pa, late], rolling_days=90)
    history.advance(date(2001, 4, 2))
    early = history.box_team_batting["away"].values.copy()
    assert early[0] == early[1] == early[9] == early[10] == 1
    assert list(history.box_batting) == ["real-batter"]
    history.advance(date(2001, 4, 3))
    later = history.box_team_batting["away"].values
    assert later[0] == later[1] == 2
    assert later[9] == later[10] == 1  # Replaced, never accumulated as 1+2.
    assert history.box_batting["real-batter"].values[1] == 1


def test_late_pa_overlap_does_not_double_count_existing_archive_team_history() -> None:
    pa = _pa(available_at=_at(2, 12))
    box = _box(hits=2)
    history = _History([box, pa], rolling_days=90)
    history.advance(date(2001, 4, 2))
    before = history.box_team_batting["away"].values.copy()
    assert "real-batter" not in history.box_batting
    history.advance(date(2001, 4, 3))
    np.testing.assert_array_equal(history.box_team_batting["away"].values, before)
    assert history.box_batting["real-batter"].values[1] == 1


def test_archive_revision_replaces_overlap_once_and_expiry_removes_both_sources() -> None:
    pa = _pa()
    original = _box(hits=1, row_id="archive-v1")
    correction = _box(hits=2, row_id="archive-v2", available_at=_at(2, 12))
    history = _History([pa, original, correction], rolling_days=2)
    history.advance(date(2001, 4, 2))
    assert history.box_team_batting["away"].values[1] == 1
    history.advance(date(2001, 4, 3))
    revised = history.box_team_batting["away"].values
    assert revised[1] == 2 and revised[10] == 1
    assert len(history.box_team_batting["away"].events) == 1
    history.advance(date(2001, 4, 4))
    assert not history.box_batting
    assert not history.box_team_batting
    assert not history.batting
    assert not history.box_inputs


def test_aggregate_availability_includes_the_latest_component() -> None:
    pa = _pa(available_at=_at(3))
    box = _box(canonical=True, available_at=_at(2))
    merged = _common_box_records([pa, box], "batting")[0]
    assert merged.available_at == _at(3)
    assert merged.event_at == _at(1, 23)
    assert merged.ingested_at == _at(3)
    assert merged.box_values[9] == merged.box_values[10] == 1


def test_empty_archive_does_not_suppress_any_usable_pa_task_or_team_field() -> None:
    pa, box = _pa(), _box()
    box.data["stats_json"] = {}
    box.box_values = _box_record_values(box.data)
    history = _History([pa, box], rolling_days=90)
    history.advance(date(2001, 4, 2))
    team = history.box_team_batting["away"].values
    assert team[0] == team[1] == team[9] == team[10] == 1
    arrays = _query_arrays(pa, box)
    assert arrays["box_pa_counts"].sum() == 1
    assert arrays["live_hit_query_ids"].tolist() == ["g1|real-batter"]


def test_partial_archive_uses_field_precedence_without_dropping_other_pa_fields() -> None:
    pa, box = _pa(), _box()
    box.data["stats_json"] = {"at_bats": 1}
    box.box_values = _box_record_values(box.data)
    history = _History([pa, box], rolling_days=90)
    history.advance(date(2001, 4, 2))
    team = history.box_team_batting["away"].values
    assert team[0] == team[9] == 1  # AB from archive, not archive+PA.
    assert team[1] == team[10] == 1  # H independently available from PA.
    assert team[4] == team[13] == 1
    assert _query_arrays(pa, box)["box_pa_counts"].sum() == 1


def test_partial_pa_never_verifies_a_complete_official_box_histogram() -> None:
    pa, box = _pa(), _box(hits=2, canonical=True)
    box.data["stats_json"].update(counts_verified=False, outcome_counts=None)
    box.box_values = _box_record_values(box.data)
    original = copy.deepcopy(box.data)
    merged = _common_box_records([pa, box], "batting")[0]
    assert merged.data["stats_json"]["hits"] == 2
    assert merged.data["stats_json"]["counts_verified"] is False
    assert merged.box_values[5] == merged.box_values[14] == 0  # TB remains unknown, not 1.
    arrays = _query_arrays(pa, box)
    assert arrays["box_pa_counts"].sum() == 1  # Actual partial PA still supervises this head.
    assert arrays["box_pa_query_ids"].tolist() == ["observed-pa-box:batting:g1:real-batter"]
    assert box.data == original


def test_usable_canonical_archive_live_hit_agrees_with_official_box_targets() -> None:
    arrays = _query_arrays(_pa(), _box(hits=2, canonical=True))
    assert arrays["live_hit_query_ids"].tolist() == ["g1|real-batter"]
    assert arrays["live_hit_pa"].tolist() == [2]
    assert arrays["live_hit_hits"].tolist() == [2]
    assert arrays["box_pa_counts"].tolist() == [[0, 0, 2, 0, 0, 0, 0, 0, 0, 0]]


def test_unresolved_partial_pa_fallback_does_not_invent_impossible_team_totals() -> None:
    pa, box = _pa(), _box(hits=2)
    box.data["stats_json"].update(counts_verified=False, outcome_counts=None)
    box.box_values = _box_record_values(box.data)
    history = _History([pa, box], rolling_days=90)
    history.advance(date(2001, 4, 2))
    values = history.box_team_batting["away"].values
    assert values[1] == 2
    assert values[5] == values[14] == 0  # One observed PA cannot supply official two-hit TB.


def test_pitching_target_precedence_masks_only_observed_archive_fields() -> None:
    pa = _pa()
    data = {
        "game_id": "g1",
        "observation_id": "archive-pitch",
        "role": "pitching",
        "player_id": "archive-pitcher",
        "identity_status": "source_observation",
        "team_id": "home",
        "opponent_team_id": "away",
        "display_name": "투수",
        "stats_json": {"hits": 1},
        "quality_json": [],
    }
    box = _record("box_pitching", "archive-pitch", data, _at(2))
    box.box_values = _box_record_values(data)
    arrays = _queries(
        [],
        [pa],
        {"real-batter": 0, "real-pitcher": 1, "archive-pitcher": 2},
        {"away": 0, "home": 1},
        _label_boxes([pa, box]),
    )
    assert len(arrays["box_pitch_targets"]) == 2
    assert arrays["box_pitch_mask"][:, 4].sum() == 1  # H appears once.
    assert arrays["box_pitch_targets"][:, 4].sum() == 1
    assert arrays["box_pitch_mask"][:, 0].sum() == 1  # BF still comes from actual PA.
    assert arrays["box_pitch_targets"][:, 0].sum() == 1
    assert arrays["box_pitch_mask"].sum() == 6


def test_label_snapshot_does_not_resurrect_an_expired_latest_revision() -> None:
    game = _record(
        "game",
        "g1",
        {"game_id": "g1", "game_status": "final", "home_score": 1, "away_score": 0},
        _at(2),
    )
    original = _pa()
    correction = copy.deepcopy(original)
    correction.row_id = "pa-corrected"
    correction.valid_from = correction.available_at = correction.ingested_at = _at(2, 12)
    correction.valid_to = _at(3)
    before = _label_records([game, original, correction], _at(2))
    assert [r.row_id for r in before[date(2001, 4, 1)] if r.kind == "pa"] == ["pa-physical"]
    after = _label_records([game, original, correction], _at(4))
    assert [r.kind for r in after[date(2001, 4, 1)]] == ["game"]


def test_label_snapshot_requires_publication_and_validity_not_only_ingestion() -> None:
    game = _record(
        "game",
        "g1",
        {"game_id": "g1", "game_status": "final", "home_score": 1, "away_score": 0},
        _at(2),
    )
    pa = _pa(available_at=_at(3))
    pa.ingested_at = _at(2)
    before = _label_records([game, pa], _at(2))
    assert [r.kind for r in before[date(2001, 4, 1)]] == ["game"]
    after = _label_records([game, pa], _at(3))
    assert {r.kind for r in after[date(2001, 4, 1)]} == {"game", "pa"}


def test_history_expired_latest_revision_does_not_resurrect_older_open_row() -> None:
    original = _pa()
    correction = copy.deepcopy(original)
    correction.row_id = "corrected-and-expired"
    correction.available_at = correction.valid_from = _at(2, 12)
    correction.valid_to = _at(3)
    history = _History([original, correction], rolling_days=90)
    history.advance(date(2001, 4, 2))
    assert history.box_batting["real-batter"].values[1] == 1
    history.advance(date(2001, 4, 3))
    assert not history.box_batting
    assert not history.batting
    assert not history.box_inputs
