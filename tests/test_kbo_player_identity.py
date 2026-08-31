from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import pytest

from cpv26.data.kbo_player_identity import historical_player_prior


def test_name_team_role_prior_is_reproducible_and_explicitly_not_person_identity() -> None:
    first = historical_player_prior("kbo-team:LG", "batting", "이병규")
    same = historical_player_prior("kbo-team:LG", "batting", "이병규")
    assert first == same
    assert first.prior_id.startswith("kbo-name-team-role-cohort:")
    assert len(first.prior_id.rsplit(":", 1)[1]) == 64
    assert first.identity_status == "source_name_team_cohort"
    assert first.normalized_name == "이병규"
    assert "player_id" not in asdict(first)
    assert "observation_id" not in asdict(first)


def test_same_name_cohort_does_not_merge_or_replace_source_query_identities() -> None:
    records = [
        {"player_id": "observation:2001:away:0", "display_name": "이병규"},
        {"player_id": "observation:2011:away:8", "display_name": "이병규"},
    ]
    before = copy.deepcopy(records)
    priors = [
        historical_player_prior("kbo-team:LG", "batting", row["display_name"])
        for row in records
    ]
    assert priors[0].prior_id == priors[1].prior_id
    assert records == before
    assert records[0]["player_id"] != records[1]["player_id"]
    assert all(
        row["player_id"] != prior.prior_id for row, prior in zip(records, priors, strict=True)
    )


def test_team_role_and_name_are_each_part_of_the_cohort_key() -> None:
    keys = {
        historical_player_prior("kbo-team:LG", "batting", "이병규").prior_id,
        historical_player_prior("kbo-team:SK", "batting", "이병규").prior_id,
        historical_player_prior("kbo-team:LG", "pitching", "이병규").prior_id,
        historical_player_prior("kbo-team:LG", "batting", "이병훈").prior_id,
    }
    assert len(keys) == 4


def test_name_normalization_only_strips_outer_whitespace() -> None:
    plain = historical_player_prior("kbo-team:LG", "batting", "홍 길동")
    padded = historical_player_prior(" kbo-team:LG ", "batting", "\n 홍 길동\t")
    compact = historical_player_prior("kbo-team:LG", "batting", "홍길동")
    assert plain == padded
    assert plain.prior_id != compact.prior_id
    assert historical_player_prior("LG", "batting", "Lee").prior_id != historical_player_prior(
        "LG", "batting", "LEE",
    ).prior_id


@pytest.mark.parametrize("missing", [None, "", " ", "\n\t"])
def test_missing_name_uses_explicit_team_role_fallback(missing: str | None) -> None:
    result = historical_player_prior("kbo-team:LG", "pitching", missing)
    assert result.prior_id == "kbo-team-role-prior:pitching:kbo-team:LG"
    assert result.identity_status == "team_role_fallback"
    assert result.normalized_name is None


def test_missing_name_fallbacks_still_separate_team_and_role() -> None:
    assert len({
        historical_player_prior("LG", "batting", None).prior_id,
        historical_player_prior("LG", "pitching", None).prior_id,
        historical_player_prior("SK", "batting", None).prior_id,
    }) == 3


def test_tuple_hash_does_not_conflate_delimiters_in_source_strings() -> None:
    assert historical_player_prior("team", "batting", "x|name").prior_id != (
        historical_player_prior("name|team", "batting", "x").prior_id
    )


@pytest.mark.parametrize("team", [None, "", " ", 1])
def test_invalid_team_does_not_create_an_unscoped_cohort(team: Any) -> None:
    with pytest.raises(ValueError, match="team_id"):
        historical_player_prior(team, "batting", "선수")


@pytest.mark.parametrize("role", [None, "", "fielder", "Batting"])
def test_invalid_role_rejected(role: Any) -> None:
    with pytest.raises(ValueError, match="role"):
        historical_player_prior("LG", role, "선수")


@pytest.mark.parametrize("name", [1, True, [], {}])
def test_nontext_name_is_not_stringified_into_identity(name: Any) -> None:
    with pytest.raises(TypeError, match="display_name"):
        historical_player_prior("LG", "batting", name)
