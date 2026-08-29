from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cpv26.data import DuckDBStore

UTC = timezone.utc


def _temporal(at: datetime, *, available_at: datetime | None = None) -> dict[str, object]:
    available = available_at or at
    return {
        "event_at": at,
        "available_at": available,
        "ingested_at": available,
        "valid_from": at,
        "valid_to": None,
    }


def _append_reference_fixture(
    store: DuckDBStore,
    *,
    at: datetime,
    player_available_at: datetime,
    entry_team_id: str = "team-a",
) -> None:
    store.append(
        "source_revision",
        {
            "source_revision_id": "source",
            "source_name": "fixture",
            "source_locator": None,
            "content_sha256": "a" * 64,
            "metadata_json": {},
            **_temporal(at),
        },
    )
    for team_id in ("team-a", "team-b"):
        store.append(
            "team",
            {
                "team_row_id": f"{team_id}-row",
                "team_id": team_id,
                "team_name": team_id,
                "short_name": None,
                "city": None,
                "active_from": None,
                "active_to": None,
                "source_revision_id": "source",
                **_temporal(at),
            },
        )
    store.append(
        "game",
        {
            "game_row_id": "game-row",
            "game_id": "game",
            "season": 2026,
            "game_type": "regular",
            "scheduled_start": at + timedelta(hours=8),
            "home_team_id": "team-a",
            "away_team_id": "team-b",
            "stadium_id": None,
            "doubleheader_number": None,
            "resumed_from_game_id": None,
            "game_status": "scheduled",
            "home_score": None,
            "away_score": None,
            "source_revision_id": "source",
            **_temporal(at),
        },
    )
    store.append(
        "player",
        {
            "player_row_id": "player-row",
            "player_id": "player",
            "display_name": "Player",
            "birth_date": None,
            "bats": "R",
            "throws": "R",
            "primary_position": "OF",
            "debut_year": 2020,
            "source_revision_id": "source",
            **_temporal(at, available_at=player_available_at),
        },
    )
    store.append(
        "lineup_version",
        {
            "lineup_version_row_id": "lineup-version-row",
            "lineup_version_id": "lineup-version",
            "game_id": "game",
            "team_id": "team-a",
            "version_number": 1,
            "lineup_status": "official",
            "published_at": at,
            "source_revision_id": "source",
            **_temporal(at),
        },
    )
    store.append(
        "lineup_entry",
        {
            "lineup_entry_row_id": "lineup-entry-row",
            "lineup_entry_id": "lineup-entry",
            "lineup_version_id": "lineup-version",
            "game_id": "game",
            "team_id": entry_team_id,
            "player_id": "player",
            "batting_order": 1,
            "fielding_position": "OF",
            "is_starter": True,
            "source_revision_id": "source",
            **_temporal(at),
        },
    )


def test_composite_reference_audit_catches_wrong_lineup_team_tuple() -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    with DuckDBStore() as store:
        _append_reference_fixture(
            store,
            at=at,
            player_available_at=at,
            entry_team_id="team-b",
        )

        assert store.reference_violations() == ()
        violations = store.composite_reference_violations()

    matching = [
        violation
        for violation in violations
        if violation.rule.child_table == "lineup_entry"
    ]
    assert len(matching) == 1
    assert matching[0].sample_values == ("lineup-version|game|team-b",)


def test_as_of_reference_audit_rejects_parent_published_after_cutoff() -> None:
    at = datetime(2026, 8, 29, 7, 0, tzinfo=UTC)
    late = at + timedelta(hours=1)
    with DuckDBStore() as store:
        _append_reference_fixture(
            store,
            at=at,
            player_available_at=late,
        )

        assert store.reference_violations() == ()
        before = store.as_of_reference_violations(cutoff_at=at + timedelta(minutes=30))
        after = store.as_of_reference_violations(cutoff_at=late + timedelta(minutes=1))

    before_player = [
        violation
        for violation in before
        if violation.rule.name == "lineup_entry.player_id -> player.player_id"
    ]
    assert len(before_player) == 1
    assert before_player[0].sample_values == ("player",)
    assert not any(
        violation.rule.name == "lineup_entry.player_id -> player.player_id"
        for violation in after
    )
