"""Explicit historical feature cohorts, not inferred real-person identities.

The archived player boxes do not contain stable player IDs. A same-name,
same-team, same-role cohort can contain several people, and one person's
observations can fall into several cohorts after a transfer or name change.
These keys are only for pooling past feature observations. Never write one to
the canonical player table or replace a game's source-observation/query ID.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalPlayerPrior:
    """Feature-pooling provenance; ``prior_id`` is not a canonical player ID."""

    prior_id: str
    identity_status: str
    normalized_name: str | None
    team_id: str
    role: str


def historical_player_prior(
    team_id: str, role: str, display_name: str | None,
) -> HistoricalPlayerPrior:
    """Return a reproducible name/team/role cohort, or an explicit team fallback.

    Only surrounding whitespace is removed from names. Spelling, punctuation,
    case, internal whitespace, and Unicode characters are never fuzzy-matched.
    No season roster, birth date, career continuity, or actual player ID is
    asserted by this fallback. Source observation identities stay separate.
    """
    if not isinstance(team_id, str) or not team_id.strip():
        raise ValueError("team_id must be a non-empty string")
    if role not in ("batting", "pitching"):
        raise ValueError("historical player prior role must be batting or pitching")
    if display_name is not None and not isinstance(display_name, str):
        raise TypeError("display_name must be a string or None")
    team_id = team_id.strip()
    name = display_name.strip() if display_name is not None else None
    if not name:
        return HistoricalPlayerPrior(
            prior_id=f"kbo-team-role-prior:{role}:{team_id}",
            identity_status="team_role_fallback", normalized_name=None,
            team_id=team_id, role=role,
        )
    # JSON tuple encoding avoids delimiter collisions between arbitrary names
    # and team IDs. Version the recipe rather than silently changing cohorts.
    payload = json.dumps(
        ["kbo-source-name-team-role-v1", name, team_id, role],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return HistoricalPlayerPrior(
        prior_id=f"kbo-name-team-role-cohort:{digest}",
        identity_status="source_name_team_cohort", normalized_name=name,
        team_id=team_id, role=role,
    )
