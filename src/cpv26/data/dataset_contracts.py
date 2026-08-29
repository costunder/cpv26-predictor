"""Pure-Python contracts for training splits and source-data quality.

The storage schema intentionally stays independent from this module.  Provider
adapters can translate their rows into these small records before writing a
training dataset, which makes the checks usable in ingestion jobs, notebooks,
and tests without requiring DuckDB, NumPy, or Torch.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from cpv26.domain import utc_datetime


def _require_text(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _require_non_negative(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class DatasetViolation:
    """One deterministic, machine-readable dataset-contract failure."""

    code: str
    message: str
    game_id: str | None = None
    entity_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.code, name="code")
        _require_text(self.message, name="message")


class DatasetContractError(ValueError):
    """Raised when a dataset audit or split contract is not satisfied."""

    def __init__(self, violations: Iterable[DatasetViolation]) -> None:
        self.violations = tuple(violations)
        if not self.violations:
            raise ValueError("DatasetContractError requires at least one violation")
        preview = "; ".join(
            f"{violation.code}"
            f"[{violation.game_id or '-'}:{violation.entity_id or '-'}]: "
            f"{violation.message}"
            for violation in self.violations[:5]
        )
        remainder = len(self.violations) - 5
        if remainder > 0:
            preview = f"{preview}; and {remainder} more"
        super().__init__(preview)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Aggregate validation result; callers may inspect or raise it."""

    violations: tuple[DatasetViolation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_invalid(self) -> None:
        if self.violations:
            raise DatasetContractError(self.violations)

    @classmethod
    def combine(cls, *audits: DatasetAudit) -> DatasetAudit:
        return cls(tuple(violation for audit in audits for violation in audit.violations))


class TemporalRole(str, Enum):
    """Mutually exclusive role of a game in the default temporal protocol."""

    BASE_TRAIN = "base_train"
    MODEL_SELECTION = "model_selection"
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class GameSample:
    """Minimal identity carried by every PA, player-game, or horizon row."""

    row_id: str
    game_id: str
    season: int

    def __post_init__(self) -> None:
        _require_text(self.row_id, name="row_id")
        _require_text(self.game_id, name="game_id")
        if isinstance(self.season, bool) or not isinstance(self.season, int):
            raise TypeError("season must be an integer")
        if self.season < 1:
            raise ValueError("season must be positive")


@dataclass(frozen=True, slots=True)
class ExpandingTemporalPolicy:
    """Season roles for model development, frozen holdout, and live use."""

    base_train_start: int = 2018
    base_train_end: int = 2022
    model_selection_season: int = 2023
    calibration_season: int = 2024
    holdout_season: int = 2025
    live_season: int = 2026

    def __post_init__(self) -> None:
        for name in (
            "base_train_start",
            "base_train_end",
            "model_selection_season",
            "calibration_season",
            "holdout_season",
            "live_season",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        expected = (
            self.base_train_end + 1,
            self.model_selection_season + 1,
            self.calibration_season + 1,
            self.holdout_season + 1,
        )
        actual = (
            self.model_selection_season,
            self.calibration_season,
            self.holdout_season,
            self.live_season,
        )
        if self.base_train_start > self.base_train_end:
            raise ValueError("base_train_start cannot be later than base_train_end")
        if actual != expected:
            raise ValueError("selection, calibration, holdout, and live seasons must be contiguous")

    def role_for_season(self, season: int) -> TemporalRole | None:
        if self.base_train_start <= season <= self.base_train_end:
            return TemporalRole.BASE_TRAIN
        role_by_season = {
            self.model_selection_season: TemporalRole.MODEL_SELECTION,
            self.calibration_season: TemporalRole.CALIBRATION,
            self.holdout_season: TemporalRole.HOLDOUT,
            self.live_season: TemporalRole.LIVE,
        }
        return role_by_season.get(season)

    def target_season(self, role: TemporalRole) -> int:
        target_by_role = {
            TemporalRole.MODEL_SELECTION: self.model_selection_season,
            TemporalRole.CALIBRATION: self.calibration_season,
            TemporalRole.HOLDOUT: self.holdout_season,
            TemporalRole.LIVE: self.live_season,
        }
        if role is TemporalRole.BASE_TRAIN:
            raise ValueError("base_train is history, not an expanding-fold target")
        return target_by_role[role]


DEFAULT_EXPANDING_TEMPORAL_POLICY = ExpandingTemporalPolicy()


@dataclass(frozen=True, slots=True)
class GameGroupAssignment:
    """All row identities for one game, assigned atomically to one role."""

    game_id: str
    season: int
    role: TemporalRole
    row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpandingTemporalFold:
    """Historical games available before one stage's target season."""

    target_role: TemporalRole
    target_season: int
    training_seasons: tuple[int, ...]
    training_game_ids: tuple[str, ...]
    target_game_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.training_game_ids).intersection(self.target_game_ids)
        if overlap:
            games = ", ".join(sorted(overlap))
            raise ValueError(f"training and target game groups overlap: {games}")


@dataclass(frozen=True, slots=True)
class ExpandingTemporalSplit:
    """Deterministic game-group assignments and expanding stage manifests."""

    assignments: tuple[GameGroupAssignment, ...]
    folds: tuple[ExpandingTemporalFold, ...]

    def games_for_role(self, role: TemporalRole) -> tuple[str, ...]:
        return tuple(item.game_id for item in self.assignments if item.role is role)

    def rows_for_role(self, role: TemporalRole) -> tuple[str, ...]:
        return tuple(
            row_id
            for item in self.assignments
            if item.role is role
            for row_id in item.row_ids
        )

    def assignment_for(self, game_id: str) -> GameGroupAssignment:
        for item in self.assignments:
            if item.game_id == game_id:
                return item
        raise KeyError(game_id)

    def fold_for(self, target_role: TemporalRole) -> ExpandingTemporalFold:
        for fold in self.folds:
            if fold.target_role is target_role:
                return fold
        raise KeyError(target_role)


def build_expanding_temporal_split(
    samples: Iterable[GameSample],
    *,
    policy: ExpandingTemporalPolicy = DEFAULT_EXPANDING_TEMPORAL_POLICY,
) -> ExpandingTemporalSplit:
    """Assign rows by game and construct strict expanding temporal folds.

    A ``game_id`` may occur in many PA/player/horizon rows, but all rows remain
    in one role.  Conflicting seasons, duplicate row identities, and seasons
    outside the declared protocol are rejected instead of being silently
    dropped.
    """

    materialized = tuple(samples)
    violations: list[DatasetViolation] = []
    duplicate_row_ids = sorted(
        row_id
        for row_id, count in Counter(sample.row_id for sample in materialized).items()
        if count > 1
    )
    for row_id in duplicate_row_ids:
        violations.append(
            DatasetViolation(
                code="DUPLICATE_SAMPLE_ROW_ID",
                message="row_id must identify exactly one training sample",
                entity_id=row_id,
            )
        )

    grouped: dict[str, list[GameSample]] = defaultdict(list)
    for sample in materialized:
        grouped[sample.game_id].append(sample)

    for game_id, group in sorted(grouped.items()):
        seasons = sorted({sample.season for sample in group})
        if len(seasons) != 1:
            violations.append(
                DatasetViolation(
                    code="GAME_SEASON_CONFLICT",
                    message=f"one game_id has multiple seasons: {seasons}",
                    game_id=game_id,
                )
            )
            continue
        if policy.role_for_season(seasons[0]) is None:
            violations.append(
                DatasetViolation(
                    code="SEASON_OUTSIDE_POLICY",
                    message=f"season {seasons[0]} is outside the declared temporal protocol",
                    game_id=game_id,
                )
            )

    DatasetAudit(tuple(violations)).raise_if_invalid()

    assignments: list[GameGroupAssignment] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (item[1][0].season, item[0]),
    )
    for game_id, group in ordered_groups:
        season = group[0].season
        role = policy.role_for_season(season)
        if role is None:  # pragma: no cover - rejected by the aggregate audit above
            raise AssertionError("validated season must have a temporal role")
        assignments.append(
            GameGroupAssignment(
                game_id=game_id,
                season=season,
                role=role,
                row_ids=tuple(sorted(sample.row_id for sample in group)),
            )
        )

    target_roles = (
        TemporalRole.MODEL_SELECTION,
        TemporalRole.CALIBRATION,
        TemporalRole.HOLDOUT,
        TemporalRole.LIVE,
    )
    folds: list[ExpandingTemporalFold] = []
    for target_role in target_roles:
        target_season = policy.target_season(target_role)
        training_seasons = tuple(range(policy.base_train_start, target_season))
        folds.append(
            ExpandingTemporalFold(
                target_role=target_role,
                target_season=target_season,
                training_seasons=training_seasons,
                training_game_ids=tuple(
                    item.game_id for item in assignments if item.season < target_season
                ),
                target_game_ids=tuple(
                    item.game_id for item in assignments if item.season == target_season
                ),
            )
        )

    return ExpandingTemporalSplit(tuple(assignments), tuple(folds))


@dataclass(frozen=True, slots=True)
class PlateAppearanceRow:
    """Provider-neutral subset of ``observed_plate_appearance`` used in audits."""

    plate_appearance_id: str
    game_id: str
    batter_id: str
    batting_team_id: str
    is_at_bat: bool
    is_hit: bool
    total_bases: int
    runs_scored: int

    def __post_init__(self) -> None:
        for name in ("plate_appearance_id", "game_id", "batter_id", "batting_team_id"):
            _require_text(getattr(self, name), name=name)
        if not isinstance(self.is_at_bat, bool) or not isinstance(self.is_hit, bool):
            raise TypeError("is_at_bat and is_hit must be booleans")
        _require_non_negative(self.total_bases, name="total_bases")
        _require_non_negative(self.runs_scored, name="runs_scored")
        if self.total_bases > 4:
            raise ValueError("total_bases cannot exceed four")


@dataclass(frozen=True, slots=True)
class PlayerGameBattingRow:
    """Official player-game box-score totals used to check PA materialization."""

    player_game_batting_id: str
    game_id: str
    player_id: str
    team_id: str
    plate_appearances: int
    at_bats: int
    hits: int
    singles: int
    doubles: int
    triples: int
    home_runs: int
    started: bool | None = None
    batting_order: int | None = None

    def __post_init__(self) -> None:
        for name in ("player_game_batting_id", "game_id", "player_id", "team_id"):
            _require_text(getattr(self, name), name=name)
        for name in (
            "plate_appearances",
            "at_bats",
            "hits",
            "singles",
            "doubles",
            "triples",
            "home_runs",
        ):
            _require_non_negative(getattr(self, name), name=name)
        if self.started is not None and not isinstance(self.started, bool):
            raise TypeError("started must be a boolean or None")
        if self.batting_order is not None and not 1 <= self.batting_order <= 9:
            raise ValueError("batting_order must be between one and nine")


@dataclass(frozen=True, slots=True)
class TeamGameRow:
    """Final team line from an official game box score."""

    team_game_id: str
    game_id: str
    team_id: str
    opponent_team_id: str
    is_home: bool
    runs: int | None

    def __post_init__(self) -> None:
        for name in ("team_game_id", "game_id", "team_id", "opponent_team_id"):
            _require_text(getattr(self, name), name=name)
        if self.team_id == self.opponent_team_id:
            raise ValueError("team_id and opponent_team_id must differ")
        if not isinstance(self.is_home, bool):
            raise TypeError("is_home must be a boolean")
        if self.runs is not None:
            _require_non_negative(self.runs, name="runs")


@dataclass(slots=True)
class _BattingTally:
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0


_COUNT_FIELDS = (
    "plate_appearances",
    "at_bats",
    "hits",
    "singles",
    "doubles",
    "triples",
    "home_runs",
)


def audit_player_game_batting(
    plate_appearances: Iterable[PlateAppearanceRow],
    batting_rows: Iterable[PlayerGameBattingRow],
) -> DatasetAudit:
    """Compare PA-derived player totals with official player-game totals."""

    appearances = tuple(plate_appearances)
    summaries = tuple(batting_rows)
    violations: list[DatasetViolation] = []

    for plate_appearance_id, count in sorted(
        Counter(row.plate_appearance_id for row in appearances).items()
    ):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_PLATE_APPEARANCE",
                    message=f"plate appearance occurs {count} times",
                    entity_id=plate_appearance_id,
                )
            )

    tallies: dict[tuple[str, str], _BattingTally] = defaultdict(_BattingTally)
    teams_by_player_game: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pa_row in appearances:
        key = (pa_row.game_id, pa_row.batter_id)
        tally = tallies[key]
        tally.plate_appearances += 1
        tally.at_bats += int(pa_row.is_at_bat)
        tally.hits += int(pa_row.is_hit)
        teams_by_player_game[key].add(pa_row.batting_team_id)
        if pa_row.is_hit:
            if not pa_row.is_at_bat:
                violations.append(
                    DatasetViolation(
                        code="HIT_WITHOUT_AT_BAT",
                        message="an official hit must also be an at-bat",
                        game_id=pa_row.game_id,
                        entity_id=pa_row.plate_appearance_id,
                    )
                )
            if pa_row.total_bases == 1:
                tally.singles += 1
            elif pa_row.total_bases == 2:
                tally.doubles += 1
            elif pa_row.total_bases == 3:
                tally.triples += 1
            elif pa_row.total_bases == 4:
                tally.home_runs += 1
            else:
                violations.append(
                    DatasetViolation(
                        code="HIT_TOTAL_BASES_INVALID",
                        message="a hit must have total_bases in the range one through four",
                        game_id=pa_row.game_id,
                        entity_id=pa_row.plate_appearance_id,
                    )
                )
        elif pa_row.total_bases != 0:
            violations.append(
                DatasetViolation(
                    code="NON_HIT_WITH_TOTAL_BASES",
                    message="a non-hit PA cannot carry official total bases",
                    game_id=pa_row.game_id,
                    entity_id=pa_row.plate_appearance_id,
                )
            )

    summary_by_key: dict[tuple[str, str], PlayerGameBattingRow] = {}
    for batting_row in summaries:
        key = (batting_row.game_id, batting_row.player_id)
        if key in summary_by_key:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_PLAYER_GAME_BATTING",
                    message="official player-game totals must be unique by game and player",
                    game_id=batting_row.game_id,
                    entity_id=batting_row.player_id,
                )
            )
        else:
            summary_by_key[key] = batting_row
        if batting_row.at_bats > batting_row.plate_appearances:
            violations.append(
                DatasetViolation(
                    code="BOX_AT_BATS_EXCEED_PA",
                    message="official at-bats cannot exceed plate appearances",
                    game_id=batting_row.game_id,
                    entity_id=batting_row.player_id,
                )
            )
        if batting_row.hits > batting_row.at_bats:
            violations.append(
                DatasetViolation(
                    code="BOX_HITS_EXCEED_AT_BATS",
                    message="official hits cannot exceed at-bats",
                    game_id=batting_row.game_id,
                    entity_id=batting_row.player_id,
                )
            )
        hit_type_sum = (
            batting_row.singles
            + batting_row.doubles
            + batting_row.triples
            + batting_row.home_runs
        )
        if hit_type_sum != batting_row.hits:
            violations.append(
                DatasetViolation(
                    code="BOX_HIT_TYPES_MISMATCH",
                    message=(
                        f"hit-type sum {hit_type_sum} does not equal hits {batting_row.hits}"
                    ),
                    game_id=batting_row.game_id,
                    entity_id=batting_row.player_id,
                )
            )

    for key in sorted(set(tallies).union(summary_by_key)):
        game_id, player_id = key
        tally = tallies.get(key, _BattingTally())
        summary = summary_by_key.get(key)
        if summary is None:
            violations.append(
                DatasetViolation(
                    code="PLAYER_GAME_BATTING_MISSING",
                    message="PA rows have no official player-game batting total",
                    game_id=game_id,
                    entity_id=player_id,
                )
            )
            continue
        teams = teams_by_player_game.get(key, set())
        if teams and teams != {summary.team_id}:
            violations.append(
                DatasetViolation(
                    code="PLAYER_GAME_TEAM_MISMATCH",
                    message=(
                        f"PA teams {sorted(teams)} do not match "
                        f"box-score team {summary.team_id}"
                    ),
                    game_id=game_id,
                    entity_id=player_id,
                )
            )
        for field_name in _COUNT_FIELDS:
            actual = int(getattr(tally, field_name))
            official = int(getattr(summary, field_name))
            if actual != official:
                violations.append(
                    DatasetViolation(
                        code=f"PLAYER_GAME_{field_name.upper()}_MISMATCH",
                        message=f"PA aggregate {actual} does not equal official total {official}",
                        game_id=game_id,
                        entity_id=player_id,
                    )
                )

    return DatasetAudit(tuple(violations))


def audit_team_scores(
    plate_appearances: Iterable[PlateAppearanceRow],
    team_games: Iterable[TeamGameRow],
) -> DatasetAudit:
    """Compare event runs with final team scores and reciprocal game rows."""

    appearances = tuple(plate_appearances)
    summaries = tuple(team_games)
    violations: list[DatasetViolation] = []
    pa_runs: dict[tuple[str, str], int] = defaultdict(int)
    games_with_pa: set[str] = set()
    for pa_row in appearances:
        pa_runs[(pa_row.game_id, pa_row.batting_team_id)] += pa_row.runs_scored
        games_with_pa.add(pa_row.game_id)

    for team_game_id, count in sorted(Counter(row.team_game_id for row in summaries).items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_TEAM_GAME_ID",
                    message=f"team_game_id occurs {count} times",
                    entity_id=team_game_id,
                )
            )

    by_key: dict[tuple[str, str], TeamGameRow] = {}
    rows_by_game: dict[str, list[TeamGameRow]] = defaultdict(list)
    for team_row in summaries:
        key = (team_row.game_id, team_row.team_id)
        rows_by_game[team_row.game_id].append(team_row)
        if key in by_key:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_TEAM_GAME_TEAM",
                    message="team result must be unique by game and team",
                    game_id=team_row.game_id,
                    entity_id=team_row.team_id,
                )
            )
        else:
            by_key[key] = team_row

    for key in sorted(pa_runs):
        if key not in by_key:
            violations.append(
                DatasetViolation(
                    code="TEAM_GAME_MISSING",
                    message="PA events have no final team-game row",
                    game_id=key[0],
                    entity_id=key[1],
                )
            )

    for key, team_row in sorted(by_key.items()):
        expected_runs = pa_runs.get(key, 0)
        if team_row.runs is None:
            violations.append(
                DatasetViolation(
                    code="TEAM_SCORE_MISSING",
                    message="training team-game row must have a final run total",
                    game_id=team_row.game_id,
                    entity_id=team_row.team_id,
                )
            )
        elif expected_runs != team_row.runs:
            violations.append(
                DatasetViolation(
                    code="TEAM_SCORE_MISMATCH",
                    message=(
                        f"PA runs {expected_runs} do not equal final score {team_row.runs}"
                    ),
                    game_id=team_row.game_id,
                    entity_id=team_row.team_id,
                )
            )

    for game_id in sorted(games_with_pa.union(rows_by_game)):
        rows = rows_by_game.get(game_id, [])
        if len(rows) != 2:
            violations.append(
                DatasetViolation(
                    code="TEAM_GAME_PAIR_INCOMPLETE",
                    message=f"completed training game must have two team rows, found {len(rows)}",
                    game_id=game_id,
                )
            )
            continue
        home_count = sum(row.is_home for row in rows)
        if home_count != 1:
            violations.append(
                DatasetViolation(
                    code="TEAM_GAME_HOME_AWAY_INVALID",
                    message="team rows must contain exactly one home and one away team",
                    game_id=game_id,
                )
            )
        first, second = rows
        if first.team_id != second.opponent_team_id or second.team_id != first.opponent_team_id:
            violations.append(
                DatasetViolation(
                    code="TEAM_GAME_OPPONENT_MISMATCH",
                    message="team rows must reference each other as opponents",
                    game_id=game_id,
                )
            )

    return DatasetAudit(tuple(violations))


def audit_box_score_consistency(
    plate_appearances: Iterable[PlateAppearanceRow],
    batting_rows: Iterable[PlayerGameBattingRow],
    team_games: Iterable[TeamGameRow],
) -> DatasetAudit:
    """Run both official-box-score reconciliation contracts."""

    appearances = tuple(plate_appearances)
    return DatasetAudit.combine(
        audit_player_game_batting(appearances, batting_rows),
        audit_team_scores(appearances, team_games),
    )


class TransitionHalf(str, Enum):
    """Half-inning label carried by a materialized historical transition."""

    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True, slots=True)
class TransitionRow:
    """PA pre/post state required to fit a sequential transition model.

    Legacy rows may set ``transition_complete=False`` and leave post-state
    fields absent.  They remain valid for PA-outcome learning, but a
    simulator-ready audit rejects them explicitly.
    """

    plate_appearance_id: str
    game_id: str
    sequence_in_game: int
    event_subsequence: int
    inning: int
    half_inning: TransitionHalf
    home_score_before: int | None
    away_score_before: int | None
    outs_before: int
    runners_before: str
    outs_added: int | None
    runners_after: str | None
    home_score_after: int | None
    away_score_after: int | None
    runs_scored: int
    transition_complete: bool

    def __post_init__(self) -> None:
        _require_text(self.plate_appearance_id, name="plate_appearance_id")
        _require_text(self.game_id, name="game_id")
        for name in (
            "sequence_in_game",
            "event_subsequence",
            "inning",
            "outs_before",
            "runs_scored",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in (
            "home_score_before",
            "away_score_before",
            "outs_added",
            "home_score_after",
            "away_score_after",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an integer or None")
        if not isinstance(self.half_inning, TransitionHalf):
            raise TypeError("half_inning must be a TransitionHalf")
        if not isinstance(self.transition_complete, bool):
            raise TypeError("transition_complete must be a boolean")
        if not isinstance(self.runners_before, str):
            raise TypeError("runners_before must be a string bitmap")
        if self.runners_after is not None and not isinstance(self.runners_after, str):
            raise TypeError("runners_after must be a string bitmap or None")

    @property
    def order_key(self) -> tuple[int, int]:
        return self.sequence_in_game, self.event_subsequence


@dataclass(frozen=True, slots=True)
class RunnerEventMarker:
    """Ordering marker that suppresses direct PA-to-PA state continuity checks."""

    runner_event_id: str
    game_id: str
    sequence_in_game: int
    event_subsequence: int

    def __post_init__(self) -> None:
        _require_text(self.runner_event_id, name="runner_event_id")
        _require_text(self.game_id, name="game_id")
        for name in ("sequence_in_game", "event_subsequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.sequence_in_game < 1:
            raise ValueError("sequence_in_game must be at least one")
        if self.event_subsequence < 0:
            raise ValueError("event_subsequence cannot be negative")

    @property
    def order_key(self) -> tuple[int, int]:
        return self.sequence_in_game, self.event_subsequence


def _valid_runner_bitmap(value: str) -> bool:
    return len(value) == 3 and set(value) <= {"0", "1"}


def _transition_required_fields_missing(row: TransitionRow) -> tuple[str, ...]:
    fields = (
        "home_score_before",
        "away_score_before",
        "outs_added",
        "runners_after",
        "home_score_after",
        "away_score_after",
    )
    return tuple(name for name in fields if getattr(row, name) is None)


def _runner_event_between(
    previous: TransitionRow,
    current: TransitionRow,
    markers_by_game: Mapping[str, tuple[RunnerEventMarker, ...]],
) -> bool:
    return any(
        previous.order_key < marker.order_key < current.order_key
        for marker in markers_by_game.get(previous.game_id, ())
    )


def _next_half(inning: int, half: TransitionHalf) -> tuple[int, TransitionHalf]:
    if half is TransitionHalf.TOP:
        return inning, TransitionHalf.BOTTOM
    return inning + 1, TransitionHalf.TOP


def audit_pa_transitions(
    rows: Iterable[TransitionRow],
    *,
    runner_events: Iterable[RunnerEventMarker] = (),
    simulator_ready: bool = False,
) -> DatasetAudit:
    """Audit PA state transitions and simple consecutive-PA continuity.

    Direct continuity is checked only when no explicitly supplied runner event
    falls between two consecutive PA order keys.  A runner event can change
    bases, outs, score, or even end a half-inning, so those pairs require a
    richer event replay rather than an equality check.
    """

    transitions = tuple(rows)
    markers = tuple(runner_events)
    violations: list[DatasetViolation] = []

    duplicate_pa_ids = Counter(row.plate_appearance_id for row in transitions)
    for plate_appearance_id, count in sorted(duplicate_pa_ids.items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_TRANSITION_PA",
                    message=f"plate appearance transition occurs {count} times",
                    entity_id=plate_appearance_id,
                )
            )

    transitions_by_game: dict[str, list[TransitionRow]] = defaultdict(list)
    for row in transitions:
        transitions_by_game[row.game_id].append(row)
    markers_by_game: dict[str, list[RunnerEventMarker]] = defaultdict(list)
    for marker in markers:
        markers_by_game[marker.game_id].append(marker)
    frozen_markers = {
        game_id: tuple(sorted(game_markers, key=lambda item: item.order_key))
        for game_id, game_markers in markers_by_game.items()
    }

    for row in transitions:
        entity_id = row.plate_appearance_id
        if row.sequence_in_game < 1:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_SEQUENCE_INVALID",
                    message="sequence_in_game must be at least one",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if row.event_subsequence < 0:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_SUBSEQUENCE_INVALID",
                    message="event_subsequence cannot be negative",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if row.inning < 1:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_INNING_INVALID",
                    message="inning must be at least one",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if not 0 <= row.outs_before <= 2:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_OUTS_BEFORE_INVALID",
                    message="outs_before must be in the range zero through two",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if not _valid_runner_bitmap(row.runners_before):
            violations.append(
                DatasetViolation(
                    code="TRANSITION_RUNNERS_BEFORE_INVALID",
                    message="runners_before must be a three-bit occupancy bitmap",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if row.runs_scored < 0:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_RUNS_INVALID",
                    message="runs_scored cannot be negative",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        for field_name in (
            "home_score_before",
            "away_score_before",
            "home_score_after",
            "away_score_after",
        ):
            value = getattr(row, field_name)
            if value is not None and value < 0:
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_SCORE_INVALID",
                        message=f"{field_name} cannot be negative",
                        game_id=row.game_id,
                        entity_id=entity_id,
                    )
                )

        missing = _transition_required_fields_missing(row)
        if row.transition_complete and missing:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_COMPLETE_FIELDS_MISSING",
                    message=f"complete transition is missing fields: {', '.join(missing)}",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if simulator_ready and not row.transition_complete:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_INCOMPLETE_FOR_SIMULATOR",
                    message="legacy PA transition cannot enter a simulator-ready dataset",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if not row.transition_complete or missing:
            continue

        if row.outs_added is None or row.runners_after is None:
            raise AssertionError("validated complete transition must have outs and runners")
        if not 0 <= row.outs_added <= 3:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_OUTS_ADDED_INVALID",
                    message="outs_added must be in the range zero through three",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        total_outs = row.outs_before + row.outs_added
        if total_outs > 3:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_OUTS_OVERFLOW",
                    message=f"outs_before + outs_added is {total_outs}, above three",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if not _valid_runner_bitmap(row.runners_after):
            violations.append(
                DatasetViolation(
                    code="TRANSITION_RUNNERS_AFTER_INVALID",
                    message="runners_after must be a three-bit occupancy bitmap",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        if total_outs == 3 and row.runners_after != "000":
            violations.append(
                DatasetViolation(
                    code="TRANSITION_RUNNERS_AFTER_THREE_OUTS",
                    message="runners_after must be empty when the half-inning ends",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )

        home_before = row.home_score_before
        away_before = row.away_score_before
        home_after = row.home_score_after
        away_after = row.away_score_after
        if home_before is None or away_before is None:
            raise AssertionError("validated complete transition must have pre-score fields")
        if home_after is None or away_after is None:
            raise AssertionError("validated complete transition must have post-score fields")
        home_delta = home_after - home_before
        away_delta = away_after - away_before
        if home_delta < 0 or away_delta < 0:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_SCORE_DECREASED",
                    message="a completed PA cannot decrease either team score",
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )
        expected_home_delta = row.runs_scored if row.half_inning is TransitionHalf.BOTTOM else 0
        expected_away_delta = row.runs_scored if row.half_inning is TransitionHalf.TOP else 0
        if home_delta != expected_home_delta or away_delta != expected_away_delta:
            violations.append(
                DatasetViolation(
                    code="TRANSITION_SCORE_DELTA_MISMATCH",
                    message=(
                        f"score delta home={home_delta}, away={away_delta} "
                        f"does not match {row.half_inning.value} runs={row.runs_scored}"
                    ),
                    game_id=row.game_id,
                    entity_id=entity_id,
                )
            )

    for game_id, game_rows in sorted(transitions_by_game.items()):
        order_counts = Counter(row.order_key for row in game_rows)
        for order_key, count in sorted(order_counts.items()):
            if count > 1:
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_ORDER_DUPLICATE",
                        message=f"order key {order_key} occurs {count} times",
                        game_id=game_id,
                    )
                )
        ordered = sorted(game_rows, key=lambda item: (*item.order_key, item.plate_appearance_id))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if (
                not previous.transition_complete
                or not current.transition_complete
                or _transition_required_fields_missing(previous)
                or _transition_required_fields_missing(current)
                or _runner_event_between(previous, current, frozen_markers)
            ):
                continue
            if previous.order_key >= current.order_key:
                continue
            if previous.outs_added is None or previous.runners_after is None:
                raise AssertionError("complete previous transition has a post-state")
            previous_total_outs = previous.outs_before + previous.outs_added
            if previous_total_outs < 3:
                expected_inning = previous.inning
                expected_half = previous.half_inning
                expected_outs = previous_total_outs
                expected_runners = previous.runners_after
            elif previous_total_outs == 3:
                expected_inning, expected_half = _next_half(
                    previous.inning,
                    previous.half_inning,
                )
                expected_outs = 0
                expected_runners = "000"
            else:
                continue

            if (current.inning, current.half_inning) != (expected_inning, expected_half):
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_NEXT_HALF_MISMATCH",
                        message=(
                            f"expected inning/half {expected_inning}/{expected_half.value}, "
                            f"found {current.inning}/{current.half_inning.value}"
                        ),
                        game_id=game_id,
                        entity_id=current.plate_appearance_id,
                    )
                )
            if current.outs_before != expected_outs:
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_NEXT_OUTS_MISMATCH",
                        message=(
                            f"expected next outs_before {expected_outs}, "
                            f"found {current.outs_before}"
                        ),
                        game_id=game_id,
                        entity_id=current.plate_appearance_id,
                    )
                )
            if current.runners_before != expected_runners:
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_NEXT_RUNNERS_MISMATCH",
                        message=(
                            f"expected next runners_before {expected_runners}, "
                            f"found {current.runners_before}"
                        ),
                        game_id=game_id,
                        entity_id=current.plate_appearance_id,
                    )
                )
            if (
                current.home_score_before != previous.home_score_after
                or current.away_score_before != previous.away_score_after
            ):
                violations.append(
                    DatasetViolation(
                        code="TRANSITION_NEXT_SCORE_MISMATCH",
                        message="next PA pre-score does not equal previous PA post-score",
                        game_id=game_id,
                        entity_id=current.plate_appearance_id,
                    )
                )

    return DatasetAudit(tuple(violations))


def assert_simulator_ready_transitions(
    rows: Iterable[TransitionRow],
    *,
    runner_events: Iterable[RunnerEventMarker] = (),
) -> None:
    """Raise unless every PA has a valid, replayable pre/post transition."""

    audit_pa_transitions(
        rows,
        runner_events=runner_events,
        simulator_ready=True,
    ).raise_if_invalid()


class V26CapturePhase(str, Enum):
    """Required point-in-time selection-rate capture horizons."""

    EARLY = "early"
    STARTER_KNOWN = "starter_known"
    LINEUP_KNOWN = "lineup_known"
    NEAR_LOCK = "near_lock"


V26_CAPTURE_PHASES: tuple[V26CapturePhase, ...] = tuple(V26CapturePhase)


@dataclass(frozen=True, slots=True)
class V26RuleVersionRow:
    """Minimal existence contract for a versioned V26 scoring rule."""

    rule_version: str

    def __post_init__(self) -> None:
        _require_text(self.rule_version, name="rule_version")


@dataclass(frozen=True, slots=True)
class V26SlateRow:
    """Immutable identities that every capture for one V26 slate must share."""

    slate_id: str
    rule_version: str
    live_card_version: str
    position_eligibility_snapshot_id: str
    lock_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "slate_id",
            "rule_version",
            "live_card_version",
            "position_eligibility_snapshot_id",
        ):
            _require_text(getattr(self, name), name=name)
        object.__setattr__(self, "lock_at", utc_datetime(self.lock_at, field_name="lock_at"))


@dataclass(frozen=True, slots=True)
class V26EligibilityRow:
    """Player-position eligibility bound to one slate/card snapshot."""

    eligibility_row_id: str
    slate_id: str
    position_eligibility_snapshot_id: str
    live_card_version: str
    player_id: str
    position: str
    is_eligible: bool

    def __post_init__(self) -> None:
        for name in (
            "eligibility_row_id",
            "slate_id",
            "position_eligibility_snapshot_id",
            "live_card_version",
            "player_id",
            "position",
        ):
            _require_text(getattr(self, name), name=name)
        if not isinstance(self.is_eligible, bool):
            raise TypeError("is_eligible must be a boolean")


def _v26_eligibility_key(row: V26EligibilityRow) -> tuple[str, str, str, str, str]:
    return (
        row.position_eligibility_snapshot_id,
        row.slate_id,
        row.live_card_version,
        row.player_id,
        row.position,
    )


@dataclass(frozen=True, slots=True)
class V26CaptureRow:
    """One player-position selection-rate capture enriched with slate identity."""

    capture_row_id: str
    selection_snapshot_id: str
    slate_id: str
    phase: V26CapturePhase
    rule_version: str
    live_card_version: str
    position_eligibility_snapshot_id: str
    player_id: str
    position: str
    captured_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "capture_row_id",
            "selection_snapshot_id",
            "slate_id",
            "rule_version",
            "live_card_version",
            "position_eligibility_snapshot_id",
            "player_id",
            "position",
        ):
            _require_text(getattr(self, name), name=name)
        if not isinstance(self.phase, V26CapturePhase):
            raise TypeError("phase must be an explicit V26CapturePhase")
        object.__setattr__(
            self,
            "captured_at",
            utc_datetime(self.captured_at, field_name="captured_at"),
        )


def audit_v26_capture_consistency(
    slates: Iterable[V26SlateRow],
    rules: Iterable[V26RuleVersionRow],
    eligibility_rows: Iterable[V26EligibilityRow],
    captures: Iterable[V26CaptureRow],
    *,
    require_all_phases: bool = True,
) -> DatasetAudit:
    """Audit V26 horizon coverage and slate/rule/card/eligibility identity."""

    slate_rows = tuple(slates)
    rule_rows = tuple(rules)
    eligibility = tuple(eligibility_rows)
    capture_rows = tuple(captures)
    violations: list[DatasetViolation] = []

    rule_counts = Counter(row.rule_version for row in rule_rows)
    for rule_version, count in sorted(rule_counts.items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="V26_RULE_VERSION_DUPLICATE",
                    message=f"rule version occurs {count} times",
                    entity_id=rule_version,
                )
            )
    known_rules = set(rule_counts)

    slate_counts = Counter(row.slate_id for row in slate_rows)
    slate_by_id: dict[str, V26SlateRow] = {}
    for slate_row in slate_rows:
        if slate_row.slate_id not in slate_by_id:
            slate_by_id[slate_row.slate_id] = slate_row
    for slate_id, count in sorted(slate_counts.items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="V26_SLATE_DUPLICATE",
                    message=f"slate_id occurs {count} times",
                    entity_id=slate_id,
                )
            )
    for slate_row in slate_by_id.values():
        if slate_row.rule_version not in known_rules:
            violations.append(
                DatasetViolation(
                    code="V26_SLATE_RULE_MISSING",
                    message=f"slate references unknown rule version {slate_row.rule_version}",
                    game_id=slate_row.slate_id,
                    entity_id=slate_row.rule_version,
                )
            )

    eligibility_counts = Counter(_v26_eligibility_key(row) for row in eligibility)
    eligibility_by_key: dict[
        tuple[str, str, str, str, str],
        V26EligibilityRow,
    ] = {}
    for eligibility_row in eligibility:
        eligibility_identity = _v26_eligibility_key(eligibility_row)
        if eligibility_identity not in eligibility_by_key:
            eligibility_by_key[eligibility_identity] = eligibility_row
    for eligibility_identity, count in sorted(eligibility_counts.items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="V26_ELIGIBILITY_DUPLICATE",
                    message=f"eligibility identity occurs {count} times",
                    game_id=eligibility_identity[1],
                    entity_id=f"{eligibility_identity[3]}:{eligibility_identity[4]}",
                )
            )
    for eligibility_row in eligibility:
        slate_contract = slate_by_id.get(eligibility_row.slate_id)
        if slate_contract is None:
            violations.append(
                DatasetViolation(
                    code="V26_ELIGIBILITY_SLATE_MISSING",
                    message="eligibility row references an unknown slate",
                    game_id=eligibility_row.slate_id,
                    entity_id=eligibility_row.eligibility_row_id,
                )
            )
            continue
        if (
            eligibility_row.position_eligibility_snapshot_id
            != slate_contract.position_eligibility_snapshot_id
        ):
            violations.append(
                DatasetViolation(
                    code="V26_ELIGIBILITY_SNAPSHOT_MISMATCH",
                    message="eligibility snapshot does not match the slate contract",
                    game_id=eligibility_row.slate_id,
                    entity_id=eligibility_row.eligibility_row_id,
                )
            )
        if eligibility_row.live_card_version != slate_contract.live_card_version:
            violations.append(
                DatasetViolation(
                    code="V26_ELIGIBILITY_CARD_MISMATCH",
                    message="eligibility live-card version does not match the slate",
                    game_id=eligibility_row.slate_id,
                    entity_id=eligibility_row.eligibility_row_id,
                )
            )

    capture_identity_counts = Counter(
        (row.slate_id, row.phase, row.player_id, row.position) for row in capture_rows
    )
    for capture_identity, count in sorted(
        capture_identity_counts.items(),
        key=lambda item: (item[0][0], item[0][1].value, item[0][2], item[0][3]),
    ):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_DUPLICATE",
                    message=f"player-position phase capture occurs {count} times",
                    game_id=capture_identity[0],
                    entity_id=(
                        f"{capture_identity[2]}:{capture_identity[3]}:"
                        f"{capture_identity[1].value}"
                    ),
                )
            )

    snapshots_by_slate_phase: dict[
        tuple[str, V26CapturePhase],
        set[str],
    ] = defaultdict(set)
    times_by_slate_phase: dict[
        tuple[str, V26CapturePhase],
        set[datetime],
    ] = defaultdict(set)
    uses_by_snapshot: dict[str, set[tuple[str, V26CapturePhase]]] = defaultdict(set)
    phases_by_entity: dict[tuple[str, str, str], set[V26CapturePhase]] = defaultdict(set)
    phases_by_slate: dict[str, set[V26CapturePhase]] = defaultdict(set)

    for capture_row in capture_rows:
        slate_contract = slate_by_id.get(capture_row.slate_id)
        phase_key = (capture_row.slate_id, capture_row.phase)
        snapshots_by_slate_phase[phase_key].add(capture_row.selection_snapshot_id)
        times_by_slate_phase[phase_key].add(capture_row.captured_at)
        uses_by_snapshot[capture_row.selection_snapshot_id].add(phase_key)
        phases_by_entity[
            (capture_row.slate_id, capture_row.player_id, capture_row.position)
        ].add(capture_row.phase)
        phases_by_slate[capture_row.slate_id].add(capture_row.phase)
        if slate_contract is None:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_SLATE_MISSING",
                    message="selection capture references an unknown slate",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
            continue
        if capture_row.rule_version != slate_contract.rule_version:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_RULE_MISMATCH",
                    message="capture rule version does not match the slate",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
        if capture_row.live_card_version != slate_contract.live_card_version:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_CARD_MISMATCH",
                    message="capture live-card version does not match the slate",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
        if (
            capture_row.position_eligibility_snapshot_id
            != slate_contract.position_eligibility_snapshot_id
        ):
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_ELIGIBILITY_SNAPSHOT_MISMATCH",
                    message="capture eligibility snapshot does not match the slate",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
        if capture_row.captured_at > slate_contract.lock_at:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_AFTER_LOCK",
                    message="selection rate was captured after the slate lock",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
        eligibility_identity = (
            capture_row.position_eligibility_snapshot_id,
            capture_row.slate_id,
            capture_row.live_card_version,
            capture_row.player_id,
            capture_row.position,
        )
        matched_eligibility = eligibility_by_key.get(eligibility_identity)
        if matched_eligibility is None:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_ELIGIBILITY_MISSING",
                    message="capture has no matching player-position eligibility row",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )
        elif not matched_eligibility.is_eligible:
            violations.append(
                DatasetViolation(
                    code="V26_CAPTURE_PLAYER_INELIGIBLE",
                    message="selection rate was captured for an ineligible player-position",
                    game_id=capture_row.slate_id,
                    entity_id=capture_row.capture_row_id,
                )
            )

    for (slate_id, phase), snapshot_ids in sorted(
        snapshots_by_slate_phase.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        if len(snapshot_ids) != 1:
            violations.append(
                DatasetViolation(
                    code="V26_PHASE_SNAPSHOT_ID_INCONSISTENT",
                    message=f"phase {phase.value} uses snapshot IDs {sorted(snapshot_ids)}",
                    game_id=slate_id,
                )
            )
    for (slate_id, phase), captured_times in sorted(
        times_by_slate_phase.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        if len(captured_times) != 1:
            violations.append(
                DatasetViolation(
                    code="V26_PHASE_CAPTURE_TIME_INCONSISTENT",
                    message=f"phase {phase.value} has multiple capture timestamps",
                    game_id=slate_id,
                )
            )
    for snapshot_id, uses in sorted(uses_by_snapshot.items()):
        if len(uses) != 1:
            rendered = sorted((slate_id, phase.value) for slate_id, phase in uses)
            violations.append(
                DatasetViolation(
                    code="V26_SELECTION_SNAPSHOT_REUSED",
                    message=f"snapshot ID is reused across slate/phases: {rendered}",
                    entity_id=snapshot_id,
                )
            )

    for slate_id in sorted(slate_by_id):
        phase_times: list[tuple[V26CapturePhase, datetime]] = []
        for phase in V26_CAPTURE_PHASES:
            times = times_by_slate_phase.get((slate_id, phase), set())
            if len(times) == 1:
                phase_times.append((phase, next(iter(times))))
        for (previous_phase, previous_time), (current_phase, current_time) in zip(
            phase_times,
            phase_times[1:],
            strict=False,
        ):
            if current_time < previous_time:
                violations.append(
                    DatasetViolation(
                        code="V26_CAPTURE_PHASE_ORDER_INVALID",
                        message=(
                            f"{current_phase.value} at {current_time.isoformat()} precedes "
                            f"{previous_phase.value} at {previous_time.isoformat()}"
                        ),
                        game_id=slate_id,
                    )
                )

    if require_all_phases:
        required_phases = set(V26_CAPTURE_PHASES)
        for slate_id in sorted(slate_by_id):
            missing = required_phases - phases_by_slate.get(slate_id, set())
            if missing:
                violations.append(
                    DatasetViolation(
                        code="V26_SLATE_CAPTURE_PHASES_INCOMPLETE",
                        message=(
                            "missing phases: "
                            + ", ".join(sorted(item.value for item in missing))
                        ),
                        game_id=slate_id,
                    )
                )
        eligible_entities = {
            (row.slate_id, row.player_id, row.position)
            for row in eligibility
            if row.is_eligible and row.slate_id in slate_by_id
        }
        for slate_id, player_id, position in sorted(eligible_entities):
            missing = required_phases - phases_by_entity.get(
                (slate_id, player_id, position),
                set(),
            )
            if missing:
                violations.append(
                    DatasetViolation(
                        code="V26_ELIGIBLE_CAPTURE_PHASES_INCOMPLETE",
                        message=(
                            "missing phases: "
                            + ", ".join(sorted(item.value for item in missing))
                        ),
                        game_id=slate_id,
                        entity_id=f"{player_id}:{position}",
                    )
                )

    return DatasetAudit(tuple(violations))


def assert_v26_capture_consistency(
    slates: Iterable[V26SlateRow],
    rules: Iterable[V26RuleVersionRow],
    eligibility_rows: Iterable[V26EligibilityRow],
    captures: Iterable[V26CaptureRow],
    *,
    require_all_phases: bool = True,
) -> None:
    """Raise unless V26 capture horizons and all joined identities are valid."""

    audit_v26_capture_consistency(
        slates,
        rules,
        eligibility_rows,
        captures,
        require_all_phases=require_all_phases,
    ).raise_if_invalid()


class WeatherSourceKind(str, Enum):
    """Whether a weather value was predicted beforehand or observed later."""

    FORECAST = "forecast"
    OBSERVATION = "observation"


class WeatherExperiment(str, Enum):
    """Weather contract attached to a reported evaluation result."""

    FORECAST = "forecast"
    ORACLE_WEATHER = "oracle_weather"


@dataclass(frozen=True, slots=True)
class WeatherFeatureRow:
    """One materialized weather feature with auditable temporal provenance."""

    weather_feature_id: str
    game_id: str
    feature_name: str
    source_kind: WeatherSourceKind
    target_at: datetime
    available_at: datetime
    issued_at: datetime | None = None
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("weather_feature_id", "game_id", "feature_name"):
            _require_text(getattr(self, name), name=name)
        if not isinstance(self.source_kind, WeatherSourceKind):
            raise TypeError("source_kind must be a WeatherSourceKind")
        object.__setattr__(self, "target_at", utc_datetime(self.target_at, field_name="target_at"))
        object.__setattr__(
            self,
            "available_at",
            utc_datetime(self.available_at, field_name="available_at"),
        )
        if self.issued_at is not None:
            object.__setattr__(
                self,
                "issued_at",
                utc_datetime(self.issued_at, field_name="issued_at"),
            )
        if self.observed_at is not None:
            object.__setattr__(
                self,
                "observed_at",
                utc_datetime(self.observed_at, field_name="observed_at"),
            )


def audit_weather_usage(
    rows: Iterable[WeatherFeatureRow],
    *,
    cutoff_by_game: Mapping[str, datetime],
    experiment: WeatherExperiment,
) -> DatasetAudit:
    """Reject forecast leakage and unlabelled use of observed weather.

    ``FORECAST`` experiments accept only forecast revisions that were issued
    and available by the game's prediction cutoff.  ``ORACLE_WEATHER`` accepts
    only post-event observations, making an oracle result impossible to report
    accidentally as an ordinary forecast backtest.
    """

    if not isinstance(experiment, WeatherExperiment):
        raise TypeError("experiment must be a WeatherExperiment")
    features = tuple(rows)
    cutoffs = {
        game_id: utc_datetime(value, field_name=f"cutoff_by_game[{game_id!r}]")
        for game_id, value in cutoff_by_game.items()
    }
    violations: list[DatasetViolation] = []

    duplicate_keys = Counter(
        (row.game_id, row.feature_name, row.target_at) for row in features
    )
    for (game_id, feature_name, target_at), count in sorted(duplicate_keys.items()):
        if count > 1:
            violations.append(
                DatasetViolation(
                    code="DUPLICATE_WEATHER_FEATURE",
                    message=(
                        f"{count} rows select the same feature and target "
                        f"{target_at.isoformat()}"
                    ),
                    game_id=game_id,
                    entity_id=feature_name,
                )
            )

    for row in features:
        cutoff = cutoffs.get(row.game_id)
        if cutoff is None:
            violations.append(
                DatasetViolation(
                    code="WEATHER_CUTOFF_MISSING",
                    message="every weather feature requires its game's prediction cutoff",
                    game_id=row.game_id,
                    entity_id=row.weather_feature_id,
                )
            )
            continue

        if row.source_kind is WeatherSourceKind.FORECAST:
            if experiment is WeatherExperiment.ORACLE_WEATHER:
                violations.append(
                    DatasetViolation(
                        code="FORECAST_IN_ORACLE_DATASET",
                        message="oracle_weather datasets must contain observation rows only",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            if row.issued_at is None:
                violations.append(
                    DatasetViolation(
                        code="FORECAST_ISSUED_AT_MISSING",
                        message="forecast rows require the source revision's issue time",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            else:
                if row.issued_at > row.available_at:
                    violations.append(
                        DatasetViolation(
                            code="FORECAST_AVAILABLE_BEFORE_ISSUE",
                            message="forecast cannot be available before it was issued",
                            game_id=row.game_id,
                            entity_id=row.weather_feature_id,
                        )
                    )
                if row.issued_at > cutoff:
                    violations.append(
                        DatasetViolation(
                            code="FORECAST_ISSUED_AFTER_CUTOFF",
                            message="forecast revision was issued after the prediction cutoff",
                            game_id=row.game_id,
                            entity_id=row.weather_feature_id,
                        )
                    )
                if row.issued_at > row.target_at:
                    violations.append(
                        DatasetViolation(
                            code="FORECAST_ISSUED_AFTER_TARGET",
                            message="forecast issue time cannot be later than its target time",
                            game_id=row.game_id,
                            entity_id=row.weather_feature_id,
                        )
                    )
            if row.observed_at is not None:
                violations.append(
                    DatasetViolation(
                        code="FORECAST_HAS_OBSERVATION_TIME",
                        message="forecast rows cannot also carry an observation timestamp",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            if row.available_at > cutoff:
                violations.append(
                    DatasetViolation(
                        code="FORECAST_AVAILABLE_AFTER_CUTOFF",
                        message="forecast feature was unavailable at prediction time",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
        else:
            if experiment is WeatherExperiment.FORECAST:
                violations.append(
                    DatasetViolation(
                        code="ORACLE_WEATHER_IN_FORECAST_DATASET",
                        message="observed weather is allowed only in an oracle_weather experiment",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            if row.observed_at is None:
                violations.append(
                    DatasetViolation(
                        code="WEATHER_OBSERVED_AT_MISSING",
                        message="observation rows require the physical observation time",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            elif row.observed_at > row.available_at:
                violations.append(
                    DatasetViolation(
                        code="WEATHER_AVAILABLE_BEFORE_OBSERVED",
                        message="weather observation cannot be available before it occurred",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )
            if row.issued_at is not None:
                violations.append(
                    DatasetViolation(
                        code="OBSERVATION_HAS_FORECAST_ISSUE_TIME",
                        message="observation rows cannot carry a forecast issue time",
                        game_id=row.game_id,
                        entity_id=row.weather_feature_id,
                    )
                )

    return DatasetAudit(tuple(violations))


def assert_box_score_consistency(
    plate_appearances: Iterable[PlateAppearanceRow],
    batting_rows: Iterable[PlayerGameBattingRow],
    team_games: Iterable[TeamGameRow],
) -> None:
    """Raise when PA, player-game, and team-game totals disagree."""

    audit_box_score_consistency(plate_appearances, batting_rows, team_games).raise_if_invalid()


def assert_weather_usage(
    rows: Iterable[WeatherFeatureRow],
    *,
    cutoff_by_game: Mapping[str, datetime],
    experiment: WeatherExperiment,
) -> None:
    """Raise when weather provenance violates the selected experiment."""

    audit_weather_usage(
        rows,
        cutoff_by_game=cutoff_by_game,
        experiment=experiment,
    ).raise_if_invalid()


__all__ = [
    "DEFAULT_EXPANDING_TEMPORAL_POLICY",
    "DatasetAudit",
    "DatasetContractError",
    "DatasetViolation",
    "ExpandingTemporalFold",
    "ExpandingTemporalPolicy",
    "ExpandingTemporalSplit",
    "GameGroupAssignment",
    "GameSample",
    "PlateAppearanceRow",
    "PlayerGameBattingRow",
    "RunnerEventMarker",
    "TeamGameRow",
    "TemporalRole",
    "TransitionHalf",
    "TransitionRow",
    "V26_CAPTURE_PHASES",
    "V26CapturePhase",
    "V26CaptureRow",
    "V26EligibilityRow",
    "V26RuleVersionRow",
    "V26SlateRow",
    "WeatherExperiment",
    "WeatherFeatureRow",
    "WeatherSourceKind",
    "assert_box_score_consistency",
    "assert_simulator_ready_transitions",
    "assert_v26_capture_consistency",
    "assert_weather_usage",
    "audit_box_score_consistency",
    "audit_player_game_batting",
    "audit_pa_transitions",
    "audit_team_scores",
    "audit_v26_capture_consistency",
    "audit_weather_usage",
    "build_expanding_temporal_split",
]
