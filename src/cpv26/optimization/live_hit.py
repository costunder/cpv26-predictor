"""Versioned V26 Live Hit scoring over joint player-hit scenarios.

Baseball probabilities and game reward rules deliberately meet only in this
module. In particular, an all-hit reward is metadata and a probability; it is
never added to the official hit-point total.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from cpv26.simulation import HitScenario

LiveHitMode = Literal["pure_hits", "fixed_300", "account"]
BonusCombinationRule = Literal["additive_percentage_points"]


@dataclass(frozen=True, slots=True)
class RosterSlot:
    slot_id: str
    position: str

    def __post_init__(self) -> None:
        if not self.slot_id or not self.position:
            raise ValueError("slot_id and position must not be empty")


@dataclass(frozen=True, slots=True)
class SelectionRateBand:
    """A half-open selection-rate interval and its multiplier component.

    ``multiplier=2.0`` represents 200%, not a +200% bonus to which another
    implicit base multiplier should be added. The final band includes 1.0.
    """

    minimum_rate: float
    maximum_rate: float
    multiplier: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.minimum_rate)
            and math.isfinite(self.maximum_rate)
            and 0.0 <= self.minimum_rate < self.maximum_rate <= 1.0
        ):
            raise ValueError("selection-rate bands must satisfy 0 <= minimum < maximum <= 1")
        if not math.isfinite(self.multiplier) or self.multiplier <= 0.0:
            raise ValueError("selection multiplier must be finite and positive")

    def contains(self, selection_rate: float) -> bool:
        return self.minimum_rate <= selection_rate < self.maximum_rate or (
            selection_rate == 1.0 and self.maximum_rate == 1.0
        )


@dataclass(frozen=True, slots=True)
class LiveHitCandidate:
    player_id: str
    team_id: str
    eligible_positions: frozenset[str]
    selection_rate_by_position: Mapping[str, float] = field(default_factory=dict)
    collection_owned: bool = False

    def __post_init__(self) -> None:
        positions = frozenset(self.eligible_positions)
        rates = dict(self.selection_rate_by_position)
        object.__setattr__(self, "eligible_positions", positions)
        object.__setattr__(self, "selection_rate_by_position", MappingProxyType(rates))
        if not self.player_id or not self.team_id:
            raise ValueError("player_id and team_id must not be empty")
        if not positions or any(not position for position in positions):
            raise ValueError("eligible positions must contain non-empty values")
        if not isinstance(self.collection_owned, bool):
            raise TypeError("collection_owned must be a bool")
        unknown_positions = set(rates).difference(positions)
        if unknown_positions:
            raise ValueError("selection rates may only be supplied for eligible positions")
        if any(not position for position in rates):
            raise ValueError("selection-rate positions must not be empty")
        if any(
            not math.isfinite(rate) or not 0.0 <= rate <= 1.0
            for rate in rates.values()
        ):
            raise ValueError("selection rates must be finite values between 0 and 1")


@dataclass(frozen=True, slots=True)
class LiveHitRuleSet:
    """A versioned and auditable Live Hit point contract.

    The only supported combination is additive percentage points. For
    example, selection 2.0 + collection 1.0 + selected-team 1.0 yields 4.0.
    """

    rule_version: str
    hit_point_table_source: str = "provisional-linear-example"
    mode: LiveHitMode = "pure_hits"
    hit_point_table: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    extra_hit_points: float = 1.0
    fixed_multiplier: float | None = 1.0
    selection_bands_by_position: Mapping[str, tuple[SelectionRateBand, ...]] = field(
        default_factory=dict
    )
    collection_multiplier_addition: float = 0.0
    selected_team_multiplier_addition: float = 0.0
    bonus_combination_rule: BonusCombinationRule = "additive_percentage_points"
    all_hit_reward_id: str | None = None

    def __post_init__(self) -> None:
        if not self.rule_version:
            raise ValueError("rule_version must not be empty")
        if not self.hit_point_table_source:
            raise ValueError("hit_point_table_source must not be empty")
        if self.mode not in {"pure_hits", "fixed_300", "account"}:
            raise ValueError("unsupported Live Hit mode")
        hit_points = tuple(self.hit_point_table)
        object.__setattr__(self, "hit_point_table", hit_points)
        if not hit_points or hit_points[0] != 0.0:
            raise ValueError("hit_point_table must start with the zero-hit score")
        if any(not math.isfinite(value) or value < 0.0 for value in hit_points):
            raise ValueError("hit points must be finite and non-negative")
        if any(
            right < left
            for left, right in zip(hit_points, hit_points[1:], strict=False)
        ):
            raise ValueError("hit_point_table must be non-decreasing")
        if not math.isfinite(self.extra_hit_points) or self.extra_hit_points < 0.0:
            raise ValueError("extra_hit_points must be finite and non-negative")
        if self.fixed_multiplier is not None and (
            not math.isfinite(self.fixed_multiplier) or self.fixed_multiplier <= 0.0
        ):
            raise ValueError("fixed_multiplier must be finite and positive")
        for addition in (
            self.collection_multiplier_addition,
            self.selected_team_multiplier_addition,
        ):
            if not math.isfinite(addition) or addition < 0.0:
                raise ValueError("multiplier additions must be finite and non-negative")
        if self.bonus_combination_rule != "additive_percentage_points":
            raise ValueError("unsupported bonus combination rule")
        if self.all_hit_reward_id is not None and not self.all_hit_reward_id:
            raise ValueError("all_hit_reward_id must be non-empty when supplied")

        bands_by_position: dict[str, tuple[SelectionRateBand, ...]] = {}
        for position, configured_bands in self.selection_bands_by_position.items():
            if not position:
                raise ValueError("selection-band positions must not be empty")
            bands = tuple(configured_bands)
            if not bands:
                raise ValueError("each configured position must have at least one band")
            if bands[0].minimum_rate != 0.0 or bands[-1].maximum_rate != 1.0:
                raise ValueError("selection bands must cover the complete [0, 1] interval")
            for left, right in zip(bands, bands[1:], strict=False):
                if not math.isclose(
                    left.maximum_rate,
                    right.minimum_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("selection bands must be ordered, contiguous, and disjoint")
            bands_by_position[position] = bands
        object.__setattr__(
            self,
            "selection_bands_by_position",
            MappingProxyType(bands_by_position),
        )

        if self.mode == "pure_hits":
            if self.fixed_multiplier != 1.0:
                raise ValueError("pure_hits requires fixed_multiplier=1.0")
            if bands_by_position:
                raise ValueError("pure_hits must not configure selection bands")
            if (
                self.collection_multiplier_addition != 0.0
                or self.selected_team_multiplier_addition != 0.0
            ):
                raise ValueError("pure_hits must not configure game bonus additions")
        elif self.mode == "fixed_300":
            if self.fixed_multiplier != 3.0:
                raise ValueError("fixed_300 requires fixed_multiplier=3.0")
            if bands_by_position or self.collection_multiplier_addition != 0.0:
                raise ValueError("fixed_300 ignores selection bands and collection ownership")
            if self.selected_team_multiplier_addition != 1.0:
                raise ValueError("fixed_300 requires a +1.0 selected-team addition")
        else:
            if self.fixed_multiplier is not None:
                raise ValueError("account mode obtains its base multiplier from selection bands")
            if not bands_by_position:
                raise ValueError("account mode requires per-position selection bands")

    @classmethod
    def pure_hits(
        cls,
        *,
        rule_version: str = "provisional-pure-hits-v1",
        hit_point_table: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
        extra_hit_points: float = 1.0,
        hit_point_table_source: str = "provisional-linear-example",
        all_hit_reward_id: str | None = None,
    ) -> LiveHitRuleSet:
        return cls(
            rule_version=rule_version,
            hit_point_table_source=hit_point_table_source,
            mode="pure_hits",
            hit_point_table=hit_point_table,
            extra_hit_points=extra_hit_points,
            fixed_multiplier=1.0,
            all_hit_reward_id=all_hit_reward_id,
        )

    @classmethod
    def fixed_300(
        cls,
        *,
        rule_version: str = "provisional-fixed-300-v1",
        hit_point_table: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
        extra_hit_points: float = 1.0,
        hit_point_table_source: str = "provisional-linear-example",
        all_hit_reward_id: str | None = None,
    ) -> LiveHitRuleSet:
        return cls(
            rule_version=rule_version,
            hit_point_table_source=hit_point_table_source,
            mode="fixed_300",
            hit_point_table=hit_point_table,
            extra_hit_points=extra_hit_points,
            fixed_multiplier=3.0,
            selected_team_multiplier_addition=1.0,
            all_hit_reward_id=all_hit_reward_id,
        )

    @classmethod
    def account(
        cls,
        *,
        rule_version: str,
        selection_bands_by_position: Mapping[str, tuple[SelectionRateBand, ...]],
        hit_point_table_source: str,
        hit_point_table: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
        extra_hit_points: float = 1.0,
        collection_multiplier_addition: float = 1.0,
        selected_team_multiplier_addition: float = 1.0,
        all_hit_reward_id: str | None = None,
    ) -> LiveHitRuleSet:
        return cls(
            rule_version=rule_version,
            hit_point_table_source=hit_point_table_source,
            mode="account",
            hit_point_table=hit_point_table,
            extra_hit_points=extra_hit_points,
            fixed_multiplier=None,
            selection_bands_by_position=selection_bands_by_position,
            collection_multiplier_addition=collection_multiplier_addition,
            selected_team_multiplier_addition=selected_team_multiplier_addition,
            all_hit_reward_id=all_hit_reward_id,
        )

    @property
    def requires_selected_team(self) -> bool:
        return self.mode in {"fixed_300", "account"}

    def hit_points(self, hits: int) -> float:
        if isinstance(hits, bool) or not isinstance(hits, int) or hits < 0:
            raise ValueError("hits must be a non-negative integer")
        if hits < len(self.hit_point_table):
            return self.hit_point_table[hits]
        extra_hits = hits - len(self.hit_point_table) + 1
        return self.hit_point_table[-1] + extra_hits * self.extra_hit_points

    def selection_multiplier(self, position: str, selection_rate: float) -> float:
        bands = self.selection_bands_by_position.get(position)
        if bands is None:
            raise ValueError(f"no selection-rate bands configured for position {position!r}")
        for band in bands:
            if band.contains(selection_rate):
                return band.multiplier
        raise ValueError(
            f"selection rate {selection_rate!r} is outside the bands for position {position!r}"
        )

    def player_multiplier(
        self,
        *,
        candidate: LiveHitCandidate,
        assigned_position: str,
        selected_team_id: str | None,
    ) -> float:
        if self.requires_selected_team and selected_team_id is None:
            raise ValueError(f"{self.mode} requires exactly one selected synergy team")
        if not self.requires_selected_team and selected_team_id is not None:
            raise ValueError("pure_hits does not use a selected synergy team")

        if self.mode in {"pure_hits", "fixed_300"}:
            if self.fixed_multiplier is None:  # pragma: no cover - construction guard
                raise RuntimeError("fixed mode has no fixed multiplier")
            multiplier = self.fixed_multiplier
        else:
            try:
                selection_rate = candidate.selection_rate_by_position[assigned_position]
            except KeyError as error:
                raise ValueError(
                    f"candidate {candidate.player_id!r} lacks a selection rate for "
                    f"position {assigned_position!r}"
                ) from error
            multiplier = self.selection_multiplier(assigned_position, selection_rate)
            if candidate.collection_owned:
                multiplier += self.collection_multiplier_addition

        if selected_team_id is not None and candidate.team_id == selected_team_id:
            multiplier += self.selected_team_multiplier_addition
        return multiplier


@dataclass(frozen=True, slots=True)
class LiveHitObjective:
    risk_aversion: float = 0.0
    all_hit_reward_utility: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and non-negative")
        if (
            not math.isfinite(self.all_hit_reward_utility)
            or self.all_hit_reward_utility < 0.0
        ):
            raise ValueError("all_hit_reward_utility must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LiveHitSearchDiagnostics:
    """Search effort and whether beam truncation could change the optimum."""

    search_mode: Literal["beam", "exact"]
    is_exact: bool
    beam_width: int | None
    expanded_states: int
    pruned_states: int
    beam_pruned_states: int
    completed_rosters: int
    slot_count: int
    candidate_count: int
    eligible_assignment_count: int
    optimality_gap: float | None


@dataclass(frozen=True, slots=True)
class LiveHitRecommendation:
    rule_version: str
    hit_point_table_source: str
    mode: LiveHitMode
    assignments: tuple[tuple[RosterSlot, LiveHitCandidate], ...]
    synergy_team_id: str | None
    expected_hit_points: float
    hit_point_variance: float
    all_hit_probability: float
    all_hit_reward_id: str | None
    custom_utility: float
    expected_hits_by_player: tuple[tuple[str, float], ...]
    diagnostics: LiveHitSearchDiagnostics

    @property
    def player_by_slot(self) -> dict[str, str]:
        return {slot.slot_id: candidate.player_id for slot, candidate in self.assignments}

    @property
    def hit_point_standard_deviation(self) -> float:
        return math.sqrt(self.hit_point_variance)


@dataclass(frozen=True, slots=True)
class _SearchState:
    assignments: tuple[tuple[RosterSlot, LiveHitCandidate], ...]
    used_players: frozenset[str]
    scenario_points: tuple[float, ...]
    all_hit_mask: tuple[bool, ...]
    rank_score: float


class LiveHitOptimizer:
    """Build legal rosters while evaluating correlated Monte Carlo outcomes."""

    def optimize(
        self,
        slots: Sequence[RosterSlot],
        candidates: Sequence[LiveHitCandidate],
        scenarios: Sequence[HitScenario],
        *,
        ruleset: LiveHitRuleSet | None = None,
        objective: LiveHitObjective | None = None,
        synergy_team_ids: Sequence[str] | None = None,
        beam_width: int = 256,
        top_k: int = 1,
        search_mode: Literal["beam", "exact"] = "beam",
    ) -> tuple[LiveHitRecommendation, ...]:
        if not slots:
            raise ValueError("at least one roster slot is required")
        if not candidates:
            raise ValueError("at least one candidate is required")
        if not scenarios:
            raise ValueError("at least one joint hit scenario is required")
        if beam_width < 1 or top_k < 1:
            raise ValueError("beam_width and top_k must be positive")
        if search_mode not in {"beam", "exact"}:
            raise ValueError("search_mode must be 'beam' or 'exact'")

        slot_ids = [slot.slot_id for slot in slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot identifiers must be unique")
        player_ids = [candidate.player_id for candidate in candidates]
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("each player must have exactly one candidate record")

        scoring = ruleset or LiveHitRuleSet.pure_hits()
        target = objective or LiveHitObjective()
        weights = self._normalized_weights(scenarios)
        candidates_by_slot = self._eligible_candidates(slots, candidates)
        ordered_slots = tuple(
            sorted(
                slots,
                key=lambda slot: (len(candidates_by_slot[slot.slot_id]), slot.slot_id),
            )
        )
        if not self._can_complete(ordered_slots, candidates_by_slot, frozenset()):
            raise ValueError("position eligibility cannot form a complete roster")
        self._validate_account_inputs(scoring, candidates_by_slot, slots)

        selected_team_choices = self._selected_team_choices(
            scoring,
            candidates,
            synergy_team_ids,
        )
        expected_hits = self._expected_hits(candidates, scenarios, weights)
        slot_order = {slot.slot_id: index for index, slot in enumerate(slots)}

        recommendations: list[LiveHitRecommendation] = []
        for selected_team_id in selected_team_choices:
            recommendations.extend(
                self._search_for_selected_team(
                    ordered_slots=ordered_slots,
                    original_slots=slots,
                    candidates=candidates,
                    candidates_by_slot=candidates_by_slot,
                    scenarios=scenarios,
                    weights=weights,
                    ruleset=scoring,
                    objective=target,
                    selected_team_id=selected_team_id,
                    expected_hits=expected_hits,
                    slot_order=slot_order,
                    beam_width=beam_width,
                    search_mode=search_mode,
                )
            )

        recommendations.sort(
            key=lambda recommendation: (
                recommendation.custom_utility,
                recommendation.expected_hit_points,
                recommendation.all_hit_probability,
                tuple(candidate.player_id for _, candidate in recommendation.assignments),
                recommendation.synergy_team_id or "",
            ),
            reverse=True,
        )
        return tuple(recommendations[:top_k])

    @staticmethod
    def _eligible_candidates(
        slots: Sequence[RosterSlot],
        candidates: Sequence[LiveHitCandidate],
    ) -> dict[str, tuple[LiveHitCandidate, ...]]:
        candidates_by_slot: dict[str, tuple[LiveHitCandidate, ...]] = {}
        for slot in slots:
            eligible = tuple(
                candidate
                for candidate in candidates
                if slot.position in candidate.eligible_positions
            )
            if not eligible:
                raise ValueError(f"slot {slot.slot_id} has no eligible candidate")
            candidates_by_slot[slot.slot_id] = eligible
        return candidates_by_slot

    @staticmethod
    def _validate_account_inputs(
        ruleset: LiveHitRuleSet,
        candidates_by_slot: Mapping[str, tuple[LiveHitCandidate, ...]],
        slots: Sequence[RosterSlot],
    ) -> None:
        if ruleset.mode != "account":
            return
        for slot in slots:
            if slot.position not in ruleset.selection_bands_by_position:
                raise ValueError(
                    f"account ruleset has no selection bands for position {slot.position!r}"
                )
            for candidate in candidates_by_slot[slot.slot_id]:
                if slot.position not in candidate.selection_rate_by_position:
                    raise ValueError(
                        f"candidate {candidate.player_id!r} lacks a selection rate for "
                        f"eligible position {slot.position!r}"
                    )

    @staticmethod
    def _selected_team_choices(
        ruleset: LiveHitRuleSet,
        candidates: Sequence[LiveHitCandidate],
        synergy_team_ids: Sequence[str] | None,
    ) -> tuple[str | None, ...]:
        if not ruleset.requires_selected_team:
            if synergy_team_ids:
                raise ValueError("pure_hits does not accept synergy_team_ids")
            return (None,)

        if synergy_team_ids is None:
            team_ids = tuple(sorted({candidate.team_id for candidate in candidates}))
        else:
            team_ids = tuple(synergy_team_ids)
            if not team_ids:
                raise ValueError(f"{ruleset.mode} requires at least one selectable synergy team")
            if any(not team_id for team_id in team_ids):
                raise ValueError("synergy team identifiers must not be empty")
            if len(set(team_ids)) != len(team_ids):
                raise ValueError("synergy team identifiers must be unique")
        return team_ids

    @classmethod
    def _search_for_selected_team(
        cls,
        *,
        ordered_slots: Sequence[RosterSlot],
        original_slots: Sequence[RosterSlot],
        candidates: Sequence[LiveHitCandidate],
        candidates_by_slot: Mapping[str, tuple[LiveHitCandidate, ...]],
        scenarios: Sequence[HitScenario],
        weights: tuple[float, ...],
        ruleset: LiveHitRuleSet,
        objective: LiveHitObjective,
        selected_team_id: str | None,
        expected_hits: Mapping[str, float],
        slot_order: Mapping[str, int],
        beam_width: int,
        search_mode: Literal["beam", "exact"],
    ) -> list[LiveHitRecommendation]:
        points_by_assignment: dict[tuple[str, str], tuple[float, ...]] = {}
        hit_masks: dict[str, tuple[bool, ...]] = {}
        for candidate in candidates:
            hit_counts = tuple(
                scenario.hits_by_player.get(candidate.player_id, 0) for scenario in scenarios
            )
            hit_masks[candidate.player_id] = tuple(hits >= 1 for hits in hit_counts)
        for slot in ordered_slots:
            for candidate in candidates_by_slot[slot.slot_id]:
                multiplier = ruleset.player_multiplier(
                    candidate=candidate,
                    assigned_position=slot.position,
                    selected_team_id=selected_team_id,
                )
                points_by_assignment[(slot.slot_id, candidate.player_id)] = tuple(
                    ruleset.hit_points(scenario.hits_by_player.get(candidate.player_id, 0))
                    * multiplier
                    for scenario in scenarios
                )

        scenario_count = len(scenarios)
        frontier = [
            _SearchState(
                assignments=(),
                used_players=frozenset(),
                scenario_points=(0.0,) * scenario_count,
                all_hit_mask=(True,) * scenario_count,
                rank_score=0.0,
            )
        ]
        expanded_states = 0
        pruned_states = 0
        beam_pruned_states = 0

        for slot_index, slot in enumerate(ordered_slots):
            expanded: list[_SearchState] = []
            remaining_slots = ordered_slots[slot_index + 1 :]
            for state in frontier:
                for candidate in candidates_by_slot[slot.slot_id]:
                    if candidate.player_id in state.used_players:
                        pruned_states += 1
                        continue
                    used = state.used_players | {candidate.player_id}
                    if not cls._can_complete(remaining_slots, candidates_by_slot, used):
                        pruned_states += 1
                        continue
                    points = tuple(
                        left + right
                        for left, right in zip(
                            state.scenario_points,
                            points_by_assignment[(slot.slot_id, candidate.player_id)],
                            strict=True,
                        )
                    )
                    all_hit = tuple(
                        left and right
                        for left, right in zip(
                            state.all_hit_mask,
                            hit_masks[candidate.player_id],
                            strict=True,
                        )
                    )
                    assignments = state.assignments + ((slot, candidate),)
                    rank = cls._partial_rank(
                        points,
                        all_hit,
                        weights,
                        completion=len(assignments) / len(ordered_slots),
                        objective=objective,
                    )
                    expanded.append(_SearchState(assignments, used, points, all_hit, rank))
                    expanded_states += 1

            if not expanded:
                raise ValueError("no legal roster survives the position constraints")
            expanded.sort(
                key=lambda state: (
                    state.rank_score,
                    tuple(candidate.player_id for _, candidate in state.assignments),
                ),
                reverse=True,
            )
            if search_mode == "beam" and len(expanded) > beam_width:
                removed = len(expanded) - beam_width
                pruned_states += removed
                beam_pruned_states += removed
                frontier = expanded[:beam_width]
            else:
                frontier = expanded

        diagnostics = LiveHitSearchDiagnostics(
            search_mode=search_mode,
            is_exact=search_mode == "exact" or beam_pruned_states == 0,
            beam_width=beam_width if search_mode == "beam" else None,
            expanded_states=expanded_states,
            pruned_states=pruned_states,
            beam_pruned_states=beam_pruned_states,
            completed_rosters=len(frontier),
            slot_count=len(original_slots),
            candidate_count=len(candidates),
            eligible_assignment_count=sum(len(values) for values in candidates_by_slot.values()),
            optimality_gap=0.0 if search_mode == "exact" or beam_pruned_states == 0 else None,
        )
        return [
            cls._evaluate_complete_roster(
                state,
                selected_team_id=selected_team_id,
                weights=weights,
                expected_hits=expected_hits,
                ruleset=ruleset,
                objective=objective,
                slot_order=slot_order,
                diagnostics=diagnostics,
            )
            for state in frontier
        ]

    @staticmethod
    def _normalized_weights(scenarios: Sequence[HitScenario]) -> tuple[float, ...]:
        total = sum(scenario.weight for scenario in scenarios)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("scenario weights must have a positive finite sum")
        return tuple(scenario.weight / total for scenario in scenarios)

    @staticmethod
    def _expected_hits(
        candidates: Sequence[LiveHitCandidate],
        scenarios: Sequence[HitScenario],
        weights: tuple[float, ...],
    ) -> dict[str, float]:
        return {
            candidate.player_id: sum(
                weight * scenario.hits_by_player.get(candidate.player_id, 0)
                for weight, scenario in zip(weights, scenarios, strict=True)
            )
            for candidate in candidates
        }

    @classmethod
    def _partial_rank(
        cls,
        points: tuple[float, ...],
        all_hit: tuple[bool, ...],
        weights: tuple[float, ...],
        *,
        completion: float,
        objective: LiveHitObjective,
    ) -> float:
        mean, variance = cls._weighted_moments(points, weights)
        all_hit_probability = sum(
            weight for weight, success in zip(weights, all_hit, strict=True) if success
        )
        return (
            mean
            - objective.risk_aversion * variance * completion
            + objective.all_hit_reward_utility * all_hit_probability * completion
        )

    @classmethod
    def _evaluate_complete_roster(
        cls,
        state: _SearchState,
        *,
        selected_team_id: str | None,
        weights: tuple[float, ...],
        expected_hits: Mapping[str, float],
        ruleset: LiveHitRuleSet,
        objective: LiveHitObjective,
        slot_order: Mapping[str, int],
        diagnostics: LiveHitSearchDiagnostics,
    ) -> LiveHitRecommendation:
        expected_hit_points, variance = cls._weighted_moments(state.scenario_points, weights)
        all_hit_probability = sum(
            weight
            for weight, success in zip(weights, state.all_hit_mask, strict=True)
            if success
        )
        custom_utility = (
            expected_hit_points
            - objective.risk_aversion * variance
            + objective.all_hit_reward_utility * all_hit_probability
        )
        ordered_assignments = tuple(
            sorted(
                state.assignments,
                key=lambda assignment: slot_order[assignment[0].slot_id],
            )
        )
        return LiveHitRecommendation(
            rule_version=ruleset.rule_version,
            hit_point_table_source=ruleset.hit_point_table_source,
            mode=ruleset.mode,
            assignments=ordered_assignments,
            synergy_team_id=selected_team_id,
            expected_hit_points=expected_hit_points,
            hit_point_variance=variance,
            all_hit_probability=all_hit_probability,
            all_hit_reward_id=ruleset.all_hit_reward_id,
            custom_utility=custom_utility,
            expected_hits_by_player=tuple(
                (candidate.player_id, expected_hits[candidate.player_id])
                for _, candidate in ordered_assignments
            ),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _weighted_moments(
        values: Iterable[float],
        weights: tuple[float, ...],
    ) -> tuple[float, float]:
        value_tuple = tuple(values)
        mean = sum(weight * value for weight, value in zip(weights, value_tuple, strict=True))
        second = sum(
            weight * value * value for weight, value in zip(weights, value_tuple, strict=True)
        )
        return mean, max(0.0, second - mean * mean)

    @staticmethod
    def _can_complete(
        remaining_slots: Sequence[RosterSlot],
        candidates_by_slot: Mapping[str, tuple[LiveHitCandidate, ...]],
        used_players: frozenset[str],
    ) -> bool:
        """Exact bipartite matching feasibility check for remaining positions."""

        if not remaining_slots:
            return True
        owner_by_player: dict[str, str] = {}

        def assign(slot: RosterSlot, visited: set[str]) -> bool:
            for candidate in candidates_by_slot[slot.slot_id]:
                player_id = candidate.player_id
                if player_id in used_players or player_id in visited:
                    continue
                visited.add(player_id)
                owner = owner_by_player.get(player_id)
                if owner is None:
                    owner_by_player[player_id] = slot.slot_id
                    return True
                previous_slot = next(item for item in remaining_slots if item.slot_id == owner)
                if assign(previous_slot, visited):
                    owner_by_player[player_id] = slot.slot_id
                    return True
            return False

        return all(assign(slot, set()) for slot in remaining_slots)
