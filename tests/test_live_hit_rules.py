from __future__ import annotations

from itertools import product

import pytest

from cpv26.optimization import (
    LiveHitCandidate,
    LiveHitObjective,
    LiveHitOptimizer,
    LiveHitRuleSet,
    LiveHitSearchDiagnostics,
    RosterSlot,
    SelectionRateBand,
)
from cpv26.simulation import HitScenario


def _account_rules() -> LiveHitRuleSet:
    return LiveHitRuleSet.account(
        rule_version="v26-account-test",
        hit_point_table_source="manual-test-fixture",
        selection_bands_by_position={
            "C": (
                SelectionRateBand(0.0, 0.5, 2.0),
                SelectionRateBand(0.5, 1.0, 1.0),
            ),
            "1B": (
                SelectionRateBand(0.0, 0.3, 2.5),
                SelectionRateBand(0.3, 1.0, 1.0),
            ),
        },
        hit_point_table=(0.0, 10.0, 25.0),
        extra_hit_points=15.0,
        all_hit_reward_id="weekly-all-hit-item",
    )


def test_multi_position_candidate_uses_assigned_position_selection_rate() -> None:
    ruleset = LiveHitRuleSet.account(
        rule_version="position-rate-test",
        hit_point_table_source="manual-test-fixture",
        selection_bands_by_position={
            position: (
                SelectionRateBand(0.0, 0.5, 3.0),
                SelectionRateBand(0.5, 1.0, 1.0),
            )
            for position in ("C", "1B")
        },
    )
    slots = (RosterSlot("catcher", "C"), RosterSlot("first", "1B"))
    candidates = (
        LiveHitCandidate(
            "dual-a",
            "team-x",
            frozenset({"C", "1B"}),
            {"C": 0.1, "1B": 0.9},
        ),
        LiveHitCandidate(
            "dual-b",
            "team-x",
            frozenset({"C", "1B"}),
            {"C": 0.9, "1B": 0.1},
        ),
    )

    recommendation = LiveHitOptimizer().optimize(
        slots,
        candidates,
        (HitScenario({"dual-a": 1, "dual-b": 1}),),
        ruleset=ruleset,
        search_mode="exact",
    )[0]

    assert recommendation.player_by_slot == {"catcher": "dual-a", "first": "dual-b"}
    assert recommendation.synergy_team_id == "team-x"
    assert recommendation.expected_hit_points == pytest.approx(8.0)


def test_fixed_300_is_three_times_base_and_four_times_selected_team() -> None:
    assert LiveHitRuleSet.fixed_300().rule_version == "provisional-fixed-300-v1"
    slots = (RosterSlot("catcher", "C"), RosterSlot("first", "1B"))
    candidates = (
        LiveHitCandidate(
            "catcher-a",
            "team-a",
            frozenset({"C"}),
            collection_owned=True,
        ),
        LiveHitCandidate("first-b", "team-b", frozenset({"1B"})),
    )

    recommendation = LiveHitOptimizer().optimize(
        slots,
        candidates,
        (HitScenario({"catcher-a": 1, "first-b": 2}),),
        ruleset=LiveHitRuleSet.fixed_300(rule_version="fixed-300-test"),
        search_mode="exact",
    )[0]

    assert recommendation.synergy_team_id == "team-b"
    assert recommendation.expected_hit_points == pytest.approx(1 * 3.0 + 2 * 4.0)
    assert recommendation.mode == "fixed_300"


def test_pure_hits_ignores_collection_and_has_no_selected_team() -> None:
    candidate = LiveHitCandidate(
        "catcher",
        "team-a",
        frozenset({"C"}),
        collection_owned=True,
    )

    recommendation = LiveHitOptimizer().optimize(
        (RosterSlot("catcher", "C"),),
        (candidate,),
        (HitScenario({"catcher": 2}),),
        ruleset=LiveHitRuleSet.pure_hits(rule_version="pure-test"),
        search_mode="exact",
    )[0]

    assert recommendation.synergy_team_id is None
    assert recommendation.expected_hit_points == pytest.approx(2.0)


def test_account_mode_collection_is_a_one_times_addition() -> None:
    ruleset = LiveHitRuleSet.account(
        rule_version="account-collection-test",
        hit_point_table_source="manual-test-fixture",
        selection_bands_by_position={
            "C": (SelectionRateBand(0.0, 1.0, 2.0),),
        },
    )
    candidates = (
        LiveHitCandidate(
            "owned",
            "team-a",
            frozenset({"C"}),
            {"C": 0.2},
            collection_owned=True,
        ),
        LiveHitCandidate(
            "zz-unowned",
            "team-a",
            frozenset({"C"}),
            {"C": 0.2},
        ),
    )

    recommendation = LiveHitOptimizer().optimize(
        (RosterSlot("catcher", "C"),),
        candidates,
        (HitScenario({"owned": 1, "zz-unowned": 1}),),
        ruleset=ruleset,
        search_mode="exact",
    )[0]

    assert recommendation.player_by_slot == {"catcher": "owned"}
    assert recommendation.expected_hit_points == pytest.approx(2.0 + 1.0 + 1.0)


def test_all_hit_reward_utility_never_changes_official_hit_points() -> None:
    slots = (RosterSlot("catcher", "C"), RosterSlot("first", "1B"))
    candidates = (
        LiveHitCandidate("catcher", "team-a", frozenset({"C"})),
        LiveHitCandidate("first", "team-b", frozenset({"1B"})),
    )
    scenarios = (
        HitScenario({"catcher": 1, "first": 1}, weight=0.4),
        HitScenario({"catcher": 3, "first": 0}, weight=0.6),
    )
    ruleset = LiveHitRuleSet.pure_hits(
        rule_version="all-hit-separation-test",
        all_hit_reward_id="reward-item-v1",
    )
    optimizer = LiveHitOptimizer()

    official_only = optimizer.optimize(
        slots,
        candidates,
        scenarios,
        ruleset=ruleset,
        search_mode="exact",
    )[0]
    custom = optimizer.optimize(
        slots,
        candidates,
        scenarios,
        ruleset=ruleset,
        objective=LiveHitObjective(risk_aversion=0.5, all_hit_reward_utility=25.0),
        search_mode="exact",
    )[0]

    assert official_only.expected_hit_points == pytest.approx(2.6)
    assert custom.expected_hit_points == pytest.approx(official_only.expected_hit_points)
    assert custom.hit_point_variance == pytest.approx(0.24)
    assert custom.all_hit_probability == pytest.approx(0.4)
    assert custom.all_hit_reward_id == "reward-item-v1"
    assert custom.hit_point_table_source == "provisional-linear-example"
    assert custom.custom_utility == pytest.approx(2.6 - 0.5 * 0.24 + 25.0 * 0.4)


def test_exact_account_mode_matches_independent_manual_oracle() -> None:
    ruleset = _account_rules()
    slots = (RosterSlot("catcher", "C"), RosterSlot("first", "1B"))
    catchers = (
        LiveHitCandidate(
            "catcher-a",
            "team-a",
            frozenset({"C"}),
            {"C": 0.1},
            collection_owned=True,
        ),
        LiveHitCandidate(
            "catcher-b",
            "team-b",
            frozenset({"C"}),
            {"C": 0.7},
        ),
    )
    first_basemen = (
        LiveHitCandidate(
            "first-a",
            "team-a",
            frozenset({"1B"}),
            {"1B": 0.8},
        ),
        LiveHitCandidate(
            "first-b",
            "team-b",
            frozenset({"1B"}),
            {"1B": 0.2},
            collection_owned=True,
        ),
    )
    candidates = (*catchers, *first_basemen)
    scenarios = (
        HitScenario(
            {"catcher-a": 2, "catcher-b": 0, "first-a": 1, "first-b": 0},
            weight=0.2,
        ),
        HitScenario(
            {"catcher-a": 0, "catcher-b": 1, "first-a": 0, "first-b": 2},
            weight=0.5,
        ),
        HitScenario(
            {"catcher-a": 1, "catcher-b": 2, "first-a": 1, "first-b": 1},
            weight=0.3,
        ),
    )
    objective = LiveHitObjective(risk_aversion=0.01, all_hit_reward_utility=5.0)

    def base_points(hits: int) -> float:
        return (0.0, 10.0, 25.0)[hits] if hits <= 2 else 25.0 + 15.0 * (hits - 2)

    def selection_multiplier(position: str, rate: float) -> float:
        if position == "C":
            return 2.0 if rate < 0.5 else 1.0
        return 2.5 if rate < 0.3 else 1.0

    oracle: list[
        tuple[float, float, float, float, tuple[str, str], str]
    ] = []
    for catcher, first_baseman, selected_team in product(
        catchers,
        first_basemen,
        ("team-a", "team-b"),
    ):
        selected = (("C", catcher), ("1B", first_baseman))
        scenario_points: list[float] = []
        all_hit_probability = 0.0
        for scenario in scenarios:
            total = 0.0
            all_hit = True
            for position, candidate in selected:
                hits = scenario.hits_by_player[candidate.player_id]
                multiplier = selection_multiplier(
                    position,
                    candidate.selection_rate_by_position[position],
                )
                multiplier += 1.0 if candidate.collection_owned else 0.0
                multiplier += 1.0 if candidate.team_id == selected_team else 0.0
                total += base_points(hits) * multiplier
                all_hit = all_hit and hits >= 1
            scenario_points.append(total)
            if all_hit:
                all_hit_probability += scenario.weight
        mean = sum(
            scenario.weight * points
            for scenario, points in zip(scenarios, scenario_points, strict=True)
        )
        second = sum(
            scenario.weight * points * points
            for scenario, points in zip(scenarios, scenario_points, strict=True)
        )
        variance = second - mean * mean
        utility = (
            mean
            - objective.risk_aversion * variance
            + objective.all_hit_reward_utility * all_hit_probability
        )
        oracle.append(
            (
                utility,
                mean,
                variance,
                all_hit_probability,
                (catcher.player_id, first_baseman.player_id),
                selected_team,
            )
        )

    expected = max(oracle, key=lambda item: (item[0], item[1], item[3], item[4], item[5]))
    actual = LiveHitOptimizer().optimize(
        slots,
        candidates,
        scenarios,
        ruleset=ruleset,
        objective=objective,
        search_mode="exact",
    )[0]

    assert tuple(candidate.player_id for _, candidate in actual.assignments) == expected[4]
    assert actual.synergy_team_id == expected[5]
    assert actual.custom_utility == pytest.approx(expected[0])
    assert actual.expected_hit_points == pytest.approx(expected[1])
    assert actual.hit_point_variance == pytest.approx(expected[2])
    assert actual.all_hit_probability == pytest.approx(expected[3])
    assert isinstance(actual.diagnostics, LiveHitSearchDiagnostics)
    assert actual.diagnostics.is_exact is True
    assert actual.diagnostics.optimality_gap == 0.0


def test_beam_search_reports_truncation_and_exact_search_finds_joint_optimum() -> None:
    slots = (RosterSlot("a-catcher", "C"), RosterSlot("b-first", "1B"))
    candidates = (
        LiveHitCandidate("a-high-marginal", "team-a", frozenset({"C"})),
        LiveHitCandidate("a-joint", "team-b", frozenset({"C"})),
        LiveHitCandidate("b-low", "team-c", frozenset({"1B"})),
        LiveHitCandidate("b-joint", "team-d", frozenset({"1B"})),
    )
    scenarios = (
        HitScenario({"a-high-marginal": 1, "a-joint": 1, "b-low": 0, "b-joint": 1}),
        HitScenario({"a-high-marginal": 0, "a-joint": 1, "b-low": 0, "b-joint": 1}),
        HitScenario({"a-high-marginal": 1, "a-joint": 0, "b-low": 0, "b-joint": 0}),
        HitScenario({"a-high-marginal": 1, "a-joint": 0, "b-low": 1, "b-joint": 0}),
    )
    objective = LiveHitObjective(all_hit_reward_utility=10.0)
    optimizer = LiveHitOptimizer()

    beam = optimizer.optimize(
        slots,
        candidates,
        scenarios,
        objective=objective,
        beam_width=1,
        search_mode="beam",
    )[0]
    exact = optimizer.optimize(
        slots,
        candidates,
        scenarios,
        objective=objective,
        search_mode="exact",
    )[0]

    assert beam.custom_utility < exact.custom_utility
    assert beam.diagnostics.is_exact is False
    assert beam.diagnostics.beam_pruned_states > 0
    assert beam.diagnostics.optimality_gap is None
    assert exact.diagnostics.is_exact is True
    assert exact.diagnostics.expanded_states > beam.diagnostics.expanded_states
    assert exact.diagnostics.optimality_gap == 0.0


def test_account_mode_rejects_missing_position_selection_rate() -> None:
    ruleset = LiveHitRuleSet.account(
        rule_version="missing-rate-test",
        hit_point_table_source="manual-test-fixture",
        selection_bands_by_position={
            "C": (SelectionRateBand(0.0, 1.0, 2.0),),
        },
    )
    candidate = LiveHitCandidate("catcher", "team-a", frozenset({"C"}))

    with pytest.raises(ValueError, match="lacks a selection rate"):
        LiveHitOptimizer().optimize(
            (RosterSlot("catcher", "C"),),
            (candidate,),
            (HitScenario({"catcher": 1}),),
            ruleset=ruleset,
        )
