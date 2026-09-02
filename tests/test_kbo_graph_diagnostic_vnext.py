from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

import cpv26.training.kbo_graph_diagnostic as diagnostic  # noqa: E402
from cpv26.graph import TorchAtomicRouteBatch  # noqa: E402
from cpv26.models.kbo_relgnn import KBO_ROUTE_NAMES  # noqa: E402
from cpv26.training.kbo_graph_diagnostic import (  # noqa: E402
    KBOGraphTransformSpec,
    transform_kbo_graph_batch,
)


def _route(
    name: str,
    source_type: str,
    destination_type: str,
    source: list[int],
    destination: list[int],
) -> TorchAtomicRouteBatch:
    count = len(source)
    return TorchAtomicRouteBatch(
        route_name=name,
        source_type=source_type,
        destination_type=destination_type,
        source_index=torch.tensor(source, dtype=torch.long),
        destination_index=torch.tensor(destination, dtype=torch.long),
        event_features=torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
        event_age_seconds=torch.arange(1, count + 1, dtype=torch.float32) * 86_400,
        publication_delay_seconds=torch.zeros(count, dtype=torch.float32),
        weights=torch.ones(count, dtype=torch.float32),
        bidirectional=True,
    )


def _vnext_batch() -> dict[str, Any]:
    return {
        "_validated_on_cpu": True,
        "day_ids": ("2025-04-01", "2025-04-02"),
        "node_graph_index": {
            "player": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            "team": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "game": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        },
        "routes": (
            _route(
                "batter_game_participation",
                "player",
                "game",
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ),
            _route(
                "team_game_context",
                "team",
                "game",
                [0, 1, 2, 3],
                [0, 1, 3, 4],
            ),
        ),
    }


def _condition_names(route_dims: dict[str, int]) -> tuple[str, ...]:
    state = {"model_config": {"route_feature_dims": route_dims}}
    route_names = diagnostic._checkpoint_route_names(state)
    return tuple(
        name for name, _ in diagnostic._condition_specs(17, False, route_names)
    )


def test_conditions_use_the_checkpoint_route_contract_not_the_global_union() -> None:
    legacy_dims = dict.fromkeys(KBO_ROUTE_NAMES, 6)
    legacy_conditions = _condition_names(legacy_dims)
    assert legacy_conditions == (
        "intact",
        "no_routes",
        "permuted_endpoints",
        *(f"without_{name}" for name in KBO_ROUTE_NAMES),
    )
    assert "without_batter_game_participation" not in legacy_conditions

    checkpoint_routes = (
        "batter_pa_pitcher",
        "batter_game_participation",
        "team_game_context",
    )
    vnext_conditions = _condition_names(dict.fromkeys(checkpoint_routes, 6))
    assert vnext_conditions == (
        "intact",
        "no_routes",
        "permuted_endpoints",
        *(f"without_{name}" for name in checkpoint_routes),
    )
    # A pre-route-contract checkpoint retains the historical four conditions.
    assert diagnostic._checkpoint_route_names({"model_config": {}}) == KBO_ROUTE_NAMES


def test_vnext_route_knockout_uses_its_explicit_checkpoint_contract() -> None:
    batch = _vnext_batch()
    route_names = tuple(route.route_name for route in batch["routes"])
    spec = KBOGraphTransformSpec(
        "route_knockout",
        seed=11,
        route_name="team_game_context",
        reviewed_route_names=route_names,
    )

    transformed, audit = transform_kbo_graph_batch(batch, spec)

    assert transformed["routes"][0] is batch["routes"][0]
    assert transformed["routes"][1].num_edges == 0
    assert audit["per_route"]["team_game_context"]["edges_removed"] == 4


def test_game_endpoint_permutation_is_degree_preserving_and_day_local() -> None:
    batch = _vnext_batch()
    originals = tuple(
        (route.source_index.clone(), route.destination_index.clone())
        for route in batch["routes"]
    )

    transformed, audit = transform_kbo_graph_batch(
        batch, KBOGraphTransformSpec("permute_endpoints", seed=29)
    )

    assert audit["source_endpoints_changed"] > 0
    assert audit["destination_endpoints_changed"] > 0
    memberships = batch["node_graph_index"]
    for original, changed in zip(batch["routes"], transformed["routes"], strict=True):
        original_days = memberships[original.source_type][original.source_index]
        changed_source_days = memberships[changed.source_type][changed.source_index]
        changed_destination_days = memberships[changed.destination_type][
            changed.destination_index
        ]
        assert torch.equal(changed_source_days, original_days)
        assert torch.equal(changed_destination_days, original_days)
        assert changed.destination_type == "game"

        for field, node_type in (
            ("source_index", changed.source_type),
            ("destination_index", changed.destination_type),
        ):
            membership = memberships[node_type]
            before = getattr(original, field)
            after = getattr(changed, field)
            for graph_index in (0, 1):
                nodes = torch.nonzero(membership == graph_index, as_tuple=False).flatten()
                before_degree = torch.bincount(
                    before[original_days == graph_index], minlength=len(membership)
                )[nodes].sort().values
                after_degree = torch.bincount(
                    after[original_days == graph_index], minlength=len(membership)
                )[nodes].sort().values
                assert torch.equal(before_degree, after_degree)

    for route, (source, destination) in zip(batch["routes"], originals, strict=True):
        assert torch.equal(route.source_index, source)
        assert torch.equal(route.destination_index, destination)
