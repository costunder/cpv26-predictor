from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cpv26.models.kbo_relgnn import (  # noqa: E402
    KBO_ROUTE_NAMES,
    KBORelGNNConfig,
    KBORelGNNModel,
    collate_kbo_day_graphs,
)
from cpv26.training.kbo_graph_diagnostic import (  # noqa: E402
    KBOGraphBatchTransform,
    KBOGraphTransformSpec,
    RelGNNDiagnosticsCollector,
    paired_prediction_sensitivity,
    recursive_numeric_metric_deltas,
    transform_kbo_graph_batch,
)


def _day(day_id: str = "2025-04-01", seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    endpoints = {
        "batter_pa_pitcher": ([0, 1, 2, 3], [1, 2, 3, 0]),
        "batter_participation_team": ([0, 1, 2, 3], [0, 1, 2, 0]),
        "pitcher_participation_team": ([3, 2, 1, 0], [2, 1, 0, 2]),
        "home_team_game_away_team": ([0, 1, 2], [1, 2, 0]),
    }
    routes: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for route_number, (name, (source, destination)) in enumerate(endpoints.items()):
        count = len(source)
        event_features = np.arange(count * 6, dtype=np.float32).reshape(count, 6)
        event_features += seed * 1000 + route_number * 100
        routes[name] = {
            "source_index": np.asarray(source, dtype=np.int64),
            "destination_index": np.asarray(destination, dtype=np.int64),
            "event_features": event_features,
            "event_age_seconds": np.arange(1, count + 1, dtype=np.float32) * 86_400,
            "publication_delay_seconds": np.arange(1, count + 1, dtype=np.float32) * 600,
            "weights": np.linspace(0.25, 1.0, count, dtype=np.float32),
        }
    return {
        "day_id": day_id,
        "node_features": {
            "player": rng.normal(size=(4, 4)).astype(np.float32),
            "team": rng.normal(size=(3, 8)).astype(np.float32),
        },
        "role_features": {
            "batting": rng.normal(size=(4, 8)).astype(np.float32),
            "pitching": rng.normal(size=(4, 8)).astype(np.float32),
        },
        "routes": routes,
        "match_home_team_index": np.asarray([0], dtype=np.int64),
        "match_away_team_index": np.asarray([1], dtype=np.int64),
        "match_targets": np.asarray([2], dtype=np.int64),
        "match_runs": np.asarray([[5, 3]], dtype=np.float32),
        "match_query_ids": (f"{day_id}:game",),
        "live_hit_player_index": np.asarray([0, 2], dtype=np.int64),
        "live_hit_team_index": np.asarray([0, 1], dtype=np.int64),
        "live_hit_opponent_index": np.asarray([1, 2], dtype=np.int64),
        "live_hit_pa": np.asarray([4, 3], dtype=np.int64),
        "live_hit_hits": np.asarray([2, 0], dtype=np.int64),
        "live_hit_query_ids": (f"{day_id}:hit:0", f"{day_id}:hit:2"),
        "pa_batter_index": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "pa_pitcher_index": np.asarray([1, 2, 3, 0], dtype=np.int64),
        "pa_targets": np.asarray([2, 0, 6, 1], dtype=np.int64),
        "pa_context": rng.normal(size=(4, 10)).astype(np.float32),
        "pa_query_ids": tuple(f"{day_id}:pa:{index}" for index in range(4)),
    }


def _route(batch: dict[str, Any], name: str) -> Any:
    return next(route for route in batch["routes"] if route.route_name == name)


def _snapshot(batch: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        route.route_name: tuple(
            getattr(route, field).clone()
            for field in (
                "source_index",
                "destination_index",
                "event_features",
                "event_age_seconds",
                "publication_delay_seconds",
                "weights",
            )
        )
        for route in batch["routes"]
    }


def _local_endpoints(
    batch: dict[str, Any], route_name: str, day_id: str
) -> tuple[list[int], list[int]]:
    route = _route(batch, route_name)
    graph_index = batch["day_ids"].index(day_id)
    source_membership = batch["node_graph_index"][route.source_type]
    destination_membership = batch["node_graph_index"][route.destination_type]
    edge_graph = source_membership[route.source_index]
    selected = edge_graph == graph_index
    source_nodes = torch.nonzero(source_membership == graph_index).flatten().tolist()
    destination_nodes = torch.nonzero(destination_membership == graph_index).flatten().tolist()
    source_lookup = {node: index for index, node in enumerate(source_nodes)}
    destination_lookup = {node: index for index, node in enumerate(destination_nodes)}
    return (
        [source_lookup[int(value)] for value in route.source_index[selected].tolist()],
        [destination_lookup[int(value)] for value in route.destination_index[selected].tolist()],
    )


@pytest.mark.parametrize("mode", ["intact", "no_routes"])
def test_intact_and_no_routes_are_immutable_and_schema_preserving(mode: str) -> None:
    batch = collate_kbo_day_graphs([_day()])
    before = _snapshot(batch)
    transformed, audit = transform_kbo_graph_batch(batch, KBOGraphTransformSpec(mode))
    assert transformed is not batch
    assert tuple(route.route_name for route in transformed["routes"]) == KBO_ROUTE_NAMES
    for route in transformed["routes"]:
        expected = _route(batch, route.route_name)
        if mode == "no_routes":
            assert route.num_edges == 0
            assert route.event_features.shape == (0, expected.event_features.shape[1])
            assert route.event_features.dtype == expected.event_features.dtype
        else:
            assert route is expected
    assert audit["edges_removed"] == (
        sum(route.num_edges for route in batch["routes"]) if mode == "no_routes" else 0
    )
    assert audit["no_op"] is (mode == "intact")
    for route_name, tensors in before.items():
        assert all(
            torch.equal(value, actual)
            for value, actual in zip(tensors, _snapshot(batch)[route_name], strict=True)
        )


@pytest.mark.parametrize("route_name", KBO_ROUTE_NAMES)
def test_route_knockout_only_zeros_the_requested_route(route_name: str) -> None:
    batch = collate_kbo_day_graphs([_day()])
    transformed, audit = transform_kbo_graph_batch(
        batch, KBOGraphTransformSpec("route_knockout", route_name=route_name)
    )
    for route in transformed["routes"]:
        original = _route(batch, route.route_name)
        if route.route_name == route_name:
            assert route.num_edges == 0
            assert audit["per_route"][route_name]["edges_removed"] == original.num_edges
        else:
            for field in (
                "source_index",
                "destination_index",
                "event_features",
                "event_age_seconds",
                "publication_delay_seconds",
                "weights",
            ):
                assert torch.equal(getattr(route, field), getattr(original, field))
    with pytest.raises(ValueError, match="reviewed KBO route"):
        KBOGraphTransformSpec("route_knockout", route_name="invented")


def test_endpoint_permutation_is_day_local_degree_preserving_and_column_preserving() -> None:
    batch = collate_kbo_day_graphs([_day(), _day("2025-04-02", 2)])
    transformed, audit = transform_kbo_graph_batch(
        batch, KBOGraphTransformSpec("permute_endpoints", seed=91)
    )
    assert audit["source_endpoints_changed"] > 0
    assert audit["destination_endpoints_changed"] > 0
    for original, changed in zip(batch["routes"], transformed["routes"], strict=True):
        source_graph = batch["node_graph_index"][changed.source_type][changed.source_index]
        destination_graph = batch["node_graph_index"][changed.destination_type][
            changed.destination_index
        ]
        assert torch.equal(source_graph, destination_graph)
        for field in (
            "event_features",
            "event_age_seconds",
            "publication_delay_seconds",
            "weights",
        ):
            assert torch.equal(getattr(original, field), getattr(changed, field))
        for field, node_type in (
            ("source_index", changed.source_type),
            ("destination_index", changed.destination_type),
        ):
            node_count = len(batch["node_graph_index"][node_type])
            before_degree = (
                torch.bincount(getattr(original, field), minlength=node_count).sort().values
            )
            after_degree = (
                torch.bincount(getattr(changed, field), minlength=node_count).sort().values
            )
            assert torch.equal(before_degree, after_degree)
    assert transformed["match_query_ids"] == batch["match_query_ids"]
    assert torch.equal(transformed["match_targets"], batch["match_targets"])


def test_endpoint_permutation_is_independent_of_batch_order() -> None:
    first, second = _day(), _day("2025-04-02", 2)
    single, _ = transform_kbo_graph_batch(
        collate_kbo_day_graphs([first]), KBOGraphTransformSpec("permute_endpoints", seed=7)
    )
    union, _ = transform_kbo_graph_batch(
        collate_kbo_day_graphs([second, first]),
        KBOGraphTransformSpec("permute_endpoints", seed=7),
    )
    for route_name in KBO_ROUTE_NAMES:
        assert _local_endpoints(single, route_name, first["day_id"]) == _local_endpoints(
            union, route_name, first["day_id"]
        )


def test_edge_attribute_permutation_keeps_endpoints_and_joint_attribute_rows() -> None:
    batch = collate_kbo_day_graphs([_day()])
    transformed, audit = transform_kbo_graph_batch(
        batch, KBOGraphTransformSpec("permute_edge_attributes", seed=13)
    )
    assert audit["edge_attribute_rows_permuted"] > 0
    for original, changed in zip(batch["routes"], transformed["routes"], strict=True):
        assert torch.equal(original.source_index, changed.source_index)
        assert torch.equal(original.destination_index, changed.destination_index)
        before_rows = sorted(
            zip(
                original.event_features.tolist(),
                original.event_age_seconds.tolist(),
                original.publication_delay_seconds.tolist(),
                original.weights.tolist(),
                strict=True,
            )
        )
        after_rows = sorted(
            zip(
                changed.event_features.tolist(),
                changed.event_age_seconds.tolist(),
                changed.publication_delay_seconds.tolist(),
                changed.weights.tolist(),
                strict=True,
            )
        )
        assert before_rows == after_rows


def test_transform_rejects_a_cross_day_edge_before_randomising() -> None:
    batch = collate_kbo_day_graphs([_day(), _day("2025-04-02", 2)])
    route = batch["routes"][0]
    destination = route.destination_index.clone()
    destination[0] = 4  # First player node of the second disjoint graph.
    bad = dict(batch)
    bad["routes"] = (replace(route, destination_index=destination), *batch["routes"][1:])
    with pytest.raises(ValueError, match="cross-day"):
        transform_kbo_graph_batch(bad, KBOGraphTransformSpec("permute_endpoints"))


def test_callable_transform_accumulates_audits_without_mutating_batches() -> None:
    transform = KBOGraphBatchTransform(KBOGraphTransformSpec("no_routes"))
    batches = [collate_kbo_day_graphs([_day()]), collate_kbo_day_graphs([_day("2025-04-02", 2)])]
    snapshots = [_snapshot(batch) for batch in batches]
    for batch in batches:
        transform(batch)
    report = transform.report()
    assert report["batches"] == 2 and report["days"] == 2
    assert report["edges_after"] == 0 and report["edges_removed"] > 0
    for batch, snapshot in zip(batches, snapshots, strict=True):
        for route_name, tensors in snapshot.items():
            assert all(
                torch.equal(value, actual)
                for value, actual in zip(tensors, _snapshot(batch)[route_name], strict=True)
            )


def test_prediction_sensitivity_joins_by_query_id_and_reports_distribution_changes() -> None:
    reference = {
        "match": [
            {"query_id": "q1", "label": 0, "probability_0": 0.8, "probability_1": 0.2},
            {"query_id": "q2", "label": 1, "probability_0": 0.1, "probability_1": 0.9},
        ]
    }
    candidate = {
        "match": [
            {"query_id": "q2", "label": 1, "probability_0": 0.6, "probability_1": 0.4},
            {"query_id": "q1", "label": 0, "probability_0": 0.6, "probability_1": 0.4},
        ]
    }
    result = paired_prediction_sensitivity(reference, candidate)["match"]
    assert result["mean_total_variation"] == pytest.approx(0.35)
    assert result["median_total_variation"] == pytest.approx(0.35)
    assert result["p95_total_variation"] == pytest.approx(0.485)
    assert result["max_absolute_probability_change"] == pytest.approx(0.5)
    assert result["argmax_flip_count"] == 1
    assert result["argmax_flip_rate"] == pytest.approx(0.5)
    duplicate = {"match": [reference["match"][0], reference["match"][0]]}
    with pytest.raises(ValueError, match="duplicate"):
        paired_prediction_sensitivity(duplicate, candidate)
    with pytest.raises(ValueError, match="populations differ"):
        paired_prediction_sensitivity(reference, {"match": candidate["match"][:1]})


def test_recursive_metric_delta_is_candidate_minus_intact_and_keeps_nulls() -> None:
    reference = {
        "selection_loss": 4.0,
        "match": {"accuracy": 0.60, "log_loss": 0.8},
        "box_pa": None,
        "label": "unchanged",
    }
    candidate = {
        "selection_loss": 4.25,
        "match": {"accuracy": 0.55, "log_loss": 0.9},
        "box_pa": None,
        "label": "different text is not a numeric metric",
    }
    delta = recursive_numeric_metric_deltas(reference, candidate)
    assert delta["selection_loss"] == pytest.approx(0.25)
    assert delta["match"] == {"accuracy": pytest.approx(-0.05), "log_loss": pytest.approx(0.1)}
    assert delta["box_pa"] is None
    assert "label" not in delta


def test_collector_separates_forced_and_competitive_attention_gates_and_updates() -> None:
    batch = collate_kbo_day_graphs([_day()])
    collector = RelGNNDiagnosticsCollector()
    collector.begin_batch(batch)
    destination_index = torch.tensor([0, 1, 1])
    collector.observe_attention(
        layer_index=0,
        route_name="batter_pa_pitcher",
        direction="forward",
        source_channel="player__batting",
        destination_channel="player__pitching",
        source_index=torch.tensor([0, 1, 2]),
        destination_index=destination_index,
        positive_weight=torch.ones(3, dtype=torch.bool),
        attention=torch.tensor([[1.0, 1.0], [0.25, 0.40], [0.75, 0.60]]),
        message=torch.ones((3, 4)),
        route_mask=torch.tensor([True, True, False]),
        destination_state=torch.full((3, 4), 2.0),
    )
    previous = torch.ones((3, 4))
    candidate = previous + 0.5
    masks = torch.tensor([[True, False], [True, True], [False, False]])
    collector.observe_gates(
        layer_index=0,
        destination_channel="team",
        route_names=("route_a", "route_b"),
        directions=("forward", "reverse"),
        source_channels=("player__batting", "team"),
        gate_keys=("route_a__forward", "route_b__reverse"),
        messages=torch.ones((3, 2, 4)),
        masks=masks,
        route_attention=torch.tensor([[1.0, 0.0], [0.25, 0.75], [0.0, 0.0]]),
        previous_state=previous,
        combined_message=torch.tensor([[1.0] * 4, [1.0] * 4, [0.0] * 4]),
        candidate_state=candidate,
        updated_state=torch.where(masks.any(dim=1, keepdim=True), candidate, previous),
    )
    report = collector.report()
    attention = report["attention"]["by_layer_route_direction"]["layer_0|batter_pa_pitcher|forward"]
    assert attention["forced_singleton_destinations"] == 1
    assert attention["competitive_destinations"] == 1
    assert attention["forced_singleton_attention"]["mean"] == pytest.approx(1.0)
    assert 0 < attention["competitive_normalized_entropy"]["mean"] <= 1
    gates = report["route_gates"]["by_layer_channel"]["layer_0|team"]
    assert gates["forced_singleton_nodes"] == 1 and gates["competitive_nodes"] == 1
    assert gates["forced_singleton_gate_weight"]["mean"] == pytest.approx(1.0)
    assert gates["forced_singleton_fraction"] == pytest.approx(0.5)
    route_gate = report["route_gates"]["by_route_direction"]["layer_0|team|route_b|reverse"]
    assert route_gate["competitive_winner_fraction"] == pytest.approx(1.0)
    assert route_gate["raw_message_norm"]["count"] == 1
    assert gates["actual_update_norm"]["count"] == 2


def test_real_model_accepts_collector_and_emits_every_diagnostic_family() -> None:
    torch.manual_seed(17)
    model = KBORelGNNModel(
        KBORelGNNConfig(
            node_feature_dims={"player": 4, "team": 8},
            role_feature_dims={"batting": 8, "pitching": 8},
            route_feature_dims=dict.fromkeys(KBO_ROUTE_NAMES, 6),
            hidden_dim=8,
            num_layers=1,
            num_attention_heads=2,
            dropout=0.0,
        )
    ).eval()
    batch = collate_kbo_day_graphs([_day()])
    collector = RelGNNDiagnosticsCollector()
    collector.begin_batch(batch)
    output = model(batch, diagnostics_observer=collector)
    report = collector.report()
    assert torch.isfinite(output["match_logits"]).all()
    assert len(report["attention"]["by_layer_route_direction"]) == 8
    assert report["route_gates"]["by_layer_channel"]
    assert report["topology"]["query_counts"]["match"] == 1


def test_transform_spec_validation_is_strict() -> None:
    for invalid in ("", "shuffle", "none"):
        with pytest.raises(ValueError, match="unsupported"):
            KBOGraphTransformSpec(invalid)
    with pytest.raises(ValueError, match="non-negative"):
        KBOGraphTransformSpec(seed=-1)
    with pytest.raises(ValueError, match="route_name"):
        KBOGraphTransformSpec("intact", route_name=KBO_ROUTE_NAMES[0])
    with pytest.raises(ValueError, match="non-negative integer"):
        # Constructor validation must not accept bool as an integer seed.
        KBOGraphTransformSpec(seed=True)
