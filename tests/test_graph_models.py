from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cpv26.graph import (
    PLAYER_ROLE_NAMES,
    AtomicRoute,
    AtomicRouteBatch,
    GraphSnapshot,
    default_route_registry,
)


def _player_graph_inputs() -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[tuple[float, ...], ...]],
]:
    return (
        {"player": ("batter-1", "pitcher-1")},
        {"player": ((1.0, 0.0), (0.0, 1.0))},
    )


def test_graph_snapshot_rejects_route_available_after_cutoff() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_pitcher",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[cutoff_at - timedelta(days=1)],
        available_at=[cutoff_at + timedelta(microseconds=1)],
    )
    node_ids, node_features = _player_graph_inputs()

    with pytest.raises(ValueError, match="after cutoff"):
        GraphSnapshot(
            snapshot_id="future-route",
            cutoff_at=cutoff_at,
            node_ids=node_ids,
            node_features=node_features,
            routes=(route,),
        )


def test_graph_snapshot_rejects_route_outside_whitelist() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    route = AtomicRouteBatch.from_columns(
        route_name="auto_discovered_foreign_key_path",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[cutoff_at - timedelta(days=1)],
        available_at=[cutoff_at],
    )
    node_ids, node_features = _player_graph_inputs()

    with pytest.raises(KeyError, match="not whitelisted"):
        GraphSnapshot(
            snapshot_id="unreviewed-route",
            cutoff_at=cutoff_at,
            node_ids=node_ids,
            node_features=node_features,
            routes=(route,),
        )


def test_graph_snapshot_validates_whitelisted_route_endpoints() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_game",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[cutoff_at - timedelta(days=1)],
        available_at=[cutoff_at],
    )
    node_ids, node_features = _player_graph_inputs()

    with pytest.raises(ValueError, match="expects 'player' -> 'game'"):
        GraphSnapshot(
            snapshot_id="wrong-endpoints",
            cutoff_at=cutoff_at,
            node_ids=node_ids,
            node_features=node_features,
            routes=(route,),
            registry=default_route_registry(),
        )


def test_route_time_contract_uses_availability_for_eligibility_and_event_for_age() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    event_at = cutoff_at - timedelta(days=3)
    available_at = cutoff_at - timedelta(days=1)
    route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_pitcher",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[event_at],
        available_at=[available_at],
    )
    node_ids, node_features = _player_graph_inputs()

    snapshot = GraphSnapshot(
        snapshot_id="event-and-availability-time",
        cutoff_at=cutoff_at,
        node_ids=node_ids,
        node_features=node_features,
        routes=(route,),
    )

    assert snapshot.routes[0].event_at == (event_at,)
    assert snapshot.routes[0].available_at == (available_at,)
    assert route.event_ages_seconds(cutoff_at) == (3 * 86400.0,)
    assert route.publication_delays_seconds == (2 * 86400.0,)
    assert route.temporal_features(cutoff_at) == ((3 * 86400.0,),)
    assert route.temporal_features(
        cutoff_at,
        include_publication_delay=True,
    ) == ((3 * 86400.0, 2 * 86400.0),)


def test_candidate_route_allows_future_event_when_already_available() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    future_game_at = cutoff_at + timedelta(hours=3)
    route = AtomicRouteBatch.from_columns(
        route_name="player_candidate_game",
        source_type="player",
        destination_type="game",
        source_index=[0],
        destination_index=[0],
        event_at=[future_game_at],
        available_at=[cutoff_at],
    )

    snapshot = GraphSnapshot(
        snapshot_id="known-future-candidate",
        cutoff_at=cutoff_at,
        node_ids={"player": ("player-1",), "game": ("game-1",)},
        node_features={"player": ((1.0, 0.0),), "game": ((0.5,),)},
        routes=(route,),
    )

    assert snapshot.routes[0].event_ages_seconds(cutoff_at) == (-10800.0,)
    assert snapshot.routes[0].publication_delays_seconds == (-10800.0,)


def test_default_routes_include_team_game_and_candidate_relationships() -> None:
    registry = default_route_registry()

    team_route = registry.require("home_team_game_away_team")
    assert (team_route.source_type, team_route.event_type, team_route.destination_type) == (
        "team",
        "game",
        "team",
    )
    candidate_route = registry.require("player_candidate_game")
    assert (
        candidate_route.source_type,
        candidate_route.event_type,
        candidate_route.destination_type,
    ) == ("player", "player_game_candidate", "game")


def test_player_endpoint_roles_are_normalized_to_encoder_role_names() -> None:
    route = AtomicRoute(
        name="alias-normalization",
        source_type="player",
        event_type="observed_plate_appearance",
        destination_type="player",
        source_role="batter",
        destination_role="pitcher",
    )

    assert route.source_role == "batting"
    assert route.destination_role == "pitching"
    assert default_route_registry().require("pitcher_pa_catcher").destination_role == "catcher"
    assert set(PLAYER_ROLE_NAMES) == {
        "batting",
        "pitching",
        "defense",
        "baserunning",
        "catcher",
    }


def test_empty_node_type_requires_and_preserves_explicit_feature_dimension() -> None:
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="requires an explicit feature dimension"):
        GraphSnapshot(
            snapshot_id="empty-without-width",
            cutoff_at=cutoff_at,
            node_ids={"game": ()},
            node_features={"game": ()},
            routes=(),
        )

    snapshot = GraphSnapshot(
        snapshot_id="empty-with-width",
        cutoff_at=cutoff_at,
        node_ids={"game": ()},
        node_features={"game": ()},
        node_feature_dims={"game": 4},
        routes=(),
    )

    assert snapshot.feature_dims["game"] == 4


def test_empty_node_type_tensor_has_declared_width() -> None:
    torch = pytest.importorskip("torch")
    snapshot = GraphSnapshot(
        snapshot_id="empty-tensor-width",
        cutoff_at=datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc),
        node_ids={"game": ()},
        node_features={"game": ()},
        node_feature_dims={"game": 4},
        routes=(),
    )

    tensor = snapshot.torch_node_features()["game"]

    assert tensor.shape == torch.Size((0, 4))


def test_relgnn_routes_messages_through_declared_player_roles() -> None:
    torch = pytest.importorskip("torch")
    from cpv26.models import CompositeRelGNNBackbone

    torch.manual_seed(7)
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    backbone = CompositeRelGNNBackbone(
        node_feature_dims={"player": 2},
        route_feature_dims={"batter_pa_pitcher": 0},
        hidden_dim=4,
        num_layers=1,
        num_attention_heads=2,
        dropout=0.0,
    )
    backbone.eval()
    route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_pitcher",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[cutoff_at - timedelta(hours=2)],
        available_at=[cutoff_at - timedelta(hours=1)],
    )
    node_features = {"player": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    pitching = torch.zeros((2, 4))
    batting_a = torch.zeros((2, 4))
    batting_b = batting_a.clone()
    batting_b[0] = torch.tensor([8.0, -4.0, 2.0, 1.0])

    state_a = backbone.forward_relational_state(
        node_features,
        (route,),
        cutoff_at=cutoff_at,
        player_role_states={"batting": batting_a, "pitching": pitching},
    )
    state_b = backbone.forward_relational_state(
        node_features,
        (route,),
        cutoff_at=cutoff_at,
        player_role_states={"batter": batting_b, "pitcher": pitching},
    )

    assert torch.equal(state_a.node_states["player"], state_b.node_states["player"])
    assert not torch.allclose(
        state_a.player_role_states["pitching"][1],
        state_b.player_role_states["pitching"][1],
    )


def test_relgnn_connects_role_aware_player_encoder_to_route_states() -> None:
    torch = pytest.importorskip("torch")
    from cpv26.models import (
        CompositeRelGNNBackbone,
        DirectPlayerGameHead,
        PlateAppearanceInteractionDecoder,
        RoleAwarePlayerEncoder,
    )

    encoder = RoleAwarePlayerEncoder(
        shared_input_dim=2,
        role_feature_dims={role: 0 for role in PLAYER_ROLE_NAMES},
        hidden_dim=4,
        dropout=0.0,
    )
    backbone = CompositeRelGNNBackbone(
        node_feature_dims={"player": 2},
        route_feature_dims={"batter_pa_pitcher": 0},
        hidden_dim=4,
        num_layers=1,
        num_attention_heads=2,
        dropout=0.0,
        player_encoder=encoder,
    )
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_pitcher",
        source_type="player",
        destination_type="player",
        source_index=[0],
        destination_index=[1],
        event_at=[cutoff_at - timedelta(hours=2)],
        available_at=[cutoff_at - timedelta(hours=1)],
    )

    state = backbone.forward_relational_state(
        {"player": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
        (route,),
        cutoff_at=cutoff_at,
        player_role_features={},
    )
    decoder = PlateAppearanceInteractionDecoder(4, hidden_dim=8, dropout=0.0)
    player_head = DirectPlayerGameHead(
        4,
        hidden_dim=8,
        max_plate_appearances=4,
        max_hits=3,
        dropout=0.0,
    )
    pa_logits = decoder(
        state.player_role_states["batting"][:1],
        state.player_role_states["pitching"][1:2],
    )
    player_loss = player_head.negative_log_likelihood(
        state.player_role_states["batting"][:1],
        torch.tensor([4]),
        torch.tensor([2]),
    )
    loss = pa_logits.square().mean() + player_loss
    loss.backward()

    assert set(state.player_role_states) == set(PLAYER_ROLE_NAMES)
    assert all(value.shape == (2, 4) for value in state.player_role_states.values())
    assert pa_logits.shape == (1, 10)
    connected_gradients = [
        parameter.grad
        for name, parameter in backbone.named_parameters()
        if "player_encoder.shared_core" in name
        or "player_encoder.adapters.batting" in name
        or "player_encoder.adapters.pitching" in name
    ]
    assert connected_gradients
    assert all(gradient is not None for gradient in connected_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in connected_gradients
        if gradient is not None
    )


def test_route_local_attention_is_invariant_to_duplicate_edges_in_other_route() -> None:
    torch = pytest.importorskip("torch")
    from cpv26.models import CompositeRelGNNBackbone

    torch.manual_seed(17)
    cutoff_at = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    backbone = CompositeRelGNNBackbone(
        node_feature_dims={"player": 2, "game": 2},
        route_feature_dims={"batter_pa_game": 0, "player_candidate_game": 0},
        hidden_dim=4,
        num_layers=1,
        num_attention_heads=2,
        dropout=0.0,
    )
    backbone.eval()
    key_route = AtomicRouteBatch.from_columns(
        route_name="batter_pa_game",
        source_type="player",
        destination_type="game",
        source_index=[0],
        destination_index=[0],
        event_at=[cutoff_at - timedelta(days=1)],
        available_at=[cutoff_at - timedelta(hours=1)],
    )

    def unrelated_route(edge_count: int) -> AtomicRouteBatch:
        return AtomicRouteBatch.from_columns(
            route_name="player_candidate_game",
            source_type="player",
            destination_type="game",
            source_index=[1] * edge_count,
            destination_index=[0] * edge_count,
            event_at=[cutoff_at + timedelta(hours=2)] * edge_count,
            available_at=[cutoff_at] * edge_count,
        )

    node_features = {
        "player": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "game": torch.tensor([[0.25, -0.5]]),
    }
    one_unrelated = backbone(
        node_features,
        (key_route, unrelated_route(1)),
        cutoff_at=cutoff_at,
    )["game"]
    many_unrelated = backbone(
        node_features,
        (key_route, unrelated_route(100)),
        cutoff_at=cutoff_at,
    )["game"]

    assert torch.allclose(one_unrelated, many_unrelated, rtol=1e-6, atol=1e-6)


def test_models_import_and_fail_clearly_when_torch_is_unavailable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_torch

        import cpv26.models as models

        assert models.torch_available() is False
        try:
            models.RoleAwarePlayerEncoder(shared_input_dim=4, role_feature_dims={})
        except models.TorchUnavailableError as error:
            assert "PyTorch is required" in str(error)
        else:
            raise AssertionError("neural model construction succeeded without PyTorch")
        """
    )
    environment = os.environ.copy()
    python_path = str(project_root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path = python_path + os.pathsep + existing_python_path
    environment["PYTHONPATH"] = python_path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
