from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from cpv26.graph import TorchAtomicRouteBatch  # noqa: E402
from cpv26.models.kbo_relgnn import (  # noqa: E402
    KBORelGNNConfig,
    KBORelGNNModel,
)
from cpv26.models.relgnn import RelGNNDiagnosticsObserver  # noqa: E402


class _RecordingObserver:
    def __init__(self) -> None:
        self.attention_events: list[dict[str, Any]] = []
        self.gate_events: list[dict[str, Any]] = []

    @staticmethod
    def _record_tensors(values: Mapping[str, Any]) -> None:
        for value in values.values():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                assert value.grad_fn is None

    def observe_attention(
        self,
        *,
        layer_index: int,
        route_name: str,
        direction: str,
        source_channel: str,
        destination_channel: str,
        source_index: Any,
        destination_index: Any,
        positive_weight: Any,
        attention: Any,
        message: Any,
        route_mask: Any,
        destination_state: Any,
    ) -> None:
        event = dict(
            layer_index=layer_index,
            route_name=route_name,
            direction=direction,
            source_channel=source_channel,
            destination_channel=destination_channel,
            source_index=source_index,
            destination_index=destination_index,
            positive_weight=positive_weight,
            attention=attention,
            message=message,
            route_mask=route_mask,
            destination_state=destination_state,
        )
        self._record_tensors(event)
        self.attention_events.append(event)

    def observe_gates(
        self,
        *,
        layer_index: int,
        destination_channel: str,
        route_names: tuple[str, ...],
        directions: tuple[str, ...],
        source_channels: tuple[str, ...],
        gate_keys: tuple[str, ...],
        messages: Any,
        masks: Any,
        route_attention: Any,
        previous_state: Any,
        combined_message: Any,
        candidate_state: Any,
        updated_state: Any,
    ) -> None:
        event = dict(
            layer_index=layer_index,
            destination_channel=destination_channel,
            route_names=route_names,
            directions=directions,
            source_channels=source_channels,
            gate_keys=gate_keys,
            messages=messages,
            masks=masks,
            route_attention=route_attention,
            previous_state=previous_state,
            combined_message=combined_message,
            candidate_state=candidate_state,
            updated_state=updated_state,
        )
        self._record_tensors(event)
        self.gate_events.append(event)


def _model_and_batch() -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(83)
    model = KBORelGNNModel(
        KBORelGNNConfig(
            node_feature_dims={"player": 2, "team": 2},
            role_feature_dims={"batting": 2, "pitching": 2},
            route_feature_dims={"batter_pa_pitcher": 1},
            hidden_dim=4,
            num_layers=2,
            num_attention_heads=2,
            dropout=0.0,
            pa_context_dim=0,
        )
    )
    route = TorchAtomicRouteBatch(
        route_name="batter_pa_pitcher",
        source_type="player",
        destination_type="player",
        source_index=torch.tensor([0, 2, 0]),
        destination_index=torch.tensor([1, 1, 2]),
        event_features=torch.tensor([[0.2], [-0.3], [0.7]]),
        event_age_seconds=torch.tensor([7200.0, 10_800.0, 14_400.0]),
        publication_delay_seconds=torch.tensor([60.0, 120.0, 180.0]),
        weights=torch.tensor([1.0, 0.0, 2.0]),
        bidirectional=True,
    )
    batch = {
        "_validated_on_cpu": True,
        "node_features": {
            "player": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]),
            "team": torch.tensor([[0.25, 0.75], [-0.25, 0.5]]),
        },
        "role_features": {
            "batting": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "pitching": torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]),
        },
        "routes": (route,),
        "match_home_team_index": torch.tensor([0]),
        "match_away_team_index": torch.tensor([1]),
        "live_hit_player_index": torch.tensor([0]),
        "live_hit_team_index": torch.tensor([0]),
        "live_hit_opponent_index": torch.tensor([1]),
        "pa_batter_index": torch.tensor([0]),
        "pa_pitcher_index": torch.tensor([1]),
        "pa_context": torch.empty((1, 0)),
        "box_pa_player_index": torch.empty(0, dtype=torch.long),
        "box_pitch_player_index": torch.empty(0, dtype=torch.long),
    }
    return cast(Any, model).eval(), batch


def _assert_tree_equal(left: Any, right: Any) -> None:
    if isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
        return
    assert torch.equal(left, right)


def test_observer_exposes_detached_directional_tensors_without_changing_forward() -> None:
    model, batch = _model_and_batch()
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    expected = model(batch)
    observer = _RecordingObserver()
    typed_observer: RelGNNDiagnosticsObserver = observer

    actual = model(batch, diagnostics_observer=typed_observer)

    _assert_tree_equal(expected, actual)
    assert state_before.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(state_before[name], value), name

    assert len(observer.attention_events) == 4
    metadata = {
        (
            event["layer_index"],
            event["direction"],
            event["source_channel"],
            event["destination_channel"],
        )
        for event in observer.attention_events
    }
    assert metadata == {
        (0, "forward", "player__batting", "player__pitching"),
        (0, "reverse", "player__pitching", "player__batting"),
        (1, "forward", "player__batting", "player__pitching"),
        (1, "reverse", "player__pitching", "player__batting"),
    }
    for event in observer.attention_events:
        assert event["route_name"] == "batter_pa_pitcher"
        assert event["positive_weight"].tolist() == [True, False, True]
        assert event["attention"].shape == (3, 2)
        assert torch.equal(event["attention"][1], torch.zeros(2))
        if event["direction"] == "forward":
            assert event["source_index"].tolist() == [0, 2, 0]
            assert event["destination_index"].tolist() == [1, 1, 2]
        else:
            assert event["source_index"].tolist() == [1, 1, 2]
            assert event["destination_index"].tolist() == [0, 2, 0]

    assert len(observer.gate_events) == 4
    for event in observer.gate_events:
        assert event["route_names"] == ("batter_pa_pitcher",)
        assert len(event["directions"]) == len(event["source_channels"]) == 1
        assert event["messages"].shape == (3, 1, 4)
        assert event["masks"].shape == (3, 1)
        assert event["route_attention"].shape == (3, 1)


def test_default_observer_does_not_add_checkpoint_state() -> None:
    model, _ = _model_and_batch()

    assert all("diagnostic" not in name for name in model.state_dict())
